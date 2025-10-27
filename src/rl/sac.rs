// WIP

use core::f64;
use std::collections::VecDeque;

use ndarray::{Array1, Array2, Axis, Slice, concatenate};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use rand::seq::SliceRandom;

use crate::{
    f,
    nn::ffn::{FFN, LayerDef},
    optim::param::{Param, ToParams},
};

#[derive(Debug, Clone)]
pub struct Experience {
    pub state: Array2<f64>,

    pub action_mean: Array2<f64>,
    pub action_log_std: Array2<f64>,
    pub action_sample: Array2<f64>,

    pub reward: f64,
    pub state_next: Array2<f64>,
}

pub struct SAC {
    pub d_state: usize,
    pub d_action: usize,

    pub q1: FFN,
    pub q2: FFN,
    pub q1_target: FFN,
    pub q2_target: FFN,

    pub policy: FFN,
    pub policy_mean_head: FFN,
    pub policy_std_head: FFN,
    pub log_alpha: f64,
    pub d_log_alpha: f64,

    pub buffer: VecDeque<Experience>,
}

impl SAC {
    pub fn new(
        d_state: usize,
        d_action: usize,
        q_layers: Vec<LayerDef>,
        policy_hidden_layers: Vec<LayerDef>,
    ) -> Self {
        let Some(q_first) = q_layers.first() else {
            panic!("Q networks must have at least 1 layer.");
        };

        if q_first.0 != d_state + d_action {
            panic!("Q network input layer must be (d_state + d_action, ..)");
        }

        let Some(q_last) = q_layers.last() else {
            panic!("Q networks must have at least 1 layer.");
        };

        let Some(policy_first) = policy_hidden_layers.first() else {
            panic!("Policy networks myst have at least 1 hidden layer.")
        };

        if policy_first.0 != d_state {
            panic!("Policy network input layer must be (d_state, ..)")
        }

        let Some(policy_last) = policy_hidden_layers.last() else {
            panic!("Policy network must have at least 1 hidden layer.")
        };

        let n_action = q_last.1;

        let q1 = FFN::new(q_layers.clone());
        let q2 = FFN::new(q_layers);
        let q1_target = q1.clone();
        let q2_target = q2.clone();

        let policy_mean_head = FFN::new(vec![(policy_last.1, n_action, f::ident, f::d_ident)]);
        let policy_std_head = FFN::new(vec![(policy_last.1, n_action, f::ident, f::d_ident)]);
        let policy = FFN::new(policy_hidden_layers);

        Self {
            d_state,
            d_action,

            q1,
            q2,
            q1_target,
            q2_target,

            policy,
            policy_mean_head,
            policy_std_head,
            log_alpha: 0.,
            d_log_alpha: 0.,

            buffer: VecDeque::new(),
        }
    }

    pub fn inference(
        &mut self,
        x: Array2<f64>,
        grad: bool,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        let hidden_policy = self.policy.forward(x, grad);
        let mean = self.policy_mean_head.forward(hidden_policy.clone(), grad);
        let log_std = self
            .policy_std_head
            .forward(hidden_policy, grad)
            .clamp(-20., 2.);
        let std = log_std.exp();

        let gaussian = Array2::random(mean.dim(), Normal::new(0., 1.).unwrap());
        let a_raw = &mean + &(&std * &gaussian);
        let a = f::tanh(&a_raw);

        (a, mean, a_raw, log_std)
    }

    pub fn gaussian_log_prob(
        a_raw: &Array2<f64>,
        u: &Array2<f64>,
        log_std: &Array2<f64>,
    ) -> Array1<f64> {
        let eps = 1e-8;
        let std = log_std.exp();
        let quadratic = (a_raw - u).powi(2) / std.powi(2);
        let norm = 2. * std.map(|v| v.max(eps).ln());
        let constant = (f64::consts::PI * 2.).ln();

        let dims = -0.5 * (quadratic + norm + constant);

        dims.sum_axis(Axis(1))
    }

    pub fn tanh_gaussian_correction(a_raw: &Array2<f64>) -> Array1<f64> {
        let eps = 1e-6;
        let d_tanh = f::d_tanh(a_raw) + eps;
        d_tanh.ln().sum_axis(Axis(1))
    }

    pub fn remember(&mut self, exp: Experience, retain: usize) {
        self.buffer.push_back(exp);

        while self.buffer.len() > retain {
            self.buffer.pop_front();
        }
    }

    pub fn batch(&self, size: usize) -> Vec<Experience> {
        let mut pool = (0..self.buffer.len()).collect::<Vec<usize>>();
        let mut rng = rand::rng();
        pool.shuffle(&mut rng);

        let mut batch = vec![];
        for i in 0..size.min(pool.len()) {
            let ix = pool[i % pool.len()];
            batch.push(self.buffer[ix].clone());
        }

        batch
    }

    pub fn backwards(&mut self, batch: Vec<Experience>, gamma: f64, tau: f64) {
        let states = batch.iter().fold(Array2::zeros((0, self.d_state)), |a, v| {
            concatenate(Axis(0), &[a.view(), v.state.view()]).unwrap()
        });
        let actions_taken = batch
            .iter()
            .fold(Array2::zeros((0, self.d_action)), |a, v| {
                concatenate(Axis(0), &[a.view(), v.action_sample.view()]).unwrap()
            });
        let sa_current = concatenate(Axis(1), &[states.view(), actions_taken.view()]).unwrap();

        let next_states = batch.iter().fold(Array2::zeros((0, self.d_state)), |a, v| {
            concatenate(Axis(0), &[a.view(), v.state_next.view()]).unwrap()
        });
        let (a, ..) = self.inference(next_states.clone(), false);

        let sa_next = concatenate(Axis(1), &[next_states.view(), a.view()]).unwrap();

        let rewards = batch
            .iter()
            .map(|v| v.reward)
            .collect::<Array1<f64>>()
            .insert_axis(Axis(1));

        let q1_t_pred = self.q1_target.forward(sa_next.clone(), false);
        let q2_t_pred = self.q2_target.forward(sa_next.clone(), false);

        let mut q1_mask = Array2::zeros(q1_t_pred.dim());
        let mut q2_mask = Array2::zeros(q2_t_pred.dim());
        let mut qt_min = q1_t_pred.clone();
        qt_min.indexed_iter_mut().for_each(|((y, x), v)| {
            let v2 = q2_t_pred[[y, x]];
            if v2 < *v {
                q2_mask[[y, x]] = 1.;
                *v = v2;
            } else {
                q1_mask[[y, x]] = 1.;
            }
        });

        let (.., mean, a_raw, log_std) = self.inference(states.clone(), true);

        let gaussian_log_probs = SAC::gaussian_log_prob(&a_raw, &mean, &log_std);
        let correction = SAC::tanh_gaussian_correction(&a_raw);
        let log_probs = (&gaussian_log_probs - &correction).insert_axis(Axis(1));

        let y_target = rewards + gamma * &(&qt_min - self.log_alpha.exp() * &log_probs);

        let q1_pred = self.q1.forward(sa_current.clone(), true);
        let q2_pred = self.q2.forward(sa_current.clone(), true);

        let q1_dloss = (2. / batch.len() as f64) * (&q1_pred - &y_target);
        let q2_dloss = (2. / batch.len() as f64) * (&q2_pred - &y_target);

        // Get dQ/da
        let dq1_dinput = self.q1.backward(Array2::ones(q1_pred.dim()));
        let dq2_dinput = self.q2.backward(Array2::ones(q2_pred.dim()));

        // Set the gradients back to real loss so the optim can correct
        self.q1.backward(q1_dloss);
        self.q2.backward(q2_dloss);

        // Slice observation off of both of these, leaving only action.
        let dq1_da = dq1_dinput
            .slice_axis(Axis(1), Slice::from(self.d_state..))
            .to_owned();
        let dq2_da = dq2_dinput
            .slice_axis(Axis(1), Slice::from(self.d_state..))
            .to_owned();
        let dqmin_da = &(dq1_da * q1_mask) + &(dq2_da * q2_mask); // (B, S + A)

        // Get derivative of a wrt a_raw
        // ∂a/∂u
        let d_tanh = f::d_tanh(&a_raw);

        // ∂logπ(a|s)/∂µ
        let dgaussian_du = (&a_raw - &mean) / log_std.exp().powi(2);
        // ∂c/∂µ
        // let dcorrection_du =
        //     (-2. * f::tanh(&a_raw) * f::d_tanh(&a_raw)) / (f::d_tanh(&a_raw) + 1e-6);
        let dcorrection_du = -2.0 * f::tanh(&a_raw);

        // Policy gradient - mean head
        // ∂L/∂µ = a(∂logπ/∂µ - ∂c/∂µ) - (∂Q/∂a * ∂a/∂µ)
        // ∂L/∂µ = a(∂logπ/∂µ - ∂c/∂µ) - (∂Q/∂u)
        let dl_dmu = self.log_alpha.exp() * &(&dgaussian_du - &dcorrection_du) // derivative of logπ(a|s) w.r.t. µ
            - &dqmin_da // ∂Q_∂a
            * &d_tanh; // ∂a/∂u

        // Mathematically undoing a_raw = µ + oe to get e
        let eps = (&a_raw - &mean) / &log_std.exp();
        // ∂logπ/∂logstd
        let dgaussian_dlogstd = (&a_raw - &mean).powi(2) / log_std.exp().powi(2) - 1.0;
        // ∂µ/∂logstd
        let du_dlogstd = log_std.exp() * &eps;
        // ∂c/∂log_std = ∂c/∂u * ∂u/∂log_std
        let dcorrection_dlogstd = &dcorrection_du * &du_dlogstd;

        // Policy gradient: logstd head
        // ∂L/∂logstd = a(∂logπ/∂logstd - ∂c/∂logstd) - (∂Q/∂a * ∂a/∂µ * ∂µ/∂logstd)
        // ∂L/∂logstd = a(∂logπ/∂logstd - ∂c/∂logstd) - (∂Q/∂logstd)
        let dl_dlogstd = self.log_alpha.exp() * (dgaussian_dlogstd - dcorrection_dlogstd)
            - &dqmin_da     // ∂Q/∂a
            * &d_tanh       // ∂a/∂µ
            * du_dlogstd; // ∂µ/∂logstd

        // Backwards and merge
        let mean_loss = self.policy_mean_head.backward(dl_dmu);
        let std_loss = self.policy_std_head.backward(dl_dlogstd);
        self.policy.backward(mean_loss + std_loss);

        // Update alpha
        // ∂L/∂loga = ∂L/∂a * ∂a/∂loga
        let h_target = -1. * self.d_action as f64;
        let mean_logp = log_probs.mean().unwrap();
        let dl_da = -(mean_logp + h_target);
        let da_dloga = self.log_alpha.exp();
        let dl_dloga = dl_da * da_dloga;
        self.d_log_alpha = dl_dloga;

        for l in 0..self.q1.layers.len() {
            self.q1_target.layers[l].w =
                (1. - tau) * &self.q1_target.layers[l].w + tau * &self.q1.layers[l].w;
            self.q1_target.layers[l].b =
                (1. - tau) * &self.q1_target.layers[l].b + tau * &self.q1.layers[l].b;
            self.q2_target.layers[l].w =
                (1. - tau) * &self.q2_target.layers[l].w + tau * &self.q2.layers[l].w;
            self.q2_target.layers[l].b =
                (1. - tau) * &self.q2_target.layers[l].b + tau * &self.q2.layers[l].b;
        }
    }
}

impl ToParams for SAC {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        let mut params = vec![];

        params.append(&mut self.q1.params());
        params.append(&mut self.q2.params());
        params.append(&mut self.policy.params());
        params.append(&mut self.policy_mean_head.params());
        params.append(&mut self.policy_std_head.params());
        params.push(Param::from_scalars(
            &mut self.log_alpha,
            &mut self.d_log_alpha,
        ));

        params
    }
}
