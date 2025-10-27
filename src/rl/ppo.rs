use std::collections::VecDeque;

use ndarray::{Array1, Array2, Axis, concatenate};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use rand::seq::SliceRandom;

use crate::{
    f,
    nn::ffn::{FFN, Layer, LayerDef},
    optim::param::ToParams,
};

pub struct Trajectory {
    pub state: Array2<f64>,
    pub action: Array2<f64>,
    pub action_raw: Array2<f64>,
    pub log_prob: Array2<f64>,
    pub pred_value: f64,
    pub reward: f64,
}

pub struct PPO {
    d_state: usize,
    d_action: usize,

    values: FFN,
    policy: FFN,
    policy_mean_head: Layer,
    policy_logstd_head: Layer,

    trajectories: VecDeque<Trajectory>,
}

impl PPO {
    pub fn new(
        d_state: usize,
        d_action: usize,
        values: Vec<LayerDef>,
        policy_hidden: Vec<LayerDef>,
    ) -> PPO {
        let policy_last_layer = policy_hidden
            .last()
            .expect("Policy network needs at least one hidden layer.");
        let policy_mean_head = Layer::new(policy_last_layer.1, d_action, f::ident, f::d_ident);
        let policy_logstd_head = Layer::new(policy_last_layer.1, d_action, f::ident, f::d_ident);
        let policy = FFN::new(policy_hidden);

        PPO {
            d_state,
            d_action,

            values: FFN::new(values),
            policy,
            policy_mean_head,
            policy_logstd_head,

            trajectories: VecDeque::new(),
        }
    }

    pub fn forward(&mut self, state: Array2<f64>) -> Trajectory {
        let hidden = self.policy.forward(state.clone(), false);
        let mean = self.policy_mean_head.forward(hidden.clone(), false);
        let logstd = self.policy_logstd_head.forward(hidden, false);

        let std = logstd.exp();
        let eps = Array2::random(mean.dim(), Normal::new(0., 1.).unwrap());
        let u = &mean + &std * eps;
        let pred_value = self.values.forward(state.clone(), false);

        let a = f::tanh(&u);

        if a.is_any_nan() {
            println!("PPO: policy exploded");
        }

        if pred_value.is_any_nan() {
            println!("PPO: values exploded");
        }

        let log_prob = f::gaussian_log_prob(&u, &mean, &std) - f::tanh_gaussian_correction_eps(&u);

        Trajectory {
            state,
            action: a,
            action_raw: u,
            log_prob: log_prob.insert_axis(Axis(1)),
            pred_value: pred_value[[0, 0]],
            reward: 0.,
        }
    }

    pub fn complete(&mut self, mut trajectory: Trajectory, reward: f64) {
        trajectory.reward = reward;
        self.trajectories.push_back(trajectory);
    }

    fn concat_from_indices(
        feature_size: usize,
        source: &Vec<&Array2<f64>>,
        indices: &[usize],
    ) -> Array2<f64> {
        indices
            .into_iter()
            .fold(Array2::zeros((0, feature_size)), |a, v| {
                concatenate(Axis(0), &[a.view(), source[*v].view()]).unwrap()
            })
    }

    pub fn backward(
        &mut self,
        gamma: f64,
        lambda: f64,
        clip: f64,
        epochs: usize,
        c1: f64,
        c2: f64,
    ) {
        if self.trajectories.len() < 2 {
            return;
        }

        let count = self.trajectories.len() - 1;
        let current = 0..self.trajectories.len() - 1;
        let next = 1..self.trajectories.len();

        let states = self
            .trajectories
            .range(current.clone())
            .map(|x| &x.state)
            .collect::<Vec<&Array2<f64>>>();

        let us = self
            .trajectories
            .range(current.clone())
            .map(|x| &x.action_raw)
            .collect::<Vec<&Array2<f64>>>();

        let log_probs = self
            .trajectories
            .range(current.clone())
            .map(|x| &x.log_prob)
            .collect::<Vec<&Array2<f64>>>();

        let rewards = self
            .trajectories
            .range(current.clone())
            .map(|t| t.reward)
            .collect::<Vec<f64>>();

        let value_pred_current = self
            .trajectories
            .range(current.clone())
            .map(|t| t.pred_value)
            .collect::<Vec<f64>>();

        let value_pred_next = self
            .trajectories
            .range(next)
            .map(|t| t.pred_value)
            .collect::<Vec<f64>>();

        let mut rolling_advantage = 0.;
        let mut advantages = Vec::with_capacity(count);
        let mut returns = Vec::with_capacity(count);

        for i in (0..count).rev() {
            let delta = rewards[i] + gamma * value_pred_next[i] - value_pred_current[i];
            rolling_advantage = delta + lambda * gamma * rolling_advantage;
            advantages.push(rolling_advantage);
            returns.push(value_pred_current[i] + rolling_advantage);
        }

        advantages.reverse();
        returns.reverse();

        let advantages = Array1::from_vec(advantages);
        let norm_advantages =
            (&advantages - advantages.mean().unwrap()) / (&advantages.std(0.) + 1e-8);
        let returns = Array1::from_vec(returns);

        let norm_advantages = norm_advantages
            .into_iter()
            .map(|a| Array1::from_vec(vec![a]).insert_axis(Axis(0)))
            .collect::<Vec<Array2<f64>>>();
        let returns = returns
            .into_iter()
            .map(|r| Array1::from_vec(vec![r]).insert_axis(Axis(0)))
            .collect::<Vec<Array2<f64>>>();

        let mut trajectory_indices = (0..count).into_iter().collect::<Vec<usize>>();
        trajectory_indices.shuffle(&mut rand::rng());
        let batch_size = (count as f64 / epochs as f64).ceil() as usize;

        for _ in 0..epochs {
            let batch_indices = trajectory_indices
                .drain(0..batch_size.min(trajectory_indices.len()))
                .collect::<Vec<usize>>();
            let states = PPO::concat_from_indices(self.d_state, &states, &batch_indices);
            let us = PPO::concat_from_indices(self.d_action, &us, &batch_indices);
            let log_probs = PPO::concat_from_indices(1, &log_probs, &batch_indices);

            let norm_advantages =
                PPO::concat_from_indices(1, &norm_advantages.iter().collect(), &batch_indices);
            let returns = PPO::concat_from_indices(1, &returns.iter().collect(), &batch_indices);

            let batch_mean = 1. / states.nrows() as f64;

            // Values network update
            let pred_returns = self.values.forward(states.clone(), true);
            let dl_dv = c1 * batch_mean * (2. / count as f64) * (&pred_returns - &returns);
            self.values.backward(dl_dv);

            // Policy network update
            let hidden = self.policy.forward(states.clone(), true);
            let mean = self.policy_mean_head.forward(hidden.clone(), true);
            let logstd = self.policy_logstd_head.forward(hidden, true);
            let std = logstd.exp();

            let log_prob_new = (f::gaussian_log_prob(&us, &mean, &std)
                - f::tanh_gaussian_correction_eps(&us))
            .insert_axis(Axis(1));

            let r = (&log_prob_new - &log_probs).exp();
            let r_clip = r.clamp(1. - clip, 1. + clip);

            let mut r_adv: Array2<f64> = &r * &norm_advantages;
            let r_clip_adv = &r_clip * &norm_advantages;

            r_adv.zip_mut_with(&r_clip_adv, |x, y| {
                *x = x.min(*y);
            });

            let clipped_surrogate_loss = -batch_mean * r_adv;

            // Create policy gradients
            let dlogprob_dmean = (&us - &mean) / &std.powi(2);
            let dlogprob_dlogstd = (&us - &mean).powi(2) / std.powi(2) - 1.;
            let entropy_grad_logstd = Array2::<f64>::zeros(std.dim()) * (c2 / batch_mean);

            let d_policy_mean = &clipped_surrogate_loss * dlogprob_dmean;
            let d_policy_logstd = &clipped_surrogate_loss * dlogprob_dlogstd + entropy_grad_logstd;

            let grad_mean = self.policy_mean_head.backward(d_policy_mean);
            let grad_logstd = self.policy_logstd_head.backward(d_policy_logstd);

            self.policy.backward(grad_mean + grad_logstd);
        }

        self.trajectories.clear();
    }
}

impl ToParams for PPO {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        let mut params = vec![];

        params.append(&mut self.values.params());
        params.append(&mut self.policy.params());
        params.append(&mut self.policy_mean_head.params());
        params.append(&mut self.policy_logstd_head.params());

        params
    }
}
