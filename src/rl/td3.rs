use std::collections::VecDeque;

use ndarray::{Array1, Array2, Axis, Slice, concatenate};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use rand::seq::IteratorRandom;

use crate::{
    nn::ffn::{FFN, LayerDef},
    optim::param::ToParams,
};

pub struct Experience {
    pub state: Array2<f64>,
    pub action: Array2<f64>,
    pub reward: f64,
    pub state_next: Array2<f64>,
    pub done: bool,
}

pub struct TD3 {
    d_state: usize,
    d_action: usize,

    q1: FFN,
    q2: FFN,
    q1_target: FFN,
    q2_target: FFN,

    policy: FFN,
    policy_target: FFN,

    total_steps: usize,
    pub policy_delay: usize,
    pub buffer: VecDeque<Experience>,
}

impl TD3 {
    pub fn new(
        d_state: usize,
        d_action: usize,
        q_layers: Vec<LayerDef>,
        policy_layers: Vec<LayerDef>,
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

        if q_last.1 != 1 {
            panic!("Q network output layer must be (.., 1)")
        }

        let Some(policy_first) = policy_layers.first() else {
            panic!("Policy networks myst have at least 1 layer.")
        };

        if policy_first.0 != d_state {
            panic!("Policy network input layer must be (d_state, ..)")
        }

        let Some(policy_last) = policy_layers.last() else {
            panic!("Policy network must have at least 1 layer.")
        };

        if policy_last.1 != d_action {
            panic!("Policy network output layer must be (.., d_action)")
        }

        let q1 = FFN::new(q_layers.clone());
        let q2 = FFN::new(q_layers);
        let q1_target = q1.clone();
        let q2_target = q2.clone();

        let policy = FFN::new(policy_layers);
        let policy_target = policy.clone();

        Self {
            d_state,
            d_action,

            q1,
            q2,
            q1_target,
            q2_target,

            policy,
            policy_target,

            buffer: VecDeque::new(),
            total_steps: 0,
            policy_delay: 2,
        }
    }

    pub fn forward(&mut self, state: Array2<f64>) -> Array2<f64> {
        self.policy.forward(state, false)
    }

    pub fn noise(action: Array2<f64>, sigma: f64) -> Array2<f64> {
        let noise = Array2::random(action.dim(), Normal::new(0., sigma).unwrap());
        action + noise
    }

    pub fn remember(&mut self, experience: Experience, retain: usize) {
        self.buffer.push_back(experience);
        while self.buffer.len() > retain {
            self.buffer.pop_front();
        }
    }

    pub fn batch(&self, batch_size: usize) -> Vec<&Experience> {
        let mut rng = rand::rng();
        self.buffer.iter().choose_multiple(&mut rng, batch_size)
    }

    fn soft_update(target: &mut FFN, net: &FFN, tau: f64) {
        net.layers
            .iter()
            .zip(target.layers.iter_mut())
            .for_each(|(l, l_t)| {
                l_t.w = (1. - tau) * &l_t.w + tau * &l.w;
                l_t.b = (1. - tau) * &l_t.b + tau * &l.b;
            });
    }

    pub fn backward(&mut self, mut batch_size: usize, gamma: f64, tau: f64, sigma: f64, clip: f64) {
        let batch = self.batch(batch_size);
        batch_size = batch.len();

        let states = batch.iter().fold(Array2::zeros((0, self.d_state)), |a, v| {
            concatenate(Axis(0), &[a.view(), v.state.view()]).unwrap()
        });

        let states_next = batch.iter().fold(Array2::zeros((0, self.d_state)), |a, v| {
            concatenate(Axis(0), &[a.view(), v.state_next.view()]).unwrap()
        });

        let actions = batch
            .iter()
            .fold(Array2::zeros((0, self.d_action)), |a, v| {
                concatenate(Axis(0), &[a.view(), v.action.view()]).unwrap()
            });

        let rewards = batch
            .iter()
            .map(|x| x.reward)
            .collect::<Array1<f64>>()
            .insert_axis(Axis(1));

        let dones = batch
            .iter()
            .map(|x| if x.done { 1. } else { 0. })
            .collect::<Array1<f64>>()
            .insert_axis(Axis(1));

        let noise = Array2::random((batch_size, self.d_action), Normal::new(0., sigma).unwrap())
            .clamp(-clip, clip);
        let mut actions_next = self.policy_target.forward(states_next.clone(), false);
        actions_next += &noise;

        let sa = concatenate(Axis(1), &[states.view(), actions.view()]).unwrap();
        let sa_next = concatenate(Axis(1), &[states_next.view(), actions_next.view()]).unwrap();

        let q1_target_pred = self.q1_target.forward(sa_next.clone(), false);
        let q2_target_pred = self.q2_target.forward(sa_next.clone(), false);
        let mut q_target_min = q1_target_pred.clone();
        q_target_min.zip_mut_with(&q2_target_pred, |x, y| {
            *x = x.min(*y);
        });

        let y_target = rewards + (1. - dones) * gamma * q_target_min;

        let q1_pred = self.q1.forward(sa.clone(), true);
        let q2_pred = self.q2.forward(sa.clone(), true);

        if self.total_steps > 0 && self.total_steps % self.policy_delay == 0 {
            let dq_da = self
                .q1
                .backward(-Array2::ones(q1_pred.dim()))
                .slice_axis(Axis(1), Slice::from(self.d_state..))
                .to_owned();

            self.policy.forward(states.clone(), true);
            self.policy.backward(dq_da);

            TD3::soft_update(&mut self.policy_target, &self.policy, tau);
            TD3::soft_update(&mut self.q1_target, &self.q1, tau);
            TD3::soft_update(&mut self.q2_target, &self.q2, tau);
        }

        let dl_dq1 = (1. / batch_size as f64) * 2. * (&q1_pred - &y_target);
        let dl_dq2 = (1. / batch_size as f64) * 2. * (&q2_pred - &y_target);

        self.q1.backward(dl_dq1);
        self.q2.backward(dl_dq2);

        self.total_steps = self.total_steps.wrapping_add(1);
    }
}

impl ToParams for TD3 {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        let mut params = vec![];

        params.append(&mut self.q1.params());
        params.append(&mut self.q2.params());
        params.append(&mut self.policy.params());

        params
    }
}
