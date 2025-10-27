#[allow(dead_code)]
use std::collections::VecDeque;

use ndarray::{Array1, Array2, Axis, concatenate};
use rand::{rng, rngs::ThreadRng};

use crate::{
    f,
    nn::ffn::{FFN, LayerDef},
};

pub struct Effect {
    reward: f64,
    new_state: Array2<f64>,
}

struct Trajectory {
    obs: Array2<f64>,
    action_ix: usize,
    action_log_prob: f64,
    pred_value: f64,

    reward: f64,
    pred_value_next: f64,
}

struct PartialTrajectory {
    obs: Array2<f64>,
    action_ix: usize,
    action_log_prob: f64,
    pred_value: f64,
}

impl PartialTrajectory {
    pub fn complete(self, reward: f64, pred_value_next: f64) -> Trajectory {
        Trajectory {
            obs: self.obs,
            action_ix: self.action_ix,
            action_log_prob: self.action_log_prob,
            pred_value: self.pred_value,

            reward,
            pred_value_next,
        }
    }
}

pub struct DiscretePPO {
    pub d_obs: usize,
    pub d_action: usize,

    pub policy: FFN,
    pub values: FFN,
    pub rng: ThreadRng,

    partial_trajectories: VecDeque<PartialTrajectory>,
    trajectories: VecDeque<Trajectory>,
}

impl DiscretePPO {
    pub fn new(
        mut policy_layers: Vec<LayerDef>,
        mut values_layers: Vec<LayerDef>,
        d_action: usize,
        d_obs: usize,
    ) -> Self {
        let d_policy_last = policy_layers.last().unwrap().1;
        let d_values_last = values_layers.last().unwrap().1;

        let policy_final: LayerDef = (d_policy_last, d_action, f::ident, f::d_ident);
        let values_final: LayerDef = (d_values_last, 1, f::ident, f::d_ident);

        policy_layers.push(policy_final);
        values_layers.push(values_final);

        Self {
            d_obs,
            d_action,

            policy: FFN::new(policy_layers),
            values: FFN::new(values_layers),
            rng: rng(),

            partial_trajectories: VecDeque::new(),
            trajectories: VecDeque::new(),
        }
    }

    pub fn forward(&mut self, x: Array2<f64>) -> Array1<f64> {
        let action = self.policy.forward(x.clone(), true);
        let pred_reward = self.values.forward(x.clone(), true);

        let action_log_probs = f::log_softmax(action).remove_axis(Axis(0));
        let action_ix = f::sample_gumbel_categorical(&action_log_probs, &mut self.rng);
        let action_log_prob = action_log_probs[action_ix];

        let partial_trajectory = PartialTrajectory {
            obs: x,
            action_ix,
            action_log_prob,
            pred_value: pred_reward[[0, 0]],
        };

        self.partial_trajectories.push_back(partial_trajectory);

        action_log_probs
    }

    pub fn backward(&mut self, effects: Vec<Effect>, epochs: usize, buffer: usize, clip: f64) {
        for e in effects {
            let Some(partial) = self.partial_trajectories.pop_front() else {
                continue;
            };

            let pred_reward_next = self.values.forward(e.new_state.clone(), false);
            self.trajectories
                .push_back(partial.complete(e.reward, pred_reward_next[[0, 0]]));
        }

        while self.trajectories.len() > buffer {
            self.trajectories.pop_front();
        }

        let lambda = 0.9;
        let gamma = 0.9;

        let mut rolling_advantage = 0.;
        let mut advantages = vec![];
        let mut returns = vec![];

        for t in self.trajectories.iter().rev() {
            let delta = t.reward + gamma * t.pred_value_next - t.pred_value;

            rolling_advantage = delta + (gamma * lambda) * rolling_advantage;

            advantages.push(rolling_advantage);
            returns.push(t.pred_value + rolling_advantage);
        }

        advantages.reverse();
        returns.reverse();

        let observations = self
            .trajectories
            .iter()
            .fold(Array2::zeros((0, self.d_obs)), |a, v| {
                concatenate(Axis(0), &[a.view(), v.obs.view()]).unwrap()
            });

        let v_y_true = Array1::from_vec(returns).insert_axis(Axis(1));

        let mut actions_prev = Array2::zeros((self.trajectories.len(), self.d_action));
        let mut actions_mask = actions_prev.clone();

        for (i, t) in self.trajectories.iter().enumerate() {
            actions_prev[[i, t.action_ix]] = t.action_log_prob;
            actions_mask[[i, t.action_ix]] = 1.;
        }

        let advantages = Array1::from_vec(advantages);

        for _ in 0..epochs {
            let v_y_pred = self.values.forward(observations.clone(), true);
            let v_err = &v_y_pred - &v_y_true;
            let _v_loss = 0.5 * &v_err * &v_err;

            self.values.backward(v_err.clone());

            let actions_current = self.policy.forward(observations.clone(), true);
            let actions_current = f::log_softmax(actions_current);
            let actions_current_masked = &actions_current * &actions_mask;

            let ratio = (&actions_current_masked - &actions_prev).exp();

            let unclip_loss = &ratio * &advantages;
            let mut p_loss = ratio.clamp(1. - clip, 1. + clip) * &advantages;
            p_loss.indexed_iter_mut().for_each(|((y, x), v)| {
                *v = v.min(unclip_loss[[y, x]]);
            });

            self.policy.backward(p_loss);
        }
    }
}
