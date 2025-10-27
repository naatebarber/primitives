use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use std::f64::consts::PI;

pub struct PendulumEnv {
    pub state: [f64; 2], // [theta, theta_dot]
    pub max_speed: f64,
    pub max_torque: f64,
    pub dt: f64,
    pub g: f64,
    pub m: f64,
    pub l: f64,
}

impl PendulumEnv {
    pub fn new() -> Self {
        Self {
            state: [0.0, 0.0],
            max_speed: 8.0,
            max_torque: 2.0,
            dt: 0.05,
            g: 10.0,
            m: 1.0,
            l: 1.0,
        }
    }

    /// Reset environment with random initial angle
    pub fn reset(&mut self, rng: &mut impl Rng) -> Array2<f64> {
        let theta = rng.random_range(-PI..PI);
        let theta_dot = rng.random_range(-1.0..1.0);
        self.state = [theta, theta_dot];
        self.get_obs()
    }

    /// Step with continuous action `a` (torque)
    pub fn step(&mut self, action: f64) -> (Array2<f64>, f64, bool) {
        // clip torque
        let u = action.clamp(-self.max_torque, self.max_torque);

        let th = self.state[0];
        let thdot = self.state[1];

        // dynamics: theta'' = (g * sin(theta) + cos(theta)*(-u)) / (l) etc
        let new_thdot = thdot
            + (3.0 * self.g / (2.0 * self.l) * th.sin() + 3.0 / (self.m * self.l.powi(2)) * u)
                * self.dt;
        let new_thdot = new_thdot.clamp(-self.max_speed, self.max_speed);

        let new_th = th + new_thdot * self.dt;

        self.state = [wrap_angle(new_th), new_thdot];

        // reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*u^2)
        let reward = -(self.state[0].powi(2) + 0.1 * new_thdot.powi(2) + 0.001 * u.powi(2));

        (self.get_obs(), reward, false) // pendulum never "done"
    }

    /// Observation: [cos(theta), sin(theta), theta_dot]
    pub fn get_obs(&self) -> Array2<f64> {
        Array1::from_vec(vec![
            self.state[0].cos(),
            self.state[0].sin(),
            self.state[1],
        ])
        .insert_axis(Axis(0))
    }
}

/// wrap angle into [-pi, pi]
fn wrap_angle(x: f64) -> f64 {
    ((x + PI) % (2.0 * PI)) - PI
}
