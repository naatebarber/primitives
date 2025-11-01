use core::f64;
use ndarray::{Array1, Array2, Axis, Slice, concatenate};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

use crate::{
    f::Activation,
    optim::param::{Param, ToParams},
};

// TODO implement full BPTT for tn -> tn+m, not just integrated last step

pub struct VertletCTRNN {
    size: usize,
    a: Activation,
    d: Activation,
    tau_min: f64,
    tau_max: f64,

    prev_state: Option<Array1<f64>>,
    pub states: Array1<f64>,
    weights: Array2<f64>,
    biases: Array1<f64>,
    taus: Array1<f64>,

    d_loss: Array2<f64>,
    step_dt: Vec<f64>,
    x_prev: Array2<f64>,
    preactivations: Array2<f64>,
    activations: Array2<f64>,
    dx_dt: Array2<f64>,

    d_taus: Array1<f64>,
    d_weights: Array2<f64>,
    d_biases: Array1<f64>,
}

impl VertletCTRNN {
    pub fn new(size: usize, a: Activation, d: Activation) -> Self {
        let tau_min = 0.01;
        let tau_max = 3.;

        Self {
            size,
            a,
            d,
            tau_min,
            tau_max,

            prev_state: None,
            states: Array1::zeros(size),
            weights: Array2::random((size, size), Uniform::new(-0.1, 0.1)),
            biases: Array1::zeros(size),
            taus: Array1::random(size, Uniform::new(tau_min, tau_max)),

            d_loss: Array2::zeros((0, size)),
            step_dt: Vec::new(),
            x_prev: Array2::zeros((0, size)),
            preactivations: Array2::zeros((0, size)),
            activations: Array2::zeros((0, size)),
            dx_dt: Array2::zeros((0, size)),

            d_taus: Array1::zeros(size),
            d_weights: Array2::zeros((size, size)),
            d_biases: Array1::zeros(size),
        }
    }

    pub fn log_taus(&mut self, low: f64, high: f64) {
        self.tau_min = low;
        self.tau_max = high;

        self.taus = Array1::random(
            self.size,
            Uniform::new(self.tau_min.ln(), self.tau_max.ln()),
        );
    }

    pub fn uniform_taus(&mut self, low: f64, high: f64) {
        self.tau_min = low;
        self.tau_max = high;

        self.taus = Array1::random(self.size, Uniform::new(self.tau_min, self.tau_max));
    }

    pub fn vertlet_step(&mut self, input: &Array1<f64>, last_state: Option<Array1<f64>>, dt: f64) {
        self.step_dt.push(dt);
        self.x_prev.push(Axis(0), self.states.view()).unwrap();
        self.prev_state = Some(self.states.clone());
        self.d_loss
            .push(Axis(0), Array1::zeros(self.size).view())
            .unwrap();

        let preactivations = &self.states + &self.biases;
        self.preactivations
            .push(Axis(0), preactivations.view())
            .unwrap();

        let activations = (self.a)(&preactivations.insert_axis(Axis(0))).remove_axis(Axis(0));
        self.activations.push(Axis(0), activations.view()).unwrap();

        let recurrent = self.weights.dot(&activations);
        let update = &recurrent + input;
        self.dx_dt.push(Axis(0), update.view()).unwrap();

        let smoothed_taus = self.tau_min + self.taus.exp();
        let rate = dt / smoothed_taus;

        let prev_state = if let Some(last_state) = last_state {
            last_state.clone()
        } else {
            let init_velocity = Array1::zeros(self.size);
            &self.states - &init_velocity * &rate + 0.5 * &update * rate.powi(2)
        };

        self.states = 2. * &self.states - prev_state + update * rate.powi(2);
    }

    pub fn forward(&mut self, input: Array1<f64>, dt: f64, step_size: f64) {
        assert!(input.len() == self.size);

        let mut t = 0.;

        while t < dt {
            self.vertlet_step(&input, self.prev_state.clone(), step_size);
            t += step_size;
        }
    }

    pub fn backward_vertlet_step(&mut self, d_loss: &Array1<f64>, t: usize) {
        let dt = self.step_dt[t];
        let preactivation = self.preactivations.row(t);
        let _activation = self.activations.row(t); // unused
        let dx_dt = self.dx_dt.row(t);

        // tau gradients
        let exp_taus = self.taus.exp();
        let smoothed_taus = self.tau_min + &exp_taus;
        let r = dt / &smoothed_taus;

        let dy_dr = 2. * &dx_dt * &r;
        let dr_dst = -dt / &smoothed_taus.powi(2);
        let dst_dt = exp_taus;

        let dy_dt = dy_dr * dr_dst * dst_dt;
        let dl_dt = d_loss * dy_dt;
        self.d_taus = dl_dt;

        // weight gradients
        let r_factor = r.powi(2);
        let grad_scale = d_loss * r_factor;
        self.d_weights = grad_scale
            .clone()
            .insert_axis(Axis(1))
            .dot(&dx_dt.insert_axis(Axis(0)));

        // bias gradients
        let d_bias_dz =
            (self.d)(&preactivation.to_owned().insert_axis(Axis(0))).remove_axis(Axis(0));
        self.d_biases = self.weights.t().dot(&grad_scale) * d_bias_dz;
    }

    pub fn backward(&mut self, d_loss: Array1<f64>) {
        let time_steps = self.step_dt.len();

        self.d_loss.row_mut(time_steps - 1).assign(&d_loss);

        let mut d_state_next = Array1::zeros(self.size);

        // calculate vertlet jacobians. each contains the ∂x+1/∂x of the prev step (next if time
        // moving backwards)

        // s_t+1 = 2s_t - s_t-1 + u * t^2
        // ∂s1/∂s = 2 - ∂s/∂s-1 + 2ut
        // let mut jacobians = vec![];
        // for t in (1..time_steps) {
        //
        // }

        for t in (0..self.step_dt.len()).rev() {
            self.backward_vertlet_step(&d_state_next, t);

            let dt = self.step_dt[t];
            let exp_taus = self.taus.exp();
            let smoothed_taus = self.tau_min + exp_taus;
            let r = dt / smoothed_taus;

            // ∂x_{t+1}/∂x_t ≈ 1 - (dt/τ) + (dt/τ)*f'(x_t)
            // for small dt, often approximated as 1/(1 + r)
            let jacobian = 1. / (1. + r);
            d_state_next = &self.d_loss.row(t) + &d_state_next * &jacobian;
        }
    }

    pub fn zero_grads(&mut self) {
        self.step_dt = Vec::new();
        self.x_prev = Array2::zeros((0, self.size));
        self.preactivations = Array2::zeros((0, self.size));
        self.activations = Array2::zeros((0, self.size));
        self.dx_dt = Array2::zeros((0, self.size));
    }

    pub fn state(&self) -> Array1<f64> {
        self.states.clone()
    }

    pub fn slice_front(&self, n: usize) -> Array1<f64> {
        if n > self.size {
            panic!("front neural slice greater than network size");
        }

        self.states
            .slice_axis(Axis(0), Slice::from(0..n))
            .to_owned()
    }

    pub fn slice_back(&self, n: usize) -> Array1<f64> {
        if n > self.size {
            panic!("back neural slice greater than network size");
        }

        self.states
            .slice_axis(Axis(0), Slice::from((self.size - n)..self.size))
            .to_owned()
    }

    pub fn concat_front(&self, x: Array1<f64>) -> Array1<f64> {
        if x.len() > self.size {
            panic!("front neural concat greater than network size");
        }

        let diff = self.size - x.len();
        let zeros = Array1::zeros(diff);
        concatenate(Axis(0), &[x.view(), zeros.view()]).unwrap()
    }

    pub fn concat_back(&self, x: Array1<f64>) -> Array1<f64> {
        if x.len() > self.size {
            panic!("back neural concat greater than network size");
        }

        let diff = self.size - x.len();
        let zeros = Array1::zeros(diff);
        concatenate(Axis(0), &[zeros.view(), x.view()]).unwrap()
    }

    pub fn reset(&mut self) {
        self.states = Array1::zeros(self.size);
        self.prev_state = None;
    }
}

impl ToParams for VertletCTRNN {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        vec![
            Param::from_array1(&mut self.taus, &mut self.d_taus),
            Param::from_array2(&mut self.weights, &mut self.d_weights),
            Param::from_array1(&mut self.biases, &mut self.d_biases),
        ]
    }
}
