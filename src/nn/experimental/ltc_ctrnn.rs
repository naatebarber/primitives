use core::f64;
use ndarray::{Array1, Array2, Axis, concatenate};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

use crate::{
    f::{self, Activation},
    optim::param::{Param, ToParams},
};

pub struct Liquid {
    size: usize,
    a: Activation,
    d: Activation,

    states: Array1<f64>,
    w: Array2<f64>,
    b: Array1<f64>,
    w_tau: Array2<f64>,
    b_tau: Array1<f64>,

    dt: f64,
    prev_states: Array2<f64>,
    state_input: Array2<f64>,
    taus_pre: Array2<f64>,
    taus: Array2<f64>,
    rate: Array2<f64>,
    inner_pre: Array2<f64>,
    inner: Array2<f64>,
    update: Array2<f64>,

    d_w_tau: Array2<f64>,
    d_b_tau: Array1<f64>,
    d_w: Array2<f64>,
    d_b: Array1<f64>,
}

const TAU_EPS: f64 = 1e-6;

impl Liquid {
    pub fn new(size: usize, a: Activation, d: Activation) -> Self {
        Self {
            size,
            a,
            d,

            states: Array1::zeros(size),
            w: Array2::random((size, size), Uniform::new(-0.1, 0.1)),
            b: Array1::zeros(size),
            w_tau: f::xavier_normal((size * 2, size)),
            b_tau: Array1::zeros(size),

            dt: 0.,
            prev_states: Array2::zeros((0, 0)),
            state_input: Array2::zeros((0, 0)),
            taus_pre: Array2::zeros((0, 0)),
            taus: Array2::zeros((0, 0)),
            rate: Array2::zeros((0, 0)),
            inner_pre: Array2::zeros((0, 0)),
            inner: Array2::zeros((0, 0)),
            update: Array2::zeros((0, 0)),

            d_w_tau: Array2::zeros((0, 0)),
            d_b_tau: Array1::zeros(0),
            d_w: Array2::zeros((0, 0)),
            d_b: Array1::zeros(0),
        }
    }

    pub fn forward(&mut self, input: Array2<f64>, dt: f64) {
        assert!(input.len() == self.size);

        self.prev_states = self.states.clone().insert_axis(Axis(0));
        self.dt = dt;

        self.state_input = concatenate(Axis(1), &[self.prev_states.view(), input.view()]).unwrap();
        self.taus_pre = &self.state_input.dot(&self.w_tau) + &self.b_tau;
        self.taus = f::softplus(&self.taus_pre) + TAU_EPS;

        self.rate = self.dt / &self.taus;

        self.inner_pre = (&self.states + &self.b).insert_axis(Axis(0));
        self.inner = (self.a)(&self.inner_pre);
        self.update = self.inner.dot(&self.w) + &input;

        let next_state = (&self.states + &self.rate * &self.update) / (1. + &self.rate);
        self.states = next_state.remove_axis(Axis(0));

        if self.states.is_any_nan() {
            panic!("exploded");
        }
    }

    // CURRENTLY DISCRETE, MAKE CONTINUOUS
    pub fn backward(&mut self, d_loss: Array2<f64>) {
        // ∂y∂w_t = ∂y∂rate * ∂rate∂w_t
        // ∂y∂b_t = ∂y∂rate * ∂rate∂b_t

        let dy_drate = (&self.update - &self.prev_states) / (1. + &self.rate).powi(2);
        let drate_dtau = -self.dt / self.taus.powi(2);
        let dtau_dtau_pre = f::d_softplus(&self.taus_pre);
        let d_tau_pre = &d_loss * &dy_drate * &drate_dtau * &dtau_dtau_pre;

        self.d_w_tau = self.state_input.t().dot(&d_tau_pre);
        self.d_b_tau = d_tau_pre.sum_axis(Axis(0));

        // ∂y∂w = d_loss * rate_factor * ∂u∂w

        let rate_factor = &self.rate / (1. + &self.rate);
        let du = &d_loss * &rate_factor;
        self.d_w = du.t().dot(&self.inner);

        let d_inner = self.w.t().dot(&du.remove_axis(Axis(0)));
        let d_sigma = (self.d)(&self.inner_pre);
        self.d_b = (d_inner * d_sigma).remove_axis(Axis(0));
    }

    pub fn state(&self) -> Array1<f64> {
        self.states.clone()
    }
}

impl ToParams for Liquid {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        vec![
            Param::from_array2(&mut self.w_tau, &mut self.d_w_tau),
            Param::from_array1(&mut self.b_tau, &mut self.d_b_tau),
            Param::from_array2(&mut self.w, &mut self.d_w),
            Param::from_array1(&mut self.b, &mut self.d_b),
        ]
    }
}
