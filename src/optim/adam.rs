use ndarray::{Array1, Array2, Axis};

use crate::f;

use super::{
    optimizer::Optimizer,
    param::{Param, ToParams},
};

pub struct AdamMatrix {
    pub m: Array2<f64>,
    pub v: Array2<f64>,
}

pub struct AdamVector {
    pub m: Array1<f64>,
    pub v: Array1<f64>,
}

pub struct AdamScalar {
    pub m: f64,
    pub v: f64,
}

#[derive(Debug, Clone)]
pub enum AdamParam {
    Scalar { m: f64, v: f64 },
    Vector { m: Array1<f64>, v: Array1<f64> },
    Matrix { m: Array2<f64>, v: Array2<f64> },
}

#[derive(Debug, Clone)]
pub struct AdamW {
    pub learning_rate: f64,
    pub clip_grad: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub t: usize,

    pub params: Vec<AdamParam>,
}

impl Default for AdamW {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            learning_rate: 3e-4,
            clip_grad: 0.3,
            weight_decay: 0.001,
            epsilon: 1e-8,
            t: 1,
            params: vec![],
        }
    }
}

impl Optimizer for AdamW {
    fn with(mut self, optimizable: &mut impl ToParams) -> Self {
        let params = optimizable.params();

        unsafe {
            for param in params {
                self.params.push(match param {
                    Param::Scalar { .. } => AdamParam::Scalar { m: 0., v: 0. },
                    Param::Vector { target, .. } => AdamParam::Vector {
                        m: Array1::zeros((*target).len()),
                        v: Array1::zeros((*target).len()),
                    },
                    Param::Matrix { target, .. } => AdamParam::Matrix {
                        m: Array2::zeros((*target).dim()),
                        v: Array2::zeros((*target).dim()),
                    },
                });
            }
        }

        self
    }

    fn step(&mut self, optimizable: &mut impl ToParams) {
        self.t += 1;
        let bc1 = self.beta1.powi(self.t as i32);
        let bc2 = self.beta2.powi(self.t as i32);

        unsafe {
            for (param, adam_param) in optimizable.params().into_iter().zip(self.params.iter_mut())
            {
                match (adam_param, param) {
                    (AdamParam::Scalar { m, v }, Param::Scalar { target, grad }) => {
                        let g = (*grad).to_owned();
                        *m = self.beta1 * *m + (1. - self.beta1) * g;
                        *v = self.beta2 * *v + (1. - self.beta2) * g.powi(2);

                        let m_hat = (*m) / (1. - bc1);
                        let v_hat = (*v) / (1. - bc2);

                        let delta = self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                        *target -= delta;
                    }
                    (AdamParam::Vector { m, v }, Param::Vector { target, grad }) => {
                        let g = (*grad).to_owned();
                        let g = f::clip_grad(g.insert_axis(Axis(0)), self.clip_grad)
                            .remove_axis(Axis(0));

                        *m = self.beta1 * &(*m) + (1. - self.beta1) * &g;
                        *v = self.beta2 * &(*v) + (1. - self.beta2) * g.powi(2);

                        let m_hat = &(*m) / (1. - bc1);
                        let v_hat = &(*v) / (1. - bc2);

                        let delta = self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                        *target -= &delta;
                    }
                    (AdamParam::Matrix { m, v }, Param::Matrix { target, grad }) => {
                        let g = (*grad).to_owned();
                        let g = f::clip_grad(g, self.clip_grad);

                        *m = self.beta1 * &(*m) + (1. - self.beta1) * &g;
                        *v = self.beta2 * &(*v) + (1. - self.beta2) * g.powi(2);

                        let m_hat = &(*m) / (1. - bc1);
                        let v_hat = &(*v) / (1. - bc2);

                        let delta = m_hat / (v_hat.sqrt() + self.epsilon);
                        *target -=
                            &(self.learning_rate * &(&delta + self.weight_decay * &(*target)));
                    }
                    _ => (),
                }
            }
        }
    }
}
