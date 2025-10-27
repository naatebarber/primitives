use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Deserialize, Serialize};

use crate::optim::param::{Param, ToParams};

const EPS: f64 = 1e-6;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LayerNorm {
    pub gamma: Array2<f64>,
    pub beta: Array1<f64>,

    pub x: Array2<f64>,
    pub x_h: Array2<f64>,
    pub mean: Array1<f64>,
    pub var: Array1<f64>,

    pub d_gamma: Array2<f64>,
    pub d_beta: Array1<f64>,
}

impl LayerNorm {
    pub fn new(d_in: usize) -> LayerNorm {
        LayerNorm {
            gamma: Array2::ones((1, d_in)),
            beta: Array1::zeros(d_in),

            x: Array2::zeros((0, 0)),
            x_h: Array2::zeros((0, 0)),
            mean: Array1::zeros(0),
            var: Array1::zeros(0),

            d_gamma: Array2::ones((1, d_in)),
            d_beta: Array1::zeros(d_in),
        }
    }

    pub fn forward(&mut self, x: Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, feature_size) = x.dim();

        let mut x = x
            .into_shape_clone((batch_size * seq_len, feature_size))
            .unwrap();

        self.x = x.clone();
        self.mean = Array1::zeros(batch_size * seq_len);
        self.var = Array1::zeros(batch_size * seq_len);

        x.axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(i, mut x_r)| {
                let mean = x_r.mean().unwrap();
                let var = x_r.mapv(|x_v| (x_v - mean).powi(2)).mean().unwrap();
                let std = (var + EPS).sqrt();

                x_r.mapv_inplace(|v| (v - mean) / std);

                self.mean[i] = mean;
                self.var[i] = var;
            });

        self.x_h = x.clone();

        let x_n = (&x * &self.gamma) + &self.beta.view().insert_axis(Axis(0));
        x_n.into_shape_clone((batch_size, seq_len, feature_size))
            .unwrap()
    }

    pub fn backward(&mut self, d_a: Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, feature_size) = d_a.dim();
        let d_a = d_a
            .into_shape_clone((batch_size * seq_len, feature_size))
            .unwrap();

        self.d_gamma = (&d_a * &self.x_h).sum_axis(Axis(0)).insert_axis(Axis(0));
        self.d_beta = d_a.sum_axis(Axis(0));

        let (.., d_in) = d_a.dim();

        let d_xh = &d_a * &self.gamma;

        let stddev = (&self.var + EPS).sqrt().insert_axis(Axis(1));

        let sum_d_xh = d_xh.sum_axis(Axis(1)).insert_axis(Axis(1));
        let sum_d_xh_xh = (&d_xh * &self.x_h).sum_axis(Axis(1)).insert_axis(Axis(1));

        let d_x = (&d_xh * (d_in as f64)) - sum_d_xh - (&self.x_h * sum_d_xh_xh);

        let d_x = &d_x * (1.0 / d_in as f64);
        let d_x = &d_x / &stddev;

        let d_x = d_x
            .into_shape_clone((batch_size, seq_len, feature_size))
            .unwrap();
        d_x
    }
}

impl ToParams for LayerNorm {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        let mut params = vec![];

        params.push(Param::from_array2(&mut self.gamma, &mut self.d_gamma));
        params.push(Param::from_array1(&mut self.beta, &mut self.d_beta));

        params
    }
}
