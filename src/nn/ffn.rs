use ndarray::{Array1, Array2, Axis};

use crate::{
    f,
    optim::param::{Param, ToParams},
};

pub type LayerDef = (usize, usize, f::Activation, f::Activation);

#[derive(Debug, Clone)]
pub struct Layer {
    pub w: Array2<f64>,
    pub b: Array1<f64>,

    pub activation: f::Activation,
    pub d_activation: f::Activation,

    pub x: Array2<f64>,
    pub z: Array2<f64>,

    pub d_w: Array2<f64>,
    pub d_b: Array1<f64>,
}

impl Layer {
    pub fn new(
        d_in: usize,
        d_out: usize,
        activation: f::Activation,
        d_activation: f::Activation,
    ) -> Layer {
        Layer {
            w: f::he((d_in, d_out)),
            b: Array1::zeros(d_out),
            activation,
            d_activation,

            x: Array2::zeros((0, 0)),
            z: Array2::zeros((0, 0)),

            d_w: Array2::zeros((d_in, d_out)),
            d_b: Array1::zeros(d_out),
        }
    }

    pub fn forward(&mut self, x: Array2<f64>, grad: bool) -> Array2<f64> {
        let z = x.dot(&self.w) + &self.b;

        if grad {
            self.x = x.clone();
            self.z = z.clone();
        }

        (self.activation)(&z)
    }

    pub fn backward(&mut self, d_a: Array2<f64>) -> Array2<f64> {
        let d_z = d_a * &(self.d_activation)(&self.z);
        self.d_w = self.x.t().dot(&d_z);
        self.d_b = d_z.sum_axis(Axis(0));

        d_z.dot(&self.w.t())
    }
}

impl ToParams for Layer {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        vec![
            Param::from_array2(&mut self.w, &mut self.d_w),
            Param::from_array1(&mut self.b, &mut self.d_b),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct FFN {
    pub layers: Vec<Layer>,
}

impl FFN {
    pub fn new(layers: Vec<LayerDef>) -> FFN {
        FFN {
            layers: layers
                .into_iter()
                .map(|(d_in, d_out, a, d_a)| Layer::new(d_in, d_out, a, d_a))
                .collect(),
        }
    }

    pub fn forward(&mut self, mut x: Array2<f64>, grad: bool) -> Array2<f64> {
        for layer in self.layers.iter_mut() {
            x = layer.forward(x, grad)
        }

        x
    }

    pub fn backward(&mut self, mut d_a: Array2<f64>) -> Array2<f64> {
        for layer in self.layers.iter_mut().rev() {
            d_a = layer.backward(d_a);
        }

        d_a
    }
}

impl ToParams for FFN {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        let mut params = vec![];
        self.layers
            .iter_mut()
            .for_each(|l| params.append(&mut l.params()));
        params
    }
}
