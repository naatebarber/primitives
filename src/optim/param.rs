use ndarray::{Array1, Array2};

pub enum Param {
    Scalar {
        target: *mut f64,
        grad: *mut f64,
    },
    Vector {
        target: *mut Array1<f64>,
        grad: *mut Array1<f64>,
    },
    Matrix {
        target: *mut Array2<f64>,
        grad: *mut Array2<f64>,
    },
}

impl Param {
    pub fn from_scalars(target: &mut f64, grad: &mut f64) -> Param {
        Param::Scalar {
            target: target as *mut f64,
            grad: grad as *mut f64,
        }
    }

    pub fn from_array1(target: &mut Array1<f64>, grad: &mut Array1<f64>) -> Param {
        Param::Vector {
            target: target as *mut Array1<f64>,
            grad: grad as *mut Array1<f64>,
        }
    }

    pub fn from_array2(target: &mut Array2<f64>, grad: &mut Array2<f64>) -> Param {
        Param::Matrix {
            target: target as *mut Array2<f64>,
            grad: grad as *mut Array2<f64>,
        }
    }
}

pub trait ToParams {
    fn params(&mut self) -> Vec<Param>;
}
