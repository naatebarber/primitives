use core::f64;

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use ndarray_stats::QuantileExt;
use rand::{Rng, rngs::ThreadRng};

pub type Activation = fn(&Array2<f64>) -> Array2<f64>;

pub fn relu(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.max(0.))
}

pub fn d_relu(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.signum().max(0.))
}

pub fn tanh(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.tanh())
}

pub fn d_tanh(x: &Array2<f64>) -> Array2<f64> {
    1. - (x.mapv(|v| v.tanh())).powi(2)
}

pub fn exp(x: &Array2<f64>) -> Array2<f64> {
    x.exp()
}

pub fn d_exp(x: &Array2<f64>) -> Array2<f64> {
    x.exp()
}

pub fn softplus(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| if v > 20.0 { v } else { (1.0 + v.exp()).ln() })
}

pub fn d_softplus(x: &Array2<f64>) -> Array2<f64> {
    sigmoid(x)
}

pub fn ident(x: &Array2<f64>) -> Array2<f64> {
    x.to_owned()
}

pub fn d_ident(x: &Array2<f64>) -> Array2<f64> {
    Array2::ones(x.dim())
}

pub fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

pub fn d_sigmoid(x: &Array2<f64>) -> Array2<f64> {
    let s = sigmoid(x);
    &s * &(1.0 - &s)
}

pub fn he(shape: (usize, usize)) -> Array2<f64> {
    let bound = f64::sqrt(2.) / f64::sqrt(shape.0 as f64);
    return Array2::random(shape, Uniform::new(-bound, bound));
}

pub fn xavier_normal(shape: (usize, usize)) -> Array2<f64> {
    let std = (2. / ((shape.0 + shape.1) as f64)).sqrt();
    Array2::random(
        shape,
        ndarray_rand::rand_distr::Normal::new(0., std).unwrap(),
    )
}

pub fn softmax(x: Array2<f64>) -> Array2<f64> {
    let maxes = x
        .map_axis(Axis(1), |row| row.max().cloned().unwrap_or(1e-4))
        .insert_axis(Axis(1));

    let mut d = &x - &maxes;

    d.mapv_inplace(|x| x.exp());

    let sums = d.map_axis(Axis(1), |row| row.sum()).insert_axis(Axis(1));

    let last = &d / &sums;
    return last;
}

pub fn softmax_vector_jacobian_product(
    upstream: &Array2<f64>,
    softmax_out: &Array2<f64>,
) -> Array2<f64> {
    let mut grad = upstream.clone();

    for ((mut g_row, s_row), u_row) in grad
        .axis_iter_mut(Axis(0))
        .zip(softmax_out.axis_iter(Axis(0)))
        .zip(upstream.axis_iter(Axis(0)))
    {
        let dot = u_row.dot(&s_row);

        for ((g, &s), &u) in g_row.iter_mut().zip(s_row.iter()).zip(u_row.iter()) {
            *g = s * (u - dot);
        }
    }

    grad
}

// TODO: Use with Gumbel max sampling, avoid standard softmax entirely.
pub fn log_softmax(x: Array2<f64>) -> Array2<f64> {
    let maxes = x
        .map_axis(Axis(1), |row| row.max().cloned().unwrap_or(1e-4))
        .insert_axis(Axis(1));

    let d = &x - &maxes;
    let sums = d
        .map_axis(Axis(1), |row| row.exp().sum())
        .insert_axis(Axis(1));

    &d - sums.ln()
}

pub fn l2(v: &Array1<f64>) -> f64 {
    v.pow2().sum().sqrt()
}

pub fn l2_norm(v: &Array1<f64>) -> Array1<f64> {
    v / l2(v)
}

pub fn clip_grad(mut grad: Array2<f64>, clip: f64) -> Array2<f64> {
    let norm_sq = grad.mapv(|x| x * x).sum();
    let norm = norm_sq.sqrt();

    if norm > clip {
        grad.mapv_inplace(|x| x * (clip / (norm + 1e-6)));
    }

    grad
}

pub fn sample_categorical(probs: &Array1<f64>, rng: &mut ThreadRng) -> usize {
    let mut u: f64 = rng.random();

    for (i, &p) in probs.iter().enumerate() {
        if u < p {
            return i;
        }

        u -= p;
    }

    return 0;
}

pub fn sample_gumbel_categorical(log_probs: &Array1<f64>, rng: &mut ThreadRng) -> usize {
    let u = (0..log_probs.len())
        .map(|_| -f64::ln(-f64::ln(rng.random())))
        .collect::<Array1<f64>>();

    (log_probs + u).argmax().unwrap()
}

pub fn gaussian_log_prob(x: &Array2<f64>, u: &Array2<f64>, o: &Array2<f64>) -> Array1<f64> {
    let eps = 1e-8;
    let quadratic = (x - u).powi(2) / o.powi(2);
    let norm = 2. * o.map(|v| v.max(eps).ln());
    let constant = (f64::consts::PI * 2.).ln();

    let dims = -0.5 * (quadratic + norm + constant);

    dims.sum_axis(Axis(1))
}

pub fn tanh_gaussian_correction(a_raw: &Array2<f64>) -> Array1<f64> {
    let d_tanh = d_tanh(a_raw);
    d_tanh.ln().sum_axis(Axis(1))
}

pub fn tanh_gaussian_correction_eps(a_raw: &Array2<f64>) -> Array1<f64> {
    let deriv = 1.0 - a_raw.mapv(|u| u.tanh().powi(2)); // d_tanh(u)
    deriv.mapv(|v| (v + 1e-8).ln()).sum_axis(Axis(1))
}
