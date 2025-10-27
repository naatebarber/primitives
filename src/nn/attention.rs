use ndarray::{Array1, Array2, Array3, Array4, Axis, stack};
use serde::{Deserialize, Serialize};

use crate::{
    f,
    optim::param::{Param, ToParams},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AttentionHead {
    pub d_in: usize,
    pub d_head: usize,
    pub n_head: usize,

    pub qkv_w: Array2<f64>,
    pub qkv_b: Array1<f64>,
    pub o_w: Array2<f64>,
    pub o_b: Array1<f64>,

    pub x: Array2<f64>,
    pub q: Array3<f64>,
    pub k: Array3<f64>,
    pub v: Array3<f64>,
    pub qkv: Array2<f64>,

    pub scores: Array3<f64>,
    pub attention: Array2<f64>,
    pub o: Array2<f64>,

    pub d_qkv_w: Array2<f64>,
    pub d_qkv_b: Array1<f64>,
    pub d_o_w: Array2<f64>,
    pub d_o_b: Array1<f64>,
}

impl AttentionHead {
    pub fn new(d_in: usize, d_head: usize, n_head: usize) -> AttentionHead {
        AttentionHead {
            d_in,
            d_head,
            n_head,

            qkv_w: f::xavier_normal((d_in, 3 * d_head * n_head)),
            qkv_b: Array1::zeros(3 * d_head * n_head),
            o_w: f::xavier_normal((d_head * n_head, d_in)),
            o_b: Array1::zeros(d_in),

            x: Array2::zeros((0, 0)),
            q: Array3::zeros((0, 0, 0)),
            k: Array3::zeros((0, 0, 0)),
            v: Array3::zeros((0, 0, 0)),
            qkv: Array2::zeros((0, 0)),

            scores: Array3::zeros((0, 0, 0)),
            attention: Array2::zeros((0, 0)),
            o: Array2::zeros((0, 0)),

            d_qkv_w: Array2::zeros((0, 0)),
            d_qkv_b: Array1::zeros(0),
            d_o_w: Array2::zeros((0, 0)),
            d_o_b: Array1::zeros(0),
        }
    }

    pub fn forward(&mut self, x: &Array3<f64>, mask: &Array2<f64>, auto: bool) -> Array3<f64> {
        let (batch_size, seq_len, feature_size) = x.dim();

        let padding_mask = mask
            .mapv(|x| if x == 0. { f64::NEG_INFINITY } else { 0. }) // (B, S)
            .insert_axis(Axis(1)) // (B, 1, S)
            .insert_axis(Axis(1)) // (B, 1, 1, S)
            .broadcast((batch_size, self.n_head, 1, seq_len)) // (B, H, 1, S)
            .unwrap()
            .to_shape((batch_size * self.n_head, 1, seq_len)) // (B * H, 1, S)
            .unwrap()
            .to_owned();

        let x = x
            .clone()
            .into_shape_clone((batch_size * seq_len, feature_size))
            .unwrap();

        let qkv = x.dot(&self.qkv_w) + &self.qkv_b.view().insert_axis(Axis(0));

        if auto {
            self.x = x.clone();
            self.qkv = qkv.clone();
        }

        let qkv = qkv
            .into_shape_clone((batch_size, seq_len, 3, self.n_head, self.d_head))
            .unwrap(); // (B, S, 3, H, dH)
        let qkv = qkv.permuted_axes([0, 3, 1, 2, 4]); // (B, H, S, 3, dH)

        let q: Array4<f64> = qkv.index_axis(Axis(3), 0).to_owned(); // (B, H, S, dH)
        let k: Array4<f64> = qkv.index_axis(Axis(3), 1).to_owned(); // ..
        let v: Array4<f64> = qkv.index_axis(Axis(3), 2).to_owned(); // ..

        let q = q
            .into_shape_clone((batch_size * self.n_head, seq_len, self.d_head))
            .unwrap(); // (B * H, S, dH)
        let k = k
            .into_shape_clone((batch_size * self.n_head, seq_len, self.d_head))
            .unwrap(); // ..
        let v = v
            .into_shape_clone((batch_size * self.n_head, seq_len, self.d_head))
            .unwrap(); // ..

        if auto {
            self.q = q.clone();
            self.k = k.clone();
            self.v = v.clone();
        }

        let mut b_scores: Array3<f64> = Array3::zeros((batch_size * self.n_head, seq_len, seq_len)); // (B * H, S, S)
        let mut b_attention: Array3<f64> =
            Array3::zeros((batch_size * self.n_head, seq_len, self.d_head)); // (B * H, S, dH)

        for i in 0..(batch_size * self.n_head) {
            let q_i = q.index_axis(Axis(0), i); // (S, dH)
            let k_i = k.index_axis(Axis(0), i); // ..
            let v_i = v.index_axis(Axis(0), i); // ..

            let padding_mask_i = padding_mask.index_axis(Axis(0), i); // (1, S)

            let mut scores = q_i.dot(&k_i.t()) / f64::sqrt(self.d_head as f64); // (S, S)

            scores += &padding_mask_i.broadcast((seq_len, seq_len)).unwrap();

            let weights = f::softmax(scores); // (S, S)
            let attention = weights.dot(&v_i); // (S, dH)

            b_scores.index_axis_mut(Axis(0), i).assign(&weights);
            b_attention.index_axis_mut(Axis(0), i).assign(&attention);
        }

        if auto {
            self.scores = b_scores.clone();
        }

        let attention = b_attention
            .into_shape_clone((batch_size, self.n_head, seq_len, self.d_head))
            .unwrap();
        let attention = attention.permuted_axes([0, 2, 1, 3]).to_owned();
        let attention = attention
            .into_shape_clone((batch_size * seq_len, self.n_head * self.d_head))
            .unwrap();

        let output = attention.dot(&self.o_w) + self.o_b.view().insert_axis(Axis(0));

        if auto {
            self.o = output.clone();
            self.attention = attention.clone();
        }

        let output = output
            .into_shape_clone((batch_size, seq_len, feature_size))
            .unwrap();
        output
    }

    pub fn backward(&mut self, d_a: Array3<f64>) -> Array3<f64> {
        let (batch_size, sequence_len, feature_size) = d_a.dim();
        let d_a = d_a
            .into_shape_clone((batch_size * sequence_len, feature_size))
            .unwrap();

        self.d_o_w = self.attention.t().dot(&d_a);
        self.d_o_b = d_a.sum_axis(Axis(0));

        let d_a = d_a.dot(&self.o_w.t()); // (B * S, H * dH)

        let d_a = d_a
            .into_shape_clone((batch_size, sequence_len, self.n_head, self.d_head))
            .unwrap(); // (B, S, H, dH)
        let d_a = d_a.permuted_axes([0, 2, 1, 3]); // (B, H, S, dH)
        let d_a = d_a
            .into_shape_clone((batch_size * self.n_head, sequence_len, self.d_head))
            .unwrap(); // (B * H, S, dH)

        let mut d_v: Array3<f64> =
            Array3::zeros((batch_size * self.n_head, sequence_len, self.d_head));
        let mut d_q: Array3<f64> =
            Array3::zeros((batch_size * self.n_head, sequence_len, self.d_head));
        let mut d_k: Array3<f64> =
            Array3::zeros((batch_size * self.n_head, sequence_len, self.d_head));

        for i in 0..(batch_size * self.n_head) {
            let attn_score = self.scores.index_axis(Axis(0), i);
            let d_out = d_a.index_axis(Axis(0), i);
            let v = self.v.index_axis(Axis(0), i);
            let q = self.q.index_axis(Axis(0), i);
            let k = self.k.index_axis(Axis(0), i);

            let grad_v = attn_score.t().dot(&d_out);
            d_v.index_axis_mut(Axis(0), i).assign(&grad_v);

            let d_softmax = d_out.dot(&v.t());
            let d_scores = f::softmax_vector_jacobian_product(&d_softmax, &attn_score.to_owned());

            let grad_q = d_scores.dot(&k) / f64::sqrt(self.d_head as f64);
            d_q.index_axis_mut(Axis(0), i).assign(&grad_q);

            let grad_k = d_scores.t().dot(&q) / f64::sqrt(self.d_head as f64);
            d_k.index_axis_mut(Axis(0), i).assign(&grad_k);
        }

        let d_v = d_v
            .into_shape_clone((batch_size, self.n_head, sequence_len, self.d_head))
            .unwrap();
        let d_q = d_q
            .into_shape_clone((batch_size, self.n_head, sequence_len, self.d_head))
            .unwrap();
        let d_k = d_k
            .into_shape_clone((batch_size, self.n_head, sequence_len, self.d_head))
            .unwrap();

        let d_qkv = stack![Axis(0), d_q.view(), d_k.view(), d_v.view()]; // (3, B, H, S, dH)

        let d_qkv = d_qkv.permuted_axes([1, 3, 0, 2, 4]); // (B, S, 3, H, dH)
        let d_qkv = d_qkv
            .into_shape_clone((batch_size * sequence_len, 3 * self.n_head * self.d_head))
            .unwrap();

        self.d_qkv_w = self.x.t().dot(&d_qkv);
        self.d_qkv_b = d_qkv.sum_axis(Axis(0));

        let d_x_qkv = d_qkv.dot(&self.qkv_w.t());

        let d_x_qkv = d_x_qkv
            .into_shape_clone((batch_size, sequence_len, feature_size))
            .unwrap();
        d_x_qkv
    }
}

impl ToParams for AttentionHead {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        let mut params = vec![];

        params.push(Param::from_array2(&mut self.qkv_w, &mut self.d_qkv_w));
        params.push(Param::from_array1(&mut self.qkv_b, &mut self.d_qkv_b));
        params.push(Param::from_array2(&mut self.o_w, &mut self.d_o_w));
        params.push(Param::from_array1(&mut self.o_b, &mut self.d_o_b));

        params
    }
}
