use ndarray::Array1;
use prims::{
    envs::lorenz::lorenz_step,
    f,
    nn::ctrnn::CTRNN,
    optim::{adam::AdamW, optimizer::Optimizer},
    util::plot_pair,
};
use rand::{rng, seq::IndexedRandom};

fn main() {
    let size = 64;
    let dt = 0.01;
    let step_size = 0.001;

    let train_episodes = 1000;
    let train_epochs = 100;
    let test_epochs = 10_000;

    let mut c = CTRNN::new(size, f::tanh, f::d_tanh);
    let mut optim = AdamW::default().with(&mut c);
    optim.learning_rate = 1e-4;

    let mut starts = vec![];
    let mut l_state = Array1::ones(3);
    for _ in 0..1_000 {
        l_state = lorenz_step(&l_state, dt);
        starts.push(l_state.clone());
    }

    for i in 0..train_episodes {
        let mut l_state = starts.choose(&mut rng()).unwrap().to_owned();
        let mut state = l_state.clone();

        let mut sum_loss = 0.;

        for _ in 0..train_epochs {
            l_state = lorenz_step(&l_state, dt);

            c.forward(c.concat_front(state.clone()), dt, step_size);
            state = c.slice_back(3);

            let diff = &l_state - &state;
            sum_loss += diff.mapv(|x| x.powi(2)).mean().unwrap();
            let d_loss = c.concat_back(2.0 * &diff);

            c.backward(-d_loss);
            optim.step(&mut c);
        }

        println!("episode={} sum_loss={}", i, sum_loss);

        c.reset();
    }

    let mut l_state = Array1::ones(3);
    let mut state = l_state.clone();

    let mut x = vec![];
    let mut y = vec![];
    let mut lx = vec![];
    let mut ly = vec![];

    for _ in 0..test_epochs {
        l_state = lorenz_step(&l_state, dt);
        c.forward(c.concat_front(state.clone()), dt, step_size);
        state = c.slice_back(3);

        lx.push(l_state[0]);
        ly.push(l_state[1]);

        x.push(state[0]);
        y.push(state[1]);
    }

    plot_pair(vec![(lx, ly), (x, y)]);
}
