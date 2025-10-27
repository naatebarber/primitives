use plotly::{Plot, Scatter, common::Mode};

pub fn plot(l: Vec<Vec<f64>>) {
    let mut plot = Plot::new();
    for x in l {
        let y = (0..x.len()).collect::<Vec<usize>>();
        let pred_trace = Scatter::new(y, x).mode(Mode::Lines);
        plot.add_trace(pred_trace);
    }
    plot.show();
}

pub fn plot_pair(l: Vec<(Vec<f64>, Vec<f64>)>) {
    let mut plot = Plot::new();
    for (x, y) in l {
        let pred_trace = Scatter::new(y, x).mode(Mode::Lines);
        plot.add_trace(pred_trace);
    }
    plot.show();
}
