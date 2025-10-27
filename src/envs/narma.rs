use rand::random;

pub struct Narma10 {
    pub ys: [f64; 10],
    pub xs: [f64; 10],
}

impl Narma10 {
    pub fn new() -> Self {
        Self {
            ys: [0.0; 10],
            xs: [0.0; 10],
        }
    }
    pub fn new_random() -> Self {
        let ys: [f64; 10] = [random(); 10];
        let xs: [f64; 10] = [random(); 10];

        Self { ys, xs }
    }

    pub fn step(&mut self, x_t: f64) -> f64 {
        // compute NARMA10 output using canonical equation
        // y_t = 0.3*y_{t-1} + 0.05*y_{t-1} * (sum_{i=1}^{10} y_{t-i})
        //       + 1.5*x_{t-10}*x_t + 0.1

        let y_prev = self.ys[0];
        let y_sum: f64 = self.ys.iter().sum();
        let x_prev10 = self.xs[9]; // x_{t-10}

        let y_t = 0.3 * y_prev + 0.05 * y_prev * y_sum + 1.5 * x_prev10 * x_t + 0.1;

        // shift histories (FIFO-style)
        self.ys.rotate_right(1);
        self.xs.rotate_right(1);

        // store new values
        self.ys[0] = y_t;
        self.xs[0] = x_t;

        y_t
    }
}
