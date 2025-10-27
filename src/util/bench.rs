use ndarray::Array1;

pub struct Bench {}

impl Bench {
    pub fn mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        (y_true - y_pred).mapv(|v| v.powi(2)).mean().unwrap()
    }

    pub fn nmse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let mse_val = Bench::mse(y_true, y_pred);
        let var_y = (y_true - y_true.mean().unwrap())
            .mapv(|v| v.powi(2))
            .mean()
            .unwrap();
        mse_val / var_y
    }

    pub fn r2(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        1.0 - Bench::nmse(y_true, y_pred)
    }

    pub fn pearson_r(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let y_true_centered = y_true - y_true.mean().unwrap();
        let y_pred_centered = y_pred - y_pred.mean().unwrap();
        let numerator = (y_true_centered.clone() * y_pred_centered.clone())
            .mean()
            .unwrap();
        let denom = (y_true_centered.mapv(|v| v.powi(2)).mean().unwrap()
            * y_pred_centered.mapv(|v| v.powi(2)).mean().unwrap())
        .sqrt();
        numerator / denom
    }

    pub fn score(y_true: &Array1<f64>, y_pred: &Array1<f64>) {
        let mse_val = Bench::mse(&y_true, &y_pred);
        let nmse_val = Bench::nmse(&y_true, &y_pred);
        let r2_val = Bench::r2(&y_true, &y_pred);
        let corr = Bench::pearson_r(&y_true, &y_pred);
        let rmse_val = mse_val.sqrt();

        println!("\n=== Bench Metrics ===");
        println!("MSE   = {:.6}", mse_val);
        println!("RMSE  = {:.6}", rmse_val);
        println!("NMSE  = {:.6}", nmse_val);
        println!("R^2   = {:.6}", r2_val);
        println!("r     = {:.6}", corr);
    }
}
