use ndarray::Array1;

pub fn lorenz_step(state: &Array1<f64>, dt: f64) -> Array1<f64> {
    // Parameters
    let sigma = 10.0;
    let rho = 28.0;
    let beta = 8.0 / 3.0;

    let x = state[0];
    let y = state[1];
    let z = state[2];

    // Lorenz ODEs
    let dx = sigma * (y - x);
    let dy = x * (rho - z) - y;
    let dz = x * y - beta * z;

    // Euler integration step
    Array1::from(vec![x + dx * dt, y + dy * dt, z + dz * dt])
}
