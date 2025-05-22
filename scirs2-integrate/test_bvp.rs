use ndarray::{array, ArrayView1};
use scirs2_integrate::bvp::solve_bvp;

fn main() {
    // Simple test case
    let fun = |_x: f64, y: ArrayView1<f64>| array![y[1], -y[0]];
    let bc = |ya: ArrayView1<f64>, yb: ArrayView1<f64>| array![ya[0], yb[0]];
    
    let x = vec![0.0, 1.57, 3.14];
    let y_init = vec![
        array![0.0, 1.0],
        array![1.0, 0.0],
        array![0.0, -1.0],
    ];
    
    match solve_bvp(fun, bc, Some(x), y_init, None) {
        Ok(result) => println!("Success: {:?}", result.success),
        Err(e) => println!("Error: {:?}", e),
    }
}