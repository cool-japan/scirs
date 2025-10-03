//! Simple test for cubature integration

use scirs2_core::ndarray::Array1;
use scirs2_integrate::cubature::{cubature, Bound};

#[allow(dead_code)]
fn main() {
    println!("Running simple cubature test...");

    // Define a 2D integrand: f(x,y) = x * y
    let f = |x: &Array1<f64>| x[0] * x[1];

    // Integrate over [0,1] × [0,1]
    let bounds = vec![
        (Bound::Finite(0.0), Bound::Finite(1.0)),
        (Bound::Finite(0.0), Bound::Finite(1.0)),
    ];

    match cubature(f, &bounds, None) {
        Ok(result) => {
            println!("Integration result: {}", result.value);
            println!("Expected result: 0.25");
            println!("Error: {}", result.abs_error);
            println!("Function evaluations: {}", result.n_evals);
            println!("Converged: {}", result.converged);
        }
        Err(e) => {
            println!("Error: {e:?}");
        }
    }

    println!("Test completed.");
}
