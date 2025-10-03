//! Matrix calculus example
//!
//! This example demonstrates the use of matrix calculus operations such as
//! gradient, Jacobian, and Hessian computation.

use scirs2_core::ndarray::{array, Array1, ArrayView1};
use scirs2_linalg::error::LinalgResult;
use scirs2_linalg::matrix_calculus::{gradient, hessian, jacobian};

#[allow(dead_code)]
fn main() -> LinalgResult<()> {
    println!("Matrix Calculus Examples");
    println!("=======================\n");

    // Example 1: Gradient of a simple function
    println!("Example 1: Gradient of f(x,y) = x^2 + y^2");

    // Define function f(x) = x^2 + y^2
    let f = |x: &ArrayView1<f64>| -> LinalgResult<f64> {
        if x.len() != 2 {
            panic!("Expected 2D input");
        }
        Ok(x[0] * x[0] + x[1] * x[1])
    };

    // Compute gradient at (1, 2)
    let x = array![1.0, 2.0];
    let grad = gradient(f, &x.view(), None)?;

    println!("Gradient at (1, 2): {:?}", grad);
    println!("Expected: [2, 4]\n");

    // Example 2: Jacobian of a vector-valued function
    println!("Example 2: Jacobian of f(x,y) = [x^2 + y, x*y]");

    // Define vector function f(x,y) = [x^2 + y, x*y]
    let g = |x: &ArrayView1<f64>| -> LinalgResult<Array1<f64>> {
        if x.len() != 2 {
            panic!("Expected 2D input");
        }
        let mut result = Array1::zeros(2);
        result[0] = x[0] * x[0] + x[1];
        result[1] = x[0] * x[1];
        Ok(result)
    };

    // Compute Jacobian at (1, 2)
    let jac = jacobian(g, &x.view(), None)?;

    println!("Jacobian at (1, 2):");
    println!("{:?}", jac);
    println!("Expected: [[2, 1], [2, 1]]\n");

    // Example 3: Hessian of a scalar function
    println!("Example 3: Hessian of f(x,y) = x^2 + x*y + y^2");

    // Define function f(x,y) = x^2 + x*y + y^2
    let h = |x: &ArrayView1<f64>| -> LinalgResult<f64> {
        if x.len() != 2 {
            panic!("Expected 2D input");
        }
        Ok(x[0] * x[0] + x[0] * x[1] + x[1] * x[1])
    };

    // Compute Hessian at (1, 2)
    let hess = hessian(h, &x.view(), None)?;

    println!("Hessian at (1, 2):");
    println!("{:?}", hess);
    println!("Expected: [[2, 1], [1, 2]]\n");

    Ok(())
}
