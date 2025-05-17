//! Example demonstrating the use of cubature for multidimensional integration
//!
//! This example shows various use cases for multidimensional integration:
//! - Regular bounded integration
//! - Semi-infinite domains
//! - Fully infinite domains
//! - Higher-dimensional integration
//! - Using nquad as a simpler interface

use ndarray::Array1;
use scirs2_integrate::cubature::{cubature, nquad, Bound, CubatureOptions};
use std::f64::consts::PI;

fn main() {
    println!("Cubature Multidimensional Integration Examples");
    println!("==============================================\n");

    // Example 1: Simple 2D integral with finite bounds
    println!("Example 1: Integrating f(x,y) = x*y over [0,1]×[0,1]");
    let f1 = |x: &Array1<f64>| x[0] * x[1];

    let bounds1 = vec![
        (Bound::Finite(0.0), Bound::Finite(1.0)),
        (Bound::Finite(0.0), Bound::Finite(1.0)),
    ];

    let result1 = cubature(f1, &bounds1, None).unwrap();
    println!("   Result: {:.10}", result1.value);
    println!("   Error estimate: {:.10e}", result1.abs_error);
    println!("   Function evaluations: {}", result1.n_evals);
    println!("   Converged: {}", result1.converged);
    println!("   Expected value: 0.25\n");

    // Example 2: 3D integral with finite bounds
    println!("Example 2: Integrating f(x,y,z) = x*y*z over [0,1]×[0,1]×[0,1]");
    let f2 = |x: &Array1<f64>| x[0] * x[1] * x[2];

    let bounds2 = vec![
        (Bound::Finite(0.0), Bound::Finite(1.0)),
        (Bound::Finite(0.0), Bound::Finite(1.0)),
        (Bound::Finite(0.0), Bound::Finite(1.0)),
    ];

    let result2 = cubature(f2, &bounds2, None).unwrap();
    println!("   Result: {:.10}", result2.value);
    println!("   Error estimate: {:.10e}", result2.abs_error);
    println!("   Function evaluations: {}", result2.n_evals);
    println!("   Converged: {}", result2.converged);
    println!("   Expected value: 0.125\n");

    // Example 3: Semi-infinite domain
    println!("Example 3: Integrating f(x) = exp(-x) over [0,∞)");
    let f3 = |x: &Array1<f64>| (-x[0]).exp();

    let bounds3 = vec![(Bound::Finite(0.0), Bound::PosInf)];

    let result3 = cubature(f3, &bounds3, None).unwrap();
    println!("   Result: {:.10}", result3.value);
    println!("   Error estimate: {:.10e}", result3.abs_error);
    println!("   Function evaluations: {}", result3.n_evals);
    println!("   Converged: {}", result3.converged);
    println!("   Expected value: 1.0");
    println!("   Note: Current implementation has limitations with infinite domains.\n");

    // Example 4: Fully infinite domain - Gaussian integral
    println!("Example 4: Integrating f(x) = exp(-x²) over (-∞,∞)");
    let f4 = |x: &Array1<f64>| (-x[0] * x[0]).exp();

    let bounds4 = vec![(Bound::NegInf, Bound::PosInf)];

    let result4 = cubature(f4, &bounds4, None).unwrap();
    println!("   Result: {:.10}", result4.value);
    println!("   Error estimate: {:.10e}", result4.abs_error);
    println!("   Function evaluations: {}", result4.n_evals);
    println!("   Converged: {}", result4.converged);
    println!("   Expected value: {:.10}", f64::sqrt(PI));
    println!("   Note: Current implementation has limitations with infinite domains.\n");

    // Example 5: 2D Gaussian integral
    println!("Example 5: Integrating f(x,y) = exp(-(x² + y²)) over R²");
    let f5 = |x: &Array1<f64>| (-x[0] * x[0] - x[1] * x[1]).exp();

    let bounds5 = vec![
        (Bound::NegInf, Bound::PosInf),
        (Bound::NegInf, Bound::PosInf),
    ];

    let result5 = cubature(f5, &bounds5, None).unwrap();
    println!("   Result: {:.10}", result5.value);
    println!("   Error estimate: {:.10e}", result5.abs_error);
    println!("   Function evaluations: {}", result5.n_evals);
    println!("   Converged: {}", result5.converged);
    println!("   Expected value: {:.10}", PI);
    println!("   Note: Current implementation has limitations with infinite domains.\n");

    // Example 6: Using nquad for simpler interface
    println!("Example 6: Using nquad to integrate f(x,y) = sin(x+y) over [0,π]×[0,π]");
    let f6 = |args: &[f64]| (args[0] + args[1]).sin();
    let ranges = vec![(0.0, PI), (0.0, PI)];

    let result6 = nquad(f6, &ranges, None).unwrap();
    println!("   Result: {:.10}", result6.value);
    println!("   Error estimate: {:.10e}", result6.abs_error);
    println!("   Function evaluations: {}", result6.n_evals);
    println!("   Converged: {}", result6.converged);
    println!("   Expected value: 4.0\n");

    // Example 7: Integration with custom options
    println!("Example 7: Integration with custom options");
    let f7 = |x: &Array1<f64>| x[0].cos() * x[1].sin();

    let bounds7 = vec![
        (Bound::Finite(0.0), Bound::Finite(PI)),
        (Bound::Finite(0.0), Bound::Finite(PI)),
    ];

    let options = CubatureOptions {
        abs_tol: 1e-10,
        rel_tol: 1e-10,
        max_evals: 10000,
        max_recursion_depth: 20,
        vectorized: false,
        log: false,
    };

    let result7 = cubature(f7, &bounds7, Some(options)).unwrap();
    println!("   Result: {:.10}", result7.value);
    println!("   Error estimate: {:.10e}", result7.abs_error);
    println!("   Function evaluations: {}", result7.n_evals);
    println!("   Converged: {}", result7.converged);
    println!("   Expected value: 2.0\n");
}
