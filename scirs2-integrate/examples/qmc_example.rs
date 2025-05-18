use ndarray::{Array1, ArrayView1};
use scirs2_integrate::qmc::{qmc_quad, Halton, Sobol};
use std::f64::consts::PI;

fn main() {
    println!("Quasi-Monte Carlo Integration Examples");
    println!("======================================");

    // Example 1: Simple 1D integral
    println!("\nExample 1: Integral of x^2 from 0 to 1");
    println!("Exact value: 1/3 = 0.3333...");

    let f1 = |x: ArrayView1<f64>| x[0].powi(2);
    let a1 = Array1::from_vec(vec![0.0]);
    let b1 = Array1::from_vec(vec![1.0]);

    let result1 = qmc_quad(f1, &a1, &b1, Some(8), Some(1000), None, false).unwrap();

    println!("QMC result: {:.10}", result1.integral);
    println!("Error estimate: {:.10}", result1.standard_error);
    println!("Actual error: {:.10}", (result1.integral - 1.0 / 3.0).abs());

    // Example 2: 2D Integral with Halton sequence
    println!("\nExample 2: Integral of sin(x)*cos(y) from 0 to π × 0 to π");
    println!("Exact value: 0.0");

    let f2 = |x: ArrayView1<f64>| (x[0].sin() * x[1].cos());
    let a2 = Array1::from_vec(vec![0.0, 0.0]);
    let b2 = Array1::from_vec(vec![PI, PI]);

    let qrng = Halton::new(2, Some(42));
    let result2 = qmc_quad(
        f2,
        &a2,
        &b2,
        Some(16),
        Some(1024),
        Some(Box::new(qrng)),
        false,
    )
    .unwrap();

    println!("QMC result: {:.10}", result2.integral);
    println!("Error estimate: {:.10}", result2.standard_error);
    println!("Actual error: {:.10}", result2.integral.abs());

    // Example 3: Higher dimensional integral with Sobol sequence
    println!("\nExample 3: 3D integral of x*y*z from 0 to 1 × 0 to 1 × 0 to 1");
    println!("Exact value: 1/8 = 0.125");

    let f3 = |x: ArrayView1<f64>| x[0] * x[1] * x[2];
    let a3 = Array1::from_vec(vec![0.0, 0.0, 0.0]);
    let b3 = Array1::from_vec(vec![1.0, 1.0, 1.0]);

    let qrng = Sobol::new(3, Some(42));
    let result3 = qmc_quad(
        f3,
        &a3,
        &b3,
        Some(16),
        Some(2048),
        Some(Box::new(qrng)),
        false,
    )
    .unwrap();

    println!("QMC result: {:.10}", result3.integral);
    println!("Error estimate: {:.10}", result3.standard_error);
    println!("Actual error: {:.10}", (result3.integral - 1.0 / 8.0).abs());

    // Example 4: Semi-infinite integral
    println!("\nExample 4: Integral of e^(-x^2) from -∞ to ∞");
    println!("Exact value: sqrt(π) ≈ 1.7724538509...");

    let f4 = |x: ArrayView1<f64>| (-x[0].powi(2)).exp();
    // Approximate infinity with large but finite values
    let a4 = Array1::from_vec(vec![-10.0]);
    let b4 = Array1::from_vec(vec![10.0]);

    let result4 = qmc_quad(f4, &a4, &b4, Some(16), Some(2048), None, false).unwrap();

    println!("QMC result: {:.10}", result4.integral);
    println!("Error estimate: {:.10}", result4.standard_error);
    println!("Actual error: {:.10}", (result4.integral - PI.sqrt()).abs());

    // Example 5: Using log integration for numerical stability
    println!("\nExample 5: Integral with log integration for numeric stability");
    println!("Integral of e^(-1000*x^2) from -∞ to ∞");
    println!("Exact value: sqrt(π/1000) ≈ 0.056...");

    // Without log integration, this would underflow
    let _f5_nolog = |x: ArrayView1<f64>| (-1000.0 * x[0].powi(2)).exp();
    let f5_log = |x: ArrayView1<f64>| -1000.0 * x[0].powi(2);

    let a5 = Array1::from_vec(vec![-0.5]);
    let b5 = Array1::from_vec(vec![0.5]);

    let result5_log = qmc_quad(f5_log, &a5, &b5, Some(16), Some(2048), None, true).unwrap();

    println!("QMC log result: {:.10}", result5_log.integral);
    println!("QMC result: {:.10}", result5_log.integral.exp());
    println!("Error estimate: {:.10}", result5_log.standard_error.exp());
    println!("Exact result: {:.10}", (PI / 1000.0).sqrt());
    println!(
        "Actual error: {:.10}",
        (result5_log.integral.exp() - (PI / 1000.0).sqrt()).abs()
    );
}
