use scirs2_signal::spline::{
    bspline_basis, bspline_coefficients, bspline_derivative, bspline_evaluate, bspline_filter,
    bspline_smooth, SplineOrder,
};
use std::error::Error;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn Error>> {
    println!("B-Spline Filtering and Interpolation Example");
    println!("--------------------------------------------");

    // Generate test signal
    println!("\nGenerating noisy test signal...");
    let n_samples = 100;
    let t: Vec<f64> = (0..n_samples).map(|i| i as f64 * 0.1).collect();

    // Signal: sine wave with noise
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| {
            let sine = (2.0 * PI * 0.5 * ti).sin(); // 0.5 Hz sine wave
            let noise = 0.2 * ((ti * 11.0).sin() + (ti * 29.7).cos()); // Noise
            sine + noise
        })
        .collect();

    println!("  Signal length: {} samples", signal.len());
    println!("  Sampling rate: 10 Hz");
    println!("  Signal frequency: 0.5 Hz");
    println!("  Added noise level: 0.2");

    // Demonstrate B-spline basis functions
    println!("\nComputing B-spline basis functions...");
    let basis_x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();

    for order in [
        SplineOrder::Constant,
        SplineOrder::Linear,
        SplineOrder::Quadratic,
        SplineOrder::Cubic,
    ] {
        let basis = bspline_basis(&basis_x, order)?;
        let max_val = basis.iter().fold(0.0_f64, |a, &b| a.max(b));

        println!(
            "  {} B-spline basis:",
            match order {
                SplineOrder::Constant => "Constant (order 0)",
                SplineOrder::Linear => "Linear (order 1)",
                SplineOrder::Quadratic => "Quadratic (order 2)",
                SplineOrder::Cubic => "Cubic (order 3)",
                SplineOrder::Quartic => "Quartic (order 4)",
                SplineOrder::Quintic => "Quintic (order 5)",
            }
        );
        println!("    Maximum value: {:.4}", max_val);
        println!("    Support width: {}", order.as_int() + 1);
    }

    // B-spline filtering
    println!("\nApplying B-spline filters to the signal...");

    // Filter using different spline orders
    for order in [
        SplineOrder::Linear,
        SplineOrder::Quadratic,
        SplineOrder::Cubic,
        SplineOrder::Quartic,
    ] {
        let filtered = bspline_filter(&signal, order)?;

        // Calculate mean squared error
        let mse: f64 = signal
            .iter()
            .zip(filtered.iter())
            .map(|(&orig, &filt)| (orig - filt).powi(2))
            .sum::<f64>()
            / signal.len() as f64;

        println!(
            "  {} filter:",
            match order {
                SplineOrder::Constant => "Constant (order 0)",
                SplineOrder::Linear => "Linear (order 1)",
                SplineOrder::Quadratic => "Quadratic (order 2)",
                SplineOrder::Cubic => "Cubic (order 3)",
                SplineOrder::Quartic => "Quartic (order 4)",
                SplineOrder::Quintic => "Quintic (order 5)",
            }
        );
        println!("    Mean squared error: {:.6}", mse);
    }

    // B-spline smoothing
    println!("\nApplying B-spline smoothing with different lambda values...");

    // Smooth using different smoothing parameters
    for lambda in [0.1, 1.0, 10.0, 100.0] {
        let smoothed = bspline_smooth(&signal, SplineOrder::Cubic, lambda)?;

        // Calculate mean squared error
        let mse: f64 = signal
            .iter()
            .zip(smoothed.iter())
            .map(|(&orig, &smooth)| (orig - smooth).powi(2))
            .sum::<f64>()
            / signal.len() as f64;

        println!("  Lambda = {:.1}:", lambda);
        println!("    Mean squared error: {:.6}", mse);
    }

    // B-spline interpolation
    println!("\nDemonstrating B-spline interpolation...");

    // Create a sparse signal (use every 5th point)
    let sparse_indices: Vec<usize> = (0..20).map(|i| i * 5).collect();
    let sparse_x: Vec<f64> = sparse_indices.iter().map(|&i| t[i]).collect();
    let sparse_y: Vec<f64> = sparse_indices.iter().map(|&i| signal[i]).collect();

    println!(
        "  Sparse signal has {} points (every 5th point)",
        sparse_y.len()
    );

    // Compute B-spline coefficients
    let coeffs = bspline_coefficients(&sparse_y, SplineOrder::Cubic)?;

    // Interpolate at original points
    let interpolated = bspline_evaluate(&coeffs, &sparse_x, SplineOrder::Cubic)?;

    // Validate interpolation at control points
    let max_error = sparse_y
        .iter()
        .zip(interpolated.iter())
        .map(|(&orig, &interp)| (orig - interp).abs())
        .fold(0.0_f64, |a, b| a.max(b));

    println!("  Interpolation error at control points: {:.10}", max_error);

    // Now interpolate between the control points
    let interp_x: Vec<f64> = t.clone();
    let interp_y = bspline_evaluate(&coeffs, &interp_x, SplineOrder::Cubic)?;

    println!("  Interpolated to {} points", interp_y.len());

    // Compute derivatives
    println!("\nComputing derivatives of the B-spline curve...");

    // First derivative
    let first_deriv = bspline_derivative(&coeffs, &interp_x, SplineOrder::Cubic, 1)?;

    // Find the maximum derivative (should correspond to steepest parts of sine)
    let max_deriv = first_deriv.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));

    // Second derivative
    let second_deriv = bspline_derivative(&coeffs, &interp_x, SplineOrder::Cubic, 2)?;

    println!("  First derivative:");
    println!("    Maximum absolute value: {:.4}", max_deriv);
    println!("  Second derivative:");
    println!(
        "    Maximum absolute value: {:.4}",
        second_deriv.iter().fold(0.0_f64, |a, &b| a.max(b.abs()))
    );

    println!("\nB-spline processing complete!");

    Ok(())
}
