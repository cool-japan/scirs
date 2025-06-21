//! Numerical Stability Monitoring Demonstration
//!
//! This example demonstrates the new numerical stability monitoring features
//! in the scirs2-interpolate library, showing how to detect and handle
//! ill-conditioned matrices during interpolation.

use ndarray::{Array1, Array2};
use scirs2_interpolate::{
    advanced::rbf::{RBFInterpolator, RBFKernel},
    numerical_stability::{assess_matrix_condition, machine_epsilon, StabilityLevel},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Numerical Stability Monitoring Demo ===\n");

    // 1. Demonstrate machine epsilon detection
    println!("1. Machine Epsilon:");
    let eps_f32: f32 = machine_epsilon();
    let eps_f64: f64 = machine_epsilon();
    println!("   f32 machine epsilon: {:.2e}", eps_f32);
    println!("   f64 machine epsilon: {:.2e}\n", eps_f64);

    // 2. Demonstrate matrix condition assessment
    println!("2. Matrix Condition Assessment:");

    // Well-conditioned matrix (identity)
    let well_conditioned = Array2::<f64>::eye(3);
    let report = assess_matrix_condition(&well_conditioned.view())?;
    println!("   Identity matrix:");
    println!("     Condition number: {:.2e}", report.condition_number);
    println!("     Stability level: {:?}", report.stability_level);
    println!("     Well-conditioned: {}", report.is_well_conditioned);

    // Ill-conditioned matrix
    let mut ill_conditioned = Array2::eye(3);
    ill_conditioned[[2, 2]] = 1e-15; // Make it nearly singular
    let report = assess_matrix_condition(&ill_conditioned.view())?;
    println!("   Nearly singular matrix:");
    println!("     Condition number: {:.2e}", report.condition_number);
    println!("     Stability level: {:?}", report.stability_level);
    println!("     Well-conditioned: {}", report.is_well_conditioned);
    if let Some(reg) = report.recommended_regularization {
        println!("     Recommended regularization: {:.2e}", reg);
    }
    println!();

    // 3. Demonstrate RBF interpolation with stability monitoring
    println!("3. RBF Interpolation with Stability Monitoring:");

    // Create well-spaced data points
    let points_good = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])?;
    let values_good = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);

    println!("   Creating RBF interpolator with well-conditioned data...");
    let rbf_good = RBFInterpolator::new(
        &points_good.view(),
        &values_good.view(),
        RBFKernel::Gaussian,
        1.0,
    )?;

    if let Some(report) = rbf_good.condition_report() {
        println!(
            "     RBF matrix condition number: {:.2e}",
            report.condition_number
        );
        println!("     Stability level: {:?}", report.stability_level);
    }

    // Create poorly-conditioned data (nearly collinear points)
    let points_bad = Array2::from_shape_vec(
        (4, 2),
        vec![
            0.0, 0.0, 1e-10, 1e-10, // Very close to first point
            2e-10, 2e-10, // Very close to first point
            3e-10, 3e-10, // Very close to first point
        ],
    )?;
    let values_bad = Array1::from_vec(vec![0.0, 1e-10, 2e-10, 3e-10]);

    println!("\n   Creating RBF interpolator with poorly-conditioned data...");
    let rbf_bad = RBFInterpolator::new(
        &points_bad.view(),
        &values_bad.view(),
        RBFKernel::Gaussian,
        1.0,
    );

    match rbf_bad {
        Ok(interpolator) => {
            if let Some(report) = interpolator.condition_report() {
                println!(
                    "     RBF matrix condition number: {:.2e}",
                    report.condition_number
                );
                println!("     Stability level: {:?}", report.stability_level);
                println!("     Well-conditioned: {}", report.is_well_conditioned);
            }
        }
        Err(e) => {
            println!("     Failed to create interpolator: {}", e);
        }
    }

    // 4. Demonstrate different stability levels
    println!("\n4. Stability Level Classification:");
    let condition_numbers = vec![1e10, 1e13, 1e15, 1e17];
    for &cond in &condition_numbers {
        let level = if cond < 1e12 {
            StabilityLevel::Excellent
        } else if cond < 1e14 {
            StabilityLevel::Good
        } else if cond < 1e16 {
            StabilityLevel::Marginal
        } else {
            StabilityLevel::Poor
        };

        println!("   Condition number {:.0e}: {:?}", cond, level);
    }

    // 5. Test interpolation with stability monitoring
    println!("\n5. Interpolation with Well-Conditioned RBF:");
    let query_points = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.25, 0.75])?;
    let results = rbf_good.interpolate(&query_points.view())?;
    println!("   Query points: [[0.5, 0.5], [0.25, 0.75]]");
    println!(
        "   Interpolated values: [{:.4}, {:.4}]",
        results[0], results[1]
    );

    if let Some(report) = rbf_good.condition_report() {
        println!(
            "   Matrix was well-conditioned: {}",
            report.is_well_conditioned
        );
    }

    println!("\n=== Demo Complete ===");
    println!("\nKey Features Demonstrated:");
    println!("• Machine epsilon detection for different floating point types");
    println!("• Matrix condition number assessment with stability classification");
    println!("• Automatic detection of ill-conditioned matrices in RBF interpolation");
    println!("• Stability-aware fallback strategies when numerical issues are detected");
    println!("• Condition reporting for post-analysis of interpolation quality");

    Ok(())
}
