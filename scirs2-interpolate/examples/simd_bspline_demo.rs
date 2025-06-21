//! SIMD-Optimized B-spline Evaluation Demonstration
//!
//! This example demonstrates the vectorized de Boor algorithm implementation
//! that can evaluate B-splines at multiple points simultaneously using SIMD
//! instructions for significant performance improvements.

#[cfg(feature = "simd")]
use ndarray::Array1;
#[cfg(feature = "simd")]
use scirs2_interpolate::{
    bspline::{BSpline, ExtrapolateMode},
    simd_bspline::SimdBSplineEvaluator,
    simd_optimized::{get_simd_config, is_simd_available},
};
#[cfg(feature = "simd")]
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SIMD B-spline Evaluation Demo ===\n");

    #[cfg(feature = "simd")]
    {
        // 1. Display SIMD capabilities
        println!("1. SIMD Capabilities:");
        let simd_config = get_simd_config();
        println!("   SIMD Available: {}", simd_config.simd_available);
        println!("   Instruction Set: {}", simd_config.instruction_set);
        println!("   f64 Vector Width: {}", simd_config.f64_width);
        println!("   f32 Vector Width: {}", simd_config.f32_width);
        println!();

        // 2. Create test B-splines
        println!("2. Creating Test B-splines:");

        // Smooth polynomial-like function
        let knots = Array1::linspace(0.0, 10.0, 25);
        let coeffs = Array1::from_iter((0..22).map(|i| {
            let x = i as f64 / 21.0;
            (2.0 * std::f64::consts::PI * x * 3.0).sin()
                + 0.5 * (std::f64::consts::PI * x * 7.0).cos()
        }));

        let smooth_spline = BSpline::new(
            &knots.view(),
            &coeffs.view(),
            3, // Cubic B-spline
            ExtrapolateMode::Extrapolate,
        )?;

        // Oscillatory function
        let knots_osc = Array1::linspace(0.0, 4.0 * std::f64::consts::PI, 30);
        let coeffs_osc = Array1::from_iter((0..27).map(|i| {
            let x = i as f64 / 26.0 * 4.0 * std::f64::consts::PI;
            x.sin() * (-x / 8.0).exp()
        }));

        let oscillatory_spline = BSpline::new(
            &knots_osc.view(),
            &coeffs_osc.view(),
            3,
            ExtrapolateMode::Extrapolate,
        )?;

        println!(
            "   Created smooth B-spline: {} knots, degree {}",
            smooth_spline.knot_vector().len(),
            smooth_spline.degree()
        );
        println!(
            "   Created oscillatory B-spline: {} knots, degree {}",
            oscillatory_spline.knot_vector().len(),
            oscillatory_spline.degree()
        );
        println!();

        // 3. Create SIMD evaluators
        println!("3. Creating SIMD Evaluators:");
        let simd_evaluator_smooth = SimdBSplineEvaluator::new(smooth_spline.clone());
        let simd_evaluator_osc = SimdBSplineEvaluator::new(oscillatory_spline.clone());

        println!(
            "   Smooth spline SIMD available: {}",
            simd_evaluator_smooth.simd_available()
        );
        println!(
            "   Oscillatory spline SIMD available: {}",
            simd_evaluator_osc.simd_available()
        );
        println!();

        // 4. Performance comparison
        println!("4. Performance Comparison:");

        // Test different batch sizes
        let batch_sizes = vec![10, 100, 1000, 10000];

        for &batch_size in &batch_sizes {
            println!("\n   Batch Size: {} points", batch_size);

            // Generate evaluation points
            let eval_points = Array1::linspace(1.0, 9.0, batch_size);

            // SIMD evaluation
            let start = Instant::now();
            let simd_results = simd_evaluator_smooth.evaluate_batch(&eval_points.view())?;
            let simd_duration = start.elapsed();

            // Scalar evaluation for comparison
            let start = Instant::now();
            let mut scalar_results = Array1::zeros(batch_size);
            for (i, &point) in eval_points.iter().enumerate() {
                scalar_results[i] = smooth_spline.evaluate(point)?;
            }
            let scalar_duration = start.elapsed();

            // Verify results are identical
            let max_diff = simd_results
                .iter()
                .zip(scalar_results.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0, f64::max);

            let speedup = scalar_duration.as_nanos() as f64 / simd_duration.as_nanos() as f64;

            println!(
                "     SIMD time: {:8.2} µs",
                simd_duration.as_nanos() as f64 / 1000.0
            );
            println!(
                "     Scalar time: {:6.2} µs",
                scalar_duration.as_nanos() as f64 / 1000.0
            );
            println!("     Speedup: {:.2}x", speedup);
            println!("     Max difference: {:.2e}", max_diff);

            if max_diff > 1e-12 {
                println!("     WARNING: Results differ significantly!");
            }
        }

        // 5. Accuracy verification
        println!("\n5. Accuracy Verification:");

        let test_points = Array1::from_vec(vec![0.5, 1.0, 2.5, 3.14159, 5.0, 7.5, 8.9, 9.5]);

        println!("   Testing {} specific points:", test_points.len());

        // Test smooth spline
        let simd_smooth = simd_evaluator_smooth.evaluate_batch(&test_points.view())?;
        println!("   Smooth spline results:");
        for (_i, (&point, &result)) in test_points.iter().zip(simd_smooth.iter()).enumerate() {
            println!("     f({:.3}) = {:8.5}", point, result);

            // Verify against scalar evaluation
            let scalar_result = smooth_spline.evaluate(point)?;
            let diff = (result - scalar_result).abs();
            if diff > 1e-12 {
                println!(
                    "       ERROR: Scalar gives {:.5} (diff: {:.2e})",
                    scalar_result, diff
                );
            }
        }

        // Test oscillatory spline
        let osc_test_points = Array1::linspace(0.0, 4.0 * std::f64::consts::PI, 8);
        let simd_osc = simd_evaluator_osc.evaluate_batch(&osc_test_points.view())?;

        println!("\n   Oscillatory spline results:");
        for (&point, &result) in osc_test_points.iter().zip(simd_osc.iter()) {
            println!("     f({:.3}) = {:8.5}", point, result);
        }

        // 6. Demonstrate batch vs individual evaluation
        println!("\n6. Batch vs Individual Evaluation:");

        let large_batch = Array1::linspace(0.1, 9.9, 5000);

        // Batch evaluation
        let start = Instant::now();
        let _batch_results = simd_evaluator_smooth.evaluate_batch(&large_batch.view())?;
        let batch_time = start.elapsed();

        // Individual evaluations
        let start = Instant::now();
        for &point in large_batch.iter().take(100) {
            // Only sample first 100 for timing
            let _ = simd_evaluator_smooth.evaluate(point)?;
        }
        let individual_time_sample = start.elapsed();
        let estimated_individual_time = individual_time_sample * 50; // Scale to full batch

        println!("   5000-point evaluation:");
        println!(
            "     Batch: {:.2} ms",
            batch_time.as_nanos() as f64 / 1_000_000.0
        );
        println!(
            "     Individual (estimated): {:.2} ms",
            estimated_individual_time.as_nanos() as f64 / 1_000_000.0
        );
        println!(
            "     Batch speedup: {:.1}x",
            estimated_individual_time.as_nanos() as f64 / batch_time.as_nanos() as f64
        );

        // 7. Memory efficiency demonstration
        println!("\n7. Memory Efficiency:");

        // Show that SIMD evaluation processes data in chunks
        let chunk_sizes = vec![3, 4, 5, 7, 8, 9, 15, 16, 17];

        for &size in &chunk_sizes {
            let points = Array1::linspace(2.0, 8.0, size);
            let results = simd_evaluator_smooth.evaluate_batch(&points.view())?;

            println!(
                "   {} points -> {} results (SIMD chunks: {})",
                size,
                results.len(),
                size.div_ceil(4)
            );
        }

        // 8. Edge cases and robustness
        println!("\n8. Edge Cases and Robustness:");

        // Test boundary values
        let boundary_points = Array1::from_vec(vec![
            knots[0],                                  // Left boundary
            knots[knots.len() - 1],                    // Right boundary
            (knots[0] + knots[knots.len() - 1]) / 2.0, // Middle
        ]);

        let boundary_results = simd_evaluator_smooth.evaluate_batch(&boundary_points.view())?;

        println!("   Boundary value tests:");
        for (i, (&point, &result)) in boundary_points
            .iter()
            .zip(boundary_results.iter())
            .enumerate()
        {
            println!("     Boundary {}: f({:.3}) = {:8.5}", i + 1, point, result);
            assert!(result.is_finite(), "Result should be finite");
        }

        // Test very small batch (should fall back to scalar)
        let tiny_batch = Array1::from_vec(vec![5.0]);
        let tiny_result = simd_evaluator_smooth.evaluate_batch(&tiny_batch.view())?;
        println!(
            "   Single point batch: f({}) = {:.5}",
            tiny_batch[0], tiny_result[0]
        );

        println!("\n=== Demo Complete ===");

        if is_simd_available() {
            println!("\n✅ SIMD optimizations are active on this platform!");
            println!("   Expected 2-4x speedup for large batches of f64 B-spline evaluations.");
        } else {
            println!("\n⚠️  SIMD optimizations not available.");
            println!("   Using optimized scalar fallback implementation.");
            println!(
                "   Compile with --features simd and run on x86_64/aarch64 for SIMD acceleration."
            );
        }

        println!("\nKey Features Demonstrated:");
        println!("• Vectorized de Boor algorithm for simultaneous evaluation of 4 points");
        println!("• Automatic SIMD detection and graceful fallback to scalar code");
        println!("• Same-span optimization for nearby evaluation points");
        println!("• Chunk-based processing for arbitrary batch sizes");
        println!("• Memory-efficient SIMD register utilization");
        println!("• Numerical accuracy identical to scalar implementation");

        Ok(())
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("⚠️  SIMD features are not enabled for this build.");
        println!("To see SIMD B-spline evaluation in action:");
        println!("  cargo run --example simd_bspline_demo --features simd");
        println!("\nThis example requires the 'simd' feature flag to demonstrate:");
        println!("• Vectorized de Boor algorithm");
        println!("• SIMD-accelerated B-spline evaluation");
        println!("• Performance comparisons with scalar implementation");
        Ok(())
    }
}
