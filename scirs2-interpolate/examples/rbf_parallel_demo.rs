//! Demonstration of parallel RBF interpolation
//!
//! This example shows how to use the new `new_parallel` method for RBF interpolation
//! and compares performance with the serial version.

use ndarray::{Array1, Array2};
use scirs2_interpolate::advanced::rbf::{RBFInterpolator, RBFKernel};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RBF Parallel Interpolation Demo");
    println!("================================");

    // Create a larger dataset for demonstrating parallel benefits
    let n_points = 100;
    let n_dims = 2;

    // Generate sample points in a grid pattern
    let mut points_data = Vec::with_capacity(n_points * n_dims);
    let mut values_data = Vec::with_capacity(n_points);

    let grid_size = (n_points as f64).sqrt() as usize;
    for i in 0..grid_size {
        for j in 0..grid_size {
            if points_data.len() >= n_points * n_dims {
                break;
            }
            let x = i as f64 / (grid_size - 1) as f64;
            let y = j as f64 / (grid_size - 1) as f64;
            points_data.push(x);
            points_data.push(y);

            // Test function: z = sin(2πx) * cos(2πy) + 0.5 * (x² + y²)
            let z = (2.0 * std::f64::consts::PI * x).sin() * (2.0 * std::f64::consts::PI * y).cos()
                + 0.5 * (x * x + y * y);
            values_data.push(z);
        }
    }

    // Truncate to exactly n_points
    points_data.truncate(n_points * n_dims);
    values_data.truncate(n_points);

    let points = Array2::from_shape_vec((n_points, n_dims), points_data)?;
    let values = Array1::from(values_data);

    println!("Dataset: {} points in {}D space", n_points, n_dims);
    println!("Test function: z = sin(2πx) * cos(2πy) + 0.5 * (x² + y²)");
    println!();

    // Test different kernels
    let kernels = [
        ("Gaussian", RBFKernel::Gaussian),
        ("Multiquadric", RBFKernel::Multiquadric),
        ("InverseMultiquadric", RBFKernel::InverseMultiquadric),
    ];

    for (kernel_name, kernel) in kernels.iter() {
        println!("Testing {} kernel:", kernel_name);

        // Serial version
        let start = Instant::now();
        let interp_serial = RBFInterpolator::new(&points.view(), &values.view(), *kernel, 1.0)?;
        let serial_time = start.elapsed();

        // Parallel version with 4 workers
        let start = Instant::now();
        let interp_parallel = RBFInterpolator::new_parallel(
            &points.view(),
            &values.view(),
            *kernel,
            1.0,
            4, // Use 4 workers
        )?;
        let parallel_time = start.elapsed();

        // Parallel version with automatic worker detection
        let start = Instant::now();
        let interp_auto = RBFInterpolator::new_parallel(
            &points.view(),
            &values.view(),
            *kernel,
            1.0,
            0, // Use automatic worker detection
        )?;
        let auto_time = start.elapsed();

        println!("  Serial construction:    {:?}", serial_time);
        println!("  Parallel (4 workers):   {:?}", parallel_time);
        println!("  Parallel (auto):        {:?}", auto_time);

        // Calculate speedup
        let speedup_4 = serial_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
        let speedup_auto = serial_time.as_nanos() as f64 / auto_time.as_nanos() as f64;
        println!("  Speedup (4 workers):    {:.2}x", speedup_4);
        println!("  Speedup (auto):         {:.2}x", speedup_auto);

        // Test that all methods give the same results
        let test_points = Array2::from_shape_vec((3, 2), vec![0.25, 0.25, 0.5, 0.75, 0.8, 0.3])?;

        let result_serial = interp_serial.interpolate(&test_points.view())?;
        let result_parallel = interp_parallel.interpolate(&test_points.view())?;
        let result_auto = interp_auto.interpolate(&test_points.view())?;

        println!("  Results consistency check:");
        for i in 0..test_points.nrows() {
            let diff_parallel = (result_serial[i] - result_parallel[i]).abs();
            let diff_auto = (result_serial[i] - result_auto[i]).abs();
            println!(
                "    Point {}: serial={:.6}, parallel={:.6}, auto={:.6}",
                i + 1,
                result_serial[i],
                result_parallel[i],
                result_auto[i]
            );
            println!(
                "              Diff (parallel): {:.2e}, Diff (auto): {:.2e}",
                diff_parallel, diff_auto
            );

            // Verify results are very close (within numerical precision)
            assert!(
                diff_parallel < 1e-10,
                "Parallel result differs too much from serial"
            );
            assert!(
                diff_auto < 1e-10,
                "Auto result differs too much from serial"
            );
        }

        println!("  ✓ All methods produce consistent results");
        println!();
    }

    println!("Recommendations:");
    println!("- Use parallel construction for datasets with > 100 points");
    println!("- For smaller datasets, serial construction may be faster due to overhead");
    println!("- Use workers=0 for automatic CPU detection in most cases");
    println!("- Adjust worker count based on your system and dataset size");

    Ok(())
}
