use ndarray::ArrayView1;
use scirs2_integrate::romberg::{multi_romberg_with_details, RombergOptions};
use std::f64::consts::PI;
use std::time::Instant;

/// A helper function to time and report the result of an integration method
fn time_integration<F, R>(name: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    println!("{}: {:?}", name, elapsed);
    result
}

fn main() {
    println!("Multidimensional Integration Examples\n");

    // Example 1: 2D integration with different methods
    println!("Example 1: 2D Integration - f(x,y) = sin(x)sin(y) over [0,π]×[0,π]");
    println!("Exact result: 4.0");

    // Create integration options with different dimension thresholds
    let romberg_opts = RombergOptions {
        max_true_dimension: 3, // Use grid-based method for 2D
        ..Default::default()
    };

    let monte_carlo_opts = RombergOptions {
        max_true_dimension: 1,          // Force Monte Carlo for 2D and higher
        min_monte_carlo_samples: 50000, // Use more samples for accuracy
        ..Default::default()
    };

    // Function to integrate: sin(x)sin(y) over [0,π]×[0,π]
    let f_2d = |x: ArrayView1<f64>| x[0].sin() * x[1].sin();
    let ranges = &[(0.0, PI), (0.0, PI)];

    // Integrate using the default method (grid-based for 2D)
    let result_default = time_integration("Default method (grid-based)", || {
        multi_romberg_with_details(f_2d, ranges, None).unwrap()
    });

    println!("  Result: {:.10}", result_default.value);
    println!("  Error estimate: {:.10}", result_default.abs_error);
    println!("  Method used: {:?}", result_default.method);
    println!(
        "  Absolute error: {:.10}",
        (result_default.value - 4.0).abs()
    );

    // Integrate using Monte Carlo
    let result_mc = time_integration("Monte Carlo method", || {
        multi_romberg_with_details(f_2d, ranges, Some(monte_carlo_opts.clone())).unwrap()
    });

    println!("  Result: {:.10}", result_mc.value);
    println!("  Error estimate: {:.10}", result_mc.abs_error);
    println!("  Method used: {:?}", result_mc.method);
    println!("  Absolute error: {:.10}", (result_mc.value - 4.0).abs());
    println!();

    // Example 2: 3D integration - unit sphere volume
    println!("Example 2: 3D Integration - Volume of unit sphere");
    println!("Exact result: {:.10}", 4.0 * PI / 3.0);

    // Function that is 1 inside the unit sphere, 0 outside
    let unit_sphere = |x: ArrayView1<f64>| {
        // Map from [0,1]³ to [-1,1]³ for easier sphere calculation
        let x_mapped = 2.0 * x[0] - 1.0;
        let y_mapped = 2.0 * x[1] - 1.0;
        let z_mapped = 2.0 * x[2] - 1.0;

        // Distance from origin
        let r_squared = x_mapped * x_mapped + y_mapped * y_mapped + z_mapped * z_mapped;

        // 1 if inside unit sphere, 0 otherwise
        if r_squared <= 1.0 {
            1.0
        } else {
            0.0
        }
    };

    // Need to multiply by 8 because we mapped from [0,1]³ to [-1,1]³ (volume factor)
    let result_3d = time_integration("3D integration (adaptive)", || {
        let raw_result = multi_romberg_with_details(
            unit_sphere,
            &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            Some(romberg_opts.clone()),
        )
        .unwrap();
        (raw_result.value * 8.0, raw_result.method)
    });

    println!("  Result: {:.10}", result_3d.0);
    println!("  Method used: {:?}", result_3d.1);
    println!(
        "  Absolute error: {:.10}",
        (result_3d.0 - 4.0 * PI / 3.0).abs()
    );
    println!();

    // Example 3: High-dimensional integration (5D)
    println!("Example 3: 5D Integration - Volume of 5D hypersphere");

    // The volume of a 5D hypersphere with radius 1 is pi^(5/2) / (5/2)!
    let exact_volume_5d = PI.powf(2.5) * 8.0 / 15.0;
    println!("Exact result: {:.10}", exact_volume_5d);

    // Function that is 1 inside the 5D hypersphere, 0 outside
    let hypersphere_5d = |x: ArrayView1<f64>| {
        // Map from [0,1]⁵ to [-1,1]⁵ for easier hypersphere calculation
        let r_squared: f64 = x
            .iter()
            .map(|&xi| 2.0 * xi - 1.0)
            .map(|xi_mapped| xi_mapped * xi_mapped)
            .sum();

        // 1 if inside unit hypersphere, 0 otherwise
        if r_squared <= 1.0 {
            1.0
        } else {
            0.0
        }
    };

    // Need to multiply by 2^5 = 32 for the domain mapping
    let result_5d = time_integration("5D integration (Monte Carlo)", || {
        let raw_result = multi_romberg_with_details(
            hypersphere_5d,
            &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            Some(RombergOptions {
                min_monte_carlo_samples: 100000, // More samples for higher dimensions
                ..Default::default()
            }),
        )
        .unwrap();
        (
            raw_result.value * 32.0,
            raw_result.method,
            raw_result.abs_error * 32.0,
        )
    });

    println!("  Result: {:.10}", result_5d.0);
    println!("  Method used: {:?}", result_5d.1);
    println!("  Error estimate: {:.10}", result_5d.2);
    println!(
        "  Absolute error: {:.10}",
        (result_5d.0 - exact_volume_5d).abs()
    );
    println!();

    // Example 4: Comparison of accuracy vs. speed for different methods
    println!("Example 4: Comparing accuracy and speed for different dimensions");

    // A function that's challenging to integrate: f(x₁,...,xₙ) = exp(-sum(xᵢ²))
    let gaussian_function = |x: ArrayView1<f64>| {
        let sum_of_squares: f64 = x.iter().map(|&xi| xi * xi).sum();
        (-sum_of_squares).exp()
    };

    // Test with different dimensions
    for dim in 1..=5 {
        println!("\nDimension {}", dim);

        // Create the appropriate ranges
        let ranges: Vec<(f64, f64)> = vec![(0.0, 1.0); dim];

        // Integration with default options
        let result = time_integration(&format!("Integration (dim={})", dim), || {
            multi_romberg_with_details(gaussian_function, &ranges, None).unwrap()
        });

        // For this special case, we can compute an analytical result
        // For dim-dimensional integral of exp(-sum(xᵢ²)) over [0,1]ᵈⁱᵐ
        // The exact result is (sqrt(π)/2 * erf(1))^dim
        let exact_result = (PI.sqrt() / 2.0 * libm::erf(1.0)).powi(dim as i32);

        println!("  Method used: {:?}", result.method);
        println!("  Result: {:.10}", result.value);
        println!("  Error estimate: {:.10}", result.abs_error);
        println!("  Exact result: {:.10}", exact_result);
        println!(
            "  Absolute error: {:.10}",
            (result.value - exact_result).abs()
        );
    }
}
