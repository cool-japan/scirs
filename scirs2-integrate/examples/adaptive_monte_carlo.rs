use ndarray::{Array1, ArrayView1};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use scirs2_integrate::monte_carlo::{importance_sampling, monte_carlo, MonteCarloOptions};
use scirs2_integrate::qmc::{qmc_quad, Halton, QRNGEngine, Sobol};
use std::f64::consts::PI;

/// Function with a singularity at (1,0) for demonstrating adaptive integration
fn singular_function(x: ArrayView1<f64>) -> f64 {
    // Function with singularity at x=1, y=0
    let r_squared = (x[0] - 1.0).powi(2) + x[1].powi(2);

    // Prevent true division by zero at the singularity
    if r_squared < 1e-14 {
        return 1e12; // Approximate the singularity with a large value
    }

    1.0 / r_squared
}

/// Heavy-tailed function that benefits from adaptive techniques
fn heavy_tailed_function(x: ArrayView1<f64>) -> f64 {
    1.0 / (1.0 + 10.0 * (x[0] - 0.5).powi(2) + 10.0 * (x[1] - 0.5).powi(2)).powi(3)
}

/// Adaptive importance sampling using domain decomposition
fn adaptive_importance_sampling<F>(
    f: F,
    ranges: &[(f64, f64)],
    n_samples_total: usize,
    n_initial_samples: usize,
    n_subregions: usize,
    seed: Option<u64>,
) -> (f64, f64)
where
    F: Fn(ArrayView1<f64>) -> f64 + Sync,
{
    // Step 1: Generate initial samples to identify important regions
    let _options = MonteCarloOptions::<f64> {
        n_samples: n_initial_samples,
        seed,
        ..Default::default()
    };

    // Generate initial points and evaluate the function
    // NOTE: We're using the same seed for all RNGs to ensure reproducibility
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(12345));

    let dim = ranges.len();
    let mut points = Vec::with_capacity(n_initial_samples);
    let mut values = Vec::with_capacity(n_initial_samples);

    // Create distributions for sampling
    let distributions: Vec<_> = ranges
        .iter()
        .map(|&(a, b)| rand_distr::Uniform::new_inclusive(a, b).unwrap())
        .collect();

    // Generate samples and evaluate function
    for _ in 0..n_initial_samples {
        let mut point = Array1::zeros(dim);
        for (i, dist) in distributions.iter().enumerate() {
            point[i] = dist.sample(&mut rng);
        }

        let value = f(point.view());

        // Filter out invalid values
        if !value.is_nan() && !value.is_infinite() {
            points.push(point);
            values.push(value);
        }
    }

    // Step 2: Sort values and identify high-contribution regions
    let mut value_indices: Vec<_> = (0..values.len()).collect();
    value_indices.sort_by(|&i, &j| {
        values[j]
            .abs()
            .partial_cmp(&values[i].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Step 3: Create subregions centered on the highest-valued points
    let n_top_points = n_subregions.min(value_indices.len());
    let mut subregion_results = Vec::with_capacity(n_top_points);

    let samples_per_region = (n_samples_total - n_initial_samples) / n_subregions;
    let domain_volume: f64 = ranges.iter().map(|&(a, b)| b - a).product();

    println!("Identified {} high-contribution regions", n_top_points);

    // Process each subregion
    for (idx, &point_idx) in value_indices.iter().take(n_top_points).enumerate() {
        let center_point = &points[point_idx];
        let center_value = values[point_idx];

        // The radius scales inversely with the function value
        // For high values, make a smaller region to capture more detail
        let radius = match center_value.abs() {
            v if v > 1e6 => 0.01,
            v if v > 1e3 => 0.05,
            v if v > 1e0 => 0.1,
            _ => 0.2,
        };

        println!(
            "Region {}: Center value = {:.6e}, Radius = {:.4}",
            idx + 1,
            center_value,
            radius
        );

        // Create a subregion with boundaries
        let mut subregion_ranges = Vec::with_capacity(dim);
        for i in 0..dim {
            let original_a = ranges[i].0;
            let original_b = ranges[i].1;

            // Compute subregion boundaries, ensuring they stay within original range
            let a = (center_point[i] - radius).max(original_a);
            let b = (center_point[i] + radius).min(original_b);

            subregion_ranges.push((a, b));
        }

        // Calculate subregion volume relative to full domain
        let subregion_volume: f64 = subregion_ranges.iter().map(|&(a, b)| b - a).product();
        let volume_ratio = subregion_volume / domain_volume;

        // Adjust sampling for this subregion based on function value and volume
        let importance = center_value.abs().min(1e12);
        let sample_weight = importance / (importance + 1.0);
        let min_samples = samples_per_region / 4;
        let adjusted_samples =
            min_samples + ((samples_per_region - min_samples) as f64 * sample_weight) as usize;

        println!(
            "  Volume ratio = {:.6}, Samples = {}",
            volume_ratio, adjusted_samples
        );

        // Clone the necessary data for the closures
        let center_point_clone = center_point.clone();
        let subregion_ranges_clone = subregion_ranges.clone();

        // Define a sampling distribution focused on the subregion center
        let normal_sampler = move |rng: &mut StdRng, dims: usize| {
            let mut point = Array1::zeros(dims);

            for i in 0..dims {
                let center = center_point_clone[i];
                let width = (subregion_ranges_clone[i].1 - subregion_ranges_clone[i].0) / 3.0;
                let normal = Normal::new(center, width).unwrap();

                // Sample and clamp to subregion
                let mut x: f64 = normal.sample(rng);
                x = x.clamp(subregion_ranges_clone[i].0, subregion_ranges_clone[i].1);
                point[i] = x;
            }

            point
        };

        // Clone again for the second closure
        let center_point_clone2 = center_point.clone();
        let subregion_ranges_clone2 = subregion_ranges.clone();

        // PDF for the normal distribution (truncated to the subregion)
        let normal_pdf = move |x: ArrayView1<f64>| {
            let mut pdf = 1.0;

            for i in 0..x.len() {
                let center = center_point_clone2[i];
                let width = (subregion_ranges_clone2[i].1 - subregion_ranges_clone2[i].0) / 3.0;
                // For smooth functions, we need to ensure we don't overshoot
                // Use a more conservative approach to the PDF calculation
                let z = (x[i] - center) / width;
                // Scale down the PDF for smoother functions to avoid overestimation
                let density = if z.abs() < 3.0 {
                    (-0.5 * z * z).exp() / (width * (2.0 * PI).sqrt())
                } else {
                    1e-10 // Very small value for points far from center
                };
                pdf *= density.max(1e-10); // Prevent underflow
            }

            pdf
        };

        // Create a new seed for this subregion
        let subregion_seed = seed.map(|s| s + idx as u64 * 1000);

        let options = MonteCarloOptions::<f64> {
            n_samples: adjusted_samples,
            seed: subregion_seed,
            ..Default::default()
        };

        // Integrate this subregion using importance sampling
        let subregion_result = importance_sampling(
            &f,
            normal_pdf,
            normal_sampler,
            &subregion_ranges,
            Some(options),
        )
        .unwrap();

        // Store the result
        subregion_results.push((
            subregion_result.value,
            subregion_result.std_error.powi(2), // Store variance for error propagation
        ));
    }

    // Step 4: Integrate the remainder of the domain with regular Monte Carlo
    let remaining_samples = n_samples_total
        - n_initial_samples
        - subregion_results
            .iter()
            .map(|(_, v)| *v as usize)
            .sum::<usize>();

    let options = MonteCarloOptions::<f64> {
        n_samples: remaining_samples,
        seed: seed.map(|s| s + 10000),
        ..Default::default()
    };

    println!("Sampling remainder with {} samples", remaining_samples);

    let remainder_result = monte_carlo(&f, ranges, Some(options)).unwrap();

    // Step 5: Combine results
    let mut total_integral = remainder_result.value;
    let mut total_variance = remainder_result.std_error.powi(2);

    for (value, variance) in subregion_results {
        total_integral += value;
        total_variance += variance;
    }

    let total_error = total_variance.sqrt();

    (total_integral, total_error)
}

/// Adaptive QMC integration using domain decomposition
fn adaptive_qmc<F>(
    f: F,
    ranges: &[(f64, f64)],
    n_initial_points: usize,
    n_subregions: usize,
    subregion_points: usize,
    seed: Option<u64>,
) -> (f64, f64)
where
    F: Fn(ArrayView1<f64>) -> f64 + Sync,
{
    let dim = ranges.len();

    // Step 1: First do a standard QMC integration of the whole domain
    // to establish a baseline and identify important regions
    let a = Array1::from_iter(ranges.iter().map(|&(a, _)| a));
    let b = Array1::from_iter(ranges.iter().map(|&(_, b)| b));

    // Get the total domain volume for later calculations
    let domain_volume: f64 = ranges.iter().map(|&(a, b)| b - a).product();

    // Use Halton sequence for the initial exploration
    let initial_qrng = Halton::new(dim, seed);
    let initial_result = qmc_quad(
        &f,
        &a,
        &b,
        Some(4), // Use multiple estimates for better error assessment
        Some(n_initial_points),
        Some(Box::new(initial_qrng)),
        false,
    )
    .unwrap();

    // Record the initial estimate for comparison
    println!(
        "Initial full-domain QMC result: {:.8}",
        initial_result.integral
    );

    // Step 2: Sample additional points to identify important regions
    // Generate QMC points from Sobol sequence for exploration (better coverage properties)
    let mut qrng = Sobol::new(dim, seed);
    let qmc_samples = qrng.random(n_initial_points);

    // Scale QMC points to the integration domain and evaluate the function
    let mut points = Vec::with_capacity(n_initial_points);
    let mut values = Vec::with_capacity(n_initial_points);
    let mut value_sum = 0.0;

    for i in 0..n_initial_points {
        let mut point = Array1::zeros(dim);
        for j in 0..dim {
            point[j] = ranges[j].0 + (ranges[j].1 - ranges[j].0) * qmc_samples[[i, j]];
        }

        let value = f(point.view());

        // Filter out invalid values
        if !value.is_nan() && !value.is_infinite() {
            // Keep track of the sum for calculating importance later
            value_sum += value.abs();
            points.push(point);
            values.push(value);
        }
    }

    // Step 3: Sort and identify important regions
    let mut value_indices: Vec<_> = (0..values.len()).collect();
    value_indices.sort_by(|&i, &j| {
        values[j]
            .abs()
            .partial_cmp(&values[i].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Step 4: Define subregions around important points
    let n_top_points = n_subregions.min(value_indices.len());
    let mut subregion_results = Vec::with_capacity(n_top_points);

    println!("Identified {} high-contribution regions", n_top_points);

    // Keep track of the total volume of all subregions to avoid double-counting
    let mut total_subregion_volume = 0.0;
    let mut subregion_volumes = Vec::with_capacity(n_top_points);

    for (idx, &point_idx) in value_indices.iter().take(n_top_points).enumerate() {
        let center_point = &points[point_idx];
        let center_value = values[point_idx];

        // Adaptive radius based on function value and position in sorted list
        // Use smaller radius for higher values to focus more precisely
        // Scale inversely with value rank to provide better coverage
        let radius_scale = 1.0 - (idx as f64 / n_top_points as f64).powf(0.5); // Square root decay
        let radius = match center_value.abs() {
            v if v > 1e6 => 0.05 * radius_scale,
            v if v > 1e3 => 0.1 * radius_scale,
            v if v > 1e0 => 0.15 * radius_scale,
            _ => 0.2 * radius_scale,
        };

        println!(
            "Region {}: Center value = {:.6e}, Radius = {:.4}",
            idx + 1,
            center_value,
            radius
        );

        // Create subregion boundaries
        let mut subregion_a = Array1::zeros(dim);
        let mut subregion_b = Array1::zeros(dim);

        for i in 0..dim {
            subregion_a[i] = (center_point[i] - radius).max(ranges[i].0);
            subregion_b[i] = (center_point[i] + radius).min(ranges[i].1);
        }

        // Calculate subregion volume
        let subregion_volume: f64 = (0..dim).map(|i| subregion_b[i] - subregion_a[i]).product();

        // Adjust points based on relative importance and volume
        let relative_importance = center_value.abs() / value_sum;
        let adjusted_points =
            (subregion_points as f64 * (relative_importance * n_top_points as f64)) as usize;
        let points_to_use = adjusted_points
            .max(subregion_points / 4)
            .min(subregion_points * 2);

        println!(
            "  Volume: {:.6}, Points: {}",
            subregion_volume, points_to_use
        );

        // Integrate the subregion using QMC with Sobol sequence for better convergence
        let subregion_qrng = Sobol::new(dim, seed.map(|s| s + idx as u64 * 1000));
        let subregion_result = qmc_quad(
            &f,
            &subregion_a,
            &subregion_b,
            Some(8), // More independent estimates for better error estimation
            Some(points_to_use),
            Some(Box::new(subregion_qrng)),
            false,
        )
        .unwrap();

        println!("  Subintegral: {:.8}", subregion_result.integral);

        subregion_results.push(subregion_result);
        subregion_volumes.push(subregion_volume);
        total_subregion_volume += subregion_volume;
    }

    // Step 5: Analyze function properties to choose the best combination method
    // Calculate coefficient of variation (CV) to determine function smoothness
    let mean_value = value_sum / values.len() as f64;
    let variance: f64 = values
        .iter()
        .map(|&v| (v.abs() - mean_value).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let std_dev = variance.sqrt();
    let cv = std_dev / mean_value;

    println!("Function smoothness analysis:");
    println!("  Mean value: {:.6e}", mean_value);
    println!("  Std deviation: {:.6e}", std_dev);
    println!("  Coefficient of variation: {:.6}", cv);

    // Check if the function is smooth or has singularities/peaks
    // Smooth functions have lower coefficient of variation

    // Calculate coverage for determining combination method
    let coverage_ratio = total_subregion_volume / domain_volume;
    println!(
        "Subregion coverage: {:.2}% of total domain",
        coverage_ratio * 100.0
    );

    // Choose combination method based on function properties and coverage
    let (total_integral, total_error) = if cv < 0.8 {
        // For very smooth functions, the initial QMC estimate is often best
        println!("Function appears very smooth, using initial QMC estimate");
        (initial_result.integral, initial_result.standard_error)
    } else if coverage_ratio > 0.5 {
        // Method 1: Weighted average of initial estimate and subregion results
        // This works well when subregions cover most of the domain
        let mut combined_integral = 0.0;
        let mut combined_variance = 0.0;

        // Weight for the initial estimate (decreases as coverage increases)
        let initial_weight = (1.0 - coverage_ratio).max(0.0).min(0.5);
        combined_integral += initial_result.integral * initial_weight;
        combined_variance += (initial_result.standard_error * initial_weight).powi(2);

        // Weight for the subregion results (increases with coverage)
        let subregion_weight = 1.0 - initial_weight;
        let mut total_subregion_integral = 0.0;
        let mut total_subregion_variance = 0.0;

        for (i, result) in subregion_results.iter().enumerate() {
            // Weight each subregion by its volume relative to total subregion volume
            let region_weight = subregion_volumes[i] / total_subregion_volume;
            total_subregion_integral += result.integral * region_weight;
            total_subregion_variance += (result.standard_error * region_weight).powi(2);
        }

        combined_integral += total_subregion_integral * subregion_weight;
        combined_variance += total_subregion_variance * subregion_weight.powi(2);

        (combined_integral, combined_variance.sqrt())
    } else {
        // Method 2: Use initial estimate as baseline and add corrective terms from subregions
        // This works better when subregions are focused on a small part of the domain
        let mut refined_integral = initial_result.integral;
        let mut refined_variance = initial_result.standard_error.powi(2);

        println!("Using corrective refinement method");
        let mut total_correction = 0.0;

        for (i, result) in subregion_results.iter().enumerate() {
            // For each subregion, estimate what portion of the initial integral
            // came from this region, and replace it with the more accurate estimate
            let volume_ratio = subregion_volumes[i] / domain_volume;
            let estimated_contribution = initial_result.integral * volume_ratio;
            let correction = result.integral - estimated_contribution;

            println!(
                "  Region {}: Est. contribution: {:.8}, Correction: {:.8}",
                i + 1,
                estimated_contribution,
                correction
            );

            total_correction += correction;
            // Add the variance of the correction
            refined_variance += (result.standard_error).powi(2)
                + (initial_result.standard_error * volume_ratio).powi(2);
        }

        refined_integral += total_correction;
        println!("Total correction: {:.8}", total_correction);

        (refined_integral, refined_variance.sqrt())
    };

    println!("Combined estimate: {:.8}", total_integral);
    (total_integral, total_error)
}

fn main() {
    println!("Advanced Adaptive Monte Carlo Integration Examples");
    println!("================================================\n");

    // Example 1: Integrating a function with a singularity
    println!("Example 1: Integrating a function with a singularity at (1,0)");
    println!("Function: f(x,y) = 1/((x-1)² + y²)");
    println!("Domain: [0,2] × [-0.5,0.5]");
    println!("Note: This function has a singularity at (1,0) that makes");
    println!("      standard integration methods challenging.\n");

    // First, try standard Monte Carlo
    let options_regular = MonteCarloOptions::<f64> {
        n_samples: 100_000,
        seed: Some(12345),
        ..Default::default()
    };

    let regular_result = monte_carlo(
        singular_function,
        &[(0.0, 2.0), (-0.5, 0.5)],
        Some(options_regular),
    )
    .unwrap();

    println!("Standard Monte Carlo result:");
    println!("  Integral estimate: {:.8}", regular_result.value);
    println!("  Standard error: {:.8}", regular_result.std_error);
    println!("  Number of evaluations: {}", regular_result.n_evals);

    // Now try adaptive importance sampling
    println!("\nAdaptive importance sampling result:");
    let (adaptive_value, adaptive_error) = adaptive_importance_sampling(
        singular_function,
        &[(0.0, 2.0), (-0.5, 0.5)],
        100_000,     // Total samples
        5_000,       // Initial exploratory samples
        8,           // Number of adaptive subregions
        Some(12345), // Seed for reproducibility
    );

    println!("  Integral estimate: {:.8}", adaptive_value);
    println!("  Error estimate: {:.8}", adaptive_error);

    // The singularity at (1,0) makes the integral unbounded
    // But we can compute a reference value for a slightly shifted domain
    // For reference, the integral of 1/(x² + y²) over [0,2]×[-0.5,0.5] without the singularity
    // would be approximately arctan(0.5/2) - arctan(-0.5/2) ≈ 0.49

    println!("\nObserve how adaptive sampling handles the singularity better by");
    println!("concentrating samples in the high-contribution regions near (1,0).");

    // Example 2: Heavy-tailed function with sharp peak
    println!("\n\nExample 2: Integrating a heavy-tailed function with sharp peak");
    println!("Function: f(x,y) = 1/(1 + 10(x-0.5)² + 10(y-0.5)²)³");
    println!("Domain: [0,1] × [0,1]");
    println!("Exact value: Approximately 0.245...\n");

    // Try standard QMC
    let a = Array1::from_vec(vec![0.0, 0.0]);
    let b = Array1::from_vec(vec![1.0, 1.0]);

    let standard_qmc = qmc_quad(
        heavy_tailed_function,
        &a,
        &b,
        Some(8),    // Number of estimates
        Some(5000), // Points per estimate
        None,       // Default QRNG
        false,
    )
    .unwrap();

    println!("Standard QMC result:");
    println!("  Integral estimate: {:.8}", standard_qmc.integral);
    println!("  Error estimate: {:.8}", standard_qmc.standard_error);

    // Try adaptive QMC
    println!("\nAdaptive QMC result:");
    let (adaptive_qmc_value, adaptive_qmc_error) = adaptive_qmc(
        heavy_tailed_function,
        &[(0.0, 1.0), (0.0, 1.0)],
        5000, // Initial points
        5,    // Number of subregions
        5000, // Points per subregion (same as initial for fair comparison)
        Some(12345),
    );

    println!("  Integral estimate: {:.8}", adaptive_qmc_value);
    println!("  Error estimate: {:.8}", adaptive_qmc_error);

    // Reference value computed with high-precision adaptive methods
    let reference_value = 0.244662;
    println!("\nReference value: {:.8}", reference_value);
    println!(
        "Standard QMC error: {:.8}",
        (standard_qmc.integral - reference_value).abs()
    );
    println!(
        "Adaptive QMC error: {:.8}",
        (adaptive_qmc_value - reference_value).abs()
    );

    println!("\nObserve how adaptive QMC provides better accuracy");
    println!("by focusing computational effort on the peak region.");

    // Example 3: Comparison on a smoother function
    println!("\n\nExample 3: Comparison on a smoother function");
    println!("Function: f(x,y) = sin(π·x)·sin(π·y)");
    println!("Domain: [0,1] × [0,1]");
    println!("Exact value: 4/π² ≈ 0.4053...\n");

    let smooth_function = |x: ArrayView1<f64>| (PI * x[0]).sin() * (PI * x[1]).sin();
    let exact_value = 4.0 / (PI * PI);

    // Try standard Monte Carlo
    let options_smooth = MonteCarloOptions::<f64> {
        n_samples: 50_000,
        seed: Some(12345),
        ..Default::default()
    };

    let smooth_mc = monte_carlo(
        smooth_function,
        &[(0.0, 1.0), (0.0, 1.0)],
        Some(options_smooth),
    )
    .unwrap();

    println!("Standard Monte Carlo result:");
    println!("  Integral estimate: {:.8}", smooth_mc.value);
    println!("  Standard error: {:.8}", smooth_mc.std_error);
    println!(
        "  Absolute error: {:.8}",
        (smooth_mc.value - exact_value).abs()
    );

    // Try standard QMC
    let smooth_qmc = qmc_quad(
        smooth_function,
        &a,
        &b,
        Some(8),    // Number of estimates
        Some(5000), // Points per estimate
        None,       // Default QRNG
        false,
    )
    .unwrap();

    println!("\nStandard QMC result:");
    println!("  Integral estimate: {:.8}", smooth_qmc.integral);
    println!("  Error estimate: {:.8}", smooth_qmc.standard_error);
    println!(
        "  Absolute error: {:.8}",
        (smooth_qmc.integral - exact_value).abs()
    );

    // For this smooth function, regular QMC should actually perform quite well
    // But we can still try the adaptive version for comparison
    let (adaptive_smooth, adaptive_smooth_error) = adaptive_qmc(
        smooth_function,
        &[(0.0, 1.0), (0.0, 1.0)],
        5000, // Initial points
        4,    // Number of subregions
        5000, // Points per subregion (same as standard for fair comparison)
        Some(12345),
    );

    println!("\nAdaptive QMC result:");
    println!("  Integral estimate: {:.8}", adaptive_smooth);
    println!("  Error estimate: {:.8}", adaptive_smooth_error);
    println!(
        "  Absolute error: {:.8}",
        (adaptive_smooth - exact_value).abs()
    );

    println!("\nExact value: {:.8} (4/π²)", exact_value);
    println!("\nFor this smooth, well-behaved function, standard QMC usually");
    println!("performs very well, while adaptive methods may not offer as much improvement");
    println!("as they do for functions with singularities or sharp features.");
}
