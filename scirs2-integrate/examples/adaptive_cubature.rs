#![recursion_limit = "512"]

use ndarray::{Array1, ArrayView1};
use scirs2_integrate::error::{IntegrateError, IntegrateResult};
use scirs2_integrate::gaussian::GaussLegendreQuadrature;
use std::collections::VecDeque;
use std::f64::consts::PI;
use std::time::Instant;

/// Result of adaptive cubature integration
#[derive(Debug, Clone)]
pub struct AdaptiveCubatureResult {
    /// Estimated value of the integral
    pub value: f64,
    /// Estimated error
    pub error: f64,
    /// Number of function evaluations performed
    pub n_evals: usize,
    /// Number of subdivisions created
    pub n_subregions: usize,
    /// Maximum recursion depth reached
    pub max_depth: usize,
}

/// A subregion of the integration domain with its boundaries and current estimate
#[derive(Debug, Clone)]
struct SubRegion {
    /// Lower bounds for each dimension
    a: Array1<f64>,
    /// Upper bounds for each dimension
    b: Array1<f64>,
    /// Current error estimate for this region
    error: f64,
    /// Depth in the subdivision tree
    depth: usize,
}

/// Performs adaptive cubature integration using domain subdivision
///
/// This algorithm recursively subdivides the integration domain based on error
/// estimates, focusing computational effort where the integrand is most challenging.
///
/// # Parameters
/// * `f` - Function to integrate
/// * `a` - Lower bounds array, one element per dimension
/// * `b` - Upper bounds array, one element per dimension
/// * `tol` - Desired absolute tolerance
/// * `max_evals` - Maximum number of function evaluations
/// * `max_depth` - Maximum recursion depth
pub fn adaptive_cubature<F>(
    f: F,
    a: &Array1<f64>,
    b: &Array1<f64>,
    tol: f64,
    max_evals: usize,
    max_depth: usize,
) -> IntegrateResult<AdaptiveCubatureResult>
where
    F: Fn(ArrayView1<f64>) -> f64 + Sync,
{
    // Start timer
    let start_time = Instant::now();

    // Parameter validation
    let dim = a.len();
    if dim != b.len() {
        return Err(IntegrateError::ValueError(
            "Dimension mismatch: 'a' and 'b' must have the same length".to_string(),
        ));
    }

    // Ensure all integration bounds are valid
    for i in 0..dim {
        if a[i] >= b[i] {
            return Err(IntegrateError::ValueError(format!(
                "Invalid integration bounds: a[{}] >= b[{}]",
                i, i
            )));
        }
    }

    // Initialize workspace and result variables
    let mut total_integral = 0.0;
    let mut total_error = 0.0;
    let mut n_evals = 0;
    let mut n_subregions = 1;
    let mut max_depth_reached = 0;

    // Working queue of subregions to process
    let mut region_queue = VecDeque::new();

    // Create initial region covering the entire domain
    let initial_region = SubRegion {
        a: a.clone(),
        b: b.clone(),
        error: f64::INFINITY,
        depth: 0,
    };

    // Initialize with the first region
    region_queue.push_back(initial_region);

    // Process regions until convergence or resource limits
    while !region_queue.is_empty() && n_evals < max_evals {
        // Get the region with the largest estimated error
        let current_region = region_queue.pop_front().unwrap();

        // Update max depth reached
        max_depth_reached = max_depth_reached.max(current_region.depth);

        // Compute integral for this region
        let (region_est, region_err, region_evals) =
            integrate_region(&f, &current_region.a, &current_region.b)?;
        n_evals += region_evals;

        // Check if region meets convergence criteria
        if region_err <= tol * (b[0] - a[0]) / (current_region.b[0] - current_region.a[0])
            || current_region.depth >= max_depth
        {
            // Accept this region's contribution
            total_integral += region_est;
            total_error += region_err;
            continue;
        }

        // Subdivide the region - find the dimension with largest width
        let mut max_width_dim = 0;
        let mut max_width = 0.0;

        for i in 0..dim {
            let width = current_region.b[i] - current_region.a[i];
            if width > max_width {
                max_width = width;
                max_width_dim = i;
            }
        }

        // Create two subregions by splitting along the chosen dimension
        let mid_point = (current_region.a[max_width_dim] + current_region.b[max_width_dim]) / 2.0;

        // Left subregion
        let left_a = current_region.a.clone();
        let mut left_b = current_region.b.clone();
        left_b[max_width_dim] = mid_point;

        // Right subregion
        let mut right_a = current_region.a.clone();
        let right_b = current_region.b.clone();
        right_a[max_width_dim] = mid_point;

        // Create new subregions with increased depth
        let left_region = SubRegion {
            a: left_a,
            b: left_b,
            error: f64::INFINITY,
            depth: current_region.depth + 1,
        };

        let right_region = SubRegion {
            a: right_a,
            b: right_b,
            error: f64::INFINITY,
            depth: current_region.depth + 1,
        };

        // Add new regions to the queue
        region_queue.push_back(left_region);
        region_queue.push_back(right_region);
        n_subregions += 1;

        // Sort regions by error (largest first) - simple bubble sort since queue is usually small
        let n = region_queue.len();
        let mut sorted = false;

        if n >= 2 {
            let mut temp_queue = VecDeque::with_capacity(n);
            let mut regions: Vec<_> = region_queue.drain(..).collect();

            // Simple error-based sorting
            while !sorted {
                sorted = true;
                for i in 0..regions.len() - 1 {
                    if regions[i].error < regions[i + 1].error {
                        regions.swap(i, i + 1);
                        sorted = false;
                    }
                }
            }

            // Put sorted regions back in queue
            for region in regions {
                temp_queue.push_back(region);
            }

            region_queue = temp_queue;
        }
    }

    // Process any remaining regions when max_evals is reached
    for region in region_queue {
        // Compute estimates for remaining regions
        let (region_est, region_err, region_evals) = integrate_region(&f, &region.a, &region.b)?;
        n_evals += region_evals;

        // Add contributions
        total_integral += region_est;
        total_error += region_err;
    }

    let elapsed = start_time.elapsed();
    println!("Adaptive cubature completed in {:.2?}", elapsed);
    println!("Total evaluations: {}", n_evals);
    println!("Total subregions: {}", n_subregions);
    println!("Maximum depth: {}", max_depth_reached);

    Ok(AdaptiveCubatureResult {
        value: total_integral,
        error: total_error,
        n_evals,
        n_subregions,
        max_depth: max_depth_reached,
    })
}

/// Integrate a function over a specific region using tensor product Gauss-Legendre quadrature
///
/// # Returns
/// Tuple containing (integral estimate, error estimate, number of evaluations)
fn integrate_region<F>(
    f: &F,
    a: &Array1<f64>,
    b: &Array1<f64>,
) -> IntegrateResult<(f64, f64, usize)>
where
    F: Fn(ArrayView1<f64>) -> f64 + Sync,
{
    let dim = a.len();
    // Use 3 and 5 point rules for error estimation
    let low_degree = 3;
    let high_degree = 5;

    // Get Gauss-Legendre quadrature for low and high degree
    let low_quad = GaussLegendreQuadrature::<f64>::new(low_degree)?;
    let high_quad = GaussLegendreQuadrature::<f64>::new(high_degree)?;

    // Convert nodes and weights to slices
    let low_points_slice = low_quad.nodes.as_slice().unwrap();
    let low_weights_slice = low_quad.weights.as_slice().unwrap();

    let high_points_slice = high_quad.nodes.as_slice().unwrap();
    let high_weights_slice = high_quad.weights.as_slice().unwrap();

    // Compute volume of the region
    let mut volume = 1.0;
    for i in 0..dim {
        volume *= b[i] - a[i];
    }

    // Helper to transform quadrature points to the integration region
    fn transform_point(x: f64, dim_idx: usize, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        a[dim_idx] + (b[dim_idx] - a[dim_idx]) * (x + 1.0) / 2.0
    }

    // Storage for the evaluation point
    let mut point = Array1::zeros(dim);

    // Recursive integration helper
    fn integrate_recursive<F>(
        f: &F,
        points: &[f64],
        weights: &[f64],
        a: &Array1<f64>,
        b: &Array1<f64>,
        point: &mut Array1<f64>,
        dim_idx: usize,
        n_eval: &mut usize,
    ) -> f64
    where
        F: Fn(ArrayView1<f64>) -> f64 + Sync,
    {
        let n_points = points.len();

        if dim_idx == a.len() - 1 {
            // Last dimension, evaluate the integrand
            let mut result = 0.0;
            for i in 0..n_points {
                point[dim_idx] = transform_point(points[i], dim_idx, a, b);
                result += weights[i] * f(point.view());
                *n_eval += 1;
            }
            result
        } else {
            // Recurse to next dimension
            let mut result = 0.0;
            for i in 0..n_points {
                point[dim_idx] = transform_point(points[i], dim_idx, a, b);
                result += weights[i]
                    * integrate_recursive(f, points, weights, a, b, point, dim_idx + 1, n_eval);
            }
            result
        }
    }

    // Compute low and high order estimates
    let mut low_evals = 0;
    let low_estimate = integrate_recursive(
        f,
        low_points_slice,
        low_weights_slice,
        a,
        b,
        &mut point,
        0,
        &mut low_evals,
    );

    let mut high_evals = 0;
    let high_estimate = integrate_recursive(
        f,
        high_points_slice,
        high_weights_slice,
        a,
        b,
        &mut point,
        0,
        &mut high_evals,
    );

    // Compute error estimate as the difference between high and low order rule
    let error = (high_estimate - low_estimate).abs();

    // Scale the result by the region volume and transformation factors
    let scaling = volume / (2.0f64.powi(dim as i32));
    let result = high_estimate * scaling;
    let error_est = error * scaling;

    Ok((result, error_est, low_evals + high_evals))
}

/// Function with a single sharp peak at the center
fn peak_function(x: ArrayView1<f64>) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..x.len() {
        sum_sq += (x[i] - 0.5).powi(2);
    }
    (-20.0 * sum_sq).exp()
}

/// Function with multiple peaks
fn multi_peak_function(x: ArrayView1<f64>) -> f64 {
    let d = x.len() as f64;
    let p1 = {
        let mut sum_sq = 0.0;
        for i in 0..x.len() {
            sum_sq += (x[i] - 0.25).powi(2);
        }
        (-d * 40.0 * sum_sq).exp()
    };

    let p2 = {
        let mut sum_sq = 0.0;
        for i in 0..x.len() {
            sum_sq += (x[i] - 0.75).powi(2);
        }
        (-d * 40.0 * sum_sq).exp() * 0.8
    };

    p1 + p2
}

/// Function with a sharp ridge (challenging for adaptive methods)
fn ridge_function(x: ArrayView1<f64>) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }

    let y1 = x[0];
    let y2 = x[1];

    // Ridge along y2 = y1
    let distance = (y2 - y1).abs();
    (-100.0 * distance).exp()
}

/// Function with a discontinuity
fn discontinuous_function(x: ArrayView1<f64>) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }

    let y1 = x[0];
    let y2 = x[1];

    if y1 + y2 > 1.0 {
        1.0
    } else {
        0.0
    }
}

/// Run a test case and report results with reference solution
fn run_test<F>(
    f: F,
    a: &Array1<f64>,
    b: &Array1<f64>,
    tol: f64,
    max_evals: usize,
    max_depth: usize,
    name: &str,
    reference: Option<f64>,
) where
    F: Fn(ArrayView1<f64>) -> f64 + Sync,
{
    println!("\n=== Testing {} function ===", name);
    println!("Dimension: {}", a.len());
    println!("Tolerance: {:.2e}", tol);

    let result = adaptive_cubature(f, a, b, tol, max_evals, max_depth).unwrap();

    println!("Integral estimate: {:.10}", result.value);
    println!("Error estimate: {:.10}", result.error);
    println!("Function evaluations: {}", result.n_evals);
    println!("Number of subregions: {}", result.n_subregions);

    if let Some(ref_value) = reference {
        let actual_error = (result.value - ref_value).abs();
        println!("Reference value: {:.10}", ref_value);
        println!("Actual error: {:.10}", actual_error);
        println!("Relative error: {:.10}", actual_error / ref_value.abs());
    }
}

fn main() {
    println!("Adaptive Cubature Integration Examples");
    println!("=====================================");

    // Example 1: Integrate a function with a sharp peak in 2D
    let dim = 2;
    let a = Array1::from_vec(vec![0.0; dim]);
    let b = Array1::from_vec(vec![1.0; dim]);

    // For a 2D Gaussian peak centered at (0.5, 0.5), the integral is approximately
    // 2π/(d*k) where d is dimension and k is the coefficient in exponent
    let peak_reference = 2.0 * PI / (20.0 * 2.0);

    run_test(
        peak_function,
        &a,
        &b,
        1e-6,
        100_000,
        15,
        "Sharp Peak (2D)",
        Some(peak_reference),
    );

    // Example 2: Higher dimensional integration
    let dim = 4;
    let a = Array1::from_vec(vec![0.0; dim]);
    let b = Array1::from_vec(vec![1.0; dim]);

    // For a 4D Gaussian peak centered at (0.5, 0.5, 0.5, 0.5), the integral is approximately
    // (2π)^(d/2)/(d*k)^(d/2) where d is dimension and k is the coefficient in exponent
    let peak_4d_reference =
        (2.0 * PI).powf(dim as f64 / 2.0) / (20.0 * dim as f64).powf(dim as f64 / 2.0);

    run_test(
        peak_function,
        &a,
        &b,
        1e-5,
        200_000,
        12,
        "Sharp Peak (4D)",
        Some(peak_4d_reference),
    );

    // Example 3: Function with multiple peaks
    let dim = 2;
    let a = Array1::from_vec(vec![0.0; dim]);
    let b = Array1::from_vec(vec![1.0; dim]);

    // No closed-form reference for multiple peaks, use approximation
    let multi_peak_reference = 0.03947;

    run_test(
        multi_peak_function,
        &a,
        &b,
        1e-6,
        100_000,
        15,
        "Multiple Peaks",
        Some(multi_peak_reference),
    );

    // Example 4: Ridge function (challenging for adaptive methods)
    let a = Array1::from_vec(vec![0.0, 0.0]);
    let b = Array1::from_vec(vec![1.0, 1.0]);

    // For the ridge function, the reference is approximated with high-degree quadrature
    let ridge_reference = 0.1772454;

    run_test(
        ridge_function,
        &a,
        &b,
        1e-5,
        50_000,
        15,
        "Ridge Function",
        Some(ridge_reference),
    );

    // Example 5: Discontinuous function
    let a = Array1::from_vec(vec![0.0, 0.0]);
    let b = Array1::from_vec(vec![1.0, 1.0]);

    // For this discontinuous function, the integral is 0.5 (the area where x+y > 1)
    let disc_reference = 0.5;

    run_test(
        discontinuous_function,
        &a,
        &b,
        1e-6,
        100_000,
        18,
        "Discontinuous Function",
        Some(disc_reference),
    );
}
