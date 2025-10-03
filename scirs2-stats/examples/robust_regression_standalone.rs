// Standalone robust regression example - doesn't require scirs2-stats compilation
use std::time::{SystemTime, UNIX_EPOCH};

/// A simple result for Theil-Sen regression
struct TheilSlopesResult {
    slope: f64,
    intercept: f64,
    slope_low: f64,
    slope_high: f64,
}

/// Result for linear regression
struct RegressionResult {
    slope: f64,
    intercept: f64,
    r_squared: f64,
    inlier_mask: Vec<bool>,
}

/// Simple implementation of Theil-Sen estimator
#[allow(dead_code)]
fn simple_theilslopes(x: &[f64], y: &[f64]) -> TheilSlopesResult {
    assert_eq!(x.len(), y.len(), "x and y must have same length");

    let n = x.len();
    let mut slopes = Vec::with_capacity(n * (n - 1) / 2);

    // Calculate all pairwise slopes
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[j] - x[i];
            if dx.abs() > f64::EPSILON {
                let dy = y[j] - y[i];
                slopes.push(dy / dx);
            }
        }
    }

    // Sort slopes to find median
    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Get median slope
    let mid = slopes.len() / 2;
    let slope = if slopes.len() % 2 == 0 && !slopes.is_empty() {
        (slopes[mid - 1] + slopes[mid]) / 2.0
    } else if !slopes.is_empty() {
        slopes[mid]
    } else {
        0.0 // No valid slopes, return zero
    };

    // Calculate the y-intercept using median values
    let x_median = x.iter().sum::<f64>() / n as f64;
    let y_median = y.iter().sum::<f64>() / n as f64;
    let intercept = y_median - slope * x_median;

    // Confidence interval for slope (simplified approximation based on Sen's method)
    let n_float = n as f64;
    let sigma = 1.0 / (1.5 * n_float.sqrt());
    let z = 1.96; // Approximate 95% confidence
    let margin = z * sigma;

    TheilSlopesResult {
        slope,
        intercept,
        slope_low: slope - margin,
        slope_high: slope + margin,
    }
}

/// Simple OLS (Ordinary Least Squares) implementation
#[allow(dead_code)]
fn simple_ols(x: &[f64], y: &[f64]) -> RegressionResult {
    assert_eq!(x.len(), y.len(), "x and y must have same length");

    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..x.len() {
        let x_dev = x[i] - x_mean;
        let y_dev = y[i] - y_mean;
        numerator += x_dev * y_dev;
        denominator += x_dev * x_dev;
    }

    let slope = numerator / denominator;
    let intercept = y_mean - slope * x_mean;

    // Calculate fitted values and residuals
    let mut fitted_values = Vec::with_capacity(x.len());
    let mut residuals = Vec::with_capacity(x.len());

    for i in 0..x.len() {
        let y_pred = intercept + slope * x[i];
        fitted_values.push(y_pred);
        residuals.push(y[i] - y_pred);
    }

    // Calculate R-squared
    let ss_total = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>();
    let ss_residual = residuals.iter().map(|&r| r.powi(2)).sum::<f64>();
    let r_squared = 1.0 - ss_residual / ss_total;

    RegressionResult {
        slope,
        intercept,
        r_squared,
        inlier_mask: vec![true; x.len()], // All points are inliers in OLS
    }
}

/// Simple implementation of RANSAC (Random Sample Consensus) regression
#[allow(dead_code)]
fn simple_ransac(x: &[f64], y: &[f64], threshold: f64, ntrials: usize) -> RegressionResult {
    assert_eq!(x.len(), y.len(), "x and y must have same length");

    let n = x.len();
    if n < 2 {
        panic!("Need at least 2 points for RANSAC");
    }

    // To keep code simple, we always sample 2 points
    let _min_samples = 2;

    // Use a simple random number generator
    let mut rng = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let mut best_inlier_count = 0;
    let mut best_model = (0.0, 0.0); // (slope, intercept)
    let mut best_inlier_mask = vec![false; n];

    for _ in 0..ntrials {
        // Select 2 random points
        let idx1 = (rng % n as u64) as usize;
        rng = (rng.wrapping_mul(1103515245).wrapping_add(12345)) % 2147483648; // Simple LCG
        let idx2 = (rng % n as u64) as usize;
        rng = (rng.wrapping_mul(1103515245).wrapping_add(12345)) % 2147483648;

        // Skip if we selected the same index twice
        if idx1 == idx2 {
            continue;
        }

        // Get the points
        let x1 = x[idx1];
        let y1 = y[idx1];
        let x2 = x[idx2];
        let y2 = y[idx2];

        // Calculate slope and intercept
        let dx = x2 - x1;
        if dx.abs() < f64::EPSILON {
            continue; // Skip vertical lines
        }

        let slope = (y2 - y1) / dx;
        let intercept = y1 - slope * x1;

        // Determine inliers
        let mut inlier_mask = vec![false; n];
        let mut inlier_count = 0;

        for i in 0..n {
            let y_pred = slope * x[i] + intercept;
            let residual = (y[i] - y_pred).abs();

            if residual < threshold {
                inlier_mask[i] = true;
                inlier_count += 1;
            }
        }

        // Update best model if we found more inliers
        if inlier_count > best_inlier_count {
            best_inlier_count = inlier_count;
            best_model = (slope, intercept);
            best_inlier_mask = inlier_mask;
        }
    }

    // Calculate R-squared for the best model
    let (slope, intercept) = best_model;

    // Use the inliers to calculate statistics
    let mut inlier_x = Vec::new();
    let mut inlier_y = Vec::new();

    for i in 0..n {
        if best_inlier_mask[i] {
            inlier_x.push(x[i]);
            inlier_y.push(y[i]);
        }
    }

    // If we found inliers, calculate R-squared
    let r_squared = if !inlier_x.is_empty() {
        let y_mean = inlier_y.iter().sum::<f64>() / inlier_y.len() as f64;

        let mut ss_total = 0.0;
        let mut ss_residual = 0.0;

        for i in 0..inlier_x.len() {
            let y_pred = intercept + slope * inlier_x[i];
            ss_total += (inlier_y[i] - y_mean).powi(2);
            ss_residual += (inlier_y[i] - y_pred).powi(2);
        }

        1.0 - ss_residual / ss_total
    } else {
        0.0 // No inliers found
    };

    RegressionResult {
        slope,
        intercept,
        r_squared,
        inlier_mask: best_inlier_mask,
    }
}

/// A simplified implementation of Huber regression
#[allow(dead_code)]
fn simple_huber(x: &[f64], y: &[f64], epsilon: f64, maxiter: usize) -> RegressionResult {
    assert_eq!(x.len(), y.len(), "x and y must have same length");

    // Start with OLS estimate
    let RegressionResult {
        mut slope,
        mut intercept,
        ..
    } = simple_ols(x, y);

    for _ in 0..maxiter {
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        // Calculate Huber weights and weighted OLS
        for i in 0..x.len() {
            let y_pred = intercept + slope * x[i];
            let residual = y[i] - y_pred;

            // Huber weight function
            let weight = if residual.abs() <= epsilon {
                1.0 // Use full weight for small residuals
            } else {
                epsilon / residual.abs() // Reduce weight for larger residuals
            };

            // Weighted contribution to slope/intercept calculation
            numerator += weight * x[i] * (y[i] - intercept);
            denominator += weight * x[i] * x[i];
        }

        let new_slope = numerator / denominator;

        // Calculate new intercept using Huber weights
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for i in 0..x.len() {
            let y_pred = intercept + new_slope * x[i];
            let residual = y[i] - y_pred;

            // Huber weight function
            let weight = if residual.abs() <= epsilon {
                1.0
            } else {
                epsilon / residual.abs()
            };

            weighted_sum += weight * (y[i] - new_slope * x[i]);
            weight_sum += weight;
        }

        let new_intercept = weighted_sum / weight_sum;

        // Check convergence
        let slope_change = (new_slope - slope).abs();
        let intercept_change = (new_intercept - intercept).abs();

        if slope_change < 1e-6 && intercept_change < 1e-6 {
            break;
        }

        slope = new_slope;
        intercept = new_intercept;
    }

    // Calculate R-squared
    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let mut ss_total = 0.0;
    let mut ss_residual = 0.0;

    for i in 0..x.len() {
        let y_pred = intercept + slope * x[i];
        ss_total += (y[i] - y_mean).powi(2);
        ss_residual += (y[i] - y_pred).powi(2);
    }

    let r_squared = 1.0 - ss_residual / ss_total;

    RegressionResult {
        slope,
        intercept,
        r_squared,
        inlier_mask: vec![true; x.len()], // All points considered inliers in Huber regression
    }
}

/// Text-based visualization of regression lines
#[allow(dead_code)]
fn visualize_regression_lines(
    x: &[f64],
    y: &[f64],
    models: &[(&str, f64, f64)], // (name, slope, intercept)
    outliers: &[bool],
) {
    println!("\nVisualization of regression lines:");

    // Determine y range for plotting
    let mut y_min = y[0];
    let mut y_max = y[0];

    for &yi in y.iter() {
        if yi < y_min {
            y_min = yi;
        }
        if yi > y_max {
            y_max = yi;
        }
    }

    // Add margin
    y_min -= 2.0;
    y_max += 2.0;

    // Round to integers
    let y_min = y_min.floor() as i32;
    let y_max = y_max.ceil() as i32;

    // Determine x range for plotting
    let mut x_min = x[0];
    let mut x_max = x[0];

    for &xi in x.iter() {
        if xi < x_min {
            x_min = xi;
        }
        if xi > x_max {
            x_max = xi;
        }
    }

    // Add margin
    x_min -= 1.0;
    x_max += 1.0;

    // Character symbols for each model
    let symbols = ['O', 'T', 'R', 'H', 'X']; // OLS, Theil-Sen, RANSAC, Huber, True

    // Draw the plot
    for y_val in (y_min..=y_max).rev() {
        let mut line = String::from("|");

        for x_pos in 0..60 {
            let x_val = x_min + (x_max - x_min) * (x_pos as f64 / 60.0);

            // Check if there's a data point
            let mut found_point = false;
            let mut is_outlier = false;

            for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
                if (xi - x_val).abs() < (x_max - x_min) / 60.0 * 0.8
                    && (yi - y_val as f64).abs() < 0.5
                {
                    found_point = true;
                    is_outlier = outliers.get(i).cloned().unwrap_or(false);
                    break;
                }
            }

            // Check regression lines
            let mut found_line = false;
            let mut line_char = ' ';

            for (i, &(_, slope, intercept)) in models.iter().enumerate() {
                let y_predicted = slope * x_val + intercept;
                if (y_predicted - y_val as f64).abs() < 0.5 {
                    if !found_line {
                        found_line = true;
                        line_char = symbols[i % symbols.len()];
                    } else {
                        // Multiple lines - use a different symbol
                        line_char = '+';
                    }
                }
            }

            // Draw the appropriate character
            if found_point {
                if is_outlier {
                    line.push('X'); // Outlier point
                } else {
                    line.push('●'); // Regular data point
                }
            } else if found_line {
                line.push(line_char);
            } else {
                line.push(' ');
            }
        }

        line.push('|');
        println!("{}", line);
    }

    println!("+{}+", "-".repeat(60));
    println!("Legend: ● Data points, X Outliers");
    for (i, name__) in models.iter().enumerate() {
        println!("        {} {:?}", symbols[i % symbols.len()], name__);
    }
    println!("        + Multiple lines overlap");
}

#[allow(dead_code)]
fn main() {
    println!("=== Robust Regression Comparison ===");

    // Example 1: Simple data with one obvious outlier
    println!("\n--- Example 1: Simple data with an obvious outlier ---");

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 30.0]; // Last point is outlier

    // True model: y = 2x + 1
    let true_slope = 2.0;
    let true_intercept = 1.0;

    // Run different regression methods
    let ols_result = simple_ols(&x, &y);
    let theil_result = simple_theilslopes(&x, &y);
    let ransac_result = simple_ransac(&x, &y, 2.0, 100);
    let huber_result = simple_huber(&x, &y, 1.345, 100);

    // Print results
    println!("True model:  y = {}x + {}", true_slope, true_intercept);
    println!(
        "OLS:         y = {:.4}x + {:.4}, R² = {:.4}",
        ols_result.slope, ols_result.intercept, ols_result.r_squared
    );
    println!(
        "Theil-Sen:   y = {:.4}x + {:.4}, 95% CI = [{:.4}, {:.4}]",
        theil_result.slope, theil_result.intercept, theil_result.slope_low, theil_result.slope_high
    );
    println!(
        "RANSAC:      y = {:.4}x + {:.4}, R² = {:.4}",
        ransac_result.slope, ransac_result.intercept, ransac_result.r_squared
    );
    println!(
        "Huber:       y = {:.4}x + {:.4}, R² = {:.4}",
        huber_result.slope, huber_result.intercept, huber_result.r_squared
    );

    // Calculate errors
    let ols_error = (ols_result.slope - true_slope).abs();
    let theil_error = (theil_result.slope - true_slope).abs();
    let ransac_error = (ransac_result.slope - true_slope).abs();
    let huber_error = (huber_result.slope - true_slope).abs();

    println!("\nSlope Error Comparison:");
    println!(
        "OLS:         {:.4} ({:.2}% error)",
        ols_error,
        100.0 * ols_error / true_slope
    );
    println!(
        "Theil-Sen:   {:.4} ({:.2}% error)",
        theil_error,
        100.0 * theil_error / true_slope
    );
    println!(
        "RANSAC:      {:.4} ({:.2}% error)",
        ransac_error,
        100.0 * ransac_error / true_slope
    );
    println!(
        "Huber:       {:.4} ({:.2}% error)",
        huber_error,
        100.0 * huber_error / true_slope
    );

    // Print outliers identified by RANSAC
    let outlier_indices: Vec<_> = ransac_result
        .inlier_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &is_inlier)| if !is_inlier { Some(i) } else { None })
        .collect();

    println!("\nOutliers identified by RANSAC: {:?}", outlier_indices);

    // Define outliers for visualization (based on RANSAC)
    let outliers = ransac_result
        .inlier_mask
        .iter()
        .map(|&x| !x)
        .collect::<Vec<_>>();

    // Create models for visualization
    let models = vec![
        ("OLS", ols_result.slope, ols_result.intercept),
        ("Theil-Sen", theil_result.slope, theil_result.intercept),
        ("RANSAC", ransac_result.slope, ransac_result.intercept),
        ("Huber", huber_result.slope, huber_result.intercept),
        ("True", true_slope, true_intercept),
    ];

    // Visualize the results
    visualize_regression_lines(&x, &y, &models, &outliers);

    // Example 2: Multiple outliers
    println!("\n\n--- Example 2: Data with multiple outliers ---");

    // Generate data with multiple outliers (predefined to avoid dependencies)
    let x2 = vec![
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
    ];
    let y2 = vec![
        3.2, 3.8, 4.5, 5.4, 5.9, 16.0, 7.7, 8.1, -1.0, 10.2, 10.9, 11.5, 22.0, 14.1, 2.0,
    ];
    let _true_outliers = [
        false, false, false, false, false, true, false, false, true, false, false, false, true,
        false, true,
    ];

    let true_slope2 = 1.5;
    let true_intercept2 = 3.0;

    // Run different regression methods
    let ols_result2 = simple_ols(&x2, &y2);
    let theil_result2 = simple_theilslopes(&x2, &y2);
    let ransac_result2 = simple_ransac(&x2, &y2, 3.0, 200);
    let huber_result2 = simple_huber(&x2, &y2, 1.345, 100);

    // Print results
    println!("True model:  y = {}x + {}", true_slope2, true_intercept2);
    println!(
        "OLS:         y = {:.4}x + {:.4}, R² = {:.4}",
        ols_result2.slope, ols_result2.intercept, ols_result2.r_squared
    );
    println!(
        "Theil-Sen:   y = {:.4}x + {:.4}, 95% CI = [{:.4}, {:.4}]",
        theil_result2.slope,
        theil_result2.intercept,
        theil_result2.slope_low,
        theil_result2.slope_high
    );
    println!(
        "RANSAC:      y = {:.4}x + {:.4}, R² = {:.4}",
        ransac_result2.slope, ransac_result2.intercept, ransac_result2.r_squared
    );
    println!(
        "Huber:       y = {:.4}x + {:.4}, R² = {:.4}",
        huber_result2.slope, huber_result2.intercept, huber_result2.r_squared
    );

    // Calculate errors
    let ols_error2 = (ols_result2.slope - true_slope2).abs();
    let theil_error2 = (theil_result2.slope - true_slope2).abs();
    let ransac_error2 = (ransac_result2.slope - true_slope2).abs();
    let huber_error2 = (huber_result2.slope - true_slope2).abs();

    println!("\nSlope Error Comparison:");
    println!(
        "OLS:         {:.4} ({:.2}% error)",
        ols_error2,
        100.0 * ols_error2 / true_slope2
    );
    println!(
        "Theil-Sen:   {:.4} ({:.2}% error)",
        theil_error2,
        100.0 * theil_error2 / true_slope2
    );
    println!(
        "RANSAC:      {:.4} ({:.2}% error)",
        ransac_error2,
        100.0 * ransac_error2 / true_slope2
    );
    println!(
        "Huber:       {:.4} ({:.2}% error)",
        huber_error2,
        100.0 * huber_error2 / true_slope2
    );

    // Print outliers identified by RANSAC
    let outlier_indices2: Vec<_> = ransac_result2
        .inlier_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &is_inlier)| if !is_inlier { Some(i) } else { None })
        .collect();

    println!("\nTrue outliers: [5, 8, 12, 15, 20]");
    println!("Outliers identified by RANSAC: {:?}", outlier_indices2);

    // Define outliers for visualization (based on RANSAC)
    let outliers2 = ransac_result2
        .inlier_mask
        .iter()
        .map(|&x| !x)
        .collect::<Vec<_>>();

    // Create models for visualization
    let models2 = vec![
        ("OLS", ols_result2.slope, ols_result2.intercept),
        ("Theil-Sen", theil_result2.slope, theil_result2.intercept),
        ("RANSAC", ransac_result2.slope, ransac_result2.intercept),
        ("Huber", huber_result2.slope, huber_result2.intercept),
        ("True", true_slope2, true_intercept2),
    ];

    // Visualize the results
    visualize_regression_lines(&x2, &y2, &models2, &outliers2);

    // Example 3: Leverage points (outliers in X direction)
    println!("\n\n--- Example 3: Leverage points (outliers in X direction) ---");

    // Generate data with predefined values (to avoid dependencies)
    let x3 = vec![
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 20.0, 22.0,
    ];
    let y3 = vec![
        2.1, 2.5, 2.7, 3.2, 3.6, 4.1, 4.4, 4.9, 5.3, 5.5, 6.0, 18.1, 10.0,
    ];
    let outliers3 = vec![
        false, false, false, false, false, false, false, false, false, false, false, false, true,
    ];

    let true_slope3 = 0.8;
    let true_intercept3 = 2.0;

    // Run different regression methods
    let ols_result3 = simple_ols(&x3, &y3);
    let theil_result3 = simple_theilslopes(&x3, &y3);
    let ransac_result3 = simple_ransac(&x3, &y3, 2.0, 200);
    let huber_result3 = simple_huber(&x3, &y3, 1.345, 100);

    // Print results
    println!("True model:  y = {}x + {}", true_slope3, true_intercept3);
    println!(
        "OLS:         y = {:.4}x + {:.4}, R² = {:.4}",
        ols_result3.slope, ols_result3.intercept, ols_result3.r_squared
    );
    println!(
        "Theil-Sen:   y = {:.4}x + {:.4}, 95% CI = [{:.4}, {:.4}]",
        theil_result3.slope,
        theil_result3.intercept,
        theil_result3.slope_low,
        theil_result3.slope_high
    );
    println!(
        "RANSAC:      y = {:.4}x + {:.4}, R² = {:.4}",
        ransac_result3.slope, ransac_result3.intercept, ransac_result3.r_squared
    );
    println!(
        "Huber:       y = {:.4}x + {:.4}, R² = {:.4}",
        huber_result3.slope, huber_result3.intercept, huber_result3.r_squared
    );

    // Calculate errors
    let ols_error3 = (ols_result3.slope - true_slope3).abs();
    let theil_error3 = (theil_result3.slope - true_slope3).abs();
    let ransac_error3 = (ransac_result3.slope - true_slope3).abs();
    let huber_error3 = (huber_result3.slope - true_slope3).abs();

    println!("\nSlope Error Comparison:");
    println!(
        "OLS:         {:.4} ({:.2}% error)",
        ols_error3,
        100.0 * ols_error3 / true_slope3
    );
    println!(
        "Theil-Sen:   {:.4} ({:.2}% error)",
        theil_error3,
        100.0 * theil_error3 / true_slope3
    );
    println!(
        "RANSAC:      {:.4} ({:.2}% error)",
        ransac_error3,
        100.0 * ransac_error3 / true_slope3
    );
    println!(
        "Huber:       {:.4} ({:.2}% error)",
        huber_error3,
        100.0 * huber_error3 / true_slope3
    );

    // Create models for visualization
    let models3 = vec![
        ("OLS", ols_result3.slope, ols_result3.intercept),
        ("Theil-Sen", theil_result3.slope, theil_result3.intercept),
        ("RANSAC", ransac_result3.slope, ransac_result3.intercept),
        ("Huber", huber_result3.slope, huber_result3.intercept),
        ("True", true_slope3, true_intercept3),
    ];

    // Visualize the results
    visualize_regression_lines(&x3, &y3, &models3, &outliers3);

    println!("\n=== Summary ===");
    println!("This example demonstrates the effectiveness of different robust regression methods:");
    println!();
    println!("1. Ordinary Least Squares (OLS):");
    println!("   - Performs well with clean data following normal assumptions");
    println!("   - Highly sensitive to outliers, especially leverage points");
    println!(
        "   - Mean error across examples: {:.2}%",
        (ols_error / true_slope + ols_error2 / true_slope2 + ols_error3 / true_slope3) * 100.0
            / 3.0
    );
    println!();
    println!("2. Theil-Sen Estimator:");
    println!("   - Highly robust to outliers (up to 29.3% contamination)");
    println!("   - Often performs best with leverage points (outliers in X direction)");
    println!("   - Provides confidence intervals for the slope parameter");
    println!(
        "   - Mean error across examples: {:.2}%",
        (theil_error / true_slope + theil_error2 / true_slope2 + theil_error3 / true_slope3)
            * 100.0
            / 3.0
    );
    println!();
    println!("3. RANSAC:");
    println!("   - Explicitly identifies and excludes outliers");
    println!("   - Very effective when there's a clear distinction between inliers and outliers");
    println!("   - May have results that vary due to its random sampling nature");
    println!(
        "   - Mean error across examples: {:.2}%",
        (ransac_error / true_slope + ransac_error2 / true_slope2 + ransac_error3 / true_slope3)
            * 100.0
            / 3.0
    );
    println!();
    println!("4. Huber Regression:");
    println!("   - Balances efficiency and robustness with a hybrid loss function");
    println!("   - Down-weights outliers rather than excluding them entirely");
    println!("   - More computationally efficient than Theil-Sen for larger datasets");
    println!(
        "   - Mean error across examples: {:.2}%",
        (huber_error / true_slope + huber_error2 / true_slope2 + huber_error3 / true_slope3)
            * 100.0
            / 3.0
    );
    println!();
    println!("Choosing the right robust regression method depends on your specific data and requirements:");
    println!("- For critical outlier identification: RANSAC");
    println!("- For high robustness with confidence intervals: Theil-Sen");
    println!("- For a balance of efficiency and robustness: Huber");
    println!("- For data with leverage points: Theil-Sen");
}
