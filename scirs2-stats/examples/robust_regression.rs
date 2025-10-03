// This is a standalone example that demonstrates robust regression concepts
// without requiring the main library implementation
use scirs2_core::numeric::Float;

/// A simplified version of TheilSlopes for demonstration
struct SimpleTheilSlopes {
    slope: f64,
    intercept: f64,
}

/// Simple implementation of Theil-Sen estimator
#[allow(dead_code)]
fn simple_theilslopes(x: &[f64], y: &[f64]) -> SimpleTheilSlopes {
    assert_eq!(x.len(), y.len(), "x and y must have same length");

    let n = x.len();
    let mut slopes = Vec::with_capacity(n * (n - 1) / 2);

    // Calculate all pairwise slopes
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[j] - x[i];
            if dx.abs() > f64::epsilon() {
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

    SimpleTheilSlopes { slope, intercept }
}

/// Simple OLS (Ordinary Least Squares) implementation
#[allow(dead_code)]
fn simple_ols(x: &[f64], y: &[f64]) -> (f64, f64) {
    // (slope, intercept)
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

    (slope, intercept)
}

/// Simple implementation of RANSAC (Random Sample Consensus) regression
#[allow(dead_code)]
fn simple_ransac(x: &[f64], y: &[f64], threshold: f64, ntrials: usize) -> (f64, f64, Vec<bool>) {
    assert_eq!(x.len(), y.len(), "x and y must have same length");

    let n = x.len();
    if n < 2 {
        panic!("Need at least 2 points for RANSAC");
    }

    // To keep code simple, we always sample 2 points
    let min_samples = 2;

    // Use a simple random number generator
    let mut rng = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let mut best_inlier_count = 0;
    let mut best_model = (0.0, 0.0); // (slope, intercept)
    let mut best_inlier_mask = vec![false; n];

    for _ in 0..ntrials {
        // Select 2 random points
        let idx1 = (rng % n as u64) as usize;
        rng = (rng * 1103515245 + 12345) % 2147483648; // Simple LCG
        let idx2 = (rng % n as u64) as usize;
        rng = (rng * 1103515245 + 12345) % 2147483648;

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
        if dx.abs() < f64::epsilon() {
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

    // Refit the model using all inliers
    if best_inlier_count >= min_samples {
        // We could refit, but for simplicity we'll just return the best model
        (best_model.0, best_model.1, best_inlier_mask)
    } else {
        // Fallback to using all data if we couldn't find a good model
        let (slope, intercept) = simple_ols(x, y);
        (slope, intercept, vec![true; n])
    }
}

/// A simplified implementation of Huber regression
#[allow(dead_code)]
fn simple_huber(x: &[f64], y: &[f64], epsilon: f64, maxiter: usize) -> (f64, f64) {
    assert_eq!(x.len(), y.len(), "x and y must have same length");

    // Start with OLS estimate
    let (mut slope, mut intercept) = simple_ols(x, y);

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

    (slope, intercept)
}

#[allow(dead_code)]
fn main() {
    println!("=== Robust Regression Methods Demonstration ===");

    // Example 1: Simple data with one outlier
    println!("\n--- Example 1: Simple data with one outlier ---");
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 30.0]; // Last point is an outlier

    // True model: y = 2x + 1
    let true_slope = 2.0;
    let true_intercept = 1.0;

    // Calculate using different methods
    let (ols_slope, ols_intercept) = simple_ols(&x, &y);
    let theil = simple_theilslopes(&x, &y);
    let (ransac_slope, ransac_intercept, ransac_inliers) = simple_ransac(&x, &y, 2.0, 100);
    let (huber_slope, huber_intercept) = simple_huber(&x, &y, 1.345, 50);

    // Print results
    println!("True model: y = {}x + {}", true_slope, true_intercept);
    println!("OLS:        y = {:.4}x + {:.4}", ols_slope, ols_intercept);
    println!(
        "Theil-Sen:  y = {:.4}x + {:.4}",
        theil.slope, theil.intercept
    );
    println!(
        "RANSAC:     y = {:.4}x + {:.4}",
        ransac_slope, ransac_intercept
    );
    println!(
        "Huber:      y = {:.4}x + {:.4}",
        huber_slope, huber_intercept
    );

    // Calculate errors
    let ols_slope_error = (ols_slope - true_slope).abs();
    let theil_slope_error = (theil.slope - true_slope).abs();
    let ransac_slope_error = (ransac_slope - true_slope).abs();
    let huber_slope_error = (huber_slope - true_slope).abs();

    println!("\nSlope Error Comparison:");
    println!("OLS:        {:.4}", ols_slope_error);
    println!("Theil-Sen:  {:.4}", theil_slope_error);
    println!("RANSAC:     {:.4}", ransac_slope_error);
    println!("Huber:      {:.4}", huber_slope_error);

    // Print outliers identified by RANSAC
    let outlier_indices: Vec<_> = ransac_inliers
        .iter()
        .enumerate()
        .filter_map(|(i, &is_inlier)| if !is_inlier { Some(i) } else { None })
        .collect();

    println!("\nOutliers identified by RANSAC: {:?}", outlier_indices);

    // Draw a simple text-based scatter plot
    println!("\nVisualization of regression lines:");
    for i in (0..20).rev() {
        let y_val = i as f64 + 0.5;
        let mut line = String::from("|");

        for j in 0..40 {
            let x_val = j as f64 / 3.0;

            // Check if there's a data point
            let mut isdata = false;
            let mut is_outlier = false;

            for k in 0..x.len() {
                if (x[k] - x_val).abs() < 0.2 && (y[k] - y_val).abs() < 0.5 {
                    isdata = true;
                    if k == x.len() - 1 {
                        // Last point is our outlier
                        is_outlier = true;
                    }
                    break;
                }
            }

            // Check if it's near a regression line
            let ols_y = ols_intercept + ols_slope * x_val;
            let theil_y = theil.intercept + theil.slope * x_val;
            let ransac_y = ransac_intercept + ransac_slope * x_val;
            let huber_y = huber_intercept + huber_slope * x_val;

            let near_ols = (ols_y - y_val).abs() < 0.3;
            let near_theil = (theil_y - y_val).abs() < 0.3;
            let near_ransac = (ransac_y - y_val).abs() < 0.3;
            let near_huber = (huber_y - y_val).abs() < 0.3;

            if isdata {
                if is_outlier {
                    line.push('X'); // Outlier
                } else {
                    line.push('●'); // Data point
                }
            } else if near_ols && near_theil && near_ransac && near_huber {
                line.push('+'); // All lines overlap
            } else if near_ols {
                line.push('o'); // OLS line
            } else if near_theil {
                line.push('t'); // Theil-Sen line
            } else if near_ransac {
                line.push('r'); // RANSAC line
            } else if near_huber {
                line.push('h'); // Huber line
            } else {
                line.push(' ');
            }
        }

        line.push('|');
        println!("{}", line);
    }

    println!("+----------------------------------------+");
    println!("Legend: ● Data points, X Outliers");
    println!("        o OLS, t Theil-Sen, r RANSAC, h Huber");

    println!("\nConclusion:");
    println!("This demonstrates that robust regression methods (Theil-Sen, RANSAC, Huber)");
    println!("perform significantly better than OLS when data contains outliers.");
    println!("The primary benefits of each method are:");
    println!(
        " - Theil-Sen: Non-parametric, works well for small datasets with up to ~30% outliers"
    );
    println!(
        " - RANSAC: Explicitly identifies outliers, good when proportion of outliers is known"
    );
    println!(" - Huber: Provides a smooth transition between L1 and L2 loss functions");
}
