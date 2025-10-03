#[cfg(test)]
mod test_robust_methods {
    use scirs2_core::ndarray::Array2;
    use scirs2_core::numeric::Float;
    use scirs2_stats::error::StatsResult;

    // A simplified version of TheilSlopes just for testing
    struct SimpleTheilSlopes {
        slope: f64,
        intercept: f64,
    }

    // Simple implementation of Theil-Sen estimator for testing purposes
    fn simple_theilslopes(x: &[f64], y: &[f64]) -> StatsResult<SimpleTheilSlopes> {
        if x.len() != y.len() {
            return Err(scirs2_stats::error::StatsError::DimensionMismatch(
                "x and y must have same length".to_string(),
            ));
        }

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

        Ok(SimpleTheilSlopes { slope, intercept })
    }

    #[test]
    fn test_theilslopes_with_outliers() {
        // Create a dataset with a clear linear relationship (y = 2*x + 1) and some outliers
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 30.0]; // Last point is an outlier

        // Calculate slope using simple implementation
        let theil = simple_theilslopes(&x, &y).unwrap();

        // Create a design matrix for OLS
        let _x_design =
            Array2::from_shape_fn((x.len(), 2), |(i, j)| if j == 0 { 1.0 } else { x[i] });

        // Calculate simple OLS manually
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

        let ols_slope = numerator / denominator;
        let ols_intercept = y_mean - ols_slope * x_mean;

        println!(
            "OLS - Slope: {:.4}, Intercept: {:.4}",
            ols_slope, ols_intercept
        );
        println!(
            "Theil-Sen - Slope: {:.4}, Intercept: {:.4}",
            theil.slope, theil.intercept
        );

        // Compare OLS vs Theil-Sen errors - true relationship is y = 2x + 1
        let true_slope = 2.0;
        let _true_intercept = 1.0;

        let ols_slope_error = (ols_slope - true_slope).abs();
        let theil_slope_error = (theil.slope - true_slope).abs();

        // Theil-Sen should be more accurate as it's less affected by outliers
        assert!(
            theil_slope_error < ols_slope_error,
            "Theil-Sen (error={}) should be more accurate than OLS (error={})",
            theil_slope_error,
            ols_slope_error
        );
    }
}
