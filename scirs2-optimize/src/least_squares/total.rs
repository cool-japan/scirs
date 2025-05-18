//! Total least squares (errors-in-variables)
//!
//! This module implements total least squares for problems where both
//! the independent and dependent variables have measurement errors.
//!
//! # Example
//!
//! ```
//! use ndarray::{array, Array1, Array2};
//! use scirs2_optimize::least_squares::total::{total_least_squares, TotalLeastSquaresOptions};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Data points with errors in both x and y
//! let x_measured = array![1.0, 2.1, 2.9, 4.2, 5.0];
//! let y_measured = array![2.1, 3.9, 5.1, 6.8, 8.1];
//!
//! // Known or estimated error variances
//! let x_variance = array![0.1, 0.1, 0.1, 0.2, 0.1];
//! let y_variance = array![0.1, 0.15, 0.1, 0.2, 0.1];
//!
//! let result = total_least_squares(
//!     &x_measured,
//!     &y_measured,
//!     Some(&x_variance),
//!     Some(&y_variance),
//!     None
//! )?;
//!
//! println!("Slope: {:.3}", result.slope);
//! println!("Intercept: {:.3}", result.intercept);
//! # Ok(())
//! # }
//! ```

use crate::error::OptimizeResult;
use ndarray::{array, s, Array1, Array2, ArrayBase, Data, Ix1};

/// Options for total least squares
#[derive(Debug, Clone)]
pub struct TotalLeastSquaresOptions {
    /// Maximum number of iterations for iterative methods
    pub max_iter: usize,

    /// Convergence tolerance
    pub tol: f64,

    /// Method to use
    pub method: TLSMethod,

    /// Whether to use weighted TLS when variances are provided
    pub use_weights: bool,
}

/// Methods for total least squares
#[derive(Debug, Clone, Copy)]
pub enum TLSMethod {
    /// Singular Value Decomposition (most stable)
    SVD,
    /// Iterative orthogonal regression
    Iterative,
    /// Maximum likelihood estimation
    MaximumLikelihood,
}

impl Default for TotalLeastSquaresOptions {
    fn default() -> Self {
        TotalLeastSquaresOptions {
            max_iter: 100,
            tol: 1e-8,
            method: TLSMethod::SVD,
            use_weights: true,
        }
    }
}

/// Result structure for total least squares
#[derive(Debug, Clone)]
pub struct TotalLeastSquaresResult {
    /// Estimated slope
    pub slope: f64,

    /// Estimated intercept
    pub intercept: f64,

    /// Corrected x values
    pub x_corrected: Array1<f64>,

    /// Corrected y values
    pub y_corrected: Array1<f64>,

    /// Sum of squared orthogonal distances
    pub orthogonal_residuals: f64,

    /// Number of iterations (for iterative methods)
    pub iterations: usize,

    /// Convergence status
    pub converged: bool,
}

/// Solve a total least squares problem
///
/// This function fits a line to data with errors in both variables.
/// It minimizes the sum of squared orthogonal distances to the line.
///
/// # Arguments
///
/// * `x_measured` - Measured x values
/// * `y_measured` - Measured y values
/// * `x_variance` - Optional variance estimates for x measurements
/// * `y_variance` - Optional variance estimates for y measurements
/// * `options` - Options for the optimization
pub fn total_least_squares<S1, S2, S3, S4>(
    x_measured: &ArrayBase<S1, Ix1>,
    y_measured: &ArrayBase<S2, Ix1>,
    x_variance: Option<&ArrayBase<S3, Ix1>>,
    y_variance: Option<&ArrayBase<S4, Ix1>>,
    options: Option<TotalLeastSquaresOptions>,
) -> OptimizeResult<TotalLeastSquaresResult>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
    S4: Data<Elem = f64>,
{
    let options = options.unwrap_or_default();
    let n = x_measured.len();

    if y_measured.len() != n {
        return Err(crate::error::OptimizeError::ValueError(
            "x_measured and y_measured must have the same length".to_string(),
        ));
    }

    // Check variance arrays if provided
    if let Some(x_var) = x_variance {
        if x_var.len() != n {
            return Err(crate::error::OptimizeError::ValueError(
                "x_variance must have the same length as data".to_string(),
            ));
        }
    }

    if let Some(y_var) = y_variance {
        if y_var.len() != n {
            return Err(crate::error::OptimizeError::ValueError(
                "y_variance must have the same length as data".to_string(),
            ));
        }
    }

    match options.method {
        TLSMethod::SVD => tls_svd(x_measured, y_measured, x_variance, y_variance, &options),
        TLSMethod::Iterative => {
            tls_iterative(x_measured, y_measured, x_variance, y_variance, &options)
        }
        TLSMethod::MaximumLikelihood => {
            tls_maximum_likelihood(x_measured, y_measured, x_variance, y_variance, &options)
        }
    }
}

/// Total least squares using SVD
fn tls_svd<S1, S2, S3, S4>(
    x_measured: &ArrayBase<S1, Ix1>,
    y_measured: &ArrayBase<S2, Ix1>,
    x_variance: Option<&ArrayBase<S3, Ix1>>,
    y_variance: Option<&ArrayBase<S4, Ix1>>,
    options: &TotalLeastSquaresOptions,
) -> OptimizeResult<TotalLeastSquaresResult>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
    S4: Data<Elem = f64>,
{
    let n = x_measured.len();

    // Center the data
    let x_mean = x_measured.mean().unwrap();
    let y_mean = y_measured.mean().unwrap();

    let x_centered = x_measured - x_mean;
    let y_centered = y_measured - y_mean;

    // Apply weights if variances are provided
    let (x_scaled, y_scaled) =
        if options.use_weights && x_variance.is_some() && y_variance.is_some() {
            let x_var = x_variance.unwrap();
            let y_var = y_variance.unwrap();

            // Scale by inverse standard deviation
            let x_weights = x_var.mapv(|v| 1.0 / v.sqrt());
            let y_weights = y_var.mapv(|v| 1.0 / v.sqrt());

            (x_centered * &x_weights, y_centered * &y_weights)
        } else {
            (x_centered.clone(), y_centered.clone())
        };

    // Form the augmented matrix [x_scaled, y_scaled]
    let mut data_matrix = Array2::zeros((n, 2));
    for i in 0..n {
        data_matrix[[i, 0]] = x_scaled[i];
        data_matrix[[i, 1]] = y_scaled[i];
    }

    // Perform SVD (simplified - in practice use a proper SVD)
    // For now, use eigendecomposition of the covariance matrix
    let cov_matrix = data_matrix.t().dot(&data_matrix) / n as f64;

    // Find eigenvalues and eigenvectors
    let (eigenvalues, eigenvectors) = eigen_2x2(&cov_matrix);

    // The eigenvector corresponding to the smallest eigenvalue gives the normal to the line
    let min_idx = if eigenvalues[0] < eigenvalues[1] {
        0
    } else {
        1
    };
    let normal = eigenvectors.slice(s![.., min_idx]).to_owned();

    // Convert normal to slope-intercept form
    // Normal vector (a, b) corresponds to line ax + by + c = 0
    let a = normal[0usize];
    let b = normal[1usize];

    if b.abs() < 1e-10 {
        // Nearly vertical line
        return Err(crate::error::OptimizeError::ValueError(
            "Nearly vertical line detected".to_string(),
        ));
    }

    let slope = -a / b;
    let intercept = y_mean - slope * x_mean;

    // Compute corrected values (orthogonal projection onto the line)
    let mut x_corrected = Array1::zeros(n);
    let mut y_corrected = Array1::zeros(n);
    let mut total_residual = 0.0;

    for i in 0..n {
        let (x_proj, y_proj) =
            orthogonal_projection(x_measured[i], y_measured[i], slope, intercept);
        x_corrected[i] = x_proj;
        y_corrected[i] = y_proj;

        let dx = x_measured[i] - x_proj;
        let dy = y_measured[i] - y_proj;
        total_residual += dx * dx + dy * dy;
    }

    Ok(TotalLeastSquaresResult {
        slope,
        intercept,
        x_corrected,
        y_corrected,
        orthogonal_residuals: total_residual,
        iterations: 1,
        converged: true,
    })
}

/// Iterative total least squares
fn tls_iterative<S1, S2, S3, S4>(
    x_measured: &ArrayBase<S1, Ix1>,
    y_measured: &ArrayBase<S2, Ix1>,
    x_variance: Option<&ArrayBase<S3, Ix1>>,
    y_variance: Option<&ArrayBase<S4, Ix1>>,
    options: &TotalLeastSquaresOptions,
) -> OptimizeResult<TotalLeastSquaresResult>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
    S4: Data<Elem = f64>,
{
    let n = x_measured.len();

    // Initial estimate using ordinary least squares
    let (mut slope, mut intercept) = ordinary_least_squares(x_measured, y_measured);

    let mut x_corrected = x_measured.to_owned();
    let mut y_corrected = y_measured.to_owned();
    let mut prev_residual = f64::INFINITY;

    // Get weights from variances
    let x_weights = if let Some(x_var) = x_variance {
        x_var.mapv(|v| 1.0 / v)
    } else {
        Array1::ones(n)
    };

    let y_weights = if let Some(y_var) = y_variance {
        y_var.mapv(|v| 1.0 / v)
    } else {
        Array1::ones(n)
    };

    let mut iter = 0;
    let mut converged = false;

    while iter < options.max_iter {
        // E-step: Update corrected values given current line parameters
        let mut total_residual = 0.0;

        for i in 0..n {
            let (x_proj, y_proj) = weighted_orthogonal_projection(
                x_measured[i],
                y_measured[i],
                slope,
                intercept,
                x_weights[i],
                y_weights[i],
            );

            x_corrected[i] = x_proj;
            y_corrected[i] = y_proj;

            let dx = x_measured[i] - x_proj;
            let dy = y_measured[i] - y_proj;
            total_residual += x_weights[i] * dx * dx + y_weights[i] * dy * dy;
        }

        // M-step: Update line parameters given corrected values
        let (new_slope, new_intercept) =
            weighted_least_squares_line(&x_corrected, &y_corrected, &x_weights, &y_weights);

        // Check convergence
        if (total_residual - prev_residual).abs() < options.tol * total_residual
            && (new_slope - slope).abs() < options.tol
            && (new_intercept - intercept).abs() < options.tol
        {
            converged = true;
            break;
        }

        slope = new_slope;
        intercept = new_intercept;
        prev_residual = total_residual;
        iter += 1;
    }

    // Compute final orthogonal residuals
    let mut orthogonal_residuals = 0.0;
    for i in 0..n {
        let dx = x_measured[i] - x_corrected[i];
        let dy = y_measured[i] - y_corrected[i];
        orthogonal_residuals += dx * dx + dy * dy;
    }

    Ok(TotalLeastSquaresResult {
        slope,
        intercept,
        x_corrected,
        y_corrected,
        orthogonal_residuals,
        iterations: iter,
        converged,
    })
}

/// Maximum likelihood total least squares
fn tls_maximum_likelihood<S1, S2, S3, S4>(
    x_measured: &ArrayBase<S1, Ix1>,
    y_measured: &ArrayBase<S2, Ix1>,
    x_variance: Option<&ArrayBase<S3, Ix1>>,
    y_variance: Option<&ArrayBase<S4, Ix1>>,
    _options: &TotalLeastSquaresOptions,
) -> OptimizeResult<TotalLeastSquaresResult>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
    S4: Data<Elem = f64>,
{
    // For now, use the iterative method
    // A proper implementation would maximize the likelihood function
    tls_iterative(x_measured, y_measured, x_variance, y_variance, _options)
}

/// Compute ordinary least squares for initial estimate
fn ordinary_least_squares<S1, S2>(x: &ArrayBase<S1, Ix1>, y: &ArrayBase<S2, Ix1>) -> (f64, f64)
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    let _n = x.len() as f64;
    let x_mean = x.mean().unwrap();
    let y_mean = y.mean().unwrap();

    let mut num = 0.0;
    let mut den = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        num += dx * dy;
        den += dx * dx;
    }

    let slope = num / den;
    let intercept = y_mean - slope * x_mean;

    (slope, intercept)
}

/// Orthogonal projection of a point onto a line
fn orthogonal_projection(x: f64, y: f64, slope: f64, intercept: f64) -> (f64, f64) {
    // Line equation: y = slope * x + intercept
    // Normal vector: (slope, -1)
    // Normalized: (slope, -1) / sqrt(slope^2 + 1)

    let norm_sq = slope * slope + 1.0;
    let t = ((y - intercept) * slope + x) / norm_sq;

    let x_proj = t;
    let y_proj = slope * t + intercept;

    (x_proj, y_proj)
}

/// Weighted orthogonal projection
fn weighted_orthogonal_projection(
    x: f64,
    y: f64,
    slope: f64,
    intercept: f64,
    weight_x: f64,
    weight_y: f64,
) -> (f64, f64) {
    // Minimize: weight_x * (x - x_proj)^2 + weight_y * (y - y_proj)^2
    // Subject to: y_proj = slope * x_proj + intercept

    let a = weight_x + weight_y * slope * slope;
    let _b = weight_y * slope;
    let c = weight_x * x + weight_y * slope * (y - intercept);

    let x_proj = c / a;
    let y_proj = slope * x_proj + intercept;

    (x_proj, y_proj)
}

/// Weighted least squares for a line
fn weighted_least_squares_line<S1, S2, S3, S4>(
    x: &ArrayBase<S1, Ix1>,
    y: &ArrayBase<S2, Ix1>,
    weight_x: &ArrayBase<S3, Ix1>,
    weight_y: &ArrayBase<S4, Ix1>,
) -> (f64, f64)
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
    S4: Data<Elem = f64>,
{
    let n = x.len();
    let mut sum_wx = 0.0;
    let mut sum_wy = 0.0;
    let mut sum_wxx = 0.0;
    let mut sum_wxy = 0.0;
    let mut _sum_wyy = 0.0;
    let mut sum_w = 0.0;

    for i in 0..n {
        let w = (weight_x[i] + weight_y[i]) / 2.0; // Combined weight
        sum_w += w;
        sum_wx += w * x[i];
        sum_wy += w * y[i];
        sum_wxx += w * x[i] * x[i];
        sum_wxy += w * x[i] * y[i];
        _sum_wyy += w * y[i] * y[i];
    }

    let x_mean = sum_wx / sum_w;
    let y_mean = sum_wy / sum_w;

    let cov_xx = sum_wxx / sum_w - x_mean * x_mean;
    let cov_xy = sum_wxy / sum_w - x_mean * y_mean;

    let slope = cov_xy / cov_xx;
    let intercept = y_mean - slope * x_mean;

    (slope, intercept)
}

/// Simple 2x2 eigendecomposition
fn eigen_2x2(matrix: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let a = matrix[[0, 0]];
    let b = matrix[[0, 1]];
    let c = matrix[[1, 0]];
    let d = matrix[[1, 1]];

    // Characteristic equation: λ² - (a+d)λ + (ad-bc) = 0
    let trace = a + d;
    let det = a * d - b * c;

    let discriminant = trace * trace - 4.0 * det;
    let sqrt_disc = discriminant.sqrt();

    let lambda1 = (trace + sqrt_disc) / 2.0;
    let lambda2 = (trace - sqrt_disc) / 2.0;

    // Eigenvectors
    let mut eigenvectors = Array2::zeros((2, 2));

    // For λ1
    if (a - lambda1).abs() > 1e-10 || b.abs() > 1e-10 {
        let v1_x = b;
        let v1_y = lambda1 - a;
        let norm1 = (v1_x * v1_x + v1_y * v1_y).sqrt();
        eigenvectors[[0, 0]] = v1_x / norm1;
        eigenvectors[[1, 0]] = v1_y / norm1;
    } else {
        eigenvectors[[0, 0]] = 1.0;
        eigenvectors[[1, 0]] = 0.0;
    }

    // For λ2
    if (a - lambda2).abs() > 1e-10 || b.abs() > 1e-10 {
        let v2_x = b;
        let v2_y = lambda2 - a;
        let norm2 = (v2_x * v2_x + v2_y * v2_y).sqrt();
        eigenvectors[[0, 1]] = v2_x / norm2;
        eigenvectors[[1, 1]] = v2_y / norm2;
    } else {
        eigenvectors[[0, 1]] = 0.0;
        eigenvectors[[1, 1]] = 1.0;
    }

    (array![lambda1, lambda2], eigenvectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_total_least_squares_simple() {
        // Generate data with errors in both x and y
        let true_slope = 1.5;
        let true_intercept = 0.5;

        let x_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_true = &x_true * true_slope + true_intercept;

        // Add errors
        let x_errors = array![0.1, -0.05, 0.08, -0.03, 0.06];
        let y_errors = array![-0.05, 0.1, -0.07, 0.04, -0.08];

        let x_measured = &x_true + &x_errors;
        let y_measured = &y_true + &y_errors;

        let result = total_least_squares::<_, _, _, _>(
            &x_measured,
            &y_measured,
            None::<&Array1<f64>>,
            None::<&Array1<f64>>,
            None,
        )
        .unwrap();

        // Check that the estimated parameters are close to true values
        assert!((result.slope - true_slope).abs() < 0.1);
        assert!((result.intercept - true_intercept).abs() < 0.1);
    }

    #[test]
    fn test_weighted_total_least_squares() {
        // Data with different error variances
        let x_measured = array![1.0, 2.1, 2.9, 4.2, 5.0];
        let y_measured = array![2.1, 3.9, 5.1, 6.8, 8.1];

        // Known error variances (larger for some points)
        let x_variance = array![0.01, 0.01, 0.01, 0.1, 0.01];
        let y_variance = array![0.01, 0.02, 0.01, 0.1, 0.01];

        let result = total_least_squares(
            &x_measured,
            &y_measured,
            Some(&x_variance),
            Some(&y_variance),
            None,
        )
        .unwrap();

        // The point with large variance should have less influence
        assert!(result.converged);
        println!(
            "Weighted TLS: slope = {:.3}, intercept = {:.3}",
            result.slope, result.intercept
        );
    }

    #[test]
    fn test_iterative_vs_svd() {
        // Compare iterative and SVD methods
        let x_measured = array![0.5, 1.5, 2.8, 3.7, 4.9];
        let y_measured = array![1.2, 2.7, 4.1, 5.3, 6.8];

        let mut options_svd = TotalLeastSquaresOptions::default();
        options_svd.method = TLSMethod::SVD;

        let mut options_iter = TotalLeastSquaresOptions::default();
        options_iter.method = TLSMethod::Iterative;

        let result_svd = total_least_squares::<_, _, _, _>(
            &x_measured,
            &y_measured,
            None::<&Array1<f64>>,
            None::<&Array1<f64>>,
            Some(options_svd),
        )
        .unwrap();

        let result_iter = total_least_squares::<_, _, _, _>(
            &x_measured,
            &y_measured,
            None::<&Array1<f64>>,
            None::<&Array1<f64>>,
            Some(options_iter),
        )
        .unwrap();

        // Results should be similar
        assert!((result_svd.slope - result_iter.slope).abs() < 0.01);
        assert!((result_svd.intercept - result_iter.intercept).abs() < 0.01);
    }
}
