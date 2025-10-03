//! Thin-plate spline implementation
//!
//! Thin-plate splines (TPS) are a generalization of cubic splines to
//! higher dimensions. They represent a special case of radial basis
//! function interpolation, designed to minimize a specific bending energy
//! measure.
//!
//! TPS are particularly useful for 2D and 3D interpolation and warping problems.

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

/// Thin-plate spline interpolator for scattered data
///
/// The thin-plate spline minimizes the energy function:
/// E(f) = ∑(f(x_i) - y_i)² + λ * ∫∫ [(∂²f/∂x²)² + 2(∂²f/∂x∂y)² + (∂²f/∂y²)²] dx dy
///
/// where the first term ensures the interpolation accuracy and the second term
/// controls the smoothness of the interpolation.
#[derive(Debug, Clone)]
pub struct ThinPlateSpline<T>
where
    T: Float + FromPrimitive + Debug,
{
    /// Centers (data points) for the basis functions
    centers: Array2<T>,
    /// Coefficients for the RBF terms
    coeffs: Array1<T>,
    /// Coefficients for the polynomial terms (a + b*x + c*y + ...)
    poly_coeffs: Array1<T>,
    /// Smoothing parameter (lambda), 0 = exact interpolation
    smoothing: T,
    /// Pre-computed values of the basis function
    basis_values: Option<Array2<T>>,
}

impl<T> ThinPlateSpline<T>
where
    T: Float + FromPrimitive + Debug,
{
    /// Get the pre-computed basis values
    pub fn basis_values(&self) -> Option<&Array2<T>> {
        self.basis_values.as_ref()
    }
}

impl<T> ThinPlateSpline<T>
where
    T: Float + FromPrimitive + Debug,
{
    /// Create a new thin-plate spline
    ///
    /// # Arguments
    ///
    /// * `x` - Input coordinates of shape (n_points, n_dims)
    /// * `y` - Target values of shape (n_points,)
    /// * `smoothing` - Smoothing parameter (regularization)
    ///
    /// # Returns
    ///
    /// A new `ThinPlateSpline` interpolator
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "linalg")]
    /// # {
    /// use scirs2_core::ndarray::{array, Array2};
    /// use scirs2_interpolate::advanced::thinplate::ThinPlateSpline;
    ///
    /// // Create 2D scattered data
    /// let points = Array2::from_shape_vec((4, 2), vec![
    ///     0.0, 0.0,
    ///     0.0, 1.0,
    ///     1.0, 0.0,
    ///     1.0, 1.0,
    /// ]).unwrap();
    ///
    /// // Function values: f(x,y) = x^2 + y^2
    /// let values = array![0.0, 1.0, 1.0, 2.0];
    ///
    /// // Create the interpolator
    /// let tps = ThinPlateSpline::<f64>::new(&points.view(), &values.view(), 0.0).unwrap();
    ///
    /// // Interpolate at a new point
    /// let new_point = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap();
    /// let result = tps.evaluate(&new_point.view()).unwrap();
    /// // The thin-plate spline interpolation may not match exactly due to numerical issues
    /// // For this example, we check that the result is reasonable (between 0 and 2)
    /// assert!(result[0] > 0.0 && result[0] < 2.0, "Result should be reasonable");
    /// # }
    /// ```
    pub fn new(x: &ArrayView2<T>, y: &ArrayView1<T>, smoothing: T) -> InterpolateResult<Self> {
        // Validate inputs
        if x.nrows() != y.len() {
            return Err(InterpolateError::invalid_input(
                "number of points must match number of values".to_string(),
            ));
        }

        if smoothing < T::zero() {
            return Err(InterpolateError::invalid_input(
                "smoothing parameter must be non-negative".to_string(),
            ));
        }

        // Get dimensions
        let n_points = x.nrows();
        let n_dims = x.ncols();

        // Need at least n_dims + 1 points for a unique solution (polynomial terms)
        if n_points < n_dims + 1 {
            return Err(InterpolateError::invalid_input(format!(
                "need at least {} points for {} dimensions",
                n_dims + 1,
                n_dims
            )));
        }

        // We need to solve a linear system to get the coefficients
        // The system includes both RBF terms and polynomial terms for the
        // TPS (affine transformation)

        // For 2D, the polynomial terms are 1, x, y
        // For 3D, they are 1, x, y, z
        // In general, 1 + n_dims terms

        let poly_terms = 1 + n_dims;

        // Compute the full system matrix
        // [K P; P^T 0]
        // where K is the kernel matrix, and P contains the polynomial terms

        // First, compute the kernel matrix K (n_points x n_points)
        let mut k = Array2::zeros((n_points, n_points));

        for i in 0..n_points {
            for j in 0..n_points {
                if i == j {
                    k[(i, j)] = smoothing; // Add regularization on the diagonal
                } else {
                    // Compute squared distance
                    let mut dist_sq = T::zero();
                    for d in 0..n_dims {
                        let diff = x[(i, d)] - x[(j, d)];
                        dist_sq = dist_sq + diff * diff;
                    }

                    // Apply the thin-plate kernel
                    k[(i, j)] = tps_kernel(dist_sq.sqrt());
                }
            }
        }

        // Second, create the polynomial matrix P (n_points x poly_terms)
        let mut p = Array2::zeros((n_points, poly_terms));

        for i in 0..n_points {
            // First term is always 1
            p[(i, 0)] = T::one();

            // Remaining terms are the coordinates
            for d in 0..n_dims {
                p[(i, d + 1)] = x[(i, d)];
            }
        }

        // Construct the full system matrix
        // [K P; P^T 0]
        let mut a = Array2::zeros((n_points + poly_terms, n_points + poly_terms));

        // Fill in K (top-left)
        for i in 0..n_points {
            for j in 0..n_points {
                a[(i, j)] = k[(i, j)];
            }
        }

        // Fill in P (top-right) and P^T (bottom-left)
        for i in 0..n_points {
            for j in 0..poly_terms {
                a[(i, n_points + j)] = p[(i, j)];
                a[(n_points + j, i)] = p[(i, j)];
            }
        }

        // The bottom-right is already zeros

        // Construct the right-hand side
        // [y; 0]
        let mut b = Array1::zeros(n_points + poly_terms);
        for i in 0..n_points {
            b[i] = y[i];
        }

        // Solve the system using scirs2-linalg
        let coeffs_full = {
            use scirs2_linalg::solve;
            let a_f64 = a.mapv(|x| x.to_f64().unwrap());
            let b_f64 = b.mapv(|x| x.to_f64().unwrap());
            match solve(&a_f64.view(), &b_f64.view(), None) {
                Ok(solution) => solution.mapv(|x| T::from_f64(x).unwrap()),
                Err(_) => {
                    // If the system is singular or near-singular, try SVD-based solution
                    use scirs2_linalg::lstsq;
                    match lstsq(&a_f64.view(), &b_f64.view(), None) {
                        Ok(result) => result.x.mapv(|x| T::from_f64(x).unwrap()),
                        Err(_) => {
                            return Err(InterpolateError::LinalgError(
                                "failed to solve linear system".to_string(),
                            ));
                        }
                    }
                }
            }
        };

        // Extract coefficients
        let coeffs = coeffs_full.slice(s![0..n_points]).to_owned();
        let poly_coeffs = coeffs_full.slice(s![n_points..]).to_owned();

        Ok(ThinPlateSpline {
            centers: x.to_owned(),
            coeffs,
            poly_coeffs,
            smoothing,
            basis_values: None,
        })
    }

    /// Evaluate the thin-plate spline at given points
    ///
    /// # Arguments
    ///
    /// * `x` - Points at which to evaluate, shape (n_eval, n_dims)
    ///
    /// # Returns
    ///
    /// Array of interpolated values, shape (n_eval,)
    pub fn evaluate(&self, x: &ArrayView2<T>) -> InterpolateResult<Array1<T>> {
        // Validate input
        if x.ncols() != self.centers.ncols() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "expected {} dimensions, got {}",
                self.centers.ncols(),
                x.ncols()
            )));
        }

        let n_eval = x.nrows();
        let n_centers = self.centers.nrows();
        let n_dims = self.centers.ncols();

        // Allocate result array
        let mut result = Array1::zeros(n_eval);

        // For each evaluation point
        for i in 0..n_eval {
            // Compute the RBF contribution first
            for j in 0..n_centers {
                // Compute the distance
                let mut dist_sq = T::zero();
                for d in 0..n_dims {
                    let diff = x[(i, d)] - self.centers[(j, d)];
                    dist_sq = dist_sq + diff * diff;
                }

                // Apply the kernel and add the contribution
                let kernel_value = tps_kernel(dist_sq.sqrt());
                result[i] = result[i] + self.coeffs[j] * kernel_value;
            }

            // Add the polynomial terms (affine transformation)
            result[i] = result[i] + self.poly_coeffs[0]; // Constant term

            // Linear terms
            for d in 0..n_dims {
                result[i] = result[i] + self.poly_coeffs[d + 1] * x[(i, d)];
            }
        }

        Ok(result)
    }

    /// Get a smoothed version of the interpolator
    ///
    /// # Arguments
    ///
    /// * `smoothing` - Smoothing parameter (0 = exact interpolation)
    ///
    /// # Returns
    ///
    /// A new `ThinPlateSpline` with the specified smoothing
    pub fn with_smoothing(&self, smoothing: T) -> InterpolateResult<Self> {
        if smoothing == self.smoothing {
            return Ok(self.clone());
        }

        // Extract original data
        let y = self.get_values()?;

        // Create a new TPS with different smoothing
        ThinPlateSpline::new(&self.centers.view(), &y.view(), smoothing)
    }

    /// Get the original data values
    ///
    /// This is mainly for internal use, to recreate the interpolator
    /// with different smoothing parameters
    fn get_values(&self) -> InterpolateResult<Array1<T>> {
        // Evaluate at the original centers
        let y = self.evaluate(&self.centers.view())?;

        Ok(y)
    }
}

/// Thin-plate spline kernel function
///
/// For 2D, the kernel is r^2 * ln(r)
/// For higher dimensions, it depends on the dimension
#[allow(dead_code)]
fn tps_kernel<T: Float + FromPrimitive>(r: T) -> T {
    if r == T::zero() {
        return T::zero();
    }

    // For 2D, we use r^2 * ln(r)
    let r_sq = r * r;

    if r_sq.is_zero() {
        T::zero()
    } else {
        r_sq * r_sq.ln()
    }
}

/// Create a thin-plate spline interpolator for scattered data
///
/// # Arguments
///
/// * `points` - Input coordinates of shape (n_points, n_dims)
/// * `values` - Target values of shape (n_points,)
/// * `smoothing` - Smoothing parameter (0 = exact interpolation)
///
/// # Returns
///
/// A new `ThinPlateSpline` interpolator
#[allow(dead_code)]
pub fn make_thinplate_interpolator<T>(
    points: &ArrayView2<T>,
    values: &ArrayView1<T>,
    smoothing: T,
) -> InterpolateResult<ThinPlateSpline<T>>
where
    T: Float + FromPrimitive + Debug,
{
    ThinPlateSpline::new(points, values, smoothing)
}

#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_thinplate_exact_fit() {
        // Create 2D scattered data
        let points =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();

        // Function values: f(x,y) = x^2 + y^2
        let values = array![0.0, 1.0, 1.0, 2.0];

        let tps = ThinPlateSpline::new(&points.view(), &values.view(), 0.0);
        assert!(tps.is_ok());
        let tps = tps.unwrap();

        // Test that we can evaluate at the original points
        let result = tps.evaluate(&points.view());
        assert!(result.is_ok());
        let interpolated = result.unwrap();

        // Check exact fit at the data points
        for i in 0..values.len() {
            assert!(
                (interpolated[i] - values[i]).abs() < 1e-10,
                "ThinPlateSpline should fit exactly at point {}: {} vs {}",
                i,
                interpolated[i],
                values[i]
            );
        }
    }

    #[test]
    fn test_thinplate_smoothing() {
        // Create 2D scattered data with noise
        let points = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.5, 0.5],
        )
        .unwrap();

        // Function values with noise: f(x,y) = x^2 + y^2 + noise
        let values = array![0.0, 1.0, 1.0, 2.0, 0.6]; // 0.5 + 0.1 noise

        let tps_exact = ThinPlateSpline::new(&points.view(), &values.view(), 0.0);
        let tps_smooth = ThinPlateSpline::new(&points.view(), &values.view(), 0.1);

        assert!(tps_exact.is_ok());
        assert!(tps_smooth.is_ok());

        let tps_exact = tps_exact.unwrap();
        let tps_smooth = tps_smooth.unwrap();

        // Test evaluation at original points
        let _result_exact = tps_exact.evaluate(&points.view()).unwrap();
        let result_smooth = tps_smooth.evaluate(&points.view()).unwrap();

        // With smoothing, the fit shouldn't be exact, but should still be reasonable
        for i in 0..values.len() {
            assert!(
                (result_smooth[i] - values[i]).abs() < 0.5,
                "Smoothed TPS value at point {} should be close to original: {} vs {}",
                i,
                result_smooth[i],
                values[i]
            );
        }
    }
}
