//! Radial Basis Function (RBF) interpolation for N-dimensional scattered data
//!
//! This module provides a clean, self-contained RBF interpolation framework
//! supporting multiple kernel functions:
//!
//! - **Thin plate spline** (`r^2 * ln(r)` in 2-D, `r` in other dims)
//! - **Multiquadric** (`sqrt(r^2 + eps^2)`)
//! - **Inverse multiquadric** (`1 / sqrt(r^2 + eps^2)`)
//! - **Gaussian** (`exp(-r^2 / eps^2)`)
//!
//! All kernels support N-dimensional input via row-major point matrices.
//!
//! The interpolant has the form
//!
//! ```text
//! s(x) = sum_i  w_i * phi(||x - x_i||) + polynomial_terms(x)
//! ```
//!
//! where the optional polynomial augmentation improves conditioning for
//! thin-plate and multiquadric kernels.
//!
//! # Key differences from `advanced::rbf`
//!
//! This module:
//! - Provides a simpler, focused API (`ScatteredRbf`)
//! - Includes an optional polynomial augmentation for conditionally positive
//!   definite kernels (thin plate, multiquadric)
//! - Has a cleaner builder interface and explicit epsilon auto-selection
//! - Provides per-kernel convenience constructors

use crate::error::{InterpolateError, InterpolateResult};
use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Kernel enum
// ---------------------------------------------------------------------------

/// Radial basis function kernel type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RbfKernel {
    /// Thin plate spline: `r^2 * ln(r)` (2-D convention).
    /// Conditionally positive definite of order 2.
    ThinPlateSpline,
    /// Multiquadric: `sqrt(r^2 + eps^2)`.
    /// Conditionally positive definite of order 1.
    Multiquadric,
    /// Inverse multiquadric: `1 / sqrt(r^2 + eps^2)`.
    /// Strictly positive definite.
    InverseMultiquadric,
    /// Gaussian: `exp(-r^2 / eps^2)`.
    /// Strictly positive definite.
    Gaussian,
}

// ---------------------------------------------------------------------------
// ScatteredRbf
// ---------------------------------------------------------------------------

/// N-dimensional RBF interpolator for scattered data.
///
/// Given `n` data sites in `d` dimensions and `n` function values, this
/// struct solves for weights `w` (and optional polynomial coefficients)
/// such that the interpolant exactly passes through all data points.
///
/// # Type parameter
///
/// `F`: a floating-point type implementing `InterpolationFloat`.
#[derive(Debug, Clone)]
pub struct ScatteredRbf<F: InterpolationFloat> {
    /// Data sites, shape `(n, d)`.
    centers: Array2<F>,
    /// RBF weights, length `n` (or `n + polynomial_terms` if augmented).
    weights: Array1<F>,
    /// The kernel used.
    kernel: RbfKernel,
    /// Shape parameter (epsilon).
    epsilon: F,
    /// Number of polynomial augmentation terms (0 = none).
    poly_terms: usize,
    /// Spatial dimension.
    dim: usize,
}

impl<F: InterpolationFloat> ScatteredRbf<F> {
    /// Create a new RBF interpolator.
    ///
    /// # Arguments
    ///
    /// * `points`  - Data sites, shape `(n, d)` where `n >= 1`.
    /// * `values`  - Function values, length `n`.
    /// * `kernel`  - Kernel type.
    /// * `epsilon` - Shape parameter. Pass `None` for automatic selection.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent or the linear system
    /// cannot be solved.
    pub fn new(
        points: &Array2<F>,
        values: &Array1<F>,
        kernel: RbfKernel,
        epsilon: Option<F>,
    ) -> InterpolateResult<Self> {
        let n = points.nrows();
        let d = points.ncols();

        if values.len() != n {
            return Err(InterpolateError::invalid_input(format!(
                "points has {} rows but values has {} elements",
                n,
                values.len()
            )));
        }
        if n == 0 {
            return Err(InterpolateError::empty_data("RBF interpolation"));
        }

        // Auto-select epsilon if not given
        let eps = match epsilon {
            Some(e) => e,
            None => auto_epsilon(points)?,
        };

        // Decide polynomial augmentation
        let poly_terms = match kernel {
            RbfKernel::ThinPlateSpline | RbfKernel::Multiquadric => {
                // Augment with constant + linear terms => 1 + d terms
                1 + d
            }
            _ => 0,
        };

        let total = n + poly_terms;

        // Build the interpolation matrix
        let mut mat = Array2::<F>::zeros((total, total));

        // Upper-left block: phi(||x_i - x_j||)
        for i in 0..n {
            for j in 0..n {
                let r = euclidean_dist(points, i, points, j)?;
                mat[[i, j]] = eval_kernel(kernel, r, eps)?;
            }
        }

        // Polynomial augmentation blocks (if any)
        if poly_terms > 0 {
            for i in 0..n {
                // Constant term
                mat[[i, n]] = F::one();
                mat[[n, i]] = F::one();
                // Linear terms
                for k in 0..d {
                    mat[[i, n + 1 + k]] = points[[i, k]];
                    mat[[n + 1 + k, i]] = points[[i, k]];
                }
            }
            // Bottom-right block stays zero (saddle-point system)
        }

        // Build RHS
        let mut rhs = Array1::<F>::zeros(total);
        for i in 0..n {
            rhs[i] = values[i];
        }
        // Polynomial constraint rows are zero

        // Solve the linear system via Gaussian elimination with partial pivoting
        let weights = solve_linear_system(&mat, &rhs)?;

        Ok(Self {
            centers: points.clone(),
            weights,
            kernel,
            epsilon: eps,
            poly_terms,
            dim: d,
        })
    }

    /// Create a new RBF interpolator with diagonal Tikhonov regularisation
    /// (nugget).
    ///
    /// Adds `nugget * I` to the `n × n` upper-left RBF block of the
    /// interpolation matrix.  A positive nugget makes the RBF a *smoothing*
    /// interpolant rather than an exact interpolant — i.e. the model need not
    /// pass exactly through each training point.
    ///
    /// Passing `nugget = 0.0` is equivalent to calling [`Self::new`].
    ///
    /// # Arguments
    ///
    /// * `points`  - Data sites, shape `(n, d)`.
    /// * `values`  - Function values, length `n`.
    /// * `kernel`  - Kernel type.
    /// * `epsilon` - Shape parameter.  Pass `None` for automatic selection.
    /// * `nugget`  - Diagonal regularisation strength ≥ 0.
    pub fn new_with_nugget(
        points: &Array2<F>,
        values: &Array1<F>,
        kernel: RbfKernel,
        epsilon: Option<F>,
        nugget: F,
    ) -> InterpolateResult<Self> {
        let n = points.nrows();
        let d = points.ncols();

        if values.len() != n {
            return Err(InterpolateError::invalid_input(format!(
                "points has {} rows but values has {} elements",
                n,
                values.len()
            )));
        }
        if n == 0 {
            return Err(InterpolateError::empty_data("RBF interpolation"));
        }

        // Auto-select epsilon if not given
        let eps = match epsilon {
            Some(e) => e,
            None => auto_epsilon(points)?,
        };

        // Decide polynomial augmentation
        let poly_terms = match kernel {
            RbfKernel::ThinPlateSpline | RbfKernel::Multiquadric => 1 + d,
            _ => 0,
        };

        let total = n + poly_terms;

        // Build the interpolation matrix
        let mut mat = Array2::<F>::zeros((total, total));

        // Upper-left block: phi(||x_i - x_j||) + nugget * delta_{ij}
        for i in 0..n {
            for j in 0..n {
                let r = euclidean_dist(points, i, points, j)?;
                mat[[i, j]] = eval_kernel(kernel, r, eps)?;
            }
            // Add nugget to the diagonal
            mat[[i, i]] = mat[[i, i]] + nugget;
        }

        // Polynomial augmentation blocks (if any)
        if poly_terms > 0 {
            for i in 0..n {
                mat[[i, n]] = F::one();
                mat[[n, i]] = F::one();
                for k in 0..d {
                    mat[[i, n + 1 + k]] = points[[i, k]];
                    mat[[n + 1 + k, i]] = points[[i, k]];
                }
            }
        }

        // Build RHS
        let mut rhs = Array1::<F>::zeros(total);
        for i in 0..n {
            rhs[i] = values[i];
        }

        let weights = solve_linear_system(&mat, &rhs)?;

        Ok(Self {
            centers: points.clone(),
            weights,
            kernel,
            epsilon: eps,
            poly_terms,
            dim: d,
        })
    }

    /// Evaluate the interpolant at a single query point.
    ///
    /// # Arguments
    ///
    /// * `point` - Query point, must have length `d`.
    pub fn evaluate(&self, point: &[F]) -> InterpolateResult<F> {
        if point.len() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "expected {} dims, got {}",
                self.dim,
                point.len()
            )));
        }

        let n = self.centers.nrows();
        let mut val = F::zero();

        // RBF kernel sum
        for i in 0..n {
            let r = euclidean_dist_to_point(&self.centers, i, point);
            let phi = eval_kernel(self.kernel, r, self.epsilon)?;
            val = val + self.weights[i] * phi;
        }

        // Polynomial terms
        if self.poly_terms > 0 {
            val = val + self.weights[n]; // constant
            for k in 0..self.dim {
                val = val + self.weights[n + 1 + k] * point[k];
            }
        }

        Ok(val)
    }

    /// Evaluate the interpolant at multiple query points.
    ///
    /// # Arguments
    ///
    /// * `query` - Query points, shape `(m, d)`.
    pub fn evaluate_points(&self, query: &Array2<F>) -> InterpolateResult<Array1<F>> {
        if query.ncols() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "expected {} dims, got {}",
                self.dim,
                query.ncols()
            )));
        }
        let m = query.nrows();
        let mut result = Array1::<F>::zeros(m);
        for i in 0..m {
            let pt: Vec<F> = (0..self.dim).map(|k| query[[i, k]]).collect();
            result[i] = self.evaluate(&pt)?;
        }
        Ok(result)
    }

    /// Return the kernel used.
    pub fn kernel(&self) -> RbfKernel {
        self.kernel
    }

    /// Return the shape parameter.
    pub fn epsilon(&self) -> F {
        self.epsilon
    }

    /// Return the weights.
    pub fn weights(&self) -> &Array1<F> {
        &self.weights
    }

    /// Return the data centers.
    pub fn centers(&self) -> &Array2<F> {
        &self.centers
    }
}

// ---------------------------------------------------------------------------
// Kernel evaluation
// ---------------------------------------------------------------------------

fn eval_kernel<F: InterpolationFloat>(kernel: RbfKernel, r: F, eps: F) -> InterpolateResult<F> {
    match kernel {
        RbfKernel::ThinPlateSpline => {
            if r < F::epsilon() {
                Ok(F::zero())
            } else {
                Ok(r * r * r.ln())
            }
        }
        RbfKernel::Multiquadric => Ok((r * r + eps * eps).sqrt()),
        RbfKernel::InverseMultiquadric => {
            let denom = (r * r + eps * eps).sqrt();
            if denom < F::epsilon() {
                return Err(InterpolateError::NumericalError(
                    "inverse multiquadric denominator near zero".to_string(),
                ));
            }
            Ok(F::one() / denom)
        }
        RbfKernel::Gaussian => {
            if eps.abs() < F::epsilon() {
                return Err(InterpolateError::invalid_input(
                    "Gaussian kernel requires eps > 0".to_string(),
                ));
            }
            Ok((-r * r / (eps * eps)).exp())
        }
    }
}

// ---------------------------------------------------------------------------
// Distance helpers
// ---------------------------------------------------------------------------

fn euclidean_dist<F: InterpolationFloat>(
    a: &Array2<F>,
    i: usize,
    b: &Array2<F>,
    j: usize,
) -> InterpolateResult<F> {
    let d = a.ncols();
    if b.ncols() != d {
        return Err(InterpolateError::DimensionMismatch(
            "point dimensions differ".to_string(),
        ));
    }
    let mut sq = F::zero();
    for k in 0..d {
        let diff = a[[i, k]] - b[[j, k]];
        sq = sq + diff * diff;
    }
    Ok(sq.sqrt())
}

fn euclidean_dist_to_point<F: InterpolationFloat>(a: &Array2<F>, i: usize, pt: &[F]) -> F {
    let d = a.ncols();
    let mut sq = F::zero();
    for k in 0..d {
        let diff = a[[i, k]] - pt[k];
        sq = sq + diff * diff;
    }
    sq.sqrt()
}

// ---------------------------------------------------------------------------
// Auto epsilon
// ---------------------------------------------------------------------------

fn auto_epsilon<F: InterpolationFloat>(points: &Array2<F>) -> InterpolateResult<F> {
    let n = points.nrows();
    if n <= 1 {
        return Ok(F::one());
    }
    // Use the average nearest-neighbor distance
    let d = points.ncols();
    let mut sum_nn = F::zero();
    for i in 0..n {
        let mut min_d = F::infinity();
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut sq = F::zero();
            for k in 0..d {
                let diff = points[[i, k]] - points[[j, k]];
                sq = sq + diff * diff;
            }
            let dist = sq.sqrt();
            if dist < min_d {
                min_d = dist;
            }
        }
        sum_nn = sum_nn + min_d;
    }
    let avg_nn = sum_nn
        / F::from_usize(n)
            .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?;
    // Use average nearest-neighbor distance as epsilon
    if avg_nn < F::epsilon() {
        Ok(F::one())
    } else {
        Ok(avg_nn)
    }
}

// ---------------------------------------------------------------------------
// Linear system solver (Gaussian elimination with partial pivoting)
// ---------------------------------------------------------------------------

fn solve_linear_system<F: InterpolationFloat>(
    a: &Array2<F>,
    b: &Array1<F>,
) -> InterpolateResult<Array1<F>> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(InterpolateError::invalid_input(
            "System matrix must be square and match RHS length".to_string(),
        ));
    }

    // Augmented matrix [A | b]
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < F::from_f64(1e-14).unwrap_or(F::epsilon()) {
            // Add small regularization to diagonal instead of failing
            aug[[col, col]] = aug[[col, col]] + F::from_f64(1e-10).unwrap_or(F::epsilon());
        }

        // Swap rows
        if max_row != col {
            for k in 0..=n {
                let tmp = aug[[col, k]];
                aug[[col, k]] = aug[[max_row, k]];
                aug[[max_row, k]] = tmp;
            }
        }

        let pivot = aug[[col, col]];
        if pivot.abs() < F::from_f64(1e-30).unwrap_or(F::epsilon()) {
            return Err(InterpolateError::LinalgError(
                "Singular or near-singular RBF matrix".to_string(),
            ));
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for k in col..=n {
                aug[[row, k]] = aug[[row, k]] - factor * aug[[col, k]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum = sum - aug[[i, j]] * x[j];
        }
        let diag = aug[[i, i]];
        if diag.abs() < F::from_f64(1e-30).unwrap_or(F::epsilon()) {
            return Err(InterpolateError::LinalgError(
                "Zero pivot in back substitution".to_string(),
            ));
        }
        x[i] = sum / diag;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Create a thin-plate spline RBF interpolator.
pub fn make_thin_plate_rbf<F: InterpolationFloat>(
    points: &Array2<F>,
    values: &Array1<F>,
) -> InterpolateResult<ScatteredRbf<F>> {
    ScatteredRbf::new(points, values, RbfKernel::ThinPlateSpline, None)
}

/// Create a multiquadric RBF interpolator.
pub fn make_multiquadric_rbf<F: InterpolationFloat>(
    points: &Array2<F>,
    values: &Array1<F>,
    epsilon: Option<F>,
) -> InterpolateResult<ScatteredRbf<F>> {
    ScatteredRbf::new(points, values, RbfKernel::Multiquadric, epsilon)
}

/// Create an inverse multiquadric RBF interpolator.
pub fn make_inverse_multiquadric_rbf<F: InterpolationFloat>(
    points: &Array2<F>,
    values: &Array1<F>,
    epsilon: Option<F>,
) -> InterpolateResult<ScatteredRbf<F>> {
    ScatteredRbf::new(points, values, RbfKernel::InverseMultiquadric, epsilon)
}

/// Create a Gaussian RBF interpolator.
pub fn make_gaussian_rbf<F: InterpolationFloat>(
    points: &Array2<F>,
    values: &Array1<F>,
    epsilon: Option<F>,
) -> InterpolateResult<ScatteredRbf<F>> {
    ScatteredRbf::new(points, values, RbfKernel::Gaussian, epsilon)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    fn make_2d_test_points() -> (Array2<f64>, Array1<f64>) {
        // z = x + y
        let points = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )
        .expect("shape");
        let values = array![0.0, 1.0, 1.0, 2.0, 1.0];
        (points, values)
    }

    fn make_1d_test_points() -> (Array2<f64>, Array1<f64>) {
        // f(x) = x^2
        let points = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).expect("shape");
        let values = array![0.0, 1.0, 4.0, 9.0, 16.0];
        (points, values)
    }

    // ===== Thin plate spline tests =====

    #[test]
    fn test_tps_interpolates_data() {
        let (pts, vals) = make_2d_test_points();
        let rbf = make_thin_plate_rbf(&pts, &vals).expect("construction");
        for i in 0..pts.nrows() {
            let pt = vec![pts[[i, 0]], pts[[i, 1]]];
            let v = rbf.evaluate(&pt).expect("eval");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_tps_evaluates_between_points() {
        let (pts, vals) = make_2d_test_points();
        let rbf = make_thin_plate_rbf(&pts, &vals).expect("construction");
        // f(0.25, 0.25) should be approximately 0.5
        let v = rbf.evaluate(&[0.25, 0.25]).expect("eval");
        assert!(v.is_finite());
        assert!((v - 0.5).abs() < 1.0); // Rough check
    }

    #[test]
    fn test_tps_1d() {
        let (pts, vals) = make_1d_test_points();
        let rbf = make_thin_plate_rbf(&pts, &vals).expect("construction");
        for i in 0..pts.nrows() {
            let v = rbf.evaluate(&[pts[[i, 0]]]).expect("eval");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_tps_dimension_mismatch() {
        let (pts, vals) = make_2d_test_points();
        let rbf = make_thin_plate_rbf(&pts, &vals).expect("construction");
        assert!(rbf.evaluate(&[1.0]).is_err()); // Wrong dimension
    }

    #[test]
    fn test_tps_evaluate_points_batch() {
        let (pts, vals) = make_2d_test_points();
        let rbf = make_thin_plate_rbf(&pts, &vals).expect("construction");
        let query = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.0, 0.0]).expect("shape");
        let result = rbf.evaluate_points(&query).expect("eval");
        assert_eq!(result.len(), 2);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-4);
    }

    // ===== Multiquadric tests =====

    #[test]
    fn test_mq_interpolates_data() {
        let (pts, vals) = make_2d_test_points();
        let rbf = make_multiquadric_rbf(&pts, &vals, None).expect("construction");
        for i in 0..pts.nrows() {
            let pt = vec![pts[[i, 0]], pts[[i, 1]]];
            let v = rbf.evaluate(&pt).expect("eval");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_mq_custom_epsilon() {
        let (pts, vals) = make_2d_test_points();
        let rbf = make_multiquadric_rbf(&pts, &vals, Some(0.5)).expect("construction");
        let v = rbf.evaluate(&[0.5, 0.5]).expect("eval");
        assert!(v.is_finite());
    }

    #[test]
    fn test_mq_1d() {
        let (pts, vals) = make_1d_test_points();
        let rbf = make_multiquadric_rbf(&pts, &vals, None).expect("construction");
        for i in 0..pts.nrows() {
            let v = rbf.evaluate(&[pts[[i, 0]]]).expect("eval");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_mq_3d_data() {
        // 3D: f(x,y,z) = x + y + z
        let points = Array2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .expect("shape");
        let values = array![0.0, 1.0, 1.0, 1.0];
        let rbf = make_multiquadric_rbf(&points, &values, None).expect("construction");
        let v = rbf.evaluate(&[0.0, 0.0, 0.0]).expect("eval");
        assert_abs_diff_eq!(v, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_mq_empty_data() {
        let points = Array2::<f64>::zeros((0, 2));
        let values = Array1::<f64>::zeros(0);
        assert!(make_multiquadric_rbf(&points, &values, None).is_err());
    }

    // ===== Inverse multiquadric tests =====

    #[test]
    fn test_imq_interpolates_data() {
        let (pts, vals) = make_2d_test_points();
        let rbf = make_inverse_multiquadric_rbf(&pts, &vals, None).expect("construction");
        for i in 0..pts.nrows() {
            let pt = vec![pts[[i, 0]], pts[[i, 1]]];
            let v = rbf.evaluate(&pt).expect("eval");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_imq_finite_output() {
        let (pts, vals) = make_2d_test_points();
        let rbf = make_inverse_multiquadric_rbf(&pts, &vals, Some(1.0)).expect("construction");
        let v = rbf.evaluate(&[0.3, 0.7]).expect("eval");
        assert!(v.is_finite());
    }

    #[test]
    fn test_imq_1d() {
        let (pts, vals) = make_1d_test_points();
        let rbf = make_inverse_multiquadric_rbf(&pts, &vals, None).expect("construction");
        for i in 0..pts.nrows() {
            let v = rbf.evaluate(&[pts[[i, 0]]]).expect("eval");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_imq_kernel_type() {
        let (pts, vals) = make_2d_test_points();
        let rbf = make_inverse_multiquadric_rbf(&pts, &vals, None).expect("construction");
        assert_eq!(rbf.kernel(), RbfKernel::InverseMultiquadric);
    }

    #[test]
    fn test_imq_weights_length() {
        let (pts, vals) = make_2d_test_points();
        let rbf = make_inverse_multiquadric_rbf(&pts, &vals, None).expect("construction");
        // No polynomial augmentation for IMQ
        assert_eq!(rbf.weights().len(), pts.nrows());
    }

    // ===== Gaussian tests =====

    #[test]
    fn test_gaussian_interpolates_data() {
        let (pts, vals) = make_2d_test_points();
        let rbf = make_gaussian_rbf(&pts, &vals, None).expect("construction");
        for i in 0..pts.nrows() {
            let pt = vec![pts[[i, 0]], pts[[i, 1]]];
            let v = rbf.evaluate(&pt).expect("eval");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_gaussian_smooth() {
        let (pts, vals) = make_2d_test_points();
        let rbf = make_gaussian_rbf(&pts, &vals, Some(1.0)).expect("construction");
        // Evaluate near center
        let v = rbf.evaluate(&[0.5, 0.5]).expect("eval");
        assert!(v.is_finite());
    }

    #[test]
    fn test_gaussian_1d() {
        let (pts, vals) = make_1d_test_points();
        let rbf = make_gaussian_rbf(&pts, &vals, None).expect("construction");
        for i in 0..pts.nrows() {
            let v = rbf.evaluate(&[pts[[i, 0]]]).expect("eval");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_gaussian_dimension_check() {
        let (pts, vals) = make_2d_test_points();
        let rbf = make_gaussian_rbf(&pts, &vals, None).expect("construction");
        assert!(rbf.evaluate(&[1.0, 2.0, 3.0]).is_err()); // Wrong dim
    }

    #[test]
    fn test_gaussian_epsilon() {
        let (pts, vals) = make_2d_test_points();
        let rbf = make_gaussian_rbf(&pts, &vals, Some(2.0)).expect("construction");
        assert_abs_diff_eq!(rbf.epsilon(), 2.0, epsilon = 1e-12);
    }

    // ===== General tests =====

    #[test]
    fn test_values_length_mismatch() {
        let points =
            Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).expect("shape");
        let values = array![0.0, 1.0]; // Wrong length
        assert!(ScatteredRbf::new(&points, &values, RbfKernel::Gaussian, Some(1.0)).is_err());
    }

    #[test]
    fn test_auto_epsilon_consistent() {
        let (pts, _) = make_2d_test_points();
        let eps = auto_epsilon(&pts).expect("auto");
        assert!(eps > 0.0);
        assert!(eps.is_finite());
    }

    #[test]
    fn test_5d_scatter() {
        // 5-dimensional: f = sum of coordinates
        let n = 8;
        let d = 5;
        let mut data = Vec::with_capacity(n * d);
        let mut vals = Vec::with_capacity(n);
        for i in 0..n {
            let mut s = 0.0_f64;
            for k in 0..d {
                let v = (i * d + k) as f64 * 0.1;
                data.push(v);
                s += v;
            }
            vals.push(s);
        }
        let points = Array2::from_shape_vec((n, d), data).expect("shape");
        let values = Array1::from_vec(vals.clone());
        let rbf = make_gaussian_rbf(&points, &values, Some(1.0)).expect("construction");
        // Check at data points
        for i in 0..n {
            let pt: Vec<f64> = (0..d).map(|k| points[[i, k]]).collect();
            let v = rbf.evaluate(&pt).expect("eval");
            assert_abs_diff_eq!(v, values[i], epsilon = 1e-2);
        }
    }
}
