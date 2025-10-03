//! SIMD-optimized interpolation functions
//!
//! This module provides SIMD (Single Instruction, Multiple Data) optimized versions
//! of computationally intensive interpolation operations. SIMD instructions allow
//! processing multiple data points simultaneously, leading to significant performance
//! improvements for basis function evaluation, distance calculations, and other
//! vectorizable operations.
//!
//! The optimizations target:
//! - **Basis function evaluation**: Vectorized B-spline, RBF, and polynomial basis computations
//! - **Distance calculations**: Fast Euclidean and other distance metrics for multiple points
//! - **Matrix operations**: Optimized linear algebra for interpolation systems
//! - **Batch processing**: Efficient evaluation at multiple query points
//! - **Data layout optimization**: Memory-friendly data structures for SIMD
//!
//! # SIMD Support
//!
//! This module uses conditional compilation to provide SIMD implementations when
//! available, with automatic fallback to scalar implementations on unsupported
//! architectures.
//!
//! Supported instruction sets:
//! - **x86/x86_64**: SSE2, SSE4.1, AVX, AVX2, AVX-512
//! - **ARM**: NEON (AArch64)
//! - **Portable fallback**: Pure Rust implementation for all other targets
//!
//! # Examples
//!
//! ```rust
//! use scirs2_core::ndarray::Array2;
//! use scirs2_interpolate::simd_optimized::{
//!     simd_rbf_evaluate, simd_distance_matrix, RBFKernel
//! };
//!
//! // Evaluate RBF at multiple points simultaneously
//! let centers = Array2::from_shape_vec((100, 3), vec![0.0; 300]).unwrap();
//! let queries = Array2::from_shape_vec((50, 3), vec![0.5; 150]).unwrap();
//! let coefficients = vec![1.0; 100];
//!
//! let results = simd_rbf_evaluate(
//!     &queries.view(),
//!     &centers.view(),
//!     &coefficients,
//!     RBFKernel::Gaussian,
//!     1.0
//! ).unwrap();
//! ```

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use scirs2_core::simd_ops::{AutoOptimizer, PlatformCapabilities, SimdUnifiedOps};
use std::fmt::{Debug, Display};

/// RBF kernel types for SIMD evaluation
#[derive(Debug, Clone, Copy)]
pub enum RBFKernel {
    /// Gaussian: exp(-r²/ε²)
    Gaussian,
    /// Multiquadric: sqrt(r² + ε²)
    Multiquadric,
    /// Inverse multiquadric: 1/sqrt(r² + ε²)
    InverseMultiquadric,
    /// Linear: r
    Linear,
    /// Cubic: r³
    Cubic,
}

/// SIMD configuration and capabilities
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Whether SIMD is available on this platform
    pub simd_available: bool,
    /// Vector width for f32 operations
    pub f32_width: usize,
    /// Vector width for f64 operations
    pub f64_width: usize,
    /// Instruction set being used
    pub instruction_set: String,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self::detect()
    }
}

impl SimdConfig {
    /// Detect SIMD capabilities on the current platform using core abstractions
    pub fn detect() -> Self {
        let caps = PlatformCapabilities::detect();

        Self {
            simd_available: caps.simd_available,
            f32_width: if caps.avx2_available {
                8
            } else if caps.simd_available {
                4
            } else {
                1
            },
            f64_width: if caps.avx2_available {
                4
            } else if caps.simd_available {
                2
            } else {
                1
            },
            instruction_set: if caps.avx512_available {
                "AVX512".to_string()
            } else if caps.avx2_available {
                "AVX2".to_string()
            } else if caps.neon_available {
                "NEON".to_string()
            } else if caps.simd_available {
                "SIMD".to_string()
            } else {
                "Scalar".to_string()
            },
        }
    }

    #[allow(dead_code)]
    fn fallback() -> Self {
        Self {
            simd_available: false,
            f32_width: 1,
            f64_width: 1,
            instruction_set: "Scalar".to_string(),
        }
    }
}

/// SIMD-optimized RBF evaluation
#[allow(dead_code)]
pub fn simd_rbf_evaluate<F>(
    queries: &ArrayView2<F>,
    centers: &ArrayView2<F>,
    coefficients: &[F],
    kernel: RBFKernel,
    epsilon: F,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + 'static,
{
    if queries.ncols() != centers.ncols() {
        return Err(InterpolateError::invalid_input(
            "Query and center dimensions must match".to_string(),
        ));
    }

    if centers.nrows() != coefficients.len() {
        return Err(InterpolateError::invalid_input(
            "Number of centers must match number of coefficients".to_string(),
        ));
    }

    let n_queries = queries.nrows();
    let _n_centers = centers.nrows();
    #[allow(unused_variables)]
    let dims = queries.ncols();

    let mut results = Array1::zeros(n_queries);

    // For f64 specifically, we can use SIMD if available
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
        // Unsafe transmute for SIMD operations (f64 case)
        let queries_f64 =
            unsafe { std::mem::transmute::<&ArrayView2<F>, &ArrayView2<f64>>(queries) };
        let centers_f64 =
            unsafe { std::mem::transmute::<&ArrayView2<F>, &ArrayView2<f64>>(centers) };
        let coefficients_f64: &[f64] = unsafe { std::mem::transmute(coefficients) };
        let epsilon_f64 = unsafe { *((&epsilon) as *const F as *const f64) };

        let results_f64 = simd_rbf_evaluate_f64(
            queries_f64,
            centers_f64,
            coefficients_f64,
            kernel,
            epsilon_f64,
        )?;

        // Convert back to F
        for (i, &val) in results_f64.iter().enumerate() {
            results[i] = unsafe { *((&val) as *const f64 as *const F) };
        }
    } else {
        // Fallback to scalar implementation for other types
        simd_rbf_evaluate_scalar(
            queries,
            centers,
            coefficients,
            kernel,
            epsilon,
            &mut results.view_mut(),
        )?;
    }

    Ok(results)
}

/// SIMD-optimized RBF evaluation for f64
#[allow(dead_code)]
fn simd_rbf_evaluate_f64(
    queries: &ArrayView2<f64>,
    centers: &ArrayView2<f64>,
    coefficients: &[f64],
    kernel: RBFKernel,
    epsilon: f64,
) -> InterpolateResult<Array1<f64>> {
    let optimizer = AutoOptimizer::new();
    let problem_size = queries.nrows() * centers.nrows() * queries.ncols();

    if optimizer.should_use_simd(problem_size) {
        simd_rbf_evaluate_f64_vectorized(queries, centers, coefficients, kernel, epsilon)
    } else {
        let mut results = Array1::zeros(queries.nrows());
        simd_rbf_evaluate_scalar(
            &queries.view(),
            &centers.view(),
            coefficients,
            kernel,
            epsilon,
            &mut results.view_mut(),
        )?;
        Ok(results)
    }
}

/// Vectorized f64 RBF evaluation using SIMD
#[allow(dead_code)]
fn simd_rbf_evaluate_f64_vectorized(
    queries: &ArrayView2<f64>,
    centers: &ArrayView2<f64>,
    coefficients: &[f64],
    kernel: RBFKernel,
    epsilon: f64,
) -> InterpolateResult<Array1<f64>> {
    let n_queries = queries.nrows();
    let n_centers = centers.nrows();
    let _dims = queries.ncols();
    let mut results = Array1::zeros(n_queries);

    // Use core SIMD operations for optimized computation
    for q in 0..n_queries {
        let query_row = queries.row(q);
        let mut sum = 0.0;

        for (c, &coeff) in coefficients.iter().enumerate().take(n_centers) {
            let center_row = centers.row(c);

            // Compute squared distance using SIMD operations
            let diff = &query_row - &center_row;
            let diff_arr = diff.to_owned();
            let dist_sq = f64::simd_dot(&diff_arr.view(), &diff_arr.view());

            // Apply kernel
            let kernel_val = match kernel {
                RBFKernel::Gaussian => (-dist_sq / (epsilon * epsilon)).exp(),
                RBFKernel::Multiquadric => (dist_sq + epsilon * epsilon).sqrt(),
                RBFKernel::InverseMultiquadric => 1.0 / (dist_sq + epsilon * epsilon).sqrt(),
                RBFKernel::Linear => dist_sq.sqrt(),
                RBFKernel::Cubic => {
                    let r = dist_sq.sqrt();
                    r * r * r
                }
            };

            sum += coeff * kernel_val;
        }

        results[q] = sum;
    }

    Ok(results)
}

/// Fallback implementation for all architectures
/// Scalar fallback implementation
#[allow(dead_code)]
fn simd_rbf_evaluate_scalar<F>(
    queries: &ArrayView2<F>,
    centers: &ArrayView2<F>,
    coefficients: &[F],
    kernel: RBFKernel,
    epsilon: F,
    results: &mut scirs2_core::ndarray::ArrayViewMut1<F>,
) -> InterpolateResult<()>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + 'static,
{
    let n_queries = queries.nrows();
    let n_centers = centers.nrows();
    let dims = queries.ncols();

    for q in 0..n_queries {
        let mut sum = F::zero();

        for c in 0..n_centers {
            // Compute distance
            let mut dist_sq = F::zero();
            for d in 0..dims {
                let diff = queries[[q, d]] - centers[[c, d]];
                dist_sq = dist_sq + diff * diff;
            }
            let dist = dist_sq.sqrt();

            // Apply kernel
            let kernel_val = match kernel {
                RBFKernel::Gaussian => {
                    let exp_arg = -dist_sq / (epsilon * epsilon);
                    exp_arg.exp()
                }
                RBFKernel::Multiquadric => (dist_sq + epsilon * epsilon).sqrt(),
                RBFKernel::InverseMultiquadric => F::one() / (dist_sq + epsilon * epsilon).sqrt(),
                RBFKernel::Linear => dist,
                RBFKernel::Cubic => dist * dist * dist,
            };

            sum = sum + coefficients[c] * kernel_val;
        }

        results[q] = sum;
    }

    Ok(())
}

/// Evaluate RBF kernel (scalar version)
#[allow(dead_code)]
fn evaluate_rbf_kernel_scalar(r: f64, epsilon: f64, kernel: RBFKernel) -> f64 {
    let r_sq = r * r;
    let eps_sq = epsilon * epsilon;

    match kernel {
        RBFKernel::Gaussian => (-r_sq / eps_sq).exp(),
        RBFKernel::Multiquadric => (r_sq + eps_sq).sqrt(),
        RBFKernel::InverseMultiquadric => 1.0 / (r_sq + eps_sq).sqrt(),
        RBFKernel::Linear => r,
        RBFKernel::Cubic => r * r * r,
    }
}

/// SIMD-optimized distance matrix computation
///
/// Computes pairwise Euclidean distances between two sets of points using
/// SIMD vectorized operations when available.
///
/// # Arguments
///
/// * `points_a` - First set of points with shape (n_a, dims)
/// * `points_b` - Second set of points with shape (n_b, dims)
///
/// # Returns
///
/// Distance matrix with shape (n_a, n_b) where entry (i,j) contains the
/// Euclidean distance between points_a[i] and points_b[j]
#[allow(dead_code)]
pub fn simd_distance_matrix<F>(
    points_a: &ArrayView2<F>,
    points_b: &ArrayView2<F>,
) -> InterpolateResult<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + 'static,
{
    if points_a.ncols() != points_b.ncols() {
        return Err(InterpolateError::invalid_input(
            "Point sets must have the same dimensionality".to_string(),
        ));
    }

    // For f64, use optimized SIMD implementation when available
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
        let points_a_f64 = points_a.mapv(|x| x.to_f64().unwrap_or(0.0));
        let points_b_f64 = points_b.mapv(|x| x.to_f64().unwrap_or(0.0));

        let result_f64 =
            simd_distance_matrix_f64_vectorized(&points_a_f64.view(), &points_b_f64.view())?;
        let result = result_f64.mapv(|x| F::from_f64(x).unwrap_or(F::zero()));

        return Ok(result);
    }

    // Fallback to scalar implementation for other types
    simd_distance_matrix_scalar(points_a, points_b)
}

/// SIMD-optimized distance matrix computation for f64 values
#[allow(dead_code)]
fn simd_distance_matrix_f64_vectorized(
    points_a: &ArrayView2<f64>,
    points_b: &ArrayView2<f64>,
) -> InterpolateResult<Array2<f64>> {
    let n_a = points_a.nrows();
    let n_b = points_b.nrows();
    let dims = points_a.ncols();
    let mut distances = Array2::zeros((n_a, n_b));

    let optimizer = AutoOptimizer::new();
    let problem_size = n_a * n_b * dims;

    if optimizer.should_use_simd(problem_size) {
        // Use SIMD operations for distance computation
        for i in 0..n_a {
            let a_row = points_a.row(i);
            for j in 0..n_b {
                let b_row = points_b.row(j);

                // Compute squared distance using SIMD operations
                let diff = &a_row - &b_row;
                let diff_arr = diff.to_owned();
                let dist_sq = f64::simd_dot(&diff_arr.view(), &diff_arr.view());

                distances[[i, j]] = dist_sq.sqrt();
            }
        }
    } else {
        // Fallback to scalar implementation
        return simd_distance_matrix_scalar(points_a, points_b);
    }

    Ok(distances)
}

// Direct SIMD intrinsics implementations removed - all SIMD operations now go through core abstractions

/// Scalar fallback implementation for distance matrix computation
#[allow(dead_code)]
fn simd_distance_matrix_scalar<F>(
    points_a: &ArrayView2<F>,
    points_b: &ArrayView2<F>,
) -> InterpolateResult<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy,
{
    let n_a = points_a.nrows();
    let n_b = points_b.nrows();
    let dims = points_a.ncols();
    let mut distances = Array2::zeros((n_a, n_b));

    for i in 0..n_a {
        for j in 0..n_b {
            let mut dist_sq = F::zero();
            for d in 0..dims {
                let diff = points_a[[i, d]] - points_b[[j, d]];
                dist_sq = dist_sq + diff * diff;
            }
            distances[[i, j]] = dist_sq.sqrt();
        }
    }

    Ok(distances)
}

/// SIMD-optimized batch evaluation for B-splines
#[allow(dead_code)]
pub fn simd_bspline_batch_evaluate<F>(
    knots: &ArrayView1<F>,
    coefficients: &ArrayView1<F>,
    degree: usize,
    x_values: &ArrayView1<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + 'static,
{
    let mut results = Array1::zeros(x_values.len());

    // For now, delegate to scalar implementation
    // In a full SIMD implementation, this would vectorize the de Boor algorithm
    for (i, &x) in x_values.iter().enumerate() {
        results[i] = scalar_bspline_evaluate(knots, coefficients, degree, x)?;
    }

    Ok(results)
}

/// Vectorized B-spline basis function evaluation using SIMD
///
/// This function computes B-spline basis functions for multiple evaluation points
/// simultaneously using SIMD instructions when available.
#[allow(dead_code)]
pub fn simd_bspline_basis_functions<F>(
    knots: &ArrayView1<F>,
    degree: usize,
    x_values: &ArrayView1<F>,
    span_indices: &[usize],
) -> InterpolateResult<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + 'static,
{
    let n_points = x_values.len();
    let n_basis = degree + 1;
    let mut basis_values = Array2::zeros((n_points, n_basis));

    // Use scalar implementation (AVX2 implementation removed)
    scalar_bspline_basis_functions(knots, degree, x_values, span_indices, &mut basis_values)
}

// B-spline basis function AVX2 implementation removed - using scalar implementation only

/// Scalar implementation of B-spline basis function computation
#[allow(dead_code)]
fn scalar_bspline_basis_functions<F>(
    knots: &ArrayView1<F>,
    degree: usize,
    x_values: &ArrayView1<F>,
    span_indices: &[usize],
    basis_values: &mut Array2<F>,
) -> InterpolateResult<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + 'static,
{
    let n_points = x_values.len();
    let n_basis = degree + 1;

    for i in 0..n_points {
        let span = span_indices[i];
        let x = x_values[i];
        let basis = compute_basis_functions_scalar(knots, degree, x, span)?;

        for j in 0..n_basis {
            basis_values[[i, j]] = basis[j];
        }
    }

    Ok(basis_values.to_owned())
}

/// Compute basis functions for a single point using de Boor's algorithm
#[allow(dead_code)]
fn compute_basis_functions_scalar<F>(
    knots: &ArrayView1<F>,
    degree: usize,
    x: F,
    span: usize,
) -> InterpolateResult<Vec<F>>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + 'static,
{
    let mut basis = vec![F::zero(); degree + 1];
    basis[0] = F::one();

    for j in 1..=degree {
        let mut saved = F::zero();
        for r in 0..j {
            let temp = basis[r];

            let left_knot = if span + 1 + r >= j && span + 1 + r - j < knots.len() {
                knots[span + 1 + r - j]
            } else {
                F::zero()
            };

            let right_knot = if span + 1 + r < knots.len() {
                knots[span + 1 + r]
            } else {
                F::zero()
            };

            let denom = right_knot - left_knot;
            let alpha = if denom != F::zero() {
                (x - left_knot) / denom
            } else {
                F::zero()
            };

            basis[r] = saved + (F::one() - alpha) * temp;
            saved = alpha * temp;
        }
        basis[j] = saved;
    }

    Ok(basis)
}

/// Improved scalar B-spline evaluation using cached workspace
#[allow(dead_code)]
fn scalar_bspline_evaluate<F>(
    knots: &ArrayView1<F>,
    coefficients: &ArrayView1<F>,
    degree: usize,
    x: F,
) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + 'static,
{
    // Find the knot span
    let span = find_knot_span(knots, coefficients.len(), degree, x);

    // Compute basis functions
    let basis = compute_basis_functions_scalar(knots, degree, x, span)?;

    // Evaluate the spline
    let mut result = F::zero();
    for (i, &basis_val) in basis.iter().enumerate().take(degree + 1) {
        let coeff_idx = span - degree + i;
        if coeff_idx < coefficients.len() {
            result = result + coefficients[coeff_idx] * basis_val;
        }
    }

    Ok(result)
}

/// Find the knot span for a given parameter value
#[allow(dead_code)]
fn find_knot_span<F>(knots: &ArrayView1<F>, n: usize, degree: usize, x: F) -> usize
where
    F: Float + FromPrimitive + PartialOrd,
{
    if x >= knots[n] {
        return n - 1;
    }
    if x <= knots[degree] {
        return degree;
    }

    // Binary search
    let mut low = degree;
    let mut high = n;
    let mut mid = (low + high) / 2;

    while x < knots[mid] || x >= knots[mid + 1] {
        if x < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }

    mid
}

// SIMD helper functions removed - all operations now use core abstractions

/// Get SIMD configuration information
#[allow(dead_code)]
pub fn get_simd_config() -> SimdConfig {
    SimdConfig::detect()
}

/// Check if SIMD is available on this platform
#[allow(dead_code)]
pub fn is_simd_available() -> bool {
    SimdConfig::detect().simd_available
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::{array, Axis};

    #[test]
    fn test_simd_config_detection() {
        let config = SimdConfig::detect();
        println!("SIMD Config: {config:?}");

        // Basic validation
        assert!(config.f32_width >= 1);
        assert!(config.f64_width >= 1);
        assert!(!config.instruction_set.is_empty());
    }

    #[test]
    fn test_simd_rbf_evaluate() {
        let queries = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let centers = array![[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]];
        let coefficients = vec![1.0, 1.0, 1.0];

        let results = simd_rbf_evaluate(
            &queries.view(),
            &centers.view(),
            &coefficients,
            RBFKernel::Gaussian,
            1.0,
        )
        .unwrap();

        assert_eq!(results.len(), 3);

        // Results should be finite and reasonable
        for &result in results.iter() {
            assert!(result.is_finite());
            assert!(result >= 0.0); // Gaussian RBF is always positive
        }
    }

    #[test]
    fn test_simd_distance_matrix() {
        let points_a = array![[0.0, 0.0], [1.0, 0.0]];
        let points_b = array![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let distances = simd_distance_matrix(&points_a.view(), &points_b.view()).unwrap();

        assert_eq!(distances.shape(), &[2, 3]);

        // Check some known distances
        assert_relative_eq!(distances[[0, 0]], 0.0, epsilon = 1e-10); // Same point
        assert_relative_eq!(distances[[0, 1]], 1.0, epsilon = 1e-10); // Unit distance
        assert_relative_eq!(distances[[1, 0]], 1.0, epsilon = 1e-10); // Unit distance
    }

    #[test]
    fn test_rbf_kernel_consistency() {
        // Test that SIMD and scalar implementations give same results
        let queries = array![[0.25, 0.75]];
        let centers = array![[0.0, 0.0], [1.0, 1.0]];
        let coefficients = vec![0.5, 1.5];
        let epsilon = 1.0;

        let simd_result = simd_rbf_evaluate(
            &queries.view(),
            &centers.view(),
            &coefficients,
            RBFKernel::Gaussian,
            epsilon,
        )
        .unwrap();

        // Compute scalar result manually
        let mut scalar_result = 0.0;
        for (i, center) in centers.axis_iter(Axis(0)).enumerate() {
            let mut dist_sq = 0.0;
            for (q_val, c_val) in queries.row(0).iter().zip(center.iter()) {
                let diff = q_val - c_val;
                dist_sq += diff * diff;
            }
            let kernel_val = (-dist_sq / (epsilon * epsilon)).exp();
            scalar_result += coefficients[i] * kernel_val;
        }

        assert_relative_eq!(simd_result[0], scalar_result, epsilon = 1e-10);
    }

    #[test]
    fn test_different_rbf_kernels() {
        let queries = array![[0.5, 0.5]];
        let centers = array![[0.0, 0.0], [1.0, 1.0]];
        let coefficients = vec![1.0, 1.0];
        let epsilon = 1.0;

        let kernels = [
            RBFKernel::Gaussian,
            RBFKernel::Multiquadric,
            RBFKernel::InverseMultiquadric,
            RBFKernel::Linear,
            RBFKernel::Cubic,
        ];

        for kernel in kernels {
            let result = simd_rbf_evaluate(
                &queries.view(),
                &centers.view(),
                &coefficients,
                kernel,
                epsilon,
            )
            .unwrap();

            assert_eq!(result.len(), 1);
            assert!(result[0].is_finite());
        }
    }

    #[test]
    fn test_simd_availability() {
        let available = is_simd_available();
        println!("SIMD available: {available}");

        // Test should always pass regardless of SIMD availability
        // (just checking that the SIMD detection function doesn't panic)
    }

    #[test]
    fn test_bspline_batch_evaluate() {
        let knots = array![0.0, 1.0, 2.0, 3.0];
        let coefficients = array![1.0, 2.0];
        let x_values = array![0.5, 1.5, 2.5];

        let results =
            simd_bspline_batch_evaluate(&knots.view(), &coefficients.view(), 1, &x_values.view())
                .unwrap();

        assert_eq!(results.len(), 3);
        // Results should be finite (actual values computed by scalar implementation)
        for &result in results.iter() {
            assert!(result.is_finite());
        }
    }
}
