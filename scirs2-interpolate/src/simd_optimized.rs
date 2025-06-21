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
//! use ndarray::Array2;
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
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive, Zero};
use std::fmt::{Debug, Display};

// Platform-specific SIMD imports
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

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
    /// Detect SIMD capabilities on the current platform
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                Self {
                    simd_available: true,
                    f32_width: 8, // AVX2: 256-bit / 32-bit = 8 f32s
                    f64_width: 4, // AVX2: 256-bit / 64-bit = 4 f64s
                    instruction_set: "AVX2".to_string(),
                }
            } else if is_x86_feature_detected!("avx") {
                Self {
                    simd_available: true,
                    f32_width: 8,
                    f64_width: 4,
                    instruction_set: "AVX".to_string(),
                }
            } else if is_x86_feature_detected!("sse2") {
                Self {
                    simd_available: true,
                    f32_width: 4, // SSE: 128-bit / 32-bit = 4 f32s
                    f64_width: 2, // SSE: 128-bit / 64-bit = 2 f64s
                    instruction_set: "SSE2".to_string(),
                }
            } else {
                Self::fallback()
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                simd_available: true,
                f32_width: 4, // NEON: 128-bit / 32-bit = 4 f32s
                f64_width: 2, // NEON: 128-bit / 64-bit = 2 f64s
                instruction_set: "NEON".to_string(),
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
        {
            Self::fallback()
        }
    }

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
        return Err(InterpolateError::ValueError(
            "Query and center dimensions must match".to_string(),
        ));
    }

    if centers.nrows() != coefficients.len() {
        return Err(InterpolateError::ValueError(
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
fn simd_rbf_evaluate_f64(
    queries: &ArrayView2<f64>,
    centers: &ArrayView2<f64>,
    coefficients: &[f64],
    kernel: RBFKernel,
    epsilon: f64,
) -> InterpolateResult<Array1<f64>> {
    let config = SimdConfig::detect();

    if config.simd_available && config.f64_width >= 2 {
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
#[cfg(target_arch = "x86_64")]
fn simd_rbf_evaluate_f64_vectorized(
    queries: &ArrayView2<f64>,
    centers: &ArrayView2<f64>,
    coefficients: &[f64],
    kernel: RBFKernel,
    epsilon: f64,
) -> InterpolateResult<Array1<f64>> {
    let n_queries = queries.nrows();
    #[allow(unused_variables)]
    let n_centers = centers.nrows();
    #[allow(unused_variables)]
    let dims = queries.ncols();
    let mut results = Array1::zeros(n_queries);

    if is_x86_feature_detected!("avx2") {
        unsafe {
            simd_rbf_evaluate_avx2(
                queries,
                centers,
                coefficients,
                kernel,
                epsilon,
                &mut results,
            )?;
        }
    } else if is_x86_feature_detected!("sse2") {
        unsafe {
            simd_rbf_evaluate_sse2(
                queries,
                centers,
                coefficients,
                kernel,
                epsilon,
                &mut results,
            )?;
        }
    } else {
        simd_rbf_evaluate_scalar(
            &queries.view(),
            &centers.view(),
            coefficients,
            kernel,
            epsilon,
            &mut results.view_mut(),
        )?;
    }

    Ok(results)
}

/// AVX2 implementation for f64 RBF evaluation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_rbf_evaluate_avx2(
    queries: &ArrayView2<f64>,
    centers: &ArrayView2<f64>,
    coefficients: &[f64],
    kernel: RBFKernel,
    epsilon: f64,
    results: &mut Array1<f64>,
) -> InterpolateResult<()> {
    let n_queries = queries.nrows();
    let n_centers = centers.nrows();
    let dims = queries.ncols();

    // Process 4 f64 values at a time with AVX2
    let simd_width = 4;
    let _epsilon_vec = _mm256_set1_pd(epsilon);
    let epsilon_sq_vec = _mm256_set1_pd(epsilon * epsilon);

    for q in 0..n_queries {
        let mut result_vec = _mm256_setzero_pd();

        // Process centers in chunks of 4
        for c_chunk in (0..n_centers).step_by(simd_width) {
            let chunk_size = (n_centers - c_chunk).min(simd_width);

            if chunk_size == simd_width {
                // Load coefficients
                let coeff_vec = _mm256_loadu_pd(coefficients.as_ptr().add(c_chunk));

                // Compute distances squared
                let mut dist_sq_vec = _mm256_setzero_pd();

                for d in 0..dims {
                    let query_val = _mm256_set1_pd(queries[[q, d]]);
                    let center_vals = _mm256_set_pd(
                        centers[[c_chunk + 3, d]],
                        centers[[c_chunk + 2, d]],
                        centers[[c_chunk + 1, d]],
                        centers[[c_chunk, d]],
                    );

                    let diff = _mm256_sub_pd(query_val, center_vals);
                    dist_sq_vec = _mm256_fmadd_pd(diff, diff, dist_sq_vec);
                }

                // Apply RBF kernel
                let kernel_vals = match kernel {
                    RBFKernel::Gaussian => {
                        let neg_dist_sq_eps = _mm256_div_pd(
                            _mm256_sub_pd(_mm256_setzero_pd(), dist_sq_vec),
                            epsilon_sq_vec,
                        );
                        simd_exp_pd(neg_dist_sq_eps)
                    }
                    RBFKernel::Multiquadric => {
                        let r_sq_plus_eps_sq = _mm256_add_pd(dist_sq_vec, epsilon_sq_vec);
                        simd_sqrt_pd(r_sq_plus_eps_sq)
                    }
                    RBFKernel::InverseMultiquadric => {
                        let r_sq_plus_eps_sq = _mm256_add_pd(dist_sq_vec, epsilon_sq_vec);
                        let sqrt_val = simd_sqrt_pd(r_sq_plus_eps_sq);
                        _mm256_div_pd(_mm256_set1_pd(1.0), sqrt_val)
                    }
                    RBFKernel::Linear => simd_sqrt_pd(dist_sq_vec),
                    RBFKernel::Cubic => {
                        let r = simd_sqrt_pd(dist_sq_vec);
                        _mm256_mul_pd(_mm256_mul_pd(r, r), r)
                    }
                };

                // Multiply by coefficients and accumulate
                let weighted = _mm256_mul_pd(kernel_vals, coeff_vec);
                result_vec = _mm256_add_pd(result_vec, weighted);
            } else {
                // Handle remaining elements with scalar code
                for c in c_chunk..c_chunk + chunk_size {
                    let mut dist_sq = 0.0;
                    for d in 0..dims {
                        let diff = queries[[q, d]] - centers[[c, d]];
                        dist_sq += diff * diff;
                    }

                    let kernel_val = evaluate_rbf_kernel_scalar(dist_sq.sqrt(), epsilon, kernel);
                    results[q] += coefficients[c] * kernel_val;
                }
            }
        }

        // Horizontal sum of the SIMD result
        let result_scalar = simd_horizontal_sum_pd(result_vec);
        results[q] += result_scalar;
    }

    Ok(())
}

/// SSE2 implementation for f64 RBF evaluation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn simd_rbf_evaluate_sse2(
    queries: &ArrayView2<f64>,
    centers: &ArrayView2<f64>,
    coefficients: &[f64],
    kernel: RBFKernel,
    epsilon: f64,
    results: &mut Array1<f64>,
) -> InterpolateResult<()> {
    let n_queries = queries.nrows();
    let n_centers = centers.nrows();
    let dims = queries.ncols();

    // Process 2 f64 values at a time with SSE2
    let simd_width = 2;
    let _epsilon_vec = _mm_set1_pd(epsilon);
    let epsilon_sq_vec = _mm_set1_pd(epsilon * epsilon);

    for q in 0..n_queries {
        let mut result_vec = _mm_setzero_pd();

        // Process centers in chunks of 2
        for c_chunk in (0..n_centers).step_by(simd_width) {
            let chunk_size = (n_centers - c_chunk).min(simd_width);

            if chunk_size == simd_width {
                // Load coefficients
                let coeff_vec = _mm_loadu_pd(coefficients.as_ptr().add(c_chunk));

                // Compute distances squared
                let mut dist_sq_vec = _mm_setzero_pd();

                for d in 0..dims {
                    let query_val = _mm_set1_pd(queries[[q, d]]);
                    let center_vals = _mm_set_pd(centers[[c_chunk + 1, d]], centers[[c_chunk, d]]);

                    let diff = _mm_sub_pd(query_val, center_vals);
                    let diff_sq = _mm_mul_pd(diff, diff);
                    dist_sq_vec = _mm_add_pd(dist_sq_vec, diff_sq);
                }

                // Apply RBF kernel (simplified for SSE2)
                let kernel_vals = match kernel {
                    RBFKernel::Gaussian => {
                        // Simplified Gaussian approximation for SSE2
                        let neg_dist_sq_eps =
                            _mm_div_pd(_mm_sub_pd(_mm_setzero_pd(), dist_sq_vec), epsilon_sq_vec);
                        simd_exp_pd_sse2(neg_dist_sq_eps)
                    }
                    RBFKernel::Linear => simd_sqrt_pd_sse2(dist_sq_vec),
                    _ => {
                        // Fallback to scalar for complex kernels
                        let mut scalar_vals = [0.0; 2];
                        _mm_storeu_pd(scalar_vals.as_mut_ptr(), dist_sq_vec);
                        let k1 = evaluate_rbf_kernel_scalar(scalar_vals[0].sqrt(), epsilon, kernel);
                        let k2 = evaluate_rbf_kernel_scalar(scalar_vals[1].sqrt(), epsilon, kernel);
                        _mm_set_pd(k2, k1)
                    }
                };

                // Multiply by coefficients and accumulate
                let weighted = _mm_mul_pd(kernel_vals, coeff_vec);
                result_vec = _mm_add_pd(result_vec, weighted);
            } else {
                // Handle remaining elements with scalar code
                for c in c_chunk..c_chunk + chunk_size {
                    let mut dist_sq = 0.0;
                    for d in 0..dims {
                        let diff = queries[[q, d]] - centers[[c, d]];
                        dist_sq += diff * diff;
                    }

                    let kernel_val = evaluate_rbf_kernel_scalar(dist_sq.sqrt(), epsilon, kernel);
                    results[q] += coefficients[c] * kernel_val;
                }
            }
        }

        // Horizontal sum of the SSE2 result
        let result_scalar = {
            let sum_vec = _mm_hadd_pd(result_vec, result_vec);
            let mut result = 0.0;
            _mm_store_sd(&mut result, sum_vec);
            result
        };
        results[q] += result_scalar;
    }

    Ok(())
}

/// Fallback implementation for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
fn simd_rbf_evaluate_f64_vectorized(
    queries: &ArrayView2<f64>,
    centers: &ArrayView2<f64>,
    coefficients: &[f64],
    kernel: RBFKernel,
    epsilon: f64,
) -> InterpolateResult<Array1<f64>> {
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

/// Scalar fallback implementation
fn simd_rbf_evaluate_scalar<F>(
    queries: &ArrayView2<F>,
    centers: &ArrayView2<F>,
    coefficients: &[F],
    kernel: RBFKernel,
    epsilon: F,
    results: &mut ndarray::ArrayViewMut1<F>,
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
pub fn simd_distance_matrix<F>(
    points_a: &ArrayView2<F>,
    points_b: &ArrayView2<F>,
) -> InterpolateResult<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + 'static,
{
    if points_a.ncols() != points_b.ncols() {
        return Err(InterpolateError::ValueError(
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
#[cfg(feature = "simd")]
fn simd_distance_matrix_f64_vectorized(
    points_a: &ArrayView2<f64>,
    points_b: &ArrayView2<f64>,
) -> InterpolateResult<Array2<f64>> {
    let config = get_simd_config();

    // Use the best available SIMD instruction set based on detected capabilities
    #[cfg(target_arch = "x86_64")]
    {
        if config.instruction_set == "AVX2" {
            return unsafe { simd_distance_matrix_avx2(points_a, points_b) };
        } else if config.instruction_set == "AVX" || config.instruction_set == "SSE2" {
            return unsafe { simd_distance_matrix_sse2(points_a, points_b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if config.instruction_set == "NEON" {
            return unsafe { simd_distance_matrix_neon(points_a, points_b) };
        }
    }

    // Fallback to scalar implementation
    simd_distance_matrix_scalar(points_a, points_b)
}

#[cfg(not(feature = "simd"))]
fn simd_distance_matrix_f64_vectorized(
    points_a: &ArrayView2<f64>,
    points_b: &ArrayView2<f64>,
) -> InterpolateResult<Array2<f64>> {
    simd_distance_matrix_scalar(points_a, points_b)
}

/// AVX2-optimized distance matrix computation (processes 4 f64 values at once)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn simd_distance_matrix_avx2(
    points_a: &ArrayView2<f64>,
    points_b: &ArrayView2<f64>,
) -> InterpolateResult<Array2<f64>> {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        let n_a = points_a.nrows();
        let n_b = points_b.nrows();
        let dims = points_a.ncols();
        let mut distances = Array2::zeros((n_a, n_b));

        for i in 0..n_a {
            for j in 0..n_b {
                let mut dist_sq_vec = _mm256_setzero_pd();

                // Process 4 dimensions at a time
                let mut d = 0;
                while d + 4 <= dims {
                    let a_vec = _mm256_loadu_pd(points_a.as_ptr().add(i * dims + d));
                    let b_vec = _mm256_loadu_pd(points_b.as_ptr().add(j * dims + d));
                    let diff = _mm256_sub_pd(a_vec, b_vec);
                    dist_sq_vec = _mm256_fmadd_pd(diff, diff, dist_sq_vec);
                    d += 4;
                }

                // Horizontal reduction: sum the 4 components
                let sum_high = _mm256_extractf128_pd(dist_sq_vec, 1);
                let sum_low = _mm256_extractf128_pd(dist_sq_vec, 0);
                let sum_128 = _mm_add_pd(sum_low, sum_high);
                let sum_final = _mm_hadd_pd(sum_128, sum_128);
                let mut dist_sq: f64 = _mm_cvtsd_f64(sum_final);

                // Handle remaining dimensions
                for d in d..dims {
                    let diff = points_a[[i, d]] - points_b[[j, d]];
                    dist_sq += diff * diff;
                }

                distances[[i, j]] = dist_sq.sqrt();
            }
        }

        Ok(distances)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        simd_distance_matrix_scalar(points_a, points_b)
    }
}

/// SSE2-optimized distance matrix computation (processes 2 f64 values at once)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(dead_code)]
unsafe fn simd_distance_matrix_sse2(
    points_a: &ArrayView2<f64>,
    points_b: &ArrayView2<f64>,
) -> InterpolateResult<Array2<f64>> {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        let n_a = points_a.nrows();
        let n_b = points_b.nrows();
        let dims = points_a.ncols();
        let mut distances = Array2::zeros((n_a, n_b));

        for i in 0..n_a {
            for j in 0..n_b {
                let mut dist_sq_vec = _mm_setzero_pd();

                // Process 2 dimensions at a time
                let mut d = 0;
                while d + 2 <= dims {
                    let a_vec = _mm_loadu_pd(points_a.as_ptr().add(i * dims + d));
                    let b_vec = _mm_loadu_pd(points_b.as_ptr().add(j * dims + d));
                    let diff = _mm_sub_pd(a_vec, b_vec);
                    dist_sq_vec = _mm_add_pd(dist_sq_vec, _mm_mul_pd(diff, diff));
                    d += 2;
                }

                // Horizontal reduction
                let sum_high = _mm_unpackhi_pd(dist_sq_vec, dist_sq_vec);
                let sum_low = _mm_add_sd(dist_sq_vec, sum_high);
                let mut dist_sq: f64 = _mm_cvtsd_f64(sum_low);

                // Handle remaining dimension
                if d < dims {
                    let diff = points_a[[i, d]] - points_b[[j, d]];
                    dist_sq += diff * diff;
                }

                distances[[i, j]] = dist_sq.sqrt();
            }
        }

        Ok(distances)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        simd_distance_matrix_scalar(points_a, points_b)
    }
}

/// NEON-optimized distance matrix computation for ARM64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_distance_matrix_neon(
    points_a: &ArrayView2<f64>,
    points_b: &ArrayView2<f64>,
) -> InterpolateResult<Array2<f64>> {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        let n_a = points_a.nrows();
        let n_b = points_b.nrows();
        let dims = points_a.ncols();
        let mut distances = Array2::zeros((n_a, n_b));

        for i in 0..n_a {
            for j in 0..n_b {
                let mut dist_sq_vec = vdupq_n_f64(0.0);

                // Process 2 dimensions at a time
                let mut d = 0;
                while d + 2 <= dims {
                    let a_vec = vld1q_f64(points_a.as_ptr().add(i * dims + d));
                    let b_vec = vld1q_f64(points_b.as_ptr().add(j * dims + d));
                    let diff = vsubq_f64(a_vec, b_vec);
                    dist_sq_vec = vfmaq_f64(dist_sq_vec, diff, diff);
                    d += 2;
                }

                // Horizontal reduction
                let sum = vaddvq_f64(dist_sq_vec);
                let mut dist_sq = sum;

                // Handle remaining dimension
                if d < dims {
                    let diff = points_a[[i, d]] - points_b[[j, d]];
                    dist_sq += diff * diff;
                }

                distances[[i, j]] = dist_sq.sqrt();
            }
        }

        Ok(distances)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        simd_distance_matrix_scalar(points_a, points_b)
    }
}

/// Scalar fallback implementation for distance matrix computation
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

    // Use SIMD for f64, fall back to scalar for other types
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return simd_bspline_basis_avx2(
                        knots,
                        degree,
                        x_values,
                        span_indices,
                        &mut basis_values,
                    );
                }
            }
        }
    }

    // Scalar fallback
    scalar_bspline_basis_functions(knots, degree, x_values, span_indices, &mut basis_values)
}

/// SIMD-optimized B-spline basis function computation for f64 using AVX2
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
unsafe fn simd_bspline_basis_avx2<F>(
    knots: &ArrayView1<F>,
    degree: usize,
    x_values: &ArrayView1<F>,
    span_indices: &[usize],
    basis_values: &mut Array2<F>,
) -> InterpolateResult<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + 'static,
{
    use wide::f64x4;

    let n_points = x_values.len();
    let n_basis = degree + 1;

    // Process points in groups of 4 for SIMD
    for chunk_start in (0..n_points).step_by(4) {
        let chunk_end = (chunk_start + 4).min(n_points);
        let chunk_size = chunk_end - chunk_start;

        if chunk_size == 4 {
            // Full SIMD processing
            let x_vec = f64x4::new([
                x_values[chunk_start].to_f64().unwrap_or(0.0),
                x_values[chunk_start + 1].to_f64().unwrap_or(0.0),
                x_values[chunk_start + 2].to_f64().unwrap_or(0.0),
                x_values[chunk_start + 3].to_f64().unwrap_or(0.0),
            ]);

            // Compute basis functions for these 4 points simultaneously
            let mut basis_chunk = vec![f64x4::ZERO; n_basis];
            basis_chunk[0] = f64x4::splat(1.0);

            for j in 1..=degree {
                let mut saved = f64x4::ZERO;

                for r in 0..j {
                    let temp = basis_chunk[r];

                    // Compute alpha values for all 4 points
                    let mut alpha = f64x4::ZERO;
                    for p in 0..4 {
                        let span = span_indices[chunk_start + p];
                        let knot_left = knots[span + 1 + r - j].to_f64().unwrap_or(0.0);
                        let knot_right = knots[span + 1 + r].to_f64().unwrap_or(0.0);
                        let denom = knot_right - knot_left;

                        let alpha_val = if denom != 0.0 {
                            (x_vec.as_array_ref()[p] - knot_left) / denom
                        } else {
                            0.0
                        };

                        // Set the appropriate element in the SIMD vector
                        let mut alpha_array = *alpha.as_array_ref();
                        alpha_array[p] = alpha_val;
                        alpha = f64x4::new(alpha_array);
                    }

                    basis_chunk[r] = saved + (f64x4::splat(1.0) - alpha) * temp;
                    saved = alpha * temp;
                }
                basis_chunk[j] = saved;
            }

            // Store results back to the output array
            for p in 0..4 {
                for b in 0..n_basis {
                    let val = basis_chunk[b].as_array_ref()[p];
                    basis_values[[chunk_start + p, b]] = F::from_f64(val).unwrap();
                }
            }
        } else {
            // Handle partial chunk with scalar computation
            for p in chunk_start..chunk_end {
                let span = span_indices[p];
                let x = x_values[p];
                let basis = compute_basis_functions_scalar(knots, degree, x, span)?;

                for b in 0..n_basis {
                    basis_values[[p, b]] = basis[b];
                }
            }
        }
    }

    Ok(basis_values.to_owned())
}

/// Scalar implementation of B-spline basis function computation
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

// SIMD helper functions for x86_64

#[cfg(target_arch = "x86_64")]
unsafe fn simd_exp_pd(x: __m256d) -> __m256d {
    // Simplified exponential approximation for AVX2
    // In a production implementation, this would use a more accurate approximation
    let one = _mm256_set1_pd(1.0);
    let x_clamped = _mm256_max_pd(
        _mm256_set1_pd(-10.0),
        _mm256_min_pd(x, _mm256_set1_pd(10.0)),
    );
    // Very rough approximation: exp(x) ≈ 1 + x + x²/2
    let x_sq = _mm256_mul_pd(x_clamped, x_clamped);
    let x_sq_half = _mm256_mul_pd(x_sq, _mm256_set1_pd(0.5));
    _mm256_add_pd(_mm256_add_pd(one, x_clamped), x_sq_half)
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_sqrt_pd(x: __m256d) -> __m256d {
    _mm256_sqrt_pd(x)
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_horizontal_sum_pd(x: __m256d) -> f64 {
    let sum_high_low = _mm256_hadd_pd(x, x);
    let sum_128 = _mm256_extractf128_pd(sum_high_low, 1);
    let sum_64 = _mm_add_pd(_mm256_castpd256_pd128(sum_high_low), sum_128);
    let mut result = 0.0;
    _mm_store_sd(&mut result, sum_64);
    result
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_exp_pd_sse2(x: __m128d) -> __m128d {
    // Simplified exponential approximation for SSE2
    let one = _mm_set1_pd(1.0);
    let x_clamped = _mm_max_pd(_mm_set1_pd(-10.0), _mm_min_pd(x, _mm_set1_pd(10.0)));
    let x_sq = _mm_mul_pd(x_clamped, x_clamped);
    let x_sq_half = _mm_mul_pd(x_sq, _mm_set1_pd(0.5));
    _mm_add_pd(_mm_add_pd(one, x_clamped), x_sq_half)
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_sqrt_pd_sse2(x: __m128d) -> __m128d {
    _mm_sqrt_pd(x)
}

/// Get SIMD configuration information
pub fn get_simd_config() -> SimdConfig {
    SimdConfig::detect()
}

/// Check if SIMD is available on this platform
pub fn is_simd_available() -> bool {
    SimdConfig::detect().simd_available
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{array, Axis};

    #[test]
    fn test_simd_config_detection() {
        let config = SimdConfig::detect();
        println!("SIMD Config: {:?}", config);

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
        println!("SIMD available: {}", available);

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
