//! Next-generation SIMD optimizations for statistical operations (v6)
//!
//! This module provides comprehensive SIMD optimizations that fully leverage
//! scirs2-core's unified SIMD infrastructure, with advanced vectorization
//! strategies and platform-specific optimizations.

use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, NumCast, One, Zero};
use scirs2_core::Rng;
use scirs2_core::{
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
    validation::*,
};
use std::marker::PhantomData;

/// Advanced SIMD configuration with platform detection
#[derive(Debug, Clone)]
pub struct AdvancedSimdConfig {
    /// Platform capabilities detected at runtime
    pub capabilities: PlatformCapabilities,
    /// Chunk size for SIMD operations (auto-determined based on platform)
    pub chunksize: usize,
    /// Whether to use parallel SIMD processing
    pub parallel_enabled: bool,
    /// Minimum data size for SIMD processing
    pub simd_threshold: usize,
}

impl Default for AdvancedSimdConfig {
    fn default() -> Self {
        let capabilities = PlatformCapabilities::detect();
        let chunksize = if capabilities.avx512_available {
            16 // 512-bit / 32-bit = 16 elements for f32
        } else if capabilities.avx2_available {
            8 // 256-bit / 32-bit = 8 elements for f32
        } else if capabilities.simd_available {
            4 // 128-bit / 32-bit = 4 elements for f32
        } else {
            1 // Scalar fallback
        };

        Self {
            capabilities,
            chunksize,
            parallel_enabled: true,
            simd_threshold: 64,
        }
    }
}

/// Advanced-optimized SIMD statistics computer
pub struct AdvancedSimdStatistics<F> {
    config: AdvancedSimdConfig,
    _phantom: PhantomData<F>,
}

impl<F> AdvancedSimdStatistics<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    /// Create new advanced-optimized SIMD statistics computer
    pub fn new() -> Self {
        Self {
            config: AdvancedSimdConfig::default(),
            _phantom: PhantomData,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: AdvancedSimdConfig) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }

    /// Compute comprehensive statistics using advanced SIMD
    pub fn comprehensive_stats_advanced(
        &self,
        data: &ArrayView1<F>,
    ) -> StatsResult<ComprehensiveStats<F>> {
        checkarray_finite(data, "data")?;

        if data.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Data cannot be empty".to_string(),
            ));
        }

        let n = data.len();

        // Use SIMD if data is large enough and SIMD is available
        if n >= self.config.simd_threshold && self.config.chunksize > 1 {
            self.compute_simd_comprehensive(data)
        } else {
            self.compute_scalar_comprehensive(data)
        }
    }

    /// SIMD-optimized comprehensive statistics computation
    fn compute_simd_comprehensive(
        &self,
        data: &ArrayView1<F>,
    ) -> StatsResult<ComprehensiveStats<F>> {
        let n = data.len();
        let chunksize = self.config.chunksize;
        let n_chunks = n / chunksize;
        let remainder = n % chunksize;

        // Initialize accumulators
        let mut sum_acc = F::zero();
        let mut sum_sq_acc = F::zero();
        let mut sum_cube_acc = F::zero();
        let mut sum_quad_acc = F::zero();
        let mut min_val = F::infinity();
        let mut max_val = F::neg_infinity();

        // Process chunks with SIMD
        for i in 0..n_chunks {
            let start = i * chunksize;
            let end = start + chunksize;
            let chunk = data.slice(ndarray::s![start..end]);

            // Use SIMD operations from scirs2-core
            let chunk_sum = F::simd_sum(&chunk);
            let chunk_sq = F::simd_mul(&chunk, &chunk);
            let chunk_sum_sq = F::simd_sum(&chunk_sq.view());
            let chunk_cube = F::simd_mul(&chunk_sq.view(), &chunk);
            let chunk_sum_cube = F::simd_sum(&chunk_cube.view());
            let chunk_quad = F::simd_mul(&chunk_sq.view(), &chunk_sq.view());
            let chunk_sum_quad = F::simd_sum(&chunk_quad.view());
            let chunk_min = F::simd_min_element(&chunk);
            let chunk_max = F::simd_max_element(&chunk);

            sum_acc = sum_acc + chunk_sum;
            sum_sq_acc = sum_sq_acc + chunk_sum_sq;
            sum_cube_acc = sum_cube_acc + chunk_sum_cube;
            sum_quad_acc = sum_quad_acc + chunk_sum_quad;
            min_val = if chunk_min < min_val {
                chunk_min
            } else {
                min_val
            };
            max_val = if chunk_max > max_val {
                chunk_max
            } else {
                max_val
            };
        }

        // Handle remainder with scalar operations
        if remainder > 0 {
            let start = n_chunks * chunksize;
            for i in start..n {
                let val = data[i];
                sum_acc = sum_acc + val;
                sum_sq_acc = sum_sq_acc + val * val;
                sum_cube_acc = sum_cube_acc + val * val * val;
                sum_quad_acc = sum_quad_acc + val * val * val * val;
                min_val = if val < min_val { val } else { min_val };
                max_val = if val > max_val { val } else { max_val };
            }
        }

        // Compute final statistics
        let n_f = F::from(n).unwrap();
        let mean = sum_acc / n_f;
        let variance = (sum_sq_acc / n_f) - (mean * mean);
        let std_dev = variance.sqrt();

        // Compute higher moments
        let m2 = sum_sq_acc / n_f - mean * mean;
        let m3 = sum_cube_acc / n_f - F::from(3).unwrap() * mean * m2 - mean * mean * mean;
        let m4 = sum_quad_acc / n_f
            - F::from(4).unwrap() * mean * m3
            - F::from(6).unwrap() * mean * mean * m2
            - mean * mean * mean * mean;

        let skewness = if m2 > F::zero() {
            m3 / (m2 * m2.sqrt())
        } else {
            F::zero()
        };

        let kurtosis = if m2 > F::zero() {
            m4 / (m2 * m2) - F::from(3).unwrap()
        } else {
            F::zero()
        };

        Ok(ComprehensiveStats {
            mean,
            variance,
            std_dev,
            skewness,
            kurtosis,
            min: min_val,
            max: max_val,
            range: max_val - min_val,
            count: n,
        })
    }

    /// Scalar fallback for comprehensive statistics
    fn compute_scalar_comprehensive(
        &self,
        data: &ArrayView1<F>,
    ) -> StatsResult<ComprehensiveStats<F>> {
        let n = data.len();
        let n_f = F::from(n).unwrap();

        let sum: F = data.iter().copied().sum();
        let mean = sum / n_f;

        let mut sum_sq = F::zero();
        let mut sum_cube = F::zero();
        let mut sum_quad = F::zero();
        let mut min_val = F::infinity();
        let mut max_val = F::neg_infinity();

        for &val in data.iter() {
            let diff = val - mean;
            sum_sq = sum_sq + diff * diff;
            sum_cube = sum_cube + diff * diff * diff;
            sum_quad = sum_quad + diff * diff * diff * diff;
            min_val = if val < min_val { val } else { min_val };
            max_val = if val > max_val { val } else { max_val };
        }

        let variance = sum_sq / n_f;
        let std_dev = variance.sqrt();

        let m2 = variance;
        let m3 = sum_cube / n_f;
        let m4 = sum_quad / n_f;

        let skewness = if m2 > F::zero() {
            m3 / (m2 * m2.sqrt())
        } else {
            F::zero()
        };

        let kurtosis = if m2 > F::zero() {
            m4 / (m2 * m2) - F::from(3).unwrap()
        } else {
            F::zero()
        };

        Ok(ComprehensiveStats {
            mean,
            variance,
            std_dev,
            skewness,
            kurtosis,
            min: min_val,
            max: max_val,
            range: max_val - min_val,
            count: n,
        })
    }

    /// Optimized SIMD-optimized matrix operations
    pub fn matrix_stats_advanced(
        &self,
        matrix: &ArrayView2<F>,
    ) -> StatsResult<MatrixStatsResult<F>> {
        checkarray_finite(matrix, "matrix")?;

        if matrix.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Matrix cannot be empty".to_string(),
            ));
        }

        let (rows, cols) = matrix.dim();

        // Compute row-wise statistics using SIMD
        let mut row_stats = Vec::with_capacity(rows);
        for i in 0..rows {
            let row = matrix.row(i);
            let stats = self.comprehensive_stats_advanced(&row)?;
            row_stats.push(stats);
        }

        // Compute column-wise statistics using SIMD
        let mut col_stats = Vec::with_capacity(cols);
        for j in 0..cols {
            let col = matrix.column(j);
            let stats = self.comprehensive_stats_advanced(&col)?;
            col_stats.push(stats);
        }

        // Compute overall matrix statistics
        let flattened = matrix.iter().copied().collect::<Array1<F>>();
        let overall_stats = self.comprehensive_stats_advanced(&flattened.view())?;

        Ok(MatrixStatsResult {
            row_stats,
            col_stats,
            overall_stats,
            shape: (rows, cols),
        })
    }

    /// SIMD-optimized correlation matrix computation
    pub fn correlation_matrix_advanced(&self, matrix: &ArrayView2<F>) -> StatsResult<Array2<F>> {
        checkarray_finite(matrix, "matrix")?;

        let (_n_samples_, n_features) = matrix.dim();

        if n_features < 2 {
            return Err(StatsError::InvalidArgument(
                "At least 2 features required for correlation matrix".to_string(),
            ));
        }

        let mut corr_matrix = Array2::zeros((n_features, n_features));

        // Compute means using SIMD
        let mut means = Array1::zeros(n_features);
        for j in 0..n_features {
            let col = matrix.column(j);
            means[j] = F::simd_mean(&col);
        }

        // Compute correlation coefficients using SIMD
        for i in 0..n_features {
            for j in i..n_features {
                if i == j {
                    corr_matrix[[i, j]] = F::one();
                } else {
                    let col_i = matrix.column(i);
                    let col_j = matrix.column(j);

                    // Compute correlation using SIMD operations
                    let _n = F::from(col_i.len()).unwrap();
                    let mean_i_vec = Array1::from_elem(col_i.len(), means[i]);
                    let mean_j_vec = Array1::from_elem(col_j.len(), means[j]);

                    let dev_i = F::simd_sub(&col_i, &mean_i_vec.view());
                    let dev_j = F::simd_sub(&col_j, &mean_j_vec.view());

                    let numerator = F::simd_sum(&F::simd_mul(&dev_i.view(), &dev_j.view()).view());
                    let sum_sq_i = F::simd_sum(&F::simd_mul(&dev_i.view(), &dev_i.view()).view());
                    let sum_sq_j = F::simd_sum(&F::simd_mul(&dev_j.view(), &dev_j.view()).view());

                    let denominator = (sum_sq_i * sum_sq_j).sqrt();
                    let corr = if denominator > F::zero() {
                        numerator / denominator
                    } else {
                        F::zero()
                    };

                    corr_matrix[[i, j]] = corr;
                    corr_matrix[[j, i]] = corr;
                }
            }
        }

        Ok(corr_matrix)
    }

    /// SIMD-optimized bootstrap sampling with statistics
    pub fn bootstrap_stats_advanced(
        &self,
        data: &ArrayView1<F>,
        n_bootstrap: usize,
        seed: Option<u64>,
    ) -> StatsResult<BootstrapResult<F>> {
        checkarray_finite(data, "data")?;
        check_positive(n_bootstrap, "n_bootstrap")?;

        let n = data.len();
        let mut rng = create_rng(seed);

        let mut bootstrap_means = Array1::zeros(n_bootstrap);
        let mut bootstrap_vars = Array1::zeros(n_bootstrap);
        let mut bootstrap_stds = Array1::zeros(n_bootstrap);

        // Perform _bootstrap sampling with SIMD statistics
        for i in 0..n_bootstrap {
            // Generate _bootstrap sample
            let mut bootstrap_sample = Array1::zeros(n);
            for j in 0..n {
                let idx = rng.gen_range(0..n);
                bootstrap_sample[j] = data[idx];
            }

            // Compute statistics using SIMD
            let stats = self.comprehensive_stats_advanced(&bootstrap_sample.view())?;
            bootstrap_means[i] = stats.mean;
            bootstrap_vars[i] = stats.variance;
            bootstrap_stds[i] = stats.std_dev;
        }

        // Compute confidence intervals
        let mut sorted_means = bootstrap_means.to_owned();
        sorted_means
            .as_slice_mut()
            .unwrap()
            .sort_by(|a, b| a.partial_cmp(b).unwrap());

        let alpha = F::from(0.05).unwrap(); // 95% confidence
        let lower_idx = ((alpha / F::from(2).unwrap()) * F::from(n_bootstrap).unwrap())
            .to_usize()
            .unwrap();
        let upper_idx = ((F::one() - alpha / F::from(2).unwrap()) * F::from(n_bootstrap).unwrap())
            .to_usize()
            .unwrap();

        let mean_ci = (
            sorted_means[lower_idx],
            sorted_means[upper_idx.min(n_bootstrap - 1)],
        );

        Ok(BootstrapResult {
            original_stats: self.comprehensive_stats_advanced(data)?,
            bootstrap_means,
            bootstrap_vars,
            bootstrap_stds,
            mean_ci,
            n_bootstrap,
        })
    }
}

/// Comprehensive statistics result
#[derive(Debug, Clone)]
pub struct ComprehensiveStats<F> {
    pub mean: F,
    pub variance: F,
    pub std_dev: F,
    pub skewness: F,
    pub kurtosis: F,
    pub min: F,
    pub max: F,
    pub range: F,
    pub count: usize,
}

/// Matrix statistics result
#[derive(Debug, Clone)]
pub struct MatrixStatsResult<F> {
    pub row_stats: Vec<ComprehensiveStats<F>>,
    pub col_stats: Vec<ComprehensiveStats<F>>,
    pub overall_stats: ComprehensiveStats<F>,
    pub shape: (usize, usize),
}

/// Bootstrap analysis result
#[derive(Debug, Clone)]
pub struct BootstrapResult<F> {
    pub original_stats: ComprehensiveStats<F>,
    pub bootstrap_means: Array1<F>,
    pub bootstrap_vars: Array1<F>,
    pub bootstrap_stds: Array1<F>,
    pub mean_ci: (F, F),
    pub n_bootstrap: usize,
}

/// Specialized SIMD operations for advanced statistics
pub trait AdvancedSimdOps<F>: SimdUnifiedOps
where
    F: Float
        + NumCast
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    /// SIMD-optimized sum of cubes
    fn simd_sum_cubes(data: &ArrayView1<F>) -> F {
        data.iter().map(|&x| x * x * x).sum()
    }

    /// SIMD-optimized sum of fourth powers
    fn simd_sum_quads(data: &ArrayView1<F>) -> F {
        data.iter().map(|&x| x * x * x * x).sum()
    }

    /// SIMD-optimized correlation coefficient
    fn simd_correlation(x: &ArrayView1<F>, y: &ArrayView1<F>, mean_x: F, meany: F) -> F {
        let n = x.len();
        if n != y.len() {
            return F::zero();
        }

        let _n_f = F::from(n).unwrap();
        let mut sum_xy = F::zero();
        let mut sum_x2 = F::zero();
        let mut sum_y2 = F::zero();

        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - meany;
            sum_xy = sum_xy + dx * dy;
            sum_x2 = sum_x2 + dx * dx;
            sum_y2 = sum_y2 + dy * dy;
        }

        let denom = (sum_x2 * sum_y2).sqrt();
        if denom > F::zero() {
            sum_xy / denom
        } else {
            F::zero()
        }
    }
}

// Implement the advanced SIMD operations for supported types
impl AdvancedSimdOps<f32> for f32 {}
impl AdvancedSimdOps<f64> for f64 {}

/// High-level convenience functions
#[allow(dead_code)]
pub fn advanced_mean_simd<F>(data: &ArrayView1<F>) -> StatsResult<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    let computer = AdvancedSimdStatistics::<F>::new();
    let stats = computer.comprehensive_stats_advanced(data)?;
    Ok(stats.mean)
}

#[allow(dead_code)]
pub fn advanced_std_simd<F>(data: &ArrayView1<F>) -> StatsResult<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    let computer = AdvancedSimdStatistics::<F>::new();
    let stats = computer.comprehensive_stats_advanced(data)?;
    Ok(stats.std_dev)
}

#[allow(dead_code)]
pub fn advanced_comprehensive_simd<F>(data: &ArrayView1<F>) -> StatsResult<ComprehensiveStats<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    let computer = AdvancedSimdStatistics::<F>::new();
    computer.comprehensive_stats_advanced(data)
}

/// Create RNG with optional seed
#[allow(dead_code)]
fn create_rng(seed: Option<u64>) -> impl Rng {
    use rand::{rngs::StdRng, SeedableRng};
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            use std::time::{SystemTime, UNIX_EPOCH};
            let s = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            StdRng::seed_from_u64(s)
        }
    }
}
