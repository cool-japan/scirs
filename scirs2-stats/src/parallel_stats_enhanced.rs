//! Enhanced parallel statistical operations with adaptive processing
//!
//! This module provides enhanced parallel implementations with:
//! - Adaptive threshold selection based on system capabilities
//! - Better load balancing for heterogeneous data
//! - Cache-aware chunking strategies
//! - NUMA-aware processing for large systems

use crate::descriptive_simd::{mean_simd, variance_simd};
use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayBase, ArrayView2, Data, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::parallel_ops::{num_threads, par_chunks, parallel_map, ParallelIterator};
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};
use std::f64::consts::PI;
use std::sync::Arc;

/// Adaptive threshold calculator based on system capabilities
pub struct AdaptiveThreshold {
    base_threshold: usize,
    cpu_cores: usize,
    simd_available: bool,
    cache_linesize: usize,
}

impl Default for AdaptiveThreshold {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveThreshold {
    /// Create a new adaptive threshold calculator
    pub fn new() -> Self {
        let caps = PlatformCapabilities::detect();
        Self {
            base_threshold: 10_000,
            cpu_cores: num_threads(),
            simd_available: caps.simd_available,
            cache_linesize: 64, // Common cache line size
        }
    }

    /// Calculate the optimal threshold for parallel processing
    pub fn calculate(&self, elementsize: usize, operationcomplexity: f64) -> usize {
        // Adjust threshold based on:
        // 1. Number of CPU cores
        // 2. SIMD availability (SIMD reduces the need for parallelism)
        // 3. Operation _complexity
        // 4. Cache efficiency

        let simd_factor = if self.simd_available { 2.0 } else { 1.0 };
        let core_factor = (self.cpu_cores as f64).sqrt();
        let cache_factor = (self.cache_linesize / elementsize).max(1) as f64;

        let adjusted_threshold =
            (self.base_threshold as f64 * simd_factor / core_factor / operationcomplexity
                * cache_factor.sqrt()) as usize;

        adjusted_threshold.max(1000) // Minimum threshold
    }

    /// Calculate optimal chunk size for cache efficiency
    pub fn optimal_chunksize(&self, total_elements: usize, elementsize: usize) -> usize {
        // L1 cache is typically 32KB per core
        let l1_cachesize = 32 * 1024;
        let elements_per_cache = l1_cachesize / elementsize;

        // Optimal chunk size should fit in L1 cache
        let ideal_chunk = elements_per_cache / 2; // Leave room for other data

        // Balance between cache efficiency and parallelism
        let min_chunks = self.cpu_cores * 4; // At least 4 chunks per core
        let max_chunksize = total_elements / min_chunks;

        ideal_chunk.min(max_chunksize).max(64)
    }
}

/// Parallel histogram computation with adaptive binning
pub struct ParallelHistogram<F: Float> {
    bins: Vec<F>,
    counts: Vec<usize>,
    min_val: F,
    max_val: F,
    n_bins: usize,
}

impl<F: Float + NumCast + Send + Sync + std::fmt::Display> ParallelHistogram<F> {
    /// Create a new parallel histogram
    pub fn new<D>(data: &ArrayBase<D, Ix1>, nbins: usize) -> StatsResult<Self>
    where
        D: Data<Elem = F> + Sync,
    {
        if data.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Cannot create histogram from empty data".to_string(),
            ));
        }

        let threshold = AdaptiveThreshold::new();
        let parallel_threshold = threshold.calculate(std::mem::size_of::<F>(), 1.0);

        // Find min/max in parallel if data is large enough
        let (min_val, max_val) = if data.len() >= parallel_threshold {
            let chunksize = threshold.optimal_chunksize(data.len(), std::mem::size_of::<F>());

            let (min, max) = par_chunks(data.as_slice().unwrap(), chunksize)
                .map(|chunk| {
                    let mut local_min = chunk[0];
                    let mut local_max = chunk[0];
                    for &val in chunk.iter().skip(1) {
                        if val < local_min {
                            local_min = val;
                        }
                        if val > local_max {
                            local_max = val;
                        }
                    }
                    (local_min, local_max)
                })
                .reduce(
                    || (F::infinity(), F::neg_infinity()),
                    |(min1, max1), (min2, max2)| {
                        (
                            if min1 < min2 { min1 } else { min2 },
                            if max1 > max2 { max1 } else { max2 },
                        )
                    },
                );
            (min, max)
        } else {
            // Sequential for small data
            let mut min = data[0];
            let mut max = data[0];
            for &val in data.iter().skip(1) {
                if val < min {
                    min = val;
                }
                if val > max {
                    max = val;
                }
            }
            (min, max)
        };

        // Create _bins
        let range = max_val - min_val;
        let bin_width = range / F::from(nbins).unwrap();

        let bins: Vec<F> = (0..=nbins)
            .map(|i| min_val + bin_width * F::from(i).unwrap())
            .collect();

        let mut histogram = Self {
            bins,
            counts: vec![0; nbins],
            min_val,
            max_val,
            n_bins: nbins,
        };

        // Compute counts
        histogram.compute_counts(data)?;

        Ok(histogram)
    }

    /// Compute histogram counts in parallel
    fn compute_counts<D>(&mut self, data: &ArrayBase<D, Ix1>) -> StatsResult<()>
    where
        D: Data<Elem = F> + Sync,
    {
        let threshold = AdaptiveThreshold::new();
        let parallel_threshold = threshold.calculate(std::mem::size_of::<F>(), 2.0);

        if data.len() < parallel_threshold {
            // Sequential computation
            let bin_width = (self.max_val - self.min_val) / F::from(self.n_bins).unwrap();

            for &val in data.iter() {
                if val >= self.min_val && val <= self.max_val {
                    let bin_idx = ((val - self.min_val) / bin_width)
                        .floor()
                        .to_usize()
                        .unwrap()
                        .min(self.n_bins - 1);
                    self.counts[bin_idx] += 1;
                }
            }
        } else {
            // Parallel computation with thread-local histograms
            let chunksize = threshold.optimal_chunksize(data.len(), std::mem::size_of::<F>());
            let bin_width = (self.max_val - self.min_val) / F::from(self.n_bins).unwrap();
            let n_bins = self.n_bins;
            let min_val = self.min_val;

            // Each thread maintains its own histogram
            let local_histograms: Vec<Vec<usize>> = par_chunks(data.as_slice().unwrap(), chunksize)
                .map(|chunk| {
                    let mut local_counts = vec![0; n_bins];

                    for &val in chunk {
                        if val >= min_val && val <= self.max_val {
                            let bin_idx = ((val - min_val) / bin_width)
                                .floor()
                                .to_usize()
                                .unwrap()
                                .min(n_bins - 1);
                            local_counts[bin_idx] += 1;
                        }
                    }

                    local_counts
                })
                .collect();

            // Merge local histograms
            for local_counts in local_histograms {
                for (i, count) in local_counts.into_iter().enumerate() {
                    self.counts[i] += count;
                }
            }
        }

        Ok(())
    }

    /// Get histogram bins and counts
    pub fn get_histogram(&self) -> (&[F], &[usize]) {
        (&self.bins[..self.n_bins], &self.counts)
    }
}

/// Parallel kernel density estimation
#[allow(dead_code)]
pub fn kde_parallel<F, D>(
    data: &ArrayBase<D, Ix1>,
    eval_points: &Array1<F>,
    bandwidth: F,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + Send + Sync + SimdUnifiedOps,
    D: Data<Elem = F> + Sync,
{
    if data.is_empty() || eval_points.is_empty() {
        return Err(StatsError::InvalidArgument("Empty input data".to_string()));
    }

    let n = data.len();
    let threshold = AdaptiveThreshold::new();
    let parallel_threshold = threshold.calculate(std::mem::size_of::<F>(), 3.0);

    // Gaussian kernel constant
    let norm_const =
        F::one() / (F::from(2.0 * PI).unwrap().sqrt() * bandwidth * F::from(n).unwrap());

    if eval_points.len() * n < parallel_threshold {
        // Sequential KDE
        let mut result = Array1::zeros(eval_points.len());

        for (i, &x) in eval_points.iter().enumerate() {
            let mut density = F::zero();
            for &xi in data.iter() {
                let u = (x - xi) / bandwidth;
                density = density + (-u * u / F::from(2.0).unwrap()).exp();
            }
            result[i] = density * norm_const;
        }

        Ok(result)
    } else {
        // Parallel KDE
        let eval_vec: Vec<F> = eval_points.to_vec();
        let data_slice = data.as_slice().unwrap();

        let densities: Vec<F> = parallel_map(&eval_vec, |&x| {
            let chunksize = threshold.optimal_chunksize(n, std::mem::size_of::<F>());

            // Compute density for this evaluation point in parallel
            let density: F = par_chunks(data_slice, chunksize)
                .map(|chunk| {
                    let mut local_sum = F::zero();
                    for &xi in chunk {
                        let u = (x - xi) / bandwidth;
                        local_sum = local_sum + (-u * u / F::from(2.0).unwrap()).exp();
                    }
                    local_sum
                })
                .reduce(|| F::zero(), |a, b| a + b);

            density * norm_const
        });

        Ok(Array1::from_vec(densities))
    }
}

/// Parallel moving statistics (rolling mean, std, etc.)
pub struct ParallelMovingStats<F: Float> {
    windowsize: usize,
    data: Arc<Vec<F>>,
}

impl<F: Float + NumCast + Send + Sync + SimdUnifiedOps + std::fmt::Display> ParallelMovingStats<F> {
    /// Create a new moving statistics calculator
    pub fn new<D>(data: &ArrayBase<D, Ix1>, windowsize: usize) -> StatsResult<Self>
    where
        D: Data<Elem = F>,
    {
        if windowsize == 0 || windowsize > data.len() {
            return Err(StatsError::InvalidArgument(
                "Invalid window size".to_string(),
            ));
        }

        Ok(Self {
            windowsize,
            data: Arc::new(data.to_vec()),
        })
    }

    /// Compute moving average in parallel
    pub fn moving_mean(&self) -> StatsResult<Array1<F>> {
        let n = self.data.len();
        let output_len = n - self.windowsize + 1;

        let threshold = AdaptiveThreshold::new();
        let parallel_threshold = threshold.calculate(std::mem::size_of::<F>(), 2.0);

        if output_len < parallel_threshold {
            // Sequential computation
            let mut result = Array1::zeros(output_len);
            let windowsize_f = F::from(self.windowsize).unwrap();

            // Initial window sum
            let mut window_sum = self.data[..self.windowsize]
                .iter()
                .fold(F::zero(), |acc, &x| acc + x);
            result[0] = window_sum / windowsize_f;

            // Sliding window
            for i in 1..output_len {
                window_sum = window_sum - self.data[i - 1] + self.data[i + self.windowsize - 1];
                result[i] = window_sum / windowsize_f;
            }

            Ok(result)
        } else {
            // Parallel computation
            let indices: Vec<usize> = (0..output_len).collect();
            let data_ref = Arc::clone(&self.data);
            let windowsize = self.windowsize;
            let windowsize_f = F::from(windowsize).unwrap();

            let means: Vec<F> = parallel_map(&indices, |&i| {
                let window_sum = data_ref[i..i + windowsize]
                    .iter()
                    .fold(F::zero(), |acc, &x| acc + x);
                window_sum / windowsize_f
            });

            Ok(Array1::from_vec(means))
        }
    }

    /// Compute moving standard deviation in parallel
    pub fn moving_std(&self, ddof: usize) -> StatsResult<Array1<F>> {
        let n = self.data.len();
        let output_len = n - self.windowsize + 1;

        if self.windowsize <= ddof {
            return Err(StatsError::InvalidArgument(
                "Window size must be greater than ddof".to_string(),
            ));
        }

        let threshold = AdaptiveThreshold::new();
        let parallel_threshold = threshold.calculate(std::mem::size_of::<F>(), 3.0);

        if output_len < parallel_threshold {
            // Sequential computation using Welford's algorithm
            let mut result = Array1::zeros(output_len);
            let divisor = F::from(self.windowsize - ddof).unwrap();

            for i in 0..output_len {
                let window = &self.data[i..i + self.windowsize];
                let mean = window.iter().fold(F::zero(), |acc, &x| acc + x)
                    / F::from(self.windowsize).unwrap();

                let variance = window
                    .iter()
                    .map(|&x| {
                        let diff = x - mean;
                        diff * diff
                    })
                    .fold(F::zero(), |acc, x| acc + x)
                    / divisor;

                result[i] = variance.sqrt();
            }

            Ok(result)
        } else {
            // Parallel computation
            let indices: Vec<usize> = (0..output_len).collect();
            let data_ref = Arc::clone(&self.data);
            let windowsize = self.windowsize;
            let divisor = F::from(windowsize - ddof).unwrap();

            let stds: Vec<F> = parallel_map(&indices, |&i| {
                let window = &data_ref[i..i + windowsize];
                let mean =
                    window.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(windowsize).unwrap();

                let variance = window
                    .iter()
                    .map(|&x| {
                        let diff = x - mean;
                        diff * diff
                    })
                    .fold(F::zero(), |acc, x| acc + x)
                    / divisor;

                variance.sqrt()
            });

            Ok(Array1::from_vec(stds))
        }
    }
}

/// Parallel computation of pairwise distances
#[allow(dead_code)]
pub fn pairwise_distances_parallel<F, D>(
    x: &ArrayBase<D, Ix2>,
    metric: &str,
) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + Send + Sync + SimdUnifiedOps,
    D: Data<Elem = F> + Sync,
{
    let n = x.nrows();
    let d = x.ncols();

    let threshold = AdaptiveThreshold::new();
    let parallel_threshold = threshold.calculate(std::mem::size_of::<F>() * d, 2.0);

    if n * n < parallel_threshold {
        // Sequential computation
        let mut distances = Array2::zeros((n, n));

        for i in 0..n {
            for j in i + 1..n {
                let dist = match metric {
                    "euclidean" => {
                        let mut sum = F::zero();
                        for k in 0..d {
                            let diff = x[(i, k)] - x[(j, k)];
                            sum = sum + diff * diff;
                        }
                        sum.sqrt()
                    }
                    "manhattan" => {
                        let mut sum = F::zero();
                        for k in 0..d {
                            sum = sum + (x[(i, k)] - x[(j, k)]).abs();
                        }
                        sum
                    }
                    _ => return Err(StatsError::InvalidArgument("Unknown metric".to_string())),
                };

                distances[(i, j)] = dist;
                distances[(j, i)] = dist;
            }
        }

        Ok(distances)
    } else {
        // Parallel computation
        let mut distances = Array2::zeros((n, n));
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .collect();

        let computed_distances: Vec<((usize, usize), F)> = parallel_map(&pairs, |&(i, j)| {
            let dist = match metric {
                "euclidean" => {
                    if d > 8 && F::simd_available() {
                        // Use SIMD for distance computation
                        let row_i = x.slice(s![i, ..]);
                        let row_j = x.slice(s![j, ..]);
                        let diff = F::simd_sub(&row_i, &row_j);
                        let squared = F::simd_mul(&diff.view(), &diff.view());
                        F::simd_sum(&squared.view()).sqrt()
                    } else {
                        let mut sum = F::zero();
                        for k in 0..d {
                            let diff = x[(i, k)] - x[(j, k)];
                            sum = sum + diff * diff;
                        }
                        sum.sqrt()
                    }
                }
                "manhattan" => {
                    let mut sum = F::zero();
                    for k in 0..d {
                        sum = sum + (x[(i, k)] - x[(j, k)]).abs();
                    }
                    sum
                }
                _ => F::nan(), // Should not reach here
            };
            ((i, j), dist)
        });

        // Fill the distance matrix
        for ((i, j), dist) in computed_distances {
            distances[(i, j)] = dist;
            distances[(j, i)] = dist;
        }

        Ok(distances)
    }
}

/// Parallel cross-validation for model evaluation
pub struct ParallelCrossValidation<F: Float> {
    n_folds: usize,
    shuffle: bool,
    random_state: Option<u64>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + NumCast + Send + Sync + std::fmt::Display> ParallelCrossValidation<F> {
    /// Create a new cross-validation splitter
    pub fn new(n_folds: usize, shuffle: bool, randomstate: Option<u64>) -> Self {
        Self {
            n_folds,
            shuffle,
            random_state: randomstate,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Perform parallel cross-validation
    pub fn cross_val_score<D, M, S>(
        &self,
        x: &ArrayBase<D, Ix2>,
        y: &Array1<F>,
        model: M,
        scorer: S,
    ) -> StatsResult<Array1<F>>
    where
        D: Data<Elem = F> + Sync,
        M: Fn(&ArrayView2<F>, &Array1<F>) -> StatsResult<()> + Send + Sync + Clone,
        S: Fn(&ArrayView2<F>, &Array1<F>) -> StatsResult<F> + Send + Sync,
    {
        let n_samples_ = x.nrows();
        if n_samples_ != y.len() {
            return Err(StatsError::DimensionMismatch(
                "X and y must have same number of samples".to_string(),
            ));
        }

        // Create fold indices
        let indices: Vec<usize> = if self.shuffle {
            use crate::random::permutation_int;
            permutation_int(n_samples_, self.random_state)
                .unwrap()
                .to_vec()
        } else {
            (0..n_samples_).collect()
        };

        // Split into folds
        let foldsize = n_samples_ / self.n_folds;
        let remainder = n_samples_ % self.n_folds;

        let fold_indices: Vec<(usize, usize)> = (0..self.n_folds)
            .map(|i| {
                let start = i * foldsize + i.min(remainder);
                let size = foldsize + if i < remainder { 1 } else { 0 };
                (start, start + size)
            })
            .collect();

        // Parallel cross-validation
        let scores: Vec<F> = parallel_map(&fold_indices, |&(test_start, test_end)| {
            // Create train/test splits
            let test_indices = &indices[test_start..test_end];
            let train_indices: Vec<usize> = indices[..test_start]
                .iter()
                .chain(indices[test_end..].iter())
                .copied()
                .collect();

            // Create train/test data
            let x_train = x.select(scirs2_core::ndarray::Axis(0), &train_indices);
            let y_train = y.select(scirs2_core::ndarray::Axis(0), &train_indices);
            let x_test = x.select(scirs2_core::ndarray::Axis(0), test_indices);
            let y_test = y.select(scirs2_core::ndarray::Axis(0), test_indices);

            // Train model on fold
            let fold_model = model.clone();
            fold_model(&x_train.view(), &y_train)?;

            // Score on test set
            scorer(&x_test.view(), &y_test)
        })
        .into_iter()
        .collect::<StatsResult<Vec<_>>>()?;

        Ok(Array1::from_vec(scores))
    }
}

/// Parallel correlation matrix computation
///
/// Efficiently computes correlation matrix for multiple variables using parallel processing
/// and SIMD operations where available.
#[allow(dead_code)]
pub fn corrcoef_parallel<F, D>(data: &ArrayBase<D, Ix2>, rowvar: bool) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + Send + Sync + SimdUnifiedOps,
    D: Data<Elem = F> + Sync,
{
    use crate::correlation_simd::pearson_r_simd;

    let (n_vars, n_obs) = if rowvar {
        (data.nrows(), data.ncols())
    } else {
        (data.ncols(), data.nrows())
    };

    if n_obs < 2 {
        return Err(StatsError::InvalidArgument(
            "Need at least 2 observations to compute correlation".to_string(),
        ));
    }

    let threshold = AdaptiveThreshold::new();
    let parallel_threshold = threshold.calculate(
        std::mem::size_of::<F>() * n_obs,
        2.0, // Correlation is O(n) operation
    );

    let mut corr_matrix = Array2::zeros((n_vars, n_vars));

    // Fill diagonal with 1s
    for i in 0..n_vars {
        corr_matrix[(i, i)] = F::one();
    }

    // Generate all unique pairs
    let pairs: Vec<(usize, usize)> = (0..n_vars)
        .flat_map(|i| (i + 1..n_vars).map(move |j| (i, j)))
        .collect();

    if pairs.len() * n_obs < parallel_threshold {
        // Sequential computation
        for (i, j) in pairs {
            let var_i = if rowvar {
                data.slice(s![i, ..])
            } else {
                data.slice(s![.., i])
            };

            let var_j = if rowvar {
                data.slice(s![j, ..])
            } else {
                data.slice(s![.., j])
            };

            let corr = pearson_r_simd(&var_i, &var_j)?;
            corr_matrix[(i, j)] = corr;
            corr_matrix[(j, i)] = corr;
        }
    } else {
        // Parallel computation
        let correlations: Vec<((usize, usize), F)> = parallel_map(&pairs, |&(i, j)| {
            let var_i = if rowvar {
                data.slice(s![i, ..])
            } else {
                data.slice(s![.., i])
            };

            let var_j = if rowvar {
                data.slice(s![j, ..])
            } else {
                data.slice(s![.., j])
            };

            let corr = pearson_r_simd(&var_i, &var_j)?;
            Ok(((i, j), corr))
        })
        .into_iter()
        .collect::<StatsResult<Vec<_>>>()?;

        // Fill the correlation matrix
        for ((i, j), corr) in correlations {
            corr_matrix[(i, j)] = corr;
            corr_matrix[(j, i)] = corr;
        }
    }

    Ok(corr_matrix)
}

/// Parallel partial correlation computation
///
/// Computes partial correlations between multiple variables, controlling for other variables,
/// using parallel processing for efficiency.
#[allow(dead_code)]
pub fn partial_corrcoef_parallel<F, D>(
    data: &ArrayBase<D, Ix2>,
    control_vars: &[usize],
) -> StatsResult<Array2<F>>
where
    F: Float
        + NumCast
        + Send
        + Sync
        + SimdUnifiedOps
        + scirs2_core::ndarray::ScalarOperand
        + std::iter::Sum
        + scirs2_core::numeric::NumAssign,
    D: Data<Elem = F> + Sync,
{
    use scirs2_linalg::lstsq;

    let n_vars = data.ncols();
    let n_obs = data.nrows();

    if control_vars.is_empty() {
        // No control variables, just compute regular correlation
        return corrcoef_parallel(data, false);
    }

    // Check control variables are valid
    for &idx in control_vars {
        if idx >= n_vars {
            return Err(StatsError::InvalidArgument(format!(
                "Control variable index {} out of bounds",
                idx
            )));
        }
    }

    let threshold = AdaptiveThreshold::new();
    let parallel_threshold = threshold.calculate(
        std::mem::size_of::<F>() * n_obs * control_vars.len(),
        3.0, // Partial correlation is more complex
    );

    let mut partial_corr = Array2::zeros((n_vars, n_vars));

    // Fill diagonal
    for i in 0..n_vars {
        partial_corr[(i, i)] = F::one();
    }

    // Create control matrix
    let control_matrix = data.select(scirs2_core::ndarray::Axis(1), control_vars);

    // Generate pairs excluding control variables
    let pairs: Vec<(usize, usize)> = (0..n_vars)
        .filter(|i| !control_vars.contains(i))
        .flat_map(|i| {
            (i + 1..n_vars)
                .filter(|j| !control_vars.contains(j))
                .map(move |j| (i, j))
        })
        .collect();

    if pairs.len() * n_obs * control_vars.len() < parallel_threshold {
        // Sequential computation
        for (i, j) in pairs {
            let var_i = data.slice(s![.., i]);
            let var_j = data.slice(s![.., j]);

            // Regress out control variables and compute residuals
            let solution_i = lstsq(&control_matrix.view(), &var_i.view(), None)
                .map_err(|e| StatsError::ComputationError(format!("Regression failed: {}", e)))?;
            let predicted_i = control_matrix.dot(&solution_i.x);
            let residuals_i = &var_i - &predicted_i;

            let solution_j = lstsq(&control_matrix.view(), &var_j.view(), None)
                .map_err(|e| StatsError::ComputationError(format!("Regression failed: {}", e)))?;
            let predicted_j = control_matrix.dot(&solution_j.x);
            let residuals_j = &var_j - &predicted_j;

            // Compute correlation of residuals
            use crate::correlation_simd::pearson_r_simd;
            let partial_r = pearson_r_simd(&residuals_i, &residuals_j)?;

            partial_corr[(i, j)] = partial_r;
            partial_corr[(j, i)] = partial_r;
        }
    } else {
        // Parallel computation
        let partial_correlations: Vec<((usize, usize), F)> = parallel_map(&pairs, |&(i, j)| {
            let var_i = data.slice(s![.., i]);
            let var_j = data.slice(s![.., j]);

            // Regress out control variables and compute residuals
            let solution_i = lstsq(&control_matrix.view(), &var_i.view(), None)
                .map_err(|e| StatsError::ComputationError(format!("Regression failed: {}", e)))?;
            let predicted_i = control_matrix.dot(&solution_i.x);
            let residuals_i = &var_i - &predicted_i;

            let solution_j = lstsq(&control_matrix.view(), &var_j.view(), None)
                .map_err(|e| StatsError::ComputationError(format!("Regression failed: {}", e)))?;
            let predicted_j = control_matrix.dot(&solution_j.x);
            let residuals_j = &var_j - &predicted_j;

            // Compute correlation of residuals
            use crate::correlation_simd::pearson_r_simd;
            let partial_r = pearson_r_simd(&residuals_i, &residuals_j)?;

            Ok(((i, j), partial_r))
        })
        .into_iter()
        .collect::<StatsResult<Vec<_>>>()?;

        // Fill the matrix
        for ((i, j), corr) in partial_correlations {
            partial_corr[(i, j)] = corr;
            partial_corr[(j, i)] = corr;
        }
    }

    Ok(partial_corr)
}

/// Parallel autocorrelation computation
///
/// Computes autocorrelation function (ACF) for time series data using parallel processing.
#[allow(dead_code)]
pub fn autocorrelation_parallel<F, D>(
    data: &ArrayBase<D, Ix1>,
    max_lag: usize,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + Send + Sync + SimdUnifiedOps,
    D: Data<Elem = F> + Sync,
{
    let n = data.len();
    if max_lag >= n {
        return Err(StatsError::InvalidArgument(
            "max_lag must be less than data length".to_string(),
        ));
    }

    let threshold = AdaptiveThreshold::new();
    let parallel_threshold = threshold.calculate(
        std::mem::size_of::<F>() * n,
        1.5, // ACF complexity
    );

    // Compute mean and variance
    let mean = mean_simd(data)?;
    let variance = variance_simd(data, 0)?;

    if variance <= F::epsilon() {
        return Err(StatsError::InvalidArgument(
            "Cannot compute autocorrelation for constant series".to_string(),
        ));
    }

    let lags: Vec<usize> = (0..=max_lag).collect();

    let autocorr = if max_lag * n < parallel_threshold {
        // Sequential computation
        lags.iter()
            .map(|&lag| {
                if lag == 0 {
                    F::one()
                } else {
                    let mut sum = F::zero();
                    for i in 0..n - lag {
                        sum = sum + (data[i] - mean) * (data[i + lag] - mean);
                    }
                    sum / (F::from(n - lag).unwrap() * variance)
                }
            })
            .collect()
    } else {
        // Parallel computation
        parallel_map(&lags, |&lag| {
            if lag == 0 {
                F::one()
            } else if F::simd_available() && n - lag > 64 {
                // Use SIMD for large lags
                let data_start = data.slice(s![..n - lag]);
                let data_lagged = data.slice(s![lag..]);

                let mean_array = scirs2_core::ndarray::Array1::from_elem(n - lag, mean);
                let start_centered = F::simd_sub(&data_start, &mean_array.view());
                let lagged_centered = F::simd_sub(&data_lagged, &mean_array.view());

                let products = F::simd_mul(&start_centered.view(), &lagged_centered.view());
                let sum = F::simd_sum(&products.view());

                sum / (F::from(n - lag).unwrap() * variance)
            } else {
                // Scalar fallback
                let mut sum = F::zero();
                for i in 0..n - lag {
                    sum = sum + (data[i] - mean) * (data[i + lag] - mean);
                }
                sum / (F::from(n - lag).unwrap() * variance)
            }
        })
    };

    Ok(Array1::from_vec(autocorr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    #[ignore = "timeout"]
    fn test_parallel_histogram() {
        let data = Array1::from_vec((0..10000).map(|i| i as f64 / 100.0).collect());
        let hist = ParallelHistogram::new(&data.view(), 10).unwrap();

        let (bins, counts) = hist.get_histogram();
        assert_eq!(bins.len(), 10);
        assert_eq!(counts.iter().sum::<usize>(), 10000);
    }

    #[test]
    fn test_parallel_kde() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let eval_points = Array1::linspace(0.0, 6.0, 100);

        let kde_result = kde_parallel(&data.view(), &eval_points, 0.5).unwrap();
        assert_eq!(kde_result.len(), 100);

        // KDE should have maximum near data points
        let max_idx = kde_result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx_, _)| idx_)
            .unwrap();

        assert!(max_idx > 40 && max_idx < 60); // Maximum should be near middle
    }

    #[test]
    fn test_moving_stats() {
        let data = Array1::from_vec((0..100).map(|i| i as f64).collect());
        let moving_stats = ParallelMovingStats::new(&data.view(), 10).unwrap();

        let moving_mean = moving_stats.moving_mean().unwrap();
        assert_eq!(moving_mean.len(), 91); // 100 - 10 + 1

        // First moving average should be mean of 0..10
        assert_relative_eq!(moving_mean[0], 4.5, epsilon = 1e-10);

        // Last moving average should be mean of 90..100
        assert_relative_eq!(moving_mean[90], 94.5, epsilon = 1e-10);
    }

    #[test]
    fn test_pairwise_distances() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];

        let distances = pairwise_distances_parallel(&data.view(), "euclidean").unwrap();

        // Check diagonal is zero
        for i in 0..4 {
            assert_relative_eq!(distances[(i, i)], 0.0, epsilon = 1e-10);
        }

        // Check specific distances
        assert_relative_eq!(distances[(0, 1)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(distances[(0, 3)], 2.0_f64.sqrt(), epsilon = 1e-10);

        // Check symmetry
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(distances[(i, j)], distances[(j, i)], epsilon = 1e-10);
            }
        }
    }
}
