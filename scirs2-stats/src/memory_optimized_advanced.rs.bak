//! Advanced memory optimization for large-scale statistical computing
//!
//! This module provides memory-aware algorithms that automatically adapt
//! to available memory constraints and optimize data layout for cache efficiency.

use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix2};
use num_traits::{Float, NumCast, One, Zero};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Memory constraints configuration
#[derive(Debug, Clone)]
pub struct MemoryConstraints {
    /// Maximum memory to use (in bytes)
    pub max_memory_bytes: usize,
    /// Preferred chunk size for processing
    pub chunksize: usize,
    /// Use memory mapping for large files
    pub use_memory_mapping: bool,
    /// Enable garbage collection hints
    pub enable_gc_hints: bool,
}

impl Default for MemoryConstraints {
    fn default() -> Self {
        // Default to 1GB max memory, 64KB chunks
        Self {
            max_memory_bytes: 1_024 * 1_024 * 1_024,
            chunksize: 64 * 1024,
            use_memory_mapping: true,
            enable_gc_hints: true,
        }
    }
}

/// Adaptive memory manager that monitors usage and adjusts strategies
pub struct AdaptiveMemoryManager {
    constraints: MemoryConstraints,
    current_usage: Arc<Mutex<usize>>,
    peak_usage: Arc<Mutex<usize>>,
    operation_history: Arc<Mutex<VecDeque<OperationMetrics>>>,
}

#[derive(Debug, Clone)]
pub struct OperationMetrics {
    operation_type: String,
    memory_used: usize,
    processing_time: std::time::Duration,
    chunksize_used: usize,
}

impl AdaptiveMemoryManager {
    pub fn new(constraints: MemoryConstraints) -> Self {
        Self {
            constraints,
            current_usage: Arc::new(Mutex::new(0)),
            peak_usage: Arc::new(Mutex::new(0)),
            operation_history: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
        }
    }

    /// Get optimal chunk size based on current memory usage and data size
    pub fn get_optimal_chunksize(&self, datasize: usize, elementsize: usize) -> usize {
        let current_usage = *self.current_usage.lock().unwrap();
        let available_memory = self
            .constraints
            .max_memory_bytes
            .saturating_sub(current_usage);

        // Use at most 80% of available memory for chunk processing
        let max_chunk_memory = available_memory * 4 / 5;
        let max_chunk_elements = max_chunk_memory / elementsize;

        // Prefer power-of-2 sizes for cache efficiency
        let optimalsize = max_chunk_elements
            .min(datasize)
            .min(self.constraints.chunksize);
        optimalsize.next_power_of_two() / 2 // Round down to nearest power of 2
    }

    /// Record memory usage for an operation
    pub fn record_operation(&self, metrics: OperationMetrics) {
        let mut history = self.operation_history.lock().unwrap();

        // Keep only recent operations
        if history.len() >= 100 {
            history.pop_front();
        }

        history.push_back(metrics.clone());

        // Update peak usage
        let mut peak = self.peak_usage.lock().unwrap();
        *peak = (*peak).max(metrics.memory_used);
    }

    /// Get memory usage statistics
    pub fn get_statistics(&self) -> MemoryStatistics {
        let current_usage = *self.current_usage.lock().unwrap();
        let peak_usage = *self.peak_usage.lock().unwrap();
        let history = self.operation_history.lock().unwrap();

        let avg_memory_per_op = if !history.is_empty() {
            history.iter().map(|m| m.memory_used).sum::<usize>() / history.len()
        } else {
            0
        };

        MemoryStatistics {
            current_usage,
            peak_usage,
            avg_memory_per_operation: avg_memory_per_op,
            operations_completed: history.len(),
            memory_efficiency: if peak_usage > 0 {
                (avg_memory_per_op as f64 / peak_usage as f64) * 100.0
            } else {
                100.0
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    pub current_usage: usize,
    pub peak_usage: usize,
    pub avg_memory_per_operation: usize,
    pub operations_completed: usize,
    pub memory_efficiency: f64,
}

/// Memory-aware correlation matrix computation
///
/// Computes correlation matrices using adaptive chunking based on available memory.
/// For very large matrices, uses block-wise computation to stay within memory constraints.
#[allow(dead_code)]
pub fn corrcoef_memory_aware<F>(
    data: &ArrayView2<F>,
    method: &str,
    manager: &mut AdaptiveMemoryManager,
) -> StatsResult<Array2<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + Copy
        + Send
        + Sync
        + std::iter::Sum<F>
        + std::fmt::Debug
        + std::fmt::Display,
{
    let start_time = std::time::Instant::now();

    checkarray_finite_2d(data, "data")?;

    let (n_obs, n_vars) = data.dim();
    let elementsize = std::mem::size_of::<F>();

    // Estimate memory requirements
    let matrix_memory = n_vars * n_vars * elementsize;
    let temp_memory = n_obs * elementsize * 2; // For column pairs
    let total_estimated = matrix_memory + temp_memory;

    let mut corr_matrix = Array2::<F>::zeros((n_vars, n_vars));

    // Set diagonal to 1.0
    for i in 0..n_vars {
        corr_matrix[[i, i]] = F::one();
    }

    if total_estimated <= manager.constraints.max_memory_bytes {
        // Can fit in memory - use standard approach
        corr_matrix = compute_correlation_matrix_standard(data, method)?;
    } else {
        // Use block-wise computation
        let blocksize = manager.get_optimal_chunksize(n_vars, elementsize * n_vars);
        corr_matrix = compute_correlation_matrix_blocked(data, method, blocksize)?;
    }

    // Record metrics
    let metrics = OperationMetrics {
        operation_type: format!("corrcoef_memory_aware_{}", method),
        memory_used: total_estimated,
        processing_time: start_time.elapsed(),
        chunksize_used: n_vars,
    };
    manager.record_operation(metrics);

    Ok(corr_matrix)
}

/// Cache-oblivious matrix multiplication for large correlation computations
#[allow(dead_code)]
pub fn cache_oblivious_matrix_mult<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    threshold: usize,
) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + Zero + One + Copy + Send + Sync + std::fmt::Display,
{
    let (m, n) = a.dim();
    let (n2, p) = b.dim();

    if n != n2 {
        return Err(StatsError::DimensionMismatch(
            "Matrix dimensions don't match for multiplication".to_string(),
        ));
    }

    let mut result = Array2::<F>::zeros((m, p));

    if m <= threshold && n <= threshold && p <= threshold {
        // Base case: use standard multiplication
        for i in 0..m {
            for j in 0..p {
                let mut sum = F::zero();
                for k in 0..n {
                    sum = sum + a[[i, k]] * b[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }
    } else {
        // Recursive case: divide matrices
        let mid_m = m / 2;
        let _mid_n = n / 2;
        let _mid_p = p / 2;

        // This is a simplified version - full implementation would handle
        // all matrix subdivisions recursively
        if m > threshold {
            let a_top = a.slice(ndarray::s![..mid_m, ..]);
            let a_bottom = a.slice(ndarray::s![mid_m.., ..]);

            let result_top = cache_oblivious_matrix_mult(&a_top, b, threshold)?;
            let result_bottom = cache_oblivious_matrix_mult(&a_bottom, b, threshold)?;

            result
                .slice_mut(ndarray::s![..mid_m, ..])
                .assign(&result_top);
            result
                .slice_mut(ndarray::s![mid_m.., ..])
                .assign(&result_bottom);
        }
    }

    Ok(result)
}

/// Streaming covariance computation for large datasets
#[allow(dead_code)]
pub fn streaming_covariance_matrix<'a, F>(
    data_chunks: impl Iterator<Item = ArrayView2<'a, F>>,
    manager: &mut AdaptiveMemoryManager,
) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + Zero + One + Copy + 'a + std::fmt::Display,
{
    let start_time = std::time::Instant::now();

    let mut n_vars = 0;
    let mut total_obs = 0;
    let mut sum_x = Array1::<F>::zeros(0);
    let mut sum_x2 = Array2::<F>::zeros((0, 0));
    let mut initialized = false;

    for chunk in data_chunks {
        let (chunk_obs, chunk_vars) = chunk.dim();

        if !initialized {
            n_vars = chunk_vars;
            sum_x = Array1::<F>::zeros(n_vars);
            sum_x2 = Array2::<F>::zeros((n_vars, n_vars));
            initialized = true;
        } else if chunk_vars != n_vars {
            return Err(StatsError::DimensionMismatch(
                "All _chunks must have the same number of variables".to_string(),
            ));
        }

        total_obs += chunk_obs;

        // Update sums
        for i in 0..chunk_obs {
            let row = chunk.row(i);

            // Update sum_x
            for j in 0..n_vars {
                sum_x[j] = sum_x[j] + row[j];
            }

            // Update sum_x2 (cross products)
            for j in 0..n_vars {
                for k in j..n_vars {
                    let product = row[j] * row[k];
                    sum_x2[[j, k]] = sum_x2[[j, k]] + product;
                    if j != k {
                        sum_x2[[k, j]] = sum_x2[[k, j]] + product;
                    }
                }
            }
        }
    }

    if total_obs == 0 {
        return Err(StatsError::InvalidArgument("No data provided".to_string()));
    }

    // Compute covariance matrix
    let mut cov_matrix = Array2::<F>::zeros((n_vars, n_vars));
    let n_f = F::from(total_obs).unwrap();

    for i in 0..n_vars {
        for j in 0..n_vars {
            let mean_i = sum_x[i] / n_f;
            let mean_j = sum_x[j] / n_f;
            let cov = (sum_x2[[i, j]] / n_f) - (mean_i * mean_j);
            cov_matrix[[i, j]] = cov;
        }
    }

    // Record metrics
    let memory_used = (n_vars * n_vars + n_vars) * std::mem::size_of::<F>();
    let metrics = OperationMetrics {
        operation_type: "streaming_covariance_matrix".to_string(),
        memory_used,
        processing_time: start_time.elapsed(),
        chunksize_used: n_vars,
    };
    manager.record_operation(metrics);

    Ok(cov_matrix)
}

/// Memory-efficient principal component analysis
#[allow(dead_code)]
pub fn pca_memory_efficient<F>(
    data: &ArrayView2<F>,
    n_components: Option<usize>,
    manager: &mut AdaptiveMemoryManager,
) -> StatsResult<PCAResult<F>>
where
    F: Float + NumCast + Zero + One + Copy + Send + Sync + std::fmt::Debug + std::fmt::Display,
{
    let start_time = std::time::Instant::now();

    let (n_obs, n_vars) = data.dim();
    let n_components = n_components.unwrap_or(n_vars.min(n_obs));

    // Center the data using streaming mean
    let mut means = Array1::<F>::zeros(n_vars);
    for i in 0..n_vars {
        let column = data.column(i);
        means[i] = column.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(n_obs).unwrap();
    }

    // Estimate memory for centered data
    let centereddata_memory = n_obs * n_vars * std::mem::size_of::<F>();

    if centereddata_memory <= manager.constraints.max_memory_bytes / 2 {
        // Can afford to store centered data
        let mut centereddata = Array2::<F>::zeros((n_obs, n_vars));
        for i in 0..n_obs {
            for j in 0..n_vars {
                centereddata[[i, j]] = data[[i, j]] - means[j];
            }
        }

        // Compute covariance matrix
        let cov_matrix = compute_covariance_from_centered(&centereddata.view())?;

        // Eigendecomposition (simplified - would use proper linear algebra library)
        let (eigenvectors, eigenvalues) =
            compute_eigendecomposition(&cov_matrix.view(), n_components)?;

        // Transform data
        let transformed = matrix_multiply(&centereddata.view(), &eigenvectors.view())?;

        let result = PCAResult {
            components: eigenvectors,
            explained_variance: eigenvalues,
            transformeddata: transformed,
            mean: means,
        };

        let metrics = OperationMetrics {
            operation_type: "pca_memory_efficient".to_string(),
            memory_used: centereddata_memory,
            processing_time: start_time.elapsed(),
            chunksize_used: n_vars,
        };
        manager.record_operation(metrics);

        Ok(result)
    } else {
        // Use incremental PCA for very large datasets
        incremental_pca(data, n_components, &means, manager)
    }
}

#[derive(Debug, Clone)]
pub struct PCAResult<F> {
    pub components: Array2<F>,
    pub explained_variance: Array1<F>,
    pub transformeddata: Array2<F>,
    pub mean: Array1<F>,
}

/// Streaming principal component analysis for very large datasets
#[allow(dead_code)]
pub fn streaming_pca_enhanced<'a, F>(
    data_chunks: impl Iterator<Item = ArrayView2<'a, F>>,
    n_components: usize,
    manager: &mut AdaptiveMemoryManager,
) -> StatsResult<PCAResult<F>>
where
    F: Float + NumCast + Zero + One + Copy + 'a + std::fmt::Display,
{
    let start_time = std::time::Instant::now();

    // First pass: compute mean and covariance incrementally
    let mut n_samples_ = 0;
    let mut n_features = 0;
    let mut running_mean = Array1::<F>::zeros(0);
    let mut running_cov = Array2::<F>::zeros((0, 0));
    let mut initialized = false;

    for chunk in data_chunks {
        let (chunk_samples, chunk_features) = chunk.dim();

        if !initialized {
            n_features = chunk_features;
            running_mean = Array1::<F>::zeros(n_features);
            running_cov = Array2::<F>::zeros((n_features, n_features));
            initialized = true;
        } else if chunk_features != n_features {
            return Err(StatsError::DimensionMismatch(
                "All _chunks must have same number of features".to_string(),
            ));
        }

        // Update running statistics using Welford's method
        for i in 0..chunk_samples {
            n_samples_ += 1;
            let row = chunk.row(i);

            // Update mean
            let n_f = F::from(n_samples_).unwrap();
            for j in 0..n_features {
                let delta = row[j] - running_mean[j];
                running_mean[j] = running_mean[j] + delta / n_f;
            }

            // Update covariance (simplified incremental update)
            if n_samples_ > 1 {
                for j in 0..n_features {
                    for k in j..n_features {
                        let prod = (row[j] - running_mean[j]) * (row[k] - running_mean[k]);
                        running_cov[[j, k]] =
                            running_cov[[j, k]] * F::from(n_samples_ - 1).unwrap() / n_f
                                + prod / n_f;
                        if j != k {
                            running_cov[[k, j]] = running_cov[[j, k]];
                        }
                    }
                }
            }
        }
    }

    if n_samples_ == 0 {
        return Err(StatsError::InvalidArgument("No data provided".to_string()));
    }

    // Simplified eigendecomposition (would use proper SVD in production)
    let (components, explained_variance) =
        compute_eigendecomposition(&running_cov.view(), n_components)?;

    // Record memory usage
    let memory_used =
        n_features * n_features * std::mem::size_of::<F>() + n_features * std::mem::size_of::<F>();
    manager.record_operation(OperationMetrics {
        operation_type: "streaming_pca_enhanced".to_string(),
        memory_used,
        processing_time: start_time.elapsed(),
        chunksize_used: manager.constraints.chunksize,
    });

    Ok(PCAResult {
        components,
        explained_variance,
        transformeddata: Array2::zeros((0, 0)), // Would project data in second pass
        mean: running_mean,
    })
}

/// Enhanced streaming histogram computation with adaptive binning
#[allow(dead_code)]
pub fn streaming_histogram_adaptive<'a, F>(
    data_chunks: impl Iterator<Item = ArrayView1<'a, F>>,
    manager: &mut AdaptiveMemoryManager,
) -> StatsResult<(Array1<F>, Array1<usize>)>
where
    F: Float + NumCast + Zero + One + Copy + PartialOrd + 'a + std::fmt::Display,
{
    let start_time = std::time::Instant::now();

    let mut min_val = F::infinity();
    let mut max_val = F::neg_infinity();
    let mut total_count = 0;
    let mut values = Vec::new(); // For adaptive binning

    // First pass: find range and collect sample for bin determination
    for chunk in data_chunks {
        for &value in chunk.iter() {
            if value < min_val {
                min_val = value;
            }
            if value > max_val {
                max_val = value;
            }
            total_count += 1;

            // Reservoir sampling to maintain memory bounds
            if values.len() < manager.constraints.chunksize {
                values.push(value);
            } else {
                let j = {
                    use rand::Rng;
                    let mut rng = rand::rng();
                    rng.gen_range(0..total_count)
                };
                if j < values.len() {
                    values[j] = value;
                }
            }
        }
    }

    if total_count == 0 {
        return Err(StatsError::InvalidArgument("No data provided".to_string()));
    }

    // Adaptive bin count using Freedman-Diaconis rule
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let q75_idx = (values.len() as f64 * 0.75) as usize;
    let q25_idx = (values.len() as f64 * 0.25) as usize;
    let iqr = values[q75_idx] - values[q25_idx];
    let h = F::from(2.0).unwrap() * iqr
        / F::from(total_count as f64)
            .unwrap()
            .powf(F::from(1.0 / 3.0).unwrap());
    let n_bins = if h > F::zero() {
        ((max_val - min_val) / h)
            .to_usize()
            .unwrap_or(50)
            .max(10)
            .min(1000)
    } else {
        50 // Default
    };

    let bin_width = (max_val - min_val) / F::from(n_bins).unwrap();
    let mut bin_edges = Array1::<F>::zeros(n_bins + 1);
    let mut counts = Array1::<usize>::zeros(n_bins);

    // Set bin edges
    for i in 0..=n_bins {
        bin_edges[i] = min_val + F::from(i).unwrap() * bin_width;
    }

    // Second pass would count values into bins (simplified here)
    // In practice, you'd iterate through _chunks again
    for &value in values.iter() {
        if value >= min_val && value <= max_val {
            let bin_idx = ((value - min_val) / bin_width)
                .to_usize()
                .unwrap_or(0)
                .min(n_bins - 1);
            counts[bin_idx] += 1;
        }
    }

    let memory_used = n_bins * (std::mem::size_of::<F>() + std::mem::size_of::<usize>());
    manager.record_operation(OperationMetrics {
        operation_type: "streaming_histogram_adaptive".to_string(),
        memory_used,
        processing_time: start_time.elapsed(),
        chunksize_used: manager.constraints.chunksize,
    });

    Ok((bin_edges, counts))
}

/// Memory-efficient streaming quantile computation using P² algorithm
#[allow(dead_code)]
pub fn streaming_quantiles_p2<'a, F>(
    data_chunks: impl Iterator<Item = ArrayView1<'a, F>>,
    quantiles: &[f64],
    manager: &mut AdaptiveMemoryManager,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + Zero + One + Copy + PartialOrd + 'a + std::fmt::Display,
{
    let start_time = std::time::Instant::now();

    let mut p2_estimators: Vec<P2Estimator<F>> =
        quantiles.iter().map(|&q| P2Estimator::new(q)).collect();

    let mut total_count = 0;

    for chunk in data_chunks {
        for &value in chunk.iter() {
            total_count += 1;
            for estimator in &mut p2_estimators {
                estimator.update(value);
            }
        }
    }

    if total_count == 0 {
        return Err(StatsError::InvalidArgument("No data provided".to_string()));
    }

    let results: Vec<F> = p2_estimators.iter().map(|est| est.quantile()).collect();

    let memory_used = quantiles.len() * std::mem::size_of::<P2Estimator<F>>();
    manager.record_operation(OperationMetrics {
        operation_type: "streaming_quantiles_p2".to_string(),
        memory_used,
        processing_time: start_time.elapsed(),
        chunksize_used: manager.constraints.chunksize,
    });

    Ok(Array1::from_vec(results))
}

/// P² algorithm estimator for streaming quantiles
#[derive(Debug, Clone)]
struct P2Estimator<F> {
    quantile: f64,
    markers: [F; 5],
    positions: [f64; 5],
    desired_positions: [f64; 5],
    increment: [f64; 5],
    count: usize,
}

impl<F> P2Estimator<F>
where
    F: Float + NumCast + Copy + PartialOrd + std::fmt::Display,
{
    fn new(quantile: f64) -> Self {
        let mut estimator = Self {
            quantile,
            markers: [F::zero(); 5],
            positions: [1.0, 2.0, 3.0, 4.0, 5.0],
            desired_positions: [
                1.0,
                1.0 + 2.0 * quantile,
                1.0 + 4.0 * quantile,
                3.0 + 2.0 * quantile,
                5.0,
            ],
            increment: [0.0, quantile / 2.0, quantile, (1.0 + quantile) / 2.0, 1.0],
            count: 0,
        };

        // Initialize desired positions
        estimator.desired_positions[1] = 1.0 + 2.0 * quantile;
        estimator.desired_positions[2] = 1.0 + 4.0 * quantile;
        estimator.desired_positions[3] = 3.0 + 2.0 * quantile;

        estimator
    }

    fn update(&mut self, value: F) {
        self.count += 1;

        if self.count <= 5 {
            // Initialize first 5 values
            self.markers[self.count - 1] = value;
            if self.count == 5 {
                self.markers
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            return;
        }

        // Find cell k such that markers[k] <= value < markers[k+1]
        let mut k = 0;
        for i in 0..4 {
            if value >= self.markers[i] {
                k = i + 1;
            }
        }

        if k == 0 {
            self.markers[0] = value;
            k = 1;
        } else if k == 5 {
            self.markers[4] = value;
            k = 4;
        }

        // Increment positions k through 4
        for i in k..5 {
            self.positions[i] += 1.0;
        }

        // Update desired positions
        for i in 0..5 {
            self.desired_positions[i] += self.increment[i];
        }

        // Adjust heights if necessary
        for i in 1..4 {
            let d = self.desired_positions[i] - self.positions[i];
            if (d >= 1.0 && self.positions[i + 1] - self.positions[i] > 1.0)
                || (d <= -1.0 && self.positions[i - 1] - self.positions[i] < -1.0)
            {
                let d_sign = if d >= 0.0 { 1.0 } else { -1.0 };
                let new_height = self.parabolic_prediction(i, d_sign);

                if self.markers[i - 1] < new_height && new_height < self.markers[i + 1] {
                    self.markers[i] = new_height;
                } else {
                    self.markers[i] = self.linear_prediction(i, d_sign);
                }
                self.positions[i] += d_sign;
            }
        }
    }

    fn parabolic_prediction(&self, i: usize, d: f64) -> F {
        let qi = self.markers[i];
        let qi_prev = self.markers[i - 1];
        let qi_next = self.markers[i + 1];
        let ni = self.positions[i];
        let ni_prev = self.positions[i - 1];
        let ni_next = self.positions[i + 1];

        let d_f = F::from(d).unwrap();
        let ni_f = F::from(ni).unwrap();
        let ni_prev_f = F::from(ni_prev).unwrap();
        let ni_next_f = F::from(ni_next).unwrap();

        let a = d_f / (ni_next_f - ni_prev_f);
        let b1 = (ni_f - ni_prev_f + d_f) * (qi_next - qi) / (ni_next_f - ni_f);
        let b2 = (ni_next_f - ni_f - d_f) * (qi - qi_prev) / (ni_f - ni_prev_f);

        qi + a * (b1 + b2)
    }

    fn linear_prediction(&self, i: usize, d: f64) -> F {
        if d > 0.0 {
            let qi_next = self.markers[i + 1];
            let qi = self.markers[i];
            let ni_next = self.positions[i + 1];
            let ni = self.positions[i];
            qi + (qi_next - qi) * F::from(d / (ni_next - ni)).unwrap()
        } else {
            let qi = self.markers[i];
            let qi_prev = self.markers[i - 1];
            let ni = self.positions[i];
            let ni_prev = self.positions[i - 1];
            qi + (qi_prev - qi) * F::from(-d / (ni - ni_prev)).unwrap()
        }
    }

    fn quantile(&self) -> F {
        if self.count < 5 {
            // For small datasets, use simple sorting
            let mut sorted = self.markers[..self.count.min(5)].to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = (self.quantile * (sorted.len() - 1) as f64) as usize;
            sorted[idx]
        } else {
            self.markers[2] // The middle marker approximates the desired quantile
        }
    }
}

/// Enhanced streaming regression for large datasets with regularization
#[allow(dead_code)]
pub fn streaming_regression_enhanced<'a, F>(
    data_chunks: impl Iterator<Item = (ArrayView2<'a, F>, ArrayView1<'a, F>)>,
    regularization: F,
    manager: &mut AdaptiveMemoryManager,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + Zero + One + Copy + 'a + std::fmt::Display,
{
    let start_time = std::time::Instant::now();

    let mut xtx = Array2::<F>::zeros((0, 0));
    let mut xty = Array1::<F>::zeros(0);
    let mut n_samples_ = 0;
    let mut n_features = 0;
    let mut initialized = false;

    // Accumulate X'X and X'y incrementally
    for (x_chunk, y_chunk) in data_chunks {
        let (chunk_samples, chunk_features) = x_chunk.dim();

        if !initialized {
            n_features = chunk_features;
            xtx = Array2::<F>::zeros((n_features, n_features));
            xty = Array1::<F>::zeros(n_features);
            initialized = true;
        } else if chunk_features != n_features {
            return Err(StatsError::DimensionMismatch(
                "All _chunks must have same number of features".to_string(),
            ));
        }

        if y_chunk.len() != chunk_samples {
            return Err(StatsError::DimensionMismatch(
                "X and y must have same number of samples".to_string(),
            ));
        }

        n_samples_ += chunk_samples;

        // Update X'X
        for i in 0..n_features {
            for j in i..n_features {
                let mut sum = F::zero();
                for k in 0..chunk_samples {
                    sum = sum + x_chunk[[k, i]] * x_chunk[[k, j]];
                }
                xtx[[i, j]] = xtx[[i, j]] + sum;
                if i != j {
                    xtx[[j, i]] = xtx[[i, j]];
                }
            }
        }

        // Update X'y
        for i in 0..n_features {
            let mut sum = F::zero();
            for k in 0..chunk_samples {
                sum = sum + x_chunk[[k, i]] * y_chunk[k];
            }
            xty[i] = xty[i] + sum;
        }
    }

    if n_samples_ == 0 {
        return Err(StatsError::InvalidArgument("No data provided".to_string()));
    }

    // Add regularization to diagonal
    for i in 0..n_features {
        xtx[[i, i]] = xtx[[i, i]] + regularization;
    }

    // Solve (X'X + λI)β = X'y using simplified method
    // In practice, would use proper matrix decomposition
    let coefficients = solve_linear_system(&xtx.view(), &xty.view())?;

    let memory_used =
        n_features * n_features * std::mem::size_of::<F>() + n_features * std::mem::size_of::<F>();
    manager.record_operation(OperationMetrics {
        operation_type: "streaming_regression_enhanced".to_string(),
        memory_used,
        processing_time: start_time.elapsed(),
        chunksize_used: manager.constraints.chunksize,
    });

    Ok(coefficients)
}

/// Simple linear system solver (would use proper LU decomposition in production)
#[allow(dead_code)]
fn solve_linear_system<F>(a: &ArrayView2<F>, b: &ArrayView1<F>) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + Zero + One + Copy + std::fmt::Display,
{
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(StatsError::DimensionMismatch(
            "Matrix dimensions incompatible".to_string(),
        ));
    }

    // Simplified Gaussian elimination (not numerically stable for production)
    let mut aug = Array2::<F>::zeros((n, n + 1));

    // Copy A and b into augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..=n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Make diagonal 1
        let pivot = aug[[i, i]];
        if pivot.abs() < F::from(1e-12).unwrap() {
            return Err(StatsError::ComputationError(
                "Matrix is singular".to_string(),
            ));
        }

        for j in i..=n {
            aug[[i, j]] = aug[[i, j]] / pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[[k, i]];
                for j in i..=n {
                    aug[[k, j]] = aug[[k, j]] - factor * aug[[i, j]];
                }
            }
        }
    }

    // Extract solution
    let mut solution = Array1::<F>::zeros(n);
    for i in 0..n {
        solution[i] = aug[[i, n]];
    }

    Ok(solution)
}

// Helper functions (simplified implementations)

#[allow(dead_code)]
fn compute_correlation_matrix_standard<F>(
    data: &ArrayView2<F>,
    method: &str,
) -> StatsResult<Array2<F>>
where
    F: Float
        + NumCast
        + Zero
        + One
        + Copy
        + std::iter::Sum<F>
        + std::fmt::Debug
        + std::fmt::Display
        + Send
        + Sync
        + scirs2_core::simd_ops::SimdUnifiedOps,
{
    // Use existing corrcoef implementation
    crate::corrcoef(data, method)
}

#[allow(dead_code)]
fn compute_correlation_matrix_blocked<F>(
    data: &ArrayView2<F>,
    method: &str,
    blocksize: usize,
) -> StatsResult<Array2<F>>
where
    F: Float
        + NumCast
        + Zero
        + One
        + Copy
        + std::iter::Sum<F>
        + std::fmt::Debug
        + std::fmt::Display
        + scirs2_core::simd_ops::SimdUnifiedOps,
{
    let (_, n_vars) = data.dim();
    let mut corr_matrix = Array2::<F>::zeros((n_vars, n_vars));

    // Set diagonal
    for i in 0..n_vars {
        corr_matrix[[i, i]] = F::one();
    }

    // Process in blocks
    for i_block in (0..n_vars).step_by(blocksize) {
        let i_end = (i_block + blocksize).min(n_vars);

        for j_block in (i_block..n_vars).step_by(blocksize) {
            let j_end = (j_block + blocksize).min(n_vars);

            // Compute correlations for this block
            for i in i_block..i_end {
                for j in j_block.max(i + 1)..j_end {
                    let col_i = data.column(i);
                    let col_j = data.column(j);

                    let corr = match method {
                        "pearson" => crate::pearson_r(&col_i, &col_j)?,
                        "spearman" => crate::spearman_r(&col_i, &col_j)?,
                        "kendall" => crate::kendall_tau(&col_i, &col_j, "b")?,
                        _ => {
                            return Err(StatsError::InvalidArgument(format!(
                                "Unknown method: {}",
                                method
                            )))
                        }
                    };

                    corr_matrix[[i, j]] = corr;
                    corr_matrix[[j, i]] = corr;
                }
            }
        }
    }

    Ok(corr_matrix)
}

#[allow(dead_code)]
fn compute_covariance_from_centered<F>(data: &ArrayView2<F>) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + Zero + Copy + std::fmt::Display,
{
    let (n_obs, n_vars) = data.dim();
    let mut cov_matrix = Array2::<F>::zeros((n_vars, n_vars));
    let n_f = F::from(n_obs - 1).unwrap(); // Sample covariance

    for i in 0..n_vars {
        for j in i..n_vars {
            let mut cov = F::zero();
            for k in 0..n_obs {
                cov = cov + data[[k, i]] * data[[k, j]];
            }
            cov = cov / n_f;
            cov_matrix[[i, j]] = cov;
            cov_matrix[[j, i]] = cov;
        }
    }

    Ok(cov_matrix)
}

#[allow(dead_code)]
fn compute_eigendecomposition<F>(
    matrix: &ArrayView2<F>,
    n_components: usize,
) -> StatsResult<(Array2<F>, Array1<F>)>
where
    F: Float + NumCast + Zero + One + Copy + std::fmt::Display,
{
    let n = matrix.dim().0;
    let n_components = n_components.min(n);

    // Power iteration method for top eigenvalues/eigenvectors
    // This is a simplified implementation - in practice would use LAPACK
    let mut eigenvalues = Array1::<F>::zeros(n_components);
    let mut eigenvectors = Array2::<F>::zeros((n, n_components));

    for k in 0..n_components {
        // Initialize random vector
        let mut v = Array1::<F>::ones(n);

        // Power iteration
        for _ in 0..100 {
            // Max iterations
            let mut new_v = Array1::<F>::zeros(n);

            // Matrix-vector multiplication
            for i in 0..n {
                let mut sum = F::zero();
                for j in 0..n {
                    sum = sum + matrix[[i, j]] * v[j];
                }
                new_v[i] = sum;
            }

            // Orthogonalize against previous eigenvectors
            for prev_k in 0..k {
                let mut dot_product = F::zero();
                for i in 0..n {
                    dot_product = dot_product + new_v[i] * eigenvectors[[i, prev_k]];
                }

                for i in 0..n {
                    new_v[i] = new_v[i] - dot_product * eigenvectors[[i, prev_k]];
                }
            }

            // Normalize
            let mut norm = F::zero();
            for i in 0..n {
                norm = norm + new_v[i] * new_v[i];
            }
            norm = norm.sqrt();

            if norm > F::epsilon() {
                for i in 0..n {
                    new_v[i] = new_v[i] / norm;
                }
            }

            // Check convergence
            let mut converged = true;
            for i in 0..n {
                if (new_v[i] - v[i]).abs() > F::from(1e-6).unwrap() {
                    converged = false;
                    break;
                }
            }

            v = new_v;

            if converged {
                break;
            }
        }

        // Compute eigenvalue
        let mut eigenvalue = F::zero();
        for i in 0..n {
            let mut sum = F::zero();
            for j in 0..n {
                sum = sum + matrix[[i, j]] * v[j];
            }
            eigenvalue = eigenvalue + v[i] * sum;
        }

        eigenvalues[k] = eigenvalue;
        for i in 0..n {
            eigenvectors[[i, k]] = v[i];
        }
    }

    Ok((eigenvectors, eigenvalues))
}

#[allow(dead_code)]
fn matrix_multiply<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + Zero + Copy + std::fmt::Display,
{
    let (m, n) = a.dim();
    let (n2, p) = b.dim();

    if n != n2 {
        return Err(StatsError::DimensionMismatch(
            "Matrix dimensions don't match".to_string(),
        ));
    }

    let mut result = Array2::<F>::zeros((m, p));

    for i in 0..m {
        for j in 0..p {
            let mut sum = F::zero();
            for k in 0..n {
                sum = sum + a[[i, k]] * b[[k, j]];
            }
            result[[i, j]] = sum;
        }
    }

    Ok(result)
}

#[allow(dead_code)]
fn incremental_pca<F>(
    data: &ArrayView2<F>,
    n_components: usize,
    means: &Array1<F>,
    manager: &mut AdaptiveMemoryManager,
) -> StatsResult<PCAResult<F>>
where
    F: Float + NumCast + Zero + One + Copy + Send + Sync + std::fmt::Debug + std::fmt::Display,
{
    let (n_obs, n_vars) = data.dim();
    let n_components = n_components.min(n_vars);

    // Batch size for incremental processing
    let batchsize = manager.get_optimal_chunksize(n_obs, std::mem::size_of::<F>() * n_vars);

    // Initialize _components with random orthogonal matrix
    let mut components = Array2::<F>::zeros((n_vars, n_components));
    for i in 0..n_components {
        components[[i % n_vars, i]] = F::one();
    }

    let mut explained_variance = Array1::<F>::zeros(n_components);
    let mut _n_samples_seen = 0;

    // Process data in batches
    for batch_start in (0..n_obs).step_by(batchsize) {
        let batch_end = (batch_start + batchsize).min(n_obs);
        let batch = data.slice(ndarray::s![batch_start..batch_end, ..]);

        // Center the batch
        let mut centered_batch = Array2::<F>::zeros(batch.dim());
        for i in 0..batch.nrows() {
            for j in 0..batch.ncols() {
                centered_batch[[i, j]] = batch[[i, j]] - means[j];
            }
        }

        // Update _components using simplified incremental update
        for k in 0..n_components {
            let component = components.column(k).to_owned();

            // Project batch onto current component
            let mut projections = Array1::<F>::zeros(batch.nrows());
            for i in 0..batch.nrows() {
                let mut projection = F::zero();
                for j in 0..n_vars {
                    projection = projection + centered_batch[[i, j]] * component[j];
                }
                projections[i] = projection;
            }

            // Update component direction
            let mut new_component = Array1::<F>::zeros(n_vars);
            for j in 0..n_vars {
                let mut sum = F::zero();
                for i in 0..batch.nrows() {
                    sum = sum + centered_batch[[i, j]] * projections[i];
                }
                new_component[j] = sum;
            }

            // Normalize
            let mut norm = F::zero();
            for j in 0..n_vars {
                norm = norm + new_component[j] * new_component[j];
            }
            norm = norm.sqrt();

            if norm > F::epsilon() {
                for j in 0..n_vars {
                    components[[j, k]] = new_component[j] / norm;
                }

                // Update explained variance
                let variance = projections
                    .iter()
                    .map(|&x| x * x)
                    .fold(F::zero(), |acc, x| acc + x);
                explained_variance[k] = variance / F::from(batch.nrows()).unwrap();
            }
        }

        _n_samples_seen += batch.nrows();
    }

    // Transform the data
    let mut transformeddata = Array2::<F>::zeros((n_obs, n_components));
    for i in 0..n_obs {
        for k in 0..n_components {
            let mut projection = F::zero();
            for j in 0..n_vars {
                let centered_val = data[[i, j]] - means[j];
                projection = projection + centered_val * components[[j, k]];
            }
            transformeddata[[i, k]] = projection;
        }
    }

    Ok(PCAResult {
        components,
        explained_variance,
        transformeddata,
        mean: means.clone(),
    })
}

#[allow(dead_code)]
fn checkarray_finite_2d<F, D>(arr: &ArrayBase<D, Ix2>, name: &str) -> StatsResult<()>
where
    F: Float,
    D: Data<Elem = F>,
{
    for &val in arr.iter() {
        if !val.is_finite() {
            return Err(StatsError::InvalidArgument(format!(
                "{} contains non-finite values",
                name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    #[ignore = "timeout"]
    fn test_adaptive_memory_manager() {
        let constraints = MemoryConstraints::default();
        let manager = AdaptiveMemoryManager::new(constraints);

        let chunksize = manager.get_optimal_chunksize(1000, 8);
        assert!(chunksize > 0);
        assert!(chunksize <= 1000);

        let metrics = OperationMetrics {
            operation_type: "test".to_string(),
            memory_used: 1024,
            processing_time: std::time::Duration::from_millis(10),
            chunksize_used: chunksize,
        };

        manager.record_operation(metrics);
        let stats = manager.get_statistics();
        assert_eq!(stats.operations_completed, 1);
    }

    #[test]
    fn test_cache_oblivious_matrix_mult() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let result = cache_oblivious_matrix_mult(&a.view(), &b.view(), 2).unwrap();

        // Expected: [[19, 22], [43, 50]]
        assert!((result[[0, 0]] - 19.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 22.0).abs() < 1e-10);
        assert!((result[[1, 0]] - 43.0).abs() < 1e-10);
        assert!((result[[1, 1]] - 50.0).abs() < 1e-10);
    }
}
