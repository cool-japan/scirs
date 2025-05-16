//! Parallel computation utilities for metrics
//!
//! This module provides tools for computing metrics in parallel using Rayon.

use ndarray::{ArrayBase, Data, Dimension};
use parking_lot;
use rayon::prelude::*;
use std::sync::Arc;

use crate::error::Result;

/// Type alias for a metric function that can be executed in parallel
pub type ParallelMetricFn<S1, S2, D1, D2> =
    dyn Fn(&ArrayBase<S1, D1>, &ArrayBase<S2, D2>) -> Result<f64> + Send + Sync;

/// Configuration for parallel metrics computation
///
/// This struct provides options for controlling parallel execution
/// of metrics calculations using Rayon.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Minimum chunk size for parallel processing
    pub min_chunk_size: usize,
    /// Whether to use parallel processing
    pub parallel_enabled: bool,
    /// Number of threads to use (None = use Rayon's default thread pool)
    pub num_threads: Option<usize>,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        ParallelConfig {
            min_chunk_size: 1000,
            parallel_enabled: true,
            num_threads: None,
        }
    }
}

impl ParallelConfig {
    /// Create a new ParallelConfig with default values
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the minimum chunk size for parallel processing
    pub fn with_min_chunk_size(mut self, size: usize) -> Self {
        self.min_chunk_size = size;
        self
    }

    /// Enable or disable parallel processing
    pub fn with_parallel_enabled(mut self, enabled: bool) -> Self {
        self.parallel_enabled = enabled;
        self
    }

    /// Set the number of threads to use
    pub fn with_num_threads(mut self, threads: Option<usize>) -> Self {
        self.num_threads = threads;
        self
    }
}

/// Trait for metrics that can be computed in parallel
pub trait ParallelMetric<T, D>
where
    T: Send + Sync,
    D: Dimension,
{
    /// Compute the metric in parallel
    fn compute_parallel(
        &self,
        x: &ArrayBase<impl Data<Elem = T>, D>,
        config: &ParallelConfig,
    ) -> Result<f64>;
}

/// Compute multiple metrics in parallel
///
/// This function computes multiple metrics in parallel using Rayon.
///
/// # Arguments
///
/// * `y_true` - True values
/// * `y_pred` - Predicted values
/// * `metric_fns` - Vector of metric functions
/// * `config` - Parallel configuration
///
/// # Returns
///
/// * Vector of metric values
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_metrics::optimization::parallel::{compute_metrics_batch, ParallelConfig};
/// use scirs2_metrics::error::Result;
/// use scirs2_metrics::classification::{accuracy_score, precision_score};
///
/// // Create sample data
/// let y_true = Array1::from_vec(vec![0, 1, 2, 0, 1, 2]);
/// let y_pred = Array1::from_vec(vec![0, 2, 1, 0, 0, 2]);
///
/// // Define metric functions
/// let metric_fns: Vec<Box<dyn Fn(&Array1<i32>, &Array1<i32>) -> Result<f64> + Send + Sync>> = vec![
///     Box::new(|a, b| accuracy_score(a, b)),
///     Box::new(|a, b| precision_score(a, b, 1)),
/// ];
///
/// // Compute metrics in parallel
/// let config = ParallelConfig::default();
/// let results = compute_metrics_batch(&y_true, &y_pred, &metric_fns, &config).unwrap();
///
/// // Check results
/// assert_eq!(results.len(), 2);
/// ```
pub fn compute_metrics_batch<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    metric_fns: &[Box<ParallelMetricFn<S1, S2, D1, D2>>],
    config: &ParallelConfig,
) -> Result<Vec<f64>>
where
    T: Clone + Send + Sync,
    S1: Data<Elem = T> + Sync,
    S2: Data<Elem = T> + Sync,
    D1: Dimension + Sync,
    D2: Dimension + Sync,
{
    if !config.parallel_enabled || metric_fns.len() < 2 {
        // Sequential computation if parallel is disabled or only one metric
        let mut results = Vec::with_capacity(metric_fns.len());
        for metric_fn in metric_fns {
            let value = metric_fn(y_true, y_pred)?;
            results.push(value);
        }
        return Ok(results);
    }

    // Parallel computation of metrics
    let results: Result<Vec<f64>> = metric_fns
        .par_iter()
        .map(|metric_fn| metric_fn(y_true, y_pred))
        .collect();

    results
}

/// Process a large array in chunks with parallel execution
///
/// This function splits a large array into chunks and processes each chunk in parallel.
///
/// # Arguments
///
/// * `data` - Input data
/// * `chunk_size` - Size of each chunk
/// * `chunk_op` - Operation to perform on each chunk
/// * `reducer` - Function to combine results from all chunks
///
/// # Returns
///
/// * Combined result
///
/// # Examples
///
/// ```
/// use scirs2_metrics::optimization::parallel::{chunked_parallel_compute, ParallelConfig};
/// use scirs2_metrics::error::Result;
///
/// // Create sample data
/// let data: Vec<f64> = (0..1000).map(|x| x as f64).collect();
///
/// // Define chunk operation (sum of squares)
/// let chunk_op = |chunk: &[f64]| -> Result<f64> {
///     Ok(chunk.iter().map(|x| x * x).sum())
/// };
///
/// // Define reducer (sum of partial results)
/// let reducer = |results: Vec<f64>| -> Result<f64> {
///     Ok(results.iter().sum())
/// };
///
/// // Process data in chunks
/// let result = chunked_parallel_compute(&data, 100, chunk_op, reducer).unwrap();
///
/// // Verify result
/// let expected: f64 = (0..1000).map(|x| (x * x) as f64).sum();
/// assert!((result - expected).abs() < 1e-10);
/// ```
pub fn chunked_parallel_compute<T, R>(
    data: &[T],
    chunk_size: usize,
    chunk_op: impl Fn(&[T]) -> Result<R> + Send + Sync,
    reducer: impl Fn(Vec<R>) -> Result<R>,
) -> Result<R>
where
    T: Clone + Send + Sync,
    R: Send + Sync,
{
    if data.len() <= chunk_size {
        // If data fits in a single chunk, just process it directly
        return chunk_op(data);
    }

    // Split data into chunks
    let chunks: Vec<&[T]> = data.chunks(chunk_size).collect();

    // Process chunks in parallel
    let results: Result<Vec<R>> = chunks.par_iter().map(|chunk| chunk_op(chunk)).collect();

    // Combine results
    reducer(results?)
}

/// Trait for defining chunked metric operations
pub trait ChunkedMetric<T> {
    /// Type for intermediate state
    type State: Send + Sync;

    /// Initialize state
    fn init_state(&self) -> Self::State;

    /// Process a chunk and update state
    fn process_chunk(&self, state: &mut Self::State, chunk: &[T]) -> Result<()>;

    /// Finalize computation from state
    fn finalize(&self, state: &Self::State) -> Result<f64>;
}

/// Process a large array using chunked metric computation
///
/// # Arguments
///
/// * `data` - Input data
/// * `metric` - Chunked metric implementation
/// * `chunk_size` - Size of each chunk
/// * `config` - Parallel configuration
///
/// # Returns
///
/// * Computed metric value
pub fn compute_chunked_metric<T, M>(
    data: &[T],
    metric: &M,
    chunk_size: usize,
    config: &ParallelConfig,
) -> Result<f64>
where
    T: Clone + Send + Sync,
    M: ChunkedMetric<T> + Send + Sync,
{
    if data.len() <= chunk_size || !config.parallel_enabled {
        // If data fits in a single chunk or parallel is disabled
        let mut state = metric.init_state();
        metric.process_chunk(&mut state, data)?;
        return metric.finalize(&state);
    }

    // Create shared state
    let state = Arc::new(parking_lot::Mutex::new(metric.init_state()));
    let metric = Arc::new(metric);

    // Split data into chunks
    let chunks: Vec<&[T]> = data.chunks(chunk_size).collect();

    // Process chunks in parallel
    let result: Result<()> = chunks.par_iter().try_for_each(|chunk| {
        let mut local_state = metric.init_state();
        metric.process_chunk(&mut local_state, chunk)?;

        // Update global state with mutex
        let mut global_state = state.lock();
        metric.process_chunk(&mut *global_state, chunk)?;
        Ok(())
    });

    // Check for errors during processing
    result?;

    // Finalize computation
    let state_lock = state.lock();
    let result = metric.finalize(&*state_lock);
    drop(state_lock); // Explicitly drop the lock before the end of the function
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MetricsError;
    use ndarray::Array1;

    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig::new()
            .with_min_chunk_size(500)
            .with_parallel_enabled(true)
            .with_num_threads(Some(4));

        assert_eq!(config.min_chunk_size, 500);
        assert!(config.parallel_enabled);
        assert_eq!(config.num_threads, Some(4));
    }

    #[test]
    fn test_compute_metrics_batch() {
        // Create sample data
        let y_true = Array1::from_vec(vec![0, 1, 2, 0, 1, 2]);
        let y_pred = Array1::from_vec(vec![0, 2, 1, 0, 0, 2]);

        // Define metric functions
        let metric_fns: Vec<Box<dyn Fn(&Array1<i32>, &Array1<i32>) -> Result<f64> + Send + Sync>> = vec![
            Box::new(|a, b| {
                if a.len() != b.len() {
                    return Err(MetricsError::InvalidInput("Lengths must match".to_string()));
                }
                // Simple accuracy calculation for test
                let correct = a.iter().zip(b.iter()).filter(|&(a, b)| a == b).count();
                Ok(correct as f64 / a.len() as f64)
            }),
            Box::new(|a, _b| {
                // Another dummy metric
                Ok(a.len() as f64)
            }),
        ];

        // Compute metrics with parallel disabled
        let config = ParallelConfig::new().with_parallel_enabled(false);
        let results = compute_metrics_batch(&y_true, &y_pred, &metric_fns, &config).unwrap();

        assert_eq!(results.len(), 2);
        assert!((results[0] - 0.5).abs() < 1e-10); // 3/6 correct
        assert!((results[1] - 6.0).abs() < 1e-10); // Length is 6

        // Compute metrics with parallel enabled
        let config = ParallelConfig::new().with_parallel_enabled(true);
        let results = compute_metrics_batch(&y_true, &y_pred, &metric_fns, &config).unwrap();

        assert_eq!(results.len(), 2);
        assert!((results[0] - 0.5).abs() < 1e-10);
        assert!((results[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_chunked_parallel_compute() {
        // Create sample data
        let data: Vec<f64> = (0..1000).map(|x| x as f64).collect();

        // Define chunk operation (sum of squares)
        let chunk_op = |chunk: &[f64]| -> Result<f64> { Ok(chunk.iter().map(|x| x * x).sum()) };

        // Define reducer (sum of partial results)
        let reducer = |results: Vec<f64>| -> Result<f64> { Ok(results.iter().sum()) };

        // Process data in chunks
        let result = chunked_parallel_compute(&data, 100, chunk_op, reducer).unwrap();

        // Verify result against direct calculation
        let expected: f64 = (0..1000).map(|x| (x * x) as f64).sum();
        assert!((result - expected).abs() < 1e-10);
    }

    // Example implementation of ChunkedMetric for testing
    struct MeanChunkedMetric;

    impl ChunkedMetric<f64> for MeanChunkedMetric {
        type State = (f64, usize); // (sum, count)

        fn init_state(&self) -> Self::State {
            (0.0, 0)
        }

        fn process_chunk(&self, state: &mut Self::State, chunk: &[f64]) -> Result<()> {
            for &value in chunk {
                state.0 += value;
                state.1 += 1;
            }
            Ok(())
        }

        fn finalize(&self, state: &Self::State) -> Result<f64> {
            if state.1 == 0 {
                return Err(MetricsError::DivisionByZero);
            }
            Ok(state.0 / state.1 as f64)
        }
    }

    #[test]
    fn test_compute_chunked_metric() {
        // Create sample data
        let data: Vec<f64> = (0..1000).map(|x| x as f64).collect();

        // Create metric
        let metric = MeanChunkedMetric;

        // Compute with chunking
        let config = ParallelConfig::default();
        let result = compute_chunked_metric(&data, &metric, 100, &config).unwrap();

        // Verify result against direct calculation
        let expected: f64 = data.iter().sum::<f64>() / data.len() as f64;
        assert!((result - expected).abs() < 1e-10);
    }
}
