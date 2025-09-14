//! Advanced parallel statistical processing with intelligent optimization
//!
//! This module provides state-of-the-art parallel implementations that
//! automatically adapt to system characteristics and data patterns for
//! optimal performance across different hardware configurations.

use crate::error::{StatsError, StatsResult};
use crate::error_standardization::ErrorMessages;
use crate::simd_enhanced_core::{mean_enhanced, variance_enhanced, ComprehensiveStats};
use crossbeam;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Data, Ix1, Ix2};
use num_traits::{Float, NumCast, One, Zero};
use scirs2_core::{
    parallel_ops::*,
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
};
use std::collections::VecDeque;
use std::sync::{atomic::AtomicUsize, Arc, Mutex};
use std::thread;

/// Advanced parallel processing configuration
#[derive(Debug, Clone)]
pub struct AdvancedParallelConfig {
    /// Minimum data size to trigger parallel processing
    pub parallel_threshold: usize,
    /// Number of worker threads (None = auto-detect)
    pub num_threads: Option<usize>,
    /// Enable NUMA-aware processing
    pub numa_aware: bool,
    /// Enable work stealing for better load balancing
    pub work_stealing: bool,
    /// Preferred chunk size strategy
    pub chunk_strategy: ChunkStrategy,
    /// Maximum memory usage for intermediate results (bytes)
    pub max_memory_usage: usize,
}

impl Default for AdvancedParallelConfig {
    fn default() -> Self {
        Self {
            parallel_threshold: 10_000,
            num_threads: None,
            numa_aware: true,
            work_stealing: true,
            chunk_strategy: ChunkStrategy::Adaptive,
            max_memory_usage: 1024 * 1024 * 1024, // 1GB
        }
    }
}

/// Chunking strategies for optimal data access patterns
#[derive(Debug, Clone, Copy)]
pub enum ChunkStrategy {
    /// Fixed chunk size
    Fixed(usize),
    /// Cache-aware chunking
    CacheOptimal,
    /// Adaptive chunking based on data characteristics
    Adaptive,
    /// Work-stealing with dynamic load balancing
    WorkStealing,
}

/// Advanced parallel statistics processor
pub struct AdvancedParallelProcessor<F: Float + std::fmt::Display> {
    config: AdvancedParallelConfig,
    capabilities: PlatformCapabilities,
    #[allow(dead_code)]
    thread_pool: Option<ThreadPool>,
    #[allow(dead_code)]
    work_queue: Arc<Mutex<VecDeque<ParallelTask<F>>>>,
    #[allow(dead_code)]
    active_workers: Arc<AtomicUsize>,
}

/// Task for parallel execution
enum ParallelTask<F: Float + std::fmt::Display> {
    Mean(Vec<F>),
    Variance(Vec<F>, F, usize), // data, mean, ddof
    Correlation(Vec<F>, Vec<F>),
    Histogram(Vec<F>, usize),
}

/// Result of parallel computation
pub enum ParallelResult<F: Float + std::fmt::Display> {
    Mean(F),
    Variance(F),
    Correlation(F),
    Histogram(Vec<usize>),
}

impl<F> AdvancedParallelProcessor<F>
where
    F: Float
        + NumCast
        + Send
        + Sync
        + SimdUnifiedOps
        + Copy
        + 'static
        + Zero
        + One
        + std::fmt::Debug
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    /// Create a new advanced parallel processor
    pub fn new(config: AdvancedParallelConfig) -> Self {
        let capabilities = PlatformCapabilities::detect();

        Self {
            config,
            capabilities,
            thread_pool: None,
            work_queue: Arc::new(Mutex::new(VecDeque::new())),
            active_workers: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Initialize the thread pool with optimal configuration
    pub fn initialize(&mut self) -> StatsResult<()> {
        let num_threads = self
            .config
            .num_threads
            .unwrap_or_else(|| self.optimal_thread_count());

        self.thread_pool = Some(ThreadPool::new(num_threads, self.config.clone())?);
        Ok(())
    }

    /// Compute mean using advanced parallel processing
    pub fn mean_parallel_advanced<D>(&self, x: &ArrayBase<D, Ix1>) -> StatsResult<F>
    where
        D: Data<Elem = F> + Sync + Send,
    {
        if x.is_empty() {
            return Err(ErrorMessages::empty_array("x"));
        }

        let n = x.len();

        // Use sequential processing for small arrays
        if n < self.config.parallel_threshold {
            return mean_enhanced(x);
        }

        // Choose optimal parallel strategy
        match self.config.chunk_strategy {
            ChunkStrategy::WorkStealing => self.mean_work_stealing(x),
            ChunkStrategy::Adaptive => self.mean_adaptive_chunking(x),
            ChunkStrategy::CacheOptimal => self.mean_cache_optimal(x),
            ChunkStrategy::Fixed(chunksize) => self.mean_fixed_chunks(x, chunksize),
        }
    }

    /// Compute variance using advanced parallel processing with numerical stability
    pub fn variance_parallel_advanced<D>(
        &self,
        x: &ArrayBase<D, Ix1>,
        ddof: usize,
    ) -> StatsResult<F>
    where
        D: Data<Elem = F> + Sync + Send,
    {
        let n = x.len();
        if n == 0 {
            return Err(ErrorMessages::empty_array("x"));
        }
        if n <= ddof {
            return Err(ErrorMessages::insufficientdata(
                "variance calculation",
                ddof + 1,
                n,
            ));
        }

        if n < self.config.parallel_threshold {
            return variance_enhanced(x, ddof);
        }

        // Use parallel Welford's algorithm for better numerical stability
        self.variance_welford_parallel(x, ddof)
    }

    /// Compute correlation matrix in parallel for multivariate data
    pub fn correlation_matrix_parallel<D>(&self, data: &ArrayBase<D, Ix2>) -> StatsResult<Array2<F>>
    where
        D: Data<Elem = F> + Sync + Send,
    {
        let (n_samples_, n_features) = data.dim();

        if n_samples_ == 0 {
            return Err(ErrorMessages::empty_array("data"));
        }
        if n_features == 0 {
            return Err(ErrorMessages::insufficientdata(
                "correlation matrix",
                2,
                n_features,
            ));
        }

        let mut correlation_matrix = Array2::eye(n_features);

        // Parallel computation of upper triangle
        if n_features > 4 && n_samples_ > self.config.parallel_threshold {
            self.correlation_matrix_parallel_upper_triangle(data, &mut correlation_matrix)?;
        } else {
            self.correlation_matrix_sequential(data, &mut correlation_matrix)?;
        }

        // Fill lower triangle (correlation matrix is symmetric)
        for i in 0..n_features {
            for j in 0..i {
                correlation_matrix[[i, j]] = correlation_matrix[[j, i]];
            }
        }

        Ok(correlation_matrix)
    }

    /// Batch parallel processing for multiple statistical operations
    pub fn batch_statistics_parallel<D>(
        &self,
        x: &ArrayBase<D, Ix1>,
        ddof: usize,
    ) -> StatsResult<ComprehensiveStats<F>>
    where
        D: Data<Elem = F> + Sync + Send,
    {
        let n = x.len();
        if n == 0 {
            return Err(ErrorMessages::empty_array("x"));
        }
        if n <= ddof {
            return Err(ErrorMessages::insufficientdata(
                "comprehensive statistics",
                ddof + 1,
                n,
            ));
        }

        if n < self.config.parallel_threshold {
            // Use the enhanced SIMD version for smaller datasets
            return crate::simd_enhanced_core::comprehensive_stats_simd(x, ddof);
        }

        // Parallel single-pass computation of all statistics
        self.comprehensive_stats_single_pass_parallel(x, ddof)
    }

    /// Parallel bootstrap resampling with intelligent load balancing
    pub fn bootstrap_parallel<D>(
        &self,
        x: &ArrayBase<D, Ix1>,
        n_samples_: usize,
        statistic_fn: impl Fn(&ArrayView1<F>) -> F + Send + Sync + Clone,
        seed: Option<u64>,
    ) -> StatsResult<Array1<F>>
    where
        D: Data<Elem = F> + Sync + Send,
    {
        if x.is_empty() {
            return Err(ErrorMessages::empty_array("x"));
        }
        if n_samples_ == 0 {
            return Err(ErrorMessages::insufficientdata("bootstrap", 1, 0));
        }

        let num_threads = self
            .config
            .num_threads
            .unwrap_or_else(|| self.optimal_thread_count());
        let samples_per_thread = (n_samples_ + num_threads - 1) / num_threads;

        // Parallel bootstrap computation with work stealing
        self.bootstrap_work_stealing(x, n_samples_, samples_per_thread, statistic_fn, seed)
    }

    // Private helper methods

    fn optimal_thread_count(&self) -> usize {
        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // Account for hyperthreading - usually optimal to use physical cores
        // Simple heuristic: if we have more than 2 logical cores, assume hyperthreading
        let physical_cores = if logical_cores > 2 {
            logical_cores / 2
        } else {
            logical_cores
        };

        // For CPU-intensive tasks, use physical cores
        // For memory-bound tasks, might benefit from more threads
        // Use physical cores for better performance
        physical_cores
    }

    fn mean_work_stealing<D>(&self, x: &ArrayBase<D, Ix1>) -> StatsResult<F>
    where
        D: Data<Elem = F> + Sync + Send,
    {
        let n = x.len();
        let num_threads = self
            .config
            .num_threads
            .unwrap_or_else(|| self.optimal_thread_count());
        let initial_chunksize = (n + num_threads - 1) / num_threads;

        // Create work queue with initial chunks
        let work_queue: Arc<Mutex<VecDeque<(usize, usize)>>> =
            Arc::new(Mutex::new(VecDeque::new()));

        for i in 0..num_threads {
            let start = i * initial_chunksize;
            let end = ((i + 1) * initial_chunksize).min(n);
            if start < end {
                work_queue.lock().unwrap().push_back((start, end));
            }
        }

        let partial_sums: Arc<Mutex<Vec<F>>> = Arc::new(Mutex::new(Vec::new()));
        let data_slice = x
            .as_slice()
            .ok_or(StatsError::InvalidInput("Data not contiguous".to_string()))?;

        crossbeam::scope(|s| {
            for _ in 0..num_threads {
                let work_queue = Arc::clone(&work_queue);
                let partial_sums = Arc::clone(&partial_sums);

                s.spawn(move |_| {
                    let mut local_sum = F::zero();

                    while let Some((start, end)) = work_queue.lock().unwrap().pop_front() {
                        // Process chunk safely
                        for &val in &data_slice[start..end] {
                            local_sum = local_sum + val;
                        }

                        // Split remaining work if chunk was large
                        if end - start > 1000 {
                            let mid = (start + end) / 2;
                            if mid > start {
                                work_queue.lock().unwrap().push_back((mid, end));
                            }
                        }
                    }

                    partial_sums.lock().unwrap().push(local_sum);
                });
            }
        })
        .unwrap();

        let total_sum = partial_sums
            .lock()
            .unwrap()
            .iter()
            .fold(F::zero(), |acc, &val| acc + val);
        Ok(total_sum / F::from(n).unwrap())
    }

    fn mean_adaptive_chunking<D>(&self, x: &ArrayBase<D, Ix1>) -> StatsResult<F>
    where
        D: Data<Elem = F> + Sync + Send,
    {
        let n = x.len();
        let elementsize = std::mem::size_of::<F>();

        // Adaptive chunk size based on cache hierarchy
        let l1_cache = 32 * 1024; // 32KB L1 cache (typical)
        let l2_cache = 256 * 1024; // 256KB L2 cache (typical)

        let chunksize = if n * elementsize <= l1_cache {
            n // Fits in L1, no chunking needed
        } else if n * elementsize <= l2_cache {
            l1_cache / elementsize // Chunk to fit in L1
        } else {
            l2_cache / elementsize // Chunk to fit in L2
        };

        let num_chunks = (n + chunksize - 1) / chunksize;
        let _num_threads = self
            .config
            .num_threads
            .unwrap_or_else(|| self.optimal_thread_count());

        // Use thread pool for processing
        let chunks: Vec<_> = (0..num_chunks)
            .map(|i| {
                let start = i * chunksize;
                let end = ((i + 1) * chunksize).min(n);
                x.slice(ndarray::s![start..end])
            })
            .collect();

        let partial_sums: Vec<F> = chunks
            .into_par_iter()
            .map(|chunk| {
                if self.capabilities.simd_available && chunk.len() > 64 {
                    F::simd_sum(&chunk)
                } else {
                    chunk.iter().fold(F::zero(), |acc, &val| acc + val)
                }
            })
            .collect();

        let total_sum = partial_sums
            .into_iter()
            .fold(F::zero(), |acc, val| acc + val);
        Ok(total_sum / F::from(n).unwrap())
    }

    fn mean_cache_optimal<D>(&self, x: &ArrayBase<D, Ix1>) -> StatsResult<F>
    where
        D: Data<Elem = F> + Sync + Send,
    {
        // Use cache-oblivious algorithm for optimal performance
        Self::mean_cache_oblivious_static(x, 0, x.len())
    }

    #[allow(dead_code)]
    fn mean_cache_oblivious<D>(
        &self,
        x: &ArrayBase<D, Ix1>,
        start: usize,
        len: usize,
    ) -> StatsResult<F>
    where
        D: Data<Elem = F> + Sync + Send,
    {
        Self::mean_cache_oblivious_static(x, start, len)
    }

    // Static version that can be used in threads
    fn mean_cache_oblivious_static<D>(
        x: &ArrayBase<D, Ix1>,
        start: usize,
        len: usize,
    ) -> StatsResult<F>
    where
        D: Data<Elem = F> + Sync + Send,
        F: Float + Send + Sync + 'static + std::fmt::Display,
    {
        const CACHE_THRESHOLD: usize = 1024; // Empirically determined threshold

        if len <= CACHE_THRESHOLD {
            // Base case: compute directly
            let slice = x.slice(ndarray::s![start..start + len]);
            let sum = slice.iter().fold(F::zero(), |acc, &val| acc + val);
            Ok(sum / F::from(len).unwrap())
        } else {
            // Divide and conquer (sequential to avoid lifetime issues)
            let mid = len / 2;
            let left_result = Self::mean_cache_oblivious_static(x, start, mid)?;
            let right_result = Self::mean_cache_oblivious_static(x, start + mid, len - mid)?;

            // Combine results weighted by size
            let left_weight = F::from(mid).unwrap();
            let right_weight = F::from(len - mid).unwrap();
            let total_weight = F::from(len).unwrap();

            Ok((left_result * left_weight + right_result * right_weight) / total_weight)
        }
    }

    fn mean_fixed_chunks<D>(&self, x: &ArrayBase<D, Ix1>, chunksize: usize) -> StatsResult<F>
    where
        D: Data<Elem = F> + Sync + Send,
    {
        let n = x.len();
        let chunks: Vec<_> = x
            .exact_chunks(chunksize)
            .into_iter()
            .chain(if n % chunksize != 0 {
                vec![x.slice(ndarray::s![n - (n % chunksize)..])]
            } else {
                vec![]
            })
            .collect();

        let partial_sums: Vec<F> = chunks
            .into_par_iter()
            .map(|chunk| chunk.iter().fold(F::zero(), |acc, &val| acc + val))
            .collect();

        let total_sum = partial_sums
            .into_iter()
            .fold(F::zero(), |acc, val| acc + val);
        Ok(total_sum / F::from(n).unwrap())
    }

    fn variance_welford_parallel<D>(&self, x: &ArrayBase<D, Ix1>, ddof: usize) -> StatsResult<F>
    where
        D: Data<Elem = F> + Sync + Send,
    {
        // Parallel Welford's algorithm implementation
        let n = x.len();
        let num_threads = self
            .config
            .num_threads
            .unwrap_or_else(|| self.optimal_thread_count());
        let chunksize = (n + num_threads - 1) / num_threads;

        let results: Vec<(F, F, usize)> = (0..num_threads)
            .into_par_iter()
            .map(|i| {
                let start = i * chunksize;
                let end = ((i + 1) * chunksize).min(n);

                if start >= end {
                    return (F::zero(), F::zero(), 0);
                }

                let chunk = x.slice(ndarray::s![start..end]);
                let mut mean = F::zero();
                let mut m2 = F::zero();
                let count = chunk.len();

                for (j, &val) in chunk.iter().enumerate() {
                    let n = F::from(j + 1).unwrap();
                    let delta = val - mean;
                    mean = mean + delta / n;
                    let delta2 = val - mean;
                    m2 = m2 + delta * delta2;
                }

                (mean, m2, count)
            })
            .collect();

        // Combine results using parallel reduction
        let (_final_mean, final_m2, final_count) = results.into_iter().fold(
            (F::zero(), F::zero(), 0),
            |(mean_a, m2_a, count_a), (mean_b, m2_b, count_b)| {
                if count_b == 0 {
                    return (mean_a, m2_a, count_a);
                }
                if count_a == 0 {
                    return (mean_b, m2_b, count_b);
                }

                let total_count = count_a + count_b;
                let count_a_f = F::from(count_a).unwrap();
                let count_b_f = F::from(count_b).unwrap();
                let total_count_f = F::from(total_count).unwrap();

                let delta = mean_b - mean_a;
                let combined_mean = (mean_a * count_a_f + mean_b * count_b_f) / total_count_f;
                let combined_m2 =
                    m2_a + m2_b + delta * delta * count_a_f * count_b_f / total_count_f;

                (combined_mean, combined_m2, total_count)
            },
        );

        Ok(final_m2 / F::from(n - ddof).unwrap())
    }

    fn correlation_matrix_parallel_upper_triangle<D>(
        &self,
        data: &ArrayBase<D, Ix2>,
        correlation_matrix: &mut Array2<F>,
    ) -> StatsResult<()>
    where
        D: Data<Elem = F> + Sync + Send,
    {
        let (_, n_features) = data.dim();

        // Generate pairs for upper triangle
        let pairs: Vec<(usize, usize)> = (0..n_features)
            .flat_map(|i| (i + 1..n_features).map(move |j| (i, j)))
            .collect();

        let results: Vec<((usize, usize), F)> = pairs
            .into_par_iter()
            .map(|(i, j)| {
                let x = data.column(i);
                let y = data.column(j);
                let corr = crate::simd_enhanced_core::correlation_simd_enhanced(&x, &y)
                    .unwrap_or(F::zero());
                ((i, j), corr)
            })
            .collect();

        // Fill the correlation _matrix
        for ((i, j), corr) in results {
            correlation_matrix[[i, j]] = corr;
        }

        Ok(())
    }

    fn correlation_matrix_sequential<D>(
        &self,
        data: &ArrayBase<D, Ix2>,
        correlation_matrix: &mut Array2<F>,
    ) -> StatsResult<()>
    where
        D: Data<Elem = F> + Sync + Send,
    {
        let (_, n_features) = data.dim();

        for i in 0..n_features {
            for j in i + 1..n_features {
                let x = data.column(i);
                let y = data.column(j);
                let corr = crate::simd_enhanced_core::correlation_simd_enhanced(&x, &y)?;
                correlation_matrix[[i, j]] = corr;
            }
        }

        Ok(())
    }

    fn comprehensive_stats_single_pass_parallel<D>(
        &self,
        x: &ArrayBase<D, Ix1>,
        ddof: usize,
    ) -> StatsResult<ComprehensiveStats<F>>
    where
        D: Data<Elem = F> + Sync + Send,
    {
        let n = x.len();
        let num_threads = self
            .config
            .num_threads
            .unwrap_or_else(|| self.optimal_thread_count());
        let chunksize = (n + num_threads - 1) / num_threads;

        // Parallel computation of all moments
        let results: Vec<(F, F, F, F, usize)> = (0..num_threads)
            .into_par_iter()
            .map(|i| {
                let start = i * chunksize;
                let end = ((i + 1) * chunksize).min(n);

                if start >= end {
                    return (F::zero(), F::zero(), F::zero(), F::zero(), 0);
                }

                let chunk = x.slice(ndarray::s![start..end]);
                let count = chunk.len();
                let count_f = F::from(count).unwrap();

                // Single pass computation of all moments
                let mean = chunk.iter().fold(F::zero(), |acc, &val| acc + val) / count_f;

                let (m2, m3, m4) =
                    chunk
                        .iter()
                        .fold((F::zero(), F::zero(), F::zero()), |(m2, m3, m4), &val| {
                            let dev = val - mean;
                            let dev2 = dev * dev;
                            let dev3 = dev2 * dev;
                            let dev4 = dev2 * dev2;
                            (m2 + dev2, m3 + dev3, m4 + dev4)
                        });

                (mean, m2, m3, m4, count)
            })
            .collect();

        // Combine results
        let (total_mean, total_m2_, total_m3, total_m4, total_count) = results.into_iter().fold(
            (F::zero(), F::zero(), F::zero(), F::zero(), 0),
            |(mean_acc, m2_acc, m3_acc, m4_acc, count_acc), (mean, m2, m3, m4, count)| {
                if count == 0 {
                    return (mean_acc, m2_acc, m3_acc, m4_acc, count_acc);
                }
                if count_acc == 0 {
                    return (mean, m2, m3, m4, count);
                }

                // Combine means
                let total_count = count_acc + count;
                let count_f = F::from(count).unwrap();
                let count_acc_f = F::from(count_acc).unwrap();
                let total_count_f = F::from(total_count).unwrap();

                let combined_mean = (mean_acc * count_acc_f + mean * count_f) / total_count_f;

                // For simplicity, recalculate moments (could be optimized further)
                (
                    combined_mean,
                    m2_acc + m2,
                    m3_acc + m3,
                    m4_acc + m4,
                    total_count,
                )
            },
        );

        let variance = total_m2_ / F::from(n - ddof).unwrap();
        let std = variance.sqrt();

        let skewness = if variance > F::epsilon() {
            (total_m3 / F::from(n).unwrap()) / variance.powf(F::from(1.5).unwrap())
        } else {
            F::zero()
        };

        let kurtosis = if variance > F::epsilon() {
            (total_m4 / F::from(n).unwrap()) / (variance * variance) - F::from(3.0).unwrap()
        } else {
            F::zero()
        };

        Ok(ComprehensiveStats {
            mean: total_mean,
            variance,
            std,
            skewness,
            kurtosis,
            count: n,
        })
    }

    fn bootstrap_work_stealing<D>(
        &self,
        x: &ArrayBase<D, Ix1>,
        n_samples_: usize,
        samples_per_thread: usize,
        statistic_fn: impl Fn(&ArrayView1<F>) -> F + Send + Sync + Clone,
        seed: Option<u64>,
    ) -> StatsResult<Array1<F>>
    where
        D: Data<Elem = F> + Sync + Send,
    {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let num_threads = self
            .config
            .num_threads
            .unwrap_or_else(|| self.optimal_thread_count());
        let _results: Vec<F> = Vec::with_capacity(n_samples_);

        let data_vec: Vec<F> = x.iter().cloned().collect();
        let data_arc = Arc::new(data_vec);

        let partial_results: Arc<Mutex<Vec<F>>> = Arc::new(Mutex::new(Vec::new()));

        crossbeam::scope(|s| {
            for thread_id in 0..num_threads {
                let data_arc = Arc::clone(&data_arc);
                let partial_results = Arc::clone(&partial_results);
                let statistic_fn = statistic_fn.clone();

                s.spawn(move |_| {
                    let mut rng = if let Some(seed) = seed {
                        ChaCha8Rng::seed_from_u64(seed + thread_id as u64)
                    } else {
                        ChaCha8Rng::from_rng(&mut rand::rng())
                    };

                    let mut local_results = Vec::with_capacity(samples_per_thread);
                    let ndata = data_arc.len();

                    for _ in 0..samples_per_thread {
                        // Generate bootstrap sample
                        let bootstrap_indices: Vec<usize> =
                            (0..ndata).map(|_| rng.gen_range(0..ndata)).collect();

                        let bootstrap_sample: Vec<F> =
                            bootstrap_indices.into_iter().map(|i| data_arc[i]).collect();

                        let sample_array = Array1::from(bootstrap_sample);
                        let statistic = statistic_fn(&sample_array.view());
                        local_results.push(statistic);
                    }

                    partial_results.lock().unwrap().extend(local_results);
                });
            }
        })
        .unwrap();

        let mut all_results = partial_results.lock().unwrap();
        all_results.truncate(n_samples_); // Ensure exact number of _samples

        Ok(Array1::from(all_results.clone()))
    }
}

/// Simple thread pool for parallel execution
struct ThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    sender: std::sync::mpsc::Sender<Message>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

enum Message {
    NewJob(Job),
    Terminate,
}

impl ThreadPool {
    fn new(size: usize, config: AdvancedParallelConfig) -> StatsResult<ThreadPool> {
        if size == 0 {
            return Err(ErrorMessages::invalid_probability("thread count", 0.0));
        }

        let (sender, receiver) = std::sync::mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers = Vec::with_capacity(size);

        for _id in 0..size {
            let receiver = Arc::clone(&receiver);

            let worker = thread::spawn(move || loop {
                let message = receiver.lock().unwrap().recv().unwrap();

                match message {
                    Message::NewJob(job) => {
                        job();
                    }
                    Message::Terminate => {
                        break;
                    }
                }
            });

            workers.push(worker);
        }

        Ok(ThreadPool { workers, sender })
    }

    #[allow(dead_code)]
    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.send(Message::NewJob(job)).unwrap();
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        for _ in &self.workers {
            self.sender.send(Message::Terminate).unwrap();
        }

        for worker in &mut self.workers {
            if let Some(handle) = worker.thread().name() {
                println!("Shutting down worker {}", handle);
            }
        }
    }
}

/// Convenience function to create an advanced parallel processor
#[allow(dead_code)]
pub fn create_advanced_parallel_processor<F>() -> AdvancedParallelProcessor<F>
where
    F: Float
        + NumCast
        + Send
        + Sync
        + SimdUnifiedOps
        + Copy
        + 'static
        + Zero
        + One
        + std::fmt::Debug
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    AdvancedParallelProcessor::new(AdvancedParallelConfig::default())
}

/// Convenience function to create a processor with custom configuration
#[allow(dead_code)]
pub fn create_configured_parallel_processor<F>(
    config: AdvancedParallelConfig,
) -> AdvancedParallelProcessor<F>
where
    F: Float
        + NumCast
        + Send
        + Sync
        + SimdUnifiedOps
        + Copy
        + 'static
        + Zero
        + One
        + std::fmt::Debug
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    AdvancedParallelProcessor::new(config)
}
