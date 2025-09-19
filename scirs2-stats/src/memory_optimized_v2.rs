//! Enhanced memory optimization for v1.0.0
//!
//! This module provides advanced memory-efficient implementations with:
//! - Zero-copy operations where possible
//! - Memory pooling for repeated calculations
//! - Lazy evaluation strategies
//! - Cache-aware algorithms

use crate::error::{StatsError, StatsResult};
use ndarray::{ArrayBase, Data, Ix1};
use num_traits::{Float, NumCast};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

#[cfg(feature = "memmap")]
use memmap2::Mmap;

/// Memory usage configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum memory to use in bytes
    pub max_memory: usize,
    /// Chunk size for streaming operations
    pub chunksize: usize,
    /// Whether to use memory pooling
    pub use_pooling: bool,
    /// Cache line size (typically 64 bytes)
    pub cache_linesize: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory: 1 << 30, // 1 GB default
            chunksize: 8192,     // 8K elements
            use_pooling: true,
            cache_linesize: 64,
        }
    }
}

/// Memory pool for reusing allocations
pub struct MemoryPool<F> {
    pools: RefCell<Vec<VecDeque<Vec<F>>>>,
    config: MemoryConfig,
}

impl<F: Float> MemoryPool<F> {
    pub fn new(config: MemoryConfig) -> Self {
        // Create pools for different sizes (powers of 2)
        let pools = vec![VecDeque::new(); 20]; // Up to 2^20 elements
        Self {
            pools: RefCell::new(pools),
            config,
        }
    }

    /// Acquire a buffer of at least the specified size
    pub fn acquire(&self, size: usize) -> Vec<F> {
        if !self.config.use_pooling {
            return vec![F::zero(); size];
        }

        let pool_idx = (size.next_power_of_two().trailing_zeros() as usize).min(19);
        let mut pools = self.pools.borrow_mut();

        if let Some(mut buffer) = pools[pool_idx].pop_front() {
            buffer.resize(size, F::zero());
            buffer
        } else {
            vec![F::zero(); size]
        }
    }

    /// Release a buffer back to the pool
    pub fn release(&self, buffer: Vec<F>) {
        if !self.config.use_pooling || buffer.is_empty() {
            return;
        }

        let capacity = buffer.capacity();
        let pool_idx = (capacity.next_power_of_two().trailing_zeros() as usize).min(19);
        let mut pools = self.pools.borrow_mut();

        // Keep at most 10 buffers per size
        if pools[pool_idx].len() < 10 {
            pools[pool_idx].push_back(buffer);
        }
    }
}

/// Zero-copy mean calculation using views
///
/// This function avoids allocations by working directly with array views
#[allow(dead_code)]
pub fn mean_zero_copy<F, D>(x: &ArrayBase<D, Ix1>) -> StatsResult<F>
where
    F: Float + NumCast,
    D: Data<Elem = F>,
{
    if x.is_empty() {
        return Err(StatsError::invalid_argument(
            "Cannot compute mean of empty array",
        ));
    }

    // Use Kahan summation for improved accuracy with no extra memory
    let mut sum = F::zero();
    let mut c = F::zero(); // Compensation for lost digits

    for &val in x.iter() {
        let y = val - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    Ok(sum / F::from(x.len()).unwrap())
}

/// Cache-friendly variance computation
///
/// Processes data in cache-sized blocks for better performance
#[allow(dead_code)]
pub fn variance_cache_aware<F, D>(
    x: &ArrayBase<D, Ix1>,
    ddof: usize,
    config: Option<MemoryConfig>,
) -> StatsResult<F>
where
    F: Float + NumCast,
    D: Data<Elem = F>,
{
    let n = x.len();
    if n <= ddof {
        return Err(StatsError::invalid_argument(
            "Not enough data points for the given degrees of freedom",
        ));
    }

    let config = config.unwrap_or_default();
    let cache_elements = config.cache_linesize / std::mem::size_of::<F>();

    // First pass: compute mean with cache-friendly access
    let mean = mean_zero_copy(x)?;

    // Second pass: compute variance in cache-sized blocks
    let mut sum_sq_dev = F::zero();
    let mut c = F::zero(); // Kahan compensation

    let chunksize = cache_elements.min(n); // Ensure we don't have empty chunks

    // Process complete chunks
    let mut processed = 0;
    for chunk in x.exact_chunks(chunksize) {
        for &val in chunk.iter() {
            let dev = val - mean;
            let sq_dev = dev * dev;

            // Kahan summation for squared deviations
            let y = sq_dev - c;
            let t = sum_sq_dev + y;
            c = (t - sum_sq_dev) - y;
            sum_sq_dev = t;
            processed += 1;
        }
    }

    // Process remainder elements
    for i in processed..n {
        let val = x[i];
        let dev = val - mean;
        let sq_dev = dev * dev;

        // Kahan summation for squared deviations
        let y = sq_dev - c;
        let t = sum_sq_dev + y;
        c = (t - sum_sq_dev) - y;
        sum_sq_dev = t;
    }

    Ok(sum_sq_dev / F::from(n - ddof).unwrap())
}

/// Lazy statistics calculator that computes values on demand
pub struct LazyStats<'a, F, D>
where
    F: Float,
    D: Data<Elem = F>,
{
    data: &'a ArrayBase<D, Ix1>,
    mean: RefCell<Option<F>>,
    variance: RefCell<Option<F>>,
    min: RefCell<Option<F>>,
    max: RefCell<Option<F>>,
    sorted: RefCell<Option<Vec<F>>>,
}

impl<'a, F, D> LazyStats<'a, F, D>
where
    F: Float + NumCast,
    D: Data<Elem = F>,
{
    pub fn new(data: &'a ArrayBase<D, Ix1>) -> Self {
        Self {
            data,
            mean: RefCell::new(None),
            variance: RefCell::new(None),
            min: RefCell::new(None),
            max: RefCell::new(None),
            sorted: RefCell::new(None),
        }
    }

    /// Get mean (computed lazily)
    pub fn mean(&self) -> StatsResult<F> {
        if let Some(mean) = *self.mean.borrow() {
            return Ok(mean);
        }

        let mean = mean_zero_copy(self.data)?;
        *self.mean.borrow_mut() = Some(mean);
        Ok(mean)
    }

    /// Get variance (computed lazily)
    pub fn variance(&self, ddof: usize) -> StatsResult<F> {
        if let Some(var) = *self.variance.borrow() {
            return Ok(var);
        }

        let var = variance_cache_aware(self.data, ddof, None)?;
        *self.variance.borrow_mut() = Some(var);
        Ok(var)
    }

    /// Get min and max (computed together for efficiency)
    pub fn minmax(&self) -> StatsResult<(F, F)> {
        if let (Some(min), Some(max)) = (*self.min.borrow(), *self.max.borrow()) {
            return Ok((min, max));
        }

        if self.data.is_empty() {
            return Err(StatsError::invalid_argument("Empty array"));
        }

        let (min, max) = self
            .data
            .iter()
            .fold((self.data[0], self.data[0]), |(min, max), &val| {
                (
                    if val < min { val } else { min },
                    if val > max { val } else { max },
                )
            });

        *self.min.borrow_mut() = Some(min);
        *self.max.borrow_mut() = Some(max);
        Ok((min, max))
    }

    /// Get median (requires sorting, cached for subsequent quantile calls)
    pub fn median(&self) -> StatsResult<F> {
        self.quantile(F::from(0.5).unwrap())
    }

    /// Get quantile (uses cached sorted data if available)
    pub fn quantile(&self, q: F) -> StatsResult<F> {
        if self.data.is_empty() {
            return Err(StatsError::invalid_argument("Empty array"));
        }

        let mut sorted_ref = self.sorted.borrow_mut();
        if sorted_ref.is_none() {
            let mut sorted: Vec<F> = self.data.iter().cloned().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            *sorted_ref = Some(sorted);
        }

        let sorted = sorted_ref.as_ref().unwrap();
        let n = sorted.len();
        let pos = q * F::from(n - 1).unwrap();
        let idx = pos.floor();
        let frac = pos - idx;

        let idx = idx.to_usize().unwrap_or(0).min(n - 1);

        if idx == n - 1 {
            Ok(sorted[idx])
        } else {
            Ok(sorted[idx] * (F::one() - frac) + sorted[idx + 1] * frac)
        }
    }
}

/// Streaming covariance matrix computation
///
/// Computes covariance matrix for large datasets without loading all data
pub struct StreamingCovariance<F> {
    n: usize,
    means: Vec<F>,
    cov: Vec<Vec<F>>,
    pool: Rc<MemoryPool<F>>,
}

impl<F: Float + NumCast + std::fmt::Display> StreamingCovariance<F> {
    pub fn new(nfeatures: usize, pool: Rc<MemoryPool<F>>) -> Self {
        Self {
            n: 0,
            means: vec![F::zero(); nfeatures],
            cov: vec![vec![F::zero(); nfeatures]; nfeatures],
            pool,
        }
    }

    /// Update with a new sample
    pub fn update(&mut self, sample: &[F]) {
        assert_eq!(sample.len(), self.means.len(), "Feature dimension mismatch");

        self.n += 1;
        let n_f = F::from(self.n).unwrap();

        // Update means and covariance using Welford's algorithm
        let mut deltas = self.pool.acquire(sample.len());

        for i in 0..sample.len() {
            deltas[i] = sample[i] - self.means[i];
            self.means[i] = self.means[i] + deltas[i] / n_f;
        }

        // Update covariance matrix
        for i in 0..sample.len() {
            for j in i..sample.len() {
                let delta_prod = deltas[i] * (sample[j] - self.means[j]);
                self.cov[i][j] = self.cov[i][j] + delta_prod;
                if i != j {
                    self.cov[j][i] = self.cov[i][j]; // Symmetric
                }
            }
        }

        self.pool.release(deltas);
    }

    /// Get the current covariance matrix
    pub fn covariance(&self, ddof: usize) -> StatsResult<Vec<Vec<F>>> {
        if self.n <= ddof {
            return Err(StatsError::invalid_argument(
                "Not enough samples for given degrees of freedom",
            ));
        }

        let factor = F::from(self.n - ddof).unwrap();
        let mut result = self.cov.clone();

        for i in 0..result.len() {
            for j in 0..result[i].len() {
                result[i][j] = result[i][j] / factor;
            }
        }

        Ok(result)
    }
}

/// Memory-mapped statistics for huge files
#[cfg(feature = "memmap")]
pub struct MemoryMappedStats {
    mmap: Mmap,
    elementsize: usize,
    n_elements: usize,
}

#[cfg(feature = "memmap")]
impl MemoryMappedStats {
    pub fn new(path: &std::path::Path) -> StatsResult<Self> {
        use std::fs::OpenOptions;

        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| StatsError::computation(format!("Failed to open file: {}", e)))?;

        let metadata = file
            .metadata()
            .map_err(|e| StatsError::computation(format!("Failed to get metadata: {}", e)))?;

        let filesize = metadata.len() as usize;
        let elementsize = std::mem::size_of::<f64>(); // Assume f64 for now
        let n_elements = filesize / elementsize;

        unsafe {
            let mmap = Mmap::map(&file)
                .map_err(|e| StatsError::computation(format!("Failed to mmap: {}", e)))?;

            Ok(Self {
                mmap,
                elementsize,
                n_elements,
            })
        }
    }

    /// Compute mean without loading file into memory
    pub fn mean(&self) -> StatsResult<f64> {
        let data = unsafe {
            std::slice::from_raw_parts(self.mmap.as_ptr() as *const f64, self.n_elements)
        };

        // Process in chunks for cache efficiency
        let chunksize = 8192;
        let mut sum = 0.0;

        for chunk in data.chunks(chunksize) {
            sum += chunk.iter().sum::<f64>();
        }

        Ok(sum / self.n_elements as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    #[ignore = "timeout"]
    fn test_memory_pool() {
        let pool = MemoryPool::<f64>::new(MemoryConfig::default());

        let buf1 = pool.acquire(100);
        assert_eq!(buf1.len(), 100);

        pool.release(buf1);

        let buf2 = pool.acquire(100);
        assert_eq!(buf2.len(), 100);
    }

    #[test]
    fn test_lazy_stats() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = LazyStats::new(&data);

        assert!((stats.mean().unwrap() - 3.0).abs() < 1e-10);
        assert!((stats.variance(1).unwrap() - 2.5).abs() < 1e-10);

        let (min, max) = stats.minmax().unwrap();
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);

        assert!((stats.median().unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_covariance() {
        let pool = Rc::new(MemoryPool::new(MemoryConfig::default()));
        let mut cov = StreamingCovariance::new(2, pool);

        // Add samples
        cov.update(&[1.0, 2.0]);
        cov.update(&[2.0, 4.0]);
        cov.update(&[3.0, 6.0]);

        let result = cov.covariance(1).unwrap();

        // Should have perfect correlation
        assert!(result[0][0] > 0.0); // Variance of first feature
        assert!(result[1][1] > 0.0); // Variance of second feature
    }
}
