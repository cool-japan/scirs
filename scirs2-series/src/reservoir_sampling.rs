//! Streaming reservoir sampling for time series data
//!
//! This module provides algorithms for maintaining representative random samples
//! of streams whose total length is unknown.  All algorithms are O(k) space and
//! O(1) amortised time per element (where k is the reservoir capacity).
//!
//! # Algorithms
//!
//! - [`ReservoirSampler`] - Vitter's Algorithm R (uniform sampling)
//! - [`WeightedReservoir`] - Efraimidis-Spirakis Algorithm A-Res (weighted sampling)
//! - [`TimeSeriesReservoir`] - Time-decayed (recency-biased) reservoir sampling
//! - [`StreamStats`] - Wrapper combining a reservoir with stream-level statistics

use std::collections::VecDeque;

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Internal PRNG — xoshiro256** for deterministic, no-dep random number generation
// ---------------------------------------------------------------------------

/// Minimal xoshiro256** PRNG so we avoid an external `rand` dependency.
#[derive(Debug, Clone)]
struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    /// Seed from a single u64 via SplitMix64.
    fn from_seed(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for si in s.iter_mut() {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            let mut tmp = z;
            tmp = (tmp ^ (tmp >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            tmp = (tmp ^ (tmp >> 27)).wrapping_mul(0x94d049bb133111eb);
            *si = tmp ^ (tmp >> 31);
        }
        Self { s }
    }

    /// Generate the next u64 pseudo-random value.
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform u64 in `[0, n)` using the fast-rejection method.
    fn gen_range(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        // Rejection sampling to avoid modulo bias
        let threshold = n.wrapping_neg() % n;
        loop {
            let r = self.next_u64();
            if r >= threshold {
                return r % n;
            }
        }
    }

    /// Uniform f64 in `[0.0, 1.0)`.
    fn gen_f64(&mut self) -> f64 {
        // Use top 53 bits
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------
// ReservoirSampler<T> — Algorithm R
// ---------------------------------------------------------------------------

/// Uniform reservoir sampler using Vitter's Algorithm R.
///
/// Maintains a fixed-size random sample of a stream.  After `n` elements have
/// been seen, each element in the reservoir is equally likely to be any of the
/// `n` elements seen so far.  The algorithm requires no knowledge of the stream
/// length.
///
/// # Type parameter
///
/// `T: Clone` — items are cloned into the reservoir.
///
/// # Examples
///
/// ```
/// use scirs2_series::reservoir_sampling::ReservoirSampler;
///
/// let mut sampler = ReservoirSampler::<u32>::new(100, 42).expect("should succeed");
/// for i in 0..10_000u32 {
///     sampler.update(i);
/// }
/// assert_eq!(sampler.sample().len(), 100);
/// ```
#[derive(Debug, Clone)]
pub struct ReservoirSampler<T: Clone> {
    /// Maximum reservoir capacity k
    capacity: usize,
    /// The reservoir (at most `capacity` items)
    reservoir: Vec<T>,
    /// Number of elements seen (including those not in the reservoir)
    n_seen: u64,
    /// Internal PRNG
    rng: Xoshiro256,
}

impl<T: Clone> ReservoirSampler<T> {
    /// Create a new reservoir sampler.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Reservoir size k.  Must be ≥ 1.
    /// * `seed` - Seed for the internal PRNG.
    pub fn new(capacity: usize, seed: u64) -> Result<Self> {
        if capacity == 0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "capacity".to_string(),
                message: "Reservoir capacity must be at least 1".to_string(),
            });
        }
        Ok(Self {
            capacity,
            reservoir: Vec::with_capacity(capacity),
            n_seen: 0,
            rng: Xoshiro256::from_seed(seed),
        })
    }

    /// Incorporate one new item from the stream (Algorithm R).
    ///
    /// If the reservoir is not yet full the item is always added.  Otherwise
    /// it replaces a randomly chosen existing element with probability k/n.
    pub fn update(&mut self, item: T) {
        self.n_seen += 1;
        if self.reservoir.len() < self.capacity {
            // Phase 1: fill up to capacity
            self.reservoir.push(item);
        } else {
            // Phase 2: replace with probability k/n_seen
            let j = self.rng.gen_range(self.n_seen) as usize;
            if j < self.capacity {
                self.reservoir[j] = item;
            }
        }
    }

    /// Return a slice containing the current reservoir contents.
    ///
    /// The slice is unordered (reservoir order, not stream order).
    pub fn sample(&self) -> &[T] {
        &self.reservoir
    }

    /// Return the number of stream elements seen so far.
    pub fn n_seen(&self) -> u64 {
        self.n_seen
    }

    /// Return the reservoir capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return whether the reservoir is full.
    pub fn is_full(&self) -> bool {
        self.reservoir.len() == self.capacity
    }

    /// Clear the reservoir and reset the counter (but keep the same seed state).
    pub fn reset(&mut self) {
        self.reservoir.clear();
        self.n_seen = 0;
    }
}

// ---------------------------------------------------------------------------
// WeightedReservoir — Algorithm A-Res (Efraimidis-Spirakis)
// ---------------------------------------------------------------------------

/// Weighted reservoir sampler using the A-Res algorithm.
///
/// Each item `i` is assigned a key `u^{1/w_i}` where `u ~ Uniform(0,1)` and
/// `w_i > 0` is the item's weight.  The k items with the largest keys are
/// retained.  This produces a weighted sample without replacement where each
/// item's inclusion probability is proportional to its weight.
///
/// ## Reference
/// Efraimidis, P. S. & Spirakis, P. G. (2006).
/// *Weighted Random Sampling with a Reservoir.*
/// Information Processing Letters, 97(5), 181–185.
///
/// # Examples
///
/// ```
/// use scirs2_series::reservoir_sampling::WeightedReservoir;
///
/// let mut wr = WeightedReservoir::new(10, 123).expect("should succeed");
/// for i in 0..100usize {
///     wr.update(i, (i + 1) as f64).expect("should succeed");
/// }
/// assert_eq!(wr.sample().len(), 10);
/// // Higher-indexed (heavier) items should dominate the sample
/// ```
#[derive(Debug, Clone)]
pub struct WeightedReservoir<T: Clone> {
    /// Maximum reservoir capacity
    capacity: usize,
    /// Reservoir items with their keys: (key, item)
    heap: Vec<(f64, T)>,
    /// Minimum key currently in the heap
    min_key: f64,
    /// Index of the minimum-key element in `heap`
    min_idx: usize,
    /// Number of stream elements seen
    n_seen: u64,
    /// Internal PRNG
    rng: Xoshiro256,
}

impl<T: Clone> WeightedReservoir<T> {
    /// Create a new weighted reservoir sampler.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Reservoir size k ≥ 1.
    /// * `seed` - Seed for the internal PRNG.
    pub fn new(capacity: usize, seed: u64) -> Result<Self> {
        if capacity == 0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "capacity".to_string(),
                message: "Reservoir capacity must be at least 1".to_string(),
            });
        }
        Ok(Self {
            capacity,
            heap: Vec::with_capacity(capacity),
            min_key: 0.0,
            min_idx: 0,
            n_seen: 0,
            rng: Xoshiro256::from_seed(seed),
        })
    }

    /// Incorporate one new item from the stream.
    ///
    /// # Arguments
    ///
    /// * `item` - The value to potentially include in the sample.
    /// * `weight` - Positive weight associated with this item.
    pub fn update(&mut self, item: T, weight: f64) -> Result<()> {
        if weight <= 0.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "weight".to_string(),
                message: "Weights must be strictly positive".to_string(),
            });
        }
        self.n_seen += 1;

        // Compute A-Res key: u^(1/w) = exp(ln(u)/w)
        let u = self.rng.gen_f64().max(f64::EPSILON);
        let key = u.ln() / weight; // maximise this (less negative = larger key)

        if self.heap.len() < self.capacity {
            self.heap.push((key, item));
            if self.heap.len() == self.capacity {
                // Find initial minimum
                self.refresh_min();
            }
        } else if key > self.min_key {
            // Replace the minimum-key element
            self.heap[self.min_idx] = (key, item);
            self.refresh_min();
        }

        Ok(())
    }

    /// Scan the heap to find and cache the minimum key position.
    fn refresh_min(&mut self) {
        self.min_key = f64::INFINITY;
        self.min_idx = 0;
        for (i, (k, _)) in self.heap.iter().enumerate() {
            if *k < self.min_key {
                self.min_key = *k;
                self.min_idx = i;
            }
        }
    }

    /// Return a slice of `(key, item)` pairs currently in the reservoir.
    ///
    /// The keys are the A-Res keys (ln(u)/w), not the original weights.
    pub fn sample(&self) -> &[(f64, T)] {
        &self.heap
    }

    /// Return the items without their keys.
    pub fn items(&self) -> Vec<&T> {
        self.heap.iter().map(|(_, item)| item).collect()
    }

    /// Return the number of stream elements seen.
    pub fn n_seen(&self) -> u64 {
        self.n_seen
    }

    /// Return the reservoir capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ---------------------------------------------------------------------------
// TimeSeriesReservoir — time-decayed reservoir
// ---------------------------------------------------------------------------

/// Time-stamped observation stored in a time-series reservoir.
#[derive(Debug, Clone)]
pub struct TimedObservation {
    /// The value
    pub value: f64,
    /// Logical timestamp (e.g. stream index or wall-clock seconds)
    pub timestamp: f64,
}

/// Time-decayed (recency-biased) reservoir sampler for time series streams.
///
/// Uses the A-Res algorithm with exponentially decayed weights so that more
/// recent observations are more likely to appear in the reservoir:
///
///   w(t) = exp(−λ · (T_now − t))
///
/// where λ > 0 controls the decay rate.  Setting λ = 0 yields a uniform
/// reservoir equivalent to Algorithm R.
///
/// # Examples
///
/// ```
/// use scirs2_series::reservoir_sampling::TimeSeriesReservoir;
///
/// let mut tsr = TimeSeriesReservoir::new(20, 0.1, 0).expect("should succeed");
/// for t in 0..1000 {
///     tsr.update(t as f64 * 0.1, t as f64).expect("should succeed");
/// }
/// let sample = tsr.sample();
/// assert_eq!(sample.len(), 20);
/// // Most sampled timestamps should be large (recent)
/// let mean_ts: f64 = sample.iter().map(|o| o.timestamp).sum::<f64>() / 20.0;
/// assert!(mean_ts > 500.0, "sample should be biased toward recent timestamps");
/// ```
#[derive(Debug, Clone)]
pub struct TimeSeriesReservoir {
    inner: WeightedReservoir<TimedObservation>,
    /// Exponential decay rate λ
    decay_rate: f64,
    /// Current logical time (latest timestamp seen)
    current_time: f64,
}

impl TimeSeriesReservoir {
    /// Create a new time-series reservoir.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Reservoir size.
    /// * `decay_rate` - λ ≥ 0.  Higher values = stronger recency bias.
    /// * `seed` - PRNG seed.
    pub fn new(capacity: usize, decay_rate: f64, seed: u64) -> Result<Self> {
        if decay_rate < 0.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "decay_rate".to_string(),
                message: "Decay rate must be non-negative".to_string(),
            });
        }
        Ok(Self {
            inner: WeightedReservoir::new(capacity, seed)?,
            decay_rate,
            current_time: 0.0,
        })
    }

    /// Add a new observation to the reservoir.
    ///
    /// # Arguments
    ///
    /// * `value` - The time series value.
    /// * `timestamp` - Logical time of this observation (must be non-decreasing).
    pub fn update(&mut self, value: f64, timestamp: f64) -> Result<()> {
        if timestamp < self.current_time {
            return Err(TimeSeriesError::InvalidInput(
                "Timestamps must be non-decreasing".to_string(),
            ));
        }
        self.current_time = timestamp;
        // Recency-biased A-Res weight: w = exp(decay_rate * t).  Larger (later)
        // timestamps receive exponentially larger weight, so the A-Res key
        // ln(u)/w is closer to 0 (the kept end) for recent observations.
        // decay_rate == 0 reduces to uniform sampling (Algorithm R).
        let exponent = self.decay_rate * timestamp;
        // Guard against overflow of exp() for very large exponents: the key
        // ln(u)/w only needs the *relative* ordering, and exp() saturates to
        // f64::MAX rather than +inf so that ln(u)/w stays finite and strictly
        // ordered (more-recent items keep larger, i.e. closer-to-zero, keys).
        let weight = {
            let w = exponent.exp();
            if w.is_finite() && w > 0.0 {
                w
            } else if exponent > 0.0 {
                f64::MAX
            } else {
                f64::MIN_POSITIVE
            }
        };
        self.inner
            .update(TimedObservation { value, timestamp }, weight)
    }

    /// Return the current reservoir sample.
    pub fn sample(&self) -> Vec<&TimedObservation> {
        self.inner.items()
    }

    /// Return the current logical time (latest timestamp seen).
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Return the reservoir capacity.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Return the number of observations seen.
    pub fn n_seen(&self) -> u64 {
        self.inner.n_seen()
    }
}

// ---------------------------------------------------------------------------
// StreamStats — reservoir + streaming statistics
// ---------------------------------------------------------------------------

/// Running statistics over the full stream, combined with a reservoir sample.
///
/// Maintains:
/// - Exact count, mean and variance of all `n_seen` observations (Welford).
/// - A fixed-size uniform reservoir sample for offline analysis.
/// - A sliding window of the most recent `window_size` observations.
///
/// # Examples
///
/// ```
/// use scirs2_series::reservoir_sampling::StreamStats;
///
/// let mut ss = StreamStats::new(50, 200, 0).expect("should succeed");
/// for x in 0..1000 {
///     ss.update(x as f64);
/// }
/// assert_eq!(ss.n_seen, 1000);
/// assert!((ss.mean() - 499.5).abs() < 1.0);
/// assert_eq!(ss.reservoir.sample().len(), 50);
/// assert_eq!(ss.recent_window().len(), 200);
/// ```
#[derive(Debug, Clone)]
pub struct StreamStats {
    /// Number of observations seen so far
    pub n_seen: u64,
    /// Uniform reservoir of the stream
    pub reservoir: ReservoirSampler<f64>,
    /// Welford mean accumulator
    mean_acc: f64,
    /// Welford M2 accumulator
    m2_acc: f64,
    /// Sliding window of most recent observations
    window: VecDeque<f64>,
    /// Maximum window size
    window_size: usize,
}

impl StreamStats {
    /// Create a new `StreamStats`.
    ///
    /// # Arguments
    ///
    /// * `reservoir_capacity` - Number of items to maintain in the random sample.
    /// * `window_size` - Number of most-recent observations to retain verbatim.
    /// * `seed` - PRNG seed for the reservoir.
    pub fn new(reservoir_capacity: usize, window_size: usize, seed: u64) -> Result<Self> {
        if window_size == 0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "window_size".to_string(),
                message: "Window size must be at least 1".to_string(),
            });
        }
        Ok(Self {
            n_seen: 0,
            reservoir: ReservoirSampler::new(reservoir_capacity, seed)?,
            mean_acc: 0.0,
            m2_acc: 0.0,
            window: VecDeque::with_capacity(window_size + 1),
            window_size,
        })
    }

    /// Incorporate one new observation.
    pub fn update(&mut self, x: f64) {
        self.n_seen += 1;

        // Welford update
        let delta = x - self.mean_acc;
        self.mean_acc += delta / self.n_seen as f64;
        let delta2 = x - self.mean_acc;
        self.m2_acc += delta * delta2;

        // Reservoir update
        self.reservoir.update(x);

        // Sliding window update
        self.window.push_back(x);
        if self.window.len() > self.window_size {
            self.window.pop_front();
        }
    }

    /// Return the running mean of all observations seen so far.
    pub fn mean(&self) -> f64 {
        self.mean_acc
    }

    /// Return the sample variance of all observations seen so far.
    pub fn variance(&self) -> f64 {
        if self.n_seen < 2 {
            0.0
        } else {
            self.m2_acc / (self.n_seen - 1) as f64
        }
    }

    /// Return the sample standard deviation.
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Return the most recent `window_size` observations (or fewer if fewer have been seen).
    pub fn recent_window(&self) -> &VecDeque<f64> {
        &self.window
    }

    /// Return the sliding window mean.
    pub fn window_mean(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            self.window.iter().sum::<f64>() / self.window.len() as f64
        }
    }

    /// Return the sliding window variance.
    pub fn window_variance(&self) -> f64 {
        if self.window.len() < 2 {
            return 0.0;
        }
        let m = self.window_mean();
        let var: f64 = self.window.iter().map(|&x| (x - m).powi(2)).sum::<f64>();
        var / (self.window.len() - 1) as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservoir_sampler_fills() {
        let mut sampler = ReservoirSampler::<u32>::new(10, 42).expect("failed to create sampler");
        for i in 0..5u32 {
            sampler.update(i);
        }
        assert_eq!(sampler.sample().len(), 5);
        assert!(!sampler.is_full());

        for i in 5..15u32 {
            sampler.update(i);
        }
        assert_eq!(sampler.sample().len(), 10);
        assert!(sampler.is_full());
    }

    #[test]
    fn test_reservoir_sampler_n_seen() {
        let mut sampler = ReservoirSampler::<i32>::new(5, 0).expect("failed to create sampler");
        for i in 0..100 {
            sampler.update(i);
        }
        assert_eq!(sampler.n_seen(), 100);
        assert_eq!(sampler.sample().len(), 5);
    }

    #[test]
    fn test_reservoir_uniformity() {
        // Statistical test: each of 10 items should appear in the reservoir with
        // roughly equal frequency when sampling repeatedly.
        let n_trials = 2000;
        let mut counts = [0u32; 10];
        for seed in 0..n_trials as u64 {
            let mut sampler =
                ReservoirSampler::<usize>::new(5, seed).expect("failed to create sampler");
            for i in 0..10 {
                sampler.update(i);
            }
            for &item in sampler.sample() {
                counts[item] += 1;
            }
        }
        // Expected count per item ≈ 5 * 2000 / 10 = 1000
        let expected = 1000.0f64;
        for (i, &cnt) in counts.iter().enumerate() {
            let diff = (cnt as f64 - expected).abs() / expected;
            assert!(
                diff < 0.15,
                "Item {i} has biased count {cnt}, expected ≈{expected}"
            );
        }
    }

    #[test]
    fn test_reservoir_invalid_capacity() {
        assert!(ReservoirSampler::<i32>::new(0, 0).is_err());
    }

    #[test]
    fn test_weighted_reservoir_basic() {
        let mut wr = WeightedReservoir::<usize>::new(5, 7).expect("failed to create wr");
        for i in 0..20 {
            wr.update(i, (i + 1) as f64)
                .expect("unexpected None or Err");
        }
        assert_eq!(wr.sample().len(), 5);
    }

    #[test]
    fn test_weighted_reservoir_invalid_weight() {
        let mut wr = WeightedReservoir::<i32>::new(5, 0).expect("failed to create wr");
        assert!(wr.update(1, 0.0).is_err());
        assert!(wr.update(1, -1.0).is_err());
    }

    #[test]
    fn test_time_series_reservoir_basic() {
        let mut tsr = TimeSeriesReservoir::new(20, 0.0, 0).expect("failed to create tsr");
        for t in 0..200 {
            tsr.update(t as f64, t as f64)
                .expect("unexpected None or Err");
        }
        assert_eq!(tsr.sample().len(), 20);
        assert_eq!(tsr.n_seen(), 200);
    }

    #[test]
    fn test_time_series_reservoir_monotone_timestamps() {
        let mut tsr = TimeSeriesReservoir::new(5, 0.1, 0).expect("failed to create tsr");
        tsr.update(1.0, 1.0).expect("unexpected None or Err");
        tsr.update(2.0, 2.0).expect("unexpected None or Err");
        // Non-monotone timestamp should fail
        assert!(tsr.update(3.0, 1.5).is_err());
    }

    #[test]
    fn test_stream_stats_basic() {
        let mut ss = StreamStats::new(50, 100, 0).expect("failed to create ss");
        for x in 1..=100 {
            ss.update(x as f64);
        }
        assert_eq!(ss.n_seen, 100);
        assert!((ss.mean() - 50.5).abs() < 1e-9, "mean = {}", ss.mean());
        assert!(ss.std() > 0.0);
        assert_eq!(ss.reservoir.sample().len(), 50);
        assert_eq!(ss.recent_window().len(), 100);
    }

    #[test]
    fn test_stream_stats_window_trimming() {
        let mut ss = StreamStats::new(10, 5, 0).expect("failed to create ss");
        for x in 0..20 {
            ss.update(x as f64);
        }
        assert_eq!(ss.recent_window().len(), 5);
        // Window should contain the last 5: 15, 16, 17, 18, 19
        let w: Vec<f64> = ss.recent_window().iter().copied().collect();
        assert_eq!(w, vec![15.0, 16.0, 17.0, 18.0, 19.0]);
    }

    #[test]
    fn test_stream_stats_window_mean_var() {
        let mut ss = StreamStats::new(10, 10, 0).expect("failed to create ss");
        for &x in &[2.0f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            ss.update(x);
        }
        let wm = ss.window_mean();
        assert!((wm - 5.0).abs() < 1e-9, "window mean = {wm}");
    }
}
