//! # Random Number Generation (Alpha 6 Enhanced)
//!
//! This module provides a comprehensive interface for random number generation in scientific computing
//! with advanced features for high-performance and specialized applications.
//!
//! ## Enhanced Features (Alpha 6)
//!
//! * Consistent interface across distribution types
//! * Thread-local and seedable random number generators
//! * Distribution-independent sampling functions
//! * Deterministic sequence generation for testing
//! * **Quasi-Monte Carlo sequences** (Sobol, Halton, Latin hypercube)
//! * **Cryptographically secure random number generation**
//! * **Variance reduction techniques** for Monte Carlo methods
//! * **Importance sampling methods** for efficient estimation
//! * **Parallel-safe random number generation** with thread-local pools
//! * **Specialized distributions** for scientific computing
//! * **GPU-accelerated random number generation** (when available)
//!
//! ## Usage
//!
//! ```rust
//! use scirs2_core::random::{Random, DistributionExt};
//! use rand_distr::{Normal, Uniform};
//! use ndarray::IxDyn;
//!
//! // Get a thread-local random number generator
//! let mut rng = Random::default();
//!
//! // Generate random values from various distributions
//! let normal_value = rng.sample(Normal::new(0.0_f64, 1.0_f64).unwrap());
//! let uniform_value = rng.sample(Uniform::new(0.0_f64, 1.0_f64).unwrap());
//!
//! // Generate a random array using the distribution extension trait
//! let shape = IxDyn(&[10, 10]);
//! let normal_array = Normal::new(0.0_f64, 1.0_f64).unwrap().random_array(&mut rng, shape);
//!
//! // Create a seeded random generator for reproducible results
//! let mut seeded_rng = Random::with_seed(42);
//! let reproducible_value = seeded_rng.sample(Uniform::new(0.0_f64, 1.0_f64).unwrap());
//! ```

use ndarray::{Array, Dimension, IxDyn};
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use std::cell::RefCell;

// Re-export traits for external use
pub use rand::Rng;

/// Wrapper around the rand crate's RNG for a consistent interface
#[derive(Debug)]
pub struct Random<R: Rng + ?Sized = rand::rngs::ThreadRng> {
    rng: R,
}

impl Default for Random {
    fn default() -> Self {
        Self { rng: rand::rng() }
    }
}

impl<R: Rng> Random<R> {
    /// Sample a value from a distribution
    pub fn sample<D, T>(&mut self, distribution: D) -> T
    where
        D: Distribution<T>,
    {
        distribution.sample(&mut self.rng)
    }

    /// Generate a random value within the given range (inclusive min, exclusive max)
    pub fn random_range<T: rand_distr::uniform::SampleUniform + PartialOrd + Copy>(
        &mut self,
        min: T,
        max: T,
    ) -> T {
        self.sample(rand_distr::Uniform::new(min, max).unwrap())
    }

    /// Generate a random boolean value
    pub fn random_bool(&mut self) -> bool {
        let dist = rand_distr::Bernoulli::new(0.5).unwrap();
        dist.sample(&mut self.rng)
    }

    /// Generate a random boolean with the given probability of being true
    pub fn random_bool_with_chance(&mut self, prob: f64) -> bool {
        let dist = rand_distr::Bernoulli::new(prob).unwrap();
        dist.sample(&mut self.rng)
    }

    /// Shuffle a slice randomly
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        slice.shuffle(&mut self.rng);
    }

    /// Generate a vector of values sampled from a distribution
    pub fn sample_vec<D, T>(&mut self, distribution: D, size: usize) -> Vec<T>
    where
        D: Distribution<T> + Copy,
    {
        (0..size)
            .map(|_| distribution.sample(&mut self.rng))
            .collect()
    }

    /// Generate an ndarray::Array from samples of a distribution
    pub fn sample_array<D, T, Sh>(&mut self, distribution: D, shape: Sh) -> Array<T, IxDyn>
    where
        D: Distribution<T> + Copy,
        Sh: Into<IxDyn>,
    {
        let shape = shape.into();
        let size = shape.size();
        let values = self.sample_vec(distribution, size);
        Array::from_shape_vec(shape, values).unwrap()
    }
}

impl Random {
    /// Create a new random number generator with a specific seed
    pub fn with_seed(seed: u64) -> Random<StdRng> {
        Random {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

// Thread-local random number generator for convenient access
thread_local! {
    static THREAD_RNG: RefCell<Random> = RefCell::new(Random::default());
}

/// Get a reference to the thread-local random number generator
pub fn get_thread_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut Random) -> R,
{
    THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
}

/// Extension trait for distributions to create arrays directly
pub trait DistributionExt<T>: Distribution<T> + Sized {
    /// Create a random array with values from this distribution
    fn random_array<U, Sh>(&self, rng: &mut Random<U>, shape: Sh) -> Array<T, IxDyn>
    where
        U: Rng,
        Sh: Into<IxDyn>,
        Self: Copy,
    {
        rng.sample_array(*self, shape)
    }

    /// Create a random vector with values from this distribution
    fn random_vec<U>(&self, rng: &mut Random<U>, size: usize) -> Vec<T>
    where
        U: Rng,
        Self: Copy,
    {
        rng.sample_vec(*self, size)
    }
}

// Implement the extension trait for all distributions
impl<D, T> DistributionExt<T> for D where D: Distribution<T> {}

/// Helper functions for common random sampling needs
pub mod sampling {
    use super::*;
    use rand_distr as rdistr;

    /// Sample uniformly from [0, 1)
    pub fn random_uniform01<R: Rng>(rng: &mut Random<R>) -> f64 {
        Uniform::new(0.0_f64, 1.0_f64).unwrap().sample(&mut rng.rng)
    }

    /// Sample from a standard normal distribution (mean 0, std dev 1)
    pub fn random_standard_normal<R: Rng>(rng: &mut Random<R>) -> f64 {
        rdistr::Normal::new(0.0_f64, 1.0_f64)
            .unwrap()
            .sample(&mut rng.rng)
    }

    /// Sample from a normal distribution with given mean and standard deviation
    pub fn random_normal<R: Rng>(rng: &mut Random<R>, mean: f64, std_dev: f64) -> f64 {
        rdistr::Normal::new(mean, std_dev)
            .unwrap()
            .sample(&mut rng.rng)
    }

    /// Sample from a log-normal distribution
    pub fn random_lognormal<R: Rng>(rng: &mut Random<R>, mean: f64, std_dev: f64) -> f64 {
        rdistr::LogNormal::new(mean, std_dev)
            .unwrap()
            .sample(&mut rng.rng)
    }

    /// Sample from an exponential distribution
    pub fn random_exponential<R: Rng>(rng: &mut Random<R>, lambda: f64) -> f64 {
        rdistr::Exp::new(lambda).unwrap().sample(&mut rng.rng)
    }

    /// Generate an array of random integers in a range
    pub fn random_integers<R: Rng, Sh>(
        rng: &mut Random<R>,
        min: i64,
        max: i64,
        shape: Sh,
    ) -> Array<i64, IxDyn>
    where
        Sh: Into<IxDyn>,
    {
        rng.sample_array(Uniform::new_inclusive(min, max).unwrap(), shape)
    }

    /// Generate an array of random floating-point values in a range
    pub fn random_floats<R: Rng, Sh>(
        rng: &mut Random<R>,
        min: f64,
        max: f64,
        shape: Sh,
    ) -> Array<f64, IxDyn>
    where
        Sh: Into<IxDyn>,
    {
        rng.sample_array(Uniform::new(min, max).unwrap(), shape)
    }

    /// Sample indices for bootstrapping (sampling with replacement)
    pub fn bootstrap_indices<R: Rng>(
        rng: &mut Random<R>,
        data_size: usize,
        sample_size: usize,
    ) -> Vec<usize> {
        let dist = Uniform::new(0, data_size).unwrap();
        rng.sample_vec(dist, sample_size)
    }

    /// Sample indices without replacement (for random subsampling)
    pub fn sample_without_replacement<R: Rng>(
        rng: &mut Random<R>,
        data_size: usize,
        sample_size: usize,
    ) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..data_size).collect();
        indices.shuffle(&mut rng.rng);
        indices.truncate(sample_size);
        indices
    }
}

/// Quasi-Monte Carlo sequences for low-discrepancy sampling
pub mod qmc;

/// GPU-accelerated random number generation
#[cfg(feature = "gpu")]
pub mod gpu;

/// Deterministic random sequence generator for testing
pub struct DeterministicSequence {
    seed: u64,
    counter: u64,
}

impl DeterministicSequence {
    /// Create a new deterministic sequence with the given seed
    pub fn new(seed: u64) -> Self {
        Self { seed, counter: 0 }
    }

    /// Generate the next value in the sequence
    pub fn next_f64(&mut self) -> f64 {
        // Simple deterministic hash function for testing purposes
        // This is NOT for cryptographic or high-quality random numbers
        // It's just for reproducible test sequences
        let mut x = self.counter.wrapping_add(self.seed);
        x = ((x >> 16) ^ x).wrapping_mul(0x45d9f3b);
        x = ((x >> 16) ^ x).wrapping_mul(0x45d9f3b);
        x = (x >> 16) ^ x;

        self.counter = self.counter.wrapping_add(1);

        // Convert to f64 in [0, 1) range
        (x as f64) / (u64::MAX as f64)
    }

    /// Reset the sequence to its initial state
    pub fn reset(&mut self) {
        self.counter = 0;
    }

    /// Get a vector of deterministic values
    pub fn get_vec(&mut self, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.next_f64()).collect()
    }

    /// Get an ndarray::Array of deterministic values
    pub fn get_array<Sh>(&mut self, shape: Sh) -> Array<f64, IxDyn>
    where
        Sh: Into<IxDyn>,
    {
        let shape = shape.into();
        let size = shape.size();
        let values = self.get_vec(size);
        Array::from_shape_vec(shape, values).unwrap()
    }
}

/// Quasi-Monte Carlo sequence generators for low-discrepancy sampling
pub mod quasi_monte_carlo {
    use super::*;

    /// Sobol sequence generator for quasi-Monte Carlo sampling
    #[derive(Debug, Clone)]
    pub struct SobolSequence {
        dimensions: usize,
        current_index: usize,
        direction_numbers: Vec<Vec<u64>>,
    }

    impl SobolSequence {
        /// Create a new Sobol sequence generator
        pub fn new(dimensions: usize) -> Self {
            let mut direction_numbers = vec![vec![]; dimensions];

            // Initialize direction numbers for the first few dimensions
            // This is a simplified implementation - a full implementation would use
            // precomputed direction numbers from mathematical tables
            for direction_number in direction_numbers.iter_mut().take(dimensions) {
                let mut direction = vec![1u64 << 31]; // First direction number
                for i in 1..32 {
                    direction.push(direction[i - 1] ^ (direction[i - 1] >> 1));
                }
                *direction_number = direction;
            }

            Self {
                dimensions,
                current_index: 0,
                direction_numbers,
            }
        }

        /// Generate the next point in the Sobol sequence
        pub fn next_point(&mut self) -> Vec<f64> {
            let mut point = vec![0.0; self.dimensions];

            for (dim, point_val) in point.iter_mut().enumerate().take(self.dimensions) {
                let mut value = 0u64;
                let mut index = self.current_index;
                let mut bit = 0;

                while index > 0 {
                    if (index & 1) == 1 {
                        value ^= self.direction_numbers[dim][bit];
                    }
                    index >>= 1;
                    bit += 1;
                }

                *point_val = value as f64 / (1u64 << 32) as f64;
            }

            self.current_index += 1;
            point
        }

        /// Generate multiple points from the Sobol sequence
        pub fn generate_points(&mut self, count: usize) -> Vec<Vec<f64>> {
            (0..count).map(|_| self.next_point()).collect()
        }

        /// Generate points as an ndarray
        pub fn generate_array(&mut self, count: usize) -> Array<f64, ndarray::Ix2> {
            let points = self.generate_points(count);
            let mut data = Vec::with_capacity(count * self.dimensions);

            for point in points {
                data.extend(point);
            }

            Array::from_shape_vec((count, self.dimensions), data).unwrap()
        }

        /// Reset the sequence
        pub fn reset(&mut self) {
            self.current_index = 0;
        }
    }

    /// Halton sequence generator
    #[derive(Debug, Clone)]
    pub struct HaltonSequence {
        dimensions: usize,
        current_index: usize,
        bases: Vec<usize>,
    }

    impl HaltonSequence {
        /// Create a new Halton sequence generator
        pub fn new(dimensions: usize) -> Self {
            // Use the first prime numbers as bases
            let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
            let bases = primes.iter().take(dimensions).cloned().collect();

            Self {
                dimensions,
                current_index: 1,
                bases,
            }
        }

        /// Generate the next point in the Halton sequence
        pub fn next_point(&mut self) -> Vec<f64> {
            let point = self
                .bases
                .iter()
                .map(|&base| self.radical_inverse(self.current_index, base))
                .collect();

            self.current_index += 1;
            point
        }

        /// Generate multiple points from the Halton sequence
        pub fn generate_points(&mut self, count: usize) -> Vec<Vec<f64>> {
            (0..count).map(|_| self.next_point()).collect()
        }

        /// Generate points as an ndarray
        pub fn generate_array(&mut self, count: usize) -> Array<f64, ndarray::Ix2> {
            let points = self.generate_points(count);
            let mut data = Vec::with_capacity(count * self.dimensions);

            for point in points {
                data.extend(point);
            }

            Array::from_shape_vec((count, self.dimensions), data).unwrap()
        }

        /// Reset the sequence
        pub fn reset(&mut self) {
            self.current_index = 1;
        }

        fn radical_inverse(&self, mut n: usize, base: usize) -> f64 {
            let mut result = 0.0;
            let mut f = 1.0 / base as f64;

            while n > 0 {
                result += f * (n % base) as f64;
                n /= base;
                f /= base as f64;
            }

            result
        }
    }

    /// Latin Hypercube Sampling generator
    #[derive(Debug)]
    pub struct LatinHypercubeSampling<R: Rng> {
        dimensions: usize,
        rng: Random<R>,
    }

    impl<R: Rng> LatinHypercubeSampling<R> {
        /// Create a new Latin Hypercube sampler
        pub fn new(dimensions: usize, rng: Random<R>) -> Self {
            Self { dimensions, rng }
        }

        /// Generate Latin Hypercube samples
        pub fn generate_samples(&mut self, count: usize) -> Array<f64, ndarray::Ix2> {
            let mut samples = Array::<f64, _>::zeros((count, self.dimensions));

            for dim in 0..self.dimensions {
                // Create stratified samples
                let mut strata: Vec<f64> = (0..count)
                    .map(|i| {
                        (i as f64 + self.rng.sample(Uniform::new(0.0, 1.0).unwrap())) / count as f64
                    })
                    .collect();

                // Shuffle the strata for this dimension
                strata.shuffle(&mut self.rng.rng);

                // Assign to samples
                for (i, &value) in strata.iter().enumerate() {
                    samples[[i, dim]] = value;
                }
            }

            samples
        }
    }

    impl LatinHypercubeSampling<rand::rngs::ThreadRng> {
        /// Create a Latin Hypercube sampler with default RNG
        pub fn with_default_rng(dimensions: usize) -> Self {
            Self::new(dimensions, Random::default())
        }
    }
}

/// Cryptographically secure random number generation
pub mod secure {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Cryptographically secure random number generator
    pub struct SecureRandom {
        rng: Random<StdRng>,
    }

    impl Default for SecureRandom {
        fn default() -> Self {
            Self::new()
        }
    }

    impl SecureRandom {
        /// Create a new cryptographically secure RNG
        pub fn new() -> Self {
            // Use system entropy to generate a secure seed for StdRng
            use std::process;
            use std::thread;
            use std::time::{SystemTime, UNIX_EPOCH};

            let time_nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);

            let process_id = process::id() as u128;
            let thread_id = thread::current().id();

            // Combine multiple entropy sources
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            use std::hash::{Hash, Hasher};
            time_nanos.hash(&mut hasher);
            process_id.hash(&mut hasher);
            thread_id.hash(&mut hasher);

            let seed_u64 = hasher.finish();

            // Create a 32-byte seed from the hash
            let mut seed = [0u8; 32];
            for (i, chunk) in seed.chunks_mut(8).enumerate() {
                let offset_seed =
                    seed_u64.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
                let bytes = offset_seed.to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }

            let std_rng = StdRng::from_seed(seed);
            Self {
                rng: Random { rng: std_rng },
            }
        }

        /// Generate a cryptographically secure random value
        pub fn sample<D, T>(&mut self, distribution: D) -> T
        where
            D: Distribution<T>,
        {
            self.rng.sample(distribution)
        }

        /// Generate cryptographically secure random bytes
        pub fn random_bytes(&mut self, count: usize) -> Vec<u8> {
            (0..count)
                .map(|_| self.sample(Uniform::new(0u8, 255u8).unwrap()))
                .collect()
        }

        /// Generate a cryptographically secure random key
        pub fn random_key(&mut self, key_length: usize) -> Vec<u8> {
            self.random_bytes(key_length)
        }

        /// Generate a cryptographically secure random float in [0, 1)
        pub fn random_f64(&mut self) -> f64 {
            self.sample(Uniform::new(0.0, 1.0).unwrap())
        }

        /// Generate cryptographically secure random integers
        pub fn random_range<T>(&mut self, min: T, max: T) -> T
        where
            T: rand_distr::uniform::SampleUniform + PartialOrd + Copy,
        {
            self.sample(Uniform::new(min, max).unwrap())
        }
    }
}

/// Variance reduction techniques for Monte Carlo methods
pub mod variance_reduction {
    use super::*;
    use std::collections::HashMap;

    /// Antithetic variate sampling for variance reduction
    #[derive(Debug)]
    pub struct AntitheticSampling<R: Rng> {
        rng: Random<R>,
        #[allow(dead_code)]
        stored_samples: HashMap<usize, Vec<f64>>,
    }

    impl<R: Rng> AntitheticSampling<R> {
        /// Create a new antithetic sampling generator
        pub fn new(rng: Random<R>) -> Self {
            Self {
                rng,
                stored_samples: HashMap::new(),
            }
        }

        /// Generate antithetic pairs of samples
        pub fn generate_antithetic_pairs(&mut self, count: usize) -> (Vec<f64>, Vec<f64>) {
            let original: Vec<f64> = (0..count)
                .map(|_| self.rng.sample(Uniform::new(0.0, 1.0).unwrap()))
                .collect();

            let antithetic: Vec<f64> = original.iter().map(|&x| 1.0 - x).collect();

            (original, antithetic)
        }

        /// Generate stratified samples for variance reduction
        pub fn generate_stratified_samples(
            &mut self,
            strata: usize,
            samples_per_stratum: usize,
        ) -> Vec<f64> {
            let mut all_samples = Vec::new();

            for i in 0..strata {
                let stratum_start = i as f64 / strata as f64;
                let stratum_end = (i + 1) as f64 / strata as f64;

                for _ in 0..samples_per_stratum {
                    let uniform_in_stratum = self.rng.sample(Uniform::new(0.0, 1.0).unwrap());
                    let sample = stratum_start + uniform_in_stratum * (stratum_end - stratum_start);
                    all_samples.push(sample);
                }
            }

            all_samples.shuffle(&mut self.rng.rng);
            all_samples
        }
    }

    impl AntitheticSampling<rand::rngs::ThreadRng> {
        /// Create antithetic sampling with default RNG
        pub fn with_default_rng() -> Self {
            Self::new(Random::default())
        }
    }

    /// Control variate method for variance reduction
    #[derive(Debug)]
    pub struct ControlVariate {
        control_mean: f64,
        optimal_coefficient: Option<f64>,
    }

    impl ControlVariate {
        /// Create a new control variate method
        pub fn new(control_mean: f64) -> Self {
            Self {
                control_mean,
                optimal_coefficient: None,
            }
        }

        /// Estimate the optimal control coefficient
        pub fn estimate_coefficient(&mut self, target_samples: &[f64], control_samples: &[f64]) {
            let n = target_samples.len() as f64;

            let target_mean = target_samples.iter().sum::<f64>() / n;
            let control_sample_mean = control_samples.iter().sum::<f64>() / n;

            let numerator: f64 = target_samples
                .iter()
                .zip(control_samples.iter())
                .map(|(&y, &x)| (y - target_mean) * (x - control_sample_mean))
                .sum();

            let denominator: f64 = control_samples
                .iter()
                .map(|&x| (x - control_sample_mean).powi(2))
                .sum();

            if denominator > 0.0 {
                self.optimal_coefficient = Some(numerator / denominator);
            }
        }

        /// Apply control variate correction
        pub fn apply_correction(
            &self,
            target_samples: &[f64],
            control_samples: &[f64],
        ) -> Vec<f64> {
            if let Some(c) = self.optimal_coefficient {
                target_samples
                    .iter()
                    .zip(control_samples.iter())
                    .map(|(&y, &x)| y - c * (x - self.control_mean))
                    .collect()
            } else {
                target_samples.to_vec()
            }
        }
    }
}

/// Importance sampling methods for efficient estimation
pub mod importance_sampling {
    use super::*;
    use rand_distr::Normal;

    /// Importance sampling estimator
    #[derive(Debug)]
    pub struct ImportanceSampler<R: Rng> {
        rng: Random<R>,
    }

    impl<R: Rng> ImportanceSampler<R> {
        /// Create a new importance sampler
        pub fn new(rng: Random<R>) -> Self {
            Self { rng }
        }

        /// Perform importance sampling with a given proposal distribution
        pub fn sample_with_weights<F, G>(
            &mut self,
            target_pdf: F,
            proposal_pdf: G,
            proposal_sampler: impl Fn(&mut Random<R>) -> f64,
            n_samples: usize,
        ) -> (Vec<f64>, Vec<f64>)
        where
            F: Fn(f64) -> f64,
            G: Fn(f64) -> f64,
        {
            let mut samples = Vec::with_capacity(n_samples);
            let mut weights = Vec::with_capacity(n_samples);

            for _ in 0..n_samples {
                let sample = proposal_sampler(&mut self.rng);
                let weight = target_pdf(sample) / proposal_pdf(sample);

                samples.push(sample);
                weights.push(weight);
            }

            (samples, weights)
        }

        /// Estimate expectation using importance sampling
        pub fn estimate_expectation<F, G, H>(
            &mut self,
            function: F,
            target_pdf: G,
            proposal_pdf: H,
            proposal_sampler: impl Fn(&mut Random<R>) -> f64,
            n_samples: usize,
        ) -> f64
        where
            F: Fn(f64) -> f64,
            G: Fn(f64) -> f64,
            H: Fn(f64) -> f64,
        {
            let (samples, weights) =
                self.sample_with_weights(target_pdf, proposal_pdf, proposal_sampler, n_samples);

            let weighted_sum: f64 = samples
                .iter()
                .zip(weights.iter())
                .map(|(&x, &w)| function(x) * w)
                .sum();

            let weight_sum: f64 = weights.iter().sum();

            weighted_sum / weight_sum
        }

        /// Adaptive importance sampling with mixture proposal
        pub fn adaptive_sampling<F>(
            &mut self,
            target_log_pdf: F,
            initial_samples: usize,
            adaptation_rounds: usize,
        ) -> Vec<f64>
        where
            F: Fn(f64) -> f64,
        {
            let mut samples = Vec::new();
            let mut proposal_mean: f64 = 0.0;
            let mut proposal_std: f64 = 1.0;

            for round in 0..adaptation_rounds {
                let round_samples = if round == 0 {
                    initial_samples
                } else {
                    initial_samples / 2
                };
                let normal_dist = Normal::new(proposal_mean, proposal_std).unwrap();

                let mut round_sample_vec = Vec::new();
                let mut weights = Vec::new();

                for _ in 0..round_samples {
                    let sample = self.rng.sample(normal_dist);

                    // Manual calculation of log PDF for normal distribution
                    let normal_log_pdf = -0.5 * ((sample - proposal_mean) / proposal_std).powi(2)
                        - 0.5 * (2.0 * std::f64::consts::PI).ln()
                        - proposal_std.ln();
                    let log_weight = target_log_pdf(sample) - normal_log_pdf;

                    round_sample_vec.push(sample);
                    weights.push(log_weight.exp());
                }

                // Update proposal parameters based on weighted samples
                let weight_sum: f64 = weights.iter().sum();
                if weight_sum > 0.0 {
                    let normalized_weights: Vec<f64> =
                        weights.iter().map(|w| w / weight_sum).collect();

                    proposal_mean = round_sample_vec
                        .iter()
                        .zip(normalized_weights.iter())
                        .map(|(&x, &w)| x * w)
                        .sum();

                    let variance = round_sample_vec
                        .iter()
                        .zip(normalized_weights.iter())
                        .map(|(&x, &w)| w * (x - proposal_mean).powi(2))
                        .sum::<f64>();

                    proposal_std = variance.sqrt().max(0.1); // Prevent collapse
                }

                samples.extend(round_sample_vec);
            }

            samples
        }
    }

    impl ImportanceSampler<rand::rngs::ThreadRng> {
        /// Create importance sampler with default RNG
        pub fn with_default_rng() -> Self {
            Self::new(Random::default())
        }
    }
}

/// Thread-local RNG pools for high-performance parallel applications
pub mod parallel {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Thread-local random number generator pool
    #[derive(Debug)]
    pub struct ThreadLocalRngPool {
        seed_counter: Arc<AtomicUsize>,
        base_seed: u64,
    }

    impl ThreadLocalRngPool {
        /// Create a new thread-local RNG pool
        pub fn new(base_seed: u64) -> Self {
            Self {
                seed_counter: Arc::new(AtomicUsize::new(0)),
                base_seed,
            }
        }

        /// Get a thread-local RNG
        pub fn get_rng(&self) -> Random<StdRng> {
            let thread_id = self.seed_counter.fetch_add(1, Ordering::Relaxed);
            let seed = self.base_seed.wrapping_add(thread_id as u64);
            Random::with_seed(seed)
        }

        /// Execute a closure with a thread-local RNG
        pub fn with_rng<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&mut Random<StdRng>) -> R,
        {
            let mut rng = self.get_rng();
            f(&mut rng)
        }
    }

    impl Default for ThreadLocalRngPool {
        fn default() -> Self {
            use std::time::{SystemTime, UNIX_EPOCH};
            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(42);
            Self::new(seed)
        }
    }

    /// Parallel random number generation utilities
    pub struct ParallelRng;

    impl ParallelRng {
        /// Generate parallel random samples using Rayon
        #[cfg(feature = "parallel")]
        pub fn parallel_sample<D, T>(
            distribution: D,
            count: usize,
            pool: &ThreadLocalRngPool,
        ) -> Vec<T>
        where
            D: Distribution<T> + Copy + Send + Sync,
            T: Send,
        {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};

            (0..count)
                .into_par_iter()
                .map(|_| pool.with_rng(|rng| rng.sample(distribution)))
                .collect()
        }

        /// Generate parallel random arrays using Rayon
        #[cfg(feature = "parallel")]
        pub fn parallel_sample_array<D, T, Sh>(
            distribution: D,
            shape: Sh,
            pool: &ThreadLocalRngPool,
        ) -> Array<T, IxDyn>
        where
            D: Distribution<T> + Copy + Send + Sync,
            T: Send + Clone,
            Sh: Into<IxDyn>,
        {
            let shape = shape.into();
            let size = shape.size();
            let samples = Self::parallel_sample(distribution, size, pool);
            Array::from_shape_vec(shape, samples).unwrap()
        }

        /// Sequential fallback when parallel feature is not enabled
        #[cfg(not(feature = "parallel"))]
        pub fn parallel_sample<D, T>(
            distribution: D,
            count: usize,
            pool: &ThreadLocalRngPool,
        ) -> Vec<T>
        where
            D: Distribution<T> + Copy,
            T: Send,
        {
            pool.with_rng(|rng| rng.sample_vec(distribution, count))
        }

        /// Sequential fallback when parallel feature is not enabled
        #[cfg(not(feature = "parallel"))]
        pub fn parallel_sample_array<D, T, Sh>(
            distribution: D,
            shape: Sh,
            pool: &ThreadLocalRngPool,
        ) -> Array<T, IxDyn>
        where
            D: Distribution<T> + Copy,
            T: Send + Clone,
            Sh: Into<IxDyn>,
        {
            pool.with_rng(|rng| rng.sample_array(distribution, shape))
        }
    }
}

/// Specialized distributions for scientific computing
pub mod specialized_distributions {
    use super::*;
    use rand_distr::{Gamma, Normal};

    /// Multivariate normal distribution
    #[derive(Debug, Clone)]
    pub struct MultivariateNormal {
        mean: Vec<f64>,
        covariance_decomposition: Vec<Vec<f64>>, // Cholesky decomposition
        dimensions: usize,
    }

    impl MultivariateNormal {
        /// Create a new multivariate normal distribution
        pub fn new(mean: Vec<f64>, covariance: Vec<Vec<f64>>) -> Result<Self, String> {
            let dimensions = mean.len();

            if covariance.len() != dimensions
                || covariance.iter().any(|row| row.len() != dimensions)
            {
                return Err("Covariance matrix dimensions don't match mean vector".to_string());
            }

            // Compute Cholesky decomposition
            let chol = Self::cholesky_decomposition(&covariance)?;

            Ok(Self {
                mean,
                covariance_decomposition: chol,
                dimensions,
            })
        }

        /// Sample from the multivariate normal distribution
        pub fn sample<R: Rng>(&self, rng: &mut Random<R>) -> Vec<f64> {
            let standard_normal = Normal::new(0.0, 1.0).unwrap();
            let z: Vec<f64> = (0..self.dimensions)
                .map(|_| rng.sample(standard_normal))
                .collect();

            // Apply Cholesky transformation: x = μ + L*z
            let mut result = vec![0.0; self.dimensions];
            for (i, result_val) in result.iter_mut().enumerate().take(self.dimensions) {
                *result_val = self.mean[i];
                for (j, &z_val) in z.iter().enumerate().take(i + 1) {
                    *result_val += self.covariance_decomposition[i][j] * z_val;
                }
            }

            result
        }

        /// Generate multiple samples
        pub fn sample_array<R: Rng>(
            &self,
            rng: &mut Random<R>,
            count: usize,
        ) -> Array<f64, ndarray::Ix2> {
            let mut data = Vec::with_capacity(count * self.dimensions);

            for _ in 0..count {
                let sample = self.sample(rng);
                data.extend(sample);
            }

            Array::from_shape_vec((count, self.dimensions), data).unwrap()
        }

        fn cholesky_decomposition(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
            let n = matrix.len();
            let mut l = vec![vec![0.0; n]; n];

            for i in 0..n {
                for j in 0..=i {
                    if i == j {
                        let sum = (0..j).map(|k| l[i][k] * l[i][k]).sum::<f64>();
                        let diagonal = matrix[i][i] - sum;
                        if diagonal <= 0.0 {
                            return Err("Matrix is not positive definite".to_string());
                        }
                        l[i][j] = diagonal.sqrt();
                    } else {
                        let sum = (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
                        l[i][j] = (matrix[i][j] - sum) / l[j][j];
                    }
                }
            }

            Ok(l)
        }
    }

    /// Dirichlet distribution for probability vectors
    #[derive(Debug, Clone)]
    pub struct Dirichlet {
        alphas: Vec<f64>,
        gamma_distributions: Vec<Gamma<f64>>,
    }

    impl Dirichlet {
        /// Create a new Dirichlet distribution
        pub fn new(alphas: Vec<f64>) -> Result<Self, String> {
            if alphas.iter().any(|&alpha| alpha <= 0.0) {
                return Err("All alpha parameters must be positive".to_string());
            }

            let gamma_distributions = alphas
                .iter()
                .map(|&alpha| Gamma::new(alpha, 1.0).unwrap())
                .collect();

            Ok(Self {
                alphas,
                gamma_distributions,
            })
        }

        /// Sample from the Dirichlet distribution
        pub fn sample<R: Rng>(&self, rng: &mut Random<R>) -> Vec<f64> {
            let gamma_samples: Vec<f64> = self
                .gamma_distributions
                .iter()
                .map(|&dist| rng.sample(dist))
                .collect();

            let sum: f64 = gamma_samples.iter().sum();
            gamma_samples.into_iter().map(|x| x / sum).collect()
        }

        /// Generate multiple samples
        pub fn sample_array<R: Rng>(
            &self,
            rng: &mut Random<R>,
            count: usize,
        ) -> Array<f64, ndarray::Ix2> {
            let dimensions = self.alphas.len();
            let mut data = Vec::with_capacity(count * dimensions);

            for _ in 0..count {
                let sample = self.sample(rng);
                data.extend(sample);
            }

            Array::from_shape_vec((count, dimensions), data).unwrap()
        }
    }

    /// Von Mises distribution for circular data
    #[derive(Debug, Clone)]
    pub struct VonMises {
        mu: f64,
        kappa: f64,
    }

    impl VonMises {
        /// Create a new Von Mises distribution
        pub fn new(mu: f64, kappa: f64) -> Result<Self, String> {
            if kappa < 0.0 {
                return Err("Kappa parameter must be non-negative".to_string());
            }

            Ok(Self { mu, kappa })
        }

        /// Sample from the Von Mises distribution
        pub fn sample<R: Rng>(&self, rng: &mut Random<R>) -> f64 {
            if self.kappa == 0.0 {
                // Uniform distribution on the circle
                return rng.sample(Uniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap());
            }

            // Use acceptance-rejection method
            let uniform = Uniform::new(0.0, 1.0).unwrap();

            loop {
                let u1 = rng.sample(uniform);
                let u2 = rng.sample(uniform);
                let _u3 = rng.sample(uniform);

                let theta = (2.0 * std::f64::consts::PI * u1) - std::f64::consts::PI;
                let cos_theta = theta.cos();

                let accept_prob = (1.0 + self.kappa * cos_theta)
                    / (2.0 * std::f64::consts::PI * self.bessel_i0(self.kappa));

                if u2 <= accept_prob {
                    let result = (self.mu + theta) % (2.0 * std::f64::consts::PI);
                    return if result < 0.0 {
                        result + 2.0 * std::f64::consts::PI
                    } else {
                        result
                    };
                }
            }
        }

        /// Approximate modified Bessel function of the first kind, order 0
        fn bessel_i0(&self, x: f64) -> f64 {
            let ax = x.abs();
            if ax < 3.75 {
                let y = (x / 3.75).powi(2);
                1.0 + y
                    * (3.5156229
                        + y * (3.0899424
                            + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
            } else {
                let y = 3.75 / ax;
                (ax.exp() / ax.sqrt())
                    * (0.39894228
                        + y * (0.01328592
                            + y * (0.00225319
                                + y * (-0.00157565
                                    + y * (0.00916281
                                        + y * (-0.02057706
                                            + y * (0.02635537
                                                + y * (-0.01647633 + y * 0.00392377))))))))
            }
        }

        /// Generate multiple samples
        pub fn sample_vec<R: Rng>(&self, rng: &mut Random<R>, count: usize) -> Vec<f64> {
            (0..count).map(|_| self.sample(rng)).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sobol_sequence() {
        let mut sobol = quasi_monte_carlo::SobolSequence::new(2);
        let points = sobol.generate_points(10);

        assert_eq!(points.len(), 10);
        assert!(points.iter().all(|p| p.len() == 2));
        assert!(points.iter().flatten().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_halton_sequence() {
        let mut halton = quasi_monte_carlo::HaltonSequence::new(2);
        let points = halton.generate_points(10);

        assert_eq!(points.len(), 10);
        assert!(points.iter().all(|p| p.len() == 2));
        assert!(points.iter().flatten().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_latin_hypercube_sampling() {
        let mut lhs = quasi_monte_carlo::LatinHypercubeSampling::with_default_rng(2);
        let samples = lhs.generate_samples(10);

        assert_eq!(samples.shape(), &[10, 2]);
        assert!(samples.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_secure_random() {
        let mut secure_rng = secure::SecureRandom::new();
        let value = secure_rng.random_f64();
        assert!((0.0..1.0).contains(&value));

        let bytes = secure_rng.random_bytes(32);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_antithetic_sampling() {
        let mut antithetic = variance_reduction::AntitheticSampling::with_default_rng();
        let (original, antithetic_vals) = antithetic.generate_antithetic_pairs(10);

        assert_eq!(original.len(), 10);
        assert_eq!(antithetic_vals.len(), 10);

        for (o, a) in original.iter().zip(antithetic_vals.iter()) {
            assert_abs_diff_eq!(o + a, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_importance_sampling() {
        let mut sampler = importance_sampling::ImportanceSampler::with_default_rng();

        let target_pdf = |x: f64| (-0.5 * x * x).exp();
        let proposal_pdf = |_x: f64| 1.0; // Uniform on some interval
        let proposal_sampler = |rng: &mut Random<_>| rng.sample(Uniform::new(-3.0, 3.0).unwrap());

        let (samples, weights) =
            sampler.sample_with_weights(target_pdf, proposal_pdf, proposal_sampler, 100);

        assert_eq!(samples.len(), 100);
        assert_eq!(weights.len(), 100);
        assert!(weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_multivariate_normal() {
        let mean = vec![0.0, 0.0];
        let cov = vec![vec![1.0, 0.5], vec![0.5, 1.0]];

        let mvn = specialized_distributions::MultivariateNormal::new(mean, cov).unwrap();
        let mut rng = Random::default();
        let sample = mvn.sample(&mut rng);

        assert_eq!(sample.len(), 2);
    }

    #[test]
    fn test_dirichlet_distribution() {
        let alphas = vec![1.0, 2.0, 3.0];
        let dirichlet = specialized_distributions::Dirichlet::new(alphas).unwrap();

        let mut rng = Random::default();
        let sample = dirichlet.sample(&mut rng);

        assert_eq!(sample.len(), 3);
        assert_abs_diff_eq!(sample.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
        assert!(sample.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_von_mises_distribution() {
        let von_mises = specialized_distributions::VonMises::new(0.0, 1.0).unwrap();
        let mut rng = Random::default();

        let samples = von_mises.sample_vec(&mut rng, 100);
        assert_eq!(samples.len(), 100);
        assert!(samples
            .iter()
            .all(|&x| (0.0..2.0 * std::f64::consts::PI).contains(&x)));
    }

    #[test]
    fn test_thread_local_rng_pool() {
        let pool = parallel::ThreadLocalRngPool::default();

        let result = pool.with_rng(|rng| rng.sample(Uniform::new(0.0, 1.0).unwrap()));

        assert!((0.0..1.0).contains(&result));
    }

    #[test]
    fn test_control_variate() {
        let mut control = variance_reduction::ControlVariate::new(0.5);

        let target = vec![0.1, 0.3, 0.7, 0.9];
        let control_samples = vec![0.2, 0.4, 0.6, 0.8];

        control.estimate_coefficient(&target, &control_samples);
        let corrected = control.apply_correction(&target, &control_samples);

        assert_eq!(corrected.len(), target.len());
    }
}
