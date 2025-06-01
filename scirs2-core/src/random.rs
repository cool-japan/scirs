//! # Random Number Generation
//!
//! This module provides a consistent interface for random number generation in scientific computing.
//!
//! ## Features
//!
//! * Consistent interface across distribution types
//! * Thread-local and seedable random number generators
//! * Distribution-independent sampling functions
//! * Deterministic sequence generation for testing
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

/// Wrapper around the rand crate's RNG for a consistent interface
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
