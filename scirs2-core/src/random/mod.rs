//! Ultra-advanced random number generation for SCIRS2 ecosystem
//!
//! This module provides the most comprehensive and cutting-edge random number generation
//! capabilities available, designed specifically for scientific computing, machine learning,
//! and quantum-inspired algorithms with unparalleled features for reproducibility,
//! performance, and specialized ultra-modern distributions.
//!
//! ## Quick Start
//!
//! ```rust
//! // For quick prototyping - use the quick module
//! use scirs2_core::random::quick::*;
//! let x = random_f64();
//! let data = random_vector(100); // Use smaller data for doc tests
//!
//! // For scientific computing - use the prelude
//! use scirs2_core::random::prelude::*;
//! let mut rng = thread_rng();
//! let sample = rng.sample(Normal::new(0.0, 1.0).unwrap());
//! ```
//!
//! ## Module Organization
//!
//! ### 🚀 **Cutting-Edge Modules**
//! - [`cutting_edge_mcmc`] - HMC, NUTS, SVGD, and advanced MCMC methods
//! - [`neural_sampling`] - Normalizing flows, VAE, diffusion models
//! - [`quantum_inspired`] - Quantum algorithms for classical computation
//! - [`advanced_numerical`] - Multi-level Monte Carlo, adaptive sampling
//! - [`ecosystem_integration`] - Seamless SCIRS2 module interoperability
//!
//! ### 🎯 **Workflow-Based Modules**
//! - [`prelude`] - Most commonly used items (Rust idiom)
//! - [`quick`] - Rapid prototyping with minimal setup
//! - [`scientific`] - Research and scientific computing workflows
//! - [`ml`] - Machine learning specific utilities
//!
//! ### ⚡ **Core Implementation Modules**
//! - [`core`] - Core Random struct and fundamental operations
//! - [`distributions`] - Advanced statistical distributions
//! - [`arrays`] - Optimized bulk array generation
//! - [`slice_ops`] - Enhanced slice operations and sampling
//!
//! ### 🔬 **Specialized Modules**
//! - [`qmc`] - Quasi-Monte Carlo sequences (Sobol, Halton, LHS)
//! - [`variance_reduction`] - Monte Carlo variance reduction techniques
//! - [`secure`] - Cryptographically secure random generation
//! - [`parallel`] - Thread-safe parallel random generation

// Core random functionality
pub mod core;
pub mod seq;
pub mod slice_ops;

// Advanced distributions and sampling
pub mod arrays;
pub mod distributions;
// TODO: Fix compilation issues in distributions_unified module
// pub mod distributions_unified;

// Monte Carlo and variance reduction
pub mod qmc;
pub mod variance_reduction;

// Security and parallel computing
pub mod parallel;
pub mod secure;

// Enhanced workflow-based modules
pub mod ml;
pub mod prelude;
pub mod quick;
pub mod scientific;

// Cutting-edge modules
pub mod advanced_numerical;
pub mod cutting_edge_mcmc;
pub mod ecosystem_integration;
pub mod neural_sampling;
pub mod quantum_inspired;

// Re-export core functionality (except Random which we redefine for compatibility)
pub use core::{seeded_rng, thread_rng, DistributionExt};

// Re-export RNG types for SCIRS2 POLICY compliance
pub use rand_chacha::{ChaCha12Rng, ChaCha20Rng, ChaCha8Rng};

// Re-export the core Random as CoreRandom for internal use
pub use core::Random as CoreRandom;

// Re-export enhanced slice operations
pub use slice_ops::{ScientificSliceRandom, SliceRandomExt};

// Note: seq module is available as scirs2_core::random::seq

// Re-export slice convenience functions under different name to avoid conflict
// pub use slice_ops::convenience as slice_convenience;

// Re-export specialized distributions
pub use distributions::{
    Beta, Categorical, Dirichlet, GammaDist, MultivariateNormal, VonMises, WeightedChoice,
};

// TODO: Re-enable when distributions_unified module is fixed
// Re-export unified distribution interface for ecosystem compatibility
// pub use distributions_unified::{
//     UnifiedDistribution,
//     UnifiedNormal, UnifiedBeta, UnifiedCauchy, UnifiedChiSquared,
//     UnifiedFisherF, UnifiedLogNormal, UnifiedStudentT, UnifiedWeibull,
//     UnifiedGamma, UnifiedExp, UnifiedBinomial, UnifiedPoisson, UnifiedDirichlet,
// };

// Re-export optimized array operations
pub use arrays::{
    random_exponential_array, random_gamma_array, random_he_weights, random_normal_array,
    random_sparse_array, random_uniform_array, random_xavier_weights, OptimizedArrayRandom,
};

// Re-export variance reduction techniques
pub use variance_reduction::{
    AntitheticSampling, CommonRatio, ControlVariate, ImportanceSplitting,
};

// Re-export QMC sequences
pub use qmc::{
    HaltonGenerator, LatinHypercubeSampler, LowDiscrepancySequence, QmcError, SobolGenerator,
};

// Re-export secure random generation
pub use secure::{utils as secure_utils, SecureRandom, SecureRngPool};

// Re-export parallel operations
pub use parallel::{BatchRng, DistributedRngPool, ParallelRng, ThreadLocalRngPool};

// Re-export cutting-edge algorithms
pub use advanced_numerical::{
    AdaptiveResult, AdaptiveSampler, ImportanceResult, ImportanceSampler, MLMCResult,
    MultiLevelMonteCarlo, SequentialMonteCarlo,
};

pub use cutting_edge_mcmc::{
    EllipticalSliceSampler, HamiltonianMonteCarlo, NoUTurnSampler, ParallelTempering,
    SteinVariationalGradientDescent,
};

pub use neural_sampling::{
    DiffusionConfig, EnergyBasedModel, NeuralPosteriorEstimation, NormalizingFlow,
    ScoreBasedDiffusion,
};

pub use quantum_inspired::{
    CoinParameters, QuantumAmplitudeAmplification, QuantumInspiredAnnealing,
    QuantumInspiredEvolutionary, QuantumWalk,
};

pub use ecosystem_integration::{
    AugmentationConfig, ExperimentalDesign, LinalgBridge, NeuralBridge, OptimizationBridge,
    StatsBridge, SyntheticDataset,
};

// Re-export external dependencies for convenience
pub use ndarray::Dimension;
pub use rand::prelude as rand_prelude;
pub use rand::rngs;
pub use rand::seq::SliceRandom;
pub use rand::{Rng, RngCore, SeedableRng};
pub use rand_distr as rand_distributions;
pub use rand_distr::uniform;

/// Convenience function to generate a random value of the inferred type
///
/// This function generates a random value using the thread-local RNG.
/// The type is inferred from context, or can be specified explicitly.
///
/// # Examples
///
/// ```
/// use scirs2_core::random::random;
///
/// // Generate random f64
/// let x: f64 = random();
/// assert!(x >= 0.0 && x < 1.0);
///
/// // Generate random bool
/// let b: bool = random();
///
/// // Explicit type annotation
/// let y = random::<f32>();
/// ```
pub fn random<T>() -> T
where
    rand::distr::StandardUniform: rand::distr::Distribution<T>,
{
    rand::random()
}

/// Convenience function to create a thread-local RNG
///
/// This is equivalent to `thread_rng()` but provides a shorter name
/// for compatibility with code that uses `rng()`.
///
/// # Examples
///
/// ```
/// use scirs2_core::random::rng;
/// use scirs2_core::random::Rng;
///
/// let mut rng = rng();
/// let x: f64 = rng.random();
/// ```
pub fn rng() -> rand::rngs::ThreadRng {
    rand::thread_rng()
}

// Comprehensive re-export of ALL rand_distr distributions for SciRS2 ecosystem compatibility
// This ensures other projects can access any distribution through scirs2-core
pub use rand_distr::{
    // Other distributions
    Alphanumeric,
    // Discrete distributions
    Bernoulli as RandBernoulli,
    // Continuous distributions
    Beta as RandBeta,
    Binomial,
    Cauchy,
    ChiSquared,
    // Multivariate distributions
    Dirichlet as RandDirichlet,
    // Distribution trait
    Distribution,
    Exp,
    FisherF,
    Gamma as RandGamma,
    Geometric,
    Hypergeometric,
    InverseGaussian,
    LogNormal,
    Normal as RandNormal,
    Open01,
    OpenClosed01,
    Pareto,
    Pert,
    Poisson,
    StandardNormal,
    StudentT,
    Triangular,
    Uniform as RandUniform,
    UnitBall,
    UnitCircle,
    UnitDisc,
    UnitSphere,
    Weibull,
    Zeta,
    Zipf,
};

// Re-export WeightedIndex from weighted submodule
pub use rand_distr::weighted::WeightedIndex;

// Clean, unprefixed type aliases for common distributions (for easier use)
// These allow `use scirs2_core::random::Normal;` instead of `use scirs2_core::random::RandNormal;`
pub use rand_distr::Bernoulli;
pub use rand_distr::Exp as Exponential; // Exponential is just Exp in rand_distr
pub use rand_distr::Gamma;
pub use rand_distr::Normal;
pub use rand_distr::Uniform;

// Re-export ndarray-rand RandomExt trait if available
#[cfg(feature = "random")]
pub use ndarray_rand::RandomExt;

// Compatibility layer for systems without random feature
#[cfg(not(feature = "random"))]
pub trait RandomExt<T, D> {
    fn random_using<R: rand::Rng>(
        shape: D,
        distribution: impl rand_distr::Distribution<T>,
        rng: &mut R,
    ) -> Self;
}

#[cfg(not(feature = "random"))]
impl<T, D> RandomExt<T, D> for ndarray::ArrayBase<ndarray::OwnedRepr<T>, D>
where
    D: ndarray::Dimension,
{
    fn random_using<R: rand::Rng>(
        shape: D,
        distribution: impl rand_distr::Distribution<T>,
        rng: &mut R,
    ) -> Self {
        let size = shape.size();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(distribution.sample(rng));
        }
        Self::from_shape_vec(shape, data).unwrap()
    }
}

/// Legacy compatibility functions for backward compatibility
pub mod legacy {
    use super::*;
    use rand_distr::Uniform;

    /// Compatibility wrapper for updated rand API
    pub fn rng() -> Random<rand::rngs::ThreadRng> {
        Random { rng: rand::rng() }
    }

    /// Generate a random f64 value between 0.0 and 1.0
    pub fn f64() -> f64 {
        rand::random::<f64>()
    }

    /// Generate a random f32 value between 0.0 and 1.0
    pub fn f32() -> f32 {
        rand::random::<f32>()
    }

    /// Generate a random usize value in the given range
    pub fn usize(range: std::ops::Range<usize>) -> usize {
        rand::thread_rng().gen_range(range)
    }
}

/// High-level convenience functions for common operations
pub mod convenience {
    use super::*;
    use ndarray::{Array, Dimension, IxDyn};
    use rand_distr::{Distribution, Normal, Uniform};

    /// Generate a uniform random number in [0, 1)
    pub fn uniform() -> f64 {
        let mut rng = thread_rng();
        rng.sample(Uniform::new(0.0, 1.0).unwrap())
    }

    /// Generate a standard normal random number (mean=0, std=1)
    pub fn normal() -> f64 {
        let mut rng = thread_rng();
        rng.sample(Normal::new(0.0, 1.0).unwrap())
    }

    /// Generate a random integer in the given range
    pub fn integer(min: i64, max: i64) -> i64 {
        let mut rng = thread_rng();
        rng.sample(Uniform::new_inclusive(min, max).unwrap())
    }

    /// Generate a random boolean
    pub fn boolean() -> bool {
        let mut rng = thread_rng();
        rng.gen_bool(0.5)
    }

    /// Generate a random array with uniform distribution
    pub fn uniform_array<Sh: Into<IxDyn>>(shape: Sh) -> Array<f64, IxDyn> {
        let mut rng = thread_rng();
        let shape = shape.into();
        let size = shape.size();
        let values: Vec<f64> = (0..size)
            .map(|_| rng.sample(Uniform::new(0.0, 1.0).unwrap()))
            .collect();
        Array::from_shape_vec(shape, values).unwrap()
    }

    /// Generate a random array with normal distribution
    pub fn normal_array<Sh: Into<IxDyn>>(shape: Sh, mean: f64, std: f64) -> Array<f64, IxDyn> {
        let mut rng = thread_rng();
        let shape = shape.into();
        let size = shape.size();
        let values: Vec<f64> = (0..size)
            .map(|_| rng.sample(Normal::new(mean, std).unwrap()))
            .collect();
        Array::from_shape_vec(shape, values).unwrap()
    }
}

/// Sampling utilities for common statistical operations
pub mod sampling {
    use super::*;
    use ndarray::{Array, Dimension, IxDyn};
    use rand_distr::{Distribution, Exp, LogNormal, Normal, Uniform};

    /// Sample uniformly from [0, 1)
    pub fn random_uniform01<R: rand::Rng>(rng: &mut Random<R>) -> f64 {
        rng.sample(Uniform::new(0.0, 1.0).unwrap())
    }

    /// Sample from a standard normal distribution (mean 0, std dev 1)
    pub fn random_standard_normal<R: rand::Rng>(rng: &mut Random<R>) -> f64 {
        rng.sample(Normal::new(0.0, 1.0).unwrap())
    }

    /// Sample from a normal distribution with given mean and standard deviation
    pub fn random_normal<R: rand::Rng>(rng: &mut Random<R>, mean: f64, stddev: f64) -> f64 {
        rng.sample(Normal::new(mean, stddev).unwrap())
    }

    /// Sample from a log-normal distribution
    pub fn random_lognormal<R: rand::Rng>(rng: &mut Random<R>, mean: f64, stddev: f64) -> f64 {
        rng.sample(LogNormal::new(mean, stddev).unwrap())
    }

    /// Sample from an exponential distribution
    pub fn random_exponential<R: rand::Rng>(rng: &mut Random<R>, lambda: f64) -> f64 {
        rng.sample(Exp::new(lambda).unwrap())
    }

    /// Generate an array of random integers in a range
    pub fn random_integers<R: rand::Rng, Sh>(
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
    pub fn random_floats<R: rand::Rng, Sh>(
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
    pub fn bootstrap_indices<R: rand::Rng>(
        rng: &mut Random<R>,
        data_size: usize,
        sample_size: usize,
    ) -> Vec<usize> {
        let dist = Uniform::new(0, data_size).unwrap();
        rng.sample_vec(dist, sample_size)
    }

    /// Sample indices without replacement (for random subsampling)
    pub fn sample_without_replacement<R: rand::Rng>(
        rng: &mut Random<R>,
        data_size: usize,
        sample_size: usize,
    ) -> Vec<usize> {
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..data_size).collect();
        indices.shuffle(&mut rng.rng);
        indices.truncate(sample_size);
        indices
    }
}

/// Importance sampling methods for efficient estimation
pub mod importance_sampling {
    use super::*;
    use rand_distr::{Normal, Uniform};

    /// Importance sampling estimator
    #[derive(Debug)]
    pub struct ImportanceSampler<R: rand::Rng> {
        rng: Random<R>,
    }

    impl<R: rand::Rng> ImportanceSampler<R> {
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

/// GPU-accelerated random number generation (when available)
#[cfg(feature = "gpu")]
pub mod gpu {
    // GPU acceleration implementation would go here
    // This is a placeholder for future GPU support
    pub struct GpuRng;

    impl Default for GpuRng {
        fn default() -> Self {
            Self::new()
        }
    }

    impl GpuRng {
        pub fn new() -> Self {
            Self
        }
    }
}

/// Legacy Random struct wrapper for backward compatibility
/// This provides the same interface as the original Random struct
/// while delegating to the new modular implementation
#[derive(Debug)]
pub struct Random<R: rand::Rng + ?Sized = rand::rngs::ThreadRng> {
    pub(crate) rng: R,
}

impl Default for Random<rand::rngs::ThreadRng> {
    fn default() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }
}

impl<R: rand::Rng + Clone> Clone for Random<R> {
    fn clone(&self) -> Self {
        Self {
            rng: self.rng.clone(),
        }
    }
}

impl<R: rand::Rng> Random<R> {
    /// Sample a value from a distribution
    pub fn sample<D, T>(&mut self, distribution: D) -> T
    where
        D: rand_distr::Distribution<T>,
    {
        use rand_distr::Distribution;
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

    /// Generate a random value within the given range (using range syntax)
    #[allow(deprecated)]
    pub fn gen_range<T, RNG>(&mut self, range: RNG) -> T
    where
        T: rand_distr::uniform::SampleUniform,
        RNG: rand_distr::uniform::SampleRange<T>,
    {
        self.rng.gen_range(range)
    }

    /// Generate a random f64 value between 0.0 and 1.0
    pub fn random_f64(&mut self) -> f64 {
        self.sample(rand_distr::Uniform::new(0.0, 1.0).unwrap())
    }

    /// Generate a random f64 value using the underlying RNG (convenience method)
    pub fn random_f64_raw(&mut self) -> f64 {
        self.rng.random()
    }

    /// Generate a random boolean value
    pub fn random_bool(&mut self) -> bool {
        use rand_distr::Distribution;
        let dist = rand_distr::Bernoulli::new(0.5).unwrap();
        dist.sample(&mut self.rng)
    }

    /// Generate a random boolean with the given probability of being true
    pub fn random_bool_with_chance(&mut self, prob: f64) -> bool {
        use rand_distr::Distribution;
        let dist = rand_distr::Bernoulli::new(prob).unwrap();
        dist.sample(&mut self.rng)
    }

    /// Shuffle a slice randomly
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        use rand::seq::SliceRandom;
        slice.shuffle(&mut self.rng);
    }

    /// Generate a vector of values sampled from a distribution
    pub fn sample_vec<D, T>(&mut self, distribution: D, size: usize) -> Vec<T>
    where
        D: rand_distr::Distribution<T> + Copy,
    {
        (0..size)
            .map(|_| distribution.sample(&mut self.rng))
            .collect()
    }

    /// Generate an ndarray::Array from samples of a distribution
    pub fn sample_array<D, T, Sh>(
        &mut self,
        distribution: D,
        shape: Sh,
    ) -> ndarray::Array<T, ndarray::IxDyn>
    where
        D: rand_distr::Distribution<T> + Copy,
        Sh: Into<ndarray::IxDyn>,
    {
        let shape = shape.into();
        let size = shape.size();
        let values = self.sample_vec(distribution, size);
        ndarray::Array::from_shape_vec(shape, values).unwrap()
    }
}

impl Random<rand::rngs::ThreadRng> {
    /// Create a new random number generator with a specific seed
    pub fn seed(seed: u64) -> Random<rand::rngs::StdRng> {
        Random {
            rng: rand::SeedableRng::seed_from_u64(seed),
        }
    }
}

// Implement required traits for the legacy Random struct
impl<R: rand::RngCore> rand::RngCore for Random<R> {
    fn next_u32(&mut self) -> u32 {
        self.rng.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.rng.fill_bytes(dest)
    }
}

impl rand::SeedableRng for Random<rand::rngs::StdRng> {
    type Seed = <rand::rngs::StdRng as rand::SeedableRng>::Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        Random {
            rng: rand::rngs::StdRng::from_seed(seed),
        }
    }

    fn seed_from_u64(state: u64) -> Self {
        Random {
            rng: rand::rngs::StdRng::seed_from_u64(state),
        }
    }
}

/// Thread-local random number generator for convenient access (legacy compatibility)
use std::cell::RefCell;
thread_local! {
    static THREAD_RNG: RefCell<Random> = RefCell::new(Random::default());
}

/// Get a reference to the thread-local random number generator (legacy compatibility)
#[allow(dead_code)]
pub fn get_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut Random) -> R,
{
    THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
}

/// Deterministic random sequence generator for testing (legacy compatibility)
pub struct DeterministicSequence {
    seed: u64,
    counter: u64,
}

impl DeterministicSequence {
    /// Create a new deterministic sequence with the given seed
    pub fn seed(seed: u64) -> Self {
        Self { seed, counter: 0 }
    }

    /// Generate the next value in the sequence
    pub fn next_f64(&mut self) -> f64 {
        // Simple deterministic hash function for testing purposes
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
    pub fn get_array<Sh>(&mut self, shape: Sh) -> ndarray::Array<f64, ndarray::IxDyn>
    where
        Sh: Into<ndarray::IxDyn>,
    {
        let shape = shape.into();
        let size = shape.size();
        let values = self.get_vec(size);
        ndarray::Array::from_shape_vec(shape, values).unwrap()
    }
}

// ===============================
// Enhanced Type Aliases & Exports
// ===============================

/// Convenient type aliases for common RNG types
pub type ThreadRng = Random<rand::rngs::ThreadRng>;
pub type StdRng = Random<rand::rngs::StdRng>;

/// Common distribution type aliases
pub type UniformDist = rand_distributions::Uniform<f64>;
pub type NormalDist = rand_distributions::Normal<f64>;
pub type ExponentialDist = rand_distributions::Exp<f64>;

/// Array type aliases for convenience
pub type Array1D<T> = ndarray::Array1<T>;
pub type Array2D<T> = ndarray::Array2<T>;
pub type Array3D<T> = ndarray::Array3<T>;

// ===============================
// Workflow Module Aliases
// ===============================

/// Alias for quick access to rapid prototyping functions
pub use quick as rapid;

/// Alias for scientific computing workflows
pub use scientific as research;

/// Alias for machine learning workflows
pub use ml as machine_learning;

/// Alias for cryptographic random generation
pub use secure as crypto;

// ===============================
// Legacy Compatibility Modules
// ===============================

/// Legacy module structure for backward compatibility
pub mod quasi_monte_carlo {
    pub use crate::random::qmc::*;

    // Legacy type aliases for backward compatibility
    pub type SobolSequence = crate::random::qmc::SobolGenerator;
    pub type HaltonSequence = crate::random::qmc::HaltonGenerator;
    pub type LatinHypercubeSampling = crate::random::qmc::LatinHypercubeSampler;
}

/// Legacy module structure for backward compatibility
pub mod specialized_distributions {
    pub use crate::random::distributions::*;
}

/// Legacy module structure for backward compatibility
pub mod optimized_arrays {
    pub use crate::random::arrays::*;
}

/// Legacy slice operations
pub mod slice_random {
    pub use crate::random::slice_ops::convenience::*;
}

// ===============================
// Enhanced Feature-Based Exports
// ===============================

/// All essential items for most use cases
pub mod essentials {
    pub use crate::random::rand_distributions::{Normal, Uniform};
    pub use crate::random::{
        random_normal_array, random_uniform_array, seeded_rng, thread_rng, Beta, Categorical,
        Random, Rng, RngCore, SeedableRng, WeightedChoice,
    };
}

/// Advanced statistical functionality
pub mod statistics {
    pub use crate::random::{
        AntitheticSampling, Beta, Categorical, CommonRatio, ControlVariate, Dirichlet,
        ExponentialDist, GammaDist, HaltonGenerator, LatinHypercubeSampler, MultivariateNormal,
        SobolGenerator, VonMises, WeightedChoice,
    };
}

/// High-performance computing functionality
pub mod hpc {
    pub use crate::random::{
        random_he_weights, random_normal_array, random_uniform_array, random_xavier_weights,
        BatchRng, DistributedRngPool, OptimizedArrayRandom, ParallelRng, ThreadLocalRngPool,
    };
}

/// 🚀 **Cutting-edge algorithms**
pub mod cutting_edge {
    pub use crate::random::{
        advanced_numerical::*, cutting_edge_mcmc::*, ecosystem_integration::*, neural_sampling::*,
        quantum_inspired::*,
    };
}

/// Advanced MCMC and Bayesian inference
pub mod bayesian {
    pub use crate::random::{
        EllipticalSliceSampler, HamiltonianMonteCarlo, ImportanceSampler, NoUTurnSampler,
        ParallelTempering, SteinVariationalGradientDescent,
    };
    // AdaptiveMetropolisHastings is available through the cutting_edge module
}

/// Neural and AI-based sampling methods
pub mod ai_sampling {
    pub use crate::random::{
        DiffusionConfig, EnergyBasedModel, NeuralBridge, NeuralPosteriorEstimation,
        NormalizingFlow, ScoreBasedDiffusion,
    };
}

/// Quantum-inspired computational methods
pub mod quantum {
    pub use crate::random::{
        CoinParameters, QuantumAmplitudeAmplification, QuantumInspiredAnnealing,
        QuantumInspiredEvolutionary, QuantumWalk,
    };
}

/// Advanced numerical methods and optimization
pub mod numerical_methods {
    pub use crate::random::{
        AdaptiveResult, AdaptiveSampler, ImportanceResult, MLMCResult, MultiLevelMonteCarlo,
        SequentialMonteCarlo,
    };
}

/// Ecosystem integration and bridge utilities
pub mod bridges {
    pub use crate::random::{
        AugmentationConfig, ExperimentalDesign, LinalgBridge, NeuralBridge, OptimizationBridge,
        StatsBridge, SyntheticDataset,
    };
}
