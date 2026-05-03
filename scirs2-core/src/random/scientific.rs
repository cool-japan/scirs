//! Scientific computing workflows for reproducible research
//!
//! This module provides high-level interfaces specifically designed for scientific computing
//! and research applications where reproducibility, statistical validity, and advanced
//! sampling methods are critical.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_core::random::scientific::*;
//!
//! // Reproducible experiments
//! let mut experiment = ReproducibleExperiment::new(42);
//! let sample1 = experiment.next_sample(1000, StandardNormal);
//! let sample2 = experiment.next_sample(1000, StandardNormal);
//!
//! // Statistical sampling
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let bootstrap = bootstrap_sample(&data, 1000, 500);
//! let jackknife = jackknife_samples(&data);
//!
//! // Monte Carlo with variance reduction
//! let mut mc = MonteCarloSampler::with_antithetic_variates(12345);
//! let samples = mc.generate_correlated_pairs(1000);
//! ```

use crate::random::{
    arrays::{random_normal_array, random_uniform_array, OptimizedArrayRandom},
    core::{
        scientific::{DeterministicState, ReproducibleSequence},
        seeded_rng, Random,
    },
    qmc::{HaltonGenerator, LatinHypercubeSampler, LowDiscrepancySequence, SobolGenerator},
    variance_reduction::{AntitheticSampling, ControlVariate},
    ParallelRng, ThreadLocalRngPool,
};
use ::ndarray::{Array, Array1, Array2, Ix2};
use rand::{Rng, RngExt};
use rand_distr::{Distribution, Normal, Uniform};
use std::collections::HashMap;

/// Standard normal distribution (mean=0, std=1) for convenience
#[derive(Debug, Clone, Copy)]
pub struct StandardNormal;

impl Distribution<f64> for StandardNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        Normal::new(0.0, 1.0).expect("Operation failed").sample(rng)
    }
}

/// Standard uniform distribution [0, 1) for convenience
#[derive(Debug, Clone, Copy)]
pub struct StandardUniform;

impl Distribution<f64> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        Uniform::new(0.0, 1.0)
            .expect("Operation failed")
            .sample(rng)
    }
}

/// Reproducible experiment manager for scientific research
pub struct ReproducibleExperiment {
    sequence: ReproducibleSequence,
    state: DeterministicState,
}

impl ReproducibleExperiment {
    /// Create a new reproducible experiment with a seed
    pub fn new(seed: u64) -> Self {
        Self {
            sequence: ReproducibleSequence::new(seed),
            state: DeterministicState::new(seed),
        }
    }

    /// Generate the next sample in the reproducible sequence
    pub fn next_sample<D: Distribution<f64> + Copy>(
        &mut self,
        size: usize,
        distribution: D,
    ) -> Vec<f64> {
        let mut rng = self.state.next_rng();
        rng.sample_vec(distribution, size)
    }

    /// Generate the next array sample
    pub fn next_array_sample<D: Distribution<f64> + Copy>(
        &mut self,
        shape: [usize; 2],
        distribution: D,
    ) -> Array2<f64> {
        let mut rng = self.state.next_rng();
        Array::random_bulk(Ix2(shape[0], shape[1]), distribution, &mut rng)
    }

    /// Reset the experiment to its initial state
    pub fn reset(&mut self) {
        self.sequence.reset();
        self.state = DeterministicState::new(self.state.current_state().0);
    }

    /// Get current experiment state for logging/checkpointing
    pub fn current_state(&self) -> (u64, u64) {
        self.state.current_state()
    }
}

/// Bootstrap sampling for statistical inference
pub fn bootstrap_sample<T: Clone + Send + Sync>(
    data: &[T],
    n_bootstrap: usize,
    sample_size: usize,
) -> Vec<Vec<T>> {
    let pool = ThreadLocalRngPool::new(42);
    ParallelRng::parallel_bootstrap(data, n_bootstrap, &pool)
        .into_iter()
        .map(|sample| sample.into_iter().take(sample_size).collect())
        .collect()
}

/// Jackknife resampling (leave-one-out)
pub fn jackknife_samples<T: Clone>(data: &[T]) -> Vec<Vec<T>> {
    (0..data.len())
        .map(|i| {
            data.iter()
                .enumerate()
                .filter(|(idx, _)| *idx != i)
                .map(|(_, item)| item.clone())
                .collect()
        })
        .collect()
}

/// Cross-validation splits
pub fn cross_validation_splits<T: Clone>(
    data: &[T],
    k_folds: usize,
    seed: u64,
) -> Vec<(Vec<T>, Vec<T>)> {
    let mut rng = seeded_rng(seed);
    let mut indices: Vec<usize> = (0..data.len()).collect();

    // Shuffle indices for random splits
    use rand::seq::SliceRandom;
    indices.shuffle(&mut rng.rng);

    let fold_size = data.len() / k_folds;

    (0..k_folds)
        .map(|fold| {
            let test_start = fold * fold_size;
            let test_end = if fold == k_folds - 1 {
                data.len()
            } else {
                test_start + fold_size
            };

            let test_indices = &indices[test_start..test_end];
            let train_indices: Vec<usize> = indices
                .iter()
                .filter(|&&idx| !test_indices.contains(&idx))
                .copied()
                .collect();

            let train_data = train_indices.iter().map(|&i| data[i].clone()).collect();
            let test_data = test_indices.iter().map(|&i| data[i].clone()).collect();

            (train_data, test_data)
        })
        .collect()
}

/// Monte Carlo sampler with variance reduction techniques
pub struct MonteCarloSampler {
    antithetic: Option<AntitheticSampling<rand::rngs::StdRng>>,
    control_variate: Option<ControlVariate>,
    base_seed: u64,
}

impl MonteCarloSampler {
    /// Create a basic Monte Carlo sampler
    pub fn new(seed: u64) -> Self {
        Self {
            antithetic: None,
            control_variate: None,
            base_seed: seed,
        }
    }

    /// Create a sampler with antithetic variates for variance reduction
    pub fn with_antithetic_variates(seed: u64) -> Self {
        Self {
            antithetic: Some(AntitheticSampling::new(seeded_rng(seed))),
            control_variate: None,
            base_seed: seed,
        }
    }

    /// Create a sampler with control variates
    pub fn with_control_variate(seed: u64, control_mean: f64) -> Self {
        Self {
            antithetic: None,
            control_variate: Some(ControlVariate::new(control_mean)),
            base_seed: seed,
        }
    }

    /// Generate correlated sample pairs for variance reduction
    pub fn generate_correlated_pairs(&mut self, count: usize) -> (Vec<f64>, Vec<f64>) {
        if let Some(ref mut antithetic) = self.antithetic {
            antithetic.generate_antithetic_pairs(count)
        } else {
            let mut rng = seeded_rng(self.base_seed);
            let samples1: Vec<f64> = (0..count)
                .map(|_| rng.sample(Uniform::new(0.0, 1.0).expect("Operation failed")))
                .collect();
            let samples2: Vec<f64> = (0..count)
                .map(|_| rng.sample(Uniform::new(0.0, 1.0).expect("Operation failed")))
                .collect();
            (samples1, samples2)
        }
    }

    /// Generate stratified samples for more uniform coverage
    pub fn stratified_samples(&mut self, strata: usize, samples_per_stratum: usize) -> Vec<f64> {
        if let Some(ref mut antithetic) = self.antithetic {
            antithetic.stratified_samples(strata, samples_per_stratum)
        } else {
            let mut rng = seeded_rng(self.base_seed);
            let mut all_samples = Vec::new();

            for i in 0..strata {
                let stratum_start = i as f64 / strata as f64;
                let stratum_end = (i + 1) as f64 / strata as f64;

                for _ in 0..samples_per_stratum {
                    let u = rng.sample(Uniform::new(0.0, 1.0).expect("Operation failed"));
                    let sample = stratum_start + u * (stratum_end - stratum_start);
                    all_samples.push(sample);
                }
            }

            all_samples
        }
    }
}

/// Quasi-Monte Carlo sampling for better coverage
pub struct QuasiMonteCarloSampler {
    sobol: Option<SobolGenerator>,
    halton: Option<HaltonGenerator>,
    lhs: Option<LatinHypercubeSampler<rand::rngs::ThreadRng>>,
    dimensions: usize,
}

impl QuasiMonteCarloSampler {
    /// Create a Sobol sequence sampler
    pub fn sobol(dimensions: usize) -> Result<Self, String> {
        Ok(Self {
            sobol: Some(SobolGenerator::dimension(dimensions).map_err(|e| e.to_string())?),
            halton: None,
            lhs: None,
            dimensions,
        })
    }

    /// Create a Halton sequence sampler
    pub fn halton(dimensions: usize) -> Self {
        Self {
            sobol: None,
            halton: Some(HaltonGenerator::new(
                &(2u32..2u32 + dimensions as u32).collect::<Vec<_>>(),
            )),
            lhs: None,
            dimensions,
        }
    }

    /// Create a Latin Hypercube sampler
    pub fn latin_hypercube(dimensions: usize) -> Self {
        Self {
            sobol: None,
            halton: None,
            lhs: Some(LatinHypercubeSampler::<rand::rngs::ThreadRng>::new(
                dimensions,
            )),
            dimensions,
        }
    }

    /// Generate quasi-random points
    pub fn generate_points(&mut self, count: usize) -> Result<Vec<Vec<f64>>, String> {
        if let Some(ref mut sobol) = self.sobol {
            let array = sobol.generate(count);
            let result: Vec<Vec<f64>> = array.rows().into_iter().map(|row| row.to_vec()).collect();
            Ok(result)
        } else if let Some(ref mut halton) = self.halton {
            let array = halton.generate(count);
            let result: Vec<Vec<f64>> = array.rows().into_iter().map(|row| row.to_vec()).collect();
            Ok(result)
        } else if let Some(ref mut lhs) = self.lhs {
            let array = lhs.sample(count).map_err(|e| e.to_string())?;
            Ok((0..count)
                .map(|i| (0..self.dimensions).map(|j| array[[i, j]]).collect())
                .collect())
        } else {
            Err("No QMC generator configured".to_string())
        }
    }
}

/// Design of experiments helper
pub struct ExperimentalDesign;

impl ExperimentalDesign {
    /// Generate a factorial design
    pub fn factorial_design(factors: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if factors.is_empty() {
            return vec![vec![]];
        }

        let mut designs = vec![vec![]];

        for factor_levels in factors {
            let mut new_designs = Vec::new();
            for design in &designs {
                for &level in factor_levels {
                    let mut new_design = design.clone();
                    new_design.push(level);
                    new_designs.push(new_design);
                }
            }
            designs = new_designs;
        }

        designs
    }

    /// Generate a random fractional factorial design
    pub fn fractional_factorial_design(
        factors: &[Vec<f64>],
        fraction: f64,
        seed: u64,
    ) -> Vec<Vec<f64>> {
        let full_design = Self::factorial_design(factors);
        let sample_size = (full_design.len() as f64 * fraction).ceil() as usize;

        let mut rng = seeded_rng(seed);
        use rand::seq::SliceRandom;
        let mut sampled_design = full_design;
        sampled_design.shuffle(&mut rng.rng);
        sampled_design.truncate(sample_size);

        sampled_design
    }

    /// Generate a central composite design
    pub fn central_composite_design(dimensions: usize, alpha: f64) -> Vec<Vec<f64>> {
        let mut design = Vec::new();

        // Factorial points (corners of hypercube)
        let factorial_factors: Vec<Vec<f64>> = (0..dimensions).map(|_| vec![-1.0, 1.0]).collect();
        design.extend(Self::factorial_design(&factorial_factors));

        // Axial points
        for dim in 0..dimensions {
            let mut point_pos = vec![0.0; dimensions];
            let mut point_neg = vec![0.0; dimensions];
            point_pos[dim] = alpha;
            point_neg[dim] = -alpha;
            design.push(point_pos);
            design.push(point_neg);
        }

        // Center point
        design.push(vec![0.0; dimensions]);

        design
    }

    /// Generate a Latin Hypercube Sampling (LHS) design.
    ///
    /// LHS is a stratified sampling method that produces samples that are
    /// well-distributed across each dimension's range. For each of the `D`
    /// dimensions, the unit interval `[0, 1)` is split into `N` equal strata
    /// and exactly one sample is drawn from each stratum; the per-dimension
    /// samples are then independently shuffled to randomize cross-dimensional
    /// pairings. Each `[0, 1)` sample is finally mapped to a discrete factor
    /// level by index `(u * K) as usize` clamped to `K - 1`, where `K` is the
    /// number of levels of that factor.
    ///
    /// # Arguments
    ///
    /// * `factors` - Per-factor level vectors (each `factors[d]` is the set of
    ///   discrete levels for dimension `d`).
    /// * `n_points` - Number of LHS points. When `None`, defaults to the
    ///   maximum number of levels across all factors so that every level of
    ///   every factor is visited at least once.
    /// * `seed` - Deterministic RNG seed.
    ///
    /// # Returns
    ///
    /// A vector of `n_points` design points; each point is a `Vec<f64>` of
    /// length `factors.len()` whose entries are drawn from the respective
    /// factor's level set. Returns a single empty point when `factors` is
    /// empty (matching the behavior of `factorial_design`), and an empty
    /// vector when the resolved `n` is zero.
    pub fn latin_hypercube_design(
        factors: &[Vec<f64>],
        n_points: Option<usize>,
        seed: u64,
    ) -> Vec<Vec<f64>> {
        // Match factorial_design's edge case for empty factors.
        if factors.is_empty() {
            return vec![vec![]];
        }

        // Refuse zero-level factors — there is nothing meaningful to sample
        // along that dimension. Returning an empty design preserves the
        // function's `Vec<Vec<f64>>` signature without panicking.
        if factors.iter().any(|f| f.is_empty()) {
            return Vec::new();
        }

        // Default N: max factor level count, so each level of each factor
        // appears at least once across the strata.
        let n =
            n_points.unwrap_or_else(|| factors.iter().map(|f| f.len()).max().unwrap_or(1).max(1));

        if n == 0 {
            return Vec::new();
        }

        let dimensions = factors.len();

        // Reuse the QMC LHS sampler (Fisher–Yates shuffle + per-stratum
        // uniform jitter, see `crate::random::qmc::LatinHypercubeSampler`).
        let mut sampler =
            LatinHypercubeSampler::<rand::prelude::ThreadRng>::with_seed(dimensions, seed);

        // The QMC sampler can fail only when n == 0 (already handled). Use
        // `unwrap_or_else` rather than `unwrap()` to honor the no-unwrap
        // policy.
        let unit_points = sampler
            .sample(n)
            .unwrap_or_else(|_| ::ndarray::Array2::<f64>::zeros((n, dimensions)));

        // Map each [0, 1) coordinate to the corresponding factor level.
        let mut design_points = Vec::with_capacity(n);
        for i in 0..n {
            let mut point = Vec::with_capacity(dimensions);
            for (d, factor) in factors.iter().enumerate() {
                let k = factor.len();
                // The QMC sampler guarantees u in [0, 1); clamp the resulting
                // index to `k - 1` to defensively handle any edge case where
                // `u * k as f64` rounds to `k`.
                let mut idx = (unit_points[[i, d]] * k as f64) as usize;
                if idx >= k {
                    idx = k - 1;
                }
                point.push(factor[idx]);
            }
            design_points.push(point);
        }

        design_points
    }
}

/// A/B testing utilities
pub mod ab_testing {
    use super::*;

    /// Split data into A/B test groups
    pub fn split_ab_groups<T: Clone>(data: &[T], split_ratio: f64, seed: u64) -> (Vec<T>, Vec<T>) {
        let mut rng = seeded_rng(seed);

        let mut group_a = Vec::new();
        let mut group_b = Vec::new();

        for item in data {
            if rng.sample(Uniform::new(0.0, 1.0).expect("Operation failed")) < split_ratio {
                group_a.push(item.clone());
            } else {
                group_b.push(item.clone());
            }
        }

        (group_a, group_b)
    }

    /// Generate balanced A/B test assignments
    pub fn balanced_assignment(user_ids: &[String], seed: u64) -> HashMap<String, String> {
        let mut rng = seeded_rng(seed);
        let mut assignments = HashMap::new();

        let mut shuffled_ids = user_ids.to_vec();
        use rand::seq::SliceRandom;
        shuffled_ids.shuffle(&mut rng.rng);

        for (i, user_id) in shuffled_ids.iter().enumerate() {
            let group = if i % 2 == 0 { "A" } else { "B" };
            assignments.insert(user_id.clone(), group.to_string());
        }

        assignments
    }

    /// Multi-arm bandit assignment with Thompson sampling
    pub fn thompson_sampling_assignment(
        arms: &[String],
        successes: &[u32],
        failures: &[u32],
        seed: u64,
    ) -> String {
        use crate::random::distributions::Beta;

        let mut rng = seeded_rng(seed);
        let mut max_sample = 0.0;
        let mut best_arm = arms[0].clone();

        for (i, arm) in arms.iter().enumerate() {
            let alpha = successes[i] as f64 + 1.0;
            let beta_param = failures[i] as f64 + 1.0;

            if let Ok(beta_dist) = Beta::new(alpha, beta_param) {
                let sample = beta_dist.sample(&mut rng);
                if sample > max_sample {
                    max_sample = sample;
                    best_arm = arm.clone();
                }
            }
        }

        best_arm
    }
}

#[cfg(test)]
mod lhs_tests {
    use super::*;

    fn make_factors() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![10.0, 20.0],
            vec![-5.0, 0.0, 5.0],
        ]
    }

    #[test]
    fn test_latin_hypercube_default_n_uses_max_levels() {
        let factors = make_factors();
        let design = ExperimentalDesign::latin_hypercube_design(&factors, None, 7);

        // Default N = max(K) = 4 (first factor has 4 levels).
        assert_eq!(design.len(), 4);
        for point in &design {
            assert_eq!(point.len(), factors.len());
            // Every coordinate must come from the corresponding factor's levels.
            for (d, value) in point.iter().enumerate() {
                assert!(
                    factors[d].iter().any(|level| (level - value).abs() < 1e-12),
                    "value {} for dim {} not in factor levels",
                    value,
                    d,
                );
            }
        }
    }

    #[test]
    fn test_latin_hypercube_explicit_n_points() {
        let factors = make_factors();
        let n = 12;
        let design = ExperimentalDesign::latin_hypercube_design(&factors, Some(n), 11);
        assert_eq!(design.len(), n);
        for point in &design {
            assert_eq!(point.len(), factors.len());
        }
    }

    #[test]
    fn test_latin_hypercube_determinism_same_seed() {
        let factors = make_factors();
        let a = ExperimentalDesign::latin_hypercube_design(&factors, Some(8), 1234);
        let b = ExperimentalDesign::latin_hypercube_design(&factors, Some(8), 1234);
        assert_eq!(a, b);
    }

    #[test]
    fn test_latin_hypercube_different_seeds_diverge() {
        let factors = make_factors();
        let a = ExperimentalDesign::latin_hypercube_design(&factors, Some(16), 42);
        let b = ExperimentalDesign::latin_hypercube_design(&factors, Some(16), 43);
        assert_ne!(a, b);
    }

    #[test]
    fn test_latin_hypercube_stratification() {
        // For N == K_d on every factor, every level must appear at least once
        // along that factor — this is the LHS guarantee (one sample per stratum).
        // Use uniform K so the default N = K satisfies the precondition.
        let factors: Vec<Vec<f64>> = (0..3).map(|_| vec![0.0, 1.0, 2.0, 3.0, 4.0]).collect();
        let design = ExperimentalDesign::latin_hypercube_design(&factors, None, 99);
        assert_eq!(design.len(), 5);

        for d in 0..3 {
            let mut seen = vec![false; 5];
            for point in &design {
                let level_idx = factors[d]
                    .iter()
                    .position(|level| (level - point[d]).abs() < 1e-12)
                    .expect("each LHS coordinate must map back to a factor level");
                seen[level_idx] = true;
            }
            assert!(
                seen.iter().all(|s| *s),
                "stratification failed on dim {}: {:?}",
                d,
                seen,
            );
        }
    }

    #[test]
    fn test_latin_hypercube_empty_factors() {
        let design = ExperimentalDesign::latin_hypercube_design(&[], None, 1);
        assert_eq!(design, vec![Vec::<f64>::new()]);
    }

    #[test]
    fn test_latin_hypercube_zero_level_factor_returns_empty() {
        let factors: Vec<Vec<f64>> = vec![vec![0.0, 1.0], Vec::new()];
        let design = ExperimentalDesign::latin_hypercube_design(&factors, Some(4), 1);
        assert!(design.is_empty());
    }
}
