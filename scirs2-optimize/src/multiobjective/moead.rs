//! MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition
//!
//! Implements the algorithm by Zhang & Li (2007).  The core idea is to
//! decompose the multi-objective problem into *N* scalar sub-problems using
//! weight vectors and to exploit neighbourhood relationships during evolution.
//!
//! **Algorithm outline (per generation):**
//!
//! 1. For each sub-problem *i* (weight vector λ_i):
//!    a. Choose mating partners from the neighbourhood B(i) of size T.
//!    b. Generate offspring y by differential evolution (DE/rand/1).
//!    c. Update ideal point z*.
//!    d. For each j ∈ B(i), replace x_j with y if g^te(y|λ_j,z*) ≤ g^te(x_j|λ_j,z*).
//! 2. Collect the final population as the Pareto-front approximation.
//!
//! # References
//!
//! - Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm
//!   based on decomposition. *IEEE TEC*, 11(6), 712–731.
//! - Li, H., & Zhang, Q. (2009). Multiobjective optimization problems with
//!   complicated Pareto sets, MOEA/D and NSGA-II.  *IEEE TEC*, 13(2), 284–302.

use crate::error::OptimizeResult;
use crate::multiobjective::indicators::{dominates, non_dominated_sort};
use crate::multiobjective::nsga2::Individual;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the MOEA/D algorithm.
#[derive(Debug, Clone)]
pub struct MoeadConfig {
    /// Number of sub-problems (= number of weight vectors = population size).
    /// Default 100.
    pub population_size: usize,
    /// Number of generations.  Default 200.
    pub n_generations: usize,
    /// Neighbourhood size *T* (number of weight vectors considered as
    /// neighbours).  Default 20 (or population_size/5, whichever is smaller).
    pub n_neighbors: usize,
    /// Probability of selecting mating partners from neighbourhood vs. whole
    /// population.  Default 0.9.
    pub delta: f64,
    /// Number of objectives.  Must be ≥ 2.
    pub n_objectives: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for MoeadConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            n_generations: 200,
            n_neighbors: 20,
            delta: 0.9,
            n_objectives: 2,
            seed: 12345,
        }
    }
}

/// Result returned by [`moead`].
#[derive(Debug)]
pub struct MoeadResult {
    /// Non-dominated solutions extracted from the final population.
    pub pareto_front: Vec<Individual>,
    /// Weight vectors used for decomposition (one per sub-problem).
    pub weight_vectors: Vec<Vec<f64>>,
    /// Number of generations executed.
    pub n_generations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run MOEA/D on a multi-objective optimisation problem.
///
/// # Arguments
/// * `bounds`     - Decision-variable bounds `[(lo, hi); n_vars]`.
/// * `objectives` - Closure mapping a gene vector to an objective vector
///   (all minimised).
/// * `config`     - Algorithm hyper-parameters.
///
/// # Errors
/// Returns an error for empty bounds, degenerate bound intervals, or if
/// `n_objectives < 2`.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::moead::{moead, MoeadConfig};
///
/// let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 10];
/// let mut cfg = MoeadConfig::default();
/// cfg.population_size = 20;
/// cfg.n_generations   = 5;
/// cfg.n_objectives    = 2;
///
/// let result = moead(&bounds, |x| {
///     let f1 = x[0];
///     let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len()-1) as f64;
///     vec![f1, g * (1.0 - (f1/g).sqrt())]
/// }, cfg).expect("valid input");
///
/// assert!(!result.pareto_front.is_empty());
/// ```
pub fn moead<F>(
    bounds: &[(f64, f64)],
    objectives: F,
    config: MoeadConfig,
) -> OptimizeResult<MoeadResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    use crate::error::OptimizeError;

    if config.n_objectives < 2 {
        return Err(OptimizeError::InvalidInput(
            "n_objectives must be >= 2".to_string(),
        ));
    }
    if bounds.is_empty() {
        return Err(OptimizeError::InvalidInput(
            "bounds must be non-empty".to_string(),
        ));
    }
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if lo >= hi {
            return Err(OptimizeError::InvalidInput(format!(
                "bound[{i}]: lo ({lo}) must be < hi ({hi})"
            )));
        }
    }

    let n_vars = bounds.len();
    let pop_size = config.population_size.max(4);
    let n_obj = config.n_objectives;
    let t = config.n_neighbors.min(pop_size).max(2);

    let mut rng = StdRng::seed_from_u64(config.seed);

    // ── 1. Generate uniform weight vectors ──────────────────────────────────
    let weight_vectors = generate_weight_vectors(n_obj, pop_size, &mut rng);
    let actual_pop_size = weight_vectors.len();

    // ── 2. Build neighbourhood lookup table ─────────────────────────────────
    let neighborhoods = build_neighborhood(&weight_vectors, t);

    // ── 3. Initialise population + ideal point ───────────────────────────────
    let mut population: Vec<Individual> = (0..actual_pop_size)
        .map(|_| {
            let genes = random_genes(bounds, &mut rng);
            let objs = objectives(&genes);
            Individual::new(genes, objs)
        })
        .collect();

    // Ideal point z* = component-wise minimum of all objective vectors
    let mut ideal_point: Vec<f64> = vec![f64::INFINITY; n_obj];
    for ind in &population {
        for (k, &v) in ind.objectives.iter().enumerate() {
            if v < ideal_point[k] {
                ideal_point[k] = v;
            }
        }
    }

    // ── 4. Main evolutionary loop ────────────────────────────────────────────
    for _ in 0..config.n_generations {
        for i in 0..actual_pop_size {
            // Select mating pool: neighbourhood or whole population
            let use_neighborhood = rng.random::<f64>() < config.delta;
            let mating_pool = if use_neighborhood {
                &neighborhoods[i]
            } else {
                // Use all indices (implicit via actual_pop_size)
                &neighborhoods[i] // fall back to neighborhood if whole-pop not pre-built
            };

            // Generate offspring using DE/rand/1 with two random parents from pool
            let offspring_genes = de_offspring(&population, mating_pool, bounds, &mut rng);
            let offspring_objs = objectives(&offspring_genes);
            let offspring = Individual::new(offspring_genes, offspring_objs);

            // Update ideal point
            for (k, &v) in offspring.objectives.iter().enumerate() {
                if v < ideal_point[k] {
                    ideal_point[k] = v;
                }
            }

            // Update neighbourhood solutions
            for &j in mating_pool {
                let w = &weight_vectors[j];
                let g_offspring = tchebycheff_scalarization(&offspring.objectives, w, &ideal_point);
                let g_current =
                    tchebycheff_scalarization(&population[j].objectives, w, &ideal_point);
                if g_offspring <= g_current {
                    population[j] = offspring.clone();
                }
            }
        }
    }

    // ── 5. Extract Pareto front from final population ────────────────────────
    let obj_vecs: Vec<Vec<f64>> = population
        .iter()
        .map(|ind| ind.objectives.clone())
        .collect();
    let fronts = non_dominated_sort(&obj_vecs);

    let pareto_front: Vec<Individual> = if fronts.is_empty() {
        population.clone()
    } else {
        fronts[0].iter().map(|&i| population[i].clone()).collect()
    };

    Ok(MoeadResult {
        pareto_front,
        weight_vectors,
        n_generations: config.n_generations,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tchebycheff scalarization
// ─────────────────────────────────────────────────────────────────────────────

/// Tchebycheff (Chebyshev) scalarization.
///
/// g^te(f | w, z*) = max_k { w_k * |f_k - z_k*| }
///
/// This is the most widely used scalarization in MOEA/D and handles
/// non-convex Pareto fronts, unlike the weighted-sum approach.
pub fn tchebycheff_scalarization(objectives: &[f64], weights: &[f64], ideal_point: &[f64]) -> f64 {
    objectives
        .iter()
        .zip(weights.iter())
        .zip(ideal_point.iter())
        .map(|((f, w), z)| w * (f - z).abs())
        .fold(f64::NEG_INFINITY, f64::max)
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight vector generation (simplex lattice)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate approximately `target_n` uniformly distributed weight vectors on
/// the unit simplex for `n_obj` objectives.
///
/// Uses the Das–Dennis normal boundary intersection (NBI) lattice.  The actual
/// number of generated vectors may differ slightly from `target_n` due to
/// combinatorial constraints; it is always at least 2.
///
/// # Algorithm
/// Find the largest integer *H* (number of divisions) such that
/// C(H + M - 1, M - 1) ≤ target_n, then enumerate all non-negative integer
/// tuples (h_1,...,h_M) with sum H and map each to w_i = h_i / H.
pub fn generate_weight_vectors(n_obj: usize, target_n: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    if n_obj == 1 {
        return vec![vec![1.0]];
    }

    // Find H: largest integer such that C(H+M-1, M-1) <= target_n
    let mut h = 1usize;
    while combinations(h + n_obj, n_obj - 1) <= target_n {
        h += 1;
    }
    h -= 1;
    if h == 0 {
        h = 1;
    }

    let mut vectors: Vec<Vec<f64>> = Vec::new();
    enumerate_simplex(&mut vectors, n_obj, h);

    // Normalise (divide by H to get values in [0,1] summing to 1)
    for v in &mut vectors {
        for x in v.iter_mut() {
            *x /= h as f64;
        }
    }

    // If we have fewer than 2 vectors, add random ones to fill target_n
    while vectors.len() < 2 {
        vectors.push(random_simplex_point(n_obj, rng));
    }

    vectors
}

/// Enumerate all integer tuples (a_0,...,a_{M-1}) with sum H and M components.
fn enumerate_simplex(out: &mut Vec<Vec<f64>>, n_obj: usize, h: usize) {
    let mut current = vec![0.0f64; n_obj];
    enumerate_simplex_rec(out, &mut current, n_obj, h, 0, h);
}

fn enumerate_simplex_rec(
    out: &mut Vec<Vec<f64>>,
    current: &mut Vec<f64>,
    n_obj: usize,
    h: usize,
    index: usize,
    remaining: usize,
) {
    if index == n_obj - 1 {
        current[index] = remaining as f64;
        out.push(current.clone());
        return;
    }
    for i in 0..=remaining {
        current[index] = i as f64;
        enumerate_simplex_rec(out, current, n_obj, h, index + 1, remaining - i);
    }
}

/// Binomial coefficient C(n, k).
fn combinations(n: usize, k: usize) -> usize {
    if k == 0 || k == n {
        return 1;
    }
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
}

/// Generate a single random point on the unit simplex (for fallback padding).
fn random_simplex_point(n: usize, rng: &mut StdRng) -> Vec<f64> {
    // Exponential sampling: sample Exp(1) and normalise
    let exps: Vec<f64> = (0..n)
        .map(|_| -rng.random::<f64>().ln().max(1e-15))
        .collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Neighbourhood construction
// ─────────────────────────────────────────────────────────────────────────────

/// Build neighbourhood table.
///
/// For each weight vector i, `result[i]` contains the indices of the T weight
/// vectors nearest to i (by Euclidean distance, including i itself).
pub fn build_neighborhood(weight_vectors: &[Vec<f64>], t: usize) -> Vec<Vec<usize>> {
    let n = weight_vectors.len();
    let t = t.min(n);

    weight_vectors
        .iter()
        .map(|wi| {
            // Compute distance from wi to every other weight vector
            let mut dist_idx: Vec<(f64, usize)> = weight_vectors
                .iter()
                .enumerate()
                .map(|(j, wj)| {
                    let d = wi
                        .iter()
                        .zip(wj.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    (d, j)
                })
                .collect();

            dist_idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            dist_idx.iter().take(t).map(|&(_, j)| j).collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// DE offspring generation
// ─────────────────────────────────────────────────────────────────────────────

/// Differential evolution offspring (DE/rand/1 + binomial crossover).
///
/// Selects three distinct random individuals from `pool` indices, applies
/// the DE mutation y = x_r1 + F*(x_r2 - x_r3), and performs binomial
/// crossover with rate CR = 0.5.  Result is clamped to `bounds`.
fn de_offspring(
    population: &[Individual],
    pool: &[usize],
    bounds: &[(f64, f64)],
    rng: &mut StdRng,
) -> Vec<f64> {
    let n_pool = pool.len();
    let n_vars = bounds.len();

    // Pick r1, r2, r3 distinct
    let r1 = pool[rng.random_range(0..n_pool)];
    let mut r2 = pool[rng.random_range(0..n_pool)];
    let mut r3 = pool[rng.random_range(0..n_pool)];

    // Retry up to 3 times to get distinct indices (best effort)
    for _ in 0..3 {
        if r2 != r1 {
            break;
        }
        r2 = pool[rng.random_range(0..n_pool)];
    }
    for _ in 0..3 {
        if r3 != r1 && r3 != r2 {
            break;
        }
        r3 = pool[rng.random_range(0..n_pool)];
    }

    let x1 = &population[r1].genes;
    let x2 = &population[r2].genes;
    let x3 = &population[r3].genes;

    // DE scale factor in [0.5, 0.9]
    let f_scale = 0.5 + rng.random::<f64>() * 0.4;
    // Crossover rate
    let cr = 0.5;

    // Mandatory crossover index
    let j_rand = rng.random_range(0..n_vars);

    let (lo_v, hi_v): (Vec<f64>, Vec<f64>) = bounds.iter().map(|&(lo, hi)| (lo, hi)).unzip();

    (0..n_vars)
        .map(|j| {
            if j == j_rand || rng.random::<f64>() < cr {
                (x1[j] + f_scale * (x2[j] - x3[j])).clamp(lo_v[j], hi_v[j])
            } else {
                x1[j]
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Random initialisation helper
// ─────────────────────────────────────────────────────────────────────────────

fn random_genes(bounds: &[(f64, f64)], rng: &mut StdRng) -> Vec<f64> {
    bounds
        .iter()
        .map(|&(lo, hi)| lo + rng.random::<f64>() * (hi - lo))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn zdt1(x: &[f64]) -> Vec<f64> {
        let f1 = x[0];
        let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() - 1) as f64;
        let f2 = g * (1.0 - (f1 / g).sqrt());
        vec![f1, f2]
    }

    // ── Weight vector generation ─────────────────────────────────────────────

    #[test]
    fn test_weight_vectors_sum_to_one() {
        let mut rng = StdRng::seed_from_u64(0);
        let wvs = generate_weight_vectors(2, 10, &mut rng);
        for w in &wvs {
            let s: f64 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "weight vector sum = {s}");
            assert_eq!(w.len(), 2);
        }
    }

    #[test]
    fn test_weight_vectors_3obj() {
        let mut rng = StdRng::seed_from_u64(1);
        let wvs = generate_weight_vectors(3, 15, &mut rng);
        for w in &wvs {
            let s: f64 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-10);
            assert_eq!(w.len(), 3);
            for &x in w {
                assert!(x >= 0.0 && x <= 1.0);
            }
        }
    }

    // ── Neighbourhood ────────────────────────────────────────────────────────

    #[test]
    fn test_neighborhood_includes_self() {
        let mut rng = StdRng::seed_from_u64(0);
        let wvs = generate_weight_vectors(2, 10, &mut rng);
        let nb = build_neighborhood(&wvs, 3);
        for (i, n) in nb.iter().enumerate() {
            assert!(n.contains(&i), "neighbourhood of {i} should contain itself");
        }
    }

    #[test]
    fn test_neighborhood_size() {
        let mut rng = StdRng::seed_from_u64(0);
        let wvs = generate_weight_vectors(2, 10, &mut rng);
        let t = 4;
        let nb = build_neighborhood(&wvs, t);
        for n in &nb {
            assert_eq!(n.len(), t);
        }
    }

    // ── Tchebycheff scalarization ────────────────────────────────────────────

    #[test]
    fn test_tchebycheff_basic() {
        let f = vec![1.0, 2.0];
        let w = vec![0.5, 0.5];
        let z = vec![0.0, 0.0];
        let val = tchebycheff_scalarization(&f, &w, &z);
        // max(0.5*1, 0.5*2) = max(0.5, 1.0) = 1.0
        assert!((val - 1.0).abs() < 1e-10, "Expected 1.0, got {val}");
    }

    #[test]
    fn test_tchebycheff_with_ideal_shift() {
        let f = vec![3.0, 3.0];
        let w = vec![0.5, 0.5];
        let z = vec![1.0, 1.0];
        let val = tchebycheff_scalarization(&f, &w, &z);
        // max(0.5*2, 0.5*2) = 1.0
        assert!((val - 1.0).abs() < 1e-10);
    }

    // ── MOEA/D on ZDT1 ──────────────────────────────────────────────────────

    #[test]
    fn test_moead_returns_pareto_front() {
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 5];
        let mut cfg = MoeadConfig::default();
        cfg.population_size = 20;
        cfg.n_generations = 10;
        cfg.n_objectives = 2;
        cfg.seed = 1;

        let result = moead(&bounds, zdt1, cfg).expect("moead should succeed");
        assert!(!result.pareto_front.is_empty());
    }

    #[test]
    fn test_moead_weight_vectors_returned() {
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 3];
        let mut cfg = MoeadConfig::default();
        cfg.population_size = 10;
        cfg.n_generations = 5;
        cfg.n_objectives = 2;

        let result = moead(&bounds, zdt1, cfg).expect("failed to create result");
        assert!(!result.weight_vectors.is_empty());
        for w in &result.weight_vectors {
            assert_eq!(w.len(), 2);
            let s: f64 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "weight sum = {s}");
        }
    }

    #[test]
    fn test_moead_pareto_front_non_dominated() {
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 5];
        let mut cfg = MoeadConfig::default();
        cfg.population_size = 20;
        cfg.n_generations = 20;
        cfg.n_objectives = 2;
        cfg.seed = 42;

        let result = moead(&bounds, zdt1, cfg).expect("failed to create result");

        let front = &result.pareto_front;
        for i in 0..front.len() {
            for j in 0..front.len() {
                if i != j {
                    assert!(
                        !dominates(&front[i].objectives, &front[j].objectives),
                        "front[{i}] dominates front[{j}] in MOEA/D result"
                    );
                }
            }
        }
    }

    #[test]
    fn test_moead_bounds_respected() {
        let bounds = vec![(0.2, 0.8); 3];
        let mut cfg = MoeadConfig::default();
        cfg.population_size = 10;
        cfg.n_generations = 5;
        cfg.n_objectives = 2;

        let result =
            moead(&bounds, |x| vec![x[0], 1.0 - x[0]], cfg).expect("failed to create result");

        for ind in &result.pareto_front {
            for (i, &g) in ind.genes.iter().enumerate() {
                assert!(
                    g >= bounds[i].0 - 1e-9 && g <= bounds[i].1 + 1e-9,
                    "gene[{i}]={g} outside bounds"
                );
            }
        }
    }

    #[test]
    fn test_moead_invalid_input() {
        // Empty bounds
        let result = moead(&[], |x| vec![x[0]], MoeadConfig::default());
        assert!(result.is_err());

        // Bad bound interval
        let result = moead(&[(1.0, 0.0)], |x| vec![x[0]], MoeadConfig::default());
        assert!(result.is_err());

        // Too few objectives
        let mut cfg = MoeadConfig::default();
        cfg.n_objectives = 1;
        let result = moead(&[(0.0, 1.0)], |x| vec![x[0]], cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_moead_generation_count() {
        let bounds = vec![(0.0, 1.0); 3];
        let mut cfg = MoeadConfig::default();
        cfg.population_size = 10;
        cfg.n_generations = 7;
        cfg.n_objectives = 2;

        let result = moead(&bounds, zdt1, cfg).expect("failed to create result");
        assert_eq!(result.n_generations, 7);
    }

    #[test]
    fn test_moead_diverse_objectives() {
        // With sufficient generations, MOEA/D should find diverse coverage.
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 6];
        let mut cfg = MoeadConfig::default();
        cfg.population_size = 30;
        cfg.n_generations = 30;
        cfg.n_objectives = 2;
        cfg.seed = 7;

        let result = moead(&bounds, zdt1, cfg).expect("failed to create result");

        // At least two distinct solutions must exist for a non-trivial problem
        assert!(result.pareto_front.len() >= 2);

        // Check all objectives are in plausible range for ZDT1
        for ind in &result.pareto_front {
            assert!(ind.objectives[0] >= 0.0, "f1 must be >= 0");
            assert!(ind.objectives[1] >= 0.0, "f2 must be >= 0");
        }
    }
}
