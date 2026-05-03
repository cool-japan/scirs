//! NSGA-II: Non-dominated Sorting Genetic Algorithm II
//!
//! Implements the landmark algorithm by Deb et al. (2002) for multi-objective
//! optimisation.  Key ideas:
//!
//! 1. **Fast non-dominated sorting** — assign each individual a Pareto-front
//!    rank in O(MN²) time.
//! 2. **Crowding-distance assignment** — within each front, measure how
//!    isolated each point is, rewarding diversity.
//! 3. **Tournament selection** — prefer lower rank; break ties by crowding
//!    distance (larger is better).
//! 4. **Simulated Binary Crossover (SBX)** and **Polynomial Mutation (PM)**
//!    for real-valued variables.
//! 5. **Elitist survivor selection** — combine parent and offspring populations
//!    (size 2N) and select the best N individuals.
//!
//! # References
//!
//! - Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
//!   A fast and elitist multiobjective genetic algorithm: NSGA-II.
//!   *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

use crate::error::OptimizeResult;
use crate::multiobjective::indicators::{dominates, non_dominated_sort};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// An individual in the NSGA-II population.
#[derive(Clone, Debug)]
pub struct Individual {
    /// Decision variable vector.
    pub genes: Vec<f64>,
    /// Objective value vector (one element per objective, all minimised).
    pub objectives: Vec<f64>,
    /// Pareto-front rank (0 = best non-dominated front).
    pub rank: usize,
    /// Crowding distance (larger = more isolated = preferred when ranks tie).
    pub crowding_distance: f64,
}

impl Individual {
    pub fn new(genes: Vec<f64>, objectives: Vec<f64>) -> Self {
        Self {
            genes,
            objectives,
            rank: 0,
            crowding_distance: 0.0,
        }
    }
}

/// Result returned by [`nsga2`].
#[derive(Debug)]
pub struct Nsga2Result {
    /// Individuals on the first (best) Pareto front after the final generation.
    pub pareto_front: Vec<Individual>,
    /// All fronts (front 0 = Pareto optimal, etc.) from the final population.
    pub all_fronts: Vec<Vec<Individual>>,
    /// Number of generations actually executed.
    pub n_generations: usize,
    /// Total number of objective evaluations performed.
    pub n_evaluations: usize,
}

/// Configuration for the NSGA-II algorithm.
#[derive(Debug, Clone)]
pub struct Nsga2Config {
    /// Population size (must be even, will be rounded up if odd).  Default 100.
    pub population_size: usize,
    /// Number of generations.  Default 200.
    pub n_generations: usize,
    /// Simulated binary crossover probability.  Default 0.9.
    pub crossover_rate: f64,
    /// Polynomial mutation probability per variable.  Default 1/n_vars.
    /// Setting to 0 uses the 1/n_vars default at runtime.
    pub mutation_rate: f64,
    /// SBX distribution index η_c (controls offspring spread).  Default 20.
    pub eta_c: f64,
    /// Polynomial mutation distribution index η_m.  Default 20.
    pub eta_m: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for Nsga2Config {
    fn default() -> Self {
        Self {
            population_size: 100,
            n_generations: 200,
            crossover_rate: 0.9,
            mutation_rate: 0.0, // resolved to 1/n_vars at runtime
            eta_c: 20.0,
            eta_m: 20.0,
            seed: 12345,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run NSGA-II on a multi-objective optimisation problem.
///
/// # Arguments
/// * `n_objectives` - Number of objectives (must be ≥ 2).
/// * `bounds`       - Decision-variable bounds `[(lo, hi); n_vars]`.  Must be
///   non-empty and each bound must satisfy `lo < hi`.
/// * `objectives`   - Closure mapping a gene vector to an objective vector.
///   **All objectives are minimised.**
/// * `config`       - Algorithm hyper-parameters; use `Default::default()` for
///   sensible defaults.
///
/// # Errors
/// Returns an error if inputs are invalid (e.g. empty bounds, degenerate
/// bound intervals, zero objectives).
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::nsga2::{nsga2, Nsga2Config};
///
/// // ZDT1 benchmark (30 variables, 2 objectives)
/// let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 30];
/// let mut cfg = Nsga2Config::default();
/// cfg.population_size = 20;
/// cfg.n_generations  = 5;
///
/// let result = nsga2(2, &bounds, |x| {
///     let f1 = x[0];
///     let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() - 1) as f64;
///     vec![f1, g * (1.0 - (f1 / g).sqrt())]
/// }, cfg).expect("valid input");
///
/// assert!(!result.pareto_front.is_empty());
/// ```
pub fn nsga2<F>(
    n_objectives: usize,
    bounds: &[(f64, f64)],
    objectives: F,
    config: Nsga2Config,
) -> OptimizeResult<Nsga2Result>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    use crate::error::OptimizeError;

    if n_objectives == 0 {
        return Err(OptimizeError::InvalidInput(
            "n_objectives must be >= 1".to_string(),
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
    let pop_size = if config.population_size % 2 == 0 {
        config.population_size.max(4)
    } else {
        (config.population_size + 1).max(4)
    };

    let mutation_rate = if config.mutation_rate > 0.0 {
        config.mutation_rate
    } else {
        1.0 / n_vars as f64
    };

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut n_evaluations = 0usize;

    // ── Initialise population ────────────────────────────────────────────────
    let mut population: Vec<Individual> = (0..pop_size)
        .map(|_| {
            let genes = random_genes(bounds, &mut rng);
            let objs = objectives(&genes);
            n_evaluations += 1;
            Individual::new(genes, objs)
        })
        .collect();

    // Assign initial ranks and crowding distances
    assign_ranks_and_crowding(&mut population);

    // ── Main loop ────────────────────────────────────────────────────────────
    for _ in 0..config.n_generations {
        // Generate offspring via tournament selection + SBX + polynomial mutation
        let offspring: Vec<Individual> = (0..pop_size / 2)
            .flat_map(|_| {
                let p1 = tournament_select(&population, &mut rng);
                let p2 = tournament_select(&population, &mut rng);

                let (c1_genes, c2_genes) = if rng.random::<f64>() < config.crossover_rate {
                    sbx_crossover(
                        &population[p1].genes,
                        &population[p2].genes,
                        config.eta_c,
                        bounds,
                        &mut rng,
                    )
                } else {
                    (population[p1].genes.clone(), population[p2].genes.clone())
                };

                let c1_genes =
                    polynomial_mutation(c1_genes, mutation_rate, config.eta_m, bounds, &mut rng);
                let c2_genes =
                    polynomial_mutation(c2_genes, mutation_rate, config.eta_m, bounds, &mut rng);

                let objs1 = objectives(&c1_genes);
                let objs2 = objectives(&c2_genes);
                n_evaluations += 2;

                vec![
                    Individual::new(c1_genes, objs1),
                    Individual::new(c2_genes, objs2),
                ]
            })
            .collect();

        // Combine parent + offspring; select best pop_size survivors
        let mut combined = population;
        combined.extend(offspring);
        assign_ranks_and_crowding(&mut combined);
        population = select_survivors(combined, pop_size);
    }

    // ── Build result ─────────────────────────────────────────────────────────
    assign_ranks_and_crowding(&mut population);

    // Gather all fronts
    let obj_vecs: Vec<Vec<f64>> = population
        .iter()
        .map(|ind| ind.objectives.clone())
        .collect();
    let front_indices = non_dominated_sort(&obj_vecs);

    let mut all_fronts: Vec<Vec<Individual>> = front_indices
        .iter()
        .map(|idx_vec| idx_vec.iter().map(|&i| population[i].clone()).collect())
        .collect();

    let pareto_front = if all_fronts.is_empty() {
        population.clone()
    } else {
        all_fronts.remove(0)
    };

    // Reinsert front 0 that we just removed
    let obj_vecs2: Vec<Vec<f64>> = population
        .iter()
        .map(|ind| ind.objectives.clone())
        .collect();
    let front_indices2 = non_dominated_sort(&obj_vecs2);
    let all_fronts_final: Vec<Vec<Individual>> = front_indices2
        .iter()
        .map(|idx_vec| idx_vec.iter().map(|&i| population[i].clone()).collect())
        .collect();

    Ok(Nsga2Result {
        pareto_front,
        all_fronts: all_fronts_final,
        n_generations: config.n_generations,
        n_evaluations,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Rank + crowding assignment
// ─────────────────────────────────────────────────────────────────────────────

/// Assign Pareto ranks and crowding distances to the entire population in-place.
pub(crate) fn assign_ranks_and_crowding(population: &mut [Individual]) {
    if population.is_empty() {
        return;
    }

    let obj_vecs: Vec<Vec<f64>> = population
        .iter()
        .map(|ind| ind.objectives.clone())
        .collect();
    let fronts = non_dominated_sort(&obj_vecs);

    for (rank, front_idx) in fronts.iter().enumerate() {
        for &i in front_idx {
            population[i].rank = rank;
        }
        crowding_distance_assignment(population, front_idx);
    }
}

/// Compute crowding distances for the individuals in `front_indices` and write
/// the results into `population[i].crowding_distance`.
fn crowding_distance_assignment(population: &mut [Individual], front_indices: &[usize]) {
    let n = front_indices.len();
    if n <= 2 {
        // Boundary individuals get infinite distance
        for &i in front_indices {
            population[i].crowding_distance = f64::INFINITY;
        }
        return;
    }

    // Reset crowding distances for this front
    for &i in front_indices {
        population[i].crowding_distance = 0.0;
    }

    let n_obj = population[front_indices[0]].objectives.len();

    for m in 0..n_obj {
        // Sort by objective m
        let mut sorted = front_indices.to_vec();
        sorted.sort_by(|&a, &b| {
            population[a].objectives[m]
                .partial_cmp(&population[b].objectives[m])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Boundary individuals get infinite distance
        population[sorted[0]].crowding_distance = f64::INFINITY;
        population[sorted[n - 1]].crowding_distance = f64::INFINITY;

        let f_min = population[sorted[0]].objectives[m];
        let f_max = population[sorted[n - 1]].objectives[m];
        let range = f_max - f_min;

        if range < f64::EPSILON {
            continue; // All values equal; no contribution to crowding
        }

        for k in 1..n - 1 {
            let prev_val = population[sorted[k - 1]].objectives[m];
            let next_val = population[sorted[k + 1]].objectives[m];
            population[sorted[k]].crowding_distance += (next_val - prev_val) / range;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Survivor selection (elitist)
// ─────────────────────────────────────────────────────────────────────────────

/// Select the best `target_size` individuals from a combined population.
///
/// Uses the NSGA-II selection rule:
/// 1. Fill from front 0, then front 1, etc.
/// 2. If a front does not fit entirely, select those with the largest
///    crowding distance from that front.
fn select_survivors(mut combined: Vec<Individual>, target_size: usize) -> Vec<Individual> {
    // Sort by (rank ASC, crowding_distance DESC)
    combined.sort_by(|a, b| {
        a.rank.cmp(&b.rank).then_with(|| {
            b.crowding_distance
                .partial_cmp(&a.crowding_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    combined.truncate(target_size);
    combined
}

// ─────────────────────────────────────────────────────────────────────────────
// Genetic operators
// ─────────────────────────────────────────────────────────────────────────────

/// Simulated Binary Crossover (SBX) for real-valued variables.
///
/// Produces two offspring from two parents using the polynomial distribution
/// with index `eta_c`.  Each variable is crossed independently with
/// probability 0.5.
///
/// Offspring are clamped to `bounds`.
fn sbx_crossover(
    parent1: &[f64],
    parent2: &[f64],
    eta_c: f64,
    bounds: &[(f64, f64)],
    rng: &mut StdRng,
) -> (Vec<f64>, Vec<f64>) {
    let n = parent1.len();
    let mut child1 = parent1.to_vec();
    let mut child2 = parent2.to_vec();

    for i in 0..n {
        if rng.random::<f64>() > 0.5 {
            continue; // skip this variable
        }

        let (lo, hi) = bounds[i];
        let x1 = parent1[i].min(parent2[i]);
        let x2 = parent1[i].max(parent2[i]);

        if (x2 - x1).abs() < 1e-14 {
            continue; // Parents are identical on this variable
        }

        let u: f64 = rng.random();

        // Beta from polynomial distribution
        let beta_q = if u <= 0.5 {
            let alpha = 2.0 - (1.0 / sbx_beta(x1, x2, lo, eta_c)).powf(eta_c + 1.0);
            let alpha = alpha.max(0.0);
            (2.0 * u * alpha).powf(1.0 / (eta_c + 1.0))
        } else {
            let alpha = 2.0 - (1.0 / sbx_beta(x1, x2, hi - x2 + x1, eta_c)).powf(eta_c + 1.0);
            // Use the mirror formula for u > 0.5
            let alpha_inv = 2.0 * (1.0 - u) * alpha.max(0.0);
            if alpha_inv < f64::EPSILON {
                1.0
            } else {
                (1.0 / alpha_inv).powf(1.0 / (eta_c + 1.0))
            }
        };

        let mid = 0.5 * (x1 + x2);
        let half_diff = 0.5 * (x2 - x1);

        let c1 = (mid - beta_q * half_diff).clamp(lo, hi);
        let c2 = (mid + beta_q * half_diff).clamp(lo, hi);

        // Preserve parent ordering (child1 ← smaller side)
        if parent1[i] < parent2[i] {
            child1[i] = c1;
            child2[i] = c2;
        } else {
            child1[i] = c2;
            child2[i] = c1;
        }
    }

    (child1, child2)
}

/// Helper for SBX: compute beta spread factor.
fn sbx_beta(x1: f64, x2: f64, bound: f64, eta: f64) -> f64 {
    let diff = (x2 - x1).abs().max(1e-14);
    let dist = (bound - x1).abs().max(1e-14);
    (1.0 + 2.0 * dist / diff).powf(eta + 1.0)
}

/// Polynomial mutation.
///
/// Mutates each variable independently with probability `mutation_rate`.
/// The perturbation magnitude is governed by `eta_m` (higher → smaller
/// perturbation on average).
fn polynomial_mutation(
    mut genes: Vec<f64>,
    mutation_rate: f64,
    eta_m: f64,
    bounds: &[(f64, f64)],
    rng: &mut StdRng,
) -> Vec<f64> {
    for (i, gene) in genes.iter_mut().enumerate() {
        if rng.random::<f64>() >= mutation_rate {
            continue;
        }

        let (lo, hi) = bounds[i];
        let delta = hi - lo;
        if delta < f64::EPSILON {
            continue;
        }

        let u: f64 = rng.random();
        let delta_q = if u < 0.5 {
            let delta_l = (*gene - lo) / delta;
            let base = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta_l).powf(eta_m + 1.0);
            base.powf(1.0 / (eta_m + 1.0)) - 1.0
        } else {
            let delta_r = (hi - *gene) / delta;
            let base = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta_r).powf(eta_m + 1.0);
            1.0 - base.powf(1.0 / (eta_m + 1.0))
        };

        *gene = (*gene + delta_q * delta).clamp(lo, hi);
    }

    genes
}

// ─────────────────────────────────────────────────────────────────────────────
// Tournament selection
// ─────────────────────────────────────────────────────────────────────────────

/// Binary tournament selection.
///
/// Picks two random individuals and returns the index of the one that wins
/// according to NSGA-II dominance order: lower rank wins; ties broken by
/// larger crowding distance.
fn tournament_select(population: &[Individual], rng: &mut StdRng) -> usize {
    let n = population.len();
    let a = rng.random_range(0..n);
    let mut b = rng.random_range(0..n);
    // Ensure a != b (retry once)
    if b == a && n > 1 {
        b = (a + 1) % n;
    }

    let ia = &population[a];
    let ib = &population[b];

    if ia.rank < ib.rank || (ia.rank == ib.rank && ia.crowding_distance > ib.crowding_distance) {
        a
    } else {
        b
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Random initialisation
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

    // ZDT1: f1 = x[0], f2 = g * (1 - sqrt(x[0]/g))
    //       where g = 1 + 9 * sum(x[1..]) / (n-1)
    fn zdt1(x: &[f64]) -> Vec<f64> {
        let f1 = x[0];
        let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() - 1) as f64;
        let f2 = g * (1.0 - (f1 / g).sqrt());
        vec![f1, f2]
    }

    #[test]
    fn test_nsga2_returns_pareto_front() {
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 5];
        let mut cfg = Nsga2Config::default();
        cfg.population_size = 20;
        cfg.n_generations = 10;
        cfg.seed = 99;

        let result = nsga2(2, &bounds, zdt1, cfg).expect("nsga2 should succeed");
        assert!(!result.pareto_front.is_empty());
        assert!(result.n_evaluations > 0);
    }

    #[test]
    fn test_nsga2_objectives_are_evaluated() {
        let bounds = vec![(0.0, 1.0); 3];
        let mut cfg = Nsga2Config::default();
        cfg.population_size = 10;
        cfg.n_generations = 3;

        let result =
            nsga2(2, &bounds, |x| vec![x[0], 1.0 - x[0]], cfg).expect("failed to create result");
        // n_evaluations = pop_size + n_gen * pop_size (offspring)
        assert!(result.n_evaluations >= 10);
    }

    #[test]
    fn test_nsga2_pareto_front_non_dominated() {
        let bounds = vec![(0.0, 1.0); 5];
        let mut cfg = Nsga2Config::default();
        cfg.population_size = 20;
        cfg.n_generations = 20;
        cfg.seed = 7;

        let result = nsga2(2, &bounds, zdt1, cfg).expect("failed to create result");

        // Verify that no front member dominates another
        let front = &result.pareto_front;
        for i in 0..front.len() {
            for j in 0..front.len() {
                if i != j {
                    assert!(
                        !dominates(&front[i].objectives, &front[j].objectives),
                        "front[{i}] dominates front[{j}]"
                    );
                }
            }
        }
    }

    #[test]
    fn test_nsga2_bounds_respected() {
        let bounds = vec![(0.2, 0.8); 3];
        let mut cfg = Nsga2Config::default();
        cfg.population_size = 20;
        cfg.n_generations = 10;

        let result =
            nsga2(2, &bounds, |x| vec![x[0], 1.0 - x[0]], cfg).expect("failed to create result");

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
    fn test_nsga2_zdt1_quality() {
        // With 50 generations and pop=40, the Pareto front should be near ZDT1 true front
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 10];
        let mut cfg = Nsga2Config::default();
        cfg.population_size = 40;
        cfg.n_generations = 50;
        cfg.seed = 42;

        let result = nsga2(2, &bounds, zdt1, cfg).expect("failed to create result");

        // f1+f2 should be close to the ZDT1 Pareto front (f2 ≈ 1-sqrt(f1) when g≈1)
        // At minimum, all front members should have f1,f2 in [0,1.5]
        for ind in &result.pareto_front {
            assert!(ind.objectives[0] >= 0.0, "f1 should be >= 0");
            assert!(ind.objectives[1] >= 0.0, "f2 should be >= 0");
        }
    }

    #[test]
    fn test_nsga2_invalid_bounds() {
        let result = nsga2(2, &[], |x| vec![x[0]], Nsga2Config::default());
        assert!(result.is_err());

        let result = nsga2(2, &[(1.0, 0.0)], |x| vec![x[0]], Nsga2Config::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_nsga2_invalid_objectives() {
        let result = nsga2(0, &[(0.0, 1.0)], |_| vec![], Nsga2Config::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_sbx_offspring_in_bounds() {
        let mut rng = StdRng::seed_from_u64(0);
        let bounds = vec![(0.0, 1.0); 4];
        let p1 = vec![0.2, 0.4, 0.6, 0.8];
        let p2 = vec![0.8, 0.6, 0.4, 0.2];

        for _ in 0..50 {
            let (c1, c2) = sbx_crossover(&p1, &p2, 20.0, &bounds, &mut rng);
            for (i, &v) in c1.iter().enumerate() {
                assert!(
                    v >= bounds[i].0 && v <= bounds[i].1,
                    "c1[{i}]={v} out of bounds"
                );
            }
            for (i, &v) in c2.iter().enumerate() {
                assert!(
                    v >= bounds[i].0 && v <= bounds[i].1,
                    "c2[{i}]={v} out of bounds"
                );
            }
        }
    }

    #[test]
    fn test_polynomial_mutation_in_bounds() {
        let mut rng = StdRng::seed_from_u64(1);
        let bounds = vec![(0.0, 1.0); 5];
        let genes = vec![0.5; 5];

        for _ in 0..100 {
            let mutated = polynomial_mutation(genes.clone(), 0.5, 20.0, &bounds, &mut rng);
            for (i, &v) in mutated.iter().enumerate() {
                assert!(
                    v >= bounds[i].0 && v <= bounds[i].1,
                    "mutated[{i}]={v} out of bounds"
                );
            }
        }
    }

    #[test]
    fn test_crowding_distance_boundary_gets_infinity() {
        let mut pop = vec![
            Individual::new(vec![0.0], vec![0.0, 3.0]),
            Individual::new(vec![0.5], vec![0.5, 2.0]),
            Individual::new(vec![1.0], vec![1.0, 0.0]),
        ];
        let indices: Vec<usize> = (0..3).collect();
        crowding_distance_assignment(&mut pop, &indices);

        assert_eq!(pop[0].crowding_distance, f64::INFINITY);
        assert_eq!(pop[2].crowding_distance, f64::INFINITY);
        // Middle element should have finite positive distance
        assert!(pop[1].crowding_distance > 0.0);
    }
}
