//! NSGA-III: Non-dominated Sorting Genetic Algorithm III
//!
//! Implements the many-objective evolutionary algorithm by Deb & Jain (2014).
//! NSGA-III extends NSGA-II with structured reference points (Das-Dennis simplex
//! lattice) instead of crowding distance for diversity preservation. It is
//! particularly effective for problems with more than 3 objectives where
//! NSGA-II's crowding distance degrades.
//!
//! # Algorithm outline
//!
//! 1. **Reference point generation**: Construct a set of structured reference
//!    points on the unit hyperplane using the Das-Dennis method.
//! 2. **Fast non-dominated sorting**: Same as NSGA-II — assign each individual
//!    a Pareto-front rank.
//! 3. **Reference-point-based niching**: For the critical (last-included) front,
//!    use association + niche counting to maintain diversity with respect to
//!    structured reference points.
//! 4. **Elitist survivor selection**: Deterministically fill the next population
//!    by adding complete fronts, then applying niching to the final partial front.
//!
//! # References
//!
//! - Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization
//!   algorithm using reference-point-based nondominated sorting approach, Part I:
//!   Solving problems with box constraints. *IEEE TEC*, 18(4), 577–601.
//! - Jain, H., & Deb, K. (2014). An evolutionary many-objective optimization
//!   algorithm using reference-point based nondominated sorting approach, Part II:
//!   Handling constraints and extending to an adaptive approach. *IEEE TEC*,
//!   18(4), 602–622.

use crate::error::{OptimizeError, OptimizeResult};
use crate::multiobjective::indicators::{dominates, non_dominated_sort};
use crate::multiobjective::nsga2::{Individual, Nsga2Config};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the NSGA-III algorithm.
///
/// Most fields are identical to [`Nsga2Config`]; the key difference is that
/// NSGA-III accepts a `reference_point_divisions` parameter controlling the
/// density of the Das-Dennis reference point lattice, and optionally a set
/// of user-supplied adaptive reference points layered on top.
#[derive(Debug, Clone)]
pub struct Nsga3Config {
    /// Population size (will be rounded to the nearest feasible size ≥ this
    /// value to accommodate the reference point lattice).  Default 100.
    pub population_size: usize,
    /// Number of generations.  Default 200.
    pub n_generations: usize,
    /// Number of divisions on each objective axis for the primary reference
    /// point lattice (Das-Dennis).  Default 12 for 2 objectives, 6 for 3-5,
    /// 3 for 6+.  Setting to 0 uses the adaptive default.
    pub n_divisions: usize,
    /// Optional second-layer divisions for an inner lattice (used in
    /// two-layer reference point generation for many-objective problems).
    /// If `Some(d)`, an additional inner lattice with `d` divisions is
    /// generated and merged with the outer lattice. Default `None`.
    pub n_divisions_inner: Option<usize>,
    /// Simulated binary crossover probability.  Default 0.9.
    pub crossover_rate: f64,
    /// Polynomial mutation probability per variable.  Default 1/n_vars.
    pub mutation_rate: f64,
    /// SBX distribution index η_c.  Default 20.
    pub eta_c: f64,
    /// Polynomial mutation distribution index η_m.  Default 20.
    pub eta_m: f64,
    /// RNG seed for reproducibility.  Default 12345.
    pub seed: u64,
}

impl Default for Nsga3Config {
    fn default() -> Self {
        Self {
            population_size: 100,
            n_generations: 200,
            n_divisions: 0, // auto-select
            n_divisions_inner: None,
            crossover_rate: 0.9,
            mutation_rate: 0.0, // resolved at runtime
            eta_c: 20.0,
            eta_m: 20.0,
            seed: 12345,
        }
    }
}

/// Result returned by [`nsga3`].
#[derive(Debug)]
pub struct Nsga3Result {
    /// Individuals on the first (best) Pareto front after the final generation.
    pub pareto_front: Vec<Individual>,
    /// All fronts from the final generation (front 0 = Pareto optimal).
    pub all_fronts: Vec<Vec<Individual>>,
    /// Reference points used during the run.
    pub reference_points: Vec<Vec<f64>>,
    /// Number of generations executed.
    pub n_generations: usize,
    /// Total number of objective evaluations.
    pub n_evaluations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run NSGA-III on a many-objective optimisation problem.
///
/// # Arguments
/// * `n_objectives` - Number of objectives (must be ≥ 2; designed for ≥ 4).
/// * `bounds`       - Decision-variable bounds `[(lo, hi); n_vars]`.
/// * `objectives`   - Closure mapping a gene vector to objective values
///   (all minimised).
/// * `config`       - Algorithm hyper-parameters.
///
/// # Errors
/// Returns an error for empty bounds, degenerate bounds, or < 2 objectives.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::nsga3::{nsga3, Nsga3Config};
///
/// // DTLZ2 benchmark: 3 variables, 3 objectives
/// let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 7];
/// let mut cfg = Nsga3Config::default();
/// cfg.population_size = 20;
/// cfg.n_generations   = 10;
///
/// let result = nsga3(3, &bounds, |x| {
///     let n = x.len();
///     let k = n - 3 + 1;
///     let g: f64 = x[n-k..].iter().map(|&xi| (xi - 0.5).powi(2)).sum();
///     let f1 = (1.0 + g) * x[0].cos() * x[1].cos();
///     let f2 = (1.0 + g) * x[0].cos() * x[1].sin();
///     let f3 = (1.0 + g) * x[0].sin();
///     vec![f1, f2, f3]
/// }, cfg).expect("valid input");
///
/// assert!(!result.pareto_front.is_empty());
/// ```
pub fn nsga3<F>(
    n_objectives: usize,
    bounds: &[(f64, f64)],
    objectives: F,
    config: Nsga3Config,
) -> OptimizeResult<Nsga3Result>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if n_objectives < 2 {
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
    let mutation_rate = if config.mutation_rate > 0.0 {
        config.mutation_rate
    } else {
        1.0 / n_vars as f64
    };

    // ── Reference point generation ───────────────────────────────────────────
    let n_div = if config.n_divisions > 0 {
        config.n_divisions
    } else {
        // Auto-select number of divisions based on number of objectives
        match n_objectives {
            2..=3 => 12,
            4..=5 => 6,
            6..=8 => 4,
            _ => 3,
        }
    };

    let mut ref_points = generate_reference_points(n_objectives, n_div);

    // Optional inner (second) lattice for two-layer reference points
    if let Some(n_div_inner) = config.n_divisions_inner {
        let inner = generate_reference_points_inner(n_objectives, n_div_inner);
        ref_points.extend(inner);
    }

    // Determine population size: must be >= number of reference points for
    // good coverage, and even for pairing in reproduction
    let n_ref = ref_points.len();
    let pop_size = {
        let desired = config.population_size.max(n_ref);
        if desired % 2 == 0 {
            desired
        } else {
            desired + 1
        }
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

    assign_ranks(&mut population);

    // ── Main evolutionary loop ───────────────────────────────────────────────
    for _ in 0..config.n_generations {
        // Generate offspring via SBX + polynomial mutation
        let offspring: Vec<Individual> = (0..pop_size / 2)
            .flat_map(|_| {
                let p1 = tournament_select_by_rank(&population, &mut rng);
                let p2 = tournament_select_by_rank(&population, &mut rng);

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

        // Combine parent + offspring
        let mut combined = population;
        combined.extend(offspring);
        assign_ranks(&mut combined);

        // NSGA-III survivor selection using reference-point niching
        population = nsga3_select(&mut combined, &ref_points, pop_size, &mut rng);
    }

    // ── Build result ─────────────────────────────────────────────────────────
    assign_ranks(&mut population);
    let obj_vecs: Vec<Vec<f64>> = population
        .iter()
        .map(|ind| ind.objectives.clone())
        .collect();
    let front_indices = non_dominated_sort(&obj_vecs);

    let all_fronts: Vec<Vec<Individual>> = front_indices
        .iter()
        .map(|idx_vec| idx_vec.iter().map(|&i| population[i].clone()).collect())
        .collect();

    let pareto_front = if all_fronts.is_empty() {
        population.clone()
    } else {
        all_fronts[0].clone()
    };

    Ok(Nsga3Result {
        pareto_front,
        all_fronts,
        reference_points: ref_points,
        n_generations: config.n_generations,
        n_evaluations,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Reference point generation (Das-Dennis simplex lattice)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate structured reference points on the unit hyperplane using the
/// Das-Dennis lattice (simplex lattice design).
///
/// For `n_obj` objectives and `n_divisions` H, generates all points
/// (a_1/H, ..., a_M/H) where each a_i is a non-negative integer and their
/// sum equals H.  The result lies on the M-dimensional unit simplex.
///
/// The total number of points is C(H + M - 1, M - 1).
pub fn generate_reference_points(n_obj: usize, n_divisions: usize) -> Vec<Vec<f64>> {
    let mut points: Vec<Vec<f64>> = Vec::new();
    let mut current = vec![0.0f64; n_obj];
    enumerate_simplex(
        &mut points,
        &mut current,
        n_obj,
        n_divisions,
        0,
        n_divisions,
    );

    // Normalise by dividing by n_divisions
    for p in &mut points {
        for x in p.iter_mut() {
            *x /= n_divisions as f64;
        }
    }
    points
}

/// Generate inner reference points for the two-layer reference point approach.
///
/// The inner lattice is scaled to lie inside the simplex, avoiding boundary
/// degeneracy for certain problem types.  Points are shifted toward the
/// centroid: p' = p * (1 - 1/M) + 1/M^2.
pub fn generate_reference_points_inner(n_obj: usize, n_divisions: usize) -> Vec<Vec<f64>> {
    let mut points = generate_reference_points(n_obj, n_divisions);
    let scale = 1.0 - 1.0 / n_obj as f64;
    let offset = 1.0 / (n_obj * n_obj) as f64;
    for p in &mut points {
        for x in p.iter_mut() {
            *x = *x * scale + offset;
        }
    }
    points
}

fn enumerate_simplex(
    out: &mut Vec<Vec<f64>>,
    current: &mut Vec<f64>,
    n_obj: usize,
    n_divisions: usize,
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
        enumerate_simplex(out, current, n_obj, n_divisions, index + 1, remaining - i);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Normalization utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the ideal point (component-wise minimum) of a population.
fn ideal_point(population: &[Individual]) -> Vec<f64> {
    if population.is_empty() {
        return vec![];
    }
    let n_obj = population[0].objectives.len();
    let mut ideal = vec![f64::INFINITY; n_obj];
    for ind in population {
        for (k, &v) in ind.objectives.iter().enumerate() {
            if k < ideal.len() && v < ideal[k] {
                ideal[k] = v;
            }
        }
    }
    ideal
}

/// Compute the nadir (approximate worst-boundary) point using extreme points
/// from each objective axis.  Uses the achievement scalarization function (ASF)
/// approach: for each objective k, find the individual minimizing
/// max_i { (f_i - z_i*) / w_i } with weight vector e_k (axis vector).
fn nadir_estimate(population: &[Individual], ideal: &[f64]) -> Vec<f64> {
    let n_obj = ideal.len();
    let mut nadir = vec![f64::NEG_INFINITY; n_obj];

    for k in 0..n_obj {
        // Construct weight vector: 1e-6 everywhere, 1.0 on axis k
        let mut w = vec![1e-6f64; n_obj];
        w[k] = 1.0;

        // Find individual minimizing ASF (axis-aligned achievement)
        let best_ind = population.iter().min_by(|a, b| {
            let asf_a = asf(&a.objectives, ideal, &w);
            let asf_b = asf(&b.objectives, ideal, &w);
            asf_a
                .partial_cmp(&asf_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(ind) = best_ind {
            for (j, &v) in ind.objectives.iter().enumerate() {
                if v > nadir[j] {
                    nadir[j] = v;
                }
            }
        }
    }

    // Fallback: ensure nadir > ideal everywhere
    for k in 0..n_obj {
        if nadir[k] <= ideal[k] {
            nadir[k] = ideal[k] + 1.0;
        }
    }

    nadir
}

/// Achievement scalarizing function (ASF):
/// ASF(f | z*, w) = max_i { (f_i - z_i*) / w_i }
fn asf(objectives: &[f64], ideal: &[f64], weights: &[f64]) -> f64 {
    objectives
        .iter()
        .zip(ideal.iter())
        .zip(weights.iter())
        .map(|((f, z), w)| (f - z) / w)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Normalize an objective vector given the ideal and nadir points.
/// Returns the translated-and-scaled vector: (f - ideal) / (nadir - ideal).
fn normalize_objectives(objectives: &[f64], ideal: &[f64], nadir: &[f64]) -> Vec<f64> {
    objectives
        .iter()
        .zip(ideal.iter())
        .zip(nadir.iter())
        .map(|((f, z), n)| {
            let denom = n - z;
            if denom.abs() < 1e-10 {
                0.0
            } else {
                (f - z) / denom
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Reference-point association and niching
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the perpendicular distance from a normalized objective vector to a
/// reference line (direction vector from origin through reference point).
///
/// dist(f, r) = ||f - (f·r / ||r||²) * r||
///
/// Both `f_norm` and `ref_point` must have the same length.
pub fn reference_line_distance(f_norm: &[f64], ref_point: &[f64]) -> f64 {
    let dot: f64 = f_norm
        .iter()
        .zip(ref_point.iter())
        .map(|(a, b)| a * b)
        .sum();
    let r_sq: f64 = ref_point.iter().map(|r| r * r).sum();

    if r_sq < 1e-14 {
        // Reference point at origin — fall back to Euclidean distance
        return f_norm.iter().map(|x| x * x).sum::<f64>().sqrt();
    }

    let proj = dot / r_sq;

    // perpendicular distance: ||f - proj * r||
    f_norm
        .iter()
        .zip(ref_point.iter())
        .map(|(f, r)| (f - proj * r).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// For each individual, find the nearest reference point and compute the
/// distance to its reference line.
///
/// Returns a vector of `(ref_idx, distance)` tuples, one per individual.
pub fn associate_to_reference_points(
    population: &[Individual],
    ref_points: &[Vec<f64>],
    ideal: &[f64],
    nadir: &[f64],
) -> Vec<(usize, f64)> {
    population
        .iter()
        .map(|ind| {
            let f_norm = normalize_objectives(&ind.objectives, ideal, nadir);

            let mut best_ref = 0usize;
            let mut best_dist = f64::INFINITY;

            for (r_idx, rp) in ref_points.iter().enumerate() {
                let d = reference_line_distance(&f_norm, rp);
                if d < best_dist {
                    best_dist = d;
                    best_ref = r_idx;
                }
            }

            (best_ref, best_dist)
        })
        .collect()
}

/// NSGA-III survivor selection using reference-point-based niching.
///
/// Selects `target_size` survivors from `combined` (size ≈ 2N):
/// 1. Greedily fill with front 0, front 1, ... until adding the next front
///    would exceed `target_size`.
/// 2. From the "critical" (last partial) front, use niche preservation:
///    count how many individuals from already-selected fronts are associated
///    with each reference point, then repeatedly pick the individual from the
///    reference point with the smallest niche count (breaking ties by distance).
fn nsga3_select(
    combined: &mut Vec<Individual>,
    ref_points: &[Vec<f64>],
    target_size: usize,
    rng: &mut StdRng,
) -> Vec<Individual> {
    let obj_vecs: Vec<Vec<f64>> = combined.iter().map(|ind| ind.objectives.clone()).collect();
    let fronts = non_dominated_sort(&obj_vecs);

    // Compute ideal + nadir over the combined population
    let ideal = ideal_point(combined);
    let nadir = nadir_estimate(combined, &ideal);

    // Association: for each individual, find nearest ref point and distance
    let assoc = associate_to_reference_points(combined, ref_points, &ideal, &nadir);

    // Greedily fill complete fronts
    let mut survivors: Vec<usize> = Vec::with_capacity(target_size);
    let mut critical_front: &[usize] = &[];

    for front in &fronts {
        if survivors.len() + front.len() <= target_size {
            survivors.extend_from_slice(front);
        } else {
            critical_front = front;
            break;
        }
    }

    let remaining = target_size - survivors.len();

    if remaining == 0 || critical_front.is_empty() {
        // Selection complete without niching
        return survivors.iter().map(|&i| combined[i].clone()).collect();
    }

    // Niche counting: count how many survivors are associated with each ref point
    let n_ref = ref_points.len();
    let mut niche_count = vec![0usize; n_ref];
    for &s in &survivors {
        let (ref_idx, _) = assoc[s];
        if ref_idx < niche_count.len() {
            niche_count[ref_idx] += 1;
        }
    }

    // Add `remaining` individuals from `critical_front` using niche preservation
    let mut available: Vec<usize> = critical_front.to_vec();
    let mut selected_from_critical: Vec<usize> = Vec::with_capacity(remaining);

    for _ in 0..remaining {
        if available.is_empty() {
            break;
        }

        // Find the minimum niche count among reference points that have candidates
        let min_niche = available
            .iter()
            .filter_map(|&idx| {
                let (ref_idx, _) = assoc[idx];
                Some(niche_count[ref_idx])
            })
            .min()
            .unwrap_or(0);

        // Collect all candidates associated with reference points at min_niche
        let candidates: Vec<usize> = available
            .iter()
            .copied()
            .filter(|&idx| {
                let (ref_idx, _) = assoc[idx];
                niche_count[ref_idx] == min_niche
            })
            .collect();

        if candidates.is_empty() {
            break;
        }

        // Among tied candidates, if niche_count == 0, pick the closest to the
        // reference line; otherwise pick a random one (uniform random selection)
        let chosen = if min_niche == 0 {
            // Pick the candidate with the smallest distance to its reference line
            *candidates
                .iter()
                .min_by(|&&a, &&b| {
                    let da = assoc[a].1;
                    let db = assoc[b].1;
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(&candidates[0])
        } else {
            // Random selection among tied candidates
            candidates[rng.random_range(0..candidates.len())]
        };

        // Update niche count for the chosen individual's reference point
        let (chosen_ref, _) = assoc[chosen];
        if chosen_ref < niche_count.len() {
            niche_count[chosen_ref] += 1;
        }

        selected_from_critical.push(chosen);

        // Remove chosen from available
        available.retain(|&x| x != chosen);
    }

    // Build final survivor list
    survivors.extend(selected_from_critical);
    survivors.iter().map(|&i| combined[i].clone()).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Rank assignment (without crowding distance — NSGA-III uses reference points)
// ─────────────────────────────────────────────────────────────────────────────

fn assign_ranks(population: &mut Vec<Individual>) {
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
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tournament selection (rank only, for NSGA-III)
// ─────────────────────────────────────────────────────────────────────────────

fn tournament_select_by_rank(population: &[Individual], rng: &mut StdRng) -> usize {
    let n = population.len();
    let a = rng.random_range(0..n);
    let mut b = rng.random_range(0..n);
    if b == a && n > 1 {
        b = (a + 1) % n;
    }

    if population[a].rank <= population[b].rank {
        a
    } else {
        b
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Genetic operators (SBX + polynomial mutation, same as NSGA-II)
// ─────────────────────────────────────────────────────────────────────────────

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
            continue;
        }

        let (lo, hi) = bounds[i];
        let x1 = parent1[i].min(parent2[i]);
        let x2 = parent1[i].max(parent2[i]);

        if (x2 - x1).abs() < 1e-14 {
            continue;
        }

        let u: f64 = rng.random();

        let beta_q = if u <= 0.5 {
            let alpha = 2.0 - (1.0 / sbx_beta(x1, x2, lo, eta_c)).powf(eta_c + 1.0);
            let alpha = alpha.max(0.0);
            (2.0 * u * alpha).powf(1.0 / (eta_c + 1.0))
        } else {
            let alpha = 2.0 - (1.0 / sbx_beta(x1, x2, hi - x2 + x1, eta_c)).powf(eta_c + 1.0);
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

fn sbx_beta(x1: f64, x2: f64, bound: f64, eta: f64) -> f64 {
    let diff = (x2 - x1).abs().max(1e-14);
    let dist = (bound - x1).abs().max(1e-14);
    (1.0 + 2.0 * dist / diff).powf(eta + 1.0)
}

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
// Random initialisation
// ─────────────────────────────────────────────────────────────────────────────

fn random_genes(bounds: &[(f64, f64)], rng: &mut StdRng) -> Vec<f64> {
    bounds
        .iter()
        .map(|&(lo, hi)| lo + rng.random::<f64>() * (hi - lo))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Adaptive reference points
// ─────────────────────────────────────────────────────────────────────────────

/// Adaptively update reference points based on the current Pareto front.
///
/// This implements the adaptive reference point mechanism from the A-NSGA-III
/// variant: after observing the current approximation, reference points that
/// have no associated solutions are moved toward the centroid of the front.
///
/// # Arguments
/// * `ref_points`    - Current reference points (modified in place).
/// * `pareto_front`  - Current Pareto front solutions' normalized objectives.
/// * `learning_rate` - Step size for reference point update (typically 0.1).
pub fn adapt_reference_points(
    ref_points: &mut Vec<Vec<f64>>,
    pareto_front_norm: &[Vec<f64>],
    learning_rate: f64,
) {
    if pareto_front_norm.is_empty() || ref_points.is_empty() {
        return;
    }

    let n_obj = ref_points[0].len();

    // Compute centroid of the normalized Pareto front
    let mut centroid = vec![0.0f64; n_obj];
    for pt in pareto_front_norm {
        for (k, &v) in pt.iter().enumerate() {
            if k < n_obj {
                centroid[k] += v;
            }
        }
    }
    let n = pareto_front_norm.len() as f64;
    for c in &mut centroid {
        *c /= n;
    }

    // For each reference point, check if it has any associated solution
    for rp in ref_points.iter_mut() {
        let has_association = pareto_front_norm.iter().any(|pt| {
            let d = reference_line_distance(pt, rp);
            d < 0.1 // threshold for "close enough"
        });

        if !has_association {
            // Move reference point toward centroid
            for k in 0..n_obj {
                rp[k] += learning_rate * (centroid[k] - rp[k]);
            }

            // Re-normalise to unit simplex: project back
            let sum: f64 = rp.iter().sum();
            if sum > 1e-10 {
                for x in rp.iter_mut() {
                    *x /= sum;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // DTLZ2 benchmark: Pareto front is on sphere surface in M-dimensional space
    fn dtlz2(x: &[f64], n_obj: usize) -> Vec<f64> {
        let n = x.len();
        let k = n - n_obj + 1;
        let g: f64 = x[n - k..].iter().map(|&xi| (xi - 0.5).powi(2)).sum();

        let mut f = vec![0.0f64; n_obj];
        for i in 0..n_obj {
            let mut val = 1.0 + g;
            for j in 0..n_obj - 1 - i {
                val *= (x[j] * std::f64::consts::FRAC_PI_2).cos();
            }
            if i > 0 {
                val *= (x[n_obj - 1 - i] * std::f64::consts::FRAC_PI_2).sin();
            }
            f[i] = val;
        }
        f
    }

    // ── Reference point generation ───────────────────────────────────────────

    #[test]
    fn test_reference_points_sum_to_one() {
        let pts = generate_reference_points(3, 4);
        for p in &pts {
            let s: f64 = p.iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "sum = {s}");
            assert_eq!(p.len(), 3);
        }
    }

    #[test]
    fn test_reference_points_count() {
        // C(H + M - 1, M - 1) reference points
        // For M=3, H=4: C(6, 2) = 15
        let pts = generate_reference_points(3, 4);
        assert_eq!(pts.len(), 15, "Expected 15 reference points for M=3, H=4");

        // For M=2, H=5: C(6, 1) = 6
        let pts2 = generate_reference_points(2, 5);
        assert_eq!(pts2.len(), 6);
    }

    #[test]
    fn test_reference_points_non_negative() {
        let pts = generate_reference_points(4, 3);
        for p in &pts {
            for &v in p {
                assert!(v >= 0.0, "Reference point component {v} is negative");
            }
        }
    }

    #[test]
    fn test_inner_reference_points_inside_simplex() {
        let pts = generate_reference_points_inner(3, 3);
        for p in &pts {
            let s: f64 = p.iter().sum();
            assert!((s - 1.0).abs() < 0.01, "inner sum = {s}");
            for &v in p {
                assert!(v >= 0.0, "negative inner component {v}");
            }
        }
    }

    // ── reference_line_distance ──────────────────────────────────────────────

    #[test]
    fn test_ref_line_distance_on_line() {
        // A point along the reference direction should have distance 0
        let f_norm = vec![0.5, 0.5];
        let ref_pt = vec![1.0, 1.0]; // direction (1,1)
        let d = reference_line_distance(&f_norm, &ref_pt);
        assert!(d < 1e-10, "point on ref line should have d≈0, got {d}");
    }

    #[test]
    fn test_ref_line_distance_perpendicular() {
        // (1, 0) has distance 1/sqrt(2) from direction (1, 1)/sqrt(2)
        let f_norm = vec![1.0, 0.0];
        let ref_pt = vec![1.0, 1.0];
        let d = reference_line_distance(&f_norm, &ref_pt);
        let expected = (0.5f64).sqrt();
        assert!((d - expected).abs() < 1e-10, "expected {expected}, got {d}");
    }

    // ── associate_to_reference_points ─────────────────────────────────────────

    #[test]
    fn test_association_nearest_ref() {
        // Two reference points: (1,0) and (0,1); individual at (0.9, 0.1) → ref 0
        let ref_points = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let pop = vec![Individual::new(vec![0.0], vec![0.9, 0.1])];
        let ideal = vec![0.0, 0.0];
        let nadir = vec![1.0, 1.0];

        let assoc = associate_to_reference_points(&pop, &ref_points, &ideal, &nadir);
        assert_eq!(assoc.len(), 1);
        assert_eq!(assoc[0].0, 0, "Should be associated with reference point 0");
    }

    // ── nsga3 on DTLZ2 ───────────────────────────────────────────────────────

    #[test]
    fn test_nsga3_returns_pareto_front() {
        let n_obj = 3;
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); n_obj + 3];
        let mut cfg = Nsga3Config::default();
        cfg.population_size = 20;
        cfg.n_generations = 10;
        cfg.n_divisions = 3;

        let result = nsga3(n_obj, &bounds, |x| dtlz2(x, n_obj), cfg).expect("nsga3 should succeed");

        assert!(!result.pareto_front.is_empty());
        assert!(!result.reference_points.is_empty());
    }

    #[test]
    fn test_nsga3_pareto_front_non_dominated() {
        let n_obj = 3;
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); n_obj + 3];
        let mut cfg = Nsga3Config::default();
        cfg.population_size = 20;
        cfg.n_generations = 15;
        cfg.n_divisions = 3;
        cfg.seed = 77;

        let result =
            nsga3(n_obj, &bounds, |x| dtlz2(x, n_obj), cfg).expect("failed to create result");
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
    fn test_nsga3_four_objectives() {
        // Many-objective: 4 objectives where NSGA-II degrades
        let n_obj = 4;
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); n_obj + 3];
        let mut cfg = Nsga3Config::default();
        cfg.population_size = 30;
        cfg.n_generations = 10;
        cfg.n_divisions = 3;

        let result =
            nsga3(n_obj, &bounds, |x| dtlz2(x, n_obj), cfg).expect("failed to create result");
        assert!(!result.pareto_front.is_empty());
    }

    #[test]
    fn test_nsga3_two_layer_reference_points() {
        let n_obj = 3;
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); n_obj + 2];
        let mut cfg = Nsga3Config::default();
        cfg.population_size = 30;
        cfg.n_generations = 10;
        cfg.n_divisions = 3;
        cfg.n_divisions_inner = Some(2);

        let result =
            nsga3(n_obj, &bounds, |x| dtlz2(x, n_obj), cfg).expect("failed to create result");
        // Two-layer should have more reference points
        assert!(result.reference_points.len() > 10);
        assert!(!result.pareto_front.is_empty());
    }

    #[test]
    fn test_nsga3_bounds_respected() {
        let bounds = vec![(0.2, 0.8); 4];
        let mut cfg = Nsga3Config::default();
        cfg.population_size = 20;
        cfg.n_generations = 10;
        cfg.n_divisions = 3;

        let result =
            nsga3(3, &bounds, |x| vec![x[0], x[1], x[2]], cfg).expect("failed to create result");

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
    fn test_nsga3_invalid_input() {
        // Empty bounds
        let result = nsga3(3, &[], |x| vec![x[0]], Nsga3Config::default());
        assert!(result.is_err());

        // Bad bound
        let result = nsga3(3, &[(1.0, 0.0)], |x| vec![x[0]], Nsga3Config::default());
        assert!(result.is_err());

        // Too few objectives
        let result = nsga3(1, &[(0.0, 1.0)], |x| vec![x[0]], Nsga3Config::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_nsga3_reference_point_coverage() {
        let ref_pts = generate_reference_points(3, 4);
        // All reference points should be on the unit simplex
        for p in &ref_pts {
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
            for &v in p {
                assert!(v >= 0.0 && v <= 1.0);
            }
        }
    }

    #[test]
    fn test_adapt_reference_points() {
        let mut ref_pts = generate_reference_points(3, 3);
        let initial_count = ref_pts.len();

        // Simulate a Pareto front concentrated in one corner
        let fake_front: Vec<Vec<f64>> = vec![vec![0.9, 0.05, 0.05], vec![0.85, 0.1, 0.05]];

        adapt_reference_points(&mut ref_pts, &fake_front, 0.1);

        // Count should be unchanged
        assert_eq!(ref_pts.len(), initial_count);

        // Reference points should still approximately sum to 1
        for p in &ref_pts {
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "adapted ref point sum = {sum}");
        }
    }
}
