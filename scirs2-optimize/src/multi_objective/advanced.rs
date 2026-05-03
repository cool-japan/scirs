//! Advanced multi-objective optimization utilities (clean Vec<f64>-based API).
//!
//! Provides:
//! - [`NsgaIII`] — struct-based NSGA-III many-objective optimizer
//! - [`NsgaResult`] — unified result type
//! - Free functions: [`dominates`], [`non_dominated_sort`], [`hypervolume`],
//!   [`crowding_distance`], [`weighted_sum_optimize`], [`epsilon_constraint_optimize`]
//!
//! All functions operate on `Vec<f64>` and `&[f64]` rather than ndarray types,
//! making them easy to use from pure Rust code without array wrappers.
//!
//! # Example
//!
//! ```rust
//! use scirs2_optimize::multi_objective::advanced::{NsgaIII, dominates, non_dominated_sort};
//!
//! assert!(dominates(&[1.0, 2.0], &[2.0, 3.0]));
//!
//! let sols = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0], vec![2.0, 3.0]];
//! let fronts = non_dominated_sort(&sols);
//! assert_eq!(fronts[0].len(), 3);
//! ```

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, RngExt, SeedableRng};

// ─────────────────────────────────────────────────────────────────────────────
// Core dominance utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` if objective vector `a` Pareto-dominates `b`.
///
/// `a` dominates `b` when:
/// - `a[i] <= b[i]` for **all** objectives `i`, **and**
/// - `a[j] < b[j]` for **at least one** objective `j`.
///
/// # Examples
/// ```
/// use scirs2_optimize::multi_objective::advanced::dominates;
/// assert!(dominates(&[1.0, 2.0], &[2.0, 3.0]));
/// assert!(!dominates(&[1.0, 2.0], &[1.0, 2.0])); // equal
/// assert!(!dominates(&[2.0, 1.0], &[1.0, 2.0])); // incomparable
/// ```
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    if a.len() != b.len() || a.is_empty() {
        return false;
    }
    let mut strictly_better = false;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Fast non-dominated sorting (NSGA-II style).
///
/// Returns a list of **fronts**, where each front is a `Vec<usize>` of solution
/// indices into `solutions`. Front 0 is the Pareto front.
///
/// # Examples
/// ```
/// use scirs2_optimize::multi_objective::advanced::non_dominated_sort;
/// let sols = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0], vec![4.0, 4.0]];
/// let fronts = non_dominated_sort(&sols);
/// assert_eq!(fronts[0].len(), 3);
/// assert_eq!(fronts[1].len(), 1);
/// ```
pub fn non_dominated_sort(solutions: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = solutions.len();
    let mut dom_count = vec![0usize; n];
    let mut dom_set: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(&solutions[i], &solutions[j]) {
                dom_set[i].push(j);
                dom_count[j] += 1;
            } else if dominates(&solutions[j], &solutions[i]) {
                dom_set[j].push(i);
                dom_count[i] += 1;
            }
        }
    }

    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front: Vec<usize> = (0..n).filter(|&i| dom_count[i] == 0).collect();

    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &p in &current_front {
            for &q in &dom_set[p] {
                dom_count[q] -= 1;
                if dom_count[q] == 0 {
                    next_front.push(q);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    fronts
}

// ─────────────────────────────────────────────────────────────────────────────
// Hypervolume indicator
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the hypervolume indicator.
///
/// - Exact for 1-D and 2-D fronts (O(n log n)).
/// - Monte Carlo approximation (100 000 samples) for dimensions ≥ 3.
///
/// # Examples
/// ```
/// use scirs2_optimize::multi_objective::advanced::hypervolume;
/// let front = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
/// let ref_pt = vec![3.0, 3.0];
/// let hv = hypervolume(&front, &ref_pt);
/// assert!((hv - 3.0).abs() < 1e-6);
/// ```
pub fn hypervolume(pareto_front: &[Vec<f64>], reference_point: &[f64]) -> f64 {
    if pareto_front.is_empty() {
        return 0.0;
    }
    match reference_point.len() {
        0 => 0.0,
        1 => hv_1d(pareto_front, reference_point),
        2 => hv_2d(pareto_front, reference_point),
        _ => hv_mc(pareto_front, reference_point, 100_000),
    }
}

fn hv_1d(front: &[Vec<f64>], ref_pt: &[f64]) -> f64 {
    let min_f = front
        .iter()
        .filter_map(|f| f.first().copied())
        .fold(f64::INFINITY, f64::min);
    (ref_pt[0] - min_f).max(0.0)
}

fn hv_2d(front: &[Vec<f64>], ref_pt: &[f64]) -> f64 {
    let mut pts: Vec<(f64, f64)> = front
        .iter()
        .filter(|f| f.len() >= 2 && f[0] < ref_pt[0] && f[1] < ref_pt[1])
        .map(|f| (f[0], f[1]))
        .collect();

    if pts.is_empty() {
        return 0.0;
    }

    // Sweep line: process points left-to-right (ascending x), tracking minimum y seen
    // so far.  For each point i the strip [x_i, x_{i+1}) × [min_y, ref_y) is added,
    // where x_{n} = ref_pt[0].  This correctly handles dominated points and arbitrary
    // orderings because min_y only decreases, preventing overcounting.
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let n = pts.len();
    let mut hv = 0.0f64;
    let mut min_y = ref_pt[1];

    for i in 0..n {
        let (x_i, y_i) = pts[i];
        let next_x = if i + 1 < n { pts[i + 1].0 } else { ref_pt[0] };
        if y_i < min_y {
            min_y = y_i;
        }
        hv += (next_x - x_i) * (ref_pt[1] - min_y);
    }
    hv
}

fn hv_mc(front: &[Vec<f64>], ref_pt: &[f64], n_samples: usize) -> f64 {
    let n_obj = ref_pt.len();
    let ideal: Vec<f64> = (0..n_obj)
        .map(|j| {
            front
                .iter()
                .filter_map(|f| f.get(j).copied())
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    let box_vol: f64 = (0..n_obj)
        .map(|j| (ref_pt[j] - ideal[j]).max(0.0))
        .product();
    if box_vol <= 0.0 {
        return 0.0;
    }

    let mut rng = StdRng::seed_from_u64(42);
    let count = (0..n_samples)
        .filter(|_| {
            let sample: Vec<f64> = (0..n_obj)
                .map(|j| ideal[j] + rng.random::<f64>() * (ref_pt[j] - ideal[j]))
                .collect();
            front
                .iter()
                .any(|f| f.len() >= n_obj && (0..n_obj).all(|j| f[j] <= sample[j]))
        })
        .count();

    box_vol * count as f64 / n_samples as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Crowding distance
// ─────────────────────────────────────────────────────────────────────────────

/// Compute crowding distances for a single Pareto front.
///
/// Boundary solutions receive `f64::INFINITY`.
///
/// # Examples
/// ```
/// use scirs2_optimize::multi_objective::advanced::crowding_distance;
/// let front = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
/// let d = crowding_distance(&front);
/// assert_eq!(d[0], f64::INFINITY);
/// assert_eq!(d[2], f64::INFINITY);
/// assert!(d[1].is_finite());
/// ```
pub fn crowding_distance(front: &[Vec<f64>]) -> Vec<f64> {
    let n = front.len();
    if n == 0 {
        return Vec::new();
    }
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let n_obj = front[0].len();
    let mut dist = vec![0.0f64; n];

    for obj in 0..n_obj {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            front[a][obj]
                .partial_cmp(&front[b][obj])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let f_min = front[order[0]][obj];
        let f_max = front[order[n - 1]][obj];
        let range = f_max - f_min;

        dist[order[0]] = f64::INFINITY;
        dist[order[n - 1]] = f64::INFINITY;

        if range < 1e-14 {
            continue;
        }

        for k in 1..(n - 1) {
            dist[order[k]] += (front[order[k + 1]][obj] - front[order[k - 1]][obj]) / range;
        }
    }

    dist
}

// ─────────────────────────────────────────────────────────────────────────────
// NsgaIII struct
// ─────────────────────────────────────────────────────────────────────────────

/// NSGA-III many-objective optimizer — struct-based API.
///
/// # Example
/// ```rust
/// use scirs2_optimize::multi_objective::advanced::NsgaIII;
/// let mut nsga = NsgaIII::new(2);
/// nsga.population_size = 20;
/// nsga.n_generations = 5;
/// let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 3];
/// let result = nsga.optimize(
///     |x| vec![x[0], 1.0 - x[0].sqrt()],
///     &bounds,
/// ).expect("valid input");
/// assert!(!result.pareto_front.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct NsgaIII {
    /// Number of objectives.
    pub n_objectives: usize,
    /// Population size per generation.
    pub population_size: usize,
    /// Number of generations.
    pub n_generations: usize,
    /// SBX crossover probability.
    pub crossover_prob: f64,
    /// Polynomial mutation probability.
    pub mutation_prob: f64,
    /// Random seed.
    pub seed: u64,
    /// Number of reference-point divisions per objective.
    pub n_divisions: usize,
}

impl NsgaIII {
    /// Create a new NSGA-III optimizer for the given number of objectives.
    pub fn new(n_objectives: usize) -> Self {
        Self {
            n_objectives,
            population_size: 100,
            n_generations: 100,
            crossover_prob: 0.9,
            mutation_prob: 0.01,
            seed: 42,
            n_divisions: 4,
        }
    }

    /// Run NSGA-III to approximate the Pareto front.
    pub fn optimize<F>(&self, objectives: F, bounds: &[(f64, f64)]) -> OptimizeResult<NsgaResult>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        if bounds.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "bounds must not be empty".to_string(),
            ));
        }

        let (pareto_dec, pareto_obj, n_eval) = run_nsga3_internal(
            &objectives,
            bounds,
            self.n_objectives,
            self.population_size,
            self.n_generations,
            self.crossover_prob,
            self.mutation_prob,
            self.seed,
            self.n_divisions,
        );

        Ok(NsgaResult {
            pareto_front: pareto_dec,
            objective_values: pareto_obj,
            n_evaluations: n_eval,
        })
    }
}

/// Result from [`NsgaIII::optimize`].
#[derive(Debug, Clone)]
pub struct NsgaResult {
    /// Decision variable vectors for Pareto-optimal solutions.
    pub pareto_front: Vec<Vec<f64>>,
    /// Corresponding objective values.
    pub objective_values: Vec<Vec<f64>>,
    /// Total function evaluations.
    pub n_evaluations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Weighted-sum scalarization
// ─────────────────────────────────────────────────────────────────────────────

/// Minimize a weighted sum of objectives via multi-start coordinate descent.
///
/// # Returns
/// `(x_best, f_best)` — optimal decision vector and objective values.
pub fn weighted_sum_optimize<F>(
    objectives: F,
    weights: &[f64],
    bounds: &[(f64, f64)],
    n_starts: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n_vars = bounds.len();
    let n_starts = n_starts.max(1);

    let scalar = |x: &[f64]| -> f64 {
        objectives(x)
            .iter()
            .zip(weights.iter())
            .map(|(&fi, &wi)| wi * fi)
            .sum::<f64>()
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut best_x = vec![0.0f64; n_vars];
    let mut best_val = f64::INFINITY;

    for _ in 0..n_starts {
        let mut x: Vec<f64> = bounds
            .iter()
            .map(|&(lb, ub)| lb + rng.random::<f64>() * (ub - lb))
            .collect();

        let avg_range =
            bounds.iter().map(|&(lb, ub)| (ub - lb).abs()).sum::<f64>() / n_vars.max(1) as f64;
        let mut step = 0.1 * avg_range;
        let mut fx = scalar(&x);

        loop {
            let mut improved = false;
            for j in 0..n_vars {
                let (lb, ub) = bounds[j];
                for &dir in &[1.0f64, -1.0] {
                    let xj_new = (x[j] + dir * step).clamp(lb, ub);
                    let mut x_try = x.clone();
                    x_try[j] = xj_new;
                    let f_try = scalar(&x_try);
                    if f_try < fx - 1e-12 {
                        x[j] = xj_new;
                        fx = f_try;
                        improved = true;
                    }
                }
            }
            if !improved {
                step *= 0.5;
                if step < 1e-8 {
                    break;
                }
            }
        }

        if fx < best_val {
            best_val = fx;
            best_x = x;
        }
    }

    let best_f = objectives(&best_x);
    (best_x, best_f)
}

// ─────────────────────────────────────────────────────────────────────────────
// Epsilon-constraint method
// ─────────────────────────────────────────────────────────────────────────────

/// Epsilon-constraint method for multi-objective optimization.
///
/// Minimises the primary objective subject to bounds on all other objectives.
///
/// # Returns
/// `(x_best, f_best)` — optimal decision vector and objective values.
pub fn epsilon_constraint_optimize<F>(
    objectives: F,
    primary_idx: usize,
    epsilon_bounds: &[(f64, f64)],
    bounds: &[(f64, f64)],
    n_starts: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n_vars = bounds.len();
    let n_starts = n_starts.max(1);
    let penalty_rho = 1e4;

    let penalized = |x: &[f64]| -> f64 {
        let f = objectives(x);
        let primary = f.get(primary_idx).copied().unwrap_or(f64::INFINITY);
        let mut viol = 0.0f64;
        let mut eps_idx = 0usize;
        for (i, &fi) in f.iter().enumerate() {
            if i == primary_idx {
                continue;
            }
            if let Some(&(lb, ub)) = epsilon_bounds.get(eps_idx) {
                viol += (lb - fi).max(0.0) + (fi - ub).max(0.0);
            }
            eps_idx += 1;
        }
        primary + penalty_rho * viol
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut best_x = vec![0.0f64; n_vars];
    let mut best_val = f64::INFINITY;

    for _ in 0..n_starts {
        let mut x: Vec<f64> = bounds
            .iter()
            .map(|&(lb, ub)| lb + rng.random::<f64>() * (ub - lb))
            .collect();

        let avg_range =
            bounds.iter().map(|&(lb, ub)| (ub - lb).abs()).sum::<f64>() / n_vars.max(1) as f64;
        let mut step = 0.1 * avg_range;
        let mut fx = penalized(&x);

        loop {
            let mut improved = false;
            for j in 0..n_vars {
                let (lb, ub) = bounds[j];
                for &dir in &[1.0f64, -1.0] {
                    let xj_new = (x[j] + dir * step).clamp(lb, ub);
                    let mut x_try = x.clone();
                    x_try[j] = xj_new;
                    let f_try = penalized(&x_try);
                    if f_try < fx - 1e-12 {
                        x[j] = xj_new;
                        fx = f_try;
                        improved = true;
                    }
                }
            }
            if !improved {
                step *= 0.5;
                if step < 1e-8 {
                    break;
                }
            }
        }

        if fx < best_val {
            best_val = fx;
            best_x = x;
        }
    }

    let best_f = objectives(&best_x);
    (best_x, best_f)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal NSGA-III evolutionary engine
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_nsga3_internal<F>(
    objectives: &F,
    bounds: &[(f64, f64)],
    n_obj: usize,
    pop_size: usize,
    n_gen: usize,
    cx_prob: f64,
    mut_prob: f64,
    seed: u64,
    n_div: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, usize)
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n_vars = bounds.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut n_eval = 0usize;

    let ref_pts = gen_reference_points(n_obj, n_div);

    let mut pop: Vec<Vec<f64>> = (0..pop_size)
        .map(|_| {
            bounds
                .iter()
                .map(|&(lb, ub)| lb + rng.random::<f64>() * (ub - lb))
                .collect()
        })
        .collect();

    let mut obj_vals: Vec<Vec<f64>> = pop
        .iter()
        .map(|x| {
            n_eval += 1;
            objectives(x)
        })
        .collect();

    for _gen in 0..n_gen {
        let mut offspring: Vec<Vec<f64>> = Vec::with_capacity(pop_size);

        while offspring.len() < pop_size {
            let p1 = rng.random_range(0..pop_size);
            let p2 = rng.random_range(0..pop_size);

            let (c1, c2) = if rng.random::<f64>() < cx_prob {
                sbx_crossover(&pop[p1], &pop[p2], bounds, &mut rng, 20.0)
            } else {
                (pop[p1].clone(), pop[p2].clone())
            };

            let c1 = poly_mutation(c1, bounds, &mut rng, mut_prob, 20.0);
            let c2 = poly_mutation(c2, bounds, &mut rng, mut_prob, 20.0);

            offspring.push(c1);
            if offspring.len() < pop_size {
                offspring.push(c2);
            }
        }

        let offspring_obj: Vec<Vec<f64>> = offspring
            .iter()
            .map(|x| {
                n_eval += 1;
                objectives(x)
            })
            .collect();

        let combined: Vec<Vec<f64>> = pop.iter().chain(offspring.iter()).cloned().collect();
        let combined_obj: Vec<Vec<f64>> = obj_vals
            .iter()
            .chain(offspring_obj.iter())
            .cloned()
            .collect();

        let fronts = non_dominated_sort(&combined_obj);

        let mut new_pop: Vec<Vec<f64>> = Vec::with_capacity(pop_size);
        let mut new_obj: Vec<Vec<f64>> = Vec::with_capacity(pop_size);
        let mut critical_indices: Vec<usize> = Vec::new();

        for front in &fronts {
            if new_pop.len() + front.len() <= pop_size {
                for &idx in front {
                    new_pop.push(combined[idx].clone());
                    new_obj.push(combined_obj[idx].clone());
                }
            } else {
                critical_indices = front.clone();
                break;
            }
        }

        let remaining = pop_size.saturating_sub(new_pop.len());
        if remaining > 0 && !critical_indices.is_empty() {
            let selected = niche_select(
                &critical_indices,
                &combined_obj,
                &new_obj,
                &ref_pts,
                remaining,
                n_obj,
                &mut rng,
            );
            for idx in selected {
                new_pop.push(combined[idx].clone());
                new_obj.push(combined_obj[idx].clone());
            }
        }

        pop = new_pop;
        obj_vals = new_obj;
    }

    let fronts = non_dominated_sort(&obj_vals);
    let pareto_idx = fronts.into_iter().next().unwrap_or_default();
    let dec: Vec<Vec<f64>> = pareto_idx.iter().map(|&i| pop[i].clone()).collect();
    let obj: Vec<Vec<f64>> = pareto_idx.iter().map(|&i| obj_vals[i].clone()).collect();

    (dec, obj, n_eval)
}

fn gen_reference_points(n_obj: usize, n_div: usize) -> Vec<Vec<f64>> {
    let mut points = Vec::new();
    let mut cur = vec![0usize; n_obj];
    gen_ref_rec(n_obj, n_div, 0, n_div, &mut cur, &mut points);
    points
}

fn gen_ref_rec(
    n_obj: usize,
    n_div: usize,
    idx: usize,
    rem: usize,
    cur: &mut Vec<usize>,
    points: &mut Vec<Vec<f64>>,
) {
    if idx == n_obj - 1 {
        cur[idx] = rem;
        points.push(
            cur.iter()
                .map(|&v| v as f64 / n_div.max(1) as f64)
                .collect(),
        );
        return;
    }
    for v in 0..=rem {
        cur[idx] = v;
        gen_ref_rec(n_obj, n_div, idx + 1, rem - v, cur, points);
    }
}

#[allow(clippy::too_many_arguments)]
fn niche_select(
    critical: &[usize],
    all_obj: &[Vec<f64>],
    selected_obj: &[Vec<f64>],
    ref_pts: &[Vec<f64>],
    n_select: usize,
    n_obj: usize,
    rng: &mut StdRng,
) -> Vec<usize> {
    let n_ref = ref_pts.len();
    if n_ref == 0 {
        let mut idx = critical.to_vec();
        idx.truncate(n_select);
        return idx;
    }

    let all_selected_obj: Vec<&Vec<f64>> = selected_obj
        .iter()
        .chain(critical.iter().map(|&i| &all_obj[i]))
        .collect();

    let ideal: Vec<f64> = (0..n_obj)
        .map(|j| {
            all_selected_obj
                .iter()
                .filter_map(|f| f.get(j).copied())
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let nadir: Vec<f64> = (0..n_obj)
        .map(|j| {
            all_selected_obj
                .iter()
                .filter_map(|f| f.get(j).copied())
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();

    let norm = |f: &[f64]| -> Vec<f64> {
        (0..n_obj)
            .map(|j| {
                let range = (nadir[j] - ideal[j]).max(1e-10);
                (f.get(j).copied().unwrap_or(0.0) - ideal[j]) / range
            })
            .collect()
    };

    let line_dist = |f_n: &[f64], rp: &[f64]| -> f64 {
        let rp_norm: f64 = rp.iter().map(|&r| r * r).sum::<f64>().sqrt().max(1e-12);
        let dot: f64 = f_n.iter().zip(rp).map(|(&fi, &ri)| fi * ri).sum();
        let proj = dot / rp_norm;
        let proj_pt: Vec<f64> = rp.iter().map(|&ri| proj * ri / rp_norm).collect();
        f_n.iter()
            .zip(&proj_pt)
            .map(|(&fi, &pi)| (fi - pi).powi(2))
            .sum::<f64>()
            .sqrt()
    };

    let mut niche_count = vec![0usize; n_ref];
    for f in selected_obj {
        let fn_ = norm(f);
        let best = (0..n_ref)
            .min_by(|&r1, &r2| {
                line_dist(&fn_, &ref_pts[r1])
                    .partial_cmp(&line_dist(&fn_, &ref_pts[r2]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        niche_count[best] += 1;
    }

    let info: Vec<(usize, usize, f64)> = critical
        .iter()
        .map(|&idx| {
            let fn_ = norm(&all_obj[idx]);
            let (r, d) = (0..n_ref)
                .map(|r| (r, line_dist(&fn_, &ref_pts[r])))
                .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, f64::INFINITY));
            (idx, r, d)
        })
        .collect();

    let mut remaining = info;
    let mut selected = Vec::with_capacity(n_select);

    while selected.len() < n_select && !remaining.is_empty() {
        let min_nc = remaining
            .iter()
            .map(|&(_, r, _)| niche_count[r])
            .min()
            .unwrap_or(0);
        let cands: Vec<usize> = remaining
            .iter()
            .enumerate()
            .filter(|(_, &(_, r, _))| niche_count[r] == min_nc)
            .map(|(i, _)| i)
            .collect();

        let chosen = if min_nc == 0 {
            cands[rng.random_range(0..cands.len())]
        } else {
            *cands
                .iter()
                .min_by(|&&a, &&b| {
                    remaining[a]
                        .2
                        .partial_cmp(&remaining[b].2)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(&cands[0])
        };

        let (sol, ref_idx, _) = remaining.remove(chosen);
        selected.push(sol);
        niche_count[ref_idx] += 1;
    }

    selected
}

fn sbx_crossover(
    p1: &[f64],
    p2: &[f64],
    bounds: &[(f64, f64)],
    rng: &mut StdRng,
    eta: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut c1 = p1.to_vec();
    let mut c2 = p2.to_vec();
    for i in 0..p1.len() {
        let (lb, ub) = bounds[i];
        if rng.random::<f64>() > 0.5 {
            continue;
        }
        let u: f64 = rng.random();
        let beta: f64 = if u <= 0.5 {
            (2.0_f64 * u).powf(1.0_f64 / (eta + 1.0))
        } else {
            (1.0_f64 / (2.0_f64 * (1.0 - u))).powf(1.0_f64 / (eta + 1.0))
        };
        c1[i] = (0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])).clamp(lb, ub);
        c2[i] = (0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])).clamp(lb, ub);
    }
    (c1, c2)
}

fn poly_mutation(
    mut x: Vec<f64>,
    bounds: &[(f64, f64)],
    rng: &mut StdRng,
    prob: f64,
    eta: f64,
) -> Vec<f64> {
    for i in 0..x.len() {
        if rng.random::<f64>() >= prob {
            continue;
        }
        let (lb, ub) = bounds[i];
        let range = (ub - lb).max(1e-12);
        let d1 = (x[i] - lb) / range;
        let d2 = (ub - x[i]) / range;
        let u: f64 = rng.random();
        let dq = if u <= 0.5 {
            let v = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - d1).powf(eta + 1.0);
            v.powf(1.0 / (eta + 1.0)) - 1.0
        } else {
            let v = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - d2).powf(eta + 1.0);
            1.0 - v.powf(1.0 / (eta + 1.0))
        };
        x[i] = (x[i] + dq * range).clamp(lb, ub);
    }
    x
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dominates_basic() {
        assert!(dominates(&[1.0, 2.0], &[2.0, 3.0]));
        assert!(!dominates(&[1.0, 2.0], &[1.0, 2.0]));
        assert!(!dominates(&[2.0, 1.0], &[1.0, 2.0]));
    }

    #[test]
    fn test_non_dominated_sort_basic() {
        let sols = vec![
            vec![1.0, 3.0],
            vec![2.0, 2.0],
            vec![3.0, 1.0],
            vec![4.0, 4.0],
        ];
        let fronts = non_dominated_sort(&sols);
        assert_eq!(fronts[0].len(), 3);
        assert!(fronts[1].contains(&3));
    }

    #[test]
    fn test_hypervolume_2d() {
        let front = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        let ref_pt = vec![3.0, 3.0];
        let hv = hypervolume(&front, &ref_pt);
        assert_abs_diff_eq!(hv, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hypervolume_empty() {
        assert_eq!(hypervolume(&[], &[3.0, 3.0]), 0.0);
    }

    #[test]
    fn test_crowding_distance_boundary() {
        let front = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
        let d = crowding_distance(&front);
        assert_eq!(d[0], f64::INFINITY);
        assert_eq!(d[2], f64::INFINITY);
        assert!(d[1].is_finite());
    }

    #[test]
    fn test_nsga3_basic() {
        let mut nsga = NsgaIII::new(2);
        nsga.population_size = 10;
        nsga.n_generations = 3;
        nsga.n_divisions = 2;

        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 2];
        let result = nsga
            .optimize(|x| vec![x[0], 1.0 - x[0].sqrt()], &bounds)
            .expect("failed to create result");
        assert!(!result.pareto_front.is_empty());
        assert!(result.n_evaluations > 0);
    }

    #[test]
    fn test_nsga3_empty_bounds_err() {
        let nsga = NsgaIII::new(2);
        assert!(nsga.optimize(|x| vec![x[0]], &[]).is_err());
    }

    #[test]
    fn test_weighted_sum_single() {
        let (x, f) = weighted_sum_optimize(|x| vec![x[0].powi(2)], &[1.0], &[(0.0, 2.0)], 5, 0);
        assert!(x[0] < 0.2, "x near 0, got {}", x[0]);
        assert!(f[0] < 0.05, "f near 0, got {}", f[0]);
    }

    #[test]
    fn test_epsilon_constraint_basic() {
        // min f0=x0 s.t. 0.3<=f1=x1<=0.7, x in [0,1]^2
        let (x, f) = epsilon_constraint_optimize(
            |x| vec![x[0], x[1]],
            0,
            &[(0.3, 0.7)],
            &[(0.0, 1.0), (0.0, 1.0)],
            5,
            42,
        );
        assert!(x[0] < 0.2, "x[0] near 0, got {}", x[0]);
        assert!(f[1] >= 0.2 && f[1] <= 0.8, "f[1] in bounds, got {}", f[1]);
    }

    #[test]
    fn test_hypervolume_3d_positive() {
        let front = vec![
            vec![0.1, 0.5, 0.9],
            vec![0.5, 0.1, 0.9],
            vec![0.9, 0.9, 0.1],
        ];
        let ref_pt = vec![1.0, 1.0, 1.0];
        let hv = hypervolume(&front, &ref_pt);
        assert!(hv > 0.0, "3D HV should be positive, got {}", hv);
    }
}
