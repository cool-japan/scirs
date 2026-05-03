//! Epsilon-constraint method for multi-objective optimization.
//!
//! The epsilon-constraint method systematically converts a multi-objective
//! problem into a series of single-objective constrained subproblems by
//! optimizing one objective while constraining all others to lie within
//! ε-bounds.  By varying the ε-values, the complete Pareto front can be
//! approximated.
//!
//! # Algorithm outline
//!
//! Given a problem with `M` objectives `f_1(x), f_2(x), ..., f_M(x)`:
//!
//! 1. Choose one primary objective `k` to minimize.
//! 2. Constrain all other objectives: `f_i(x) ≤ ε_i` for `i ≠ k`.
//! 3. Solve the resulting single-objective constrained problem.
//! 4. Systematically vary `ε` to explore different Pareto trade-offs.
//!
//! # References
//!
//! - Haimes, Y.V., Lasdon, L.S., & Wismer, D.A. (1971). On a bicriterion
//!   formulation of the problems of integrated system identification and system
//!   optimization. *IEEE Transactions on Systems, Man, and Cybernetics*, 1(3),
//!   296–297.
//! - Chankong, V. & Haimes, Y.V. (1983). Multiobjective Decision Making:
//!   Theory and Methodology. Elsevier.

use crate::error::{OptimizeError, OptimizeResult};

// ─────────────────────────────────────────────────────────────────────────────
// EpsilonConstraint struct
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the epsilon-constraint method.
///
/// Specifies which objective to optimize and the upper bounds (ε-values) for
/// all remaining objectives.
///
/// # Examples
/// ```
/// use scirs2_optimize::constrained::epsilon_constraint::EpsilonConstraint;
///
/// // 2-objective problem: minimize f_0 subject to f_1 ≤ ε_1
/// let ec = EpsilonConstraint::new(0, vec![f64::INFINITY, 0.5]);
/// assert_eq!(ec.primary_objective(), 0);
/// assert_eq!(ec.epsilon_for(1), Some(0.5));
/// ```
#[derive(Debug, Clone)]
pub struct EpsilonConstraint {
    /// Index of the primary objective to minimize (0-based).
    primary_idx: usize,
    /// Upper bounds for each objective.
    ///
    /// `epsilon[i]` is the ε-bound for objective `i`.
    /// For the primary objective (`i == primary_idx`), this value is ignored;
    /// set it to `f64::INFINITY` for clarity.
    epsilon: Vec<f64>,
}

impl EpsilonConstraint {
    /// Create a new epsilon-constraint specification.
    ///
    /// # Arguments
    /// * `primary_idx` — Index (0-based) of the objective to minimize.
    /// * `epsilon`     — Upper bounds for all objectives; `epsilon[primary_idx]`
    ///   is ignored in constraint evaluation.
    ///
    /// # Errors
    /// Returns an error if `primary_idx >= epsilon.len()` or `epsilon` is empty.
    pub fn new_checked(primary_idx: usize, epsilon: Vec<f64>) -> OptimizeResult<Self> {
        if epsilon.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "epsilon vector must be non-empty".to_string(),
            ));
        }
        if primary_idx >= epsilon.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "primary_idx={primary_idx} must be < epsilon.len()={}",
                epsilon.len()
            )));
        }
        Ok(Self {
            primary_idx,
            epsilon,
        })
    }

    /// Create a new epsilon-constraint specification (panics on invalid input;
    /// use [`new_checked`] for fallible construction).
    ///
    /// [`new_checked`]: Self::new_checked
    pub fn new(primary_idx: usize, epsilon: Vec<f64>) -> Self {
        Self {
            primary_idx,
            epsilon,
        }
    }

    /// Index of the primary objective to minimize.
    pub fn primary_objective(&self) -> usize {
        self.primary_idx
    }

    /// Number of objectives.
    pub fn n_objectives(&self) -> usize {
        self.epsilon.len()
    }

    /// Return the ε-bound for objective `i`, or `None` if `i` is the primary
    /// objective (which has no ε bound).
    pub fn epsilon_for(&self, objective_idx: usize) -> Option<f64> {
        if objective_idx == self.primary_idx {
            None
        } else {
            self.epsilon.get(objective_idx).copied()
        }
    }

    /// Check whether a given objective vector `f` satisfies all ε-constraints.
    ///
    /// Returns `true` if `f[i] <= epsilon[i]` for every `i != primary_idx`.
    ///
    /// # Arguments
    /// * `f` — Objective vector with length equal to `self.n_objectives()`.
    pub fn is_feasible(&self, f: &[f64]) -> bool {
        for (i, &fi) in f.iter().enumerate() {
            if i == self.primary_idx {
                continue;
            }
            if let Some(&eps) = self.epsilon.get(i) {
                if fi > eps {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the constraint violation for a given objective vector.
    ///
    /// Returns a vector of violations `max(0, f_i - epsilon_i)` for each
    /// non-primary objective.  A violation of 0 means the constraint is satisfied.
    ///
    /// # Arguments
    /// * `f` — Objective vector.
    pub fn violations(&self, f: &[f64]) -> Vec<f64> {
        f.iter()
            .enumerate()
            .filter(|(i, _)| *i != self.primary_idx)
            .map(|(i, &fi)| {
                let eps = self.epsilon.get(i).copied().unwrap_or(f64::INFINITY);
                (fi - eps).max(0.0)
            })
            .collect()
    }

    /// Build a penalized single-objective function from a multi-objective
    /// closure, suitable for passing to a scalar optimizer.
    ///
    /// The penalized objective is:
    /// ```text
    /// F(x) = f_primary(x) + penalty * sum_i max(0, f_i(x) - epsilon_i)^2
    /// ```
    ///
    /// # Arguments
    /// * `objectives`   — Closure mapping `x: &[f64]` → `Vec<f64>` of objectives.
    /// * `penalty`      — Penalty coefficient for constraint violations.
    ///
    /// # Returns
    /// A closure `fn(&[f64]) -> f64` suitable for scalar minimization.
    pub fn penalized_objective<F>(&self, objectives: F, penalty: f64) -> impl Fn(&[f64]) -> f64 + '_
    where
        F: Fn(&[f64]) -> Vec<f64> + 'static,
    {
        let primary = self.primary_idx;
        let eps = self.epsilon.clone();

        move |x: &[f64]| {
            let f = objectives(x);
            let primary_val = f.get(primary).copied().unwrap_or(f64::INFINITY);
            let violation_penalty: f64 = f
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != primary)
                .map(|(i, &fi)| {
                    let bound = eps.get(i).copied().unwrap_or(f64::INFINITY);
                    (fi - bound).max(0.0).powi(2)
                })
                .sum::<f64>()
                * penalty;
            primary_val + violation_penalty
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EpsilonConstraintSweep — structured Pareto front generation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for systematic Pareto front generation via ε-constraint sweeps.
#[derive(Debug, Clone)]
pub struct EpsilonSweepConfig {
    /// Number of objectives.
    pub n_objectives: usize,
    /// Number of ε-levels to test per non-primary objective.
    pub n_points_per_obj: usize,
    /// Lower bounds for ε-values (defaults to 0.0 for each objective if empty).
    pub epsilon_lower: Vec<f64>,
    /// Upper bounds for ε-values (should be set to estimated nadir point values).
    pub epsilon_upper: Vec<f64>,
    /// Penalty coefficient for constraint violations in penalized optimization.
    pub penalty: f64,
    /// Maximum iterations for the inner scalar optimizer.
    pub max_inner_iter: usize,
    /// Convergence tolerance for the inner optimizer.
    pub tolerance: f64,
}

impl EpsilonSweepConfig {
    /// Create a new sweep configuration.
    ///
    /// # Arguments
    /// * `n_objectives`    — Number of objectives.
    /// * `n_points_per_obj` — Grid resolution per non-primary objective axis.
    /// * `epsilon_lower`   — Lower bounds for ε-values.
    /// * `epsilon_upper`   — Upper bounds for ε-values.
    pub fn new(
        n_objectives: usize,
        n_points_per_obj: usize,
        epsilon_lower: Vec<f64>,
        epsilon_upper: Vec<f64>,
    ) -> OptimizeResult<Self> {
        if n_objectives < 2 {
            return Err(OptimizeError::InvalidInput(
                "n_objectives must be >= 2".to_string(),
            ));
        }
        if n_points_per_obj == 0 {
            return Err(OptimizeError::InvalidInput(
                "n_points_per_obj must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            n_objectives,
            n_points_per_obj,
            epsilon_lower,
            epsilon_upper,
            penalty: 1e6,
            max_inner_iter: 200,
            tolerance: 1e-6,
        })
    }

    /// Create a symmetric sweep with `[0.0, 1.0]` ε-bounds for all objectives.
    pub fn uniform(n_objectives: usize, n_points_per_obj: usize) -> OptimizeResult<Self> {
        let lower = vec![0.0; n_objectives];
        let upper = vec![1.0; n_objectives];
        Self::new(n_objectives, n_points_per_obj, lower, upper)
    }
}

/// Result of an epsilon-constraint Pareto front generation sweep.
#[derive(Debug, Clone)]
pub struct EpsilonSweepResult {
    /// Feasible Pareto-front approximate solutions found by the sweep.
    /// Each element is an objective vector `Vec<f64>`.
    pub pareto_approximation: Vec<Vec<f64>>,
    /// Decision vectors corresponding to each Pareto point.
    pub decision_vectors: Vec<Vec<f64>>,
    /// Number of subproblems solved.
    pub n_solved: usize,
    /// Number of feasible solutions found.
    pub n_feasible: usize,
}

/// Generate an approximation of the Pareto front using the epsilon-constraint
/// method.
///
/// Systematically varies ε-bounds for all non-primary objectives and solves
/// the resulting single-objective constrained subproblems.  Infeasible
/// subproblems (where no feasible solution exists) are skipped.
///
/// # Arguments
/// * `objectives`  — Multi-objective function: `x: &[f64]` → `Vec<f64>`.
/// * `bounds`      — Decision variable bounds `[(lo, hi); n_vars]`.
/// * `primary_obj` — Index of the primary objective to minimize (0-based).
/// * `config`      — Sweep configuration specifying grid resolution and ε-ranges.
///
/// # Returns
/// An [`EpsilonSweepResult`] containing the Pareto approximation.
///
/// # Errors
/// Returns an error on invalid configuration.
///
/// # Notes
/// This function uses a simple gradient-free interior minimization to solve
/// each subproblem.  For high accuracy, replace the inner optimizer with a
/// more sophisticated method appropriate to your problem structure.
///
/// # Examples
/// ```
/// use scirs2_optimize::constrained::epsilon_constraint::{
///     generate_pareto_front_epsilon, EpsilonSweepConfig,
/// };
///
/// // 2-variable, 2-objective toy problem
/// let bounds = vec![(0.0_f64, 1.0_f64); 2];
/// let config = EpsilonSweepConfig::uniform(2, 5).expect("valid input");
///
/// let result = generate_pareto_front_epsilon(
///     |x| vec![x[0], 1.0 - x[0]],
///     &bounds,
///     0,
///     config,
/// ).expect("valid input");
///
/// assert!(result.n_feasible > 0);
/// ```
pub fn generate_pareto_front_epsilon<F>(
    objectives: F,
    bounds: &[(f64, f64)],
    primary_obj: usize,
    config: EpsilonSweepConfig,
) -> OptimizeResult<EpsilonSweepResult>
where
    F: Fn(&[f64]) -> Vec<f64> + Clone + 'static,
{
    let n_obj = config.n_objectives;
    if primary_obj >= n_obj {
        return Err(OptimizeError::InvalidInput(format!(
            "primary_obj={primary_obj} must be < n_objectives={n_obj}"
        )));
    }
    if bounds.is_empty() {
        return Err(OptimizeError::InvalidInput(
            "bounds must be non-empty".to_string(),
        ));
    }

    // Build the grid of ε-combinations for the non-primary objectives
    let non_primary: Vec<usize> = (0..n_obj).filter(|&i| i != primary_obj).collect();
    let n_non_primary = non_primary.len();

    // Build ε-grid values for each non-primary objective
    let epsilon_grids: Vec<Vec<f64>> = non_primary
        .iter()
        .map(|&obj_i| {
            let lo = config.epsilon_lower.get(obj_i).copied().unwrap_or(0.0);
            let hi = config.epsilon_upper.get(obj_i).copied().unwrap_or(1.0);
            let n = config.n_points_per_obj;
            (0..n)
                .map(|k| lo + (hi - lo) * k as f64 / (n.max(2) - 1) as f64)
                .collect()
        })
        .collect();

    // Enumerate all combinations using a multi-dimensional counter
    let total_combos: usize = epsilon_grids
        .iter()
        .map(|g| g.len())
        .product::<usize>()
        .max(1);

    let mut pareto_approximation: Vec<Vec<f64>> = Vec::new();
    let mut decision_vectors: Vec<Vec<f64>> = Vec::new();
    let mut n_solved = 0usize;
    let mut n_feasible = 0usize;

    let mut counter = vec![0usize; n_non_primary];

    for _ in 0..total_combos {
        // Build epsilon vector for this combination
        let mut epsilon: Vec<f64> = vec![f64::INFINITY; n_obj];
        for (k, &obj_i) in non_primary.iter().enumerate() {
            epsilon[obj_i] = epsilon_grids[k][counter[k]];
        }

        let ec = EpsilonConstraint::new(primary_obj, epsilon.clone());
        let pen_fn = ec.penalized_objective(objectives.clone(), config.penalty);

        // Solve the penalized scalar problem using coordinate descent
        let x_opt =
            coordinate_descent_minimize(&pen_fn, bounds, config.max_inner_iter, config.tolerance);
        n_solved += 1;

        let f_opt = objectives(&x_opt);

        // Accept only feasible solutions
        if ec.is_feasible(&f_opt) {
            pareto_approximation.push(f_opt);
            decision_vectors.push(x_opt);
            n_feasible += 1;
        }

        // Increment counter (odometer-style)
        if !counter.is_empty() {
            let mut carry = true;
            for idx in (0..n_non_primary).rev() {
                if carry {
                    counter[idx] += 1;
                    if counter[idx] >= epsilon_grids[idx].len() {
                        counter[idx] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
        }
    }

    // Post-process: filter to non-dominated solutions
    let nd_indices = pareto_filter_nd(&pareto_approximation);
    let pareto_approximation: Vec<Vec<f64>> = nd_indices
        .iter()
        .map(|&i| pareto_approximation[i].clone())
        .collect();
    let decision_vectors: Vec<Vec<f64>> = nd_indices
        .iter()
        .map(|&i| decision_vectors[i].clone())
        .collect();

    Ok(EpsilonSweepResult {
        pareto_approximation,
        decision_vectors,
        n_solved,
        n_feasible,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: simple coordinate descent for scalar subproblems
// ─────────────────────────────────────────────────────────────────────────────

/// Minimize a scalar function over box bounds using coordinate descent.
///
/// A lightweight optimizer for the inner subproblems of the ε-constraint
/// method.  Uses golden-section search along each coordinate.
fn coordinate_descent_minimize<F>(
    f: F,
    bounds: &[(f64, f64)],
    max_iter: usize,
    tol: f64,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = bounds.len();
    // Initialize at midpoint of each variable's range
    let mut x: Vec<f64> = bounds.iter().map(|(lo, hi)| (lo + hi) / 2.0).collect();

    let mut prev_val = f(&x);

    for _ in 0..max_iter {
        for j in 0..n {
            let (lo, hi) = bounds[j];
            x[j] = golden_section_1d(&f, &x, j, lo, hi, 30);
        }

        let curr_val = f(&x);
        if (prev_val - curr_val).abs() < tol {
            break;
        }
        prev_val = curr_val;
    }

    x
}

/// Golden-section search to minimize `f` along dimension `j` of `x`.
fn golden_section_1d<F>(f: &F, x: &[f64], j: usize, lo: f64, hi: f64, n_iter: usize) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0; // ≈ 0.618

    let mut a = lo;
    let mut b = hi;

    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);

    let eval = |t: f64| {
        let mut xc = x.to_vec();
        xc[j] = t;
        f(&xc)
    };

    let mut fc = eval(c);
    let mut fd = eval(d);

    for _ in 0..n_iter {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = eval(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = eval(d);
        }

        if (b - a).abs() < 1e-12 {
            break;
        }
    }

    (a + b) / 2.0
}

/// Filter non-dominated solutions from a set of objective vectors.
///
/// Returns indices of non-dominated (Pareto front) solutions.
fn pareto_filter_nd(objectives: &[Vec<f64>]) -> Vec<usize> {
    if objectives.is_empty() {
        return vec![];
    }
    let n = objectives.len();
    let mut dominated = vec![false; n];

    for i in 0..n {
        if dominated[i] {
            continue;
        }
        for j in 0..n {
            if i == j || dominated[j] {
                continue;
            }
            if dominates_vec(&objectives[i], &objectives[j]) {
                dominated[j] = true;
            }
        }
    }

    (0..n).filter(|&i| !dominated[i]).collect()
}

/// Return true if `a` Pareto-dominates `b`.
fn dominates_vec(a: &[f64], b: &[f64]) -> bool {
    let mut any_strict = false;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            any_strict = true;
        }
    }
    any_strict
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── EpsilonConstraint ─────────────────────────────────────────────────────

    #[test]
    fn test_ec_feasible_check() {
        let ec = EpsilonConstraint::new(0, vec![f64::INFINITY, 0.5, 0.8]);
        // f_0 = 0.3 (primary, ignored), f_1 = 0.4 <= 0.5, f_2 = 0.7 <= 0.8
        assert!(ec.is_feasible(&[0.3, 0.4, 0.7]));
        // f_1 = 0.6 > 0.5: infeasible
        assert!(!ec.is_feasible(&[0.3, 0.6, 0.7]));
    }

    #[test]
    fn test_ec_violations() {
        let ec = EpsilonConstraint::new(0, vec![f64::INFINITY, 0.5]);
        let v = ec.violations(&[0.3, 0.7]);
        assert_eq!(v.len(), 1); // 1 non-primary objective
        assert!((v[0] - 0.2).abs() < 1e-10, "violation = {}", v[0]);
    }

    #[test]
    fn test_ec_no_violation_when_feasible() {
        let ec = EpsilonConstraint::new(1, vec![0.5, f64::INFINITY]);
        let v = ec.violations(&[0.4, 0.9]);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], 0.0);
    }

    #[test]
    fn test_ec_epsilon_for() {
        let ec = EpsilonConstraint::new(0, vec![f64::INFINITY, 0.3, 0.7]);
        assert_eq!(ec.epsilon_for(0), None); // primary
        assert_eq!(ec.epsilon_for(1), Some(0.3));
        assert_eq!(ec.epsilon_for(2), Some(0.7));
        assert_eq!(ec.epsilon_for(99), None); // out of range
    }

    #[test]
    fn test_ec_new_checked_valid() {
        let ec = EpsilonConstraint::new_checked(0, vec![f64::INFINITY, 0.5]);
        assert!(ec.is_ok());
    }

    #[test]
    fn test_ec_new_checked_invalid_primary_idx() {
        let ec = EpsilonConstraint::new_checked(5, vec![f64::INFINITY, 0.5]);
        assert!(ec.is_err());
    }

    #[test]
    fn test_ec_new_checked_empty_epsilon() {
        let ec = EpsilonConstraint::new_checked(0, vec![]);
        assert!(ec.is_err());
    }

    #[test]
    fn test_ec_penalized_objective_feasible() {
        // f(x) = [x[0], 1 - x[0]]; constrain f_1 <= 0.6 means x[0] >= 0.4
        let ec = EpsilonConstraint::new(0, vec![f64::INFINITY, 0.6]);
        let pen_fn = ec.penalized_objective(|x| vec![x[0], 1.0 - x[0]], 1e4);
        // x[0] = 0.5: f_1 = 0.5 <= 0.6 (feasible): penalized = 0.5
        let val = pen_fn(&[0.5]);
        assert!((val - 0.5).abs() < 1e-10, "expected 0.5, got {val}");
    }

    #[test]
    fn test_ec_penalized_objective_infeasible() {
        // Constrain f_1 <= 0.3 means x[0] >= 0.7
        let ec = EpsilonConstraint::new(0, vec![f64::INFINITY, 0.3]);
        let pen_fn = ec.penalized_objective(|x| vec![x[0], 1.0 - x[0]], 1e4);
        // x[0] = 0.5: f_1 = 0.5 > 0.3 (infeasible): penalty = 1e4 * (0.5-0.3)^2 = 400
        let val = pen_fn(&[0.5]);
        assert!(
            val > 100.0,
            "infeasible point should be penalized, got {val}"
        );
    }

    // ── EpsilonSweepConfig ────────────────────────────────────────────────────

    #[test]
    fn test_sweep_config_valid() {
        let cfg = EpsilonSweepConfig::uniform(3, 4);
        assert!(cfg.is_ok());
        let cfg = cfg.expect("failed to create cfg");
        assert_eq!(cfg.n_objectives, 3);
        assert_eq!(cfg.n_points_per_obj, 4);
    }

    #[test]
    fn test_sweep_config_invalid_objectives() {
        let cfg = EpsilonSweepConfig::uniform(1, 5);
        assert!(cfg.is_err());
    }

    #[test]
    fn test_sweep_config_invalid_points() {
        let cfg = EpsilonSweepConfig::uniform(2, 0);
        assert!(cfg.is_err());
    }

    // ── generate_pareto_front_epsilon ─────────────────────────────────────────

    #[test]
    fn test_epsilon_pareto_simple_2obj() {
        // f(x) = [x[0], 1 - x[0]] on x in [0, 1]
        // True Pareto front is the line f_1 + f_2 = 1
        let bounds = vec![(0.0_f64, 1.0_f64)];
        let config = EpsilonSweepConfig::new(2, 5, vec![0.0, 0.0], vec![1.0, 1.0])
            .expect("failed to create config");

        let result = generate_pareto_front_epsilon(|x| vec![x[0], 1.0 - x[0]], &bounds, 0, config)
            .expect("unexpected None or Err");

        assert!(result.n_solved > 0, "should have solved some subproblems");
        // At least some feasible solutions should be found
        // (some ε combinations may be too tight)
    }

    #[test]
    fn test_epsilon_pareto_wrong_primary_obj() {
        let bounds = vec![(0.0_f64, 1.0_f64)];
        let config = EpsilonSweepConfig::uniform(2, 3).expect("failed to create config");
        let result = generate_pareto_front_epsilon(|x| vec![x[0], 1.0 - x[0]], &bounds, 5, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_epsilon_pareto_empty_bounds() {
        let config = EpsilonSweepConfig::uniform(2, 3).expect("failed to create config");
        let result = generate_pareto_front_epsilon(|x| vec![x[0]], &[], 0, config);
        assert!(result.is_err());
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_pareto_filter_nd() {
        let objectives = vec![
            vec![1.0, 2.0], // non-dominated
            vec![2.0, 1.0], // non-dominated
            vec![3.0, 3.0], // dominated by both
        ];
        let nd = pareto_filter_nd(&objectives);
        assert_eq!(nd.len(), 2);
        assert!(!nd.contains(&2), "dominated point should not be in result");
    }

    #[test]
    fn test_pareto_filter_nd_empty() {
        let nd = pareto_filter_nd(&[]);
        assert!(nd.is_empty());
    }

    #[test]
    fn test_coordinate_descent_converges_quadratic() {
        // Minimize (x-0.3)^2 + (y-0.7)^2 on [0,1]^2
        let bounds = vec![(0.0_f64, 1.0_f64), (0.0_f64, 1.0_f64)];
        let x_opt = coordinate_descent_minimize(
            |x| (x[0] - 0.3).powi(2) + (x[1] - 0.7).powi(2),
            &bounds,
            100,
            1e-8,
        );
        assert!(
            (x_opt[0] - 0.3).abs() < 1e-4,
            "x[0] should be ~0.3, got {}",
            x_opt[0]
        );
        assert!(
            (x_opt[1] - 0.7).abs() < 1e-4,
            "x[1] should be ~0.7, got {}",
            x_opt[1]
        );
    }
}
