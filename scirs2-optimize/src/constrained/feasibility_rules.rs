//! Feasibility-rules and constraint-handling strategies for evolutionary
//! optimization.
//!
//! Provides deterministic and stochastic rules for comparing candidate
//! solutions in the presence of constraints — without relying on explicit
//! penalty functions.
//!
//! # Strategies
//!
//! | Strategy | Description |
//! |----------|-------------|
//! | [`FeasibilityRule`]       | Deb's feasibility tournament rule (2002) |
//! | [`EpsilonFeasibility`]    | ε-constrained feasibility (Takahama & Sakai 2006) |
//! | [`StochasticRanking`]     | Stochastic ranking (Runarsson & Yao 2000) |
//! | [`AdaptiveFeasibility`]   | Adaptive feasibility pressure with generation control |
//!
//! # References
//!
//! - Deb, K. (2000). An efficient constraint handling method for genetic
//!   algorithms. *Computer Methods in Applied Mechanics and Engineering*,
//!   186(2–4), 311–338.
//! - Takahama, T. & Sakai, S. (2006). Constrained optimization by the ε
//!   constrained differential evolution with gradient-based mutation and
//!   feasible elites. *IEEE CEC 2006*.
//! - Runarsson, T.P. & Yao, X. (2000). Stochastic ranking for constrained
//!   evolutionary optimization. *IEEE Transactions on Evolutionary
//!   Computation*, 4(3), 284–294.

use crate::error::{OptimizeError, OptimizeResult};

// ─────────────────────────────────────────────────────────────────────────────
// Constraint violation summary
// ─────────────────────────────────────────────────────────────────────────────

/// Summary of constraint violations for a single candidate solution.
///
/// Violations are computed as `max(0, g_i(x))` for inequality constraints
/// (`g_i(x) <= 0`) and `|h_j(x)|` for equality constraints.
#[derive(Debug, Clone, PartialEq)]
pub struct ViolationSummary {
    /// L1 sum of constraint violations.
    pub total_violation: f64,
    /// Maximum single-constraint violation.
    pub max_violation: f64,
    /// Number of violated constraints.
    pub n_violated: usize,
    /// Per-constraint violations (non-negative).
    pub violations: Vec<f64>,
}

impl ViolationSummary {
    /// Create a `ViolationSummary` from a vector of violation amounts.
    ///
    /// # Arguments
    /// * `violations` — Non-negative violation per constraint.
    ///   For inequality `g_i(x) <= 0`: pass `max(0, g_i(x))`.
    ///   For equality `h_j(x) = 0`: pass `|h_j(x)|`.
    pub fn new(violations: Vec<f64>) -> Self {
        let total: f64 = violations.iter().sum();
        let max = violations.iter().cloned().fold(0.0_f64, f64::max);
        let n_violated = violations.iter().filter(|&&v| v > 0.0).count();
        Self {
            total_violation: total,
            max_violation: max,
            n_violated,
            violations,
        }
    }

    /// Returns `true` if the solution is strictly feasible (all violations zero).
    pub fn is_feasible(&self) -> bool {
        self.total_violation == 0.0
    }

    /// Returns `true` if the solution is approximately feasible within `tol`.
    pub fn is_approximately_feasible(&self, tol: f64) -> bool {
        self.total_violation <= tol
    }

    /// Compute the L2 (sum-of-squares) violation norm.
    pub fn l2_norm(&self) -> f64 {
        self.violations
            .iter()
            .map(|v| v.powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Deb's Feasibility Tournament Rule
// ─────────────────────────────────────────────────────────────────────────────

/// Deb's feasibility tournament rule for comparing candidate solutions.
///
/// The rule defines a total order on solutions:
/// 1. A **feasible** solution always beats an **infeasible** one.
/// 2. Among two feasible solutions, the one with the better objective wins.
/// 3. Among two infeasible solutions, the one with the smaller total
///    constraint violation wins.
///
/// # References
/// Deb, K. (2000). An efficient constraint handling method for genetic
/// algorithms. *Computer Methods in Applied Mechanics and Engineering*, 186.
#[derive(Debug, Clone, Copy, Default)]
pub struct FeasibilityRule {
    /// Feasibility tolerance: violations smaller than this are treated as zero.
    pub feasibility_tol: f64,
}

impl FeasibilityRule {
    /// Create a new feasibility rule.
    pub fn new(feasibility_tol: f64) -> Self {
        Self { feasibility_tol }
    }

    /// Compare two solutions `a` and `b` according to Deb's tournament rule.
    ///
    /// Returns `std::cmp::Ordering::Less` if `a` is preferred,
    /// `Equal` if they are equivalent, and `Greater` if `b` is preferred.
    ///
    /// # Arguments
    /// * `f_a`, `f_b`  — Objective values (minimisation).
    /// * `viol_a`, `viol_b` — Constraint violations.
    pub fn compare(
        &self,
        f_a: f64,
        viol_a: &ViolationSummary,
        f_b: f64,
        viol_b: &ViolationSummary,
    ) -> std::cmp::Ordering {
        let feasible_a = viol_a.total_violation <= self.feasibility_tol;
        let feasible_b = viol_b.total_violation <= self.feasibility_tol;

        match (feasible_a, feasible_b) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (true, true) => f_a.partial_cmp(&f_b).unwrap_or(std::cmp::Ordering::Equal),
            (false, false) => viol_a
                .total_violation
                .partial_cmp(&viol_b.total_violation)
                .unwrap_or(std::cmp::Ordering::Equal),
        }
    }

    /// Sort a list of `(objective, violation)` pairs in-place using the
    /// feasibility rule (best first).
    ///
    /// Returns the sorted indices into the original slice.
    pub fn sort_population(&self, population: &[(f64, ViolationSummary)]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..population.len()).collect();
        indices.sort_by(|&a, &b| {
            self.compare(
                population[a].0,
                &population[a].1,
                population[b].0,
                &population[b].1,
            )
        });
        indices
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ε-Constrained Feasibility (Takahama & Sakai)
// ─────────────────────────────────────────────────────────────────────────────

/// ε-constrained feasibility — a relaxed version of Deb's rule that treats
/// solutions with violation ≤ ε as "feasible" for comparison purposes.
///
/// The ε threshold is adapted over generations: it starts at a user-specified
/// `epsilon_0` and decreases to 0 following a schedule, gradually tightening
/// the feasibility criterion.
///
/// # References
/// Takahama, T. & Sakai, S. (2006). Constrained optimization by the ε
/// constrained differential evolution with gradient-based mutation and
/// feasible elites. *IEEE CEC 2006*.
#[derive(Debug, Clone)]
pub struct EpsilonFeasibility {
    /// Current ε threshold.
    epsilon: f64,
    /// Initial ε (at generation 0).
    epsilon_0: f64,
    /// Generation at which ε reaches 0.
    t_c: usize,
    /// Control parameter (typically 0.1–0.5 for CP schedule, or exponent for
    /// the exponential schedule).
    cp: f64,
    /// Schedule type for ε decay.
    schedule: EpsilonSchedule,
    /// Current generation counter.
    generation: usize,
}

/// Decay schedule for the ε threshold in [`EpsilonFeasibility`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EpsilonSchedule {
    /// Linear decay: `ε(t) = ε_0 * max(0, 1 - t/t_c)`.
    Linear,
    /// Exponential decay: `ε(t) = ε_0 * exp(-cp * t)`.
    Exponential,
    /// Power-law decay: `ε(t) = ε_0 * (1 - t/t_c)^cp`.
    PowerLaw,
}

impl EpsilonFeasibility {
    /// Create a new ε-feasibility handler.
    ///
    /// # Arguments
    /// * `epsilon_0`  — Initial ε threshold (must be ≥ 0).
    /// * `t_c`        — Generation at which ε → 0 (must be > 0).
    /// * `cp`         — Control parameter for the schedule.
    /// * `schedule`   — Decay schedule.
    pub fn new(
        epsilon_0: f64,
        t_c: usize,
        cp: f64,
        schedule: EpsilonSchedule,
    ) -> OptimizeResult<Self> {
        if epsilon_0 < 0.0 {
            return Err(OptimizeError::InvalidInput(
                "epsilon_0 must be >= 0".to_string(),
            ));
        }
        if t_c == 0 {
            return Err(OptimizeError::InvalidInput("t_c must be > 0".to_string()));
        }
        Ok(Self {
            epsilon: epsilon_0,
            epsilon_0,
            t_c,
            cp,
            schedule,
            generation: 0,
        })
    }

    /// Current ε threshold.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Advance to the next generation and update ε accordingly.
    pub fn step(&mut self) {
        self.generation += 1;
        self.epsilon = self.compute_epsilon(self.generation);
    }

    /// Compute the ε value for an arbitrary generation `t`.
    fn compute_epsilon(&self, t: usize) -> f64 {
        if t >= self.t_c {
            return 0.0;
        }
        let ratio = t as f64 / self.t_c as f64;
        match self.schedule {
            EpsilonSchedule::Linear => self.epsilon_0 * (1.0 - ratio),
            EpsilonSchedule::Exponential => self.epsilon_0 * (-self.cp * t as f64).exp(),
            EpsilonSchedule::PowerLaw => self.epsilon_0 * (1.0 - ratio).powf(self.cp),
        }
    }

    /// Compare two solutions using the ε-constrained rule.
    ///
    /// A solution with violation ≤ ε is treated as "feasible" for this
    /// comparison.  The rule is otherwise identical to [`FeasibilityRule`].
    pub fn compare(
        &self,
        f_a: f64,
        viol_a: &ViolationSummary,
        f_b: f64,
        viol_b: &ViolationSummary,
    ) -> std::cmp::Ordering {
        let eps_feasible_a = viol_a.total_violation <= self.epsilon;
        let eps_feasible_b = viol_b.total_violation <= self.epsilon;

        match (eps_feasible_a, eps_feasible_b) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (true, true) => f_a.partial_cmp(&f_b).unwrap_or(std::cmp::Ordering::Equal),
            (false, false) => viol_a
                .total_violation
                .partial_cmp(&viol_b.total_violation)
                .unwrap_or(std::cmp::Ordering::Equal),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stochastic Ranking (Runarsson & Yao 2000)
// ─────────────────────────────────────────────────────────────────────────────

/// Stochastic ranking for constrained evolutionary optimization.
///
/// A bubble-sort-based ranking that probabilistically compares solutions.
/// With probability `p_f`, feasible-feasible comparisons are based on
/// the objective; with probability `1 - p_f`, the violation is used.
/// Infeasible-infeasible comparisons always use the violation.
///
/// This balances exploration of infeasible regions (large `p_f` → more weight
/// on objectives even for infeasible solutions) against driving toward
/// feasibility (small `p_f`).  `p_f = 0.45` is a commonly recommended value.
///
/// # References
/// Runarsson, T.P. & Yao, X. (2000). Stochastic ranking for constrained
/// evolutionary optimization. *IEEE Transactions on Evolutionary Computation*,
/// 4(3), 284–294.
#[derive(Debug, Clone)]
pub struct StochasticRanking {
    /// Probability of using objective comparison between feasible-feasible pairs.
    pub p_f: f64,
    /// RNG seed for reproducibility.
    seed: u64,
}

impl StochasticRanking {
    /// Create a new stochastic ranking instance.
    ///
    /// # Arguments
    /// * `p_f`  — Probability of objective-based comparison for feasible
    ///   pairs.  Typical range: 0.3–0.5.
    /// * `seed` — RNG seed.
    pub fn new(p_f: f64, seed: u64) -> OptimizeResult<Self> {
        if !(0.0..=1.0).contains(&p_f) {
            return Err(OptimizeError::InvalidInput(
                "p_f must be in [0, 1]".to_string(),
            ));
        }
        Ok(Self { p_f, seed })
    }

    /// Rank a population by stochastic ranking (bubble-sort variant).
    ///
    /// # Arguments
    /// * `objectives`  — Objective values (one per individual).
    /// * `violations`  — Constraint violation summaries.
    /// * `n_passes`    — Number of bubble-sort passes to perform.
    ///
    /// # Returns
    /// Ranked indices (index 0 = best-ranked individual).
    pub fn rank(
        &self,
        objectives: &[f64],
        violations: &[ViolationSummary],
        n_passes: usize,
    ) -> OptimizeResult<Vec<usize>> {
        let n = objectives.len();
        if n != violations.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "objectives.len()={n} must equal violations.len()={}",
                violations.len()
            )));
        }
        if n == 0 {
            return Ok(vec![]);
        }

        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng_state = self.seed;

        let n_bubble = n_passes.max(1);

        for _ in 0..n_bubble {
            for j in 0..(n - 1) {
                let a = indices[j];
                let b = indices[j + 1];

                let fa = objectives[a];
                let fb = objectives[b];
                let va = &violations[a];
                let vb = &violations[b];

                // Pseudo-random value from LCG
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (rng_state >> 33) as f64 / (u32::MAX as f64);

                let should_swap = if va.is_feasible() && vb.is_feasible() {
                    // Both feasible: compare by objective with prob p_f
                    if u < self.p_f {
                        fa > fb // swap if a is worse
                    } else {
                        false
                    }
                } else if !va.is_feasible() && !vb.is_feasible() {
                    // Both infeasible: always compare by violation
                    va.total_violation > vb.total_violation
                } else {
                    // Mixed: feasible beats infeasible
                    !va.is_feasible() && vb.is_feasible()
                };

                if should_swap {
                    indices.swap(j, j + 1);
                }
            }
        }

        Ok(indices)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adaptive Feasibility Pressure
// ─────────────────────────────────────────────────────────────────────────────

/// Adaptive feasibility pressure that adjusts constraint-handling behavior
/// based on the ratio of feasible to infeasible solutions in the population.
///
/// When the feasibility ratio is low, the comparison rule leans toward
/// preferring solutions with smaller violations (exploration mode).
/// As feasibility improves, it transitions toward objective-based comparison
/// (exploitation mode).
///
/// This prevents premature convergence into infeasible regions while still
/// allowing useful gradient information from near-feasible solutions.
#[derive(Debug, Clone)]
pub struct AdaptiveFeasibility {
    /// Current feasibility ratio (fraction of feasible solutions in population).
    feasibility_ratio: f64,
    /// Target feasibility ratio (typically 0.5).
    target_ratio: f64,
    /// Learning rate for updating the feasibility ratio estimate.
    alpha: f64,
    /// Feasibility tolerance.
    feasibility_tol: f64,
}

impl AdaptiveFeasibility {
    /// Create a new adaptive feasibility handler.
    ///
    /// # Arguments
    /// * `target_ratio`    — Desired fraction of feasible solutions (0 < t ≤ 1).
    /// * `alpha`           — Update learning rate for the ratio estimate.
    /// * `feasibility_tol` — Tolerance below which violations are treated as zero.
    pub fn new(target_ratio: f64, alpha: f64, feasibility_tol: f64) -> OptimizeResult<Self> {
        if !(0.0..=1.0).contains(&target_ratio) {
            return Err(OptimizeError::InvalidInput(
                "target_ratio must be in (0, 1]".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(OptimizeError::InvalidInput(
                "alpha must be in [0, 1]".to_string(),
            ));
        }
        Ok(Self {
            feasibility_ratio: target_ratio,
            target_ratio,
            alpha,
            feasibility_tol,
        })
    }

    /// Update the feasibility ratio estimate from the current population.
    ///
    /// # Arguments
    /// * `violations` — Violation summaries for all population members.
    pub fn update(&mut self, violations: &[ViolationSummary]) {
        if violations.is_empty() {
            return;
        }
        let n_feasible = violations
            .iter()
            .filter(|v| v.total_violation <= self.feasibility_tol)
            .count();
        let observed_ratio = n_feasible as f64 / violations.len() as f64;
        // Exponential moving average update
        self.feasibility_ratio =
            (1.0 - self.alpha) * self.feasibility_ratio + self.alpha * observed_ratio;
    }

    /// Current estimated feasibility ratio.
    pub fn feasibility_ratio(&self) -> f64 {
        self.feasibility_ratio
    }

    /// Compute the effective penalty weight for constraint violations.
    ///
    /// When feasibility_ratio < target_ratio, penalty is amplified to drive
    /// solutions toward feasibility.  When ratio >= target_ratio, penalty
    /// is reduced to emphasize objective improvement.
    ///
    /// The weight is: `base_penalty * (target / current)^2` if below target,
    /// or `base_penalty * (current / target)^(-1)` if above.
    pub fn effective_penalty_weight(&self, base_penalty: f64) -> f64 {
        if self.feasibility_ratio <= 0.0 {
            return base_penalty * 10.0;
        }
        let ratio = self.target_ratio / self.feasibility_ratio.max(1e-10);
        if self.feasibility_ratio < self.target_ratio {
            base_penalty * ratio.powi(2)
        } else {
            // Above target: reduce penalty
            base_penalty / ratio
        }
    }

    /// Compare two solutions using the adaptive feasibility-weighted rule.
    ///
    /// The effective comparison weight between objective and violation is
    /// controlled by the current feasibility pressure.
    pub fn compare(
        &self,
        f_a: f64,
        viol_a: &ViolationSummary,
        f_b: f64,
        viol_b: &ViolationSummary,
    ) -> std::cmp::Ordering {
        // Use Deb's rule as base, but with adaptive tolerance
        let feasible_a = viol_a.total_violation <= self.feasibility_tol;
        let feasible_b = viol_b.total_violation <= self.feasibility_tol;

        match (feasible_a, feasible_b) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (true, true) => f_a.partial_cmp(&f_b).unwrap_or(std::cmp::Ordering::Equal),
            (false, false) => {
                // Weighted combination: emphasize violation when ratio is low
                let w = self.effective_penalty_weight(1.0);
                let score_a = f_a + w * viol_a.total_violation;
                let score_b = f_b + w * viol_b.total_violation;
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: violation computation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute inequality constraint violations from a function slice.
///
/// For each `g_i(x) <= 0` constraint function in `g_fns`, computes
/// `max(0, g_i(x))`.
///
/// # Arguments
/// * `x`     — Decision variable vector.
/// * `g_fns` — Slice of inequality constraint functions `g_i: &[f64] -> f64`.
///   Feasible: `g_i(x) <= 0`.
///
/// # Returns
/// [`ViolationSummary`] with per-constraint violations.
///
/// # Examples
/// ```
/// use scirs2_optimize::constrained::feasibility_rules::ineq_violations;
///
/// // Constraint: x[0] + x[1] <= 2
/// let g = |x: &[f64]| x[0] + x[1] - 2.0;
/// let x = &[1.5_f64, 1.5_f64]; // violates: 1.5+1.5-2=1.0 > 0
/// let v = ineq_violations(x, &[g]);
/// assert!((v.total_violation - 1.0).abs() < 1e-10);
/// ```
pub fn ineq_violations<F>(x: &[f64], g_fns: &[F]) -> ViolationSummary
where
    F: Fn(&[f64]) -> f64,
{
    let violations: Vec<f64> = g_fns.iter().map(|g| g(x).max(0.0)).collect();
    ViolationSummary::new(violations)
}

/// Compute equality constraint violations from a function slice.
///
/// For each `h_j(x) = 0` equality constraint, computes `|h_j(x)|`.
///
/// # Arguments
/// * `x`     — Decision variable vector.
/// * `h_fns` — Slice of equality constraint functions `h_j: &[f64] -> f64`.
///   Feasible: `h_j(x) = 0`.
///
/// # Returns
/// [`ViolationSummary`] with per-constraint violations.
pub fn eq_violations<F>(x: &[f64], h_fns: &[F]) -> ViolationSummary
where
    F: Fn(&[f64]) -> f64,
{
    let violations: Vec<f64> = h_fns.iter().map(|h| h(x).abs()).collect();
    ViolationSummary::new(violations)
}

/// Combine inequality and equality violations into a single summary.
///
/// Concatenates the violation vectors from both and recomputes the summary.
pub fn combined_violations<G, H>(x: &[f64], g_fns: &[G], h_fns: &[H]) -> ViolationSummary
where
    G: Fn(&[f64]) -> f64,
    H: Fn(&[f64]) -> f64,
{
    let mut violations: Vec<f64> = g_fns.iter().map(|g| g(x).max(0.0)).collect();
    violations.extend(h_fns.iter().map(|h| h(x).abs()));
    ViolationSummary::new(violations)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ViolationSummary ──────────────────────────────────────────────────────

    #[test]
    fn test_violation_summary_feasible() {
        let vs = ViolationSummary::new(vec![0.0, 0.0]);
        assert!(vs.is_feasible());
        assert_eq!(vs.total_violation, 0.0);
        assert_eq!(vs.n_violated, 0);
    }

    #[test]
    fn test_violation_summary_infeasible() {
        let vs = ViolationSummary::new(vec![0.3, 0.0, 0.5]);
        assert!(!vs.is_feasible());
        assert!((vs.total_violation - 0.8).abs() < 1e-10);
        assert!((vs.max_violation - 0.5).abs() < 1e-10);
        assert_eq!(vs.n_violated, 2);
    }

    #[test]
    fn test_violation_summary_l2_norm() {
        let vs = ViolationSummary::new(vec![3.0, 4.0]);
        assert!((vs.l2_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_violation_summary_approximately_feasible() {
        let vs = ViolationSummary::new(vec![0.05]);
        assert!(vs.is_approximately_feasible(0.1));
        assert!(!vs.is_approximately_feasible(0.01));
    }

    // ── FeasibilityRule ───────────────────────────────────────────────────────

    #[test]
    fn test_feasibility_rule_feasible_beats_infeasible() {
        let rule = FeasibilityRule::new(1e-8);
        let feas = ViolationSummary::new(vec![0.0]);
        let infeas = ViolationSummary::new(vec![0.5]);
        // feasible wins regardless of objective
        assert_eq!(
            rule.compare(100.0, &feas, 0.0, &infeas),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn test_feasibility_rule_both_feasible_compare_objective() {
        let rule = FeasibilityRule::new(1e-8);
        let feas = ViolationSummary::new(vec![0.0]);
        assert_eq!(
            rule.compare(1.0, &feas, 2.0, &feas),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            rule.compare(2.0, &feas, 1.0, &feas),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            rule.compare(1.5, &feas, 1.5, &feas),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn test_feasibility_rule_both_infeasible_compare_violation() {
        let rule = FeasibilityRule::new(1e-8);
        let v1 = ViolationSummary::new(vec![0.3]);
        let v2 = ViolationSummary::new(vec![0.8]);
        // v1 has smaller violation → wins (Less)
        assert_eq!(rule.compare(0.0, &v1, 0.0, &v2), std::cmp::Ordering::Less);
    }

    #[test]
    fn test_feasibility_rule_sort_population() {
        let rule = FeasibilityRule::new(1e-8);
        let pop = vec![
            (5.0, ViolationSummary::new(vec![0.0])), // feasible, bad obj
            (1.0, ViolationSummary::new(vec![1.0])), // infeasible, good obj
            (2.0, ViolationSummary::new(vec![0.0])), // feasible, medium obj
        ];
        let sorted = rule.sort_population(&pop);
        // Best first: feasible with obj=2.0 (idx 2), then feasible with obj=5.0 (idx 0), then infeasible (idx 1)
        assert_eq!(sorted[0], 2);
        assert_eq!(sorted[1], 0);
        assert_eq!(sorted[2], 1);
    }

    // ── EpsilonFeasibility ────────────────────────────────────────────────────

    #[test]
    fn test_epsilon_feasibility_linear_decay() {
        let mut ef = EpsilonFeasibility::new(1.0, 10, 1.0, EpsilonSchedule::Linear)
            .expect("failed to create ef");
        assert!((ef.epsilon() - 1.0).abs() < 1e-10);
        ef.step(); // gen 1
        assert!((ef.epsilon() - 0.9).abs() < 1e-10);
        for _ in 0..9 {
            ef.step();
        }
        assert_eq!(ef.epsilon(), 0.0);
    }

    #[test]
    fn test_epsilon_feasibility_power_law_decay() {
        let mut ef = EpsilonFeasibility::new(1.0, 100, 2.0, EpsilonSchedule::PowerLaw)
            .expect("failed to create ef");
        ef.step(); // gen 1: ε = 1 * (1 - 1/100)^2 ≈ 0.9801
        assert!(ef.epsilon() > 0.97 && ef.epsilon() < 1.0);
    }

    #[test]
    fn test_epsilon_feasibility_invalid_epsilon() {
        let result = EpsilonFeasibility::new(-1.0, 10, 1.0, EpsilonSchedule::Linear);
        assert!(result.is_err());
    }

    #[test]
    fn test_epsilon_feasibility_invalid_tc() {
        let result = EpsilonFeasibility::new(1.0, 0, 1.0, EpsilonSchedule::Linear);
        assert!(result.is_err());
    }

    #[test]
    fn test_epsilon_feasibility_compare_relaxed_feasibility() {
        let ef = EpsilonFeasibility::new(0.5, 10, 1.0, EpsilonSchedule::Linear)
            .expect("failed to create ef");
        // Both have violation <= 0.5, so both treated as "feasible" → compare by objective
        let v1 = ViolationSummary::new(vec![0.3]);
        let v2 = ViolationSummary::new(vec![0.4]);
        // f_a=1.0, f_b=2.0: a wins (Less)
        assert_eq!(ef.compare(1.0, &v1, 2.0, &v2), std::cmp::Ordering::Less);
    }

    // ── StochasticRanking ─────────────────────────────────────────────────────

    #[test]
    fn test_stochastic_ranking_correct_length() {
        let sr = StochasticRanking::new(0.45, 42).expect("failed to create sr");
        let objectives = vec![1.0, 2.0, 3.0, 4.0];
        let violations = vec![
            ViolationSummary::new(vec![0.0]),
            ViolationSummary::new(vec![0.5]),
            ViolationSummary::new(vec![0.0]),
            ViolationSummary::new(vec![0.2]),
        ];
        let ranked = sr
            .rank(&objectives, &violations, 5)
            .expect("failed to create ranked");
        assert_eq!(ranked.len(), 4);
        // All indices should be present
        let mut sorted_ranked = ranked.clone();
        sorted_ranked.sort_unstable();
        assert_eq!(sorted_ranked, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_stochastic_ranking_feasible_prefers_better_obj() {
        // With enough passes, feasible better-objective solution should rank first
        let sr = StochasticRanking::new(0.45, 42).expect("failed to create sr");
        let objectives = vec![2.0, 1.0]; // idx 1 has better objective
        let violations = vec![
            ViolationSummary::new(vec![0.0]), // feasible
            ViolationSummary::new(vec![0.0]), // feasible
        ];
        let ranked = sr
            .rank(&objectives, &violations, 50)
            .expect("failed to create ranked");
        assert_eq!(ranked[0], 1, "better objective should rank first");
    }

    #[test]
    fn test_stochastic_ranking_invalid_p_f() {
        let result = StochasticRanking::new(1.5, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_stochastic_ranking_mismatch_lengths() {
        let sr = StochasticRanking::new(0.45, 42).expect("failed to create sr");
        let result = sr.rank(&[1.0, 2.0], &[ViolationSummary::new(vec![0.0])], 5);
        assert!(result.is_err());
    }

    // ── AdaptiveFeasibility ───────────────────────────────────────────────────

    #[test]
    fn test_adaptive_feasibility_initial_ratio() {
        let af = AdaptiveFeasibility::new(0.5, 0.1, 1e-8).expect("failed to create af");
        assert!((af.feasibility_ratio() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_feasibility_update_all_feasible() {
        let mut af = AdaptiveFeasibility::new(0.5, 0.5, 1e-8).expect("failed to create af");
        let violations = vec![
            ViolationSummary::new(vec![0.0]),
            ViolationSummary::new(vec![0.0]),
        ];
        af.update(&violations);
        // After update: ratio should move toward 1.0
        assert!(af.feasibility_ratio() > 0.5);
    }

    #[test]
    fn test_adaptive_feasibility_penalty_amplified_when_low() {
        // When all solutions are infeasible, ratio → 0, penalty should increase
        let mut af = AdaptiveFeasibility::new(0.5, 0.9, 1e-8).expect("failed to create af");
        let violations: Vec<ViolationSummary> = vec![
            ViolationSummary::new(vec![1.0]),
            ViolationSummary::new(vec![2.0]),
        ];
        af.update(&violations);
        let weight = af.effective_penalty_weight(1.0);
        assert!(
            weight > 1.0,
            "penalty should be amplified when feasibility is low"
        );
    }

    #[test]
    fn test_adaptive_feasibility_compare() {
        let af = AdaptiveFeasibility::new(0.5, 0.1, 1e-8).expect("failed to create af");
        let feas = ViolationSummary::new(vec![0.0]);
        let infeas = ViolationSummary::new(vec![1.0]);
        assert_eq!(
            af.compare(100.0, &feas, 0.0, &infeas),
            std::cmp::Ordering::Less
        );
    }

    // ── violation helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_ineq_violations_satisfied() {
        let x = &[0.5_f64];
        let g = |x: &[f64]| x[0] - 1.0; // x[0] <= 1 (g <= 0)
        let v = ineq_violations(x, &[g]);
        assert_eq!(v.total_violation, 0.0);
    }

    #[test]
    fn test_ineq_violations_violated() {
        let x = &[1.5_f64];
        let g = |x: &[f64]| x[0] - 1.0; // x[0] <= 1
        let v = ineq_violations(x, &[g]);
        assert!((v.total_violation - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_eq_violations() {
        let x = &[2.0_f64];
        let h = |x: &[f64]| x[0] - 3.0; // h(x) = x - 3 should be 0
        let v = eq_violations(x, &[h]);
        assert!((v.total_violation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_combined_violations() {
        let x = &[1.5_f64, 2.0_f64];
        let g = |x: &[f64]| x[0] - 1.0; // violated by 0.5
        let h = |x: &[f64]| x[1] - 2.0; // satisfied (|0| = 0)
        let v = combined_violations(x, &[g], &[h]);
        assert!((v.total_violation - 0.5).abs() < 1e-10);
        assert_eq!(v.n_violated, 1);
    }
}
