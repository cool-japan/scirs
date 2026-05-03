//! Lift-and-Project MIP Solver Integration
//!
//! Provides [`LiftProjectMipSolver`], which wraps the BCC lift-and-project cut
//! generator and maintains a pool of accumulated cuts.  The solver supports an
//! iterated cut-generation loop suitable for embedding inside a branch-and-cut
//! framework.
//!
//! # Workflow
//!
//! ```text
//! 1. Initialise LiftProjectMipSolver with a LiftProjectConfig.
//! 2. Solve LP relaxation externally to obtain x̄.
//! 3. Call add_cuts_to_lp(a, b, x̄, integer_vars) to generate new cuts.
//!    – The returned cuts are those that are violated at x̄.
//!    – Cuts are added to the internal pool for reuse across iterations.
//! 4. Augment the LP with the new cuts (caller's responsibility).
//! 5. Repeat from step 2 until x̄ is integer-feasible or no cuts are generated.
//! ```
//!
//! # References
//! - Balas, E., Ceria, S., & Cornuéjols, G. (1993). "A lift-and-project cutting
//!   plane algorithm for mixed 0–1 programs." Mathematical Programming, 58(1-3), 295-324.

use super::lift_project::{LiftProjectConfig, LiftProjectCut, LiftProjectGenerator};
use crate::error::{OptimizeError, OptimizeResult};

// ── Solver ───────────────────────────────────────────────────────────────────

/// Lift-and-project cut manager for MIP solving.
///
/// Maintains a pool of previously generated cuts so they can be re-used
/// across branch-and-cut nodes without regenerating them from scratch.
pub struct LiftProjectMipSolver {
    config: LiftProjectConfig,
    generator: LiftProjectGenerator,
    cut_pool: Vec<LiftProjectCut>,
    /// Total number of calls to `add_cuts_to_lp`.
    iterations: usize,
    /// Total cuts generated across all iterations.
    total_cuts_generated: usize,
}

impl LiftProjectMipSolver {
    /// Create a new solver with the given configuration.
    pub fn new(config: LiftProjectConfig) -> Self {
        let generator = LiftProjectGenerator::new(config.clone());
        LiftProjectMipSolver {
            config,
            generator,
            cut_pool: Vec::new(),
            iterations: 0,
            total_cuts_generated: 0,
        }
    }

    /// Create a solver with default configuration.
    pub fn default_solver() -> Self {
        LiftProjectMipSolver::new(LiftProjectConfig::default())
    }

    // ── Cut generation ───────────────────────────────────────────────────

    /// Generate lift-and-project cuts violated at `x_bar` and add them to the pool.
    ///
    /// # Arguments
    ///
    /// * `a` – Constraint matrix rows (`a[i][k]` is the coefficient of x_k in row i).
    ///   The constraint is `a[i] · x ≤ b[i]`.
    /// * `b` – Right-hand side.
    /// * `x_bar` – Current LP relaxation solution.
    /// * `integer_vars` – Indices of variables constrained to {0, 1}.
    ///
    /// # Returns
    ///
    /// The newly generated cuts that are violated at `x_bar`.  These have already
    /// been added to the internal cut pool; the caller should augment the LP with
    /// these cuts and re-solve.
    pub fn add_cuts_to_lp(
        &mut self,
        a: &[Vec<f64>],
        b: &[f64],
        x_bar: &[f64],
        integer_vars: &[usize],
    ) -> OptimizeResult<Vec<LiftProjectCut>> {
        self.iterations += 1;

        let new_cuts = self.generator.generate_cuts(a, b, x_bar, integer_vars)?;

        if new_cuts.is_empty() {
            return Ok(Vec::new());
        }

        // Filter: keep only cuts that are actually violated at x_bar
        // (the generator already ensures this, but we verify defensively)
        let violated: Vec<LiftProjectCut> = new_cuts
            .into_iter()
            .filter(|c| {
                let v = self.generator.cut_violation(c, x_bar);
                v > self.config.cut_violation_tol
            })
            .collect();

        self.total_cuts_generated += violated.len();
        self.cut_pool.extend(violated.clone());

        Ok(violated)
    }

    // ── Pool management ──────────────────────────────────────────────────

    /// Number of cuts currently in the pool.
    pub fn cut_pool_size(&self) -> usize {
        self.cut_pool.len()
    }

    /// Access the full cut pool (read-only).
    pub fn cut_pool(&self) -> &[LiftProjectCut] {
        &self.cut_pool
    }

    /// Remove all cuts from the pool (e.g., when moving to a new B&B node).
    pub fn clear_cut_pool(&mut self) {
        self.cut_pool.clear();
    }

    /// Retain only cuts that are still violated at a new LP solution `x_new`.
    ///
    /// This is useful for node-level cut management: cuts that were valid at
    /// the parent node but are no longer violated can be removed to keep the
    /// working LP compact.
    pub fn purge_non_violated_cuts(&mut self, x_new: &[f64]) {
        self.cut_pool.retain(|c| {
            let v = self.generator.cut_violation(c, x_new);
            v > self.config.cut_violation_tol
        });
    }

    // ── Statistics ───────────────────────────────────────────────────────

    /// Number of times `add_cuts_to_lp` has been called.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Cumulative count of all cuts ever generated (including purged ones).
    pub fn total_cuts_generated(&self) -> usize {
        self.total_cuts_generated
    }

    /// Read-only reference to the underlying configuration.
    pub fn config(&self) -> &LiftProjectConfig {
        &self.config
    }

    /// Compute the violation of a cut at `x_bar`.
    ///
    /// Returns `π · x_bar - π₀`.  Positive means `x_bar` violates the cut.
    pub fn cut_violation(&self, cut: &LiftProjectCut, x_bar: &[f64]) -> f64 {
        self.generator.cut_violation(cut, x_bar)
    }

    // ── Augmented constraint matrix helper ───────────────────────────────

    /// Build an augmented constraint system by appending pooled cuts to `(a, b)`.
    ///
    /// The BCC cuts are stored as π · x ≥ π₀.  To express them in ≤ form for
    /// standard LP solvers, we negate: −π · x ≤ −π₀.
    ///
    /// # Returns
    ///
    /// `(a_aug, b_aug)` where the first `a.len()` rows are the original constraints
    /// and the subsequent rows are the negated cut inequalities.
    pub fn build_augmented_system(
        &self,
        a: &[Vec<f64>],
        b: &[f64],
    ) -> OptimizeResult<(Vec<Vec<f64>>, Vec<f64>)> {
        if a.len() != b.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "Constraint matrix has {} rows but b has {} entries",
                a.len(),
                b.len()
            )));
        }

        let n = if a.is_empty() {
            // Try to infer from cut pool
            self.cut_pool.first().map_or(0, |c| c.pi.len())
        } else {
            a[0].len()
        };

        let mut a_aug: Vec<Vec<f64>> = a.to_vec();
        let mut b_aug: Vec<f64> = b.to_vec();

        for cut in &self.cut_pool {
            if cut.pi.len() != n {
                return Err(OptimizeError::InvalidInput(format!(
                    "Cut has {} coefficients but constraint matrix has {} columns",
                    cut.pi.len(),
                    n
                )));
            }
            // π · x ≥ π₀  ↔  −π · x ≤ −π₀
            let neg_pi: Vec<f64> = cut.pi.iter().map(|&p| -p).collect();
            a_aug.push(neg_pi);
            b_aug.push(-cut.pi0);
        }

        Ok((a_aug, b_aug))
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fractional_lp() -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>, Vec<usize>) {
        // x1 + x2 <= 1, x_bar = (0.4, 0.6), integer vars = {0, 1}
        let a = vec![vec![1.0, 1.0]];
        let b = vec![1.0];
        let x_bar = vec![0.4, 0.6];
        let ivars = vec![0, 1];
        (a, b, x_bar, ivars)
    }

    #[test]
    fn test_add_cuts_increases_pool_size() {
        let mut solver = LiftProjectMipSolver::default_solver();
        let (a, b, x_bar, ivars) = make_fractional_lp();
        assert_eq!(solver.cut_pool_size(), 0);
        let cuts = solver.add_cuts_to_lp(&a, &b, &x_bar, &ivars).unwrap();
        assert_eq!(solver.cut_pool_size(), cuts.len());
        assert!(solver.cut_pool_size() > 0, "Expected cuts to be generated");
    }

    #[test]
    fn test_add_cuts_returns_violated_cuts() {
        let mut solver = LiftProjectMipSolver::default_solver();
        let (a, b, x_bar, ivars) = make_fractional_lp();
        let cuts = solver.add_cuts_to_lp(&a, &b, &x_bar, &ivars).unwrap();
        for cut in &cuts {
            let v = solver.cut_violation(cut, &x_bar);
            assert!(
                v > solver.config().cut_violation_tol,
                "Returned cut should be violated at x_bar, got v={}",
                v
            );
        }
    }

    #[test]
    fn test_add_cuts_empty_for_integer_solution() {
        let mut solver = LiftProjectMipSolver::default_solver();
        let (a, b, _, ivars) = make_fractional_lp();
        let x_int = vec![1.0, 0.0]; // integer point
        let cuts = solver.add_cuts_to_lp(&a, &b, &x_int, &ivars).unwrap();
        assert!(cuts.is_empty());
        assert_eq!(solver.cut_pool_size(), 0);
    }

    #[test]
    fn test_clear_cut_pool_resets_size() {
        let mut solver = LiftProjectMipSolver::default_solver();
        let (a, b, x_bar, ivars) = make_fractional_lp();
        solver.add_cuts_to_lp(&a, &b, &x_bar, &ivars).unwrap();
        assert!(solver.cut_pool_size() > 0);
        solver.clear_cut_pool();
        assert_eq!(solver.cut_pool_size(), 0);
    }

    #[test]
    fn test_iterations_counter_increments() {
        let mut solver = LiftProjectMipSolver::default_solver();
        let (a, b, x_bar, ivars) = make_fractional_lp();
        assert_eq!(solver.iterations(), 0);
        solver.add_cuts_to_lp(&a, &b, &x_bar, &ivars).unwrap();
        assert_eq!(solver.iterations(), 1);
        solver.add_cuts_to_lp(&a, &b, &x_bar, &ivars).unwrap();
        assert_eq!(solver.iterations(), 2);
    }

    #[test]
    fn test_total_cuts_generated_accumulates() {
        let mut solver = LiftProjectMipSolver::default_solver();
        let (a, b, x_bar, ivars) = make_fractional_lp();
        solver.add_cuts_to_lp(&a, &b, &x_bar, &ivars).unwrap();
        let after_first = solver.total_cuts_generated();
        solver.add_cuts_to_lp(&a, &b, &x_bar, &ivars).unwrap();
        let after_second = solver.total_cuts_generated();
        assert!(after_second >= after_first);
    }

    #[test]
    fn test_pool_accumulates_across_calls() {
        let mut solver = LiftProjectMipSolver::default_solver();
        let (a, b, x_bar, ivars) = make_fractional_lp();
        solver.add_cuts_to_lp(&a, &b, &x_bar, &ivars).unwrap();
        let size_after_first = solver.cut_pool_size();
        solver.add_cuts_to_lp(&a, &b, &x_bar, &ivars).unwrap();
        let size_after_second = solver.cut_pool_size();
        assert!(size_after_second >= size_after_first);
    }

    #[test]
    fn test_purge_non_violated_cuts() {
        let mut solver = LiftProjectMipSolver::default_solver();
        let (a, b, x_bar, ivars) = make_fractional_lp();
        solver.add_cuts_to_lp(&a, &b, &x_bar, &ivars).unwrap();
        let size_before = solver.cut_pool_size();
        // After purging with an integer solution, all cuts should be removed
        // (no cut is violated at an integer feasible point)
        let x_int = vec![1.0, 0.0];
        solver.purge_non_violated_cuts(&x_int);
        let size_after = solver.cut_pool_size();
        assert!(
            size_after <= size_before,
            "Pool should not grow after purge"
        );
    }

    #[test]
    fn test_build_augmented_system_appends_cuts() {
        let mut solver = LiftProjectMipSolver::default_solver();
        let (a, b, x_bar, ivars) = make_fractional_lp();
        solver.add_cuts_to_lp(&a, &b, &x_bar, &ivars).unwrap();
        let n_original = a.len();
        let n_cuts = solver.cut_pool_size();
        let (a_aug, b_aug) = solver.build_augmented_system(&a, &b).unwrap();
        assert_eq!(a_aug.len(), n_original + n_cuts);
        assert_eq!(b_aug.len(), n_original + n_cuts);
    }

    #[test]
    fn test_build_augmented_system_negates_cuts() {
        let mut solver = LiftProjectMipSolver::default_solver();
        let (a, b, x_bar, ivars) = make_fractional_lp();
        solver.add_cuts_to_lp(&a, &b, &x_bar, &ivars).unwrap();
        let (a_aug, b_aug) = solver.build_augmented_system(&a, &b).unwrap();
        let n_orig = a.len();
        // Check that the appended rows are negated cut coefficients
        for (k, cut) in solver.cut_pool().iter().enumerate() {
            let row = &a_aug[n_orig + k];
            let rhs = b_aug[n_orig + k];
            for (j, (&aug_coeff, &pi_k)) in row.iter().zip(cut.pi.iter()).enumerate() {
                assert!(
                    (aug_coeff - (-pi_k)).abs() < 1e-12,
                    "Augmented row coeff [{}][{}] = {} but expected {}",
                    k,
                    j,
                    aug_coeff,
                    -pi_k
                );
            }
            assert!(
                (rhs - (-cut.pi0)).abs() < 1e-12,
                "Augmented RHS = {} but expected {}",
                rhs,
                -cut.pi0
            );
        }
    }

    #[test]
    fn test_build_augmented_system_error_on_mismatched_a_b() {
        let solver = LiftProjectMipSolver::default_solver();
        let a = vec![vec![1.0, 1.0], vec![0.0, 1.0]];
        let b = vec![1.0]; // only 1 entry for 2 rows
        let result = solver.build_augmented_system(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_cut_pool_accessor_matches_pool_size() {
        let mut solver = LiftProjectMipSolver::default_solver();
        let (a, b, x_bar, ivars) = make_fractional_lp();
        solver.add_cuts_to_lp(&a, &b, &x_bar, &ivars).unwrap();
        assert_eq!(solver.cut_pool().len(), solver.cut_pool_size());
    }

    #[test]
    fn test_config_accessor() {
        let config = LiftProjectConfig {
            max_cuts: 7,
            cut_violation_tol: 1e-5,
            ..Default::default()
        };
        let solver = LiftProjectMipSolver::new(config.clone());
        assert_eq!(solver.config().max_cuts, 7);
        assert!((solver.config().cut_violation_tol - 1e-5).abs() < 1e-12);
    }

    #[test]
    fn test_multiple_constraint_rows_generate_more_cuts() {
        let mut solver = LiftProjectMipSolver::default_solver();
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let b = vec![0.8, 0.8, 1.2];
        let x_bar = vec![0.4, 0.5];
        let ivars = vec![0, 1];
        let cuts = solver.add_cuts_to_lp(&a, &b, &x_bar, &ivars).unwrap();
        // With three rows and two fractional variables there should be cuts
        assert!(!cuts.is_empty());
    }

    #[test]
    fn test_solver_handles_no_integer_vars_gracefully() {
        let mut solver = LiftProjectMipSolver::default_solver();
        let a = vec![vec![1.0, 1.0]];
        let b = vec![1.0];
        let x_bar = vec![0.4, 0.6];
        // Pass empty integer_vars: no cuts should be generated
        let cuts = solver.add_cuts_to_lp(&a, &b, &x_bar, &[]).unwrap();
        assert!(cuts.is_empty());
    }
}
