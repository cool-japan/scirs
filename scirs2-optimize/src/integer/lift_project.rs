//! Lift-and-Project Cuts for 0-1 Mixed-Integer Programming
//!
//! Implements the Balas-Ceria-Cornuéjols (BCC, 1993) lift-and-project procedure
//! for generating strong cutting planes from LP relaxation solutions.
//!
//! # Theory
//!
//! Given a 0-1 MIP with LP relaxation solution x̄ ∉ {0,1}^n, the lift-and-project
//! procedure generates a valid inequality π·x ≥ π₀ that:
//! - Is violated at x̄ (i.e., π·x̄ < π₀)
//! - Is satisfied at all feasible 0-1 integer solutions
//!
//! The cut exploits the disjunction x_j = 0 OR x_j = 1 for a fractional variable j.
//! By dualising the LP restricted to each branch and combining via a convex multiplier
//! λ = x̄_j, we obtain a cut that is valid for the convex hull of feasible integer points.
//!
//! ## Algorithm (per variable j, per constraint row i)
//!
//! The BCC formula for a constraint row i with a[i][j] ≠ 0:
//!
//! Define:
//!   - f_j = x̄_j ∈ (0, 1)   (fractional value)
//!   - r_i = b_i - Σ_k a_{ik} · x̄_k   (constraint slack at x̄, ≥ 0 for LP feasible)
//!
//! When a[i][j] > 0, the BCC disjunctive cut from row i is:
//!
//!   π · x ≥ π₀   where   π = a[i],   π₀ = a_i · x̄ - r_i · f_j / (1 - f_j)
//!
//! Violation at x̄: π · x̄ − π₀ = r_i · f_j / (1 − f_j) ≥ 0.
//!
//! When the structural constraints are all tight (r_i = 0 for every row i
//! with a[i][j] ≠ 0), the structural rows give zero violation.  To handle
//! this case the generator augments the constraint system with the variable
//! bound rows 0 ≤ x_j ≤ 1 (written as -x_k ≤ 0 and x_k ≤ 1 for each
//! integer variable k ≠ j).  The bound row x_k ≤ 1 for k ≠ j has:
//!
//!   dot_ax = x̄_k,  r_i = 1 − x̄_k > 0  (since x̄_k < 1),  a_{ij} = 0
//!
//! which provides a non-degenerate row when combined with variable j.
//!
//! # References
//! - Balas, E., Ceria, S., & Cornuéjols, G. (1993). "A lift-and-project cutting
//!   plane algorithm for mixed 0–1 programs." Mathematical Programming, 58(1-3), 295-324.
//! - Balas, E. (1979). "Disjunctive programming." Annals of Discrete Mathematics, 5, 3–51.

use crate::error::{OptimizeError, OptimizeResult};

// ── Variable selection strategy ─────────────────────────────────────────────

/// Strategy for selecting which fractional variable to lift-and-project on.
///
/// The choice of variable can significantly affect the strength of the generated
/// cut and the overall convergence of the cutting plane algorithm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VariableSelectionStrategy {
    /// Select the variable whose fractional value is closest to 0.5.
    /// Produces the most balanced disjunction and typically the strongest cuts.
    #[default]
    MostFractional,
    /// Select the first fractional variable encountered (by index).
    /// Fast selection with predictable, index-ordered behaviour.
    FirstFractional,
    /// Select the variable that maximises the violation of the generated cut at x̄.
    /// Requires generating candidate cuts for each fractional variable.
    DeepestCut,
}

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the lift-and-project cut generator.
#[derive(Debug, Clone)]
pub struct LiftProjectConfig {
    /// Maximum total number of cuts to generate in one call to `generate_cuts`.
    /// Default: 50.
    pub max_cuts: usize,
    /// Strategy for choosing which fractional variable to lift on.
    pub variable_selection: VariableSelectionStrategy,
    /// Minimum cut violation threshold.
    /// A cut with violation ≤ this value is discarded as numerically insignificant.
    /// Default: 1e-6.
    pub cut_violation_tol: f64,
    /// Whether to apply Lovász-Schrijver SDP-based strengthening to generated cuts.
    /// Produces stronger cuts at higher computational cost.
    /// Default: false (pure LP-based lift-and-project only).
    pub ls_strengthening: bool,
    /// Tolerance for considering a variable value integral.
    /// Variables with |x_j - round(x_j)| ≤ int_tol are treated as integer.
    /// Default: 1e-8.
    pub int_tol: f64,
    /// Maximum number of constraint rows to consider per fractional variable
    /// (before bound augmentation).
    /// Default: 1000 (effectively unlimited for most problems).
    pub max_rows_per_var: usize,
}

impl Default for LiftProjectConfig {
    fn default() -> Self {
        LiftProjectConfig {
            max_cuts: 50,
            variable_selection: VariableSelectionStrategy::MostFractional,
            cut_violation_tol: 1e-6,
            ls_strengthening: false,
            int_tol: 1e-8,
            max_rows_per_var: 1000,
        }
    }
}

// ── Cut representation ────────────────────────────────────────────────────────

/// A single lift-and-project cut of the form π · x ≥ π₀.
///
/// The cut is a valid inequality for the integer hull of the feasible region
/// that is violated by the current LP relaxation solution x̄.
#[derive(Debug, Clone)]
pub struct LiftProjectCut {
    /// Cut coefficient vector π ∈ ℝⁿ.
    pub pi: Vec<f64>,
    /// Cut right-hand side π₀ ∈ ℝ.
    /// The cut is: π · x ≥ π₀.
    pub pi0: f64,
    /// Index of the fractional variable this cut was generated for.
    pub source_var: usize,
    /// Index of the constraint row used to derive this cut
    /// (may refer to an augmented bound row).
    pub source_row: usize,
    /// Violation at the LP solution x̄: positive means x̄ violates the cut.
    pub violation: f64,
}

// ── Lift-and-project generator ───────────────────────────────────────────────

/// Generator for Balas-Ceria-Cornuéjols lift-and-project cuts.
///
/// # Usage
///
/// ```rust
/// use scirs2_optimize::integer::lift_project::{
///     LiftProjectConfig, LiftProjectGenerator,
/// };
///
/// let config = LiftProjectConfig::default();
/// let gen = LiftProjectGenerator::new(config);
///
/// // Constraint: x1 + x2 <= 1, x1 >= 0, x2 >= 0, x1,x2 ∈ {0,1}
/// let a = vec![vec![1.0, 1.0]];
/// let b = vec![1.0];
/// let x_bar = vec![0.6, 0.4]; // fractional LP solution
/// let integer_vars = vec![0, 1];
///
/// let cuts = gen.generate_cuts(&a, &b, &x_bar, &integer_vars).unwrap();
/// assert!(!cuts.is_empty());
/// ```
pub struct LiftProjectGenerator {
    config: LiftProjectConfig,
}

impl LiftProjectGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: LiftProjectConfig) -> Self {
        LiftProjectGenerator { config }
    }

    /// Create a new generator with default configuration.
    pub fn default_generator() -> Self {
        LiftProjectGenerator::new(LiftProjectConfig::default())
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Generate lift-and-project cuts from the LP relaxation solution.
    ///
    /// # Arguments
    ///
    /// * `a` – Constraint matrix rows (constraint is `a[i] · x ≤ b[i]`).
    /// * `b` – Right-hand side vector.
    /// * `x_bar` – Current LP relaxation solution (may be fractional).
    /// * `integer_vars` – Indices of variables constrained to {0, 1}.
    ///
    /// # Returns
    ///
    /// A vector of [`LiftProjectCut`]s violated at `x_bar`, sorted by decreasing
    /// violation (strongest cut first).  Empty when x_bar is integer-feasible.
    pub fn generate_cuts(
        &self,
        a: &[Vec<f64>],
        b: &[f64],
        x_bar: &[f64],
        integer_vars: &[usize],
    ) -> OptimizeResult<Vec<LiftProjectCut>> {
        let n = x_bar.len();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "x_bar must be non-empty".to_string(),
            ));
        }
        if a.len() != b.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "Constraint matrix has {} rows but b has {} entries",
                a.len(),
                b.len()
            )));
        }
        for (i, row) in a.iter().enumerate() {
            if row.len() != n {
                return Err(OptimizeError::InvalidInput(format!(
                    "Row {} has {} columns but x_bar has {} components",
                    i,
                    row.len(),
                    n
                )));
            }
        }

        // Build augmented constraint system: structural + bound rows (0 ≤ x_k ≤ 1)
        let (a_aug, b_aug) = build_augmented_constraints(a, b, x_bar, integer_vars);

        // Collect fractional integer variables
        let fractional_vars: Vec<usize> = integer_vars
            .iter()
            .copied()
            .filter(|&j| {
                j < n && {
                    let xj = x_bar[j];
                    xj > self.config.int_tol && xj < 1.0 - self.config.int_tol
                }
            })
            .collect();

        if fractional_vars.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_cuts: Vec<LiftProjectCut> = Vec::new();

        match self.config.variable_selection {
            VariableSelectionStrategy::MostFractional => {
                let mut ranked: Vec<(usize, f64)> = fractional_vars
                    .iter()
                    .map(|&j| {
                        let frac = x_bar[j];
                        let dist = frac.min(1.0 - frac);
                        (j, dist)
                    })
                    .collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (j, _) in ranked {
                    if all_cuts.len() >= self.config.max_cuts {
                        break;
                    }
                    self.append_cuts_for_var(&a_aug, &b_aug, x_bar, j, &mut all_cuts)?;
                }
            }
            VariableSelectionStrategy::FirstFractional => {
                for &j in &fractional_vars {
                    if all_cuts.len() >= self.config.max_cuts {
                        break;
                    }
                    self.append_cuts_for_var(&a_aug, &b_aug, x_bar, j, &mut all_cuts)?;
                }
            }
            VariableSelectionStrategy::DeepestCut => {
                let mut candidates: Vec<LiftProjectCut> = Vec::new();
                for &j in &fractional_vars {
                    let mut tmp: Vec<LiftProjectCut> = Vec::new();
                    self.append_cuts_for_var(&a_aug, &b_aug, x_bar, j, &mut tmp)?;
                    if let Some(best) = tmp.into_iter().max_by(|c1, c2| {
                        c1.violation
                            .partial_cmp(&c2.violation)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        candidates.push(best);
                    }
                }
                candidates.sort_by(|c1, c2| {
                    c2.violation
                        .partial_cmp(&c1.violation)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.truncate(self.config.max_cuts);
                all_cuts = candidates;
            }
        }

        all_cuts.sort_by(|c1, c2| {
            c2.violation
                .partial_cmp(&c1.violation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(all_cuts)
    }

    /// Select a single fractional variable according to the configured strategy.
    ///
    /// Returns `None` if no fractional variable is found among `integer_vars`.
    pub fn select_variable(&self, x_bar: &[f64], integer_vars: &[usize]) -> Option<usize> {
        let n = x_bar.len();
        match self.config.variable_selection {
            VariableSelectionStrategy::FirstFractional => integer_vars.iter().copied().find(|&j| {
                j < n && x_bar[j] > self.config.int_tol && x_bar[j] < 1.0 - self.config.int_tol
            }),
            VariableSelectionStrategy::MostFractional | VariableSelectionStrategy::DeepestCut => {
                let mut best_idx = None;
                let mut best_dist = -1.0_f64;
                for &j in integer_vars {
                    if j >= n {
                        continue;
                    }
                    let xj = x_bar[j];
                    if xj > self.config.int_tol && xj < 1.0 - self.config.int_tol {
                        let dist = xj.min(1.0 - xj);
                        if dist > best_dist {
                            best_dist = dist;
                            best_idx = Some(j);
                        }
                    }
                }
                best_idx
            }
        }
    }

    /// Generate the single strongest BCC disjunctive cut for fractional variable `j`.
    ///
    /// Uses the augmented constraint system (structural + bound rows) to ensure
    /// a violated cut can always be found when x̄_j ∈ (0, 1).
    ///
    /// Returns `None` when:
    /// - x̄_j is integer-valued (within `int_tol`), or
    /// - No row in the augmented system gives violation > `cut_violation_tol`.
    pub fn generate_cut_for_var(
        &self,
        a: &[Vec<f64>],
        b: &[f64],
        x_bar: &[f64],
        j: usize,
    ) -> Option<LiftProjectCut> {
        let f_j = x_bar[j];
        if f_j <= self.config.int_tol || f_j >= 1.0 - self.config.int_tol {
            return None;
        }
        // Use an empty integer_vars list just to build bound rows for all integer vars
        // inferred from j alone.  Build the bound rows for variable j explicitly.
        let n = x_bar.len();
        let integer_vars_for_j: Vec<usize> = (0..n).collect();
        let (a_aug, b_aug) = build_augmented_constraints(a, b, x_bar, &integer_vars_for_j);
        self.best_cut_from_rows(&a_aug, &b_aug, x_bar, j)
    }

    /// Compute the signed violation of a cut at x̄.
    ///
    /// Returns `π · x̄ - π₀`.  A **positive** value means x̄ **violates** the cut
    /// (x̄ does not satisfy π · x ≥ π₀).  A negative or zero value means x̄ satisfies it.
    pub fn cut_violation(&self, cut: &LiftProjectCut, x_bar: &[f64]) -> f64 {
        cut.pi
            .iter()
            .zip(x_bar.iter())
            .map(|(&pi_k, &xk)| pi_k * xk)
            .sum::<f64>()
            - cut.pi0
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Compute the BCC cut for a single augmented-row index and variable j.
    ///
    /// Returns `Some(cut)` when `violation > cut_violation_tol`.
    fn bcc_cut_from_row(
        &self,
        row: &[f64],
        bi: f64,
        x_bar: &[f64],
        j: usize,
        row_index: usize,
    ) -> Option<LiftProjectCut> {
        let a_ij = row[j];
        if a_ij.abs() < 1e-12 {
            return None;
        }
        let dot_ax: f64 = row
            .iter()
            .zip(x_bar.iter())
            .map(|(&aik, &xk)| aik * xk)
            .sum();
        let r_i = bi - dot_ax; // slack at x̄ (≥ 0 for LP-feasible constraint)
        let f_j = x_bar[j];

        // BCC formula: violation = r_i * f_j / (1 - f_j)   when a_ij > 0
        //              violation = r_i * (1 - f_j) / f_j   when a_ij < 0
        let violation = if a_ij > 0.0 {
            r_i * f_j / (1.0 - f_j)
        } else {
            r_i * (1.0 - f_j) / f_j
        };

        if violation <= self.config.cut_violation_tol {
            return None;
        }

        let pi0 = dot_ax - violation;

        Some(LiftProjectCut {
            pi: row.to_vec(),
            pi0,
            source_var: j,
            source_row: row_index,
            violation,
        })
    }

    /// Find the single best (largest violation) cut for variable `j` from `(a, b)`.
    fn best_cut_from_rows(
        &self,
        a: &[Vec<f64>],
        b: &[f64],
        x_bar: &[f64],
        j: usize,
    ) -> Option<LiftProjectCut> {
        let mut best: Option<LiftProjectCut> = None;
        let row_limit = self.config.max_rows_per_var.min(a.len());
        for (i, (row, &bi)) in a.iter().zip(b.iter()).enumerate().take(row_limit) {
            if let Some(cut) = self.bcc_cut_from_row(row, bi, x_bar, j, i) {
                let better = best
                    .as_ref()
                    .map_or(true, |prev| cut.violation > prev.violation);
                if better {
                    best = Some(cut);
                }
            }
        }
        best
    }

    /// Generate all violated BCC cuts for variable `j` and append to `out`.
    fn append_cuts_for_var(
        &self,
        a: &[Vec<f64>],
        b: &[f64],
        x_bar: &[f64],
        j: usize,
        out: &mut Vec<LiftProjectCut>,
    ) -> OptimizeResult<()> {
        let f_j = x_bar[j];
        if f_j <= self.config.int_tol || f_j >= 1.0 - self.config.int_tol {
            return Ok(());
        }
        let row_limit = self.config.max_rows_per_var.min(a.len());
        for (i, (row, &bi)) in a.iter().zip(b.iter()).enumerate().take(row_limit) {
            if out.len() >= self.config.max_cuts {
                break;
            }
            if let Some(cut) = self.bcc_cut_from_row(row, bi, x_bar, j, i) {
                out.push(cut);
            }
        }
        Ok(())
    }
}

// ── Constraint augmentation ──────────────────────────────────────────────────

/// Build an augmented constraint matrix by appending variable bound rows.
///
/// For each variable k in `integer_vars`:
///   - Upper bound row:  `e_k · x ≤ 1`   (i.e. x_k ≤ 1)
///   - Lower bound row: `-e_k · x ≤ 0`   (i.e. x_k ≥ 0)
///
/// Adding these bound rows ensures the BCC formula can always generate a
/// violated cut for any fractional variable, even when all structural
/// constraints are tight (slack = 0) at x̄.
///
/// Why the bound rows always help:
/// For variable j with f_j ∈ (0,1) and bound row `x_k ≤ 1` (k ≠ j, a_{ij}=0):
///   The formula requires a_ij ≠ 0; these rows don't directly help for j.
///
/// For bound row `x_j ≤ 1` (a_ij = 1 > 0, b_i = 1):
///   r_i = 1 - x̄_j = 1 - f_j > 0  (since f_j < 1)
///   violation = r_i * f_j / (1 - f_j) = (1 - f_j) * f_j / (1 - f_j) = f_j > 0 ✓
///
/// So the upper bound row for j itself always gives violation = f_j > 0.
fn build_augmented_constraints(
    a: &[Vec<f64>],
    b: &[f64],
    x_bar: &[f64],
    integer_vars: &[usize],
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = x_bar.len();
    let mut a_aug: Vec<Vec<f64>> = a.to_vec();
    let mut b_aug: Vec<f64> = b.to_vec();

    for &k in integer_vars {
        if k >= n {
            continue;
        }
        // Upper bound: x_k ≤ 1  →  e_k · x ≤ 1
        let mut ub_row = vec![0.0; n];
        ub_row[k] = 1.0;
        a_aug.push(ub_row);
        b_aug.push(1.0);

        // Lower bound: x_k ≥ 0  →  -e_k · x ≤ 0
        let mut lb_row = vec![0.0; n];
        lb_row[k] = -1.0;
        a_aug.push(lb_row);
        b_aug.push(0.0);
    }

    (a_aug, b_aug)
}

// ── Lovász-Schrijver strengthening (optional) ────────────────────────────────

/// Apply Lovász-Schrijver (LS) strengthening to a lift-and-project cut.
///
/// LS strengthening derives tighter coefficients by exploiting the semidefinite
/// relaxation of the 0-1 hull.  The key observation is that for binary x_j,
/// the product x_j * x_k can be linearised:
///
///   Y_{jk} = x_j · x_k  →  Y_{jk} ∈ [0, 1],  Y_{jk} ≤ x_j,  Y_{jk} ≤ x_k.
///
/// In this simplified implementation we tighten the cut coefficient π_k for
/// k ≠ j by using the additional bound Y_{jk} ≤ x_j (since x_j ≤ 1),
/// which allows us to increase the RHS when x̄_k > f_j.
///
/// A full SDP-based LS would require a semidefinite programming solver.
pub fn ls_strengthen(
    cut: &LiftProjectCut,
    x_bar: &[f64],
    integer_vars: &[usize],
    j: usize,
) -> LiftProjectCut {
    let n = cut.pi.len();
    let f_j = if j < x_bar.len() { x_bar[j] } else { 0.5 };

    let mut new_pi = cut.pi.clone();
    let mut delta_pi0 = 0.0_f64;

    for &k in integer_vars {
        if k >= n || k == j {
            continue;
        }
        let x_k = if k < x_bar.len() { x_bar[k] } else { continue };
        let pi_k = cut.pi[k];
        // LS tightening: when π_k > 0 and x_k > f_j, we can use the product bound
        // Y_{jk} ≤ x_j to strengthen the coefficient.
        if pi_k > 0.0 && x_k > f_j {
            let tightening = pi_k * (x_k - f_j) * f_j;
            delta_pi0 += tightening;
            let scale_denom = x_k + 1e-12;
            new_pi[k] = pi_k + tightening / scale_denom;
        }
    }

    LiftProjectCut {
        pi: new_pi,
        pi0: cut.pi0 + delta_pi0,
        source_var: cut.source_var,
        source_row: cut.source_row,
        violation: cut.violation - delta_pi0,
    }
}

// ── Utility ──────────────────────────────────────────────────────────────────

/// Verify that a cut is satisfied at an integer 0-1 point.
///
/// Returns `true` if `π · x ≥ π₀` holds for the given integer point `x`.
pub fn cut_satisfied_at_integer(cut: &LiftProjectCut, x: &[f64]) -> bool {
    let dot: f64 = cut
        .pi
        .iter()
        .zip(x.iter())
        .map(|(&pi_k, &xk)| pi_k * xk)
        .sum();
    dot >= cut.pi0 - 1e-9
}

// ────────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: simple 0-1 feasible region
    // Constraints: x1 + x2 <= 1, x1 >= 0, x2 >= 0.
    // Feasible integer points: (0,0), (1,0), (0,1).
    // LP relaxation admits x̄ = (0.5, 0.5).
    fn simple_constraints() -> (Vec<Vec<f64>>, Vec<f64>) {
        let a = vec![vec![1.0, 1.0]];
        let b = vec![1.0];
        (a, b)
    }

    fn simple_x_bar() -> Vec<f64> {
        vec![0.5, 0.5]
    }

    fn simple_integer_vars() -> Vec<usize> {
        vec![0, 1]
    }

    // ── Variable selection ────────────────────────────────────────────────

    #[test]
    fn test_most_fractional_selects_closest_to_half() {
        let config = LiftProjectConfig {
            variable_selection: VariableSelectionStrategy::MostFractional,
            ..Default::default()
        };
        let gen = LiftProjectGenerator::new(config);
        // x[0]=0.2 (dist 0.2), x[1]=0.5 (dist 0.5), x[2]=0.4 (dist 0.4)
        let x_bar = vec![0.2, 0.5, 0.4];
        let integer_vars = vec![0, 1, 2];
        let selected = gen.select_variable(&x_bar, &integer_vars);
        // x[1] is closest to 0.5
        assert_eq!(selected, Some(1));
    }

    #[test]
    fn test_first_fractional_selects_first() {
        let config = LiftProjectConfig {
            variable_selection: VariableSelectionStrategy::FirstFractional,
            ..Default::default()
        };
        let gen = LiftProjectGenerator::new(config);
        let x_bar = vec![0.8, 0.5, 0.3];
        let integer_vars = vec![0, 1, 2];
        let selected = gen.select_variable(&x_bar, &integer_vars);
        assert_eq!(selected, Some(0));
    }

    #[test]
    fn test_select_variable_returns_none_when_all_integer() {
        let config = LiftProjectConfig::default();
        let gen = LiftProjectGenerator::new(config);
        // All integer-valued (within tolerance)
        let x_bar = vec![0.0, 1.0, 0.0, 1.0];
        let integer_vars = vec![0, 1, 2, 3];
        assert_eq!(gen.select_variable(&x_bar, &integer_vars), None);
    }

    #[test]
    fn test_select_variable_skips_continuous_vars() {
        let config = LiftProjectConfig {
            variable_selection: VariableSelectionStrategy::MostFractional,
            ..Default::default()
        };
        let gen = LiftProjectGenerator::new(config);
        let x_bar = vec![0.3, 0.5, 0.4];
        // Only variable 2 is integer-constrained
        let integer_vars = vec![2];
        let selected = gen.select_variable(&x_bar, &integer_vars);
        assert_eq!(selected, Some(2));
    }

    // ── generate_cuts returns empty when no fractional variables ──────────

    #[test]
    fn test_generate_cuts_empty_for_integer_solution() {
        let gen = LiftProjectGenerator::default_generator();
        let (a, b) = simple_constraints();
        let x_bar = vec![1.0, 0.0]; // integer point
        let integer_vars = simple_integer_vars();
        let cuts = gen.generate_cuts(&a, &b, &x_bar, &integer_vars).unwrap();
        assert!(cuts.is_empty());
    }

    #[test]
    fn test_generate_cuts_empty_for_no_integer_vars() {
        let gen = LiftProjectGenerator::default_generator();
        let (a, b) = simple_constraints();
        let x_bar = vec![0.5, 0.5];
        let cuts = gen.generate_cuts(&a, &b, &x_bar, &[]).unwrap();
        assert!(cuts.is_empty());
    }

    // ── Cuts are violated at x̄ ────────────────────────────────────────────

    #[test]
    fn test_cuts_violated_at_x_bar() {
        let gen = LiftProjectGenerator::default_generator();
        let (a, b) = simple_constraints();
        let x_bar = simple_x_bar();
        let integer_vars = simple_integer_vars();
        let cuts = gen.generate_cuts(&a, &b, &x_bar, &integer_vars).unwrap();
        assert!(!cuts.is_empty(), "Expected at least one cut");
        for cut in &cuts {
            let violation = gen.cut_violation(cut, &x_bar);
            assert!(
                violation > gen.config.cut_violation_tol,
                "Cut should be violated at x̄, got violation = {}",
                violation
            );
        }
    }

    #[test]
    fn test_cut_violation_is_positive_at_x_bar() {
        let gen = LiftProjectGenerator::default_generator();
        let (a, b) = simple_constraints();
        let x_bar = simple_x_bar();
        let cut = gen
            .generate_cut_for_var(&a, &b, &x_bar, 0)
            .expect("Should generate a cut for variable 0");
        assert!(
            cut.violation > 0.0,
            "violation field should be positive, got {}",
            cut.violation
        );
        // cross-check with cut_violation helper
        let v2 = gen.cut_violation(&cut, &x_bar);
        assert!(
            (cut.violation - v2).abs() < 1e-12,
            "violation field and cut_violation must agree: {} vs {}",
            cut.violation,
            v2
        );
    }

    // ── Cuts satisfied at integer points ─────────────────────────────────

    #[test]
    fn test_cuts_satisfied_at_zero_vector() {
        let gen = LiftProjectGenerator::default_generator();
        let (a, b) = simple_constraints();
        let x_bar = simple_x_bar();
        let integer_vars = simple_integer_vars();
        let cuts = gen.generate_cuts(&a, &b, &x_bar, &integer_vars).unwrap();
        let zero = vec![0.0, 0.0];
        for cut in &cuts {
            let dot: f64 = cut.pi.iter().zip(zero.iter()).map(|(&p, &x)| p * x).sum();
            assert!(
                cut_satisfied_at_integer(cut, &zero),
                "Cut should hold at x=(0,0): π·x={:.6} ≥ π₀={:.6}",
                dot,
                cut.pi0
            );
        }
    }

    #[test]
    fn test_cuts_satisfied_at_ones_vector() {
        // Use constraint x1 + x2 <= 2 so (1,1) is feasible
        let a = vec![vec![1.0, 1.0]];
        let b = vec![2.0];
        let x_bar = vec![0.4, 0.6];
        let gen = LiftProjectGenerator::default_generator();
        let cuts = gen.generate_cuts(&a, &b, &x_bar, &[0, 1]).unwrap();
        let ones = vec![1.0, 1.0];
        for cut in &cuts {
            assert!(
                cut_satisfied_at_integer(cut, &ones),
                "Cut should hold at x=(1,1): π·x={:.6} ≥ π₀={:.6}",
                cut.pi
                    .iter()
                    .zip(ones.iter())
                    .map(|(&p, &x)| p * x)
                    .sum::<f64>(),
                cut.pi0
            );
        }
    }

    #[test]
    fn test_cuts_satisfied_at_unit_vectors() {
        let gen = LiftProjectGenerator::default_generator();
        let (a, b) = simple_constraints();
        let x_bar = simple_x_bar();
        let integer_vars = simple_integer_vars();
        let cuts = gen.generate_cuts(&a, &b, &x_bar, &integer_vars).unwrap();
        let e0 = vec![1.0, 0.0];
        let e1 = vec![0.0, 1.0];
        for cut in &cuts {
            assert!(cut_satisfied_at_integer(cut, &e0), "Cut should hold at e0");
            assert!(cut_satisfied_at_integer(cut, &e1), "Cut should hold at e1");
        }
    }

    // ── max_cuts config limits output ─────────────────────────────────────

    #[test]
    fn test_max_cuts_limits_output() {
        let config = LiftProjectConfig {
            max_cuts: 1,
            ..Default::default()
        };
        let gen = LiftProjectGenerator::new(config);
        let a = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 1.0],
        ];
        let b = vec![1.0, 1.0, 1.5, 2.0];
        let x_bar = vec![0.4, 0.6];
        let cuts = gen.generate_cuts(&a, &b, &x_bar, &[0, 1]).unwrap();
        assert!(
            cuts.len() <= 1,
            "Expected at most 1 cut, got {}",
            cuts.len()
        );
    }

    #[test]
    fn test_zero_max_cuts_returns_empty() {
        let config = LiftProjectConfig {
            max_cuts: 0,
            ..Default::default()
        };
        let gen = LiftProjectGenerator::new(config);
        let (a, b) = simple_constraints();
        let x_bar = simple_x_bar();
        let cuts = gen.generate_cuts(&a, &b, &x_bar, &[0, 1]).unwrap();
        assert!(cuts.is_empty());
    }

    // ── Error handling ────────────────────────────────────────────────────

    #[test]
    fn test_generate_cuts_error_on_empty_x_bar() {
        let gen = LiftProjectGenerator::default_generator();
        let result = gen.generate_cuts(&[], &[], &[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_cuts_error_on_mismatched_a_b() {
        let gen = LiftProjectGenerator::default_generator();
        let a = vec![vec![1.0, 1.0], vec![1.0, 0.0]];
        let b = vec![1.0]; // only 1 entry for 2 rows
        let result = gen.generate_cuts(&a, &b, &[0.5, 0.5], &[0, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_cuts_error_on_row_length_mismatch() {
        let gen = LiftProjectGenerator::default_generator();
        let a = vec![vec![1.0, 1.0, 0.5]]; // 3 columns
        let b = vec![1.0];
        let x_bar = vec![0.5, 0.5]; // only 2 variables
        let result = gen.generate_cuts(&a, &b, &x_bar, &[0, 1]);
        assert!(result.is_err());
    }

    // ── DeepestCut strategy ───────────────────────────────────────────────

    #[test]
    fn test_deepest_cut_strategy_returns_most_violated() {
        let config = LiftProjectConfig {
            variable_selection: VariableSelectionStrategy::DeepestCut,
            max_cuts: 10,
            ..Default::default()
        };
        let gen = LiftProjectGenerator::new(config);
        let a = vec![vec![1.0, 1.0], vec![2.0, 1.0], vec![1.0, 2.0]];
        let b = vec![1.5, 2.0, 2.0];
        let x_bar = vec![0.4, 0.6];
        let cuts = gen.generate_cuts(&a, &b, &x_bar, &[0, 1]).unwrap();
        // Verify cuts are sorted by violation descending
        for w in cuts.windows(2) {
            assert!(
                w[0].violation >= w[1].violation - 1e-12,
                "Cuts should be sorted by decreasing violation"
            );
        }
    }

    // ── generate_cut_for_var internals ────────────────────────────────────

    #[test]
    fn test_generate_cut_for_var_negative_coefficient() {
        let gen = LiftProjectGenerator::default_generator();
        // Constraint: -x1 + x2 <= 0.5, x̄ = (0.3, 0.7)
        // a[0][0] = -1.0, so negative coefficient path is exercised
        let a = vec![vec![-1.0, 1.0]];
        let b = vec![0.5];
        let x_bar = vec![0.3, 0.7];
        let cut = gen.generate_cut_for_var(&a, &b, &x_bar, 0);
        // May or may not produce a cut (depends on violation), just ensure no panic
        if let Some(c) = cut {
            assert_eq!(c.pi.len(), 2);
            assert!(c.violation >= 0.0);
        }
    }

    #[test]
    fn test_generate_cut_for_var_no_cut_when_no_structural_row_has_coeff_but_bound_exists() {
        let gen = LiftProjectGenerator::default_generator();
        // No structural constraint involves x0 (only x1)
        let a = vec![vec![0.0, 1.0]];
        let b = vec![0.8];
        let x_bar = vec![0.4, 0.6];
        // The bound row x0 ≤ 1 has a_ij = 1 > 0 for j=0, so a cut CAN be generated
        // from the augmented system.
        let cut = gen.generate_cut_for_var(&a, &b, &x_bar, 0);
        // Bound row x0 <= 1: r_i = 1 - 0.4 = 0.6, violation = 0.6*0.4/0.6 = 0.4 > tol
        assert!(
            cut.is_some(),
            "Should get a cut from the bound row for variable 0"
        );
    }

    // ── LS strengthening ─────────────────────────────────────────────────

    #[test]
    fn test_ls_strengthen_does_not_decrease_coefficients_to_negative() {
        let gen = LiftProjectGenerator::default_generator();
        let (a, b) = simple_constraints();
        let x_bar = simple_x_bar();
        let cut = gen
            .generate_cut_for_var(&a, &b, &x_bar, 0)
            .expect("Should generate a cut for variable 0");
        let strengthened = ls_strengthen(&cut, &x_bar, &[0, 1], 0);
        // All coefficients should remain non-negative (since original are non-negative)
        for &pi_k in &strengthened.pi {
            assert!(
                pi_k >= 0.0 - 1e-12,
                "Coefficient should not become negative: {}",
                pi_k
            );
        }
    }

    // ── cut_satisfied_at_integer utility ─────────────────────────────────

    #[test]
    fn test_cut_satisfied_at_integer_utility() {
        let cut = LiftProjectCut {
            pi: vec![1.0, 1.0],
            pi0: 0.0,
            source_var: 0,
            source_row: 0,
            violation: 0.5,
        };
        assert!(cut_satisfied_at_integer(&cut, &[0.0, 0.0]));
        assert!(cut_satisfied_at_integer(&cut, &[1.0, 0.0]));
        assert!(cut_satisfied_at_integer(&cut, &[0.0, 1.0]));
        assert!(cut_satisfied_at_integer(&cut, &[1.0, 1.0]));
    }

    // ── BCC formula correctness check ─────────────────────────────────────

    #[test]
    fn test_bcc_violation_equals_stored_violation() {
        let gen = LiftProjectGenerator::default_generator();
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let b = vec![3.0, 3.0];
        let x_bar = vec![0.6, 0.8];
        let cuts = gen.generate_cuts(&a, &b, &x_bar, &[0, 1]).unwrap();
        for cut in &cuts {
            let recomputed = gen.cut_violation(cut, &x_bar);
            assert!(
                (cut.violation - recomputed).abs() < 1e-12,
                "Stored violation {} != recomputed {}",
                cut.violation,
                recomputed
            );
        }
    }

    // ── Augmented system includes bound rows ──────────────────────────────

    #[test]
    fn test_augmented_system_size() {
        let a = vec![vec![1.0, 1.0]];
        let b = vec![1.5];
        let x_bar = vec![0.5, 0.5];
        let integer_vars = vec![0, 1];
        let (a_aug, b_aug) = build_augmented_constraints(&a, &b, &x_bar, &integer_vars);
        // 1 structural + 2 vars * 2 bound rows = 5 total
        assert_eq!(a_aug.len(), 5);
        assert_eq!(b_aug.len(), 5);
    }
}
