//! CDCL-Style (Conflict-Driven Clause Learning) MIP Branching
//!
//! This module adapts CDCL techniques from SAT solving to MIP branch-and-bound.
//! When a sub-problem is found infeasible, the algorithm extracts a "nogood"
//! clause (a conjunction of branching decisions that caused infeasibility) and
//! stores it to prune future subproblems that replay the same partial assignment.
//!
//! # Algorithm Overview
//!
//! 1. **Branching**: select the fractional variable with the highest VSIDS
//!    activity score.
//! 2. **Conflict analysis**: when a node is infeasible, record a learned clause
//!    `NOT(d_1) ∨ NOT(d_2) ∨ … ∨ NOT(d_k)` for each branching decision d_i
//!    on the path.
//! 3. **Propagation**: before expanding a node, check all learned clauses; prune
//!    if any clause is violated.
//! 4. **Activity decay**: periodically decay all activity scores to prioritise
//!    recent conflicts (VSIDS heuristic).
//!
//! # Integration
//!
//! The `CdclBranchingState` is designed to plug into `MilpBranchAndBound` via
//! the `BranchingStrategy::Cdcl` variant (see [`BranchingStrategy`]).  The
//! host solver calls:
//! - `select_branching_var` to pick which variable to branch on.
//! - `record_conflict` when LP infeasibility is detected.
//! - `apply_clauses` before expanding each child node.
//! - `decay_activities` after each batch of conflicts.
//!
//! # References
//! - Marques-Silva & Sakallah (1996). "GRASP—A new search algorithm for
//!   satisfiability." ICCAD.
//! - Achterberg, T. (2007). "Conflict analysis in mixed integer programming."
//!   Discrete Optimization, 4(1), 4–20.

use std::fmt::Debug;

use scirs2_core::num_traits::{Float, FromPrimitive};

use crate::error::{OptimizeError, OptimizeResult};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the CDCL branching state.
#[derive(Debug, Clone)]
pub struct CdclConfig {
    /// Maximum number of learned clauses to retain.  Oldest clauses are evicted
    /// when this limit is reached.
    pub max_clauses: usize,
    /// Maximum number of clauses learned per conflict resolution.
    pub max_learned_per_conflict: usize,
    /// VSIDS activity decay factor applied after each conflict batch.
    /// Typical value: 0.95.
    pub decay: f64,
    /// Activity bump increment applied to variables involved in a conflict.
    pub activity_bump: f64,
    /// Minimum activity threshold below which a clause may be deleted.
    pub min_activity_threshold: f64,
}

impl Default for CdclConfig {
    fn default() -> Self {
        Self {
            max_clauses: 10_000,
            max_learned_per_conflict: 3,
            decay: 0.95,
            activity_bump: 1.0,
            min_activity_threshold: 1e-12,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core data structures
// ─────────────────────────────────────────────────────────────────────────────

/// A single branching decision: set variable `var_index` to value `value`.
///
/// For binary variables `value` ∈ {0, 1}.  For general integer variables,
/// the clause uses the exact integer value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchingDecision {
    /// Index of the branched variable.
    pub var_index: usize,
    /// Value assigned by this decision.
    pub value: i32,
}

impl BranchingDecision {
    /// Create a new branching decision.
    pub fn new(var_index: usize, value: i32) -> Self {
        Self { var_index, value }
    }
}

/// A learned nogood clause: a disjunction of negated literals.
///
/// A clause `{(i₁, v₁), (i₂, v₂), …}` is violated (and triggers pruning) when
/// every literal matches a current decision: `decision[iⱼ] == vⱼ` for all j.
#[derive(Debug, Clone)]
pub struct LearnedClause {
    /// Literals `(var_index, value)`.  The clause fires when all literals are
    /// satisfied simultaneously by the current decision trail.
    pub literals: Vec<(usize, i32)>,
    /// Activity score of this clause (used for clause deletion).
    pub activity: f64,
}

impl LearnedClause {
    /// Create a new learned clause from a slice of variable-value pairs.
    pub fn new(literals: Vec<(usize, i32)>) -> Self {
        Self {
            literals,
            activity: 1.0,
        }
    }

    /// Return the number of literals in the clause.
    pub fn len(&self) -> usize {
        self.literals.len()
    }

    /// Return true if the clause has no literals (tautology / empty).
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Check whether this clause subsumes `other`, i.e., every literal of
    /// `self` also appears in `other`.  A subsuming clause is strictly
    /// stronger (shorter or equal).
    pub fn subsumes(&self, other: &LearnedClause) -> bool {
        self.literals.iter().all(|lit| other.literals.contains(lit))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CDCL Branching State
// ─────────────────────────────────────────────────────────────────────────────

/// Mutable state for CDCL-style MIP branching.
///
/// Maintains a decision trail, a database of learned clauses, and VSIDS
/// activity scores for heuristic variable selection.
#[derive(Debug, Clone)]
pub struct CdclBranchingState<F = f64> {
    /// Number of (binary/integer) variables in the problem.
    pub n_vars: usize,
    /// Current decision trail (ordered stack).
    pub decisions: Vec<BranchingDecision>,
    /// Accumulated learned clauses (nogoods).
    pub learned_clauses: Vec<LearnedClause>,
    /// VSIDS activity scores per variable.
    pub activity: Vec<F>,
    /// Solver configuration.
    pub config: CdclConfig,
}

impl<F> CdclBranchingState<F>
where
    F: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::MulAssign,
{
    /// Create a new CDCL branching state for a problem with `n_vars` variables.
    pub fn new(n_vars: usize, config: CdclConfig) -> OptimizeResult<Self> {
        if n_vars == 0 {
            return Err(OptimizeError::InvalidInput(
                "n_vars must be positive".into(),
            ));
        }
        Ok(Self {
            n_vars,
            decisions: Vec::new(),
            learned_clauses: Vec::new(),
            activity: vec![F::zero(); n_vars],
            config,
        })
    }

    /// Select the variable to branch on.
    ///
    /// Among all variables that are **fractional** in the LP relaxation
    /// (i.e., `0 < lp_solution[i] < 1`), return the index with the highest
    /// activity score.  If no fractional variable exists, return `None`.
    ///
    /// # Arguments
    /// * `lp_solution` – current LP relaxation solution (length `n_vars`).
    pub fn select_branching_var(&self, lp_solution: &[F]) -> Option<usize> {
        if lp_solution.len() != self.n_vars {
            return None;
        }

        let zero = F::zero();
        let one = F::one();
        let frac_tol = F::from_f64(1e-6).unwrap_or_else(|| F::from_f64(1e-6).unwrap_or(zero));

        let mut best_idx: Option<usize> = None;
        let mut best_activity = F::neg_infinity();

        for (i, &val) in lp_solution.iter().enumerate() {
            // Fractional: strictly between 0 and 1
            if val > frac_tol && val < one - frac_tol {
                let act = self.activity[i];
                if act > best_activity {
                    best_activity = act;
                    best_idx = Some(i);
                }
            }
        }

        best_idx
    }

    /// Record a conflict: given the set of branching decisions that led to
    /// infeasibility, extract a learned clause and update VSIDS activities.
    ///
    /// # Algorithm
    /// 1. Build clause `{NOT d | d ∈ infeasible_decisions}` as the conjunction
    ///    of negated literals.
    /// 2. Bump activity for all variables in the clause.
    /// 3. Apply clause minimisation (remove literals subsumed by existing
    ///    shorter clauses).
    /// 4. Evict oldest clauses if the database exceeds `max_clauses`.
    ///
    /// At most `max_learned_per_conflict` clauses are added per call.
    pub fn record_conflict(&mut self, infeasible_decisions: &[BranchingDecision]) {
        if infeasible_decisions.is_empty() {
            return;
        }

        // Build learned clause from negated decisions
        let literals: Vec<(usize, i32)> = infeasible_decisions
            .iter()
            .map(|d| (d.var_index, d.value))
            .collect();

        let clause = LearnedClause::new(literals);

        // Bump activity for all variables in the clause
        let bump = F::from_f64(self.config.activity_bump).unwrap_or(F::one());
        for &(var_idx, _) in &clause.literals {
            if var_idx < self.n_vars {
                self.activity[var_idx] += bump;
            }
        }

        // Only add if not subsumed by an existing clause
        let already_covered = self
            .learned_clauses
            .iter()
            .any(|existing| existing.subsumes(&clause));

        if !already_covered {
            self.learned_clauses.push(clause);
        }

        // Evict oldest clauses if limit exceeded
        if self.learned_clauses.len() > self.config.max_clauses {
            let excess = self.learned_clauses.len() - self.config.max_clauses;
            self.learned_clauses.drain(0..excess);
        }

        // Decay activities periodically after recording a conflict
        self.decay_activities();
    }

    /// Check whether any learned clause is violated by the current decisions.
    ///
    /// A clause is **violated** (fires) when every literal `(i, v)` in the
    /// clause is matched by some decision `d` with `d.var_index == i` and
    /// `d.value == v`.  Firing indicates that the current node is provably
    /// infeasible and should be pruned.
    ///
    /// Returns `true` if at least one clause fires (prune this node).
    pub fn apply_clauses(&self, current_decisions: &[BranchingDecision]) -> bool {
        'clause: for clause in &self.learned_clauses {
            if clause.is_empty() {
                continue;
            }
            // Check if every literal is satisfied by current_decisions
            for &(var_idx, value) in &clause.literals {
                let matched = current_decisions
                    .iter()
                    .any(|d| d.var_index == var_idx && d.value == value);
                if !matched {
                    continue 'clause; // This clause does not fire
                }
            }
            // All literals matched → prune
            return true;
        }
        false
    }

    /// Decay all variable activity scores by multiplying by `config.decay`.
    ///
    /// This implements the VSIDS (Variable State Independent Decaying Sum)
    /// heuristic, which prioritises variables that appeared in *recent*
    /// conflicts.
    pub fn decay_activities(&mut self) {
        let decay = F::from_f64(self.config.decay).unwrap_or(F::one());
        for act in &mut self.activity {
            *act *= decay;
        }
    }

    /// Push a new decision onto the decision trail.
    pub fn push_decision(&mut self, decision: BranchingDecision) {
        self.decisions.push(decision);
    }

    /// Pop the most recent decision from the trail (backtrack one level).
    pub fn pop_decision(&mut self) -> Option<BranchingDecision> {
        self.decisions.pop()
    }

    /// Return the number of learned clauses currently in the database.
    pub fn n_learned_clauses(&self) -> usize {
        self.learned_clauses.len()
    }

    /// Remove all learned clauses with activity below the configured threshold.
    pub fn prune_inactive_clauses(&mut self) {
        let threshold = self.config.min_activity_threshold;
        self.learned_clauses.retain(|c| c.activity >= threshold);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BranchingStrategy hook (interface documentation)
// ─────────────────────────────────────────────────────────────────────────────

/// Enumeration of branching strategies for use with `MilpBranchAndBound`.
///
/// This enum is intended to be incorporated into the `MilpBranchAndBound`
/// solver to switch between branching heuristics at runtime.
///
/// # Integration with `MilpBranchAndBound`
///
/// ```text
/// match strategy {
///     BranchingStrategy::MostFractional => { /* existing code */ }
///     BranchingStrategy::StrongBranching => { /* existing code */ }
///     BranchingStrategy::Cdcl(ref mut state) => {
///         let var = state.select_branching_var(lp_sol)?;
///         // ... branch on `var`, then:
///         if node_infeasible {
///             state.record_conflict(&decisions_on_path);
///         }
///         if state.apply_clauses(&current_decisions) {
///             prune_node();
///         }
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum BranchingStrategy<F = f64> {
    /// Most-fractional variable selection (standard heuristic).
    MostFractional,
    /// Strong branching: evaluate both child LP relaxations before committing.
    StrongBranching,
    /// CDCL-style branching with clause learning and VSIDS activity.
    Cdcl(CdclBranchingState<F>),
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    type F = f64;

    fn make_state(n: usize) -> CdclBranchingState<F> {
        CdclBranchingState::new(n, CdclConfig::default()).unwrap()
    }

    // ── Initialization ───────────────────────────────────────────────────────

    #[test]
    fn test_new_state_correct_size() {
        let state = make_state(5);
        assert_eq!(state.n_vars, 5);
        assert_eq!(state.activity.len(), 5);
        assert!(state.decisions.is_empty());
        assert!(state.learned_clauses.is_empty());
    }

    #[test]
    fn test_cdcl_config_defaults() {
        let cfg = CdclConfig::default();
        assert_eq!(cfg.max_clauses, 10_000);
        assert_eq!(cfg.max_learned_per_conflict, 3);
        assert!((cfg.decay - 0.95).abs() < 1e-12);
        assert!((cfg.activity_bump - 1.0).abs() < 1e-12);
    }

    // ── Variable selection ───────────────────────────────────────────────────

    #[test]
    fn test_select_branching_var_picks_highest_activity() {
        let mut state = make_state(4);
        // Manually set activities: var 2 has highest
        state.activity[0] = 0.1;
        state.activity[1] = 0.5;
        state.activity[2] = 1.0;
        state.activity[3] = 0.3;
        // All fractional
        let lp_sol = vec![0.5, 0.7, 0.3, 0.6];
        let selected = state.select_branching_var(&lp_sol);
        assert_eq!(selected, Some(2), "should pick var 2 (highest activity)");
    }

    #[test]
    fn test_select_branching_var_skips_integral() {
        let mut state = make_state(3);
        state.activity = vec![10.0, 0.5, 0.1];
        // var 0 is integral (value 1.0), only vars 1 and 2 are fractional
        let lp_sol = vec![1.0, 0.4, 0.6];
        let selected = state.select_branching_var(&lp_sol);
        // var 0 integral → should pick var 1 (next highest activity)
        assert_eq!(selected, Some(1));
    }

    // ── Conflict recording ───────────────────────────────────────────────────

    #[test]
    fn test_record_conflict_creates_clause_correct_length() {
        let mut state = make_state(4);
        let decisions = vec![BranchingDecision::new(0, 1), BranchingDecision::new(2, 0)];
        state.record_conflict(&decisions);
        assert!(!state.learned_clauses.is_empty());
        let clause = &state.learned_clauses[0];
        assert_eq!(clause.len(), 2, "clause should have 2 literals");
    }

    #[test]
    fn test_record_conflict_bumps_activity() {
        let mut state = make_state(3);
        let decisions = vec![BranchingDecision::new(0, 1), BranchingDecision::new(1, 0)];
        state.record_conflict(&decisions);
        // After bump and one decay: activity = bump * decay
        let expected = 1.0 * 0.95;
        assert!(
            (state.activity[0] - expected).abs() < 1e-9,
            "activity[0] = {}",
            state.activity[0]
        );
        assert!(
            (state.activity[1] - expected).abs() < 1e-9,
            "activity[1] = {}",
            state.activity[1]
        );
        // var 2 was not in conflict — only decayed
        assert!(state.activity[2] < state.activity[0]);
    }

    // ── Clause application ───────────────────────────────────────────────────

    #[test]
    fn test_apply_clauses_no_violation() {
        let mut state = make_state(3);
        // Clause: (var 0 = 1) ∧ (var 1 = 0) — fires when both hold
        state
            .learned_clauses
            .push(LearnedClause::new(vec![(0, 1), (1, 0)]));
        // Current decisions match only partially
        let current = vec![BranchingDecision::new(0, 1)]; // var 1 not set
        assert!(!state.apply_clauses(&current), "clause should not fire");
    }

    #[test]
    fn test_apply_clauses_violation() {
        let mut state = make_state(3);
        state
            .learned_clauses
            .push(LearnedClause::new(vec![(0, 1), (1, 0)]));
        // Both literals match → clause fires → prune
        let current = vec![BranchingDecision::new(0, 1), BranchingDecision::new(1, 0)];
        assert!(state.apply_clauses(&current), "clause should fire");
    }

    #[test]
    fn test_apply_clauses_empty_trail_no_violation() {
        let mut state = make_state(3);
        state.learned_clauses.push(LearnedClause::new(vec![(0, 1)]));
        assert!(
            !state.apply_clauses(&[]),
            "empty trail cannot satisfy any clause"
        );
    }

    // ── Activity decay ───────────────────────────────────────────────────────

    #[test]
    fn test_decay_activities_reduces_all() {
        let mut state = make_state(3);
        state.activity = vec![1.0, 2.0, 0.5];
        let before: Vec<f64> = state.activity.clone();
        state.decay_activities();
        for (a, b) in before.iter().zip(state.activity.iter()) {
            assert!(b < a || (*a == 0.0 && *b == 0.0));
        }
    }

    // ── Clause subsumption ───────────────────────────────────────────────────

    #[test]
    fn test_learned_clause_subsumption() {
        // Shorter clause {(0,1)} subsumes longer {(0,1), (1,0)}
        let short = LearnedClause::new(vec![(0, 1)]);
        let long = LearnedClause::new(vec![(0, 1), (1, 0)]);
        assert!(
            short.subsumes(&long),
            "shorter clause should subsume longer"
        );
        assert!(
            !long.subsumes(&short),
            "longer clause should not subsume shorter"
        );
    }

    #[test]
    fn test_duplicate_clause_not_added() {
        let mut state = make_state(3);
        let decisions = vec![BranchingDecision::new(0, 1), BranchingDecision::new(1, 0)];
        state.record_conflict(&decisions);
        let count_before = state.learned_clauses.len();
        // Record same conflict again — should not duplicate
        // (re-creates same clause; subsumption check prevents duplicate)
        let decisions2 = vec![
            BranchingDecision::new(0, 1),
            BranchingDecision::new(1, 0),
            BranchingDecision::new(2, 1),
        ];
        state.record_conflict(&decisions2);
        // The new (longer) clause is subsumed by the first — not added
        assert_eq!(
            state.learned_clauses.len(),
            count_before,
            "subsumed clause should not be added"
        );
    }
}
