//! Constrained symbolic regression.
//!
//! Wraps [`fn@discover`] with constraint filtering — formulas that violate
//! the supplied constraints are penalized in fitness or rejected outright.
//!
//! # Constraint kinds
//! - `Constraint::BoundedOutput` — predictions must lie in `[lo, hi]`
//! - `Constraint::Monotonic` — formula's gradient w.r.t. a variable must
//!   have a fixed sign at all training samples
//! - `Constraint::DimensionMatch` — formula's dimensional units must match
//!   the given target dimension (uses `units::infer_dimension`)
//!
//! # Phase 2 follow-ups
//! - SMT-certified constraint discharge via `crate::cas::smt`
//! - Symbolic (not just sample-based) monotonicity checks
//! - Boolean combinations of constraints
//!
//! # Examples
//!
//! ```
//! use ndarray::{Array1, Array2};
//! use scirs2_symbolic::regression::{with_constraints, ConstrainedConfig, Constraint, SrConfig};
//!
//! let xs: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
//! let features = Array2::from_shape_vec((30, 1), xs.clone()).expect("shape");
//! let targets = Array1::from_vec(xs);
//!
//! let config = ConstrainedConfig::new(
//!     SrConfig::default().with_max_iter(10),
//!     vec![Constraint::BoundedOutput { lo: -1.0, hi: 5.0 }],
//! );
//! let results = with_constraints(features.view(), targets.view(), &config);
//! assert!(!results.is_empty());
//! ```

use crate::eml::eval::{eval_real, EvalCtx};
use crate::eml::{grad, LoweredOp};
use crate::regression::{discover, DiscoveredFormula, SrConfig};
use ndarray::{ArrayView1, ArrayView2};

/// A single constraint on the discovered formula.
#[derive(Clone, Debug)]
pub enum Constraint {
    /// Predictions must lie in `[lo, hi]` at all training samples.
    BoundedOutput {
        /// Lower bound (inclusive).
        lo: f64,
        /// Upper bound (inclusive).
        hi: f64,
    },

    /// The formula's partial derivative w.r.t. variable `wrt` must be
    /// non-negative (if `increasing`) or non-positive at all training samples.
    Monotonic {
        /// Index of the variable to differentiate against.
        wrt: usize,
        /// `true` for non-negative gradient, `false` for non-positive.
        increasing: bool,
    },

    /// Enforce that the expression's physical dimension matches the target.
    ///
    /// `var_dims[i]` gives the dimension of `Var(i)` in the expression.
    /// Expressions with mismatched or incommensurate dimensions incur one
    /// violation (or are rejected outright in strict mode).
    ///
    /// Uses [`crate::units::infer_dimension`] for purely structural inference.
    DimensionMatch {
        /// The expected dimension of the formula's output.
        target: crate::units::Dimension,
        /// Dimension of `Var(i)` for each variable index used in the formula.
        var_dims: Vec<crate::units::Dimension>,
    },
}

/// Configuration for constrained symbolic-regression search.
#[derive(Clone, Debug)]
pub struct ConstrainedConfig {
    /// Underlying SR config (drives the candidate-generation engine).
    pub sr_config: SrConfig,
    /// Constraints that candidate formulas must satisfy.
    pub constraints: Vec<Constraint>,
    /// Penalty added to combined fitness per violated constraint
    /// (when not in strict mode).
    pub violation_penalty: f64,
    /// If true, reject violators outright (vs. penalize). Default: `false`.
    pub strict: bool,
}

impl Default for ConstrainedConfig {
    fn default() -> Self {
        Self {
            sr_config: SrConfig::default(),
            constraints: Vec::new(),
            violation_penalty: 1e6,
            strict: false,
        }
    }
}

impl ConstrainedConfig {
    /// New config with the given SR config and constraints.
    pub fn new(sr_config: SrConfig, constraints: Vec<Constraint>) -> Self {
        Self {
            sr_config,
            constraints,
            violation_penalty: 1e6,
            strict: false,
        }
    }

    /// Builder: strict mode (reject violators rather than penalize).
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Builder: violation penalty (added to combined fitness per violated
    /// constraint when `strict == false`).
    pub fn with_penalty(mut self, p: f64) -> Self {
        self.violation_penalty = p;
        self
    }

    /// Builder: append a single constraint.
    pub fn with_constraint(mut self, c: Constraint) -> Self {
        self.constraints.push(c);
        self
    }

    /// Builder: require the discovered formula to have the given physical dimension.
    ///
    /// `target` is the expected dimension of the formula's output.
    /// `var_dims` maps each `Var(i)` to its physical dimension.
    ///
    /// Internally appends a [`Constraint::DimensionMatch`].
    pub fn with_dimension(
        self,
        target: crate::units::Dimension,
        var_dims: Vec<crate::units::Dimension>,
    ) -> Self {
        self.with_constraint(Constraint::DimensionMatch { target, var_dims })
    }
}

/// Discover formulas subject to constraints.
///
/// Internally calls [`fn@discover`] (with a widened `top_n` to feed the
/// post-filter) and then re-scores or filters results based on
/// constraint satisfaction.
///
/// In `strict` mode, violators are removed; otherwise their combined fitness
/// is penalized by `violation_penalty * n_violations` and they remain in
/// the result set.
///
/// Returns up to `config.sr_config.top_n` formulas, ranked by combined
/// fitness (lower is better).
pub fn with_constraints(
    features: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
    config: &ConstrainedConfig,
) -> Vec<DiscoveredFormula> {
    // Widen the underlying search's output pool 4× so the post-filter
    // has more candidates to work with after rejection / re-scoring.
    let mut widened = config.sr_config.clone();
    widened.top_n = (config.sr_config.top_n * 4).max(8);

    let candidates = discover(features, targets, &widened);
    let bindings: Vec<Vec<f64>> = (0..features.nrows())
        .map(|i| (0..features.ncols()).map(|j| features[(i, j)]).collect())
        .collect();

    let mut scored: Vec<DiscoveredFormula> = candidates
        .into_iter()
        .filter_map(|formula| score_with_constraints(formula, &bindings, config))
        .collect();

    scored.sort_by(|a, b| {
        a.fitness
            .combined
            .partial_cmp(&b.fitness.combined)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(config.sr_config.top_n);
    scored
}

/// Score a candidate against the constraint set.
///
/// In strict mode, returns `None` when any constraint is violated.
/// In non-strict mode, returns `Some` with the combined fitness penalized
/// by `violation_penalty * n_violations`.
fn score_with_constraints(
    mut formula: DiscoveredFormula,
    bindings: &[Vec<f64>],
    config: &ConstrainedConfig,
) -> Option<DiscoveredFormula> {
    let violations = count_violations(&formula.op, bindings, &config.constraints);
    if violations > 0 {
        if config.strict {
            return None;
        }
        let mut fit = formula.fitness;
        fit.combined += config.violation_penalty * (violations as f64);
        formula.fitness = fit;
    }
    Some(formula)
}

/// Count how many of `constraints` are violated by `op` on `bindings`.
///
/// Each constraint contributes at most 1 to the count — early-breaking on
/// the first sample-level violation keeps the penalty bounded and the
/// strict-mode check fast.
fn count_violations(op: &LoweredOp, bindings: &[Vec<f64>], constraints: &[Constraint]) -> usize {
    let mut count = 0;
    for constraint in constraints {
        match constraint {
            Constraint::BoundedOutput { lo, hi } => {
                for vars in bindings {
                    let ctx = EvalCtx::new(vars);
                    match eval_real(op, &ctx) {
                        Ok(v) => {
                            if !v.is_finite() || v < *lo || v > *hi {
                                count += 1;
                                break;
                            }
                        }
                        Err(_) => {
                            count += 1;
                            break;
                        }
                    }
                }
            }
            Constraint::Monotonic { wrt, increasing } => {
                let g = grad(op, *wrt);
                for vars in bindings {
                    let ctx = EvalCtx::new(vars);
                    match eval_real(&g, &ctx) {
                        Ok(d) => {
                            if !d.is_finite() {
                                count += 1;
                                break;
                            }
                            if *increasing && d < 0.0 {
                                count += 1;
                                break;
                            }
                            if !*increasing && d > 0.0 {
                                count += 1;
                                break;
                            }
                        }
                        Err(_) => {
                            count += 1;
                            break;
                        }
                    }
                }
            }
            Constraint::DimensionMatch { target, var_dims } => {
                match crate::units::infer_dimension(op, var_dims) {
                    Ok(ref dim) if dim == target => {}
                    _ => count += 1,
                }
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::eval::{eval_real, EvalCtx};
    use crate::regression::SrConfig;
    use ndarray::{Array1, Array2};

    #[test]
    fn unconstrained_matches_discover() {
        let xs: Vec<f64> = (0..30).map(|i| (i as f64) * 0.1).collect();
        let features = Array2::from_shape_vec((30, 1), xs.clone()).expect("shape");
        let targets = Array1::from_vec(xs);

        let config = ConstrainedConfig::default();
        let results = with_constraints(features.view(), targets.view(), &config);

        assert!(!results.is_empty());
        assert!(results[0].fitness.mse < 1e-10);
    }

    #[test]
    fn bounded_output_rejects_unbounded() {
        // Target = x in [0, 2.9]; constraint: predictions must be in [0, 5].
        let xs: Vec<f64> = (0..30).map(|i| (i as f64) * 0.1).collect();
        let features = Array2::from_shape_vec((30, 1), xs.clone()).expect("shape");
        let targets = Array1::from_vec(xs.clone());

        let config = ConstrainedConfig::new(
            SrConfig::default().with_max_iter(15).with_top_n(3),
            vec![Constraint::BoundedOutput { lo: 0.0, hi: 5.0 }],
        )
        .with_strict(true);
        let results = with_constraints(features.view(), targets.view(), &config);

        // All returned formulas must produce values in [0, 5] at every sample.
        for formula in &results {
            for x in &xs {
                let r = eval_real(&formula.op, &EvalCtx::new(&[*x])).unwrap_or(f64::NAN);
                if r.is_finite() {
                    assert!(
                        (0.0..=5.0).contains(&r),
                        "formula {:?} produced {} at x={}",
                        formula.op,
                        r,
                        x
                    );
                }
            }
        }
    }

    #[test]
    fn monotonic_increasing_filters() {
        // Target = x (monotonically increasing); constraint: dx/dx >= 0.
        let xs: Vec<f64> = (0..30).map(|i| (i as f64) * 0.1).collect();
        let features = Array2::from_shape_vec((30, 1), xs.clone()).expect("shape");
        let targets = Array1::from_vec(xs);

        let config = ConstrainedConfig::new(
            SrConfig::default().with_max_iter(10).with_top_n(3),
            vec![Constraint::Monotonic {
                wrt: 0,
                increasing: true,
            }],
        );
        let results = with_constraints(features.view(), targets.view(), &config);

        // `Var(0)` trivially satisfies dx/dx = 1 >= 0, so we must get something.
        assert!(!results.is_empty());
    }

    #[test]
    fn strict_mode_can_filter_violators() {
        let xs: Vec<f64> = (0..10).map(|i| (i as f64) * 0.1).collect();
        let features = Array2::from_shape_vec((10, 1), xs.clone()).expect("shape");
        let targets = Array1::from_vec(vec![0.5; 10]);

        // Bound that excludes constants outside [100, 200] and most other shapes.
        let config = ConstrainedConfig::new(
            SrConfig::default().with_max_iter(5),
            vec![Constraint::BoundedOutput {
                lo: 100.0,
                hi: 200.0,
            }],
        )
        .with_strict(true);

        let results = with_constraints(features.view(), targets.view(), &config);
        // Strict mode: every survivor must satisfy the bound at every sample.
        for f in &results {
            for x in &xs {
                let r = eval_real(&f.op, &EvalCtx::new(&[*x])).unwrap_or(f64::NAN);
                if r.is_finite() {
                    assert!(
                        (100.0..=200.0).contains(&r),
                        "strict-mode survivor produced out-of-bounds {}",
                        r
                    );
                }
            }
        }
    }

    #[test]
    fn penalty_mode_keeps_violators_but_penalizes() {
        let xs: Vec<f64> = (0..10).map(|i| (i as f64) * 0.1).collect();
        let features = Array2::from_shape_vec((10, 1), xs.clone()).expect("shape");
        let targets = Array1::from_vec(xs);

        // Tight bound that excludes the natural identity-fit.
        let config = ConstrainedConfig::new(
            SrConfig::default().with_max_iter(5).with_top_n(3),
            vec![Constraint::BoundedOutput { lo: 0.5, hi: 0.7 }],
        )
        .with_penalty(1000.0);

        let results = with_constraints(features.view(), targets.view(), &config);
        // Non-strict: violators stay (penalized), so we expect a non-empty list.
        assert!(!results.is_empty());
    }

    #[test]
    fn config_defaults_have_no_constraints() {
        let config = ConstrainedConfig::default();
        assert!(config.constraints.is_empty());
        assert!(!config.strict);
        assert!(config.violation_penalty > 0.0);
    }

    #[test]
    fn dimension_match_accepts_dimensionless_formula() {
        use crate::units::Dimension;
        // Target = constant (dimensionless); expect dimensionless formula to pass.
        let xs: Vec<f64> = (0..20).map(|i| (i as f64) * 0.1).collect();
        let features = Array2::from_shape_vec((20, 1), xs.clone()).expect("shape");
        let targets = Array1::from_vec(xs);

        // Var(0) is dimensionless; target is also dimensionless.
        let config =
            ConstrainedConfig::new(SrConfig::default().with_max_iter(5).with_top_n(4), vec![])
                .with_dimension(Dimension::dimensionless(), vec![Dimension::dimensionless()]);

        let results = with_constraints(features.view(), targets.view(), &config);
        assert!(!results.is_empty());
    }

    #[test]
    fn dimension_match_constraint_count_violations() {
        use crate::eml::LoweredOp;
        use crate::units::Dimension;
        // Var(0) has dimension [m]; Var(1) has dimension [s].
        // Add(Var(0), Var(1)) has a dimension mismatch → UnitError::Mismatch.
        // DimensionMatch { target: length, var_dims: [m, s] } → 1 violation.
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let constraints = vec![Constraint::DimensionMatch {
            target: Dimension::length(),
            var_dims: vec![Dimension::length(), Dimension::time()],
        }];
        let bindings: Vec<Vec<f64>> = vec![]; // no sample bindings needed for unit check
        let violations = count_violations(&op, &bindings, &constraints);
        assert_eq!(
            violations, 1,
            "dimension mismatch should count as 1 violation"
        );
    }

    #[test]
    fn dimension_match_no_violation_when_correct() {
        use crate::eml::LoweredOp;
        use crate::units::Dimension;
        // Var(0) has dimension [m]; Add(Var(0), Var(0)) = [m] → matches target [m].
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(0)));
        let constraints = vec![Constraint::DimensionMatch {
            target: Dimension::length(),
            var_dims: vec![Dimension::length()],
        }];
        let bindings: Vec<Vec<f64>> = vec![];
        let violations = count_violations(&op, &bindings, &constraints);
        assert_eq!(violations, 0, "correct dimension should have 0 violations");
    }

    #[test]
    fn with_dimension_builder_appends_constraint() {
        use crate::units::Dimension;
        let config = ConstrainedConfig::default().with_dimension(
            Dimension::velocity(),
            vec![Dimension::length(), Dimension::time()],
        );
        assert_eq!(config.constraints.len(), 1);
        assert!(matches!(
            &config.constraints[0],
            Constraint::DimensionMatch { .. }
        ));
    }
}
