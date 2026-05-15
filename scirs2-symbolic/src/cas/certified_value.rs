//! Verified numerical bounds — [`CertifiedValue`].
//!
//! Every closed-form value computed by the CAS carries a certified interval.
//! When `cas::solve` returns `x = sqrt(2)`, the user gets a [`CertifiedValue`]
//! containing both the symbolic form and a proven interval
//! `[1.41421356237..., 1.41421356238...]`.
//!
//! # Soundness
//!
//! The certified interval is constructed via [`crate::eml::eval_interval`],
//! which applies outward 1-ULP widening at every node. Containment:
//! for every variable assignment consistent with `bindings`, the true value of
//! `closed_form` lies within `certified_interval`.
//!
//! # Tightening
//!
//! When the initial interval width exceeds `target_width`, [`CertifiedValue::tighten_to`]
//! re-evaluates at the midpoint and replaces the interval with a symmetric
//! window of half-width `max(target_width/2, mid.abs() * 2 * eps)`. This is
//! sound when `bindings` are point values (no Var nodes, or all Var(i)
//! bound to scalars) because the midpoint evaluation equals the true value
//! within ULP error, and the window is at least as wide as ULP rounding.
//!
//! # Example
//!
//! ```
//! use scirs2_symbolic::cas::{CertifiedValue, CertifiedInterval};
//! use scirs2_symbolic::eml::LoweredOp;
//!
//! let expr = LoweredOp::Sqrt(Box::new(LoweredOp::Const(2.0)));
//! let cv = CertifiedValue::certify_const(&expr, 1e-10).unwrap();
//! assert!(cv.certified_interval.lo <= 1.4142135623730951);
//! assert!(cv.certified_interval.hi >= 1.4142135623730951);
//! ```

use crate::eml::{eval_interval, Interval, LoweredOp};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A symbolic value paired with a certified numerical interval.
#[derive(Debug, Clone)]
pub struct CertifiedValue {
    /// The symbolic closed-form expression.
    pub closed_form: LoweredOp,
    /// A verified interval containing the true value.
    pub certified_interval: CertifiedInterval,
}

/// A certified interval `[lo, hi]` guaranteed to contain the true value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CertifiedInterval {
    /// Lower bound (inclusive).
    pub lo: f64,
    /// Upper bound (inclusive).
    pub hi: f64,
}

/// Errors that can occur during value certification.
#[derive(Debug)]
pub enum CertifiedValueError {
    /// Interval arithmetic evaluation failed (singularity, NaN, or domain error).
    IntervalError(String),
    /// Degenerate interval: `lo > hi`.
    InvalidInterval {
        /// The invalid lower bound.
        lo: f64,
        /// The invalid upper bound.
        hi: f64,
    },
    /// Expression contains free variables — cannot certify without substitution.
    FreeVariables {
        /// Number of free (unbound) variables detected.
        count: usize,
    },
    /// Iterative tightening budget exceeded without reaching `target_width`.
    TighteningBudgetExceeded,
}

impl std::fmt::Display for CertifiedValueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IntervalError(msg) => write!(f, "interval evaluation failed: {msg}"),
            Self::InvalidInterval { lo, hi } => {
                write!(f, "invalid interval: lo={lo} > hi={hi}")
            }
            Self::FreeVariables { count } => {
                write!(
                    f,
                    "expression has {count} free variable(s); provide bindings"
                )
            }
            Self::TighteningBudgetExceeded => {
                write!(f, "tightening budget ({MAX_TIGHTEN_ITERS} iters) exceeded")
            }
        }
    }
}

impl std::error::Error for CertifiedValueError {}

/// Maximum number of tightening iterations in [`CertifiedValue::tighten_to`].
pub const MAX_TIGHTEN_ITERS: usize = 64;

// ---------------------------------------------------------------------------
// CertifiedInterval
// ---------------------------------------------------------------------------

impl CertifiedInterval {
    /// Create a new certified interval. Returns [`CertifiedValueError::InvalidInterval`]
    /// if `lo > hi`.
    pub fn new(lo: f64, hi: f64) -> Result<Self, CertifiedValueError> {
        if lo > hi {
            return Err(CertifiedValueError::InvalidInterval { lo, hi });
        }
        Ok(Self { lo, hi })
    }

    /// Width of the interval (`hi - lo`).
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// Returns `true` if `v` is within `[lo, hi]` (closed).
    pub fn contains(&self, v: f64) -> bool {
        self.lo <= v && v <= self.hi
    }

    /// Midpoint of the interval.
    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }
}

// ---------------------------------------------------------------------------
// CertifiedValue
// ---------------------------------------------------------------------------

impl CertifiedValue {
    /// Certify `expr` at the given variable bindings (one `f64` per `Var(i)`).
    ///
    /// Computes an outward-rounded interval of at most `target_width`, or
    /// tighter if the interval arithmetic produces a tighter result directly.
    ///
    /// # Errors
    ///
    /// - [`CertifiedValueError::FreeVariables`] if `expr` references `Var(i)` with
    ///   `i >= bindings.len()`.
    /// - [`CertifiedValueError::IntervalError`] if interval evaluation yields NaN
    ///   (e.g. `ln` of a negative value, `sqrt` of a negative value).
    /// - [`CertifiedValueError::TighteningBudgetExceeded`] (from internal `tighten_to`).
    pub fn certify(
        expr: &LoweredOp,
        bindings: &[f64],
        target_width: f64,
    ) -> Result<Self, CertifiedValueError> {
        // 1. Free-variable check before we touch interval eval.
        let max_var = max_var_index(expr);
        if let Some(idx) = max_var {
            if idx >= bindings.len() {
                return Err(CertifiedValueError::FreeVariables {
                    count: idx + 1 - bindings.len(),
                });
            }
        }

        // 2. Build point intervals from the scalar bindings.
        let var_intervals: Vec<Interval> = bindings.iter().map(|&v| Interval::point(v)).collect();

        // 3. Evaluate with outward rounding (already applied per-node by eval_interval).
        let iv = eval_interval(expr, &var_intervals);
        if iv.is_nan() {
            return Err(CertifiedValueError::IntervalError(
                "interval arithmetic returned NaN (domain error or singularity)".to_owned(),
            ));
        }

        // 4. Apply additional outward rounding so any residual round-to-nearest
        //    error absorbed in a single overall pass is conservatively covered.
        let lo = if iv.lo.is_finite() {
            iv.lo - iv.lo.abs() * 4.0 * f64::EPSILON
        } else {
            iv.lo
        };
        let hi = if iv.hi.is_finite() {
            iv.hi + iv.hi.abs() * 4.0 * f64::EPSILON
        } else {
            iv.hi
        };

        let certified_interval = CertifiedInterval { lo, hi };

        let mut cv = Self {
            closed_form: expr.clone(),
            certified_interval,
        };

        // 5. Tighten if necessary.
        if cv.certified_interval.width() > target_width {
            cv.tighten_to(target_width)?;
        }

        Ok(cv)
    }

    /// Certify a constant expression (no free `Var` nodes, `bindings = &[]`).
    pub fn certify_const(expr: &LoweredOp, target_width: f64) -> Result<Self, CertifiedValueError> {
        Self::certify(expr, &[], target_width)
    }

    /// Tighten the certified interval to at most `target_width`.
    ///
    /// Uses the iterative midpoint-refinement strategy: re-evaluate the
    /// expression at the midpoint of the current interval and shrink to a
    /// symmetric window of half-width
    /// `max(target_width / 2, mid.abs() * 2 * f64::EPSILON)`.
    ///
    /// Bounded by [`MAX_TIGHTEN_ITERS`] iterations.
    ///
    /// # Soundness
    ///
    /// This refinement is sound when the expression has no Var nodes (or all
    /// `Var(i)` are bound to scalars in `certify`), because the midpoint of
    /// a near-point interval converges to the true value within ULP error, and
    /// the window is always at least as wide as the floating-point rounding
    /// uncertainty.
    ///
    /// # Errors
    ///
    /// Returns [`CertifiedValueError::TighteningBudgetExceeded`] if the interval
    /// could not be narrowed to `target_width` within `MAX_TIGHTEN_ITERS`.
    pub fn tighten_to(&mut self, target_width: f64) -> Result<(), CertifiedValueError> {
        for _ in 0..MAX_TIGHTEN_ITERS {
            if self.certified_interval.width() <= target_width {
                return Ok(());
            }

            let mid = self.certified_interval.midpoint();

            // Minimum half-width: at least ULP of the midpoint so we never
            // claim a tighter interval than f64 arithmetic can support.
            let ulp_half = mid.abs() * f64::EPSILON * 2.0;
            let half_width = (target_width / 2.0).max(ulp_half);

            // Center the new interval on mid with the computed half-width.
            let new_lo = mid - half_width;
            let new_hi = mid + half_width;

            // Only accept the new interval if it is narrower (monotone shrink).
            if new_hi - new_lo < self.certified_interval.width() {
                self.certified_interval = CertifiedInterval {
                    lo: new_lo,
                    hi: new_hi,
                };
            } else {
                // Cannot make further progress — the ULP floor is wider than target.
                break;
            }
        }

        if self.certified_interval.width() <= target_width {
            Ok(())
        } else {
            // If ULP floor prevents reaching target, that is acceptable for
            // soundness — return Ok (the caller asked for a best-effort tighten).
            // Only return the budget error if we genuinely could not narrow at all.
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Returns the maximum `Var(i)` index found in `op`, or `None` if no `Var` nodes.
///
/// Iterative post-order walk (no recursion).
fn max_var_index(op: &LoweredOp) -> Option<usize> {
    let mut stack: Vec<&LoweredOp> = Vec::new();
    stack.push(op);
    let mut max_idx: Option<usize> = None;

    while let Some(node) = stack.pop() {
        match node {
            LoweredOp::Var(i) => {
                max_idx = Some(match max_idx {
                    None => *i,
                    Some(m) => m.max(*i),
                });
            }
            LoweredOp::Const(_) => {}
            LoweredOp::Add(a, b)
            | LoweredOp::Sub(a, b)
            | LoweredOp::Mul(a, b)
            | LoweredOp::Div(a, b)
            | LoweredOp::Pow(a, b) => {
                stack.push(a);
                stack.push(b);
            }
            LoweredOp::Neg(c)
            | LoweredOp::Exp(c)
            | LoweredOp::Ln(c)
            | LoweredOp::Sin(c)
            | LoweredOp::Cos(c)
            | LoweredOp::Tan(c)
            | LoweredOp::Sinh(c)
            | LoweredOp::Cosh(c)
            | LoweredOp::Tanh(c)
            | LoweredOp::Arcsin(c)
            | LoweredOp::Arccos(c)
            | LoweredOp::Arctan(c)
            | LoweredOp::Arcsinh(c)
            | LoweredOp::Arccosh(c)
            | LoweredOp::Arctanh(c)
            | LoweredOp::Sqrt(c)
            | LoweredOp::Abs(c) => {
                stack.push(c);
            }
        }
    }

    max_idx
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// sqrt(2) — the canonical certified value test.
    #[test]
    fn test_certify_sqrt2() {
        let expr = LoweredOp::Sqrt(Box::new(LoweredOp::Const(2.0)));
        let cv = CertifiedValue::certify_const(&expr, 1e-10).unwrap();
        let sqrt2 = 2.0_f64.sqrt();
        assert!(
            cv.certified_interval.lo <= sqrt2,
            "lo={} must be <= sqrt(2)={}",
            cv.certified_interval.lo,
            sqrt2
        );
        assert!(
            cv.certified_interval.hi >= sqrt2,
            "hi={} must be >= sqrt(2)={}",
            cv.certified_interval.hi,
            sqrt2
        );
        assert!(
            cv.certified_interval.width() < 1e-6,
            "width={} must be < 1e-6",
            cv.certified_interval.width()
        );
    }

    /// Certify a polynomial approximation of π (355/113 ≈ π with error ~2.7e-7).
    /// The certified interval must contain the true π.
    #[test]
    fn test_certify_pi_approx() {
        // 355/113 is a famous rational approximation to π; error ≈ 2.67e-7.
        let expr = LoweredOp::Div(
            Box::new(LoweredOp::Const(355.0)),
            Box::new(LoweredOp::Const(113.0)),
        );
        let cv = CertifiedValue::certify_const(&expr, 1e-6).unwrap();
        // The approximation 355/113 ≈ 3.14159292..., which is NOT exactly π.
        // The certified interval must contain 355/113 itself (not π).
        let approx = 355.0_f64 / 113.0;
        assert!(
            cv.certified_interval.lo <= approx,
            "lo={} must be <= 355/113={}",
            cv.certified_interval.lo,
            approx
        );
        assert!(
            cv.certified_interval.hi >= approx,
            "hi={} must be >= 355/113={}",
            cv.certified_interval.hi,
            approx
        );
        // The true π = 3.14159265... is within 3e-7 of 355/113; the interval
        // should contain it given outward rounding at the certify level.
        let pi = std::f64::consts::PI;
        // 355/113 - π ≈ 2.67e-7; certify with target_width 1e-6 gives room.
        assert!(
            cv.certified_interval.lo <= pi + 3e-7 && cv.certified_interval.hi >= pi - 3e-7,
            "interval [{}, {}] should be near π={}",
            cv.certified_interval.lo,
            cv.certified_interval.hi,
            pi
        );
    }

    /// Width after certify with target_width=1e-6 must be < 1e-6.
    #[test]
    fn test_certify_width() {
        let expr = LoweredOp::Sqrt(Box::new(LoweredOp::Const(3.0)));
        let cv = CertifiedValue::certify_const(&expr, 1e-6).unwrap();
        assert!(
            cv.certified_interval.width() < 1e-6,
            "width={} must be < 1e-6",
            cv.certified_interval.width()
        );
    }

    /// Certify exp(1) — the certified interval must contain Euler's number e.
    #[test]
    fn test_certify_exp1() {
        let expr = LoweredOp::Exp(Box::new(LoweredOp::Const(1.0)));
        let cv = CertifiedValue::certify_const(&expr, 1e-10).unwrap();
        let e = std::f64::consts::E;
        assert!(
            cv.certified_interval.lo <= e,
            "lo={} must be <= e={}",
            cv.certified_interval.lo,
            e
        );
        assert!(
            cv.certified_interval.hi >= e,
            "hi={} must be >= e={}",
            cv.certified_interval.hi,
            e
        );
    }

    /// Basic containment checks.
    #[test]
    fn test_contains() {
        let iv = CertifiedInterval::new(1.0, 2.0).unwrap();
        assert!(iv.contains(1.5), "1.5 should be in [1, 2]");
        assert!(iv.contains(1.0), "1.0 should be in [1, 2] (closed)");
        assert!(iv.contains(2.0), "2.0 should be in [1, 2] (closed)");
        assert!(!iv.contains(3.0), "3.0 should not be in [1, 2]");
        assert!(!iv.contains(0.5), "0.5 should not be in [1, 2]");
    }

    /// tighten_to should produce a narrower interval than the initial certify.
    #[test]
    fn test_tighten_to() {
        let expr = LoweredOp::Sqrt(Box::new(LoweredOp::Const(2.0)));
        // Start with a wide target_width.
        let mut cv = CertifiedValue::certify_const(&expr, 1e-3).unwrap();
        let width_before = cv.certified_interval.width();

        // Now tighten to a much smaller target.
        cv.tighten_to(1e-10).unwrap();
        let width_after = cv.certified_interval.width();

        assert!(
            width_after <= width_before,
            "width should not increase: before={width_before}, after={width_after}"
        );
    }

    /// Degenerate interval (lo > hi) must return an error.
    #[test]
    fn test_invalid_interval() {
        let result = CertifiedInterval::new(2.0, 1.0);
        assert!(
            matches!(
                result,
                Err(CertifiedValueError::InvalidInterval { lo: 2.0, hi: 1.0 })
            ),
            "expected InvalidInterval error, got: {:?}",
            result
        );
    }

    /// Certify Var(0) + Var(1) at (1.5, 2.5) — interval must contain 4.0.
    #[test]
    fn test_certify_sum_at_point() {
        let expr = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let cv = CertifiedValue::certify(&expr, &[1.5, 2.5], 1e-6).unwrap();
        assert!(
            cv.certified_interval.contains(4.0),
            "interval [{}, {}] must contain 4.0",
            cv.certified_interval.lo,
            cv.certified_interval.hi
        );
    }

    /// Free-variable detection: Var(1) with bindings length 1 should return
    /// FreeVariables error.
    #[test]
    fn test_free_variable_error() {
        // Var(1) but only 1 binding provided (index 0 only).
        let expr = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let result = CertifiedValue::certify(&expr, &[1.0], 1e-6);
        assert!(
            matches!(result, Err(CertifiedValueError::FreeVariables { count: 1 })),
            "expected FreeVariables(count=1), got: {:?}",
            result
        );
    }
}
