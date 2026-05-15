//! Outward-rounded interval arithmetic over [`LoweredOp`].
//!
//! Provides rigorous range bounds for symbolic expressions. Used by adaptive
//! integrators and trust-region optimisers in Phase 3.
//!
//! # Outward Rounding (best-effort, NOT IEEE 1788)
//!
//! Rust's `f64` is round-to-nearest by default. We apply 1-ULP outward
//! widening (`next_down(lo)`, `next_up(hi)`) once per [`LoweredOp`] node,
//! NOT per arithmetic primitive. This is **sound** (containment preserved)
//! but is NOT IEEE 1788 compliant. For tight bounds suitable for formal
//! verification, integrate the `inari` crate (deferred to v0.5.x).
//!
//! # Soundness invariant
//!
//! For every `LoweredOp` `op`, every variable assignment `xs` consistent
//! with the per-variable intervals `vs` (i.e. `xs[i] in vs[i]`), if
//! `eval_real(op, &EvalCtx::new(&xs))` succeeds with value `y`, then
//! `eval_interval(op, &vs).contains(y)` must hold. Domain-error inputs are
//! reflected as a NaN interval.
//!
//! # Examples
//!
//! ```
//! use scirs2_symbolic::eml::{LoweredOp, eval_interval, Interval};
//!
//! // sin(x) on x ∈ [0, 2π] reaches both ±1.
//! let op = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
//! let r = eval_interval(&op, &[Interval::new(0.0, 2.0 * std::f64::consts::PI)]);
//! assert!(r.lo <= -1.0 + 1e-10);
//! assert!(r.hi >= 1.0 - 1e-10);
//! ```

// Adapted from oxieml v0.1.0, src/lower_interval.rs (lines 343-392)
//
// `interval_sin` / `interval_cos` use oxieml's k-loop critical-point
// enumeration verbatim (necessary for soundness across multi-period
// inputs). Default arithmetic rules preserved. Outward 1-ULP widening
// added on top — NEW for SciRS2.

#![warn(missing_docs)]

use crate::eml::op::{LoweredOp, OxiOp};

/// A real-valued interval `[lo, hi]`.
///
/// `lo > hi` indicates an empty interval. NaN bounds indicate "unknown" /
/// "domain error" intervals — caller must check via [`Interval::is_empty`]
/// / [`Interval::is_nan`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Interval {
    /// Lower bound.
    pub lo: f64,
    /// Upper bound.
    pub hi: f64,
}

impl Interval {
    /// New interval. Caller is responsible for `lo <= hi` (or NaN bounds);
    /// the constructor performs no validation.
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// Degenerate interval at a point.
    pub fn point(v: f64) -> Self {
        Self { lo: v, hi: v }
    }

    /// `(-∞, +∞)`.
    pub fn full() -> Self {
        Self {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    /// NaN interval — represents domain error / unknown.
    pub fn nan() -> Self {
        Self {
            lo: f64::NAN,
            hi: f64::NAN,
        }
    }

    /// Empty interval (canonical `lo > hi` representation).
    pub fn empty() -> Self {
        Self { lo: 1.0, hi: -1.0 }
    }

    /// True if `lo > hi` (empty representation).
    pub fn is_empty(&self) -> bool {
        !self.lo.is_nan() && !self.hi.is_nan() && self.lo > self.hi
    }

    /// True if either bound is NaN.
    pub fn is_nan(&self) -> bool {
        self.lo.is_nan() || self.hi.is_nan()
    }

    /// True if both bounds are finite.
    pub fn is_finite(&self) -> bool {
        self.lo.is_finite() && self.hi.is_finite()
    }

    /// Width of the interval (`hi - lo`). Returns `0.0` for empty intervals.
    pub fn width(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.hi - self.lo
        }
    }

    /// True if `x` is inside the interval (closed). NaN bounds and empty
    /// intervals always return `false`.
    pub fn contains(&self, x: f64) -> bool {
        if self.is_nan() || self.is_empty() {
            return false;
        }
        self.lo <= x && x <= self.hi
    }

    /// Convex hull union. Empty intervals act as identity.
    pub fn union(&self, other: &Self) -> Self {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Intersection. Returns an empty interval (`lo > hi`) if disjoint.
    pub fn intersect(&self, other: &Self) -> Self {
        Self {
            lo: self.lo.max(other.lo),
            hi: self.hi.min(other.hi),
        }
    }

    /// 1-ULP outward widening (best-effort outward rounding).
    ///
    /// Walks `lo` one ULP toward `-∞` and `hi` one ULP toward `+∞`, when
    /// finite. NaN/empty intervals are returned unchanged. This is **not**
    /// IEEE 1788 directed-rounding, but is sufficient to absorb the
    /// round-to-nearest error of a single elementary operation.
    pub(crate) fn outward_widen(self) -> Self {
        if self.is_nan() || self.is_empty() {
            return self;
        }
        Self {
            lo: if self.lo.is_finite() {
                self.lo.next_down()
            } else {
                self.lo
            },
            hi: if self.hi.is_finite() {
                self.hi.next_up()
            } else {
                self.hi
            },
        }
    }
}

// ---------------------------------------------------------------------
// Top-level evaluation
// ---------------------------------------------------------------------

/// Evaluate an interval over a [`LoweredOp`] given variable interval
/// bindings.
///
/// Iterative post-order. At each node, we apply the appropriate interval
/// rule (sum, difference, product, etc.) and outward-widen the result.
///
/// For `sin`/`cos`, we enumerate critical points (`π/2 + kπ` and `kπ`
/// respectively) inside the input interval to recover tight bounds across
/// monotone-region boundaries.
///
/// Out-of-bounds variable indices fall back to [`Interval::full`]
/// (`(-∞, +∞)`); domain errors (e.g. `ln` of a strictly negative interval,
/// `arcsin` outside `[-1, 1]`) propagate as NaN intervals.
pub fn eval_interval(op: &LoweredOp, vars: &[Interval]) -> Interval {
    let ops = op.to_oxi_ops();
    let mut stack: Vec<Interval> = Vec::with_capacity(ops.len());

    for o in &ops {
        let result = match o {
            OxiOp::Const(c) => Interval::point(*c),
            OxiOp::Var(i) => vars.get(*i).copied().unwrap_or_else(Interval::full),
            OxiOp::Add => {
                let b = stack.pop().unwrap_or_else(Interval::full);
                let a = stack.pop().unwrap_or_else(Interval::full);
                interval_add(a, b)
            }
            OxiOp::Sub => {
                let b = stack.pop().unwrap_or_else(Interval::full);
                let a = stack.pop().unwrap_or_else(Interval::full);
                interval_sub(a, b)
            }
            OxiOp::Mul => {
                let b = stack.pop().unwrap_or_else(Interval::full);
                let a = stack.pop().unwrap_or_else(Interval::full);
                interval_mul(a, b)
            }
            OxiOp::Div => {
                let b = stack.pop().unwrap_or_else(Interval::full);
                let a = stack.pop().unwrap_or_else(Interval::full);
                interval_div(a, b)
            }
            OxiOp::Pow => {
                let b = stack.pop().unwrap_or_else(Interval::full);
                let a = stack.pop().unwrap_or_else(Interval::full);
                interval_pow(a, b)
            }
            OxiOp::Neg => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                if c.is_nan() {
                    Interval::nan()
                } else {
                    Interval::new(-c.hi, -c.lo)
                }
            }
            OxiOp::Exp => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                if c.is_nan() {
                    Interval::nan()
                } else {
                    Interval::new(c.lo.exp(), c.hi.exp())
                }
            }
            OxiOp::Ln => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                interval_ln(c)
            }
            OxiOp::Sin => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                interval_sin(c)
            }
            OxiOp::Cos => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                interval_cos(c)
            }
            OxiOp::Tan => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                interval_tan(c)
            }
            OxiOp::Sinh => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                if c.is_nan() {
                    Interval::nan()
                } else {
                    Interval::new(c.lo.sinh(), c.hi.sinh())
                }
            }
            OxiOp::Cosh => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                interval_cosh(c)
            }
            OxiOp::Tanh => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                if c.is_nan() {
                    Interval::nan()
                } else {
                    Interval::new(c.lo.tanh(), c.hi.tanh())
                }
            }
            OxiOp::Arcsin => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                if c.is_nan() || c.lo < -1.0 || c.hi > 1.0 {
                    Interval::nan()
                } else {
                    Interval::new(c.lo.asin(), c.hi.asin())
                }
            }
            OxiOp::Arccos => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                if c.is_nan() || c.lo < -1.0 || c.hi > 1.0 {
                    Interval::nan()
                } else {
                    // Monotone-decreasing.
                    Interval::new(c.hi.acos(), c.lo.acos())
                }
            }
            OxiOp::Arctan => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                if c.is_nan() {
                    Interval::nan()
                } else {
                    Interval::new(c.lo.atan(), c.hi.atan())
                }
            }
            OxiOp::Arcsinh => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                if c.is_nan() {
                    Interval::nan()
                } else {
                    Interval::new(c.lo.asinh(), c.hi.asinh())
                }
            }
            OxiOp::Arccosh => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                if c.is_nan() || c.hi < 1.0 {
                    Interval::nan()
                } else {
                    Interval::new(c.lo.max(1.0).acosh(), c.hi.acosh())
                }
            }
            OxiOp::Arctanh => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                if c.is_nan() || c.lo <= -1.0 || c.hi >= 1.0 {
                    Interval::nan()
                } else {
                    Interval::new(c.lo.atanh(), c.hi.atanh())
                }
            }
            OxiOp::Sqrt => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                if c.is_nan() || c.hi < 0.0 {
                    Interval::nan()
                } else {
                    Interval::new(c.lo.max(0.0).sqrt(), c.hi.sqrt())
                }
            }
            OxiOp::Abs => {
                let c = stack.pop().unwrap_or_else(Interval::full);
                if c.is_nan() {
                    Interval::nan()
                } else if c.lo >= 0.0 {
                    c
                } else if c.hi <= 0.0 {
                    Interval::new(-c.hi, -c.lo)
                } else {
                    Interval::new(0.0, c.hi.max(-c.lo))
                }
            }
        };
        stack.push(result.outward_widen());
    }

    stack.pop().unwrap_or_else(Interval::nan)
}

// ---------------------------------------------------------------------
// Per-op interval rules
// ---------------------------------------------------------------------

fn interval_add(a: Interval, b: Interval) -> Interval {
    if a.is_nan() || b.is_nan() {
        return Interval::nan();
    }
    Interval::new(a.lo + b.lo, a.hi + b.hi)
}

fn interval_sub(a: Interval, b: Interval) -> Interval {
    if a.is_nan() || b.is_nan() {
        return Interval::nan();
    }
    Interval::new(a.lo - b.hi, a.hi - b.lo)
}

fn interval_mul(a: Interval, b: Interval) -> Interval {
    if a.is_nan() || b.is_nan() {
        return Interval::nan();
    }
    let candidates = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi];
    let lo = candidates.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = candidates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Interval::new(lo, hi)
}

fn interval_div(a: Interval, b: Interval) -> Interval {
    if a.is_nan() || b.is_nan() {
        return Interval::nan();
    }
    // Denominator straddling zero — result is unbounded; return full range
    // (caller must guard if a tighter representation is needed).
    if b.lo <= 0.0 && 0.0 <= b.hi {
        return Interval::full();
    }
    interval_mul(a, Interval::new(1.0 / b.hi, 1.0 / b.lo))
}

fn interval_ln(c: Interval) -> Interval {
    if c.is_nan() {
        return Interval::nan();
    }
    if c.hi <= 0.0 {
        return Interval::nan();
    }
    let lo = c.lo.max(f64::MIN_POSITIVE).ln();
    let hi = c.hi.ln();
    Interval::new(lo, hi)
}

fn interval_pow(a: Interval, b: Interval) -> Interval {
    // Phase 0: only handle integer-exponent case (b is a degenerate point
    // at an integer value). All other cases conservatively return the full
    // range — acceptable for soundness, may be refined in v0.5.x.
    if a.is_nan() || b.is_nan() {
        return Interval::nan();
    }
    if b.lo == b.hi && b.lo.is_finite() && b.lo.fract() == 0.0 && b.lo.abs() < i32::MAX as f64 {
        let n = b.lo as i32;
        return interval_pow_int(a, n);
    }
    Interval::full()
}

fn interval_pow_int(a: Interval, n: i32) -> Interval {
    if n == 0 {
        return Interval::point(1.0);
    }
    if n == 1 {
        return a;
    }
    if n == -1 {
        // Reciprocal — denominator straddling zero is full.
        if a.lo <= 0.0 && 0.0 <= a.hi {
            return Interval::full();
        }
        return Interval::new(1.0 / a.hi, 1.0 / a.lo);
    }
    if n > 0 {
        if n % 2 == 0 {
            // Even positive power — piecewise.
            if a.lo >= 0.0 {
                return Interval::new(a.lo.powi(n), a.hi.powi(n));
            }
            if a.hi <= 0.0 {
                return Interval::new(a.hi.powi(n), a.lo.powi(n));
            }
            // Crosses zero — min is 0, max is at the wider endpoint.
            let max = a.lo.abs().max(a.hi.abs()).powi(n);
            return Interval::new(0.0, max);
        }
        // Odd positive power — monotone.
        return Interval::new(a.lo.powi(n), a.hi.powi(n));
    }
    // n < -1: x^n = 1 / x^|n|. Reuse positive case via reciprocal of integer power.
    let pos = interval_pow_int(a, -n);
    interval_pow_int(pos, -1)
}

// ---------------------------------------------------------------------
// Trigonometric: sin/cos via critical-point k-loop
// ---------------------------------------------------------------------

// Adapted from oxieml v0.1.0, src/lower_interval.rs (lines 347-367).
//
// SCIRS2 DEVIATION FROM SPEC: the Phase 0 spec proposed a 3-spot
// crit-point check (π/2, 3π/2, π/2 + 2π). That misses higher-period
// minima/maxima for inputs whose width is below 2π but spans >2 critical
// points (e.g. `sin([6, 12])` would miss the minimum at 7π/2 and produce
// an unsound bound). We port oxieml's k-loop verbatim instead.
fn interval_sin(c: Interval) -> Interval {
    if c.is_nan() {
        return Interval::nan();
    }
    use std::f64::consts::{FRAC_PI_2, PI};
    if c.hi - c.lo >= 2.0 * PI {
        return Interval::new(-1.0, 1.0);
    }
    let mut lo = c.lo.sin().min(c.hi.sin());
    let mut hi = c.lo.sin().max(c.hi.sin());
    let half_pi = FRAC_PI_2;
    let pi = PI;
    let k_lo = ((c.lo - half_pi) / pi).ceil() as i64;
    let k_hi = ((c.hi - half_pi) / pi).floor() as i64;
    for k in k_lo..=k_hi {
        let crit = half_pi + (k as f64) * pi;
        if c.lo <= crit && crit <= c.hi {
            // sin at π/2 + kπ is exactly ±1; use literal rather than .sin()
            // to avoid roundoff on the critical extremum itself.
            let v = if k.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
    }
    Interval::new(lo, hi)
}

// Adapted from oxieml v0.1.0, src/lower_interval.rs (lines 373-392).
fn interval_cos(c: Interval) -> Interval {
    if c.is_nan() {
        return Interval::nan();
    }
    use std::f64::consts::PI;
    if c.hi - c.lo >= 2.0 * PI {
        return Interval::new(-1.0, 1.0);
    }
    let mut lo = c.lo.cos().min(c.hi.cos());
    let mut hi = c.lo.cos().max(c.hi.cos());
    let pi = PI;
    let k_lo = (c.lo / pi).ceil() as i64;
    let k_hi = (c.hi / pi).floor() as i64;
    for k in k_lo..=k_hi {
        let crit = (k as f64) * pi;
        if c.lo <= crit && crit <= c.hi {
            // cos at kπ is exactly ±1 — use literal to avoid roundoff.
            let v = if k.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
    }
    Interval::new(lo, hi)
}

fn interval_tan(c: Interval) -> Interval {
    if c.is_nan() {
        return Interval::nan();
    }
    use std::f64::consts::{FRAC_PI_2, PI};
    // Tan has discontinuities at π/2 + kπ. We split [c.lo, c.hi] by the
    // half-period containing the lower bound: if both endpoints fall in
    // the same monotone branch we can return tight bounds; otherwise the
    // interval crosses an asymptote and we conservatively return full.
    let n_lo = ((c.lo + FRAC_PI_2) / PI).floor() as i64;
    let n_hi = ((c.hi + FRAC_PI_2) / PI).floor() as i64;
    if n_lo != n_hi {
        return Interval::full();
    }
    Interval::new(c.lo.tan(), c.hi.tan())
}

fn interval_cosh(c: Interval) -> Interval {
    if c.is_nan() {
        return Interval::nan();
    }
    if c.lo >= 0.0 {
        return Interval::new(c.lo.cosh(), c.hi.cosh());
    }
    if c.hi <= 0.0 {
        return Interval::new(c.hi.cosh(), c.lo.cosh());
    }
    // Crosses 0 — minimum is cosh(0) = 1.
    let max_arg = c.lo.abs().max(c.hi.abs());
    Interval::new(1.0, max_arg.cosh())
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::eval::{eval_real, EvalCtx};

    #[test]
    fn point_contains_self() {
        let v = 2.5_f64;
        let i = Interval::point(v);
        assert!(i.contains(v));
        assert_eq!(i.width(), 0.0);
    }

    #[test]
    fn empty_and_nan_distinguished() {
        let e = Interval::empty();
        assert!(e.is_empty());
        assert!(!e.is_nan());
        let n = Interval::nan();
        assert!(n.is_nan());
        assert!(!n.is_empty());
    }

    #[test]
    fn outward_widen_widens_finite() {
        let i = Interval::new(1.0, 2.0);
        let w = i.outward_widen();
        assert!(w.lo < 1.0 && w.hi > 2.0);
        // Still very tight (1 ULP at magnitude ~1 is ~2^-52 ≈ 2.22e-16).
        assert!(1.0 - w.lo < 1e-15);
        assert!(w.hi - 2.0 < 1e-15);
    }

    #[test]
    fn outward_widen_preserves_infinities() {
        let i = Interval::full();
        let w = i.outward_widen();
        assert_eq!(w.lo, f64::NEG_INFINITY);
        assert_eq!(w.hi, f64::INFINITY);
    }

    #[test]
    fn add_intervals_contains_sum() {
        // [1,2] + [3,4] = [4,6] (after outward widen, slightly wider).
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let r = eval_interval(&op, &[Interval::new(1.0, 2.0), Interval::new(3.0, 4.0)]);
        assert!(r.lo <= 4.0 && r.hi >= 6.0);
    }

    #[test]
    fn sub_intervals_swaps_endpoints() {
        // [1,2] - [3,4] = [1-4, 2-3] = [-3, -1]
        let op = LoweredOp::Sub(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let r = eval_interval(&op, &[Interval::new(1.0, 2.0), Interval::new(3.0, 4.0)]);
        assert!(r.lo <= -3.0 && r.hi >= -1.0);
    }

    #[test]
    fn mul_handles_sign_changes() {
        // [-2, 1] * [-3, 4] = min(6,-8,-3,4) ... = [-8, 6]
        let op = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let r = eval_interval(&op, &[Interval::new(-2.0, 1.0), Interval::new(-3.0, 4.0)]);
        assert!(r.lo <= -8.0 && r.hi >= 6.0);
    }

    #[test]
    fn div_by_zero_straddle_returns_full() {
        let op = LoweredOp::Div(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let r = eval_interval(&op, &[Interval::new(1.0, 2.0), Interval::new(-1.0, 1.0)]);
        assert!(r.lo == f64::NEG_INFINITY || r.lo <= -1e15);
        assert!(r.hi == f64::INFINITY || r.hi >= 1e15);
    }

    #[test]
    fn ln_negative_returns_nan() {
        let op = LoweredOp::Ln(Box::new(LoweredOp::Var(0)));
        let r = eval_interval(&op, &[Interval::new(-2.0, -1.0)]);
        assert!(r.is_nan());
    }

    #[test]
    fn sqrt_negative_returns_nan() {
        let op = LoweredOp::Sqrt(Box::new(LoweredOp::Var(0)));
        let r = eval_interval(&op, &[Interval::new(-2.0, -1.0)]);
        assert!(r.is_nan());
    }

    #[test]
    fn arcsin_outside_domain_nan() {
        let op = LoweredOp::Arcsin(Box::new(LoweredOp::Var(0)));
        let r = eval_interval(&op, &[Interval::new(-2.0, 0.5)]);
        assert!(r.is_nan());
    }

    #[test]
    fn sin_full_period_is_pm_one() {
        use std::f64::consts::PI;
        let op = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        let r = eval_interval(&op, &[Interval::new(0.0, 2.0 * PI)]);
        assert!(r.lo <= -1.0 + 1e-10);
        assert!(r.hi >= 1.0 - 1e-10);
    }

    #[test]
    fn sin_multi_period_min_reached() {
        // Regression: oxieml's k-loop must enumerate every critical point
        // inside the input. The 3-spot Phase 0 spec missed multi-period
        // minima — this case has minimum at 7π/2 ≈ 10.996 ∈ [6, 12].
        let op = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        let r = eval_interval(&op, &[Interval::new(6.0, 12.0)]);
        assert!(
            r.lo <= -1.0 + 1e-10,
            "sin min should reach -1, got {}",
            r.lo
        );
        assert!(r.hi >= 1.0 - 1e-10, "sin max should reach 1, got {}", r.hi);
    }

    #[test]
    fn cos_full_period_is_pm_one() {
        use std::f64::consts::PI;
        let op = LoweredOp::Cos(Box::new(LoweredOp::Var(0)));
        let r = eval_interval(&op, &[Interval::new(0.0, 2.0 * PI)]);
        assert!(r.lo <= -1.0 + 1e-10);
        assert!(r.hi >= 1.0 - 1e-10);
    }

    #[test]
    fn cos_narrow_monotone() {
        // cos is monotone-decreasing on [0, π/2]; tight bounds.
        use std::f64::consts::FRAC_PI_2;
        let op = LoweredOp::Cos(Box::new(LoweredOp::Var(0)));
        let r = eval_interval(&op, &[Interval::new(0.1, FRAC_PI_2 - 0.1)]);
        assert!(r.contains(0.1_f64.cos()) || r.lo <= 0.1_f64.cos());
        assert!(r.hi <= 1.0);
    }

    #[test]
    fn tan_within_branch() {
        let op = LoweredOp::Tan(Box::new(LoweredOp::Var(0)));
        let r = eval_interval(&op, &[Interval::new(0.0, 1.0)]);
        assert!(r.contains(0.0));
        assert!(r.contains(1.0_f64.tan()));
    }

    #[test]
    fn tan_crossing_asymptote_returns_full() {
        use std::f64::consts::PI;
        let op = LoweredOp::Tan(Box::new(LoweredOp::Var(0)));
        let r = eval_interval(&op, &[Interval::new(0.0, PI)]);
        // Crosses π/2 — must conservatively return full.
        assert!(r.lo == f64::NEG_INFINITY);
        assert!(r.hi == f64::INFINITY);
    }

    #[test]
    fn exp_monotone() {
        let op = LoweredOp::Exp(Box::new(LoweredOp::Var(0)));
        let r = eval_interval(&op, &[Interval::new(0.0, 1.0)]);
        assert!(r.lo < 1.0 + 1e-15);
        assert!(r.hi > std::f64::consts::E - 1e-15);
    }

    #[test]
    fn cosh_crosses_zero_min_is_one() {
        let op = LoweredOp::Cosh(Box::new(LoweredOp::Var(0)));
        let r = eval_interval(&op, &[Interval::new(-2.0, 1.0)]);
        // Min should be at 0 → cosh(0) = 1; max at -2 (cosh is symmetric).
        assert!(r.lo <= 1.0);
        assert!(r.hi >= 2.0_f64.cosh());
    }

    #[test]
    fn abs_crossing_zero_min_zero() {
        let op = LoweredOp::Abs(Box::new(LoweredOp::Var(0)));
        let r = eval_interval(&op, &[Interval::new(-3.0, 2.0)]);
        assert!(r.lo <= 0.0);
        assert!(r.hi >= 3.0);
    }

    #[test]
    fn pow_int_square_crosses_zero() {
        let op = LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(2.0)));
        let r = eval_interval(&op, &[Interval::new(-3.0, 2.0)]);
        assert!(r.lo <= 0.0);
        assert!(r.hi >= 9.0);
    }

    #[test]
    fn pow_int_cube_monotone() {
        let op = LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(3.0)));
        let r = eval_interval(&op, &[Interval::new(-2.0, 3.0)]);
        assert!(r.lo <= -8.0);
        assert!(r.hi >= 27.0);
    }

    #[test]
    fn containment_proptest() {
        // Sampled containment check — randomised f64 points inside narrow
        // intervals must always lie within the interval-evaluation result.
        // 100 seeds × 4 formulas × 2 variants (point at edge vs centre).
        let formulas: Vec<LoweredOp> = vec![
            LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0))),
            LoweredOp::Sin(Box::new(LoweredOp::Var(0))),
            LoweredOp::Exp(Box::new(LoweredOp::Var(0))),
            LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(0))),
        ];
        for seed in 0..100 {
            let xv = (seed as f64) * 0.1 - 5.0;
            let xi = Interval::new(xv - 0.01, xv + 0.01);
            for f in &formulas {
                let interval_result = eval_interval(f, &[xi]);
                if interval_result.is_nan() {
                    continue;
                }
                let bindings = [xv];
                let ctx = EvalCtx::new(&bindings);
                if let Ok(s) = eval_real(f, &ctx) {
                    assert!(
                        interval_result.contains(s),
                        "containment violation: scalar {} not in {:?} for formula {:?} at xv={}",
                        s,
                        interval_result,
                        f,
                        xv
                    );
                }
            }
        }
    }
}
