//! `Extrapolate` trait — a composable extension point for 1-D interpolators.
//!
//! Any type that knows its domain `[x_min, x_max]` and can evaluate an
//! interior value via `value_at(x)` automatically gets five well-defined
//! extrapolation modes through the blanket default implementation of
//! [`Extrapolate::extrapolate`].
//!
//! # Extrapolation modes
//!
//! | `ExtrapolationBehavior`      | Behaviour outside the domain |
//! |------------------------|------------------------------|
//! | `Nearest`              | Clamp `x` to boundary, evaluate |
//! | `Linear`               | Finite-difference slope at boundary |
//! | `Polynomial(degree)`   | Neville's algorithm using `degree+1` interior pts |
//! | `Reflection`           | Mirror `x` about the boundary |
//! | `Periodic`             | Wrap `x` with period `x_max - x_min` |
//!
//! # Example
//!
//! ```rust
//! use scirs2_interpolate::extrapolation_trait::{ExtrapolationBehavior, Extrapolate};
//!
//! // A simple linear function on [0, 1].
//! struct LinearFn;
//!
//! impl Extrapolate for LinearFn {
//!     fn domain(&self) -> (f64, f64) { (0.0, 1.0) }
//!     fn value_at(&self, x: f64) -> f64 { 2.0 * x }
//! }
//!
//! let f = LinearFn;
//! let y = f.extrapolate(1.5, &ExtrapolationBehavior::Linear);
//! assert!((y - 3.0).abs() < 1e-6);
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// ExtrapolationBehavior
// ─────────────────────────────────────────────────────────────────────────────

/// Extrapolation behaviour for queries outside `[x_min, x_max]`.
///
/// This enum is deliberately kept simple and orthogonal.  For the richer
/// physics-informed and ensemble strategies, see [`crate::extrapolation`].
#[derive(Debug, Clone, PartialEq)]
pub enum ExtrapolationBehavior {
    /// Clamp `x` to the nearest boundary value.
    Nearest,
    /// Linearly extrapolate using the finite-difference slope at the boundary.
    ///
    /// Step size `h = (x_max - x_min) * 1e-6`.
    Linear,
    /// Neville polynomial extrapolation of the given degree.
    ///
    /// Uses `degree + 1` equally-spaced interior points near the boundary.
    /// The degree is clamped to 9 (10-point stencil) to avoid blow-up.
    Polynomial(usize),
    /// Mirror `x` about the boundary, then evaluate the interior function.
    Reflection,
    /// Wrap `x` with period `x_max - x_min`, then evaluate the interior function.
    Periodic,
}

// ─────────────────────────────────────────────────────────────────────────────
// Extrapolate trait
// ─────────────────────────────────────────────────────────────────────────────

/// Composable extrapolation for 1-D interpolators.
///
/// Implementors must provide:
/// - [`domain`](Extrapolate::domain) — the closed interval `[x_min, x_max]`.
/// - [`value_at`](Extrapolate::value_at) — interior evaluation (assumed valid
///   only within the domain; behaviour outside is undefined).
///
/// The provided default [`extrapolate`](Extrapolate::extrapolate) method
/// dispatches to the five modes in [`ExtrapolationBehavior`].
pub trait Extrapolate {
    /// Returns `(x_min, x_max)` — the closed interpolation domain.
    fn domain(&self) -> (f64, f64);

    /// Evaluate the interpolant at `x`.
    ///
    /// Callers within this crate guarantee `x_min <= x <= x_max`.
    fn value_at(&self, x: f64) -> f64;

    /// Evaluate at `x`, applying `mode` for out-of-domain queries.
    ///
    /// If `x` is inside the domain the function falls through to `value_at`.
    fn extrapolate(&self, x: f64, mode: &ExtrapolationBehavior) -> f64 {
        let (x_min, x_max) = self.domain();

        // In-domain: delegate to interior evaluation.
        if x >= x_min && x <= x_max {
            return self.value_at(x);
        }

        match mode {
            ExtrapolationBehavior::Nearest => {
                let clamped = x.max(x_min).min(x_max);
                self.value_at(clamped)
            }

            ExtrapolationBehavior::Linear => {
                let span = x_max - x_min;
                let h = (span * 1e-6).max(1e-12);
                if x < x_min {
                    let slope = (self.value_at(x_min + h) - self.value_at(x_min)) / h;
                    self.value_at(x_min) + slope * (x - x_min)
                } else {
                    let slope = (self.value_at(x_max) - self.value_at(x_max - h)) / h;
                    self.value_at(x_max) + slope * (x - x_max)
                }
            }

            ExtrapolationBehavior::Polynomial(degree) => {
                // Neville stencil: `n_pts` equally-spaced points taken from
                // the relevant boundary region (left or right).
                let n_pts = (*degree + 1).min(10).max(2);
                let span = x_max - x_min;
                let stencil_xs: Vec<f64>;
                let stencil_ys: Vec<f64>;
                if x < x_min {
                    let step = span / (n_pts as f64 - 1.0);
                    stencil_xs = (0..n_pts).map(|i| x_min + i as f64 * step).collect();
                } else {
                    let step = span / (n_pts as f64 - 1.0);
                    stencil_xs = (0..n_pts)
                        .map(|i| x_max - (n_pts - 1 - i) as f64 * step)
                        .collect();
                }
                stencil_ys = stencil_xs.iter().map(|&xi| self.value_at(xi)).collect();
                neville_eval(&stencil_xs, &stencil_ys, x)
            }

            ExtrapolationBehavior::Reflection => {
                let span = x_max - x_min;
                let reflected = reflect_into_domain(x, x_min, x_max, span);
                self.value_at(reflected)
            }

            ExtrapolationBehavior::Periodic => {
                let span = x_max - x_min;
                if span < 1e-30 {
                    return self.value_at(x_min);
                }
                let periodic = ((x - x_min).rem_euclid(span)) + x_min;
                self.value_at(periodic)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Neville's algorithm for polynomial interpolation / extrapolation.
///
/// `xs` and `ys` must have the same length ≥ 1.
/// Returns the value of the unique polynomial passing through all nodes,
/// evaluated at `x`.
pub fn neville_eval(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len().min(ys.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut d = ys[..n].to_vec();
    for k in 1..n {
        for i in 0..n - k {
            let den = xs[i] - xs[i + k];
            if den.abs() > 1e-14 {
                let num = (x - xs[i + k]) * d[i] - (x - xs[i]) * d[i + 1];
                d[i] = num / den;
            }
            // If denominator is tiny, keep d[i] unchanged (degenerate stencil).
        }
    }
    d[0]
}

/// Map `x` (possibly outside `[x_min, x_max]`) to a reflected point inside.
///
/// The mapping folds `x` with period `2 * span` and mirrors on odd periods.
pub fn reflect_into_domain(x: f64, x_min: f64, x_max: f64, span: f64) -> f64 {
    if span < 1e-30 {
        return x_min;
    }
    let norm = (x - x_min) / span;
    // period = 2: [0,1] is the first half (un-reflected), [1,2] second (reflected)
    let period = norm.rem_euclid(2.0);
    let mapped = if period > 1.0 { 2.0 - period } else { period };
    (mapped * span + x_min).clamp(x_min, x_max)
}

// ─────────────────────────────────────────────────────────────────────────────
// Blanket impl for closures
// ─────────────────────────────────────────────────────────────────────────────

/// Convenience wrapper for using a closure as an `Extrapolate` implementor.
pub struct ClosureInterpolator<F: Fn(f64) -> f64> {
    pub x_min: f64,
    pub x_max: f64,
    pub f: F,
}

impl<F: Fn(f64) -> f64> Extrapolate for ClosureInterpolator<F> {
    fn domain(&self) -> (f64, f64) {
        (self.x_min, self.x_max)
    }
    fn value_at(&self, x: f64) -> f64 {
        (self.f)(x)
    }
}
