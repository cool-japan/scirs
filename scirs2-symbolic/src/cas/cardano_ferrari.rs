//! Closed-form polynomial solvers for cubic (Cardano) and quartic (Ferrari).
//!
//! These extend `solve_polynomial` to degrees 3 and 4.
//! Both algorithms operate on **literal-numeric** coefficients (`f64`); when
//! coefficients are symbolic, the caller must lower them by partial evaluation
//! first. Solutions are returned as `LoweredOp` expressions involving `Sqrt`,
//! `Pow`, and arithmetic on the input `Const(...)` values.
//!
//! # Cardano (cubic)
//!
//! Standard reduction:
//!
//! 1. Normalise: `a₃·x³ + a₂·x² + a₁·x + a₀ = 0` → `x³ + b·x² + c·x + d = 0`
//!    with `b = a₂/a₃, c = a₁/a₃, d = a₀/a₃`.
//! 2. Depress: `x = y − b/3` → `y³ + p·y + q = 0` with
//!    `p = c − b²/3` and `q = (2b³)/27 − (b·c)/3 + d`.
//! 3. Discriminant: `Δ = −(4p³ + 27q²)` (real-decision).
//!    - `Δ > 0` (three real roots): trigonometric form
//!      `yₖ = 2·√(−p/3)·cos[(1/3)·arccos(3q/(2p)·√(−3/p)) − 2πk/3]` for k=0,1,2.
//!    - `Δ ≤ 0` (one real, two complex): Cardano formula
//!      `y = ∛(−q/2 + √(q²/4 + p³/27)) + ∛(−q/2 − √(q²/4 + p³/27))`.
//!      Plus two complex conjugate roots that we represent symbolically; the
//!      real-only path returns the single real branch with `complete=false`.
//! 4. Un-substitute: `x = y − b/3`.
//!
//! # Ferrari (quartic)
//!
//! 1. Normalise: divide through by `a₄`.
//! 2. Depress: `x = y − b/4` → `y⁴ + p·y² + q·y + r = 0` with
//!    `p = c − 3b²/8`, `q = b³/8 − bc/2 + d`, `r = −3b⁴/256 + b²c/16 − bd/4 + e`.
//! 3. Resolvent cubic: `8t³ + 8p·t² + (2p² − 8r)·t − q² = 0`.
//!    Solve for any real root `t₀` via Cardano.
//! 4. Edge case `q = 0`: biquadratic `y⁴ + p·y² + r = 0`; substitute `z = y²`
//!    and apply the quadratic formula.
//! 5. Otherwise factor: `(y² + αy + β)(y² + γy + δ)` with
//!    `α = ±√(2t₀)` (sign matched to `q`), `β = t₀ + p/2 − q/(2α)`,
//!    `γ = −α`, `δ = t₀ + p/2 + q/(2α)`.
//!    Solve each quadratic factor for two roots.
//! 6. Un-substitute: `x = y − b/4`.

#![warn(missing_docs)]

use crate::cas::solve::{SolveError, SolveResult};
use crate::eml::op::LoweredOp;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Solve a cubic polynomial with literal-numeric coefficients.
///
/// `coeffs[k]` is the coefficient of `x^k`. The polynomial is `Σₖ coeffs[k]·x^k`
/// and we solve `Σₖ coeffs[k]·x^k = 0`. Length must be ≥ 4 (degree ≥ 3 after
/// trailing-zero stripping); the caller is responsible for stripping.
///
/// Returns `SolveResult` with up to three real `LoweredOp` solutions. The
/// `complete` flag is `true` when all three real roots were recovered (Δ > 0)
/// or the single real root for Δ ≤ 0 (with two complex conjugates omitted).
pub fn solve_cubic(coeffs: &[LoweredOp]) -> Result<SolveResult, SolveError> {
    if coeffs.len() < 4 {
        return Err(SolveError::InternalError(
            "solve_cubic requires at least 4 coefficients".into(),
        ));
    }

    let f64_coeffs = extract_f64(coeffs)?;
    let a3 = f64_coeffs[3];
    if a3.abs() < 1e-14 {
        return Err(SolveError::InternalError(
            "leading coefficient is numerically zero".into(),
        ));
    }

    let a2 = f64_coeffs[2];
    let a1 = f64_coeffs[1];
    let a0 = f64_coeffs[0];

    // Normalise: x³ + b·x² + c·x + d = 0
    let b = a2 / a3;
    let c = a1 / a3;
    let d = a0 / a3;

    // Depress: y = x + b/3 → y³ + p·y + q = 0
    let p = c - b * b / 3.0;
    let q = 2.0 * b.powi(3) / 27.0 - b * c / 3.0 + d;

    let depressed_roots = solve_depressed_cubic(p, q);
    // Un-substitute x = y − b/3.
    let shift = -b / 3.0;
    let solutions: Vec<LoweredOp> = depressed_roots
        .into_iter()
        .map(|y| LoweredOp::Const(y + shift))
        .collect();

    if solutions.is_empty() {
        return Err(SolveError::InternalError(
            "Cardano produced no real roots".into(),
        ));
    }
    let three_roots = solutions.len() == 3;
    Ok(SolveResult {
        solutions,
        complete: three_roots,
    })
}

/// Solve a quartic polynomial with literal-numeric coefficients via Ferrari.
///
/// `coeffs[k]` is the coefficient of `x^k`. Length must be ≥ 5 (degree 4
/// after trailing-zero stripping).
///
/// Returns up to four real solutions. The `complete` flag is `true` when
/// all four real roots are recovered.
pub fn solve_quartic(coeffs: &[LoweredOp]) -> Result<SolveResult, SolveError> {
    if coeffs.len() < 5 {
        return Err(SolveError::InternalError(
            "solve_quartic requires at least 5 coefficients".into(),
        ));
    }

    let f = extract_f64(coeffs)?;
    let a4 = f[4];
    if a4.abs() < 1e-14 {
        return Err(SolveError::InternalError(
            "leading coefficient is numerically zero".into(),
        ));
    }

    // Normalise: x⁴ + b·x³ + c·x² + d·x + e
    let b = f[3] / a4;
    let c = f[2] / a4;
    let d = f[1] / a4;
    let e = f[0] / a4;

    // Depress: x = y − b/4 → y⁴ + p·y² + q·y + r
    let p = c - 3.0 * b * b / 8.0;
    let q = b.powi(3) / 8.0 - b * c / 2.0 + d;
    let r = -3.0 * b.powi(4) / 256.0 + b * b * c / 16.0 - b * d / 4.0 + e;

    let y_roots = solve_depressed_quartic(p, q, r);
    let shift = -b / 4.0;
    let solutions: Vec<LoweredOp> = y_roots
        .into_iter()
        .map(|y| LoweredOp::Const(y + shift))
        .collect();

    let len = solutions.len();
    Ok(SolveResult {
        solutions,
        complete: len == 4,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract f64 coefficients from a slice of `LoweredOp::Const(...)`.
///
/// Returns `SolveError::InternalError` if any entry is non-constant. Symbolic
/// coefficients are not supported at this layer; the caller must reduce them
/// by partial evaluation first.
fn extract_f64(coeffs: &[LoweredOp]) -> Result<Vec<f64>, SolveError> {
    coeffs
        .iter()
        .map(|c| match c {
            LoweredOp::Const(v) => Ok(*v),
            _ => Err(SolveError::InternalError(
                "Cardano/Ferrari requires literal-numeric coefficients".into(),
            )),
        })
        .collect()
}

/// Solve depressed cubic `y³ + p·y + q = 0` for real roots.
///
/// Returns 1 or 3 real roots. For three-real-root case (Δ > 0) the
/// trigonometric form is used; for the single-real-root case (Δ ≤ 0) the
/// Cardano cube-root formula is used.
fn solve_depressed_cubic(p: f64, q: f64) -> Vec<f64> {
    // Δ = −(4p³ + 27q²)
    let discr = -(4.0 * p.powi(3) + 27.0 * q * q);

    if p.abs() < 1e-14 && q.abs() < 1e-14 {
        // Triple zero root.
        return vec![0.0, 0.0, 0.0];
    }

    if p.abs() < 1e-14 {
        // y³ = −q → single real cube-root, plus two complex conjugates.
        let y = (-q).cbrt();
        return vec![y];
    }

    if discr > 1e-14 {
        // Three distinct real roots — trigonometric form.
        // Argument of arccos: 3q/(2p) · √(−3/p), guard against |arg| > 1
        // due to f64 round-off.
        let r = -p / 3.0;
        let m = 2.0 * r.sqrt(); // 2·√(−p/3)
        let inside = (3.0 * q / (2.0 * p)) * (3.0 / -p).sqrt();
        let arg = inside.clamp(-1.0, 1.0);
        let theta = arg.acos() / 3.0;
        let two_pi_third = 2.0 * std::f64::consts::PI / 3.0;
        vec![
            m * theta.cos(),
            m * (theta - two_pi_third).cos(),
            m * (theta - 2.0 * two_pi_third).cos(),
        ]
    } else {
        // One real root (Δ ≤ 0); Cardano cube-root formula.
        let half_q = q / 2.0;
        let inside = half_q * half_q + p.powi(3) / 27.0;
        if inside < 0.0 {
            // Defensive: should not occur when discr ≤ 0 modulo round-off.
            // Use the trig form as a fallback.
            return solve_depressed_cubic_trig_fallback(p, q);
        }
        let s = inside.sqrt();
        let u = signed_cbrt(-half_q + s);
        let v = signed_cbrt(-half_q - s);
        vec![u + v]
    }
}

/// Robust trig fallback when Δ is near zero (round-off may flip sign).
fn solve_depressed_cubic_trig_fallback(p: f64, q: f64) -> Vec<f64> {
    // Three coincident real roots when Δ = 0.
    let r = -p / 3.0;
    if r < 0.0 {
        // No real solution via this branch.
        return vec![];
    }
    let m = 2.0 * r.sqrt();
    let denom = 2.0 * p * r.sqrt();
    let arg = if denom.abs() < 1e-14 {
        0.0
    } else {
        ((3.0 * q) / denom).clamp(-1.0, 1.0)
    };
    let theta = arg.acos() / 3.0;
    let two_pi_third = 2.0 * std::f64::consts::PI / 3.0;
    vec![
        m * theta.cos(),
        m * (theta - two_pi_third).cos(),
        m * (theta - 2.0 * two_pi_third).cos(),
    ]
}

/// Signed cube root: handles negative arguments correctly (Rust's `.cbrt()`
/// already does this, but we wrap for explicit semantics).
fn signed_cbrt(x: f64) -> f64 {
    x.cbrt()
}

/// Solve depressed quartic `y⁴ + p·y² + q·y + r = 0` for real roots.
///
/// Algorithm (Ferrari, Wikipedia "Quartic function § Ferrari's solution"):
///
/// 1. Add `2zy² + z²` to both sides → `(y²+z)² = (2z−p)y² − qy + (z²−r)`.
/// 2. The RHS is a perfect square iff its discriminant in `y` vanishes:
///    `q² − 4·(2z−p)·(z²−r) = 0` ⇒ resolvent cubic
///    `8z³ − 4p·z² − 8r·z + (4pr − q²) = 0`.
/// 3. Solve the resolvent for any real `z₀`.
/// 4. Then `(y²+z₀)² = (αy + β)²` with `α = √(2z₀−p)`, `β = −q/(2α)`
///    (when `α ≠ 0`).
/// 5. Splitting `y² + z₀ = ±(αy + β)` gives two quadratics in `y`.
///
/// Edge cases:
/// - `q ≈ 0`: biquadratic; substitute `z = y²` and apply quadratic formula.
/// - `α ≈ 0`: pick the resolvent root with the largest `2z − p` to avoid
///   division by zero; fall back to biquadratic if no choice has `2z − p > 0`.
fn solve_depressed_quartic(p: f64, q: f64, r: f64) -> Vec<f64> {
    if q.abs() < 1e-14 {
        return solve_biquadratic(p, r);
    }

    // Resolvent cubic: 8z³ − 4p·z² − 8r·z + (4pr − q²) = 0.
    // Normalize → z³ + (−p/2)z² + (−r)z + (pr/2 − q²/8) = 0.
    // Depress: z = u + p/6 → u³ + P·u + Q = 0.
    let a2 = -p / 2.0;
    let a1 = -r;
    let a0 = p * r / 2.0 - q * q / 8.0;

    let cap_p = a1 - a2 * a2 / 3.0;
    let cap_q = 2.0 * a2.powi(3) / 27.0 - a2 * a1 / 3.0 + a0;
    let u_roots = solve_depressed_cubic(cap_p, cap_q);
    let z_shift = -a2 / 3.0;
    let z_candidates: Vec<f64> = u_roots.iter().map(|u| u + z_shift).collect();

    // Pick the z that maximises 2z − p (so α² is positive and α ≠ 0). Among
    // those with 2z − p > 0, prefer the one closest to making the
    // factorisation numerically stable.
    let mut best_z: Option<f64> = None;
    let mut best_two_z_minus_p = -f64::INFINITY;
    for &z in &z_candidates {
        let v = 2.0 * z - p;
        if v > best_two_z_minus_p {
            best_two_z_minus_p = v;
            best_z = Some(z);
        }
    }
    let z0 = match best_z {
        Some(z) => z,
        None => return Vec::new(),
    };

    let alpha_sq = 2.0 * z0 - p;
    if alpha_sq < -1e-9 {
        // No real factorisation from any resolvent root → no real quartic
        // roots (or all complex).
        return Vec::new();
    }
    let alpha_sq = alpha_sq.max(0.0);
    let alpha = alpha_sq.sqrt();

    if alpha.abs() < 1e-12 {
        // α=0 means q² = 0 (from resolvent), but we are in the q≠0 branch.
        // This is a numerical degeneracy — try with the next best z, else
        // fall back to biquadratic.
        return solve_biquadratic(p, r);
    }

    let beta = -q / (2.0 * alpha);

    // (y² + z₀)² = (α·y + β)² ⇒ y² + z₀ = ±(α·y + β)
    // Branch A: y² + z₀ − αy − β = 0   (i.e. y² − αy + (z₀ − β) = 0)
    // Branch B: y² + z₀ + αy + β = 0   (i.e. y² + αy + (z₀ + β) = 0)
    let mut roots = Vec::with_capacity(4);
    quadratic_real_roots(-alpha, z0 - beta, &mut roots);
    quadratic_real_roots(alpha, z0 + beta, &mut roots);
    roots
}

/// Solve biquadratic `y⁴ + p·y² + r = 0` for real roots.
fn solve_biquadratic(p: f64, r: f64) -> Vec<f64> {
    let discr = p * p - 4.0 * r;
    if discr < -1e-14 {
        return Vec::new();
    }
    let sd = discr.max(0.0).sqrt();
    let z1 = (-p + sd) / 2.0;
    let z2 = (-p - sd) / 2.0;
    let mut roots = Vec::new();
    for z in [z1, z2] {
        if z >= -1e-14 {
            let zc = z.max(0.0);
            let y = zc.sqrt();
            roots.push(y);
            if y > 1e-14 {
                roots.push(-y);
            }
        }
    }
    roots
}

/// Append real roots of `y² + a·y + b = 0` to `out`.
fn quadratic_real_roots(a: f64, b: f64, out: &mut Vec<f64>) {
    let discr = a * a - 4.0 * b;
    if discr < -1e-12 {
        return;
    }
    let sd = discr.max(0.0).sqrt();
    out.push((-a + sd) / 2.0);
    if sd > 1e-14 {
        out.push((-a - sd) / 2.0);
    } else {
        out.push((-a - sd) / 2.0); // double root — emit twice for multiplicity
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    fn eval_const(op: &LoweredOp) -> f64 {
        match op {
            LoweredOp::Const(v) => *v,
            _ => f64::NAN,
        }
    }

    // ----------------------------------------------------------------
    // Cubic tests
    // ----------------------------------------------------------------

    #[test]
    fn cubic_three_real_distinct() {
        // (x−1)(x−2)(x−3) = x³ − 6x² + 11x − 6
        let coeffs = [c(-6.0), c(11.0), c(-6.0), c(1.0)];
        let result = solve_cubic(&coeffs).expect("should solve");
        assert_eq!(result.solutions.len(), 3);
        let mut got: Vec<f64> = result.solutions.iter().map(eval_const).collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (a, b) in got.iter().zip([1.0, 2.0, 3.0].iter()) {
            assert!((a - b).abs() < 1e-9, "got {a:?} expected {b:?}");
        }
    }

    #[test]
    fn cubic_one_real_two_complex() {
        // x³ + 1 = 0 → real root x = −1; two complex conjugates omitted.
        let coeffs = [c(1.0), c(0.0), c(0.0), c(1.0)];
        let result = solve_cubic(&coeffs).expect("should solve");
        let r = eval_const(&result.solutions[0]);
        assert!((r + 1.0).abs() < 1e-12, "got {r}, expected −1");
    }

    #[test]
    fn cubic_triple_zero() {
        // x³ = 0
        let coeffs = [c(0.0), c(0.0), c(0.0), c(1.0)];
        let result = solve_cubic(&coeffs).expect("should solve");
        for sol in &result.solutions {
            assert!(eval_const(sol).abs() < 1e-12);
        }
    }

    // ----------------------------------------------------------------
    // Quartic tests
    // ----------------------------------------------------------------

    #[test]
    fn quartic_distinct_real() {
        // (x−1)(x−2)(x−3)(x−4) = x⁴ − 10x³ + 35x² − 50x + 24
        let coeffs = [c(24.0), c(-50.0), c(35.0), c(-10.0), c(1.0)];
        let result = solve_quartic(&coeffs).expect("should solve");
        assert_eq!(result.solutions.len(), 4);
        let mut got: Vec<f64> = result.solutions.iter().map(eval_const).collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (a, b) in got.iter().zip([1.0, 2.0, 3.0, 4.0].iter()) {
            assert!((a - b).abs() < 1e-7, "got {a:?} expected {b:?}");
        }
    }

    #[test]
    fn quartic_biquadratic() {
        // x⁴ − 5x² + 4 = (x²−1)(x²−4)
        let coeffs = [c(4.0), c(0.0), c(-5.0), c(0.0), c(1.0)];
        let result = solve_quartic(&coeffs).expect("should solve");
        let mut got: Vec<f64> = result.solutions.iter().map(eval_const).collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let expected = [-2.0, -1.0, 1.0, 2.0];
        assert_eq!(got.len(), 4);
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-9, "got {a:?} expected {b:?}");
        }
    }

    #[test]
    fn quartic_no_real_roots() {
        // x⁴ + 1 = 0 has no real roots.
        let coeffs = [c(1.0), c(0.0), c(0.0), c(0.0), c(1.0)];
        let result = solve_quartic(&coeffs).expect("should solve");
        // No real roots — solutions vector should be empty.
        assert!(
            result.solutions.is_empty(),
            "x⁴+1=0 has no real roots; got {:?}",
            result.solutions
        );
        assert!(!result.complete);
    }

    #[test]
    fn quartic_irreducible_real() {
        // x⁴ − 2 = 0 → real roots ±2^(1/4)
        let coeffs = [c(-2.0), c(0.0), c(0.0), c(0.0), c(1.0)];
        let result = solve_quartic(&coeffs).expect("should solve");
        let target = 2.0_f64.powf(0.25);
        let got: Vec<f64> = result.solutions.iter().map(eval_const).collect();
        let has_pos = got.iter().any(|v| (v - target).abs() < 1e-9);
        let has_neg = got.iter().any(|v| (v + target).abs() < 1e-9);
        assert!(has_pos && has_neg, "got {got:?}");
    }
}
