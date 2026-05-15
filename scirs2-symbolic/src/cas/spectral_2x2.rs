//! `cas::spectral_2x2` — closed-form eigenvalues and eigenvectors for symmetric 2×2 matrices.
//!
//! # Math
//!
//! For a symmetric 2×2 matrix `[[a, b], [b, d]]`:
//!
//! - `tr = a + d`
//! - `discriminant = (a − d)² + 4b²`  (always ≥ 0 for real symmetric)
//! - `delta = sqrt(discriminant)`
//! - `λ₁ = (tr + delta) / 2`  (larger eigenvalue)
//! - `λ₂ = (tr − delta) / 2`  (smaller eigenvalue)
//! - Eigenvector for `λ₁`: `v1 = [b, λ₁ − a]`
//! - Eigenvector for `λ₂`: `v2 = [b, λ₂ − a]`
//!
//! Orthogonality proof:
//! `<v1, v2> = b² + (λ₁−a)(λ₂−a) = b² + ((d−a)² − δ²)/4 = b² − b² = 0`. ✓
//!
//! # Degenerate case (b = 0)
//!
//! When the off-diagonal entry `b` is symbolically zero, `v1 = [0, λ₁−a]`
//! and `v2 = [0, λ₂−a]`. These share the same first component and are not
//! orthogonal in the usual sense. Callers that need well-defined eigenvectors
//! for the degenerate case should branch on `b == 0` before calling this
//! function and handle the diagonal case separately.
//!
//! # All outputs are canonicalized
//!
//! Every [`LoweredOp`] in the returned [`SymmetricEig2`] has been run through
//! [`crate::cas::canonicalize::canonicalize`] once. Constant-only inputs
//! therefore produce `Const(value)` leaves, enabling direct numerical
//! comparison without evaluation.

use crate::cas::canonicalize::canonicalize;
use crate::eml::op::LoweredOp;

/// Eigenvalues and unnormalized eigenvectors for a symmetric 2×2 matrix.
///
/// # Convention
///
/// - `m[0][1] == m[1][0]` is the caller's contract (symmetry is not checked).
/// - Eigenvectors are **not** normalized — they are symbolically compact.
/// - Column order: `eigenvalues[0]` is the larger (or equal) eigenvalue;
///   `eigenvectors[0]` is the corresponding eigenvector.
/// - When the off-diagonal entry is zero, `eigenvectors[i][0] = 0` for both
///   columns (see module-level docs).
#[derive(Clone, Debug)]
pub struct SymmetricEig2 {
    /// `[lambda_1, lambda_2]` — larger eigenvalue first.
    pub eigenvalues: [LoweredOp; 2],
    /// `[v1, v2]` as column vectors — each `[LoweredOp; 2]` is `[x, y]`.
    pub eigenvectors: [[LoweredOp; 2]; 2],
}

// ── private arithmetic helpers ────────────────────────────────────────────────

#[inline]
fn c(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}

#[inline]
fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Box::new(a), Box::new(b))
}

#[inline]
fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(Box::new(a), Box::new(b))
}

#[inline]
fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Box::new(a), Box::new(b))
}

#[inline]
fn div(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Div(Box::new(a), Box::new(b))
}

#[inline]
fn sq(a: LoweredOp) -> LoweredOp {
    LoweredOp::Pow(Box::new(a), Box::new(LoweredOp::Const(2.0)))
}

#[inline]
fn sqrt_op(a: LoweredOp) -> LoweredOp {
    LoweredOp::Sqrt(Box::new(a))
}

/// Apply canonicalization and extract the inner `LoweredOp`.
#[inline]
fn canon(op: LoweredOp) -> LoweredOp {
    canonicalize(&op).into_op()
}

// ── public function ───────────────────────────────────────────────────────────

/// Compute the closed-form eigenvalues and unnormalized eigenvectors of a
/// symmetric 2×2 matrix.
///
/// # Arguments
///
/// * `m` — the matrix as `[[a, b], [b, d]]` where `m[i][j]` is a `LoweredOp`.
///   Symmetry (`m[0][1] == m[1][0]`) is the caller's contract.
///
/// # Returns
///
/// A [`SymmetricEig2`] whose fields are all canonicalized [`LoweredOp`]
/// expressions.
///
/// # Panics
///
/// Never panics.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::cas::spectral_2x2::{eig_symmetric_2x2};
/// use scirs2_symbolic::eml::{eval_real, EvalCtx};
/// use scirs2_symbolic::LoweredOp;
///
/// let a = LoweredOp::Const(3.0);
/// let b = LoweredOp::Const(1.0);
/// let d = LoweredOp::Const(3.0);
/// let m = [[a.clone(), b.clone()], [b, d]];
/// let eig = eig_symmetric_2x2(&m);
/// let ctx = EvalCtx::new(&[]);
/// let lam1 = eval_real(&eig.eigenvalues[0], &ctx).unwrap();
/// let lam2 = eval_real(&eig.eigenvalues[1], &ctx).unwrap();
/// assert!((lam1 - 4.0).abs() < 1e-10);
/// assert!((lam2 - 2.0).abs() < 1e-10);
/// ```
pub fn eig_symmetric_2x2(m: &[[LoweredOp; 2]; 2]) -> SymmetricEig2 {
    // Extract matrix entries, cloning each one as each is used multiple times.
    let a = m[0][0].clone(); // top-left
    let b = m[0][1].clone(); // off-diagonal
    let d = m[1][1].clone(); // bottom-right

    // tr = a + d
    let tr = add(a.clone(), d.clone());

    // discriminant = (a - d)^2 + 4*b^2
    let a_minus_d = sub(a.clone(), d.clone());
    let discriminant = add(sq(a_minus_d), mul(c(4.0), sq(b.clone())));

    // delta = sqrt(discriminant)
    let delta = sqrt_op(discriminant);

    // lambda_1 = (tr + delta) / 2
    let lambda_1 = div(add(tr.clone(), delta.clone()), c(2.0));

    // lambda_2 = (tr - delta) / 2
    let lambda_2 = div(sub(tr, delta), c(2.0));

    // Canonicalize eigenvalues
    let lam1 = canon(lambda_1.clone());
    let lam2 = canon(lambda_2.clone());

    // Eigenvector for lambda_1: v1 = [b, lambda_1 - a]
    // (correct when b != 0; see module docs for degenerate case)
    let v1x = canon(b.clone());
    let v1y = canon(sub(lambda_1, a.clone()));

    // Eigenvector for lambda_2: v2 = [b, lambda_2 - a]
    // Orthogonality: <v1, v2> = b^2 + (λ₁-a)(λ₂-a) = b^2 - b^2 = 0
    let v2x = canon(b.clone());
    let v2y = canon(sub(lambda_2, a.clone()));

    SymmetricEig2 {
        eigenvalues: [lam1, lam2],
        eigenvectors: [[v1x, v1y], [v2x, v2y]],
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::eval::{eval_real, EvalCtx};

    /// Convenience: build a constant matrix.
    fn const_mat(a: f64, b: f64, d: f64) -> [[LoweredOp; 2]; 2] {
        [
            [LoweredOp::Const(a), LoweredOp::Const(b)],
            [LoweredOp::Const(b), LoweredOp::Const(d)],
        ]
    }

    /// Evaluate a `LoweredOp` with an empty binding (constant expression).
    fn ev(op: &LoweredOp) -> f64 {
        eval_real(op, &EvalCtx::new(&[])).expect("eval failed")
    }

    /// Evaluate a `LoweredOp` with a binding slice.
    fn ev_at(op: &LoweredOp, bindings: &[f64]) -> f64 {
        eval_real(op, &EvalCtx::new(bindings)).expect("eval failed")
    }

    // Test 1: [[3,1],[1,3]] — eigenvalues 4 and 2.
    #[test]
    fn test_eigenvalues_3_1_1_3() {
        let m = const_mat(3.0, 1.0, 3.0);
        let eig = eig_symmetric_2x2(&m);
        let lam1 = ev(&eig.eigenvalues[0]);
        let lam2 = ev(&eig.eigenvalues[1]);
        assert!((lam1 - 4.0).abs() < 1e-10, "lambda_1={lam1}");
        assert!((lam2 - 2.0).abs() < 1e-10, "lambda_2={lam2}");
    }

    // Test 2: [[2,0],[0,5]] — diagonal; larger eigenvalue first.
    #[test]
    fn test_eigenvalues_diagonal_2_5() {
        let m = const_mat(2.0, 0.0, 5.0);
        let eig = eig_symmetric_2x2(&m);
        let lam1 = ev(&eig.eigenvalues[0]);
        let lam2 = ev(&eig.eigenvalues[1]);
        assert!((lam1 - 5.0).abs() < 1e-10, "lambda_1={lam1}");
        assert!((lam2 - 2.0).abs() < 1e-10, "lambda_2={lam2}");
    }

    // Test 3: [[1,0],[0,1]] — identity; both eigenvalues are 1.
    #[test]
    fn test_eigenvalues_identity() {
        let m = const_mat(1.0, 0.0, 1.0);
        let eig = eig_symmetric_2x2(&m);
        let lam1 = ev(&eig.eigenvalues[0]);
        let lam2 = ev(&eig.eigenvalues[1]);
        assert!((lam1 - 1.0).abs() < 1e-10, "lambda_1={lam1}");
        assert!((lam2 - 1.0).abs() < 1e-10, "lambda_2={lam2}");
    }

    // Test 4: [[5,2],[2,-1]] — verify tr and det relations.
    // tr = 4, det = -9, so λ₁+λ₂ = 4 and λ₁*λ₂ = -9.
    #[test]
    fn test_eigenvalues_trace_det_relations() {
        let m = const_mat(5.0, 2.0, -1.0);
        let eig = eig_symmetric_2x2(&m);
        let lam1 = ev(&eig.eigenvalues[0]);
        let lam2 = ev(&eig.eigenvalues[1]);
        let trace = lam1 + lam2;
        let det = lam1 * lam2;
        assert!((trace - 4.0).abs() < 1e-9, "trace={trace}");
        assert!((det - (-9.0)).abs() < 1e-9, "det={det}");
        // Larger eigenvalue is first.
        assert!(lam1 > lam2, "expected lam1={lam1} > lam2={lam2}");
    }

    // Test 5: Eigenvector orthogonality for [[3,1],[1,3]].
    // v1=[1,1], v2=[1,-1], <v1,v2>=0.
    #[test]
    fn test_eigenvector_orthogonality_3_1_1_3() {
        let m = const_mat(3.0, 1.0, 3.0);
        let eig = eig_symmetric_2x2(&m);
        let v1 = &eig.eigenvectors[0];
        let v2 = &eig.eigenvectors[1];
        let dot = ev(&v1[0]) * ev(&v2[0]) + ev(&v1[1]) * ev(&v2[1]);
        assert!(dot.abs() < 1e-10, "<v1,v2>={dot}");
    }

    // Test 6: Eigenvector orthogonality for [[5,2],[2,-1]].
    #[test]
    fn test_eigenvector_orthogonality_5_2_2_m1() {
        let m = const_mat(5.0, 2.0, -1.0);
        let eig = eig_symmetric_2x2(&m);
        let v1 = &eig.eigenvectors[0];
        let v2 = &eig.eigenvectors[1];
        let dot = ev(&v1[0]) * ev(&v2[0]) + ev(&v1[1]) * ev(&v2[1]);
        assert!(dot.abs() < 1e-9, "<v1,v2>={dot}");
    }

    // Test 7: Symbolic [[a,b],[b,a]] — eval eigenvalues at a=3, b=1.
    // lambda_1 should eval to a+b=4, lambda_2 to a-b=2.
    #[test]
    fn test_symbolic_aab_eigenvalues() {
        // Var(0) = a, Var(1) = b
        let m = [
            [LoweredOp::Var(0), LoweredOp::Var(1)],
            [LoweredOp::Var(1), LoweredOp::Var(0)],
        ];
        let eig = eig_symmetric_2x2(&m);
        // At a=3, b=1: lambda_1 = a+b = 4, lambda_2 = a-b = 2.
        let bindings = [3.0_f64, 1.0_f64];
        let lam1 = ev_at(&eig.eigenvalues[0], &bindings);
        let lam2 = ev_at(&eig.eigenvalues[1], &bindings);
        assert!((lam1 - 4.0).abs() < 1e-9, "lam1={lam1}");
        assert!((lam2 - 2.0).abs() < 1e-9, "lam2={lam2}");
    }

    // Test 8: Repeated root [[a,0],[0,a]] — both eigenvalues equal a.
    #[test]
    fn test_repeated_root_scalar_multiple() {
        // Var(0) = a
        let m = [
            [LoweredOp::Var(0), LoweredOp::Const(0.0)],
            [LoweredOp::Const(0.0), LoweredOp::Var(0)],
        ];
        let eig = eig_symmetric_2x2(&m);
        let bindings = [5.0_f64];
        let lam1 = ev_at(&eig.eigenvalues[0], &bindings);
        let lam2 = ev_at(&eig.eigenvalues[1], &bindings);
        assert!((lam1 - 5.0).abs() < 1e-9, "lam1={lam1}");
        assert!((lam2 - 5.0).abs() < 1e-9, "lam2={lam2}");
    }
}
