//! `cas::matrix_exp` — closed-form symbolic matrix exponentials.
//!
//! Supports diagonal, nilpotent, 2×2 (exact via Cayley–Hamilton mean-shift),
//! and 3×3 (all-constant path via scaling-and-squaring Taylor series).
//!
//! # Algorithm: `expm_2x2` — Cayley–Hamilton mean-shift
//!
//! For a 2×2 matrix `M` with `trace t = (m00 + m11)/2`, define
//! `M' = M - tI`. Then:
//!
//! ```text
//! exp(M) = exp(t) * [cosh(δ)·I + sinh(δ)/δ · M']
//! ```
//!
//! where `δ = sqrt(det(-M') ) = sqrt(m'01·m'10 - m'00·m'11)`.
//!
//! The `sinh(δ)/δ` factor is well-defined for `δ → 0` (limit = 1) but
//! the raw symbolic expression evaluates to `0/0`; therefore **all output
//! cells are canonicalized** via [`crate::cas::canonicalize::canonicalize`]
//! before return, which folds the `Mul(Const(0), ...)` trees away. Additionally
//! a fast path bypasses the formula when `M'` is the zero matrix.
//!
//! # 3×3 path
//!
//! Only all-constant 3×3 matrices are supported symbolically. The entry values
//! are extracted as `f64`, the matrix exponential is computed numerically via
//! scaling-and-squaring with a degree-20 Taylor series, and the result is
//! reconstructed as `LoweredOp::Const` entries.

use std::fmt;

use crate::cas::canonicalize::canonicalize;
use crate::eml::eval::{eval_real, EvalCtx};
use crate::eml::op::LoweredOp;

// ---------------------------------------------------------------------------
// Private constructor helpers
// ---------------------------------------------------------------------------

#[inline]
fn c(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}

#[inline]
fn add2(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Box::new(a), Box::new(b))
}

#[inline]
fn sub2(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(Box::new(a), Box::new(b))
}

#[inline]
fn mul2(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Box::new(a), Box::new(b))
}

#[inline]
fn div2(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Div(Box::new(a), Box::new(b))
}

#[inline]
fn neg1(a: LoweredOp) -> LoweredOp {
    LoweredOp::Neg(Box::new(a))
}

#[inline]
fn exp1(a: LoweredOp) -> LoweredOp {
    LoweredOp::Exp(Box::new(a))
}

#[inline]
fn sqrt1(a: LoweredOp) -> LoweredOp {
    LoweredOp::Sqrt(Box::new(a))
}

#[inline]
fn cosh1(a: LoweredOp) -> LoweredOp {
    LoweredOp::Cosh(Box::new(a))
}

#[inline]
fn sinh1(a: LoweredOp) -> LoweredOp {
    LoweredOp::Sinh(Box::new(a))
}

/// Canonicalize a `LoweredOp` and extract the inner simplified expression.
#[inline]
fn canon(op: LoweredOp) -> LoweredOp {
    canonicalize(&op).into_op()
}

/// Test whether a `LoweredOp` is symbolically zero (after canonicalization).
///
/// Returns `true` iff the canonical form is `Const(v)` with `|v| < 1e-14`.
#[inline]
fn is_zero(op: &LoweredOp) -> bool {
    match canonicalize(op).into_op() {
        LoweredOp::Const(v) => v.abs() < 1e-14,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// 2×2 matrix multiplication in LoweredOp
// ---------------------------------------------------------------------------

/// Multiply two 2×2 symbolic matrices, returning a new 2×2 symbolic matrix.
///
/// Each entry `C[i][j] = Σ_k A[i][k] * B[k][j]` — expanded as `Add(Mul, Mul)`.
fn matmul_2x2(a: &[[LoweredOp; 2]; 2], b: &[[LoweredOp; 2]; 2]) -> [[LoweredOp; 2]; 2] {
    // c[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j]
    let entry = |i: usize, j: usize| -> LoweredOp {
        add2(
            mul2(a[i][0].clone(), b[0][j].clone()),
            mul2(a[i][1].clone(), b[1][j].clone()),
        )
    };
    [[entry(0, 0), entry(0, 1)], [entry(1, 0), entry(1, 1)]]
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Error type returned by [`expm_3x3`].
#[derive(Clone, Debug)]
pub enum MatrixExpError {
    /// The 3×3 matrix contains symbolic (non-constant) entries. Only
    /// all-constant matrices are supported in `expm_3x3`.
    CubicRootSymbolic,
    /// Evaluation of a constant entry failed (malformed constant subtree).
    EvalError(String),
}

impl fmt::Display for MatrixExpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixExpError::CubicRootSymbolic => write!(
                f,
                "expm_3x3 requires all-constant matrix entries; \
                 symbolic entries are not supported (use expm_2x2 for the 2×2 symbolic case)"
            ),
            MatrixExpError::EvalError(msg) => write!(f, "evaluation error in expm_3x3: {msg}"),
        }
    }
}

impl std::error::Error for MatrixExpError {}

// ---------------------------------------------------------------------------
// Diagonal helpers
// ---------------------------------------------------------------------------

/// Compute the symbolic matrix exponential of a **diagonal** 2×2 matrix.
///
/// Returns `None` if any off-diagonal entry is not symbolically zero.
/// The diagonal entries may be arbitrary symbolic `LoweredOp` expressions.
///
/// The result is `[[exp(m[0][0]), 0], [0, exp(m[1][1])]]`.
pub fn expm_diag_2x2(m: &[[LoweredOp; 2]; 2]) -> Option<[[LoweredOp; 2]; 2]> {
    if !is_zero(&m[0][1]) || !is_zero(&m[1][0]) {
        return None;
    }
    Some([
        [canon(exp1(m[0][0].clone())), c(0.0)],
        [c(0.0), canon(exp1(m[1][1].clone()))],
    ])
}

/// Compute the symbolic matrix exponential of a **diagonal** 3×3 matrix.
///
/// Returns `None` if any off-diagonal entry is not symbolically zero.
/// The diagonal entries may be arbitrary symbolic `LoweredOp` expressions.
///
/// The result is `diag(exp(m[0][0]), exp(m[1][1]), exp(m[2][2]))`.
pub fn expm_diag_3x3(m: &[[LoweredOp; 3]; 3]) -> Option<[[LoweredOp; 3]; 3]> {
    // Check all six off-diagonal entries.
    let off_diag_indices = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)];
    for (i, j) in off_diag_indices {
        if !is_zero(&m[i][j]) {
            return None;
        }
    }
    Some([
        [canon(exp1(m[0][0].clone())), c(0.0), c(0.0)],
        [c(0.0), canon(exp1(m[1][1].clone())), c(0.0)],
        [c(0.0), c(0.0), canon(exp1(m[2][2].clone()))],
    ])
}

// ---------------------------------------------------------------------------
// Nilpotent 2×2
// ---------------------------------------------------------------------------

/// Compute the matrix exponential of a **nilpotent** 2×2 matrix via truncated
/// Taylor series `exp(M) = I + M + M²/2! + … + M^(degree-1)/(degree-1)!`.
///
/// The caller asserts that `M^degree = 0`; `degree` must be ≥ 1.
/// For `degree = 1` only the identity term is included, giving `I`.
///
/// All entries are canonicalized before return.
pub fn expm_nilpotent_2x2(m: &[[LoweredOp; 2]; 2], degree: usize) -> [[LoweredOp; 2]; 2] {
    // Start with identity matrix (k=0 term: M^0 / 0! = I).
    let mut result: [[LoweredOp; 2]; 2] = [[c(1.0), c(0.0)], [c(0.0), c(1.0)]];

    // Current matrix power: starts at M^1 = M.
    let mut power: [[LoweredOp; 2]; 2] = [
        [m[0][0].clone(), m[0][1].clone()],
        [m[1][0].clone(), m[1][1].clone()],
    ];

    // Iteratively compute M^k / k! and accumulate.
    // k! computed as an integer to stay exact.
    let mut factorial: u64 = 1;
    for k in 1..degree {
        factorial = factorial.saturating_mul(k as u64);
        let inv_fact = 1.0 / (factorial as f64);
        // Add M^k / k! to result.
        for i in 0..2 {
            for j in 0..2 {
                let term = mul2(power[i][j].clone(), c(inv_fact));
                let old = result[i][j].clone();
                result[i][j] = add2(old, term);
            }
        }
        // Advance power: M^{k+1} = M^k * M (only needed if k+1 < degree).
        if k + 1 < degree {
            power = matmul_2x2(&power, m);
        }
    }

    // Canonicalize each entry.
    for row in &mut result {
        for entry in row.iter_mut() {
            let old = entry.clone();
            *entry = canon(old);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// General 2×2 — Cayley–Hamilton mean-shift
// ---------------------------------------------------------------------------

/// Compute the matrix exponential of a general **2×2 symbolic matrix** using
/// the Cayley–Hamilton mean-shift formula.
///
/// The result is exact in `LoweredOp` form using `cosh` and `sinh`:
///
/// ```text
/// exp(M) = exp(t) * [cosh(δ)·I + sinh(δ)/δ · M']
/// ```
///
/// where `t = (m00 + m11)/2`, `M' = M − tI`, and
/// `δ = sqrt(m'01·m'10 − m'00·m'11)`.
///
/// When `δ = 0` (i.e. `M'` is the zero matrix), the formula degenerates to
/// `exp(t)·I`. A fast path handles this case directly; otherwise all output
/// entries are canonicalized so that `Mul(Const(0), sinh_δ_over_δ)` subtrees
/// fold away and the expression remains well-behaved at `δ = 0`.
pub fn expm_2x2(m: &[[LoweredOp; 2]; 2]) -> [[LoweredOp; 2]; 2] {
    // t = (m00 + m11) / 2
    let trace_sum = add2(m[0][0].clone(), m[1][1].clone());
    let t = div2(trace_sum, c(2.0));

    // M' = M − tI
    let mp = [
        [sub2(m[0][0].clone(), t.clone()), m[0][1].clone()],
        [m[1][0].clone(), sub2(m[1][1].clone(), t.clone())],
    ];

    // Fast path: if M' is the zero matrix, result = exp(t)·I.
    let all_zero =
        is_zero(&mp[0][0]) && is_zero(&mp[0][1]) && is_zero(&mp[1][0]) && is_zero(&mp[1][1]);

    if all_zero {
        let et = canon(exp1(t));
        return [[et.clone(), c(0.0)], [c(0.0), et]];
    }

    // δ² = m'01 * m'10 - m'00 * m'11   (i.e. -det(M'))
    let delta_sq = sub2(
        mul2(mp[0][1].clone(), mp[1][0].clone()),
        mul2(mp[0][0].clone(), mp[1][1].clone()),
    );

    // δ = sqrt(δ²)
    let delta = sqrt1(delta_sq);

    // Scalar factors
    let et = exp1(t);
    let cosh_d = cosh1(delta.clone());
    let sinh_d = sinh1(delta.clone());
    // sinh(δ) / δ  — well-defined symbolically; canonicalize will fold
    // Mul(Const(0), <this>) to 0 for M'=0, but the fast path above
    // already handles that case.
    let sinh_over_delta = div2(sinh_d, delta);

    // result[i][j] = exp(t) * (cosh(δ) * I[i][j] + sinh(δ)/δ * m'[i][j])
    let entry = |i: usize, j: usize| -> LoweredOp {
        let identity_ij = if i == j { c(1.0) } else { c(0.0) };
        let cosh_term = mul2(cosh_d.clone(), identity_ij);
        let sinh_term = mul2(sinh_over_delta.clone(), mp[i][j].clone());
        mul2(et.clone(), add2(cosh_term, sinh_term))
    };

    // Build and canonicalize each entry.
    [
        [canon(entry(0, 0)), canon(entry(0, 1))],
        [canon(entry(1, 0)), canon(entry(1, 1))],
    ]
}

// ---------------------------------------------------------------------------
// 3×3 — all-constant path via scaling-and-squaring Taylor series
// ---------------------------------------------------------------------------

/// Matrix-multiply two 3×3 `f64` arrays (for the internal 3×3 computation).
fn matmul_3x3_f64(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c_mat = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0_f64;
            for k in 0..3 {
                s += a[i][k] * b[k][j];
            }
            c_mat[i][j] = s;
        }
    }
    c_mat
}

/// Add two 3×3 `f64` matrices element-wise.
fn matadd_3x3_f64(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    result
}

/// Scale a 3×3 `f64` matrix by a scalar.
fn matscale_3x3_f64(a: &[[f64; 3]; 3], s: f64) -> [[f64; 3]; 3] {
    let mut result = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i][j] = a[i][j] * s;
        }
    }
    result
}

/// 3×3 identity matrix as `f64`.
fn identity_3x3_f64() -> [[f64; 3]; 3] {
    let mut m = [[0.0_f64; 3]; 3];
    m[0][0] = 1.0;
    m[1][1] = 1.0;
    m[2][2] = 1.0;
    m
}

/// Infinity norm of a 3×3 matrix (max row-sum of absolute values).
fn inf_norm_3x3(m: &[[f64; 3]; 3]) -> f64 {
    let mut max_row = 0.0_f64;
    for row in m {
        let row_sum: f64 = row.iter().map(|v| v.abs()).sum();
        if row_sum > max_row {
            max_row = row_sum;
        }
    }
    max_row
}

/// Matrix exponential of a 3×3 `f64` matrix via scaling-and-squaring with
/// a degree-20 Taylor series: `exp(M) = (exp(M/s))^s` where `s = 2^k`.
///
/// `k` is chosen so `‖M/s‖_∞ ≤ 0.5`.
fn expm_3x3_f64(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    const TAYLOR_TERMS: usize = 20;

    // --- Scaling ---
    let norm = inf_norm_3x3(m);
    let mut k: u32 = 0;
    let mut scale = 1.0_f64;
    while norm / scale > 0.5 {
        k += 1;
        scale *= 2.0;
        if k > 30 {
            // Limit scaling to avoid underflow in entries.
            break;
        }
    }

    // A = M / 2^k
    let a = matscale_3x3_f64(m, 1.0 / scale);

    // --- Taylor series: T = Σ_{i=0}^{TAYLOR_TERMS-1} A^i / i! ---
    let mut t = identity_3x3_f64(); // A^0 / 0! = I
    let mut power = identity_3x3_f64(); // current A^i
    let mut factorial = 1.0_f64;

    for i in 1..TAYLOR_TERMS {
        factorial *= i as f64;
        power = matmul_3x3_f64(&power, &a);
        let term = matscale_3x3_f64(&power, 1.0 / factorial);
        t = matadd_3x3_f64(&t, &term);
    }

    // --- Squaring: T^(2^k) ---
    for _ in 0..k {
        t = matmul_3x3_f64(&t, &t);
    }

    t
}

/// Compute the matrix exponential of a general **3×3 symbolic matrix**.
///
/// Only matrices whose every entry is a `LoweredOp::Const` (after
/// canonicalization) are supported. For matrices with symbolic (non-constant)
/// entries, returns `Err(MatrixExpError::CubicRootSymbolic)`.
///
/// The computation proceeds by:
/// 1. Extracting each entry as an `f64` value.
/// 2. Computing the matrix exponential numerically via scaling-and-squaring
///    with a 20-term Taylor series.
/// 3. Reconstructing the result as `LoweredOp::Const` entries.
pub fn expm_3x3(m: &[[LoweredOp; 3]; 3]) -> Result<[[LoweredOp; 3]; 3], MatrixExpError> {
    // Extract constant f64 values from each entry.
    let ctx = EvalCtx::new(&[]);
    let mut vals = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let canonical = canonicalize(&m[i][j]).into_op();
            match canonical {
                LoweredOp::Const(v) => {
                    vals[i][j] = v;
                }
                _ => {
                    // Try evaluating (in case it simplifies to a constant
                    // expression like Sub(Const(2), Const(1))).
                    match eval_real(&m[i][j], &ctx) {
                        Ok(v) => vals[i][j] = v,
                        Err(_) => return Err(MatrixExpError::CubicRootSymbolic),
                    }
                }
            }
        }
    }

    let result_f64 = expm_3x3_f64(&vals);

    // Reconstruct as LoweredOp::Const.
    let result_op = [
        [
            c(result_f64[0][0]),
            c(result_f64[0][1]),
            c(result_f64[0][2]),
        ],
        [
            c(result_f64[1][0]),
            c(result_f64[1][1]),
            c(result_f64[1][2]),
        ],
        [
            c(result_f64[2][0]),
            c(result_f64[2][1]),
            c(result_f64[2][2]),
        ],
    ];
    Ok(result_op)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::eval::{eval_real, EvalCtx};

    /// Evaluate a `LoweredOp` with no variable bindings.
    fn eval_const(op: &LoweredOp) -> f64 {
        let ctx = EvalCtx::new(&[]);
        eval_real(op, &ctx).expect("eval_const: expression must be constant")
    }

    // -----------------------------------------------------------------------
    // Test 1 — diagonal 2×2
    // -----------------------------------------------------------------------
    #[test]
    fn test_expm_diag_2x2_basic() {
        let m = [
            [LoweredOp::Const(1.0), LoweredOp::Const(0.0)],
            [LoweredOp::Const(0.0), LoweredOp::Const(2.0)],
        ];
        let result = expm_diag_2x2(&m).expect("should be diagonal");
        let e = std::f64::consts::E;
        let tol = 1e-12;
        assert!((eval_const(&result[0][0]) - e).abs() < tol, "diag[0][0]");
        assert!((eval_const(&result[0][1])).abs() < tol, "off[0][1]");
        assert!((eval_const(&result[1][0])).abs() < tol, "off[1][0]");
        assert!(
            (eval_const(&result[1][1]) - e * e).abs() < tol,
            "diag[1][1]"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2 — nilpotent 2×2
    // -----------------------------------------------------------------------
    #[test]
    fn test_expm_nilpotent_2x2_shift() {
        // M = [[0, 1],[0, 0]] is nilpotent, M² = 0.
        let m = [
            [LoweredOp::Const(0.0), LoweredOp::Const(1.0)],
            [LoweredOp::Const(0.0), LoweredOp::Const(0.0)],
        ];
        let result = expm_nilpotent_2x2(&m, 2);
        let tol = 1e-12;
        // exp(M) = I + M = [[1,1],[0,1]]
        assert!((eval_const(&result[0][0]) - 1.0).abs() < tol, "[0][0]");
        assert!((eval_const(&result[0][1]) - 1.0).abs() < tol, "[0][1]");
        assert!((eval_const(&result[1][0])).abs() < tol, "[1][0]");
        assert!((eval_const(&result[1][1]) - 1.0).abs() < tol, "[1][1]");
    }

    // -----------------------------------------------------------------------
    // Test 3 — expm_2x2 of zero matrix → identity
    // -----------------------------------------------------------------------
    #[test]
    fn test_expm_2x2_zero_matrix() {
        let m = [
            [LoweredOp::Const(0.0), LoweredOp::Const(0.0)],
            [LoweredOp::Const(0.0), LoweredOp::Const(0.0)],
        ];
        let result = expm_2x2(&m);
        let tol = 1e-12;
        assert!((eval_const(&result[0][0]) - 1.0).abs() < tol, "[0][0]");
        assert!((eval_const(&result[0][1])).abs() < tol, "[0][1]");
        assert!((eval_const(&result[1][0])).abs() < tol, "[1][0]");
        assert!((eval_const(&result[1][1]) - 1.0).abs() < tol, "[1][1]");
    }

    // -----------------------------------------------------------------------
    // Test 4 — diagonal 3×3
    // -----------------------------------------------------------------------
    #[test]
    fn test_expm_diag_3x3_basic() {
        let z = LoweredOp::Const(0.0);
        let m = [
            [LoweredOp::Const(1.0), z.clone(), z.clone()],
            [z.clone(), LoweredOp::Const(2.0), z.clone()],
            [z.clone(), z.clone(), LoweredOp::Const(3.0)],
        ];
        let result = expm_diag_3x3(&m).expect("should be diagonal");
        let e = std::f64::consts::E;
        let tol = 1e-10;
        assert!((eval_const(&result[0][0]) - e).abs() < tol, "diag[0]");
        assert!((eval_const(&result[1][1]) - e * e).abs() < tol, "diag[1]");
        assert!(
            (eval_const(&result[2][2]) - e * e * e).abs() < tol,
            "diag[2]"
        );
        // Off-diagonal entries should be zero.
        assert!((eval_const(&result[0][1])).abs() < tol, "off[0][1]");
        assert!((eval_const(&result[1][2])).abs() < tol, "off[1][2]");
    }

    // -----------------------------------------------------------------------
    // Test 5 — expm_2x2 of identity → [[e, 0],[0, e]]
    // -----------------------------------------------------------------------
    #[test]
    fn test_expm_2x2_identity() {
        let m = [
            [LoweredOp::Const(1.0), LoweredOp::Const(0.0)],
            [LoweredOp::Const(0.0), LoweredOp::Const(1.0)],
        ];
        let result = expm_2x2(&m);
        let e = std::f64::consts::E;
        let tol = 1e-10;
        assert!((eval_const(&result[0][0]) - e).abs() < tol, "[0][0]");
        assert!((eval_const(&result[0][1])).abs() < tol, "[0][1]");
        assert!((eval_const(&result[1][0])).abs() < tol, "[1][0]");
        assert!((eval_const(&result[1][1]) - e).abs() < tol, "[1][1]");
    }

    // -----------------------------------------------------------------------
    // Test 6 — expm_3x3 of identity → diag(e, e, e)
    // -----------------------------------------------------------------------
    #[test]
    fn test_expm_3x3_identity() {
        let z = LoweredOp::Const(0.0);
        let m = [
            [LoweredOp::Const(1.0), z.clone(), z.clone()],
            [z.clone(), LoweredOp::Const(1.0), z.clone()],
            [z.clone(), z.clone(), LoweredOp::Const(1.0)],
        ];
        let result = expm_3x3(&m).expect("should succeed for constant matrix");
        let e = std::f64::consts::E;
        let tol = 1e-10;
        assert!((eval_const(&result[0][0]) - e).abs() < tol, "[0][0]");
        assert!((eval_const(&result[1][1]) - e).abs() < tol, "[1][1]");
        assert!((eval_const(&result[2][2]) - e).abs() < tol, "[2][2]");
        assert!((eval_const(&result[0][1])).abs() < tol, "[0][1]");
        assert!((eval_const(&result[0][2])).abs() < tol, "[0][2]");
    }

    // -----------------------------------------------------------------------
    // Test 7 — expm_2x2(M) · expm_2x2(−M) ≈ I  (inverse property)
    // -----------------------------------------------------------------------
    #[test]
    fn test_expm_2x2_inverse_property() {
        let m = [
            [LoweredOp::Const(1.0), LoweredOp::Const(2.0)],
            [LoweredOp::Const(3.0), LoweredOp::Const(4.0)],
        ];
        let neg_m = [
            [LoweredOp::Const(-1.0), LoweredOp::Const(-2.0)],
            [LoweredOp::Const(-3.0), LoweredOp::Const(-4.0)],
        ];
        let em = expm_2x2(&m);
        let enm = expm_2x2(&neg_m);

        // Extract f64 values.
        let em_f = [
            [eval_const(&em[0][0]), eval_const(&em[0][1])],
            [eval_const(&em[1][0]), eval_const(&em[1][1])],
        ];
        let enm_f = [
            [eval_const(&enm[0][0]), eval_const(&enm[0][1])],
            [eval_const(&enm[1][0]), eval_const(&enm[1][1])],
        ];

        // Multiply em_f * enm_f = should be I.
        let prod = [
            [
                em_f[0][0] * enm_f[0][0] + em_f[0][1] * enm_f[1][0],
                em_f[0][0] * enm_f[0][1] + em_f[0][1] * enm_f[1][1],
            ],
            [
                em_f[1][0] * enm_f[0][0] + em_f[1][1] * enm_f[1][0],
                em_f[1][0] * enm_f[0][1] + em_f[1][1] * enm_f[1][1],
            ],
        ];
        let tol = 1e-8;
        assert!((prod[0][0] - 1.0).abs() < tol, "I[0][0] = {}", prod[0][0]);
        assert!((prod[0][1]).abs() < tol, "I[0][1] = {}", prod[0][1]);
        assert!((prod[1][0]).abs() < tol, "I[1][0] = {}", prod[1][0]);
        assert!((prod[1][1] - 1.0).abs() < tol, "I[1][1] = {}", prod[1][1]);
    }

    // -----------------------------------------------------------------------
    // Test 8 — nilpotent degree=1 → identity
    // -----------------------------------------------------------------------
    #[test]
    fn test_expm_nilpotent_2x2_degree1() {
        let m = [
            [LoweredOp::Const(0.0), LoweredOp::Const(1.0)],
            [LoweredOp::Const(0.0), LoweredOp::Const(0.0)],
        ];
        let result = expm_nilpotent_2x2(&m, 1);
        let tol = 1e-12;
        // Only k=0 term: I.
        assert!((eval_const(&result[0][0]) - 1.0).abs() < tol, "[0][0]");
        assert!((eval_const(&result[0][1])).abs() < tol, "[0][1]");
        assert!((eval_const(&result[1][0])).abs() < tol, "[1][0]");
        assert!((eval_const(&result[1][1]) - 1.0).abs() < tol, "[1][1]");
    }

    // -----------------------------------------------------------------------
    // Test 9 — off-diagonal non-zero returns None
    // -----------------------------------------------------------------------
    #[test]
    fn test_expm_diag_2x2_non_diagonal_returns_none() {
        // m[0][1] = Const(1.0) — non-zero off-diagonal.
        let m = [
            [LoweredOp::Var(0), LoweredOp::Const(1.0)],
            [LoweredOp::Const(0.0), LoweredOp::Const(0.0)],
        ];
        assert!(
            expm_diag_2x2(&m).is_none(),
            "non-diagonal should return None"
        );
    }

    // -----------------------------------------------------------------------
    // Test 10 — canonicalize idempotent on expm_2x2 result
    // -----------------------------------------------------------------------
    #[test]
    fn test_expm_2x2_canon_idempotent() {
        let m = [
            [LoweredOp::Const(1.0), LoweredOp::Const(2.0)],
            [LoweredOp::Const(3.0), LoweredOp::Const(4.0)],
        ];
        let result = expm_2x2(&m);
        // Canonicalize the top-left entry twice and check hash equality.
        let c1 = canonicalize(&result[0][0]);
        let c2 = canonicalize(c1.op());
        assert_eq!(
            c1.hash(),
            c2.hash(),
            "canonicalize should be idempotent (same hash on second pass)"
        );
    }
}
