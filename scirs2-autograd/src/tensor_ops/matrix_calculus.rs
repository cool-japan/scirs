//! Matrix-calculus kernels shared by the matrix-function operations.
//!
//! This module holds the numerical engine behind the *forward* values and the
//! *vector-Jacobian products* (VJPs) of the matrix functions in
//! [`crate::tensor_ops::matrix_ops`], [`crate::tensor_ops::matrix_trig_functions`] and
//! [`crate::tensor_ops::decomposition_ops::cholesky_eigen_ops`].
//!
//! # Why a shared engine
//!
//! Every one of those ops previously shipped a *different* placeholder: a diagonal-only
//! "matrix exponential", a matrix logarithm that took `ln` of the diagonal, a matrix
//! power that ignored the off-diagonal entirely, and backward passes that either passed
//! the cotangent straight through or multiplied it element-wise by `cos(A)`. All of them
//! produce plausible-looking numbers that are simply wrong. This module replaces them
//! with algorithms that are correct for general (not just diagonal) matrices.
//!
//! # The Fréchet derivative and the VJP
//!
//! For a matrix function `f`, the Fréchet derivative `L_f(A, E)` is the unique linear
//! operator with `f(A + E) = f(A) + L_f(A, E) + o(||E||)`. Reverse-mode needs its
//! *adjoint*: given an output cotangent `Ḡ`, the gradient w.r.t. `A` is the matrix `Ā`
//! satisfying `<Ā, E> = <Ḡ, L_f(A, E)>` for every `E`.
//!
//! Two facts make this computable exactly:
//!
//! 1. **Block-matrix identity** (Higham, *Functions of Matrices*, Thm. 3.6):
//!    ```text
//!    f( [ A  E ]  ) = [ f(A)  L_f(A, E) ]
//!      ( [ 0  A ]  )   [  0      f(A)   ]
//!    ```
//!    so one evaluation of `f` on a `2n x 2n` block-triangular matrix yields the Fréchet
//!    derivative in its top-right block.
//!
//! 2. **Adjoint identity**: for real `A` and a power series with real coefficients,
//!    `L_f(A, ·)* = L_f(Aᵀ, ·)`. (Each term `A^j E A^{k-1-j}` contributes
//!    `tr(Ḡᵀ A^j E A^{k-1-j}) = <(Aᵀ)^j Ḡ (Aᵀ)^{k-1-j}, E>`.)
//!
//! Combining them, the VJP is the top-right block of `f([[Aᵀ, Ḡ], [0, Aᵀ]])`, which is
//! exactly what [`frechet_vjp`] computes. Because `L_f(A, ·)` is linear in its second
//! argument, `Ḡ` is normalised before the block is formed and the result is rescaled;
//! that keeps the `2n x 2n` argument well scaled no matter how large the cotangent is.

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::Float;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Ix2};

// ---------------------------------------------------------------------------
// Small dense-linear-algebra helpers
// ---------------------------------------------------------------------------

/// Largest absolute entry of a matrix.
pub(crate) fn max_abs<F: Float>(a: &ArrayView2<F>) -> F {
    a.iter().fold(F::zero(), |acc, &v| {
        let m = v.abs();
        if m > acc {
            m
        } else {
            acc
        }
    })
}

/// Maximum absolute row sum (the induced infinity norm).
pub(crate) fn inf_norm<F: Float>(a: &ArrayView2<F>) -> F {
    let mut best = F::zero();
    for row in a.rows() {
        let s = row.iter().fold(F::zero(), |acc, &v| acc + v.abs());
        if s > best {
            best = s;
        }
    }
    best
}

fn constant<F: Float>(v: f64) -> Result<F, OpError> {
    F::from(v).ok_or_else(|| {
        OpError::Other(format!(
            "matrix calculus: cannot represent the constant {v} in the working float type"
        ))
    })
}

/// Dense inverse by Gauss-Jordan elimination with partial pivoting.
pub(crate) fn inverse<F: Float>(a: &ArrayView2<F>) -> Result<Array2<F>, OpError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(OpError::IncompatibleShape(
            "matrix inverse requires a square matrix".into(),
        ));
    }
    let mut work = a.to_owned();
    let mut inv = Array2::<F>::eye(n);
    let scale = max_abs(a);
    // Relative singularity threshold: an absolute `epsilon` test declares a well
    // conditioned but small-magnitude matrix singular, and a badly scaled but singular
    // matrix non-singular.
    let tol = if scale > F::zero() {
        scale * F::epsilon() * constant::<F>(16.0)?
    } else {
        F::epsilon()
    };

    for col in 0..n {
        let mut pivot_row = col;
        for row in (col + 1)..n {
            if work[[row, col]].abs() > work[[pivot_row, col]].abs() {
                pivot_row = row;
            }
        }
        if work[[pivot_row, col]].abs() <= tol {
            return Err(OpError::Other(
                "matrix calculus: matrix is singular to working precision".into(),
            ));
        }
        if pivot_row != col {
            for j in 0..n {
                work.swap((col, j), (pivot_row, j));
                inv.swap((col, j), (pivot_row, j));
            }
        }
        let pivot = work[[col, col]];
        for j in 0..n {
            work[[col, j]] /= pivot;
            inv[[col, j]] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = work[[row, col]];
            if factor == F::zero() {
                continue;
            }
            for j in 0..n {
                let wj = work[[col, j]];
                let ij = inv[[col, j]];
                work[[row, j]] -= factor * wj;
                inv[[row, j]] -= factor * ij;
            }
        }
    }
    Ok(inv)
}

/// `true` when `a` equals its own transpose to a *relative* tolerance.
pub(crate) fn is_symmetric<F: Float>(a: &ArrayView2<F>) -> bool {
    let n = a.nrows();
    if a.ncols() != n {
        return false;
    }
    let scale = max_abs(a);
    let tol = match F::from(1e-10) {
        Some(t) => t * (scale + F::one()),
        None => F::epsilon(),
    };
    for i in 0..n {
        for j in (i + 1)..n {
            if (a[[i, j]] - a[[j, i]]).abs() > tol {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Symmetric eigendecomposition (cyclic Jacobi)
// ---------------------------------------------------------------------------

/// Eigendecomposition of a real symmetric matrix by the cyclic Jacobi method.
///
/// Returns `(values, vectors)` with the eigenvalues sorted in **descending** order and
/// `vectors` holding the matching unit eigenvectors as **columns**, so that
/// `A == vectors * diag(values) * vectorsᵀ`.
///
/// Each eigenvector's sign is fixed by requiring its largest-magnitude component to be
/// positive. Without that the output would flip sign arbitrarily between calls on nearly
/// identical inputs, which breaks any downstream finite-difference comparison.
///
/// The input is symmetrised (`(A + Aᵀ)/2`) first: callers that reach here after a
/// tolerance check may still carry rounding-level asymmetry, and Jacobi assumes exact
/// symmetry.
pub(crate) fn symmetric_eigen<F: Float>(
    a: &ArrayView2<F>,
) -> Result<(Array1<F>, Array2<F>), OpError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(OpError::IncompatibleShape(
            "symmetric eigendecomposition requires a square matrix".into(),
        ));
    }
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }

    let two = constant::<F>(2.0)?;
    let half = constant::<F>(0.5)?;

    let mut work = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            work[[i, j]] = (a[[i, j]] + a[[j, i]]) * half;
        }
    }
    let mut vecs = Array2::<F>::eye(n);

    // 60 sweeps is far beyond the 6-10 quadratically convergent sweeps Jacobi needs in
    // practice; it exists only so a pathological input cannot spin forever.
    let max_sweeps = 60;
    let scale = max_abs(&work.view());
    let tol = F::epsilon() * (scale + F::one());

    for _ in 0..max_sweeps {
        let mut off = F::zero();
        for i in 0..n {
            for j in (i + 1)..n {
                off += work[[i, j]].abs();
            }
        }
        if off <= tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = work[[p, q]];
                if apq.abs() <= tol {
                    continue;
                }
                let app = work[[p, p]];
                let aqq = work[[q, q]];
                let theta = (aqq - app) / (two * apq);
                let t = if theta >= F::zero() {
                    F::one() / (theta + (F::one() + theta * theta).sqrt())
                } else {
                    -F::one() / (-theta + (F::one() + theta * theta).sqrt())
                };
                let c = F::one() / (F::one() + t * t).sqrt();
                let s = t * c;

                for k in 0..n {
                    let akp = work[[k, p]];
                    let akq = work[[k, q]];
                    work[[k, p]] = c * akp - s * akq;
                    work[[k, q]] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = work[[p, k]];
                    let aqk = work[[q, k]];
                    work[[p, k]] = c * apk - s * aqk;
                    work[[q, k]] = s * apk + c * aqk;
                }
                work[[p, q]] = F::zero();
                work[[q, p]] = F::zero();

                for k in 0..n {
                    let vkp = vecs[[k, p]];
                    let vkq = vecs[[k, q]];
                    vecs[[k, p]] = c * vkp - s * vkq;
                    vecs[[k, q]] = s * vkp + c * vkq;
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        work[[j, j]]
            .partial_cmp(&work[[i, i]])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut values = Array1::<F>::zeros(n);
    let mut vectors = Array2::<F>::zeros((n, n));
    for (new_idx, &old_idx) in order.iter().enumerate() {
        values[new_idx] = work[[old_idx, old_idx]];
        // Deterministic sign: largest-magnitude component made positive.
        let mut pivot = 0usize;
        let mut best = F::zero();
        for k in 0..n {
            let m = vecs[[k, old_idx]].abs();
            if m > best {
                best = m;
                pivot = k;
            }
        }
        let flip = vecs[[pivot, old_idx]] < F::zero();
        for k in 0..n {
            let v = vecs[[k, old_idx]];
            vectors[[k, new_idx]] = if flip { -v } else { v };
        }
    }
    Ok((values, vectors))
}

// ---------------------------------------------------------------------------
// Matrix functions
// ---------------------------------------------------------------------------

/// Matrix exponential by scaling-and-squaring with a truncated Taylor series.
///
/// `A` is halved until `||A||_inf <= 1/4`, where 24 Taylor terms are accurate to well
/// below `f64` rounding, and the result is squared back.
pub(crate) fn expm<F: Float>(a: &ArrayView2<F>) -> Result<Array2<F>, OpError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(OpError::IncompatibleShape(
            "matrix exponential requires a square matrix".into(),
        ));
    }
    let half = constant::<F>(0.5)?;
    let quarter = constant::<F>(0.25)?;

    let mut scaled = a.to_owned();
    let mut squarings = 0usize;
    let mut norm = inf_norm(&scaled.view());
    if !norm.is_finite() {
        return Err(OpError::Other(
            "matrix exponential: input contains non-finite entries".into(),
        ));
    }
    while norm > quarter && squarings < 128 {
        scaled.mapv_inplace(|v| v * half);
        squarings += 1;
        norm = inf_norm(&scaled.view());
    }

    let mut result = Array2::<F>::eye(n);
    let mut term = Array2::<F>::eye(n);
    for k in 1..=24u32 {
        let kf = constant::<F>(f64::from(k))?;
        term = term.dot(&scaled).mapv(|v| v / kf);
        result += &term;
    }
    for _ in 0..squarings {
        result = result.dot(&result);
    }
    Ok(result)
}

/// Principal matrix square root by the Denman-Beavers iteration.
///
/// Converges for every matrix with no eigenvalue on the closed negative real axis, and
/// preserves block-triangular structure (each step is built from inverses and sums),
/// which is what makes the Fréchet block trick work through [`logm`].
pub(crate) fn sqrtm<F: Float>(a: &ArrayView2<F>) -> Result<Array2<F>, OpError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(OpError::IncompatibleShape(
            "matrix square root requires a square matrix".into(),
        ));
    }
    let half = constant::<F>(0.5)?;
    let mut y = a.to_owned();
    let mut z = Array2::<F>::eye(n);
    let tol = F::epsilon() * constant::<F>(64.0)?;

    for _ in 0..100 {
        let y_inv = inverse(&y.view())?;
        let z_inv = inverse(&z.view())?;
        let y_next = (&y + &z_inv).mapv(|v| v * half);
        let z_next = (&z + &y_inv).mapv(|v| v * half);
        let delta = max_abs(&(&y_next - &y).view());
        y = y_next;
        z = z_next;
        if delta <= tol * (F::one() + max_abs(&y.view())) {
            break;
        }
    }
    Ok(y)
}

/// Principal matrix logarithm by inverse scaling and squaring.
///
/// Repeated square roots pull `A` towards the identity; once `||X - I||` is small the
/// logarithm is evaluated from the rapidly convergent `2 atanh(Z)` series with
/// `Z = (X - I)(X + I)^{-1}`, and the result is scaled back by `2^k`.
pub(crate) fn logm<F: Float>(a: &ArrayView2<F>) -> Result<Array2<F>, OpError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(OpError::IncompatibleShape(
            "matrix logarithm requires a square matrix".into(),
        ));
    }
    let two = constant::<F>(2.0)?;
    let quarter = constant::<F>(0.25)?;
    let eye = Array2::<F>::eye(n);

    let mut x = a.to_owned();
    let mut roots = 0usize;
    while max_abs(&(&x - &eye).view()) > quarter {
        if roots >= 60 {
            return Err(OpError::Other(
                "matrix logarithm: inverse scaling and squaring did not reach the \
                 convergence region (the matrix may have a negative real eigenvalue)"
                    .into(),
            ));
        }
        x = sqrtm(&x.view())?;
        roots += 1;
    }

    let z = (&x - &eye).dot(&inverse(&(&x + &eye).view())?);
    let z2 = z.dot(&z);
    let mut term = z.clone();
    let mut acc = z;
    for j in 1..=24u32 {
        term = term.dot(&z2);
        let denom = constant::<F>(f64::from(2 * j + 1))?;
        acc += &term.mapv(|v| v / denom);
    }

    let mut scale = two;
    for _ in 0..roots {
        scale *= two;
    }
    Ok(acc.mapv(|v| v * scale))
}

/// Matrix sine and cosine, computed together by scaling and double-angle recovery.
pub(crate) fn sin_cos_m<F: Float>(a: &ArrayView2<F>) -> Result<(Array2<F>, Array2<F>), OpError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(OpError::IncompatibleShape(
            "matrix sine/cosine requires a square matrix".into(),
        ));
    }
    let half = constant::<F>(0.5)?;
    let quarter = constant::<F>(0.25)?;
    let two = constant::<F>(2.0)?;

    let mut scaled = a.to_owned();
    let mut doublings = 0usize;
    let mut norm = inf_norm(&scaled.view());
    if !norm.is_finite() {
        return Err(OpError::Other(
            "matrix sine/cosine: input contains non-finite entries".into(),
        ));
    }
    while norm > quarter && doublings < 128 {
        scaled.mapv_inplace(|v| v * half);
        doublings += 1;
        norm = inf_norm(&scaled.view());
    }

    let a2 = scaled.dot(&scaled);
    let mut sin = scaled.clone();
    let mut sin_term = scaled;
    let mut cos = Array2::<F>::eye(n);
    let mut cos_term = Array2::<F>::eye(n);
    for k in 1..=16u32 {
        let sd = constant::<F>(f64::from((2 * k) * (2 * k + 1)))?;
        sin_term = sin_term.dot(&a2).mapv(|v| -v / sd);
        sin += &sin_term;

        let cd = constant::<F>(f64::from((2 * k - 1) * (2 * k)))?;
        cos_term = cos_term.dot(&a2).mapv(|v| -v / cd);
        cos += &cos_term;
    }

    for _ in 0..doublings {
        // sin(2X) = 2 sin X cos X, cos(2X) = cos^2 X - sin^2 X.
        let sin_next = sin.dot(&cos).mapv(|v| v * two);
        let cos_next = cos.dot(&cos) - sin.dot(&sin);
        sin = sin_next;
        cos = cos_next;
    }
    Ok((sin, cos))
}

/// Matrix hyperbolic sine and cosine from `expm(A)` and `expm(-A)`.
pub(crate) fn sinh_cosh_m<F: Float>(a: &ArrayView2<F>) -> Result<(Array2<F>, Array2<F>), OpError> {
    let half = constant::<F>(0.5)?;
    let exp_a = expm(a)?;
    let neg = a.mapv(|v| -v);
    let exp_neg = expm(&neg.view())?;
    let sinh = (&exp_a - &exp_neg).mapv(|v| v * half);
    let cosh = (&exp_a + &exp_neg).mapv(|v| v * half);
    Ok((sinh, cosh))
}

/// Matrix sign function by the (unscaled) Newton iteration `X <- (X + X^{-1})/2`.
pub(crate) fn signm<F: Float>(a: &ArrayView2<F>) -> Result<Array2<F>, OpError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(OpError::IncompatibleShape(
            "matrix sign requires a square matrix".into(),
        ));
    }
    let half = constant::<F>(0.5)?;
    let tol = F::epsilon() * constant::<F>(64.0)?;
    let mut x = a.to_owned();
    for _ in 0..200 {
        let x_inv = inverse(&x.view())?;
        let x_next = (&x + &x_inv).mapv(|v| v * half);
        let delta = max_abs(&(&x_next - &x).view());
        x = x_next;
        if delta <= tol * (F::one() + max_abs(&x.view())) {
            break;
        }
    }
    Ok(x)
}

/// Integer matrix power by binary exponentiation.
fn int_powm<F: Float>(a: &ArrayView2<F>, exponent: u64) -> Array2<F> {
    let n = a.nrows();
    let mut result = Array2::<F>::eye(n);
    let mut base = a.to_owned();
    let mut e = exponent;
    while e > 0 {
        if e & 1 == 1 {
            result = result.dot(&base);
        }
        e >>= 1;
        if e > 0 {
            base = base.dot(&base);
        }
    }
    result
}

/// Matrix power `A^p`.
///
/// Integer exponents use binary exponentiation (exact, and valid for singular `A` when
/// `p >= 0`); non-integer exponents use the principal branch `exp(p log A)`.
pub(crate) fn powm<F: Float>(a: &ArrayView2<F>, p: f64) -> Result<Array2<F>, OpError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(OpError::IncompatibleShape(
            "matrix power requires a square matrix".into(),
        ));
    }
    if p == 0.0 {
        return Ok(Array2::<F>::eye(n));
    }
    if p.fract() == 0.0 && p.abs() <= 4096.0 {
        let magnitude = p.abs() as u64;
        if p > 0.0 {
            return Ok(int_powm(a, magnitude));
        }
        let inv = inverse(a)?;
        return Ok(int_powm(&inv.view(), magnitude));
    }
    let log_a = logm(a)?;
    let pf = constant::<F>(p)?;
    expm(&log_a.mapv(|v| v * pf).view())
}

// ---------------------------------------------------------------------------
// Function selector + Fréchet-derivative VJP
// ---------------------------------------------------------------------------

/// Which matrix function a [`MatrixFnVjpOp`] differentiates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum MatrixFnKind {
    /// `exp(A)`
    Exp,
    /// `log(A)` (principal branch)
    Log,
    /// `sin(A)`
    Sin,
    /// `cos(A)`
    Cos,
    /// `sinh(A)`
    Sinh,
    /// `cosh(A)`
    Cosh,
    /// `sign(A)`
    Sign,
    /// `A^p`
    Power(f64),
}

impl MatrixFnKind {
    /// Human-readable name, used in error messages.
    pub(crate) fn label(self) -> &'static str {
        match self {
            MatrixFnKind::Exp => "matrix exponential",
            MatrixFnKind::Log => "matrix logarithm",
            MatrixFnKind::Sin => "matrix sine",
            MatrixFnKind::Cos => "matrix cosine",
            MatrixFnKind::Sinh => "matrix hyperbolic sine",
            MatrixFnKind::Cosh => "matrix hyperbolic cosine",
            MatrixFnKind::Sign => "matrix sign",
            MatrixFnKind::Power(_) => "matrix power",
        }
    }
}

/// Evaluates the selected matrix function.
pub(crate) fn apply_matrix_fn<F: Float>(
    a: &ArrayView2<F>,
    kind: MatrixFnKind,
) -> Result<Array2<F>, OpError> {
    match kind {
        MatrixFnKind::Exp => expm(a),
        MatrixFnKind::Log => logm(a),
        MatrixFnKind::Sin => sin_cos_m(a).map(|(s, _)| s),
        MatrixFnKind::Cos => sin_cos_m(a).map(|(_, c)| c),
        MatrixFnKind::Sinh => sinh_cosh_m(a).map(|(s, _)| s),
        MatrixFnKind::Cosh => sinh_cosh_m(a).map(|(_, c)| c),
        MatrixFnKind::Sign => signm(a),
        MatrixFnKind::Power(p) => powm(a, p),
    }
}

/// Vector-Jacobian product of a matrix function.
///
/// Returns `Ā` with `<Ā, E> = <gy, L_f(A, E)>` for every direction `E`, computed as the
/// top-right block of `f([[Aᵀ, gy], [0, Aᵀ]])` (see the module docs for the derivation).
///
/// `gy` is rescaled to unit magnitude before the block matrix is built, because every
/// algorithm here reduces its argument towards a convergence region and an unnormalised
/// cotangent would inflate that argument for no reason. Linearity in `gy` makes the
/// rescaling exact.
pub(crate) fn frechet_vjp<F: Float>(
    a: &ArrayView2<F>,
    gy: &ArrayView2<F>,
    kind: MatrixFnKind,
) -> Result<Array2<F>, OpError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(OpError::IncompatibleShape(format!(
            "{}: gradient requires a square matrix",
            kind.label()
        )));
    }
    if gy.nrows() != n || gy.ncols() != n {
        return Err(OpError::IncompatibleShape(format!(
            "{}: output cotangent has shape {:?} but the input is {n}x{n}",
            kind.label(),
            gy.shape()
        )));
    }

    let scale = max_abs(gy);
    if scale == F::zero() {
        return Ok(Array2::<F>::zeros((n, n)));
    }

    let mut block = Array2::<F>::zeros((2 * n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            // Transposed diagonal blocks: the adjoint of L_f(A, .) is L_f(Aᵀ, .).
            let at = a[[j, i]];
            block[[i, j]] = at;
            block[[n + i, n + j]] = at;
            block[[i, n + j]] = gy[[i, j]] / scale;
        }
    }

    let applied = apply_matrix_fn(&block.view(), kind)?;
    let mut out = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = applied[[i, n + j]] * scale;
        }
    }
    Ok(out)
}

/// Vector-Jacobian product of `f(A) = V diag(f(λ)) Vᵀ` for a **symmetric** `A` and an
/// arbitrary scalar function `f` supplied as a plain `fn(F) -> F`.
///
/// Uses the Daleckii-Krein formula: with `W = Vᵀ gy V` and the divided-difference matrix
/// `D_ij = (f(λ_i) - f(λ_j)) / (λ_i - λ_j)` (and `D_ii = f'(λ_i)` on the diagonal and
/// wherever the eigenvalues coincide), the gradient is `V (D ∘ W) Vᵀ`.
///
/// `f'` is not available through a bare `fn(F) -> F`, so the diagonal entries use a
/// central difference with a step scaled to the eigenvalue. That is a genuine
/// approximation and is documented as such on the public `funm` entry point; the
/// off-diagonal entries — which dominate for a non-degenerate spectrum — are exact.
pub(crate) fn scalar_fn_symmetric_vjp<F: Float>(
    a: &ArrayView2<F>,
    gy: &ArrayView2<F>,
    f: fn(F) -> F,
) -> Result<Array2<F>, OpError> {
    let n = a.nrows();
    if !is_symmetric(a) {
        return Err(OpError::Other(
            "funm: the gradient of a scalar-valued matrix function is only implemented for \
             symmetric inputs (a non-symmetric argument needs a Schur-Parlett evaluation, \
             which this crate does not provide)"
                .into(),
        ));
    }
    if gy.nrows() != n || gy.ncols() != n {
        return Err(OpError::IncompatibleShape(
            "funm: output cotangent shape does not match the input".into(),
        ));
    }

    let (values, vectors) = symmetric_eigen(a)?;
    let two = constant::<F>(2.0)?;
    let fd_step = constant::<F>(1e-5)?;

    // W = Vᵀ gy V
    let w = vectors.t().dot(gy).dot(&vectors);

    let mut d = Array2::<F>::zeros((n, n));
    let degeneracy_tol = F::epsilon().sqrt() * (F::one() + max_abs(a));
    for i in 0..n {
        for j in 0..n {
            let li = values[i];
            let lj = values[j];
            d[[i, j]] = if (li - lj).abs() > degeneracy_tol {
                (f(li) - f(lj)) / (li - lj)
            } else {
                let mid = (li + lj) / two;
                let h = fd_step * (F::one() + mid.abs());
                (f(mid + h) - f(mid - h)) / (two * h)
            };
        }
    }

    let mut dw = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            dw[[i, j]] = d[[i, j]] * w[[i, j]];
        }
    }
    Ok(vectors.dot(&dw).dot(&vectors.t()))
}

// ---------------------------------------------------------------------------
// Graph ops
// ---------------------------------------------------------------------------

fn as_square<'a, F: Float>(
    view: &'a crate::ndarray_ext::NdArrayView<'a, F>,
    what: &str,
) -> Result<ArrayView2<'a, F>, OpError> {
    let v = view
        .view()
        .into_dimensionality::<Ix2>()
        .map_err(|_| OpError::IncompatibleShape(format!("{what}: expected a 2-D array")))?;
    if v.nrows() != v.ncols() {
        return Err(OpError::IncompatibleShape(format!(
            "{what}: expected a square matrix, got {:?}",
            v.shape()
        )));
    }
    Ok(v)
}

/// Lazy backward node for the matrix functions handled by [`MatrixFnKind`].
///
/// Inputs are `(A, gy)`; the output is the Fréchet-derivative adjoint applied to `gy`.
/// Building a node (instead of evaluating during `grad`) keeps the tape intact and works
/// even when the forward graph still has unfed placeholders at the time `grad` is called.
pub(crate) struct MatrixFnVjpOp {
    pub(crate) kind: MatrixFnKind,
}

impl<F: Float> Op<F> for MatrixFnVjpOp {
    fn name(&self) -> &'static str {
        "MatrixFnVjp"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a_in = ctx.input(0);
        let gy_in = ctx.input(1);
        let a = as_square(&a_in, self.kind.label())?;
        let gy = as_square(&gy_in, self.kind.label())?;
        let out = frechet_vjp(&a, &gy, self.kind)?;
        ctx.append_output(out.into_dyn());
        Ok(())
    }

    fn grad<'a, 'graph>(&self, ctx: &mut GradientContext<'a, 'graph, F>) {
        append_second_order_unsupported(ctx, self.kind.label());
    }
}

/// Lazy backward node for `funm` (a scalar function applied through the spectrum).
pub(crate) struct ScalarFnSymmetricVjpOp<F: Float> {
    pub(crate) function: fn(F) -> F,
}

impl<F: Float> Op<F> for ScalarFnSymmetricVjpOp<F> {
    fn name(&self) -> &'static str {
        "ScalarMatrixFnVjp"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a_in = ctx.input(0);
        let gy_in = ctx.input(1);
        let a = as_square(&a_in, "funm")?;
        let gy = as_square(&gy_in, "funm")?;
        let out = scalar_fn_symmetric_vjp(&a, &gy, self.function)?;
        ctx.append_output(out.into_dyn());
        Ok(())
    }

    fn grad<'a, 'graph>(&self, ctx: &mut GradientContext<'a, 'graph, F>) {
        append_second_order_unsupported(ctx, "funm");
    }
}

/// A node that always fails to evaluate, carrying an explanatory message.
///
/// This is how an unimplemented gradient is reported. Returning `None` from `Op::grad`
/// would make the backward pass substitute an explicit **zero**, which is a wrong answer
/// dressed up as a real one; a node that refuses to evaluate is loud and cannot be
/// mistaken for a computed gradient.
pub(crate) struct UnsupportedGradOp {
    pub(crate) message: String,
}

impl<F: Float> Op<F> for UnsupportedGradOp {
    fn name(&self) -> &'static str {
        "UnsupportedGrad"
    }

    fn compute(&self, _ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        Err(OpError::Other(self.message.clone()))
    }

    fn grad<'a, 'graph>(&self, ctx: &mut GradientContext<'a, 'graph, F>) {
        let n = ctx.num_inputs();
        for i in 0..n {
            ctx.append_input_grad(i, None);
        }
    }
}

/// Builds an [`UnsupportedGradOp`] node fed by `anchor` (so the node has a well-defined
/// place in the graph) and returns it as the gradient of every input of `ctx`.
pub(crate) fn append_unsupported_grad<'a, 'graph, F: Float>(
    ctx: &mut GradientContext<'a, 'graph, F>,
    message: String,
) {
    let anchor = *ctx.input(0);
    let g = ctx.graph();
    let node = crate::tensor::Tensor::builder(g)
        .append_input(anchor, false)
        .build(UnsupportedGradOp { message });
    let n = ctx.num_inputs();
    for i in 0..n {
        ctx.append_input_grad(i, Some(node));
    }
}

fn append_second_order_unsupported<'a, 'graph, F: Float>(
    ctx: &mut GradientContext<'a, 'graph, F>,
    label: &str,
) {
    append_unsupported_grad(
        ctx,
        format!(
            "{label}: second-order differentiation is not implemented. The first-order \
             gradient is computed from a Fréchet derivative evaluated numerically, which \
             has no reverse-mode rule of its own."
        ),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    fn close(a: &Array2<f64>, b: &Array2<f64>, tol: f64) {
        assert_eq!(a.shape(), b.shape());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x - y).abs() <= tol * (1.0 + y.abs()),
                "expected {y}, got {x}"
            );
        }
    }

    /// A deliberately non-symmetric, non-diagonal matrix: a diagonal-only "matrix
    /// exponential" cannot reproduce these values.
    fn sample() -> Array2<f64> {
        arr2(&[[0.3, 0.7, -0.2], [0.1, -0.4, 0.5], [-0.3, 0.2, 0.6]])
    }

    #[test]
    fn expm_matches_series_on_a_non_diagonal_matrix() {
        let a = sample();
        let got = expm(&a.view()).expect("expm failed");
        // Independent reference: plain 40-term Taylor series (converges for this norm).
        let mut reference = Array2::<f64>::eye(3);
        let mut term = Array2::<f64>::eye(3);
        for k in 1..=40 {
            term = term.dot(&a) / f64::from(k);
            reference += &term;
        }
        close(&got, &reference, 1e-12);
        // A diagonal approximation would give exp(0.3), exp(-0.4), exp(0.6) on the
        // diagonal and zeros elsewhere; assert we are nowhere near that.
        assert!(got[[0, 1]].abs() > 1e-3);
    }

    #[test]
    fn logm_inverts_expm() {
        let a = sample();
        let e = expm(&a.view()).expect("expm failed");
        let back = logm(&e.view()).expect("logm failed");
        close(&back, &a, 1e-9);
    }

    #[test]
    fn powm_half_squares_back() {
        let a = arr2(&[[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]]);
        let root = powm(&a.view(), 0.5).expect("powm failed");
        close(&root.dot(&root), &a, 1e-9);
    }

    #[test]
    fn sin_cos_satisfy_the_pythagorean_identity() {
        let a = sample();
        let (s, c) = sin_cos_m(&a.view()).expect("sin_cos failed");
        let identity = s.dot(&s) + c.dot(&c);
        close(&identity, &Array2::<f64>::eye(3), 1e-12);
    }

    #[test]
    fn signm_squares_to_identity() {
        let a = sample();
        let s = signm(&a.view()).expect("signm failed");
        close(&s.dot(&s), &Array2::<f64>::eye(3), 1e-9);
    }

    #[test]
    fn symmetric_eigen_reconstructs_the_matrix() {
        let a = arr2(&[[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]]);
        let (vals, vecs) = symmetric_eigen(&a.view()).expect("eigen failed");
        assert!(
            vals[0] > vals[1] && vals[1] > vals[2],
            "not sorted: {vals:?}"
        );
        let mut diag = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            diag[[i, i]] = vals[i];
        }
        close(&vecs.dot(&diag).dot(&vecs.t()), &a, 1e-10);
        close(&vecs.t().dot(&vecs), &Array2::<f64>::eye(3), 1e-10);
    }

    /// The Fréchet VJP must reproduce a directional derivative computed by central
    /// differences with a non-uniform cotangent.
    #[test]
    fn frechet_vjp_matches_finite_differences() {
        let a = sample();
        let gy = arr2(&[
            [0.31, -0.72, 0.15],
            [0.44, 0.09, -0.53],
            [-0.26, 0.68, 0.37],
        ]);

        for kind in [
            MatrixFnKind::Exp,
            MatrixFnKind::Sin,
            MatrixFnKind::Cos,
            MatrixFnKind::Sinh,
            MatrixFnKind::Cosh,
        ] {
            let grad = frechet_vjp(&a.view(), &gy.view(), kind).expect("vjp failed");
            for i in 0..3 {
                for j in 0..3 {
                    let h = 1e-6;
                    let mut plus = a.clone();
                    plus[[i, j]] += h;
                    let mut minus = a.clone();
                    minus[[i, j]] -= h;
                    let fp = apply_matrix_fn(&plus.view(), kind).expect("f+ failed");
                    let fm = apply_matrix_fn(&minus.view(), kind).expect("f- failed");
                    let numeric = (&fp - &fm)
                        .iter()
                        .zip(gy.iter())
                        .map(|(d, g)| d * g)
                        .sum::<f64>()
                        / (2.0 * h);
                    assert!(
                        (grad[[i, j]] - numeric).abs() <= 1e-5 * (1.0 + numeric.abs()),
                        "{kind:?} d/dA[{i},{j}]: analytic {} vs numeric {numeric}",
                        grad[[i, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn frechet_vjp_of_log_matches_finite_differences() {
        // Positive definite so the principal logarithm exists.
        let a = arr2(&[[3.0, 0.4, 0.2], [0.4, 2.0, -0.3], [0.2, -0.3, 1.5]]);
        let gy = arr2(&[
            [0.31, -0.72, 0.15],
            [0.44, 0.09, -0.53],
            [-0.26, 0.68, 0.37],
        ]);
        let grad = frechet_vjp(&a.view(), &gy.view(), MatrixFnKind::Log).expect("vjp failed");
        for i in 0..3 {
            for j in 0..3 {
                let h = 1e-6;
                let mut plus = a.clone();
                plus[[i, j]] += h;
                let mut minus = a.clone();
                minus[[i, j]] -= h;
                let fp = logm(&plus.view()).expect("logm+ failed");
                let fm = logm(&minus.view()).expect("logm- failed");
                let numeric = (&fp - &fm)
                    .iter()
                    .zip(gy.iter())
                    .map(|(d, g)| d * g)
                    .sum::<f64>()
                    / (2.0 * h);
                assert!(
                    (grad[[i, j]] - numeric).abs() <= 1e-5 * (1.0 + numeric.abs()),
                    "log d/dA[{i},{j}]: analytic {} vs numeric {numeric}",
                    grad[[i, j]]
                );
            }
        }
    }

    #[test]
    fn scalar_fn_symmetric_vjp_matches_finite_differences() {
        fn cube(x: f64) -> f64 {
            x * x * x
        }
        let b = arr2(&[[1.3, 0.4, -0.2], [0.4, 0.9, 0.35], [-0.2, 0.35, 1.7]]);
        let gy = arr2(&[
            [0.31, -0.72, 0.15],
            [0.44, 0.09, -0.53],
            [-0.26, 0.68, 0.37],
        ]);
        let grad = scalar_fn_symmetric_vjp(&b.view(), &gy.view(), cube).expect("vjp failed");

        let eval = |m: &Array2<f64>| -> Array2<f64> {
            let (vals, vecs) = symmetric_eigen(&m.view()).expect("eigen failed");
            let mut d = Array2::<f64>::zeros((3, 3));
            for i in 0..3 {
                d[[i, i]] = cube(vals[i]);
            }
            vecs.dot(&d).dot(&vecs.t())
        };

        // Perturb symmetrically so the finite difference stays inside the symmetric
        // manifold the formula is derived on.
        for i in 0..3 {
            for j in 0..3 {
                let h = 1e-6;
                let mut plus = b.clone();
                plus[[i, j]] += h;
                plus[[j, i]] = plus[[i, j]];
                let mut minus = b.clone();
                minus[[i, j]] -= h;
                minus[[j, i]] = minus[[i, j]];
                let numeric = (&eval(&plus) - &eval(&minus))
                    .iter()
                    .zip(gy.iter())
                    .map(|(d, g)| d * g)
                    .sum::<f64>()
                    / (2.0 * h);
                // Symmetric perturbation of an off-diagonal pair moves two entries, so
                // compare against the matching symmetrised gradient entry.
                let analytic = if i == j {
                    grad[[i, j]]
                } else {
                    grad[[i, j]] + grad[[j, i]]
                };
                assert!(
                    (analytic - numeric).abs() <= 1e-5 * (1.0 + numeric.abs()),
                    "cube d/dB[{i},{j}]: analytic {analytic} vs numeric {numeric}"
                );
            }
        }
    }
}
