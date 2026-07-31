//! Backward compatibility layer for ndarray-linalg-style trait-based API
//!
//! This module provides trait-based extensions to `ArrayBase` types that mirror
//! the old `ndarray-linalg` API, making it easier to migrate existing code.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::ndarray::array;
//! use scirs2_linalg::compat::ArrayLinalgExt;
//!
//! let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
//!
//! // Old ndarray-linalg style (now works via compat layer)
//! let (u, s, vt) = a.svd(true).expect("valid input");
//! let inv_a = a.inv().expect("valid input");
//! ```

// ✅ SciRS2 POLICY: Use scirs2_core for all external dependencies
use crate::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, Zero};
use scirs2_core::Complex;
use std::iter::Sum;

/// UPLO parameter for symmetric/Hermitian matrices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UPLO {
    /// Upper triangular
    Upper,
    /// Lower triangular
    Lower,
}

/// Trait providing ndarray-linalg-compatible linear algebra operations
pub trait ArrayLinalgExt<A, S: scirs2_core::ndarray::RawData> {
    /// Singular Value Decomposition: A = U Σ Vᵀ
    fn svd(&self, compute_uv: bool) -> LinalgResult<(Array2<A>, Array1<A>, Array2<A>)>;

    /// Eigenvalues and eigenvectors of a general matrix
    #[allow(clippy::type_complexity)]
    fn eig(
        &self,
    ) -> LinalgResult<(
        Array1<scirs2_core::Complex<A>>,
        Array2<scirs2_core::Complex<A>>,
    )>;

    /// Eigenvalues and eigenvectors of a symmetric/Hermitian matrix
    fn eigh(&self, uplo: UPLO) -> LinalgResult<(Array1<A>, Array2<A>)>;

    /// Eigenvalues only of a symmetric/Hermitian matrix
    fn eigvalsh(&self, uplo: UPLO) -> LinalgResult<Array1<A>>;

    /// Matrix inverse
    fn inv(&self) -> LinalgResult<Array2<A>>;

    /// Solve linear system Ax = b for vector b
    fn solve(&self, b: &ArrayBase<S, Ix1>) -> LinalgResult<Array1<A>>;

    /// Solve linear system Ax = B for matrix B
    fn solve_into(&self, b: &ArrayBase<S, Ix2>) -> LinalgResult<Array2<A>>;

    /// L2 norm (Euclidean norm)
    fn norm_l2(&self) -> A;

    /// Frobenius norm
    fn norm_fro(&self) -> A;

    /// Determinant
    fn det(&self) -> LinalgResult<A>;

    /// QR decomposition: A = QR
    fn qr(&self) -> LinalgResult<(Array2<A>, Array2<A>)>;

    /// LU decomposition: PA = LU
    fn lu(&self) -> LinalgResult<(Array2<A>, Array2<A>, Array2<A>)>;

    /// Cholesky decomposition: A = LLᵀ
    fn cholesky(&self) -> LinalgResult<Array2<A>>;
}

impl<A, S> ArrayLinalgExt<A, S> for ArrayBase<S, Ix2>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    fn svd(&self, compute_uv: bool) -> LinalgResult<(Array2<A>, Array1<A>, Array2<A>)> {
        crate::svd(&self.view(), compute_uv, None)
    }

    fn eig(
        &self,
    ) -> LinalgResult<(
        Array1<scirs2_core::Complex<A>>,
        Array2<scirs2_core::Complex<A>>,
    )> {
        crate::eig(&self.view(), None)
    }

    fn eigh(&self, _uplo: UPLO) -> LinalgResult<(Array1<A>, Array2<A>)> {
        // scirs2-linalg's eigh doesn't use UPLO parameter currently
        crate::eigh(&self.view(), None)
    }

    fn eigvalsh(&self, _uplo: UPLO) -> LinalgResult<Array1<A>> {
        crate::eigvalsh(&self.view(), None)
    }

    fn inv(&self) -> LinalgResult<Array2<A>> {
        crate::inv(&self.view(), None)
    }

    fn solve(&self, b: &ArrayBase<S, Ix1>) -> LinalgResult<Array1<A>> {
        crate::solve(&self.view(), &b.view(), None)
    }

    fn solve_into(&self, b: &ArrayBase<S, Ix2>) -> LinalgResult<Array2<A>> {
        crate::solve_multiple(&self.view(), &b.view(), None)
    }

    fn norm_l2(&self) -> A {
        // Calculate Frobenius norm for matrices (equivalent to L2 for flattened matrix)
        self.iter().map(|&x| x * x).sum::<A>().sqrt()
    }

    fn norm_fro(&self) -> A {
        self.iter().map(|&x| x * x).sum::<A>().sqrt()
    }

    fn det(&self) -> LinalgResult<A> {
        crate::det(&self.view(), None)
    }

    fn qr(&self) -> LinalgResult<(Array2<A>, Array2<A>)> {
        crate::qr(&self.view(), None)
    }

    fn lu(&self) -> LinalgResult<(Array2<A>, Array2<A>, Array2<A>)> {
        crate::lu(&self.view(), None)
    }

    fn cholesky(&self) -> LinalgResult<Array2<A>> {
        crate::cholesky(&self.view(), None)
    }
}

/// Trait for solving linear systems (compatibility)
pub trait Solve<A> {
    /// Output type for the solution
    type Output;

    /// Solve the linear system
    fn solve(&self, rhs: &Self) -> LinalgResult<Self::Output>;
}

/// Trait for computing singular value decomposition
pub trait SVD {
    /// Singular values type
    type S;
    /// Left singular vectors type
    type U;
    /// Right singular vectors type (transposed)
    type Vt;

    /// Compute SVD
    fn svd(&self, compute_uv: bool) -> LinalgResult<(Self::U, Self::S, Self::Vt)>;
}

impl<A, S> SVD for ArrayBase<S, Ix2>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    type S = Array1<A>;
    type U = Array2<A>;
    type Vt = Array2<A>;

    fn svd(&self, compute_uv: bool) -> LinalgResult<(Self::U, Self::S, Self::Vt)> {
        ArrayLinalgExt::svd(self, compute_uv)
    }
}

/// Trait for computing eigenvalues and eigenvectors
pub trait Eig {
    /// Eigenvalue type
    type EigVal;
    /// Eigenvector type
    type EigVec;

    /// Compute eigenvalues and eigenvectors
    fn eig(&self) -> LinalgResult<(Self::EigVal, Self::EigVec)>;
}

impl<A, S> Eig for ArrayBase<S, Ix2>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    type EigVal = Array1<scirs2_core::Complex<A>>;
    type EigVec = Array2<scirs2_core::Complex<A>>;

    fn eig(&self) -> LinalgResult<(Self::EigVal, Self::EigVec)> {
        ArrayLinalgExt::eig(self)
    }
}

/// Trait for computing eigenvalues and eigenvectors of Hermitian matrices
pub trait Eigh {
    /// Eigenvalue type
    type EigVal;
    /// Eigenvector type
    type EigVec;

    /// Compute eigenvalues and eigenvectors
    fn eigh(&self, uplo: UPLO) -> LinalgResult<(Self::EigVal, Self::EigVec)>;
}

impl<A, S> Eigh for ArrayBase<S, Ix2>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    type EigVal = Array1<A>;
    type EigVec = Array2<A>;

    fn eigh(&self, uplo: UPLO) -> LinalgResult<(Self::EigVal, Self::EigVec)> {
        ArrayLinalgExt::eigh(self, uplo)
    }
}

/// Trait for computing matrix inverse
pub trait Inverse {
    /// Output type
    type Output;

    /// Compute matrix inverse
    fn inv(&self) -> LinalgResult<Self::Output>;
}

impl<A, S> Inverse for ArrayBase<S, Ix2>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn inv(&self) -> LinalgResult<Self::Output> {
        ArrayLinalgExt::inv(self)
    }
}

/// Trait for computing matrix norms
pub trait Norm<A> {
    /// Compute matrix norm
    fn norm(&self) -> A;

    /// Compute L2 norm
    fn norm_l2(&self) -> A;

    /// Compute Frobenius norm
    fn norm_fro(&self) -> A;
}

impl<A, S> Norm<A> for ArrayBase<S, Ix2>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    fn norm(&self) -> A {
        ArrayLinalgExt::norm_fro(self)
    }

    fn norm_l2(&self) -> A {
        ArrayLinalgExt::norm_l2(self)
    }

    fn norm_fro(&self) -> A {
        ArrayLinalgExt::norm_fro(self)
    }
}

impl<A, S> Norm<A> for ArrayBase<S, Ix1>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    fn norm(&self) -> A {
        self.norm_l2()
    }

    fn norm_l2(&self) -> A {
        self.iter().map(|&x| x * x).sum::<A>().sqrt()
    }

    fn norm_fro(&self) -> A {
        self.norm_l2()
    }
}

// ============================================================================
// Standalone wrapper functions for scipy.linalg compatibility
// ============================================================================
// These functions provide a scipy.linalg-style API by wrapping the internal
// implementation functions. They are designed to be used via scipy_compat module.

/// Type alias for SVD result (U, S, Vt)
pub type SvdResult<A> = (Array2<A>, Array1<A>, Array2<A>);

/// Singular Value Decomposition: A = U Σ Vᵀ
pub fn svd<A, S>(a: &ArrayBase<S, Ix2>, compute_uv: bool) -> LinalgResult<SvdResult<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    crate::svd(&a.view(), compute_uv, None)
}

/// Compute eigenvalues and eigenvectors of a general matrix
#[allow(clippy::type_complexity)]
pub fn eig<A, S>(
    a: &ArrayBase<S, Ix2>,
) -> LinalgResult<(
    Array1<scirs2_core::Complex<A>>,
    Array2<scirs2_core::Complex<A>>,
)>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    crate::eig(&a.view(), None)
}

/// Compute eigenvalues and eigenvectors of a symmetric/Hermitian matrix
pub fn eigh<A, S>(a: &ArrayBase<S, Ix2>, uplo: UPLO) -> LinalgResult<(Array1<A>, Array2<A>)>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    let _ = uplo; // Currently unused
    crate::eigh(&a.view(), None)
}

/// Compute eigenvalues only of a symmetric/Hermitian matrix
pub fn eigvalsh<A, S>(a: &ArrayBase<S, Ix2>, uplo: UPLO) -> LinalgResult<Array1<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    let _ = uplo; // Currently unused
    crate::eigvalsh(&a.view(), None)
}

/// Compute eigenvalues only of a general matrix
pub fn eigvals<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<Array1<scirs2_core::Complex<A>>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    let (vals, _) = crate::eig(&a.view(), None)?;
    Ok(vals)
}

/// Compute eigenvalues of a banded symmetric matrix
pub fn eigvals_banded<A, S>(a: &ArrayBase<S, Ix2>, uplo: UPLO) -> LinalgResult<Array1<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    eigvalsh(a, uplo)
}

/// Compute eigenvalues and eigenvectors of a banded symmetric matrix
pub fn eig_banded<A, S>(a: &ArrayBase<S, Ix2>, uplo: UPLO) -> LinalgResult<(Array1<A>, Array2<A>)>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    eigh(a, uplo)
}

/// Compute eigenvalues and eigenvectors of a tridiagonal symmetric matrix
pub fn eigh_tridiagonal<A>(d: &Array1<A>, e: &Array1<A>) -> LinalgResult<(Array1<A>, Array2<A>)>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    // Construct tridiagonal matrix and use eigh
    let n = d.len();
    let mut mat = Array2::zeros((n, n));
    for i in 0..n {
        mat[[i, i]] = d[i];
        if i < n - 1 {
            mat[[i, i + 1]] = e[i];
            mat[[i + 1, i]] = e[i];
        }
    }
    eigh(&mat, UPLO::Lower)
}

/// Compute eigenvalues only of a tridiagonal symmetric matrix
pub fn eigvalsh_tridiagonal<A>(d: &Array1<A>, e: &Array1<A>) -> LinalgResult<Array1<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (vals, _) = eigh_tridiagonal(d, e)?;
    Ok(vals)
}

/// Matrix inverse
pub fn inv<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    crate::inv(&a.view(), None)
}

/// Determinant of a square matrix
pub fn det<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<A>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    crate::det(&a.view(), None)
}

/// QR decomposition: A = QR
pub fn qr<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<(Array2<A>, Array2<A>)>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    crate::qr(&a.view(), None)
}

/// RQ decomposition: A = RQ
pub fn rq<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<(Array2<A>, Array2<A>)>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    // RQ = reverse of QR on transposed matrix
    let t = a.t();
    let (q, r) = crate::qr(&t.view(), None)?;
    Ok((r.reversed_axes(), q.reversed_axes()))
}

/// LU decomposition: PA = LU
pub fn lu<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<(Array2<A>, Array2<A>, Array2<A>)>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    crate::lu(&a.view(), None)
}

/// Cholesky decomposition: A = LLᵀ
pub fn cholesky<A, S>(a: &ArrayBase<S, Ix2>, uplo: UPLO) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    let _ = uplo; // Currently unused
    crate::cholesky(&a.view(), None)
}

/// Solve linear system Ax = b
pub fn compat_solve<A, S1, S2>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix1>,
) -> LinalgResult<Array1<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    crate::solve(&a.view(), &b.view(), None)
}

/// Solve banded linear system
pub fn solve_banded<A, S1, S2>(
    l_and_u: (usize, usize),
    ab: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix1>,
) -> LinalgResult<Array1<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let (l, u) = l_and_u;
    crate::structured_solvers::solve_banded(l, u, &ab.view(), &b.view())
}

/// Solve triangular system
///
/// Solves Ax = b where A is either lower-triangular (lower=true) or
/// upper-triangular (lower=false) using forward/backward substitution.
pub fn solve_triangular<A, S1, S2>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix1>,
    lower: bool,
) -> LinalgResult<Array1<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "solve_triangular: coefficient matrix must be square".to_string(),
        ));
    }
    if b.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "solve_triangular: b length {} does not match matrix size {}",
            b.len(),
            n
        )));
    }

    let mut x = Array1::<A>::zeros(n);

    if lower {
        // Forward substitution: L x = b
        for i in 0..n {
            let mut val = b[i];
            for k in 0..i {
                val -= a[[i, k]] * x[k];
            }
            let diag = a[[i, i]];
            if diag.abs() < A::epsilon() {
                return Err(LinalgError::SingularMatrixError(format!(
                    "solve_triangular: zero pivot at ({i},{i})"
                )));
            }
            x[i] = val / diag;
        }
    } else {
        // Backward substitution: U x = b
        for i in (0..n).rev() {
            let mut val = b[i];
            for k in (i + 1)..n {
                val -= a[[i, k]] * x[k];
            }
            let diag = a[[i, i]];
            if diag.abs() < A::epsilon() {
                return Err(LinalgError::SingularMatrixError(format!(
                    "solve_triangular: zero pivot at ({i},{i})"
                )));
            }
            x[i] = val / diag;
        }
    }

    Ok(x)
}

/// Least squares solution
pub fn lstsq<A, S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix1>) -> LinalgResult<Array1<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let result = crate::lstsq(&a.view(), &b.view(), None)?;
    Ok(result.x)
}

/// Pseudo-inverse (Moore-Penrose inverse)
pub fn pinv<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    // Use SVD-based pseudo-inverse: A† = V Σ† Uᵀ
    let (u, s, vt) = crate::svd(&a.view(), true, None)?;
    let threshold = A::from(1e-15)
        .ok_or_else(|| LinalgError::ComputationError("Failed to convert threshold".to_string()))?
        * s[[0]];
    let s_inv: Array1<A> = s.map(|&val| {
        if val > threshold {
            A::one() / val
        } else {
            A::zero()
        }
    });
    Ok(vt.t().dot(&Array2::from_diag(&s_inv)).dot(&u.t()))
}

/// Matrix rank
pub fn matrix_rank<A, S>(a: &ArrayBase<S, Ix2>, tol: Option<A>) -> LinalgResult<usize>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    let (_, s, _) = crate::svd(&a.view(), false, None)?;
    let threshold = tol.unwrap_or_else(|| {
        let max_singular = s.iter().fold(A::zero(), |a, &b| if b > a { b } else { a });
        let dim_factor = A::from(a.nrows().max(a.ncols())).unwrap_or_else(|| A::one());
        max_singular * dim_factor * A::epsilon()
    });
    Ok(s.iter().filter(|&&val| val > threshold).count())
}

/// Condition number
pub fn cond<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<A>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    let (_, s, _) = crate::svd(&a.view(), false, None)?;
    let s_max = s.iter().fold(A::zero(), |a, &b| if b > a { b } else { a });
    let s_min = s
        .iter()
        .fold(s_max, |a, &b| if b < a && b > A::zero() { b } else { a });
    if s_min == A::zero() {
        return Ok(A::infinity());
    }
    Ok(s_max / s_min)
}

/// Matrix norm
pub fn norm<A, S>(a: &ArrayBase<S, Ix2>, ord: Option<&str>) -> LinalgResult<A>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    match ord {
        None | Some("fro") => Ok(ArrayLinalgExt::norm_fro(a)),
        Some("2") => {
            let (_, s, _) = crate::svd(&a.view(), false, None)?;
            Ok(s[[0]])
        }
        _ => Err(LinalgError::ComputationError(format!(
            "norm ord={:?} not implemented",
            ord
        ))),
    }
}

/// Vector norm
///
/// `ord` follows the SciPy `scipy.linalg.norm` convention for vectors:
/// `None` or `Some(2)` is the Euclidean (L2) norm, `Some(1)` is the
/// Manhattan (L1) norm, `Some(0)` counts the non-zero entries, and any other
/// (possibly negative) integer `p` computes the general `(sum(|x_i|^p))^(1/p)`
/// formula. Unlike the sibling matrix [`norm`] function (which returns an
/// error for an unsupported `ord`), every integer `ord` has a well-defined
/// value here, so this function does not need to fail.
pub fn vector_norm<A, S>(a: &ArrayBase<S, Ix1>, ord: Option<i32>) -> A
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    match ord {
        None | Some(2) => a.iter().map(|&x| x * x).sum::<A>().sqrt(),
        Some(1) => a.iter().map(|&x| x.abs()).sum::<A>(),
        Some(0) => {
            // SciPy convention: the "0-norm" counts non-zero entries.
            let count = a.iter().filter(|&&x| x != A::zero()).count();
            A::from(count).unwrap_or(A::zero())
        }
        Some(p) => {
            // General L^p (quasi-)norm: (sum |x_i|^p)^(1/p), valid for any
            // nonzero integer p (SciPy also accepts negative orders, e.g.
            // ord=-1, ord=-2, for vectors).
            let p_float = A::from(p).unwrap_or(A::one());
            let sum = a.iter().fold(A::zero(), |acc, &x| acc + x.abs().powi(p));
            sum.powf(A::one() / p_float)
        }
    }
}

/// Schur decomposition
pub fn schur<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<(Array2<A>, Array2<A>)>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    crate::schur(&a.view())
}

/// Polar decomposition: A = UP where U is unitary and P is positive semidefinite
pub fn polar<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<(Array2<A>, Array2<A>)>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    let (u, s, vt) = crate::svd(&a.view(), true, None)?;
    let unitary = u.dot(&vt);
    let hermitian = vt.t().dot(&Array2::from_diag(&s)).dot(&vt);
    Ok((unitary, hermitian))
}

/// Matrix exponential: exp(A)
pub fn expm<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    crate::expm(&a.view(), None)
}

/// Matrix logarithm: log(A)
pub fn logm<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    crate::logm(&a.view())
}

/// Matrix square root: sqrt(A)
pub fn sqrtm<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    let tol = A::from(1e-8)
        .ok_or_else(|| LinalgError::ComputationError("Failed to convert tolerance".to_string()))?;
    crate::sqrtm(&a.view(), 100, tol)
}

/// Matrix sine: sin(A)
pub fn sinm<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    crate::sinm(&a.view())
}

/// Matrix cosine: cos(A)
pub fn cosm<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    crate::cosm(&a.view())
}

/// Matrix tangent: tan(A)
pub fn tanm<A, S>(a: &ArrayBase<S, Ix2>) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    crate::tanm(&a.view())
}

/// General matrix function: f(A) using eigendecomposition
pub fn funm<A, S, F>(a: &ArrayBase<S, Ix2>, func: F) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
    F: Fn(A) -> A,
{
    // For symmetric matrices, use eigendecomposition
    let (vals, vecs) = crate::eigh(&a.view(), None)?;
    let f_vals: Array1<A> = vals.map(|&v| func(v));
    Ok(vecs.dot(&Array2::from_diag(&f_vals)).dot(&vecs.t()))
}

/// Fractional matrix power: A^p using eigendecomposition
pub fn fractionalmatrix_power<A, S>(a: &ArrayBase<S, Ix2>, p: A) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
    S: Data<Elem = A>,
{
    funm(a, |x| x.powf(p))
}

/// Block diagonal matrix construction
pub fn block_diag<A>(blocks: &[Array2<A>]) -> Array2<A>
where
    A: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static + Zero,
{
    if blocks.is_empty() {
        return Array2::zeros((0, 0));
    }

    let total_rows: usize = blocks.iter().map(|b| b.nrows()).sum();
    let total_cols: usize = blocks.iter().map(|b| b.ncols()).sum();
    let mut result = Array2::zeros((total_rows, total_cols));

    let mut row_offset = 0;
    let mut col_offset = 0;

    for block in blocks {
        let nrows = block.nrows();
        let ncols = block.ncols();
        result
            .slice_mut(scirs2_core::ndarray::s![
                row_offset..row_offset + nrows,
                col_offset..col_offset + ncols
            ])
            .assign(block);
        row_offset += nrows;
        col_offset += ncols;
    }

    result
}

#[cfg(test)]
mod tests_compat {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_solve_triangular_lower_2x2() {
        // L = [[2, 0], [3, 4]], b = [4, 17]  =>  x = [2, (17-6)/4] = [2, 11/4]
        let l = array![[2.0_f64, 0.0], [3.0, 4.0]];
        let b = array![4.0_f64, 17.0];
        let x = solve_triangular(&l, &b, true).expect("lower triangular solve");
        assert_abs_diff_eq!(x[0], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], 11.0 / 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_solve_triangular_upper_2x2() {
        // U = [[3, 1], [0, 2]], b = [7, 4]  =>  x[1] = 2, x[0] = (7-2)/3 = 5/3
        let u = array![[3.0_f64, 1.0], [0.0, 2.0]];
        let b = array![7.0_f64, 4.0];
        let x = solve_triangular(&u, &b, false).expect("upper triangular solve");
        assert_abs_diff_eq!(x[1], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[0], 5.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_solve_triangular_identity() {
        let eye = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![3.0_f64, 5.0];
        let x_l = solve_triangular(&eye, &b, true).expect("lower");
        let x_u = solve_triangular(&eye, &b, false).expect("upper");
        assert_abs_diff_eq!(x_l[0], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x_l[1], 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x_u[0], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x_u[1], 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_solve_triangular_singular_error() {
        let singular = array![[0.0_f64, 0.0], [1.0, 2.0]];
        let b = array![1.0_f64, 1.0];
        assert!(solve_triangular(&singular, &b, true).is_err());
    }

    #[test]
    fn test_vector_norm_general_lp_not_silently_l2() {
        // Non-constant vector: L1=6, L2=sqrt(14)=3.7416..., L3=36^(1/3)=3.3019...
        // These are all genuinely different, so an `ord` that silently fell
        // back to L2 would fail this check.
        let v = array![1.0_f64, 2.0, 3.0];

        let l1 = vector_norm(&v, Some(1));
        let l2 = vector_norm(&v, Some(2));
        let l3 = vector_norm(&v, Some(3));

        assert_abs_diff_eq!(l1, 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(l2, 14.0_f64.sqrt(), epsilon = 1e-10);
        let expected_l3 = 36.0_f64.powf(1.0 / 3.0);
        assert_abs_diff_eq!(l3, expected_l3, epsilon = 1e-9);
        // The whole point: L3 must NOT equal the (previously hard-coded) L2 fallback.
        assert!((l3 - l2).abs() > 0.1, "L3 collapsed onto L2: {l3} vs {l2}");
    }

    #[test]
    fn test_vector_norm_zero_norm_counts_nonzero() {
        let v = array![0.0_f64, 3.0, 0.0, -5.0, 2.0];
        let l0 = vector_norm(&v, Some(0));
        assert_abs_diff_eq!(l0, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_vector_norm_negative_order_matches_general_formula() {
        // Non-zero, non-constant vector so ord=-1 is well defined:
        // (sum |x_i|^-1)^(1/-1) = (1/2 + 1/4)^(-1) = (0.75)^(-1) = 4/3.
        let v = array![2.0_f64, 4.0];
        let neg1 = vector_norm(&v, Some(-1));
        assert_abs_diff_eq!(neg1, 4.0 / 3.0, epsilon = 1e-10);
    }
}
