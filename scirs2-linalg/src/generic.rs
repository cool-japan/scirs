//! Type-generic linear algebra operations
//!
//! This module provides unified interfaces for linear algebra operations
//! that work with different numeric types (f32, f64, Complex<f32>, Complex<f64>).

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array2, ArrayView2};
use num_traits::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;

/// A trait that unifies numeric types suitable for linear algebra operations
pub trait LinalgScalar:
    Clone
    + Debug
    + Default
    + PartialEq
    + NumAssign
    + Sum
    + for<'a> Sum<&'a Self>
    + ndarray::ScalarOperand
    + 'static
{
    /// Type used for norms and condition numbers (always real)
    type Real: Float + NumAssign + Sum + Debug + Default + 'static;

    /// Convert to f64 for certain calculations
    fn to_f64(&self) -> Result<f64, LinalgError>;

    /// Create from f64
    fn from_f64(v: f64) -> Result<Self, LinalgError>;

    /// Get the absolute value
    fn abs(&self) -> Self::Real;

    /// Check if value is zero
    fn is_zero(&self) -> bool;

    /// Get the zero value
    fn zero() -> Self;

    /// Get the one value  
    fn one() -> Self;

    /// Square root
    fn sqrt(&self) -> Self;

    /// Get conjugate (for complex numbers, identity for real)
    fn conj(&self) -> Self;

    /// Get real part
    fn real(&self) -> Self::Real;

    /// Get epsilon for this type
    fn epsilon() -> Self::Real;
}

// Implement LinalgScalar for f32
impl LinalgScalar for f32 {
    type Real = f32;

    fn to_f64(&self) -> Result<f64, LinalgError> {
        Ok(*self as f64)
    }

    fn from_f64(v: f64) -> Result<Self, LinalgError> {
        if v.is_finite() && v.abs() <= f32::MAX as f64 {
            Ok(v as f32)
        } else {
            Err(LinalgError::ComputationError(
                "Value out of range for f32".to_string(),
            ))
        }
    }

    fn abs(&self) -> Self::Real {
        <f32>::abs(*self)
    }

    fn is_zero(&self) -> bool {
        self.abs() < f32::EPSILON
    }

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn sqrt(&self) -> Self {
        <f32>::sqrt(*self)
    }

    fn conj(&self) -> Self {
        *self
    }

    fn real(&self) -> Self::Real {
        *self
    }

    fn epsilon() -> Self::Real {
        f32::EPSILON
    }
}

// Implement LinalgScalar for f64
impl LinalgScalar for f64 {
    type Real = f64;

    fn to_f64(&self) -> Result<f64, LinalgError> {
        Ok(*self)
    }

    fn from_f64(v: f64) -> Result<Self, LinalgError> {
        if v.is_finite() {
            Ok(v)
        } else {
            Err(LinalgError::ComputationError(
                "Non-finite value".to_string(),
            ))
        }
    }

    fn abs(&self) -> Self::Real {
        <f64>::abs(*self)
    }

    fn is_zero(&self) -> bool {
        self.abs() < f64::EPSILON
    }

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn sqrt(&self) -> Self {
        <f64>::sqrt(*self)
    }

    fn conj(&self) -> Self {
        *self
    }

    fn real(&self) -> Self::Real {
        *self
    }

    fn epsilon() -> Self::Real {
        f64::EPSILON
    }
}

/// Generic matrix multiplication - wrapper using ndarray's dot
pub fn gemm<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> LinalgResult<Array2<T>>
where
    T: LinalgScalar + ndarray::LinalgScalar,
{
    if a.ncols() != b.nrows() {
        return Err(LinalgError::DimensionError(format!(
            "Matrix dimensions don't match for multiplication: ({}, {}) x ({}, {})",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        )));
    }

    Ok(a.dot(b))
}

/// Generic matrix-vector multiplication - wrapper using ndarray's dot
pub fn gemv<T>(a: &ArrayView2<T>, x: &ndarray::ArrayView1<T>) -> LinalgResult<ndarray::Array1<T>>
where
    T: LinalgScalar + ndarray::LinalgScalar,
{
    if a.ncols() != x.len() {
        return Err(LinalgError::DimensionError(format!(
            "Matrix and vector dimensions don't match: ({}, {}) x {}",
            a.nrows(),
            a.ncols(),
            x.len()
        )));
    }

    Ok(a.dot(x))
}

/// Generic determinant calculation (only for real floats)
pub fn gdet<T: LinalgScalar + Float>(a: &ArrayView2<T>) -> LinalgResult<T> {
    crate::basic::det(a, None)
}

/// Generic matrix inversion (only for real floats)
pub fn ginv<T: LinalgScalar + Float>(a: &ArrayView2<T>) -> LinalgResult<Array2<T>> {
    crate::basic::inv(a, None)
}

/// Generic matrix norm (only for real floats)
pub fn gnorm<T: LinalgScalar + Float>(a: &ArrayView2<T>, norm_type: &str) -> LinalgResult<T> {
    crate::norm::matrix_norm(a, norm_type, None)
}

/// Generic SVD decomposition result
pub struct GenericSVD<T: LinalgScalar> {
    pub u: Array2<T>,
    pub s: ndarray::Array1<T>,
    pub vt: Array2<T>,
}

/// Generic SVD decomposition (only for real floats)  
pub fn gsvd<T: LinalgScalar + Float>(
    a: &ArrayView2<T>,
    full_matrices: bool,
) -> LinalgResult<GenericSVD<T>> {
    let result = crate::lapack::svd(a, full_matrices)?;
    Ok(GenericSVD {
        u: result.u,
        s: result.s,
        vt: result.vt,
    })
}

/// Generic QR decomposition result
pub struct GenericQR<T: LinalgScalar> {
    pub q: Array2<T>,
    pub r: Array2<T>,
}

/// Generic QR decomposition (only for real floats)
pub fn gqr<T: LinalgScalar + Float>(a: &ArrayView2<T>) -> LinalgResult<GenericQR<T>> {
    let result = crate::lapack::qr_factor(a)?;
    Ok(GenericQR {
        q: result.q,
        r: result.r,
    })
}

/// Generic eigendecomposition result (complex for real matrices)
pub struct GenericEigen<T: LinalgScalar> {
    pub eigenvalues: ndarray::Array1<num_complex::Complex<T>>,
    pub eigenvectors: Array2<num_complex::Complex<T>>,
}

/// Generic eigendecomposition (only for real floats, returns complex)
pub fn geig<T: LinalgScalar + Float>(a: &ArrayView2<T>) -> LinalgResult<GenericEigen<T>> {
    let (eigenvalues, eigenvectors) = crate::eigen::eig(a, None)?;
    Ok(GenericEigen {
        eigenvalues,
        eigenvectors,
    })
}

/// Generic linear solve (only for real floats)
pub fn gsolve<T: LinalgScalar + Float>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
) -> LinalgResult<Array2<T>> {
    crate::solve::solve_multiple(a, b, None)
}

/// Precision trait for automatic precision selection
pub trait PrecisionSelector {
    type HighPrecision: LinalgScalar;
    type LowPrecision: LinalgScalar;

    fn should_use_high_precision(input_condition: f64) -> bool {
        input_condition > 1e6
    }
}

impl PrecisionSelector for f32 {
    type HighPrecision = f64;
    type LowPrecision = f32;
}

impl PrecisionSelector for f64 {
    type HighPrecision = f64;
    type LowPrecision = f32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gemm() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = gemm(&a.view(), &b.view()).unwrap();
        assert_eq!(c[[0, 0]], 19.0);
        assert_eq!(c[[0, 1]], 22.0);
        assert_eq!(c[[1, 0]], 43.0);
        assert_eq!(c[[1, 1]], 50.0);
    }

    #[test]
    fn test_gemv() {
        let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let x = array![5.0, 6.0];
        let y = gemv(&a.view(), &x.view()).unwrap();
        assert_eq!(y[0], 17.0);
        assert_eq!(y[1], 39.0);
    }

    #[test]
    fn test_gdet() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let det = gdet(&a.view()).unwrap();
        assert!((det - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_ginv() {
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let a_inv = ginv(&a.view()).unwrap();
        assert!((a_inv[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((a_inv[[1, 1]] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gnorm() {
        let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let norm = gnorm(&a.view(), "fro").unwrap();
        let expected = (1.0 + 4.0 + 9.0 + 16.0_f32).sqrt();
        assert!((norm - expected).abs() < 1e-6);
    }

    #[test]
    fn test_gsvd() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let svd = gsvd(&a.view(), false).unwrap();

        // Check that U and V are orthogonal
        let u_t_u = svd.u.t().dot(&svd.u);
        for i in 0..u_t_u.nrows() {
            for j in 0..u_t_u.ncols() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((u_t_u[[i, j]] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_gqr() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let qr = gqr(&a.view()).unwrap();

        // Check that Q is orthogonal
        let q_t_q = qr.q.t().dot(&qr.q);
        for i in 0..q_t_q.nrows() {
            for j in 0..q_t_q.ncols() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((q_t_q[[i, j]] - expected).abs() < 1e-10);
            }
        }

        // Check that A = Q * R
        let reconstructed = qr.q.dot(&qr.r);
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert!((reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_geig() {
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let eigen = geig(&a.view()).unwrap();

        // For diagonal matrix, eigenvalues should be the diagonal elements
        // but they might not be in order
        let mut eigenvalues_real: Vec<f64> = eigen.eigenvalues.iter().map(|e| e.re).collect();
        eigenvalues_real.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let expected_eigenvalues = [1.0, 2.0];
        for (i, &expected) in expected_eigenvalues.iter().enumerate() {
            assert!((eigenvalues_real[i] - expected).abs() < 1e-10);
            assert!(eigen.eigenvalues[0].im.abs() < 1e-10);
            assert!(eigen.eigenvalues[1].im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_gsolve() {
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        let b = array![[1.0], [1.0]];
        let x = gsolve(&a.view(), &b.view()).unwrap();

        // Check that A * x = b
        let ax = a.dot(&x);
        for i in 0..b.nrows() {
            for j in 0..b.ncols() {
                assert!((ax[[i, j]] - b[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_precision_selector() {
        assert!(!f32::should_use_high_precision(100.0));
        assert!(f32::should_use_high_precision(1e7));
    }
}
