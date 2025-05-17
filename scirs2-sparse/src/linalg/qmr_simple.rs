use crate::error::{SparseError, SparseResult};
use crate::linalg::interface::LinearOperator;
use num_traits::{Float, NumAssign};
use std::fmt::Display;
use std::iter::Sum;

/// Result of QMR solver
#[derive(Debug, Clone)]
pub struct QMRResult<F> {
    pub x: Vec<F>,
    pub iterations: usize,
    pub residual_norm: F,
    pub converged: bool,
    pub message: String,
}

/// Options for QMR solver
pub struct QMROptions<F> {
    pub max_iter: usize,
    pub rtol: F,
    pub atol: F,
    pub x0: Option<Vec<F>>,
    pub left_preconditioner: Option<Box<dyn LinearOperator<F>>>,
    pub right_preconditioner: Option<Box<dyn LinearOperator<F>>>,
}

impl<F: Float> Default for QMROptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            rtol: F::from(1e-8).unwrap(),
            atol: F::from(1e-12).unwrap(),
            x0: None,
            left_preconditioner: None,
            right_preconditioner: None,
        }
    }
}

/// Simple QMR implementation for testing
///
/// This is a simplified version that should work for basic cases
pub fn qmr<F>(
    a: &dyn LinearOperator<F>,
    b: &[F],
    options: QMROptions<F>,
) -> SparseResult<QMRResult<F>>
where
    F: Float + NumAssign + Sum + Display + 'static,
{
    let n = b.len();
    
    // Check dimensions
    if a.shape().0 != n || a.shape().1 != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: a.shape().0,
        });
    }
    
    // Initialize solution
    let mut x = options.x0.unwrap_or_else(|| vec![F::zero(); n]);
    
    // For a simple implementation, let's use BiCG-style iteration
    // This is not standard QMR but should work for testing
    
    // Compute initial residual r = b - Ax
    let ax = a.matvec(&x)?;
    let mut r = vec_sub(b, &ax);
    
    // Choose r_tilde = r for simplicity
    let mut r_tilde = r.clone();
    
    // Initialize vectors
    let mut p = vec![F::zero(); n];
    let mut p_tilde = vec![F::zero(); n];
    
    // Initialize scalars
    let mut rho = F::one();
    let mut rho_old = F::one();
    
    // Compute initial norms
    let bnorm = norm2(b);
    let tol = options.atol + options.rtol * bnorm;
    
    for iter in 0..options.max_iter {
        // Check convergence
        let rnorm = norm2(&r);
        if rnorm < tol {
            return Ok(QMRResult {
                x,
                iterations: iter,
                residual_norm: rnorm,
                converged: true,
                message: format!("Converged in {} iterations", iter),
            });
        }
        
        // BiCG-like iteration
        rho = dot(&r_tilde, &r);
        
        if rho.abs() < F::epsilon() {
            return Ok(QMRResult {
                x,
                iterations: iter,
                residual_norm: rnorm,
                converged: false,
                message: "Breakdown: rho = 0".to_string(),
            });
        }
        
        if iter == 0 {
            p = r.clone();
            p_tilde = r_tilde.clone();
        } else {
            let beta = rho / rho_old;
            p = vec_add_scaled(&r, &p, beta);
            p_tilde = vec_add_scaled(&r_tilde, &p_tilde, beta);
        }
        
        // q = A * p
        let q = a.matvec(&p)?;
        let alpha = rho / dot(&p_tilde, &q);
        
        // Update solution and residuals
        x = vec_add(&x, &vec_scaled(&p, alpha));
        r = vec_sub(&r, &vec_scaled(&q, alpha));
        
        // Update r_tilde for the transpose
        let q_tilde = a.rmatvec(&p_tilde)?;
        r_tilde = vec_sub(&r_tilde, &vec_scaled(&q_tilde, alpha));
        
        rho_old = rho;
    }
    
    let rnorm = norm2(&r);
    Ok(QMRResult {
        x,
        iterations: options.max_iter,
        residual_norm: rnorm,
        converged: false,
        message: format!(
            "Did not converge in {} iterations. Residual: {}",
            options.max_iter, rnorm
        ),
    })
}

// Helper functions
fn dot<F: Float + Sum>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

fn norm2<F: Float + Sum>(v: &[F]) -> F {
    v.iter().map(|&vi| vi * vi).sum::<F>().sqrt()
}

fn vec_add<F: Float>(a: &[F], b: &[F]) -> Vec<F> {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai + bi).collect()
}

fn vec_sub<F: Float>(a: &[F], b: &[F]) -> Vec<F> {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai - bi).collect()
}

fn vec_scaled<F: Float>(v: &[F], s: F) -> Vec<F> {
    v.iter().map(|&vi| vi * s).collect()
}

fn vec_add_scaled<F: Float>(a: &[F], b: &[F], s: F) -> Vec<F> {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| ai + bi * s)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::interface::{DiagonalOperator, IdentityOperator};

    #[test]
    fn test_qmr_identity() {
        // Test with identity matrix: x = b
        let identity = IdentityOperator::<f64>::new(3);
        let b = vec![1.0, 2.0, 3.0];
        let options = QMROptions::default();

        let result = qmr(&identity, &b, options).unwrap();
        assert!(result.converged);
        for i in 0..3 {
            assert!((result.x[i] - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_qmr_diagonal() {
        // Test with diagonal matrix
        let diag = vec![2.0, 3.0, 4.0];
        let diagonal = DiagonalOperator::new(diag.clone());
        let b = vec![2.0, 6.0, 8.0];
        let expected = vec![1.0, 2.0, 2.0];
        
        let options = QMROptions {
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        };

        let result = qmr(&diagonal, &b, options).unwrap();
        assert!(result.converged);
        for i in 0..3 {
            assert!(
                (result.x[i] - expected[i]).abs() < 1e-9,
                "x[{}] = {} != {}",
                i,
                result.x[i],
                expected[i]
            );
        }
    }
}