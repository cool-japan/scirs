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

/// QMR (Quasi-Minimal Residual) solver for non-symmetric systems
///
/// This implementation is a simplified version that provides QMR-like
/// behavior using a BiCG-based approach with quasi-minimization.
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

    // Compute initial residual r = b - Ax
    let mut r = if !x.iter().all(|&xi| xi == F::zero()) {
        let ax = a.matvec(&x)?;
        vec_sub(b, &ax)
    } else {
        b.to_vec()
    };

    // Apply left preconditioner if provided
    if let Some(ref ml) = options.left_preconditioner {
        r = ml.matvec(&r)?;
    }

    // Choose r_tilde = r for simplicity (can be randomized in practice)
    let mut r_tilde = r.clone();

    // Initialize Lanczos vectors
    let mut p = vec![F::zero(); n];
    let mut p_tilde = vec![F::zero(); n];
    let mut q = vec![F::zero(); n];
    let mut q_tilde;

    // Initialize scalars
    let mut rho = F::one();
    let mut rho_old;
    let mut alpha = F::zero();
    let mut omega = F::one();

    // Compute initial norms
    let bnorm = norm2(b);
    let mut rnorm = norm2(&r);
    let tol = options.atol + options.rtol * bnorm;

    // Check initial convergence
    if rnorm < tol {
        return Ok(QMRResult {
            x,
            iterations: 0,
            residual_norm: rnorm,
            converged: true,
            message: "Converged at initial guess".to_string(),
        });
    }

    // Main QMR loop
    for iter in 0..options.max_iter {
        // Store old rho
        rho_old = rho;

        // Compute rho = <r_tilde, r>
        rho = dot(&r_tilde, &r);

        // Check for breakdown
        if rho.abs() < F::epsilon() * F::from(10).unwrap() {
            return Ok(QMRResult {
                x,
                iterations: iter,
                residual_norm: rnorm,
                converged: false,
                message: "Breakdown: rho = 0".to_string(),
            });
        }

        // Compute beta
        let beta = if iter == 0 {
            F::zero()
        } else {
            (rho / rho_old) * (alpha / omega)
        };

        // Update p and p_tilde
        p = if iter == 0 {
            r.clone()
        } else {
            vec_add(&r, &vec_scaled(&vec_sub(&p, &vec_scaled(&q, omega)), beta))
        };

        // Apply preconditioner and operator
        let p_prec = if let Some(ref mr) = options.right_preconditioner {
            mr.matvec(&p)?
        } else {
            p.clone()
        };

        q = a.matvec(&p_prec)?;

        if let Some(ref ml) = options.left_preconditioner {
            q = ml.matvec(&q)?;
        }

        // Similarly for p_tilde and q_tilde
        p_tilde = if iter == 0 {
            r_tilde.clone()
        } else {
            let diff = vec_sub(&p_tilde, &vec_scaled(&q_tilde, omega));
            vec_add(&r_tilde, &vec_scaled(&diff, beta))
        };

        // Apply transpose operators
        let p_tilde_prec = if let Some(ref ml) = options.left_preconditioner {
            ml.rmatvec(&p_tilde)?
        } else {
            p_tilde.clone()
        };

        q_tilde = a.rmatvec(&p_tilde_prec)?;

        if let Some(ref mr) = options.right_preconditioner {
            q_tilde = mr.rmatvec(&q_tilde)?;
        }

        // Compute alpha
        let dot_pq = dot(&p_tilde, &q);
        if dot_pq.abs() < F::epsilon() * F::from(10).unwrap() {
            return Ok(QMRResult {
                x,
                iterations: iter,
                residual_norm: rnorm,
                converged: false,
                message: "Breakdown: <p_tilde, q> = 0".to_string(),
            });
        }

        alpha = rho / dot_pq;

        // Update residuals
        let s = vec_sub(&r, &vec_scaled(&q, alpha));
        let s_tilde = vec_sub(&r_tilde, &vec_scaled(&q_tilde, alpha));

        // Apply preconditioner and operator to s
        let s_prec = if let Some(ref mr) = options.right_preconditioner {
            mr.matvec(&s)?
        } else {
            s.clone()
        };

        let t = a.matvec(&s_prec)?;
        let t = if let Some(ref ml) = options.left_preconditioner {
            ml.matvec(&t)?
        } else {
            t
        };

        // Compute omega (quasi-minimization parameter)
        let dot_tt = dot(&t, &t);
        if dot_tt == F::zero() {
            omega = F::zero();
        } else {
            omega = dot(&t, &s) / dot_tt;
        }

        // Update solution
        x = vec_add(&x, &vec_scaled(&p_prec, alpha));
        x = vec_add(&x, &vec_scaled(&s_prec, omega));

        // Update residuals
        r = vec_sub(&s, &vec_scaled(&t, omega));

        // Update residual norm
        rnorm = norm2(&r);

        // Check convergence
        if rnorm < tol {
            return Ok(QMRResult {
                x,
                iterations: iter + 1,
                residual_norm: rnorm,
                converged: true,
                message: format!("Converged in {} iterations", iter + 1),
            });
        }

        // Check for stagnation
        if omega.abs() < F::epsilon() {
            return Ok(QMRResult {
                x,
                iterations: iter + 1,
                residual_norm: rnorm,
                converged: false,
                message: "Breakdown: omega = 0".to_string(),
            });
        }
    }

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
        // Test QMR with identity matrix: x = b
        let identity = IdentityOperator::<f64>::new(3);
        let b = vec![1.0, 2.0, 3.0];
        let options = QMROptions::default();

        let result = qmr(&identity, &b, options).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 1); // Should converge in 1 iteration
        for i in 0..3 {
            assert!((result.x[i] - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_qmr_diagonal() {
        // Test QMR with diagonal matrix
        let diag = vec![2.0, 3.0, 4.0];
        let diagonal = DiagonalOperator::new(diag.clone());
        let b = vec![2.0, 6.0, 8.0]; // [2, 3, 4] * [1, 2, 2] = [2, 6, 8]
        let expected = vec![1.0, 2.0, 2.0];

        let options = QMROptions {
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        };

        let result = qmr(&diagonal, &b, options).unwrap();
        assert!(result.converged);
        assert!(result.iterations <= 3);
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

    #[test]
    fn test_qmr_with_initial_guess() {
        // Test QMR with a good initial guess
        let identity = IdentityOperator::<f64>::new(3);
        let b = vec![1.0, 2.0, 3.0];
        let x0 = vec![0.9, 1.9, 2.9]; // Close to solution

        let options = QMROptions {
            x0: Some(x0),
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        };

        let result = qmr(&identity, &b, options).unwrap();
        assert!(result.converged);
        assert!(result.iterations <= 1); // Should converge very quickly
        for i in 0..3 {
            assert!((result.x[i] - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_qmr_max_iterations() {
        // Test QMR with a hard problem and limited iterations
        let diag = vec![1e-8, 1.0, 1.0]; // Ill-conditioned diagonal matrix
        let diagonal = DiagonalOperator::new(diag.clone());
        let b = vec![1.0, 1.0, 1.0];

        let options = QMROptions {
            max_iter: 5,
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        };

        let result = qmr(&diagonal, &b, options).unwrap();
        if !result.converged {
            assert_eq!(result.iterations, 5);
            assert!(result.message.contains("Did not converge"));
        }
    }
}
