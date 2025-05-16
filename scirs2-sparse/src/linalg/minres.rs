use crate::error::{SparseError, SparseResult};
use crate::linalg::interface::LinearOperator;
use num_traits::{Float, NumAssign};
use std::iter::Sum;

/// Result of MINRES solver
#[derive(Debug, Clone)]
pub struct MINRESResult<F> {
    pub x: Vec<F>,
    pub iterations: usize,
    pub residual_norm: F,
    pub converged: bool,
    pub message: String,
}

/// Options for MINRES solver
pub struct MINRESOptions<F> {
    pub max_iter: usize,
    pub rtol: F,
    pub atol: F,
    pub x0: Option<Vec<F>>,
    pub preconditioner: Option<Box<dyn LinearOperator<F>>>,
}

impl<F: Float> Default for MINRESOptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            rtol: F::from(1e-8).unwrap(),
            atol: F::from(1e-12).unwrap(),
            x0: None,
            preconditioner: None,
        }
    }
}

/// MINRES solver for symmetric indefinite systems
///
/// Solves Ax = b where A is symmetric (possibly indefinite)
/// using the Minimal Residual method.
///
/// Based on:
/// C. C. Paige and M. A. Saunders (1975),
/// "Solution of sparse indefinite systems of linear equations",
/// SIAM J. Numer. Anal. 12(4), pp. 617-629.
///
/// # Arguments
/// * `a` - The linear operator representing the matrix A
/// * `b` - The right-hand side vector
/// * `options` - Solver options including maximum iterations and tolerances
///
/// # Returns
/// A `MINRESResult` containing the solution vector, iteration count, residual norm,
/// and convergence status.
///
/// # Example
/// ```
/// use scirs2_sparse::linalg::{minres, MINRESOptions, IdentityOperator};
///
/// let identity = IdentityOperator::<f64>::new(3);
/// let b = vec![1.0, 2.0, 3.0];
/// let options = MINRESOptions::default();
/// let result = minres(&identity, &b, options).unwrap();
/// assert!(result.converged);
/// ```
pub fn minres<F>(
    a: &dyn LinearOperator<F>,
    b: &[F],
    options: MINRESOptions<F>,
) -> SparseResult<MINRESResult<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    let (m, n) = a.shape();
    if m != n {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: n,
        });
    }

    if b.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: b.len(),
        });
    }

    // Initial guess
    let mut x = options.x0.clone().unwrap_or_else(|| vec![F::zero(); n]);
    let bnorm = norm2(b);

    if bnorm < options.atol {
        return Ok(MINRESResult {
            x,
            iterations: 0,
            residual_norm: F::zero(),
            converged: true,
            message: "System has zero right-hand side".to_string(),
        });
    }

    // Compute initial residual
    let r1 = compute_residual(a, b, &x)?;

    // Apply preconditioner if available
    let y = if let Some(ref prec) = options.preconditioner {
        prec.matvec(&r1)?
    } else {
        r1.clone()
    };

    // beta1 = sqrt(r1' * M^{-1} * r1)
    let beta1_sq = inner(&r1, &y);

    if beta1_sq < F::zero() {
        return Err(SparseError::ComputationError(
            "Indefinite preconditioner".to_string(),
        ));
    }

    if beta1_sq == F::zero() {
        return Ok(MINRESResult {
            x,
            iterations: 0,
            residual_norm: F::zero(),
            converged: true,
            message: "Initial residual is zero".to_string(),
        });
    }

    let beta1 = beta1_sq.sqrt();

    // Initialize quantities for the iteration
    let mut oldb = F::zero();
    let mut beta = beta1;
    let mut dbar = F::zero();
    let mut epsln = F::zero();
    let mut phibar = beta1;
    let mut cs = F::from(-1.0).unwrap();
    let mut sn = F::zero();
    let mut w = vec![F::zero(); n];
    let mut w2 = vec![F::zero(); n];
    let mut r2 = r1.clone();
    let mut v = vec![F::zero(); n];
    let mut r1 = vec![F::zero(); n];
    let mut y_vec = y;

    let mut gmax = F::zero();
    let mut gmin = F::max_value();
    let mut tnorm2 = F::zero();
    let mut qrnorm = beta1;

    let eps = F::epsilon();

    for itn in 0..options.max_iter {
        // Lanczos iteration
        let s = F::one() / beta;
        for i in 0..n {
            v[i] = s * y_vec[i];
        }

        let mut y_new = a.matvec(&v)?;

        if itn >= 1 {
            for i in 0..n {
                y_new[i] -= (beta / oldb) * r1[i];
            }
        }

        let alfa = inner(&v, &y_new);

        for i in 0..n {
            y_new[i] -= (alfa / beta) * r2[i];
        }

        r1 = r2;
        r2 = y_new;

        y_vec = if let Some(ref prec) = options.preconditioner {
            prec.matvec(&r2)?
        } else {
            r2.clone()
        };

        oldb = beta;
        let beta_sq = inner(&r2, &y_vec);

        if beta_sq < F::zero() {
            return Err(SparseError::ComputationError(
                "Non-symmetric matrix".to_string(),
            ));
        }

        beta = beta_sq.sqrt();
        tnorm2 = tnorm2 + alfa * alfa + oldb * oldb + beta * beta;

        // Apply previous rotation
        let oldeps = epsln;
        let delta = cs * dbar + sn * alfa;
        let gbar = sn * dbar - cs * alfa;
        epsln = sn * beta;
        dbar = -cs * beta;

        // Compute next rotation
        let gamma = (gbar * gbar + beta * beta).sqrt();
        let gamma_clamped = if gamma > eps { gamma } else { eps };
        cs = gbar / gamma_clamped;
        sn = beta / gamma_clamped;
        let phi = cs * phibar;
        phibar = sn * phibar;

        // Update solution
        let denom = F::one() / gamma_clamped;
        let w1 = w2;
        w2 = w;
        w = vec![F::zero(); n];
        for i in 0..n {
            w[i] = (v[i] - oldeps * w1[i] - delta * w2[i]) * denom;
            x[i] += phi * w[i];
        }

        // Update residual norm
        gmax = gmax.max(gamma);
        gmin = gmin.min(gamma);
        qrnorm = phibar;
        let rnorm = qrnorm;

        // Check convergence
        let anorm = tnorm2.sqrt();
        let ynorm = norm2(&x);

        let test1 = if ynorm == F::zero() || anorm == F::zero() {
            F::infinity()
        } else {
            rnorm / (anorm * ynorm)
        };

        if test1 <= options.rtol || rnorm <= options.atol {
            return Ok(MINRESResult {
                x,
                iterations: itn + 1,
                residual_norm: rnorm,
                converged: true,
                message: "Converged".to_string(),
            });
        }
    }

    Ok(MINRESResult {
        x,
        iterations: options.max_iter,
        residual_norm: qrnorm,
        converged: false,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Compute residual r = b - Ax
fn compute_residual<F>(a: &dyn LinearOperator<F>, b: &[F], x: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign,
{
    let ax = a.matvec(x)?;
    Ok(b.iter()
        .zip(ax.iter())
        .map(|(&bi, &axi)| bi - axi)
        .collect())
}

/// Compute 2-norm of a vector
fn norm2<F: Float + Sum>(v: &[F]) -> F {
    v.iter().map(|&x| x * x).sum::<F>().sqrt()
}

/// Compute inner product of two vectors
fn inner<F: Float + Sum>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;
    use crate::linalg::interface::{AsLinearOperator, DiagonalOperator, IdentityOperator};

    #[test]
    fn test_minres_identity() {
        // Test MINRES on identity matrix: I * x = b => x = b
        let identity = IdentityOperator::<f64>::new(3);
        let b = vec![1.0, 2.0, 3.0];
        let options = MINRESOptions::default();
        let result = minres(&identity, &b, options).unwrap();

        println!("Identity test result: {:?}", result);
        println!("Expected x: {:?}", b);
        println!("Actual x: {:?}", result.x);

        // Verify solution
        let ax = identity.matvec(&result.x).unwrap();
        println!("Ax = {:?}", ax);
        println!("b  = {:?}", b);

        assert!(result.converged);
        for (i, (xi, bi)) in result.x.iter().zip(&b).enumerate() {
            let diff = (xi - bi).abs();
            println!("x[{}]: expected {}, got {}, diff {}", i, bi, xi, diff);
            assert!(
                diff < 1e-10,
                "x[{}]: expected {}, got {}, diff {}",
                i,
                bi,
                xi,
                diff
            );
        }
    }

    #[test]
    fn test_minres_diagonal() {
        // Test MINRES on diagonal matrix
        let diag = DiagonalOperator::new(vec![2.0, -3.0, 4.0]); // Note: can have negative values
        let b = vec![2.0, -6.0, 12.0];
        let options = MINRESOptions::default();
        let result = minres(&diag, &b, options).unwrap();

        println!("Result: {:?}", result);
        println!("Solution x: {:?}", result.x);
        println!("Iterations: {}", result.iterations);
        println!("Converged: {}", result.converged);
        println!("Residual norm: {}", result.residual_norm);

        // Verify solution
        let ax = diag.matvec(&result.x).unwrap();
        println!("Ax = {:?}", ax);
        println!("b  = {:?}", b);

        assert!(result.converged);
        let expected = vec![1.0, 2.0, 3.0];
        for (xi, ei) in result.x.iter().zip(&expected) {
            assert!((xi - ei).abs() < 1e-9, "Expected {}, got {}", ei, xi);
        }
    }

    #[test]
    fn test_minres_symmetric_indefinite() {
        // Test MINRES on a symmetric indefinite matrix
        // A = [ 4  -1  0]
        //     [-1   2 -1]
        //     [ 0  -1  4]
        // This is symmetric but not positive definite
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let data = vec![4.0, -1.0, -1.0, 2.0, -1.0, -1.0, 4.0];
        let shape = (3, 3);

        let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
        let op = matrix.as_linear_operator();

        let b = vec![1.0, 0.0, 1.0];
        let options = MINRESOptions::default();
        let result = minres(op.as_ref(), &b, options).unwrap();

        assert!(result.converged);
        // Verify solution by checking Ax = b
        let ax = op.matvec(&result.x).unwrap();
        for (axi, bi) in ax.iter().zip(&b) {
            assert!((axi - bi).abs() < 1e-9);
        }
    }

    #[test]
    fn test_minres_preconditioned() {
        // Test MINRES with a preconditioner
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let data = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let shape = (3, 3);

        let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
        let op = matrix.as_linear_operator();

        let b = vec![1.0, 2.0, 3.0];

        // Use Jacobi preconditioner (diagonal matrix)
        let preconditioner = Box::new(DiagonalOperator::new(vec![1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0]));

        let options = MINRESOptions {
            preconditioner: Some(preconditioner),
            ..Default::default()
        };

        let result = minres(op.as_ref(), &b, options).unwrap();

        assert!(result.converged);
        // Verify solution by checking Ax = b
        let ax = op.matvec(&result.x).unwrap();
        for (axi, bi) in ax.iter().zip(&b) {
            assert!((axi - bi).abs() < 1e-9);
        }
    }
}
