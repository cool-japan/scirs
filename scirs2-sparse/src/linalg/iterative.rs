use crate::error::{SparseError, SparseResult};
use crate::linalg::interface::LinearOperator;
use num_traits::{Float, NumAssign};
use std::iter::Sum;

/// Result of an iterative solver
#[derive(Debug, Clone)]
pub struct IterationResult<F> {
    pub x: Vec<F>,
    pub iterations: usize,
    pub residual_norm: F,
    pub converged: bool,
    pub message: String,
}

/// Options for conjugate gradient solver
pub struct CGOptions<F> {
    pub max_iter: usize,
    pub rtol: F,
    pub atol: F,
    pub x0: Option<Vec<F>>,
    pub preconditioner: Option<Box<dyn LinearOperator<F>>>,
}

impl<F: Float> Default for CGOptions<F> {
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

/// Conjugate gradient solver for symmetric positive definite systems
///
/// Solves Ax = b where A is symmetric positive definite
pub fn cg<F>(
    a: &dyn LinearOperator<F>,
    b: &[F],
    options: CGOptions<F>,
) -> SparseResult<IterationResult<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(SparseError::ValueError(
            "Matrix must be square for CG solver".to_string(),
        ));
    }
    if b.len() != rows {
        return Err(SparseError::DimensionMismatch {
            expected: rows,
            found: b.len(),
        });
    }

    let n = rows;

    // Initialize solution
    let mut x: Vec<F> = match &options.x0 {
        Some(x0) => {
            if x0.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: x0.len(),
                });
            }
            x0.clone()
        }
        None => vec![F::zero(); n],
    };

    // Compute initial residual: r = b - A*x
    let ax = a.matvec(&x)?;
    let mut r: Vec<F> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();

    // Apply preconditioner if provided
    let mut z = if let Some(m) = &options.preconditioner {
        m.matvec(&r)?
    } else {
        r.clone()
    };

    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);

    // Check for convergence
    let bnorm = norm2(b);
    let tolerance = F::max(options.atol, options.rtol * bnorm);

    let mut iterations = 0;
    while iterations < options.max_iter {
        // Compute Ap
        let ap = a.matvec(&p)?;

        // Compute alpha = (r,z) / (p,Ap)
        let pap = dot(&p, &ap);
        if pap <= F::zero() {
            return Ok(IterationResult {
                x,
                iterations,
                residual_norm: norm2(&r),
                converged: false,
                message: "Matrix not positive definite (p^T*A*p <= 0)".to_string(),
            });
        }
        let alpha = rz_old / pap;

        // Update solution: x = x + alpha*p
        for i in 0..n {
            x[i] += alpha * p[i];
        }

        // Update residual: r = r - alpha*Ap
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }

        // Check for convergence
        let rnorm = norm2(&r);
        if rnorm <= tolerance {
            return Ok(IterationResult {
                x,
                iterations: iterations + 1,
                residual_norm: rnorm,
                converged: true,
                message: "Converged".to_string(),
            });
        }

        // Apply preconditioner
        z = if let Some(m) = &options.preconditioner {
            m.matvec(&r)?
        } else {
            r.clone()
        };

        // Compute beta = (r_{i+1},z_{i+1}) / (r_i,z_i)
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz_old;

        // Update direction: p = z + beta*p
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }

        rz_old = rz_new;
        iterations += 1;
    }

    Ok(IterationResult {
        x,
        iterations,
        residual_norm: norm2(&r),
        converged: false,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Options for BiCG solver
pub struct BiCGOptions<F> {
    pub max_iter: usize,
    pub rtol: F,
    pub atol: F,
    pub x0: Option<Vec<F>>,
    pub left_preconditioner: Option<Box<dyn LinearOperator<F>>>,
    pub right_preconditioner: Option<Box<dyn LinearOperator<F>>>,
}

impl<F: Float> Default for BiCGOptions<F> {
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

/// Biconjugate Gradient solver
///
/// Solves Ax = b where A is non-symmetric.
pub fn bicg<F>(
    a: &dyn LinearOperator<F>,
    b: &[F],
    options: BiCGOptions<F>,
) -> SparseResult<IterationResult<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(SparseError::ValueError(
            "Matrix must be square for BiCG solver".to_string(),
        ));
    }
    if b.len() != rows {
        return Err(SparseError::DimensionMismatch {
            expected: rows,
            found: b.len(),
        });
    }

    let n = rows;

    // Initialize solution
    let mut x: Vec<F> = match &options.x0 {
        Some(x0) => {
            if x0.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: x0.len(),
                });
            }
            x0.clone()
        }
        None => vec![F::zero(); n],
    };

    // Compute initial residual: r = b - A*x
    let ax = a.matvec(&x)?;
    let mut r: Vec<F> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();

    // Initialize r_star (shadow residual) = r
    let mut r_star = r.clone();

    // Apply preconditioners to initial residuals
    let mut z = if let Some(m1) = &options.left_preconditioner {
        m1.matvec(&r)?
    } else {
        r.clone()
    };

    let mut z_star = if let Some(m2) = &options.right_preconditioner {
        m2.matvec(&r_star)?
    } else {
        r_star.clone()
    };

    let mut p = z.clone();
    let mut p_star = z_star.clone();

    let mut rho_old = dot(&r_star, &z);

    let bnorm = norm2(b);
    let tolerance = F::max(options.atol, options.rtol * bnorm);

    let mut iterations = 0;
    while iterations < options.max_iter {
        // Compute q = A*p and q_star = A^T*p_star
        let mut q = a.matvec(&p)?;
        if let Some(m2) = &options.right_preconditioner {
            q = m2.matvec(&q)?;
        }

        let mut q_star = a.rmatvec(&p_star)?;
        if let Some(m1) = &options.left_preconditioner {
            q_star = m1.matvec(&q_star)?;
        }

        // Compute alpha = rho_old / (p_star, q)
        let alpha_den = dot(&p_star, &q);
        if alpha_den.abs() < F::epsilon() * F::from(10).unwrap() {
            return Ok(IterationResult {
                x,
                iterations,
                residual_norm: norm2(&r),
                converged: false,
                message: "BiCG breakdown: (p_star, q) ≈ 0".to_string(),
            });
        }
        let alpha = rho_old / alpha_den;

        // Update solution and residuals
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
            r_star[i] -= alpha * q_star[i];
        }

        // Check for convergence - compute residual norm BEFORE the next iteration
        let rnorm = norm2(&r);
        if rnorm <= tolerance {
            return Ok(IterationResult {
                x,
                iterations: iterations + 1,
                residual_norm: rnorm,
                converged: true,
                message: "Converged".to_string(),
            });
        }

        // Apply preconditioners
        z = if let Some(m1) = &options.left_preconditioner {
            m1.matvec(&r)?
        } else {
            r.clone()
        };

        z_star = if let Some(m2) = &options.right_preconditioner {
            m2.matvec(&r_star)?
        } else {
            r_star.clone()
        };

        // Compute new rho
        let rho = dot(&r_star, &z);
        if rho.abs() < F::epsilon() * F::from(10).unwrap() {
            return Ok(IterationResult {
                x,
                iterations: iterations + 1,
                residual_norm: rnorm,
                converged: false,
                message: "BiCG breakdown: rho ≈ 0".to_string(),
            });
        }

        // Compute beta = rho / rho_old
        let beta = rho / rho_old;

        // Update search directions
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
            p_star[i] = z_star[i] + beta * p_star[i];
        }

        rho_old = rho;
        iterations += 1;
    }

    Ok(IterationResult {
        x,
        iterations,
        residual_norm: norm2(&r),
        converged: false,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Options for BiCGSTAB solver
pub struct BiCGSTABOptions<F> {
    pub max_iter: usize,
    pub rtol: F,
    pub atol: F,
    pub x0: Option<Vec<F>>,
    pub left_preconditioner: Option<Box<dyn LinearOperator<F>>>,
    pub right_preconditioner: Option<Box<dyn LinearOperator<F>>>,
}

impl<F: Float> Default for BiCGSTABOptions<F> {
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

/// Result from BiCGSTAB solver
pub type BiCGSTABResult<F> = IterationResult<F>;

/// BiConjugate Gradient Stabilized method
///
/// An improved version of BiCG that avoids the irregular convergence patterns
/// and has better numerical stability. Works for general non-symmetric systems.
pub fn bicgstab<F>(
    a: &dyn LinearOperator<F>,
    b: &[F],
    options: BiCGSTABOptions<F>,
) -> SparseResult<BiCGSTABResult<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(SparseError::ValueError(
            "Matrix must be square for BiCGSTAB solver".to_string(),
        ));
    }
    if b.len() != rows {
        return Err(SparseError::DimensionMismatch {
            expected: rows,
            found: b.len(),
        });
    }

    let n = rows;

    // Initialize solution
    let mut x: Vec<F> = match &options.x0 {
        Some(x0) => {
            if x0.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: x0.len(),
                });
            }
            x0.clone()
        }
        None => vec![F::zero(); n],
    };

    // Compute initial residual: r = b - A*x
    let ax = a.matvec(&x)?;
    let mut r: Vec<F> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();

    // Check if initial guess is solution
    let mut rnorm = norm2(&r);
    let bnorm = norm2(b);
    let tolerance = F::max(options.atol, options.rtol * bnorm);

    if rnorm <= tolerance {
        return Ok(BiCGSTABResult {
            x,
            iterations: 0,
            residual_norm: rnorm,
            converged: true,
            message: "Converged with initial guess".to_string(),
        });
    }

    // Choose shadow residual (r_hat) as r0
    let r_hat = r.clone();

    let mut v = vec![F::zero(); n];
    let mut p = vec![F::zero(); n];
    let mut y = vec![F::zero(); n];
    let mut s = vec![F::zero(); n];
    let mut t: Vec<F>;

    let mut rho_old = F::one();
    let mut alpha = F::zero();
    let mut omega = F::one();

    // Main iteration loop
    let mut iterations = 0;
    while iterations < options.max_iter {
        // Compute rho = (r_hat, r)
        let rho = dot(&r_hat, &r);

        // Check for breakdown
        if rho.abs() < F::epsilon() * F::from(10).unwrap() {
            return Ok(BiCGSTABResult {
                x,
                iterations,
                residual_norm: rnorm,
                converged: false,
                message: "BiCGSTAB breakdown: rho ≈ 0".to_string(),
            });
        }

        // Compute beta and update p
        let beta = (rho / rho_old) * (alpha / omega);

        // p = r + beta * (p - omega * v)
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // Apply left preconditioner if provided
        let p_tilde = match &options.left_preconditioner {
            Some(m1) => m1.matvec(&p)?,
            None => p.clone(),
        };

        // Compute v = A * p_tilde
        v = a.matvec(&p_tilde)?;

        // Apply right preconditioner if provided
        if let Some(m2) = &options.right_preconditioner {
            v = m2.matvec(&v)?;
        }

        // Compute alpha = rho / (r_hat, v)
        let den = dot(&r_hat, &v);
        if den.abs() < F::epsilon() * F::from(10).unwrap() {
            return Ok(BiCGSTABResult {
                x,
                iterations,
                residual_norm: rnorm,
                converged: false,
                message: "BiCGSTAB breakdown: (r_hat, v) ≈ 0".to_string(),
            });
        }
        alpha = rho / den;

        // Check if alpha is reasonable
        if !alpha.is_finite() {
            return Ok(BiCGSTABResult {
                x,
                iterations,
                residual_norm: rnorm,
                converged: false,
                message: "BiCGSTAB breakdown: alpha is not finite".to_string(),
            });
        }

        // Update solution and residual: s = r - alpha * v
        for i in 0..n {
            y[i] = x[i] + alpha * p_tilde[i];
            s[i] = r[i] - alpha * v[i];
        }

        // Check convergence
        let snorm = norm2(&s);
        if snorm <= tolerance {
            // Final update: x = y
            x = y;

            // Apply right preconditioner to final solution if provided
            if let Some(m2) = &options.right_preconditioner {
                x = m2.matvec(&x)?;
            }

            return Ok(BiCGSTABResult {
                x,
                iterations: iterations + 1,
                residual_norm: snorm,
                converged: true,
                message: "Converged".to_string(),
            });
        }

        // Apply left preconditioner to s if provided
        let s_tilde = match &options.left_preconditioner {
            Some(m1) => m1.matvec(&s)?,
            None => s.clone(),
        };

        // Compute t = A * s_tilde
        t = a.matvec(&s_tilde)?;

        // Apply right preconditioner if provided
        if let Some(m2) = &options.right_preconditioner {
            t = m2.matvec(&t)?;
        }

        // Compute omega = (t, s) / (t, t)
        let ts = dot(&t, &s);
        let tt = dot(&t, &t);

        if tt < F::epsilon() * F::from(10).unwrap() {
            return Ok(BiCGSTABResult {
                x,
                iterations,
                residual_norm: rnorm,
                converged: false,
                message: "BiCGSTAB breakdown: (t, t) ≈ 0".to_string(),
            });
        }

        omega = ts / tt;

        // Check if omega is reasonable
        if !omega.is_finite() || omega.abs() < F::epsilon() * F::from(10).unwrap() {
            return Ok(BiCGSTABResult {
                x,
                iterations,
                residual_norm: rnorm,
                converged: false,
                message: "BiCGSTAB breakdown: omega is not finite or too small".to_string(),
            });
        }

        // Update solution: x = y + omega * s_tilde
        for i in 0..n {
            x[i] = y[i] + omega * s_tilde[i];
            r[i] = s[i] - omega * t[i];
        }

        // Apply right preconditioner to final solution if provided
        if let Some(m2) = &options.right_preconditioner {
            x = m2.matvec(&x)?;
        }

        rnorm = norm2(&r);

        // Check for convergence
        if rnorm <= tolerance {
            return Ok(BiCGSTABResult {
                x,
                iterations: iterations + 1,
                residual_norm: rnorm,
                converged: true,
                message: "Converged".to_string(),
            });
        }

        rho_old = rho;
        iterations += 1;
    }

    Ok(BiCGSTABResult {
        x,
        iterations,
        residual_norm: rnorm,
        converged: false,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Options for GMRES solver
pub struct GMRESOptions<F> {
    pub max_iter: usize,
    pub restart: usize,
    pub rtol: F,
    pub atol: F,
    pub x0: Option<Vec<F>>,
    pub preconditioner: Option<Box<dyn LinearOperator<F>>>,
}

impl<F: Float> Default for GMRESOptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            restart: 30,
            rtol: F::from(1e-8).unwrap(),
            atol: F::from(1e-12).unwrap(),
            x0: None,
            preconditioner: None,
        }
    }
}

/// Generalized Minimal Residual Method
///
/// Solves Ax = b for general non-symmetric systems. GMRES is particularly
/// robust but requires more memory than other methods due to the need to
/// store the Krylov basis vectors.
pub fn gmres<F>(
    a: &dyn LinearOperator<F>,
    b: &[F],
    options: GMRESOptions<F>,
) -> SparseResult<IterationResult<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(SparseError::ValueError(
            "Matrix must be square for GMRES solver".to_string(),
        ));
    }
    if b.len() != rows {
        return Err(SparseError::DimensionMismatch {
            expected: rows,
            found: b.len(),
        });
    }

    let n = rows;
    let restart = options.restart.min(n);

    // Initialize solution
    let mut x: Vec<F> = match &options.x0 {
        Some(x0) => {
            if x0.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: x0.len(),
                });
            }
            x0.clone()
        }
        None => vec![F::zero(); n],
    };

    // Compute initial residual: r = b - A*x
    let ax = a.matvec(&x)?;
    let mut r: Vec<F> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();

    // Apply preconditioner if provided
    if let Some(m) = &options.preconditioner {
        r = m.matvec(&r)?;
    }

    let mut rnorm = norm2(&r);
    let bnorm = norm2(b);
    let tolerance = F::max(options.atol, options.rtol * bnorm);

    let mut outer_iterations = 0;

    // Outer iteration loop (restarts)
    while outer_iterations < options.max_iter && rnorm > tolerance {
        // Initialize Krylov subspace
        let mut v = vec![vec![F::zero(); n]; restart + 1];
        let mut h = vec![vec![F::zero(); restart]; restart + 1];
        let mut cs = vec![F::zero(); restart]; // Cosines for Givens rotations
        let mut sn = vec![F::zero(); restart]; // Sines for Givens rotations
        let mut s = vec![F::zero(); restart + 1]; // RHS for triangular system

        // Set up initial vector
        v[0] = r.iter().map(|&ri| ri / rnorm).collect();
        s[0] = rnorm;

        // Arnoldi iteration
        let mut inner_iter = 0;
        while inner_iter < restart && inner_iter + outer_iterations < options.max_iter {
            // Compute w = A * v[j]
            let mut w = a.matvec(&v[inner_iter])?;

            // Apply preconditioner if provided
            if let Some(m) = &options.preconditioner {
                w = m.matvec(&w)?;
            }

            // Orthogonalize against previous vectors
            for i in 0..=inner_iter {
                h[i][inner_iter] = dot(&v[i], &w);
                for (k, w_elem) in w.iter_mut().enumerate().take(n) {
                    *w_elem -= h[i][inner_iter] * v[i][k];
                }
            }

            h[inner_iter + 1][inner_iter] = norm2(&w);

            // Check for breakdown
            if h[inner_iter + 1][inner_iter] < F::epsilon() * F::from(10).unwrap() {
                break;
            }

            // Normalize w and store in v[j+1]
            v[inner_iter + 1] = w
                .iter()
                .map(|&wi| wi / h[inner_iter + 1][inner_iter])
                .collect();

            // Apply previous Givens rotations
            for i in 0..inner_iter {
                let temp = cs[i] * h[i][inner_iter] + sn[i] * h[i + 1][inner_iter];
                h[i + 1][inner_iter] = -sn[i] * h[i][inner_iter] + cs[i] * h[i + 1][inner_iter];
                h[i][inner_iter] = temp;
            }

            // Compute new Givens rotation
            let rho = (h[inner_iter][inner_iter] * h[inner_iter][inner_iter]
                + h[inner_iter + 1][inner_iter] * h[inner_iter + 1][inner_iter])
                .sqrt();
            cs[inner_iter] = h[inner_iter][inner_iter] / rho;
            sn[inner_iter] = h[inner_iter + 1][inner_iter] / rho;

            // Apply new Givens rotation
            h[inner_iter][inner_iter] = rho;
            h[inner_iter + 1][inner_iter] = F::zero();

            let temp = cs[inner_iter] * s[inner_iter] + sn[inner_iter] * s[inner_iter + 1];
            s[inner_iter + 1] =
                -sn[inner_iter] * s[inner_iter] + cs[inner_iter] * s[inner_iter + 1];
            s[inner_iter] = temp;

            inner_iter += 1;

            // Check for convergence
            let residual = s[inner_iter].abs();
            if residual <= tolerance {
                break;
            }
        }

        // Solve the upper triangular system
        let mut y = vec![F::zero(); inner_iter];
        for i in (0..inner_iter).rev() {
            y[i] = s[i];
            for j in i + 1..inner_iter {
                y[i] = y[i] - h[i][j] * y[j];
            }
            y[i] /= h[i][i];
        }

        // Update solution
        for i in 0..inner_iter {
            for j in 0..n {
                x[j] += y[i] * v[i][j];
            }
        }

        // Compute new residual
        let ax = a.matvec(&x)?;
        r = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();

        // Apply preconditioner if provided
        if let Some(m) = &options.preconditioner {
            r = m.matvec(&r)?;
        }

        rnorm = norm2(&r);
        outer_iterations += inner_iter;

        if rnorm <= tolerance {
            break;
        }
    }

    Ok(IterationResult {
        x,
        iterations: outer_iterations,
        residual_norm: rnorm,
        converged: rnorm <= tolerance,
        message: if rnorm <= tolerance {
            "Converged".to_string()
        } else {
            "Maximum iterations reached".to_string()
        },
    })
}

/// Trait for iterative solvers
pub trait IterativeSolver<F: Float> {
    /// Solve the linear system Ax = b
    fn solve(&self, a: &dyn LinearOperator<F>, b: &[F]) -> SparseResult<IterationResult<F>>;
}

// Helper functions

/// Compute the dot product of two vectors
pub(crate) fn dot<F: Float + Sum>(x: &[F], y: &[F]) -> F {
    x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum()
}

/// Compute the 2-norm of a vector
pub(crate) fn norm2<F: Float + Sum>(x: &[F]) -> F {
    dot(x, x).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;
    use crate::linalg::interface::AsLinearOperator;

    #[test]
    fn test_cg_identity() {
        // Test CG on identity matrix: I * x = b => x = b
        let identity = crate::linalg::interface::IdentityOperator::<f64>::new(3);
        let b = vec![1.0, 2.0, 3.0];
        let options = CGOptions::default();
        let result = cg(&identity, &b, options).unwrap();

        assert!(result.converged);
        for (xi, bi) in result.x.iter().zip(&b) {
            assert!((xi - bi).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cg_diagonal() {
        // Test CG on diagonal matrix
        let diag = crate::linalg::interface::DiagonalOperator::new(vec![2.0, 3.0, 4.0]);
        let b = vec![2.0, 6.0, 12.0];
        let options = CGOptions::default();
        let result = cg(&diag, &b, options).unwrap();

        assert!(result.converged);
        let expected = vec![1.0, 2.0, 3.0];
        for (xi, ei) in result.x.iter().zip(&expected) {
            assert!((xi - ei).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cg_sparse_matrix() {
        // Test CG on a sparse positive definite matrix
        let rows = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let cols = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let data = vec![4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0];
        let shape = (3, 3);

        let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
        let op = matrix.as_linear_operator();

        let b = vec![1.0, 2.0, 3.0];
        let options = CGOptions::default();
        let result = cg(op.as_ref(), &b, options).unwrap();

        assert!(result.converged);
        // Verify solution by checking Ax = b
        let ax = op.matvec(&result.x).unwrap();
        for (axi, bi) in ax.iter().zip(&b) {
            assert!((axi - bi).abs() < 1e-9);
        }
    }

    #[test]
    fn test_bicgstab_identity() {
        // Test BiCGSTAB on identity matrix: I * x = b => x = b
        let identity = crate::linalg::interface::IdentityOperator::<f64>::new(3);
        let b = vec![1.0, 2.0, 3.0];
        let options = BiCGSTABOptions::default();
        let result = bicgstab(&identity, &b, options).unwrap();

        assert!(result.converged);
        for (xi, bi) in result.x.iter().zip(&b) {
            assert!((xi - bi).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bicgstab_diagonal() {
        // Test BiCGSTAB on diagonal matrix
        let diag = crate::linalg::interface::DiagonalOperator::new(vec![2.0, 3.0, 4.0]);
        let b = vec![2.0, 6.0, 12.0];
        let options = BiCGSTABOptions::default();
        let result = bicgstab(&diag, &b, options).unwrap();

        assert!(result.converged);
        let expected = vec![1.0, 2.0, 3.0];
        for (xi, ei) in result.x.iter().zip(&expected) {
            assert!((xi - ei).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bicgstab_non_symmetric() {
        // Test BiCGSTAB on a non-symmetric matrix
        let rows = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let cols = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let data = vec![4.0, -1.0, -2.0, -1.0, 4.0, -1.0, 0.0, -1.0, 3.0];
        let shape = (3, 3);

        let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
        let op = matrix.as_linear_operator();

        let b = vec![1.0, 2.0, 1.0];
        let options = BiCGSTABOptions::default();
        let result = bicgstab(op.as_ref(), &b, options).unwrap();

        assert!(result.converged);
        // Verify solution by checking Ax = b
        let ax = op.matvec(&result.x).unwrap();
        for (axi, bi) in ax.iter().zip(&b) {
            assert!((axi - bi).abs() < 1e-9);
        }
    }
}
