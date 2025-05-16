//! Loose GMRES (LGMRES) linear solver

use crate::error::{SparseError, SparseResult};
use crate::linalg::interface::LinearOperator;
use num_traits::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;

/// Options for the LGMRES solver
pub struct LGMRESOptions<F> {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Relative tolerance
    pub rtol: F,
    /// Absolute tolerance
    pub atol: F,
    /// Number of inner GMRES iterations
    pub inner_m: usize,
    /// Number of outer iterations
    pub outer_k: usize,
    /// Initial guess (if None, uses zero vector)
    pub x0: Option<Vec<F>>,
    /// Preconditioner
    pub preconditioner: Option<Box<dyn LinearOperator<F>>>,
}

impl<F: Float> Default for LGMRESOptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 500,
            rtol: F::from(1e-8).unwrap(),
            atol: F::from(1e-8).unwrap(),
            inner_m: 30,
            outer_k: 3,
            x0: None,
            preconditioner: None,
        }
    }
}

/// Result of the LGMRES solver
#[derive(Debug, Clone)]
pub struct LGMRESResult<F> {
    /// Solution vector
    pub x: Vec<F>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Residual norm
    pub residual_norm: F,
    /// Whether the solver converged
    pub converged: bool,
}

/// Solve A*x = b using the LGMRES method
///
/// LGMRES (Loose GMRES) is a variant of restarted GMRES that supplements
/// the Krylov subspace with approximations to error from previous restart cycles.
pub fn lgmres<F>(
    a: &dyn LinearOperator<F>,
    b: &[F],
    options: LGMRESOptions<F>,
) -> SparseResult<LGMRESResult<F>>
where
    F: Float + NumAssign + Sum + Debug + 'static,
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

    // Initial setup
    let mut x = options.x0.clone().unwrap_or_else(|| vec![F::zero(); n]);
    let b_norm = b.iter().map(|&bi| bi * bi).sum::<F>().sqrt();

    if b_norm < options.atol {
        return Ok(LGMRESResult {
            x,
            iterations: 0,
            residual_norm: F::zero(),
            converged: true,
        });
    }

    // Compute initial residual
    let mut r = if let Some(ref m) = options.preconditioner {
        let ax = a.matvec(&x)?;
        let residual: Vec<F> = b
            .iter()
            .zip(ax.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect();
        m.matvec(&residual)?
    } else {
        let ax = a.matvec(&x)?;
        b.iter()
            .zip(ax.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect()
    };

    let mut r_norm = r.iter().map(|&ri| ri * ri).sum::<F>().sqrt();

    if r_norm < options.atol || r_norm / b_norm < options.rtol {
        return Ok(LGMRESResult {
            x,
            iterations: 0,
            residual_norm: r_norm,
            converged: true,
        });
    }

    // Outer iterations
    let mut augmented_vectors: Vec<Vec<F>> = Vec::new();
    let mut total_iter = 0;

    for _ in 0..options.max_iter {
        // Perform inner GMRES iterations
        let (y, _new_r_norm, v_list) = inner_gmres(
            a,
            &r,
            options.inner_m,
            &augmented_vectors,
            options.preconditioner.as_deref(),
        )?;

        // Update solution
        for (xi, &yi) in x.iter_mut().zip(y.iter()) {
            *xi += yi;
        }

        total_iter += 1;

        // Update residual
        if let Some(ref m) = options.preconditioner {
            let ax = a.matvec(&x)?;
            let residual: Vec<F> = b
                .iter()
                .zip(ax.iter())
                .map(|(&bi, &axi)| bi - axi)
                .collect();
            r = m.matvec(&residual)?;
        } else {
            let ax = a.matvec(&x)?;
            r = b
                .iter()
                .zip(ax.iter())
                .map(|(&bi, &axi)| bi - axi)
                .collect();
        }

        r_norm = r.iter().map(|&ri| ri * ri).sum::<F>().sqrt();

        // Check convergence
        if r_norm < options.atol || r_norm / b_norm < options.rtol {
            return Ok(LGMRESResult {
                x,
                iterations: total_iter,
                residual_norm: r_norm,
                converged: true,
            });
        }

        // Update augmented vectors (keep last outer_k vectors)
        for v in v_list {
            augmented_vectors.push(v);
        }
        if augmented_vectors.len() > options.outer_k {
            augmented_vectors.drain(0..augmented_vectors.len() - options.outer_k);
        }
    }

    Ok(LGMRESResult {
        x,
        iterations: total_iter,
        residual_norm: r_norm,
        converged: false,
    })
}

/// Inner GMRES iteration
fn inner_gmres<F>(
    a: &dyn LinearOperator<F>,
    r0: &[F],
    m: usize,
    _augmented_vectors: &[Vec<F>],
    preconditioner: Option<&dyn LinearOperator<F>>,
) -> SparseResult<(Vec<F>, F, Vec<Vec<F>>)>
where
    F: Float + NumAssign + Sum + Debug + 'static,
{
    let n = r0.len();

    // Initialize

    let mut v = vec![vec![F::zero(); n]; m + 1];
    let r0_norm = r0.iter().map(|&ri| ri * ri).sum::<F>().sqrt();

    if r0_norm < F::epsilon() {
        return Ok((vec![F::zero(); n], F::zero(), vec![]));
    }

    v[0] = r0.iter().map(|&ri| ri / r0_norm).collect();

    let mut h = vec![vec![F::zero(); m]; m + 1];
    let mut s = vec![F::zero(); m + 1];
    let mut c = vec![F::zero(); m + 1];
    let mut beta = vec![F::zero(); m + 2];
    beta[0] = r0_norm;

    // Standard Arnoldi process (augmented vectors not implemented yet)
    let mut k = 0; // Track actual iterations completed
    for j in 0..m {
        // Compute A*v[j] with preconditioning
        let w = if let Some(prec) = preconditioner {
            let av = a.matvec(&v[j])?;
            prec.matvec(&av)?
        } else {
            a.matvec(&v[j])?
        };

        // Gram-Schmidt orthogonalization
        let mut w_orth = w.clone();
        for i in 0..=j {
            let h_ij = w
                .iter()
                .zip(v[i].iter())
                .map(|(&wi, &vi)| wi * vi)
                .sum::<F>();
            h[i][j] = h_ij;
            for idx in 0..n {
                w_orth[idx] -= h_ij * v[i][idx];
            }
        }

        let h_jp1_j = w_orth.iter().map(|&wi| wi * wi).sum::<F>().sqrt();

        if h_jp1_j > F::epsilon() {
            h[j + 1][j] = h_jp1_j;
            v[j + 1] = w_orth.iter().map(|&wi| wi / h_jp1_j).collect();
        } else {
            // Linear dependence detected, early exit
            k = j + 1;
            break;
        }

        // Apply Givens rotations
        for i in 0..j {
            let temp = c[i] * h[i][j] + s[i] * h[i + 1][j];
            h[i + 1][j] = -s[i] * h[i][j] + c[i] * h[i + 1][j];
            h[i][j] = temp;
        }

        // Compute new Givens rotation
        let h_jj = h[j][j];
        let h_jp1_j = h[j + 1][j];
        let rho = (h_jj * h_jj + h_jp1_j * h_jp1_j).sqrt();

        if rho > F::epsilon() {
            c[j] = h_jj / rho;
            s[j] = h_jp1_j / rho;

            h[j][j] = c[j] * h_jj + s[j] * h_jp1_j;
            h[j + 1][j] = F::zero();

            // Update residual
            beta[j + 1] = -s[j] * beta[j];
            beta[j] = c[j] * beta[j];

            k = j + 1; // Update iterations completed

            // Check for early termination
            if beta[j + 1].abs() < F::from(1e-10).unwrap() {
                break;
            }
        } else {
            // Division by zero avoided
            k = j;
            break;
        }
    }

    // If no iterations were performed, return zero solution
    if k == 0 {
        return Ok((vec![F::zero(); n], r0_norm, vec![]));
    }

    // Solve the least squares problem
    let mut y = vec![F::zero(); k];
    for i in (0..k).rev() {
        y[i] = beta[i];
        for j in (i + 1)..k {
            y[i] = y[i] - h[i][j] * y[j];
        }
        if h[i][i].abs() > F::epsilon() {
            y[i] /= h[i][i];
        } else {
            y[i] = F::zero();
        }
    }

    // Compute solution in original space
    let mut x = vec![F::zero(); n];
    for i in 0..k {
        for j in 0..n {
            x[j] += y[i] * v[i][j];
        }
    }

    // Collect Krylov vectors for augmentation
    let v_list: Vec<Vec<F>> = if k > 1 {
        v.into_iter().skip(1).take(k - 1).collect()
    } else {
        vec![]
    };

    Ok((x, beta[k].abs(), v_list))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;
    use crate::linalg::interface::{AsLinearOperator, IdentityOperator};

    #[test]
    fn test_lgmres_identity() {
        // Test with identity matrix
        let a = IdentityOperator::new(3);
        let b = vec![1.0, 2.0, 3.0];
        let options = LGMRESOptions::default();

        let result = lgmres(&a, &b, options).unwrap();

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-10);
        assert!((result.x[1] - 2.0).abs() < 1e-10);
        assert!((result.x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_lgmres_spd_matrix() {
        // Test with a simple SPD matrix
        let data = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let indptr = vec![0, 2, 5, 7];
        let indices = vec![0, 1, 0, 1, 2, 1, 2];
        let matrix = CsrMatrix::from_raw_csr(data, indptr, indices, (3, 3)).unwrap();
        let linear_op = matrix.as_linear_operator();

        let b = vec![1.0, 2.0, 3.0];
        let options = LGMRESOptions::default();

        let result = lgmres(linear_op.as_ref(), &b, options).unwrap();

        assert!(result.converged);
        assert!(result.residual_norm < 1e-8);
    }
}
