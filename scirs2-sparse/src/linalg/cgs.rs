use crate::error::{SparseError, SparseResult};
use crate::linalg::interface::LinearOperator;
use crate::linalg::iterative::{dot, norm2, BiCGOptions, IterationResult};
use num_traits::{Float, NumAssign};
use std::iter::Sum;

/// Options for CGS solver
pub type CGSOptions<F> = BiCGOptions<F>;
pub type CGSResult<F> = IterationResult<F>;

/// Conjugate Gradient Squared solver (CGS)
///
/// Implementation following the algorithm from "Templates for the Solution of Linear Systems"
/// by Barrett et al. This is for non-symmetric linear systems.
pub fn cgs<F>(
    a: &dyn LinearOperator<F>,
    b: &[F],
    options: CGSOptions<F>,
) -> SparseResult<CGSResult<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(SparseError::ValueError(
            "Matrix must be square for CGS solver".to_string(),
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
        return Ok(CGSResult {
            x,
            iterations: 0,
            residual_norm: rnorm,
            converged: true,
            message: "Converged with initial guess".to_string(),
        });
    }

    // Choose arbitrary r̃ (usually r̃ = r)
    let r_tilde = r.clone();

    // Initialize vectors
    let mut u = vec![F::zero(); n];
    let mut p = vec![F::zero(); n];
    let mut q = vec![F::zero(); n];

    let mut rho = F::one();
    let mut iterations = 0;

    // Main CGS iteration
    while iterations < options.max_iter {
        // Compute ρ = (r̃, r)
        let rho_new = dot(&r_tilde, &r);

        // Check for breakdown
        if rho_new.abs() < F::epsilon() * F::from(10).unwrap() {
            return Ok(CGSResult {
                x,
                iterations,
                residual_norm: rnorm,
                converged: false,
                message: "CGS breakdown: rho ≈ 0".to_string(),
            });
        }

        // Compute β = ρ_i / ρ_{i-1}
        let beta = if iterations == 0 {
            F::zero()
        } else {
            rho_new / rho
        };

        // Update u and p
        for i in 0..n {
            u[i] = r[i] + beta * q[i];
            p[i] = u[i] + beta * (q[i] + beta * p[i]);
        }

        // Apply right preconditioner if provided
        let p_prec = if let Some(m) = &options.right_preconditioner {
            m.matvec(&p)?
        } else {
            p.clone()
        };

        // v = A * M^{-1} * p
        let v = a.matvec(&p_prec)?;

        // σ = (r̃, v)
        let sigma = dot(&r_tilde, &v);

        // Check for breakdown
        if sigma.abs() < F::epsilon() * F::from(10).unwrap() {
            return Ok(CGSResult {
                x,
                iterations,
                residual_norm: rnorm,
                converged: false,
                message: "CGS breakdown: sigma ≈ 0".to_string(),
            });
        }

        // α = ρ / σ
        let alpha = rho_new / sigma;

        // Update q
        for i in 0..n {
            q[i] = u[i] - alpha * v[i];
        }

        // Compute u + q
        let u_plus_q: Vec<F> = u.iter().zip(&q).map(|(&ui, &qi)| ui + qi).collect();

        // Apply right preconditioner if provided
        let u_plus_q_prec = if let Some(m) = &options.right_preconditioner {
            m.matvec(&u_plus_q)?
        } else {
            u_plus_q
        };

        // Update x
        for i in 0..n {
            x[i] += alpha * u_plus_q_prec[i];
        }

        // Apply right preconditioner to q
        let q_prec = if let Some(m) = &options.right_preconditioner {
            m.matvec(&q)?
        } else {
            q.clone()
        };

        // Compute A * M^{-1} * q
        let aq = a.matvec(&q_prec)?;

        // Update r
        for i in 0..n {
            r[i] -= alpha * (v[i] + aq[i]);
        }

        rho = rho_new;
        iterations += 1;

        // Check convergence
        rnorm = norm2(&r);
        if rnorm <= tolerance {
            break;
        }
    }

    Ok(CGSResult {
        x,
        iterations,
        residual_norm: rnorm,
        converged: rnorm <= tolerance,
        message: if rnorm <= tolerance {
            "Converged".to_string()
        } else {
            "Maximum iterations reached".to_string()
        },
    })
}
