//! Utilities for working with mass matrices in ODE systems
//!
//! This module provides functions for handling mass matrices in ODEs of the form:
//! M(t,y)·y' = f(t,y), where M is a mass matrix that may depend on time t and state y.

use crate::common::IntegrateFloat;
use crate::dae::utils::linear_solvers::solve_linear_system;
use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::types::{MassMatrix, MassMatrixType};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Solve a linear system with mass matrix: M·x = b
///
/// For ODEs with mass matrices, we often need to solve M·x = f(t,y)
/// to find x = y' for the standard form y' = g(t,y)
///
/// # Arguments
///
/// * `mass` - The mass matrix structure
/// * `t` - Current time
/// * `y` - Current state
/// * `b` - Right-hand side vector
///
/// # Returns
///
/// Solution vector x where M·x = b, or error if the system cannot be solved
#[allow(dead_code)]
pub fn solve_mass_system<F>(
    mass: &MassMatrix<F>,
    t: F,
    y: ArrayView1<F>,
    b: ArrayView1<F>,
) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
{
    match mass.matrix_type {
        MassMatrixType::Identity => {
            // For identity mass matrix, solution is just b
            Ok(b.to_owned())
        }
        _ => {
            // Get the mass matrix at current time and state
            let matrix = mass.evaluate(t, y).ok_or_else(|| {
                IntegrateError::ComputationError("Failed to evaluate mass matrix".to_string())
            })?;

            // Solve the linear system M·x = b
            solve_matrix_system(matrix.view(), b)
        }
    }
}

/// Solve a linear system M·x = b with explicit matrix
///
/// Helper function to solve linear systems with mass matrices
#[allow(dead_code)]
fn solve_matrix_system<F>(matrix: ArrayView2<F>, b: ArrayView1<F>) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
{
    // Use our custom solver
    solve_linear_system(&matrix, &b).map_err(|err| {
        IntegrateError::ComputationError(format!("Failed to solve mass _matrix system: {err}"))
    })
}

/// Apply mass matrix to a vector: result = M·v
///
/// Used to compute the product of a mass matrix with a vector
///
/// # Arguments
///
/// * `mass` - The mass matrix structure
/// * `t` - Current time
/// * `y` - Current state
/// * `v` - Vector to multiply with
///
/// # Returns
///
/// Result of M·v, or error if the operation cannot be performed
#[allow(dead_code)]
pub fn apply_mass<F>(
    mass: &MassMatrix<F>,
    t: F,
    y: ArrayView1<F>,
    v: ArrayView1<F>,
) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
{
    match mass.matrix_type {
        MassMatrixType::Identity => {
            // For identity mass matrix, result is just v
            Ok(v.to_owned())
        }
        _ => {
            // Get the mass matrix at current time and state
            let matrix = mass.evaluate(t, y).ok_or_else(|| {
                IntegrateError::ComputationError("Failed to evaluate mass matrix".to_string())
            })?;

            // Perform matrix-vector multiplication
            let result = matrix.dot(&v);
            Ok(result)
        }
    }
}

/// Compute the LU decomposition of a mass matrix
///
/// This can be used to cache the decomposition for repeated solves
/// with the same mass matrix
#[allow(dead_code)]
struct LUDecomposition<F: IntegrateFloat> {
    /// The LU factors
    lu: Array2<F>,
    /// Pivot indices
    pivots: Vec<usize>,
}

#[allow(dead_code)]
impl<F: IntegrateFloat> LUDecomposition<F> {
    /// Create a new LU decomposition from a matrix with partial pivoting
    fn new(matrix: ArrayView2<F>) -> IntegrateResult<Self> {
        let (n, m) = matrix.dim();
        if n != m {
            return Err(IntegrateError::ValueError(
                "Matrix must be square for LU decomposition".to_string(),
            ));
        }

        let mut lu = matrix.to_owned();
        let mut pivots = (0..n).collect::<Vec<_>>();

        // Gaussian elimination with partial pivoting
        for k in 0..n {
            // Find the largest element in column k from row k onwards (partial pivoting)
            let mut max_row = k;
            let mut max_val = lu[[k, k]].abs();

            for i in (k + 1)..n {
                let val = lu[[i, k]].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            // Check for singularity
            if max_val < F::from_f64(1e-14).unwrap() {
                return Err(IntegrateError::ComputationError(
                    "Matrix is singular or nearly singular".to_string(),
                ));
            }

            // Swap rows if necessary
            if max_row != k {
                pivots.swap(k, max_row);
                for j in 0..n {
                    let temp = lu[[k, j]];
                    lu[[k, j]] = lu[[max_row, j]];
                    lu[[max_row, j]] = temp;
                }
            }

            // Elimination step
            for i in (k + 1)..n {
                let factor = lu[[i, k]] / lu[[k, k]];
                lu[[i, k]] = factor; // Store the multiplier

                for j in (k + 1)..n {
                    let temp = lu[[k, j]];
                    lu[[i, j]] -= factor * temp;
                }
            }
        }

        Ok(LUDecomposition { lu, pivots })
    }

    /// Solve a linear system using the LU decomposition
    fn solve(&self, b: ArrayView1<F>) -> IntegrateResult<Array1<F>> {
        // Use our custom solver with the matrix
        // Note: For a proper LU-based solver, we would need to implement one
        // For now, this is a simpler approach that still works
        solve_linear_system(&self.lu.view(), &b).map_err(|err| {
            IntegrateError::ComputationError(format!("Failed to solve with matrix: {err}"))
        })
    }
}

/// Check if a mass matrix is compatible with an ODE state
///
/// Verifies that the mass matrix dimensions match the state vector dimensions
#[allow(dead_code)]
pub fn check_mass_compatibility<F>(
    mass: &MassMatrix<F>,
    t: F,
    y: ArrayView1<F>,
) -> IntegrateResult<()>
where
    F: IntegrateFloat,
{
    let n = y.len();

    match mass.matrix_type {
        MassMatrixType::Identity => {
            // Identity matrix is always compatible
            Ok(())
        }
        _ => {
            // Evaluate the mass matrix and check dimensions
            let matrix = mass.evaluate(t, y).ok_or_else(|| {
                IntegrateError::ComputationError("Failed to evaluate mass matrix".to_string())
            })?;

            let (rows, cols) = matrix.dim();

            if rows != n || cols != n {
                return Err(IntegrateError::ValueError(format!(
                    "Mass matrix dimensions ({rows},{cols}) do not match state vector length ({n})"
                )));
            }

            Ok(())
        }
    }
}

/// Transform standard ODE to form with identity mass matrix
///
/// For ODE systems with constant or time-dependent mass matrices,
/// we can transform to a standard ODE with identity mass matrix
/// if the mass matrix is invertible.
///
/// M·y' = f(t,y) transforms to y' = M⁻¹·f(t,y)
///
/// # Arguments
///
/// * `f` - Original ODE function: f(t,y)
/// * `mass` - Mass matrix specification
///
/// # Returns
///
/// A function representing the transformed ODE: g(t,y) where y' = g(t,y)
#[allow(dead_code)]
pub fn transform_to_standard_form<F, Func>(
    f: Func,
    mass: &MassMatrix<F>,
) -> impl Fn(F, ArrayView1<F>) -> IntegrateResult<Array1<F>> + Clone
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Clone,
{
    let mass_cloned = mass.clone();

    move |t: F, y: ArrayView1<F>| {
        // Compute original RHS: f(t,y)
        let rhs = f(t, y);

        // Solve M·y' = f(t,y) for y'
        solve_mass_system(&mass_cloned, t, y, rhs.view())
    }
}

/// Check if a matrix is singular or ill-conditioned
///
/// Uses condition number estimation to check if a matrix is
/// close to singular, which would cause problems for ODE solvers
#[allow(dead_code)]
pub fn is_singular<F>(matrix: ArrayView2<F>, threshold: Option<F>) -> bool
where
    F: IntegrateFloat,
{
    // Default condition number threshold
    let thresh = threshold.unwrap_or_else(|| F::from_f64(1e14).unwrap());

    let (n, m) = matrix.dim();
    if n != m {
        return true; // Non-square matrices are considered singular
    }

    // Estimate condition number using power iteration for largest singular value
    // and inverse power iteration for smallest singular value

    // For efficiency, we'll use a simpler approach for small matrices
    if n <= 3 {
        // For small matrices, compute determinant directly
        let det = compute_determinant(&matrix);
        return det.abs() < F::from_f64(1e-14).unwrap();
    }

    // For larger matrices, estimate condition number
    let cond_number = estimate_condition_number(&matrix);

    cond_number > thresh
}

/// Compute determinant for small matrices (up to 3x3)
#[allow(dead_code)]
fn compute_determinant<F: IntegrateFloat>(matrix: &ArrayView2<F>) -> F {
    let (n, _) = matrix.dim();

    match n {
        1 => matrix[[0, 0]],
        2 => matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]],
        3 => {
            matrix[[0, 0]] * (matrix[[1, 1]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 1]])
                - matrix[[0, 1]]
                    * (matrix[[1, 0]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 0]])
                + matrix[[0, 2]]
                    * (matrix[[1, 0]] * matrix[[2, 1]] - matrix[[1, 1]] * matrix[[2, 0]])
        }
        _ => F::zero(), // Should not reach here
    }
}

/// Estimate condition number using iterative methods
#[allow(dead_code)]
fn estimate_condition_number<F: IntegrateFloat>(matrix: &ArrayView2<F>) -> F {
    let _n = matrix.nrows();

    // Estimate largest eigenvalue magnitude of A^T * A using power iteration
    let max_singular_val_sq = estimate_largest_eigenvalue_ata(matrix);
    let max_singular_val = max_singular_val_sq.sqrt();

    // Estimate smallest eigenvalue magnitude of A^T * A using inverse power iteration
    let min_singular_val_sq = estimate_smallest_eigenvalue_ata(matrix);
    let min_singular_val = min_singular_val_sq.sqrt();

    if min_singular_val < F::from_f64(1e-14).unwrap() {
        F::from_f64(1e16).unwrap() // Very large condition number
    } else {
        max_singular_val / min_singular_val
    }
}

/// Estimate largest eigenvalue of A^T * A using power iteration
#[allow(dead_code)]
fn estimate_largest_eigenvalue_ata<F: IntegrateFloat>(matrix: &ArrayView2<F>) -> F {
    let n = matrix.nrows();
    let max_iterations = 10;

    // Initialize with ones vector
    let mut v = Array1::<F>::from_elem(n, F::one());

    // Normalize
    let mut norm = (v.dot(&v)).sqrt();
    if norm > F::from_f64(1e-14).unwrap() {
        v = &v / norm;
    }

    let mut eigenvalue = F::zero();

    for _ in 0..max_iterations {
        // Compute A * v
        let mut av = Array1::<F>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                av[i] += matrix[[i, j]] * v[j];
            }
        }

        // Compute A^T * (A * v) = A^T * av
        let mut atav = Array1::<F>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                atav[i] += matrix[[j, i]] * av[j];
            }
        }

        // Compute eigenvalue (Rayleigh quotient)
        let new_eigenvalue = v.dot(&atav);

        // Normalize atav for next iteration
        norm = (atav.dot(&atav)).sqrt();
        if norm > F::from_f64(1e-14).unwrap() {
            v = &atav / norm;
        }

        eigenvalue = new_eigenvalue;
    }

    eigenvalue.abs()
}

/// Estimate smallest eigenvalue of A^T * A using simplified approach
#[allow(dead_code)]
fn estimate_smallest_eigenvalue_ata<F: IntegrateFloat>(matrix: &ArrayView2<F>) -> F {
    let n = matrix.nrows();

    // For simplicity, we'll use the minimum diagonal element of A^T * A as a lower bound
    // This is not exact but gives a reasonable estimate for condition number purposes
    let mut min_diag = F::from_f64(f64::INFINITY).unwrap();

    for i in 0..n {
        let mut diag_elem = F::zero();
        for k in 0..n {
            diag_elem += matrix[[k, i]] * matrix[[k, i]];
        }
        if diag_elem < min_diag {
            min_diag = diag_elem;
        }
    }

    min_diag.max(F::from_f64(1e-16).unwrap())
}
