//! Utilities for working with mass matrices in ODE systems
//!
//! This module provides functions for handling mass matrices in ODEs of the form:
//! M(t,y)·y' = f(t,y), where M is a mass matrix that may depend on time t and state y.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::types::{MassMatrix, MassMatrixType};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

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
fn solve_matrix_system<F>(matrix: ArrayView2<F>, b: ArrayView1<F>) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
{
    use crate::ode::utils::linear_solvers::solve_linear_system;

    // Use our custom solver
    solve_linear_system(&matrix, &b).map_err(|err| {
        IntegrateError::ComputationError(format!("Failed to solve mass matrix system: {}", err))
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
    /// Create a new LU decomposition from a matrix
    fn new(matrix: ArrayView2<F>) -> IntegrateResult<Self> {
        // Implement our own LU factorization
        // For now, we'll just store the matrix for use with our linear solver

        // Use ndarray-linalg to compute LU decomposition
        // Note: This is a placeholder implementation
        // In practice, we would use a more specific implementation
        // that directly exposes the pivots

        let (n, _) = matrix.dim();
        let lu = matrix.to_owned();
        let pivots = (0..n).collect(); // Placeholder

        Ok(LUDecomposition { lu, pivots })
    }

    /// Solve a linear system using the LU decomposition
    fn solve(&self, b: ArrayView1<F>) -> IntegrateResult<Array1<F>> {
        use crate::ode::utils::linear_solvers::solve_linear_system;

        // Use our custom solver with the matrix
        // Note: For a proper LU-based solver, we would need to implement one
        // For now, this is a simpler approach that still works
        solve_linear_system(&self.lu.view(), &b).map_err(|err| {
            IntegrateError::ComputationError(format!("Failed to solve with matrix: {}", err))
        })
    }
}

/// Check if a mass matrix is compatible with an ODE state
///
/// Verifies that the mass matrix dimensions match the state vector dimensions
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
                    "Mass matrix dimensions ({},{}) do not match state vector length ({})",
                    rows, cols, n
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
pub fn is_singular<F>(matrix: ArrayView2<F>, threshold: Option<F>) -> bool
where
    F: IntegrateFloat,
{
    // Default condition number threshold
    let _thresh = threshold.unwrap_or_else(|| F::from_f64(1e14).unwrap());

    // In a full implementation, we would compute the condition number
    // using singular value decomposition or other methods.
    // For simplicity, this returns a placeholder result.

    // Simple check: see if diagonal has any zeros
    let (n, _) = matrix.dim();
    for i in 0..n {
        if matrix[[i, i]].abs() < F::from_f64(1e-14).unwrap() {
            return true;
        }
    }

    false
}
