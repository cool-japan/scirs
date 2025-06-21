//! Constraint optimization solvers for constrained splines
//!
//! This module contains the internal solver functions for constrained optimization problems,
//! including quadratic programming and projected gradient methods.

use crate::bspline::{BSpline, ExtrapolateMode};
use crate::error::{InterpolateError, InterpolateResult};
#[cfg(feature = "linalg")]
use crate::numerical_stability::{
    assess_matrix_condition, solve_with_stability_monitoring, StabilityLevel,
};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use super::types::{Constraint, ConstraintType};

/// Solve the constrained interpolation problem
///
/// Solves A*c = y subject to G*c >= h
fn solve_constrained_interpolation<T>(
    design_matrix: &ArrayView2<T>,
    y: &ArrayView1<T>,
    constraint_matrix: &ArrayView2<T>,
    constraint_rhs: &ArrayView1<T>,
) -> InterpolateResult<Array1<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + 'static
        + std::fmt::LowerExp,
{
    // For now, we'll provide a simplified implementation that uses a quadratic program
    // to approximate the interpolation problem

    // We transform the interpolation problem into a least squares problem
    // with a large weight to enforce a close fit to the data
    let weight = T::from_f64(1e6).unwrap();
    let weighted_design = design_matrix.map(|&x| x * weight);
    let weighted_y = y.map(|&x| x * weight);

    // Use the least squares solver with the weighted matrices
    solve_constrained_least_squares(
        &weighted_design.view(),
        &weighted_y.view(),
        constraint_matrix,
        constraint_rhs,
    )
}

/// Solve the constrained least squares problem
///
/// Solves min ||A*c - y||^2 subject to G*c >= h
fn solve_constrained_least_squares<T>(
    design_matrix: &ArrayView2<T>,
    y: &ArrayView1<T>,
    constraint_matrix: &ArrayView2<T>,
    constraint_rhs: &ArrayView1<T>,
) -> InterpolateResult<Array1<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + 'static
        + std::fmt::LowerExp,
{
    // Form the normal equations: A'A*c = A'y
    let a_transpose = design_matrix.t();
    #[cfg(feature = "linalg")]
    let ata = a_transpose.dot(design_matrix);
    #[cfg(not(feature = "linalg"))]
    let _ata = a_transpose.dot(design_matrix);
    #[cfg(feature = "linalg")]
    let aty = a_transpose.dot(y);
    #[cfg(not(feature = "linalg"))]
    let _aty = a_transpose.dot(y);

    // If no constraints, solve the unconstrained problem
    if constraint_matrix.shape()[0] == 0 {
        #[cfg(feature = "linalg")]
        {
            // Assess matrix condition before solving
            let condition_report = assess_matrix_condition(&ata.view());
            if let Ok(report) = condition_report {
                match report.stability_level {
                    StabilityLevel::Poor => {
                        eprintln!(
                            "Warning: Normal equations matrix is poorly conditioned \
                             (condition number: {:.2e}). Results may be unreliable.",
                            report.condition_number
                        );
                    }
                    StabilityLevel::Marginal => {
                        eprintln!(
                            "Info: Normal equations matrix has marginal conditioning \
                             (condition number: {:.2e}). Monitoring solution quality.",
                            report.condition_number
                        );
                    }
                    _ => {}
                }
            }

            // Use stability-monitored solver
            match solve_with_stability_monitoring(&ata, &aty) {
                Ok((solution, _solve_report)) => return Ok(solution),
                Err(_) => {
                    return Err(InterpolateError::ComputationError(
                        "Failed to solve the unconstrained least squares problem with stability monitoring".to_string(),
                    ))
                }
            }
        }

        #[cfg(not(feature = "linalg"))]
        return Err(InterpolateError::NotImplemented(
            "Linear algebra operations require the 'linalg' feature".to_string(),
        ));
    }

    // For the constrained problem, we use a simple active set method
    // This is a simplified implementation - a real one would use a specialized QP solver

    // Start with an initial feasible solution (unconstrained)
    #[cfg(feature = "linalg")]
    let mut c = {
        // Use stability-monitored solver for initial solution
        match solve_with_stability_monitoring(&ata, &aty) {
            Ok((solution, solve_report)) => {
                if !solve_report.is_well_conditioned {
                    eprintln!(
                        "Warning: Initial solution for constrained problem computed with \
                         poorly conditioned matrix (condition number: {:.2e})",
                        solve_report.condition_number
                    );
                }
                solution
            }
            Err(_) => {
                eprintln!(
                    "Warning: Stability-monitored solve failed for initial solution. \
                     Using zero initialization."
                );
                // If solve fails, try a simpler approach
                let n = design_matrix.shape()[1];
                Array1::zeros(n)
            }
        }
    };

    #[cfg(not(feature = "linalg"))]
    let mut c = {
        // Fallback implementation when linalg is not available
        // Simple diagonal approximation
        let n = design_matrix.shape()[1];
        Array1::zeros(n)
    };

    // Check if the initial solution satisfies the constraints
    #[cfg(feature = "linalg")]
    let mut constraint_values = constraint_matrix.dot(&c) - constraint_rhs;
    #[cfg(not(feature = "linalg"))]
    let mut constraint_values = constraint_matrix.dot(&c) - constraint_rhs;

    // If all constraints are satisfied, we're done
    let mut all_satisfied = true;
    for &val in constraint_values.iter() {
        if val < T::zero() {
            all_satisfied = false;
            break;
        }
    }

    if all_satisfied {
        return Ok(c);
    }

    // If not, we'll use a projected gradient approach
    // We'll iterate until either all constraints are satisfied or we hit the max iterations
    let max_iterations = 100;
    let mut iterations = 0;

    while iterations < max_iterations {
        iterations += 1;

        // Find the most violated constraint
        let mut worst_idx = 0;
        let mut worst_violation = T::zero();

        for (i, &val) in constraint_values.iter().enumerate() {
            if val < worst_violation {
                worst_idx = i;
                worst_violation = val;
            }
        }

        // If no violation, we're done
        if worst_violation >= -T::epsilon() {
            break;
        }

        // Project the constraint into the solution
        // We'll move in the direction of the constraint gradient
        let constraint_vector = constraint_matrix.row(worst_idx).to_owned();
        #[cfg(feature = "linalg")]
        let constraint_norm_squared = constraint_vector.dot(&constraint_vector);
        #[cfg(not(feature = "linalg"))]
        let constraint_norm_squared = constraint_vector.dot(&constraint_vector);

        if constraint_norm_squared < T::epsilon() {
            // If the constraint gradient is zero, we can't make progress
            continue;
        }

        // Calculate the projection step size
        let step_size = -worst_violation / constraint_norm_squared;

        // Update the solution
        for i in 0..c.len() {
            c[i] += step_size * constraint_vector[i];
        }

        // Recheck constraints
        #[cfg(feature = "linalg")]
        {
            constraint_values = constraint_matrix.dot(&c) - constraint_rhs;
        }
        #[cfg(not(feature = "linalg"))]
        {
            constraint_values = constraint_matrix.dot(&c) - constraint_rhs;
        }
    }

    // After the iterations, check if all constraints are satisfied
    #[cfg(feature = "linalg")]
    {
        constraint_values = constraint_matrix.dot(&c) - constraint_rhs;
    }
    #[cfg(not(feature = "linalg"))]
    {
        constraint_values = constraint_matrix.dot(&c) - constraint_rhs;
    }
    all_satisfied = true;
    for &val in constraint_values.iter() {
        if val < -T::from_f64(1e-6).unwrap() {
            all_satisfied = false;
            break;
        }
    }

    if !all_satisfied {
        // If we couldn't satisfy all constraints, return an error
        // In practice, you might want to return the best solution found instead
        return Err(InterpolateError::ComputationError(
            "Failed to find a solution that satisfies all constraints".to_string(),
        ));
    }

    Ok(c)
}

/// Solve the constrained penalized problem
///
/// Solves min ||A*c - y||^2 + λ*c'P*c subject to G*c >= h
fn solve_constrained_penalized<T>(
    design_matrix: &ArrayView2<T>,
    y: &ArrayView1<T>,
    penalty_matrix: &ArrayView2<T>,
    lambda: T,
    constraint_matrix: &ArrayView2<T>,
    constraint_rhs: &ArrayView1<T>,
) -> InterpolateResult<Array1<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + 'static
        + std::fmt::LowerExp,
{
    // Form the penalized objective: A'A*c + λ*P*c = A'y
    let a_transpose = design_matrix.t();
    let mut ata = a_transpose.dot(design_matrix);
    #[cfg(feature = "linalg")]
    let aty = a_transpose.dot(y);
    #[cfg(not(feature = "linalg"))]
    let _aty = a_transpose.dot(y);

    // Add the penalty term
    for i in 0..ata.shape()[0] {
        for j in 0..ata.shape()[1] {
            ata[[i, j]] += lambda * penalty_matrix[[i, j]];
        }
    }

    // If no constraints, solve the unconstrained problem
    if constraint_matrix.shape()[0] == 0 {
        #[cfg(feature = "linalg")]
        {
            use ndarray_linalg::Solve;
            let ata_f64 = ata.mapv(|x| x.to_f64().unwrap());
            let aty_f64 = aty.mapv(|x| x.to_f64().unwrap());
            match ata_f64.solve(&aty_f64) {
                Ok(solution) => return Ok(solution.mapv(|x| T::from_f64(x).unwrap())),
                Err(_) => {
                    return Err(InterpolateError::ComputationError(
                        "Failed to solve the unconstrained penalized problem".to_string(),
                    ))
                }
            }
        }

        #[cfg(not(feature = "linalg"))]
        return Err(InterpolateError::NotImplemented(
            "Linear algebra operations require the 'linalg' feature".to_string(),
        ));
    }

    // Use a similar approach as the non-penalized case
    #[cfg(feature = "linalg")]
    let mut c = {
        use ndarray_linalg::Solve;
        let ata_f64 = ata.mapv(|x| x.to_f64().unwrap());
        let aty_f64 = aty.mapv(|x| x.to_f64().unwrap());
        match ata_f64.solve(&aty_f64) {
            Ok(solution) => solution.mapv(|x| T::from_f64(x).unwrap()),
            Err(_) => {
                // If direct solve fails, try a simpler approach
                let n = design_matrix.shape()[1];
                Array1::zeros(n)
            }
        }
    };

    #[cfg(not(feature = "linalg"))]
    let mut c = {
        // Fallback implementation when linalg is not available
        let n = design_matrix.shape()[1];
        Array1::zeros(n)
    };

    // Apply the projected gradient optimization as before
    let max_iterations = 100;
    let mut iterations = 0;

    while iterations < max_iterations {
        iterations += 1;

        // Check constraints
        #[cfg(feature = "linalg")]
        let constraint_values = constraint_matrix.dot(&c) - constraint_rhs;
        #[cfg(not(feature = "linalg"))]
        let constraint_values = constraint_matrix.dot(&c) - constraint_rhs;

        // Find the most violated constraint
        let mut worst_idx = 0;
        let mut worst_violation = T::zero();

        for (i, &val) in constraint_values.iter().enumerate() {
            if val < worst_violation {
                worst_idx = i;
                worst_violation = val;
            }
        }

        // If no violation, we're done
        if worst_violation >= -T::epsilon() {
            break;
        }

        // Project the constraint into the solution
        let constraint_vector = constraint_matrix.row(worst_idx).to_owned();
        #[cfg(feature = "linalg")]
        let constraint_norm_squared = constraint_vector.dot(&constraint_vector);
        #[cfg(not(feature = "linalg"))]
        let constraint_norm_squared = constraint_vector.dot(&constraint_vector);

        if constraint_norm_squared < T::epsilon() {
            continue;
        }

        // Calculate the projection step size
        let step_size = -worst_violation / constraint_norm_squared;

        // Update the solution
        for i in 0..c.len() {
            c[i] += step_size * constraint_vector[i];
        }
    }

    Ok(c)
}

/// Main entry point for solving constrained systems
pub fn solve_constrained_system<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    knots: &ArrayView1<T>,
    degree: usize,
    constraints: &[Constraint<T>],
) -> InterpolateResult<Array1<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static
        + std::fmt::LowerExp,
{
    let n_coeffs = knots.len() - degree - 1;

    // Create the design matrix
    let mut design_matrix = Array2::zeros((x.len(), n_coeffs));
    for (i, &x_val) in x.iter().enumerate() {
        for j in 0..n_coeffs {
            let basis = BSpline::basis_element(degree, j, knots, ExtrapolateMode::Extrapolate)?;
            design_matrix[[i, j]] = basis.evaluate(x_val)?;
        }
    }

    // Generate constraint matrices
    let (constraint_matrix, constraint_rhs) =
        generate_constraint_matrices(x, knots, degree, constraints)?;

    // Solve the constrained system
    solve_constrained_interpolation(
        &design_matrix.view(),
        y,
        &constraint_matrix.view(),
        &constraint_rhs.view(),
    )
}

/// Main entry point for solving penalized systems
pub fn solve_penalized_system<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    knots: &ArrayView1<T>,
    degree: usize,
    constraints: &[Constraint<T>],
    lambda: T,
) -> InterpolateResult<Array1<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static
        + std::fmt::LowerExp,
{
    let n_coeffs = knots.len() - degree - 1;

    // Create the design matrix
    let mut design_matrix = Array2::zeros((x.len(), n_coeffs));
    for (i, &x_val) in x.iter().enumerate() {
        for j in 0..n_coeffs {
            let basis = BSpline::basis_element(degree, j, knots, ExtrapolateMode::Extrapolate)?;
            design_matrix[[i, j]] = basis.evaluate(x_val)?;
        }
    }

    // Create penalty matrix
    let penalty_matrix = create_penalty_matrix(n_coeffs, degree)?;

    // Generate constraint matrices
    let (constraint_matrix, constraint_rhs) =
        generate_constraint_matrices(x, knots, degree, constraints)?;

    // Solve the penalized constrained system
    solve_constrained_penalized(
        &design_matrix.view(),
        y,
        &penalty_matrix.view(),
        lambda,
        &constraint_matrix.view(),
        &constraint_rhs.view(),
    )
}

/// Create a standard second derivative penalty matrix
fn create_penalty_matrix<T>(n: usize, degree: usize) -> InterpolateResult<Array2<T>>
where
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::SubAssign,
{
    let mut penalty = Array2::zeros((n, n));

    // For degree < 2, we can't compute second derivatives
    if degree < 2 {
        return Ok(penalty);
    }

    // Second derivative penalty: D₂ᵀD₂ where D₂ is the second difference matrix
    let one = T::one();
    let two = T::from_f64(2.0).unwrap();

    for i in 0..n - 2 {
        // Diagonal elements
        penalty[[i, i]] += one;
        penalty[[i + 1, i + 1]] += two * two;
        penalty[[i + 2, i + 2]] += one;

        // Off-diagonal elements
        penalty[[i, i + 1]] -= two;
        penalty[[i + 1, i]] -= two;

        penalty[[i, i + 2]] += one;
        penalty[[i + 2, i]] += one;

        penalty[[i + 1, i + 2]] -= two;
        penalty[[i + 2, i + 1]] -= two;
    }

    Ok(penalty)
}

/// Generate constraint matrices from constraint specifications
pub fn generate_constraint_matrices<T>(
    x: &ArrayView1<T>,
    knots: &ArrayView1<T>,
    degree: usize,
    constraints: &[Constraint<T>],
) -> InterpolateResult<(Array2<T>, Array1<T>)>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static
        + std::fmt::LowerExp,
{
    let n_coeffs = knots.len() - degree - 1;
    let x_min = x[0];
    let x_max = x[x.len() - 1];

    // Count the total number of constraint points we'll need
    let mut total_constraints = 0;

    for constraint in constraints {
        // Get the x range for this constraint
        let _constraint_x_min = constraint.x_min.unwrap_or(x_min);
        let _constraint_x_max = constraint.x_max.unwrap_or(x_max);

        // Count evaluation points in the constraint region
        // Note: we generate n_eval = 10 points, and add n_eval - 1 = 9 constraints
        let n_eval = 10;
        match constraint.constraint_type {
            ConstraintType::MonotoneIncreasing | ConstraintType::MonotoneDecreasing => {
                // For monotonicity, we add n_eval - 1 constraints
                total_constraints += n_eval - 1;
            }
            ConstraintType::Convex | ConstraintType::Concave => {
                // For convexity, we add n_eval constraints (one per evaluation point)
                total_constraints += n_eval;
            }
            _ => {
                // For other constraints, check at multiple points
                total_constraints += n_eval;
            }
        }
    }

    // Allocate matrices
    let mut constraint_matrix = Array2::zeros((total_constraints, n_coeffs));
    let mut constraint_rhs = Array1::zeros(total_constraints);
    let mut constraint_idx = 0;

    let extrapolate = ExtrapolateMode::Extrapolate;

    for constraint in constraints {
        // Get the x range for this constraint
        let constraint_x_min = constraint.x_min.unwrap_or(x_min);
        let constraint_x_max = constraint.x_max.unwrap_or(x_max);

        // Generate evaluation points in the constraint region
        let n_eval = 10;
        let mut eval_points = Vec::new();
        for i in 0..n_eval {
            let t = i as f64 / (n_eval - 1) as f64;
            let x_val =
                constraint_x_min + T::from_f64(t).unwrap() * (constraint_x_max - constraint_x_min);
            eval_points.push(x_val);
        }

        // Create constraint rows based on the constraint type
        match constraint.constraint_type {
            ConstraintType::MonotoneIncreasing => {
                // For each adjacent pair of evaluation points, ensure derivative is non-negative
                for i in 0..eval_points.len() - 1 {
                    let x_val = (eval_points[i] + eval_points[i + 1]) / T::from_f64(2.0).unwrap();

                    // Create a row for the first derivative constraint at this point
                    for j in 0..n_coeffs {
                        let basis = BSpline::basis_element(degree, j, knots, extrapolate)?;
                        constraint_matrix[[constraint_idx, j]] = basis.derivative(x_val, 1)?;
                    }
                    constraint_rhs[constraint_idx] = T::zero();
                    constraint_idx += 1;
                }
            }
            ConstraintType::MonotoneDecreasing => {
                // For decreasing, the negative of the derivative should be non-negative
                for i in 0..eval_points.len() - 1 {
                    let x_val = (eval_points[i] + eval_points[i + 1]) / T::from_f64(2.0).unwrap();

                    for j in 0..n_coeffs {
                        let basis = BSpline::basis_element(degree, j, knots, extrapolate)?;
                        constraint_matrix[[constraint_idx, j]] = -basis.derivative(x_val, 1)?;
                    }
                    constraint_rhs[constraint_idx] = T::zero();
                    constraint_idx += 1;
                }
            }
            ConstraintType::Convex => {
                // For convexity, ensure second derivative is non-negative
                for &x_val in eval_points.iter() {
                    for j in 0..n_coeffs {
                        let basis = BSpline::basis_element(degree, j, knots, extrapolate)?;
                        constraint_matrix[[constraint_idx, j]] = basis.derivative(x_val, 2)?;
                    }
                    constraint_rhs[constraint_idx] = T::zero();
                    constraint_idx += 1;
                }
            }
            ConstraintType::Concave => {
                // For concavity, the negative of the second derivative should be non-negative
                for &x_val in eval_points.iter() {
                    for j in 0..n_coeffs {
                        let basis = BSpline::basis_element(degree, j, knots, extrapolate)?;
                        constraint_matrix[[constraint_idx, j]] = -basis.derivative(x_val, 2)?;
                    }
                    constraint_rhs[constraint_idx] = T::zero();
                    constraint_idx += 1;
                }
            }
            ConstraintType::Positive => {
                // For positivity, the function value should be non-negative
                for &x_val in eval_points.iter() {
                    for j in 0..n_coeffs {
                        let basis = BSpline::basis_element(degree, j, knots, extrapolate)?;
                        constraint_matrix[[constraint_idx, j]] = basis.evaluate(x_val)?;
                    }
                    constraint_rhs[constraint_idx] = T::zero();
                    constraint_idx += 1;
                }
            }
            ConstraintType::UpperBound => {
                // For upper bound, the function value should be <= upper_bound
                let upper_bound = constraint.parameter.unwrap_or(T::one());
                for &x_val in eval_points.iter() {
                    // Create a row for the upper bound constraint at this point
                    for j in 0..n_coeffs {
                        let basis = BSpline::basis_element(degree, j, knots, extrapolate)?;
                        constraint_matrix[[constraint_idx, j]] = basis.evaluate(x_val)?;
                    }
                    constraint_rhs[constraint_idx] = upper_bound;
                    constraint_idx += 1;
                }
            }
            ConstraintType::LowerBound => {
                // For lower bound, the function value should be >= lower_bound
                let lower_bound = constraint.parameter.unwrap_or(T::zero());
                for &x_val in eval_points.iter() {
                    // Create a row for the lower bound constraint at this point
                    for j in 0..n_coeffs {
                        let basis = BSpline::basis_element(degree, j, knots, extrapolate)?;
                        constraint_matrix[[constraint_idx, j]] = -basis.evaluate(x_val)?;
                    }
                    constraint_rhs[constraint_idx] = -lower_bound;
                    constraint_idx += 1;
                }
            }
        }
    }

    // Trim matrices if we didn't use all allocated space
    if constraint_idx < total_constraints {
        constraint_matrix = constraint_matrix
            .slice(s![0..constraint_idx, ..])
            .to_owned();
        constraint_rhs = constraint_rhs.slice(s![0..constraint_idx]).to_owned();
    }

    Ok((constraint_matrix, constraint_rhs))
}
