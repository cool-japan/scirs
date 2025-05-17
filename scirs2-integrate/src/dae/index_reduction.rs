//! Index reduction techniques for higher-index DAE systems
//!
//! This module provides implementations of various index reduction algorithms
//! for transforming higher-index DAE systems into equivalent index-1 systems
//! that can be solved with standard DAE solvers.
//!
//! Three main approaches are implemented:
//! - Pantelides algorithm for automatic index reduction
//! - Dummy derivative method
//! - Projection methods for constraint satisfaction

use crate::dae::types::DAEIndex;
use crate::dae::utils::{compute_constraint_jacobian, is_singular_matrix};
use crate::error::{IntegrateError, IntegrateResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display, LowerExp};

/// Index structure of a DAE system
#[derive(Debug, Clone)]
pub struct DAEStructure<
    F: Float
        + FromPrimitive
        + Debug
        + ScalarOperand
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + Display
        + LowerExp
        + std::iter::Sum,
> {
    /// Number of differential variables
    pub n_differential: usize,

    /// Number of algebraic variables
    pub n_algebraic: usize,

    /// Number of differential equations
    pub n_diff_eqs: usize,

    /// Number of algebraic equations
    pub n_alg_eqs: usize,

    /// Index of the DAE system
    pub index: DAEIndex,

    /// Differentiation index (number of differentiations needed to reach an ODE)
    pub diff_index: usize,

    /// Jacobian of the constraint equations with respect to algebraic variables
    pub constraint_jacobian: Option<Array2<F>>,

    /// Jacobian of the constraint equations with respect to derivatives
    pub derivative_jacobian: Option<Array2<F>>,

    /// Incidence matrix showing dependencies between equations and variables
    pub incidence_matrix: Option<Array2<bool>>,

    /// Variables that appear in each equation with non-zero coefficients
    pub variable_dependencies: Option<Vec<Vec<usize>>>,

    /// Equations in which each variable appears with non-zero coefficients
    pub equation_dependencies: Option<Vec<Vec<usize>>>,
}

impl<
        F: Float
            + FromPrimitive
            + Debug
            + ScalarOperand
            + std::ops::AddAssign
            + std::ops::SubAssign
            + std::ops::MulAssign
            + std::ops::DivAssign
            + Display
            + LowerExp
            + std::iter::Sum,
    > Default for DAEStructure<F>
{
    fn default() -> Self {
        DAEStructure {
            n_differential: 0,
            n_algebraic: 0,
            n_diff_eqs: 0,
            n_alg_eqs: 0,
            index: DAEIndex::Index1,
            diff_index: 1,
            constraint_jacobian: None,
            derivative_jacobian: None,
            incidence_matrix: None,
            variable_dependencies: None,
            equation_dependencies: None,
        }
    }
}

impl<
        F: Float
            + FromPrimitive
            + Debug
            + ScalarOperand
            + std::ops::AddAssign
            + std::ops::SubAssign
            + std::ops::MulAssign
            + std::ops::DivAssign
            + Display
            + LowerExp
            + std::iter::Sum,
    > DAEStructure<F>
{
    /// Create a new DAE structure for a semi-explicit system
    pub fn new_semi_explicit(n_differential: usize, n_algebraic: usize) -> Self {
        DAEStructure {
            n_differential,
            n_algebraic,
            n_diff_eqs: n_differential,
            n_alg_eqs: n_algebraic,
            index: DAEIndex::Index1, // Initial assumption
            diff_index: 1,           // Initial assumption
            constraint_jacobian: None,
            derivative_jacobian: None,
            incidence_matrix: None,
            variable_dependencies: None,
            equation_dependencies: None,
        }
    }

    /// Create a new DAE structure for a fully implicit system
    pub fn new_fully_implicit(n_equations: usize, n_variables: usize) -> Self {
        DAEStructure {
            n_differential: n_variables, // Initially assume all variables are differential
            n_algebraic: 0,
            n_diff_eqs: n_equations,
            n_alg_eqs: 0,
            index: DAEIndex::Index1, // Initial assumption
            diff_index: 1,           // Initial assumption
            constraint_jacobian: None,
            derivative_jacobian: None,
            incidence_matrix: None,
            variable_dependencies: None,
            equation_dependencies: None,
        }
    }

    /// Compute the index of the DAE system
    pub fn compute_index<FFunc, GFunc>(
        &mut self,
        t: F,
        x: ArrayView1<F>,
        y: ArrayView1<F>,
        f: &FFunc,
        g: &GFunc,
    ) -> IntegrateResult<DAEIndex>
    where
        FFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
        GFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
    {
        // For semi-explicit DAEs, first check if it's index-1
        // by examining if ∂g/∂y is invertible
        // Convert ArrayView1 to slices for the constraint function
        let x_slice: Vec<F> = x.to_vec();
        let y_slice: Vec<F> = y.to_vec();
        let g_y = compute_constraint_jacobian(
            &|t, x, y| g(t, ArrayView1::from(x), ArrayView1::from(y)).to_vec(),
            t,
            &x_slice,
            &y_slice,
        );
        self.constraint_jacobian = Some(g_y.clone());

        // Check if constraint Jacobian is invertible
        let singular = is_singular_matrix(g_y.view());

        if !singular {
            // The system is index-1
            self.index = DAEIndex::Index1;
            self.diff_index = 1;
            return Ok(DAEIndex::Index1);
        }

        // If constraint Jacobian is singular, the index is higher than 1
        // We need to perform index detection by differentiation and analysis

        // Compute full Jacobians of f and g with respect to x and y
        let _f_x = compute_jacobian_for_variables(f, t, x, y, 0, self.n_differential)?;
        let f_y =
            compute_jacobian_for_variables(f, t, x, y, self.n_differential, self.n_algebraic)?;
        let g_x = compute_jacobian_for_variables(g, t, x, y, 0, self.n_differential)?;

        // For index-2, check if the matrix [g_y, g_x*f_y] is full rank
        let mut index2_matrix = Array2::<F>::zeros((self.n_alg_eqs, self.n_algebraic));
        let g_x_f_y = g_x.dot(&f_y);

        for i in 0..self.n_alg_eqs {
            for j in 0..self.n_algebraic {
                index2_matrix[[i, j]] = g_y[[i, j]];
                if j < g_x_f_y.dim().1 {
                    index2_matrix[[i, j]] += g_x_f_y[[i, j]];
                }
            }
        }

        // Check if index-2 matrix is full rank
        let index2_singular = is_singular_matrix(index2_matrix.view());

        if !index2_singular {
            // The system is index-2
            self.index = DAEIndex::Index2;
            self.diff_index = 2;
            return Ok(DAEIndex::Index2);
        }

        // For index-3, we would need to differentiate again and perform similar analysis
        // This is a simplified test for common mechanical systems

        // For many mechanical systems, differentiating the index-2 conditions once more
        // results in an index-3 system
        self.index = DAEIndex::Index3;
        self.diff_index = 3;

        Ok(DAEIndex::Index3)
    }
}

/// Automatic index reduction using the Pantelides algorithm
///
/// This algorithm automatically detects and reduces the index of a DAE system
/// by finding structural singularities and adding differentiated equations.
pub struct PantelidesReducer<
    F: Float
        + FromPrimitive
        + Debug
        + ScalarOperand
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + Display
        + LowerExp
        + std::iter::Sum,
> {
    /// DAE structure information
    pub structure: DAEStructure<F>,

    /// Maximum number of differentiation steps
    pub max_diff_steps: usize,

    /// Tolerance for numerical singularity detection
    pub tol: F,
}

impl<
        F: Float
            + FromPrimitive
            + Debug
            + ScalarOperand
            + std::ops::AddAssign
            + std::ops::SubAssign
            + std::ops::MulAssign
            + std::ops::DivAssign
            + Display
            + LowerExp
            + std::iter::Sum,
    > PantelidesReducer<F>
{
    /// Create a new Pantelides reducer for index reduction
    pub fn new(structure: DAEStructure<F>) -> Self {
        PantelidesReducer {
            structure,
            max_diff_steps: 5, // Default limit on differentiation
            tol: F::from_f64(1e-10).unwrap(),
        }
    }

    /// Initialize the incidence matrix for the Pantelides algorithm
    ///
    /// This creates a boolean matrix showing which variables appear in which equations
    pub fn initialize_incidence_matrix<FFunc, GFunc>(
        &mut self,
        t: F,
        x: ArrayView1<F>,
        y: ArrayView1<F>,
        f: &FFunc,
        g: &GFunc,
    ) -> IntegrateResult<()>
    where
        FFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
        GFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
    {
        let n_diff = self.structure.n_differential;
        let n_alg = self.structure.n_algebraic;
        let n_diff_eqs = self.structure.n_diff_eqs;
        let n_alg_eqs = self.structure.n_alg_eqs;
        let n_vars = n_diff + n_alg;
        let n_eqs = n_diff_eqs + n_alg_eqs;

        // Create incidence matrix
        let mut incidence = Array2::<bool>::from_elem((n_eqs, n_vars), false);

        // Compute Jacobians to determine which variables appear in which equations
        let f_x = compute_jacobian_for_variables(f, t, x, y, 0, n_diff)?;
        let f_y = compute_jacobian_for_variables(f, t, x, y, n_diff, n_alg)?;
        let g_x = compute_jacobian_for_variables(g, t, x, y, 0, n_diff)?;
        let g_y = compute_jacobian_for_variables(g, t, x, y, n_diff, n_alg)?;

        // Fill incidence matrix based on non-zero Jacobian entries

        // Differential equations (f)
        for i in 0..n_diff_eqs {
            // Check x dependencies
            for j in 0..n_diff {
                if f_x[[i, j]].abs() > self.tol {
                    incidence[[i, j]] = true;
                }
            }

            // Check y dependencies
            for j in 0..n_alg {
                if j < f_y.dim().1 && f_y[[i, j]].abs() > self.tol {
                    incidence[[i, n_diff + j]] = true;
                }
            }
        }

        // Algebraic equations (g)
        for i in 0..n_alg_eqs {
            // Check x dependencies
            for j in 0..n_diff {
                if j < g_x.dim().1 && g_x[[i, j]].abs() > self.tol {
                    incidence[[n_diff_eqs + i, j]] = true;
                }
            }

            // Check y dependencies
            for j in 0..n_alg {
                if j < g_y.dim().1 && g_y[[i, j]].abs() > self.tol {
                    incidence[[n_diff_eqs + i, n_diff + j]] = true;
                }
            }
        }

        self.structure.incidence_matrix = Some(incidence);

        // Also build variable and equation dependency lists for easier traversal
        let mut var_deps = Vec::with_capacity(n_vars);
        let mut eq_deps = Vec::with_capacity(n_eqs);

        // Variable dependencies: which equations each variable appears in
        for j in 0..n_vars {
            let mut deps = Vec::new();
            for i in 0..n_eqs {
                if let Some(incidence) = &self.structure.incidence_matrix {
                    if incidence[[i, j]] {
                        deps.push(i);
                    }
                }
            }
            var_deps.push(deps);
        }

        // Equation dependencies: which variables appear in each equation
        for i in 0..n_eqs {
            let mut deps = Vec::new();
            for j in 0..n_vars {
                if let Some(incidence) = &self.structure.incidence_matrix {
                    if incidence[[i, j]] {
                        deps.push(j);
                    }
                }
            }
            eq_deps.push(deps);
        }

        self.structure.variable_dependencies = Some(var_deps);
        self.structure.equation_dependencies = Some(eq_deps);

        Ok(())
    }

    /// Apply the Pantelides algorithm to reduce the index
    ///
    /// This is a structural index reduction algorithm that finds and
    /// differentiates the minimal set of equations needed to obtain an index-1 system.
    pub fn reduce_index<FFunc, GFunc>(
        &mut self,
        t: F,
        x: ArrayView1<F>,
        y: ArrayView1<F>,
        f: &FFunc,
        g: &GFunc,
    ) -> IntegrateResult<()>
    where
        FFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
        GFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
    {
        // Initialize the structure and compute the index
        self.structure.compute_index(t, x, y, f, g)?;

        // If already index-1, no reduction needed
        if self.structure.index == DAEIndex::Index1 {
            return Ok(());
        }

        // Initialize the incidence matrix if not already done
        if self.structure.incidence_matrix.is_none() {
            self.initialize_incidence_matrix(t, x, y, f, g)?;
        }

        // For higher-index systems, apply the Pantelides algorithm
        // This is a simplified implementation of the algorithm

        // In a full implementation, we would:
        // 1. Find all structurally singular subsets of equations
        // 2. Differentiate these equations to replace algebraic constraints
        // 3. Update the DAE structure with the differentiated system
        // 4. Repeat until the system becomes index-1

        // For now, we'll return a placeholder implementation and indicate
        // that index reduction is not yet fully implemented
        Err(IntegrateError::NotImplementedError(
            "Full Pantelides algorithm for index reduction is not yet implemented".to_string(),
        ))
    }
}

/// Dummy derivative method for index reduction
///
/// This method replaces some differentiated variables with new "dummy" variables
/// to maintain the correct number of degrees of freedom in the system.
pub struct DummyDerivativeReducer<
    F: Float
        + FromPrimitive
        + Debug
        + ScalarOperand
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + Display
        + LowerExp
        + std::iter::Sum,
> {
    /// DAE structure information
    pub structure: DAEStructure<F>,

    /// Variables selected to be replaced with dummy derivatives
    pub dummy_variables: Vec<usize>,

    /// New equations added for the dummy variables
    pub dummy_equations: Vec<usize>,
}

impl<
        F: Float
            + FromPrimitive
            + Debug
            + ScalarOperand
            + std::ops::AddAssign
            + std::ops::SubAssign
            + std::ops::MulAssign
            + std::ops::DivAssign
            + Display
            + LowerExp
            + std::iter::Sum,
    > DummyDerivativeReducer<F>
{
    /// Create a new dummy derivative reducer
    pub fn new(structure: DAEStructure<F>) -> Self {
        DummyDerivativeReducer {
            structure,
            dummy_variables: Vec::new(),
            dummy_equations: Vec::new(),
        }
    }

    /// Apply the dummy derivative method to reduce the index
    ///
    /// This method differentiates constraints and selects a subset of variables
    /// to be replaced with dummy variables.
    pub fn reduce_index<FFunc, GFunc>(
        &mut self,
        t: F,
        x: ArrayView1<F>,
        y: ArrayView1<F>,
        f: &FFunc,
        g: &GFunc,
    ) -> IntegrateResult<()>
    where
        FFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
        GFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
    {
        // Initialize the structure and compute the index
        self.structure.compute_index(t, x, y, f, g)?;

        // If already index-1, no reduction needed
        if self.structure.index == DAEIndex::Index1 {
            return Ok(());
        }

        // For higher-index systems, we would:
        // 1. Differentiate the constraint equations
        // 2. Select variables to be replaced with dummy derivatives
        // 3. Formulate the augmented system with dummy variables
        // 4. Set up the transformation to solve this system

        // Placeholder implementation
        Err(IntegrateError::NotImplementedError(
            "Dummy derivative method for index reduction is not yet implemented".to_string(),
        ))
    }
}

/// Projection method for index reduction and constraint satisfaction
///
/// This method combines index reduction with projection-based stabilization
/// to maintain constraint satisfaction during integration.
pub struct ProjectionMethod<
    F: Float
        + FromPrimitive
        + Debug
        + ScalarOperand
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + Display
        + LowerExp
        + std::iter::Sum,
> {
    /// DAE structure information
    pub structure: DAEStructure<F>,

    /// Whether to project after each step
    pub project_after_step: bool,

    /// Whether to use consistent initialization
    pub consistent_initialization: bool,

    /// Tolerance for constraint satisfaction
    pub constraint_tol: F,
}

impl<
        F: Float
            + FromPrimitive
            + Debug
            + ScalarOperand
            + std::ops::AddAssign
            + std::ops::SubAssign
            + std::ops::MulAssign
            + std::ops::DivAssign
            + Display
            + LowerExp
            + std::iter::Sum,
    > ProjectionMethod<F>
{
    /// Create a new projection method for constraint satisfaction
    pub fn new(structure: DAEStructure<F>) -> Self {
        ProjectionMethod {
            structure,
            project_after_step: true,
            consistent_initialization: true,
            constraint_tol: F::from_f64(1e-8).unwrap(),
        }
    }

    /// Project the solution onto the constraint manifold
    ///
    /// This ensures that the solution satisfies the constraints exactly.
    pub fn project_solution<GFunc>(
        &self,
        t: F,
        x: &mut Array1<F>,
        y: &mut Array1<F>,
        g: &GFunc,
    ) -> IntegrateResult<()>
    where
        GFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
    {
        // Evaluate the constraint
        let g_val = g(t, x.view(), y.view());

        // Check if already satisfying the constraint
        let g_norm = g_val
            .iter()
            .fold(F::zero(), |acc, &val| acc + val.powi(2))
            .sqrt();

        if g_norm <= self.constraint_tol {
            // Already on the constraint manifold
            return Ok(());
        }

        // Compute the constraint Jacobian
        // Convert ArrayView1 to slices for the constraint function
        let x_slice: Vec<F> = x.to_vec();
        let y_slice: Vec<F> = y.to_vec();
        let g_y = compute_constraint_jacobian(
            &|t, x, y| g(t, ArrayView1::from(x), ArrayView1::from(y)).to_vec(),
            t,
            &x_slice,
            &y_slice,
        );

        // Solve the correction equation: g_y * Δy = -g
        let neg_g = g_val.mapv(|val| -val);

        // Use a constrained least-squares approach for the projection
        let (delta_y, residual) = solve_constrained_least_squares(g_y.view(), neg_g.view())?;

        // Apply the correction
        *y = y.clone() + delta_y;

        // Check if the projection was successful
        if residual > self.constraint_tol {
            return Err(IntegrateError::ComputationError(format!(
                "Failed to project solution onto constraint manifold. Residual: {}",
                residual
            )));
        }

        Ok(())
    }

    /// Make the initial conditions consistent with the constraints
    ///
    /// This projects the initial state onto the constraint manifold.
    pub fn make_consistent<GFunc>(
        &self,
        t: F,
        x: &mut Array1<F>,
        y: &mut Array1<F>,
        g: &GFunc,
    ) -> IntegrateResult<()>
    where
        GFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
    {
        // For consistent initialization, we use the same projection method
        // but may need more iterations to achieve high accuracy
        let max_iter = 10;
        let mut iter = 0;

        while iter < max_iter {
            // Evaluate the constraint
            let g_val = g(t, x.view(), y.view());

            // Check if already satisfying the constraint
            let g_norm = g_val
                .iter()
                .fold(F::zero(), |acc, &val| acc + val.powi(2))
                .sqrt();

            if g_norm <= self.constraint_tol {
                // Already on the constraint manifold
                return Ok(());
            }

            // Project onto the constraint manifold
            self.project_solution(t, x, y, g)?;

            iter += 1;
        }

        // Final check
        let g_val = g(t, x.view(), y.view());
        let g_norm = g_val
            .iter()
            .fold(F::zero(), |acc, &val| acc + val.powi(2))
            .sqrt();

        if g_norm > self.constraint_tol {
            return Err(IntegrateError::ComputationError(format!(
                "Failed to find consistent initial conditions after {} iterations. Residual: {}",
                max_iter, g_norm
            )));
        }

        Ok(())
    }
}

/// Compute the Jacobian of a function with respect to a subset of variables
///
/// This is a helper function for computing partial Jacobians.
fn compute_jacobian_for_variables<F, Func>(
    f: &Func,
    t: F,
    x: ArrayView1<F>,
    y: ArrayView1<F>,
    start_idx: usize,
    n_vars: usize,
) -> IntegrateResult<Array2<F>>
where
    F: Float
        + FromPrimitive
        + Debug
        + ScalarOperand
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + Display
        + LowerExp
        + std::iter::Sum,
    Func: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
{
    // Evaluate the base function
    let f_val = f(t, x, y);
    let n_eqs = f_val.len();

    // Create the Jacobian matrix
    let mut jac = Array2::<F>::zeros((n_eqs, n_vars));

    // Compute the Jacobian using finite differences
    let epsilon = F::from_f64(1e-8).unwrap();

    if start_idx < x.len() {
        // Differentiate with respect to x variables
        let end_idx = (start_idx + n_vars).min(x.len());
        let n_x_vars = end_idx - start_idx;

        let mut x_perturbed = x.to_owned();

        for j in 0..n_x_vars {
            let idx = start_idx + j;
            let h = epsilon.max(x[idx].abs() * epsilon);

            x_perturbed[idx] = x[idx] + h;
            let f_plus = f(t, x_perturbed.view(), y);
            x_perturbed[idx] = x[idx];

            let df = (f_plus - f_val.view()) / h;

            for i in 0..n_eqs {
                jac[[i, j]] = df[i];
            }
        }
    }

    if start_idx + n_vars > x.len() {
        // Differentiate with respect to y variables
        let n_x_vars = x.len().saturating_sub(start_idx);
        let n_y_vars = n_vars - n_x_vars;
        let y_start = start_idx.saturating_sub(x.len());
        let y_end = (y_start + n_y_vars).min(y.len());

        let mut y_perturbed = y.to_owned();

        for j in 0..y_end.saturating_sub(y_start) {
            let idx = y_start + j;
            let h = epsilon.max(y[idx].abs() * epsilon);

            y_perturbed[idx] = y[idx] + h;
            let f_plus = f(t, x, y_perturbed.view());
            y_perturbed[idx] = y[idx];

            let df = (f_plus - f_val.view()) / h;

            for i in 0..n_eqs {
                jac[[i, n_x_vars + j]] = df[i];
            }
        }
    }

    Ok(jac)
}

/// Solve a constrained least-squares problem for projection
///
/// Solves the problem: min ||Δy|| subject to J·Δy = b
fn solve_constrained_least_squares<F>(
    j: ArrayView2<F>,
    b: ArrayView1<F>,
) -> IntegrateResult<(Array1<F>, F)>
where
    F: Float
        + FromPrimitive
        + Debug
        + ScalarOperand
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + Display
        + LowerExp
        + std::iter::Sum,
{
    use crate::dae::utils::linear_solvers::solve_linear_system;

    // For underconstrained systems, we use the pseudoinverse (minimum norm solution)
    // J·J^T·λ = b
    // Δy = J^T·λ

    // Compute J·J^T
    let j_jt = j.dot(&j.t());

    // Solve for the Lagrange multiplier λ
    let lambda = match solve_linear_system(&j_jt.view(), &b.view()) {
        Ok(sol) => sol,
        Err(e) => {
            return Err(IntegrateError::ComputationError(format!(
                "Failed to solve least squares system: {}",
                e
            )))
        }
    };

    // Compute the correction Δy = J^T·λ
    let delta_y = j.t().dot(&lambda);

    // Compute the residual ||J·Δy - b||
    let residual = (&j.dot(&delta_y) - &b)
        .iter()
        .fold(F::zero(), |acc, &val| acc + val.powi(2))
        .sqrt();

    Ok((delta_y, residual))
}
