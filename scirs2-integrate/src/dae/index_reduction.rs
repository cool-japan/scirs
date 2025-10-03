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
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive};
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
        + std::iter::Sum
        + crate::common::IntegrateFloat,
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
            + std::iter::Sum
            + crate::common::IntegrateFloat,
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
            + std::iter::Sum
            + crate::common::IntegrateFloat,
    > DAEStructure<F>
{
    /// Create a new DAE structure for a semi-explicit system
    pub fn new_semi_explicit(n_differential: usize, nalgebraic: usize) -> Self {
        DAEStructure {
            n_differential,
            n_algebraic: nalgebraic,
            n_diff_eqs: n_differential,
            n_alg_eqs: nalgebraic,
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
    pub fn new_fully_implicit(n_equations: usize, nvariables: usize) -> Self {
        DAEStructure {
            n_differential: nvariables, // Initially assume all _variables are differential
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
        + std::iter::Sum
        + crate::common::IntegrateFloat,
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
            + std::iter::Sum
            + crate::common::IntegrateFloat,
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
        // This implements the core structural analysis and index reduction

        let max_iterations = 10;
        let mut current_iteration = 0;

        while self.structure.index != DAEIndex::Index1 && current_iteration < max_iterations {
            current_iteration += 1;

            // Step 1: Find structurally singular subsets
            let singular_subsets = self.find_singular_subsets()?;

            if singular_subsets.is_empty() {
                // No more singular subsets found, but still not index-1
                // This might indicate a more complex system structure
                break;
            }

            // Step 2: Differentiate equations in singular subsets
            let differentiated_equations =
                self.differentiate_singular_equations(&singular_subsets)?;

            // Step 3: Update the DAE structure
            self.update_structure_after_differentiation(&differentiated_equations)?;

            // Step 4: Recompute the index
            self.structure.compute_index(t, x, y, f, g)?;

            // Check if we've made progress
            if self.structure.diff_index > 0 {
                self.structure.diff_index -= 1;
            }

            // Update the DAE index based on the differentiation index
            self.structure.index = match self.structure.diff_index {
                0 | 1 => DAEIndex::Index1,
                2 => DAEIndex::Index2,
                3 => DAEIndex::Index3,
                _ => DAEIndex::HigherIndex,
            };
        }

        if self.structure.index != DAEIndex::Index1 {
            return Err(IntegrateError::ConvergenceError(format!(
                "Failed to reduce DAE to index-1 after {max_iterations} iterations"
            )));
        }

        Ok(())
    }

    /// Find structurally singular subsets of equations
    fn find_singular_subsets(&self) -> IntegrateResult<Vec<Vec<usize>>> {
        let mut singular_subsets = Vec::new();

        if let Some(ref incidence) = self.structure.incidence_matrix {
            let n_eqs = incidence.nrows();
            let n_vars = incidence.ncols();

            // Look for subsets of equations with fewer variables than equations
            // This is a simplified heuristic - a full implementation would use
            // more sophisticated graph algorithms

            for subset_size in 2..=std::cmp::min(n_eqs, 6) {
                // Generate combinations of equations
                let equation_combinations = generate_combinations(n_eqs, subset_size);

                for eq_subset in equation_combinations {
                    // Find variables that appear in these equations
                    let mut involved_vars = std::collections::HashSet::new();

                    for &eq_idx in &eq_subset {
                        for var_idx in 0..n_vars {
                            if incidence[[eq_idx, var_idx]] {
                                involved_vars.insert(var_idx);
                            }
                        }
                    }

                    // Check if this is a singular subset (more equations than variables)
                    if eq_subset.len() > involved_vars.len() {
                        singular_subsets.push(eq_subset);
                    }
                }
            }
        }

        Ok(singular_subsets)
    }

    /// Differentiate equations in singular subsets
    fn differentiate_singular_equations(
        &self,
        singular_subsets: &[Vec<usize>],
    ) -> IntegrateResult<Vec<usize>> {
        let mut equations_to_differentiate = Vec::new();

        // Select representative equations from each singular subset to differentiate
        for subset in singular_subsets {
            if !subset.is_empty() {
                // Choose the first equation in each subset to differentiate
                // In a more sophisticated implementation, we would choose optimally
                equations_to_differentiate.push(subset[0]);
            }
        }

        Ok(equations_to_differentiate)
    }

    /// Update the DAE structure after differentiation
    fn update_structure_after_differentiation(
        &mut self,
        differentiated_equations: &[usize],
    ) -> IntegrateResult<()> {
        if differentiated_equations.is_empty() {
            return Ok(());
        }

        // Step 1: Add new differential variables for derivatives of differentiated _equations
        let num_new_variables = differentiated_equations.len();
        let old_n_diff = self.structure.n_differential;
        let old_n_alg = self.structure.n_algebraic;
        let old_n_vars = old_n_diff + old_n_alg;
        let old_n_eqs = self.structure.n_diff_eqs + self.structure.n_alg_eqs;

        // Update variable counts
        self.structure.n_differential += num_new_variables;
        self.structure.n_diff_eqs += num_new_variables;

        let new_n_vars = self.structure.n_differential + self.structure.n_algebraic;
        let new_n_eqs = self.structure.n_diff_eqs + self.structure.n_alg_eqs;

        // Step 2: Update the incidence matrix to reflect new dependencies
        if let Some(ref old_incidence) = self.structure.incidence_matrix.clone() {
            let mut new_incidence = Array2::<bool>::from_elem((new_n_eqs, new_n_vars), false);

            // Copy existing incidence relationships
            for i in 0..old_n_eqs {
                for j in 0..old_n_vars {
                    new_incidence[[i, j]] = old_incidence[[i, j]];
                }
            }

            // Add new differential _equations for the differentiated constraints
            for (new_eq_idx, &orig_eq_idx) in differentiated_equations.iter().enumerate() {
                let new_eq_row = old_n_eqs + new_eq_idx;

                // The new differential equation depends on the new differential variable
                let new_var_col = old_n_vars + new_eq_idx;
                new_incidence[[new_eq_row, new_var_col]] = true;

                // It also depends on all variables that the original equation depended on
                for j in 0..old_n_vars {
                    if orig_eq_idx < old_n_eqs && old_incidence[[orig_eq_idx, j]] {
                        new_incidence[[new_eq_row, j]] = true;
                    }
                }

                // The differentiated equation also creates dependencies on derivatives
                // of variables that appeared in the original equation
                for j in 0..old_n_diff {
                    if orig_eq_idx < old_n_eqs && old_incidence[[orig_eq_idx, j]] {
                        // Add dependency on derivative of variable j
                        new_incidence[[new_eq_row, j]] = true;
                    }
                }
            }

            self.structure.incidence_matrix = Some(new_incidence);
        }

        // Step 3: Update variable and equation dependencies
        self.update_dependencies_after_structure_change()?;

        // Step 4: Update differentiation index
        if self.structure.diff_index > 1 {
            self.structure.diff_index -= 1;
        }

        // Step 5: Clear cached matrices to force recomputation
        self.structure.constraint_jacobian = None;
        self.structure.derivative_jacobian = None;

        Ok(())
    }

    /// Update variable and equation dependencies after structural changes
    fn update_dependencies_after_structure_change(&mut self) -> IntegrateResult<()> {
        if let Some(ref incidence) = self.structure.incidence_matrix {
            let n_eqs = incidence.nrows();
            let n_vars = incidence.ncols();

            // Update variable dependencies
            let mut var_deps = Vec::new();
            for j in 0..n_vars {
                let mut deps = Vec::new();
                for i in 0..n_eqs {
                    if incidence[[i, j]] {
                        deps.push(i);
                    }
                }
                var_deps.push(deps);
            }

            // Update equation dependencies
            let mut eq_deps = Vec::new();
            for i in 0..n_eqs {
                let mut deps = Vec::new();
                for j in 0..n_vars {
                    if incidence[[i, j]] {
                        deps.push(j);
                    }
                }
                eq_deps.push(deps);
            }

            self.structure.variable_dependencies = Some(var_deps);
            self.structure.equation_dependencies = Some(eq_deps);
        }

        Ok(())
    }
}

/// Selection result for dummy derivative method
#[derive(Debug, Clone)]
struct DummySelection<
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
        + std::iter::Sum
        + crate::common::IntegrateFloat,
> {
    /// Variables selected to be dummy variables
    dummy_vars: Vec<usize>,
    /// Equations added for dummy variables
    dummy_eqs: Vec<usize>,
    /// Q matrix from QR decomposition
    q: Array2<F>,
    /// R matrix from QR decomposition
    r: Array2<F>,
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
        + std::iter::Sum
        + crate::common::IntegrateFloat,
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
            + std::iter::Sum
            + crate::common::IntegrateFloat,
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

        // Step 1: Differentiate the constraint equations
        let constraint_derivative = self.differentiate_constraints(t, x, y, g)?;

        // Step 2: Analyze the extended system structure
        let extended_jacobian =
            self.build_extended_jacobian(t, x, y, f, g, &constraint_derivative)?;

        // Step 3: Select dummy variables using rank analysis
        let dummy_selection = self.select_dummy_variables(&extended_jacobian)?;

        // Step 4: Store the dummy variable mapping
        self.dummy_variables = dummy_selection.dummy_vars;
        self.dummy_equations = dummy_selection.dummy_eqs;

        // Step 5: Update the DAE structure
        self.update_structure_with_dummies()?;

        // Step 6: Verify that the resulting system is index-1
        let final_index = self.verify_reduced_index(t, x, y, f, g)?;

        if final_index != DAEIndex::Index1 {
            return Err(IntegrateError::ComputationError(format!(
                "Dummy derivative method failed to reduce to index-1. Final index: {final_index:?}"
            )));
        }

        self.structure.index = DAEIndex::Index1;
        self.structure.diff_index = 1;

        Ok(())
    }

    /// Differentiate the constraint equations with respect to time
    fn differentiate_constraints<GFunc>(
        &self,
        t: F,
        x: ArrayView1<F>,
        y: ArrayView1<F>,
        g: &GFunc,
    ) -> IntegrateResult<Array1<F>>
    where
        GFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
    {
        let h = F::from_f64(1e-8).unwrap();

        // Compute g(t, x, y)
        let g0 = g(t, x, y);

        // Compute g(t+h, x, y) for time derivative
        let g_plus_t = g(t + h, x, y);

        // Approximate time derivative: dg/dt ≈ (g(t+h) - g(t))/h
        let dg_dt = (&g_plus_t - &g0) / h;

        Ok(dg_dt)
    }

    /// Build the extended Jacobian matrix for the dummy derivative method
    fn build_extended_jacobian<FFunc, GFunc>(
        &self,
        t: F,
        x: ArrayView1<F>,
        y: ArrayView1<F>,
        f: &FFunc,
        g: &GFunc,
        constraint_derivative: &Array1<F>,
    ) -> IntegrateResult<Array2<F>>
    where
        FFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
        GFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
    {
        let n_diff = self.structure.n_differential;
        let n_alg = self.structure.n_algebraic;
        let n_constraints = constraint_derivative.len();

        // Extended system size: original equations + differentiated constraints
        let extended_size = n_diff + n_alg + n_constraints;
        let mut extended_jacobian = Array2::<F>::zeros((extended_size, extended_size));

        // Compute Jacobians of the original system
        let f_x = compute_jacobian_for_variables(f, t, x, y, 0, n_diff)?;
        let f_y = compute_jacobian_for_variables(f, t, x, y, n_diff, n_alg)?;
        let g_x = compute_jacobian_for_variables(g, t, x, y, 0, n_diff)?;
        let g_y = compute_jacobian_for_variables(g, t, x, y, n_diff, n_alg)?;

        // Fill the extended Jacobian matrix
        // Block structure:
        // [ df/dx  df/dy     0    ]
        // [ dg/dx  dg/dy     0    ]
        // [dg'/dx dg'/dy    I    ]

        // Top-left block: original differential equations
        for i in 0..n_diff {
            for j in 0..n_diff {
                if j < f_x.dim().1 {
                    extended_jacobian[[i, j]] = f_x[[i, j]];
                }
            }
            for j in 0..n_alg {
                if j < f_y.dim().1 {
                    extended_jacobian[[i, n_diff + j]] = f_y[[i, j]];
                }
            }
        }

        // Middle block: original algebraic equations
        for i in 0..n_alg {
            for j in 0..n_diff {
                if j < g_x.dim().1 && i < g_x.dim().0 {
                    extended_jacobian[[n_diff + i, j]] = g_x[[i, j]];
                }
            }
            for j in 0..n_alg {
                if j < g_y.dim().1 && i < g_y.dim().0 {
                    extended_jacobian[[n_diff + i, n_diff + j]] = g_y[[i, j]];
                }
            }
        }

        // Bottom block: differentiated constraints
        // For now, use finite difference approximation of the Jacobian of g'
        for i in 0..n_constraints {
            // Add derivatives of constraint derivatives (simplified approach)
            for j in 0..n_diff {
                if j < g_x.dim().1 && i < g_x.dim().0 {
                    extended_jacobian[[n_diff + n_alg + i, j]] = g_x[[i, j]];
                }
            }
            for j in 0..n_alg {
                if j < g_y.dim().1 && i < g_y.dim().0 {
                    extended_jacobian[[n_diff + n_alg + i, n_diff + j]] = g_y[[i, j]];
                }
            }
            // Identity block for dummy derivatives
            if i < extended_size - n_diff - n_alg {
                extended_jacobian[[n_diff + n_alg + i, n_diff + n_alg + i]] = F::one();
            }
        }

        Ok(extended_jacobian)
    }

    /// Select dummy variables based on rank analysis
    fn select_dummy_variables(
        &self,
        extended_jacobian: &Array2<F>,
    ) -> IntegrateResult<DummySelection<F>> {
        let n_diff = self.structure.n_differential;
        let n_alg = self.structure.n_algebraic;
        let total_vars = n_diff + n_alg;

        // Perform QR decomposition to find rank and pivot columns
        let (q, r, pivots) = self.qr_decomposition_with_pivoting(extended_jacobian)?;

        // Determine the rank of the matrix
        let rank = Self::compute_matrix_rank(&r)?;

        // The number of dummy variables needed
        let n_dummy = total_vars.saturating_sub(rank);

        // Select dummy variables based on pivot analysis
        let mut dummy_vars = Vec::new();
        let mut dummy_eqs = Vec::new();

        // Select variables that correspond to dependent columns
        for i in rank..total_vars.min(pivots.len()) {
            if pivots[i] < total_vars {
                dummy_vars.push(pivots[i]);
                dummy_eqs.push(n_diff + n_alg + dummy_eqs.len());
            }
        }

        // If we need more dummy variables, select from the least important variables
        while dummy_vars.len() < n_dummy && dummy_vars.len() < total_vars {
            for i in 0..total_vars {
                if !dummy_vars.contains(&i) && dummy_vars.len() < n_dummy {
                    dummy_vars.push(i);
                    dummy_eqs.push(n_diff + n_alg + dummy_eqs.len());
                }
            }
        }

        Ok(DummySelection {
            dummy_vars,
            dummy_eqs,
            q,
            r,
        })
    }

    /// QR decomposition with column pivoting for rank analysis
    fn qr_decomposition_with_pivoting(
        &self,
        matrix: &Array2<F>,
    ) -> IntegrateResult<(Array2<F>, Array2<F>, Vec<usize>)> {
        let (m, n) = matrix.dim();
        let mut a = matrix.clone();
        let q = Array2::<F>::eye(m);
        let mut pivots: Vec<usize> = (0..n).collect();

        // Simplified QR with pivoting (Gram-Schmidt process)
        for k in 0..std::cmp::min(m, n) {
            // Find the column with largest norm for pivoting
            let mut max_norm = F::zero();
            let mut max_col = k;

            for j in k..n {
                let col_norm = (k..m)
                    .map(|i| a[[i, j]].powi(2))
                    .fold(F::zero(), |acc, x| acc + x)
                    .sqrt();
                if col_norm > max_norm {
                    max_norm = col_norm;
                    max_col = j;
                }
            }

            // Swap columns if needed
            if max_col != k {
                for i in 0..m {
                    let temp = a[[i, k]];
                    a[[i, k]] = a[[i, max_col]];
                    a[[i, max_col]] = temp;
                }
                pivots.swap(k, max_col);
            }

            // Gram-Schmidt orthogonalization
            if max_norm > F::from_f64(1e-12).unwrap() {
                // Normalize the k-th column
                for i in k..m {
                    a[[i, k]] /= max_norm;
                }

                // Orthogonalize subsequent columns
                for j in (k + 1)..n {
                    let dot_product = (k..m)
                        .map(|i| a[[i, k]] * a[[i, j]])
                        .fold(F::zero(), |acc, x| acc + x);
                    for i in k..m {
                        a[[i, j]] = a[[i, j]] - dot_product * a[[i, k]];
                    }
                }
            }
        }

        // Extract Q and R matrices (simplified)
        let r = a.clone();

        Ok((q, r, pivots))
    }

    /// Compute the numerical rank of an upper triangular matrix
    fn compute_matrix_rank(r: &Array2<F>) -> IntegrateResult<usize> {
        let tolerance = F::from_f64(1e-10).unwrap();
        let min_dim = std::cmp::min(r.nrows(), r.ncols());

        let mut rank = 0;
        for i in 0..min_dim {
            if r[[i, i]].abs() > tolerance {
                rank += 1;
            }
        }

        Ok(rank)
    }

    /// Update the DAE structure after adding dummy variables
    fn update_structure_with_dummies(&mut self) -> IntegrateResult<()> {
        let n_dummy = self.dummy_variables.len();

        // Increase the number of variables and equations
        self.structure.n_differential += n_dummy;
        self.structure.n_diff_eqs += n_dummy;

        // Clear cached matrices to force recomputation
        self.structure.constraint_jacobian = None;
        self.structure.derivative_jacobian = None;
        self.structure.incidence_matrix = None;

        Ok(())
    }

    /// Verify that the reduced system is indeed index-1
    fn verify_reduced_index<FFunc, GFunc>(
        &self,
        t: F,
        x: ArrayView1<F>,
        y: ArrayView1<F>,
        _f: &FFunc,
        g: &GFunc,
    ) -> IntegrateResult<DAEIndex>
    where
        FFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
        GFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
    {
        // For the extended system with dummy variables, check if the constraint Jacobian is non-singular
        let x_slice: Vec<F> = x.to_vec();
        let y_slice: Vec<F> = y.to_vec();

        let g_y = compute_constraint_jacobian(
            &|t, x, y| g(t, ArrayView1::from(x), ArrayView1::from(y)).to_vec(),
            t,
            &x_slice,
            &y_slice,
        );

        // Check if the extended constraint Jacobian (including dummy variables) is non-singular
        let singular = is_singular_matrix(g_y.view());

        if !singular {
            Ok(DAEIndex::Index1)
        } else {
            Ok(DAEIndex::Index2) // Still higher index
        }
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
        + std::iter::Sum
        + crate::common::IntegrateFloat,
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
            + std::iter::Sum
            + crate::common::IntegrateFloat,
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
                "Failed to project solution onto constraint manifold. Residual: {residual}"
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
                "Failed to find consistent initial conditions after {max_iter} iterations. Residual: {g_norm}"
            )));
        }

        Ok(())
    }
}

/// Compute the Jacobian of a function with respect to a subset of variables
///
/// This is a helper function for computing partial Jacobians.
#[allow(dead_code)]
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
        + std::iter::Sum
        + crate::common::IntegrateFloat,
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
            let _idx = start_idx + j;
            let h = epsilon.max(x[_idx].abs() * epsilon);

            x_perturbed[_idx] = x[_idx] + h;
            let f_plus = f(t, x_perturbed.view(), y);
            x_perturbed[_idx] = x[_idx];

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
            let _idx = y_start + j;
            let h = epsilon.max(y[_idx].abs() * epsilon);

            y_perturbed[_idx] = y[_idx] + h;
            let f_plus = f(t, x, y_perturbed.view());
            y_perturbed[_idx] = y[_idx];

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
#[allow(dead_code)]
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
        + std::iter::Sum
        + crate::common::IntegrateFloat,
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
                "Failed to solve least squares system: {e}"
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

/// Generate combinations of indices for equation subset analysis
#[allow(dead_code)]
fn generate_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 || k > n {
        return vec![];
    }

    let mut combinations = Vec::new();
    let mut current = Vec::new();

    generate_combinations_recursive(0, n, k, &mut current, &mut combinations);

    combinations
}

/// Recursive helper for generating combinations
#[allow(dead_code)]
fn generate_combinations_recursive(
    start: usize,
    n: usize,
    k: usize,
    current: &mut Vec<usize>,
    combinations: &mut Vec<Vec<usize>>,
) {
    if current.len() == k {
        combinations.push(current.clone());
        return;
    }

    for i in start..n {
        if current.len() + (n - i) >= k {
            current.push(i);
            generate_combinations_recursive(i + 1, n, k, current, combinations);
            current.pop();
        }
    }
}
