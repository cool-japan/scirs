//! Extended Boundary Value Problem solvers with Robin and mixed boundary conditions
//!
//! This module extends the basic BVP solver to support more general boundary
//! conditions including Robin conditions (a*u + b*u' = c) and complex mixed
//! boundary conditions.

use crate::bvp::{BVPOptions, BVPResult};
use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// Boundary condition types for extended BVP solver
#[derive(Debug, Clone)]
pub enum BoundaryConditionType<F: IntegrateFloat> {
    /// Dirichlet: u = value
    Dirichlet { value: F },
    /// Neumann: u' = value  
    Neumann { value: F },
    /// Robin: a*u + b*u' = c
    Robin { a: F, b: F, c: F },
    /// Periodic: u(a) = u(b), u'(a) = u'(b)
    Periodic,
}

impl<F: IntegrateFloat> BoundaryConditionType<F> {
    /// Evaluate the boundary condition residual
    pub fn evaluate_residual(
        &self,
        x: F,
        y: ArrayView1<F>,
        dydt: ArrayView1<F>,
        component: usize,
    ) -> F {
        match self {
            BoundaryConditionType::Dirichlet { value } => y[component] - *value,
            BoundaryConditionType::Neumann { value } => dydt[component] - *value,
            BoundaryConditionType::Robin { a, b, c } => {
                *a * y[component] + *b * dydt[component] - *c
            }
            BoundaryConditionType::Periodic => {
                // This is handled specially in the solver
                F::zero()
            }
        }
    }

    /// Get derivative of residual with respect to y[component]
    pub fn derivative_y(&self, component: usize) -> F {
        match self {
            BoundaryConditionType::Dirichlet { .. } => F::one(),
            BoundaryConditionType::Neumann { .. } => F::zero(),
            BoundaryConditionType::Robin { a, .. } => *a,
            BoundaryConditionType::Periodic => F::one(),
        }
    }

    /// Get derivative of residual with respect to dydt[component]  
    pub fn derivative_dydt(&self, component: usize) -> F {
        match self {
            BoundaryConditionType::Dirichlet { .. } => F::zero(),
            BoundaryConditionType::Neumann { .. } => F::one(),
            BoundaryConditionType::Robin { b, .. } => *b,
            BoundaryConditionType::Periodic => F::zero(),
        }
    }
}

/// Extended boundary condition specification
#[derive(Debug, Clone)]
pub struct ExtendedBoundaryConditions<F: IntegrateFloat> {
    /// Boundary conditions at left endpoint (x = a)
    pub left: Vec<BoundaryConditionType<F>>,
    /// Boundary conditions at right endpoint (x = b)  
    pub right: Vec<BoundaryConditionType<F>>,
    /// Whether the problem has periodic boundary conditions
    pub is_periodic: bool,
}

impl<F: IntegrateFloat> ExtendedBoundaryConditions<F> {
    /// Create Dirichlet boundary conditions
    pub fn dirichlet(_left_values: Vec<F>, rightvalues: Vec<F>) -> Self {
        let left = _left_values
            .into_iter()
            .map(|value| BoundaryConditionType::Dirichlet { value })
            .collect();

        let right = rightvalues
            .into_iter()
            .map(|value| BoundaryConditionType::Dirichlet { value })
            .collect();

        Self {
            left,
            right,
            is_periodic: false,
        }
    }

    /// Create Neumann boundary conditions
    pub fn neumann(_left_values: Vec<F>, rightvalues: Vec<F>) -> Self {
        let left = _left_values
            .into_iter()
            .map(|value| BoundaryConditionType::Neumann { value })
            .collect();

        let right = rightvalues
            .into_iter()
            .map(|value| BoundaryConditionType::Neumann { value })
            .collect();

        Self {
            left,
            right,
            is_periodic: false,
        }
    }

    /// Create Robin boundary conditions: a*u + b*u' = c
    pub fn robin(
        left_coeffs: Vec<(F, F, F)>, // (a, b, c) for each component
        right_coeffs: Vec<(F, F, F)>,
    ) -> Self {
        let left = left_coeffs
            .into_iter()
            .map(|(a, b, c)| BoundaryConditionType::Robin { a, b, c })
            .collect();

        let right = right_coeffs
            .into_iter()
            .map(|(a, b, c)| BoundaryConditionType::Robin { a, b, c })
            .collect();

        Self {
            left,
            right,
            is_periodic: false,
        }
    }

    /// Create periodic boundary conditions
    pub fn periodic(dimension: usize) -> Self {
        let condition = BoundaryConditionType::Periodic;
        Self {
            left: vec![condition.clone(); dimension],
            right: vec![condition; dimension],
            is_periodic: true,
        }
    }

    /// Create mixed boundary conditions
    pub fn mixed(
        left: Vec<BoundaryConditionType<F>>,
        right: Vec<BoundaryConditionType<F>>,
    ) -> Self {
        Self {
            left,
            right,
            is_periodic: false,
        }
    }
}

/// Solve BVP with extended boundary condition support
#[allow(dead_code)]
pub fn solve_bvp_extended<F, FunType>(
    fun: FunType,
    x_span: [F; 2],
    boundary_conditions: ExtendedBoundaryConditions<F>,
    n_points: usize,
    options: Option<BVPOptions<F>>,
) -> IntegrateResult<BVPResult<F>>
where
    F: IntegrateFloat,
    FunType: Fn(F, ArrayView1<F>) -> Array1<F> + Copy,
{
    let [a, b] = x_span;

    if a >= b {
        return Err(IntegrateError::ValueError(
            "Invalid interval: left bound must be less than right bound".to_string(),
        ));
    }

    let ndim = boundary_conditions.left.len();
    if boundary_conditions.right.len() != ndim {
        return Err(IntegrateError::ValueError(
            "Left and right boundary _conditions must have same dimension".to_string(),
        ));
    }

    // Generate uniform mesh
    let mesh: Vec<F> = (0..n_points)
        .map(|i| a + (b - a) * F::from_usize(i).unwrap() / F::from_usize(n_points - 1).unwrap())
        .collect();

    // Generate initial guess (zero solution for now - could be improved)
    let mut y_init = Vec::with_capacity(n_points);
    for _i in 0..n_points {
        y_init.push(Array1::zeros(ndim));
    }

    // Apply initial guess based on boundary _conditions
    match boundary_conditions.left[0] {
        BoundaryConditionType::Dirichlet { value } => {
            if let BoundaryConditionType::Dirichlet { value: right_value } =
                boundary_conditions.right[0]
            {
                // Linear interpolation between boundary values
                for (i, y_val) in y_init.iter_mut().enumerate().take(n_points) {
                    let t = F::from_usize(i).unwrap() / F::from_usize(n_points - 1).unwrap();
                    y_val[0] = value * (F::one() - t) + right_value * t;
                }
            }
        }
        _ => {
            // For other boundary conditions, use zero initial guess
        }
    }

    // Create boundary condition function for the main BVP solver
    let bc_func = create_extended_bc_function(boundary_conditions, fun, a, b);

    // Solve using the main BVP solver
    crate::bvp::solve_bvp(fun, bc_func, Some(mesh), y_init, options)
}

/// Create boundary condition function for extended boundary conditions
#[allow(dead_code)]
fn create_extended_bc_function<F, FunType>(
    boundary_conditions: ExtendedBoundaryConditions<F>,
    fun: FunType,
    a: F,
    b: F,
) -> impl Fn(ArrayView1<F>, ArrayView1<F>) -> Array1<F>
where
    F: IntegrateFloat,
    FunType: Fn(F, ArrayView1<F>) -> Array1<F> + Copy,
{
    move |ya: ArrayView1<F>, yb: ArrayView1<F>| {
        let ndim = ya.len();

        if boundary_conditions.is_periodic {
            // For periodic boundary _conditions: u(a) = u(b), u'(a) = u'(b)
            let f_a = fun(a, ya);
            let f_b = fun(b, yb);

            let mut residuals = Array1::zeros(ndim * 2);
            for i in 0..ndim {
                residuals[i] = ya[i] - yb[i]; // u(a) = u(b)
                residuals[i + ndim] = f_a[i] - f_b[i]; // u'(a) = u'(b)
            }
            residuals
        } else {
            // General boundary _conditions
            let f_a = fun(a, ya);
            let f_b = fun(b, yb);

            let mut residuals = Array1::zeros(ndim * 2);

            // Left boundary _conditions
            for (i, bc) in boundary_conditions.left.iter().enumerate() {
                residuals[i] = bc.evaluate_residual(a, ya, f_a.view(), i);
            }

            // Right boundary _conditions
            for (i, bc) in boundary_conditions.right.iter().enumerate() {
                residuals[i + ndim] = bc.evaluate_residual(b, yb, f_b.view(), i);
            }

            residuals
        }
    }
}

/// Robin boundary condition builder for convenience
#[derive(Debug, Clone)]
pub struct RobinBC<F: IntegrateFloat> {
    /// Coefficient of u
    pub a: F,
    /// Coefficient of u'
    pub b: F,
    /// Right-hand side value
    pub c: F,
}

impl<F: IntegrateFloat> RobinBC<F> {
    /// Create new Robin boundary condition: a*u + b*u' = c
    pub fn new(a: F, b: F, c: F) -> Self {
        Self { a, b, c }
    }

    /// Create Dirichlet condition: u = c
    pub fn dirichlet(c: F) -> Self {
        Self {
            a: F::one(),
            b: F::zero(),
            c,
        }
    }

    /// Create Neumann condition: u' = c
    pub fn neumann(c: F) -> Self {
        Self {
            a: F::zero(),
            b: F::one(),
            c,
        }
    }

    /// Create insulated boundary condition: u' = 0
    pub fn insulated() -> Self {
        Self::neumann(F::zero())
    }

    /// Create convective boundary condition: u' + h*(u - u_env) = 0
    /// where h is heat transfer coefficient and u_env is environment temperature
    pub fn convective(h: F, uenv: F) -> Self {
        Self {
            a: h,
            b: F::one(),
            c: h * uenv,
        }
    }
}

/// Multipoint boundary value problem support
#[derive(Debug, Clone)]
pub struct MultipointBVP<F: IntegrateFloat> {
    /// Interior points where conditions are specified
    pub interior_points: Vec<F>,
    /// Boundary conditions at interior points
    pub interior_conditions: Vec<Vec<BoundaryConditionType<F>>>,
}

impl<F: IntegrateFloat> Default for MultipointBVP<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IntegrateFloat> MultipointBVP<F> {
    /// Create new multipoint BVP
    pub fn new() -> Self {
        Self {
            interior_points: Vec::new(),
            interior_conditions: Vec::new(),
        }
    }

    /// Add interior point with conditions
    pub fn add_interior_point(&mut self, x: F, conditions: Vec<BoundaryConditionType<F>>) {
        self.interior_points.push(x);
        self.interior_conditions.push(conditions);
    }
}

/// Solve multipoint boundary value problem
#[allow(dead_code)]
pub fn solve_multipoint_bvp<F, FunType>(
    fun: FunType,
    x_span: [F; 2],
    boundary_conditions: ExtendedBoundaryConditions<F>,
    multipoint: MultipointBVP<F>,
    n_points: usize,
    options: Option<BVPOptions<F>>,
) -> IntegrateResult<BVPResult<F>>
where
    F: IntegrateFloat,
    FunType: Fn(F, ArrayView1<F>) -> Array1<F> + Copy,
{
    if multipoint.interior_points.is_empty() {
        // No interior points, solve as regular BVP
        solve_bvp_extended(fun, x_span, boundary_conditions, n_points, options)
    } else {
        // Multipoint BVP using segmented collocation approach

        // Validate interior _points are within domain and sorted
        let [a, b] = x_span;
        let mut all_points = vec![a];
        all_points.extend(multipoint.interior_points.clone());
        all_points.push(b);

        // Sort and check uniqueness
        for i in 1..all_points.len() {
            if all_points[i] <= all_points[i - 1] {
                return Err(IntegrateError::ValueError(
                    "Interior _points must be unique and in ascending order".to_string(),
                ));
            }
        }

        // Determine dimensions
        let ndim = boundary_conditions.left.len();
        let n_segments = all_points.len() - 1;
        let points_per_segment = (n_points - 1) / n_segments + 1;

        // Build global mesh
        let mut global_mesh = Vec::new();
        for i in 0..n_segments {
            let segment_start = all_points[i];
            let segment_end = all_points[i + 1];
            let n_seg_points = if i == n_segments - 1 {
                points_per_segment
            } else {
                points_per_segment - 1 // Avoid duplicating interior _points
            };

            for j in 0..n_seg_points {
                let t = F::from_usize(j).unwrap() / F::from_usize(n_seg_points - 1).unwrap();
                let x = segment_start + (segment_end - segment_start) * t;
                global_mesh.push(x);
            }
        }

        // Initialize solution with zeros
        let total_points = global_mesh.len();
        let mut y_solution: Array2<F> = Array2::zeros((total_points, ndim));

        // Apply boundary _conditions at endpoints
        apply_initial_boundary_values(&boundary_conditions, &mut y_solution, total_points, ndim);

        // Set up collocation system
        let options = options.unwrap_or_default();
        let mut residuals = vec![F::zero(); total_points * ndim];
        let mut max_residual = F::zero();

        // Newton's method for solving the collocation system
        for _iter in 0..options.max_iter {
            // Compute residuals at all collocation _points
            compute_multipoint_residuals(
                &fun,
                &global_mesh,
                &y_solution,
                &boundary_conditions,
                &multipoint,
                &mut residuals,
                ndim,
            )?;

            // Check convergence
            max_residual =
                residuals
                    .iter()
                    .map(|&r| r.abs())
                    .fold(F::zero(), |a, b| if a > b { a } else { b });

            if max_residual < options.tol {
                // Converged
                let x = global_mesh.clone();
                let y = transpose_solution(y_solution);

                return Ok(BVPResult {
                    x: x.to_vec(),
                    y,
                    n_iter: _iter + 1,
                    success: true,
                    message: Some("Converged".to_string()),
                    residual_norm: max_residual,
                });
            }

            // Compute Jacobian and solve linear system
            let jacobian = compute_multipoint_jacobian(
                &fun,
                &global_mesh,
                &y_solution,
                &boundary_conditions,
                &multipoint,
                ndim,
                F::from(1e-6).unwrap(), // Default jacobian epsilon
            )?;

            // Solve J * delta_y = -residuals
            let delta_y = solve_sparse_system(&jacobian, &residuals)?;

            // Update solution
            for (i, delta) in delta_y.iter().enumerate() {
                let row = i / ndim;
                let col = i % ndim;
                y_solution[[row, col]] -= *delta;
            }
        }

        // Did not converge
        let x = global_mesh;
        let y = transpose_solution(y_solution);

        Ok(BVPResult {
            x,
            y,
            n_iter: options.max_iter,
            success: false,
            message: Some("Did not converge within max iterations".to_string()),
            residual_norm: max_residual,
        })
    }
}

/// Apply initial boundary values to solution array
#[allow(dead_code)]
fn apply_initial_boundary_values<F: IntegrateFloat>(
    boundary_conditions: &ExtendedBoundaryConditions<F>,
    y_solution: &mut Array2<F>,
    n_points: usize,
    _ndim: usize,
) {
    // Apply Dirichlet _conditions at boundaries if available
    for (dim, bc) in boundary_conditions.left.iter().enumerate() {
        if let BoundaryConditionType::Dirichlet { value } = bc {
            y_solution[[0, dim]] = *value;
        }
    }

    for (dim, bc) in boundary_conditions.right.iter().enumerate() {
        if let BoundaryConditionType::Dirichlet { value } = bc {
            y_solution[[n_points - 1, dim]] = *value;
        }
    }
}

/// Compute residuals for multipoint BVP
#[allow(dead_code)]
fn compute_multipoint_residuals<F: IntegrateFloat, FunType>(
    fun: &FunType,
    mesh: &[F],
    y_solution: &Array2<F>,
    boundary_conditions: &ExtendedBoundaryConditions<F>,
    multipoint: &MultipointBVP<F>,
    residuals: &mut [F],
    ndim: usize,
) -> IntegrateResult<()>
where
    FunType: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let n_points = mesh.len();
    let h = mesh[1] - mesh[0]; // Assuming uniform spacing for simplicity

    // Interior point residuals (differential equations)
    for i in 1..n_points - 1 {
        let y_prev = y_solution.row(i - 1);
        let y_curr = y_solution.row(i);
        let y_next = y_solution.row(i + 1);

        // Compute derivatives using central differences
        let dydt = (&y_next - &y_prev) / (F::from_f64(2.0).unwrap() * h);

        // Evaluate ODE
        let f_val = fun(mesh[i], y_curr);

        // Residual: dy/dt - f(t, y) = 0
        for j in 0..ndim {
            residuals[i * ndim + j] = dydt[j] - f_val[j];
        }
    }

    // Boundary condition residuals
    apply_boundary_residuals(
        boundary_conditions,
        y_solution,
        residuals,
        n_points,
        ndim,
        h,
    );

    // Interior point condition residuals
    apply_interior_residuals(multipoint, mesh, y_solution, residuals, ndim, h)?;

    Ok(())
}

/// Apply boundary condition residuals
#[allow(dead_code)]
fn apply_boundary_residuals<F: IntegrateFloat>(
    boundary_conditions: &ExtendedBoundaryConditions<F>,
    y_solution: &Array2<F>,
    residuals: &mut [F],
    n_points: usize,
    ndim: usize,
    h: F,
) {
    // Left boundary
    let y_left = y_solution.row(0);
    let y_left_next = y_solution.row(1);
    let dydt_left = (&y_left_next - &y_left) / h;

    for (dim, bc) in boundary_conditions.left.iter().enumerate() {
        residuals[dim] = bc.evaluate_residual(F::zero(), y_left, dydt_left.view(), dim);
    }

    // Right boundary
    let y_right = y_solution.row(n_points - 1);
    let y_right_prev = y_solution.row(n_points - 2);
    let dydt_right = (&y_right - &y_right_prev) / h;

    for (dim, bc) in boundary_conditions.right.iter().enumerate() {
        residuals[(n_points - 1) * ndim + dim] =
            bc.evaluate_residual(F::zero(), y_right, dydt_right.view(), dim);
    }
}

/// Apply interior point condition residuals
#[allow(dead_code)]
fn apply_interior_residuals<F: IntegrateFloat>(
    multipoint: &MultipointBVP<F>,
    mesh: &[F],
    y_solution: &Array2<F>,
    residuals: &mut [F],
    ndim: usize,
    h: F,
) -> IntegrateResult<()> {
    // Find indices of interior condition points
    for (point_idx, &interior_x) in multipoint.interior_points.iter().enumerate() {
        // Find closest mesh point
        let mesh_idx = mesh
            .iter()
            .position(|&x| (x - interior_x).abs() < F::from_f64(1e-10).unwrap())
            .ok_or_else(|| {
                IntegrateError::ValueError("Interior point not found in mesh".to_string())
            })?;

        let y_at_point = y_solution.row(mesh_idx);

        // Compute derivative at interior point
        let dydt_at_point = if mesh_idx > 0 && mesh_idx < mesh.len() - 1 {
            let y_prev = y_solution.row(mesh_idx - 1);
            let y_next = y_solution.row(mesh_idx + 1);
            (&y_next - &y_prev) / (F::from_f64(2.0).unwrap() * h)
        } else {
            // Use one-sided difference at boundaries
            if mesh_idx == 0 {
                let y_next = y_solution.row(1);
                (&y_next - &y_at_point) / h
            } else {
                let y_prev = y_solution.row(mesh_idx - 1);
                (&y_at_point - &y_prev) / h
            }
        };

        // Apply each condition at this interior point
        for (cond_idx, condition) in multipoint.interior_conditions[point_idx].iter().enumerate() {
            residuals[mesh_idx * ndim + cond_idx] =
                condition.evaluate_residual(interior_x, y_at_point, dydt_at_point.view(), cond_idx);
        }
    }

    Ok(())
}

/// Compute Jacobian for multipoint BVP (simplified version)
#[allow(dead_code)]
fn compute_multipoint_jacobian<F: IntegrateFloat, FunType>(
    fun: &FunType,
    mesh: &[F],
    y_solution: &Array2<F>,
    boundary_conditions: &ExtendedBoundaryConditions<F>,
    multipoint: &MultipointBVP<F>,
    ndim: usize,
    eps: F,
) -> IntegrateResult<Vec<Vec<F>>>
where
    FunType: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let n_points = mesh.len();
    let total_size = n_points * ndim;
    let mut jacobian = vec![vec![F::zero(); total_size]; total_size];

    // Use finite differences to approximate Jacobian
    let mut residuals_base = vec![F::zero(); total_size];
    let mut residuals_pert = vec![F::zero(); total_size];

    // Compute base residuals
    compute_multipoint_residuals(
        fun,
        mesh,
        y_solution,
        boundary_conditions,
        multipoint,
        &mut residuals_base,
        ndim,
    )?;

    // Perturb each variable and compute Jacobian columns
    let mut y_pert = y_solution.clone();

    for col in 0..total_size {
        let row_idx = col / ndim;
        let dim_idx = col % ndim;

        // Perturb
        let original = y_pert[[row_idx, dim_idx]];
        y_pert[[row_idx, dim_idx]] = original + eps;

        // Compute perturbed residuals
        compute_multipoint_residuals(
            fun,
            mesh,
            &y_pert,
            boundary_conditions,
            multipoint,
            &mut residuals_pert,
            ndim,
        )?;

        // Compute Jacobian column
        for row in 0..total_size {
            jacobian[row][col] = (residuals_pert[row] - residuals_base[row]) / eps;
        }

        // Restore original value
        y_pert[[row_idx, dim_idx]] = original;
    }

    Ok(jacobian)
}

/// Solve sparse linear system (simplified dense solver)
#[allow(dead_code)]
fn solve_sparse_system<F: IntegrateFloat>(
    jacobian: &[Vec<F>],
    residuals: &[F],
) -> IntegrateResult<Vec<F>> {
    // Convert to dense matrix and use LU decomposition
    // In a real implementation, we'd use a sparse solver
    let n = jacobian.len();
    let mut a = Array2::zeros((n, n));
    let mut b = Array1::zeros(n);

    for i in 0..n {
        for j in 0..n {
            a[[i, j]] = jacobian[i][j];
        }
        b[i] = residuals[i];
    }

    // Use scirs2-linalg for solving
    // For now, use a simple Gaussian elimination
    let solution = gaussian_elimination(a, b)?;

    Ok(solution.to_vec())
}

/// Simple Gaussian elimination solver
#[allow(dead_code)]
fn gaussian_elimination<F: IntegrateFloat>(
    mut a: Array2<F>,
    mut b: Array1<F>,
) -> IntegrateResult<Array1<F>> {
    let n = a.nrows();

    // Forward elimination
    for k in 0..n - 1 {
        // Find pivot
        let mut max_row = k;
        for i in k + 1..n {
            if a[[i, k]].abs() > a[[max_row, k]].abs() {
                max_row = i;
            }
        }

        // Swap rows
        if max_row != k {
            for j in 0..n {
                let temp = a[[k, j]];
                a[[k, j]] = a[[max_row, j]];
                a[[max_row, j]] = temp;
            }
            let temp = b[k];
            b[k] = b[max_row];
            b[max_row] = temp;
        }

        // Check for singular matrix
        if a[[k, k]].abs() < F::from_f64(1e-12).unwrap() {
            return Err(IntegrateError::ComputationError(
                "Singular matrix in Gaussian elimination".to_string(),
            ));
        }

        // Eliminate column
        for i in k + 1..n {
            let factor = a[[i, k]] / a[[k, k]];
            for j in k + 1..n {
                a[[i, j]] = a[[i, j]] - factor * a[[k, j]];
            }
            b[i] = b[i] - factor * b[k];
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in i + 1..n {
            x[i] = x[i] - a[[i, j]] * x[j];
        }
        x[i] /= a[[i, i]];
    }

    Ok(x)
}

/// Transpose solution from row-major to column-major format
#[allow(dead_code)]
fn transpose_solution<F: IntegrateFloat>(solution: Array2<F>) -> Vec<Array1<F>> {
    let n_points = solution.nrows();
    let _ndim = solution.ncols();

    let mut result = Vec::with_capacity(n_points);
    for i in 0..n_points {
        result.push(solution.row(i).to_owned());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_robin_boundary_conditions() {
        // Test Robin BC builder
        let robin = RobinBC::new(2.0, 1.0, 3.0); // 2*u + 1*u' = 3
        assert_abs_diff_eq!(robin.a, 2.0);
        assert_abs_diff_eq!(robin.b, 1.0);
        assert_abs_diff_eq!(robin.c, 3.0);

        // Test convenience methods
        let dirichlet = RobinBC::dirichlet(5.0); // u = 5
        assert_abs_diff_eq!(dirichlet.a, 1.0);
        assert_abs_diff_eq!(dirichlet.b, 0.0);
        assert_abs_diff_eq!(dirichlet.c, 5.0);

        let neumann = RobinBC::neumann(2.0); // u' = 2
        assert_abs_diff_eq!(neumann.a, 0.0);
        assert_abs_diff_eq!(neumann.b, 1.0);
        assert_abs_diff_eq!(neumann.c, 2.0);

        let insulated: RobinBC<f64> = RobinBC::insulated(); // u' = 0
        assert_abs_diff_eq!(insulated.a, 0.0);
        assert_abs_diff_eq!(insulated.b, 1.0);
        assert_abs_diff_eq!(insulated.c, 0.0);
    }

    #[test]
    fn test_boundary_condition_evaluation() {
        let y = Array1::from_vec(vec![2.0, 3.0]);
        let dydt = Array1::from_vec(vec![1.0, -1.0]);

        // Test Dirichlet: u = 5
        let dirichlet = BoundaryConditionType::Dirichlet { value: 5.0 };
        let residual = dirichlet.evaluate_residual(0.0, y.view(), dydt.view(), 0);
        assert_abs_diff_eq!(residual, -3.0); // 2 - 5 = -3

        // Test Neumann: u' = 0.5
        let neumann = BoundaryConditionType::Neumann { value: 0.5 };
        let residual = neumann.evaluate_residual(0.0, y.view(), dydt.view(), 0);
        assert_abs_diff_eq!(residual, 0.5); // 1 - 0.5 = 0.5

        // Test Robin: 2*u + 3*u' = 10
        let robin = BoundaryConditionType::Robin {
            a: 2.0,
            b: 3.0,
            c: 10.0,
        };
        let residual = robin.evaluate_residual(0.0, y.view(), dydt.view(), 0);
        assert_abs_diff_eq!(residual, -3.0); // 2*2 + 3*1 - 10 = -3
    }

    #[test]
    fn test_extended_boundary_conditions_creation() {
        // Test Dirichlet creation
        let dirichlet = ExtendedBoundaryConditions::dirichlet(vec![1.0, 2.0], vec![3.0, 4.0]);
        assert!(!dirichlet.is_periodic);
        assert_eq!(dirichlet.left.len(), 2);
        assert_eq!(dirichlet.right.len(), 2);

        // Test Robin creation
        let robin = ExtendedBoundaryConditions::robin(
            vec![(1.0, 0.0, 5.0), (0.0, 1.0, 2.0)], // u = 5, u' = 2
            vec![(1.0, 0.0, 3.0), (0.0, 1.0, 1.0)], // u = 3, u' = 1
        );
        assert!(!robin.is_periodic);
        assert_eq!(robin.left.len(), 2);
        assert_eq!(robin.right.len(), 2);

        // Test periodic creation
        let periodic: ExtendedBoundaryConditions<f64> = ExtendedBoundaryConditions::periodic(3);
        assert!(periodic.is_periodic);
        assert_eq!(periodic.left.len(), 3);
        assert_eq!(periodic.right.len(), 3);
    }
}
