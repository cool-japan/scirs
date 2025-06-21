//! Extended Boundary Value Problem solvers with Robin and mixed boundary conditions
//!
//! This module extends the basic BVP solver to support more general boundary
//! conditions including Robin conditions (a*u + b*u' = c) and complex mixed
//! boundary conditions.

use crate::bvp::{BVPOptions, BVPResult};
use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use ndarray::{Array1, ArrayView1};

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
        _x: F,
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
    pub fn derivative_y(&self, _component: usize) -> F {
        match self {
            BoundaryConditionType::Dirichlet { .. } => F::one(),
            BoundaryConditionType::Neumann { .. } => F::zero(),
            BoundaryConditionType::Robin { a, .. } => *a,
            BoundaryConditionType::Periodic => F::one(),
        }
    }

    /// Get derivative of residual with respect to dydt[component]  
    pub fn derivative_dydt(&self, _component: usize) -> F {
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
    pub fn dirichlet(left_values: Vec<F>, right_values: Vec<F>) -> Self {
        let left = left_values
            .into_iter()
            .map(|value| BoundaryConditionType::Dirichlet { value })
            .collect();

        let right = right_values
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
    pub fn neumann(left_values: Vec<F>, right_values: Vec<F>) -> Self {
        let left = left_values
            .into_iter()
            .map(|value| BoundaryConditionType::Neumann { value })
            .collect();

        let right = right_values
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

    let n_dim = boundary_conditions.left.len();
    if boundary_conditions.right.len() != n_dim {
        return Err(IntegrateError::ValueError(
            "Left and right boundary conditions must have same dimension".to_string(),
        ));
    }

    // Generate uniform mesh
    let mesh: Vec<F> = (0..n_points)
        .map(|i| a + (b - a) * F::from_usize(i).unwrap() / F::from_usize(n_points - 1).unwrap())
        .collect();

    // Generate initial guess (zero solution for now - could be improved)
    let mut y_init = Vec::with_capacity(n_points);
    for _i in 0..n_points {
        y_init.push(Array1::zeros(n_dim));
    }

    // Apply initial guess based on boundary conditions
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
        let n_dim = ya.len();

        if boundary_conditions.is_periodic {
            // For periodic boundary conditions: u(a) = u(b), u'(a) = u'(b)
            let f_a = fun(a, ya);
            let f_b = fun(b, yb);

            let mut residuals = Array1::zeros(n_dim * 2);
            for i in 0..n_dim {
                residuals[i] = ya[i] - yb[i]; // u(a) = u(b)
                residuals[i + n_dim] = f_a[i] - f_b[i]; // u'(a) = u'(b)
            }
            residuals
        } else {
            // General boundary conditions
            let f_a = fun(a, ya);
            let f_b = fun(b, yb);

            let mut residuals = Array1::zeros(n_dim * 2);

            // Left boundary conditions
            for (i, bc) in boundary_conditions.left.iter().enumerate() {
                residuals[i] = bc.evaluate_residual(a, ya, f_a.view(), i);
            }

            // Right boundary conditions
            for (i, bc) in boundary_conditions.right.iter().enumerate() {
                residuals[i + n_dim] = bc.evaluate_residual(b, yb, f_b.view(), i);
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
    pub fn convective(h: F, u_env: F) -> Self {
        Self {
            a: h,
            b: F::one(),
            c: h * u_env,
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
    // For now, this is a placeholder for multipoint BVP
    // Full implementation would require more sophisticated collocation methods

    if multipoint.interior_points.is_empty() {
        // No interior points, solve as regular BVP
        solve_bvp_extended(fun, x_span, boundary_conditions, n_points, options)
    } else {
        // TODO: Implement full multipoint BVP solver
        Err(IntegrateError::ValueError(
            "Multipoint BVP solver not yet fully implemented".to_string(),
        ))
    }
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
