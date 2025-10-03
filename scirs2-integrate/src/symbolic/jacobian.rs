//! Automatic Jacobian generation using symbolic differentiation
//!
//! This module provides functionality for automatically generating
//! Jacobian matrices from symbolic expressions, eliminating the need
//! for finite difference approximations.

use super::expression::{simplify, SymbolicExpression, Variable};
use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array2, ArrayView1};
use std::collections::HashMap;

// Helper functions for creating symbolic expressions
#[allow(dead_code)]
fn var<F: IntegrateFloat>(name: &str) -> SymbolicExpression<F> {
    SymbolicExpression::var(name)
}

#[allow(dead_code)]
fn indexed_var<F: IntegrateFloat>(name: &str, index: usize) -> SymbolicExpression<F> {
    SymbolicExpression::indexedvar(name, index)
}

#[allow(dead_code)]
fn constant<F: IntegrateFloat>(value: F) -> SymbolicExpression<F> {
    SymbolicExpression::constant(value)
}

/// Represents a symbolic Jacobian matrix
pub struct SymbolicJacobian<F: IntegrateFloat> {
    /// The symbolic expressions for each element of the Jacobian
    pub elements: Array2<SymbolicExpression<F>>,
    /// The state variables with respect to which we differentiate
    pub state_vars: Vec<Variable>,
    /// The time variable (if time-dependent)
    pub time_var: Option<Variable>,
}

impl<F: IntegrateFloat> SymbolicJacobian<F> {
    /// Create a new symbolic Jacobian
    pub fn new(
        elements: Array2<SymbolicExpression<F>>,
        state_vars: Vec<Variable>,
        time_var: Option<Variable>,
    ) -> Self {
        SymbolicJacobian {
            elements,
            state_vars,
            time_var,
        }
    }

    /// Evaluate the Jacobian at given state values
    pub fn evaluate(&self, t: F, y: ArrayView1<F>) -> IntegrateResult<Array2<F>> {
        let n = self.state_vars.len();
        if y.len() != n {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} states, got {}",
                n,
                y.len()
            )));
        }

        // Build value map
        let mut values = HashMap::new();
        for (i, var) in self.state_vars.iter().enumerate() {
            values.insert(var.clone(), y[i]);
        }
        if let Some(ref t_var) = self.time_var {
            values.insert(t_var.clone(), t);
        }

        // Evaluate each element
        let (rows, cols) = self.elements.dim();
        let mut result = Array2::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                result[[i, j]] = self.elements[[i, j]].evaluate(&values)?;
            }
        }

        Ok(result)
    }

    /// Simplify all expressions in the Jacobian
    pub fn simplify(&mut self) {
        let (rows, cols) = self.elements.dim();
        for i in 0..rows {
            for j in 0..cols {
                self.elements[[i, j]] = simplify(&self.elements[[i, j]]);
            }
        }
    }
}

/// Generate a symbolic Jacobian from a vector of symbolic expressions
///
/// # Arguments
/// * `expressions` - Vector of symbolic expressions representing the ODE system
/// * `state_vars` - Variables with respect to which to differentiate
/// * `time_var` - Optional time variable
///
/// # Returns
/// A symbolic Jacobian matrix where J[i,j] = ∂f[i]/∂y[j]
#[allow(dead_code)]
pub fn generate_jacobian<F: IntegrateFloat>(
    expressions: &[SymbolicExpression<F>],
    state_vars: &[Variable],
    time_var: Option<Variable>,
) -> IntegrateResult<SymbolicJacobian<F>> {
    let n = expressions.len();
    let m = state_vars.len();

    if n == 0 || m == 0 {
        return Err(IntegrateError::ValueError(
            "Empty expressions or state variables".to_string(),
        ));
    }

    let mut jacobian = Array2::from_elem((n, m), SymbolicExpression::Constant(F::zero()));

    // Compute partial derivatives
    for (i, expr) in expressions.iter().enumerate() {
        for (j, var) in state_vars.iter().enumerate() {
            jacobian[[i, j]] = expr.differentiate(var);
        }
    }

    Ok(SymbolicJacobian::new(
        jacobian,
        state_vars.to_vec(),
        time_var,
    ))
}

/// Builder for creating symbolic ODE systems
pub struct SymbolicODEBuilder<F: IntegrateFloat> {
    expressions: Vec<SymbolicExpression<F>>,
    state_vars: Vec<Variable>,
    time_var: Option<Variable>,
}

impl<F: IntegrateFloat> SymbolicODEBuilder<F> {
    /// Create a new builder
    pub fn new() -> Self {
        SymbolicODEBuilder {
            expressions: Vec::new(),
            state_vars: Vec::new(),
            time_var: None,
        }
    }

    /// Set the number of state variables
    pub fn with_state_vars(mut self, n: usize) -> Self {
        self.state_vars = (0..n).map(|i| Variable::indexed("y", i)).collect();
        self
    }

    /// Set custom state variable names
    pub fn with_named_vars(mut self, names: Vec<String>) -> Self {
        self.state_vars = names.into_iter().map(Variable::new).collect();
        self
    }

    /// Enable time dependence
    pub fn with_time(mut self) -> Self {
        self.time_var = Some(Variable::new("t"));
        self
    }

    /// Add an ODE expression
    pub fn add_equation(&mut self, expr: SymbolicExpression<F>) -> &mut Self {
        self.expressions.push(expr);
        self
    }

    /// Build the symbolic Jacobian
    pub fn build_jacobian(&self) -> IntegrateResult<SymbolicJacobian<F>> {
        generate_jacobian(&self.expressions, &self.state_vars, self.time_var.clone())
    }
}

impl<F: IntegrateFloat> Default for SymbolicODEBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Example: Create a symbolic Jacobian for the Van der Pol oscillator
#[allow(dead_code)]
pub fn example_van_der_pol<F: IntegrateFloat>(mu: F) -> IntegrateResult<SymbolicJacobian<F>> {
    use SymbolicExpression::*;

    // Variables: y[0] = x, y[1] = x'
    let y0 = Var(Variable::indexed("y", 0));
    let y1 = Var(Variable::indexed("y", 1));

    // Van der Pol equations:
    // dy[0]/dt = y[1]
    // dy[1]/dt = _mu * (1 - y[0]^2) * y[1] - y[0]

    let expr1 = y1.clone();
    let expr2 = Sub(
        Box::new(Mul(
            Box::new(Mul(
                Box::new(Constant(mu)),
                Box::new(Sub(
                    Box::new(Constant(F::one())),
                    Box::new(Pow(
                        Box::new(y0.clone()),
                        Box::new(Constant(F::from(2.0).unwrap())),
                    )),
                )),
            )),
            Box::new(y1),
        )),
        Box::new(y0),
    );

    SymbolicODEBuilder::new()
        .with_state_vars(2)
        .add_equation(expr1)
        .add_equation(expr2)
        .build_jacobian()
}

/// Example: Create a symbolic Jacobian for a stiff chemical reaction system
#[allow(dead_code)]
pub fn example_stiff_chemical<F: IntegrateFloat>() -> IntegrateResult<SymbolicJacobian<F>> {
    // Robertson's chemical reaction problem (stiff ODE)
    // dy1/dt = -0.04*y1 + 1e4*y2*y3
    // dy2/dt = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
    // dy3/dt = 3e7*y2^2

    let y1 = SymbolicExpression::indexedvar("y", 0);
    let y2 = SymbolicExpression::indexedvar("y", 1);
    let y3 = SymbolicExpression::indexedvar("y", 2);

    let k1 = SymbolicExpression::constant(F::from(0.04).unwrap());
    let k2 = SymbolicExpression::constant(F::from(1e4).unwrap());
    let k3 = SymbolicExpression::constant(F::from(3e7).unwrap());

    // Using operator overloading for cleaner syntax
    let expr1 = -k1.clone() * y1.clone() + k2.clone() * y2.clone() * y3.clone();
    let expr2 = k1 * y1 - k2 * y2.clone() * y3 - k3.clone() * y2.clone() * y2.clone();
    let expr3 = k3 * y2.clone() * y2;

    SymbolicODEBuilder::new()
        .with_state_vars(3)
        .add_equation(expr1)
        .add_equation(expr2)
        .add_equation(expr3)
        .build_jacobian()
}

/// Example: Create a symbolic Jacobian for a predator-prey system with seasonal effects
#[allow(dead_code)]
pub fn example_seasonal_predator_prey<F: IntegrateFloat>() -> IntegrateResult<SymbolicJacobian<F>> {
    // Lotka-Volterra with seasonal variation
    // dx/dt = a*x*(1 + b*sin(2π*t)) - c*x*y
    // dy/dt = -d*y + e*x*y

    let x = indexed_var("y", 0);
    let y = indexed_var("y", 1);
    let t = var("t");

    let a = constant(F::from(1.5).unwrap());
    let b = constant(F::from(0.1).unwrap());
    let c = constant(F::from(0.5).unwrap());
    let d = constant(F::from(0.75).unwrap());
    let e = constant(F::from(0.25).unwrap());
    let two_pi = constant(F::from(std::f64::consts::TAU).unwrap());

    // Seasonal growth term: 1 + b*sin(2π*t)
    let seasonal = constant(F::one()) + b * SymbolicExpression::Sin(Box::new(two_pi * t));

    let expr1 = a * x.clone() * seasonal - c * x.clone() * y.clone();
    let expr2 = -d * y.clone() + e * x * y;

    let mut builder = SymbolicODEBuilder::new().with_state_vars(2).with_time();
    builder.add_equation(expr1);
    builder.add_equation(expr2);

    builder.build_jacobian()
}

/// Integration with the ODE solver autodiff module
#[cfg(feature = "autodiff")]
#[allow(dead_code)]
pub fn create_autodiff_jacobian<F, Func>(
    symbolic_jacobian: &SymbolicJacobian<F>,
) -> impl Fn(F, ArrayView1<F>) -> IntegrateResult<Array2<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> IntegrateResult<ArrayView1<F>>,
{
    let jac = symbolic_jacobian.clone();
    move |t: F, y: ArrayView1<F>| jac.evaluate(t, y)
}

impl<F: IntegrateFloat> Clone for SymbolicJacobian<F> {
    fn clone(&self) -> Self {
        SymbolicJacobian {
            elements: self.elements.clone(),
            state_vars: self.state_vars.clone(),
            time_var: self.time_var.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        generate_jacobian,
        SymbolicExpression::{Neg, Var},
        Variable,
    };
    use scirs2_core::ndarray::ArrayView1;
    use std::collections::HashMap;

    #[test]
    fn test_simple_jacobian() {
        // System: dy/dt = -y
        let y = Var(Variable::new("y"));
        let expr = Neg(Box::new(y));

        let jacobian = generate_jacobian(&[expr], &[Variable::new("y")], None).unwrap();

        // Jacobian should be [[-1]]
        let mut values = HashMap::new();
        values.insert(Variable::new("y"), 1.0);

        let j = jacobian.evaluate(0.0, ArrayView1::from(&[1.0])).unwrap();
        assert_eq!(j.dim(), (1, 1));
        assert!((j[[0, 0]] + 1.0_f64).abs() < 1e-10);
    }
}
