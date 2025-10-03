//! Higher-order ODE to first-order system conversion
//!
//! This module provides functionality to automatically convert higher-order
//! ODEs into systems of first-order ODEs, which is required by most
//! numerical integration methods.

use super::expression::{SymbolicExpression, Variable};
use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use std::collections::HashMap;
use SymbolicExpression::{Add, Constant, Cos, Div, Exp, Ln, Mul, Neg, Pow, Sin, Sqrt, Sub, Var};

/// Represents a higher-order ODE
pub struct HigherOrderODE<F: IntegrateFloat> {
    /// The order of the ODE
    pub order: usize,
    /// The dependent variable name
    pub dependent_var: String,
    /// The independent variable name (usually time)
    pub independent_var: String,
    /// The symbolic expression for the highest derivative
    /// as a function of lower derivatives and the independent variable
    pub expression: SymbolicExpression<F>,
}

impl<F: IntegrateFloat> HigherOrderODE<F> {
    /// Create a new higher-order ODE
    pub fn new(
        order: usize,
        dependent_var: impl Into<String>,
        independent_var: impl Into<String>,
        expression: SymbolicExpression<F>,
    ) -> IntegrateResult<Self> {
        if order == 0 {
            return Err(IntegrateError::ValueError(
                "ODE order must be at least 1".to_string(),
            ));
        }

        Ok(HigherOrderODE {
            order,
            dependent_var: dependent_var.into(),
            independent_var: independent_var.into(),
            expression,
        })
    }

    /// Get the state variable names for the first-order system
    pub fn state_variables(&self) -> Vec<Variable> {
        (0..self.order)
            .map(|i| Variable::indexed(&self.dependent_var, i))
            .collect()
    }
}

/// Result of converting a higher-order ODE to first-order system
pub struct FirstOrderSystem<F: IntegrateFloat> {
    /// The state variables (y[0] = x, y[1] = x', y[2] = x'', etc.)
    pub state_vars: Vec<Variable>,
    /// The expressions for dy/dt
    pub expressions: Vec<SymbolicExpression<F>>,
    /// Mapping from derivative notation to state variables
    pub variable_map: HashMap<String, Variable>,
}

impl<F: IntegrateFloat> FirstOrderSystem<F> {
    /// Convert to a function suitable for ODE solvers
    pub fn to_function(&self) -> impl Fn(F, ArrayView1<F>) -> IntegrateResult<Array1<F>> {
        let expressions = self.expressions.clone();
        let state_vars = self.state_vars.clone();

        move |t: F, y: ArrayView1<F>| {
            if y.len() != state_vars.len() {
                return Err(IntegrateError::DimensionMismatch(format!(
                    "Expected {} states, got {}",
                    state_vars.len(),
                    y.len()
                )));
            }

            // Build value map
            let mut values = HashMap::new();
            for (i, var) in state_vars.iter().enumerate() {
                values.insert(var.clone(), y[i]);
            }
            values.insert(Variable::new("t"), t);

            // Evaluate expressions
            let mut result = Array1::zeros(expressions.len());
            for (i, expr) in expressions.iter().enumerate() {
                result[i] = expr.evaluate(&values)?;
            }

            Ok(result)
        }
    }
}

/// Convert a higher-order ODE to a first-order system
///
/// # Arguments
/// * `ode` - The higher-order ODE to convert
///
/// # Returns
/// A first-order system equivalent to the input ODE
///
/// # Example
/// For a second-order ODE: x'' = -x - 0.1*x'
/// This converts to:
/// - y[0] = x
/// - y[1] = x'
/// - dy[0]/dt = y[1]
/// - dy[1]/dt = -y[0] - 0.1*y[1]
#[allow(dead_code)]
pub fn higher_order_to_first_order<F: IntegrateFloat>(
    ode: &HigherOrderODE<F>,
) -> IntegrateResult<FirstOrderSystem<F>> {
    use SymbolicExpression::*;

    let mut state_vars = Vec::new();
    let mut expressions = Vec::new();
    let mut variable_map = HashMap::new();

    // Create state variables for each derivative
    for i in 0..ode.order {
        let var = Variable::indexed(&ode.dependent_var, i);
        state_vars.push(var.clone());

        // Map derivative notation to state variable
        let deriv_notation = match i {
            0 => ode.dependent_var.clone(),
            1 => format!("{}'", ode.dependent_var),
            n => format!("{}^({})", ode.dependent_var, n),
        };
        variable_map.insert(deriv_notation, var);
    }

    // Create expressions for the first-order system
    // dy[i]/dt = y[i+1] for i < order-1
    for i in 0..ode.order - 1 {
        expressions.push(Var(state_vars[i + 1].clone()));
    }

    // For the highest derivative, substitute state variables
    let mut highest_deriv_expr = ode.expression.clone();
    highest_deriv_expr = substitute_derivatives(&highest_deriv_expr, &variable_map);
    expressions.push(highest_deriv_expr);

    Ok(FirstOrderSystem {
        state_vars,
        expressions,
        variable_map,
    })
}

/// Substitute derivative notation with state variables in an expression
#[allow(dead_code)]
fn substitute_derivatives<F: IntegrateFloat>(
    expr: &SymbolicExpression<F>,
    variable_map: &HashMap<String, Variable>,
) -> SymbolicExpression<F> {
    match expr {
        Var(v) => {
            // Check if this variable should be substituted
            if let Some(state_var) = variable_map.get(&v.name) {
                Var(state_var.clone())
            } else {
                expr.clone()
            }
        }
        Add(a, b) => Add(
            Box::new(substitute_derivatives(a, variable_map)),
            Box::new(substitute_derivatives(b, variable_map)),
        ),
        Sub(a, b) => Sub(
            Box::new(substitute_derivatives(a, variable_map)),
            Box::new(substitute_derivatives(b, variable_map)),
        ),
        Mul(a, b) => Mul(
            Box::new(substitute_derivatives(a, variable_map)),
            Box::new(substitute_derivatives(b, variable_map)),
        ),
        Div(a, b) => Div(
            Box::new(substitute_derivatives(a, variable_map)),
            Box::new(substitute_derivatives(b, variable_map)),
        ),
        Pow(a, b) => Pow(
            Box::new(substitute_derivatives(a, variable_map)),
            Box::new(substitute_derivatives(b, variable_map)),
        ),
        Neg(a) => Neg(Box::new(substitute_derivatives(a, variable_map))),
        Sin(a) => Sin(Box::new(substitute_derivatives(a, variable_map))),
        Cos(a) => Cos(Box::new(substitute_derivatives(a, variable_map))),
        Exp(a) => Exp(Box::new(substitute_derivatives(a, variable_map))),
        Ln(a) => Ln(Box::new(substitute_derivatives(a, variable_map))),
        Sqrt(a) => Sqrt(Box::new(substitute_derivatives(a, variable_map))),
        _ => expr.clone(),
    }
}

/// Example: Convert a damped harmonic oscillator to first-order system
#[allow(dead_code)]
pub fn example_damped_oscillator<F: IntegrateFloat>(
    omega: F,
    damping: F,
) -> IntegrateResult<FirstOrderSystem<F>> {
    // Second-order ODE: x'' + 2*damping*x' + omega^2*x = 0
    // Rearranged: x'' = -2*damping*x' - omega^2*x

    let x = Var(Variable::new("x"));
    let x_prime = Var(Variable::new("x'"));

    let expression = Neg(Box::new(Add(
        Box::new(Mul(
            Box::new(Mul(
                Box::new(Constant(F::from(2.0).unwrap())),
                Box::new(Constant(damping)),
            )),
            Box::new(x_prime),
        )),
        Box::new(Mul(
            Box::new(Pow(
                Box::new(Constant(omega)),
                Box::new(Constant(F::from(2.0).unwrap())),
            )),
            Box::new(x),
        )),
    )));

    let ode = HigherOrderODE::new(2, "x", "t", expression)?;
    higher_order_to_first_order(&ode)
}

/// Example: Convert a driven pendulum equation to first-order system
#[allow(dead_code)]
pub fn example_driven_pendulum<F: IntegrateFloat>(
    g: F,     // gravity
    l: F,     // length
    gamma: F, // damping coefficient
    a: F,     // driving amplitude
    omega: F, // driving frequency
) -> IntegrateResult<FirstOrderSystem<F>> {
    // Pendulum equation: θ'' + (g/l)*sin(θ) + γ*θ' = A*cos(ω*t)
    // Rearranged: θ'' = -γ*θ' - (g/l)*sin(θ) + A*cos(ω*t)

    let theta = SymbolicExpression::var("θ");
    let theta_prime = SymbolicExpression::var("θ'");
    let t = SymbolicExpression::var("t");

    let g_over_l = SymbolicExpression::constant(g / l);
    let gamma_const = SymbolicExpression::constant(gamma);
    let a_const = SymbolicExpression::constant(a);
    let omega_const = SymbolicExpression::constant(omega);

    // Using operator overloading
    let damping_term = -gamma_const * theta_prime;
    let gravity_term = -g_over_l * SymbolicExpression::Sin(Box::new(theta));
    let driving_term = a_const * SymbolicExpression::Cos(Box::new(omega_const * t));

    let expression = damping_term + gravity_term + driving_term;

    let ode = HigherOrderODE::new(2, "θ", "t", expression)?;
    higher_order_to_first_order(&ode)
}

/// Example: Convert a beam equation (4th order) to first-order system
#[allow(dead_code)]
pub fn example_euler_bernoulli_beam<F: IntegrateFloat>(
    ei: F,     // flexural rigidity
    _rho_a: F, // mass per unit length
    f: F,      // distributed load
) -> IntegrateResult<FirstOrderSystem<F>> {
    // Euler-Bernoulli beam equation: EI*w'''' + ρA*∂²w/∂t² = f(x,t)
    // For static case: EI*w'''' = f(x)
    // Rearranged: w'''' = f/(EI)

    let f_over_ei = SymbolicExpression::constant(f / ei);

    let ode = HigherOrderODE::new(4, "w", "x", f_over_ei)?;
    higher_order_to_first_order(&ode)
}

/// Convert a system of higher-order ODEs to first-order
pub struct SystemConverter<F: IntegrateFloat> {
    odes: Vec<HigherOrderODE<F>>,
    total_states: usize,
}

impl<F: IntegrateFloat> SystemConverter<F> {
    /// Create a new system converter
    pub fn new() -> Self {
        SystemConverter {
            odes: Vec::new(),
            total_states: 0,
        }
    }

    /// Add a higher-order ODE to the system
    pub fn add_ode(&mut self, ode: HigherOrderODE<F>) -> &mut Self {
        self.total_states += ode.order;
        self.odes.push(ode);
        self
    }

    /// Convert the entire system to first-order
    pub fn convert(&self) -> IntegrateResult<FirstOrderSystem<F>> {
        let mut all_state_vars = Vec::new();
        let mut all_expressions = Vec::new();
        let mut all_variable_map = HashMap::new();

        for ode in &self.odes {
            let system = higher_order_to_first_order(ode)?;
            all_state_vars.extend(system.state_vars);
            all_expressions.extend(system.expressions);
            all_variable_map.extend(system.variable_map);
        }

        Ok(FirstOrderSystem {
            state_vars: all_state_vars,
            expressions: all_expressions,
            variable_map: all_variable_map,
        })
    }
}

impl<F: IntegrateFloat> Default for SystemConverter<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        higher_order_to_first_order, HigherOrderODE, SymbolicExpression,
        SymbolicExpression::{Neg, Var},
        Variable,
    };

    #[test]
    fn test_second_order_conversion() {
        // Test x'' = -x
        let x: SymbolicExpression<f64> = Var(Variable::new("x"));
        let expr = Neg(Box::new(x));

        let ode = HigherOrderODE::new(2, "x", "t", expr).unwrap();
        let system = higher_order_to_first_order(&ode).unwrap();

        assert_eq!(system.state_vars.len(), 2);
        assert_eq!(system.expressions.len(), 2);

        // Check that dy[0]/dt = y[1]
        if let Var(v) = &system.expressions[0] {
            assert_eq!(v.name, "x");
            assert_eq!(v.index, Some(1));
        } else {
            panic!(
                "Expected variable expression, got {:?}",
                system.expressions[0]
            );
        }
    }
}
