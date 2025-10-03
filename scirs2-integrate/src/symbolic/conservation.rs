//! Conservation law detection and enforcement
//!
//! This module provides functionality to detect and enforce conservation
//! laws in dynamical systems, such as energy conservation in Hamiltonian
//! systems or momentum conservation in mechanical systems.

use super::expression::{simplify, SymbolicExpression, Variable};
use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use scirs2_core::ndarray::{Array1, ArrayView1};
use std::collections::HashMap;
use SymbolicExpression::{Add, Constant, Cos, Div, Exp, Ln, Mul, Neg, Pow, Sin, Sqrt, Sub, Var};

/// Represents a conservation law
#[derive(Clone)]
pub struct ConservationLaw<F: IntegrateFloat> {
    /// Name of the conserved quantity
    pub name: String,
    /// The symbolic expression for the conserved quantity
    pub expression: SymbolicExpression<F>,
    /// The expected conserved value
    pub conserved_value: Option<F>,
    /// Tolerance for conservation checking
    pub tolerance: F,
}

impl<F: IntegrateFloat> ConservationLaw<F> {
    /// Create a new conservation law
    pub fn new(name: impl Into<String>, expression: SymbolicExpression<F>, tolerance: F) -> Self {
        ConservationLaw {
            name: name.into(),
            expression,
            conserved_value: None,
            tolerance,
        }
    }

    /// Evaluate the conserved quantity at a given state
    pub fn evaluate(&self, t: F, y: ArrayView1<F>) -> IntegrateResult<F> {
        let mut values = HashMap::new();
        values.insert(Variable::new("t"), t);

        // Assume state variables are y[0], y[1], ...
        for (i, &val) in y.iter().enumerate() {
            values.insert(Variable::indexed("y", i), val);
        }

        self.expression.evaluate(&values)
    }

    /// Check if the conservation law is satisfied
    pub fn is_conserved(&self, t: F, y: ArrayView1<F>) -> IntegrateResult<bool> {
        let current_value = self.evaluate(t, y)?;

        if let Some(expected) = self.conserved_value {
            Ok((current_value - expected).abs() <= self.tolerance)
        } else {
            // If no expected value is set, we can't check conservation
            Ok(true)
        }
    }

    /// Set the conserved value based on initial conditions
    pub fn set_initial_value(&mut self, t0: F, y0: ArrayView1<F>) -> IntegrateResult<()> {
        self.conserved_value = Some(self.evaluate(t0, y0)?);
        Ok(())
    }
}

/// Detect conservation laws in an ODE system
///
/// This function analyzes the structure of the ODE system to identify
/// potential conservation laws. It looks for:
/// 1. Hamiltonian structure (energy conservation)
/// 2. Symmetries (Noether's theorem)
/// 3. Linear/quadratic invariants
#[allow(dead_code)]
pub fn detect_conservation_laws<F: IntegrateFloat>(
    expressions: &[SymbolicExpression<F>],
    state_vars: &[Variable],
) -> IntegrateResult<Vec<ConservationLaw<F>>> {
    let mut laws = Vec::new();

    // Check for Hamiltonian structure
    if let Some(hamiltonian_law) = detect_hamiltonian_conservation(expressions, state_vars)? {
        laws.push(hamiltonian_law);
    }

    // Check for linear conservation laws
    laws.extend(detect_linear_conservation(expressions, state_vars)?);

    // Check for quadratic conservation laws
    laws.extend(detect_quadratic_conservation(expressions, state_vars)?);

    Ok(laws)
}

/// Detect if the system has Hamiltonian structure
#[allow(dead_code)]
fn detect_hamiltonian_conservation<F: IntegrateFloat>(
    expressions: &[SymbolicExpression<F>],
    state_vars: &[Variable],
) -> IntegrateResult<Option<ConservationLaw<F>>> {
    use SymbolicExpression::*;

    let n = expressions.len();

    // For Hamiltonian systems, we need even dimension
    if !n.is_multiple_of(2) {
        return Ok(None);
    }

    let half_n = n / 2;

    // Check if the system has the form:
    // dq/dt = ∂H/∂p
    // dp/dt = -∂H/∂q

    // Assume first half are position variables, second half are momentum
    let q_vars: Vec<_> = state_vars[..half_n].to_vec();
    let p_vars: Vec<_> = state_vars[half_n..].to_vec();

    // Try to construct a Hamiltonian by integration
    // For a Hamiltonian system, we need:
    // dq_i/dt = ∂H/∂p_i  =>  H contains terms ∫ (dq_i/dt) dp_i
    // dp_i/dt = -∂H/∂q_i  =>  H contains terms -∫ (dp_i/dt) dq_i

    // Start with kinetic energy term (quadratic in momenta)
    let mut hamiltonian = Constant(F::zero());

    // Add kinetic energy terms by integrating dq/dt expressions
    for (i, q_expr) in expressions[..half_n].iter().enumerate() {
        // If dq/dt = p/m, then T = p²/(2m)
        // Check if expression is linear in corresponding momentum
        if let Some(coeff) = extract_linear_coefficient(q_expr, &p_vars[i]) {
            // H += p²/(2*coeff)
            hamiltonian = Add(
                Box::new(hamiltonian),
                Box::new(Div(
                    Box::new(Pow(
                        Box::new(Var(p_vars[i].clone())),
                        Box::new(Constant(F::from(2.0).unwrap())),
                    )),
                    Box::new(Mul(
                        Box::new(Constant(F::from(2.0).unwrap())),
                        Box::new(Constant(coeff)),
                    )),
                )),
            );
        }
    }

    // Add potential energy terms by integrating -dp/dt expressions
    for (i, p_expr) in expressions[half_n..].iter().enumerate() {
        // If dp/dt = -∂V/∂q, then V = -∫ (dp/dt) dq
        // For now, handle polynomial potentials
        if let Some(potential_term) =
            integrate_expression(&Neg(Box::new(p_expr.clone())), &q_vars[i])
        {
            hamiltonian = Add(Box::new(hamiltonian), Box::new(potential_term));
        }
    }

    // Verify that this is indeed a Hamiltonian by checking Hamilton's equations
    let mut is_hamiltonian = true;

    // Check dq/dt = ∂H/∂p
    for (i, q_expr) in expressions[..half_n].iter().enumerate() {
        let h_deriv_p = hamiltonian.differentiate(&p_vars[i]);
        let h_deriv_p_simplified = simplify(&h_deriv_p);
        let q_expr_simplified = simplify(q_expr);

        if !expressions_equal(&q_expr_simplified, &h_deriv_p_simplified) {
            is_hamiltonian = false;
            break;
        }
    }

    // Check dp/dt = -∂H/∂q
    if is_hamiltonian {
        for (i, p_expr) in expressions[half_n..].iter().enumerate() {
            let h_deriv_q = hamiltonian.differentiate(&q_vars[i]);
            let neg_h_deriv_q = Neg(Box::new(h_deriv_q));
            let neg_h_deriv_q_simplified = simplify(&neg_h_deriv_q);
            let p_expr_simplified = simplify(p_expr);

            if !expressions_equal(&p_expr_simplified, &neg_h_deriv_q_simplified) {
                is_hamiltonian = false;
                break;
            }
        }
    }

    if is_hamiltonian {
        Ok(Some(ConservationLaw::new(
            "Hamiltonian (Total Energy)",
            simplify(&hamiltonian),
            F::from(1e-10).unwrap(),
        )))
    } else {
        Ok(None)
    }
}

/// Extract linear coefficient if expression is linear in variable
#[allow(dead_code)]
fn extract_linear_coefficient<F: IntegrateFloat>(
    expr: &SymbolicExpression<F>,
    var: &Variable,
) -> Option<F> {
    match expr {
        Var(v) if v == var => Some(F::one()),
        Mul(a, b) => {
            // Check if one side is the variable and other is constant
            match (a.as_ref(), b.as_ref()) {
                (Var(v), Constant(c)) if v == var => Some(*c),
                (Constant(c), Var(v)) if v == var => Some(*c),
                _ => None,
            }
        }
        Div(a, b) => {
            // Check if numerator is the variable and denominator is constant
            match (a.as_ref(), b.as_ref()) {
                (Var(v), Constant(c)) if v == var => Some(F::one() / *c),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Simple symbolic integration for polynomial expressions
#[allow(dead_code)]
fn integrate_expression<F: IntegrateFloat>(
    expr: &SymbolicExpression<F>,
    var: &Variable,
) -> Option<SymbolicExpression<F>> {
    match expr {
        Constant(c) => Some(Mul(Box::new(Constant(*c)), Box::new(Var(var.clone())))),
        Var(v) if v == var => Some(Div(
            Box::new(Pow(
                Box::new(Var(var.clone())),
                Box::new(Constant(F::from(2.0).unwrap())),
            )),
            Box::new(Constant(F::from(2.0).unwrap())),
        )),
        Pow(base, exp) => {
            if let (Var(v), Constant(n)) = (base.as_ref(), exp.as_ref()) {
                if v == var && (*n + F::one()).abs() > F::epsilon() {
                    // ∫x^n dx = x^(n+1)/(n+1)
                    return Some(Div(
                        Box::new(Pow(
                            Box::new(Var(var.clone())),
                            Box::new(Constant(*n + F::one())),
                        )),
                        Box::new(Constant(*n + F::one())),
                    ));
                }
            }
            None
        }
        Mul(a, b) => {
            // Try to integrate if one factor doesn't depend on var
            if !depends_on_var(a, var) {
                if let Some(b_int) = integrate_expression(b, var) {
                    return Some(Mul(a.clone(), Box::new(b_int)));
                }
            } else if !depends_on_var(b, var) {
                if let Some(a_int) = integrate_expression(a, var) {
                    return Some(Mul(Box::new(a_int), b.clone()));
                }
            }
            None
        }
        Add(a, b) => {
            let a_int = integrate_expression(a, var)?;
            let b_int = integrate_expression(b, var)?;
            Some(Add(Box::new(a_int), Box::new(b_int)))
        }
        Sub(a, b) => {
            let a_int = integrate_expression(a, var)?;
            let b_int = integrate_expression(b, var)?;
            Some(Sub(Box::new(a_int), Box::new(b_int)))
        }
        Neg(a) => {
            let a_int = integrate_expression(a, var)?;
            Some(Neg(Box::new(a_int)))
        }
        Sin(a) => {
            if let Var(v) = a.as_ref() {
                if v == var {
                    // ∫sin(x)dx = -cos(x)
                    return Some(Neg(Box::new(Cos(Box::new(Var(var.clone()))))));
                }
            }
            None
        }
        Cos(a) => {
            if let Var(v) = a.as_ref() {
                if v == var {
                    // ∫cos(x)dx = sin(x)
                    return Some(Sin(Box::new(Var(var.clone()))));
                }
            }
            None
        }
        _ => None,
    }
}

/// Check if expression depends on variable
#[allow(dead_code)]
fn depends_on_var<F: IntegrateFloat>(expr: &SymbolicExpression<F>, var: &Variable) -> bool {
    expr.variables().contains(var)
}

/// Check if two expressions are structurally equal
#[allow(dead_code)]
fn expressions_equal<F: IntegrateFloat>(
    expr1: &SymbolicExpression<F>,
    expr2: &SymbolicExpression<F>,
) -> bool {
    match (expr1, expr2) {
        (Constant(a), Constant(b)) => (*a - *b).abs() < F::epsilon(),
        (Var(a), Var(b)) => a == b,
        (Add(a1, b1), Add(a2, b2))
        | (Sub(a1, b1), Sub(a2, b2))
        | (Mul(a1, b1), Mul(a2, b2))
        | (Div(a1, b1), Div(a2, b2))
        | (Pow(a1, b1), Pow(a2, b2)) => expressions_equal(a1, a2) && expressions_equal(b1, b2),
        (Neg(a), Neg(b))
        | (Sin(a), Sin(b))
        | (Cos(a), Cos(b))
        | (Exp(a), Exp(b))
        | (Ln(a), Ln(b))
        | (Sqrt(a), Sqrt(b)) => expressions_equal(a, b),
        _ => false,
    }
}

/// Detect linear conservation laws of the form c^T * y = constant
#[allow(dead_code)]
fn detect_linear_conservation<F: IntegrateFloat>(
    expressions: &[SymbolicExpression<F>],
    state_vars: &[Variable],
) -> IntegrateResult<Vec<ConservationLaw<F>>> {
    let mut laws = Vec::new();
    let _n = state_vars.len();

    // For linear conservation, we need c^T * f(y) = 0
    // where f is the vector field

    // Check for simple cases like sum conservation
    let mut sum_expr = Constant(F::zero());
    for var in state_vars {
        sum_expr = Add(Box::new(sum_expr), Box::new(Var(var.clone())));
    }

    // Check if d/dt(sum) = 0
    let mut sum_derivative = Constant(F::zero());
    for expr in expressions {
        sum_derivative = Add(Box::new(sum_derivative), Box::new(expr.clone()));
    }

    let simplified = simplify(&sum_derivative);
    if let Constant(val) = simplified {
        if val.abs() < F::epsilon() {
            laws.push(ConservationLaw::new(
                "Sum conservation",
                sum_expr,
                F::from(1e-10).unwrap(),
            ));
        }
    }

    Ok(laws)
}

/// Detect quadratic conservation laws
#[allow(dead_code)]
fn detect_quadratic_conservation<F: IntegrateFloat>(
    expressions: &[SymbolicExpression<F>],
    state_vars: &[Variable],
) -> IntegrateResult<Vec<ConservationLaw<F>>> {
    let mut laws = Vec::new();

    // Check for norm conservation (common in many physical systems)
    let mut norm_expr = Constant(F::zero());
    for var in state_vars {
        norm_expr = Add(
            Box::new(norm_expr),
            Box::new(Pow(
                Box::new(Var(var.clone())),
                Box::new(Constant(F::from(2.0).unwrap())),
            )),
        );
    }

    // For norm conservation, check if y^T * f(y) = 0
    let mut inner_product = Constant(F::zero());
    for (i, var) in state_vars.iter().enumerate() {
        inner_product = Add(
            Box::new(inner_product),
            Box::new(Mul(
                Box::new(Var(var.clone())),
                Box::new(expressions[i].clone()),
            )),
        );
    }

    let simplified = simplify(&inner_product);
    if let Constant(val) = simplified {
        if val.abs() < F::epsilon() {
            laws.push(ConservationLaw::new(
                "Norm conservation",
                norm_expr,
                F::from(1e-10).unwrap(),
            ));
        }
    }

    Ok(laws)
}

/// Conservation law enforcer for ODE integration
pub struct ConservationEnforcer<F: IntegrateFloat> {
    laws: Vec<ConservationLaw<F>>,
}

impl<F: IntegrateFloat> ConservationEnforcer<F> {
    /// Create a new conservation enforcer
    pub fn new(laws: Vec<ConservationLaw<F>>) -> Self {
        ConservationEnforcer { laws }
    }

    /// Initialize conservation laws with initial conditions
    pub fn initialize(&mut self, t0: F, y0: ArrayView1<F>) -> IntegrateResult<()> {
        for law in &mut self.laws {
            law.set_initial_value(t0, y0)?;
        }
        Ok(())
    }

    /// Project a state onto the conservation manifold
    pub fn project(&self, t: F, y: ArrayView1<F>) -> IntegrateResult<Array1<F>> {
        let mut y_proj = y.to_owned();

        // Simple projection: scale to maintain conservation
        // More sophisticated methods would use Lagrange multipliers
        for law in &self.laws {
            if let Some(target) = law.conserved_value {
                let current = law.evaluate(t, y_proj.view())?;
                if (current - target).abs() > law.tolerance {
                    // Simple scaling for norm-type conservation
                    if law.name.contains("Norm") && current > F::zero() {
                        let scale = (target / current).sqrt();
                        y_proj *= scale;
                    }
                }
            }
        }

        Ok(y_proj)
    }

    /// Check all conservation laws
    pub fn check_all(&self, t: F, y: ArrayView1<F>) -> IntegrateResult<Vec<(String, bool)>> {
        let mut results = Vec::new();

        for law in &self.laws {
            let is_conserved = law.is_conserved(t, y)?;
            results.push((law.name.clone(), is_conserved));
        }

        Ok(results)
    }

    /// Get conservation errors
    pub fn get_errors(&self, t: F, y: ArrayView1<F>) -> IntegrateResult<Vec<(String, F)>> {
        let mut errors = Vec::new();

        for law in &self.laws {
            if let Some(target) = law.conserved_value {
                let current = law.evaluate(t, y)?;
                errors.push((law.name.clone(), (current - target).abs()));
            }
        }

        Ok(errors)
    }
}

/// Example: Create conservation laws for a pendulum
#[allow(dead_code)]
pub fn example_pendulum_conservation<F: IntegrateFloat>() -> Vec<ConservationLaw<F>> {
    // For a pendulum with state [theta, omega]
    // Energy = 0.5 * omega^2 - cos(theta)

    let theta = Var(Variable::indexed("y", 0));
    let omega = Var(Variable::indexed("y", 1));

    let energy = Sub(
        Box::new(Mul(
            Box::new(Constant(F::from(0.5).unwrap())),
            Box::new(Pow(
                Box::new(omega),
                Box::new(Constant(F::from(2.0).unwrap())),
            )),
        )),
        Box::new(Cos(Box::new(theta))),
    );

    vec![ConservationLaw::new(
        "Total Energy",
        energy,
        F::from(1e-10).unwrap(),
    )]
}

/// Example: Create conservation laws for N-body gravitational system
#[allow(dead_code)]
pub fn example_nbody_conservation<F: IntegrateFloat>(n: usize) -> Vec<ConservationLaw<F>> {
    let mut laws = Vec::new();

    // State vector: [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2, ...]

    // Total linear momentum conservation
    let mut px = Constant(F::zero());
    let mut py = Constant(F::zero());
    let mut pz = Constant(F::zero());

    // Total angular momentum conservation
    let mut lx = Constant(F::zero());
    let mut ly = Constant(F::zero());
    let mut lz = Constant(F::zero());

    for i in 0..n {
        let base = i * 6;
        let x = Var(Variable::indexed("y", base));
        let y = Var(Variable::indexed("y", base + 1));
        let z = Var(Variable::indexed("y", base + 2));
        let vx = Var(Variable::indexed("y", base + 3));
        let vy = Var(Variable::indexed("y", base + 4));
        let vz = Var(Variable::indexed("y", base + 5));

        // Linear momentum (assuming unit masses)
        px = Add(Box::new(px), Box::new(vx.clone()));
        py = Add(Box::new(py), Box::new(vy.clone()));
        pz = Add(Box::new(pz), Box::new(vz.clone()));

        // Angular momentum L = r × v
        lx = Add(
            Box::new(lx),
            Box::new(Sub(
                Box::new(Mul(Box::new(y.clone()), Box::new(vz.clone()))),
                Box::new(Mul(Box::new(z.clone()), Box::new(vy.clone()))),
            )),
        );
        ly = Add(
            Box::new(ly),
            Box::new(Sub(
                Box::new(Mul(Box::new(z.clone()), Box::new(vx.clone()))),
                Box::new(Mul(Box::new(x.clone()), Box::new(vz))),
            )),
        );
        lz = Add(
            Box::new(lz),
            Box::new(Sub(
                Box::new(Mul(Box::new(x), Box::new(vy))),
                Box::new(Mul(Box::new(y), Box::new(vx))),
            )),
        );
    }

    laws.push(ConservationLaw::new(
        "Linear Momentum X",
        px,
        F::from(1e-10).unwrap(),
    ));
    laws.push(ConservationLaw::new(
        "Linear Momentum Y",
        py,
        F::from(1e-10).unwrap(),
    ));
    laws.push(ConservationLaw::new(
        "Linear Momentum Z",
        pz,
        F::from(1e-10).unwrap(),
    ));
    laws.push(ConservationLaw::new(
        "Angular Momentum X",
        lx,
        F::from(1e-10).unwrap(),
    ));
    laws.push(ConservationLaw::new(
        "Angular Momentum Y",
        ly,
        F::from(1e-10).unwrap(),
    ));
    laws.push(ConservationLaw::new(
        "Angular Momentum Z",
        lz,
        F::from(1e-10).unwrap(),
    ));

    laws
}

/// Example: Create conservation laws for a coupled oscillator system
#[allow(dead_code)]
pub fn example_coupled_oscillators<F: IntegrateFloat>(n: usize) -> Vec<ConservationLaw<F>> {
    let mut laws = Vec::new();

    // State: [x1, v1, x2, v2, ..., xn, vn]
    // Total energy = kinetic + potential
    let mut energy = Constant(F::zero());

    // Kinetic energy
    for i in 0..n {
        let v = Var(Variable::indexed("y", 2 * i + 1));
        energy = Add(
            Box::new(energy),
            Box::new(Mul(
                Box::new(Constant(F::from(0.5).unwrap())),
                Box::new(Pow(Box::new(v), Box::new(Constant(F::from(2.0).unwrap())))),
            )),
        );
    }

    // Potential energy (nearest neighbor coupling)
    for i in 0..n {
        let x = Var(Variable::indexed("y", 2 * i));

        // On-site potential
        energy = Add(
            Box::new(energy),
            Box::new(Mul(
                Box::new(Constant(F::from(0.5).unwrap())),
                Box::new(Pow(
                    Box::new(x.clone()),
                    Box::new(Constant(F::from(2.0).unwrap())),
                )),
            )),
        );

        // Coupling potential
        if i < n - 1 {
            let x_next = Var(Variable::indexed("y", 2 * (i + 1)));
            let diff = Sub(Box::new(x_next), Box::new(x));
            energy = Add(
                Box::new(energy),
                Box::new(Mul(
                    Box::new(Constant(F::from(0.5).unwrap())),
                    Box::new(Pow(
                        Box::new(diff),
                        Box::new(Constant(F::from(2.0).unwrap())),
                    )),
                )),
            );
        }
    }

    laws.push(ConservationLaw::new(
        "Total Energy",
        energy,
        F::from(1e-10).unwrap(),
    ));

    // For periodic boundary conditions, add momentum conservation
    let mut total_momentum = Constant(F::zero());
    for i in 0..n {
        let v = Var(Variable::indexed("y", 2 * i + 1));
        total_momentum = Add(Box::new(total_momentum), Box::new(v));
    }
    laws.push(ConservationLaw::new(
        "Total Momentum",
        total_momentum,
        F::from(1e-10).unwrap(),
    ));

    laws
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ConservationLaw,
        SymbolicExpression::{Add, Constant, Pow, Var},
        Variable,
    };
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_conservation_evaluation() {
        // Test norm conservation: x^2 + y^2
        let x = Var(Variable::indexed("y", 0));
        let y = Var(Variable::indexed("y", 1));

        let norm = Add(
            Box::new(Pow(Box::new(x), Box::new(Constant(2.0)))),
            Box::new(Pow(Box::new(y), Box::new(Constant(2.0)))),
        );

        let law = ConservationLaw::new("Norm", norm, 1e-10);

        let state = Array1::from_vec(vec![3.0, 4.0]);
        let value = law.evaluate(0.0, state.view()).unwrap();

        assert!((value - 25.0_f64).abs() < 1e-10); // 3^2 + 4^2 = 25
    }
}
