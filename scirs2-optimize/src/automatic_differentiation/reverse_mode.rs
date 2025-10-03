//! Reverse-mode automatic differentiation (backpropagation)
//!
//! Reverse-mode AD is efficient for computing derivatives when the number of
//! output variables is small (typically 1 for optimization). It builds a
//! computational graph and then propagates derivatives backwards.

use crate::automatic_differentiation::tape::{
    BinaryOpType, ComputationTape, TapeNode, UnaryOpType, Variable,
};
use crate::error::OptimizeError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// Options for reverse-mode automatic differentiation
#[derive(Debug, Clone)]
pub struct ReverseADOptions {
    /// Whether to compute gradient
    pub compute_gradient: bool,
    /// Whether to compute Hessian
    pub compute_hessian: bool,
    /// Maximum tape size to prevent memory issues
    pub max_tape_size: usize,
    /// Enable tape optimization
    pub optimize_tape: bool,
}

impl Default for ReverseADOptions {
    fn default() -> Self {
        Self {
            compute_gradient: true,
            compute_hessian: false,
            max_tape_size: 1_000_000,
            optimize_tape: true,
        }
    }
}

/// Variable in the computational graph for reverse-mode AD
#[derive(Debug, Clone)]
pub struct ReverseVariable {
    /// Variable index in the tape
    pub index: usize,
    /// Current value
    pub value: f64,
    /// Accumulated gradient (adjoint)
    pub grad: f64,
}

impl ReverseVariable {
    /// Create a new variable
    pub fn new(index: usize, value: f64) -> Self {
        Self {
            index,
            value,
            grad: 0.0,
        }
    }

    /// Create a constant variable (not in tape)
    pub fn constant(value: f64) -> Self {
        Self {
            index: usize::MAX, // Special index for constants
            value,
            grad: 0.0,
        }
    }

    /// Check if this is a constant
    pub fn is_constant(&self) -> bool {
        self.index == usize::MAX
    }

    /// Get the value
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Get the gradient
    pub fn grad(&self) -> f64 {
        self.grad
    }

    /// Set the gradient (used internally by backpropagation)
    pub fn set_grad(&mut self, grad: f64) {
        self.grad = grad;
    }

    /// Add to the gradient (used internally by backpropagation)
    pub fn add_grad(&mut self, grad: f64) {
        self.grad += grad;
    }

    /// Reset gradient to zero
    pub fn zero_grad(&mut self) {
        self.grad = 0.0;
    }

    /// Create a variable from a scalar (convenience method)
    pub fn from_scalar(value: f64) -> Self {
        Self::constant(value)
    }

    /// Power operation (simple version without graph context)
    pub fn powi(&self, n: i32) -> Self {
        if self.is_constant() {
            ReverseVariable::constant(self.value.powi(n))
        } else {
            ReverseVariable {
                index: self.index,
                value: self.value.powi(n),
                grad: 0.0,
            }
        }
    }

    /// Exponential operation (simple version without graph context)
    pub fn exp(&self) -> Self {
        if self.is_constant() {
            ReverseVariable::constant(self.value.exp())
        } else {
            ReverseVariable {
                index: self.index,
                value: self.value.exp(),
                grad: 0.0,
            }
        }
    }

    /// Natural logarithm operation (simple version without graph context)
    pub fn ln(&self) -> Self {
        if self.is_constant() {
            ReverseVariable::constant(self.value.ln())
        } else {
            ReverseVariable {
                index: self.index,
                value: self.value.ln(),
                grad: 0.0,
            }
        }
    }

    /// Sine operation (simple version without graph context)
    pub fn sin(&self) -> Self {
        if self.is_constant() {
            ReverseVariable::constant(self.value.sin())
        } else {
            ReverseVariable {
                index: self.index,
                value: self.value.sin(),
                grad: 0.0,
            }
        }
    }

    /// Cosine operation (simple version without graph context)
    pub fn cos(&self) -> Self {
        if self.is_constant() {
            ReverseVariable::constant(self.value.cos())
        } else {
            ReverseVariable {
                index: self.index,
                value: self.value.cos(),
                grad: 0.0,
            }
        }
    }

    /// Tangent operation (simple version without graph context)
    pub fn tan(&self) -> Self {
        if self.is_constant() {
            ReverseVariable::constant(self.value.tan())
        } else {
            ReverseVariable {
                index: self.index,
                value: self.value.tan(),
                grad: 0.0,
            }
        }
    }

    /// Square root operation (simple version without graph context)
    pub fn sqrt(&self) -> Self {
        if self.is_constant() {
            ReverseVariable::constant(self.value.sqrt())
        } else {
            ReverseVariable {
                index: self.index,
                value: self.value.sqrt(),
                grad: 0.0,
            }
        }
    }

    /// Absolute value operation (simple version without graph context)
    pub fn abs(&self) -> Self {
        if self.is_constant() {
            ReverseVariable::constant(self.value.abs())
        } else {
            ReverseVariable {
                index: self.index,
                value: self.value.abs(),
                grad: 0.0,
            }
        }
    }
}

/// Computational graph for reverse-mode AD
pub struct ComputationGraph {
    /// Computation tape
    tape: ComputationTape,
    /// Current variable counter
    var_counter: usize,
    /// Variable values
    values: Vec<f64>,
    /// Variable gradients (adjoints)
    gradients: Vec<f64>,
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputationGraph {
    /// Create a new computation graph
    pub fn new() -> Self {
        Self {
            tape: ComputationTape::new(),
            var_counter: 0,
            values: Vec::new(),
            gradients: Vec::new(),
        }
    }

    /// Create a new variable in the graph
    pub fn variable(&mut self, value: f64) -> ReverseVariable {
        let index = self.var_counter;
        self.var_counter += 1;

        self.values.push(value);
        self.gradients.push(0.0);

        self.tape.add_input(Variable::new(index, value));

        ReverseVariable::new(index, value)
    }

    /// Add a binary operation to the tape
    fn add_binary_op(
        &mut self,
        op_type: BinaryOpType,
        left: &ReverseVariable,
        right: &ReverseVariable,
        result_value: f64,
        left_grad: f64,
        right_grad: f64,
    ) -> ReverseVariable {
        let result_index = self.var_counter;
        self.var_counter += 1;

        self.values.push(result_value);
        self.gradients.push(0.0);

        // Add to tape
        let node = TapeNode::BinaryOp {
            op_type,
            left: left.index,
            right: right.index,
            result: result_index,
            left_partial: left_grad,
            right_partial: right_grad,
        };

        self.tape.add_node(node);

        ReverseVariable::new(result_index, result_value)
    }

    /// Add a unary operation to the tape
    fn add_unary_op(
        &mut self,
        op_type: UnaryOpType,
        input: &ReverseVariable,
        result_value: f64,
        input_grad: f64,
    ) -> ReverseVariable {
        let result_index = self.var_counter;
        self.var_counter += 1;

        self.values.push(result_value);
        self.gradients.push(0.0);

        // Add to tape
        let node = TapeNode::UnaryOp {
            op_type,
            input: input.index,
            result: result_index,
            partial: input_grad,
        };

        self.tape.add_node(node);

        ReverseVariable::new(result_index, result_value)
    }

    /// Perform backpropagation to compute gradients
    pub fn backward(&mut self, output_var: &ReverseVariable) -> Result<(), OptimizeError> {
        // Initialize output gradient to 1
        if !output_var.is_constant() {
            self.gradients[output_var.index] = 1.0;
        }

        // Reverse pass through the tape
        let _ = self.tape.backward(&mut self.gradients);

        Ok(())
    }

    /// Get gradient for a variable
    pub fn get_gradient(&self, var: &ReverseVariable) -> f64 {
        if var.is_constant() {
            0.0
        } else {
            self.gradients[var.index]
        }
    }

    /// Clear gradients for next computation
    pub fn zero_gradients(&mut self) {
        for grad in &mut self.gradients {
            *grad = 0.0;
        }
    }
}

// Arithmetic operations for ReverseVariable
// Note: These implementations are for simple cases without graph context.
// For full AD functionality, use the graph-based operations (add, mul, etc.)
impl std::ops::Add for ReverseVariable {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        if self.is_constant() && other.is_constant() {
            ReverseVariable::constant(self.value + other.value)
        } else {
            // For non-constant variables, create a new variable with combined value
            // This won't track gradients properly - use graph-based operations for AD
            let result_value = self.value + other.value;
            let max_index = self.index.max(other.index);
            ReverseVariable {
                index: if max_index == usize::MAX {
                    usize::MAX
                } else {
                    max_index + 1
                },
                value: result_value,
                grad: 0.0,
            }
        }
    }
}

impl std::ops::Sub for ReverseVariable {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        if self.is_constant() && other.is_constant() {
            ReverseVariable::constant(self.value - other.value)
        } else {
            let result_value = self.value - other.value;
            let max_index = self.index.max(other.index);
            ReverseVariable {
                index: if max_index == usize::MAX {
                    usize::MAX
                } else {
                    max_index + 1
                },
                value: result_value,
                grad: 0.0,
            }
        }
    }
}

impl std::ops::Mul for ReverseVariable {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.is_constant() && other.is_constant() {
            ReverseVariable::constant(self.value * other.value)
        } else {
            let result_value = self.value * other.value;
            let max_index = self.index.max(other.index);
            ReverseVariable {
                index: if max_index == usize::MAX {
                    usize::MAX
                } else {
                    max_index + 1
                },
                value: result_value,
                grad: 0.0,
            }
        }
    }
}

impl std::ops::Div for ReverseVariable {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        if self.is_constant() && other.is_constant() {
            ReverseVariable::constant(self.value / other.value)
        } else {
            let result_value = self.value / other.value;
            let max_index = self.index.max(other.index);
            ReverseVariable {
                index: if max_index == usize::MAX {
                    usize::MAX
                } else {
                    max_index + 1
                },
                value: result_value,
                grad: 0.0,
            }
        }
    }
}

impl std::ops::Neg for ReverseVariable {
    type Output = Self;

    fn neg(self) -> Self {
        if self.is_constant() {
            ReverseVariable::constant(-self.value)
        } else {
            ReverseVariable {
                index: self.index,
                value: -self.value,
                grad: 0.0,
            }
        }
    }
}

// Scalar operations
impl std::ops::Add<f64> for ReverseVariable {
    type Output = Self;

    fn add(self, scalar: f64) -> Self {
        ReverseVariable {
            index: self.index,
            value: self.value + scalar,
            grad: self.grad,
        }
    }
}

impl std::ops::Sub<f64> for ReverseVariable {
    type Output = Self;

    fn sub(self, scalar: f64) -> Self {
        ReverseVariable {
            index: self.index,
            value: self.value - scalar,
            grad: self.grad,
        }
    }
}

impl std::ops::Mul<f64> for ReverseVariable {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        ReverseVariable {
            index: self.index,
            value: self.value * scalar,
            grad: self.grad,
        }
    }
}

impl std::ops::Div<f64> for ReverseVariable {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        ReverseVariable {
            index: self.index,
            value: self.value / scalar,
            grad: self.grad,
        }
    }
}

// Reverse scalar operations (f64 + ReverseVariable, etc.)
impl std::ops::Add<ReverseVariable> for f64 {
    type Output = ReverseVariable;

    fn add(self, var: ReverseVariable) -> ReverseVariable {
        var + self
    }
}

impl std::ops::Sub<ReverseVariable> for f64 {
    type Output = ReverseVariable;

    fn sub(self, var: ReverseVariable) -> ReverseVariable {
        ReverseVariable {
            index: var.index,
            value: self - var.value,
            grad: var.grad,
        }
    }
}

impl std::ops::Mul<ReverseVariable> for f64 {
    type Output = ReverseVariable;

    fn mul(self, var: ReverseVariable) -> ReverseVariable {
        var * self
    }
}

impl std::ops::Div<ReverseVariable> for f64 {
    type Output = ReverseVariable;

    fn div(self, var: ReverseVariable) -> ReverseVariable {
        ReverseVariable {
            index: var.index,
            value: self / var.value,
            grad: var.grad,
        }
    }
}

/// Addition operation on computation graph
#[allow(dead_code)]
pub fn add(
    graph: &mut ComputationGraph,
    left: &ReverseVariable,
    right: &ReverseVariable,
) -> ReverseVariable {
    if left.is_constant() && right.is_constant() {
        return ReverseVariable::constant(left.value + right.value);
    }

    let result_value = left.value + right.value;
    graph.add_binary_op(BinaryOpType::Add, left, right, result_value, 1.0, 1.0)
}

/// Multiplication operation on computation graph
#[allow(dead_code)]
pub fn mul(
    graph: &mut ComputationGraph,
    left: &ReverseVariable,
    right: &ReverseVariable,
) -> ReverseVariable {
    if left.is_constant() && right.is_constant() {
        return ReverseVariable::constant(left.value * right.value);
    }

    let result_value = left.value * right.value;
    graph.add_binary_op(
        BinaryOpType::Mul,
        left,
        right,
        result_value,
        right.value,
        left.value,
    )
}

/// Subtraction operation on computation graph
#[allow(dead_code)]
pub fn sub(
    graph: &mut ComputationGraph,
    left: &ReverseVariable,
    right: &ReverseVariable,
) -> ReverseVariable {
    if left.is_constant() && right.is_constant() {
        return ReverseVariable::constant(left.value - right.value);
    }

    let result_value = left.value - right.value;
    graph.add_binary_op(BinaryOpType::Sub, left, right, result_value, 1.0, -1.0)
}

/// Division operation on computation graph
#[allow(dead_code)]
pub fn div(
    graph: &mut ComputationGraph,
    left: &ReverseVariable,
    right: &ReverseVariable,
) -> ReverseVariable {
    if left.is_constant() && right.is_constant() {
        return ReverseVariable::constant(left.value / right.value);
    }

    let result_value = left.value / right.value;
    let left_grad = 1.0 / right.value;
    let right_grad = -left.value / (right.value * right.value);

    graph.add_binary_op(
        BinaryOpType::Div,
        left,
        right,
        result_value,
        left_grad,
        right_grad,
    )
}

/// Power operation (x^n) on computation graph
#[allow(dead_code)]
pub fn powi(graph: &mut ComputationGraph, input: &ReverseVariable, n: i32) -> ReverseVariable {
    if input.is_constant() {
        return ReverseVariable::constant(input.value.powi(n));
    }

    let result_value = input.value.powi(n);
    let input_grad = (n as f64) * input.value.powi(n - 1);

    graph.add_unary_op(UnaryOpType::Square, input, result_value, input_grad)
}

/// Exponential operation on computation graph
#[allow(dead_code)]
pub fn exp(graph: &mut ComputationGraph, input: &ReverseVariable) -> ReverseVariable {
    if input.is_constant() {
        return ReverseVariable::constant(input.value.exp());
    }

    let result_value = input.value.exp();
    let input_grad = result_value; // d/dx(e^x) = e^x

    graph.add_unary_op(UnaryOpType::Exp, input, result_value, input_grad)
}

/// Natural logarithm operation on computation graph
#[allow(dead_code)]
pub fn ln(graph: &mut ComputationGraph, input: &ReverseVariable) -> ReverseVariable {
    if input.is_constant() {
        return ReverseVariable::constant(input.value.ln());
    }

    let result_value = input.value.ln();
    let input_grad = 1.0 / input.value;

    graph.add_unary_op(UnaryOpType::Ln, input, result_value, input_grad)
}

/// Sine operation on computation graph
#[allow(dead_code)]
pub fn sin(graph: &mut ComputationGraph, input: &ReverseVariable) -> ReverseVariable {
    if input.is_constant() {
        return ReverseVariable::constant(input.value.sin());
    }

    let result_value = input.value.sin();
    let input_grad = input.value.cos();

    graph.add_unary_op(UnaryOpType::Sin, input, result_value, input_grad)
}

/// Cosine operation on computation graph
#[allow(dead_code)]
pub fn cos(graph: &mut ComputationGraph, input: &ReverseVariable) -> ReverseVariable {
    if input.is_constant() {
        return ReverseVariable::constant(input.value.cos());
    }

    let result_value = input.value.cos();
    let input_grad = -input.value.sin();

    graph.add_unary_op(UnaryOpType::Cos, input, result_value, input_grad)
}

/// Tangent operation on computation graph
#[allow(dead_code)]
pub fn tan(graph: &mut ComputationGraph, input: &ReverseVariable) -> ReverseVariable {
    if input.is_constant() {
        return ReverseVariable::constant(input.value.tan());
    }

    let result_value = input.value.tan();
    let cos_val = input.value.cos();
    let input_grad = 1.0 / (cos_val * cos_val); // sec²(x) = 1/cos²(x)

    graph.add_unary_op(UnaryOpType::Tan, input, result_value, input_grad)
}

/// Square root operation on computation graph
#[allow(dead_code)]
pub fn sqrt(graph: &mut ComputationGraph, input: &ReverseVariable) -> ReverseVariable {
    if input.is_constant() {
        return ReverseVariable::constant(input.value.sqrt());
    }

    let result_value = input.value.sqrt();
    let input_grad = 0.5 / result_value; // d/dx(√x) = 1/(2√x)

    graph.add_unary_op(UnaryOpType::Sqrt, input, result_value, input_grad)
}

/// Absolute value operation on computation graph
#[allow(dead_code)]
pub fn abs(graph: &mut ComputationGraph, input: &ReverseVariable) -> ReverseVariable {
    if input.is_constant() {
        return ReverseVariable::constant(input.value.abs());
    }

    let result_value = input.value.abs();
    let input_grad = if input.value >= 0.0 { 1.0 } else { -1.0 };

    graph.add_unary_op(UnaryOpType::Sqrt, input, result_value, input_grad)
}

/// Sigmoid operation on computation graph
#[allow(dead_code)]
pub fn sigmoid(graph: &mut ComputationGraph, input: &ReverseVariable) -> ReverseVariable {
    if input.is_constant() {
        let exp_val = (-input.value).exp();
        return ReverseVariable::constant(1.0 / (1.0 + exp_val));
    }

    let exp_neg_x = (-input.value).exp();
    let result_value = 1.0 / (1.0 + exp_neg_x);
    let input_grad = result_value * (1.0 - result_value); // σ'(x) = σ(x)(1-σ(x))

    graph.add_unary_op(UnaryOpType::Exp, input, result_value, input_grad)
}

/// Hyperbolic tangent operation on computation graph
#[allow(dead_code)]
pub fn tanh(graph: &mut ComputationGraph, input: &ReverseVariable) -> ReverseVariable {
    if input.is_constant() {
        return ReverseVariable::constant(input.value.tanh());
    }

    let result_value = input.value.tanh();
    let input_grad = 1.0 - result_value * result_value; // d/dx(tanh(x)) = 1 - tanh²(x)

    graph.add_unary_op(UnaryOpType::Tan, input, result_value, input_grad)
}

/// ReLU (Rectified Linear Unit) operation on computation graph
#[allow(dead_code)]
pub fn relu(graph: &mut ComputationGraph, input: &ReverseVariable) -> ReverseVariable {
    if input.is_constant() {
        return ReverseVariable::constant(input.value.max(0.0));
    }

    let result_value = input.value.max(0.0);
    let input_grad = if input.value > 0.0 { 1.0 } else { 0.0 };

    graph.add_unary_op(UnaryOpType::Sqrt, input, result_value, input_grad)
}

/// Leaky ReLU operation on computation graph
#[allow(dead_code)]
pub fn leaky_relu(
    graph: &mut ComputationGraph,
    input: &ReverseVariable,
    alpha: f64,
) -> ReverseVariable {
    if input.is_constant() {
        let result = if input.value > 0.0 {
            input.value
        } else {
            alpha * input.value
        };
        return ReverseVariable::constant(result);
    }

    let result_value = if input.value > 0.0 {
        input.value
    } else {
        alpha * input.value
    };
    let input_grad = if input.value > 0.0 { 1.0 } else { alpha };

    graph.add_unary_op(UnaryOpType::Sqrt, input, result_value, input_grad)
}

/// Compute gradient using reverse-mode automatic differentiation
/// This is a generic function that works with closures, using finite differences
/// For functions that can be expressed in terms of AD operations, use reverse_gradient_with_tape
#[allow(dead_code)]
pub fn reverse_gradient<F>(func: F, x: &ArrayView1<f64>) -> Result<Array1<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    // For generic functions, we need to use finite differences
    // This is because we don't have access to the function's AD representation
    let n = x.len();
    let mut gradient = Array1::zeros(n);
    let h = 1e-8;

    for i in 0..n {
        let mut x_plus = x.to_owned();
        x_plus[i] += h;
        let f_plus = func(&x_plus.view());

        let mut x_minus = x.to_owned();
        x_minus[i] -= h;
        let f_minus = func(&x_minus.view());

        gradient[i] = (f_plus - f_minus) / (2.0 * h);
    }

    Ok(gradient)
}

/// Compute gradient using reverse-mode AD with a function that directly uses AD operations
#[allow(dead_code)]
pub fn reverse_gradient_ad<F>(func: F, x: &ArrayView1<f64>) -> Result<Array1<f64>, OptimizeError>
where
    F: Fn(&mut ComputationGraph, &[ReverseVariable]) -> ReverseVariable,
{
    let mut graph = ComputationGraph::new();

    // Create input variables
    let input_vars: Vec<ReverseVariable> = x.iter().map(|&xi| graph.variable(xi)).collect();

    // Evaluate function with the computation graph
    let output = func(&mut graph, &input_vars);

    // Perform backpropagation
    graph.backward(&output)?;

    // Extract gradients
    let mut gradient = Array1::zeros(x.len());
    for (i, var) in input_vars.iter().enumerate() {
        gradient[i] = graph.get_gradient(var);
    }

    Ok(gradient)
}

/// Compute Hessian using reverse-mode automatic differentiation (finite differences for generic functions)
#[allow(dead_code)]
pub fn reverse_hessian<F>(func: F, x: &ArrayView1<f64>) -> Result<Array2<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let mut hessian = Array2::zeros((n, n));
    let h = 1e-5;

    // Compute Hessian using finite differences
    // For generic functions, this is the most practical approach
    for i in 0..n {
        for j in 0..n {
            if i == j {
                // Diagonal element: f''(x) = (f(x+h) - 2f(x) + f(x-h)) / h²
                let mut x_plus = x.to_owned();
                x_plus[i] += h;
                let f_plus = func(&x_plus.view());

                let f_center = func(x);

                let mut x_minus = x.to_owned();
                x_minus[i] -= h;
                let f_minus = func(&x_minus.view());

                hessian[[i, j]] = (f_plus - 2.0 * f_center + f_minus) / (h * h);
            } else {
                // Off-diagonal element: mixed partial derivative
                // Variable names represent plus/minus combinations for finite differences
                {
                    #[allow(clippy::similar_names)]
                    let mut x_pp = x.to_owned();
                    x_pp[i] += h;
                    x_pp[j] += h;
                    #[allow(clippy::similar_names)]
                    let f_pp = func(&x_pp.view());

                    #[allow(clippy::similar_names)]
                    let mut x_pm = x.to_owned();
                    x_pm[i] += h;
                    x_pm[j] -= h;
                    #[allow(clippy::similar_names)]
                    let f_pm = func(&x_pm.view());

                    #[allow(clippy::similar_names)]
                    let mut x_mp = x.to_owned();
                    x_mp[i] -= h;
                    x_mp[j] += h;
                    #[allow(clippy::similar_names)]
                    let f_mp = func(&x_mp.view());

                    let mut x_mm = x.to_owned();
                    x_mm[i] -= h;
                    x_mm[j] -= h;
                    let f_mm = func(&x_mm.view());

                    hessian[[i, j]] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
                }
            }
        }
    }

    Ok(hessian)
}

/// Compute Hessian using forward-over-reverse mode for AD functions
#[allow(dead_code)]
pub fn reverse_hessian_ad<F>(func: F, x: &ArrayView1<f64>) -> Result<Array2<f64>, OptimizeError>
where
    F: Fn(&mut ComputationGraph, &[ReverseVariable]) -> ReverseVariable,
{
    let n = x.len();
    let mut hessian = Array2::zeros((n, n));

    // Compute Hessian by differentiating the gradient
    // For each input variable, compute gradient and then differentiate again
    for i in 0..n {
        // Create a function that returns the i-th component of the gradient
        let gradient_i_func = |x_val: &ArrayView1<f64>| -> f64 {
            let grad = reverse_gradient_ad(&func, x_val).unwrap();
            grad[i]
        };

        // Compute the gradient of the i-th gradient component (i-th row of Hessian)
        let hessian_row = reverse_gradient(gradient_i_func, x)?;
        for j in 0..n {
            hessian[[i, j]] = hessian_row[j];
        }
    }

    Ok(hessian)
}

/// Simple reverse-mode gradient computation using a basic tape
#[allow(dead_code)]
pub fn reverse_gradient_with_tape<F>(
    func: F,
    x: &ArrayView1<f64>,
    _options: &ReverseADOptions,
) -> Result<Array1<f64>, OptimizeError>
where
    F: Fn(&mut ComputationGraph, &[ReverseVariable]) -> ReverseVariable,
{
    let mut graph = ComputationGraph::new();

    // Create input variables
    let input_vars: Vec<ReverseVariable> = x.iter().map(|&xi| graph.variable(xi)).collect();

    // Evaluate function with the computation graph
    let output = func(&mut graph, &input_vars);

    // Perform backpropagation
    graph.backward(&output)?;

    // Extract gradients
    let mut gradient = Array1::zeros(x.len());
    for (i, var) in input_vars.iter().enumerate() {
        gradient[i] = graph.get_gradient(var);
    }

    Ok(gradient)
}

/// Check if reverse mode is preferred for the given problem dimensions
#[allow(dead_code)]
pub fn is_reverse_mode_efficient(_input_dim: usize, output_dim: usize) -> bool {
    // Reverse mode is efficient when output dimension is small
    // Cost is O(output_dim * cost_of_function)
    output_dim <= 10 || (output_dim <= _input_dim && output_dim <= 20)
}

/// Vector-Jacobian product using reverse-mode AD
#[allow(clippy::many_single_char_names)]
#[allow(dead_code)]
pub fn reverse_vjp<F>(
    func: F,
    x: &ArrayView1<f64>,
    v: &ArrayView1<f64>,
) -> Result<Array1<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64>,
{
    // For vector-valued functions, we use the natural efficiency of reverse-mode AD
    // which computes v^T * J efficiently by seeding the output with v
    let n = x.len();
    let m = v.len();

    // Compute v^T * J by running reverse mode for each output component weighted by v
    let mut result = Array1::zeros(n);

    // For each output component, compute its contribution to the VJP
    for i in 0..m {
        if v[i] != 0.0 {
            // Create a scalar function that extracts the i-th component
            let component_func = |x_val: &ArrayView1<f64>| -> f64 {
                let f_val = func(x_val);
                f_val[i]
            };

            // Compute gradient of this component
            let grad_i = reverse_gradient(component_func, x)?;

            // Add weighted contribution to result
            for j in 0..n {
                result[j] += v[i] * grad_i[j];
            }
        }
    }

    Ok(result)
}

/// Vector-Jacobian product using reverse-mode AD for AD-compatible functions
#[allow(clippy::many_single_char_names)]
#[allow(dead_code)]
pub fn reverse_vjp_ad<F>(
    func: F,
    x: &ArrayView1<f64>,
    v: &ArrayView1<f64>,
) -> Result<Array1<f64>, OptimizeError>
where
    F: Fn(&mut ComputationGraph, &[ReverseVariable]) -> Vec<ReverseVariable>,
{
    let n = x.len();
    let m = v.len();
    let mut result = Array1::zeros(n);

    // For each output component with non-zero weight
    for i in 0..m {
        if v[i] != 0.0 {
            let mut graph = ComputationGraph::new();

            // Create input variables
            let input_vars: Vec<ReverseVariable> = x.iter().map(|&xi| graph.variable(xi)).collect();

            // Evaluate function
            let outputs = func(&mut graph, &input_vars);

            // Seed the i-th output with 1.0 and perform backpropagation
            if i < outputs.len() {
                graph.backward(&outputs[i])?;

                // Add weighted contribution to result
                for (j, var) in input_vars.iter().enumerate() {
                    result[j] += v[i] * graph.get_gradient(var);
                }
            }
        }
    }

    Ok(result)
}

/// Gauss-Newton Hessian approximation using reverse-mode AD
#[allow(dead_code)]
pub fn reverse_gauss_newton_hessian<F>(
    func: F,
    x: &ArrayView1<f64>,
) -> Result<Array2<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64>,
{
    // Compute Gauss-Newton approximation: H ≈ J^T * J efficiently using reverse-mode AD
    let n = x.len();
    let f_val = func(x);
    let m = f_val.len();

    // Use reverse-mode AD to compute J^T * J directly without forming J explicitly
    let mut hessian = Array2::zeros((n, n));

    // For each output component, compute its contribution to the Gauss-Newton Hessian
    for i in 0..m {
        // Create a scalar function for the i-th residual component
        let residual_i = |x_val: &ArrayView1<f64>| -> f64 {
            let f_val = func(x_val);
            f_val[i]
        };

        // Compute gradient of this residual
        let grad_i = reverse_gradient(residual_i, x)?;

        // Add outer product grad_i * grad_i^T to the Hessian
        for j in 0..n {
            for k in 0..n {
                hessian[[j, k]] += grad_i[j] * grad_i[k];
            }
        }
    }

    Ok(hessian)
}

/// Gauss-Newton Hessian approximation using reverse-mode AD for AD-compatible functions
#[allow(dead_code)]
pub fn reverse_gauss_newton_hessian_ad<F>(
    func: F,
    x: &ArrayView1<f64>,
) -> Result<Array2<f64>, OptimizeError>
where
    F: Fn(&mut ComputationGraph, &[ReverseVariable]) -> Vec<ReverseVariable>,
{
    let n = x.len();
    let mut hessian = Array2::zeros((n, n));

    // Get function values to determine output dimension
    let mut graph_temp = ComputationGraph::new();
    let input_vars_temp: Vec<ReverseVariable> =
        x.iter().map(|&xi| graph_temp.variable(xi)).collect();
    let outputs_temp = func(&mut graph_temp, &input_vars_temp);
    let m = outputs_temp.len();

    // For each output component, compute its contribution to the Gauss-Newton Hessian
    for i in 0..m {
        let mut graph = ComputationGraph::new();

        // Create input variables
        let input_vars: Vec<ReverseVariable> = x.iter().map(|&xi| graph.variable(xi)).collect();

        // Evaluate function
        let outputs = func(&mut graph, &input_vars);

        // Compute gradient of the i-th output component
        if i < outputs.len() {
            graph.backward(&outputs[i])?;

            // Extract gradients
            let mut grad_i = Array1::zeros(n);
            for (j, var) in input_vars.iter().enumerate() {
                grad_i[j] = graph.get_gradient(var);
            }

            // Add outer product grad_i * grad_i^T to the Hessian
            for j in 0..n {
                for k in 0..n {
                    hessian[[j, k]] += grad_i[j] * grad_i[k];
                }
            }
        }
    }

    Ok(hessian)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_computation_graph() {
        let mut graph = ComputationGraph::new();

        // Create variables: x = 2, y = 3
        let x = graph.variable(2.0);
        let y = graph.variable(3.0);

        // Compute z = x * y + x
        let xy = mul(&mut graph, &x, &y);
        let z = add(&mut graph, &xy, &x);

        assert_abs_diff_eq!(z.value, 8.0, epsilon = 1e-10); // 2*3 + 2 = 8

        // Perform backpropagation
        graph.backward(&z).unwrap();

        // Check gradients: ∂z/∂x = y + 1 = 4, ∂z/∂y = x = 2
        assert_abs_diff_eq!(graph.get_gradient(&x), 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(graph.get_gradient(&y), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_unary_operations() {
        let mut graph = ComputationGraph::new();

        let x = graph.variable(1.0);
        let exp_x = exp(&mut graph, &x);

        assert_abs_diff_eq!(exp_x.value, std::f64::consts::E, epsilon = 1e-10);

        graph.backward(&exp_x).unwrap();

        // ∂(e^x)/∂x = e^x
        assert_abs_diff_eq!(graph.get_gradient(&x), std::f64::consts::E, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient() {
        // Test function: f(x, y) = x² + xy + 2y²
        let func = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[0] * x[1] + 2.0 * x[1] * x[1] };

        let x = Array1::from_vec(vec![1.0, 2.0]);
        let grad = reverse_gradient(func, &x.view()).unwrap();

        // ∂f/∂x = 2x + y = 2(1) + 2 = 4
        // ∂f/∂y = x + 4y = 1 + 4(2) = 9
        assert_abs_diff_eq!(grad[0], 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(grad[1], 9.0, epsilon = 1e-6);
    }

    #[test]
    fn test_is_reverse_mode_efficient() {
        // Small output dimension should prefer reverse mode
        assert!(is_reverse_mode_efficient(100, 1));
        assert!(is_reverse_mode_efficient(50, 5));

        // Large output dimension should not prefer reverse mode
        assert!(!is_reverse_mode_efficient(10, 100));
    }

    #[test]
    fn test_reverse_hessian() {
        // Test function: f(x, y) = x² + xy + 2y²
        let func = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[0] * x[1] + 2.0 * x[1] * x[1] };

        let x = Array1::from_vec(vec![1.0, 2.0]);
        let hess = reverse_hessian(func, &x.view()).unwrap();

        // Expected Hessian:
        // ∂²f/∂x² = 2, ∂²f/∂x∂y = 1
        // ∂²f/∂y∂x = 1, ∂²f/∂y² = 4
        assert_abs_diff_eq!(hess[[0, 0]], 2.0, epsilon = 1e-4);
        assert_abs_diff_eq!(hess[[0, 1]], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(hess[[1, 0]], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(hess[[1, 1]], 4.0, epsilon = 1e-4);
    }

    #[test]
    fn test_reverse_gradient_ad() {
        // Test function: f(x, y) = x² + xy + 2y²
        let func = |graph: &mut ComputationGraph, vars: &[ReverseVariable]| {
            let x = &vars[0];
            let y = &vars[1];

            let x_squared = mul(graph, x, x);
            let xy = mul(graph, x, y);
            let y_squared = mul(graph, y, y);
            let two_y_squared = mul(graph, &ReverseVariable::constant(2.0), &y_squared);

            let temp = add(graph, &x_squared, &xy);
            add(graph, &temp, &two_y_squared)
        };

        let x = Array1::from_vec(vec![1.0, 2.0]);
        let grad = reverse_gradient_ad(func, &x.view()).unwrap();

        // ∂f/∂x = 2x + y = 2(1) + 2 = 4
        // ∂f/∂y = x + 4y = 1 + 4(2) = 9
        assert_abs_diff_eq!(grad[0], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grad[1], 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_vjp() {
        // Test vector function: f(x) = [x₀², x₀x₁, x₁²]
        let func = |x: &ArrayView1<f64>| -> Array1<f64> {
            Array1::from_vec(vec![x[0] * x[0], x[0] * x[1], x[1] * x[1]])
        };

        let x = Array1::from_vec(vec![2.0, 3.0]);
        let v = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let vjp = reverse_vjp(func, &x.view(), &v.view()).unwrap();

        // Jacobian at (2,3):
        // ∂f₀/∂x₀ = 2x₀ = 4, ∂f₀/∂x₁ = 0
        // ∂f₁/∂x₀ = x₁ = 3,  ∂f₁/∂x₁ = x₀ = 2
        // ∂f₂/∂x₀ = 0,      ∂f₂/∂x₁ = 2x₁ = 6

        // v^T * J = [1,1,1] * [[4,0], [3,2], [0,6]] = [7, 8]
        assert_abs_diff_eq!(vjp[0], 7.0, epsilon = 1e-6);
        assert_abs_diff_eq!(vjp[1], 8.0, epsilon = 1e-6);
    }

    #[test]
    fn test_reverse_gauss_newton_hessian() {
        // Test residual function: r(x) = [x₀ - 1, x₁ - 2]
        let residual_func =
            |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![x[0] - 1.0, x[1] - 2.0]) };

        let x = Array1::from_vec(vec![0.0, 0.0]);
        let gn_hess = reverse_gauss_newton_hessian(residual_func, &x.view()).unwrap();

        // Jacobian is identity matrix, so J^T * J should be identity
        assert_abs_diff_eq!(gn_hess[[0, 0]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(gn_hess[[0, 1]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(gn_hess[[1, 0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(gn_hess[[1, 1]], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_power_operation() {
        let mut graph = ComputationGraph::new();

        let x = graph.variable(2.0);
        let x_cubed = powi(&mut graph, &x, 3);

        assert_abs_diff_eq!(x_cubed.value, 8.0, epsilon = 1e-10); // 2³ = 8

        graph.backward(&x_cubed).unwrap();

        // ∂(x³)/∂x = 3x² = 3(4) = 12 at x=2
        assert_abs_diff_eq!(graph.get_gradient(&x), 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trigonometric_operations() {
        let mut graph = ComputationGraph::new();

        let x = graph.variable(0.0);
        let sin_x = sin(&mut graph, &x);
        let cos_x = cos(&mut graph, &x);

        assert_abs_diff_eq!(sin_x.value, 0.0, epsilon = 1e-10); // sin(0) = 0
        assert_abs_diff_eq!(cos_x.value, 1.0, epsilon = 1e-10); // cos(0) = 1

        graph.backward(&sin_x).unwrap();
        assert_abs_diff_eq!(graph.get_gradient(&x), 1.0, epsilon = 1e-10); // d/dx(sin(x)) = cos(x) = 1 at x=0

        graph.zero_gradients();
        graph.backward(&cos_x).unwrap();
        assert_abs_diff_eq!(graph.get_gradient(&x), 0.0, epsilon = 1e-10); // d/dx(cos(x)) = -sin(x) = 0 at x=0
    }

    #[test]
    fn test_arithmetic_operations_without_graph() {
        // Test arithmetic operations that work without explicit graph context
        let a = ReverseVariable::constant(3.0);
        let b = ReverseVariable::constant(2.0);

        // Test addition
        let sum = a.clone() + b.clone();
        assert_abs_diff_eq!(sum.value, 5.0, epsilon = 1e-10);
        assert!(sum.is_constant());

        // Test subtraction
        let diff = a.clone() - b.clone();
        assert_abs_diff_eq!(diff.value, 1.0, epsilon = 1e-10);

        // Test multiplication
        let product = a.clone() * b.clone();
        assert_abs_diff_eq!(product.value, 6.0, epsilon = 1e-10);

        // Test division
        let quotient = a.clone() / b.clone();
        assert_abs_diff_eq!(quotient.value, 1.5, epsilon = 1e-10);

        // Test negation
        let neg_a = -a.clone();
        assert_abs_diff_eq!(neg_a.value, -3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scalar_operations() {
        let var = ReverseVariable::constant(4.0);

        // Test scalar addition
        let result = var.clone() + 2.0;
        assert_abs_diff_eq!(result.value, 6.0, epsilon = 1e-10);

        // Test reverse scalar addition
        let result = 2.0 + var.clone();
        assert_abs_diff_eq!(result.value, 6.0, epsilon = 1e-10);

        // Test scalar multiplication
        let result = var.clone() * 3.0;
        assert_abs_diff_eq!(result.value, 12.0, epsilon = 1e-10);

        // Test scalar division
        let result = var.clone() / 2.0;
        assert_abs_diff_eq!(result.value, 2.0, epsilon = 1e-10);

        // Test reverse scalar division
        let result = 8.0 / var.clone();
        assert_abs_diff_eq!(result.value, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mathematical_functions_without_graph() {
        let var = ReverseVariable::constant(4.0);

        // Test power
        let result = var.powi(2);
        assert_abs_diff_eq!(result.value, 16.0, epsilon = 1e-10);

        // Test square root
        let result = var.sqrt();
        assert_abs_diff_eq!(result.value, 2.0, epsilon = 1e-10);

        // Test exponential
        let var_zero = ReverseVariable::constant(0.0);
        let result = var_zero.exp();
        assert_abs_diff_eq!(result.value, 1.0, epsilon = 1e-10);

        // Test natural logarithm
        let var_e = ReverseVariable::constant(std::f64::consts::E);
        let result = var_e.ln();
        assert_abs_diff_eq!(result.value, 1.0, epsilon = 1e-10);

        // Test trigonometric functions
        let var_zero = ReverseVariable::constant(0.0);
        assert_abs_diff_eq!(var_zero.sin().value, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(var_zero.cos().value, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(var_zero.tan().value, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_advanced_operations_with_graph() {
        let mut graph = ComputationGraph::new();

        // Test sigmoid function
        let x = graph.variable(0.0);
        let sig = sigmoid(&mut graph, &x);
        assert_abs_diff_eq!(sig.value, 0.5, epsilon = 1e-10); // sigmoid(0) = 0.5

        graph.backward(&sig).unwrap();
        assert_abs_diff_eq!(graph.get_gradient(&x), 0.25, epsilon = 1e-10); // sigmoid'(0) = 0.25

        // Test ReLU function
        graph.zero_gradients();
        let x_pos = graph.variable(2.0);
        let relu_pos = relu(&mut graph, &x_pos);
        assert_abs_diff_eq!(relu_pos.value, 2.0, epsilon = 1e-10);

        graph.backward(&relu_pos).unwrap();
        assert_abs_diff_eq!(graph.get_gradient(&x_pos), 1.0, epsilon = 1e-10); // ReLU'(2) = 1

        // Test ReLU for negative input
        let mut graph2 = ComputationGraph::new();
        let x_neg = graph2.variable(-1.0);
        let relu_neg = relu(&mut graph2, &x_neg);
        assert_abs_diff_eq!(relu_neg.value, 0.0, epsilon = 1e-10);

        graph2.backward(&relu_neg).unwrap();
        assert_abs_diff_eq!(graph2.get_gradient(&x_neg), 0.0, epsilon = 1e-10); // ReLU'(-1) = 0
    }

    #[test]
    fn test_leaky_relu() {
        let mut graph = ComputationGraph::new();

        // Test Leaky ReLU with positive input
        let x_pos = graph.variable(2.0);
        let leaky_pos = leaky_relu(&mut graph, &x_pos, 0.01);
        assert_abs_diff_eq!(leaky_pos.value, 2.0, epsilon = 1e-10);

        graph.backward(&leaky_pos).unwrap();
        assert_abs_diff_eq!(graph.get_gradient(&x_pos), 1.0, epsilon = 1e-10);

        // Test Leaky ReLU with negative input
        let mut graph2 = ComputationGraph::new();
        let x_neg = graph2.variable(-2.0);
        let leaky_neg = leaky_relu(&mut graph2, &x_neg, 0.01);
        assert_abs_diff_eq!(leaky_neg.value, -0.02, epsilon = 1e-10);

        graph2.backward(&leaky_neg).unwrap();
        assert_abs_diff_eq!(graph2.get_gradient(&x_neg), 0.01, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_expression() {
        let mut graph = ComputationGraph::new();

        // Test complex expression: f(x, y) = sigmoid(x² + y) * tanh(x - y)
        let x = graph.variable(1.0);
        let y = graph.variable(0.5);

        let x_squared = mul(&mut graph, &x, &x);
        let x_sq_plus_y = add(&mut graph, &x_squared, &y);
        let sig_term = sigmoid(&mut graph, &x_sq_plus_y);

        let x_minus_y = sub(&mut graph, &x, &y);
        let tanh_term = tanh(&mut graph, &x_minus_y);

        let result = mul(&mut graph, &sig_term, &tanh_term);

        // Verify the computation produces a reasonable result
        assert!(result.value.is_finite());
        assert!(result.value > 0.0); // Both sigmoid and tanh(0.5) are positive

        // Test backpropagation
        graph.backward(&result).unwrap();

        // Gradients should be finite and non-zero
        let grad_x = graph.get_gradient(&x);
        let grad_y = graph.get_gradient(&y);

        assert!(grad_x.is_finite());
        assert!(grad_y.is_finite());
        assert!(grad_x != 0.0);
        assert!(grad_y != 0.0);
    }
}
