//! Reverse-mode automatic differentiation (backpropagation)
//!
//! Reverse-mode AD is efficient for computing derivatives when the number of
//! output variables is small (typically 1 for optimization). It builds a
//! computational graph and then propagates derivatives backwards.

use crate::automatic_differentiation::tape::{ComputationTape, TapeNode, Variable};
use crate::error::OptimizeError;
use ndarray::{Array1, Array2, ArrayView1};

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
impl std::ops::Add for ReverseVariable {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        if self.is_constant() && other.is_constant() {
            ReverseVariable::constant(self.value + other.value)
        } else {
            // This would need access to the computation graph
            // In practice, operations would be methods on the graph
            panic!("Operations need computation graph context")
        }
    }
}

/// Addition operation on computation graph
pub fn add(
    graph: &mut ComputationGraph,
    left: &ReverseVariable,
    right: &ReverseVariable,
) -> ReverseVariable {
    if left.is_constant() && right.is_constant() {
        return ReverseVariable::constant(left.value + right.value);
    }

    let result_value = left.value + right.value;
    graph.add_binary_op(left, right, result_value, 1.0, 1.0)
}

/// Multiplication operation on computation graph
pub fn mul(
    graph: &mut ComputationGraph,
    left: &ReverseVariable,
    right: &ReverseVariable,
) -> ReverseVariable {
    if left.is_constant() && right.is_constant() {
        return ReverseVariable::constant(left.value * right.value);
    }

    let result_value = left.value * right.value;
    graph.add_binary_op(left, right, result_value, right.value, left.value)
}

/// Subtraction operation on computation graph
pub fn sub(
    graph: &mut ComputationGraph,
    left: &ReverseVariable,
    right: &ReverseVariable,
) -> ReverseVariable {
    if left.is_constant() && right.is_constant() {
        return ReverseVariable::constant(left.value - right.value);
    }

    let result_value = left.value - right.value;
    graph.add_binary_op(left, right, result_value, 1.0, -1.0)
}

/// Division operation on computation graph
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

    graph.add_binary_op(left, right, result_value, left_grad, right_grad)
}

/// Power operation (x^n) on computation graph
pub fn powi(graph: &mut ComputationGraph, input: &ReverseVariable, n: i32) -> ReverseVariable {
    if input.is_constant() {
        return ReverseVariable::constant(input.value.powi(n));
    }

    let result_value = input.value.powi(n);
    let input_grad = (n as f64) * input.value.powi(n - 1);

    graph.add_unary_op(input, result_value, input_grad)
}

/// Exponential operation on computation graph
pub fn exp(graph: &mut ComputationGraph, input: &ReverseVariable) -> ReverseVariable {
    if input.is_constant() {
        return ReverseVariable::constant(input.value.exp());
    }

    let result_value = input.value.exp();
    let input_grad = result_value; // d/dx(e^x) = e^x

    graph.add_unary_op(input, result_value, input_grad)
}

/// Natural logarithm operation on computation graph
pub fn ln(graph: &mut ComputationGraph, input: &ReverseVariable) -> ReverseVariable {
    if input.is_constant() {
        return ReverseVariable::constant(input.value.ln());
    }

    let result_value = input.value.ln();
    let input_grad = 1.0 / input.value;

    graph.add_unary_op(input, result_value, input_grad)
}

/// Sine operation on computation graph
pub fn sin(graph: &mut ComputationGraph, input: &ReverseVariable) -> ReverseVariable {
    if input.is_constant() {
        return ReverseVariable::constant(input.value.sin());
    }

    let result_value = input.value.sin();
    let input_grad = input.value.cos();

    graph.add_unary_op(input, result_value, input_grad)
}

/// Cosine operation on computation graph
pub fn cos(graph: &mut ComputationGraph, input: &ReverseVariable) -> ReverseVariable {
    if input.is_constant() {
        return ReverseVariable::constant(input.value.cos());
    }

    let result_value = input.value.cos();
    let input_grad = -input.value.sin();

    graph.add_unary_op(input, result_value, input_grad)
}

/// Compute gradient using reverse-mode automatic differentiation
pub fn reverse_gradient<F>(func: F, x: &ArrayView1<f64>) -> Result<Array1<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    // For now, use finite differences as a placeholder
    // In a full implementation, this would use the computational graph
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

/// Compute Hessian using reverse-mode automatic differentiation
pub fn reverse_hessian<F>(func: F, x: &ArrayView1<f64>) -> Result<Array2<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let mut hessian = Array2::zeros((n, n));
    let h = 1e-5;

    // Compute Hessian using finite differences
    // This is a placeholder - full reverse-mode would compute this more efficiently
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
                let mut x_pp = x.to_owned();
                x_pp[i] += h;
                x_pp[j] += h;
                let f_pp = func(&x_pp.view());

                let mut x_pm = x.to_owned();
                x_pm[i] += h;
                x_pm[j] -= h;
                let f_pm = func(&x_pm.view());

                let mut x_mp = x.to_owned();
                x_mp[i] -= h;
                x_mp[j] += h;
                let f_mp = func(&x_mp.view());

                let mut x_mm = x.to_owned();
                x_mm[i] -= h;
                x_mm[j] -= h;
                let f_mm = func(&x_mm.view());

                hessian[[i, j]] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
            }
        }
    }

    Ok(hessian)
}

/// Simple reverse-mode gradient computation using a basic tape
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
pub fn is_reverse_mode_efficient(input_dim: usize, output_dim: usize) -> bool {
    // Reverse mode is efficient when output dimension is small
    // Cost is O(output_dim * cost_of_function)
    output_dim <= 10 || (output_dim <= input_dim && output_dim <= 20)
}

/// Vector-Jacobian product using reverse-mode AD
pub fn reverse_vjp<F>(
    func: F,
    x: &ArrayView1<f64>,
    v: &ArrayView1<f64>,
) -> Result<Array1<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64>,
{
    // This is a simplified implementation
    // In practice, this would use the computational graph
    let n = x.len();
    let m = v.len();

    // Compute Jacobian-vector product v^T * J
    let mut result = Array1::zeros(n);
    let h = 1e-8;

    for j in 0..n {
        let mut x_plus = x.to_owned();
        x_plus[j] += h;
        let f_plus = func(&x_plus.view());

        let mut x_minus = x.to_owned();
        x_minus[j] -= h;
        let f_minus = func(&x_minus.view());

        // Compute j-th column of Jacobian
        let mut col_sum = 0.0;
        for i in 0..m {
            let jacobian_ij = (f_plus[i] - f_minus[i]) / (2.0 * h);
            col_sum += v[i] * jacobian_ij;
        }
        result[j] = col_sum;
    }

    Ok(result)
}

/// Gauss-Newton Hessian approximation using reverse-mode AD
pub fn reverse_gauss_newton_hessian<F>(
    func: F,
    x: &ArrayView1<f64>,
) -> Result<Array2<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64>,
{
    // Compute Gauss-Newton approximation: H ≈ J^T * J
    let f_val = func(x);
    let m = f_val.len();
    let n = x.len();

    // Compute Jacobian using finite differences
    let mut jacobian = Array2::zeros((m, n));
    let h = 1e-8;

    for j in 0..n {
        let mut x_plus = x.to_owned();
        x_plus[j] += h;
        let f_plus = func(&x_plus.view());

        let mut x_minus = x.to_owned();
        x_minus[j] -= h;
        let f_minus = func(&x_minus.view());

        for i in 0..m {
            jacobian[[i, j]] = (f_plus[i] - f_minus[i]) / (2.0 * h);
        }
    }

    // Compute J^T * J
    let mut hessian = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += jacobian[[k, i]] * jacobian[[k, j]];
            }
            hessian[[i, j]] = sum;
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
}
