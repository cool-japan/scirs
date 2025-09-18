//! Forward-mode automatic differentiation
//!
//! This module implements forward-mode automatic differentiation for computing
//! directional derivatives and Jacobian-vector products efficiently.

use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::HashMap;

use crate::error::{OptimError, Result};

/// Dual number for forward-mode automatic differentiation
#[derive(Debug, Clone)]
pub struct DualNumber<T: Float> {
    /// Primal value
    pub value: T,

    /// Tangent (derivative) value
    pub tangent: T,
}

/// Multi-dimensional dual number for vector-valued functions
#[derive(Debug, Clone)]
pub struct VectorDual<T: Float> {
    /// Primal value (vector)
    pub value: Array1<T>,

    /// Tangent matrix (Jacobian-vector product)
    pub tangent: Array1<T>,
}

/// Forward-mode AD engine
pub struct ForwardModeEngine<T: Float + Default + Clone + std::iter::Sum + 'static> {
    /// Computation graph
    tape: Vec<ForwardOperation<T>>,

    /// Variable registry
    variables: HashMap<String, usize>,

    /// Current seed vectors for directional derivatives
    seed_vectors: Vec<Array1<T>>,

    /// Enable higher-order derivatives
    higher_order: bool,
}

/// Forward-mode operation
#[derive(Debug, Clone)]
struct ForwardOperation<T: Float> {
    /// Operation type
    optype: ForwardOpType,

    /// Input variable indices
    inputs: Vec<usize>,

    /// Output variable index
    output: usize,

    /// Operation metadata
    metadata: ForwardOpMetadata<T>,
}

/// Forward operation types
#[derive(Debug, Clone)]
enum ForwardOpType {
    Variable,
    Constant,
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    ReLU,
    MatMul,
    Dot,
    Sum,
    Mean,
    Norm,
}

/// Operation metadata for forward mode
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ForwardOpMetadata<T: Float> {
    /// Partial derivatives with respect to inputs
    partials: Vec<T>,

    /// Shape information
    shape: Vec<usize>,

    /// Operation-specific data
    data: ForwardOpData<T>,
}

/// Operation-specific data
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum ForwardOpData<T: Float> {
    None,
    ConstantValue(T),
    PowerExponent(T),
    MatMulDims { m: usize, n: usize, k: usize },
    ReductionAxis(usize),
}

impl<T: Float + Default + Clone> DualNumber<T> {
    /// Create a new dual number
    pub fn new(value: T, tangent: T) -> Self {
        Self { value, tangent }
    }

    /// Create a constant (zero tangent)
    pub fn constant(value: T) -> Self {
        Self::new(value, T::zero())
    }

    /// Create a variable (unit tangent)
    pub fn variable(value: T) -> Self {
        Self::new(value, T::one())
    }
}

impl<T: Float + Default + Clone> VectorDual<T> {
    /// Create a new vector dual number
    pub fn new(value: Array1<T>, tangent: Array1<T>) -> Self {
        Self { value, tangent }
    }

    /// Create a constant vector
    pub fn constant(value: Array1<T>) -> Self {
        let tangent = Array1::zeros(value.len());
        Self::new(value, tangent)
    }

    /// Create a variable vector with unit tangent in direction i
    pub fn variable(value: Array1<T>, direction: usize) -> Self {
        let mut tangent = Array1::zeros(value.len());
        if direction < tangent.len() {
            tangent[direction] = T::one();
        }
        Self::new(value, tangent)
    }
}

impl<T: Float + Default + Clone + std::iter::Sum + 'static> ForwardModeEngine<T> {
    /// Create a new forward-mode AD engine
    pub fn new() -> Self {
        Self {
            tape: Vec::new(),
            variables: HashMap::new(),
            seed_vectors: Vec::new(),
            higher_order: false,
        }
    }

    /// Enable higher-order derivatives
    pub fn enable_higher_order(&mut self, enabled: bool) {
        self.higher_order = enabled;
    }

    /// Set seed vectors for computing directional derivatives
    pub fn set_seed_vectors(&mut self, seeds: Vec<Array1<T>>) {
        self.seed_vectors = seeds;
    }

    /// Create a variable
    pub fn create_variable(&mut self, name: &str, value: Array1<T>) -> usize {
        let var_id = self.tape.len();

        let op = ForwardOperation {
            optype: ForwardOpType::Variable,
            inputs: Vec::new(),
            output: var_id,
            metadata: ForwardOpMetadata {
                partials: vec![T::one()],
                shape: value.shape().to_vec(),
                data: ForwardOpData::None,
            },
        };

        self.tape.push(op);
        self.variables.insert(name.to_string(), var_id);
        var_id
    }

    /// Create a constant
    pub fn create_constant(&mut self, value: Array1<T>) -> usize {
        let const_id = self.tape.len();

        let op = ForwardOperation {
            optype: ForwardOpType::Constant,
            inputs: Vec::new(),
            output: const_id,
            metadata: ForwardOpMetadata {
                partials: vec![T::zero()],
                shape: value.shape().to_vec(),
                data: ForwardOpData::ConstantValue(value[0]),
            },
        };

        self.tape.push(op);
        const_id
    }

    /// Add two variables
    pub fn add(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(ForwardOpType::Add, lhs, rhs)
    }

    /// Subtract two variables
    pub fn subtract(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(ForwardOpType::Subtract, lhs, rhs)
    }

    /// Multiply two variables
    pub fn multiply(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(ForwardOpType::Multiply, lhs, rhs)
    }

    /// Divide two variables
    pub fn divide(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(ForwardOpType::Divide, lhs, rhs)
    }

    /// Raise to power
    pub fn power(&mut self, base: usize, exponent: T) -> Result<usize> {
        let output_id = self.tape.len();

        let op = ForwardOperation {
            optype: ForwardOpType::Power,
            inputs: vec![base],
            output: output_id,
            metadata: ForwardOpMetadata {
                partials: vec![T::zero()], // Will be computed during forward pass
                shape: vec![],             // Will be inferred
                data: ForwardOpData::PowerExponent(exponent),
            },
        };

        self.tape.push(op);
        Ok(output_id)
    }

    /// Exponential function
    pub fn exp(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::Exp, input)
    }

    /// Natural logarithm
    pub fn log(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::Log, input)
    }

    /// Sine function
    pub fn sin(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::Sin, input)
    }

    /// Cosine function
    pub fn cos(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::Cos, input)
    }

    /// Hyperbolic tangent
    pub fn tanh(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::Tanh, input)
    }

    /// Sigmoid function
    pub fn sigmoid(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::Sigmoid, input)
    }

    /// ReLU function
    pub fn relu(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::ReLU, input)
    }

    /// Matrix multiplication
    pub fn matmul(&mut self, lhs: usize, rhs: usize, dims: (usize, usize, usize)) -> Result<usize> {
        let output_id = self.tape.len();

        let op = ForwardOperation {
            optype: ForwardOpType::MatMul,
            inputs: vec![lhs, rhs],
            output: output_id,
            metadata: ForwardOpMetadata {
                partials: vec![T::zero(), T::zero()],
                shape: vec![dims.0, dims.2],
                data: ForwardOpData::MatMulDims {
                    m: dims.0,
                    n: dims.1,
                    k: dims.2,
                },
            },
        };

        self.tape.push(op);
        Ok(output_id)
    }

    /// Dot product
    pub fn dot(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(ForwardOpType::Dot, lhs, rhs)
    }

    /// Sum reduction
    pub fn sum(&mut self, input: usize, axis: Option<usize>) -> Result<usize> {
        let output_id = self.tape.len();

        let op = ForwardOperation {
            optype: ForwardOpType::Sum,
            inputs: vec![input],
            output: output_id,
            metadata: ForwardOpMetadata {
                partials: vec![T::zero()],
                shape: vec![], // Will be computed
                data: ForwardOpData::ReductionAxis(axis.unwrap_or(0)),
            },
        };

        self.tape.push(op);
        Ok(output_id)
    }

    /// Mean reduction
    pub fn mean(&mut self, input: usize, axis: Option<usize>) -> Result<usize> {
        let output_id = self.tape.len();

        let op = ForwardOperation {
            optype: ForwardOpType::Mean,
            inputs: vec![input],
            output: output_id,
            metadata: ForwardOpMetadata {
                partials: vec![T::zero()],
                shape: vec![], // Will be computed
                data: ForwardOpData::ReductionAxis(axis.unwrap_or(0)),
            },
        };

        self.tape.push(op);
        Ok(output_id)
    }

    /// L2 norm
    pub fn norm(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::Norm, input)
    }

    /// Compute forward pass with dual numbers
    pub fn forward_pass(
        &self,
        inputs: &HashMap<String, Array1<T>>,
        seed_direction: &Array1<T>,
    ) -> Result<Vec<VectorDual<T>>> {
        let mut values = Vec::with_capacity(self.tape.len());

        // Initialize with input values
        for op in &self.tape {
            match op.optype {
                ForwardOpType::Variable => {
                    // Find corresponding input value
                    let var_name = self
                        .variables
                        .iter()
                        .find(|(_, &id)| id == op.output)
                        .map(|(name_, _)| name_.clone())
                        .ok_or_else(|| {
                            OptimError::InvalidConfig("Variable not found".to_string())
                        })?;

                    let value = inputs
                        .get(&var_name)
                        .ok_or_else(|| {
                            OptimError::InvalidConfig("Input value not provided".to_string())
                        })?
                        .clone();

                    let tangent = if op.output < seed_direction.len() {
                        Array1::from_elem(value.len(), seed_direction[op.output])
                    } else {
                        Array1::zeros(value.len())
                    };

                    values.push(VectorDual::new(value, tangent));
                }
                ForwardOpType::Constant => {
                    if let ForwardOpData::ConstantValue(val) = op.metadata.data {
                        let value = Array1::from_elem(1, val);
                        let tangent = Array1::zeros(1);
                        values.push(VectorDual::new(value, tangent));
                    } else {
                        return Err(OptimError::InvalidConfig(
                            "Invalid constant data".to_string(),
                        ));
                    }
                }
                _ => {
                    // Compute operation
                    let result = self.compute_forward_operation(op, &values)?;
                    values.push(result);
                }
            }
        }

        Ok(values)
    }

    /// Compute Jacobian-vector product
    pub fn jacobian_vector_product(
        &self,
        inputs: &HashMap<String, Array1<T>>,
        output_id: usize,
        direction: &Array1<T>,
    ) -> Result<Array1<T>> {
        let results = self.forward_pass(inputs, direction)?;

        if output_id >= results.len() {
            return Err(OptimError::InvalidConfig("Invalid output ID".to_string()));
        }

        Ok(results[output_id].tangent.clone())
    }

    /// Compute full Jacobian matrix
    pub fn jacobian_matrix(
        &self,
        inputs: &HashMap<String, Array1<T>>,
        output_id: usize,
        input_size: usize,
    ) -> Result<Array2<T>> {
        let mut jacobian = Array2::zeros((inputs.len(), input_size));

        // Compute Jacobian one column at a time using unit vectors
        for i in 0..input_size {
            let mut direction = Array1::zeros(input_size);
            direction[i] = T::one();

            let jvp = self.jacobian_vector_product(inputs, output_id, &direction)?;

            for (j, &val) in jvp.iter().enumerate() {
                if j < jacobian.nrows() {
                    jacobian[[j, i]] = val;
                }
            }
        }

        Ok(jacobian)
    }

    fn binary_op(&mut self, optype: ForwardOpType, lhs: usize, rhs: usize) -> Result<usize> {
        let output_id = self.tape.len();

        let op = ForwardOperation {
            optype,
            inputs: vec![lhs, rhs],
            output: output_id,
            metadata: ForwardOpMetadata {
                partials: vec![T::zero(), T::zero()],
                shape: vec![], // Will be inferred
                data: ForwardOpData::None,
            },
        };

        self.tape.push(op);
        Ok(output_id)
    }

    fn unary_op(&mut self, optype: ForwardOpType, input: usize) -> Result<usize> {
        let output_id = self.tape.len();

        let op = ForwardOperation {
            optype,
            inputs: vec![input],
            output: output_id,
            metadata: ForwardOpMetadata {
                partials: vec![T::zero()],
                shape: vec![], // Will be inferred
                data: ForwardOpData::None,
            },
        };

        self.tape.push(op);
        Ok(output_id)
    }

    fn compute_forward_operation(
        &self,
        op: &ForwardOperation<T>,
        values: &[VectorDual<T>],
    ) -> Result<VectorDual<T>> {
        match op.optype {
            ForwardOpType::Add => {
                let lhs = &values[op.inputs[0]];
                let rhs = &values[op.inputs[1]];
                let value = &lhs.value + &rhs.value;
                let tangent = &lhs.tangent + &rhs.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Subtract => {
                let lhs = &values[op.inputs[0]];
                let rhs = &values[op.inputs[1]];
                let value = &lhs.value - &rhs.value;
                let tangent = &lhs.tangent - &rhs.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Multiply => {
                let lhs = &values[op.inputs[0]];
                let rhs = &values[op.inputs[1]];

                // Element-wise multiplication: (u*v)' = u'*v + u*v'
                let value = &lhs.value * &rhs.value;
                let tangent = &lhs.tangent * &rhs.value + &lhs.value * &rhs.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Divide => {
                let lhs = &values[op.inputs[0]];
                let rhs = &values[op.inputs[1]];

                // Division rule: (u/v)' = (u'*v - u*v') / v^2
                let value = &lhs.value / &rhs.value;
                let numerator = &lhs.tangent * &rhs.value - &lhs.value * &rhs.tangent;
                let denominator = &rhs.value * &rhs.value;
                let tangent = numerator / denominator;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Power => {
                let base = &values[op.inputs[0]];
                if let ForwardOpData::PowerExponent(exp) = op.metadata.data {
                    // Power rule: (u^n)' = n * u^(n-1) * u'
                    let value = base.value.mapv(|x| x.powf(exp));
                    let derivative = base.value.mapv(|x| exp * x.powf(exp - T::one()));
                    let tangent = derivative * &base.tangent;
                    Ok(VectorDual::new(value, tangent))
                } else {
                    Err(OptimError::InvalidConfig(
                        "Invalid power operation".to_string(),
                    ))
                }
            }
            ForwardOpType::Exp => {
                let input = &values[op.inputs[0]];
                // (e^u)' = e^u * u'
                let value = input.value.mapv(|x| x.exp());
                let tangent = &value * &input.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Log => {
                let input = &values[op.inputs[0]];
                // (ln(u))' = u' / u
                let value = input.value.mapv(|x| x.ln());
                let tangent = &input.tangent / &input.value;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Sin => {
                let input = &values[op.inputs[0]];
                // (sin(u))' = cos(u) * u'
                let value = input.value.mapv(|x| x.sin());
                let derivative = input.value.mapv(|x| x.cos());
                let tangent = derivative * &input.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Cos => {
                let input = &values[op.inputs[0]];
                // (cos(u))' = -sin(u) * u'
                let value = input.value.mapv(|x| x.cos());
                let derivative = input.value.mapv(|x| -x.sin());
                let tangent = derivative * &input.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Tanh => {
                let input = &values[op.inputs[0]];
                // (tanh(u))' = sech^2(u) * u' = (1 - tanh^2(u)) * u'
                let value = input.value.mapv(|x| x.tanh());
                let derivative = value.mapv(|y| T::one() - y * y);
                let tangent = derivative * &input.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Sigmoid => {
                let input = &values[op.inputs[0]];
                // (sigmoid(u))' = sigmoid(u) * (1 - sigmoid(u)) * u'
                let value = input.value.mapv(|x| T::one() / (T::one() + (-x).exp()));
                let derivative = value.mapv(|y| y * (T::one() - y));
                let tangent = derivative * &input.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::ReLU => {
                let input = &values[op.inputs[0]];
                // (ReLU(u))' = u' if u > 0, else 0
                let value = input
                    .value
                    .mapv(|x| if x > T::zero() { x } else { T::zero() });
                let derivative = input
                    .value
                    .mapv(|x| if x > T::zero() { T::one() } else { T::zero() });
                let tangent = derivative * &input.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Dot => {
                let lhs = &values[op.inputs[0]];
                let rhs = &values[op.inputs[1]];
                // (u·v)' = u'·v + u·v'
                let value = Array1::from_elem(1, lhs.value.dot(&rhs.value));
                let tangent =
                    Array1::from_elem(1, lhs.tangent.dot(&rhs.value) + lhs.value.dot(&rhs.tangent));
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Sum => {
                let input = &values[op.inputs[0]];
                // Sum derivative is sum of input derivatives
                let value = Array1::from_elem(1, input.value.sum());
                let tangent = Array1::from_elem(1, input.tangent.sum());
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Mean => {
                let input = &values[op.inputs[0]];
                // Mean derivative is mean of input derivatives
                let n = T::from(input.value.len()).unwrap();
                let value = Array1::from_elem(1, input.value.sum() / n);
                let tangent = Array1::from_elem(1, input.tangent.sum() / n);
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Norm => {
                let input = &values[op.inputs[0]];
                // ||u||' = (u·u')/ ||u||
                let norm = input.value.iter().map(|&x| x * x).sum::<T>().sqrt();
                let value = Array1::from_elem(1, norm);
                let tangent = if norm > T::zero() {
                    let dot_product = input.value.dot(&input.tangent);
                    Array1::from_elem(1, dot_product / norm)
                } else {
                    Array1::zeros(1)
                };
                Ok(VectorDual::new(value, tangent))
            }
            _ => Err(OptimError::InvalidConfig(
                "Unsupported operation".to_string(),
            )),
        }
    }

    /// Get computation graph statistics
    pub fn get_graph_stats(&self) -> ForwardModeStats {
        ForwardModeStats {
            num_operations: self.tape.len(),
            num_variables: self.variables.len(),
            memory_usage_estimate: self.estimate_memory_usage(),
            max_depth: self.compute_max_depth(),
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        self.tape.len() * std::mem::size_of::<ForwardOperation<T>>()
    }

    fn compute_max_depth(&self) -> usize {
        // Simplified depth computation
        self.tape.len()
    }
}

/// Forward-mode AD statistics
#[derive(Debug, Clone)]
pub struct ForwardModeStats {
    pub num_operations: usize,
    pub num_variables: usize,
    pub memory_usage_estimate: usize,
    pub max_depth: usize,
}

// Implement arithmetic operations for dual numbers
impl<T: Float + Default + Clone> std::ops::Add for DualNumber<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            tangent: self.tangent + rhs.tangent,
        }
    }
}

impl<T: Float + Default + Clone> std::ops::Sub for DualNumber<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            tangent: self.tangent - rhs.tangent,
        }
    }
}

impl<T: Float + Default + Clone> std::ops::Mul for DualNumber<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            tangent: self.tangent * rhs.value + self.value * rhs.tangent,
        }
    }
}

impl<T: Float + Default + Clone> std::ops::Div for DualNumber<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self {
            value: self.value / rhs.value,
            tangent: (self.tangent * rhs.value - self.value * rhs.tangent)
                / (rhs.value * rhs.value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_number_creation() {
        let dual = DualNumber::new(2.0, 1.0);
        assert_eq!(dual.value, 2.0);
        assert_eq!(dual.tangent, 1.0);
    }

    #[test]
    fn test_dual_number_arithmetic() {
        let x = DualNumber::new(3.0, 1.0);
        let y = DualNumber::new(2.0, 0.0);

        let sum = x.clone() + y.clone();
        assert_eq!(sum.value, 5.0);
        assert_eq!(sum.tangent, 1.0);

        let product = x * y;
        assert_eq!(product.value, 6.0);
        assert_eq!(product.tangent, 2.0);
    }

    #[test]
    fn test_forward_mode_engine() {
        let mut engine = ForwardModeEngine::<f64>::new();

        let x_val = Array1::from_vec(vec![2.0]);
        let x_id = engine.create_variable("x", x_val.clone());

        let y_val = Array1::from_vec(vec![3.0]);
        let y_id = engine.create_variable("y", y_val.clone());

        let sum_id = engine.add(x_id, y_id).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x_val);
        inputs.insert("y".to_string(), y_val);

        let direction = Array1::from_vec(vec![1.0, 0.0]);
        let results = engine.forward_pass(&inputs, &direction).unwrap();

        assert!(results.len() > sum_id);
        assert_eq!(results[sum_id].value[0], 5.0);
    }

    #[test]
    fn test_vector_dual_operations() {
        let value1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let tangent1 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let dual1 = VectorDual::new(value1, tangent1);

        assert_eq!(dual1.value.len(), 3);
        assert_eq!(dual1.tangent.len(), 3);
        assert_eq!(dual1.tangent[0], 1.0);
    }
}
