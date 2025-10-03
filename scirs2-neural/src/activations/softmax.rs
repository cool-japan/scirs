//! Softmax activation function implementation

use crate::activations::Activation;
use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, Axis, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Softmax activation function.
///
/// Softmax applies the softmax function along a specified axis.
/// It's commonly used as the final activation in classification networks.
///
/// # Examples
/// ```
/// use scirs2_neural::activations::Softmax;
/// use scirs2_neural::activations::Activation;
/// use scirs2_core::ndarray::Array;
/// 
/// let softmax = Softmax::new(0);
/// let input = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
/// let output = softmax.forward(&input).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Softmax {
    /// The axis along which to apply softmax
    axis: usize,
}

impl Softmax {
    /// Create a new Softmax activation function.
    ///
    /// # Arguments
    /// * `axis` - The axis along which to apply softmax
    pub fn new(axis: usize) -> Self {
        Self { _axis }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<F: Float + Debug> Activation<F> for Softmax {
    fn forward(&self, input: &Array<F, scirs2_core::ndarray::IxDyn>) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        if input.ndim() <= self.axis {
            return Err(NeuralError::InferenceError(format!(
                "Softmax axis {} is out of bounds for input with {} dimensions",
                self.axis,
                input.ndim()
            )));
        }

        // Special case for 1D arrays
        if input.ndim() == 1 && self.axis == 0 {
            // Find max for numerical stability
            let max_val = input.fold(F::neg_infinity(), |a, &b| a.max(b));
            // Compute exp(x - max)
            let mut output = input.clone();
            for val in output.iter_mut() {
                *val = (*val - max_val).exp();
            }
            // Compute sum
            let sum = output.fold(F::zero(), |a, &b| a + b);
            // Normalize
            for val in output.iter_mut() {
                *val = *val / sum;
            }
            return Ok(output);
        }

        // General case for multi-dimensional arrays
        // Numerical stability: subtract the maximum value to avoid overflow
        let max_vals = input.map_axis(Axis(self.axis), |view| {
            view.fold(F::neg_infinity(), |a, &b| a.max(b))
        });

        let mut output = input.clone();
        
        // Apply exp(x - max) for numerical stability
        for (mut out_subview, &max_val) in
            output.axis_iter_mut(Axis(self.axis)).zip(max_vals.iter())
        {
            for val in out_subview.iter_mut() {
                *val = (*val - max_val).exp();
            }
        }

        // Calculate sum of exp(x - max) for each sample
        let sum_vals = output.map_axis(Axis(self.axis), |view| view.fold(F::zero(), |a, &b| a + b));

        // Normalize to get softmax values
        for (mut out_subview, &sum_val) in
            output.axis_iter_mut(Axis(self.axis)).zip(sum_vals.iter())
        {
            for val in out_subview.iter_mut() {
                *val = *val / sum_val;
            }
        }

        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        output: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        // Softmax backward pass: grad_input = softmax * (grad_output - sum(grad_output * softmax))
        // This implements the full Jacobian-vector product for softmax
        
        if output.ndim() == 1 && self.axis == 0 {
            // Compute dot product of grad_output and _output (softmax values)
            let dot_product = grad_output
                .iter()
                .zip(_output.iter())
                .map(|(&g, &s)| g * s)
                .fold(F::zero(), |a, b| a + b);

            // Compute gradient: s * (grad_output - dot_product)
            let grad_input = _output
                .iter()
                .zip(grad_output.iter())
                .map(|(&s, &g)| s * (g - dot_product))
                .collect::<Vec<_>>();

            return Ok(Array::from_vec(grad_input).into_dyn());
        }

        // Multi-dimensional case
        // Compute sum(grad_output * softmax) along the softmax axis
        let weighted_sum = (grad_output * output).sum_axis(Axis(self.axis));

        // Broadcast the weighted sum back to original shape
        let mut sumshape = output.shape().to_vec();
        sumshape[self.axis] = 1;
        let weighted_sum_reshaped = weighted_sum.into_shape_with_order(sumshape)?;
        let weighted_sum_broadcast = weighted_sum_reshaped.broadcast(_output.shape()).unwrap();

        // Compute gradient: softmax * (grad_output - weighted_sum)
        let grad_input = _output * (grad_output - &weighted_sum_broadcast);
        
        Ok(grad_input)
    }
}

impl<F: Float + Debug + ScalarOperand> Layer<F> for Softmax {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        <Self as Activation<F>>::forward(self, input)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // For softmax, we need the output, not the input for backward pass
        let _output = self.forward(input)?;
        <Self as Activation<F>>::backward(self, grad_output, &_output)
    }

    fn update(&mut self, learningrate: F) -> Result<()> {
        Ok(())
    }
}
