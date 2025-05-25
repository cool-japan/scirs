//! Softmax activation function implementation

use crate::activations::Activation;
use crate::error::{NeuralError, Result};
use ndarray::{Array, Axis};
use num_traits::Float;
use std::fmt::Debug;

/// Softmax activation function.
///
/// The softmax function is defined as:
/// f(x_i) = exp(x_i) / sum_j(exp(x_j))
///
/// It transforms a vector of real values into a probability distribution.
///
/// # Examples
///
/// ```
/// use scirs2_neural::activations::{Softmax, Activation};
/// use ndarray::arr1;
///
/// // Create softmax activation for 1D array (axis 0)
/// let softmax = Softmax::new(0);
/// let input = arr1(&[1.0f64, 2.0, 3.0]).into_dyn();
/// let output = softmax.forward(&input).unwrap();
///
/// // Check that the output sums to 1.0
/// let sum: f64 = output.sum();
/// assert!((sum - 1.0).abs() < 1e-6);
///
/// // Check that all values are between 0 and 1
/// for val in output.iter() {
///     assert!(*val >= 0.0 && *val <= 1.0);
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Softmax {
    /// Axis along which to apply softmax (typically 0 for 1D arrays or the last dimension for batches)
    axis: usize,
}

impl Softmax {
    /// Create a new Softmax activation function.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to compute the softmax.
    pub fn new(axis: usize) -> Self {
        Self { axis }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<F: Float + Debug> Activation<F> for Softmax {
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>> {
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
        grad_output: &Array<F, ndarray::IxDyn>,
        output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>> {
        // Softmax backward pass: grad_input = softmax * (grad_output - sum(grad_output * softmax))
        // This implements the full Jacobian-vector product for softmax

        // Special case for 1D arrays
        if output.ndim() == 1 && self.axis == 0 {
            // Compute dot product of grad_output and output (softmax values)
            let dot_product = grad_output
                .iter()
                .zip(output.iter())
                .map(|(&g, &s)| g * s)
                .fold(F::zero(), |a, b| a + b);

            // Compute gradient: s * (grad_output - dot_product)
            let grad_input = output
                .iter()
                .zip(grad_output.iter())
                .map(|(&s, &g)| s * (g - dot_product))
                .collect::<Vec<_>>();

            return Ok(Array::from_vec(grad_input).into_dyn());
        }

        // General case for multi-dimensional arrays
        // Compute sum(grad_output * softmax) along the softmax axis
        let weighted_sum = (grad_output * output).sum_axis(Axis(self.axis));

        // Broadcast the weighted sum back to original shape
        let mut sum_shape = output.shape().to_vec();
        sum_shape[self.axis] = 1;
        let weighted_sum_reshaped = weighted_sum.into_shape_with_order(sum_shape)?;
        let weighted_sum_broadcast = weighted_sum_reshaped.broadcast(output.shape()).unwrap();

        // Compute gradient: softmax * (grad_output - weighted_sum)
        let grad_input = output * (grad_output - &weighted_sum_broadcast);

        Ok(grad_input)
    }
}
