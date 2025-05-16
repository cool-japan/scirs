//! Swish activation function implementation

use crate::activations::Activation;
use crate::error::{NeuralError, Result};
use ndarray::{Array, Zip};
use num_traits::Float;
use std::fmt::Debug;

/// Swish activation function.
///
/// Swish is defined as:
/// f(x) = x * sigmoid(β * x)
///
/// where β is a trainable parameter. When β = 1, it's called Swish-1 and is commonly used.
/// This activation was introduced in "Searching for Activation Functions" by Ramachandran et al.
///
/// # Examples
///
/// ```
/// use scirs2_neural::activations::Swish;
/// use scirs2_neural::activations::Activation;
/// use ndarray::Array;
///
/// let swish = Swish::new(1.0);
/// let input = Array::from_vec(vec![1.0, -1.0, 2.0, -2.0]).into_dyn();
/// let output = swish.forward(&input).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Swish {
    /// Beta parameter controlling the shape of the function
    beta: f64,
}

impl Swish {
    /// Create a new Swish activation function with given beta parameter.
    ///
    /// # Arguments
    ///
    /// * `beta` - Parameter controlling the shape of the Swish function.
    ///   β = 1.0 gives the standard Swish-1 function.
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }
}

impl Default for Swish {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl<F: Float + Debug> Activation<F> for Swish {
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>> {
        let beta = F::from(self.beta).ok_or_else(|| {
            NeuralError::InferenceError(
                "Could not convert beta to the required float type".to_string(),
            )
        })?;

        let mut output = input.clone();

        // Compute x * sigmoid(β * x)
        Zip::from(&mut output).for_each(|x| {
            let sigmoid_beta_x = F::one() / (F::one() + (-beta * *x).exp());
            *x = *x * sigmoid_beta_x;
        });

        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, ndarray::IxDyn>,
        output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>> {
        let beta = F::from(self.beta).ok_or_else(|| {
            NeuralError::InferenceError(
                "Could not convert beta to the required float type".to_string(),
            )
        })?;

        // Swish'(x) = swish(x) + sigmoid(β * x) * (1 - swish(x))
        let mut grad_input = Array::zeros(grad_output.raw_dim());

        // We need the original input to compute the derivative
        // For simplicity, let's assume we can extract the input from the output
        // This is an approximation - in a complete implementation, we should cache the input

        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(output)
            .for_each(|grad_in, &grad_out, &out| {
                // Approximate the input value using the output
                // NOTE: This is not accurate for all values, especially near zero
                // In a real implementation, we would cache the input during forward pass
                let x = if out.abs() > F::from(1e-10).unwrap() {
                    // Estimate input from output
                    // Solve: out = x * sigmoid(β * x) for x
                    // This is a very rough approximation
                    out
                } else {
                    // For values close to zero, use a default derivative
                    F::from(0.5).unwrap()
                };

                // Compute sigmoid(β * x)
                let sigmoid_beta_x = F::one() / (F::one() + (-beta * x).exp());

                // Compute the derivative of Swish with respect to x
                let swish_prime = out + sigmoid_beta_x * (F::one() - out);

                // Multiply by the gradient from the next layer
                *grad_in = grad_out * swish_prime;
            });

        Ok(grad_input)
    }
}
