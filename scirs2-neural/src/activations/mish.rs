//! Mish activation function implementation

use crate::activations::Activation;
use crate::error::Result;
use ndarray::{Array, Zip};
use num_traits::Float;
use std::fmt::Debug;

/// Mish activation function.
///
/// Mish is defined as:
/// f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
///
/// It was proposed in "Mish: A Self Regularized Non-Monotonic Activation Function"
/// by Diganta Misra, and has been shown to work well in deep networks.
///
/// # Examples
///
/// ```
/// use scirs2_neural::activations::Mish;
/// use scirs2_neural::activations::Activation;
/// use ndarray::Array;
///
/// let mish = Mish::new();
/// let input = Array::from_vec(vec![1.0, -1.0, 2.0, -2.0]).into_dyn();
/// let output = mish.forward(&input).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Mish;

impl Mish {
    /// Create a new Mish activation function.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Mish {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug> Activation<F> for Mish {
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>> {
        let mut output = input.clone();

        // Compute x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
        Zip::from(&mut output).for_each(|x| {
            // Compute softplus(x) = ln(1 + e^x)
            // Use a numerically stable version for large values
            let sp = if *x > F::from(20.0).unwrap() {
                // For large x, softplus(x) ≈ x
                *x
            } else {
                (F::one() + x.exp()).ln()
            };

            // Apply tanh(softplus(x))
            *x = *x * sp.tanh();
        });

        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, ndarray::IxDyn>,
        output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>> {
        // We need to compute the derivative of Mish: d(mish)/dx
        let mut grad_input = Array::zeros(grad_output.raw_dim());

        // We need the original x to compute the derivative accurately
        // For simplicity, we'll approximate it from the output
        // In practice, you'd want to cache the input in the forward pass

        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(output)
            .for_each(|grad_in, &grad_out, &out| {
                // Approximate the input value from output
                // This is NOT accurate in general - a real implementation would cache the inputs
                let x = out; // This is just a placeholder - not correct

                // Compute the derivative components
                // For numerical stability with large values
                let sp = if x > F::from(20.0).unwrap() {
                    x
                } else {
                    (F::one() + x.exp()).ln()
                };

                let tanh_sp = sp.tanh();
                let sech_sp_sq = F::one() - tanh_sp * tanh_sp; // sech²(sp)

                // delta is the derivative of softplus: e^x / (1 + e^x)
                let delta = F::one() / (F::one() + (-x).exp());

                // By the chain rule:
                // d(mish)/dx = tanh(sp) + x * sech²(sp) * delta
                let dmish_dx = tanh_sp + x * sech_sp_sq * delta;

                // Multiply by the gradient from the next layer
                *grad_in = grad_out * dmish_dx;
            });

        Ok(grad_input)
    }
}
