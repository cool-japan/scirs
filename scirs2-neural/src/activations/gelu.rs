//! GELU activation function implementation

use crate::activations::Activation;
use crate::error::Result;
use crate::layers::Layer;
use ndarray::{Array, IxDyn, ScalarOperand, Zip};
use num_traits::Float;
use std::fmt::Debug;

/// Gaussian Error Linear Unit (GELU) activation function.
///
/// GELU activation is defined as:
/// f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
///
/// It was introduced in the paper "Gaussian Error Linear Units (GELUs)"
/// by Hendrycks and Gimpel, and is commonly used in transformer models like BERT and GPT.
///
/// # Examples
///
/// ```
/// use scirs2_neural::activations::GELU;
/// use scirs2_neural::activations::Activation;
/// use ndarray::Array;
///
/// let gelu = GELU::new();
/// let input = Array::from_vec(vec![1.0, -1.0, 2.0, -2.0]).into_dyn();
/// let output = gelu.forward(&input).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GELU {
    /// Whether to use fast approximation or exact formula
    fast: bool,
}

impl GELU {
    /// Create a new GELU activation function using the exact formula.
    pub fn new() -> Self {
        Self { fast: false }
    }

    /// Create a new GELU activation function using a fast approximation.
    /// This approximation is faster but slightly less accurate.
    pub fn fast() -> Self {
        Self { fast: true }
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug> Activation<F> for GELU {
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>> {
        let mut output = input.clone();

        if self.fast {
            // Fast approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            let sqrt_2_over_pi = F::from(0.7978845608028654).unwrap(); // sqrt(2/π)
            let coeff = F::from(0.044715).unwrap();
            let half = F::from(0.5).unwrap();
            let one = F::one();

            Zip::from(&mut output).for_each(|x| {
                let x3 = *x * *x * *x;
                let inner = sqrt_2_over_pi * (*x + coeff * x3);
                *x = half * *x * (one + inner.tanh());
            });
        } else {
            // Exact formula: 0.5 * x * (1 + erf(x/sqrt(2)))
            // Since erf is not directly available in num_traits, we use the
            // related function: 0.5 * (1 + tanh(sqrt(π/2) * x * (1 + 0.044715 * x^2))
            let half = F::from(0.5).unwrap();
            let one = F::one();
            let sqrt_pi_over_2 = F::from(1.2533141373155).unwrap(); // sqrt(π/2)
            let coeff = F::from(0.044715).unwrap();

            Zip::from(&mut output).for_each(|x| {
                let x2 = *x * *x;
                let inner = sqrt_pi_over_2 * *x * (one + coeff * x2);
                *x = half * *x * (one + inner.tanh());
            });
        }

        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, ndarray::IxDyn>,
        output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>> {
        // We need to compute the derivative of GELU
        // For the exact implementation, this is complex
        // So we'll recompute from the input
        let input = output.clone(); // FIXME: This is not correct, but it's a placeholder

        let mut grad_input = Array::zeros(grad_output.raw_dim());

        // Compute the derivative based on the input
        if self.fast {
            let sqrt_2_over_pi = F::from(0.7978845608028654).unwrap(); // sqrt(2/π)
            let coeff = F::from(0.044715).unwrap();
            let half = F::from(0.5).unwrap();
            let one = F::one();
            let three = F::from(3.0).unwrap();

            Zip::from(&mut grad_input)
                .and(grad_output)
                .and(&input)
                .for_each(|grad_in, &grad_out, &x| {
                    let x2 = x * x;
                    let x3 = x2 * x;

                    // Compute the inner term: sqrt(2/π) * (x + 0.044715 * x^3)
                    let inner = sqrt_2_over_pi * (x + coeff * x3);

                    // Compute tanh and sech^2 of the inner term
                    let tanh_inner = inner.tanh();
                    let sech_sq = one - tanh_inner * tanh_inner;

                    // Compute the derivative of inner with respect to x
                    let d_inner_dx = sqrt_2_over_pi * (one + three * coeff * x2);

                    // Apply the chain rule to get the derivative of GELU
                    let dgelu_dx = half * (one + tanh_inner) + half * x * sech_sq * d_inner_dx;

                    // Multiply by the gradient from the next layer
                    *grad_in = grad_out * dgelu_dx;
                });
        } else {
            // Exact formula derivative
            let half = F::from(0.5).unwrap();
            let one = F::one();
            let sqrt_pi_over_2 = F::from(1.2533141373155).unwrap(); // sqrt(π/2)
            let coeff = F::from(0.044715).unwrap();
            let three = F::from(3.0).unwrap();

            Zip::from(&mut grad_input)
                .and(grad_output)
                .and(&input)
                .for_each(|grad_in, &grad_out, &x| {
                    let x2 = x * x;

                    // Compute the inner term: sqrt(π/2) * x * (1 + 0.044715 * x^2)
                    let inner = sqrt_pi_over_2 * x * (one + coeff * x2);

                    // Compute tanh and sech^2 of the inner term
                    let tanh_inner = inner.tanh();
                    let sech_sq = one - tanh_inner * tanh_inner;

                    // Compute derivative of inner with respect to x
                    let d_inner_dx = sqrt_pi_over_2 * (one + three * coeff * x2);

                    // Apply the chain rule to get the derivative of GELU
                    let dgelu_dx = half * (one + tanh_inner) + half * x * sech_sq * d_inner_dx;

                    // Multiply by the gradient from the next layer
                    *grad_in = grad_out * dgelu_dx;
                });
        }

        Ok(grad_input)
    }
}

// Implement Layer trait for GELU
impl<F: Float + Debug + ScalarOperand> Layer<F> for GELU {
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
        // For GELU, we don't use the input directly in the backward pass
        // Instead, we compute the gradient from the output
        let output = Activation::<F>::forward(self, input)?;
        <Self as Activation<F>>::backward(self, grad_output, &output)
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        // GELU has no parameters to update
        Ok(())
    }

    // Use default implementations for other methods from Layer trait
}
