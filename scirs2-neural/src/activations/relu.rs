//! ReLU activation function implementations including ReLU, LeakyReLU, and ELU

use crate::activations::Activation;
use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;
/// Rectified Linear Unit (ReLU) activation function.
///
/// The ReLU function returns the input if it's positive, and 0 otherwise:
/// f(x) = max(0, x)
/// # Examples
/// ```
/// use scirs2_neural::activations::ReLU;
/// use scirs2_neural::activations::Activation;
/// use scirs2_core::ndarray::Array;
/// let relu = ReLU::new();
/// let input = Array::from_vec(vec![1.0, -1.0, 2.0, -2.0]).into_dyn();
/// let output = relu.forward(&input).unwrap();
/// assert_eq!(output, Array::from_vec(vec![1.0, 0.0, 2.0, 0.0]).into_dyn());
#[derive(Debug, Clone, Copy)]
pub struct ReLU {
    /// Alpha parameter for leaky ReLU.
    /// If alpha = 0, it's a standard ReLU.
    /// If alpha > 0, it's a leaky ReLU.
    alpha: f64,
}
impl ReLU {
    /// Create a new ReLU activation function.
    pub fn new() -> Self {
        Self { alpha: 0.0 }
    }
    /// Create a new Leaky ReLU activation function with given alpha.
    pub fn leaky(alpha: f64) -> Self {
        Self { _alpha }
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug> Activation<F> for ReLU {
    fn forward(&self, input: &Array<F, scirs2_core::ndarray::IxDyn>) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let alpha = F::from(self.alpha).ok_or_else(|| {
            NeuralError::InferenceError(
                "Could not convert alpha to the required float type".to_string(),
            )
        })?;
        let mut output = input.clone();
        let zero = F::zero();
        Zip::from(&mut output).for_each(|x| {
            if *x < zero {
                *x = alpha * *x;
            }
        });
        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        output: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let one = F::one();
        let zero = F::zero();
        let alpha = F::from(self.alpha).ok_or_else(|| {
            NeuralError::InferenceError(
                "Could not convert alpha to the required float type".to_string(),
            )
        })?;
        // Create derivative mask based on the _output (1 where x > 0, alpha where x <= 0)
        let mut mask = Array::from_elem(_output.dim(), one);
        Zip::from(&mut mask).and(_output).for_each(|mask_val, &out| {
            if out <= zero {
                *mask_val = alpha;
            }
        });
        // Multiply element-wise with the gradient from the next layer
        let mut grad_input = Array::zeros(grad_output.raw_dim());
        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(&mask)
            .for_each(|grad_in, &grad_out, &mask_val| {
                *grad_in = grad_out * mask_val;
            });
        Ok(grad_input)
    }
}

/// Leaky Rectified Linear Unit (LeakyReLU) activation function.
/// The LeakyReLU function is similar to ReLU but allows a small, non-zero
/// gradient when the unit is not active:
/// f(x) = max(alpha*x, x) or f(x) = x if x > 0, alpha*x otherwise
/// use scirs2_neural::activations::LeakyReLU;
/// let leaky_relu = LeakyReLU::new(0.01);
/// let output = leaky_relu.forward(&input).unwrap();
/// assert_eq!(output, Array::from_vec(vec![1.0, -0.01, 2.0, -0.02]).into_dyn());
pub struct LeakyReLU {
    /// Alpha parameter for leaky ReLU (small positive value)
    alpha: f64,
}

impl LeakyReLU {
    /// Create a new LeakyReLU activation function with given alpha.
    pub fn new(alpha: f64) -> Self {
        Self { _alpha }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01) // Common default for LeakyReLU
    }
}

impl<F: Float + Debug> Activation<F> for LeakyReLU {
    fn forward(&self, input: &Array<F, scirs2_core::ndarray::IxDyn>) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        // Use ReLU implementation with the alpha parameter
        let relu = ReLU::leaky(self.alpha);
        relu.forward(input)
    }

    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        output: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        // Use ReLU implementation with the alpha parameter
        let relu = ReLU::leaky(self.alpha);
        relu.backward(grad_output_output)
    }
}
