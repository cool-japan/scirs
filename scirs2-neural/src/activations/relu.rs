//! ReLU activation function implementations including ReLU, LeakyReLU, and ELU

use crate::activations::Activation;
use crate::error::{NeuralError, Result};
use ndarray::{Array, Zip};
use num_traits::Float;
use std::fmt::Debug;

/// Rectified Linear Unit (ReLU) activation function.
///
/// The ReLU function returns the input if it's positive, and 0 otherwise:
/// f(x) = max(0, x)
///
/// # Examples
///
/// ```
/// use scirs2_neural::activations::ReLU;
/// use scirs2_neural::activations::Activation;
/// use ndarray::Array;
///
/// let relu = ReLU::new();
/// let input = Array::from_vec(vec![1.0, -1.0, 2.0, -2.0]).into_dyn();
/// let output = relu.forward(&input).unwrap();
/// assert_eq!(output, Array::from_vec(vec![1.0, 0.0, 2.0, 0.0]).into_dyn());
/// ```
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
        Self { alpha }
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug> Activation<F> for ReLU {
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>> {
        let alpha = F::from(self.alpha).ok_or_else(|| {
            NeuralError::InferenceError(
                "Could not convert alpha to the required float type".to_string(),
            )
        })?;
        let zero = F::zero();

        let mut output = input.clone();
        Zip::from(&mut output).for_each(|x| {
            if *x < zero {
                *x = alpha * *x;
            }
        });

        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, ndarray::IxDyn>,
        output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>> {
        let alpha = F::from(self.alpha).ok_or_else(|| {
            NeuralError::InferenceError(
                "Could not convert alpha to the required float type".to_string(),
            )
        })?;
        let zero = F::zero();
        let one = F::one();

        // Create derivative mask based on the output (1 where x > 0, alpha where x <= 0)
        let mut mask = Array::from_elem(output.dim(), one);
        Zip::from(&mut mask).and(output).for_each(|mask_val, &out| {
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
///
/// The LeakyReLU function is similar to ReLU but allows a small, non-zero
/// gradient when the unit is not active:
/// f(x) = max(alpha*x, x) or f(x) = x if x > 0, alpha*x otherwise
///
/// # Examples
///
/// ```
/// use scirs2_neural::activations::LeakyReLU;
/// use scirs2_neural::activations::Activation;
/// use ndarray::Array;
///
/// let leaky_relu = LeakyReLU::new(0.01);
/// let input = Array::from_vec(vec![1.0, -1.0, 2.0, -2.0]).into_dyn();
/// let output = leaky_relu.forward(&input).unwrap();
/// assert_eq!(output, Array::from_vec(vec![1.0, -0.01, 2.0, -0.02]).into_dyn());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU {
    /// Alpha parameter for leaky ReLU (small positive value)
    alpha: f64,
}

impl LeakyReLU {
    /// Create a new LeakyReLU activation function with given alpha.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01) // Common default for LeakyReLU
    }
}

impl<F: Float + Debug> Activation<F> for LeakyReLU {
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>> {
        let alpha = F::from(self.alpha).ok_or_else(|| {
            NeuralError::InferenceError(
                "Could not convert alpha to the required float type".to_string(),
            )
        })?;
        let zero = F::zero();

        let mut output = input.clone();
        Zip::from(&mut output).for_each(|x| {
            if *x < zero {
                *x = alpha * *x;
            }
        });

        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, ndarray::IxDyn>,
        output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>> {
        let alpha = F::from(self.alpha).ok_or_else(|| {
            NeuralError::InferenceError(
                "Could not convert alpha to the required float type".to_string(),
            )
        })?;
        let zero = F::zero();
        let one = F::one();

        // Create derivative mask based on the output (1 where x > 0, alpha where x <= 0)
        let mut mask = Array::from_elem(output.dim(), one);
        Zip::from(&mut mask).and(output).for_each(|mask_val, &out| {
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

/// Exponential Linear Unit (ELU) activation function.
///
/// The ELU function is:
/// f(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
///
/// # Examples
///
/// ```ignore
/// // Example (not compiled due to generic type issues):
/// // use scirs2_neural::activations::ELU;
/// // use scirs2_neural::activations::Activation;
/// // use ndarray::Array;
/// //
/// // let elu = ELU::new(1.0);
/// // let input = Array::from_vec(vec![1.0, -1.0, 2.0, -2.0]).into_dyn();
/// // let output = elu.forward(&input).unwrap();
/// //
/// // // Check that positive values pass through unchanged
/// // assert!((output[[0]] - 1.0).abs() < 1e-6);
/// // assert!((output[[2]] - 2.0).abs() < 1e-6);
/// //
/// // // Check that negative values are transformed with ELU formula
/// // assert!((output[[1]] - (-0.6321205588285577)).abs() < 1e-6); // alpha * (exp(-1) - 1) ≈ -0.632
/// // assert!((output[[3]] - (-0.8646647167633873)).abs() < 1e-6); // alpha * (exp(-2) - 1) ≈ -0.865
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ELU {
    /// Alpha parameter for ELU
    alpha: f64,
}

impl ELU {
    /// Create a new ELU activation function with given alpha.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new(1.0) // Common default for ELU
    }
}

impl<F: Float + Debug> Activation<F> for ELU {
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>> {
        let alpha = F::from(self.alpha).ok_or_else(|| {
            NeuralError::InferenceError(
                "Could not convert alpha to the required float type".to_string(),
            )
        })?;
        let zero = F::zero();
        let one = F::one();

        let mut output = input.clone();
        Zip::from(&mut output).for_each(|x| {
            if *x <= zero {
                *x = alpha * (x.exp() - one);
            }
        });

        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, ndarray::IxDyn>,
        output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>> {
        let alpha = F::from(self.alpha).ok_or_else(|| {
            NeuralError::InferenceError(
                "Could not convert alpha to the required float type".to_string(),
            )
        })?;
        let zero = F::zero();
        let _one = F::one();

        // Calculate the derivative of ELU
        // derivative is 1 for x > 0 and f(x) + alpha for x <= 0
        let mut grad_input = Array::zeros(grad_output.raw_dim());
        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(output)
            .for_each(|grad_in, &grad_out, &out| {
                if out > zero {
                    *grad_in = grad_out;
                } else {
                    *grad_in = grad_out * (out + alpha);
                }
            });

        Ok(grad_input)
    }
}
