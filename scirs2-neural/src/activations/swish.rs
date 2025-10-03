//! Swish activation function implementation

use crate::activations::Activation;
use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Swish activation function, also known as SiLU (Sigmoid-weighted Linear Unit).
///
/// Swish(x) = x * sigmoid(β * x), where β is a trainable parameter.
/// When β = 1, it reduces to the SiLU function.
///
/// # Examples
/// ```
/// use scirs2_neural::activations::Swish;
/// use scirs2_neural::activations::Activation;
/// use scirs2_core::ndarray::Array;
/// 
/// let swish = Swish::new(1.0);
/// let input = Array::from_vec(vec![1.0, -1.0, 2.0, -2.0]).into_dyn();
/// let output = swish.forward(&input).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Swish {
    /// The beta parameter
    beta: f64,
}

impl Swish {
    /// Create a new Swish activation function with specified beta.
    ///
    /// # Arguments
    /// * `beta` - The beta parameter for the swish function
    pub fn new(beta: f64) -> Self {
        Self { _beta }
    }
}

impl Default for Swish {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl<F: Float + Debug> Activation<F> for Swish {
    fn forward(&self, input: &Array<F, scirs2_core::ndarray::IxDyn>) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
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
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let beta = F::from(self.beta).ok_or_else(|| {
            NeuralError::InferenceError(
                "Could not convert beta to the required float type".to_string(),
            )
        })?;

        let mut grad_input = Array::zeros(grad_output.raw_dim());

        // Swish'(x) = β * swish(x) + sigmoid(β * x) * (1 - β * swish(x))
        // This is computed as: sigmoid(β * x) * (β * x * (1 - sigmoid(β * x)) + 1), Zip::from(&mut grad_input)
            .and(grad_output)
            .and(input)
            .for_each(|grad_in, &grad_out, &x| {
                let beta_x = beta * x;
                let sigmoid_beta_x = F::one() / (F::one() + (-beta_x).exp());
                let swish_x = x * sigmoid_beta_x;
                
                // Derivative: β * swish(x) + sigmoid(β * x) * (1 - β * swish(x))
                let derivative = beta * swish_x + sigmoid_beta_x * (F::one() - beta * swish_x);
                *grad_in = grad_out * derivative;
            });

        Ok(grad_input)
    }
}

impl<F: Float + Debug + ScalarOperand> Layer<F> for Swish {
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
        <Self as Activation<F>>::backward(self, grad_output, input)
    }

    fn update(&mut self, learningrate: F) -> Result<()> {
        Ok(())
    }
}
