//! Tanh activation function implementation

use crate::activations::Activation;
use crate::error::Result;
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Hyperbolic tangent (Tanh) activation function.
///
/// Tanh activation is defined as:
/// f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
///
/// It's similar to sigmoid but with output range [-1, 1].
///
/// # Examples
/// ```
/// use scirs2_neural::activations::Tanh;
/// use scirs2_neural::activations::Activation;
/// use scirs2_core::ndarray::Array;
/// 
/// let tanh = Tanh::new();
/// let input = Array::from_vec(vec![1.0, -1.0, 2.0, -2.0]).into_dyn();
/// let output = tanh.forward(&input).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Tanh;

impl Tanh {
    /// Create a new Tanh activation function.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug> Activation<F> for Tanh {
    fn forward(&self, input: &Array<F, scirs2_core::ndarray::IxDyn>) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut output = input.clone();
        Zip::from(&mut output).for_each(|x| {
            *x = x.tanh();
        });
        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut grad_input = Array::zeros(grad_output.raw_dim());
        
        // Derivative of tanh(x) is 1 - tanh^2(x)
        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(input)
            .for_each(|grad_in, &grad_out, &x| {
                let tanh_x = x.tanh();
                let derivative = F::one() - tanh_x * tanh_x;
                *grad_in = grad_out * derivative;
            });

        Ok(grad_input)
    }
}

impl<F: Float + Debug + ScalarOperand> Layer<F> for Tanh {
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
