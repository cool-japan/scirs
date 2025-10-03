//! Sigmoid activation function implementation

use crate::activations::Activation;
use crate::error::Result;
use scirs2_core::ndarray::{Array, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;
/// Sigmoid activation function.
///
/// The sigmoid function is defined as:
/// f(x) = 1 / (1 + exp(-x))
/// It maps any input value to a value between 0 and 1.
/// # Examples
/// ```
/// use scirs2_neural::activations::Sigmoid;
/// use scirs2_neural::activations::Activation;
/// use scirs2_core::ndarray::Array;
/// let sigmoid = Sigmoid::new();
/// let input = Array::from_vec(vec![0.0f64, 1.0, -1.0]).into_dyn();
/// let output = sigmoid.forward(&input).unwrap();
/// // Check that values are in the expected range
/// assert!(output.iter().all(|&x| x >= 0.0f64 && x <= 1.0f64));
/// // Sigmoid(0) should be 0.5
/// assert!((output[0] - 0.5f64).abs() < 1e-6);
#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;
impl Sigmoid {
    /// Create a new Sigmoid activation function.
    pub fn new() -> Self {
        Self
    }
}
impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug> Activation<F> for Sigmoid {
    fn forward(&self, input: &Array<F, scirs2_core::ndarray::IxDyn>) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let one = F::one();
        let mut output = input.clone();
        Zip::from(&mut output).for_each(|x| {
            *x = one / (one + (-*x).exp());
        });
        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        output: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let one = F::one();
        let mut grad_input = Array::zeros(_output.raw_dim());
        // For sigmoid: derivative = sigmoid(x) * (1 - sigmoid(x))
        // _output already contains sigmoid(x), so we compute _output * (1 - output)
        // grad_input = grad_output * (_output * (1 - output))
        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(_output)
            .for_each(|grad_in, &grad_out, &out| {
                *grad_in = grad_out * out * (one - out);
            });
        Ok(grad_input)
    }
}
