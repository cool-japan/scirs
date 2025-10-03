//! Minimal activation functions without Layer trait dependencies

use crate::error::Result;
use scirs2_core::ndarray::{Array, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Trait for activation functions
pub trait Activation<F> {
    /// Forward pass of the activation function
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>>;

    /// Backward pass of the activation function
    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>>;
}

/// GELU activation function
#[derive(Debug, Clone, Copy)]
pub struct GELU {
    fast: bool,
}

impl GELU {
    pub fn new() -> Self {
        Self { fast: false }
    }

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
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut output = input.clone();

        if self.fast {
            let sqrt_2_over_pi = F::from(0.7978845608028654).unwrap();
            let coeff = F::from(0.044715).unwrap();
            let half = F::from(0.5).unwrap();
            let one = F::one();

            Zip::from(&mut output).for_each(|x| {
                let x3 = *x * *x * *x;
                let inner = sqrt_2_over_pi * (*x + coeff * x3);
                *x = half * *x * (one + inner.tanh());
            });
        } else {
            let sqrt_pi_over_2 = F::from(1.2533141373155).unwrap();
            let coeff = F::from(0.044715).unwrap();
            let half = F::from(0.5).unwrap();
            let one = F::one();

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
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut grad_input = Array::zeros(grad_output.raw_dim());

        if self.fast {
            let sqrt_2_over_pi = F::from(0.7978845608028654).unwrap();
            let coeff = F::from(0.044715).unwrap();
            let half = F::from(0.5).unwrap();
            let one = F::one();
            let three = F::from(3.0).unwrap();

            Zip::from(&mut grad_input)
                .and(grad_output)
                .and(input)
                .for_each(|grad_in, &grad_out, &x| {
                    let x2 = x * x;
                    let x3 = x2 * x;
                    let inner = sqrt_2_over_pi * (x + coeff * x3);
                    let tanh_inner = inner.tanh();
                    let sech_sq = one - tanh_inner * tanh_inner;
                    let d_inner_dx = sqrt_2_over_pi * (one + three * coeff * x2);
                    let dgelu_dx = half * (one + tanh_inner) + half * x * sech_sq * d_inner_dx;
                    *grad_in = grad_out * dgelu_dx;
                });
        } else {
            let sqrt_pi_over_2 = F::from(1.2533141373155).unwrap();
            let coeff = F::from(0.044715).unwrap();
            let half = F::from(0.5).unwrap();
            let one = F::one();
            let three = F::from(3.0).unwrap();

            Zip::from(&mut grad_input)
                .and(grad_output)
                .and(input)
                .for_each(|grad_in, &grad_out, &x| {
                    let x2 = x * x;
                    let inner = sqrt_pi_over_2 * x * (one + coeff * x2);
                    let tanh_inner = inner.tanh();
                    let sech_sq = one - tanh_inner * tanh_inner;
                    let d_inner_dx = sqrt_pi_over_2 * (one + three * coeff * x2);
                    let dgelu_dx = half * (one + tanh_inner) + half * x * sech_sq * d_inner_dx;
                    *grad_in = grad_out * dgelu_dx;
                });
        }

        Ok(grad_input)
    }
}

/// Tanh activation function
#[derive(Debug, Clone, Copy)]
pub struct Tanh;

impl Tanh {
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
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
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

/// Sigmoid activation function
#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

impl Sigmoid {
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
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut output = input.clone();
        let one = F::one();
        Zip::from(&mut output).for_each(|x| {
            *x = one / (one + (-*x).exp());
        });
        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut grad_input = Array::zeros(grad_output.raw_dim());
        let one = F::one();

        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(input)
            .for_each(|grad_in, &grad_out, &x| {
                let sigmoid_x = one / (one + (-x).exp());
                let derivative = sigmoid_x * (one - sigmoid_x);
                *grad_in = grad_out * derivative;
            });

        Ok(grad_input)
    }
}

/// ReLU activation function
#[derive(Debug, Clone, Copy)]
pub struct ReLU {
    alpha: f64,
}

impl ReLU {
    pub fn new() -> Self {
        Self { alpha: 0.0 }
    }

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
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut output = input.clone();
        let zero = F::zero();
        let alpha = F::from(self.alpha).unwrap_or(zero);

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
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut grad_input = Array::zeros(grad_output.raw_dim());
        let zero = F::zero();
        let one = F::one();
        let alpha = F::from(self.alpha).unwrap_or(zero);

        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(input)
            .for_each(|grad_in, &grad_out, &x| {
                let derivative = if x > zero { one } else { alpha };
                *grad_in = grad_out * derivative;
            });

        Ok(grad_input)
    }
}

/// Softmax activation function
#[derive(Debug, Clone, Copy)]
pub struct Softmax {
    axis: isize,
}

impl Softmax {
    pub fn new(axis: isize) -> Self {
        Self { axis }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl<F: Float + Debug> Activation<F> for Softmax {
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut output = input.clone();

        // Simple softmax implementation for the last axis
        if self.axis == -1 || self.axis as usize == input.ndim() - 1 {
            // For 1D case or applying to last axis
            let max_val = input.fold(F::neg_infinity(), |acc, &x| if x > acc { x } else { acc });

            // Subtract max for numerical stability
            Zip::from(&mut output).for_each(|x| {
                *x = (*x - max_val).exp();
            });

            // Sum all exponentials
            let sum = output.sum();

            // Normalize
            Zip::from(&mut output).for_each(|x| {
                *x = *x / sum;
            });
        }

        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        // Forward pass to get softmax _output
        let softmax_output = self.forward(input)?;
        let mut grad_input = Array::zeros(grad_output.raw_dim());

        // For softmax: grad = softmax * (grad_out - (softmax * grad_out).sum())
        let sum_grad = Zip::from(&softmax_output)
            .and(grad_output)
            .fold(F::zero(), |acc, &s, &g| acc + s * g);

        Zip::from(&mut grad_input)
            .and(&softmax_output)
            .and(grad_output)
            .for_each(|grad_in, &s, &grad_out| {
                *grad_in = s * (grad_out - sum_grad);
            });

        Ok(grad_input)
    }
}
