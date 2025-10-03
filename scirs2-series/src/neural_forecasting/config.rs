//! Configuration and Common Types for Neural Forecasting
//!
//! This module contains common configuration structures, enums, and utility types
//! used across all neural forecasting architectures.

use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// Activation functions for neural networks
#[derive(Debug, Clone)]
pub enum ActivationFunction {
    /// Sigmoid activation
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Rectified Linear Unit
    ReLU,
    /// Gaussian Error Linear Unit
    GELU,
    /// Swish activation
    Swish,
    /// Linear activation (identity)
    Linear,
}

impl ActivationFunction {
    /// Apply activation function
    pub fn apply<F: Float>(&self, x: F) -> F {
        match self {
            ActivationFunction::Sigmoid => {
                let one = F::one();
                one / (one + (-x).exp())
            }
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::ReLU => x.max(F::zero()),
            ActivationFunction::GELU => {
                // Approximation of GELU
                let half = F::from(0.5).unwrap();
                let one = F::one();
                let sqrt_2_pi = F::from(0.7978845608).unwrap(); // sqrt(2/π)
                let coeff = F::from(0.044715).unwrap();

                half * x * (one + (sqrt_2_pi * (x + coeff * x * x * x)).tanh())
            }
            ActivationFunction::Swish => {
                let sigmoid = F::one() / (F::one() + (-x).exp());
                x * sigmoid
            }
            ActivationFunction::Linear => x,
        }
    }

    /// Apply derivative of activation function
    pub fn derivative<F: Float>(&self, x: F) -> F {
        match self {
            ActivationFunction::Sigmoid => {
                let sigmoid = self.apply(x);
                sigmoid * (F::one() - sigmoid)
            }
            ActivationFunction::Tanh => {
                let tanh_x = x.tanh();
                F::one() - tanh_x * tanh_x
            }
            ActivationFunction::ReLU => {
                if x > F::zero() {
                    F::one()
                } else {
                    F::zero()
                }
            }
            ActivationFunction::GELU => {
                // Simplified derivative approximation
                F::one() / (F::one() + (-x).exp())
            }
            ActivationFunction::Swish => {
                let sigmoid = F::one() / (F::one() + (-x).exp());
                sigmoid * (F::one() + x * (F::one() - sigmoid))
            }
            ActivationFunction::Linear => F::one(),
        }
    }
}
