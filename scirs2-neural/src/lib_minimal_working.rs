//! Minimal working scirs2-neural implementation
//!
//! This is a simplified version that focuses on core functionality
//! that has been verified to compile and work.

pub mod error;
pub mod activations {
    pub mod gelu;
    pub mod softmax;
    pub mod swish;
    pub mod tanh;
    
    pub use gelu::GELU;
    pub use softmax::Softmax;
    pub use swish::Swish;
    pub use tanh::Tanh;
    
    use crate::error::Result;
    use scirs2_core::ndarray::Array;
    
    /// Trait for activation functions
    pub trait Activation<F> {
        /// Forward pass of the activation function
        fn forward(&self, input: &Array<F, scirs2_core::ndarray::IxDyn>) -> Result<Array<F, scirs2_core::ndarray::IxDyn>>;
        
        /// Backward pass of the activation function
        fn backward(
            &self,
            grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
            input: &Array<F, scirs2_core::ndarray::IxDyn>,
        ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>>;
    }
}

/// Simplified prelude with working components
pub mod prelude {
    pub use crate::activations::{Activation, GELU, Softmax, Swish, Tanh};
    pub use crate::error::{Error, NeuralError, Result};
}

// Re-export key types at crate level
pub use error::{Error, NeuralError, Result};
pub use activations::{Activation, GELU, Softmax, Swish, Tanh};
