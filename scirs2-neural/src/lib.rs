#![allow(deprecated)]
//! Enhanced scirs2-neural implementation with core activation functions

pub mod activations_minimal;
pub mod autograd;
pub mod error;
// pub mod gpu; // Disabled in minimal version - has syntax errors
pub mod layers;
pub mod losses;
pub mod training;
pub mod utils;

pub use activations_minimal::{Activation, ReLU, Sigmoid, Softmax, Tanh, GELU};
pub use error::{Error, NeuralError, Result};
pub use layers::{BatchNorm, Conv2D, Dense, Dropout, Layer, LayerNorm, Sequential, LSTM};
pub use losses::{
    ContrastiveLoss, CrossEntropyLoss, FocalLoss, Loss, MeanSquaredError, TripletLoss,
};
pub use training::{TrainingConfig, TrainingSession};

/// Working prelude with core functionality
pub mod prelude {
    pub use crate::{
        activations_minimal::{Activation, ReLU, Sigmoid, Softmax, Tanh, GELU},
        error::{Error, NeuralError, Result},
        layers::{BatchNorm, Conv2D, Dense, Dropout, Layer, LayerNorm, Sequential, LSTM},
        losses::{
            ContrastiveLoss, CrossEntropyLoss, FocalLoss, Loss, MeanSquaredError, TripletLoss,
        },
        training::{TrainingConfig, TrainingSession},
    };
}
