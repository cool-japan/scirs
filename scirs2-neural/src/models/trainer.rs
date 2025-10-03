//! Model trainer implementation
//!
//! This module provides utilities for training neural network models.

use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Training history that stores metrics during training
#[derive(Debug, Clone)]
pub struct History<F: Float> {
    /// Training loss per epoch
    pub train_loss: Vec<F>,
    /// Validation loss per epoch  
    pub val_loss: Vec<F>,
    /// Additional metrics
    pub metrics: HashMap<String, Vec<F>>,
}

impl<F: Float> Default for History<F> {
    fn default() -> Self {
        Self {
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            metrics: HashMap::new(),
        }
    }
}

/// Configuration for model training
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of epochs to train
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Whether to shuffle the data
    pub shuffle: bool,
    /// Validation split ratio (0.0 to 1.0)
    pub validation_split: f64,
    /// Verbose level (0: silent, 1: progress bar, 2: one line per epoch)
    pub verbose: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 32,
            learning_rate: 0.01,
            shuffle: true,
            validation_split: 0.2,
            verbose: 1,
        }
    }
}

/// Neural network model trainer
pub struct Trainer<F: Float + Debug + ScalarOperand> {
    /// Training configuration
    pub config: TrainingConfig,
    /// Training history
    pub history: History<F>,
}

impl<F: Float + Debug + ScalarOperand> Trainer<F> {
    /// Create a new trainer
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            history: History::default(),
        }
    }
}
