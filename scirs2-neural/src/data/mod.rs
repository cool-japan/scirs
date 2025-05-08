//! Data loading and processing utilities for neural networks
//!
//! This module provides utilities for loading, preprocessing, and batching
//! data for neural network training and evaluation.

use crate::error::{NeuralError, Result};
use ndarray::{Array, Axis, IxDyn, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::fmt::Debug;

mod augmentation;
mod dataloader;
mod dataset;
mod transforms;
mod utils;

pub use augmentation::*;
pub use dataloader::*;
pub use dataset::*;
pub use transforms::*;
pub use utils::*;

/// Dataset trait for accessing data
pub trait Dataset<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync>:
    Send + Sync
{
    /// Get the number of samples in the dataset
    fn len(&self) -> usize;

    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a single item from the dataset
    fn get(&self, index: usize) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)>;

    /// Clone the dataset (we need to implement it as a method since we can't derive Clone for trait objects)
    fn box_clone(&self) -> Box<dyn Dataset<F> + Send + Sync>;
}

/// Implementation of the Dataset trait for boxed datasets
impl<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync> Dataset<F>
    for Box<dyn Dataset<F> + Send + Sync>
{
    fn len(&self) -> usize {
        (**self).len()
    }

    fn get(&self, index: usize) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        (**self).get(index)
    }

    fn box_clone(&self) -> Box<dyn Dataset<F> + Send + Sync> {
        (**self).box_clone()
    }
}

/// In-memory dataset implementation
#[derive(Debug, Clone)]
pub struct InMemoryDataset<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync> {
    /// Features (inputs)
    pub features: Array<F, IxDyn>,
    /// Labels (targets)
    pub labels: Array<F, IxDyn>,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync> InMemoryDataset<F> {
    /// Create a new in-memory dataset
    pub fn new(features: Array<F, IxDyn>, labels: Array<F, IxDyn>) -> Result<Self> {
        if features.shape()[0] != labels.shape()[0] {
            return Err(NeuralError::InferenceError(format!(
                "Features and labels have different number of samples: {} vs {}",
                features.shape()[0],
                labels.shape()[0]
            )));
        }

        Ok(Self { features, labels })
    }

    /// Split the dataset into training and validation sets
    pub fn train_test_split(&self, test_size: f64) -> Result<(Self, Self)> {
        if test_size <= 0.0 || test_size >= 1.0 {
            return Err(NeuralError::InferenceError(
                "test_size must be between 0 and 1".to_string(),
            ));
        }

        let n_samples = self.len();
        let n_test = (n_samples as f64 * test_size).round() as usize;
        let n_train = n_samples - n_test;

        if n_train == 0 || n_test == 0 {
            return Err(NeuralError::InferenceError(
                "Split would result in empty training or test set".to_string(),
            ));
        }

        // Create shuffled indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = SmallRng::seed_from_u64(42);
        indices.shuffle(&mut rng);

        // Split indices
        let train_indices = &indices[0..n_train];
        let test_indices = &indices[n_train..];

        // Create training set
        let train_features = self.features.select(Axis(0), train_indices);
        let train_labels = self.labels.select(Axis(0), train_indices);

        // Create test set
        let test_features = self.features.select(Axis(0), test_indices);
        let test_labels = self.labels.select(Axis(0), test_indices);

        Ok((
            Self::new(train_features, train_labels)?,
            Self::new(test_features, test_labels)?,
        ))
    }
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync> Dataset<F>
    for InMemoryDataset<F>
{
    fn len(&self) -> usize {
        self.features.shape()[0]
    }

    fn get(&self, index: usize) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        if index >= self.len() {
            return Err(NeuralError::InferenceError(format!(
                "Index {} out of bounds for dataset with length {}",
                index,
                self.len()
            )));
        }

        // Get slices and convert to dynamic dimension arrays
        let x_slice = self.features.slice(ndarray::s![index, ..]);
        let y_slice = self.labels.slice(ndarray::s![index, ..]);

        let x_shape = x_slice.shape().to_vec();
        let y_shape = y_slice.shape().to_vec();

        let x = x_slice
            .to_owned()
            .into_shape_with_order(IxDyn(&x_shape))
            .unwrap();
        let y = y_slice
            .to_owned()
            .into_shape_with_order(IxDyn(&y_shape))
            .unwrap();

        Ok((x, y))
    }

    fn box_clone(&self) -> Box<dyn Dataset<F> + Send + Sync> {
        Box::new(InMemoryDataset {
            features: self.features.clone(),
            labels: self.labels.clone(),
        })
    }
}
