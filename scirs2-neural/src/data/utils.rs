//! Utility functions for data processing

use crate::error::Result;
use scirs2_core::ndarray::{Array, Axis, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::SeedableRng;
use std::fmt::Debug;
/// Type alias for train-val split return type
type SplitResult<F> = Result<(
    Array<F, IxDyn>,
)>;
/// Split data into training and validation sets
#[allow(dead_code)]
pub fn train_val_split<F: Float + Debug + ScalarOperand>(
    x: &Array<F, IxDyn>,
    y: &Array<F, IxDyn>,
    val_size: f64,
    shuffle: bool,
) -> SplitResult<F> {
    if val_size <= 0.0 || val_size >= 1.0 {
        return Err(crate::error::NeuralError::InferenceError(
            "Validation size must be between 0 and 1".to_string(),
        ));
    }
    let n_samples = x.shape()[0];
    if n_samples != y.shape()[0] {
        return Err(crate::error::NeuralError::InferenceError(format!(
            "X and y have different number of samples: {} vs {}",
            n_samples,
            y.shape()[0]
        )));
    let n_val = (n_samples as f64 * val_size).round() as usize;
    let n_train = n_samples - n_val;
    if n_train == 0 || n_val == 0 {
            "Split would result in empty training or validation set".to_string(),
    if shuffle {
        // Create shuffled indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = rng();
        indices.shuffle(&mut rng);
        // Split indices
        let train_indices = &indices[0..n_train];
        let val_indices = &indices[n_train..];
        // Split data
        let x_train_2d = x.select(Axis(0), train_indices);
        let y_train_2d = y.select(Axis(0), train_indices);
        let x_val_2d = x.select(Axis(0), val_indices);
        let y_val_2d = y.select(Axis(0), val_indices);
        // Convert to IxDyn
        let x_trainshape = x_train_2d.shape().to_vec();
        let y_trainshape = y_train_2d.shape().to_vec();
        let x_valshape = x_val_2d.shape().to_vec();
        let y_valshape = y_val_2d.shape().to_vec();
        let x_train = x_train_2d
            .into_shape_with_order(IxDyn(&x_trainshape))
            .unwrap();
        let y_train = y_train_2d
            .into_shape_with_order(IxDyn(&y_trainshape))
        let x_val = x_val_2d.into_shape_with_order(IxDyn(&x_valshape)).unwrap();
        let y_val = y_val_2d.into_shape_with_order(IxDyn(&y_valshape)).unwrap();
        Ok((x_train, y_train, x_val, y_val))
    } else {
        // Split data without shuffling
        let x_train_2d = x.slice(scirs2_core::ndarray::s![0..n_train, ..]).to_owned();
        let y_train_2d = y.slice(scirs2_core::ndarray::s![0..n_train, ..]).to_owned();
        let x_val_2d = x.slice(scirs2_core::ndarray::s![n_train.., ..]).to_owned();
        let y_val_2d = y.slice(scirs2_core::ndarray::s![n_train.., ..]).to_owned();
}
/// K-fold cross-validation indices generator
pub struct KFold {
    /// Number of folds
    n_splits: usize,
    /// Whether to shuffle the data
    /// Random seed for reproducibility
    random_state: Option<u64>,
impl KFold {
    /// Create a new K-fold cross-validation generator
    pub fn new(_n_splits: usize, shuffle: bool, randomstate: Option<u64>) -> Self {
        Self {
            n_splits,
            shuffle,
            random_state,
        }
    /// Generate train/test indices for each fold
    pub fn split(&self, nsamples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if self.n_splits < 2 {
            return Err(crate::error::NeuralError::InvalidArgument(
                "n_splits must be >= 2".to_string(),
            ));
        if self.n_splits > n_samples {
                "n_splits must be <= n_samples".to_string(),
        // Create indices
        // Shuffle if needed
        if self.shuffle {
            // Use separate branches for different RNG types
            if let Some(seed) = self.random_state {
                let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(seed);
                indices.shuffle(&mut rng);
            } else {
                let mut rng = rng();
            }
        // Generate folds
        let fold_sizes = vec![n_samples / self.n_splits; self.n_splits];
        let fold_sizes_updated = fold_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                if i < n_samples % self.n_splits {
                    size + 1
                } else {
                    size
                }
            })
            .collect::<Vec<_>>();
        // Generate train/test indices for each fold
        let mut result = Vec::with_capacity(self.n_splits);
        let mut current = 0;
        for &fold_size in fold_sizes_updated.iter() {
            let start = current;
            let end = current + fold_size;
            let test_indices = indices[start..end].to_vec();
            let mut train_indices = Vec::with_capacity(n_samples - test_indices.len());
            train_indices.extend_from_slice(&indices[0..start]);
            train_indices.extend_from_slice(&indices[end..]);
            result.push((train_indices, test_indices));
            current = end;
        Ok(result)
/// Create batches from data
#[allow(dead_code)]
pub fn create_batches<F: Float + Debug + ScalarOperand>(
    batch_size: usize,
) -> Vec<(Array<F, IxDyn>, Array<F, IxDyn>)> {
    let n_batches = n_samples.div_ceil(batch_size); // Ceiling division
    // Create indices
    let mut indices: Vec<usize> = (0..n_samples).collect();
    // Shuffle if needed
    // Create batches
    let mut batches = Vec::with_capacity(n_batches);
    for i in 0..n_batches {
        let start = i * batch_size;
        let end = std::cmp::min(start + batch_size, n_samples);
        let batch_indices = &indices[start..end];
        let x_batch = x.select(Axis(0), batch_indices);
        let y_batch = y.select(Axis(0), batch_indices);
        batches.push((x_batch, y_batch));
    batches
