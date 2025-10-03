//! DataLoader implementation for efficient batch loading

use crate::data::Dataset;
use crate::error::Result;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use num_integer::div_ceil;
use scirs2_core::numeric::{Float, FromPrimitive};
// use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::seq::SliceRandom;
// use scirs2_core::random::SeedableRng;
use std::fmt::Debug;
use std::marker::PhantomData;
/// Type alias for batch result
type BatchResult<F> = Result<(Array<F, IxDyn>, Array<F, IxDyn>)>;
/// Data loader for efficient batch processing
pub struct DataLoader<
    F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync,
    D: Dataset<F> + Send + Sync,
> {
    /// Dataset to load from
    pub dataset: D,
    /// Batch size
    pub batch_size: usize,
    /// Whether to shuffle the data
    pub shuffle: bool,
    /// Whether to drop the last batch if it's smaller than batch_size
    pub drop_last: bool,
    /// Current indices for iteration
    indices: Vec<usize>,
    /// Current position in indices
    position: usize,
    /// Phantom data for float type
    _phantom: PhantomData<F>,
}
impl<
        F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync,
        D: Dataset<F> + Send + Sync,
    > DataLoader<F, D>
{
    /// Create a new data loader
    ///
    /// # Arguments
    /// * `dataset` - Dataset to load from
    /// * `batch_size` - Number of samples per batch
    /// * `shuffle` - Whether to shuffle the data
    /// * `drop_last` - Whether to drop the last batch if it's smaller than batch_size
    pub fn new(_dataset: D, batch_size: usize, shuffle: bool, droplast: bool) -> Self {
        let indices: Vec<usize> = (0.._dataset.len()).collect();
        Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
            indices,
            position: 0, phantom: PhantomData,
        }
    }
    /// Reset the data loader state
    pub fn reset(&mut self) {
        if self.shuffle {
            let mut rng = rng();
            self.indices.shuffle(&mut rng);
        self.position = 0;
    /// Get the number of batches in the dataset
    pub fn num_batches(&self) -> usize {
        let num = div_ceil(self.dataset.len(), self.batch_size);
        if self.drop_last && num > 0 && self.dataset.len() % self.batch_size != 0 {
            num - 1
        } else {
            num
    /// Get the dataset len
    pub fn len(&self) -> usize {
    /// Check if the dataloader is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    /// Get the next batch from the dataset
    pub fn next_batch(&mut self) -> Option<BatchResult<F>> {
        if self.position >= self.dataset.len() {
            return None;
        let remaining = self.dataset.len() - self.position;
        let batch_size = if remaining < self.batch_size {
            if self.drop_last {
                return None;
            }
            remaining
            self.batch_size
        };
        // Collect batch indices
        let batch_indices: Vec<usize> =
            self.indices[self.position..self.position + batch_size].to_vec();
        self.position += batch_size;
        // Load data
        let result = self.load_batch(&batch_indices);
        Some(result)
    /// Load a batch of data using the given indices
    fn load_batch(&self, indices: &[usize]) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        // Load first sample to determine shapes
        let (first_x, first_y) = self.dataset.get(indices[0])?;
        // Create batch arrays
        let batch_xshape = [indices.len()]
            .iter()
            .chain(first_x.shape())
            .cloned()
            .collect::<Vec<_>>();
        let batch_yshape = [indices.len()]
            .chain(first_y.shape())
        let mut batch_x = Array::zeros(IxDyn(&batch_xshape));
        let mut batch_y = Array::zeros(IxDyn(&batch_yshape));
        // Fill batch arrays
        for (i, &idx) in indices.iter().enumerate() {
            let (x, y) = self.dataset.get(idx)?;
            // Copy data into batch arrays
            let mut batch_x_slice = batch_x.slice_mut(scirs2_core::ndarray::s![i, ..]);
            batch_x_slice.assign(&x);
            let mut batch_y_slice = batch_y.slice_mut(scirs2_core::ndarray::s![i, ..]);
            batch_y_slice.assign(&y);
        Ok((batch_x, batch_y))
    > Iterator for DataLoader<F, D>
    type Item = Result<(Array<F, IxDyn>, Array<F, IxDyn>)>;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
/// Helper function to create an iterator over the dataset in batches
#[allow(dead_code)]
pub fn iter_batches<
>(
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
) -> impl Iterator<Item = Result<(Array<F, IxDyn>, Array<F, IxDyn>)>> {
    let mut loader = DataLoader::new(dataset, batch_size, shuffle, drop_last);
    loader.reset();
    loader
