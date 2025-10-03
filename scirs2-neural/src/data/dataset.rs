//! Dataset implementations for different data sources

use crate::data::{Dataset, Transform};
use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::path::Path;
/// CSV dataset implementation
#[derive(Debug)]
pub struct CSVDataset<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync> {
    /// Features (inputs)
    features: Array<F, IxDyn>,
    /// Labels (targets)
    labels: Array<F, IxDyn>,
    /// Transform to apply to features
    feature_transform: Option<Box<dyn Transform<F> + Send + Sync>>,
    /// Transform to apply to labels
    label_transform: Option<Box<dyn Transform<F> + Send + Sync>>,
}
impl<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync> Clone for CSVDataset<F> {
    fn clone(&self) -> Self {
        // Manual clone implementation that uses box_clone for dyn Transform<F> + Send + Sync
        Self {
            features: self.features.clone(),
            labels: self.labels.clone(),
            feature_transform: match &self.feature_transform {
                Some(t) => Some(t.box_clone()),
                None => None,
            },
            label_transform: match &self.label_transform {
        }
    }
impl<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync> CSVDataset<F> {
    /// Create a new dataset from CSV file
    pub fn from_csv<P: AsRef<Path>>(
        _path: P_has, header: bool, _feature_cols: &[usize], _label_cols: &[usize], _delimiter: char,
    ) -> Result<Self> {
        // In a real implementation, we'd use a CSV reader here
        // For now, just return an error
        Err(NeuralError::InferenceError(
            "CSV loading not yet implemented".to_string(),
        ))
    /// Set feature transform
    pub fn with_feature_transform<T: Transform<F> + 'static>(mut self, transform: T) -> Self {
        self.feature_transform = Some(Box::new(transform));
        self
    /// Set label transform
    pub fn with_label_transform<T: Transform<F> + 'static>(mut self, transform: T) -> Self {
        self.label_transform = Some(Box::new(transform));
impl<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync> Dataset<F> for CSVDataset<F> {
    fn len(&self) -> usize {
        self.features.shape()[0]
    fn get(&self, index: usize) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        if index >= self.len() {
            return Err(NeuralError::InferenceError(format!(
                "Index {} out of bounds for dataset with length {}",
                index,
                self.len()
            )));
        // Get slices of the data and convert to owned arrays
        let x_slice = self.features.slice(scirs2_core::ndarray::s![index, ..]);
        let y_slice = self.labels.slice(scirs2_core::ndarray::s![index, ..]);
        // Convert to dynamic dimension arrays
        let xshape = x_slice.shape().to_vec();
        let yshape = y_slice.shape().to_vec();
        let mut x = x_slice
            .to_owned()
            .into_shape_with_order(IxDyn(&xshape))
            .unwrap();
        let mut y = y_slice
            .into_shape_with_order(IxDyn(&yshape))
        // Apply transforms if available
        if let Some(ref transform) = self.feature_transform {
            x = transform.apply(&x)?;
        if let Some(ref transform) = self.label_transform {
            y = transform.apply(&y)?;
        Ok((x, y))
    fn box_clone(&self) -> Box<dyn Dataset<F> + Send + Sync> {
        Box::new(self.clone())
/// Transformed dataset wrapper
pub struct TransformedDataset<
    F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync,
    D: Dataset<F> + Clone,
> {
    /// Base dataset
    dataset: D,
impl<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync, D: Dataset<F> + Clone> Clone
    for TransformedDataset<F, D>
{
            dataset: self.dataset.clone(),
impl<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync, D: Dataset<F> + Clone>
    TransformedDataset<F, D>
    /// Create a new transformed dataset
    pub fn new(dataset: D) -> Self {
            dataset,
            feature_transform: None,
            label_transform: None,
impl<
        F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync,
        D: Dataset<F> + Clone + 'static,
    > Dataset<F> for TransformedDataset<F, D>
        self._dataset.len()
        // Get the data from the underlying _dataset
        let (mut x, mut y) = self._dataset.get(index)?;
/// Subset _dataset wrapper
#[derive(Debug, Clone)]
pub struct SubsetDataset<
    /// Indices to include in the subset
    indices: Vec<usize>,
    /// Phantom data for float type
    _phantom: PhantomData<F>,
    SubsetDataset<F, D>
    /// Create a new subset _dataset
    pub fn new(dataset: D, indices: Vec<usize>) -> Result<Self> {
        // Validate indices
        for &idx in &indices {
            if idx >= dataset.len() {
                return Err(NeuralError::InferenceError(format!(
                    "Index {} out of bounds for dataset with length {}",
                    idx,
                    dataset.len()
                )));
            }
        Ok(Self {
            indices_phantom: PhantomData,
        })
    > Dataset<F> for SubsetDataset<F, D>
        self.indices.len()
                "Index {} out of bounds for subset dataset with length {}",
        let dataset_index = self.indices[index];
        self.dataset.get(dataset_index)
