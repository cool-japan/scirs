//! Dataset implementations for different data sources

use crate::data::{Dataset, Transform};
use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array, Array2, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign};
use std::fmt::Debug;
use std::io::BufRead;
use std::marker::PhantomData;
use std::path::Path;

/// CSV dataset implementation
#[derive(Debug)]
pub struct CSVDataset<F: Float + NumAssign + Debug + ScalarOperand + FromPrimitive + Send + Sync> {
    /// Features (inputs)
    features: Array<F, IxDyn>,
    /// Labels (targets)
    labels: Array<F, IxDyn>,
    /// Transform to apply to features
    feature_transform: Option<Box<dyn Transform<F> + Send + Sync>>,
    /// Transform to apply to labels
    label_transform: Option<Box<dyn Transform<F> + Send + Sync>>,
}

impl<F: Float + NumAssign + Debug + ScalarOperand + FromPrimitive + Send + Sync> Clone
    for CSVDataset<F>
{
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
                Some(t) => Some(t.box_clone()),
                None => None,
            },
        }
    }
}

impl<F: Float + NumAssign + Debug + ScalarOperand + FromPrimitive + Send + Sync> CSVDataset<F> {
    /// Create a new dataset from CSV file
    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        has_header: bool,
        feature_cols: &[usize],
        label_cols: &[usize],
        delimiter: char,
    ) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())
            .map_err(|e| NeuralError::IOError(format!("Failed to open CSV file: {e}")))?;
        let reader = std::io::BufReader::new(file);
        let mut lines = reader.lines();

        // Optionally skip header row
        if has_header {
            lines.next();
        }

        let mut feature_rows: Vec<Vec<f64>> = Vec::new();
        let mut label_rows: Vec<Vec<f64>> = Vec::new();

        let delimiter_str = delimiter.to_string();

        for (line_idx, line_result) in lines.enumerate() {
            let line = line_result.map_err(|e| {
                NeuralError::IOError(format!("Failed to read CSV line {line_idx}: {e}"))
            })?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let fields: Vec<&str> = line.split(delimiter_str.as_str()).collect();

            // Parse all required column indices first to validate bounds
            let max_col = feature_cols
                .iter()
                .chain(label_cols.iter())
                .copied()
                .max()
                .unwrap_or(0);
            if fields.len() <= max_col {
                return Err(NeuralError::InvalidArgument(format!(
                    "Row {line_idx} has {} fields but column index {max_col} was requested",
                    fields.len()
                )));
            }

            let mut feat_row = Vec::with_capacity(feature_cols.len());
            for &col in feature_cols {
                let val: f64 = fields[col].trim().parse::<f64>().map_err(|e| {
                    NeuralError::InvalidArgument(format!(
                        "Failed to parse float at row {line_idx}, col {col}: {e}"
                    ))
                })?;
                feat_row.push(val);
            }

            let mut label_row = Vec::with_capacity(label_cols.len());
            for &col in label_cols {
                let val: f64 = fields[col].trim().parse::<f64>().map_err(|e| {
                    NeuralError::InvalidArgument(format!(
                        "Failed to parse float at row {line_idx}, col {col}: {e}"
                    ))
                })?;
                label_row.push(val);
            }

            feature_rows.push(feat_row);
            label_rows.push(label_row);
        }

        if feature_rows.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "CSV file contains no data rows".to_string(),
            ));
        }

        let num_rows = feature_rows.len();
        let num_feat_cols = feature_cols.len();
        let num_label_cols = label_cols.len();

        // Flatten rows into column-major buffers for Array2 construction
        let feat_flat: Vec<f64> = feature_rows.into_iter().flatten().collect();
        let label_flat: Vec<f64> = label_rows.into_iter().flatten().collect();

        let features_f64 = Array2::<f64>::from_shape_vec((num_rows, num_feat_cols), feat_flat)
            .map_err(|e| NeuralError::ShapeMismatch(format!("Feature array shape error: {e}")))?;
        let labels_f64 = Array2::<f64>::from_shape_vec((num_rows, num_label_cols), label_flat)
            .map_err(|e| NeuralError::ShapeMismatch(format!("Label array shape error: {e}")))?;

        // Convert f64 arrays to the generic float type F
        let features: Array<F, IxDyn> = features_f64
            .mapv(|v| F::from_f64(v).unwrap_or(F::zero()))
            .into_dyn();
        let labels: Array<F, IxDyn> = labels_f64
            .mapv(|v| F::from_f64(v).unwrap_or(F::zero()))
            .into_dyn();

        Ok(Self {
            features,
            labels,
            feature_transform: None,
            label_transform: None,
        })
    }

    /// Set feature transform
    pub fn with_feature_transform<T: Transform<F> + 'static>(mut self, transform: T) -> Self {
        self.feature_transform = Some(Box::new(transform));
        self
    }

    /// Set label transform
    pub fn with_label_transform<T: Transform<F> + 'static>(mut self, transform: T) -> Self {
        self.label_transform = Some(Box::new(transform));
        self
    }
}

impl<F: Float + NumAssign + Debug + ScalarOperand + FromPrimitive + Send + Sync> Dataset<F>
    for CSVDataset<F>
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

        // Get slices of the data and convert to owned arrays
        let x_slice = self.features.slice(scirs2_core::ndarray::s![index, ..]);
        let y_slice = self.labels.slice(scirs2_core::ndarray::s![index, ..]);

        // Convert to dynamic dimension arrays
        let xshape = x_slice.shape().to_vec();
        let yshape = y_slice.shape().to_vec();

        let mut x = x_slice
            .to_owned()
            .into_shape_with_order(IxDyn(&xshape))
            .expect("Operation failed");
        let mut y = y_slice
            .to_owned()
            .into_shape_with_order(IxDyn(&yshape))
            .expect("Operation failed");

        // Apply transforms if available
        if let Some(ref transform) = self.feature_transform {
            x = transform.apply(&x)?;
        }

        if let Some(ref transform) = self.label_transform {
            y = transform.apply(&y)?;
        }

        Ok((x, y))
    }

    fn box_clone(&self) -> Box<dyn Dataset<F> + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Transformed dataset wrapper
#[derive(Debug)]
pub struct TransformedDataset<
    F: Float + NumAssign + Debug + ScalarOperand + FromPrimitive + Send + Sync,
    D: Dataset<F> + Clone,
> {
    /// Base dataset
    dataset: D,
    /// Transform to apply to features
    feature_transform: Option<Box<dyn Transform<F> + Send + Sync>>,
    /// Transform to apply to labels
    label_transform: Option<Box<dyn Transform<F> + Send + Sync>>,
    /// Phantom data for float type
    _phantom: PhantomData<F>,
}

impl<
        F: Float + NumAssign + Debug + ScalarOperand + FromPrimitive + Send + Sync,
        D: Dataset<F> + Clone,
    > Clone for TransformedDataset<F, D>
{
    fn clone(&self) -> Self {
        // Manual clone implementation that uses box_clone for dyn Transform<F> + Send + Sync
        Self {
            dataset: self.dataset.clone(),
            feature_transform: match &self.feature_transform {
                Some(t) => Some(t.box_clone()),
                None => None,
            },
            label_transform: match &self.label_transform {
                Some(t) => Some(t.box_clone()),
                None => None,
            },
            _phantom: PhantomData,
        }
    }
}

impl<
        F: Float + NumAssign + Debug + ScalarOperand + FromPrimitive + Send + Sync,
        D: Dataset<F> + Clone,
    > TransformedDataset<F, D>
{
    /// Create a new transformed dataset
    pub fn new(dataset: D) -> Self {
        Self {
            dataset,
            feature_transform: None,
            label_transform: None,
            _phantom: PhantomData,
        }
    }

    /// Set feature transform
    pub fn with_feature_transform<T: Transform<F> + 'static>(mut self, transform: T) -> Self {
        self.feature_transform = Some(Box::new(transform));
        self
    }

    /// Set label transform
    pub fn with_label_transform<T: Transform<F> + 'static>(mut self, transform: T) -> Self {
        self.label_transform = Some(Box::new(transform));
        self
    }
}

impl<
        F: Float + NumAssign + Debug + ScalarOperand + FromPrimitive + Send + Sync,
        D: Dataset<F> + Clone + 'static,
    > Dataset<F> for TransformedDataset<F, D>
{
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        // Get the data from the underlying dataset
        let (mut x, mut y) = self.dataset.get(index)?;

        // Apply transforms if available
        if let Some(ref transform) = self.feature_transform {
            x = transform.apply(&x)?;
        }

        if let Some(ref transform) = self.label_transform {
            y = transform.apply(&y)?;
        }

        Ok((x, y))
    }

    fn box_clone(&self) -> Box<dyn Dataset<F> + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Subset dataset wrapper
#[derive(Debug, Clone)]
pub struct SubsetDataset<
    F: Float + NumAssign + Debug + ScalarOperand + FromPrimitive + Send + Sync,
    D: Dataset<F> + Clone,
> {
    /// Base dataset
    dataset: D,
    /// Indices to include in the subset
    indices: Vec<usize>,
    /// Phantom data for float type
    _phantom: PhantomData<F>,
}

impl<
        F: Float + NumAssign + Debug + ScalarOperand + FromPrimitive + Send + Sync,
        D: Dataset<F> + Clone,
    > SubsetDataset<F, D>
{
    /// Create a new subset dataset
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
        }

        Ok(Self {
            dataset,
            indices,
            _phantom: PhantomData,
        })
    }
}

impl<
        F: Float + NumAssign + Debug + ScalarOperand + FromPrimitive + Send + Sync,
        D: Dataset<F> + Clone + 'static,
    > Dataset<F> for SubsetDataset<F, D>
{
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        if index >= self.len() {
            return Err(NeuralError::InferenceError(format!(
                "Index {} out of bounds for subset dataset with length {}",
                index,
                self.len()
            )));
        }

        let dataset_index = self.indices[index];
        self.dataset.get(dataset_index)
    }

    fn box_clone(&self) -> Box<dyn Dataset<F> + Send + Sync> {
        Box::new(self.clone())
    }
}
