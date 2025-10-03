//! Pipeline API for chaining transformations
//!
//! This module provides utilities for creating pipelines of transformations
//! that can be applied sequentially, similar to scikit-learn's Pipeline.

// mod adapters;

// pub use adapters::boxed;

use scirs2_core::ndarray::{Array2, ArrayBase, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use std::any::Any;

use crate::error::{Result, TransformError};

/// Trait for all transformers that can be used in pipelines
pub trait Transformer: Send + Sync {
    /// Fits the transformer to the input data
    fn fit(&mut self, x: &Array2<f64>) -> Result<()>;

    /// Transforms the input data
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>;

    /// Fits and transforms the data in one step
    fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns a boxed clone of the transformer
    fn clone_box(&self) -> Box<dyn Transformer>;

    /// Returns the transformer as Any for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Returns the transformer as mutable Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// A pipeline of transformations to be applied sequentially
pub struct Pipeline {
    /// List of named steps in the pipeline
    steps: Vec<(String, Box<dyn Transformer>)>,
    /// Whether the pipeline has been fitted
    fitted: bool,
}

impl Pipeline {
    /// Creates a new empty pipeline
    pub fn new() -> Self {
        Pipeline {
            steps: Vec::new(),
            fitted: false,
        }
    }

    /// Adds a step to the pipeline
    ///
    /// # Arguments
    /// * `name` - Name of the step
    /// * `transformer` - The transformer to add
    ///
    /// # Returns
    /// * `Self` - The pipeline for chaining
    pub fn add_step(mut self, name: impl Into<String>, transformer: Box<dyn Transformer>) -> Self {
        self.steps.push((name.into(), transformer));
        self
    }

    /// Fits all steps in the pipeline
    ///
    /// # Arguments
    /// * `x` - The input data
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let mut x_transformed = x.mapv(|x| NumCast::from(x).unwrap_or(0.0));

        for (name, transformer) in &mut self.steps {
            transformer.fit(&x_transformed).map_err(|e| {
                TransformError::TransformationError(format!("Failed to fit step '{name}': {e}"))
            })?;

            x_transformed = transformer.transform(&x_transformed).map_err(|e| {
                TransformError::TransformationError(format!(
                    "Failed to transform in step '{name}': {e}"
                ))
            })?;
        }

        self.fitted = true;
        Ok(())
    }

    /// Transforms data through all steps in the pipeline
    ///
    /// # Arguments
    /// * `x` - The input data
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if !self.fitted {
            return Err(TransformError::TransformationError(
                "Pipeline has not been fitted".to_string(),
            ));
        }

        let mut x_transformed = x.mapv(|x| NumCast::from(x).unwrap_or(0.0));

        for (name, transformer) in &self.steps {
            x_transformed = transformer.transform(&x_transformed).map_err(|e| {
                TransformError::TransformationError(format!(
                    "Failed to transform in step '{name}': {e}"
                ))
            })?;
        }

        Ok(x_transformed)
    }

    /// Fits and transforms data through all steps in the pipeline
    ///
    /// # Arguments
    /// * `x` - The input data
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the number of steps in the pipeline
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns whether the pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Gets a reference to a step by name
    pub fn get_step(&self, name: &str) -> Option<&dyn Transformer> {
        self.steps
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t.as_ref())
    }

    /// Gets a mutable reference to a step by name
    pub fn get_step_mut(&mut self, name: &str) -> Option<&mut Box<dyn Transformer>> {
        self.steps
            .iter_mut()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t)
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// ColumnTransformer applies different transformers to different columns
pub struct ColumnTransformer {
    /// List of transformers with their column indices
    transformers: Vec<(String, Box<dyn Transformer>, Vec<usize>)>,
    /// Whether to pass through columns not specified
    remainder: RemainderOption,
    /// Whether the transformer has been fitted
    fitted: bool,
}

/// Options for handling columns not specified in transformers
#[derive(Debug, Clone, Copy)]
pub enum RemainderOption {
    /// Drop unspecified columns
    Drop,
    /// Pass through unspecified columns unchanged
    Passthrough,
}

impl ColumnTransformer {
    /// Creates a new ColumnTransformer
    ///
    /// # Arguments
    /// * `remainder` - How to handle unspecified columns
    pub fn new(remainder: RemainderOption) -> Self {
        ColumnTransformer {
            transformers: Vec::new(),
            remainder,
            fitted: false,
        }
    }

    /// Adds a transformer for specific columns
    ///
    /// # Arguments
    /// * `name` - Name of the transformer
    /// * `transformer` - The transformer to apply
    /// * `columns` - Column indices to apply the transformer to
    ///
    /// # Returns
    /// * `Self` - The ColumnTransformer for chaining
    pub fn add_transformer(
        mut self,
        name: impl Into<String>,
        transformer: Box<dyn Transformer>,
        columns: Vec<usize>,
    ) -> Self {
        self.transformers.push((name.into(), transformer, columns));
        self
    }

    /// Fits all transformers to their respective columns
    ///
    /// # Arguments
    /// * `x` - The input data
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| NumCast::from(x).unwrap_or(0.0));
        let n_features = x_f64.shape()[1];

        // Validate column indices
        for (name_, transformer, columns) in &self.transformers {
            for &col in columns {
                if col >= n_features {
                    return Err(TransformError::InvalidInput(format!(
                        "Column index {col} in transformer '{name_}' exceeds number of features {n_features}"
                    )));
                }
            }
        }

        // Fit each transformer on its columns
        for (name, transformer, columns) in &mut self.transformers {
            // Extract relevant columns
            let subset = extract_columns(&x_f64, columns);

            transformer.fit(&subset).map_err(|e| {
                TransformError::TransformationError(format!(
                    "Failed to fit transformer '{name}': {e}"
                ))
            })?;
        }

        self.fitted = true;
        Ok(())
    }

    /// Transforms data using all configured transformers
    ///
    /// # Arguments
    /// * `x` - The input data
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if !self.fitted {
            return Err(TransformError::TransformationError(
                "ColumnTransformer has not been fitted".to_string(),
            ));
        }

        let x_f64 = x.mapv(|x| NumCast::from(x).unwrap_or(0.0));
        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        // Track which columns have been transformed
        let mut used_columns = vec![false; n_features];
        let mut transformed_parts = Vec::new();

        // Transform each group of columns
        for (name, transformer, columns) in &self.transformers {
            // Mark columns as used
            for &col in columns {
                used_columns[col] = true;
            }

            // Extract and transform columns
            let subset = extract_columns(&x_f64, columns);
            let transformed = transformer.transform(&subset).map_err(|e| {
                TransformError::TransformationError(format!(
                    "Failed to transform with '{name}': {e}"
                ))
            })?;

            transformed_parts.push(transformed);
        }

        // Handle remainder columns
        match self.remainder {
            RemainderOption::Passthrough => {
                // Collect unused columns
                let unused_columns: Vec<usize> =
                    (0..n_features).filter(|&i| !used_columns[i]).collect();

                if !unused_columns.is_empty() {
                    let remainder = extract_columns(&x_f64, &unused_columns);
                    transformed_parts.push(remainder);
                }
            }
            RemainderOption::Drop => {
                // Do nothing - unused columns are dropped
            }
        }

        // Concatenate all parts horizontally
        if transformed_parts.is_empty() {
            return Ok(Array2::zeros((n_samples, 0)));
        }

        concatenate_horizontal(&transformed_parts)
    }

    /// Fits and transforms data in one step
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }
}

/// Extracts specific columns from a 2D array
#[allow(dead_code)]
fn extract_columns(data: &Array2<f64>, columns: &[usize]) -> Array2<f64> {
    let n_samples = data.shape()[0];
    let n_cols = columns.len();

    let mut result = Array2::zeros((n_samples, n_cols));

    for (j, &col_idx) in columns.iter().enumerate() {
        for i in 0..n_samples {
            result[[i, j]] = data[[i, col_idx]];
        }
    }

    result
}

/// Concatenates arrays horizontally
#[allow(dead_code)]
fn concatenate_horizontal(arrays: &[Array2<f64>]) -> Result<Array2<f64>> {
    if arrays.is_empty() {
        return Err(TransformError::InvalidInput(
            "Cannot concatenate empty array list".to_string(),
        ));
    }

    let n_samples = arrays[0].shape()[0];
    let total_features: usize = arrays.iter().map(|a| a.shape()[1]).sum();

    // Verify all _arrays have the same number of samples
    for arr in arrays {
        if arr.shape()[0] != n_samples {
            return Err(TransformError::InvalidInput(
                "All _arrays must have the same number of samples".to_string(),
            ));
        }
    }

    let mut result = Array2::zeros((n_samples, total_features));
    let mut col_offset = 0;

    for arr in arrays {
        let n_cols = arr.shape()[1];
        for i in 0..n_samples {
            for j in 0..n_cols {
                result[[i, col_offset + j]] = arr[[i, j]];
            }
        }
        col_offset += n_cols;
    }

    Ok(result)
}

/// Make a pipeline from a list of (name, transformer) tuples
#[allow(dead_code)]
pub fn make_pipeline(steps: Vec<(&str, Box<dyn Transformer>)>) -> Pipeline {
    let mut pipeline = Pipeline::new();
    for (name, transformer) in steps {
        pipeline = pipeline.add_step(name, transformer);
    }
    pipeline
}

/// Make a column transformer from a list of (name, transformer, columns) tuples
#[allow(dead_code)]
pub fn make_column_transformer(
    transformers: Vec<(&str, Box<dyn Transformer>, Vec<usize>)>,
    remainder: RemainderOption,
) -> ColumnTransformer {
    let mut ct = ColumnTransformer::new(remainder);
    for (name, transformer, columns) in transformers {
        ct = ct.add_transformer(name, transformer, columns);
    }
    ct
}
