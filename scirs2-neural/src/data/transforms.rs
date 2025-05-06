//! Data transforms for preprocessing inputs

use crate::error::Result;
use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

/// Trait for data transforms
pub trait Transform<F: Float + Debug + ScalarOperand> {
    /// Apply the transform to the input
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>>;

    /// Get a description of the transform
    fn description(&self) -> String;
}

/// Standard scaler transform
#[derive(Debug, Clone)]
pub struct StandardScaler<F: Float + Debug + ScalarOperand> {
    /// Mean values for each feature
    mean: Option<Array<F, IxDyn>>,
    /// Standard deviation values for each feature
    std: Option<Array<F, IxDyn>>,
    /// Whether to fit on the first dimension (samples)
    fit_per_sample: bool,
}

impl<F: Float + Debug + ScalarOperand> StandardScaler<F> {
    /// Create a new standard scaler
    pub fn new(fit_per_sample: bool) -> Self {
        Self {
            mean: None,
            std: None,
            fit_per_sample,
        }
    }

    /// Fit the scaler to the data
    pub fn fit(&mut self, data: &Array<F, IxDyn>) -> Result<&mut Self> {
        if data.ndim() < 2 {
            // Just compute global mean and std
            let mean = data.mean().unwrap_or(F::zero());
            let std = data.std(0.0);

            self.mean = Some(Array::from_elem(IxDyn(&[1]), mean));
            self.std = Some(Array::from_elem(IxDyn(&[1]), std));
        } else if self.fit_per_sample {
            // Compute mean and std for each sample
            let axis = 1; // Fit on feature dimension
            let mean = data
                .mean_axis(ndarray::Axis(axis))
                .unwrap_or(Array::zeros(IxDyn(&[data.shape()[0]])));
            let std = data.std_axis(ndarray::Axis(axis), 0.0);

            self.mean = Some(mean);
            self.std = Some(std);
        } else {
            // Compute mean and std for each feature
            let axis = 0; // Fit on sample dimension
            let mean = data
                .mean_axis(ndarray::Axis(axis))
                .unwrap_or(Array::zeros(IxDyn(&[data.shape()[1]])));
            let std = data.std_axis(ndarray::Axis(axis), 0.0);

            self.mean = Some(mean);
            self.std = Some(std);
        }

        Ok(self)
    }

    /// Transform data using the fitted parameters
    pub fn transform(&self, data: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        if self.mean.is_none() || self.std.is_none() {
            return Err(crate::error::NeuralError::InferenceError(
                "StandardScaler has not been fitted".to_string(),
            ));
        }

        let mean = self.mean.as_ref().unwrap();
        let std = self.std.as_ref().unwrap();

        let mut result = data.clone();

        if data.ndim() < 2 {
            // Apply global mean and std
            let mean_val = mean[[0]];
            let std_val = std[[0]].max(F::epsilon());

            for item in result.iter_mut() {
                *item = (*item - mean_val) / std_val;
            }
        } else if self.fit_per_sample {
            // Apply per-sample normalization
            for i in 0..data.shape()[0] {
                let mean_val = mean[[i]];
                let std_val = std[[i]].max(F::epsilon());

                for j in 0..data.shape()[1] {
                    result[[i, j]] = (data[[i, j]] - mean_val) / std_val;
                }
            }
        } else {
            // Apply per-feature normalization
            for j in 0..data.shape()[1] {
                let mean_val = mean[[j]];
                let std_val = std[[j]].max(F::epsilon());

                for i in 0..data.shape()[0] {
                    result[[i, j]] = (data[[i, j]] - mean_val) / std_val;
                }
            }
        }

        Ok(result)
    }

    /// Fit to data and then transform it
    pub fn fit_transform(&mut self, data: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.fit(data)?;
        self.transform(data)
    }
}

impl<F: Float + Debug + ScalarOperand> Transform<F> for StandardScaler<F> {
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.transform(input)
    }

    fn description(&self) -> String {
        if self.fit_per_sample {
            "StandardScaler (per-sample)".to_string()
        } else {
            "StandardScaler (per-feature)".to_string()
        }
    }
}

/// MinMax scaler transform
#[derive(Debug, Clone)]
pub struct MinMaxScaler<F: Float + Debug + ScalarOperand> {
    /// Minimum values for each feature
    min: Option<Array<F, IxDyn>>,
    /// Maximum values for each feature
    max: Option<Array<F, IxDyn>>,
    /// Target range for scaling (default: [0, 1])
    range: (F, F),
    /// Whether to fit on the first dimension (samples)
    fit_per_sample: bool,
}

impl<F: Float + Debug + ScalarOperand> MinMaxScaler<F> {
    /// Create a new MinMax scaler with default range [0, 1]
    pub fn new(fit_per_sample: bool) -> Self {
        Self::with_range(F::zero(), F::one(), fit_per_sample)
    }

    /// Create a new MinMax scaler with custom range
    pub fn with_range(min_val: F, max_val: F, fit_per_sample: bool) -> Self {
        Self {
            min: None,
            max: None,
            range: (min_val, max_val),
            fit_per_sample,
        }
    }

    /// Fit the scaler to the data
    pub fn fit(&mut self, data: &Array<F, IxDyn>) -> Result<&mut Self> {
        if data.ndim() < 2 {
            // Just compute global min and max
            let min = match data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                Some(&val) => val,
                None => F::zero(),
            };

            let max = match data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                Some(&val) => val,
                None => F::one(),
            };

            self.min = Some(Array::from_elem(IxDyn(&[1]), min));
            self.max = Some(Array::from_elem(IxDyn(&[1]), max));
        } else if self.fit_per_sample {
            // Compute min and max for each sample
            let mut min_vals = Array::zeros(IxDyn(&[data.shape()[0]]));
            let mut max_vals = Array::zeros(IxDyn(&[data.shape()[0]]));

            for i in 0..data.shape()[0] {
                let row = data.slice(ndarray::s![i, ..]);

                let min = match row.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                    Some(&val) => val,
                    None => F::zero(),
                };

                let max = match row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                    Some(&val) => val,
                    None => F::one(),
                };

                min_vals[[i]] = min;
                max_vals[[i]] = max;
            }

            self.min = Some(min_vals);
            self.max = Some(max_vals);
        } else {
            // Compute min and max for each feature
            let mut min_vals = Array::zeros(IxDyn(&[data.shape()[1]]));
            let mut max_vals = Array::zeros(IxDyn(&[data.shape()[1]]));

            for j in 0..data.shape()[1] {
                let col = data.slice(ndarray::s![.., j]);

                let min = match col.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                    Some(&val) => val,
                    None => F::zero(),
                };

                let max = match col.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                    Some(&val) => val,
                    None => F::one(),
                };

                min_vals[[j]] = min;
                max_vals[[j]] = max;
            }

            self.min = Some(min_vals);
            self.max = Some(max_vals);
        }

        Ok(self)
    }

    /// Transform data using the fitted parameters
    pub fn transform(&self, data: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        if self.min.is_none() || self.max.is_none() {
            return Err(crate::error::NeuralError::InferenceError(
                "MinMaxScaler has not been fitted".to_string(),
            ));
        }

        let min = self.min.as_ref().unwrap();
        let max = self.max.as_ref().unwrap();
        let (range_min, range_max) = self.range;
        let range_diff = range_max - range_min;

        let mut result = data.clone();

        if data.ndim() < 2 {
            // Apply global min and max
            let min_val = min[[0]];
            let max_val = max[[0]];
            let scale = if max_val > min_val {
                F::one() / (max_val - min_val)
            } else {
                F::one()
            };

            for item in result.iter_mut() {
                *item = range_min + range_diff * ((*item - min_val) * scale);
            }
        } else if self.fit_per_sample {
            // Apply per-sample normalization
            for i in 0..data.shape()[0] {
                let min_val = min[[i]];
                let max_val = max[[i]];
                let scale = if max_val > min_val {
                    F::one() / (max_val - min_val)
                } else {
                    F::one()
                };

                for j in 0..data.shape()[1] {
                    result[[i, j]] = range_min + range_diff * ((data[[i, j]] - min_val) * scale);
                }
            }
        } else {
            // Apply per-feature normalization
            for j in 0..data.shape()[1] {
                let min_val = min[[j]];
                let max_val = max[[j]];
                let scale = if max_val > min_val {
                    F::one() / (max_val - min_val)
                } else {
                    F::one()
                };

                for i in 0..data.shape()[0] {
                    result[[i, j]] = range_min + range_diff * ((data[[i, j]] - min_val) * scale);
                }
            }
        }

        Ok(result)
    }

    /// Fit to data and then transform it
    pub fn fit_transform(&mut self, data: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.fit(data)?;
        self.transform(data)
    }
}

impl<F: Float + Debug + ScalarOperand> Transform<F> for MinMaxScaler<F> {
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.transform(input)
    }

    fn description(&self) -> String {
        format!(
            "MinMaxScaler (range: [{:.1}, {:.1}], {})",
            self.range.0.to_f64().unwrap_or(0.0),
            self.range.1.to_f64().unwrap_or(1.0),
            if self.fit_per_sample {
                "per-sample"
            } else {
                "per-feature"
            }
        )
    }
}

/// One-hot encoder transform
#[derive(Debug, Clone)]
pub struct OneHotEncoder<F: Float + Debug + ScalarOperand> {
    /// Number of classes
    n_classes: usize,
    /// Phantom data for generic type
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + Debug + ScalarOperand> OneHotEncoder<F> {
    /// Create a new one-hot encoder
    pub fn new(n_classes: usize) -> Self {
        Self {
            n_classes,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Transform class indices to one-hot encoded vectors
    pub fn transform(&self, data: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = data.shape();
        let n_samples = shape[0];

        // Create output array with shape [n_samples, n_classes]
        let mut result = Array::zeros(IxDyn(&[n_samples, self.n_classes]));

        // Fill one-hot encoded values
        for i in 0..n_samples {
            let class_idx = data[[i]].to_usize().unwrap_or(0);

            if class_idx >= self.n_classes {
                return Err(crate::error::NeuralError::InferenceError(format!(
                    "Class index {} is out of bounds for {} classes",
                    class_idx, self.n_classes
                )));
            }

            result[[i, class_idx]] = F::one();
        }

        Ok(result)
    }
}

impl<F: Float + Debug + ScalarOperand> Transform<F> for OneHotEncoder<F> {
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.transform(input)
    }

    fn description(&self) -> String {
        format!("OneHotEncoder (n_classes: {})", self.n_classes)
    }
}

/// Compose multiple transforms into a single transform
#[derive(Debug, Clone)]
pub struct ComposeTransform<F: Float + Debug + ScalarOperand> {
    /// List of transforms to apply in sequence
    transforms: Vec<Box<dyn Transform<F>>>,
}

impl<F: Float + Debug + ScalarOperand> ComposeTransform<F> {
    /// Create a new composition of transforms
    pub fn new(transforms: Vec<Box<dyn Transform<F>>>) -> Self {
        Self { transforms }
    }
}

impl<F: Float + Debug + ScalarOperand> Transform<F> for ComposeTransform<F> {
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut data = input.clone();

        for transform in &self.transforms {
            data = transform.apply(&data)?;
        }

        Ok(data)
    }

    fn description(&self) -> String {
        let descriptions: Vec<String> = self.transforms.iter().map(|t| t.description()).collect();

        format!("Compose({})", descriptions.join(", "))
    }
}
