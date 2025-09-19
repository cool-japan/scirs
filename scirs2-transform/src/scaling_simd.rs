//! SIMD-accelerated scaling operations
//!
//! This module provides SIMD-optimized implementations of scaling operations
//! using the unified SIMD operations from scirs2-core.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use num_traits::{Float, NumCast};
use scirs2_core::simd_ops::SimdUnifiedOps;

use crate::error::{Result, TransformError};
use crate::scaling::EPSILON;

/// SIMD-accelerated MaxAbsScaler
pub struct SimdMaxAbsScaler<F: Float + NumCast + SimdUnifiedOps> {
    /// Maximum absolute values for each feature
    max_abs_: Option<Array1<F>>,
    /// Scale factors for each feature
    scale_: Option<Array1<F>>,
}

impl<F: Float + NumCast + SimdUnifiedOps> SimdMaxAbsScaler<F> {
    /// Creates a new SIMD-accelerated MaxAbsScaler
    pub fn new() -> Self {
        SimdMaxAbsScaler {
            max_abs_: None,
            scale_: None,
        }
    }

    /// Fits the scaler to the input data using SIMD operations
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        let mut max_abs = Array1::zeros(n_features);

        // Compute maximum absolute value for each feature using SIMD
        for j in 0..n_features {
            let col = x.column(j);
            let col_array = col.to_owned();
            let abs_col = F::simd_abs(&col_array.view());
            max_abs[j] = F::simd_max_element(&abs_col.view());
        }

        // Compute scale factors
        let scale = max_abs.mapv(|max_abs_val| {
            if max_abs_val > F::from(EPSILON).unwrap() {
                F::one() / max_abs_val
            } else {
                F::one()
            }
        });

        self.max_abs_ = Some(max_abs);
        self.scale_ = Some(scale);

        Ok(())
    }

    /// Transforms the input data using SIMD operations
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<F>>
    where
        S: Data<Elem = F>,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if self.scale_.is_none() {
            return Err(TransformError::TransformationError(
                "Scaler has not been fitted".to_string(),
            ));
        }

        let scale = self.scale_.as_ref().unwrap();

        if n_features != scale.len() {
            return Err(TransformError::InvalidInput(format!(
                "X has {} features, but scaler was fitted with {} features",
                n_features,
                scale.len()
            )));
        }

        let mut result = Array2::zeros((n_samples, n_features));

        // Transform each row using SIMD operations
        for i in 0..n_samples {
            let row = x.row(i);
            let row_array = row.to_owned();
            let scaled_row = F::simd_mul(&row_array.view(), &scale.view());

            for j in 0..n_features {
                result[[i, j]] = scaled_row[j];
            }
        }

        Ok(result)
    }

    /// Fits and transforms the data in one step
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<F>>
    where
        S: Data<Elem = F>,
    {
        self.fit(x)?;
        self.transform(x)
    }
}

/// SIMD-accelerated robust scaling using median and IQR
pub struct SimdRobustScaler<F: Float + NumCast + SimdUnifiedOps> {
    /// Median values for each feature
    median_: Option<Array1<F>>,
    /// IQR values for each feature
    iqr_: Option<Array1<F>>,
    /// Scale factors (1/IQR) for each feature
    scale_: Option<Array1<F>>,
}

impl<F: Float + NumCast + SimdUnifiedOps> SimdRobustScaler<F> {
    /// Creates a new SIMD-accelerated RobustScaler
    pub fn new() -> Self {
        SimdRobustScaler {
            median_: None,
            iqr_: None,
            scale_: None,
        }
    }

    /// Fits the scaler to the input data
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        let mut median = Array1::zeros(n_features);
        let mut iqr = Array1::zeros(n_features);

        // Compute median and IQR for each feature
        for j in 0..n_features {
            let col = x.column(j);
            let mut col_data: Vec<F> = col.to_vec();
            col_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let n = col_data.len();

            // Calculate median
            median[j] = if n % 2 == 0 {
                (col_data[n / 2 - 1] + col_data[n / 2]) / F::from(2.0).unwrap()
            } else {
                col_data[n / 2]
            };

            // Calculate IQR
            let q1_idx = n / 4;
            let q3_idx = 3 * n / 4;
            let q1 = col_data[q1_idx];
            let q3 = col_data[q3_idx];
            iqr[j] = q3 - q1;
        }

        // Compute scale factors
        let scale = iqr.mapv(|iqr_val| {
            if iqr_val > F::from(EPSILON).unwrap() {
                F::one() / iqr_val
            } else {
                F::one()
            }
        });

        self.median_ = Some(median);
        self.iqr_ = Some(iqr);
        self.scale_ = Some(scale);

        Ok(())
    }

    /// Transforms the input data using SIMD operations
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<F>>
    where
        S: Data<Elem = F>,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if self.median_.is_none() || self.scale_.is_none() {
            return Err(TransformError::TransformationError(
                "Scaler has not been fitted".to_string(),
            ));
        }

        let median = self.median_.as_ref().unwrap();
        let scale = self.scale_.as_ref().unwrap();

        if n_features != median.len() {
            return Err(TransformError::InvalidInput(format!(
                "X has {} features, but scaler was fitted with {} features",
                n_features,
                median.len()
            )));
        }

        let mut result = Array2::zeros((n_samples, n_features));

        // Transform each row: (x - median) * scale
        for i in 0..n_samples {
            let row = x.row(i);
            let row_array = row.to_owned();

            // Subtract median
            let centered = F::simd_sub(&row_array.view(), &median.view());

            // Scale by IQR
            let scaled = F::simd_mul(&centered.view(), &scale.view());

            for j in 0..n_features {
                result[[i, j]] = scaled[j];
            }
        }

        Ok(result)
    }

    /// Fits and transforms the data in one step
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<F>>
    where
        S: Data<Elem = F>,
    {
        self.fit(x)?;
        self.transform(x)
    }
}

/// SIMD-accelerated standard scaling (Z-score normalization)
pub struct SimdStandardScaler<F: Float + NumCast + SimdUnifiedOps> {
    /// Mean values for each feature
    mean_: Option<Array1<F>>,
    /// Standard deviation values for each feature
    std_: Option<Array1<F>>,
    /// Whether to center the data
    with_mean: bool,
    /// Whether to scale to unit variance
    with_std: bool,
}

impl<F: Float + NumCast + SimdUnifiedOps> SimdStandardScaler<F> {
    /// Creates a new SIMD-accelerated StandardScaler
    pub fn new(with_mean: bool, with_std: bool) -> Self {
        SimdStandardScaler {
            mean_: None,
            std_: None,
            with_mean,
            with_std,
        }
    }

    /// Fits the scaler to the input data using SIMD operations
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        let n_samples_f = F::from(n_samples).unwrap();
        let mut mean = Array1::zeros(n_features);
        let mut std = Array1::ones(n_features);

        if self.with_mean {
            // Compute mean for each feature using SIMD
            for j in 0..n_features {
                let col = x.column(j);
                let col_array = col.to_owned();
                mean[j] = F::simd_sum(&col_array.view()) / n_samples_f;
            }
        }

        if self.with_std {
            // Compute standard deviation for each feature using SIMD
            for j in 0..n_features {
                let col = x.column(j);
                let col_array = col.to_owned();

                // Compute variance
                let m = if self.with_mean { mean[j] } else { F::zero() };

                let mean_array = Array1::from_elem(n_samples, m);
                let centered = F::simd_sub(&col_array.view(), &mean_array.view());
                let squared = F::simd_mul(&centered.view(), &centered.view());
                let variance = F::simd_sum(&squared.view()) / n_samples_f;

                std[j] = variance.sqrt();

                // Avoid division by zero
                if std[j] <= F::from(EPSILON).unwrap() {
                    std[j] = F::one();
                }
            }
        }

        self.mean_ = Some(mean);
        self.std_ = Some(std);

        Ok(())
    }

    /// Transforms the input data using SIMD operations
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<F>>
    where
        S: Data<Elem = F>,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if self.mean_.is_none() || self.std_.is_none() {
            return Err(TransformError::TransformationError(
                "Scaler has not been fitted".to_string(),
            ));
        }

        let mean = self.mean_.as_ref().unwrap();
        let std = self.std_.as_ref().unwrap();

        if n_features != mean.len() {
            return Err(TransformError::InvalidInput(format!(
                "X has {} features, but scaler was fitted with {} features",
                n_features,
                mean.len()
            )));
        }

        let mut result = Array2::zeros((n_samples, n_features));

        // Transform each row: (x - mean) / std
        for i in 0..n_samples {
            let row = x.row(i);
            let mut row_array = row.to_owned();

            if self.with_mean {
                // Center the data
                row_array = F::simd_sub(&row_array.view(), &mean.view());
            }

            if self.with_std {
                // Scale to unit variance
                row_array = F::simd_div(&row_array.view(), &std.view());
            }

            for j in 0..n_features {
                result[[i, j]] = row_array[j];
            }
        }

        Ok(result)
    }

    /// Fits and transforms the data in one step
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<F>>
    where
        S: Data<Elem = F>,
    {
        self.fit(x)?;
        self.transform(x)
    }
}
