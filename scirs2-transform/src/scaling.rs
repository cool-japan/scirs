//! Advanced scaling and transformation methods
//!
//! This module provides sophisticated scaling methods that go beyond basic normalization,
//! including quantile transformations and robust scaling methods.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use num_traits::{Float, NumCast};

use crate::error::{Result, TransformError};

// Define a small value to use for comparison with zero
const EPSILON: f64 = 1e-10;

/// QuantileTransformer for non-linear transformations
///
/// This transformer transforms features to follow a uniform or normal distribution
/// using quantiles information. This method reduces the impact of outliers.
pub struct QuantileTransformer {
    /// Number of quantiles to estimate
    n_quantiles: usize,
    /// Output distribution ('uniform' or 'normal')
    output_distribution: String,
    /// Whether to clip transformed values to bounds [0, 1] for uniform distribution
    clip: bool,
    /// The quantiles for each feature
    quantiles: Option<Array2<f64>>,
    /// References values for each quantile
    references: Option<Array1<f64>>,
}

impl QuantileTransformer {
    /// Creates a new QuantileTransformer
    ///
    /// # Arguments
    /// * `n_quantiles` - Number of quantiles to estimate (default: 1000)
    /// * `output_distribution` - Target distribution ('uniform' or 'normal')
    /// * `clip` - Whether to clip transformed values
    ///
    /// # Returns
    /// * A new QuantileTransformer instance
    pub fn new(n_quantiles: usize, output_distribution: &str, clip: bool) -> Result<Self> {
        if n_quantiles < 2 {
            return Err(TransformError::InvalidInput(
                "n_quantiles must be at least 2".to_string(),
            ));
        }

        if output_distribution != "uniform" && output_distribution != "normal" {
            return Err(TransformError::InvalidInput(
                "output_distribution must be 'uniform' or 'normal'".to_string(),
            ));
        }

        Ok(QuantileTransformer {
            n_quantiles,
            output_distribution: output_distribution.to_string(),
            clip,
            quantiles: None,
            references: None,
        })
    }

    /// Fits the QuantileTransformer to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if self.n_quantiles > n_samples {
            return Err(TransformError::InvalidInput(format!(
                "n_quantiles ({}) cannot be greater than n_samples ({})",
                self.n_quantiles, n_samples
            )));
        }

        // Compute quantiles for each feature
        let mut quantiles = Array2::zeros((n_features, self.n_quantiles));

        for j in 0..n_features {
            // Extract feature data and sort it
            let mut feature_data: Vec<f64> = x_f64.column(j).to_vec();
            feature_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Compute quantiles
            for i in 0..self.n_quantiles {
                let q = i as f64 / (self.n_quantiles - 1) as f64;
                let idx = (q * (feature_data.len() - 1) as f64).round() as usize;
                quantiles[[j, i]] = feature_data[idx];
            }
        }

        // Generate reference distribution
        let references = if self.output_distribution == "uniform" {
            // Uniform distribution references
            Array1::from_shape_fn(self.n_quantiles, |i| {
                i as f64 / (self.n_quantiles - 1) as f64
            })
        } else {
            // Normal distribution references (using inverse normal CDF approximation)
            Array1::from_shape_fn(self.n_quantiles, |i| {
                let u = i as f64 / (self.n_quantiles - 1) as f64;
                // Clamp u to avoid extreme values
                let u_clamped = u.clamp(1e-7, 1.0 - 1e-7);
                inverse_normal_cdf(u_clamped)
            })
        };

        self.quantiles = Some(quantiles);
        self.references = Some(references);

        Ok(())
    }

    /// Transforms the input data using the fitted QuantileTransformer
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if self.quantiles.is_none() || self.references.is_none() {
            return Err(TransformError::TransformationError(
                "QuantileTransformer has not been fitted".to_string(),
            ));
        }

        let quantiles = self.quantiles.as_ref().unwrap();
        let references = self.references.as_ref().unwrap();

        if n_features != quantiles.shape()[0] {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but QuantileTransformer was fitted with {} features",
                n_features,
                quantiles.shape()[0]
            )));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                let value = x_f64[[i, j]];

                // Find the position of the value in the quantiles
                let feature_quantiles = quantiles.row(j);

                // Find the index where value would be inserted
                let mut lower_idx = 0;
                let mut upper_idx = self.n_quantiles - 1;

                // Handle edge cases
                if value <= feature_quantiles[0] {
                    transformed[[i, j]] = references[0];
                    continue;
                }
                if value >= feature_quantiles[self.n_quantiles - 1] {
                    transformed[[i, j]] = references[self.n_quantiles - 1];
                    continue;
                }

                // Binary search to find the interval
                while upper_idx - lower_idx > 1 {
                    let mid = (lower_idx + upper_idx) / 2;
                    if value <= feature_quantiles[mid] {
                        upper_idx = mid;
                    } else {
                        lower_idx = mid;
                    }
                }

                // Linear interpolation between reference values
                let lower_quantile = feature_quantiles[lower_idx];
                let upper_quantile = feature_quantiles[upper_idx];
                let lower_ref = references[lower_idx];
                let upper_ref = references[upper_idx];

                if (upper_quantile - lower_quantile).abs() < EPSILON {
                    transformed[[i, j]] = lower_ref;
                } else {
                    let ratio = (value - lower_quantile) / (upper_quantile - lower_quantile);
                    transformed[[i, j]] = lower_ref + ratio * (upper_ref - lower_ref);
                }
            }
        }

        // Apply clipping if requested and output distribution is uniform
        if self.clip && self.output_distribution == "uniform" {
            for i in 0..n_samples {
                for j in 0..n_features {
                    transformed[[i, j]] = transformed[[i, j]].clamp(0.0, 1.0);
                }
            }
        }

        Ok(transformed)
    }

    /// Fits the QuantileTransformer to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
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

    /// Returns the quantiles for each feature
    ///
    /// # Returns
    /// * `Option<&Array2<f64>>` - The quantiles, shape (n_features, n_quantiles)
    pub fn quantiles(&self) -> Option<&Array2<f64>> {
        self.quantiles.as_ref()
    }
}

/// Approximation of the inverse normal cumulative distribution function
///
/// This uses the Beasley-Springer-Moro algorithm for approximating the inverse normal CDF
fn inverse_normal_cdf(u: f64) -> f64 {
    // Constants for the Beasley-Springer-Moro algorithm
    const A0: f64 = 2.50662823884;
    const A1: f64 = -18.61500062529;
    const A2: f64 = 41.39119773534;
    const A3: f64 = -25.44106049637;
    const B1: f64 = -8.47351093090;
    const B2: f64 = 23.08336743743;
    const B3: f64 = -21.06224101826;
    const B4: f64 = 3.13082909833;
    const C0: f64 = 0.3374754822726147;
    const C1: f64 = 0.9761690190917186;
    const C2: f64 = 0.1607979714918209;
    const C3: f64 = 0.0276438810333863;
    const C4: f64 = 0.0038405729373609;
    const C5: f64 = 0.0003951896511919;
    const C6: f64 = 0.0000321767881768;
    const C7: f64 = 0.0000002888167364;
    const C8: f64 = 0.0000003960315187;

    let y = u - 0.5;

    if y.abs() < 0.42 {
        // Central region
        let r = y * y;
        y * (((A3 * r + A2) * r + A1) * r + A0) / ((((B4 * r + B3) * r + B2) * r + B1) * r + 1.0)
    } else {
        // Tail region
        let r = if y > 0.0 { 1.0 - u } else { u };
        let r = (-r.ln()).ln();

        let result = C0
            + r * (C1 + r * (C2 + r * (C3 + r * (C4 + r * (C5 + r * (C6 + r * (C7 + r * C8)))))));

        if y < 0.0 {
            -result
        } else {
            result
        }
    }
}

/// MaxAbsScaler for scaling features by their maximum absolute value
///
/// This scaler scales each feature individually such that the maximal absolute value
/// of each feature in the training set will be 1.0. It does not shift/center the data,
/// and thus does not destroy any sparsity.
pub struct MaxAbsScaler {
    /// Maximum absolute values for each feature (learned during fit)
    max_abs_: Option<Array1<f64>>,
    /// Scale factors for each feature (1 / max_abs_)
    scale_: Option<Array1<f64>>,
}

impl MaxAbsScaler {
    /// Creates a new MaxAbsScaler
    ///
    /// # Returns
    /// * A new MaxAbsScaler instance
    ///
    /// # Examples
    /// ```
    /// use scirs2_transform::scaling::MaxAbsScaler;
    ///
    /// let scaler = MaxAbsScaler::new();
    /// ```
    pub fn new() -> Self {
        MaxAbsScaler {
            max_abs_: None,
            scale_: None,
        }
    }

    /// Creates a MaxAbsScaler with default settings (same as new())
    pub fn with_defaults() -> Self {
        Self::new()
    }

    /// Fits the MaxAbsScaler to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        // Compute maximum absolute value for each feature
        let mut max_abs = Array1::zeros(n_features);

        for j in 0..n_features {
            let feature_data = x_f64.column(j);
            let max_abs_value = feature_data
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, |acc, x| acc.max(x));

            max_abs[j] = max_abs_value;
        }

        // Compute scale factors (avoid division by zero)
        let scale = max_abs.mapv(|max_abs_val| {
            if max_abs_val > EPSILON {
                1.0 / max_abs_val
            } else {
                1.0 // If max_abs is 0, don't scale (feature is constant zero)
            }
        });

        self.max_abs_ = Some(max_abs);
        self.scale_ = Some(scale);

        Ok(())
    }

    /// Transforms the input data using the fitted MaxAbsScaler
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The scaled data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if self.scale_.is_none() {
            return Err(TransformError::TransformationError(
                "MaxAbsScaler has not been fitted".to_string(),
            ));
        }

        let scale = self.scale_.as_ref().unwrap();

        if n_features != scale.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but MaxAbsScaler was fitted with {} features",
                n_features,
                scale.len()
            )));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        // Scale each feature by its scale factor
        for i in 0..n_samples {
            for j in 0..n_features {
                transformed[[i, j]] = x_f64[[i, j]] * scale[j];
            }
        }

        Ok(transformed)
    }

    /// Fits the MaxAbsScaler to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The scaled data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Inverse transforms the scaled data back to original scale
    ///
    /// # Arguments
    /// * `x` - The scaled data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The data in original scale
    pub fn inverse_transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if self.max_abs_.is_none() {
            return Err(TransformError::TransformationError(
                "MaxAbsScaler has not been fitted".to_string(),
            ));
        }

        let max_abs = self.max_abs_.as_ref().unwrap();

        if n_features != max_abs.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but MaxAbsScaler was fitted with {} features",
                n_features,
                max_abs.len()
            )));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        // Scale back by multiplying with max_abs values
        for i in 0..n_samples {
            for j in 0..n_features {
                transformed[[i, j]] = x_f64[[i, j]] * max_abs[j];
            }
        }

        Ok(transformed)
    }

    /// Returns the maximum absolute values for each feature
    ///
    /// # Returns
    /// * `Option<&Array1<f64>>` - The maximum absolute values
    pub fn max_abs(&self) -> Option<&Array1<f64>> {
        self.max_abs_.as_ref()
    }

    /// Returns the scale factors for each feature
    ///
    /// # Returns
    /// * `Option<&Array1<f64>>` - The scale factors (1 / max_abs)
    pub fn scale(&self) -> Option<&Array1<f64>> {
        self.scale_.as_ref()
    }
}

impl Default for MaxAbsScaler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    #[test]
    fn test_quantile_transformer_uniform() {
        // Create test data with different distributions
        let data = Array::from_shape_vec(
            (6, 2),
            vec![
                1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 100.0, 1000.0,
            ], // Last row has outliers
        )
        .unwrap();

        let mut transformer = QuantileTransformer::new(5, "uniform", true).unwrap();
        let transformed = transformer.fit_transform(&data).unwrap();

        // Check that the shape is preserved
        assert_eq!(transformed.shape(), &[6, 2]);

        // For uniform distribution, values should be between 0 and 1
        for i in 0..6 {
            for j in 0..2 {
                assert!(
                    transformed[[i, j]] >= 0.0 && transformed[[i, j]] <= 1.0,
                    "Value at [{}, {}] = {} is not in [0, 1]",
                    i,
                    j,
                    transformed[[i, j]]
                );
            }
        }

        // The smallest value should map to 0 and largest to 1
        assert_abs_diff_eq!(transformed[[0, 0]], 0.0, epsilon = 1e-10); // min of column 0
        assert_abs_diff_eq!(transformed[[5, 0]], 1.0, epsilon = 1e-10); // max of column 0
        assert_abs_diff_eq!(transformed[[0, 1]], 0.0, epsilon = 1e-10); // min of column 1
        assert_abs_diff_eq!(transformed[[5, 1]], 1.0, epsilon = 1e-10); // max of column 1
    }

    #[test]
    fn test_quantile_transformer_normal() {
        // Create test data
        let data = Array::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        let mut transformer = QuantileTransformer::new(5, "normal", false).unwrap();
        let transformed = transformer.fit_transform(&data).unwrap();

        // Check that the shape is preserved
        assert_eq!(transformed.shape(), &[5, 1]);

        // The middle value should be close to 0 (median of normal distribution)
        assert_abs_diff_eq!(transformed[[2, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quantile_transformer_errors() {
        // Test invalid n_quantiles
        assert!(QuantileTransformer::new(1, "uniform", true).is_err());

        // Test invalid output_distribution
        assert!(QuantileTransformer::new(100, "invalid", true).is_err());

        // Test fitting with insufficient data
        let small_data = Array::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let mut transformer = QuantileTransformer::new(10, "uniform", true).unwrap();
        assert!(transformer.fit(&small_data).is_err());
    }

    #[test]
    fn test_inverse_normal_cdf() {
        // Test some known values
        assert_abs_diff_eq!(inverse_normal_cdf(0.5), 0.0, epsilon = 1e-6);
        assert!(inverse_normal_cdf(0.1) < 0.0); // Should be negative
        assert!(inverse_normal_cdf(0.9) > 0.0); // Should be positive
    }

    #[test]
    fn test_max_abs_scaler_basic() {
        // Create test data with different ranges
        // Feature 0: [-4, -2, 0, 2, 4] -> max_abs = 4
        // Feature 1: [-10, -5, 0, 5, 10] -> max_abs = 10
        let data = Array::from_shape_vec(
            (5, 2),
            vec![-4.0, -10.0, -2.0, -5.0, 0.0, 0.0, 2.0, 5.0, 4.0, 10.0],
        )
        .unwrap();

        let mut scaler = MaxAbsScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();

        // Check that the shape is preserved
        assert_eq!(scaled.shape(), &[5, 2]);

        // Check the maximum absolute values
        let max_abs = scaler.max_abs().unwrap();
        assert_abs_diff_eq!(max_abs[0], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(max_abs[1], 10.0, epsilon = 1e-10);

        // Check the scale factors
        let scale = scaler.scale().unwrap();
        assert_abs_diff_eq!(scale[0], 0.25, epsilon = 1e-10); // 1/4
        assert_abs_diff_eq!(scale[1], 0.1, epsilon = 1e-10); // 1/10

        // Check that the maximum absolute value in each feature is 1.0
        for j in 0..2 {
            let feature_max = scaled
                .column(j)
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, f64::max);
            assert_abs_diff_eq!(feature_max, 1.0, epsilon = 1e-10);
        }

        // Check specific scaled values
        assert_abs_diff_eq!(scaled[[0, 0]], -1.0, epsilon = 1e-10); // -4 / 4 = -1
        assert_abs_diff_eq!(scaled[[0, 1]], -1.0, epsilon = 1e-10); // -10 / 10 = -1
        assert_abs_diff_eq!(scaled[[2, 0]], 0.0, epsilon = 1e-10); // 0 / 4 = 0
        assert_abs_diff_eq!(scaled[[2, 1]], 0.0, epsilon = 1e-10); // 0 / 10 = 0
        assert_abs_diff_eq!(scaled[[4, 0]], 1.0, epsilon = 1e-10); // 4 / 4 = 1
        assert_abs_diff_eq!(scaled[[4, 1]], 1.0, epsilon = 1e-10); // 10 / 10 = 1
    }

    #[test]
    fn test_max_abs_scaler_positive_only() {
        // Test with positive-only data
        let data = Array::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 6.0, 5.0, 10.0]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();

        // Check maximum absolute values
        let max_abs = scaler.max_abs().unwrap();
        assert_abs_diff_eq!(max_abs[0], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(max_abs[1], 10.0, epsilon = 1e-10);

        // Check scaled values
        assert_abs_diff_eq!(scaled[[0, 0]], 0.2, epsilon = 1e-10); // 1 / 5
        assert_abs_diff_eq!(scaled[[0, 1]], 0.2, epsilon = 1e-10); // 2 / 10
        assert_abs_diff_eq!(scaled[[2, 0]], 1.0, epsilon = 1e-10); // 5 / 5
        assert_abs_diff_eq!(scaled[[2, 1]], 1.0, epsilon = 1e-10); // 10 / 10
    }

    #[test]
    fn test_max_abs_scaler_inverse_transform() {
        let data = Array::from_shape_vec((3, 2), vec![-6.0, 8.0, 0.0, -4.0, 3.0, 12.0]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();
        let inverse = scaler.inverse_transform(&scaled).unwrap();

        // Check that inverse transform recovers original data
        assert_eq!(inverse.shape(), data.shape());
        for i in 0..3 {
            for j in 0..2 {
                assert_abs_diff_eq!(inverse[[i, j]], data[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_max_abs_scaler_constant_feature() {
        // Test with a constant feature (all zeros)
        let data = Array::from_shape_vec((3, 2), vec![0.0, 5.0, 0.0, 10.0, 0.0, 15.0]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();

        // Constant zero feature should remain zero
        for i in 0..3 {
            assert_abs_diff_eq!(scaled[[i, 0]], 0.0, epsilon = 1e-10);
        }

        // Second feature should be scaled normally
        assert_abs_diff_eq!(scaled[[0, 1]], 1.0 / 3.0, epsilon = 1e-10); // 5 / 15
        assert_abs_diff_eq!(scaled[[2, 1]], 1.0, epsilon = 1e-10); // 15 / 15
    }

    #[test]
    fn test_max_abs_scaler_errors() {
        // Test with empty data
        let empty_data = Array::<f64, _>::zeros((0, 2));
        let mut scaler = MaxAbsScaler::new();
        assert!(scaler.fit(&empty_data).is_err());

        // Test transform before fit
        let data = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let unfitted_scaler = MaxAbsScaler::new();
        assert!(unfitted_scaler.transform(&data).is_err());
        assert!(unfitted_scaler.inverse_transform(&data).is_err());

        // Test feature dimension mismatch
        let train_data = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let test_data = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        scaler.fit(&train_data).unwrap();
        assert!(scaler.transform(&test_data).is_err());
        assert!(scaler.inverse_transform(&test_data).is_err());
    }

    #[test]
    fn test_max_abs_scaler_single_feature() {
        // Test with single feature
        let data = Array::from_shape_vec((4, 1), vec![-8.0, -2.0, 4.0, 6.0]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();

        // Maximum absolute value should be 8.0
        let max_abs = scaler.max_abs().unwrap();
        assert_abs_diff_eq!(max_abs[0], 8.0, epsilon = 1e-10);

        // Check scaled values
        assert_abs_diff_eq!(scaled[[0, 0]], -1.0, epsilon = 1e-10); // -8 / 8
        assert_abs_diff_eq!(scaled[[1, 0]], -0.25, epsilon = 1e-10); // -2 / 8
        assert_abs_diff_eq!(scaled[[2, 0]], 0.5, epsilon = 1e-10); // 4 / 8
        assert_abs_diff_eq!(scaled[[3, 0]], 0.75, epsilon = 1e-10); // 6 / 8
    }

    #[test]
    fn test_max_abs_scaler_sparse_preservation() {
        // Test that zero values remain zero (sparsity preservation)
        let data = Array::from_shape_vec(
            (4, 3),
            vec![
                0.0, 5.0, 0.0, // Row with zeros
                10.0, 0.0, -8.0, // Another row with zeros
                0.0, 0.0, 4.0, // Row with multiple zeros
                -5.0, 10.0, 0.0, // Row with zero at end
            ],
        )
        .unwrap();

        let mut scaler = MaxAbsScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();

        // Check that zeros remain zeros
        assert_abs_diff_eq!(scaled[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[0, 2]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[1, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[2, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[2, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[3, 2]], 0.0, epsilon = 1e-10);

        // Check that non-zero values are scaled correctly
        // Feature 0: max_abs = 10, Feature 1: max_abs = 10, Feature 2: max_abs = 8
        assert_abs_diff_eq!(scaled[[0, 1]], 0.5, epsilon = 1e-10); // 5 / 10
        assert_abs_diff_eq!(scaled[[1, 0]], 1.0, epsilon = 1e-10); // 10 / 10
        assert_abs_diff_eq!(scaled[[1, 2]], -1.0, epsilon = 1e-10); // -8 / 8
        assert_abs_diff_eq!(scaled[[2, 2]], 0.5, epsilon = 1e-10); // 4 / 8
        assert_abs_diff_eq!(scaled[[3, 0]], -0.5, epsilon = 1e-10); // -5 / 10
        assert_abs_diff_eq!(scaled[[3, 1]], 1.0, epsilon = 1e-10); // 10 / 10
    }
}
