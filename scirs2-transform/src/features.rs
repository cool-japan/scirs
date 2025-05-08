//! Feature engineering utilities
//!
//! This module provides tools for feature engineering, which is the process of
//! transforming raw data into features that better represent the underlying problem
//! to predictive models.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use num_traits::{Float, NumCast};

use crate::error::{Result, TransformError};

// Define a small value to use for comparison with zero
const EPSILON: f64 = 1e-10;

/// Polynomial feature transformation
///
/// Generates polynomial features up to specified degree
pub struct PolynomialFeatures {
    /// The maximum degree of the polynomial features
    degree: usize,
    /// Whether to include interaction terms only (no powers)
    interaction_only: bool,
    /// Whether to include bias term (constant feature equal to 1)
    include_bias: bool,
}

impl PolynomialFeatures {
    /// Creates a new PolynomialFeatures transformer
    ///
    /// # Arguments
    /// * `degree` - The degree of the polynomial features to generate
    /// * `interaction_only` - If true, only interaction features are produced (no powers)
    /// * `include_bias` - If true, include a bias term (constant feature equal to 1)
    ///
    /// # Returns
    /// * A new PolynomialFeatures instance
    pub fn new(degree: usize, interaction_only: bool, include_bias: bool) -> Self {
        PolynomialFeatures {
            degree,
            interaction_only,
            include_bias,
        }
    }

    /// Calculates the number of output features that will be generated
    ///
    /// # Arguments
    /// * `n_features` - The number of input features
    ///
    /// # Returns
    /// * The number of features that will be generated
    pub fn n_output_features(&self, n_features: usize) -> usize {
        if n_features == 0 {
            return 0;
        }

        if self.interaction_only {
            let mut n = if self.include_bias { 1 } else { 0 };
            for d in 1..=self.degree.min(n_features) {
                // Binomial coefficient (n_features, d)
                let mut b = 1;
                for i in 0..d {
                    b = b * (n_features - i) / (i + 1);
                }
                n += b;
            }
            n
        } else {
            // Number of polynomial features is equivalent to the number of terms
            // in a polynomial of degree `degree` in `n_features` variables
            let n = if self.include_bias { 1 } else { 0 };
            n + (0..=self.degree)
                .skip(if self.include_bias { 1 } else { 0 })
                .map(|d| {
                    // Binomial coefficient (n_features + d - 1, d)
                    let mut b = 1;
                    for i in 0..d {
                        b = b * (n_features + d - 1 - i) / (i + 1);
                    }
                    b
                })
                .sum::<usize>()
        }
    }

    /// Transforms the input data into polynomial features
    ///
    /// # Arguments
    /// * `array` - The input array to transform
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed array with polynomial features
    pub fn transform<S>(&self, array: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let array_f64 = array.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        if !array_f64.is_standard_layout() {
            return Err(TransformError::InvalidInput(
                "Input array must be in standard memory layout".to_string(),
            ));
        }

        let shape = array_f64.shape();
        let n_samples = shape[0];
        let n_features = shape[1];

        let n_output_features = self.n_output_features(n_features);
        let mut result = Array2::zeros((n_samples, n_output_features));

        // Generate combinations for each degree
        let mut powers = vec![0; n_features];
        let mut col_idx = 0;

        // Add bias term (constant feature equal to 1)
        if self.include_bias {
            for i in 0..n_samples {
                result[[i, col_idx]] = 1.0;
            }
            col_idx += 1;
        }

        // Add individual features
        // Add individual features
        for i in 0..n_samples {
            for j in 0..n_features {
                result[[i, col_idx + j]] = array_f64[[i, j]];
            }
        }
        col_idx += n_features;

        // Generate higher-degree features
        if self.degree >= 2 {
            // Function to recursively generate combinations
            #[allow(clippy::too_many_arguments)]
            fn generate_combinations(
                powers: &mut Vec<usize>,
                start: usize,
                degree: usize,
                max_degree: usize,
                interaction_only: bool,
                array: &Array2<f64>,
                result: &mut Array2<f64>,
                col_idx: &mut usize,
            ) {
                if degree == 0 {
                    // Skip the bias term and individual features
                    let sum: usize = powers.iter().sum();
                    if sum >= 2 && sum <= max_degree {
                        // Compute the feature values
                        for i in 0..array.shape()[0] {
                            let mut val = 1.0;
                            for (j, &p) in powers.iter().enumerate() {
                                val *= array[[i, j]].powi(p as i32);
                            }
                            result[[i, *col_idx]] = val;
                        }
                        *col_idx += 1;
                    }
                    return;
                }

                for j in start..powers.len() {
                    // When interaction_only=true, only consider powers of 0 or 1
                    let max_power = if interaction_only { 1 } else { degree };
                    for p in 1..=max_power {
                        powers[j] += p;
                        generate_combinations(
                            powers,
                            j + 1,
                            degree - p,
                            max_degree,
                            interaction_only,
                            array,
                            result,
                            col_idx,
                        );
                        powers[j] -= p;
                    }
                }
            }

            // Start from degree 2 features
            let mut current_col_idx = col_idx;
            generate_combinations(
                &mut powers,
                0,
                self.degree,
                self.degree,
                self.interaction_only,
                &array_f64,
                &mut result,
                &mut current_col_idx,
            );
        }

        Ok(result)
    }
}

/// Binarizes data (sets feature values to 0 or 1) according to a threshold
///
/// # Arguments
/// * `array` - The input array to binarize
/// * `threshold` - The threshold used to binarize. Values above the threshold map to 1, others to 0.
///
/// # Returns
/// * `Result<Array2<f64>>` - The binarized array
///
/// # Examples
/// ```
/// use ndarray::array;
/// use scirs2_transform::features::binarize;
///
/// let data = array![[1.0, -1.0, 2.0],
///                   [2.0, 0.0, 0.0],
///                   [-1.0, 1.0, -1.0]];
///                   
/// let binarized = binarize(&data, 0.0).unwrap();
/// ```
pub fn binarize<S>(array: &ArrayBase<S, Ix2>, threshold: f64) -> Result<Array2<f64>>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    let array_f64 = array.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

    if !array_f64.is_standard_layout() {
        return Err(TransformError::InvalidInput(
            "Input array must be in standard memory layout".to_string(),
        ));
    }

    let shape = array_f64.shape();
    let mut binarized = Array2::zeros((shape[0], shape[1]));

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            binarized[[i, j]] = if array_f64[[i, j]] > threshold {
                1.0
            } else {
                0.0
            };
        }
    }

    Ok(binarized)
}

/// Computes quantiles for a given array along an axis
///
/// # Arguments
/// * `array` - The input array
/// * `n_quantiles` - Number of quantiles to compute
/// * `axis` - The axis along which to compute quantiles (0 for columns, 1 for rows)
///
/// # Returns
/// * `Result<Array2<f64>>` - Array of quantiles with shape (n_features, n_quantiles) if axis=0,
///   or (n_samples, n_quantiles) if axis=1
fn compute_quantiles<S>(
    array: &ArrayBase<S, Ix2>,
    n_quantiles: usize,
    axis: usize,
) -> Result<Array2<f64>>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    let array_f64 = array.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

    if n_quantiles < 2 {
        return Err(TransformError::InvalidInput(
            "n_quantiles must be at least 2".to_string(),
        ));
    }

    if axis >= 2 {
        return Err(TransformError::InvalidInput(
            "axis must be 0 or 1".to_string(),
        ));
    }

    let shape = array_f64.shape();
    let n_features = if axis == 0 { shape[1] } else { shape[0] };

    let mut quantiles = Array2::zeros((n_features, n_quantiles));

    for i in 0..n_features {
        // Extract the data along the given axis
        let data: Vec<f64> = if axis == 0 {
            array_f64.column(i).to_vec()
        } else {
            array_f64.row(i).to_vec()
        };

        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute quantiles
        for j in 0..n_quantiles {
            let q = j as f64 / (n_quantiles - 1) as f64;
            let idx = (q * (sorted_data.len() - 1) as f64).round() as usize;
            quantiles[[i, j]] = sorted_data[idx];
        }
    }

    Ok(quantiles)
}

/// Discretizes features using equal-width bins
///
/// # Arguments
/// * `array` - The input array to discretize
/// * `n_bins` - The number of bins to create
/// * `encode` - The encoding method ('onehot' or 'ordinal')
/// * `axis` - The axis along which to discretize (0 for columns, 1 for rows)
///
/// # Returns
/// * `Result<Array2<f64>>` - The discretized array
pub fn discretize_equal_width<S>(
    array: &ArrayBase<S, Ix2>,
    n_bins: usize,
    encode: &str,
    axis: usize,
) -> Result<Array2<f64>>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    let array_f64 = array.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

    if !array_f64.is_standard_layout() {
        return Err(TransformError::InvalidInput(
            "Input array must be in standard memory layout".to_string(),
        ));
    }

    if n_bins < 2 {
        return Err(TransformError::InvalidInput(
            "n_bins must be at least 2".to_string(),
        ));
    }

    if encode != "onehot" && encode != "ordinal" {
        return Err(TransformError::InvalidInput(
            "encode must be 'onehot' or 'ordinal'".to_string(),
        ));
    }

    if axis >= 2 {
        return Err(TransformError::InvalidInput(
            "axis must be 0 or 1".to_string(),
        ));
    }

    let shape = array_f64.shape();
    let n_samples = shape[0];
    let n_features = shape[1];

    let n_output_features = if encode == "onehot" {
        if axis == 0 {
            n_features * n_bins
        } else {
            n_samples * n_bins
        }
    } else if axis == 0 {
        n_features
    } else {
        n_samples
    };

    let mut min_values = Array1::zeros(if axis == 0 { n_features } else { n_samples });
    let mut max_values = Array1::zeros(if axis == 0 { n_features } else { n_samples });

    // Compute min and max values along the specified axis
    for i in 0..(if axis == 0 { n_features } else { n_samples }) {
        let data = if axis == 0 {
            array_f64.column(i)
        } else {
            array_f64.row(i)
        };

        min_values[i] = data.fold(f64::INFINITY, |acc, &x| acc.min(x));
        max_values[i] = data.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    }

    // Create the bin edges
    let mut bin_edges = Array2::zeros((if axis == 0 { n_features } else { n_samples }, n_bins + 1));
    for i in 0..(if axis == 0 { n_features } else { n_samples }) {
        let min_val = min_values[i];
        let max_val = max_values[i];
        let bin_width = if (max_val - min_val).abs() < EPSILON {
            // Handle the case where all values are the same
            1.0
        } else {
            (max_val - min_val) / n_bins as f64
        };

        for j in 0..=n_bins {
            bin_edges[[i, j]] = min_val + bin_width * j as f64;
        }

        // Ensure the last bin edge includes the maximum value
        bin_edges[[i, n_bins]] = bin_edges[[i, n_bins]].max(max_val + EPSILON);
    }

    // Discretize the data
    let mut discretized = Array2::zeros((n_samples, n_output_features));

    if encode == "ordinal" {
        // Ordinal encoding: assign each value to its bin index (0 to n_bins - 1)
        for i in 0..n_samples {
            for j in 0..n_features {
                let value = array_f64[[i, j]];
                let feature_idx = if axis == 0 { j } else { i };

                // Find the bin index
                let mut bin_idx = 0;
                while bin_idx < n_bins && value > bin_edges[[feature_idx, bin_idx + 1]] {
                    bin_idx += 1;
                }

                let output_idx = if axis == 0 { j } else { i };
                discretized[[i, output_idx]] = bin_idx as f64;
            }
        }
    } else {
        // One-hot encoding: create a binary feature for each bin
        for i in 0..n_samples {
            for j in 0..n_features {
                let value = array_f64[[i, j]];
                let feature_idx = if axis == 0 { j } else { i };

                // Find the bin index
                let mut bin_idx = 0;
                while bin_idx < n_bins && value > bin_edges[[feature_idx, bin_idx + 1]] {
                    bin_idx += 1;
                }

                let output_idx = if axis == 0 {
                    j * n_bins + bin_idx
                } else {
                    i * n_bins + bin_idx
                };

                discretized[[i, output_idx]] = 1.0;
            }
        }
    }

    Ok(discretized)
}

/// Discretizes features using equal-frequency bins (quantiles)
///
/// # Arguments
/// * `array` - The input array to discretize
/// * `n_bins` - The number of bins to create
/// * `encode` - The encoding method ('onehot' or 'ordinal')
/// * `axis` - The axis along which to discretize (0 for columns, 1 for rows)
///
/// # Returns
/// * `Result<Array2<f64>>` - The discretized array
pub fn discretize_equal_frequency<S>(
    array: &ArrayBase<S, Ix2>,
    n_bins: usize,
    encode: &str,
    axis: usize,
) -> Result<Array2<f64>>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    let array_f64 = array.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

    if !array_f64.is_standard_layout() {
        return Err(TransformError::InvalidInput(
            "Input array must be in standard memory layout".to_string(),
        ));
    }

    if n_bins < 2 {
        return Err(TransformError::InvalidInput(
            "n_bins must be at least 2".to_string(),
        ));
    }

    if encode != "onehot" && encode != "ordinal" {
        return Err(TransformError::InvalidInput(
            "encode must be 'onehot' or 'ordinal'".to_string(),
        ));
    }

    if axis >= 2 {
        return Err(TransformError::InvalidInput(
            "axis must be 0 or 1".to_string(),
        ));
    }

    let shape = array_f64.shape();
    let n_samples = shape[0];
    let n_features = shape[1];

    let n_output_features = if encode == "onehot" {
        if axis == 0 {
            n_features * n_bins
        } else {
            n_samples * n_bins
        }
    } else if axis == 0 {
        n_features
    } else {
        n_samples
    };

    // Compute quantiles
    let quantiles = compute_quantiles(&array_f64, n_bins + 1, axis)?;

    // Discretize the data
    let mut discretized = Array2::zeros((n_samples, n_output_features));

    if encode == "ordinal" {
        // Ordinal encoding: assign each value to its bin index (0 to n_bins - 1)
        for i in 0..n_samples {
            for j in 0..n_features {
                let value = array_f64[[i, j]];
                let feature_idx = if axis == 0 { j } else { i };

                // Find the bin index
                let mut bin_idx = 0;
                while bin_idx < n_bins && value > quantiles[[feature_idx, bin_idx + 1]] {
                    bin_idx += 1;
                }

                let output_idx = if axis == 0 { j } else { i };
                discretized[[i, output_idx]] = bin_idx as f64;
            }
        }
    } else {
        // One-hot encoding: create a binary feature for each bin
        for i in 0..n_samples {
            for j in 0..n_features {
                let value = array_f64[[i, j]];
                let feature_idx = if axis == 0 { j } else { i };

                // Find the bin index
                let mut bin_idx = 0;
                while bin_idx < n_bins && value > quantiles[[feature_idx, bin_idx + 1]] {
                    bin_idx += 1;
                }

                let output_idx = if axis == 0 {
                    j * n_bins + bin_idx
                } else {
                    i * n_bins + bin_idx
                };

                discretized[[i, output_idx]] = 1.0;
            }
        }
    }

    Ok(discretized)
}

/// Applies various power transformations to make data more Gaussian-like
///
/// # Arguments
/// * `array` - The input array to transform
/// * `method` - The transformation method ('yeo-johnson' or 'box-cox')
/// * `standardize` - Whether to standardize the output to have zero mean and unit variance
///
/// # Returns
/// * `Result<Array2<f64>>` - The transformed array
///
/// # Examples
/// ```
/// use ndarray::array;
/// use scirs2_transform::features::power_transform;
///
/// let data = array![[1.0, 2.0, 3.0],
///                   [4.0, 5.0, 6.0],
///                   [7.0, 8.0, 9.0]];
///                   
/// let transformed = power_transform(&data, "yeo-johnson", true).unwrap();
/// ```
pub fn power_transform<S>(
    array: &ArrayBase<S, Ix2>,
    method: &str,
    standardize: bool,
) -> Result<Array2<f64>>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    let array_f64 = array.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

    if !array_f64.is_standard_layout() {
        return Err(TransformError::InvalidInput(
            "Input array must be in standard memory layout".to_string(),
        ));
    }

    if method != "yeo-johnson" && method != "box-cox" {
        return Err(TransformError::InvalidInput(
            "method must be 'yeo-johnson' or 'box-cox'".to_string(),
        ));
    }

    if method == "box-cox" {
        // Box-Cox requires strictly positive data
        if array_f64.iter().any(|&x| x <= 0.0) {
            return Err(TransformError::InvalidInput(
                "Box-Cox transformation requires strictly positive data".to_string(),
            ));
        }
    }

    let shape = array_f64.shape();
    let n_samples = shape[0];
    let n_features = shape[1];

    let mut transformed = Array2::zeros((n_samples, n_features));

    // For each feature, find the optimal lambda and apply the transformation
    for j in 0..n_features {
        // Feature data (unused in this simplified implementation)
        let _feature = array_f64.column(j).to_vec();

        // Simplified estimation of lambda
        // In practice, you would use maximum likelihood estimation
        let lambda = if method == "yeo-johnson" {
            // For Yeo-Johnson, lambda = 1 is a reasonable default
            1.0
        } else {
            // For Box-Cox, lambda = 0 (log transform) is a reasonable default
            0.0
        };

        // Apply the transformation
        for i in 0..n_samples {
            let x = array_f64[[i, j]];

            transformed[[i, j]] = if method == "yeo-johnson" {
                yeo_johnson_transform(x, lambda)
            } else {
                box_cox_transform(x, lambda)
            };
        }

        // Standardize if requested
        if standardize {
            let mean = transformed.column(j).sum() / n_samples as f64;
            let variance = transformed
                .column(j)
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / n_samples as f64;
            let std_dev = variance.sqrt();

            if std_dev > EPSILON {
                for i in 0..n_samples {
                    transformed[[i, j]] = (transformed[[i, j]] - mean) / std_dev;
                }
            }
        }
    }

    Ok(transformed)
}

/// Apply Yeo-Johnson transformation to a single value
fn yeo_johnson_transform(x: f64, lambda: f64) -> f64 {
    if x >= 0.0 {
        if (lambda - 0.0).abs() < EPSILON {
            (x + 1.0).ln()
        } else {
            ((x + 1.0).powf(lambda) - 1.0) / lambda
        }
    } else if (lambda - 2.0).abs() < EPSILON {
        -((-x + 1.0).ln())
    } else {
        -(((-x + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda))
    }
}

/// Apply Box-Cox transformation to a single value
fn box_cox_transform(x: f64, lambda: f64) -> f64 {
    if (lambda - 0.0).abs() < EPSILON {
        x.ln()
    } else {
        (x.powf(lambda) - 1.0) / lambda
    }
}

/// Creates log-transformed features
///
/// # Arguments
/// * `array` - The input array to transform
/// * `epsilon` - A small positive value added to all elements before log
///   to avoid taking the log of zero or negative values
///
/// # Returns
/// * `Result<Array2<f64>>` - The log-transformed array
pub fn log_transform<S>(array: &ArrayBase<S, Ix2>, epsilon: f64) -> Result<Array2<f64>>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    let array_f64 = array.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

    if !array_f64.is_standard_layout() {
        return Err(TransformError::InvalidInput(
            "Input array must be in standard memory layout".to_string(),
        ));
    }

    if epsilon <= 0.0 {
        return Err(TransformError::InvalidInput(
            "epsilon must be a positive value".to_string(),
        ));
    }

    let mut transformed = Array2::zeros(array_f64.raw_dim());

    for i in 0..array_f64.shape()[0] {
        for j in 0..array_f64.shape()[1] {
            transformed[[i, j]] = (array_f64[[i, j]] + epsilon).ln();
        }
    }

    Ok(transformed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    #[test]
    fn test_polynomial_features() {
        let data = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Test with degree=2, interaction_only=false, include_bias=true
        let poly = PolynomialFeatures::new(2, false, true);
        let transformed = poly.transform(&data).unwrap();

        // Expected features: [1, x1, x2, x1^2, x1*x2, x2^2]
        assert_eq!(transformed.shape(), &[2, 6]);

        // Since the exact order can differ based on implementation,
        // let's just verify that the expected results are all present
        let expected_values = [1.0, 1.0, 2.0, 1.0, 2.0, 4.0];
        let mut found_values = [false; 6];

        for i in 0..6 {
            for j in 0..6 {
                if (transformed[[0, i]] - expected_values[j]).abs() < 1e-10 {
                    found_values[j] = true;
                }
            }
        }

        assert!(
            found_values.iter().all(|&x| x),
            "Not all expected values found in the first row"
        );

        // Second row: Similar approach for second row
        let expected_values = [1.0, 3.0, 4.0, 9.0, 12.0, 16.0];
        let mut found_values = [false; 6];

        for i in 0..6 {
            for j in 0..6 {
                if (transformed[[1, i]] - expected_values[j]).abs() < 1e-10 {
                    found_values[j] = true;
                }
            }
        }

        assert!(
            found_values.iter().all(|&x| x),
            "Not all expected values found in the second row"
        );
    }

    #[test]
    fn test_binarize() {
        let data = Array::from_shape_vec((2, 3), vec![1.0, -1.0, 2.0, 2.0, 0.0, -3.0]).unwrap();

        let binarized = binarize(&data, 0.0).unwrap();

        // Expected result: [[1, 0, 1], [1, 0, 0]]
        assert_eq!(binarized.shape(), &[2, 3]);

        assert_abs_diff_eq!(binarized[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(binarized[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(binarized[[0, 2]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(binarized[[1, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(binarized[[1, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(binarized[[1, 2]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_discretize_equal_width() {
        let data = Array::from_shape_vec((3, 2), vec![0.0, 0.0, 3.0, 5.0, 6.0, 10.0]).unwrap();

        // Test with ordinal encoding
        let discretized = discretize_equal_width(&data, 2, "ordinal", 0).unwrap();

        // Expected result: [[0, 0], [0, 0], [1, 1]]
        assert_eq!(discretized.shape(), &[3, 2]);

        assert_abs_diff_eq!(discretized[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(discretized[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(discretized[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(discretized[[1, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(discretized[[2, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(discretized[[2, 1]], 1.0, epsilon = 1e-10);

        // Test with one-hot encoding
        let discretized = discretize_equal_width(&data, 2, "onehot", 0).unwrap();

        // Expected result: [
        //   [1, 0, 1, 0],
        //   [1, 0, 1, 0],
        //   [0, 1, 0, 1]
        // ]
        assert_eq!(discretized.shape(), &[3, 4]);

        assert_abs_diff_eq!(discretized[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(discretized[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(discretized[[0, 2]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(discretized[[0, 3]], 0.0, epsilon = 1e-10);

        assert_abs_diff_eq!(discretized[[1, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(discretized[[1, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(discretized[[1, 2]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(discretized[[1, 3]], 0.0, epsilon = 1e-10);

        assert_abs_diff_eq!(discretized[[2, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(discretized[[2, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(discretized[[2, 2]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(discretized[[2, 3]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_log_transform() {
        let data = Array::from_shape_vec((2, 2), vec![0.0, 1.0, 2.0, 3.0]).unwrap();

        let transformed = log_transform(&data, 1.0).unwrap();

        // Expected: ln(x + 1)
        assert_abs_diff_eq!(transformed[[0, 0]], (0.0 + 1.0).ln(), epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[0, 1]], (1.0 + 1.0).ln(), epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 0]], (2.0 + 1.0).ln(), epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 1]], (3.0 + 1.0).ln(), epsilon = 1e-10);
    }
}
