//! Feature engineering utilities
//!
//! This module provides tools for feature engineering, which is the process of
//! transforming raw data into features that better represent the underlying problem
//! to predictive models.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use num_traits::{Float, NumCast};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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

/// Optimized PowerTransformer for making data more Gaussian-like
///
/// This is an enhanced version of the power transformation that includes:
/// - Optimal lambda parameter estimation using maximum likelihood
/// - Vectorized operations for better performance
/// - Parallel processing for multiple features
/// - Fit/transform API for reusable transformations
/// - Inverse transformation capability
/// - Enhanced numerical stability
#[derive(Debug, Clone)]
pub struct PowerTransformer {
    /// Transformation method ('yeo-johnson' or 'box-cox')
    method: String,
    /// Whether to standardize the output
    standardize: bool,
    /// Optimal lambda parameters for each feature (computed during fit)
    lambdas_: Option<Array1<f64>>,
    /// Means for standardization (computed during fit)
    means_: Option<Array1<f64>>,
    /// Standard deviations for standardization (computed during fit)
    stds_: Option<Array1<f64>>,
    /// Whether the transformer has been fitted
    is_fitted: bool,
}

impl PowerTransformer {
    /// Creates a new PowerTransformer
    ///
    /// # Arguments
    /// * `method` - The transformation method ('yeo-johnson' or 'box-cox')
    /// * `standardize` - Whether to standardize the output to have zero mean and unit variance
    ///
    /// # Returns
    /// * A new PowerTransformer instance
    pub fn new(method: &str, standardize: bool) -> Result<Self> {
        if method != "yeo-johnson" && method != "box-cox" {
            return Err(TransformError::InvalidInput(
                "method must be 'yeo-johnson' or 'box-cox'".to_string(),
            ));
        }

        Ok(PowerTransformer {
            method: method.to_string(),
            standardize,
            lambdas_: None,
            means_: None,
            stds_: None,
            is_fitted: false,
        })
    }

    /// Creates a new PowerTransformer with Yeo-Johnson method
    pub fn yeo_johnson(standardize: bool) -> Self {
        PowerTransformer {
            method: "yeo-johnson".to_string(),
            standardize,
            lambdas_: None,
            means_: None,
            stds_: None,
            is_fitted: false,
        }
    }

    /// Creates a new PowerTransformer with Box-Cox method
    pub fn box_cox(standardize: bool) -> Self {
        PowerTransformer {
            method: "box-cox".to_string(),
            standardize,
            lambdas_: None,
            means_: None,
            stds_: None,
            is_fitted: false,
        }
    }

    /// Fits the PowerTransformer to the input data
    ///
    /// This computes the optimal lambda parameters for each feature using maximum likelihood estimation
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

        if !x_f64.is_standard_layout() {
            return Err(TransformError::InvalidInput(
                "Input array must be in standard memory layout".to_string(),
            ));
        }

        let shape = x_f64.shape();
        let n_samples = shape[0];
        let n_features = shape[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if self.method == "box-cox" {
            // Box-Cox requires strictly positive data
            if x_f64.iter().any(|&x| x <= 0.0) {
                return Err(TransformError::InvalidInput(
                    "Box-Cox transformation requires strictly positive data".to_string(),
                ));
            }
        }

        // Compute optimal lambda for each feature in parallel
        let lambdas: Vec<f64> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                let feature = x_f64.column(j).to_vec();
                self.optimize_lambda(&feature)
            })
            .collect();

        self.lambdas_ = Some(Array1::from_vec(lambdas));

        // If standardization is requested, we need to compute means and stds after transformation
        if self.standardize {
            let mut means = Array1::zeros(n_features);
            let mut stds = Array1::zeros(n_features);

            // Transform data with optimal lambdas and compute statistics
            for j in 0..n_features {
                let lambda = self.lambdas_.as_ref().unwrap()[j];
                let mut transformed_feature = Array1::zeros(n_samples);

                // Apply transformation to each sample in the feature
                for i in 0..n_samples {
                    let x = x_f64[[i, j]];
                    transformed_feature[i] = if self.method == "yeo-johnson" {
                        yeo_johnson_transform(x, lambda)
                    } else {
                        box_cox_transform(x, lambda)
                    };
                }

                // Compute mean and standard deviation
                let mean = transformed_feature.sum() / n_samples as f64;
                let variance = transformed_feature
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / n_samples as f64;
                let std_dev = variance.sqrt();

                means[j] = mean;
                stds[j] = if std_dev > EPSILON { std_dev } else { 1.0 };
            }

            self.means_ = Some(means);
            self.stds_ = Some(stds);
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Transforms the input data using the fitted parameters
    ///
    /// # Arguments
    /// * `x` - The input data to transform
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if !self.is_fitted {
            return Err(TransformError::InvalidInput(
                "PowerTransformer must be fitted before transform".to_string(),
            ));
        }

        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        if !x_f64.is_standard_layout() {
            return Err(TransformError::InvalidInput(
                "Input array must be in standard memory layout".to_string(),
            ));
        }

        let shape = x_f64.shape();
        let n_samples = shape[0];
        let n_features = shape[1];

        let lambdas = self.lambdas_.as_ref().unwrap();

        if n_features != lambdas.len() {
            return Err(TransformError::InvalidInput(
                "Number of features in transform data does not match fitted data".to_string(),
            ));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        // Apply transformation in parallel for each feature
        let transformed_data: Vec<Vec<f64>> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                let lambda = lambdas[j];
                // Transform all samples for this feature
                (0..n_samples)
                    .map(|i| {
                        let x = x_f64[[i, j]];
                        if self.method == "yeo-johnson" {
                            yeo_johnson_transform(x, lambda)
                        } else {
                            box_cox_transform(x, lambda)
                        }
                    })
                    .collect()
            })
            .collect();

        // Copy results back to the array
        for j in 0..n_features {
            for i in 0..n_samples {
                transformed[[i, j]] = transformed_data[j][i];
            }
        }

        // Apply standardization if requested
        if self.standardize {
            let means = self.means_.as_ref().unwrap();
            let stds = self.stds_.as_ref().unwrap();

            for j in 0..n_features {
                let mean = means[j];
                let std = stds[j];

                for i in 0..n_samples {
                    transformed[[i, j]] = (transformed[[i, j]] - mean) / std;
                }
            }
        }

        Ok(transformed)
    }

    /// Fits the transformer and transforms the data in one step
    ///
    /// # Arguments
    /// * `x` - The input data to fit and transform
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

    /// Applies the inverse transformation to recover the original data
    ///
    /// # Arguments
    /// * `x` - The transformed data to inverse transform
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The inverse transformed data
    pub fn inverse_transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if !self.is_fitted {
            return Err(TransformError::InvalidInput(
                "PowerTransformer must be fitted before inverse_transform".to_string(),
            ));
        }

        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        if !x_f64.is_standard_layout() {
            return Err(TransformError::InvalidInput(
                "Input array must be in standard memory layout".to_string(),
            ));
        }

        let shape = x_f64.shape();
        let n_samples = shape[0];
        let n_features = shape[1];

        let lambdas = self.lambdas_.as_ref().unwrap();

        if n_features != lambdas.len() {
            return Err(TransformError::InvalidInput(
                "Number of features in inverse transform data does not match fitted data"
                    .to_string(),
            ));
        }

        let mut x_normalized = x_f64.clone();

        // Reverse standardization if it was applied
        if self.standardize {
            let means = self.means_.as_ref().unwrap();
            let stds = self.stds_.as_ref().unwrap();

            for j in 0..n_features {
                let mean = means[j];
                let std = stds[j];

                for i in 0..n_samples {
                    x_normalized[[i, j]] = x_normalized[[i, j]] * std + mean;
                }
            }
        }

        let mut original = Array2::zeros((n_samples, n_features));

        // Apply inverse transformation in parallel for each feature
        let original_data: Vec<Vec<f64>> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                let lambda = lambdas[j];
                // Apply inverse transformation to all samples for this feature
                (0..n_samples)
                    .map(|i| {
                        let y = x_normalized[[i, j]];
                        if self.method == "yeo-johnson" {
                            yeo_johnson_inverse_transform(y, lambda)
                        } else {
                            box_cox_inverse_transform(y, lambda)
                        }
                    })
                    .collect()
            })
            .collect();

        // Copy results back to the array
        for j in 0..n_features {
            for i in 0..n_samples {
                original[[i, j]] = original_data[j][i];
            }
        }

        Ok(original)
    }

    /// Optimizes the lambda parameter for a single feature using maximum likelihood estimation
    ///
    /// # Arguments
    /// * `data` - The feature data to optimize lambda for
    ///
    /// # Returns
    /// * The optimal lambda value
    fn optimize_lambda(&self, data: &[f64]) -> f64 {
        // Use golden section search to find optimal lambda
        let mut a = -2.0;
        let mut b = 2.0;
        let tolerance = 1e-6;
        let golden_ratio = (5.0_f64.sqrt() - 1.0) / 2.0;

        // Golden section search
        let mut c = b - golden_ratio * (b - a);
        let mut d = a + golden_ratio * (b - a);

        while (b - a).abs() > tolerance {
            let fc = -self.log_likelihood(data, c);
            let fd = -self.log_likelihood(data, d);

            if fc < fd {
                b = d;
                d = c;
                c = b - golden_ratio * (b - a);
            } else {
                a = c;
                c = d;
                d = a + golden_ratio * (b - a);
            }
        }

        (a + b) / 2.0
    }

    /// Computes the log-likelihood for a given lambda parameter
    ///
    /// # Arguments
    /// * `data` - The feature data
    /// * `lambda` - The lambda parameter to evaluate
    ///
    /// # Returns
    /// * The log-likelihood value
    fn log_likelihood(&self, data: &[f64], lambda: f64) -> f64 {
        let n = data.len() as f64;
        let mut log_likelihood = 0.0;

        // Transform the data
        let transformed: Vec<f64> = data
            .iter()
            .map(|&x| {
                if self.method == "yeo-johnson" {
                    yeo_johnson_transform(x, lambda)
                } else {
                    box_cox_transform(x, lambda)
                }
            })
            .collect();

        // Compute mean and variance
        let mean = transformed.iter().sum::<f64>() / n;
        let variance = transformed.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

        if variance <= 0.0 {
            return f64::NEG_INFINITY;
        }

        // Log-likelihood of normal distribution
        log_likelihood -= 0.5 * n * (2.0 * std::f64::consts::PI * variance).ln();
        log_likelihood -=
            0.5 * transformed.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / variance;

        // Add Jacobian term
        for &x in data {
            if self.method == "yeo-johnson" {
                log_likelihood += yeo_johnson_log_jacobian(x, lambda);
            } else {
                log_likelihood += box_cox_log_jacobian(x, lambda);
            }
        }

        log_likelihood
    }

    /// Returns the fitted lambda parameters
    pub fn lambdas(&self) -> Option<&Array1<f64>> {
        self.lambdas_.as_ref()
    }

    /// Returns whether the transformer has been fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }
}

/// Apply Yeo-Johnson inverse transformation to a single value
fn yeo_johnson_inverse_transform(y: f64, lambda: f64) -> f64 {
    if y >= 0.0 {
        if (lambda - 0.0).abs() < EPSILON {
            y.exp() - 1.0
        } else {
            (lambda * y + 1.0).powf(1.0 / lambda) - 1.0
        }
    } else if (lambda - 2.0).abs() < EPSILON {
        1.0 - (-y).exp()
    } else {
        1.0 - (-(2.0 - lambda) * y + 1.0).powf(1.0 / (2.0 - lambda))
    }
}

/// Apply Box-Cox inverse transformation to a single value
fn box_cox_inverse_transform(y: f64, lambda: f64) -> f64 {
    if (lambda - 0.0).abs() < EPSILON {
        y.exp()
    } else {
        (lambda * y + 1.0).powf(1.0 / lambda)
    }
}

/// Compute the log of the Jacobian for Yeo-Johnson transformation
fn yeo_johnson_log_jacobian(x: f64, lambda: f64) -> f64 {
    if x >= 0.0 {
        (lambda - 1.0) * (x + 1.0).ln()
    } else {
        (1.0 - lambda) * (-x + 1.0).ln()
    }
}

/// Compute the log of the Jacobian for Box-Cox transformation
fn box_cox_log_jacobian(x: f64, lambda: f64) -> f64 {
    (lambda - 1.0) * x.ln()
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

    #[test]
    fn test_power_transformer_yeo_johnson() {
        // Test data that includes negative values (Yeo-Johnson can handle these)
        let data =
            Array::from_shape_vec((4, 2), vec![1.0, -1.0, 2.0, 0.5, 3.0, -2.0, 0.1, 1.5]).unwrap();

        let mut transformer = PowerTransformer::yeo_johnson(false);

        // Test that transformer is not fitted initially
        assert!(!transformer.is_fitted());

        // Fit the transformer
        transformer.fit(&data).unwrap();
        assert!(transformer.is_fitted());

        // Check that lambdas were computed
        let lambdas = transformer.lambdas().unwrap();
        assert_eq!(lambdas.len(), 2);

        // Transform the data
        let transformed = transformer.transform(&data).unwrap();
        assert_eq!(transformed.shape(), data.shape());

        // Test inverse transformation
        let inverse = transformer.inverse_transform(&transformed).unwrap();
        assert_eq!(inverse.shape(), data.shape());

        // Check that inverse transformation approximately recovers original data
        for i in 0..data.shape()[0] {
            for j in 0..data.shape()[1] {
                assert_abs_diff_eq!(inverse[[i, j]], data[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_power_transformer_box_cox() {
        // Test data with strictly positive values (Box-Cox requirement)
        let data =
            Array::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]).unwrap();

        let mut transformer = PowerTransformer::box_cox(false);

        // Fit the transformer
        transformer.fit(&data).unwrap();

        // Transform the data
        let transformed = transformer.transform(&data).unwrap();
        assert_eq!(transformed.shape(), data.shape());

        // Test inverse transformation
        let inverse = transformer.inverse_transform(&transformed).unwrap();

        // Check that inverse transformation approximately recovers original data
        for i in 0..data.shape()[0] {
            for j in 0..data.shape()[1] {
                assert_abs_diff_eq!(inverse[[i, j]], data[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_power_transformer_standardized() {
        let data = Array::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0],
        )
        .unwrap();

        let mut transformer = PowerTransformer::yeo_johnson(true);

        // Fit and transform with standardization
        let transformed = transformer.fit_transform(&data).unwrap();

        // Check that each feature has approximately zero mean and unit variance
        for j in 0..transformed.shape()[1] {
            let column = transformed.column(j);
            let mean = column.sum() / column.len() as f64;
            let variance =
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64;

            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(variance, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_power_transformer_box_cox_negative_data() {
        // Test that Box-Cox fails with negative data
        let data = Array::from_shape_vec((2, 2), vec![1.0, -1.0, 2.0, 3.0]).unwrap();

        let mut transformer = PowerTransformer::box_cox(false);

        // Should fail because data contains negative values
        assert!(transformer.fit(&data).is_err());
    }

    #[test]
    fn test_power_transformer_fit_transform() {
        let data = Array::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let mut transformer = PowerTransformer::yeo_johnson(false);

        // Test fit_transform convenience method
        let transformed1 = transformer.fit_transform(&data).unwrap();

        // Compare with separate fit and transform
        let mut transformer2 = PowerTransformer::yeo_johnson(false);
        transformer2.fit(&data).unwrap();
        let transformed2 = transformer2.transform(&data).unwrap();

        // Results should be identical
        for i in 0..transformed1.shape()[0] {
            for j in 0..transformed1.shape()[1] {
                assert_abs_diff_eq!(transformed1[[i, j]], transformed2[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_power_transformer_different_data_sizes() {
        let train_data = Array::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let test_data = Array::from_shape_vec((2, 2), vec![2.5, 3.5, 4.5, 5.5]).unwrap();

        let mut transformer = PowerTransformer::yeo_johnson(false);

        // Fit on training data
        transformer.fit(&train_data).unwrap();

        // Transform test data (different number of samples)
        let transformed = transformer.transform(&test_data).unwrap();
        assert_eq!(transformed.shape(), test_data.shape());

        // Test inverse transformation on test data
        let inverse = transformer.inverse_transform(&transformed).unwrap();

        for i in 0..test_data.shape()[0] {
            for j in 0..test_data.shape()[1] {
                assert_abs_diff_eq!(inverse[[i, j]], test_data[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_power_transformer_mismatched_features() {
        let train_data = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let test_data = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let mut transformer = PowerTransformer::yeo_johnson(false);
        transformer.fit(&train_data).unwrap();

        // Should fail because number of features doesn't match
        assert!(transformer.transform(&test_data).is_err());
    }

    #[test]
    fn test_power_transformer_not_fitted() {
        let data = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let transformer = PowerTransformer::yeo_johnson(false);

        // Should fail because transformer hasn't been fitted
        assert!(transformer.transform(&data).is_err());
        assert!(transformer.inverse_transform(&data).is_err());
    }

    #[test]
    fn test_power_transformer_creation_methods() {
        // Test different creation methods
        let transformer1 = PowerTransformer::new("yeo-johnson", true).unwrap();
        let transformer2 = PowerTransformer::yeo_johnson(true);

        // Should be equivalent
        assert_eq!(transformer1.method, transformer2.method);
        assert_eq!(transformer1.standardize, transformer2.standardize);

        let transformer3 = PowerTransformer::new("box-cox", false).unwrap();
        let transformer4 = PowerTransformer::box_cox(false);

        assert_eq!(transformer3.method, transformer4.method);
        assert_eq!(transformer3.standardize, transformer4.standardize);

        // Test invalid method
        assert!(PowerTransformer::new("invalid", false).is_err());
    }

    #[test]
    fn test_power_transformer_empty_data() {
        let empty_data = Array2::<f64>::zeros((0, 2));
        let mut transformer = PowerTransformer::yeo_johnson(false);

        // Should fail with empty data
        assert!(transformer.fit(&empty_data).is_err());
    }

    #[test]
    fn test_yeo_johnson_inverse_functions() {
        let test_values = vec![-2.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let lambdas = vec![0.0, 0.5, 1.0, 1.5, 2.0];

        for &lambda in &lambdas {
            for &x in &test_values {
                let y = yeo_johnson_transform(x, lambda);
                let x_recovered = yeo_johnson_inverse_transform(y, lambda);
                assert_abs_diff_eq!(x_recovered, x, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_box_cox_inverse_functions() {
        let test_values = vec![0.1, 0.5, 1.0, 2.0, 5.0]; // Positive values only
        let lambdas = vec![0.0, 0.5, 1.0, 1.5, 2.0];

        for &lambda in &lambdas {
            for &x in &test_values {
                let y = box_cox_transform(x, lambda);
                let x_recovered = box_cox_inverse_transform(y, lambda);
                assert_abs_diff_eq!(x_recovered, x, epsilon = 1e-10);
            }
        }
    }
}
