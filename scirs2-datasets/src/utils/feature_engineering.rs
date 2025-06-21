//! Feature engineering utilities for creating and transforming features
//!
//! This module provides various methods for generating new features from existing data,
//! including polynomial features, statistical features, and binned features. These
//! transformations can help improve machine learning model performance by creating
//! more informative representations of the data.

use crate::error::{DatasetsError, Result};
use ndarray::{Array1, Array2};

/// Binning strategies for discretization
#[derive(Debug, Clone, Copy)]
pub enum BinningStrategy {
    /// Uniform-width bins based on min-max range
    Uniform,
    /// Quantile-based bins (equal frequency)
    Quantile,
}

/// Generates polynomial features up to a specified degree
///
/// Creates polynomial combinations of features up to the specified degree.
/// For example, with degree=2 and features [a, b], generates [1, a, b, a², ab, b²].
///
/// # Arguments
///
/// * `data` - Input feature matrix (n_samples, n_features)
/// * `degree` - Maximum polynomial degree (must be >= 1)
/// * `include_bias` - Whether to include the bias column (all ones)
///
/// # Returns
///
/// A new array with polynomial features
///
/// # Examples
///
/// ```rust
/// use ndarray::Array2;
/// use scirs2_datasets::utils::polynomial_features;
///
/// let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let poly_features = polynomial_features(&data, 2, true).unwrap();
/// // Result includes: [1, x1, x2, x1², x1*x2, x2²]
/// ```
pub fn polynomial_features(
    data: &Array2<f64>,
    degree: usize,
    include_bias: bool,
) -> Result<Array2<f64>> {
    if degree == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Polynomial degree must be at least 1".to_string(),
        ));
    }

    let n_samples = data.nrows();
    let n_features = data.ncols();

    // Calculate number of polynomial features
    let mut n_output_features = 0;
    if include_bias {
        n_output_features += 1;
    }

    // Count features for each degree
    for d in 1..=degree {
        // Number of multivariate polynomials of degree d with n_features variables
        // This uses the formula for combinations with repetition: C(n+k-1, k)
        let mut combinations = 1;
        for i in 0..d {
            combinations = combinations * (n_features + i) / (i + 1);
        }
        n_output_features += combinations;
    }

    let mut output = Array2::zeros((n_samples, n_output_features));
    let mut col_idx = 0;

    // Add bias column if requested
    if include_bias {
        output.column_mut(col_idx).fill(1.0);
    }

    // Generate polynomial features
    for sample_idx in 0..n_samples {
        let sample = data.row(sample_idx);
        col_idx = if include_bias { 1 } else { 0 };

        // Degree 1 features (original features)
        for &feature_val in sample.iter() {
            output[[sample_idx, col_idx]] = feature_val;
            col_idx += 1;
        }

        // Higher degree features
        for deg in 2..=degree {
            generate_polynomial_combinations(
                &sample.to_owned(),
                deg,
                sample_idx,
                &mut output,
                &mut col_idx,
            );
        }
    }

    Ok(output)
}

/// Helper function to generate polynomial combinations recursively
fn generate_polynomial_combinations(
    features: &Array1<f64>,
    degree: usize,
    sample_idx: usize,
    output: &mut Array2<f64>,
    col_idx: &mut usize,
) {
    fn combinations_recursive(
        features: &Array1<f64>,
        degree: usize,
        start_idx: usize,
        current_product: f64,
        sample_idx: usize,
        output: &mut Array2<f64>,
        col_idx: &mut usize,
    ) {
        if degree == 0 {
            output[[sample_idx, *col_idx]] = current_product;
            *col_idx += 1;
            return;
        }

        for i in start_idx..features.len() {
            combinations_recursive(
                features,
                degree - 1,
                i, // Allow repetition by using i instead of i+1
                current_product * features[i],
                sample_idx,
                output,
                col_idx,
            );
        }
    }

    combinations_recursive(features, degree, 0, 1.0, sample_idx, output, col_idx);
}

/// Extracts statistical features from the data
///
/// Computes various statistical measures for each feature, including central tendency,
/// dispersion, and shape statistics.
///
/// # Arguments
///
/// * `data` - Input feature matrix (n_samples, n_features)
///
/// # Returns
///
/// A new array with statistical features: [mean, std, min, max, median, q25, q75, skewness, kurtosis] for each original feature
///
/// # Examples
///
/// ```rust
/// use ndarray::Array2;
/// use scirs2_datasets::utils::statistical_features;
///
/// let data = Array2::from_shape_vec((5, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0]).unwrap();
/// let stats_features = statistical_features(&data).unwrap();
/// // Result includes 9 statistical measures for each of the 2 original features
/// ```
pub fn statistical_features(data: &Array2<f64>) -> Result<Array2<f64>> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_samples == 0 || n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Data cannot be empty for statistical feature extraction".to_string(),
        ));
    }

    // 9 statistical features per original feature
    let n_stat_features = 9;
    let mut stats = Array2::zeros((n_samples, n_features * n_stat_features));

    for sample_idx in 0..n_samples {
        for feature_idx in 0..n_features {
            let feature_values = data.column(feature_idx);

            // Calculate basic statistics
            let mean = feature_values.mean().unwrap_or(0.0);
            let std = feature_values.std(0.0);
            let min_val = feature_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = feature_values
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            // Calculate quantiles
            let mut sorted_values: Vec<f64> = feature_values.to_vec();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let median = calculate_quantile(&sorted_values, 0.5);
            let q25 = calculate_quantile(&sorted_values, 0.25);
            let q75 = calculate_quantile(&sorted_values, 0.75);

            // Calculate skewness and kurtosis
            let skewness = calculate_skewness(&feature_values, mean, std);
            let kurtosis = calculate_kurtosis(&feature_values, mean, std);

            // Store statistical features
            let base_idx = feature_idx * n_stat_features;
            stats[[sample_idx, base_idx]] = mean;
            stats[[sample_idx, base_idx + 1]] = std;
            stats[[sample_idx, base_idx + 2]] = min_val;
            stats[[sample_idx, base_idx + 3]] = max_val;
            stats[[sample_idx, base_idx + 4]] = median;
            stats[[sample_idx, base_idx + 5]] = q25;
            stats[[sample_idx, base_idx + 6]] = q75;
            stats[[sample_idx, base_idx + 7]] = skewness;
            stats[[sample_idx, base_idx + 8]] = kurtosis;
        }
    }

    Ok(stats)
}

/// Calculates a specific quantile from sorted data
fn calculate_quantile(sorted_data: &[f64], quantile: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }

    let n = sorted_data.len();
    let index = quantile * (n - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        sorted_data[lower]
    } else {
        let weight = index - lower as f64;
        sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
    }
}

/// Calculates skewness (third moment)
fn calculate_skewness(data: &ndarray::ArrayView1<f64>, mean: f64, std: f64) -> f64 {
    if std <= 1e-10 {
        return 0.0;
    }

    let n = data.len() as f64;
    let sum_cubed_deviations: f64 = data.iter().map(|&x| ((x - mean) / std).powi(3)).sum();

    sum_cubed_deviations / n
}

/// Calculates kurtosis (fourth moment)
fn calculate_kurtosis(data: &ndarray::ArrayView1<f64>, mean: f64, std: f64) -> f64 {
    if std <= 1e-10 {
        return 0.0;
    }

    let n = data.len() as f64;
    let sum_fourth_deviations: f64 = data.iter().map(|&x| ((x - mean) / std).powi(4)).sum();

    (sum_fourth_deviations / n) - 3.0 // Excess kurtosis (subtract 3 for normal distribution)
}

/// Creates binned (discretized) features from continuous features
///
/// Transforms continuous features into categorical features by binning values
/// into specified ranges. This can be useful for creating non-linear features
/// or reducing the impact of outliers.
///
/// # Arguments
///
/// * `data` - Input feature matrix (n_samples, n_features)
/// * `n_bins` - Number of bins per feature
/// * `strategy` - Binning strategy to use
///
/// # Returns
///
/// A new array with binned features (encoded as bin indices)
///
/// # Examples
///
/// ```rust
/// use ndarray::Array2;
/// use scirs2_datasets::utils::{create_binned_features, BinningStrategy};
///
/// let data = Array2::from_shape_vec((5, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0]).unwrap();
/// let binned = create_binned_features(&data, 3, BinningStrategy::Uniform).unwrap();
/// // Each feature is now discretized into 3 bins (values 0, 1, 2)
/// ```
pub fn create_binned_features(
    data: &Array2<f64>,
    n_bins: usize,
    strategy: BinningStrategy,
) -> Result<Array2<f64>> {
    if n_bins < 2 {
        return Err(DatasetsError::InvalidFormat(
            "Number of bins must be at least 2".to_string(),
        ));
    }

    let n_samples = data.nrows();
    let n_features = data.ncols();
    let mut binned = Array2::zeros((n_samples, n_features));

    for j in 0..n_features {
        let column = data.column(j);
        let bin_edges = calculate_bin_edges(&column, n_bins, &strategy)?;

        for i in 0..n_samples {
            let value = column[i];
            let bin_idx = find_bin_index(value, &bin_edges);
            binned[[i, j]] = bin_idx as f64;
        }
    }

    Ok(binned)
}

/// Calculate bin edges based on the specified strategy
fn calculate_bin_edges(
    data: &ndarray::ArrayView1<f64>,
    n_bins: usize,
    strategy: &BinningStrategy,
) -> Result<Vec<f64>> {
    match strategy {
        BinningStrategy::Uniform => {
            let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            if (max_val - min_val).abs() <= 1e-10 {
                return Ok(vec![min_val, min_val + 1e-10]);
            }

            let bin_width = (max_val - min_val) / n_bins as f64;
            let mut edges = Vec::with_capacity(n_bins + 1);

            for i in 0..=n_bins {
                edges.push(min_val + i as f64 * bin_width);
            }

            Ok(edges)
        }
        BinningStrategy::Quantile => {
            let mut sorted_data: Vec<f64> = data.to_vec();
            sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut edges = Vec::with_capacity(n_bins + 1);
            edges.push(sorted_data[0]);

            for i in 1..n_bins {
                let quantile = i as f64 / n_bins as f64;
                let edge = calculate_quantile(&sorted_data, quantile);
                edges.push(edge);
            }

            edges.push(sorted_data[sorted_data.len() - 1]);

            Ok(edges)
        }
    }
}

/// Find the bin index for a given value
fn find_bin_index(value: f64, bin_edges: &[f64]) -> usize {
    for (i, &edge) in bin_edges.iter().enumerate().skip(1) {
        if value <= edge {
            return i - 1;
        }
    }
    bin_edges.len() - 2 // Last bin
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_polynomial_features_degree_2() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let poly = polynomial_features(&data, 2, true).unwrap();

        // Should have: [bias, x1, x2, x1², x1*x2, x2²] = 6 features
        assert_eq!(poly.ncols(), 6);
        assert_eq!(poly.nrows(), 2);

        // Check first sample: [1, 1, 2, 1, 2, 4]
        assert!((poly[[0, 0]] - 1.0).abs() < 1e-10); // bias
        assert!((poly[[0, 1]] - 1.0).abs() < 1e-10); // x1
        assert!((poly[[0, 2]] - 2.0).abs() < 1e-10); // x2
        assert!((poly[[0, 3]] - 1.0).abs() < 1e-10); // x1²
        assert!((poly[[0, 4]] - 2.0).abs() < 1e-10); // x1*x2
        assert!((poly[[0, 5]] - 4.0).abs() < 1e-10); // x2²
    }

    #[test]
    fn test_polynomial_features_no_bias() {
        let data = Array2::from_shape_vec((1, 2), vec![2.0, 3.0]).unwrap();
        let poly = polynomial_features(&data, 2, false).unwrap();

        // Should have: [x1, x2, x1², x1*x2, x2²] = 5 features (no bias)
        assert_eq!(poly.ncols(), 5);

        // Check values: [2, 3, 4, 6, 9]
        assert!((poly[[0, 0]] - 2.0).abs() < 1e-10); // x1
        assert!((poly[[0, 1]] - 3.0).abs() < 1e-10); // x2
        assert!((poly[[0, 2]] - 4.0).abs() < 1e-10); // x1²
        assert!((poly[[0, 3]] - 6.0).abs() < 1e-10); // x1*x2
        assert!((poly[[0, 4]] - 9.0).abs() < 1e-10); // x2²
    }

    #[test]
    fn test_polynomial_features_invalid_degree() {
        let data = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        assert!(polynomial_features(&data, 0, true).is_err());
    }

    #[test]
    fn test_statistical_features() {
        let data = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let stats = statistical_features(&data).unwrap();

        // Should have 9 statistical features for 1 original feature
        assert_eq!(stats.ncols(), 9);
        assert_eq!(stats.nrows(), 5);

        // All samples should have the same statistical features (global statistics)
        for i in 0..stats.nrows() {
            assert!((stats[[i, 0]] - 3.0).abs() < 1e-10); // mean
            assert!((stats[[i, 2]] - 1.0).abs() < 1e-10); // min
            assert!((stats[[i, 3]] - 5.0).abs() < 1e-10); // max
            assert!((stats[[i, 4]] - 3.0).abs() < 1e-10); // median
        }
    }

    #[test]
    fn test_statistical_features_empty_data() {
        let data = Array2::zeros((0, 1));
        assert!(statistical_features(&data).is_err());
    }

    #[test]
    fn test_create_binned_features_uniform() {
        let data = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let binned = create_binned_features(&data, 3, BinningStrategy::Uniform).unwrap();

        assert_eq!(binned.nrows(), 5);
        assert_eq!(binned.ncols(), 1);

        // Check that all values are valid bin indices (0, 1, or 2)
        for i in 0..binned.nrows() {
            let bin_val = binned[[i, 0]] as usize;
            assert!(bin_val < 3);
        }
    }

    #[test]
    fn test_create_binned_features_quantile() {
        let data = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let binned = create_binned_features(&data, 3, BinningStrategy::Quantile).unwrap();

        assert_eq!(binned.nrows(), 6);
        assert_eq!(binned.ncols(), 1);

        // With quantile binning, each bin should have roughly equal number of samples
        let mut bin_counts = vec![0; 3];
        for i in 0..binned.nrows() {
            let bin_val = binned[[i, 0]] as usize;
            bin_counts[bin_val] += 1;
        }

        // Each bin should have 2 samples (6 samples / 3 bins)
        for &count in &bin_counts {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn test_create_binned_features_invalid_bins() {
        let data = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(create_binned_features(&data, 1, BinningStrategy::Uniform).is_err());
        assert!(create_binned_features(&data, 0, BinningStrategy::Uniform).is_err());
    }

    #[test]
    fn test_calculate_quantile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(calculate_quantile(&data, 0.0), 1.0);
        assert_eq!(calculate_quantile(&data, 0.5), 3.0);
        assert_eq!(calculate_quantile(&data, 1.0), 5.0);
        assert_eq!(calculate_quantile(&data, 0.25), 2.0);
        assert_eq!(calculate_quantile(&data, 0.75), 4.0);
    }

    #[test]
    fn test_calculate_skewness() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let view = data.view();
        let mean = view.mean().unwrap();
        let std = view.std(0.0);

        let skewness = calculate_skewness(&view, mean, std);
        // Symmetric distribution should have skewness near 0
        assert!(skewness.abs() < 1e-10);
    }

    #[test]
    fn test_calculate_kurtosis() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let view = data.view();
        let mean = view.mean().unwrap();
        let std = view.std(0.0);

        let kurtosis = calculate_kurtosis(&view, mean, std);
        // Uniform distribution should have negative excess kurtosis
        assert!(kurtosis < 0.0);
    }

    #[test]
    fn test_feature_extraction_pipeline() {
        // Test a complete feature extraction pipeline
        let data = Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
            .unwrap();

        // Step 1: Generate polynomial features
        let poly_data = polynomial_features(&data, 2, false).unwrap();

        // Step 2: Create binned features
        let binned_data = create_binned_features(&poly_data, 2, BinningStrategy::Uniform).unwrap();

        // Step 3: Generate statistical features
        let stats_data = statistical_features(&data).unwrap();

        // Verify pipeline produces expected shapes
        assert_eq!(poly_data.ncols(), 5); // [x1, x2, x1², x1*x2, x2²]
        assert_eq!(binned_data.ncols(), 5); // Same number of features, but binned
        assert_eq!(stats_data.ncols(), 18); // 9 statistics × 2 original features
        assert_eq!(binned_data.nrows(), 4); // Same number of samples
        assert_eq!(stats_data.nrows(), 4); // Same number of samples
    }

    #[test]
    fn test_binning_strategies_comparison() {
        // Create data with outliers
        let data =
            Array2::from_shape_vec((7, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]).unwrap();

        let uniform_binned = create_binned_features(&data, 3, BinningStrategy::Uniform).unwrap();
        let quantile_binned = create_binned_features(&data, 3, BinningStrategy::Quantile).unwrap();

        // With uniform binning, the outlier will dominate the range
        // With quantile binning, each bin should have roughly equal frequency

        // Count bin distributions
        let mut uniform_counts = vec![0; 3];
        let mut quantile_counts = vec![0; 3];

        for i in 0..data.nrows() {
            uniform_counts[uniform_binned[[i, 0]] as usize] += 1;
            quantile_counts[quantile_binned[[i, 0]] as usize] += 1;
        }

        // Quantile binning should have more balanced distribution
        let uniform_max = *uniform_counts.iter().max().unwrap();
        let uniform_min = *uniform_counts.iter().min().unwrap();
        let quantile_max = *quantile_counts.iter().max().unwrap();
        let quantile_min = *quantile_counts.iter().min().unwrap();

        // Quantile binning should be more balanced (smaller difference between max and min)
        assert!((quantile_max - quantile_min) <= (uniform_max - uniform_min));
    }
}
