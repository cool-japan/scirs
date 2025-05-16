//! Robustness metrics for fairness assessment
//!
//! This module provides metrics to evaluate the robustness of models with respect to fairness,
//! including performance invariance measures, influence functions, and sensitivity
//! to perturbations. These metrics help identify how sensitive a model's fairness
//! properties are to changes in the data or model parameters.

use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2};
use num_traits::real::Real;
use rand::{random, rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::collections::HashMap;

use crate::error::{MetricsError, Result};

/// Measures how invariant model performance is across different protected groups.
///
/// Performance invariance assesses whether a model's performance metrics remain consistent
/// across different demographic groups, which is a key aspect of fairness. This function
/// calculates the variance or dispersion of performance metrics across specified groups.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels or scores
/// * `protected_groups` - A matrix where each column represents membership in a different protected group
/// * `group_names` - Names of the protected groups (corresponding to columns in `protected_groups`)
/// * `metric_fn` - Performance metric function to use (e.g., accuracy, F1 score)
///
/// # Returns
///
/// * A HashMap mapping group names to their performance metrics and a summary invariance score
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use std::collections::HashMap;
/// use scirs2_metrics::fairness::robustness::performance_invariance;
/// use scirs2_metrics::classification::accuracy_score;
///
/// // Sample data
/// let y_true = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
/// let y_pred = array![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
///
/// // Protected group memberships (gender and age)
/// let protected_groups = Array2::from_shape_vec((8, 2), vec![
///     // Column 1: Gender (0=male, 1=female)
///     // Column 2: Age group (0=young, 1=old)
///     0.0, 0.0,
///     0.0, 1.0,
///     0.0, 0.0,
///     0.0, 1.0,
///     1.0, 0.0,
///     1.0, 1.0,
///     1.0, 0.0,
///     1.0, 1.0,
/// ]).unwrap();
///
/// let group_names = vec!["gender".to_string(), "age".to_string()];
///
/// // Calculate performance invariance using accuracy
/// let result = performance_invariance(
///     &y_true,
///     &y_pred,
///     &protected_groups,
///     &group_names,
///     |yt, yp| {
///         // Convert to arrays for accuracy_score
///         let yt_array = ndarray::Array::from_vec(yt.to_vec());
///         let yp_array = ndarray::Array::from_vec(yp.to_vec());
///         accuracy_score(&yt_array, &yp_array).unwrap_or(0.0)
///     }
/// ).unwrap();
///
/// // Examine results
/// println!("Overall invariance score: {}", result.invariance_score);
/// for (group, metrics) in &result.group_metrics {
///     println!("Group {}: performance = {}", group, metrics);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct InvarianceResult {
    /// Performance metrics for each group (group_name -> performance)
    pub group_metrics: HashMap<String, f64>,
    /// Overall invariance score (lower is better, 0 is perfect invariance)
    pub invariance_score: f64,
}

/// Measures how invariant model performance is across different protected groups.
///
/// Performance invariance assesses whether a model's performance metrics remain consistent
/// across different demographic groups, which is a key aspect of fairness. This function
/// calculates the variance or dispersion of performance metrics across specified groups.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels or scores
/// * `protected_groups` - A matrix where each column represents membership in a different protected group
/// * `group_names` - Names of the protected groups (corresponding to columns in `protected_groups`)
/// * `metric_fn` - Performance metric function to use (e.g., accuracy, F1 score)
///
/// # Returns
///
/// * A result containing group-specific performance metrics and an overall invariance score
pub fn performance_invariance<T, S1, S2, S3, F>(
    y_true: &ArrayBase<S1, Ix1>,
    y_pred: &ArrayBase<S2, Ix1>,
    protected_groups: &ArrayBase<S3, Ix2>,
    group_names: &[String],
    metric_fn: F,
) -> Result<InvarianceResult>
where
    T: Real + PartialOrd + Clone,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    S3: Data<Elem = T>,
    F: Fn(&[T], &[T]) -> f64,
{
    // Check dimensions
    let n_samples = y_true.len();
    if n_samples != y_pred.len() || n_samples != protected_groups.nrows() {
        return Err(MetricsError::InvalidInput(format!(
            "Dimensions mismatch: y_true ({}), y_pred ({}), protected_groups ({} rows)",
            n_samples,
            y_pred.len(),
            protected_groups.nrows()
        )));
    }

    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let n_groups = protected_groups.ncols();
    if n_groups != group_names.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Number of group columns ({}) doesn't match number of group names ({})",
            n_groups,
            group_names.len()
        )));
    }

    let y_true_vec: Vec<T> = y_true.iter().cloned().collect();
    let y_pred_vec: Vec<T> = y_pred.iter().cloned().collect();

    // Calculate overall performance
    let overall_performance = metric_fn(&y_true_vec, &y_pred_vec);

    let mut group_metrics = HashMap::new();
    group_metrics.insert("overall".to_string(), overall_performance);

    // For each protected group attribute
    for (group_idx, group_name) in group_names.iter().enumerate() {
        // Find unique values for this group attribute
        let mut unique_values = std::collections::BTreeSet::new();
        for i in 0..n_samples {
            let value = protected_groups[[i, group_idx]].to_f64().unwrap();
            let rounded_value = (value * 1000.0).round() as i64;
            unique_values.insert(rounded_value);
        }

        // Calculate performance for each value of this attribute
        for &int_value in &unique_values {
            let value = int_value as f64 / 1000.0;

            // Filter data for this group
            let mut group_y_true = Vec::new();
            let mut group_y_pred = Vec::new();

            for i in 0..n_samples {
                let group_value = protected_groups[[i, group_idx]].to_f64().unwrap();
                let rounded_value = (group_value * 1000.0).round() as i64;

                if rounded_value == int_value {
                    group_y_true.push(y_true[i].clone());
                    group_y_pred.push(y_pred[i].clone());
                }
            }

            // Skip if empty group
            if group_y_true.is_empty() {
                continue;
            }

            // Calculate performance for this group
            let group_performance = metric_fn(&group_y_true, &group_y_pred);
            group_metrics.insert(format!("{}={}", group_name, value), group_performance);
        }
    }

    // Calculate invariance score (standard deviation of performances)
    let performances: Vec<f64> = group_metrics.values().copied().collect();
    let mean_performance = performances.iter().sum::<f64>() / performances.len() as f64;

    let variance = performances
        .iter()
        .map(|&p| (p - mean_performance).powi(2))
        .sum::<f64>()
        / performances.len() as f64;

    let invariance_score = variance.sqrt(); // Standard deviation as the invariance score

    Ok(InvarianceResult {
        group_metrics,
        invariance_score,
    })
}

/// Calculates the influence of individual samples on model fairness.
///
/// Influence functions approximate how removing each training point would affect
/// model performance and fairness metrics. This is a computationally efficient way
/// to identify which data points have the most impact on a model's fairness properties.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels or scores
/// * `protected_group` - Binary indicator of protected group membership (1 for protected, 0 otherwise)
/// * `fairness_metric` - Function to compute fairness metric (e.g., demographic parity)
/// * `n_samples` - Number of samples to use for approximation (default: all samples)
///
/// # Returns
///
/// * Vector of influence scores for each sample (higher absolute values indicate more influence)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::fairness::robustness::influence_function;
/// use scirs2_metrics::fairness::demographic_parity_difference;
///
/// // Sample data
/// let y_true = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
/// let y_pred = array![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
/// let protected_group = array![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
///
/// // Calculate influence scores using demographic parity
/// let influence_scores = influence_function(
///     &y_true,
///     &y_pred,
///     &protected_group,
///     |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
///     None
/// ).unwrap();
///
/// // Higher absolute scores indicate samples with more influence on fairness
/// for (i, score) in influence_scores.iter().enumerate() {
///     println!("Sample {}: influence = {:.4}", i, score);
/// }
/// ```
pub fn influence_function<T, F>(
    y_true: &Array1<T>,
    y_pred: &Array1<T>,
    protected_group: &Array1<T>,
    fairness_metric: F,
    n_samples: Option<usize>,
) -> Result<Vec<f64>>
where
    T: Real + PartialOrd + Clone,
    F: Fn(&Array1<T>, &Array1<T>) -> f64,
{
    // Check dimensions
    let n_total_samples = y_true.len();
    if n_total_samples != y_pred.len() || n_total_samples != protected_group.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Dimensions mismatch: y_true ({}), y_pred ({}), protected_group ({})",
            n_total_samples,
            y_pred.len(),
            protected_group.len()
        )));
    }

    if n_total_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Determine number of samples to use
    let n_samples_to_use = n_samples.unwrap_or(n_total_samples);
    if n_samples_to_use > n_total_samples {
        return Err(MetricsError::InvalidInput(format!(
            "Requested number of samples ({}) exceeds available samples ({})",
            n_samples_to_use, n_total_samples
        )));
    }

    // Calculate baseline fairness metric
    let baseline_fairness = fairness_metric(y_pred, protected_group);

    // Calculate influence for each sample (leave-one-out approximation)
    let mut influence_scores = vec![0.0; n_total_samples];

    for i in 0..n_samples_to_use {
        // Create a modified y_pred with this sample removed
        let mut y_pred_modified = Vec::with_capacity(n_total_samples - 1);
        let mut protected_group_modified = Vec::with_capacity(n_total_samples - 1);

        for j in 0..n_total_samples {
            if j != i {
                y_pred_modified.push(y_pred[j].clone());
                protected_group_modified.push(protected_group[j].clone());
            }
        }

        // Create modified arrays
        let y_pred_temp = Array1::from_vec(y_pred_modified);
        let protected_group_temp = Array1::from_vec(protected_group_modified);

        // Calculate fairness metric without this sample
        let modified_fairness = fairness_metric(&y_pred_temp, &protected_group_temp);

        // Influence is the difference removing this sample makes
        influence_scores[i] = baseline_fairness - modified_fairness;
    }

    Ok(influence_scores)
}

/// Evaluates model fairness sensitivity to data perturbations.
///
/// This function measures how robust a model's fairness properties are to
/// perturbations in the input data, by adding controlled noise or
/// transformations and observing the impact on fairness metrics.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels or scores
/// * `protected_group` - Binary indicator of protected group membership (1 for protected, 0 otherwise)
/// * `perturbation_type` - Type of perturbation to apply ("label_flip", "subsample", or "noise")
/// * `fairness_metric` - Function to compute fairness metric
/// * `perturbation_level` - Intensity of perturbation (interpretation depends on perturbation_type)
/// * `n_iterations` - Number of times to repeat perturbation for statistical validity
/// * `random_seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Information about fairness metric sensitivity to perturbations
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::fairness::robustness::{
///     perturbation_sensitivity, PerturbationType, SensitivityResult
/// };
/// use scirs2_metrics::fairness::demographic_parity_difference;
///
/// // Sample data
/// let y_true = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
/// let y_pred = array![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
/// let protected_group = array![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
///
/// // Evaluate sensitivity to label flips
/// let result = perturbation_sensitivity(
///     &y_true,
///     &y_pred,
///     &protected_group,
///     PerturbationType::LabelFlip,
///     |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
///     0.1, // 10% of labels will be flipped
///     10,  // Repeat 10 times for statistical significance
///     Some(42)
/// ).unwrap();
///
/// println!("Original fairness: {:.4}", result.original_fairness);
/// println!("Mean perturbed fairness: {:.4}", result.mean_fairness);
/// println!("Standard deviation: {:.4}", result.std_deviation);
/// println!("Sensitivity score: {:.4}", result.sensitivity_score);
/// ```
#[derive(Debug, Clone, Copy)]
pub enum PerturbationType {
    /// Randomly flip prediction labels
    LabelFlip,
    /// Randomly subsample data
    Subsample,
    /// Add Gaussian noise to predictions
    Noise,
}

#[derive(Debug, Clone)]
/// Result of sensitivity analysis for fairness metrics
///
/// This struct contains detailed information about the sensitivity of
/// fairness metrics to data perturbations, including both summary statistics
/// and the full distribution of perturbed fairness values.
pub struct SensitivityResult {
    /// Original fairness metric value
    pub original_fairness: f64,
    /// Mean fairness metric across perturbation iterations
    pub mean_fairness: f64,
    /// Standard deviation of fairness metric across iterations
    pub std_deviation: f64,
    /// Sensitivity score (higher means more sensitive to perturbations)
    pub sensitivity_score: f64,
    /// Perturbed fairness values for each iteration
    pub perturbed_values: Vec<f64>,
    /// Type of perturbation applied
    pub perturbation_type: String,
    /// Level of perturbation applied
    pub perturbation_level: f64,
}

/// Evaluates model fairness sensitivity to data perturbations.
///
/// This function measures how robust a model's fairness properties are to
/// perturbations in the input data, by adding controlled noise or
/// transformations and observing the impact on fairness metrics.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels or scores
/// * `protected_group` - Binary indicator of protected group membership (1 for protected, 0 otherwise)
/// * `perturbation_type` - Type of perturbation to apply (LabelFlip, Subsample, or Noise)
/// * `fairness_metric` - Function to compute fairness metric
/// * `perturbation_level` - Intensity of perturbation (interpretation depends on perturbation_type)
/// * `n_iterations` - Number of times to repeat perturbation for statistical validity
/// * `random_seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Information about fairness metric sensitivity to perturbations
#[allow(clippy::too_many_arguments)]
pub fn perturbation_sensitivity<T, F>(
    y_true: &Array1<T>,
    y_pred: &Array1<T>,
    protected_group: &Array1<T>,
    perturbation_type: PerturbationType,
    fairness_metric: F,
    perturbation_level: f64,
    n_iterations: usize,
    random_seed: Option<u64>,
) -> Result<SensitivityResult>
where
    T: Real + PartialOrd + Clone,
    F: Fn(&Array1<T>, &Array1<T>) -> f64,
{
    // Check dimensions
    let n_samples = y_true.len();
    if n_samples != y_pred.len() || n_samples != protected_group.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Dimensions mismatch: y_true ({}), y_pred ({}), protected_group ({})",
            n_samples,
            y_pred.len(),
            protected_group.len()
        )));
    }

    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Validate perturbation level
    if perturbation_level <= 0.0 || perturbation_level >= 1.0 {
        return Err(MetricsError::InvalidInput(
            "Perturbation level must be between 0 and 1 exclusive".to_string(),
        ));
    }

    if n_iterations == 0 {
        return Err(MetricsError::InvalidInput(
            "Number of iterations must be positive".to_string(),
        ));
    }

    // Initialize random number generator
    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::seed_from_u64(random()),
    };

    // Calculate original fairness
    let original_fairness = fairness_metric(y_pred, protected_group);

    // Collect perturbed fairness values
    let mut perturbed_values = Vec::with_capacity(n_iterations);

    for _ in 0..n_iterations {
        // Create perturbed predictions based on the specified type
        let perturbed_y_pred = match perturbation_type {
            PerturbationType::LabelFlip => {
                perturb_by_label_flip(y_pred, perturbation_level, &mut rng)?
            }
            PerturbationType::Subsample => {
                perturb_by_subsample(y_pred, protected_group, perturbation_level, &mut rng)?
            }
            PerturbationType::Noise => perturb_by_noise(y_pred, perturbation_level, &mut rng)?,
        };

        // Calculate fairness with perturbed data
        let perturbed_fairness = fairness_metric(&perturbed_y_pred, protected_group);
        perturbed_values.push(perturbed_fairness);
    }

    // Calculate statistics
    let mean_fairness = perturbed_values.iter().sum::<f64>() / n_iterations as f64;

    let variance = perturbed_values
        .iter()
        .map(|&v| (v - mean_fairness).powi(2))
        .sum::<f64>()
        / n_iterations as f64;

    let std_deviation = variance.sqrt();

    // Sensitivity score is the ratio of standard deviation to perturbation level
    let sensitivity_score = std_deviation / perturbation_level;

    Ok(SensitivityResult {
        original_fairness,
        mean_fairness,
        std_deviation,
        sensitivity_score,
        perturbed_values,
        perturbation_type: format!("{:?}", perturbation_type),
        perturbation_level,
    })
}

// Helper function to perturb by flipping labels randomly
fn perturb_by_label_flip<T>(
    y_pred: &Array1<T>,
    flip_prob: f64,
    rng: &mut StdRng,
) -> Result<Array1<T>>
where
    T: Real + PartialOrd + Clone,
{
    let n_samples = y_pred.len();
    let n_flips = (n_samples as f64 * flip_prob).round() as usize;

    // Convert to vector for easier manipulation
    let (mut perturbed, _) = y_pred.to_owned().into_raw_vec_and_offset();

    // Create indices to flip
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(rng);
    let flip_indices = &indices[0..n_flips];

    // Flip labels (assuming binary 0/1 labels)
    let zero = T::zero();
    let one = T::one();

    for &idx in flip_indices {
        if perturbed[idx] > zero {
            perturbed[idx] = zero.clone();
        } else {
            perturbed[idx] = one.clone();
        }
    }

    // Convert back to Array1
    Ok(Array1::from_vec(perturbed))
}

// Helper function to perturb by subsampling
fn perturb_by_subsample<T>(
    y_pred: &Array1<T>,
    protected_group: &Array1<T>,
    sample_fraction: f64,
    rng: &mut StdRng,
) -> Result<Array1<T>>
where
    T: Real + PartialOrd + Clone,
{
    let n_samples = y_pred.len();
    let n_subsample = (n_samples as f64 * sample_fraction).round() as usize;

    // Convert to vectors
    let (y_pred_vec, _) = y_pred.to_owned().into_raw_vec_and_offset();
    let (protected_vec, _) = protected_group.to_owned().into_raw_vec_and_offset();

    // Create sample of indices
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(rng);
    let subsample_indices = &indices[0..n_subsample];

    // Create new array using the subsample
    let mut perturbed = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        if subsample_indices.contains(&i) {
            // Keep original prediction
            perturbed.push(y_pred_vec[i].clone());
        } else {
            // For non-sampled points, use the average prediction from their group
            let group_val = &protected_vec[i];
            let zero = T::zero();
            let is_protected = group_val > &zero;

            // Find average prediction for the group
            let mut group_sum = T::zero();
            let mut group_count = 0;

            for j in 0..n_samples {
                let j_group_val = &protected_vec[j];
                let j_is_protected = j_group_val > &zero;

                if j_is_protected == is_protected {
                    group_sum = group_sum + y_pred_vec[j].clone();
                    group_count += 1;
                }
            }

            if group_count > 0 {
                // Use group average
                let group_avg = group_sum / T::from(group_count).unwrap();
                perturbed.push(group_avg);
            } else {
                // Fallback to original value
                perturbed.push(y_pred_vec[i].clone());
            }
        }
    }

    // Convert back to Array1
    Ok(Array1::from_vec(perturbed))
}

// Helper function to perturb by adding noise
fn perturb_by_noise<T>(
    y_pred: &Array1<T>,
    noise_level: f64,
    _rng: &mut StdRng, // Prefix with underscore since we're not using it directly
) -> Result<Array1<T>>
where
    T: Real + PartialOrd + Clone,
{
    let n_samples = y_pred.len();

    // Get the values from the array
    let (y_pred_vec, _) = y_pred.to_owned().into_raw_vec_and_offset();
    let mut perturbed = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let y_val = y_pred_vec[i].to_f64().unwrap();

        // Generate Gaussian noise manually without using the problematic Distribution trait
        // Box-Muller transform
        let u1: f64 = random();
        let u2: f64 = random();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        // Scale by noise_level
        let noise = z0 * noise_level;
        let mut perturbed_val = y_val + noise;

        // Clamp for binary predictions
        if (y_val == 0.0 || y_val == 1.0) && !(0.0..=1.0).contains(&perturbed_val) {
            perturbed_val = perturbed_val.clamp(0.0, 1.0);
        }

        perturbed.push(T::from(perturbed_val).unwrap());
    }

    // Convert back to Array1
    Ok(Array1::from_vec(perturbed))
}
