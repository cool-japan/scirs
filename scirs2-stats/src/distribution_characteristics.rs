//! Distribution characteristic statistics
//!
//! This module provides functions for analyzing characteristics of data distributions,
//! including modes, entropy measures, and confidence intervals for skewness and kurtosis.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::Sum;

/// Mode calculation method to use
pub enum ModeMethod {
    /// Returns the most common value (unimodal case)
    Unimodal,
    /// Returns all local maxima in the probability mass function
    MultiModal,
}

/// Structure to represent the mode(s) of a distribution
#[derive(Debug, Clone)]
pub struct Mode<T>
where
    T: Copy + Debug,
{
    /// The mode values
    pub values: Vec<T>,
    /// The frequency (count) of each mode
    pub counts: Vec<usize>,
}

/// Find the mode(s) of a data set.
///
/// # Arguments
///
/// * `x` - Input data
/// * `method` - The mode calculation method to use
///
/// # Returns
///
/// * Mode structure containing the mode value(s) and their frequencies
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::distribution_characteristics::{mode, ModeMethod};
///
/// // Unimodal data
/// let data = array![1, 2, 2, 3, 2, 4, 5];
/// let unimodal_result = mode(&data.view(), ModeMethod::Unimodal).unwrap();
/// assert_eq!(unimodal_result.values, vec![2]);
/// assert_eq!(unimodal_result.counts, vec![3]);
///
/// // Multimodal data
/// let multidata = array![1, 2, 2, 3, 3, 4];
/// let multimodal_result = mode(&multidata.view(), ModeMethod::MultiModal).unwrap();
/// assert_eq!(multimodal_result.values, vec![2, 3]);
/// assert_eq!(multimodal_result.counts, vec![2, 2]);
/// ```
#[allow(dead_code)]
pub fn mode<T>(x: &ArrayView1<T>, method: ModeMethod) -> StatsResult<Mode<T>>
where
    T: Copy + Eq + Hash + Debug + Ord,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    // Count the occurrences of each value
    let mut counts: HashMap<T, usize> = HashMap::new();
    for &value in x.iter() {
        *counts.entry(value).or_insert(0) += 1;
    }

    // Find the maximum count
    let max_count = counts.values().cloned().max().unwrap_or(0);

    match method {
        ModeMethod::Unimodal => {
            // Find the single value with the highest count
            let mode_value = counts
                .iter()
                .filter(|(_, &count)| count == max_count)
                .map(|(&value_, _)| value_)
                .min()
                .ok_or_else(|| StatsError::InvalidArgument("Failed to compute mode".to_string()))?;

            Ok(Mode {
                values: vec![mode_value],
                counts: vec![max_count],
            })
        }
        ModeMethod::MultiModal => {
            // Find all values with the highest count
            let mut mode_values: Vec<T> = counts
                .iter()
                .filter(|(_, &count)| count == max_count)
                .map(|(&value_, _)| value_)
                .collect();

            // Sort the mode values for consistent output
            mode_values.sort();

            let mode_counts = vec![max_count; mode_values.len()];

            Ok(Mode {
                values: mode_values,
                counts: mode_counts,
            })
        }
    }
}

/// Calculate the Shannon entropy of a data set.
///
/// # Arguments
///
/// * `x` - Input data
/// * `base` - The logarithm base to use (default: e for natural logarithm)
///
/// # Returns
///
/// * The entropy value in the specified base
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::distribution_characteristics::entropy;
///
/// // Uniform distribution (maximum entropy)
/// let uniform = array![1, 2, 3, 4, 5, 6];
/// let entropy_uniform = entropy(&uniform.view(), Some(2.0)).unwrap();
/// assert!((entropy_uniform - 2.58496).abs() < 1e-5);
///
/// // Less uniform distribution (lower entropy)
/// let less_uniform = array![1, 1, 1, 2, 3, 4];
/// let entropy_less = entropy(&less_uniform.view(), Some(2.0)).unwrap();
/// assert!(entropy_less < entropy_uniform);
/// ```
#[allow(dead_code)]
pub fn entropy<T>(x: &ArrayView1<T>, base: Option<f64>) -> StatsResult<f64>
where
    T: Eq + Hash + Copy,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    let base_val = base.unwrap_or(std::f64::consts::E);
    if base_val <= 0.0 {
        return Err(StatsError::InvalidArgument(
            "Base must be positive".to_string(),
        ));
    }

    // Count occurrences
    let mut counts: HashMap<T, usize> = HashMap::new();
    for &value in x.iter() {
        *counts.entry(value).or_insert(0) += 1;
    }

    let n = x.len() as f64;

    // Calculate entropy: -sum(p_i * log(p_i))
    let mut entropy = 0.0;
    for (_, &count) in counts.iter() {
        let p = count as f64 / n;
        if p > 0.0 {
            entropy -= p * p.log(base_val);
        }
    }

    Ok(entropy)
}

/// Calculate the Kullback-Leibler divergence between two probability distributions.
///
/// # Arguments
///
/// * `p` - First probability distribution (reference)
/// * `q` - Second probability distribution (approximation)
///
/// # Returns
///
/// * The KL divergence from q to p
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::distribution_characteristics::kl_divergence;
///
/// // Create two probability distributions
/// let p = array![0.5f64, 0.5];
/// let q = array![0.9f64, 0.1];
///
/// let div = kl_divergence(&p.view(), &q.view()).unwrap();
/// assert!((div - 0.5108256238).abs() < 1e-10);
///
/// // Identical distributions have zero divergence
/// let same = kl_divergence(&p.view(), &p.view()).unwrap();
/// assert!(same == 0.0);
/// ```
#[allow(dead_code)]
pub fn kl_divergence<F>(p: &ArrayView1<F>, q: &ArrayView1<F>) -> StatsResult<F>
where
    F: Float + std::fmt::Debug + Sum + std::fmt::Display,
{
    if p.is_empty() || q.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    if p.len() != q.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "Distributions must have same length, got p({}) and q({})",
            p.len(),
            q.len()
        )));
    }

    // Verify p and q are valid probability distributions
    let p_sum: F = p.iter().cloned().sum();
    let q_sum: F = q.iter().cloned().sum();

    let one = F::one();
    let tol = F::from(1e-10).unwrap();

    if (p_sum - one).abs() > tol || (q_sum - one).abs() > tol {
        return Err(StatsError::InvalidArgument(
            "Inputs must be valid probability distributions that sum to 1".to_string(),
        ));
    }

    // Calculate KL divergence: sum(p_i * log(p_i / q_i))
    let mut divergence = F::zero();

    for (p_i, q_i) in p.iter().zip(q.iter()) {
        if *p_i < F::zero() || *q_i < F::zero() {
            return Err(StatsError::InvalidArgument(
                "Probability values must be non-negative".to_string(),
            ));
        }

        if *p_i > F::zero() {
            if *q_i <= F::zero() {
                return Err(StatsError::DomainError(
                    "KL divergence undefined when q_i = 0 and p_i > 0".to_string(),
                ));
            }

            let ratio = *p_i / *q_i;
            divergence = divergence + *p_i * ratio.ln();
        }
    }

    Ok(divergence)
}

/// Calculate the cross-entropy between two probability distributions.
///
/// # Arguments
///
/// * `p` - First probability distribution (reference)
/// * `q` - Second probability distribution (approximation)
///
/// # Returns
///
/// * The cross-entropy between p and q
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::distribution_characteristics::cross_entropy;
///
/// // Create two probability distributions
/// let p = array![0.5f64, 0.5];
/// let q = array![0.9f64, 0.1];
///
/// let cross_ent = cross_entropy(&p.view(), &q.view()).unwrap();
/// ```
#[allow(dead_code)]
pub fn cross_entropy<F>(p: &ArrayView1<F>, q: &ArrayView1<F>) -> StatsResult<F>
where
    F: Float + std::fmt::Debug + Sum + std::fmt::Display,
{
    if p.is_empty() || q.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    if p.len() != q.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "Distributions must have same length, got p({}) and q({})",
            p.len(),
            q.len()
        )));
    }

    // Verify p and q are valid probability distributions
    let p_sum: F = p.iter().cloned().sum();
    let q_sum: F = q.iter().cloned().sum();

    let one = F::one();
    let tol = F::from(1e-10).unwrap();

    if (p_sum - one).abs() > tol || (q_sum - one).abs() > tol {
        return Err(StatsError::InvalidArgument(
            "Inputs must be valid probability distributions that sum to 1".to_string(),
        ));
    }

    // Calculate cross-entropy: -sum(p_i * log(q_i))
    let mut cross_ent = F::zero();

    for (p_i, q_i) in p.iter().zip(q.iter()) {
        if *p_i < F::zero() || *q_i < F::zero() {
            return Err(StatsError::InvalidArgument(
                "Probability values must be non-negative".to_string(),
            ));
        }

        if *p_i > F::zero() {
            if *q_i <= F::zero() {
                return Err(StatsError::DomainError(
                    "Cross-entropy undefined when q_i = 0 and p_i > 0".to_string(),
                ));
            }

            cross_ent = cross_ent - *p_i * q_i.ln();
        }
    }

    Ok(cross_ent)
}

/// Structure to hold confidence interval information
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceInterval<F>
where
    F: Float + std::fmt::Display,
{
    /// The estimated statistic value
    pub estimate: F,
    /// The lower bound of the confidence interval
    pub lower: F,
    /// The upper bound of the confidence interval
    pub upper: F,
    /// The confidence level (e.g., 0.95 for 95% confidence)
    pub confidence: F,
}

/// Calculate the skewness with confidence interval using bootstrap method.
///
/// # Arguments
///
/// * `x` - Input data
/// * `bias` - Whether to use the biased estimator
/// * `confidence` - Confidence level (default: 0.95)
/// * `n_bootstrap` - Number of bootstrap samples (default: 1000)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
///
/// * A ConfidenceInterval structure containing the estimate and confidence bounds
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::distribution_characteristics::skewness_ci;
///
/// // Calculate skewness with 95% confidence interval
/// let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 10.0];
/// let result = skewness_ci(&data.view(), false, None, None, Some(42)).unwrap();
/// println!("Skewness: {} (95% CI: {}, {})", result.estimate, result.lower, result.upper);
/// ```
#[allow(dead_code)]
pub fn skewness_ci<F>(
    x: &ArrayView1<F>,
    bias: bool,
    confidence: Option<F>,
    n_bootstrap: Option<usize>,
    seed: Option<u64>,
) -> StatsResult<ConfidenceInterval<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Debug
        + std::fmt::Display
        + Send
        + Sync
        + scirs2_core::simd_ops::SimdUnifiedOps,
{
    use crate::sampling::bootstrap;
    use crate::skew;

    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    if x.len() < 3 {
        return Err(StatsError::DomainError(
            "At least 3 data points required to calculate skewness".to_string(),
        ));
    }

    let conf = confidence.unwrap_or(F::from(0.95).unwrap());
    let n_boot = n_bootstrap.unwrap_or(1000);

    if conf <= F::zero() || conf >= F::one() {
        return Err(StatsError::InvalidArgument(
            "Confidence level must be between 0 and 1 exclusive".to_string(),
        ));
    }

    // Calculate point estimate
    let estimate = skew(x, bias, None)?;

    // Generate _bootstrap samples
    let samples = bootstrap(x, n_boot, seed)?;

    // Calculate skewness for each _bootstrap sample
    let mut bootstrap_skew = Vec::with_capacity(n_boot);

    for i in 0..n_boot {
        let sample_view = samples.slice(scirs2_core::ndarray::s![i, ..]).to_owned();
        if let Ok(sk) = skew(&sample_view.view(), bias, None) {
            bootstrap_skew.push(sk);
        }
    }

    // Sort _bootstrap statistics for percentile confidence intervals
    bootstrap_skew.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate percentile indices
    let alpha = (F::one() - conf) / (F::one() + F::one());
    let lower_idx = (alpha * F::from(bootstrap_skew.len()).unwrap())
        .to_usize()
        .unwrap();
    let upper_idx = ((F::one() - alpha) * F::from(bootstrap_skew.len()).unwrap())
        .to_usize()
        .unwrap();

    // Get percentile values
    let lower = bootstrap_skew.get(lower_idx).cloned().unwrap_or(F::zero());
    let upper = bootstrap_skew.get(upper_idx).cloned().unwrap_or(F::zero());

    Ok(ConfidenceInterval {
        estimate,
        lower,
        upper,
        confidence: conf,
    })
}

/// Calculate the kurtosis with confidence interval using bootstrap method.
///
/// # Arguments
///
/// * `x` - Input data
/// * `fisher` - Whether to use Fisher's definition (excess kurtosis)
/// * `bias` - Whether to use the biased estimator
/// * `confidence` - Confidence level (default: 0.95)
/// * `n_bootstrap` - Number of bootstrap samples (default: 1000)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
///
/// * A ConfidenceInterval structure containing the estimate and confidence bounds
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::distribution_characteristics::kurtosis_ci;
///
/// // Calculate kurtosis with 95% confidence interval
/// let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 10.0];
/// let result = kurtosis_ci(&data.view(), true, false, None, None, Some(42)).unwrap();
/// println!("Kurtosis: {} (95% CI: {}, {})", result.estimate, result.lower, result.upper);
/// ```
#[allow(dead_code)]
pub fn kurtosis_ci<F>(
    x: &ArrayView1<F>,
    fisher: bool,
    bias: bool,
    confidence: Option<F>,
    n_bootstrap: Option<usize>,
    seed: Option<u64>,
) -> StatsResult<ConfidenceInterval<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Debug
        + std::fmt::Display
        + Send
        + Sync
        + scirs2_core::simd_ops::SimdUnifiedOps,
{
    use crate::kurtosis;
    use crate::sampling::bootstrap;

    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    if x.len() < 4 {
        return Err(StatsError::DomainError(
            "At least 4 data points required to calculate kurtosis".to_string(),
        ));
    }

    let conf = confidence.unwrap_or(F::from(0.95).unwrap());
    let n_boot = n_bootstrap.unwrap_or(1000);

    if conf <= F::zero() || conf >= F::one() {
        return Err(StatsError::InvalidArgument(
            "Confidence level must be between 0 and 1 exclusive".to_string(),
        ));
    }

    // Calculate point estimate
    let estimate = kurtosis(x, fisher, bias, None)?;

    // Generate _bootstrap samples
    let samples = bootstrap(x, n_boot, seed)?;

    // Calculate kurtosis for each _bootstrap sample
    let mut bootstrap_kurt = Vec::with_capacity(n_boot);

    for i in 0..n_boot {
        let sample_view = samples.slice(scirs2_core::ndarray::s![i, ..]).to_owned();
        if let Ok(k) = kurtosis(&sample_view.view(), fisher, bias, None) {
            bootstrap_kurt.push(k);
        }
    }

    // Sort _bootstrap statistics for percentile confidence intervals
    bootstrap_kurt.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate percentile indices
    let alpha = (F::one() - conf) / (F::one() + F::one());
    let lower_idx = (alpha * F::from(bootstrap_kurt.len()).unwrap())
        .to_usize()
        .unwrap();
    let upper_idx = ((F::one() - alpha) * F::from(bootstrap_kurt.len()).unwrap())
        .to_usize()
        .unwrap();

    // Get percentile values
    let lower = bootstrap_kurt.get(lower_idx).cloned().unwrap_or(F::zero());
    let upper = bootstrap_kurt.get(upper_idx).cloned().unwrap_or(F::zero());

    Ok(ConfidenceInterval {
        estimate,
        lower,
        upper,
        confidence: conf,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{kurtosis, skew};
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mode_unimodal() {
        let data = array![1, 2, 2, 3, 2, 4, 5];
        let result = mode(&data.view(), ModeMethod::Unimodal).unwrap();
        assert_eq!(result.values.len(), 1);
        assert_eq!(result.values[0], 2);
        assert_eq!(result.counts[0], 3);
    }

    #[test]
    fn test_mode_multimodal() {
        let data = array![1, 2, 2, 3, 3, 4];
        let result = mode(&data.view(), ModeMethod::MultiModal).unwrap();
        assert_eq!(result.values.len(), 2);
        assert_eq!(result.values, vec![2, 3]);
        assert_eq!(result.counts, vec![2, 2]);
    }

    #[test]
    fn test_entropy() {
        // Uniform distribution (maximum entropy)
        let uniform = array![1, 2, 3, 4, 5, 6];
        let entropy_uniform = entropy(&uniform.view(), Some(2.0)).unwrap();
        assert_relative_eq!(entropy_uniform, 2.58496, epsilon = 1e-5);

        // Less uniform distribution (lower entropy)
        let less_uniform = array![1, 1, 1, 2, 3, 4];
        let entropy_less = entropy(&less_uniform.view(), Some(2.0)).unwrap();
        assert!(entropy_less < entropy_uniform);

        // Single value (zero entropy)
        let single = array![1, 1, 1, 1, 1];
        let entropy_single = entropy(&single.view(), Some(2.0)).unwrap();
        assert_relative_eq!(entropy_single, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence() {
        // Create two probability distributions
        let p = array![0.5f64, 0.5];
        let q = array![0.9f64, 0.1];

        let div = kl_divergence(&p.view(), &q.view()).unwrap();
        assert_relative_eq!(div, 0.5108256238, epsilon = 1e-10);

        // KL divergence is not symmetric
        let div_reverse = kl_divergence(&q.view(), &p.view()).unwrap();
        assert!(div != div_reverse);

        // Identical distributions have zero divergence
        let same = kl_divergence(&p.view(), &p.view()).unwrap();
        assert_relative_eq!(same, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cross_entropy() {
        // Create two probability distributions
        let p = array![0.5f64, 0.5];
        let q = array![0.9f64, 0.1];

        let cross_ent = cross_entropy(&p.view(), &q.view()).unwrap();

        // Cross entropy equals entropy(p) + KL(p||q)
        let entropy_p = -0.5f64 * (0.5f64.ln()) - 0.5 * (0.5f64.ln());
        let kl = kl_divergence(&p.view(), &q.view()).unwrap();

        assert_relative_eq!(cross_ent, entropy_p + kl, epsilon = 1e-10);
    }

    #[test]
    fn test_skewness_ci() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 10.0];
        let result = skewness_ci(&data.view(), false, None, Some(100), Some(42)).unwrap();

        // Check that the estimate is correct
        let direct_skew = skew(&data.view(), false, None).unwrap();
        assert_relative_eq!(result.estimate, direct_skew, epsilon = 1e-10);

        // Check confidence interval contains the estimate
        assert!(result.lower <= result.estimate);
        assert!(result.upper >= result.estimate);

        // Check confidence level
        assert_relative_eq!(result.confidence, 0.95, epsilon = 1e-10);
    }

    #[test]
    fn test_kurtosis_ci() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 10.0];
        let result = kurtosis_ci(&data.view(), true, false, None, Some(100), Some(42)).unwrap();

        // Check that the estimate is correct
        let direct_kurt = kurtosis(&data.view(), true, false, None).unwrap();
        assert_relative_eq!(result.estimate, direct_kurt, epsilon = 1e-10);

        // Check confidence interval contains the estimate
        assert!(result.lower <= result.estimate);
        assert!(result.upper >= result.estimate);

        // Check confidence level
        assert_relative_eq!(result.confidence, 0.95, epsilon = 1e-10);
    }
}
