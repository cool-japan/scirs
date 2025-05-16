//! Numerically stable implementations of metrics
//!
//! This module provides implementations of metrics that are designed to be
//! numerically stable, particularly for edge cases and extreme values.

use ndarray::{ArrayBase, Data, Dimension};
use num_traits::{Float, NumCast};

use crate::error::{MetricsError, Result};

/// Trait for metrics that have numerically stable implementations
pub trait StableMetric<T, D>
where
    T: Float,
    D: Dimension,
{
    /// Compute the metric in a numerically stable way
    fn compute_stable(&self, x: &ArrayBase<impl Data<Elem = T>, D>) -> Result<T>;
}

/// Numerically stable computation of common operations
#[derive(Debug, Clone)]
pub struct StableMetrics<T> {
    /// Minimum value to avoid division by zero or log of zero
    pub epsilon: T,
    /// Maximum value to clip extreme values
    pub max_value: T,
    /// Whether to clip values
    pub clip_values: bool,
    /// Whether to use log-sum-exp trick for numerical stability
    pub use_logsumexp: bool,
}

impl<T: Float + NumCast> Default for StableMetrics<T> {
    fn default() -> Self {
        StableMetrics {
            epsilon: T::from(1e-10).unwrap(),
            max_value: T::from(1e10).unwrap(),
            clip_values: true,
            use_logsumexp: true,
        }
    }
}

impl<T: Float + NumCast> StableMetrics<T> {
    /// Create a new StableMetrics with default settings
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the epsilon value
    pub fn with_epsilon(mut self, epsilon: T) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the maximum value
    pub fn with_max_value(mut self, max_value: T) -> Self {
        self.max_value = max_value;
        self
    }

    /// Set whether to clip values
    pub fn with_clip_values(mut self, clip_values: bool) -> Self {
        self.clip_values = clip_values;
        self
    }

    /// Set whether to use log-sum-exp trick
    pub fn with_logsumexp(mut self, use_logsumexp: bool) -> Self {
        self.use_logsumexp = use_logsumexp;
        self
    }

    /// Numerically stable calculation of the mean
    ///
    /// Implements Welford's online algorithm for computing the mean,
    /// which is more stable for large datasets.
    ///
    /// # Arguments
    ///
    /// * `values` - Input array of values
    ///
    /// # Returns
    ///
    /// * The mean of the values
    pub fn stable_mean(&self, values: &[T]) -> Result<T> {
        if values.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Cannot compute mean of empty array".to_string(),
            ));
        }

        let mut mean = T::zero();
        let mut count = T::zero();

        for &value in values {
            count = count + T::one();
            // Using Welford's online algorithm
            let delta = value - mean;
            mean = mean + delta / count;
        }

        Ok(mean)
    }

    /// Numerically stable calculation of variance
    ///
    /// Implements Welford's online algorithm for computing variance,
    /// which is more stable for large datasets.
    ///
    /// # Arguments
    ///
    /// * `values` - Input array of values
    /// * `ddof` - Delta degrees of freedom (0 for population variance, 1 for sample variance)
    ///
    /// # Returns
    ///
    /// * The variance of the values
    pub fn stable_variance(&self, values: &[T], ddof: usize) -> Result<T> {
        if values.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Cannot compute variance of empty array".to_string(),
            ));
        }

        if values.len() <= ddof {
            return Err(MetricsError::InvalidInput(format!(
                "Not enough values to compute variance with ddof={}",
                ddof
            )));
        }

        let mut mean = T::zero();
        let mut m2 = T::zero();
        let mut count = T::zero();

        for &value in values {
            count = count + T::one();
            // Using Welford's online algorithm
            let delta = value - mean;
            mean = mean + delta / count;
            let delta2 = value - mean;
            m2 = m2 + delta * delta2;
        }

        // Convert to f64 for calculation, then back to T
        let n = T::from(values.len()).unwrap();
        let ddof_t = T::from(ddof).unwrap();

        Ok(m2 / (n - ddof_t))
    }

    /// Numerically stable calculation of standard deviation
    ///
    /// Uses the stable variance calculation and takes the square root.
    ///
    /// # Arguments
    ///
    /// * `values` - Input array of values
    /// * `ddof` - Delta degrees of freedom (0 for population std, 1 for sample std)
    ///
    /// # Returns
    ///
    /// * The standard deviation of the values
    pub fn stable_std(&self, values: &[T], ddof: usize) -> Result<T> {
        let var = self.stable_variance(values, ddof)?;

        // Handle possibly negative variance due to numerical precision
        if var < T::zero() {
            if var.abs() < self.epsilon {
                Ok(T::zero())
            } else {
                Err(MetricsError::CalculationError(
                    "Computed negative variance in stable_std".to_string(),
                ))
            }
        } else {
            Ok(var.sqrt())
        }
    }

    /// Safely compute logarithm
    ///
    /// Avoids taking the logarithm of zero by adding a small epsilon value.
    ///
    /// # Arguments
    ///
    /// * `x` - Input value
    ///
    /// # Returns
    ///
    /// * The logarithm of the input, with epsilon added to avoid log(0)
    pub fn safe_log(&self, x: T) -> T {
        x.max(self.epsilon).ln()
    }

    /// Safely compute reciprocal
    ///
    /// Avoids division by zero by adding a small epsilon value to the denominator.
    ///
    /// # Arguments
    ///
    /// * `numer` - Numerator
    /// * `denom` - Denominator
    ///
    /// # Returns
    ///
    /// * The result of numer / (denom + epsilon)
    pub fn safe_div(&self, numer: T, denom: T) -> T {
        numer / (denom + self.epsilon)
    }

    /// Clip values to a reasonable range
    ///
    /// Limits extreme values to prevent numerical instability.
    ///
    /// # Arguments
    ///
    /// * `x` - Input value
    ///
    /// # Returns
    ///
    /// * The input value, clipped to the range [epsilon, max_value]
    pub fn clip(&self, x: T) -> T {
        if self.clip_values {
            x.max(self.epsilon).min(self.max_value)
        } else {
            x
        }
    }

    /// Compute log-sum-exp in a numerically stable way
    ///
    /// Uses the log-sum-exp trick to prevent overflow in exponentials.
    ///
    /// # Arguments
    ///
    /// * `x` - Array of values
    ///
    /// # Returns
    ///
    /// * The log-sum-exp of the values
    pub fn logsumexp(&self, x: &[T]) -> T {
        if x.is_empty() {
            return T::neg_infinity();
        }

        // Find the maximum value
        let max_val = x.iter().cloned().fold(T::neg_infinity(), T::max);

        // If max is -infinity, all values are -infinity, so return -infinity
        if max_val == T::neg_infinity() {
            return T::neg_infinity();
        }

        // Compute exp(x - max) and sum
        let sum = x
            .iter()
            .map(|&v| (v - max_val).exp())
            .fold(T::zero(), |acc, v| acc + v);

        // Return max + log(sum)
        max_val + sum.ln()
    }

    /// Compute softmax in a numerically stable way
    ///
    /// Uses the log-sum-exp trick to prevent overflow in exponentials.
    ///
    /// # Arguments
    ///
    /// * `x` - Array of values
    ///
    /// # Returns
    ///
    /// * Array with softmax values
    pub fn softmax(&self, x: &[T]) -> Vec<T> {
        if x.is_empty() {
            return Vec::new();
        }

        // Find the maximum value
        let max_val = x.iter().cloned().fold(T::neg_infinity(), T::max);

        // If max is -infinity, all values are -infinity
        if max_val == T::neg_infinity() {
            let n = x.len();
            return vec![T::from(1.0).unwrap() / T::from(n).unwrap(); n];
        }

        // Compute exp(x - max)
        let mut exp_vals: Vec<T> = x.iter().map(|&v| (v - max_val).exp()).collect();

        // Compute sum of exp_vals
        let sum = exp_vals.iter().fold(T::zero(), |acc, &v| acc + v);

        // Divide each value by the sum
        for val in &mut exp_vals {
            *val = *val / sum;
        }

        exp_vals
    }

    /// Compute cross-entropy in a numerically stable way
    ///
    /// # Arguments
    ///
    /// * `y_true` - True probabilities
    /// * `y_pred` - Predicted probabilities
    ///
    /// # Returns
    ///
    /// * The cross-entropy loss
    pub fn cross_entropy(&self, y_true: &[T], y_pred: &[T]) -> Result<T> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::DimensionMismatch(format!(
                "y_true and y_pred must have the same length, got {} and {}",
                y_true.len(),
                y_pred.len()
            )));
        }

        let mut loss = T::zero();
        for (p, q) in y_true.iter().zip(y_pred.iter()) {
            // Skip if true probability is zero (0 * log(q) = 0)
            if *p > T::zero() {
                // Clip predicted probability to avoid log(0)
                let q_clipped = q.max(self.epsilon).min(T::one());
                loss = loss - (*p * q_clipped.ln());
            }
        }

        Ok(loss)
    }

    /// Compute Kullback-Leibler divergence in a numerically stable way
    ///
    /// # Arguments
    ///
    /// * `p` - True probability distribution
    /// * `q` - Predicted probability distribution
    ///
    /// # Returns
    ///
    /// * The KL divergence
    pub fn kl_divergence(&self, p: &[T], q: &[T]) -> Result<T> {
        if p.len() != q.len() {
            return Err(MetricsError::DimensionMismatch(format!(
                "p and q must have the same length, got {} and {}",
                p.len(),
                q.len()
            )));
        }

        let mut kl = T::zero();
        for (p_i, q_i) in p.iter().zip(q.iter()) {
            // Skip if p_i is zero (0 * log(p_i/q_i) = 0)
            if *p_i > T::zero() {
                // Clip q_i to avoid division by zero
                let q_clipped = q_i.max(self.epsilon);

                // Calculate log(p_i/q_clipped)
                let log_ratio = (*p_i).ln() - q_clipped.ln();

                kl = kl + (*p_i * log_ratio);
            }
        }

        Ok(kl)
    }

    /// Compute Jensen-Shannon divergence in a numerically stable way
    ///
    /// # Arguments
    ///
    /// * `p` - First probability distribution
    /// * `q` - Second probability distribution
    ///
    /// # Returns
    ///
    /// * The JS divergence
    pub fn js_divergence(&self, p: &[T], q: &[T]) -> Result<T> {
        if p.len() != q.len() {
            return Err(MetricsError::DimensionMismatch(format!(
                "p and q must have the same length, got {} and {}",
                p.len(),
                q.len()
            )));
        }

        // Compute midpoint distribution m = (p + q) / 2
        let mut m = Vec::with_capacity(p.len());
        for (p_i, q_i) in p.iter().zip(q.iter()) {
            m.push((*p_i + *q_i) / T::from(2.0).unwrap());
        }

        // Compute KL(p || m) and KL(q || m)
        let kl_p_m = self.kl_divergence(p, &m)?;
        let kl_q_m = self.kl_divergence(q, &m)?;

        // JS = (KL(p || m) + KL(q || m)) / 2
        Ok((kl_p_m + kl_q_m) / T::from(2.0).unwrap())
    }

    /// Compute Wasserstein distance between 1D probability distributions
    ///
    /// # Arguments
    ///
    /// * `u_values` - First distribution sample values
    /// * `v_values` - Second distribution sample values
    ///
    /// # Returns
    ///
    /// * The Wasserstein distance
    pub fn wasserstein_distance(&self, u_values: &[T], v_values: &[T]) -> Result<T> {
        if u_values.is_empty() || v_values.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Input arrays must not be empty".to_string(),
            ));
        }

        // Sort the values
        let mut u_sorted = u_values.to_vec();
        let mut v_sorted = v_values.to_vec();

        u_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute the Wasserstein distance
        let n_u = u_sorted.len();
        let n_v = v_sorted.len();

        let mut distance = T::zero();
        for i in 0..n_u.max(n_v) {
            let u_quantile = if i < n_u {
                u_sorted[i]
            } else {
                u_sorted[n_u - 1]
            };

            let v_quantile = if i < n_v {
                v_sorted[i]
            } else {
                v_sorted[n_v - 1]
            };

            distance = distance + (u_quantile - v_quantile).abs();
        }

        // Normalize by the number of points
        Ok(distance / T::from(n_u.max(n_v)).unwrap())
    }

    /// Compute maximum mean discrepancy (MMD) between samples
    ///
    /// # Arguments
    ///
    /// * `x` - First sample
    /// * `y` - Second sample
    /// * `gamma` - RBF kernel parameter
    ///
    /// # Returns
    ///
    /// * The MMD value
    pub fn maximum_mean_discrepancy(&self, x: &[T], y: &[T], gamma: Option<T>) -> Result<T> {
        if x.is_empty() || y.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Input arrays must not be empty".to_string(),
            ));
        }

        // Default gamma value as 1.0 / median_distance
        let gamma = gamma.unwrap_or_else(|| T::one());

        // Compute kernel matrices
        let xx = self.rbf_kernel_mean(x, x, gamma);
        let yy = self.rbf_kernel_mean(y, y, gamma);
        let xy = self.rbf_kernel_mean(x, y, gamma);

        // Compute MMD
        Ok(xx + yy - T::from(2.0).unwrap() * xy)
    }

    // Helper function to compute mean of RBF kernel evaluations
    fn rbf_kernel_mean(&self, x: &[T], y: &[T], gamma: T) -> T {
        let mut sum = T::zero();
        let n_x = x.len();
        let n_y = y.len();

        for &x_i in x {
            for &y_j in y {
                let squared_dist = (x_i - y_j).powi(2);
                sum = sum + (-gamma * squared_dist).exp();
            }
        }

        sum / (T::from(n_x).unwrap() * T::from(n_y).unwrap())
    }

    /// Safely compute matrix exponential trace
    ///
    /// Computes tr(exp(A)) in a numerically stable way.
    /// This is useful for computing the nuclear norm of a matrix.
    ///
    /// # Arguments
    ///
    /// * `eigenvalues` - Eigenvalues of matrix A
    ///
    /// # Returns
    ///
    /// * The trace of the matrix exponential
    pub fn matrix_exp_trace(&self, eigenvalues: &[T]) -> Result<T> {
        if eigenvalues.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Cannot compute matrix exponential trace with empty eigenvalues".to_string(),
            ));
        }

        // Compute the sum of exponentials of eigenvalues
        let mut sum = T::zero();
        for &eigenvalue in eigenvalues {
            // Clip extreme values to prevent overflow
            let clipped = self.clip(eigenvalue);
            sum = sum + clipped.exp();
        }

        Ok(sum)
    }

    /// Compute stable matrix logarithm determinant
    ///
    /// Computes log(det(A)) in a numerically stable way by summing
    /// the logarithms of eigenvalues.
    ///
    /// # Arguments
    ///
    /// * `eigenvalues` - Eigenvalues of matrix A
    ///
    /// # Returns
    ///
    /// * The logarithm of the determinant
    pub fn matrix_logdet(&self, eigenvalues: &[T]) -> Result<T> {
        if eigenvalues.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Cannot compute matrix logarithm determinant with empty eigenvalues".to_string(),
            ));
        }

        // Check if any eigenvalues are negative or zero
        for &eigenvalue in eigenvalues {
            if eigenvalue <= T::zero() {
                return Err(MetricsError::CalculationError(
                    "Cannot compute logarithm of non-positive eigenvalues".to_string(),
                ));
            }
        }

        // Compute the sum of logarithms of eigenvalues
        let mut log_det = T::zero();
        for &eigenvalue in eigenvalues {
            log_det = log_det + self.safe_log(eigenvalue);
        }

        Ok(log_det)
    }

    /// Compute numerically stable log1p (log(1+x))
    ///
    /// More accurate than log(1+x) for small values of x.
    ///
    /// # Arguments
    ///
    /// * `x` - Input value
    ///
    /// # Returns
    ///
    /// * log(1+x) computed in a numerically stable way
    pub fn log1p(&self, x: T) -> T {
        // For very small x, use Taylor series approximation
        if x.abs() < T::from(1e-4).unwrap() {
            let x2 = x * x;
            let x3 = x2 * x;
            let x4 = x2 * x2;
            return x - x2 / T::from(2).unwrap() + x3 / T::from(3).unwrap()
                - x4 / T::from(4).unwrap();
        }

        // Otherwise use log(1+x) directly
        (T::one() + x).ln()
    }

    /// Compute numerically stable expm1 (exp(x)-1)
    ///
    /// More accurate than exp(x)-1 for small values of x.
    ///
    /// # Arguments
    ///
    /// * `x` - Input value
    ///
    /// # Returns
    ///
    /// * exp(x)-1 computed in a numerically stable way
    pub fn expm1(&self, x: T) -> T {
        // For very small x, use Taylor series approximation
        if x.abs() < T::from(1e-4).unwrap() {
            let x2 = x * x;
            let x3 = x2 * x;
            let x4 = x3 * x;
            return x
                + x2 / T::from(2).unwrap()
                + x3 / T::from(6).unwrap()
                + x4 / T::from(24).unwrap();
        }

        // Otherwise use exp(x)-1 directly
        x.exp() - T::one()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_safe_log() {
        let stable = StableMetrics::<f64>::default();

        // Test normal case
        assert_abs_diff_eq!(stable.safe_log(2.0), 2.0f64.ln(), epsilon = 1e-10);

        // Test zero
        assert_abs_diff_eq!(stable.safe_log(0.0), stable.epsilon.ln(), epsilon = 1e-10);

        // Test small value
        let small = 1e-15;
        assert_abs_diff_eq!(stable.safe_log(small), stable.epsilon.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_safe_div() {
        let stable = StableMetrics::<f64>::default();

        // Test normal case
        assert_abs_diff_eq!(stable.safe_div(10.0, 2.0), 5.0, epsilon = 1e-8);

        // Test division by zero
        assert_abs_diff_eq!(
            stable.safe_div(10.0, 0.0),
            10.0 / stable.epsilon,
            epsilon = 1e-10
        );

        // Test division by small value
        let small = 1e-15;
        assert_abs_diff_eq!(
            stable.safe_div(10.0, small),
            10.0 / (small + stable.epsilon),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_clip() {
        let stable = StableMetrics::<f64>::default()
            .with_epsilon(1e-5)
            .with_max_value(1e5);

        // Test normal case
        assert_abs_diff_eq!(stable.clip(50.0), 50.0, epsilon = 1e-10);

        // Test small value
        assert_abs_diff_eq!(stable.clip(1e-10), 1e-5, epsilon = 1e-10);

        // Test large value
        assert_abs_diff_eq!(stable.clip(1e10), 1e5, epsilon = 1e-10);
    }

    #[test]
    fn test_logsumexp() {
        let stable = StableMetrics::<f64>::default();

        // Test standard case
        let x = vec![1.0, 2.0, 3.0];
        let expected = (1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln();
        assert_abs_diff_eq!(stable.logsumexp(&x), expected, epsilon = 1e-10);

        // Test large values that would overflow with naive approach
        let large_vals = vec![1000.0, 1000.0, 1000.0];
        let expected = 1000.0 + (3.0f64).ln();
        assert_abs_diff_eq!(stable.logsumexp(&large_vals), expected, epsilon = 1e-10);

        // Test empty array
        assert_eq!(stable.logsumexp(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn test_softmax() {
        let stable = StableMetrics::<f64>::default();

        // Test standard case
        let x = vec![1.0, 2.0, 3.0];
        let softmax = stable.softmax(&x);
        let total: f64 = softmax.iter().sum();

        // Verify softmax properties
        assert_abs_diff_eq!(total, 1.0, epsilon = 1e-10);
        assert!(softmax[2] > softmax[1] && softmax[1] > softmax[0]);

        // Test large values that would overflow with naive approach
        let large_vals = vec![1000.0, 999.0, 998.0];
        let softmax_large = stable.softmax(&large_vals);
        let total_large: f64 = softmax_large.iter().sum();

        // Verify softmax properties for large values
        assert_abs_diff_eq!(total_large, 1.0, epsilon = 1e-10);
        assert!(softmax_large[0] > softmax_large[1] && softmax_large[1] > softmax_large[2]);
    }

    #[test]
    fn test_cross_entropy() {
        let stable = StableMetrics::<f64>::default();

        // Test standard case
        let y_true = vec![0.0, 1.0, 0.0];
        let y_pred = vec![0.1, 0.8, 0.1];

        // Expected: -sum(y_true * log(y_pred)) = -1.0 * log(0.8) = -log(0.8)
        let expected = -0.8f64.ln();
        let ce = stable.cross_entropy(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(ce, expected, epsilon = 1e-10);

        // Test with zero in prediction
        let y_pred_zero = vec![0.0, 0.8, 0.2];
        let ce_zero = stable.cross_entropy(&y_true, &y_pred_zero).unwrap();
        // Should use epsilon instead of zero
        assert!(ce_zero.is_finite());

        // Test dimensions mismatch
        let y_pred_short = vec![0.1, 0.9];
        assert!(stable.cross_entropy(&y_true, &y_pred_short).is_err());
    }

    #[test]
    fn test_kl_divergence() {
        let stable = StableMetrics::<f64>::default();

        // Test standard case
        let p = vec![0.5, 0.5, 0.0];
        let q = vec![0.25, 0.25, 0.5];

        // Calculate expected KL divergence
        // KL(p||q) = sum(p_i * log(p_i/q_i))
        let expected = 0.5 * (0.5 / 0.25).ln() + 0.5 * (0.5 / 0.25).ln();
        let kl = stable.kl_divergence(&p, &q).unwrap();
        assert_abs_diff_eq!(kl, expected, epsilon = 1e-10);

        // Test with zero in q where p is zero (should be fine)
        let q_zero = vec![0.5, 0.5, 0.0];
        let kl_zero = stable.kl_divergence(&p, &q_zero).unwrap();
        assert_abs_diff_eq!(kl_zero, 0.0, epsilon = 1e-10);

        // Test with zero in q where p is non-zero (should use epsilon)
        let p_nonzero = vec![0.4, 0.3, 0.3];
        let q_more_zeros = vec![0.6, 0.4, 0.0];
        let kl_safe = stable.kl_divergence(&p_nonzero, &q_more_zeros).unwrap();
        assert!(kl_safe.is_finite());
    }

    #[test]
    fn test_js_divergence() {
        let stable = StableMetrics::<f64>::default();

        // Test standard case
        let p = vec![0.5, 0.5, 0.0];
        let q = vec![0.25, 0.25, 0.5];

        // Manually compute JS divergence
        let m = [0.375, 0.375, 0.25]; // (p + q) / 2

        let kl_p_m_expected = p[0] * (p[0] / m[0]).ln() + p[1] * (p[1] / m[1]).ln();
        let kl_q_m_expected =
            q[0] * (q[0] / m[0]).ln() + q[1] * (q[1] / m[1]).ln() + q[2] * (q[2] / m[2]).ln();
        let expected = 0.5 * (kl_p_m_expected + kl_q_m_expected);

        let js = stable.js_divergence(&p, &q).unwrap();
        assert_abs_diff_eq!(js, expected, epsilon = 1e-10);

        // JS divergence should be symmetric
        let js_reverse = stable.js_divergence(&q, &p).unwrap();
        assert_abs_diff_eq!(js, js_reverse, epsilon = 1e-10);

        // JS divergence should be 0 for identical distributions
        let js_identical = stable.js_divergence(&p, &p).unwrap();
        assert_abs_diff_eq!(js_identical, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_wasserstein_distance() {
        let stable = StableMetrics::<f64>::default();

        // Test with uniform distributions
        let u = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let distance = stable.wasserstein_distance(&u, &v).unwrap();
        assert_abs_diff_eq!(distance, 0.0, epsilon = 1e-10);

        // Test with shifted distributions
        let w = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let distance = stable.wasserstein_distance(&u, &w).unwrap();
        assert_abs_diff_eq!(distance, 1.0, epsilon = 1e-10);

        // Test with different distribution sizes
        let x = vec![1.0, 3.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];

        let distance = stable.wasserstein_distance(&x, &y).unwrap();
        assert!(distance > 0.0);
    }

    #[test]
    fn test_maximum_mean_discrepancy() {
        let stable = StableMetrics::<f64>::default();

        // Test with identical distributions
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mmd = stable.maximum_mean_discrepancy(&x, &y, Some(0.1)).unwrap();
        assert!(mmd < 1e-10); // Should be close to 0

        // Test with different distributions
        let z = vec![6.0, 7.0, 8.0, 9.0, 10.0];

        let mmd = stable.maximum_mean_discrepancy(&x, &z, Some(0.1)).unwrap();
        assert!(mmd > 0.1); // Should be significantly positive
    }

    #[test]
    fn test_stable_mean() {
        let stable = StableMetrics::<f64>::default();

        // Test standard case
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = stable.stable_mean(&values).unwrap();
        assert_abs_diff_eq!(mean, 3.0, epsilon = 1e-10);

        // Test with single value
        let single = vec![42.0];
        let mean_single = stable.stable_mean(&single).unwrap();
        assert_abs_diff_eq!(mean_single, 42.0, epsilon = 1e-10);

        // Test with empty array
        assert!(stable.stable_mean(&[]).is_err());
    }

    #[test]
    fn test_stable_variance() {
        let stable = StableMetrics::<f64>::default();

        // Test standard case with population variance (ddof=0)
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let var = stable.stable_variance(&values, 0).unwrap();
        let expected_var = 2.0; // Variance of [1,2,3,4,5] is 2
        assert_abs_diff_eq!(var, expected_var, epsilon = 1e-10);

        // Test standard case with sample variance (ddof=1)
        let sample_var = stable.stable_variance(&values, 1).unwrap();
        let expected_sample_var = 2.5; // Sample variance of [1,2,3,4,5] is 2.5
        assert_abs_diff_eq!(sample_var, expected_sample_var, epsilon = 1e-10);

        // Test with not enough values
        assert!(stable.stable_variance(&[1.0], 1).is_err());

        // Test with empty array
        assert!(stable.stable_variance(&[], 0).is_err());
    }

    #[test]
    fn test_stable_std() {
        let stable = StableMetrics::<f64>::default();

        // Test standard case with population std (ddof=0)
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std_dev = stable.stable_std(&values, 0).unwrap();
        let expected_std = 2.0f64.sqrt(); // STD of [1,2,3,4,5] is sqrt(2)
        assert_abs_diff_eq!(std_dev, expected_std, epsilon = 1e-10);

        // Test standard case with sample std (ddof=1)
        let sample_std = stable.stable_std(&values, 1).unwrap();
        let expected_sample_std = 2.5f64.sqrt(); // Sample STD of [1,2,3,4,5] is sqrt(2.5)
        assert_abs_diff_eq!(sample_std, expected_sample_std, epsilon = 1e-10);
    }

    #[test]
    fn test_log1p_expm1() {
        let stable = StableMetrics::<f64>::default();

        // Test log1p for small values
        let small_x = 1e-8;
        assert_abs_diff_eq!(stable.log1p(small_x), (1.0 + small_x).ln(), epsilon = 1e-15);

        // Test log1p for regular values
        let x = 1.5;
        assert_abs_diff_eq!(stable.log1p(x), (1.0 + x).ln(), epsilon = 1e-10);

        // Test expm1 for small values
        let small_y = 1e-8;
        assert_abs_diff_eq!(stable.expm1(small_y), small_y.exp() - 1.0, epsilon = 1e-15);

        // Test expm1 for regular values
        let y = 1.5;
        assert_abs_diff_eq!(stable.expm1(y), y.exp() - 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_operations() {
        let stable = StableMetrics::<f64>::default();

        // Test matrix_exp_trace
        let eigenvalues = vec![1.0, 2.0, 3.0];
        let exp_trace = stable.matrix_exp_trace(&eigenvalues).unwrap();
        let expected = 1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp();
        assert_abs_diff_eq!(exp_trace, expected, epsilon = 1e-10);

        // Test matrix_logdet
        let positive_eigenvalues = vec![1.0, 2.0, 5.0];
        let logdet = stable.matrix_logdet(&positive_eigenvalues).unwrap();
        let expected = 1.0f64.ln() + 2.0f64.ln() + 5.0f64.ln();
        assert_abs_diff_eq!(logdet, expected, epsilon = 1e-10);

        // Test matrix_logdet with negative eigenvalues (should fail)
        let negative_eigenvalues = vec![1.0, -2.0, 3.0];
        assert!(stable.matrix_logdet(&negative_eigenvalues).is_err());
    }
}
