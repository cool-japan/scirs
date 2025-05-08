//! Statistical functions module
//!
//! This module provides implementations of various statistical algorithms,
//! modeled after SciPy's stats module.
//!
//! ## Overview
//!
//! * Descriptive statistics
//!   - Basic statistics (mean, median, variance, etc.)
//!   - Advanced statistics (skewness, kurtosis, moments)
//!   - Correlation measures (Pearson, Spearman, Kendall tau, partial correlation)
//!   - Dispersion measures (MAD, median absolute deviation, IQR, range, coefficient of variation)
//!
//! * Statistical distributions
//!   - Normal distribution
//!   - Uniform distribution
//!   - Student's t distribution
//!   - Chi-square distribution
//!   - F distribution
//!   - Poisson distribution
//!   - Gamma distribution
//!   - Beta distribution
//!   - Exponential distribution
//!   - Hypergeometric distribution
//!   - Laplace distribution
//!   - Logistic distribution
//!   - Cauchy distribution
//!   - Pareto distribution
//!   - Weibull distribution
//!   - Multivariate distributions (multivariate normal, multivariate t, dirichlet, wishart, etc.)
//!
//! * Statistical tests
//!   - Parametric tests (t-tests, ANOVA)
//!   - Non-parametric tests (Mann-Whitney U)
//!   - Normality tests (Shapiro-Wilk, Anderson-Darling, D'Agostino's K²)
//!   - Goodness-of-fit tests (Chi-square)
//! * Random number generation
//! * Regression models
//! * Contingency table functions
//! * Masked array statistics
//! * Quasi-Monte Carlo
//! * Statistical sampling
//!
//! ## Examples
//!
//! ### Descriptive Statistics
//!
//! ```
//! use ndarray::array;
//! use scirs2_stats::{mean, median, std, var, skew, kurtosis};
//!
//! let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
//!
//! // Calculate basic statistics
//! let mean_val = mean(&data.view()).unwrap();
//! let median_val = median(&data.view()).unwrap();
//! let var_val = var(&data.view(), 0).unwrap();  // ddof = 0 for population variance
//! let std_val = std(&data.view(), 0).unwrap();  // ddof = 0 for population standard deviation
//!
//! // Advanced statistics
//! let skewness = skew(&data.view(), false).unwrap();  // bias = false
//! let kurt = kurtosis(&data.view(), true, false).unwrap();  // fisher = true, bias = false
//! ```
//!
//! ### Correlation Measures
//!
//! ```
//! use ndarray::{array, Array2};
//! use scirs2_stats::{pearson_r, spearman_r, kendall_tau, corrcoef};
//!
//! let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
//!
//! // Calculate Pearson correlation coefficient (linear correlation)
//! let r = pearson_r(&x.view(), &y.view()).unwrap();
//! println!("Pearson correlation: {}", r);  // Should be -1.0 (perfect negative correlation)
//!
//! // Spearman rank correlation (monotonic relationship)
//! let rho = spearman_r(&x.view(), &y.view()).unwrap();
//! println!("Spearman correlation: {}", rho);
//!
//! // Kendall tau rank correlation
//! let tau = kendall_tau(&x.view(), &y.view(), "b").unwrap();
//! println!("Kendall tau correlation: {}", tau);
//!
//! // Correlation matrix for multiple variables
//! let data = array![
//!     [1.0, 5.0, 10.0],
//!     [2.0, 4.0, 9.0],
//!     [3.0, 3.0, 8.0],
//!     [4.0, 2.0, 7.0],
//!     [5.0, 1.0, 6.0]
//! ];
//!
//! let corr_matrix = corrcoef(&data.view(), "pearson").unwrap();
//! println!("Correlation matrix:\n{:?}", corr_matrix);
//! ```
//!
//! ### Dispersion Measures
//!
//! ```
//! use ndarray::array;
//! use scirs2_stats::{
//!     mean_abs_deviation, median_abs_deviation, iqr, data_range, coef_variation
//! };
//!
//! let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];  // Note the outlier
//!
//! // Mean absolute deviation (from mean)
//! let mad = mean_abs_deviation(&data.view(), None).unwrap();
//! println!("Mean absolute deviation: {}", mad);
//!
//! // Median absolute deviation (robust to outliers)
//! let median_ad = median_abs_deviation(&data.view(), None, None).unwrap();
//! println!("Median absolute deviation: {}", median_ad);
//!
//! // Scaled median absolute deviation (consistent with std dev for normal distributions)
//! let median_ad_scaled = median_abs_deviation(&data.view(), None, Some(1.4826)).unwrap();
//! println!("Scaled median absolute deviation: {}", median_ad_scaled);
//!
//! // Interquartile range (Q3 - Q1)
//! let iqr_val = iqr(&data.view(), None).unwrap();
//! println!("Interquartile range: {}", iqr_val);
//!
//! // Range (max - min)
//! let range_val = data_range(&data.view()).unwrap();
//! println!("Range: {}", range_val);
//!
//! // Coefficient of variation (std/mean, unitless measure)
//! let cv = coef_variation(&data.view(), 1).unwrap();
//! println!("Coefficient of variation: {}", cv);
//! ```
//!
//! ### Statistical Distributions
//!
//! ```
//! use scirs2_stats::distributions;
//!
//! // Normal distribution
//! let normal = distributions::norm(0.0f64, 1.0).unwrap();
//! let pdf = normal.pdf(0.0);
//! let cdf = normal.cdf(1.96);
//! let samples = normal.rvs(100).unwrap();
//!
//! // Poisson distribution
//! let poisson = distributions::poisson(3.0f64, 0.0).unwrap();
//! let pmf = poisson.pmf(2.0);
//! let cdf = poisson.cdf(4.0);
//! let samples = poisson.rvs(100).unwrap();
//!
//! // Gamma distribution
//! let gamma = distributions::gamma(2.0f64, 1.0, 0.0).unwrap();
//! let pdf = gamma.pdf(1.0);
//! let cdf = gamma.cdf(2.0);
//! let samples = gamma.rvs(100).unwrap();
//!
//! // Beta distribution
//! let beta = distributions::beta(2.0f64, 3.0, 0.0, 1.0).unwrap();
//! let pdf = beta.pdf(0.5);
//! let samples = beta.rvs(100).unwrap();
//!
//! // Exponential distribution
//! let exp = distributions::expon(1.0f64, 0.0).unwrap();
//! let pdf = exp.pdf(1.0);
//! let mean = exp.mean(); // Should be 1.0
//!
//! // Multivariate normal distribution
//! use ndarray::array;
//! let mvn_mean = array![0.0, 0.0];
//! let mvn_cov = array![[1.0, 0.5], [0.5, 2.0]];
//! let mvn = distributions::multivariate::multivariate_normal(mvn_mean, mvn_cov).unwrap();
//! let pdf = mvn.pdf(&array![0.0, 0.0]);
//! let samples = mvn.rvs(100).unwrap();
//! ```
//!
//! ### Statistical Tests
//!
//! ```
//! use ndarray::{array, Array2};
//! use scirs2_stats::{
//!     ttest_1samp, ttest_ind, ttest_rel, kstest, shapiro, mannwhitneyu,
//!     shapiro_wilk, anderson_darling, dagostino_k2, wilcoxon, kruskal_wallis, friedman,
//!     distributions
//! };
//!
//! // One-sample t-test (we'll use a larger sample for normality tests)
//! let data = array![
//!     5.1, 4.9, 6.2, 5.7, 5.5, 5.1, 5.2, 5.0, 5.3, 5.4,
//!     5.6, 5.8, 5.9, 6.0, 5.2, 5.4, 5.3, 5.1, 5.2, 5.0
//! ];
//! let (t_stat, p_value) = ttest_1samp(&data.view(), 5.0).unwrap();
//! println!("One-sample t-test: t={}, p={}", t_stat, p_value);
//!
//! // Two-sample t-test
//! let group1 = array![5.1, 4.9, 6.2, 5.7, 5.5];
//! let group2 = array![4.8, 5.2, 5.1, 4.7, 4.9];
//! let (t_stat, p_value) = ttest_ind(&group1.view(), &group2.view(), true).unwrap();
//! println!("Two-sample t-test: t={}, p={}", t_stat, p_value);
//!
//! // Normality tests
//! let (w_stat, p_value) = shapiro(&data.view()).unwrap();
//! println!("Shapiro-Wilk test: W={}, p={}", w_stat, p_value);
//!
//! // More accurate Shapiro-Wilk test implementation
//! let (w_stat, p_value) = shapiro_wilk(&data.view()).unwrap();
//! println!("Improved Shapiro-Wilk test: W={}, p={}", w_stat, p_value);
//!
//! // Anderson-Darling test for normality
//! let (a2_stat, p_value) = anderson_darling(&data.view()).unwrap();
//! println!("Anderson-Darling test: A²={}, p={}", a2_stat, p_value);
//!
//! // D'Agostino's K² test combining skewness and kurtosis
//! let (k2_stat, p_value) = dagostino_k2(&data.view()).unwrap();
//! println!("D'Agostino K² test: K²={}, p={}", k2_stat, p_value);
//!
//! // Non-parametric tests
//!
//! // Wilcoxon signed-rank test (paired samples)
//! let before = array![125.0, 115.0, 130.0, 140.0, 140.0];
//! let after = array![110.0, 122.0, 125.0, 120.0, 140.0];
//! let (w, p_value) = wilcoxon(&before.view(), &after.view(), "wilcox", true).unwrap();
//! println!("Wilcoxon signed-rank test: W={}, p={}", w, p_value);
//!
//! // Kruskal-Wallis test (unpaired samples)
//! let group1 = array![2.9, 3.0, 2.5, 2.6, 3.2];
//! let group2 = array![3.8, 3.7, 3.9, 4.0, 4.2];
//! let group3 = array![2.8, 3.4, 3.7, 2.2, 2.0];
//! let samples = vec![group1.view(), group2.view(), group3.view()];
//! let (h, p_value) = kruskal_wallis(&samples).unwrap();
//! println!("Kruskal-Wallis test: H={}, p={}", h, p_value);
//!
//! // Friedman test (repeated measures)
//! let data = array![
//!     [7.0, 9.0, 8.0],
//!     [6.0, 5.0, 7.0],
//!     [9.0, 7.0, 6.0],
//!     [8.0, 5.0, 6.0]
//! ];
//! let (chi2, p_value) = friedman(&data.view()).unwrap();
//! println!("Friedman test: Chi²={}, p={}", chi2, p_value);
//!
//! // Distribution fit test
//! let normal = distributions::norm(0.0f64, 1.0).unwrap();
//! let standardized_data = array![0.1, -0.2, 0.3, -0.1, 0.2];
//! let (ks_stat, p_value) = kstest(&standardized_data.view(), |x| normal.cdf(x)).unwrap();
//! println!("Kolmogorov-Smirnov test: D={}, p={}", ks_stat, p_value);
//! ```
//!
//! ### Random Number Generation
//!
//! ```
//! use scirs2_stats::random::{uniform, randn, randint, choice};
//! use ndarray::array;
//!
//! // Generate uniform random numbers between 0 and 1
//! let uniform_samples = uniform(0.0, 1.0, 10, Some(42)).unwrap();
//!
//! // Generate standard normal random numbers
//! let normal_samples = randn(10, Some(123)).unwrap();
//!
//! // Generate random integers between 1 and 100
//! let int_samples = randint(1, 101, 5, Some(456)).unwrap();
//!
//! // Randomly choose elements from an array
//! let options = array!["apple", "banana", "cherry", "date", "elderberry"];
//! let choices = choice(&options.view(), 3, false, None, Some(789)).unwrap();
//! ```
//!
//! ### Statistical Sampling
//!
//! ```
//! use scirs2_stats::sampling;
//! use ndarray::array;
//!
//! // Create an array
//! let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
//!
//! // Generate bootstrap samples
//! let bootstrap_samples = sampling::bootstrap(&data.view(), 10, Some(42)).unwrap();
//!
//! // Generate a random permutation
//! let permutation = sampling::permutation(&data.view(), Some(123)).unwrap();
//! ```
extern crate openblas_src;

// Export error types
pub mod error;
pub use error::{StatsError, StatsResult};

// Module substructure following SciPy's organization
pub mod contingency; // Contingency table functions
pub mod distributions; // Statistical distributions
pub mod mstats; // Masked array statistics
pub mod qmc; // Quasi-Monte Carlo
pub mod sampling; // Sampling utilities
pub mod traits; // Trait definitions for distributions and statistical objects

// Core functions for descriptive statistics
mod descriptive;
pub use descriptive::*;

// Statistical tests module
pub mod tests;
pub use tests::anova::{one_way_anova, tukey_hsd};
pub use tests::chi2_test::{chi2_gof, chi2_independence, chi2_yates};
pub use tests::nonparametric::{friedman, kruskal_wallis, wilcoxon};
pub use tests::normality::{anderson_darling, dagostino_k2, shapiro_wilk};
pub use tests::*;

// Correlation measures
mod correlation;
pub use correlation::{corrcoef, kendall_tau, partial_corr, pearson_r, point_biserial, spearman_r};

// Dispersion and variability measures
mod dispersion;
pub use dispersion::{coef_variation, data_range, iqr, mean_abs_deviation, median_abs_deviation};

// Core functions for regression analysis
mod regression;
pub use regression::*;

// Core functions for random number generation
pub mod random;
pub use random::*;

#[cfg(test)]
mod test_utils {
    // Common utilities for testing statistical functions

    /// Generate a simple test array
    pub fn test_array() -> ndarray::Array1<f64> {
        ndarray::array![1.0, 2.0, 3.0, 4.0, 5.0]
    }
}
