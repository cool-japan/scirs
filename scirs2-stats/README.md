# scirs2-stats

[![crates.io](https://img.shields.io/crates/v/scirs2-stats.svg)](https://crates.io/crates/scirs2-stats)
[![License](https://img.shields.io/crates/l/scirs2-stats.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-stats)](https://docs.rs/scirs2-stats)

Statistical functions module for the scirs2 project.

## Overview

This module provides implementations of various statistical algorithms, modeled after SciPy's stats module.

### Features

- Descriptive statistics
  - Basic measures: mean, median, variance, standard deviation
  - Advanced statistics: skewness, kurtosis, moments
  - Correlation measures: Pearson, Spearman, Kendall tau, partial correlation, point-biserial
  - Dispersion measures: MAD, median absolute deviation, IQR, range, coefficient of variation
- Statistical distributions
  - Continuous: Normal, Uniform, Student's t, Chi-square, F, Gamma, Beta, Exponential, Laplace, Logistic, Cauchy, Pareto, Weibull
  - Discrete: Poisson, Binomial, Hypergeometric, Bernoulli, Geometric, Negative Binomial
  - Multivariate: Multivariate Normal, Multivariate t, Dirichlet, Wishart, InverseWishart, Multinomial
- Statistical tests
  - Parametric tests: t-tests (one-sample, two-sample, paired), ANOVA
  - Non-parametric tests: Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Friedman
  - Normality tests: Shapiro-Wilk, Anderson-Darling, D'Agostino's K²
  - Goodness-of-fit tests: Chi-square, Kolmogorov-Smirnov
- Random number generation
- Regression models
- Sampling techniques (bootstrap, stratified sampling)
- Contingency table functions

## Installation

Add scirs2-stats to your Cargo.toml:

```toml
[dependencies]
scirs2-stats = "0.1.0-alpha.1"
ndarray = "0.16.1"
```

## Requirements

- Rust 1.65 or later
- [rand](https://crates.io/crates/rand) 0.9.0 or later
- [ndarray](https://crates.io/crates/ndarray) 0.16.0 or later

## Usage Examples

### Descriptive Statistics

```rust
use ndarray::array;
use scirs2_stats::{mean, median, std, var, skew, kurtosis};

let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

// Calculate basic statistics
let mean_val = mean(&data.view()).unwrap();
let median_val = median(&data.view()).unwrap();
let var_val = var(&data.view(), 0).unwrap();  // ddof = 0 for population variance
let std_val = std(&data.view(), 0).unwrap();  // ddof = 0 for population standard deviation

// Advanced statistics
let skewness = skew(&data.view(), false).unwrap();  // bias = false
let kurt = kurtosis(&data.view(), true, false).unwrap();  // fisher = true, bias = false
```

### Statistical Distributions

```rust
use scirs2_stats::distributions;

// Normal distribution
let normal = distributions::norm(0.0f64, 1.0).unwrap();
let pdf = normal.pdf(0.0);
let cdf = normal.cdf(1.96);
let samples = normal.rvs(100).unwrap();

// Poisson distribution
let poisson = distributions::poisson(3.0f64, 0.0).unwrap();
let pmf = poisson.pmf(2.0);
let cdf = poisson.cdf(4.0);
let samples = poisson.rvs(100).unwrap();
```

### Correlation Measures

```rust
use ndarray::{array, Array2};
use scirs2_stats::{pearson_r, spearman_r, kendall_tau, corrcoef};

// Calculate Pearson correlation coefficient (linear correlation)
let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
let y = array![5.0, 4.0, 3.0, 2.0, 1.0];

let r = pearson_r(&x.view(), &y.view()).unwrap();
println!("Pearson correlation: {}", r);  // Should be -1.0 (perfect negative correlation)

// Spearman rank correlation (monotonic relationship)
let rho = spearman_r(&x.view(), &y.view()).unwrap();
println!("Spearman correlation: {}", rho);

// Kendall tau rank correlation
let tau = kendall_tau(&x.view(), &y.view(), "b").unwrap();
println!("Kendall tau correlation: {}", tau);

// Correlation matrix for multiple variables
let data = array![
    [1.0, 5.0, 10.0],
    [2.0, 4.0, 9.0],
    [3.0, 3.0, 8.0],
    [4.0, 2.0, 7.0],
    [5.0, 1.0, 6.0]
];

let corr_matrix = corrcoef(&data.view(), "pearson").unwrap();
println!("Correlation matrix:\n{:?}", corr_matrix);
```

### Dispersion Measures

```rust
use ndarray::array;
use scirs2_stats::{
    mean_abs_deviation, median_abs_deviation, iqr, data_range, coef_variation
};

let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];  // Note the outlier

// Mean absolute deviation (from mean)
let mad = mean_abs_deviation(&data.view(), None).unwrap();
println!("Mean absolute deviation: {}", mad);

// Median absolute deviation (robust to outliers)
let median_ad = median_abs_deviation(&data.view(), None, None).unwrap();
println!("Median absolute deviation: {}", median_ad);

// Scaled median absolute deviation (to match std. dev. in normal distributions)
let scaled_mad = median_abs_deviation(&data.view(), None, Some(1.4826)).unwrap();
println!("Scaled median absolute deviation: {}", scaled_mad);

// Interquartile range (Q3 - Q1)
let iqr_val = iqr(&data.view(), None).unwrap();
println!("Interquartile range: {}", iqr_val);

// Range (max - min)
let range_val = data_range(&data.view()).unwrap();
println!("Range: {}", range_val);

// Coefficient of variation (std/mean)
let cv = coef_variation(&data.view(), 1).unwrap(); // 1 = sample
println!("Coefficient of variation: {}", cv);
```

### Statistical Tests

```rust
use ndarray::{array, Array2};
use scirs2_stats::{
    ttest_1samp, ttest_ind, ttest_rel, kstest, shapiro, 
    shapiro_wilk, anderson_darling, dagostino_k2,
    wilcoxon, kruskal_wallis, friedman
};

// Parametric tests
let data = array![5.1, 4.9, 6.2, 5.7, 5.5, 5.1, 5.2, 5.0];
let (t_stat, p_value) = ttest_1samp(&data.view(), 5.0).unwrap();
println!("One-sample t-test: t={}, p={}", t_stat, p_value);

let group1 = array![5.1, 4.9, 6.2, 5.7, 5.5];
let group2 = array![4.8, 5.2, 5.1, 4.7, 4.9];
let (t_stat, p_value) = ttest_ind(&group1.view(), &group2.view(), true).unwrap();
println!("Two-sample t-test: t={}, p={}", t_stat, p_value);

// Normality tests
let (w_stat, p_value) = shapiro_wilk(&data.view()).unwrap();
println!("Shapiro-Wilk test: W={}, p={}", w_stat, p_value);

let (a2_stat, p_value) = anderson_darling(&data.view()).unwrap();
println!("Anderson-Darling test: A²={}, p={}", a2_stat, p_value);

// Non-parametric tests
let before = array![125.0, 115.0, 130.0, 140.0, 140.0];
let after = array![110.0, 122.0, 125.0, 120.0, 140.0];
let (w, p_value) = wilcoxon(&before.view(), &after.view(), "wilcox", true).unwrap();
println!("Wilcoxon signed-rank test: W={}, p={}", w, p_value);

// Kruskal-Wallis test for independent samples
let group3 = array![2.8, 3.4, 3.7, 2.2, 2.0];
let samples = vec![group1.view(), group2.view(), group3.view()];
let (h, p_value) = kruskal_wallis(&samples).unwrap();
println!("Kruskal-Wallis test: H={}, p={}", h, p_value);

// Friedman test for repeated measures
let repeated_data = array![
    [7.0, 9.0, 8.0],
    [6.0, 5.0, 7.0],
    [9.0, 7.0, 6.0],
    [8.0, 5.0, 6.0]
];
let (chi2, p_value) = friedman(&repeated_data.view()).unwrap();
println!("Friedman test: Chi²={}, p={}", chi2, p_value);
```

### Random Number Generation

```rust
use scirs2_stats::random::{uniform, randn, randint, choice};
use ndarray::array;

// Generate uniform random numbers between 0 and 1
let uniform_samples = uniform(0.0, 1.0, 10, Some(42)).unwrap();

// Generate standard normal random numbers
let normal_samples = randn(10, Some(123)).unwrap();

// Generate random integers between 1 and 100
let int_samples = randint(1, 101, 5, Some(456)).unwrap();

// Randomly choose elements from an array
let options = array!["apple", "banana", "cherry", "date", "elderberry"];
let choices = choice(&options.view(), 3, false, None, Some(789)).unwrap();
```

### Statistical Sampling

```rust
use scirs2_stats::sampling;
use ndarray::array;

// Create an array
let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

// Generate bootstrap samples
let bootstrap_samples = sampling::bootstrap(&data.view(), 10, Some(42)).unwrap();

// Generate a random permutation
let permutation = sampling::permutation(&data.view(), Some(123)).unwrap();
```

## Recent Updates

- Added dispersion measures:
  - Mean absolute deviation (MAD)
  - Median absolute deviation
  - Interquartile range (IQR)
  - Range
  - Coefficient of variation
- Added correlation measures:
  - Pearson correlation coefficient
  - Spearman rank correlation
  - Kendall tau correlation
  - Partial correlation
  - Point-biserial correlation
  - Correlation matrix computation
- Added non-parametric statistical tests:
  - Wilcoxon signed-rank test
  - Kruskal-Wallis test
  - Friedman test
- Improved normality tests:
  - Enhanced Shapiro-Wilk implementation
  - Anderson-Darling test
  - D'Agostino's K² test
- Added discrete distributions:
  - Hypergeometric distribution
- Updated to use rand 0.9.0 API
  - Changed `gen()` to `random()`
  - Changed `thread_rng()` to `rng()`
  - Fixed RNG initialization and type mismatches
- Implemented SampleableDistribution trait for all distributions
- Added integration tests for statistical functionality
- Improved code quality with clippy fixes
- Enhanced documentation with examples

## See Also

Check the [TODO.md](./TODO.md) file for planned features and improvements.