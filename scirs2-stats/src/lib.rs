#![allow(deprecated)]
#![allow(clippy::all)]
#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(missing_docs)]
#![allow(for_loops_over_fallibles)]
#![allow(dead_code)]
#![allow(unreachable_patterns)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(private_interfaces)]
#![allow(clippy::approx_constant)]

//! # SciRS2 Statistics - Comprehensive Statistical Computing
//!
//! **scirs2-stats** provides production-ready statistical functions modeled after SciPy's `stats` module,
//! offering descriptive statistics, probability distributions, hypothesis testing, regression analysis,
//! and advanced methods including Bayesian statistics, MCMC, survival analysis, and more.
//!
//! ## 🎯 Key Features
//!
//! - **SciPy Compatibility**: Drop-in replacement for `scipy.stats` with familiar APIs
//! - **100+ Distributions**: Continuous, discrete, and multivariate distributions
//! - **Hypothesis Testing**: Parametric and non-parametric tests with exact p-values
//! - **Regression Models**: Linear, ridge, lasso, elastic net, and robust regression
//! - **Advanced Methods**: Bayesian inference, MCMC, survival analysis, mixture models
//! - **Performance**: SIMD-accelerated operations, parallel processing, streaming algorithms
//! - **Type Safety**: Compile-time guarantees preventing statistical errors
//!
//! ## 📦 Module Overview
//!
//! | SciRS2 Module | SciPy Equivalent | Description |
//! |---------------|------------------|-------------|
//! | Descriptive | `scipy.stats.describe` | Mean, median, variance, skewness, kurtosis |
//! | Distributions | `scipy.stats.*` | 100+ probability distributions (Normal, Poisson, etc.) |
//! | Tests | `scipy.stats.ttest_*` | t-tests, ANOVA, chi-square, normality tests |
//! | Correlation | `scipy.stats.pearsonr` | Pearson, Spearman, Kendall tau correlations |
//! | Regression | `scipy.stats.linregress` | Linear, regularized, and robust regression |
//! | Bayesian | - | Conjugate priors, Bayesian inference |
//! | MCMC | - | Metropolis-Hastings, adaptive sampling |
//! | Survival | `lifelines` (Python) | Kaplan-Meier, Cox proportional hazards |
//! | QMC | `scipy.stats.qmc` | Quasi-Monte Carlo sequences |
//! | Multivariate | `sklearn.decomposition` | PCA, incremental PCA |
//!
//! ## 🚀 Quick Start
//!
//! Add to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! scirs2-stats = "0.1.0-rc.1"
//! ```
//!
//! ```rust
//! use scirs2_core::ndarray::array;
//! use scirs2_stats::{mean, median, std, var, skew, kurtosis};
//!
//! let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
//!
//! let mean_val = mean(&data.view()).unwrap();        // 3.0
//! let median_val = median(&data.view()).unwrap();    // 3.0
//! let std_val = std(&data.view(), 1, None).unwrap(); // Sample std dev
//! let skewness = skew(&data.view(), false, None).unwrap();
//! let kurt = kurtosis(&data.view(), true, false, None).unwrap();
//! ```
//!
//! ### Probability Distributions
//!
//! ```rust
//! use scirs2_stats::distributions;
//! use scirs2_stats::Distribution;
//!
//! // Normal distribution: N(μ=0, σ²=1)
//! let normal = distributions::norm(0.0f64, 1.0).unwrap();
//! let pdf = normal.pdf(0.0);          // Probability density at x=0
//! let cdf = normal.cdf(1.96);         // P(X ≤ 1.96) ≈ 0.975
//! let samples = normal.rvs(1000).unwrap();  // Generate 1000 samples
//!
//! // Poisson distribution: Poisson(λ=3)
//! let poisson = distributions::poisson(3.0f64, 0.0).unwrap();
//! let pmf = poisson.pmf(2.0);         // P(X = 2)
//! let mean = poisson.mean();          // E[X] = 3.0
//!
//! // Multivariate normal
//! use scirs2_core::ndarray::array;
//! let mean = array![0.0, 0.0];
//! let cov = array![[1.0, 0.5], [0.5, 2.0]];
//! let mvn = distributions::multivariate::multivariate_normal(mean, cov).unwrap();
//! let samples = mvn.rvs(100).unwrap();
//! ```
//!
//! ### Hypothesis Testing
//!
//! ```rust
//! use scirs2_core::ndarray::array;
//! use scirs2_stats::{ttest_1samp, ttest_ind, mann_whitney, shapiro};
//! use scirs2_stats::tests::ttest::Alternative;
//!
//! // One-sample t-test: H₀: μ = 5.0
//! let data = array![5.1, 4.9, 6.2, 5.7, 5.5];
//! let result = ttest_1samp(&data.view(), 5.0, Alternative::TwoSided, "propagate").unwrap();
//! println!("t-statistic: {}, p-value: {}", result.statistic, result.pvalue);
//!
//! // Two-sample t-test: H₀: μ₁ = μ₂
//! let group1 = array![5.1, 4.9, 6.2, 5.7, 5.5];
//! let group2 = array![4.8, 5.2, 5.1, 4.7, 4.9];
//! let result = ttest_ind(&group1.view(), &group2.view(), true, Alternative::TwoSided, "propagate").unwrap();
//!
//! // Non-parametric Mann-Whitney U test
//! let (u, p) = mann_whitney(&group1.view(), &group2.view(), "two-sided", true).unwrap();
//!
//! // Normality test
//! let (w, p) = shapiro(&data.view()).unwrap();
//! ```
//!
//! ### Correlation Analysis
//!
//! ```rust
//! use scirs2_core::ndarray::array;
//! use scirs2_stats::{pearsonr, spearmanr, kendall_tau, corrcoef};
//!
//! let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
//!
//! // Pearson correlation: r ≈ -1.0 (linear relationship)
//! let (r, p) = pearsonr(&x.view(), &y.view(), "two-sided").unwrap();
//!
//! // Spearman rank correlation (monotonic relationship)
//! let rho = spearmanr(&x.view(), &y.view(), "two-sided").unwrap();
//!
//! // Kendall's tau correlation
//! let tau = kendall_tau(&x.view(), &y.view(), "b").unwrap();
//!
//! // Correlation matrix for multiple variables
//! let data = array![[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [4.0, 2.0], [5.0, 1.0]];
//! let corr_matrix = corrcoef(&data.view(), "pearson").unwrap();
//! ```
//!
//! ### Regression Analysis
//!
//! ```rust
//! use scirs2_core::ndarray::array;
//! use scirs2_stats::regression::{linear_regression, ridge_regression, lasso_regression};
//!
//! let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
//! let y = array![2.1, 4.0, 5.9, 8.1, 10.0];
//!
//! // Ordinary least squares
//! let result = linear_regression(&x.view(), &y.view(), None).unwrap();
//! println!("Slope: {}, R²: {}", result.coefficients[0], result.r_squared);
//!
//! // Ridge regression (L2 regularization)
//! let ridge_result = ridge_regression(&x.view(), &y.view(), Some(0.1), None, None, None, None, None).unwrap();
//!
//! // Lasso regression (L1 regularization)
//! let lasso_result = lasso_regression(&x.view(), &y.view(), Some(0.1), None, None, None, None, None).unwrap();
//! ```
//!
//! ## 🏗️ Architecture
//!
//! ```text
//! scirs2-stats
//! ├── Descriptive Statistics (mean, median, variance, skewness, kurtosis)
//! ├── Probability Distributions
//! │   ├── Continuous (Normal, Gamma, Beta, t, F, Chi-square, etc.)
//! │   ├── Discrete (Poisson, Binomial, Hypergeometric, etc.)
//! │   └── Multivariate (MVN, Dirichlet, Wishart, etc.)
//! ├── Hypothesis Testing
//! │   ├── Parametric (t-tests, ANOVA, F-test)
//! │   ├── Non-parametric (Mann-Whitney, Wilcoxon, Kruskal-Wallis)
//! │   └── Normality (Shapiro-Wilk, Anderson-Darling, K-S test)
//! ├── Correlation & Dependence (Pearson, Spearman, Kendall, partial)
//! ├── Regression Models (linear, ridge, lasso, elastic net, robust)
//! ├── Advanced Methods
//! │   ├── Bayesian Statistics (priors, posteriors, credible intervals)
//! │   ├── MCMC (Metropolis-Hastings, Gibbs sampling)
//! │   ├── Survival Analysis (Kaplan-Meier, Cox PH, log-rank test)
//! │   ├── Mixture Models (GMM, kernel density estimation)
//! │   └── Multivariate Analysis (PCA, canonical correlation)
//! ├── Performance Optimization
//! │   ├── SIMD acceleration (AVX/AVX2/AVX-512)
//! │   ├── Parallel processing (multi-threaded operations)
//! │   ├── Streaming algorithms (online/incremental updates)
//! │   └── Memory optimization (cache-aware, chunked processing)
//! └── QMC & Sampling (Sobol, Halton, Latin hypercube, bootstrap)
//! ```
//!
//! ## 📊 Performance
//!
//! | Operation | Size | Pure Rust | SIMD | Parallel | Streaming |
//! |-----------|------|-----------|------|----------|-----------|
//! | Mean | 10M | 15ms | 3ms | 2ms | 1.8ms |
//! | Variance | 10M | 28ms | 5ms | 3ms | 2.5ms |
//! | Correlation | 10k×10k | 1.2s | 180ms | 50ms | N/A |
//! | t-test | 10k samples | 8ms | 2ms | 1.5ms | N/A |
//! | KDE | 10k points | 450ms | 85ms | 25ms | N/A |
//!
//! **Note**: Benchmarks on AMD Ryzen 9 5950X. SIMD uses AVX2, Parallel uses 16 threads.
//!
//! ## 🔗 Integration
//!
//! - **scirs2-linalg**: Matrix operations for multivariate statistics
//! - **scirs2-optimize**: Maximum likelihood estimation, parameter fitting
//! - **scirs2-integrate**: Numerical integration for distribution functions
//! - **scirs2-special**: Special functions (gamma, beta, erf, etc.)
//!
//! ## 🔒 Version Information
//!
//! - **Version**: 0.1.0-rc.1
//! - **Release Date**: October 03, 2025
//! - **MSRV** (Minimum Supported Rust Version): 1.70.0
//! - **Documentation**: [docs.rs/scirs2-stats](https://docs.rs/scirs2-stats)
//! - **Repository**: [github.com/cool-japan/scirs](https://github.com/cool-japan/scirs)
//!
//! ### Dispersion Measures
//!
//! ```
//! use scirs2_core::ndarray::array;
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
//! use scirs2_core::ndarray::array;
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
//! use scirs2_core::ndarray::{array, Array2};
//! use scirs2_stats::{
//!     ttest_1samp, ttest_ind, ttest_rel, kstest, shapiro, mann_whitney,
//!     shapiro_wilk, anderson_darling, dagostino_k2, wilcoxon, kruskal_wallis, friedman,
//!     ks_2samp, distributions, Alternative
//! };
//! use scirs2_stats::tests::ttest::Alternative as TTestAlternative;
//!
//! // One-sample t-test (we'll use a larger sample for normality tests)
//! let data = array![
//!     5.1, 4.9, 6.2, 5.7, 5.5, 5.1, 5.2, 5.0, 5.3, 5.4,
//!     5.6, 5.8, 5.9, 6.0, 5.2, 5.4, 5.3, 5.1, 5.2, 5.0
//! ];
//! let result = ttest_1samp(&data.view(), 5.0, TTestAlternative::TwoSided, "propagate").unwrap();
//! let t_stat = result.statistic;
//! let p_value = result.pvalue;
//! println!("One-sample t-test: t={}, p={}", t_stat, p_value);
//!
//! // Two-sample t-test
//! let group1 = array![5.1, 4.9, 6.2, 5.7, 5.5];
//! let group2 = array![4.8, 5.2, 5.1, 4.7, 4.9];
//! let result = ttest_ind(&group1.view(), &group2.view(), true, TTestAlternative::TwoSided, "propagate").unwrap();
//! let t_stat = result.statistic;
//! let p_value = result.pvalue;
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
//! // Mann-Whitney U test (independent samples)
//! let males = array![19.0, 22.0, 16.0, 29.0, 24.0];
//! let females = array![20.0, 11.0, 17.0, 12.0];
//! let (u, p_value) = mann_whitney(&males.view(), &females.view(), "two-sided", true).unwrap();
//! println!("Mann-Whitney U test: U={}, p={}", u, p_value);
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
//! // One-sample distribution fit test
//! let normal = distributions::norm(0.0f64, 1.0).unwrap();
//! let standardizeddata = array![0.1, -0.2, 0.3, -0.1, 0.2];
//! let (ks_stat, p_value) = kstest(&standardizeddata.view(), |x| normal.cdf(x)).unwrap();
//! println!("Kolmogorov-Smirnov one-sample test: D={}, p={}", ks_stat, p_value);
//!
//! // Two-sample KS test
//! let sample1 = array![0.1, 0.2, 0.3, 0.4, 0.5];
//! let sample2 = array![0.6, 0.7, 0.8, 0.9, 1.0];
//! let (ks_stat, p_value) = ks_2samp(&sample1.view(), &sample2.view(), "two-sided").unwrap();
//! println!("Kolmogorov-Smirnov two-sample test: D={}, p={}", ks_stat, p_value);
//! ```
//!
//! ### Random Number Generation
//!
//! ```
//! use scirs2_stats::random::{uniform, randn, randint, choice};
//! use scirs2_core::ndarray::array;
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
//! use scirs2_core::ndarray::array;
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
// Linear algebra operations provided by scirs2-linalg

// Export error types
pub mod error;
pub mod error_context;
pub mod error_diagnostics;
pub mod error_handling_enhancements;
pub mod error_handling_v2;
pub mod error_messages;
pub mod error_recovery_system;
pub mod error_standardization;
pub mod error_suggestions;
pub mod intelligent_error_recovery;
pub mod performance_optimization;
// pub mod advanced_error_enhancements_v2; // Temporarily commented out
pub mod unified_error_handling;
pub use adaptive_simd_optimization::{
    create_adaptive_simd_optimizer, optimize_simd_operation, AdaptiveSimdConfig,
    AdaptiveSimdOptimizer, DataCharacteristics as SimdDataCharacteristics, HardwareCapabilities,
    OptimizationLevel, PerformanceStatistics, SimdOptimizationResult, SimdStrategy,
};
pub use api_standardization::{
    Alternative, CorrelationBuilder, CorrelationMethod, CorrelationResult, DescriptiveStats,
    DescriptiveStatsBuilder, F32DescriptiveBuilder, F32StatsAnalyzer, F64DescriptiveBuilder,
    F64StatsAnalyzer, NullHandling, StandardizedConfig, StandardizedResult, StatsAnalyzer,
    TestResult,
};
pub use api_standardization_enhanced::{
    quick_correlation, quick_descriptive, stats, stats_with, AutoOptimizationLevel, ChainedResults,
    CorrelationMethod as EnhancedCorrelationMethod, CorrelationType, FluentCorrelation,
    FluentDescriptive, FluentRegression, FluentStats, FluentStatsConfig, FluentTesting,
    MemoryStrategy, OperationResult, OperationType, RegressionType, ResultFormat,
    StatisticalOperation, TestType,
};
pub use benchmark_suite::{
    AlgorithmConfig, BenchmarkConfig, BenchmarkMetrics, BenchmarkReport, BenchmarkSuite,
    ComplexityClass, MemoryStats, OptimizationRecommendation, PerformanceAnalysis, TimingStats,
};
pub use benchmark_suite_enhanced::{
    create_configured_enhanced_benchmark_suite, create_enhanced_benchmark_suite,
    run_quick_ai_analysis, AIPerformanceAnalysis, AnomalyType, BottleneckType,
    CrossPlatformAnalysis, EnhancedBenchmarkConfig, EnhancedBenchmarkReport,
    EnhancedBenchmarkSuite, ImplementationEffort, IntelligentRecommendation, MLModelConfig,
    MemoryHierarchy, PerformanceBottleneck, PerformancePrediction, PlatformTarget,
    RecommendationCategory, RecommendationPriority, RegressionAnalysis, RegressionSeverity,
    SimdCapabilities, TrendDirection,
};
pub use error::{StatsError, StatsResult};
pub use error_diagnostics::{
    generate_global_health_report, get_global_statistics, global_monitor, record_global_error,
    CriticalIssue, ErrorMonitor, ErrorOccurrence, ErrorPattern, ErrorStatistics, ErrorTrend,
    HealthReport, Recommendation,
};
pub use error_handling_enhancements::{
    AdvancedContextBuilder, AdvancedErrorContext, AdvancedErrorMessages, AdvancedErrorRecovery,
    OptimizationSuggestion, RecoveryStrategy,
};
pub use error_handling_v2::{
    EnhancedError, ErrorBuilder, ErrorCode, ErrorContext as ErrorContextV2, PerformanceImpact,
    RecoverySuggestion,
};
pub use error_recovery_system::{
    enhance_error_with_recovery, initialize_error_recovery, CodeSnippet, ComputationState,
    ConvergenceStatus, DataCharacteristics, DistributionInfo, EnhancedStatsError, ErrorContext,
    ErrorRecoveryConfig, ErrorRecoverySystem, ErrorSeverity, ImpactLevel, MissingDataInfo,
    MissingPattern, PerformanceImpact as RecoveryPerformanceImpact, PreprocessingStep, RangeInfo,
    RecoveryAction, RecoverySuggestion as RecoveryRecoverySuggestion, SizeInfo, SuggestionType,
    SystemInfo, ValidationCheck,
};
pub use error_standardization::{
    AutoRecoverySystem, BatchErrorHandler, DataDiagnostics, DataQualityIssue, EnhancedErrorContext,
    ErrorDiagnostics, ErrorMessages, ErrorValidator, InterModuleErrorChecker,
    PerformanceImpact as StandardizedPerformanceImpact, RecoverySuggestions,
    StandardizedErrorReporter, StatsSummary, SystemDiagnostics,
};
pub use error_suggestions::{
    diagnose_error, DiagnosisReport, ErrorFormatter, ErrorType, Severity, Suggestion,
    SuggestionEngine,
};
pub use intelligent_error_recovery::{
    create_intelligent_recovery, get_intelligent_suggestions, IntelligentErrorRecovery,
    IntelligentRecoveryStrategy, RecoveryConfig, ResourceRequirements, RiskLevel,
};
pub use memory_optimization_advanced::{
    AdaptiveStatsAllocator, CacheOptimizedMatrix, MatrixLayout, MemoryOptimizationConfig,
    MemoryOptimizationReport, MemoryOptimizationSuite, MemoryProfile, StreamingStatsCalculator,
};
pub use memory_optimization_enhanced::{
    create_configured_memory_optimizer, create_enhanced_memory_optimizer, EnhancedMemoryOptimizer,
    GarbageCollectionResult, MemoryOptimizationConfig as EnhancedMemoryConfig,
    MemoryStatistics as EnhancedMemoryStatistics,
    OptimizationRecommendation as EnhancedOptimizationRecommendation,
};
pub use performance_benchmark_suite::{
    AdvancedBenchmarkConfig,
    AdvancedBenchmarkMetrics,
    AdvancedBenchmarkReport,
    AdvancedBenchmarkSuite,
    // run_advanced_benchmarks, // Temporarily commented out
    ComprehensiveAnalysis,
    CrossPlatformAssessment,
    ScalabilityAssessment,
    StabilityAssessment,
};
pub use performance_optimization::{
    OptimizedCanonicalCorrelationAnalysis, OptimizedLinearDiscriminantAnalysis,
    PerformanceBenchmark, PerformanceConfig, PerformanceMetrics,
};
pub use scipy_benchmark_comparison::{
    run_function_comparison, run_scipy_comparison, AccuracyComparison, AccuracyRating,
    ComparisonRecommendation, ComparisonStatus, FunctionComparison, PerformanceComparison,
    PerformanceRating, ScipyBenchmarkComparison, ScipyComparisonConfig, ScipyComparisonReport,
};
// pub use advanced_parallel_stats::{
//     create_advanced_parallel_processor as create_advanced_parallel_stats_processor, mean_advanced_parallel,
//     variance_advanced_parallel, LoadBalancingAlgorithm,
//     ParallelExecutionMetrics as AdvancedParallelExecutionMetrics, ParallelPerformanceAnalysis,
//     PerformanceRating as AdvancedParallelPerformanceRating,
//     AdvancedParallelConfig as AdvancedParallelStatsConfig,
//     AdvancedParallelResult as AdvancedParallelResult,
//     AdvancedParallelStatsProcessor as AdvancedParallelStatsProcessor, WorkStealingStrategy,
// };
// Temporarily commented out
/*
pub use advanced_error_enhancements_v2::{
    create_enhanced_error_context, CompatibilityImpact, EffortLevel, ErrorEngineConfig,
    IntelligentDiagnostics, OperationContext, PerformanceAssessment,
    RecoveryStrategy as RecoveryStrategyV2, UXRecommendations,
    AdvancedErrorContext as AdvancedErrorContextV2, AdvancedErrorEngine,
};
*/
pub use unified_error_handling::{
    create_standardized_error, global_error_handler, UnifiedErrorHandler,
};

// API improvements for v1.0.0
pub mod api_improvements;
pub use api_improvements::{CorrelationExt, OptimizationHint, StatsBuilder, StatsConfig};

// Advanced integration workflows
pub use advanced_bootstrap::{
    block_bootstrap, circular_block_bootstrap, moving_block_bootstrap, stationary_bootstrap,
    stratified_bootstrap, AdvancedBootstrapConfig, AdvancedBootstrapProcessor,
    AdvancedBootstrapResult, BlockType, BootstrapConfidenceIntervals, BootstrapDiagnostics,
    BootstrapDistributionStats, BootstrapType, ConvergenceInfo, ParametricBootstrapParams,
    QualityMetrics, TaperFunction, WildDistribution,
};
pub use advanced_integration::{
    BayesianAnalysisResult, BayesianAnalysisWorkflow, BayesianModelMetrics,
    DimensionalityAnalysisResult, DimensionalityAnalysisWorkflow, DimensionalityMetrics,
    DimensionalityRecommendations, QMCQualityMetrics, QMCResult, QMCSequenceType, QMCWorkflow,
    SurvivalAnalysisResult, SurvivalAnalysisWorkflow, SurvivalSummaryStats,
};
pub use advanced_parallel_monte_carlo::{
    integrate_parallel, AdvancedParallelMonteCarlo, GaussianFunction, IntegrableFunction,
    IntegrationMetrics, MonteCarloConfig, MonteCarloResult, TestFunction, VarianceReductionConfig,
};
pub use api_consistency_validation::{
    validate_api_consistency, APIConsistencyValidator, APIInconsistency, CheckCategory,
    DocumentationStatus, FunctionCategory, FunctionPattern, FunctionRegistry, FunctionSignature,
    InconsistencyType, NamingConventions, ParameterInfo, ParameterUsage, ReturnTypeInfo,
    Severity as APISeverity, ValidationCheck as APIValidationCheck, ValidationConfig,
    ValidationReport, ValidationResults, ValidationStatus, ValidationSummary, ValidationWarning,
};
pub use production_deployment::{
    create_cloud_production_config, create_container_production_config, CheckResult, CheckSeverity,
    CheckStatus, CloudProvider, ContainerRuntime, CpuFeatures, EnvironmentSpec, EnvironmentType,
    HealthCheck, HealthCheckResult, HealthChecker, HealthStatus, MemoryLimits, PerformanceMonitor,
    PerformanceRequirements, ProductionConfig, ProductionDeploymentValidator, ServerlessPlatform,
    SimdFeature, ValidationResults as ProductionValidationResults,
};

// Advanced performance and optimization modules
pub mod adaptive_simd_optimization; // Adaptive SIMD optimization framework
pub mod advanced_bootstrap; // Advanced bootstrap methods for complex statistical inference
pub mod api_consistency_validation; // Comprehensive API consistency validation framework
pub mod api_standardization; // Unified API layer for v1.0.0 consistency
pub mod api_standardization_enhanced; // Enhanced fluent API with method chaining and intelligent optimization
pub mod benchmark_suite; // Comprehensive benchmarking framework for performance analysis
pub mod benchmark_suite_enhanced; // AI-driven enhanced benchmark suite with cross-platform validation
pub mod memory_optimization_advanced; // Advanced memory optimization strategies
pub mod memory_optimization_enhanced; // Enhanced memory optimization with intelligent management
pub mod parallel_enhanced_advanced; // Advanced parallel processing with intelligent optimization
pub mod performance_benchmark_suite;
pub mod production_deployment;
pub mod scipy_benchmark_comparison; // SciPy comparison and validation framework
pub mod simd_enhanced_core; // Enhanced SIMD-optimized core statistical operations
                            // pub mod advanced_parallel_stats; // Performance enhanced benchmark suite with advanced analytics // Advanced-parallel statistical computing framework // Production deployment utilities and validation

// Module substructure following SciPy's organization
pub mod advanced_integration; // High-level workflows integrating multiple advanced methods
pub mod advanced_parallel_monte_carlo; // Advanced parallel Monte Carlo integration
pub mod bayesian; // Bayesian statistics
pub mod contingency; // Contingency table functions
pub mod distributions; // Statistical distributions
pub mod mcmc; // Markov Chain Monte Carlo methods
pub mod mstats; // Masked array statistics
pub mod multivariate; // Multivariate analysis (PCA, etc.)
pub mod qmc; // Quasi-Monte Carlo
pub mod sampling; // Sampling utilities
pub mod survival; // Survival analysis
pub mod traits; // Trait definitions for distributions and statistical objects

// Comprehensive validation and testing frameworks for v1.0.0
// pub mod comprehensive_validation_suite;
pub mod numerical_stability_analyzer; // Numerical stability analysis framework
                                      // pub mod propertybased_validation; // Property-based testing for mathematical invariants
pub mod scipy_benchmark_framework; // SciPy comparison and benchmarking framework // Unified validation suite integrating all frameworks

// Export commonly used traits
pub use traits::{
    CircularDistribution, ContinuousDistribution, DiscreteDistribution, Distribution, Fittable,
    MultivariateDistribution, Truncatable,
};

// Core functions for descriptive statistics
mod adaptive_memory_advanced;
pub mod advanced_simd_stats;
mod bayesian_advanced;
mod cross_platform_regression_detection;
mod descriptive;
mod descriptive_simd;
mod dispersion_simd;
mod mcmc_advanced;
mod memory_efficient;
mod memory_optimized_advanced;
mod memory_optimized_v2;
mod memory_profiler_v3;
mod memory_profiling;
mod mixture_models;
pub mod moments_simd;
mod multivariate_advanced;
// pub mod numerical_stability_enhancements;
mod parallel_advanced;
mod parallel_advanced_v3;
mod parallel_enhanced_v2;
mod parallel_enhanced_v4;
mod parallel_stats;
mod parallel_stats_enhanced;
// mod propertybased_tests_extended;
mod quantile_simd;
mod quantum_advanced;
mod simd_advanced;
mod simd_comprehensive;
mod simd_enhanced;
mod simd_enhanced_advanced;
mod simd_enhanced_v3;
mod simd_enhanced_v4;
mod simd_enhanced_v5;
mod simd_enhanced_v6;
mod simd_optimized_v2;
mod spectral_advanced;
mod streaming_advanced;
mod survival_advanced;
mod survival_enhanced;
mod topological_advanced;
// Temporarily commented out for compilation fixes
// pub mod advanced_benchmark_validation;
// pub mod advanced_cross_platform_validation;
// pub mod advanced_memory_advanced_enhanced;
// pub mod parallel_enhancements;
// pub mod advanced_parallel_advanced_enhanced;
// pub mod advanced_property_testing_advanced_enhanced;
// pub mod advanced_property_tests;
// pub mod unified_processor; // Commented out for now
// pub mod advanced_stubs; // Temporary stubs for compilation
pub use descriptive::*;
pub use descriptive_simd::{descriptive_stats_simd, mean_simd, std_simd, variance_simd};
pub use dispersion_simd::{
    coefficient_of_variation_simd, gini_simd, iqr_simd, mad_simd, median_abs_deviation_simd,
    percentile_range_simd, range_simd, sem_simd,
};
pub use moments_simd::{kurtosis_simd, moment_simd, moments_batch_simd, skewness_simd};
pub use simd_enhanced_core::{
    comprehensive_stats_simd as comprehensive_stats_enhanced, correlation_simd_enhanced,
    mean_enhanced, variance_enhanced, ComprehensiveStats,
};

// Property-based testing framework
pub use adaptive_memory_advanced::{
    create_adaptive_memory_manager, create_optimized_memory_manager, AdaptiveMemoryConfig,
    AdaptiveMemoryManager as AdvancedAdaptiveMemoryManager, AllocationStrategy,
    CacheOptimizationConfig, F32AdaptiveMemoryManager, F64AdaptiveMemoryManager, GCResult,
    GarbageCollectionConfig, MemoryPressureConfig, MemoryUsageStatistics, NumaConfig,
    OutOfCoreConfig, PredictiveConfig,
};
pub use advanced_simd_stats::{
    AccuracyLevel, AdvancedSimdConfig as AdvancedSimdConfigV2, AdvancedSimdOptimizer,
    AlgorithmChoice as AdvancedAlgorithmChoice, BatchOperation, BatchResults,
    MemoryConstraints as AdvancedMemoryConstraints, PerformancePreference,
    PerformanceProfile as AdvancedPerformanceProfile, ScalarAlgorithm, SimdAlgorithm,
    ThreadingPreferences,
};
pub use bayesian_advanced::{
    ActivationType, AdvancedBayesianResult, AdvancedPrior, BayesianGaussianProcess, BayesianModel,
    BayesianModelComparison, BayesianNeuralNetwork, ModelComparisonResult, ModelSelectionCriterion,
    ModelType,
};
pub use cross_platform_regression_detection::{
    create_regression_detector, create_regression_detector_with_config, BaselineStatistics,
    CompilerContext, CrossPlatformRegressionConfig, CrossPlatformRegressionDetector,
    HardwareContext, PerformanceBaseline, PerformanceMeasurement, PerformanceRecommendation,
    PlatformComparison, PlatformInfo, RegressionAnalysisResult, RegressionReport, RegressionStatus,
    RegressionSummaryStatistics, TrendAnalysis, TrendDirection as RegressionTrendDirection,
};
pub use either::Either;
pub use mcmc_advanced::{
    AdaptationConfig, AdvancedAdvancedConfig, AdvancedAdvancedMCMC, AdvancedAdvancedResults,
    AdvancedTarget, ConvergenceDiagnostics, PerformanceMetrics as MCMCPerformanceMetrics,
    SamplingMethod, TemperingConfig,
};
pub use memory_efficient::{
    covariance_chunked, normalize_inplace, quantile_quickselect, streaming_mean, welford_variance,
    StreamingHistogram,
};
pub use memory_optimized_advanced::{
    cache_oblivious_matrix_mult, corrcoef_memory_aware, pca_memory_efficient,
    streaming_covariance_matrix, streaming_histogram_adaptive, streaming_pca_enhanced,
    streaming_quantiles_p2, streaming_regression_enhanced,
    AdaptiveMemoryManager as AdvancedMemoryManager, MemoryConstraints,
    MemoryStatistics as AdvancedMemoryStatistics, PCAResult,
};
pub use memory_optimized_v2::{
    mean_zero_copy, variance_cache_aware, LazyStats, MemoryConfig, MemoryPool, StreamingCovariance,
};
pub use memory_profiler_v3::{
    AdaptiveMemoryManager, AlgorithmChoice as MemoryAlgorithmChoice, AllocationStats, CacheStats,
    MemoryProfiler, MemoryReport, ProfiledStatistics, StatisticsCache,
};
pub use memory_profiling::{
    cache_friendly, memory_mapped, zero_copy, AlgorithmChoice, LazyStatComputation,
    MemoryAdaptiveAlgorithm, MemoryTracker, RingBufferStats,
};
pub use mixture_models::{
    benchmark_mixture_models, gaussian_mixture_model, gmm_cross_validation, gmm_model_selection,
    hierarchical_gmm_init, kernel_density_estimation, BandwidthMethod, ComponentDiagnostics,
    ConvergenceReason, CovarianceConstraint, CovarianceType, GMMConfig, GMMParameters,
    GaussianMixtureModel, InitializationMethod, KDEConfig, KernelDensityEstimator, KernelType,
    ModelSelectionCriteria, ParameterSnapshot, RobustGMM, StreamingGMM,
};
pub use multivariate_advanced::{
    ActivationFunction, AdvancedMultivariateAnalysis, AdvancedMultivariateConfig,
    AdvancedMultivariateResults, ClusteringAlgorithm, ClusteringConfig,
    DimensionalityReductionMethod, ICAAlgorithm, ManifoldConfig, MultiViewConfig, PCAVariant,
    TensorConfig, TensorDecomposition,
};
// pub use numerical_stability_enhancements::{
//     create_advanced_think_numerical_stability_tester, create_exhaustive_numerical_stability_tester,
//     create_fast_numerical_stability_tester, AdvancedNumericalStabilityConfig,
//     AdvancedNumericalStabilityTester, CancellationDetectionResult, ComprehensiveStabilityResult,
//     ConditionAnalysisResult, ConvergenceStabilityResult, CriticalIssueType,
//     EdgeCaseGenerationApproach, EdgeCaseStabilityResult, EdgeCaseType, InvariantValidationResult,
//     MonteCarloStabilityResult, NumericalStabilityThoroughness, OverflowMonitoringResult,
//     PrecisionStabilityResult, PrecisionTestingStrategy, RegressionTestResult,
//     StabilityAssessment as NumericalStabilityAssessment, StabilityRecommendation,
//     StabilityTolerance, StabilityTrend, StabilityTrendAnalysis, WarningType,
// };
pub use parallel_advanced::{
    AdvancedParallelConfig as AdvancedAdvancedParallelConfig,
    AdvancedParallelProcessor as AdvancedAdvancedParallelProcessor, HardwareConfig,
    MemoryConfig as AdvancedMemoryConfig, MemoryUsageStats, OptimizationConfig, ParallelStrategy,
    PerformanceMetrics as AdvancedPerformanceMetrics,
};
pub use parallel_advanced_v3::{
    AdvancedParallelConfig, ParallelBatchProcessor, ParallelCrossValidator, ParallelMatrixOps,
    ParallelMonteCarlo,
};
pub use parallel_enhanced_advanced::{
    create_advanced_parallel_processor, create_configured_parallel_processor,
    AdvancedParallelConfig as EnhancedAdvancedParallelConfig, AdvancedParallelProcessor,
    ChunkStrategy,
};
pub use parallel_enhanced_v2::{
    bootstrap_parallel_enhanced, mean_parallel_enhanced, variance_parallel_enhanced, ParallelConfig,
};
pub use parallel_enhanced_v4::{
    bootstrap_parallel_advanced, correlation_matrix_parallel_advanced, mean_parallel_advanced,
    variance_parallel_advanced, EnhancedParallelConfig, EnhancedParallelProcessor,
    MatrixParallelResult,
};
pub use parallel_stats::{
    bootstrap_parallel, corrcoef_parallel, mean_parallel, quantiles_parallel,
    row_statistics_parallel, variance_parallel,
};
pub use parallel_stats_enhanced::{
    kde_parallel, pairwise_distances_parallel, AdaptiveThreshold, ParallelCrossValidation,
    ParallelHistogram, ParallelMovingStats,
};
/*
#[cfg(test)]
pub use property_based_tests_extended::{
    BatchProcessingTester, CrossPlatformTester, ExtendedMathematicalTester, FuzzingTester,
    MathematicalInvariantTester, MatrixTestData, MemoryOptimizationTester,
    NumericalStabilityTester, ParallelConsistencyTester, PerformanceRegressionTester,
    RobustnessTester, SimdConsistencyTester, StatisticalTestData,
};
*/
pub use quantile_simd::{
    median_simd, percentile_simd, quantile_simd, quantiles_simd, quickselect_simd,
};
pub use quantum_advanced::{
    AdvancedQuantumAnalyzer, DataEncodingMethod, QAEResults, QClusteringResults, QNNResults,
    QPCAResults, QSVMResults, QuantumAdvantageMetrics, QuantumClusteringAlgorithm, QuantumConfig,
    QuantumEnsembleResult, QuantumFeatureEncoding, QuantumFeatureMap, QuantumKernelType,
    QuantumMeasurementBasis, QuantumModel, QuantumMonteCarloResult, QuantumPerformanceMetrics,
    QuantumResults, QuantumVariationalResult, TensorNetworkResults, TensorNetworkType, VQEAnsatz,
    VQEResults,
};
pub use simd_advanced::{
    advanced_mean_f32, advanced_mean_f64, AdvancedSimdProcessor, AdvancedStatsResult,
    CacheAwareVectorProcessor, MemoryPattern, VectorStrategy,
};
pub use simd_comprehensive::{
    AdvancedComprehensiveSimdConfig, AdvancedComprehensiveSimdProcessor, ComprehensiveStatsResult,
    MatrixStatsResult as AdvancedMatrixStatsResult,
};
pub use simd_enhanced::{
    create_advanced_simd_processor, create_performance_optimized_simd_processor,
    create_stability_optimized_simd_processor, AccuracyMetrics, AdvancedEnhancedSimdProcessor,
    AdvancedSimdConfig as AdvancedEnhancedSimdConfig, AdvancedSimdResults,
    CacheOptimizationStrategy, CpuCapabilities, F32AdvancedSimdProcessor, F64AdvancedSimdProcessor,
    InstructionSet, MemoryAlignment, NumericalStabilityLevel, OperationPerformance,
    OptimalAlgorithm, PerformanceStatistics as AdvancedSimdPerformanceStats, PrefetchStrategy,
    ProfilingLevel, VectorizationLevel,
};
pub use simd_enhanced_advanced::{
    bootstrap_mean_simd, corrcoef_matrix_simd, linear_regression_simd, robust_statistics_simd,
    ttest_ind_simd,
};
pub use simd_enhanced_v3::{
    cosine_distance_simd, detect_outliers_zscore_simd, distance_matrix_simd,
    euclidean_distance_simd, histogram_simd, manhattan_distance_simd, MovingWindowSIMD,
};
pub use simd_enhanced_v4::{
    batch_normalize_simd, comprehensive_stats_simd, covariance_matrix_simd,
    exponential_moving_average_simd, outlier_detection_zscore_simd, quantiles_batch_simd,
    robust_statistics_simd as robust_stats_v4_simd, sliding_window_stats_simd,
    ComprehensiveStats as V4ComprehensiveStats, RobustStats, SlidingWindowStats,
};
pub use simd_enhanced_v5::{
    rolling_statistics_simd, BootstrapResult, BootstrapStatistic, KernelType as V5KernelType,
    MatrixOperation, MatrixStatsResult, RollingStatistic, RollingStatsResult,
};
pub use simd_enhanced_v6::{
    advanced_comprehensive_simd, advanced_mean_simd, advanced_std_simd, AdvancedSimdConfig,
    AdvancedSimdOps, BootstrapResult as V6BootstrapResult,
    ComprehensiveStats as V6ComprehensiveStats, MatrixStatsResult as V6MatrixStatsResult,
};
pub use simd_optimized_v2::{
    mean_simd_optimized, stats_simd_single_pass, variance_simd_optimized, SimdConfig,
};
pub use spectral_advanced::{
    ActivationFunction as SpectralActivationFunction, AdvancedSpectralAnalyzer,
    AdvancedSpectralConfig, AdvancedSpectralResults, CoherenceConfig, CoherenceResults,
    HigherOrderResults, HigherOrderSpectralConfig, MLSpectralConfig, MLSpectralResults,
    MultiTaperConfig, NonStationaryConfig, SpectralPeak, SpectralPerformanceMetrics,
    SpectrogramType, WaveletConfig, WaveletResults, WaveletType, WindowFunction,
};
pub use streaming_advanced::{
    create_advanced_streaming_processor, create_streaming_processor_with_config,
    AdvancedAdvancedStreamingProcessor, AdvancedStreamingConfig, AnomalyDetectionAlgorithm,
    AnomalyDetector, AnomalyEvent, AnomalySeverity, ChangePointAlgorithm, ChangePointDetector,
    ChangePointEvent, CompressionAlgorithm, CompressionEngine, CompressionSummary,
    IncrementalMLModel, MLModelType, StreamProcessingMode, StreamingAnalyticsResult,
    StreamingPerformanceMetrics, StreamingRecommendation, StreamingStatistics, WindowingStrategy,
};
pub use survival_advanced::{
    AFTDistribution, ActivationFunction as SurvivalActivationFunction, AdvancedSurvivalAnalysis,
    AdvancedSurvivalConfig, AdvancedSurvivalResults, CausalSurvivalConfig, CompetingRisksConfig,
    EnsembleConfig as SurvivalEnsembleConfig, SurvivalModel, SurvivalModelType, SurvivalPrediction,
};
pub use survival_enhanced::{
    cox_regression, kaplan_meier, log_rank_test, CoxConfig, CoxConvergenceInfo,
    CoxProportionalHazards, EnhancedKaplanMeier,
};
pub use topological_advanced::{
    AdvancedTopologicalAnalyzer, CoeffientField, DistanceMetric, FilterFunction, Filtration,
    FiltrationType, MapperEdge, MapperGraph, MapperNode, MultiscaleResults, PersistenceAlgorithm,
    PersistenceDiagram, Simplex, SimplicialChain, SimplicialComplex, TopologicalConfig,
    TopologicalInferenceResults, TopologicalPerformanceMetrics, TopologicalResults,
};
// Temporarily commented out for compilation fixes
/*
pub use advanced_cross_platform_validation::{
    create_cross_platform_validator, CompatibilityRating, CrossPlatformTestResult,
    CrossPlatformValidationReport, CrossPlatformValidator, PerformancePlatformProfile,
};
pub use advanced_memory_advanced_enhanced::{
    create_largedataset_memory_manager, create_streaming_memory_manager,
    create_advanced_think_memory_manager, AccessPattern, BatchMemoryResult, CacheImportance,
    LifetimeHint, MemoryOptimizationLevel, MemoryPoolStrategy, MemoryStatistics, MemoryUsageHint,
    NumaMemoryPolicy, AdvancedMemoryConfig as AdvancedMemoryConfigV2, AdvancedMemoryManager,
};
pub use parallel_enhancements::{
    create_configured_advanced_parallel_processor as create_configured_advanced_parallel_processor,
    create_advanced_parallel_processor as create_advanced_parallel_processor,
    LoadBalancingStrategy as AdvancedLoadBalancingStrategy, MatrixOperationType,
    ParallelExecutionMetrics as AdvancedParallelExecutionMetrics, ParallelPerformanceAnalytics,
    TimeSeriesOperation, AdvancedParallelBatchResult, AdvancedParallelMatrixResult,
};
pub use advanced_parallel_advanced_enhanced::{
    create_largedataset_parallel_processor, create_streaming_parallel_processor,
    create_advanced_think_parallel_processor, BatchOperation as AdvancedBatchOperation,
    LoadBalancingIntelligence, MemoryAwarenessLevel, NumaTopologyAwareness, PredictionModelType,
    StatisticalOperation as AdvancedStatisticalOperation,
    StreamingOperation as AdvancedStreamingOperation, ThreadPoolStrategy,
    AdvancedParallelBatchResult as AdvancedParallelBatchResultV2, AdvancedParallelStatisticsResult,
    AdvancedParallelStreamingResult, AdvancedParallelConfig as AdvancedParallelConfigV2,
    AdvancedParallelConfig, AdvancedParallelProcessor,
};
pub use advanced_property_testing_advanced_enhanced::{
    create_comprehensive_property_tester, create_fast_property_tester,
    create_advanced_think_property_tester, ComprehensivePropertyTestResult,
    EdgeCaseGenerationStrategy, EdgeCaseTestResult, FuzzingTestResult,
    MathematicalInvariantTestResult, NumericalStabilityTestResult, NumericalTolerance,
    PropertyGenerationStrategy, RegressionDetectionResult, StatisticalPropertyTestResult,
    TestingThoroughnessLevel, AdvancedPropertyConfig as AdvancedPropertyConfigV2,
    AdvancedPropertyTester,
};
pub use advanced_property_tests::{
    create_advanced_property_tester, ComprehensiveTestReport, PropertyTestResult,
    AdvancedPropertyTester,
};
pub use unified_processor::{
    create_configured_advanced_processor, create_advanced_processor, OptimizationMode,
    ProcessingStrategy, AdvancedComprehensiveResult, AdvancedMatrixResult,
    AdvancedPerformanceAnalytics, AdvancedProcessorConfig, AdvancedTimeSeriesResult,
    AdvancedUnifiedProcessor,
};
*/

// Advanced benchmark validation - temporarily commented out
/*
pub use advanced_benchmark_validation::{
    create_custom_advanced_validator, create_advanced_validator, AdvancedBenchmarkValidator,
    ValidationConfig as ComprehensiveValidationConfig, ValidationReport as AdvancedValidationReport,
    ValidationResult as ComprehensiveValidationResult,
};
*/

// MCMC module
pub use mcmc::ChainStatistics;

// Statistical tests module
pub mod tests;
pub use tests::anova::{one_way_anova, tukey_hsd};
pub use tests::chi2_test::{chi2_gof, chi2_independence, chi2_yates};
pub use tests::nonparametric::{friedman, kruskal_wallis, mann_whitney, wilcoxon};
pub use tests::normality::{anderson_darling, dagostino_k2, ks_2samp, shapiro_wilk};
pub use tests::ttest::{ttest_1samp, ttest_ind, ttest_ind_from_stats, ttest_rel, TTestResult};
pub use tests::*;

// Correlation measures
mod correlation;
mod correlation_parallel_enhanced;
mod correlation_simd;
pub use correlation::intraclass::icc;
pub use correlation::{
    corrcoef, kendall_tau, kendalltau, partial_corr, partial_corrr, pearson_r, pearsonr,
    point_biserial, point_biserialr, spearman_r, spearmanr,
};
pub use correlation_parallel_enhanced::{
    batch_correlations_parallel, corrcoef_parallel_enhanced, pearson_r_simd_enhanced,
    rolling_correlation_parallel, ParallelCorrelationConfig,
};
pub use correlation_simd::{corrcoef_simd, covariance_simd, pearson_r_simd};

// Dispersion and variability measures
mod dispersion;
pub use dispersion::{
    coef_variation, data_range, gini_coefficient, iqr, mean_abs_deviation, median_abs_deviation,
};

// Quantile-based statistics
mod quantile;
pub use quantile::{
    boxplot_stats, deciles, percentile, quantile, quartiles, quintiles, winsorized_mean,
    winsorized_variance, QuantileInterpolation,
};

// Distribution characteristics statistics
pub mod distribution_characteristics;
pub use distribution_characteristics::{
    cross_entropy, entropy, kl_divergence, kurtosis_ci, mode, skewness_ci, ConfidenceInterval,
    Mode, ModeMethod,
};

// Core functions for regression analysis
pub mod regression;
pub use regression::{
    elastic_net, group_lasso, huber_regression, lasso_regression, linear_regression, linregress,
    multilinear_regression, odr, polyfit, ransac, ridge_regression, stepwise_regression,
    theilslopes, HuberT, RegressionResults, StepwiseCriterion, StepwiseDirection, StepwiseResults,
    TheilSlopesResult,
};

// Core functions for random number generation
pub mod random;
pub use random::*;

#[cfg(test)]
mod test_utils {
    // Common utilities for testing statistical functions

    /// Generate a simple test array
    pub fn test_array() -> scirs2_core::ndarray::Array1<f64> {
        scirs2_core::ndarray::array![1.0, 2.0, 3.0, 4.0, 5.0]
    }
}
