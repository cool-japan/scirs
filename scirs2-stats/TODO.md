# scirs2-stats TODO

## 1. Random Module and RNG Improvements

- [x] RNG infrastructure updates:
  - [x] Update rand API to 0.9.0
  - [x] Fix `gen` → `random` migration
  - [x] Fix `thread_rng` → `rng` migration
  - [x] Standardize RNG initialization approach
  - [x] Ensure consistent seed handling

- [x] Code quality improvements:
  - [x] Fix remaining clippy warnings in new distributions
  - [x] Fix `len_zero` comparisons to use `is_empty()`
  - [x] Fix `needless_return` statements in new distributions
  - [x] Add type alias for complex return type in tukey_hsd

- [ ] Advanced sampling capabilities:
  - [ ] Add permutation_by_key function for sorting permutations
  - [ ] Implement multidimensional random sampling
  - [ ] Add Stratified bootstrap sampling
  - [ ] Add weighted bootstrap sampling
  - [ ] Add inverse transform sampling for arbitrary distributions

- [ ] Custom RNG features:
  - [ ] Implement seedable PCG random number generator
  - [ ] Add cryptographically secure RNG option
  - [ ] Implement thread-local RNG pools for performance
  - [ ] Add reproducible parallel random generation

## 2. Testing and Benchmarking

- [x] Integration testing:
  - [x] Add integration tests for random and sampling modules
  - [x] Test sampling distributions
  - [x] Test bootstrap sampling
  - [x] Test statistical properties
  - [x] Mark statistical tests with random failures as ignored

- [ ] Performance benchmarking:
  - [ ] Basic operations:
    - [ ] RNG generation (uniform, normal, etc.)
    - [ ] Array operations (permutation, sampling)
  - [ ] Distribution evaluations:
    - [ ] Distribution PDF/CDF/PPF calculations
    - [ ] Multivariate distribution operations
  - [ ] Complex operations:
    - [ ] Bootstrap sampling
    - [ ] Stratified sampling
    - [ ] MCMC methods
  - [ ] Comparative benchmarks:
    - [ ] Compare against SciPy implementations
    - [ ] Compare against native Rust alternatives

- [ ] Testing methodology improvements:
  - [ ] Property-based testing:
    - [ ] Confirm distribution properties
    - [ ] Test invariants like sample mean convergence
    - [ ] Test entropy and KL divergence calculations
  - [ ] Stress testing:
    - [ ] Large-scale data handling
    - [ ] Edge case behavior
    - [ ] Numerical stability tests
  - [ ] Reproducibility testing:
    - [ ] Seed-based test verification
    - [ ] Cross-platform consistency checks

## 3. Distribution Enhancements

- [x] Support core distributions:
  - [x] Normal distribution
  - [x] Uniform distribution
  - [x] Student's t distribution
  - [x] Chi-square distribution
  - [x] F distribution
  - [x] Poisson distribution

- [ ] Add more distributions:
  - [x] Implement Gamma distribution
  - [x] Implement Beta distribution
  - [x] Implement Exponential distribution
  - [ ] Implement discrete distributions:
    - [x] Bernoulli distribution
    - [x] Binomial distribution
    - [x] Negative Binomial distribution
    - [x] Geometric distribution
    - [x] Hypergeometric distribution
  - [ ] Implement continuous distributions:
    - [x] Weibull distribution
    - [x] Lognormal distribution
    - [x] Pareto distribution
    - [x] Cauchy distribution
    - [x] Laplace distribution
    - [x] Logistic distribution
  - [ ] Implement circular distributions:
    - [ ] von Mises distribution
    - [ ] Wrapped normal distribution
    - [ ] Wrapped Cauchy distribution

- [ ] Advanced distribution features:
  - [x] Support multivariate distributions
    - [x] MultivariateNormal
    - [x] MultivariateT
    - [x] Dirichlet
    - [x] Wishart
    - [x] InverseWishart
    - [x] Multinomial
    - [x] MultivariateLognormal
  - [ ] Add mixture models
  - [ ] Implement kernel density estimation
  - [ ] Support truncated distributions
  - [ ] Add goodness-of-fit tests

- [x] Improve the distribution trait system:
  - [x] Create unified interface for all distributions
  - [ ] Support composition of distributions
  - [x] Add methods for KL divergence, entropy
  - [x] Add trait for discrete vs continuous distributions
  - [x] Implement traits for:
    - [x] Normal distribution
    - [x] Uniform distribution
    - [x] Poisson distribution
    - [x] Laplace distribution
    - [x] Cauchy distribution
    - [x] Beta distribution
    - [x] Gamma distribution
    - [x] Exponential distribution
    - [x] Student's t distribution
    - [x] Chi-square distribution
    - [x] MultivariateNormal distribution

## 4. Statistical Tests with Improved Confidence Intervals

- [ ] Enhance statistical tests with confidence intervals:
  - [ ] Add confidence interval calculations to all statistical test results
  - [ ] Implement bootstrap-based confidence intervals for complex statistics
  - [ ] Support for confidence interval visualization data

- [ ] Add exact p-value calculations:
  - [ ] Implement exact p-values for non-parametric tests
  - [ ] Add exact combinatorial calculators for small sample sizes
  - [ ] Support for permutation test-based p-values

- [ ] Expand statistical test capabilities:
  - [ ] Add power analysis functionality for hypothesis tests
  - [ ] Include effect size calculations for all tests
  - [ ] Support multiple test correction methods

## 5. Performance Optimizations

- [ ] Add parallel implementations using rayon:
  - [ ] Parallel bootstrap sampling for large datasets
  - [ ] Parallel permutation for large arrays
  - [ ] Multithreaded random number generation 
  - [ ] Parallel computation of distribution statistics
  - [ ] Add standard `workers` parameter to parallelizable functions

- [ ] Memory optimizations:
  - [ ] Add chunked processing for large samples
  - [ ] Support out-of-memory datasets
  - [ ] Implement lazy evaluation for distribution statistics
  - [ ] Add streaming computation for statistics when possible

- [ ] Algorithmic optimizations:
  - [ ] Optimize distribution quantile functions
  - [ ] Improve accuracy of tail probabilities
  - [ ] Optimize special function calculations
  - [ ] Add fast approximations with error bounds for performance-critical functions

## 6. Statistical Functions Enhancement

- [ ] Add hypothesis testing functions:
  - [ ] Parametric tests:
    - [ ] T-tests (one-sample, two-sample, paired)
    - [ ] ANOVA (one-way, multi-way)
    - [ ] MANOVA (multivariate analysis of variance)
    - [ ] ANCOVA (analysis of covariance)
  - [ ] Non-parametric tests:
    - [ ] Chi-square tests (goodness-of-fit, independence)
    - [ ] Kolmogorov-Smirnov test (one-sample, two-sample)
    - [ ] Mann-Whitney U test
    - [ ] Wilcoxon signed-rank test
    - [ ] Kruskal-Wallis test
    - [ ] Friedman test
  - [ ] Normality tests:
    - [ ] Shapiro-Wilk test
    - [ ] Anderson-Darling test
    - [ ] D'Agostino's K-squared test
  - [ ] Other tests:
    - [ ] Levene's test for homogeneity of variance
    - [ ] Mauchly's test for sphericity
    - [ ] Bartlett's test for equal variances
    - [ ] Brown-Forsythe test for equal variances

- [ ] Add descriptive statistics capabilities:
  - [ ] Correlation measures:
    - [ ] Pearson correlation coefficient
    - [ ] Spearman rank correlation
    - [ ] Kendall tau correlation
    - [ ] Point-biserial correlation
    - [ ] Partial correlation
    - [ ] Intraclass correlation coefficient
  - [ ] Dispersion measures:
    - [ ] Mean absolute deviation
    - [ ] Median absolute deviation
    - [ ] Interquartile range
    - [ ] Range
    - [ ] Coefficient of variation
    - [ ] Gini coefficient
  - [ ] Quantile-based statistics:
    - [ ] Percentiles and quantiles
    - [ ] Quartiles and quintiles
    - [ ] Box plot statistics
    - [ ] Winsorized mean and variance
  - [ ] Distribution characteristics:
    - [ ] Modes (unimodal, bimodal, multimodal)
    - [ ] Entropy and information measures
    - [ ] Skewness and kurtosis with confidence intervals

- [ ] Add regression and modeling features:
  - [ ] Linear models:
    - [ ] Improve linear regression implementation
    - [ ] Multiple linear regression
    - [ ] Polynomial regression
    - [ ] Stepwise regression
    - [ ] Robust regression methods
  - [ ] Regularized models:
    - [ ] Ridge regression (L2 regularization)
    - [ ] Lasso regression (L1 regularization)
    - [ ] Elastic Net (L1 + L2 regularization)
    - [ ] Group lasso and sparse group lasso
  - [ ] Generalized linear models:
    - [ ] Logistic regression
    - [ ] Poisson regression
    - [ ] Negative binomial regression
    - [ ] Gamma regression
    - [ ] Quasi-likelihood models
  - [ ] Model diagnostics:
    - [ ] Residual analysis
    - [ ] Influence measures
    - [ ] Heteroscedasticity tests
    - [ ] VIF calculation for multicollinearity
    - [ ] Model selection criteria (AIC, BIC, etc.)
    - [ ] Cross-validation utilities

## 7. Monte Carlo and Sampling Methods

- [ ] Enhance sampling capabilities:
  - [ ] Bootstrap methods:
    - [ ] Parametric bootstrap
    - [ ] Non-parametric bootstrap
    - [ ] Block bootstrap for time series
    - [ ] Stratified bootstrap
    - [ ] Balanced bootstrap
  - [ ] Monte Carlo methods:
    - [ ] Importance sampling
    - [ ] Rejection sampling
    - [ ] Adaptive sampling
    - [ ] Multilevel Monte Carlo
  - [ ] Markov Chain Monte Carlo (MCMC):
    - [ ] Metropolis-Hastings algorithm
    - [ ] Gibbs sampling
    - [ ] Hamiltonian Monte Carlo
    - [ ] MCMC diagnostics (Gelman-Rubin, autocorrelation)
    - [ ] No-U-Turn Sampler (NUTS)
  - [ ] Sequential Monte Carlo methods:
    - [ ] Particle filtering
    - [ ] SMC samplers
    - [ ] Annealed importance sampling

- [ ] Enhance Quasi-Monte Carlo (QMC) implementation:
  - [ ] Low-discrepancy sequences:
    - [ ] Sobol sequence generator
    - [ ] Halton sequence generator
    - [ ] Faure sequence generator
    - [ ] Scrambled sequence variants
  - [ ] Sampling designs:
    - [ ] Latin hypercube sampling
    - [ ] Orthogonal array sampling
    - [ ] Good lattice point sampling
    - [ ] Digital nets and sequences
  - [ ] QMC integration methods:
    - [ ] QMC for numerical integration
    - [ ] Randomized QMC
    - [ ] Multi-level QMC
    - [ ] Component-by-component construction

## 8. Documentation and Examples

- [x] Update all documentation for rand 0.9.0
- [ ] Add examples demonstrating common statistical tasks:
  - [ ] Sampling and resampling:
    - [ ] Bootstrap confidence intervals
    - [ ] Monte Carlo simulation examples
    - [ ] Random dataset generation
  - [ ] Statistical inference:
    - [ ] Parameter estimation
    - [ ] Hypothesis testing workflows
    - [ ] Power analysis
  - [ ] Statistical modeling:
    - [ ] Regression analysis examples
    - [ ] ANOVA examples
    - [ ] Model selection and validation
  - [ ] Multivariate analysis:
    - [ ] Principal component analysis
    - [ ] Factor analysis
    - [ ] Discriminant analysis
  - [ ] Bayesian analysis:
    - [ ] Bayesian estimation
    - [ ] MCMC examples
    - [ ] Bayesian model comparison

## 9. Error Handling and Robustness

- [ ] Error handling improvements:
  - [ ] Add consistent error handling across all functions
  - [ ] Create comprehensive error taxonomy
  - [ ] Add detailed error messages with recovery suggestions
  - [ ] Implement custom error types for statistical edge cases
- [ ] Input validation:
  - [ ] Implement input validation for all statistical functions
  - [ ] Add bounds checking for distribution parameters
  - [ ] Add dimensional consistency checks for matrix operations
- [ ] Numerical stability:
  - [ ] Add numerical stability improvements for extreme distribution tails
  - [ ] Implement log-space calculations for small probabilities
  - [ ] Add overflow/underflow protection mechanisms
- [ ] Data quality handling:
  - [ ] Support for handling missing/invalid data
  - [ ] Implement outlier detection methods
  - [ ] Add data transformation utilities for normality

## 10. API Improvements

- [ ] API ergonomics:
  - [ ] Create builder patterns for complex function calls
  - [ ] Add fluent interfaces for distributions and sampling
  - [ ] Implement chainable method calls for data transformation
- [ ] Type system improvements:
  - [ ] Create type aliases for complex return types
  - [ ] Add proper generic constraints for numeric types
  - [ ] Use const generics for dimension-aware computations
- [ ] API consistency:
  - [ ] Ensure consistent naming conventions across modules
  - [ ] Standardize parameter ordering in similar functions
  - [ ] Align API design with SciPy where appropriate
- [ ] Performance hints:
  - [ ] Add explicit inlining for performance-critical functions
  - [ ] Provide zero-copy alternatives where appropriate
  - [ ] Document performance characteristics of algorithms

## 11. Integration with Other Modules

- [ ] Data processing integration:
  - [ ] Seamless integration with scirs2-datasets for data loading
  - [ ] Data transformation pipelines for preprocessing
  - [ ] Support for streaming data processing
- [ ] Computational integration:
  - [ ] Integration with scirs2-linalg for efficient matrix operations
  - [ ] Integration with scirs2-optimize for maximum likelihood estimation
  - [ ] Integration with scirs2-fft for spectral analysis methods
- [ ] Visualization integration:
  - [ ] Integration with scirs2-plot (future) for statistical visualization
  - [ ] Support for interactive exploratory data analysis
  - [ ] Generate visualization-ready data structures
- [ ] Extended functionality:
  - [ ] Integration with scirs2-interpolate for density estimation
  - [ ] Support for time series analysis with scirs2-series
  - [ ] Integration with spatial statistics modules

## 12. Bayesian Statistical Methods

- [ ] Bayesian inference:
  - [ ] Conjugate prior framework
  - [ ] Bayesian linear regression
  - [ ] Bayesian hierarchical models
  - [ ] Variational inference methods
- [ ] Bayesian model selection:
  - [ ] Bayes factors
  - [ ] Bayesian information criteria
  - [ ] Cross-validation methods
- [ ] Bayesian nonparametrics:
  - [ ] Dirichlet process models
  - [ ] Gaussian process regression
  - [ ] Bayesian additive regression trees

## 13. Advanced Statistical Methods

- [ ] Multivariate methods:
  - [ ] Principal component analysis with rotation methods
  - [ ] Factor analysis with multiple extraction methods
  - [ ] Multidimensional scaling
  - [ ] Canonical correlation analysis
- [ ] Classification methods:
  - [ ] Discriminant analysis (linear, quadratic)
  - [ ] Naive Bayes classifier
  - [ ] k-Nearest Neighbors
- [ ] Survival analysis:
  - [ ] Kaplan-Meier estimator
  - [ ] Cox proportional hazards model
  - [ ] Parametric survival models
- [ ] Categorical data analysis:
  - [ ] Log-linear models
  - [ ] Correspondence analysis
  - [ ] Item response theory