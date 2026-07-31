# scirs2-stats

[![crates.io](https://img.shields.io/crates/v/scirs2-stats.svg)](https://crates.io/crates/scirs2-stats)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-stats)](https://docs.rs/scirs2-stats)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)]()

**Comprehensive statistical computing for Rust** — part of the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.

`scirs2-stats` is the statistical backbone of SciRS2, providing a production-ready, pure-Rust implementation of probability distributions, hypothesis testing, Bayesian inference, survival analysis, MCMC sampling, Gaussian processes, copulas, and much more. The API mirrors SciPy's `stats` module where sensible, while going considerably further in v0.5.0 with nonparametric Bayes, causal inference, sequential Monte Carlo, and advanced time-series-oriented statistics.

**Tests:** 2529/2529 passing (default features), 2561/2561 passing (`--all-features`) as of 2026-07-22 baseline; not independently re-run for the 0.6.5 docs update.

**v0.6.5:** fixed several correctness bugs surfaced by a workspace-wide `#[ignore]`-legitimacy audit: the one-sided Kolmogorov-Smirnov test (`tests/normality.rs`) had an asymmetric/backwards p-value formula, printing logically-inverted "Rejected"/"Not rejected" conclusions; `polyfit`'s (`regression/polynomial.rs`) regression F-test hardcoded `f_p_value = F::zero()` instead of computing it from the F-statistic; `fisher_exact` (`contingency/mod.rs`) and the QMC low-discrepancy sequence generators (`qmc/advanced.rs`, `qmc/enhanced_sequences.rs` — Niederreiter/Sobol) produced garbage output on the affected code paths, now fixed; `ErrorMonitor::get_statistics()` (`error_diagnostics.rs`) self-deadlocked by re-locking its own non-reentrant `Mutex` from `detect_active_patterns()`. See [CHANGELOG.md](../CHANGELOG.md) `[0.6.5]` for full detail.

**v0.6.3:** fixed `AdaptiveMemoryManager::infer_deallocation_strategy` (`src/adaptive_memory_advanced/`), which re-derived the strategy from config instead of the strategy `allocate` actually resolved for that pointer — wrong for `Adaptive`-configured pools, which could be freed through the wrong allocator/layout and corrupt the heap. `allocate` now records the resolved strategy per pointer for `deallocate` to look up.

---

## Overview

Modern statistical workflows demand more than descriptive statistics and p-values. `scirs2-stats` covers:

- **Classical statistics**: descriptive measures, 100+ distributions, hypothesis tests, regression
- **Bayesian inference**: conjugate priors, MCMC (MH, HMC, NUTS, Gibbs, slice), SMC/particle filters, hierarchical models, Bayesian networks
- **Survival analysis**: Kaplan-Meier, Nelson-Aalen, Cox PH, AFT models, competing risks
- **Nonparametric Bayes**: Dirichlet process mixtures, Chinese restaurant process, Indian buffet process
- **Causal inference**: causal graphs (DAGs), do-calculus queries, cointegration tests
- **Dependence modelling**: copulas (Frank, Clayton, Gumbel, Student-t, vine), Bayesian copulas
- **Advanced distributions**: GPD, stable, von Mises-Fisher, truncated, Tweedie
- **Multiple testing & effect sizes**: FDR control, Bonferroni, BH, BY, Holm; Cohen's d, eta-squared, omega-squared

---

## Feature List (v0.6.5)

### Descriptive Statistics
- Mean, median, trimmed mean, geometric mean, harmonic mean
- Variance, standard deviation, MAD, IQR, range, coefficient of variation, Gini coefficient
- Skewness, kurtosis (Fisher and Pearson), moments
- Pearson, Spearman, Kendall tau, partial correlation, intraclass correlation (ICC)
- Quantiles, percentiles, box-plot statistics, winsorized statistics

### Probability Distributions (100+)
- **Continuous**: Normal, Uniform, Student-t, Chi-square, F, Gamma, Beta, Exponential, Laplace, Logistic, Cauchy, Pareto, Weibull, Lognormal, Rayleigh, Gumbel, Extreme Value, ...
- **Discrete**: Poisson, Bernoulli, Binomial, Geometric, Hypergeometric, Negative Binomial, ...
- **Multivariate**: Multivariate Normal, Multivariate-t, Dirichlet, Wishart, Inverse-Wishart, Multinomial
- **Circular**: von Mises, wrapped Cauchy, wrapped Normal
- **Heavy-tailed / special**:
  - Generalized Pareto Distribution (GPD) with MLE and PWM fitting
  - Stable distributions (alpha-stable) with characteristic function
  - von Mises-Fisher distribution on the sphere
  - Truncated distributions (any base distribution truncated to an interval)
  - Tweedie distribution (compound Poisson-Gamma, power variance)

### Hypothesis Testing
- **Parametric**: one-/two-/paired-sample t-tests, one-way ANOVA, Tukey HSD
- **Nonparametric**: Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Friedman
- **Normality**: Shapiro-Wilk, Anderson-Darling, D'Agostino's K²
- **Goodness-of-fit**: Kolmogorov-Smirnov (one- and two-sample), Chi-square
- **Homogeneity**: Levene, Bartlett, Brown-Forsythe
- **Multiple testing corrections**: Bonferroni, Benjamini-Hochberg (BH), Benjamini-Yekutieli (BY), Holm, Hochberg
- **Effect size measures**: Cohen's d, Cohen's f², eta-squared, partial eta-squared, omega-squared, Cramer's V, epsilon-squared

### Regression Analysis
- Simple and multiple linear regression, polynomial regression
- Ridge (L2), Lasso (L1), Elastic Net regularized regression
- Robust regression: RANSAC, Huber, Theil-Sen
- Stepwise selection, cross-validation utilities, AIC/BIC model criteria
- Residual analysis, influence measures, VIF calculation

### Bayesian Statistics & MCMC
- Conjugate priors (Beta-Binomial, Gamma-Poisson, Normal-Normal, ...)
- Markov Chain Monte Carlo:
  - Metropolis-Hastings with adaptive proposals
  - Hamiltonian Monte Carlo (HMC)
  - No-U-Turn Sampler (NUTS)
  - Gibbs sampling
  - Slice sampling
- **Sequential Monte Carlo (SMC) / particle filters**: Bootstrap particle filter, auxiliary particle filter, resample-move, tempering
- Hierarchical Bayesian models
- Bayesian networks (DAG structure, exact and approximate inference)
- Variational inference utilities

### Gaussian Processes
- Squared-exponential, Matern (1/2, 3/2, 5/2), rational quadratic, periodic, linear kernels
- Kernel composition (sum, product, scale)
- GP regression and classification
- Sparse GP (inducing-point methods: FITC, VFE)
- Deep GP (stacked latent GP layers)
- Hyperparameter optimization via marginal likelihood

### Survival Analysis
- Kaplan-Meier estimator with confidence bands
- Nelson-Aalen cumulative hazard estimator
- Cox proportional hazards model (partial likelihood, Breslow baseline)
- Accelerated Failure Time (AFT) models (Weibull, log-normal, log-logistic parametrizations)
- Competing risks analysis (cause-specific and sub-distribution hazard models)
- Log-rank test and generalized Wilcoxon test
- Restricted mean survival time (RMST) estimation

### Copulas & Dependence Modelling
- Parametric copulas: Frank, Clayton, Gumbel, Gaussian, Student-t
- Vine copulas (C-vine, D-vine, R-vine structures)
- Copula fitting (MLE, canonical ML), tail dependence coefficients
- Conditional simulation from fitted copulas

### Mixture Models
- Gaussian Mixture Models (GMM) with EM and variational EM
- Finite mixture models (general distributions)
- Nonparametric Bayesian mixtures:
  - Dirichlet Process Mixture Models (DPMM)
  - Chinese Restaurant Process (CRP) samplers
  - Indian Buffet Process (IBP) for latent feature models

### Causal Inference
- Causal graph representation (DAGs, CPDAGs, MAGs)
- D-separation and Markov blanket queries
- Cointegration testing (Engle-Granger, Johansen trace and eigenvalue tests)
- Structural equation models (SEM) — basic linear
- Causal impact estimation via Bayesian structural time series

### Time-Series-Oriented Statistics
- Dynamic Factor Models (DFM) with EM fitting
- Time-Varying Parameter VAR (TVP-VAR) with Kalman filter
- Hidden Markov Models (HMM) — Baum-Welch, Viterbi
- Stationarity tests: ADF, KPSS, Phillips-Perron, DFGLS, Zivot-Andrews
- Spectral density estimation: periodogram, Welch, multitaper

### Compositional & Spatial Data
- Compositional data analysis: Aitchison geometry, ALR/CLR/ILR transforms, Dirichlet MLE
- Spatial statistics: variogram estimation and modelling, Kriging (ordinary, simple, universal), Moran's I
- Spatial scan statistics, point pattern analysis (K-function, L-function, Ripley)

### Panel Data & Hierarchical Models
- Panel data models: fixed effects, random effects (GLS), pooled OLS
- Hierarchical linear models (HLM / multilevel models)
- Cross-sectional dependence tests, Hausman test

### Quasi-Monte Carlo Sampling
- Sobol sequences, Halton sequences, Faure sequences
- Latin hypercube sampling (LHS) with optimization
- Scrambled nets (Owen's scrambling)

---

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-stats = "0.6.5"
```

### Basic Descriptive Statistics

```rust
use scirs2_core::ndarray::array;
use scirs2_stats::{mean, median, std, var, skew, kurtosis, pearson_r};

let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

let m    = mean(&data.view()).unwrap();          // 5.5
let med  = median(&data.view()).unwrap();         // 5.5
let s    = std(&data.view(), 1, None).unwrap();  // sample std dev
let sk   = skew(&data.view(), false, None).unwrap();
let kurt = kurtosis(&data.view(), true, false, None).unwrap();

println!("mean={m:.3}, median={med:.3}, std={s:.3}, skew={sk:.3}, kurtosis={kurt:.3}");
```

### Hypothesis Testing with Multiple Testing Correction

```rust
use scirs2_core::ndarray::array;
use scirs2_stats::tests::ttest::Alternative;
use scirs2_stats::{multiple_testing::benjamini_hochberg, ttest_ind};

let group_a = array![5.1f64, 4.9, 6.2, 5.7, 5.5, 5.0];
let group_b = array![4.8f64, 5.2, 5.1, 4.7, 4.9, 4.6];

let result = ttest_ind(&group_a.view(), &group_b.view(), true, Alternative::TwoSided, "omit").unwrap();
println!("t = {:.4}, p = {:.4}", result.statistic, result.pvalue);

// Multiple testing correction over a collection of p-values
let p_values = vec![0.01, 0.04, 0.20, 0.003, 0.15];
let corrected = benjamini_hochberg(&p_values, 0.05).unwrap();
println!("BH-adjusted p-values: {:?}", corrected.pvalues_corrected);
```

### Sequential Monte Carlo (Particle Filter)

```rust
use scirs2_core::ndarray::array;
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::SeedableRng;
use scirs2_stats::mcmc::smc::{effective_sample_size, resample, ResamplingStrategy};

// Particle weights (e.g. from an observation likelihood update)
let weights = array![0.4_f64, 0.3, 0.2, 0.1];
let ess = effective_sample_size(&weights); // diagnostic: resample when ess/n < threshold

let mut rng = SmallRng::seed_from_u64(42);
let parent_indices = resample(&weights, ResamplingStrategy::Systematic, &mut rng).unwrap();
```

The full `ParticleFilter<O, T, P>` struct (bootstrap/auxiliary particle filters, tempering,
resample-move) is generic over user-supplied observation-model, transition-kernel, and prior
types — see `scirs2_stats::mcmc::smc` on docs.rs for a complete end-to-end example.

### Survival Analysis

```rust
use scirs2_stats::survival::{KaplanMeier, CoxPH, NelsonAalen};

// Kaplan-Meier estimator (events: true = event occurred, false = censored)
let times  = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
let events = vec![true, true, false, true, false, true, true, false, true];

let km = KaplanMeier::fit(&times, &events).unwrap();
println!("Median survival time: {:?}", km.median_survival());

// Nelson-Aalen cumulative hazard: `cumulative_hazard` is a `Vec<f64>` field
// parallel to `times` (H(t) at each unique event time), not a callable.
let na = NelsonAalen::fit(&times, &events).unwrap();
println!("Cumulative hazard at each event time: {:?}", na.cumulative_hazard);
```

### Copula Modelling

```rust
use scirs2_stats::copula::ClaytonCopula;

let clayton = ClaytonCopula::new(2.0).unwrap(); // theta=2; `new` validates theta > 0
let (u, v) = (0.3f64, 0.7);
println!("C({u},{v}) = {:.4}", clayton.cdf(u, v));
println!("c({u},{v}) = {:.4}", clayton.pdf(u, v));
```

(Vine copulas are exposed as `CVine` / `DVine` / `PairCopula` in `scirs2_stats::copula::vine`,
not a unified `VineCopula` type.)

### Gaussian Process Regression

```rust
use scirs2_stats::gaussian_process::{GaussianProcess, Matern52, ZeroPrior};

let kernel = Matern52::new(1.0, 1.0);           // length_scale=1.0, signal_variance=1.0
let mut gp = GaussianProcess::new(kernel, ZeroPrior, 1e-6); // (kernel, mean prior, noise)

gp.fit(&x_train, &y_train).unwrap();
let (mean_pred, std_pred) = gp.predict_with_std(&x_test).unwrap();
```

---

## API Overview

| Module | Description |
|---|---|
| `descriptive` | Descriptive statistics: mean, variance, skewness, quantiles |
| `distributions` | 100+ probability distributions with pdf/cdf/rvs |
| `distributions::multivariate` | Multivariate Normal, Dirichlet, Wishart, von Mises-Fisher |
| `distributions::generalized_pareto` | GPD fitting and inference |
| `distributions::stable` | Alpha-stable distributions |
| `distributions::truncated` | Truncated wrappers for any distribution |
| `distributions::tweedie` | Tweedie / compound Poisson-Gamma |
| `tests` | Parametric and nonparametric hypothesis tests |
| `multiple_testing` | Bonferroni, BH, BY, Holm corrections |
| `effect_size` | Cohen's d, eta-squared, omega-squared, Cramer's V |
| `regression` | Linear, ridge, lasso, elastic net, robust regression |
| `bayesian` | Conjugate priors, variational inference utilities |
| `mcmc` | Metropolis-Hastings, HMC, NUTS, Gibbs, slice, SMC |
| `mcmc::smc` | Sequential Monte Carlo / particle filters |
| `gaussian_process` | GP regression and classification |
| `gaussian_process::advanced` | Sparse GP, deep GP, non-Gaussian likelihoods |
| `survival` | Kaplan-Meier, Nelson-Aalen, Cox PH, AFT, competing risks |
| `copula` | Frank, Clayton, Gumbel, Gaussian, Student-t, vine copulas |
| `mixture` | GMM, finite mixtures, Dirichlet process mixtures |
| `nonparametric_bayes` | CRP, IBP, stick-breaking constructions |
| `causal` | Causal DAGs, d-separation, do-calculus, SEM |
| `causal_graph` | Graph-based causal inference utilities |
| `cointegration` | Engle-Granger, Johansen tests |
| `hmm` | Hidden Markov Models (Baum-Welch, Viterbi) |
| `bayesian_network` | Bayesian network inference |
| `dynamic_factor` | Dynamic factor models (DFM) |
| `tvp_var` | Time-varying parameter VAR |
| `information` | Mutual information, KL divergence, entropy |
| `stationarity` | ADF, KPSS, Phillips-Perron, DFGLS, Zivot-Andrews |
| `spectral_density` | Periodogram, Welch, multitaper PSD |
| `compositional` | Aitchison geometry, ALR/CLR/ILR transforms |
| `spatial` | Variogram, Kriging, Moran's I, K-function |
| `spatial_stats` | Spatial scan statistics, point processes |
| `panel` | Fixed/random effects panel models |
| `hierarchical` | Hierarchical linear models |
| `extreme_value` | GEV, GPD, block maxima, peaks-over-threshold |
| `nonparametric` | Kernel density estimation, rank tests |
| `robust` | Robust location and scale estimators |
| `resampling` | Bootstrap (simple, stratified, block), jackknife, permutation |
| `sampling` | QMC: Sobol, Halton, LHS, scrambled nets |
| `random` | Random variate generation utilities |

---

## Feature Flags

| Flag | Description |
|---|---|
| `parallel` | Enable Rayon-based parallel computation (recommended; delegates to `scirs2-core/parallel`) |
| `memmap` | Memory-mapped file support for large datasets |
| `python` | Python interop layer (delegates to `scirs2-core/python`) |
| `symbolic` | Symbolic MLE via `scirs2-symbolic` — exact symbolic gradients for `mle::derive` / `mle_symbolic` |
| `gpu` | GPU abstraction layer (delegates to `scirs2-core/gpu`) |
| `wgpu` | Pure-Rust WebGPU compute kernels for batch distribution evaluation (requires `gpu`) |
| `cuda` | Pure-Rust direct `oxicuda-*` CUDA path (NVIDIA-only, runtime-probed, no C/Fortran) |
| `memory_tracking` | Real RSS-based memory tracking for the SciPy benchmark framework (opt-in) |

SIMD-accelerated inner loops (via `scirs2-core`) and `serde`-based serialization are compiled in
unconditionally — they are dependencies of this crate, not optional Cargo features of it.

Default features: none (pure Rust, no C/Fortran dependencies).

---

## Links

- [SciRS2 project](https://github.com/cool-japan/scirs)
- [docs.rs](https://docs.rs/scirs2-stats)
- [crates.io](https://crates.io/crates/scirs2-stats)
- [TODO.md](./TODO.md)

## License

Apache License 2.0. See [LICENSE](../LICENSE) for details.
