# scirs2-stats TODO

## Status: v0.3.4 Released (March 18, 2026)

19,685 workspace tests pass (100% pass rate). All v0.3.4 features are complete and production-ready.

---

## v0.3.3 Completed

### Classical Statistics
- [x] Descriptive statistics: mean, median, trimmed mean, geometric/harmonic mean, variance, std, MAD, IQR, skewness, kurtosis, moments
- [x] Pearson, Spearman, Kendall tau, partial correlation, ICC
- [x] SIMD-accelerated variance, std, weighted mean (via scirs2-core)

### Probability Distributions (100+)
- [x] Continuous: Normal, Uniform, t, Chi-square, F, Gamma, Beta, Exponential, Laplace, Logistic, Cauchy, Pareto, Weibull, Lognormal, Rayleigh, Gumbel, and more
- [x] Discrete: Poisson, Bernoulli, Binomial, Geometric, Hypergeometric, Negative Binomial
- [x] Multivariate: Multivariate Normal, Dirichlet, Wishart, Inverse-Wishart, Multinomial, Multivariate-t
- [x] Circular: von Mises, wrapped Cauchy, wrapped Normal
- [x] Generalized Pareto Distribution (GPD) — MLE and PWM fitting, POT methodology
- [x] Alpha-stable distributions — characteristic function parametrization, simulation
- [x] von Mises-Fisher distribution on the d-sphere — MLE concentration parameter
- [x] Truncated distributions — arbitrary interval truncation of any base distribution
- [x] Tweedie distribution — compound Poisson-Gamma, power variance family

### Hypothesis Testing
- [x] t-tests (one-sample, two-sample, paired), one-way ANOVA, Tukey HSD
- [x] Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Friedman
- [x] Shapiro-Wilk, Anderson-Darling, D'Agostino's K²
- [x] Kolmogorov-Smirnov (one- and two-sample), Chi-square goodness-of-fit
- [x] Levene, Bartlett, Brown-Forsythe homogeneity tests
- [x] Multiple testing corrections: Bonferroni, BH, BY, Holm, Hochberg
- [x] Effect size measures: Cohen's d, Cohen's f², eta-squared, partial eta-squared, omega-squared, Cramer's V, epsilon-squared

### Regression
- [x] Simple and multiple linear regression, polynomial regression
- [x] Ridge (L2), Lasso (L1), Elastic Net
- [x] RANSAC, Huber regression, Theil-Sen
- [x] Stepwise selection, cross-validation, AIC/BIC, VIF, residual analysis

### Bayesian & MCMC
- [x] Conjugate priors (Beta-Binomial, Gamma-Poisson, Normal-Normal, Dirichlet-Multinomial)
- [x] Metropolis-Hastings with adaptive proposals
- [x] Hamiltonian Monte Carlo (HMC) with leapfrog integrator
- [x] No-U-Turn Sampler (NUTS)
- [x] Gibbs sampling (systematic and random-scan)
- [x] Slice sampling (stepping-out and doubling procedures)
- [x] Sequential Monte Carlo (SMC) / particle filters: Bootstrap PF, Auxiliary PF, resample-move, tempering
- [x] Hierarchical Bayesian models
- [x] Bayesian networks (exact inference via variable elimination, approximate via loopy BP)
- [x] Variational inference utilities

### Gaussian Processes
- [x] Kernels: SE, Matern (1/2, 3/2, 5/2), rational quadratic, periodic, linear, polynomial, neural network
- [x] Kernel composition: sum, product, scale
- [x] GP regression (exact) with marginal likelihood optimization
- [x] GP classification (Laplace and EP approximations)
- [x] Sparse GP: FITC, VFE (inducing-point methods)
- [x] Deep GP (stacked latent layers with doubly stochastic VI)

### Survival Analysis
- [x] Kaplan-Meier estimator with Greenwood variance and confidence bands
- [x] Nelson-Aalen cumulative hazard estimator
- [x] Cox proportional hazards (partial likelihood, Breslow baseline, time-varying covariates)
- [x] Accelerated Failure Time (AFT) models: Weibull, log-normal, log-logistic
- [x] Competing risks: cause-specific hazard and Fine-Gray sub-distribution hazard
- [x] Log-rank test, Wilcoxon test, restricted mean survival time (RMST)

### Copulas & Dependence
- [x] Parametric copulas: Frank, Clayton, Gumbel, Gaussian (normal), Student-t
- [x] Vine copulas: C-vine, D-vine, R-vine with pair-copula construction
- [x] Copula fitting (MLE, canonical ML), tail dependence coefficients
- [x] Conditional simulation from fitted copula

### Nonparametric Bayes
- [x] Dirichlet Process Mixture Models (DPMM) via collapsed Gibbs sampling
- [x] Chinese Restaurant Process (CRP) — prior and posterior samplers
- [x] Indian Buffet Process (IBP) — binary latent feature models
- [x] Stick-breaking and Polya urn representations

### Mixture Models
- [x] Gaussian Mixture Models (GMM) — EM and variational EM
- [x] Finite mixture models (general base distributions)
- [x] Bayesian GMM with automatic component selection

### Causal Inference
- [x] Causal DAG and CPDAG representation
- [x] D-separation, Markov blanket, skeleton algorithms
- [x] Cointegration: Engle-Granger two-step, Johansen trace and max-eigenvalue tests
- [x] Structural equation models (linear SEM, path coefficients)
- [x] Causal impact analysis via Bayesian structural time series

### Time-Series Statistics
- [x] Dynamic Factor Models (DFM) with EM fitting and Kalman smoother
- [x] Time-Varying Parameter VAR (TVP-VAR) with Kalman filter and forgetting factors
- [x] Hidden Markov Models (HMM): Baum-Welch, Viterbi, forward-backward
- [x] Stationarity tests: ADF, KPSS, Phillips-Perron, DFGLS, Zivot-Andrews structural break
- [x] Spectral density: periodogram, Welch, multitaper (DPSS)

### Compositional & Spatial
- [x] Compositional data: Aitchison geometry, ALR/CLR/ILR transforms, Dirichlet MLE, closure, perturbation
- [x] Spatial: empirical variogram, theoretical models (spherical, exponential, Gaussian), kriging (ordinary, simple, universal)
- [x] Moran's I spatial autocorrelation, K-function, L-function, Ripley's edge correction
- [x] Spatial scan statistics (circular window, Kulldorff)

### Panel Data & Hierarchical
- [x] Fixed effects (within estimator, Mundlak), random effects (GLS/FGLS), pooled OLS
- [x] Hausman specification test, cross-sectional dependence (Pesaran CD)
- [x] Hierarchical linear models (HLM) with random intercepts and slopes

### Extreme Value Analysis
- [x] GEV distribution, block maxima method
- [x] Peaks-over-threshold (POT) with GPD tail fitting
- [x] Return level and return period estimation

### Sampling & QMC
- [x] Sobol sequences (Joe-Kuo direction numbers), Halton, Faure
- [x] Latin hypercube sampling (LHS) with maximin and correlation optimization
- [x] Owen's scrambled nets
- [x] Bootstrap: non-parametric, stratified, block (circular and non-overlapping)
- [x] Jackknife, permutation tests

---

## v0.4.0 Roadmap

### Variational Inference — Partially implemented in v0.4.0
- [x] Automatic Differentiation Variational Inference (ADVI) with normalizing flows — Implemented in v0.4.0
- [x] Stein Variational Gradient Descent (SVGD) — Implemented in v0.4.0
- [x] Black-box VI with variance reduction (REINFORCE, VIMCO) — Implemented in v0.4.0 (`variational/bbvi.rs`)

### Causal Inference (Extended)
- [x] Full do-calculus identification engine (ID algorithm, hedge criterion) — Implemented in v0.4.0 (`causal/id_algorithm.rs`, `causal/hedge.rs`)
- [x] PC algorithm and FCI algorithm for causal discovery from observational data — Implemented in v0.4.0 (`causal/pc_algorithm.rs`, `causal/fci_algorithm.rs`)
- [x] Instrumental variable (IV) estimation and 2SLS — Implemented in v0.4.0
- [x] Difference-in-differences and synthetic control methods — Implemented in v0.4.0

### Online & Streaming Bayesian Learning
- [x] Online variational Bayes for conjugate models — Implemented in v0.4.0 (`online_bayes/online_vb.rs`)
- [x] Streaming Gaussian processes with sparse updates — Implemented in v0.4.0 (`online_bayes/streaming_gp.rs`)
- [x] Sequential Bayesian model comparison (SMC-based) — Implemented in v0.4.0 (`mcmc/smc_model_compare.rs`)

### Advanced Nonparametric Bayes
- [x] Hierarchical Dirichlet Process (HDP) for topic models — Implemented in v0.4.0 (`nonparametric_bayes/hdp.rs`, `hdp_topic_model.rs`)
- [x] Beta process for sparse feature learning — Implemented in v0.4.0 (`nonparametric_bayes/beta_process.rs`)
- [x] Normalized random measures with independent increments (NRMI) — Implemented in v0.4.0 (`nonparametric_bayes/nrmi.rs`)

### Multivariate Volatility — Implemented in v0.4.0
- [x] DCC-GARCH (Dynamic Conditional Correlation) — Implemented in v0.4.0
- [x] BEKK-GARCH for multivariate financial data — Implemented in v0.4.0
- [x] Realized covariance and HAR-RV models — Implemented in v0.4.0

### High-Dimensional Statistics
- [x] Graphical Lasso (GLASSO) for sparse precision matrix estimation — Implemented in v0.4.0 (`graphical_lasso.rs`, `high_dimensional/glasso.rs`)
- [x] Factor-adjusted robust multiple testing — Implemented in v0.4.0 (`high_dimensional/factor_adjusted_testing.rs`)
- [x] High-dimensional t-tests and principal component regression — Implemented in v0.4.0 (`high_dimensional/hd_tests.rs`)

---

## Known Issues

- Slice sampling performance degrades on very high-dimensional posteriors (>100 dimensions) — use HMC/NUTS instead
- Deep GP fitting is memory-intensive for large datasets (>10k points); sparse approximation recommended
- TVP-VAR with large lag orders (>4) and many series (>8) may be slow without parallel feature enabled

---

## Wave 72 — Symbolic MLE derive (2026-05-06)

- [x] **Symbolic MLE / method-of-moments — pdf-driven** (completed 2026-05-07)
  - **Goal:** `scirs2_stats::mle::derive` takes parametric pdf as `LoweredOp`, parameter Var indices, and data Var index; returns an `Estimator` callable on data. Closes the last open Phase 3 cross-crate item.
  - **Design:** New `scirs2-stats/src/mle/derive.rs` behind `symbolic` feature. Builds symbolic log-likelihood ℓ(θ)=Σᵢln(pdf(xᵢ;θ)) via balanced-add-tree (depth O(log n)). Differentiates w.r.t. each θⱼ via `cas::ad::grad`; score_equations[j] = ∂ℓ/∂θⱼ. Calls `cas::solve_system(score_equations as (eq, Const(0.0)), params)`. On Ok → closed_form. On CannotEliminateTranscendental/PartialGroebner → falls_back_to_numeric=true. `Estimator::fit`: closed-form JIT-eval or Newton with finite-difference Hessian fallback. Structs: `Estimator { closed_form: Option<Vec<LoweredOp>>, score_equations: Vec<LoweredOp>, falls_back_to_numeric: bool }`. Reorganize: current `mle_symbolic.rs` → `mle/symbolic.rs`; new `mle/derive.rs`; `mle/mod.rs`.
  - **Files:** `scirs2-stats/src/mle/derive.rs` (new); `scirs2-stats/src/mle/mod.rs` (new); `scirs2-stats/src/mle/symbolic.rs` (renamed from mle_symbolic.rs); `scirs2-stats/src/lib.rs` (export under symbolic feature).
  - **Prerequisites:** `scirs2_symbolic::cas::solve_system` (scirs2-symbolic Wave 72); `cas::mle_catalog` (✓); `cas::ad::grad` (✓).
  - **Tests:** ≥8 in `tests/mle_derive_tests.rs`: Normal→μ̂=x̄ σ̂²=sample_var, fit on synthetic Normal(2.0,1.5) to 1e-6; Exponential→λ̂=1/x̄; Bernoulli→p̂=x̄; Geometric; Cauchy→closed_form is None, numeric Newton within 0.05 of true location; dim mismatch error; n_samples=0 rejected; canonical invariance (twice same hash).
  - **Risk:** Score equations for heavy-tailed transcendental. Mitigation: Newton with finite-difference Hessian always ships; document "closed-form" subset (exponential family with full sufficient statistic).
