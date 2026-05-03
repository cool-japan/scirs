//! Bayesian Optimization module for `scirs2-optimize`.
//!
//! Provides a comprehensive Bayesian optimization framework for black-box,
//! expensive-to-evaluate objective functions. The approach uses a Gaussian
//! Process surrogate to model the objective and acquisition functions to
//! decide where to sample next.
//!
//! # Architecture
//!
//! ```text
//! +-----------------+     +-------------------+     +------------------+
//! |  GP Surrogate   |<--->| Acquisition Func  |<--->| Bayesian Optim   |
//! |  (gp.rs)        |     | (acquisition.rs)  |     | (optimizer.rs)   |
//! +-----------------+     +-------------------+     +------------------+
//!        ^                                                  |
//!        |                 +-------------------+            |
//!        +-----------------| Sampling Design   |<-----------+
//!                          | (sampling.rs)     |
//!                          +-------------------+
//! ```
//!
//! # Modules
//!
//! - [`gp`] -- Gaussian Process surrogate with multiple kernels
//! - [`acquisition`] -- Acquisition functions (EI, PI, UCB, KG, Thompson, batch variants)
//! - [`optimizer`] -- Main optimizer loop (sequential, batch, constrained, multi-objective)
//! - [`sampling`] -- Initial design strategies (LHS, Sobol, Halton, random)
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_optimize::bayesian::optimize;
//! use scirs2_core::ndarray::ArrayView1;
//!
//! // Minimize a simple quadratic function
//! let result = optimize(
//!     |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2),
//!     &[(-5.0, 5.0), (-5.0, 5.0)],
//!     20,
//!     None,
//! ).expect("optimization failed");
//!
//! println!("Best x: {:?}", result.x_best);
//! println!("Best f: {:.6}", result.f_best);
//! ```
//!
//! # Advanced Usage
//!
//! ## Custom Kernel & Acquisition
//!
//! ```rust
//! use scirs2_optimize::bayesian::{
//!     BayesianOptimizer, BayesianOptimizerConfig,
//!     MaternKernel, MaternVariant,
//!     AcquisitionType, GpSurrogateConfig,
//! };
//! use scirs2_core::ndarray::ArrayView1;
//!
//! let config = BayesianOptimizerConfig {
//!     acquisition: AcquisitionType::UCB { kappa: 2.5 },
//!     n_initial: 8,
//!     seed: Some(42),
//!     gp_config: GpSurrogateConfig {
//!         noise_variance: 1e-4,
//!         optimize_hyperparams: false,
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//!
//! let kernel = Box::new(MaternKernel::new(MaternVariant::FiveHalves, 1.0, 1.0));
//! let mut opt = BayesianOptimizer::with_kernel(
//!     vec![(-5.0, 5.0), (-5.0, 5.0)],
//!     kernel,
//!     config,
//! ).expect("create optimizer");
//!
//! let result = opt.optimize(
//!     |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2),
//!     20,
//! ).expect("optimization ok");
//! ```

pub mod acquisition;
pub mod constrained_bo;
pub mod gp;
pub mod multi_fidelity;
pub mod optimizer;
pub mod sampling;
pub mod transfer_bo;
pub mod warm_start;

// ---- Re-exports for convenient access ----

// GP surrogate
pub use gp::{
    GpSurrogate,
    GpSurrogateConfig,
    // Kernels
    MaternKernel,
    MaternVariant,
    ProductKernel,
    RationalQuadraticKernel,
    RbfKernel,
    SumKernel,
    SurrogateKernel,
};

// Acquisition functions
pub use acquisition::{
    AcquisitionFn, AcquisitionType, BatchExpectedImprovement, BatchUpperConfidenceBound,
    ExpectedImprovement, KnowledgeGradient, ProbabilityOfImprovement, ThompsonSampling,
    UpperConfidenceBound,
};

// Optimizer
pub use optimizer::{
    optimize, BayesianOptResult, BayesianOptimizer, BayesianOptimizerConfig, Constraint,
    DimensionType, Observation,
};

// Sampling
pub use sampling::{generate_samples, SamplingConfig, SamplingStrategy};
