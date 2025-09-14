#![allow(deprecated)]
#![allow(dead_code)]
#![allow(unreachable_patterns)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(private_interfaces)]
//! Optimization module for SciRS
//!
//! This module provides implementations of various optimization algorithms,
//! modeled after SciPy's `optimize` module.
//!

#![allow(clippy::field_reassign_with_default)]
#![recursion_limit = "512"]
// Allow common mathematical conventions in optimization code
#![allow(clippy::many_single_char_names)] // x, f, g, h, n, m etc. are standard in optimization
#![allow(clippy::similar_names)] // x_pp, x_pm, x_mp, x_mm are standard for finite differences
//! ## Submodules
//!
//! * `unconstrained`: Unconstrained optimization algorithms
//! * `constrained`: Constrained optimization algorithms
//! * `least_squares`: Least squares minimization (including robust methods)
//! * `roots`: Root finding algorithms
//! * `scalar`: Scalar (univariate) optimization algorithms
//! * `global`: Global optimization algorithms
//!
//! ## Optimization Methods
//!
//! The following optimization methods are currently implemented:
//!
//! ### Unconstrained:
//! - **Nelder-Mead**: A derivative-free method using simplex-based approach
//! - **Powell**: Derivative-free method using conjugate directions
//! - **BFGS**: Quasi-Newton method with BFGS update
//! - **CG**: Nonlinear conjugate gradient method
//!
//! ### Constrained:
//! - **SLSQP**: Sequential Least SQuares Programming
//! - **TrustConstr**: Trust-region constrained optimizer
//!
//! ### Scalar (Univariate) Optimization:
//! - **Brent**: Combines parabolic interpolation with golden section search
//! - **Bounded**: Brent's method with bounds constraints
//! - **Golden**: Golden section search
//!
//! ### Global:
//! - **Differential Evolution**: Stochastic global optimization method
//! - **Basin-hopping**: Random perturbations with local minimization
//! - **Dual Annealing**: Simulated annealing with fast annealing
//! - **Particle Swarm**: Population-based optimization inspired by swarm behavior
//! - **Simulated Annealing**: Probabilistic optimization with cooling schedule
//!
//! ### Least Squares:
//! - **Levenberg-Marquardt**: Trust-region algorithm for nonlinear least squares
//! - **Trust Region Reflective**: Bounds-constrained least squares
//! - **Robust Least Squares**: M-estimators for outlier-resistant regression
//!   - Huber loss: Reduces influence of moderate outliers
//!   - Bisquare loss: Completely rejects extreme outliers
//!   - Cauchy loss: Provides very strong outlier resistance
//! - **Weighted Least Squares**: Handles heteroscedastic data (varying variance)
//! - **Bounded Least Squares**: Box constraints on parameters
//! - **Separable Least Squares**: Variable projection for partially linear models
//! - **Total Least Squares**: Errors-in-variables regression
//! ## Bounds Support
//!
//! The `unconstrained` module now supports bounds constraints for variables.
//! You can specify lower and upper bounds for each variable, and the optimizer
//! will ensure that all iterates remain within these bounds.
//!
//! The following methods support bounds constraints:
//! - Powell
//! - Nelder-Mead
//! - BFGS
//! - CG (Conjugate Gradient)
//!
//! ## Examples
//!
//! ### Basic Optimization
//!
//! ```
//! // Example of minimizing a function using BFGS
//! use ndarray::{array, ArrayView1};
//! use scirs2_optimize::unconstrained::{minimize, Method};
//!
//! fn rosenbrock(x: &ArrayView1<f64>) -> f64 {
//!     let a = 1.0;
//!     let b = 100.0;
//!     let x0 = x[0];
//!     let x1 = x[1];
//!     (a - x0).powi(2) + b * (x1 - x0.powi(2)).powi(2)
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let initial_guess = [0.0, 0.0];
//! let result = minimize(rosenbrock, &initial_guess, Method::BFGS, None)?;
//!
//! println!("Solution: {:?}", result.x);
//! println!("Function value at solution: {}", result.fun);
//! println!("Number of nit: {}", result.nit);
//! println!("Success: {}", result.success);
//! # Ok(())
//! # }
//! ```
//!
//! ### Optimization with Bounds
//!
//! ```
//! // Example of minimizing a function with bounds constraints
//! use ndarray::{array, ArrayView1};
//! use scirs2_optimize::{Bounds, unconstrained::{minimize, Method, Options}};
//!
//! // A function with minimum at (-1, -1)
//! fn func(x: &ArrayView1<f64>) -> f64 {
//!     (x[0] + 1.0).powi(2) + (x[1] + 1.0).powi(2)
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create bounds: x >= 0, y >= 0
//! // This will constrain the optimization to the positive quadrant
//! let bounds = Bounds::new(&[(Some(0.0), None), (Some(0.0), None)]);
//!
//! let initial_guess = [0.5, 0.5];
//! let mut options = Options::default();
//! options.bounds = Some(bounds);
//!
//! // Use Powell's method which supports bounds
//! let result = minimize(func, &initial_guess, Method::Powell, Some(options))?;
//!
//! // The constrained minimum should be at [0, 0] with value 2.0
//! println!("Solution: {:?}", result.x);
//! println!("Function value at solution: {}", result.fun);
//! # Ok(())
//! # }
//! ```
//!
//! ### Bounds Creation Options
//!
//! ```
//! use scirs2_optimize::Bounds;
//!
//! // Create bounds from pairs
//! // Format: [(min_x1, max_x1), (min_x2, max_x2), ...] where None = unbounded
//! let bounds1 = Bounds::new(&[
//!     (Some(0.0), Some(1.0)),  // 0 <= x[0] <= 1
//!     (Some(-1.0), None),      // x[1] >= -1, no upper bound
//!     (None, Some(10.0)),      // x[2] <= 10, no lower bound
//!     (None, None)             // x[3] is completely unbounded
//! ]);
//!
//! // Alternative: create from separate lower and upper bound vectors
//! let lb = vec![Some(0.0), Some(-1.0), None, None];
//! let ub = vec![Some(1.0), None, Some(10.0), None];
//! let bounds2 = Bounds::from_vecs(lb, ub).unwrap();
//! ```
//!
//! ### Robust Least Squares Example
//!
//! ```
//! use ndarray::{array, Array1, Array2};
//! use scirs2_optimize::least_squares::{robust_least_squares, HuberLoss};
//!
//! // Define residual function for linear regression
//! fn residual(params: &[f64], data: &[f64]) -> Array1<f64> {
//!     let n = data.len() / 2;
//!     let x_vals = &data[0..n];
//!     let y_vals = &data[n..];
//!     
//!     let mut res = Array1::zeros(n);
//!     for i in 0..n {
//!         res[i] = y_vals[i] - (_params[0] + params[1] * x_vals[i]);
//!     }
//!     res
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Data with outliers
//! let data = array![0., 1., 2., 3., 4., 0.1, 0.9, 2.1, 2.9, 10.0];
//! let x0 = array![0.0, 0.0];
//!
//! // Use Huber loss for robustness
//! let huber_loss = HuberLoss::new(1.0);
//! let result = robust_least_squares(
//!     residual,
//!     &x0,
//!     huber_loss,
//!     None::<fn(&[f64], &[f64]) -> Array2<f64>>,
//!     &data,
//!     None
//! )?;
//!
//! println!("Robust solution: intercept={:.3}, slope={:.3}",
//!          result.x[0], result.x[1]);
//! # Ok(())
//! # }
//! ```

// BLAS backend linking handled through scirs2-core

// Export error types
pub mod error;
pub use error::{OptimizeError, OptimizeResult};

// Module structure
pub mod advanced_coordinator;
#[cfg(feature = "async")]
pub mod async_parallel;
pub mod automatic_differentiation;
pub mod benchmarking;
pub mod constrained;
pub mod distributed;
pub mod distributed_gpu;
pub mod global;
pub mod gpu;
pub mod jit_optimization;
pub mod learned_optimizers;
pub mod least_squares;
pub mod ml_optimizers;
pub mod multi_objective;
pub mod neural_integration;
pub mod neuromorphic;
pub mod parallel;
pub mod quantum_inspired;
pub mod reinforcement_learning;
pub mod roots;
pub mod roots_anderson;
pub mod roots_krylov;
pub mod scalar;
pub mod self_tuning;
pub mod simd_ops;
pub mod sparse_numdiff; // Refactored into a module with submodules
pub mod stochastic;
pub mod streaming;
pub mod unconstrained;
pub mod unified_pipeline;
pub mod visualization;

// Common optimization result structure
pub mod result;
pub use result::OptimizeResults;

// Convenience re-exports for common functions
pub use advanced_coordinator::{
    advanced_optimize, AdvancedConfig, AdvancedCoordinator, AdvancedStats, AdvancedStrategy,
    StrategyPerformance,
};
#[cfg(feature = "async")]
pub use async_parallel::{
    AsyncDifferentialEvolution, AsyncOptimizationConfig, AsyncOptimizationStats,
    SlowEvaluationStrategy,
};
pub use automatic_differentiation::{
    autodiff, create_ad_gradient, create_ad_hessian, optimize_ad_mode, ADMode, ADResult,
    AutoDiffFunction, AutoDiffOptions,
};
pub use benchmarking::{
    benchmark_suites, test_functions, AlgorithmRanking, BenchmarkConfig, BenchmarkResults,
    BenchmarkRun, BenchmarkSummary, BenchmarkSystem, ProblemCharacteristics, RuntimeStats,
    TestProblem,
};
pub use constrained::minimize_constrained;
pub use distributed::{
    algorithms::{DistributedDifferentialEvolution, DistributedParticleSwarm},
    DistributedConfig, DistributedOptimizationContext, DistributedStats, DistributionStrategy,
    MPIInterface, WorkAssignment,
};
pub use distributed_gpu::{
    DistributedGpuConfig, DistributedGpuOptimizer, DistributedGpuResults, DistributedGpuStats,
    GpuCommunicationStrategy, IterationStats,
};
pub use global::{
    basinhopping, bayesian_optimization, differential_evolution, dual_annealing,
    generate_diverse_start_points, multi_start, multi_start_with_clustering, particle_swarm,
    simulated_annealing,
};
pub use gpu::{
    acceleration::{
        AccelerationConfig, AccelerationManager, AccelerationStrategy, PerformanceStats,
    },
    algorithms::{GpuDifferentialEvolution, GpuParticleSwarm},
    GpuFunction, GpuOptimizationConfig, GpuOptimizationContext, GpuPrecision,
};
pub use jit_optimization::{optimize_function, FunctionPattern, JitCompiler, JitOptions, JitStats};
pub use learned_optimizers::{
    learned_optimize, ActivationType, AdaptationStatistics, AdaptiveNASSystem,
    AdaptiveTransformerOptimizer, FewShotLearningOptimizer, LearnedHyperparameterTuner,
    LearnedOptimizationConfig, LearnedOptimizer, MetaOptimizerState, NeuralAdaptiveOptimizer,
    OptimizationNetwork, OptimizationProblem, ParameterDistribution, ProblemEncoder, TrainingTask,
};
pub use least_squares::{
    bounded_least_squares, least_squares, robust_least_squares, separable_least_squares,
    total_least_squares, weighted_least_squares, BisquareLoss, CauchyLoss, HuberLoss,
};
pub use ml_optimizers::{
    ml_problems, ADMMOptimizer, CoordinateDescentOptimizer, ElasticNetOptimizer,
    GroupLassoOptimizer, LassoOptimizer,
};
pub use multi_objective::{
    MultiObjectiveConfig, MultiObjectiveResult, MultiObjectiveSolution, NSGAII, NSGAIII,
};
pub use neural_integration::{optimizers, NeuralOptimizer, NeuralParameters, NeuralTrainer};
pub use neuromorphic::{
    neuromorphic_optimize, BasicNeuromorphicOptimizer, NeuromorphicConfig, NeuromorphicNetwork,
    NeuromorphicOptimizer, NeuronState, SpikeEvent,
};
pub use quantum_inspired::{
    quantum_optimize, quantum_particle_swarm_optimize, Complex, CoolingSchedule,
    QuantumAnnealingSchedule, QuantumInspiredOptimizer, QuantumOptimizationStats, QuantumState,
};
pub use reinforcement_learning::{
    actor_critic_optimize, bandit_optimize, evolutionary_optimize, meta_learning_optimize,
    policy_gradient_optimize, BanditOptimizer, EvolutionaryStrategy, Experience,
    MetaLearningOptimizer, OptimizationAction, OptimizationState, QLearningOptimizer,
    RLOptimizationConfig, RLOptimizer,
};
pub use roots::root;
pub use scalar::minimize_scalar;
pub use self_tuning::{
    presets, AdaptationResult, AdaptationStrategy, ParameterChange, ParameterValue,
    PerformanceMetrics, SelfTuningConfig, SelfTuningOptimizer, TunableParameter,
};
pub use sparse_numdiff::{sparse_hessian, sparse_jacobian, SparseFiniteDiffOptions};
pub use stochastic::{
    minimize_adam, minimize_adamw, minimize_rmsprop, minimize_sgd, minimize_sgd_momentum,
    minimize_stochastic, AdamOptions, AdamWOptions, DataProvider, InMemoryDataProvider,
    LearningRateSchedule, MomentumOptions, RMSPropOptions, SGDOptions, StochasticGradientFunction,
    StochasticMethod, StochasticOptions,
};
pub use streaming::{
    exponentially_weighted_rls, incremental_bfgs, incremental_lbfgs,
    incremental_lbfgs_linear_regression, kalman_filter_estimator, online_gradient_descent,
    online_linear_regression, online_logistic_regression, real_time_linear_regression,
    recursive_least_squares, rolling_window_gradient_descent, rolling_window_least_squares,
    rolling_window_linear_regression, rolling_window_weighted_least_squares,
    streaming_trust_region_linear_regression, streaming_trust_region_logistic_regression,
    IncrementalNewton, IncrementalNewtonMethod, LinearRegressionObjective,
    LogisticRegressionObjective, RealTimeEstimator, RealTimeMethod, RollingWindowOptimizer,
    StreamingConfig, StreamingDataPoint, StreamingObjective, StreamingOptimizer, StreamingStats,
    StreamingTrustRegion,
};
pub use unconstrained::{minimize, Bounds};
pub use unified_pipeline::{
    presets as unified_presets, UnifiedOptimizationConfig, UnifiedOptimizationResults,
    UnifiedOptimizer,
};
pub use visualization::{
    tracking::TrajectoryTracker, ColorScheme, OptimizationTrajectory, OptimizationVisualizer,
    OutputFormat, VisualizationConfig,
};

// Prelude module for convenient imports
pub mod prelude {
    pub use crate::advanced_coordinator::{
        advanced_optimize, AdvancedConfig, AdvancedCoordinator, AdvancedStats, AdvancedStrategy,
        StrategyPerformance,
    };
    #[cfg(feature = "async")]
    pub use crate::async_parallel::{
        AsyncDifferentialEvolution, AsyncOptimizationConfig, AsyncOptimizationStats,
        SlowEvaluationStrategy,
    };
    pub use crate::automatic_differentiation::{
        autodiff, create_ad_gradient, create_ad_hessian, optimize_ad_mode, ADMode, ADResult,
        AutoDiffFunction, AutoDiffOptions, Dual, DualNumber,
    };
    pub use crate::benchmarking::{
        benchmark_suites, test_functions, AlgorithmRanking, BenchmarkConfig, BenchmarkResults,
        BenchmarkRun, BenchmarkSummary, BenchmarkSystem, ProblemCharacteristics, RuntimeStats,
        TestProblem,
    };
    pub use crate::constrained::{minimize_constrained, Method as ConstrainedMethod};
    pub use crate::distributed::{
        algorithms::{DistributedDifferentialEvolution, DistributedParticleSwarm},
        DistributedConfig, DistributedOptimizationContext, DistributedStats, DistributionStrategy,
        MPIInterface, WorkAssignment,
    };
    pub use crate::distributed_gpu::{
        DistributedGpuConfig, DistributedGpuOptimizer, DistributedGpuResults, DistributedGpuStats,
        GpuCommunicationStrategy, IterationStats,
    };
    pub use crate::error::{OptimizeError, OptimizeResult};
    pub use crate::global::{
        basinhopping, bayesian_optimization, differential_evolution, dual_annealing,
        generate_diverse_start_points, multi_start_with_clustering, particle_swarm,
        simulated_annealing, AcquisitionFunctionType, BasinHoppingOptions,
        BayesianOptimizationOptions, BayesianOptimizer, ClusterCentroid, ClusteringAlgorithm,
        ClusteringOptions, ClusteringResult, DifferentialEvolutionOptions, DualAnnealingOptions,
        InitialPointGenerator, KernelType, LocalMinimum, Parameter, ParticleSwarmOptions,
        SimulatedAnnealingOptions, Space, StartPointStrategy,
    };
    pub use crate::gpu::{
        acceleration::{
            AccelerationConfig, AccelerationManager, AccelerationStrategy, PerformanceStats,
        },
        algorithms::{GpuDifferentialEvolution, GpuParticleSwarm},
        GpuFunction, GpuOptimizationConfig, GpuOptimizationContext, GpuPrecision,
    };
    pub use crate::jit_optimization::{
        optimize_function, FunctionPattern, JitCompiler, JitOptions, JitStats,
    };
    pub use crate::learned_optimizers::{
        learned_optimize, ActivationType, AdaptationStatistics, AdaptiveNASSystem,
        AdaptiveTransformerOptimizer, FewShotLearningOptimizer, LearnedHyperparameterTuner,
        LearnedOptimizationConfig, LearnedOptimizer, MetaOptimizerState, NeuralAdaptiveOptimizer,
        OptimizationNetwork, OptimizationProblem, ParameterDistribution, ProblemEncoder,
        TrainingTask,
    };
    pub use crate::least_squares::{
        bounded_least_squares, least_squares, robust_least_squares, separable_least_squares,
        total_least_squares, weighted_least_squares, BisquareLoss, BoundedOptions, CauchyLoss,
        HuberLoss, LinearSolver, Method as LeastSquaresMethod, RobustLoss, RobustOptions,
        SeparableOptions, SeparableResult, TLSMethod, TotalLeastSquaresOptions,
        TotalLeastSquaresResult, WeightedOptions,
    };
    pub use crate::ml_optimizers::{
        ml_problems, ADMMOptimizer, CoordinateDescentOptimizer, ElasticNetOptimizer,
        GroupLassoOptimizer, LassoOptimizer,
    };
    pub use crate::multi_objective::{
        MultiObjectiveConfig, MultiObjectiveResult, MultiObjectiveSolution, NSGAII, NSGAIII,
    };
    pub use crate::neural_integration::{
        optimizers, NeuralOptimizer, NeuralParameters, NeuralTrainer,
    };
    pub use crate::neuromorphic::{
        neuromorphic_optimize, BasicNeuromorphicOptimizer, NeuromorphicConfig, NeuromorphicNetwork,
        NeuromorphicOptimizer, NeuronState, SpikeEvent,
    };
    pub use crate::parallel::{
        parallel_evaluate_batch, parallel_finite_diff_gradient, ParallelOptions,
    };
    pub use crate::quantum_inspired::{
        quantum_optimize, quantum_particle_swarm_optimize, Complex, CoolingSchedule,
        QuantumAnnealingSchedule, QuantumInspiredOptimizer, QuantumOptimizationStats, QuantumState,
    };
    pub use crate::reinforcement_learning::{
        bandit_optimize, evolutionary_optimize, meta_learning_optimize, policy_gradient_optimize,
        BanditOptimizer, EvolutionaryStrategy, Experience, MetaLearningOptimizer,
        OptimizationAction, OptimizationState, QLearningOptimizer, RLOptimizationConfig,
        RLOptimizer,
    };
    pub use crate::result::OptimizeResults;
    pub use crate::roots::{root, Method as RootMethod};
    pub use crate::scalar::{
        minimize_scalar, Method as ScalarMethod, Options as ScalarOptions, ScalarOptimizeResult,
    };
    pub use crate::self_tuning::{
        presets, AdaptationResult, AdaptationStrategy, ParameterChange, ParameterValue,
        PerformanceMetrics, SelfTuningConfig, SelfTuningOptimizer, TunableParameter,
    };
    pub use crate::sparse_numdiff::{sparse_hessian, sparse_jacobian, SparseFiniteDiffOptions};
    pub use crate::streaming::{
        exponentially_weighted_rls, incremental_bfgs, incremental_lbfgs,
        incremental_lbfgs_linear_regression, kalman_filter_estimator, online_gradient_descent,
        online_linear_regression, online_logistic_regression, real_time_linear_regression,
        recursive_least_squares, rolling_window_gradient_descent, rolling_window_least_squares,
        rolling_window_linear_regression, rolling_window_weighted_least_squares,
        streaming_trust_region_linear_regression, streaming_trust_region_logistic_regression,
        IncrementalNewton, IncrementalNewtonMethod, LinearRegressionObjective,
        LogisticRegressionObjective, RealTimeEstimator, RealTimeMethod, RollingWindowOptimizer,
        StreamingConfig, StreamingDataPoint, StreamingObjective, StreamingOptimizer,
        StreamingStats, StreamingTrustRegion,
    };
    pub use crate::unconstrained::{minimize, Bounds, Method as UnconstrainedMethod, Options};
    pub use crate::unified_pipeline::{
        presets as unified_presets, UnifiedOptimizationConfig, UnifiedOptimizationResults,
        UnifiedOptimizer,
    };
    pub use crate::visualization::{
        tracking::TrajectoryTracker, ColorScheme, OptimizationTrajectory, OptimizationVisualizer,
        OutputFormat, VisualizationConfig,
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
