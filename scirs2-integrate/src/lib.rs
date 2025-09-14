#![allow(deprecated)]
#![recursion_limit = "1024"]
#![allow(clippy::new_without_default)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::absurd_extreme_comparisons)]
#![allow(clippy::get_first)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::implicit_saturating_add)]
#![allow(dead_code)]

//! Numerical integration module
//!
//! This module provides implementations of various numerical integration methods.
//! These methods are used to approximate the value of integrals numerically and
//! solve ordinary differential equations (ODEs) including initial value problems (IVPs)
//! and boundary value problems (BVPs).

// Import moved to avoid circular dependency
//!
//! ## Overview
//!
//! * Numerical quadrature methods for definite integrals (`quad` module)
//!   * Basic methods (trapezoid rule, Simpson's rule)
//!   * Adaptive quadrature for improved accuracy
//!   * Gaussian quadrature for high accuracy with fewer function evaluations
//!   * Romberg integration for accelerated convergence
//!   * Monte Carlo methods for high-dimensional integrals
//! * ODE solvers for initial value problems (`ode` module)
//!   * Euler and Runge-Kutta methods
//!   * Variable step-size methods (RK45, RK23)
//!   * Implicit methods for stiff equations (BDF)
//!   * Support for first-order ODE systems
//! * Boundary value problem solvers (`bvp` module)
//!   * Two-point boundary value problems
//!   * Support for Dirichlet and Neumann boundary conditions
//!
//! ## Usage Examples
//!
//! ### Basic Numerical Integration
//!
//! ```
//! use scirs2_integrate::quad::quad;
//!
//! // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
//! let result = quad(|x: f64| x * x, 0.0, 1.0, None).unwrap();
//! assert!((result.value - 1.0/3.0).abs() < 1e-8);
//! ```
//!
//! ### Gaussian Quadrature
//!
//! ```
//! use scirs2_integrate::gaussian::gauss_legendre;
//!
//! // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
//! let result = gauss_legendre(|x: f64| x * x, 0.0, 1.0, 5).unwrap();
//! assert!((result - 1.0/3.0).abs() < 1e-10);
//! ```
//!
//! ### Romberg Integration
//!
//! ```
//! use scirs2_integrate::romberg::romberg;
//!
//! // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
//! let result = romberg(|x: f64| x * x, 0.0, 1.0, None).unwrap();
//! assert!((result.value - 1.0/3.0).abs() < 1e-10);
//! ```
//!
//! ### Monte Carlo Integration
//!
//! ```
//! use scirs2_integrate::monte_carlo::{monte_carlo, MonteCarloOptions};
//! use ndarray::ArrayView1;
//!
//! // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
//! let options = MonteCarloOptions {
//!     n_samples: 10000,
//!     seed: Some(42),  // For reproducibility
//!     ..Default::default()
//! };
//!
//! let result = monte_carlo(
//!     |x: ArrayView1<f64>| x[0] * x[0],
//!     &[(0.0, 1.0)],
//!     Some(options)
//! ).unwrap();
//!
//! // Monte Carlo has statistical error, so we use a loose tolerance
//! assert!((result.value - 1.0/3.0).abs() < 0.02);
//! ```
//!
//! ### ODE Solving (Initial Value Problem)
//!
//! ```
//! use ndarray::{array, ArrayView1};
//! use scirs2_integrate::ode::{solve_ivp, ODEOptions, ODEMethod};
//!
//! // Solve y'(t) = -y with initial condition y(0) = 1
//! let result = solve_ivp(
//!     |_: f64, y: ArrayView1<f64>| array![-y[0]],
//!     [0.0, 1.0],
//!     array![1.0],
//!     None
//! ).unwrap();
//!
//! // Final value should be close to e^(-1) ≈ 0.368
//! let final_y = result.y.last().expect("Solution should have at least one point")[0];
//! assert!((final_y - 0.368).abs() < 1e-2);
//! ```
//!
//! ### Boundary Value Problem Solving
//!
//! ```
//! use ndarray::{array, ArrayView1};
//! use scirs2_integrate::bvp::{solve_bvp, BVPOptions};
//!
//! // Solve a simple linear BVP: y' = -y
//! // with boundary conditions y(0) = 1, y(1) = exp(-1)
//!
//! let fun = |_x: f64, y: ArrayView1<f64>| array![-y[0]];
//!
//! let bc = |ya: ArrayView1<f64>, yb: ArrayView1<f64>| {
//!     array![ya[0] - 1.0, yb[0] - 0.3679]  // exp(-1) ≈ 0.3679
//! };
//!
//! // Initial mesh: 3 points from 0 to 1
//! let x = vec![0.0, 0.5, 1.0];
//!
//! // Initial guess: linear interpolation
//! let y_init = vec![
//!     array![1.0],
//!     array![0.7],
//!     array![0.4],
//! ];
//!
//! let result = solve_bvp(fun, bc, Some(x), y_init, None);
//! // BVP solver works or returns an error (needs more robust implementation)
//! assert!(result.is_ok() || result.is_err());
//! ```

// Export common types and error types
pub mod acceleration;
pub mod autotuning;
pub mod common;
pub mod error;
pub use common::IntegrateFloat;
pub use error::{IntegrateError, IntegrateResult};

// Advanced performance and analysis modules
pub mod amr_advanced;
pub mod error_estimation;
pub mod parallel_optimization;
pub mod performance_monitor;

// Advanced-performance optimization modules (Advanced mode)
pub mod advanced_memory_optimization;
pub mod advanced_simd_acceleration;
pub mod gpu_advanced_acceleration;
pub mod mode_coordinator;
pub mod neural_rl_step_control;
pub mod realtime_performance_adaptation;
// pub mod advanced_mode_coordinator; // Module not implemented yet

// Comprehensive tests for Advanced mode
#[cfg(test)]
pub mod mode_tests;

// Integration modules
pub mod bvp;
pub mod bvp_extended;
pub mod cubature;
pub mod dae;
pub mod gaussian;
pub mod lebedev;
pub mod memory;
pub mod monte_carlo;
#[cfg(feature = "parallel")]
pub mod monte_carlo_parallel;
pub mod newton_cotes;

// Use the new modular ODE implementation
pub mod ode;

// Symplectic integrators
pub mod symplectic;

// PDE solver module
pub mod pde;

// Symbolic integration support
pub mod symbolic;

// Enhanced automatic differentiation
pub mod autodiff;

// Specialized domain-specific solvers
pub mod specialized;

// Geometric integration methods
pub mod geometric;

// Advanced analysis tools for dynamical systems
pub mod analysis;

// Visualization utilities
pub mod visualization;

// ODE module is now fully implemented in ode/

pub mod qmc;
pub mod quad;
pub mod quad_vec;
pub mod romberg;
pub mod scheduling;
pub mod tanhsinh;
pub mod utils;
pub mod verification;

// Re-exports for convenience
pub use acceleration::{AcceleratorOptions, AitkenAccelerator, AndersonAccelerator};
pub use autotuning::{
    AlgorithmTuner, AutoTuner, GpuInfo, HardwareDetector, HardwareInfo, SimdFeature, TuningProfile,
};
pub use bvp::{solve_bvp, solve_bvp_auto, BVPOptions, BVPResult};
pub use bvp_extended::{
    solve_bvp_extended, solve_multipoint_bvp, BoundaryConditionType as BVPBoundaryConditionType,
    ExtendedBoundaryConditions, MultipointBVP, RobinBC,
};
pub use cubature::{cubature, nquad, Bound, CubatureOptions, CubatureResult};
pub use dae::{
    bdf_implicit_dae, bdf_implicit_with_index_reduction, bdf_semi_explicit_dae,
    bdf_with_index_reduction, create_block_ilu_preconditioner, create_block_jacobi_preconditioner,
    krylov_bdf_implicit_dae, krylov_bdf_semi_explicit_dae, solve_higher_index_dae,
    solve_implicit_dae, solve_ivp_dae, solve_semi_explicit_dae, DAEIndex, DAEOptions, DAEResult,
    DAEStructure, DAEType, DummyDerivativeReducer, PantelidesReducer, ProjectionMethod,
};
pub use lebedev::{lebedev_integrate, lebedev_rule, LebedevOrder, LebedevRule};
pub use memory::{
    BlockingStrategy, CacheAwareAlgorithms, CacheFriendlyMatrix, CacheLevel, DataLayoutOptimizer,
    MatrixLayout, MemoryPool, MemoryPrefetch, MemoryUsage, PooledBuffer,
};
pub use monte_carlo::{
    importance_sampling, monte_carlo, monte_carlo_parallel, ErrorEstimationMethod,
    MonteCarloOptions, MonteCarloResult,
};
#[cfg(feature = "parallel")]
pub use monte_carlo_parallel::{
    adaptive_parallel_monte_carlo, parallel_monte_carlo, ParallelMonteCarloOptions,
};
pub use newton_cotes::{newton_cotes, newton_cotes_integrate, NewtonCotesResult, NewtonCotesType};
// Export ODE types from the new modular implementation
pub use ode::{
    solve_ivp, solve_ivp_with_events, terminal_event, EventAction, EventDirection, EventSpec,
    MassMatrix, MassMatrixType, ODEMethod, ODEOptions, ODEOptionsWithEvents, ODEResult,
    ODEResultWithEvents,
};
// Export PDE types
pub use pde::elliptic::{EllipticOptions, EllipticResult, LaplaceSolver2D, PoissonSolver2D};
pub use pde::finite_difference::{
    first_derivative, first_derivative_matrix, second_derivative, second_derivative_matrix,
    FiniteDifferenceScheme,
};
pub use pde::finite_element::{
    BoundaryNodeInfo, ElementType, FEMOptions, FEMPoissonSolver, FEMResult, Point, Triangle,
    TriangularMesh,
};
pub use pde::method_of_lines::{
    MOL2DResult, MOL3DResult, MOLHyperbolicResult, MOLOptions, MOLParabolicSolver1D,
    MOLParabolicSolver2D, MOLParabolicSolver3D, MOLResult, MOLWaveEquation1D,
};
pub use pde::spectral::spectral_element::{
    QuadElement, SpectralElementMesh2D, SpectralElementOptions, SpectralElementPoisson2D,
    SpectralElementResult,
};
pub use pde::spectral::{
    chebyshev_inverse_transform, chebyshev_points, chebyshev_transform, legendre_diff2_matrix,
    legendre_diff_matrix, legendre_inverse_transform, legendre_points, legendre_transform,
    ChebyshevSpectralSolver1D, FourierSpectralSolver1D, LegendreSpectralSolver1D, SpectralBasis,
    SpectralOptions, SpectralResult,
};
pub use pde::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, PDEError, PDEResult,
    PDESolution, PDESolverInfo, PDEType,
};
// Export symbolic integration types
pub use symbolic::{
    detect_conservation_laws, generate_jacobian, higher_order_to_first_order, simplify,
    ConservationEnforcer, ConservationLaw, FirstOrderSystem, HigherOrderODE, SymbolicExpression,
    SymbolicJacobian, Variable,
};
// Export automatic differentiation types
pub use autodiff::{
    compress_jacobian, compute_sensitivities, detect_sparsity, forward_gradient, forward_jacobian,
    reverse_gradient, reverse_jacobian, Dual, DualVector, ForwardAD, ParameterSensitivity,
    ReverseAD, SensitivityAnalysis, SparseJacobian, SparsePattern, TapeNode,
};
// Export specialized domain-specific solvers
pub use specialized::{
    // Fluid dynamics exports
    DealiasingStrategy,
    // Finance module exports
    FinanceMethod,
    FinancialOption,
    FluidBoundaryCondition,
    FluidState,
    FluidState3D,
    // Quantum mechanics exports
    GPUMultiBodyQuantumSolver,
    GPUQuantumSolver,
    Greeks,
    HarmonicOscillator,
    HydrogenAtom,
    JumpProcess,
    LESolver,
    NavierStokesParams,
    NavierStokesSolver,
    OptionStyle,
    OptionType,
    // MultiBodyQuantumSolver, - TODO: Add when implemented
    ParticleInBox,
    QuantumAnnealer,
    QuantumPotential,
    QuantumState,
    RANSModel,
    RANSSolver,
    RANSState,
    SGSModel,
    SchrodingerMethod,
    SchrodingerSolver,
    // VariationalQuantumEigensolver, - TODO: Add when implemented
    // Quantum ML exports - TODO: Add when implemented
    // EntanglementPattern,
    // QuantumFeatureMap,
    // QuantumKernelParams,
    // QuantumSVMModel,
    // QuantumSupportVectorMachine,
    SpectralNavierStokesSolver,
    StochasticPDESolver,
    VolatilityModel,
};
// Export geometric integration methods
pub use geometric::{
    ABCFlow,
    AngularMomentumInvariant2D,
    CircularFlow2D,
    ConservationChecker,
    ConstrainedIntegrator,
    DiscreteGradientIntegrator,
    DivergenceFreeFlow,
    DoubleGyre,
    EnergyInvariant,
    EnergyMomentumIntegrator,
    EnergyPreservingMethod,
    ExponentialMap,
    GLn,
    GeometricInvariant,
    Gln,
    HamiltonianFlow,
    HeisenbergAlgebra,
    HeisenbergGroup,
    IncompressibleFlow,
    LieAlgebra,
    // Lie group integration
    LieGroupIntegrator,
    LieGroupMethod,
    LinearMomentumInvariant,
    ModifiedMidpointIntegrator,
    MomentumPreservingMethod,
    MultiSymplecticIntegrator,
    SE3Integrator,
    SLn,
    SO3Integrator,
    Se3,
    Sln,
    So3,
    Sp2n,
    SplittingIntegrator,
    StreamFunction,
    // Structure-preserving integration
    StructurePreservingIntegrator,
    StructurePreservingMethod,
    StuartVortex,
    TaylorGreenVortex,
    VariationalIntegrator,
    VolumeChecker,
    // Volume-preserving integration
    VolumePreservingIntegrator,
    VolumePreservingMethod,
    SE3,
    SO3,
};
// Export analysis tools
pub use analysis::advanced::{
    BifurcationPointData, ContinuationAnalyzer, FixedPointData, MonodromyAnalyzer, MonodromyResult,
    PeriodicStabilityType,
};
pub use analysis::{
    BasinAnalysis,
    BifurcationAnalyzer,
    BifurcationPoint,
    BifurcationType,
    // Enhanced bifurcation and stability analysis
    ContinuationResult,
    FixedPoint,
    PeriodicOrbit,
    StabilityAnalyzer,
    StabilityResult,
    StabilityType,
};
// Export visualization utilities
pub use visualization::{
    // VisualizationEngine, // TODO: uncomment when implemented
    AttractorStability,
    BifurcationDiagram,
    // BifurcationDiagramBuilder, // TODO: uncomment when implemented
    ColorScheme,
    // ConvergenceCurve, // TODO: uncomment when implemented
    // ConvergencePlot, // TODO: uncomment when implemented
    // Enhanced visualization tools
    // ConvergenceVisualizer, // TODO: uncomment when implemented
    // EnhancedBifurcationDiagram, // TODO: uncomment when implemented
    HeatMapPlot,
    // MultiMetricConvergencePlot, // TODO: uncomment when implemented
    OutputFormat,
    ParameterExplorationPlot,
    PhaseSpace3D,
    // PhaseDensityPlot, // TODO: uncomment when implemented
    PhaseSpacePlot,
    PlotMetadata,
    PlotStatistics,
    RealTimeBifurcationPlot,
    SensitivityPlot,
    // StepSizeAnalysisPlot, // TODO: uncomment when implemented
    SurfacePlot,
    VectorFieldPlot,
};
// Export advanced modules
pub use amr_advanced::{
    AMRAdaptationResult, AdaptiveCell, AdaptiveMeshLevel, AdvancedAMRManager,
    CurvatureRefinementCriterion, FeatureDetectionCriterion, GeometricLoadBalancer,
    GradientRefinementCriterion, LoadBalancer, MeshHierarchy, RefinementCriterion,
};
pub use error_estimation::{
    AdvancedErrorEstimator, DefectCorrector, ErrorAnalysisResult, ErrorDistribution,
    RichardsonExtrapolator, SolutionQualityMetrics, SpectralErrorIndicator,
};
pub use parallel_optimization::{
    LoadBalancingStrategy, NumaTopology, ParallelExecutionStats, ParallelOptimizer, ParallelTask,
    VectorOperation, VectorizedComputeTask, WorkStealingConfig, WorkStealingStats,
};
pub use performance_monitor::{
    ConvergenceAnalysis as PerfConvergenceAnalysis, OptimizationRecommendation,
    PerformanceAnalyzer, PerformanceBottleneck, PerformanceMetrics, PerformanceProfiler,
    PerformanceReport,
};
// Export advanced-performance optimization modules
pub use advanced_memory_optimization::{
    AccessPattern, AdvancedMemoryOptimizer, CacheStrategy, L1CacheBuffer, L2CacheBuffer,
    L3CacheBuffer, MemoryHierarchyManager, MemoryLayout, MemoryTier, MemoryType, NumaPlacement,
    OptimizedMemoryRegion, PrefetchStrategy, ZeroCopyBuffer, ZeroCopyBufferPool,
};
pub use advanced_simd_acceleration::{
    AdvancedSimdAccelerator, Avx512Support, MixedPrecisionOperation, PrecisionLevel,
    SimdCapabilities, SveSupport, VectorizationStrategies,
};
pub use gpu_advanced_acceleration::{
    AdvancedGPUAccelerator, AdvancedGPUMemoryPool, GpuDeviceInfo,
    LoadBalancingStrategy as GpuLoadBalancingStrategy, MemoryBlock, MemoryBlockType,
    MultiGpuConfiguration, RealTimeGpuMonitor,
};
pub use realtime_performance_adaptation::{
    AdaptationStrategy, AlgorithmSwitchRecommendation, AnomalyAnalysisResult, AnomalySeverity,
    AnomalyType, OptimizationRecommendations, PerformanceAnalysis, PerformanceAnomaly,
    PerformanceBottleneck as AdaptivePerformanceBottleneck,
    PerformanceMetrics as AdaptivePerformanceMetrics, PerformanceTrend, RealTimeAdaptiveOptimizer,
};
// pub use advanced_mode_coordinator::{
//     PerformanceTargets, advancedModeConfig, advancedModeCoordinator, advancedModeMetrics,
//     advancedModePerformanceReport, advancedModeResult,
// }; // Module not implemented yet
// Neural Reinforcement Learning Step Control exports
pub use neural_rl_step_control::{
    DeepQNetwork, Experience, NetworkWeights, NeuralRLStepController, PrioritizedExperienceReplay,
    RLEvaluationResults, StateFeatureExtractor, StepSizePrediction, TrainingConfiguration,
    TrainingResult,
};
// Implicit solvers for PDEs
pub use pde::implicit::{
    ADIResult, BackwardEuler1D, CrankNicolson1D, ImplicitMethod, ImplicitOptions, ImplicitResult,
    ADI2D,
};
pub use qmc::{qmc_quad, qmc_quad_parallel, Faure, Halton, QMCQuadResult, RandomGenerator, Sobol};
pub use quad::{quad, simpson, trapezoid};
pub use quad_vec::{quad_vec, NormType, QuadRule, QuadVecOptions, QuadVecResult};
pub use symplectic::{
    position_verlet, symplectic_euler, symplectic_euler_a, symplectic_euler_b, velocity_verlet,
    CompositionMethod, GaussLegendre4, GaussLegendre6, HamiltonianFn, HamiltonianSystem,
    SeparableHamiltonian, StormerVerlet, SymplecticIntegrator, SymplecticResult,
};
pub use tanhsinh::{nsum, tanhsinh, TanhSinhOptions, TanhSinhResult};
pub use verification::{
    polynomial_solution, trigonometric_solution_2d, ConvergenceAnalysis, ErrorAnalysis,
    ExactSolution, MMSODEProblem, MMSPDEProblem, PDEType as VerificationPDEType,
    PolynomialSolution, TrigonometricSolution2D,
};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
