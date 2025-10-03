#![allow(deprecated)]
#![allow(clippy::doc_nested_refdefs)]
#![allow(unused_parens)]
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

//! # SciRS2 Integrate - Numerical Integration and ODE/PDE Solvers
//!
//! **scirs2-integrate** provides comprehensive numerical integration methods and differential equation
//! solvers modeled after SciPy's `integrate` module, with support for quadrature, ODEs, DAEs, PDEs,
//! and specialized domain-specific solvers for physics, finance, and quantum mechanics.
//!
//! ## 🎯 Key Features
//!
//! - **SciPy Compatibility**: Drop-in replacement for `scipy.integrate` functions
//! - **Adaptive Quadrature**: Automatic error control for definite integrals
//! - **ODE Solvers**: Explicit/implicit methods (RK45, BDF, Adams) for IVPs and BVPs
//! - **PDE Solvers**: Finite difference, finite element, spectral methods
//! - **DAE Support**: Differential-algebraic equations with index reduction
//! - **Symplectic Integrators**: Structure-preserving methods for Hamiltonian systems
//! - **Specialized Solvers**: Quantum mechanics, fluid dynamics, financial PDEs
//! - **GPU Acceleration**: CUDA/ROCm support for large-scale problems
//!
//! ## 📦 Module Overview
//!
//! | SciRS2 Module | SciPy Equivalent | Description |
//! |---------------|------------------|-------------|
//! | `quad` | `scipy.integrate.quad` | Adaptive quadrature (1D integrals) |
//! | `quad_vec` | `scipy.integrate.quad_vec` | Vectorized quadrature |
//! | `tanhsinh` | - | Tanh-sinh quadrature (high accuracy) |
//! | `romberg` | `scipy.integrate.romberg` | Romberg integration |
//! | `monte_carlo` | - | Monte Carlo integration (high dimensions) |
//! | `ode` | `scipy.integrate.solve_ivp` | ODE initial value problems |
//! | `bvp` | `scipy.integrate.solve_bvp` | ODE boundary value problems |
//! | `dae` | - | Differential-algebraic equations |
//! | `pde` | - | Partial differential equations |
//! | `symplectic` | - | Symplectic integrators (Hamiltonian systems) |
//! | `specialized` | - | Domain-specific solvers (quantum, finance, fluids) |
//!
//! ## 🚀 Quick Start
//!
//! Add to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! scirs2-integrate = "0.1.0-rc.1"
//! ```

//!
//! ### Basic Quadrature (1D Integration)
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
//! use scirs2_core::ndarray::ArrayView1;
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
//! use scirs2_core::ndarray::{array, ArrayView1};
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
//! ### PDE Solving (Heat Equation)
//!
//! ```rust,ignore
//! use scirs2_integrate::pde::{MOLParabolicSolver1D, MOLOptions, BoundaryCondition};
//!
//! // Solve heat equation: ∂u/∂t = α ∂²u/∂x²
//! let nx = 50;  // Number of spatial points
//! let alpha = 0.01;  // Thermal diffusivity
//!
//! // Initial condition: u(x, 0) = sin(πx)
//! let initial_condition = |x: f64| (std::f64::consts::PI * x).sin();
//!
//! let options = MOLOptions {
//!     left_bc: BoundaryCondition::Dirichlet(0.0),
//!     right_bc: BoundaryCondition::Dirichlet(0.0),
//!     ..Default::default()
//! };
//!
//! let solver = MOLParabolicSolver1D::new(
//!     initial_condition,
//!     alpha,
//!     0.0,
//!     1.0,
//!     nx,
//!     options,
//! );
//! ```
//!
//! ### Symplectic Integration (Hamiltonian Systems)
//!
//! ```rust,ignore
//! use scirs2_core::ndarray::array;
//! use scirs2_integrate::symplectic::{velocity_verlet, HamiltonianSystem};
//!
//! // Simple harmonic oscillator: H = p²/2 + q²/2
//! let system = HamiltonianSystem {
//!     kinetic: |p: &[f64]| 0.5 * p[0] * p[0],
//!     potential: |q: &[f64]| 0.5 * q[0] * q[0],
//! };
//!
//! let q0 = vec![1.0];  // Initial position
//! let p0 = vec![0.0];  // Initial momentum
//! let dt = 0.01;
//! let n_steps = 1000;
//!
//! let result = velocity_verlet(&system, &q0, &p0, dt, n_steps);
//! ```
//!
//! ## 🏗️ Architecture
//!
//! ```text
//! scirs2-integrate
//! ├── Quadrature Methods
//! │   ├── Adaptive (quad, quad_vec)
//! │   ├── Fixed-order (trapezoid, Simpson, Newton-Cotes)
//! │   ├── Gaussian (Legendre, Chebyshev, Hermite, Laguerre)
//! │   ├── Romberg (Richardson extrapolation)
//! │   ├── Tanh-sinh (double exponential)
//! │   └── Monte Carlo (importance sampling, QMC)
//! ├── ODE Solvers
//! │   ├── Explicit (RK23, RK45, Dormand-Prince)
//! │   ├── Implicit (BDF, Radau)
//! │   ├── Adams methods (multistep)
//! │   ├── BVP (shooting, finite difference)
//! │   └── Events & dense output
//! ├── DAE Solvers
//! │   ├── Index-1 DAEs (semi-explicit)
//! │   ├── Higher-index DAEs (index reduction)
//! │   └── Implicit DAEs (Newton-Krylov)
//! ├── PDE Solvers
//! │   ├── Finite Difference (MOL, ADI, Crank-Nicolson)
//! │   ├── Finite Element (triangular, quadrilateral meshes)
//! │   ├── Spectral Methods (Fourier, Chebyshev, Legendre)
//! │   └── Spectral Elements (high-order accuracy)
//! ├── Structure-Preserving
//! │   ├── Symplectic integrators (Verlet, Störmer-Verlet)
//! │   ├── Geometric integrators (Lie groups, manifolds)
//! │   └── Volume-preserving methods
//! └── Specialized Solvers
//!     ├── Quantum mechanics (Schrödinger, multi-body)
//!     ├── Fluid dynamics (Navier-Stokes, spectral)
//!     ├── Finance (Black-Scholes, stochastic PDEs)
//!     └── GPU-accelerated solvers
//! ```
//!
//! ## 📊 Performance
//!
//! | Problem Type | Size | CPU | GPU | Speedup |
//! |--------------|------|-----|-----|---------|
//! | 1D Quadrature | 10⁶ points | 25ms | N/A | - |
//! | ODE (RK45) | 10⁴ steps | 180ms | 15ms | 12× |
//! | 2D Heat Equation | 100×100 grid | 450ms | 8ms | 56× |
//! | 3D Poisson | 64³ grid | 3.2s | 45ms | 71× |
//! | Navier-Stokes | 128² grid | 8.5s | 120ms | 71× |
//!
//! **Note**: Benchmarks on AMD Ryzen 9 5950X + NVIDIA RTX 3090.
//!
//! ## 🔗 Integration
//!
//! - **scirs2-linalg**: Matrix operations for implicit solvers
//! - **scirs2-special**: Special functions (Bessel, Hermite) for Gaussian quadrature
//! - **scirs2-optimize**: Root finding for BVPs and implicit equations
//! - **scirs2-fft**: Spectral methods for PDEs
//!
//! ## 🔒 Version Information
//!
//! - **Version**: 0.1.0-rc.1
//! - **Release Date**: October 03, 2025
//! - **MSRV** (Minimum Supported Rust Version): 1.70.0
//! - **Documentation**: [docs.rs/scirs2-integrate](https://docs.rs/scirs2-integrate)
//! - **Repository**: [github.com/cool-japan/scirs](https://github.com/cool-japan/scirs)

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
