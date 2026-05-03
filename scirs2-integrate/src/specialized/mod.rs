//! Specialized solvers for domain-specific problems
//!
//! This module provides optimized solvers for specific scientific domains:
//! - Quantum mechanics (Schrödinger equation)
//! - Fluid dynamics (Navier-Stokes)
//! - Financial modeling (stochastic PDEs)

pub mod finance;
pub mod finance_legacy;
pub mod fluid_dynamics;
pub mod quantum;

pub use finance::{
    FinanceMethod, FinancialOption, Greeks, JumpProcess, OptionStyle, OptionType,
    StochasticPDESolver, VolatilityModel,
};
// Advanced-performance financial computing exports
// TODO: These modules need to be created or mapped to existing modules
// pub use finance::advanced_monte_carlo_engine::{
//     AdvancedMonteCarloEngine, OptionPricingResult, QuantumInspiredRNG, VarianceReductionSuite,
// };
// pub use finance::realtime_risk_engine::{
//     AlertSeverity, RealTimeRiskMonitor, RiskAlert, RiskAlertType, RiskDashboard, RiskSnapshot,
// };
// pub use fluid_dynamics::turbulence_models::{
//     FluidState3D, LESolver, RANSModel, RANSSolver, RANSState, SGSModel,
// };
pub use fluid_dynamics::{
    DealiasingStrategy, FluidBoundaryCondition, FluidState, FluidState3D, LESolver,
    NavierStokesParams, NavierStokesSolver, RANSModel, RANSSolver, RANSState, SGSModel,
    SpectralNavierStokesSolver,
};
// Advanced-performance fluid dynamics exports (commented out until implemented)
// pub use fluid_dynamics::advanced_gpu_acceleration::{AdvancedGPUKernel, GPUMemoryPool};
// pub use fluid_dynamics::neural_adaptive_solver::{
//     AdaptiveAlgorithmSelector, AlgorithmRecommendation, ProblemCharacteristics,
// };
// pub use fluid_dynamics::streaming_optimization::StreamingComputeManager;
// Enhanced multiphase flow exports (commented out until implemented)
// pub use fluid_dynamics::multiphase_flow::{
//     InterfaceTrackingMethod, MultiphaseFlowSolver, MultiphaseFlowState, PhaseProperties,
// };
pub use quantum::algorithms::{
    MultiBodyEigenResult, MultiBodyQuantumSolver, QuantumAnnealer, VariationalQuantumEigensolver,
};
pub use quantum::{
    GPUMultiBodyQuantumSolver, GPUQuantumSolver, HarmonicOscillator, HydrogenAtom, ParticleInBox,
    QuantumPotential, QuantumState, SchrodingerMethod, SchrodingerSolver,
};
// Quantum machine learning exports - TODO: Add when implemented
// pub use quantum::algorithms::{
//     EntanglementPattern, QuantumFeatureMap, QuantumKernelParams, QuantumSVMModel,
//     QuantumSupportVectorMachine,
// };
// Enhanced quantum optimization exports - already imported above
// pub use quantum::QuantumAnnealingSolver;
// Enhanced financial modeling exports
// TODO: Map to actual existing modules
// pub use finance::exotic_options::{
//     AveragingType, ExoticOption, ExoticOptionPricer, ExoticOptionType, PricingResult,
//     RainbowPayoffType,
// };
// pub use finance::risk_management::{PortfolioRiskMetrics, RiskAnalyzer, StressScenario};
