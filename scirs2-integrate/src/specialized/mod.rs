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
//
// `QuantumInspiredRNG` is intentionally not exported here — see
// `TODO.md` "Proposed follow-ups" for why.
pub use finance::advanced_monte_carlo_engine::{
    AdvancedMonteCarloEngine, OptionPricingResult, VarianceReductionSuite,
};
pub use finance::realtime_risk_engine::{
    AlertSeverity, RealTimeRiskMonitor, RiskAlert, RiskAlertType, RiskDashboard, RiskSnapshot,
};
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
// Enhanced financial modeling exports
//
// `RainbowPayoffType` and `StressScenario` are intentionally not exported
// here — see `TODO.md` "Proposed follow-ups" for why.
pub use finance::exotic_options::{ExoticOptionPricer, ExoticOptionType, PricingResult};
pub use finance::risk_management::{PortfolioRiskMetrics, RiskAnalyzer};
