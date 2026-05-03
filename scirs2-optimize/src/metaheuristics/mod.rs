//! # Metaheuristic Optimization Algorithms
//!
//! This module provides a collection of advanced metaheuristic optimization algorithms
//! for solving complex optimization problems. These algorithms are nature-inspired and
//! population-based methods that can handle non-convex, multi-modal, and combinatorial
//! optimization problems.
//!
//! ## Algorithms
//!
//! - **Simulated Annealing (SA)**: Probabilistic optimization inspired by metallurgical
//!   annealing, with configurable cooling schedules, reheating, and multi-start variants.
//!
//! - **Ant Colony Optimization (ACO)**: Swarm intelligence algorithm inspired by ant
//!   foraging behavior, suitable for combinatorial optimization (TSP, permutation problems).
//!
//! - **Differential Evolution (DE)**: Population-based stochastic optimizer for continuous
//!   domains, with multiple mutation strategies and self-adaptive parameter control.
//!
//! - **Harmony Search (HS)**: Music-inspired optimization with dynamic parameter adaptation,
//!   supporting both single and multi-objective optimization.

pub mod aco;
pub mod de;
pub mod harmony;
pub mod sa;

// Re-exports for Simulated Annealing
pub use sa::{
    AdaptiveCoolingState, ConstraintHandler, CoolingSchedule as SaCoolingSchedule,
    MetaheuristicSaOptions, MetaheuristicSaResult, MultiStartSaOptions, PenaltyConstraint,
    ReheatingStrategy, SimulatedAnnealingOptimizer,
};

// Re-exports for Ant Colony Optimization
pub use aco::{
    AcoResult, AntColonyOptimizer, AntSystemOptions, CombinatorialProblem, MaxMinAntSystem,
    MaxMinAntSystemOptions, PermutationProblem, TspProblem,
};

// Re-exports for Differential Evolution
pub use de::{
    CrossoverType, DeConstraintHandler, DeOptions, DeResult, DeStrategy,
    DifferentialEvolutionOptimizer, JdeOptions, OppositionBasedInit,
};

// Re-exports for Harmony Search
pub use harmony::{
    GlobalBestHarmonySearch, HarmonySearchOptimizer, HarmonySearchOptions, HarmonySearchResult,
    ImprovedHarmonySearchOptions, MultiObjectiveHarmonySearch, MultiObjectiveHsResult,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_accessible() {
        // Basic smoke test that modules are loadable
        let _opts = MetaheuristicSaOptions::default();
        let _de_opts = DeOptions::default();
        let _hs_opts = HarmonySearchOptions::default();
    }
}
