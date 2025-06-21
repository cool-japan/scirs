//! Global optimization algorithms
//!
//! This module provides various global optimization algorithms for finding
//! the global minimum of a multivariate function.

#[allow(dead_code)]
mod basinhopping;
#[allow(dead_code)]
mod bayesian;
#[allow(dead_code)]
mod clustering;
#[allow(dead_code)]
mod differential_evolution;
#[allow(dead_code)]
mod dual_annealing;
#[allow(dead_code)]
mod multi_start;
#[allow(dead_code)]
mod particle_swarm;
#[allow(dead_code)]
mod simulated_annealing;

#[cfg(test)]
mod tests;

pub use basinhopping::{basinhopping, BasinHoppingOptions};
pub use bayesian::{
    bayesian_optimization, AcquisitionFunctionType, BayesianOptimizationOptions, BayesianOptimizer,
    InitialPointGenerator, KernelType, Parameter, Space,
};
pub use clustering::{
    generate_diverse_start_points, multi_start_with_clustering, ClusterCentroid,
    ClusteringAlgorithm, ClusteringOptions, ClusteringResult, LocalMinimum, StartPointStrategy,
};
pub use differential_evolution::{differential_evolution, DifferentialEvolutionOptions};
pub use dual_annealing::{dual_annealing, DualAnnealingOptions};
pub use multi_start::{multi_start, MultiStartOptions, StartingPointStrategy};
pub use particle_swarm::{particle_swarm, ParticleSwarmOptions};
pub use simulated_annealing::{simulated_annealing, SimulatedAnnealingOptions};
