//! Markov Decision Process (MDP) solvers
//!
//! This module provides exact and approximate MDP solvers including:
//! - Value Iteration
//! - Policy Iteration
//! - Modified Policy Iteration
//! - Linear Programming approach
//! - Q-Learning (model-free)
//! - SARSA (on-policy TD)
//! - RTDP (Real-Time Dynamic Programming)
//! - Prioritized Sweeping
//! - Stochastic Shortest Path
//! - Inverse Reinforcement Learning (MaxEnt IRL)

pub mod planning;
pub mod tabular;

pub use planning::*;
pub use tabular::*;
