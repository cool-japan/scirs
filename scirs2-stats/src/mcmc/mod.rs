//! Markov Chain Monte Carlo (MCMC) methods
//!
//! This module provides implementations of MCMC algorithms for sampling from
//! complex probability distributions including:
//! - Metropolis-Hastings
//! - Gibbs sampling
//! - Hamiltonian Monte Carlo
//! - No-U-Turn Sampler (NUTS) - the gold standard adaptive HMC algorithm
//! - Advanced methods (Multiple-try Metropolis, Parallel Tempering, Slice Sampling, Ensemble Methods)
//! - Convergence diagnostics (R-hat, ESS, MCSE, autocorrelation)

mod advanced;
pub mod diagnostics;
mod enhanced_hamiltonian;
mod gibbs;
mod hamiltonian;
mod metropolis;
pub mod nuts;
pub mod smc;
pub mod smc_model_compare;

pub use advanced::*;
pub use diagnostics::*;
pub use enhanced_hamiltonian::*;
pub use gibbs::*;
pub use hamiltonian::*;
pub use metropolis::*;
pub use nuts::*;

#[allow(unused_imports)]
use crate::error::StatsResult as Result;
#[allow(unused_imports)]
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};
