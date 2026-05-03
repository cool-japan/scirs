//! Bayesian Nonparametric methods for scirs2-stats
//!
//! This module provides implementations of Bayesian nonparametric models,
//! including:
//!
//! - **Dirichlet Process (DP)**: The fundamental object of Bayesian nonparametrics,
//!   a distribution over distributions with concentration parameter α.
//! - **Stick-breaking construction (GEM)**: Sethuraman's constructive representation
//!   of the DP, producing mixture weights from Beta(1, α) draws.
//! - **Chinese Restaurant Process (CRP)**: Sequential construction of DP partitions
//!   via a metaphorical seating arrangement.
//! - **Pitman-Yor Process (PYP)**: A two-parameter generalisation of the DP that
//!   exhibits power-law cluster-size distributions.
//! - **DP Gaussian Mixture Model (DP-GMM)**: Infinite mixture of Gaussians with
//!   collapsed Gibbs sampling using the Normal-Inverse-Wishart conjugate prior.

pub mod beta_process;
pub mod dirichlet_process;
pub mod dp_mixture;
pub mod hdp;
pub mod hdp_topic_model;
pub mod ibp_linear_gaussian;
pub mod nrmi;

pub use dirichlet_process::{
    chinese_restaurant_process, crp_posterior_tables, crp_predictive, estimate_alpha_from_clusters,
    expected_clusters, pitman_yor_process, stick_breaking,
};
pub use dp_mixture::{
    dp_gmm_cluster, dp_gmm_log_likelihood, DpCluster, DpGaussianMixture, DpGmmResult,
};
