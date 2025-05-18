//! Regularization techniques for machine learning
//!
//! This module provides various regularization techniques commonly used in
//! machine learning to prevent overfitting, such as L1 (Lasso), L2 (Ridge),
//! ElasticNet, and Dropout.

use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::Result;

/// Trait for regularizers that can be applied to parameters and gradients
pub trait Regularizer<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    /// Apply regularization to parameters and gradients
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to regularize
    /// * `gradients` - The gradients to modify
    ///
    /// # Returns
    ///
    /// The regularization penalty value
    fn apply(&self, params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A>;

    /// Compute the regularization penalty value
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to compute the penalty for
    ///
    /// # Returns
    ///
    /// The regularization penalty value
    fn penalty(&self, params: &Array<A, D>) -> Result<A>;
}

mod activity;
mod dropconnect;
mod dropout;
mod elastic_net;
mod entropy;
mod l1;
mod l2;
mod label_smoothing;
mod manifold;
mod mixup;
mod orthogonal;
mod shakedrop;
mod spatial_dropout;
mod spectral_norm;
mod stochastic_depth;
mod weight_standardization;

// Re-export regularizers
pub use activity::{ActivityNorm, ActivityRegularization};
pub use dropconnect::DropConnect;
pub use dropout::Dropout;
pub use elastic_net::ElasticNet;
pub use entropy::{EntropyRegularization, EntropyRegularizerType};
pub use l1::L1;
pub use l2::L2;
pub use label_smoothing::LabelSmoothing;
pub use manifold::ManifoldRegularization;
pub use mixup::{CutMix, MixUp};
pub use orthogonal::OrthogonalRegularization;
pub use shakedrop::ShakeDrop;
pub use spatial_dropout::{FeatureDropout, SpatialDropout};
pub use spectral_norm::SpectralNorm;
pub use stochastic_depth::StochasticDepth;
pub use weight_standardization::WeightStandardization;
