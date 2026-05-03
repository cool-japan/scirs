//! Topic modelling sub-crate.
//!
//! Currently provides the Hierarchical Dirichlet Process (HDP) model, which
//! automatically selects the number of topics from data.

/// Hierarchical Dirichlet Process topic model.
pub mod hdp;

pub use hdp::{Hdp, HdpConfig, HdpState, HdpTopicConfig, HdpTopicModel, TopicError};
