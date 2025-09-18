//! Cluster management for distributed computing
//!
//! This module provides comprehensive cluster management capabilities
//! including node discovery, health monitoring, resource allocation,
//! and fault-tolerant cluster coordination.
//!
//! The implementation has been modularized into separate components
//! for better maintainability and code organization.

// Re-export everything from the cluster module
pub use self::cluster::*;

// Include the modularized cluster implementation
pub mod cluster;