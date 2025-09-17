//! Dataset generators
//!
//! This module provides comprehensive dataset generation functionality for machine learning
//! and data science applications. It includes:
//!
//! - **Basic generators**: Classification, regression, clustering, and time series datasets
//! - **Manifold generators**: Swiss roll, S-curve, torus, and other manifold datasets
//! - **Noise injection**: Missing data and outlier injection utilities
//! - **GPU acceleration**: GPU-accelerated versions of basic generators
//! - **Configuration**: Types and utilities for generator configuration

pub mod basic;
pub mod config;
pub mod gpu;
pub mod manifold;
pub mod noise;

// Re-export all public functions and types for backward compatibility
pub use basic::*;
pub use config::*;
pub use gpu::*;
pub use manifold::*;
pub use noise::*;
