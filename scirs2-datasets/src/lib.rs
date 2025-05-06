//! Datasets module for SciRS2
//!
//! This module provides dataset loading utilities similar to scikit-learn's datasets module.
//! It includes toy datasets, sample datasets, time series datasets, data generators,
//! and utilities for loading and processing datasets.

#![warn(missing_docs)]

pub mod cache;
pub mod error;
pub mod generators;
pub mod loaders;
pub mod sample;
pub mod time_series;
pub mod toy;
/// Core utilities for working with datasets
///
/// This module provides the Dataset struct and helper functions for
/// manipulating and transforming datasets.
pub mod utils;

// Re-export commonly used functionality
pub use generators::*;
pub use sample::*;
pub use toy::*;
pub use utils::Dataset;
