//! Time series feature extraction modules
//!
//! This module provides a modular approach to time series feature extraction,
//! organized by functionality for better maintainability and performance.

pub mod complexity;
pub mod config;
pub mod frequency;
pub mod statistical;
pub mod temporal;
pub mod turning_points;
pub mod utils;
pub mod wavelet;
pub mod window_based;

// Re-export commonly used items for convenience
#[allow(ambiguous_glob_reexports)]
pub use complexity::*;
#[allow(ambiguous_glob_reexports)]
pub use config::*;
// pub use frequency::*;
#[allow(ambiguous_glob_reexports)]
pub use statistical::*;
#[allow(ambiguous_glob_reexports)]
pub use temporal::*;
#[allow(ambiguous_glob_reexports)]
pub use turning_points::*;
#[allow(ambiguous_glob_reexports)]
pub use utils::*;
#[allow(ambiguous_glob_reexports)]
pub use wavelet::*;
#[allow(ambiguous_glob_reexports)]
pub use window_based::*;
