//! AI-Driven Adaptive Processing - Next-Generation Intelligent Image Processing
//!
//! This module has been refactored into focused components for better maintainability.
//! See the submodules for specific functionality.

// Re-export all module components for backward compatibility
pub use self::{config::*, knowledge::*, learning::*, processing::*, state::*, strategies::*};

// Module declarations
pub mod config;
pub mod knowledge;
pub mod learning;
pub mod processing;
pub mod state;
pub mod strategies;
