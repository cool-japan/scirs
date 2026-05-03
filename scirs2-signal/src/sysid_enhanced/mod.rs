//! Enhanced System Identification with advanced algorithms
//!
//! This module provides advanced system identification methods including:
//! - Recursive identification with forgetting factors
//! - Multi-model adaptive estimation
//! - Nonlinear system identification
//! - Closed-loop identification
//! - MIMO system identification

// Core enhanced system identification modules
pub mod core;
pub mod recursive;
pub mod statistics;
pub mod types;

// Import the original implementation for backward compatibility
mod legacy;
pub use legacy::*;
