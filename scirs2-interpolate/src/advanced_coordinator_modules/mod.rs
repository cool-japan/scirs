//! Advanced Interpolation Coordinator Modules
//!
//! This module contains the refactored components of the advanced interpolation
//! coordinator, broken down into focused, maintainable modules.

// Core types and configuration
pub mod types;
pub mod config;

// Core coordinator functionality
pub mod core;

// Specialized optimization engines
pub mod method_selection;
pub mod accuracy_optimization;
pub mod pattern_analysis;
pub mod performance_tuning;
pub mod quantum_optimization;

// Knowledge and memory systems
pub mod knowledge_transfer;
pub mod memory_management;

// Public API re-exports
pub use types::*;
pub use config::*;
pub use core::AdvancedInterpolationCoordinator;

// Factory functions
pub use core::{
    create_advanced_interpolation_coordinator,
    create_advanced_interpolation_coordinator_with_config,
};

// Performance and metrics
pub use core::InterpolationPerformanceMetrics;