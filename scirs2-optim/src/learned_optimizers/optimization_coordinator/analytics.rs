//! Analytics and monitoring system for optimization coordinator
//!
//! This module provides comprehensive analytics capabilities for optimization monitoring,
//! performance analysis, convergence detection, resource monitoring, pattern detection,
//! anomaly detection, trend analysis, and reporting.
//!
//! The implementation has been refactored into a modular structure for better maintainability:
//! - Each analytics component is separated into focused modules under `analytics/`
//! - All original functionality is preserved through comprehensive re-exports
//! - New convenience functions and improved APIs are available
//! - Enhanced testing and documentation
//!
//! # Migration Note
//! All existing imports and usage patterns remain unchanged. The modular refactoring is
//! internal and does not affect the public API.

// Re-export all functionality from the modular implementation
pub use self::analytics::*;

// Declare the submodule
mod analytics;