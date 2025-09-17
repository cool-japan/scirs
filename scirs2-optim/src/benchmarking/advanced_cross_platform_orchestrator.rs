//! Advanced Cross-Platform Testing Orchestrator
//!
//! This module provides sophisticated cross-platform testing capabilities with cloud provider
//! integration, containerized testing environments, automated test matrix generation,
//! and comprehensive platform compatibility validation.
//!
//! The implementation has been refactored into a modular structure for better maintainability:
//! - Each component is separated into focused modules under `advanced_cross_platform_orchestrator/`
//! - All original functionality is preserved through comprehensive re-exports
//! - New convenience functions and improved APIs are available
//!
//! # Migration Note
//! All existing imports and usage patterns remain unchanged. The modular refactoring is
//! internal and does not affect the public API.

// Re-export all functionality from the modular implementation
pub use self::advanced_cross_platform_orchestrator::*;

// Declare the submodule
mod advanced_cross_platform_orchestrator;