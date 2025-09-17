//! CI/CD Automation for Performance Testing
//!
//! This module provides automated performance testing integration for CI/CD pipelines,
//! including GitHub Actions, GitLab CI, Jenkins, and other systems. It handles
//! automated benchmarking, regression detection, and report generation.
//!
//! The module has been refactored into a modular architecture for better maintainability:
//!
//! - **config**: Configuration management and platform settings
//! - **test_execution**: Test suite management and execution logic
//! - **reporting**: Report generation, templates, and formatting
//! - **artifact_management**: Storage providers and artifact handling
//! - **integrations**: External service integrations
//! - **performance_gates**: Performance monitoring and gate evaluation
//! - **core_automation**: Main automation engine and orchestration
//!
//! ## Usage
//!
//! ```rust
//! use scirs2_optim::benchmarking::ci_cd_automation::{
//!     CiCdAutomation, CiCdAutomationConfig, CiCdPlatform
//! };
//!
//! // Create automation configuration
//! let config = CiCdAutomationConfig {
//!     enable_automation: true,
//!     platform: CiCdPlatform::GitHubActions,
//!     // ... other configuration
//! };
//!
//! // Initialize automation system
//! let automation = CiCdAutomation::new(config)?;
//!
//! // Run performance tests with CI/CD integration
//! let results = automation.run_automated_tests().await?;
//! ```

// Re-export all functionality from the modular implementation
mod ci_cd_automation;

pub use ci_cd_automation::*;