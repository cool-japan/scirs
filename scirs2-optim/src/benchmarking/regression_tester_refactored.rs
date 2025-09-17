//! Performance regression testing framework for CI/CD integration
//!
//! This module provides comprehensive performance regression detection capabilities
//! including baseline establishment, historical tracking, statistical analysis,
//! and automated CI/CD integration for continuous performance monitoring.
//!
//! ## Architecture
//!
//! This module has been refactored into focused submodules for better maintainability:
//!
//! - **config**: Configuration structures and validation
//! - **types**: Core data types and structures
//! - **database**: Performance database operations and persistence
//! - **detectors**: Regression detection algorithms (statistical, sliding window, change point)
//! - **statistics**: Statistical analysis implementations (trend analysis, outlier detection)
//! - **alerts**: Alert system with multi-channel notifications (email, Slack, GitHub)
//! - **mod**: Main framework orchestration and CI integration
//!
//! ## Example Usage
//!
//! ```rust
//! use scirs2_optim::benchmarking::regression_tester::{RegressionTester, RegressionConfig};
//! use scirs2_optim::benchmarking::BenchmarkResult;
//! use std::time::Duration;
//!
//! // Create regression tester with configuration
//! let config = RegressionConfig::default();
//! let mut tester = RegressionTester::new(config)?;
//!
//! // Run regression test on benchmark result
//! let benchmark_result = BenchmarkResult {
//!     mean_time: Duration::from_millis(100),
//!     std_dev: Duration::from_millis(5),
//!     median_time: Duration::from_millis(98),
//!     min_time: Duration::from_millis(95),
//!     max_time: Duration::from_millis(110),
//!     memory_usage: 1024 * 1024, // 1MB
//!     converged_value: Some(0.001),
//!     iterations: Some(1000),
//! };
//!
//! let test_result = tester.run_regression_test("optimizer_performance", &benchmark_result)?;
//!
//! // Generate CI report
//! let ci_report = tester.generate_ci_report(&[test_result])?;
//! println!("CI Report: {}", ci_report);
//! ```

// Re-export all public items from the modular implementation
mod regression_tester;

pub use regression_tester::*;