//! Work-stealing scheduler implementation for dynamic load balancing
//!
//! This module provides a work-stealing scheduler that dynamically balances
//! work across threads, with timing analysis and adaptive chunking based on
//! workload characteristics.

pub mod cache_aware;
pub mod core;
pub mod matrix_ops;
pub mod queue;
pub mod scheduler;

// Re-export all public types and functions for backward compatibility

// Core types
pub use core::*;

// Queue implementation
pub use queue::WorkQueue;

// Scheduler implementation
pub use scheduler::WorkStealingScheduler;

// Matrix operations (all functions from the matrix_ops module)
pub use matrix_ops::*;

// Cache-aware implementations
pub use cache_aware::*;
