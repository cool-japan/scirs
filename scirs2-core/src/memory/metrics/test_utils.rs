//! Test utilities for memory metrics tests
//!
//! This module provides common utilities for testing memory metrics functionality,
//! including a global test mutex to ensure test isolation.

use std::sync::Mutex;

/// Global test mutex to ensure memory metrics tests run in isolation
///
/// Since memory metrics uses a global static collector, tests must be synchronized
/// to prevent interference when running in parallel.
pub static MEMORY_METRICS_TEST_MUTEX: Mutex<()> = Mutex::new(());

/// Helper macro to lock the test mutex with proper error handling
#[macro_export]
macro_rules! lock_test_mutex {
    () => {
        $crate::memory::metrics::test_utils::MEMORY_METRICS_TEST_MUTEX
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    };
}
