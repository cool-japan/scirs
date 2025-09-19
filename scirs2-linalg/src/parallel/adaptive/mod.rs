//! Adaptive algorithm selection based on data size and worker configuration

use super::WorkerConfig;

/// Algorithm selection strategy
#[derive(Debug, Clone, Copy)]
pub enum Strategy {
    /// Always use serial processing
    Serial,
    /// Always use parallel processing
    Parallel,
    /// Automatically choose based on data size
    Adaptive,
}

/// Choose processing strategy based on data size and configuration
///
/// # Arguments
///
/// * `datasize` - Size of the data to process
/// * `config` - Worker configuration
///
/// # Returns
///
/// * Recommended processing strategy
pub fn choose_strategy(_datasize: usize, config: &WorkerConfig) -> Strategy {
    if _datasize < config.parallel_threshold {
        Strategy::Serial
    } else {
        Strategy::Parallel
    }
}

/// Check if parallel processing is recommended
///
/// # Arguments
///
/// * `datasize` - Size of the data to process
/// * `config` - Worker configuration
///
/// # Returns
///
/// * true if parallel processing is recommended
pub fn should_use_parallel(_datasize: usize, config: &WorkerConfig) -> bool {
    matches!(choose_strategy(_datasize, config), Strategy::Parallel)
}
