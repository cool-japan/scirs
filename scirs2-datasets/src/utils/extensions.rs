//! Trait extensions for ndarray and other types
//!
//! This module provides trait extensions that add useful methods to existing types,
//! particularly for statistical operations and data manipulation that complement
//! the core ndarray functionality.

// Re-export the StatsExt trait from scaling where it's defined
pub use crate::utils::scaling::StatsExt;

// Additional trait extensions can be added here in the future
// For example: Extensions for DataFrame operations, specialized array operations, etc.

#[cfg(test)]
mod tests {
    use ndarray::array;

    #[test]
    fn test_stats_ext_integration() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let view = data.view();

        // Test that we can use the StatsExt trait methods
        let mean = view.mean().unwrap();
        assert!((mean - 3.0_f64).abs() < 1e-10);

        let std = view.std(0.0);
        assert!(std > 0.0);
    }
}
