//! Cutting-edge clustering algorithms including quantum-inspired methods and advanced online learning
//!
//! This module provides implementations of state-of-the-art clustering algorithms that push
//! the boundaries of traditional clustering methods. It includes quantum-inspired algorithms
//! that leverage quantum computing principles, advanced online learning variants, reinforcement
//! learning approaches, transfer learning methods, and deep clustering techniques.
//!
//! # Examples
//!
//! ## Quantum-Inspired Clustering
//!
//! ```rust
//! use scirs2_cluster::advanced::quantum::{quantum_kmeans, QuantumConfig};
//! use scirs2_core::ndarray::Array2;
//!
//! let data = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
//! let config = QuantumConfig::default();
//! let (centroids, labels) = quantum_kmeans(data.view(), 3, Some(config)).unwrap();
//! ```
//!
//! ## Adaptive Online Clustering
//!
//! ```rust
//! use scirs2_cluster::advanced::online::{adaptive_online_clustering, AdaptiveOnlineConfig};
//! use scirs2_core::ndarray::Array2;
//!
//! let data = Array2::from_shape_vec((20, 3), (0..60).map(|x| x as f64).collect()).unwrap();
//! let config = AdaptiveOnlineConfig::default();
//! let (centers, labels) = adaptive_online_clustering(data.view(), Some(config)).unwrap();
//! ```

pub mod online;
pub mod quantum;

// For now, we'll include placeholder modules for the remaining sections
// These would be fully implemented in a complete refactoring
pub mod deep;
pub mod quantum_algorithms;
pub mod reinforcement;
pub mod transfer;

// Re-export main types for convenience
pub use online::{adaptive_online_clustering, AdaptiveOnlineClustering, AdaptiveOnlineConfig};
pub use quantum::{quantum_kmeans, QuantumConfig, QuantumKMeans, QuantumState};

// Re-export types from placeholder modules
pub use deep::*;
pub use quantum_algorithms::*;
pub use reinforcement::*;
pub use transfer::*;

/// Convenience function to create a default quantum configuration
pub fn default_quantum_config() -> QuantumConfig {
    QuantumConfig::default()
}

/// Convenience function to create a default adaptive online configuration
pub fn default_adaptive_online_config() -> AdaptiveOnlineConfig {
    AdaptiveOnlineConfig::default()
}

/// Convenience function for quick quantum clustering with default parameters
pub fn quick_quantum_clustering<F>(
    data: scirs2_core::ndarray::ArrayView2<F>,
    n_clusters: usize,
) -> crate::error::Result<(
    scirs2_core::ndarray::Array2<F>,
    scirs2_core::ndarray::Array1<usize>,
)>
where
    F: scirs2_core::numeric::Float + scirs2_core::numeric::FromPrimitive + std::fmt::Debug,
{
    quantum_kmeans(data, n_clusters, None)
}

/// Convenience function for quick online clustering with default parameters
pub fn quick_online_clustering<F>(
    data: scirs2_core::ndarray::ArrayView2<F>,
) -> crate::error::Result<(
    scirs2_core::ndarray::Array2<F>,
    scirs2_core::ndarray::Array1<usize>,
)>
where
    F: scirs2_core::numeric::Float + scirs2_core::numeric::FromPrimitive + std::fmt::Debug,
{
    adaptive_online_clustering(data, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_default_quantum_config() {
        let config = default_quantum_config();
        assert_eq!(config.n_quantum_states, 8);
        assert_eq!(config.quantum_iterations, 50);
    }

    #[test]
    fn test_default_adaptive_online_config() {
        let config = default_adaptive_online_config();
        assert_eq!(config.initial_learning_rate, 0.1);
        assert_eq!(config.max_clusters, 50);
    }

    #[test]
    fn test_quick_quantum_clustering() {
        let data = Array2::from_shape_vec((8, 2), (0..16).map(|x| x as f64).collect()).unwrap();
        let result = quick_quantum_clustering(data.view(), 2);
        assert!(result.is_ok());

        let (centroids, labels) = result.unwrap();
        assert_eq!(centroids.nrows(), 2);
        assert_eq!(labels.len(), 8);
    }

    #[test]
    fn test_quick_online_clustering() {
        let data = Array2::from_shape_vec((6, 2), (0..12).map(|x| x as f64).collect()).unwrap();
        let result = quick_online_clustering(data.view());
        assert!(result.is_ok());

        let (centers, labels) = result.unwrap();
        assert_eq!(labels.len(), 6);
    }
}
