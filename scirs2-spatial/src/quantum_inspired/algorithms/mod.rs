//! Quantum-Inspired Algorithms
//!
//! This module provides implementations of quantum-inspired algorithms for spatial
//! computing, including clustering, search, optimization, and machine learning algorithms
//! that leverage quantum computing principles for enhanced performance.

pub mod quantum_clustering;
pub mod quantum_optimization;
pub mod quantum_search;

// Export main algorithm structures
pub use quantum_clustering::QuantumClusterer;
pub use quantum_optimization::QuantumSpatialOptimizer;
pub use quantum_search::QuantumNearestNeighbor;

pub mod quantum_machine_learning;

// Re-export key types from quantum_machine_learning
pub use quantum_machine_learning::{QuantumClassifier, QuantumSVMModel};
