//! Quantum-Enhanced Search Algorithms
//!
//! This module provides quantum-inspired search algorithms that leverage quantum computing
//! principles for enhanced spatial data retrieval and neighbor searching.

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// Import quantum concepts
use super::super::concepts::QuantumState;

/// Quantum-Enhanced Nearest Neighbor Search
///
/// This structure implements a quantum-inspired nearest neighbor search algorithm
/// that uses quantum state representations and amplitude amplification to enhance
/// search performance. The algorithm can operate in pure quantum mode or fall back
/// to classical computation for compatibility.
///
/// # Features
/// - Quantum state encoding of reference points
/// - Amplitude amplification using Grover-like algorithms
/// - Quantum fidelity-based distance computation
/// - Classical fallback for robustness
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_spatial::quantum_inspired::algorithms::QuantumNearestNeighbor;
///
/// let points = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
/// let mut searcher = QuantumNearestNeighbor::new(&points.view())
///     .unwrap()
///     .with_quantum_encoding(true)
///     .with_amplitude_amplification(true);
///
/// let query = scirs2_core::ndarray::arr1(&[0.5, 0.5]);
/// let (indices, distances) = searcher.query_quantum(&query.view(), 2).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct QuantumNearestNeighbor {
    /// Reference points encoded as quantum states
    quantum_points: Vec<QuantumState>,
    /// Classical reference points
    classical_points: Array2<f64>,
    /// Enable quantum encoding
    quantum_encoding: bool,
    /// Enable amplitude amplification
    amplitude_amplification: bool,
    /// Grover iterations for search enhancement
    grover_iterations: usize,
}

impl QuantumNearestNeighbor {
    /// Create new quantum nearest neighbor searcher
    ///
    /// # Arguments
    /// * `points` - Reference points for nearest neighbor search
    ///
    /// # Returns
    /// A new `QuantumNearestNeighbor` instance with default configuration
    pub fn new(points: &ArrayView2<'_, f64>) -> SpatialResult<Self> {
        let classical_points = points.to_owned();
        let quantum_points = Vec::new(); // Will be initialized when quantum encoding is enabled

        Ok(Self {
            quantum_points,
            classical_points,
            quantum_encoding: false,
            amplitude_amplification: false,
            grover_iterations: 3,
        })
    }

    /// Enable quantum encoding of reference points
    ///
    /// When enabled, reference points are encoded as quantum states which can
    /// provide enhanced search performance through quantum parallelism.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable quantum encoding
    pub fn with_quantum_encoding(mut self, enabled: bool) -> Self {
        self.quantum_encoding = enabled;

        if enabled {
            // Initialize quantum encoding
            if let Ok(encoded) = self.encode_reference_points() {
                self.quantum_points = encoded;
            }
        }

        self
    }

    /// Enable amplitude amplification (Grover-like algorithm)
    ///
    /// Amplitude amplification can enhance search performance by amplifying
    /// the probability amplitudes of good solutions.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable amplitude amplification
    pub fn with_amplitude_amplification(mut self, enabled: bool) -> Self {
        self.amplitude_amplification = enabled;
        self
    }

    /// Configure Grover iterations
    ///
    /// Sets the number of Grover iterations used in amplitude amplification.
    /// More iterations can improve search quality but increase computation time.
    ///
    /// # Arguments
    /// * `iterations` - Number of Grover iterations (typically 3-5 for best results)
    pub fn with_grover_iterations(mut self, iterations: usize) -> Self {
        self.grover_iterations = iterations;
        self
    }

    /// Perform quantum-enhanced nearest neighbor search
    ///
    /// Finds the k nearest neighbors to a query point using quantum-enhanced
    /// distance computation when quantum encoding is enabled, otherwise falls
    /// back to classical Euclidean distance.
    ///
    /// # Arguments
    /// * `query_point` - Point to search for neighbors
    /// * `k` - Number of nearest neighbors to find
    ///
    /// # Returns
    /// Tuple of (indices, distances) for the k nearest neighbors
    pub fn query_quantum(
        &self,
        query_point: &ArrayView1<f64>,
        k: usize,
    ) -> SpatialResult<(Vec<usize>, Vec<f64>)> {
        let n_points = self.classical_points.nrows();

        if k > n_points {
            return Err(SpatialError::InvalidInput(
                "k cannot be larger than number of points".to_string(),
            ));
        }

        let mut distances = if self.quantum_encoding && !self.quantum_points.is_empty() {
            // Quantum-enhanced search
            self.quantum_distance_computation(query_point)?
        } else {
            // Classical fallback
            self.classical_distance_computation(query_point)
        };

        // Apply amplitude amplification if enabled
        if self.amplitude_amplification {
            distances = self.apply_amplitude_amplification(distances)?;
        }

        // Find k nearest neighbors
        let mut indexed_distances: Vec<(usize, f64)> = distances.into_iter().enumerate().collect();
        indexed_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let indices: Vec<usize> = indexed_distances
            .iter()
            .take(k)
            .map(|(i_, _)| *i_)
            .collect();
        let dists: Vec<f64> = indexed_distances.iter().take(k).map(|(_, d)| *d).collect();

        Ok((indices, dists))
    }

    /// Encode reference points into quantum states
    ///
    /// Converts classical reference points into quantum state representations
    /// using phase encoding and entangling gates for better quantum parallelism.
    fn encode_reference_points(&self) -> SpatialResult<Vec<QuantumState>> {
        let (n_points, n_dims) = self.classical_points.dim();
        let mut encoded_points = Vec::new();

        for i in 0..n_points {
            let point = self.classical_points.row(i);

            // Determine number of qubits needed
            let numqubits = (n_dims).next_power_of_two().trailing_zeros() as usize + 2;
            let mut quantum_point = QuantumState::zero_state(numqubits);

            // Encode each dimension
            for (dim, &coord) in point.iter().enumerate() {
                if dim < numqubits - 1 {
                    // Normalize coordinate to [0, π] range
                    let normalized_coord = (coord + 10.0) / 20.0; // Assumes data in [-10, 10]
                    let angle = normalized_coord.clamp(0.0, 1.0) * PI;
                    quantum_point.phase_rotation(dim, angle)?;
                }
            }

            // Apply entangling gates for better representation
            for i in 0..numqubits - 1 {
                quantum_point.controlled_rotation(i, i + 1, PI / 4.0)?;
            }

            encoded_points.push(quantum_point);
        }

        Ok(encoded_points)
    }

    /// Compute distances using quantum state overlap
    ///
    /// Calculates distances between query point and reference points using
    /// quantum state fidelity as a distance metric.
    fn quantum_distance_computation(
        &self,
        query_point: &ArrayView1<f64>,
    ) -> SpatialResult<Vec<f64>> {
        let n_dims = query_point.len();
        let mut distances = Vec::new();

        // Encode query point as quantum state
        let numqubits = n_dims.next_power_of_two().trailing_zeros() as usize + 2;
        let mut query_state = QuantumState::zero_state(numqubits);

        for (dim, &coord) in query_point.iter().enumerate() {
            if dim < numqubits - 1 {
                let normalized_coord = (coord + 10.0) / 20.0;
                let angle = normalized_coord.clamp(0.0, 1.0) * PI;
                query_state.phase_rotation(dim, angle)?;
            }
        }

        // Apply entangling gates to query state
        for i in 0..numqubits - 1 {
            query_state.controlled_rotation(i, i + 1, PI / 4.0)?;
        }

        // Calculate quantum fidelity with each reference point
        for quantum_ref in &self.quantum_points {
            let fidelity =
                QuantumNearestNeighbor::calculate_quantum_fidelity(&query_state, quantum_ref);

            // Convert fidelity to distance (higher fidelity = lower distance)
            let quantum_distance = 1.0 - fidelity;
            distances.push(quantum_distance);
        }

        Ok(distances)
    }

    /// Calculate classical distances as fallback
    ///
    /// Computes standard Euclidean distances when quantum encoding is disabled
    /// or as a fallback mechanism.
    fn classical_distance_computation(&self, query_point: &ArrayView1<f64>) -> Vec<f64> {
        let mut distances = Vec::new();

        for i in 0..self.classical_points.nrows() {
            let ref_point = self.classical_points.row(i);
            let distance: f64 = query_point
                .iter()
                .zip(ref_point.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            distances.push(distance);
        }

        distances
    }

    /// Calculate quantum state fidelity
    ///
    /// Computes the fidelity between two quantum states as |⟨ψ₁|ψ₂⟩|²
    fn calculate_quantum_fidelity(state1: &QuantumState, state2: &QuantumState) -> f64 {
        if state1.amplitudes.len() != state2.amplitudes.len() {
            return 0.0;
        }

        // Calculate inner product of quantum states
        let inner_product: Complex64 = state1
            .amplitudes
            .iter()
            .zip(state2.amplitudes.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();

        // Fidelity is |⟨ψ₁|ψ₂⟩|²
        inner_product.norm_sqr()
    }

    /// Apply amplitude amplification (Grover-like enhancement)
    ///
    /// Implements a Grover-like amplitude amplification algorithm to enhance
    /// the probability of finding good solutions (nearest neighbors).
    fn apply_amplitude_amplification(&self, mut distances: Vec<f64>) -> SpatialResult<Vec<f64>> {
        if distances.is_empty() {
            return Ok(distances);
        }

        // Find average distance
        let avg_distance: f64 = distances.iter().sum::<f64>() / distances.len() as f64;

        // Apply Grover-like amplitude amplification
        for _ in 0..self.grover_iterations {
            // Inversion about average (diffusion operator)
            #[allow(clippy::manual_slice_fill)]
            for distance in &mut distances {
                *distance = 2.0 * avg_distance - *distance;
            }

            // Oracle: amplify distances below average
            for distance in &mut distances {
                if *distance < avg_distance {
                    *distance *= 0.9; // Amplify by reducing distance
                }
            }
        }

        // Ensure all distances are positive
        let min_distance = distances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        if min_distance < 0.0 {
            for distance in &mut distances {
                *distance -= min_distance;
            }
        }

        Ok(distances)
    }

    /// Get number of reference points
    pub fn len(&self) -> usize {
        self.classical_points.nrows()
    }

    /// Check if searcher is empty
    pub fn is_empty(&self) -> bool {
        self.classical_points.nrows() == 0
    }

    /// Get reference to classical points
    pub fn classical_points(&self) -> &Array2<f64> {
        &self.classical_points
    }

    /// Check if quantum encoding is enabled
    pub fn is_quantum_enabled(&self) -> bool {
        self.quantum_encoding
    }

    /// Check if amplitude amplification is enabled
    pub fn is_amplification_enabled(&self) -> bool {
        self.amplitude_amplification
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_quantum_nearest_neighbor_creation() {
        let points = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let searcher = QuantumNearestNeighbor::new(&points.view()).unwrap();

        assert_eq!(searcher.len(), 3);
        assert!(!searcher.is_quantum_enabled());
        assert!(!searcher.is_amplification_enabled());
    }

    #[test]
    fn test_classical_search() {
        let points = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let searcher = QuantumNearestNeighbor::new(&points.view()).unwrap();

        let query = scirs2_core::ndarray::arr1(&[0.5, 0.5]);
        let (indices, distances) = searcher.query_quantum(&query.view(), 2).unwrap();

        assert_eq!(indices.len(), 2);
        assert_eq!(distances.len(), 2);
        // Should find point [0,0] and [1,1] as nearest
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
    }

    #[test]
    fn test_quantum_configuration() {
        let points = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let searcher = QuantumNearestNeighbor::new(&points.view())
            .unwrap()
            .with_quantum_encoding(true)
            .with_amplitude_amplification(true)
            .with_grover_iterations(5);

        assert!(searcher.is_quantum_enabled());
        assert!(searcher.is_amplification_enabled());
        assert_eq!(searcher.grover_iterations, 5);
    }

    #[test]
    fn test_empty_points() {
        let points = Array2::from_shape_vec((0, 2), vec![]).unwrap();
        let searcher = QuantumNearestNeighbor::new(&points.view()).unwrap();

        assert!(searcher.is_empty());
        assert_eq!(searcher.len(), 0);
    }

    #[test]
    fn test_invalid_k() {
        let points = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let searcher = QuantumNearestNeighbor::new(&points.view()).unwrap();

        let query = scirs2_core::ndarray::arr1(&[0.5, 0.5]);
        let result = searcher.query_quantum(&query.view(), 5);

        assert!(result.is_err());
    }
}
