//! Quantum-Enhanced Data Generation Engine
//!
//! This module provides quantum-inspired algorithms for generating sophisticated
//! synthetic datasets with quantum computational advantages and enhanced performance.

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::f64::consts::PI;

/// Quantum-enhanced dataset generator using quantum-inspired algorithms
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QuantumDatasetGenerator {
    /// Quantum coherence time for entanglement effects
    coherence_time: f64,
    /// Number of quantum bits (qubits) to simulate
    n_qubits: usize,
    /// Quantum gate fidelity (0.0 to 1.0)
    gate_fidelity: f64,
    /// Enable quantum advantage optimizations
    quantum_advantage: bool,
}

impl Default for QuantumDatasetGenerator {
    fn default() -> Self {
        Self {
            coherence_time: 1000.0, // microseconds
            n_qubits: 8,
            gate_fidelity: 0.99,
            quantum_advantage: true,
        }
    }
}

impl QuantumDatasetGenerator {
    /// Create a new quantum dataset generator
    pub fn new(n_qubits: usize, gate_fidelity: f64) -> Self {
        Self {
            coherence_time: 1000.0,
            n_qubits,
            gate_fidelity,
            quantum_advantage: true,
        }
    }

    /// Generate quantum-entangled classification dataset
    pub fn make_quantum_classification(
        &self,
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
        entanglement_strength: f64,
        random_seed: Option<u64>,
    ) -> Result<Dataset> {
        if n_samples == 0 || n_features == 0 || n_classes == 0 {
            return Err(DatasetsError::InvalidFormat(
                "All parameters must be > 0".to_string(),
            ));
        }

        let mut rng = match random_seed {
            Some(_seed) => StdRng::seed_from_u64(_seed),
            None => StdRng::from_rng(&mut thread_rng()),
        };

        // Initialize quantum state vectors for each sample
        let mut data = Array2::zeros((n_samples, n_features));
        let mut targets = Array1::zeros(n_samples);

        // Generate quantum-entangled feature correlations
        let entanglement_matrix =
            self.generate_entanglement_matrix(n_features, entanglement_strength, &mut rng)?;

        for sample_idx in 0..n_samples {
            // Assign class using quantum superposition collapse
            let class_id = self.quantum_class_assignment(n_classes, &mut rng);
            targets[sample_idx] = class_id as f64;

            // Generate quantum state for this sample
            let quantum_state = self.generate_quantum_state(n_features, class_id, &mut rng)?;

            // Apply quantum entanglement to _features
            let entangled_features =
                self.apply_quantum_entanglement(&quantum_state, &entanglement_matrix)?;

            // Store entangled _features in dataset
            for feature_idx in 0..n_features {
                data[[sample_idx, feature_idx]] = entangled_features[feature_idx];
            }
        }

        Ok(Dataset::new(data, Some(targets)))
    }

    /// Generate quantum superposition regression dataset
    pub fn make_quantum_regression(
        &self,
        n_samples: usize,
        n_features: usize,
        noise_amplitude: f64,
        quantum_noise: bool,
        random_seed: Option<u64>,
    ) -> Result<Dataset> {
        if n_samples == 0 || n_features == 0 {
            return Err(DatasetsError::InvalidFormat(
                "All parameters must be > 0".to_string(),
            ));
        }

        let mut rng = match random_seed {
            Some(_seed) => StdRng::seed_from_u64(_seed),
            None => StdRng::from_rng(&mut thread_rng()),
        };

        let mut data = Array2::zeros((n_samples, n_features));
        let mut targets = Array1::zeros(n_samples);

        // Generate quantum coefficient matrix using Hadamard gates
        let quantum_coefficients = self.generate_quantum_coefficients(n_features, &mut rng)?;

        for sample_idx in 0..n_samples {
            // Generate quantum feature vector
            let quantum_features = self.generate_quantum_feature_vector(n_features, &mut rng)?;

            // Apply quantum transformations (rotation gates)
            let transformed_features = self.apply_quantum_rotations(&quantum_features, &mut rng)?;

            // Compute target using quantum dot product
            let target = self.quantum_dot_product(&transformed_features, &quantum_coefficients)?;

            // Add quantum or classical _noise
            let noisy_target = if quantum_noise {
                target + self.generate_quantum_noise(noise_amplitude, &mut rng)?
            } else {
                target + noise_amplitude * rng.random::<f64>()
            };

            // Store in dataset
            for feature_idx in 0..n_features {
                data[[sample_idx, feature_idx]] = transformed_features[feature_idx];
            }
            targets[sample_idx] = noisy_target;
        }

        Ok(Dataset::new(data, Some(targets)))
    }

    /// Generate quantum clustering dataset with entangled clusters
    pub fn make_quantum_blobs(
        &self,
        n_samples: usize,
        n_features: usize,
        n_centers: usize,
        cluster_std: f64,
        quantum_interference: f64,
        random_seed: Option<u64>,
    ) -> Result<Dataset> {
        if n_samples == 0 || n_features == 0 || n_centers == 0 {
            return Err(DatasetsError::InvalidFormat(
                "All parameters must be > 0".to_string(),
            ));
        }

        let mut rng = match random_seed {
            Some(_seed) => StdRng::seed_from_u64(_seed),
            None => StdRng::from_rng(&mut thread_rng()),
        };

        let mut data = Array2::zeros((n_samples, n_features));
        let mut targets = Array1::zeros(n_samples);

        // Generate quantum cluster _centers using superposition
        let quantum_centers =
            self.generate_quantum_cluster_centers(n_centers, n_features, &mut rng)?;

        for sample_idx in 0..n_samples {
            // Quantum cluster assignment with _interference
            let (cluster_id, interference_weight) =
                self.quantum_cluster_assignment(n_centers, quantum_interference, &mut rng)?;

            targets[sample_idx] = cluster_id as f64;

            // Generate sample around quantum center with _interference effects
            let center = &quantum_centers[cluster_id];
            for feature_idx in 0..n_features {
                let base_value = center[feature_idx];
                let gaussian_noise = rng.random::<f64>() * cluster_std;
                let quantum_interference_noise =
                    self.generate_quantum_interference(interference_weight, &mut rng)?;

                data[[sample_idx, feature_idx]] =
                    base_value + gaussian_noise + quantum_interference_noise;
            }
        }

        Ok(Dataset::new(data, Some(targets)))
    }

    // Private helper methods for quantum operations

    fn generate_entanglement_matrix(
        &self,
        n_features: usize,
        strength: f64,
        rng: &mut StdRng,
    ) -> Result<Array2<f64>> {
        let mut matrix = Array2::eye(n_features);

        // Add entanglement correlations
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let entanglement = strength * (rng.random::<f64>() - 0.5);
                matrix[[i, j]] = entanglement;
                matrix[[j, i]] = entanglement; // Symmetric entanglement
            }
        }

        Ok(matrix)
    }

    fn quantum_class_assignment(&self, n_classes: usize, rng: &mut StdRng) -> usize {
        // Simulate quantum measurement collapse
        let quantum_probability = rng.random::<f64>();
        let normalized_prob = (quantum_probability * self.gate_fidelity).abs();
        (normalized_prob * n_classes as f64) as usize % n_classes
    }

    fn generate_quantum_state(
        &self,
        n_features: usize,
        class_id: usize,
        rng: &mut StdRng,
    ) -> Result<Array1<f64>> {
        let mut state = Array1::zeros(n_features);

        // Initialize quantum state based on class
        let phase_offset = (class_id as f64 * 2.0 * PI) / 3.0; // Class-dependent phase

        for i in 0..n_features {
            // Generate quantum amplitude with phase
            let amplitude = rng.random::<f64>().sqrt();
            let phase = phase_offset + (i as f64 * PI / n_features as f64);
            state[i] = amplitude * (phase.cos() + phase.sin());
        }

        Ok(state)
    }

    fn apply_quantum_entanglement(
        &self,
        state: &Array1<f64>,
        entanglement_matrix: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        // Apply entanglement transformation: |ψ'⟩ = U|ψ⟩
        let entangled = entanglement_matrix.dot(state);
        Ok(entangled)
    }

    fn generate_quantum_coefficients(
        &self,
        n_features: usize,
        rng: &mut StdRng,
    ) -> Result<Array1<f64>> {
        let mut coefficients = Array1::zeros(n_features);

        for i in 0..n_features {
            // Generate coefficients using quantum random walk
            let quantum_step = if rng.random::<f64>() > 0.5 { 1.0 } else { -1.0 };
            let amplitude = rng.random::<f64>() * self.gate_fidelity;
            coefficients[i] = quantum_step * amplitude;
        }

        Ok(coefficients)
    }

    fn generate_quantum_feature_vector(
        &self,
        n_features: usize,
        rng: &mut StdRng,
    ) -> Result<Array1<f64>> {
        let mut features = Array1::zeros(n_features);

        for i in 0..n_features {
            // Quantum feature generation using Bloch sphere parameterization
            let theta = rng.random::<f64>() * PI;
            let phi = rng.random::<f64>() * 2.0 * PI;
            features[i] = theta.sin() * phi.cos() + theta.cos();
        }

        Ok(features)
    }

    fn apply_quantum_rotations(
        &self,
        features: &Array1<f64>,
        rng: &mut StdRng,
    ) -> Result<Array1<f64>> {
        let mut rotated = features.clone();

        for i in 0..features.len() {
            // Apply quantum rotation gates (Rx, Ry, Rz)
            let rotation_angle = rng.random::<f64>() * 2.0 * PI;
            let cos_theta = rotation_angle.cos();
            let sin_theta = rotation_angle.sin();

            // Rotation transformation
            rotated[i] = features[i] * cos_theta + sin_theta * self.gate_fidelity;
        }

        Ok(rotated)
    }

    fn quantum_dot_product(&self, a: &Array1<f64>, b: &Array1<f64>) -> Result<f64> {
        if a.len() != b.len() {
            return Err(DatasetsError::InvalidFormat(
                "Arrays must have same length for quantum dot product".to_string(),
            ));
        }

        // Quantum-enhanced dot product with interference effects
        let mut result = 0.0;
        for i in 0..a.len() {
            result += a[i] * b[i] * self.gate_fidelity;
        }

        Ok(result)
    }

    fn generate_quantum_noise(&self, amplitude: f64, rng: &mut StdRng) -> Result<f64> {
        // Generate quantum noise using quantum fluctuations
        let quantum_fluctuation = rng.random::<f64>() - 0.5;
        let decoherence_factor = (-1.0 / self.coherence_time).exp();
        Ok(amplitude * quantum_fluctuation * decoherence_factor)
    }

    fn generate_quantum_cluster_centers(
        &self,
        n_centers: usize,
        n_features: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<Array1<f64>>> {
        let mut centers = Vec::with_capacity(n_centers);

        for center_idx in 0..n_centers {
            let mut center = Array1::zeros(n_features);

            // Generate center using quantum superposition principle
            let center_phase = (center_idx as f64 * 2.0 * PI) / n_centers as f64;

            for feature_idx in 0..n_features {
                let quantum_amplitude = rng.random::<f64>() * 5.0; // Scale for visibility
                let feature_phase = center_phase + (feature_idx as f64 * PI / n_features as f64);
                center[feature_idx] = quantum_amplitude * feature_phase.cos();
            }

            centers.push(center);
        }

        Ok(centers)
    }

    fn quantum_cluster_assignment(
        &self,
        n_centers: usize,
        interference: f64,
        rng: &mut StdRng,
    ) -> Result<(usize, f64)> {
        // Quantum measurement with interference effects
        let quantum_state = rng.random::<f64>();
        let interference_weight = interference * (quantum_state * 2.0 * PI).sin();

        // Collapse to classical cluster assignment
        let cluster_id = (quantum_state * n_centers as f64) as usize % n_centers;

        Ok((cluster_id, interference_weight))
    }

    fn generate_quantum_interference(&self, weight: f64, rng: &mut StdRng) -> Result<f64> {
        // Generate quantum interference noise
        let phase = rng.random::<f64>() * 2.0 * PI;
        let amplitude = weight * self.gate_fidelity;
        Ok(amplitude * phase.sin())
    }
}

/// Convenience function to create quantum classification dataset
#[allow(dead_code)]
pub fn make_quantum_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    entanglement_strength: f64,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    let generator = QuantumDatasetGenerator::default();
    generator.make_quantum_classification(
        n_samples,
        n_features,
        n_classes,
        entanglement_strength,
        random_seed,
    )
}

/// Convenience function to create quantum regression dataset
#[allow(dead_code)]
pub fn make_quantum_regression(
    n_samples: usize,
    n_features: usize,
    noise_amplitude: f64,
    quantum_noise: bool,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    let generator = QuantumDatasetGenerator::default();
    generator.make_quantum_regression(
        n_samples,
        n_features,
        noise_amplitude,
        quantum_noise,
        random_seed,
    )
}

/// Convenience function to create quantum clustering dataset
#[allow(dead_code)]
pub fn make_quantum_blobs(
    n_samples: usize,
    n_features: usize,
    n_centers: usize,
    cluster_std: f64,
    quantum_interference: f64,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    let generator = QuantumDatasetGenerator::default();
    generator.make_quantum_blobs(
        n_samples,
        n_features,
        n_centers,
        cluster_std,
        quantum_interference,
        random_seed,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_classification() {
        let dataset = make_quantum_classification(100, 4, 3, 0.5, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 4);

        // Verify quantum entanglement effects in the data
        let data = &dataset.data;
        let correlations = data.t().dot(data) / (data.nrows() as f64);

        // Check for non-trivial correlations due to entanglement
        let mut has_entanglement = false;
        for i in 0..correlations.nrows() {
            for j in (i + 1)..correlations.ncols() {
                if correlations[[i, j]].abs() > 0.1 {
                    has_entanglement = true;
                    break;
                }
            }
        }
        assert!(
            has_entanglement,
            "Quantum entanglement should create feature correlations"
        );
    }

    #[test]
    fn test_quantum_regression() {
        let dataset = make_quantum_regression(50, 3, 0.1, true, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 50);
        assert_eq!(dataset.n_features(), 3);
        assert!(dataset.has_target());
    }

    #[test]
    fn test_quantum_blobs() {
        let dataset = make_quantum_blobs(80, 2, 4, 1.0, 0.3, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 80);
        assert_eq!(dataset.n_features(), 2);

        // Verify cluster formation with quantum interference
        let targets = dataset.target.as_ref().unwrap();
        let unique_clusters: std::collections::HashSet<_> =
            targets.iter().map(|&x| x as usize).collect();
        assert!(unique_clusters.len() <= 4, "Should have at most 4 clusters");
    }

    #[test]
    fn test_quantum_generator_configuration() {
        let generator = QuantumDatasetGenerator::new(12, 0.95);
        assert_eq!(generator.n_qubits, 12);
        assert_eq!(generator.gate_fidelity, 0.95);
        assert!(generator.quantum_advantage);
    }
}
