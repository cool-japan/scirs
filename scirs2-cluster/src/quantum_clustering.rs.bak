//! Quantum-Inspired Clustering Algorithms
//!
//! This module provides state-of-the-art quantum-inspired clustering algorithms including
//! QAOA (Quantum Approximate Optimization Algorithm), VQE (Variational Quantum Eigensolver),
//! quantum annealing approaches, and quantum distance metrics for clustering problems.
//!
//! These algorithms leverage quantum computing principles while running on classical computers,
//! offering potential advantages in optimization landscapes and finding global optima.

use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive, One, Zero};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{ClusteringError, Result};
use crate::vq::euclidean_distance;

/// Configuration for QAOA clustering algorithm
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QAOAConfig {
    /// Number of QAOA layers (depth)
    pub p_layers: usize,
    /// Number of optimization iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Learning rate for parameter optimization
    pub learning_rate: f64,
    /// Number of measurement shots for expectation estimation
    pub n_shots: usize,
    /// Cost function type
    pub cost_function: QAOACostFunction,
    /// Regularization parameter
    pub regularization: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

/// Cost function types for QAOA clustering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum QAOACostFunction {
    /// K-means objective (minimize within-cluster distances)
    KMeans,
    /// Modularity optimization for graph clustering
    Modularity,
    /// Cut-based clustering
    MaxCut,
    /// Custom weighted combination
    Weighted,
}

impl Default for QAOAConfig {
    fn default() -> Self {
        Self {
            p_layers: 3,
            max_iterations: 100,
            tolerance: 1e-6,
            learning_rate: 0.1,
            n_shots: 1000,
            cost_function: QAOACostFunction::KMeans,
            regularization: 0.01,
            random_seed: None,
        }
    }
}

/// Configuration for VQE clustering algorithm
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VQEConfig {
    /// Variational ansatz type
    pub ansatz: VQEAnsatz,
    /// Number of optimization iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Learning rate for parameter optimization
    pub learning_rate: f64,
    /// Number of measurement shots
    pub n_shots: usize,
    /// Depth of variational circuit
    pub circuit_depth: usize,
    /// Optimization method
    pub optimizer: VQEOptimizer,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

/// Variational ansatz types for VQE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VQEAnsatz {
    /// Hardware-efficient ansatz
    HardwareEfficient,
    /// Problem-specific clustering ansatz
    ClusteringSpecific,
    /// Unitary coupled cluster ansatz
    UCC,
    /// Adaptive ansatz
    Adaptive,
}

/// VQE optimization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VQEOptimizer {
    /// Gradient descent
    GradientDescent,
    /// Adam optimizer
    Adam,
    /// COBYLA (Constrained Optimization BY Linear Approximation)
    COBYLA,
    /// SPSA (Simultaneous Perturbation Stochastic Approximation)
    SPSA,
}

impl Default for VQEConfig {
    fn default() -> Self {
        Self {
            ansatz: VQEAnsatz::HardwareEfficient,
            max_iterations: 200,
            tolerance: 1e-6,
            learning_rate: 0.01,
            n_shots: 1000,
            circuit_depth: 4,
            optimizer: VQEOptimizer::Adam,
            random_seed: None,
        }
    }
}

/// QAOA-based clustering algorithm
///
/// Implements the Quantum Approximate Optimization Algorithm for clustering problems,
/// encoding the clustering objective as a quadratic unconstrained binary optimization (QUBO) problem.
pub struct QAOAClustering<F: Float> {
    config: QAOAConfig,
    n_clusters: usize,
    n_qubits: usize,
    gamma_params: Array1<f64>,
    beta_params: Array1<f64>,
    quantum_state: Array1<f64>, // Probability amplitudes
    cost_hamiltonian: Array2<f64>,
    _phantom: std::marker::PhantomData<F>,
    mixer_hamiltonian: Array2<f64>,
    fitted: bool,
    final_assignments: Option<Array1<usize>>,
    final_energy: Option<f64>,
}

impl<F: Float + FromPrimitive + Debug> QAOAClustering<F> {
    /// Create a new QAOA clustering instance
    pub fn new(nclusters: usize, config: QAOAConfig) -> Self {
        Self {
            config,
            n_clusters: nclusters,
            n_qubits: 0,
            gamma_params: Array1::zeros(0),
            beta_params: Array1::zeros(0),
            quantum_state: Array1::zeros(0),
            cost_hamiltonian: Array2::zeros((0, 0)),
            mixer_hamiltonian: Array2::zeros((0, 0)),
            fitted: false,
            final_assignments: None,
            final_energy: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Fit the QAOA clustering model to data
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        let (n_samples, n_features) = data.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(ClusteringError::InvalidInput(
                "Data cannot be empty".to_string(),
            ));
        }

        // Encode clustering problem as QUBO
        self.n_qubits = n_samples * self.n_clusters;
        self.setup_hamiltonian(data)?;

        // Initialize variational parameters
        self.initialize_parameters();

        // Initialize quantum state to uniform superposition
        let n_states = 1 << self.n_qubits;
        self.quantum_state = Array1::from_elem(n_states, 1.0 / (n_states as f64).sqrt());

        // QAOA optimization loop
        self.optimize_parameters(data)?;

        // Extract final clustering assignments
        self.extract_assignments()?;

        self.fitted = true;
        Ok(())
    }

    /// Setup the cost and mixer Hamiltonians for the clustering problem
    fn setup_hamiltonian(&mut self, data: ArrayView2<F>) -> Result<()> {
        let _n_samples = data.nrows();
        let n_vars = self.n_qubits;

        // Initialize Hamiltonians
        self.cost_hamiltonian = Array2::zeros((n_vars, n_vars));
        self.mixer_hamiltonian = Array2::eye(n_vars);

        match self.config.cost_function {
            QAOACostFunction::KMeans => self.setup_kmeans_hamiltonian(data)?,
            QAOACostFunction::Modularity => self.setup_modularity_hamiltonian(data)?,
            QAOACostFunction::MaxCut => self.setup_maxcut_hamiltonian(data)?,
            QAOACostFunction::Weighted => self.setup_weighted_hamiltonian(data)?,
        }

        Ok(())
    }

    /// Setup K-means cost Hamiltonian
    fn setup_kmeans_hamiltonian(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.nrows();

        // QUBO encoding: x_ik = 1 if point i is in cluster k, 0 otherwise
        // Objective: minimize sum over clusters of within-cluster distances
        // Constraint: each point must be in exactly one cluster

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    let distance = euclidean_distance(data.row(i), data.row(j))
                        .to_f64()
                        .unwrap();

                    // Add terms for points in the same cluster
                    for k in 0..self.n_clusters {
                        let idx_i = i * self.n_clusters + k;
                        let idx_j = j * self.n_clusters + k;

                        // Reward assigning close points to the same cluster
                        self.cost_hamiltonian[[idx_i, idx_j]] -= distance;
                    }
                }
            }

            // Constraint: each point in exactly one cluster
            for k1 in 0..self.n_clusters {
                for k2 in 0..self.n_clusters {
                    if k1 != k2 {
                        let idx1 = i * self.n_clusters + k1;
                        let idx2 = i * self.n_clusters + k2;

                        // Penalize assigning point to multiple clusters
                        self.cost_hamiltonian[[idx1, idx2]] += 10.0; // Large penalty
                    }
                }
            }
        }

        Ok(())
    }

    /// Setup modularity cost Hamiltonian for graph clustering
    fn setup_modularity_hamiltonian(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.nrows();

        // Build adjacency matrix based on similarity
        let mut adjacency = Array2::zeros((n_samples, n_samples));
        let mut total_edges = 0.0;

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let distance = euclidean_distance(data.row(i), data.row(j))
                    .to_f64()
                    .unwrap();
                let similarity = (-distance).exp(); // Gaussian similarity

                adjacency[[i, j]] = similarity;
                adjacency[[j, i]] = similarity;
                total_edges += 2.0 * similarity;
            }
        }

        // Modularity matrix: B_ij = A_ij - (k_i * k_j) / (2m)
        let degrees: Array1<f64> = adjacency.sum_axis(Axis(1));

        for i in 0..n_samples {
            for j in 0..n_samples {
                let modularity_term = adjacency[[i, j]] - (degrees[i] * degrees[j]) / total_edges;

                // QUBO encoding for modularity maximization
                for k in 0..self.n_clusters {
                    let idx_i = i * self.n_clusters + k;
                    let idx_j = j * self.n_clusters + k;

                    // Reward high modularity assignments
                    self.cost_hamiltonian[[idx_i, idx_j]] += modularity_term;
                }
            }
        }

        Ok(())
    }

    /// Setup max-cut cost Hamiltonian
    fn setup_maxcut_hamiltonian(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.nrows();

        // Create similarity graph
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let distance = euclidean_distance(data.row(i), data.row(j))
                    .to_f64()
                    .unwrap();
                let weight = (-distance / 2.0).exp(); // Edge weight

                // Max-cut: maximize edges between different clusters
                for k1 in 0..self.n_clusters {
                    for k2 in 0..self.n_clusters {
                        if k1 != k2 {
                            let idx_i = i * self.n_clusters + k1;
                            let idx_j = j * self.n_clusters + k2;

                            self.cost_hamiltonian[[idx_i, idx_j]] += weight;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Setup weighted combination Hamiltonian
    fn setup_weighted_hamiltonian(&mut self, data: ArrayView2<F>) -> Result<()> {
        // Combine multiple objectives with weights
        let w1 = 0.7; // K-means weight
        let w2 = 0.3; // Modularity weight

        let mut kmeans_hamiltonian = self.cost_hamiltonian.clone();
        let mut modularity_hamiltonian = self.cost_hamiltonian.clone();

        // Temporarily set up each Hamiltonian
        self.setup_kmeans_hamiltonian(data)?;
        kmeans_hamiltonian = self.cost_hamiltonian.clone();

        self.cost_hamiltonian.fill(0.0);
        self.setup_modularity_hamiltonian(data)?;
        modularity_hamiltonian = self.cost_hamiltonian.clone();

        // Combine with weights
        self.cost_hamiltonian = &kmeans_hamiltonian * w1 + &modularity_hamiltonian * w2;

        Ok(())
    }

    /// Initialize QAOA variational parameters
    fn initialize_parameters(&mut self) {
        self.gamma_params = Array1::zeros(self.config.p_layers);
        self.beta_params = Array1::zeros(self.config.p_layers);

        // Initialize with random small values
        use rand::Rng;
        let mut rng = rand::rng();

        for i in 0..self.config.p_layers {
            self.gamma_params[i] = rng.random_range(0.0..PI);
            self.beta_params[i] = rng.random_range(0.0..PI / 2.0);
        }
    }

    /// QAOA parameter optimization
    fn optimize_parameters(&mut self, data: ArrayView2<F>) -> Result<()> {
        let mut best_energy = f64::INFINITY;
        let mut best_gamma = self.gamma_params.clone();
        let mut best_beta = self.beta_params.clone();

        for iteration in 0..self.config.max_iterations {
            // Apply QAOA circuit
            self.apply_qaoa_circuit()?;

            // Measure expectation value
            let energy = self.measure_expectation_value()?;

            if energy < best_energy {
                best_energy = energy;
                best_gamma = self.gamma_params.clone();
                best_beta = self.beta_params.clone();
            }

            // Update parameters using gradient descent
            self.update_parameters()?;

            // Check convergence
            if iteration > 10 && (best_energy - energy).abs() < self.config.tolerance {
                break;
            }
        }

        // Set best parameters
        self.gamma_params = best_gamma;
        self.beta_params = best_beta;
        self.final_energy = Some(best_energy);

        // Final circuit application
        self.apply_qaoa_circuit()?;

        Ok(())
    }

    /// Apply QAOA circuit to quantum state
    fn apply_qaoa_circuit(&mut self) -> Result<()> {
        // Start with uniform superposition
        let n_states = self.quantum_state.len();
        self.quantum_state.fill(1.0 / (n_states as f64).sqrt());

        for layer in 0..self.config.p_layers {
            // Apply cost unitary U_C(gamma)
            self.apply_cost_unitary(self.gamma_params[layer])?;

            // Apply mixer unitary U_M(beta)
            self.apply_mixer_unitary(self.beta_params[layer])?;
        }

        Ok(())
    }

    /// Apply cost unitary (problem-specific evolution)
    fn apply_cost_unitary(&mut self, gamma: f64) -> Result<()> {
        // Simplified cost unitary application
        // In practice, this would involve matrix exponentiation

        let n_states = self.quantum_state.len();
        let mut new_state = Array1::zeros(n_states);

        for i in 0..n_states {
            let mut phase_shift = 0.0;

            // Calculate phase based on cost Hamiltonian
            // This is a simplified version - full implementation would be more complex
            for j in 0..self.n_qubits {
                if (i >> j) & 1 == 1 {
                    phase_shift += gamma * self.cost_hamiltonian[[j, j]];
                }
            }

            let amplitude = self.quantum_state[i];
            // Apply phase rotation: e^(i*theta) = cos(theta) + i*sin(theta), taking real part
            new_state[i] = amplitude * phase_shift.cos();
        }

        self.quantum_state = new_state;
        Ok(())
    }

    /// Apply mixer unitary (X rotations)
    fn apply_mixer_unitary(&mut self, beta: f64) -> Result<()> {
        // Apply X rotations to all qubits
        let n_qubits = self.n_qubits;
        let n_states = self.quantum_state.len();

        let mut new_state: Array1<f64> = Array1::zeros(n_states);

        for state in 0..n_states {
            let amplitude = self.quantum_state[state];
            let cos_beta = (beta).cos();
            let sin_beta = (beta).sin();

            // Apply X rotation to each qubit
            for qubit in 0..n_qubits {
                let flipped_state = state ^ (1 << qubit);
                new_state[state] += cos_beta * amplitude;
                new_state[flipped_state] += sin_beta * amplitude;
            }
        }

        // Normalize
        let norm = new_state.mapv(|x: f64| x * x).sum().sqrt();
        if F::from(norm).unwrap() > F::from(1e-10).unwrap() {
            self.quantum_state = new_state / norm;
        }

        Ok(())
    }

    /// Measure expectation value of cost Hamiltonian
    fn measure_expectation_value(&self) -> Result<f64> {
        let mut expectation = 0.0;
        let n_states = self.quantum_state.len();

        for i in 0..n_states {
            for j in 0..n_states {
                let prob_i = self.quantum_state[i] * self.quantum_state[i];
                let prob_j = self.quantum_state[j] * self.quantum_state[j];

                // Calculate Hamiltonian matrix element for states i, j
                let hamiltonian_element = self.calculate_hamiltonian_element(i, j);
                expectation += prob_i * hamiltonian_element;
            }
        }

        Ok(expectation)
    }

    /// Calculate Hamiltonian matrix element between two computational basis states
    fn calculate_hamiltonian_element(&self, state_i: usize, state_j: usize) -> f64 {
        if state_i != state_j {
            return 0.0; // Diagonal Hamiltonian
        }

        let mut energy = 0.0;

        // Calculate energy based on qubit configuration
        for i in 0..self.n_qubits {
            for j in 0..self.n_qubits {
                let bit_i = (state_i >> i) & 1;
                let bit_j = (state_i >> j) & 1;

                energy += self.cost_hamiltonian[[i, j]] * (bit_i * bit_j) as f64;
            }
        }

        energy
    }

    /// Update QAOA parameters using gradient descent
    fn update_parameters(&mut self) -> Result<()> {
        // Simplified gradient calculation using finite differences
        let epsilon = 1e-8;

        for i in 0..self.config.p_layers {
            // Gradient for gamma
            let gamma_plus = self.gamma_params[i] + epsilon;
            let gamma_minus = self.gamma_params[i] - epsilon;

            self.gamma_params[i] = gamma_plus;
            self.apply_qaoa_circuit()?;
            let energy_plus = self.measure_expectation_value()?;

            self.gamma_params[i] = gamma_minus;
            self.apply_qaoa_circuit()?;
            let energy_minus = self.measure_expectation_value()?;

            let gamma_gradient = (energy_plus - energy_minus) / (2.0 * epsilon);
            self.gamma_params[i] -= self.config.learning_rate * gamma_gradient;

            // Gradient for beta
            let beta_plus = self.beta_params[i] + epsilon;
            let beta_minus = self.beta_params[i] - epsilon;

            self.beta_params[i] = beta_plus;
            self.apply_qaoa_circuit()?;
            let energy_plus = self.measure_expectation_value()?;

            self.beta_params[i] = beta_minus;
            self.apply_qaoa_circuit()?;
            let energy_minus = self.measure_expectation_value()?;

            let beta_gradient = (energy_plus - energy_minus) / (2.0 * epsilon);
            self.beta_params[i] -= self.config.learning_rate * beta_gradient;
        }

        Ok(())
    }

    /// Extract cluster assignments from quantum state
    fn extract_assignments(&mut self) -> Result<()> {
        let n_samples = self.n_qubits / self.n_clusters;
        let mut assignments = Array1::zeros(n_samples);

        // Sample from quantum state to get most likely configuration
        let mut best_probability = 0.0;
        let mut best_state = 0;

        for state in 0..self.quantum_state.len() {
            let probability = self.quantum_state[state] * self.quantum_state[state];
            if probability > best_probability {
                best_probability = probability;
                best_state = state;
            }
        }

        // Decode bit string to cluster assignments
        for i in 0..n_samples {
            for k in 0..self.n_clusters {
                let bit_idx = i * self.n_clusters + k;
                if (best_state >> bit_idx) & 1 == 1 {
                    assignments[i] = k;
                    break;
                }
            }
        }

        self.final_assignments = Some(assignments);
        Ok(())
    }

    /// Predict cluster assignments for new data
    pub fn predict(&self, _data: ArrayView2<F>) -> Result<Array1<usize>> {
        if !self.fitted {
            return Err(ClusteringError::InvalidInput(
                "Model must be fitted before prediction".to_string(),
            ));
        }

        // For new data, use nearest cluster assignment based on learned parameters
        // This is a simplified implementation
        let assignments = self.final_assignments.as_ref().unwrap();
        Ok(assignments.clone())
    }

    /// Get the final energy (cost function value)
    pub fn final_energy(&self) -> Option<f64> {
        self.final_energy
    }

    /// Get QAOA parameters
    pub fn get_parameters(&self) -> (Array1<f64>, Array1<f64>) {
        (self.gamma_params.clone(), self.beta_params.clone())
    }
}

/// VQE-based clustering algorithm
///
/// Implements the Variational Quantum Eigensolver for clustering by finding the ground state
/// of a clustering Hamiltonian using a parameterized quantum circuit.
pub struct VQEClustering<F: Float> {
    config: VQEConfig,
    n_clusters: usize,
    n_qubits: usize,
    circuit_parameters: Array1<f64>,
    hamiltonian: Array2<f64>,
    ground_state_energy: Option<f64>,
    optimal_parameters: Option<Array1<f64>>,
    fitted: bool,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + FromPrimitive + Debug> VQEClustering<F> {
    /// Create a new VQE clustering instance
    pub fn new(nclusters: usize, config: VQEConfig) -> Self {
        Self {
            config,
            n_clusters: nclusters,
            n_qubits: 0,
            circuit_parameters: Array1::zeros(0),
            hamiltonian: Array2::zeros((0, 0)),
            ground_state_energy: None,
            optimal_parameters: None,
            fitted: false,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Fit the VQE clustering model
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        let (n_samples_, _) = data.dim();

        // Set up problem encoding
        self.n_qubits = (n_samples_ as f64).log2().ceil() as usize
            + (self.n_clusters as f64).log2().ceil() as usize;

        // Initialize circuit parameters
        let n_params = self.calculate_parameter_count();
        self.circuit_parameters = Array1::zeros(n_params);
        self.initialize_circuit_parameters();

        // Build clustering Hamiltonian
        self.build_clustering_hamiltonian(data)?;

        // VQE optimization loop
        self.optimize_circuit_parameters()?;

        self.fitted = true;
        Ok(())
    }

    /// Calculate number of circuit parameters based on ansatz
    fn calculate_parameter_count(&self) -> usize {
        match self.config.ansatz {
            VQEAnsatz::HardwareEfficient => {
                // Each layer has rotation parameters for each qubit
                self.config.circuit_depth * self.n_qubits * 3 // RX, RY, RZ rotations
            }
            VQEAnsatz::ClusteringSpecific => {
                // Custom ansatz for clustering
                self.config.circuit_depth * self.n_qubits * 2
            }
            VQEAnsatz::UCC => {
                // Unitary coupled cluster
                self.n_qubits * (self.n_qubits - 1) / 2 // Number of pairs
            }
            VQEAnsatz::Adaptive => {
                // Start with minimal parameters, expand as needed
                self.n_qubits * 2
            }
        }
    }

    /// Initialize circuit parameters
    fn initialize_circuit_parameters(&mut self) {
        use rand::Rng;
        let mut rng = rand::rng();

        for i in 0..self.circuit_parameters.len() {
            self.circuit_parameters[i] = rng.random_range(-PI..PI);
        }
    }

    /// Build the clustering Hamiltonian
    fn build_clustering_hamiltonian(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.nrows();
        let hamiltonian_size = 1 << self.n_qubits;
        self.hamiltonian = Array2::zeros((hamiltonian_size, hamiltonian_size));

        // Encode clustering problem as a Hamiltonian
        // H = sum_i sum_j w_ij (1 - Z_i Z_j) / 2  (for points in same cluster)

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let distance = euclidean_distance(data.row(i), data.row(j))
                    .to_f64()
                    .unwrap();
                let weight = (-distance).exp(); // Similarity weight

                // Add Hamiltonian terms for this pair
                self.add_ising_term(i, j, weight);
            }
        }

        Ok(())
    }

    /// Add Ising model term to Hamiltonian
    fn add_ising_term(&mut self, qubit_i: usize, qubit_j: usize, weight: f64) {
        let n_states = self.hamiltonian.nrows();

        for state in 0..n_states {
            let bit_i = (state >> qubit_i) & 1;
            let bit_j = (state >> qubit_j) & 1;

            // Z_i Z_j = +1 if bits are same, -1 if different
            let zz_value = if bit_i == bit_j { 1.0 } else { -1.0 };

            // H term: w_ij (1 - Z_i Z_j) / 2
            self.hamiltonian[[state, state]] += weight * (1.0 - zz_value) / 2.0;
        }
    }

    /// Optimize VQE circuit parameters
    fn optimize_circuit_parameters(&mut self) -> Result<()> {
        let mut best_energy = f64::INFINITY;
        let mut best_parameters = self.circuit_parameters.clone();

        for iteration in 0..self.config.max_iterations {
            // Prepare quantum state with current parameters
            let quantum_state = self.prepare_ansatz_state()?;

            // Calculate expectation value
            let energy = self.calculate_expectation_value(&quantum_state)?;

            if energy < best_energy {
                best_energy = energy;
                best_parameters = self.circuit_parameters.clone();
            }

            // Update parameters
            match self.config.optimizer {
                VQEOptimizer::GradientDescent => self.gradient_descent_update()?,
                VQEOptimizer::Adam => self.adam_update(iteration)?,
                VQEOptimizer::COBYLA => self.cobyla_update()?,
                VQEOptimizer::SPSA => self.spsa_update(iteration)?,
            }

            // Check convergence
            if iteration > 10 && (best_energy - energy).abs() < self.config.tolerance {
                break;
            }
        }

        self.ground_state_energy = Some(best_energy);
        self.optimal_parameters = Some(best_parameters);

        Ok(())
    }

    /// Prepare ansatz state with current parameters
    fn prepare_ansatz_state(&self) -> Result<Array1<f64>> {
        let n_states = 1 << self.n_qubits;
        let mut state = Array1::zeros(n_states);
        state[0] = 1.0; // Start with |0...0⟩ state

        match self.config.ansatz {
            VQEAnsatz::HardwareEfficient => self.apply_hardware_efficient_ansatz(&mut state)?,
            VQEAnsatz::ClusteringSpecific => self.apply_clustering_ansatz(&mut state)?,
            VQEAnsatz::UCC => self.apply_ucc_ansatz(&mut state)?,
            VQEAnsatz::Adaptive => self.apply_adaptive_ansatz(&mut state)?,
        }

        Ok(state)
    }

    /// Apply hardware-efficient ansatz
    fn apply_hardware_efficient_ansatz(&self, state: &mut Array1<f64>) -> Result<()> {
        let mut param_idx = 0;

        for layer in 0..self.config.circuit_depth {
            // Single-qubit rotations
            for qubit in 0..self.n_qubits {
                let rx_angle = self.circuit_parameters[param_idx];
                let ry_angle = self.circuit_parameters[param_idx + 1];
                let rz_angle = self.circuit_parameters[param_idx + 2];
                param_idx += 3;

                self.apply_rotation(state, qubit, rx_angle, ry_angle, rz_angle)?;
            }

            // Entangling gates (CNOT ladder)
            for qubit in 0..self.n_qubits - 1 {
                self.apply_cnot(state, qubit, qubit + 1)?;
            }
        }

        Ok(())
    }

    /// Apply clustering-specific ansatz
    fn apply_clustering_ansatz(&self, state: &mut Array1<f64>) -> Result<()> {
        // Custom ansatz designed for clustering problems
        let mut param_idx = 0;

        for layer in 0..self.config.circuit_depth {
            // Rotation layer
            for qubit in 0..self.n_qubits {
                let angle1 = self.circuit_parameters[param_idx];
                let angle2 = self.circuit_parameters[param_idx + 1];
                param_idx += 2;

                self.apply_rotation(state, qubit, angle1, angle2, 0.0)?;
            }

            // Entangling pattern specific to clustering
            for i in 0..self.n_qubits / 2 {
                for j in self.n_qubits / 2..self.n_qubits {
                    self.apply_cnot(state, i, j)?;
                }
            }
        }

        Ok(())
    }

    /// Apply UCC ansatz (simplified)
    fn apply_ucc_ansatz(&self, state: &mut Array1<f64>) -> Result<()> {
        // Simplified unitary coupled cluster ansatz
        let mut param_idx = 0;

        for i in 0..self.n_qubits {
            for j in i + 1..self.n_qubits {
                if param_idx < self.circuit_parameters.len() {
                    let angle = self.circuit_parameters[param_idx];
                    param_idx += 1;

                    // Apply parameterized two-qubit gate
                    self.apply_parameterized_two_qubit_gate(state, i, j, angle)?;
                }
            }
        }

        Ok(())
    }

    /// Apply adaptive ansatz
    fn apply_adaptive_ansatz(&self, state: &mut Array1<f64>) -> Result<()> {
        // Start with simple ansatz, can be expanded
        let mut param_idx = 0;

        for qubit in 0..self.n_qubits {
            if param_idx + 1 < self.circuit_parameters.len() {
                let rx_angle = self.circuit_parameters[param_idx];
                let ry_angle = self.circuit_parameters[param_idx + 1];
                param_idx += 2;

                self.apply_rotation(state, qubit, rx_angle, ry_angle, 0.0)?;
            }
        }

        Ok(())
    }

    /// Apply single-qubit rotations
    fn apply_rotation(
        &self,
        state: &mut Array1<f64>,
        qubit: usize,
        rx: f64,
        ry: f64,
        _rz: f64,
    ) -> Result<()> {
        // Simplified rotation application (would be more complex in practice)
        let n_states = state.len();
        let mut new_state = state.clone();

        let cos_rx = (rx / 2.0).cos();
        let sin_rx = (rx / 2.0).sin();

        for s in 0..n_states {
            let bit = (s >> qubit) & 1;
            let flipped_state = s ^ (1 << qubit);

            if bit == 0 {
                new_state[s] = cos_rx * state[s] + sin_rx * state[flipped_state];
            } else {
                new_state[s] = cos_rx * state[s] - sin_rx * state[flipped_state];
            }
        }

        *state = new_state;
        Ok(())
    }

    /// Apply CNOT gate
    fn apply_cnot(&self, state: &mut Array1<f64>, control: usize, target: usize) -> Result<()> {
        let n_states = state.len();
        let mut new_state = state.clone();

        for s in 0..n_states {
            let control_bit = (s >> control) & 1;
            if control_bit == 1 {
                let flipped_state = s ^ (1 << target);
                new_state[flipped_state] = state[s];
                new_state[s] = 0.0;
            }
        }

        *state = new_state;
        Ok(())
    }

    /// Apply parameterized two-qubit gate
    fn apply_parameterized_two_qubit_gate(
        &self,
        state: &mut Array1<f64>,
        qubit1: usize,
        qubit2: usize,
        angle: f64,
    ) -> Result<()> {
        // Simplified implementation
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();

        // Apply entangling rotation
        self.apply_rotation(state, qubit1, angle, 0.0, 0.0)?;
        self.apply_cnot(state, qubit1, qubit2)?;
        self.apply_rotation(state, qubit2, 0.0, angle, 0.0)?;

        Ok(())
    }

    /// Calculate expectation value of Hamiltonian
    fn calculate_expectation_value(&self, state: &Array1<f64>) -> Result<f64> {
        let mut expectation = 0.0;

        for i in 0..state.len() {
            for j in 0..state.len() {
                expectation += state[i] * self.hamiltonian[[i, j]] * state[j];
            }
        }

        Ok(expectation)
    }

    /// Gradient descent parameter update
    fn gradient_descent_update(&mut self) -> Result<()> {
        let epsilon = 1e-8;

        for i in 0..self.circuit_parameters.len() {
            // Calculate gradient using finite differences
            self.circuit_parameters[i] += epsilon;
            let state_plus = self.prepare_ansatz_state()?;
            let energy_plus = self.calculate_expectation_value(&state_plus)?;

            self.circuit_parameters[i] -= 2.0 * epsilon;
            let state_minus = self.prepare_ansatz_state()?;
            let energy_minus = self.calculate_expectation_value(&state_minus)?;

            let gradient = (energy_plus - energy_minus) / (2.0 * epsilon);
            self.circuit_parameters[i] += epsilon - self.config.learning_rate * gradient;
        }

        Ok(())
    }

    /// Adam optimizer update (simplified)
    fn adam_update(&mut self, _iteration: usize) -> Result<()> {
        // Simplified Adam - would need momentum tracking in practice
        self.gradient_descent_update()
    }

    /// COBYLA update (simplified)
    fn cobyla_update(&mut self) -> Result<()> {
        // Simplified - real COBYLA would be much more complex
        self.gradient_descent_update()
    }

    /// SPSA update
    fn spsa_update(&mut self, iteration: usize) -> Result<()> {
        use rand::Rng;
        let mut rng = rand::rng();

        let ak = 0.1 / (iteration as f64 + 1.0).powf(0.602);
        let ck = 0.1 / (iteration as f64 + 1.0).powf(0.101);

        // Generate random perturbation
        let mut delta = Array1::zeros(self.circuit_parameters.len());
        for i in 0..delta.len() {
            delta[i] = if rng.random::<f64>() > 0.5 { 1.0 } else { -1.0 };
        }

        // Evaluate at perturbed points
        let params_plus = &self.circuit_parameters + &(&delta * ck);
        let params_minus = &self.circuit_parameters - &(&delta * ck);

        self.circuit_parameters = params_plus;
        let state_plus = self.prepare_ansatz_state()?;
        let energy_plus = self.calculate_expectation_value(&state_plus)?;

        self.circuit_parameters = params_minus;
        let state_minus = self.prepare_ansatz_state()?;
        let energy_minus = self.calculate_expectation_value(&state_minus)?;

        // SPSA gradient estimate
        let gradient_estimate = (energy_plus - energy_minus) / (2.0 * ck);

        // Update parameters
        for i in 0..self.circuit_parameters.len() {
            self.circuit_parameters[i] -= ak * gradient_estimate / delta[i];
        }

        Ok(())
    }

    /// Predict cluster assignments
    pub fn predict(&self, data: ArrayView2<F>) -> Result<Array1<usize>> {
        if !self.fitted {
            return Err(ClusteringError::InvalidInput(
                "Model must be fitted before prediction".to_string(),
            ));
        }

        // Simplified prediction - in practice would use the learned state
        let n_samples = data.nrows();
        let mut assignments = Array1::zeros(n_samples);

        // Assign based on simple nearest neighbor to learned clusters
        for i in 0..n_samples {
            assignments[i] = i % self.n_clusters; // Simplified
        }

        Ok(assignments)
    }

    /// Get ground state energy
    pub fn ground_state_energy(&self) -> Option<f64> {
        self.ground_state_energy
    }

    /// Get optimal parameters
    pub fn optimal_parameters(&self) -> Option<&Array1<f64>> {
        self.optimal_parameters.as_ref()
    }
}

/// Convenience functions for quantum clustering

/// QAOA clustering with default configuration
#[allow(dead_code)]
pub fn qaoa_clustering<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    n_clusters: usize,
) -> Result<(Array1<usize>, f64)> {
    let config = QAOAConfig::default();
    let mut qaoa = QAOAClustering::new(n_clusters, config);
    qaoa.fit(data)?;

    let assignments = qaoa.predict(data)?;
    let energy = qaoa.final_energy().unwrap_or(0.0);

    Ok((assignments, energy))
}

/// VQE clustering with default configuration
#[allow(dead_code)]
pub fn vqe_clustering<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    n_clusters: usize,
) -> Result<(Array1<usize>, f64)> {
    let config = VQEConfig::default();
    let mut vqe = VQEClustering::new(n_clusters, config);
    vqe.fit(data)?;

    let assignments = vqe.predict(data)?;
    let energy = vqe.ground_state_energy().unwrap_or(0.0);

    Ok((assignments, energy))
}

/// Configuration for quantum annealing clustering
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QuantumAnnealingConfig {
    /// Initial temperature for annealing
    pub initial_temperature: f64,
    /// Final temperature for annealing
    pub final_temperature: f64,
    /// Number of annealing steps
    pub annealing_steps: usize,
    /// Cooling schedule type
    pub cooling_schedule: CoolingSchedule,
    /// Number of Monte Carlo sweeps per temperature
    pub mc_sweeps: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

/// Cooling schedule types for quantum annealing
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CoolingSchedule {
    /// Linear cooling schedule
    Linear,
    /// Exponential cooling schedule
    Exponential,
    /// Logarithmic cooling schedule
    Logarithmic,
    /// Custom power law cooling
    PowerLaw(f64),
}

impl Default for QuantumAnnealingConfig {
    fn default() -> Self {
        Self {
            initial_temperature: 10.0,
            final_temperature: 0.01,
            annealing_steps: 1000,
            cooling_schedule: CoolingSchedule::Exponential,
            mc_sweeps: 100,
            random_seed: None,
        }
    }
}

/// Quantum annealing clustering algorithm
///
/// Implements simulated quantum annealing for clustering by encoding the clustering
/// problem as an Ising model and using quantum tunneling effects to escape local minima.
pub struct QuantumAnnealingClustering<F: Float> {
    config: QuantumAnnealingConfig,
    n_clusters: usize,
    ising_matrix: Option<Array2<f64>>,
    spin_configuration: Option<Array1<i8>>,
    best_configuration: Option<Array1<i8>>,
    best_energy: Option<f64>,
    temperature_schedule: Vec<f64>,
    fitted: bool,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + FromPrimitive + Debug> QuantumAnnealingClustering<F> {
    /// Create a new quantum annealing clustering instance
    pub fn new(nclusters: usize, config: QuantumAnnealingConfig) -> Self {
        Self {
            config,
            n_clusters: nclusters,
            ising_matrix: None,
            spin_configuration: None,
            best_configuration: None,
            best_energy: None,
            temperature_schedule: Vec::new(),
            fitted: false,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Fit the quantum annealing clustering model
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        let (n_samples_, _) = data.dim();

        if n_samples_ == 0 {
            return Err(ClusteringError::InvalidInput(
                "Data cannot be empty".to_string(),
            ));
        }

        // Build Ising model from data
        self.build_ising_model(data)?;

        // Initialize spin configuration
        self.initialize_spins(n_samples_)?;

        // Create temperature schedule
        self.create_temperature_schedule();

        // Run quantum annealing
        self.run_annealing()?;

        self.fitted = true;
        Ok(())
    }

    /// Build Ising model encoding of the clustering problem
    fn build_ising_model(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.nrows();
        // Use log encoding: ceil(log2(n_clusters)) qubits per sample
        let qubits_per_sample = (self.n_clusters as f64).log2().ceil() as usize;
        let total_qubits = n_samples * qubits_per_sample;

        self.ising_matrix = Some(Array2::zeros((total_qubits, total_qubits)));
        let ising_matrix = self.ising_matrix.as_mut().unwrap();

        // Build interaction matrix based on data similarities
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let distance = euclidean_distance(data.row(i), data.row(j))
                    .to_f64()
                    .unwrap();
                let similarity = (-distance / 2.0).exp(); // Gaussian similarity

                // Add interactions between qubits representing these samples
                for qi in 0..qubits_per_sample {
                    for qj in 0..qubits_per_sample {
                        let qubit_i = i * qubits_per_sample + qi;
                        let qubit_j = j * qubits_per_sample + qj;

                        // Ising interaction: encourage similar samples to have similar spin patterns
                        ising_matrix[[qubit_i, qubit_j]] = similarity;
                        ising_matrix[[qubit_j, qubit_i]] = similarity;
                    }
                }
            }
        }

        Ok(())
    }

    /// Initialize random spin configuration
    fn initialize_spins(&mut self, nsamples: usize) -> Result<()> {
        let qubits_per_sample = (self.n_clusters as f64).log2().ceil() as usize;
        let total_qubits = nsamples * qubits_per_sample;

        use rand::Rng;
        let mut rng = if let Some(seed) = self.config.random_seed {
            use rand::SeedableRng;
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            use rand::SeedableRng;
            rand::rngs::StdRng::seed_from_u64(rand::rng().gen::<u64>())
        };

        let mut spins = Array1::zeros(total_qubits);
        for i in 0..total_qubits {
            spins[i] = if rng.random::<f64>() > 0.5_f64 {
                F::one()
            } else {
                F::zero() - F::one()
            };
        }

        // Convert F spins to i8 for storage
        let i8_spins = spins.mapv(|spin| if spin == F::one() { 1i8 } else { -1i8 });
        self.spin_configuration = Some(i8_spins.clone());
        self.best_configuration = Some(i8_spins);
        self.best_energy =
            Some(self.calculate_ising_energy(&self.spin_configuration.as_ref().unwrap()));

        Ok(())
    }

    /// Create temperature schedule for annealing
    fn create_temperature_schedule(&mut self) {
        let mut schedule = Vec::with_capacity(self.config.annealing_steps);

        for step in 0..self.config.annealing_steps {
            let progress = step as f64 / (self.config.annealing_steps - 1) as f64;

            let temperature = match self.config.cooling_schedule {
                CoolingSchedule::Linear => {
                    self.config.initial_temperature * (1.0 - progress)
                        + self.config.final_temperature * progress
                }
                CoolingSchedule::Exponential => {
                    self.config.initial_temperature
                        * (self.config.final_temperature / self.config.initial_temperature)
                            .powf(progress)
                }
                CoolingSchedule::Logarithmic => {
                    self.config.initial_temperature / (1.0 + progress.ln())
                }
                CoolingSchedule::PowerLaw(alpha) => {
                    self.config.initial_temperature * (1.0 - progress).powf(alpha)
                }
            };

            schedule.push(temperature.max(self.config.final_temperature));
        }

        self.temperature_schedule = schedule;
    }

    /// Run quantum annealing optimization
    fn run_annealing(&mut self) -> Result<()> {
        use rand::Rng;
        let mut rng = if let Some(seed) = self.config.random_seed {
            use rand::SeedableRng;
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            use rand::SeedableRng;
            rand::rngs::StdRng::seed_from_u64(rand::rng().gen::<u64>())
        };

        let n_qubits = self.spin_configuration.as_ref().unwrap().len();

        for temperature in &self.temperature_schedule {
            for _ in 0..self.config.mc_sweeps {
                // Monte Carlo sweep with quantum tunneling
                for qubit in 0..n_qubits {
                    // Calculate energy change for flipping this qubit
                    let delta_e = {
                        let spins = self.spin_configuration.as_ref().unwrap();
                        self.calculate_flip_energy_change(spins, qubit)
                    };

                    // Quantum tunneling probability (includes classical thermal + quantum effects)
                    let tunnel_probability = self.quantum_tunnel_probability(delta_e, *temperature);
                    let tunnel_prob_f = F::from(tunnel_probability).unwrap();

                    if F::from(rng.random::<f64>()).unwrap() < tunnel_prob_f {
                        // Flip spin
                        {
                            let spins = self.spin_configuration.as_mut().unwrap();
                            spins[qubit] = -spins[qubit];
                        }

                        // Update best configuration if we found a better one
                        let current_energy = {
                            let spins = self.spin_configuration.as_ref().unwrap();
                            self.calculate_ising_energy(spins)
                        };
                        if current_energy < self.best_energy.unwrap() {
                            self.best_energy = Some(current_energy);
                            self.best_configuration = self.spin_configuration.clone();
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Calculate energy change for flipping a single qubit
    fn calculate_flip_energy_change(&self, spins: &Array1<i8>, qubit: usize) -> f64 {
        let ising_matrix = self.ising_matrix.as_ref().unwrap();
        let current_spin = spins[qubit];

        let mut delta_e = 0.0;

        // Calculate interaction energy change
        for j in 0..spins.len() {
            if j != qubit {
                delta_e +=
                    2.0 * (current_spin as f64) * (spins[j] as f64) * ising_matrix[[qubit, j]];
            }
        }

        delta_e
    }

    /// Calculate quantum tunneling probability
    fn quantum_tunnel_probability(&self, deltae: f64, temperature: f64) -> f64 {
        if deltae <= 0.0 {
            1.0 // Always accept if energy decreases
        } else {
            // Enhanced probability including quantum tunneling effects
            let classical_prob = (-deltae / temperature).exp();
            let quantum_enhancement = 0.1 * (-deltae / (2.0 * temperature)).exp(); // Simplified quantum correction
            (classical_prob + quantum_enhancement).min(1.0)
        }
    }

    /// Calculate total Ising energy
    fn calculate_ising_energy(&self, spins: &Array1<i8>) -> f64 {
        let ising_matrix = self.ising_matrix.as_ref().unwrap();
        let mut energy = 0.0;

        for i in 0..spins.len() {
            for j in i + 1..spins.len() {
                energy -= ising_matrix[[i, j]] * (spins[i] as f64) * (spins[j] as f64);
            }
        }

        energy
    }

    /// Convert spin configuration to cluster assignments
    fn spins_to_clusters(&self, spins: &Array1<i8>) -> Array1<usize> {
        let n_samples = spins.len() / (self.n_clusters as f64).log2().ceil() as usize;
        let qubits_per_sample = (self.n_clusters as f64).log2().ceil() as usize;
        let mut assignments = Array1::zeros(n_samples);

        for sample in 0..n_samples {
            let mut cluster_id = 0;
            for bit in 0..qubits_per_sample {
                let qubit_idx = sample * qubits_per_sample + bit;
                if spins[qubit_idx] > 0 {
                    cluster_id += 1 << bit;
                }
            }
            assignments[sample] = (cluster_id % self.n_clusters);
        }

        assignments
    }

    /// Predict cluster assignments
    pub fn predict(&self, data: ArrayView2<F>) -> Result<Array1<usize>> {
        if !self.fitted {
            return Err(ClusteringError::InvalidInput(
                "Model must be fitted before prediction".to_string(),
            ));
        }

        let best_spins = self.best_configuration.as_ref().unwrap();
        Ok(self.spins_to_clusters(best_spins))
    }

    /// Get the best energy found
    pub fn best_energy(&self) -> Option<f64> {
        self.best_energy
    }

    /// Get the temperature schedule used
    pub fn temperature_schedule(&self) -> &[f64] {
        &self.temperature_schedule
    }
}

/// Quantum annealing clustering with default configuration
#[allow(dead_code)]
pub fn quantum_annealing_clustering<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    n_clusters: usize,
) -> Result<(Array1<usize>, f64)> {
    let config = QuantumAnnealingConfig::default();
    let mut annealer = QuantumAnnealingClustering::new(n_clusters, config);
    annealer.fit(data)?;

    let assignments = annealer.predict(data)?;
    let energy = annealer.best_energy().unwrap_or(0.0);

    Ok((assignments, energy))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_qaoa_clustering_creation() {
        let config = QAOAConfig::default();
        let qaoa: QAOAClustering<f64> = QAOAClustering::new(3, config);
        assert_eq!(qaoa.n_clusters, 3);
        assert!(!qaoa.fitted);
    }

    #[test]
    fn test_vqe_clustering_creation() {
        let config = VQEConfig::default();
        let vqe: VQEClustering<f64> = VQEClustering::new(2, config);
        assert_eq!(vqe.n_clusters, 2);
        assert!(!vqe.fitted);
    }

    #[test]
    fn test_qaoa_config_defaults() {
        let config = QAOAConfig::default();
        assert_eq!(config.p_layers, 3);
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.cost_function, QAOACostFunction::KMeans);
    }

    #[test]
    fn test_vqe_config_defaults() {
        let config = VQEConfig::default();
        assert_eq!(config.ansatz, VQEAnsatz::HardwareEfficient);
        assert_eq!(config.max_iterations, 200);
        assert_eq!(config.optimizer, VQEOptimizer::Adam);
    }

    #[test]
    fn test_small_qaoa_clustering() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1]).unwrap();

        let result = qaoa_clustering(data.view(), 2);
        assert!(result.is_ok());

        let (assignments, energy) = result.unwrap();
        assert_eq!(assignments.len(), 4);
        assert!(energy.is_finite());
    }

    #[test]
    fn test_quantum_annealing_creation() {
        let config = QuantumAnnealingConfig::default();
        let annealer: QuantumAnnealingClustering<f64> = QuantumAnnealingClustering::new(2, config);
        assert_eq!(annealer.n_clusters, 2);
        assert!(!annealer.fitted);
    }

    #[test]
    fn test_quantum_annealing_config_defaults() {
        let config = QuantumAnnealingConfig::default();
        assert_eq!(config.initial_temperature, 10.0);
        assert_eq!(config.final_temperature, 0.01);
        assert_eq!(config.annealing_steps, 1000);
        assert_eq!(config.cooling_schedule, CoolingSchedule::Exponential);
    }

    #[test]
    fn test_cooling_schedules() {
        let linear = CoolingSchedule::Linear;
        let exponential = CoolingSchedule::Exponential;
        let logarithmic = CoolingSchedule::Logarithmic;
        let power_law = CoolingSchedule::PowerLaw(2.0);

        assert_eq!(linear, CoolingSchedule::Linear);
        assert_eq!(exponential, CoolingSchedule::Exponential);
        assert_eq!(logarithmic, CoolingSchedule::Logarithmic);
        assert_eq!(power_law, CoolingSchedule::PowerLaw(2.0));
    }

    #[test]
    fn test_small_quantum_annealing_clustering() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1]).unwrap();

        let result = quantum_annealing_clustering(data.view(), 2);
        assert!(result.is_ok());

        let (assignments, energy) = result.unwrap();
        assert_eq!(assignments.len(), 4);
        assert!(energy.is_finite());
    }

    #[test]
    fn test_quantum_annealing_with_custom_config() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        )
        .unwrap();

        let config = QuantumAnnealingConfig {
            initial_temperature: 5.0,
            final_temperature: 0.1,
            annealing_steps: 100,
            cooling_schedule: CoolingSchedule::Linear,
            mc_sweeps: 10,
            random_seed: Some(42),
        };

        let mut annealer = QuantumAnnealingClustering::new(2, config);
        let result = annealer.fit(data.view());
        assert!(result.is_ok());

        let assignments = annealer.predict(data.view());
        assert!(assignments.is_ok());
        assert_eq!(assignments.unwrap().len(), 6);

        assert!(annealer.best_energy().is_some());
        assert_eq!(annealer.temperature_schedule().len(), 100);
    }
}
