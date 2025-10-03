//! Quantum Machine Learning Metrics Collection
//!
//! This module provides specialized metrics for evaluating quantum machine learning
//! algorithms, quantum circuits, and hybrid quantum-classical models.
//!
//! # Features
//!
//! - **Quantum Fidelity Metrics**: State fidelity, process fidelity, gate fidelity
//! - **Entanglement Analysis**: Concurrence, negativity, entanglement entropy
//! - **Quantum Circuit Metrics**: Circuit depth, gate count, connectivity metrics
//! - **Hybrid Model Evaluation**: Quantum advantage metrics, classical-quantum comparison
//! - **Quantum Error Metrics**: Error rates, noise characterization, fault tolerance
//! - **Quantum Speedup Analysis**: Quantum supremacy indicators, computational complexity
//!
//! # Examples
//!
//! ```
//! use scirs2_metrics::domains::quantum_ml::QuantumMLSuite;
//! use scirs2_core::ndarray::array;
//! use scirs2_core::numeric::Complex64;
//!
//! let mut qml_suite = QuantumMLSuite::new();
//!
//! // Example quantum state vectors (normalized)
//! let state1 = array![
//!     Complex64::new(0.6, 0.0),  // |0⟩ amplitude
//!     Complex64::new(0.8, 0.0),  // |1⟩ amplitude
//! ];
//!
//! let state2 = array![
//!     Complex64::new(0.8, 0.0),
//!     Complex64::new(0.6, 0.0),
//! ];
//!
//! // Compute quantum fidelity
//! let fidelity = qml_suite.quantum_state_fidelity(&state1, &state2).unwrap();
//! println!("Quantum State Fidelity: {:.4}", fidelity);
//! ```

use crate::domains::{DomainEvaluationResult, DomainMetrics};
use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{Float, One, Zero};
use std::collections::HashMap;

/// Quantum Machine Learning metrics suite
#[derive(Debug, Clone)]
pub struct QuantumMLSuite {
    /// Configuration for quantum metrics computation
    config: QuantumMetricsConfig,
    /// Cache for expensive quantum computations
    #[allow(dead_code)]
    computation_cache: HashMap<String, f64>,
}

/// Configuration for quantum metrics computation
#[derive(Debug, Clone)]
pub struct QuantumMetricsConfig {
    /// Tolerance for numerical comparisons
    pub numerical_tolerance: f64,
    /// Enable caching of expensive computations
    pub enable_caching: bool,
    /// Maximum number of qubits to handle efficiently
    pub max_qubits: usize,
    /// Quantum noise model parameters
    pub noise_model: Option<QuantumNoiseModel>,
    /// Enable quantum advantage detection
    pub detect_quantum_advantage: bool,
}

/// Quantum noise model for realistic simulations
#[derive(Debug, Clone)]
pub struct QuantumNoiseModel {
    /// Single-qubit gate error rate
    pub single_qubit_error_rate: f64,
    /// Two-qubit gate error rate
    pub two_qubit_error_rate: f64,
    /// Measurement error rate
    pub measurement_error_rate: f64,
    /// Decoherence time T1 (microseconds)
    pub t1_decoherence: f64,
    /// Dephasing time T2 (microseconds)
    pub t2_dephasing: f64,
}

/// Quantum circuit description
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub depth: usize,
    /// Gate sequence
    pub gates: Vec<QuantumGate>,
    /// Connectivity graph
    pub connectivity: Option<Array2<bool>>,
}

/// Quantum gate representation
#[derive(Debug, Clone)]
pub enum QuantumGate {
    /// Single-qubit gates
    X(usize),
    Y(usize),
    Z(usize),
    H(usize),
    S(usize),
    T(usize),
    /// Parameterized single-qubit gates
    RX(usize, f64),
    RY(usize, f64),
    RZ(usize, f64),
    /// Two-qubit gates
    CNOT(usize, usize),
    CZ(usize, usize),
    SWAP(usize, usize),
    /// Custom unitary
    Custom(Vec<usize>, Array2<Complex64>),
}

/// Quantum entanglement metrics
#[derive(Debug, Clone)]
pub struct EntanglementMetrics {
    /// Concurrence measure
    pub concurrence: f64,
    /// Negativity measure
    pub negativity: f64,
    /// Von Neumann entropy
    pub von_neumann_entropy: f64,
    /// Entanglement of formation
    pub entanglement_of_formation: f64,
    /// Schmidt rank
    pub schmidt_rank: usize,
}

/// Quantum fidelity metrics
#[derive(Debug, Clone)]
pub struct FidelityMetrics {
    /// State fidelity
    pub state_fidelity: f64,
    /// Process fidelity
    pub process_fidelity: Option<f64>,
    /// Average gate fidelity
    pub average_gate_fidelity: Option<f64>,
    /// Trace distance
    pub trace_distance: f64,
}

/// Quantum circuit complexity metrics
#[derive(Debug, Clone)]
pub struct CircuitComplexityMetrics {
    /// Circuit depth
    pub depth: usize,
    /// Total gate count
    pub gate_count: usize,
    /// Two-qubit gate count
    pub two_qubit_gate_count: usize,
    /// Connectivity requirements
    pub connectivity_degree: f64,
    /// Circuit volume
    pub circuit_volume: f64,
}

/// Quantum advantage metrics
#[derive(Debug, Clone)]
pub struct QuantumAdvantageMetrics {
    /// Estimated quantum speedup
    pub quantum_speedup: f64,
    /// Classical simulation complexity
    pub classical_complexity: f64,
    /// Quantum resource requirements
    pub quantum_resources: f64,
    /// Quantum advantage score
    pub advantage_score: f64,
}

/// Quantum error analysis results
#[derive(Debug, Clone)]
pub struct QuantumErrorMetrics {
    /// Error rate estimates
    pub error_rates: HashMap<String, f64>,
    /// Noise characterization
    pub noise_metrics: HashMap<String, f64>,
    /// Fault tolerance requirements
    pub fault_tolerance_threshold: f64,
    /// Error correction overhead
    pub error_correction_overhead: f64,
}

impl Default for QuantumMetricsConfig {
    fn default() -> Self {
        Self {
            numerical_tolerance: 1e-10,
            enable_caching: true,
            max_qubits: 20,
            noise_model: None,
            detect_quantum_advantage: true,
        }
    }
}

impl QuantumMLSuite {
    /// Create a new quantum ML metrics suite
    pub fn new() -> Self {
        Self {
            config: QuantumMetricsConfig::default(),
            computation_cache: HashMap::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: QuantumMetricsConfig) -> Self {
        Self {
            config,
            computation_cache: HashMap::new(),
        }
    }

    /// Compute quantum state fidelity between two quantum states
    pub fn quantum_state_fidelity(
        &self,
        state1: &Array1<Complex64>,
        state2: &Array1<Complex64>,
    ) -> Result<f64> {
        if state1.len() != state2.len() {
            return Err(MetricsError::InvalidInput(
                "Quantum states must have the same dimension".to_string(),
            ));
        }

        // Verify states are normalized
        let norm1 = Self::state_norm(state1);
        let norm2 = Self::state_norm(state2);

        if (norm1 - 1.0).abs() > self.config.numerical_tolerance
            || (norm2 - 1.0).abs() > self.config.numerical_tolerance
        {
            return Err(MetricsError::InvalidInput(
                "Quantum states must be normalized".to_string(),
            ));
        }

        // Compute fidelity F = |⟨ψ₁|ψ₂⟩|²
        let inner_product = state1
            .iter()
            .zip(state2.iter())
            .map(|(a, b)| a.conj() * b)
            .fold(Complex64::zero(), |acc, x| acc + x);

        let fidelity = inner_product.norm_sqr();
        Ok(fidelity)
    }

    /// Compute process fidelity between two quantum channels
    pub fn process_fidelity(
        &self,
        channel1: &Array2<Complex64>,
        channel2: &Array2<Complex64>,
    ) -> Result<f64> {
        if channel1.shape() != channel2.shape() {
            return Err(MetricsError::InvalidInput(
                "Quantum channels must have the same dimensions".to_string(),
            ));
        }

        // Convert to Choi matrices and compute fidelity
        let choi1 = self.channel_to_choi(channel1)?;
        let choi2 = self.channel_to_choi(channel2)?;

        // Compute process fidelity using Choi matrices
        let trace_product = self.trace_product(&choi1, &choi2)?;
        let fidelity = trace_product.norm();

        Ok(fidelity)
    }

    /// Compute entanglement metrics for a quantum state
    pub fn entanglement_analysis(
        &self,
        state: &Array1<Complex64>,
        partition: (Vec<usize>, Vec<usize>),
    ) -> Result<EntanglementMetrics> {
        let num_qubits = (state.len() as f64).log2() as usize;
        if state.len() != 2_usize.pow(num_qubits as u32) {
            return Err(MetricsError::InvalidInput(
                "State dimension must be a power of 2".to_string(),
            ));
        }

        // Compute reduced density matrix for the first subsystem
        let rho_a = self.partial_trace(state, &partition)?;

        // Compute concurrence (for 2-qubit systems)
        let concurrence = if num_qubits == 2 {
            self.compute_concurrence(state)?
        } else {
            0.0 // General concurrence computation is more complex
        };

        // Compute negativity
        let negativity = self.compute_negativity(&rho_a)?;

        // Compute von Neumann entropy
        let von_neumann_entropy = self.von_neumann_entropy(&rho_a)?;

        // Compute entanglement of formation (approximation for larger systems)
        let entanglement_of_formation = if num_qubits == 2 {
            self.entanglement_of_formation(concurrence)?
        } else {
            von_neumann_entropy // Upper bound approximation
        };

        // Compute Schmidt rank
        let schmidt_rank = self.schmidt_rank(state, &partition)?;

        Ok(EntanglementMetrics {
            concurrence,
            negativity,
            von_neumann_entropy,
            entanglement_of_formation,
            schmidt_rank,
        })
    }

    /// Analyze quantum circuit complexity
    pub fn circuit_complexity_analysis(
        &self,
        circuit: &QuantumCircuit,
    ) -> Result<CircuitComplexityMetrics> {
        let depth = circuit.depth;
        let gate_count = circuit.gates.len();

        // Count two-qubit gates
        let two_qubit_gate_count = circuit
            .gates
            .iter()
            .filter(|gate| self.is_two_qubit_gate(gate))
            .count();

        // Compute connectivity degree
        let connectivity_degree = if let Some(connectivity) = &circuit.connectivity {
            self.compute_connectivity_degree(connectivity)?
        } else {
            // Estimate based on two-qubit gates
            two_qubit_gate_count as f64 / circuit.num_qubits as f64
        };

        // Compute circuit volume (depth × width)
        let circuit_volume = (depth * circuit.num_qubits) as f64;

        Ok(CircuitComplexityMetrics {
            depth,
            gate_count,
            two_qubit_gate_count,
            connectivity_degree,
            circuit_volume,
        })
    }

    /// Assess quantum advantage potential
    pub fn quantum_advantage_analysis(
        &self,
        quantum_results: &[f64],
        classical_results: &[f64],
        circuit: &QuantumCircuit,
    ) -> Result<QuantumAdvantageMetrics> {
        if quantum_results.len() != classical_results.len() {
            return Err(MetricsError::InvalidInput(
                "Quantum and classical _results must have the same length".to_string(),
            ));
        }

        // Estimate quantum speedup based on performance comparison
        let quantum_speedup = self.estimate_speedup(quantum_results, classical_results)?;

        // Estimate classical simulation complexity
        let classical_complexity = self.estimate_classical_complexity(circuit)?;

        // Estimate quantum resource requirements
        let quantum_resources = self.estimate_quantum_resources(circuit)?;

        // Compute overall quantum advantage score
        let advantage_score = quantum_speedup * classical_complexity / quantum_resources;

        Ok(QuantumAdvantageMetrics {
            quantum_speedup,
            classical_complexity,
            quantum_resources,
            advantage_score,
        })
    }

    /// Comprehensive quantum error analysis
    pub fn quantum_error_analysis(
        &self,
        ideal_results: &[Complex64],
        noisy_results: &[Complex64],
        circuit: &QuantumCircuit,
    ) -> Result<QuantumErrorMetrics> {
        let mut error_rates = HashMap::new();
        let mut noise_metrics = HashMap::new();

        // Compute gate error rates
        if let Some(noise_model) = &self.config.noise_model {
            error_rates.insert(
                "single_qubit_error".to_string(),
                noise_model.single_qubit_error_rate,
            );
            error_rates.insert(
                "two_qubit_error".to_string(),
                noise_model.two_qubit_error_rate,
            );
            error_rates.insert(
                "measurement_error".to_string(),
                noise_model.measurement_error_rate,
            );
        }

        // Compute fidelity loss
        let fidelity_loss = self.compute_fidelity_loss(ideal_results, noisy_results)?;
        noise_metrics.insert("fidelity_loss".to_string(), fidelity_loss);

        // Estimate fault tolerance threshold
        let fault_tolerance_threshold = self.estimate_fault_tolerance_threshold(circuit)?;

        // Estimate error correction overhead
        let error_correction_overhead = self.estimate_error_correction_overhead(circuit)?;

        Ok(QuantumErrorMetrics {
            error_rates,
            noise_metrics,
            fault_tolerance_threshold,
            error_correction_overhead,
        })
    }

    /// Comprehensive quantum ML evaluation
    pub fn evaluate_quantum_model(
        &self,
        quantum_predictions: &[f64],
        classical_predictions: &[f64],
        true_labels: &[f64],
        circuit: &QuantumCircuit,
    ) -> Result<DomainEvaluationResult> {
        let mut result = DomainEvaluationResult::new();

        // Compute basic ML metrics
        let quantum_mse = self.mean_squared_error(quantum_predictions, true_labels)?;
        let classical_mse = self.mean_squared_error(classical_predictions, true_labels)?;

        result.add_primary_metric("quantum_mse".to_string(), quantum_mse);
        result.add_primary_metric("classical_mse".to_string(), classical_mse);
        result.add_primary_metric(
            "mse_improvement".to_string(),
            (classical_mse - quantum_mse) / classical_mse,
        );

        // Compute quantum-specific metrics
        let complexity_metrics = self.circuit_complexity_analysis(circuit)?;
        result.add_secondary_metric("circuit_depth".to_string(), complexity_metrics.depth as f64);
        result.add_secondary_metric(
            "gate_count".to_string(),
            complexity_metrics.gate_count as f64,
        );
        result.add_secondary_metric(
            "two_qubit_gates".to_string(),
            complexity_metrics.two_qubit_gate_count as f64,
        );

        // Quantum advantage analysis
        let advantage_metrics =
            self.quantum_advantage_analysis(quantum_predictions, classical_predictions, circuit)?;
        result.add_primary_metric(
            "quantum_advantage_score".to_string(),
            advantage_metrics.advantage_score,
        );
        result.add_secondary_metric(
            "estimated_speedup".to_string(),
            advantage_metrics.quantum_speedup,
        );

        // Set summary
        let summary = if advantage_metrics.advantage_score > 1.0 {
            format!(
                "Quantum advantage detected with score {:.2}. MSE improvement: {:.1}%",
                advantage_metrics.advantage_score,
                ((classical_mse - quantum_mse) / classical_mse) * 100.0
            )
        } else {
            format!(
                "No clear quantum advantage. Classical MSE: {:.4}, Quantum MSE: {:.4}",
                classical_mse, quantum_mse
            )
        };
        result.set_summary(summary);

        Ok(result)
    }

    // Helper methods

    fn state_norm(state: &Array1<Complex64>) -> f64 {
        state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
    }

    fn channel_to_choi(&self, channel: &Array2<Complex64>) -> Result<Array2<Complex64>> {
        // Simplified Choi matrix computation
        // In a full implementation, this would properly construct the Choi representation
        Ok(channel.clone())
    }

    fn trace_product(&self, a: &Array2<Complex64>, b: &Array2<Complex64>) -> Result<Complex64> {
        // Compute Tr(A† B)
        let mut trace = Complex64::zero();
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                trace += a[[i, j]].conj() * b[[i, j]];
            }
        }
        Ok(trace)
    }

    fn partial_trace(
        &self,
        state: &Array1<Complex64>,
        partition: &(Vec<usize>, Vec<usize>),
    ) -> Result<Array2<Complex64>> {
        let _num_qubits = (state.len() as f64).log2() as usize;
        let dim_a = 2_usize.pow(partition.0.len() as u32);
        let dim_b = 2_usize.pow(partition.1.len() as u32);

        if dim_a * dim_b != state.len() {
            return Err(MetricsError::InvalidInput(
                "Partition dimensions don't match state dimension".to_string(),
            ));
        }

        // Construct density matrix
        let mut density_matrix = Array2::zeros((state.len(), state.len()));
        for i in 0..state.len() {
            for j in 0..state.len() {
                density_matrix[[i, j]] = state[i].conj() * state[j];
            }
        }

        // Compute partial trace (simplified implementation)
        let mut rho_a = Array2::zeros((dim_a, dim_a));
        for i in 0..dim_a {
            for j in 0..dim_a {
                for k in 0..dim_b {
                    let idx1 = i * dim_b + k;
                    let idx2 = j * dim_b + k;
                    rho_a[[i, j]] += density_matrix[[idx1, idx2]];
                }
            }
        }

        Ok(rho_a)
    }

    fn compute_concurrence(&self, state: &Array1<Complex64>) -> Result<f64> {
        // Concurrence for a 2-qubit pure state |ψ⟩ = a|00⟩ + b|01⟩ + c|10⟩ + d|11⟩
        if state.len() != 4 {
            return Err(MetricsError::InvalidInput(
                "Concurrence calculation requires a 2-qubit state".to_string(),
            ));
        }

        let a = state[0];
        let b = state[1];
        let c = state[2];
        let d = state[3];

        // C = 2|ad - bc|
        let concurrence = 2.0 * (a * d - b * c).norm();
        Ok(concurrence)
    }

    fn compute_negativity(&self, rho: &Array2<Complex64>) -> Result<f64> {
        // Compute partial transpose and its eigenvalues
        let rho_pt = self.partial_transpose(rho)?;
        let eigenvalues = self.compute_eigenvalues(&rho_pt)?;

        // Negativity is the sum of absolute values of negative eigenvalues
        let negativity = eigenvalues
            .iter()
            .filter(|&lambda| *lambda < 0.0)
            .map(|lambda| lambda.abs())
            .sum::<f64>();

        Ok(negativity)
    }

    fn partial_transpose(&self, matrix: &Array2<Complex64>) -> Result<Array2<Complex64>> {
        // Simplified partial transpose implementation
        // In practice, this would depend on the specific bipartition
        Ok(matrix.t().to_owned())
    }

    fn compute_eigenvalues(&self, matrix: &Array2<Complex64>) -> Result<Vec<f64>> {
        // Simplified eigenvalue computation
        // In practice, would use proper linear algebra library
        let mut eigenvalues = Vec::new();
        for i in 0..matrix.nrows().min(matrix.ncols()) {
            eigenvalues.push(matrix[[i, i]].re);
        }
        Ok(eigenvalues)
    }

    fn von_neumann_entropy(&self, rho: &Array2<Complex64>) -> Result<f64> {
        let eigenvalues = self.compute_eigenvalues(rho)?;

        let entropy = eigenvalues
            .iter()
            .filter(|&lambda| *lambda > self.config.numerical_tolerance)
            .map(|&lambda| -lambda * lambda.ln())
            .sum::<f64>();

        Ok(entropy)
    }

    fn entanglement_of_formation(&self, concurrence: f64) -> Result<f64> {
        if !(0.0..=1.0).contains(&concurrence) {
            return Err(MetricsError::InvalidInput(
                "Concurrence must be between 0 and 1".to_string(),
            ));
        }

        let h = |x: f64| {
            if x == 0.0 {
                0.0
            } else {
                -x * x.log2() - (1.0 - x) * (1.0 - x).log2()
            }
        };

        let x = (1.0 + (1.0 - concurrence.powi(2)).sqrt()) / 2.0;
        Ok(h(x))
    }

    fn schmidt_rank(
        &self,
        state: &Array1<Complex64>,
        partition: &(Vec<usize>, Vec<usize>),
    ) -> Result<usize> {
        // Compute reduced density matrix and count non-zero eigenvalues
        let rho_a = self.partial_trace(state, partition)?;
        let eigenvalues = self.compute_eigenvalues(&rho_a)?;

        let rank = eigenvalues
            .iter()
            .filter(|&lambda| lambda.abs() > self.config.numerical_tolerance)
            .count();

        Ok(rank)
    }

    fn is_two_qubit_gate(&self, gate: &QuantumGate) -> bool {
        matches!(
            gate,
            QuantumGate::CNOT(_, _) | QuantumGate::CZ(_, _) | QuantumGate::SWAP(_, _)
        )
    }

    fn compute_connectivity_degree(&self, connectivity: &Array2<bool>) -> Result<f64> {
        let total_edges = connectivity.iter().filter(|&&connected| connected).count();
        let max_edges = connectivity.nrows() * (connectivity.nrows() - 1) / 2;
        Ok(total_edges as f64 / max_edges as f64)
    }

    fn estimate_speedup(&self, quantum_results: &[f64], classicalresults: &[f64]) -> Result<f64> {
        // Simple performance ratio estimation
        let quantum_avg = quantum_results.iter().sum::<f64>() / quantum_results.len() as f64;
        let classical_avg = classicalresults.iter().sum::<f64>() / classicalresults.len() as f64;

        if quantum_avg == 0.0 {
            Ok(1.0)
        } else {
            Ok(classical_avg / quantum_avg)
        }
    }

    fn estimate_classical_complexity(&self, circuit: &QuantumCircuit) -> Result<f64> {
        // Exponential scaling with number of qubits
        Ok(2.0_f64.powf(circuit.num_qubits as f64))
    }

    fn estimate_quantum_resources(&self, circuit: &QuantumCircuit) -> Result<f64> {
        // Resource estimate based on circuit depth and gate count
        Ok((circuit.depth * circuit.gates.len()) as f64)
    }

    fn compute_fidelity_loss(&self, ideal: &[Complex64], noisy: &[Complex64]) -> Result<f64> {
        if ideal.len() != noisy.len() {
            return Err(MetricsError::InvalidInput(
                "Ideal and noisy results must have the same length".to_string(),
            ));
        }

        let total_error = ideal
            .iter()
            .zip(noisy.iter())
            .map(|(i, n)| (i - n).norm_sqr())
            .sum::<f64>();

        Ok(total_error / ideal.len() as f64)
    }

    fn estimate_fault_tolerance_threshold(&self, circuit: &QuantumCircuit) -> Result<f64> {
        // Rough estimate based on circuit complexity
        // Float threshold depends on specific error correction code
        let base_threshold = 1e-3; // 0.1% error rate
        let complexity_factor = circuit.depth as f64 * circuit.num_qubits as f64;
        Ok(base_threshold / complexity_factor.sqrt())
    }

    fn estimate_error_correction_overhead(&self, circuit: &QuantumCircuit) -> Result<f64> {
        // Estimate overhead factor for quantum error correction
        // Surface code typically requires ~1000 physical qubits per logical qubit
        let logical_qubits = circuit.num_qubits as f64;
        let physical_qubits = logical_qubits * 1000.0; // Rough estimate
        Ok(physical_qubits / logical_qubits)
    }

    fn mean_squared_error(&self, predictions: &[f64], targets: &[f64]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(MetricsError::InvalidInput(
                "Predictions and targets must have the same length".to_string(),
            ));
        }

        let mse = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;

        Ok(mse)
    }
}

impl DomainMetrics for QuantumMLSuite {
    type Result = DomainEvaluationResult;

    fn domain_name(&self) -> &'static str {
        "Quantum Machine Learning"
    }

    fn available_metrics(&self) -> Vec<&'static str> {
        vec![
            "quantum_state_fidelity",
            "process_fidelity",
            "concurrence",
            "negativity",
            "von_neumann_entropy",
            "entanglement_of_formation",
            "schmidt_rank",
            "circuit_depth",
            "gate_count",
            "connectivity_degree",
            "quantum_advantage_score",
            "error_rates",
            "fault_tolerance_threshold",
        ]
    }

    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str> {
        let mut descriptions = HashMap::new();
        descriptions.insert(
            "quantum_state_fidelity",
            "Overlap between two quantum states",
        );
        descriptions.insert("process_fidelity", "Similarity between quantum processes");
        descriptions.insert("concurrence", "Measure of two-qubit entanglement");
        descriptions.insert("negativity", "Entanglement measure via partial transpose");
        descriptions.insert("von_neumann_entropy", "Quantum information entropy");
        descriptions.insert(
            "entanglement_of_formation",
            "Cost to create entangled state",
        );
        descriptions.insert("schmidt_rank", "Number of entangled degrees of freedom");
        descriptions.insert("circuit_depth", "Number of sequential gate layers");
        descriptions.insert("gate_count", "Total number of quantum gates");
        descriptions.insert("connectivity_degree", "Required qubit connectivity");
        descriptions.insert(
            "quantum_advantage_score",
            "Quantum vs classical performance",
        );
        descriptions.insert("error_rates", "Quantum gate and measurement errors");
        descriptions.insert("fault_tolerance_threshold", "Error rate tolerance limit");
        descriptions
    }
}

impl Default for QuantumMLSuite {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use scirs2_core::numeric::Complex64;
    use std::f64::consts::PI;

    #[test]
    fn test_quantum_state_fidelity() {
        let suite = QuantumMLSuite::new();

        // Identical states should have fidelity 1
        let state1 = array![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        let state2 = state1.clone();

        let fidelity = suite.quantum_state_fidelity(&state1, &state2).unwrap();
        assert!((fidelity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_state_fidelity_orthogonal() {
        let suite = QuantumMLSuite::new();

        // Orthogonal states should have fidelity 0
        let state1 = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let state2 = array![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];

        let fidelity = suite.quantum_state_fidelity(&state1, &state2).unwrap();
        assert!(fidelity.abs() < 1e-10);
    }

    #[test]
    fn test_concurrence_computation() {
        let suite = QuantumMLSuite::new();

        // Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 should have concurrence 1
        let bell_state = array![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0), // |00⟩
            Complex64::new(0.0, 0.0),                  // |01⟩
            Complex64::new(0.0, 0.0),                  // |10⟩
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0), // |11⟩
        ];

        let concurrence = suite.compute_concurrence(&bell_state).unwrap();
        assert!((concurrence - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_circuit_complexity_analysis() {
        let suite = QuantumMLSuite::new();

        let circuit = QuantumCircuit {
            num_qubits: 3,
            depth: 4,
            gates: vec![
                QuantumGate::H(0),
                QuantumGate::CNOT(0, 1),
                QuantumGate::RZ(2, PI / 4.0),
                QuantumGate::CNOT(1, 2),
            ],
            connectivity: None,
        };

        let metrics = suite.circuit_complexity_analysis(&circuit).unwrap();
        assert_eq!(metrics.depth, 4);
        assert_eq!(metrics.gate_count, 4);
        assert_eq!(metrics.two_qubit_gate_count, 2);
        assert_eq!(metrics.circuit_volume, 12.0);
    }

    #[test]
    fn test_domain_metrics_trait() {
        let suite = QuantumMLSuite::new();

        assert_eq!(suite.domain_name(), "Quantum Machine Learning");
        assert!(suite
            .available_metrics()
            .contains(&"quantum_state_fidelity"));
        assert!(suite.available_metrics().contains(&"concurrence"));

        let descriptions = suite.metric_descriptions();
        assert!(descriptions.contains_key("quantum_state_fidelity"));
        assert!(descriptions.contains_key("concurrence"));
    }

    #[test]
    fn test_mean_squared_error() {
        let suite = QuantumMLSuite::new();

        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.1, 1.9, 3.1];

        let mse = suite.mean_squared_error(&predictions, &targets).unwrap();
        let expected_mse = (0.01 + 0.01 + 0.01) / 3.0;
        assert!((mse - expected_mse).abs() < 1e-10);
    }
}
