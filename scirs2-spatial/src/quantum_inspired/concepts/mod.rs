//! Core Quantum Computing Concepts
//!
//! This module provides fundamental quantum computing concepts and structures
//! used throughout the quantum-inspired spatial algorithms. It includes quantum
//! state representations, quantum gates, and basic quantum operations.

use crate::error::{SpatialError, SpatialResult};
use ndarray::Array1;
use num_complex::Complex64;
use rand::Rng;
use std::f64::consts::SQRT_2;

/// Complex number type for quantum states
pub type QuantumAmplitude = Complex64;

/// Quantum state vector representation
///
/// A quantum state is represented as a vector of complex amplitudes in the
/// computational basis. The state |ψ⟩ = Σᵢ αᵢ|i⟩ where αᵢ are the complex
/// amplitudes and |i⟩ are the computational basis states.
///
/// # Properties
/// - The state is normalized: Σᵢ |αᵢ|² = 1
/// - The number of amplitudes must be a power of 2 (2ⁿ for n qubits)
/// - Supports common quantum gates and operations
///
/// # Example
/// ```rust
/// use scirs2_spatial::quantum_inspired::concepts::QuantumState;
/// use ndarray::Array1;
/// use num_complex::Complex64;
///
/// // Create a 2-qubit zero state |00⟩
/// let zero_state = QuantumState::zero_state(2);
/// assert_eq!(zero_state.num_qubits(), 2);
/// assert_eq!(zero_state.probability(0), 1.0);
///
/// // Create uniform superposition |+⟩⊗²
/// let superposition = QuantumState::uniform_superposition(2);
/// assert_eq!(superposition.probability(0), 0.25);
/// ```
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Amplitudes for each basis state
    pub amplitudes: Array1<QuantumAmplitude>,
    /// Number of qubits
    pub numqubits: usize,
}

impl QuantumState {
    /// Create a new quantum state with given amplitudes
    ///
    /// # Arguments
    /// * `amplitudes` - Complex amplitudes for each computational basis state
    ///
    /// # Returns
    /// A new `QuantumState` if the amplitudes vector has a power-of-2 length
    ///
    /// # Errors
    /// Returns `SpatialError::InvalidInput` if the number of amplitudes is not a power of 2
    pub fn new(amplitudes: Array1<QuantumAmplitude>) -> SpatialResult<Self> {
        let num_states = amplitudes.len();
        if !num_states.is_power_of_two() {
            return Err(SpatialError::InvalidInput(
                "Number of amplitudes must be a power of 2".to_string(),
            ));
        }

        let numqubits = (num_states as f64).log2() as usize;

        Ok(Self {
            amplitudes,
            numqubits,
        })
    }

    /// Create a quantum state in computational basis |0⟩⊗ⁿ
    ///
    /// Creates an n-qubit state where all qubits are in the |0⟩ state.
    /// This corresponds to the basis state |00...0⟩.
    ///
    /// # Arguments
    /// * `numqubits` - Number of qubits in the state
    ///
    /// # Returns
    /// A new `QuantumState` in the |0⟩⊗ⁿ state
    pub fn zero_state(numqubits: usize) -> Self {
        let num_states = 1 << numqubits;
        let mut amplitudes = Array1::zeros(num_states);
        amplitudes[0] = Complex64::new(1.0, 0.0);

        Self {
            amplitudes,
            numqubits,
        }
    }

    /// Create a uniform superposition state |+⟩⊗ⁿ
    ///
    /// Creates an n-qubit state where each qubit is in the |+⟩ = (|0⟩ + |1⟩)/√2 state.
    /// This results in a uniform superposition over all 2ⁿ computational basis states.
    ///
    /// # Arguments
    /// * `numqubits` - Number of qubits in the state
    ///
    /// # Returns
    /// A new `QuantumState` in uniform superposition
    pub fn uniform_superposition(numqubits: usize) -> Self {
        let num_states = 1 << numqubits;
        let amplitude = Complex64::new(1.0 / (num_states as f64).sqrt(), 0.0);
        let amplitudes = Array1::from_elem(num_states, amplitude);

        Self {
            amplitudes,
            numqubits,
        }
    }

    /// Measure the quantum state and collapse to classical state
    ///
    /// Performs a measurement in the computational basis, collapsing the quantum
    /// state to a classical state according to the Born rule. The probability of
    /// measuring state |i⟩ is |αᵢ|².
    ///
    /// # Returns
    /// The index of the measured computational basis state
    pub fn measure(&self) -> usize {
        let mut rng = rand::rng();

        // Calculate probabilities from amplitudes
        let probabilities: Vec<f64> = self.amplitudes.iter().map(|amp| amp.norm_sqr()).collect();

        // Cumulative probability distribution
        let mut cumulative = 0.0;
        let random_value = rng.gen_range(0.0..1.0);

        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                return i;
            }
        }

        // Fallback to last state
        probabilities.len() - 1
    }

    /// Get the probability of measuring a specific state
    ///
    /// Calculates the probability of measuring the quantum state in a specific
    /// computational basis state according to the Born rule.
    ///
    /// # Arguments
    /// * `state` - Index of the computational basis state
    ///
    /// # Returns
    /// Probability of measuring the given state (between 0.0 and 1.0)
    pub fn probability(&self, state: usize) -> f64 {
        if state >= self.amplitudes.len() {
            0.0
        } else {
            self.amplitudes[state].norm_sqr()
        }
    }

    /// Apply Hadamard gate to specific qubit
    ///
    /// The Hadamard gate creates superposition by mapping:
    /// - |0⟩ → (|0⟩ + |1⟩)/√2
    /// - |1⟩ → (|0⟩ - |1⟩)/√2
    ///
    /// # Arguments
    /// * `qubit` - Index of the qubit to apply the gate to (0-indexed)
    ///
    /// # Errors
    /// Returns `SpatialError::InvalidInput` if the qubit index is out of range
    pub fn hadamard(&mut self, qubit: usize) -> SpatialResult<()> {
        if qubit >= self.numqubits {
            return Err(SpatialError::InvalidInput(format!(
                "Qubit index {qubit} out of range"
            )));
        }

        let mut new_amplitudes = self.amplitudes.clone();
        let qubit_mask = 1 << qubit;

        for i in 0..self.amplitudes.len() {
            let j = i ^ qubit_mask; // Flip the target qubit
            if i < j {
                let amp_i = self.amplitudes[i];
                let amp_j = self.amplitudes[j];

                new_amplitudes[i] = (amp_i + amp_j) / SQRT_2;
                new_amplitudes[j] = (amp_i - amp_j) / SQRT_2;
            }
        }

        self.amplitudes = new_amplitudes;
        Ok(())
    }

    /// Apply phase rotation gate
    ///
    /// The phase rotation gate applies a phase e^(iθ) to the |1⟩ component
    /// of the specified qubit, leaving the |0⟩ component unchanged.
    ///
    /// # Arguments
    /// * `qubit` - Index of the qubit to apply the gate to
    /// * `angle` - Rotation angle in radians
    ///
    /// # Errors
    /// Returns `SpatialError::InvalidInput` if the qubit index is out of range
    pub fn phase_rotation(&mut self, qubit: usize, angle: f64) -> SpatialResult<()> {
        if qubit >= self.numqubits {
            return Err(SpatialError::InvalidInput(format!(
                "Qubit index {qubit} out of range"
            )));
        }

        let phase = Complex64::new(0.0, angle).exp();
        let qubit_mask = 1 << qubit;

        for i in 0..self.amplitudes.len() {
            if (i & qubit_mask) != 0 {
                self.amplitudes[i] *= phase;
            }
        }

        Ok(())
    }

    /// Apply controlled rotation between two qubits
    ///
    /// Applies a rotation to the target qubit conditioned on the control qubit
    /// being in the |1⟩ state. This creates entanglement between the qubits.
    ///
    /// # Arguments
    /// * `control` - Index of the control qubit
    /// * `target` - Index of the target qubit
    /// * `angle` - Rotation angle in radians
    ///
    /// # Errors
    /// Returns `SpatialError::InvalidInput` if either qubit index is out of range
    pub fn controlled_rotation(
        &mut self,
        control: usize,
        target: usize,
        angle: f64,
    ) -> SpatialResult<()> {
        if control >= self.numqubits || target >= self.numqubits {
            return Err(SpatialError::InvalidInput(
                "Qubit indices out of range".to_string(),
            ));
        }

        let control_mask = 1 << control;
        let target_mask = 1 << target;
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        let mut new_amplitudes = self.amplitudes.clone();

        for i in 0..self.amplitudes.len() {
            if (i & control_mask) != 0 {
                // Control qubit is |1⟩
                let j = i ^ target_mask; // Flip target qubit
                if i < j {
                    let amp_i = self.amplitudes[i];
                    let amp_j = self.amplitudes[j];

                    new_amplitudes[i] = Complex64::new(cos_half, 0.0) * amp_i
                        - Complex64::new(0.0, sin_half) * amp_j;
                    new_amplitudes[j] = Complex64::new(0.0, sin_half) * amp_i
                        + Complex64::new(cos_half, 0.0) * amp_j;
                }
            }
        }

        self.amplitudes = new_amplitudes;
        Ok(())
    }

    /// Apply Pauli-X gate (bit flip) to specific qubit
    ///
    /// The Pauli-X gate performs a bit flip: |0⟩ ↔ |1⟩
    ///
    /// # Arguments
    /// * `qubit` - Index of the qubit to apply the gate to
    ///
    /// # Errors
    /// Returns `SpatialError::InvalidInput` if the qubit index is out of range
    pub fn pauli_x(&mut self, qubit: usize) -> SpatialResult<()> {
        if qubit >= self.numqubits {
            return Err(SpatialError::InvalidInput(format!(
                "Qubit index {qubit} out of range"
            )));
        }

        let qubit_mask = 1 << qubit;
        let mut new_amplitudes = self.amplitudes.clone();

        for i in 0..self.amplitudes.len() {
            let j = i ^ qubit_mask; // Flip the target qubit
            new_amplitudes[i] = self.amplitudes[j];
        }

        self.amplitudes = new_amplitudes;
        Ok(())
    }

    /// Apply Pauli-Y gate to specific qubit
    ///
    /// The Pauli-Y gate performs: |0⟩ → i|1⟩, |1⟩ → -i|0⟩
    ///
    /// # Arguments
    /// * `qubit` - Index of the qubit to apply the gate to
    ///
    /// # Errors
    /// Returns `SpatialError::InvalidInput` if the qubit index is out of range
    pub fn pauli_y(&mut self, qubit: usize) -> SpatialResult<()> {
        if qubit >= self.numqubits {
            return Err(SpatialError::InvalidInput(format!(
                "Qubit index {qubit} out of range"
            )));
        }

        let qubit_mask = 1 << qubit;
        let mut new_amplitudes = self.amplitudes.clone();
        let i_complex = Complex64::new(0.0, 1.0);

        for i in 0..self.amplitudes.len() {
            let j = i ^ qubit_mask; // Flip the target qubit
            if (i & qubit_mask) == 0 {
                // |0⟩ → i|1⟩
                new_amplitudes[j] = i_complex * self.amplitudes[i];
                new_amplitudes[i] = Complex64::new(0.0, 0.0);
            } else {
                // |1⟩ → -i|0⟩
                new_amplitudes[j] = -i_complex * self.amplitudes[i];
                new_amplitudes[i] = Complex64::new(0.0, 0.0);
            }
        }

        self.amplitudes = new_amplitudes;
        Ok(())
    }

    /// Apply Pauli-Z gate (phase flip) to specific qubit
    ///
    /// The Pauli-Z gate performs a phase flip: |0⟩ → |0⟩, |1⟩ → -|1⟩
    ///
    /// # Arguments
    /// * `qubit` - Index of the qubit to apply the gate to
    ///
    /// # Errors
    /// Returns `SpatialError::InvalidInput` if the qubit index is out of range
    pub fn pauli_z(&mut self, qubit: usize) -> SpatialResult<()> {
        if qubit >= self.numqubits {
            return Err(SpatialError::InvalidInput(format!(
                "Qubit index {qubit} out of range"
            )));
        }

        let qubit_mask = 1 << qubit;

        for i in 0..self.amplitudes.len() {
            if (i & qubit_mask) != 0 {
                self.amplitudes[i] *= -1.0;
            }
        }

        Ok(())
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.numqubits
    }

    /// Get number of basis states (2^n)
    pub fn num_states(&self) -> usize {
        self.amplitudes.len()
    }

    /// Check if the state is normalized
    pub fn is_normalized(&self) -> bool {
        let norm_squared: f64 = self.amplitudes.iter().map(|amp| amp.norm_sqr()).sum();
        (norm_squared - 1.0).abs() < 1e-10
    }

    /// Normalize the quantum state
    pub fn normalize(&mut self) {
        let norm: f64 = self
            .amplitudes
            .iter()
            .map(|amp| amp.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm > 1e-10 {
            for amp in self.amplitudes.iter_mut() {
                *amp /= norm;
            }
        }
    }

    /// Get the amplitude for a specific basis state
    pub fn amplitude(&self, state: usize) -> Option<QuantumAmplitude> {
        self.amplitudes.get(state).copied()
    }

    /// Set the amplitude for a specific basis state
    pub fn set_amplitude(
        &mut self,
        state: usize,
        amplitude: QuantumAmplitude,
    ) -> SpatialResult<()> {
        if state >= self.amplitudes.len() {
            return Err(SpatialError::InvalidInput(
                "State index out of range".to_string(),
            ));
        }
        self.amplitudes[state] = amplitude;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_zero_state_creation() {
        let state = QuantumState::zero_state(2);
        assert_eq!(state.num_qubits(), 2);
        assert_eq!(state.num_states(), 4);
        assert_eq!(state.probability(0), 1.0);
        assert_eq!(state.probability(1), 0.0);
        assert!(state.is_normalized());
    }

    #[test]
    fn test_uniform_superposition() {
        let state = QuantumState::uniform_superposition(2);
        assert_eq!(state.num_qubits(), 2);
        assert_eq!(state.num_states(), 4);

        // Each state should have equal probability
        for i in 0..4 {
            assert!((state.probability(i) - 0.25).abs() < 1e-10);
        }
        assert!(state.is_normalized());
    }

    #[test]
    fn test_hadamard_gate() {
        let mut state = QuantumState::zero_state(1);
        state.hadamard(0).unwrap();

        // Should create equal superposition
        assert!((state.probability(0) - 0.5).abs() < 1e-10);
        assert!((state.probability(1) - 0.5).abs() < 1e-10);
        assert!(state.is_normalized());
    }

    #[test]
    fn test_pauli_x_gate() {
        let mut state = QuantumState::zero_state(1);
        state.pauli_x(0).unwrap();

        // Should flip |0⟩ to |1⟩
        assert_eq!(state.probability(0), 0.0);
        assert_eq!(state.probability(1), 1.0);
        assert!(state.is_normalized());
    }

    #[test]
    fn test_phase_rotation() {
        let mut state = QuantumState::uniform_superposition(1);
        state.phase_rotation(0, PI).unwrap();

        // Should apply -1 phase to |1⟩ component
        assert!(state.is_normalized());
        assert!((state.probability(0) - 0.5).abs() < 1e-10);
        assert!((state.probability(1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_controlled_rotation() {
        let mut state = QuantumState::zero_state(2);
        // First create entanglement
        state.hadamard(0).unwrap();
        state.controlled_rotation(0, 1, PI).unwrap();

        assert!(state.is_normalized());
        // Should be entangled Bell state
        assert!((state.probability(0) - 0.5).abs() < 1e-10);
        assert!((state.probability(3) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_measurement() {
        let state = QuantumState::zero_state(2);
        let result = state.measure();
        assert_eq!(result, 0); // Should always measure |00⟩
    }

    #[test]
    fn test_invalid_qubit_index() {
        let mut state = QuantumState::zero_state(2);
        assert!(state.hadamard(2).is_err()); // Out of range
        assert!(state.pauli_x(2).is_err());
        assert!(state.phase_rotation(2, PI).is_err());
    }

    #[test]
    fn test_amplitude_access() {
        let state = QuantumState::zero_state(2);
        assert_eq!(state.amplitude(0), Some(Complex64::new(1.0, 0.0)));
        assert_eq!(state.amplitude(1), Some(Complex64::new(0.0, 0.0)));
        assert_eq!(state.amplitude(10), None); // Out of range
    }

    #[test]
    fn test_normalization() {
        let amplitudes = Array1::from_vec(vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        let mut state = QuantumState::new(amplitudes).unwrap();

        assert!(!state.is_normalized());
        state.normalize();
        assert!(state.is_normalized());
        assert!((state.probability(0) - 0.5).abs() < 1e-10);
        assert!((state.probability(1) - 0.5).abs() < 1e-10);
    }
}
