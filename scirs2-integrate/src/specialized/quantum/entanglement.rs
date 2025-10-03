//! Multi-particle entanglement and Bell state systems
//!
//! This module provides functionality for creating and manipulating
//! multi-particle entangled quantum states, including Bell states
//! and general entanglement measures.

use crate::error::{IntegrateError, IntegrateResult as Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;

/// Multi-particle entanglement handling system
#[derive(Debug, Clone)]
pub struct MultiParticleEntanglement {
    /// Number of particles
    pub n_particles: usize,
    /// Hilbert space dimension
    pub hilbert_dim: usize,
    /// Entangled state representation
    pub state: Array1<Complex64>,
    /// Particle masses
    pub masses: Array1<f64>,
    /// Interaction strength matrix
    pub interactions: Array2<f64>,
}

impl MultiParticleEntanglement {
    /// Create new multi-particle entangled system
    pub fn new(nparticles: usize, masses: Array1<f64>) -> Self {
        let hilbert_dim = 2_usize.pow(nparticles as u32); // For spin-1/2 particles
        let state = Array1::zeros(hilbert_dim);
        let interactions = Array2::zeros((nparticles, nparticles));

        Self {
            n_particles: nparticles,
            hilbert_dim,
            state,
            masses,
            interactions,
        }
    }

    /// Create Bell state (two-particle entanglement)
    pub fn create_bell_state(&mut self, belltype: BellState) -> Result<()> {
        if self.n_particles != 2 {
            return Err(IntegrateError::InvalidInput(
                "Bell states require exactly 2 particles".to_string(),
            ));
        }

        let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
        self.state = Array1::zeros(4);

        match belltype {
            BellState::PhiPlus => {
                // |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
                self.state[0] = Complex64::new(inv_sqrt2, 0.0); // |00⟩
                self.state[3] = Complex64::new(inv_sqrt2, 0.0); // |11⟩
            }
            BellState::PhiMinus => {
                // |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
                self.state[0] = Complex64::new(inv_sqrt2, 0.0); // |00⟩
                self.state[3] = Complex64::new(-inv_sqrt2, 0.0); // |11⟩
            }
            BellState::PsiPlus => {
                // |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
                self.state[1] = Complex64::new(inv_sqrt2, 0.0); // |01⟩
                self.state[2] = Complex64::new(inv_sqrt2, 0.0); // |10⟩
            }
            BellState::PsiMinus => {
                // |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
                self.state[1] = Complex64::new(inv_sqrt2, 0.0); // |01⟩
                self.state[2] = Complex64::new(-inv_sqrt2, 0.0); // |10⟩
            }
        }

        Ok(())
    }

    /// Create GHZ state (multi-particle entanglement)
    pub fn create_ghz_state(&mut self) -> Result<()> {
        if self.n_particles < 3 {
            return Err(IntegrateError::InvalidInput(
                "GHZ states require at least 3 particles".to_string(),
            ));
        }

        let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
        self.state = Array1::zeros(self.hilbert_dim);

        // |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2
        self.state[0] = Complex64::new(inv_sqrt2, 0.0); // |00...0⟩
        self.state[self.hilbert_dim - 1] = Complex64::new(inv_sqrt2, 0.0); // |11...1⟩

        Ok(())
    }

    /// Create W state (symmetric entanglement)
    pub fn create_w_state(&mut self) -> Result<()> {
        if self.n_particles < 3 {
            return Err(IntegrateError::InvalidInput(
                "W states require at least 3 particles".to_string(),
            ));
        }

        let inv_sqrt_n = 1.0 / (self.n_particles as f64).sqrt();
        self.state = Array1::zeros(self.hilbert_dim);

        // |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
        for i in 0..self.n_particles {
            let state_index = 1 << (self.n_particles - 1 - i);
            self.state[state_index] = Complex64::new(inv_sqrt_n, 0.0);
        }

        Ok(())
    }

    /// Calculate entanglement entropy (von Neumann entropy)
    pub fn calculate_entanglement_entropy(&self, subsystemqubits: &[usize]) -> Result<f64> {
        // Calculate reduced density matrix for the subsystem
        let rho_sub = self.reduced_density_matrix(subsystemqubits)?;

        // Calculate eigenvalues of reduced density matrix
        let eigenvalues = self.compute_eigenvalues(&rho_sub)?;

        // Calculate von Neumann entropy: S = -Tr(ρ log ρ)
        let mut entropy = 0.0;
        for &lambda in &eigenvalues {
            if lambda > 1e-12 {
                entropy += -lambda * lambda.ln();
            }
        }

        Ok(entropy)
    }

    /// Calculate reduced density matrix for a subsystem
    fn reduced_density_matrix(&self, subsystemqubits: &[usize]) -> Result<Array2<Complex64>> {
        let subsystem_size = subsystemqubits.len();
        let subsystem_dim = 1 << subsystem_size;
        let mut rho_sub = Array2::zeros((subsystem_dim, subsystem_dim));

        // Trace out the environment qubits
        for i in 0..subsystem_dim {
            for j in 0..subsystem_dim {
                let mut sum = Complex64::new(0.0, 0.0);

                // Sum over all environment configurations
                let env_size = self.n_particles - subsystem_size;
                let env_dim = 1 << env_size;

                for env_config in 0..env_dim {
                    let full_i = self.combine_subsystem_env(i, env_config, subsystemqubits);
                    let full_j = self.combine_subsystem_env(j, env_config, subsystemqubits);

                    if full_i < self.hilbert_dim && full_j < self.hilbert_dim {
                        sum += self.state[full_i].conj() * self.state[full_j];
                    }
                }

                rho_sub[[i, j]] = sum;
            }
        }

        Ok(rho_sub)
    }

    /// Combine subsystem and environment configurations
    fn combine_subsystem_env(
        &self,
        sub_config: usize,
        env_config: usize,
        subsystem_qubits: &[usize],
    ) -> usize {
        let mut full_config = 0;
        let mut env_bit = 0;

        for qubit in 0..self.n_particles {
            if subsystem_qubits.contains(&qubit) {
                // This qubit is in the subsystem
                let sub_bit_pos = subsystem_qubits.iter().position(|&x| x == qubit).unwrap();
                if (sub_config >> sub_bit_pos) & 1 == 1 {
                    full_config |= 1 << qubit;
                }
            } else {
                // This qubit is in the environment
                if (env_config >> env_bit) & 1 == 1 {
                    full_config |= 1 << qubit;
                }
                env_bit += 1;
            }
        }

        full_config
    }

    /// Compute eigenvalues of a density matrix (simplified)
    fn compute_eigenvalues(&self, rho: &Array2<Complex64>) -> Result<Vec<f64>> {
        let n = rho.nrows();
        let mut eigenvalues = Vec::new();

        // For simplicity, just extract diagonal elements as approximate eigenvalues
        // In a real implementation, this would use proper eigenvalue decomposition
        for i in 0..n {
            eigenvalues.push(rho[[i, i]].re);
        }

        Ok(eigenvalues)
    }

    /// Calculate concurrence (entanglement measure for two qubits)
    pub fn calculate_concurrence(&self) -> Result<f64> {
        if self.n_particles != 2 {
            return Err(IntegrateError::InvalidInput(
                "Concurrence is defined only for two-qubit systems".to_string(),
            ));
        }

        // Create density matrix from state vector
        let mut rho = Array2::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                rho[[i, j]] = self.state[i].conj() * self.state[j];
            }
        }

        // Calculate concurrence using Wootters' formula
        // C = max{0, λ₁ - λ₂ - λ₃ - λ₄}
        // where λᵢ are the eigenvalues of ρ(σy ⊗ σy)ρ*(σy ⊗ σy) in decreasing order

        // Simplified calculation - for a pure state |ψ⟩ = a|00⟩ + b|01⟩ + c|10⟩ + d|11⟩
        // C = 2|ad - bc|
        let a = self.state[0];
        let b = self.state[1];
        let c = self.state[2];
        let d = self.state[3];

        let concurrence = 2.0 * (a * d - b * c).norm();

        Ok(concurrence)
    }

    /// Apply controlled-NOT gate between two qubits
    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<()> {
        if control >= self.n_particles || target >= self.n_particles {
            return Err(IntegrateError::InvalidInput(
                "Qubit indices out of range".to_string(),
            ));
        }

        let mut new_state = Array1::zeros(self.hilbert_dim);

        for i in 0..self.hilbert_dim {
            let control_bit = (i >> (self.n_particles - 1 - control)) & 1;
            let target_bit = (i >> (self.n_particles - 1 - target)) & 1;

            let new_i = if control_bit == 1 {
                // Flip target bit
                i ^ (1 << (self.n_particles - 1 - target))
            } else {
                i
            };

            new_state[new_i] = self.state[i];
        }

        self.state = new_state;
        Ok(())
    }

    /// Apply Hadamard gate to a qubit
    pub fn apply_hadamard(&mut self, qubit: usize) -> Result<()> {
        if qubit >= self.n_particles {
            return Err(IntegrateError::InvalidInput(
                "Qubit index out of range".to_string(),
            ));
        }

        let mut new_state = Array1::zeros(self.hilbert_dim);
        let inv_sqrt2 = Complex64::new(1.0 / (2.0_f64).sqrt(), 0.0);

        for i in 0..self.hilbert_dim {
            let bit = (i >> (self.n_particles - 1 - qubit)) & 1;
            let flipped_i = i ^ (1 << (self.n_particles - 1 - qubit));

            if bit == 0 {
                // |0⟩ → (|0⟩ + |1⟩)/√2
                new_state[i] += inv_sqrt2 * self.state[i];
                new_state[flipped_i] += inv_sqrt2 * self.state[i];
            } else {
                // |1⟩ → (|0⟩ - |1⟩)/√2
                new_state[flipped_i] += inv_sqrt2 * self.state[i];
                new_state[i] -= inv_sqrt2 * self.state[i];
            }
        }

        self.state = new_state;
        Ok(())
    }

    /// Measure entanglement witness
    pub fn measure_entanglement_witness(
        &self,
        witness_operator: &Array2<Complex64>,
    ) -> Result<f64> {
        if witness_operator.nrows() != self.hilbert_dim
            || witness_operator.ncols() != self.hilbert_dim
        {
            return Err(IntegrateError::InvalidInput(
                "Witness operator dimension mismatch".to_string(),
            ));
        }

        // Calculate expectation value ⟨ψ|W|ψ⟩
        let mut expectation = Complex64::new(0.0, 0.0);

        for i in 0..self.hilbert_dim {
            for j in 0..self.hilbert_dim {
                expectation += self.state[i].conj() * witness_operator[[i, j]] * self.state[j];
            }
        }

        Ok(expectation.re)
    }

    /// Normalize the quantum state
    pub fn normalize(&mut self) {
        let norm_squared: f64 = self.state.iter().map(|&c| (c.conj() * c).re).sum();

        let norm = norm_squared.sqrt();
        if norm > 1e-12 {
            self.state.mapv_inplace(|c| c / norm);
        }
    }

    /// Get the current quantum state
    pub fn get_state(&self) -> &Array1<Complex64> {
        &self.state
    }

    /// Set the quantum state (with normalization)
    pub fn set_state(&mut self, newstate: Array1<Complex64>) -> Result<()> {
        if newstate.len() != self.hilbert_dim {
            return Err(IntegrateError::InvalidInput(
                "State dimension mismatch".to_string(),
            ));
        }

        self.state = newstate;
        self.normalize();
        Ok(())
    }
}

/// Bell state types for two-qubit entanglement
#[derive(Debug, Clone, Copy)]
pub enum BellState {
    /// |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    PhiPlus,
    /// |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    PhiMinus,
    /// |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    PsiPlus,
    /// |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    PsiMinus,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bell_state_creation() {
        let masses = Array1::from_vec(vec![1.0, 1.0]);
        let mut system = MultiParticleEntanglement::new(2, masses);

        // Test Φ⁺ state
        system.create_bell_state(BellState::PhiPlus).unwrap();
        let state = system.get_state();

        let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
        assert_relative_eq!(state[0].re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(state[3].re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(state[1].norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[2].norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ghz_state() {
        let masses = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let mut system = MultiParticleEntanglement::new(3, masses);

        system.create_ghz_state().unwrap();
        let state = system.get_state();

        let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
        assert_relative_eq!(state[0].re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(state[7].re, inv_sqrt2, epsilon = 1e-10);

        // Other states should be zero
        for i in 1..7 {
            assert_relative_eq!(state[i].norm(), 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_w_state() {
        let masses = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let mut system = MultiParticleEntanglement::new(3, masses);

        system.create_w_state().unwrap();
        let state = system.get_state();

        let inv_sqrt3 = 1.0 / 3.0_f64.sqrt();
        assert_relative_eq!(state[1].re, inv_sqrt3, epsilon = 1e-10); // |001⟩
        assert_relative_eq!(state[2].re, inv_sqrt3, epsilon = 1e-10); // |010⟩
        assert_relative_eq!(state[4].re, inv_sqrt3, epsilon = 1e-10); // |100⟩

        // Other states should be zero
        assert_relative_eq!(state[0].norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[3].norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[5].norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[6].norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[7].norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_concurrence() {
        let masses = Array1::from_vec(vec![1.0, 1.0]);
        let mut system = MultiParticleEntanglement::new(2, masses);

        // Test maximally entangled Bell state
        system.create_bell_state(BellState::PhiPlus).unwrap();
        let concurrence = system.calculate_concurrence().unwrap();
        assert_relative_eq!(concurrence, 1.0, epsilon = 1e-10);

        // Test separable state |00⟩
        let separable_state = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        system.set_state(separable_state).unwrap();
        let concurrence = system.calculate_concurrence().unwrap();
        assert_relative_eq!(concurrence, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quantum_gates() {
        let masses = Array1::from_vec(vec![1.0, 1.0]);
        let mut system = MultiParticleEntanglement::new(2, masses);

        // Start with |00⟩
        let initial_state = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        system.set_state(initial_state).unwrap();

        // Apply Hadamard to first qubit: |00⟩ → (|00⟩ + |10⟩)/√2
        system.apply_hadamard(0).unwrap();
        let state = system.get_state();

        let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
        assert_relative_eq!(state[0].re, inv_sqrt2, epsilon = 1e-10); // |00⟩
        assert_relative_eq!(state[2].re, inv_sqrt2, epsilon = 1e-10); // |10⟩

        // Apply CNOT: (|00⟩ + |10⟩)/√2 → (|00⟩ + |11⟩)/√2
        system.apply_cnot(0, 1).unwrap();
        let state = system.get_state();

        assert_relative_eq!(state[0].re, inv_sqrt2, epsilon = 1e-10); // |00⟩
        assert_relative_eq!(state[3].re, inv_sqrt2, epsilon = 1e-10); // |11⟩
        assert_relative_eq!(state[1].norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[2].norm(), 0.0, epsilon = 1e-10);
    }
}
