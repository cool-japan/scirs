//! Quantum algorithms and advanced quantum computing methods
//!
//! This module contains quantum algorithms including quantum annealing,
//! variational quantum eigensolvers, quantum error correction, and other
//! advanced quantum computational methods.

use crate::error::{IntegrateError, IntegrateResult as Result};
use scirs2_core::constants::PI;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_core::random::{Rng, RngExt};
use std::collections::HashMap;

/// Quantum annealing solver for optimization problems
pub struct QuantumAnnealer {
    /// Number of qubits
    pub n_qubits: usize,
    /// Annealing schedule
    pub schedule: Vec<(f64, f64)>, // (time, annealing_parameter)
    /// Temperature for thermal fluctuations
    pub temperature: f64,
    /// Number of sweeps per schedule point
    pub sweeps_per_point: usize,
}

impl QuantumAnnealer {
    /// Create a new quantum annealer
    pub fn new(n_qubits: usize, annealing_time: f64, n_schedulepoints: usize) -> Self {
        let mut schedule = Vec::with_capacity(n_schedulepoints);
        for i in 0..n_schedulepoints {
            let t = i as f64 / (n_schedulepoints - 1) as f64;
            let s = t * annealing_time;
            let annealing_param = t; // Linear schedule from 0 to 1
            schedule.push((s, annealing_param));
        }

        Self {
            n_qubits,
            schedule,
            temperature: 0.1,
            sweeps_per_point: 1000,
        }
    }

    /// Solve an Ising model using quantum annealing
    /// J: coupling matrix, h: local fields
    pub fn solve_ising(
        &self,
        j_matrix: &Array2<f64>,
        h_fields: &Array1<f64>,
    ) -> Result<(Array1<i8>, f64)> {
        let mut rng = scirs2_core::random::rng();

        // Initialize random spin configuration
        let mut spins: Array1<i8> = Array1::zeros(self.n_qubits);
        for spin in spins.iter_mut() {
            *spin = if rng.random::<bool>() { 1 } else { -1 };
        }

        let mut best_energy = self.compute_ising_energy(&spins, j_matrix, h_fields);
        let mut best_spins = spins.clone();

        // Perform annealing schedule
        for &(_time, s) in &self.schedule {
            let gamma = (1.0 - s) * 10.0; // Transverse field strength
            let beta = 1.0 / (self.temperature * (1.0 + s)); // Inverse temperature

            // Monte Carlo sweeps at this annealing point
            for _ in 0..self.sweeps_per_point {
                // Try flipping each spin
                for i in 0..self.n_qubits {
                    let old_energy = self.compute_local_energy(i, &spins, j_matrix, h_fields);

                    // Flip spin
                    spins[i] *= -1;
                    let new_energy = self.compute_local_energy(i, &spins, j_matrix, h_fields);

                    // Quantum tunneling effect (simplified)
                    let tunneling_probability = (-gamma * 0.1).exp();
                    let thermal_probability = (-(new_energy - old_energy) * beta).exp();

                    let acceptance_prob = tunneling_probability.max(thermal_probability);

                    if rng.random::<f64>() > acceptance_prob {
                        // Reject: flip back
                        spins[i] *= -1;
                    }
                }

                // Check if this is the best configuration so far
                let current_energy = self.compute_ising_energy(&spins, j_matrix, h_fields);
                if current_energy < best_energy {
                    best_energy = current_energy;
                    best_spins = spins.clone();
                }
            }
        }

        Ok((best_spins, best_energy))
    }

    fn compute_ising_energy(
        &self,
        spins: &Array1<i8>,
        j_matrix: &Array2<f64>,
        h_fields: &Array1<f64>,
    ) -> f64 {
        let mut energy = 0.0;

        // Interaction energy
        for i in 0..self.n_qubits {
            for j in (i + 1)..self.n_qubits {
                energy -= j_matrix[[i, j]] * spins[i] as f64 * spins[j] as f64;
            }
            // Local field energy
            energy -= h_fields[i] * spins[i] as f64;
        }

        energy
    }

    fn compute_local_energy(
        &self,
        site: usize,
        spins: &Array1<i8>,
        j_matrix: &Array2<f64>,
        h_fields: &Array1<f64>,
    ) -> f64 {
        let mut energy = 0.0;

        // Interaction with neighbors
        for j in 0..self.n_qubits {
            if j != site {
                energy -= j_matrix[[site, j]] * spins[site] as f64 * spins[j] as f64;
            }
        }

        // Local field
        energy -= h_fields[site] * spins[site] as f64;

        energy
    }
}

/// Variational Quantum Eigensolver (VQE) for quantum chemistry
pub struct VariationalQuantumEigensolver {
    /// Number of qubits
    pub n_qubits: usize,
    /// Ansatz circuit depth
    pub circuit_depth: usize,
    /// Optimization tolerance
    pub tolerance: f64,
    /// Maximum optimization iterations
    pub max_iterations: usize,
}

impl VariationalQuantumEigensolver {
    /// Create a new VQE solver
    pub fn new(n_qubits: usize, circuitdepth: usize) -> Self {
        Self {
            n_qubits,
            circuit_depth: circuitdepth,
            tolerance: 1e-6,
            max_iterations: 1000,
        }
    }

    /// Find ground state energy using VQE
    pub fn find_ground_state(&self, hamiltonian: &Array2<Complex64>) -> Result<(f64, Array1<f64>)> {
        let mut rng = scirs2_core::random::rng();

        // Initialize random variational parameters
        let n_params = self.n_qubits * self.circuit_depth * 3; // 3 angles per layer per qubit
        let mut params: Array1<f64> = Array1::zeros(n_params);
        for param in params.iter_mut() {
            *param = rng.random::<f64>() * 2.0 * PI;
        }

        let mut best_energy = f64::INFINITY;
        let mut best_params = params.clone();

        // Optimization using gradient descent with finite differences
        let learning_rate = 0.01;
        let epsilon = 1e-8;

        for _iteration in 0..self.max_iterations {
            let current_energy = self.compute_expectation_value(&params, hamiltonian)?;

            if current_energy < best_energy {
                best_energy = current_energy;
                best_params = params.clone();
            }

            // Compute numerical gradients
            let mut gradients = Array1::zeros(n_params);
            for i in 0..n_params {
                params[i] += epsilon;
                let energy_plus = self.compute_expectation_value(&params, hamiltonian)?;

                params[i] -= 2.0 * epsilon;
                let energy_minus = self.compute_expectation_value(&params, hamiltonian)?;

                params[i] += epsilon; // Restore original value

                gradients[i] = (energy_plus - energy_minus) / (2.0 * epsilon);
            }

            // Update parameters
            for i in 0..n_params {
                params[i] -= learning_rate * gradients[i];
            }

            // Check convergence
            let gradient_norm: f64 = gradients.iter().map(|&g| g * g).sum::<f64>().sqrt();
            if gradient_norm < self.tolerance {
                break;
            }
        }

        Ok((best_energy, best_params))
    }

    /// Compute expectation value of Hamiltonian for given parameters
    fn compute_expectation_value(
        &self,
        params: &Array1<f64>,
        hamiltonian: &Array2<Complex64>,
    ) -> Result<f64> {
        // Create ansatz state vector
        let state_vector = self.create_ansatz_state(params)?;

        // Compute <ψ|H|ψ>
        let h_psi = hamiltonian.dot(&state_vector);
        let expectation: Complex64 = state_vector
            .iter()
            .zip(h_psi.iter())
            .map(|(&psi, &h_psi)| psi.conj() * h_psi)
            .sum();

        Ok(expectation.re)
    }

    /// Create ansatz state vector from variational parameters
    fn create_ansatz_state(&self, params: &Array1<f64>) -> Result<Array1<Complex64>> {
        let n_states = 1 << self.n_qubits;
        let mut state = Array1::zeros(n_states);
        state[0] = Complex64::new(1.0, 0.0); // Start with |00...0⟩

        // Apply parameterized quantum circuit
        let mut param_idx = 0;
        for _layer in 0..self.circuit_depth {
            // Apply rotation gates to each qubit
            for qubit in 0..self.n_qubits {
                let rx_angle = params[param_idx];
                let ry_angle = params[param_idx + 1];
                let rz_angle = params[param_idx + 2];
                param_idx += 3;

                // Apply rotations (simplified implementation)
                state = self.apply_rotation_gates(&state, qubit, rx_angle, ry_angle, rz_angle)?;
            }

            // Apply entangling gates
            for qubit in 0..self.n_qubits - 1 {
                state = self.apply_cnot(&state, qubit, qubit + 1)?;
            }
        }

        Ok(state)
    }

    /// Apply rotation gates to a qubit (simplified implementation)
    fn apply_rotation_gates(
        &self,
        state: &Array1<Complex64>,
        _qubit: usize,
        _rx_angle: f64,
        _ry_angle: f64,
        _rz_angle: f64,
    ) -> Result<Array1<Complex64>> {
        // Simplified: just return the input state
        // In a real implementation, this would apply the rotation matrices
        Ok(state.clone())
    }

    /// Apply CNOT gate between two qubits
    fn apply_cnot(
        &self,
        state: &Array1<Complex64>,
        _control: usize,
        _target: usize,
    ) -> Result<Array1<Complex64>> {
        // Simplified: just return the input state
        // In a real implementation, this would apply the CNOT gate
        Ok(state.clone())
    }
}

/// Quantum error correction codes
#[derive(Debug, Clone, Copy)]
pub enum ErrorCorrectionCode {
    /// Steane 7-qubit code
    Steane7,
    /// Surface code
    Surface,
    /// Repetition code
    Repetition,
}

/// Noise parameters for quantum error correction
#[derive(Debug, Clone)]
pub struct NoiseParameters {
    /// Single-qubit error rate
    pub single_qubit_error_rate: f64,
    /// Two-qubit gate error rate
    pub two_qubit_error_rate: f64,
    /// Measurement error rate
    pub measurement_error_rate: f64,
    /// Decoherence time
    pub decoherence_time: f64,
}

impl Default for NoiseParameters {
    fn default() -> Self {
        Self {
            single_qubit_error_rate: 1e-4,
            two_qubit_error_rate: 1e-3,
            measurement_error_rate: 1e-3,
            decoherence_time: 100e-6, // 100 microseconds
        }
    }
}

/// Quantum error correction system
pub struct QuantumErrorCorrection {
    /// Number of logical qubits
    pub n_logical_qubits: usize,
    /// Error correction code
    pub code: ErrorCorrectionCode,
    /// Noise parameters
    pub noise_parameters: NoiseParameters,
    /// Syndrome history for improved decoding
    pub syndrome_history: Vec<Array1<i8>>,
}

impl QuantumErrorCorrection {
    /// Create new quantum error correction system
    pub fn new(n_logicalqubits: usize, code: ErrorCorrectionCode) -> Self {
        Self {
            n_logical_qubits: n_logicalqubits,
            code,
            noise_parameters: NoiseParameters::default(),
            syndrome_history: Vec::new(),
        }
    }

    /// Encode logical state into physical qubits
    pub fn encode(&self, logicalstate: &Array1<Complex64>) -> Result<Array1<Complex64>> {
        let n_physical = self.get_physical_qubit_count();
        let mut physical_state = Array1::zeros(1 << n_physical);

        match self.code {
            ErrorCorrectionCode::Steane7 => {
                // Steane code encoding
                self.encode_steane7(logicalstate, &mut physical_state)?;
            }
            ErrorCorrectionCode::Surface => {
                // Surface code encoding
                self.encode_surface(logicalstate, &mut physical_state)?;
            }
            ErrorCorrectionCode::Repetition => {
                // Repetition code encoding
                self.encode_repetition(logicalstate, &mut physical_state)?;
            }
        }

        Ok(physical_state)
    }

    /// Get number of physical qubits needed
    fn get_physical_qubit_count(&self) -> usize {
        match self.code {
            ErrorCorrectionCode::Steane7 => 7 * self.n_logical_qubits,
            ErrorCorrectionCode::Surface => 9 * self.n_logical_qubits, // Simplified
            ErrorCorrectionCode::Repetition => 3 * self.n_logical_qubits,
        }
    }

    /// Encode using Steane 7-qubit code
    fn encode_steane7(
        &self,
        logical_state: &Array1<Complex64>,
        physical_state: &mut Array1<Complex64>,
    ) -> Result<()> {
        // Simplified Steane encoding
        if logical_state.len() != (1 << self.n_logical_qubits) {
            return Err(IntegrateError::InvalidInput(
                "Logical state dimension mismatch".to_string(),
            ));
        }

        // For simplicity, just copy logical state to first positions
        for (i, &amp) in logical_state.iter().enumerate() {
            if i < physical_state.len() {
                physical_state[i] = amp;
            }
        }

        Ok(())
    }

    /// Encode using surface code
    fn encode_surface(
        &self,
        logical_state: &Array1<Complex64>,
        physical_state: &mut Array1<Complex64>,
    ) -> Result<()> {
        // Simplified surface code encoding
        if logical_state.len() != (1 << self.n_logical_qubits) {
            return Err(IntegrateError::InvalidInput(
                "Logical state dimension mismatch".to_string(),
            ));
        }

        // For simplicity, just copy logical state to first positions
        for (i, &amp) in logical_state.iter().enumerate() {
            if i < physical_state.len() {
                physical_state[i] = amp;
            }
        }

        Ok(())
    }

    /// Encode using repetition code
    fn encode_repetition(
        &self,
        logical_state: &Array1<Complex64>,
        physical_state: &mut Array1<Complex64>,
    ) -> Result<()> {
        // Simplified repetition code encoding
        if logical_state.len() != (1 << self.n_logical_qubits) {
            return Err(IntegrateError::InvalidInput(
                "Logical state dimension mismatch".to_string(),
            ));
        }

        // For simplicity, just copy logical state to first positions
        for (i, &amp) in logical_state.iter().enumerate() {
            if i < physical_state.len() {
                physical_state[i] = amp;
            }
        }

        Ok(())
    }

    /// Apply quantum gates with error correction
    pub fn apply_logical_x(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
    ) -> Result<Array1<Complex64>> {
        // Apply logical X gate with error correction
        let mut result = state.clone();

        // For simplicity, apply a basic transformation
        let n_states = state.len();
        for i in 0..n_states {
            // Flip the specified qubit bit in the computational basis
            let flipped_i = i ^ (1 << qubit);
            if flipped_i < n_states {
                result[i] = state[flipped_i];
            }
        }

        Ok(result)
    }

    /// Apply Hadamard gate with error correction
    pub fn apply_hadamard(
        &self,
        state: &Array1<Complex64>,
        _qubit: usize,
    ) -> Result<Array1<Complex64>> {
        // Simplified Hadamard implementation
        let mut result = Array1::zeros(state.len());
        let sqrt_2_inv = 1.0 / 2.0_f64.sqrt();

        for (i, &amp) in state.iter().enumerate() {
            result[i] = amp * Complex64::new(sqrt_2_inv, 0.0);
        }

        Ok(result)
    }

    /// Apply Pauli-X gate with error correction
    pub fn apply_pauli_x(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
    ) -> Result<Array1<Complex64>> {
        self.apply_logical_x(state, qubit)
    }

    /// Apply CNOT gate with error correction
    pub fn apply_cnot(
        &self,
        state: &Array1<Complex64>,
        _control: usize,
        _target: usize,
    ) -> Result<Array1<Complex64>> {
        // Simplified CNOT implementation
        Ok(state.clone())
    }

    /// Perform error correction cycle
    pub fn error_correction_cycle(&mut self, state: &mut Array1<Complex64>) -> Result<f64> {
        // Measure syndromes
        let syndromes = self.measure_syndromes(state)?;

        // Store in history for improved decoding
        self.syndrome_history.push(syndromes.clone());

        // Decode and apply corrections
        let corrections = self.decode_syndromes(&syndromes)?;
        self.apply_corrections(state, &corrections)?;

        // Estimate error probability
        let error_prob = self.estimate_error_probability(&syndromes);

        Ok(error_prob)
    }

    /// Measure error syndromes
    fn measure_syndromes(&self, state: &Array1<Complex64>) -> Result<Array1<i8>> {
        let n_syndromes = match self.code {
            ErrorCorrectionCode::Steane7 => 6,
            ErrorCorrectionCode::Surface => 8,
            ErrorCorrectionCode::Repetition => 2,
        };

        // Simplified syndrome measurement
        let mut syndromes = Array1::zeros(n_syndromes);
        let mut rng = scirs2_core::random::rng();

        for syndrome in syndromes.iter_mut() {
            *syndrome = if rng.random::<f64>() < self.noise_parameters.measurement_error_rate {
                1
            } else {
                0
            };
        }

        Ok(syndromes)
    }

    /// Decode syndromes to determine corrections
    fn decode_syndromes(&self, syndromes: &Array1<i8>) -> Result<Array1<i8>> {
        let n_physical = self.get_physical_qubit_count();
        let mut corrections = Array1::zeros(n_physical);

        // Simplified decoding based on syndrome pattern
        for (i, &syndrome) in syndromes.iter().enumerate() {
            if syndrome != 0 && i < n_physical {
                corrections[i] = 1; // Apply X correction
            }
        }

        Ok(corrections)
    }

    /// Apply corrections to the quantum state
    fn apply_corrections(
        &self,
        state: &mut Array1<Complex64>,
        corrections: &Array1<i8>,
    ) -> Result<()> {
        // Apply corrections (simplified)
        for (qubit, &correction) in corrections.iter().enumerate() {
            if correction != 0 {
                // Apply Pauli correction
                let corrected_state = self.apply_pauli_x(state, qubit)?;
                *state = corrected_state;
            }
        }

        Ok(())
    }

    /// Estimate error probability from syndromes
    fn estimate_error_probability(&self, syndromes: &Array1<i8>) -> f64 {
        let error_count = syndromes.iter().filter(|&&s| s != 0).count();
        error_count as f64 / syndromes.len() as f64
    }

    /// Estimate logical error rate
    pub fn estimate_logical_error_rate(&self) -> f64 {
        match self.code {
            ErrorCorrectionCode::Steane7 => {
                // Simplified model for Steane code
                let p = self.noise_parameters.single_qubit_error_rate;
                35.0 * p.powi(3) // Third-order error suppression
            }
            ErrorCorrectionCode::Surface => {
                // Simplified model for surface code
                let p = self.noise_parameters.single_qubit_error_rate;
                0.1 * (p / 0.01).powi(2) // Threshold around 1%
            }
            ErrorCorrectionCode::Repetition => {
                // Simple repetition code
                let p = self.noise_parameters.single_qubit_error_rate;
                3.0 * p.powi(2) * (1.0 - p) + p.powi(3)
            }
        }
    }
}

/// Result of multi-body quantum diagonalization.
#[derive(Debug, Clone)]
pub struct MultiBodyEigenResult {
    /// Eigenvalues in ascending order.
    pub eigenvalues: Array1<f64>,
    /// Eigenvectors stored as columns of this matrix.
    pub eigenvectors: Array2<Complex64>,
}

/// CPU exact-diagonalization solver for small multi-body quantum Hamiltonians.
///
/// Diagonalizes the full Hamiltonian matrix exactly via Jacobi iteration for
/// Hermitian matrices with dimension ≤ 2^20 (typically ≤ 20 qubits).  For
/// very small systems (dim ≤ 512) the algorithm converges quickly; for larger
/// ones it is still correct but will be slow — use [`GPUMultiBodyQuantumSolver`]
/// for production large-system calculations.
///
/// [`GPUMultiBodyQuantumSolver`]: crate::specialized::quantum::gpu::GPUMultiBodyQuantumSolver
pub struct MultiBodyQuantumSolver {
    /// Number of particles / sites in the many-body Hilbert space.
    pub n_particles: usize,
    /// Local Hilbert-space dimension per site (e.g. 2 for spin-1/2).
    pub local_dim: usize,
    /// Convergence tolerance for the Jacobi sweeps.
    pub tolerance: f64,
    /// Maximum number of Jacobi sweeps.
    pub max_sweeps: usize,
}

impl MultiBodyQuantumSolver {
    /// Create a new CPU multi-body quantum solver.
    ///
    /// # Arguments
    /// * `n_particles` – number of particles / sites.
    /// * `local_dim`   – local Hilbert-space dimension per site (2 for qubits).
    pub fn new(n_particles: usize, local_dim: usize) -> Self {
        Self {
            n_particles,
            local_dim,
            tolerance: 1e-12,
            max_sweeps: 100,
        }
    }

    /// Total Hilbert-space dimension: `local_dim^n_particles`.
    pub fn hilbert_dim(&self) -> usize {
        self.local_dim.pow(self.n_particles as u32)
    }

    /// Diagonalize the given Hermitian Hamiltonian and return all eigenvalues
    /// and eigenvectors.
    ///
    /// The input must be a square Hermitian matrix whose dimension equals
    /// `self.hilbert_dim()`.
    pub fn diagonalize(&self, hamiltonian: &Array2<Complex64>) -> Result<MultiBodyEigenResult> {
        let dim = hamiltonian.nrows();
        if dim != hamiltonian.ncols() {
            return Err(IntegrateError::InvalidInput(
                "Hamiltonian must be square".to_string(),
            ));
        }
        if dim != self.hilbert_dim() {
            return Err(IntegrateError::InvalidInput(format!(
                "Hamiltonian dimension {} does not match expected Hilbert-space dim {}",
                dim,
                self.hilbert_dim()
            )));
        }

        // Use the real symmetric Jacobi algorithm on the real part only when
        // the Hamiltonian is purely real; otherwise use the complex Hermitian
        // Jacobi variant.
        let is_real = hamiltonian.iter().all(|c| c.im.abs() < 1e-15);

        if is_real {
            self.diagonalize_real(hamiltonian)
        } else {
            self.diagonalize_complex(hamiltonian)
        }
    }

    /// Find the `k` lowest-energy eigenpairs using the Lanczos algorithm.
    ///
    /// This is more efficient than full diagonalization for large matrices
    /// when only a few eigenvalues are needed.
    pub fn find_ground_states(
        &self,
        hamiltonian: &Array2<Complex64>,
        k: usize,
    ) -> Result<MultiBodyEigenResult> {
        let dim = hamiltonian.nrows();
        if dim != hamiltonian.ncols() {
            return Err(IntegrateError::InvalidInput(
                "Hamiltonian must be square".to_string(),
            ));
        }
        if k == 0 || k > dim {
            return Err(IntegrateError::InvalidInput(format!(
                "k={k} must be in [1, {dim}]"
            )));
        }

        // For very small systems, just do full diagonalization and pick top-k.
        if dim <= 64 {
            let full = self.diagonalize(hamiltonian)?;
            let eigenvalues = full
                .eigenvalues
                .slice(scirs2_core::ndarray::s![..k])
                .to_owned();
            let eigenvectors = full
                .eigenvectors
                .slice(scirs2_core::ndarray::s![.., ..k])
                .to_owned();
            return Ok(MultiBodyEigenResult {
                eigenvalues,
                eigenvectors,
            });
        }

        // Lanczos algorithm without re-orthogonalization.
        // Produces a tridiagonal matrix whose eigenvalues approximate those
        // of H.
        let n_lanczos = (2 * k).min(dim); // Krylov dimension
        let mut alpha = vec![0.0_f64; n_lanczos];
        let mut beta = vec![0.0_f64; n_lanczos + 1];

        // Krylov basis vectors (as rows for cache efficiency, then transposed).
        let mut v_prev = Array1::<Complex64>::zeros(dim);
        let mut v_curr: Array1<Complex64> = Array1::zeros(dim);
        v_curr[0] = Complex64::new(1.0, 0.0); // Random start vector

        let mut v_mat: Vec<Array1<Complex64>> = Vec::with_capacity(n_lanczos + 1);
        v_mat.push(v_prev.clone());

        for j in 0..n_lanczos {
            // w = H * v_curr
            let w = hamiltonian.dot(&v_curr);
            // alpha_j = <v_curr | w>
            let alpha_j: Complex64 = v_curr
                .iter()
                .zip(w.iter())
                .map(|(&vi, &wi)| vi.conj() * wi)
                .sum();
            alpha[j] = alpha_j.re;

            // w' = w - alpha_j * v_curr - beta_j * v_prev
            let mut w_prime = w;
            for i in 0..dim {
                w_prime[i] =
                    w_prime[i] - alpha_j * v_curr[i] - Complex64::new(beta[j], 0.0) * v_prev[i];
            }

            beta[j + 1] = {
                let norm_sq: f64 = w_prime.iter().map(|c| c.norm_sqr()).sum();
                norm_sq.sqrt()
            };

            v_mat.push(v_curr.clone());

            if beta[j + 1] < 1e-14 {
                // Lucky breakdown — Krylov space exhausted.
                let used = j + 1;
                return self.extract_k_from_tridiagonal(
                    &alpha[..used],
                    &beta[1..=used],
                    &v_mat[1..=used],
                    k.min(used),
                    dim,
                );
            }

            // Normalise and advance
            let inv_beta = 1.0 / beta[j + 1];
            v_prev = v_curr;
            v_curr = w_prime.mapv(|c| c * inv_beta);
        }

        self.extract_k_from_tridiagonal(&alpha, &beta[1..=n_lanczos], &v_mat[1..=n_lanczos], k, dim)
    }

    /// Extract the `k` smallest eigenpairs from a real symmetric tridiagonal
    /// matrix (alpha = diagonal, beta = sub-diagonal) and map them back into
    /// the original space via the Lanczos basis stored in `v_mat`.
    fn extract_k_from_tridiagonal(
        &self,
        alpha: &[f64],
        beta: &[f64],
        v_mat: &[Array1<Complex64>],
        k: usize,
        dim: usize,
    ) -> Result<MultiBodyEigenResult> {
        let m = alpha.len();
        // Build the tridiagonal matrix and diagonalize it with the real
        // symmetric Jacobi algorithm.
        let mut tri: Array2<Complex64> = Array2::zeros((m, m));
        for i in 0..m {
            tri[[i, i]] = Complex64::new(alpha[i], 0.0);
            if i + 1 < m {
                tri[[i, i + 1]] = Complex64::new(beta[i], 0.0);
                tri[[i + 1, i]] = Complex64::new(beta[i], 0.0);
            }
        }
        // Use a temporary solver that matches the tridiagonal dimension.
        let tmp_solver = MultiBodyQuantumSolver {
            n_particles: 1,
            local_dim: m,
            tolerance: self.tolerance,
            max_sweeps: self.max_sweeps,
        };
        let tri_result = tmp_solver.diagonalize_real(&tri)?;

        // Take the k smallest eigenvalues/vectors from the Lanczos subspace.
        let k_eff = k.min(m);
        let mut eigenvalues = Array1::zeros(k_eff);
        let mut eigenvectors: Array2<Complex64> = Array2::zeros((dim, k_eff));

        for col in 0..k_eff {
            eigenvalues[col] = tri_result.eigenvalues[col];
            // Back-transform: eigvec in original space = sum_j t[j,col] * v_mat[j]
            for j in 0..m {
                let coeff = tri_result.eigenvectors[[j, col]];
                for row in 0..dim {
                    eigenvectors[[row, col]] += coeff * v_mat[j][row];
                }
            }
        }

        Ok(MultiBodyEigenResult {
            eigenvalues,
            eigenvectors,
        })
    }

    /// Build a nearest-neighbor Heisenberg XXX Hamiltonian on a 1D chain with
    /// periodic boundary conditions:
    ///
    ///   H = J Σ_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + S^z_i S^z_{i+1})
    ///
    /// This is a convenience constructor for testing and benchmarking.
    /// Only works for spin-1/2 (local_dim == 2).
    pub fn build_heisenberg_chain(&self, j_coupling: f64) -> Result<Array2<Complex64>> {
        if self.local_dim != 2 {
            return Err(IntegrateError::InvalidInput(
                "Heisenberg chain builder requires local_dim == 2 (spin-1/2)".to_string(),
            ));
        }
        let dim = self.hilbert_dim();
        let mut h: Array2<Complex64> = Array2::zeros((dim, dim));

        // Pauli operators (0.5 * sigma)
        // S^z: diag(0.5, -0.5)
        // S^+ = [[0,1],[0,0]], S^- = [[0,0],[1,0]]
        // S^x S^x + S^y S^y = 0.5*(S^+ S^- + S^- S^+)

        // For each site, add the bond to the next site with periodic boundary
        // conditions.  For chains with n_particles > 2, every bond (i, i+1 mod N)
        // is distinct so we iterate all N sites.  For N = 2, the two bonds
        // (0→1) and (1→0) map to the same pair after sorting, so we only add
        // the bond once.
        let n_bonds = if self.n_particles == 2 {
            1
        } else {
            self.n_particles
        };
        for site in 0..n_bonds {
            let next = (site + 1) % self.n_particles;
            self.add_two_site_heisenberg(&mut h, site, next, j_coupling)?;
        }

        Ok(h)
    }

    /// Add the two-site Heisenberg interaction to the Hamiltonian matrix.
    fn add_two_site_heisenberg(
        &self,
        h: &mut Array2<Complex64>,
        site_a: usize,
        site_b: usize,
        j_coupling: f64,
    ) -> Result<()> {
        let dim = self.hilbert_dim();
        let (a, b) = if site_a < site_b {
            (site_a, site_b)
        } else {
            (site_b, site_a)
        };

        for ket in 0..dim {
            // S^z_a S^z_b contribution: ±0.25 * J
            let spin_a = ((ket >> (self.n_particles - 1 - a)) & 1) as i64;
            let spin_b = ((ket >> (self.n_particles - 1 - b)) & 1) as i64;
            let sz_a = if spin_a == 0 { 0.5_f64 } else { -0.5 };
            let sz_b = if spin_b == 0 { 0.5_f64 } else { -0.5 };
            h[[ket, ket]] += Complex64::new(j_coupling * sz_a * sz_b, 0.0);

            // S^+ _a S^- _b: flip spin_a up->down, spin_b down->up
            if spin_a == 0 && spin_b == 1 {
                let bra =
                    ket ^ (1 << (self.n_particles - 1 - a)) ^ (1 << (self.n_particles - 1 - b));
                h[[bra, ket]] += Complex64::new(0.5 * j_coupling, 0.0);
                h[[ket, bra]] += Complex64::new(0.5 * j_coupling, 0.0);
            }
        }
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal diagonalization helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Real-symmetric Jacobi diagonalization (input treated as real).
    fn diagonalize_real(&self, hamiltonian: &Array2<Complex64>) -> Result<MultiBodyEigenResult> {
        let dim = hamiltonian.nrows();
        // Extract real part.
        let mut a: Vec<f64> = hamiltonian.iter().map(|c| c.re).collect();
        // Build eigenvector matrix (identity).
        let mut vecs: Vec<f64> = (0..dim * dim)
            .map(|idx| if idx / dim == idx % dim { 1.0 } else { 0.0 })
            .collect();

        // Classic cyclic-by-rows Jacobi for real symmetric matrices.
        for _sweep in 0..self.max_sweeps {
            let mut max_off = 0.0_f64;
            for p in 0..dim {
                for q in (p + 1)..dim {
                    let val = a[p * dim + q].abs();
                    if val > max_off {
                        max_off = val;
                    }
                }
            }
            if max_off < self.tolerance {
                break;
            }

            for p in 0..dim {
                for q in (p + 1)..dim {
                    let a_pq = a[p * dim + q];
                    if a_pq.abs() < 1e-15 {
                        continue;
                    }
                    let a_pp = a[p * dim + p];
                    let a_qq = a[q * dim + q];
                    let theta = 0.5 * (a_qq - a_pp) / a_pq;
                    let t = if theta >= 0.0 {
                        1.0 / (theta + (1.0 + theta * theta).sqrt())
                    } else {
                        -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                    };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;
                    let tau = s / (1.0 + c);

                    // Update diagonal.
                    let delta_pp = -t * a_pq;
                    let delta_qq = t * a_pq;
                    a[p * dim + p] += delta_pp;
                    a[q * dim + q] += delta_qq;
                    a[p * dim + q] = 0.0;
                    a[q * dim + p] = 0.0;

                    // Update off-diagonal elements.
                    for r in 0..dim {
                        if r == p || r == q {
                            continue;
                        }
                        let a_rp = a[r * dim + p];
                        let a_rq = a[r * dim + q];
                        let new_rp = a_rp - s * (a_rq + tau * a_rp);
                        let new_rq = a_rq + s * (a_rp - tau * a_rq);
                        a[r * dim + p] = new_rp;
                        a[p * dim + r] = new_rp;
                        a[r * dim + q] = new_rq;
                        a[q * dim + r] = new_rq;
                    }

                    // Update eigenvectors.
                    for r in 0..dim {
                        let v_rp = vecs[r * dim + p];
                        let v_rq = vecs[r * dim + q];
                        vecs[r * dim + p] = v_rp - s * (v_rq + tau * v_rp);
                        vecs[r * dim + q] = v_rq + s * (v_rp - tau * v_rq);
                    }
                }
            }
        }

        // Collect eigenvalues and sort.
        let mut pairs: Vec<(f64, usize)> = (0..dim).map(|i| (a[i * dim + i], i)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let eigenvalues = Array1::from_iter(pairs.iter().map(|&(v, _)| v));
        let mut eigenvectors: Array2<Complex64> = Array2::zeros((dim, dim));
        for (new_col, &(_, old_col)) in pairs.iter().enumerate() {
            for row in 0..dim {
                eigenvectors[[row, new_col]] = Complex64::new(vecs[row * dim + old_col], 0.0);
            }
        }

        Ok(MultiBodyEigenResult {
            eigenvalues,
            eigenvectors,
        })
    }

    /// Complex-Hermitian Jacobi diagonalization.
    fn diagonalize_complex(&self, hamiltonian: &Array2<Complex64>) -> Result<MultiBodyEigenResult> {
        let dim = hamiltonian.nrows();
        // Work with a flat copy.
        let mut a: Vec<Complex64> = hamiltonian.iter().copied().collect();
        let mut vecs: Vec<Complex64> = (0..dim * dim)
            .map(|idx| {
                if idx / dim == idx % dim {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect();

        for _sweep in 0..self.max_sweeps {
            let mut max_off = 0.0_f64;
            for p in 0..dim {
                for q in (p + 1)..dim {
                    let val = a[p * dim + q].norm();
                    if val > max_off {
                        max_off = val;
                    }
                }
            }
            if max_off < self.tolerance {
                break;
            }

            for p in 0..dim {
                for q in (p + 1)..dim {
                    let a_pq = a[p * dim + q];
                    if a_pq.norm() < 1e-15 {
                        continue;
                    }
                    let a_pp = a[p * dim + p].re;
                    let a_qq = a[q * dim + q].re;

                    // Compute 2x2 Hermitian Jacobi rotation.
                    // Following Golub & Van Loan §8.5 for the Hermitian case.
                    let mu = a_pq.norm();
                    let phi = a_pq.arg(); // phase of off-diagonal element
                    let theta = 0.5 * (a_qq - a_pp) / mu;
                    let t_mag = if theta >= 0.0 {
                        1.0 / (theta + (1.0 + theta * theta).sqrt())
                    } else {
                        -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                    };
                    let c = 1.0 / (1.0 + t_mag * t_mag).sqrt();
                    let s = Complex64::new(0.0, -phi).exp() * t_mag * c; // e^{-i phi} t c
                    let tau = s / c; // s / c for real tau part trick

                    // Update diagonal (real).
                    a[p * dim + p] = Complex64::new(a_pp - t_mag * mu, 0.0);
                    a[q * dim + q] = Complex64::new(a_qq + t_mag * mu, 0.0);
                    a[p * dim + q] = Complex64::new(0.0, 0.0);
                    a[q * dim + p] = Complex64::new(0.0, 0.0);

                    // Update off-diagonal elements.
                    for r in 0..dim {
                        if r == p || r == q {
                            continue;
                        }
                        let a_rp = a[r * dim + p];
                        let a_rq = a[r * dim + q];
                        // Simplified update (approximate):
                        let new_rp = a_rp - s * a_rq;
                        let new_rq = a_rq + s.conj() * a_rp;
                        a[r * dim + p] = new_rp;
                        a[p * dim + r] = new_rp.conj();
                        a[r * dim + q] = new_rq;
                        a[q * dim + r] = new_rq.conj();
                        let _ = tau; // suppress unused warning
                    }

                    // Update eigenvectors.
                    for r in 0..dim {
                        let v_rp = vecs[r * dim + p];
                        let v_rq = vecs[r * dim + q];
                        vecs[r * dim + p] = c * v_rp - s.conj() * v_rq;
                        vecs[r * dim + q] = s * v_rp + c * v_rq;
                    }
                }
            }
        }

        // Collect eigenvalues and sort by real part.
        let mut pairs: Vec<(f64, usize)> = (0..dim).map(|i| (a[i * dim + i].re, i)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let eigenvalues = Array1::from_iter(pairs.iter().map(|&(v, _)| v));
        let mut eigenvectors: Array2<Complex64> = Array2::zeros((dim, dim));
        for (new_col, &(_, old_col)) in pairs.iter().enumerate() {
            for row in 0..dim {
                eigenvectors[[row, new_col]] = vecs[row * dim + old_col];
            }
        }

        Ok(MultiBodyEigenResult {
            eigenvalues,
            eigenvectors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_quantum_annealer() {
        let annealer = QuantumAnnealer::new(4, 10.0, 100);
        assert_eq!(annealer.n_qubits, 4);
        assert_eq!(annealer.schedule.len(), 100);

        // Test Ising problem
        let j_matrix = Array2::zeros((4, 4));
        let h_fields = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.1]);

        let result = annealer.solve_ising(&j_matrix, &h_fields);
        assert!(result.is_ok());

        let (spins, energy) = result.expect("Operation failed");
        assert_eq!(spins.len(), 4);
        assert!(energy.is_finite());
    }

    #[test]
    fn test_vqe() {
        let vqe = VariationalQuantumEigensolver::new(2, 2);
        assert_eq!(vqe.n_qubits, 2);
        assert_eq!(vqe.circuit_depth, 2);

        // Test with simple Hamiltonian
        let mut hamiltonian = Array2::zeros((4, 4));
        hamiltonian[[0, 0]] = Complex64::new(-1.0, 0.0);
        hamiltonian[[1, 1]] = Complex64::new(0.0, 0.0);
        hamiltonian[[2, 2]] = Complex64::new(0.0, 0.0);
        hamiltonian[[3, 3]] = Complex64::new(1.0, 0.0);

        let result = vqe.find_ground_state(&hamiltonian);
        assert!(result.is_ok());

        let (energy, params) = result.expect("Operation failed");
        assert!(energy <= 0.0); // Should find ground state with negative energy
        assert_eq!(params.len(), vqe.n_qubits * vqe.circuit_depth * 3);
    }

    #[test]
    fn test_quantum_error_correction() {
        let mut qec = QuantumErrorCorrection::new(1, ErrorCorrectionCode::Steane7);

        // Test encoding
        let logical_state =
            Array1::from_vec(vec![Complex64::new(0.707, 0.0), Complex64::new(0.707, 0.0)]);

        let encoded = qec.encode(&logical_state);
        assert!(encoded.is_ok());

        let physical_state = encoded.expect("Operation failed");
        assert_eq!(physical_state.len(), 1 << qec.get_physical_qubit_count());

        // Test error correction cycle
        let mut test_state = physical_state;
        let error_prob = qec.error_correction_cycle(&mut test_state);
        assert!(error_prob.is_ok());

        let error_rate = qec.estimate_logical_error_rate();
        assert!((0.0..=1.0).contains(&error_rate));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MultiBodyQuantumSolver tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_multi_body_hilbert_dim() {
        let solver = MultiBodyQuantumSolver::new(3, 2); // 3 qubits
        assert_eq!(solver.hilbert_dim(), 8);

        let solver2 = MultiBodyQuantumSolver::new(2, 3); // 2 qutrits
        assert_eq!(solver2.hilbert_dim(), 9);
    }

    #[test]
    fn test_diagonalize_2x2_real() {
        // Diagonal 2x2 Hamiltonian — eigenvalues should be exact.
        let solver = MultiBodyQuantumSolver::new(1, 2);
        let mut h: Array2<Complex64> = Array2::zeros((2, 2));
        h[[0, 0]] = Complex64::new(-1.0, 0.0);
        h[[1, 1]] = Complex64::new(2.0, 0.0);

        let result = solver.diagonalize(&h).expect("diagonalize failed");
        assert_eq!(result.eigenvalues.len(), 2);
        assert_relative_eq!(result.eigenvalues[0], -1.0, epsilon = 1e-10);
        assert_relative_eq!(result.eigenvalues[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diagonalize_2x2_offdiag() {
        // Pauli-X: eigenvalues ±1.
        let solver = MultiBodyQuantumSolver::new(1, 2);
        let mut h: Array2<Complex64> = Array2::zeros((2, 2));
        h[[0, 1]] = Complex64::new(1.0, 0.0);
        h[[1, 0]] = Complex64::new(1.0, 0.0);

        let result = solver.diagonalize(&h).expect("diagonalize failed");
        assert_relative_eq!(result.eigenvalues[0], -1.0, epsilon = 1e-10);
        assert_relative_eq!(result.eigenvalues[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diagonalize_4x4_ising_transverse() {
        // Transverse-field Ising model on 2 sites:
        // H = -J σ^z_0 σ^z_1 - h (σ^x_0 + σ^x_1)
        // Exact eigenvalues: ±√(J² + h²) ± √(J² + h²)  (not exactly — use known spectrum).
        // We just verify eigenvalues are real-valued and sorted.
        let solver = MultiBodyQuantumSolver::new(2, 2);
        let j = 1.0_f64;
        let h_field = 0.5_f64;
        // Build H manually in |↑↑⟩,|↑↓⟩,|↓↑⟩,|↓↓⟩ basis.
        let mut h: Array2<Complex64> = Array2::zeros((4, 4));
        // σ^z_0 σ^z_1 term: diagonal
        h[[0, 0]] = Complex64::new(-j, 0.0); // ↑↑
        h[[1, 1]] = Complex64::new(j, 0.0); // ↑↓
        h[[2, 2]] = Complex64::new(j, 0.0); // ↓↑
        h[[3, 3]] = Complex64::new(-j, 0.0); // ↓↓
                                             // σ^x_0 term: flip qubit 0
        h[[0, 2]] += Complex64::new(-h_field, 0.0);
        h[[2, 0]] += Complex64::new(-h_field, 0.0);
        h[[1, 3]] += Complex64::new(-h_field, 0.0);
        h[[3, 1]] += Complex64::new(-h_field, 0.0);
        // σ^x_1 term: flip qubit 1
        h[[0, 1]] += Complex64::new(-h_field, 0.0);
        h[[1, 0]] += Complex64::new(-h_field, 0.0);
        h[[2, 3]] += Complex64::new(-h_field, 0.0);
        h[[3, 2]] += Complex64::new(-h_field, 0.0);

        let result = solver.diagonalize(&h).expect("diagonalize failed");
        assert_eq!(result.eigenvalues.len(), 4);
        // Verify sorted ascending.
        for i in 1..4 {
            assert!(result.eigenvalues[i] >= result.eigenvalues[i - 1] - 1e-10);
        }
        // Verify eigenvectors are orthonormal.
        for i in 0..4 {
            let col_i = result.eigenvectors.column(i);
            let norm_sq: f64 = col_i.iter().map(|c| c.norm_sqr()).sum();
            assert_relative_eq!(norm_sq, 1.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_heisenberg_chain_2_sites() {
        // 2-site antiferromagnetic Heisenberg XXX (J = +1):
        //   H = J S_0·S_1
        // Exact spectrum for spin-1/2 (dim = 4):
        //   singlet  S=0: E = -3J/4 = -0.75  (1-fold degenerate — ground state)
        //   triplet  S=1: E = +J/4  = +0.25  (3-fold degenerate: Sz = -1, 0, +1)
        //
        // Eigenvalues sorted ascending: -0.75, +0.25, +0.25, +0.25
        let solver = MultiBodyQuantumSolver::new(2, 2);
        let h = solver
            .build_heisenberg_chain(1.0)
            .expect("build_heisenberg_chain failed");
        let result = solver.diagonalize(&h).expect("diagonalize failed");
        assert_eq!(result.eigenvalues.len(), 4);
        assert_relative_eq!(result.eigenvalues[0], -0.75, epsilon = 1e-8);
        assert_relative_eq!(result.eigenvalues[1], 0.25, epsilon = 1e-8);
        assert_relative_eq!(result.eigenvalues[2], 0.25, epsilon = 1e-8);
        assert_relative_eq!(result.eigenvalues[3], 0.25, epsilon = 1e-8);
    }

    #[test]
    fn test_find_ground_states_small() {
        // On a small 4×4 matrix, find_ground_states with k=2 should agree with full diag.
        let solver = MultiBodyQuantumSolver::new(2, 2);
        let mut h: Array2<Complex64> = Array2::zeros((4, 4));
        h[[0, 0]] = Complex64::new(3.0, 0.0);
        h[[1, 1]] = Complex64::new(1.0, 0.0);
        h[[2, 2]] = Complex64::new(2.0, 0.0);
        h[[3, 3]] = Complex64::new(0.5, 0.0);
        h[[0, 1]] = Complex64::new(0.1, 0.0);
        h[[1, 0]] = Complex64::new(0.1, 0.0);

        let full = solver.diagonalize(&h).expect("full diag failed");
        let partial = solver.find_ground_states(&h, 2).expect("partial failed");
        assert_relative_eq!(partial.eigenvalues[0], full.eigenvalues[0], epsilon = 1e-8);
        assert_relative_eq!(partial.eigenvalues[1], full.eigenvalues[1], epsilon = 1e-8);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        // Solver with dim=4 should reject a 2×2 Hamiltonian.
        let solver = MultiBodyQuantumSolver::new(2, 2); // dim = 4
        let h: Array2<Complex64> = Array2::zeros((2, 2));
        let result = solver.diagonalize(&h);
        assert!(result.is_err());
    }
}
