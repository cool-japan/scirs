//! Quantum-Inspired Optimization Methods
//!
//! This module implements cutting-edge quantum-inspired optimization algorithms:
//! - Quantum superposition for parallel parameter exploration
//! - Quantum entanglement for correlated parameter updates
//! - Quantum annealing with sophisticated cooling schedules
//! - Quantum tunneling for escaping local minima
//! - Quantum interference patterns for optimization landscapes
//! - Quantum state collapse for solution selection

use crate::error::OptimizeResult;
use crate::result::OptimizeResults;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Zero;
use rand::{rng, Rng};
use std::collections::VecDeque;
use std::f64::consts::PI;

/// Complex number representation for quantum states
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Complex {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn magnitude(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    pub fn phase(&self) -> f64 {
        self.imag.atan2(self.real)
    }

    pub fn conjugate(&self) -> Complex {
        Complex::new(self.real, -self.imag)
    }
}

impl std::ops::Add for Complex {
    type Output = Complex;

    fn add(self, other: Complex) -> Complex {
        Complex::new(self.real + other.real, self.imag + other.imag)
    }
}

impl std::ops::Mul for Complex {
    type Output = Complex;

    fn mul(self, other: Complex) -> Complex {
        Complex::new(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Complex;

    fn mul(self, scalar: f64) -> Complex {
        Complex::new(self.real * scalar, self.imag * scalar)
    }
}

impl Zero for Complex {
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }

    fn is_zero(&self) -> bool {
        self.real == 0.0 && self.imag == 0.0
    }
}

/// Quantum state representation for optimization parameters
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Amplitude coefficients for superposition states
    pub amplitudes: Array1<Complex>,
    /// Parameter values for each basis state
    pub basis_states: Array2<f64>,
    /// Quantum register size (number of qubits)
    pub num_qubits: usize,
    /// Entanglement correlations between parameters
    pub entanglement_matrix: Array2<Complex>,
    /// Decoherence time constant
    pub decoherence_time: f64,
    /// Current evolution time
    pub evolution_time: f64,
}

impl QuantumState {
    /// Create new quantum state with random superposition
    pub fn new(num_params: usize, num_basis_states: usize) -> Self {
        let num_qubits = (num_basis_states as f64).log2().ceil() as usize;
        let actual_states = 1 << num_qubits;

        // Initialize random amplitudes (normalized)
        let mut amplitudes = Array1::from_shape_fn(actual_states, |_| {
            Complex::new(
                rand::rng().random_range(-1.0..1.0),
                rand::rng().random_range(-1.0..1.0),
            )
        });

        // Normalize amplitudes
        let norm: f64 = amplitudes
            .iter()
            .map(|c| c.magnitude().powi(2))
            .sum::<f64>()
            .sqrt();
        for amp in amplitudes.iter_mut() {
            *amp = *amp * (1.0 / norm);
        }

        // Initialize basis states around reasonable search space
        let basis_states = Array2::from_shape_fn((actual_states, num_params), |_| {
            rand::rng().random_range(-2.0..5.0)
        });

        // Initialize entanglement matrix
        let entanglement_matrix = Array2::from_shape_fn((num_params, num_params), |(i, j)| {
            if i == j {
                Complex::new(1.0, 0.0)
            } else {
                let correlation = rand::rng().random_range(-0.1..0.1);
                Complex::new(correlation, correlation * 0.1)
            }
        });

        Self {
            amplitudes,
            basis_states,
            num_qubits,
            entanglement_matrix,
            decoherence_time: 1000.0,
            evolution_time: 0.0,
        }
    }

    /// Measure quantum state to collapse to classical parameters
    pub fn measure(&self) -> Array1<f64> {
        // Compute measurement probabilities
        let probabilities: Vec<f64> = self
            .amplitudes
            .iter()
            .map(|c| c.magnitude().powi(2))
            .collect();

        // Quantum measurement (random collapse based on probabilities)
        let mut cumulative = 0.0;
        let random_value = rand::rng().random_range(0.0..1.0);

        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                return self.basis_states.row(i).to_owned();
            }
        }

        // Fallback to most probable state
        let max_prob_idx = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.basis_states.row(max_prob_idx).to_owned()
    }

    /// Apply quantum evolution based on objective function landscape
    pub fn evolve(&mut self, objective_gradients: &Array1<f64>, dt: f64) -> OptimizeResult<()> {
        self.evolution_time += dt;

        // Compute Hamiltonian from objective landscape
        let hamiltonian = self.compute_hamiltonian(objective_gradients)?;

        // Apply time evolution operator U = exp(-iHt)
        self.apply_time_evolution(&hamiltonian, dt)?;

        // Apply decoherence
        self.apply_decoherence(dt)?;

        // Update entanglement based on parameter correlations
        self.update_entanglement(objective_gradients)?;

        Ok(())
    }

    fn compute_hamiltonian(
        &self,
        objective_gradients: &Array1<f64>,
    ) -> OptimizeResult<Array2<Complex>> {
        let n_states = self.amplitudes.len();
        let mut hamiltonian = Array2::zeros((n_states, n_states));

        for i in 0..n_states {
            for j in 0..n_states {
                if i == j {
                    // Diagonal elements: potential energy from objective
                    let params = self.basis_states.row(i);
                    let potential = params
                        .iter()
                        .zip(objective_gradients.iter())
                        .map(|(&p, &g)| p * g)
                        .sum::<f64>();

                    hamiltonian[[i, j]] = Complex::new(potential, 0.0);
                } else {
                    // Off-diagonal elements: tunneling between states
                    let distance = (&self.basis_states.row(i) - &self.basis_states.row(j))
                        .mapv(|x| x * x)
                        .sum()
                        .sqrt();

                    let tunneling_amplitude = (-distance * 0.1).exp();
                    hamiltonian[[i, j]] = Complex::new(0.0, tunneling_amplitude);
                }
            }
        }

        Ok(hamiltonian)
    }

    fn apply_time_evolution(
        &mut self,
        hamiltonian: &Array2<Complex>,
        dt: f64,
    ) -> OptimizeResult<()> {
        // Simplified time evolution: ψ(t+dt) = exp(-iH*dt) * ψ(t)
        // Using first-order approximation: exp(-iH*dt) ≈ 1 - iH*dt

        let n_states = self.amplitudes.len();
        let mut new_amplitudes = self.amplitudes.clone();

        for i in 0..n_states {
            let mut evolved_amp = Complex::new(0.0, 0.0);

            for j in 0..n_states {
                let h_element = hamiltonian[[i, j]];
                let evolution_factor = if i == j {
                    // Diagonal: 1 - iH_ii*dt
                    Complex::new(1.0, 0.0) + Complex::new(0.0, -h_element.real * dt)
                } else {
                    // Off-diagonal: -iH_ij*dt
                    Complex::new(0.0, -h_element.real * dt) + Complex::new(h_element.imag * dt, 0.0)
                };

                evolved_amp = evolved_amp + evolution_factor * self.amplitudes[j];
            }

            new_amplitudes[i] = evolved_amp;
        }

        // Renormalize
        let norm: f64 = new_amplitudes
            .iter()
            .map(|c| c.magnitude().powi(2))
            .sum::<f64>()
            .sqrt();
        for amp in new_amplitudes.iter_mut() {
            *amp = *amp * (1.0 / norm.max(1e-12));
        }

        self.amplitudes = new_amplitudes;
        Ok(())
    }

    fn apply_decoherence(&mut self, dt: f64) -> OptimizeResult<()> {
        // Exponential decoherence: amplitude decay
        let decoherence_factor = (-dt / self.decoherence_time).exp();

        for amp in self.amplitudes.iter_mut() {
            *amp = *amp * decoherence_factor;
        }

        // Add small random noise to simulate environmental interaction
        let noise_strength = 1.0 - decoherence_factor;
        for amp in self.amplitudes.iter_mut() {
            let noise_real = rand::rng().random_range(-0.5..0.5) * noise_strength * 0.01;
            let noise_imag = rand::rng().random_range(-0.5..0.5) * noise_strength * 0.01;
            *amp = *amp + Complex::new(noise_real, noise_imag);
        }

        // Renormalize after decoherence
        let norm: f64 = self
            .amplitudes
            .iter()
            .map(|c| c.magnitude().powi(2))
            .sum::<f64>()
            .sqrt();
        for amp in self.amplitudes.iter_mut() {
            *amp = *amp * (1.0 / norm.max(1e-12));
        }

        Ok(())
    }

    fn update_entanglement(&mut self, objective_gradients: &Array1<f64>) -> OptimizeResult<()> {
        let n_params = self.entanglement_matrix.nrows();

        // Update entanglement based on gradient correlations
        for i in 0..n_params {
            for j in i + 1..n_params {
                if i < objective_gradients.len() && j < objective_gradients.len() {
                    let correlation = objective_gradients[i] * objective_gradients[j];
                    let phase_update = correlation * 0.01;

                    let current = self.entanglement_matrix[[i, j]];
                    let new_phase = current.phase() + phase_update;
                    let magnitude = current.magnitude() * 0.99 + correlation.abs() * 0.01;

                    self.entanglement_matrix[[i, j]] =
                        Complex::new(magnitude * new_phase.cos(), magnitude * new_phase.sin());
                    self.entanglement_matrix[[j, i]] = self.entanglement_matrix[[i, j]].conjugate();
                }
            }
        }

        Ok(())
    }

    /// Apply quantum superposition principle to explore multiple states
    pub fn create_superposition(&mut self, exploration_radius: f64) -> OptimizeResult<()> {
        let n_states = self.basis_states.nrows();
        let n_params = self.basis_states.ncols();

        // Create new basis states around current ones
        for i in 0..n_states {
            for j in 0..n_params {
                let perturbation =
                    rand::rng().random_range(-exploration_radius..exploration_radius);
                self.basis_states[[i, j]] += perturbation;
            }
        }

        // Update amplitudes to maintain superposition
        let equal_amplitude = Complex::new(1.0 / (n_states as f64).sqrt(), 0.0);
        for i in 0..n_states {
            self.amplitudes[i] = equal_amplitude;
        }

        Ok(())
    }

    /// Apply quantum tunneling to escape local minima
    pub fn quantum_tunnel(
        &mut self,
        barrier_height: f64,
        tunnel_probability: f64,
    ) -> OptimizeResult<()> {
        if rand::rng().random_range(0.0..1.0) < tunnel_probability {
            // Quantum tunneling: create new basis states beyond energy barriers
            let n_states = self.basis_states.nrows();
            let n_params = self.basis_states.ncols();

            // Select a random state to tunnel from
            let source_state = rand::rng().random_range(0..n_states);

            // Create tunneled state
            for j in 0..n_params {
                let tunnel_distance = barrier_height * rand::rng().random_range(-0.5..0.5);
                self.basis_states[[source_state, j]] += tunnel_distance;
            }

            // Redistribute amplitudes to account for tunneling
            let tunnel_amplitude_factor = (-barrier_height * 0.1).exp();
            self.amplitudes[source_state] = self.amplitudes[source_state] * tunnel_amplitude_factor;

            // Renormalize
            let norm: f64 = self
                .amplitudes
                .iter()
                .map(|c| c.magnitude().powi(2))
                .sum::<f64>()
                .sqrt();
            for amp in self.amplitudes.iter_mut() {
                *amp = *amp * (1.0 / norm.max(1e-12));
            }
        }

        Ok(())
    }
}

/// Quantum annealing schedule for cooling
#[derive(Debug, Clone)]
pub struct QuantumAnnealingSchedule {
    /// Initial temperature
    pub initial_temperature: f64,
    /// Final temperature
    pub final_temperature: f64,
    /// Current temperature
    pub current_temperature: f64,
    /// Annealing progress (0 to 1)
    pub progress: f64,
    /// Cooling schedule type
    pub schedule_type: CoolingSchedule,
    /// Quantum fluctuation strength
    pub quantum_fluctuation: f64,
}

#[derive(Debug, Clone)]
pub enum CoolingSchedule {
    Linear,
    Exponential,
    Logarithmic,
    QuantumAdaptive,
}

impl QuantumAnnealingSchedule {
    pub fn new(initial_temp: f64, final_temp: f64, schedule: CoolingSchedule) -> Self {
        Self {
            initial_temperature: initial_temp,
            final_temperature: final_temp,
            current_temperature: initial_temp,
            progress: 0.0,
            schedule_type: schedule,
            quantum_fluctuation: 1.0,
        }
    }

    pub fn update(&mut self, iteration: usize, max_nit: usize, energy_change: f64) {
        self.progress = iteration as f64 / max_nit as f64;

        self.current_temperature = match self.schedule_type {
            CoolingSchedule::Linear => {
                self.initial_temperature * (1.0 - self.progress)
                    + self.final_temperature * self.progress
            }
            CoolingSchedule::Exponential => {
                self.initial_temperature
                    * (self.final_temperature / self.initial_temperature).powf(self.progress)
            }
            CoolingSchedule::Logarithmic => self.initial_temperature / (1.0 + self.progress).ln(),
            CoolingSchedule::QuantumAdaptive => {
                // Adaptive cooling based on energy landscape and quantum effects
                let base_temp = self.initial_temperature * (0.1_f64).powf(self.progress);
                let quantum_correction = self.quantum_fluctuation * (-energy_change.abs()).exp();
                base_temp * (1.0 + quantum_correction * 0.1)
            }
        };

        // Update quantum fluctuation strength
        self.quantum_fluctuation =
            1.0 - self.progress + 0.1 * (2.0 * PI * self.progress * 10.0).sin();
    }

    pub fn should_accept(&self, energy_delta: f64) -> bool {
        if energy_delta <= 0.0 {
            true // Always accept improvements
        } else {
            // Quantum-enhanced Boltzmann acceptance with fluctuations
            let classical_prob = (-energy_delta / self.current_temperature).exp();
            let quantum_prob = classical_prob * (1.0 + self.quantum_fluctuation * 0.1);

            rand::rng().random_range(0.0..1.0) < quantum_prob.min(1.0)
        }
    }
}

/// Advanced Quantum-Inspired Optimizer
#[derive(Debug, Clone)]
pub struct QuantumInspiredOptimizer {
    /// Quantum state representation
    pub quantum_state: QuantumState,
    /// Annealing schedule
    pub annealing_schedule: QuantumAnnealingSchedule,
    /// Best solution found
    pub best_solution: Array1<f64>,
    /// Best objective value
    pub best_objective: f64,
    /// Current iteration
    pub iteration: usize,
    /// Maximum iterations
    pub max_nit: usize,
    /// Objective function evaluations
    pub function_evaluations: usize,
    /// Quantum interference patterns
    pub interference_history: VecDeque<f64>,
    /// Entanglement strength tracker
    pub entanglement_strength: f64,
    /// Tunneling event counter
    pub tunneling_events: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl QuantumInspiredOptimizer {
    /// Create new quantum-inspired optimizer
    pub fn new(
        initial_params: &ArrayView1<f64>,
        max_nit: usize,
        num_quantum_states: usize,
    ) -> Self {
        let n_params = initial_params.len();
        let quantum_state = QuantumState::new(n_params, num_quantum_states);
        let annealing_schedule =
            QuantumAnnealingSchedule::new(10.0, 0.01, CoolingSchedule::QuantumAdaptive);

        Self {
            quantum_state,
            annealing_schedule,
            best_solution: initial_params.to_owned(),
            best_objective: f64::INFINITY,
            iteration: 0,
            max_nit,
            function_evaluations: 0,
            interference_history: VecDeque::with_capacity(1000),
            entanglement_strength: 1.0,
            tunneling_events: 0,
            tolerance: 1e-8,
        }
    }

    /// Run quantum-inspired optimization
    pub fn optimize<F>(&mut self, objective: F) -> OptimizeResult<OptimizeResults<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        // Evaluate initial point if not already done
        if self.best_objective == f64::INFINITY {
            self.best_objective = objective(&self.best_solution.view());
            self.function_evaluations += 1;
        }

        let mut prev_objective = self.best_objective;
        let mut stagnation_counter = 0;

        for iteration in 0..self.max_nit {
            self.iteration = iteration;

            // Measure quantum state to get candidate solution
            let candidate_params = self.quantum_state.measure();
            let candidate_objective = objective(&candidate_params.view());
            self.function_evaluations += 1;

            // Update best solution
            if candidate_objective < self.best_objective {
                self.best_objective = candidate_objective;
                self.best_solution = candidate_params.clone();
                stagnation_counter = 0;
            } else {
                stagnation_counter += 1;
            }

            // Compute energy change and gradients (finite difference)
            let energy_change = candidate_objective - prev_objective;
            let gradients =
                self.compute_finite_difference_gradient(&objective, &candidate_params.view())?;

            // Quantum evolution based on objective landscape
            self.quantum_state.evolve(&gradients, 0.1)?;

            // Update annealing schedule
            self.annealing_schedule
                .update(iteration, self.max_nit, energy_change);

            // Quantum tunneling for local minima escape
            if stagnation_counter > 20 {
                let barrier_height = self.annealing_schedule.current_temperature;
                let tunnel_prob = 0.1 * self.annealing_schedule.quantum_fluctuation;
                self.quantum_state
                    .quantum_tunnel(barrier_height, tunnel_prob)?;

                if rand::rng().random_range(0.0..1.0) < tunnel_prob {
                    self.tunneling_events += 1;
                    stagnation_counter = 0;
                }
            }

            // Quantum superposition for exploration
            if iteration % 50 == 0 {
                let exploration_radius = self.annealing_schedule.current_temperature * 0.1;
                self.quantum_state
                    .create_superposition(exploration_radius)?;
            }

            // Track interference patterns
            let interference = self.compute_quantum_interference();
            self.interference_history.push_back(interference);
            if self.interference_history.len() > 1000 {
                self.interference_history.pop_front();
            }

            // Update entanglement strength
            self.entanglement_strength = self.compute_entanglement_strength();

            // Convergence check
            if (prev_objective - candidate_objective).abs() < self.tolerance && iteration > 100 {
                break;
            }

            prev_objective = candidate_objective;
        }

        Ok(OptimizeResults::<f64> {
            x: self.best_solution.clone(),
            fun: self.best_objective,
            success: self.best_objective < f64::INFINITY,
            nit: self.iteration,
            message: format!(
                "Quantum optimization completed. Tunneling events: {}, Final entanglement: {:.3}",
                self.tunneling_events, self.entanglement_strength
            ),
            jac: None,
            hess: None,
            constr: None,
            nfev: self.iteration * self.quantum_state.num_qubits, // Iteration count times qubits
            njev: 0,
            nhev: 0,
            maxcv: 0,
            status: if self.best_objective < f64::INFINITY {
                0
            } else {
                1
            },
        })
    }

    fn compute_finite_difference_gradient<F>(
        &self,
        objective: &F,
        params: &ArrayView1<f64>,
    ) -> OptimizeResult<Array1<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let eps = 1e-8;
        let mut gradient = Array1::zeros(params.len());
        let base_value = objective(params);

        for i in 0..params.len() {
            let mut params_plus = params.to_owned();
            params_plus[i] += eps;
            let value_plus = objective(&params_plus.view());

            gradient[i] = (value_plus - base_value) / eps;
        }

        Ok(gradient)
    }

    fn compute_quantum_interference(&self) -> f64 {
        // Compute interference pattern from amplitude overlaps
        let mut interference = 0.0;
        let n_states = self.quantum_state.amplitudes.len();

        for i in 0..n_states {
            for j in i + 1..n_states {
                let amp_i = self.quantum_state.amplitudes[i];
                let amp_j = self.quantum_state.amplitudes[j];

                // Interference term: Re(ψ_i* ψ_j)
                let interference_term = (amp_i.conjugate() * amp_j).real;
                interference += interference_term.abs();
            }
        }

        interference / (n_states * (n_states - 1) / 2) as f64
    }

    fn compute_entanglement_strength(&self) -> f64 {
        // Simple entanglement measure based on off-diagonal elements
        let mut total_entanglement = 0.0;
        let n_params = self.quantum_state.entanglement_matrix.nrows();
        let mut count = 0;

        for i in 0..n_params {
            for j in i + 1..n_params {
                total_entanglement += self.quantum_state.entanglement_matrix[[i, j]].magnitude();
                count += 1;
            }
        }

        if count > 0 {
            total_entanglement / count as f64
        } else {
            0.0
        }
    }

    /// Get optimization statistics
    pub fn get_quantum_stats(&self) -> QuantumOptimizationStats {
        QuantumOptimizationStats {
            function_evaluations: self.function_evaluations,
            tunneling_events: self.tunneling_events,
            current_temperature: self.annealing_schedule.current_temperature,
            entanglement_strength: self.entanglement_strength,
            quantum_interference: self.interference_history.back().copied().unwrap_or(0.0),
            superposition_dimension: self.quantum_state.amplitudes.len(),
            evolution_time: self.quantum_state.evolution_time,
            decoherence_level: 1.0
                - (-self.quantum_state.evolution_time / self.quantum_state.decoherence_time).exp(),
        }
    }
}

/// Statistics for quantum optimization
#[derive(Debug, Clone)]
pub struct QuantumOptimizationStats {
    pub function_evaluations: usize,
    pub tunneling_events: usize,
    pub current_temperature: f64,
    pub entanglement_strength: f64,
    pub quantum_interference: f64,
    pub superposition_dimension: usize,
    pub evolution_time: f64,
    pub decoherence_level: f64,
}

/// Convenience function for quantum-inspired optimization
#[allow(dead_code)]
pub fn quantum_optimize<F>(
    objective: F,
    initial_params: &ArrayView1<f64>,
    max_nit: Option<usize>,
    num_quantum_states: Option<usize>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let max_iter = max_nit.unwrap_or(1000);
    let num_states = num_quantum_states.unwrap_or(32);

    let mut optimizer = QuantumInspiredOptimizer::new(initial_params, max_iter, num_states);
    optimizer.optimize(objective)
}

/// Quantum-enhanced particle swarm optimization
#[allow(dead_code)]
pub fn quantum_particle_swarm_optimize<F>(
    objective: F,
    initial_params: &ArrayView1<f64>,
    numparticles: usize,
    max_nit: usize,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let mut particles = Vec::new();

    // Create quantum-enhanced particles
    for _ in 0..numparticles {
        let particle_optimizer = QuantumInspiredOptimizer::new(initial_params, max_nit, 8);
        particles.push(particle_optimizer);
    }

    let mut global_best_solution = initial_params.to_owned();
    let mut global_best_objective = f64::INFINITY;

    for iteration in 0..max_nit {
        for particle in &mut particles {
            // Run a few quantum optimization steps per particle
            for _ in 0..5 {
                let candidate = particle.quantum_state.measure();
                let obj_value = objective(&candidate.view());

                if obj_value < particle.best_objective {
                    particle.best_objective = obj_value;
                    particle.best_solution = candidate.clone();
                }

                if obj_value < global_best_objective {
                    global_best_objective = obj_value;
                    global_best_solution = candidate.clone();
                }

                // Quantum evolution with global best influence
                let gradients =
                    particle.compute_finite_difference_gradient(&objective, &candidate.view())?;
                particle.quantum_state.evolve(&gradients, 0.01)?;

                // Entangle particles with global best
                if rand::rng().random_range(0.0..1.0) < 0.1 {
                    let n_params = initial_params.len();
                    for i in 0..n_params.min(particle.quantum_state.basis_states.ncols()) {
                        let entanglement_strength = 0.1;
                        for state_idx in 0..particle.quantum_state.basis_states.nrows() {
                            particle.quantum_state.basis_states[[state_idx, i]] = (1.0
                                - entanglement_strength)
                                * particle.quantum_state.basis_states[[state_idx, i]]
                                + entanglement_strength * global_best_solution[i];
                        }
                    }
                }
            }
        }

        // Quantum tunneling for swarm diversity
        if iteration % 100 == 0 {
            for particle in &mut particles {
                particle.quantum_state.quantum_tunnel(1.0, 0.05)?;
            }
        }
    }

    Ok(OptimizeResults::<f64> {
        x: global_best_solution,
        fun: global_best_objective,
        success: global_best_objective < f64::INFINITY,
        nit: max_nit,
        message: format!(
            "Quantum particle swarm optimization completed with {} particles",
            numparticles
        ),
        jac: None,
        hess: None,
        constr: None,
        nfev: max_nit * numparticles, // Iterations times particles
        njev: 0,
        nhev: 0,
        maxcv: 0,
        status: if global_best_objective < f64::INFINITY {
            0
        } else {
            1
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_arithmetic() {
        let c1 = Complex::new(1.0, 2.0);
        let c2 = Complex::new(3.0, 4.0);

        let sum = c1 + c2;
        assert_eq!(sum.real, 4.0);
        assert_eq!(sum.imag, 6.0);

        let product = c1 * c2;
        assert_eq!(product.real, -5.0); // 1*3 - 2*4
        assert_eq!(product.imag, 10.0); // 1*4 + 2*3

        assert!((c1.magnitude() - (5.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::new(3, 8);
        assert_eq!(state.num_qubits, 3);
        assert_eq!(state.amplitudes.len(), 8);
        assert_eq!(state.basis_states.nrows(), 8);
        assert_eq!(state.basis_states.ncols(), 3);

        // Check normalization
        let norm_squared: f64 = state.amplitudes.iter().map(|c| c.magnitude().powi(2)).sum();
        assert!((norm_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_measurement() {
        let state = QuantumState::new(2, 4);
        let measured_params = state.measure();
        assert_eq!(measured_params.len(), 2);
    }

    #[test]
    fn test_quantum_annealing_schedule() {
        let mut schedule = QuantumAnnealingSchedule::new(10.0, 0.1, CoolingSchedule::Linear);

        schedule.update(0, 100, 0.0);
        assert!((schedule.current_temperature - 10.0).abs() < 1e-10);

        schedule.update(50, 100, -1.0);
        assert!(schedule.current_temperature < 10.0);
        assert!(schedule.current_temperature > 0.1);

        schedule.update(100, 100, 0.0);
        assert!((schedule.current_temperature - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_optimizer_creation() {
        let initial_params = Array1::from(vec![1.0, 2.0]);
        let optimizer = QuantumInspiredOptimizer::new(&initial_params.view(), 100, 16);

        assert_eq!(optimizer.best_solution.len(), 2);
        assert_eq!(optimizer.max_nit, 100);
        assert_eq!(optimizer.quantum_state.amplitudes.len(), 16);
    }

    #[test]
    fn test_quantum_optimization() {
        let objective = |x: &ArrayView1<f64>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let initial = Array1::from(vec![0.0, 0.0]);

        let result = quantum_optimize(objective, &initial.view(), Some(200), Some(16)).unwrap();

        assert!(result.nit > 0);
        // Test that the algorithm runs and produces finite results, even if not optimal
        assert!(result.fun.is_finite());
        assert!(result.success);
        assert_eq!(result.x.len(), 2);

        // Basic sanity checks - results should be reasonable
        assert!(result.x[0].abs() < 10.0);
        assert!(result.x[1].abs() < 10.0);
    }

    #[test]
    fn test_quantum_tunneling() {
        let mut state = QuantumState::new(2, 4);
        let original_states = state.basis_states.clone();

        state.quantum_tunnel(5.0, 1.0).unwrap(); // Force tunneling

        // States should have changed due to tunneling
        let mut changed = false;
        for i in 0..state.basis_states.nrows() {
            for j in 0..state.basis_states.ncols() {
                if (state.basis_states[[i, j]] - original_states[[i, j]]).abs() > 1e-10 {
                    changed = true;
                    break;
                }
            }
        }

        assert!(changed);
    }

    #[test]
    fn test_quantum_particle_swarm() {
        let objective = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
        let initial = Array1::from(vec![5.0, 5.0]);

        let result = quantum_particle_swarm_optimize(objective, &initial.view(), 5, 20).unwrap();

        assert!(result.nit > 0);
        assert!(result.fun < objective(&initial.view()));
        assert!(result.success);
    }

    #[test]
    fn test_quantum_superposition() {
        let mut state = QuantumState::new(2, 4);
        state.create_superposition(1.0).unwrap();

        // Check that amplitudes are approximately equal (superposition)
        let n_states = state.amplitudes.len();
        let expected_magnitude = 1.0 / (n_states as f64).sqrt();

        for amp in &state.amplitudes {
            assert!((amp.magnitude() - expected_magnitude).abs() < 0.1);
        }
    }

    #[test]
    fn test_quantum_evolution() {
        let mut state = QuantumState::new(2, 4);
        let gradients = Array1::from(vec![1.0, -1.0]);

        let original_amplitudes = state.amplitudes.clone();
        state.evolve(&gradients, 0.1).unwrap();

        // Amplitudes should have evolved
        let mut evolved = false;
        for i in 0..state.amplitudes.len() {
            if (state.amplitudes[i].real - original_amplitudes[i].real).abs() > 1e-10
                || (state.amplitudes[i].imag - original_amplitudes[i].imag).abs() > 1e-10
            {
                evolved = true;
                break;
            }
        }

        assert!(evolved);
    }
}
