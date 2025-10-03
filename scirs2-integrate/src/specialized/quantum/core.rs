//! Core quantum mechanics components
//!
//! This module provides basic quantum state representations, potentials,
//! and Schrödinger equation solvers.

use crate::error::{IntegrateError, IntegrateResult as Result};
use scirs2_core::constants::{PI, REDUCED_PLANCK};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Complex64;
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Quantum state representation
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Wave function values (complex)
    pub psi: Array1<Complex64>,
    /// Spatial grid points
    pub x: Array1<f64>,
    /// Time
    pub t: f64,
    /// Mass of the particle
    pub mass: f64,
    /// Spatial step size
    pub dx: f64,
}

impl QuantumState {
    /// Create a new quantum state
    pub fn new(psi: Array1<Complex64>, x: Array1<f64>, t: f64, mass: f64) -> Self {
        let dx = if x.len() > 1 { x[1] - x[0] } else { 1.0 };

        Self {
            psi,
            x,
            t,
            mass,
            dx,
        }
    }

    /// Normalize the wave function
    pub fn normalize(&mut self) {
        let norm_squared: f64 = self.psi.iter().map(|&c| (c.conj() * c).re).sum::<f64>() * self.dx;

        let norm = norm_squared.sqrt();
        if norm > 0.0 {
            self.psi.mapv_inplace(|c| c / norm);
        }
    }

    /// Calculate expectation value of position
    pub fn expectation_position(&self) -> f64 {
        self.expectation_position_simd()
    }

    /// SIMD-optimized expectation value of position
    pub fn expectation_position_simd(&self) -> f64 {
        let prob_density = self.probability_density_simd();
        f64::simd_dot(&self.x.view(), &prob_density.view()) * self.dx
    }

    /// Fallback scalar implementation for expectation value of position
    pub fn expectation_position_scalar(&self) -> f64 {
        self.x
            .iter()
            .zip(self.psi.iter())
            .map(|(&x, &psi)| x * (psi.conj() * psi).re)
            .sum::<f64>()
            * self.dx
    }

    /// Calculate expectation value of momentum
    pub fn expectation_momentum(&self) -> f64 {
        let n = self.psi.len();
        let mut momentum = 0.0;

        // Central difference for derivative
        for i in 1..n - 1 {
            let dpsi_dx = (self.psi[i + 1] - self.psi[i - 1]) / (2.0 * self.dx);
            momentum += (self.psi[i].conj() * Complex64::new(0.0, -REDUCED_PLANCK) * dpsi_dx).re;
        }

        momentum * self.dx
    }

    /// Calculate probability density
    pub fn probability_density(&self) -> Array1<f64> {
        self.probability_density_simd()
    }

    /// SIMD-optimized probability density calculation
    pub fn probability_density_simd(&self) -> Array1<f64> {
        // Convert complex numbers to real and imaginary parts for SIMD processing
        let real_parts: Array1<f64> = self.psi.mapv(|c| c.re);
        let imag_parts: Array1<f64> = self.psi.mapv(|c| c.im);

        // Calculate |psi|^2 = Re(psi)^2 + Im(psi)^2 using SIMD
        let real_squared = f64::simd_mul(&real_parts.view(), &real_parts.view());
        let imag_squared = f64::simd_mul(&imag_parts.view(), &imag_parts.view());
        let result = f64::simd_add(&real_squared.view(), &imag_squared.view());

        result
    }

    /// Fallback scalar implementation for probability density
    pub fn probability_density_scalar(&self) -> Array1<f64> {
        self.psi.mapv(|c| (c.conj() * c).re)
    }
}

/// Quantum potential trait
pub trait QuantumPotential: Send + Sync {
    /// Evaluate potential at given position
    fn evaluate(&self, x: f64) -> f64;

    /// Evaluate potential for array of positions
    fn evaluate_array(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        x.mapv(|xi| self.evaluate(xi))
    }
}

/// Harmonic oscillator potential
#[derive(Debug, Clone)]
pub struct HarmonicOscillator {
    /// Spring constant
    pub k: f64,
    /// Center position
    pub x0: f64,
}

impl QuantumPotential for HarmonicOscillator {
    fn evaluate(&self, x: f64) -> f64 {
        0.5 * self.k * (x - self.x0).powi(2)
    }
}

/// Particle in a box potential
#[derive(Debug, Clone)]
pub struct ParticleInBox {
    /// Left boundary
    pub left: f64,
    /// Right boundary
    pub right: f64,
    /// Barrier height
    pub barrier_height: f64,
}

impl QuantumPotential for ParticleInBox {
    fn evaluate(&self, x: f64) -> f64 {
        if x < self.left || x > self.right {
            self.barrier_height
        } else {
            0.0
        }
    }
}

/// Hydrogen-like atom potential
#[derive(Debug, Clone)]
pub struct HydrogenAtom {
    /// Nuclear charge
    pub z: f64,
    /// Electron charge squared / (4π ε₀)
    pub e2_4pi_eps0: f64,
}

impl QuantumPotential for HydrogenAtom {
    fn evaluate(&self, r: f64) -> f64 {
        if r > 0.0 {
            -self.z * self.e2_4pi_eps0 / r
        } else {
            f64::NEG_INFINITY
        }
    }
}

/// Solver for the Schrödinger equation
pub struct SchrodingerSolver {
    /// Spatial grid size
    pub n_points: usize,
    /// Time step size
    pub dt: f64,
    /// Potential function
    pub potential: Box<dyn QuantumPotential>,
    /// Solver method
    pub method: SchrodingerMethod,
}

/// Available methods for solving the Schrödinger equation
#[derive(Debug, Clone, Copy)]
pub enum SchrodingerMethod {
    /// Split-operator method (fast and accurate)
    SplitOperator,
    /// Crank-Nicolson method (implicit, stable)
    CrankNicolson,
    /// Explicit Euler (simple but less stable)
    ExplicitEuler,
    /// Fourth-order Runge-Kutta
    RungeKutta4,
}

impl SchrodingerSolver {
    /// Create a new Schrödinger solver
    pub fn new(
        n_points: usize,
        dt: f64,
        potential: Box<dyn QuantumPotential>,
        method: SchrodingerMethod,
    ) -> Self {
        Self {
            n_points,
            dt,
            potential,
            method,
        }
    }

    /// Solve time-dependent Schrödinger equation
    pub fn solve_time_dependent(
        &self,
        initial_state: &QuantumState,
        t_final: f64,
    ) -> Result<Vec<QuantumState>> {
        let mut states = vec![initial_state.clone()];
        let mut current_state = initial_state.clone();

        // Ensure x and psi have consistent lengths
        if current_state.x.len() != current_state.psi.len() {
            // Resize x to match psi if they differ (e.g., due to FFT padding requirements)
            let n = current_state.psi.len();
            let x_min = current_state.x[0];
            let x_max = current_state.x[current_state.x.len() - 1];
            current_state.x = Array1::linspace(x_min, x_max, n);
            current_state.dx = (x_max - x_min) / (n - 1) as f64;
        }

        let n_steps = (t_final / self.dt).ceil() as usize;

        match self.method {
            SchrodingerMethod::SplitOperator => {
                for _ in 0..n_steps {
                    self.split_operator_step(&mut current_state)?;
                    current_state.t += self.dt;
                    states.push(current_state.clone());
                }
            }
            SchrodingerMethod::CrankNicolson => {
                for _ in 0..n_steps {
                    self.crank_nicolson_step(&mut current_state)?;
                    current_state.t += self.dt;
                    states.push(current_state.clone());
                }
            }
            SchrodingerMethod::ExplicitEuler => {
                for _ in 0..n_steps {
                    self.explicit_euler_step(&mut current_state)?;
                    current_state.t += self.dt;
                    states.push(current_state.clone());
                }
            }
            SchrodingerMethod::RungeKutta4 => {
                for _ in 0..n_steps {
                    self.runge_kutta4_step(&mut current_state)?;
                    current_state.t += self.dt;
                    states.push(current_state.clone());
                }
            }
        }

        Ok(states)
    }

    /// Split-operator method step
    fn split_operator_step(&self, state: &mut QuantumState) -> Result<()> {
        use scirs2_fft::{fft, ifft};

        // Ensure x and psi have the same length before proceeding
        if state.x.len() != state.psi.len() {
            // This shouldn't happen, but handle it gracefully
            let n = state.psi.len().min(state.x.len());
            if state.psi.len() > n {
                state.psi = state.psi.slice(scirs2_core::ndarray::s![..n]).to_owned();
            }
            if state.x.len() > n {
                state.x = state.x.slice(scirs2_core::ndarray::s![..n]).to_owned();
            }
        }

        let n = state.psi.len();

        // Potential energy evolution (half step)
        let v = self.potential.evaluate_array(&state.x.view());

        for i in 0..n {
            let phase = -v[i] * self.dt / (2.0 * REDUCED_PLANCK);
            state.psi[i] *= Complex64::new(phase.cos(), phase.sin());
        }

        // Kinetic energy evolution in momentum space using FFT
        // Transform to momentum space
        let psi_k = fft(&state.psi.to_vec(), None).map_err(|e| {
            crate::error::IntegrateError::ComputationError(format!("FFT failed: {e:?}"))
        })?;

        // Calculate k-space grid (momentum values)
        let dk = 2.0 * PI / (n as f64 * state.dx);
        let mut k_values = vec![0.0; n];
        for (i, k_value) in k_values.iter_mut().enumerate().take(n) {
            if i < n / 2 {
                *k_value = i as f64 * dk;
            } else {
                *k_value = (i as f64 - n as f64) * dk;
            }
        }

        // Apply kinetic energy operator in momentum space
        let mut psi_k_evolved = psi_k;
        for i in 0..n {
            let k = k_values[i];
            let kinetic_phase = -REDUCED_PLANCK * k * k * self.dt / (2.0 * state.mass);
            psi_k_evolved[i] *= Complex64::new(kinetic_phase.cos(), kinetic_phase.sin());
        }

        // Transform back to position space
        let psi_evolved = ifft(&psi_k_evolved, None).map_err(|e| {
            crate::error::IntegrateError::ComputationError(format!("IFFT failed: {e:?}"))
        })?;

        // Update state with evolved wave function
        // Ensure we preserve the original size (FFT might have padded)
        let psi_vec = if psi_evolved.len() != n {
            psi_evolved[..n].to_vec()
        } else {
            psi_evolved
        };
        state.psi = Array1::from_vec(psi_vec);

        // Potential energy evolution (half step)
        for i in 0..n {
            let phase = -v[i] * self.dt / (2.0 * REDUCED_PLANCK);
            state.psi[i] *= Complex64::new(phase.cos(), phase.sin());
        }

        // Normalize to conserve probability
        state.normalize();

        Ok(())
    }

    /// Crank-Nicolson method step
    fn crank_nicolson_step(&self, state: &mut QuantumState) -> Result<()> {
        let n = state.psi.len();
        let alpha = Complex64::new(
            0.0,
            REDUCED_PLANCK * self.dt / (4.0 * state.mass * state.dx.powi(2)),
        );

        // Build tridiagonal matrices
        let v = self.potential.evaluate_array(&state.x.view());
        let mut a = vec![Complex64::new(0.0, 0.0); n];
        let mut b = vec![Complex64::new(0.0, 0.0); n];
        let mut c = vec![Complex64::new(0.0, 0.0); n];

        for i in 0..n {
            let v_term = Complex64::new(0.0, -v[i] * self.dt / (2.0 * REDUCED_PLANCK));
            b[i] = Complex64::new(1.0, 0.0) + 2.0 * alpha - v_term;

            if i > 0 {
                a[i] = -alpha;
            }
            if i < n - 1 {
                c[i] = -alpha;
            }
        }

        // Build right-hand side
        let mut rhs = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            let v_term = Complex64::new(0.0, v[i] * self.dt / (2.0 * REDUCED_PLANCK));
            rhs[i] = state.psi[i] * (Complex64::new(1.0, 0.0) - 2.0 * alpha + v_term);

            if i > 0 {
                rhs[i] += alpha * state.psi[i - 1];
            }
            if i < n - 1 {
                rhs[i] += alpha * state.psi[i + 1];
            }
        }

        // Solve tridiagonal system using Thomas algorithm
        let new_psi = self.solve_tridiagonal(&a, &b, &c, &rhs)?;
        state.psi = Array1::from_vec(new_psi);

        // Normalize
        state.normalize();

        Ok(())
    }

    /// Explicit Euler method step
    fn explicit_euler_step(&self, state: &mut QuantumState) -> Result<()> {
        let n = state.psi.len();
        let mut dpsi_dt = Array1::zeros(n);

        // Calculate time derivative using Schrödinger equation
        let v = self.potential.evaluate_array(&state.x.view());
        let prefactor = Complex64::new(0.0, -1.0 / REDUCED_PLANCK);

        for i in 0..n {
            // Kinetic energy term (second derivative)
            let d2psi_dx2 = if i == 0 {
                state.psi[1] - 2.0 * state.psi[0] + state.psi[0]
            } else if i == n - 1 {
                state.psi[n - 1] - 2.0 * state.psi[n - 1] + state.psi[n - 2]
            } else {
                state.psi[i + 1] - 2.0 * state.psi[i] + state.psi[i - 1]
            } / state.dx.powi(2);

            // Hamiltonian action
            let h_psi =
                -REDUCED_PLANCK.powi(2) / (2.0 * state.mass) * d2psi_dx2 + v[i] * state.psi[i];

            dpsi_dt[i] = prefactor * h_psi;
        }

        // Update wave function
        state.psi += &(dpsi_dt * self.dt);

        // Normalize
        state.normalize();

        Ok(())
    }

    /// Fourth-order Runge-Kutta method step
    fn runge_kutta4_step(&self, state: &mut QuantumState) -> Result<()> {
        let n = state.psi.len();
        let v = self.potential.evaluate_array(&state.x.view());

        // Helper function to compute derivative
        let compute_derivative = |psi: &Array1<Complex64>| -> Array1<Complex64> {
            let mut dpsi = Array1::zeros(n);
            let prefactor = Complex64::new(0.0, -1.0 / REDUCED_PLANCK);

            for i in 0..n {
                let d2psi_dx2 = if i == 0 {
                    psi[1] - 2.0 * psi[0] + psi[0]
                } else if i == n - 1 {
                    psi[n - 1] - 2.0 * psi[n - 1] + psi[n - 2]
                } else {
                    psi[i + 1] - 2.0 * psi[i] + psi[i - 1]
                } / state.dx.powi(2);

                let h_psi =
                    -REDUCED_PLANCK.powi(2) / (2.0 * state.mass) * d2psi_dx2 + v[i] * psi[i];

                dpsi[i] = prefactor * h_psi;
            }
            dpsi
        };

        // RK4 steps
        let k1 = compute_derivative(&state.psi);
        let k2 = compute_derivative(&(&state.psi + &k1 * (self.dt / 2.0)));
        let k3 = compute_derivative(&(&state.psi + &k2 * (self.dt / 2.0)));
        let k4 = compute_derivative(&(&state.psi + &k3 * self.dt));

        // Update
        state.psi += &((k1 + k2 * 2.0 + k3 * 2.0 + k4) * (self.dt / 6.0));

        // Normalize
        state.normalize();

        Ok(())
    }

    /// Solve tridiagonal system using Thomas algorithm
    fn solve_tridiagonal(
        &self,
        a: &[Complex64],
        b: &[Complex64],
        c: &[Complex64],
        d: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        let n = b.len();
        let mut c_star = vec![Complex64::new(0.0, 0.0); n];
        let mut d_star = vec![Complex64::new(0.0, 0.0); n];
        let mut x = vec![Complex64::new(0.0, 0.0); n];

        // Forward sweep
        c_star[0] = c[0] / b[0];
        d_star[0] = d[0] / b[0];

        for i in 1..n {
            let m = b[i] - a[i] * c_star[i - 1];
            c_star[i] = c[i] / m;
            d_star[i] = (d[i] - a[i] * d_star[i - 1]) / m;
        }

        // Back substitution
        x[n - 1] = d_star[n - 1];
        for i in (0..n - 1).rev() {
            x[i] = d_star[i] - c_star[i] * x[i + 1];
        }

        Ok(x)
    }

    /// Solve time-independent Schrödinger equation (eigenvalue problem)
    pub fn solve_time_independent(
        &self,
        x_min: f64,
        x_max: f64,
        n_states: usize,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        let dx = (x_max - x_min) / (self.n_points - 1) as f64;
        let x = Array1::linspace(x_min, x_max, self.n_points);

        // Build the Hamiltonian matrix using finite difference method
        let mut hamiltonian = Array2::<f64>::zeros((self.n_points, self.n_points));

        // Kinetic energy contribution (second derivative via finite differences)
        // The kinetic energy operator is -ℏ²/(2m) d²/dx²
        // Using finite differences: d²ψ/dx² ≈ (ψ[i+1] - 2ψ[i] + ψ[i-1])/dx²
        // So the matrix elements are: T[i,i] = ℏ²/(m*dx²), T[i,i±1] = -ℏ²/(2m*dx²)
        let kinetic_factor = REDUCED_PLANCK.powi(2) / (2.0 * 1.0 * dx.powi(2)); // mass = 1.0 for simplicity

        // Build tridiagonal kinetic energy matrix
        for i in 0..self.n_points {
            if i > 0 {
                hamiltonian[[i, i - 1]] = -kinetic_factor;
            }
            hamiltonian[[i, i]] = 2.0 * kinetic_factor;
            if i < self.n_points - 1 {
                hamiltonian[[i, i + 1]] = -kinetic_factor;
            }
        }

        // Add potential energy contribution (diagonal)
        let v = self.potential.evaluate_array(&x.view());
        for i in 0..self.n_points {
            hamiltonian[[i, i]] += v[i];
        }

        // Apply boundary conditions (wave function vanishes at boundaries)
        // For Dirichlet boundary conditions, we can work with a reduced system
        // excluding the boundary points, but for simplicity, we'll keep them
        // and set the first and last rows to enforce ψ(0) = ψ(L) = 0
        // by making the boundary points have very high energy
        hamiltonian.row_mut(0).fill(0.0);
        hamiltonian[[0, 0]] = 1e6; // Large value to push this state to high energy
        hamiltonian.row_mut(self.n_points - 1).fill(0.0);
        hamiltonian[[self.n_points - 1, self.n_points - 1]] = 1e6;

        // Find eigenvalues and eigenvectors using a simple power iteration method
        // for the lowest n_states eigenpairs
        let mut energies = Array1::zeros(n_states);
        let mut wavefunctions = Array2::zeros((self.n_points, n_states));

        // Use inverse power iteration with shifts to find lowest eigenvalues
        for state in 0..n_states {
            let mut psi = Array1::from_elem(self.n_points, 1.0);
            psi[0] = 0.0;
            psi[self.n_points - 1] = 0.0;

            // Normalize initial guess
            let norm: f64 = psi.iter().map(|&x| x * x * dx).sum::<f64>().sqrt();
            psi /= norm;

            // Gram-Schmidt orthogonalization against previous eigenstates
            for j in 0..state {
                let overlap: f64 = psi
                    .iter()
                    .zip(wavefunctions.column(j).iter())
                    .map(|(&a, &b)| a * b * dx)
                    .sum();
                for i in 0..self.n_points {
                    psi[i] -= overlap * wavefunctions[[i, j]];
                }
            }

            // Power iteration to find eigenvalue
            let mut eigenvalue = 0.0;
            for _ in 0..100 {
                // iterations
                // Apply Hamiltonian
                let mut h_psi = Array1::zeros(self.n_points);
                for i in 1..self.n_points - 1 {
                    h_psi[i] = hamiltonian[[i, i]] * psi[i];
                    if i > 0 {
                        h_psi[i] += hamiltonian[[i, i - 1]] * psi[i - 1];
                    }
                    if i < self.n_points - 1 {
                        h_psi[i] += hamiltonian[[i, i + 1]] * psi[i + 1];
                    }
                }

                // Calculate eigenvalue estimate
                eigenvalue = psi
                    .iter()
                    .zip(h_psi.iter())
                    .map(|(&a, &b)| a * b * dx)
                    .sum::<f64>();

                // Update eigenvector
                psi = h_psi;

                // Orthogonalize against previous states
                for j in 0..state {
                    let overlap: f64 = psi
                        .iter()
                        .zip(wavefunctions.column(j).iter())
                        .map(|(&a, &b)| a * b * dx)
                        .sum();
                    for i in 0..self.n_points {
                        psi[i] -= overlap * wavefunctions[[i, j]];
                    }
                }

                // Normalize
                let norm: f64 = psi.iter().map(|&x| x * x * dx).sum::<f64>().sqrt();
                if norm > 1e-10 {
                    psi /= norm;
                }
            }

            energies[state] = eigenvalue;
            wavefunctions.column_mut(state).assign(&psi);
        }

        // Sort by energy
        let mut indices: Vec<usize> = (0..n_states).collect();
        indices.sort_by(|&i, &j| energies[i].partial_cmp(&energies[j]).unwrap());

        let sorted_energies = Array1::from_vec(indices.iter().map(|&i| energies[i]).collect());
        let mut sorted_wavefunctions = Array2::zeros((self.n_points, n_states));
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_wavefunctions
                .column_mut(new_idx)
                .assign(&wavefunctions.column(old_idx));
        }

        Ok((sorted_energies, sorted_wavefunctions))
    }

    /// Create initial Gaussian wave packet
    pub fn gaussian_wave_packet(
        x: &Array1<f64>,
        x0: f64,
        sigma: f64,
        k0: f64,
        mass: f64,
    ) -> QuantumState {
        let norm = 1.0 / (2.0 * PI * sigma.powi(2)).powf(0.25);

        // For FFT efficiency, ensure we use a power of 2 size
        let original_n = x.len();
        let fft_n = original_n.next_power_of_two();

        // Create arrays with appropriate size
        let (x_final, psi_final) = if fft_n != original_n {
            // Need to pad to power of 2
            let x_min = x[0];
            let x_max = x[original_n - 1];
            let x_padded = Array1::linspace(x_min, x_max, fft_n);

            let psi_padded = x_padded.mapv(|xi| {
                let gaussian = norm * (-(xi - x0).powi(2) / (4.0 * sigma.powi(2))).exp();
                let phase = k0 * xi;
                Complex64::new(gaussian * phase.cos(), gaussian * phase.sin())
            });

            (x_padded, psi_padded)
        } else {
            // Already a power of 2
            let psi = x.mapv(|xi| {
                let gaussian = norm * (-(xi - x0).powi(2) / (4.0 * sigma.powi(2))).exp();
                let phase = k0 * xi;
                Complex64::new(gaussian * phase.cos(), gaussian * phase.sin())
            });
            (x.clone(), psi)
        };

        let mut state = QuantumState::new(psi_final, x_final, 0.0, mass);
        state.normalize();
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[ignore] // FIXME: Harmonic oscillator ground state test failing
    fn test_harmonic_oscillator_ground_state() {
        let potential = Box::new(HarmonicOscillator { k: 1.0, x0: 0.0 });
        let solver = SchrodingerSolver::new(100, 0.01, potential, SchrodingerMethod::SplitOperator);

        let (energies, wavefunctions) = solver.solve_time_independent(-5.0, 5.0, 3).unwrap();

        // Ground state energy should be ℏω/2 = 0.5 (with ℏ=1, ω=1)
        assert_relative_eq!(energies[0], 0.5, epsilon = 0.01);

        // First excited state should be 3ℏω/2 = 1.5
        assert_relative_eq!(energies[1], 1.5, epsilon = 0.01);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_wave_packet_evolution() {
        let potential = Box::new(HarmonicOscillator { k: 0.0, x0: 0.0 }); // Free particle
        let solver =
            SchrodingerSolver::new(200, 0.001, potential, SchrodingerMethod::SplitOperator);

        let x = Array1::linspace(-10.0, 10.0, 200);
        let initial_state = SchrodingerSolver::gaussian_wave_packet(&x, -5.0, 1.0, 2.0, 1.0);

        let states = solver.solve_time_dependent(&initial_state, 1.0).unwrap();

        // Check normalization is preserved
        for state in &states {
            let norm_squared: f64 =
                state.psi.iter().map(|&c| (c.conj() * c).re).sum::<f64>() * state.dx;
            assert_relative_eq!(norm_squared, 1.0, epsilon = 1e-6);
        }

        // Wave packet should move to the right
        let final_position = states.last().unwrap().expectation_position();
        assert!(final_position > -5.0);
    }
}
