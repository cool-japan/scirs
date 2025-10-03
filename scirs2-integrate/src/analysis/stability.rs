//! Stability analysis tools for dynamical systems
//!
//! This module provides the StabilityAnalyzer and related functionality
//! for assessing the stability of fixed points and periodic orbits.

use crate::analysis::advanced;
use crate::analysis::types::*;
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;

/// Stability analyzer for dynamical systems
pub struct StabilityAnalyzer {
    /// System dimension
    pub dimension: usize,
    /// Tolerance for fixed point detection
    pub tolerance: f64,
    /// Integration time for trajectory analysis
    pub integration_time: f64,
    /// Number of test points for basin analysis
    pub basin_grid_size: usize,
}

impl StabilityAnalyzer {
    /// Create a new stability analyzer
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            tolerance: 1e-8,
            integration_time: 100.0,
            basin_grid_size: 50,
        }
    }

    /// Perform comprehensive stability analysis
    pub fn analyze_stability<F>(
        &self,
        system: F,
        domain: &[(f64, f64)],
    ) -> IntegrateResult<StabilityResult>
    where
        F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync + Clone + 'static,
    {
        // Find fixed points
        let fixed_points = self.find_fixed_points(&system, domain)?;

        // Find periodic orbits (simplified)
        let periodic_orbits = self.find_periodic_orbits(&system, domain)?;

        // Compute Lyapunov exponents
        let lyapunov_exponents = self.compute_lyapunov_exponents(&system)?;

        // Analyze basins of attraction
        let basin_analysis = if self.dimension == 2 {
            Some(self.analyze_basins(&system, domain, &fixed_points)?)
        } else {
            None
        };

        Ok(StabilityResult {
            fixed_points,
            periodic_orbits,
            lyapunov_exponents,
            basin_analysis,
        })
    }

    /// Find fixed points in the given domain
    fn find_fixed_points<F>(
        &self,
        system: &F,
        domain: &[(f64, f64)],
    ) -> IntegrateResult<Vec<FixedPoint>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let mut fixed_points: Vec<FixedPoint> = Vec::new();
        let grid_size = 10; // Number of initial guesses per dimension

        // Generate grid of initial guesses
        let mut initial_guesses = Vec::new();
        self.generate_grid_points(domain, grid_size, &mut initial_guesses);

        for guess in initial_guesses {
            if let Ok(fixed_point) = self.newton_raphson_fixed_point(system, &guess) {
                // Check if this fixed point is already found
                let mut is_duplicate = false;
                for existing_fp in &fixed_points {
                    let distance = (&fixed_point - &existing_fp.location)
                        .iter()
                        .map(|&x| x * x)
                        .sum::<f64>()
                        .sqrt();

                    if distance < self.tolerance * 10.0 {
                        is_duplicate = true;
                        break;
                    }
                }

                if !is_duplicate {
                    // Compute stability
                    let jacobian = self.compute_jacobian_at_point(system, &fixed_point)?;
                    let eigenvalues = self.compute_real_eigenvalues(&jacobian)?;
                    let eigenvectors = self.compute_eigenvectors(&jacobian, &eigenvalues)?;
                    let stability = self.classify_stability(&eigenvalues);

                    fixed_points.push(FixedPoint {
                        location: fixed_point,
                        stability,
                        eigenvalues,
                        eigenvectors,
                    });
                }
            }
        }

        Ok(fixed_points)
    }

    /// Generate grid of points in domain
    fn generate_grid_points(
        &self,
        domain: &[(f64, f64)],
        grid_size: usize,
        points: &mut Vec<Array1<f64>>,
    ) {
        fn generate_recursive(
            domain: &[(f64, f64)],
            grid_size: usize,
            current: &mut Vec<f64>,
            dim: usize,
            points: &mut Vec<Array1<f64>>,
        ) {
            if dim == domain.len() {
                points.push(Array1::from_vec(current.clone()));
                return;
            }

            // Check for division by zero in step calculation
            if grid_size <= 1 {
                return; // Skip invalid grid size
            }
            let step = (domain[dim].1 - domain[dim].0) / (grid_size - 1) as f64;
            for i in 0..grid_size {
                let value = domain[dim].0 + i as f64 * step;
                current.push(value);
                generate_recursive(domain, grid_size, current, dim + 1, points);
                current.pop();
            }
        }

        let mut current = Vec::new();
        generate_recursive(domain, grid_size, &mut current, 0, points);
    }

    /// Find fixed point using Newton-Raphson method
    fn newton_raphson_fixed_point<F>(
        &self,
        system: &F,
        initial_guess: &Array1<f64>,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let mut x = initial_guess.clone();
        let max_iterations = 100;

        for _ in 0..max_iterations {
            let f_x = system(&x);
            let sum_squares = f_x.iter().map(|&v| v * v).sum::<f64>();
            if sum_squares < 0.0 {
                return Err(IntegrateError::ComputationError(
                    "Negative sum of squares in residual norm calculation".to_string(),
                ));
            }
            let residual_norm = sum_squares.sqrt();

            if residual_norm < self.tolerance {
                return Ok(x);
            }

            let jacobian = self.compute_jacobian_at_point(system, &x)?;

            // Solve J * dx = -f(x)
            let mut augmented = Array2::zeros((self.dimension, self.dimension + 1));
            for i in 0..self.dimension {
                for j in 0..self.dimension {
                    augmented[[i, j]] = jacobian[[i, j]];
                }
                augmented[[i, self.dimension]] = -f_x[i];
            }

            let dx = self.gaussian_elimination(&augmented)?;
            x += &dx;
        }

        Err(IntegrateError::ConvergenceError(
            "Newton-Raphson did not converge".to_string(),
        ))
    }

    /// Compute Jacobian at a specific point
    fn compute_jacobian_at_point<F>(
        &self,
        system: &F,
        x: &Array1<f64>,
    ) -> IntegrateResult<Array2<f64>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let h = 1e-8_f64;
        let n = x.len();
        let mut jacobian = Array2::zeros((n, n));

        let f0 = system(x);

        // Check for valid step size
        if h.abs() < 1e-15 {
            return Err(IntegrateError::ComputationError(
                "Step size too small for finite difference calculation".to_string(),
            ));
        }

        for j in 0..n {
            let mut x_plus = x.clone();
            x_plus[j] += h;
            let f_plus = system(&x_plus);

            for i in 0..n {
                jacobian[[i, j]] = (f_plus[i] - f0[i]) / h;
            }
        }

        Ok(jacobian)
    }

    /// Solve linear system using Gaussian elimination
    fn gaussian_elimination(&self, augmented: &Array2<f64>) -> IntegrateResult<Array1<f64>> {
        let n = augmented.nrows();
        let mut a = augmented.clone();

        // Forward elimination
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            for i in k + 1..n {
                if a[[i, k]].abs() > a[[max_row, k]].abs() {
                    max_row = i;
                }
            }

            // Swap rows
            if max_row != k {
                for j in k..n + 1 {
                    let temp = a[[k, j]];
                    a[[k, j]] = a[[max_row, j]];
                    a[[max_row, j]] = temp;
                }
            }

            // Check for singularity
            if a[[k, k]].abs() < 1e-14 {
                return Err(IntegrateError::ComputationError(
                    "Matrix is singular".to_string(),
                ));
            }

            // Eliminate
            for i in k + 1..n {
                let factor = a[[i, k]] / a[[k, k]];
                for j in k..n + 1 {
                    a[[i, j]] -= factor * a[[k, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = a[[i, n]];
            for j in i + 1..n {
                x[i] -= a[[i, j]] * x[j];
            }
            // Check for zero diagonal element
            if a[[i, i]].abs() < 1e-14 {
                return Err(IntegrateError::ComputationError(
                    "Zero diagonal element in back substitution".to_string(),
                ));
            }
            x[i] /= a[[i, i]];
        }

        Ok(x)
    }

    /// Compute real eigenvalues (simplified implementation)
    fn compute_real_eigenvalues(&self, matrix: &Array2<f64>) -> IntegrateResult<Vec<Complex64>> {
        // For now, use a simplified approach for 2x2 matrices
        let n = matrix.nrows();

        if n == 2 {
            let a = matrix[[0, 0]];
            let b = matrix[[0, 1]];
            let c = matrix[[1, 0]];
            let d = matrix[[1, 1]];

            let trace = a + d;
            let det = a * d - b * c;
            let discriminant = trace * trace - 4.0 * det;

            if discriminant >= 0.0 {
                let sqrt_disc = discriminant.sqrt();
                let lambda1 = (trace + sqrt_disc) / 2.0;
                let lambda2 = (trace - sqrt_disc) / 2.0;
                Ok(vec![
                    Complex64::new(lambda1, 0.0),
                    Complex64::new(lambda2, 0.0),
                ])
            } else {
                let real_part = trace / 2.0;
                let imag_part = (-discriminant).sqrt() / 2.0;
                Ok(vec![
                    Complex64::new(real_part, imag_part),
                    Complex64::new(real_part, -imag_part),
                ])
            }
        } else {
            // For higher dimensions, use the QR algorithm
            self.eigenvalues_qr_algorithm(matrix)
        }
    }

    /// Compute eigenvalues using QR algorithm for larger matrices
    fn eigenvalues_qr_algorithm(&self, matrix: &Array2<f64>) -> IntegrateResult<Vec<Complex64>> {
        let n = matrix.nrows();
        let mut a = matrix.clone();
        let max_iterations = 100;
        let tolerance = 1e-10;

        // First, reduce to upper Hessenberg form for better convergence
        a = self.reduce_to_hessenberg(&a)?;

        // Apply QR iterations
        for _ in 0..max_iterations {
            let (q, r) = self.qr_decomposition_real(&a)?;
            a = r.dot(&q);

            // Check convergence by examining sub-diagonal elements
            let mut converged = true;
            for i in 1..n {
                if a[[i, i - 1]].abs() > tolerance {
                    converged = false;
                    break;
                }
            }

            if converged {
                break;
            }
        }

        // Extract eigenvalues from the diagonal
        let mut eigenvalues = Vec::new();
        let mut i = 0;
        while i < n {
            if i == n - 1 || a[[i + 1, i]].abs() < tolerance {
                // Real eigenvalue
                eigenvalues.push(Complex64::new(a[[i, i]], 0.0));
                i += 1;
            } else {
                // Complex conjugate pair
                let trace = a[[i, i]] + a[[i + 1, i + 1]];
                let det = a[[i, i]] * a[[i + 1, i + 1]] - a[[i, i + 1]] * a[[i + 1, i]];
                let discriminant = trace * trace - 4.0 * det;

                if discriminant >= 0.0 {
                    let sqrt_disc = discriminant.sqrt();
                    eigenvalues.push(Complex64::new((trace + sqrt_disc) / 2.0, 0.0));
                    eigenvalues.push(Complex64::new((trace - sqrt_disc) / 2.0, 0.0));
                } else {
                    let real_part = trace / 2.0;
                    let imag_part = (-discriminant).sqrt() / 2.0;
                    eigenvalues.push(Complex64::new(real_part, imag_part));
                    eigenvalues.push(Complex64::new(real_part, -imag_part));
                }
                i += 2;
            }
        }

        Ok(eigenvalues)
    }

    /// Reduce matrix to upper Hessenberg form using Householder reflections
    fn reduce_to_hessenberg(&self, matrix: &Array2<f64>) -> IntegrateResult<Array2<f64>> {
        let n = matrix.nrows();
        let mut h = matrix.clone();

        for k in 0..(n - 2) {
            // Extract the column below the diagonal
            let mut x = Array1::<f64>::zeros(n - k - 1);
            for i in 0..(n - k - 1) {
                x[i] = h[[k + 1 + i, k]];
            }

            if x.iter().map(|&v| v * v).sum::<f64>().sqrt() > 1e-15 {
                // Compute Householder vector
                let alpha = if x[0] >= 0.0 {
                    -x.iter().map(|&v| v * v).sum::<f64>().sqrt()
                } else {
                    x.iter().map(|&v| v * v).sum::<f64>().sqrt()
                };

                let mut v = x.clone();
                v[0] -= alpha;
                let v_norm = v.iter().map(|&vi| vi * vi).sum::<f64>().sqrt();

                if v_norm > 1e-15 {
                    v.mapv_inplace(|vi| vi / v_norm);

                    // Apply Householder reflection: H = I - 2*v*v^T
                    // H * A
                    for j in k..n {
                        let dot_product: f64 =
                            (0..(n - k - 1)).map(|i| v[i] * h[[k + 1 + i, j]]).sum();
                        for i in 0..(n - k - 1) {
                            h[[k + 1 + i, j]] -= 2.0 * v[i] * dot_product;
                        }
                    }

                    // A * H
                    for i in 0..n {
                        let dot_product: f64 =
                            (0..(n - k - 1)).map(|j| h[[i, k + 1 + j]] * v[j]).sum();
                        for j in 0..(n - k - 1) {
                            h[[i, k + 1 + j]] -= 2.0 * v[j] * dot_product;
                        }
                    }
                }
            }
        }

        Ok(h)
    }

    /// QR decomposition for real matrices
    fn qr_decomposition_real(
        &self,
        matrix: &Array2<f64>,
    ) -> IntegrateResult<(Array2<f64>, Array2<f64>)> {
        let (m, n) = matrix.dim();
        let mut q = Array2::<f64>::eye(m);
        let mut r = matrix.clone();

        for k in 0..std::cmp::min(m - 1, n) {
            // Extract column k from row k onwards
            let mut x = Array1::<f64>::zeros(m - k);
            for i in 0..(m - k) {
                x[i] = r[[k + i, k]];
            }

            // Compute Householder vector
            let norm_x = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm_x > 1e-15 {
                let alpha = if x[0] >= 0.0 { -norm_x } else { norm_x };

                let mut v = x.clone();
                v[0] -= alpha;
                let v_norm = v.iter().map(|&vi| vi * vi).sum::<f64>().sqrt();

                if v_norm > 1e-15 {
                    v.mapv_inplace(|vi| vi / v_norm);

                    // Apply Householder reflection to R
                    for j in k..n {
                        let dot_product: f64 = (0..(m - k)).map(|i| v[i] * r[[k + i, j]]).sum();
                        for i in 0..(m - k) {
                            r[[k + i, j]] -= 2.0 * v[i] * dot_product;
                        }
                    }

                    // Apply Householder reflection to Q
                    for i in 0..m {
                        let dot_product: f64 = (0..(m - k)).map(|j| q[[i, k + j]] * v[j]).sum();
                        for j in 0..(m - k) {
                            q[[i, k + j]] -= 2.0 * v[j] * dot_product;
                        }
                    }
                }
            }
        }

        Ok((q, r))
    }

    /// Compute eigenvectors (simplified)
    fn compute_eigenvectors(
        &self,
        _matrix: &Array2<f64>,
        eigenvalues: &[Complex64],
    ) -> IntegrateResult<Array2<Complex64>> {
        let n = eigenvalues.len();
        let eigenvectors = Array2::<Complex64>::zeros((n, n));

        // Simplified: return identity matrix
        // In practice, would solve (A - λI)v = 0 for each eigenvalue
        Ok(eigenvectors)
    }

    /// Classify stability based on eigenvalues
    fn classify_stability(&self, eigenvalues: &[Complex64]) -> StabilityType {
        let mut has_positive_real = false;
        let mut has_negative_real = false;
        let mut has_zero_real = false;

        for eigenvalue in eigenvalues {
            if eigenvalue.re > 1e-10 {
                has_positive_real = true;
            } else if eigenvalue.re < -1e-10 {
                has_negative_real = true;
            } else {
                has_zero_real = true;
            }
        }

        if has_zero_real {
            StabilityType::Degenerate
        } else if has_positive_real && has_negative_real {
            StabilityType::Saddle
        } else if has_positive_real {
            StabilityType::Unstable
        } else if has_negative_real {
            StabilityType::Stable
        } else {
            StabilityType::Center
        }
    }

    /// Find periodic orbits using multiple detection methods
    fn find_periodic_orbits<F>(
        &self,
        system: &F,
        domain: &[(f64, f64)],
    ) -> IntegrateResult<Vec<PeriodicOrbit>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let mut periodic_orbits = Vec::new();

        // Method 1: Shooting method for periodic orbits
        if let Ok(shooting_orbits) = self.shooting_method_periodic_orbits(system, domain) {
            periodic_orbits.extend(shooting_orbits);
        }

        // Method 2: Return map analysis
        if let Ok(return_map_orbits) = self.return_map_periodic_orbits(system, domain) {
            periodic_orbits.extend(return_map_orbits);
        }

        // Method 3: Fourier analysis of trajectories
        if let Ok(fourier_orbits) = self.fourier_analysis_periodic_orbits(system, domain) {
            periodic_orbits.extend(fourier_orbits);
        }

        // Remove duplicates based on spatial proximity
        let filtered_orbits = self.remove_duplicate_periodic_orbits(periodic_orbits);

        Ok(filtered_orbits)
    }

    /// Use shooting method to find periodic orbits
    fn shooting_method_periodic_orbits<F>(
        &self,
        system: &F,
        domain: &[(f64, f64)],
    ) -> IntegrateResult<Vec<PeriodicOrbit>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let mut periodic_orbits = Vec::new();

        if self.dimension != 2 {
            return Ok(periodic_orbits); // Shooting method implementation for 2D systems only
        }

        // Generate initial guesses for periodic orbits
        let n_guesses = 20;
        let mut initial_points = Vec::new();
        self.generate_grid_points(domain, n_guesses, &mut initial_points);

        // Try different periods
        let periods = vec![
            std::f64::consts::PI,       // π
            2.0 * std::f64::consts::PI, // 2π
            std::f64::consts::PI / 2.0, // π/2
            4.0 * std::f64::consts::PI, // 4π
        ];

        for initial_point in &initial_points {
            for &period in &periods {
                if let Ok(orbit) = self.shooting_method_single_orbit(system, initial_point, period)
                {
                    periodic_orbits.push(orbit);
                }
            }
        }

        Ok(periodic_orbits)
    }

    /// Single orbit detection using shooting method
    fn shooting_method_single_orbit<F>(
        &self,
        system: &F,
        initial_guess: &Array1<f64>,
        period: f64,
    ) -> IntegrateResult<PeriodicOrbit>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let max_iterations = 50;
        let tolerance = 1e-8;
        let dt = period / 100.0; // Integration step size

        let mut current_guess = initial_guess.clone();

        // Newton iteration for shooting method
        for _iter in 0..max_iterations {
            // Integrate forward for one period
            let final_state =
                self.integrate_trajectory_period(system, &current_guess, period, dt)?;

            // Compute the shooting function: F(x0) = x(T) - x0
            let shooting_residual = &final_state - &current_guess;
            let residual_norm = shooting_residual.iter().map(|&x| x * x).sum::<f64>().sqrt();

            if residual_norm < tolerance {
                // Found a periodic orbit
                let floquet_multipliers =
                    self.compute_floquet_multipliers(system, &current_guess, period)?;
                let stability = self.classify_periodic_orbit_stability(&floquet_multipliers);

                return Ok(PeriodicOrbit {
                    representative_point: current_guess,
                    period,
                    stability,
                    floquet_multipliers,
                });
            }

            // Compute Jacobian of the flow map
            let flow_jacobian = self.compute_flow_jacobian(system, &current_guess, period, dt)?;

            // Newton step: solve (∂F/∂x0) * Δx0 = -F(x0)
            let identity = Array2::<f64>::eye(self.dimension);
            let shooting_jacobian = &flow_jacobian - &identity;

            // Solve the linear system
            let newton_step =
                self.solve_linear_system_for_shooting(&shooting_jacobian, &(-&shooting_residual))?;
            current_guess += &newton_step;
        }

        Err(IntegrateError::ConvergenceError(
            "Shooting method did not converge to periodic orbit".to_string(),
        ))
    }

    /// Integrate trajectory for a specified period
    fn integrate_trajectory_period<F>(
        &self,
        system: &F,
        initial_state: &Array1<f64>,
        period: f64,
        dt: f64,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let n_steps = (period / dt) as usize;
        let mut state = initial_state.clone();

        // Fourth-order Runge-Kutta integration
        for _ in 0..n_steps {
            let k1 = system(&state);
            let k2 = system(&(&state + &(&k1 * (dt / 2.0))));
            let k3 = system(&(&state + &(&k2 * (dt / 2.0))));
            let k4 = system(&(&state + &(&k3 * dt)));

            state += &((&k1 + &k2 * 2.0 + &k3 * 2.0 + &k4) * (dt / 6.0));
        }

        Ok(state)
    }

    /// Compute flow map Jacobian using finite differences
    fn compute_flow_jacobian<F>(
        &self,
        system: &F,
        initial_state: &Array1<f64>,
        period: f64,
        dt: f64,
    ) -> IntegrateResult<Array2<f64>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let h = 1e-8;
        let n = initial_state.len();
        let mut jacobian = Array2::<f64>::zeros((n, n));

        // Base trajectory
        let base_final = self.integrate_trajectory_period(system, initial_state, period, dt)?;

        // Perturb each component and compute finite differences
        for j in 0..n {
            let mut perturbed_initial = initial_state.clone();
            perturbed_initial[j] += h;

            let perturbed_final =
                self.integrate_trajectory_period(system, &perturbed_initial, period, dt)?;

            for i in 0..n {
                jacobian[[i, j]] = (perturbed_final[i] - base_final[i]) / h;
            }
        }

        Ok(jacobian)
    }

    /// Compute Floquet multipliers for periodic orbit stability analysis
    fn compute_floquet_multipliers<F>(
        &self,
        system: &F,
        representative_point: &Array1<f64>,
        period: f64,
    ) -> IntegrateResult<Vec<Complex64>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let dt = period / 100.0;
        let flow_jacobian = self.compute_flow_jacobian(system, representative_point, period, dt)?;

        // Compute eigenvalues of the flow map Jacobian (Floquet multipliers)
        let multipliers = self.compute_real_eigenvalues(&flow_jacobian)?;

        Ok(multipliers)
    }

    /// Classify periodic orbit stability based on Floquet multipliers
    fn classify_periodic_orbit_stability(
        &self,
        floquet_multipliers: &[Complex64],
    ) -> StabilityType {
        // For periodic orbits, stability is determined by Floquet multipliers
        // Stable if all multipliers have |λ| < 1
        // Unstable if any multiplier has |λ| > 1

        let max_magnitude = floquet_multipliers
            .iter()
            .map(|m| m.norm())
            .fold(0.0, f64::max);

        if max_magnitude < 1.0 - 1e-10 {
            StabilityType::Stable
        } else if max_magnitude > 1.0 + 1e-10 {
            StabilityType::Unstable
        } else {
            // One or more multipliers on unit circle
            let on_unit_circle = floquet_multipliers
                .iter()
                .any(|m| (m.norm() - 1.0).abs() < 1e-10);

            if on_unit_circle {
                StabilityType::Center
            } else {
                StabilityType::Degenerate
            }
        }
    }

    /// Return map analysis for periodic orbit detection
    fn return_map_periodic_orbits<F>(
        &self,
        system: &F,
        domain: &[(f64, f64)],
    ) -> IntegrateResult<Vec<PeriodicOrbit>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let mut periodic_orbits = Vec::new();

        if self.dimension != 2 {
            return Ok(periodic_orbits); // Return map analysis for 2D systems only
        }

        // Define a Poincaré section (e.g., x = 0)
        let section_plane = Array1::from_vec(vec![1.0, 0.0]); // Normal to x-axis
        let section_point = Array1::zeros(2); // Origin

        // Generate several trajectories and find their intersections with the Poincaré section
        let n_trajectories = 10;
        let mut initial_points = Vec::new();
        self.generate_grid_points(domain, n_trajectories, &mut initial_points);

        for initial_point in &initial_points {
            if let Ok(return_points) = self.compute_poincare_return_map(
                system,
                initial_point,
                &section_plane,
                &section_point,
            ) {
                // Analyze return points for periodicity
                if let Ok(orbit) = self.analyze_return_map_for_periodicity(&return_points) {
                    periodic_orbits.push(orbit);
                }
            }
        }

        Ok(periodic_orbits)
    }

    /// Compute Poincaré return map
    fn compute_poincare_return_map<F>(
        &self,
        system: &F,
        initial_point: &Array1<f64>,
        section_normal: &Array1<f64>,
        section_point: &Array1<f64>,
    ) -> IntegrateResult<Vec<Array1<f64>>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let mut return_points = Vec::new();
        let dt = 0.01;
        let max_time = 50.0;
        let n_steps = (max_time / dt) as usize;

        let mut state = initial_point.clone();
        let mut prev_distance = self.distance_to_section(&state, section_normal, section_point);

        for _ in 0..n_steps {
            // Integrate one step
            let derivative = system(&state);
            state += &(derivative * dt);

            // Check for section crossing
            let curr_distance = self.distance_to_section(&state, section_normal, section_point);

            if prev_distance * curr_distance < 0.0 {
                // Crossed the section, refine the crossing point
                if let Ok(crossing_point) =
                    self.refine_section_crossing(system, &state, dt, section_normal, section_point)
                {
                    return_points.push(crossing_point);

                    if return_points.len() > 20 {
                        break; // Collect enough return points
                    }
                }
            }

            prev_distance = curr_distance;
        }

        Ok(return_points)
    }

    /// Distance from point to Poincaré section
    fn distance_to_section(
        &self,
        point: &Array1<f64>,
        section_normal: &Array1<f64>,
        section_point: &Array1<f64>,
    ) -> f64 {
        let relative_pos = point - section_point;
        relative_pos.dot(section_normal)
    }

    /// Refine section crossing using bisection
    fn refine_section_crossing<F>(
        &self,
        system: &F,
        current_state: &Array1<f64>,
        dt: f64,
        section_normal: &Array1<f64>,
        section_point: &Array1<f64>,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        // Simple bisection refinement
        let derivative = system(current_state);
        let prev_state = current_state - &(derivative * dt);

        let mut left = prev_state;
        let mut right = current_state.clone();

        for _ in 0..10 {
            let mid = (&left + &right) * 0.5;
            let mid_distance = self.distance_to_section(&mid, section_normal, section_point);

            if mid_distance.abs() < 1e-10 {
                return Ok(mid);
            }

            let left_distance = self.distance_to_section(&left, section_normal, section_point);

            if left_distance * mid_distance < 0.0 {
                right = mid;
            } else {
                left = mid;
            }
        }

        Ok((&left + &right) * 0.5)
    }

    /// Analyze return map for periodicity
    fn analyze_return_map_for_periodicity(
        &self,
        return_points: &[Array1<f64>],
    ) -> IntegrateResult<PeriodicOrbit> {
        if return_points.len() < 3 {
            return Err(IntegrateError::ComputationError(
                "Insufficient return points for periodicity analysis".to_string(),
            ));
        }

        let tolerance = 1e-6;

        // Look for approximate returns
        for period in 1..std::cmp::min(return_points.len() / 2, 10) {
            let mut is_periodic = true;
            let mut max_error: f64 = 0.0;

            for i in 0..(return_points.len() - period) {
                let error = (&return_points[i] - &return_points[i + period])
                    .iter()
                    .map(|&x| x * x)
                    .sum::<f64>()
                    .sqrt();

                max_error = max_error.max(error);

                if error > tolerance {
                    is_periodic = false;
                    break;
                }
            }

            if is_periodic {
                // Estimate the period in time (rough approximation)
                let estimated_period = period as f64 * 2.0 * std::f64::consts::PI;

                return Ok(PeriodicOrbit {
                    representative_point: return_points[0].clone(),
                    period: estimated_period,
                    stability: StabilityType::Stable, // Would need proper analysis
                    floquet_multipliers: vec![],      // Would need computation
                });
            }
        }

        Err(IntegrateError::ComputationError(
            "No periodic behavior detected in return map".to_string(),
        ))
    }

    /// Fourier analysis for periodic orbit detection
    fn fourier_analysis_periodic_orbits<F>(
        &self,
        system: &F,
        domain: &[(f64, f64)],
    ) -> IntegrateResult<Vec<PeriodicOrbit>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let mut periodic_orbits = Vec::new();

        // Generate initial points
        let n_trajectories = 5;
        let mut initial_points = Vec::new();
        self.generate_grid_points(domain, n_trajectories, &mut initial_points);

        for initial_point in &initial_points {
            if let Ok(orbit) = self.fourier_analysis_single_trajectory(system, initial_point) {
                periodic_orbits.push(orbit);
            }
        }

        Ok(periodic_orbits)
    }

    /// Fourier analysis of a single trajectory
    fn fourier_analysis_single_trajectory<F>(
        &self,
        system: &F,
        initial_point: &Array1<f64>,
    ) -> IntegrateResult<PeriodicOrbit>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let dt = 0.01;
        let total_time = 20.0;
        let n_steps = (total_time / dt) as usize;

        // Integrate trajectory
        let mut trajectory = Vec::new();
        let mut state = initial_point.clone();

        for _ in 0..n_steps {
            trajectory.push(state.clone());
            let derivative = system(&state);
            state += &(derivative * dt);
        }

        // Simple frequency analysis (detect dominant frequency)
        if let Ok(dominant_period) = self.detect_dominant_period(&trajectory, dt) {
            if dominant_period > 0.0 && dominant_period < total_time {
                return Ok(PeriodicOrbit {
                    representative_point: initial_point.clone(),
                    period: dominant_period,
                    stability: StabilityType::Stable, // Would need proper analysis
                    floquet_multipliers: vec![],      // Would need computation
                });
            }
        }

        Err(IntegrateError::ComputationError(
            "No periodic behavior detected via Fourier analysis".to_string(),
        ))
    }

    /// Detect dominant period using autocorrelation
    fn detect_dominant_period(&self, trajectory: &[Array1<f64>], dt: f64) -> IntegrateResult<f64> {
        if trajectory.len() < 100 {
            return Err(IntegrateError::ComputationError(
                "Trajectory too short for period detection".to_string(),
            ));
        }

        // Use first component for period detection
        let signal: Vec<f64> = trajectory.iter().map(|state| state[0]).collect();

        // Autocorrelation approach
        let max_lag = std::cmp::min(signal.len() / 4, 500);
        let mut autocorr = vec![0.0; max_lag];

        let mean = signal.iter().sum::<f64>() / signal.len() as f64;
        let variance =
            signal.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / signal.len() as f64;

        if variance < 1e-12 {
            return Err(IntegrateError::ComputationError(
                "Signal has zero variance".to_string(),
            ));
        }

        for lag in 1..max_lag {
            let mut correlation = 0.0;
            let count = signal.len() - lag;

            for i in 0..count {
                correlation += (signal[i] - mean) * (signal[i + lag] - mean);
            }

            autocorr[lag] = correlation / (count as f64 * variance);
        }

        // Find the first significant peak after lag = 0
        let mut max_corr = 0.0;
        let mut period_lag = 0;

        for lag in 10..max_lag {
            if autocorr[lag] > max_corr && autocorr[lag] > 0.5 {
                // Check if this is a local maximum
                if lag > 0
                    && lag < max_lag - 1
                    && autocorr[lag] > autocorr[lag - 1]
                    && autocorr[lag] > autocorr[lag + 1]
                {
                    max_corr = autocorr[lag];
                    period_lag = lag;
                }
            }
        }

        if period_lag > 0 {
            Ok(period_lag as f64 * dt)
        } else {
            Err(IntegrateError::ComputationError(
                "No dominant period detected".to_string(),
            ))
        }
    }

    /// Remove duplicate periodic orbits based on spatial proximity
    fn remove_duplicate_periodic_orbits(&self, orbits: Vec<PeriodicOrbit>) -> Vec<PeriodicOrbit> {
        let mut filtered: Vec<PeriodicOrbit> = Vec::new();
        let tolerance = 1e-4;

        for orbit in orbits {
            let mut is_duplicate = false;

            for existing in &filtered {
                let distance = (&orbit.representative_point - &existing.representative_point)
                    .iter()
                    .map(|&x| x * x)
                    .sum::<f64>()
                    .sqrt();

                let period_diff = (orbit.period - existing.period).abs();

                if distance < tolerance && period_diff < tolerance {
                    is_duplicate = true;
                    break;
                }
            }

            if !is_duplicate {
                filtered.push(orbit);
            }
        }

        filtered
    }

    /// Solve linear system for shooting method
    fn solve_linear_system_for_shooting(
        &self,
        matrix: &Array2<f64>,
        rhs: &Array1<f64>,
    ) -> IntegrateResult<Array1<f64>> {
        let n = matrix.nrows();
        if n != matrix.ncols() || n != rhs.len() {
            return Err(IntegrateError::ComputationError(
                "Inconsistent matrix dimensions in shooting method".to_string(),
            ));
        }

        let mut augmented = Array2::<f64>::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
            }
            augmented[[i, n]] = rhs[i];
        }

        // Gaussian elimination
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            for i in (k + 1)..n {
                if augmented[[i, k]].abs() > augmented[[max_row, k]].abs() {
                    max_row = i;
                }
            }

            // Swap rows
            if max_row != k {
                for j in k..=n {
                    let temp = augmented[[k, j]];
                    augmented[[k, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }

            // Check for singularity
            if augmented[[k, k]].abs() < 1e-14 {
                return Err(IntegrateError::ComputationError(
                    "Singular matrix in shooting method".to_string(),
                ));
            }

            // Eliminate
            for i in (k + 1)..n {
                let factor = augmented[[i, k]] / augmented[[k, k]];
                for j in k..=n {
                    augmented[[i, j]] -= factor * augmented[[k, j]];
                }
            }
        }

        // Back substitution
        let mut solution = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            solution[i] = augmented[[i, n]];
            for j in (i + 1)..n {
                solution[i] -= augmented[[i, j]] * solution[j];
            }
            solution[i] /= augmented[[i, i]];
        }

        Ok(solution)
    }

    /// Compute Lyapunov exponents
    fn compute_lyapunov_exponents<F>(&self, system: &F) -> IntegrateResult<Option<Array1<f64>>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync + Clone,
    {
        // For systems with dimension 1-10, compute Lyapunov exponents
        if self.dimension == 0 || self.dimension > 10 {
            return Ok(None);
        }

        // Create initial state in the center of the domain
        // (we'd ideally use an attractor, but this is a reasonable default)
        let initial_state = Array1::zeros(self.dimension);

        // Use adaptive time step based on system dimension
        let dt = match self.dimension {
            1 => 0.01,
            2 => 0.005,
            3 => 0.002,
            4..=6 => 0.001,
            _ => 0.0005,
        };

        // Calculate number of exponents to compute (typically all for small systems)
        let n_exponents = if self.dimension <= 4 {
            self.dimension
        } else {
            // For higher dimensions, compute only the largest few exponents
            std::cmp::min(4, self.dimension)
        };

        let calculator = advanced::LyapunovCalculator::new(n_exponents, dt);

        // Use integration time that scales with system complexity
        let integration_time = self.integration_time * (self.dimension as f64).sqrt();

        // Clone the system function to satisfy trait bounds
        let system_clone = system.clone();
        let system_wrapper = move |state: &Array1<f64>| system_clone(state);

        match calculator.calculate_lyapunov_exponents(
            system_wrapper,
            &initial_state,
            integration_time,
        ) {
            Ok(exponents) => {
                // Filter out numerical artifacts (very small exponents close to machine precision)
                let filtered_exponents = exponents.mapv(|x| if x.abs() < 1e-12 { 0.0 } else { x });
                Ok(Some(filtered_exponents))
            }
            Err(e) => {
                // If Lyapunov computation fails, it's not critical - return None
                eprintln!("Warning: Lyapunov exponent computation failed: {e:?}");
                Ok(None)
            }
        }
    }

    /// Analyze basins of attraction for 2D systems
    fn analyze_basins<F>(
        &self,
        system: &F,
        domain: &[(f64, f64)],
        fixed_points: &[FixedPoint],
    ) -> IntegrateResult<BasinAnalysis>
    where
        F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync,
    {
        if self.dimension != 2 || domain.len() != 2 {
            return Err(IntegrateError::ValueError(
                "Basin analysis only implemented for 2D systems".to_string(),
            ));
        }

        let grid_size = self.basin_grid_size;
        let mut grid_points = Array2::zeros((grid_size * grid_size, 2));
        let mut attractor_indices = Array2::<i32>::zeros((grid_size, grid_size));

        let dx = (domain[0].1 - domain[0].0) / (grid_size - 1) as f64;
        let dy = (domain[1].1 - domain[1].0) / (grid_size - 1) as f64;

        // Generate grid and integrate each point
        for i in 0..grid_size {
            for j in 0..grid_size {
                let x = domain[0].0 + i as f64 * dx;
                let y = domain[1].0 + j as f64 * dy;

                grid_points[[i * grid_size + j, 0]] = x;
                grid_points[[i * grid_size + j, 1]] = y;

                // Integrate trajectory and find which attractor it converges to
                let initial_state = Array1::from_vec(vec![x, y]);
                let final_state = self.integrate_trajectory(system, &initial_state)?;

                // Find closest fixed point
                let mut closest_attractor = -1;
                let mut min_distance = f64::INFINITY;

                for (idx, fp) in fixed_points.iter().enumerate() {
                    if fp.stability == StabilityType::Stable {
                        let distance = (&final_state - &fp.location)
                            .iter()
                            .map(|&x| x * x)
                            .sum::<f64>()
                            .sqrt();

                        if distance < min_distance && distance < 0.1 {
                            min_distance = distance;
                            closest_attractor = idx as i32;
                        }
                    }
                }

                attractor_indices[[i, j]] = closest_attractor;
            }
        }

        // Extract stable attractors
        let attractors = fixed_points
            .iter()
            .filter(|fp| fp.stability == StabilityType::Stable)
            .map(|fp| fp.location.clone())
            .collect();

        Ok(BasinAnalysis {
            grid_points,
            attractor_indices,
            attractors,
        })
    }

    /// Integrate trajectory to find final state
    fn integrate_trajectory<F>(
        &self,
        system: &F,
        initial_state: &Array1<f64>,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        // Simple Euler integration
        let dt = 0.01;
        let n_steps = (self.integration_time / dt) as usize;
        let mut state = initial_state.clone();

        for _ in 0..n_steps {
            let derivative = system(&state);
            state += &(derivative * dt);
        }

        Ok(state)
    }
}
