//! Structure-preserving integrators
//!
//! This module provides integrators that preserve various geometric structures
//! such as energy, momentum, symplectic structure, and other invariants.

use crate::error::{IntegrateResult, IntegrateResult as Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

// Type aliases for complex function types
type HamiltonianFn = Box<dyn Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64>;
type ConstraintFn = Box<dyn Fn(&ArrayView1<f64>) -> Array1<f64>>;
type ConstraintJacobianFn = Box<dyn Fn(&ArrayView1<f64>) -> Array2<f64>>;
type GradientFn = dyn Fn(&ArrayView1<f64>) -> Array1<f64>;
type ForceFn = Box<dyn Fn(&ArrayView1<f64>) -> Array1<f64>>;
type KineticFn = Box<dyn Fn(&ArrayView1<f64>) -> f64>;
type PotentialFn = Box<dyn Fn(&ArrayView1<f64>) -> f64>;
type StiffnessFn = Box<dyn Fn(&ArrayView1<f64>) -> Array1<f64>>;

/// Trait for geometric invariants
pub trait GeometricInvariant {
    /// Evaluate the invariant quantity
    fn evaluate(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>, t: f64) -> f64;

    /// Name of the invariant (for debugging)
    fn name(&self) -> &'static str;
}

/// General structure-preserving integrator
pub struct StructurePreservingIntegrator {
    /// Time step
    pub dt: f64,
    /// Integration method
    pub method: StructurePreservingMethod,
    /// Invariants to preserve
    pub invariants: Vec<Box<dyn GeometricInvariant>>,
    /// Tolerance for invariant preservation
    pub tol: f64,
}

/// Available structure-preserving methods
#[derive(Debug, Clone, Copy)]
pub enum StructurePreservingMethod {
    /// Discrete gradient method
    DiscreteGradient,
    /// Average vector field method
    AverageVectorField,
    /// Energy-momentum method
    EnergyMomentum,
    /// Variational integrator
    Variational,
}

impl StructurePreservingIntegrator {
    /// Create a new structure-preserving integrator
    pub fn new(dt: f64, method: StructurePreservingMethod) -> Self {
        Self {
            dt,
            method,
            invariants: Vec::new(),
            tol: 1e-10,
        }
    }

    /// Add an invariant to preserve
    pub fn add_invariant(&mut self, invariant: Box<dyn GeometricInvariant>) -> &mut Self {
        self.invariants.push(invariant);
        self
    }

    /// Check invariant preservation
    pub fn check_invariants(
        &self,
        x0: &ArrayView1<f64>,
        v0: &ArrayView1<f64>,
        x1: &ArrayView1<f64>,
        v1: &ArrayView1<f64>,
        t: f64,
    ) -> Vec<(String, f64)> {
        let mut errors = Vec::new();

        for invariant in &self.invariants {
            let i0 = invariant.evaluate(x0, v0, t);
            let i1 = invariant.evaluate(x1, v1, t + self.dt);
            let error = (i1 - i0).abs() / (1.0 + i0.abs());
            errors.push((invariant.name().to_string(), error));
        }

        errors
    }
}

/// Energy-preserving integrator for Hamiltonian systems
pub struct EnergyPreservingMethod {
    /// Hamiltonian function
    hamiltonian: HamiltonianFn,
    /// Dimension
    dim: usize,
}

impl EnergyPreservingMethod {
    /// Create a new energy-preserving integrator
    pub fn new(hamiltonian: HamiltonianFn, dim: usize) -> Self {
        Self { hamiltonian, dim }
    }

    /// Discrete gradient method
    pub fn discrete_gradient_step(
        &self,
        q: &ArrayView1<f64>,
        p: &ArrayView1<f64>,
        dt: f64,
    ) -> IntegrateResult<(Array1<f64>, Array1<f64>)> {
        let h = 1e-8;

        // Compute gradients using finite differences
        let mut grad_q = Array1::zeros(self.dim);
        let mut grad_p = Array1::zeros(self.dim);

        for i in 0..self.dim {
            let mut q_plus = q.to_owned();
            let mut q_minus = q.to_owned();
            q_plus[i] += h;
            q_minus[i] -= h;
            grad_q[i] = ((self.hamiltonian)(&q_plus.view(), p)
                - (self.hamiltonian)(&q_minus.view(), p))
                / (2.0 * h);

            let mut p_plus = p.to_owned();
            let mut p_minus = p.to_owned();
            p_plus[i] += h;
            p_minus[i] -= h;
            grad_p[i] = ((self.hamiltonian)(q, &p_plus.view())
                - (self.hamiltonian)(q, &p_minus.view()))
                / (2.0 * h);
        }

        // Implicit midpoint with discrete gradient
        let q_mid = q + &grad_p * (dt / 2.0);
        let p_mid = p - &grad_q * (dt / 2.0);

        // Compute discrete gradient at midpoint
        let mut grad_q_mid = Array1::zeros(self.dim);
        let mut grad_p_mid = Array1::zeros(self.dim);

        for i in 0..self.dim {
            let mut q_plus = q_mid.clone();
            let mut q_minus = q_mid.clone();
            q_plus[i] += h;
            q_minus[i] -= h;
            grad_q_mid[i] = ((self.hamiltonian)(&q_plus.view(), &p_mid.view())
                - (self.hamiltonian)(&q_minus.view(), &p_mid.view()))
                / (2.0 * h);

            let mut p_plus = p_mid.clone();
            let mut p_minus = p_mid.clone();
            p_plus[i] += h;
            p_minus[i] -= h;
            grad_p_mid[i] = ((self.hamiltonian)(&q_mid.view(), &p_plus.view())
                - (self.hamiltonian)(&q_mid.view(), &p_minus.view()))
                / (2.0 * h);
        }

        let q_new = q + &grad_p_mid * dt;
        let p_new = p - &grad_q_mid * dt;

        Ok((q_new, p_new))
    }

    /// Average vector field method
    pub fn average_vector_field_step(
        &self,
        q: &ArrayView1<f64>,
        p: &ArrayView1<f64>,
        dt: f64,
    ) -> IntegrateResult<(Array1<f64>, Array1<f64>)> {
        // Simplified AVF - uses quadrature to average the vector field
        let _n_quad = 3; // Number of quadrature points
        let weights = [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0]; // Simpson's rule weights
        let nodes = [0.0, 0.5, 1.0]; // Quadrature nodes

        let mut q_avg = Array1::zeros(self.dim);
        let mut p_avg = Array1::zeros(self.dim);

        for (&w, &s) in weights.iter().zip(nodes.iter()) {
            // Linear interpolation
            let q_s = q * (1.0 - s) + &(q + &(p * dt)) * s;
            let p_s = p * (1.0 - s) + &(p - &(q * dt)) * s; // Simplified

            // Compute gradients at interpolated point
            let h = 1e-8;
            for j in 0..self.dim {
                let mut q_plus = q_s.clone();
                let mut q_minus = q_s.clone();
                q_plus[j] += h;
                q_minus[j] -= h;

                p_avg[j] += w * (self.hamiltonian)(&q_plus.view(), &p_s.view())
                    - (self.hamiltonian)(&q_minus.view(), &p_s.view()) / (2.0 * h);

                let mut p_plus = p_s.clone();
                let mut p_minus = p_s.clone();
                p_plus[j] += h;
                p_minus[j] -= h;

                q_avg[j] += w * (self.hamiltonian)(&q_s.view(), &p_plus.view())
                    - (self.hamiltonian)(&q_s.view(), &p_minus.view()) / (2.0 * h);
            }
        }

        let q_new = q + &q_avg * dt;
        let p_new = p - &p_avg * dt;

        Ok((q_new, p_new))
    }
}

/// Momentum-preserving integrator
pub struct MomentumPreservingMethod {
    /// System dimension
    #[allow(dead_code)]
    dim: usize,
    /// Force function
    force: ForceFn,
    /// Mass matrix (diagonal)
    mass: Array1<f64>,
}

impl MomentumPreservingMethod {
    /// Create a new momentum-preserving integrator
    pub fn new(dim: usize, force: ForceFn, mass: Array1<f64>) -> Self {
        Self { dim, force, mass }
    }

    /// Integrate one step preserving total momentum
    pub fn step(
        &self,
        x: &ArrayView1<f64>,
        v: &ArrayView1<f64>,
        dt: f64,
    ) -> IntegrateResult<(Array1<f64>, Array1<f64>)> {
        // Compute forces
        let f = (self.force)(x);

        // Check momentum conservation (for internal forces, total force should be zero)
        let total_force: f64 = f.sum();
        if total_force.abs() > 1e-10 {
            // Apply momentum correction
            let f_corrected = &f - total_force / self.dim as f64;

            // Velocity Verlet with corrected forces
            let a = &f_corrected / &self.mass;
            let x_new = x + v * dt + &a * (dt * dt / 2.0);

            let f_new = (self.force)(&x_new.view());
            let f_new_corrected = &f_new - f_new.sum() / self.dim as f64;
            let a_new = &f_new_corrected / &self.mass;

            let v_new = v + (&a + &a_new) * (dt / 2.0);

            Ok((x_new, v_new))
        } else {
            // Standard velocity Verlet
            let a = &f / &self.mass;
            let x_new = x + v * dt + &a * (dt * dt / 2.0);

            let f_new = (self.force)(&x_new.view());
            let a_new = &f_new / &self.mass;

            let v_new = v + (&a + &a_new) * (dt / 2.0);

            Ok((x_new, v_new))
        }
    }
}

/// Conservation checker for verifying invariant preservation
pub struct ConservationChecker;

impl ConservationChecker {
    /// Check energy conservation
    pub fn check_energy<H>(trajectory: &[(Array1<f64>, Array1<f64>)], hamiltonian: H) -> Vec<f64>
    where
        H: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
    {
        trajectory
            .iter()
            .map(|(q, p)| hamiltonian(&q.view(), &p.view()))
            .collect()
    }

    /// Check momentum conservation
    pub fn check_momentum(
        trajectory: &[(Array1<f64>, Array1<f64>)],
        masses: &ArrayView1<f64>,
    ) -> Vec<Array1<f64>> {
        trajectory.iter().map(|(_, v)| v * masses).collect()
    }

    /// Check angular momentum conservation
    pub fn check_angular_momentum(
        trajectory: &[(Array1<f64>, Array1<f64>)],
        masses: &ArrayView1<f64>,
    ) -> Vec<Array1<f64>> {
        trajectory
            .iter()
            .map(|(x, v)| {
                // For 3D systems, compute L = r × p
                if x.len() == 3 && v.len() == 3 {
                    let px = v[0] * masses[0];
                    let py = v[1] * masses[1];
                    let pz = v[2] * masses[2];

                    Array1::from_vec(vec![
                        x[1] * pz - x[2] * py,
                        x[2] * px - x[0] * pz,
                        x[0] * py - x[1] * px,
                    ])
                } else {
                    Array1::zeros(3)
                }
            })
            .collect()
    }

    /// Compute relative error in conservation
    pub fn relative_error(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let initial = values[0];
        let max_deviation = values
            .iter()
            .map(|&v| (v - initial).abs())
            .fold(0.0, f64::max);

        max_deviation / (1.0 + initial.abs())
    }
}

/// Splitting method for separable Hamiltonians
pub struct SplittingIntegrator {
    /// Kinetic energy part T(p)
    kinetic: KineticFn,
    /// Potential energy part V(q)
    potential: PotentialFn,
    /// System dimension
    #[allow(dead_code)]
    dim: usize,
    /// Splitting coefficients
    coefficients: Vec<(f64, f64)>,
}

impl SplittingIntegrator {
    /// Create a new splitting integrator with Strang splitting
    pub fn strang(kinetic: KineticFn, potential: PotentialFn, dim: usize) -> Self {
        let coefficients = vec![(0.5, 1.0), (0.5, 0.0)];
        Self {
            kinetic,
            potential,
            dim,
            coefficients,
        }
    }

    /// Create with Yoshida 4th order coefficients
    pub fn yoshida4(kinetic: KineticFn, potential: PotentialFn, dim: usize) -> Self {
        let x1 = 1.0 / (2.0 - 2.0_f64.powf(1.0 / 3.0));
        let x0 = -2.0_f64.powf(1.0 / 3.0) * x1;

        let coefficients = vec![
            (x1 / 2.0, x1),
            ((x0 + x1) / 2.0, x0),
            ((x0 + x1) / 2.0, x1),
            (x1 / 2.0, 0.0),
        ];

        Self {
            kinetic,
            potential,
            dim,
            coefficients,
        }
    }

    /// Perform one splitting step
    pub fn step(
        &self,
        q: &ArrayView1<f64>,
        p: &ArrayView1<f64>,
        dt: f64,
    ) -> IntegrateResult<(Array1<f64>, Array1<f64>)> {
        let mut q_current = q.to_owned();
        let mut p_current = p.to_owned();

        for &(a, b) in &self.coefficients {
            // Kick: p' = p - a*dt*∇V(q)
            if a != 0.0 {
                let grad_v = self.gradient_potential(&q_current.view());
                p_current = p_current - grad_v * (a * dt);
            }

            // Drift: q' = q + b*dt*∇T(p')
            if b != 0.0 {
                let grad_t = self.gradient_kinetic(&p_current.view());
                q_current = q_current + grad_t * (b * dt);
            }
        }

        Ok((q_current, p_current))
    }

    /// Compute gradient of kinetic energy
    fn gradient_kinetic(&self, p: &ArrayView1<f64>) -> Array1<f64> {
        let h = 1e-8;
        let mut grad = Array1::zeros(self.dim);

        for i in 0..self.dim {
            let mut p_plus = p.to_owned();
            let mut p_minus = p.to_owned();
            p_plus[i] += h;
            p_minus[i] -= h;

            grad[i] =
                ((self.kinetic)(&p_plus.view()) - (self.kinetic)(&p_minus.view())) / (2.0 * h);
        }

        grad
    }

    /// Compute gradient of potential energy
    fn gradient_potential(&self, q: &ArrayView1<f64>) -> Array1<f64> {
        let h = 1e-8;
        let mut grad = Array1::zeros(self.dim);

        for i in 0..self.dim {
            let mut q_plus = q.to_owned();
            let mut q_minus = q.to_owned();
            q_plus[i] += h;
            q_minus[i] -= h;

            grad[i] =
                ((self.potential)(&q_plus.view()) - (self.potential)(&q_minus.view())) / (2.0 * h);
        }

        grad
    }
}

/// Energy-momentum conserving integrator for nonlinear elastodynamics
pub struct EnergyMomentumIntegrator {
    /// Mass matrix
    mass: Array1<f64>,
    /// Stiffness function
    stiffness: StiffnessFn,
    /// System dimension
    #[allow(dead_code)]
    dim: usize,
}

impl EnergyMomentumIntegrator {
    /// Create a new energy-momentum integrator
    pub fn new(mass: Array1<f64>, stiffness: StiffnessFn) -> Self {
        let dim = mass.len();
        Self {
            mass,
            stiffness,
            dim,
        }
    }

    /// Integrate one step with energy-momentum conservation
    pub fn step(
        &self,
        u: &ArrayView1<f64>,
        v: &ArrayView1<f64>,
        dt: f64,
    ) -> IntegrateResult<(Array1<f64>, Array1<f64>)> {
        // Predict displacement
        let u_pred = u + v * dt;

        // Compute average internal force
        let f0 = (self.stiffness)(u);
        let f1 = (self.stiffness)(&u_pred.view());
        let f_avg = (&f0 + &f1) / 2.0;

        // Algorithmic acceleration
        let a_alg = &f_avg / &self.mass;

        // Update
        let u_new = u + v * dt + &a_alg * (dt * dt / 2.0);
        let v_new = v + &a_alg * dt;

        // Energy-momentum correction
        let momentum_error: f64 = (&v_new * &self.mass).sum() - (v.to_owned() * &self.mass).sum();
        if momentum_error.abs() > 1e-12 {
            let v_corrected = &v_new - momentum_error / self.mass.sum();
            Ok((u_new, v_corrected))
        } else {
            Ok((u_new, v_new))
        }
    }
}

/// Störmer-Verlet method for constrained systems
pub struct ConstrainedIntegrator {
    /// Constraint function g(q) = 0
    constraints: ConstraintFn,
    /// Constraint Jacobian
    constraint_jacobian: ConstraintJacobianFn,
    /// System dimension
    #[allow(dead_code)]
    dim: usize,
    /// Number of constraints
    #[allow(dead_code)]
    n_constraints: usize,
    /// Tolerance for constraint satisfaction
    tol: f64,
}

impl ConstrainedIntegrator {
    /// Create a new constrained integrator
    pub fn new(
        constraints: ConstraintFn,
        constraint_jacobian: ConstraintJacobianFn,
        dim: usize,
        n_constraints: usize,
    ) -> Self {
        Self {
            constraints,
            constraint_jacobian,
            dim,
            n_constraints,
            tol: 1e-10,
        }
    }

    /// SHAKE algorithm for position constraints
    pub fn shake_step(
        &self,
        q: &ArrayView1<f64>,
        p: &ArrayView1<f64>,
        dt: f64,
        force: &Array1<f64>,
    ) -> IntegrateResult<(Array1<f64>, Array1<f64>)> {
        // Unconstrained step
        let q_tilde = q + p * dt;
        let p_tilde = p + force * dt;

        // SHAKE iteration for position constraints
        let mut q_new = q_tilde.clone();
        let mut lambda;

        for _ in 0..100 {
            let g = (self.constraints)(&q_new.view());
            if g.mapv(f64::abs).sum() < self.tol {
                break;
            }

            let g_matrix = (self.constraint_jacobian)(&q_new.view());

            // Solve for Lagrange multipliers
            // G * G^T * λ = -g
            let ggt = g_matrix.dot(&g_matrix.t());
            lambda = self.solve_linear_system(&ggt, &(-&g))?;

            // Update position
            let correction = g_matrix.t().dot(&lambda);
            q_new = &q_new + &correction * dt * dt;
        }

        // RATTLE for velocity constraints
        let g_new = (self.constraint_jacobian)(&q_new.view());
        let gv = g_new.dot(&p_tilde);

        // Solve G * G^T * μ = -G * v
        let ggt = g_new.dot(&g_new.t());
        let mu = self.solve_linear_system(&ggt, &(-&gv))?;

        let p_correction = g_new.t().dot(&mu);
        let p_new = &p_tilde + &p_correction;

        Ok((q_new, p_new))
    }

    /// Simple linear system solver (for small systems)
    fn solve_linear_system(
        &self,
        a: &scirs2_core::ndarray::Array2<f64>,
        b: &Array1<f64>,
    ) -> IntegrateResult<Array1<f64>> {
        // LU decomposition would be more robust
        let n = b.len();
        let mut x = Array1::zeros(n);

        // Simplified Gaussian elimination
        let mut a_copy = a.clone();
        let mut b_copy = b.clone();

        for i in 0..n {
            // Pivot
            let pivot = a_copy[[i, i]];
            if pivot.abs() < 1e-14 {
                return Err(crate::error::IntegrateError::ComputationError(
                    "Singular constraint matrix".to_string(),
                ));
            }

            // Eliminate
            for j in (i + 1)..n {
                let factor = a_copy[[j, i]] / pivot;
                for k in i..n {
                    a_copy[[j, k]] -= factor * a_copy[[i, k]];
                }
                b_copy[j] -= factor * b_copy[i];
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            let mut sum = b_copy[i];
            for j in (i + 1)..n {
                sum -= a_copy[[i, j]] * x[j];
            }
            x[i] = sum / a_copy[[i, i]];
        }

        Ok(x)
    }
}

/// Multi-symplectic integrator for PDEs
pub struct MultiSymplecticIntegrator {
    /// Spatial dimension
    #[allow(dead_code)]
    spatial_dim: usize,
    /// Number of fields
    #[allow(dead_code)]
    n_fields: usize,
    /// Symplectic structure matrices
    k: scirs2_core::ndarray::Array2<f64>,
    l: scirs2_core::ndarray::Array2<f64>,
}

impl MultiSymplecticIntegrator {
    /// Create a new multi-symplectic integrator
    pub fn new(
        spatial_dim: usize,
        n_fields: usize,
        k: scirs2_core::ndarray::Array2<f64>,
        l: scirs2_core::ndarray::Array2<f64>,
    ) -> Self {
        Self {
            spatial_dim,
            n_fields,
            k,
            l,
        }
    }

    /// Preissman box scheme
    pub fn preissman_step(
        &self,
        z: &scirs2_core::ndarray::Array2<f64>,
        s: &GradientFn,
        dt: f64,
        dx: f64,
    ) -> IntegrateResult<scirs2_core::ndarray::Array2<f64>> {
        let (nx_, _) = z.dim();
        let mut z_new = z.clone();

        // Iterate through spatial grid
        for i in 1..nx_ {
            // Box average
            let z_avg = (&z.row(i - 1) + &z.row(i) + z_new.row(i - 1) + z_new.row(i)) / 4.0;

            // Compute gradient of S
            let grad_s = s(&z_avg.view());

            // Multi-symplectic conservation law
            // K(z_t) + L(z_x) = ∇S(z)
            let z_t = (&z_new.row(i) - &z.row(i) + z_new.row(i - 1) - z.row(i - 1)) / (2.0 * dt);
            let z_x = (&z_new.row(i) + &z.row(i) - z_new.row(i - 1) - z.row(i - 1)) / (2.0 * dx);

            let residual = self.k.dot(&z_t) + self.l.dot(&z_x) - grad_s;

            // Newton iteration (simplified)
            let current_row = z_new.row(i).to_owned();
            let update = &current_row - &residual * 0.5;
            z_new.row_mut(i).assign(&update);
        }

        Ok(z_new)
    }
}

/// Example invariants
pub mod invariants {
    use super::*;

    /// Energy invariant for Hamiltonian systems
    pub struct EnergyInvariant {
        hamiltonian: HamiltonianFn,
    }

    impl EnergyInvariant {
        pub fn new(hamiltonian: HamiltonianFn) -> Self {
            Self { hamiltonian }
        }
    }

    impl GeometricInvariant for EnergyInvariant {
        fn evaluate(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>, t: f64) -> f64 {
            (self.hamiltonian)(x, v)
        }

        fn name(&self) -> &'static str {
            "Energy"
        }
    }

    /// Linear momentum invariant
    pub struct LinearMomentumInvariant {
        masses: Array1<f64>,
        component: usize,
    }

    impl LinearMomentumInvariant {
        pub fn new(masses: Array1<f64>, component: usize) -> Self {
            Self { masses, component }
        }
    }

    impl GeometricInvariant for LinearMomentumInvariant {
        fn evaluate(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>, t: f64) -> f64 {
            v[self.component] * self.masses[self.component]
        }

        fn name(&self) -> &'static str {
            "Linear Momentum"
        }
    }

    /// Angular momentum invariant (for 2D systems)
    pub struct AngularMomentumInvariant2D {
        masses: Array1<f64>,
    }

    impl AngularMomentumInvariant2D {
        pub fn new(masses: Array1<f64>) -> Self {
            Self { masses }
        }
    }

    impl GeometricInvariant for AngularMomentumInvariant2D {
        fn evaluate(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>, t: f64) -> f64 {
            // L = m(xv_y - yv_x) for 2D
            let n = x.len() / 2;
            let mut l = 0.0;

            for i in 0..n {
                let xi = x[2 * i];
                let yi = x[2 * i + 1];
                let vxi = v[2 * i];
                let vyi = v[2 * i + 1];
                l += self.masses[i] * (xi * vyi - yi * vxi);
            }

            l
        }

        fn name(&self) -> &'static str {
            "Angular Momentum"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EnergyPreservingMethod, MomentumPreservingMethod};
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::{Array1, ArrayView1};

    #[test]
    fn test_energy_preservation() {
        // Simple harmonic oscillator: H = p²/2m + kx²/2
        let m = 1.0;
        let k = 1.0;
        let hamiltonian = Box::new(move |q: &ArrayView1<f64>, p: &ArrayView1<f64>| {
            p[0] * p[0] / (2.0 * m) + k * q[0] * q[0] / 2.0
        });

        let integrator = EnergyPreservingMethod::new(hamiltonian.clone(), 1);
        let q0 = Array1::from_vec(vec![1.0]);
        let p0 = Array1::from_vec(vec![0.0]);

        let initial_energy = hamiltonian(&q0.view(), &p0.view());

        // Integrate for one period
        let dt = 0.1;
        let n_steps = 63; // approximately 2π

        let mut q = q0.clone();
        let mut p = p0.clone();

        for _ in 0..n_steps {
            let (q_new, p_new) = integrator
                .discrete_gradient_step(&q.view(), &p.view(), dt)
                .unwrap();
            q = q_new;
            p = p_new;
        }

        let final_energy = hamiltonian(&q.view(), &p.view());
        assert_relative_eq!(initial_energy, final_energy, epsilon = 1e-3);
    }

    #[test]
    fn test_momentum_preservation() {
        // Two-particle system with internal forces
        let dim = 4; // 2 particles in 2D
        let force = Box::new(|x: &ArrayView1<f64>| {
            // Spring force between particles
            let dx = x[2] - x[0];
            let dy = x[3] - x[1];
            let r = (dx * dx + dy * dy).sqrt();
            let f = if r > 0.0 { 1.0 / r } else { 0.0 };

            Array1::from_vec(vec![f * dx, f * dy, -f * dx, -f * dy])
        });

        let mass = Array1::from_vec(vec![1.0, 1.0, 2.0, 2.0]);
        let integrator = MomentumPreservingMethod::new(dim, force, mass.clone());

        let x0 = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0]);
        let v0 = Array1::from_vec(vec![0.0, 0.1, 0.0, -0.05]);

        let initial_momentum: f64 = (&v0 * &mass).sum();

        let (_x1, v1) = integrator.step(&x0.view(), &v0.view(), 0.01).unwrap();
        let final_momentum: f64 = (&v1 * &mass).sum();

        assert_relative_eq!(initial_momentum, final_momentum, epsilon = 1e-12);
    }
}
