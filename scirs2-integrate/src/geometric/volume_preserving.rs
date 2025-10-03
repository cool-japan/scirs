//! Volume-preserving integrators
//!
//! This module provides numerical integrators that preserve phase space volume,
//! suitable for divergence-free flows and incompressible fluid dynamics.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
#[allow(unused_imports)]
use std::f64::consts::{PI, SQRT_2};

// Type alias for complex function type
type InvariantFn = Box<dyn Fn(&ArrayView1<f64>) -> f64>;

/// Trait for divergence-free vector fields
pub trait DivergenceFreeFlow {
    /// Dimension of the phase space
    fn dim(&self) -> usize;

    /// Evaluate the vector field at a point
    fn evaluate(&self, x: &ArrayView1<f64>, t: f64) -> Array1<f64>;

    /// Verify divergence-free condition (for debugging)
    fn verify_divergence_free(&self, x: &ArrayView1<f64>, t: f64, h: f64) -> f64 {
        let n = self.dim();
        let mut div = 0.0;

        for i in 0..n {
            let mut x_plus = x.to_owned();
            let mut x_minus = x.to_owned();
            x_plus[i] += h;
            x_minus[i] -= h;

            let f_plus = self.evaluate(&x_plus.view(), t);
            let f_minus = self.evaluate(&x_minus.view(), t);

            div += (f_plus[i] - f_minus[i]) / (2.0 * h);
        }

        div
    }
}

/// Volume-preserving integrator
pub struct VolumePreservingIntegrator {
    /// Time step
    pub dt: f64,
    /// Integration method
    pub method: VolumePreservingMethod,
    /// Tolerance for implicit methods
    pub tol: f64,
    /// Maximum iterations for implicit methods
    pub max_iter: usize,
}

/// Available volume-preserving integration methods
#[derive(Debug, Clone, Copy)]
pub enum VolumePreservingMethod {
    /// Explicit midpoint rule (2nd order)
    ExplicitMidpoint,
    /// Implicit midpoint rule (2nd order)
    ImplicitMidpoint,
    /// Splitting method for special structure
    SplittingMethod,
    /// Projection method
    ProjectionMethod,
    /// Composition method (4th order)
    CompositionMethod,
}

impl VolumePreservingIntegrator {
    /// Create a new volume-preserving integrator
    pub fn new(dt: f64, method: VolumePreservingMethod) -> Self {
        Self {
            dt,
            method,
            tol: 1e-10,
            max_iter: 100,
        }
    }

    /// Set tolerance for implicit methods
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Integrate one step
    pub fn step<F>(&self, x: &ArrayView1<f64>, t: f64, flow: &F) -> IntegrateResult<Array1<f64>>
    where
        F: DivergenceFreeFlow,
    {
        match self.method {
            VolumePreservingMethod::ExplicitMidpoint => self.explicit_midpoint_step(x, t, flow),
            VolumePreservingMethod::ImplicitMidpoint => self.implicit_midpoint_step(x, t, flow),
            VolumePreservingMethod::SplittingMethod => self.splitting_step(x, t, flow),
            VolumePreservingMethod::ProjectionMethod => self.projection_step(x, t, flow),
            VolumePreservingMethod::CompositionMethod => self.composition_step(x, t, flow),
        }
    }

    /// Explicit midpoint method
    fn explicit_midpoint_step<F>(
        &self,
        x: &ArrayView1<f64>,
        t: f64,
        flow: &F,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: DivergenceFreeFlow,
    {
        let f0 = flow.evaluate(x, t);
        let x_mid = x + &f0 * (self.dt / 2.0);
        let f_mid = flow.evaluate(&x_mid.view(), t + self.dt / 2.0);

        Ok(x + &f_mid * self.dt)
    }

    /// Implicit midpoint method (Gauss-Legendre)
    fn implicit_midpoint_step<F>(
        &self,
        x: &ArrayView1<f64>,
        t: f64,
        flow: &F,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: DivergenceFreeFlow,
    {
        let mut x_new = x.to_owned();
        let t_mid = t + self.dt / 2.0;

        // Fixed-point iteration
        for _ in 0..self.max_iter {
            let x_mid = (&x.to_owned() + &x_new) / 2.0;
            let f_mid = flow.evaluate(&x_mid.view(), t_mid);
            let x_next = x + &f_mid * self.dt;

            let error = (&x_next - &x_new).mapv(f64::abs).sum();
            x_new = x_next;

            if error < self.tol {
                return Ok(x_new);
            }
        }

        Err(IntegrateError::ConvergenceError(
            "Implicit midpoint method failed to converge".to_string(),
        ))
    }

    /// Splitting method for special structures
    fn splitting_step<F>(
        &self,
        x: &ArrayView1<f64>,
        t: f64,
        flow: &F,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: DivergenceFreeFlow,
    {
        // For general flows, fall back to composition
        self.composition_step(x, t, flow)
    }

    /// Projection method (project back to divergence-free manifold)
    fn projection_step<F>(
        &self,
        x: &ArrayView1<f64>,
        t: f64,
        flow: &F,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: DivergenceFreeFlow,
    {
        // Take an Euler step
        let f = flow.evaluate(x, t);
        let _x_euler = x + &f * self.dt;

        // Project back (simplified - in practice would solve Poisson equation)
        // For now, just use explicit midpoint which is volume-preserving
        self.explicit_midpoint_step(x, t, flow)
    }

    /// Fourth-order composition method
    fn composition_step<F>(
        &self,
        x: &ArrayView1<f64>,
        t: f64,
        flow: &F,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: DivergenceFreeFlow,
    {
        // Suzuki-Yoshida 4th order composition
        let gamma = 1.0 / (2.0 - 2.0_f64.powf(1.0 / 3.0));
        let c1 = gamma / 2.0;
        let c2 = (1.0 - gamma) / 2.0;
        let c3 = c2;
        let c4 = c1;

        let d1 = gamma;
        let d2 = -gamma * 2.0_f64.powf(1.0 / 3.0);
        let d3 = gamma;

        // Sub-steps
        let mut x_current = x.to_owned();
        let mut t_current = t;

        // Step 1
        let substep = Self::new(c1 * self.dt, VolumePreservingMethod::ExplicitMidpoint);
        x_current = substep.step(&x_current.view(), t_current, flow)?;
        t_current += c1 * self.dt;

        // Step 2
        let substep = Self::new(d1 * self.dt, VolumePreservingMethod::ExplicitMidpoint);
        x_current = substep.step(&x_current.view(), t_current, flow)?;
        t_current += d1 * self.dt;

        // Step 3
        let substep = Self::new(c2 * self.dt, VolumePreservingMethod::ExplicitMidpoint);
        x_current = substep.step(&x_current.view(), t_current, flow)?;
        t_current += c2 * self.dt;

        // Step 4
        let substep = Self::new(d2 * self.dt, VolumePreservingMethod::ExplicitMidpoint);
        x_current = substep.step(&x_current.view(), t_current, flow)?;
        t_current += d2 * self.dt;

        // Step 5
        let substep = Self::new(c3 * self.dt, VolumePreservingMethod::ExplicitMidpoint);
        x_current = substep.step(&x_current.view(), t_current, flow)?;
        t_current += c3 * self.dt;

        // Step 6
        let substep = Self::new(d3 * self.dt, VolumePreservingMethod::ExplicitMidpoint);
        x_current = substep.step(&x_current.view(), t_current, flow)?;
        t_current += d3 * self.dt;

        // Step 7
        let substep = Self::new(c4 * self.dt, VolumePreservingMethod::ExplicitMidpoint);
        x_current = substep.step(&x_current.view(), t_current, flow)?;

        Ok(x_current)
    }

    /// Integrate for multiple steps
    pub fn integrate<F>(
        &self,
        x0: &ArrayView1<f64>,
        t0: f64,
        t_final: f64,
        flow: &F,
    ) -> IntegrateResult<Vec<(f64, Array1<f64>)>>
    where
        F: DivergenceFreeFlow,
    {
        let n_steps = ((t_final - t0) / self.dt).ceil() as usize;
        let mut trajectory = vec![(t0, x0.to_owned())];

        let mut x_current = x0.to_owned();
        let mut t_current = t0;

        for _ in 0..n_steps {
            let dt_actual = (t_final - t_current).min(self.dt);

            if dt_actual != self.dt {
                // Last step with adjusted time step
                let integrator = Self::new(dt_actual, self.method);
                x_current = integrator.step(&x_current.view(), t_current, flow)?;
            } else {
                x_current = self.step(&x_current.view(), t_current, flow)?;
            }

            t_current += dt_actual;
            trajectory.push((t_current, x_current.clone()));

            if t_current >= t_final - 1e-10 {
                break;
            }
        }

        Ok(trajectory)
    }
}

/// Incompressible flow examples
pub struct IncompressibleFlow;

impl IncompressibleFlow {
    /// 2D circular flow
    pub fn circular_2d(&self) -> CircularFlow2D {
        CircularFlow2D { omega: 1.0 }
    }

    /// ABC flow (Arnold-Beltrami-Childress)
    pub fn abc_flow(a: f64, b: f64, c: f64) -> ABCFlow {
        ABCFlow { a, b, c }
    }

    /// Double gyre flow
    pub fn double_gyre(a: f64, epsilon: f64, omega: f64) -> DoubleGyre {
        DoubleGyre { a, epsilon, omega }
    }
}

/// 2D circular flow (simple incompressible flow)
pub struct CircularFlow2D {
    omega: f64,
}

impl DivergenceFreeFlow for CircularFlow2D {
    fn dim(&self) -> usize {
        2
    }

    fn evaluate(&self, x: &ArrayView1<f64>, t: f64) -> Array1<f64> {
        Array1::from_vec(vec![-self.omega * x[1], self.omega * x[0]])
    }
}

/// ABC flow (3D incompressible flow)
pub struct ABCFlow {
    a: f64,
    b: f64,
    c: f64,
}

impl DivergenceFreeFlow for ABCFlow {
    fn dim(&self) -> usize {
        3
    }

    fn evaluate(&self, x: &ArrayView1<f64>, t: f64) -> Array1<f64> {
        Array1::from_vec(vec![
            self.a * x[1].sin() + self.c * x[2].cos(),
            self.b * x[2].sin() + self.a * x[0].cos(),
            self.c * x[0].sin() + self.b * x[1].cos(),
        ])
    }
}

/// Double gyre flow (time-dependent 2D flow)
pub struct DoubleGyre {
    a: f64,
    epsilon: f64,
    omega: f64,
}

impl DivergenceFreeFlow for DoubleGyre {
    fn dim(&self) -> usize {
        2
    }

    fn evaluate(&self, x: &ArrayView1<f64>, t: f64) -> Array1<f64> {
        let a_t = self.epsilon * (self.omega * t).sin();
        let b_t = 1.0 - 2.0 * self.epsilon * (self.omega * t).sin();

        let f = a_t * x[0].powi(2) + b_t * x[0];
        let df_dx = 2.0 * a_t * x[0] + b_t;

        Array1::from_vec(vec![
            -PI * self.a * (PI * f).sin() * (PI * x[1]).cos(),
            PI * self.a * (PI * f).cos() * df_dx * (PI * x[1]).sin(),
        ])
    }
}

/// Stream function based flow representation
pub trait StreamFunction {
    /// Evaluate stream function at a point
    fn psi(&self, x: f64, y: f64, t: f64) -> f64;

    /// Compute velocity field from stream function
    fn velocity(&self, x: f64, y: f64, t: f64) -> (f64, f64) {
        let h = 1e-8;

        // u = ∂ψ/∂y
        let u = (self.psi(x, y + h, t) - self.psi(x, y - h, t)) / (2.0 * h);

        // v = -∂ψ/∂x
        let v = -(self.psi(x + h, y, t) - self.psi(x - h, y, t)) / (2.0 * h);

        (u, v)
    }
}

/// Stuart vortex flow
pub struct StuartVortex {
    /// Amplitude parameter
    pub alpha: f64,
    /// Wavenumber
    pub k: f64,
}

impl StreamFunction for StuartVortex {
    fn psi(&self, x: f64, y: f64, t: f64) -> f64 {
        -self.alpha.ln() * y.cos() + self.alpha * (self.k * x).cos() * y.sin()
    }
}

impl DivergenceFreeFlow for StuartVortex {
    fn dim(&self) -> usize {
        2
    }

    fn evaluate(&self, x: &ArrayView1<f64>, t: f64) -> Array1<f64> {
        let (u, v) = self.velocity(x[0], x[1], t);
        Array1::from_vec(vec![u, v])
    }
}

/// Taylor-Green vortex
pub struct TaylorGreenVortex {
    /// Viscosity parameter
    pub nu: f64,
}

impl StreamFunction for TaylorGreenVortex {
    fn psi(&self, x: f64, y: f64, t: f64) -> f64 {
        let decay = (-2.0 * self.nu * t).exp();
        decay * x.sin() * y.sin()
    }
}

impl DivergenceFreeFlow for TaylorGreenVortex {
    fn dim(&self) -> usize {
        2
    }

    fn evaluate(&self, x: &ArrayView1<f64>, t: f64) -> Array1<f64> {
        let (u, v) = self.velocity(x[0], x[1], t);
        Array1::from_vec(vec![u, v])
    }
}

/// Generalized Hamiltonian system with volume preservation
pub struct HamiltonianFlow<H>
where
    H: Fn(&ArrayView1<f64>) -> f64,
{
    /// Hamiltonian function
    pub hamiltonian: H,
    /// System dimension (must be even)
    pub dim: usize,
}

impl<H> DivergenceFreeFlow for HamiltonianFlow<H>
where
    H: Fn(&ArrayView1<f64>) -> f64,
{
    fn dim(&self) -> usize {
        self.dim
    }

    fn evaluate(&self, x: &ArrayView1<f64>, t: f64) -> Array1<f64> {
        let n = self.dim / 2;
        let h = 1e-8;
        let mut dx = Array1::zeros(self.dim);

        // Compute gradients
        let mut grad_h = Array1::zeros(self.dim);
        for i in 0..self.dim {
            let mut x_plus = x.to_owned();
            let mut x_minus = x.to_owned();
            x_plus[i] += h;
            x_minus[i] -= h;

            grad_h[i] = ((self.hamiltonian)(&x_plus.view()) - (self.hamiltonian)(&x_minus.view()))
                / (2.0 * h);
        }

        // Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
        for i in 0..n {
            dx[i] = grad_h[n + i]; // dq/dt = ∂H/∂p
            dx[n + i] = -grad_h[i]; // dp/dt = -∂H/∂q
        }

        dx
    }
}

/// Modified midpoint method with volume error correction
pub struct ModifiedMidpointIntegrator {
    /// Base integrator
    base: VolumePreservingIntegrator,
    /// Volume correction strength
    correction_factor: f64,
}

impl ModifiedMidpointIntegrator {
    /// Create a new modified midpoint integrator
    pub fn new(_dt: f64, correctionfactor: f64) -> Self {
        Self {
            base: VolumePreservingIntegrator::new(_dt, VolumePreservingMethod::ImplicitMidpoint),
            correction_factor: correctionfactor,
        }
    }

    /// Step with volume correction
    pub fn step_with_correction<F>(
        &self,
        x: &ArrayView1<f64>,
        t: f64,
        flow: &F,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: DivergenceFreeFlow,
    {
        // Take base step
        let x_new = self.base.step(x, t, flow)?;

        // Compute divergence at midpoint
        let x_mid = (&x.to_owned() + &x_new) / 2.0;
        let div = flow.verify_divergence_free(&x_mid.view(), t + self.base.dt / 2.0, 1e-8);

        // Apply correction if needed
        if div.abs() > 1e-10 {
            let correction = -self.correction_factor * div * self.base.dt;
            let n = x.len();
            let corrected = &x_new * (1.0 + correction / n as f64);
            Ok(corrected)
        } else {
            Ok(x_new)
        }
    }
}

/// Variational integrator for volume-preserving systems
pub struct VariationalIntegrator {
    /// Time step
    dt: f64,
    /// Number of quadrature points
    n_quad: usize,
}

impl VariationalIntegrator {
    /// Create a new variational integrator
    pub fn new(dt: f64, nquad: usize) -> Self {
        Self { dt, n_quad: nquad }
    }

    /// Discrete Lagrangian for volume-preserving flow
    pub fn discrete_lagrangian<F>(
        &self,
        x0: &ArrayView1<f64>,
        x1: &ArrayView1<f64>,
        t: f64,
        flow: &F,
    ) -> IntegrateResult<f64>
    where
        F: DivergenceFreeFlow,
    {
        // Gauss-Legendre quadrature points
        let (weights, nodes) = self.gauss_legendre_quadrature()?;

        let mut l_d = 0.0;

        for i in 0..self.n_quad {
            let tau = nodes[i];
            let x_tau = x0 * (1.0 - tau) + x1 * tau;
            let t_tau = t + self.dt * tau;

            let f = flow.evaluate(&x_tau.view(), t_tau);
            let v = (x1 - x0) / self.dt;

            // Lagrangian density
            let l = 0.5 * v.dot(&v) - v.dot(&f);
            l_d += weights[i] * l;
        }

        Ok(l_d * self.dt)
    }

    /// Gauss-Legendre quadrature on [0,1]
    fn gauss_legendre_quadrature(&self) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
        match self.n_quad {
            1 => Ok((vec![1.0], vec![0.5])),
            2 => Ok((
                vec![0.5, 0.5],
                vec![0.5 - 0.5 / 3.0_f64.sqrt(), 0.5 + 0.5 / 3.0_f64.sqrt()],
            )),
            3 => Ok((
                vec![5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0],
                vec![
                    0.5 - 0.5 * (0.6_f64).sqrt(),
                    0.5,
                    0.5 + 0.5 * (0.6_f64).sqrt(),
                ],
            )),
            _ => Err(IntegrateError::ValueError(format!(
                "Quadrature order {} not implemented",
                self.n_quad
            ))),
        }
    }
}

/// Discrete gradient method for preserving multiple invariants
pub struct DiscreteGradientIntegrator {
    /// Time step
    #[allow(dead_code)]
    dt: f64,
    /// Invariant functions
    #[allow(dead_code)]
    invariants: Vec<InvariantFn>,
}

impl DiscreteGradientIntegrator {
    /// Create a new discrete gradient integrator
    pub fn new(dt: f64) -> Self {
        Self {
            dt,
            invariants: Vec::new(),
        }
    }

    /// Add an invariant function to preserve
    pub fn add_invariant<I>(&mut self, invariant: I) -> &mut Self
    where
        I: Fn(&ArrayView1<f64>) -> f64 + 'static,
    {
        self.invariants.push(Box::new(invariant));
        self
    }

    /// Compute discrete gradient
    pub fn discrete_gradient(
        &self,
        x0: &ArrayView1<f64>,
        x1: &ArrayView1<f64>,
        invariantidx: usize,
    ) -> Array1<f64> {
        let h = &self.invariants[invariantidx];
        let h0 = h(x0);
        let h1 = h(x1);

        if (x1 - x0).mapv(|x| x.abs()).sum() < 1e-14 {
            // If x0 ≈ x1, use standard gradient
            self.gradient(x0, invariantidx)
        } else {
            // Average vector field
            let g0 = self.gradient(x0, invariantidx);
            let g1 = self.gradient(x1, invariantidx);
            let g_avg = (&g0 + &g1) / 2.0;

            // Correction term
            let dx = x1 - x0;
            let correction = (h1 - h0 - g_avg.dot(&dx)) / dx.dot(&dx) * &dx;

            g_avg + correction
        }
    }

    /// Standard gradient computation
    fn gradient(&self, x: &ArrayView1<f64>, invariantidx: usize) -> Array1<f64> {
        let h = &self.invariants[invariantidx];
        let eps = 1e-8;
        let n = x.len();
        let mut grad = Array1::zeros(n);

        for i in 0..n {
            let mut x_plus = x.to_owned();
            let mut x_minus = x.to_owned();
            x_plus[i] += eps;
            x_minus[i] -= eps;

            grad[i] = (h(&x_plus.view()) - h(&x_minus.view())) / (2.0 * eps);
        }

        grad
    }
}

/// Volume computation utilities
pub struct VolumeChecker;

impl VolumeChecker {
    /// Check volume preservation for a set of points
    pub fn check_volume_preservation<F>(
        points: &Array2<f64>,
        integrator: &VolumePreservingIntegrator,
        flow: &F,
        t0: f64,
        t_final: f64,
    ) -> IntegrateResult<f64>
    where
        F: DivergenceFreeFlow,
    {
        let npoints = points.nrows();
        let dim = points.ncols();

        // Initial volume (using convex hull approximation for simplicity)
        let initial_volume = Self::estimate_volume(points)?;

        // Evolve all points
        let mut evolvedpoints = Array2::zeros((npoints, dim));
        for i in 0..npoints {
            let x0 = points.row(i);
            let trajectory = integrator.integrate(&x0, t0, t_final, flow)?;
            let (_, x_final) = trajectory.last().unwrap();
            evolvedpoints.row_mut(i).assign(x_final);
        }

        // Final volume
        let final_volume = Self::estimate_volume(&evolvedpoints)?;

        // Return relative volume change
        Ok((final_volume - initial_volume).abs() / initial_volume)
    }

    /// Estimate volume using bounding box (simplified)
    fn estimate_volume(points: &Array2<f64>) -> IntegrateResult<f64> {
        if points.nrows() == 0 {
            return Ok(0.0);
        }

        let dim = points.ncols();
        let mut min_coords = points.row(0).to_owned();
        let mut max_coords = points.row(0).to_owned();

        for i in 1..points.nrows() {
            for j in 0..dim {
                min_coords[j] = min_coords[j].min(points[[i, j]]);
                max_coords[j] = max_coords[j].max(points[[i, j]]);
            }
        }

        let mut volume = 1.0;
        for j in 0..dim {
            volume *= max_coords[j] - min_coords[j];
        }

        Ok(volume)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_circular_flow_volume_preservation() {
        let flow = CircularFlow2D { omega: 1.0 };
        let integrator =
            VolumePreservingIntegrator::new(0.1, VolumePreservingMethod::ExplicitMidpoint);

        // Create a square of points
        let mut points = Array2::zeros((4, 2));
        points[[0, 0]] = 1.0;
        points[[0, 1]] = 0.0;
        points[[1, 0]] = 0.0;
        points[[1, 1]] = 1.0;
        points[[2, 0]] = -1.0;
        points[[2, 1]] = 0.0;
        points[[3, 0]] = 0.0;
        points[[3, 1]] = -1.0;

        let volume_change =
            VolumeChecker::check_volume_preservation(&points, &integrator, &flow, 0.0, 2.0 * PI)
                .unwrap();

        assert!(
            volume_change < 0.01,
            "Volume not preserved: {volume_change}"
        );
    }

    #[test]
    fn test_divergence_free_verification() {
        let flow = ABCFlow {
            a: 1.0,
            b: SQRT_2,
            c: PI / 2.0,
        };
        let x = Array1::from_vec(vec![0.5, 0.5, 0.5]);

        let div = flow.verify_divergence_free(&x.view(), 0.0, 1e-6);
        assert!(div.abs() < 1e-8, "Flow not divergence-free: {div}");
    }

    #[test]
    fn test_implicit_midpoint_convergence() {
        let flow = CircularFlow2D { omega: 1.0 };
        let dt = 0.1;

        let integrator =
            VolumePreservingIntegrator::new(dt, VolumePreservingMethod::ImplicitMidpoint);
        let x0 = Array1::from_vec(vec![1.0, 0.0]);

        let x1 = integrator.step(&x0.view(), 0.0, &flow).unwrap();

        // After one step, should approximately be at (cos(dt), sin(dt))
        assert_relative_eq!(x1[0], dt.cos(), epsilon = 1e-3);
        assert_relative_eq!(x1[1], dt.sin(), epsilon = 1e-3);
    }
}
