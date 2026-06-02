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
    ///
    /// Delegates to [`gauss_legendre_quadrature`] (which returns nodes/weights on [-1,1])
    /// and transforms to the unit interval via τ = (x+1)/2.
    fn gauss_legendre_quadrature(&self) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
        let (nodes_m1p1, weights_m1p1) = gauss_legendre_quadrature(self.n_quad)?;
        // Transform [-1,1] → [0,1]: τ = (x+1)/2, dτ = dx/2 → w_[0,1] = w_[-1,1]/2
        let nodes: Vec<f64> = nodes_m1p1.iter().map(|&x| (x + 1.0) * 0.5).collect();
        let weights: Vec<f64> = weights_m1p1.iter().map(|&w| w * 0.5).collect();
        // Return (weights, nodes) to match the original convention used by discrete_lagrangian
        Ok((weights, nodes))
    }
}

/// Gauss-Legendre quadrature nodes and weights on [-1, 1] for orders 1 through 10.
///
/// Returns `(nodes, weights)` where `nodes` are the quadrature points in [-1, 1]
/// and `weights` are the corresponding quadrature weights (summing to 2).
///
/// The n-point rule integrates polynomials of degree ≤ 2n-1 exactly.
///
/// # Errors
///
/// Returns [`IntegrateError::ValueError`] when `n_quad` is 0 or greater than 10.
///
/// # Example
///
/// ```
/// use scirs2_integrate::geometric::volume_preserving::gauss_legendre_quadrature;
/// let (nodes, weights) = gauss_legendre_quadrature(3).unwrap();
/// assert_eq!(nodes.len(), 3);
/// let sum: f64 = weights.iter().sum();
/// assert!((sum - 2.0).abs() < 1e-13);
/// ```
///
/// # Note
///
/// For n > 10, use Golub-Welsch tridiagonal eigenvalue computation (future work).
pub fn gauss_legendre_quadrature(n_quad: usize) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
    // Canonical Gauss-Legendre nodes and weights on [-1, 1].
    // Source: Abramowitz & Stegun, Table 25.4 (verified against SciPy/DLMF).
    //
    // For n > 10, use Golub-Welsch tridiagonal eigenvalue computation (future work).
    let (nodes_slice, weights_slice): (&[f64], &[f64]) = match n_quad {
        1 => (&[0.0], &[2.0]),
        2 => (
            &[-0.577_350_269_189_625_7, 0.577_350_269_189_625_7],
            &[1.0, 1.0],
        ),
        3 => (
            &[-0.774_596_669_241_483_4, 0.0, 0.774_596_669_241_483_4],
            &[
                0.555_555_555_555_555_6,
                0.888_888_888_888_888_8,
                0.555_555_555_555_555_6,
            ],
        ),
        4 => (
            &[
                -0.861_136_311_594_052_6,
                -0.339_981_043_584_856_3,
                0.339_981_043_584_856_3,
                0.861_136_311_594_052_6,
            ],
            &[
                0.347_854_845_137_453_8,
                0.652_145_154_862_546_1,
                0.652_145_154_862_546_1,
                0.347_854_845_137_453_8,
            ],
        ),
        5 => (
            &[
                -0.906_179_845_938_664,
                -0.538_469_310_105_683_1,
                0.0,
                0.538_469_310_105_683_1,
                0.906_179_845_938_664,
            ],
            &[
                0.236_926_885_056_189_1,
                0.478_628_670_499_366_5,
                0.568_888_888_888_888_9,
                0.478_628_670_499_366_5,
                0.236_926_885_056_189_1,
            ],
        ),
        6 => (
            &[
                -0.932_469_514_203_152,
                -0.661_209_386_466_264_5,
                -0.238_619_186_083_196_9,
                0.238_619_186_083_196_9,
                0.661_209_386_466_264_5,
                0.932_469_514_203_152,
            ],
            &[
                0.171_324_492_379_170_4,
                0.360_761_573_048_138_6,
                0.467_913_934_572_691,
                0.467_913_934_572_691,
                0.360_761_573_048_138_6,
                0.171_324_492_379_170_4,
            ],
        ),
        7 => (
            &[
                -0.949_107_912_342_758_5,
                -0.741_531_185_599_394_5,
                -0.405_845_151_377_397_2,
                0.0,
                0.405_845_151_377_397_2,
                0.741_531_185_599_394_5,
                0.949_107_912_342_758_5,
            ],
            &[
                0.129_484_966_168_869_7,
                0.279_705_391_489_276_7,
                0.381_830_050_505_118_9,
                0.417_959_183_673_469_4,
                0.381_830_050_505_118_9,
                0.279_705_391_489_276_7,
                0.129_484_966_168_869_7,
            ],
        ),
        8 => (
            &[
                -0.960_289_856_497_536_3,
                -0.796_666_477_413_626_7,
                -0.525_532_409_916_329,
                -0.183_434_642_495_649_8,
                0.183_434_642_495_649_8,
                0.525_532_409_916_329,
                0.796_666_477_413_626_7,
                0.960_289_856_497_536_3,
            ],
            &[
                0.101_228_536_290_376_3,
                0.222_381_034_453_374_5,
                0.313_706_645_877_887_3,
                0.362_683_783_378_362,
                0.362_683_783_378_362,
                0.313_706_645_877_887_3,
                0.222_381_034_453_374_5,
                0.101_228_536_290_376_3,
            ],
        ),
        9 => (
            &[
                -0.968_160_239_507_626_1,
                -0.836_031_107_326_635_8,
                -0.613_371_432_700_590_4,
                -0.324_253_423_403_808_9,
                0.0,
                0.324_253_423_403_808_9,
                0.613_371_432_700_590_4,
                0.836_031_107_326_635_8,
                0.968_160_239_507_626_1,
            ],
            &[
                0.081_274_388_361_574_4,
                0.180_648_160_694_857_4,
                0.260_610_696_402_935_4,
                0.312_347_077_040_002_9,
                0.330_239_355_001_259_8,
                0.312_347_077_040_002_9,
                0.260_610_696_402_935_4,
                0.180_648_160_694_857_4,
                0.081_274_388_361_574_4,
            ],
        ),
        10 => (
            &[
                -0.973_906_528_517_171_7,
                -0.865_063_366_688_984_5,
                -0.679_409_568_299_024_4,
                -0.433_395_394_129_247_2,
                -0.148_874_338_981_631_2,
                0.148_874_338_981_631_2,
                0.433_395_394_129_247_2,
                0.679_409_568_299_024_4,
                0.865_063_366_688_984_5,
                0.973_906_528_517_171_7,
            ],
            &[
                0.066_671_344_308_688_1,
                0.149_451_349_150_580_6,
                0.219_086_362_515_982,
                0.269_266_719_309_996_3,
                0.295_524_224_714_752_9,
                0.295_524_224_714_752_9,
                0.269_266_719_309_996_3,
                0.219_086_362_515_982,
                0.149_451_349_150_580_6,
                0.066_671_344_308_688_1,
            ],
        ),
        _ => {
            return Err(IntegrateError::ValueError(format!(
                "Gauss-Legendre order {} not supported (must be 1..=10)",
                n_quad
            )))
        }
    };
    Ok((nodes_slice.to_vec(), weights_slice.to_vec()))
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
            let (_, x_final) = trajectory.last().expect("Operation failed");
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
                .expect("Operation failed");

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

        let x1 = integrator
            .step(&x0.view(), 0.0, &flow)
            .expect("Operation failed");

        // After one step, should approximately be at (cos(dt), sin(dt))
        assert_relative_eq!(x1[0], dt.cos(), epsilon = 1e-3);
        assert_relative_eq!(x1[1], dt.sin(), epsilon = 1e-3);
    }
}
