//! Lie group integrators
//!
//! This module provides numerical integrators for differential equations on Lie groups,
//! preserving the group structure throughout the integration.

use crate::error::{IntegrateResult, IntegrateResult as Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// Trait for Lie algebra operations
pub trait LieAlgebra: Clone {
    /// Dimension of the Lie algebra
    fn dim(&self) -> usize;

    /// Lie bracket [X, Y] = XY - YX
    fn bracket(&self, other: &Self) -> Self;

    /// Convert from vector representation to algebra element
    fn from_vector(v: &ArrayView1<f64>) -> Self;

    /// Convert to vector representation
    fn to_vector(&self) -> Array1<f64>;

    /// Norm of the algebra element
    fn norm(&self) -> f64;
}

/// Trait for exponential map on Lie groups
pub trait ExponentialMap: Sized {
    /// Associated Lie algebra type
    type Algebra: LieAlgebra;

    /// Exponential map from Lie algebra to Lie group
    fn exp(algebra: &Self::Algebra) -> Self;

    /// Logarithm map from Lie group to Lie algebra
    fn log(&self) -> Self::Algebra;

    /// Group multiplication
    fn multiply(&self, other: &Self) -> Self;

    /// Group inverse
    fn inverse(&self) -> Self;

    /// Identity element
    fn identity(&self) -> Self;
}

/// General Lie group integrator
pub struct LieGroupIntegrator<G: ExponentialMap> {
    /// Time step
    pub dt: f64,
    /// Integration method
    pub method: LieGroupMethod,
    /// Phantom data for group type
    _phantom: std::marker::PhantomData<G>,
}

/// Available Lie group integration methods
#[derive(Debug, Clone, Copy)]
pub enum LieGroupMethod {
    /// Lie-Euler method (first order)
    LieEuler,
    /// Lie-Midpoint method (second order)
    LieMidpoint,
    /// Lie-Trapezoidal method (second order)
    LieTrapezoidal,
    /// Runge-Kutta-Munthe-Kaas method (fourth order)
    RKMK4,
    /// Crouch-Grossman method
    CrouchGrossman,
}

impl<G: ExponentialMap> LieGroupIntegrator<G> {
    /// Create a new Lie group integrator
    pub fn new(dt: f64, method: LieGroupMethod) -> Self {
        Self {
            dt,
            method,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Integrate one step
    pub fn step<F>(&self, g: &G, f: F) -> IntegrateResult<G>
    where
        F: Fn(&G) -> G::Algebra,
    {
        match self.method {
            LieGroupMethod::LieEuler => self.lie_euler_step(g, f),
            LieGroupMethod::LieMidpoint => self.lie_midpoint_step(g, f),
            LieGroupMethod::LieTrapezoidal => self.lie_trapezoidal_step(g, f),
            LieGroupMethod::RKMK4 => self.rkmk4_step(g, f),
            LieGroupMethod::CrouchGrossman => self.crouch_grossman_step(g, f),
        }
    }

    /// Lie-Euler method: g_{n+1} = g_n * exp(dt * f(g_n))
    fn lie_euler_step<F>(&self, g: &G, f: F) -> IntegrateResult<G>
    where
        F: Fn(&G) -> G::Algebra,
    {
        let xi = f(g);
        let mut scaled_xi = xi.to_vector();
        scaled_xi *= self.dt;
        let scaledalgebra = G::Algebra::from_vector(&scaled_xi.view());
        Ok(g.multiply(&G::exp(&scaledalgebra)))
    }

    /// Lie-Midpoint method
    fn lie_midpoint_step<F>(&self, g: &G, f: F) -> IntegrateResult<G>
    where
        F: Fn(&G) -> G::Algebra,
    {
        let xi1 = f(g);
        let mut half_xi = xi1.to_vector();
        half_xi *= self.dt / 2.0;
        let halfalgebra = G::Algebra::from_vector(&half_xi.view());
        let g_mid = g.multiply(&G::exp(&halfalgebra));

        let xi_mid = f(&g_mid);
        let mut full_xi = xi_mid.to_vector();
        full_xi *= self.dt;
        let fullalgebra = G::Algebra::from_vector(&full_xi.view());
        Ok(g.multiply(&G::exp(&fullalgebra)))
    }

    /// Lie-Trapezoidal method
    fn lie_trapezoidal_step<F>(&self, g: &G, f: F) -> IntegrateResult<G>
    where
        F: Fn(&G) -> G::Algebra,
    {
        let xi1 = f(g);
        let mut full_xi = xi1.to_vector();
        full_xi *= self.dt;
        let fullalgebra = G::Algebra::from_vector(&full_xi.view());
        let g_euler = g.multiply(&G::exp(&fullalgebra));

        let xi2 = f(&g_euler);
        let mut avg_xi = (xi1.to_vector() + xi2.to_vector()) / 2.0;
        avg_xi *= self.dt;
        let avgalgebra = G::Algebra::from_vector(&avg_xi.view());
        Ok(g.multiply(&G::exp(&avgalgebra)))
    }

    /// Runge-Kutta-Munthe-Kaas 4th order method
    fn rkmk4_step<F>(&self, g: &G, f: F) -> IntegrateResult<G>
    where
        F: Fn(&G) -> G::Algebra,
    {
        // k1
        let k1 = f(g);

        // k2
        let mut exp_arg = k1.to_vector() * (self.dt / 2.0);
        let exp_k1_2 = G::exp(&G::Algebra::from_vector(&exp_arg.view()));
        let g2 = g.multiply(&exp_k1_2);
        let k2 = f(&g2);

        // k3
        exp_arg = k2.to_vector() * (self.dt / 2.0);
        let exp_k2_2 = G::exp(&G::Algebra::from_vector(&exp_arg.view()));
        let g3 = g.multiply(&exp_k2_2);
        let k3 = f(&g3);

        // k4
        exp_arg = k3.to_vector() * self.dt;
        let exp_k3 = G::exp(&G::Algebra::from_vector(&exp_arg.view()));
        let g4 = g.multiply(&exp_k3);
        let k4 = f(&g4);

        // Combine using BCH formula approximation
        let combined =
            (k1.to_vector() + k2.to_vector() * 2.0 + k3.to_vector() * 2.0 + k4.to_vector()) / 6.0;
        let mut final_xi = combined * self.dt;

        // Second-order BCH correction
        let comm = k1.bracket(&k2);
        final_xi = final_xi + comm.to_vector() * (self.dt * self.dt / 12.0);

        let finalalgebra = G::Algebra::from_vector(&final_xi.view());
        Ok(g.multiply(&G::exp(&finalalgebra)))
    }

    /// Crouch-Grossman method
    fn crouch_grossman_step<F>(&self, g: &G, f: F) -> IntegrateResult<G>
    where
        F: Fn(&G) -> G::Algebra,
    {
        let c2 = 0.5;
        let c3 = 1.0;
        let b1 = 1.0 / 6.0;
        let b2 = 2.0 / 3.0;
        let b3 = 1.0 / 6.0;

        // Stage 1
        let k1 = f(g);

        // Stage 2
        let mut exp_arg = k1.to_vector() * (c2 * self.dt);
        let y2 = g.multiply(&G::exp(&G::Algebra::from_vector(&exp_arg.view())));
        let k2 = f(&y2);

        // Stage 3
        exp_arg = k2.to_vector() * (c3 * self.dt);
        let y3 = g.multiply(&G::exp(&G::Algebra::from_vector(&exp_arg.view())));
        let k3 = f(&y3);

        // Final update
        let combined = k1.to_vector() * b1 + k2.to_vector() * b2 + k3.to_vector() * b3;
        exp_arg = combined * self.dt;
        Ok(g.multiply(&G::exp(&G::Algebra::from_vector(&exp_arg.view()))))
    }
}

/// SO(3) Lie algebra (skew-symmetric 3x3 matrices)
#[derive(Debug, Clone)]
pub struct So3 {
    /// Components [ω_x, ω_y, ω_z] representing angular velocity
    pub omega: Array1<f64>,
}

impl LieAlgebra for So3 {
    fn dim(&self) -> usize {
        3
    }

    fn bracket(&self, other: &Self) -> Self {
        // [ω1, ω2] = ω1 × ω2 (cross product)
        let omega = Array1::from_vec(vec![
            self.omega[1] * other.omega[2] - self.omega[2] * other.omega[1],
            self.omega[2] * other.omega[0] - self.omega[0] * other.omega[2],
            self.omega[0] * other.omega[1] - self.omega[1] * other.omega[0],
        ]);
        So3 { omega }
    }

    fn from_vector(v: &ArrayView1<f64>) -> Self {
        So3 {
            omega: v.to_owned(),
        }
    }

    fn to_vector(&self) -> Array1<f64> {
        self.omega.clone()
    }

    fn norm(&self) -> f64 {
        self.omega.dot(&self.omega).sqrt()
    }
}

impl So3 {
    /// Convert to 3x3 skew-symmetric matrix
    pub fn to_matrix(&self) -> Array2<f64> {
        let mut mat = Array2::zeros((3, 3));
        mat[[0, 1]] = -self.omega[2];
        mat[[0, 2]] = self.omega[1];
        mat[[1, 0]] = self.omega[2];
        mat[[1, 2]] = -self.omega[0];
        mat[[2, 0]] = -self.omega[1];
        mat[[2, 1]] = self.omega[0];
        mat
    }
}

/// SO(3) Lie group (3D rotation matrices)
#[derive(Debug, Clone)]
pub struct SO3 {
    /// Rotation matrix
    pub matrix: Array2<f64>,
}

impl ExponentialMap for SO3 {
    type Algebra = So3;

    fn exp(algebra: &Self::Algebra) -> Self {
        let theta = algebra.norm();

        if theta < 1e-10 {
            // Small angle approximation
            SO3 {
                matrix: Array2::eye(3) + algebra.to_matrix(),
            }
        } else {
            // Rodrigues' formula
            let k = algebra.to_matrix() / theta;
            let k2 = k.dot(&k);

            SO3 {
                matrix: Array2::eye(3) + k * theta.sin() + k2 * (1.0 - theta.cos()),
            }
        }
    }

    fn log(&self) -> Self::Algebra {
        // Extract axis-angle from rotation matrix
        let trace = self.matrix[[0, 0]] + self.matrix[[1, 1]] + self.matrix[[2, 2]];
        let theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();

        if theta.abs() < 1e-10 {
            // Small rotation
            So3 {
                omega: Array1::from_vec(vec![
                    self.matrix[[2, 1]] - self.matrix[[1, 2]],
                    self.matrix[[0, 2]] - self.matrix[[2, 0]],
                    self.matrix[[1, 0]] - self.matrix[[0, 1]],
                ]) / 2.0,
            }
        } else {
            // General case
            let factor = theta / (2.0 * theta.sin());
            So3 {
                omega: Array1::from_vec(vec![
                    self.matrix[[2, 1]] - self.matrix[[1, 2]],
                    self.matrix[[0, 2]] - self.matrix[[2, 0]],
                    self.matrix[[1, 0]] - self.matrix[[0, 1]],
                ]) * factor,
            }
        }
    }

    fn multiply(&self, other: &Self) -> Self {
        SO3 {
            matrix: self.matrix.dot(&other.matrix),
        }
    }

    fn inverse(&self) -> Self {
        SO3 {
            matrix: self.matrix.t().to_owned(),
        }
    }

    fn identity(&self) -> Self {
        SO3 {
            matrix: Array2::eye(3),
        }
    }
}

/// Integrator specifically for SO(3)
pub struct SO3Integrator {
    /// Base Lie group integrator
    base: LieGroupIntegrator<SO3>,
}

impl SO3Integrator {
    /// Create a new SO(3) integrator
    pub fn new(dt: f64, method: LieGroupMethod) -> Self {
        Self {
            base: LieGroupIntegrator::new(dt, method),
        }
    }

    /// Integrate rigid body dynamics
    pub fn integrate_rigid_body(
        &self,
        orientation: &SO3,
        angular_velocity: &Array1<f64>,
        inertia_tensor: &Array2<f64>,
        external_torque: &Array1<f64>,
        n_steps: usize,
    ) -> IntegrateResult<Vec<(SO3, Array1<f64>)>> {
        let mut states = vec![(orientation.clone(), angular_velocity.clone())];
        let mut current_orientation = orientation.clone();
        let mut current_omega = angular_velocity.clone();

        let inertia_inv = SO3Integrator::invertinertia(inertia_tensor)?;

        for _ in 0..n_steps {
            // Compute angular acceleration
            let omega_cross_i_omega =
                SO3Integrator::cross_product(&current_omega, &inertia_tensor.dot(&current_omega));
            let angular_accel = inertia_inv.dot(&(external_torque - &omega_cross_i_omega));

            // Update angular _velocity
            current_omega = &current_omega + &angular_accel * self.base.dt;

            // Update orientation
            let omegaalgebra = So3 {
                omega: current_omega.clone(),
            };
            current_orientation = self
                .base
                .step(&current_orientation, |_| omegaalgebra.clone())?;

            states.push((current_orientation.clone(), current_omega.clone()));
        }

        Ok(states)
    }

    /// Cross product for 3D vectors
    fn cross_product(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
        Array1::from_vec(vec![
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ])
    }

    /// Invert 3x3 matrix (for inertia tensor)
    fn invertinertia(inertia: &Array2<f64>) -> IntegrateResult<Array2<f64>> {
        // Simple 3x3 matrix inversion
        let det = inertia[[0, 0]]
            * (inertia[[1, 1]] * inertia[[2, 2]] - inertia[[1, 2]] * inertia[[2, 1]])
            - inertia[[0, 1]]
                * (inertia[[1, 0]] * inertia[[2, 2]] - inertia[[1, 2]] * inertia[[2, 0]])
            + inertia[[0, 2]]
                * (inertia[[1, 0]] * inertia[[2, 1]] - inertia[[1, 1]] * inertia[[2, 0]]);

        if det.abs() < 1e-10 {
            return Err(crate::error::IntegrateError::ValueError(
                "Singular inertia tensor".to_string(),
            ));
        }

        let mut inv = Array2::zeros((3, 3));
        inv[[0, 0]] = (inertia[[1, 1]] * inertia[[2, 2]] - inertia[[1, 2]] * inertia[[2, 1]]) / det;
        inv[[0, 1]] = (inertia[[0, 2]] * inertia[[2, 1]] - inertia[[0, 1]] * inertia[[2, 2]]) / det;
        inv[[0, 2]] = (inertia[[0, 1]] * inertia[[1, 2]] - inertia[[0, 2]] * inertia[[1, 1]]) / det;
        inv[[1, 0]] = (inertia[[1, 2]] * inertia[[2, 0]] - inertia[[1, 0]] * inertia[[2, 2]]) / det;
        inv[[1, 1]] = (inertia[[0, 0]] * inertia[[2, 2]] - inertia[[0, 2]] * inertia[[2, 0]]) / det;
        inv[[1, 2]] = (inertia[[0, 2]] * inertia[[1, 0]] - inertia[[0, 0]] * inertia[[1, 2]]) / det;
        inv[[2, 0]] = (inertia[[1, 0]] * inertia[[2, 1]] - inertia[[1, 1]] * inertia[[2, 0]]) / det;
        inv[[2, 1]] = (inertia[[0, 1]] * inertia[[2, 0]] - inertia[[0, 0]] * inertia[[2, 1]]) / det;
        inv[[2, 2]] = (inertia[[0, 0]] * inertia[[1, 1]] - inertia[[0, 1]] * inertia[[1, 0]]) / det;

        Ok(inv)
    }
}

/// SE(3) Lie algebra (rigid body motions)
#[derive(Debug, Clone)]
pub struct Se3 {
    /// Linear velocity
    pub v: Array1<f64>,
    /// Angular velocity
    pub omega: Array1<f64>,
}

impl LieAlgebra for Se3 {
    fn dim(&self) -> usize {
        6
    }

    fn bracket(&self, other: &Self) -> Self {
        // [ξ1, ξ2] = ad_ξ1(ξ2)
        let omega_bracket = So3 {
            omega: self.omega.clone(),
        }
        .bracket(&So3 {
            omega: other.omega.clone(),
        });

        let v_bracket = self.cross_3d(&self.omega, &other.v) - self.cross_3d(&other.omega, &self.v);

        Se3 {
            v: v_bracket,
            omega: omega_bracket.omega,
        }
    }

    fn from_vector(v: &ArrayView1<f64>) -> Self {
        Se3 {
            v: v.slice(scirs2_core::ndarray::s![0..3]).to_owned(),
            omega: v.slice(scirs2_core::ndarray::s![3..6]).to_owned(),
        }
    }

    fn to_vector(&self) -> Array1<f64> {
        let mut vec = Array1::zeros(6);
        vec.slice_mut(scirs2_core::ndarray::s![0..3])
            .assign(&self.v);
        vec.slice_mut(scirs2_core::ndarray::s![3..6])
            .assign(&self.omega);
        vec
    }

    fn norm(&self) -> f64 {
        (self.v.dot(&self.v) + self.omega.dot(&self.omega)).sqrt()
    }
}

impl Se3 {
    fn cross_3d(&self, a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
        Array1::from_vec(vec![
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ])
    }
}

/// SE(3) Lie group (rigid body transformations)
#[derive(Debug, Clone)]
pub struct SE3 {
    /// Rotation part (SO(3))
    pub rotation: SO3,
    /// Translation part
    pub translation: Array1<f64>,
}

impl ExponentialMap for SE3 {
    type Algebra = Se3;

    fn exp(algebra: &Self::Algebra) -> Self {
        let omega_norm = algebra.omega.dot(&algebra.omega).sqrt();
        let rotation = SO3::exp(&So3 {
            omega: algebra.omega.clone(),
        });

        let translation = if omega_norm < 1e-10 {
            // Small angle: V ≈ I
            algebra.v.clone()
        } else {
            // General case: compute V matrix
            let axis = &algebra.omega / omega_norm;
            let axis_cross = So3 {
                omega: axis.clone(),
            }
            .to_matrix();
            let axis_cross2 = axis_cross.dot(&axis_cross);

            let v_matrix = Array2::eye(3)
                + axis_cross * ((1.0 - omega_norm.cos()) / omega_norm)
                + axis_cross2 * ((omega_norm - omega_norm.sin()) / omega_norm);

            v_matrix.dot(&algebra.v)
        };

        SE3 {
            rotation,
            translation,
        }
    }

    fn log(&self) -> Self::Algebra {
        let omegaalgebra = self.rotation.log();
        let omega = &omegaalgebra.omega;
        let omega_norm = omega.dot(omega).sqrt();

        let v = if omega_norm < 1e-10 {
            // Small rotation: V^(-1) ≈ I
            self.translation.clone()
        } else {
            // General case: compute V^(-1)
            let axis = omega / omega_norm;
            let axis_cross = So3 {
                omega: axis.clone(),
            }
            .to_matrix();
            let axis_cross2 = axis_cross.dot(&axis_cross);

            let cot_half = 1.0 / (omega_norm / 2.0).tan();
            let v_inv = Array2::eye(3) - axis_cross / 2.0
                + axis_cross2 * (1.0 / omega_norm.powi(2)) * (1.0 - omega_norm / 2.0 * cot_half);

            v_inv.dot(&self.translation)
        };

        Se3 {
            v,
            omega: omegaalgebra.omega,
        }
    }

    fn multiply(&self, other: &Self) -> Self {
        SE3 {
            rotation: self.rotation.multiply(&other.rotation),
            translation: &self.translation + &self.rotation.matrix.dot(&other.translation),
        }
    }

    fn inverse(&self) -> Self {
        let rotation_inv = self.rotation.inverse();
        SE3 {
            rotation: rotation_inv.clone(),
            translation: -rotation_inv.matrix.dot(&self.translation),
        }
    }

    fn identity(&self) -> Self {
        SE3 {
            rotation: SO3 {
                matrix: Array2::eye(3),
            },
            translation: Array1::zeros(3),
        }
    }
}

/// Integrator specifically for SE(3)
pub struct SE3Integrator {
    /// Base Lie group integrator
    base: LieGroupIntegrator<SE3>,
}

impl SE3Integrator {
    /// Create a new SE(3) integrator
    pub fn new(dt: f64, method: LieGroupMethod) -> Self {
        Self {
            base: LieGroupIntegrator::new(dt, method),
        }
    }

    /// Integrate rigid body motion with forces and torques
    pub fn integrate_rigid_body_6dof(
        &self,
        pose: &SE3,
        velocity: &Se3,
        mass: f64,
        inertia: &Array2<f64>,
        forces: &Array1<f64>,
        torques: &Array1<f64>,
        n_steps: usize,
    ) -> IntegrateResult<Vec<(SE3, Se3)>> {
        let mut states = vec![(pose.clone(), velocity.clone())];
        let mut current_pose = pose.clone();
        let mut current_velocity = velocity.clone();

        for _ in 0..n_steps {
            // Update velocities with Newton-Euler equations
            let linear_accel = forces / mass;
            let angular_accel =
                self.compute_angular_acceleration(&current_velocity.omega, inertia, torques)?;

            // Update velocity
            current_velocity.v = &current_velocity.v + &linear_accel * self.base.dt;
            current_velocity.omega = &current_velocity.omega + &angular_accel * self.base.dt;

            // Update pose
            current_pose = self
                .base
                .step(&current_pose, |_| current_velocity.clone())?;

            states.push((current_pose.clone(), current_velocity.clone()));
        }

        Ok(states)
    }

    fn compute_angular_acceleration(
        &self,
        omega: &Array1<f64>,
        inertia: &Array2<f64>,
        torque: &Array1<f64>,
    ) -> IntegrateResult<Array1<f64>> {
        // τ = Iα + ω × (Iω)
        // α = I^(-1)(τ - ω × (Iω))
        let i_omega = inertia.dot(omega);
        let omega_cross_i_omega = Array1::from_vec(vec![
            omega[1] * i_omega[2] - omega[2] * i_omega[1],
            omega[2] * i_omega[0] - omega[0] * i_omega[2],
            omega[0] * i_omega[1] - omega[1] * i_omega[0],
        ]);

        let inertia_inv = SO3Integrator::invertinertia(inertia)?;

        Ok(inertia_inv.dot(&(torque - &omega_cross_i_omega)))
    }
}

/// SL(n) Lie algebra (traceless matrices)
#[derive(Debug, Clone)]
pub struct Sln {
    /// Dimension n
    pub n: usize,
    /// Matrix representation (traceless)
    pub matrix: Array2<f64>,
}

impl LieAlgebra for Sln {
    fn dim(&self) -> usize {
        // For SL(n), dimension is n^2 - 1
        0 // This should be set dynamically based on n
    }

    fn bracket(&self, other: &Self) -> Self {
        // Matrix commutator [A, B] = AB - BA
        let ab = self.matrix.dot(&other.matrix);
        let ba = other.matrix.dot(&self.matrix);
        Sln {
            n: self.n,
            matrix: ab - ba,
        }
    }

    fn from_vector(v: &ArrayView1<f64>) -> Self {
        // This needs the dimension n to be known
        // For now, assume n=2 (SL(2))
        let n = 2;
        let mut matrix = Array2::zeros((n, n));

        // SL(2) basis: σ1, σ2, σ3 (Pauli matrices without i)
        matrix[[0, 0]] = v[2];
        matrix[[0, 1]] = v[0] - v[1];
        matrix[[1, 0]] = v[0] + v[1];
        matrix[[1, 1]] = -v[2];

        Sln { n, matrix }
    }

    fn to_vector(&self) -> Array1<f64> {
        if self.n == 2 {
            // Extract coordinates for SL(2)
            Array1::from_vec(vec![
                (self.matrix[[1, 0]] + self.matrix[[0, 1]]) / 2.0,
                (self.matrix[[1, 0]] - self.matrix[[0, 1]]) / 2.0,
                self.matrix[[0, 0]],
            ])
        } else {
            // General case: vectorize the independent components
            let mut v = Vec::new();
            for i in 0..self.n {
                for j in 0..self.n {
                    if i != self.n - 1 || j != self.n - 1 {
                        v.push(self.matrix[[i, j]]);
                    }
                }
            }
            Array1::from_vec(v)
        }
    }

    fn norm(&self) -> f64 {
        // Frobenius norm
        self.matrix.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

/// SL(n) Lie group (special linear group)
#[derive(Debug, Clone)]
pub struct SLn {
    /// Dimension n
    pub n: usize,
    /// Matrix with determinant 1
    pub matrix: Array2<f64>,
}

impl ExponentialMap for SLn {
    type Algebra = Sln;

    fn exp(algebra: &Self::Algebra) -> Self {
        // Matrix exponential for SL(n)
        // Use scaling and squaring method
        let n = algebra.n;
        let a = &algebra.matrix;
        let norm = algebra.norm();

        // Scaling
        let s = (norm.log2().ceil() as i32).max(0);
        let a_scaled = a / 2.0f64.powi(s);

        // Padé approximation (order 6)
        let a2 = a_scaled.dot(&a_scaled);
        let a4 = a2.dot(&a2);
        let a6 = a4.dot(&a2);

        let mut u = Array2::eye(n) * 1.0;
        u = u + &a_scaled * 1.0;
        u = u + &a2 * (1.0 / 2.0);
        u = u + &a4 * (1.0 / 24.0);
        u = u + &a6 * (1.0 / 720.0);

        let mut v = Array2::eye(n) * 1.0;
        v = v - &a_scaled * 1.0;
        v = v + &a2 * (1.0 / 2.0);
        v = v - &a4 * (1.0 / 24.0);
        v = v + &a6 * (1.0 / 720.0);

        // Solve (V)(exp(A)) = U
        let exp_a = Self::solve_linear_system(&v, &u).unwrap_or(Array2::eye(n));

        // Squaring
        let mut result = exp_a;
        for _ in 0..s {
            result = result.dot(&result);
        }

        SLn { n, matrix: result }
    }

    fn log(&self) -> Self::Algebra {
        // Matrix logarithm - simplified implementation
        let n = self.n;
        let i = Array2::<f64>::eye(n);
        let a = &self.matrix - &i;

        // Series expansion for log(I + A)
        let mut log_m = a.clone();
        let mut a_power = a.clone();

        for k in 2..10 {
            a_power = a_power.dot(&a);
            if k % 2 == 0 {
                log_m = log_m - &a_power / (k as f64);
            } else {
                log_m = log_m + &a_power / (k as f64);
            }
        }

        Sln { n, matrix: log_m }
    }

    fn multiply(&self, other: &Self) -> Self {
        SLn {
            n: self.n,
            matrix: self.matrix.dot(&other.matrix),
        }
    }

    fn inverse(&self) -> Self {
        // For SL(n), compute matrix inverse
        SLn {
            n: self.n,
            matrix: Self::matrix_inverse(&self.matrix).unwrap_or(Array2::eye(self.n)),
        }
    }

    fn identity(&self) -> Self {
        let n = 2; // Default to SL(2)
        SLn {
            n,
            matrix: Array2::eye(n),
        }
    }
}

impl SLn {
    /// Solve linear system Ax = b
    fn solve_linear_system(a: &Array2<f64>, b: &Array2<f64>) -> Option<Array2<f64>> {
        // Simple Gaussian elimination for small matrices
        let n = a.nrows();
        let mut aug = Array2::zeros((n, 2 * n));

        // Create augmented matrix [A | B]
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
                aug[[i, n + j]] = b[[i, j]];
            }
        }

        // Forward elimination
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            for i in (k + 1)..n {
                if aug[[i, k]].abs() > aug[[max_row, k]].abs() {
                    max_row = i;
                }
            }

            // Swap rows
            for j in 0..2 * n {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }

            // Eliminate
            for i in (k + 1)..n {
                let factor = aug[[i, k]] / aug[[k, k]];
                for j in k..2 * n {
                    aug[[i, j]] -= factor * aug[[k, j]];
                }
            }
        }

        // Back substitution
        let mut result = Array2::zeros((n, n));
        for j in 0..n {
            for i in (0..n).rev() {
                let mut sum = aug[[i, n + j]];
                for k in (i + 1)..n {
                    sum -= aug[[i, k]] * result[[k, j]];
                }
                result[[i, j]] = sum / aug[[i, i]];
            }
        }

        Some(result)
    }

    /// Compute matrix inverse
    fn matrix_inverse(a: &Array2<f64>) -> Option<Array2<f64>> {
        let n = a.nrows();
        Self::solve_linear_system(a, &Array2::eye(n))
    }
}

/// Symplectic Lie algebra sp(2n)
#[derive(Debug, Clone)]
pub struct Sp2n {
    /// n (half dimension)
    pub n: usize,
    /// Matrix representation (Hamiltonian matrix)
    pub matrix: Array2<f64>,
}

impl Sp2n {
    /// Create the standard symplectic form
    pub fn omega_matrix(n: usize) -> Array2<f64> {
        let mut omega = Array2::zeros((2 * n, 2 * n));
        for i in 0..n {
            omega[[i, n + i]] = 1.0;
            omega[[n + i, i]] = -1.0;
        }
        omega
    }

    /// Check if matrix is Hamiltonian (in sp(2n))
    pub fn is_hamiltonian(&self) -> bool {
        let omega = Self::omega_matrix(self.n);
        let test = &self.matrix.t().dot(&omega) + &omega.dot(&self.matrix);
        test.iter().all(|&x| x.abs() < 1e-10)
    }
}

impl LieAlgebra for Sp2n {
    fn dim(&self) -> usize {
        // For sp(2n), dimension is 2n^2 + n
        0 // Set dynamically based on n
    }

    fn bracket(&self, other: &Self) -> Self {
        let ab = self.matrix.dot(&other.matrix);
        let ba = other.matrix.dot(&self.matrix);
        Sp2n {
            n: self.n,
            matrix: ab - ba,
        }
    }

    fn from_vector(v: &ArrayView1<f64>) -> Self {
        // Reconstruct Hamiltonian matrix from vector representation
        // Expected vector format: [A_elements, B_upper_triangle, C_upper_triangle]
        // where A is n×n, B and C are symmetric n×n

        // For now, assume n=2 (sp(4)) - can be generalized later
        let n = 2;
        let expected_size = n * n + n * (n + 1) / 2 + n * (n + 1) / 2;

        if v.len() != expected_size {
            // Fallback: try to infer n from vector length
            // For sp(2n): dim = 2n² + n = n(2n + 1)
            // Solve n(2n + 1) = v.len() for n
            let len = v.len() as f64;
            let n_float = (-1.0 + (1.0 + 8.0 * len).sqrt()) / 4.0;
            let n = n_float.round() as usize;

            if n * (2 * n + 1) != v.len() {
                // Use default n=2 and pad with zeros if necessary
                let n = 2;
                let mut matrix = Array2::zeros((2 * n, 2 * n));

                // Fill what we can from the vector
                let available = v.len().min(n * n);
                for i in 0..n {
                    for j in 0..n {
                        let idx = i * n + j;
                        if idx < available {
                            matrix[[i, j]] = v[idx];
                        }
                    }
                }

                return Sp2n { n, matrix };
            }
        }

        let n = 2; // For sp(4)
        let mut matrix = Array2::zeros((2 * n, 2 * n));
        let mut offset = 0;

        // Extract A (n×n block in upper-left)
        for i in 0..n {
            for j in 0..n {
                matrix[[i, j]] = v[offset];
                offset += 1;
            }
        }

        // Extract B (symmetric, upper-right block)
        for i in 0..n {
            for j in i..n {
                let val = v[offset];
                matrix[[i, n + j]] = val;
                if i != j {
                    matrix[[j, n + i]] = val; // Symmetry
                }
                offset += 1;
            }
        }

        // Extract C (symmetric, lower-left block)
        for i in 0..n {
            for j in i..n {
                let val = v[offset];
                matrix[[n + i, j]] = val;
                if i != j {
                    matrix[[n + j, i]] = val; // Symmetry
                }
                offset += 1;
            }
        }

        // Set -A^T in lower-right block
        for i in 0..n {
            for j in 0..n {
                matrix[[n + i, n + j]] = -matrix[[j, i]];
            }
        }

        Sp2n { n, matrix }
    }

    fn to_vector(&self) -> Array1<f64> {
        // Extract independent components
        let n = self.n;
        let mut v = Vec::new();

        // A Hamiltonian matrix has the form:
        // [ A   B  ]
        // [ C  -A^T]
        // where B and C are symmetric

        // Extract A (n x n)
        for i in 0..n {
            for j in 0..n {
                v.push(self.matrix[[i, j]]);
            }
        }

        // Extract upper triangle of B
        for i in 0..n {
            for j in i..n {
                v.push(self.matrix[[i, n + j]]);
            }
        }

        // Extract upper triangle of C
        for i in 0..n {
            for j in i..n {
                v.push(self.matrix[[n + i, j]]);
            }
        }

        Array1::from_vec(v)
    }

    fn norm(&self) -> f64 {
        self.matrix.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

/// Heisenberg Lie algebra
#[derive(Debug, Clone)]
pub struct HeisenbergAlgebra {
    /// Position component
    pub p: f64,
    /// Momentum component
    pub q: f64,
    /// Central component
    pub z: f64,
}

impl LieAlgebra for HeisenbergAlgebra {
    fn dim(&self) -> usize {
        3
    }

    fn bracket(&self, other: &Self) -> Self {
        // [·,·] gives only central component
        HeisenbergAlgebra {
            p: 0.0,
            q: 0.0,
            z: self.p * other.q - self.q * other.p,
        }
    }

    fn from_vector(v: &ArrayView1<f64>) -> Self {
        HeisenbergAlgebra {
            p: v[0],
            q: v[1],
            z: v[2],
        }
    }

    fn to_vector(&self) -> Array1<f64> {
        Array1::from_vec(vec![self.p, self.q, self.z])
    }

    fn norm(&self) -> f64 {
        (self.p * self.p + self.q * self.q + self.z * self.z).sqrt()
    }
}

/// Heisenberg group
#[derive(Debug, Clone)]
pub struct HeisenbergGroup {
    /// Position
    pub p: f64,
    /// Momentum
    pub q: f64,
    /// Phase
    pub z: f64,
}

impl ExponentialMap for HeisenbergGroup {
    type Algebra = HeisenbergAlgebra;

    fn exp(algebra: &Self::Algebra) -> Self {
        // Exponential map for Heisenberg group
        // exp(p, q, z) = (p, q, z + pq/2)
        HeisenbergGroup {
            p: algebra.p,
            q: algebra.q,
            z: algebra.z + algebra.p * algebra.q / 2.0,
        }
    }

    fn log(&self) -> Self::Algebra {
        // Logarithm map
        HeisenbergAlgebra {
            p: self.p,
            q: self.q,
            z: self.z - self.p * self.q / 2.0,
        }
    }

    fn multiply(&self, other: &Self) -> Self {
        // Group multiplication with Baker-Campbell-Hausdorff formula
        HeisenbergGroup {
            p: self.p + other.p,
            q: self.q + other.q,
            z: self.z + other.z + self.p * other.q,
        }
    }

    fn inverse(&self) -> Self {
        HeisenbergGroup {
            p: -self.p,
            q: -self.q,
            z: -self.z,
        }
    }

    fn identity(&self) -> Self {
        HeisenbergGroup {
            p: 0.0,
            q: 0.0,
            z: 0.0,
        }
    }
}

/// General Linear group GL(n)
#[derive(Debug, Clone)]
pub struct GLn {
    /// Matrix dimension
    pub n: usize,
    /// Invertible matrix
    pub matrix: Array2<f64>,
}

impl ExponentialMap for GLn {
    type Algebra = Gln;

    fn exp(algebra: &Self::Algebra) -> Self {
        // Matrix exponential
        let n = algebra.n;
        let exp_matrix = matrix_exponential(&algebra.matrix);
        GLn {
            n,
            matrix: exp_matrix,
        }
    }

    fn log(&self) -> Self::Algebra {
        // Matrix logarithm
        let log_matrix = matrix_logarithm(&self.matrix);
        Gln {
            n: self.n,
            matrix: log_matrix,
        }
    }

    fn multiply(&self, other: &Self) -> Self {
        GLn {
            n: self.n,
            matrix: self.matrix.dot(&other.matrix),
        }
    }

    fn inverse(&self) -> Self {
        GLn {
            n: self.n,
            matrix: SLn::matrix_inverse(&self.matrix).unwrap_or(Array2::eye(self.n)),
        }
    }

    fn identity(&self) -> Self {
        let n = 2; // Default size
        GLn {
            n,
            matrix: Array2::eye(n),
        }
    }
}

/// GL(n) Lie algebra
#[derive(Debug, Clone)]
pub struct Gln {
    /// Matrix dimension
    pub n: usize,
    /// Matrix
    pub matrix: Array2<f64>,
}

impl LieAlgebra for Gln {
    fn dim(&self) -> usize {
        // n^2 for gl(n)
        0 // Set dynamically
    }

    fn bracket(&self, other: &Self) -> Self {
        let ab = self.matrix.dot(&other.matrix);
        let ba = other.matrix.dot(&self.matrix);
        Gln {
            n: self.n,
            matrix: ab - ba,
        }
    }

    fn from_vector(v: &ArrayView1<f64>) -> Self {
        let n = (v.len() as f64).sqrt() as usize;
        let matrix = Array2::from_shape_vec((n, n), v.to_vec()).unwrap();
        Gln { n, matrix }
    }

    fn to_vector(&self) -> Array1<f64> {
        Array1::from_vec(self.matrix.as_slice().unwrap().to_vec())
    }

    fn norm(&self) -> f64 {
        self.matrix.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

/// Helper function for matrix exponential
#[allow(dead_code)]
fn matrix_exponential(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut result = Array2::eye(n);
    let mut term = Array2::eye(n);

    for k in 1..20 {
        term = term.dot(a) / (k as f64);
        result += &term;

        if term.iter().map(|&x| x.abs()).sum::<f64>() < 1e-12 {
            break;
        }
    }

    result
}

/// Helper function for matrix logarithm
#[allow(dead_code)]
fn matrix_logarithm(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let i = Array2::<f64>::eye(n);
    let x = a - &i;

    let mut result = Array2::zeros((n, n));
    let mut term = x.clone();

    for k in 1..20 {
        if k % 2 == 1 {
            result = result + &term / (k as f64);
        } else {
            result = result - &term / (k as f64);
        }

        term = term.dot(&x);

        if term.iter().map(|&x| x.abs()).sum::<f64>() < 1e-12 {
            break;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_so3_exp_log() {
        let omega = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let algebra = So3 { omega };

        let group = SO3::exp(&algebra);
        let algebra_recovered = group.log();

        for i in 0..3 {
            assert_relative_eq!(
                algebra.omega[i],
                algebra_recovered.omega[i],
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_rigid_body_energy_conservation() {
        let dt = 0.01;
        let integrator = SO3Integrator::new(dt, LieGroupMethod::RKMK4);

        // Initial conditions
        let orientation = SO3 {
            matrix: Array2::eye(3),
        };
        let angular_velocity = Array1::from_vec(vec![0.1, 0.5, 0.3]);
        let inertia =
            Array2::from_shape_vec((3, 3), vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0])
                .unwrap();
        let external_torque = Array1::zeros(3);

        // Initial energy
        let initial_energy = 0.5 * angular_velocity.dot(&inertia.dot(&angular_velocity));

        // Integrate
        let states = integrator
            .integrate_rigid_body(
                &orientation,
                &angular_velocity,
                &inertia,
                &external_torque,
                100,
            )
            .unwrap();

        // Check energy conservation
        let (_, final_omega) = &states.last().unwrap();
        let final_energy = 0.5 * final_omega.dot(&inertia.dot(final_omega));

        assert_relative_eq!(initial_energy, final_energy, epsilon = 1e-4);
    }
}
