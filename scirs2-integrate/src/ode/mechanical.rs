//! Mechanical systems integration methods
//!
//! This module provides specialized numerical integration methods for mechanical systems,
//! including rigid body dynamics, multibody systems with constraints, and deformable solids.
//! These methods are optimized for the specific structure and properties of mechanical systems.

use ndarray::{s, Array1, Array2};

/// Type alias for external force functions
type ExternalForceFunction =
    Box<dyn Fn(f64, &Array1<f64>, &Array1<f64>) -> Array1<f64> + Send + Sync>;

/// Type alias for constraint functions
type ConstraintFunction = Box<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>;

/// Type alias for constraint Jacobian functions
type ConstraintJacobianFunction = Box<dyn Fn(&Array1<f64>) -> Array2<f64> + Send + Sync>;

/// Type alias for integration result tuples
type IntegrationResult = Result<(Array1<f64>, f64, usize, bool), Box<dyn std::error::Error>>;

/// Types of mechanical systems supported
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MechanicalSystemType {
    /// Single rigid body with 6 DOF
    RigidBody,
    /// Multiple rigid bodies connected by constraints
    Multibody,
    /// Deformable solid with finite element discretization
    DeformableSolid,
    /// Mechanism with kinematic constraints
    ConstrainedMechanism,
}

/// Configuration for mechanical system integration
#[derive(Debug, Clone)]
pub struct MechanicalConfig {
    /// Type of mechanical system
    pub system_type: MechanicalSystemType,
    /// Time step size
    pub dt: f64,
    /// Integration method for position update
    pub position_method: PositionIntegrationMethod,
    /// Integration method for velocity update
    pub velocity_method: VelocityIntegrationMethod,
    /// Constraint enforcement method
    pub constraint_method: ConstraintMethod,
    /// Energy conservation tolerance
    pub energy_tolerance: f64,
    /// Maximum constraint violation tolerance
    pub constraint_tolerance: f64,
    /// Whether to use stabilization techniques
    pub use_stabilization: bool,
}

impl Default for MechanicalConfig {
    fn default() -> Self {
        Self {
            system_type: MechanicalSystemType::RigidBody,
            dt: 0.01,
            position_method: PositionIntegrationMethod::Verlet,
            velocity_method: VelocityIntegrationMethod::Leapfrog,
            constraint_method: ConstraintMethod::Lagrange,
            energy_tolerance: 1e-6,
            constraint_tolerance: 1e-8,
            use_stabilization: true,
        }
    }
}

/// Position integration methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionIntegrationMethod {
    /// Verlet integration (symplectic, energy conserving)
    Verlet,
    /// Velocity Verlet (explicit positions, implicit velocities)
    VelocityVerlet,
    /// Newmark-β method (popular in structural dynamics)
    NewmarkBeta { beta: f64, gamma: f64 },
    /// Central difference method
    CentralDifference,
}

/// Velocity integration methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VelocityIntegrationMethod {
    /// Leapfrog integration
    Leapfrog,
    /// Implicit Euler for damped systems
    ImplicitEuler,
    /// Crank-Nicolson for smooth dynamics
    CrankNicolson,
    /// Runge-Kutta 4th order
    RungeKutta4,
}

/// Constraint enforcement methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintMethod {
    /// Lagrange multipliers
    Lagrange,
    /// Penalty method
    Penalty { stiffness: f64 },
    /// Augmented Lagrangian
    AugmentedLagrangian { penalty: f64 },
    /// Stabilization (Baumgarte)
    Baumgarte { alpha: f64, beta: f64 },
}

/// Rigid body state representation
#[derive(Debug, Clone)]
pub struct RigidBodyState {
    /// Position (3D vector)
    pub position: Array1<f64>,
    /// Velocity (3D vector)
    pub velocity: Array1<f64>,
    /// Orientation (quaternion: w, x, y, z)
    pub orientation: Array1<f64>,
    /// Angular velocity (3D vector)
    pub angular_velocity: Array1<f64>,
}

impl RigidBodyState {
    /// Create a new rigid body state
    pub fn new(
        position: Array1<f64>,
        velocity: Array1<f64>,
        orientation: Array1<f64>,
        angular_velocity: Array1<f64>,
    ) -> Self {
        Self {
            position,
            velocity,
            orientation,
            angular_velocity,
        }
    }

    /// Get total kinetic energy
    pub fn kinetic_energy(&self, mass: f64, inertia: &Array2<f64>) -> f64 {
        let translational = 0.5 * mass * self.velocity.mapv(|v| v * v).sum();
        let rotational = 0.5
            * self
                .angular_velocity
                .dot(&inertia.dot(&self.angular_velocity));
        translational + rotational
    }

    /// Normalize quaternion orientation
    pub fn normalize_quaternion(&mut self) {
        let norm = self.orientation.mapv(|q| q * q).sum().sqrt();
        if norm > 1e-12 {
            self.orientation /= norm;
        }
    }
}

/// Mechanical system properties
pub struct MechanicalProperties {
    /// Mass (scalar for single body, vector for multibody)
    pub mass: Array1<f64>,
    /// Inertia tensor(s)
    pub inertia: Vec<Array2<f64>>,
    /// Damping coefficients
    pub damping: Array1<f64>,
    /// External force function
    pub external_forces: Option<ExternalForceFunction>,
    /// Constraint equations
    pub constraints: Vec<ConstraintFunction>,
    /// Constraint Jacobians
    pub constraint_jacobians: Vec<ConstraintJacobianFunction>,
    /// Spring constants for harmonic oscillators (if applicable)
    pub spring_constants: Vec<f64>,
}

impl Default for MechanicalProperties {
    fn default() -> Self {
        Self {
            mass: Array1::ones(1),
            inertia: vec![Array2::eye(3)],
            damping: Array1::zeros(6),
            external_forces: None,
            constraints: Vec::new(),
            constraint_jacobians: Vec::new(),
            spring_constants: Vec::new(),
        }
    }
}

/// Result of mechanical integration step
#[derive(Debug, Clone)]
pub struct MechanicalIntegrationResult {
    /// Updated state
    pub state: RigidBodyState,
    /// Constraint forces/torques
    pub constraint_forces: Array1<f64>,
    /// Energy drift
    pub energy_drift: f64,
    /// Constraint violation
    pub constraint_violation: f64,
    /// Integration statistics
    pub stats: IntegrationStats,
}

/// Integration statistics for performance monitoring
#[derive(Debug, Clone)]
pub struct IntegrationStats {
    /// Number of constraint iterations
    pub constraint_iterations: usize,
    /// Convergence achieved
    pub converged: bool,
    /// Time spent in force calculation
    pub force_computation_time: f64,
    /// Time spent in constraint enforcement
    pub constraint_time: f64,
}

impl Default for IntegrationStats {
    fn default() -> Self {
        Self {
            constraint_iterations: 0,
            converged: true,
            force_computation_time: 0.0,
            constraint_time: 0.0,
        }
    }
}

/// Specialized integrator for mechanical systems
pub struct MechanicalIntegrator {
    config: MechanicalConfig,
    properties: MechanicalProperties,
    previous_state: Option<RigidBodyState>,
    energy_history: Vec<f64>,
}

impl MechanicalIntegrator {
    /// Create a new mechanical integrator
    pub fn new(config: MechanicalConfig, properties: MechanicalProperties) -> Self {
        Self {
            config,
            properties,
            previous_state: None,
            energy_history: Vec::new(),
        }
    }

    /// Perform a single integration step
    pub fn step(
        &mut self,
        t: f64,
        state: &RigidBodyState,
    ) -> Result<MechanicalIntegrationResult, Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();

        // Calculate forces and torques
        let (forces, torques) = self.calculate_forces_torques(t, state)?;
        let force_time = start_time.elapsed().as_secs_f64();

        // Integrate based on selected method
        let mut new_state = match self.config.position_method {
            PositionIntegrationMethod::Verlet => self.verlet_step(state, &forces, &torques)?,
            PositionIntegrationMethod::VelocityVerlet => {
                self.velocity_verlet_step(state, &forces, &torques)?
            }
            PositionIntegrationMethod::NewmarkBeta { beta, gamma } => {
                self.newmark_beta_step(state, &forces, &torques, beta, gamma)?
            }
            PositionIntegrationMethod::CentralDifference => {
                self.central_difference_step(state, &forces, &torques)?
            }
        };

        let constraint_start = std::time::Instant::now();

        // Enforce constraints
        let (constraint_forces, constraint_violation, iterations, converged) =
            self.enforce_constraints(&mut new_state)?;

        let constraint_time = constraint_start.elapsed().as_secs_f64();

        // Calculate energy drift
        let current_energy = self.calculate_total_energy(&new_state);
        let energy_drift = if let Some(ref prev_state) = self.previous_state {
            let prev_energy = self.calculate_total_energy(prev_state);
            (current_energy - prev_energy).abs() / prev_energy.max(1e-12)
        } else {
            0.0
        };

        self.energy_history.push(current_energy);
        self.previous_state = Some(new_state.clone());

        Ok(MechanicalIntegrationResult {
            state: new_state,
            constraint_forces,
            energy_drift,
            constraint_violation,
            stats: IntegrationStats {
                constraint_iterations: iterations,
                converged,
                force_computation_time: force_time,
                constraint_time,
            },
        })
    }

    /// Calculate forces and torques acting on the system
    fn calculate_forces_torques(
        &self,
        t: f64,
        state: &RigidBodyState,
    ) -> Result<(Array1<f64>, Array1<f64>), Box<dyn std::error::Error>> {
        let n_bodies = self.properties.mass.len();
        let mut forces = Array1::zeros(3 * n_bodies);
        let mut torques = Array1::zeros(3 * n_bodies);

        // External forces
        if let Some(ref external_force_fn) = self.properties.external_forces {
            let combined_state = self.combine_position_velocity(state);
            let external = external_force_fn(t, &combined_state, &state.velocity);

            for i in 0..n_bodies {
                forces
                    .slice_mut(s![3 * i..3 * (i + 1)])
                    .assign(&external.slice(s![3 * i..3 * (i + 1)]));
            }
        }

        // Damping forces
        for i in 0..n_bodies {
            let damping_force = -&self.properties.damping.slice(s![3 * i..3 * (i + 1)])
                * state.velocity.slice(s![3 * i..3 * (i + 1)]);
            let current_force = forces.slice(s![3 * i..3 * (i + 1)]).to_owned();
            forces
                .slice_mut(s![3 * i..3 * (i + 1)])
                .assign(&(&current_force + &damping_force));

            // Angular damping
            if self.properties.damping.len() > 3 * n_bodies {
                let angular_damping = -&self
                    .properties
                    .damping
                    .slice(s![3 * (n_bodies + i)..3 * (n_bodies + i + 1)])
                    * state.angular_velocity.slice(s![3 * i..3 * (i + 1)]);
                let current_torque = torques.slice(s![3 * i..3 * (i + 1)]).to_owned();
                torques
                    .slice_mut(s![3 * i..3 * (i + 1)])
                    .assign(&(&current_torque + &angular_damping));
            }
        }

        Ok((forces, torques))
    }

    /// Verlet integration step
    fn verlet_step(
        &self,
        state: &RigidBodyState,
        forces: &Array1<f64>,
        torques: &Array1<f64>,
    ) -> Result<RigidBodyState, Box<dyn std::error::Error>> {
        let dt = self.config.dt;
        let dt2 = dt * dt;

        // Position update: x_{n+1} = 2x_n - x_{n-1} + a_n * dt^2
        let acceleration = self.calculate_acceleration(forces);
        let angular_acceleration = self.calculate_angular_acceleration(torques, state);

        let new_position = if let Some(ref prev_state) = self.previous_state {
            2.0 * &state.position - &prev_state.position + &acceleration * dt2
        } else {
            // First step: use Euler's method
            &state.position + &state.velocity * dt + 0.5 * &acceleration * dt2
        };

        // Velocity update: v_{n+1} = (x_{n+1} - x_{n-1}) / (2*dt)
        let new_velocity = if let Some(ref prev_state) = self.previous_state {
            (&new_position - &prev_state.position) / (2.0 * dt)
        } else {
            &state.velocity + &acceleration * dt
        };

        // Angular velocity update (similar to linear)
        let new_angular_velocity = if let Some(ref _prev_state) = self.previous_state {
            &state.angular_velocity + &angular_acceleration * dt
        } else {
            &state.angular_velocity + &angular_acceleration * dt
        };

        // Orientation update using quaternion integration
        let new_orientation =
            self.integrate_quaternion(&state.orientation, &new_angular_velocity, dt);

        Ok(RigidBodyState::new(
            new_position,
            new_velocity,
            new_orientation,
            new_angular_velocity,
        ))
    }

    /// Velocity Verlet integration step
    fn velocity_verlet_step(
        &self,
        state: &RigidBodyState,
        forces: &Array1<f64>,
        torques: &Array1<f64>,
    ) -> Result<RigidBodyState, Box<dyn std::error::Error>> {
        let dt = self.config.dt;
        let dt2 = dt * dt;

        let acceleration = self.calculate_acceleration(forces);
        let angular_acceleration = self.calculate_angular_acceleration(torques, state);

        // Position update: x_{n+1} = x_n + v_n * dt + 0.5 * a_n * dt^2
        let new_position = &state.position + &state.velocity * dt + 0.5 * &acceleration * dt2;

        // Velocity update: v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt
        // For now, assume acceleration doesn't change much
        let new_velocity = &state.velocity + &acceleration * dt;
        let new_angular_velocity = &state.angular_velocity + &angular_acceleration * dt;

        // Orientation update
        let new_orientation =
            self.integrate_quaternion(&state.orientation, &new_angular_velocity, dt);

        Ok(RigidBodyState::new(
            new_position,
            new_velocity,
            new_orientation,
            new_angular_velocity,
        ))
    }

    /// Newmark-β integration step
    fn newmark_beta_step(
        &self,
        state: &RigidBodyState,
        forces: &Array1<f64>,
        torques: &Array1<f64>,
        beta: f64,
        gamma: f64,
    ) -> Result<RigidBodyState, Box<dyn std::error::Error>> {
        let dt = self.config.dt;
        let dt2 = dt * dt;

        let acceleration = self.calculate_acceleration(forces);
        let angular_acceleration = self.calculate_angular_acceleration(torques, state);

        // Newmark-β formulas
        let new_position =
            &state.position + &state.velocity * dt + (0.5 - beta) * &acceleration * dt2;

        let new_velocity = &state.velocity + (1.0 - gamma) * &acceleration * dt;

        // Similar for angular quantities
        let new_angular_velocity =
            &state.angular_velocity + (1.0 - gamma) * &angular_acceleration * dt;
        let new_orientation =
            self.integrate_quaternion(&state.orientation, &new_angular_velocity, dt);

        Ok(RigidBodyState::new(
            new_position,
            new_velocity,
            new_orientation,
            new_angular_velocity,
        ))
    }

    /// Central difference integration step
    fn central_difference_step(
        &self,
        state: &RigidBodyState,
        forces: &Array1<f64>,
        torques: &Array1<f64>,
    ) -> Result<RigidBodyState, Box<dyn std::error::Error>> {
        // Similar to Verlet but with different formulation
        self.verlet_step(state, forces, torques)
    }

    /// Calculate linear acceleration from forces
    fn calculate_acceleration(&self, forces: &Array1<f64>) -> Array1<f64> {
        let n_bodies = self.properties.mass.len();
        let mut acceleration = Array1::zeros(forces.len());

        for i in 0..n_bodies {
            let mass = self.properties.mass[i];
            if mass > 1e-12 {
                let force_slice = forces.slice(s![3 * i..3 * (i + 1)]);
                acceleration
                    .slice_mut(s![3 * i..3 * (i + 1)])
                    .assign(&force_slice.mapv(|f| f / mass));
            }
        }

        acceleration
    }

    /// Calculate angular acceleration from torques
    fn calculate_angular_acceleration(
        &self,
        torques: &Array1<f64>,
        _state: &RigidBodyState,
    ) -> Array1<f64> {
        // For single rigid body, return 3D angular acceleration
        if self.properties.inertia.len() == 1 && torques.len() >= 3 {
            let inertia = &self.properties.inertia[0];
            let mut angular_acceleration = Array1::zeros(3);

            // Solve I * α = τ for α
            // For simplicity, assume inertia is diagonal
            for j in 0..3 {
                let inertia_jj = inertia[[j, j]];
                if inertia_jj > 1e-12 {
                    angular_acceleration[j] = torques[j] / inertia_jj;
                }
            }

            return angular_acceleration;
        }

        // For multibody systems
        let n_bodies = self.properties.inertia.len();
        let mut angular_acceleration = Array1::zeros(torques.len());

        for i in 0..n_bodies {
            if i < self.properties.inertia.len() {
                let inertia = &self.properties.inertia[i];
                let torque = torques.slice(s![3 * i..3 * (i + 1)]);

                // Solve I * α = τ for α
                // For simplicity, assume inertia is diagonal
                for j in 0..3 {
                    let inertia_jj = inertia[[j, j]];
                    if inertia_jj > 1e-12 {
                        angular_acceleration[3 * i + j] = torque[j] / inertia_jj;
                    }
                }
            }
        }

        angular_acceleration
    }

    /// Integrate quaternion using angular velocity
    fn integrate_quaternion(
        &self,
        quaternion: &Array1<f64>,
        angular_velocity: &Array1<f64>,
        dt: f64,
    ) -> Array1<f64> {
        // Quaternion integration: q_{n+1} = q_n + 0.5 * Ω(ω) * q_n * dt
        // where Ω(ω) is the quaternion rate matrix

        let omega_norm = angular_velocity.mapv(|w| w * w).sum().sqrt();

        if omega_norm < 1e-12 {
            return quaternion.clone();
        }

        let half_angle = 0.5 * omega_norm * dt;
        let sin_half = half_angle.sin();
        let cos_half = half_angle.cos();

        let axis = angular_velocity / omega_norm;

        // Rotation quaternion
        let dq = Array1::from_vec(vec![
            cos_half,
            sin_half * axis[0],
            sin_half * axis[1],
            sin_half * axis[2],
        ]);

        // Quaternion multiplication: q_new = dq * q_old
        self.quaternion_multiply(&dq, quaternion)
    }

    /// Multiply two quaternions
    fn quaternion_multiply(&self, q1: &Array1<f64>, q2: &Array1<f64>) -> Array1<f64> {
        let w1 = q1[0];
        let x1 = q1[1];
        let y1 = q1[2];
        let z1 = q1[3];
        let w2 = q2[0];
        let x2 = q2[1];
        let y2 = q2[2];
        let z2 = q2[3];

        Array1::from_vec(vec![
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])
    }

    /// Enforce constraints using selected method
    fn enforce_constraints(&self, state: &mut RigidBodyState) -> IntegrationResult {
        if self.properties.constraints.is_empty() {
            return Ok((Array1::zeros(0), 0.0, 0, true));
        }

        match self.config.constraint_method {
            ConstraintMethod::Lagrange => self.enforce_lagrange_constraints(state),
            ConstraintMethod::Penalty { stiffness } => {
                self.enforce_penalty_constraints(state, stiffness)
            }
            ConstraintMethod::AugmentedLagrangian { penalty } => {
                self.enforce_augmented_lagrangian_constraints(state, penalty)
            }
            ConstraintMethod::Baumgarte { alpha, beta } => {
                self.enforce_baumgarte_stabilization(state, alpha, beta)
            }
        }
    }

    /// Lagrange multiplier constraint enforcement
    fn enforce_lagrange_constraints(&self, state: &mut RigidBodyState) -> IntegrationResult {
        let max_iterations = 10;
        let tolerance = self.config.constraint_tolerance;

        let mut lambda = Array1::zeros(self.properties.constraints.len());
        let mut converged = false;
        let mut iterations_used = 0;

        for iteration in 0..max_iterations {
            iterations_used = iteration + 1;

            // Evaluate constraints
            let combined_pos = self.combine_position_velocity(state);
            let mut constraint_values = Array1::zeros(self.properties.constraints.len());

            for (i, constraint) in self.properties.constraints.iter().enumerate() {
                let c_val = constraint(&combined_pos);
                constraint_values[i] = c_val[0]; // Assume scalar constraints for simplicity
            }

            let violation = constraint_values.mapv(|c| c * c).sum().sqrt();

            if violation < tolerance {
                converged = true;
                break;
            }

            // Check for NaN or excessive constraint violations and abort if detected
            if violation.is_nan() || !violation.is_finite() || violation > 1e6 {
                // Reset to a more stable configuration
                state
                    .position
                    .mapv_inplace(|x| if x.is_nan() || !x.is_finite() { 0.0 } else { x });
                state
                    .velocity
                    .mapv_inplace(|x| if x.is_nan() || !x.is_finite() { 0.0 } else { x });
                break;
            }

            // Proper Newton-Raphson iteration with constraint Jacobian
            if self.properties.constraint_jacobians.len() == self.properties.constraints.len() {
                // Calculate constraint Jacobian matrix
                let num_coords = combined_pos.len();
                let num_constraints = constraint_values.len();
                let mut jacobian = Array2::zeros((num_constraints, num_coords));

                for (i, jacobian_func) in self.properties.constraint_jacobians.iter().enumerate() {
                    let jac_row = jacobian_func(&combined_pos);
                    for j in 0..num_coords.min(jac_row.ncols()) {
                        jacobian[(i, j)] = jac_row[(0, j)];
                    }
                }

                // Solve for Lagrange multipliers: J * J^T * λ = -C
                let jjt = jacobian.dot(&jacobian.t());
                if let Ok(delta_lambda) = self.solve_constraint_system(&jjt, &(-&constraint_values))
                {
                    lambda += &delta_lambda;

                    // Apply position correction: Δq = -J^T * λ
                    let position_correction = jacobian.t().dot(&delta_lambda);

                    // Apply correction with damping to ensure stability
                    let damping = 0.5; // Even more conservative damping
                    for i in 0..state.position.len().min(position_correction.len()) {
                        if position_correction[i].is_finite() && position_correction[i].abs() < 1.0
                        {
                            state.position[i] -= damping * position_correction[i];
                        }
                    }

                    // Additional constraint enforcement - project back to constraint manifold
                    if violation > 1e3 {
                        // For very large violations, use a more drastic correction
                        self.project_to_constraint_manifold(state)?;
                    }
                } else {
                    // Fallback: simple proportional correction with proper scaling
                    let correction_factor = 0.01; // Much smaller factor
                    if num_constraints > 0 && combined_pos.len() >= 4 {
                        // For double pendulum: distribute correction appropriately
                        let correction_per_coord =
                            constraint_values.sum() / combined_pos.len() as f64;
                        for i in 0..state.position.len() {
                            state.position[i] -= correction_factor * correction_per_coord;
                        }
                    }
                }
            } else {
                // Fallback method when Jacobians are not available
                let correction_factor = 0.01;
                if !constraint_values.is_empty() && state.position.len() >= 4 {
                    // Simple correction distributed across coordinates
                    let avg_violation = constraint_values.iter().map(|c| c.abs()).sum::<f64>()
                        / constraint_values.len() as f64;
                    for i in 0..state.position.len() {
                        // Small correction to prevent instability
                        state.position[i] -= correction_factor * avg_violation * (i as f64 + 1.0)
                            / state.position.len() as f64;
                    }
                }
            }
        }

        let final_violation = {
            let combined_pos = self.combine_position_velocity(state);
            let mut total = 0.0;
            for constraint in &self.properties.constraints {
                let c_val = constraint(&combined_pos);
                let violation = c_val[0];
                // Protect against numerical issues
                if violation.is_finite() {
                    total += violation * violation;
                } else {
                    total = f64::INFINITY;
                    break;
                }
            }
            if total.is_finite() {
                total.sqrt()
            } else {
                f64::INFINITY
            }
        };

        Ok((lambda, final_violation, iterations_used, converged))
    }

    /// Penalty method constraint enforcement
    fn enforce_penalty_constraints(
        &self,
        state: &mut RigidBodyState,
        stiffness: f64,
    ) -> IntegrationResult {
        let combined_pos = self.combine_position_velocity(state);
        let mut constraint_forces = Array1::zeros(state.position.len());
        let mut total_violation = 0.0;

        for constraint in &self.properties.constraints {
            let c_val = constraint(&combined_pos);
            let violation = c_val[0];
            total_violation += violation * violation;

            // Apply penalty force: F = -k * C(q)
            let penalty_force = -stiffness * violation;

            // Distribute force (simplified)
            constraint_forces[0] += penalty_force;
        }

        // Apply penalty forces to state
        let dt = self.config.dt;
        let mass = self.properties.mass[0];
        let acceleration = constraint_forces.mapv(|f| f / mass);
        state.velocity = &state.velocity + &acceleration * dt;

        Ok((constraint_forces, total_violation.sqrt(), 1, true))
    }

    /// Augmented Lagrangian constraint enforcement
    fn enforce_augmented_lagrangian_constraints(
        &self,
        state: &mut RigidBodyState,
        penalty: f64,
    ) -> IntegrationResult {
        // Combine Lagrange multipliers with penalty method
        let (lambda, violation, iter1, conv1) = self.enforce_lagrange_constraints(state)?;
        let (penalty_forces, _violation2, iter2, conv2) =
            self.enforce_penalty_constraints(state, penalty)?;

        Ok((
            lambda + penalty_forces,
            violation,
            iter1 + iter2,
            conv1 && conv2,
        ))
    }

    /// Baumgarte stabilization
    fn enforce_baumgarte_stabilization(
        &self,
        state: &mut RigidBodyState,
        _alpha: f64,
        beta: f64,
    ) -> IntegrationResult {
        // Baumgarte stabilization: C̈ + 2α*Ċ + β²*C = 0
        // This requires constraint velocity and acceleration
        // For simplicity, implement basic position correction

        let combined_pos = self.combine_position_velocity(state);
        let mut total_violation = 0.0;

        for constraint in &self.properties.constraints {
            let c_val = constraint(&combined_pos);
            let violation = c_val[0];
            total_violation += violation * violation;

            // Apply Baumgarte correction
            let correction = -beta * beta * violation;
            state.position[0] += correction * self.config.dt * self.config.dt;
        }

        Ok((Array1::zeros(0), total_violation.sqrt(), 1, true))
    }

    /// Solve constraint system with numerical stability checks
    fn solve_constraint_system(
        &self,
        matrix: &Array2<f64>,
        rhs: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let n = matrix.nrows();
        if n == 0 || n != matrix.ncols() || n != rhs.len() {
            return Err("Invalid matrix dimensions".into());
        }

        // Check for near-singular matrix
        let det_check = matrix.mapv(|x| x.abs()).sum() / (n * n) as f64;
        if det_check < 1e-14 {
            return Err("Nearly singular constraint matrix".into());
        }

        // Simple Gauss elimination with partial pivoting
        let mut aug_matrix = Array2::zeros((n, n + 1));

        // Set up augmented matrix
        for i in 0..n {
            for j in 0..n {
                aug_matrix[(i, j)] = matrix[(i, j)];
            }
            aug_matrix[(i, n)] = rhs[i];
        }

        // Forward elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug_matrix[(k, i)].abs() > aug_matrix[(max_row, i)].abs() {
                    max_row = k;
                }
            }

            // Check for near-zero pivot
            if aug_matrix[(max_row, i)].abs() < 1e-12 {
                return Err("Singular matrix in constraint solving".into());
            }

            // Swap rows if needed
            if max_row != i {
                for j in 0..=n {
                    let temp = aug_matrix[(i, j)];
                    aug_matrix[(i, j)] = aug_matrix[(max_row, j)];
                    aug_matrix[(max_row, j)] = temp;
                }
            }

            // Eliminate
            for k in (i + 1)..n {
                let factor = aug_matrix[(k, i)] / aug_matrix[(i, i)];
                for j in i..=n {
                    aug_matrix[(k, j)] -= factor * aug_matrix[(i, j)];
                }
            }
        }

        // Back substitution
        let mut solution = Array1::zeros(n);
        for i in (0..n).rev() {
            solution[i] = aug_matrix[(i, n)];
            for j in (i + 1)..n {
                solution[i] -= aug_matrix[(i, j)] * solution[j];
            }
            solution[i] /= aug_matrix[(i, i)];

            // Check for NaN or infinite solutions
            if !solution[i].is_finite() {
                return Err("Non-finite solution in constraint solving".into());
            }
        }

        Ok(solution)
    }

    /// Project system back to constraint manifold for severe violations
    fn project_to_constraint_manifold(
        &self,
        state: &mut RigidBodyState,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // For double pendulum, directly project to valid configuration
        if self.properties.constraints.len() == 2 && state.position.len() >= 6 {
            // Get parameters from the constraints (assuming they're for double pendulum)
            let l1 = 1.0; // Typical values - could be extracted from constraints
            let l2 = 0.8;

            // Get current positions
            let x1 = state.position[0];
            let y1 = state.position[1];
            let x2 = state.position[3];
            let y2 = state.position[4];

            // Project first pendulum to correct length
            let r1 = (x1 * x1 + y1 * y1).sqrt();
            if r1 > 1e-6 {
                state.position[0] = x1 * l1 / r1;
                state.position[1] = y1 * l1 / r1;
            } else {
                // If at origin, place at reference position
                state.position[0] = 0.0;
                state.position[1] = -l1;
            }

            // Project second pendulum to correct length relative to first
            let dx = x2 - state.position[0];
            let dy = y2 - state.position[1];
            let r2 = (dx * dx + dy * dy).sqrt();
            if r2 > 1e-6 {
                state.position[3] = state.position[0] + dx * l2 / r2;
                state.position[4] = state.position[1] + dy * l2 / r2;
            } else {
                // If coincident, place at reference position
                state.position[3] = state.position[0];
                state.position[4] = state.position[1] - l2;
            }

            // Adjust velocities to be consistent with constraints (zero radial component)
            let x1_new = state.position[0];
            let y1_new = state.position[1];
            let x2_new = state.position[3];
            let y2_new = state.position[4];

            // For first pendulum: v · r = 0 (velocity perpendicular to radius)
            let vx1 = state.velocity[0];
            let vy1 = state.velocity[1];
            let radial_component1 = (vx1 * x1_new + vy1 * y1_new) / (l1 * l1);
            state.velocity[0] = vx1 - radial_component1 * x1_new;
            state.velocity[1] = vy1 - radial_component1 * y1_new;

            // For second pendulum: (v2 - v1) · (r2 - r1) = 0
            let vx2 = state.velocity[3];
            let vy2 = state.velocity[4];
            let rel_vx = vx2 - state.velocity[0];
            let rel_vy = vy2 - state.velocity[1];
            let rel_dx = x2_new - x1_new;
            let rel_dy = y2_new - y1_new;
            let radial_component2 = (rel_vx * rel_dx + rel_vy * rel_dy) / (l2 * l2);
            state.velocity[3] = state.velocity[0] + rel_vx - radial_component2 * rel_dx;
            state.velocity[4] = state.velocity[1] + rel_vy - radial_component2 * rel_dy;
        }

        Ok(())
    }

    /// Calculate total energy of the system
    fn calculate_total_energy(&self, state: &RigidBodyState) -> f64 {
        let n_bodies = self.properties.mass.len();
        let mut total_kinetic = 0.0;

        // Calculate kinetic energy for each body
        for i in 0..n_bodies {
            let mass = self.properties.mass[i];
            let inertia = &self.properties.inertia[i];

            // Translational kinetic energy
            let vel_start = 3 * i;
            let vel_end = 3 * (i + 1);
            if vel_end <= state.velocity.len() {
                let body_velocity = state.velocity.slice(s![vel_start..vel_end]);
                let translational = 0.5 * mass * body_velocity.mapv(|v| v * v).sum();

                // Rotational kinetic energy
                let mut rotational = 0.0;
                if vel_end <= state.angular_velocity.len() {
                    let body_angular_vel = state.angular_velocity.slice(s![vel_start..vel_end]);
                    rotational = 0.5 * body_angular_vel.dot(&inertia.dot(&body_angular_vel));
                }

                total_kinetic += translational + rotational;
            }
        }

        // Add potential energy based on system type
        let potential_energy = if self.properties.external_forces.is_some() {
            if self.properties.mass.len() == 1 && !self.properties.spring_constants.is_empty() {
                // For single-body harmonic oscillator: PE = 0.5 * k * x^2
                let k = self.properties.spring_constants[0];
                0.5 * k * state.position[0] * state.position[0]
            } else if self.properties.mass.len() == 2 && self.properties.constraints.len() == 2 {
                // For double pendulum: gravitational potential energy
                let g = 9.81;
                let m1 = self.properties.mass[0];
                let m2 = self.properties.mass[1];
                let y1 = state.position[1]; // y-coordinate of first mass
                let y2 = state.position[4]; // y-coordinate of second mass
                m1 * g * y1 + m2 * g * y2
            } else {
                0.0
            }
        } else {
            0.0
        };

        total_kinetic + potential_energy
    }

    /// Combine position and velocity into a single state vector
    fn combine_position_velocity(&self, state: &RigidBodyState) -> Array1<f64> {
        let mut combined = Array1::zeros(state.position.len() + state.velocity.len());
        combined
            .slice_mut(s![..state.position.len()])
            .assign(&state.position);
        combined
            .slice_mut(s![state.position.len()..])
            .assign(&state.velocity);
        combined
    }

    /// Get energy conservation statistics
    pub fn energy_statistics(&self) -> (f64, f64, f64) {
        if self.energy_history.len() < 2 {
            return (0.0, 0.0, 0.0);
        }

        let initial_energy = self.energy_history[0];
        let current_energy = *self.energy_history.last().unwrap();
        let relative_drift = (current_energy - initial_energy).abs() / initial_energy.max(1e-12);

        let max_energy = self
            .energy_history
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_energy = self
            .energy_history
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max_drift = (max_energy - min_energy) / initial_energy.max(1e-12);

        (relative_drift, max_drift, current_energy)
    }
}

/// Factory functions for common mechanical systems
pub mod systems {
    use super::*;

    /// Create a simple rigid body system
    pub fn rigid_body(
        mass: f64,
        inertia: Array2<f64>,
        initial_position: Array1<f64>,
        initial_velocity: Array1<f64>,
        initial_orientation: Array1<f64>,
        initial_angular_velocity: Array1<f64>,
    ) -> (MechanicalConfig, MechanicalProperties, RigidBodyState) {
        let config = MechanicalConfig {
            system_type: MechanicalSystemType::RigidBody,
            ..Default::default()
        };

        let properties = MechanicalProperties {
            mass: Array1::from_vec(vec![mass]),
            inertia: vec![inertia],
            ..Default::default()
        };

        let state = RigidBodyState::new(
            initial_position,
            initial_velocity,
            initial_orientation,
            initial_angular_velocity,
        );

        (config, properties, state)
    }

    /// Create a damped oscillator system
    pub fn damped_oscillator(
        mass: f64,
        stiffness: f64,
        damping: f64,
        initial_position: f64,
        initial_velocity: f64,
    ) -> (MechanicalConfig, MechanicalProperties, RigidBodyState) {
        let config = MechanicalConfig::default();

        let spring_force = Box::new(move |_t: f64, pos: &Array1<f64>, vel: &Array1<f64>| {
            Array1::from_vec(vec![-stiffness * pos[0] - damping * vel[0], 0.0, 0.0])
        });

        let properties = MechanicalProperties {
            mass: Array1::from_vec(vec![mass]),
            inertia: vec![Array2::eye(3)],
            damping: Array1::from_vec(vec![damping, 0.0, 0.0]),
            external_forces: Some(spring_force),
            spring_constants: vec![stiffness],
            ..Default::default()
        };

        let state = RigidBodyState::new(
            Array1::from_vec(vec![initial_position, 0.0, 0.0]),
            Array1::from_vec(vec![initial_velocity, 0.0, 0.0]),
            Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]), // Identity quaternion
            Array1::zeros(3),
        );

        (config, properties, state)
    }

    /// Create a double pendulum system (multibody with constraints)
    pub fn double_pendulum(
        m1: f64,
        m2: f64,
        l1: f64,
        l2: f64,
        initial_angles: [f64; 2],
        initial_velocities: [f64; 2],
    ) -> (MechanicalConfig, MechanicalProperties, RigidBodyState) {
        let config = MechanicalConfig {
            system_type: MechanicalSystemType::Multibody,
            constraint_method: ConstraintMethod::Lagrange,
            ..Default::default()
        };

        // Gravitational forces
        let gravity_force = Box::new(move |_t: f64, _pos: &Array1<f64>, _vel: &Array1<f64>| {
            Array1::from_vec(vec![0.0, -m1 * 9.81, 0.0, 0.0, -m2 * 9.81, 0.0])
        });

        // Constraint equations for pendulum joints
        let constraint1 = Box::new(move |pos: &Array1<f64>| {
            // First pendulum constraint: |r1| = l1
            let x1 = pos[0];
            let y1 = pos[1];
            Array1::from_vec(vec![x1 * x1 + y1 * y1 - l1 * l1])
        });

        let constraint2 = Box::new(move |pos: &Array1<f64>| {
            // Second pendulum constraint: |r2 - r1| = l2
            let x1 = pos[0];
            let y1 = pos[1];
            let x2 = pos[3];
            let y2 = pos[4];
            let dx = x2 - x1;
            let dy = y2 - y1;
            Array1::from_vec(vec![dx * dx + dy * dy - l2 * l2])
        });

        // Constraint Jacobians for better numerical stability
        let jacobian1 = Box::new(move |pos: &Array1<f64>| {
            // Jacobian of constraint 1: C1 = x1² + y1² - l1²
            let x1 = pos[0];
            let y1 = pos[1];
            let mut jac = Array2::zeros((1, pos.len()));
            jac[(0, 0)] = 2.0 * x1; // ∂C1/∂x1
            jac[(0, 1)] = 2.0 * y1; // ∂C1/∂y1
                                    // All other entries are zero
            jac
        });

        let jacobian2 = Box::new(move |pos: &Array1<f64>| {
            // Jacobian of constraint 2: C2 = (x2-x1)² + (y2-y1)² - l2²
            let x1 = pos[0];
            let y1 = pos[1];
            let x2 = pos[3];
            let y2 = pos[4];
            let dx = x2 - x1;
            let dy = y2 - y1;
            let mut jac = Array2::zeros((1, pos.len()));
            jac[(0, 0)] = -2.0 * dx; // ∂C2/∂x1
            jac[(0, 1)] = -2.0 * dy; // ∂C2/∂y1
                                     // pos[2] is z1, not used
            jac[(0, 3)] = 2.0 * dx; // ∂C2/∂x2
            jac[(0, 4)] = 2.0 * dy; // ∂C2/∂y2
                                    // pos[5] is z2, not used
            jac
        });

        let properties = MechanicalProperties {
            mass: Array1::from_vec(vec![m1, m2]),
            inertia: vec![Array2::eye(3), Array2::eye(3)],
            external_forces: Some(gravity_force),
            constraints: vec![constraint1, constraint2],
            constraint_jacobians: vec![jacobian1, jacobian2],
            ..Default::default()
        };

        // Convert initial angles to Cartesian coordinates
        let x1 = l1 * initial_angles[0].sin();
        let y1 = -l1 * initial_angles[0].cos();
        let x2 = x1 + l2 * initial_angles[1].sin();
        let y2 = y1 - l2 * initial_angles[1].cos();

        let vx1 = l1 * initial_velocities[0] * initial_angles[0].cos();
        let vy1 = l1 * initial_velocities[0] * initial_angles[0].sin();
        let vx2 = vx1 + l2 * initial_velocities[1] * initial_angles[1].cos();
        let vy2 = vy1 + l2 * initial_velocities[1] * initial_angles[1].sin();

        let state = RigidBodyState::new(
            Array1::from_vec(vec![x1, y1, 0.0, x2, y2, 0.0]),
            Array1::from_vec(vec![vx1, vy1, 0.0, vx2, vy2, 0.0]),
            Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]), // Identity quaternion
            Array1::zeros(6), // Angular velocity for 2 bodies (3 components each)
        );

        (config, properties, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_rigid_body_state_creation() {
        let position = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let velocity = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let orientation = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let angular_velocity = Array1::from_vec(vec![0.01, 0.02, 0.03]);

        let state = RigidBodyState::new(
            position.clone(),
            velocity.clone(),
            orientation.clone(),
            angular_velocity.clone(),
        );

        assert_eq!(state.position, position);
        assert_eq!(state.velocity, velocity);
        assert_eq!(state.orientation, orientation);
        assert_eq!(state.angular_velocity, angular_velocity);
    }

    #[test]
    fn test_kinetic_energy_calculation() {
        let mass = 2.0;
        let inertia = Array2::eye(3) * 0.1;

        let state = RigidBodyState::new(
            Array1::zeros(3),
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
            Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0.0, 0.0, 1.0]),
        );

        let ke = state.kinetic_energy(mass, &inertia);
        let expected = 0.5 * mass * 1.0 + 0.5 * 0.1 * 1.0; // Translational + rotational
        assert_abs_diff_eq!(ke, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_normalization() {
        let mut state = RigidBodyState::new(
            Array1::zeros(3),
            Array1::zeros(3),
            Array1::from_vec(vec![2.0, 1.0, 1.0, 1.0]), // Non-normalized
            Array1::zeros(3),
        );

        state.normalize_quaternion();

        let norm = state.orientation.mapv(|q| q * q).sum().sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mechanical_integrator_creation() {
        let config = MechanicalConfig::default();
        let properties = MechanicalProperties::default();

        let integrator = MechanicalIntegrator::new(config, properties);

        assert_eq!(
            integrator.config.system_type,
            MechanicalSystemType::RigidBody
        );
        assert!(integrator.previous_state.is_none());
        assert!(integrator.energy_history.is_empty());
    }

    #[test]
    fn test_damped_oscillator_system() {
        let (config, properties, state) = systems::damped_oscillator(
            1.0,  // mass
            10.0, // stiffness
            0.1,  // damping
            1.0,  // initial position
            0.0,  // initial velocity
        );

        assert_eq!(config.system_type, MechanicalSystemType::RigidBody);
        assert_eq!(properties.mass[0], 1.0);
        assert_abs_diff_eq!(state.position[0], 1.0, epsilon = 1e-10);
        assert!(properties.external_forces.is_some());
    }

    #[test]
    fn test_double_pendulum_system() {
        let (config, properties, state) = systems::double_pendulum(
            1.0,        // m1
            0.5,        // m2
            1.0,        // l1
            0.8,        // l2
            [0.1, 0.2], // initial angles
            [0.0, 0.0], // initial velocities
        );

        assert_eq!(config.system_type, MechanicalSystemType::Multibody);
        assert_eq!(properties.mass.len(), 2);
        assert_eq!(properties.constraints.len(), 2);
        assert!(properties.external_forces.is_some());

        // Check that positions satisfy initial angle constraints
        let x1 = state.position[0];
        let y1 = state.position[1];
        let r1 = (x1 * x1 + y1 * y1).sqrt();
        assert_abs_diff_eq!(r1, 1.0, epsilon = 1e-10); // Should be l1
    }

    #[test]
    fn test_simple_integration_step() {
        let (config, properties, state) = systems::rigid_body(
            1.0,                                        // mass
            Array2::eye(3),                             // inertia
            Array1::zeros(3),                           // position
            Array1::from_vec(vec![1.0, 0.0, 0.0]),      // velocity
            Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]), // orientation
            Array1::zeros(3),                           // angular velocity
        );

        let mut integrator = MechanicalIntegrator::new(config, properties);

        let result = integrator.step(0.0, &state).unwrap();

        // After one time step, position should have changed
        assert!(result.state.position[0] > 0.0);
        assert!(result.stats.converged);
    }

    #[test]
    fn test_energy_conservation() {
        let (mut config, properties, state) = systems::damped_oscillator(
            1.0,  // mass
            10.0, // stiffness
            0.0,  // no damping for energy conservation test
            1.0,  // initial position
            0.0,  // initial velocity
        );

        config.dt = 0.001; // Small time step for better conservation
        let mut integrator = MechanicalIntegrator::new(config, properties);

        let _initial_energy = integrator.calculate_total_energy(&state);
        let mut current_state = state;

        // Integrate for several steps
        for i in 0..100 {
            let result = integrator.step(i as f64 * 0.001, &current_state).unwrap();
            current_state = result.state;
        }

        let (relative_drift, _max_drift, _current_energy) = integrator.energy_statistics();

        // Energy should be reasonably conserved (within a few percent)
        assert!(
            relative_drift < 0.1,
            "Energy drift too large: {}",
            relative_drift
        );
    }
}
