//! Port-Hamiltonian Systems and Structure-Preserving Integration
//!
//! This module implements the port-Hamiltonian (pH) framework for modeling and
//! simulating physical systems that obey energy balance laws, combined with
//! structure-preserving numerical integrators that maintain these laws at the
//! discrete level.
//!
//! # Port-Hamiltonian Systems
//!
//! A port-Hamiltonian system is described by:
//!
//! ```text
//! dx/dt = (J(x) - R(x)) ∇H(x) + B(x) u
//!     y = B(x)^T ∇H(x)
//! ```
//!
//! where:
//! - `x ∈ ℝⁿ` is the state (energy variables)
//! - `H: ℝⁿ → ℝ` is the Hamiltonian (stored energy)
//! - `J(x) = -J(x)^T` is the skew-symmetric interconnection matrix
//! - `R(x) = R(x)^T ≥ 0` is the positive semi-definite dissipation matrix
//! - `B(x)` is the input/output (port) matrix
//! - `u` is the port input (external forces, voltages, etc.)
//! - `y` is the power-conjugate output
//!
//! ## Energy Balance
//!
//! The fundamental property of pH systems is the **power balance**:
//! ```text
//! dH/dt = -∇H^T R ∇H + y^T u
//!       ≤ y^T u    (passivity)
//! ```
//! The system is passive: it can only dissipate or store, not generate, energy.
//!
//! # Structure-Preserving Integrators
//!
//! To maintain the energy balance at the discrete level, we provide:
//!
//! | Method | Energy preservation | Order | Implicit |
//! |--------|--------------------|---------| --------|
//! | [`DiscreteGradientGonzalez`] | Exact (conservative) | 2 | Yes |
//! | [`DiscreteGradientItohAbe`] | Exact (conservative) | 2 | Yes |
//! | [`AverageVectorField`] | Exact (conservative) | 2 | Yes |
//! | [`ImplicitMidpoint`] | Quadratic H exactly | 2 | Yes |
//! | [`StormerVerletPH`] | Symplectic | 2 | Partially |
//! | [`Rattle`] | Symplectic + constraints | 2 | Yes |
//!
//! # Example: Simple Pendulum
//!
//! ```rust
//! use scirs2_integrate::port_hamiltonian::{
//!     PortHamiltonianBuilder, DiscreteGradientGonzalez,
//! };
//! use scirs2_core::ndarray::array;
//!
//! // Build a conservative pendulum (m=1, l=1, g=9.81, no damping)
//! let system = PortHamiltonianBuilder::new(2, 1)
//!     .with_j(|_x| Ok(array![[0.0, 1.0], [-1.0, 0.0]]))
//!     .with_r(|_x| Ok(array![[0.0, 0.0], [0.0, 0.0]]))
//!     .with_hamiltonian(|x| {
//!         let q = x[0]; let p = x[1];
//!         Ok(p * p / 2.0 + 9.81 * (1.0 - q.cos()))
//!     })
//!     .with_grad_hamiltonian(|x| {
//!         Ok(array![9.81 * x[0].sin(), x[1]])
//!     })
//!     .with_b(array![[0.0], [1.0]])
//!     .build()
//!     .expect("Failed to build system");
//!
//! let integrator = DiscreteGradientGonzalez::new();
//! let result = integrator
//!     .integrate(&system, &[0.5, 0.0], 0.0, 10.0, 0.05, None)
//!     .expect("Integration failed");
//!
//! // Check energy conservation
//! let h0 = result.energy[0];
//! let h_final = result.energy.last().copied().unwrap_or(h0);
//! assert!((h_final - h0).abs() < 1e-10, "Energy drift: {}", (h_final - h0).abs());
//! ```
//!
//! # Example: RLC Circuit
//!
//! ```rust
//! use scirs2_integrate::port_hamiltonian::{rlc_circuit_ph, AverageVectorField};
//!
//! // Series RLC: L=1mH, C=1μF, R=100Ω
//! let circuit = rlc_circuit_ph(1e-3, 1e-6, 100.0).expect("Failed to create RLC");
//!
//! // Initial state: capacitor charged to 1V (q_c = C * V = 1μC), no current
//! let x0 = vec![1e-6, 0.0];
//!
//! let avf = AverageVectorField::new();
//! let result = avf
//!     .integrate(&circuit, &x0, 0.0, 1e-4, 1e-7, None)
//!     .expect("Integration failed");
//!
//! // Energy must decay (resistor dissipates)
//! let h_final = result.energy.last().copied().unwrap_or(0.0);
//! assert!(h_final < result.energy[0]);
//! ```

pub mod dissipation;
pub mod examples;
pub mod integrators;
pub mod system;

// ─── Re-exports from system ───────────────────────────────────────────────────
pub use system::{PortHamiltonianBuilder, PortHamiltonianConfig, PortHamiltonianSystem};

// ─── Re-exports from integrators ─────────────────────────────────────────────
pub use integrators::{
    AverageVectorField, DiscreteGradientGonzalez, DiscreteGradientItohAbe, ImplicitMidpoint,
    IntegratorOptions, PortHamiltonianResult, Rattle, StepResult, StormerVerletPH,
};

// ─── Re-exports from dissipation ─────────────────────────────────────────────
pub use dissipation::{
    LinearDissipation, NonlinearDissipation, PortDissipation, RayleighDissipation,
    StructuredDissipation,
};

// ─── Re-exports from examples ────────────────────────────────────────────────
pub use examples::{
    coupled_oscillators_ph, double_pendulum_ph, mass_spring_damper_ph, pendulum_ph, rlc_circuit_ph,
};
