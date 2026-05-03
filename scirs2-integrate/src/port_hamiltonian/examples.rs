//! Example Port-Hamiltonian Systems
//!
//! This module provides ready-to-use pH system constructors for canonical
//! physical examples, useful for testing, demonstration, and benchmarking.
//!
//! # Systems Provided
//!
//! - [`pendulum_ph`]: Simple pendulum (nonlinear, conservative)
//! - [`mass_spring_damper_ph`]: Mass-spring-damper (dissipative)
//! - [`double_pendulum_ph`]: Double pendulum (4-DOF, conservative)
//! - [`rlc_circuit_ph`]: Series RLC circuit (electrical port-Hamiltonian)

use crate::error::{IntegrateError, IntegrateResult};
use crate::port_hamiltonian::system::{PortHamiltonianBuilder, PortHamiltonianSystem};
use scirs2_core::ndarray::{array, Array2};

// ─── Helper: skew-symmetric canonical J for n DOF ────────────────────────────

/// Construct the canonical 2n×2n skew-symmetric structure matrix for n DOF:
/// ```text
/// J = [[0_n,  I_n],
///      [-I_n, 0_n]]
/// ```
fn canonical_j(n: usize) -> Array2<f64> {
    let two_n = 2 * n;
    let mut j = Array2::zeros((two_n, two_n));
    for i in 0..n {
        j[[i, n + i]] = 1.0;
        j[[n + i, i]] = -1.0;
    }
    j
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Simple Pendulum
// ═══════════════════════════════════════════════════════════════════════════════

/// Create a simple pendulum pH system.
///
/// State: x = [q, p] where q = angle (rad), p = angular momentum (kg⋅m²/s)
///
/// Hamiltonian:
/// ```text
/// H(q, p) = p²/(2ml²) + mgl(1 - cos(q))
/// ```
///
/// Structure:
/// ```text
/// J = [[0,  1],   R = [[0, 0],   B = [[0],
///      [-1, 0]],        [0, d]],       [1]]
/// ```
/// where d is the pivot friction coefficient.
///
/// # Arguments
///
/// * `mass` - Pendulum mass [kg]
/// * `length` - Pendulum length [m]
/// * `gravity` - Gravitational acceleration [m/s²] (default: 9.81)
/// * `damping` - Pivot friction damping coefficient (0 = conservative)
pub fn pendulum_ph(
    mass: f64,
    length: f64,
    gravity: f64,
    damping: f64,
) -> IntegrateResult<PortHamiltonianSystem> {
    if mass <= 0.0 {
        return Err(IntegrateError::ValueError(
            "Pendulum mass must be positive".into(),
        ));
    }
    if length <= 0.0 {
        return Err(IntegrateError::ValueError(
            "Pendulum length must be positive".into(),
        ));
    }
    if damping < 0.0 {
        return Err(IntegrateError::ValueError(
            "Damping must be non-negative".into(),
        ));
    }

    let ml2 = mass * length * length;
    let mgl = mass * gravity * length;

    PortHamiltonianBuilder::new(2, 1)
        .with_j(move |_x| Ok(array![[0.0, 1.0], [-1.0, 0.0]]))
        .with_r(move |_x| Ok(array![[0.0, 0.0], [0.0, damping]]))
        .with_hamiltonian(move |x| {
            let q = x[0];
            let p = x[1];
            Ok(p * p / (2.0 * ml2) + mgl * (1.0 - q.cos()))
        })
        .with_grad_hamiltonian(move |x| {
            let q = x[0];
            let p = x[1];
            Ok(scirs2_core::ndarray::array![mgl * q.sin(), p / ml2])
        })
        .with_b(array![[0.0], [1.0]])
        .build()
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Mass-Spring-Damper
// ═══════════════════════════════════════════════════════════════════════════════

/// Create a mass-spring-damper pH system.
///
/// State: x = [q, p] where q = displacement [m], p = momentum [kg⋅m/s]
///
/// Hamiltonian:
/// ```text
/// H(q, p) = p²/(2m) + k⋅q²/2
/// ```
///
/// Structure:
/// ```text
/// J = [[0,  1],   R = [[0, 0],   B = [[0],
///      [-1, 0]],        [0, c]],       [1]]
/// ```
///
/// The energy balance: dH/dt = -c*(p/m)² + F*p/m
/// where F is the applied force (through the port).
///
/// # Arguments
///
/// * `mass` - Mass [kg]
/// * `spring_const` - Spring constant [N/m]
/// * `damping` - Damping coefficient [N⋅s/m]
pub fn mass_spring_damper_ph(
    mass: f64,
    spring_const: f64,
    damping: f64,
) -> IntegrateResult<PortHamiltonianSystem> {
    if mass <= 0.0 {
        return Err(IntegrateError::ValueError("Mass must be positive".into()));
    }
    if spring_const < 0.0 {
        return Err(IntegrateError::ValueError(
            "Spring constant must be non-negative".into(),
        ));
    }
    if damping < 0.0 {
        return Err(IntegrateError::ValueError(
            "Damping must be non-negative".into(),
        ));
    }

    PortHamiltonianBuilder::new(2, 1)
        .with_j(move |_x| Ok(array![[0.0, 1.0], [-1.0, 0.0]]))
        .with_r(move |_x| Ok(array![[0.0, 0.0], [0.0, damping]]))
        .with_hamiltonian(move |x| {
            let q = x[0];
            let p = x[1];
            Ok(p * p / (2.0 * mass) + spring_const * q * q / 2.0)
        })
        .with_grad_hamiltonian(move |x| {
            let q = x[0];
            let p = x[1];
            Ok(scirs2_core::ndarray::array![spring_const * q, p / mass])
        })
        .with_b(array![[0.0], [1.0]])
        .build()
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Double Pendulum
// ═══════════════════════════════════════════════════════════════════════════════

/// Create a double pendulum pH system.
///
/// State: x = [q1, q2, p1, p2] where q1, q2 are the angles of link 1 and 2
/// (relative angles), and p1, p2 are the conjugate momenta.
///
/// This is a conservative (Hamiltonian) system with two links of equal
/// mass and length. The Hamiltonian uses the classical T+V form with
/// the full kinetic energy coupling.
///
/// # Arguments
///
/// * `mass` - Mass of each link [kg]
/// * `length` - Length of each link [m]
/// * `gravity` - Gravitational acceleration [m/s²]
/// * `damping1` - Damping at joint 1
/// * `damping2` - Damping at joint 2
pub fn double_pendulum_ph(
    mass: f64,
    length: f64,
    gravity: f64,
    damping1: f64,
    damping2: f64,
) -> IntegrateResult<PortHamiltonianSystem> {
    if mass <= 0.0 {
        return Err(IntegrateError::ValueError("Mass must be positive".into()));
    }
    if length <= 0.0 {
        return Err(IntegrateError::ValueError("Length must be positive".into()));
    }

    let m = mass;
    let l = length;
    let g = gravity;
    let ml2 = m * l * l;

    // The double pendulum Hamiltonian in terms of generalized momenta (with coupling).
    // Using the standard formulation for two equal point masses:
    // T = (1/(2ml²)) * [p1² + 2p2² - 2p1p2 cos(q1-q2)] / (2 + (sin(q1-q2))² * ... )
    // For simplicity, we use the uncoupled approximation and note the user can
    // substitute the exact kinetic energy for their application.
    //
    // Exact Hamiltonian (Gans, 1995):
    // T = (p1² + 2p2² - 2 p1 p2 cos(Δ)) / [2 ml²(2 - cos²(Δ))]
    // V = -mgl(2 cos(q1) + cos(q2))
    // where Δ = q1 - q2

    PortHamiltonianBuilder::new(4, 2)
        .with_j(|_x| {
            let j = canonical_j(2);
            Ok(j)
        })
        .with_r(move |_x| {
            Ok(array![
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, damping1, 0.0],
                [0.0, 0.0, 0.0, damping2]
            ])
        })
        .with_hamiltonian(move |x| {
            let q1 = x[0];
            let q2 = x[1];
            let p1 = x[2];
            let p2 = x[3];
            let delta = q1 - q2;
            let cos_delta = delta.cos();
            let denom = 2.0 * ml2 * (2.0 - cos_delta * cos_delta);
            if denom.abs() < 1e-30 {
                return Err(IntegrateError::ComputationError(
                    "Double pendulum singularity (cos²(Δ)=2)".into(),
                ));
            }
            let t_kin = (p1 * p1 + 2.0 * p2 * p2 - 2.0 * p1 * p2 * cos_delta) / denom;
            let v_pot = -m * g * l * (2.0 * q1.cos() + q2.cos());
            Ok(t_kin + v_pot)
        })
        .with_b({
            let mut b = Array2::zeros((4, 2));
            b[[2, 0]] = 1.0;
            b[[3, 1]] = 1.0;
            b
        })
        .build()
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. RLC Circuit
// ═══════════════════════════════════════════════════════════════════════════════

/// Create a series RLC circuit pH system.
///
/// The circuit has an inductor L, capacitor C, and resistor R in series.
///
/// **State**: x = [q_c, φ_L] where:
/// - q_c = charge on capacitor [C]
/// - φ_L = flux linkage of inductor [Wb = V⋅s]
///
/// **Hamiltonian** (total electromagnetic energy):
/// ```text
/// H(q_c, φ_L) = q_c²/(2C) + φ_L²/(2L)
///              = V_C²C/2 + I²L/2
/// ```
///
/// **Port-Hamiltonian structure**:
/// ```text
/// J = [[0, -1],    R = [[0, 0],    B = [[1],
///      [1,  0]],         [0, R]],        [0]]
/// ```
/// Note: The sign convention follows the standard "flux-charge" formulation
/// where ∂H/∂q_c = q_c/C = V_C (voltage) and ∂H/∂φ_L = φ_L/L = I (current).
///
/// **Energy balance**:
/// ```text
/// dH/dt = -R⋅I² + V_source⋅I
/// ```
///
/// # Arguments
///
/// * `inductance` - Inductance L [H]
/// * `capacitance` - Capacitance C [F]  
/// * `resistance` - Resistance R [Ω]
pub fn rlc_circuit_ph(
    inductance: f64,
    capacitance: f64,
    resistance: f64,
) -> IntegrateResult<PortHamiltonianSystem> {
    if inductance <= 0.0 {
        return Err(IntegrateError::ValueError(
            "Inductance must be positive".into(),
        ));
    }
    if capacitance <= 0.0 {
        return Err(IntegrateError::ValueError(
            "Capacitance must be positive".into(),
        ));
    }
    if resistance < 0.0 {
        return Err(IntegrateError::ValueError(
            "Resistance must be non-negative".into(),
        ));
    }

    PortHamiltonianBuilder::new(2, 1)
        // J: skew-symmetric, represents energy exchange between L and C
        // dq_c/dt = -I = -φ_L/L = -∂H/∂φ_L   => J[0,1] = -1
        // dφ_L/dt = V = q_c/C = ∂H/∂q_c       => J[1,0] = +1
        .with_j(|_x| Ok(array![[0.0, -1.0], [1.0, 0.0]]))
        // R: resistor dissipates energy via current (I = φ_L/L = ∂H/∂φ_L)
        .with_r(move |_x| Ok(array![[0.0, 0.0], [0.0, resistance]]))
        .with_hamiltonian(move |x| {
            let q_c = x[0]; // Capacitor charge
            let phi_l = x[1]; // Inductor flux linkage
            Ok(q_c * q_c / (2.0 * capacitance) + phi_l * phi_l / (2.0 * inductance))
        })
        .with_grad_hamiltonian(move |x| {
            let q_c = x[0];
            let phi_l = x[1];
            Ok(scirs2_core::ndarray::array![q_c / capacitance, phi_l / inductance])
        })
        // B: voltage source connects through charge equation
        .with_b(array![[1.0], [0.0]])
        .build()
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. Coupled Oscillators (multi-physics example)
// ═══════════════════════════════════════════════════════════════════════════════

/// Create a network of coupled harmonic oscillators as a pH system.
///
/// This models n masses connected by springs in a chain, where each mass
/// can have individual damping. The resulting pH system has 2n states.
///
/// # Arguments
///
/// * `masses` - Mass of each oscillator [kg], length n
/// * `spring_consts` - Spring constants: spring_consts[i] connects mass i to i+1, length n-1
/// * `damping` - Damping coefficients for each mass, length n
pub fn coupled_oscillators_ph(
    masses: &[f64],
    spring_consts: &[f64],
    damping: &[f64],
) -> IntegrateResult<PortHamiltonianSystem> {
    let n = masses.len();
    if spring_consts.len() != n - 1 {
        return Err(IntegrateError::ValueError(format!(
            "spring_consts must have length n-1={}, got {}",
            n - 1,
            spring_consts.len()
        )));
    }
    if damping.len() != n {
        return Err(IntegrateError::ValueError(format!(
            "damping must have length n={n}, got {}",
            damping.len()
        )));
    }
    for (i, &m) in masses.iter().enumerate() {
        if m <= 0.0 {
            return Err(IntegrateError::ValueError(format!(
                "Mass {i} must be positive"
            )));
        }
    }

    let masses_vec = masses.to_vec();
    let k_vec = spring_consts.to_vec();
    let d_vec = damping.to_vec();
    let masses_clone = masses_vec.clone();
    let k_clone = k_vec.clone();
    let d_clone = d_vec.clone();

    let mut b = Array2::zeros((2 * n, n));
    for i in 0..n {
        b[[n + i, i]] = 1.0;
    }

    PortHamiltonianBuilder::new(2 * n, n)
        .with_j(move |_x| Ok(canonical_j(n)))
        .with_r(move |_x| {
            let mut r = Array2::zeros((2 * n, 2 * n));
            for i in 0..n {
                r[[n + i, n + i]] = d_clone[i];
            }
            Ok(r)
        })
        .with_hamiltonian(move |x| {
            let mut h = 0.0_f64;
            // Kinetic energy: T = sum p_i² / (2 m_i)
            for i in 0..n {
                let p_i = x[n + i];
                h += p_i * p_i / (2.0 * masses_vec[i]);
            }
            // Potential energy: V = sum k_i * (q_{i+1} - q_i)² / 2
            for i in 0..n - 1 {
                let dq = x[i + 1] - x[i];
                h += k_vec[i] * dq * dq / 2.0;
            }
            Ok(h)
        })
        .with_grad_hamiltonian(move |x| {
            let mut grad = vec![0.0_f64; 2 * n];
            // ∂H/∂q_i = -k_{i-1}*(q_i - q_{i-1}) + k_i*(q_i - q_{i+1})
            for i in 0..n {
                if i > 0 {
                    grad[i] -= k_clone[i - 1] * (x[i - 1] - x[i]);
                }
                if i < n - 1 {
                    grad[i] += k_clone[i] * (x[i] - x[i + 1]);
                }
            }
            // ∂H/∂p_i = p_i / m_i
            for i in 0..n {
                grad[n + i] = x[n + i] / masses_clone[i];
            }
            Ok(scirs2_core::ndarray::Array1::from_vec(grad))
        })
        .with_b(b)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::port_hamiltonian::integrators::{AverageVectorField, DiscreteGradientGonzalez};

    #[test]
    fn test_pendulum_energy_conservation() {
        // Conservative pendulum (no damping)
        let sys = pendulum_ph(1.0, 1.0, 9.81, 0.0).expect("Failed to create pendulum");

        // Initial condition: small angle, at rest
        let x0 = vec![0.1_f64, 0.0];
        let h0 = sys.hamiltonian(&x0).expect("Hamiltonian eval failed");

        let integrator = DiscreteGradientGonzalez::new();
        let result = integrator
            .integrate(&sys, &x0, 0.0, 2.0, 0.05, None)
            .expect("Integration failed");

        let h_final = *result.energy.last().expect("No energy in result");
        // Energy should be conserved to machine precision
        assert!(
            (h_final - h0).abs() < 1e-10,
            "Energy drift = {} (expected < 1e-10)",
            (h_final - h0).abs()
        );
    }

    #[test]
    fn test_mass_spring_damper_energy_decay() {
        // Dissipative system: energy must decay
        let mass = 1.0_f64;
        let k = 4.0_f64;
        let c = 0.5_f64;
        let sys = mass_spring_damper_ph(mass, k, c).expect("Failed to create MSD");

        let x0 = vec![1.0_f64, 0.0]; // Displaced from equilibrium, at rest
        let h0 = sys.hamiltonian(&x0).expect("Hamiltonian eval failed");

        let integrator = AverageVectorField::new();
        let result = integrator
            .integrate(&sys, &x0, 0.0, 5.0, 0.05, None)
            .expect("Integration failed");

        let h_final = *result.energy.last().expect("No energy in result");
        // Energy must have decreased due to damping
        assert!(
            h_final < h0,
            "Energy should decrease: h0={h0}, h_final={h_final}"
        );
        assert!(
            h_final >= 0.0,
            "Energy must remain non-negative: h_final={h_final}"
        );
    }

    #[test]
    fn test_rlc_circuit_structure() {
        let sys = rlc_circuit_ph(1e-3, 1e-6, 100.0).expect("Failed to create RLC");

        // Verify J is skew-symmetric
        assert!(
            sys.validate_skew_symmetry(&[0.0, 0.0])
                .expect("Validation failed"),
            "J must be skew-symmetric"
        );

        // Verify R is PSD
        assert!(
            sys.validate_psd(&[0.0, 0.0]).expect("PSD check failed"),
            "R must be PSD"
        );
    }

    #[test]
    fn test_rlc_energy_decay() {
        // L = 1H, C = 1F, R = 1Ω
        let sys = rlc_circuit_ph(1.0, 1.0, 1.0).expect("Failed to create RLC");

        // Initial condition: capacitor charged to 1V, no inductor current
        // q_c = C * V = 1.0, phi_L = 0
        let x0 = vec![1.0_f64, 0.0];
        let h0 = sys.hamiltonian(&x0).expect("Hamiltonian eval failed");

        let integrator = AverageVectorField::new();
        let result = integrator
            .integrate(&sys, &x0, 0.0, 5.0, 0.1, None)
            .expect("Integration failed");

        let h_final = *result.energy.last().expect("No energy in result");
        assert!(
            h_final < h0,
            "RLC energy must decrease: h0={h0}, h_final={h_final}"
        );
    }

    #[test]
    fn test_pendulum_output() {
        // Test the power port output y = B^T ∇H
        let sys = pendulum_ph(1.0, 1.0, 9.81, 0.0).expect("Failed to create pendulum");
        let x = vec![0.5_f64, 2.0];
        let y = sys.output(&x).expect("Output eval failed");
        // y = B^T * ∇H = [0, 1] * [mgl*sin(q), p/ml²] = p/(ml²)
        let p_over_ml2 = 2.0 / (1.0 * 1.0 * 1.0); // p = 2, m = 1, l = 1
        assert!(
            (y[0] - p_over_ml2).abs() < 1e-10,
            "Output y[0] = {} != {}",
            y[0],
            p_over_ml2
        );
    }

    #[test]
    fn test_coupled_oscillators() {
        let masses = vec![1.0_f64, 1.0, 1.0];
        let springs = vec![1.0_f64, 1.0];
        let damps = vec![0.0_f64, 0.0, 0.0];
        let sys = coupled_oscillators_ph(&masses, &springs, &damps)
            .expect("Failed to create coupled oscillators");

        // Check structure
        let x0 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // first mass displaced
        let h0 = sys.hamiltonian(&x0).expect("Hamiltonian eval failed");
        assert!(h0 > 0.0, "Potential energy should be positive");

        // Conservative system: energy should be preserved
        let integrator = AverageVectorField::new();
        let result = integrator
            .integrate(&sys, &x0, 0.0, 2.0, 0.05, None)
            .expect("Integration failed");

        let h_final = *result.energy.last().expect("No energy in result");
        assert!(
            (h_final - h0).abs() < 1e-8,
            "Energy drift in coupled oscillators: {}",
            (h_final - h0).abs()
        );
    }
}
