//! 1D compressible Euler equations: state, flux, and conversion utilities.
//!
//! Conservative variables: U = (ρ, ρu, E)
//! where ρ is density, u is velocity, E = ρe + ½ρu² is total energy.
//! Flux: F(U) = (ρu, ρu² + p, u(E + p))

use std::ops::{Add, Mul, Sub};

/// Conservative state vector for the 1D Euler equations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EulerState {
    /// Density ρ
    pub rho: f64,
    /// Momentum ρu
    pub rho_u: f64,
    /// Total energy E = ρe + ½ρu²
    pub energy: f64,
}

impl EulerState {
    /// Zero state (vacuum).
    pub fn zero() -> Self {
        Self {
            rho: 0.0,
            rho_u: 0.0,
            energy: 0.0,
        }
    }
}

impl Add for EulerState {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            rho: self.rho + rhs.rho,
            rho_u: self.rho_u + rhs.rho_u,
            energy: self.energy + rhs.energy,
        }
    }
}

impl Sub for EulerState {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            rho: self.rho - rhs.rho,
            rho_u: self.rho_u - rhs.rho_u,
            energy: self.energy - rhs.energy,
        }
    }
}

impl Mul<f64> for EulerState {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        Self {
            rho: self.rho * scalar,
            rho_u: self.rho_u * scalar,
            energy: self.energy * scalar,
        }
    }
}

/// Physical flux F(U) = (ρu, ρu² + p, u(E + p)).
#[derive(Clone, Copy, Debug)]
pub struct EulerFlux {
    /// Mass flux ρu
    pub mass: f64,
    /// Momentum flux ρu² + p
    pub momentum: f64,
    /// Energy flux u(E + p)
    pub energy: f64,
}

impl EulerFlux {
    /// Zero flux.
    pub fn zero() -> Self {
        Self {
            mass: 0.0,
            momentum: 0.0,
            energy: 0.0,
        }
    }
}

/// Pressure from conservative variables via ideal-gas EOS: p = (γ-1)(E - ½ρu²/ρ).
///
/// Returns `None` if density is non-positive (unphysical state) or if the
/// computed pressure is negative.
pub fn pressure_eos(state: &EulerState, gamma: f64) -> Option<f64> {
    if state.rho <= 0.0 {
        return None;
    }
    let u = state.rho_u / state.rho;
    let p = (gamma - 1.0) * (state.energy - 0.5 * state.rho * u * u);
    if p < 0.0 {
        None
    } else {
        Some(p)
    }
}

/// Sound speed c = sqrt(γp/ρ). Returns `None` for unphysical state.
pub fn sound_speed(state: &EulerState, gamma: f64) -> Option<f64> {
    let p = pressure_eos(state, gamma)?;
    Some((gamma * p / state.rho).sqrt())
}

/// Maximum wave speed estimate |u| + c. Returns `None` for unphysical state.
pub fn max_wave_speed(state: &EulerState, gamma: f64) -> Option<f64> {
    let u = state.rho_u / state.rho;
    let c = sound_speed(state, gamma)?;
    Some(u.abs() + c)
}

/// Physical flux F(U) = (ρu, ρu² + p, u(E + p)).
///
/// Returns `None` if the state is unphysical (non-positive density or pressure).
pub fn euler_flux(state: &EulerState, gamma: f64) -> Option<EulerFlux> {
    let p = pressure_eos(state, gamma)?;
    let u = state.rho_u / state.rho;
    Some(EulerFlux {
        mass: state.rho_u,
        momentum: state.rho_u * u + p,
        energy: u * (state.energy + p),
    })
}

/// Convert primitive variables (ρ, u, p) to conservative (ρ, ρu, E).
pub fn primitives_to_conservative(rho: f64, u: f64, p: f64, gamma: f64) -> EulerState {
    EulerState {
        rho,
        rho_u: rho * u,
        energy: p / (gamma - 1.0) + 0.5 * rho * u * u,
    }
}

/// Convert conservative variables (ρ, ρu, E) to primitive (ρ, u, p).
///
/// Returns `None` if state is unphysical.
pub fn conservative_to_primitives(state: &EulerState, gamma: f64) -> Option<(f64, f64, f64)> {
    if state.rho <= 0.0 {
        return None;
    }
    let u = state.rho_u / state.rho;
    let p = pressure_eos(state, gamma)?;
    Some((state.rho, u, p))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GAMMA: f64 = 1.4;

    #[test]
    fn test_primitives_conservative_round_trip() {
        let (rho, u, p) = (1.0, 2.5, 1.0);
        let cons = primitives_to_conservative(rho, u, p, GAMMA);
        let (rho2, u2, p2) = conservative_to_primitives(&cons, GAMMA).expect("physical state");
        assert!((rho2 - rho).abs() < 1e-14);
        assert!((u2 - u).abs() < 1e-14);
        assert!((p2 - p).abs() < 1e-14);
    }

    #[test]
    fn test_pressure_positivity() {
        // At rest: E = p/(γ-1)
        let state = primitives_to_conservative(1.0, 0.0, 1.0, GAMMA);
        let p = pressure_eos(&state, GAMMA).expect("positive pressure");
        assert!((p - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_flux_at_rest() {
        // u=0: flux = (0, p, 0)
        let state = primitives_to_conservative(1.0, 0.0, 1.0, GAMMA);
        let f = euler_flux(&state, GAMMA).expect("valid flux");
        assert!(f.mass.abs() < 1e-14);
        assert!((f.momentum - 1.0).abs() < 1e-14);
        assert!(f.energy.abs() < 1e-14);
    }

    #[test]
    fn test_state_arithmetic() {
        let a = EulerState {
            rho: 1.0,
            rho_u: 2.0,
            energy: 3.0,
        };
        let b = EulerState {
            rho: 0.5,
            rho_u: 1.0,
            energy: 1.5,
        };
        let sum = a + b;
        assert_eq!(sum.rho, 1.5);
        let scaled = a * 2.0;
        assert_eq!(scaled.rho, 2.0);
    }

    #[test]
    fn test_invalid_state_returns_none() {
        let vacuum = EulerState {
            rho: -1.0,
            rho_u: 0.0,
            energy: 0.0,
        };
        assert!(pressure_eos(&vacuum, GAMMA).is_none());
        assert!(sound_speed(&vacuum, GAMMA).is_none());
    }

    #[test]
    fn test_sound_speed_at_rest() {
        // c = sqrt(γp/ρ) = sqrt(1.4 * 1.0 / 1.0) = sqrt(1.4)
        let state = primitives_to_conservative(1.0, 0.0, 1.0, GAMMA);
        let c = sound_speed(&state, GAMMA).expect("valid sound speed");
        assert!((c - GAMMA.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_max_wave_speed() {
        let state = primitives_to_conservative(1.0, 2.0, 1.0, GAMMA);
        let s = max_wave_speed(&state, GAMMA).expect("valid wave speed");
        let c = sound_speed(&state, GAMMA).expect("sound speed");
        assert!((s - (2.0 + c)).abs() < 1e-12);
    }

    #[test]
    fn test_zero_state() {
        let z = EulerState::zero();
        assert_eq!(z.rho, 0.0);
        assert_eq!(z.rho_u, 0.0);
        assert_eq!(z.energy, 0.0);
    }

    #[test]
    fn test_zero_flux() {
        let z = EulerFlux::zero();
        assert_eq!(z.mass, 0.0);
        assert_eq!(z.momentum, 0.0);
        assert_eq!(z.energy, 0.0);
    }
}
