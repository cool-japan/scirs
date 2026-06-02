//! HLLC Riemann solver for the 1D compressible Euler equations.
//!
//! Reference: Toro (2009), "Riemann Solvers and Numerical Methods for Fluid Dynamics",
//! Chapter 10. Wave speed estimates from Davis-Einfeldt.

use super::euler_1d::{euler_flux, pressure_eos, sound_speed, EulerFlux, EulerState};

/// Compute the HLLC star state for wave-side `k`.
///
/// Implements Toro (2009) Eq. 10.39:
///   U*_k = ρ_k * (S_k - u_k)/(S_k - S*) * [1, S*, E_k/ρ_k + (S* - u_k)(S* + p_k/(ρ_k(S_k - u_k)))]
fn hllc_star_state(
    u_k: f64,
    rho_k: f64,
    p_k: f64,
    s_k: f64,
    s_star: f64,
    state_k: &EulerState,
) -> EulerState {
    let factor = rho_k * (s_k - u_k) / (s_k - s_star);
    EulerState {
        rho: factor,
        rho_u: factor * s_star,
        energy: factor
            * (state_k.energy / rho_k + (s_star - u_k) * (s_star + p_k / (rho_k * (s_k - u_k)))),
    }
}

/// HLLC numerical flux at the interface between `left` and `right` states.
///
/// Uses Davis-Einfeldt wave speed estimates and Toro's HLLC contact-wave formulation.
/// Returns `None` if either state is unphysical (non-positive density or pressure).
pub fn hllc_flux(left: &EulerState, right: &EulerState, gamma: f64) -> Option<EulerFlux> {
    // Primitive variables
    let rho_l = left.rho;
    let u_l = left.rho_u / rho_l;
    let p_l = pressure_eos(left, gamma)?;
    let c_l = sound_speed(left, gamma)?;

    let rho_r = right.rho;
    let u_r = right.rho_u / rho_r;
    let p_r = pressure_eos(right, gamma)?;
    let c_r = sound_speed(right, gamma)?;

    // Davis-Einfeldt wave speed estimates
    let s_l = f64::min(u_l - c_l, u_r - c_r);
    let s_r = f64::max(u_l + c_l, u_r + c_r);

    // Pure upwind cases
    if s_l >= 0.0 {
        return euler_flux(left, gamma);
    }
    if s_r <= 0.0 {
        return euler_flux(right, gamma);
    }

    // Contact wave speed (Toro Eq. 10.37)
    let s_star = (p_r - p_l + rho_l * u_l * (s_l - u_l) - rho_r * u_r * (s_r - u_r))
        / (rho_l * (s_l - u_l) - rho_r * (s_r - u_r));

    if s_star >= 0.0 {
        // F_HLLC = F_L + S_L * (U*_L - U_L)
        let f_l = euler_flux(left, gamma)?;
        let u_star_l = hllc_star_state(u_l, rho_l, p_l, s_l, s_star, left);
        let diff = u_star_l - *left;
        Some(EulerFlux {
            mass: f_l.mass + s_l * diff.rho,
            momentum: f_l.momentum + s_l * diff.rho_u,
            energy: f_l.energy + s_l * diff.energy,
        })
    } else {
        // F_HLLC = F_R + S_R * (U*_R - U_R)
        let f_r = euler_flux(right, gamma)?;
        let u_star_r = hllc_star_state(u_r, rho_r, p_r, s_r, s_star, right);
        let diff = u_star_r - *right;
        Some(EulerFlux {
            mass: f_r.mass + s_r * diff.rho,
            momentum: f_r.momentum + s_r * diff.rho_u,
            energy: f_r.energy + s_r * diff.energy,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pde::dg_systems::euler_1d::{euler_flux, primitives_to_conservative};

    const GAMMA: f64 = 1.4;

    #[test]
    fn test_hllc_pure_left_upwind() {
        // Both wave speeds positive → pure upwind → F = F_L
        let left = primitives_to_conservative(1.0, 5.0, 1.0, GAMMA);
        let right = primitives_to_conservative(0.5, 4.0, 0.5, GAMMA);
        let f = hllc_flux(&left, &right, GAMMA).expect("valid flux");
        let f_l = euler_flux(&left, GAMMA).expect("valid left flux");
        assert!((f.mass - f_l.mass).abs() < 1e-10);
        assert!((f.momentum - f_l.momentum).abs() < 1e-10);
        assert!((f.energy - f_l.energy).abs() < 1e-10);
    }

    #[test]
    fn test_hllc_sod_initial_left_right_gives_finite_flux() {
        // Sod left (ρ=1, u=0, p=1) and right (ρ=0.125, u=0, p=0.1)
        let left = primitives_to_conservative(1.0, 0.0, 1.0, GAMMA);
        let right = primitives_to_conservative(0.125, 0.0, 0.1, GAMMA);
        let f = hllc_flux(&left, &right, GAMMA).expect("valid flux");
        // Flux should be finite and mass flux should be positive (wave moves right)
        assert!(f.mass.is_finite());
        assert!(f.momentum.is_finite());
        assert!(f.energy.is_finite());
        assert!(f.mass >= 0.0); // net mass flow non-negative from left high pressure
    }

    #[test]
    fn test_hllc_symmetric_at_rest() {
        // Identical states at rest → flux should equal physical flux
        let state = primitives_to_conservative(1.0, 0.0, 1.0, GAMMA);
        let f = hllc_flux(&state, &state, GAMMA).expect("valid flux");
        let f_phys = euler_flux(&state, GAMMA).expect("valid phys flux");
        assert!((f.mass - f_phys.mass).abs() < 1e-12);
        assert!((f.momentum - f_phys.momentum).abs() < 1e-12);
        assert!((f.energy - f_phys.energy).abs() < 1e-12);
    }

    #[test]
    fn test_hllc_unphysical_state_returns_none() {
        let bad = EulerState {
            rho: -1.0,
            rho_u: 0.0,
            energy: 0.0,
        };
        let good = primitives_to_conservative(1.0, 0.0, 1.0, GAMMA);
        assert!(hllc_flux(&bad, &good, GAMMA).is_none());
    }

    #[test]
    fn test_hllc_pure_right_upwind() {
        // Both wave speeds negative → pure upwind → F = F_R
        let left = primitives_to_conservative(0.5, -4.0, 0.5, GAMMA);
        let right = primitives_to_conservative(1.0, -5.0, 1.0, GAMMA);
        let f = hllc_flux(&left, &right, GAMMA).expect("valid flux");
        let f_r = euler_flux(&right, GAMMA).expect("valid right flux");
        assert!((f.mass - f_r.mass).abs() < 1e-10);
        assert!((f.momentum - f_r.momentum).abs() < 1e-10);
        assert!((f.energy - f_r.energy).abs() < 1e-10);
    }

    #[test]
    fn test_hllc_conservation_property() {
        // The HLLC flux should satisfy: for identical states the flux matches physical flux
        // Here we test a supersonic rightward flow
        let state = primitives_to_conservative(1.0, 3.0, 1.0, GAMMA);
        let f = hllc_flux(&state, &state, GAMMA).expect("valid flux");
        let f_phys = euler_flux(&state, GAMMA).expect("valid phys flux");
        assert!((f.mass - f_phys.mass).abs() < 1e-10);
        assert!((f.momentum - f_phys.momentum).abs() < 1e-10);
        assert!((f.energy - f_phys.energy).abs() < 1e-10);
    }
}
