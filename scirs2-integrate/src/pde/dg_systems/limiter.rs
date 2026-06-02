//! Slope limiters and troubled-cell indicators for the DG Euler solver.
//!
//! - `SlopeLimiter` trait: applied to a single element after each time step.
//! - `MinmodTvbLimiter`: Cockburn-Shu TVB-modified minmod limiter.
//! - `PerssonPeraireIndicator` trait: identifies troubled elements.
//! - `StandardPerssonPeraire`: spectral smoothness indicator (Persson 2006).

use super::euler_1d::EulerState;

/// Trait for slope limiters applied to each DG element.
///
/// The limiter receives modal coefficients for the current element and its two
/// neighbours (left and right), and may modify the current element's modal coefficients
/// to reduce spurious oscillations.
pub trait SlopeLimiter: Send + Sync {
    /// Limit the modal coefficients of `current_element`.
    ///
    /// # Arguments
    /// - `modal_coeffs`: mutable slice of modal coefficients for the current element
    ///   (length = polynomial_order + 1). Coefficient 0 is the cell average.
    /// - `left_avg`: cell average (mode 0) of the left neighbour.
    /// - `right_avg`: cell average (mode 0) of the right neighbour.
    /// - `h`: element width (for TVB constant scaling).
    fn limit(
        &self,
        modal_coeffs: &mut [EulerState],
        left_avg: EulerState,
        right_avg: EulerState,
        h: f64,
    );
}

/// TVB-modified minmod slope limiter (Cockburn & Shu 1989, 1998).
///
/// Limits each component of `EulerState` independently using the generalized minmod
/// function with TVB constant `m_constant`. Sets all modes above p=1 to zero for
/// troubled cells (piecewise-linear reconstruction).
pub struct MinmodTvbLimiter {
    /// TVB constant M. Larger values allow smoother solutions near smooth extrema.
    /// Typical: M = 1..50 for Euler; M = 0 recovers the strict minmod limiter.
    pub m_constant: f64,
}

impl MinmodTvbLimiter {
    /// Create a new TVB limiter with the given M constant.
    pub fn new(m_constant: f64) -> Self {
        Self { m_constant }
    }
}

/// Minmod function: min(a, b, c) in magnitude if same sign, else 0.
fn minmod(a: f64, b: f64, c: f64) -> f64 {
    if a > 0.0 && b > 0.0 && c > 0.0 {
        a.min(b).min(c)
    } else if a < 0.0 && b < 0.0 && c < 0.0 {
        a.max(b).max(c)
    } else {
        0.0
    }
}

/// TVB-modified minmod: returns `a` unchanged if |a| ≤ M·h².
fn minmod_tvb(a: f64, b: f64, c: f64, m: f64, h: f64) -> f64 {
    if a.abs() <= m * h * h {
        a
    } else {
        minmod(a, b, c)
    }
}

/// Apply the minmod limiter component-wise to a single `f64` field of `EulerState`.
fn limit_component(
    slope: f64,
    left_avg: f64,
    cell_avg: f64,
    right_avg: f64,
    m: f64,
    h: f64,
) -> f64 {
    let fwd = right_avg - cell_avg;
    let bwd = cell_avg - left_avg;
    minmod_tvb(slope, fwd, bwd, m, h)
}

impl SlopeLimiter for MinmodTvbLimiter {
    fn limit(
        &self,
        modal_coeffs: &mut [EulerState],
        left_avg: EulerState,
        right_avg: EulerState,
        h: f64,
    ) {
        if modal_coeffs.is_empty() {
            return;
        }
        let cell_avg = modal_coeffs[0];
        if modal_coeffs.len() == 1 {
            return; // p=0 — nothing to limit
        }
        // Limit the linear (p=1) mode, which represents the slope.
        // For GLL nodal DG, mode 1 ≈ slope * h/2 (Legendre normalisation varies by convention).
        // We limit it using neighbour cell averages.
        let slope = modal_coeffs[1];
        let new_rho = limit_component(
            slope.rho,
            left_avg.rho,
            cell_avg.rho,
            right_avg.rho,
            self.m_constant,
            h,
        );
        let new_rho_u = limit_component(
            slope.rho_u,
            left_avg.rho_u,
            cell_avg.rho_u,
            right_avg.rho_u,
            self.m_constant,
            h,
        );
        let new_energy = limit_component(
            slope.energy,
            left_avg.energy,
            cell_avg.energy,
            right_avg.energy,
            self.m_constant,
            h,
        );
        // Check if limiting was active (any component changed)
        let limited = (new_rho - slope.rho).abs() > f64::EPSILON
            || (new_rho_u - slope.rho_u).abs() > f64::EPSILON
            || (new_energy - slope.energy).abs() > f64::EPSILON;
        if limited {
            // Set the linear mode to the limited slope
            modal_coeffs[1] = EulerState {
                rho: new_rho,
                rho_u: new_rho_u,
                energy: new_energy,
            };
            // Zero out all higher modes (piecewise-linear reconstruction)
            for coeff in modal_coeffs[2..].iter_mut() {
                *coeff = EulerState::zero();
            }
        }
    }
}

/// Trait for troubled-cell indicators used to decide which elements need limiting.
pub trait PerssonPeraireIndicator: Send + Sync {
    /// Compute the smoothness indicator S_k for an element.
    ///
    /// Returns a dimensionless value; elements with S_k above the threshold are "troubled".
    ///
    /// # Arguments
    /// - `modal_coeffs`: modal coefficients of the density field for this element.
    fn evaluate(&self, modal_coeffs: &[f64]) -> f64;
}

/// Persson-Peraire spectral smoothness indicator (Persson & Peraire 2006).
///
/// `S_k = log10( <u - u_{p-1}>² / <u>² )`
///
/// where `<·>²` is the L2 norm and `u_{p-1}` is the projection onto the (p-1)-th order space.
/// Elements are flagged as troubled if `S_k > threshold` (typically `-kappa * log10(p)`).
pub struct StandardPerssonPeraire {
    /// Controls the threshold: troubled if S_k > -kappa * log10(p).
    /// Typical kappa = 4.0 (Persson 2006).
    pub kappa: f64,
}

impl StandardPerssonPeraire {
    /// Create a new Persson-Peraire indicator with the given kappa.
    pub fn new(kappa: f64) -> Self {
        Self { kappa }
    }

    /// Compute the threshold for a given polynomial order `p`.
    pub fn threshold(&self, p: usize) -> f64 {
        if p == 0 {
            f64::NEG_INFINITY
        } else {
            -self.kappa * (p as f64).log10()
        }
    }
}

impl PerssonPeraireIndicator for StandardPerssonPeraire {
    fn evaluate(&self, modal_coeffs: &[f64]) -> f64 {
        let p = modal_coeffs.len().saturating_sub(1);
        if p == 0 {
            return f64::NEG_INFINITY; // p=0: always smooth
        }
        // L2 norm of full representation: sum of squares of all modal coefficients
        // (using L2-orthonormal Legendre basis where ||φ_j||² = 1)
        let norm_u_sq: f64 = modal_coeffs.iter().map(|c| c * c).sum();
        if norm_u_sq < f64::EPSILON {
            return f64::NEG_INFINITY;
        }
        // L2 norm of the highest mode (approximation of u - u_{p-1})
        let highest_mode_sq = modal_coeffs[p] * modal_coeffs[p];
        (highest_mode_sq / norm_u_sq).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn es(rho: f64, rho_u: f64, energy: f64) -> EulerState {
        EulerState { rho, rho_u, energy }
    }

    #[test]
    fn test_minmod_basic() {
        assert!((minmod(1.0, 2.0, 3.0) - 1.0).abs() < 1e-14);
        assert!((minmod(-1.0, -2.0, -3.0) + 1.0).abs() < 1e-14);
        assert!(minmod(1.0, -1.0, 0.5).abs() < 1e-14); // different signs
    }

    #[test]
    fn test_minmod_tvb_passthrough_small_slope() {
        // If |a| <= M*h^2, return a unchanged
        let a = 0.001;
        let h = 0.1;
        let m = 10.0; // M*h^2 = 0.1 > 0.001
        let result = minmod_tvb(a, -1.0, 1.0, m, h); // b, c would normally zero it
        assert!((result - a).abs() < 1e-14);
    }

    #[test]
    fn test_limiter_leaves_low_order_smooth_solution() {
        let limiter = MinmodTvbLimiter::new(0.0); // strict minmod, no TVB
                                                  // Smooth linear solution: cell avg = 1.0, slope = 0.1
                                                  // Neighbours have averages 0.9 and 1.1 (matching the slope perfectly)
        let mut modal = vec![es(1.0, 0.5, 3.0), es(0.1, 0.05, 0.3)];
        let left = es(0.9, 0.45, 2.7);
        let right = es(1.1, 0.55, 3.3);
        let slope_before = modal[1];
        limiter.limit(&mut modal, left, right, 1.0);
        // slope should be unchanged because minmod(0.1, 0.1, 0.1) = 0.1
        assert!((modal[1].rho - slope_before.rho).abs() < 1e-10);
    }

    #[test]
    fn test_limiter_zeros_out_oscillatory_high_modes() {
        let limiter = MinmodTvbLimiter::new(0.0);
        // Slope is 1.0, but neighbours both have avg = 1.0 (no variation)
        // → minmod(1.0, 0.0, 0.0) = 0 → slope zeroed; higher modes also zeroed
        let mut modal = vec![
            es(1.0, 1.0, 3.0), // cell avg
            es(1.0, 1.0, 1.0), // linear mode (large slope)
            es(0.5, 0.5, 0.5), // quadratic mode
        ];
        let same = es(1.0, 1.0, 3.0);
        limiter.limit(&mut modal, same, same, 1.0);
        // Slope should be zeroed, higher modes too
        assert!(modal[1].rho.abs() < 1e-10);
        assert!(modal[2].rho.abs() < 1e-10);
    }

    #[test]
    fn test_persson_peraire_smooth_element_negative_indicator() {
        let indicator = StandardPerssonPeraire::new(4.0);
        // Dominant low-frequency mode: most energy in mode 0
        let modal = vec![1.0, 0.01, 0.001]; // p=2
        let s = indicator.evaluate(&modal);
        assert!(
            s < -3.0,
            "smooth element should have very negative indicator: s={s}"
        );
    }

    #[test]
    fn test_persson_peraire_troubled_element_positive_indicator() {
        let indicator = StandardPerssonPeraire::new(4.0);
        // Most energy in highest mode → troubled
        let modal = vec![0.001, 0.001, 1.0]; // p=2, high-mode dominant
        let s = indicator.evaluate(&modal);
        assert!(
            s > -3.0,
            "troubled element should have indicator above threshold: s={s}"
        );
    }

    #[test]
    fn test_persson_peraire_threshold() {
        let indicator = StandardPerssonPeraire::new(4.0);
        // p=1: threshold = -4.0 * log10(1) = 0.0
        assert!((indicator.threshold(1) - 0.0).abs() < 1e-14);
        // p=10: threshold = -4.0 * log10(10) = -4.0
        assert!((indicator.threshold(10) - (-4.0)).abs() < 1e-14);
        // p=0: always -inf
        assert_eq!(indicator.threshold(0), f64::NEG_INFINITY);
    }

    #[test]
    fn test_limiter_p0_element_unchanged() {
        let limiter = MinmodTvbLimiter::new(0.0);
        let mut modal = vec![es(1.0, 0.5, 3.0)]; // p=0
        let left = es(0.5, 0.25, 1.5);
        let right = es(1.5, 0.75, 4.5);
        let avg_before = modal[0];
        limiter.limit(&mut modal, left, right, 1.0);
        // p=0: cell average should remain unchanged
        assert!((modal[0].rho - avg_before.rho).abs() < 1e-14);
    }

    #[test]
    fn test_limiter_empty_unchanged() {
        let limiter = MinmodTvbLimiter::new(1.0);
        let mut modal: Vec<EulerState> = vec![];
        // Should not panic on empty input
        limiter.limit(&mut modal, EulerState::zero(), EulerState::zero(), 1.0);
        assert!(modal.is_empty());
    }
}
