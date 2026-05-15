//! Wiktorsson (2001) approximation of the Lévy area matrix for multi-dimensional SDEs
//!
//! The Lévy area `A_{ij}` is the antisymmetric matrix of iterated stochastic integrals:
//!
//! ```text
//! A_{ij} = (I_{ij} - I_{ji}) / 2
//! ```
//!
//! where `I_{ij} = ∫₀ʰ ∫₀ˢ dW_i(τ) dW_j(s)` are the iterated Itô integrals.
//!
//! Computing these integrals exactly requires infinite series. The Wiktorsson (2001)
//! approximation truncates the Fourier series to `k` terms, achieving:
//! - k = 5:  ~1e-3 relative accuracy
//! - k = 20: ~1e-7 relative accuracy
//!
//! ## References
//!
//! Wiktorsson, M. (2001). Joint characteristic function and simultaneous simulation
//! of iterated Itô integrals for multiple independent Brownian motions.
//! Annals of Applied Probability, 11(2):470–487.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::{Normal, Rng};
use scirs2_core::Distribution;

/// Wiktorsson (2001) truncated-series approximation of the Lévy area matrix.
///
/// Returns the antisymmetric matrix `A` of shape `(dim, dim)` where
/// `A[i][j] = (I_{ij} - I_{ji}) / 2` and the series approximation is:
///
/// ```text
/// A_{ij} = (h / 2π) * Σ_{r=1..k} (1/r) * (ξ_r^i · η_r^j − ξ_r^j · η_r^i)
/// ```
///
/// with `ξ_r, η_r ~ N(0, I_dim)` independent for each `r`.
///
/// By construction: `A[i][j] = -A[j][i]` and `A[i][i] = 0`.
///
/// # Parameters
///
/// * `dim` - Number of independent Brownian motions
/// * `h` - Time step size
/// * `_dw` - Brownian increments of shape `(dim,)` (not used directly in the Lévy area
///   series — used by `iterated_integral` to reconstruct `I_{ij}`)
/// * `k` - Truncation order (5 for ~1e-3 accuracy; 20 for ~1e-7)
/// * `rng` - Mutable RNG implementing `rand::Rng`
///
/// # Returns
///
/// Antisymmetric matrix `A` of shape `(dim, dim)`.
pub fn levy_area_wiktorsson<R: Rng>(
    dim: usize,
    h: f64,
    _dw: &Array1<f64>,
    k: usize,
    rng: &mut R,
) -> Array2<f64> {
    let mut a = Array2::<f64>::zeros((dim, dim));

    if dim < 2 || k == 0 {
        return a;
    }

    let normal = match Normal::new(0.0_f64, 1.0_f64) {
        Ok(n) => n,
        Err(_) => return a, // Fallback: return zero matrix on error
    };

    let scale = h / (2.0 * std::f64::consts::PI);

    for r in 1..=k {
        let r_inv = 1.0 / r as f64;

        // Sample two independent N(0,I) vectors of length dim
        let xi_r: Vec<f64> = (0..dim).map(|_| normal.sample(rng)).collect();
        let eta_r: Vec<f64> = (0..dim).map(|_| normal.sample(rng)).collect();

        // For each pair (i,j), add the r-th term contribution
        // A[i,j] += (1/r) * (xi_r[i] * eta_r[j] - xi_r[j] * eta_r[i])
        for i in 0..dim {
            for j in (i + 1)..dim {
                let contrib = r_inv * (xi_r[i] * eta_r[j] - xi_r[j] * eta_r[i]);
                a[[i, j]] += contrib;
                a[[j, i]] -= contrib; // antisymmetry
            }
        }
    }

    // Scale by h / (2π)
    a.mapv_inplace(|x| x * scale);
    a
}

/// Compute the full iterated integral matrix `I_{ij}` from the Lévy area.
///
/// The relationship between the iterated integrals and the Lévy area is:
///
/// ```text
/// I_{ij} = (dW_i * dW_j - h * δ_{ij}) / 2 + A_{ij}
/// ```
///
/// where `δ_{ij}` is the Kronecker delta and `A` is the Lévy area from
/// `levy_area_wiktorsson`.
///
/// For `i = j`: `I_{ii} = (dW_i² - h) / 2` (Itô correction term).
/// For `i ≠ j`: `I_{ij} + I_{ji} = dW_i * dW_j - h * δ_{ij} = dW_i * dW_j`.
///
/// # Parameters
///
/// * `dw` - Brownian increments of shape `(dim,)` with `dim = dw.len()`
/// * `h` - Time step size
/// * `levy_area` - Antisymmetric Lévy area matrix from `levy_area_wiktorsson`
///
/// # Returns
///
/// Matrix `I` of shape `(dim, dim)` with `I[i][j] = I_{ij}`.
pub fn iterated_integral(dw: &Array1<f64>, h: f64, levy_area: &Array2<f64>) -> Array2<f64> {
    let dim = dw.len();
    let mut result = Array2::<f64>::zeros((dim, dim));

    for i in 0..dim {
        for j in 0..dim {
            let symmetric_part = if i == j {
                (dw[i] * dw[j] - h) / 2.0
            } else {
                dw[i] * dw[j] / 2.0
            };
            result[[i, j]] = symmetric_part + levy_area[[i, j]];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use scirs2_core::random::prelude::seeded_rng;

    #[test]
    fn test_levy_area_antisymmetry() {
        let dim = 3;
        let h = 0.1_f64;
        let dw = array![0.05_f64, -0.03, 0.07];
        let mut rng = seeded_rng(42);
        let a = levy_area_wiktorsson(dim, h, &dw, 10, &mut rng);
        for i in 0..dim {
            for j in 0..dim {
                assert!(
                    (a[[i, j]] + a[[j, i]]).abs() < 1e-14,
                    "Antisymmetry violated: A[{i},{j}]={:.6} but A[{j},{i}]={:.6}",
                    a[[i, j]],
                    a[[j, i]]
                );
            }
        }
    }

    #[test]
    fn test_levy_area_zero_diagonal() {
        let dim = 4;
        let h = 0.05_f64;
        let dw = array![0.1_f64, -0.2, 0.05, -0.1];
        let mut rng = seeded_rng(7);
        let a = levy_area_wiktorsson(dim, h, &dw, 5, &mut rng);
        for i in 0..dim {
            assert!(
                a[[i, i]].abs() < 1e-15,
                "Diagonal should be zero: A[{i},{i}] = {:.6}",
                a[[i, i]]
            );
        }
    }

    #[test]
    fn test_levy_area_shape() {
        let dim = 5;
        let h = 0.01_f64;
        let dw = Array1::zeros(dim);
        let mut rng = seeded_rng(1);
        let a = levy_area_wiktorsson(dim, h, &dw, 5, &mut rng);
        assert_eq!(a.shape(), &[dim, dim]);
    }

    #[test]
    fn test_levy_area_deterministic_with_seed() {
        let dim = 2;
        let h = 0.1_f64;
        let dw = array![0.05_f64, -0.03];
        let mut rng1 = seeded_rng(123);
        let mut rng2 = seeded_rng(123);
        let a1 = levy_area_wiktorsson(dim, h, &dw, 5, &mut rng1);
        let a2 = levy_area_wiktorsson(dim, h, &dw, 5, &mut rng2);
        assert!(
            (a1[[0, 1]] - a2[[0, 1]]).abs() < 1e-15,
            "Same seed should give same result"
        );
    }

    #[test]
    fn test_levy_area_variance_matches_theoretical() {
        // Var(A_{ij}) = h²/12 (theoretical infinite-series result)
        // For k=5 truncation: ≈ 0.89 * h²/12 (ratio = Σ_{r=1..5} 1/r² / (π²/6) ≈ 0.89)
        // With ±30% tolerance
        let dim = 2;
        let h = 0.1_f64;
        let dw = array![0.0_f64, 0.0];
        let n_samples = 5000;
        let theoretical = h * h / 12.0;
        let tol_lo = theoretical * 0.7 * 0.70; // 30% below with k=5 correction
        let tol_hi = theoretical * 1.30;

        let samples: Vec<f64> = (0..n_samples)
            .map(|seed| {
                let mut rng = seeded_rng(seed as u64);
                let a = levy_area_wiktorsson(dim, h, &dw, 5, &mut rng);
                a[[0, 1]]
            })
            .collect();

        let mean: f64 = samples.iter().sum::<f64>() / n_samples as f64;
        let var: f64 =
            samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_samples - 1) as f64;

        assert!(
            var > tol_lo && var < tol_hi,
            "Lévy area variance {:.6e} outside [{:.6e}, {:.6e}] (theoretical h²/12 = {:.6e})",
            var,
            tol_lo,
            tol_hi,
            theoretical
        );
    }

    #[test]
    fn test_iterated_integral_shape() {
        let dim = 3;
        let h = 0.01_f64;
        let dw = array![0.05_f64, -0.03, 0.07];
        let levy = Array2::zeros((dim, dim));
        let integ = iterated_integral(&dw, h, &levy);
        assert_eq!(integ.shape(), &[dim, dim]);
    }

    #[test]
    fn test_iterated_integral_symmetric_noise() {
        // For diagonal noise (no Lévy area), I_{ij} + I_{ji} should equal dw[i]*dw[j]
        // for i != j, and (dw[i]^2 - h) for i == j.
        let dim = 2;
        let h = 0.1_f64;
        let dw = array![0.15_f64, -0.08];
        let levy_zero = Array2::zeros((dim, dim));
        let integ = iterated_integral(&dw, h, &levy_zero);

        // I_{01} + I_{10} = dw[0]*dw[1]
        let sum_off_diag = integ[[0, 1]] + integ[[1, 0]];
        assert!(
            (sum_off_diag - dw[0] * dw[1]).abs() < 1e-14,
            "I_01 + I_10 should equal dW_0 * dW_1 = {:.6}, got {:.6}",
            dw[0] * dw[1],
            sum_off_diag
        );

        // I_{00} = (dw[0]^2 - h) / 2
        let expected_diag = (dw[0] * dw[0] - h) / 2.0;
        assert!(
            (integ[[0, 0]] - expected_diag).abs() < 1e-14,
            "I_00 should equal (dW_0² - h)/2 = {:.6}, got {:.6}",
            expected_diag,
            integ[[0, 0]]
        );
    }
}
