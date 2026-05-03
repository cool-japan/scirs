//! HiPPO: High-Order Polynomial Projection Operators
//! (Gu, Dao, Ermon, Rudra, Re, 2020)
//!
//! Constructs state matrices (A, B) for continuous-time state-space models
//! that optimally project input history onto polynomial bases.
//!
//! The HiPPO framework provides a principled way to initialise
//! the (A, B) matrices of a continuous-time SSM
//!
//!   x'(t) = A x(t) + B u(t)
//!
//! so that the state x(t) approximates the coefficients of an
//! orthogonal polynomial expansion of the input history u(s), s <= t,
//! with respect to a particular measure.

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// HiPPO variant enumeration
// ---------------------------------------------------------------------------

/// HiPPO matrix variants describing different polynomial projection measures.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HiPPOVariant {
    /// LegS: Legendre (Scaled) -- uniform measure over [0, t].
    ///
    /// This is the original HiPPO matrix and the default initialisation
    /// used in S4.  A is lower-triangular with negative diagonal.
    LegS,

    /// LagT: Laguerre (Translated) -- exponentially-decaying measure.
    ///
    /// Suitable when recent inputs should be weighted more heavily.
    LagT,

    /// LegT: Legendre (Translated) -- sliding window [t - theta, t].
    ///
    /// `theta` is the window length.
    LegT {
        /// Sliding window length (must be > 0).
        theta: f64,
    },

    /// Fourier basis with exponential decay.
    ///
    /// `decay` controls the rate of exponential weighting (must be > 0).
    FourierDecay {
        /// Exponential decay rate.
        decay: f64,
    },
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Construct the HiPPO (A, B) matrices for a given variant and state
/// dimension `n`.
///
/// Returns `(A, B)` where
///   - `A` is an `n x n` matrix (row-major `Vec<Vec<f64>>`)
///   - `B` is an `n`-element column vector (`Vec<f64>`)
///
/// The matrices satisfy the continuous-time ODE
///
///   x'(t) = A x(t) + B u(t)
///
/// # Errors
///
/// Returns [`TimeSeriesError::InvalidInput`] if `n == 0`.
pub fn hippo_matrix(variant: &HiPPOVariant, n: usize) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "state dimension n must be > 0".into(),
        ));
    }
    match variant {
        HiPPOVariant::LegS => hippo_legs(n),
        HiPPOVariant::LagT => hippo_lagt(n),
        HiPPOVariant::LegT { theta } => hippo_legt(n, *theta),
        HiPPOVariant::FourierDecay { decay } => hippo_fourier_decay(n, *decay),
    }
}

// ---------------------------------------------------------------------------
// LegS (Scaled Legendre)
// ---------------------------------------------------------------------------
//
// A[n][k] = -(2n+1)^{1/2} (2k+1)^{1/2}  if n > k
// A[n][n] = -(n+1)
// A[n][k] = 0                              if n < k
// B[n]    = (2n+1)^{1/2}

fn hippo_legs(n: usize) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    let mut a = vec![vec![0.0_f64; n]; n];
    let mut b = vec![0.0_f64; n];

    for i in 0..n {
        let sqrt_2i1 = ((2 * i + 1) as f64).sqrt();
        b[i] = sqrt_2i1;

        // Diagonal: A[i][i] = -(i+1)
        a[i][i] = -((i + 1) as f64);

        // Lower triangle: n > k
        for k in 0..i {
            let sqrt_2k1 = ((2 * k + 1) as f64).sqrt();
            a[i][k] = -sqrt_2i1 * sqrt_2k1;
        }
    }

    Ok((a, b))
}

// ---------------------------------------------------------------------------
// LagT (Translated Laguerre)
// ---------------------------------------------------------------------------
//
// A[n][k] = -1  if n >= k
// A[n][k] =  0  if n < k
// B[n]    =  1  for all n

fn hippo_lagt(n: usize) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    let mut a = vec![vec![0.0_f64; n]; n];
    let b = vec![1.0_f64; n];

    for i in 0..n {
        for k in 0..=i {
            a[i][k] = -1.0;
        }
    }

    Ok((a, b))
}

// ---------------------------------------------------------------------------
// LegT (Translated Legendre, sliding window)
// ---------------------------------------------------------------------------
//
// For window length theta:
//   A[n][k] = (1/theta) * -(2n+1)^{1/2} (2k+1)^{1/2}  if n > k
//   A[n][n] = (1/theta) * -(2n+1)
//   B[n]    = (1/theta) * (2n+1)^{1/2} * (-1)^n    (alternating sign for window edge)
//
// But the canonical form simply scales the LegS matrices by 1/theta
// and adjusts the B sign:
//   A_legT = A_legS / theta   (with modified diagonal)
//   B_legT = B_legS / theta   (with alternating sign)

fn hippo_legt(n: usize, theta: f64) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    if theta <= 0.0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "theta".into(),
            message: "sliding window length must be > 0".into(),
        });
    }

    let inv_theta = 1.0 / theta;
    let mut a = vec![vec![0.0_f64; n]; n];
    let mut b = vec![0.0_f64; n];

    for i in 0..n {
        let val_2i1 = (2 * i + 1) as f64;
        let sqrt_2i1 = val_2i1.sqrt();

        // B with alternating sign
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        b[i] = inv_theta * sqrt_2i1 * sign;

        // Diagonal
        a[i][i] = -inv_theta * val_2i1;

        // Lower triangle
        for k in 0..i {
            let sqrt_2k1 = ((2 * k + 1) as f64).sqrt();
            a[i][k] = -inv_theta * sqrt_2i1 * sqrt_2k1;
        }
    }

    Ok((a, b))
}

// ---------------------------------------------------------------------------
// Fourier with exponential decay
// ---------------------------------------------------------------------------
//
// Uses pairs of (cos, sin) basis functions with exponential decay.
// For pair index p (0-indexed), frequency w_p = pi * (p+1):
//   A block for pair p = [[-decay, -w_p], [w_p, -decay]]
//   B block for pair p = [1, 0]
//
// If n is odd, the last row uses a single decaying exponential.

fn hippo_fourier_decay(n: usize, decay: f64) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    if decay <= 0.0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "decay".into(),
            message: "decay rate must be > 0".into(),
        });
    }

    let mut a = vec![vec![0.0_f64; n]; n];
    let mut b = vec![0.0_f64; n];

    let num_pairs = n / 2;
    for p in 0..num_pairs {
        let w = std::f64::consts::PI * ((p + 1) as f64);
        let i = 2 * p;
        // 2x2 block
        a[i][i] = -decay;
        a[i][i + 1] = -w;
        a[i + 1][i] = w;
        a[i + 1][i + 1] = -decay;

        b[i] = 1.0;
        // b[i+1] = 0.0 (already initialised)
    }

    // If n is odd, last component is a single decaying exponential
    if n % 2 == 1 {
        let last = n - 1;
        a[last][last] = -decay;
        b[last] = 1.0;
    }

    Ok((a, b))
}

// ---------------------------------------------------------------------------
// Utility: check eigenvalue stability (all real parts < 0)
// ---------------------------------------------------------------------------

/// Check that every diagonal element of a (lower-triangular or general)
/// matrix is negative, which is a necessary (though not sufficient for
/// non-triangular) condition for stability.
pub fn check_diagonal_stability(a: &[Vec<f64>]) -> bool {
    a.iter()
        .enumerate()
        .all(|(i, row)| if i < row.len() { row[i] < 0.0 } else { false })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legs_matrix_dimensions() {
        let n = 8;
        let (a, b) = hippo_matrix(&HiPPOVariant::LegS, n).expect("should succeed");
        assert_eq!(a.len(), n);
        assert_eq!(a[0].len(), n);
        assert_eq!(b.len(), n);
    }

    #[test]
    fn test_legs_diagonal_negative() {
        let n = 16;
        let (a, _b) = hippo_matrix(&HiPPOVariant::LegS, n).expect("should succeed");
        for i in 0..n {
            assert!(
                a[i][i] < 0.0,
                "diagonal element A[{}][{}] = {} should be negative",
                i,
                i,
                a[i][i]
            );
        }
    }

    #[test]
    fn test_legs_stability() {
        let n = 32;
        let (a, _b) = hippo_matrix(&HiPPOVariant::LegS, n).expect("should succeed");
        assert!(
            check_diagonal_stability(&a),
            "LegS matrix should have all negative diagonal entries"
        );
    }

    #[test]
    fn test_legs_lower_triangular() {
        let n = 10;
        let (a, _b) = hippo_matrix(&HiPPOVariant::LegS, n).expect("should succeed");
        for i in 0..n {
            for k in (i + 1)..n {
                assert!(
                    a[i][k].abs() < 1e-15,
                    "A[{}][{}] = {} should be zero (upper triangle)",
                    i,
                    k,
                    a[i][k]
                );
            }
        }
    }

    #[test]
    fn test_legs_b_values() {
        let n = 8;
        let (_a, b) = hippo_matrix(&HiPPOVariant::LegS, n).expect("should succeed");
        for i in 0..n {
            let expected = ((2 * i + 1) as f64).sqrt();
            assert!(
                (b[i] - expected).abs() < 1e-12,
                "B[{}] = {} expected {}",
                i,
                b[i],
                expected
            );
        }
    }

    #[test]
    fn test_legs_specific_values() {
        let (a, b) = hippo_matrix(&HiPPOVariant::LegS, 3).expect("should succeed");
        // A[0][0] = -(0+1) = -1
        assert!((a[0][0] - (-1.0)).abs() < 1e-12);
        // A[1][1] = -(1+1) = -2
        assert!((a[1][1] - (-2.0)).abs() < 1e-12);
        // A[2][2] = -(2+1) = -3
        assert!((a[2][2] - (-3.0)).abs() < 1e-12);
        // A[1][0] = -sqrt(3)*sqrt(1) = -sqrt(3)
        assert!((a[1][0] - (-3.0_f64.sqrt())).abs() < 1e-12);
        // B[0] = sqrt(1) = 1
        assert!((b[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_lagt_matrix() {
        let n = 5;
        let (a, b) = hippo_matrix(&HiPPOVariant::LagT, n).expect("should succeed");
        // All B = 1
        for i in 0..n {
            assert!((b[i] - 1.0).abs() < 1e-15);
        }
        // Lower triangular with -1
        for i in 0..n {
            for k in 0..=i {
                assert!((a[i][k] - (-1.0)).abs() < 1e-15);
            }
            for k in (i + 1)..n {
                assert!(a[i][k].abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_legt_matrix() {
        let n = 4;
        let theta = 2.0;
        let (a, b) = hippo_matrix(&HiPPOVariant::LegT { theta }, n).expect("should succeed");
        assert_eq!(a.len(), n);
        assert_eq!(b.len(), n);
        // Diagonal should be negative
        for i in 0..n {
            assert!(
                a[i][i] < 0.0,
                "LegT diagonal A[{}][{}] should be negative",
                i,
                i
            );
        }
    }

    #[test]
    fn test_legt_invalid_theta() {
        let result = hippo_matrix(&HiPPOVariant::LegT { theta: 0.0 }, 4);
        assert!(result.is_err());
        let result = hippo_matrix(&HiPPOVariant::LegT { theta: -1.0 }, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_fourier_decay_matrix() {
        let n = 6;
        let decay = 0.5;
        let (a, b) =
            hippo_matrix(&HiPPOVariant::FourierDecay { decay }, n).expect("should succeed");
        assert_eq!(a.len(), n);
        assert_eq!(b.len(), n);
        // Check 2x2 block structure for first pair
        assert!((a[0][0] - (-decay)).abs() < 1e-12);
        assert!((a[1][1] - (-decay)).abs() < 1e-12);
        let w1 = std::f64::consts::PI;
        assert!((a[0][1] - (-w1)).abs() < 1e-12);
        assert!((a[1][0] - w1).abs() < 1e-12);
    }

    #[test]
    fn test_fourier_decay_odd_n() {
        let n = 5;
        let decay = 1.0;
        let (a, b) =
            hippo_matrix(&HiPPOVariant::FourierDecay { decay }, n).expect("should succeed");
        // Last element: single decaying exponential
        assert!((a[4][4] - (-1.0)).abs() < 1e-12);
        assert!((b[4] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_zero_state_dim_error() {
        let result = hippo_matrix(&HiPPOVariant::LegS, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_fourier_invalid_decay() {
        let result = hippo_matrix(&HiPPOVariant::FourierDecay { decay: 0.0 }, 4);
        assert!(result.is_err());
        let result = hippo_matrix(&HiPPOVariant::FourierDecay { decay: -1.0 }, 4);
        assert!(result.is_err());
    }
}
