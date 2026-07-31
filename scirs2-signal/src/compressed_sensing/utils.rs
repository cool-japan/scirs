//! Utility functions for compressed sensing.
//!
//! Provides threshold operators, sparse signal generators, and helper
//! routines shared across CS algorithms.
//!
//! # Contents
//!
//! - [`soft_threshold`] – Proximal operator for the L1 norm (scalar).
//! - [`soft_threshold_vec`] – Element-wise soft thresholding of a vector.
//! - [`hard_threshold`] – Keep only the `k` largest-magnitude coefficients.
//! - [`hard_threshold_val`] – Scalar hard threshold with cutoff τ.
//! - [`generate_sparse_signal`] – Random k-sparse vector for testing.
//! - [`support_of`] – Indices of non-zero entries.
//! - [`restrict_to`] – Zero out all entries not in a given support set.
//!
//! Pure Rust, no `unwrap()`, snake_case naming throughout.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{RngExt, SeedableRng};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Soft thresholding
// ---------------------------------------------------------------------------

/// Scalar soft-threshold (proximal operator for the L1 norm).
///
/// Returns `sign(v) · max(|v| − τ, 0)`.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::utils::soft_threshold;
/// assert!((soft_threshold(3.0, 1.0) - 2.0).abs() < 1e-15);
/// assert_eq!(soft_threshold(0.5, 1.0), 0.0);
/// assert!((soft_threshold(-3.0, 1.0) + 2.0).abs() < 1e-15);
/// ```
#[inline]
pub fn soft_threshold(v: f64, tau: f64) -> f64 {
    if v > tau {
        v - tau
    } else if v < -tau {
        v + tau
    } else {
        0.0
    }
}

/// Element-wise soft thresholding of a vector.
///
/// Applies [`soft_threshold`] independently to every element of `x`.
///
/// # Arguments
///
/// * `x`   – Input vector.
/// * `tau` – Threshold parameter (≥ 0).
///
/// # Errors
///
/// Returns [`SignalError::ValueError`] if `tau < 0`.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::utils::soft_threshold_vec;
/// use scirs2_core::ndarray::Array1;
/// let x = Array1::from_vec(vec![3.0, -0.5, 1.0, -2.0]);
/// let y = soft_threshold_vec(&x, 1.0).expect("operation should succeed");
/// assert!((y[0] - 2.0).abs() < 1e-15);
/// assert_eq!(y[1], 0.0);
/// assert_eq!(y[2], 0.0);
/// assert!((y[3] + 1.0).abs() < 1e-15);
/// ```
pub fn soft_threshold_vec(x: &Array1<f64>, tau: f64) -> SignalResult<Array1<f64>> {
    if tau < 0.0 {
        return Err(SignalError::ValueError(
            "soft_threshold_vec: tau must be non-negative".to_string(),
        ));
    }
    Ok(x.mapv(|v| soft_threshold(v, tau)))
}

// ---------------------------------------------------------------------------
// Hard thresholding
// ---------------------------------------------------------------------------

/// Scalar hard-threshold: keep the value if `|v| > τ`, else zero.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::utils::hard_threshold_val;
/// assert_eq!(hard_threshold_val(3.0, 1.0), 3.0);
/// assert_eq!(hard_threshold_val(0.5, 1.0), 0.0);
/// assert_eq!(hard_threshold_val(-3.0, 1.0), -3.0);
/// ```
#[inline]
pub fn hard_threshold_val(v: f64, tau: f64) -> f64 {
    if v.abs() > tau {
        v
    } else {
        0.0
    }
}

/// Keep only the `k` largest-magnitude entries of `x`; zero out the rest.
///
/// If `k >= x.len()`, `x` is returned unchanged.  Ties are broken by index
/// (lower index wins).
///
/// # Arguments
///
/// * `x` – Input vector.
/// * `k` – Number of non-zero entries to retain.
///
/// # Errors
///
/// Returns [`SignalError::ValueError`] if `k == 0`.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::utils::hard_threshold;
/// use scirs2_core::ndarray::Array1;
/// let x = Array1::from_vec(vec![1.0, 5.0, -3.0, 2.0, -0.1]);
/// let y = hard_threshold(&x, 2).expect("operation should succeed");
/// // Two largest: index 1 (5.0) and index 2 (-3.0)
/// assert_eq!(y[1], 5.0);
/// assert_eq!(y[2], -3.0);
/// assert_eq!(y[0], 0.0);
/// ```
pub fn hard_threshold(x: &Array1<f64>, k: usize) -> SignalResult<Array1<f64>> {
    if k == 0 {
        return Err(SignalError::ValueError(
            "hard_threshold: k must be positive".to_string(),
        ));
    }
    let n = x.len();
    if k >= n {
        return Ok(x.clone());
    }

    // Sort indices by descending absolute value
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        x[b].abs()
            .partial_cmp(&x[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out = Array1::zeros(n);
    for &idx in &indices[..k] {
        out[idx] = x[idx];
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Sparse signal generation
// ---------------------------------------------------------------------------

/// Generate a random k-sparse signal of length `n` for testing.
///
/// Exactly `k` entries are chosen uniformly at random and assigned i.i.d.
/// N(0, 1) amplitudes.  All other entries are zero.
///
/// # Arguments
///
/// * `n`       – Signal dimension.
/// * `k`       – Sparsity (number of non-zero entries, must satisfy `k ≤ n`).
/// * `seed`    – RNG seed for reproducibility.
///
/// # Returns
///
/// A dense `Array1<f64>` of length `n` with exactly `k` non-zero entries.
///
/// # Errors
///
/// Returns [`SignalError::ValueError`] if `k > n` or either is zero.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::utils::generate_sparse_signal;
/// let x = generate_sparse_signal(64, 5, 42).expect("operation should succeed");
/// assert_eq!(x.len(), 64);
/// let nnz = x.iter().filter(|&&v| v != 0.0).count();
/// assert_eq!(nnz, 5);
/// ```
pub fn generate_sparse_signal(n: usize, k: usize, seed: u64) -> SignalResult<Array1<f64>> {
    if n == 0 {
        return Err(SignalError::ValueError(
            "generate_sparse_signal: n must be positive".to_string(),
        ));
    }
    if k == 0 {
        return Err(SignalError::ValueError(
            "generate_sparse_signal: k must be positive".to_string(),
        ));
    }
    if k > n {
        return Err(SignalError::ValueError(format!(
            "generate_sparse_signal: sparsity k={k} > signal length n={n}"
        )));
    }

    let mut rng = StdRng::seed_from_u64(seed);

    // Fisher-Yates shuffle to select k distinct support indices
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (rng.random::<u64>() as usize % (n - i));
        indices.swap(i, j);
    }
    let support = &indices[..k];

    let mut x = Array1::zeros(n);
    for &idx in support.iter() {
        // Box-Muller Gaussian sample
        let u1: f64 = {
            let v: f64 = rng.random();
            if v < 1e-300 {
                1e-300
            } else {
                v
            }
        };
        let u2: f64 = rng.random();
        let amp = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        x[idx] = amp;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Support utilities
// ---------------------------------------------------------------------------

/// Return the indices of non-zero entries (absolute value > `tol`).
///
/// # Arguments
///
/// * `x`   – Input vector.
/// * `tol` – Entries with `|x[i]|` ≤ tol are treated as zero.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::utils::support_of;
/// use scirs2_core::ndarray::Array1;
/// let x = Array1::from_vec(vec![0.0, 1.0, 0.0, -2.0, 1e-16]);
/// let s = support_of(&x, 1e-10);
/// assert_eq!(s, vec![1, 3]);
/// ```
pub fn support_of(x: &Array1<f64>, tol: f64) -> Vec<usize> {
    x.iter()
        .enumerate()
        .filter_map(|(i, &v)| if v.abs() > tol { Some(i) } else { None })
        .collect()
}

/// Zero out all entries of `x` whose index is not in `support`.
///
/// Returns a new vector of the same length as `x`.
///
/// # Arguments
///
/// * `x`       – Input vector.
/// * `support` – Indices of entries to keep.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::utils::restrict_to;
/// use scirs2_core::ndarray::Array1;
/// let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let y = restrict_to(&x, &[1, 3]);
/// assert_eq!(y[0], 0.0);
/// assert_eq!(y[1], 2.0);
/// assert_eq!(y[2], 0.0);
/// assert_eq!(y[3], 4.0);
/// ```
pub fn restrict_to(x: &Array1<f64>, support: &[usize]) -> Array1<f64> {
    let mut out = Array1::zeros(x.len());
    for &idx in support.iter() {
        if idx < x.len() {
            out[idx] = x[idx];
        }
    }
    out
}

/// Compute the L2 norm of a vector.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::utils::l2_norm;
/// use scirs2_core::ndarray::Array1;
/// let x = Array1::from_vec(vec![3.0, 4.0]);
/// assert!((l2_norm(&x) - 5.0).abs() < 1e-15);
/// ```
#[inline]
pub fn l2_norm(x: &Array1<f64>) -> f64 {
    x.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

/// Compute the L1 norm (sum of absolute values) of a vector.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::utils::l1_norm;
/// use scirs2_core::ndarray::Array1;
/// let x = Array1::from_vec(vec![-1.0, 2.0, -3.0]);
/// assert!((l1_norm(&x) - 6.0).abs() < 1e-15);
/// ```
#[inline]
pub fn l1_norm(x: &Array1<f64>) -> f64 {
    x.iter().map(|&v| v.abs()).sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_soft_threshold_positive() {
        assert!((soft_threshold(3.0, 1.0) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_soft_threshold_negative() {
        assert!((soft_threshold(-3.0, 1.0) + 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_soft_threshold_inside_band() {
        assert_eq!(soft_threshold(0.5, 1.0), 0.0);
        assert_eq!(soft_threshold(-0.9, 1.0), 0.0);
    }

    #[test]
    fn test_soft_threshold_vec() {
        let x = Array1::from_vec(vec![3.0, -0.5, 1.0, -2.0]);
        let y = soft_threshold_vec(&x, 1.0).expect("should succeed");
        assert!((y[0] - 2.0).abs() < 1e-15);
        assert_eq!(y[1], 0.0);
        assert_eq!(y[2], 0.0);
        assert!((y[3] + 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_soft_threshold_vec_negative_tau() {
        let x = Array1::from_vec(vec![1.0]);
        assert!(soft_threshold_vec(&x, -0.1).is_err());
    }

    #[test]
    fn test_hard_threshold_val() {
        assert_eq!(hard_threshold_val(3.0, 1.0), 3.0);
        assert_eq!(hard_threshold_val(0.5, 1.0), 0.0);
        assert_eq!(hard_threshold_val(-3.0, 1.0), -3.0);
        assert_eq!(hard_threshold_val(1.0, 1.0), 0.0); // not strictly greater
    }

    #[test]
    fn test_hard_threshold_top_k() {
        let x = Array1::from_vec(vec![1.0, 5.0, -3.0, 2.0, -0.1]);
        let y = hard_threshold(&x, 2).expect("should succeed");
        assert_eq!(y[1], 5.0);
        assert_eq!(y[2], -3.0);
        assert_eq!(y[0], 0.0);
        assert_eq!(y[3], 0.0);
        assert_eq!(y[4], 0.0);
    }

    #[test]
    fn test_hard_threshold_zero_k_error() {
        let x = Array1::from_vec(vec![1.0, 2.0]);
        assert!(hard_threshold(&x, 0).is_err());
    }

    #[test]
    fn test_generate_sparse_signal_sparsity() {
        let x = generate_sparse_signal(64, 5, 42).expect("should succeed");
        assert_eq!(x.len(), 64);
        let nnz = x.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nnz, 5);
    }

    #[test]
    fn test_generate_sparse_signal_errors() {
        assert!(generate_sparse_signal(0, 1, 0).is_err());
        assert!(generate_sparse_signal(4, 0, 0).is_err());
        assert!(generate_sparse_signal(4, 5, 0).is_err());
    }

    #[test]
    fn test_support_of() {
        let x = Array1::from_vec(vec![0.0, 1.0, 0.0, -2.0, 1e-16]);
        let s = support_of(&x, 1e-10);
        assert_eq!(s, vec![1, 3]);
    }

    #[test]
    fn test_restrict_to() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y = restrict_to(&x, &[1, 3]);
        assert_eq!(y[0], 0.0);
        assert_eq!(y[1], 2.0);
        assert_eq!(y[2], 0.0);
        assert_eq!(y[3], 4.0);
    }

    #[test]
    fn test_l2_norm() {
        let x = Array1::from_vec(vec![3.0, 4.0]);
        assert!((l2_norm(&x) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_l1_norm() {
        let x = Array1::from_vec(vec![-1.0, 2.0, -3.0]);
        assert!((l1_norm(&x) - 6.0).abs() < 1e-15);
    }
}
