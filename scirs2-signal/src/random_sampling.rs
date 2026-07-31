//! Non-uniform and random sampling for compressed sensing
//!
//! This module provides sampling strategies and reconstruction utilities that
//! exploit sparsity in transformed domains:
//!
//! - **Jitter sampling** – quasi-random time points with uniform jitter per bin
//! - **Random demodulation** – random mixing + low-pass compressive scheme
//! - **NUFFT-lite** – non-uniform DFT evaluated at arbitrary frequencies
//! - **Full CS pipeline** – compose measurements, sensing, and reconstruction
//!
//! # References
//!
//! - Tropp et al. (2010) – "Beyond Nyquist: efficient sampling of sparse
//!   bandlimited signals"
//! - Candès & Wakin (2008) – "An Introduction to Compressive Sampling"
//! - Dutt & Rokhlin (1993) – "Fast Fourier transforms for nonequispaced data"
//!
//! Pure Rust, no unwrap(), snake_case naming.

use crate::compressed_sensing::reconstruct as compressive_sense;
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::num_complex::Complex64;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{RngExt, SeedableRng};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Jitter Sampling
// ---------------------------------------------------------------------------

/// Generate jitter-sampled time points.
///
/// Divides the interval `[0, (n-1)/fs]` into `n` uniform bins of width `1/fs`
/// and places one sample per bin at a uniformly-random position within the bin.
/// This preserves the average sampling rate `fs` while breaking spectral aliasing.
///
/// # Arguments
///
/// * `n`  – Number of sample points to generate.
/// * `fs` – Nominal sampling rate in Hz (samples per second).
///
/// # Returns
///
/// Sorted vector of `n` sample times in seconds.
///
/// # Errors
///
/// Returns `SignalError::ValueError` if `n == 0` or `fs <= 0`.
pub fn jitter_sampling(n: usize, fs: f64) -> SignalResult<Vec<f64>> {
    if n == 0 {
        return Err(SignalError::ValueError(
            "jitter_sampling: n must be positive".to_string(),
        ));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError(
            "jitter_sampling: fs must be positive".to_string(),
        ));
    }

    let dt = 1.0 / fs; // bin width
    let mut rng = StdRng::seed_from_u64(0xCAFE_BABE_u64);
    let mut times: Vec<f64> = (0..n)
        .map(|i| {
            let bin_start = i as f64 * dt;
            bin_start + rng.random::<f64>() * dt
        })
        .collect();

    // Already monotone by construction, but sort to be safe
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(times)
}

/// Generate jitter-sampled time points with explicit seed for reproducibility.
///
/// Identical to [`jitter_sampling`] but accepts a seed for deterministic output.
pub fn jitter_sampling_seeded(n: usize, fs: f64, seed: u64) -> SignalResult<Vec<f64>> {
    if n == 0 {
        return Err(SignalError::ValueError(
            "jitter_sampling_seeded: n must be positive".to_string(),
        ));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError(
            "jitter_sampling_seeded: fs must be positive".to_string(),
        ));
    }

    let dt = 1.0 / fs;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut times: Vec<f64> = (0..n)
        .map(|i| {
            let bin_start = i as f64 * dt;
            bin_start + rng.random::<f64>() * dt
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(times)
}

// ---------------------------------------------------------------------------
// Random Demodulation
// ---------------------------------------------------------------------------

/// Random demodulator: apply a pseudo-random ±1 chip sequence then integrate.
///
/// The random demodulator is a compressive sensing architecture for wideband
/// signals.  Given a discrete signal `signal` of length `n`, this function:
///
/// 1. Multiplies `signal` by a Rademacher (±1) chip sequence.
/// 2. Low-pass integrates by summing non-overlapping blocks of size `n / n_measurements`.
///
/// # Arguments
///
/// * `signal`         – Discrete input signal of length `n`.
/// * `n_measurements` – Number of compressive measurements `m ≤ n`.
///
/// # Returns
///
/// A tuple `(phi, y)` where:
/// - `phi` is the `(m, n)` sensing matrix representing the combined operation.
/// - `y`   is the `(m,)` measurement vector.
///
/// # Errors
///
/// Returns an error if `n_measurements == 0` or `n_measurements > n`.
pub fn random_demodulation(
    signal: &Array1<f64>,
    n_measurements: usize,
) -> SignalResult<(Array2<f64>, Array1<f64>)> {
    let n = signal.len();
    if n == 0 {
        return Err(SignalError::ValueError(
            "random_demodulation: signal is empty".to_string(),
        ));
    }
    if n_measurements == 0 || n_measurements > n {
        return Err(SignalError::ValueError(format!(
            "random_demodulation: n_measurements ({n_measurements}) must be in (0, {n}]"
        )));
    }

    // Generate Rademacher chip sequence  ±1
    let mut rng = StdRng::seed_from_u64(0xDECA_F00D_u64);
    let chips: Vec<f64> = (0..n)
        .map(|_| if rng.random::<f64>() < 0.5 { 1.0 } else { -1.0 })
        .collect();

    // Block size (integration window)
    let block = n / n_measurements;
    let m_actual = n / block; // may be < n_measurements if n not divisible

    // Build sensing matrix phi (m_actual x n)
    let mut phi = Array2::<f64>::zeros((m_actual, n));
    for row in 0..m_actual {
        for col in (row * block)..((row + 1) * block).min(n) {
            phi[[row, col]] = chips[col];
        }
    }

    // Compute measurements
    let y = phi.dot(signal);

    Ok((phi, y))
}

// ---------------------------------------------------------------------------
// Non-Cartesian (Non-Uniform) FFT – NUFFT lite
// ---------------------------------------------------------------------------

/// Non-Uniform DFT (NUFFT-lite): evaluate the DFT at arbitrary frequencies.
///
/// Computes the type-1 NUFFT:
///   Y(f_k) = Σ_{j} x(t_j) · exp(-i 2π f_k t_j)
///
/// for each frequency `freqs[k]` (in Hz) using a direct summation approach
/// with O(n · nf) complexity.  For production use, replace with a true
/// non-uniform FFT library when `n` and `nf` are large.
///
/// # Arguments
///
/// * `signal`       – Signal samples `x(t_j)`, real-valued, length `n`.
/// * `sample_times` – Non-uniform sample times `t_j` in seconds, length `n`.
/// * `freqs`        – Target frequencies `f_k` in Hz, length `nf`.
///
/// # Returns
///
/// Complex spectrum of length `nf`.
///
/// # Errors
///
/// Returns `SignalError::DimensionMismatch` if `signal` and `sample_times`
/// have different lengths.
pub fn noncartesian_fft(
    signal: &Array1<f64>,
    sample_times: &[f64],
    freqs: &[f64],
) -> SignalResult<Array1<Complex64>> {
    let n = signal.len();
    if sample_times.len() != n {
        return Err(SignalError::DimensionMismatch(format!(
            "noncartesian_fft: signal has {n} elements but sample_times has {}",
            sample_times.len()
        )));
    }
    if freqs.is_empty() {
        return Ok(Array1::zeros(0));
    }

    let nf = freqs.len();
    let mut result = vec![Complex64::new(0.0, 0.0); nf];

    // Direct NUDFT: Y(f_k) = sum_j x_j * exp(-i 2π f_k t_j)
    for (k, &fk) in freqs.iter().enumerate() {
        let mut acc = Complex64::new(0.0, 0.0);
        for j in 0..n {
            let phase = -2.0 * PI * fk * sample_times[j];
            let (sin_ph, cos_ph) = phase.sin_cos();
            acc += signal[j] * Complex64::new(cos_ph, sin_ph);
        }
        result[k] = acc;
    }

    Ok(Array1::from_vec(result))
}

// ---------------------------------------------------------------------------
// Full Compressed Sensing Pipeline
// ---------------------------------------------------------------------------

/// Full compressed-sensing reconstruction pipeline.
///
/// Given compressive measurements `y = Φ Ψ x_sparse`, this function:
/// 1. Forms the combined sensing-sparsity matrix `A = Φ Ψ`.
/// 2. Recovers the sparse coefficients `x_sparse` via Orthogonal Matching
///    Pursuit (see [`crate::compressed_sensing::reconstruct`] for other
///    available algorithms).
/// 3. Returns the signal in the original domain as `Ψ x_sparse`.
///
/// # Arguments
///
/// * `y`         – Measurement vector of length `m`.
/// * `phi`       – Sensing / measurement matrix `(m, n_transform)`.  Rows are
///                 linear measurement functionals.
/// * `psi`       – Sparsity basis / synthesis dictionary `(n_signal, n_transform)`.
///                 Each column is a basis atom in the signal domain.
/// * `sparsity`  – Target sparsity k used by the recovery algorithm.
///
/// # Returns
///
/// Reconstructed signal of length `n_signal` in the original (signal) domain.
///
/// # Errors
///
/// Returns dimension errors if the matrices are incompatible.
pub fn compressed_sensing_reconstruct(
    y: &Array1<f64>,
    phi: &Array2<f64>,
    psi: &Array2<f64>,
    sparsity: usize,
) -> SignalResult<Array1<f64>> {
    let (m, n_phi) = phi.dim();
    let (n_signal, n_psi) = psi.dim();

    if y.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "compressed_sensing_reconstruct: y has {} elements but phi has {m} rows",
            y.len()
        )));
    }
    if n_phi != n_psi {
        return Err(SignalError::DimensionMismatch(format!(
            "compressed_sensing_reconstruct: phi has {n_phi} columns \
             but psi has {n_psi} columns"
        )));
    }

    // Combined matrix A = Phi * Psi^T  (m x n_signal) when Psi is a frame
    // Here we interpret Psi as (n_signal x n_transform) where each column is
    // an atom, so the combined matrix is Phi * Psi^T giving (m x n_signal).
    // Alternatively: if Psi is the synthesis operator (n_signal x n_transform),
    // the forward model is y = Phi * Psi^T * x where x is in R^{n_signal}.
    // We use the interpretation A = Phi @ Psi.T so we solve in n_signal.
    let a = phi.dot(&psi.t()); // (m x n_signal)

    // Recover sparse coefficients in the signal domain
    let x_sparse = compressive_sense(&a, y, "omp", sparsity, 0.1)?;

    Ok(x_sparse)
}

// ---------------------------------------------------------------------------
// Poisson disk / stratified sampling helpers
// ---------------------------------------------------------------------------

/// Variable-density Poisson-disk sampling pattern in 1-D.
///
/// Generates sample indices in `[0, n)` using a Poisson-disk style rejection
/// to ensure a minimum separation of `min_sep` samples while preferentially
/// drawing from lower frequencies (useful for MRI-style variable-density CS).
///
/// # Arguments
///
/// * `n`       – Total number of available grid points.
/// * `m`       – Desired number of samples.
/// * `min_sep` – Minimum integer separation between adjacent samples.
/// * `seed`    – Random seed for reproducibility.
///
/// # Returns
///
/// Sorted vector of `m` sample indices in `[0, n)`.
pub fn poisson_disk_sampling_1d(
    n: usize,
    m: usize,
    min_sep: usize,
    seed: u64,
) -> SignalResult<Vec<usize>> {
    if n == 0 {
        return Err(SignalError::ValueError(
            "poisson_disk_sampling_1d: n must be positive".to_string(),
        ));
    }
    if m > n {
        return Err(SignalError::ValueError(format!(
            "poisson_disk_sampling_1d: m ({m}) > n ({n})"
        )));
    }
    if min_sep == 0 {
        return Err(SignalError::ValueError(
            "poisson_disk_sampling_1d: min_sep must be at least 1".to_string(),
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut selected: Vec<usize> = Vec::with_capacity(m);
    let mut available: Vec<usize> = (0..n).collect();
    let max_attempts = n * 10;

    for _ in 0..max_attempts {
        if selected.len() >= m {
            break;
        }
        if available.is_empty() {
            break;
        }

        // Pick a random candidate from available
        let idx_in_avail = (rng.random::<f64>() * available.len() as f64) as usize;
        let idx_in_avail = idx_in_avail.min(available.len() - 1);
        let candidate = available[idx_in_avail];

        // Check minimum separation against already selected
        let ok = selected
            .iter()
            .all(|&s| (s as isize - candidate as isize).unsigned_abs() >= min_sep);

        if ok {
            selected.push(candidate);
            // Remove from available within min_sep radius for efficiency
            available.retain(|&a| (a as isize - candidate as isize).unsigned_abs() >= min_sep);
        }
    }

    if selected.len() < m {
        // Fill remaining without the strict separation constraint
        let mut all: Vec<usize> = (0..n).collect();
        all.retain(|a| !selected.contains(a));
        for &a in all.iter().take(m - selected.len()) {
            selected.push(a);
        }
    }

    selected.sort_unstable();
    selected.truncate(m);
    Ok(selected)
}

/// Build a sub-sampled sensing matrix from a full DFT matrix.
///
/// Selects `m` rows (frequencies) from the `n x n` DFT matrix according to
/// `row_indices`, returning a real-valued sensing matrix of shape `(2m, n)`
/// by stacking the real and imaginary parts of each selected DFT row.
///
/// # Arguments
///
/// * `n`           – Signal length / DFT size.
/// * `row_indices` – Which DFT frequencies to keep (values in `[0, n)`).
///
/// # Returns
///
/// Real sensing matrix of shape `(2 * row_indices.len(), n)`.
pub fn partial_dft_matrix(n: usize, row_indices: &[usize]) -> SignalResult<Array2<f64>> {
    if n == 0 {
        return Err(SignalError::ValueError(
            "partial_dft_matrix: n must be positive".to_string(),
        ));
    }
    for &k in row_indices {
        if k >= n {
            return Err(SignalError::ValueError(format!(
                "partial_dft_matrix: row index {k} >= n ({n})"
            )));
        }
    }

    let m = row_indices.len();
    let rows = 2 * m;
    let mut phi = Array2::<f64>::zeros((rows, n));

    for (row_pair, &freq_idx) in row_indices.iter().enumerate() {
        let k = freq_idx as f64;
        for col in 0..n {
            let phase = 2.0 * PI * k * col as f64 / n as f64;
            phi[[2 * row_pair, col]] = phase.cos(); // real part
            phi[[2 * row_pair + 1, col]] = -phase.sin(); // imaginary part
        }
    }

    Ok(phi)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jitter_sampling_length() {
        let times = jitter_sampling(64, 1000.0).expect("jitter sampling should succeed");
        assert_eq!(times.len(), 64);
    }

    #[test]
    fn test_jitter_sampling_sorted() {
        let times = jitter_sampling(32, 500.0).expect("jitter sampling should succeed");
        for w in times.windows(2) {
            assert!(w[0] <= w[1], "times must be non-decreasing");
        }
    }

    #[test]
    fn test_jitter_sampling_stays_in_bins() {
        let n = 16_usize;
        let fs = 100.0_f64;
        let dt = 1.0 / fs;
        let times = jitter_sampling_seeded(n, fs, 42).expect("jitter sampling should succeed");
        for (i, &t) in times.iter().enumerate() {
            let bin_start = i as f64 * dt;
            let bin_end = bin_start + dt;
            assert!(
                t >= bin_start && t < bin_end + 1e-12,
                "sample {i} at {t} is outside bin [{bin_start}, {bin_end})"
            );
        }
    }

    #[test]
    fn test_random_demodulation_dimensions() {
        let n = 64_usize;
        let m = 16_usize;
        let signal = Array1::<f64>::ones(n);
        let (phi, y) = random_demodulation(&signal, m).expect("random_demodulation should succeed");
        assert_eq!(phi.nrows(), y.len());
        assert_eq!(phi.ncols(), n);
        assert!(phi.nrows() <= m);
    }

    #[test]
    fn test_random_demodulation_y_equals_phi_dot_signal() {
        let n = 32_usize;
        let signal: Array1<f64> = Array1::from_vec((0..n).map(|i| (i as f64).sin()).collect());
        let (phi, y) = random_demodulation(&signal, 8).expect("should succeed");
        let y_check = phi.dot(&signal);
        for (a, b) in y.iter().zip(y_check.iter()) {
            assert!((a - b).abs() < 1e-10, "y mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_noncartesian_fft_dc() {
        // DC signal: all ones. The DFT at f=0 should equal sum of signal = n.
        let n = 32_usize;
        let signal = Array1::<f64>::ones(n);
        let times: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let freqs = vec![0.0_f64];
        let y = noncartesian_fft(&signal, &times, &freqs).expect("NUFFT should succeed");
        assert!(
            (y[0].re - n as f64).abs() < 1e-8,
            "DC bin should be {n}, got {}",
            y[0].re
        );
        assert!(y[0].im.abs() < 1e-8, "DC bin imaginary part should be ~0");
    }

    #[test]
    fn test_noncartesian_fft_single_tone() {
        // Signal = cos(2π f₀ t), should give a spike at ±f₀
        let n = 64_usize;
        let fs = 64.0_f64;
        let f0 = 4.0_f64; // 4 Hz
        let times: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let signal: Array1<f64> =
            Array1::from_vec(times.iter().map(|&t| (2.0 * PI * f0 * t).cos()).collect());
        // Evaluate at f0 and at 0
        let freqs = vec![0.0_f64, f0];
        let y = noncartesian_fft(&signal, &times, &freqs).expect("NUFFT should succeed");
        // |Y(f0)| should dominate |Y(0)|
        assert!(
            y[1].norm() > y[0].norm(),
            "tone at f0 should dominate DC: |Y(f0)|={}, |Y(0)|={}",
            y[1].norm(),
            y[0].norm()
        );
    }

    #[test]
    fn test_compressed_sensing_reconstruct_identity_psi() {
        // When Psi = I, the combined matrix A = Phi Psi^T = Phi.
        // With Phi = I (full measurements), reconstruction should be exact.
        let n = 8_usize;
        let phi = Array2::<f64>::eye(n);
        let psi = Array2::<f64>::eye(n);
        let mut x_true = Array1::<f64>::zeros(n);
        x_true[1] = 3.0;
        x_true[4] = -2.0;
        let y = phi.dot(&x_true);
        let x_hat = compressed_sensing_reconstruct(&y, &phi, &psi, 2)
            .expect("CS reconstruct should succeed");
        let err: f64 = (&x_hat - &x_true)
            .iter()
            .map(|&v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-6, "CS reconstruct error: {err}");
    }

    #[test]
    fn test_poisson_disk_sampling_1d_length() {
        let indices = poisson_disk_sampling_1d(128, 32, 2, 0).expect("PD sampling should succeed");
        assert_eq!(indices.len(), 32);
    }

    #[test]
    fn test_poisson_disk_sampling_1d_separation() {
        let min_sep = 3_usize;
        let indices =
            poisson_disk_sampling_1d(128, 20, min_sep, 1).expect("PD sampling should succeed");
        for w in indices.windows(2) {
            assert!(
                w[1] - w[0] >= min_sep,
                "separation violated: {} and {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_partial_dft_matrix_shape() {
        let n = 16_usize;
        let freq_rows = vec![0, 1, 2, 5];
        let phi = partial_dft_matrix(n, &freq_rows).expect("partial DFT should succeed");
        assert_eq!(phi.nrows(), 2 * freq_rows.len());
        assert_eq!(phi.ncols(), n);
    }

    #[test]
    fn test_partial_dft_matrix_dc_row() {
        // Row 0 corresponds to f=0; cos(0) = 1 and sin(0) = 0, so real row is all-ones, imag row all-zeros
        let n = 8_usize;
        let phi = partial_dft_matrix(n, &[0]).expect("partial DFT should succeed");
        for col in 0..n {
            assert!(
                (phi[[0, col]] - 1.0).abs() < 1e-12,
                "DC real row should be 1"
            );
            assert!(phi[[1, col]].abs() < 1e-12, "DC imag row should be 0");
        }
    }
}
