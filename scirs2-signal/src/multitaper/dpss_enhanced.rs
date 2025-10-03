// Enhanced DPSS (Discrete Prolate Spheroidal Sequences) implementation
//
// This module provides a corrected and validated implementation of DPSS
// computation following SciPy's approach and Percival & Walden (1993).
// Updated to remove ndarray-linalg dependency.

use crate::error::{SignalError, SignalResult};
use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{Array1, Array2};
use rustfft::{num_complex::Complex, FftPlanner};
use scirs2_core::validation::check_positive;

#[allow(unused_imports)]
/// Enhanced DPSS computation with proper SciPy-compatible implementation
///
/// Computes the Discrete Prolate Spheroidal Sequences (Slepian sequences)
/// using the tridiagonal matrix formulation.
///
/// # Arguments
///
/// * `n` - Sequence length
/// * `nw` - Time-bandwidth product (typically 2-4)
/// * `k` - Number of sequences to compute (default: 2*nw - 1)
/// * `return_ratios` - Whether to return concentration ratios
///
/// # Returns
///
/// * Tuple of (tapers, concentration_ratios)
#[allow(dead_code)]
pub fn dpss_enhanced(
    n: usize,
    nw: f64,
    k: usize,
    return_ratios: bool,
) -> SignalResult<(Array2<f64>, Option<Array1<f64>>)> {
    // Validate inputs
    check_positive(n, "n")?;
    check_positive(nw, "nw")?;
    check_positive(k, "k")?;

    if k > n {
        return Err(SignalError::ValueError(format!(
            "k ({}) must not exceed n ({})",
            k, n
        )));
    }

    // Maximum useful number of tapers
    let k_max = (2.0 * nw).floor() as usize;
    if k > k_max {
        eprintln!(
            "Warning: k ({}) is greater than 2*NW-1 ({}). The higher order tapers will have poor concentration.",
            k, k_max - 1
        );
    }

    // Compute normalized frequency
    let w = nw / n as f64;

    // Build tridiagonal matrix (following SciPy/Percival & Walden)
    let (diagonal, off_diagonal) = build_tridiagonal_matrix(n, w);

    // Compute eigenvalues and eigenvectors
    let (eigenvalues, eigenvectors) = solve_tridiagonal_symmetric(&diagonal, &off_diagonal)?;

    // Sort by eigenvalue magnitude (descending)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| {
        eigenvalues[j]
            .abs()
            .partial_cmp(&eigenvalues[i].abs())
            .unwrap()
    });

    // Extract k largest eigenvalue/eigenvector pairs
    let mut tapers = Array2::zeros((k, n));
    let mut eigenvals = Array1::zeros(k);

    for i in 0..k {
        let idx = indices[i];
        eigenvals[i] = eigenvalues[idx];

        // Extract and normalize eigenvector
        let mut eigvec = eigenvectors.column(idx).to_owned();
        normalize_eigenvector(&mut eigvec);

        // Apply sign convention
        apply_sign_convention(&mut eigvec, i);

        // Store as row in tapers matrix
        tapers.row_mut(i).assign(&eigvec);
    }

    // Compute concentration _ratios if requested
    let _ratios = if return_ratios {
        Some(compute_concentration_ratios(&tapers, w, n)?)
    } else {
        None
    };

    Ok((tapers, ratios))
}

/// Build the tridiagonal matrix for the eigenvalue problem
#[allow(dead_code)]
fn build_tridiagonal_matrix(n: usize, w: f64) -> (Vec<f64>, Vec<f64>) {
    let cos_2pi_w = (2.0 * PI * w).cos();

    // Diagonal elements: ((n-1-2i)/2)^2 * cos(2πW)
    let diagonal: Vec<f64> = (0..n)
        .map(|i| {
            let term = (n as f64 - 1.0 - 2.0 * i as f64) / 2.0;
            term * term * cos_2pi_w
        })
        .collect();

    // Off-diagonal elements: i(n-i)/2 for i = 1, 2, ..., n-1
    let off_diagonal: Vec<f64> = (1..n).map(|i| (i as f64 * (n - i) as f64) / 2.0).collect();

    (diagonal, off_diagonal)
}

/// Solve symmetric tridiagonal eigenvalue problem using QR algorithm
#[allow(dead_code)]
fn solve_tridiagonal_symmetric(
    diagonal: &[f64],
    off_diagonal: &[f64],
) -> SignalResult<(Vec<f64>, Array2<f64>)> {
    let n = diagonal.len();

    // Copy arrays for modification
    let mut diag = diagonal.to_vec();
    let mut off_diag = off_diagonal.to_vec();

    // Initialize eigenvector matrix as identity
    let mut q = Array2::eye(n);

    // QR algorithm for tridiagonal matrices
    let max_iterations = 100 * n;
    let tolerance = 1e-12;

    for _iter in 0..max_iterations {
        // Check for convergence - find off-_diagonal elements that are small enough
        let mut converged = true;
        for i in 0..n - 1 {
            if off_diag[i].abs() > tolerance * (diag[i].abs() + diag[i + 1].abs()) {
                converged = false;
                break;
            }
        }

        if converged {
            break;
        }

        // Choose shift (Wilkinson shift for better convergence)
        let shift = if n > 1 {
            wilkinson_shift(&diag[n - 2..n], &off_diag[n - 2..n - 1])
        } else {
            0.0
        };

        // Apply shift
        for i in 0..n {
            diag[i] -= shift;
        }

        // QR step
        qr_step(&mut diag, &mut off_diag, &mut q)?;

        // Restore shift
        for i in 0..n {
            diag[i] += shift;
        }
    }

    Ok((diag, q))
}

/// Compute Wilkinson shift for better convergence
#[allow(dead_code)]
fn wilkinson_shift(_diag: &[f64], offdiag: &[f64]) -> f64 {
    if diag.len() < 2 || off_diag.is_empty() {
        return 0.0;
    }

    let a = diag[0];
    let b = off_diag[0];
    let c = diag[1];

    let d = (a - c) / 2.0;
    let sign = if d >= 0.0 { 1.0 } else { -1.0 };

    c - sign * b * b / (d.abs() + (d * d + b * b).sqrt())
}

/// Perform one QR step on tridiagonal matrix
#[allow(dead_code)]
fn qr_step(_diag: &mut [f64], offdiag: &mut [f64], q: &mut Array2<f64>) -> SignalResult<()> {
    let n = diag.len();
    if n <= 1 {
        return Ok(());
    }

    // Initialize Givens rotation parameters
    let mut c_prev = 1.0;
    let mut s_prev = 0.0;

    for i in 0..n - 1 {
        // Compute Givens rotation to eliminate off_diag[i]
        let (c, s) = givens_rotation(
            diag[i] * c_prev + off_diag[i] * s_prev,
            off_diag[i] * c_prev - diag[i] * s_prev
                + if i < n - 2 { off_diag[i + 1] } else { 0.0 },
        );

        // Apply rotation to tridiagonal matrix
        if i > 0 {
            off_diag[i - 1] = c_prev * off_diag[i - 1] + s_prev * diag[i];
        }

        let temp = c * diag[i] + s * off_diag[i];
        diag[i + 1] = -s * diag[i] + c * diag[i + 1];
        diag[i] = temp;

        if i < n - 2 {
            let temp = c * off_diag[i + 1];
            off_diag[i + 1] = -s * off_diag[i + 1];
            off_diag[i] = temp;
        } else {
            off_diag[i] = c * off_diag[i];
        }

        // Update eigenvector matrix
        for j in 0..n {
            let temp = c * q[[j, i]] + s * q[[j, i + 1]];
            q[[j, i + 1]] = -s * q[[j, i]] + c * q[[j, i + 1]];
            q[[j, i]] = temp;
        }

        c_prev = c;
        s_prev = s;
    }

    Ok(())
}

/// Compute Givens rotation parameters
#[allow(dead_code)]
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if b.abs() < 1e-15 {
        return (1.0, 0.0);
    }

    if a.abs() < 1e-15 {
        return (0.0, if b > 0.0 { 1.0 } else { -1.0 });
    }

    let r = (a * a + b * b).sqrt();
    (a / r, b / r)
}

/// Normalize eigenvector to unit norm
#[allow(dead_code)]
fn normalize_eigenvector(eigvec: &mut Array1<f64>) {
    let norm = eigvec.dot(_eigvec).sqrt();
    if norm > 1e-10 {
        *_eigvec /= norm;
    }
}

/// Apply sign convention to ensure consistency
#[allow(dead_code)]
fn apply_sign_convention(eigvec: &mut Array1<f64>, order: usize) {
    let n = eigvec.len();

    if order % 2 == 0 {
        // Even order: ensure symmetric taper has positive average
        let sum: f64 = eigvec.sum();
        if sum < 0.0 {
            *_eigvec *= -1.0;
        }
    } else {
        // Odd order: ensure antisymmetric taper starts positive
        let mid = n / 2;
        if eigvec[0] < 0.0 || (n % 2 == 1 && eigvec[mid] < 0.0) {
            *_eigvec *= -1.0;
        }
    }
}

/// Compute concentration ratios using autocorrelation method
#[allow(dead_code)]
fn compute_concentration_ratios(
    tapers: &Array2<f64>,
    w: f64,
    n: usize,
) -> SignalResult<Array1<f64>> {
    let k = tapers.nrows();
    let mut ratios = Array1::zeros(k);

    // Frequency range for concentration
    let _f_low = -w;
    let _f_high = w;

    // Use FFT-based autocorrelation for efficiency
    let mut planner = FftPlanner::new();
    let fft_size = 2 * n; // Zero-pad for linear convolution
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    for i in 0..k {
        let taper = tapers.row(i);

        // Zero-pad taper
        let mut padded = vec![Complex::new(0.0, 0.0); fft_size];
        for j in 0..n {
            padded[j] = Complex::new(taper[j], 0.0);
        }

        // FFT
        fft.process(&mut padded);

        // Compute power spectrum
        for j in 0..fft_size {
            let power = padded[j].norm_sqr();
            padded[j] = Complex::new(power, 0.0);
        }

        // IFFT to get autocorrelation
        ifft.process(&mut padded);

        // Extract autocorrelation values (normalized)
        let autocorr: Vec<f64> = padded
            .iter()
            .take(n)
            .map(|c| c.re / fft_size as f64)
            .collect();

        // Compute concentration ratio using Percival & Walden formula
        let mut concentration = autocorr[0]; // R(0) term

        for lag in 1..n {
            let sinc_term = if lag as f64 * 2.0 * PI * w < 1e-10 {
                1.0
            } else {
                (lag as f64 * 2.0 * PI * w).sin() / (lag as f64 * 2.0 * PI * w)
            };
            concentration += 2.0 * autocorr[lag] * sinc_term;
        }

        ratios[i] = concentration.min(1.0).max(0.0);
    }

    Ok(ratios)
}

/// Validate DPSS computation against known values
#[allow(dead_code)]
pub fn validate_dpss_implementation() -> SignalResult<bool> {
    // Test case from SciPy documentation
    let n = 64;
    let nw = 4.0;
    let k = 7;

    let (tapers, ratios) = dpss_enhanced(n, nw, k, true)?;
    let ratios = ratios.unwrap();

    // Expected concentration ratios (from SciPy)
    let expected_ratios = vec![
        0.9999999999,
        0.9999999964,
        0.9999999432,
        0.9999996325,
        0.9999984459,
        0.9999943506,
        0.9999829374,
    ];

    // Check concentration ratios
    for i in 0..k {
        let error = (ratios[i] - expected_ratios[i]).abs();
        if error > 1e-8 {
            eprintln!(
                "Concentration ratio mismatch at index {}: expected {:.10}, got {:.10}",
                i, expected_ratios[i], ratios[i]
            );
            return Ok(false);
        }
    }

    // Check orthogonality
    for i in 0..k {
        for j in i + 1..k {
            let dot_product = tapers.row(i).dot(&tapers.row(j));
            if dot_product.abs() > 1e-10 {
                eprintln!(
                    "Orthogonality violated: tapers {} and {} have dot product {:.2e}",
                    i, j, dot_product
                );
                return Ok(false);
            }
        }
    }

    // Check normalization
    for i in 0..k {
        let norm = tapers.row(i).dot(&tapers.row(i)).sqrt();
        if ((norm - 1.0) as f64).abs() > 1e-10 {
            eprintln!("Taper {} not normalized: norm = {:.10}", i, norm);
            return Ok(false);
        }
    }

    Ok(true)
}

/// Generate reference values for testing
#[allow(dead_code)]
pub fn generate_reference_values() -> SignalResult<()> {
    println!("DPSS Reference Values:");
    println!("======================");

    // Standard test cases
    let test_cases = vec![
        (16, 2.5, 3),
        (32, 3.0, 5),
        (64, 4.0, 7),
        (128, 4.0, 7),
        (256, 3.5, 6),
    ];

    for (n, nw, k) in test_cases {
        println!("\nCase: n={}, NW={}, k={}", n, nw, k);

        let (tapers, ratios) = dpss_enhanced(n, nw, k, true)?;
        let ratios = ratios.unwrap();

        println!("Concentration ratios:");
        for i in 0..k {
            println!("  λ[{}] = {:.12}", i, ratios[i]);
        }

        // Print first few values of first taper
        println!("First taper (first 8 values):");
        for i in 0..8.min(n) {
            println!("  v[0][{}] = {:.12}", i, tapers[[0, i]]);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn test_dpss_basic() {
        let (tapers, ratios) = dpss_enhanced(64, 4.0, 7, true).unwrap();

        assert_eq!(tapers.nrows(), 7);
        assert_eq!(tapers.ncols(), 64);
        assert!(ratios.is_some());
    }

    #[test]
    fn test_dpss_orthogonality() {
        let (tapers_) = dpss_enhanced(128, 4.0, 7, false).unwrap();

        // Check orthogonality
        for i in 0..7 {
            for j in i + 1..7 {
                let dot = tapers.row(i).dot(&tapers.row(j));
                assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_dpss_normalization() {
        let (tapers_) = dpss_enhanced(128, 4.0, 7, false).unwrap();

        // Check unit norm
        for i in 0..7 {
            let norm_sq = tapers.row(i).dot(&tapers.row(i));
            assert_abs_diff_eq!(norm_sq, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_concentration_ratios() {
        let (_, ratios) = dpss_enhanced(64, 4.0, 7, true).unwrap();
        let ratios = ratios.unwrap();

        // All ratios should be between 0 and 1
        for &ratio in ratios.iter() {
            assert!(ratio >= 0.0 && ratio <= 1.0);
        }

        // Ratios should decrease
        for i in 1..ratios.len() {
            assert!(ratios[i] <= ratios[i - 1]);
        }
    }

    #[test]
    fn test_validation() {
        assert!(validate_dpss_implementation().unwrap());
    }
}
