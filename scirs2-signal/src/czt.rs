// Chirp Z-Transform (CZT)
//
// This module provides functions for computing the Chirp Z-Transform (CZT),
// which is a generalization of the Discrete Fourier Transform (DFT) that
// allows evaluation of the Z-transform on arbitrary contours in the complex plane.
//
// The CZT is particularly useful for analyzing frequency components with
// non-uniform spacing or for "zooming in" on specific frequency ranges.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{Float, NumCast};

use std::f64::consts::PI;
use std::fmt::Debug;

/// Calculate the points at which the chirp z-transform is computed
///
/// # Arguments
///
/// * `m` - Number of points to evaluate
/// * `w` - Step size between points on the contour (default: unit circle w = exp(-j*2π/m))
/// * `a` - Starting point on the contour (default: a = 1)
///
/// # Returns
///
/// * A vector of points in the complex plane
///
/// # Examples
///
/// ```
/// use scirs2_signal::czt::czt_points;
///
/// // Generate 10 points on the unit circle
/// let points = czt_points(10, None, None).expect("Operation failed");
/// assert_eq!(points.len(), 10);
/// ```
pub fn czt_points(
    m: usize,
    w: Option<Complex64>,
    a: Option<Complex64>,
) -> SignalResult<Vec<Complex64>> {
    // Default values
    let a_val = a.unwrap_or(Complex64::new(1.0, 0.0));
    let w_val = w.unwrap_or_else(|| {
        // Default to unit circle: w = exp(-j*2π/m)
        let arg = -2.0 * PI / m as f64;
        Complex64::new(arg.cos(), arg.sin())
    });

    // Create the points
    let mut points = Vec::with_capacity(m);
    let mut current = a_val;

    for _ in 0..m {
        points.push(current);
        current *= w_val;
    }

    Ok(points)
}

/// Compute the Chirp Z-Transform
///
/// Evaluates `X(k) = sum_{n=0}^{N-1} x[n] * A^{-n} * W^{nk}` for `k = 0..M-1`.
///
/// When called with default parameters (`m = N`, `w = exp(-j*2π/N)`, `a = 1`),
/// the result is identical to the standard DFT (FFT output).
///
/// # Arguments
///
/// * `x` - Input signal
/// * `m` - Number of output points (default: same as input length)
/// * `w` - Step size between points on the contour (default: unit circle w = exp(-j*2π/m))
/// * `a` - Starting point on the contour (default: a = 1)
/// * `axis` - Axis along which to compute the transform (only -1 or 0 supported)
///
/// # Returns
///
/// * The Chirp Z-Transform of the input signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::czt::czt;
///
/// // Generate a simple signal
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
///
/// // Compute the CZT (equivalent to the DFT in this case)
/// let result = czt(&signal, None, None, None, None).expect("Operation failed");
/// assert_eq!(result.len(), 4);
/// ```
///
/// Zoom in on a specific frequency range:
///
/// ```
/// use scirs2_signal::czt::czt;
/// use scirs2_core::numeric::Complex64;
///
/// // Generate a simple signal
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
///
/// // w = exp(-j*π/8) -> 1/8 of a full circle per step
/// let arg = -std::f64::consts::PI / 8.0;
/// let w = Complex64::new(arg.cos(), arg.sin());
/// let result = czt(&signal, Some(16), Some(w), None, None).expect("Operation failed");
/// assert_eq!(result.len(), 16);
/// ```
pub fn czt<T>(
    x: &[T],
    m: Option<usize>,
    w: Option<Complex64>,
    a: Option<Complex64>,
    axis: Option<isize>,
) -> SignalResult<Vec<Complex64>>
where
    T: Float + NumCast + Debug,
{
    // Check input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Default values
    let n = x.len();
    let m_val = m.unwrap_or(n);
    let a_val = a.unwrap_or(Complex64::new(1.0, 0.0));
    let w_val = w.unwrap_or_else(|| {
        // Default to unit circle: w = exp(-j*2π/m)
        let arg = -2.0 * PI / m_val as f64;
        Complex64::new(arg.cos(), arg.sin())
    });

    // Convert input to complex
    let x_complex: Vec<Complex64> = x
        .iter()
        .map(|&val| {
            let val_f64 = NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })?;
            Ok(Complex64::new(val_f64, 0.0))
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Ignore axis parameter for now (only 1D transform is implemented)
    if let Some(ax) = axis {
        if ax != -1 && ax != 0 {
            return Err(SignalError::ValueError(
                "Only axis=-1 or axis=0 is supported".to_string(),
            ));
        }
    }

    // Compute the CZT using Bluestein's algorithm
    czt_bluestein(&x_complex, m_val, w_val, a_val)
}

/// Compute the Chirp Z-Transform using Bluestein's algorithm.
///
/// Evaluates X(k) = sum_{n=0}^{N-1} x[n] * a^{-n} * w^{nk}  for k=0..M-1
///
/// Using the Bluestein identity `nk = n²/2 - (k-n)²/2 + k²/2`:
///
/// ```text
/// X(k) = w^{k²/2} * sum_{n=0}^{N-1} [x[n] * a^{-n} * w^{n²/2}] * w^{-(k-n)²/2}
/// ```
///
/// The inner sum is a convolution that can be computed with FFTs.
fn czt_bluestein(
    x: &[Complex64],
    m: usize,
    w: Complex64,
    a: Complex64,
) -> SignalResult<Vec<Complex64>> {
    let n = x.len();

    // Length for the FFT-based convolution: next power of 2 >= (n + m - 1)
    let conv_len = next_power_of_two(n + m - 1);

    // Extract |w| and angle(w) to handle complex w with |w| != 1 correctly.
    let w_angle = w.im.atan2(w.re);
    let w_mag = (w.re * w.re + w.im * w.im).sqrt();

    // chirp_w(n_val) = w^{n_val^2 / 2}
    //   magnitude:  w_mag^{n_val^2 / 2}
    //   phase:      exp(j * w_angle * n_val^2 / 2)
    let chirp_w = |n_val: i64| -> Complex64 {
        let sq = n_val * n_val;
        let mag = w_mag.powf(sq as f64 / 2.0);
        let phase = w_angle * sq as f64 / 2.0;
        Complex64::new(mag * phase.cos(), mag * phase.sin())
    };

    // Build yn[n] = x[n] * a^{-n} * w^{n²/2},  n = 0..N-1, zero-padded to conv_len
    let mut yn: Vec<Complex64> = Vec::with_capacity(conv_len);
    let a_inv = Complex64::new(1.0, 0.0) / a;
    let mut a_pow = Complex64::new(1.0, 0.0); // tracks a^{-n}
    for ni in 0..n {
        let chirp_n = chirp_w(ni as i64);
        yn.push(x[ni] * a_pow * chirp_n);
        a_pow *= a_inv;
    }
    while yn.len() < conv_len {
        yn.push(Complex64::new(0.0, 0.0));
    }

    // Build hn: the filter kernel h[k] = w^{-k²/2}
    //   h[k]              for k = 0..M-1  (positive lags)
    //   h[conv_len - k]   for k = 1..N-1  (negative lags, wrapped)
    let mut hn: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); conv_len];
    for ki in 0..m {
        let c = chirp_w(ki as i64);
        hn[ki] = Complex64::new(c.re, -c.im); // conjugate = w^{-k^2/2}
    }
    for ni in 1..n {
        let c = chirp_w(ni as i64);
        hn[conv_len - ni] = Complex64::new(c.re, -c.im);
    }

    // FFT-based convolution
    let yn_fft = fft_complex(&yn)?;
    let hn_fft = fft_complex(&hn)?;

    let mut product: Vec<Complex64> = yn_fft
        .iter()
        .zip(hn_fft.iter())
        .map(|(&y, &h)| y * h)
        .collect();

    ifft_in_place(&mut product)?;

    // Extract M points, multiply by post-chirp w^{k²/2}
    let result: Vec<Complex64> = (0..m).map(|ki| product[ki] * chirp_w(ki as i64)).collect();

    Ok(result)
}

/// Find the next power of 2 greater than or equal to n.
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Compute FFT of a complex sequence using in-place Cooley-Tukey.
fn fft_complex(x: &[Complex64]) -> SignalResult<Vec<Complex64>> {
    if x.is_empty() {
        return Ok(Vec::new());
    }
    let mut buf = x.to_vec();
    fft_inplace(&mut buf, false)?;
    Ok(buf)
}

/// In-place Cooley-Tukey radix-2 DIT FFT/IFFT. Length must be a power of 2.
fn fft_inplace(buf: &mut Vec<Complex64>, inverse: bool) -> SignalResult<()> {
    let n = buf.len();
    if n <= 1 {
        return Ok(());
    }
    if n & (n - 1) != 0 {
        return Err(SignalError::ValueError(format!(
            "FFT length must be a power of 2, got {}",
            n
        )));
    }

    // Bit-reversal permutation
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if j > i {
            buf.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly
    let mut len = 2_usize;
    while len <= n {
        let half = len / 2;
        let angle = if inverse {
            2.0 * PI / len as f64
        } else {
            -2.0 * PI / len as f64
        };
        let wlen = Complex64::new(angle.cos(), angle.sin());

        let mut i = 0;
        while i < n {
            let mut w = Complex64::new(1.0, 0.0);
            for j in 0..half {
                let u = buf[i + j];
                let v = buf[i + j + half] * w;
                buf[i + j] = u + v;
                buf[i + j + half] = u - v;
                w *= wlen;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for c in buf.iter_mut() {
            *c = Complex64::new(c.re * scale, c.im * scale);
        }
    }

    Ok(())
}

/// Reverse the bits of `x` using `bits` significant bits.
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// In-place IFFT.
fn ifft_in_place(buf: &mut Vec<Complex64>) -> SignalResult<()> {
    fft_inplace(buf, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_czt_points() {
        // Generate 4 points on the unit circle
        let points = czt_points(4, None, None).expect("Operation failed");

        // Check length
        assert_eq!(points.len(), 4);

        // First point should be 1+0j
        assert_relative_eq!(points[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(points[0].im, 0.0, epsilon = 1e-10);

        // Check that points are evenly spaced on unit circle (W = exp(-j*2π/4))
        points.iter().enumerate().take(4).for_each(|(i, point)| {
            let angle = -2.0 * PI * i as f64 / 4.0;
            let expected = Complex64::new(angle.cos(), angle.sin());

            assert_relative_eq!(point.re, expected.re, epsilon = 1e-10);
            assert_relative_eq!(point.im, expected.im, epsilon = 1e-10);
        });
    }

    /// Helper: compute an N-point DFT directly (O(N²)) for reference comparison.
    fn dft_direct(x: &[f64]) -> Vec<Complex64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut sum = Complex64::new(0.0, 0.0);
                for (ni, &xn) in x.iter().enumerate() {
                    let angle = -2.0 * PI * k as f64 * ni as f64 / n as f64;
                    sum += Complex64::new(xn, 0.0) * Complex64::new(angle.cos(), angle.sin());
                }
                sum
            })
            .collect()
    }

    #[test]
    fn test_czt_dft_equivalence() {
        // CZT with default parameters (w = exp(-j*2π/N), a = 1, m = N)
        // must be identical to the standard DFT within floating-point tolerance.
        let signals: &[&[f64]] = &[
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 0.0, -1.0, 0.0],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ];

        for signal in signals {
            let czt_result = czt(*signal, None, None, None, None).expect("CZT failed");
            let dft_result = dft_direct(signal);

            assert_eq!(czt_result.len(), dft_result.len());

            for (k, (c, d)) in czt_result.iter().zip(dft_result.iter()).enumerate() {
                let err_re = (c.re - d.re).abs();
                let err_im = (c.im - d.im).abs();
                assert!(
                    err_re < 1e-9,
                    "signal index {}, bin {}: CZT re={} vs DFT re={} (err={})",
                    signal.len(),
                    k,
                    c.re,
                    d.re,
                    err_re
                );
                assert!(
                    err_im < 1e-9,
                    "signal index {}, bin {}: CZT im={} vs DFT im={} (err={})",
                    signal.len(),
                    k,
                    c.im,
                    d.im,
                    err_im
                );
            }
        }
    }

    #[test]
    fn test_czt_dft_properties() {
        // Verify DFT properties when CZT uses default parameters.
        let signal = vec![1.0_f64, 2.0, 3.0, 4.0];
        let czt_result = czt(&signal, None, None, None, None).expect("CZT failed");

        // 1. Length
        assert_eq!(czt_result.len(), 4);

        // 2. DC component should be purely real and equal to sum of signal
        let dc_sum: f64 = signal.iter().sum();
        assert_relative_eq!(czt_result[0].re, dc_sum, epsilon = 1e-9);
        assert_relative_eq!(czt_result[0].im, 0.0, epsilon = 1e-9);

        // 3. Nyquist bin (index N/2) should be real for real input
        assert_relative_eq!(czt_result[2].im, 0.0, epsilon = 1e-9);

        // 4. Conjugate symmetry: X[N-k] = conj(X[k])
        assert_relative_eq!(czt_result[1].re, czt_result[3].re, epsilon = 1e-9);
        assert_relative_eq!(czt_result[1].im, -czt_result[3].im, epsilon = 1e-9);

        // 5. Linearity: CZT(2x) = 2 * CZT(x)
        let signal2: Vec<f64> = signal.iter().map(|&v| 2.0 * v).collect();
        let czt_result2 = czt(&signal2, None, None, None, None).expect("CZT failed");
        for k in 0..4 {
            assert_relative_eq!(czt_result2[k].re, 2.0 * czt_result[k].re, epsilon = 1e-9);
            assert_relative_eq!(czt_result2[k].im, 2.0 * czt_result[k].im, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_czt_zoom() {
        // Test CZT for "zooming in" on a specific frequency range
        let signal = vec![1.0, 0.0, 1.0, 0.0]; // Simple 2Hz signal (when sampled at 8Hz)

        // Compute 8-point CZT that zooms in on the first quarter of the spectrum
        // This means w = exp(-j*π/16)
        let arg = -PI / 16.0;
        let w = Complex64::new(arg.cos(), arg.sin());

        let czt_result = czt(&signal, Some(8), Some(w), None, None).expect("Operation failed");

        // Check length
        assert_eq!(czt_result.len(), 8);

        // Find max magnitude bin
        let mut max_idx = 0;
        let mut max_val = 0.0;
        for (i, val) in czt_result.iter().enumerate() {
            let mag = val.norm();
            if mag > max_val {
                max_val = mag;
                max_idx = i;
            }
        }

        // Check that we have significant energy somewhere in the array
        assert!(max_val > 1.0);

        println!(
            "Max energy found at bin {} with magnitude {}",
            max_idx, max_val
        );
    }

    #[test]
    fn test_czt_non_power_of_two_length() {
        // CZT can handle non-power-of-2 input lengths unlike standard FFT
        let signal: Vec<f64> = (0..7).map(|i| i as f64).collect();
        let czt_result = czt(&signal, None, None, None, None).expect("CZT failed");
        let dft_result = dft_direct(&signal);

        assert_eq!(czt_result.len(), 7);

        for (k, (c, d)) in czt_result.iter().zip(dft_result.iter()).enumerate() {
            let err = (c - d).norm();
            assert!(
                err < 1e-8,
                "bin {}: CZT=({},{}) vs DFT=({},{}) err={}",
                k,
                c.re,
                c.im,
                d.re,
                d.im,
                err
            );
        }
    }

    #[test]
    fn test_czt_m_greater_than_n() {
        // CZT can produce more output points than input length
        let signal = vec![1.0_f64, 0.0, 0.0, 0.0];
        // With a = 1 and w = exp(-j*2π/8), 8 output points on the unit circle
        let w_arg = -2.0 * PI / 8.0;
        let w = Complex64::new(w_arg.cos(), w_arg.sin());
        let czt_result = czt(&signal, Some(8), Some(w), None, None).expect("CZT failed");

        // For x = [1, 0, 0, 0], all DFT bins have value 1.0+0j
        assert_eq!(czt_result.len(), 8);
        for (k, c) in czt_result.iter().enumerate() {
            let err = (c.norm() - 1.0).abs();
            assert!(
                err < 1e-9,
                "bin {} magnitude should be 1.0, got {}",
                k,
                c.norm()
            );
        }
    }
}
