//! Frequency Domain Image Processing
//!
//! This module provides image analysis and filtering in the frequency domain using
//! the 2D Discrete Fourier Transform (DFT).  It builds on top of [`scirs2_fft`]
//! for efficient FFT computation and provides high-level image processing utilities:
//!
//! - **FFT2 / IFFT2**: Forward and inverse 2-D DFT with complex output.
//! - **FFT shift**: Centre zero-frequency component for visualisation.
//! - **Frequency-domain filtering**: Generic filter via a user-supplied gain function.
//! - **Built-in filters**: ideal low-pass / high-pass, Butterworth, Gaussian.
//! - **Phase correlation**: Sub-pixel image registration via normalised cross-power spectrum.
//!
//! # References
//!
//! - Gonzalez, R.C. & Woods, R.E. (2018). *Digital Image Processing*, 4th ed.
//! - Oppenheim, A.V. & Schafer, R.W. (2010). *Discrete-Time Signal Processing*, 3rd ed.
//! - Kuglin, C.D. & Hines, D.C. (1975). "The Phase Correlation Image Alignment Method."

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::Complex64;
use scirs2_fft::{fft2, fftfreq, fftshift2, ifft2, ifftshift2};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert a real `Array2<f64>` to a complex `Array2<Complex64>`.
fn real_to_complex(image: &Array2<f64>) -> Array2<Complex64> {
    image.mapv(|v| Complex64::new(v, 0.0))
}

/// Extract the real part of a complex array (clamp imaginary part silently).
fn complex_to_real(spec: &Array2<Complex64>) -> Array2<f64> {
    spec.mapv(|z| z.re)
}

/// Forward 2-D FFT — returns complex spectrum as flat `Array2<Complex64>`.
fn fft2_complex(image: &Array2<f64>) -> NdimageResult<Array2<Complex64>> {
    let complex_input = real_to_complex(image);
    fft2(&complex_input, None, None, None)
        .map_err(|e| NdimageError::ComputationError(format!("fft2: {e}")))
}

/// Inverse 2-D FFT — takes complex spectrum, returns real image.
fn ifft2_real(spec: &Array2<Complex64>) -> NdimageResult<Array2<f64>> {
    let out = ifft2(spec, None, None, None)
        .map_err(|e| NdimageError::ComputationError(format!("ifft2: {e}")))?;
    Ok(complex_to_real(&out))
}

/// Build a pair of normalised frequency grids (u_grid, v_grid) for an image
/// of shape (rows, cols).  Each element is in the range [-0.5, 0.5).
fn freq_grids(rows: usize, cols: usize) -> NdimageResult<(Array2<f64>, Array2<f64>)> {
    let fy = fftfreq(rows, 1.0)
        .map_err(|e| NdimageError::ComputationError(format!("fftfreq rows: {e}")))?;
    let fx = fftfreq(cols, 1.0)
        .map_err(|e| NdimageError::ComputationError(format!("fftfreq cols: {e}")))?;

    let v_grid = Array2::from_shape_fn((rows, cols), |(r, _)| fy[r]);
    let u_grid = Array2::from_shape_fn((rows, cols), |(_, c)| fx[c]);
    Ok((v_grid, u_grid))
}

// ---------------------------------------------------------------------------
// Public API — FFT2 / IFFT2 / fftshift2
// ---------------------------------------------------------------------------

/// Forward 2-D Discrete Fourier Transform.
///
/// Returns a complex array encoded as `Array2<[f64; 2]>` where each element is
/// `[real, imaginary]`.  The zero-frequency component is at index `[0, 0]`
/// (not centred); use [`fft2_shift`] to centre it for display.
///
/// # Arguments
///
/// * `image` - Real-valued input image.
///
/// # Errors
///
/// Returns [`NdimageError`] if the underlying FFT fails.
pub fn fft2_image(image: &Array2<f64>) -> NdimageResult<Array2<[f64; 2]>> {
    let spec = fft2_complex(image)?;
    Ok(spec.mapv(|z| [z.re, z.im]))
}

/// Inverse 2-D Discrete Fourier Transform.
///
/// Takes a complex spectrum encoded as `Array2<[f64; 2]>` (each element is
/// `[real, imaginary]`) and returns the real-valued image.
///
/// # Errors
///
/// Returns [`NdimageError`] if the underlying IFFT fails.
pub fn ifft2_image(spectrum: &Array2<[f64; 2]>) -> NdimageResult<Array2<f64>> {
    let complex = spectrum.mapv(|a| Complex64::new(a[0], a[1]));
    ifft2_real(&complex)
}

/// Shift the zero-frequency component of a 2-D spectrum to the centre.
///
/// Applies the standard "fftshift" rearrangement: quadrants are swapped so
/// that DC is at `[rows/2, cols/2]`.
pub fn fft2_shift(spectrum: &Array2<[f64; 2]>) -> Array2<[f64; 2]> {
    let complex = spectrum.mapv(|a| Complex64::new(a[0], a[1]));
    let shifted = fftshift2(&complex);
    shifted.mapv(|z| [z.re, z.im])
}

/// Inverse of [`fft2_shift`]: moves the zero-frequency component back to `[0, 0]`.
pub fn ifft2_shift(spectrum: &Array2<[f64; 2]>) -> Array2<[f64; 2]> {
    let complex = spectrum.mapv(|a| Complex64::new(a[0], a[1]));
    let shifted = ifftshift2(&complex);
    shifted.mapv(|z| [z.re, z.im])
}

// ---------------------------------------------------------------------------
// Generic frequency-domain filter
// ---------------------------------------------------------------------------

/// Apply a custom frequency-domain filter to an image.
///
/// The filter function receives the normalised frequencies `(u, v)` — both in
/// the range `[-0.5, 0.5)` — and returns a real gain value that multiplies the
/// corresponding DFT coefficient.
///
/// The pipeline is: real image → FFT2 → multiply by gain → IFFT2 → real image.
///
/// # Arguments
///
/// * `image`     - Input image.
/// * `filter_fn` - Closure `|u: f64, v: f64| -> f64` returning the gain at each
///                 frequency.  `u` is the column (horizontal) frequency;
///                 `v` is the row (vertical) frequency.
///
/// # Errors
///
/// Returns [`NdimageError`] on FFT failure.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::frequency_domain::frequency_filter;
/// use scirs2_core::ndarray::Array2;
///
/// let img = Array2::<f64>::zeros((16, 16));
/// // Zero-out all frequencies: output should be zeros
/// let out = frequency_filter(&img, |_u, _v| 0.0).expect("ok");
/// assert!(out.iter().all(|&v| v.abs() < 1e-10));
/// ```
pub fn frequency_filter(
    image: &Array2<f64>,
    filter_fn: impl Fn(f64, f64) -> f64,
) -> NdimageResult<Array2<f64>> {
    let (rows, cols) = (image.nrows(), image.ncols());
    let (v_grid, u_grid) = freq_grids(rows, cols)?;

    let mut spectrum = fft2_complex(image)?;

    for r in 0..rows {
        for c in 0..cols {
            let gain = filter_fn(u_grid[[r, c]], v_grid[[r, c]]);
            spectrum[[r, c]] =
                Complex64::new(spectrum[[r, c]].re * gain, spectrum[[r, c]].im * gain);
        }
    }

    ifft2_real(&spectrum)
}

// ---------------------------------------------------------------------------
// Ideal filters
// ---------------------------------------------------------------------------

/// Ideal (brick-wall) low-pass filter.
///
/// All frequencies with normalised radius ≤ `cutoff` are passed; all others
/// are zeroed.  `cutoff` is in the range (0, 0.5].
///
/// # Errors
///
/// Returns [`NdimageError`] on FFT failure or if `cutoff ≤ 0`.
pub fn lowpass_filter(image: &Array2<f64>, cutoff: f64) -> NdimageResult<Array2<f64>> {
    if cutoff <= 0.0 {
        return Err(NdimageError::InvalidInput("cutoff must be positive".into()));
    }
    let cutoff2 = cutoff * cutoff;
    frequency_filter(
        image,
        move |u, v| {
            if u * u + v * v <= cutoff2 {
                1.0
            } else {
                0.0
            }
        },
    )
}

/// Ideal (brick-wall) high-pass filter.
///
/// All frequencies with normalised radius > `cutoff` are passed; all others
/// are zeroed.
///
/// # Errors
///
/// Returns [`NdimageError`] on FFT failure or if `cutoff ≤ 0`.
pub fn highpass_filter(image: &Array2<f64>, cutoff: f64) -> NdimageResult<Array2<f64>> {
    if cutoff <= 0.0 {
        return Err(NdimageError::InvalidInput("cutoff must be positive".into()));
    }
    let cutoff2 = cutoff * cutoff;
    frequency_filter(
        image,
        move |u, v| {
            if u * u + v * v > cutoff2 {
                1.0
            } else {
                0.0
            }
        },
    )
}

// ---------------------------------------------------------------------------
// Butterworth filter
// ---------------------------------------------------------------------------

/// Butterworth filter (low-pass or high-pass).
///
/// The Butterworth frequency response transitions smoothly between 0 and 1:
///
/// ```text
/// H(r) = 1 / (1 + (r / cutoff)^(2 * order))   [low-pass]
/// H(r) = 1 / (1 + (cutoff / r)^(2 * order))   [high-pass]
/// ```
///
/// where `r = sqrt(u² + v²)` is the normalised frequency radius.
///
/// # Arguments
///
/// * `image`     - Input image.
/// * `cutoff`    - 3 dB cutoff frequency (normalised, typically 0.05–0.45).
/// * `order`     - Filter order (higher = steeper roll-off).
/// * `high_pass` - If `true`, apply high-pass; otherwise low-pass.
///
/// # Errors
///
/// Returns [`NdimageError`] on FFT failure or invalid parameters.
pub fn butterworth_filter(
    image: &Array2<f64>,
    cutoff: f64,
    order: u32,
    high_pass: bool,
) -> NdimageResult<Array2<f64>> {
    if cutoff <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "Butterworth cutoff must be positive".into(),
        ));
    }
    if order == 0 {
        return Err(NdimageError::InvalidInput(
            "Butterworth order must be at least 1".into(),
        ));
    }
    let two_n = 2 * order as i32;
    frequency_filter(image, move |u, v| {
        let r2 = u * u + v * v;
        if r2 == 0.0 {
            if high_pass {
                0.0
            } else {
                1.0
            }
        } else {
            let ratio = if high_pass {
                cutoff * cutoff / r2
            } else {
                r2 / (cutoff * cutoff)
            };
            1.0 / (1.0 + ratio.powi(two_n))
        }
    })
}

// ---------------------------------------------------------------------------
// Gaussian frequency filter
// ---------------------------------------------------------------------------

/// Gaussian frequency-domain filter (low-pass or high-pass).
///
/// The Gaussian gain is:
///
/// ```text
/// H(r) = exp(-r² / (2 * sigma²))   [low-pass]
/// H(r) = 1 - exp(-r² / (2 * sigma²))   [high-pass]
/// ```
///
/// where `r = sqrt(u² + v²)` and `sigma` controls the bandwidth.
///
/// # Arguments
///
/// * `image`     - Input image.
/// * `sigma`     - Standard deviation of the Gaussian in normalised frequency units.
/// * `high_pass` - If `true`, apply high-pass; otherwise low-pass.
///
/// # Errors
///
/// Returns [`NdimageError`] on FFT failure or if `sigma ≤ 0`.
pub fn gaussian_freq_filter(
    image: &Array2<f64>,
    sigma: f64,
    high_pass: bool,
) -> NdimageResult<Array2<f64>> {
    if sigma <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "Gaussian sigma must be positive".into(),
        ));
    }
    let two_sigma2 = 2.0 * sigma * sigma;
    frequency_filter(image, move |u, v| {
        let r2 = u * u + v * v;
        let gauss = (-r2 / two_sigma2).exp();
        if high_pass {
            1.0 - gauss
        } else {
            gauss
        }
    })
}

// ---------------------------------------------------------------------------
// Phase correlation
// ---------------------------------------------------------------------------

/// Estimate the translational shift between two images via phase correlation.
///
/// Computes the normalised cross-power spectrum:
///
/// ```text
/// R(u,v) = F1(u,v) · conj(F2(u,v)) / |F1(u,v) · conj(F2(u,v))|
/// ```
///
/// and finds the location of its IFFT peak, which corresponds to the
/// (row, col) shift between `image1` and `image2`.
///
/// Both images must have the same shape.
///
/// # Arguments
///
/// * `image1` - Reference image.
/// * `image2` - Shifted image.
///
/// # Returns
///
/// `(row_shift, col_shift)` — the estimated integer pixel shift.
///
/// # Errors
///
/// Returns [`NdimageError`] if image shapes differ, are too small, or FFT fails.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::frequency_domain::phase_correlation;
/// use scirs2_core::ndarray::Array2;
///
/// let img1 = Array2::<f64>::zeros((32, 32));
/// let img2 = Array2::<f64>::zeros((32, 32));
/// let (dr, dc) = phase_correlation(&img1, &img2).expect("ok");
/// // Zero shift expected for identical images
/// assert!((dr.abs() + dc.abs()) < 2.0);
/// ```
pub fn phase_correlation(image1: &Array2<f64>, image2: &Array2<f64>) -> NdimageResult<(f64, f64)> {
    let (rows, cols) = (image1.nrows(), image1.ncols());
    if rows != image2.nrows() || cols != image2.ncols() {
        return Err(NdimageError::DimensionError(
            "phase_correlation: images must have the same shape".into(),
        ));
    }
    if rows < 2 || cols < 2 {
        return Err(NdimageError::InvalidInput(
            "phase_correlation: images must be at least 2×2".into(),
        ));
    }

    let f1 = fft2_complex(image1)?;
    let f2 = fft2_complex(image2)?;

    // Normalised cross-power spectrum
    let mut cross_power = Array2::<Complex64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            // conj(F1) * F2 yields a IFFT peak at +d when img2 = img1 shifted by d.
            let product = f1[[r, c]].conj() * f2[[r, c]];
            let mag = product.norm();
            cross_power[[r, c]] = if mag > 1e-12 {
                Complex64::new(product.re / mag, product.im / mag)
            } else {
                Complex64::new(0.0, 0.0)
            };
        }
    }

    // IFFT → peak location = shift
    let corr = ifft2_real(&cross_power)?;

    // Find the peak
    let mut best_r = 0usize;
    let mut best_c = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for r in 0..rows {
        for c in 0..cols {
            let v = corr[[r, c]];
            if v > best_val {
                best_val = v;
                best_r = r;
                best_c = c;
            }
        }
    }

    // Convert to signed shift (wrap around at half image size)
    let row_shift = if best_r > rows / 2 {
        best_r as f64 - rows as f64
    } else {
        best_r as f64
    };
    let col_shift = if best_c > cols / 2 {
        best_c as f64 - cols as f64
    } else {
        best_c as f64
    };

    Ok((row_shift, col_shift))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    // ─── FFT2 round-trip ─────────────────────────────────────────────────────

    #[test]
    fn fft2_ifft2_round_trip() {
        let img = Array2::from_shape_fn((16, 16), |(r, c)| ((r + c) % 4) as f64);
        let spec = fft2_image(&img).expect("fft2 failed");
        let recovered = ifft2_image(&spec).expect("ifft2 failed");
        for ((r, c), &orig) in img.indexed_iter() {
            let rec = recovered[[r, c]];
            assert!(
                (orig - rec).abs() < 1e-8,
                "Round-trip error at ({r},{c}): {orig} vs {rec}"
            );
        }
    }

    #[test]
    fn fft2_zeros_is_zeros() {
        let img = Array2::<f64>::zeros((8, 8));
        let spec = fft2_image(&img).expect("ok");
        for v in spec.iter() {
            assert!(v[0].abs() < 1e-12 && v[1].abs() < 1e-12);
        }
    }

    // ─── fft2_shift ──────────────────────────────────────────────────────────

    #[test]
    fn fft2_shift_size_preserved() {
        let img = Array2::<f64>::from_elem((8, 12), 1.0);
        let spec = fft2_image(&img).expect("ok");
        let shifted = fft2_shift(&spec);
        assert_eq!(shifted.dim(), spec.dim());
    }

    // ─── Frequency filter ────────────────────────────────────────────────────

    #[test]
    fn frequency_filter_zero_gain_zeroes_output() {
        let img = Array2::from_shape_fn((16, 16), |(r, c)| (r * c) as f64);
        let out = frequency_filter(&img, |_u, _v| 0.0).expect("ok");
        for &v in out.iter() {
            assert!(v.abs() < 1e-8, "Expected zero, got {v}");
        }
    }

    #[test]
    fn frequency_filter_unity_gain_preserves_image() {
        let img = Array2::from_shape_fn((16, 16), |(r, c)| ((r + c) % 3) as f64);
        let out = frequency_filter(&img, |_u, _v| 1.0).expect("ok");
        for ((r, c), &orig) in img.indexed_iter() {
            let rec = out[[r, c]];
            assert!((orig - rec).abs() < 1e-8, "Unity filter error at ({r},{c})");
        }
    }

    // ─── Low-pass filter ─────────────────────────────────────────────────────

    #[test]
    fn lowpass_reduces_high_frequencies() {
        // A checkerboard has maximum high-frequency energy
        let img = Array2::from_shape_fn(
            (32, 32),
            |(r, c)| {
                if (r + c) % 2 == 0 {
                    1.0f64
                } else {
                    -1.0f64
                }
            },
        );
        let filtered = lowpass_filter(&img, 0.1).expect("lowpass failed");
        // Energy should be significantly reduced
        let orig_energy: f64 = img.iter().map(|&v| v * v).sum();
        let filt_energy: f64 = filtered.iter().map(|&v| v * v).sum();
        assert!(
            filt_energy < orig_energy * 0.5,
            "Lowpass should reduce energy: orig={orig_energy:.3}, filt={filt_energy:.3}"
        );
    }

    #[test]
    fn lowpass_invalid_cutoff_returns_error() {
        let img = Array2::<f64>::zeros((8, 8));
        assert!(lowpass_filter(&img, 0.0).is_err());
        assert!(lowpass_filter(&img, -0.1).is_err());
    }

    // ─── High-pass filter ────────────────────────────────────────────────────

    #[test]
    fn highpass_removes_dc() {
        // Constant image → all energy is DC
        let img = Array2::<f64>::from_elem((16, 16), 1.0);
        let filtered = highpass_filter(&img, 0.01).expect("highpass ok");
        // After high-pass the output should be near zero
        let max_abs: f64 = filtered.iter().map(|&v| v.abs()).fold(0.0f64, f64::max);
        assert!(
            max_abs < 0.1,
            "High-pass on constant image should give near-zero output, got max {max_abs}"
        );
    }

    #[test]
    fn highpass_invalid_cutoff_returns_error() {
        let img = Array2::<f64>::zeros((8, 8));
        assert!(highpass_filter(&img, 0.0).is_err());
    }

    // ─── Butterworth filter ──────────────────────────────────────────────────

    #[test]
    fn butterworth_lowpass_dc_passes() {
        // A constant image should pass through a low-pass Butterworth unchanged
        let img = Array2::<f64>::from_elem((16, 16), 2.0);
        let out = butterworth_filter(&img, 0.3, 4, false).expect("ok");
        // DC should be approximately preserved
        let mean_out: f64 = out.iter().sum::<f64>() / (16.0 * 16.0);
        assert!(
            (mean_out - 2.0).abs() < 0.1,
            "Butterworth LP DC mean {mean_out} not close to 2.0"
        );
    }

    #[test]
    fn butterworth_highpass_reduces_dc() {
        let img = Array2::<f64>::from_elem((16, 16), 1.0);
        let out = butterworth_filter(&img, 0.1, 2, true).expect("ok");
        let max_abs = out.iter().map(|&v| v.abs()).fold(0.0f64, f64::max);
        assert!(
            max_abs < 0.5,
            "Butterworth HP on constant: max {max_abs} should be small"
        );
    }

    #[test]
    fn butterworth_invalid_order_returns_error() {
        let img = Array2::<f64>::zeros((8, 8));
        assert!(butterworth_filter(&img, 0.1, 0, false).is_err());
    }

    // ─── Gaussian frequency filter ───────────────────────────────────────────

    #[test]
    fn gaussian_freq_lowpass_dc_preserved() {
        let img = Array2::<f64>::from_elem((16, 16), 3.0);
        let out = gaussian_freq_filter(&img, 0.3, false).expect("ok");
        let mean_out: f64 = out.iter().sum::<f64>() / (16.0 * 16.0);
        assert!(
            (mean_out - 3.0).abs() < 0.3,
            "Gaussian LP DC mean {mean_out}"
        );
    }

    #[test]
    fn gaussian_freq_highpass_removes_dc() {
        let img = Array2::<f64>::from_elem((16, 16), 1.0);
        let out = gaussian_freq_filter(&img, 0.1, true).expect("ok");
        let max_abs = out.iter().map(|&v| v.abs()).fold(0.0f64, f64::max);
        assert!(max_abs < 0.5, "Gaussian HP on constant: max {max_abs}");
    }

    #[test]
    fn gaussian_freq_invalid_sigma_returns_error() {
        let img = Array2::<f64>::zeros((8, 8));
        assert!(gaussian_freq_filter(&img, 0.0, false).is_err());
        assert!(gaussian_freq_filter(&img, -1.0, false).is_err());
    }

    // ─── Phase correlation ───────────────────────────────────────────────────

    #[test]
    fn phase_correlation_zero_shift() {
        // Two identical images should produce (0, 0) shift
        let img = Array2::from_shape_fn((32, 32), |(r, c)| {
            let dr = r as f64 - 16.0;
            let dc = c as f64 - 16.0;
            (-0.05 * (dr * dr + dc * dc)).exp()
        });
        let (dr, dc) = phase_correlation(&img, &img).expect("ok");
        assert!(
            dr.abs() < 2.0 && dc.abs() < 2.0,
            "Expected zero shift, got ({dr}, {dc})"
        );
    }

    #[test]
    fn phase_correlation_known_shift() {
        let rows = 32usize;
        let cols = 32usize;
        let shift_r = 3i32;
        let shift_c = 5i32;

        // Create a reference image with a distinct Gaussian blob
        let img1 = Array2::from_shape_fn((rows, cols), |(r, c)| {
            let dr = r as f64 - 16.0;
            let dc = c as f64 - 16.0;
            (-0.02 * (dr * dr + dc * dc)).exp()
        });

        // Shift image2 by (shift_r, shift_c) using wrap-around
        let img2 = Array2::from_shape_fn((rows, cols), |(r, c)| {
            let r2 = ((r as i32 - shift_r).rem_euclid(rows as i32)) as usize;
            let c2 = ((c as i32 - shift_c).rem_euclid(cols as i32)) as usize;
            img1[[r2, c2]]
        });

        let (dr, dc) = phase_correlation(&img1, &img2).expect("ok");
        assert!(
            (dr - shift_r as f64).abs() <= 2.0,
            "Row shift: expected {shift_r}, got {dr}"
        );
        assert!(
            (dc - shift_c as f64).abs() <= 2.0,
            "Col shift: expected {shift_c}, got {dc}"
        );
    }

    #[test]
    fn phase_correlation_shape_mismatch_returns_error() {
        let img1 = Array2::<f64>::zeros((8, 8));
        let img2 = Array2::<f64>::zeros((8, 16));
        assert!(phase_correlation(&img1, &img2).is_err());
    }

    #[test]
    fn phase_correlation_too_small_returns_error() {
        let img1 = Array2::<f64>::zeros((1, 8));
        let img2 = Array2::<f64>::zeros((1, 8));
        assert!(phase_correlation(&img1, &img2).is_err());
    }
}
