//! N-Dimensional FFT Utilities
//!
//! This module provides convenient wrappers and utilities for N-dimensional
//! Fourier transforms operating directly on `ndarray` arrays of complex
//! numbers, as well as 2-D shift helpers and N-D frequency bin generation.
//!
//! # Overview
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`fftn_complex`]  | N-D FFT on `ArrayD<Complex<f64>>` |
//! | [`ifftn_complex`] | N-D inverse FFT on `ArrayD<Complex<f64>>` |
//! | [`fftshift2`]     | Move zero-frequency to the centre of a 2-D array |
//! | [`ifftshift2`]    | Inverse of [`fftshift2`] |
//! | [`fftfreq_nd`]    | Frequency bins for each axis of an N-D transform |
//!
//! ## Relationship to existing helpers
//!
//! * For generic `D`-dimensional arrays use [`crate::helper::fftshift`] /
//!   [`crate::helper::ifftshift`].
//! * For standard `ArrayD<T>` with real input see [`crate::fft::fftn`] /
//!   [`crate::fft::ifftn`].
//! * The functions here operate specifically on *complex-valued* `ArrayD` /
//!   `Array2` and expose a simpler axes-only interface.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use scirs2_core::ndarray::{Array2, ArrayD, Axis};
use scirs2_core::numeric::Complex64;

// ─────────────────────────────────────────────────────────────────────────────
//  fftn_complex / ifftn_complex
// ─────────────────────────────────────────────────────────────────────────────

/// N-dimensional FFT of a complex-valued array.
///
/// Applies a 1-D FFT along each axis listed in `axes` (or along all axes when
/// `axes` is `None`), producing a complex output array of the same shape.
///
/// # Arguments
///
/// * `x`    - Input complex array of any dimensionality.
/// * `axes` - Axes to transform.  `None` → transform all axes.
///
/// # Errors
///
/// Returns an error if any axis index is out of bounds.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::ndim::fftn_complex;
/// use scirs2_core::ndarray::{ArrayD, IxDyn};
/// use scirs2_core::numeric::Complex64;
///
/// // 2 × 4 complex array
/// let data: Vec<Complex64> = (0..8).map(|i| Complex64::new(i as f64, 0.0)).collect();
/// let x = ArrayD::from_shape_vec(IxDyn(&[2, 4]), data).expect("shape ok");
///
/// let spectrum = fftn_complex(&x, None).expect("fftn failed");
/// assert_eq!(spectrum.shape(), x.shape());
/// ```
pub fn fftn_complex(x: &ArrayD<Complex64>, axes: Option<&[usize]>) -> FFTResult<ArrayD<Complex64>> {
    let ndim = x.ndim();
    let axes_to_transform: Vec<usize> = match axes {
        Some(a) => {
            for &ax in a {
                if ax >= ndim {
                    return Err(FFTError::ValueError(format!(
                        "axis {ax} out of bounds for array of ndim={ndim}"
                    )));
                }
            }
            a.to_vec()
        }
        None => (0..ndim).collect(),
    };

    let mut result = x.to_owned();
    for ax in axes_to_transform {
        apply_fft1d_along_axis(&mut result, ax, false)?;
    }
    Ok(result)
}

/// N-dimensional inverse FFT of a complex-valued array.
///
/// Applies a 1-D inverse FFT along each axis listed in `axes` (or along all
/// axes when `axes` is `None`).
///
/// # Arguments
///
/// * `x`    - Input complex array.
/// * `axes` - Axes to transform inversely.  `None` → transform all axes.
///
/// # Errors
///
/// Returns an error if any axis index is out of bounds.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::ndim::{fftn_complex, ifftn_complex};
/// use scirs2_core::ndarray::{ArrayD, IxDyn};
/// use scirs2_core::numeric::Complex64;
///
/// let data: Vec<Complex64> = (0..8).map(|i| Complex64::new(i as f64, 0.0)).collect();
/// let x = ArrayD::from_shape_vec(IxDyn(&[2, 4]), data).expect("shape ok");
///
/// let spectrum  = fftn_complex(&x, None).expect("fftn failed");
/// let recovered = ifftn_complex(&spectrum, None).expect("ifftn failed");
///
/// // Round-trip should recover the original (within floating-point tolerance)
/// for (a, b) in x.iter().zip(recovered.iter()) {
///     assert!((a.re - b.re).abs() < 1e-10);
///     assert!((a.im - b.im).abs() < 1e-10);
/// }
/// ```
pub fn ifftn_complex(
    x: &ArrayD<Complex64>,
    axes: Option<&[usize]>,
) -> FFTResult<ArrayD<Complex64>> {
    let ndim = x.ndim();
    let axes_to_transform: Vec<usize> = match axes {
        Some(a) => {
            for &ax in a {
                if ax >= ndim {
                    return Err(FFTError::ValueError(format!(
                        "axis {ax} out of bounds for array of ndim={ndim}"
                    )));
                }
            }
            a.to_vec()
        }
        None => (0..ndim).collect(),
    };

    let mut result = x.to_owned();
    for ax in axes_to_transform {
        apply_fft1d_along_axis(&mut result, ax, true)?;
    }
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  2-D shift helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Shift the zero-frequency component to the centre of a 2-D complex array.
///
/// For a 2-D FFT output of shape `(M, N)` the DC component is at `[0, 0]`.
/// `fftshift2` moves it to the centre position `[M/2, N/2]` (integer division),
/// which is the natural representation for visualisation.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::ndim::fftshift2;
/// use scirs2_core::ndarray::Array2;
/// use scirs2_core::numeric::Complex64;
///
/// // 4×4 array where position [0,0] has value 1 (DC component)
/// let mut data = Array2::<Complex64>::zeros((4, 4));
/// data[[0, 0]] = Complex64::new(1.0, 0.0);
///
/// let shifted = fftshift2(&data);
/// // After shift the DC component is at [2, 2]
/// assert!((shifted[[2, 2]].re - 1.0).abs() < 1e-12);
/// ```
pub fn fftshift2(x: &Array2<Complex64>) -> Array2<Complex64> {
    shift2_impl(x, false)
}

/// Inverse of [`fftshift2`]: move the zero-frequency back to position `[0, 0]`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::ndim::{fftshift2, ifftshift2};
/// use scirs2_core::ndarray::Array2;
/// use scirs2_core::numeric::Complex64;
///
/// let mut data = Array2::<Complex64>::zeros((4, 4));
/// data[[0, 0]] = Complex64::new(1.0, 0.0);
///
/// let shifted   = fftshift2(&data);
/// let recovered = ifftshift2(&shifted);
/// assert!((recovered[[0, 0]].re - 1.0).abs() < 1e-12);
/// ```
pub fn ifftshift2(x: &Array2<Complex64>) -> Array2<Complex64> {
    shift2_impl(x, true)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Frequency bins for N-D FFT
// ─────────────────────────────────────────────────────────────────────────────

/// Compute frequency bins for each axis of an N-dimensional FFT.
///
/// Returns a vector (one entry per axis) of frequency bin arrays in cycles per
/// unit, using the per-axis sample spacings supplied in `d`.  This generalises
/// [`crate::helper::fftfreq`] to multiple axes at once.
///
/// # Arguments
///
/// * `shape` - Shape of the N-D array (one entry per dimension).
/// * `d`     - Sample spacing for each dimension.  Must have the same length as
/// `shape`; a value of `1.0` gives frequencies in cycles/sample.
///
/// # Returns
///
/// `Vec<Vec<f64>>` where `result[i]` contains the `shape[i]` frequency values
/// for axis `i`.
///
/// # Errors
///
/// Returns an error if `shape.len() != d.len()` or if any spacing is ≤ 0.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::ndim::fftfreq_nd;
///
/// // 4×8 array, sample spacing 0.5 in first axis and 1.0 in second
/// let freqs = fftfreq_nd(&[4, 8], &[0.5, 1.0]).expect("fftfreq_nd failed");
///
/// assert_eq!(freqs.len(), 2);
/// assert_eq!(freqs[0].len(), 4);
/// assert_eq!(freqs[1].len(), 8);
///
/// // DC component is always 0
/// assert_eq!(freqs[0][0], 0.0);
/// assert_eq!(freqs[1][0], 0.0);
/// ```
pub fn fftfreq_nd(shape: &[usize], d: &[f64]) -> FFTResult<Vec<Vec<f64>>> {
    if shape.len() != d.len() {
        return Err(FFTError::ValueError(format!(
            "shape.len()={} must equal d.len()={}",
            shape.len(),
            d.len()
        )));
    }
    for (i, &spacing) in d.iter().enumerate() {
        if spacing <= 0.0 {
            return Err(FFTError::ValueError(format!(
                "sample spacing d[{i}]={spacing} must be > 0"
            )));
        }
    }

    shape
        .iter()
        .zip(d.iter())
        .map(|(&n, &spacing)| fftfreq_1d(n, spacing))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a 1-D FFT or IFFT along the given axis of a dynamic-dim complex array.
fn apply_fft1d_along_axis(
    data: &mut ArrayD<Complex64>,
    axis: usize,
    inverse: bool,
) -> FFTResult<()> {
    let axis_len = data.shape()[axis];
    let mut buf = vec![Complex64::new(0.0, 0.0); axis_len];

    for mut lane in data.lanes_mut(Axis(axis)) {
        buf.iter_mut().zip(lane.iter()).for_each(|(b, &x)| *b = x);

        // Pass explicit size to avoid auto-padding to next power of two
        let n = buf.len();
        let transformed = if inverse {
            ifft(&buf, Some(n))?
        } else {
            fft(&buf, Some(n))?
        };

        lane.iter_mut()
            .zip(transformed.iter())
            .for_each(|(d, &s)| *d = s);
    }
    Ok(())
}

/// Shared implementation for fftshift2 / ifftshift2.
///
/// `inverse = false` → forward shift (DC to centre).
/// `inverse = true`  → inverse shift (centre to DC).
fn shift2_impl(x: &Array2<Complex64>, inverse: bool) -> Array2<Complex64> {
    let (rows, cols) = x.dim();
    let row_shift = if inverse {
        // For odd n: forward shift by n/2 (floor), inverse by ceil
        rows - rows / 2
    } else {
        rows / 2
    };
    let col_shift = if inverse { cols - cols / 2 } else { cols / 2 };

    let mut out = Array2::<Complex64>::zeros((rows, cols));
    for r in 0..rows {
        let new_r = (r + row_shift) % rows;
        for c in 0..cols {
            let new_c = (c + col_shift) % cols;
            out[[new_r, new_c]] = x[[r, c]];
        }
    }
    out
}

/// 1-D fftfreq: frequency values for n samples with spacing d.
///
/// Matches the convention of `numpy.fft.fftfreq` / `scipy.fft.fftfreq`:
/// - Even n: `[0, 1, ..., n/2-1, -n/2, -(n/2-1), ..., -1] / (n * d)`
/// - Odd  n: `[0, 1, ..., (n-1)/2, -((n-1)/2), ..., -1] / (n * d)`
fn fftfreq_1d(n: usize, d: f64) -> FFTResult<Vec<f64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let scale = 1.0 / (n as f64 * d);

    let mut freqs = Vec::with_capacity(n);
    let p = (n / 2) as i64; // positive half length (floor(n/2))

    // Positive frequencies: 0, 1, ..., p  (for even n, p = n/2; for odd, p = (n-1)/2)
    // But for even n the Nyquist bin n/2 is represented as *negative* (-n/2)
    for i in 0..n as i64 {
        let k = if i <= p - (if n % 2 == 0 { 1 } else { 0 }) as i64 {
            // Positive frequencies: 0 .. floor((n-1)/2)
            i
        } else {
            // Negative frequencies: -floor(n/2) .. -1
            i - n as i64
        };
        freqs.push(k as f64 * scale);
    }
    Ok(freqs)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::IxDyn;
    use std::f64::consts::PI;

    // ── fftn_complex / ifftn_complex roundtrip ───────────────────────────────

    fn make_complex_array(shape: &[usize]) -> ArrayD<Complex64> {
        let n: usize = shape.iter().product();
        let data: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(i as f64, -(i as f64) * 0.5))
            .collect();
        ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape ok")
    }

    #[test]
    fn test_fftn_ifftn_roundtrip_1d() {
        let x = make_complex_array(&[16]);
        let s = fftn_complex(&x, None).expect("fftn");
        let r = ifftn_complex(&s, None).expect("ifftn");
        for (a, b) in x.iter().zip(r.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fftn_ifftn_roundtrip_2d() {
        let x = make_complex_array(&[4, 8]);
        let s = fftn_complex(&x, None).expect("fftn 2d");
        let r = ifftn_complex(&s, None).expect("ifftn 2d");
        for (a, b) in x.iter().zip(r.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fftn_ifftn_roundtrip_3d() {
        let x = make_complex_array(&[2, 3, 4]);
        let s = fftn_complex(&x, None).expect("fftn 3d");
        let r = ifftn_complex(&s, None).expect("ifftn 3d");
        for (a, b) in x.iter().zip(r.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fftn_partial_axes() {
        let x = make_complex_array(&[4, 8]);
        // Only transform axis 1
        let s1 = fftn_complex(&x, Some(&[1])).expect("fftn axis 1");
        let r1 = ifftn_complex(&s1, Some(&[1])).expect("ifftn axis 1");
        for (a, b) in x.iter().zip(r1.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fftn_out_of_bounds_axis() {
        let x = make_complex_array(&[4, 8]);
        assert!(fftn_complex(&x, Some(&[2])).is_err()); // only 2 axes (0, 1)
        assert!(ifftn_complex(&x, Some(&[5])).is_err());
    }

    #[test]
    fn test_fftn_shape_preserved() {
        let x = make_complex_array(&[3, 5, 7]);
        let s = fftn_complex(&x, None).expect("fftn");
        assert_eq!(s.shape(), x.shape());
    }

    // ── fftshift2 / ifftshift2 ───────────────────────────────────────────────

    #[test]
    fn test_fftshift2_roundtrip_even() {
        let rows = 4;
        let cols = 6;
        let data: Vec<Complex64> = (0..(rows * cols) as i32)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect();
        let x = Array2::from_shape_vec((rows, cols), data).expect("shape");
        let shifted = fftshift2(&x);
        let recovered = ifftshift2(&shifted);
        for r in 0..rows {
            for c in 0..cols {
                assert_relative_eq!(x[[r, c]].re, recovered[[r, c]].re, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_fftshift2_roundtrip_odd() {
        let rows = 5;
        let cols = 7;
        let data: Vec<Complex64> = (0..(rows * cols) as i32)
            .map(|i| Complex64::new(i as f64, i as f64 * 0.1))
            .collect();
        let x = Array2::from_shape_vec((rows, cols), data).expect("shape");
        let shifted = fftshift2(&x);
        let recovered = ifftshift2(&shifted);
        for r in 0..rows {
            for c in 0..cols {
                assert_relative_eq!(x[[r, c]].re, recovered[[r, c]].re, epsilon = 1e-12);
                assert_relative_eq!(x[[r, c]].im, recovered[[r, c]].im, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_fftshift2_dc_to_centre() {
        let mut data = Array2::<Complex64>::zeros((4, 4));
        data[[0, 0]] = Complex64::new(1.0, 0.0);
        let shifted = fftshift2(&data);
        // For n=4, shift = 2 → DC moves to [2, 2]
        assert_relative_eq!(shifted[[2, 2]].re, 1.0, epsilon = 1e-12);
        assert_relative_eq!(shifted[[0, 0]].re, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_ifftshift2_dc_back() {
        let mut data = Array2::<Complex64>::zeros((4, 4));
        data[[0, 0]] = Complex64::new(1.0, 0.0);
        let shifted = fftshift2(&data);
        let recovered = ifftshift2(&shifted);
        assert_relative_eq!(recovered[[0, 0]].re, 1.0, epsilon = 1e-12);
    }

    // ── fftfreq_nd ───────────────────────────────────────────────────────────

    #[test]
    fn test_fftfreq_nd_basic() {
        let freqs = fftfreq_nd(&[4, 8], &[1.0, 1.0]).expect("fftfreq_nd");
        assert_eq!(freqs.len(), 2);
        assert_eq!(freqs[0].len(), 4);
        assert_eq!(freqs[1].len(), 8);
        // DC is always 0
        assert_relative_eq!(freqs[0][0], 0.0, epsilon = 1e-15);
        assert_relative_eq!(freqs[1][0], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_fftfreq_nd_matches_1d_fftfreq() {
        // Compare with the scalar fftfreq from crate::helper
        use crate::helper::fftfreq;
        let n = 16;
        let d = 0.5;
        let nd_freqs = fftfreq_nd(&[n], &[d]).expect("nd");
        let scalar_freqs = fftfreq(n, d).expect("1d");
        assert_eq!(nd_freqs[0].len(), scalar_freqs.len());
        for (a, b) in nd_freqs[0].iter().zip(scalar_freqs.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_fftfreq_nd_spacing() {
        // With d=0.5 the max positive frequency doubles compared to d=1.0
        let f1 = fftfreq_nd(&[8], &[1.0]).expect("d=1");
        let f2 = fftfreq_nd(&[8], &[0.5]).expect("d=0.5");
        // Max positive freq for n=8, d=1: 3/8; for d=0.5: 3/4
        assert_relative_eq!(f1[0][3], 3.0 / 8.0, epsilon = 1e-14);
        assert_relative_eq!(f2[0][3], 3.0 / 4.0, epsilon = 1e-14);
    }

    #[test]
    fn test_fftfreq_nd_mismatch_error() {
        assert!(fftfreq_nd(&[4, 8], &[1.0]).is_err()); // lengths differ
        assert!(fftfreq_nd(&[4], &[0.0]).is_err()); // zero spacing
        assert!(fftfreq_nd(&[4], &[-1.0]).is_err()); // negative spacing
    }

    #[test]
    fn test_fftfreq_nd_empty_axis() {
        let freqs = fftfreq_nd(&[0, 4], &[1.0, 1.0]).expect("empty axis ok");
        assert_eq!(freqs[0].len(), 0);
        assert_eq!(freqs[1].len(), 4);
    }

    // ── Correctness: 2D FFT shift is consistent with element-wise check ──────

    #[test]
    fn test_fftshift2_known_pattern() {
        // Build a 4×4 array with known values at corners
        let rows = 4;
        let cols = 4;
        let mut x = Array2::<Complex64>::zeros((rows, cols));
        x[[0, 0]] = Complex64::new(1.0, 0.0); // top-left (DC)
        x[[0, 2]] = Complex64::new(2.0, 0.0); // top-right region
        x[[2, 0]] = Complex64::new(3.0, 0.0); // bottom-left region
        x[[2, 2]] = Complex64::new(4.0, 0.0); // bottom-right region

        let shifted = fftshift2(&x);
        // For n=4 (even), shift = 2 → each element at [r,c] moves to [(r+2)%4, (c+2)%4]
        assert_relative_eq!(shifted[[2, 2]].re, 1.0, epsilon = 1e-12); // was [0,0]
        assert_relative_eq!(shifted[[2, 0]].re, 2.0, epsilon = 1e-12); // was [0,2]
        assert_relative_eq!(shifted[[0, 2]].re, 3.0, epsilon = 1e-12); // was [2,0]
        assert_relative_eq!(shifted[[0, 0]].re, 4.0, epsilon = 1e-12); // was [2,2]
    }

    // ── Integration: fftn + fftshift2 on a sinusoidal image ─────────────────

    #[test]
    fn test_fftn_then_shift_preserves_energy() {
        use std::f64::consts::PI;
        let n = 8;
        // Simple 2D sinusoid
        let data: Vec<Complex64> = (0..n * n)
            .map(|k| {
                let r = k / n;
                let c = k % n;
                let re =
                    (2.0 * PI * r as f64 / n as f64).cos() * (2.0 * PI * c as f64 / n as f64).cos();
                Complex64::new(re, 0.0)
            })
            .collect();
        let x = ArrayD::from_shape_vec(IxDyn(&[n, n]), data).expect("shape");
        let spec = fftn_complex(&x, None).expect("fftn");
        // Parseval: sum |X[k]|^2 = n^2 * sum |x[n]|^2
        let energy_x: f64 = x.iter().map(|c| c.norm_sqr()).sum();
        let energy_s: f64 = spec.iter().map(|c| c.norm_sqr()).sum();
        let n2 = (n * n) as f64;
        assert_relative_eq!(energy_s, n2 * energy_x, epsilon = 1e-8 * energy_s.max(1.0));
    }
}
