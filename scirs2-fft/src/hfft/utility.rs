//! Utility functions for HFFT operations
//!
//! This module contains helper functions for the Hermitian Fast Fourier Transform operations.

use crate::error::{FFTError, FFTResult};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::NumCast;
use std::fmt::Debug;

/// Parsed `norm` argument for the HFFT/IHFFT family, matching the three modes
/// documented by NumPy/SciPy (`numpy.fft.hfft`/`numpy.fft.ihfft`):
/// `"backward"` (default), `"forward"`, and `"ortho"`.
///
/// `hfft`/`hfft2`/`hfftn` play the role of the *forward*-type half of the
/// pair (a Hermitian-symmetric input is transformed into a real output,
/// analogous to how `rfft` transforms a real input into a complex output),
/// while `ihfft`/`ihfft2`/`ihfftn` play the role of the matching
/// *inverse*-type half. The two halves use opposite defaults for where the
/// `1/n` scaling goes, exactly as NumPy documents:
///
/// * `"backward"`: no scaling on the forward-type transform, `1/n` on the
///   inverse-type transform.
/// * `"forward"`: `1/n` on the forward-type transform, no scaling on the
///   inverse-type transform.
/// * `"ortho"`: `1/sqrt(n)` on both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HfftNorm {
    Backward,
    Forward,
    Ortho,
}

impl HfftNorm {
    /// Parse a `norm` string, defaulting to `"backward"` when `None`.
    ///
    /// Returns an error for anything other than the three documented mode
    /// strings, matching NumPy's own `ValueError` on an invalid `norm`.
    pub(crate) fn parse(norm: Option<&str>) -> FFTResult<Self> {
        match norm {
            None | Some("backward") => Ok(HfftNorm::Backward),
            Some("forward") => Ok(HfftNorm::Forward),
            Some("ortho") => Ok(HfftNorm::Ortho),
            Some(other) => Err(FFTError::ValueError(format!(
                "Invalid norm value '{other}': expected \"backward\", \"forward\", or \"ortho\""
            ))),
        }
    }

    /// Absolute scale factor to apply to a *raw* (completely unnormalized)
    /// forward-type transform of size `n`, i.e. the factor used by
    /// `hfft`/`hfft2`/`hfftn`.
    pub(crate) fn forward_scale(self, n: usize) -> FFTResult<f64> {
        match self {
            HfftNorm::Backward => Ok(1.0),
            HfftNorm::Forward => Ok(1.0 / Self::checked_n(n)?),
            HfftNorm::Ortho => Ok(1.0 / Self::checked_n(n)?.sqrt()),
        }
    }

    /// Absolute scale factor to apply to a *raw* (completely unnormalized)
    /// inverse-type transform of size `n`, i.e. the factor used by
    /// `ihfft`/`ihfft2`/`ihfftn`.
    pub(crate) fn inverse_scale(self, n: usize) -> FFTResult<f64> {
        match self {
            HfftNorm::Forward => Ok(1.0),
            HfftNorm::Backward => Ok(1.0 / Self::checked_n(n)?),
            HfftNorm::Ortho => Ok(1.0 / Self::checked_n(n)?.sqrt()),
        }
    }

    fn checked_n(n: usize) -> FFTResult<f64> {
        if n == 0 {
            return Err(FFTError::ValueError(
                "Cannot normalize a zero-length HFFT/IHFFT transform".to_string(),
            ));
        }
        Ok(n as f64)
    }
}

/// Enumerate every fixed combination of indices for all axes other than
/// `axis`, as full `shape.len()`-length index vectors (the slot at `axis`
/// is left at `0` as a placeholder; callers overwrite it while walking the
/// fiber along that axis).
///
/// This is the standard "hold every axis but one fixed, sweep the
/// remaining one" iteration needed to apply a 1-D transform along a single
/// axis of an N-D array: for an `(n_rows, n_cols)` array transformed along
/// axis 1, this yields one combination per row (`n_rows` combinations
/// total), not just the single row at index 0.
pub(crate) fn other_axis_index_combinations(shape: &[usize], axis: usize) -> Vec<Vec<usize>> {
    let ndim = shape.len();
    let mut combos: Vec<Vec<usize>> = vec![vec![0; ndim]];
    for (dim, &size) in shape.iter().enumerate() {
        if dim == axis {
            continue;
        }
        let mut expanded = Vec::with_capacity(combos.len() * size.max(1));
        for combo in &combos {
            for v in 0..size {
                let mut next = combo.clone();
                next[dim] = v;
                expanded.push(next);
            }
        }
        combos = expanded;
    }
    combos
}

/// Try to convert a value to Complex64
///
/// This function attempts to convert different types to Complex64:
/// - Complex64 values are passed through
/// - Complex32 values are converted to Complex64
/// - Other complex types are parsed from their debug representation
/// - Primitive numeric types are converted to Complex64 with zero imaginary part
///
/// # Arguments
///
/// * `val` - The value to convert
///
/// # Returns
///
/// * `Some(Complex64)` if the conversion was successful
/// * `None` if the conversion failed
pub(crate) fn try_as_complex<T: Copy + Debug + 'static + NumCast>(val: T) -> Option<Complex64> {
    // Check if the value is a Complex64 directly
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex64>() {
        unsafe {
            let ptr = &val as *const T as *const Complex64;
            return Some(*ptr);
        }
    }

    // Check for complex32
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<scirs2_core::numeric::Complex32>() {
        unsafe {
            let ptr = &val as *const T as *const scirs2_core::numeric::Complex32;
            let complex32 = *ptr;
            return Some(Complex64::new(complex32.re as f64, complex32.im as f64));
        }
    }

    // Handle other common complex number types by name-based detection
    // This is safer than trying to convert directly, as it avoids potential memory issues
    let type_name = std::any::type_name::<T>();
    if type_name.contains("Complex") {
        // For complex types, try to get the representation and parse it
        let debug_str = format!("{val:?}");

        // Try to extract re and im values using split and parse
        let re_im: Vec<f64> = debug_str
            .split(&[',', '(', ')', '{', '}', ':', ' '][..])
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();

        // If we found exactly two numbers, assume they're re and im
        if re_im.len() == 2 {
            return Some(Complex64::new(re_im[0], re_im[1]));
        }
    }

    // Handle primitive number types directly for better performance
    // For numeric primitives, we convert to Complex64 with zero imaginary part
    macro_rules! handle_primitive {
        ($type:ty) => {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$type>() {
                unsafe {
                    let ptr = &val as *const T as *const $type;
                    return Some(Complex64::new(*ptr as f64, 0.0));
                }
            }
        };
    }

    // Handle common numeric types
    handle_primitive!(f64);
    handle_primitive!(f32);
    handle_primitive!(i32);
    handle_primitive!(i64);
    handle_primitive!(u32);
    handle_primitive!(u64);
    handle_primitive!(i16);
    handle_primitive!(u16);
    handle_primitive!(i8);
    handle_primitive!(u8);

    // For other potential complex types, try to parse from Debug representation
    // This is a more robust approach for complex types from other libraries
    let debug_str = format!("{val:?}");
    if debug_str.contains("Complex") || (debug_str.contains("re") && debug_str.contains("im")) {
        // Extract numbers from the debug string
        let re_im: Vec<f64> = debug_str
            .split(&[',', '(', ')', '{', '}', ':', ' '][..])
            .filter_map(|s| {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    trimmed.parse::<f64>().ok()
                } else {
                    None
                }
            })
            .collect();

        // Try different approaches to extract values
        if re_im.len() == 2 {
            // If we found exactly two numbers, assume they're re and im
            return Some(Complex64::new(re_im[0], re_im[1]));
        } else if debug_str.contains("re:") && debug_str.contains("im:") {
            // For more complex representations like { re: 1.0, im: 2.0 }
            let re_str = debug_str
                .split("re:")
                .nth(1)
                .and_then(|s| s.split(',').next());
            let im_str = debug_str
                .split("im:")
                .nth(1)
                .and_then(|s| s.split('}').next());

            if let (Some(re_s), Some(im_s)) = (re_str, im_str) {
                if let (Ok(re), Ok(im)) = (re_s.trim().parse::<f64>(), im_s.trim().parse::<f64>()) {
                    return Some(Complex64::new(re, im));
                }
            }
        }
    }

    // As a last resort, try generic NumCast conversion
    NumCast::from(val).map(|v| Complex64::new(v, 0.0))
}
