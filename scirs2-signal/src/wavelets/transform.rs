// Core wavelet transform implementations

use super::cwt::{convolve_complex_same_complex, convolve_complex_same_real};
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::NumCast;
use std::fmt::Debug;

#[allow(unused_imports)]
/// Continuous wavelet transform
///
/// # Arguments
///
/// * `data` - The input signal (real or complex)
/// * `wavelet` - A function that generates wavelet coefficients. This function should
///   take the number of points and the scale parameter and return a vector of
///   wavelet coefficients.
/// * `scales` - The scales at which to compute the transform
///
/// # Returns
///
/// * The continuous wavelet transform as a matrix where rows correspond to scales
///   and columns correspond to time points
///
/// # Examples
///
/// ```
/// use scirs2_signal::wavelets::{cwt, ricker};
///
/// // Generate a signal
/// let signal: Vec<f64> = (0..100).map(|i| (i as f64 / 10.0).sin()).collect();
///
/// // Define scales
/// let scales: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0];
///
/// // Compute CWT using the Ricker wavelet
/// let result = cwt(&signal, |points, scale| ricker(points, scale), &scales).unwrap();
/// ```
///
/// You can also use it with complex signals:
///
/// ```rust
/// use scirs2_signal::wavelets::{cwt, morlet};
///
/// // Generate a real signal (CWT also works with complex signals)
/// let signal: Vec<f64> = (0..100)
///     .map(|i| {
///         let t = i as f64 / 10.0;
///         (2.0 * std::f64::consts::PI * t).sin()
///     })
///     .collect();
///
/// // Define scales
/// let scales: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0];
///
/// // Compute CWT using the Morlet wavelet with 5.0 as central frequency parameter
/// let result = cwt(&signal, |points, scale| morlet(points, 5.0, scale), &scales).unwrap();
///
/// // Check results
/// assert_eq!(result.len(), scales.len());
/// assert_eq!(result[0].len(), signal.len());
/// ```
#[allow(dead_code)]
pub fn cwt<T, F, W>(data: &[T], wavelet: F, scales: &[f64]) -> SignalResult<Vec<Vec<Complex64>>>
where
    T: NumCast + Debug + Copy,
    F: Fn(usize, f64) -> SignalResult<Vec<W>>,
    W: Into<Complex64> + Copy,
{
    if data.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    if scales.is_empty() {
        return Err(SignalError::ValueError("Scales array is empty".to_string()));
    }

    // Try to convert to f64 first for real-valued input
    let mut is_complex = false;
    let data_real: Result<Vec<f64>, ()> = data
        .iter()
        .map(|&val| NumCast::from(val).ok_or(()))
        .collect();

    // Process data based on type
    let data_complex: Vec<Complex64> = if let Ok(realdata) = data_real {
        // Real data
        realdata.iter().map(|&r| Complex64::new(r, 0.0)).collect()
    } else {
        // Complex data
        is_complex = true;
        data.iter()
            .map(|&val| {
                NumCast::from(val).ok_or_else(|| {
                    SignalError::ValueError(format!("Could not convert {:?} to Complex64", val))
                })
            })
            .collect::<SignalResult<Vec<_>>>()?
    };

    // Validate scales
    for &scale in scales {
        if scale <= 0.0 {
            return Err(SignalError::ValueError(
                "Scales must be positive".to_string(),
            ));
        }
    }

    // Initialize output
    let mut output = Vec::with_capacity(scales.len());

    // Compute transform for each scale
    for &scale in scales {
        // Determine wavelet size - use at least data.len() points, but limit to reasonable size
        let n = std::cmp::min(
            data.len() * 10,
            std::cmp::max(data.len(), 10 * scale as usize),
        );

        // Generate wavelet coefficients
        let waveletdata = wavelet(n, scale)?;

        // Convert to complex and take conjugate (for convolution)
        let mut wavelet_complex = Vec::with_capacity(n);
        for &w in &waveletdata {
            let complex_val: Complex64 = w.into();
            wavelet_complex.push(complex_val.conj());
        }

        // Reverse for convolution
        wavelet_complex.reverse();

        // Convolve with 'same' mode - choose the right convolution function based on input type
        let convolved = if is_complex {
            convolve_complex_same_complex(&data_complex, &wavelet_complex)
        } else {
            // For real data, we can use the simpler convolution
            convolve_complex_same_real(&data_complex, &wavelet_complex)
        };

        output.push(convolved);
    }

    Ok(output)
}
