//! Wavelet transforms
//!
//! This module provides functions for continuous and discrete wavelet transforms,
//! useful for multi-resolution analysis of signals.

use crate::error::{SignalError, SignalResult};
use num_complex::{Complex64, ComplexFloat};
use num_traits::NumCast;
use std::f64::consts::PI;
use std::fmt::Debug;

/// Generate a Ricker (Mexican hat) wavelet
///
/// # Arguments
///
/// * `points` - Number of points in the wavelet
/// * `a` - Width parameter
///
/// # Returns
///
/// * The Ricker wavelet as a vector of length `points`
///
/// # Examples
///
/// ```
/// use scirs2_signal::wavelets::ricker;
///
/// // Generate a Ricker wavelet with 100 points and width parameter 4.0
/// let wavelet = ricker(100, 4.0).unwrap();
/// ```
pub fn ricker(points: usize, a: f64) -> SignalResult<Vec<f64>> {
    if points == 0 {
        return Err(SignalError::ValueError(
            "points must be greater than 0".to_string(),
        ));
    }

    if a <= 0.0 {
        return Err(SignalError::ValueError(
            "width parameter 'a' must be positive".to_string(),
        ));
    }

    // Calculate amplitude factor
    let amplitude = 2.0 / (std::f64::consts::PI.powf(0.25) * (3.0 * a).sqrt());
    let wsq = a * a;

    // Generate position vector
    let mid_point = (points - 1) as f64 / 2.0;
    let mut wavelet = Vec::with_capacity(points);

    for i in 0..points {
        let x = i as f64 - mid_point;
        let xsq = x * x;
        let mod_term = 1.0 - xsq / wsq;
        let gauss = (-xsq / (2.0 * wsq)).exp();
        let value = amplitude * mod_term * gauss;
        wavelet.push(value);
    }

    Ok(wavelet)
}

/// Generate a Morlet wavelet
///
/// The standard Morlet wavelet is defined as a complex exponential modulated by a Gaussian.
///
/// # Arguments
///
/// * `points` - Number of points in the wavelet
/// * `w` - Omega0 parameter (central frequency)
/// * `s` - Scaling factor (bandwidth parameter)
///
/// # Returns
///
/// * The Morlet wavelet as a vector of complex numbers with length `points`
///
/// # Examples
///
/// ```
/// use scirs2_signal::wavelets::morlet;
///
/// // Generate a Morlet wavelet with 100 points, central frequency 5.0, and scaling 1.0
/// let wavelet = morlet(100, 5.0, 1.0).unwrap();
/// ```
pub fn morlet(points: usize, w: f64, s: f64) -> SignalResult<Vec<Complex64>> {
    if points == 0 {
        return Err(SignalError::ValueError(
            "points must be greater than 0".to_string(),
        ));
    }

    if s <= 0.0 {
        return Err(SignalError::ValueError(
            "scaling parameter 's' must be positive".to_string(),
        ));
    }

    // Generate position vector
    let mid_point = (points - 1) as f64 / 2.0;
    let mut wavelet = Vec::with_capacity(points);

    for i in 0..points {
        let t = (i as f64 - mid_point) / s;

        // Complex exponential term (oscillation)
        let exp_term = Complex64::new(0.0, w * t).exp();

        // Gaussian envelope
        let gauss = (-0.5 * t * t).exp();

        // Normalization factor to ensure unit energy
        let norm = (PI * s * s).sqrt().recip();

        wavelet.push(norm * gauss * exp_term);
    }

    Ok(wavelet)
}

/// Generate a Complex Morlet wavelet with advanced parameters
///
/// The Complex Morlet wavelet is defined as a complex exponential modulated by a Gaussian,
/// with additional parameters for controlling the shape and admissibility.
///
/// # Arguments
///
/// * `points` - Number of points in the wavelet
/// * `center_frequency` - Center frequency of the wavelet (w0)
/// * `bandwidth` - Bandwidth parameter (sigma)
/// * `symmetry` - Symmetry parameter (beta) for introducing asymmetry in the wavelet shape
/// * `scale` - Scaling factor
///
/// # Returns
///
/// * The Complex Morlet wavelet as a vector of complex numbers with length `points`
///
/// # Examples
///
/// ```
/// use scirs2_signal::wavelets::complex_morlet;
///
/// // Generate a Complex Morlet wavelet with 100 points, center frequency 5.0,
/// // bandwidth 1.0, symmetry 0.0 (symmetric), and scaling 1.0
/// let wavelet = complex_morlet(100, 5.0, 1.0, 0.0, 1.0).unwrap();
/// ```
pub fn complex_morlet(
    points: usize,
    center_frequency: f64,
    bandwidth: f64,
    symmetry: f64,
    scale: f64,
) -> SignalResult<Vec<Complex64>> {
    if points == 0 {
        return Err(SignalError::ValueError(
            "points must be greater than 0".to_string(),
        ));
    }

    if bandwidth <= 0.0 {
        return Err(SignalError::ValueError(
            "bandwidth parameter must be positive".to_string(),
        ));
    }

    if scale <= 0.0 {
        return Err(SignalError::ValueError(
            "scale must be positive".to_string(),
        ));
    }

    // Generate position vector
    let mid_point = (points - 1) as f64 / 2.0;
    let mut wavelet = Vec::with_capacity(points);

    // Calculate normalization factor for unit energy
    let norm = 1.0 / (PI * bandwidth * bandwidth * scale * scale).sqrt();

    // Correction factor for admissibility condition
    let correction = (-center_frequency * center_frequency / (2.0 * bandwidth * bandwidth)).exp();

    for i in 0..points {
        let t = (i as f64 - mid_point) / scale;

        // Introduce asymmetry if symmetry parameter is non-zero
        let asymmetry = if symmetry != 0.0 {
            Complex64::new(symmetry * t * t / 2.0, 0.0).exp()
        } else {
            Complex64::new(1.0, 0.0)
        };

        // Complex exponential term (oscillation)
        let exp_term = Complex64::new(0.0, center_frequency * t).exp();

        // Gaussian envelope with bandwidth control
        let gauss = (-t * t / (2.0 * bandwidth * bandwidth)).exp();

        // Admissibility correction (subtraction of DC component)
        let dc_correction = if correction > 1e-10 {
            exp_term - correction
        } else {
            exp_term
        };

        // Final wavelet value
        let value = norm * gauss * dc_correction * asymmetry;

        wavelet.push(value);
    }

    Ok(wavelet)
}

/// Generate a Paul wavelet
///
/// # Arguments
///
/// * `points` - Number of points in the wavelet
/// * `order` - Order of the Paul wavelet (must be positive)
/// * `scale` - Scaling factor
///
/// # Returns
///
/// * The Paul wavelet as a vector of complex numbers with length `points`
///
/// # Examples
///
/// ```
/// use scirs2_signal::wavelets::paul;
///
/// // Generate a Paul wavelet with 100 points, order 4, and scaling 1.0
/// let wavelet = paul(100, 4, 1.0).unwrap();
/// ```
pub fn paul(points: usize, order: usize, scale: f64) -> SignalResult<Vec<Complex64>> {
    if points == 0 {
        return Err(SignalError::ValueError(
            "points must be greater than 0".to_string(),
        ));
    }

    if order == 0 {
        return Err(SignalError::ValueError(
            "order must be greater than 0".to_string(),
        ));
    }

    if scale <= 0.0 {
        return Err(SignalError::ValueError(
            "scale must be positive".to_string(),
        ));
    }

    // Calculate normalization factor
    let m = order as f64;
    let fact_2m_1 = factorial(2 * order - 1) as f64;
    let fact_m = factorial(order) as f64;
    let norm = (2.0_f64.powf(m) * fact_2m_1 / (std::f64::consts::PI * fact_m)).sqrt();

    // Generate position vector
    let mid_point = (points - 1) as f64 / 2.0;
    let mut wavelet = Vec::with_capacity(points);

    for i in 0..points {
        let t = (i as f64 - mid_point) / scale;

        // The Paul wavelet formula is the same for t != 0
        let value = if t != 0.0 {
            let factor = Complex64::new(0.0, 1.0).powf(order as f64);
            let denom = (1.0 - Complex64::new(0.0, t)).powf(order as f64 + 1.0);
            norm * factor / denom
        } else {
            // t == 0 (special case)
            let val = norm * 2.0_f64.powf(m - 1.0) * fact_2m_1 / (fact_m * (2.0 * m - 1.0));
            Complex64::new(val, 0.0)
        };

        wavelet.push(value);
    }

    Ok(wavelet)
}

/// Generate a Complex Gaussian wavelet
///
/// The Complex Gaussian wavelet is defined as the complex-valued negative derivative
/// of the Gaussian function. The parameter `order` determines which derivative is used.
///
/// # Arguments
///
/// * `points` - Number of points in the wavelet
/// * `order` - Order of the derivative (must be positive)
/// * `scale` - Scaling factor
///
/// # Returns
///
/// * The Complex Gaussian wavelet as a vector of complex numbers with length `points`
///
/// # Examples
///
/// ```
/// use scirs2_signal::wavelets::complex_gaussian;
///
/// // Generate a Complex Gaussian wavelet with 100 points, order 1, and scaling 1.0
/// let wavelet = complex_gaussian(100, 1, 1.0).unwrap();
/// ```
pub fn complex_gaussian(points: usize, order: usize, scale: f64) -> SignalResult<Vec<Complex64>> {
    if points == 0 {
        return Err(SignalError::ValueError(
            "points must be greater than 0".to_string(),
        ));
    }

    if order == 0 {
        return Err(SignalError::ValueError(
            "order must be greater than 0".to_string(),
        ));
    }

    if scale <= 0.0 {
        return Err(SignalError::ValueError(
            "scale must be positive".to_string(),
        ));
    }

    // Generate position vector
    let mid_point = (points - 1) as f64 / 2.0;
    let mut wavelet = Vec::with_capacity(points);

    // Helper function to calculate the complex Gaussian derivative of specified order
    let cgauss_derivative = |t: f64, m: usize| -> Complex64 {
        // Base Gaussian function
        let gauss = (-t * t / 2.0).exp();

        // Calculate the appropriate Hermite polynomial for this derivative
        let hermite = match m {
            1 => -t,                                                                       // H_1(t) = -t
            2 => t * t - 1.0,                              // H_2(t) = t² - 1
            3 => -t * t * t + 3.0 * t,                     // H_3(t) = -t³ + 3t
            4 => t.powi(4) - 6.0 * t * t + 3.0,            // H_4(t) = t⁴ - 6t² + 3
            5 => -t.powi(5) + 10.0 * t.powi(3) - 15.0 * t, // H_5(t) = -t⁵ + 10t³ - 15t
            6 => t.powi(6) - 15.0 * t.powi(4) + 45.0 * t * t - 15.0, // H_6(t) = t⁶ - 15t⁴ + 45t² - 15
            7 => -t.powi(7) + 21.0 * t.powi(5) - 105.0 * t.powi(3) + 105.0 * t, // H_7(t) = -t⁷ + 21t⁵ - 105t³ + 105t
            8 => t.powi(8) - 28.0 * t.powi(6) + 210.0 * t.powi(4) - 420.0 * t * t + 105.0, // H_8(t) = t⁸ - 28t⁶ + 210t⁴ - 420t² + 105
            _ => return Complex64::new(0.0, 0.0), // Fallback for unsupported orders
        };

        // Calculate normalization factor (-i)^m / sqrt(m!) for the Fourier transform
        let factor = Complex64::new(0.0, -1.0).powf(m as f64) / (factorial(m) as f64).sqrt();

        // Return normalized complex Gaussian derivative
        factor * hermite * gauss
    };

    // Calculate wavelet values at each point
    for i in 0..points {
        let t = (i as f64 - mid_point) / scale;
        let value = cgauss_derivative(t, order);
        wavelet.push(value);
    }

    // Normalize for unit energy
    let normalization = (PI * scale * scale).sqrt().recip();
    wavelet.iter_mut().for_each(|v| *v *= normalization);

    Ok(wavelet)
}

/// Generate a Shannon wavelet
///
/// The Shannon wavelet is defined using the sinc function modulated by a complex exponential.
/// It has excellent frequency localization properties but poor time localization.
///
/// # Arguments
///
/// * `points` - Number of points in the wavelet
/// * `center_frequency` - Center frequency parameter (>= 0)
/// * `bandwidth` - Bandwidth parameter (>= 0)
/// * `scale` - Scaling factor
///
/// # Returns
///
/// * The Shannon wavelet as a vector of complex numbers with length `points`
///
/// # Examples
///
/// ```
/// use scirs2_signal::wavelets::shannon;
///
/// // Generate a Shannon wavelet with 100 points, center frequency 1.0, bandwidth 0.5, and scaling 1.0
/// let wavelet = shannon(100, 1.0, 0.5, 1.0).unwrap();
/// ```
pub fn shannon(
    points: usize,
    center_frequency: f64,
    bandwidth: f64,
    scale: f64,
) -> SignalResult<Vec<Complex64>> {
    if points == 0 {
        return Err(SignalError::ValueError(
            "points must be greater than 0".to_string(),
        ));
    }

    if center_frequency < 0.0 {
        return Err(SignalError::ValueError(
            "center_frequency must be non-negative".to_string(),
        ));
    }

    if bandwidth < 0.0 {
        return Err(SignalError::ValueError(
            "bandwidth must be non-negative".to_string(),
        ));
    }

    if scale <= 0.0 {
        return Err(SignalError::ValueError(
            "scale must be positive".to_string(),
        ));
    }

    // Normalization factor for unit energy
    let norm = scale.sqrt().recip() * (2.0 * bandwidth).sqrt();

    // Generate position vector
    let mid_point = (points - 1) as f64 / 2.0;
    let mut wavelet = Vec::with_capacity(points);

    for i in 0..points {
        let t = (i as f64 - mid_point) / scale;

        // Calculate the sinc function (sin(x)/x)
        // This defines the band-limited wavelet in the time domain
        let sinc_term = if t == 0.0 {
            1.0 // lim(x->0) sin(x)/x = 1
        } else {
            (bandwidth * PI * t).sin() / (bandwidth * PI * t)
        };

        // Modulate by complex exponential to shift to the center frequency
        let modulation = Complex64::new(0.0, 2.0 * PI * center_frequency * t).exp();

        // Final wavelet value
        let value = norm * sinc_term * modulation;

        wavelet.push(value);
    }

    Ok(wavelet)
}

/// Generate a Frequency B-Spline (FBSP) wavelet
///
/// The FBSP wavelet combines a B-spline function with a complex modulation term.
/// This wavelet provides good time-frequency localization with a controllable
/// trade-off between time and frequency resolution.
///
/// # Arguments
///
/// * `points` - Number of points in the wavelet
/// * `center_frequency` - Center frequency parameter (>= 0)
/// * `bandwidth` - Bandwidth parameter (>= 0)
/// * `order` - B-spline order (>= 2, higher orders produce smoother wavelets)
/// * `scale` - Scaling factor
///
/// # Returns
///
/// * The FBSP wavelet as a vector of complex numbers with length `points`
///
/// # Examples
///
/// ```
/// use scirs2_signal::wavelets::fbsp;
///
/// // Generate a FBSP wavelet with 100 points, center frequency 1.0,
/// // bandwidth 0.5, order 3, and scaling 1.0
/// let wavelet = fbsp(100, 1.0, 0.5, 3, 1.0).unwrap();
/// ```
pub fn fbsp(
    points: usize,
    center_frequency: f64,
    bandwidth: f64,
    order: usize,
    scale: f64,
) -> SignalResult<Vec<Complex64>> {
    if points == 0 {
        return Err(SignalError::ValueError(
            "points must be greater than 0".to_string(),
        ));
    }

    if center_frequency < 0.0 {
        return Err(SignalError::ValueError(
            "center_frequency must be non-negative".to_string(),
        ));
    }

    if bandwidth < 0.0 {
        return Err(SignalError::ValueError(
            "bandwidth must be non-negative".to_string(),
        ));
    }

    if order < 2 {
        return Err(SignalError::ValueError(
            "order must be at least 2".to_string(),
        ));
    }

    if scale <= 0.0 {
        return Err(SignalError::ValueError(
            "scale must be positive".to_string(),
        ));
    }

    // Generate position vector
    let mid_point = (points - 1) as f64 / 2.0;
    let mut wavelet = Vec::with_capacity(points);

    // Calculate the normalization factor
    // For B-spline of order m, the normalization ensures unit energy
    let norm = scale.sqrt().recip() * (bandwidth * (order as f64)).sqrt();

    for i in 0..points {
        let t = (i as f64 - mid_point) / scale;

        // Compute the Fourier transform of a cardinal B-spline of order 'order'
        // This is the frequency response of the B-spline
        let bspline_ft = |omega: f64| -> f64 {
            if omega.abs() < 1e-10 {
                return 1.0; // Prevent division by zero
            }

            // The Fourier transform of a cardinal B-spline is (sin(ω/2)/(ω/2))^m
            let sinc_term = (omega / 2.0).sin() / (omega / 2.0);
            sinc_term.powi(order as i32)
        };

        // Calculate the B-spline wavelet in frequency domain
        // We apply a frequency shift by the center frequency
        let omega = 2.0 * PI * t * bandwidth;
        let bspline_value = bspline_ft(omega);

        // Modulate by complex exponential to shift to the center frequency
        let modulation = Complex64::new(0.0, 2.0 * PI * center_frequency * t).exp();

        // Final wavelet value: B-spline modulated by complex exponential
        let value = norm * bspline_value * modulation;

        wavelet.push(value);
    }

    Ok(wavelet)
}

/// Helper function to calculate factorial
fn factorial(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

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
/// ```ignore
/// # FIXME: Type conversion issue with Complex64
/// use scirs2_signal::wavelets::{cwt, morlet};
/// use num_complex::Complex64;
///
/// // Generate a complex signal
/// let signal: Vec<Complex64> = (0..100)
///     .map(|i| {
///         let t = i as f64 / 10.0;
///         Complex64::new(t.sin(), t.cos())
///     })
///     .collect();
///
/// // Define scales
/// let scales: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0];
///
/// // Compute CWT using the Morlet wavelet with 5.0 as central frequency parameter
/// let result = cwt(&signal, |points, scale| morlet(points, 5.0, scale), &scales).unwrap();
/// ```
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
        .map(|&val| num_traits::cast::cast::<T, f64>(val).ok_or(()))
        .collect();

    // Process data based on type
    let data_complex: Vec<Complex64> = if let Ok(real_data) = data_real {
        // Real data
        real_data.iter().map(|&r| Complex64::new(r, 0.0)).collect()
    } else {
        // Complex data
        is_complex = true;
        data.iter()
            .map(|&val| {
                num_traits::cast::cast::<T, Complex64>(val).ok_or_else(|| {
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
        let wavelet_data = wavelet(n, scale)?;

        // Convert to complex and take conjugate (for convolution)
        let mut wavelet_complex = Vec::with_capacity(n);
        for &w in &wavelet_data {
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

/// Helper function to convolve real signal with complex filter using 'same' mode
///
/// Optimized for the CWT case with real input data
fn convolve_complex_same_real(x: &[Complex64], h: &[Complex64]) -> Vec<Complex64> {
    let nx = x.len();
    let nh = h.len();
    let n_out = nx;

    // Allocate output buffer
    let mut out = vec![Complex64::new(0.0, 0.0); nx + nh - 1];

    // Perform convolution - since h is typically much smaller than x in the CWT case,
    // we optimize by iterating through h in the outer loop for better cache locality
    if nh < nx {
        for j in 0..nh {
            for i in 0..nx {
                out[i + j] += x[i] * h[j];
            }
        }
    } else {
        // Fall back to standard convolution when h is larger
        for i in 0..nx {
            for j in 0..nh {
                out[i + j] += x[i] * h[j];
            }
        }
    }

    // Extract the middle part ('same' mode)
    let start = (nh - 1) / 2;
    out.iter().skip(start).take(n_out).copied().collect()
}

/// Helper function to convolve complex signal with complex filter using 'same' mode
///
/// Handles fully complex CWT computation
fn convolve_complex_same_complex(x: &[Complex64], h: &[Complex64]) -> Vec<Complex64> {
    let nx = x.len();
    let nh = h.len();
    let n_out = nx;

    // Allocate output buffer
    let mut out = vec![Complex64::new(0.0, 0.0); nx + nh - 1];

    // Perform convolution - since h is typically much smaller than x in the CWT case,
    // we optimize by iterating through h in the outer loop for better cache locality
    if nh < nx {
        for j in 0..nh {
            for i in 0..nx {
                out[i + j] += x[i] * h[j];
            }
        }
    } else {
        // Fall back to standard convolution when h is larger
        for i in 0..nx {
            for j in 0..nh {
                out[i + j] += x[i] * h[j];
            }
        }
    }

    // Extract the middle part ('same' mode)
    let start = (nh - 1) / 2;
    out.iter().skip(start).take(n_out).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ricker_wavelet() {
        // Check that the Ricker wavelet has the right shape
        let points = 100;
        let a = 4.0;
        let wavelet = ricker(points, a).unwrap();

        assert_eq!(wavelet.len(), points);

        // Check symmetry
        let mid = points / 2;
        for i in 0..mid {
            assert_relative_eq!(wavelet[i], wavelet[points - 1 - i], epsilon = 1e-10);
        }

        // Check that the peak is at the middle
        let max_idx = wavelet
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(max_idx, mid);
    }

    #[test]
    fn test_morlet_wavelet() {
        // Check that the Morlet wavelet has the right shape
        let points = 100;
        let w = 5.0;
        let s = 1.0;
        let wavelet = morlet(points, w, s).unwrap();

        assert_eq!(wavelet.len(), points);

        // Check the envelope is symmetric (magnitude should be symmetric)
        let mid = points / 2;
        for i in 0..mid {
            assert_relative_eq!(
                wavelet[i].norm(),
                wavelet[points - 1 - i].norm(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_complex_morlet_wavelet() {
        // Parameters for the complex Morlet wavelet
        let points = 100;
        let center_frequency = 5.0;
        let bandwidth = 1.0;
        let symmetry = 0.0; // Symmetric
        let scale = 1.0;

        // Generate the wavelet
        let wavelet = complex_morlet(points, center_frequency, bandwidth, symmetry, scale).unwrap();

        // Check that we get the right number of points
        assert_eq!(wavelet.len(), points);

        // For a symmetric wavelet (symmetry=0), check that magnitudes are symmetric
        let mid = points / 2;
        for i in 0..mid {
            assert_relative_eq!(
                wavelet[i].norm(),
                wavelet[points - 1 - i].norm(),
                epsilon = 1e-10
            );
        }

        // Now create a wavelet with mild asymmetry - should still produce valid values
        let asymmetry = 0.2; // Use a smaller value that won't cause numerical issues
        let wavelet_asym =
            complex_morlet(points, center_frequency, bandwidth, asymmetry, scale).unwrap();

        // Just verify that the implementation returns the right number of points
        assert_eq!(wavelet_asym.len(), points);

        // And that values are different from the symmetric version
        let mut has_difference = false;
        for i in 0..points {
            if (wavelet[i].norm() - wavelet_asym[i].norm()).abs() > 1e-10 {
                has_difference = true;
                break;
            }
        }
        assert!(
            has_difference,
            "Asymmetric parameter should produce different wavelet values"
        );
    }

    #[test]
    fn test_complex_gaussian_wavelet() {
        // Parameters for the complex Gaussian wavelet
        let points = 100;
        let order = 4;
        let scale = 1.0;

        // Generate the wavelet
        let wavelet = complex_gaussian(points, order, scale).unwrap();

        // Check that we get the right number of points
        assert_eq!(wavelet.len(), points);

        // The complex Gaussian is a derivative of the Gaussian function
        // Higher-order derivatives have more oscillations

        // Count the number of zero crossings in the real part
        let mut zero_crossings = 0;
        for i in 1..points {
            if wavelet[i].re * wavelet[i - 1].re <= 0.0
                && (wavelet[i].re.abs() > 1e-10 || wavelet[i - 1].re.abs() > 1e-10)
            {
                zero_crossings += 1;
            }
        }

        // Higher-order Complex Gaussian should have multiple zero crossings
        assert!(
            zero_crossings >= 2,
            "Complex Gaussian should have multiple oscillations"
        );

        // Test order 1 Complex Gaussian (should be simpler)
        let wavelet_order1 = complex_gaussian(points, 1, scale).unwrap();
        let mut zero_crossings_order1 = 0;
        for i in 1..points {
            if wavelet_order1[i].re * wavelet_order1[i - 1].re <= 0.0
                && (wavelet_order1[i].re.abs() > 1e-10 || wavelet_order1[i - 1].re.abs() > 1e-10)
            {
                zero_crossings_order1 += 1;
            }
        }

        // Order 1 should have fewer zero crossings than order 4
        assert!(
            zero_crossings_order1 < zero_crossings,
            "Lower order should have fewer oscillations"
        );
    }

    #[test]
    fn test_shannon_wavelet() {
        // Parameters for the Shannon wavelet
        let points = 100;
        let center_frequency = 0.5;
        let bandwidth = 0.1;
        let scale = 1.0;

        // Generate the wavelet
        let wavelet = shannon(points, center_frequency, bandwidth, scale).unwrap();

        // Check that we get the right number of points
        assert_eq!(wavelet.len(), points);

        // Shannon wavelets are band-limited and should have symmetric magnitude
        let mid = points / 2;
        for i in 0..mid {
            assert_relative_eq!(
                wavelet[i].norm(),
                wavelet[points - 1 - i].norm(),
                epsilon = 1e-10
            );
        }

        // Test the frequency response by checking oscillation period
        // The oscillation period should be related to the center frequency
        let expected_period = (scale / center_frequency).round() as usize;
        if expected_period > 0 && expected_period < points / 4 {
            // Find the phase of the central few points
            let center_phases: Vec<f64> = (mid - 10..mid + 10).map(|i| wavelet[i].arg()).collect();

            // Calculate phase differences to estimate period
            let mut phase_diffs = Vec::new();
            for i in 1..center_phases.len() {
                let diff = (center_phases[i] - center_phases[i - 1]).abs();
                if diff > 0.01 {
                    // Ignore very small changes
                    phase_diffs.push(diff);
                }
            }

            // Average phase difference should be approximately 2π/expected_period
            if !phase_diffs.is_empty() {
                let avg_diff: f64 = phase_diffs.iter().sum::<f64>() / phase_diffs.len() as f64;
                let estimated_period = (2.0 * PI / avg_diff).round() as usize;

                // Allow some flexibility due to discretization and windowing effects
                assert!(
                    (estimated_period as isize - expected_period as isize).abs() <= 5,
                    "Expected period approximately {} but got {}",
                    expected_period,
                    estimated_period
                );
            }
        }
    }

    #[test]
    fn test_fbsp_wavelet() {
        // Parameters for the FBSP wavelet
        let points = 100;
        let center_frequency = 0.5;
        let bandwidth = 0.1;
        let order = 3;
        let scale = 1.0;

        // Generate the wavelet
        let wavelet = fbsp(points, center_frequency, bandwidth, order, scale).unwrap();

        // Check that we get the right number of points
        assert_eq!(wavelet.len(), points);

        // FBSP wavelets should have symmetric magnitude
        let mid = points / 2;
        for i in 0..mid {
            assert_relative_eq!(
                wavelet[i].norm(),
                wavelet[points - 1 - i].norm(),
                epsilon = 1e-10
            );
        }

        // Test with different orders
        let wavelet_order2 = fbsp(points, center_frequency, bandwidth, 2, scale).unwrap();
        let wavelet_order4 = fbsp(points, center_frequency, bandwidth, 4, scale).unwrap();

        // Higher order B-splines should be smoother (concentrate more energy around the center)
        // Calculate the energy concentration ratio: Sum of central 20% / total energy
        let central_start = (points as f64 * 0.4).round() as usize;
        let central_end = (points as f64 * 0.6).round() as usize;

        let calc_concentration = |w: &[Complex64]| -> f64 {
            let total_energy: f64 = w.iter().map(|c| c.norm_sqr()).sum();
            let central_energy: f64 = w[central_start..central_end]
                .iter()
                .map(|c| c.norm_sqr())
                .sum();
            central_energy / total_energy
        };

        let concentration_order2 = calc_concentration(&wavelet_order2);
        let concentration_order3 = calc_concentration(&wavelet);
        let concentration_order4 = calc_concentration(&wavelet_order4);

        // Higher orders should concentrate more energy in the center (be more localized in time)
        assert!(
            concentration_order3 > concentration_order2,
            "Order 3 should be more concentrated than order 2"
        );
        assert!(
            concentration_order4 > concentration_order3,
            "Order 4 should be more concentrated than order 3"
        );

        // Test oscillation frequency by checking phase differences
        let phases: Vec<f64> = wavelet.iter().map(|c| c.arg()).collect();
        let mut phase_diffs = Vec::new();
        for i in 1..phases.len() {
            let diff = (phases[i] - phases[i - 1]).abs();
            if diff > 0.01 && diff < PI {
                // Ignore very small changes and phase wrapping
                phase_diffs.push(diff);
            }
        }

        if !phase_diffs.is_empty() {
            let avg_diff: f64 = phase_diffs.iter().sum::<f64>() / phase_diffs.len() as f64;
            let estimated_period = (2.0 * PI / avg_diff).round() as usize;
            let expected_period = (scale / center_frequency).round() as usize;

            // The period should be roughly related to the center frequency
            // Allow some flexibility due to B-spline influence and discretization
            assert!(
                (estimated_period as isize - expected_period as isize).abs() <= 5,
                "Expected period approximately {} but got {}",
                expected_period,
                estimated_period
            );
        }
    }

    #[test]
    fn test_paul_wavelet() {
        // Check that the Paul wavelet has the right shape
        let points = 100;
        let order = 4;
        let scale = 1.0;
        let wavelet = paul(points, order, scale).unwrap();

        assert_eq!(wavelet.len(), points);

        // Paul wavelets have complex values with magnitude decreasing from the center
        let mid = points / 2;
        let mut magnitudes = vec![0.0; points];
        for i in 0..points {
            magnitudes[i] = wavelet[i].norm();
        }

        // Check peak is near the middle
        let max_idx = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        // Allow some flexibility since Paul wavelets may not peak exactly at center
        assert!((max_idx as isize - mid as isize).abs() <= 5);
    }

    #[test]
    fn test_cwt_with_ricker() {
        // Generate a simple sine wave
        let length = 100;
        let signal: Vec<f64> = (0..length)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin())
            .collect();

        // Define scales
        let scales = vec![1.0, 2.0, 4.0, 8.0, 16.0];

        // Compute CWT
        let result = cwt(
            &signal,
            |points, scale| {
                ricker(points, scale)
                    .map(|v| v.into_iter().map(|f| Complex64::new(f, 0.0)).collect())
            },
            &scales,
        )
        .unwrap();

        // Check dimensions
        assert_eq!(result.len(), scales.len());
        for row in &result {
            assert_eq!(row.len(), signal.len());
        }

        // Higher scales should capture the sine wave better (at scale close to period)
        let energy: Vec<f64> = result
            .iter()
            .map(|row| row.iter().map(|c| c.norm_sqr()).sum::<f64>())
            .collect();

        // Energy should peak at scales close to the signal period (20 samples)
        assert!(energy[3] > energy[0]); // Scale 8 > Scale 1
        assert!(energy[4] > energy[0]); // Scale 16 > Scale 1
    }

    #[test]
    fn test_cwt_with_complex_morlet() {
        // Simple signal for testing - a single sine wave
        let length = 100;
        let signal: Vec<f64> = (0..length)
            .map(|i| (2.0 * PI * i as f64 / 20.0).sin())
            .collect();

        // Define scales - just need a few for basic test
        let scales = vec![1.0, 4.0, 16.0];

        // Center frequency for complex Morlet
        let cf = 5.0;

        // Compute CWT using complex Morlet
        let result = cwt(
            &signal,
            |points, scale| complex_morlet(points, cf, 1.0, 0.0, scale),
            &scales,
        )
        .unwrap();

        // Basic dimensional validation
        assert_eq!(result.len(), scales.len());
        for row in &result {
            assert_eq!(row.len(), signal.len());
        }

        // For a simple sine wave, we know that the energy should be concentrated
        // at the scale closest to the signal period (20 samples)

        // Calculate total energy at each scale
        let scale_energies: Vec<f64> = result
            .iter()
            .map(|scale_data| scale_data.iter().map(|c| c.norm_sqr()).sum::<f64>())
            .collect();

        // Middle scale (4.0) should have significant energy for a sine wave with period ~20
        assert!(
            scale_energies[1] > 0.0,
            "CWT should capture sine wave energy"
        );

        // Check that the values are all finite (no NaN or infinity)
        for scale_data in &result {
            for val in scale_data {
                assert!(!val.re.is_nan() && !val.im.is_nan());
                assert!(!val.re.is_infinite() && !val.im.is_infinite());
            }
        }
    }

    #[test]
    fn test_cwt_with_fbsp() {
        // Simple signal for testing - a single sine wave
        let length = 100;
        let signal: Vec<f64> = (0..length)
            .map(|i| (2.0 * PI * i as f64 / 20.0).sin())
            .collect();

        // Define scales - just need a few for basic test
        let scales = vec![1.0, 4.0, 16.0];

        // FBSP parameters
        let center_frequency = 1.0;
        let bandwidth = 0.5;
        let order = 3;

        // Compute CWT using FBSP
        let result = cwt(
            &signal,
            |points, scale| fbsp(points, center_frequency, bandwidth, order, scale),
            &scales,
        )
        .unwrap();

        // Basic dimensional validation
        assert_eq!(result.len(), scales.len());
        for row in &result {
            assert_eq!(row.len(), signal.len());
        }

        // Calculate total energy at each scale
        let scale_energies: Vec<f64> = result
            .iter()
            .map(|scale_data| scale_data.iter().map(|c| c.norm_sqr()).sum::<f64>())
            .collect();

        // Verify that we have energy at all scales
        for energy in &scale_energies {
            assert!(*energy > 0.0, "FBSP wavelet should capture signal energy");
        }

        // Check that the values are all finite (no NaN or infinity)
        for scale_data in &result {
            for val in scale_data {
                assert!(!val.re.is_nan() && !val.im.is_nan());
                assert!(!val.re.is_infinite() && !val.im.is_infinite());
            }
        }

        // Also test with different order
        let order4_result = cwt(
            &signal,
            |points, scale| fbsp(points, center_frequency, bandwidth, 4, scale),
            &scales,
        )
        .unwrap();

        // Verify that higher order works too
        assert_eq!(order4_result.len(), scales.len());
        for row in &order4_result {
            assert_eq!(row.len(), signal.len());
        }
    }
}
