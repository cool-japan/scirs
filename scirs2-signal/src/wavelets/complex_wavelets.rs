// Complex-valued wavelets (Morlet, Paul, etc.)

use super::utils::factorial;
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

#[allow(unused_imports)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

        // Modulate by complex exponential to shift to the center _frequency
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
#[allow(dead_code)]
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
        // This is the _frequency response of the B-spline
        let bspline_ft = |omega: f64| -> f64 {
            if omega.abs() < 1e-10 {
                return 1.0; // Prevent division by zero
            }

            // The Fourier transform of a cardinal B-spline is (sin(ω/2)/(ω/2))^m
            let sinc_term = (omega / 2.0).sin() / (omega / 2.0);
            sinc_term.powi(order as i32)
        };

        // Calculate the B-spline wavelet in _frequency domain
        // We apply a _frequency shift by the center _frequency
        let omega = 2.0 * PI * t * bandwidth;
        let bspline_value = bspline_ft(omega);

        // Modulate by complex exponential to shift to the center _frequency
        let modulation = Complex64::new(0.0, 2.0 * PI * center_frequency * t).exp();

        // Final wavelet value: B-spline modulated by complex exponential
        let value = norm * bspline_value * modulation;

        wavelet.push(value);
    }

    Ok(wavelet)
}
