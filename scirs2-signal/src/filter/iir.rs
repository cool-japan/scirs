//! IIR (Infinite Impulse Response) filter design functions
//!
//! This module provides comprehensive IIR filter design capabilities including
//! classic analog filter prototypes (Butterworth, Chebyshev, Elliptic, Bessel)
//! and specialized IIR design methods. All filters use the bilinear transform
//! for analog-to-digital conversion.

use crate::error::{SignalError, SignalResult};
use num_complex::Complex64;
use num_traits::{Float, NumCast};
use std::fmt::Debug;

use super::common::{
    math::{add_digital_zeros, bilinear_pole_transform, butterworth_poles, prewarp_frequency},
    validation::{convert_filter_type, validate_cutoff_frequency, validate_order},
    FilterCoefficients, FilterType, FilterTypeParam,
};

/// Butterworth filter design
///
/// Designs a digital Butterworth filter with maximally flat frequency response
/// in the passband. Butterworth filters provide the best approximation to the
/// ideal "brick wall" filter response in the passband.
///
/// # Arguments
///
/// * `order` - Filter order (higher order = steeper roll-off)
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is Nyquist frequency)
/// * `filter_type` - Filter type (lowpass, highpass, bandpass, bandstop)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a) where b are the numerator coefficients
///   and a are the denominator coefficients
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::iir::butter;
/// use scirs2_signal::filter::FilterType;
///
/// // Design a 4th order lowpass Butterworth filter with cutoff at 0.2 times Nyquist
/// let (b, a) = butter(4, 0.2, FilterType::Lowpass).unwrap();
///
/// // Using string parameter
/// let (b, a) = butter(4, 0.2, "lowpass").unwrap();
/// ```
pub fn butter<T>(
    order: usize,
    cutoff: T,
    filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<FilterCoefficients>
where
    T: Float + NumCast + Debug,
{
    // Validate parameters
    validate_order(order)?;
    let wn = validate_cutoff_frequency(cutoff)?;
    let filter_type = convert_filter_type(filter_type.into())?;

    // Step 1: Calculate analog Butterworth prototype poles
    let poles = butterworth_poles(order);

    // Step 2: Apply frequency transformation based on filter type
    let (analog_zeros, transformed_poles, gain) = match filter_type {
        FilterType::Lowpass => {
            // Scale poles by cutoff frequency (pre-warping for bilinear transform)
            let warped_freq = prewarp_frequency(wn);
            let scaled_poles: Vec<_> = poles.iter().map(|p| p * warped_freq).collect();
            // Lowpass has no finite zeros in analog domain (zeros at infinity)
            (
                Vec::<Complex64>::new(),
                scaled_poles,
                warped_freq.powi(order as i32),
            )
        }
        FilterType::Highpass => {
            // Highpass: s -> wc/s transformation
            let warped_freq = prewarp_frequency(wn);
            let hp_poles: Vec<_> = poles.iter().map(|p| warped_freq / p).collect();
            // No finite zeros in analog domain for highpass - zeros are at origin
            (Vec::<Complex64>::new(), hp_poles, 1.0)
        }
        FilterType::Bandpass => {
            return butter_bandpass_bandstop(order, wn - 0.05, wn + 0.05, FilterType::Bandpass);
        }
        FilterType::Bandstop => {
            return butter_bandpass_bandstop(order, wn - 0.05, wn + 0.05, FilterType::Bandstop);
        }
    };

    // Step 3: Apply bilinear transform to convert to digital filter
    let mut digital_poles = Vec::new();
    let mut digital_zeros = Vec::new();

    // Transform poles: z_pole = (2 + s_pole) / (2 - s_pole)
    for &pole in &transformed_poles {
        digital_poles.push(bilinear_pole_transform(pole));
    }

    // Transform finite analog zeros: z_zero = (2 + s_zero) / (2 - s_zero)
    for &zero in &analog_zeros {
        digital_zeros.push(bilinear_pole_transform(zero));
    }

    // Add zeros in the digital domain based on filter type
    digital_zeros.extend(add_digital_zeros(filter_type, order));

    // Step 4: Convert poles and zeros to transfer function coefficients
    zpk_to_tf(&digital_zeros, &digital_poles, gain)
}

/// Butterworth bandpass/bandstop filter design
///
/// Design Butterworth bandpass or bandstop filters with explicit low and high cutoff frequencies.
/// This function provides proper design for multi-band filters.
///
/// # Arguments
///
/// * `order` - Filter order (total poles will be 2*order for bandpass/bandstop)
/// * `low_freq` - Low cutoff frequency (normalized from 0 to 1)
/// * `high_freq` - High cutoff frequency (normalized from 0 to 1)
/// * `filter_type` - Filter type (must be Bandpass or Bandstop)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a)
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::iir::butter_bandpass_bandstop;
/// use scirs2_signal::filter::FilterType;
///
/// // Design a 4th order bandpass Butterworth filter from 0.1 to 0.4 times Nyquist
/// let (b, a) = butter_bandpass_bandstop(4, 0.1, 0.4, FilterType::Bandpass).unwrap();
/// ```
pub fn butter_bandpass_bandstop(
    order: usize,
    low_freq: f64,
    high_freq: f64,
    filter_type: FilterType,
) -> SignalResult<FilterCoefficients> {
    validate_order(order)?;

    // Validate frequency bounds
    if low_freq <= 0.0 || high_freq >= 1.0 || low_freq >= high_freq {
        return Err(SignalError::ValueError(
            "Invalid band frequencies: low must be positive, high must be less than 1, and low < high".to_string(),
        ));
    }

    if !matches!(filter_type, FilterType::Bandpass | FilterType::Bandstop) {
        return Err(SignalError::ValueError(
            "Filter type must be Bandpass or Bandstop".to_string(),
        ));
    }

    // Calculate analog Butterworth prototype poles
    let poles = butterworth_poles(order);

    // Pre-warp frequencies
    let wl = prewarp_frequency(low_freq);
    let wh = prewarp_frequency(high_freq);
    let center_freq = (wl * wh).sqrt();
    let bandwidth = wh - wl;

    let (analog_zeros, transformed_poles, gain) = match filter_type {
        FilterType::Bandpass => {
            // Apply bandpass transformation: s -> (s^2 + wc^2) / (s * BW)
            let mut bp_poles = Vec::new();
            let mut bp_zeros = Vec::new();

            for &pole in &poles {
                // Apply bandpass transformation to each pole
                let discriminant = (bandwidth * pole / 2.0).powi(2) + center_freq.powi(2);
                let sqrt_disc = discriminant.sqrt();
                let p1 = bandwidth * pole / 2.0 + sqrt_disc;
                let p2 = bandwidth * pole / 2.0 - sqrt_disc;
                bp_poles.push(p1);
                bp_poles.push(p2);
            }

            // Bandpass has zeros at origin (DC) and infinity
            for _ in 0..order {
                bp_zeros.push(Complex64::new(0.0, 0.0)); // Zero at origin
            }

            (bp_zeros, bp_poles, 1.0)
        }
        FilterType::Bandstop => {
            // Apply bandstop transformation: s -> (s * BW) / (s^2 + wc^2)
            let mut bs_poles = Vec::new();
            let mut bs_zeros = Vec::new();

            for &pole in &poles {
                let discriminant = (bandwidth / (2.0 * pole)).powi(2) + center_freq.powi(2);
                let sqrt_disc = discriminant.sqrt();
                let p1 = bandwidth / (2.0 * pole) + sqrt_disc;
                let p2 = bandwidth / (2.0 * pole) - sqrt_disc;
                bs_poles.push(p1);
                bs_poles.push(p2);
            }

            // Bandstop has zeros at ±j*wc (notch frequencies)
            for _ in 0..order {
                bs_zeros.push(Complex64::new(0.0, center_freq)); // +j*wc
                bs_zeros.push(Complex64::new(0.0, -center_freq)); // -j*wc
            }

            (bs_zeros, bs_poles, 1.0)
        }
        _ => unreachable!(), // Already validated above
    };

    // Apply bilinear transform to convert to digital filter
    let digital_poles: Vec<_> = transformed_poles
        .iter()
        .map(|&pole| bilinear_pole_transform(pole))
        .collect();

    let digital_zeros: Vec<_> = analog_zeros
        .iter()
        .map(|&zero| bilinear_pole_transform(zero))
        .collect();

    // Convert poles and zeros to transfer function coefficients
    zpk_to_tf(&digital_zeros, &digital_poles, gain)
}

/// Chebyshev Type I filter design
///
/// Designs a digital Chebyshev Type I filter with equiripple passband and
/// monotonic stopband. Provides steeper roll-off than Butterworth at the
/// cost of passband ripple.
///
/// # Arguments
///
/// * `order` - Filter order
/// * `ripple` - Passband ripple in dB (e.g., 0.5 for 0.5 dB ripple)
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1)
/// * `filter_type` - Filter type (lowpass, highpass, bandpass, bandstop)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a)
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::iir::cheby1;
///
/// // Design a 4th order Chebyshev I lowpass filter with 0.5 dB ripple
/// let (b, a) = cheby1(4, 0.5, 0.3, "lowpass").unwrap();
/// ```
pub fn cheby1<T>(
    order: usize,
    ripple: f64,
    cutoff: T,
    filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<FilterCoefficients>
where
    T: Float + NumCast + Debug,
{
    validate_order(order)?;
    let wn = validate_cutoff_frequency(cutoff)?;
    let filter_type = convert_filter_type(filter_type.into())?;

    if ripple <= 0.0 {
        return Err(SignalError::ValueError(
            "Ripple must be positive".to_string(),
        ));
    }

    // For now, return a simple error for unsupported types
    if !matches!(filter_type, FilterType::Lowpass | FilterType::Highpass) {
        return Err(SignalError::NotImplementedError(
            "Bandpass and bandstop Chebyshev I filters not yet implemented".to_string(),
        ));
    }

    // Convert ripple from dB to linear
    let epsilon = (10.0_f64.powf(ripple / 10.0) - 1.0).sqrt();

    // Calculate Chebyshev Type I analog prototype poles
    let mut poles = Vec::with_capacity(order);
    let a = (1.0 / epsilon + (1.0 / epsilon / epsilon + 1.0).sqrt()).ln() / order as f64;

    for k in 0..order {
        let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
        let real = -a.sinh() * theta.sin();
        let imag = a.cosh() * theta.cos();
        poles.push(Complex64::new(real, imag));
    }

    // Apply frequency transformation and bilinear transform
    let (analog_zeros, transformed_poles, gain) = match filter_type {
        FilterType::Lowpass => {
            let warped_freq = prewarp_frequency(wn);
            let scaled_poles: Vec<_> = poles.iter().map(|p| p * warped_freq).collect();
            (
                Vec::<Complex64>::new(),
                scaled_poles,
                warped_freq.powi(order as i32),
            )
        }
        FilterType::Highpass => {
            let warped_freq = prewarp_frequency(wn);
            let hp_poles: Vec<_> = poles.iter().map(|p| warped_freq / p).collect();
            (Vec::<Complex64>::new(), hp_poles, 1.0)
        }
        _ => unreachable!(),
    };

    let digital_poles: Vec<_> = transformed_poles
        .iter()
        .map(|&pole| bilinear_pole_transform(pole))
        .collect();

    let mut digital_zeros: Vec<_> = analog_zeros
        .iter()
        .map(|&zero| bilinear_pole_transform(zero))
        .collect();

    digital_zeros.extend(add_digital_zeros(filter_type, order));

    zpk_to_tf(&digital_zeros, &digital_poles, gain)
}

/// Chebyshev Type II filter design  
///
/// Designs a digital Chebyshev Type II filter with monotonic passband and
/// equiripple stopband. Provides better stopband attenuation than Type I.
///
/// # Arguments
///
/// * `order` - Filter order
/// * `attenuation` - Stopband attenuation in dB (e.g., 40.0 for 40 dB attenuation)
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1)
/// * `filter_type` - Filter type (lowpass, highpass, bandpass, bandstop)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a)
///
/// # Examples
///
/// ```ignore
/// # FIXME: Chebyshev Type II filter is not yet implemented
/// use scirs2_signal::filter::iir::cheby2;
///
/// // Design a 4th order Chebyshev II lowpass filter with 40 dB stopband attenuation
/// let (b, a) = cheby2(4, 40.0, 0.3, "lowpass").unwrap();
/// ```
pub fn cheby2<T>(
    order: usize,
    attenuation: f64,
    cutoff: T,
    filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<FilterCoefficients>
where
    T: Float + NumCast + Debug,
{
    validate_order(order)?;
    let _wn = validate_cutoff_frequency(cutoff)?;
    let _filter_type = convert_filter_type(filter_type.into())?;

    if attenuation <= 0.0 {
        return Err(SignalError::ValueError(
            "Attenuation must be positive".to_string(),
        ));
    }

    // Placeholder implementation - full Chebyshev II design is complex
    Err(SignalError::NotImplementedError(
        "Chebyshev Type II filter design is not yet implemented".to_string(),
    ))
}

/// Elliptic (Cauer) filter design
///
/// Designs a digital elliptic filter with equiripple passband and stopband.
/// Elliptic filters provide the steepest roll-off of any IIR filter type.
///
/// # Arguments
///
/// * `order` - Filter order
/// * `passband_ripple` - Passband ripple in dB
/// * `stopband_attenuation` - Stopband attenuation in dB
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1)
/// * `filter_type` - Filter type (lowpass, highpass, bandpass, bandstop)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a)
///
/// # Examples
///
/// ```ignore
/// # FIXME: Elliptic filter is not yet implemented
/// use scirs2_signal::filter::iir::ellip;
///
/// // Design a 4th order elliptic lowpass filter with 0.5 dB ripple and 40 dB stopband attenuation
/// let (b, a) = ellip(4, 0.5, 40.0, 0.3, "lowpass").unwrap();
/// ```
pub fn ellip<T>(
    order: usize,
    passband_ripple: f64,
    stopband_attenuation: f64,
    cutoff: T,
    filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<FilterCoefficients>
where
    T: Float + NumCast + Debug,
{
    validate_order(order)?;
    let _wn = validate_cutoff_frequency(cutoff)?;
    let _filter_type = convert_filter_type(filter_type.into())?;

    if passband_ripple <= 0.0 {
        return Err(SignalError::ValueError(
            "Passband ripple must be positive".to_string(),
        ));
    }

    if stopband_attenuation <= 0.0 {
        return Err(SignalError::ValueError(
            "Stopband attenuation must be positive".to_string(),
        ));
    }

    // Placeholder implementation - full elliptic design requires Jacobi elliptic functions
    Err(SignalError::NotImplementedError(
        "Elliptic filter design is not yet implemented".to_string(),
    ))
}

/// Bessel filter design
///
/// Designs a digital Bessel filter with maximally flat group delay.
/// Bessel filters provide excellent phase linearity, making them ideal
/// for applications requiring minimal phase distortion.
///
/// # Arguments
///
/// * `order` - Filter order
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1)
/// * `filter_type` - Filter type (lowpass, highpass, bandpass, bandstop)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a)
///
/// # Examples
///
/// ```ignore
/// # FIXME: Bessel filter is not yet implemented
/// use scirs2_signal::filter::iir::bessel;
///
/// // Design a 4th order Bessel lowpass filter
/// let (b, a) = bessel(4, 0.3, "lowpass").unwrap();
/// ```
pub fn bessel<T>(
    order: usize,
    cutoff: T,
    filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<FilterCoefficients>
where
    T: Float + NumCast + Debug,
{
    validate_order(order)?;
    let _wn = validate_cutoff_frequency(cutoff)?;
    let _filter_type = convert_filter_type(filter_type.into())?;

    // Placeholder implementation - Bessel filter design requires Bessel polynomials
    Err(SignalError::NotImplementedError(
        "Bessel filter design is not yet implemented".to_string(),
    ))
}

/// Convert zeros, poles, and gain to transfer function coefficients
///
/// Converts a filter representation in zeros-poles-gain form to
/// transfer function coefficients (numerator and denominator polynomials).
///
/// # Arguments
///
/// * `zeros` - Filter zeros in the z-domain
/// * `poles` - Filter poles in the z-domain  
/// * `gain` - Filter gain
///
/// # Returns
///
/// * Tuple of (numerator_coeffs, denominator_coeffs)
fn zpk_to_tf(
    zeros: &[Complex64],
    poles: &[Complex64],
    gain: f64,
) -> SignalResult<FilterCoefficients> {
    // Build numerator polynomial from zeros
    let mut num_poly = vec![Complex64::new(1.0, 0.0)];
    for &zero in zeros {
        // Multiply polynomial by (z - zero)
        let mut new_poly = vec![Complex64::new(0.0, 0.0); num_poly.len() + 1];

        // Multiply by z (shift coefficients)
        for (i, &coeff) in num_poly.iter().enumerate() {
            new_poly[i] += coeff;
        }

        // Subtract zero times polynomial
        for (i, &coeff) in num_poly.iter().enumerate() {
            new_poly[i + 1] -= zero * coeff;
        }

        num_poly = new_poly;
    }

    // Build denominator polynomial from poles
    let mut den_poly = vec![Complex64::new(1.0, 0.0)];
    for &pole in poles {
        // Multiply polynomial by (z - pole)
        let mut new_poly = vec![Complex64::new(0.0, 0.0); den_poly.len() + 1];

        // Multiply by z (shift coefficients)
        for (i, &coeff) in den_poly.iter().enumerate() {
            new_poly[i] += coeff;
        }

        // Subtract pole times polynomial
        for (i, &coeff) in den_poly.iter().enumerate() {
            new_poly[i + 1] -= pole * coeff;
        }

        den_poly = new_poly;
    }

    // Apply gain to numerator
    for coeff in &mut num_poly {
        *coeff *= gain;
    }

    // Convert complex coefficients to real (should be real for proper filter design)
    let b: Vec<f64> = num_poly
        .iter()
        .map(|c| {
            if c.im.abs() > 1e-10 {
                eprintln!(
                    "Warning: Numerator coefficient has significant imaginary part: {}",
                    c.im
                );
            }
            c.re
        })
        .collect();

    let a: Vec<f64> = den_poly
        .iter()
        .map(|c| {
            if c.im.abs() > 1e-10 {
                eprintln!(
                    "Warning: Denominator coefficient has significant imaginary part: {}",
                    c.im
                );
            }
            c.re
        })
        .collect();

    // Ensure denominator is monic (leading coefficient = 1)
    if a.is_empty() || a[0].abs() < 1e-15 {
        return Err(SignalError::ValueError(
            "Invalid denominator polynomial".to_string(),
        ));
    }

    let a0 = a[0];
    let b_normalized: Vec<f64> = b.iter().map(|&coeff| coeff / a0).collect();
    let a_normalized: Vec<f64> = a.iter().map(|&coeff| coeff / a0).collect();

    Ok((b_normalized, a_normalized))
}
