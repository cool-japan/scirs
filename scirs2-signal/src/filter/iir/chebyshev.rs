//! Chebyshev Type I (equiripple passband) and Type II (equiripple stopband)
//! IIR filter design.

use super::zpk_to_tf;
use crate::error::{SignalError, SignalResult};
use crate::filter::common::{
    math::{add_digital_zeros, bilinear_pole_transform, prewarp_frequency},
    validation::{convert_filter_type, validate_cutoff_frequency, validate_order},
    FilterCoefficients, FilterType, FilterTypeParam,
};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{Float, NumCast};
use std::fmt::Debug;

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
/// let (b, a) = cheby1(4, 0.5, 0.3, "lowpass").expect("Operation failed");
/// ```
#[allow(dead_code)]
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

    // Convert ripple from dB to linear
    let epsilon = (10.0_f64.powf(ripple / 10.0) - 1.0).sqrt();

    // Calculate Chebyshev Type I analog prototype poles
    let mut poles = Vec::with_capacity(order);
    let a = ((1.0 / epsilon + (1.0 / epsilon / epsilon + 1.0)).sqrt()).ln() / order as f64;

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
        FilterType::Bandpass => {
            let half_bw = (wn * 0.1).max(0.05).min(wn.min(1.0 - wn) * 0.4);
            let low = (wn - half_bw).max(1e-4);
            let high = (wn + half_bw).min(1.0 - 1e-4);
            return cheby1_bandpass_bandstop(order, ripple, low, high, FilterType::Bandpass);
        }
        FilterType::Bandstop => {
            let half_bw = (wn * 0.1).max(0.05).min(wn.min(1.0 - wn) * 0.4);
            let low = (wn - half_bw).max(1e-4);
            let high = (wn + half_bw).min(1.0 - 1e-4);
            return cheby1_bandpass_bandstop(order, ripple, low, high, FilterType::Bandstop);
        }
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

/// Chebyshev Type I bandpass and bandstop filter design
///
/// Designs digital Chebyshev Type I bandpass or bandstop filters with specified
/// passband ripple and frequency band. The total filter order will be 2*order.
///
/// # Arguments
///
/// * `order` - Filter order (total poles will be 2*order for bandpass/bandstop)
/// * `ripple` - Passband ripple in dB (e.g., 0.5 for 0.5 dB ripple)
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
/// use scirs2_signal::filter::iir::cheby1_bandpass_bandstop;
/// use scirs2_signal::filter::FilterType;
///
/// // Design a 2nd order Chebyshev I bandpass filter (4 poles total)
/// let (b, a) = cheby1_bandpass_bandstop(2, 0.5, 0.2, 0.6, FilterType::Bandpass).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn cheby1_bandpass_bandstop<T, U>(
    order: usize,
    ripple: f64,
    low_freq: T,
    high_freq: U,
    filter_type: FilterType,
) -> SignalResult<FilterCoefficients>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    validate_order(order)?;
    let low_wn = validate_cutoff_frequency(low_freq)?;
    let high_wn = validate_cutoff_frequency(high_freq)?;

    if ripple <= 0.0 {
        return Err(SignalError::ValueError(
            "Ripple must be positive".to_string(),
        ));
    }

    if !matches!(filter_type, FilterType::Bandpass | FilterType::Bandstop) {
        return Err(SignalError::ValueError(
            "Filter _type must be Bandpass or Bandstop".to_string(),
        ));
    }

    if low_wn >= high_wn {
        return Err(SignalError::ValueError(
            "Low frequency must be less than high frequency".to_string(),
        ));
    }

    // Convert ripple from dB to linear
    let epsilon = (10.0_f64.powf(ripple / 10.0) - 1.0).sqrt();

    // Calculate Chebyshev Type I analog prototype poles
    let mut prototype_poles = Vec::with_capacity(order);
    let a = ((1.0 / epsilon + (1.0 / epsilon / epsilon + 1.0)).sqrt()).ln() / order as f64;

    for k in 0..order {
        let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
        let real = -a.sinh() * theta.sin();
        let imag = a.cosh() * theta.cos();
        prototype_poles.push(Complex64::new(real, imag));
    }

    // Prewarp frequencies
    let w1 = prewarp_frequency(low_wn);
    let w2 = prewarp_frequency(high_wn);
    let w0 = (w1 * w2).sqrt(); // Center frequency
    let bw = w2 - w1; // Bandwidth

    let (analog_zeros, analog_poles, gain) = match filter_type {
        FilterType::Bandpass => {
            let mut bp_zeros = Vec::new();
            let mut bp_poles = Vec::new();

            // Transform each prototype pole to bandpass using s -> (s^2 + w0^2)/(s*bw)
            for &pole in &prototype_poles {
                let temp = (pole * bw / 2.0).powi(2) - w0 * w0;
                let sqrt_term = temp.sqrt();

                bp_poles.push(pole * bw / 2.0 + sqrt_term);
                bp_poles.push(pole * bw / 2.0 - sqrt_term);
            }

            // Bandpass has zeros at origin (DC) and infinity
            for _ in 0..order {
                bp_zeros.push(Complex64::new(0.0, 0.0)); // Zero at origin
            }

            let bp_gain = bw.powi(order as i32);
            (bp_zeros, bp_poles, bp_gain)
        }
        FilterType::Bandstop => {
            let mut bs_zeros = Vec::new();
            let mut bs_poles = Vec::new();

            // Transform each prototype pole to bandstop using s -> (s*bw)/(s^2 + w0^2)
            for &pole in &prototype_poles {
                if pole.norm() > 1e-10 {
                    let temp = (bw / (2.0 * pole)).powi(2) - w0 * w0;
                    let sqrt_term = temp.sqrt();

                    bs_poles.push(bw / (2.0 * pole) + sqrt_term);
                    bs_poles.push(bw / (2.0 * pole) - sqrt_term);
                }
            }

            // Bandstop has zeros at ±j*w0
            for _ in 0..order {
                bs_zeros.push(Complex64::new(0.0, w0));
                bs_zeros.push(Complex64::new(0.0, -w0));
            }

            (bs_zeros, bs_poles, 1.0)
        }
        _ => unreachable!(),
    };

    // Apply bilinear transform
    let digital_poles: Vec<_> = analog_poles
        .iter()
        .map(|&pole| bilinear_pole_transform(pole))
        .collect();

    let digital_zeros: Vec<_> = analog_zeros
        .iter()
        .map(|&zero| bilinear_pole_transform(zero))
        .collect();

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
/// ```
/// use scirs2_signal::filter::iir::cheby2;
///
/// // Design a 4th order Chebyshev II lowpass filter with 40 dB stopband attenuation
/// let (b, a) = cheby2(4, 40.0, 0.3, "lowpass").expect("Operation failed");
/// assert_eq!(b.len(), 5); // Order + 1 coefficients
/// assert_eq!(a.len(), 5);
/// ```
#[allow(dead_code)]
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
    let wn = validate_cutoff_frequency(cutoff)?;
    let filter_type = convert_filter_type(filter_type.into())?;

    if attenuation <= 0.0 {
        return Err(SignalError::ValueError(
            "Attenuation must be positive".to_string(),
        ));
    }

    // Redirect bandpass/bandstop to companion function
    match filter_type {
        FilterType::Bandpass => {
            let half_bw = (wn * 0.1).max(0.05).min(wn.min(1.0 - wn) * 0.4);
            let low = (wn - half_bw).max(1e-4);
            let high = (wn + half_bw).min(1.0 - 1e-4);
            return cheby2_bandpass_bandstop(order, attenuation, low, high, FilterType::Bandpass);
        }
        FilterType::Bandstop => {
            let half_bw = (wn * 0.1).max(0.05).min(wn.min(1.0 - wn) * 0.4);
            let low = (wn - half_bw).max(1e-4);
            let high = (wn + half_bw).min(1.0 - 1e-4);
            return cheby2_bandpass_bandstop(order, attenuation, low, high, FilterType::Bandstop);
        }
        _ => {}
    }

    // Convert attenuation from dB to linear
    let epsilon = 1.0 / (10.0_f64.powf(attenuation / 10.0) - 1.0).sqrt();

    // Calculate Chebyshev Type II analog prototype poles and zeros
    let mut poles = Vec::with_capacity(order);
    let mut zeros = Vec::with_capacity(order);

    // Calculate the parameter related to ripple
    let a = ((epsilon + (epsilon * epsilon + 1.0)).sqrt()).ln() / order as f64;

    // Generate poles for Type II (inverse Chebyshev)
    for k in 0..order {
        let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);

        // For Type II, poles are inverted from Type I
        let real = -a.sinh() * theta.sin();
        let imag = a.cosh() * theta.cos();

        // Invert to get Type II poles
        let pole = Complex64::new(real, imag);
        let inv_pole = 1.0 / pole;
        poles.push(inv_pole);
    }

    // Type II has zeros on the imaginary axis
    for k in 0..order {
        let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
        let zero_imag = 1.0 / theta.cos();
        zeros.push(Complex64::new(0.0, zero_imag));
    }

    // Apply frequency transformation and bilinear transform
    let (analog_zeros, transformed_poles, gain) = match filter_type {
        FilterType::Lowpass => {
            let warped_freq = prewarp_frequency(wn);
            let scaled_poles: Vec<_> = poles.iter().map(|p| p * warped_freq).collect();
            let scaled_zeros: Vec<_> = zeros.iter().map(|z| z * warped_freq).collect();
            (scaled_zeros, scaled_poles, 1.0)
        }
        FilterType::Highpass => {
            let warped_freq = prewarp_frequency(wn);
            let hp_poles: Vec<_> = poles.iter().map(|p| warped_freq / p).collect();
            let hp_zeros: Vec<_> = zeros.iter().map(|z| warped_freq / z).collect();
            (hp_zeros, hp_poles, 1.0)
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

    // Add additional zeros if needed based on filter _type
    let additional_zeros = order.saturating_sub(analog_zeros.len());
    for _ in 0..additional_zeros {
        if filter_type == FilterType::Highpass {
            digital_zeros.push(Complex64::new(1.0, 0.0)); // Zero at z=1 (DC)
        } else {
            digital_zeros.push(Complex64::new(-1.0, 0.0)); // Zero at z=-1 (Nyquist)
        }
    }

    zpk_to_tf(&digital_zeros, &digital_poles, gain)
}

/// Chebyshev Type II bandpass and bandstop filter design
///
/// Designs digital Chebyshev Type II bandpass or bandstop filters with specified
/// stopband attenuation and frequency band. The total filter order will be 2*order.
///
/// # Arguments
///
/// * `order` - Filter order (total poles will be 2*order for bandpass/bandstop)
/// * `attenuation` - Stopband attenuation in dB (e.g., 40.0 for 40 dB)
/// * `low_freq` - Low cutoff frequency (normalized 0..1)
/// * `high_freq` - High cutoff frequency (normalized 0..1)
/// * `filter_type` - Filter type (must be Bandpass or Bandstop)
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::iir::cheby2_bandpass_bandstop;
/// use scirs2_signal::filter::FilterType;
///
/// let (b, a) = cheby2_bandpass_bandstop(2, 40.0, 0.2, 0.6, FilterType::Bandpass).expect("Operation failed");
/// assert_eq!(b.len(), 5); // 2*order + 1
/// ```
#[allow(dead_code)]
pub fn cheby2_bandpass_bandstop<T, U>(
    order: usize,
    attenuation: f64,
    low_freq: T,
    high_freq: U,
    filter_type: FilterType,
) -> SignalResult<FilterCoefficients>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    validate_order(order)?;
    let low_wn = validate_cutoff_frequency(low_freq)?;
    let high_wn = validate_cutoff_frequency(high_freq)?;

    if attenuation <= 0.0 {
        return Err(SignalError::ValueError(
            "Attenuation must be positive".to_string(),
        ));
    }
    if !matches!(filter_type, FilterType::Bandpass | FilterType::Bandstop) {
        return Err(SignalError::ValueError(
            "Filter type must be Bandpass or Bandstop".to_string(),
        ));
    }
    if low_wn >= high_wn {
        return Err(SignalError::ValueError(
            "Low frequency must be less than high frequency".to_string(),
        ));
    }

    // Build Chebyshev II analog prototype poles and zeros
    let epsilon = 1.0 / (10.0_f64.powf(attenuation / 10.0) - 1.0).sqrt();
    let a = ((epsilon + (epsilon * epsilon + 1.0).sqrt()).ln()) / order as f64;

    let mut proto_poles = Vec::with_capacity(order);
    let mut proto_zeros = Vec::with_capacity(order);

    for k in 0..order {
        let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
        let pole = Complex64::new(-a.sinh() * theta.sin(), a.cosh() * theta.cos());
        // Type II: invert poles
        proto_poles.push(Complex64::new(1.0, 0.0) / pole);
        // Zeros on imaginary axis
        let zero_imag = 1.0 / theta.cos();
        proto_zeros.push(Complex64::new(0.0, zero_imag));
    }

    // Prewarp and compute band parameters
    let w1 = prewarp_frequency(low_wn);
    let w2 = prewarp_frequency(high_wn);
    let w0 = (w1 * w2).sqrt();
    let bw = w2 - w1;

    let (analog_zeros, analog_poles, gain) = match filter_type {
        FilterType::Bandpass => {
            let mut bp_poles = Vec::new();
            let mut bp_zeros = Vec::new();

            for &pole in &proto_poles {
                let disc = (pole * bw / 2.0).powi(2) - w0 * w0;
                let sqrt_disc = disc.sqrt();
                bp_poles.push(pole * bw / 2.0 + sqrt_disc);
                bp_poles.push(pole * bw / 2.0 - sqrt_disc);
            }
            for &zero in &proto_zeros {
                let disc = (zero * bw / 2.0).powi(2) - w0 * w0;
                let sqrt_disc = disc.sqrt();
                bp_zeros.push(zero * bw / 2.0 + sqrt_disc);
                bp_zeros.push(zero * bw / 2.0 - sqrt_disc);
            }
            let gain = bw.powi(order as i32);
            (bp_zeros, bp_poles, gain)
        }
        FilterType::Bandstop => {
            let mut bs_poles = Vec::new();
            let mut bs_zeros = Vec::new();

            for &pole in &proto_poles {
                if pole.norm() > 1e-10 {
                    let disc = (bw / (2.0 * pole)).powi(2) - w0 * w0;
                    let sqrt_disc = disc.sqrt();
                    bs_poles.push(bw / (2.0 * pole) + sqrt_disc);
                    bs_poles.push(bw / (2.0 * pole) - sqrt_disc);
                }
            }
            for &zero in &proto_zeros {
                if zero.norm() > 1e-10 {
                    let disc = (bw / (2.0 * zero)).powi(2) - w0 * w0;
                    let sqrt_disc = disc.sqrt();
                    bs_zeros.push(bw / (2.0 * zero) + sqrt_disc);
                    bs_zeros.push(bw / (2.0 * zero) - sqrt_disc);
                }
            }
            // Zeros at ±jw0
            for _ in 0..order {
                bs_zeros.push(Complex64::new(0.0, w0));
                bs_zeros.push(Complex64::new(0.0, -w0));
            }
            (bs_zeros, bs_poles, 1.0)
        }
        _ => unreachable!(),
    };

    let digital_poles: Vec<_> = analog_poles
        .iter()
        .map(|&p| bilinear_pole_transform(p))
        .collect();
    let digital_zeros: Vec<_> = analog_zeros
        .iter()
        .map(|&z| bilinear_pole_transform(z))
        .collect();

    zpk_to_tf(&digital_zeros, &digital_poles, gain)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Basic sanity check: result must have non-empty coefficient vectors and monic denominator.
    fn check_filter_basic(b: &[f64], a: &[f64], label: &str) {
        assert!(!b.is_empty(), "{}: numerator empty", label);
        assert!(!a.is_empty(), "{}: denominator empty", label);
        // Denominator should be monic (normalised)
        assert!(
            (a[0] - 1.0).abs() < 1e-6,
            "{}: denominator not monic: a[0] = {}",
            label,
            a[0]
        );
        // All coefficients must be finite
        for (i, &v) in b.iter().enumerate() {
            assert!(v.is_finite(), "{}: b[{}] = {} is not finite", label, i, v);
        }
        for (i, &v) in a.iter().enumerate() {
            assert!(v.is_finite(), "{}: a[{}] = {} is not finite", label, i, v);
        }
    }

    #[test]
    fn test_cheby2_bandpass() {
        let (b, a) = cheby2_bandpass_bandstop(2, 40.0, 0.2_f64, 0.6_f64, FilterType::Bandpass)
            .expect("cheby2 bandpass failed");
        check_filter_basic(&b, &a, "cheby2 bandpass order=2");
        // Denominator degree must be 2*order = 4 (5 coefficients)
        assert_eq!(a.len(), 5, "cheby2 bandpass denominator length");
    }

    #[test]
    fn test_cheby2_bandstop() {
        let (b, a) = cheby2_bandpass_bandstop(2, 40.0, 0.2_f64, 0.6_f64, FilterType::Bandstop)
            .expect("cheby2 bandstop failed");
        check_filter_basic(&b, &a, "cheby2 bandstop order=2");
    }

    #[test]
    fn test_cheby2_bandpass_invalid_freq() {
        // low >= high should fail
        let result = cheby2_bandpass_bandstop(2, 40.0, 0.6_f64, 0.2_f64, FilterType::Bandpass);
        assert!(result.is_err());
    }
}
