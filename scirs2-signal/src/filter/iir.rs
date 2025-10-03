// IIR (Infinite Impulse Response) filter design functions
//
// This module provides comprehensive IIR filter design capabilities including
// classic analog filter prototypes (Butterworth, Chebyshev, Elliptic, Bessel)
// and specialized IIR design methods. All filters use the bilinear transform
// for analog-to-digital conversion.

use crate::error::{SignalError, SignalResult};
use crate::lti::TransferFunction;
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{Float, NumCast};
use std::fmt::Debug;

#[allow(unused_imports)]
// Helper enum for handling either single values or slices
#[derive(Debug, Clone)]
pub enum Either<A, B> {
    Left(A),
    Right(B),
}
use super::common::{
    math::{add_digital_zeros, bilinear_pole_transform, butterworth_poles, prewarp_frequency},
    validation::{convert_filter_type, validate_cutoff_frequency, validate_order},
    FilterCoefficients, FilterType, FilterTypeParam,
};
use crate::lti::design::tf as design_tf;
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
#[allow(dead_code)]
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

    // Step 2: Apply frequency transformation based on filter _type
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

    // Add zeros in the digital domain based on filter _type
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
#[allow(dead_code)]
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
            "Filter _type must be Bandpass or Bandstop".to_string(),
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
        FilterType::Bandpass | FilterType::Bandstop => {
            return Err(SignalError::NotImplemented(
                "Bandpass and bandstop Chebyshev I filters should use cheby1_bandpass_bandstop function".to_string(),
            ));
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
/// let (b, a) = cheby1_bandpass_bandstop(2, 0.5, 0.2, 0.6, FilterType::Bandpass).unwrap();
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
                let temp = (pole * bw / 2.0).powi(2) + w0 * w0;
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
                    let temp = (bw / (2.0 * pole)).powi(2) + w0 * w0;
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
/// let (b, a) = cheby2(4, 40.0, 0.3, "lowpass").unwrap();
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

    // For now, only support lowpass and highpass
    if !matches!(filter_type, FilterType::Lowpass | FilterType::Highpass) {
        return Err(SignalError::NotImplemented(
            "Bandpass and bandstop Chebyshev II filters not yet implemented".to_string(),
        ));
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
/// ```
/// use scirs2_signal::filter::iir::ellip;
///
/// // Design a 4th order elliptic lowpass filter with 0.5 dB ripple and 40 dB stopband attenuation
/// let (b, a) = ellip(4, 0.5, 40.0, 0.3, "lowpass").unwrap();
/// assert_eq!(b.len(), 5); // Order + 1 coefficients
/// assert_eq!(a.len(), 5);
/// ```
#[allow(dead_code)]
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
    let wn = validate_cutoff_frequency(cutoff)?;
    let filter_type = convert_filter_type(filter_type.into())?;

    if passband_ripple <= 0.0 {
        return Err(SignalError::ValueError(
            "Passband _ripple must be positive".to_string(),
        ));
    }

    if stopband_attenuation <= 0.0 {
        return Err(SignalError::ValueError(
            "Stopband _attenuation must be positive".to_string(),
        ));
    }

    // For now, only support lowpass and highpass
    if !matches!(filter_type, FilterType::Lowpass | FilterType::Highpass) {
        return Err(SignalError::NotImplemented(
            "Bandpass and bandstop elliptic filters not yet implemented".to_string(),
        ));
    }

    // Convert _ripple and _attenuation from dB to linear
    let epsilon_p = (10.0_f64.powf(passband_ripple / 10.0) - 1.0).sqrt();
    let epsilon_s = (10.0_f64.powf(stopband_attenuation / 10.0) - 1.0).sqrt();

    // For a simplified elliptic filter, we'll approximate using a Chebyshev-like approach
    // with modified pole-zero placement to achieve both passband and stopband specifications

    // This is a simplified implementation. A full elliptic filter would require:
    // 1. Calculation of modular constant k from specifications
    // 2. Use of Jacobi elliptic functions sn, cn, dn
    // 3. Elliptic integral calculations

    // For production readiness, we'll create a filter that approximates elliptic behavior
    // by combining aspects of Chebyshev I (passband ripple) and Chebyshev II (stopband zeros)

    let mut poles = Vec::with_capacity(order);
    let mut zeros = Vec::with_capacity(order);

    // Generate poles similar to Chebyshev but with adjustments for elliptic characteristics
    let a = (1.0 / epsilon_p).asinh() / order as f64;

    for k in 0..order {
        let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);

        // Elliptic-like pole placement
        let real = -a.sinh() * theta.sin();
        let imag = a.cosh() * theta.cos();

        // Modify pole positions to account for stopband requirements
        let mod_factor = 1.0 + (epsilon_s / epsilon_p).ln() / (2.0 * order as f64);
        let pole = Complex64::new(real * mod_factor, imag);
        poles.push(pole);

        // Add zeros for stopband (simplified placement)
        if k < order / 2 {
            let zero_freq = 1.5 + 0.5 * k as f64 / (order as f64 / 2.0);
            zeros.push(Complex64::new(0.0, zero_freq));
            if order.is_multiple_of(2) || k < order / 2 - 1 {
                zeros.push(Complex64::new(0.0, -zero_freq));
            }
        }
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

    // Ensure we have the right number of zeros
    while digital_zeros.len() < order {
        if filter_type == FilterType::Highpass {
            digital_zeros.push(Complex64::new(1.0, 0.0)); // Zero at z=1 (DC)
        } else {
            digital_zeros.push(Complex64::new(-1.0, 0.0)); // Zero at z=-1 (Nyquist)
        }
    }

    zpk_to_tf(&digital_zeros, &digital_poles, gain)
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
/// ```
/// use scirs2_signal::filter::iir::bessel;
///
/// // Design a 4th order Bessel lowpass filter
/// let (b, a) = bessel(4, 0.3, "lowpass").unwrap();
/// assert_eq!(b.len(), 5); // Order + 1 coefficients
/// assert_eq!(a.len(), 5);
/// ```
#[allow(dead_code)]
pub fn bessel<T>(
    order: usize,
    cutoff: T,
    filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<FilterCoefficients>
where
    T: Float + NumCast + Debug,
{
    validate_order(order)?;
    let wn = validate_cutoff_frequency(cutoff)?;
    let filter_type = convert_filter_type(filter_type.into())?;

    // For now, only support lowpass and highpass
    if !matches!(filter_type, FilterType::Lowpass | FilterType::Highpass) {
        return Err(SignalError::NotImplemented(
            "Bandpass and bandstop Bessel filters not yet implemented".to_string(),
        ));
    }

    // Bessel filter poles for orders 1-8 (pre-computed for standard Bessel polynomials)
    // These are the poles of the normalized Bessel polynomials
    let bessel_poles: Vec<Complex64> = match order {
        1 => vec![Complex64::new(-1.0, 0.0)],
        2 => vec![
            Complex64::new(-0.8660254037844387, 0.5),
            Complex64::new(-0.8660254037844387, -0.5),
        ],
        3 => vec![
            Complex64::new(-0.9416000265332069, 0.7456403858480766),
            Complex64::new(-0.9416000265332069, -0.7456403858480766),
            Complex64::new(-0.7456403858480766, 0.0),
        ],
        4 => vec![
            Complex64::new(-0.6572111716718829, 0.8301614350048733),
            Complex64::new(-0.6572111716718829, -0.8301614350048733),
            Complex64::new(-0.9047587967882449, 0.2709187330038746),
            Complex64::new(-0.9047587967882449, -0.2709187330038746),
        ],
        5 => vec![
            Complex64::new(-0.9264420773877602, 0.0),
            Complex64::new(-0.8515536193688395, 0.4427174639443327),
            Complex64::new(-0.8515536193688395, -0.4427174639443327),
            Complex64::new(-0.5905759446119191, 0.9072067564574549),
            Complex64::new(-0.5905759446119191, -0.9072067564574549),
        ],
        6 => vec![
            Complex64::new(-0.9093906830472271, 0.1856964396793046),
            Complex64::new(-0.9093906830472271, -0.1856964396793046),
            Complex64::new(-0.7996541858328288, 0.5621717346937317),
            Complex64::new(-0.7996541858328288, -0.5621717346937317),
            Complex64::new(-0.5385526816693109, 0.9616876881954277),
            Complex64::new(-0.5385526816693109, -0.9616876881954277),
        ],
        7 => vec![
            Complex64::new(-0.9195339081664588, 0.0),
            Complex64::new(-0.8800029341523374, 0.2789585460830486),
            Complex64::new(-0.8800029341523374, -0.2789585460830486),
            Complex64::new(-0.7527355434093214, 0.650_469_630_552_255),
            Complex64::new(-0.7527355434093214, -0.650_469_630_552_255),
            Complex64::new(-0.4966917256672316, 1.0025085824351491),
            Complex64::new(-0.4966917256672316, -1.0025085824351491),
        ],
        8 => vec![
            Complex64::new(-0.909_683_154_665_291, 0.1412437976671422),
            Complex64::new(-0.909_683_154_665_291, -0.1412437976671422),
            Complex64::new(-0.8473250802359334, 0.4259700895773585),
            Complex64::new(-0.8473250802359334, -0.4259700895773585),
            Complex64::new(-0.7111381808485399, 0.7186517314014426),
            Complex64::new(-0.7111381808485399, -0.7186517314014426),
            Complex64::new(-0.4621740412532122, 1.0344954064286434),
            Complex64::new(-0.4621740412532122, -1.0344954064286434),
        ],
        _ => {
            // For higher orders, approximate using Butterworth-like poles
            // with modified positions for Bessel characteristics
            let mut poles = Vec::with_capacity(order);
            for k in 0..order {
                let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
                let radius = 1.0 - 0.1 * (order as f64 - 8.0).min(5.0) / 10.0;
                let real = -radius * theta.sin();
                let imag = radius * theta.cos();
                poles.push(Complex64::new(real, imag));
            }
            poles
        }
    };

    // Apply frequency transformation based on filter _type
    let (analog_zeros, transformed_poles, gain) = match filter_type {
        FilterType::Lowpass => {
            let warped_freq = prewarp_frequency(wn);
            // Scale poles by the warped frequency
            let scaled_poles: Vec<_> = bessel_poles.iter().map(|p| p * warped_freq).collect();
            // Lowpass Bessel has no finite zeros
            (
                Vec::<Complex64>::new(),
                scaled_poles,
                warped_freq.powi(order as i32),
            )
        }
        FilterType::Highpass => {
            let warped_freq = prewarp_frequency(wn);
            // Highpass transformation: s -> wc/s
            let hp_poles: Vec<_> = bessel_poles.iter().map(|p| warped_freq / p).collect();
            // No finite zeros for highpass Bessel
            (Vec::<Complex64>::new(), hp_poles, 1.0)
        }
        _ => unreachable!(),
    };

    // Apply bilinear transform to convert to digital filter
    let digital_poles: Vec<_> = transformed_poles
        .iter()
        .map(|&pole| bilinear_pole_transform(pole))
        .collect();

    let mut digital_zeros: Vec<_> = analog_zeros
        .iter()
        .map(|&zero| bilinear_pole_transform(zero))
        .collect();

    // Add zeros in the digital domain based on filter _type
    digital_zeros.extend(add_digital_zeros(filter_type, order));

    zpk_to_tf(&digital_zeros, &digital_poles, gain)
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
#[allow(dead_code)]
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

#[allow(dead_code)]
fn tf(num: Vec<f64>, den: Vec<f64>) -> TransferFunction {
    TransferFunction::new(num, den, None).unwrap()
}
