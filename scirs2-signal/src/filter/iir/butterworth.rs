//! Butterworth IIR filter design (maximally-flat-magnitude passband).

use super::zpk_to_tf;
use crate::error::{SignalError, SignalResult};
use crate::filter::common::{
    math::{add_digital_zeros, bilinear_pole_transform, butterworth_poles, prewarp_frequency},
    validation::{convert_filter_type, validate_cutoff_frequency, validate_order},
    FilterCoefficients, FilterType, FilterTypeParam,
};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{Float, NumCast};
use std::fmt::Debug;

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
/// let (b, a) = butter(4, 0.2, FilterType::Lowpass).expect("Operation failed");
///
/// // Using string parameter
/// let (b, a) = butter(4, 0.2, "lowpass").expect("Operation failed");
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

    // Step 4: Compute the correct digital gain by evaluating at the
    // appropriate normalisation point:
    //   lowpass  -> z = 1 (DC, frequency = 0)
    //   highpass -> z = -1 (Nyquist)
    let eval_z = match filter_type {
        FilterType::Lowpass => Complex64::new(1.0, 0.0),
        FilterType::Highpass => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(1.0, 0.0),
    };

    // H(z) = gain * prod(z - z_k) / prod(z - p_k)
    // We want |H(eval_z)| = 1, so compute the ratio and invert.
    let num_val: Complex64 = digital_zeros
        .iter()
        .fold(Complex64::new(1.0, 0.0), |acc, &z| acc * (eval_z - z));
    let den_val: Complex64 = digital_poles
        .iter()
        .fold(Complex64::new(1.0, 0.0), |acc, &p| acc * (eval_z - p));
    let ratio = num_val / den_val;
    let digital_gain = if ratio.norm() > 1e-30 {
        1.0 / ratio.norm()
    } else {
        gain
    };

    // Step 5: Convert poles and zeros to transfer function coefficients
    zpk_to_tf(&digital_zeros, &digital_poles, digital_gain)
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
/// let (b, a) = butter_bandpass_bandstop(4, 0.1, 0.4, FilterType::Bandpass).expect("Operation failed");
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
