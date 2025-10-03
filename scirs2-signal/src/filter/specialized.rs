// Specialized filter designs
//
// This module provides specialized filter designs including notch filters, comb filters,
// allpass filters, and other special-purpose filters for specific signal processing
// applications such as noise removal, echo cancellation, and phase shifting.

use super::common::{validation::validate_cutoff_frequency, FilterCoefficients};
use super::transform::zpk_to_tf;
use crate::error::{SignalError, SignalResult};
use crate::lti::design::tf as design_tf;
use crate::lti::TransferFunction;
use scirs2_core::numeric::Complex64;

#[allow(unused_imports)]
/// Design a notch filter to remove a specific frequency
///
/// A notch filter provides sharp attenuation at a specific frequency while
/// preserving other frequencies. Commonly used to remove power line interference
/// (50/60 Hz) or other unwanted sinusoidal components.
///
/// # Arguments
///
/// * `notch_freq` - Frequency to notch (normalized from 0 to 1, where 1 is Nyquist)
/// * `quality_factor` - Q factor controlling notch width (higher Q = narrower notch)
///
/// # Returns
///
/// * Filter coefficients (b, a) for the notch filter
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::specialized::notch_filter;
///
/// // Remove 60 Hz interference from signal sampled at 1000 Hz
/// // Normalized frequency = 60 / (1000/2) = 0.12
/// let (b, a) = notch_filter(0.12, 35.0).unwrap();
/// ```
#[allow(dead_code)]
pub fn notch_filter(notch_freq: f64, qualityfactor: f64) -> SignalResult<FilterCoefficients> {
    validate_cutoff_frequency(notch_freq)?;

    if qualityfactor <= 0.0 {
        return Err(SignalError::ValueError(
            "Quality factor must be positive".to_string(),
        ));
    }

    // Convert normalized frequency to angular frequency
    let omega = std::f64::consts::PI * notch_freq;

    // Calculate pole radius and zero locations
    let r = 1.0 - std::f64::consts::PI * notch_freq / qualityfactor;

    // Zeros are on the unit circle at the notch frequency
    let zeros = vec![
        Complex64::new(omega.cos(), omega.sin()),  // e^(j*omega)
        Complex64::new(omega.cos(), -omega.sin()), // e^(-j*omega)
    ];

    // Poles are inside the unit circle at the same angle
    let poles = vec![
        Complex64::new(r * omega.cos(), r * omega.sin()), // r * e^(j*omega)
        Complex64::new(r * omega.cos(), -r * omega.sin()), // r * e^(-j*omega)
    ];

    // Unity gain at DC
    let gain = 1.0;

    zpk_to_tf(&zeros, &poles, gain)
}

/// Design a comb filter for echo/delay effects
///
/// A comb filter creates a series of equally spaced notches or peaks in the
/// frequency response, resembling a comb. Used for echo effects, reverb,
/// and periodic noise removal.
///
/// # Arguments
///
/// * `delay_samples` - Delay in samples (determines comb spacing)
/// * `feedback_gain` - Feedback gain (-1.0 to 1.0, negative for notches)
/// * `feedforward_gain` - Feedforward gain (0.0 to 1.0)
///
/// # Returns
///
/// * Filter coefficients (b, a) for the comb filter
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::specialized::comb_filter;
///
/// // Create echo effect with 100-sample delay
/// let (b, a) = comb_filter(100, 0.5, 1.0).unwrap();
///
/// // Create notch comb for periodic noise removal
/// let (b, a) = comb_filter(50, -0.8, 1.0).unwrap();
/// ```
#[allow(dead_code)]
pub fn comb_filter(
    delay_samples: usize,
    feedback_gain: f64,
    feedforward_gain: f64,
) -> SignalResult<FilterCoefficients> {
    if delay_samples == 0 {
        return Err(SignalError::ValueError(
            "Delay must be greater than 0".to_string(),
        ));
    }

    if feedback_gain.abs() >= 1.0 {
        return Err(SignalError::ValueError(
            "Feedback _gain must be between -1 and 1 for stability".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&feedforward_gain) {
        return Err(SignalError::ValueError(
            "Feedforward _gain must be between 0 and 1".to_string(),
        ));
    }

    // Create coefficients for y[n] = feedforward_gain * x[n] + feedback_gain * y[n-D]
    let mut b = vec![0.0; delay_samples + 1];
    let mut a = vec![0.0; delay_samples + 1];

    // Feedforward path
    b[0] = feedforward_gain;

    // Feedback path
    a[0] = 1.0;
    if delay_samples < a.len() {
        a[delay_samples] = -feedback_gain;
    }

    Ok((b, a))
}

/// Design an allpass filter for phase shifting
///
/// An allpass filter has unity magnitude response at all frequencies but
/// provides frequency-dependent phase shift. Used for phase equalization,
/// delay effects, and reverb design.
///
/// # Arguments
///
/// * `pole_frequency` - Pole frequency (normalized from 0 to 1)
/// * `pole_radius` - Pole radius (0 to 1, closer to 1 = more phase shift)
///
/// # Returns
///
/// * Filter coefficients (b, a) for the allpass filter
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::specialized::allpass_filter;
///
/// // Create allpass filter with 90-degree phase shift around 0.2 normalized frequency
/// let (b, a) = allpass_filter(0.2, 0.9).unwrap();
/// ```
#[allow(dead_code)]
pub fn allpass_filter(pole_frequency: f64, poleradius: f64) -> SignalResult<FilterCoefficients> {
    validate_cutoff_frequency(pole_frequency)?;

    if !(0.0..1.0).contains(&poleradius) {
        return Err(SignalError::ValueError(
            "Pole radius must be between 0 and 1".to_string(),
        ));
    }

    let omega = std::f64::consts::PI * pole_frequency;

    // For real allpass filter: H(z) = (a* + z^-1) / (1 + a*z^-1)
    // where a* is the complex conjugate of the pole
    let pole = Complex64::new(poleradius * omega.cos(), poleradius * omega.sin());

    // Allpass property: zero is the complex conjugate of pole reflected across unit circle
    let zero = 1.0 / pole.conj();

    let zeros = vec![zero];
    let poles = vec![pole];
    let gain = poleradius; // Normalize for unity DC gain

    zpk_to_tf(&zeros, &poles, gain)
}

/// Design a second-order allpass filter section
///
/// A second-order allpass section provides more control over phase response
/// and is commonly used in reverb and phaser designs.
///
/// # Arguments
///
/// * `pole_frequency` - Pole frequency (normalized from 0 to 1)
/// * `pole_radius` - Pole radius (0 to 1)
/// * `pole_angle_offset` - Additional phase offset (radians)
///
/// # Returns
///
/// * Filter coefficients (b, a) for the second-order allpass filter
#[allow(dead_code)]
pub fn allpass_second_order(
    pole_frequency: f64,
    pole_radius: f64,
    pole_angle_offset: f64,
) -> SignalResult<FilterCoefficients> {
    validate_cutoff_frequency(pole_frequency)?;

    if !(0.0..1.0).contains(&pole_radius) {
        return Err(SignalError::ValueError(
            "Pole radius must be between 0 and 1".to_string(),
        ));
    }

    let omega = std::f64::consts::PI * pole_frequency + pole_angle_offset;

    // Create complex conjugate pole pair
    let pole1 = Complex64::new(pole_radius * omega.cos(), pole_radius * omega.sin());
    let pole2 = pole1.conj();

    // Zeros are reflections of poles across unit circle
    let zero1 = 1.0 / pole1.conj();
    let zero2 = 1.0 / pole2.conj();

    let zeros = vec![zero1, zero2];
    let poles = vec![pole1, pole2];
    let gain = pole_radius.powi(2); // Normalize for unity DC gain

    zpk_to_tf(&zeros, &poles, gain)
}

/// Design a Hilbert transform filter (90-degree phase shifter)
///
/// A Hilbert transform filter provides a 90-degree phase shift across
/// all frequencies. Used for analytic signal generation, SSB modulation,
/// and frequency shifting.
///
/// # Arguments
///
/// * `num_taps` - Number of filter taps (should be odd)
///
/// # Returns
///
/// * Filter coefficients for FIR Hilbert transformer
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::specialized::hilbert_filter;
///
/// // Design 65-tap Hilbert transformer
/// let h = hilbert_filter(65).unwrap();
/// ```
#[allow(dead_code)]
pub fn hilbert_filter(numtaps: usize) -> SignalResult<Vec<f64>> {
    if numtaps < 3 {
        return Err(SignalError::ValueError(
            "Number of taps must be at least 3".to_string(),
        ));
    }

    if numtaps.is_multiple_of(2) {
        return Err(SignalError::ValueError(
            "Number of taps should be odd for linear phase".to_string(),
        ));
    }

    let mut h = vec![0.0; numtaps];
    let center = numtaps / 2;

    for (i, item) in h.iter_mut().enumerate() {
        if i == center {
            *item = 0.0; // Central coefficient is always zero
        } else {
            let n = i as i32 - center as i32;
            if n % 2 != 0 {
                // Odd indices get non-zero values
                *item = 2.0 / (std::f64::consts::PI * n as f64);
            } else {
                // Even indices (except center) are zero
                *item = 0.0;
            }
        }
    }

    // Apply Hamming window to reduce sidelobes
    for (i, coeff) in h.iter_mut().enumerate() {
        let window_val =
            0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (numtaps - 1) as f64).cos();
        *coeff *= window_val;
    }

    Ok(h)
}

/// Design a differentiator filter
///
/// A differentiator filter approximates the derivative of the input signal.
/// The ideal frequency response is jω, providing increasing gain with frequency
/// and a 90-degree phase shift.
///
/// # Arguments
///
/// * `num_taps` - Number of filter taps (should be odd)
///
/// # Returns
///
/// * Filter coefficients for FIR differentiator
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::specialized::differentiator_filter;
///
/// // Design 21-tap differentiator
/// let h = differentiator_filter(21).unwrap();
/// ```
#[allow(dead_code)]
pub fn differentiator_filter(numtaps: usize) -> SignalResult<Vec<f64>> {
    if numtaps < 3 {
        return Err(SignalError::ValueError(
            "Number of taps must be at least 3".to_string(),
        ));
    }

    if numtaps.is_multiple_of(2) {
        return Err(SignalError::ValueError(
            "Number of taps should be odd for linear phase".to_string(),
        ));
    }

    let mut h = vec![0.0; numtaps];
    let center = numtaps / 2;

    for (i, item) in h.iter_mut().enumerate() {
        if i == center {
            *item = 0.0; // Central coefficient is always zero
        } else {
            let n = i as i32 - center as i32;
            *item = (-1.0_f64).powi(n + 1) / n as f64;
        }
    }

    // Apply Hamming window to reduce high-frequency noise
    for (i, coeff) in h.iter_mut().enumerate() {
        let window_val =
            0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (numtaps - 1) as f64).cos();
        *coeff *= window_val;
    }

    Ok(h)
}

/// Design an integrator filter
///
/// An integrator filter approximates the integral of the input signal.
/// The ideal frequency response is 1/(jω), providing decreasing gain with frequency
/// and a -90-degree phase shift.
///
/// # Arguments
///
/// * `num_taps` - Number of filter taps
///
/// # Returns
///
/// * Filter coefficients for FIR integrator
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::specialized::integrator_filter;
///
/// // Design 21-tap integrator
/// let h = integrator_filter(21).unwrap();
/// ```
#[allow(dead_code)]
pub fn integrator_filter(numtaps: usize) -> SignalResult<Vec<f64>> {
    if numtaps < 3 {
        return Err(SignalError::ValueError(
            "Number of taps must be at least 3".to_string(),
        ));
    }

    // Simple rectangular integration (cumulative sum approximation)
    let h = vec![1.0; numtaps];

    // Normalize to prevent DC buildup
    let normalized_h: Vec<f64> = h.iter().map(|&x| x / numtaps as f64).collect();

    Ok(normalized_h)
}

/// Design a fractional delay filter
///
/// A fractional delay filter provides non-integer sample delays, useful for
/// interpolation, sample rate conversion, and fine timing adjustments.
///
/// # Arguments
///
/// * `delay` - Fractional delay in samples (can be non-integer)
/// * `num_taps` - Number of filter taps
///
/// # Returns
///
/// * Filter coefficients for fractional delay filter
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::specialized::fractional_delay_filter;
///
/// // Design filter with 2.5 sample delay
/// let h = fractional_delay_filter(2.5, 21).unwrap();
/// ```
#[allow(dead_code)]
pub fn fractional_delay_filter(_delay: f64, numtaps: usize) -> SignalResult<Vec<f64>> {
    if numtaps < 3 {
        return Err(SignalError::ValueError(
            "Number of taps must be at least 3".to_string(),
        ));
    }

    if _delay < 0.0 {
        return Err(SignalError::ValueError(
            "Delay must be non-negative".to_string(),
        ));
    }

    let mut h = vec![0.0; numtaps];
    let center = (numtaps - 1) as f64 / 2.0;

    // Use sinc interpolation for fractional _delay
    for (i, item) in h.iter_mut().enumerate() {
        let n = i as f64 - center;
        let shifted_n = n - _delay;

        if shifted_n.abs() < 1e-10 {
            *item = 1.0; // sinc(0) = 1
        } else {
            let arg = std::f64::consts::PI * shifted_n;
            *item = arg.sin() / arg;
        }
    }

    // Apply Hamming window to reduce sidelobes
    for (i, coeff) in h.iter_mut().enumerate() {
        let window_val =
            0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (numtaps - 1) as f64).cos();
        *coeff *= window_val;
    }

    Ok(h)
}

/// Design a DC blocking filter (high-pass with very low cutoff)
///
/// A DC blocking filter removes the DC component and very low frequencies
/// while preserving the rest of the signal. Commonly used in audio processing.
///
/// # Arguments
///
/// * `pole_location` - Pole location (0 to 1, closer to 1 = lower cutoff)
///
/// # Returns
///
/// * Filter coefficients for DC blocker
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::specialized::dc_blocker;
///
/// // Design DC blocker with very low cutoff
/// let (b, a) = dc_blocker(0.995).unwrap();
/// ```
#[allow(dead_code)]
pub fn dc_blocker(polelocation: f64) -> SignalResult<FilterCoefficients> {
    if polelocation <= 0.0 || polelocation >= 1.0 {
        return Err(SignalError::ValueError(
            "Pole location must be between 0 and 1".to_string(),
        ));
    }

    // Simple first-order highpass: H(z) = (1 - z^-1) / (1 - p*z^-1)
    let b = vec![1.0, -1.0];
    let a = vec![1.0, -polelocation];

    Ok((b, a))
}

/// Design a peak filter for parametric equalization
///
/// A peak filter provides boost or cut at a specific frequency with
/// adjustable gain and bandwidth. Used in graphic and parametric equalizers.
///
/// # Arguments
///
/// * `center_freq` - Center frequency (normalized from 0 to 1)
/// * `gain_db` - Gain in dB (positive for boost, negative for cut)
/// * `bandwidth` - Bandwidth in octaves
///
/// # Returns
///
/// * Filter coefficients for peak filter
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::specialized::peak_filter;
///
/// // Boost 6 dB at 0.3 normalized frequency with 1 octave bandwidth
/// let (b, a) = peak_filter(0.3, 6.0, 1.0).unwrap();
/// ```
#[allow(dead_code)]
pub fn peak_filter(
    center_freq: f64,
    gain_db: f64,
    bandwidth: f64,
) -> SignalResult<FilterCoefficients> {
    validate_cutoff_frequency(center_freq)?;

    if bandwidth <= 0.0 {
        return Err(SignalError::ValueError(
            "Bandwidth must be positive".to_string(),
        ));
    }

    let omega = std::f64::consts::PI * center_freq;
    let a_gain = 10.0_f64.powf(gain_db / 40.0); // Convert dB to linear
    let q = 1.0 / (2.0 * (bandwidth * std::f64::consts::LN_2 / 2.0).sinh());

    let alpha = omega.sin() / (2.0 * q);
    let cos_omega = omega.cos();

    // Peaking EQ coefficients
    let b0 = 1.0 + alpha * a_gain;
    let b1 = -2.0 * cos_omega;
    let b2 = 1.0 - alpha * a_gain;
    let a0 = 1.0 + alpha / a_gain;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha / a_gain;

    // Normalize by a0
    let b = vec![b0 / a0, b1 / a0, b2 / a0];
    let a = vec![1.0, a1 / a0, a2 / a0];

    Ok((b, a))
}

#[allow(dead_code)]
fn tf(num: Vec<f64>, den: Vec<f64>) -> TransferFunction {
    TransferFunction::new(num, den, None).unwrap()
}
