//! Bessel IIR filter design (maximally-flat group delay / linear phase).

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
/// let (b, a) = bessel(4, 0.3, "lowpass").expect("Operation failed");
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

    // Redirect bandpass/bandstop to companion function
    match filter_type {
        FilterType::Bandpass => {
            let half_bw = (wn * 0.1).max(0.05).min(wn.min(1.0 - wn) * 0.4);
            let low = (wn - half_bw).max(1e-4);
            let high = (wn + half_bw).min(1.0 - 1e-4);
            return bessel_bandpass_bandstop(order, low, high, FilterType::Bandpass);
        }
        FilterType::Bandstop => {
            let half_bw = (wn * 0.1).max(0.05).min(wn.min(1.0 - wn) * 0.4);
            let low = (wn - half_bw).max(1e-4);
            let high = (wn + half_bw).min(1.0 - 1e-4);
            return bessel_bandpass_bandstop(order, low, high, FilterType::Bandstop);
        }
        _ => {}
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

/// Bessel bandpass and bandstop filter design
///
/// Designs digital Bessel bandpass or bandstop filters with maximally flat group
/// delay. The total filter order will be 2*order.
///
/// # Arguments
///
/// * `order` - Filter order (1..=8 uses exact poles; higher uses approximation)
/// * `low_freq` - Low cutoff frequency (normalized 0..1)
/// * `high_freq` - High cutoff frequency (normalized 0..1)
/// * `filter_type` - Filter type (must be Bandpass or Bandstop)
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::iir::bessel_bandpass_bandstop;
/// use scirs2_signal::filter::FilterType;
///
/// let (b, a) = bessel_bandpass_bandstop(2, 0.2, 0.6, FilterType::Bandpass).expect("Operation failed");
/// assert_eq!(b.len(), 5);
/// ```
#[allow(dead_code)]
pub fn bessel_bandpass_bandstop(
    order: usize,
    low_freq: f64,
    high_freq: f64,
    filter_type: FilterType,
) -> SignalResult<FilterCoefficients> {
    validate_order(order)?;
    let low_wn = validate_cutoff_frequency(low_freq)?;
    let high_wn = validate_cutoff_frequency(high_freq)?;

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

    // Normalised Bessel prototype poles (same table as in bessel())
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
            let mut poles = Vec::with_capacity(order);
            for k in 0..order {
                let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
                let radius = 1.0 - 0.1 * (order as f64 - 8.0).min(5.0) / 10.0;
                poles.push(Complex64::new(-radius * theta.sin(), radius * theta.cos()));
            }
            poles
        }
    };

    // Prewarp and compute band parameters
    let w1 = prewarp_frequency(low_wn);
    let w2 = prewarp_frequency(high_wn);
    let w0 = (w1 * w2).sqrt();
    let bw = w2 - w1;

    let (analog_zeros, analog_poles, gain) = match filter_type {
        FilterType::Bandpass => {
            let mut bp_poles = Vec::new();
            for &pole in &bessel_poles {
                let disc = (pole * bw / 2.0).powi(2) - w0 * w0;
                let sqrt_disc = disc.sqrt();
                bp_poles.push(pole * bw / 2.0 + sqrt_disc);
                bp_poles.push(pole * bw / 2.0 - sqrt_disc);
            }
            // Bandpass Bessel has zeros at DC
            let bp_zeros: Vec<Complex64> = (0..order).map(|_| Complex64::new(0.0, 0.0)).collect();
            let bp_gain = bw.powi(order as i32);
            (bp_zeros, bp_poles, bp_gain)
        }
        FilterType::Bandstop => {
            let mut bs_poles = Vec::new();
            for &pole in &bessel_poles {
                if pole.norm() > 1e-10 {
                    let disc = (bw / (2.0 * pole)).powi(2) - w0 * w0;
                    let sqrt_disc = disc.sqrt();
                    bs_poles.push(bw / (2.0 * pole) + sqrt_disc);
                    bs_poles.push(bw / (2.0 * pole) - sqrt_disc);
                }
            }
            // Zeros at ±jw0
            let bs_zeros: Vec<Complex64> = (0..order)
                .flat_map(|_| [Complex64::new(0.0, w0), Complex64::new(0.0, -w0)])
                .collect();
            (bs_zeros, bs_poles, 1.0)
        }
        _ => unreachable!(),
    };

    let digital_poles: Vec<_> = analog_poles
        .iter()
        .map(|&p| bilinear_pole_transform(p))
        .collect();
    let mut digital_zeros: Vec<_> = analog_zeros
        .iter()
        .map(|&z| bilinear_pole_transform(z))
        .collect();

    // Balance numerator degree to denominator degree: an all-pole prototype
    // (Bessel) bandpass has its analog DC zeros mapped to z=1 by the bilinear
    // transform; the remaining zeros required to make the transfer function
    // proper appear at z=-1 (Nyquist), matching scipy's lp2bp + bilinear result.
    while digital_zeros.len() < digital_poles.len() {
        digital_zeros.push(Complex64::new(-1.0, 0.0));
    }

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
    fn test_bessel_bandpass() {
        let (b, a) = bessel_bandpass_bandstop(2, 0.2, 0.6, FilterType::Bandpass)
            .expect("bessel bandpass failed");
        check_filter_basic(&b, &a, "bessel bandpass order=2");
        // Denominator degree must be 2*order = 4 (5 coefficients)
        assert_eq!(a.len(), 5, "bessel bandpass denominator length");
    }

    #[test]
    fn test_bessel_bandstop() {
        let (b, a) = bessel_bandpass_bandstop(2, 0.2, 0.6, FilterType::Bandstop)
            .expect("bessel bandstop failed");
        check_filter_basic(&b, &a, "bessel bandstop order=2");
    }

    #[test]
    fn test_bessel_bandpass_order_4() {
        let (b, a) = bessel_bandpass_bandstop(4, 0.1, 0.4, FilterType::Bandpass)
            .expect("bessel bandpass order 4 failed");
        check_filter_basic(&b, &a, "bessel bandpass order=4");
        // Denominator degree must be 2*order = 8 (9 coefficients)
        assert_eq!(a.len(), 9, "bessel bandpass order=4 denominator length");
    }
}
