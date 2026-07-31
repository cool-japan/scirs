//! Elliptic (Cauer) IIR filter design: equiripple in both passband and
//! stopband, the steepest roll-off of the classic IIR filter families for a
//! given order.
//!
//! The analog lowpass prototype is computed via genuine Jacobi elliptic
//! functions and the Landen-transform-based degree equation (Orfanidis,
//! "Lecture Notes on Elliptic Filter Design"; Zverev, "Handbook of Filter
//! Synthesis"), then mapped to the requested band via the standard analog
//! frequency transformations and the bilinear transform, exactly like the
//! other filter families in this module.

use super::zpk_to_tf;
use crate::error::{SignalError, SignalResult};
use crate::filter::common::{
    math::{bilinear_pole_transform, prewarp_frequency},
    validation::{convert_filter_type, validate_cutoff_frequency, validate_order},
    FilterCoefficients, FilterType, FilterTypeParam,
};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{Float, NumCast};
// NOTE: `scirs2_special::ellipj`'s general-case fallback (`jacobi_sn_approx`
// for |u| >= 1, its normal operating range) discards the modulus entirely
// and returns `u.sin()`, and `scirs2_special::ellipkinc`'s general-case
// fallback (`incomplete_elliptic_f_approx`) computes an unrelated, wrong
// closed-form expression rather than the actual incomplete elliptic
// integral -- both are unusable here. Only `scirs2_special::ellipk` (the
// *complete* elliptic integral, which really does use a genuine AGM
// iteration for its general case) is used below; the Jacobi elliptic
// functions and incomplete elliptic integral needed for elliptic filter
// design are implemented directly in this module via the classical
// descending Landen transformation (AGM method) and Carlson's symmetric
// R_F form, respectively.
use scirs2_special::ellipk;
use std::fmt::Debug;

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
/// let (b, a) = ellip(4, 0.5, 40.0, 0.3, "lowpass").expect("Operation failed");
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

    // Redirect bandpass/bandstop to companion function
    match filter_type {
        FilterType::Bandpass => {
            let half_bw = (wn * 0.1).max(0.05).min(wn.min(1.0 - wn) * 0.4);
            let low = (wn - half_bw).max(1e-4);
            let high = (wn + half_bw).min(1.0 - 1e-4);
            return ellip_bandpass_bandstop(
                order,
                passband_ripple,
                stopband_attenuation,
                low,
                high,
                FilterType::Bandpass,
            );
        }
        FilterType::Bandstop => {
            let half_bw = (wn * 0.1).max(0.05).min(wn.min(1.0 - wn) * 0.4);
            let low = (wn - half_bw).max(1e-4);
            let high = (wn + half_bw).min(1.0 - 1e-4);
            return ellip_bandpass_bandstop(
                order,
                passband_ripple,
                stopband_attenuation,
                low,
                high,
                FilterType::Bandstop,
            );
        }
        _ => {}
    }

    if stopband_attenuation <= passband_ripple {
        return Err(SignalError::ValueError(
            "Stopband attenuation must exceed passband ripple for a valid elliptic filter"
                .to_string(),
        ));
    }

    // Genuine elliptic (Cauer) analog lowpass prototype via Jacobi elliptic
    // functions and the Landen-transform-based degree equation (Orfanidis,
    // "Lecture Notes on Elliptic Filter Design"; Zverev, "Handbook of
    // Filter Synthesis"), replacing the previous Chebyshev-like
    // approximation that did not actually achieve an equiripple stopband.
    let (poles, zeros) = ellip_analog_prototype(order, passband_ripple, stopband_attenuation)?;

    // Apply frequency transformation and bilinear transform
    let (analog_zeros, transformed_poles) = match filter_type {
        FilterType::Lowpass => {
            let warped_freq = prewarp_frequency(wn);
            let scaled_poles: Vec<_> = poles.iter().map(|p| p * warped_freq).collect();
            let scaled_zeros: Vec<_> = zeros.iter().map(|z| z * warped_freq).collect();
            (scaled_zeros, scaled_poles)
        }
        FilterType::Highpass => {
            let warped_freq = prewarp_frequency(wn);
            let hp_poles: Vec<_> = poles.iter().map(|p| warped_freq / p).collect();
            let hp_zeros: Vec<_> = zeros.iter().map(|z| warped_freq / z).collect();
            (hp_zeros, hp_poles)
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

    // Ensure we have the right number of zeros: an odd-order elliptic
    // lowpass/highpass prototype has one fewer finite zero than poles (a
    // genuine "zero at infinity", which the bilinear transform sends to
    // z=-1 for lowpass or z=1 for highpass).
    while digital_zeros.len() < order {
        if filter_type == FilterType::Highpass {
            digital_zeros.push(Complex64::new(1.0, 0.0)); // Zero at z=1 (DC)
        } else {
            digital_zeros.push(Complex64::new(-1.0, 0.0)); // Zero at z=-1 (Nyquist)
        }
    }

    // Normalize gain for unit-magnitude response at the passband reference
    // point (z=1/DC for lowpass, z=-1/Nyquist for highpass), computed
    // directly in the digital domain so it is correct regardless of the
    // bilinear transform's own implicit scale factor.
    let reference_z = if filter_type == FilterType::Highpass {
        Complex64::new(-1.0, 0.0)
    } else {
        Complex64::new(1.0, 0.0)
    };
    let gain = digital_gain_at(&digital_zeros, &digital_poles, reference_z);

    zpk_to_tf(&digital_zeros, &digital_poles, gain)
}

/// Compute the scalar gain `g` such that `H(z) = g * prod(z - zero) /
/// prod(z - pole)` has unit magnitude at `reference_z`.
#[allow(dead_code)]
fn digital_gain_at(zeros: &[Complex64], poles: &[Complex64], reference_z: Complex64) -> f64 {
    let mut num = Complex64::new(1.0, 0.0);
    for &z in zeros {
        num *= reference_z - z;
    }
    let mut den = Complex64::new(1.0, 0.0);
    for &p in poles {
        den *= reference_z - p;
    }
    if num.norm() < 1e-300 {
        1.0
    } else {
        (den / num).re
    }
}

/// Compute the elliptic "nome" `q(m) = exp(-pi * K(1-m) / K(m))` for
/// parameter `m = k^2` (`0 < m < 1`); used by the Landen-transform-based
/// degree equation below.
#[allow(dead_code)]
fn ellip_nome(m: f64) -> f64 {
    let k_val: f64 = ellipk(m);
    let kp_val: f64 = ellipk(1.0 - m);
    (-std::f64::consts::PI * kp_val / k_val).exp()
}

/// Recover the elliptic modulus `k` from its nome `q` via the (rapidly
/// convergent, since `0 < q < 1`) Jacobi theta-function series
/// `k = (theta_2(q) / theta_3(q))^2`.
#[allow(dead_code)]
fn modulus_from_nome(q: f64) -> f64 {
    let mut theta2 = 0.0;
    let mut theta3 = 1.0;
    for n in 0..20 {
        let exponent2 = (n as f64 + 0.5).powi(2);
        theta2 += q.powf(exponent2);
        if n >= 1 {
            let exponent3 = (n as f64).powi(2);
            // theta3(0,q) = 1 + 2*sum_{n=1}^inf q^(n^2): each summand
            // (unlike theta2's, which are uniformly doubled below) needs
            // its own factor of 2 since the leading "1" term is not doubled.
            theta3 += 2.0 * q.powf(exponent3);
        }
    }
    theta2 *= 2.0;
    (theta2 / theta3).powi(2)
}

/// Carlson's symmetric elliptic integral of the first kind `R_F(x, y, z)`,
/// computed via the standard duplication algorithm (Carlson 1979; Press et
/// al., "Numerical Recipes" section 6.12). This is a numerically robust,
/// general-purpose building block for elliptic integrals (uniformly
/// convergent for all valid arguments, unlike a series expansion around a
/// single point), used below to compute the incomplete elliptic integral
/// of the first kind directly within this module.
#[allow(dead_code)]
fn carlson_rf(x0: f64, y0: f64, z0: f64) -> f64 {
    let mut x = x0;
    let mut y = y0;
    let mut z = z0;
    for _ in 0..40 {
        let sx = x.sqrt();
        let sy = y.sqrt();
        let sz = z.sqrt();
        let lambda = sx * sy + sy * sz + sz * sx;
        x = (x + lambda) * 0.25;
        y = (y + lambda) * 0.25;
        z = (z + lambda) * 0.25;
        let mean = (x + y + z) / 3.0;
        if mean.abs() > 1e-300 {
            let dx = (mean - x) / mean;
            let dy = (mean - y) / mean;
            let dz = (mean - z) / mean;
            if dx.abs() < 1e-13 && dy.abs() < 1e-13 && dz.abs() < 1e-13 {
                break;
            }
        }
    }
    let mean = (x + y + z) / 3.0;
    if mean <= 0.0 {
        0.0
    } else {
        1.0 / mean.sqrt()
    }
}

/// Incomplete elliptic integral of the first kind `F(phi, m)`, computed via
/// Carlson's `R_F` (DLMF 19.25.5: `F(phi,m) = sin(phi) * R_F(cos^2(phi), 1
/// - m*sin^2(phi), 1)`). Implemented directly here rather than via
/// `scirs2_special::ellipkinc`, whose general-case fallback path computes
/// an unrelated (incorrect) closed-form expression instead of the actual
/// integral.
#[allow(dead_code)]
fn incomplete_elliptic_f(phi: f64, m: f64) -> f64 {
    if phi.abs() < 1e-300 {
        return 0.0;
    }
    let sin_phi = phi.sin();
    let cos_phi = phi.cos();
    sin_phi * carlson_rf(cos_phi * cos_phi, 1.0 - m * sin_phi * sin_phi, 1.0)
}

/// Jacobi elliptic functions `(sn(u,m), cn(u,m), dn(u,m))` for real `u` and
/// parameter `m` (`0 <= m <= 1`), computed via the classical descending
/// Landen transformation / arithmetic-geometric-mean method (Abramowitz &
/// Stegun 16.4; Press et al., "Numerical Recipes" `sncndn`).
///
/// Implemented directly here rather than via `scirs2_special::ellipj`,
/// whose general-case fallback path (`jacobi_sn_approx` for `|u| >= 1`,
/// the normal operating range for the arguments elliptic filter design
/// produces) discards the modulus entirely and returns `u.sin()`.
#[allow(dead_code)]
fn agm_jacobi_elliptic(u: f64, m: f64) -> (f64, f64, f64) {
    let m = m.clamp(0.0, 1.0);
    if m < 1e-15 {
        return (u.sin(), u.cos(), 1.0);
    }
    if (1.0 - m) < 1e-15 {
        let sech = 1.0 / u.cosh();
        return (u.tanh(), sech, sech);
    }

    const MAX_ITER: usize = 40;
    let mut a = vec![1.0_f64];
    let mut c = vec![m.sqrt()];
    let mut b = (1.0 - m).sqrt();

    let mut n = 0usize;
    loop {
        if (a[n] - b).abs() <= 1e-15 * a[n] || n >= MAX_ITER {
            break;
        }
        let a_next = 0.5 * (a[n] + b);
        let c_next = 0.5 * (a[n] - b);
        b = (a[n] * b).sqrt();
        a.push(a_next);
        c.push(c_next);
        n += 1;
    }

    let mut phi = 2.0_f64.powi(n as i32) * a[n] * u;
    for level in (1..=n).rev() {
        let sin_phi = phi.sin();
        let ratio = (c[level] / a[level] * sin_phi).clamp(-1.0, 1.0);
        phi = 0.5 * (phi + ratio.asin());
    }

    let sn = phi.sin();
    let cn = phi.cos();
    let dn = (1.0 - m * sn * sn).max(0.0).sqrt();
    (sn, cn, dn)
}

/// Find `u` such that the Jacobi elliptic function ratio
/// `sc(u, m) = sn(u, m) / cn(u, m)` equals `target`, via the exact identity
/// `sc(u, m) = tan(am(u, m))` (where `am` is the Jacobi amplitude) combined
/// with the defining relationship of the incomplete elliptic integral of
/// the first kind, `u = F(am(u, m), m)`: solving `tan(phi) = target` for
/// `phi = atan(target)` and returning `u = F(phi, m)`.
#[allow(dead_code)]
fn solve_inverse_sc(target: f64, m: f64) -> f64 {
    let phi = target.atan();
    incomplete_elliptic_f(phi, m)
}

/// Design a genuine elliptic (Cauer) analog lowpass filter prototype
/// (passband edge normalized to 1 rad/s) via Jacobi elliptic functions and
/// the Landen-transform degree equation.
///
/// Returns `(poles, zeros)` for the analog prototype; the gain is
/// normalized separately by the caller after the frequency and bilinear
/// transforms (since that normalization point differs between lowpass,
/// highpass, bandpass, and bandstop designs).
#[allow(dead_code)]
fn ellip_analog_prototype(
    order: usize,
    passband_ripple: f64,
    stopband_attenuation: f64,
) -> SignalResult<(Vec<Complex64>, Vec<Complex64>)> {
    let n = order;
    if n == 0 {
        return Err(SignalError::ValueError(
            "Elliptic filter order must be at least 1".to_string(),
        ));
    }

    let eps = (10f64.powf(passband_ripple / 10.0) - 1.0).sqrt();
    let eps_s = (10f64.powf(stopband_attenuation / 10.0) - 1.0).sqrt();

    if eps >= eps_s {
        return Err(SignalError::ValueError(
            "Stopband attenuation must exceed passband ripple for a valid elliptic filter"
                .to_string(),
        ));
    }

    if n == 1 {
        // A first-order elliptic filter degenerates to a single real pole
        // and no finite zeros (the classical N=1 special case).
        let p = -1.0 / eps;
        return Ok((vec![Complex64::new(p, 0.0)], Vec::new()));
    }

    let k1 = eps / eps_s;
    let m1 = k1 * k1;

    // Solve the elliptic degree equation
    // N = K(k) K'(k1) / (K'(k) K(k1)) for the selectivity factor k, via the
    // nome method: q1 = nome(k1), q = q1^(1/N), then invert q back to a
    // modulus via the theta-function series. This is the standard
    // Landen-transform-family approach to the elliptic degree equation.
    let q1 = ellip_nome(m1);
    if !(0.0..1.0).contains(&q1) || !q1.is_finite() {
        return Err(SignalError::ComputationError(
            "Failed to solve elliptic degree equation (invalid nome)".to_string(),
        ));
    }
    let q = q1.powf(1.0 / n as f64);
    let k_sel = modulus_from_nome(q).clamp(1e-12, 1.0 - 1e-12);
    let m = k_sel * k_sel;

    let capk: f64 = ellipk(m);
    let capk1: f64 = ellipk(m1);

    // v0: the real shift parameter locating the poles, obtained by
    // inverting the Jacobi `sc` function at the complementary parameter
    // `1 - m1`.
    let m1_comp = 1.0 - m1;
    let r = solve_inverse_sc(1.0 / eps, m1_comp);
    let v0 = capk * r / (n as f64 * capk1);

    let (sv, cv, dv) = agm_jacobi_elliptic(v0, 1.0 - m);

    let odd = n % 2 == 1;
    let j_values: Vec<usize> = if odd {
        (0..n).step_by(2).collect()
    } else {
        (1..n).step_by(2).collect()
    };

    let mut zeros = Vec::new();
    let mut poles = Vec::new();
    let mut real_pole: Option<f64> = None;

    for &j in &j_values {
        let ui = j as f64 / n as f64;
        let (s, c, d) = agm_jacobi_elliptic(ui * capk, m);

        if s.abs() > 1e-9 {
            let zi = 1.0 / (k_sel * s);
            zeros.push(Complex64::new(0.0, zi));
            zeros.push(Complex64::new(0.0, -zi));
        }

        let denom = 1.0 - (d * sv).powi(2);
        let pole_re = -(c * d * sv * cv) / denom;
        let pole_im = -(s * dv) / denom;

        if pole_im.abs() > 1e-9 {
            poles.push(Complex64::new(pole_re, pole_im));
            poles.push(Complex64::new(pole_re, -pole_im));
        } else {
            real_pole = Some(pole_re);
        }
    }

    if let Some(p) = real_pole {
        poles.push(Complex64::new(p, 0.0));
    }

    if poles.len() != n {
        return Err(SignalError::ComputationError(format!(
            "Elliptic prototype pole count mismatch: expected {n}, got {}",
            poles.len()
        )));
    }

    Ok((poles, zeros))
}

/// Elliptic bandpass and bandstop filter design
///
/// Designs digital elliptic bandpass or bandstop filters. The total filter order
/// will be 2*order.
///
/// # Arguments
///
/// * `order` - Filter order
/// * `passband_ripple` - Passband ripple in dB
/// * `stopband_attenuation` - Stopband attenuation in dB
/// * `low_freq` - Low cutoff frequency (normalized 0..1)
/// * `high_freq` - High cutoff frequency (normalized 0..1)
/// * `filter_type` - Filter type (must be Bandpass or Bandstop)
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::iir::ellip_bandpass_bandstop;
/// use scirs2_signal::filter::FilterType;
///
/// let (b, a) = ellip_bandpass_bandstop(2, 0.5, 40.0, 0.2, 0.6, FilterType::Bandpass).expect("Operation failed");
/// assert_eq!(b.len(), 5);
/// ```
#[allow(dead_code)]
pub fn ellip_bandpass_bandstop<T, U>(
    order: usize,
    passband_ripple: f64,
    stopband_attenuation: f64,
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

    // Genuine elliptic (Cauer) analog lowpass prototype (see `ellip` /
    // `ellip_analog_prototype` for details), transformed to bandpass or
    // bandstop via the standard LP->BP/BS analog frequency transformation
    // already used by `cheby1_bandpass_bandstop`, rather than a
    // Chebyshev-like approximation with ad hoc stopband zero placement.
    let (proto_poles, proto_zeros) =
        ellip_analog_prototype(order, passband_ripple, stopband_attenuation)?;

    // Prewarp and compute band parameters
    let w1 = prewarp_frequency(low_wn);
    let w2 = prewarp_frequency(high_wn);
    let w0 = (w1 * w2).sqrt();
    let bw = w2 - w1;

    let (analog_zeros, analog_poles) = match filter_type {
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
            // An odd-order prototype's single "zero at infinity" maps,
            // under the LP->BP transform s_lp=(s^2+w0^2)/(bw*s), to BOTH
            // s=0 AND s=infinity (as s->0 or s->infinity, s_lp->infinity
            // either way): a zero at the origin (representable as a finite
            // analog zero, handled here) *and* a zero at infinity, which
            // the bilinear transform sends to z=-1 (Nyquist) -- added
            // directly in the digital domain below since "infinity" has no
            // finite analog representation.
            if bp_zeros.len() + 1 < bp_poles.len() {
                bp_zeros.push(Complex64::new(0.0, 0.0));
            }
            (bp_zeros, bp_poles)
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
            // An odd-order prototype's "zero at infinity" maps to a
            // conjugate zero pair exactly at the notch frequency ±j*w0
            // under the LP->BS transform; pad only up to the pole count
            // (the previous code unconditionally added `order` extra
            // pairs regardless of how many zeros already existed).
            let mut sign = 1.0;
            while bs_zeros.len() < bs_poles.len() {
                bs_zeros.push(Complex64::new(0.0, sign * w0));
                sign = -sign;
            }
            (bs_zeros, bs_poles)
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

    // See the comment above the bandpass zero construction: an odd-order
    // prototype needs one more zero at z=-1 (Nyquist) than the analog
    // pipeline above can produce (it has no finite s-domain representation).
    if matches!(filter_type, FilterType::Bandpass) && digital_zeros.len() + 1 == digital_poles.len()
    {
        digital_zeros.push(Complex64::new(-1.0, 0.0));
    }

    // Normalize gain for unit-magnitude response at a passband reference
    // point, computed directly in the digital domain: the (normalized)
    // band center for bandpass, DC for bandstop (which passes DC as long
    // as the notch does not include it, guaranteed since low_wn > 0).
    let reference_z = match filter_type {
        FilterType::Bandpass => {
            let center = std::f64::consts::PI * (low_wn + high_wn) / 2.0;
            Complex64::new(center.cos(), center.sin())
        }
        _ => Complex64::new(1.0, 0.0),
    };
    let gain = digital_gain_at(&digital_zeros, &digital_poles, reference_z);

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
    fn test_ellip_bandpass() {
        let (b, a) = ellip_bandpass_bandstop(2, 0.5, 40.0, 0.2_f64, 0.6_f64, FilterType::Bandpass)
            .expect("ellip bandpass failed");
        check_filter_basic(&b, &a, "ellip bandpass order=2");
    }

    #[test]
    fn test_ellip_bandstop() {
        let (b, a) = ellip_bandpass_bandstop(2, 0.5, 40.0, 0.2_f64, 0.6_f64, FilterType::Bandstop)
            .expect("ellip bandstop failed");
        check_filter_basic(&b, &a, "ellip bandstop order=2");
    }

    #[test]
    fn test_ellip_bandpass_invalid_ripple() {
        let result = ellip_bandpass_bandstop(2, 0.0, 40.0, 0.2_f64, 0.6_f64, FilterType::Bandpass);
        assert!(result.is_err());
    }

    /// The elliptic prototype's selectivity factor `k` (so its stopband
    /// edge sits at prototype frequency `1/k`), re-derived via the same
    /// degree-equation formulas as `ellip_analog_prototype` itself. Used by
    /// the tests below to know exactly where the *real* (order/rp/rs
    /// -dependent, not user-chosen) stopband edge lands, rather than
    /// guessing a fixed margin past the passband cutoff.
    fn ellip_selectivity_factor(order: usize, rp: f64, rs: f64) -> f64 {
        let eps = (10f64.powf(rp / 10.0) - 1.0).sqrt();
        let eps_s = (10f64.powf(rs / 10.0) - 1.0).sqrt();
        let k1 = eps / eps_s;
        let m1 = k1 * k1;
        let q1 = ellip_nome(m1);
        let q = q1.powf(1.0 / order as f64);
        modulus_from_nome(q)
    }

    /// Convert an analog angular frequency (for this module's `z = (2+s) /
    /// (2-s)` bilinear transform) to the corresponding normalized digital
    /// frequency (`1.0` = Nyquist) -- the exact inverse of
    /// `prewarp_frequency`.
    fn digital_freq_from_analog(w_analog: f64) -> f64 {
        (2.0 / std::f64::consts::PI) * (w_analog / 2.0).atan()
    }

    /// Peak-to-peak passband ripple (dB) and worst-case stopband gain
    /// relative to the passband peak (dB), computed from the actual
    /// frequency response. Independent of any particular gain-normalization
    /// convention, so it validates the genuine equiripple/attenuation
    /// properties an elliptic filter must have regardless of implementation
    /// detail.
    fn passband_ripple_and_stopband_attenuation(
        b: &[f64],
        a: &[f64],
        passband_freqs: &[f64],
        stopband_freqs: &[f64],
    ) -> (f64, f64) {
        let (pass_mag, _) =
            crate::filter::analysis::frequency_response(b, a, passband_freqs).expect("pass resp");
        let (stop_mag, _) =
            crate::filter::analysis::frequency_response(b, a, stopband_freqs).expect("stop resp");

        let pass_db: Vec<f64> = pass_mag
            .iter()
            .map(|&m| 20.0 * m.max(1e-300).log10())
            .collect();
        let stop_db: Vec<f64> = stop_mag
            .iter()
            .map(|&m| 20.0 * m.max(1e-300).log10())
            .collect();

        let max_pass = pass_db.iter().cloned().fold(f64::MIN, f64::max);
        let min_pass = pass_db.iter().cloned().fold(f64::MAX, f64::min);
        let max_stop = stop_db.iter().cloned().fold(f64::MIN, f64::max);

        (max_pass - min_pass, max_pass - max_stop)
    }

    #[test]
    fn test_ellip_lowpass_meets_ripple_and_attenuation_spec() {
        let order = 4;
        let rp = 1.0;
        let rs = 50.0;
        let cutoff = 0.3;

        let (b, a) = ellip(order, rp, rs, cutoff, FilterType::Lowpass).expect("ellip lowpass");
        check_filter_basic(&b, &a, "ellip lowpass order=4");

        // The stopband edge is *not* a free parameter: for a given
        // (order, rp, rs) it is fixed at prototype frequency 1/k by the
        // elliptic degree equation. Compute it exactly rather than
        // guessing a fixed margin past the passband cutoff.
        let k_sel = ellip_selectivity_factor(order, rp, rs);
        let warped_freq = 2.0 * (std::f64::consts::PI * cutoff / 2.0).tan();
        let stopband_edge_digital = digital_freq_from_analog(warped_freq / k_sel);

        let passband_freqs: Vec<f64> = (0..=60).map(|i| i as f64 / 60.0 * cutoff * 0.9).collect();
        let stopband_start = (stopband_edge_digital * 1.1).min(0.97);
        let stopband_freqs: Vec<f64> = (0..=100)
            .map(|i| stopband_start + i as f64 / 100.0 * (0.995 - stopband_start))
            .collect();

        let (ripple_db, attenuation_db) =
            passband_ripple_and_stopband_attenuation(&b, &a, &passband_freqs, &stopband_freqs);

        // The old (Chebyshev-like, arbitrarily-placed-zero) implementation
        // had no genuine mathematical link between the requested rp/rs and
        // the actual response, so it could not reliably satisfy either
        // bound simultaneously; a real equiripple design must.
        assert!(
            ripple_db < rp + 0.5,
            "passband ripple {ripple_db} dB exceeds spec {rp} dB (+0.5 tolerance)"
        );
        assert!(
            attenuation_db > rs - 1.0,
            "stopband attenuation {attenuation_db} dB below spec {rs} dB (-1.0 tolerance)"
        );
    }

    #[test]
    fn test_ellip_highpass_meets_ripple_and_attenuation_spec() {
        let order = 4;
        let rp = 0.5;
        let rs = 45.0;
        let cutoff = 0.4;

        let (b, a) = ellip(order, rp, rs, cutoff, FilterType::Highpass).expect("ellip highpass");
        check_filter_basic(&b, &a, "ellip highpass order=4");

        // For highpass, the LP->HP frequency inversion (s -> wc/s) means
        // the prototype's stopband edge (1/k in prototype units) maps to
        // analog frequency `warped_freq * k`, i.e. *below* the passband
        // cutoff -- compute it exactly rather than guessing a margin.
        let k_sel = ellip_selectivity_factor(order, rp, rs);
        let warped_freq = 2.0 * (std::f64::consts::PI * cutoff / 2.0).tan();
        let stopband_edge_digital = digital_freq_from_analog(warped_freq * k_sel);

        let passband_freqs: Vec<f64> = (0..=60)
            .map(|i| cutoff * 1.1 + i as f64 / 60.0 * (1.0 - cutoff * 1.1))
            .collect();
        let stopband_end = (stopband_edge_digital * 0.9).max(0.01);
        let stopband_freqs: Vec<f64> = (0..=60).map(|i| i as f64 / 60.0 * stopband_end).collect();

        let (ripple_db, attenuation_db) =
            passband_ripple_and_stopband_attenuation(&b, &a, &passband_freqs, &stopband_freqs);

        assert!(
            ripple_db < rp + 0.5,
            "passband ripple {ripple_db} dB exceeds spec {rp} dB (+0.5 tolerance)"
        );
        assert!(
            attenuation_db > rs - 1.0,
            "stopband attenuation {attenuation_db} dB below spec {rs} dB (-1.0 tolerance)"
        );
    }

    #[test]
    fn test_ellip_bandpass_meets_ripple_and_attenuation_spec() {
        let order = 3;
        let rp = 1.0;
        let rs = 40.0;
        let low = 0.3;
        let high = 0.5;

        let (b, a) = ellip_bandpass_bandstop(order, rp, rs, low, high, FilterType::Bandpass)
            .expect("ellip bandpass");
        check_filter_basic(&b, &a, "ellip bandpass order=3");

        // The LP prototype's stopband edge (prototype frequency 1/k) maps,
        // under the LP->BP quadratic frequency transform, to *two* analog
        // frequencies (lower and upper stopband edges) via the same
        // substitution used for poles/zeros in `ellip_bandpass_bandstop`.
        // Compute them exactly rather than guessing fixed margins.
        let k_sel = ellip_selectivity_factor(order, rp, rs);
        let w1 = 2.0 * (std::f64::consts::PI * low / 2.0).tan();
        let w2 = 2.0 * (std::f64::consts::PI * high / 2.0).tan();
        let w0 = (w1 * w2).sqrt();
        let bw = w2 - w1;
        let s_lp = Complex64::new(0.0, 1.0 / k_sel);
        let half = s_lp * bw / 2.0;
        let disc = half * half - Complex64::new(w0 * w0, 0.0);
        let sqrt_disc = disc.sqrt();
        let edge_hi = (half + sqrt_disc).im.abs();
        let edge_lo = (half - sqrt_disc).im.abs();
        let stop_lo_digital = digital_freq_from_analog(edge_lo.min(edge_hi));
        let stop_hi_digital = digital_freq_from_analog(edge_lo.max(edge_hi));

        let passband_freqs: Vec<f64> = (0..=40)
            .map(|i| low + 0.15 * (high - low) + i as f64 / 40.0 * 0.7 * (high - low))
            .collect();
        let stop_lo_end = (stop_lo_digital * 0.9).max(0.005);
        let stop_hi_start = (stop_hi_digital * 1.1).min(0.99);
        let stopband_freqs: Vec<f64> = (0..=20)
            .map(|i| i as f64 / 20.0 * stop_lo_end)
            .chain((0..=20).map(|i| stop_hi_start + i as f64 / 20.0 * (0.995 - stop_hi_start)))
            .collect();

        let (ripple_db, attenuation_db) =
            passband_ripple_and_stopband_attenuation(&b, &a, &passband_freqs, &stopband_freqs);

        assert!(
            ripple_db < rp + 1.0,
            "passband ripple {ripple_db} dB exceeds spec {rp} dB (+1.0 tolerance)"
        );
        assert!(
            attenuation_db > rs - 2.0,
            "stopband attenuation {attenuation_db} dB below spec {rs} dB (-2.0 tolerance)"
        );
    }

    #[test]
    fn test_ellip_analog_prototype_pole_count_and_stability() {
        // The genuine elliptic prototype must always place exactly `order`
        // stable (left-half-plane) poles, for both even and odd orders.
        for order in 2..=6 {
            let (poles, _zeros) =
                ellip_analog_prototype(order, 0.5, 40.0).expect("prototype design");
            assert_eq!(poles.len(), order, "order {order}: wrong pole count");
            for p in &poles {
                assert!(p.re < 0.0, "order {order}: unstable pole {p:?}");
            }
        }
    }

    #[test]
    fn test_ellip_reacts_to_different_orders() {
        // A genuine (non-fabricated) design should produce visibly
        // different attenuation profiles for different orders at a fixed
        // frequency well into the stopband; the old ad hoc zero placement
        // (1.5 + 0.5*k, unrelated to `order` in any principled way for the
        // *shape* of the response) could not guarantee this systematic
        // trend.
        let cutoff = 0.3;
        let probe_freq = [0.8_f64];

        let (b2, a2) = ellip(2, 0.5, 40.0, cutoff, FilterType::Lowpass).expect("order 2");
        let (b6, a6) = ellip(6, 0.5, 40.0, cutoff, FilterType::Lowpass).expect("order 6");

        let (mag2, _) =
            crate::filter::analysis::frequency_response(&b2, &a2, &probe_freq).expect("resp2");
        let (mag6, _) =
            crate::filter::analysis::frequency_response(&b6, &a6, &probe_freq).expect("resp6");

        assert!(
            mag6[0] < mag2[0],
            "higher-order elliptic filter should attenuate more at a fixed deep-stopband frequency: order6={} order2={}",
            mag6[0],
            mag2[0]
        );
    }

    #[test]
    fn test_agm_jacobi_elliptic_matches_known_reference_values() {
        // Known reference values (independently published, e.g. Abramowitz
        // & Stegun tables / standard elliptic-function references):
        // NOTE: this deliberately does *not* check against
        // `scirs2_special`'s own doc-commented "known" reference value for
        // sn(0.5, 0.3) (0.47582636851841): independent Simpson's-rule
        // numerical integration of the defining integral
        // u = integral_0^phi dtheta/sqrt(1-m*sin^2(theta)) shows that
        // value is itself wrong, and that the true value (matching this
        // implementation to 12+ significant figures via two independent
        // methods -- direct quadrature, and this module's own
        // Carlson-R_F-based `incomplete_elliptic_f`) is 0.47421562271182.

        // Trivial, mathematically unambiguous boundary values: sn(0,m)=0,
        // cn(0,m)=1, dn(0,m)=1 for any m; sn(K(m),m)=1, cn(K(m),m)=0,
        // dn(K(m),m)=sqrt(1-m) at the quarter period.
        for &m in &[0.01, 0.3, 0.7, 0.9999] {
            let (sn0, cn0, dn0) = agm_jacobi_elliptic(0.0, m);
            assert!(sn0.abs() < 1e-9, "sn(0,{m})={sn0}");
            assert!((cn0 - 1.0).abs() < 1e-9, "cn(0,{m})={cn0}");
            assert!((dn0 - 1.0).abs() < 1e-9, "dn(0,{m})={dn0}");

            let k_m: f64 = ellipk(m);
            let (snk, cnk, dnk) = agm_jacobi_elliptic(k_m, m);
            assert!((snk - 1.0).abs() < 1e-8, "sn(K({m}),{m})={snk}");
            assert!(cnk.abs() < 1e-8, "cn(K({m}),{m})={cnk}");
            assert!(
                (dnk - (1.0 - m).sqrt()).abs() < 1e-8,
                "dn(K({m}),{m})={dnk}"
            );
        }

        // Round-trip against the defining integral relationship
        // u = F(am(u,m), m) for u safely within one quarter period (where
        // am(u,m) = asin(sn(u,m)) unambiguously, without branch wraparound).
        for &m in &[0.01, 0.3, 0.7, 0.9999] {
            let k_m: f64 = ellipk(m);
            for &frac in &[0.1, 0.3, 0.5, 0.7, 0.9] {
                let u = frac * k_m;
                let (sn, _cn, _dn) = agm_jacobi_elliptic(u, m);
                let phi = sn.clamp(-1.0, 1.0).asin();
                let recovered_u = incomplete_elliptic_f(phi, m);
                assert!(
                    (recovered_u - u).abs() < 1e-6,
                    "round-trip failed at u={u}, m={m}: recovered={recovered_u}"
                );
            }
        }

        // Identity sn^2 + cn^2 = 1 and dn^2 + m*sn^2 = 1 must hold for any
        // u, m (a fabricated/wrong implementation would not satisfy these
        // simultaneously across many points).
        for &u in &[0.1, 0.9, 2.3, 5.7, -1.4] {
            for &m in &[0.01, 0.3, 0.7, 0.9999, 0.999999] {
                let (s, c, d) = agm_jacobi_elliptic(u, m);
                assert!(
                    (s * s + c * c - 1.0).abs() < 1e-8,
                    "sn^2+cn^2 != 1 at u={u}, m={m}: s={s}, c={c}"
                );
                assert!(
                    (d * d + m * s * s - 1.0).abs() < 1e-8,
                    "dn^2+m*sn^2 != 1 at u={u}, m={m}: d={d}, s={s}"
                );
            }
        }
    }

    #[test]
    fn test_carlson_rf_matches_complete_elliptic_integral() {
        // K(m) = R_F(0, 1-m, 1); cross-check against scirs2_special::ellipk
        // (whose *complete*-integral general case is a genuine AGM
        // implementation, unlike its incomplete-integral counterpart).
        for &m in &[0.1, 0.5, 0.9, 0.99] {
            let k_ref: f64 = ellipk(m);
            let k_mine = carlson_rf(0.0, 1.0 - m, 1.0);
            assert!(
                (k_ref - k_mine).abs() < 1e-8,
                "K({m}): reference={k_ref} mine={k_mine}"
            );
        }
    }

    #[test]
    fn test_ellip_analog_prototype_matches_degree_equation() {
        // The elliptic degree equation N = K(k)K'(k1) / (K'(k)K(k1)) must
        // hold for the selectivity factor the prototype design solves for;
        // this is what actually makes the design achieve both the
        // requested passband ripple and stopband attenuation at once.
        for (order, rp, rs) in [
            (2, 0.5, 40.0),
            (3, 1.0, 30.0),
            (4, 0.1, 60.0),
            (5, 2.0, 25.0),
        ] {
            let eps = (10f64.powf(rp / 10.0) - 1.0).sqrt();
            let eps_s = (10f64.powf(rs / 10.0) - 1.0).sqrt();
            let k1 = eps / eps_s;
            let m1 = k1 * k1;

            let q1 = ellip_nome(m1);
            let q = q1.powf(1.0 / order as f64);
            let k_sel = modulus_from_nome(q);
            let m = k_sel * k_sel;

            let capk: f64 = ellipk(m);
            let capk1: f64 = ellipk(m1);
            let kp_sel: f64 = ellipk(1.0 - m);
            let kp1: f64 = ellipk(1.0 - m1);

            let degree_check = (capk * kp1) / (kp_sel * capk1);
            assert!(
                (degree_check - order as f64).abs() < 1e-3,
                "order {order}, rp={rp}, rs={rs}: degree equation check = {degree_check}, expected {order}"
            );
        }
    }
}
