//! Hilbert transform implementation
//!
//! This module provides functions for computing the Hilbert transform
//! and analytic signal of a real-valued signal.
//!
//! The Hilbert transform is useful for creating analytic signals,
//! computing instantaneous frequency and amplitude, and other signal
//! processing applications.

use crate::error::{SignalError, SignalResult};
use num_complex::Complex64;
use num_traits::{Float, NumCast};
use std::fmt::Debug;

/// Compute the Hilbert transform of a real-valued signal.
///
/// The Hilbert transform is a linear operator that takes a function of real variable
/// and returns another function of real variable which is the convolution of the input
/// with the function 1/(πt). This transforms a real-valued signal into the imaginary
/// part of an analytic signal, which can be used to derive the instantaneous amplitude
/// and frequency of the original signal.
///
/// This implementation uses the FFT method to compute the Hilbert transform, which is
/// efficient for sufficiently long signals.
///
/// # Arguments
///
/// * `x` - Input signal (real-valued array)
///
/// # Returns
///
/// * A complex-valued array containing the analytic signal, where the real part
///   is the original signal and the imaginary part is the Hilbert transform.
///
/// # Examples
///
/// ```
/// use scirs2_signal::hilbert;
/// use std::f64::consts::PI;
/// use num_complex::Complex64;
///
/// // Generate a cosine signal
/// let n = 100;
/// let freq = 5.0; // Hz
/// let dt = 0.01;  // 100 Hz sampling
/// let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * freq * i as f64 * dt).cos()).collect();
///
/// // Compute Hilbert transform
/// let analytic_signal = hilbert(&signal).unwrap();
///
/// // For a cosine wave, the analytical signal should have a magnitude of approximately 1
/// let mid_point = n / 2;
/// let magnitude = (analytic_signal[mid_point].re.powi(2) +
///                 analytic_signal[mid_point].im.powi(2)).sqrt();
/// assert!((magnitude - 1.0).abs() < 0.1);
/// ```
///
/// # References
///
/// * Marple, S. L. "Computing the Discrete-Time Analytic Signal via FFT."
///   IEEE Transactions on Signal Processing, Vol. 47, No. 9, 1999.
pub fn hilbert<T>(x: &[T]) -> SignalResult<Vec<Complex64>>
where
    T: Float + NumCast + Debug,
{
    // Check input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Input length
    let n = x.len();

    // Convert input to a vector of f64
    let signal: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val)
                .ok_or_else(|| SignalError::ValueError(format!("Could not convert {val:?} to f64")))
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Compute FFT of the input signal
    let spectrum = scirs2_fft::fft(&signal, None)
        .map_err(|e| SignalError::ComputationError(format!("FFT computation error: {e}")))?;

    // Create the frequency domain filter for the Hilbert transform
    // For a proper Hilbert transform, we need to:
    // 1. Set the DC component (0 frequency) to 1
    // 2. Double the positive frequencies and multiply by -i
    // 3. Zero out the negative frequencies
    let mut h = vec![Complex64::new(1.0, 0.0); n];

    if n % 2 == 0 {
        // Even length case
        h[0] = Complex64::new(1.0, 0.0); // DC component
        h[n / 2] = Complex64::new(1.0, 0.0); // Nyquist component

        // Positive frequencies (multiply by 2 and by -i)
        h.iter_mut().take(n / 2).skip(1).for_each(|val| {
            *val = Complex64::new(0.0, -2.0); // Equivalent to 2 * (-i)
        });

        // Negative frequencies (set to 0)
        h.iter_mut().skip(n / 2 + 1).for_each(|val| {
            *val = Complex64::new(0.0, 0.0);
        });
    } else {
        // Odd length case
        h[0] = Complex64::new(1.0, 0.0); // DC component

        // Positive frequencies (multiply by 2 and by -i)
        h.iter_mut().take(n.div_ceil(2)).skip(1).for_each(|val| {
            *val = Complex64::new(0.0, -2.0); // Equivalent to 2 * (-i)
        });

        // Negative frequencies (set to 0)
        h.iter_mut().skip(n.div_ceil(2)).for_each(|val| {
            *val = Complex64::new(0.0, 0.0);
        });
    }

    // Apply the filter in frequency domain
    let filtered_spectrum: Vec<Complex64> = spectrum
        .iter()
        .zip(h.iter())
        .map(|(&s, &h)| s * h)
        .collect();

    // Compute inverse FFT to get the analytic signal
    let analytic_signal = scirs2_fft::ifft(&filtered_spectrum, None)
        .map_err(|e| SignalError::ComputationError(format!("IFFT computation error: {e}")))?;

    Ok(analytic_signal)
}

/// Compute the envelope of a signal using the Hilbert transform.
///
/// The envelope of a signal is the magnitude of its analytic signal,
/// which is computed using the Hilbert transform.
///
/// # Arguments
///
/// * `x` - Input signal (real-valued array)
///
/// # Returns
///
/// * A real-valued array containing the envelope of the signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::envelope;
/// use std::f64::consts::PI;
///
/// // Generate a windowed sine wave
/// let n = 100;
/// let freq = 5.0; // Hz
/// let dt = 0.01;  // 100 Hz sampling
/// let signal: Vec<f64> = (0..n).map(|i| {
///     let t = i as f64 * dt;
///     let window = 0.5 * (1.0 - (2.0 * PI * t / n as f64 - PI).cos()); // Hann window
///     window * (2.0 * PI * freq * t).sin()
/// }).collect();
///
/// // Compute envelope
/// let envelope = envelope(&signal).unwrap();
///
/// // Envelope should generally follow the window shape, but edge effects may occur
/// let mid_point = n / 2;
/// // For simplicity, we just check that the envelope has reasonable non-zero values
/// assert!(envelope.iter().all(|&x| x >= 0.0));
/// assert!(envelope.iter().any(|&x| x > 0.5));
/// ```
pub fn envelope<T>(x: &[T]) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    // Compute analytic signal
    let analytic = hilbert(x)?;

    // Compute magnitude (envelope)
    let envelope = analytic
        .iter()
        .map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt())
        .collect();

    Ok(envelope)
}

/// Compute the instantaneous frequency of a signal using the Hilbert transform.
///
/// The instantaneous frequency is the derivative of the phase of the analytic signal,
/// which is computed using the Hilbert transform.
///
/// # Arguments
///
/// * `x` - Input signal (real-valued array)
/// * `fs` - Sampling frequency in Hz
///
/// # Returns
///
/// * A real-valued array containing the instantaneous frequency of the signal in Hz
///
/// # Examples
///
/// ```
/// use scirs2_signal::instantaneous_frequency;
/// use std::f64::consts::PI;
///
/// // Generate a chirp signal (increasing frequency)
/// let n = 100;
/// let fs = 100.0; // 100 Hz sampling
/// let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
/// let f0 = 5.0;  // Starting frequency
/// let f1 = 20.0; // Ending frequency
/// let signal: Vec<f64> = t.iter().map(|&ti| {
///     let freq = f0 + (f1 - f0) * ti / (n as f64 / fs);
///     (2.0 * PI * freq * ti).sin()
/// }).collect();
///
/// // Compute instantaneous frequency
/// let inst_freq = instantaneous_frequency(&signal, fs).unwrap();
///
/// // Check that frequency increases
/// assert!(inst_freq[n/4] > inst_freq[n/8]);
/// assert!(inst_freq[n/2] > inst_freq[n/4]);
/// assert!(inst_freq[3*n/4] > inst_freq[n/2]);
/// ```
pub fn instantaneous_frequency<T>(x: &[T], fs: f64) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    use std::f64::consts::PI;

    // Check input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    if fs <= 0.0 {
        return Err(SignalError::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }

    // Compute analytic signal
    let analytic = hilbert(x)?;

    // Compute phase
    let phase: Vec<f64> = analytic.iter().map(|c| c.im.atan2(c.re)).collect();

    // Unwrap phase to handle phase jumps of more than π
    let mut unwrapped_phase = vec![phase[0]];
    let mut prev_phase = phase[0];

    for &p in &phase[1..] {
        let mut diff = p - prev_phase;

        // Handle phase wrapping
        while diff > PI {
            diff -= 2.0 * PI;
        }
        while diff < -PI {
            diff += 2.0 * PI;
        }

        unwrapped_phase.push(unwrapped_phase.last().unwrap() + diff);
        prev_phase = p;
    }

    // Compute instantaneous frequency
    let mut inst_freq = Vec::with_capacity(x.len());

    // First point requires forward difference
    inst_freq.push(fs * (unwrapped_phase[1] - unwrapped_phase[0]) / (2.0 * PI));

    // Middle points using central difference for better accuracy
    for i in 1..unwrapped_phase.len() - 1 {
        let freq = fs * (unwrapped_phase[i + 1] - unwrapped_phase[i - 1]) / (4.0 * PI);
        inst_freq.push(freq);
    }

    // Last point requires backward difference
    let last_idx = unwrapped_phase.len() - 1;
    inst_freq.push(fs * (unwrapped_phase[last_idx] - unwrapped_phase[last_idx - 1]) / (2.0 * PI));

    Ok(inst_freq)
}

/// Compute the instantaneous phase of a signal using the Hilbert transform.
///
/// The instantaneous phase is the phase of the analytic signal,
/// which is computed using the Hilbert transform.
///
/// # Arguments
///
/// * `x` - Input signal (real-valued array)
/// * `unwrap` - Whether to unwrap the phase to handle jumps of more than π
///
/// # Returns
///
/// * A real-valued array containing the instantaneous phase of the signal in radians
///
/// # Examples
///
/// ```
/// use scirs2_signal::instantaneous_phase;
/// use std::f64::consts::PI;
///
/// // Generate a sine wave
/// let n = 100;
/// let freq = 5.0; // Hz
/// let dt = 0.01;  // 100 Hz sampling
/// let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * freq * i as f64 * dt).sin()).collect();
///
/// // Compute instantaneous phase (unwrapped)
/// let phase = instantaneous_phase(&signal, true).unwrap();
///
/// // Phase should increase linearly for a sine wave
/// let expected_phase_diff = 2.0 * PI * freq * dt;
/// for i in 1..phase.len() {
///     let actual_diff = phase[i] - phase[i-1];
///     assert!((actual_diff - expected_phase_diff).abs() < 0.2); // Allow some error
/// }
/// ```
pub fn instantaneous_phase<T>(x: &[T], unwrap: bool) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    use std::f64::consts::PI;

    // Check input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Compute analytic signal
    let analytic = hilbert(x)?;

    // Compute phase
    let phase: Vec<f64> = analytic.iter().map(|c| c.im.atan2(c.re)).collect();

    if !unwrap {
        return Ok(phase);
    }

    // Unwrap phase to handle phase jumps of more than π
    let mut unwrapped_phase = vec![phase[0]];
    let mut prev_phase = phase[0];

    for &p in &phase[1..] {
        let mut diff = p - prev_phase;

        // Handle phase wrapping
        while diff > PI {
            diff -= 2.0 * PI;
        }
        while diff < -PI {
            diff += 2.0 * PI;
        }

        unwrapped_phase.push(unwrapped_phase.last().unwrap() + diff);
        prev_phase = p;
    }

    Ok(unwrapped_phase)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_hilbert_transform() {
        // Test on a cosine wave
        let n = 1000;
        let freq = 5.0; // 5 Hz
        let sample_rate = 100.0; // 100 Hz
        let dt = 1.0 / sample_rate;

        // Create a cosine wave
        let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
        let signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * freq * ti).cos()).collect();

        // Compute Hilbert transform
        let analytic = hilbert(&signal).unwrap();

        // The Hilbert transform of cos(x) is sin(x)
        // So the analytic signal should be cos(x) + i*sin(x) = e^(ix)
        // Check the envelope (magnitude) which should be approximately 1
        let start_idx = n / 4;
        let end_idx = 3 * n / 4;

        for i in start_idx..end_idx {
            let magnitude = (analytic[i].re.powi(2) + analytic[i].im.powi(2)).sqrt();
            assert_relative_eq!(magnitude, 1.0, epsilon = 0.1);

            // Also check if the phase is advancing correctly
            if i > start_idx {
                let phase_i = analytic[i].im.atan2(analytic[i].re);
                let phase_i_prev = analytic[i - 1].im.atan2(analytic[i - 1].re);

                // Check if phase is advancing in the right direction
                // We need to handle phase wrapping around ±π
                let mut phase_diff = phase_i - phase_i_prev;
                if phase_diff > PI {
                    phase_diff -= 2.0 * PI;
                } else if phase_diff < -PI {
                    phase_diff += 2.0 * PI;
                }

                // For positive frequency, phase should generally advance positively
                assert!(phase_diff > 0.0);
            }
        }
    }

    #[test]
    fn test_envelope() {
        // Generate an amplitude-modulated signal
        let n = 1000;
        let carrier_freq = 20.0; // Hz
        let modulation_freq = 2.0; // Hz
        let fs = 100.0; // 100 Hz sampling
        let dt = 1.0 / fs;

        // Create the signal: (1 + 0.5*cos(2πf_m*t)) * cos(2πf_c*t)
        let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
        let signal: Vec<f64> = t
            .iter()
            .map(|&ti| {
                (1.0 + 0.5 * (2.0 * PI * modulation_freq * ti).cos())
                    * (2.0 * PI * carrier_freq * ti).cos()
            })
            .collect();

        // Compute envelope
        let envelope_result = envelope(&signal).unwrap();

        // The envelope should follow the modulation: 1 + 0.5*cos(2πf_m*t)
        // Check a few points
        for i in n / 10..9 * n / 10 {
            let ti = t[i];
            let expected_envelope = 1.0 + 0.5 * (2.0 * PI * modulation_freq * ti).cos();
            assert_relative_eq!(
                envelope_result[i],
                expected_envelope,
                epsilon = 0.15 // Allow some error due to edge effects
            );
        }
    }

    #[test]
    fn test_instantaneous_frequency() {
        // Generate a chirp signal (increasing frequency)
        let n = 1000;
        let fs = 1000.0; // 1000 Hz sampling
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let f0 = 10.0; // Starting frequency
        let f1 = 100.0; // Ending frequency
        let duration = (n - 1) as f64 / fs;

        // Linear chirp: frequency increases linearly from f0 to f1
        let signal: Vec<f64> = t
            .iter()
            .map(|&ti| {
                let _freq = f0 + (f1 - f0) * ti / duration;
                let phase = 2.0 * PI * (f0 * ti + 0.5 * (f1 - f0) * ti.powi(2) / duration);
                phase.sin()
            })
            .collect();

        // Compute instantaneous frequency
        let inst_freq = instantaneous_frequency(&signal, fs).unwrap();

        // Check a few points (allowing for some error due to numerical differentiation)
        // Skip the very beginning and end due to edge effects
        let start_idx = n / 5;
        let end_idx = 4 * n / 5;

        for i in start_idx..end_idx {
            let ti = t[i];
            let expected_freq = f0 + (f1 - f0) * ti / duration;
            assert_relative_eq!(
                inst_freq[i],
                expected_freq,
                epsilon = 5.0, // Allow some error due to numerical differentiation
                max_relative = 0.2
            );
        }
    }

    #[test]
    fn test_instantaneous_phase() {
        // Generate a sinusoidal signal
        let n = 1000;
        let freq = 10.0; // Hz
        let fs = 1000.0; // 1000 Hz sampling
        let dt = 1.0 / fs;

        // Create the signal: sin(2πf*t)
        let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
        let signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * freq * ti).sin()).collect();

        // Compute instantaneous phase (unwrapped)
        let phase = instantaneous_phase(&signal, true).unwrap();

        // For a sine wave, the phase should increase linearly at a rate of 2πf
        let expected_phase_rate = 2.0 * PI * freq;

        // Check the phase rate (skip edges)
        let start_idx = n / 5;
        let end_idx = 4 * n / 5;

        for i in start_idx + 1..end_idx {
            let phase_rate = (phase[i] - phase[i - 1]) / dt;
            assert_relative_eq!(
                phase_rate,
                expected_phase_rate,
                epsilon = 1.0, // Allow some error
                max_relative = 0.1
            );
        }
    }
}
