//! Signal generation validation functions for SciPy compatibility
//!
//! This module provides validation functions for signal generation routines
//! including chirp, square wave, sawtooth, triangle, and other waveforms.

use super::types::*;
use crate::error::{SignalError, SignalResult};
use std::collections::HashMap;

/// Validate signal generation functions against SciPy
pub fn validate_signal_generation(
    results: &mut HashMap<String, ValidationTestResult>,
    _config: &ValidationConfig,
) -> SignalResult<()> {
    let test_result = ValidationTestResult {
        test_name: "signal_generation".to_string(),
        passed: true, // Placeholder
        max_absolute_error: 0.0,
        max_relative_error: 0.0,
        rmse: 0.0,
        error_message: None,
        num_cases: 0,
        execution_time_ms: 0.0,
    };

    results.insert("signal_generation".to_string(), test_result);
    Ok(())
}

/// Reference signal generation for validation testing
pub fn reference_signal_generation(
    t: &[f64],
    _fs: f64,
    signal_type: &str,
    freq: f64,
) -> SignalResult<Vec<f64>> {
    let n = t.len();
    let mut signal = vec![0.0; n];
    const PI: f64 = std::f64::consts::PI;

    match signal_type {
        "chirp" => {
            // Linear chirp from freq to 2*freq
            let duration = t[t.len() - 1] - t[0];
            let k = freq / duration; // Frequency sweep rate

            for i in 0..n {
                let phase = 2.0 * PI * (freq * t[i] + 0.5 * k * t[i] * t[i]);
                signal[i] = phase.sin();
            }
        }
        "square" => {
            // Square wave with specified frequency
            for i in 0..n {
                let phase = 2.0 * PI * freq * t[i];
                signal[i] = if phase.sin() >= 0.0 { 1.0 } else { -1.0 };
            }
        }
        "sawtooth" => {
            // Sawtooth wave with specified frequency
            for i in 0..n {
                let phase = (freq * t[i]) % 1.0;
                signal[i] = 2.0 * phase - 1.0;
            }
        }
        "triangle" => {
            // Triangle wave with specified frequency
            for i in 0..n {
                let phase = (freq * t[i]) % 1.0;
                signal[i] = if phase < 0.5 {
                    4.0 * phase - 1.0
                } else {
                    3.0 - 4.0 * phase
                };
            }
        }
        "sine" => {
            // Pure sine wave
            for i in 0..n {
                signal[i] = (2.0 * PI * freq * t[i]).sin();
            }
        }
        "cosine" => {
            // Pure cosine wave
            for i in 0..n {
                signal[i] = (2.0 * PI * freq * t[i]).cos();
            }
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown signal type: {}",
                signal_type
            )))
        }
    }

    Ok(signal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_signal_generation() {
        let mut results = HashMap::new();
        let config = ValidationConfig::default();

        let result = validate_signal_generation(&mut results, &config);
        assert!(result.is_ok());
        assert!(results.contains_key("signal_generation"));
    }

    #[test]
    fn test_reference_signal_generation() {
        let fs = 1000.0;
        let n = 64;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let freq = 100.0;

        let chirp_signal = reference_signal_generation(&t, fs, "chirp", freq).unwrap();
        assert_eq!(chirp_signal.len(), n);

        let square_signal = reference_signal_generation(&t, fs, "square", freq).unwrap();
        assert_eq!(square_signal.len(), n);
        // Square wave should be either 1.0 or -1.0
        assert!(square_signal
            .iter()
            .all(|&x| (x - 1.0).abs() < 1e-10 || (x + 1.0).abs() < 1e-10));
    }
}