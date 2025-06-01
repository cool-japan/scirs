//! Tests for the wavelets module

use super::*;
use approx::assert_relative_eq;
use std::f64::consts::PI;

#[test]
fn test_morlet_wavelet() {
    // Test basic creation
    let points = 32;
    let wavelet = morlet(points, 6.0, 1.0).unwrap();

    // Check correct length
    assert_eq!(wavelet.len(), points);

    // Check symmetry (energy is centered in the wavelet)
    let mid_point = (points - 1) / 2;
    assert!(wavelet[mid_point].norm() > wavelet[0].norm());
    assert!(wavelet[mid_point].norm() > wavelet[points - 1].norm());
}

#[test]
fn test_scale_to_frequency() {
    // Define scales
    let scales: Vec<f64> = (1..32).map(|i| i as f64).collect();
    let dt = 0.01; // 100 Hz sampling rate

    // Define central frequency directly
    let central_freq = 0.85; // Representative value for Morlet with w0=6.0

    // Compute corresponding frequencies
    let freqs = scale_to_frequency(&scales, central_freq, dt).unwrap();

    // Should have same length as scales
    assert_eq!(freqs.len(), scales.len());

    // Frequencies should decrease as scales increase
    for i in 1..freqs.len() {
        assert!(freqs[i] < freqs[i - 1]);
    }

    // Check formula: f = central_freq / (scale * dt)
    for (i, &scale) in scales.iter().enumerate() {
        let expected_freq = central_freq / (scale * dt);
        assert_relative_eq!(freqs[i], expected_freq);
    }
}

#[test]
fn test_cwt_simple_signal() {
    // Create a simple signal (sine wave)
    let n = 128;
    let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 32.0).sin()).collect();

    // Define scales
    let scales: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0];

    // Define wavelet generator function
    let wavelet_fn = |points, scale| morlet(points, 6.0, scale);

    // Compute CWT
    let coeffs = transform::cwt(&signal, wavelet_fn, &scales).unwrap();

    // Shape check
    assert_eq!(coeffs.len(), scales.len());
    assert_eq!(coeffs[0].len(), signal.len());

    // Compute magnitude
    let magnitude = cwt_magnitude(&signal, wavelet_fn, &scales, None).unwrap();

    // Check shape
    assert_eq!(magnitude.len(), scales.len());
    assert_eq!(magnitude[0].len(), signal.len());

    // Check that magnitude values are non-negative
    for row in &magnitude {
        for &val in row {
            assert!(val >= 0.0);
        }
    }
}
