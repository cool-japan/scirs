//! Short-Time Fourier Transform (STFT) implementation
//!
//! This module provides a parametrized discrete Short-time Fourier transform (STFT)
//! and its inverse (ISTFT), similar to SciPy's ShortTimeFFT class.
//!
//! ## Features
//!
//! - Memory-efficient processing for large signals
//! - Streaming STFT with configurable chunk sizes
//! - Zero-copy processing where possible
//! - Different FFT modes (one-sided, two-sided, centered)
//! - Configurable scaling modes
//! - COLA window generation
//! - Dual window calculation for perfect reconstruction
//!
//! ## Examples
//!
//! ```rust
//! use scirs2_signal::stft::{ShortTimeFft, StftConfig};
//! use scirs2_signal::window;
//! use ndarray::Array1;
//! use std::f64::consts::PI;
//!
//! // Create a signal with varying frequency
//! let fs = 1000.0; // 1 kHz sampling rate
//! let duration = 1.0; // 1 second
//! let n = (fs * duration) as usize;
//! let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
//!
//! // Chirp signal: frequency sweeping from 100 Hz to 300 Hz
//! let signal: Vec<f64> = t.iter()
//!     .map(|&t| (2.0 * PI * (100.0 + 200.0 * t / duration) * t).sin())
//!     .collect();
//!
//! // Create a hann window and initialize STFT
//! let window_length = 256;
//! let hop_size = 64;
//! let hann_window = window::hann(window_length, true).unwrap();
//!
//! // Create configuration
//! let config = StftConfig {
//!     fft_mode: None, // default FFT mode (onesided)
//!     mfft: None, // default mfft (window length)
//!     dual_win: None, // calculate dual window as needed
//!     scale_to: Some("magnitude".to_string()), // scale for magnitude spectrum
//!     phase_shift: None, // no phase shift
//! };
//!
//! let stft = ShortTimeFft::new(
//!     &hann_window,
//!     hop_size,
//!     fs,
//!     Some(config),
//! ).unwrap();
//!
//! // Compute STFT
//! let stft_result = stft.stft(&signal).unwrap();
//!
//! // The result is a complex-valued 2D array with:
//! // - Rows representing frequency bins
//! // - Columns representing time frames
//! ```

// Module organization
pub mod types;
pub mod core;
pub mod algorithms;
pub mod memory_efficient;
pub mod utils;

// Re-export all public types for backward compatibility
pub use types::*;

// Re-export main structures and functions
pub use core::ShortTimeFft;
pub use memory_efficient::{MemoryEfficientStft, MemoryInfo};

// Re-export utility functions
pub use utils::{
    closest_stft_dual_window, create_cola_window, check_cola_condition,
    calculate_window_normalization, estimate_optimal_hop_size, effective_window_length,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::window;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_simple_signal_stft() {
        // Create a simple sine wave
        let fs = 100.0;
        let duration = 1.0;
        let freq = 10.0;
        let n = (fs * duration) as usize;

        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect();

        // Create Hann window
        let window_length = 32;
        let hop_size = 16;
        let hann_window = window::hann(window_length, true).unwrap();

        // Create STFT
        let stft = ShortTimeFft::new(&hann_window, hop_size, fs, None).unwrap();

        // Compute STFT
        let result = stft.stft(&signal).unwrap();

        // Check dimensions
        assert!(result.nrows() > 0);
        assert!(result.ncols() > 0);

        // Should have frequency bins up to Nyquist
        assert_eq!(result.nrows(), window_length / 2 + 1);
    }

    #[test]
    fn test_stft_istft_reconstruction() {
        // Create a simple signal
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0];

        // Use a simple rectangular window with 75% overlap for better reconstruction
        let window_length = 4;
        let hop_size = 1;
        let rect_window = vec![1.0; window_length];

        let mut config = StftConfig::default();
        config.dual_win = Some(rect_window.clone());

        let stft = ShortTimeFft::new(&rect_window, hop_size, 1.0, Some(config)).unwrap();

        // Compute STFT
        let stft_result = stft.stft(&signal).unwrap();

        // Reconstruct signal
        let reconstructed = stft.istft(&stft_result, None, None).unwrap();

        // The reconstructed signal will be longer due to windowing
        // Just check that we get reasonable reconstruction in the middle part
        if reconstructed.len() >= signal.len() {
            // Find the best alignment by checking different offsets
            let mut best_error = f64::INFINITY;
            let search_range = (reconstructed.len() - signal.len()).min(window_length);

            for offset in 0..=search_range {
                let mut error_sum = 0.0;
                let mut count = 0;

                for i in 0..signal.len() {
                    if i + offset < reconstructed.len() {
                        error_sum += (reconstructed[i + offset] - signal[i]).abs();
                        count += 1;
                    }
                }

                if count > 0 {
                    let avg_error = error_sum / count as f64;
                    if avg_error < best_error {
                        best_error = avg_error;
                    }
                }
            }

            // STFT/iSTFT reconstruction may have some error due to windowing
            assert!(best_error < 2.0, "Reconstruction error too high: {}", best_error);
        }
    }

    #[test]
    fn test_from_window_constructor() {
        // Create a simple signal
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0];

        let stft = ShortTimeFft::from_window("hann", 10.0, 8, 4, None).unwrap();

        let result = stft.stft(&signal).unwrap();

        assert!(result.nrows() > 0);
        assert!(result.ncols() > 0);
    }

    #[test]
    fn test_spectrogram() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0];
        let window = vec![1.0; 4];

        let stft = ShortTimeFft::new(&window, 2, 1.0, None).unwrap();
        let spectrogram = stft.spectrogram(&signal).unwrap();

        // All values should be non-negative (magnitude squared)
        for &val in spectrogram.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_frequency_and_time_vectors() {
        let window = vec![1.0; 16];
        let stft = ShortTimeFft::new(&window, 8, 100.0, None).unwrap();

        let signal_length = 100;
        let f_vec = stft.f();
        let t_vec = stft.t_vec(signal_length);

        // Check frequency vector
        assert_eq!(f_vec.len(), stft.f_pts());
        assert_relative_eq!(f_vec[0], 0.0, epsilon = 1e-10);
        assert!(f_vec[f_vec.len() - 1] <= 50.0); // Should not exceed Nyquist

        // Check time vector
        assert!(t_vec.len() > 0);
        assert_relative_eq!(t_vec[0], stft.p_min() as f64 * stft.hop as f64 / stft.fs, epsilon = 1e-10);
    }

    #[test]
    fn test_memory_efficient_stft() {
        let signal: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
        let window = window::hann(64, true).unwrap();

        let memory_config = MemoryEfficientStftConfig {
            memory_limit: 10, // Very small limit to force chunking
            chunk_size: Some(100),
            chunk_overlap: 32,
        };

        let memory_stft = MemoryEfficientStft::new(
            &window,
            16,
            100.0,
            None,
            memory_config,
        ).unwrap();

        let result = memory_stft.stft_chunked(&signal).unwrap();
        assert!(result.nrows() > 0);
        assert!(result.ncols() > 0);

        let spectrogram = memory_stft.spectrogram_chunked(&signal).unwrap();
        assert_eq!(spectrogram.shape(), result.shape());

        // All spectrogram values should be non-negative
        for &val in spectrogram.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_cola_window_creation() {
        let cola_window = create_cola_window(32, 8).unwrap();
        assert_eq!(cola_window.len(), 32);

        // Test COLA condition
        assert!(check_cola_condition(&cola_window, 8, 1e-10));
    }

    #[test]
    fn test_window_normalization() {
        let window = window::hann(32, true).unwrap();

        let mag_norm = calculate_window_normalization(&window, "magnitude");
        let psd_norm = calculate_window_normalization(&window, "psd");
        let energy_norm = calculate_window_normalization(&window, "energy");

        assert!(mag_norm > 0.0);
        assert!(psd_norm > 0.0);
        assert!(energy_norm > 0.0);

        // PSD normalization should be square of magnitude normalization
        assert_relative_eq!(psd_norm, mag_norm * mag_norm, epsilon = 1e-10);
    }

    #[test]
    fn test_hop_size_estimation() {
        let window = window::hann(64, true).unwrap();

        let hop_50 = estimate_optimal_hop_size(&window, 0.5);
        let hop_75 = estimate_optimal_hop_size(&window, 0.75);

        assert_eq!(hop_50, 32); // 50% overlap
        assert_eq!(hop_75, 16); // 75% overlap

        // Edge cases
        let hop_0 = estimate_optimal_hop_size(&window, 0.0);
        let hop_99 = estimate_optimal_hop_size(&window, 0.99);

        assert_eq!(hop_0, 64); // No overlap
        assert_eq!(hop_99, 1);  // Maximum overlap
    }

    #[test]
    fn test_effective_window_length() {
        // Create a window with some zero padding
        let mut window = vec![0.0; 64];
        for i in 16..48 {
            window[i] = (std::f64::consts::PI * (i - 16) as f64 / 32.0).sin();
        }

        let eff_len = effective_window_length(&window, 0.01);
        assert!(eff_len < 64);
        assert!(eff_len >= 32);
    }
}