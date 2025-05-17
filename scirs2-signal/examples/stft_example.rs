// use ndarray::Array1;
// use num_complex::Complex64;
use scirs2_signal::stft::ShortTimeFft;
use scirs2_signal::waveforms::chirp;
use scirs2_signal::window;
use std::error::Error;
// use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Short-Time Fourier Transform (STFT) Example");
    println!("-------------------------------------------");

    // Create a chirp signal (frequency sweeping from 100 Hz to 300 Hz)
    let fs = 1000.0; // 1 kHz sampling rate
    let duration = 2.0; // 2 seconds
    let n = (fs * duration) as usize;

    println!("Generating chirp signal...");
    println!("  Sampling rate: {} Hz", fs);
    println!("  Duration: {} seconds", duration);
    println!("  Number of samples: {}", n);
    println!("  Frequency range: 100 Hz to 300 Hz");

    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal = chirp(&t, 100.0, duration, 300.0, "linear", 0.0)?;

    // Window parameters
    let window_length = 256;
    let hop_size = 64;

    println!("\nSTFT parameters:");
    println!(
        "  Window length: {} samples ({} ms)",
        window_length,
        window_length as f64 * 1000.0 / fs
    );
    println!(
        "  Hop size: {} samples ({} ms)",
        hop_size,
        hop_size as f64 * 1000.0 / fs
    );
    println!("  Window type: Hann");

    // Create window
    let hann_window = window::hann(window_length, true)?;

    // Create ShortTimeFft instance
    let stft = ShortTimeFft::new(
        &hann_window,
        hop_size,
        fs,
        Some("onesided"),
        Some(512), // zero-padded FFT length
        None,
        Some("magnitude"),
        None,
    )?;

    println!("  FFT size: {} points", stft.mfft);
    println!("  Frequency resolution: {:.2} Hz", stft.delta_f());
    println!("  Time resolution: {:.2} ms", stft.delta_t() * 1000.0);

    // Compute STFT
    println!("\nComputing STFT...");
    let stft_result = stft.stft(&signal)?;

    println!(
        "  STFT matrix shape: {} frequency bins × {} time frames",
        stft_result.shape()[0],
        stft_result.shape()[1]
    );

    // Compute spectrogram (squared magnitude)
    println!("Computing spectrogram...");
    let spectrogram = stft.spectrogram(&signal)?;

    // Find maximum value in spectrogram
    let mut max_value = 0.0;
    let mut max_freq = 0.0;
    let mut max_time = 0.0;

    for i in 0..spectrogram.shape()[0] {
        for j in 0..spectrogram.shape()[1] {
            if spectrogram[[i, j]] > max_value {
                max_value = spectrogram[[i, j]];
                max_freq = i as f64 * stft.delta_f();
                max_time = j as f64 * stft.delta_t();
            }
        }
    }

    println!(
        "  Maximum energy at: {:.2} Hz, {:.2} seconds (value: {:.2})",
        max_freq, max_time, max_value
    );

    // Compute spectrogram statistics
    let spec_mean = spectrogram.sum() / (spectrogram.shape()[0] * spectrogram.shape()[1]) as f64;
    println!("  Average spectrogram energy: {:.6}", spec_mean);

    // Reconstruct signal from STFT
    println!("\nReconstructing signal from STFT...");
    let reconstructed = stft.istft(&stft_result, None, Some(n))?;

    // Calculate reconstruction error
    let mut mse = 0.0;
    for i in window_length..n - window_length {
        mse += (signal[i] - reconstructed[i]).powi(2);
    }
    mse /= (n - 2 * window_length) as f64;

    println!("  Mean squared error (excluding edges): {:.6e}", mse);
    println!("  Reconstruction complete!");

    // Test creating a ShortTimeFft from a window
    println!("\nCreating ShortTimeFft from a window function...");
    let stft2 = ShortTimeFft::from_window(
        "hamming",
        fs,
        256,
        192,
        Some("onesided"),
        Some(512),
        Some("psd"),
        None,
    )?;

    println!("  Created successfully!");
    println!("  Hop size: {} samples", stft2.hop);
    println!("  Is invertible: {}", stft2.invertible());

    // Test creating a self-dual window
    println!("\nCreating a window that equals its dual (COLA window)...");
    let cola_window = scirs2_signal::stft::create_cola_window(256, 64)?;

    println!("  COLA window created successfully!");
    println!("  Window length: {} samples", cola_window.len());

    let _stft3 = ShortTimeFft::from_win_equals_dual(&cola_window, 64, fs, None, None, None, None)?;

    println!("  Created STFT with self-dual window successfully!");

    println!("\nAll tests passed successfully!");

    Ok(())
}
