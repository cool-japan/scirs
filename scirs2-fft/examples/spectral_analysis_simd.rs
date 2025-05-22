//! Spectral Analysis Application using SIMD-accelerated FFT
//!
//! This example demonstrates a scientific computing application
//! performing spectral analysis on time series data.

use num_complex::Complex64;
use scirs2_fft::{fft_adaptive, fftfreq, ifft_adaptive, simd_support_available, window};
use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    println!("Spectral Analysis Application with SIMD-accelerated FFT");
    println!("======================================================");

    // Check for SIMD support
    let simd_available = simd_support_available();
    println!("SIMD acceleration available: {}", simd_available);

    // Parameters for the analysis
    let sample_rate = 1000.0; // Hz
    let duration = 1.0; // seconds
    let num_samples = (sample_rate * duration) as usize;

    println!("\nGenerating test signal:");
    println!("  - Sample rate: {} Hz", sample_rate);
    println!("  - Duration: {} seconds", duration);
    println!("  - Number of samples: {}", num_samples);

    // Generate a test signal with multiple frequency components
    let time = linspace(0.0, duration, num_samples);
    let signal = generate_test_signal(&time);

    println!("\nSignal contains frequencies: 10 Hz, 50 Hz, 120 Hz, and noise");

    // Perform spectral analysis
    println!("\nPerforming spectral analysis...");
    let start = Instant::now();
    let (frequencies, power_spectrum) = spectral_analysis(&signal, sample_rate);
    let elapsed = start.elapsed();

    println!("Analysis completed in {:?}", elapsed);

    // Find peak frequencies
    let peaks = find_peak_frequencies(&frequencies, &power_spectrum, 0.1);

    println!("\nDetected peak frequencies:");
    for (i, &(freq, power)) in peaks.iter().enumerate() {
        println!("  Peak {}: {:.2} Hz (Power: {:.4})", i + 1, freq, power);
    }

    // Apply a bandpass filter in the frequency domain
    println!("\nApplying a bandpass filter (40-60 Hz)...");
    let start = Instant::now();
    let filtered_signal = bandpass_filter(&signal, sample_rate, 40.0, 60.0);
    let elapsed = start.elapsed();

    println!("Filtering completed in {:?}", elapsed);

    // Compare signal energy before and after filtering
    let original_energy = signal.iter().map(|&x| x * x).sum::<f64>();
    let filtered_energy = filtered_signal.iter().map(|&x| x * x).sum::<f64>();
    let energy_ratio = filtered_energy / original_energy;

    println!("\nSignal energy:");
    println!("  Original signal: {:.4}", original_energy);
    println!("  Filtered signal: {:.4}", filtered_energy);
    println!("  Energy ratio: {:.2}%", energy_ratio * 100.0);

    // Perform time-frequency analysis
    println!("\nPerforming time-frequency analysis...");
    let window_size = 128;
    let overlap = 64;

    let start = Instant::now();
    let (time_points, freq_bins, spectrogram) =
        compute_spectrogram(&signal, sample_rate, window_size, overlap);
    let elapsed = start.elapsed();

    println!("Time-frequency analysis completed in {:?}", elapsed);

    println!("\nSpectrogram dimensions:");
    println!("  Time points: {}", time_points.len());
    println!("  Frequency bins: {}", freq_bins.len());
    println!(
        "  Values range: {:.4} to {:.4}",
        spectrogram.iter().fold(f64::MAX, |a, &b| a.min(b)),
        spectrogram.iter().fold(f64::MIN, |a, &b| a.max(b))
    );

    // Performance comparison
    println!("\nPerformance comparison with larger signals:");
    for &size in &[4096, 16384, 65536, 262144] {
        let large_signal = generate_large_signal(size);

        println!("\nSignal size: {} samples", size);

        // Measure FFT performance
        let start = Instant::now();
        let _ = fft_adaptive(&large_signal, None, None);
        let elapsed = start.elapsed();

        println!("  FFT computation time: {:?}", elapsed);

        // Calculate operations per second
        let ops_per_sec = size as f64 * (size as f64).log2() / elapsed.as_secs_f64();
        println!("  Operations per second: {:.2e}", ops_per_sec);
    }
}

// Generate linearly spaced points
fn linspace(start: f64, end: f64, num: usize) -> Vec<f64> {
    let step = (end - start) / (num - 1) as f64;
    (0..num).map(|i| start + i as f64 * step).collect()
}

// Generate a test signal with multiple frequency components
fn generate_test_signal(time: &[f64]) -> Vec<f64> {
    let mut signal = Vec::with_capacity(time.len());

    for &t in time {
        // Components at 10 Hz, 50 Hz and 120 Hz
        let value = 1.0 * (2.0 * PI * 10.0 * t).sin()
            + 2.0 * (2.0 * PI * 50.0 * t).sin()
            + 0.5 * (2.0 * PI * 120.0 * t).sin();

        // Add some noise
        let noise = rand_normal(0.0, 0.1);

        signal.push(value + noise);
    }

    signal
}

// Generate a larger signal for performance testing
fn generate_large_signal(size: usize) -> Vec<f64> {
    let mut signal = Vec::with_capacity(size);

    for i in 0..size {
        let t = i as f64 / 1000.0;

        // Multiple frequency components
        let value = 1.0 * (2.0 * PI * 10.0 * t).sin()
            + 0.8 * (2.0 * PI * 25.0 * t).sin()
            + 0.6 * (2.0 * PI * 50.0 * t).sin()
            + 0.4 * (2.0 * PI * 100.0 * t).sin()
            + 0.2 * (2.0 * PI * 200.0 * t).sin();

        // Add some noise
        let noise = rand_normal(0.0, 0.05);

        signal.push(value + noise);
    }

    signal
}

// Generate normally distributed random numbers using Box-Muller transform
fn rand_normal(mean: f64, stddev: f64) -> f64 {
    let x: f64 = rand::random::<f64>();
    let y: f64 = rand::random::<f64>();

    let normal = (-2.0 * x.ln()).sqrt() * (2.0 * PI * y).cos();
    mean + stddev * normal
}

// Perform spectral analysis on the signal
fn spectral_analysis(signal: &[f64], sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    // Apply a window function to reduce spectral leakage
    let window_func = window::get_window(window::Window::Hann, signal.len(), true).unwrap();

    let windowed_signal: Vec<f64> = signal
        .iter()
        .zip(window_func.iter())
        .map(|(&s, &w)| s * w)
        .collect();

    // Compute FFT with adaptive SIMD acceleration
    let spectrum = fft_adaptive(&windowed_signal, None, None).unwrap();

    // Calculate frequency axis
    let freqs = fftfreq(signal.len(), 1.0 / sample_rate).unwrap();

    // Create our own fftshift since we're working with Vec instead of ndarray
    let mut shifted_freqs = vec![0.0; freqs.len()];
    let mut shifted_spectrum = vec![Complex64::new(0.0, 0.0); spectrum.len()];

    let half_len = signal.len() / 2;
    for i in 0..half_len {
        shifted_freqs[i] = freqs[i + half_len];
        shifted_freqs[i + half_len] = freqs[i];

        shifted_spectrum[i] = spectrum[i + half_len];
        shifted_spectrum[i + half_len] = spectrum[i];
    }

    // If odd length, handle the middle element
    if signal.len() % 2 == 1 {
        shifted_freqs[half_len] = freqs[0];
        shifted_spectrum[half_len] = spectrum[0];
    }

    // Calculate power spectrum (magnitude squared)
    let power_spectrum: Vec<f64> = shifted_spectrum
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im) / signal.len() as f64)
        .collect();

    (shifted_freqs, power_spectrum)
}

// Find peak frequencies in the power spectrum
fn find_peak_frequencies(freqs: &[f64], power: &[f64], threshold_factor: f64) -> Vec<(f64, f64)> {
    // Find maximum power value for threshold calculation
    let max_power = power.iter().fold(0.0f64, |a, &b| a.max(b));
    let threshold = max_power * threshold_factor;

    let mut peaks = Vec::new();

    // Skip first and last points
    for i in 1..power.len() - 1 {
        // Check if this point is a local maximum
        if power[i] > power[i - 1] && power[i] > power[i + 1] && power[i] >= threshold {
            peaks.push((freqs[i].abs(), power[i]));
        }
    }

    // Sort by frequency
    peaks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    peaks
}

// Apply a bandpass filter in the frequency domain
fn bandpass_filter(
    signal: &[f64],
    sample_rate: f64,
    low_cutoff: f64,
    high_cutoff: f64,
) -> Vec<f64> {
    // Compute FFT with adaptive SIMD acceleration
    let mut spectrum = fft_adaptive(signal, None, None).unwrap();

    // Calculate frequency resolution
    let freq_resolution = sample_rate / signal.len() as f64;

    // Create bandpass filter mask
    for i in 0..spectrum.len() {
        // Calculate frequency for this bin
        let freq = if i <= spectrum.len() / 2 {
            i as f64 * freq_resolution
        } else {
            (i as f64 - spectrum.len() as f64) * freq_resolution
        };

        // Apply filter (keep frequencies between low_cutoff and high_cutoff)
        if freq.abs() < low_cutoff || freq.abs() > high_cutoff {
            spectrum[i] = Complex64::new(0.0, 0.0);
        }
    }

    // Compute inverse FFT to get filtered signal
    let filtered_signal = ifft_adaptive(&spectrum, None, None).unwrap();

    // Extract real part
    filtered_signal.iter().map(|c| c.re).collect()
}

// Compute spectrogram using STFT
fn compute_spectrogram(
    signal: &[f64],
    sample_rate: f64,
    window_size: usize,
    overlap: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Calculate hop size (step between windows)
    let hop_size = window_size - overlap;

    // Calculate number of frames
    let num_frames = (signal.len() - overlap) / hop_size;

    // Create window function
    let window_func = window::get_window(window::Window::Hann, window_size, true).unwrap();

    // Prepare output structures
    let mut spectrogram = Vec::with_capacity(num_frames * (window_size / 2 + 1));

    // Process each frame
    for frame in 0..num_frames {
        let start_idx = frame * hop_size;
        let end_idx = start_idx + window_size;

        if end_idx > signal.len() {
            break;
        }

        // Extract frame and apply window
        let mut windowed_frame = Vec::with_capacity(window_size);
        for i in 0..window_size {
            windowed_frame.push(signal[start_idx + i] * window_func[i]);
        }

        // Compute FFT
        let spectrum = fft_adaptive(&windowed_frame, None, None).unwrap();

        // Calculate power for each frequency bin (use only positive frequencies)
        for i in 0..=window_size / 2 {
            let power = spectrum[i].norm_sqr() / window_size as f64;
            spectrogram.push(power);
        }
    }

    // Create time axis
    let time_points = (0..num_frames)
        .map(|i| i as f64 * hop_size as f64 / sample_rate)
        .collect();

    // Create frequency axis
    let nyquist = sample_rate / 2.0;
    let freq_bins = (0..=window_size / 2)
        .map(|i| i as f64 * nyquist / (window_size / 2) as f64)
        .collect();

    (time_points, freq_bins, spectrogram)
}
