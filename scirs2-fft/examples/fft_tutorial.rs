//! Tutorial for common FFT operations
//!
//! This example demonstrates common Fast Fourier Transform (FFT) operations
//! using the scirs2-fft library.

use ndarray::array;
use num_complex::Complex64;
use scirs2_fft::{
    fft, fft2, fftfreq, frft, hilbert, ifft, ifft2, irfft, rfft, spectrogram,
    window::{get_window, Window},
};
use std::f64::consts::PI;

fn main() {
    println!("=== FFT Tutorial ===\n");

    // Example 1: Basic FFT and IFFT
    basic_fft_example();

    // Example 2: Real FFT (optimized for real-valued input)
    real_fft_example();

    // Example 2: 2D FFT (for images or matrices)
    multi_dimensional_fft_example();

    // Example 3: Zero padding and resolution
    zero_padding_example();

    // Example 4: Windowing functions
    windowing_example();

    // Example 5: Frequency analysis
    frequency_analysis_example();

    // Example 6: Signal filtering in frequency domain
    frequency_domain_filtering();

    // Example 7: Hilbert transform for analytic signal
    hilbert_transform_example();

    // Example 8: Fractional Fourier Transform - temporarily disabled due to complex conversion issues
    // fractional_fourier_transform_example();

    // Example 9: Time-frequency analysis with spectrograms
    time_frequency_analysis();

    println!("\nTutorial completed!");
}

/// Basic 1D FFT and IFFT operations
fn basic_fft_example() {
    println!("\n--- Basic FFT and IFFT ---");

    // Create a simple signal
    let signal = vec![1.0, 2.0, 3.0, 4.0];
    println!("Input signal: {:?}", signal);

    // Compute the FFT
    let spectrum = fft(&signal, None).unwrap();

    // Display the result (magnitude and phase)
    println!("FFT result:");
    println!("  DC component (sum): {:.2}", spectrum[0].re);
    println!(
        "  Magnitudes: {:.2}, {:.2}, {:.2}, {:.2}",
        spectrum[0].norm(),
        spectrum[1].norm(),
        spectrum[2].norm(),
        spectrum[3].norm()
    );

    // Recover the original signal using IFFT
    let recovered = ifft(&spectrum, None).unwrap();
    println!("Recovered signal (real parts):");
    println!(
        "  {:.2}, {:.2}, {:.2}, {:.2}",
        recovered[0].re, recovered[1].re, recovered[2].re, recovered[3].re
    );
}

/// Real-to-complex and complex-to-real FFT
fn real_fft_example() {
    println!("\n--- Real FFT (RFFT) ---");

    // Create a real-valued signal
    let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    println!("Input signal length: {}", signal.len());

    // Compute the real FFT (more efficient than regular FFT for real input)
    let spectrum = rfft(&signal, None).unwrap();

    // Notice that rfft returns only n//2 + 1 complex values (non-redundant part)
    println!(
        "RFFT output length: {} (vs regular FFT: {})",
        spectrum.len(),
        signal.len()
    );

    println!("RFFT result (first few components):");
    println!("  DC component (sum): {:.2}", spectrum[0].re);
    println!(
        "  First harmonic: {:.2} + {:.2}i",
        spectrum[1].re, spectrum[1].im
    );

    // Recover the original signal using inverse RFFT
    let recovered = irfft(&spectrum, Some(signal.len())).unwrap();
    println!(
        "Recovered signal matches original: {}",
        recovered
            .iter()
            .zip(signal.iter())
            .all(|(a, b)| (a - b).abs() < 1e-10)
    );
}

/// 2D FFT for images or matrices
fn multi_dimensional_fft_example() {
    println!("\n--- 2D FFT Example ---");

    // Create a 2D array (e.g., simple image or matrix)
    let data = array![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ];

    println!("Input 2D array:");
    for row in data.rows() {
        println!("  {:?}", row);
    }

    // Compute 2D FFT
    let spectrum_2d = fft2(&data.to_owned(), None, None, None).unwrap();

    // DC component should be the sum of all elements
    let total_sum: f64 = data.iter().sum();
    println!("Sum of all elements: {:.2}", total_sum);
    println!(
        "DC component (FFT[0,0]): {:.2} + {:.2}i",
        spectrum_2d[[0, 0]].re,
        spectrum_2d[[0, 0]].im
    );

    // Horizontal and vertical frequencies
    println!(
        "Horizontal frequency component (FFT[0,1]): {:.2} + {:.2}i",
        spectrum_2d[[0, 1]].re,
        spectrum_2d[[0, 1]].im
    );
    println!(
        "Vertical frequency component (FFT[1,0]): {:.2} + {:.2}i",
        spectrum_2d[[1, 0]].re,
        spectrum_2d[[1, 0]].im
    );

    // Recover original data
    let recovered_2d = ifft2(&spectrum_2d.to_owned(), None, None, None).unwrap();
    println!(
        "Recovered 2D array matches original: {}",
        recovered_2d
            .iter()
            .zip(data.iter())
            .all(|(a, b)| (a.re - b).abs() < 1e-10)
    );
}

/// Zero padding for increased frequency resolution
fn zero_padding_example() {
    println!("\n--- Zero Padding Example ---");

    // Create a signal with a pure sine wave
    let n = 64;
    let freq = 5.0; // 5 cycles in the signal
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * freq * i as f64 / n as f64).sin())
        .collect();

    println!("Original signal length: {}", signal.len());

    // FFT without zero padding
    let spectrum = fft(&signal, None).unwrap();

    // FFT with zero padding (4x length)
    let padded_spectrum = fft(&signal, Some(4 * n)).unwrap();

    println!("Zero-padded FFT length: {}", padded_spectrum.len());

    // Get corresponding frequency values
    let freqs = fftfreq(spectrum.len(), 1.0);
    let padded_freqs = fftfreq(padded_spectrum.len(), 1.0);

    // Find peak in original spectrum
    let mut max_idx = 0;
    let mut max_val = 0.0;
    for (i, val) in spectrum.iter().enumerate() {
        let magnitude = val.norm();
        if magnitude > max_val {
            max_val = magnitude;
            max_idx = i;
        }
    }

    // Find peak in padded spectrum
    let mut padded_max_idx = 0;
    let mut padded_max_val = 0.0;
    for (i, val) in padded_spectrum.iter().enumerate() {
        let magnitude = val.norm();
        if magnitude > padded_max_val {
            padded_max_val = magnitude;
            padded_max_idx = i;
        }
    }

    let freqs_vec = freqs.unwrap();
    let padded_freqs_vec = padded_freqs.unwrap();

    println!("Original FFT peak at frequency {:.2}", freqs_vec[max_idx]);
    println!(
        "Zero-padded FFT peak at frequency {:.2}",
        padded_freqs_vec[padded_max_idx]
    );

    println!("Zero padding improves frequency resolution but doesn't add new information.");
    println!("It's like interpolating between frequency bins to find peaks more precisely.");
}

/// Windowing functions to reduce spectral leakage
fn windowing_example() {
    println!("\n--- Windowing Example ---");

    // Create a signal with a non-integer number of cycles (causes leakage)
    let n = 64;
    let freq = 4.5; // 4.5 cycles (non-integer) in the signal
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * freq * i as f64 / n as f64).sin())
        .collect();

    // FFT without window
    let spectrum_no_window = fft(&signal, None).unwrap();

    // Get window functions
    let hann_window = get_window(Window::Hann, n, true).unwrap();
    let blackman_window = get_window(Window::Blackman, n, true).unwrap();

    // Apply windows to signal
    let windowed_signal_hann: Vec<f64> = signal
        .iter()
        .zip(hann_window.iter())
        .map(|(&s, &w)| s * w)
        .collect();

    let windowed_signal_blackman: Vec<f64> = signal
        .iter()
        .zip(blackman_window.iter())
        .map(|(&s, &w)| s * w)
        .collect();

    // FFT with windows
    let spectrum_hann = fft(&windowed_signal_hann, None).unwrap();
    let spectrum_blackman = fft(&windowed_signal_blackman, None).unwrap();

    // Calculate spectral leakage (energy in non-peak bins)
    let calc_leakage = |spectrum: &[Complex64], peak_idx: usize| -> f64 {
        let peak_energy = spectrum[peak_idx].norm_sqr();
        let total_energy: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum();
        (total_energy - peak_energy) / total_energy
    };

    // Find peak indices
    let find_peak = |spectrum: &[Complex64]| -> usize {
        spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };

    let peak_no_window = find_peak(&spectrum_no_window);
    let peak_hann = find_peak(&spectrum_hann);
    let peak_blackman = find_peak(&spectrum_blackman);

    let leakage_no_window = calc_leakage(&spectrum_no_window, peak_no_window);
    let leakage_hann = calc_leakage(&spectrum_hann, peak_hann);
    let leakage_blackman = calc_leakage(&spectrum_blackman, peak_blackman);

    println!("Spectral leakage comparison:");
    println!("  No window: {:.1}%", leakage_no_window * 100.0);
    println!("  Hann window: {:.1}%", leakage_hann * 100.0);
    println!("  Blackman window: {:.1}%", leakage_blackman * 100.0);

    println!("Windows reduce spectral leakage but affect amplitude and width of peaks.");
    println!("Choose based on your needs: Rectangular (no window) for maximum resolution,");
    println!("Hann for good balance, Blackman for maximum leakage reduction.");
}

/// Frequency analysis of signals
fn frequency_analysis_example() {
    println!("\n--- Frequency Analysis Example ---");

    // Create a signal with multiple frequency components
    let n = 256;
    let dt = 0.01; // 100 Hz sampling rate
    let freqs = [5.0, 15.0, 25.0]; // Hz
    let amps = [1.0, 0.5, 0.25]; // Amplitudes

    let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    let mut signal = vec![0.0; n];

    for i in 0..n {
        for (&freq, &amp) in freqs.iter().zip(amps.iter()) {
            signal[i] += amp * (2.0 * PI * freq * t[i]).sin();
        }
    }

    // Compute the FFT
    let spectrum = fft(&signal, None).unwrap();

    // Compute the power spectrum (|FFT|²)
    let power_spectrum: Vec<f64> = spectrum.iter().map(|c| c.norm_sqr() / (n as f64)).collect();

    // Get the frequency axis
    let freq_axis = fftfreq(n, dt).unwrap();

    // Find peaks in the first half of the spectrum (positive frequencies)
    let mut peaks = Vec::new();
    for i in 1..n / 2 {
        if power_spectrum[i] > power_spectrum[i - 1] && power_spectrum[i] > power_spectrum[i + 1] {
            peaks.push((freq_axis[i], power_spectrum[i], i));
        }
    }

    // Sort peaks by amplitude (descending)
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top frequency components detected:");
    for (i, &(freq, power, _)) in peaks.iter().take(3).enumerate() {
        println!(
            "  Peak {}: {:.1} Hz, amplitude: {:.2}",
            i + 1,
            freq,
            power.sqrt()
        );
    }

    println!("Expected components:");
    for (i, (&freq, &amp)) in freqs.iter().zip(amps.iter()).enumerate() {
        println!(
            "  Component {}: {:.1} Hz, amplitude: {:.2}",
            i + 1,
            freq,
            amp
        );
    }
}

/// Signal filtering in the frequency domain
fn frequency_domain_filtering() {
    println!("\n--- Frequency Domain Filtering ---");

    // Create a signal with a clean tone and high-frequency noise
    let n = 512;
    let dt = 0.001; // 1000 Hz sampling rate

    // Signal components
    let signal_freq = 50.0; // Hz - this is our desired signal
    let noise_freq = 250.0; // Hz - this is noise we want to filter out

    // Create the composite signal
    let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    let mut signal = vec![0.0; n];

    for i in 0..n {
        // Clean tone
        signal[i] = (2.0 * PI * signal_freq * t[i]).sin();
        // Add high-frequency noise
        signal[i] += 0.2 * (2.0 * PI * noise_freq * t[i]).sin();
        // Add some random noise
        signal[i] += 0.05 * (i as f64).sin();
    }

    // Compute the FFT
    let mut spectrum = fft(&signal, None).unwrap();

    // Get the frequency axis
    let freq_axis = fftfreq(n, dt).unwrap();

    // Design a low-pass filter (cutoff at 100 Hz)
    let cutoff_freq = 100.0; // Hz

    // Apply the filter in the frequency domain
    for i in 0..n {
        let freq = freq_axis[i].abs();
        if freq > cutoff_freq {
            // Simple brick-wall filter (in practice, use smoother filters)
            spectrum[i] = Complex64::new(0.0, 0.0);
        }
    }

    // Transform back to time domain
    let filtered_signal = ifft(&spectrum, None).unwrap();

    // Calculate signal-to-noise improvement
    let calc_snr = |s: &[f64]| -> f64 {
        // Estimate signal and noise power by frequency components
        let spectrum = fft(s, None).unwrap();
        let freq = fftfreq(s.len(), dt).unwrap();

        let mut signal_power = 0.0;
        let mut noise_power = 0.0;

        for i in 0..s.len() {
            let power = spectrum[i].norm_sqr() / (s.len() as f64);
            if freq[i].abs() <= cutoff_freq {
                signal_power += power;
            } else {
                noise_power += power;
            }
        }

        10.0 * (signal_power / noise_power).log10()
    };

    let original_snr = calc_snr(&signal);
    let filtered_snr = calc_snr(&filtered_signal.iter().map(|c| c.re).collect::<Vec<f64>>());

    println!("Signal-to-noise ratio comparison:");
    println!("  Original signal: {:.1} dB", original_snr);
    println!("  Filtered signal: {:.1} dB", filtered_snr);
    println!("  Improvement: {:.1} dB", filtered_snr - original_snr);

    println!("Frequency domain filtering allows precise control over which");
    println!("frequency components to keep or remove from a signal.");
}

/// Hilbert transform for analytic signal
fn hilbert_transform_example() {
    println!("\n--- Hilbert Transform Example ---");

    // Create a sinusoidal signal
    let n = 256;
    let frequency = 5.0; // Hz
    let fs = 100.0; // 100 Hz sampling rate
    let dt = 1.0 / fs;

    let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * frequency * ti).cos())
        .collect();

    println!("Created a {} Hz cosine signal", frequency);

    // Compute the Hilbert transform (analytic signal)
    let analytic = hilbert(&signal).unwrap();

    // Extract amplitude and instantaneous phase
    let amplitude: Vec<f64> = analytic.iter().map(|c| c.norm()).collect();
    let phase: Vec<f64> = analytic.iter().map(|c| c.im.atan2(c.re)).collect();

    // Calculate instantaneous frequency (derivative of phase)
    let mut inst_freq = vec![0.0; n - 1];
    for i in 0..n - 1 {
        // Unwrap phase to handle 2π jumps
        let mut phase_diff = phase[i + 1] - phase[i];
        if phase_diff > PI {
            phase_diff -= 2.0 * PI;
        } else if phase_diff < -PI {
            phase_diff += 2.0 * PI;
        }

        // Convert phase difference to frequency
        inst_freq[i] = phase_diff / (2.0 * PI * dt);
    }

    // Calculate average amplitude and frequency
    let avg_amplitude: f64 = amplitude.iter().sum::<f64>() / (n as f64);
    let avg_freq: f64 = inst_freq.iter().sum::<f64>() / ((n - 1) as f64);

    println!("Analytic signal properties:");
    println!("  Average amplitude: {:.2} (expected: 1.0)", avg_amplitude);
    println!(
        "  Average instantaneous frequency: {:.2} Hz (expected: {})",
        avg_freq, frequency
    );

    println!("The Hilbert transform creates an analytic signal,");
    println!("which lets us extract instantaneous amplitude and frequency.");
}

/// Fractional Fourier Transform
#[allow(dead_code)]
fn fractional_fourier_transform_example() {
    println!("\n--- Fractional Fourier Transform Example ---");

    // Create a simple Gaussian pulse signal
    let n = 256;
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i as f64 - n as f64 / 2.0) / (n as f64 / 8.0);
            (-x * x / 2.0).exp()
        })
        .collect();

    println!("Created a Gaussian pulse signal");

    // Calculate regular FFT (alpha = 1)
    let _spectrum = fft(&signal, None).unwrap();

    // Calculate Fractional Fourier Transforms at different angles
    let angles = [0.25, 0.5, 0.75];

    println!("Computing Fractional Fourier Transforms:");
    for &alpha in &angles {
        let frft_result = frft(&signal, alpha, None).unwrap();

        // Calculate energy preservation
        let input_energy: f64 = signal.iter().map(|&x| x * x).sum();
        let output_energy: f64 = frft_result.iter().map(|c| c.norm_sqr()).sum();
        let energy_ratio = output_energy / input_energy;

        println!("  FrFT at α = {:.2}π:", alpha);
        println!("    Energy preservation ratio: {:.4}", energy_ratio);

        // For α = 0, the transform should be the original signal
        // For α = 0.5, it should be similar to the regular FFT
        // For α = 1, it should be the time-reversed signal

        let expected;
        if (alpha - 0.0).abs() < 0.01 {
            expected = "original signal";
        } else if (alpha - 0.5).abs() < 0.01 {
            expected = "regular FFT";
        } else if (alpha - 1.0).abs() < 0.01 {
            expected = "time-reversed signal";
        } else {
            expected = "intermediate representation";
        }

        println!("    This transform is close to the {}", expected);
    }

    println!("The Fractional Fourier Transform provides a continuous");
    println!("rotation in the time-frequency plane, generalizing the");
    println!("standard Fourier transform to arbitrary angles.");
}

/// Time-frequency analysis with spectrograms
fn time_frequency_analysis() {
    println!("\n--- Time-Frequency Analysis Example ---");

    // Create a signal with time-varying frequency (chirp)
    let n = 1000;
    let fs = 1000.0; // 1000 Hz sampling rate
    let dt = 1.0 / fs;

    let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();

    // Linear chirp from 50 Hz to 250 Hz
    let start_freq = 50.0;
    let end_freq = 250.0;
    let rate = (end_freq - start_freq) / t[n - 1];

    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| {
            let inst_freq = start_freq + rate * ti;
            (2.0 * PI * inst_freq * ti).sin()
        })
        .collect();

    println!(
        "Created a chirp signal from {} Hz to {} Hz",
        start_freq, end_freq
    );

    // Compute the spectrogram
    let window_size = 128;
    let hop_size = 32;
    let (frequencies, times, sxx) = spectrogram(
        &signal,
        Some(fs),
        Some(Window::Hann),
        Some(window_size),
        Some(window_size - hop_size),
        None,
        None,
        Some("spectrum"),
        None, // Default mode
    )
    .unwrap();

    // Find the frequency with maximum energy at each time point
    let mut peak_frequencies = Vec::new();
    for t_idx in 0..sxx.shape()[1] {
        let mut max_idx = 0;
        let mut max_val = 0.0;

        for f_idx in 0..sxx.shape()[0] {
            if sxx[[f_idx, t_idx]] > max_val {
                max_val = sxx[[f_idx, t_idx]];
                max_idx = f_idx;
            }
        }

        peak_frequencies.push(frequencies[max_idx]);
    }

    // Check if peak frequencies follow the expected chirp pattern
    let expected_freqs: Vec<f64> = times.iter().map(|&t| start_freq + rate * t).collect();

    // Calculate mean absolute error between measured and expected frequencies
    let mut error_sum = 0.0;
    let mut count = 0;
    for (i, &freq) in peak_frequencies.iter().enumerate() {
        if freq > 0.0 && freq < fs / 2.0 {
            // Skip invalid frequencies
            error_sum += (freq - expected_freqs[i]).abs();
            count += 1;
        }
    }
    let mean_error = error_sum / count as f64;

    println!("Spectrogram analysis:");
    println!("  Number of time frames: {}", times.len());
    println!("  Number of frequency bins: {}", frequencies.len());
    println!("  Mean frequency estimation error: {:.2} Hz", mean_error);

    println!("Spectrograms reveal how frequency content changes over time,");
    println!("making them ideal for analyzing signals with time-varying properties.");
}
