// The signal_processing module is currently gated behind a feature flag
// This example is temporarily disabled until the module is available

#[allow(dead_code)]
fn main() {
    println!("Signal Processing Example");
    println!("------------------------");
    println!(
        "This example is currently disabled as the signal_processing module is not available."
    );
    println!("Please use other examples in this directory for FFT functionality.");
}

/*
// Original code preserved for future use:

use scirs2_core::Complex64;
use scirs2_fft::fft::{self, fft};
use scirs2_fft::signal_processing::{
    convolve, cross_correlate, design_fir_filter, fir_filter, frequency_filter, FilterSpec,
    FilterType, FilterWindow,
};
use std::time::Instant;

#[allow(dead_code)]
fn main() {
    println!("Signal Processing Example");
    println!("------------------------");

    // Generate example signals with limited size to prevent timeouts
    println!("\nGenerating sample signals...");
    let sample_rate = 1000.0; // Hz
    let duration = 0.5; // seconds
    let n_samples = (sample_rate * duration) as usize;

    // Generate a signal with multiple frequency components
    let t: Vec<f64> = (0..n_samples)
        .map(|i| i as f64 / sample_rate)
        .collect();

    // Create a composite signal
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| {
            10.0 * (2.0 * PI * 50.0 * ti).sin()     // 50 Hz
                + 5.0 * (2.0 * PI * 120.0 * ti).sin()  // 120 Hz
                + 2.0 * (2.0 * PI * 200.0 * ti).sin()  // 200 Hz
                + 0.5 * scirs2_core::random::random::<f64>()          // Noise
        })
        .collect();

    // Convolution Example
    println!("\n1. Convolution Example");
    let kernel = vec![0.25, 0.5, 0.25]; // Simple smoothing kernel
    let start = Instant::now();
    let smoothed = convolve(&signal, &kernel).unwrap();
    let duration = start.elapsed();
    println!("  Convolution completed in: {:?}", duration);
    println!("  Input length: {}, Kernel length: {}, Output length: {}",
        signal.len(), kernel.len(), smoothed.len());

    // Cross-correlation Example
    println!("\n2. Cross-correlation Example");
    let pattern = &signal[100..150]; // Extract a pattern from the signal
    let start = Instant::now();
    let correlation = cross_correlate(&signal, pattern).unwrap();
    let duration = start.elapsed();
    println!("  Cross-correlation completed in: {:?}", duration);

    // Find the best match position
    let best_match_index = correlation
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index_)| index)
        .unwrap();

    println!("  Best match found at position: {}", best_match_index);

    // FIR Filter Design Example
    println!("\n3. FIR Filter Design Example");
    let spec = FilterSpec {
        filter_type: FilterType::Lowpass,
        cutoff_freq: 80.0,
        stopband_freq: Some(150.0),
        sampling_rate: sample_rate,
        num_taps: 65,
        window: FilterWindow::Hamming,
        passband_ripple: 0.1,
        stopband_attenuation: 60.0,
        transition_width: Some(50.0),
    };

    let start = Instant::now();
    let filter_coeffs = design_fir_filter(&spec).unwrap();
    let duration = start.elapsed();
    println!("  Filter designed in: {:?}", duration);
    println!("  Number of coefficients: {}", filter_coeffs.len());

    // Apply FIR Filter
    println!("\n4. FIR Filtering Example");
    let start = Instant::now();
    let filtered_signal = fir_filter(&signal, &filter_coeffs).unwrap();
    let duration = start.elapsed();
    println!("  FIR filtering completed in: {:?}", duration);
    println!("  Filtered signal length: {}", filtered_signal.len());

    // Frequency-domain Filtering Example
    println!("\n5. Frequency-domain Filtering Example");

    // Create a bandpass filter
    let lower_freq = 40.0;
    let upper_freq = 100.0;
    let n_freq = n_samples;

    let mut filter_response = vec![Complex64::new(0.0, 0.0); n_freq];
    for (i, h) in filter_response.iter_mut().enumerate() {
        let freq = i as f64 * sample_rate / n_freq as f64;
        if freq >= lower_freq && freq <= upper_freq {
            *h = Complex64::new(1.0, 0.0);
        }
    }

    let start = Instant::now();
    let freq_filtered = frequency_filter(&signal, &filter_response).unwrap();
    let duration = start.elapsed();
    println!("  Frequency filtering completed in: {:?}", duration);
    println!("  Bandpass filter: {} Hz to {} Hz", lower_freq, upper_freq);

    // Analyze the filtered results
    println!("\n6. Spectrum Analysis");

    // Compute FFT of original signal
    let original_fft = fft(&signal, None).unwrap();

    // Compute FFT of filtered signal
    let filtered_fft = fft(&filtered_signal, None).unwrap();

    // Find peak frequencies
    let mut freq_magnitude: Vec<(f64, f64)> = Vec::new();
    for i in 0..n_samples/2 {
        let freq = i as f64 * sample_rate / n_samples as f64;
        let magoriginal = original_fft[i].norm();
        freq_magnitude.push((freq, magoriginal));
    }

    // Find the top 3 peaks
    freq_magnitude.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n  Top frequency components in original signal:");
    for i in 0..3.min(freq_magnitude.len()) {
        println!("    {:.1} Hz: magnitude = {:.2}",
            freq_magnitude[i].0,
            freq_magnitude[i].1);
    }

    // Performance comparison
    println!("\n7. Performance Summary");
    println!("  Signal length: {} samples", n_samples);
    println!("  Sample rate: {} Hz", sample_rate);
    println!("  Processing completed successfully!");
}
*/
