use num_complex::Complex64;
use plotly::common::Title;
use plotly::{common::Mode, layout::Axis, Layout, Plot, Scatter};
use scirs2_fft::{
    reconstruct_filtered, reconstruct_high_resolution, reconstruct_time_domain, sparse_fft,
    sparse_fft::SparseFFTAlgorithm, sparse_fft::WindowFunction,
};
use std::f64::consts::PI;

fn main() {
    println!("Sparse FFT Reconstruction Example");
    println!("=================================\n");

    // Create a signal with a few frequency components plus some noise
    let n = 1024;
    println!(
        "Creating a signal with n = {} samples and 3 frequency components",
        n
    );

    // Parameters for our frequency components
    let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25)];

    // Generate clean signal
    let clean_signal = create_sparse_signal(n, &frequencies);

    // Add some noise
    let mut noisy_signal = clean_signal.clone();
    for i in 0..n {
        noisy_signal[i] += 0.1 * rand::random::<f64>();
    }

    // 1. Basic sparse FFT with efficient detection of important components
    println!("\nPerforming sparse FFT on noisy signal...");
    let sparse_result = sparse_fft(
        &noisy_signal,
        6,
        Some(SparseFFTAlgorithm::SpectralFlatness),
        Some(WindowFunction::Blackman), // Use windowing to reduce spectral leakage
    )
    .unwrap();

    println!(
        "Found {} significant frequency components",
        sparse_result.values.len()
    );

    // 2. Basic reconstruction - get our signal back from sparse FFT result
    println!(
        "\nBasic reconstruction: Converting sparse frequency components back to time domain..."
    );
    let reconstructed = reconstruct_time_domain(&sparse_result, n).unwrap();

    // Compute error between original clean signal and reconstruction
    let clean_error = compute_error(&clean_signal, &reconstructed);
    println!(
        "Error between original clean signal and reconstruction: {:.6}",
        clean_error
    );

    // Compute error between noisy signal and reconstruction
    let noisy_error = compute_error(&noisy_signal, &reconstructed);
    println!(
        "Error between noisy signal and reconstruction: {:.6}",
        noisy_error
    );
    println!("(Lower error with clean signal shows noise reduction effect)");

    // 3. High-resolution reconstruction
    println!("\nHigh-resolution reconstruction: Enhancing frequency resolution 2x...");
    let target_length = n * 2;
    let high_res = reconstruct_high_resolution(&sparse_result, n, target_length).unwrap();

    println!("Original signal length: {}", n);
    println!("High-resolution signal length: {}", high_res.len());

    // 4. Filtered reconstruction - demonstrating low-pass filter
    println!("\nFiltered reconstruction: Applying low-pass filter to keep only lowest 10% frequencies...");

    // Create a lowpass filter
    let lowpass = |idx: usize, n: usize| -> f64 {
        let nyquist = n / 2;
        let cutoff = nyquist / 10; // 10% of Nyquist frequency

        // Handle wrapping for negative frequencies
        let freq_idx = if idx <= nyquist { idx } else { n - idx };

        if freq_idx <= cutoff {
            1.0 // Pass
        } else {
            0.0 // Block
        }
    };

    // Apply filter
    let lowpass_signal = reconstruct_filtered(&sparse_result, n, lowpass).unwrap();

    // 5. Filtered reconstruction - demonstrating band-pass filter
    println!("\nFiltered reconstruction: Applying band-pass filter (30-70% of Nyquist)...");

    // Create a bandpass filter
    let bandpass = |idx: usize, n: usize| -> f64 {
        let nyquist = n / 2;
        let low_cutoff = (nyquist as f64 * 0.3) as usize; // 30% of Nyquist
        let high_cutoff = (nyquist as f64 * 0.7) as usize; // 70% of Nyquist

        // Handle wrapping for negative frequencies
        let freq_idx = if idx <= nyquist { idx } else { n - idx };

        if freq_idx >= low_cutoff && freq_idx <= high_cutoff {
            1.0 // Pass
        } else {
            0.0 // Block
        }
    };

    // Apply filter
    let bandpass_signal = reconstruct_filtered(&sparse_result, n, bandpass).unwrap();

    // 6. Visualization
    println!("\nCreating visualization...");
    create_plots(
        &noisy_signal,
        &reconstructed,
        &high_res,
        &lowpass_signal,
        &bandpass_signal,
    );

    println!("\nExample completed successfully!");
}

// Helper function to create a sparse signal
fn create_sparse_signal(n: usize, frequencies: &[(usize, f64)]) -> Vec<f64> {
    let mut signal = vec![0.0; n];

    for i in 0..n {
        let t = 2.0 * PI * (i as f64) / (n as f64);
        for &(freq, amp) in frequencies {
            signal[i] += amp * (freq as f64 * t).sin();
        }
    }

    signal
}

// Helper function to compute error between original and reconstructed signals
fn compute_error(original: &[f64], reconstructed: &[Complex64]) -> f64 {
    if original.len() != reconstructed.len() {
        // Handle simple case for high-resolution where lengths don't match
        // by just comparing overlapping parts
        let min_len = original.len().min(reconstructed.len());
        return compute_error(&original[..min_len], &reconstructed[..min_len]);
    }

    // Convert original signal to complex for comparison
    let original_complex: Vec<Complex64> =
        original.iter().map(|&x| Complex64::new(x, 0.0)).collect();

    // Normalize signals before comparing
    let orig_energy: f64 = original_complex.iter().map(|&x| x.norm_sqr()).sum();
    let recon_energy: f64 = reconstructed.iter().map(|&x| x.norm_sqr()).sum();

    // Compute scaling factors
    let orig_scale = 1.0 / orig_energy.sqrt();
    let recon_scale = 1.0 / recon_energy.sqrt();

    // Compute error between normalized signals
    let mut error_sum = 0.0;
    for i in 0..original.len() {
        let orig = original_complex[i] * orig_scale;
        let recon = reconstructed[i] * recon_scale;
        error_sum += (orig - recon).norm_sqr();
    }

    // Error ranges from 0 (identical) to 2 (completely different)
    // Scale to 0-1 range
    (error_sum / (2.0 * original.len() as f64)).sqrt()
}

// Create visualization plots
fn create_plots(
    noisy_signal: &[f64],
    basic_recon: &[Complex64],
    high_res: &[Complex64],
    lowpass: &[Complex64],
    bandpass: &[Complex64],
) {
    // Convert complex signals to real for plotting
    let basic_recon_real: Vec<f64> = basic_recon.iter().map(|c| c.re).collect();
    let high_res_real: Vec<f64> = high_res.iter().map(|c| c.re).collect();
    let lowpass_real: Vec<f64> = lowpass.iter().map(|c| c.re).collect();
    let bandpass_real: Vec<f64> = bandpass.iter().map(|c| c.re).collect();

    // Create time domain comparison plot - shows a subset for clarity
    let slice_start = 0;
    let slice_len = 200.min(noisy_signal.len());
    let slice_end = slice_start + slice_len;

    let mut time_plot = Plot::new();

    // Original noisy signal
    let noisy_trace = Scatter::new(
        (slice_start..slice_end).collect::<Vec<_>>(),
        noisy_signal[slice_start..slice_end].to_vec(),
    )
    .mode(Mode::Lines)
    .name("Noisy Signal");

    // Basic reconstruction
    let basic_trace = Scatter::new(
        (slice_start..slice_end).collect::<Vec<_>>(),
        basic_recon_real[slice_start..slice_end].to_vec(),
    )
    .mode(Mode::Lines)
    .name("Basic Reconstruction");

    // Lowpass filtered
    let lowpass_trace = Scatter::new(
        (slice_start..slice_end).collect::<Vec<_>>(),
        lowpass_real[slice_start..slice_end].to_vec(),
    )
    .mode(Mode::Lines)
    .name("Lowpass Filtered");

    // Bandpass filtered
    let bandpass_trace = Scatter::new(
        (slice_start..slice_end).collect::<Vec<_>>(),
        bandpass_real[slice_start..slice_end].to_vec(),
    )
    .mode(Mode::Lines)
    .name("Bandpass Filtered");

    // Add traces to plot
    time_plot.add_trace(noisy_trace);
    time_plot.add_trace(basic_trace);
    time_plot.add_trace(lowpass_trace);
    time_plot.add_trace(bandpass_trace);

    // Set layout
    time_plot.set_layout(
        Layout::new()
            .title(Title::with_text("Time Domain Signal Comparison"))
            .x_axis(Axis::new().title(Title::with_text("Sample Index")))
            .y_axis(Axis::new().title(Title::with_text("Amplitude"))),
    );

    // Save the plot
    time_plot.write_html("sparse_recon_time_domain.html");

    // Create high-resolution plot (showing only a segment)
    let mut highres_plot = Plot::new();

    // Define a narrower slice for high-resolution comparison
    let hires_slice_start = 0;
    let hires_slice_len = 100.min(noisy_signal.len());
    let hires_slice_end = hires_slice_start + hires_slice_len;

    // Original time points
    let orig_times: Vec<f64> = (hires_slice_start..hires_slice_end)
        .map(|i| i as f64)
        .collect();

    // High-resolution time points (twice as many)
    let hires_times: Vec<f64> = (0..(2 * hires_slice_len))
        .map(|i| hires_slice_start as f64 + i as f64 / 2.0)
        .collect();

    // Original noisy signal
    let orig_trace = Scatter::new(
        orig_times.clone(),
        noisy_signal[hires_slice_start..hires_slice_end].to_vec(),
    )
    .mode(Mode::Lines)
    .name("Original Signal");

    // High-resolution signal (using twice as many points)
    let hires_trace = Scatter::new(
        hires_times,
        high_res_real[2 * hires_slice_start..2 * hires_slice_end].to_vec(),
    )
    .mode(Mode::Lines)
    .name("High-Resolution");

    // Add traces to plot
    highres_plot.add_trace(orig_trace);
    highres_plot.add_trace(hires_trace);

    // Set layout
    highres_plot.set_layout(
        Layout::new()
            .title(Title::with_text("High-Resolution Reconstruction"))
            .x_axis(Axis::new().title(Title::with_text("Sample Index")))
            .y_axis(Axis::new().title(Title::with_text("Amplitude"))),
    );

    // Save the plot
    highres_plot.write_html("sparse_recon_high_res.html");

    println!("Plots saved as 'sparse_recon_time_domain.html' and 'sparse_recon_high_res.html'");
}
