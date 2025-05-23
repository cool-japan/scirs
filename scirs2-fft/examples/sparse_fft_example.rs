use num_complex::Complex64;
use plotly::common::Title;
use plotly::{common::Mode, layout::Axis, Layout, Plot, Scatter};
use scirs2_fft::{
    adaptive_sparse_fft, fft, frequency_pruning_sparse_fft, reconstruct_spectrum, sparse_fft,
    sparse_fft::{SparseFFTAlgorithm, WindowFunction},
    sparse_fft2, spectral_flatness_sparse_fft,
};
use std::f64::consts::PI;

fn main() {
    println!("Sparse FFT Example");
    println!("==================\n");

    // 1. Create a signal with a few frequency components
    let n = 1024;
    println!(
        "Creating a signal with n = {} samples and 3 frequency components",
        n
    );
    let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25)];
    let signal = create_sparse_signal(n, &frequencies);

    // 2. Compute regular FFT for comparison
    println!("\nComputing regular FFT for comparison...");
    let full_fft_result = fft(&signal, None).unwrap();
    let full_magnitudes: Vec<f64> = full_fft_result.iter().map(|c| c.norm()).collect();

    // 3. Compute sparse FFT with different algorithms
    println!("\nComputing sparse FFT with different algorithms...");
    let algorithms = [
        SparseFFTAlgorithm::Sublinear,
        SparseFFTAlgorithm::CompressedSensing,
        SparseFFTAlgorithm::Iterative,
        SparseFFTAlgorithm::Deterministic,
        SparseFFTAlgorithm::FrequencyPruning,
    ];

    for &alg in &algorithms {
        println!("\n* Using {:?} algorithm", alg);
        let start = std::time::Instant::now();
        let sparse_result = sparse_fft(&signal, 6, Some(alg), Some(WindowFunction::Hann)).unwrap();
        let elapsed = start.elapsed();

        println!(
            "  - Found {} frequency components in {:?}",
            sparse_result.values.len(),
            elapsed
        );
        println!(
            "  - Estimated sparsity: {}",
            sparse_result.estimated_sparsity
        );

        // Print the top frequency components
        println!("  - Top frequency components:");

        // Get unique index-value pairs sorted by magnitude
        let mut unique_components: Vec<(usize, Complex64)> = Vec::new();
        for (&idx, &val) in sparse_result
            .indices
            .iter()
            .zip(sparse_result.values.iter())
        {
            if !unique_components.iter().any(|(i, _)| *i == idx) {
                unique_components.push((idx, val));
            }
        }

        // Sort by magnitude in descending order
        unique_components.sort_by(|(_, a), (_, b)| {
            b.norm()
                .partial_cmp(&a.norm())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Display top 3 components
        for (i, (idx, val)) in unique_components.iter().take(3).enumerate() {
            println!(
                "    {}. Index {}: magnitude = {:.3}",
                i + 1,
                idx,
                val.norm()
            );
        }

        // Reconstruct spectrum
        let reconstructed_spectrum = reconstruct_spectrum(&sparse_result, n).unwrap();

        // Get time-domain signal from the reconstructed spectrum
        let reconstructed_signal = scirs2_fft::ifft(&reconstructed_spectrum, None).unwrap();

        // Convert original signal to complex for comparison
        let signal_complex: Vec<Complex64> =
            signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();

        // Compute error in time domain
        let error = compute_relative_error(&signal_complex, &reconstructed_signal);
        println!("  - Relative error: {:.6}", error);
    }

    // 4. Try adaptive sparse FFT
    println!("\nComputing adaptive sparse FFT with automatic sparsity estimation...");
    let adaptive_result = adaptive_sparse_fft(&signal, 0.1).unwrap();
    println!(
        "- Found {} frequency components",
        adaptive_result.values.len()
    );
    println!(
        "- Estimated sparsity: {}",
        adaptive_result.estimated_sparsity
    );

    // 5. Try frequency pruning algorithm (new algorithm added)
    println!("\nComputing frequency pruning sparse FFT with statistical thresholding...");
    let pruning_result = frequency_pruning_sparse_fft(&signal, 2.0).unwrap();
    println!(
        "- Found {} frequency components",
        pruning_result.values.len()
    );
    println!(
        "- Estimated sparsity: {}",
        pruning_result.estimated_sparsity
    );

    // Print the top frequency components
    println!("- Top frequency components with pruning approach:");

    // Get unique index-value pairs sorted by magnitude
    let mut unique_components: Vec<(usize, Complex64)> = Vec::new();
    for (&idx, &val) in pruning_result
        .indices
        .iter()
        .zip(pruning_result.values.iter())
    {
        if !unique_components.iter().any(|(i, _)| *i == idx) {
            unique_components.push((idx, val));
        }
    }

    // Sort by magnitude in descending order
    unique_components.sort_by(|(_, a), (_, b)| {
        b.norm()
            .partial_cmp(&a.norm())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Display top 3 components
    for (i, (idx, val)) in unique_components.iter().take(3).enumerate() {
        println!(
            "    {}. Index {}: magnitude = {:.3}",
            i + 1,
            idx,
            val.norm()
        );
    }

    // Compare error with original signal
    let reconstructed_spectrum = reconstruct_spectrum(&pruning_result, n).unwrap();
    let reconstructed_signal = scirs2_fft::ifft(&reconstructed_spectrum, None).unwrap();
    let signal_complex: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    let error = compute_relative_error(&signal_complex, &reconstructed_signal);
    println!("- Relative error with pruning algorithm: {:.6}", error);

    // 6. Try spectral flatness algorithm (new algorithm added)
    println!("\nComputing spectral flatness sparse FFT with enhanced noise tolerance...");

    // Create a noisy version of the signal
    let mut noisy_signal = signal.clone();
    for i in 0..n {
        // Add some noise (ramping up with frequency to simulate real-world conditions)
        noisy_signal[i] += 0.05 * ((i % 64) as f64 / 64.0 - 0.5);
    }

    // Using Hamming window for better frequency resolution with noise
    let flatness_result =
        spectral_flatness_sparse_fft(&noisy_signal, 0.3, 32, Some(WindowFunction::Hamming))
            .unwrap();
    println!(
        "- Found {} frequency components",
        flatness_result.values.len()
    );
    println!(
        "- Estimated sparsity: {}",
        flatness_result.estimated_sparsity
    );

    // Print the top frequency components
    println!("- Top frequency components with spectral flatness approach:");

    // Get unique index-value pairs sorted by magnitude
    let mut unique_components: Vec<(usize, Complex64)> = Vec::new();
    for (&idx, &val) in flatness_result
        .indices
        .iter()
        .zip(flatness_result.values.iter())
    {
        if !unique_components.iter().any(|(i, _)| *i == idx) {
            unique_components.push((idx, val));
        }
    }

    // Sort by magnitude in descending order
    unique_components.sort_by(|(_, a), (_, b)| {
        b.norm()
            .partial_cmp(&a.norm())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Display top 3 components
    for (i, (idx, val)) in unique_components.iter().take(3).enumerate() {
        println!(
            "    {}. Index {}: magnitude = {:.3}",
            i + 1,
            idx,
            val.norm()
        );
    }

    // Compare error with original signal
    let reconstructed_spectrum = reconstruct_spectrum(&flatness_result, n).unwrap();
    let reconstructed_signal = scirs2_fft::ifft(&reconstructed_spectrum, None).unwrap();
    let error = compute_relative_error(&signal_complex, &reconstructed_signal);
    println!(
        "- Relative error with spectral flatness algorithm: {:.6}",
        error
    );

    // Compare how it performs on noisy signal
    let noisy_signal_complex: Vec<Complex64> = noisy_signal
        .iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    let error_on_noise = compute_relative_error(&noisy_signal_complex, &reconstructed_signal);
    println!(
        "- Relative error compared to noisy input: {:.6}",
        error_on_noise
    );

    // 7. 2D Sparse FFT Example
    println!("\nComputing 2D sparse FFT example...");
    let rows = 32;
    let cols = 32;
    let signal_2d = create_2d_sparse_signal(rows, cols);

    let start = std::time::Instant::now();
    // Using Blackman window for reduced spectral leakage
    let sparse_2d_result =
        sparse_fft2(&signal_2d, (rows, cols), 8, Some(WindowFunction::Blackman)).unwrap();
    let elapsed = start.elapsed();

    println!(
        "- 2D Sparse FFT of {}x{} signal: found {} components in {:?}",
        rows,
        cols,
        sparse_2d_result.values.len(),
        elapsed
    );

    // 8. Create visualization
    println!("\nCreating visualization...");
    create_plots(&signal, &full_magnitudes, &pruning_result);

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

// Helper function to create a 2D sparse signal
fn create_2d_sparse_signal(rows: usize, cols: usize) -> Vec<f64> {
    let mut signal = vec![0.0; rows * cols];

    for i in 0..rows {
        for j in 0..cols {
            let x = 2.0 * PI * (i as f64) / (rows as f64);
            let y = 2.0 * PI * (j as f64) / (cols as f64);
            signal[i * cols + j] = (2.0 * x + 3.0 * y).sin() + 0.5 * (5.0 * x).sin();
        }
    }

    signal
}

// Helper function to compute relative error
fn compute_relative_error(original: &[Complex64], reconstructed: &[Complex64]) -> f64 {
    // Make sure we're comparing signals of the same length
    let len = std::cmp::min(original.len(), reconstructed.len());

    if len == 0 {
        return 1.0; // Return max error if signals are empty
    }

    // Normalize signals before comparing
    let orig_energy: f64 = original.iter().take(len).map(|c| c.norm_sqr()).sum();
    let recon_energy: f64 = reconstructed.iter().take(len).map(|c| c.norm_sqr()).sum();

    // Compute scaling factors
    let orig_scale = if orig_energy > 0.0 {
        1.0 / orig_energy.sqrt()
    } else {
        1.0
    };
    let recon_scale = if recon_energy > 0.0 {
        1.0 / recon_energy.sqrt()
    } else {
        1.0
    };

    // Compute error between normalized signals
    let mut error_sum = 0.0;
    for i in 0..len {
        let orig = original[i] * orig_scale;
        let recon = reconstructed[i] * recon_scale;
        error_sum += (orig - recon).norm_sqr();
    }

    // Error ranges from 0 (identical) to 2 (completely different)
    // Scale to 0-1 range
    (error_sum / (2.0 * len as f64)).sqrt()
}

// Create visualization plots
fn create_plots(
    signal: &[f64],
    full_magnitudes: &[f64],
    sparse_result: &scirs2_fft::sparse_fft::SparseFFTResult,
) {
    // Create time domain plot
    let mut time_plot = Plot::new();
    let time_trace = Scatter::new((0..signal.len()).collect::<Vec<_>>(), signal.to_vec())
        .mode(Mode::Lines)
        .name("Original Signal");

    time_plot.add_trace(time_trace);
    time_plot.set_layout(
        Layout::new()
            .title(Title::with_text("Time Domain Signal"))
            .x_axis(Axis::new().title(Title::with_text("Time")))
            .y_axis(Axis::new().title(Title::with_text("Amplitude"))),
    );

    time_plot.write_html("sparse_fft_time_domain.html");

    // Create frequency domain plot
    let mut freq_plot = Plot::new();

    // Full FFT
    let full_trace = Scatter::new(
        (0..full_magnitudes.len()).collect::<Vec<_>>(),
        full_magnitudes.to_vec(),
    )
    .mode(Mode::Lines)
    .name("Full FFT");

    // Sparse FFT
    let sparse_x: Vec<_> = sparse_result.indices.clone();
    let sparse_y: Vec<_> = sparse_result.values.iter().map(|c| c.norm()).collect();

    let sparse_trace = Scatter::new(sparse_x, sparse_y)
        .mode(Mode::Markers)
        .name("Sparse FFT Components");

    freq_plot.add_trace(full_trace);
    freq_plot.add_trace(sparse_trace);
    freq_plot.set_layout(
        Layout::new()
            .title(Title::with_text("Frequency Domain Comparison"))
            .x_axis(Axis::new().title(Title::with_text("Frequency Bin")))
            .y_axis(Axis::new().title(Title::with_text("Magnitude"))),
    );

    freq_plot.write_html("sparse_fft_frequency_domain.html");

    println!("Plots saved as 'sparse_fft_time_domain.html' and 'sparse_fft_frequency_domain.html'");
}
