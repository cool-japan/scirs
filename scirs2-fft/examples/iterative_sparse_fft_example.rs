use num_complex::Complex64;
use plotly::{
    common::{Mode, Title},
    Layout, Plot, Scatter,
};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use scirs2_fft::{
    sparse_fft::{reconstruct_time_domain, SparseFFTAlgorithm, SparseFFTResult},
    sparse_fft_cuda_kernels::execute_cuda_sublinear_sparse_fft,
    sparse_fft_cuda_kernels_iterative::execute_cuda_iterative_sparse_fft,
    sparse_fft_gpu::GPUBackend,
    sparse_fft_gpu_cuda::{cuda_sparse_fft, get_cuda_devices},
    sparse_fft_gpu_memory::is_cuda_available,
    sparse_fft_gpu_memory::{init_global_memory_manager, AllocationStrategy},
};
use std::f64::consts::PI;
use std::time::Instant;

/// Create a sparse signal with known frequencies in the spectrum
fn create_sparse_signal(n: usize, frequencies: &[(usize, f64)], noise_level: f64) -> Vec<f64> {
    // Create deterministic RNG for reproducible results
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, noise_level).unwrap();

    let mut signal = vec![0.0; n];

    // Add sinusoidal components
    for i in 0..n {
        let t = 2.0 * PI * (i as f64) / (n as f64);
        for &(freq, amp) in frequencies {
            signal[i] += amp * (freq as f64 * t).sin();
        }
    }

    // Add noise
    if noise_level > 0.0 {
        for sample in &mut signal {
            *sample += normal.sample(&mut rng);
        }
    }

    signal
}

/// Evaluate the accuracy of sparse FFT results given the ground truth frequencies
fn evaluate_accuracy(
    result: &SparseFFTResult,
    true_frequencies: &[(usize, f64)],
    n: usize,
) -> (f64, f64, usize) {
    // Calculate true positive rate (how many true frequencies were found)
    let mut true_positives = 0;
    let mut _false_positives = 0;
    let mut found_indices = vec![false; true_frequencies.len()];

    // For each found frequency, check if it corresponds to a true frequency
    for &idx in &result.indices {
        let mut found = false;
        for (i, &(freq, _)) in true_frequencies.iter().enumerate() {
            // Consider frequencies within a small tolerance window as matches
            let tolerance = std::cmp::max(1, n / 1000);
            if (idx as i64 - freq as i64).abs() <= tolerance as i64 {
                found = true;
                found_indices[i] = true;
                break;
            }
        }

        if found {
            true_positives += 1;
        } else {
            _false_positives += 1; // Keep track of false positives for debugging
        }
    }

    let precision = true_positives as f64 / result.indices.len() as f64;
    let recall = true_positives as f64 / true_frequencies.len() as f64;

    (precision, recall, true_positives)
}

/// Run sparse FFT with multiple algorithms and compare results
fn run_algorithm_comparison(n: usize, sparsity: usize, noise_level: f64) {
    println!("\nRunning algorithm comparison:");
    println!("  Signal size: {}", n);
    println!("  Expected sparsity: {}", sparsity);
    println!("  Noise level: {:.3}", noise_level);

    // Create a sparse signal with known frequencies
    let frequencies = vec![
        (30, 1.0),   // Frequency 30, amplitude 1.0
        (70, 0.5),   // Frequency 70, amplitude 0.5
        (150, 0.25), // Frequency 150, amplitude 0.25
        (350, 0.15), // Frequency 350, amplitude 0.15
        (700, 0.1),  // Frequency 700, amplitude 0.1
    ];

    println!("\nTrue frequency components:");
    for (i, &(freq, amp)) in frequencies.iter().enumerate() {
        println!("  {}. Frequency {} with amplitude {:.3}", i + 1, freq, amp);
    }

    let signal = create_sparse_signal(n, &frequencies, noise_level);

    // Define algorithms to test
    let algorithms = [
        SparseFFTAlgorithm::Sublinear,
        SparseFFTAlgorithm::CompressedSensing,
        SparseFFTAlgorithm::Iterative,
    ];

    // Create plots for visualization
    let mut time_domain_plot = Plot::new();
    let mut frequency_domain_plot = Plot::new();

    // Add time domain signal
    let time_trace = Scatter::new(
        (0..std::cmp::min(n, 500)).collect::<Vec<_>>(),
        signal[0..std::cmp::min(n, 500)].to_vec(),
    )
    .mode(Mode::Lines)
    .name("Time Domain Signal (first 500 samples)");

    time_domain_plot.add_trace(time_trace);

    // Results table
    println!("\nResults:");
    println!(
        "{:<20} {:<15} {:<15} {:<15} {:<15}",
        "Algorithm", "CPU Time (ms)", "GPU Time (ms)", "Precision", "Recall"
    );
    println!("{:-<80}", "");

    // Run each algorithm on CPU and GPU
    for &algorithm in &algorithms {
        // CPU implementation
        let cpu_start = Instant::now();
        let cpu_result =
            scirs2_fft::sparse_fft::sparse_fft(&signal, sparsity, Some(algorithm), None).unwrap();
        let cpu_time = cpu_start.elapsed().as_millis();

        // Evaluate CPU accuracy
        let (cpu_precision, cpu_recall, cpu_true_positives) =
            evaluate_accuracy(&cpu_result, &frequencies, n);

        // GPU implementation (if available)
        let gpu_time;
        let gpu_result;
        let gpu_accuracy;

        if is_cuda_available() {
            // Initialize memory manager if not already done
            let _ = init_global_memory_manager(
                GPUBackend::CUDA,
                0, // First device
                AllocationStrategy::CacheBySize,
                1024 * 1024 * 1024, // 1 GB limit
            );

            let gpu_start = Instant::now();
            gpu_result = cuda_sparse_fft(
                &signal,
                sparsity,
                0, // Device ID
                Some(algorithm),
                None,
            )
            .unwrap();
            gpu_time = gpu_start.elapsed().as_millis();

            // Evaluate GPU accuracy
            gpu_accuracy = evaluate_accuracy(&gpu_result, &frequencies, n);
        } else {
            gpu_time = 0;
            gpu_result = cpu_result.clone();
            gpu_accuracy = (cpu_precision, cpu_recall, cpu_true_positives);
        }

        // Print results
        println!(
            "{:<20} {:<15} {:<15} {:<15.3} {:<15.3}",
            format!("{:?}", algorithm),
            cpu_time,
            if is_cuda_available() {
                gpu_time.to_string()
            } else {
                "N/A".to_string()
            },
            gpu_accuracy.0,
            gpu_accuracy.1
        );

        // Add frequency components to plot
        let mut indices: Vec<usize> = Vec::new();
        let mut amplitudes: Vec<f64> = Vec::new();

        let max_freq_to_plot = n / 2;
        for (&idx, &val) in gpu_result.indices.iter().zip(gpu_result.values.iter()) {
            if idx < max_freq_to_plot {
                indices.push(idx);
                amplitudes.push(val.norm());
            }
        }

        let freq_trace = Scatter::new(indices, amplitudes)
            .mode(Mode::Markers)
            .name(format!("{:?} Algorithm", algorithm));

        frequency_domain_plot.add_trace(freq_trace);

        // Print identified frequencies
        println!(
            "\n  {:?} Algorithm found {} significant frequencies:",
            algorithm,
            gpu_result.values.len()
        );

        // Sort by magnitude
        let mut components: Vec<(usize, Complex64)> = gpu_result
            .indices
            .iter()
            .zip(gpu_result.values.iter())
            .map(|(&idx, &val)| (idx, val))
            .collect();

        components.sort_by(|a, b| b.1.norm().partial_cmp(&a.1.norm()).unwrap());

        for (i, (idx, val)) in components.iter().take(10).enumerate() {
            println!(
                "    {}. Frequency {} with magnitude {:.6}",
                i + 1,
                idx,
                val.norm()
            );
        }
    }

    // Set plot layouts
    time_domain_plot.set_layout(
        Layout::new()
            .title(Title::with_text("<b>Time Domain Signal</b>"))
            .x_axis(plotly::layout::Axis::new().title(Title::with_text("Sample")))
            .y_axis(plotly::layout::Axis::new().title(Title::with_text("Amplitude"))),
    );

    frequency_domain_plot.set_layout(
        Layout::new()
            .title(Title::with_text("<b>Frequency Domain Components</b>"))
            .x_axis(plotly::layout::Axis::new().title(Title::with_text("Frequency Bin")))
            .y_axis(plotly::layout::Axis::new().title(Title::with_text("Magnitude"))),
    );

    // Save plots
    time_domain_plot.write_html("sparse_fft_time_domain.html");
    frequency_domain_plot.write_html("sparse_fft_frequency_domain.html");

    println!("\nPlots saved as sparse_fft_time_domain.html and sparse_fft_frequency_domain.html");
}

/// Run benchmark for different signal sizes
fn run_size_benchmark() {
    println!("\nRunning size benchmark:");

    if !is_cuda_available() {
        println!("CUDA is not available. Skipping GPU benchmarks.");
        return;
    }

    // Initialize memory manager
    let _ = init_global_memory_manager(
        GPUBackend::CUDA,
        0, // First device
        AllocationStrategy::CacheBySize,
        1024 * 1024 * 1024, // 1 GB limit
    );

    // Test different signal sizes
    let sizes = [
        1024,
        4 * 1024,
        16 * 1024,
        64 * 1024,
        256 * 1024,
        1024 * 1024,
    ];

    // Results table
    println!(
        "\n{:<15} {:<20} {:<20} {:<20}",
        "Signal Size", "Sublinear (ms)", "CompressedSensing (ms)", "Iterative (ms)"
    );
    println!("{:-<80}", "");

    for &size in &sizes {
        // Create a signal with fixed sparsity
        let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25), (350, 0.15), (700, 0.1)];
        let signal = create_sparse_signal(size, &frequencies, 0.01);

        // Measure times for each algorithm
        let mut times = Vec::new();

        for algorithm in [
            SparseFFTAlgorithm::Sublinear,
            SparseFFTAlgorithm::CompressedSensing,
            SparseFFTAlgorithm::Iterative,
        ] {
            let start = Instant::now();
            let _ = cuda_sparse_fft(
                &signal,
                6, // Sparsity
                0, // Device ID
                Some(algorithm),
                None,
            )
            .unwrap();
            let elapsed = start.elapsed().as_millis();
            times.push(elapsed);
        }

        println!(
            "{:<15} {:<20} {:<20} {:<20}",
            size, times[0], times[1], times[2]
        );
    }
}

/// Run noise tolerance benchmark
fn run_noise_benchmark() {
    println!("\nRunning noise tolerance benchmark:");

    if !is_cuda_available() {
        println!("CUDA is not available. Skipping GPU benchmarks.");
        return;
    }

    // Initialize memory manager
    let _ = init_global_memory_manager(
        GPUBackend::CUDA,
        0, // First device
        AllocationStrategy::CacheBySize,
        1024 * 1024 * 1024, // 1 GB limit
    );

    // Test different noise levels
    let noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0];

    // Signal parameters
    let n = 16 * 1024;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25), (350, 0.15), (700, 0.1)];

    // Results table
    println!(
        "\n{:<15} {:<15} {:<15} {:<15}",
        "Noise Level", "Sublinear", "CompressedSensing", "Iterative"
    );
    println!("{:-<65}", "");

    for &noise in &noise_levels {
        // Create a signal with the current noise level
        let signal = create_sparse_signal(n, &frequencies, noise);

        // Measure accuracy for each algorithm
        let mut accuracies = Vec::new();

        for algorithm in [
            SparseFFTAlgorithm::Sublinear,
            SparseFFTAlgorithm::CompressedSensing,
            SparseFFTAlgorithm::Iterative,
        ] {
            let result = cuda_sparse_fft(
                &signal,
                6, // Sparsity
                0, // Device ID
                Some(algorithm),
                None,
            )
            .unwrap();

            let (_, recall, _) = evaluate_accuracy(&result, &frequencies, n);
            accuracies.push(recall);
        }

        println!(
            "{:<15.3} {:<15.3} {:<15.3} {:<15.3}",
            noise, accuracies[0], accuracies[1], accuracies[2]
        );
    }
}

/// Demonstrate the impact of iterations on iterative sparse FFT accuracy
fn run_iteration_comparison() {
    println!("\nDemonstrating impact of iterations on Iterative Sparse FFT:");

    if !is_cuda_available() {
        println!("CUDA is not available. Skipping GPU benchmarks.");
        return;
    }

    // Signal parameters
    let n = 16 * 1024;
    let sparsity = 10; // Detect 10 components

    // Create signal with varying component magnitudes
    // The small magnitude components will be harder to detect without multiple iterations
    let frequencies = vec![
        (30, 1.0),    // Very large component
        (70, 0.5),    // Large component
        (150, 0.25),  // Medium component
        (200, 0.2),   // Medium component
        (270, 0.15),  // Medium-small component
        (350, 0.1),   // Small component
        (420, 0.075), // Small component
        (500, 0.05),  // Very small component
        (600, 0.025), // Very small component
        (700, 0.01),  // Extremely small component
    ];

    println!("\nTrue frequency components (in descending magnitude):");
    for (i, &(freq, amp)) in frequencies.iter().enumerate() {
        println!("  {}. Frequency {} with amplitude {:.6}", i + 1, freq, amp);
    }

    let signal = create_sparse_signal(n, &frequencies, 0.005); // Small amount of noise

    // Test with different numbers of iterations
    let iterations_to_test = [1, 2, 3, 5, 8, 10];

    println!(
        "\n{:<15} {:<15} {:<15} {:<20}",
        "Iterations", "True Positives", "Recall", "Time (ms)"
    );
    println!("{:-<65}", "");

    // Create plot for visualization
    let mut plot = Plot::new();

    for &iterations in &iterations_to_test {
        let start = Instant::now();

        // Use our direct implementation instead of the generic cuda_sparse_fft
        let result = execute_cuda_iterative_sparse_fft(&signal, sparsity, iterations).unwrap();

        let elapsed = start.elapsed().as_millis();

        // Evaluate accuracy
        let (_, recall, true_positives) = evaluate_accuracy(&result, &frequencies, n);

        println!(
            "{:<15} {:<15} {:<15.3} {:<20}",
            iterations, true_positives, recall, elapsed
        );

        // Calculate reconstruction error
        let reconstructed = reconstruct_time_domain(&result, n).unwrap();
        let mut error = 0.0;
        for i in 0..n {
            let sample_error = (signal[i] - reconstructed[i].re).abs();
            error += sample_error * sample_error;
        }
        error = (error / n as f64).sqrt();

        // Add to plot
        let trace = Scatter::new(
            vec![iterations],
            vec![recall * 100.0], // Convert to percentage
        )
        .mode(Mode::Markers)
        .marker(plotly::common::Marker::new().size(10 + 2 * iterations))
        .name(format!("{} iterations (RMSE: {:.5})", iterations, error));

        plot.add_trace(trace);

        // Print found components (top 10)
        if iterations == 1 || iterations == *iterations_to_test.last().unwrap() {
            println!("\n  With {} iterations, found components:", iterations);

            // Sort by magnitude
            let mut components: Vec<(usize, Complex64)> = result
                .indices
                .iter()
                .zip(result.values.iter())
                .map(|(&idx, &val)| (idx, val))
                .collect();

            components.sort_by(|a, b| b.1.norm().partial_cmp(&a.1.norm()).unwrap());

            for (i, (idx, val)) in components.iter().take(10).enumerate() {
                // Check if this is a true component
                let mut is_true = false;
                for &(freq, _) in &frequencies {
                    let tolerance = std::cmp::max(1, n / 1000);
                    if (*idx as i64 - freq as i64).abs() <= tolerance as i64 {
                        is_true = true;
                        break;
                    }
                }

                println!(
                    "    {}. Frequency {} with magnitude {:.6} {}",
                    i + 1,
                    idx,
                    val.norm(),
                    if is_true {
                        "(TRUE)"
                    } else {
                        "(False positive)"
                    }
                );
            }
        }
    }

    // Set plot layout
    plot.set_layout(
        Layout::new()
            .title(Title::with_text(
                "<b>Effect of Iterations on Iterative Sparse FFT Accuracy</b>",
            ))
            .x_axis(plotly::layout::Axis::new().title(Title::with_text("Number of Iterations")))
            .y_axis(
                plotly::layout::Axis::new()
                    .title(Title::with_text("Recall (%)"))
                    .range(vec![0.0, 100.0]),
            ),
    );

    // Save plot
    plot.write_html("iterative_sparse_fft_iterations.html");
    println!("\nPlot saved as iterative_sparse_fft_iterations.html");

    // Compare with sublinear algorithm
    println!("\nComparing with Sublinear algorithm (single pass):");
    let start = Instant::now();
    let sublinear_result =
        execute_cuda_sublinear_sparse_fft(&signal, sparsity, SparseFFTAlgorithm::Sublinear)
            .unwrap();
    let elapsed = start.elapsed().as_millis();

    let (_, recall, true_positives) = evaluate_accuracy(&sublinear_result, &frequencies, n);
    println!(
        "Sublinear: Found {}/{} components (Recall: {:.3}) in {} ms",
        true_positives,
        frequencies.len(),
        recall,
        elapsed
    );

    // Calculate reconstruction error
    let reconstructed = reconstruct_time_domain(&sublinear_result, n).unwrap();
    let mut error = 0.0;
    for i in 0..n {
        let sample_error = (signal[i] - reconstructed[i].re).abs();
        error += sample_error * sample_error;
    }
    error = (error / n as f64).sqrt();
    println!("Sublinear reconstruction RMSE: {:.6}", error);

    // Print found components (top 10)
    println!("\n  Sublinear algorithm found components:");

    // Sort by magnitude
    let mut components: Vec<(usize, Complex64)> = sublinear_result
        .indices
        .iter()
        .zip(sublinear_result.values.iter())
        .map(|(&idx, &val)| (idx, val))
        .collect();

    components.sort_by(|a, b| b.1.norm().partial_cmp(&a.1.norm()).unwrap());

    for (i, (idx, val)) in components.iter().take(10).enumerate() {
        // Check if this is a true component
        let mut is_true = false;
        for &(freq, _) in &frequencies {
            let tolerance = std::cmp::max(1, n / 1000);
            if (*idx as i64 - freq as i64).abs() <= tolerance as i64 {
                is_true = true;
                break;
            }
        }

        println!(
            "    {}. Frequency {} with magnitude {:.6} {}",
            i + 1,
            idx,
            val.norm(),
            if is_true {
                "(TRUE)"
            } else {
                "(False positive)"
            }
        );
    }
}

fn main() {
    println!("GPU-Accelerated Iterative Sparse FFT Example");
    println!("============================================");

    // Check CUDA availability
    if is_cuda_available() {
        let devices = get_cuda_devices().unwrap();
        println!("\nCUDA is available with {} device(s):", devices.len());

        for (idx, device) in devices.iter().enumerate() {
            println!("  - Device {} (initialized: {})", idx, device.initialized);
        }
    } else {
        println!("\nCUDA is not available. This example will use CPU implementations.");
    }

    // First demonstrate the impact of iterations
    run_iteration_comparison();

    // Run algorithm comparison
    run_algorithm_comparison(16 * 1024, 6, 0.05);

    // Run benchmarks
    run_size_benchmark();
    run_noise_benchmark();

    println!("\nExample completed successfully!");
}
