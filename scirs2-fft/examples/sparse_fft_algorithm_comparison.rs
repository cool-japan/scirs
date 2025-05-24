use plotly::{
    common::{Mode, Title},
    Layout, Plot, Scatter,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use scirs2_fft::{
    sparse_fft::{SparseFFTAlgorithm, SparseFFTResult, WindowFunction},
    sparse_fft_gpu::GPUBackend,
    sparse_fft_gpu_cuda::{cuda_sparse_fft, get_cuda_devices, is_cuda_available},
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

/// Calculate relative error between identified frequencies and ground truth
fn calculate_frequency_error(result: &SparseFFTResult, true_frequencies: &[(usize, f64)]) -> f64 {
    let mut min_error_sum = 0.0;
    let mut found_count = 0;

    // For each true frequency, find the closest detected frequency
    for &(true_freq, _true_amp) in true_frequencies {
        let mut min_error = std::f64::MAX;

        // Find the closest detected frequency
        for (_i, &detected_freq) in result.indices.iter().enumerate() {
            let error =
                (detected_freq as f64 - true_freq as f64).abs() / (true_freq as f64).max(1.0);
            if error < min_error {
                min_error = error;
                if min_error < 0.05 {
                    // Consider it found if within 5% error
                    found_count += 1;
                }
            }
        }

        min_error_sum += min_error;
    }

    // Return average error
    if found_count > 0 {
        min_error_sum / found_count as f64
    } else {
        1.0 // All frequencies missed
    }
}

/// Run a benchmark of all algorithms on signals with different characteristics
fn run_algorithm_benchmarks() {
    // Print header
    println!(
        "\n{:<15} {:<15} {:<15} {:<15} {:<15} {:<15}",
        "Signal Type", "Size", "Sublinear", "CompressedSensing", "Iterative", "FrequencyPruning"
    );
    println!("{:-<80}", "");

    // List of signal sizes to test
    let signal_sizes = [1024, 8 * 1024, 64 * 1024, 256 * 1024];

    // List of noise levels to test
    let noise_levels = [0.0, 0.05, 0.2, 0.5];

    // Initialize GPU if available
    if is_cuda_available() {
        let _ = init_global_memory_manager(
            GPUBackend::CUDA,
            0, // First device
            AllocationStrategy::CacheBySize,
            1024 * 1024 * 1024, // 1 GB limit
        );
    }

    // List of algorithms to benchmark
    let algorithms = [
        SparseFFTAlgorithm::Sublinear,
        SparseFFTAlgorithm::CompressedSensing,
        SparseFFTAlgorithm::Iterative,
        SparseFFTAlgorithm::FrequencyPruning,
    ];

    // Test over different signal sizes
    for &size in &signal_sizes {
        // Create frequency components (scaled by signal size)
        let frequencies = vec![
            (size / 100, 1.0),
            (size / 40, 0.5),
            (size / 20, 0.25),
            (size / 10, 0.15),
            (size / 5, 0.1),
        ];

        // Test clean signal
        let clean_signal = create_sparse_signal(size, &frequencies, 0.0);
        let mut clean_times = Vec::new();

        for &algorithm in &algorithms {
            let start = Instant::now();
            let _ = cuda_sparse_fft(
                &clean_signal,
                10, // Sparsity parameter
                0,  // Device ID
                Some(algorithm),
                Some(WindowFunction::Hann),
            )
            .unwrap();
            let elapsed = start.elapsed().as_millis();
            clean_times.push(elapsed);
        }

        println!(
            "{:<15} {:<15} {:<15} {:<15} {:<15} {:<15}",
            "Clean", size, clean_times[0], clean_times[1], clean_times[2], clean_times[3]
        );

        // Test different noise levels
        for &noise in &noise_levels {
            if noise > 0.0 {
                let noisy_signal = create_sparse_signal(size, &frequencies, noise);
                let mut noisy_times = Vec::new();

                for &algorithm in &algorithms {
                    let start = Instant::now();
                    let _ = cuda_sparse_fft(
                        &noisy_signal,
                        10, // Sparsity parameter
                        0,  // Device ID
                        Some(algorithm),
                        Some(WindowFunction::Hann),
                    )
                    .unwrap();
                    let elapsed = start.elapsed().as_millis();
                    noisy_times.push(elapsed);
                }

                println!(
                    "{:<15} {:<15} {:<15} {:<15} {:<15} {:<15}",
                    format!("Noise {:.2}", noise),
                    size,
                    noisy_times[0],
                    noisy_times[1],
                    noisy_times[2],
                    noisy_times[3]
                );
            }
        }
    }
}

/// Analyze algorithm accuracy on signals with increasing noise levels
fn analyze_algorithm_accuracy() {
    println!("\nAnalyzing algorithm accuracy with increasing noise levels:");

    // Signal parameters
    let n = 16 * 1024;
    let frequencies = vec![(30, 1.0), (100, 0.5), (300, 0.25), (700, 0.1), (1500, 0.05)];

    // List of noise levels to test
    let noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0];

    // Create plots
    let mut accuracy_plot = Plot::new();
    let mut time_plot = Plot::new();

    // Test each algorithm
    let algorithms = [
        SparseFFTAlgorithm::Sublinear,
        SparseFFTAlgorithm::CompressedSensing,
        SparseFFTAlgorithm::Iterative,
        SparseFFTAlgorithm::FrequencyPruning,
    ];

    // Set up result arrays
    let mut algorithm_errors = vec![Vec::new(); algorithms.len()];
    let mut algorithm_times = vec![Vec::new(); algorithms.len()];

    // Print header
    println!(
        "\n{:<15} {:<20} {:<20} {:<15}",
        "Noise Level", "Algorithm", "Frequency Error (%)", "Time (ms)"
    );
    println!("{:-<75}", "");

    // Test over different noise levels
    for &noise in &noise_levels {
        // Create signal with this noise level
        let signal = create_sparse_signal(n, &frequencies, noise);

        // Test each algorithm
        for (i, &algorithm) in algorithms.iter().enumerate() {
            // Run the algorithm
            let start = Instant::now();
            let result = cuda_sparse_fft(
                &signal,
                10, // Sparsity parameter (more than actual to ensure we find all)
                0,  // Device ID
                Some(algorithm),
                Some(WindowFunction::Hann),
            )
            .unwrap();
            let elapsed = start.elapsed().as_millis();

            // Calculate error
            let error = calculate_frequency_error(&result, &frequencies);
            let error_percent = error * 100.0;

            // Store results
            algorithm_errors[i].push(error_percent);
            algorithm_times[i].push(elapsed as f64);

            // Print results
            println!(
                "{:<15.3} {:<20} {:<20.2} {:<15}",
                noise,
                format!("{:?}", algorithm),
                error_percent,
                elapsed
            );
        }
    }

    // Add traces to plots
    for (i, &algorithm) in algorithms.iter().enumerate() {
        // Accuracy plot
        let error_trace = Scatter::new(
            noise_levels
                .iter()
                .map(|&n| format!("{:.2}", n))
                .collect::<Vec<_>>(),
            algorithm_errors[i].clone(),
        )
        .name(format!("{:?}", algorithm))
        .mode(Mode::LinesMarkers);

        accuracy_plot.add_trace(error_trace);

        // Time plot
        let time_trace = Scatter::new(
            noise_levels
                .iter()
                .map(|&n| format!("{:.2}", n))
                .collect::<Vec<_>>(),
            algorithm_times[i].clone(),
        )
        .name(format!("{:?}", algorithm))
        .mode(Mode::LinesMarkers);

        time_plot.add_trace(time_trace);
    }

    // Set layouts
    accuracy_plot.set_layout(
        Layout::new()
            .title(Title::with_text("<b>Algorithm Accuracy vs Noise Level</b>"))
            .x_axis(plotly::layout::Axis::new().title(Title::with_text("Noise Level (σ)")))
            .y_axis(
                plotly::layout::Axis::new()
                    .title(Title::with_text("Frequency Error (%)"))
                    .range(vec![0.0, 30.0]),
            ),
    );

    time_plot.set_layout(
        Layout::new()
            .title(Title::with_text(
                "<b>Algorithm Execution Time vs Noise Level</b>",
            ))
            .x_axis(plotly::layout::Axis::new().title(Title::with_text("Noise Level (σ)")))
            .y_axis(
                plotly::layout::Axis::new()
                    .title(Title::with_text("Execution Time (ms)"))
                    .type_(plotly::layout::AxisType::Log),
            ),
    );

    // Save plots
    accuracy_plot.write_html("sparse_fft_algorithm_accuracy.html");
    time_plot.write_html("sparse_fft_algorithm_timing.html");

    println!(
        "\nPlots saved as sparse_fft_algorithm_accuracy.html and sparse_fft_algorithm_timing.html"
    );
}

/// Analyze the performance of different algorithms with increasing signal size
fn analyze_scaling_behavior() {
    println!("\nAnalyzing algorithm scaling behavior with increasing signal size:");

    // Signal sizes to test
    let sizes = [
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
    ];

    // Create plots
    let mut scaling_plot = Plot::new();

    // Test each algorithm
    let algorithms = [
        SparseFFTAlgorithm::Sublinear,
        SparseFFTAlgorithm::CompressedSensing,
        SparseFFTAlgorithm::Iterative,
        SparseFFTAlgorithm::FrequencyPruning,
    ];

    // Set up result arrays
    let mut algorithm_times = vec![Vec::new(); algorithms.len()];

    // Print header
    println!(
        "\n{:<15} {:<20} {:<15}",
        "Signal Size", "Algorithm", "Time (ms)"
    );
    println!("{:-<55}", "");

    // Test over different signal sizes
    for &size in &sizes {
        // Create frequency components (scaled by signal size)
        let frequencies = vec![
            (size / 100, 1.0),
            (size / 40, 0.5),
            (size / 20, 0.25),
            (size / 10, 0.15),
            (size / 5, 0.1),
        ];

        // Create signal
        let signal = create_sparse_signal(size, &frequencies, 0.05); // Light noise

        // Test each algorithm
        for (i, &algorithm) in algorithms.iter().enumerate() {
            // Run the algorithm
            let start = Instant::now();
            let _ = cuda_sparse_fft(
                &signal,
                10, // Sparsity parameter
                0,  // Device ID
                Some(algorithm),
                Some(WindowFunction::Hann),
            )
            .unwrap();
            let elapsed = start.elapsed().as_millis();

            // Store results
            algorithm_times[i].push(elapsed as f64);

            // Print results
            println!(
                "{:<15} {:<20} {:<15}",
                size,
                format!("{:?}", algorithm),
                elapsed
            );
        }
    }

    // Add traces to plot
    for (i, &algorithm) in algorithms.iter().enumerate() {
        // Scaling plot
        let time_trace = Scatter::new(
            sizes.iter().map(|&s| format!("{}", s)).collect::<Vec<_>>(),
            algorithm_times[i].clone(),
        )
        .name(format!("{:?}", algorithm))
        .mode(Mode::LinesMarkers);

        scaling_plot.add_trace(time_trace);
    }

    // Set layout
    scaling_plot.set_layout(
        Layout::new()
            .title(Title::with_text(
                "<b>Algorithm Scaling with Signal Size</b>",
            ))
            .x_axis(
                plotly::layout::Axis::new()
                    .title(Title::with_text("Signal Size (samples)"))
                    .type_(plotly::layout::AxisType::Log),
            )
            .y_axis(
                plotly::layout::Axis::new()
                    .title(Title::with_text("Execution Time (ms)"))
                    .type_(plotly::layout::AxisType::Log),
            ),
    );

    // Save plot
    scaling_plot.write_html("sparse_fft_algorithm_scaling.html");

    println!("\nPlot saved as sparse_fft_algorithm_scaling.html");
}

fn main() {
    println!("GPU-Accelerated Sparse FFT Algorithm Comparison");
    println!("===============================================");

    // Check CUDA availability
    if is_cuda_available() {
        let devices = get_cuda_devices().unwrap();
        println!("\nCUDA is available with {} device(s):", devices.len());

        for device in &devices {
            println!(
                "  - {} (Device {}, Compute Capability {}.{})",
                device.name,
                device.device_id,
                device.compute_capability.0,
                device.compute_capability.1
            );
        }

        // Initialize GPU memory manager
        let _ = init_global_memory_manager(
            GPUBackend::CUDA,
            0, // First device
            AllocationStrategy::CacheBySize,
            1024 * 1024 * 1024, // 1 GB limit
        );

        // Run benchmarks
        run_algorithm_benchmarks();

        // Analyze accuracy
        analyze_algorithm_accuracy();

        // Analyze scaling behavior
        analyze_scaling_behavior();

        println!("\nAnalysis completed successfully!");
    } else {
        println!("\nCUDA is not available. This example requires GPU acceleration.");
    }
}
