use plotly::{
    common::{Mode, Title},
    Layout, Plot, Scatter,
};
use scirs2_fft::{
    sparse_fft::{SparseFFTAlgorithm, WindowFunction},
    sparse_fft_gpu::GPUBackend,
    sparse_fft_gpu_cuda::{cuda_sparse_fft, get_cuda_devices, is_cuda_available},
    sparse_fft_gpu_memory::{init_global_memory_manager, AllocationStrategy},
};
use std::f64::consts::PI;
use std::time::Instant;

/// Create a sparse signal with known frequencies in the spectrum
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

/// Benchmark all sparse FFT algorithms and visualize the results
fn benchmark_and_visualize() {
    println!("Running performance benchmarks for sparse FFT algorithms...");

    // Initialize GPU if available
    let cuda_available = is_cuda_available();
    if cuda_available {
        // Initialize memory manager
        let _ = init_global_memory_manager(
            GPUBackend::CUDA,
            0, // First device
            AllocationStrategy::CacheBySize,
            1024 * 1024 * 1024, // 1 GB limit
        );
    }

    // Define algorithms to benchmark
    let algorithms = [
        SparseFFTAlgorithm::Sublinear,
        SparseFFTAlgorithm::CompressedSensing,
        SparseFFTAlgorithm::Iterative,
    ];

    // Define signal sizes
    let sizes = [
        1024,
        4 * 1024,
        16 * 1024,
        64 * 1024,
        256 * 1024,
        1024 * 1024,
    ];

    // Collect benchmark results
    let mut cpu_times = Vec::new();
    let mut gpu_times = Vec::new();
    let mut speedups = Vec::new();

    for &algorithm in &algorithms {
        let mut cpu_size_times = Vec::new();
        let mut gpu_size_times = Vec::new();
        let mut algorithm_speedups = Vec::new();

        for &size in &sizes {
            // Create test signal
            let frequencies = vec![
                (size / 100, 1.0),
                (size / 50, 0.5),
                (size / 20, 0.25),
                (size / 10, 0.15),
                (size / 5, 0.1),
            ];
            let signal = create_sparse_signal(size, &frequencies);

            // CPU benchmark
            let cpu_start = Instant::now();
            let _ = scirs2_fft::sparse_fft::sparse_fft(
                &signal,
                10, // Sparsity
                Some(algorithm),
                Some(WindowFunction::Hann),
            )
            .unwrap();
            let cpu_time = cpu_start.elapsed().as_millis() as f64;
            cpu_size_times.push(cpu_time);

            // GPU benchmark (if available)
            if cuda_available {
                let gpu_start = Instant::now();
                let _ = cuda_sparse_fft(
                    &signal,
                    10, // Sparsity
                    0,  // Device ID
                    Some(algorithm),
                    Some(WindowFunction::Hann),
                )
                .unwrap();
                let gpu_time = gpu_start.elapsed().as_millis() as f64;
                gpu_size_times.push(gpu_time);

                // Calculate speedup
                let speedup = if gpu_time > 0.0 {
                    cpu_time / gpu_time
                } else {
                    0.0
                };
                algorithm_speedups.push(speedup);

                println!(
                    "Algorithm: {:?}, Size: {}, CPU: {:.2} ms, GPU: {:.2} ms, Speedup: {:.2}x",
                    algorithm, size, cpu_time, gpu_time, speedup
                );
            } else {
                gpu_size_times.push(0.0);
                algorithm_speedups.push(0.0);
                println!(
                    "Algorithm: {:?}, Size: {}, CPU: {:.2} ms, GPU: N/A",
                    algorithm, size, cpu_time
                );
            }
        }

        cpu_times.push(cpu_size_times);
        gpu_times.push(gpu_size_times);
        speedups.push(algorithm_speedups);
    }

    // Create performance plots
    let mut cpu_plot = Plot::new();
    let mut gpu_plot = Plot::new();
    let mut speedup_plot = Plot::new();

    // Convert sizes to strings for x-axis labels
    let size_labels: Vec<String> = sizes.iter().map(|&s| format!("{}K", s / 1024)).collect();

    // Add traces for each algorithm
    for (i, &algorithm) in algorithms.iter().enumerate() {
        // CPU times plot
        let cpu_trace = Scatter::new(size_labels.clone(), cpu_times[i].clone())
            .name(format!("{:?}", algorithm))
            .mode(Mode::LinesMarkers);
        cpu_plot.add_trace(cpu_trace);

        if cuda_available {
            // GPU times plot
            let gpu_trace = Scatter::new(size_labels.clone(), gpu_times[i].clone())
                .name(format!("{:?}", algorithm))
                .mode(Mode::LinesMarkers);
            gpu_plot.add_trace(gpu_trace);

            // Speedup plot
            let speedup_trace = Scatter::new(size_labels.clone(), speedups[i].clone())
                .name(format!("{:?}", algorithm))
                .mode(Mode::LinesMarkers);
            speedup_plot.add_trace(speedup_trace);
        }
    }

    // Set layouts
    cpu_plot.set_layout(
        Layout::new()
            .title(Title::with_text("<b>CPU Execution Time</b>"))
            .x_axis(plotly::layout::Axis::new().title(Title::with_text("Signal Size")))
            .y_axis(
                plotly::layout::Axis::new()
                    .title(Title::with_text("Time (ms)"))
                    .type_(plotly::layout::AxisType::Log),
            ),
    );

    if cuda_available {
        gpu_plot.set_layout(
            Layout::new()
                .title(Title::with_text("<b>GPU Execution Time</b>"))
                .x_axis(plotly::layout::Axis::new().title(Title::with_text("Signal Size")))
                .y_axis(
                    plotly::layout::Axis::new()
                        .title(Title::with_text("Time (ms)"))
                        .type_(plotly::layout::AxisType::Log),
                ),
        );

        speedup_plot.set_layout(
            Layout::new()
                .title(Title::with_text("<b>GPU vs CPU Speedup</b>"))
                .x_axis(plotly::layout::Axis::new().title(Title::with_text("Signal Size")))
                .y_axis(plotly::layout::Axis::new().title(Title::with_text("Speedup (x)"))),
        );
    }

    // Save plots
    cpu_plot.write_html("sparse_fft_cpu_performance.html");
    if cuda_available {
        gpu_plot.write_html("sparse_fft_gpu_performance.html");
        speedup_plot.write_html("sparse_fft_speedup.html");
        println!(
            "\nPerformance plots saved as sparse_fft_cpu_performance.html, sparse_fft_gpu_performance.html, and sparse_fft_speedup.html"
        );
    } else {
        println!("\nPerformance plot saved as sparse_fft_cpu_performance.html");
    }
}

/// Compare algorithm accuracy on noisy signals
fn benchmark_accuracy() {
    println!("\nComparing algorithm accuracy with noisy signals...");

    // Define algorithms to benchmark
    let algorithms = [
        SparseFFTAlgorithm::Sublinear,
        SparseFFTAlgorithm::CompressedSensing,
        SparseFFTAlgorithm::Iterative,
    ];

    // Define noise levels
    let noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0];

    // Signal parameters
    let n = 16 * 1024;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25), (350, 0.15), (700, 0.1)];

    // Collect accuracy results
    let mut accuracies = Vec::new();

    for &algorithm in &algorithms {
        let mut algorithm_accuracies = Vec::new();

        for &noise_level in &noise_levels {
            // Create signal with noise
            let mut signal = create_sparse_signal(n, &frequencies);

            // Add noise
            if noise_level > 0.0 {
                use rand_distr::{Distribution, Normal};

                let mut rng = rand::rng();
                let normal = Normal::new(0.0, noise_level).unwrap();

                for sample in &mut signal {
                    *sample += normal.sample(&mut rng);
                }
            }

            // Run sparse FFT
            let result = scirs2_fft::sparse_fft::sparse_fft(
                &signal,
                10, // Sparsity
                Some(algorithm),
                Some(WindowFunction::Hann),
            )
            .unwrap();

            // Calculate accuracy (how many true frequencies were found)
            let mut found_count = 0;
            for &true_freq in frequencies.iter().map(|(f, _)| f) {
                for &found_freq in &result.indices {
                    // Consider frequencies within a small tolerance as matches
                    let tolerance = 2;
                    if (found_freq as i64 - true_freq as i64).abs() <= tolerance {
                        found_count += 1;
                        break;
                    }
                }
            }

            let accuracy = found_count as f64 / frequencies.len() as f64;
            algorithm_accuracies.push(accuracy);

            println!(
                "Algorithm: {:?}, Noise Level: {:.2}, Accuracy: {:.2}",
                algorithm, noise_level, accuracy
            );
        }

        accuracies.push(algorithm_accuracies);
    }

    // Create accuracy plot
    let mut accuracy_plot = Plot::new();

    // Add traces for each algorithm
    for (i, &algorithm) in algorithms.iter().enumerate() {
        let trace = Scatter::new(
            noise_levels
                .iter()
                .map(|&n| format!("{:.2}", n))
                .collect::<Vec<_>>(),
            accuracies[i].clone(),
        )
        .name(format!("{:?}", algorithm))
        .mode(Mode::LinesMarkers);

        accuracy_plot.add_trace(trace);
    }

    // Set layout
    accuracy_plot.set_layout(
        Layout::new()
            .title(Title::with_text("<b>Algorithm Accuracy vs Noise Level</b>"))
            .x_axis(plotly::layout::Axis::new().title(Title::with_text("Noise Level (σ)")))
            .y_axis(
                plotly::layout::Axis::new()
                    .title(Title::with_text("Accuracy"))
                    .range(vec![0.0, 1.0]),
            ),
    );

    // Save plot
    accuracy_plot.write_html("sparse_fft_accuracy.html");
    println!("\nAccuracy plot saved as sparse_fft_accuracy.html");
}

fn main() {
    println!("Sparse FFT Performance Visualization Tool");
    println!("=======================================");

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
    } else {
        println!("\nCUDA is not available. Only CPU benchmarks will be run.");
    }

    // Run benchmarks and generate visualizations
    benchmark_and_visualize();
    benchmark_accuracy();

    println!("\nVisualization completed successfully!");
}
