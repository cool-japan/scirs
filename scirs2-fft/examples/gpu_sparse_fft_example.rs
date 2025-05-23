use num_complex::Complex64;
use plotly::common::Title;
use plotly::{common::Mode, layout::Axis, Layout, Plot, Scatter};
use scirs2_fft::{
    sparse_fft,
    sparse_fft::{SparseFFTAlgorithm, WindowFunction},
    sparse_fft_gpu::{gpu_sparse_fft, GPUBackend},
};
use std::f64::consts::PI;

fn main() {
    println!("GPU-Accelerated Sparse FFT Example");
    println!("==================================\n");

    // 1. Create a signal with a few frequency components
    let n = 1024;
    println!(
        "Creating a signal with n = {} samples and 3 frequency components",
        n
    );
    let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25)];
    let signal = create_sparse_signal(n, &frequencies);

    // 2. Compute regular CPU sparse FFT for comparison
    println!("\nComputing regular CPU sparse FFT for comparison...");
    let cpu_start = std::time::Instant::now();
    let cpu_result = sparse_fft(
        &signal,
        6,
        Some(SparseFFTAlgorithm::Sublinear),
        Some(WindowFunction::Hann),
    )
    .unwrap();
    let cpu_elapsed = cpu_start.elapsed();

    println!(
        "CPU Sparse FFT: Found {} frequency components in {:?}",
        cpu_result.values.len(),
        cpu_elapsed
    );

    // 3. Compute GPU-accelerated sparse FFT
    println!("\nComputing GPU-accelerated sparse FFT...");
    println!("Note: Currently using CPU fallback as actual GPU implementation is pending");

    let backends = [
        GPUBackend::CPUFallback,
        GPUBackend::CUDA,
        GPUBackend::HIP,
        GPUBackend::SYCL,
    ];

    // We'll actually only run the CPU fallback as the other backends are simulated
    for &backend in &backends[0..1] {
        println!("\n* Using {:?} backend", backend);
        let gpu_start = std::time::Instant::now();
        let gpu_result = gpu_sparse_fft(
            &signal,
            6,
            backend,
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
        )
        .unwrap();
        let gpu_elapsed = gpu_start.elapsed();

        println!(
            "  - Found {} frequency components in {:?}",
            gpu_result.values.len(),
            gpu_elapsed
        );

        // Display top frequency components
        println!("  - Top frequency components:");

        // Get unique index-value pairs sorted by magnitude
        let mut unique_components: Vec<(usize, Complex64)> = Vec::new();
        for (&idx, &val) in gpu_result.indices.iter().zip(gpu_result.values.iter()) {
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
    }

    // 4. Create visualization to compare results
    println!("\nCreating visualization...");
    create_comparison_plot(&signal, &cpu_result, &cpu_result); // Use CPU result twice as GPU is simulated

    // 5. Future capabilities (to be implemented)
    println!("\nFuture GPU acceleration capabilities:");
    println!("- CUDA/ROCm/SYCL backend support for high-performance computing");
    println!("- Batch processing of multiple signals");
    println!("- Mixed precision computation for improved performance");
    println!("- Automatic memory management for large signals");
    println!("- Multi-stream concurrent execution");
    println!("- Hybrid CPU/GPU execution for optimal resource utilization");

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

// Create visualization plots comparing CPU and GPU results
fn create_comparison_plot(
    signal: &[f64],
    cpu_result: &scirs2_fft::sparse_fft::SparseFFTResult,
    gpu_result: &scirs2_fft::sparse_fft::SparseFFTResult,
) {
    // Create time domain plot
    let mut time_plot = Plot::new();
    let time_trace = Scatter::new((0..200).collect::<Vec<_>>(), signal[0..200].to_vec())
        .mode(Mode::Lines)
        .name("Original Signal (first 200 samples)");

    time_plot.add_trace(time_trace);
    time_plot.set_layout(
        Layout::new()
            .title(Title::with_text("Time Domain Signal"))
            .x_axis(Axis::new().title(Title::with_text("Time")))
            .y_axis(Axis::new().title(Title::with_text("Amplitude"))),
    );

    time_plot.write_html("gpu_sparse_fft_time_domain.html");

    // Create frequency domain comparison plot
    let mut freq_plot = Plot::new();

    // Compute full spectrum for comparison
    let signal_complex: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    let full_spectrum = scirs2_fft::fft(&signal_complex, None).unwrap();
    let full_magnitudes: Vec<f64> = full_spectrum.iter().map(|c| c.norm()).collect();

    // Full FFT trace
    let full_trace = Scatter::new(
        (0..full_magnitudes.len().min(100)).collect::<Vec<_>>(),
        full_magnitudes[0..full_magnitudes.len().min(100)].to_vec(),
    )
    .mode(Mode::Lines)
    .name("Full FFT (first 100 bins)");

    // CPU sparse FFT trace
    let cpu_indices: Vec<_> = cpu_result
        .indices
        .iter()
        .cloned()
        .filter(|&idx| idx < 100)
        .collect();
    let cpu_values: Vec<_> = cpu_indices
        .iter()
        .map(|&idx| {
            let pos = cpu_result.indices.iter().position(|&i| i == idx).unwrap();
            cpu_result.values[pos].norm()
        })
        .collect();

    let cpu_trace = Scatter::new(cpu_indices, cpu_values)
        .mode(Mode::Markers)
        .name("CPU Sparse FFT Components");

    // GPU sparse FFT trace (using a different marker style)
    let gpu_indices: Vec<_> = gpu_result
        .indices
        .iter()
        .cloned()
        .filter(|&idx| idx < 100)
        .collect();
    let gpu_values: Vec<_> = gpu_indices
        .iter()
        .map(|&idx| {
            let pos = gpu_result.indices.iter().position(|&i| i == idx).unwrap();
            gpu_result.values[pos].norm()
        })
        .collect();

    let gpu_trace = Scatter::new(gpu_indices, gpu_values)
        .mode(Mode::Markers)
        .name("GPU Sparse FFT Components");

    freq_plot.add_trace(full_trace);
    freq_plot.add_trace(cpu_trace);
    freq_plot.add_trace(gpu_trace);
    freq_plot.set_layout(
        Layout::new()
            .title(Title::with_text("Frequency Domain Comparison"))
            .x_axis(Axis::new().title(Title::with_text("Frequency Bin")))
            .y_axis(Axis::new().title(Title::with_text("Magnitude"))),
    );

    freq_plot.write_html("gpu_sparse_fft_frequency_domain.html");

    println!("Plots saved as 'gpu_sparse_fft_time_domain.html' and 'gpu_sparse_fft_frequency_domain.html'");
}
