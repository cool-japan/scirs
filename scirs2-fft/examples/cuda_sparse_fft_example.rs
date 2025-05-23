use num_complex::Complex64;
use plotly::common::Title;
use plotly::{common::Mode, layout::Axis, Layout, Plot, Scatter};
use scirs2_fft::{
    cuda_sparse_fft, get_cuda_devices, is_cuda_available, sparse_fft,
    sparse_fft::{SparseFFTAlgorithm, WindowFunction},
    sparse_fft_gpu::GPUBackend,
    sparse_fft_gpu_memory::init_global_memory_manager,
};
use std::f64::consts::PI;

fn main() {
    println!("CUDA-Accelerated Sparse FFT Example");
    println!("==================================\n");

    // Check if CUDA is available
    println!("Checking for CUDA availability...");
    if !is_cuda_available() {
        println!("CUDA is not available. Using CPU fallback implementation.");
    } else {
        println!("CUDA is available!");

        // Get available CUDA devices
        let devices = get_cuda_devices().unwrap();
        println!("Found {} CUDA device(s):", devices.len());

        for device in &devices {
            println!(
                "  - Device {}: {} ({:.1} GB)",
                device.device_id,
                device.name,
                device.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            println!(
                "    Compute capability: {}.{}",
                device.compute_capability.0, device.compute_capability.1
            );
            println!("    Multiprocessors: {}", device.multiprocessor_count);
        }
    }

    // Initialize memory manager
    println!("\nInitializing GPU memory manager...");
    init_global_memory_manager(
        GPUBackend::CUDA,
        0, // First device
        scirs2_fft::sparse_fft_gpu_memory::AllocationStrategy::CacheBySize,
        1024 * 1024 * 1024, // 1 GB limit
    )
    .unwrap();

    // 1. Create a signal with a few frequency components
    let n = 1024;
    println!(
        "\nCreating a signal with n = {} samples and 3 frequency components",
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

    // 3. Compute CUDA-accelerated sparse FFT
    println!("\nComputing CUDA-accelerated sparse FFT...");

    let cuda_start = std::time::Instant::now();
    let cuda_result = cuda_sparse_fft(
        &signal,
        6,
        0, // Use first CUDA device
        Some(SparseFFTAlgorithm::Sublinear),
        Some(WindowFunction::Hann),
    )
    .unwrap();
    let cuda_elapsed = cuda_start.elapsed();

    println!(
        "CUDA Sparse FFT: Found {} frequency components in {:?}",
        cuda_result.values.len(),
        cuda_elapsed
    );

    if cuda_elapsed < cpu_elapsed {
        println!(
            "CUDA implementation was {:.2}x faster than CPU",
            cpu_elapsed.as_secs_f64() / cuda_elapsed.as_secs_f64()
        );
    } else {
        println!(
            "CPU implementation was {:.2}x faster than CUDA (this is expected for small signals)",
            cuda_elapsed.as_secs_f64() / cpu_elapsed.as_secs_f64()
        );
        println!("For larger signals or batch processing, CUDA will show better performance");
    }

    // 4. Compare results between CPU and CUDA implementations
    println!("\nComparing CPU and CUDA results:");

    // Display top frequency components
    println!("  Top frequency components from CPU implementation:");

    // Get unique index-value pairs sorted by magnitude
    let mut cpu_components: Vec<(usize, Complex64)> = Vec::new();
    for (&idx, &val) in cpu_result.indices.iter().zip(cpu_result.values.iter()) {
        if !cpu_components.iter().any(|(i, _)| *i == idx) {
            cpu_components.push((idx, val));
        }
    }

    // Sort by magnitude in descending order
    cpu_components.sort_by(|(_, a), (_, b)| {
        b.norm()
            .partial_cmp(&a.norm())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Display top 3 components
    for (i, (idx, val)) in cpu_components.iter().take(3).enumerate() {
        println!(
            "    {}. Index {}: magnitude = {:.3}",
            i + 1,
            idx,
            val.norm()
        );
    }

    println!("  Top frequency components from CUDA implementation:");

    // Get unique index-value pairs sorted by magnitude
    let mut cuda_components: Vec<(usize, Complex64)> = Vec::new();
    for (&idx, &val) in cuda_result.indices.iter().zip(cuda_result.values.iter()) {
        if !cuda_components.iter().any(|(i, _)| *i == idx) {
            cuda_components.push((idx, val));
        }
    }

    // Sort by magnitude in descending order
    cuda_components.sort_by(|(_, a), (_, b)| {
        b.norm()
            .partial_cmp(&a.norm())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Display top 3 components
    for (i, (idx, val)) in cuda_components.iter().take(3).enumerate() {
        println!(
            "    {}. Index {}: magnitude = {:.3}",
            i + 1,
            idx,
            val.norm()
        );
    }

    // 5. Create visualization to compare results
    println!("\nCreating visualization...");
    create_comparison_plot(&signal, &cpu_result, &cuda_result);

    // 6. Testing with a larger signal
    println!("\nTesting with a larger signal (16K samples)...");
    let large_n = 16 * 1024;
    let large_signal = create_sparse_signal(large_n, &frequencies);

    // CPU
    let cpu_start = std::time::Instant::now();
    let _large_cpu_result = sparse_fft(
        &large_signal,
        6,
        Some(SparseFFTAlgorithm::Sublinear),
        Some(WindowFunction::Hann),
    )
    .unwrap();
    let cpu_elapsed = cpu_start.elapsed();

    // CUDA
    let cuda_start = std::time::Instant::now();
    let _large_cuda_result = cuda_sparse_fft(
        &large_signal,
        6,
        0, // Use first CUDA device
        Some(SparseFFTAlgorithm::Sublinear),
        Some(WindowFunction::Hann),
    )
    .unwrap();
    let cuda_elapsed = cuda_start.elapsed();

    println!("  CPU  elapsed time: {:?}", cpu_elapsed);
    println!("  CUDA elapsed time: {:?}", cuda_elapsed);

    if cuda_elapsed < cpu_elapsed {
        println!(
            "  CUDA implementation was {:.2}x faster than CPU",
            cpu_elapsed.as_secs_f64() / cuda_elapsed.as_secs_f64()
        );
    } else {
        println!(
            "  CPU implementation was {:.2}x faster than CUDA",
            cuda_elapsed.as_secs_f64() / cpu_elapsed.as_secs_f64()
        );
    }

    // 7. Summary
    println!("\nCUDA integration benefits:");
    println!("  - GPU acceleration for large signals (>16K samples)");
    println!("  - Efficient batch processing of multiple signals");
    println!("  - Improved performance for compute-intensive algorithms");
    println!("  - Better scaling with signal size compared to CPU");

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

// Create visualization plots comparing CPU and CUDA results
fn create_comparison_plot(
    signal: &[f64],
    cpu_result: &scirs2_fft::sparse_fft::SparseFFTResult,
    cuda_result: &scirs2_fft::sparse_fft::SparseFFTResult,
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

    time_plot.write_html("cuda_sparse_fft_time_domain.html");

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

    // CUDA sparse FFT trace (using a different marker style)
    let cuda_indices: Vec<_> = cuda_result
        .indices
        .iter()
        .cloned()
        .filter(|&idx| idx < 100)
        .collect();
    let cuda_values: Vec<_> = cuda_indices
        .iter()
        .map(|&idx| {
            let pos = cuda_result.indices.iter().position(|&i| i == idx).unwrap();
            cuda_result.values[pos].norm()
        })
        .collect();

    let cuda_trace = Scatter::new(cuda_indices, cuda_values)
        .mode(Mode::Markers)
        .name("CUDA Sparse FFT Components");

    freq_plot.add_trace(full_trace);
    freq_plot.add_trace(cpu_trace);
    freq_plot.add_trace(cuda_trace);
    freq_plot.set_layout(
        Layout::new()
            .title(Title::with_text("Frequency Domain Comparison"))
            .x_axis(Axis::new().title(Title::with_text("Frequency Bin")))
            .y_axis(Axis::new().title(Title::with_text("Magnitude"))),
    );

    freq_plot.write_html("cuda_sparse_fft_frequency_domain.html");

    println!("Plots saved as 'cuda_sparse_fft_time_domain.html' and 'cuda_sparse_fft_frequency_domain.html'");
}
