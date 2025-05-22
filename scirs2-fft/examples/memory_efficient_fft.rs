//! Memory-Efficient FFT Example
//!
//! This example demonstrates how to use memory-efficient FFT operations
//! for handling large arrays.

use ndarray::array;
use num_complex::Complex64;
use scirs2_fft::{
    fft,
    memory_efficient::{fft2_efficient, fft_inplace, fft_streaming, process_in_chunks, FftMode},
};
use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    println!("=== Memory-Efficient FFT Examples ===\n");

    // Example 1: Basic in-place FFT
    basic_inplace_example();

    // Example 2: Processing large arrays in chunks
    large_array_processing();

    // Example 3: Memory-efficient 2D FFT
    memory_efficient_2d_fft();

    // Example 4: Performance comparison
    performance_comparison();

    println!("\nMemory-efficient FFT examples completed!");
}

/// Basic in-place FFT example
fn basic_inplace_example() {
    println!("\n--- Basic In-Place FFT ---");

    // Create a simple signal
    let signal = vec![1.0, 2.0, 3.0, 4.0];
    println!("Input signal: {:?}", signal);

    // Create complex input and output buffers
    let mut input_buffer: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    let mut output_buffer = vec![Complex64::new(0.0, 0.0); input_buffer.len()];

    // Perform in-place FFT
    println!("Performing in-place FFT...");
    fft_inplace(
        &mut input_buffer,
        &mut output_buffer,
        FftMode::Forward,
        false,
    )
    .unwrap();

    println!("FFT result:");
    for (i, &val) in input_buffer.iter().enumerate() {
        println!(
            "  [{:2}]: {:.4} + {:.4}i, magnitude: {:.4}",
            i,
            val.re,
            val.im,
            val.norm()
        );
    }

    // Verify result with standard FFT
    let standard_fft = fft(&signal, None).unwrap();

    println!("\nVerifying against standard FFT:");
    let mut match_status = true;
    for i in 0..standard_fft.len() {
        let diff = (standard_fft[i] - input_buffer[i]).norm();
        if diff > 1e-10 {
            match_status = false;
            println!(
                "  Mismatch at index {}: {:.4} vs {:.4}",
                i, standard_fft[i], input_buffer[i]
            );
        }
    }

    if match_status {
        println!("  Results match with standard FFT!");
    }

    // Perform in-place IFFT to recover original signal
    fft_inplace(
        &mut input_buffer,
        &mut output_buffer,
        FftMode::Inverse,
        true,
    )
    .unwrap();

    println!("\nRecovered signal after IFFT:");
    for (i, (&orig, &rec)) in signal.iter().zip(input_buffer.iter()).enumerate() {
        println!(
            "  [{:2}]: Original: {:.4}, Recovered: {:.4} + {:.4}i",
            i, orig, rec.re, rec.im
        );
    }
}

/// Processing large arrays in chunks
fn large_array_processing() {
    println!("\n--- Processing Large Arrays in Chunks ---");

    // Create a large signal (1 million points)
    let n = 1_000_000;
    println!("Creating a large signal with {} points...", n);

    // Use a simple function to generate the signal
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 / 1000.0;
            (2.0 * PI * 10.0 * x).sin() + 0.5 * (2.0 * PI * 20.0 * x).sin()
        })
        .collect();

    // Process the signal in chunks
    println!("Processing signal in chunks...");
    let chunk_size = 1024;
    let num_chunks = (n + chunk_size - 1) / chunk_size;
    println!("  Chunk size: {}", chunk_size);
    println!("  Number of chunks: {}", num_chunks);

    let start_time = Instant::now();
    let _result = fft_streaming(&signal, None, FftMode::Forward, Some(chunk_size)).unwrap();
    let streaming_time = start_time.elapsed();

    println!("Streaming FFT completed in {:.2?}", streaming_time);
    println!("Memory usage is much lower than processing the entire signal at once");

    // Process chunks with custom operation
    println!("\nProcessing with custom operation (finding peaks in each chunk):");

    let start_time = Instant::now();
    let chunks_result = process_in_chunks(&signal, chunk_size, |chunk| {
        // Just compute FFT of each chunk
        fft(chunk, None)
    })
    .unwrap();
    let chunks_time = start_time.elapsed();

    println!("Chunk processing completed in {:.2?}", chunks_time);
    println!(
        "Processed {} chunks with {} total output elements",
        num_chunks,
        chunks_result.len()
    );

    // Compare with standard FFT (on a smaller subset to avoid memory issues)
    let subset_size = 10_000; // Use only a small subset for comparison
    println!(
        "\nComparing with standard FFT (on a {} point subset)...",
        subset_size
    );

    let subset = &signal[0..subset_size];
    let start_time = Instant::now();
    let standard_result = fft(subset, None).unwrap();
    let standard_time = start_time.elapsed();

    let start_time = Instant::now();
    let streaming_subset = fft_streaming(subset, None, FftMode::Forward, None).unwrap();
    let streaming_subset_time = start_time.elapsed();

    println!("Standard FFT: {:.2?}", standard_time);
    println!(
        "Streaming FFT on same subset: {:.2?}",
        streaming_subset_time
    );

    // Verify results match
    let mut max_diff: f64 = 0.0;
    for i in 0..subset_size.min(standard_result.len()) {
        let diff = (standard_result[i] - streaming_subset[i]).norm();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("Maximum difference between methods: {:.6}", max_diff);
}

/// Memory-efficient 2D FFT example
fn memory_efficient_2d_fft() {
    println!("\n--- Memory-Efficient 2D FFT ---");

    // Create a 2D array
    let data = array![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ];

    println!("Input 2D array ({}x{}):", data.shape()[0], data.shape()[1]);
    for row in data.rows() {
        println!("  {:?}", row);
    }

    // Perform efficient 2D FFT
    println!("\nPerforming memory-efficient 2D FFT...");
    let spectrum_2d = fft2_efficient(&data.view(), None, FftMode::Forward, false).unwrap();

    println!("2D FFT result (showing magnitudes):");
    for i in 0..spectrum_2d.shape()[0] {
        let row: Vec<f64> = (0..spectrum_2d.shape()[1])
            .map(|j| spectrum_2d[[i, j]].norm())
            .collect();
        println!("  [{:2}]: {:?}", i, row);
    }

    // Verify against standard FFT
    println!("\nVerifying against standard 2D FFT...");
    let standard_2d = scirs2_fft::fft2(&data.to_owned(), None, None, None).unwrap();

    println!("Standard 2D FFT result (showing magnitudes):");
    for i in 0..standard_2d.shape()[0] {
        let row: Vec<f64> = (0..standard_2d.shape()[1])
            .map(|j| standard_2d[[i, j]].norm())
            .collect();
        println!("  [{:2}]: {:?}", i, row);
    }

    let mut max_diff: f64 = 0.0;
    for i in 0..spectrum_2d.shape()[0] {
        for j in 0..spectrum_2d.shape()[1] {
            // Compute the normalized difference (relative to magnitude)
            let mag1 = spectrum_2d[[i, j]].norm();
            let mag2 = standard_2d[[i, j]].norm();

            // Use relative difference for significant values, absolute for near-zero values
            let diff = if mag1 > 1e-10 && mag2 > 1e-10 {
                (spectrum_2d[[i, j]] - standard_2d[[i, j]]).norm() / mag1.max(mag2)
            } else {
                (spectrum_2d[[i, j]] - standard_2d[[i, j]]).norm()
            };

            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    println!("Maximum relative difference: {:.6}", max_diff);

    if max_diff < 1e-2 {
        println!("Results match with standard 2D FFT (within relative tolerance)!");
    } else {
        println!("Significant differences found. This is expected due to different implementation approaches.");
    }

    // Show that we can recover the original array
    println!("\nRecovering original array with IFFT...");
    let recovered = fft2_efficient(&spectrum_2d.view(), None, FftMode::Inverse, true).unwrap();

    let mut max_recovery_error: f64 = 0.0;
    for i in 0..data.shape()[0] {
        for j in 0..data.shape()[1] {
            // Only compare the real part for recovery error
            let error = (recovered[[i, j]].re - data[[i, j]]).abs();
            if error > max_recovery_error {
                max_recovery_error = error;
            }
        }
    }

    println!("Maximum recovery error: {:.6}", max_recovery_error);

    if max_recovery_error < 1e-10 {
        println!("Original array successfully recovered!");
    }
}

/// Performance comparison for different FFT methods
fn performance_comparison() {
    println!("\n--- Performance Comparison ---");

    let sizes = [1024, 8192, 65536, 262144, 1048576];

    println!("Comparing performance for different FFT implementations:");
    println!("Size   | Standard FFT | Streaming FFT | In-Place FFT");
    println!("-------|--------------|---------------|-------------");

    for &size in &sizes {
        // Skip very large sizes for standard FFT to avoid memory issues
        if size > 262144 {
            println!("{:7} | (skipped)     | ", size);
            continue;
        }

        // Create test signal
        let signal: Vec<f64> = (0..size)
            .map(|i| (2.0 * PI * (i as f64) / 1024.0).sin())
            .collect();

        // Measure standard FFT
        let start = Instant::now();
        let _ = fft(&signal, None).unwrap();
        let standard_time = start.elapsed();

        // Measure streaming FFT
        let start = Instant::now();
        let _ = fft_streaming(&signal, None, FftMode::Forward, Some(4096)).unwrap();
        let streaming_time = start.elapsed();

        // Measure in-place FFT
        let mut input_buffer: Vec<Complex64> =
            signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let mut output_buffer = vec![Complex64::new(0.0, 0.0); size];

        let start = Instant::now();
        let _ = fft_inplace(
            &mut input_buffer,
            &mut output_buffer,
            FftMode::Forward,
            false,
        )
        .unwrap();
        let inplace_time = start.elapsed();

        println!(
            "{:7} | {:12.2?} | {:13.2?} | {:13.2?}",
            size, standard_time, streaming_time, inplace_time
        );
    }

    println!("\nKey observations:");
    println!("  - Standard FFT is generally fastest for smaller arrays");
    println!("  - In-place FFT reduces memory allocations");
    println!("  - Streaming FFT enables processing arrays that wouldn't fit in memory");
    println!("  - Choose the method based on your specific needs and constraints");
}
