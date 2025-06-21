//! Example demonstrating memory-efficient STFT processing for large signals
//!
//! This example shows how to use the memory-efficient STFT implementation
//! to process large signals without consuming excessive memory.

use scirs2_signal::stft::{MemoryEfficientStft, MemoryEfficientStftConfig, StftConfig};
use scirs2_signal::window;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Memory-Efficient STFT Examples");
    println!("===============================\n");

    // Example 1: Basic memory-efficient STFT
    println!("1. Basic Memory-Efficient STFT");
    basic_memory_efficient_stft()?;
    println!();

    // Example 2: Auto memory management
    println!("2. Automatic Memory Management");
    auto_memory_management()?;
    println!();

    // Example 3: Processing very large signals
    println!("3. Large Signal Processing");
    large_signal_processing()?;
    println!();

    // Example 4: Memory usage analysis
    println!("4. Memory Usage Analysis");
    memory_usage_analysis()?;
    println!();

    // Example 5: Magnitude-only processing
    println!("5. Magnitude-Only Processing");
    magnitude_only_processing()?;
    println!();

    #[cfg(feature = "parallel")]
    {
        println!("6. Parallel Chunked Processing");
        parallel_processing()?;
        println!();
    }

    println!("Memory-efficient STFT examples completed successfully!");
    Ok(())
}

fn basic_memory_efficient_stft() -> Result<(), Box<dyn std::error::Error>> {
    // Create a test signal
    let fs = 1000.0;
    let duration = 5.0;
    let n = (fs * duration) as usize;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Multi-tone signal
    let signal: Vec<f64> = t
        .iter()
        .map(|&t| {
            (2.0 * PI * 50.0 * t).sin()
                + 0.5 * (2.0 * PI * 120.0 * t).sin()
                + 0.3 * (2.0 * PI * 200.0 * t).sin()
        })
        .collect();

    // STFT parameters
    let window_length = 512;
    let hop_size = 256;
    let window = window::hann(window_length, true)?;

    // Memory-efficient configuration
    let memory_config = MemoryEfficientStftConfig {
        max_memory_mb: 20,
        chunk_size: Some(2000),
        parallel: false,
        magnitude_only: false,
    };

    let stft_config = StftConfig::default();
    let mem_stft =
        MemoryEfficientStft::new(&window, hop_size, fs, Some(stft_config), memory_config)?;

    // Process STFT
    let result = mem_stft.stft_chunked(&signal)?;

    println!("   Signal length: {} samples", signal.len());
    println!(
        "   STFT result shape: {} x {} (freq x time)",
        result.shape()[0],
        result.shape()[1]
    );
    println!(
        "   Memory estimate: {:.2} MB",
        mem_stft.memory_estimate(signal.len())
    );

    Ok(())
}

fn auto_memory_management() -> Result<(), Box<dyn std::error::Error>> {
    let fs = 8000.0;

    // Create signals of different sizes
    let small_signal: Vec<f64> = (0..2000).map(|i| (i as f64 * 0.01).sin()).collect();
    let medium_signal: Vec<f64> = (0..50000).map(|i| (i as f64 * 0.001).sin()).collect();
    let large_signal: Vec<f64> = (0..200000).map(|i| (i as f64 * 0.0001).sin()).collect();

    let window_length = 256;
    let hop_size = 128;
    let window = window::hann(window_length, true)?;

    let memory_config = MemoryEfficientStftConfig {
        max_memory_mb: 30,
        chunk_size: None, // Auto-calculate
        parallel: false,
        magnitude_only: false,
    };

    let stft_config = StftConfig::default();
    let mem_stft =
        MemoryEfficientStft::new(&window, hop_size, fs, Some(stft_config), memory_config)?;

    // Process signals with automatic memory management
    println!("   Processing different signal sizes:");

    let small_result = mem_stft.stft_auto(&small_signal)?;
    println!(
        "     Small signal ({} samples) -> {} x {} STFT",
        small_signal.len(),
        small_result.shape()[0],
        small_result.shape()[1]
    );

    let medium_result = mem_stft.stft_auto(&medium_signal)?;
    println!(
        "     Medium signal ({} samples) -> {} x {} STFT",
        medium_signal.len(),
        medium_result.shape()[0],
        medium_result.shape()[1]
    );

    let large_result = mem_stft.stft_auto(&large_signal)?;
    println!(
        "     Large signal ({} samples) -> {} x {} STFT",
        large_signal.len(),
        large_result.shape()[0],
        large_result.shape()[1]
    );

    Ok(())
}

fn large_signal_processing() -> Result<(), Box<dyn std::error::Error>> {
    // Create a very large signal (simulating real-world scenario)
    let fs = 16000.0;
    let duration = 30.0; // 30 seconds of audio
    let n = (fs * duration) as usize;

    println!(
        "   Creating large signal: {} samples ({:.1} MB)",
        n,
        n as f64 * 8.0 / 1_000_000.0
    );

    // Create a complex signal with multiple frequency components
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            let base_freq = 440.0; // A4 note
            let harmonic1 = base_freq * 2.0;
            let harmonic2 = base_freq * 3.0;

            // Amplitude modulation
            let am_freq = 5.0;
            let am = 1.0 + 0.3 * (2.0 * PI * am_freq * t).sin();

            am * ((2.0 * PI * base_freq * t).sin()
                + 0.5 * (2.0 * PI * harmonic1 * t).sin()
                + 0.3 * (2.0 * PI * harmonic2 * t).sin())
        })
        .collect();

    let window_length = 2048;
    let hop_size = 1024;
    let window = window::hann(window_length, true)?;

    let memory_config = MemoryEfficientStftConfig {
        max_memory_mb: 100, // Reasonable memory limit
        chunk_size: None,   // Auto-calculate optimal chunk size
        parallel: false,
        magnitude_only: true, // Save memory by storing only magnitudes
    };

    let stft_config = StftConfig::default();
    let mem_stft =
        MemoryEfficientStft::new(&window, hop_size, fs, Some(stft_config), memory_config)?;

    // Get memory information
    let info = mem_stft.memory_info(signal.len());
    println!("   Memory info:");
    println!("     Chunk size: {} samples", info.chunk_size);
    println!("     Number of chunks: {}", info.num_chunks);
    println!("     Total frames: {}", info.total_frames);
    println!(
        "     Memory without chunking: {:.2} MB",
        info.total_memory_mb
    );
    println!("     Memory per chunk: {:.2} MB", info.chunk_memory_mb);
    println!(
        "     Memory reduction factor: {:.2}x",
        info.memory_reduction_factor
    );

    // Process the large signal
    let start_time = std::time::Instant::now();
    let spec_result = mem_stft.spectrogram_auto(&signal)?;
    let processing_time = start_time.elapsed();

    println!("   Processing results:");
    println!(
        "     Spectrogram shape: {} x {} (freq x time)",
        spec_result.shape()[0],
        spec_result.shape()[1]
    );
    println!(
        "     Processing time: {:.2} seconds",
        processing_time.as_secs_f64()
    );
    println!(
        "     Processing rate: {:.2} x real-time",
        duration / processing_time.as_secs_f64()
    );

    Ok(())
}

fn memory_usage_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let fs = 44100.0; // CD quality sample rate
    let window_length = 1024;
    let hop_size = 512;
    let window = window::hann(window_length, true)?;

    // Test different memory limits
    let memory_limits = vec![10, 50, 100, 200, 500];
    let signal_length = 1_000_000; // 1M samples

    println!("   Memory usage analysis for {} samples:", signal_length);
    println!("   Memory Limit | Chunk Size | Num Chunks | Reduction Factor");
    println!("   -------------|------------|------------|------------------");

    for &memory_mb in &memory_limits {
        let memory_config = MemoryEfficientStftConfig {
            max_memory_mb: memory_mb,
            chunk_size: None,
            parallel: false,
            magnitude_only: false,
        };

        let stft_config = StftConfig::default();
        let mem_stft =
            MemoryEfficientStft::new(&window, hop_size, fs, Some(stft_config), memory_config)?;

        let info = mem_stft.memory_info(signal_length);

        println!(
            "   {:>11} MB | {:>10} | {:>10} | {:>15.2}x",
            memory_mb, info.chunk_size, info.num_chunks, info.memory_reduction_factor
        );
    }

    Ok(())
}

fn magnitude_only_processing() -> Result<(), Box<dyn std::error::Error>> {
    let fs = 8000.0;
    let duration = 10.0;
    let n = (fs * duration) as usize;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Chirp signal (frequency sweep)
    let signal: Vec<f64> = t
        .iter()
        .map(|&t| {
            let freq = 100.0 + 300.0 * t / duration; // 100 to 400 Hz sweep
            (2.0 * PI * freq * t).sin()
        })
        .collect();

    let window_length = 512;
    let hop_size = 256;
    let window = window::hann(window_length, true)?;

    // Compare complex vs magnitude-only processing
    let configs = vec![
        (
            "Complex values",
            MemoryEfficientStftConfig {
                max_memory_mb: 50,
                chunk_size: Some(4000),
                parallel: false,
                magnitude_only: false,
            },
        ),
        (
            "Magnitude only",
            MemoryEfficientStftConfig {
                max_memory_mb: 50,
                chunk_size: Some(4000),
                parallel: false,
                magnitude_only: true,
            },
        ),
    ];

    for (name, config) in configs {
        let stft_config = StftConfig::default();
        let mem_stft = MemoryEfficientStft::new(&window, hop_size, fs, Some(stft_config), config)?;

        let info = mem_stft.memory_info(signal.len());
        let memory_est = mem_stft.memory_estimate(signal.len());

        println!("   {} processing:", name);
        println!("     Memory per chunk: {:.2} MB", info.chunk_memory_mb);
        println!("     Memory estimate: {:.2} MB", memory_est);

        if name.contains("Complex") {
            let _complex_result = mem_stft.stft_chunked(&signal)?;
            println!("     Output: Complex64 values");
        } else {
            let _mag_result = mem_stft.spectrogram_chunked(&signal)?;
            println!("     Output: f64 magnitudes");
        }
    }

    Ok(())
}

#[cfg(feature = "parallel")]
fn parallel_processing() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    let fs = 16000.0;
    let duration = 20.0;
    let n = (fs * duration) as usize;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Multi-component signal
    let signal: Vec<f64> = t
        .iter()
        .map(|&t| {
            (2.0 * PI * 60.0 * t).sin()
                + 0.7 * (2.0 * PI * 150.0 * t).sin()
                + 0.5 * (2.0 * PI * 300.0 * t).sin()
                + 0.1 * (2.0 * PI * 1000.0 * t * t).sin() // Chirp component
        })
        .collect();

    let window_length = 1024;
    let hop_size = 512;
    let window = window::hann(window_length, true)?;

    let memory_config = MemoryEfficientStftConfig {
        max_memory_mb: 80,
        chunk_size: Some(8000),
        parallel: true,
        magnitude_only: false,
    };

    let stft_config = StftConfig::default();
    let mem_stft =
        MemoryEfficientStft::new(&window, hop_size, fs, Some(stft_config), memory_config)?;

    println!("   Comparing sequential vs parallel processing:");

    // Sequential processing
    let start = Instant::now();
    let sequential_result = mem_stft.stft_chunked(&signal)?;
    let sequential_time = start.elapsed();

    // Parallel processing
    let start = Instant::now();
    let parallel_result = mem_stft.stft_parallel_chunked(&signal)?;
    let parallel_time = start.elapsed();

    println!(
        "     Sequential time: {:.3} seconds",
        sequential_time.as_secs_f64()
    );
    println!(
        "     Parallel time: {:.3} seconds",
        parallel_time.as_secs_f64()
    );
    println!(
        "     Speedup: {:.2}x",
        sequential_time.as_secs_f64() / parallel_time.as_secs_f64()
    );

    // Verify results are equivalent
    assert_eq!(sequential_result.shape(), parallel_result.shape());
    let max_diff = sequential_result
        .iter()
        .zip(parallel_result.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);

    println!("     Maximum difference: {:.2e}", max_diff);
    println!(
        "     Results are numerically equivalent: {}",
        max_diff < 1e-10
    );

    Ok(())
}
