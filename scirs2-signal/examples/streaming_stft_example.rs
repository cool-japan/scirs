//! Streaming STFT Example
//!
//! This example demonstrates real-time Short-Time Fourier Transform (STFT)
//! processing for continuous data streams, showing various configurations
//! and use cases for low-latency spectral analysis.

use ndarray::Array1;
use scirs2_signal::streaming_stft::{RealTimeStft, StreamingStft, StreamingStftConfig};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Streaming STFT Example ===\n");

    // Example 1: Basic Streaming STFT
    println!("1. Basic Streaming STFT Processing");
    println!("===================================");

    let config = StreamingStftConfig {
        frame_length: 512,
        hop_length: 256,
        window: "hann".to_string(),
        center: true,
        ..Default::default()
    };

    let mut streaming_stft = StreamingStft::new(config.clone())?;

    println!("Configuration:");
    println!("- Frame length: {} samples", config.frame_length);
    println!("- Hop length: {} samples", config.hop_length);
    println!("- Window: {}", config.window);
    println!("- Centering: {}", config.center);

    // Generate test signal with multiple frequency components
    let sample_rate = 8000.0; // 8 kHz sample rate
    let frame_duration = 0.032; // 32 ms frames (256 samples at 8 kHz)
    let frame_size = (sample_rate * frame_duration) as usize;

    println!("\nTest signal parameters:");
    println!("- Sample rate: {:.0} Hz", sample_rate);
    println!("- Frame duration: {:.3} s", frame_duration);
    println!("- Frame size: {} samples", frame_size);

    // Simulate streaming processing
    let mut _total_frames_processed = 0;
    let mut frequency_bins = 0;

    for frame_num in 0..10 {
        // Generate a frame with changing frequency content
        let base_freq = 440.0 + frame_num as f64 * 100.0; // Sweep from 440 to 1340 Hz
        let noise_level = 0.1;

        let frame_data: Vec<f64> = (0..frame_size)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let signal = (2.0 * PI * base_freq * t).sin()
                    + 0.5 * (2.0 * PI * (base_freq * 2.0) * t).sin(); // Harmonic
                let noise = noise_level * (rand::random::<f64>() - 0.5);
                signal + noise
            })
            .collect();

        let input_frame = Array1::from(frame_data);

        // Process frame
        if let Some(spectrum) = streaming_stft.process_frame(&input_frame)? {
            if frequency_bins == 0 {
                frequency_bins = spectrum.len();
                println!("\nSpectrum details:");
                println!("- Frequency bins: {}", frequency_bins);
                println!(
                    "- Frequency resolution: {:.2} Hz",
                    sample_rate / config.frame_length as f64
                );
            }

            // Find peak frequency
            let mut max_magnitude = 0.0;
            let mut peak_bin = 0;
            for (i, &complex_val) in spectrum.iter().enumerate() {
                let magnitude = complex_val.norm();
                if magnitude > max_magnitude {
                    max_magnitude = magnitude;
                    peak_bin = i;
                }
            }

            let peak_frequency = peak_bin as f64 * sample_rate / config.frame_length as f64;

            println!(
                "Frame {}: Peak at {:.1} Hz (expected ~{:.1} Hz), magnitude = {:.3}",
                frame_num + 1,
                peak_frequency,
                base_freq,
                max_magnitude
            );

            _total_frames_processed += 1;
        }
    }

    // Get processing statistics
    let stats = streaming_stft.get_statistics();
    println!("\nProcessing Statistics:");
    println!("- Samples processed: {}", stats.samples_processed);
    println!("- Frames generated: {}", stats.frames_generated);
    println!("- Current buffer size: {}", stats.buffer_size);
    println!(
        "- Latency: {} samples ({:.3} s at {:.0} Hz)",
        stats.latency_samples,
        stats.latency_samples as f64 / sample_rate,
        sample_rate
    );

    // Example 2: Real-Time STFT with Block Processing
    println!("\n2. Real-Time STFT Block Processing");
    println!("===================================");

    let rt_config = StreamingStftConfig {
        frame_length: 256,
        hop_length: 128,
        window: "hamming".to_string(),
        center: false,
        magnitude_only: true,
        log_magnitude: true,
        power: 2.0, // Power spectrum
        ..Default::default()
    };

    let block_size = 64; // Small blocks for real-time processing
    let max_buffer_size = 20; // Keep last 20 spectra

    let mut rt_stft = RealTimeStft::new(rt_config.clone(), block_size, max_buffer_size)?;

    println!("Real-time configuration:");
    println!("- Block size: {} samples", block_size);
    println!("- Max buffer size: {} spectra", max_buffer_size);
    println!("- Processing: magnitude only with log scaling");

    // Simulate real-time audio blocks
    let mut blocks_processed = 0;
    let mut total_spectra_generated = 0;

    for block_num in 0..20 {
        // Generate test block with chirp signal
        let start_freq = 200.0;
        let end_freq = 2000.0;
        let block_time = block_num as f64 * block_size as f64 / sample_rate;
        let sweep_duration = 20.0 * block_size as f64 / sample_rate; // Total sweep time

        let block_data: Vec<f64> = (0..block_size)
            .map(|i| {
                let t = block_time + i as f64 / sample_rate;
                let freq = start_freq + (end_freq - start_freq) * t / sweep_duration;
                (2.0 * PI * freq * t).sin()
            })
            .collect();

        let input_block = Array1::from(block_data);

        // Process block
        let new_spectra = rt_stft.process_block(&input_block)?;
        total_spectra_generated += new_spectra;
        blocks_processed += 1;

        if new_spectra > 0 {
            println!(
                "Block {}: Generated {} new spectra",
                block_num + 1,
                new_spectra
            );

            // Check latest spectrum
            if let Some(latest_spectrum) = rt_stft.peek_latest_spectrum() {
                // Find peak in log magnitude spectrum
                let mut max_log_magnitude = f64::NEG_INFINITY;
                let mut peak_bin = 0;
                for (i, &magnitude) in latest_spectrum.iter().enumerate() {
                    let log_mag = magnitude.re; // Real part contains log magnitude
                    if log_mag > max_log_magnitude {
                        max_log_magnitude = log_mag;
                        peak_bin = i;
                    }
                }

                let peak_frequency = peak_bin as f64 * sample_rate / rt_config.frame_length as f64;
                println!(
                    "  Peak frequency: {:.1} Hz, Log magnitude: {:.2}",
                    peak_frequency, max_log_magnitude
                );
            }
        }
    }

    // Get all available spectra
    let all_spectra = rt_stft.get_all_spectra();
    println!("\nReal-time processing summary:");
    println!("- Blocks processed: {}", blocks_processed);
    println!("- Total spectra generated: {}", total_spectra_generated);
    println!("- Final buffer contents: {} spectra", all_spectra.len());

    let rt_stats = rt_stft.get_statistics();
    println!(
        "- Output buffer usage: {}/{}",
        rt_stats.output_buffer_size, rt_stats.output_buffer_capacity
    );

    // Example 3: Low-Latency Configuration
    println!("\n3. Low-Latency Configuration");
    println!("=============================");

    let low_latency_config = StreamingStftConfig {
        frame_length: 128,
        hop_length: 64,
        window: "blackman".to_string(),
        center: false, // No centering to reduce latency
        magnitude_only: true,
        ..Default::default()
    };

    let mut low_latency_stft = StreamingStft::new(low_latency_config.clone())?;

    println!("Low-latency configuration:");
    println!(
        "- Frame length: {} samples",
        low_latency_config.frame_length
    );
    println!("- Hop length: {} samples", low_latency_config.hop_length);
    println!(
        "- Centering: {} (reduces latency)",
        low_latency_config.center
    );

    let latency_samples = low_latency_stft.get_latency_samples();
    let latency_ms = low_latency_stft.get_latency_seconds(sample_rate) * 1000.0;

    println!(
        "- Total latency: {} samples ({:.2} ms at {:.0} Hz)",
        latency_samples, latency_ms, sample_rate
    );

    // Process a few frames to demonstrate
    for i in 0..3 {
        let test_tone = 1000.0 + i as f64 * 500.0; // 1000, 1500, 2000 Hz
        let frame_data: Vec<f64> = (0..64)
            .map(|j| (2.0 * PI * test_tone * j as f64 / sample_rate).sin())
            .collect();

        let input_frame = Array1::from(frame_data);

        if let Some(magnitude_spectrum) = low_latency_stft.process_magnitude_frame(&input_frame)? {
            let mut peak_bin = 0;
            let mut peak_magnitude = 0.0;
            for (bin, &magnitude) in magnitude_spectrum.iter().enumerate() {
                if magnitude > peak_magnitude {
                    peak_magnitude = magnitude;
                    peak_bin = bin;
                }
            }

            let detected_freq =
                peak_bin as f64 * sample_rate / low_latency_config.frame_length as f64;
            println!(
                "Frame {}: Input {:.0} Hz → Detected {:.1} Hz (magnitude: {:.3})",
                i + 1,
                test_tone,
                detected_freq,
                peak_magnitude
            );
        }
    }

    // Example 4: Batch Processing
    println!("\n4. Batch Processing Demonstration");
    println!("==================================");

    let batch_config = StreamingStftConfig {
        frame_length: 256,
        hop_length: 128,
        window: "hann".to_string(),
        center: true,
        ..Default::default()
    };

    let mut batch_stft = StreamingStft::new(batch_config.clone())?;

    // Generate longer signal for batch processing
    let signal_duration = 1.0; // 1 second
    let n_samples = (sample_rate * signal_duration) as usize;
    let input_signal: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            // Multi-component signal
            (2.0 * PI * 300.0 * t).sin() +   // 300 Hz
            0.7 * (2.0 * PI * 800.0 * t).sin() + // 800 Hz
            0.4 * (2.0 * PI * 1500.0 * t).sin() // 1500 Hz
        })
        .collect();

    let input_array = Array1::from(input_signal);
    let batch_frame_size = 128;

    println!("Batch processing parameters:");
    println!("- Signal duration: {:.1} s", signal_duration);
    println!("- Total samples: {}", n_samples);
    println!("- Batch frame size: {}", batch_frame_size);

    // Process in batch
    let batch_results = batch_stft.process_batch(&input_array, batch_frame_size)?;

    println!("\nBatch processing results:");
    println!("- Generated {} spectra", batch_results.len());

    // Analyze frequency content
    let frequency_resolution = sample_rate / batch_config.frame_length as f64;
    println!("- Frequency resolution: {:.2} Hz", frequency_resolution);

    // Find peaks in first spectrum
    if !batch_results.is_empty() {
        let first_spectrum = &batch_results[0];
        let mut peaks = Vec::new();

        for (bin, &complex_val) in first_spectrum.iter().enumerate() {
            let magnitude = complex_val.norm();
            let frequency = bin as f64 * frequency_resolution;

            // Simple peak detection (magnitude > threshold and local maximum)
            if magnitude > 0.1 && bin > 0 && bin < first_spectrum.len() - 1 {
                let prev_mag = first_spectrum[bin - 1].norm();
                let next_mag = first_spectrum[bin + 1].norm();
                if magnitude > prev_mag && magnitude > next_mag {
                    peaks.push((frequency, magnitude));
                }
            }
        }

        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Sort by magnitude

        println!("\nDetected peaks in first spectrum:");
        for (i, (freq, mag)) in peaks.iter().take(5).enumerate() {
            println!("  Peak {}: {:.1} Hz (magnitude: {:.3})", i + 1, freq, mag);
        }
    }

    // Example 5: Performance Comparison
    println!("\n5. Performance Comparison");
    println!("=========================");

    let configs = vec![
        (
            "Ultra Low Latency",
            StreamingStftConfig {
                frame_length: 64,
                hop_length: 32,
                center: false,
                ..Default::default()
            },
        ),
        (
            "Low Latency",
            StreamingStftConfig {
                frame_length: 128,
                hop_length: 64,
                center: false,
                ..Default::default()
            },
        ),
        (
            "Balanced",
            StreamingStftConfig {
                frame_length: 256,
                hop_length: 128,
                center: true,
                ..Default::default()
            },
        ),
        (
            "High Resolution",
            StreamingStftConfig {
                frame_length: 512,
                hop_length: 256,
                center: true,
                ..Default::default()
            },
        ),
    ];

    println!("Configuration comparison:");
    println!(
        "{:<20} {:<12} {:<15} {:<15}",
        "Name", "Latency (ms)", "Freq Res (Hz)", "Time Res (ms)"
    );
    println!("{}", "-".repeat(65));

    for (name, config) in configs {
        let stft = StreamingStft::new(config.clone())?;
        let latency_ms = stft.get_latency_seconds(sample_rate) * 1000.0;
        let freq_resolution = sample_rate / config.frame_length as f64;
        let time_resolution = config.hop_length as f64 / sample_rate * 1000.0;

        println!(
            "{:<20} {:<12.2} {:<15.2} {:<15.2}",
            name, latency_ms, freq_resolution, time_resolution
        );
    }

    println!("\n=== Streaming STFT Example Complete ===");

    Ok(())
}
