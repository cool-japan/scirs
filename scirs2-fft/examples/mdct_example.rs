//! Example demonstrating MDCT and MDST usage for audio processing

use scirs2_core::ndarray::{array, Array1};
use scirs2_fft::mdct::{imdct, imdst, mdct, mdct_overlap_add, mdst};
use scirs2_fft::window::Window;
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MDCT/MDST Example ===");
    println!();

    // Example 1: Basic MDCT transform
    println!("1. Basic MDCT Transform:");
    let blocksize = 8;
    let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    println!("   Input signal: {signal:?}");

    let mdct_coeffs = mdct(&signal, blocksize, Some(Window::Hann))?;
    println!("   MDCT coefficients: {mdct_coeffs:?}");
    println!(
        "   Number of coefficients: {} (half of block size)",
        mdct_coeffs.len()
    );

    let reconstructed = imdct(&mdct_coeffs, Some(Window::Hann))?;
    println!("   Reconstructed signal: {reconstructed:?}");
    println!();

    // Example 2: Perfect reconstruction with overlap-add
    println!("2. Perfect Reconstruction with Overlap-Add:");

    // Create a longer signal
    let signal_len = 32;
    let mut long_signal = Array1::zeros(signal_len);
    for i in 0..signal_len {
        long_signal[i] = (2.0 * PI * 440.0 * i as f64 / 8000.0).sin();
    }

    // Process with overlapping blocks
    let blocksize = 16;
    let hopsize = blocksize / 2;
    let mut blocks = Vec::new();

    println!("   Signal length: {signal_len}");
    println!("   Block size: {blocksize}");
    println!("   Hop size: {hopsize}");

    // Extract and transform overlapping blocks
    let num_blocks = (signal_len - blocksize) / hopsize + 1;
    for i in 0..num_blocks {
        let start = i * hopsize;
        let end = start + blocksize;

        if end <= signal_len {
            let block = long_signal
                .slice(scirs2_core::ndarray::s![start..end])
                .to_owned();
            let mdct_block = mdct(&block, blocksize, Some(Window::Hann))?;
            blocks.push(mdct_block);
        }
    }

    println!("   Number of MDCT blocks: {}", blocks.len());

    // Reconstruct with overlap-add
    let reconstructed = mdct_overlap_add(&blocks, Some(Window::Hann), hopsize)?;

    // Calculate reconstruction error (excluding boundaries)
    let start_idx = blocksize / 2;
    let end_idx = signal_len - blocksize / 2;
    let mut error = 0.0;
    for i in start_idx..end_idx {
        error += (long_signal[i] - reconstructed[i]).abs();
    }

    println!("   Reconstruction error (excluding boundaries): {error:.6e}");
    println!();

    // Example 3: MDST transform
    println!("3. MDST Transform:");
    let test_signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let mdst_coeffs = mdst(&test_signal, 8, Some(Window::Hann))?;
    println!("   MDST coefficients: {mdst_coeffs:?}");

    let mdst_reconstructed = imdst(&mdst_coeffs, Some(Window::Hann))?;
    println!("   MDST reconstructed: {mdst_reconstructed:?}");
    println!();

    // Example 4: Comparison of MDCT vs MDST
    println!("4. MDCT vs MDST Comparison:");

    // Create a test signal with both cosine and sine components
    let mut test_signal = Array1::zeros(16);
    for i in 0..16 {
        test_signal[i] = (2.0 * PI * i as f64 / 16.0).cos() + (4.0 * PI * i as f64 / 16.0).sin();
    }

    let mdct_result = mdct(&test_signal, 16, None)?;
    let mdst_result = mdst(&test_signal, 16, None)?;

    println!("   Test signal (cos + sin components)");
    println!("   MDCT coefficients: {mdct_result:?}");
    println!("   MDST coefficients: {mdst_result:?}");

    // Energy comparison
    let mdct_energy: f64 = mdct_result.mapv(|x| x * x).sum();
    let mdst_energy: f64 = mdst_result.mapv(|x| x * x).sum();

    println!("   MDCT energy: {mdct_energy:.4}");
    println!("   MDST energy: {mdst_energy:.4}");
    println!();

    // Example 5: Audio coding simulation
    println!("5. Audio Coding Simulation:");

    // Create a simple audio signal
    let sample_rate = 44100.0;
    let duration = 0.1; // 100ms
    let freq = 440.0; // A4 note
    let signal_len = (sample_rate * duration) as usize;
    let mut audio_signal = Array1::zeros(signal_len);

    for i in 0..signal_len {
        let t = i as f64 / sample_rate;
        audio_signal[i] = (2.0 * PI * freq * t).sin() * (-10.0 * t).exp(); // Decaying sine
    }

    // Process with MDCT
    let blocksize = 2048;
    let hopsize = blocksize / 2;
    let mut mdct_blocks = Vec::new();

    let num_blocks = (signal_len - blocksize) / hopsize + 1;
    for i in 0..num_blocks {
        let start = i * hopsize;
        let end = (start + blocksize).min(signal_len);

        if end - start == blocksize {
            let block = audio_signal
                .slice(scirs2_core::ndarray::s![start..end])
                .to_owned();
            let mdct_block = mdct(&block, blocksize, Some(Window::Hann))?;

            // Simulate quantization (lossy compression)
            let quantized: Array1<f64> = mdct_block.mapv(|x| (x * 100.0).round() / 100.0);
            mdct_blocks.push(quantized);
        }
    }

    // Reconstruct
    let reconstructed_audio = mdct_overlap_add(&mdct_blocks, Some(Window::Hann), hopsize)?;

    // Calculate SNR
    let signal_power: f64 = audio_signal.mapv(|x| x * x).sum() / signal_len as f64;
    let noise = &audio_signal.slice(scirs2_core::ndarray::s![..reconstructed_audio.len()])
        - &reconstructed_audio;
    let noise_power: f64 = noise.mapv(|x| x * x).sum() / noise.len() as f64;
    let snr_db = 10.0 * (signal_power / noise_power).log10();

    println!("   Audio signal length: {signal_len} samples");
    println!("   Block size: {blocksize} samples");
    println!("   Number of MDCT blocks: {}", mdct_blocks.len());
    println!("   SNR after quantization: {snr_db:.2} dB");

    Ok(())
}
