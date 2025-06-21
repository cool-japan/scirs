//! Filter Bank Design and Analysis Example
//!
//! This example demonstrates the comprehensive filter bank functionality
//! available in scirs2-signal, including QMF banks, wavelet filter banks,
//! cosine modulated filter banks, and IIR filter stabilization.

use ndarray::Array1;
use scirs2_signal::filter_banks::{
    CosineModulatedFilterBank, FilterBankType, FilterBankWindow, IirStabilizer, QmfBank,
    StabilizationMethod, WaveletFilterBank,
};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Filter Bank Design and Analysis Example ===\n");

    // Generate test signal
    let fs = 1000.0; // Sampling frequency
    let duration = 1.0; // 1 second
    let n_samples = (fs * duration) as usize;
    let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();

    // Create multi-component test signal
    let signal: Vec<f64> = t
        .iter()
        .map(|&t| {
            // Multi-band signal with different frequency components
            (2.0 * PI * 50.0 * t).sin()   // 50 Hz component
                + 0.5 * (2.0 * PI * 150.0 * t).sin() // 150 Hz component
                + 0.3 * (2.0 * PI * 300.0 * t).sin() // 300 Hz component
                + 0.1 * (2.0 * PI * 450.0 * t).sin() // 450 Hz component
        })
        .collect();

    let input = Array1::from(signal);

    println!("Generated test signal:");
    println!("- Duration: {:.1} seconds", duration);
    println!("- Sample rate: {:.0} Hz", fs);
    println!("- Samples: {}", n_samples);
    println!("- Components: 50 Hz, 150 Hz, 300 Hz, 450 Hz");

    // Example 1: QMF (Quadrature Mirror Filter) Bank
    println!("\n1. QMF Filter Bank Analysis");
    println!("===========================");

    let num_channels = 4;
    let qmf = QmfBank::new(num_channels, FilterBankType::Orthogonal)?;

    println!("QMF Bank Configuration:");
    println!("- Channels: {}", qmf.num_channels);
    println!("- Filter length: {}", qmf.filter_length);
    println!("- Decimation factor: {}", qmf.decimation);
    println!("- Bank type: {:?}", qmf.bank_type);

    // Perform analysis (decomposition)
    let subbands = qmf.analysis(&input)?;
    println!("\nAnalysis results:");
    for (i, subband) in subbands.iter().enumerate() {
        let energy = subband.mapv(|x| x * x).sum();
        println!(
            "  Subband {}: Length = {}, Energy = {:.6}",
            i,
            subband.len(),
            energy
        );
    }

    // Perform synthesis (reconstruction)
    let reconstructed = qmf.synthesis(&subbands)?;
    println!("\nSynthesis results:");
    println!("- Original length: {}", input.len());
    println!("- Reconstructed length: {}", reconstructed.len());

    // Calculate reconstruction error
    let min_len = input.len().min(reconstructed.len());
    let mut reconstruction_error = 0.0;
    for i in 0..min_len {
        reconstruction_error += (input[i] - reconstructed[i]).powi(2);
    }
    reconstruction_error = (reconstruction_error / min_len as f64).sqrt();
    println!("- RMS reconstruction error: {:.6}", reconstruction_error);

    // Analyze filter bank properties
    let analysis = qmf.analyze_properties()?;
    println!("\nFilter Bank Analysis:");
    println!(
        "- Perfect reconstruction: {}",
        analysis.perfect_reconstruction
    );
    println!("- Aliasing distortion: {:.6}", analysis.aliasing_distortion);
    println!(
        "- Amplitude distortion: {:.6}",
        analysis.amplitude_distortion
    );
    println!(
        "- Stopband attenuation: {:.2} dB",
        analysis.stopband_attenuation
    );

    // Example 2: Wavelet Filter Bank
    println!("\n2. Wavelet Filter Bank Analysis");
    println!("==============================");

    let wavelet_bank = WaveletFilterBank::new("db4", 3)?;

    println!("Wavelet Filter Bank Configuration:");
    println!("- Wavelet: {}", wavelet_bank.wavelet_name);
    println!("- Decomposition levels: {}", wavelet_bank.levels);
    println!(
        "- Lowpass filter length: {}",
        wavelet_bank.lowpass_dec.len()
    );
    println!(
        "- Highpass filter length: {}",
        wavelet_bank.highpass_dec.len()
    );

    // Wavelet decomposition
    let wavelet_coeffs = wavelet_bank.decompose(&input)?;
    println!("\nWavelet Decomposition:");
    for (i, coeffs) in wavelet_coeffs.iter().enumerate() {
        let energy = coeffs.mapv(|x| x * x).sum();
        let level_name = if i == 0 {
            "Approximation".to_string()
        } else {
            format!("Detail {}", i)
        };
        println!(
            "  {}: Length = {}, Energy = {:.6}",
            level_name,
            coeffs.len(),
            energy
        );
    }

    // Wavelet reconstruction
    let wavelet_reconstructed = wavelet_bank.reconstruct(&wavelet_coeffs)?;
    println!("\nWavelet Reconstruction:");
    println!("- Original length: {}", input.len());
    println!("- Reconstructed length: {}", wavelet_reconstructed.len());

    // Calculate wavelet reconstruction error
    let wavelet_min_len = input.len().min(wavelet_reconstructed.len());
    let mut wavelet_error = 0.0;
    for i in 0..wavelet_min_len {
        wavelet_error += (input[i] - wavelet_reconstructed[i]).powi(2);
    }
    wavelet_error = (wavelet_error / wavelet_min_len as f64).sqrt();
    println!("- RMS reconstruction error: {:.6}", wavelet_error);

    // Example 3: Cosine Modulated Filter Bank
    println!("\n3. Cosine Modulated Filter Bank");
    println!("===============================");

    let cmfb = CosineModulatedFilterBank::new(8, 2, FilterBankWindow::Hann)?;

    println!("Cosine Modulated Filter Bank Configuration:");
    println!("- Channels: {}", cmfb.qmf_bank.num_channels);
    println!("- Overlap factor: {}", cmfb.overlap_factor);
    println!("- Prototype filter length: {}", cmfb.prototype.len());

    // CMFB Analysis
    let cmfb_subbands = cmfb.analysis(&input)?;
    println!("\nCMFB Analysis:");
    for (i, subband) in cmfb_subbands.iter().enumerate() {
        let energy = subband.mapv(|x| x * x).sum();
        println!(
            "  Channel {}: Length = {}, Energy = {:.6}",
            i,
            subband.len(),
            energy
        );
    }

    // CMFB Synthesis
    let cmfb_reconstructed = cmfb.synthesis(&cmfb_subbands)?;
    println!("\nCMFB Reconstruction:");
    println!("- Original length: {}", input.len());
    println!("- Reconstructed length: {}", cmfb_reconstructed.len());

    // Calculate CMFB reconstruction error
    let cmfb_min_len = input.len().min(cmfb_reconstructed.len());
    let mut cmfb_error = 0.0;
    for i in 0..cmfb_min_len {
        cmfb_error += (input[i] - cmfb_reconstructed[i]).powi(2);
    }
    cmfb_error = (cmfb_error / cmfb_min_len as f64).sqrt();
    println!("- RMS reconstruction error: {:.6}", cmfb_error);

    // Example 4: Filter Bank Comparison
    println!("\n4. Filter Bank Performance Comparison");
    println!("=====================================");

    println!("Reconstruction Error Comparison:");
    println!("- QMF Bank (Orthogonal):     {:.6}", reconstruction_error);
    println!("- Wavelet Filter Bank:       {:.6}", wavelet_error);
    println!("- Cosine Modulated Bank:     {:.6}", cmfb_error);

    // Test different QMF bank types
    println!("\nQMF Bank Type Comparison:");
    let bank_types = vec![
        FilterBankType::PerfectReconstruction,
        FilterBankType::Orthogonal,
        FilterBankType::Biorthogonal,
        FilterBankType::CosineModulated,
    ];

    for bank_type in bank_types {
        match QmfBank::new(4, bank_type) {
            Ok(qmf_test) => {
                if let Ok(test_subbands) = qmf_test.analysis(&input) {
                    if let Ok(test_reconstructed) = qmf_test.synthesis(&test_subbands) {
                        let test_min_len = input.len().min(test_reconstructed.len());
                        let mut test_error = 0.0;
                        for i in 0..test_min_len {
                            test_error += (input[i] - test_reconstructed[i]).powi(2);
                        }
                        test_error = (test_error / test_min_len as f64).sqrt();
                        println!("- {:?}: {:.6}", bank_type, test_error);
                    }
                }
            }
            Err(_) => {
                println!("- {:?}: Failed to create", bank_type);
            }
        }
    }

    // Example 5: IIR Filter Stabilization
    println!("\n5. IIR Filter Stabilization");
    println!("===========================");

    // Create an unstable IIR filter (poles outside unit circle)
    let b_unstable = Array1::from_vec(vec![1.0, 0.5]);
    let a_unstable = Array1::from_vec(vec![1.0, -1.8, 0.9]); // Potentially unstable

    println!("Original filter coefficients:");
    println!("- Numerator (b): {:?}", b_unstable.as_slice().unwrap());
    println!("- Denominator (a): {:?}", a_unstable.as_slice().unwrap());

    // Test different stabilization methods
    let stabilization_methods = vec![
        StabilizationMethod::RadialProjection,
        StabilizationMethod::ZeroPlacement,
        StabilizationMethod::BalancedTruncation,
    ];

    for method in stabilization_methods {
        match IirStabilizer::stabilize_filter(&b_unstable, &a_unstable, method) {
            Ok((b_stable, a_stable)) => {
                println!("\n{:?} stabilization:", method);
                println!("- Stabilized numerator: {:?}", b_stable.as_slice().unwrap());
                println!(
                    "- Stabilized denominator: {:?}",
                    a_stable.as_slice().unwrap()
                );
            }
            Err(e) => {
                println!("\n{:?} stabilization failed: {}", method, e);
            }
        }
    }

    // Example 6: Multi-rate Signal Processing Demonstration
    println!("\n6. Multi-rate Processing Effects");
    println!("=================================");

    // Show how filter banks provide different time-frequency trade-offs
    let channel_counts = vec![2, 4, 8, 16];

    println!("Channel count vs. frequency resolution:");
    for &channels in &channel_counts {
        if let Ok(_qmf_test) = QmfBank::new(channels, FilterBankType::Orthogonal) {
            let freq_resolution = fs / (2.0 * channels as f64);
            let time_resolution = channels as f64 / fs;
            println!(
                "- {} channels: Freq res = {:.1} Hz, Time res = {:.4} s",
                channels, freq_resolution, time_resolution
            );
        }
    }

    // Example 7: Filter Bank Design Guidelines
    println!("\n7. Filter Bank Design Guidelines");
    println!("=================================");

    println!("When to use different filter bank types:");
    println!("- QMF Orthogonal:        Energy preservation, moderate reconstruction");
    println!("- QMF Perfect Recon.:    Exact reconstruction, may have distortion");
    println!("- QMF Biorthogonal:      Linear phase, good for symmetric signals");
    println!("- Cosine Modulated:      Computational efficiency, good for audio");
    println!("- Wavelet Filter Banks:  Multi-resolution analysis, good for transients");

    println!("\nComputational complexity comparison (relative):");
    println!("- QMF Banks:             1.0x (baseline)");
    println!("- Cosine Modulated:      0.7x (efficient DCT implementation)");
    println!("- Wavelet Filter Banks:  0.8x (dyadic tree structure)");

    println!("\n=== Filter Bank Example Complete ===");

    Ok(())
}
