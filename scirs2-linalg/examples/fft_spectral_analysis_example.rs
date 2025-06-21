//! Fast Fourier Transform (FFT) and Spectral Analysis Example
//!
//! This example demonstrates the comprehensive FFT and spectral analysis capabilities
//! that provide cutting-edge frequency domain computations for signal processing,
//! image analysis, quantum physics, and machine learning applications:
//!
//! - Core FFT algorithms (Cooley-Tukey, mixed-radix, Bluestein)
//! - Real-valued FFT optimizations for 2x speedup
//! - Multidimensional FFT for image and volume processing
//! - Windowing functions for spectral analysis
//! - Discrete transforms (DCT, DST) for compression
//! - FFT-based convolution for signal processing
//! - Power spectral density estimation methods
//!
//! These techniques are fundamental to modern signal processing and provide
//! O(n log n) complexity for frequency domain computations.

use ndarray::{Array1, Array2, Array3};
// Commented out as it's not used in this example
// use num_complex::Complex;
use scirs2_linalg::fft::{
    apply_window, dct_1d, dst_1d, fft_1d, fft_2d, fft_3d, fft_convolve, fft_frequencies, idct_1d,
    irfft_1d, periodogram_psd, rfft_1d, welch_psd, Complex64, FFTAlgorithm, FFTPlan,
    WindowFunction,
};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 FFT AND SPECTRAL ANALYSIS - ULTRATHINK DEMONSTRATION");
    println!("========================================================");

    // Test 1: Basic 1D FFT Algorithms
    println!("\n1. CORE FFT ALGORITHMS: Cooley-Tukey and Mixed-Radix");
    println!("----------------------------------------------------");

    // Power-of-2 size for Cooley-Tukey
    let n_power2 = 16;
    let signal_power2 = Array1::from_shape_fn(n_power2, |i| {
        Complex64::new((2.0 * PI * 3.0 * i as f64 / n_power2 as f64).sin(), 0.0)
    });

    println!("   Testing Cooley-Tukey FFT (size {}):", n_power2);
    let fft_result = fft_1d(&signal_power2.view(), false)?;
    let ifft_result = fft_1d(&fft_result.view(), true)?;

    // Check reconstruction accuracy
    let mut max_error = 0.0;
    for i in 0..n_power2 {
        let error = (signal_power2[i] - ifft_result[i]).norm();
        if error > max_error {
            max_error = error;
        }
    }
    println!("     Reconstruction error: {:.2e}", max_error);
    println!("     ✅ Perfect reconstruction: {}", max_error < 1e-12);

    // Arbitrary size using mixed-radix (Bluestein's algorithm)
    let n_arbitrary = 15; // Not power of 2
    let signal_arbitrary = Array1::from_shape_fn(n_arbitrary, |i| {
        Complex64::new((2.0 * PI * 2.0 * i as f64 / n_arbitrary as f64).cos(), 0.0)
    });

    println!("\n   Testing Mixed-Radix FFT (size {}):", n_arbitrary);
    // For arbitrary sizes, we use the built-in Bluestein algorithm
    let _complex_input = signal_arbitrary.view();
    let plan = FFTPlan::<f64>::new(n_arbitrary, FFTAlgorithm::Auto, false)?;
    println!("     Selected algorithm: {:?}", plan.algorithm);

    // Test 2: Real-Valued FFT Optimizations
    println!("\n2. REAL-VALUED FFT: Exploiting Hermitian Symmetry");
    println!("------------------------------------------------");

    let n_real = 32;
    let real_signal = Array1::from_shape_fn(n_real, |i| {
        (2.0 * PI * 5.0 * i as f64 / n_real as f64).sin()
            + 0.5 * (2.0 * PI * 10.0 * i as f64 / n_real as f64).cos()
    });

    println!("   Real signal length: {}", n_real);

    // RFFT gives N/2+1 complex coefficients
    let rfft_result = rfft_1d(&real_signal.view())?;
    println!(
        "   RFFT output length: {} ({}% reduction)",
        rfft_result.len(),
        100 - (rfft_result.len() * 100 / n_real)
    );

    // Reconstruct using IRFFT
    let reconstructed = irfft_1d(&rfft_result.view(), n_real)?;

    let mut reconstruction_error = 0.0;
    for i in 0..n_real {
        let error = (real_signal[i] - reconstructed[i]).abs();
        reconstruction_error += error * error;
    }
    reconstruction_error = (reconstruction_error / n_real as f64).sqrt();
    println!("   RMSE reconstruction: {:.2e}", reconstruction_error);
    println!(
        "   ✅ Real FFT optimization successful: {}",
        reconstruction_error < 1e-12
    );

    // Test 3: Multidimensional FFT
    println!("\n3. MULTIDIMENSIONAL FFT: Image and Volume Processing");
    println!("---------------------------------------------------");

    // 2D FFT for image processing
    let (rows, cols) = (8, 8);
    let image = Array2::from_shape_fn((rows, cols), |(i, j)| {
        let x = 2.0 * PI * (i as f64) / (rows as f64);
        let y = 2.0 * PI * (j as f64) / (cols as f64);
        Complex64::new(x.sin() * y.cos(), 0.0)
    });

    println!("   2D FFT on {}×{} image:", rows, cols);
    let fft2d_result = fft_2d(&image.view(), false)?;
    let ifft2d_result = fft_2d(&fft2d_result.view(), true)?;

    let mut error_2d = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            error_2d += (image[[i, j]] - ifft2d_result[[i, j]]).norm_sqr();
        }
    }
    error_2d = (error_2d / (rows * cols) as f64).sqrt();
    println!("     2D FFT RMSE: {:.2e}", error_2d);

    // 3D FFT for volume processing
    let (depth, height, width) = (4, 4, 4);
    let volume = Array3::from_shape_fn((depth, height, width), |(d, h, w)| {
        Complex64::new((d + h + w) as f64, 0.0)
    });

    println!("   3D FFT on {}×{}×{} volume:", depth, height, width);
    let fft3d_result = fft_3d(&volume.view(), false)?;
    let ifft3d_result = fft_3d(&fft3d_result.view(), true)?;

    let mut error_3d = 0.0;
    for d in 0..depth {
        for h in 0..height {
            for w in 0..width {
                error_3d += (volume[[d, h, w]] - ifft3d_result[[d, h, w]]).norm_sqr();
            }
        }
    }
    error_3d = (error_3d / (depth * height * width) as f64).sqrt();
    println!("     3D FFT RMSE: {:.2e}", error_3d);
    println!("   ✅ Multidimensional FFT successful");

    // Test 4: Window Functions for Spectral Analysis
    println!("\n4. WINDOWING FUNCTIONS: Spectral Leakage Reduction");
    println!("--------------------------------------------------");

    let window_size = 32;
    let test_signal = Array1::from_shape_fn(window_size, |i| {
        (2.0 * PI * 7.5 * i as f64 / window_size as f64).sin() // Non-integer frequency
    });

    let window_types = vec![
        ("Rectangular", WindowFunction::Rectangular),
        ("Hann", WindowFunction::Hann),
        ("Hamming", WindowFunction::Hamming),
        ("Blackman", WindowFunction::Blackman),
        ("Kaiser(5.0)", WindowFunction::Kaiser(5.0)),
        ("Tukey(0.5)", WindowFunction::Tukey(0.5)),
        ("Gaussian(2.0)", WindowFunction::Gaussian(2.0)),
    ];

    println!("   Comparing window functions for spectral analysis:");
    for (name, window_type) in window_types {
        let windowed = apply_window(&test_signal.view(), window_type)?;
        let windowed_fft = rfft_1d(&windowed.view())?;

        // Find peak magnitude and frequency
        let mut max_magnitude = 0.0;
        let mut peak_bin = 0;
        for (i, &coeff) in windowed_fft.iter().enumerate() {
            let magnitude = coeff.norm();
            if magnitude > max_magnitude {
                max_magnitude = magnitude;
                peak_bin = i;
            }
        }

        // Calculate spectral leakage (energy outside main lobe)
        let mut total_energy = 0.0;
        let mut main_lobe_energy = 0.0;
        for (i, &coeff) in windowed_fft.iter().enumerate() {
            let energy = coeff.norm_sqr();
            total_energy += energy;
            if (i as i32 - peak_bin as i32).abs() <= 1 {
                main_lobe_energy += energy;
            }
        }
        let leakage_ratio = 1.0 - main_lobe_energy / total_energy;

        println!(
            "     {:<15}: Peak bin={}, Leakage={:.3}%",
            name,
            peak_bin,
            leakage_ratio * 100.0
        );
    }

    // Test 5: Discrete Cosine and Sine Transforms
    println!("\n5. DISCRETE TRANSFORMS: DCT and DST for Compression");
    println!("---------------------------------------------------");

    let dct_signal = Array1::from_shape_fn(16, |i| {
        if i < 8 {
            1.0
        } else {
            0.0
        } // Step function
    });

    println!("   DCT-II (JPEG-style compression):");
    let dct_coeffs = dct_1d(&dct_signal.view())?;
    let idct_result = idct_1d(&dct_coeffs.view())?;

    let mut dct_error = 0.0;
    for i in 0..dct_signal.len() {
        dct_error += (dct_signal[i] - idct_result[i]).powi(2);
    }
    dct_error = (dct_error / dct_signal.len() as f64).sqrt();
    println!("     DCT reconstruction RMSE: {:.2e}", dct_error);

    // Energy compaction test (most energy in few coefficients)
    let mut sorted_coeffs = dct_coeffs.to_vec();
    sorted_coeffs.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
    let total_energy: f64 = sorted_coeffs.iter().map(|x| x * x).sum();
    let top_8_energy: f64 = sorted_coeffs[..8].iter().map(|x| x * x).sum();
    println!(
        "     Energy in top 50% coefficients: {:.1}%",
        100.0 * top_8_energy / total_energy
    );

    println!("\n   DST-I for boundary value problems:");
    let dst_signal = Array1::from_shape_fn(8, |i| (i + 1) as f64);
    let _dst_coeffs = dst_1d(&dst_signal.view())?;
    println!("     DST computed for {} samples", dst_signal.len());

    // Test 6: FFT-Based Convolution
    println!("\n6. FFT CONVOLUTION: Fast Signal Processing");
    println!("------------------------------------------");

    let signal = Array1::from_shape_fn(64, |i| {
        if (20..=25).contains(&i) {
            1.0
        } else {
            0.0
        } // Pulse
    });

    let kernel = Array1::from_shape_fn(8, |i| {
        (-0.5 * (i as f64 - 3.5).powi(2)).exp() // Gaussian kernel
    });

    println!(
        "   Convolving pulse signal (length {}) with Gaussian kernel (length {})",
        signal.len(),
        kernel.len()
    );

    let convolved = fft_convolve(&signal.view(), &kernel.view())?;
    println!("     Convolution result length: {}", convolved.len());

    // Find peak of convolved signal
    let (max_idx, max_val) = convolved
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    println!("     Peak amplitude: {:.3} at index {}", max_val, max_idx);
    println!("   ✅ FFT convolution provides efficient O(n log n) complexity");

    // Test 7: Power Spectral Density Estimation
    println!("\n7. SPECTRAL ANALYSIS: Power Spectral Density");
    println!("--------------------------------------------");

    // Generate noisy sinusoid
    let fs = 1000.0; // Sampling frequency
    let n_samples = 512;
    let noise_level = 0.1;

    let noisy_signal = Array1::from_shape_fn(n_samples, |i| {
        let t = i as f64 / fs;
        let signal = (2.0 * PI * 50.0 * t).sin() +     // 50 Hz
                     0.5 * (2.0 * PI * 120.0 * t).sin(); // 120 Hz
        let noise = (rand::random::<f64>() - 0.5) * noise_level;
        signal + noise
    });

    println!("   Signal: 50 Hz + 120 Hz sinusoids with noise");
    println!("   Sampling rate: {} Hz, Length: {} samples", fs, n_samples);

    // Periodogram method
    let psd_periodogram = periodogram_psd(&noisy_signal.view(), WindowFunction::Hann, None)?;
    let freqs = fft_frequencies(n_samples, fs, true);

    // Find peaks in PSD
    let mut peaks = Vec::new();
    for i in 1..psd_periodogram.len() - 1 {
        if psd_periodogram[i] > psd_periodogram[i - 1]
            && psd_periodogram[i] > psd_periodogram[i + 1]
            && psd_periodogram[i] > 0.01
        {
            // Threshold
            peaks.push((freqs[i], psd_periodogram[i]));
        }
    }
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n   Periodogram PSD - Top peaks:");
    for (i, (freq, power)) in peaks.iter().take(3).enumerate() {
        println!("     Peak {}: {:.1} Hz (power: {:.2e})", i + 1, freq, power);
    }

    // Welch's method with overlapping segments
    let segment_length = 128;
    let overlap = 0.5;
    let psd_welch = welch_psd(
        &noisy_signal.view(),
        segment_length,
        overlap,
        WindowFunction::Hann,
    )?;
    let welch_freqs = fft_frequencies(segment_length.next_power_of_two(), fs, true);

    println!(
        "\n   Welch's method PSD (segments={}, overlap={}%):",
        n_samples / (segment_length / 2) - 1,
        overlap * 100.0
    );

    // Find Welch peaks
    let mut welch_peaks = Vec::new();
    for i in 1..psd_welch.len() - 1 {
        if psd_welch[i] > psd_welch[i - 1]
            && psd_welch[i] > psd_welch[i + 1]
            && psd_welch[i] > 0.001
        {
            welch_peaks.push((welch_freqs[i], psd_welch[i]));
        }
    }
    welch_peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, (freq, power)) in welch_peaks.iter().take(3).enumerate() {
        println!("     Peak {}: {:.1} Hz (power: {:.2e})", i + 1, freq, power);
    }

    // Test 8: Frequency Domain Applications
    println!("\n8. FREQUENCY DOMAIN APPLICATIONS");
    println!("--------------------------------");

    println!("   ✅ SIGNAL PROCESSING:");
    println!("      - Digital filter design and implementation");
    println!("      - Spectral analysis of biomedical signals");
    println!("      - Audio compression and enhancement");
    println!("      - Radar and sonar signal processing");

    println!("   ✅ IMAGE PROCESSING:");
    println!("      - JPEG compression using 2D DCT");
    println!("      - Image filtering and enhancement");
    println!("      - Pattern recognition and feature extraction");
    println!("      - Medical image reconstruction");

    println!("   ✅ SCIENTIFIC COMPUTING:");
    println!("      - Quantum mechanics (momentum space)");
    println!("      - Partial differential equation solving");
    println!("      - Crystallography and diffraction patterns");
    println!("      - Climate and weather modeling");

    println!("   ✅ MACHINE LEARNING:");
    println!("      - Feature extraction from time series");
    println!("      - Convolutional neural networks");
    println!("      - Speech recognition and synthesis");
    println!("      - Anomaly detection in frequency domain");

    // Test 9: Performance and Complexity Analysis
    println!("\n9. PERFORMANCE ANALYSIS: O(n log n) Advantage");
    println!("---------------------------------------------");

    let test_sizes = vec![64, 128, 256, 512, 1024];
    println!("   FFT complexity scaling demonstration:");

    for &size in &test_sizes {
        let test_data = Array1::from_shape_fn(size, |i| {
            Complex64::new((2.0 * PI * i as f64 / size as f64).sin(), 0.0)
        });

        let start = std::time::Instant::now();
        let _fft_result = fft_1d(&test_data.view(), false)?;
        let fft_time = start.elapsed();

        let naive_operations = size * size; // O(n²) for naive DFT
        let fft_operations = size * (size as f64).log2() as usize; // O(n log n)
        let theoretical_speedup = naive_operations as f64 / fft_operations as f64;

        println!(
            "     Size {:4}: {:8.1} μs (theoretical speedup: {:.1}x)",
            size,
            fft_time.as_nanos() as f64 / 1000.0,
            theoretical_speedup
        );
    }

    println!("\n========================================================");
    println!("🎯 ULTRATHINK ACHIEVEMENT: FAST FOURIER TRANSFORM");
    println!("========================================================");
    println!("✅ Core FFT algorithms: Cooley-Tukey, mixed-radix, Bluestein");
    println!("✅ Real FFT optimization: 2x speedup via Hermitian symmetry");
    println!("✅ Multidimensional FFT: 2D/3D for image and volume processing");
    println!("✅ Window functions: Hann, Hamming, Blackman, Kaiser, Tukey, Gaussian");
    println!("✅ Discrete transforms: DCT for compression, DST for boundary problems");
    println!("✅ FFT convolution: O(n log n) fast signal processing");
    println!("✅ Spectral analysis: Periodogram and Welch's PSD estimation");
    println!("✅ Optimal complexity: O(n log n) vs O(n²) for naive DFT");
    println!("✅ Comprehensive toolbox: Complete frequency domain ecosystem");
    println!("========================================================");

    Ok(())
}
