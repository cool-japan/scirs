/// Example demonstrating sparse signal recovery techniques
use ndarray::{Array1, Array2};
use rand::{seq::SliceRandom, Rng};
use scirs2_linalg::vector_norm;
use scirs2_signal::sparse::{
    compressed_sensing_recover, image_inpainting, measure_sparsity, random_sensing_matrix,
    recover_missing_samples, sparse_denoise, SparseRecoveryConfig, SparseRecoveryMethod,
    SparseTransform,
};

fn main() {
    println!("# Sparse Signal Recovery Examples");

    // Example 1: Basic Compressed Sensing
    println!("\n## 1. Basic Compressed Sensing Recovery");
    basic_compressed_sensing_example();

    // Example 2: Missing Data Recovery
    println!("\n## 2. Signal Recovery with Missing Samples");
    missing_samples_recovery_example();

    // Example 3: Sparse Signal Denoising
    println!("\n## 3. Sparse Signal Denoising");
    sparse_denoising_example();

    // Example 4: Method Comparison
    println!("\n## 4. Comparing Different Recovery Methods");
    compare_recovery_methods();

    // Example 5: Image Inpainting Demo
    println!("\n## 5. Image Inpainting Example");
    image_inpainting_example();
}

/// Basic compressed sensing recovery example
fn basic_compressed_sensing_example() {
    // Create a sparse signal
    let n = 100; // Signal dimension
    let k = 5; // Sparsity level (number of non-zero entries)

    let mut rng = rand::rng();

    // Generate random sparse signal
    let mut original_signal = Array1::zeros(n);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    indices.truncate(k);

    for &idx in &indices {
        // Random non-zero values between -10 and 10
        original_signal[idx] = 20.0 * rng.random::<f64>() - 10.0;
    }

    // Number of measurements
    let m = 30; // m < n (compressed sensing)

    // Create sensing matrix
    let phi = random_sensing_matrix(m, n, Some(42));

    // Generate measurements
    let y = phi.dot(&original_signal);

    // Add a small amount of noise
    let noise_level = 0.01;
    let noisy_y = y
        .clone()
        .mapv(|v| v + noise_level * (2.0 * rng.random::<f64>() - 1.0));

    // Sparse recovery configuration
    let config = SparseRecoveryConfig {
        sparsity: Some(k),
        max_iterations: 200,
        convergence_threshold: 1e-6,
        ..Default::default()
    };

    // Recover signal using OMP
    let recovered_signal =
        compressed_sensing_recover(&noisy_y, &phi, SparseRecoveryMethod::OMP, &config).unwrap();

    // Calculate recovery error
    let diff = &original_signal - &recovered_signal;
    let recovery_error =
        vector_norm(&diff.view(), 2).unwrap() / vector_norm(&original_signal.view(), 2).unwrap();

    println!(
        "Original signal sparsity: {:.3}",
        measure_sparsity(&original_signal, 1e-6).unwrap()
    );
    println!(
        "Recovered signal sparsity: {:.3}",
        measure_sparsity(&recovered_signal, 1e-6).unwrap()
    );
    println!("Relative recovery error: {:.6}", recovery_error);

    // Print some elements for comparison
    println!("\nOriginal vs Recovered (first 10 elements):");
    for i in 0..10 {
        println!(
            "{:2}: {:10.5} vs {:10.5}",
            i, original_signal[i], recovered_signal[i]
        );
    }
}

/// Example of recovering signals with missing samples
fn missing_samples_recovery_example() {
    // Create a sparse signal in frequency domain
    let n = 128;
    let mut rng = rand::rng();

    // Generate a signal with a few frequency components
    let mut signal = Array1::zeros(n);
    for i in 0..n {
        let t = i as f64 / n as f64;
        signal[i] = 3.0 * (2.0 * std::f64::consts::PI * 5.0 * t).sin()
            + 2.0 * (2.0 * std::f64::consts::PI * 10.0 * t).sin()
            + 1.0 * (2.0 * std::f64::consts::PI * 20.0 * t).sin();
    }

    // Create a signal with missing samples (marked as NaN)
    let missing_ratio = 0.4; // 40% of samples are missing
    let mut observed_signal = signal.clone();

    for i in 0..n {
        if rng.random::<f64>() < missing_ratio {
            observed_signal[i] = f64::NAN;
        }
    }

    // Count missing samples
    let missing_count = observed_signal.iter().filter(|&&x| x.is_nan()).count();
    println!(
        "Signal length: {}, Missing samples: {} ({}%)",
        n,
        missing_count,
        100.0 * missing_count as f64 / n as f64
    );

    // Configure recovery
    let config = SparseRecoveryConfig {
        max_iterations: 200,
        convergence_threshold: 1e-6,
        sparsity: Some(10), // Expected number of significant frequencies
        ..Default::default()
    };

    // Recover signal using OMP
    let recovered_signal =
        recover_missing_samples(&observed_signal, SparseRecoveryMethod::OMP, &config).unwrap();

    // Calculate recovery error on the missing positions only
    let mut error_sum = 0.0;
    let mut count = 0;

    for i in 0..n {
        if observed_signal[i].is_nan() {
            error_sum += (signal[i] - recovered_signal[i]).powi(2);
            count += 1;
        }
    }

    let rmse = (error_sum / count as f64).sqrt();
    println!("RMSE on missing samples: {:.6}", rmse);

    // Print a few samples for comparison
    println!("\nOriginal vs Recovered (first 10 missing samples):");
    let mut shown = 0;
    for i in 0..n {
        if observed_signal[i].is_nan() && shown < 10 {
            println!(
                "{:3}: {:10.5} vs {:10.5}",
                i, signal[i], recovered_signal[i]
            );
            shown += 1;
        }
    }
}

/// Example of sparse signal denoising
fn sparse_denoising_example() {
    // Create a signal that is sparse in the frequency domain
    let n = 128;
    let mut rng = rand::rng();

    // Generate a clean signal with a few frequency components
    let mut clean_signal = Array1::zeros(n);
    for i in 0..n {
        let t = i as f64 / n as f64;
        clean_signal[i] = 4.0 * (2.0 * std::f64::consts::PI * 3.0 * t).sin()
            + 2.0 * (2.0 * std::f64::consts::PI * 7.0 * t).sin();
    }

    // Add noise
    let noise_level = 0.5;
    let noisy_signal = clean_signal
        .clone()
        .mapv(|v| v + noise_level * (2.0 * rng.random::<f64>() - 1.0));

    // Calculate SNR of noisy signal
    let signal_power = clean_signal.mapv(|v| v * v).sum() / n as f64;
    let noise_power = (&clean_signal - &noisy_signal).mapv(|v| v * v).sum() / n as f64;
    let snr_before = 10.0 * (signal_power / noise_power).log10();

    println!("Original signal SNR: {:.2} dB", snr_before);

    // Configure denoising
    let config = SparseRecoveryConfig {
        lambda: 0.1,       // Regularization parameter for L1 methods
        sparsity: Some(5), // Expected number of significant frequencies
        max_iterations: 200,
        convergence_threshold: 1e-6,
        ..Default::default()
    };

    // Apply sparse recovery for denoising
    let denoised_signal = sparse_denoise(
        &noisy_signal,
        SparseTransform::Frequency,
        SparseRecoveryMethod::OMP,
        &config,
    )
    .unwrap();

    // Calculate SNR after denoising
    let noise_power_after = (&clean_signal - &denoised_signal).mapv(|v| v * v).sum() / n as f64;
    let snr_after = 10.0 * (signal_power / noise_power_after).log10();

    println!("Denoised signal SNR: {:.2} dB", snr_after);
    println!("SNR improvement: {:.2} dB", snr_after - snr_before);

    // Calculate RMSE
    let diff_before = &clean_signal - &noisy_signal;
    let diff_after = &clean_signal - &denoised_signal;
    let rmse_before = vector_norm(&diff_before.view(), 2).unwrap() / (n as f64).sqrt();
    let rmse_after = vector_norm(&diff_after.view(), 2).unwrap() / (n as f64).sqrt();

    println!("RMSE before denoising: {:.6}", rmse_before);
    println!("RMSE after denoising: {:.6}", rmse_after);
}

/// Compare different recovery methods
fn compare_recovery_methods() {
    // Create a sparse signal
    let n = 200; // Signal dimension
    let k = 10; // Sparsity level

    let mut rng = rand::rng();

    // Generate random sparse signal
    let mut original_signal = Array1::zeros(n);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    indices.truncate(k);

    for &idx in &indices {
        original_signal[idx] = 20.0 * rng.random::<f64>() - 10.0;
    }

    // Number of measurements
    let m = 60;

    // Create sensing matrix
    let phi = random_sensing_matrix(m, n, Some(42));

    // Generate measurements
    let y = phi.dot(&original_signal);

    // Add a small amount of noise
    let noise_level = 0.02;
    let noisy_y = y
        .clone()
        .mapv(|v| v + noise_level * (2.0 * rng.random::<f64>() - 1.0));

    // Recovery methods to compare
    let methods = vec![
        SparseRecoveryMethod::OMP,
        SparseRecoveryMethod::MP,
        SparseRecoveryMethod::ISTA,
        SparseRecoveryMethod::FISTA,
        SparseRecoveryMethod::IHT,
        SparseRecoveryMethod::CoSaMP,
        SparseRecoveryMethod::SmoothL0,
    ];

    // Basic configuration
    let config = SparseRecoveryConfig {
        sparsity: Some(k),
        max_iterations: 300,
        convergence_threshold: 1e-6,
        lambda: 0.1, // For L1 methods
        ..Default::default()
    };

    println!("\nMethod Comparison (m={}, n={}, k={}):", m, n, k);
    println!("| Method | Error | Time (ms) | Sparsity |");
    println!("|--------|-------|-----------|----------|");

    // Run each method and measure performance
    for method in methods {
        let start_time = std::time::Instant::now();

        let recovered_signal = compressed_sensing_recover(&noisy_y, &phi, method, &config).unwrap();

        let elapsed = start_time.elapsed().as_millis();

        // Calculate recovery error
        let diff = &original_signal - &recovered_signal;
        let recovery_error = vector_norm(&diff.view(), 2).unwrap()
            / vector_norm(&original_signal.view(), 2).unwrap();

        // Measure output sparsity
        let sparsity = measure_sparsity(&recovered_signal, 1e-6).unwrap();

        println!(
            "| {:?} | {:.6} | {:5} | {:.6} |",
            method, recovery_error, elapsed, sparsity
        );
    }
}

/// Example of image inpainting with sparse recovery
fn image_inpainting_example() {
    // Create a simple test image (a gradient pattern)
    let n_rows = 32;
    let n_cols = 32;

    let mut image = Array2::zeros((n_rows, n_cols));

    // Fill with a pattern
    for i in 0..n_rows {
        for j in 0..n_cols {
            // Create a pattern that will be sparse in frequency domain
            let x = i as f64 / n_rows as f64;
            let y = j as f64 / n_cols as f64;

            image[[i, j]] = 0.5
                + 0.5
                    * (2.0 * std::f64::consts::PI * 3.0 * x).sin()
                    * (2.0 * std::f64::consts::PI * 2.0 * y).cos();
        }
    }

    // Create random missing pixels
    let mut rng = rand::rng();
    let missing_ratio = 0.3; // 30% of pixels are missing

    let mut masked_image = image.clone();

    for i in 0..n_rows {
        for j in 0..n_cols {
            if rng.random::<f64>() < missing_ratio {
                masked_image[[i, j]] = f64::NAN;
            }
        }
    }

    // Count missing pixels
    let missing_count = masked_image.iter().filter(|&&x| x.is_nan()).count();
    println!(
        "Image size: {}x{}, Missing pixels: {} ({}%)",
        n_rows,
        n_cols,
        missing_count,
        100.0 * missing_count as f64 / (n_rows * n_cols) as f64
    );

    // Configure inpainting
    let config = SparseRecoveryConfig {
        sparsity: Some(20), // Expected sparsity in frequency domain
        max_iterations: 200,
        convergence_threshold: 1e-6,
        ..Default::default()
    };

    // Perform inpainting using 8x8 patches
    let inpainted_image = image_inpainting(
        &masked_image,
        8, // Patch size
        SparseRecoveryMethod::OMP,
        &config,
    )
    .unwrap();

    // Calculate error on missing pixels
    let mut error_sum = 0.0;
    let mut count = 0;

    for i in 0..n_rows {
        for j in 0..n_cols {
            if masked_image[[i, j]].is_nan() {
                error_sum += (image[[i, j]] - inpainted_image[[i, j]]).powi(2);
                count += 1;
            }
        }
    }

    let rmse = (error_sum / count as f64).sqrt();
    println!("RMSE on missing pixels: {:.6}", rmse);

    println!("\nOriginal vs Inpainted (sample of 5 missing pixels):");
    let mut shown = 0;

    for i in 0..n_rows {
        for j in 0..n_cols {
            if masked_image[[i, j]].is_nan() && shown < 5 {
                println!(
                    "[{:2},{:2}]: {:10.5} vs {:10.5}",
                    i,
                    j,
                    image[[i, j]],
                    inpainted_image[[i, j]]
                );
                shown += 1;
            }
        }
    }
}
