use ndarray::{s, Array1, Array2};
// use plotters::prelude::*; // Commented out until plotters is added as dependency
use rand::rng;
use rand_distr::{Distribution, Normal};
use scirs2_signal::{deconvolution, SignalError, SignalResult};
// use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

fn main() -> SignalResult<()> {
    println!("Running deconvolution examples...\n");

    basic_deconvolution()?;
    compare_synthetic_signals()?;
    demonstration_with_different_psfs()?;
    blind_deconvolution_example()?;
    regularization_parameter_selection()?;

    println!("\nAll examples completed successfully!");
    Ok(())
}

/// Basic deconvolution example showing different methods
fn basic_deconvolution() -> SignalResult<()> {
    println!("=== Basic Deconvolution ===");

    let n = 512;
    let x = Array1::linspace(0.0, 10.0, n);

    // Create a simple signal with two Gaussian peaks
    let mut true_signal = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi: f64 = x[i];
        true_signal[i] =
            (-((xi - 3.0).powi(2) / 0.5)).exp() + 0.7 * (-((xi - 7.0).powi(2) / 0.3)).exp();
    }

    // Create a Gaussian point spread function (PSF)
    let psf_width: f64 = 0.3;
    let mut psf = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi: f64 = x[i];
        psf[i] = (-(xi.powi(2) / (2.0 * psf_width.powi(2)))).exp();
    }

    // Normalize the PSF
    let psf_sum = psf.sum();
    psf /= psf_sum;

    // Convolve the signal with PSF to get the blurred signal
    let blurred_vec = scirs2_signal::convolve::convolve(
        true_signal.as_slice().unwrap(),
        psf.as_slice().unwrap(),
        "same",
    )?;
    let blurred = Array1::from(blurred_vec);

    // Add noise
    let noise_level = 0.02;
    let mut rng = rng();
    let normal = Normal::new(0.0, noise_level).unwrap();

    let mut noisy_blurred_vec = blurred.to_vec();
    for val in &mut noisy_blurred_vec {
        *val += normal.sample(&mut rng);
    }
    let noisy_blurred = Array1::from(noisy_blurred_vec);

    // Configure deconvolution
    let config = deconvolution::DeconvolutionConfig {
        reg_param: 0.01,
        max_iterations: 50,
        convergence_threshold: 1e-6,
        positivity_constraint: true,
        use_fft: true,
        pad_signal: true,
        auto_regularization: false,
        prefilter: false,
        prefilter_sigma: 1.0,
        enforce_boundary: true,
    };

    // Apply Wiener deconvolution
    let wiener_result =
        deconvolution::wiener_deconvolution_1d(&noisy_blurred, &psf, noise_level.powi(2), &config)?;

    // Apply Richardson-Lucy deconvolution
    let lucy_result =
        deconvolution::richardson_lucy_deconvolution_1d(&noisy_blurred, &psf, Some(30), &config)?;

    // Apply Tikhonov regularized deconvolution
    let tikhonov_result =
        deconvolution::tikhonov_deconvolution_1d(&noisy_blurred, &psf, Some(0.01), &config)?;

    // Calculate MSE for each method
    let wiener_mse = wiener_result
        .iter()
        .zip(true_signal.iter())
        .map(|(est, true_val)| (est - true_val).powi(2))
        .sum::<f64>()
        / n as f64;

    let lucy_mse = lucy_result
        .iter()
        .zip(true_signal.iter())
        .map(|(est, true_val)| (est - true_val).powi(2))
        .sum::<f64>()
        / n as f64;

    let tikhonov_mse = tikhonov_result
        .iter()
        .zip(true_signal.iter())
        .map(|(est, true_val)| (est - true_val).powi(2))
        .sum::<f64>()
        / n as f64;

    println!("MSE Results:");
    println!("  Wiener Deconvolution: {:.6}", wiener_mse);
    println!("  Richardson-Lucy: {:.6}", lucy_mse);
    println!("  Tikhonov: {:.6}", tikhonov_mse);

    // Export results for plotting with another tool
    export_to_csv(
        "basic_deconvolution.csv",
        &[
            ("True Signal", &true_signal),
            ("Blurred", &blurred),
            ("Noisy", &noisy_blurred),
            ("Wiener", &wiener_result),
            ("Richardson-Lucy", &lucy_result),
            ("Tikhonov", &tikhonov_result),
        ],
    )?;

    println!("Results exported to basic_deconvolution.csv");
    println!();

    Ok(())
}

/// Compare deconvolution performance on synthetic signals
fn compare_synthetic_signals() -> SignalResult<()> {
    println!("=== Compare Synthetic Signals ===");

    let n = 512;
    let x = Array1::linspace(0.0, 10.0, n);

    // Create different types of signals
    // 1. Sharp spikes (sparse signal)
    let mut sparse_signal = Array1::<f64>::zeros(n);
    sparse_signal[n / 4] = 1.0;
    sparse_signal[n / 2] = 0.7;
    sparse_signal[3 * n / 4] = 0.5;

    // 2. Smooth signal (Gaussian mixture)
    let mut smooth_signal = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi: f64 = x[i];
        smooth_signal[i] =
            0.8 * (-((xi - 3.0).powi(2) / 1.0)).exp() + (-((xi - 7.0).powi(2) / 0.5)).exp();
    }

    // Create PSF
    let psf_width: f64 = 0.2;
    let mut psf = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi: f64 = x[i];
        psf[i] = (-(xi.powi(2) / (2.0 * psf_width.powi(2)))).exp();
    }
    let psf_sum = psf.sum();
    psf /= psf_sum;

    // Test on both signal types
    for (signal_name, true_signal) in &[("Sparse", sparse_signal), ("Smooth", smooth_signal)] {
        println!("Testing on {} signal:", signal_name);

        // Blur the signal
        let blurred_vec = scirs2_signal::convolve::convolve(
            true_signal.as_slice().unwrap(),
            psf.as_slice().unwrap(),
            "same",
        )?;
        let blurred = Array1::from(blurred_vec);

        // Add noise
        let noise_level = 0.01;
        let mut rng = rng();
        let normal = Normal::new(0.0, noise_level).unwrap();

        let mut noisy_blurred_vec = blurred.to_vec();
        for val in &mut noisy_blurred_vec {
            *val += normal.sample(&mut rng);
        }
        let noisy_blurred = Array1::from(noisy_blurred_vec);

        // Configure deconvolution
        let config = deconvolution::DeconvolutionConfig::default();

        // Apply different methods
        let wiener_result = deconvolution::wiener_deconvolution_1d(
            &noisy_blurred,
            &psf,
            noise_level.powi(2),
            &config,
        )?;
        let lucy_result = deconvolution::richardson_lucy_deconvolution_1d(
            &noisy_blurred,
            &psf,
            Some(30),
            &config,
        )?;
        let tikhonov_result =
            deconvolution::tikhonov_deconvolution_1d(&noisy_blurred, &psf, Some(0.005), &config)?;

        // Add CLEAN method for sparse signals
        let clean_result =
            deconvolution::clean_deconvolution_1d(&noisy_blurred, &psf, 0.1, 0.01, &config)?;

        // Add MEM method
        let mem_result = deconvolution::mem_deconvolution_1d(
            &noisy_blurred,
            &psf,
            noise_level.powi(2),
            &config,
        )?;

        // Calculate MSE for each method
        let mse_results = vec![
            ("Wiener", calculate_mse(&wiener_result, true_signal)),
            ("Richardson-Lucy", calculate_mse(&lucy_result, true_signal)),
            ("Tikhonov", calculate_mse(&tikhonov_result, true_signal)),
            ("CLEAN", calculate_mse(&clean_result, true_signal)),
            ("MEM", calculate_mse(&mem_result, true_signal)),
        ];

        println!("  MSE Results:");
        for (method, mse) in &mse_results {
            println!("    {}: {:.6}", method, mse);
        }

        // Export results
        export_to_csv(
            &format!("{}_signal_deconvolution.csv", signal_name.to_lowercase()),
            &[
                ("True Signal", true_signal),
                ("Blurred", &blurred),
                ("Noisy", &noisy_blurred),
                ("Wiener", &wiener_result),
                ("Richardson-Lucy", &lucy_result),
                ("Tikhonov", &tikhonov_result),
                ("CLEAN", &clean_result),
                ("MEM", &mem_result),
            ],
        )?;
    }

    println!();
    Ok(())
}

/// Demonstrate deconvolution with different PSF types
fn demonstration_with_different_psfs() -> SignalResult<()> {
    println!("=== Different PSF Types ===");

    let n = 512;
    let x = Array1::linspace(0.0, 10.0, n);

    // Create a test signal
    let mut true_signal = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi: f64 = x[i];
        true_signal[i] = (-((xi - 3.0).powi(2) / 0.5)).exp()
            + 0.5 * (-((xi - 6.0).powi(2) / 0.3)).exp()
            + 0.3 * (-((xi - 8.0).powi(2) / 0.1)).exp();
    }

    // Different PSF types
    let psfs = vec![
        // 1. Gaussian PSF
        {
            let width: f64 = 0.2;
            let mut psf = Array1::<f64>::zeros(n);
            for i in 0..n {
                let xi: f64 = x[i];
                psf[i] = (-(xi.powi(2) / (2.0f64 * width.powi(2)))).exp();
            }
            psf /= psf.sum();
            ("Gaussian", psf)
        },
        // 2. Motion blur PSF (horizontal line)
        {
            let mut psf = Array1::<f64>::zeros(n);
            let blur_length = 10;
            for i in 0..blur_length {
                psf[n / 2 - blur_length / 2 + i] = 1.0 / blur_length as f64;
            }
            ("Motion", psf)
        },
        // 3. Out-of-focus PSF (disk)
        {
            let mut psf = Array1::<f64>::zeros(n);
            let radius = 5.0;
            for i in 0..n {
                let xi = x[i] - x[n / 2];
                if xi.abs() <= radius {
                    psf[i] = 1.0;
                }
            }
            psf /= psf.sum();
            ("Out-of-focus", psf)
        },
    ];

    for (psf_name, psf) in psfs {
        println!("Testing with {} PSF:", psf_name);

        // Blur the signal
        let blurred_vec = scirs2_signal::convolve::convolve(
            true_signal.as_slice().unwrap(),
            psf.as_slice().unwrap(),
            "same",
        )?;
        let blurred = Array1::from(blurred_vec);

        // Add noise
        let noise_level = 0.02;
        let mut rng = rng();
        let normal = Normal::new(0.0, noise_level).unwrap();

        let mut noisy_blurred_vec = blurred.to_vec();
        for val in &mut noisy_blurred_vec {
            *val += normal.sample(&mut rng);
        }
        let noisy_blurred = Array1::from(noisy_blurred_vec);

        // Configure deconvolution
        let config = deconvolution::DeconvolutionConfig::default();

        // Apply different methods
        let wiener_result = deconvolution::wiener_deconvolution_1d(
            &noisy_blurred,
            &psf,
            noise_level.powi(2),
            &config,
        )?;

        let richardson_lucy_result = deconvolution::richardson_lucy_deconvolution_1d(
            &noisy_blurred,
            &psf,
            Some(50),
            &config,
        )?;

        // Calculate and print results
        let wiener_mse = calculate_mse(&wiener_result, &true_signal);
        let rl_mse = calculate_mse(&richardson_lucy_result, &true_signal);

        println!(
            "  MSE - Wiener: {:.6}, Richardson-Lucy: {:.6}",
            wiener_mse, rl_mse
        );

        // Export results
        export_to_csv(
            &format!("{}_psf_deconvolution.csv", psf_name.to_lowercase()),
            &[
                ("True Signal", &true_signal),
                ("Blurred", &blurred),
                ("Noisy", &noisy_blurred),
                ("Wiener", &wiener_result),
                ("Richardson-Lucy", &richardson_lucy_result),
            ],
        )?;
    }

    println!();
    Ok(())
}

/// Demonstrate blind deconvolution (when PSF is unknown)
fn blind_deconvolution_example() -> SignalResult<()> {
    println!("=== Blind Deconvolution ===");

    let n = 512;
    let x = Array1::linspace(0.0, 10.0, n);

    // Create a test signal
    let mut true_signal = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi: f64 = x[i];
        true_signal[i] =
            (-((xi - 3.0f64).powi(2) / 0.5)).exp() + 0.7 * (-((xi - 7.0f64).powi(2) / 0.3)).exp();
    }

    // Create true PSF (unknown to the algorithm)
    let true_psf_width: f64 = 0.25;
    let mut true_psf = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi: f64 = x[i];
        true_psf[i] = (-(xi.powi(2) / (2.0f64 * true_psf_width.powi(2)))).exp();
    }
    true_psf /= true_psf.sum();

    // Blur the signal
    let blurred_vec = scirs2_signal::convolve::convolve(
        true_signal.as_slice().unwrap(),
        true_psf.as_slice().unwrap(),
        "same",
    )?;
    let blurred = Array1::from(blurred_vec);

    // Add noise
    let noise_level = 0.02;
    let mut rng = rng();
    let normal = Normal::new(0.0, noise_level).unwrap();

    let mut noisy_blurred_vec = blurred.to_vec();
    for val in &mut noisy_blurred_vec {
        *val += normal.sample(&mut rng);
    }
    let noisy_blurred = Array1::from(noisy_blurred_vec);

    // Configure deconvolution
    let config = deconvolution::DeconvolutionConfig::default();

    // Apply blind deconvolution
    println!("Estimating PSF from blurred signal...");
    let psf_size = 51; // Estimated size of the PSF
    let (blind_result, estimated_psf) =
        deconvolution::blind_deconvolution_1d(&noisy_blurred, psf_size, &config)?;

    // For comparison, also apply Wiener deconvolution with true PSF
    let wiener_with_true_psf = deconvolution::wiener_deconvolution_1d(
        &noisy_blurred,
        &true_psf,
        noise_level.powi(2),
        &config,
    )?;

    // Calculate MSE
    let blind_mse = calculate_mse(&blind_result, &true_signal);
    let wiener_mse = calculate_mse(&wiener_with_true_psf, &true_signal);
    let psf_mse = calculate_mse(&estimated_psf, &true_psf.slice(s![..psf_size]).to_owned());

    println!("MSE Results:");
    println!("  Blind Deconvolution: {:.6}", blind_mse);
    println!("  Wiener (true PSF): {:.6}", wiener_mse);
    println!("  Estimated PSF: {:.6}", psf_mse);

    // Export results
    export_to_csv(
        "blind_deconvolution.csv",
        &[
            ("True Signal", &true_signal),
            ("Blurred", &blurred),
            ("Noisy", &noisy_blurred),
            ("Blind Result", &blind_result),
            ("Wiener (true PSF)", &wiener_with_true_psf),
        ],
    )?;

    // Save estimated PSF
    export_to_csv(
        "estimated_psf.csv",
        &[
            ("True PSF", &true_psf.slice(s![..psf_size]).to_owned()),
            ("Estimated PSF", &estimated_psf),
        ],
    )?;

    println!();
    Ok(())
}

/// Demonstrate optimal regularization parameter selection
fn regularization_parameter_selection() -> SignalResult<()> {
    println!("=== Regularization Parameter Selection ===");

    let n = 512;
    let x = Array1::linspace(0.0, 10.0, n);

    // Create a test signal with multiple features
    let mut true_signal = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi: f64 = x[i];
        true_signal[i] = 2.0 * (-((xi - 2.0).powi(2) / 0.2)).exp()
            + 1.5 * (-((xi - 5.0).powi(2) / 0.3)).exp()
            + 1.0 * (-((xi - 8.0).powi(2) / 0.1)).exp();
    }

    // Create PSF
    let psf_width: f64 = 0.3;
    let mut psf = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi: f64 = x[i];
        psf[i] = (-(xi.powi(2) / (2.0f64 * psf_width.powi(2)))).exp();
    }
    psf /= psf.sum();

    // Blur the signal
    let blurred_vec = scirs2_signal::convolve::convolve(
        true_signal.as_slice().unwrap(),
        psf.as_slice().unwrap(),
        "same",
    )?;
    let blurred = Array1::from(blurred_vec);

    // Add noise
    let noise_level = 0.02;
    let mut rng = rng();
    let normal = Normal::new(0.0, noise_level).unwrap();

    let mut noisy_blurred_vec = blurred.to_vec();
    for val in &mut noisy_blurred_vec {
        *val += normal.sample(&mut rng);
    }
    let noisy_blurred = Array1::from(noisy_blurred_vec);

    // Test different regularization parameter values
    let param_values = vec![1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0];
    let mut mse_values = Vec::new();

    // Use automatic parameter estimation
    println!("Estimating optimal regularization parameter...");
    let estimated_param =
        deconvolution::estimate_regularization_param(&noisy_blurred, &psf, 1e-6, 1.0, 20)?;
    println!("Estimated parameter: {:.6}", estimated_param);

    // Configure deconvolution
    let config = deconvolution::DeconvolutionConfig::default();

    // Test different parameter values
    println!("Testing different regularization parameters:");
    for &param in &param_values {
        let wiener_result = deconvolution::wiener_deconvolution_1d(
            &noisy_blurred,
            &psf,
            noise_level.powi(2) + param,
            &config,
        )?;

        let mse = calculate_mse(&wiener_result, &true_signal);
        mse_values.push(mse);
        println!("  λ = {:.1e}: MSE = {:.6}", param, mse);
    }

    // Find optimal parameter from tested values
    let min_mse_idx = mse_values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    let optimal_param = param_values[min_mse_idx];

    println!(
        "\nOptimal parameter (from grid search): {:.1e}",
        optimal_param
    );
    println!("Minimum MSE: {:.6}", mse_values[min_mse_idx]);

    // Compare manual vs automatic parameter selection
    let manual_result = deconvolution::wiener_deconvolution_1d(
        &noisy_blurred,
        &psf,
        noise_level.powi(2) + optimal_param,
        &config,
    )?;

    let auto_result = deconvolution::wiener_deconvolution_1d(
        &noisy_blurred,
        &psf,
        noise_level.powi(2) + estimated_param,
        &config,
    )?;

    let manual_mse = calculate_mse(&manual_result, &true_signal);
    let auto_mse = calculate_mse(&auto_result, &true_signal);

    println!("\nComparison:");
    println!("  Manual parameter selection: MSE = {:.6}", manual_mse);
    println!("  Automatic parameter selection: MSE = {:.6}", auto_mse);

    // Export results
    export_to_csv(
        "parameter_selection.csv",
        &[
            ("True Signal", &true_signal),
            ("Blurred", &blurred),
            ("Noisy", &noisy_blurred),
            ("Manual Selection", &manual_result),
            ("Auto Selection", &auto_result),
        ],
    )?;

    // Save parameter sweep results
    let mut file =
        File::create("parameter_sweep.csv").map_err(|e| SignalError::Compute(e.to_string()))?;
    writeln!(file, "Parameter,MSE").map_err(|e| SignalError::Compute(e.to_string()))?;
    for i in 0..param_values.len() {
        writeln!(file, "{},{}", param_values[i], mse_values[i])
            .map_err(|e| SignalError::Compute(e.to_string()))?;
    }

    println!();
    Ok(())
}

/// Calculate mean squared error between two signals
fn calculate_mse(estimate: &Array1<f64>, true_signal: &Array1<f64>) -> f64 {
    estimate
        .iter()
        .zip(true_signal.iter())
        .map(|(est, true_val)| (est - true_val).powi(2))
        .sum::<f64>()
        / estimate.len() as f64
}

/// Export signals to CSV file for external plotting
fn export_to_csv(file_name: &str, signals: &[(&str, &Array1<f64>)]) -> SignalResult<()> {
    let mut file = File::create(file_name).map_err(|e| {
        scirs2_signal::error::SignalError::Compute(format!("Failed to create file: {}", e))
    })?;

    // Write header
    let header = signals
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(",");
    writeln!(file, "Index,{}", header).map_err(|e| {
        scirs2_signal::error::SignalError::Compute(format!("Failed to write header: {}", e))
    })?;

    // Write data
    let n = signals[0].1.len();
    for i in 0..n {
        let mut line = format!("{}", i);
        for (_, signal) in signals {
            line.push_str(&format!(",{}", signal[i]));
        }
        writeln!(file, "{}", line).map_err(|e| {
            scirs2_signal::error::SignalError::Compute(format!("Failed to write data: {}", e))
        })?;
    }

    Ok(())
}

/// Save 2D image data as CSV
#[allow(dead_code)]
fn save_image_as_csv(file_name: &str, image: &Array2<f64>) -> SignalResult<()> {
    let mut file = File::create(file_name).map_err(|e| {
        scirs2_signal::error::SignalError::Compute(format!("Failed to create file: {}", e))
    })?;

    for row in image.rows() {
        let line = row
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        writeln!(file, "{}", line).map_err(|e| {
            scirs2_signal::error::SignalError::Compute(format!("Failed to write data: {}", e))
        })?;
    }

    Ok(())
}
