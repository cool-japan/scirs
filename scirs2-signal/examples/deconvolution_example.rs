use ndarray::{s, Array1, Array2, ArrayView1};
use plotters::prelude::*;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use scirs2_signal::{convolve, deconvolution, SignalResult};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

fn main() -> SignalResult<()> {
    println!("Signal Deconvolution Examples");

    // Example 1: 1D signal deconvolution
    deconvolve_signal_1d()?;

    // Example 2: 1D signal with multiple deconvolution methods
    compare_deconvolution_methods_1d()?;

    // Example 3: Blind deconvolution
    blind_deconvolution_example()?;

    // Example 4: Image deconvolution
    deconvolve_image_2d()?;

    // Example 5: Optimal parameter selection
    optimal_parameter_selection()?;

    Ok(())
}

/// Example of 1D signal deconvolution
fn deconvolve_signal_1d() -> SignalResult<()> {
    println!("1D Signal Deconvolution Example");

    // Generate a test signal
    let n = 200;
    let x = Array1::linspace(0.0, 10.0, n);

    // Create a "true" signal (square wave)
    let mut true_signal = Array1::zeros(n);
    for i in 0..n {
        if i >= n / 4 && i < 3 * n / 4 {
            true_signal[i] = 1.0;
        }
    }

    // Create a PSF (Gaussian)
    let psf_size = 21;
    let mut psf = Array1::zeros(psf_size);
    let sigma = 2.0;

    for i in 0..psf_size {
        let x = (i as f64) - (psf_size as f64 - 1.0) / 2.0;
        psf[i] = (-x * x / (2.0 * sigma * sigma)).exp();
    }

    // Normalize PSF
    let psf_sum = psf.sum();
    psf /= psf_sum;

    // Convolve the signal with PSF to get the blurred signal
    let blurred = convolve::convolve(&true_signal, &psf, "same")?;

    // Add noise
    let noise_level = 0.02;
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, noise_level).unwrap();

    let mut noisy_blurred = blurred.clone();
    for i in 0..n {
        noisy_blurred[i] += normal.sample(&mut rng);
    }

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
        prefilter_sigma: 0.5,
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

    // Calculate error metrics
    let wiener_mse = calculate_mse(&wiener_result, &true_signal);
    let lucy_mse = calculate_mse(&lucy_result, &true_signal);
    let tikhonov_mse = calculate_mse(&tikhonov_result, &true_signal);

    println!("Wiener deconvolution MSE: {:.6}", wiener_mse);
    println!("Richardson-Lucy deconvolution MSE: {:.6}", lucy_mse);
    println!("Tikhonov deconvolution MSE: {:.6}", tikhonov_mse);

    // Export data for plotting
    export_to_csv(
        "deconvolution_1d.csv",
        &[
            ("True", &true_signal),
            ("Blurred", &blurred),
            ("Noisy", &noisy_blurred),
            ("Wiener", &wiener_result),
            ("Lucy", &lucy_result),
            ("Tikhonov", &tikhonov_result),
        ],
    )?;

    Ok(())
}

/// Example of comparing different deconvolution methods on a 1D signal
fn compare_deconvolution_methods_1d() -> SignalResult<()> {
    println!("Comparing Deconvolution Methods Example");

    // Generate a test signal (superposition of sine waves)
    let n = 256;
    let x = Array1::linspace(0.0, 8.0 * PI, n);

    // Create a "true" signal
    let mut true_signal = Array1::zeros(n);
    for i in 0..n {
        let xi = x[i];
        true_signal[i] = (xi * 0.5).sin() + (xi * 1.5).sin() * 0.3 + (xi * 3.0).sin() * 0.2;
    }

    // Create a PSF (Gaussian)
    let psf_size = 31;
    let mut psf = Array1::zeros(psf_size);
    let sigma = 3.0;

    for i in 0..psf_size {
        let x = (i as f64) - (psf_size as f64 - 1.0) / 2.0;
        psf[i] = (-x * x / (2.0 * sigma * sigma)).exp();
    }

    // Normalize PSF
    let psf_sum = psf.sum();
    psf /= psf_sum;

    // Convolve the signal with PSF to get the blurred signal
    let blurred = convolve::convolve(&true_signal, &psf, "same")?;

    // Add noise
    let noise_level = 0.01;
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, noise_level).unwrap();

    let mut noisy_blurred = blurred.clone();
    for i in 0..n {
        noisy_blurred[i] += normal.sample(&mut rng);
    }

    // Configure deconvolution
    let config = deconvolution::DeconvolutionConfig {
        reg_param: 0.005,
        max_iterations: 50,
        convergence_threshold: 1e-6,
        positivity_constraint: false, // Allow negative values for this signal
        use_fft: true,
        pad_signal: true,
        auto_regularization: false,
        prefilter: false,
        prefilter_sigma: 0.5,
        enforce_boundary: true,
    };

    // Apply different deconvolution methods
    let wiener_result =
        deconvolution::wiener_deconvolution_1d(&noisy_blurred, &psf, noise_level.powi(2), &config)?;

    let lucy_result =
        deconvolution::richardson_lucy_deconvolution_1d(&noisy_blurred, &psf, Some(30), &config)?;

    let tikhonov_result =
        deconvolution::tikhonov_deconvolution_1d(&noisy_blurred, &psf, Some(0.005), &config)?;

    let clean_result = deconvolution::clean_deconvolution_1d(
        &noisy_blurred,
        &psf,
        0.1,  // Gain factor
        0.01, // Threshold
        &config,
    )?;

    let mem_result =
        deconvolution::mem_deconvolution_1d(&noisy_blurred, &psf, noise_level.powi(2), &config)?;

    // Calculate error metrics
    let wiener_mse = calculate_mse(&wiener_result, &true_signal);
    let lucy_mse = calculate_mse(&lucy_result, &true_signal);
    let tikhonov_mse = calculate_mse(&tikhonov_result, &true_signal);
    let clean_mse = calculate_mse(&clean_result, &true_signal);
    let mem_mse = calculate_mse(&mem_result, &true_signal);

    println!("Wiener deconvolution MSE: {:.6}", wiener_mse);
    println!("Richardson-Lucy deconvolution MSE: {:.6}", lucy_mse);
    println!("Tikhonov deconvolution MSE: {:.6}", tikhonov_mse);
    println!("CLEAN deconvolution MSE: {:.6}", clean_mse);
    println!("MEM deconvolution MSE: {:.6}", mem_mse);

    // Export data for plotting
    export_to_csv(
        "deconvolution_methods_comparison.csv",
        &[
            ("True", &true_signal),
            ("Blurred", &blurred),
            ("Noisy", &noisy_blurred),
            ("Wiener", &wiener_result),
            ("Lucy", &lucy_result),
            ("Tikhonov", &tikhonov_result),
            ("CLEAN", &clean_result),
            ("MEM", &mem_result),
        ],
    )?;

    Ok(())
}

/// Example of blind deconvolution
fn blind_deconvolution_example() -> SignalResult<()> {
    println!("Blind Deconvolution Example");

    // Generate a test signal
    let n = 256;

    // Create a "true" signal (combination of Gaussians)
    let mut true_signal = Array1::zeros(n);
    for i in 0..n {
        let x = (i as f64) - (n as f64) / 2.0;

        let g1 = 2.0 * (-(x - 50.0).powi(2) / 100.0).exp();
        let g2 = 1.5 * (-(x + 30.0).powi(2) / 200.0).exp();
        let g3 = 1.0 * (-(x - 10.0).powi(2) / 50.0).exp();

        true_signal[i] = g1 + g2 + g3;
    }

    // Create a PSF (asymmetric to make it more challenging)
    let psf_size = 21;
    let mut psf = Array1::zeros(psf_size);
    let sigma1 = 2.0;
    let sigma2 = 3.0;

    for i in 0..psf_size {
        let x = (i as f64) - (psf_size as f64 - 1.0) / 2.0;
        let sigma = if x < 0.0 { sigma1 } else { sigma2 };
        psf[i] = (-x * x / (2.0 * sigma * sigma)).exp();
    }

    // Normalize PSF
    let psf_sum = psf.sum();
    psf /= psf_sum;

    // Convolve the signal with PSF to get the blurred signal
    let blurred = convolve::convolve(&true_signal, &psf, "same")?;

    // Add noise
    let noise_level = 0.01;
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, noise_level).unwrap();

    let mut noisy_blurred = blurred.clone();
    for i in 0..n {
        noisy_blurred[i] += normal.sample(&mut rng);
    }

    // Configure deconvolution
    let config = deconvolution::DeconvolutionConfig {
        reg_param: 0.01,
        max_iterations: 30,
        convergence_threshold: 1e-6,
        positivity_constraint: true,
        use_fft: true,
        pad_signal: true,
        auto_regularization: false,
        prefilter: true,
        prefilter_sigma: 0.5,
        enforce_boundary: true,
    };

    // Apply blind deconvolution (without knowing the true PSF)
    let (blind_signal, estimated_psf) =
        deconvolution::blind_deconvolution_1d(&noisy_blurred, psf_size, &config)?;

    // Also apply standard (non-blind) deconvolution using the true PSF
    let wiener_result =
        deconvolution::wiener_deconvolution_1d(&noisy_blurred, &psf, noise_level.powi(2), &config)?;

    // Calculate error metrics
    let blind_mse = calculate_mse(&blind_signal, &true_signal);
    let wiener_mse = calculate_mse(&wiener_result, &true_signal);

    println!("Blind deconvolution MSE: {:.6}", blind_mse);
    println!(
        "Wiener deconvolution MSE (with true PSF): {:.6}",
        wiener_mse
    );

    // Export data for plotting
    export_to_csv(
        "blind_deconvolution.csv",
        &[
            ("True", &true_signal),
            ("Blurred", &blurred),
            ("Noisy", &noisy_blurred),
            ("Blind", &blind_signal),
            ("Wiener", &wiener_result),
        ],
    )?;

    // Export PSFs for comparison
    export_to_csv(
        "psf_comparison.csv",
        &[("True PSF", &psf), ("Estimated PSF", &estimated_psf)],
    )?;

    Ok(())
}

/// Example of 2D image deconvolution
fn deconvolve_image_2d() -> SignalResult<()> {
    println!("2D Image Deconvolution Example");

    // Generate a test image (simple geometric shapes)
    let height = 64;
    let width = 64;
    let mut true_image = Array2::zeros((height, width));

    // Add a rectangle
    for i in 20..40 {
        for j in 15..45 {
            true_image[[i, j]] = 1.0;
        }
    }

    // Add a circle
    let center_i = 32;
    let center_j = 32;
    let radius = 10;

    for i in 0..height {
        for j in 0..width {
            let di = (i as f64) - (center_i as f64);
            let dj = (j as f64) - (center_j as f64);
            let dist = (di * di + dj * dj).sqrt();

            if dist <= radius as f64 {
                true_image[[i, j]] = 0.5;
            }
        }
    }

    // Create a 2D PSF (Gaussian)
    let psf_size = 7;
    let mut psf = Array2::zeros((psf_size, psf_size));
    let sigma = 1.5;

    for i in 0..psf_size {
        for j in 0..psf_size {
            let di = (i as f64) - (psf_size as f64 - 1.0) / 2.0;
            let dj = (j as f64) - (psf_size as f64 - 1.0) / 2.0;
            psf[[i, j]] = (-(di * di + dj * dj) / (2.0 * sigma * sigma)).exp();
        }
    }

    // Normalize PSF
    let psf_sum = psf.sum();
    psf /= psf_sum;

    // Convolve the image with PSF to get the blurred image
    let blurred = convolve::convolve2d(&true_image, &psf, "same")?;

    // Add noise
    let noise_level = 0.01;
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, noise_level).unwrap();

    let mut noisy_blurred = blurred.clone();
    for i in 0..height {
        for j in 0..width {
            noisy_blurred[[i, j]] += normal.sample(&mut rng);
        }
    }

    // Configure deconvolution
    let config = deconvolution::DeconvolutionConfig {
        reg_param: 0.01,
        max_iterations: 30,
        convergence_threshold: 1e-6,
        positivity_constraint: true,
        use_fft: true,
        pad_signal: true,
        auto_regularization: false,
        prefilter: false,
        prefilter_sigma: 0.5,
        enforce_boundary: true,
    };

    // Apply Wiener deconvolution
    let wiener_result =
        deconvolution::wiener_deconvolution_2d(&noisy_blurred, &psf, noise_level.powi(2), &config)?;

    // Apply Richardson-Lucy deconvolution
    let lucy_result =
        deconvolution::richardson_lucy_deconvolution_2d(&noisy_blurred, &psf, Some(20), &config)?;

    // Apply TV deconvolution
    let tv_result = deconvolution::tv_deconvolution_2d(&noisy_blurred, &psf, 0.1, &config)?;

    // Calculate error metrics
    let wiener_mse = calculate_mse_2d(&wiener_result, &true_image);
    let lucy_mse = calculate_mse_2d(&lucy_result, &true_image);
    let tv_mse = calculate_mse_2d(&tv_result, &true_image);

    println!("Wiener deconvolution MSE: {:.6}", wiener_mse);
    println!("Richardson-Lucy deconvolution MSE: {:.6}", lucy_mse);
    println!("TV deconvolution MSE: {:.6}", tv_mse);

    // Export data for visualization (just a few rows for demonstration)
    let row_idx = height / 2;

    export_to_csv(
        "deconvolution_2d_slice.csv",
        &[
            ("True", &true_image.slice(s![row_idx, ..]).to_owned()),
            ("Blurred", &blurred.slice(s![row_idx, ..]).to_owned()),
            ("Noisy", &noisy_blurred.slice(s![row_idx, ..]).to_owned()),
            ("Wiener", &wiener_result.slice(s![row_idx, ..]).to_owned()),
            ("Lucy", &lucy_result.slice(s![row_idx, ..]).to_owned()),
            ("TV", &tv_result.slice(s![row_idx, ..]).to_owned()),
        ],
    )?;

    // Save images as CSV files
    save_image_as_csv("true_image.csv", &true_image)?;
    save_image_as_csv("blurred_image.csv", &blurred)?;
    save_image_as_csv("noisy_blurred_image.csv", &noisy_blurred)?;
    save_image_as_csv("wiener_image.csv", &wiener_result)?;
    save_image_as_csv("lucy_image.csv", &lucy_result)?;
    save_image_as_csv("tv_image.csv", &tv_result)?;

    Ok(())
}

/// Example of optimal parameter selection for deconvolution
fn optimal_parameter_selection() -> SignalResult<()> {
    println!("Optimal Parameter Selection Example");

    // Generate a test signal
    let n = 256;
    let x = Array1::linspace(0.0, 10.0, n);

    // Create a "true" signal (combination of peaks)
    let mut true_signal = Array1::zeros(n);
    for i in 0..n {
        let xi = x[i];
        true_signal[i] = 2.0 * (-(xi - 2.0).powi(2) / 0.2).exp()
            + 1.5 * (-(xi - 5.0).powi(2) / 0.3).exp()
            + 1.0 * (-(xi - 8.0).powi(2) / 0.1).exp();
    }

    // Create a PSF (Gaussian)
    let psf_size = 21;
    let mut psf = Array1::zeros(psf_size);
    let sigma = 2.0;

    for i in 0..psf_size {
        let x = (i as f64) - (psf_size as f64 - 1.0) / 2.0;
        psf[i] = (-x * x / (2.0 * sigma * sigma)).exp();
    }

    // Normalize PSF
    let psf_sum = psf.sum();
    psf /= psf_sum;

    // Convolve the signal with PSF to get the blurred signal
    let blurred = convolve::convolve(&true_signal, &psf, "same")?;

    // Add noise
    let noise_level = 0.02;
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, noise_level).unwrap();

    let mut noisy_blurred = blurred.clone();
    for i in 0..n {
        noisy_blurred[i] += normal.sample(&mut rng);
    }

    // Estimate optimal regularization parameter
    let optimal_param =
        deconvolution::estimate_regularization_param(&noisy_blurred, &psf, 1e-6, 1.0, 20)?;

    println!(
        "Estimated optimal regularization parameter: {:.6}",
        optimal_param
    );

    // Configure deconvolution
    let manual_config = deconvolution::DeconvolutionConfig {
        reg_param: 0.01, // Manual parameter selection
        max_iterations: 50,
        convergence_threshold: 1e-6,
        positivity_constraint: true,
        use_fft: true,
        pad_signal: true,
        auto_regularization: false,
        prefilter: false,
        prefilter_sigma: 0.5,
        enforce_boundary: true,
    };

    let auto_config = deconvolution::DeconvolutionConfig {
        reg_param: optimal_param, // Use estimated parameter
        max_iterations: 50,
        convergence_threshold: 1e-6,
        positivity_constraint: true,
        use_fft: true,
        pad_signal: true,
        auto_regularization: false,
        prefilter: false,
        prefilter_sigma: 0.5,
        enforce_boundary: true,
    };

    // Apply Wiener deconvolution with manual parameter
    let manual_result = deconvolution::wiener_deconvolution_1d(
        &noisy_blurred,
        &psf,
        manual_config.reg_param,
        &manual_config,
    )?;

    // Apply Wiener deconvolution with optimal parameter
    let auto_result = deconvolution::wiener_deconvolution_1d(
        &noisy_blurred,
        &psf,
        auto_config.reg_param,
        &auto_config,
    )?;

    // Calculate error metrics
    let manual_mse = calculate_mse(&manual_result, &true_signal);
    let auto_mse = calculate_mse(&auto_result, &true_signal);

    println!(
        "Manual parameter Wiener deconvolution MSE: {:.6}",
        manual_mse
    );
    println!(
        "Optimal parameter Wiener deconvolution MSE: {:.6}",
        auto_mse
    );

    // Export data for plotting
    export_to_csv(
        "optimal_parameter_selection.csv",
        &[
            ("True", &true_signal),
            ("Blurred", &blurred),
            ("Noisy", &noisy_blurred),
            ("Manual", &manual_result),
            ("Auto", &auto_result),
        ],
    )?;

    // Try different parameter values and calculate MSE
    let mut param_values = Vec::new();
    let mut mse_values = Vec::new();

    for i in 0..20 {
        let param = 10.0_f64.powf(-6.0 + 0.3 * i as f64);
        param_values.push(param);

        let config = deconvolution::DeconvolutionConfig {
            reg_param: param,
            ..manual_config
        };

        let result = deconvolution::wiener_deconvolution_1d(&noisy_blurred, &psf, param, &config)?;

        let mse = calculate_mse(&result, &true_signal);
        mse_values.push(mse);
    }

    // Export parameter sweep data
    let mut file = File::create("parameter_sweep.csv")?;
    writeln!(file, "Parameter,MSE")?;

    for i in 0..param_values.len() {
        writeln!(file, "{},{}", param_values[i], mse_values[i])?;
    }

    Ok(())
}

/// Calculate mean squared error between two signals
fn calculate_mse(signal: &Array1<f64>, reference: &Array1<f64>) -> f64 {
    if signal.len() != reference.len() {
        return f64::NAN;
    }

    let n = signal.len();
    let mut sum_squared_error = 0.0;

    for i in 0..n {
        let error = signal[i] - reference[i];
        sum_squared_error += error * error;
    }

    sum_squared_error / (n as f64)
}

/// Calculate mean squared error between two images
fn calculate_mse_2d(image: &Array2<f64>, reference: &Array2<f64>) -> f64 {
    if image.dim() != reference.dim() {
        return f64::NAN;
    }

    let (height, width) = image.dim();
    let n = height * width;
    let mut sum_squared_error = 0.0;

    for i in 0..height {
        for j in 0..width {
            let error = image[[i, j]] - reference[[i, j]];
            sum_squared_error += error * error;
        }
    }

    sum_squared_error / (n as f64)
}

/// Export signal data to CSV for external plotting
fn export_to_csv(file_name: &str, signals: &[(&str, &Array1<f64>)]) -> SignalResult<()> {
    let mut file = File::create(file_name)?;

    // Write header
    let header = signals
        .iter()
        .map(|(name, _)| name.to_string())
        .collect::<Vec<String>>()
        .join(",");
    writeln!(file, "{}", header)?;

    // Find common signal length
    let min_len = signals.iter().map(|(_, data)| data.len()).min().unwrap();

    // Write data
    for i in 0..min_len {
        let line = signals
            .iter()
            .map(|(_, data)| data[i].to_string())
            .collect::<Vec<String>>()
            .join(",");
        writeln!(file, "{}", line)?;
    }

    println!("Data exported to {}", file_name);
    Ok(())
}

/// Save an image as a CSV file
fn save_image_as_csv(file_name: &str, image: &Array2<f64>) -> SignalResult<()> {
    let mut file = File::create(file_name)?;
    let (height, width) = image.dim();

    for i in 0..height {
        let line = (0..width)
            .map(|j| image[[i, j]].to_string())
            .collect::<Vec<String>>()
            .join(",");
        writeln!(file, "{}", line)?;
    }

    println!("Image exported to {}", file_name);
    Ok(())
}
