use ndarray::{Array1, Array2};
use rand::Rng;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

use scirs2_signal::waveforms;
use scirs2_signal::wiener::{
    iterative_wiener_filter, kalman_wiener_filter, psd_wiener_filter, spectral_subtraction,
    wiener_filter, wiener_filter_2d, wiener_filter_freq, wiener_filter_time, WienerConfig,
};

fn main() {
    println!("Wiener Filtering Examples");
    println!("------------------------");

    // Create test signals
    println!("\nGenerating test signals...");
    let (clean_signal, noisy_signal) = generate_test_signal();
    let (clean_signal2, noisy_signal2) = generate_nonstationary_signal();
    let (clean_image, noisy_image) = generate_test_image();

    // Benchmark SNR before denoising
    let input_snr = calculate_snr(&clean_signal, &noisy_signal);
    println!("Input SNR: {:.2} dB", input_snr);

    // Standard Wiener filter
    println!("\n1. Basic Wiener filtering...");
    let denoised_basic = wiener_filter(&noisy_signal, None, None).unwrap();
    let basic_snr = calculate_snr(&clean_signal, &denoised_basic);
    println!("   Basic Wiener filter SNR: {:.2} dB", basic_snr);
    save_signals_to_csv(
        "wiener_basic.csv",
        &clean_signal,
        &noisy_signal,
        &denoised_basic,
        None,
    );

    // Frequency domain vs. time domain
    println!("\n2. Comparing frequency and time domain implementations...");

    // Configure filters
    let freq_config = WienerConfig {
        frequency_domain: true,
        ..Default::default()
    };

    let time_config = WienerConfig {
        frequency_domain: false,
        window_size: 21,
        ..Default::default()
    };

    // Apply filters
    let denoised_freq = wiener_filter_freq(&noisy_signal, &freq_config).unwrap();
    let denoised_time = wiener_filter_time(&noisy_signal, &time_config).unwrap();

    // Evaluate performance
    let freq_snr = calculate_snr(&clean_signal, &denoised_freq);
    let time_snr = calculate_snr(&clean_signal, &denoised_time);

    println!("   Frequency domain SNR: {:.2} dB", freq_snr);
    println!("   Time domain SNR: {:.2} dB", time_snr);

    save_signals_to_csv(
        "wiener_freq_vs_time.csv",
        &clean_signal,
        &noisy_signal,
        &denoised_freq,
        Some(&denoised_time),
    );

    // Iterative Wiener filter
    println!("\n3. Iterative Wiener filtering...");

    // Configure iterative filter
    let iter_config = WienerConfig {
        max_iterations: 3,
        ..Default::default()
    };

    // Apply filter
    let denoised_iter = iterative_wiener_filter(&noisy_signal, &iter_config).unwrap();
    let iter_snr = calculate_snr(&clean_signal, &denoised_iter);

    println!("   Iterative Wiener filter SNR: {:.2} dB", iter_snr);
    println!("   Improvement over basic: {:.2} dB", iter_snr - basic_snr);

    save_signals_to_csv(
        "wiener_iterative.csv",
        &clean_signal,
        &noisy_signal,
        &denoised_basic,
        Some(&denoised_iter),
    );

    // Spectral subtraction
    println!("\n4. Spectral subtraction...");

    // Apply with different parameters
    let denoised_spec1 = spectral_subtraction(&noisy_signal, None, Some(1.0), Some(0.01)).unwrap();
    let denoised_spec2 = spectral_subtraction(&noisy_signal, None, Some(2.0), Some(0.001)).unwrap();

    let spec1_snr = calculate_snr(&clean_signal, &denoised_spec1);
    let spec2_snr = calculate_snr(&clean_signal, &denoised_spec2);

    println!("   Spectral subtraction (α=1.0) SNR: {:.2} dB", spec1_snr);
    println!("   Spectral subtraction (α=2.0) SNR: {:.2} dB", spec2_snr);

    save_signals_to_csv(
        "spectral_subtraction.csv",
        &clean_signal,
        &noisy_signal,
        &denoised_spec1,
        Some(&denoised_spec2),
    );

    // PSD-based Wiener filter
    println!("\n5. PSD-based Wiener filtering...");

    // Apply filter
    let denoised_psd = psd_wiener_filter(&noisy_signal, None, None).unwrap();
    let psd_snr = calculate_snr(&clean_signal, &denoised_psd);

    println!("   PSD-based Wiener filter SNR: {:.2} dB", psd_snr);

    save_signals_to_csv(
        "wiener_psd.csv",
        &clean_signal,
        &noisy_signal,
        &denoised_psd,
        None,
    );

    // Non-stationary signal with Kalman-Wiener filter
    println!("\n6. Kalman-Wiener filter for non-stationary signals...");

    // Kalman filter requires process and measurement variances
    let process_var = 0.01; // State model variance
    let measurement_var = 0.5; // Observation variance

    // Apply standard Wiener and Kalman-Wiener filters
    let nonstat_wiener = wiener_filter(&noisy_signal2, None, None).unwrap();
    let nonstat_kalman =
        kalman_wiener_filter(&noisy_signal2, process_var, measurement_var).unwrap();

    // Calculate SNRs
    let nonstat_input_snr = calculate_snr(&clean_signal2, &noisy_signal2);
    let nonstat_wiener_snr = calculate_snr(&clean_signal2, &nonstat_wiener);
    let nonstat_kalman_snr = calculate_snr(&clean_signal2, &nonstat_kalman);

    println!(
        "   Non-stationary signal input SNR: {:.2} dB",
        nonstat_input_snr
    );
    println!(
        "   Standard Wiener filter SNR: {:.2} dB",
        nonstat_wiener_snr
    );
    println!("   Kalman-Wiener filter SNR: {:.2} dB", nonstat_kalman_snr);

    save_signals_to_csv(
        "wiener_nonstationary.csv",
        &clean_signal2,
        &noisy_signal2,
        &nonstat_wiener,
        Some(&nonstat_kalman),
    );

    // 2D image denoising
    println!("\n7. 2D image denoising...");

    // Apply 2D Wiener filter
    let denoised_image = wiener_filter_2d(&noisy_image, None, Some([5, 5])).unwrap();

    // Calculate image SNR
    let image_input_snr = calculate_image_snr(&clean_image, &noisy_image);
    let image_output_snr = calculate_image_snr(&clean_image, &denoised_image);

    println!("   Image input SNR: {:.2} dB", image_input_snr);
    println!("   Image output SNR: {:.2} dB", image_output_snr);
    println!(
        "   Improvement: {:.2} dB",
        image_output_snr - image_input_snr
    );

    save_image_to_csv("wiener_image_clean.csv", &clean_image);
    save_image_to_csv("wiener_image_noisy.csv", &noisy_image);
    save_image_to_csv("wiener_image_denoised.csv", &denoised_image);

    println!("\nDone! Results saved to CSV files for plotting.");
}

/// Generates a test signal with additive noise
fn generate_test_signal() -> (Array1<f64>, Array1<f64>) {
    // Signal parameters
    let n_samples = 1000;
    let sampling_rate = 1000.0;
    let t = Array1::linspace(0.0, (n_samples as f64 - 1.0) / sampling_rate, n_samples);

    // Generate a clean chirp signal
    let clean_signal = Array1::from_vec(
        waveforms::chirp(t.as_slice().unwrap(), 10.0, 1.0, 100.0, "linear", 0.0).unwrap(),
    );

    // Add Gaussian noise
    let noise_level = 0.5;
    let mut rng = rand::rng();
    let mut noisy_signal = clean_signal.clone();

    for i in 0..n_samples {
        noisy_signal[i] += noise_level * rng.random_range(-1.0..1.0);
    }

    (clean_signal, noisy_signal)
}

/// Generates a non-stationary signal with time-varying noise
fn generate_nonstationary_signal() -> (Array1<f64>, Array1<f64>) {
    // Signal parameters
    let n_samples = 1000;
    let sampling_rate = 1000.0;
    let t = Array1::linspace(0.0, (n_samples as f64 - 1.0) / sampling_rate, n_samples);

    // Generate a clean multi-component signal
    let f1 = 10.0;
    let f2 = 40.0;

    let clean_signal = t.mapv(|ti| (2.0 * PI * f1 * ti).sin() + 0.5 * (2.0 * PI * f2 * ti).sin());

    // Add time-varying noise (stronger in the middle)
    let mut rng = rand::rng();
    let mut noisy_signal = clean_signal.clone();

    for i in 0..n_samples {
        // Noise amplitude varies with time (parabolic shape)
        let noise_level = 0.2 + 0.8 * (1.0 - (i as f64 / n_samples as f64 * 2.0 - 1.0).powi(2));

        noisy_signal[i] += noise_level * rng.random_range(-1.0..1.0);
    }

    (clean_signal, noisy_signal)
}

/// Generates a test image with additive noise
fn generate_test_image() -> (Array2<f64>, Array2<f64>) {
    // Image parameters
    let height = 32;
    let width = 32;

    // Create a clean test image (a simple pattern)
    let mut clean_image = Array2::zeros((height, width));

    // Create a pattern with a circle and gradients
    for i in 0..height {
        for j in 0..width {
            // Normalize coordinates to [-1, 1]
            let x = j as f64 / width as f64 * 2.0 - 1.0;
            let y = i as f64 / height as f64 * 2.0 - 1.0;

            // Circle pattern
            let distance = (x.powi(2) + y.powi(2)).sqrt();
            if distance < 0.5 {
                clean_image[[i, j]] = 1.0;
            }

            // Add gradient
            clean_image[[i, j]] += 0.4 * x;
        }
    }

    // Normalize to [0, 1] range
    let min_val = clean_image.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = clean_image.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if (max_val - min_val).abs() > 1e-10 {
        clean_image.mapv_inplace(|x| (x - min_val) / (max_val - min_val));
    }

    // Add Gaussian noise
    let noise_level = 0.2;
    let mut rng = rand::rng();
    let mut noisy_image = clean_image.clone();

    for i in 0..height {
        for j in 0..width {
            noisy_image[[i, j]] += noise_level * rng.random_range(-1.0..1.0);
        }
    }

    // Clip to [0, 1] range
    noisy_image.mapv_inplace(|x| x.clamp(0.0, 1.0));

    (clean_image, noisy_image)
}

/// Calculates the Signal-to-Noise Ratio (SNR) in dB
fn calculate_snr(clean: &Array1<f64>, noisy: &Array1<f64>) -> f64 {
    if clean.len() != noisy.len() {
        return f64::NEG_INFINITY;
    }

    let mut signal_power = 0.0;
    let mut noise_power = 0.0;

    for i in 0..clean.len() {
        signal_power += clean[i].powi(2);
        noise_power += (clean[i] - noisy[i]).powi(2);
    }

    if noise_power < 1e-10 {
        return f64::INFINITY;
    }

    10.0 * (signal_power / noise_power).log10()
}

/// Calculates the SNR for images
fn calculate_image_snr(clean: &Array2<f64>, noisy: &Array2<f64>) -> f64 {
    if clean.dim() != noisy.dim() {
        return f64::NEG_INFINITY;
    }

    let mut signal_power = 0.0;
    let mut noise_power = 0.0;

    for i in 0..clean.dim().0 {
        for j in 0..clean.dim().1 {
            signal_power += clean[[i, j]].powi(2);
            noise_power += (clean[[i, j]] - noisy[[i, j]]).powi(2);
        }
    }

    if noise_power < 1e-10 {
        return f64::INFINITY;
    }

    10.0 * (signal_power / noise_power).log10()
}

/// Saves signals to a CSV file for plotting
fn save_signals_to_csv(
    filename: &str,
    clean: &Array1<f64>,
    noisy: &Array1<f64>,
    denoised1: &Array1<f64>,
    denoised2: Option<&Array1<f64>>,
) {
    let mut file = File::create(filename).expect("Failed to create file");

    // Write header
    if denoised2.is_some() {
        writeln!(file, "sample,clean,noisy,denoised1,denoised2").expect("Failed to write header");
    } else {
        writeln!(file, "sample,clean,noisy,denoised").expect("Failed to write header");
    }

    // Write data
    for i in 0..clean.len() {
        write!(file, "{},{},{},{}", i, clean[i], noisy[i], denoised1[i])
            .expect("Failed to write data");

        if let Some(d2) = denoised2 {
            write!(file, ",{}", d2[i]).expect("Failed to write data");
        }

        writeln!(file).expect("Failed to write data");
    }
}

/// Saves an image to CSV for visualization
fn save_image_to_csv(filename: &str, image: &Array2<f64>) {
    let mut file = File::create(filename).expect("Failed to create file");

    // Write header
    let (height, width) = image.dim();

    for j in 0..width {
        if j == 0 {
            write!(file, "row").expect("Failed to write header");
        }
        write!(file, ",{}", j).expect("Failed to write header");
    }
    writeln!(file).expect("Failed to write header");

    // Write data
    for i in 0..height {
        write!(file, "{}", i).expect("Failed to write data");

        for j in 0..width {
            write!(file, ",{:.6}", image[[i, j]]).expect("Failed to write data");
        }

        writeln!(file).expect("Failed to write data");
    }
}
