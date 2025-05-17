use ndarray::{Array1, Array2, Array3};
use rand::{rng, Rng};
use std::fs::File;
use std::io::Write;

use scirs2_signal::nlm::{
    nlm_block_matching_2d, nlm_color_image, nlm_denoise_1d, nlm_denoise_2d, nlm_multiscale_2d,
    NlmConfig,
};

fn main() {
    println!("Non-Local Means Denoising Examples");
    println!("---------------------------------");

    // Create test signals and images
    println!("\nGenerating test data...");
    let (clean_signal, noisy_signal) = generate_test_signal();
    let (clean_image, noisy_image) = generate_test_image();
    let (clean_color_image, noisy_color_image) = generate_color_image();

    // 1D NLM denoising
    println!("\n1. 1D Signal Non-Local Means Denoising");

    // Apply NLM with different parameters
    let basic_config = NlmConfig::default();
    let denoised_signal = nlm_denoise_1d(&noisy_signal, &basic_config).unwrap();

    // Calculate SNR improvement
    let input_snr = calculate_snr(&clean_signal, &noisy_signal);
    let output_snr = calculate_snr(&clean_signal, &denoised_signal);

    println!("   Input SNR: {:.2} dB", input_snr);
    println!("   Output SNR: {:.2} dB", output_snr);
    println!("   Improvement: {:.2} dB", output_snr - input_snr);

    // Save results to CSV for plotting
    save_signal_to_csv(
        "nlm_1d_results.csv",
        &clean_signal,
        &noisy_signal,
        &denoised_signal,
    );
    println!("   Saved results to nlm_1d_results.csv");

    // 2D Image NLM denoising
    println!("\n2. 2D Image Non-Local Means Denoising");

    // Create configurations with different parameters
    let mut standard_config = NlmConfig::default();
    standard_config.patch_size = 5;
    standard_config.search_window = 17;
    standard_config.h = 0.4;

    let mut fast_config = standard_config.clone();
    fast_config.fast_mode = true;
    fast_config.step_size = 2;

    // Apply different NLM variants
    let denoised_standard = nlm_denoise_2d(&noisy_image, &standard_config).unwrap();
    let denoised_fast = nlm_denoise_2d(&noisy_image, &fast_config).unwrap();
    let denoised_block = nlm_block_matching_2d(&noisy_image, &standard_config, 16).unwrap();

    // Calculate image SNR for each method
    let image_input_snr = calculate_image_snr(&clean_image, &noisy_image);
    let standard_snr = calculate_image_snr(&clean_image, &denoised_standard);
    let fast_snr = calculate_image_snr(&clean_image, &denoised_fast);
    let block_snr = calculate_image_snr(&clean_image, &denoised_block);

    println!("   Image input SNR: {:.2} dB", image_input_snr);
    println!(
        "   Standard NLM SNR: {:.2} dB (gain: {:.2} dB)",
        standard_snr,
        standard_snr - image_input_snr
    );
    println!(
        "   Fast NLM SNR: {:.2} dB (gain: {:.2} dB)",
        fast_snr,
        fast_snr - image_input_snr
    );
    println!(
        "   Block-matching NLM SNR: {:.2} dB (gain: {:.2} dB)",
        block_snr,
        block_snr - image_input_snr
    );

    // Save images to CSV for visualization
    save_image_to_csv("nlm_original.csv", &clean_image);
    save_image_to_csv("nlm_noisy.csv", &noisy_image);
    save_image_to_csv("nlm_standard.csv", &denoised_standard);
    save_image_to_csv("nlm_fast.csv", &denoised_fast);
    save_image_to_csv("nlm_block.csv", &denoised_block);

    println!("   Saved image results to CSV files");

    // Multi-scale NLM denoising
    println!("\n3. Multi-scale Non-Local Means Denoising");

    let denoised_multiscale = nlm_multiscale_2d(&noisy_image, &standard_config, 2).unwrap();
    let multiscale_snr = calculate_image_snr(&clean_image, &denoised_multiscale);

    println!(
        "   Multi-scale NLM SNR: {:.2} dB (gain: {:.2} dB)",
        multiscale_snr,
        multiscale_snr - image_input_snr
    );

    save_image_to_csv("nlm_multiscale.csv", &denoised_multiscale);

    // Color image denoising
    println!("\n4. Color Image Non-Local Means Denoising");

    let denoised_color = nlm_color_image(&noisy_color_image, &standard_config).unwrap();
    let color_input_snr = calculate_color_image_snr(&clean_color_image, &noisy_color_image);
    let color_output_snr = calculate_color_image_snr(&clean_color_image, &denoised_color);

    println!("   Color image input SNR: {:.2} dB", color_input_snr);
    println!(
        "   Color image output SNR: {:.2} dB (gain: {:.2} dB)",
        color_output_snr,
        color_output_snr - color_input_snr
    );

    // Save color channels to CSV
    for c in 0..3 {
        let channel_name = match c {
            0 => "red",
            1 => "green",
            2 => "blue",
            _ => "unknown",
        };

        save_image_to_csv(
            &format!("nlm_color_clean_{}.csv", channel_name),
            &extract_color_channel(&clean_color_image, c),
        );

        save_image_to_csv(
            &format!("nlm_color_noisy_{}.csv", channel_name),
            &extract_color_channel(&noisy_color_image, c),
        );

        save_image_to_csv(
            &format!("nlm_color_denoised_{}.csv", channel_name),
            &extract_color_channel(&denoised_color, c),
        );
    }

    println!("   Saved color image results to CSV files");

    // Parameter sensitivity analysis
    println!("\n5. Parameter Sensitivity Analysis");

    // Test different patch sizes
    let patch_sizes = [3, 5, 7, 9];
    for &patch_size in &patch_sizes {
        let mut config = standard_config.clone();
        config.patch_size = patch_size;

        let denoised = nlm_denoise_2d(&noisy_image, &config).unwrap();
        let snr = calculate_image_snr(&clean_image, &denoised);

        println!("   Patch size {}: SNR = {:.2} dB", patch_size, snr);
    }

    // Test different filtering parameters (h)
    let h_values = [0.1, 0.2, 0.4, 0.8, 1.6];
    for &h in &h_values {
        let mut config = standard_config.clone();
        config.h = h;

        let denoised = nlm_denoise_2d(&noisy_image, &config).unwrap();
        let snr = calculate_image_snr(&clean_image, &denoised);

        println!("   Filtering parameter h = {}: SNR = {:.2} dB", h, snr);
    }

    println!("\nDone! All results saved to CSV files for visualization.");
}

/// Generates a test 1D signal with additive noise
fn generate_test_signal() -> (Array1<f64>, Array1<f64>) {
    // Create a piecewise signal with edges and smooth regions
    let n = 500;
    let mut clean_signal = Array1::zeros(n);

    // Add step functions
    for i in 0..n {
        if i < 100 {
            clean_signal[i] = 0.0;
        } else if i < 200 {
            clean_signal[i] = 1.0;
        } else if i < 300 {
            clean_signal[i] = 0.5;
        } else if i < 400 {
            clean_signal[i] = 0.8;
        } else {
            clean_signal[i] = 0.2;
        }
    }

    // Smooth transitions
    let smooth_width = 10;
    for edge in [100, 200, 300, 400] {
        if edge >= smooth_width && edge < n - smooth_width {
            for i in edge - smooth_width..edge + smooth_width {
                let x = (i as f64 - edge as f64) / smooth_width as f64;
                let weight = 0.5 * (1.0 + (x * std::f64::consts::PI).tanh());

                if i < edge {
                    let left_val = clean_signal[edge - smooth_width - 1];
                    let right_val = clean_signal[edge + smooth_width];
                    clean_signal[i] = left_val * (1.0 - weight) + right_val * weight;
                }
            }
        }
    }

    // Add noise
    let mut rng = rng();
    let noise_level = 0.15;
    let mut noisy_signal = clean_signal.clone();

    for i in 0..n {
        noisy_signal[i] += noise_level * rng.random_range(-1.0..1.0);
    }

    (clean_signal, noisy_signal)
}

/// Generates a test 2D image with additive noise
fn generate_test_image() -> (Array2<f64>, Array2<f64>) {
    // Create a test image with geometric shapes
    let size = 64;
    let mut clean_image = Array2::zeros((size, size));

    // Add shapes
    for i in 0..size {
        for j in 0..size {
            // Normalize coordinates to [-1, 1]
            let x = j as f64 / size as f64 * 2.0 - 1.0;
            let y = i as f64 / size as f64 * 2.0 - 1.0;

            // Circle
            if (x * x + y * y) < 0.4 {
                clean_image[[i, j]] = 1.0;
            }

            // Rectangle
            if x.abs() < 0.2 && y.abs() < 0.6 {
                clean_image[[i, j]] = 0.8;
            }

            // Cross
            if x.abs() < 0.1 || y.abs() < 0.1 {
                if x.abs() < 0.7 && y.abs() < 0.7 {
                    clean_image[[i, j]] = 0.6;
                }
            }

            // Background gradient
            clean_image[[i, j]] += 0.2 * x;
        }
    }

    // Normalize to [0, 1]
    let min_val = clean_image.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = clean_image.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if (max_val - min_val).abs() > 1e-10 {
        clean_image.mapv_inplace(|x| (x - min_val) / (max_val - min_val));
    }

    // Add noise
    let mut rng = rng();
    let noise_level = 0.1;
    let mut noisy_image = clean_image.clone();

    for i in 0..size {
        for j in 0..size {
            noisy_image[[i, j]] += noise_level * rng.random_range(-1.0..1.0);
        }
    }

    // Clip to [0, 1]
    noisy_image.mapv_inplace(|x| x.max(0.0).min(1.0));

    (clean_image, noisy_image)
}

/// Generates a test color image with additive noise
fn generate_color_image() -> (Array3<f64>, Array3<f64>) {
    // Create a color image with RGB channels
    let size = 64;
    let mut clean_image = Array3::zeros((size, size, 3));

    // Add color patterns
    for i in 0..size {
        for j in 0..size {
            // Normalize coordinates to [-1, 1]
            let x = j as f64 / size as f64 * 2.0 - 1.0;
            let y = i as f64 / size as f64 * 2.0 - 1.0;

            // Red channel: radial gradient
            clean_image[[i, j, 0]] = 0.5 * (1.0 - (x * x + y * y).sqrt().min(1.0));

            // Green channel: vertical stripes
            clean_image[[i, j, 1]] = if ((j as f64 / 8.0).floor() as i32) % 2 == 0 {
                0.8
            } else {
                0.2
            };

            // Blue channel: horizontal gradient
            clean_image[[i, j, 2]] = 0.5 + 0.5 * y;

            // Add a colored square in the middle
            if x.abs() < 0.4 && y.abs() < 0.4 {
                clean_image[[i, j, 0]] = 0.9; // Red
                clean_image[[i, j, 1]] = 0.7; // Green
                clean_image[[i, j, 2]] = 0.2; // Blue
            }
        }
    }

    // Add noise
    let mut rng = rng();
    let noise_level = 0.1;
    let mut noisy_image = clean_image.clone();

    for i in 0..size {
        for j in 0..size {
            for c in 0..3 {
                noisy_image[[i, j, c]] += noise_level * rng.random_range(-1.0..1.0);
            }
        }
    }

    // Clip to [0, 1]
    noisy_image.mapv_inplace(|x| x.max(0.0).min(1.0));

    (clean_image, noisy_image)
}

/// Helper function to extract a color channel from a color image
fn extract_color_channel(image: &Array3<f64>, channel: usize) -> Array2<f64> {
    let (height, width, _) = image.dim();
    let mut result = Array2::zeros((height, width));

    for i in 0..height {
        for j in 0..width {
            result[[i, j]] = image[[i, j, channel]];
        }
    }

    result
}

/// Calculates the Signal-to-Noise Ratio (SNR) in dB for 1D signals
fn calculate_snr(clean: &Array1<f64>, noisy: &Array1<f64>) -> f64 {
    if clean.len() != noisy.len() {
        return f64::NEG_INFINITY;
    }

    let mut signal_power = 0.0;
    let mut noise_power = 0.0;

    for i in 0..clean.len() {
        let diff = clean[i] - noisy[i];
        signal_power += clean[i] * clean[i];
        noise_power += diff * diff;
    }

    if noise_power < 1e-10 {
        return f64::INFINITY;
    }

    10.0 * (signal_power / noise_power).log10()
}

/// Calculates the Signal-to-Noise Ratio (SNR) in dB for 2D images
fn calculate_image_snr(clean: &Array2<f64>, noisy: &Array2<f64>) -> f64 {
    if clean.dim() != noisy.dim() {
        return f64::NEG_INFINITY;
    }

    let (height, width) = clean.dim();
    let mut signal_power = 0.0;
    let mut noise_power = 0.0;

    for i in 0..height {
        for j in 0..width {
            let diff = clean[[i, j]] - noisy[[i, j]];
            signal_power += clean[[i, j]] * clean[[i, j]];
            noise_power += diff * diff;
        }
    }

    if noise_power < 1e-10 {
        return f64::INFINITY;
    }

    10.0 * (signal_power / noise_power).log10()
}

/// Calculates the Signal-to-Noise Ratio (SNR) in dB for color images
fn calculate_color_image_snr(clean: &Array3<f64>, noisy: &Array3<f64>) -> f64 {
    if clean.dim() != noisy.dim() {
        return f64::NEG_INFINITY;
    }

    let (height, width, channels) = clean.dim();
    let mut signal_power = 0.0;
    let mut noise_power = 0.0;

    for i in 0..height {
        for j in 0..width {
            for c in 0..channels {
                let diff = clean[[i, j, c]] - noisy[[i, j, c]];
                signal_power += clean[[i, j, c]] * clean[[i, j, c]];
                noise_power += diff * diff;
            }
        }
    }

    if noise_power < 1e-10 {
        return f64::INFINITY;
    }

    10.0 * (signal_power / noise_power).log10()
}

/// Saves 1D signals to a CSV file for plotting
fn save_signal_to_csv(
    filename: &str,
    clean: &Array1<f64>,
    noisy: &Array1<f64>,
    denoised: &Array1<f64>,
) {
    let mut file = File::create(filename).expect("Failed to create file");

    // Write header
    writeln!(file, "index,clean,noisy,denoised").expect("Failed to write header");

    // Write data
    for i in 0..clean.len() {
        writeln!(file, "{},{},{},{}", i, clean[i], noisy[i], denoised[i])
            .expect("Failed to write data");
    }
}

/// Saves a 2D image to a CSV file for visualization
fn save_image_to_csv(filename: &str, image: &Array2<f64>) {
    let mut file = File::create(filename).expect("Failed to create file");

    let (height, width) = image.dim();

    // Write data in CSV format suitable for heatmap visualization
    for i in 0..height {
        for j in 0..width {
            if j > 0 {
                write!(file, ",").expect("Failed to write separator");
            }
            write!(file, "{:.6}", image[[i, j]]).expect("Failed to write data");
        }
        writeln!(file).expect("Failed to write newline");
    }
}
