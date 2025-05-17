use ndarray::{Array1, Array2, Array3};
// use std::f64::consts::PI;
use rand::Rng;
use std::fs::File;
use std::io::Write;

use scirs2_signal::tv::{
    tv_bregman_1d, tv_bregman_2d, tv_denoise_1d, tv_denoise_2d, tv_denoise_color, tv_inpaint,
    TvConfig, TvVariant,
};

fn main() {
    println!("Total Variation Denoising Examples");
    println!("---------------------------------");

    // Create test signals and images
    println!("\nGenerating test data...");
    let (clean_signal, noisy_signal) = generate_test_signal();
    let (clean_image, noisy_image) = generate_test_image();
    let (clean_color_image, noisy_color_image) = generate_color_image();

    // 1D Total Variation denoising
    println!("\n1. 1D Signal Total Variation Denoising");

    // Create configurations with different parameters
    let standard_config = TvConfig::default();

    let mut anisotropic_config = TvConfig::default();
    anisotropic_config.variant = TvVariant::Anisotropic;

    // Apply TV denoising with different parameters
    let weight = 0.5; // Regularization parameter

    let denoised_standard = tv_denoise_1d(&noisy_signal, weight, &standard_config).unwrap();
    let denoised_anisotropic = tv_denoise_1d(&noisy_signal, weight, &anisotropic_config).unwrap();

    // Calculate SNR improvement
    let input_snr = calculate_snr(&clean_signal, &noisy_signal);
    let standard_snr = calculate_snr(&clean_signal, &denoised_standard);
    let anisotropic_snr = calculate_snr(&clean_signal, &denoised_anisotropic);

    println!("   Input SNR: {:.2} dB", input_snr);
    println!(
        "   Isotropic TV SNR: {:.2} dB (gain: {:.2} dB)",
        standard_snr,
        standard_snr - input_snr
    );
    println!(
        "   Anisotropic TV SNR: {:.2} dB (gain: {:.2} dB)",
        anisotropic_snr,
        anisotropic_snr - input_snr
    );

    // Save results to CSV for plotting
    save_signal_to_csv(
        "tv_1d_results.csv",
        &clean_signal,
        &noisy_signal,
        &denoised_standard,
        Some(&denoised_anisotropic),
    );
    println!("   Saved results to tv_1d_results.csv");

    // 2D Image Total Variation denoising
    println!("\n2. 2D Image Total Variation Denoising");

    // Apply TV denoising to the noisy image
    let image_weight = 0.3;

    let denoised_image = tv_denoise_2d(&noisy_image, image_weight, &standard_config).unwrap();
    let denoised_aniso_image =
        tv_denoise_2d(&noisy_image, image_weight, &anisotropic_config).unwrap();

    // Calculate image SNR
    let image_input_snr = calculate_image_snr(&clean_image, &noisy_image);
    let image_standard_snr = calculate_image_snr(&clean_image, &denoised_image);
    let image_aniso_snr = calculate_image_snr(&clean_image, &denoised_aniso_image);

    println!("   Image input SNR: {:.2} dB", image_input_snr);
    println!(
        "   Isotropic TV SNR: {:.2} dB (gain: {:.2} dB)",
        image_standard_snr,
        image_standard_snr - image_input_snr
    );
    println!(
        "   Anisotropic TV SNR: {:.2} dB (gain: {:.2} dB)",
        image_aniso_snr,
        image_aniso_snr - image_input_snr
    );

    // Save images to CSV for visualization
    save_image_to_csv("tv_clean_image.csv", &clean_image);
    save_image_to_csv("tv_noisy_image.csv", &noisy_image);
    save_image_to_csv("tv_denoised_image.csv", &denoised_image);
    save_image_to_csv("tv_denoised_aniso_image.csv", &denoised_aniso_image);

    println!("   Saved image results to CSV files");

    // 3. Bregman iterations for improved detail preservation
    println!("\n3. TV with Bregman Iterations for Detail Preservation");

    // Apply TV with Bregman iterations
    let bregman_1d = tv_bregman_1d(&noisy_signal, weight, 3, &standard_config).unwrap();
    let bregman_2d = tv_bregman_2d(&noisy_image, image_weight, 3, &standard_config).unwrap();

    // Calculate SNR
    let bregman_1d_snr = calculate_snr(&clean_signal, &bregman_1d);
    let bregman_2d_snr = calculate_image_snr(&clean_image, &bregman_2d);

    println!(
        "   1D TV with Bregman SNR: {:.2} dB (gain: {:.2} dB)",
        bregman_1d_snr,
        bregman_1d_snr - input_snr
    );
    println!(
        "   2D TV with Bregman SNR: {:.2} dB (gain: {:.2} dB)",
        bregman_2d_snr,
        bregman_2d_snr - image_input_snr
    );

    // Save results
    save_image_to_csv("tv_bregman_image.csv", &bregman_2d);

    // Append Bregman 1D results to the 1D results file
    append_signal_to_csv("tv_1d_results.csv", "bregman", &bregman_1d);

    // 4. Color image TV denoising
    println!("\n4. Color Image Total Variation Denoising");

    // Apply TV to color image
    let denoised_color_sep =
        tv_denoise_color(&noisy_color_image, 0.2, &standard_config, false).unwrap();
    let denoised_color_vec =
        tv_denoise_color(&noisy_color_image, 0.2, &standard_config, true).unwrap();

    // Calculate SNR
    let color_input_snr = calculate_color_snr(&clean_color_image, &noisy_color_image);
    let color_sep_snr = calculate_color_snr(&clean_color_image, &denoised_color_sep);
    let color_vec_snr = calculate_color_snr(&clean_color_image, &denoised_color_vec);

    println!("   Color image input SNR: {:.2} dB", color_input_snr);
    println!(
        "   Channel-by-channel TV SNR: {:.2} dB (gain: {:.2} dB)",
        color_sep_snr,
        color_sep_snr - color_input_snr
    );
    println!(
        "   Vectorial TV SNR: {:.2} dB (gain: {:.2} dB)",
        color_vec_snr,
        color_vec_snr - color_input_snr
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
            &format!("tv_color_clean_{}.csv", channel_name),
            &extract_channel(&clean_color_image, c),
        );

        save_image_to_csv(
            &format!("tv_color_noisy_{}.csv", channel_name),
            &extract_channel(&noisy_color_image, c),
        );

        save_image_to_csv(
            &format!("tv_color_sep_{}.csv", channel_name),
            &extract_channel(&denoised_color_sep, c),
        );

        save_image_to_csv(
            &format!("tv_color_vec_{}.csv", channel_name),
            &extract_channel(&denoised_color_vec, c),
        );
    }

    println!("   Saved color image results to CSV files");

    // 5. TV inpainting for image restoration
    println!("\n5. TV Inpainting for Image Restoration");

    // Generate an image with missing data
    let (original_image, corrupted_image) = generate_inpainting_image();

    // Apply TV inpainting
    let inpainted_image = tv_inpaint(&corrupted_image, 0.1, &standard_config).unwrap();

    // Calculate recovery quality
    let inpaint_input_psnr = calculate_inpaint_psnr(&original_image, &corrupted_image);
    let inpaint_output_psnr = calculate_inpaint_psnr(&original_image, &inpainted_image);

    println!("   Corrupted image PSNR: {:.2} dB", inpaint_input_psnr);
    println!(
        "   Inpainted image PSNR: {:.2} dB (gain: {:.2} dB)",
        inpaint_output_psnr,
        inpaint_output_psnr - inpaint_input_psnr
    );

    // Save inpainting results
    save_image_to_csv("tv_original_image.csv", &original_image);
    save_image_to_csv(
        "tv_corrupted_image.csv",
        &corrupted_image_for_csv(&corrupted_image),
    );
    save_image_to_csv("tv_inpainted_image.csv", &inpainted_image);

    println!("   Saved inpainting results to CSV files");

    // 6. Parameter sensitivity analysis
    println!("\n6. Parameter Sensitivity Analysis");

    // Test different weights for TV denoising
    let weights = [0.1, 0.2, 0.5, 1.0, 2.0];

    println!("   Testing different regularization weights:");
    for &w in &weights {
        let denoised = tv_denoise_2d(&noisy_image, w, &standard_config).unwrap();
        let snr = calculate_image_snr(&clean_image, &denoised);

        println!("     Weight {:.1}: SNR = {:.2} dB", w, snr);

        save_image_to_csv(&format!("tv_weight_{:.1}.csv", w), &denoised);
    }

    println!("\nDone! All results saved to CSV files for visualization.");
}

/// Generates a test 1D signal with additive noise
fn generate_test_signal() -> (Array1<f64>, Array1<f64>) {
    // Create a piecewise signal with sharp edges
    let n = 500;
    let mut clean_signal = Array1::zeros(n);

    // Add step functions and ramps
    for i in 0..n {
        if i < 100 {
            clean_signal[i] = 0.0;
        } else if i < 200 {
            clean_signal[i] = 1.0;
        } else if i < 300 {
            clean_signal[i] = 0.5;
        } else if i < 400 {
            // Ramp
            clean_signal[i] = 0.5 + (i - 300) as f64 / 100.0 * 0.5;
        } else {
            clean_signal[i] = 0.0;
        }
    }

    // Add noise
    let mut rng = rand::rng();
    let noise_level = 0.15;
    let mut noisy_signal = clean_signal.clone();

    for i in 0..n {
        noisy_signal[i] += noise_level * rng.random_range(-1.0..1.0);
    }

    (clean_signal, noisy_signal)
}

/// Generates a test 2D image with additive noise
fn generate_test_image() -> (Array2<f64>, Array2<f64>) {
    // Create a test image with sharp edges and smooth regions
    let size = 64;
    let mut clean_image = Array2::zeros((size, size));

    // Add geometric shapes
    for i in 0..size {
        for j in 0..size {
            // Normalize coordinates to [-1, 1]
            let x = j as f64 / size as f64 * 2.0 - 1.0;
            let y = i as f64 / size as f64 * 2.0 - 1.0;

            // Circle
            if (x * x + y * y) < 0.3 {
                clean_image[[i, j]] = 1.0;
            }

            // Square
            if x.abs() < 0.5 && y.abs() < 0.5 && x.abs() > 0.3 && y.abs() > 0.3 {
                clean_image[[i, j]] = 0.8;
            }

            // Triangle
            if y > -0.7 && y < -0.3 && x.abs() < 0.7 && y < -0.3 + 0.5 * x.abs() {
                clean_image[[i, j]] = 0.6;
            }

            // Add a smooth gradient in the background
            if clean_image[[i, j]] < 0.1 {
                clean_image[[i, j]] = 0.2 * (x + y);
            }
        }
    }

    // Normalize to [0, 1]
    let min_val = clean_image.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = clean_image.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if (max_val - min_val).abs() > 1e-10 {
        clean_image.mapv_inplace(|x| (x - min_val) / (max_val - min_val));
    }

    // Add noise
    let mut rng = rand::rng();
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

            // Red channel: circle
            if (x * x + y * y) < 0.3 {
                clean_image[[i, j, 0]] = 0.9;
                clean_image[[i, j, 1]] = 0.2;
                clean_image[[i, j, 2]] = 0.2;
            }

            // Green channel: square
            if x.abs() < 0.5 && y.abs() < 0.5 && x.abs() > 0.3 && y.abs() > 0.3 {
                clean_image[[i, j, 0]] = 0.2;
                clean_image[[i, j, 1]] = 0.8;
                clean_image[[i, j, 2]] = 0.2;
            }

            // Blue channel: triangle
            if y > -0.7 && y < -0.3 && x.abs() < 0.7 && y < -0.3 + 0.5 * x.abs() {
                clean_image[[i, j, 0]] = 0.2;
                clean_image[[i, j, 1]] = 0.2;
                clean_image[[i, j, 2]] = 0.8;
            }

            // Add a smooth gradient in the background
            if clean_image[[i, j, 0]] < 0.1
                && clean_image[[i, j, 1]] < 0.1
                && clean_image[[i, j, 2]] < 0.1
            {
                clean_image[[i, j, 0]] = 0.3 + 0.1 * x;
                clean_image[[i, j, 1]] = 0.3 + 0.1 * y;
                clean_image[[i, j, 2]] = 0.3 - 0.1 * (x + y);
            }
        }
    }

    // Add noise
    let mut rng = rand::rng();
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

/// Generates a test image with missing pixels for inpainting
fn generate_inpainting_image() -> (Array2<f64>, Array2<f64>) {
    // Start with a clean image
    let (clean_image, _) = generate_test_image();
    let (height, width) = clean_image.dim();

    // Create a copy with missing pixels (represented as NaN)
    let mut corrupted_image = clean_image.clone();

    // Create a random mask for missing pixels
    let mut rng = rand::rng();
    let missing_ratio = 0.3; // 30% of pixels will be missing

    for i in 0..height {
        for j in 0..width {
            if rng.random_range(0.0..1.0) < missing_ratio {
                corrupted_image[[i, j]] = f64::NAN;
            }
        }
    }

    // Add some structured missing regions (e.g., lines or rectangles)
    let line_width = 3;

    // Horizontal line
    let line_y = height / 2;
    for i in line_y..(line_y + line_width).min(height) {
        for j in 0..width {
            corrupted_image[[i, j]] = f64::NAN;
        }
    }

    // Vertical line
    let line_x = width / 3;
    for i in 0..height {
        for j in line_x..(line_x + line_width).min(width) {
            corrupted_image[[i, j]] = f64::NAN;
        }
    }

    // Small rectangle
    let rect_x = 3 * width / 4;
    let rect_y = height / 4;
    let rect_size = 8;

    for i in rect_y..(rect_y + rect_size).min(height) {
        for j in rect_x..(rect_x + rect_size).min(width) {
            corrupted_image[[i, j]] = f64::NAN;
        }
    }

    (clean_image, corrupted_image)
}

/// Helper function to extract a channel from a color image
fn extract_channel(image: &Array3<f64>, channel: usize) -> Array2<f64> {
    let (height, width, _) = image.dim();
    let mut result = Array2::zeros((height, width));

    for i in 0..height {
        for j in 0..width {
            result[[i, j]] = image[[i, j, channel]];
        }
    }

    result
}

/// Helper function to convert NaN values to a fixed value for CSV export
fn corrupted_image_for_csv(image: &Array2<f64>) -> Array2<f64> {
    let (height, width) = image.dim();
    let mut result = Array2::zeros((height, width));

    for i in 0..height {
        for j in 0..width {
            if image[[i, j]].is_nan() {
                result[[i, j]] = -0.1; // Use -0.1 to represent missing values in CSV
            } else {
                result[[i, j]] = image[[i, j]];
            }
        }
    }

    result
}

/// Calculates the Signal-to-Noise Ratio (SNR) in dB for 1D signals
fn calculate_snr(clean: &Array1<f64>, noisy: &Array1<f64>) -> f64 {
    let mut signal_power = 0.0;
    let mut noise_power = 0.0;

    for i in 0..clean.len() {
        signal_power += clean[i] * clean[i];
        noise_power += (clean[i] - noisy[i]).powi(2);
    }

    if noise_power < 1e-10 {
        return f64::INFINITY;
    }

    10.0 * (signal_power / noise_power).log10()
}

/// Calculates the Signal-to-Noise Ratio (SNR) in dB for 2D images
fn calculate_image_snr(clean: &Array2<f64>, noisy: &Array2<f64>) -> f64 {
    let (height, width) = clean.dim();
    let mut signal_power = 0.0;
    let mut noise_power = 0.0;

    for i in 0..height {
        for j in 0..width {
            signal_power += clean[[i, j]] * clean[[i, j]];
            noise_power += (clean[[i, j]] - noisy[[i, j]]).powi(2);
        }
    }

    if noise_power < 1e-10 {
        return f64::INFINITY;
    }

    10.0 * (signal_power / noise_power).log10()
}

/// Calculates the Signal-to-Noise Ratio (SNR) in dB for color images
fn calculate_color_snr(clean: &Array3<f64>, noisy: &Array3<f64>) -> f64 {
    let (height, width, channels) = clean.dim();
    let mut signal_power = 0.0;
    let mut noise_power = 0.0;

    for i in 0..height {
        for j in 0..width {
            for c in 0..channels {
                signal_power += clean[[i, j, c]] * clean[[i, j, c]];
                noise_power += (clean[[i, j, c]] - noisy[[i, j, c]]).powi(2);
            }
        }
    }

    if noise_power < 1e-10 {
        return f64::INFINITY;
    }

    10.0 * (signal_power / noise_power).log10()
}

/// Calculates the Peak Signal-to-Noise Ratio (PSNR) in dB for images with missing data
fn calculate_inpaint_psnr(original: &Array2<f64>, inpainted: &Array2<f64>) -> f64 {
    let (height, width) = original.dim();
    let mut mse = 0.0;
    let mut count = 0;

    for i in 0..height {
        for j in 0..width {
            if !inpainted[[i, j]].is_nan() {
                mse += (original[[i, j]] - inpainted[[i, j]]).powi(2);
                count += 1;
            }
        }
    }

    if count == 0 || mse < 1e-10 {
        return f64::INFINITY;
    }

    mse /= count as f64;

    // PSNR with maximum value of 1.0
    10.0 * (1.0 / mse).log10()
}

/// Saves 1D signals to a CSV file for plotting
fn save_signal_to_csv(
    filename: &str,
    clean: &Array1<f64>,
    noisy: &Array1<f64>,
    denoised1: &Array1<f64>,
    denoised2: Option<&Array1<f64>>,
) {
    let mut file = File::create(filename).expect("Failed to create file");

    // Write header
    if let Some(_) = denoised2 {
        writeln!(file, "index,clean,noisy,isotropic,anisotropic").expect("Failed to write header");
    } else {
        writeln!(file, "index,clean,noisy,denoised").expect("Failed to write header");
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

/// Appends a signal column to an existing CSV file
fn append_signal_to_csv(filename: &str, column_name: &str, signal: &Array1<f64>) {
    // Read existing file
    let contents = std::fs::read_to_string(filename).expect("Failed to read file");
    let mut lines: Vec<String> = contents.lines().map(|s| s.to_string()).collect();

    // Update header
    let header = &mut lines[0];
    *header = format!("{},{}", header, column_name);

    // Update data rows
    for (i, line) in lines.iter_mut().enumerate().skip(1) {
        if i - 1 < signal.len() {
            *line = format!("{},{}", line, signal[i - 1]);
        }
    }

    // Write back to file
    let mut file = File::create(filename).expect("Failed to create file");
    for line in lines {
        writeln!(file, "{}", line).expect("Failed to write data");
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
