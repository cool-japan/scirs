use ndarray::{Array1, Array2, Array3};
use rand::Rng;
use std::fs::File;
use std::io::Write;

use scirs2_signal::median::{
    hybrid_median_filter_2d, median_filter_1d, median_filter_2d, median_filter_color,
    rank_filter_1d, EdgeMode, MedianConfig,
};

fn main() {
    println!("Median Filtering Examples");
    println!("-------------------------");

    // Create test signals and images
    println!("\nGenerating test data...");
    let (clean_signal, noisy_signal) = generate_impulse_signal();
    let (clean_image, noisy_image) = generate_impulse_image();
    let (clean_color_image, noisy_color_image) = generate_color_impulse_image();

    // 1D Signal Median Filtering
    println!("\n1. 1D Signal Median Filtering");

    // Create configurations with different parameters
    let standard_config = MedianConfig::default();

    let mut adaptive_config = MedianConfig::default();
    adaptive_config.adaptive = true;
    adaptive_config.max_kernel_size = 9;

    let mut center_weighted_config = MedianConfig::default();
    center_weighted_config.center_weighted = true;
    center_weighted_config.center_weight = 3;

    // Apply median filtering with different settings
    let filtered_standard = median_filter_1d(&noisy_signal, 5, &standard_config).unwrap();
    let filtered_adaptive = median_filter_1d(&noisy_signal, 5, &adaptive_config).unwrap();
    let filtered_weighted = median_filter_1d(&noisy_signal, 5, &center_weighted_config).unwrap();

    // Calculate error metrics
    let mse_noisy = calculate_mse(&clean_signal, &noisy_signal);
    let mse_standard = calculate_mse(&clean_signal, &filtered_standard);
    let mse_adaptive = calculate_mse(&clean_signal, &filtered_adaptive);
    let mse_weighted = calculate_mse(&clean_signal, &filtered_weighted);

    println!("   Noisy signal MSE: {:.4}", mse_noisy);
    println!(
        "   Standard median filter MSE: {:.4} (improvement: {:.2}%)",
        mse_standard,
        100.0 * (mse_noisy - mse_standard) / mse_noisy
    );
    println!(
        "   Adaptive median filter MSE: {:.4} (improvement: {:.2}%)",
        mse_adaptive,
        100.0 * (mse_noisy - mse_adaptive) / mse_noisy
    );
    println!(
        "   Center-weighted median filter MSE: {:.4} (improvement: {:.2}%)",
        mse_weighted,
        100.0 * (mse_noisy - mse_weighted) / mse_noisy
    );

    // Save results to CSV for plotting
    save_signal_to_csv(
        "median_1d_results.csv",
        &clean_signal,
        &noisy_signal,
        &filtered_standard,
        Some(&filtered_adaptive),
        Some(&filtered_weighted),
    );
    println!("   Saved results to median_1d_results.csv");

    // 2. Rank Filtering
    println!("\n2. Rank Filtering for 1D Signals");

    // Apply rank filters with different ranks
    let min_filter = rank_filter_1d(&noisy_signal, 5, 0.0, EdgeMode::Reflect).unwrap();
    let median_filter = rank_filter_1d(&noisy_signal, 5, 0.5, EdgeMode::Reflect).unwrap();
    let max_filter = rank_filter_1d(&noisy_signal, 5, 1.0, EdgeMode::Reflect).unwrap();

    // Save rank filter results to CSV
    save_signal_to_csv(
        "rank_filter_results.csv",
        &clean_signal,
        &noisy_signal,
        &min_filter,
        Some(&median_filter),
        Some(&max_filter),
    );
    println!("   Saved rank filter results to rank_filter_results.csv");

    // 3. 2D Image Median Filtering
    println!("\n3. 2D Image Median Filtering");

    // Apply different filters to the noisy image
    let image_standard = median_filter_2d(&noisy_image, 3, &standard_config).unwrap();
    let image_adaptive = median_filter_2d(&noisy_image, 3, &adaptive_config).unwrap();
    let image_weighted = median_filter_2d(&noisy_image, 3, &center_weighted_config).unwrap();
    let image_hybrid = hybrid_median_filter_2d(&noisy_image, 5, &standard_config).unwrap();

    // Calculate image quality metrics
    let image_mse_noisy = calculate_image_mse(&clean_image, &noisy_image);
    let image_mse_standard = calculate_image_mse(&clean_image, &image_standard);
    let image_mse_adaptive = calculate_image_mse(&clean_image, &image_adaptive);
    let image_mse_weighted = calculate_image_mse(&clean_image, &image_weighted);
    let image_mse_hybrid = calculate_image_mse(&clean_image, &image_hybrid);

    println!("   Noisy image MSE: {:.4}", image_mse_noisy);
    println!(
        "   Standard median filter MSE: {:.4} (improvement: {:.2}%)",
        image_mse_standard,
        100.0 * (image_mse_noisy - image_mse_standard) / image_mse_noisy
    );
    println!(
        "   Adaptive median filter MSE: {:.4} (improvement: {:.2}%)",
        image_mse_adaptive,
        100.0 * (image_mse_noisy - image_mse_adaptive) / image_mse_noisy
    );
    println!(
        "   Center-weighted median filter MSE: {:.4} (improvement: {:.2}%)",
        image_mse_weighted,
        100.0 * (image_mse_noisy - image_mse_weighted) / image_mse_noisy
    );
    println!(
        "   Hybrid median filter MSE: {:.4} (improvement: {:.2}%)",
        image_mse_hybrid,
        100.0 * (image_mse_noisy - image_mse_hybrid) / image_mse_noisy
    );

    // Save image results to CSV files
    save_image_to_csv("median_clean_image.csv", &clean_image);
    save_image_to_csv("median_noisy_image.csv", &noisy_image);
    save_image_to_csv("median_standard_image.csv", &image_standard);
    save_image_to_csv("median_adaptive_image.csv", &image_adaptive);
    save_image_to_csv("median_weighted_image.csv", &image_weighted);
    save_image_to_csv("median_hybrid_image.csv", &image_hybrid);

    println!("   Saved image results to CSV files");

    // 4. Color Image Median Filtering
    println!("\n4. Color Image Median Filtering");

    // Apply median filtering to color image
    let color_channel =
        median_filter_color(&noisy_color_image, 3, &standard_config, false).unwrap();
    let color_vector = median_filter_color(&noisy_color_image, 3, &standard_config, true).unwrap();

    // Calculate color image quality metrics
    let color_mse_noisy = calculate_color_mse(&clean_color_image, &noisy_color_image);
    let color_mse_channel = calculate_color_mse(&clean_color_image, &color_channel);
    let color_mse_vector = calculate_color_mse(&clean_color_image, &color_vector);

    println!("   Noisy color image MSE: {:.4}", color_mse_noisy);
    println!(
        "   Channel-by-channel median filter MSE: {:.4} (improvement: {:.2}%)",
        color_mse_channel,
        100.0 * (color_mse_noisy - color_mse_channel) / color_mse_noisy
    );
    println!(
        "   Vector median filter MSE: {:.4} (improvement: {:.2}%)",
        color_mse_vector,
        100.0 * (color_mse_noisy - color_mse_vector) / color_mse_noisy
    );

    // Save color channels to CSV files
    for c in 0..3 {
        let channel_name = match c {
            0 => "red",
            1 => "green",
            2 => "blue",
            _ => "unknown",
        };

        save_image_to_csv(
            &format!("median_color_clean_{}.csv", channel_name),
            &extract_channel(&clean_color_image, c),
        );

        save_image_to_csv(
            &format!("median_color_noisy_{}.csv", channel_name),
            &extract_channel(&noisy_color_image, c),
        );

        save_image_to_csv(
            &format!("median_color_channel_{}.csv", channel_name),
            &extract_channel(&color_channel, c),
        );

        save_image_to_csv(
            &format!("median_color_vector_{}.csv", channel_name),
            &extract_channel(&color_vector, c),
        );
    }

    println!("   Saved color image results to CSV files");

    // 5. Edge Mode Comparison
    println!("\n5. Edge Mode Comparison");

    // Create configurations with different edge modes
    let mut reflect_config = MedianConfig::default();
    reflect_config.edge_mode = EdgeMode::Reflect;

    let mut nearest_config = MedianConfig::default();
    nearest_config.edge_mode = EdgeMode::Nearest;

    let mut constant_config = MedianConfig::default();
    constant_config.edge_mode = EdgeMode::Constant(0.0);

    let mut wrap_config = MedianConfig::default();
    wrap_config.edge_mode = EdgeMode::Wrap;

    // Apply median filtering with different edge modes
    let edge_reflect = median_filter_2d(&noisy_image, 3, &reflect_config).unwrap();
    let edge_nearest = median_filter_2d(&noisy_image, 3, &nearest_config).unwrap();
    let edge_constant = median_filter_2d(&noisy_image, 3, &constant_config).unwrap();
    let edge_wrap = median_filter_2d(&noisy_image, 3, &wrap_config).unwrap();

    // Save edge mode results to CSV files
    save_image_to_csv("median_edge_reflect.csv", &edge_reflect);
    save_image_to_csv("median_edge_nearest.csv", &edge_nearest);
    save_image_to_csv("median_edge_constant.csv", &edge_constant);
    save_image_to_csv("median_edge_wrap.csv", &edge_wrap);

    println!("   Saved edge mode comparison results to CSV files");

    // 6. Kernel Size Comparison
    println!("\n6. Kernel Size Comparison");

    // Apply median filtering with different kernel sizes
    let kernel_sizes = [3, 5, 7, 9];

    for &size in &kernel_sizes {
        let filtered = median_filter_2d(&noisy_image, size, &standard_config).unwrap();
        let mse = calculate_image_mse(&clean_image, &filtered);

        println!("   Kernel size {}: MSE = {:.4}", size, mse);
        save_image_to_csv(&format!("median_kernel_{}.csv", size), &filtered);
    }

    println!("   Saved kernel size comparison results to CSV files");

    println!("\nDone! All results saved to CSV files for visualization.");
}

/// Generates a test signal with impulse noise
fn generate_impulse_signal() -> (Array1<f64>, Array1<f64>) {
    // Create clean signal with step edges and smooth regions
    let n = 500;
    let mut clean_signal = Array1::zeros(n);

    // Create piecewise signal with some smooth regions
    for i in 0..n {
        if i < 100 {
            clean_signal[i] = 1.0;
        } else if i < 200 {
            clean_signal[i] = 2.0;
        } else if i < 300 {
            // Ramp
            clean_signal[i] = 2.0 + (i - 300) as f64 / 100.0 * 1.0;
        } else if i < 400 {
            // Sine wave
            clean_signal[i] = 3.0 + (i - 350) as f64 / 50.0 * std::f64::consts::PI;
            clean_signal[i] = 3.0 + (clean_signal[i].sin() + 1.0) / 2.0;
        } else {
            clean_signal[i] = 4.0;
        }
    }

    // Add impulse noise (salt and pepper)
    let mut rng = rand::rng();
    let mut noisy_signal = clean_signal.clone();

    // Salt and pepper rate
    let impulse_rate = 0.1;

    for i in 0..n {
        // Random value between 0 and 1
        let r = rng.random_range(0.0..1.0);

        if r < impulse_rate {
            // Add high impulse (salt)
            noisy_signal[i] = 10.0;
        } else if r < 2.0 * impulse_rate {
            // Add low impulse (pepper)
            noisy_signal[i] = -2.0;
        }
    }

    (clean_signal, noisy_signal)
}

/// Generates a test image with impulse noise
fn generate_impulse_image() -> (Array2<f64>, Array2<f64>) {
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

            // Square
            if x.abs() < 0.3 && y.abs() < 0.3 {
                clean_image[[i, j]] = 0.7;
            }

            // Cross
            if (x.abs() < 0.1 || y.abs() < 0.1) && x.abs() < 0.7 && y.abs() < 0.7 {
                clean_image[[i, j]] = 0.5;
            }

            // Add gradient in background
            if clean_image[[i, j]] < 0.01 {
                clean_image[[i, j]] = 0.3 * (x + y + 2.0) / 4.0;
            }
        }
    }

    // Add impulse noise (salt and pepper)
    let mut rng = rand::rng();
    let mut noisy_image = clean_image.clone();

    // Salt and pepper rate
    let impulse_rate = 0.1;

    for i in 0..size {
        for j in 0..size {
            // Random value between 0 and 1
            let r = rng.random_range(0.0..1.0);

            if r < impulse_rate {
                // Add high impulse (salt)
                noisy_image[[i, j]] = 1.0;
            } else if r < 2.0 * impulse_rate {
                // Add low impulse (pepper)
                noisy_image[[i, j]] = 0.0;
            }
        }
    }

    (clean_image, noisy_image)
}

/// Generates a test color image with impulse noise
fn generate_color_impulse_image() -> (Array3<f64>, Array3<f64>) {
    // Create a color image with RGB channels
    let size = 64;
    let mut clean_image = Array3::zeros((size, size, 3));

    // Add color patterns
    for i in 0..size {
        for j in 0..size {
            // Normalize coordinates to [-1, 1]
            let x = j as f64 / size as f64 * 2.0 - 1.0;
            let y = i as f64 / size as f64 * 2.0 - 1.0;

            // Red channel
            if (x * x + y * y) < 0.4 {
                clean_image[[i, j, 0]] = 0.9;
                clean_image[[i, j, 1]] = 0.2;
                clean_image[[i, j, 2]] = 0.2;
            }

            // Green channel
            if x.abs() < 0.3 && y.abs() < 0.3 {
                clean_image[[i, j, 0]] = 0.2;
                clean_image[[i, j, 1]] = 0.8;
                clean_image[[i, j, 2]] = 0.2;
            }

            // Blue channel
            if (x.abs() < 0.1 || y.abs() < 0.1) && x.abs() < 0.7 && y.abs() < 0.7 {
                clean_image[[i, j, 0]] = 0.3;
                clean_image[[i, j, 1]] = 0.3;
                clean_image[[i, j, 2]] = 0.9;
            }

            // Add gradient in background
            if clean_image[[i, j, 0]] < 0.01
                && clean_image[[i, j, 1]] < 0.01
                && clean_image[[i, j, 2]] < 0.01
            {
                clean_image[[i, j, 0]] = 0.3 * (x + 1.0) / 2.0;
                clean_image[[i, j, 1]] = 0.3 * (y + 1.0) / 2.0;
                clean_image[[i, j, 2]] = 0.3 * ((1.0 - x) + 1.0) / 2.0;
            }
        }
    }

    // Add impulse noise (salt and pepper)
    let mut rng = rand::rng();
    let mut noisy_image = clean_image.clone();

    // Salt and pepper rate
    let impulse_rate = 0.1;

    for i in 0..size {
        for j in 0..size {
            // Random value between 0 and 1
            let r = rng.random_range(0.0..1.0);

            if r < impulse_rate {
                // Add high impulse (salt) to a random channel
                let channel = rng.random_range(0..3);
                noisy_image[[i, j, channel]] = 1.0;
            } else if r < 2.0 * impulse_rate {
                // Add low impulse (pepper) to a random channel
                let channel = rng.random_range(0..3);
                noisy_image[[i, j, channel]] = 0.0;
            }
        }
    }

    (clean_image, noisy_image)
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

/// Calculates Mean Squared Error (MSE) between two signals
fn calculate_mse(signal1: &Array1<f64>, signal2: &Array1<f64>) -> f64 {
    if signal1.len() != signal2.len() {
        return f64::INFINITY;
    }

    let mut sum_squared_diff = 0.0;

    for i in 0..signal1.len() {
        let diff = signal1[i] - signal2[i];
        sum_squared_diff += diff * diff;
    }

    sum_squared_diff / signal1.len() as f64
}

/// Calculates Mean Squared Error (MSE) between two images
fn calculate_image_mse(image1: &Array2<f64>, image2: &Array2<f64>) -> f64 {
    if image1.dim() != image2.dim() {
        return f64::INFINITY;
    }

    let (height, width) = image1.dim();
    let mut sum_squared_diff = 0.0;

    for i in 0..height {
        for j in 0..width {
            let diff = image1[[i, j]] - image2[[i, j]];
            sum_squared_diff += diff * diff;
        }
    }

    sum_squared_diff / (height * width) as f64
}

/// Calculates Mean Squared Error (MSE) between two color images
fn calculate_color_mse(image1: &Array3<f64>, image2: &Array3<f64>) -> f64 {
    if image1.dim() != image2.dim() {
        return f64::INFINITY;
    }

    let (height, width, channels) = image1.dim();
    let mut sum_squared_diff = 0.0;

    for i in 0..height {
        for j in 0..width {
            for c in 0..channels {
                let diff = image1[[i, j, c]] - image2[[i, j, c]];
                sum_squared_diff += diff * diff;
            }
        }
    }

    sum_squared_diff / (height * width * channels) as f64
}

/// Saves 1D signals to a CSV file for plotting
fn save_signal_to_csv(
    filename: &str,
    clean: &Array1<f64>,
    noisy: &Array1<f64>,
    filtered1: &Array1<f64>,
    filtered2: Option<&Array1<f64>>,
    filtered3: Option<&Array1<f64>>,
) {
    let mut file = File::create(filename).expect("Failed to create file");

    // Write header
    if filtered3.is_some() {
        writeln!(file, "index,clean,noisy,standard,adaptive,weighted")
            .expect("Failed to write header");
    } else if filtered2.is_some() {
        writeln!(file, "index,clean,noisy,min,median,max").expect("Failed to write header");
    } else {
        writeln!(file, "index,clean,noisy,filtered").expect("Failed to write header");
    }

    // Write data
    for i in 0..clean.len() {
        write!(file, "{},{},{},{}", i, clean[i], noisy[i], filtered1[i])
            .expect("Failed to write data");

        if let Some(f2) = filtered2 {
            write!(file, ",{}", f2[i]).expect("Failed to write data");
        }

        if let Some(f3) = filtered3 {
            write!(file, ",{}", f3[i]).expect("Failed to write data");
        }

        writeln!(file).expect("Failed to write data");
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
