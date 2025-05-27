//! Denoising and enhancement examples

use image::{DynamicImage, ImageBuffer, Luma};
use ndarray::{Array2, Array3};
use scirs2_vision::error::Result;
use scirs2_vision::preprocessing::{
    adaptive_gamma_correction, auto_gamma_correction, gamma_correction, nlm_denoise,
    nlm_denoise_color, nlm_denoise_parallel,
};

fn main() -> Result<()> {
    // Load input image
    let img_path = "examples/input/input.jpg";
    let img = image::open(img_path).expect("Failed to load image");

    println!("Demonstrating denoising and enhancement techniques...");

    // 1. Non-local means denoising
    demonstrate_nlm_denoising(&img)?;

    // 2. Gamma correction for contrast enhancement
    demonstrate_gamma_correction(&img)?;

    println!("\nAll examples completed successfully!");
    Ok(())
}

fn demonstrate_nlm_denoising(img: &DynamicImage) -> Result<()> {
    println!("\n1. Non-Local Means Denoising:");

    // Add synthetic noise to the image
    let noisy_img = add_gaussian_noise(img, 0.05);
    noisy_img
        .save("examples/output/noisy_input.png")
        .expect("Failed to save noisy image");

    // Convert to grayscale for denoising
    let gray = noisy_img.to_luma8();
    let (width, height) = gray.dimensions();

    // Convert to ndarray
    let mut array = Array2::zeros((height as usize, width as usize));
    for y in 0..height {
        for x in 0..width {
            array[[y as usize, x as usize]] = gray.get_pixel(x, y)[0] as f32 / 255.0;
        }
    }

    // Apply NLM denoising with different parameters
    println!("  - Testing different h values (noise filtering strength)");
    let h_values = vec![0.05, 0.1, 0.15];

    for h in h_values {
        println!("    Denoising with h={}", h);
        let denoised = nlm_denoise(&array, h, 7, 21)?;

        // Convert back to image
        let mut output_img = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let val = (denoised[[y as usize, x as usize]] * 255.0).clamp(0.0, 255.0) as u8;
                output_img.put_pixel(x, y, Luma([val]));
            }
        }

        let output_path = format!("examples/output/nlm_denoised_h{}.png", h);
        output_img.save(&output_path).expect("Failed to save image");
        println!("      Saved: {}", output_path);
    }

    // Test parallel version
    println!("  - Testing parallel NLM denoising");
    let start = std::time::Instant::now();
    let _ = nlm_denoise(&array, 0.1, 7, 21)?;
    let serial_time = start.elapsed();

    let start = std::time::Instant::now();
    let denoised_parallel = nlm_denoise_parallel(&array, 0.1, 7, 21)?;
    let parallel_time = start.elapsed();

    println!(
        "    Serial time: {:?}, Parallel time: {:?}",
        serial_time, parallel_time
    );

    // Save parallel result
    let mut parallel_output = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let val = (denoised_parallel[[y as usize, x as usize]] * 255.0).clamp(0.0, 255.0) as u8;
            parallel_output.put_pixel(x, y, Luma([val]));
        }
    }
    parallel_output
        .save("examples/output/nlm_denoised_parallel.png")
        .expect("Failed to save parallel result");

    // Test color denoising if image is color
    if img.color() != image::ColorType::L8 {
        println!("  - Testing color NLM denoising");

        let rgb = noisy_img.to_rgb8();
        let mut color_array = Array3::zeros((height as usize, width as usize, 3));

        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x, y);
                for c in 0..3 {
                    color_array[[y as usize, x as usize, c]] = pixel[c] as f32 / 255.0;
                }
            }
        }

        let color_denoised = nlm_denoise_color(&color_array, 0.1, 7, 21)?;

        let mut color_output = image::RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let r =
                    (color_denoised[[y as usize, x as usize, 0]] * 255.0).clamp(0.0, 255.0) as u8;
                let g =
                    (color_denoised[[y as usize, x as usize, 1]] * 255.0).clamp(0.0, 255.0) as u8;
                let b =
                    (color_denoised[[y as usize, x as usize, 2]] * 255.0).clamp(0.0, 255.0) as u8;
                color_output.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }
        color_output
            .save("examples/output/nlm_denoised_color.png")
            .expect("Failed to save color denoised image");
    }

    Ok(())
}

fn demonstrate_gamma_correction(img: &DynamicImage) -> Result<()> {
    println!("\n2. Gamma Correction Examples:");

    // Basic gamma correction with different values
    println!("  - Testing different gamma values");
    let gamma_values = vec![0.5, 1.0, 1.5, 2.2];

    for gamma in gamma_values {
        let corrected = gamma_correction(img, gamma)?;
        let output_path = format!("examples/output/gamma_{}.png", gamma);

        match corrected {
            DynamicImage::ImageLuma8(gray) => gray.save(&output_path),
            DynamicImage::ImageRgb8(rgb) => rgb.save(&output_path),
            _ => corrected.save(&output_path),
        }
        .expect("Failed to save gamma corrected image");

        println!("    Gamma {}: saved to {}", gamma, output_path);
    }

    // Auto gamma correction
    println!("  - Testing automatic gamma correction");
    let target_brightnesses = vec![0.3, 0.5, 0.7];

    for target in target_brightnesses {
        let auto_corrected = auto_gamma_correction(img, target)?;
        let output_path = format!("examples/output/auto_gamma_target_{}.png", target);

        match auto_corrected {
            DynamicImage::ImageLuma8(gray) => gray.save(&output_path),
            DynamicImage::ImageRgb8(rgb) => rgb.save(&output_path),
            _ => auto_corrected.save(&output_path),
        }
        .expect("Failed to save auto gamma corrected image");

        println!("    Target brightness {}: saved to {}", target, output_path);
    }

    // Adaptive gamma correction
    println!("  - Testing adaptive gamma correction");
    let window_sizes = vec![5, 15, 25];

    for window_size in window_sizes {
        let adaptive = adaptive_gamma_correction(img, window_size, (0.5, 2.0))?;
        let output_path = format!("examples/output/adaptive_gamma_window_{}.png", window_size);

        match adaptive {
            DynamicImage::ImageLuma8(gray) => gray.save(&output_path),
            DynamicImage::ImageRgb8(rgb) => rgb.save(&output_path),
            _ => adaptive.save(&output_path),
        }
        .expect("Failed to save adaptive gamma corrected image");

        println!("    Window size {}: saved to {}", window_size, output_path);
    }

    Ok(())
}

/// Add Gaussian noise to an image
fn add_gaussian_noise(img: &DynamicImage, noise_level: f32) -> DynamicImage {
    use rand_distr::{Distribution, Normal};

    let mut rng = rand::rng();
    let normal = Normal::new(0.0, noise_level).unwrap();

    match img {
        DynamicImage::ImageLuma8(gray) => {
            let (width, height) = gray.dimensions();
            let mut noisy = ImageBuffer::new(width, height);

            for (x, y, pixel) in gray.enumerate_pixels() {
                let value = pixel[0] as f32 / 255.0;
                let noise: f32 = normal.sample(&mut rng);
                let noisy_value = (value + noise).clamp(0.0, 1.0);
                noisy.put_pixel(x, y, Luma([(noisy_value * 255.0) as u8]));
            }

            DynamicImage::ImageLuma8(noisy)
        }
        _ => {
            let rgb = img.to_rgb8();
            let (width, height) = rgb.dimensions();
            let mut noisy = image::RgbImage::new(width, height);

            for (x, y, pixel) in rgb.enumerate_pixels() {
                let mut new_pixel = [0u8; 3];
                for i in 0..3 {
                    let value = pixel[i] as f32 / 255.0;
                    let noise: f32 = normal.sample(&mut rng);
                    let noisy_value = (value + noise).clamp(0.0, 1.0);
                    new_pixel[i] = (noisy_value * 255.0) as u8;
                }
                noisy.put_pixel(x, y, image::Rgb(new_pixel));
            }

            DynamicImage::ImageRgb8(noisy)
        }
    }
}
