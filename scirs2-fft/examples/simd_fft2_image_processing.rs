//! Example of using SIMD-accelerated 2D FFT for image processing
//!
//! This example demonstrates the use of SIMD-accelerated 2D FFT functions for
//! a simple image processing task: frequency domain filtering. The example
//! shows how to use optimized 2D FFT functions with adaptive dispatching,
//! which automatically selects the most efficient implementation based on
//! hardware capabilities.

use num_complex::Complex64;
use scirs2_fft::{fft2_adaptive, ifft2_adaptive, simd_support_available};
use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    println!("SIMD-accelerated 2D FFT Image Processing Example");
    println!("------------------------------------------------");

    // Check for SIMD support
    let simd_available = simd_support_available();
    println!("SIMD support available: {}", simd_available);

    // Image dimensions (small test image)
    let width = 128;
    let height = 128;
    println!("\nGenerating test image ({} x {})", width, height);

    // Generate a test image with various frequency components
    let test_image = generate_test_image(width, height);

    // Apply frequency domain filtering
    println!("\nApplying frequency domain filtering with a lowpass filter...");
    let start = Instant::now();
    let _filtered_image = frequency_domain_filter(&test_image, width, height, "lowpass");
    let lowpass_time = start.elapsed();
    println!("Lowpass filter processing time: {:?}", lowpass_time);

    // Apply frequency domain filtering with a highpass filter
    println!("\nApplying frequency domain filtering with a highpass filter...");
    let start = Instant::now();
    let _filtered_image = frequency_domain_filter(&test_image, width, height, "highpass");
    let highpass_time = start.elapsed();
    println!("Highpass filter processing time: {:?}", highpass_time);

    // Apply frequency domain filtering with a bandpass filter
    println!("\nApplying frequency domain filtering with a bandpass filter...");
    let start = Instant::now();
    let _filtered_image = frequency_domain_filter(&test_image, width, height, "bandpass");
    let bandpass_time = start.elapsed();
    println!("Bandpass filter processing time: {:?}", bandpass_time);

    // Performance comparison with larger images
    println!("\nPerformance comparison with larger images:");

    for &size in &[256, 512] {
        println!("\nGenerating test image ({} x {})", size, size);
        let large_image = generate_test_image(size, size);

        // Standard processing
        let start = Instant::now();
        let _filtered = frequency_domain_filter(&large_image, size, size, "lowpass");
        let time = start.elapsed();
        println!("Processing time for {} x {} image: {:?}", size, size, time);

        // Run another filter type to show performance consistency
        let start = Instant::now();
        let _filtered = frequency_domain_filter(&large_image, size, size, "highpass");
        let highpass_time = start.elapsed();
        println!(
            "Highpass filter time for {} x {} image: {:?}",
            size, size, highpass_time
        );
    }
}

/// Generate a test image with various frequency components
fn generate_test_image(width: usize, height: usize) -> Vec<f64> {
    let mut image = vec![0.0; width * height];

    for y in 0..height {
        for x in 0..width {
            let x_norm = x as f64 / width as f64;
            let y_norm = y as f64 / height as f64;

            // Create a pattern with various frequency components
            let low_freq = (2.0 * PI * x_norm).sin() * (2.0 * PI * y_norm).cos();
            let mid_freq = 0.5 * (8.0 * PI * x_norm).sin() * (8.0 * PI * y_norm).cos();
            let high_freq = 0.25 * (16.0 * PI * x_norm).sin() * (16.0 * PI * y_norm).cos();

            // Combine the frequencies
            image[y * width + x] = low_freq + mid_freq + high_freq;
        }
    }

    image
}

/// Apply a frequency domain filter to an image
fn frequency_domain_filter(
    image: &[f64],
    width: usize,
    height: usize,
    filter_type: &str,
) -> Vec<f64> {
    // Step 1: Compute the 2D FFT of the image
    let spectrum = fft2_adaptive(image, [height, width], None, None).unwrap();

    // Step 2: Create a frequency domain filter
    let mut filter = vec![Complex64::new(0.0, 0.0); width * height];

    // Calculate filter mask
    let center_x = width / 2;
    let center_y = height / 2;
    let max_distance = (center_x.pow(2) + center_y.pow(2)) as f64;

    for y in 0..height {
        for x in 0..width {
            // Calculate frequency coordinates (shifted to center)
            let freq_x = if x < center_x { x } else { x - width };
            let freq_y = if y < center_y { y } else { y - height };

            // Calculate normalized distance from center
            let distance = ((freq_x.pow(2) + freq_y.pow(2)) as f64) / max_distance;

            // Apply different filter types
            let filter_value = match filter_type {
                "lowpass" => {
                    // Lowpass: keep low frequencies, attenuate high frequencies
                    let cutoff = 0.1;
                    if distance <= cutoff {
                        1.0
                    } else if distance <= cutoff * 2.0 {
                        0.5 * (1.0 + (PI * (distance - cutoff) / cutoff).cos())
                    } else {
                        0.0
                    }
                }
                "highpass" => {
                    // Highpass: attenuate low frequencies, keep high frequencies
                    let cutoff = 0.1;
                    if distance <= cutoff {
                        0.0
                    } else if distance <= cutoff * 2.0 {
                        0.5 * (1.0 - (PI * (distance - cutoff) / cutoff).cos())
                    } else {
                        1.0
                    }
                }
                "bandpass" => {
                    // Bandpass: keep middle frequencies
                    let low_cutoff = 0.1;
                    let high_cutoff = 0.3;
                    if distance <= low_cutoff {
                        0.0
                    } else if distance <= low_cutoff * 2.0 {
                        0.5 * (1.0 - (PI * (distance - low_cutoff) / low_cutoff).cos())
                    } else if distance <= high_cutoff {
                        1.0
                    } else if distance <= high_cutoff * 2.0 {
                        0.5 * (1.0 + (PI * (distance - high_cutoff) / high_cutoff).cos())
                    } else {
                        0.0
                    }
                }
                _ => 1.0, // No filtering
            };

            filter[y * width + x] = Complex64::new(filter_value, 0.0);
        }
    }

    // Step 3: Apply the filter in the frequency domain
    let filtered_spectrum: Vec<Complex64> = spectrum
        .iter()
        .zip(filter.iter())
        .map(|(&s, &f)| s * f)
        .collect();

    // Step 4: Compute the inverse 2D FFT
    let filtered_image_complex =
        ifft2_adaptive(&filtered_spectrum, [height, width], None, None).unwrap();

    // Step 5: Extract real part (the filtered image)
    let filtered_image: Vec<f64> = filtered_image_complex.iter().map(|c| c.re).collect();

    filtered_image
}
