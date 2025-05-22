//! Example of using SIMD-accelerated N-dimensional FFT for volumetric data processing
//!
//! This example demonstrates the use of SIMD-accelerated ND FFT functions for
//! processing volumetric (3D) data, which is common in scientific applications
//! like medical imaging, fluid dynamics, and seismic data analysis.

use num_complex::Complex64;
use scirs2_fft::{fftn_adaptive, ifftn_adaptive, simd_support_available};
use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    println!("SIMD-accelerated N-dimensional FFT Volumetric Data Example");
    println!("-------------------------------------------------------");

    // Check for SIMD support
    let simd_available = simd_support_available();
    println!("SIMD support available: {}", simd_available);

    // Define volume dimensions
    let width = 64;
    let height = 64;
    let depth = 64;
    let shape = [width, height, depth];
    let total_voxels = width * height * depth;

    println!(
        "\nGenerating volumetric test data ({} x {} x {} = {} voxels)",
        width, height, depth, total_voxels
    );

    // Generate a test volumetric dataset with various frequency components
    let volume_data = generate_test_volume(width, height, depth);

    // Apply a low-pass filter in the frequency domain
    println!("\nApplying 3D low-pass filter in the frequency domain...");
    let start = Instant::now();
    let filtered_volume = frequency_domain_filter_3d(&volume_data, &shape, "lowpass");
    let processing_time = start.elapsed();
    println!("Processing time: {:?}", processing_time);

    // Measure the difference between original and filtered data
    let mut sum_diff = 0.0;
    for i in 0..total_voxels {
        sum_diff += (volume_data[i] - filtered_volume[i]).abs();
    }
    let mean_diff = sum_diff / total_voxels as f64;
    println!("Mean absolute difference after filtering: {:.6}", mean_diff);

    // Performance comparison
    println!("\nPerformance comparison for different volume sizes:");
    for &size in &[16, 32, 64] {
        let test_shape = [size, size, size];
        let test_total = size * size * size;
        let test_data = generate_test_volume(size, size, size);

        println!(
            "\nVolume size: {} x {} x {} = {} voxels",
            size, size, size, test_total
        );
        let start = Instant::now();
        let _filtered = frequency_domain_filter_3d(&test_data, &test_shape, "lowpass");
        let time = start.elapsed();
        println!("Processing time: {:?}", time);

        // Calculate operations per second metric
        let ops_per_sec = test_total as f64 / time.as_secs_f64();
        println!("Voxels processed per second: {:.2e}", ops_per_sec);
    }
}

/// Generate a test volume with various frequency components
fn generate_test_volume(width: usize, height: usize, depth: usize) -> Vec<f64> {
    let mut volume = vec![0.0; width * height * depth];

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let x_norm = x as f64 / width as f64;
                let y_norm = y as f64 / height as f64;
                let z_norm = z as f64 / depth as f64;

                // Create patterns with various frequency components
                // Low-frequency component
                let low_freq = (2.0 * PI * x_norm).sin()
                    * (2.0 * PI * y_norm).cos()
                    * (2.0 * PI * z_norm).sin();

                // Medium-frequency component
                let mid_freq = 0.5
                    * (6.0 * PI * x_norm).sin()
                    * (6.0 * PI * y_norm).cos()
                    * (6.0 * PI * z_norm).sin();

                // High-frequency component (noise)
                let high_freq = 0.2
                    * (12.0 * PI * x_norm).sin()
                    * (12.0 * PI * y_norm).cos()
                    * (12.0 * PI * z_norm).sin();

                // Combine the frequency components
                let idx = z * width * height + y * width + x;
                volume[idx] = low_freq + mid_freq + high_freq;
            }
        }
    }

    volume
}

/// Apply a frequency domain filter to volumetric (3D) data
fn frequency_domain_filter_3d(volume: &[f64], shape: &[usize], filter_type: &str) -> Vec<f64> {
    // Step 1: Compute the N-dimensional FFT of the volume
    let spectrum = fftn_adaptive(volume, shape, None, None).unwrap();

    // Step 2: Create a frequency domain filter
    let mut filter = vec![Complex64::new(0.0, 0.0); volume.len()];

    // Calculate the center of the frequency domain
    let center_x = shape[0] / 2;
    let center_y = shape[1] / 2;
    let center_z = shape[2] / 2;

    // Calculate maximum possible distance for normalization
    let max_distance = (center_x.pow(2) + center_y.pow(2) + center_z.pow(2)) as f64;

    // Fill in the filter values based on distance from center frequency
    for z in 0..shape[2] {
        for y in 0..shape[1] {
            for x in 0..shape[0] {
                // Calculate frequency coordinates (shifted to center)
                let freq_x = if x < center_x {
                    x as isize
                } else {
                    x as isize - shape[0] as isize
                };
                let freq_y = if y < center_y {
                    y as isize
                } else {
                    y as isize - shape[1] as isize
                };
                let freq_z = if z < center_z {
                    z as isize
                } else {
                    z as isize - shape[2] as isize
                };

                // Calculate normalized distance from center
                let distance =
                    ((freq_x.pow(2) + freq_y.pow(2) + freq_z.pow(2)) as f64) / max_distance;

                // Apply different filter types
                let filter_value = match filter_type {
                    "lowpass" => {
                        // Lowpass: keep low frequencies, attenuate high frequencies
                        let cutoff = 0.1;
                        if distance <= cutoff {
                            1.0
                        } else if distance <= cutoff * 2.0 {
                            // Smooth transition using cosine
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
                            // Smooth transition using cosine
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
                            // Smooth lower transition
                            0.5 * (1.0 - (PI * (distance - low_cutoff) / low_cutoff).cos())
                        } else if distance <= high_cutoff {
                            1.0
                        } else if distance <= high_cutoff * 2.0 {
                            // Smooth upper transition
                            0.5 * (1.0 + (PI * (distance - high_cutoff) / high_cutoff).cos())
                        } else {
                            0.0
                        }
                    }
                    _ => 1.0, // No filtering by default
                };

                // Set the filter value
                let idx = z * shape[0] * shape[1] + y * shape[0] + x;
                filter[idx] = Complex64::new(filter_value, 0.0);
            }
        }
    }

    // Step 3: Apply the filter in the frequency domain
    let filtered_spectrum: Vec<Complex64> = spectrum
        .iter()
        .zip(filter.iter())
        .map(|(&s, &f)| s * f)
        .collect();

    // Step 4: Compute the inverse N-dimensional FFT
    let filtered_volume_complex = ifftn_adaptive(&filtered_spectrum, shape, None, None).unwrap();

    // Step 5: Extract real part (the filtered volume)
    let filtered_volume: Vec<f64> = filtered_volume_complex.iter().map(|c| c.re).collect();

    filtered_volume
}

/// Visualization information (simulation for a real system)
#[allow(dead_code)]
fn simulate_visualization(volume: &[f64], shape: &[usize]) {
    println!("Visualizing volumetric data:");
    println!(
        "- Volume dimensions: {} x {} x {}",
        shape[0], shape[1], shape[2]
    );

    // Find data range
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    for &val in volume {
        min_val = min_val.min(val);
        max_val = max_val.max(val);
    }

    println!("- Data range: {:.6} to {:.6}", min_val, max_val);

    // Print a few data points for verification
    println!("- Sample data points:");
    for z in [0, shape[2] / 2, shape[2] - 1] {
        for y in [0, shape[1] / 2, shape[1] - 1] {
            for x in [0, shape[0] / 2, shape[0] - 1] {
                let idx = z * shape[0] * shape[1] + y * shape[0] + x;
                println!("  ({},{},{}) = {:.6}", x, y, z, volume[idx]);
            }
        }
    }

    println!(
        "In a real application, this data would be rendered using 3D visualization techniques"
    );
    println!("such as volume rendering, isosurfaces, or slice views.");
}
