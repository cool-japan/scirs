use ndarray::{Array2, ArrayView2};
use scirs2_signal::dwt::Wavelet;
use scirs2_signal::dwt2d::{dwt2d_decompose, dwt2d_reconstruct, wavedec2, waverec2, Dwt2dResult};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("2D Wavelet Transform Example");
    println!("===========================\n");

    // Create a sample 8x8 image
    println!("Creating a simple 8x8 test image (gradient pattern)");
    let mut image = Array2::zeros((8, 8));
    for i in 0..8 {
        for j in 0..8 {
            image[[i, j]] = (i * j) as f64;
        }
    }

    // Print the original image
    println!("\nOriginal Image:");
    print_image(image.view());

    // Perform a single-level 2D DWT using Haar wavelet
    println!("\nPerforming single-level 2D DWT using Haar wavelet...");
    let decomposition = dwt2d_decompose(&image, Wavelet::Haar, None)?;

    // Print the subbands
    println!("\nApproximation Coefficients (LL band):");
    print_image(decomposition.approx.view());

    println!("\nHorizontal Detail Coefficients (LH band):");
    print_image(decomposition.detail_h.view());

    println!("\nVertical Detail Coefficients (HL band):");
    print_image(decomposition.detail_v.view());

    println!("\nDiagonal Detail Coefficients (HH band):");
    print_image(decomposition.detail_d.view());

    // Reconstruct the image
    println!("\nReconstructing the image from wavelet coefficients...");
    let reconstructed = dwt2d_reconstruct(&decomposition, Wavelet::Haar, None)?;

    println!("\nReconstructed Image:");
    print_image(reconstructed.view());

    // Calculate reconstruction error
    let mut max_error = 0.0;
    for i in 0..8 {
        for j in 0..8 {
            let error = (image[[i, j]] - reconstructed[[i, j]]).abs();
            if error > max_error {
                max_error = error;
            }
        }
    }
    println!("\nMaximum reconstruction error: {:.2e}", max_error);

    // Now perform a multi-level decomposition (just one level to avoid overflow issues)
    println!("\n\nPerforming multi-level 2D DWT (1 level) using DB4 wavelet...");
    let coeffs = wavedec2(&image, Wavelet::DB(4), 1, None)?;

    println!("\nNumber of decomposition levels: {}", coeffs.len());

    // Print approximation coefficients at the lowest level
    println!("\nApproximation Coefficients at level 1:");
    print_image(coeffs[0].approx.view());

    // Reconstruct from multi-level decomposition
    println!("\nReconstructing from multi-level decomposition...");
    let multi_reconstructed = waverec2(&coeffs, Wavelet::DB(4), None)?;

    // Calculate multi-level reconstruction error
    let mut max_multi_error = 0.0;
    for i in 0..8 {
        for j in 0..8 {
            let error = (image[[i, j]] - multi_reconstructed[[i, j]]).abs();
            if error > max_multi_error {
                max_multi_error = error;
            }
        }
    }
    println!(
        "\nMaximum multi-level reconstruction error: {:.2e}",
        max_multi_error
    );

    // Demonstrate image compression by thresholding small coefficients
    println!("\n\nDemonstrating wavelet-based compression by zeroing small coefficients...");

    // Apply a simple threshold to all detail coefficients
    let mut thresholded_coeffs = coeffs.clone();
    let threshold = 1.0;

    for level_coeffs in &mut thresholded_coeffs {
        threshold_coefficients(level_coeffs, threshold);
    }

    // Count non-zero coefficients before and after thresholding
    let original_nonzero = count_nonzero_coefficients(&coeffs);
    let thresholded_nonzero = count_nonzero_coefficients(&thresholded_coeffs);

    println!(
        "\nOriginal coefficients: {} non-zero values",
        original_nonzero
    );
    println!(
        "After thresholding: {} non-zero values",
        thresholded_nonzero
    );
    println!(
        "Compression ratio: {:.2}x",
        original_nonzero as f64 / thresholded_nonzero as f64
    );

    // Reconstruct from thresholded coefficients
    println!("\nReconstructing from thresholded coefficients...");
    let compressed_image = waverec2(&thresholded_coeffs, Wavelet::DB(4), None)?;

    println!("\nCompressed Image:");
    print_image(compressed_image.view());

    // Calculate compression error
    let mut max_compression_error = 0.0;
    let mut mean_squared_error = 0.0;

    for i in 0..8 {
        for j in 0..8 {
            let error = (image[[i, j]] - compressed_image[[i, j]]).abs();
            if error > max_compression_error {
                max_compression_error = error;
            }
            mean_squared_error += error * error;
        }
    }

    mean_squared_error /= 64.0; // 8x8 = 64 pixels

    println!("\nCompression quality metrics:");
    println!("Maximum error: {:.2e}", max_compression_error);
    println!("Mean squared error: {:.2e}", mean_squared_error);
    println!("PSNR (dB): {:.2}", -10.0 * mean_squared_error.log10());

    println!("\nExample complete!");
    Ok(())
}

// Helper function to print a 2D array in a nicely formatted way
fn print_image(image: ArrayView2<f64>) {
    let (rows, cols) = image.dim();

    for i in 0..rows {
        for j in 0..cols {
            print!("{:6.1} ", image[[i, j]]);
        }
        println!();
    }
}

// Apply a threshold to the detail coefficients of a decomposition
fn threshold_coefficients(decomp: &mut Dwt2dResult, threshold: f64) {
    // Apply threshold to all detail coefficients
    for h in decomp.detail_h.iter_mut() {
        if h.abs() < threshold {
            *h = 0.0;
        }
    }

    for v in decomp.detail_v.iter_mut() {
        if v.abs() < threshold {
            *v = 0.0;
        }
    }

    for d in decomp.detail_d.iter_mut() {
        if d.abs() < threshold {
            *d = 0.0;
        }
    }
}

// Count non-zero coefficients in a decomposition
fn count_nonzero_coefficients(coeffs: &[Dwt2dResult]) -> usize {
    let mut count = 0;

    for decomp in coeffs {
        // Count non-zero values in approximation coefficients (only for the first level)
        if decomp == coeffs.first().unwrap_or(decomp) {
            for &val in decomp.approx.iter() {
                if val != 0.0 {
                    count += 1;
                }
            }
        }

        // Count non-zero values in detail coefficients
        for &val in decomp.detail_h.iter() {
            if val != 0.0 {
                count += 1;
            }
        }

        for &val in decomp.detail_v.iter() {
            if val != 0.0 {
                count += 1;
            }
        }

        for &val in decomp.detail_d.iter() {
            if val != 0.0 {
                count += 1;
            }
        }
    }

    count
}
