use scirs2_core::ndarray::Array2;
use scirs2_fft::{hfft2, ihfft2};
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() {
    // Create a simple 2D signal - a cosine pattern
    let n = 8;
    let mut signal = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let x = i as f64 / n as f64;
            let y = j as f64 / n as f64;
            signal[[i, j]] = (2.0 * PI * x).cos() * (4.0 * PI * y).cos();
        }
    }

    println!("Original real 2D signal:");
    for i in 0..n {
        for j in 0..n {
            print!("{:.3} ", signal[[i, j]]);
        }
        println!();
    }

    // Perform IHFFT2 to get complex spectrum with "backward" normalization
    let spectrum = ihfft2(&signal.view(), None, None, Some("backward")).unwrap();

    println!("\nComplex spectrum (Hermitian-symmetric):");
    println!("Shape: {:?}", spectrum.shape());

    // Try to convert back to real with HFFT2 using "backward" normalization
    match hfft2(&spectrum.view(), Some((n, n)), None, Some("backward")) {
        Ok(reconstructed) => {
            println!("\nReconstructed signal:");
            for i in 0..n {
                for j in 0..n {
                    print!("{:.3} ", reconstructed[[i, j]]);
                }
                println!();
            }

            // Calculate error
            let mut max_error: f64 = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let error = (reconstructed[[i, j]] - signal[[i, j]]).abs();
                    max_error = max_error.max(error);
                }
            }
            println!("\nMaximum reconstruction error: {max_error:.6e}");
        }
        Err(e) => {
            println!("\nError in reconstruction: {e:?}");
        }
    }
}
