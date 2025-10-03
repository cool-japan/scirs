use scirs2_core::ndarray::{s, Array2};
use scirs2_core::Complex64;
use scirs2_fft::{fft, fft_strided, fft_strided_complex, ifft_strided};
use std::time::Instant;

#[allow(dead_code)]
fn main() {
    println!("Advanced Strided FFT Example");
    println!("----------------------------");

    // Create a 2D test array of random values
    let rows = 256;
    let cols = 512;
    println!("Creating a {rows}x{cols} 2D array");

    let mut arr = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            arr[[i, j]] = (i * j) as f64 / 1000.0;
        }
    }

    println!("Array created successfully.");

    // Compare performance of standard vs strided FFT (along first axis)
    println!("\nPerforming FFT along first axis (axis 0):");

    // Standard FFT
    let start = Instant::now();
    let result_standard = perform_standard_fft_axis0(&arr);
    let standard_time = start.elapsed();
    println!("Standard FFT time: {standard_time:?}");

    // Strided FFT
    let start = Instant::now();
    let result_strided = fft_strided(&arr, 0).unwrap();
    let strided_time = start.elapsed();
    println!("Strided FFT time: {strided_time:?}");

    // Calculate maximum difference
    let max_diff = calculate_max_diff(&result_standard, &result_strided);
    println!("Maximum difference between approaches: {max_diff}");

    // Compare performance of standard vs strided FFT (along second axis)
    println!("\nPerforming FFT along second axis (axis 1):");

    // Standard FFT
    let start = Instant::now();
    let result_standard = perform_standard_fft_axis1(&arr);
    let standard_time = start.elapsed();
    println!("Standard FFT time: {standard_time:?}");

    // Strided FFT
    let start = Instant::now();
    let result_strided = fft_strided(&arr, 1).unwrap();
    let strided_time = start.elapsed();
    println!("Strided FFT time: {strided_time:?}");

    // Calculate maximum difference
    let max_diff = calculate_max_diff(&result_standard, &result_strided);
    println!("Maximum difference between approaches: {max_diff}");

    // Demonstrate round-trip accuracy
    println!("\nTesting round-trip accuracy (forward + inverse FFT):");

    // Create complex array
    let mut complex_arr = Array2::zeros((64, 64));
    for i in 0..64 {
        for j in 0..64 {
            complex_arr[[i, j]] = Complex64::new(i as f64, j as f64);
        }
    }

    // Forward FFT using strided implementation
    let fwd = fft_strided_complex(&complex_arr, 0).unwrap();

    // Inverse FFT using strided implementation
    let inv = ifft_strided(&fwd, 0).unwrap();

    // Calculate maximum difference
    let mut max_error: f64 = 0.0;
    for i in 0..64 {
        for j in 0..64 {
            let diff = (complex_arr[[i, j]] - inv[[i, j]]).norm();
            max_error = max_error.max(diff);
        }
    }

    println!("Maximum round-trip error: {max_error}");
}

// Implement standard FFT along axis 0 (for comparison)
#[allow(dead_code)]
fn perform_standard_fft_axis0(arr: &Array2<f64>) -> Array2<Complex64> {
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
    let mut result = Array2::zeros((rows, cols));

    // Process each column
    for j in 0..cols {
        let column: Vec<f64> = arr.slice(s![.., j]).to_vec();
        let fft_result = fft(&column, None).unwrap();

        // Copy back to result
        for i in 0..rows {
            result[[i, j]] = fft_result[i];
        }
    }

    result
}

// Implement standard FFT along axis 1 (for comparison)
#[allow(dead_code)]
fn perform_standard_fft_axis1(arr: &Array2<f64>) -> Array2<Complex64> {
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
    let mut result = Array2::zeros((rows, cols));

    // Process each row
    for i in 0..rows {
        let row: Vec<f64> = arr.slice(s![i, ..]).to_vec();
        let fft_result = fft(&row, None).unwrap();

        // Copy back to result
        for j in 0..cols {
            result[[i, j]] = fft_result[j];
        }
    }

    result
}

// Calculate maximum difference between two arrays
#[allow(dead_code)]
fn calculate_max_diff(a: &Array2<Complex64>, b: &Array2<Complex64>) -> f64 {
    let mut max_diff: f64 = 0.0;

    for i in 0..a.shape()[0] {
        for j in 0..a.shape()[1] {
            let diff = (a[[i, j]] - b[[i, j]]).norm();
            max_diff = max_diff.max(diff);
        }
    }

    max_diff
}
