//! Hermitian Symmetry Utilities
//!
//! This module contains utility functions for working with Hermitian-symmetric arrays.
//! Hermitian symmetry is a property where a complex array satisfies a[i] = conj(a[-i])
//! for all indices i. This property is important for algorithms like HFFT and IHFFT.

use ndarray::{Array, Array2, Dimension, IxDyn};
use num_complex::Complex64;
use std::ops::Not;

/// Enforce Hermitian symmetry on a 2D complex array.
///
/// This function modifies the input array to ensure it satisfies Hermitian symmetry properties.
/// Specifically, it ensures that array[i, j] = conj(array[rows-i, cols-j]) for key components
/// of the array.
///
/// # Arguments
///
/// * `array` - Mutable reference to a 2D complex array
pub fn enforce_hermitian_symmetry(array: &mut Array2<Complex64>) {
    let (rows, cols) = array.dim();

    // Make DC component real
    if rows > 0 && cols > 0 {
        array[[0, 0]] = Complex64::new(array[[0, 0]].re, 0.0);
    }

    // For a truly Hermitian-symmetric result, we need to:
    // 1. Ensure that array[i, j] = conj(array[rows-i, cols-j]) for all i, j
    // 2. Ensure that array[i, 0] and array[0, j] have special properties

    // Ensure first row and column elements satisfy Hermitian constraints
    if rows > 1 && cols > 1 {
        // First row, special case
        for j in 1..cols / 2 + (cols % 2).not() {
            let conj_val = array[[0, cols - j]].conj();
            array[[0, j]] = conj_val;
        }

        // First column, special case
        for i in 1..rows / 2 + (rows % 2).not() {
            let conj_val = array[[rows - i, 0]].conj();
            array[[i, 0]] = conj_val;
        }

        // Handle the corners and mid-points for even-sized dimensions
        if rows % 2 == 0 && rows > 0 {
            array[[rows / 2, 0]] = Complex64::new(array[[rows / 2, 0]].re, 0.0);
        }

        if cols % 2 == 0 && cols > 0 {
            array[[0, cols / 2]] = Complex64::new(array[[0, cols / 2]].re, 0.0);
        }
    }
}

/// Enforce Hermitian symmetry on an N-dimensional complex array.
///
/// This function is a generalized version of `enforce_hermitian_symmetry` for
/// N-dimensional arrays. It ensures that key symmetry properties are satisfied.
///
/// # Arguments
///
/// * `array` - Mutable reference to an N-dimensional complex array
pub fn enforce_hermitian_symmetry_nd(array: &mut Array<Complex64, IxDyn>) {
    let shape = array.shape().to_vec();
    let ndim = shape.len();

    if ndim == 0 || array.is_empty() {
        return;
    }

    // Make DC component real
    // Use direct array access instead of indices
    if let Some(slice) = array.as_slice_mut() {
        if !slice.is_empty() {
            slice[0] = Complex64::new(slice[0].re, 0.0);
        }
    }

    // For higher dimensions (1D+), handle special cases
    match ndim {
        1 => {
            // 1D case - simpler to handle directly
            if let Some(slice) = array.as_slice_mut() {
                let n = slice.len();

                // Make elements Hermitian-symmetric
                for i in 1..n / 2 + 1 {
                    if i < n && (n - i) < n {
                        let avg = (slice[i] + slice[n - i].conj()) * 0.5;
                        slice[i] = avg;
                        slice[n - i] = avg.conj();
                    }
                }

                // If n is even, make the Nyquist frequency component real
                if n % 2 == 0 && n >= 2 {
                    slice[n / 2] = Complex64::new(slice[n / 2].re, 0.0);
                }
            }
        }
        2 => {
            // Convert to 2D view for easier access
            if let Ok(mut array2) = array.clone().into_dimensionality::<ndarray::Ix2>() {
                // Apply 2D symmetry (same as in enforce_hermitian_symmetry)
                enforce_hermitian_symmetry(&mut array2);

                // Copy back to the original array
                let view2 = array2.view();
                let flat = view2.as_slice().unwrap();
                if let Some(target) = array.as_slice_mut() {
                    target.copy_from_slice(flat);
                }
            }
        }
        _ => {
            // For 3+ dimensions, we do a simplified enforcement
            // This doesn't handle all possible Hermitian symmetry properties
            // but covers the most important ones

            // For each 2D plane along the first two dimensions, apply 2D symmetry
            if let Ok(mut view) = array.view_mut().into_dimensionality::<ndarray::Ix3>() {
                let (dim1, dim2, _) = view.dim();

                for k in 0..view.dim().2 {
                    let mut slice = view.slice_mut(ndarray::s![.., .., k]);
                    let mut array2 = Array2::zeros((dim1, dim2));

                    // Copy data to a temporary 2D array
                    for i in 0..dim1 {
                        for j in 0..dim2 {
                            array2[[i, j]] = slice[[i, j]];
                        }
                    }

                    // Apply 2D symmetry
                    enforce_hermitian_symmetry(&mut array2);

                    // Copy back
                    for i in 0..dim1 {
                        for j in 0..dim2 {
                            slice[[i, j]] = array2[[i, j]];
                        }
                    }
                }
            }
        }
    }
}

/// Check if an array is approximately Hermitian-symmetric.
///
/// This function verifies whether a complex array satisfies Hermitian symmetry properties
/// within a specified tolerance.
///
/// # Arguments
///
/// * `array` - Reference to a complex array
/// * `tolerance` - Maximum allowed deviation from Hermitian symmetry (default: 1e-10)
///
/// # Returns
///
/// * `true` if the array is approximately Hermitian-symmetric, `false` otherwise
pub fn is_hermitian_symmetric<D>(array: &Array<Complex64, D>, tolerance: Option<f64>) -> bool
where
    D: Dimension,
{
    let tol = tolerance.unwrap_or(1e-10);
    let shape = array.shape();

    // For multi-dimensional arrays, check symmetry across all dimensions
    // This is a simplified version that focuses on key symmetry points

    // Check DC component is real (or close to it)
    if !shape.is_empty() && !array.is_empty() {
        let dc_val = &array.as_slice().unwrap()[0];
        if dc_val.im.abs() > tol {
            return false;
        }
    }

    // For a full check, we would need to verify that a[i] = conj(a[-i]) for all indices
    // This could be computationally expensive for large arrays
    // Here we check a sample of key points

    // Simple implementation for 1D arrays
    if shape.len() == 1 && shape[0] > 1 {
        let n = shape[0];
        let data = array.as_slice().unwrap();
        for i in 1..n / 2 + 1 {
            if i < n && (n - i) < n {
                let a = &data[i];
                let b = data[n - i].conj();

                if (a.re - b.re).abs() > tol || (a.im - b.im).abs() > tol {
                    return false;
                }
            }
        }
        return true;
    }

    // Simple implementation for 2D arrays
    if shape.len() == 2 {
        let (rows, cols) = (shape[0], shape[1]);

        // Check the array using direct indexing for 2D
        let array2 = array
            .to_owned()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();

        // Check first row
        for j in 1..cols / 2 + 1 {
            if j < cols && (cols - j) < cols {
                let a = &array2[[0, j]];
                let b = array2[[0, cols - j]].conj();

                if (a.re - b.re).abs() > tol || (a.im - b.im).abs() > tol {
                    return false;
                }
            }
        }

        // Check first column
        for i in 1..rows / 2 + 1 {
            if i < rows && (rows - i) < rows {
                let a = &array2[[i, 0]];
                let b = array2[[rows - i, 0]].conj();

                if (a.re - b.re).abs() > tol || (a.im - b.im).abs() > tol {
                    return false;
                }
            }
        }
    }

    // For higher dimensions, we would need a more sophisticated approach
    // This is a simplified check that may not catch all deviations from Hermitian symmetry
    true
}

/// Create a Hermitian-symmetric array from a real-valued amplitude spectrum.
///
/// This function builds a complex array with Hermitian symmetry, where the amplitudes
/// are specified by the input array and phases are generated to ensure symmetry.
///
/// # Arguments
///
/// * `amplitudes` - Real-valued amplitudes for the frequency components
/// * `randomize_phases` - Whether to use random phases (if true) or zero phases (if false)
///
/// # Returns
///
/// * A complex array with Hermitian symmetry
pub fn create_hermitian_symmetric_signal(
    amplitudes: &[f64],
    randomize_phases: bool,
) -> Vec<Complex64> {
    use rand::Rng;

    let n = amplitudes.len();
    let mut result = Vec::with_capacity(n);

    // DC component is always real
    result.push(Complex64::new(amplitudes[0], 0.0));

    let mut rng = rand::rng();

    // For each frequency component (excluding DC and Nyquist)
    for (_i, &amp) in amplitudes.iter().enumerate().skip(1).take(n / 2 - 1) {
        // Generate a phase (either random or zero)
        let phase = if randomize_phases {
            2.0 * std::f64::consts::PI * rng.random::<f64>()
        } else {
            0.0
        };

        // Create complex number with given amplitude and phase
        let value = Complex64::from_polar(amp, phase);
        result.push(value);

        // Add conjugate for negative frequency
        result.push(value.conj());
    }

    // If n is even, add Nyquist frequency component (must be real)
    if n % 2 == 0 && n > 0 {
        result.push(Complex64::new(amplitudes[n / 2], 0.0));
    }

    result
}
