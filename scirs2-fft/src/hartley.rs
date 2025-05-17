//! Hartley Transform implementation
//!
//! The Hartley transform is a real-valued alternative to the Fourier transform.
//! It is related to the FFT by: H(f) = Re(FFT(f)) - Im(FFT(f))

use ndarray::{Array1, ArrayBase, Data, Dimension};
use num_complex::Complex64;

use crate::error::{FFTError, FFTResult};
use crate::fft::fft;

/// Compute the Discrete Hartley Transform (DHT) of a real-valued sequence.
///
/// The Hartley transform is defined as:
/// H\[k\] = sum_{n=0}^{N-1} x\[n\] * cas(2*pi*k*n/N)
///
/// where cas(x) = cos(x) + sin(x) = sqrt(2) * cos(x - pi/4)
///
/// # Arguments
///
/// * `x` - Input array (can be complex, but imaginary part is ignored)
///
/// # Returns
///
/// The Hartley transform of the input array.
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_fft::hartley::dht;
///
/// let x = array![1.0, 2.0, 3.0, 4.0];
/// let h = dht(&x).unwrap();
/// println!("Hartley transform: {:?}", h);
/// ```
pub fn dht<S, D>(x: &ArrayBase<S, D>) -> FFTResult<Array1<f64>>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    // Flatten input to 1D for processing
    let x_flat = x.iter().cloned().collect::<Vec<f64>>();
    let n = x_flat.len();

    if n == 0 {
        return Err(FFTError::ValueError("empty array".to_string()));
    }

    // Convert to complex array
    let mut x_complex = Vec::new();
    for &val in x_flat.iter() {
        x_complex.push(Complex64::new(val, 0.0));
    }

    // Compute FFT
    let fft_result = fft(&x_complex, None)?;

    // Compute Hartley transform: H[k] = Re(F[k]) - Im(F[k])
    let mut hartley = Array1::zeros(n);
    for i in 0..n {
        hartley[i] = fft_result[i].re - fft_result[i].im;
    }

    Ok(hartley)
}

/// Compute the inverse Discrete Hartley Transform (IDHT).
///
/// The inverse Hartley transform has the same form as the forward transform,
/// but with a scaling factor of 1/N.
///
/// # Arguments
///
/// * `h` - Input Hartley-transformed array
///
/// # Returns
///
/// The inverse Hartley transform of the input array.
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_fft::hartley::{dht, idht};
///
/// let x = array![1.0, 2.0, 3.0, 4.0];
/// let h = dht(&x).unwrap();
/// let x_recovered = idht(&h).unwrap();
/// assert!((x_recovered[0] - 1.0).abs() < 1e-10);
/// ```
pub fn idht<S>(h: &ArrayBase<S, ndarray::Ix1>) -> FFTResult<Array1<f64>>
where
    S: Data<Elem = f64>,
{
    let n = h.len();

    if n == 0 {
        return Err(FFTError::ValueError("empty array".to_string()));
    }

    // The Hartley transform is self-inverse up to a scaling factor
    let mut result = dht(h)?;

    // Apply scaling factor
    let scale = 1.0 / n as f64;
    result.map_inplace(|x| *x *= scale);

    Ok(result)
}

/// Compute the 2D Discrete Hartley Transform.
///
/// # Arguments
///
/// * `x` - Input 2D array
/// * `axes` - Axes along which to compute the transform (default: (0, 1))
///
/// # Returns
///
/// The 2D Hartley transform of the input array.
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_fft::hartley::dht2;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0]];
/// let h = dht2(&x, None).unwrap();
/// println!("2D Hartley transform: {:?}", h);
/// ```
pub fn dht2<S>(
    x: &ArrayBase<S, ndarray::Ix2>,
    axes: Option<(usize, usize)>,
) -> FFTResult<ndarray::Array2<f64>>
where
    S: Data<Elem = f64>,
{
    let axes = axes.unwrap_or((0, 1));
    let shape = x.shape();

    if axes.0 >= 2 || axes.1 >= 2 {
        return Err(FFTError::ValueError(format!(
            "Axes out of bounds: {:?}",
            axes
        )));
    }

    // Apply 1D Hartley transform along first axis
    let mut result = ndarray::Array2::zeros((shape[0], shape[1]));

    if axes.0 == 0 {
        // Transform along rows
        for j in 0..shape[1] {
            let column = x.slice(ndarray::s![.., j]);
            let transformed = dht(&column)?;
            for i in 0..shape[0] {
                result[[i, j]] = transformed[i];
            }
        }
    } else {
        // Transform along columns
        for i in 0..shape[0] {
            let row = x.slice(ndarray::s![i, ..]);
            let transformed = dht(&row)?;
            for j in 0..shape[1] {
                result[[i, j]] = transformed[j];
            }
        }
    }

    // Apply 1D Hartley transform along second axis
    let mut final_result = ndarray::Array2::zeros((shape[0], shape[1]));

    if axes.1 == 1 {
        // Transform along columns
        for i in 0..shape[0] {
            let row = result.slice(ndarray::s![i, ..]);
            let transformed = dht(&row)?;
            for j in 0..shape[1] {
                final_result[[i, j]] = transformed[j];
            }
        }
    } else {
        // Transform along rows
        for j in 0..shape[1] {
            let column = result.slice(ndarray::s![.., j]);
            let transformed = dht(&column)?;
            for i in 0..shape[0] {
                final_result[[i, j]] = transformed[i];
            }
        }
    }

    Ok(final_result)
}

/// Fast Hartley Transform using FFT
///
/// This is an optimized version that uses FFT directly for better performance.
pub fn fht<S, D>(x: &ArrayBase<S, D>) -> FFTResult<Array1<f64>>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    // This is an alias for dht, but could be optimized further in the future
    dht(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_hartley_transform() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let h = dht(&x).unwrap();

        // Test inverse
        let x_recovered = idht(&h).unwrap();
        for i in 0..x.len() {
            assert!(
                (x[i] - x_recovered[i]).abs() < 1e-10,
                "Failed at index {}: expected {}, got {}",
                i,
                x[i],
                x_recovered[i]
            );
        }
    }

    #[test]
    fn test_hartley_properties() {
        // Test that the Hartley transform of a real signal is real
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let h = dht(&x).unwrap();

        // All values should be real (they already are by construction)
        for &val in h.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_2d_hartley() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let h = dht2(&x, None).unwrap();

        // Test that the result has the same shape
        assert_eq!(h.shape(), x.shape());
    }

    #[test]
    fn test_empty_input() {
        let x: Array1<f64> = array![];
        let result = dht(&x);
        assert!(result.is_err());
    }
}
