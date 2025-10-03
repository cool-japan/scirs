//! Utility functions for SciRS2
//!
//! This module provides common utility functions used throughout the library.

// SCIRS2 POLICY: Use scirs2_core re-exports
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Dimension, Ix1, Ix2};
use scirs2_core::numeric::{Float, FloatConst, FromPrimitive, Zero};

use crate::error::{SciRS2Error, SciRS2Result, check_value};

/// Generate a linearly spaced array between start and stop
///
/// # Arguments
///
/// * `start` - The starting value of the sequence
/// * `stop` - The end value of the sequence
/// * `num` - Number of samples to generate
/// * `endpoint` - If true, stop is the last sample. Otherwise, it is not included
///
/// # Returns
///
/// A 1D array with linearly spaced values
///
/// # Examples
///
/// ```
/// use scirs2::utils::linspace;
/// let x = linspace(0.0, 1.0, 5, true);
/// assert_eq!(x.len(), 5);
/// assert!((x[0] - 0.0).abs() < 1e-10);
/// assert!((x[4] - 1.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn linspace<T>(start: T, stop: T, num: usize, endpoint: bool) -> Array1<T>
where
    T: Float + FromPrimitive + std::fmt::Debug,
{
    if num < 1 {
        return Array1::zeros(0);
    }
    
    if num == 1 {
        return Array1::from_elem(1, start);
    }

    let mut result = Array1::zeros(num);
    let div = if endpoint {
        T::from(num - 1).unwrap()
    } else {
        T::from(num).unwrap()
    };

    let step = (stop - start) / div;
    
    for (i, val) in result.iter_mut().enumerate() {
        *val = start + step * T::from(i).unwrap();
    }

    result
}

/// Generate an array with logarithmically spaced values between start and stop
///
/// # Arguments
///
/// * `start` - The starting value in log-space (10^start)
/// * `stop` - The end value in log-space (10^stop)
/// * `num` - Number of samples to generate
/// * `endpoint` - If true, stop is the last sample. Otherwise, it is not included
/// * `base` - The base of the logarithm
///
/// # Returns
///
/// A 1D array with logarithmically spaced values
///
/// # Examples
///
/// ```
/// use scirs2::utils::logspace;
/// let x = logspace(0.0, 3.0, 4, true, 10.0);
/// assert_eq!(x.len(), 4);
/// assert!((x[0] - 1.0).abs() < 1e-10);
/// assert!((x[3] - 1000.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn logspace<T>(start: T, stop: T, num: usize, endpoint: bool, base: T) -> SciRS2Result<Array1<T>>
where
    T: Float + FromPrimitive + std::fmt::Debug,
{
    check_value(base > T::zero(), "Base must be positive")?;
    
    let log_base = base.ln();
    let bases = linspace(start, stop, num, endpoint);
    
    Ok(bases.mapv(|x| (x * log_base).exp()))
}

/// Generate an array with values that are evenly spaced on a log scale
///
/// # Arguments
///
/// * `start` - The starting value
/// * `stop` - The end value
/// * `num` - Number of samples to generate
/// * `endpoint` - If true, stop is the last sample. Otherwise, it is not included
/// * `base` - The base of the logarithm
///
/// # Returns
///
/// A 1D array with values that are evenly spaced on a log scale
///
/// # Examples
///
/// ```
/// use scirs2::utils::geomspace;
/// let x = geomspace(1.0, 1000.0, 4, true, 10.0).unwrap();
/// assert_eq!(x.len(), 4);
/// assert!((x[0] - 1.0).abs() < 1e-10);
/// assert!((x[3] - 1000.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn geomspace<T>(start: T, stop: T, num: usize, endpoint: bool, base: T) -> SciRS2Result<Array1<T>>
where
    T: Float + FromPrimitive + std::fmt::Debug,
{
    check_value(start > T::zero() && stop > T::zero(), "Start and stop must be positive")?;
    check_value(base > T::zero(), "Base must be positive")?;
    
    let log_base = base.ln();
    let log_start = start.ln() / log_base;
    let log_stop = stop.ln() / log_base;
    
    let bases = linspace(log_start, log_stop, num, endpoint);
    
    Ok(bases.mapv(|x| (x * log_base).exp()))
}

/// Check if two arrays are approximately equal
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
/// * `rtol` - Relative tolerance
/// * `atol` - Absolute tolerance
///
/// # Returns
///
/// `true` if arrays are close, `false` otherwise
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2::utils::allclose;
///
/// let a = array![1.0, 2.0, 3.0];
/// let b = array![1.0, 2.0, 3.0];
/// assert!(allclose(&a.view(), &b.view(), 1e-8, 1e-8));
///
/// let c = array![1.0, 2.001, 3.0];
/// assert!(!allclose(&a.view(), &c.view(), 1e-8, 1e-8));
/// assert!(allclose(&a.view(), &c.view(), 1e-2, 1e-2));
/// ```
#[allow(dead_code)]
pub fn allclose<A, D, T>(a: &A, b: &A, rtol: T, atol: T) -> bool
where
    A: scirs2_core::ndarray::NdIndex<D>,
    D: Dimension,
    T: Float,
{
    if a.shape() != b.shape() {
        return false;
    }

    for (a_val, b_val) in a.iter().zip(b.iter()) {
        let diff = (*a_val - *b_val).abs();
        if diff > atol + rtol * b_val.abs() {
            return false;
        }
    }

    true
}

/// Get an array of evenly spaced angles in the interval [0, 2pi)
///
/// # Arguments
///
/// * `n` - Number of points
/// * `endpoint` - If true, the endpoint is excluded
///
/// # Returns
///
/// An array of evenly spaced angles
///
/// # Examples
///
/// ```
/// use scirs2::utils::meshgrid;
/// let x = vec![1.0, 2.0, 3.0];
/// let y = vec![4.0, 5.0];
/// let (xx, yy) = meshgrid(&x, &y);
/// assert_eq!(xx.shape(), &[2, 3]);
/// assert_eq!(yy.shape(), &[2, 3]);
/// assert_eq!(xx[[0, 0]], 1.0);
/// assert_eq!(xx[[0, 1]], 2.0);
/// assert_eq!(yy[[0, 0]], 4.0);
/// assert_eq!(yy[[1, 0]], 5.0);
/// ```
#[allow(dead_code)]
pub fn meshgrid<T: Float + Copy>(x: &[T], y: &[T]) -> (scirs2_core::ndarray::Array2<T>, scirs2_core::ndarray::Array2<T>) {
    let nx = x.len();
    let ny = y.len();

    let mut xx = scirs2_core::ndarray::Array2::<T>::zeros((ny, nx));
    let mut yy = scirs2_core::ndarray::Array2::<T>::zeros((ny, nx));
    
    for i in 0..ny {
        for j in 0..nx {
            xx[[i, j]] = x[j];
            yy[[i, j]] = y[i];
        }
    }
    
    (xx, yy)
}

/// Check if two values are approximately equal
///
/// # Arguments
///
/// * `a` - First value
/// * `b` - Second value
/// * `rtol` - Relative tolerance
/// * `atol` - Absolute tolerance
///
/// # Returns
///
/// * `true` if values are close, `false` otherwise
#[allow(dead_code)]
pub fn isclose<T: Float>(a: T, b: T, rtol: T, atol: T) -> bool {
    let diff = (a - b).abs();
    diff <= atol + rtol * b.abs()
}

/// Compute the Cartesian product of input arrays
///
/// # Arguments
///
/// * `arrays` - A list of arrays
///
/// # Returns
///
/// * Array with shape (n, len(arrays)) where n is the product of the input array lengths
#[allow(dead_code)]
pub fn cartesian_product<T: Clone + Copy>(arrays: &[&[T]]) -> Vec<Vec<T>> {
    if arrays.is_empty() {
        return vec![vec![]];
    }
    
    let mut result = vec![];
    let mut temp = vec![vec![arrays[0][0]]];
    
    for i in 0..arrays.len() {
        result = vec![];
        for t in temp.iter() {
            for &item in arrays[i].iter() {
                let mut new_tuple = t.clone();
                new_tuple.push(item);
                result.push(new_tuple);
            }
        }
        temp = result.clone();
    }
    
    result
}

/// Generate evenly spaced numbers like numpy's arange
///
/// # Arguments
///
/// * `start` - Start of the interval
/// * `stop` - End of the interval
/// * `step` - Step size
///
/// # Returns
///
/// * Array with evenly spaced values
#[allow(dead_code)]
pub fn arange<T: Float + FromPrimitive>(start: T, stop: T, step: T) -> Array1<T> {
    if start >= stop && step > T::zero() || start <= stop && step < T::zero() {
        return Array1::zeros(0);
    }
    
    let size = ((stop - start) / step).abs().floor().to_usize().unwrap_or(0);
    let mut result = Array1::zeros(size);
    
    for i in 0..size {
        result[i] = start + T::from(i).unwrap() * step;
    }
    
    result
}

/// Fill diagonal elements of a matrix with a given value
///
/// # Arguments
///
/// * `a` - Matrix to modify
/// * `val` - Value to put on the diagonal
///
/// # Returns
///
/// * Matrix with diagonal filled with val
#[allow(dead_code)]
pub fn fill_diagonal<T: Clone>(mut a: Array2<T>, val: T) -> Array2<T> {
    let min_dim = a.nrows().min(a.ncols());
    
    for i in 0..min_dim {
        a[[i, i]] = val.clone();
    }
    
    a
}

/// Check if a 2D array is symmetric
///
/// # Arguments
///
/// * `a` - Array to check
/// * `tol` - Tolerance for comparison
///
/// # Returns
///
/// * `true` if array is symmetric, `false` otherwise
#[allow(dead_code)]
pub fn is_symmetric<T: Float>(a: &ArrayView2<T>, tol: T) -> bool {
    if a.nrows() != a.ncols() {
        return false;
    }
    
    for i in 0..a.nrows() {
        for j in 0..i {
            if (a[[i, j]] - a[[j, i]]).abs() > tol {
                return false;
            }
        }
    }
    
    true
}
