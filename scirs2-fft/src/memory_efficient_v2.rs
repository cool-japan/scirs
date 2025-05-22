//! Memory-efficient FFT operations with minimal allocations
//!
//! This module provides memory-efficient implementations of FFT operations
//! that minimize temporary allocations and reuse buffers when possible.

use crate::error::{FFTError, FFTResult};
use crate::fft::NormMode;
use ndarray::{Array, Array1, Array2, ArrayD, Dimension, IxDyn, ShapeBuilder};
use num_complex::Complex64;
use num_traits::NumCast;
use rustfft::{num_complex::Complex as RustComplex, FftPlanner};
use std::fmt::Debug;
use std::sync::Arc;

// Thread-local buffer cache for reusing allocated memory
thread_local! {
    static BUFFER_CACHE: std::cell::RefCell<Option<Vec<RustComplex<f64>>>> = std::cell::RefCell::new(None);
}

/// Get a buffer from the thread-local cache or create a new one
fn get_or_create_buffer(size: usize) -> Vec<RustComplex<f64>> {
    BUFFER_CACHE.with(|cache| {
        let mut cache_ref = cache.borrow_mut();
        if let Some(buffer) = cache_ref.take() {
            if buffer.capacity() >= size {
                // Reuse existing buffer
                let mut buffer = buffer;
                buffer.resize(size, RustComplex::new(0.0, 0.0));
                return buffer;
            }
        }
        // Create new buffer
        Vec::with_capacity(size)
    })
}

/// Return a buffer to the thread-local cache for future reuse
fn return_buffer_to_cache(buffer: Vec<RustComplex<f64>>) {
    BUFFER_CACHE.with(|cache| {
        *cache.borrow_mut() = Some(buffer);
    });
}

/// Convert a value to Complex64 with minimal allocations
fn to_complex_value<T>(val: T) -> FFTResult<Complex64>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Handle Complex64 type directly
    if let Some(complex) = try_as_complex(&val) {
        return Ok(complex);
    }
    
    // Handle real value
    let real = num_traits::cast::<T, f64>(val)
        .ok_or_else(|| FFTError::ValueError(format!("Could not convert {:?} to f64", val)))?;
    
    Ok(Complex64::new(real, 0.0))
}

/// Try to convert a value to Complex64
fn try_as_complex<T: 'static>(val: &T) -> Option<Complex64> {
    use std::any::Any;
    
    // Try direct cast
    if let Some(complex) = (val as &dyn Any).downcast_ref::<Complex64>() {
        return Some(*complex);
    }
    
    // Try f32 complex
    if let Some(complex) = (val as &dyn Any).downcast_ref::<num_complex::Complex<f32>>() {
        return Some(Complex64::new(complex.re as f64, complex.im as f64));
    }
    
    // Try rustfft complex types
    if let Some(complex) = (val as &dyn Any).downcast_ref::<RustComplex<f64>>() {
        return Some(Complex64::new(complex.re, complex.im));
    }
    
    if let Some(complex) = (val as &dyn Any).downcast_ref::<RustComplex<f32>>() {
        return Some(Complex64::new(complex.re as f64, complex.im as f64));
    }
    
    None
}

/// Compute the 1-dimensional Fast Fourier Transform with optimized memory usage
///
/// This version minimizes memory allocations by reusing internal buffers when possible.
///
/// # Arguments
///
/// * `input` - Input data array
/// * `n` - Length of the output (optional)
/// * `norm` - Normalization mode (optional)
/// * `out` - Optional pre-allocated output buffer
///
/// # Returns
///
/// A vector of complex values representing the FFT result
pub fn fft_optimized<T>(
    input: &[T],
    n: Option<usize>,
    norm: Option<NormMode>,
    out: Option<&mut Vec<Complex64>>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Input validation
    if input.is_empty() {
        return Err(FFTError::ValueError("Input cannot be empty".to_string()));
    }

    // Determine the FFT size
    let input_len = input.len();
    let fft_size = n.unwrap_or_else(|| input_len.next_power_of_two());
    
    // Get the normalization mode
    let norm_mode = norm.unwrap_or(NormMode::None);
    
    // Get or create the buffer
    let mut buffer = get_or_create_buffer(fft_size);
    buffer.resize(fft_size, RustComplex::new(0.0, 0.0));
    
    // Convert input to complex numbers with minimal allocations
    for (i, val) in input.iter().enumerate() {
        if i < fft_size {
            let complex = to_complex_value(*val)?;
            buffer[i] = RustComplex::new(complex.re, complex.im);
        }
    }
    
    // Use rustfft library for computation with planner caching
    static PLANNER_CACHE: std::sync::OnceLock<std::sync::Mutex<FftPlanner<f64>>> = std::sync::OnceLock::new();
    let planner = PLANNER_CACHE.get_or_init(|| std::sync::Mutex::new(FftPlanner::new()));
    
    let fft_plan = {
        let mut planner = planner.lock().unwrap();
        planner.plan_fft_forward(fft_size)
    };
    
    // Perform FFT in-place
    fft_plan.process(&mut buffer);
    
    // Apply normalization if needed
    if norm_mode != NormMode::None {
        let scale = match norm_mode {
            NormMode::Forward => 1.0 / (fft_size as f64),
            NormMode::Backward => 1.0,  // Not applied for forward FFT
            NormMode::Ortho => 1.0 / (fft_size as f64).sqrt(),
            NormMode::None => 1.0,  // Never happens due to check above
        };
        
        if scale != 1.0 {
            buffer.iter_mut().for_each(|c| {
                c.re *= scale;
                c.im *= scale;
            });
        }
    }
    
    // Prepare output
    let result = if let Some(output_vec) = out {
        // Reuse provided buffer
        output_vec.resize(fft_size, Complex64::new(0.0, 0.0));
        for (i, c) in buffer.iter().enumerate() {
            output_vec[i] = Complex64::new(c.re, c.im);
        }
        std::mem::take(output_vec)
    } else {
        // Create new output vector with minimal allocations
        let mut result = Vec::with_capacity(fft_size);
        unsafe {
            result.set_len(fft_size);
        }
        
        for (i, c) in buffer.iter().enumerate() {
            result[i] = Complex64::new(c.re, c.im);
        }
        result
    };
    
    // Return buffer to cache
    return_buffer_to_cache(buffer);
    
    Ok(result)
}

/// Compute the inverse 1-dimensional Fast Fourier Transform with optimized memory usage
///
/// This version minimizes memory allocations by reusing internal buffers when possible.
///
/// # Arguments
///
/// * `input` - Input complex data array
/// * `n` - Length of the output (optional)
/// * `norm` - Normalization mode (optional)
/// * `out` - Optional pre-allocated output buffer
///
/// # Returns
///
/// A vector of complex values representing the inverse FFT result
pub fn ifft_optimized<T>(
    input: &[T],
    n: Option<usize>,
    norm: Option<NormMode>,
    out: Option<&mut Vec<Complex64>>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Input validation
    if input.is_empty() {
        return Err(FFTError::ValueError("Input cannot be empty".to_string()));
    }

    // Determine the FFT size
    let input_len = input.len();
    let fft_size = n.unwrap_or_else(|| input_len.next_power_of_two());
    
    // Get the normalization mode
    let norm_mode = norm.unwrap_or(NormMode::Backward);
    
    // Get or create the buffer
    let mut buffer = get_or_create_buffer(fft_size);
    buffer.resize(fft_size, RustComplex::new(0.0, 0.0));
    
    // Convert input to complex numbers with minimal allocations
    for (i, val) in input.iter().enumerate() {
        if i < fft_size {
            let complex = to_complex_value(*val)?;
            buffer[i] = RustComplex::new(complex.re, complex.im);
        }
    }
    
    // Use rustfft library for computation with planner caching
    static PLANNER_CACHE: std::sync::OnceLock<std::sync::Mutex<FftPlanner<f64>>> = std::sync::OnceLock::new();
    let planner = PLANNER_CACHE.get_or_init(|| std::sync::Mutex::new(FftPlanner::new()));
    
    let ifft_plan = {
        let mut planner = planner.lock().unwrap();
        planner.plan_fft_inverse(fft_size)
    };
    
    // Perform IFFT in-place
    ifft_plan.process(&mut buffer);
    
    // Apply normalization if needed
    if norm_mode != NormMode::None {
        let scale = match norm_mode {
            NormMode::Forward => 1.0,  // Not applied for inverse FFT
            NormMode::Backward => 1.0 / (fft_size as f64),
            NormMode::Ortho => 1.0 / (fft_size as f64).sqrt(),
            NormMode::None => 1.0,  // Never happens due to check above
        };
        
        if scale != 1.0 {
            buffer.iter_mut().for_each(|c| {
                c.re *= scale;
                c.im *= scale;
            });
        }
    }
    
    // Prepare output
    let result = if let Some(output_vec) = out {
        // Reuse provided buffer
        output_vec.resize(fft_size, Complex64::new(0.0, 0.0));
        for (i, c) in buffer.iter().enumerate() {
            output_vec[i] = Complex64::new(c.re, c.im);
        }
        std::mem::take(output_vec)
    } else {
        // Create new output vector with minimal allocations
        let mut result = Vec::with_capacity(fft_size);
        unsafe {
            result.set_len(fft_size);
        }
        
        for (i, c) in buffer.iter().enumerate() {
            result[i] = Complex64::new(c.re, c.im);
        }
        result
    };
    
    // Return buffer to cache
    return_buffer_to_cache(buffer);
    
    Ok(result)
}

/// Compute the 2-dimensional Fast Fourier Transform with optimized memory usage
///
/// This version minimizes memory allocations by reusing internal buffers when possible
/// and by computing the FFT along each dimension separately.
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `shape` - Shape of the output (optional)
/// * `axes` - Axes along which to compute the FFT (optional)
/// * `norm` - Normalization mode (optional)
///
/// # Returns
///
/// A 2D array of complex values representing the FFT result
pub fn fft2_optimized<T>(
    input: &Array2<T>,
    shape: Option<(usize, usize)>,
    axes: Option<(i32, i32)>,
    norm: Option<&str>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Get input array shape
    let input_shape = input.shape();
    
    // Determine output shape
    let (n_rows_out, n_cols_out) = shape.unwrap_or((input_shape[0], input_shape[1]));
    
    // Determine axes to transform
    let (axis1, axis2) = axes.unwrap_or((0, 1));
    
    // Validate axes
    if axis1 < 0 || axis1 > 1 || axis2 < 0 || axis2 > 1 || axis1 == axis2 {
        return Err(FFTError::ValueError("Invalid axes for 2D FFT".to_string()));
    }
    
    // Parse normalization mode
    let norm_mode = match norm {
        Some("forward") => NormMode::Forward,
        Some("backward") => NormMode::Backward,
        Some("ortho") => NormMode::Ortho,
        _ => NormMode::Backward, // Default
    };
    
    // Create output array
    let mut output = Array2::<Complex64>::zeros((n_rows_out, n_cols_out));
    
    // Convert input to complex with minimal allocations
    let mut temp_buffer = Vec::with_capacity(input_shape[0].max(input_shape[1]));
    let mut output_buffer = Vec::with_capacity(n_rows_out.max(n_cols_out));
    
    // First, transform along rows
    for i in 0..input_shape[0].min(n_rows_out) {
        // Extract row
        temp_buffer.clear();
        for j in 0..input_shape[1] {
            let complex = to_complex_value(input[[i, j]])?;
            temp_buffer.push(complex);
        }
        
        // Compute FFT of row
        let row_fft = fft_optimized(&temp_buffer, Some(n_cols_out), Some(NormMode::None), Some(&mut output_buffer))?;
        
        // Store result
        for (j, &val) in row_fft.iter().enumerate() {
            output[[i, j]] = val;
        }
    }
    
    // Zero-fill any remaining rows
    for i in input_shape[0].min(n_rows_out)..n_rows_out {
        for j in 0..n_cols_out {
            output[[i, j]] = Complex64::new(0.0, 0.0);
        }
    }
    
    // Now, transform along columns
    temp_buffer.clear();
    temp_buffer.resize(n_rows_out, Complex64::new(0.0, 0.0));
    
    for j in 0..n_cols_out {
        // Extract column
        for i in 0..n_rows_out {
            temp_buffer[i] = output[[i, j]];
        }
        
        // Compute FFT of column
        let col_fft = fft_optimized(&temp_buffer, Some(n_rows_out), Some(NormMode::None), Some(&mut output_buffer))?;
        
        // Store result
        for (i, &val) in col_fft.iter().enumerate() {
            output[[i, j]] = val;
        }
    }
    
    // Apply normalization if needed
    if norm_mode != NormMode::None {
        let scale = match norm_mode {
            NormMode::Forward => 1.0 / (n_rows_out * n_cols_out) as f64,
            NormMode::Backward => 1.0, // No normalization for forward transform
            NormMode::Ortho => 1.0 / ((n_rows_out * n_cols_out) as f64).sqrt(),
            NormMode::None => 1.0, // Never happens due to check above
        };
        
        if scale != 1.0 {
            output.iter_mut().for_each(|c| *c *= scale);
        }
    }
    
    Ok(output)
}

/// Compute the inverse 2-dimensional Fast Fourier Transform with optimized memory usage
///
/// This version minimizes memory allocations by reusing internal buffers when possible
/// and by computing the FFT along each dimension separately.
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `shape` - Shape of the output (optional)
/// * `axes` - Axes along which to compute the inverse FFT (optional)
/// * `norm` - Normalization mode (optional)
///
/// # Returns
///
/// A 2D array of complex values representing the inverse FFT result
pub fn ifft2_optimized<T>(
    input: &Array2<T>,
    shape: Option<(usize, usize)>,
    axes: Option<(i32, i32)>,
    norm: Option<&str>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Get input array shape
    let input_shape = input.shape();
    
    // Determine output shape
    let (n_rows_out, n_cols_out) = shape.unwrap_or((input_shape[0], input_shape[1]));
    
    // Determine axes to transform
    let (axis1, axis2) = axes.unwrap_or((0, 1));
    
    // Validate axes
    if axis1 < 0 || axis1 > 1 || axis2 < 0 || axis2 > 1 || axis1 == axis2 {
        return Err(FFTError::ValueError("Invalid axes for 2D IFFT".to_string()));
    }
    
    // Parse normalization mode
    let norm_mode = match norm {
        Some("forward") => NormMode::Forward,
        Some("backward") => NormMode::Backward,
        Some("ortho") => NormMode::Ortho,
        _ => NormMode::Backward, // Default
    };
    
    // Create output array
    let mut output = Array2::<Complex64>::zeros((n_rows_out, n_cols_out));
    
    // Convert input to complex with minimal allocations
    let mut temp_buffer = Vec::with_capacity(input_shape[0].max(input_shape[1]));
    let mut output_buffer = Vec::with_capacity(n_rows_out.max(n_cols_out));
    
    // First, transform along rows
    for i in 0..input_shape[0].min(n_rows_out) {
        // Extract row
        temp_buffer.clear();
        for j in 0..input_shape[1] {
            let complex = to_complex_value(input[[i, j]])?;
            temp_buffer.push(complex);
        }
        
        // Compute IFFT of row
        let row_ifft = ifft_optimized(&temp_buffer, Some(n_cols_out), Some(NormMode::None), Some(&mut output_buffer))?;
        
        // Store result
        for (j, &val) in row_ifft.iter().enumerate() {
            output[[i, j]] = val;
        }
    }
    
    // Zero-fill any remaining rows
    for i in input_shape[0].min(n_rows_out)..n_rows_out {
        for j in 0..n_cols_out {
            output[[i, j]] = Complex64::new(0.0, 0.0);
        }
    }
    
    // Now, transform along columns
    temp_buffer.clear();
    temp_buffer.resize(n_rows_out, Complex64::new(0.0, 0.0));
    
    for j in 0..n_cols_out {
        // Extract column
        for i in 0..n_rows_out {
            temp_buffer[i] = output[[i, j]];
        }
        
        // Compute IFFT of column
        let col_ifft = ifft_optimized(&temp_buffer, Some(n_rows_out), Some(NormMode::None), Some(&mut output_buffer))?;
        
        // Store result
        for (i, &val) in col_ifft.iter().enumerate() {
            output[[i, j]] = val;
        }
    }
    
    // Apply normalization if needed
    if norm_mode != NormMode::None {
        let scale = match norm_mode {
            NormMode::Forward => 1.0, // No normalization for inverse transform
            NormMode::Backward => 1.0 / (n_rows_out * n_cols_out) as f64,
            NormMode::Ortho => 1.0 / ((n_rows_out * n_cols_out) as f64).sqrt(),
            NormMode::None => 1.0, // Never happens due to check above
        };
        
        if scale != 1.0 {
            output.iter_mut().for_each(|c| *c *= scale);
        }
    }
    
    Ok(output)
}