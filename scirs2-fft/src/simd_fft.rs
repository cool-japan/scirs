//! SIMD-accelerated FFT operations
//!
//! This module provides SIMD-accelerated implementations of FFT operations
//! for improved performance on modern CPUs with SIMD extensions.

#![allow(clippy::needless_range_loop)]

use crate::error::{FFTError, FFTResult};
use num_complex::Complex64;
use num_traits::NumCast;
use rustfft::{num_complex::Complex as RustComplex, FftPlanner};
use std::fmt::Debug;

/// Normalization mode for FFT operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormMode {
    /// No normalization (default for forward transforms)
    None,
    /// Normalize by 1/n (default for inverse transforms)
    Backward,
    /// Normalize by 1/sqrt(n) (unitary transform)
    Ortho,
    /// Normalize by 1/n for both forward and inverse transforms
    Forward,
}

// Import SIMD intrinsics based on target architecture
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Check if SIMD support is available at runtime
pub fn simd_support_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return true;
        }
        if is_x86_feature_detected!("sse4.1") {
            return true;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is required in all ARMv8-A implementations (aarch64)
        // so we can just return true here
        return true;
    }

    false
}

/// Convert a 1D array of Complex64 to SIMD-compatible representation
#[cfg(target_arch = "x86_64")]
#[allow(unused)]
fn complex_to_simd_array(input: &[Complex64]) -> Vec<(f64, f64)> {
    let mut result = Vec::with_capacity(input.len());
    for &c in input {
        result.push((c.re, c.im));
    }
    result
}

/// Convert SIMD representation back to Complex64
#[cfg(target_arch = "x86_64")]
#[allow(unused)]
fn simd_array_to_complex(input: &[(f64, f64)]) -> Vec<Complex64> {
    let mut result = Vec::with_capacity(input.len());
    for &(re, im) in input {
        result.push(Complex64::new(re, im));
    }
    result
}

/// Apply SIMD normalization to a complex array
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_simd_normalization_avx2(data: &mut [Complex64], scale: f64) {
    // Ensure data is aligned properly
    let len = data.len();
    let len_aligned = len - (len % 2); // Process 2 complex numbers at a time with AVX2

    // Convert scale to AVX2 register
    let scale_avx = _mm256_set1_pd(scale);

    // Process 2 complex numbers (4 doubles) at a time
    for i in (0..len_aligned).step_by(2) {
        // Load real and imaginary parts into AVX registers
        // Assuming data is laid out as [re0, im0, re1, im1, ...]
        let ptr = data.as_ptr() as *const f64;
        let values = _mm256_loadu_pd(ptr.add(i * 2));

        // Multiply by scale
        let scaled = _mm256_mul_pd(values, scale_avx);

        // Store back to memory
        let mut_ptr = data.as_mut_ptr() as *mut f64;
        _mm256_storeu_pd(mut_ptr.add(i * 2), scaled);
    }

    // Handle remaining elements
    for i in len_aligned..len {
        data[i] *= scale;
    }
}

/// Apply SIMD normalization to a complex array (SSE version)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn apply_simd_normalization_sse41(data: &mut [Complex64], scale: f64) {
    // Ensure data is aligned properly
    let len = data.len();
    // All elements will be processed since we're processing 1 at a time
    let len_aligned = len;

    // Convert scale to SSE register
    let scale_sse = _mm_set1_pd(scale);

    // Process 1 complex number (2 doubles) at a time
    for i in 0..len_aligned {
        // Load real and imaginary parts into SSE registers
        let ptr = data.as_ptr() as *const f64;
        let values = _mm_loadu_pd(ptr.add(i * 2));

        // Multiply by scale
        let scaled = _mm_mul_pd(values, scale_sse);

        // Store back to memory
        let mut_ptr = data.as_mut_ptr() as *mut f64;
        _mm_storeu_pd(mut_ptr.add(i * 2), scaled);
    }

    // No remaining elements to handle with SSE
}

/// Apply SIMD normalization to a complex array (ARM NEON version)
#[cfg(target_arch = "aarch64")]
unsafe fn apply_simd_normalization_neon(data: &mut [Complex64], scale: f64) {
    // Ensure data is aligned properly
    let len = data.len();
    let len_aligned = len - (len % 2); // Process 2 complex numbers at a time with NEON

    // Convert scale to NEON register (load 2 copies of the same value)
    let scale_neon = vdupq_n_f64(scale);

    // Process 2 complex numbers (4 doubles) at a time
    for i in (0..len_aligned).step_by(2) {
        // Load real and imaginary parts into NEON registers
        // Assuming data is laid out as [re0, im0, re1, im1, ...]
        let ptr = data.as_ptr() as *const f64;
        let values1 = vld1q_f64(ptr.add(i * 2));
        let values2 = vld1q_f64(ptr.add(i * 2 + 2));

        // Multiply by scale
        let scaled1 = vmulq_f64(values1, scale_neon);
        let scaled2 = vmulq_f64(values2, scale_neon);

        // Store back to memory
        let mut_ptr = data.as_mut_ptr() as *mut f64;
        vst1q_f64(mut_ptr.add(i * 2), scaled1);
        vst1q_f64(mut_ptr.add(i * 2 + 2), scaled2);
    }

    // Handle remaining elements
    for i in len_aligned..len {
        data[i] *= scale;
    }
}

/// Apply normalization to a complex array using the most efficient SIMD instructions
pub fn apply_simd_normalization(data: &mut [Complex64], scale: f64) {
    // Skip normalization if scale is 1.0
    if scale == 1.0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Use AVX2 if available
        if is_x86_feature_detected!("avx2") {
            unsafe {
                apply_simd_normalization_avx2(data, scale);
                return;
            }
        }

        // Fall back to SSE4.1 if available
        if is_x86_feature_detected!("sse4.1") {
            unsafe {
                apply_simd_normalization_sse41(data, scale);
                return;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Use NEON (always available on aarch64)
        unsafe {
            apply_simd_normalization_neon(data, scale);
            return;
        }
    }

    // Fall back to scalar implementation if no SIMD is available
    data.iter_mut().for_each(|c| *c *= scale);
}

/// Convert f64 to complex using NEON SIMD instructions
#[cfg(target_arch = "aarch64")]
unsafe fn simd_f64_to_complex_neon(input: &[f64]) -> FFTResult<Vec<Complex64>> {
    let len = input.len();
    let mut result = Vec::<Complex64>::with_capacity(len);

    // Set up result with correct length
    result.set_len(len);

    // Create zero vector for imaginary parts
    let zero = vdupq_n_f64(0.0);

    // Process 2 doubles at a time (producing 2 complex numbers)
    let len_aligned = len - (len % 2);
    for i in (0..len_aligned).step_by(2) {
        // Load 2 real values
        let real_values = vld1q_f64(&input[i]);

        // Set imaginary parts to zero
        let result_ptr = result.as_mut_ptr() as *mut f64;

        // Store real values
        vst1q_f64(result_ptr.add(i * 2), real_values);

        // Store imaginary parts (zeros)
        vst1q_f64(result_ptr.add(i * 2 + 1), zero);
    }

    // Handle remaining elements
    for i in len_aligned..len {
        result[i] = Complex64::new(input[i], 0.0);
    }

    Ok(result)
}

/// Convert input to complex with SIMD optimization when applicable
fn to_complex_simd<T>(input: &[T]) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    let mut result = Vec::with_capacity(input.len());

    // Check for available SIMD instructions
    #[cfg(target_arch = "x86_64")]
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
        && is_x86_feature_detected!("avx2")
    {
        // Fast path for f64 with AVX2
        if let Some(f64_input) = try_cast_to_f64_slice(input) {
            unsafe {
                return simd_f64_to_complex(f64_input);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        // Fast path for f64 with NEON (always available on aarch64)
        if let Some(f64_input) = try_cast_to_f64_slice(input) {
            unsafe {
                return simd_f64_to_complex_neon(f64_input);
            }
        }
    }

    // Fallback to scalar implementation
    for &val in input {
        // First try to cast directly to f64 (for real numbers)
        if let Some(real) = num_traits::cast::<T, f64>(val) {
            result.push(Complex64::new(real, 0.0));
            continue;
        }

        // If direct casting fails, check if it's already a Complex64
        use std::any::Any;
        if let Some(complex) = (&val as &dyn Any).downcast_ref::<Complex64>() {
            result.push(*complex);
            continue;
        }

        // Try to handle f32 complex numbers
        if let Some(complex32) = (&val as &dyn Any).downcast_ref::<num_complex::Complex<f32>>() {
            result.push(Complex64::new(complex32.re as f64, complex32.im as f64));
            continue;
        }

        return Err(FFTError::ValueError(format!(
            "Could not convert {:?} to numeric type",
            val
        )));
    }

    Ok(result)
}

/// Try to cast a slice of type T to a slice of f64
fn try_cast_to_f64_slice<T: 'static>(input: &[T]) -> Option<&[f64]> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let ptr = input.as_ptr() as *const f64;
        let len = input.len();
        unsafe {
            return Some(std::slice::from_raw_parts(ptr, len));
        }
    }
    None
}

/// Convert f64 to complex using SIMD instructions
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_f64_to_complex(input: &[f64]) -> FFTResult<Vec<Complex64>> {
    let len = input.len();
    // Initialize the vector with zeros to avoid uninitialized memory
    let mut result = vec![Complex64::new(0.0, 0.0); len];

    // Process 4 real numbers at a time, producing 4 complex numbers
    let len_aligned = len - (len % 4);
    for i in (0..len_aligned).step_by(4) {
        // Load 4 real values
        let real_values = _mm256_loadu_pd(&input[i]);

        // Store real values and zero imaginary parts
        let result_ptr = result.as_mut_ptr() as *mut f64;

        // Store the vector into a temporary array to extract values
        let mut temp_array = [0.0; 4];
        _mm256_storeu_pd(temp_array.as_mut_ptr(), real_values);

        // Store real parts (stride 2) - copying from temp array
        for (j, &val) in temp_array.iter().enumerate() {
            *result_ptr.add((i + j) * 2) = val;
            *result_ptr.add((i + j) * 2 + 1) = 0.0; // Imaginary part
        }
    }

    // Handle remaining elements
    for i in len_aligned..len {
        result[i] = Complex64::new(input[i], 0.0);
    }

    Ok(result)
}

/// Compute the 1-dimensional Fast Fourier Transform with SIMD acceleration
///
/// # Arguments
///
/// * `input` - Input data array
/// * `n` - Length of the output (optional)
/// * `norm` - Normalization mode (optional)
///
/// # Returns
///
/// A vector of complex values representing the FFT result
pub fn fft_simd<T>(
    input: &[T],
    n: Option<usize>,
    norm: Option<NormMode>,
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

    // Convert input to complex numbers with SIMD optimization when applicable
    let mut data = to_complex_simd(input)?;

    // Pad or truncate data to match fft_size
    if fft_size != input_len {
        if fft_size > input_len {
            // Pad with zeros
            data.resize(fft_size, Complex64::new(0.0, 0.0));
        } else {
            // Truncate
            data.truncate(fft_size);
        }
    }

    // Convert to rustfft-compatible complex type
    let mut buffer: Vec<RustComplex<f64>> =
        data.iter().map(|c| RustComplex::new(c.re, c.im)).collect();

    // Use rustfft library for computation
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Perform FFT in-place
    fft.process(&mut buffer);

    // Convert back to Complex64
    let mut result: Vec<Complex64> = buffer.iter().map(|c| Complex64::new(c.re, c.im)).collect();

    // Apply normalization with SIMD if needed
    if norm_mode != NormMode::None {
        let scale = match norm_mode {
            NormMode::Forward => 1.0 / (fft_size as f64),
            NormMode::Backward => 1.0, // No normalization for forward FFT
            NormMode::Ortho => 1.0 / (fft_size as f64).sqrt(),
            NormMode::None => 1.0, // Never happens due to check above
        };

        apply_simd_normalization(&mut result, scale);
    }

    Ok(result)
}

/// Compute the inverse 1-dimensional Fast Fourier Transform with SIMD acceleration
///
/// # Arguments
///
/// * `input` - Input complex data array
/// * `n` - Length of the output (optional)
/// * `norm` - Normalization mode (optional)
///
/// # Returns
///
/// A vector of complex values representing the inverse FFT result
pub fn ifft_simd<T>(
    input: &[T],
    n: Option<usize>,
    norm: Option<NormMode>,
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

    // Convert input to complex numbers
    let mut data = to_complex_simd(input)?;

    // Pad or truncate data to match fft_size
    if fft_size != input_len {
        if fft_size > input_len {
            // Pad with zeros
            data.resize(fft_size, Complex64::new(0.0, 0.0));
        } else {
            // Truncate
            data.truncate(fft_size);
        }
    }

    // Convert to rustfft-compatible complex type
    let mut buffer: Vec<RustComplex<f64>> =
        data.iter().map(|c| RustComplex::new(c.re, c.im)).collect();

    // Use rustfft library for computation
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(fft_size);

    // Perform inverse FFT in-place
    ifft.process(&mut buffer);

    // Convert back to Complex64
    let mut result: Vec<Complex64> = buffer.iter().map(|c| Complex64::new(c.re, c.im)).collect();

    // Apply normalization with SIMD if needed
    if norm_mode != NormMode::None {
        let scale = match norm_mode {
            NormMode::Forward => 1.0, // No normalization for inverse FFT
            NormMode::Backward => 1.0 / (fft_size as f64),
            NormMode::Ortho => 1.0 / (fft_size as f64).sqrt(),
            NormMode::None => 1.0, // Never happens due to check above
        };

        apply_simd_normalization(&mut result, scale);
    }

    Ok(result)
}

/// Complex number multiplication using SIMD instructions
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unused)]
unsafe fn complex_multiply_simd_avx2(a: &[Complex64], b: &[Complex64], out: &mut [Complex64]) {
    assert!(a.len() == b.len() && a.len() == out.len());

    let len = a.len();
    let len_aligned = len - (len % 2); // Process 2 complex numbers at a time with AVX2

    for i in (0..len_aligned).step_by(2) {
        // Load complex numbers
        let a_ptr = a.as_ptr() as *const f64;
        let b_ptr = b.as_ptr() as *const f64;

        // Load a's components: [a0.re, a0.im, a1.re, a1.im]
        let a_vals = _mm256_loadu_pd(a_ptr.add(i * 2));

        // Load b's components: [b0.re, b0.im, b1.re, b1.im]
        let b_vals = _mm256_loadu_pd(b_ptr.add(i * 2));

        // Create register with swapped real/imaginary parts for a: [a0.im, a0.re, a1.im, a1.re]
        let a_swapped = _mm256_permute_pd(a_vals, 0b0101);

        // Create b_real = [b0.re, b0.re, b1.re, b1.re]
        let b_real = _mm256_permute_pd(b_vals, 0b0000);

        // Create b_imag = [b0.im, b0.im, b1.im, b1.im]
        let b_imag = _mm256_permute_pd(b_vals, 0b1111);

        // Compute real parts: a.re * b.re - a.im * b.im
        let real_part = _mm256_fmsub_pd(a_vals, b_real, _mm256_mul_pd(a_swapped, b_imag));

        // Compute imaginary parts: a.re * b.im + a.im * b.re
        let imag_part = _mm256_fmadd_pd(a_vals, b_imag, _mm256_mul_pd(a_swapped, b_real));

        // Interleave real and imaginary parts
        let out_ptr = out.as_mut_ptr() as *mut f64;

        // Store results by storing to temp arrays and copying
        let mut real_temp = [0.0; 4];
        let mut imag_temp = [0.0; 4];

        _mm256_storeu_pd(real_temp.as_mut_ptr(), real_part);
        _mm256_storeu_pd(imag_temp.as_mut_ptr(), imag_part);

        for j in 0..2 {
            let re_idx = i + j;
            *out_ptr.add(re_idx * 2) = real_temp[j];
            *out_ptr.add(re_idx * 2 + 1) = imag_temp[j];
        }
    }

    // Handle remaining elements
    for i in len_aligned..len {
        let re = a[i].re * b[i].re - a[i].im * b[i].im;
        let im = a[i].re * b[i].im + a[i].im * b[i].re;
        out[i] = Complex64::new(re, im);
    }
}

/// Apply a window function to an array using SIMD instructions
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unused)]
unsafe fn apply_window_simd_avx2(data: &mut [f64], window: &[f64]) {
    assert_eq!(data.len(), window.len());

    let len = data.len();
    let len_aligned = len - (len % 4); // Process 4 values at a time with AVX2

    for i in (0..len_aligned).step_by(4) {
        let data_val = _mm256_loadu_pd(&data[i]);
        let window_val = _mm256_loadu_pd(&window[i]);

        let result = _mm256_mul_pd(data_val, window_val);

        _mm256_storeu_pd(&mut data[i], result);
    }

    // Handle remaining elements
    for i in len_aligned..len {
        data[i] *= window[i];
    }
}

/// Adaptive FFT dispatcher that selects the best implementation based on hardware support
pub fn fft_adaptive<T>(
    input: &[T],
    n: Option<usize>,
    norm: Option<NormMode>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Check for SIMD support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return fft_simd(input, n, norm);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return fft_simd(input, n, norm);
    }

    // Fall back to standard implementation
    crate::fft::fft(input, n)
}

/// Adaptive iFFT dispatcher that selects the best implementation based on hardware support
pub fn ifft_adaptive<T>(
    input: &[T],
    n: Option<usize>,
    norm: Option<NormMode>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Check for SIMD support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return ifft_simd(input, n, norm);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return ifft_simd(input, n, norm);
    }

    // Fall back to standard implementation
    crate::fft::ifft(input, n)
}

/// Compute the 2-dimensional Fast Fourier Transform with SIMD acceleration
///
/// # Arguments
///
/// * `input` - Input 2D array (flattened in row-major order)
/// * `shape` - Dimensions of the input array [rows, cols]
/// * `axes` - Axes along which to perform the FFT (optional, default is [-2, -1])
/// * `norm` - Normalization mode (optional)
///
/// # Returns
///
/// A vector of complex values representing the 2D FFT result
pub fn fft2_simd<T>(
    input: &[T],
    shape: [usize; 2],
    axes: Option<[isize; 2]>,
    norm: Option<NormMode>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Input validation
    if input.is_empty() {
        return Err(FFTError::ValueError("Input cannot be empty".to_string()));
    }

    if input.len() != shape[0] * shape[1] {
        return Err(FFTError::ValueError(format!(
            "Input length ({}) does not match the provided shape ({} x {})",
            input.len(),
            shape[0],
            shape[1]
        )));
    }

    // Default axes are [-2, -1] which corresponds to [0, 1] for 2D array
    let axes = axes.unwrap_or([-2, -1]);
    let axes = [
        if axes[0] < 0 { axes[0] + 2 } else { axes[0] },
        if axes[1] < 0 { axes[1] + 2 } else { axes[1] },
    ];

    // Validate axes
    if axes[0] < 0 || axes[0] > 1 || axes[1] < 0 || axes[1] > 1 || axes[0] == axes[1] {
        return Err(FFTError::ValueError(format!(
            "Invalid FFT axes: {:?}",
            axes
        )));
    }

    // Get the normalization mode
    let norm_mode = norm.unwrap_or(NormMode::None);

    // Convert input to complex
    let mut data = to_complex_simd(input)?;

    // Reshape to 2D for processing
    let n_rows = shape[0];
    let n_cols = shape[1];

    // First FFT pass: transform each row
    if axes.contains(&0) {
        for i in 0..n_rows {
            let row_start = i * n_cols;
            let row_end = row_start + n_cols;
            let row_fft = fft_simd(&data[row_start..row_end], None, None)?;
            data[row_start..row_end].copy_from_slice(&row_fft);
        }
    }

    // Second FFT pass: transform each column
    if axes.contains(&1) {
        // We need to transpose, transform, and transpose back
        // Allocate temporary storage for column-wise processing
        let mut temp_col = vec![Complex64::new(0.0, 0.0); n_rows];

        for j in 0..n_cols {
            // Extract column
            for i in 0..n_rows {
                temp_col[i] = data[i * n_cols + j];
            }

            // Perform FFT on the column
            let col_fft = fft_simd(&temp_col, None, None)?;

            // Put the transformed column back
            for i in 0..n_rows {
                data[i * n_cols + j] = col_fft[i];
            }
        }
    }

    // Apply normalization if needed
    if norm_mode != NormMode::None {
        let total_elements = n_rows * n_cols;
        let scale = match norm_mode {
            NormMode::Forward => 1.0 / (total_elements as f64),
            NormMode::Backward => 1.0, // No normalization for forward FFT
            NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),
            NormMode::None => 1.0, // Never happens due to check above
        };

        apply_simd_normalization(&mut data, scale);
    }

    Ok(data)
}

/// Compute the inverse 2-dimensional Fast Fourier Transform with SIMD acceleration
///
/// # Arguments
///
/// * `input` - Input 2D array (flattened in row-major order)
/// * `shape` - Dimensions of the input array [rows, cols]
/// * `axes` - Axes along which to perform the IFFT (optional, default is [-2, -1])
/// * `norm` - Normalization mode (optional)
///
/// # Returns
///
/// A vector of complex values representing the inverse 2D FFT result
pub fn ifft2_simd<T>(
    input: &[T],
    shape: [usize; 2],
    axes: Option<[isize; 2]>,
    norm: Option<NormMode>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Input validation
    if input.is_empty() {
        return Err(FFTError::ValueError("Input cannot be empty".to_string()));
    }

    if input.len() != shape[0] * shape[1] {
        return Err(FFTError::ValueError(format!(
            "Input length ({}) does not match the provided shape ({} x {})",
            input.len(),
            shape[0],
            shape[1]
        )));
    }

    // Default axes are [-2, -1] which corresponds to [0, 1] for 2D array
    let axes = axes.unwrap_or([-2, -1]);
    let axes = [
        if axes[0] < 0 { axes[0] + 2 } else { axes[0] },
        if axes[1] < 0 { axes[1] + 2 } else { axes[1] },
    ];

    // Validate axes
    if axes[0] < 0 || axes[0] > 1 || axes[1] < 0 || axes[1] > 1 || axes[0] == axes[1] {
        return Err(FFTError::ValueError(format!(
            "Invalid IFFT axes: {:?}",
            axes
        )));
    }

    // Get the normalization mode
    let norm_mode = norm.unwrap_or(NormMode::Backward);

    // Convert input to complex
    let mut data = to_complex_simd(input)?;

    // Reshape to 2D for processing
    let n_rows = shape[0];
    let n_cols = shape[1];

    // First IFFT pass: transform each row
    if axes.contains(&0) {
        for i in 0..n_rows {
            let row_start = i * n_cols;
            let row_end = row_start + n_cols;
            let row_ifft = ifft_simd(&data[row_start..row_end], None, Some(NormMode::None))?;
            data[row_start..row_end].copy_from_slice(&row_ifft);
        }
    }

    // Second IFFT pass: transform each column
    if axes.contains(&1) {
        // We need to transpose, transform, and transpose back
        // Allocate temporary storage for column-wise processing
        let mut temp_col = vec![Complex64::new(0.0, 0.0); n_rows];

        for j in 0..n_cols {
            // Extract column
            for i in 0..n_rows {
                temp_col[i] = data[i * n_cols + j];
            }

            // Perform IFFT on the column
            let col_ifft = ifft_simd(&temp_col, None, Some(NormMode::None))?;

            // Put the transformed column back
            for i in 0..n_rows {
                data[i * n_cols + j] = col_ifft[i];
            }
        }
    }

    // Apply normalization if needed
    if norm_mode != NormMode::None {
        let total_elements = n_rows * n_cols;
        let scale = match norm_mode {
            NormMode::Forward => 1.0, // No normalization for inverse FFT with Forward
            NormMode::Backward => 1.0 / (total_elements as f64),
            NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),
            NormMode::None => 1.0, // Never happens due to check above
        };

        apply_simd_normalization(&mut data, scale);
    }

    Ok(data)
}

/// Adaptive 2D FFT dispatcher that selects the best implementation based on hardware support
pub fn fft2_adaptive<T>(
    input: &[T],
    shape: [usize; 2],
    axes: Option<[isize; 2]>,
    norm: Option<NormMode>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Check for SIMD support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return fft2_simd(input, shape, axes, norm);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is required in all ARMv8-A implementations (aarch64)
        // so we can always use the SIMD implementation
        return fft2_simd(input, shape, axes, norm);
    }

    // Fall back to standard implementation by converting to ndarray Array2
    // This is necessary because the core implementation uses ndarray types

    // Convert input to Vec<T> if not already
    let input_vec: Vec<T> = input.to_vec();

    // Create Array2 from 1D vector
    let array2d = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[shape[0], shape[1]]), input_vec)
        .map_err(|e| FFTError::ValueError(format!("Failed to reshape input: {}", e)))?;

    // Convert to Array2 for fft2
    let array2d_ref = array2d
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| FFTError::ValueError(format!("Failed to create 2D array: {}", e)))?;

    // Convert axes format
    let ndarray_axes = axes.map(|[a, b]| (a as i32, b as i32));

    // Convert norm mode
    let norm_str = match norm {
        Some(NormMode::Forward) => Some("forward"),
        Some(NormMode::Backward) => Some("backward"),
        Some(NormMode::Ortho) => Some("ortho"),
        Some(NormMode::None) | None => None,
    };

    // Call standard implementation
    let result = crate::fft::fft2(
        &array2d_ref,
        Some((shape[0], shape[1])),
        ndarray_axes,
        norm_str,
    )?;

    // Convert result back to Vec
    let (result_vec, _) = result.into_raw_vec_and_offset();

    Ok(result_vec)
}

/// Compute the N-dimensional Fast Fourier Transform with SIMD acceleration
///
/// # Arguments
///
/// * `input` - Input ND array (flattened in row-major order)
/// * `shape` - Dimensions of the input array
/// * `axes` - Axes along which to perform the FFT (optional)
/// * `norm` - Normalization mode (optional)
///
/// # Returns
///
/// A vector of complex values representing the ND FFT result
pub fn fftn_simd<T>(
    input: &[T],
    shape: &[usize],
    axes: Option<&[isize]>,
    norm: Option<NormMode>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Input validation
    if input.is_empty() {
        return Err(FFTError::ValueError("Input cannot be empty".to_string()));
    }

    let n_dims = shape.len();

    // Calculate total number of elements
    let total_elements: usize = shape.iter().product();
    if input.len() != total_elements {
        return Err(FFTError::ValueError(format!(
            "Input length ({}) does not match the provided shape ({:?})",
            input.len(),
            shape
        )));
    }

    // Default axes is all dimensions
    let axes = match axes {
        Some(a) => a.to_vec(),
        None => (0..n_dims as isize).collect::<Vec<_>>(),
    };

    // Validate axes
    for &axis in &axes {
        let dim_index = if axis < 0 {
            axis + n_dims as isize
        } else {
            axis
        } as usize;
        if dim_index >= n_dims {
            return Err(FFTError::ValueError(format!("Invalid FFT axis: {}", axis)));
        }
    }

    // Get the normalization mode
    let norm_mode = norm.unwrap_or(NormMode::None);

    // Convert input to complex
    let mut data = to_complex_simd(input)?;

    // Strides for each dimension
    let mut strides = vec![1usize; n_dims];
    for i in (0..n_dims - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Process each axis
    for &axis in &axes {
        // Convert negative axis to positive
        let dim_index = if axis < 0 {
            axis + n_dims as isize
        } else {
            axis
        } as usize;

        // Get size along this dimension
        let dim_size = shape[dim_index];

        // Skip if dimension size is 1 (no FFT needed)
        if dim_size <= 1 {
            continue;
        }

        // Stride for this dimension
        let stride = strides[dim_index];

        // Number of separate FFTs to compute
        let num_ffts = total_elements / dim_size;

        // For each group of elements along this axis
        for i in 0..num_ffts {
            // Calculate offset for this group
            let offset = (i / stride) * stride * dim_size + (i % stride);

            // Extract elements for this FFT
            let mut temp = Vec::with_capacity(dim_size);
            for j in 0..dim_size {
                temp.push(data[offset + j * stride]);
            }

            // Compute 1D FFT
            let fft_result = fft_simd(&temp, None, None)?;

            // Store back the results
            for j in 0..dim_size {
                data[offset + j * stride] = fft_result[j];
            }
        }
    }

    // Apply normalization if needed
    if norm_mode != NormMode::None {
        let scale = match norm_mode {
            NormMode::Forward => 1.0 / (total_elements as f64),
            NormMode::Backward => 1.0, // No normalization for forward FFT
            NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),
            NormMode::None => 1.0, // Never happens due to check above
        };

        apply_simd_normalization(&mut data, scale);
    }

    Ok(data)
}

/// Compute the inverse N-dimensional Fast Fourier Transform with SIMD acceleration
///
/// # Arguments
///
/// * `input` - Input ND array (flattened in row-major order)
/// * `shape` - Dimensions of the input array
/// * `axes` - Axes along which to perform the IFFT (optional)
/// * `norm` - Normalization mode (optional)
///
/// # Returns
///
/// A vector of complex values representing the inverse ND FFT result
pub fn ifftn_simd<T>(
    input: &[T],
    shape: &[usize],
    axes: Option<&[isize]>,
    norm: Option<NormMode>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Input validation
    if input.is_empty() {
        return Err(FFTError::ValueError("Input cannot be empty".to_string()));
    }

    let n_dims = shape.len();

    // Calculate total number of elements
    let total_elements: usize = shape.iter().product();
    if input.len() != total_elements {
        return Err(FFTError::ValueError(format!(
            "Input length ({}) does not match the provided shape ({:?})",
            input.len(),
            shape
        )));
    }

    // Default axes is all dimensions
    let axes = match axes {
        Some(a) => a.to_vec(),
        None => (0..n_dims as isize).collect::<Vec<_>>(),
    };

    // Validate axes
    for &axis in &axes {
        let dim_index = if axis < 0 {
            axis + n_dims as isize
        } else {
            axis
        } as usize;
        if dim_index >= n_dims {
            return Err(FFTError::ValueError(format!("Invalid IFFT axis: {}", axis)));
        }
    }

    // Get the normalization mode
    let norm_mode = norm.unwrap_or(NormMode::Backward);

    // Convert input to complex
    let mut data = to_complex_simd(input)?;

    // Strides for each dimension
    let mut strides = vec![1usize; n_dims];
    for i in (0..n_dims - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Process each axis
    for &axis in &axes {
        // Convert negative axis to positive
        let dim_index = if axis < 0 {
            axis + n_dims as isize
        } else {
            axis
        } as usize;

        // Get size along this dimension
        let dim_size = shape[dim_index];

        // Skip if dimension size is 1 (no IFFT needed)
        if dim_size <= 1 {
            continue;
        }

        // Stride for this dimension
        let stride = strides[dim_index];

        // Number of separate IFFTs to compute
        let num_iffts = total_elements / dim_size;

        // For each group of elements along this axis
        for i in 0..num_iffts {
            // Calculate offset for this group
            let offset = (i / stride) * stride * dim_size + (i % stride);

            // Extract elements for this IFFT
            let mut temp = Vec::with_capacity(dim_size);
            for j in 0..dim_size {
                temp.push(data[offset + j * stride]);
            }

            // Compute 1D IFFT (with no normalization yet)
            let ifft_result = ifft_simd(&temp, None, Some(NormMode::None))?;

            // Store back the results
            for j in 0..dim_size {
                data[offset + j * stride] = ifft_result[j];
            }
        }
    }

    // Apply normalization if needed
    if norm_mode != NormMode::None {
        let scale = match norm_mode {
            NormMode::Forward => 1.0, // No normalization for inverse FFT with Forward mode
            NormMode::Backward => 1.0 / (total_elements as f64),
            NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),
            NormMode::None => 1.0, // Never happens due to check above
        };

        apply_simd_normalization(&mut data, scale);
    }

    Ok(data)
}

/// Adaptive N-dimensional FFT dispatcher that selects the best implementation based on hardware support
pub fn fftn_adaptive<T>(
    input: &[T],
    shape: &[usize],
    axes: Option<&[isize]>,
    norm: Option<NormMode>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Check for SIMD support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return fftn_simd(input, shape, axes, norm);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is required in all ARMv8-A implementations (aarch64)
        // so we can always use the SIMD implementation
        return fftn_simd(input, shape, axes, norm);
    }

    // Fall back to standard implementation by converting to ndarray ArrayD

    // Convert input to Vec<T> if not already
    let input_vec: Vec<T> = input.to_vec();

    // Create ArrayD from 1D vector
    let arrayd = ndarray::Array::from_shape_vec(ndarray::IxDyn(shape), input_vec)
        .map_err(|e| FFTError::ValueError(format!("Failed to reshape input: {}", e)))?;

    // Convert axes format
    let ndarray_axes: Option<Vec<usize>> = match axes {
        Some(a) => {
            // Convert from isize to usize, handling negative indices
            let n_dims = shape.len();
            Some(
                a.iter()
                    .map(|&ax| {
                        let ax_pos = if ax < 0 { ax + n_dims as isize } else { ax };
                        ax_pos as usize
                    })
                    .collect(),
            )
        }
        None => None,
    };

    // Convert norm mode
    let norm_str = match norm {
        Some(NormMode::Forward) => Some("forward"),
        Some(NormMode::Backward) => Some("backward"),
        Some(NormMode::Ortho) => Some("ortho"),
        Some(NormMode::None) | None => None,
    };

    // Call standard implementation with all required parameters
    let result = crate::fft::fftn(&arrayd, None, ndarray_axes, norm_str, None, None)?;

    // Convert result back to Vec
    let (result_vec, _) = result.into_raw_vec_and_offset();

    Ok(result_vec)
}

/// Adaptive N-dimensional IFFT dispatcher that selects the best implementation based on hardware support
pub fn ifftn_adaptive<T>(
    input: &[T],
    shape: &[usize],
    axes: Option<&[isize]>,
    norm: Option<NormMode>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Check for SIMD support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return ifftn_simd(input, shape, axes, norm);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is required in all ARMv8-A implementations (aarch64)
        // so we can always use the SIMD implementation
        return ifftn_simd(input, shape, axes, norm);
    }

    // Fall back to standard implementation by converting to ndarray ArrayD

    // Convert input to Vec<T> if not already
    let input_vec: Vec<T> = input.to_vec();

    // Create ArrayD from 1D vector
    let arrayd = ndarray::Array::from_shape_vec(ndarray::IxDyn(shape), input_vec)
        .map_err(|e| FFTError::ValueError(format!("Failed to reshape input: {}", e)))?;

    // Convert axes format
    let ndarray_axes: Option<Vec<usize>> = match axes {
        Some(a) => {
            // Convert from isize to usize, handling negative indices
            let n_dims = shape.len();
            Some(
                a.iter()
                    .map(|&ax| {
                        let ax_pos = if ax < 0 { ax + n_dims as isize } else { ax };
                        ax_pos as usize
                    })
                    .collect(),
            )
        }
        None => None,
    };

    // Convert norm mode
    let norm_str = match norm {
        Some(NormMode::Forward) => Some("forward"),
        Some(NormMode::Backward) => Some("backward"),
        Some(NormMode::Ortho) => Some("ortho"),
        Some(NormMode::None) | None => None,
    };

    // Call standard implementation with all required parameters
    let result = crate::fft::ifftn(&arrayd, None, ndarray_axes, norm_str, None, None)?;

    // Convert result back to Vec
    let (result_vec, _) = result.into_raw_vec_and_offset();

    Ok(result_vec)
}

/// Adaptive 2D IFFT dispatcher that selects the best implementation based on hardware support
pub fn ifft2_adaptive<T>(
    input: &[T],
    shape: [usize; 2],
    axes: Option<[isize; 2]>,
    norm: Option<NormMode>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Check for SIMD support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return ifft2_simd(input, shape, axes, norm);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is required in all ARMv8-A implementations (aarch64)
        // so we can always use the SIMD implementation
        return ifft2_simd(input, shape, axes, norm);
    }

    // Fall back to standard implementation by converting to ndarray Array2
    // This is necessary because the core implementation uses ndarray types

    // Convert input to Vec<T> if not already
    let input_vec: Vec<T> = input.to_vec();

    // Create Array2 from 1D vector
    let array2d = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[shape[0], shape[1]]), input_vec)
        .map_err(|e| FFTError::ValueError(format!("Failed to reshape input: {}", e)))?;

    // Convert to Array2 for ifft2
    let array2d_ref = array2d
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| FFTError::ValueError(format!("Failed to create 2D array: {}", e)))?;

    // Convert axes format
    let ndarray_axes = axes.map(|[a, b]| (a as i32, b as i32));

    // Convert norm mode
    let norm_str = match norm {
        Some(NormMode::Forward) => Some("forward"),
        Some(NormMode::Backward) => Some("backward"),
        Some(NormMode::Ortho) => Some("ortho"),
        Some(NormMode::None) | None => None,
    };

    // Call standard implementation
    let result = crate::fft::ifft2(
        &array2d_ref,
        Some((shape[0], shape[1])),
        ndarray_axes,
        norm_str,
    )?;

    // Convert result back to Vec
    let (result_vec, _) = result.into_raw_vec_and_offset();

    Ok(result_vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_simd_support_detection() {
        // This just tests that the function runs without error
        let _ = simd_support_available();
    }

    #[test]
    fn test_fft_simd_vs_standard() {
        // Only run this test if SIMD is available
        if !simd_support_available() {
            return;
        }

        // Create a test signal
        let signal: Vec<f64> = (0..16).map(|i| (i as f64).sin()).collect();

        // Compute FFT with standard implementation
        let standard_result = crate::fft::fft(&signal, None).unwrap();

        // Compute FFT with SIMD implementation
        let simd_result = fft_simd(&signal, None, None).unwrap();

        // Compare results
        assert_eq!(standard_result.len(), simd_result.len());
        for i in 0..standard_result.len() {
            assert_relative_eq!(standard_result[i].re, simd_result[i].re, epsilon = 1e-10);
            assert_relative_eq!(standard_result[i].im, simd_result[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normalization() {
        // Only run this test if SIMD is available
        if !simd_support_available() {
            return;
        }

        // Create a test array
        let mut data: Vec<Complex64> = (0..16).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let mut expected = data.clone();

        // Apply normalization
        let scale = 0.25;
        expected.iter_mut().for_each(|c| *c *= scale);

        // Apply SIMD normalization
        apply_simd_normalization(&mut data, scale);

        // Compare results
        for i in 0..data.len() {
            assert_relative_eq!(data[i].re, expected[i].re, epsilon = 1e-10);
            assert_relative_eq!(data[i].im, expected[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_adaptive_dispatch() {
        // Create a test signal
        let signal: Vec<f64> = (0..16).map(|i| (i as f64).sin()).collect();

        // Compute FFT with adaptive implementation
        let result = fft_adaptive(&signal, None, None).unwrap();

        // Just verify it returns data of the right size
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_fft2_simd() {
        // Only run this test if SIMD is available
        if !simd_support_available() {
            return;
        }

        // Create a 2D test signal (4x4 grid)
        let n_rows = 4;
        let n_cols = 4;
        let mut signal = Vec::with_capacity(n_rows * n_cols);

        for i in 0..n_rows {
            for j in 0..n_cols {
                let x = i as f64 / n_rows as f64;
                let y = j as f64 / n_cols as f64;
                let value = (2.0 * PI * x).sin() * (2.0 * PI * y).cos();
                signal.push(value);
            }
        }

        // Compute 2D FFT with SIMD implementation
        let fft_result = fft2_simd(&signal, [n_rows, n_cols], None, None).unwrap();

        // Verify some basic properties of the result
        assert_eq!(fft_result.len(), n_rows * n_cols);

        // Check if DC component is approximately equal to sum of all signal values
        let signal_sum: f64 = signal.iter().sum();
        assert_relative_eq!(fft_result[0].re, signal_sum, epsilon = 1e-10);

        // Compute IFFT to verify round-trip
        let ifft_result = ifft2_simd(&fft_result, [n_rows, n_cols], None, None).unwrap();

        // Verify that the round-trip recovers the original signal
        for (i, &val) in signal.iter().enumerate() {
            assert_relative_eq!(val, ifft_result[i].re, epsilon = 1e-10);
            assert_relative_eq!(0.0, ifft_result[i].im.abs(), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ifft2_simd_roundtrip() {
        // Only run this test if SIMD is available
        if !simd_support_available() {
            return;
        }

        // Create a 2D test signal (8x8 grid)
        let n_rows = 8;
        let n_cols = 8;
        let mut signal = Vec::with_capacity(n_rows * n_cols);

        for i in 0..n_rows {
            for j in 0..n_cols {
                let x = i as f64 / n_rows as f64;
                let y = j as f64 / n_cols as f64;
                let value = (2.0 * PI * 2.0 * x).sin() * (2.0 * PI * 3.0 * y).cos();
                signal.push(value);
            }
        }

        // Forward 2D FFT
        let spectrum = fft2_simd(&signal, [n_rows, n_cols], None, None).unwrap();

        // Inverse 2D FFT
        let recovered = ifft2_simd(&spectrum, [n_rows, n_cols], None, None).unwrap();

        // Compare original and recovered signals
        assert_eq!(signal.len(), recovered.len());
        for i in 0..signal.len() {
            assert_relative_eq!(signal[i], recovered[i].re, epsilon = 1e-10);
            assert_relative_eq!(0.0, recovered[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft2_axes_selection() {
        // Only run this test if SIMD is available
        if !simd_support_available() {
            return;
        }

        // Create a 2D test signal (4x4 grid)
        let n_rows = 4;
        let n_cols = 4;
        let mut signal = Vec::with_capacity(n_rows * n_cols);

        for i in 0..n_rows {
            for j in 0..n_cols {
                let x = i as f64 / n_rows as f64;
                let y = j as f64 / n_cols as f64;
                let value = (2.0 * PI * x).sin() * (2.0 * PI * y).cos();
                signal.push(value);
            }
        }

        // Compute 2D FFT with all axes
        let all_axes = fft2_simd(&signal, [n_rows, n_cols], None, None).unwrap();

        // Compute 2D FFT with default axes [0, 1] (both dimensions)
        let default_axes = fft2_simd(&signal, [n_rows, n_cols], Some([0, 1]), None).unwrap();

        // The all_axes result should be the same as default axes
        assert_eq!(all_axes.len(), default_axes.len());
        for i in 0..all_axes.len() {
            assert!((all_axes[i].re - default_axes[i].re).abs() < 1e-10);
            assert!((all_axes[i].im - default_axes[i].im).abs() < 1e-10);
        }

        // Test validates that specifying axes explicitly produces the same result as default
    }

    #[test]
    fn test_adaptive_2d_dispatch() {
        // Create a 2D test signal
        let n_rows = 4;
        let n_cols = 4;
        let signal = vec![1.0; n_rows * n_cols];

        // Compute FFT with adaptive implementation
        let result = fft2_adaptive(&signal, [n_rows, n_cols], None, None).unwrap();

        // Just verify it returns data of the right size
        assert_eq!(result.len(), n_rows * n_cols);
    }

    #[test]
    fn test_fftn_simd_basic() {
        // Only run this test if SIMD is available
        if !simd_support_available() {
            return;
        }

        // Create a 3D test signal (2x3x4)
        let shape = [2, 3, 4];
        let total_elements = shape.iter().product();
        let signal: Vec<f64> = (0..total_elements).map(|i| i as f64).collect();

        // Compute N-dimensional FFT with SIMD acceleration
        let result = fftn_simd(&signal, &shape, None, None).unwrap();

        // Verify dimensions
        assert_eq!(result.len(), total_elements);

        // The DC component (first element) should be the sum of all elements
        let sum: f64 = signal.iter().sum();
        assert_relative_eq!(result[0].re, sum, epsilon = 1e-10);
    }

    #[test]
    fn test_ifftn_simd_roundtrip() {
        // Only run this test if SIMD is available
        if !simd_support_available() {
            return;
        }

        // Create a 3D test signal (4x4x4)
        let shape = [4, 4, 4];
        let total_elements = shape.iter().product();

        // Generate a test signal
        let mut signal = Vec::with_capacity(total_elements);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let x = i as f64 / shape[0] as f64;
                    let y = j as f64 / shape[1] as f64;
                    let z = k as f64 / shape[2] as f64;
                    let value = (2.0 * PI * x).sin() + (4.0 * PI * y).cos() + (2.0 * PI * z).sin();
                    signal.push(value);
                }
            }
        }

        // Compute FFT
        let spectrum = fftn_simd(&signal, &shape, None, None).unwrap();

        // Compute IFFT
        let recovered = ifftn_simd(&spectrum, &shape, None, None).unwrap();

        // Verify dimensions
        assert_eq!(recovered.len(), total_elements);

        // Verify that the recovered signal matches the original
        for i in 0..total_elements {
            assert_relative_eq!(signal[i], recovered[i].re, epsilon = 1e-10);
            assert_relative_eq!(0.0, recovered[i].im.abs(), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fftn_axis_selection() {
        // Only run this test if SIMD is available
        if !simd_support_available() {
            return;
        }

        // Create a 3D test signal (2x3x4)
        let shape = [2, 3, 4];
        let total_elements = shape.iter().product();
        let signal: Vec<f64> = (0..total_elements).map(|i| i as f64).collect();

        // Compute FFT along all axes
        let all_axes = fftn_simd(&signal, &shape, None, None).unwrap();

        // Compute FFT along first axis only
        let axis0_only = fftn_simd(&signal, &shape, Some(&[0]), None).unwrap();

        // The results should be different
        let mut all_same = true;
        for i in 0..total_elements {
            if (all_axes[i].re - axis0_only[i].re).abs() > 1e-10
                || (all_axes[i].im - axis0_only[i].im).abs() > 1e-10
            {
                all_same = false;
                break;
            }
        }
        assert!(
            !all_same,
            "FFT along different axes should produce different results"
        );
    }

    #[test]
    fn test_adaptive_nd_dispatch() {
        // Create a 3D test signal
        let shape = [2, 3, 4];
        let total_elements = shape.iter().product();
        let signal = vec![1.0; total_elements];

        // Compute FFT with adaptive implementation
        let result = fftn_adaptive(&signal, &shape, None, None).unwrap();

        // Just verify it returns data of the right size
        assert_eq!(result.len(), total_elements);
    }
}
