// Direct SIMD → Scalar mappings for missing SIMD functions
// 
// This file provides scalar fallbacks for missing SIMD implementations.
// These can be replaced with proper SIMD implementations in the future.

use crate::error::{SignalError, SignalResult};

// Direct mappings for functions with existing scalar implementations

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
    super::scalar_fir_filter(input, coeffs, output)
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
    super::scalar_fir_filter(input, coeffs, output)
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_autocorrelation(signal: &[f64], maxlag: usize) -> SignalResult<Vec<f64>> {
    super::scalar_autocorrelation(signal, max_lag)
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_autocorrelation(signal: &[f64], maxlag: usize) -> SignalResult<Vec<f64>> {
    super::scalar_autocorrelation(signal, max_lag)
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    super::scalar_cross_correlation(signal1, signal2, output)
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    super::scalar_cross_correlation(signal1, signal2, output)
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_complex_butterfly(
    data: &mut [scirs2_core::numeric::Complex<f64>],
    twiddles: &[scirs2_core::numeric::Complex<f64>],
) -> SignalResult<()> {
    super::scalar_complex_butterfly(data, twiddles)
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_complex_butterfly(
    data: &mut [scirs2_core::numeric::Complex<f64>],
    twiddles: &[scirs2_core::numeric::Complex<f64>],
) -> SignalResult<()> {
    super::scalar_complex_butterfly(data, twiddles)
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn avx2_enhanced_convolution(
    signal: &[f64],
    kernel: &[f64],
    output: &mut [f64]
) -> SignalResult<()> {
    super::scalar_enhanced_convolution(signal, kernel, output)
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_enhanced_convolution(
    signal: &[f64],
    kernel: &[f64],
    output: &mut [f64]
) -> SignalResult<()> {
    super::scalar_enhanced_convolution(signal, kernel, output)
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_complex_multiply(
    a_real: &[f64],
    a_imag: &[f64],
    b_real: &[f64],
    b_imag: &[f64],
    result_real: &mut [f64],
    result_imag: &mut [f64],
) -> SignalResult<()> {
    super::scalar_complex_multiply(a_real, a_imag, b_real, b_imag, result_real, result_imag)
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_complex_multiply(
    a_real: &[f64],
    a_imag: &[f64],
    b_real: &[f64],
    b_imag: &[f64],
    result_real: &mut [f64],
    result_imag: &mut [f64],
) -> SignalResult<()> {
    super::scalar_complex_multiply(a_real, a_imag, b_real, b_imag, result_real, result_imag)
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_power_spectrum(real: &[f64], imag: &[f64], power: &mut [f64]) -> SignalResult<()> {
    super::scalar_power_spectrum(real, imag, power)
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_power_spectrum(real: &[f64], imag: &[f64], power: &mut [f64]) -> SignalResult<()> {
    super::scalar_power_spectrum(real, imag, power)
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_weighted_average_spectra(
    spectra: &[&[f64]],
    weights: &[f64],
    result: &mut [f64],
) -> SignalResult<()> {
    super::scalar_weighted_average_spectra(spectra, weights, result)
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_weighted_average_spectra(
    spectra: &[&[f64]],
    weights: &[f64],
    result: &mut [f64],
) -> SignalResult<()> {
    super::scalar_weighted_average_spectra(spectra, weights, result)
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_apply_window(signal: &[f64], window: &[f64], result: &mut [f64]) -> SignalResult<()> {
    super::scalar_apply_window(signal, window, result)
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_apply_window(signal: &[f64], window: &[f64], result: &mut [f64]) -> SignalResult<()> {
    super::scalar_apply_window(signal, window, result)
}

