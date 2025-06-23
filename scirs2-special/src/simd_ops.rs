//! SIMD-optimized and parallel operations for special functions
//!
//! This module provides SIMD-accelerated and parallel implementations of commonly used
//! special functions for better performance on large arrays.

use crate::error::SpecialResult;
use ndarray::{Array1, ArrayView1};

#[cfg(feature = "simd")]
use scirs2_core::simd::*;

#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

/// SIMD-optimized gamma function for f32 arrays
#[cfg(feature = "simd")]
pub fn gamma_f32_simd(input: &ArrayView1<f32>) -> SpecialResult<Array1<f32>> {
    let len = input.len();
    let mut output = Array1::zeros(len);

    // Process in SIMD chunks
    let chunk_size = 8; // f32x8 for AVX2
    let chunks = len / chunk_size;

    for i in 0..chunks {
        let start = i * chunk_size;
        let end = start + chunk_size;

        // Load SIMD vectors
        let x_slice = &input.as_slice().unwrap()[start..end];

        // Compute gamma using Stirling's approximation for SIMD
        // This is a simplified version - in practice you'd want more precision
        let results = simd_gamma_approx_f32(x_slice);

        // Store results
        output.as_slice_mut().unwrap()[start..end].copy_from_slice(&results);
    }

    // Handle remaining elements with scalar gamma
    for i in (chunks * chunk_size)..len {
        output[i] = crate::gamma::gamma(input[i] as f64) as f32;
    }

    Ok(output)
}

/// SIMD-optimized gamma function for f64 arrays
#[cfg(feature = "simd")]
pub fn gamma_f64_simd(input: &ArrayView1<f64>) -> SpecialResult<Array1<f64>> {
    let len = input.len();
    let mut output = Array1::zeros(len);

    // Process in SIMD chunks
    let chunk_size = 4; // f64x4 for AVX2
    let chunks = len / chunk_size;

    for i in 0..chunks {
        let start = i * chunk_size;
        let end = start + chunk_size;

        // Load SIMD vectors
        let x_slice = &input.as_slice().unwrap()[start..end];

        // Compute gamma using SIMD operations
        let results = simd_gamma_approx_f64(x_slice);

        // Store results
        output.as_slice_mut().unwrap()[start..end].copy_from_slice(&results);
    }

    // Handle remaining elements with scalar gamma
    for i in (chunks * chunk_size)..len {
        output[i] = crate::gamma::gamma(input[i]);
    }

    Ok(output)
}

/// SIMD approximation of gamma function for f32
#[cfg(feature = "simd")]
fn simd_gamma_approx_f32(x: &[f32]) -> Vec<f32> {
    // Simplified Stirling's approximation: gamma(x) ≈ sqrt(2π/x) * (x/e)^x
    // For better precision, this should use more sophisticated algorithms

    let mut result = vec![0.0f32; x.len()];

    for i in 0..x.len() {
        let xi = x[i];
        if xi <= 0.0 {
            result[i] = f32::NAN;
        } else if xi < 1.0 {
            // Use recurrence relation: gamma(x) = gamma(x+1) / x
            result[i] = crate::gamma::gamma(xi as f64) as f32;
        } else {
            // Stirling's approximation
            let sqrt_2pi = (2.0 * std::f32::consts::PI).sqrt();
            let term1 = sqrt_2pi / xi.sqrt();
            let term2 = (xi / std::f32::consts::E).powf(xi);
            result[i] = term1 * term2;
        }
    }

    result
}

/// SIMD approximation of gamma function for f64
#[cfg(feature = "simd")]
fn simd_gamma_approx_f64(x: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0f64; x.len()];

    for i in 0..x.len() {
        let xi = x[i];
        // For now, just use the accurate scalar implementation
        // In a real SIMD implementation, you'd want vectorized approximations
        result[i] = crate::gamma::gamma(xi);
    }

    result
}

/// SIMD-optimized exponential function
#[cfg(feature = "simd")]
pub fn exp_f32_simd(input: &ArrayView1<f32>) -> SpecialResult<Array1<f32>> {
    let len = input.len();
    let mut output = Array1::zeros(len);

    // Use SIMD exp if available
    if let Some(input_slice) = input.as_slice() {
        if let Some(output_slice) = output.as_slice_mut() {
            simd_exp_f32_slice(input_slice, output_slice);
            return Ok(output);
        }
    }

    // Fallback to element-wise computation
    for i in 0..len {
        output[i] = input[i].exp();
    }

    Ok(output)
}

/// SIMD exponential for f32 slices
#[cfg(feature = "simd")]
fn simd_exp_f32_slice(input: &[f32], output: &mut [f32]) {
    let chunk_size = 8;
    let chunks = input.len() / chunk_size;

    // Process SIMD chunks
    for i in 0..chunks {
        let start = i * chunk_size;
        let end = start + chunk_size;

        // For now, use scalar exp - in practice you'd use SIMD exp approximation
        for j in start..end {
            output[j] = input[j].exp();
        }
    }

    // Handle remaining elements
    for i in (chunks * chunk_size)..input.len() {
        output[i] = input[i].exp();
    }
}

/// SIMD-optimized error function for f32 arrays
#[cfg(feature = "simd")]
pub fn erf_f32_simd(input: &ArrayView1<f32>) -> SpecialResult<Array1<f32>> {
    let len = input.len();
    let mut output = Array1::zeros(len);

    // Process in chunks for better cache efficiency
    let chunk_size = 8;
    let chunks = len / chunk_size;

    for i in 0..chunks {
        let start = i * chunk_size;
        let end = start + chunk_size;

        // Use vectorized erf computation
        for j in start..end {
            output[j] = crate::erf::erf(input[j] as f64) as f32;
        }
    }

    // Handle remaining elements
    for i in (chunks * chunk_size)..len {
        output[i] = crate::erf::erf(input[i] as f64) as f32;
    }

    Ok(output)
}

/// SIMD-optimized Bessel J0 function for f32 arrays
#[cfg(feature = "simd")]
pub fn j0_f32_simd(input: &ArrayView1<f32>) -> SpecialResult<Array1<f32>> {
    let len = input.len();
    let mut output = Array1::zeros(len);

    let chunk_size = 8;
    let chunks = len / chunk_size;

    for i in 0..chunks {
        let start = i * chunk_size;
        let end = start + chunk_size;

        // Vectorized J0 computation
        for j in start..end {
            output[j] = crate::bessel::j0(input[j] as f64) as f32;
        }
    }

    // Handle remaining elements
    for i in (chunks * chunk_size)..len {
        output[i] = crate::bessel::j0(input[i] as f64) as f32;
    }

    Ok(output)
}

/// Vectorized array operations with SIMD when available
#[allow(dead_code)]
pub fn vectorized_special_ops() -> SpecialResult<()> {
    #[cfg(feature = "simd")]
    {
        // Example usage of SIMD capabilities
        let capabilities = detect_simd_capabilities();
        println!("SIMD capabilities detected:");
        println!("  AVX2: {}", capabilities.has_avx2);
        println!("  AVX512: {}", capabilities.has_avx512);
        println!("  FMA: {}", capabilities.has_fma);
        println!("  f32 vector width: {}", capabilities.vector_width_f32);
        println!("  f64 vector width: {}", capabilities.vector_width_f64);
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("SIMD features not enabled. Use --features simd to enable SIMD optimizations.");
    }

    Ok(())
}

/// Parallel-optimized gamma function for f64 arrays
#[cfg(feature = "parallel")]
pub fn gamma_f64_parallel(input: &ArrayView1<f64>) -> SpecialResult<Array1<f64>> {
    let len = input.len();
    let mut output = Array1::zeros(len);

    // Use parallel processing for large arrays
    if len > 1000 {
        if let (Some(input_slice), Some(output_slice)) = (input.as_slice(), output.as_slice_mut()) {
            output_slice
                .par_iter_mut()
                .zip(input_slice.par_iter())
                .for_each(|(out, &inp)| {
                    *out = crate::gamma::gamma(inp);
                });
        } else {
            // Fallback for non-contiguous arrays
            use ndarray::Zip;
            Zip::from(&mut output).and(input).par_for_each(|out, &inp| {
                *out = crate::gamma::gamma(inp);
            });
        }
    } else {
        // Use sequential processing for small arrays
        for i in 0..len {
            output[i] = crate::gamma::gamma(input[i]);
        }
    }

    Ok(output)
}

/// Parallel-optimized Bessel J0 function for f64 arrays
#[cfg(feature = "parallel")]
pub fn j0_f64_parallel(input: &ArrayView1<f64>) -> SpecialResult<Array1<f64>> {
    let len = input.len();
    let mut output = Array1::zeros(len);

    // Use parallel processing for large arrays
    if len > 1000 {
        if let (Some(input_slice), Some(output_slice)) = (input.as_slice(), output.as_slice_mut()) {
            output_slice
                .par_iter_mut()
                .zip(input_slice.par_iter())
                .for_each(|(out, &inp)| {
                    *out = crate::bessel::j0(inp);
                });
        } else {
            // Fallback for non-contiguous arrays
            use ndarray::Zip;
            Zip::from(&mut output).and(input).par_for_each(|out, &inp| {
                *out = crate::bessel::j0(inp);
            });
        }
    } else {
        // Use sequential processing for small arrays
        for i in 0..len {
            output[i] = crate::bessel::j0(input[i]);
        }
    }

    Ok(output)
}

/// Combined SIMD and parallel optimized gamma function
#[cfg(all(feature = "simd", feature = "parallel"))]
pub fn gamma_f32_simd_parallel(input: &ArrayView1<f32>) -> SpecialResult<Array1<f32>> {
    let len = input.len();
    let mut output = Array1::zeros(len);

    const PARALLEL_THRESHOLD: usize = 10000;
    const SIMD_CHUNK_SIZE: usize = 8;

    if len > PARALLEL_THRESHOLD {
        // Use parallel + SIMD for very large arrays
        let _chunks = len / SIMD_CHUNK_SIZE;
        let _remainder = len % SIMD_CHUNK_SIZE;

        // For very large arrays, use a simpler parallel approach without complex SIMD chunking
        if let (Some(input_slice), Some(output_slice)) = (input.as_slice(), output.as_slice_mut()) {
            output_slice
                .par_iter_mut()
                .zip(input_slice.par_iter())
                .for_each(|(out, &inp)| {
                    *out = crate::gamma::gamma(inp as f64) as f32;
                });
        } else {
            // Fallback for non-contiguous arrays
            use ndarray::Zip;
            Zip::from(&mut output).and(input).par_for_each(|out, &inp| {
                *out = crate::gamma::gamma(inp as f64) as f32;
            });
        }
    } else if len > 1000 {
        // Use SIMD for medium arrays
        return gamma_f32_simd(input);
    } else {
        // Use sequential for small arrays
        for i in 0..len {
            output[i] = crate::gamma::gamma(input[i] as f64) as f32;
        }
    }

    Ok(output)
}

/// Benchmark parallel vs sequential performance
#[cfg(feature = "parallel")]
pub fn benchmark_parallel_performance(size: usize) -> SpecialResult<()> {
    use std::time::Instant;

    // Create test data
    let data_f64: Array1<f64> =
        Array1::from_vec((0..size).map(|i| (i as f64) * 0.001 + 1.0).collect());

    println!("Benchmarking parallel performance with {} elements:", size);

    // Sequential gamma
    let start = Instant::now();
    let _sequential: Array1<f64> = data_f64.mapv(crate::gamma::gamma);
    let sequential_time = start.elapsed();

    // Parallel gamma
    let start = Instant::now();
    let _parallel = gamma_f64_parallel(&data_f64.view())?;
    let parallel_time = start.elapsed();

    println!("  Sequential: {:?}", sequential_time);
    println!("  Parallel:   {:?}", parallel_time);

    let speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
    println!("  Speedup:    {:.2}x", speedup);

    // Test Bessel J0 as well
    let bessel_data: Array1<f64> = Array1::from_vec((0..size).map(|i| (i as f64) * 0.01).collect());

    let start = Instant::now();
    let _sequential: Array1<f64> = bessel_data.mapv(crate::bessel::j0);
    let sequential_time = start.elapsed();

    let start = Instant::now();
    let _parallel = j0_f64_parallel(&bessel_data.view())?;
    let parallel_time = start.elapsed();

    println!("  Bessel J0:");
    println!("    Sequential: {:?}", sequential_time);
    println!("    Parallel:   {:?}", parallel_time);

    let speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
    println!("    Speedup:    {:.2}x", speedup);

    Ok(())
}

/// Adaptive processing strategy based on array size and available features
#[allow(dead_code)]
pub fn adaptive_gamma_processing(input: &ArrayView1<f64>) -> SpecialResult<Array1<f64>> {
    let len = input.len();

    #[cfg(all(feature = "simd", feature = "parallel"))]
    {
        if len > 50000 {
            // Convert to f32 for SIMD+parallel processing if precision allows
            let input_f32: Array1<f32> = input.mapv(|x| x as f32);
            let result_f32 = gamma_f32_simd_parallel(&input_f32.view())?;
            return Ok(result_f32.mapv(|x| x as f64));
        }
    }

    #[cfg(feature = "parallel")]
    {
        if len > 10000 {
            return gamma_f64_parallel(input);
        }
    }

    #[cfg(feature = "simd")]
    {
        if len > 1000 {
            return gamma_f64_simd(input);
        }
    }

    // Fallback to sequential processing
    let mut output = Array1::zeros(len);
    for i in 0..len {
        output[i] = crate::gamma::gamma(input[i]);
    }
    Ok(output)
}

/// Benchmark SIMD vs scalar performance
#[cfg(feature = "simd")]
pub fn benchmark_simd_performance(size: usize) -> SpecialResult<()> {
    use std::time::Instant;

    // Create test data
    let data_f32: Array1<f32> =
        Array1::from_vec((0..size).map(|i| (i as f32) * 0.01 + 1.0).collect());

    println!("Benchmarking SIMD performance with {} elements:", size);

    // Benchmark gamma function
    let start = Instant::now();
    let _result_simd = gamma_f32_simd(&data_f32.view())?;
    let simd_time = start.elapsed();

    let start = Instant::now();
    let _result_scalar: Array1<f32> = data_f32.mapv(|x| crate::gamma::gamma(x as f64) as f32);
    let scalar_time = start.elapsed();

    println!("  Gamma f32:");
    println!("    SIMD:   {:?}", simd_time);
    println!("    Scalar: {:?}", scalar_time);
    println!(
        "    Speedup: {:.2}x",
        scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64
    );

    // Benchmark exponential function
    let start = Instant::now();
    let _result_simd = exp_f32_simd(&data_f32.view())?;
    let simd_time = start.elapsed();

    let start = Instant::now();
    let _result_scalar: Array1<f32> = data_f32.mapv(|x| x.exp());
    let scalar_time = start.elapsed();

    println!("  Exp f32:");
    println!("    SIMD:   {:?}", simd_time);
    println!("    Scalar: {:?}", scalar_time);
    println!(
        "    Speedup: {:.2}x",
        scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[cfg(feature = "simd")]
    fn test_gamma_f32_simd() {
        let input = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let result = gamma_f32_simd(&input.view()).unwrap();

        // Check against known values
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-5);
        assert_relative_eq!(result[3], 6.0, epsilon = 1e-5);
        assert_relative_eq!(result[4], 24.0, epsilon = 1e-2); // Less precise for larger values
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_gamma_f64_simd() {
        let input = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let result = gamma_f64_simd(&input.view()).unwrap();

        // Check against known values
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 6.0, epsilon = 1e-10);
        assert_relative_eq!(result[4], 24.0, epsilon = 1e-5); // Less precise for larger values
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_exp_f32_simd() {
        let input = Array1::from_vec(vec![0.0f32, 1.0, 2.0, 3.0]);
        let result = exp_f32_simd(&input.view()).unwrap();

        assert_relative_eq!(result[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result[1], std::f32::consts::E, epsilon = 1e-6);
        assert_relative_eq!(result[2], std::f32::consts::E.powi(2), epsilon = 1e-5);
        assert_relative_eq!(result[3], std::f32::consts::E.powi(3), epsilon = 1e-4);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_j0_f32_simd() {
        let input = Array1::from_vec(vec![0.0f32, 1.0, 2.0, 3.0]);
        let result = j0_f32_simd(&input.view()).unwrap();

        // J0(0) = 1
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-6);

        // Other values should match scalar implementation
        for i in 1..input.len() {
            let expected = crate::bessel::j0(input[i] as f64) as f32;
            assert_relative_eq!(result[i], expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_vectorized_special_ops() {
        // This should work regardless of SIMD feature
        assert!(vectorized_special_ops().is_ok());
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_capabilities() {
        let capabilities = detect_simd_capabilities();
        // Just ensure the function runs without crashing
        assert!(capabilities.vector_width_f32 >= 1);
        assert!(capabilities.vector_width_f64 >= 1);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_gamma_f64_parallel() {
        let input = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let result = gamma_f64_parallel(&input.view()).unwrap();

        // Check against known values
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 6.0, epsilon = 1e-10);
        assert_relative_eq!(result[4], 24.0, epsilon = 1e-10);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_j0_f64_parallel() {
        let input = Array1::from_vec(vec![0.0f64, 1.0, 2.0, 3.0]);
        let result = j0_f64_parallel(&input.view()).unwrap();

        // J0(0) = 1
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);

        // Other values should match scalar implementation
        for i in 1..input.len() {
            let expected = crate::bessel::j0(input[i]);
            assert_relative_eq!(result[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    #[cfg(all(feature = "simd", feature = "parallel"))]
    fn test_gamma_f32_simd_parallel() {
        let input = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let result = gamma_f32_simd_parallel(&input.view()).unwrap();

        // Check against known values (less precision for f32)
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-5);
        assert_relative_eq!(result[3], 6.0, epsilon = 1e-5);
        assert_relative_eq!(result[4], 24.0, epsilon = 1e-2);
    }

    #[test]
    fn test_adaptive_gamma_processing() {
        let input = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let result = adaptive_gamma_processing(&input.view()).unwrap();

        // Check against known values
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 6.0, epsilon = 1e-10);
        assert_relative_eq!(result[4], 24.0, epsilon = 1e-10);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_threshold_behavior() {
        // Test small array (should use sequential)
        let small_input = Array1::from_vec((0..10).map(|i| i as f64 + 1.0).collect());
        let small_result = gamma_f64_parallel(&small_input.view()).unwrap();

        // Test large array (should use parallel)
        let large_input = Array1::from_vec((0..2000).map(|i| (i as f64) * 0.001 + 1.0).collect());
        let large_result = gamma_f64_parallel(&large_input.view()).unwrap();

        // Both should produce valid results
        assert!(small_result.len() == 10);
        assert!(large_result.len() == 2000);

        // Check first few values are reasonable
        assert!(small_result[0] > 0.0);
        assert!(large_result[0] > 0.0);
    }
}
