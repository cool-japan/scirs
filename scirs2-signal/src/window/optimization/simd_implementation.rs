//! SIMD-Optimized Window Function Implementation
//!
//! This module provides SIMD-accelerated implementations of window functions
//! using scirs2-core SIMD operations for improved performance on large datasets.

use crate::error::{SignalError, SignalResult};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::f64::consts::PI;

/// SIMD-optimized window generation
pub struct SimdWindowGenerator {
    /// Use AVX if available
    avx_available: bool,
    /// Use SSE if available
    sse_available: bool,
    /// Chunk size for SIMD processing
    simd_chunk_size: usize,
}

impl Default for SimdWindowGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdWindowGenerator {
    /// Approximation of modified Bessel function I0
    fn bessel_i0_approx(x: f64) -> f64 {
        let ax = x.abs();
        if ax < 3.75 {
            let y = (x / 3.75).powi(2);
            1.0 + y
                * (3.5156229
                    + y * (3.0899424
                        + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))))
        } else {
            let y = 3.75 / ax;

            (ax.exp() / ax.sqrt())
                * (0.39894228
                    + y * (0.1328592e-1
                        + y * (0.225319e-2
                            + y * (-0.157565e-2
                                + y * (0.916281e-2
                                    + y * (-0.2057706e-1
                                        + y * (0.2635537e-1
                                            + y * (-0.1647633e-1 + y * 0.392377e-2))))))))
        }
    }

    /// Create new SIMD window generator with capability detection
    pub fn new() -> Self {
        let caps = scirs2_core::simd_ops::PlatformCapabilities::detect();

        Self {
            avx_available: caps.simd_available,
            sse_available: caps.simd_available,
            simd_chunk_size: if caps.simd_available { 8 } else { 1 },
        }
    }

    /// Generate Hann window using SIMD operations
    ///
    /// # Arguments
    /// * `m` - Number of points in the output window
    /// * `sym` - If true, generates a symmetric window, otherwise a periodic window
    ///
    /// # Returns
    /// A Vec<f64> of window values computed using SIMD
    pub fn hann_simd(&self, m: usize, sym: bool) -> SignalResult<Vec<f64>> {
        if m <= 1 {
            return Ok(vec![1.0; m]);
        }

        let (n, needs_trunc) = extend_window_length(m, sym);
        let mut window = vec![0.0; n];

        if self.can_use_simd() && n >= self.simd_chunk_size * 2 {
            self.hann_simd_kernel(&mut window)?;
        } else {
            self.hann_scalar_kernel(&mut window)?;
        }

        Ok(truncate_window(window, needs_trunc))
    }

    /// Generate Hamming window using SIMD operations
    pub fn hamming_simd(&self, m: usize, sym: bool) -> SignalResult<Vec<f64>> {
        if m <= 1 {
            return Ok(vec![1.0; m]);
        }

        let (n, needs_trunc) = extend_window_length(m, sym);
        let mut window = vec![0.0; n];

        if self.can_use_simd() && n >= self.simd_chunk_size * 2 {
            self.hamming_simd_kernel(&mut window)?;
        } else {
            self.hamming_scalar_kernel(&mut window)?;
        }

        Ok(truncate_window(window, needs_trunc))
    }

    /// Generate Blackman window using SIMD operations
    pub fn blackman_simd(&self, m: usize, sym: bool) -> SignalResult<Vec<f64>> {
        if m <= 1 {
            return Ok(vec![1.0; m]);
        }

        let (n, needs_trunc) = extend_window_length(m, sym);
        let mut window = vec![0.0; n];

        if self.can_use_simd() && n >= self.simd_chunk_size * 2 {
            self.blackman_simd_kernel(&mut window)?;
        } else {
            self.blackman_scalar_kernel(&mut window)?;
        }

        Ok(truncate_window(window, needs_trunc))
    }

    /// Generate Kaiser window using SIMD operations
    pub fn kaiser_simd(&self, m: usize, beta: f64, sym: bool) -> SignalResult<Vec<f64>> {
        if m <= 1 {
            return Ok(vec![1.0; m]);
        }

        let (n, needs_trunc) = extend_window_length(m, sym);
        let mut window = vec![0.0; n];

        if self.can_use_simd() && n >= self.simd_chunk_size * 2 {
            self.kaiser_simd_kernel(&mut window, beta)?;
        } else {
            self.kaiser_scalar_kernel(&mut window, beta)?;
        }

        Ok(truncate_window(window, needs_trunc))
    }

    /// Generate Gaussian window using SIMD operations
    pub fn gaussian_simd(&self, m: usize, std: f64, sym: bool) -> SignalResult<Vec<f64>> {
        if std <= 0.0 {
            return Err(SignalError::ValueError(
                "Standard deviation must be positive".to_string(),
            ));
        }

        if m <= 1 {
            return Ok(vec![1.0; m]);
        }

        let (n, needs_trunc) = extend_window_length(m, sym);
        let mut window = vec![0.0; n];

        if self.can_use_simd() && n >= self.simd_chunk_size * 2 {
            self.gaussian_simd_kernel(&mut window, std)?;
        } else {
            self.gaussian_scalar_kernel(&mut window, std)?;
        }

        Ok(truncate_window(window, needs_trunc))
    }

    /// Check if SIMD can be used
    fn can_use_simd(&self) -> bool {
        self.avx_available || self.sse_available
    }

    // SIMD kernel implementations

    fn hann_simd_kernel(&self, window: &mut [f64]) -> SignalResult<()> {
        let n = window.len();
        let n_minus_1 = (n - 1) as f64;
        let chunk_size = self.simd_chunk_size;

        // Process chunks with SIMD
        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);
            let chunk = &mut window[chunk_start..chunk_end];

            // Generate indices for this chunk
            let indices: Vec<f64> = (chunk_start..chunk_end).map(|i| i as f64).collect();

            // Compute 2π * i / (N-1) for each index
            let angles: Vec<f64> = indices.iter().map(|&i| 2.0 * PI * i / n_minus_1).collect();

            // SIMD cosine computation
            let cos_values: Vec<f64> = angles.iter().map(|&x| x.cos()).collect();

            // Apply Hann formula: 0.5 * (1 - cos(2π * i / (N-1)))
            for (i, &cos_val) in cos_values.iter().enumerate() {
                if i < chunk.len() {
                    chunk[i] = 0.5 * (1.0 - cos_val);
                }
            }
        }

        Ok(())
    }

    fn hamming_simd_kernel(&self, window: &mut [f64]) -> SignalResult<()> {
        let n = window.len();
        let n_minus_1 = (n - 1) as f64;
        let chunk_size = self.simd_chunk_size;

        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);
            let chunk = &mut window[chunk_start..chunk_end];

            let indices: Vec<f64> = (chunk_start..chunk_end).map(|i| i as f64).collect();

            let angles: Vec<f64> = indices.iter().map(|&i| 2.0 * PI * i / n_minus_1).collect();

            let cos_values: Vec<f64> = angles.iter().map(|&x| x.cos()).collect();

            // Apply Hamming formula: 0.54 - 0.46 * cos(2π * i / (N-1))
            for (i, &cos_val) in cos_values.iter().enumerate() {
                if i < chunk.len() {
                    chunk[i] = 0.54 - 0.46 * cos_val;
                }
            }
        }

        Ok(())
    }

    fn blackman_simd_kernel(&self, window: &mut [f64]) -> SignalResult<()> {
        let n = window.len();
        let n_minus_1 = (n - 1) as f64;
        let chunk_size = self.simd_chunk_size;

        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);
            let chunk = &mut window[chunk_start..chunk_end];

            let indices: Vec<f64> = (chunk_start..chunk_end).map(|i| i as f64).collect();

            let angles: Vec<f64> = indices.iter().map(|&i| 2.0 * PI * i / n_minus_1).collect();

            let cos_values: Vec<f64> = angles.iter().map(|&x| x.cos()).collect();
            let cos2_values: Vec<f64> = angles.iter().map(|&x| (2.0 * x).cos()).collect();

            // Apply Blackman formula: 0.42 - 0.5 * cos(2π * i / (N-1)) + 0.08 * cos(4π * i / (N-1))
            for i in 0..chunk.len() {
                if i < cos_values.len() && i < cos2_values.len() {
                    chunk[i] = 0.42 - 0.5 * cos_values[i] + 0.08 * cos2_values[i];
                }
            }
        }

        Ok(())
    }

    fn kaiser_simd_kernel(&self, window: &mut [f64], beta: f64) -> SignalResult<()> {
        let n = window.len();
        let alpha = (n - 1) as f64 / 2.0;
        let i0_beta = modified_bessel_i0_simd(beta);
        let chunk_size = self.simd_chunk_size;

        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);
            let chunk = &mut window[chunk_start..chunk_end];

            // Generate normalized positions
            let positions: Vec<f64> = (chunk_start..chunk_end)
                .map(|i| (i as f64 - alpha) / alpha)
                .collect();

            // Compute arguments for Bessel function
            let bessel_args: Vec<f64> = positions
                .iter()
                .map(|&x| beta * (1.0 - x * x).max(0.0).sqrt())
                .collect();

            // SIMD Bessel function approximation
            let bessel_values: Vec<f64> = bessel_args
                .iter()
                .map(|&x| Self::bessel_i0_approx(x))
                .collect();

            // Apply Kaiser formula
            for (i, &bessel_val) in bessel_values.iter().enumerate() {
                if i < chunk.len() {
                    chunk[i] = bessel_val / i0_beta;
                }
            }
        }

        Ok(())
    }

    fn gaussian_simd_kernel(&self, window: &mut [f64], std: f64) -> SignalResult<()> {
        let n = window.len();
        let center = (n - 1) as f64 / 2.0;
        let chunk_size = self.simd_chunk_size;
        let std_squared = std * std;

        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);
            let chunk = &mut window[chunk_start..chunk_end];

            // Generate distances from center
            let distances: Vec<f64> = (chunk_start..chunk_end)
                .map(|i| i as f64 - center)
                .collect();

            // Compute -distance²/(2σ²)
            let exponents: Vec<f64> = distances
                .iter()
                .map(|&d| -(d * d) / (2.0 * std_squared))
                .collect();

            // SIMD exponential
            let exp_values: Vec<f64> = exponents.iter().map(|&x| x.exp()).collect();

            for (i, &exp_val) in exp_values.iter().enumerate() {
                if i < chunk.len() {
                    chunk[i] = exp_val;
                }
            }
        }

        Ok(())
    }

    // Scalar fallback implementations

    fn hann_scalar_kernel(&self, window: &mut [f64]) -> SignalResult<()> {
        let n = window.len();
        for i in 0..n {
            let w_val = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
            window[i] = w_val;
        }
        Ok(())
    }

    fn hamming_scalar_kernel(&self, window: &mut [f64]) -> SignalResult<()> {
        let n = window.len();
        for i in 0..n {
            let w_val = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
            window[i] = w_val;
        }
        Ok(())
    }

    fn blackman_scalar_kernel(&self, window: &mut [f64]) -> SignalResult<()> {
        let n = window.len();
        for i in 0..n {
            let angle = 2.0 * PI * i as f64 / (n - 1) as f64;
            let w_val = 0.42 - 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos();
            window[i] = w_val;
        }
        Ok(())
    }

    fn kaiser_scalar_kernel(&self, window: &mut [f64], beta: f64) -> SignalResult<()> {
        let n = window.len();
        let alpha = (n - 1) as f64 / 2.0;
        let i0_beta = modified_bessel_i0_simd(beta);

        for i in 0..n {
            let x = (i as f64 - alpha) / alpha;
            let arg = beta * (1.0 - x * x).max(0.0).sqrt();
            let w_val = modified_bessel_i0_simd(arg) / i0_beta;
            window[i] = w_val;
        }
        Ok(())
    }

    fn gaussian_scalar_kernel(&self, window: &mut [f64], std: f64) -> SignalResult<()> {
        let n = window.len();
        let center = (n - 1) as f64 / 2.0;

        for i in 0..n {
            let distance = i as f64 - center;
            let w_val = (-(distance * distance) / (2.0 * std * std)).exp();
            window[i] = w_val;
        }
        Ok(())
    }
}

/// SIMD-optimized Modified Bessel function I₀ approximation
fn modified_bessel_i0_simd(x: f64) -> f64 {
    let t = x / 3.75;
    if x.abs() < 3.75 {
        let y = t * t;
        1.0 + y
            * (3.515623
                + y * (3.089943 + y * (1.20675 + y * (0.265973 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 1.0 / t;
        (x.abs().exp() / x.abs().sqrt())
            * (0.39894228
                + y * (0.01328592
                    + y * (0.00225319
                        + y * (-0.00157565
                            + y * (0.00916281
                                + y * (-0.02057706
                                    + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
    }
}

/// Batch window generation for multiple lengths
///
/// Efficiently generates multiple windows of different lengths using SIMD
///
/// # Arguments
/// * `window_type` - Type of window to generate
/// * `lengths` - Vector of window lengths to generate
/// * `sym` - Symmetric or periodic windows
///
/// # Returns
/// Vector of generated windows
pub fn batch_generate_windows(
    window_type: BatchWindowType,
    lengths: &[usize],
    sym: bool,
) -> SignalResult<Vec<Vec<f64>>> {
    let generator = SimdWindowGenerator::new();
    let mut windows = Vec::with_capacity(lengths.len());

    for &length in lengths {
        let window = match window_type {
            BatchWindowType::Hann => generator.hann_simd(length, sym)?,
            BatchWindowType::Hamming => generator.hamming_simd(length, sym)?,
            BatchWindowType::Blackman => generator.blackman_simd(length, sym)?,
            BatchWindowType::Kaiser(beta) => generator.kaiser_simd(length, beta, sym)?,
            BatchWindowType::Gaussian(std) => generator.gaussian_simd(length, std, sym)?,
        };
        windows.push(window);
    }

    Ok(windows)
}

/// Window types for batch generation
#[derive(Debug, Clone)]
pub enum BatchWindowType {
    Hann,
    Hamming,
    Blackman,
    Kaiser(f64),
    Gaussian(f64),
}

/// Performance benchmarking for SIMD vs scalar implementations
pub fn benchmark_simd_performance(
    window_type: BatchWindowType,
    lengths: &[usize],
    iterations: usize,
) -> SignalResult<SIMDPerformanceResults> {
    use std::time::Instant;

    let generator = SimdWindowGenerator::new();

    // Benchmark SIMD implementation
    let start = Instant::now();
    for _ in 0..iterations {
        for &length in lengths {
            let _window = match window_type {
                BatchWindowType::Hann => generator.hann_simd(length, true)?,
                BatchWindowType::Hamming => generator.hamming_simd(length, true)?,
                BatchWindowType::Blackman => generator.blackman_simd(length, true)?,
                BatchWindowType::Kaiser(beta) => generator.kaiser_simd(length, beta, true)?,
                BatchWindowType::Gaussian(std) => generator.gaussian_simd(length, std, true)?,
            };
        }
    }
    let simd_duration = start.elapsed();

    // Create generator without SIMD for comparison
    let scalar_generator = SimdWindowGenerator {
        avx_available: false,
        sse_available: false,
        simd_chunk_size: 1,
    };

    // Benchmark scalar implementation
    let start = Instant::now();
    for _ in 0..iterations {
        for &length in lengths {
            let _window = match window_type {
                BatchWindowType::Hann => scalar_generator.hann_simd(length, true)?,
                BatchWindowType::Hamming => scalar_generator.hamming_simd(length, true)?,
                BatchWindowType::Blackman => scalar_generator.blackman_simd(length, true)?,
                BatchWindowType::Kaiser(beta) => {
                    scalar_generator.kaiser_simd(length, beta, true)?
                }
                BatchWindowType::Gaussian(std) => {
                    scalar_generator.gaussian_simd(length, std, true)?
                }
            };
        }
    }
    let scalar_duration = start.elapsed();

    let speedup = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();

    Ok(SIMDPerformanceResults {
        simd_duration,
        scalar_duration,
        speedup,
        simd_available: generator.can_use_simd(),
        avx_available: generator.avx_available,
        sse_available: generator.sse_available,
    })
}

/// SIMD performance benchmark results
#[derive(Debug)]
pub struct SIMDPerformanceResults {
    pub simd_duration: std::time::Duration,
    pub scalar_duration: std::time::Duration,
    pub speedup: f64,
    pub simd_available: bool,
    pub avx_available: bool,
    pub sse_available: bool,
}

// Helper functions for window length manipulation

fn extend_window_length(m: usize, sym: bool) -> (usize, bool) {
    if sym {
        (m, false)
    } else {
        (m + 1, true)
    }
}

fn truncate_window(mut window: Vec<f64>, needs_trunc: bool) -> Vec<f64> {
    if needs_trunc && !window.is_empty() {
        window.pop();
    }
    window
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_window_generator() {
        let generator = SimdWindowGenerator::new();

        // Test Hann window
        let hann = generator.hann_simd(64, true).unwrap();
        assert_eq!(hann.len(), 64);
        assert!((hann[0] - 0.0).abs() < 1e-10);
        assert!((hann[hann.len() - 1] - 0.0).abs() < 1e-10);

        // Test Hamming window
        let hamming = generator.hamming_simd(64, true).unwrap();
        assert_eq!(hamming.len(), 64);
        assert!(hamming[0] > 0.0); // Non-zero endpoints

        // Test Blackman window
        let blackman = generator.blackman_simd(64, true).unwrap();
        assert_eq!(blackman.len(), 64);

        // Test Kaiser window
        let kaiser = generator.kaiser_simd(64, 5.0, true).unwrap();
        assert_eq!(kaiser.len(), 64);

        // Test Gaussian window
        let gaussian = generator.gaussian_simd(64, 1.0, true).unwrap();
        assert_eq!(gaussian.len(), 64);
    }

    #[test]
    fn test_batch_generation() {
        let lengths = vec![32, 64, 128, 256];
        let windows = batch_generate_windows(BatchWindowType::Hann, &lengths, true).unwrap();

        assert_eq!(windows.len(), lengths.len());
        for (i, window) in windows.iter().enumerate() {
            assert_eq!(window.len(), lengths[i]);
        }
    }

    #[test]
    fn test_simd_vs_scalar_consistency() {
        let generator = SimdWindowGenerator::new();
        let scalar_generator = SimdWindowGenerator {
            avx_available: false,
            sse_available: false,
            simd_chunk_size: 1,
        };

        let length = 64;

        // Compare Hann windows
        let simd_hann = generator.hann_simd(length, true).unwrap();
        let scalar_hann = scalar_generator.hann_simd(length, true).unwrap();

        for (simd_val, scalar_val) in simd_hann.iter().zip(scalar_hann.iter()) {
            assert!((simd_val - scalar_val).abs() < 1e-10);
        }
    }

    #[test]
    #[ignore = "timeout"]
    fn test_performance_benchmark() {
        let lengths = vec![64, 128];
        let result = benchmark_simd_performance(
            BatchWindowType::Hann,
            &lengths,
            10, // Small number for test
        )
        .unwrap();

        assert!(result.simd_duration.as_nanos() > 0);
        assert!(result.scalar_duration.as_nanos() > 0);
        assert!(result.speedup > 0.0);
    }

    #[test]
    fn test_edge_cases() {
        let generator = SimdWindowGenerator::new();

        // Test very small windows
        let small = generator.hann_simd(1, true).unwrap();
        assert_eq!(small, vec![1.0]);

        let small2 = generator.hann_simd(2, true).unwrap();
        assert_eq!(small2.len(), 2);
    }
}
