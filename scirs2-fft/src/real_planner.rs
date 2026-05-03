//! Real FFT planner with trait object support
//!
//! This module provides trait object interfaces for real-to-complex and complex-to-real
//! FFT operations, matching the API patterns used by realfft crate for easier migration.
//! Uses OxiFFT as the backend (COOLJAPAN Pure Rust policy).
//!
//! # Features
//!
//! - `RealToComplex` trait for forward real-to-complex FFT operations
//! - `ComplexToReal` trait for inverse complex-to-real FFT operations
//! - `RealFftPlanner` for creating and caching FFT plans
//! - Support for both f32 and f64 precision
//! - Thread-safe plan caching with `Arc<dyn Trait>`
//!
//! # Examples
//!
//! ```
//! use scirs2_fft::real_planner::{RealFftPlanner, RealToComplex, ComplexToReal};
//! use std::sync::Arc;
//!
//! // Create a planner
//! let mut planner = RealFftPlanner::<f64>::new();
//!
//! // Plan forward FFT
//! let forward_fft = planner.plan_fft_forward(1024);
//!
//! // Plan inverse FFT
//! let inverse_fft = planner.plan_fft_inverse(1024);
//!
//! // Use in struct (common VoiRS pattern)
//! struct AudioProcessor {
//!     forward: Arc<dyn RealToComplex<f64>>,
//!     backward: Arc<dyn ComplexToReal<f64>>,
//! }
//! ```

use crate::error::{FFTError, FFTResult};
#[cfg(feature = "oxifft")]
use crate::oxifft_plan_cache;
#[cfg(feature = "oxifft")]
use oxifft::{Complex as OxiComplex, Direction};
use scirs2_core::numeric::Complex;
use scirs2_core::numeric::Float;

/// Trait for real-to-complex FFT operations
///
/// This trait defines the interface for forward FFT transforms that convert
/// real-valued input data to complex-valued frequency domain output.
pub trait RealToComplex<T: Float>: Send + Sync {
    /// Process a real-valued input and produce complex-valued output
    ///
    /// # Arguments
    ///
    /// * `input` - Real-valued input samples
    /// * `output` - Complex-valued frequency domain output (length = input.len()/2 + 1)
    fn process(&self, input: &[T], output: &mut [Complex<T>]) -> FFTResult<()>;

    /// Get the length of the input this FFT is configured for
    fn len(&self) -> usize;

    /// Check if this FFT is empty (length 0)
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the length of the output this FFT produces
    fn output_len(&self) -> usize {
        self.len() / 2 + 1
    }
}

/// Trait for complex-to-real FFT operations
///
/// This trait defines the interface for inverse FFT transforms that convert
/// complex-valued frequency domain data back to real-valued time domain output.
pub trait ComplexToReal<T: Float>: Send + Sync {
    /// Process a complex-valued input and produce real-valued output
    ///
    /// # Arguments
    ///
    /// * `input` - Complex-valued frequency domain samples (length = output.len()/2 + 1)
    /// * `output` - Real-valued time domain output
    fn process(&self, input: &[Complex<T>], output: &mut [T]) -> FFTResult<()>;

    /// Get the length of the output this IFFT is configured for
    fn len(&self) -> usize;

    /// Check if this IFFT is empty (length 0)
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the length of the input this IFFT expects
    fn input_len(&self) -> usize {
        self.len() / 2 + 1
    }
}

/// Real FFT plan implementation for f64 using OxiFFT backend
struct RealFftPlanF64 {
    length: usize,
}

impl RealFftPlanF64 {
    fn new(length: usize) -> Self {
        Self { length }
    }
}

impl RealToComplex<f64> for RealFftPlanF64 {
    fn process(&self, input: &[f64], output: &mut [Complex<f64>]) -> FFTResult<()> {
        // Validate input/output lengths
        if input.len() != self.length {
            return Err(FFTError::ValueError(format!(
                "Input length {} doesn't match plan length {}",
                input.len(),
                self.length
            )));
        }
        if output.len() != self.output_len() {
            return Err(FFTError::ValueError(format!(
                "Output length {} doesn't match expected length {}",
                output.len(),
                self.output_len()
            )));
        }

        #[cfg(feature = "oxifft")]
        {
            // Convert real input to complex for full FFT
            let input_oxi: Vec<OxiComplex<f64>> =
                input.iter().map(|&x| OxiComplex::new(x, 0.0)).collect();
            let mut output_oxi: Vec<OxiComplex<f64>> = vec![OxiComplex::new(0.0, 0.0); self.length];

            oxifft_plan_cache::execute_c2c(&input_oxi, &mut output_oxi, Direction::Forward)?;

            // Copy first n/2 + 1 elements to output (real FFT Hermitian symmetry property)
            let out_len = self.output_len();
            for (i, dst) in output.iter_mut().enumerate().take(out_len) {
                *dst = Complex::new(output_oxi[i].re, output_oxi[i].im);
            }
        }

        #[cfg(not(feature = "oxifft"))]
        {
            // Fallback: zero-fill output when no backend available
            for dst in output.iter_mut() {
                *dst = Complex::new(0.0, 0.0);
            }
        }

        Ok(())
    }

    fn len(&self) -> usize {
        self.length
    }
}

/// Inverse real FFT plan implementation for f64 using OxiFFT backend
struct InverseRealFftPlanF64 {
    length: usize,
}

impl InverseRealFftPlanF64 {
    fn new(length: usize) -> Self {
        Self { length }
    }
}

impl ComplexToReal<f64> for InverseRealFftPlanF64 {
    fn process(&self, input: &[Complex<f64>], output: &mut [f64]) -> FFTResult<()> {
        // Validate input/output lengths
        if input.len() != self.input_len() {
            return Err(FFTError::ValueError(format!(
                "Input length {} doesn't match expected length {}",
                input.len(),
                self.input_len()
            )));
        }
        if output.len() != self.length {
            return Err(FFTError::ValueError(format!(
                "Output length {} doesn't match plan length {}",
                output.len(),
                self.length
            )));
        }

        #[cfg(feature = "oxifft")]
        {
            // Reconstruct full spectrum using Hermitian symmetry
            let mut buffer_oxi: Vec<OxiComplex<f64>> = Vec::with_capacity(self.length);

            // Add the provided half-spectrum
            for &c in input.iter() {
                buffer_oxi.push(OxiComplex::new(c.re, c.im));
            }

            // Add conjugate symmetric part
            let start_idx = if self.length % 2 == 0 {
                input.len() - 1
            } else {
                input.len()
            };

            for i in (1..start_idx).rev() {
                buffer_oxi.push(OxiComplex::new(input[i].re, -input[i].im));
            }

            // Pad to full length if needed
            while buffer_oxi.len() < self.length {
                buffer_oxi.push(OxiComplex::new(0.0, 0.0));
            }

            let mut out_oxi: Vec<OxiComplex<f64>> = vec![OxiComplex::new(0.0, 0.0); self.length];

            oxifft_plan_cache::execute_c2c(&buffer_oxi, &mut out_oxi, Direction::Backward)?;

            // Extract real parts and normalize
            let scale = 1.0 / self.length as f64;
            for (i, dst) in output.iter_mut().enumerate() {
                *dst = out_oxi[i].re * scale;
            }
        }

        #[cfg(not(feature = "oxifft"))]
        {
            for dst in output.iter_mut() {
                *dst = 0.0;
            }
        }

        Ok(())
    }

    fn len(&self) -> usize {
        self.length
    }
}

/// Real FFT plan implementation for f32 using OxiFFT backend
///
/// OxiFFT operates on f64 internally; f32 input/output is converted.
struct RealFftPlanF32 {
    length: usize,
}

impl RealFftPlanF32 {
    fn new(length: usize) -> Self {
        Self { length }
    }
}

impl RealToComplex<f32> for RealFftPlanF32 {
    fn process(&self, input: &[f32], output: &mut [Complex<f32>]) -> FFTResult<()> {
        // Validate input/output lengths
        if input.len() != self.length {
            return Err(FFTError::ValueError(format!(
                "Input length {} doesn't match plan length {}",
                input.len(),
                self.length
            )));
        }
        if output.len() != self.output_len() {
            return Err(FFTError::ValueError(format!(
                "Output length {} doesn't match expected length {}",
                output.len(),
                self.output_len()
            )));
        }

        #[cfg(feature = "oxifft")]
        {
            // Convert f32 real input to f64 complex for OxiFFT
            let input_oxi: Vec<OxiComplex<f64>> = input
                .iter()
                .map(|&x| OxiComplex::new(x as f64, 0.0))
                .collect();
            let mut output_oxi: Vec<OxiComplex<f64>> = vec![OxiComplex::new(0.0, 0.0); self.length];

            oxifft_plan_cache::execute_c2c(&input_oxi, &mut output_oxi, Direction::Forward)?;

            // Copy first n/2 + 1 elements with f64->f32 conversion
            let out_len = self.output_len();
            for (i, dst) in output.iter_mut().enumerate().take(out_len) {
                *dst = Complex::new(output_oxi[i].re as f32, output_oxi[i].im as f32);
            }
        }

        #[cfg(not(feature = "oxifft"))]
        {
            for dst in output.iter_mut() {
                *dst = Complex::new(0.0f32, 0.0f32);
            }
        }

        Ok(())
    }

    fn len(&self) -> usize {
        self.length
    }
}

/// Inverse real FFT plan implementation for f32 using OxiFFT backend
struct InverseRealFftPlanF32 {
    length: usize,
}

impl InverseRealFftPlanF32 {
    fn new(length: usize) -> Self {
        Self { length }
    }
}

impl ComplexToReal<f32> for InverseRealFftPlanF32 {
    fn process(&self, input: &[Complex<f32>], output: &mut [f32]) -> FFTResult<()> {
        // Validate input/output lengths
        if input.len() != self.input_len() {
            return Err(FFTError::ValueError(format!(
                "Input length {} doesn't match expected length {}",
                input.len(),
                self.input_len()
            )));
        }
        if output.len() != self.length {
            return Err(FFTError::ValueError(format!(
                "Output length {} doesn't match plan length {}",
                output.len(),
                self.length
            )));
        }

        #[cfg(feature = "oxifft")]
        {
            // Reconstruct full spectrum using Hermitian symmetry (with f32->f64 conversion)
            let mut buffer_oxi: Vec<OxiComplex<f64>> = Vec::with_capacity(self.length);

            for &c in input.iter() {
                buffer_oxi.push(OxiComplex::new(c.re as f64, c.im as f64));
            }

            let start_idx = if self.length % 2 == 0 {
                input.len() - 1
            } else {
                input.len()
            };

            for i in (1..start_idx).rev() {
                buffer_oxi.push(OxiComplex::new(input[i].re as f64, -(input[i].im as f64)));
            }

            while buffer_oxi.len() < self.length {
                buffer_oxi.push(OxiComplex::new(0.0, 0.0));
            }

            let mut out_oxi: Vec<OxiComplex<f64>> = vec![OxiComplex::new(0.0, 0.0); self.length];

            oxifft_plan_cache::execute_c2c(&buffer_oxi, &mut out_oxi, Direction::Backward)?;

            // Extract real parts, normalize, and convert back to f32
            let scale = 1.0 / self.length as f64;
            for (i, dst) in output.iter_mut().enumerate() {
                *dst = (out_oxi[i].re * scale) as f32;
            }
        }

        #[cfg(not(feature = "oxifft"))]
        {
            for dst in output.iter_mut() {
                *dst = 0.0f32;
            }
        }

        Ok(())
    }

    fn len(&self) -> usize {
        self.length
    }
}

/// Real FFT planner for creating and managing FFT plans
///
/// This planner creates reusable FFT plans optimized for real-valued input/output.
/// Plans are thread-safe and can be shared across threads using Arc.
/// Uses OxiFFT as the backend (COOLJAPAN Pure Rust policy).
///
/// # Type Parameters
///
/// * `T` - Float type (f32 or f64)
///
/// # Examples
///
/// ```
/// use scirs2_fft::real_planner::RealFftPlanner;
///
/// let mut planner = RealFftPlanner::<f64>::new();
/// let forward = planner.plan_fft_forward(1024);
/// let inverse = planner.plan_fft_inverse(1024);
/// ```
pub struct RealFftPlanner<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl RealFftPlanner<f64> {
    /// Create a new planner for f64 precision
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a forward FFT plan for real-to-complex transformation
    ///
    /// # Arguments
    ///
    /// * `length` - Length of the input real-valued array
    ///
    /// # Returns
    ///
    /// Arc-wrapped trait object implementing RealToComplex
    pub fn plan_fft_forward(&mut self, length: usize) -> std::sync::Arc<dyn RealToComplex<f64>> {
        std::sync::Arc::new(RealFftPlanF64::new(length))
    }

    /// Create an inverse FFT plan for complex-to-real transformation
    ///
    /// # Arguments
    ///
    /// * `length` - Length of the output real-valued array
    ///
    /// # Returns
    ///
    /// Arc-wrapped trait object implementing ComplexToReal
    pub fn plan_fft_inverse(&mut self, length: usize) -> std::sync::Arc<dyn ComplexToReal<f64>> {
        std::sync::Arc::new(InverseRealFftPlanF64::new(length))
    }
}

impl Default for RealFftPlanner<f64> {
    fn default() -> Self {
        Self::new()
    }
}

impl RealFftPlanner<f32> {
    /// Create a new planner for f32 precision
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a forward FFT plan for real-to-complex transformation
    ///
    /// # Arguments
    ///
    /// * `length` - Length of the input real-valued array
    ///
    /// # Returns
    ///
    /// Arc-wrapped trait object implementing RealToComplex
    pub fn plan_fft_forward(&mut self, length: usize) -> std::sync::Arc<dyn RealToComplex<f32>> {
        std::sync::Arc::new(RealFftPlanF32::new(length))
    }

    /// Create an inverse FFT plan for complex-to-real transformation
    ///
    /// # Arguments
    ///
    /// * `length` - Length of the output real-valued array
    ///
    /// # Returns
    ///
    /// Arc-wrapped trait object implementing ComplexToReal
    pub fn plan_fft_inverse(&mut self, length: usize) -> std::sync::Arc<dyn ComplexToReal<f32>> {
        std::sync::Arc::new(InverseRealFftPlanF32::new(length))
    }
}

impl Default for RealFftPlanner<f32> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::numeric::Complex64;
    use std::f64::consts::PI;

    #[test]
    fn test_real_fft_planner_f64() {
        let mut planner = RealFftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(8);
        let inverse = planner.plan_fft_inverse(8);

        // Test input
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut spectrum = vec![Complex64::new(0.0, 0.0); 5]; // 8/2 + 1 = 5

        // Forward transform
        forward
            .process(&input, &mut spectrum)
            .expect("Forward FFT failed");

        // Check DC component
        let sum: f64 = input.iter().sum();
        assert!((spectrum[0].re - sum).abs() < 1e-10);
        assert!(spectrum[0].im.abs() < 1e-10);

        // Inverse transform
        let mut recovered = vec![0.0; 8];
        inverse
            .process(&spectrum, &mut recovered)
            .expect("Inverse FFT failed");

        // Check round-trip accuracy
        for (i, (&orig, &recov)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - recov).abs() < 1e-10,
                "Mismatch at index {}: {} vs {}",
                i,
                orig,
                recov
            );
        }
    }

    #[test]
    fn test_real_fft_planner_f32() {
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(8);
        let inverse = planner.plan_fft_inverse(8);

        // Test input
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut spectrum = vec![Complex::new(0.0f32, 0.0); 5]; // 8/2 + 1 = 5

        // Forward transform
        forward
            .process(&input, &mut spectrum)
            .expect("Forward FFT failed");

        // Inverse transform
        let mut recovered = vec![0.0f32; 8];
        inverse
            .process(&spectrum, &mut recovered)
            .expect("Inverse FFT failed");

        // Check round-trip accuracy (lower precision for f32)
        for (i, (&orig, &recov)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - recov).abs() < 1e-5,
                "Mismatch at index {}: {} vs {}",
                i,
                orig,
                recov
            );
        }
    }

    #[test]
    fn test_sine_wave_fft() {
        let mut planner = RealFftPlanner::<f64>::new();
        let length = 128;
        let forward = planner.plan_fft_forward(length);

        // Generate sine wave at frequency bin 5
        let freq_index = 5;
        let input: Vec<f64> = (0..length)
            .map(|i| (2.0 * PI * freq_index as f64 * i as f64 / length as f64).sin())
            .collect();

        let mut spectrum = vec![Complex64::new(0.0, 0.0); length / 2 + 1];
        forward.process(&input, &mut spectrum).expect("FFT failed");

        // Check that energy is concentrated at the expected frequency
        let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();

        // Find peak
        let (peak_idx, &peak_mag) = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
            .expect("Operation failed");

        assert_eq!(
            peak_idx, freq_index,
            "Peak should be at frequency index {}",
            freq_index
        );
        assert!(peak_mag > length as f64 / 4.0, "Peak magnitude too small");
    }

    #[test]
    fn test_plan_properties() {
        let mut planner = RealFftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(1024);

        assert_eq!(forward.len(), 1024);
        assert_eq!(forward.output_len(), 513); // 1024/2 + 1
        assert!(!forward.is_empty());
    }

    #[test]
    fn test_voirs_usage_pattern() {
        // This test demonstrates the VoiRS usage pattern with Arc<dyn Trait>
        struct AudioProcessor {
            forward: std::sync::Arc<dyn RealToComplex<f64>>,
            backward: std::sync::Arc<dyn ComplexToReal<f64>>,
        }

        impl AudioProcessor {
            fn new(size: usize) -> Self {
                let mut planner = RealFftPlanner::<f64>::new();
                Self {
                    forward: planner.plan_fft_forward(size),
                    backward: planner.plan_fft_inverse(size),
                }
            }

            fn process(&self, input: &[f64]) -> Vec<f64> {
                let mut spectrum = vec![Complex64::new(0.0, 0.0); self.forward.output_len()];
                self.forward
                    .process(input, &mut spectrum)
                    .expect("Forward FFT failed");

                let mut output = vec![0.0; self.backward.len()];
                self.backward
                    .process(&spectrum, &mut output)
                    .expect("Inverse FFT failed");

                output
            }
        }

        let processor = AudioProcessor::new(16);
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let output = processor.process(&input);

        // Verify round-trip
        for (i, (&orig, &recov)) in input.iter().zip(output.iter()).enumerate() {
            assert!((orig - recov).abs() < 1e-10, "Mismatch at {}", i);
        }
    }
}
