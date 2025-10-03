//! Real FFT planner with trait object support
//!
//! This module provides trait object interfaces for real-to-complex and complex-to-real
//! FFT operations, matching the API patterns used by realfft crate for easier migration.
//!
//! # Features
//!
//! - `RealToComplex` trait for forward real-to-complex FFT operations
//! - `ComplexToReal` trait for inverse complex-to-real FFT operations
//! - `RealFftPlanner` for creating and caching FFT plans
//! - Support for both f32 and f64 precision
//! - Thread-safe plan caching with Arc<dyn Trait>
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
use scirs2_core::numeric::Complex;
use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

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
    fn process(&self, input: &[T], output: &mut [Complex<T>]);

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
    fn process(&self, input: &[Complex<T>], output: &mut [T]);

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

/// Real FFT plan implementation for f64
struct RealFftPlanF64 {
    length: usize,
    planner: Arc<Mutex<rustfft::FftPlanner<f64>>>,
}

impl RealFftPlanF64 {
    fn new(length: usize, planner: Arc<Mutex<rustfft::FftPlanner<f64>>>) -> Self {
        Self { length, planner }
    }
}

impl RealToComplex<f64> for RealFftPlanF64 {
    fn process(&self, input: &[f64], output: &mut [Complex<f64>]) {
        // Validate input/output lengths
        assert_eq!(
            input.len(),
            self.length,
            "Input length {} doesn't match plan length {}",
            input.len(),
            self.length
        );
        assert_eq!(
            output.len(),
            self.output_len(),
            "Output length {} doesn't match expected length {}",
            output.len(),
            self.output_len()
        );

        // Convert input to complex for full FFT
        let mut buffer: Vec<Complex<f64>> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Get FFT plan and process
        let mut planner = self.planner.lock().unwrap();
        let fft = planner.plan_fft_forward(self.length);
        drop(planner); // Release lock before processing

        fft.process(&mut buffer);

        // Copy first n/2 + 1 elements to output (real FFT property)
        output.copy_from_slice(&buffer[..self.output_len()]);
    }

    fn len(&self) -> usize {
        self.length
    }
}

/// Inverse real FFT plan implementation for f64
struct InverseRealFftPlanF64 {
    length: usize,
    planner: Arc<Mutex<rustfft::FftPlanner<f64>>>,
}

impl InverseRealFftPlanF64 {
    fn new(length: usize, planner: Arc<Mutex<rustfft::FftPlanner<f64>>>) -> Self {
        Self { length, planner }
    }
}

impl ComplexToReal<f64> for InverseRealFftPlanF64 {
    fn process(&self, input: &[Complex<f64>], output: &mut [f64]) {
        // Validate input/output lengths
        assert_eq!(
            input.len(),
            self.input_len(),
            "Input length {} doesn't match expected length {}",
            input.len(),
            self.input_len()
        );
        assert_eq!(
            output.len(),
            self.length,
            "Output length {} doesn't match plan length {}",
            output.len(),
            self.length
        );

        // Reconstruct full spectrum using Hermitian symmetry
        let mut buffer: Vec<Complex<f64>> = Vec::with_capacity(self.length);
        buffer.extend_from_slice(input);

        // Add conjugate symmetric part
        let start_idx = if self.length.is_multiple_of(2) {
            input.len() - 1
        } else {
            input.len()
        };

        for i in (1..start_idx).rev() {
            buffer.push(input[i].conj());
        }

        // Pad to full length if needed
        while buffer.len() < self.length {
            buffer.push(Complex::new(0.0, 0.0));
        }

        // Get inverse FFT plan and process
        let mut planner = self.planner.lock().unwrap();
        let ifft = planner.plan_fft_inverse(self.length);
        drop(planner); // Release lock before processing

        ifft.process(&mut buffer);

        // Extract real parts and normalize
        let scale = 1.0 / self.length as f64;
        for (i, &val) in buffer.iter().enumerate() {
            output[i] = val.re * scale;
        }
    }

    fn len(&self) -> usize {
        self.length
    }
}

/// Real FFT plan implementation for f32
struct RealFftPlanF32 {
    length: usize,
    planner: Arc<Mutex<rustfft::FftPlanner<f32>>>,
}

impl RealFftPlanF32 {
    fn new(length: usize, planner: Arc<Mutex<rustfft::FftPlanner<f32>>>) -> Self {
        Self { length, planner }
    }
}

impl RealToComplex<f32> for RealFftPlanF32 {
    fn process(&self, input: &[f32], output: &mut [Complex<f32>]) {
        // Validate input/output lengths
        assert_eq!(
            input.len(),
            self.length,
            "Input length {} doesn't match plan length {}",
            input.len(),
            self.length
        );
        assert_eq!(
            output.len(),
            self.output_len(),
            "Output length {} doesn't match expected length {}",
            output.len(),
            self.output_len()
        );

        // Convert input to complex for full FFT
        let mut buffer: Vec<Complex<f32>> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Get FFT plan and process
        let mut planner = self.planner.lock().unwrap();
        let fft = planner.plan_fft_forward(self.length);
        drop(planner); // Release lock before processing

        fft.process(&mut buffer);

        // Copy first n/2 + 1 elements to output (real FFT property)
        output.copy_from_slice(&buffer[..self.output_len()]);
    }

    fn len(&self) -> usize {
        self.length
    }
}

/// Inverse real FFT plan implementation for f32
struct InverseRealFftPlanF32 {
    length: usize,
    planner: Arc<Mutex<rustfft::FftPlanner<f32>>>,
}

impl InverseRealFftPlanF32 {
    fn new(length: usize, planner: Arc<Mutex<rustfft::FftPlanner<f32>>>) -> Self {
        Self { length, planner }
    }
}

impl ComplexToReal<f32> for InverseRealFftPlanF32 {
    fn process(&self, input: &[Complex<f32>], output: &mut [f32]) {
        // Validate input/output lengths
        assert_eq!(
            input.len(),
            self.input_len(),
            "Input length {} doesn't match expected length {}",
            input.len(),
            self.input_len()
        );
        assert_eq!(
            output.len(),
            self.length,
            "Output length {} doesn't match plan length {}",
            output.len(),
            self.length
        );

        // Reconstruct full spectrum using Hermitian symmetry
        let mut buffer: Vec<Complex<f32>> = Vec::with_capacity(self.length);
        buffer.extend_from_slice(input);

        // Add conjugate symmetric part
        let start_idx = if self.length.is_multiple_of(2) {
            input.len() - 1
        } else {
            input.len()
        };

        for i in (1..start_idx).rev() {
            buffer.push(input[i].conj());
        }

        // Pad to full length if needed
        while buffer.len() < self.length {
            buffer.push(Complex::new(0.0, 0.0));
        }

        // Get inverse FFT plan and process
        let mut planner = self.planner.lock().unwrap();
        let ifft = planner.plan_fft_inverse(self.length);
        drop(planner); // Release lock before processing

        ifft.process(&mut buffer);

        // Extract real parts and normalize
        let scale = 1.0 / self.length as f32;
        for (i, &val) in buffer.iter().enumerate() {
            output[i] = val.re * scale;
        }
    }

    fn len(&self) -> usize {
        self.length
    }
}

/// Real FFT planner for creating and managing FFT plans
///
/// This planner creates reusable FFT plans optimized for real-valued input/output.
/// Plans are thread-safe and can be shared across threads using Arc.
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
    pub fn plan_fft_forward(&mut self, length: usize) -> Arc<dyn RealToComplex<f64>> {
        let planner = Arc::new(Mutex::new(rustfft::FftPlanner::<f64>::new()));
        Arc::new(RealFftPlanF64::new(length, planner))
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
    pub fn plan_fft_inverse(&mut self, length: usize) -> Arc<dyn ComplexToReal<f64>> {
        let planner = Arc::new(Mutex::new(rustfft::FftPlanner::<f64>::new()));
        Arc::new(InverseRealFftPlanF64::new(length, planner))
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
    pub fn plan_fft_forward(&mut self, length: usize) -> Arc<dyn RealToComplex<f32>> {
        let planner = Arc::new(Mutex::new(rustfft::FftPlanner::<f32>::new()));
        Arc::new(RealFftPlanF32::new(length, planner))
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
    pub fn plan_fft_inverse(&mut self, length: usize) -> Arc<dyn ComplexToReal<f32>> {
        let planner = Arc::new(Mutex::new(rustfft::FftPlanner::<f32>::new()));
        Arc::new(InverseRealFftPlanF32::new(length, planner))
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
        forward.process(&input, &mut spectrum);

        // Check DC component
        let sum: f64 = input.iter().sum();
        assert!((spectrum[0].re - sum).abs() < 1e-10);
        assert!(spectrum[0].im.abs() < 1e-10);

        // Inverse transform
        let mut recovered = vec![0.0; 8];
        inverse.process(&spectrum, &mut recovered);

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
        forward.process(&input, &mut spectrum);

        // Inverse transform
        let mut recovered = vec![0.0f32; 8];
        inverse.process(&spectrum, &mut recovered);

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
        forward.process(&input, &mut spectrum);

        // Check that energy is concentrated at the expected frequency
        let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();

        // Find peak
        let (peak_idx, &peak_mag) = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

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
            forward: Arc<dyn RealToComplex<f64>>,
            backward: Arc<dyn ComplexToReal<f64>>,
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
                self.forward.process(input, &mut spectrum);

                let mut output = vec![0.0; self.backward.len()];
                self.backward.process(&spectrum, &mut output);

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
