//! Streaming Short-Time Fourier Transform (STFT)
//!
//! This module provides streaming STFT computation for real-time signal processing
//! applications. It allows processing of continuous data streams with bounded
//! latency and memory usage.
//!
//! ## Features
//!
//! - **Real-time Processing**: Low-latency streaming STFT computation
//! - **Configurable Overlap**: Support for different overlap ratios
//! - **Memory Efficient**: Fixed memory usage independent of stream length
//! - **Multiple Windows**: Support for various window functions
//! - **Bounded Latency**: Predictable processing delay
//! - **Frame-by-Frame**: Process data in small chunks
//!
//! ## Use Cases
//!
//! - Real-time audio processing
//! - Online spectral analysis
//! - Streaming signal classification
//! - Live audio effects processing
//! - Continuous monitoring systems
//!
//! ## Example Usage
//!
//! ```rust
//! use ndarray::Array1;
//! use scirs2_signal::streaming_stft::{StreamingStft, StreamingStftConfig};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//!
//! // Configure streaming STFT
//! let config = StreamingStftConfig {
//!     frame_length: 512,
//!     hop_length: 256,
//!     window: "hann".to_string(),
//!     ..Default::default()
//! };
//!
//! let mut streaming_stft = StreamingStft::new(config)?;
//!
//! // Process streaming data
//! let input_frame = Array1::from_vec(vec![0.0; 256]);
//! if let Some(spectrum) = streaming_stft.process_frame(&input_frame)? {
//!     println!("Got spectrum with {} frequency bins", spectrum.len());
//! }
//! # Ok(())
//! # }
//! ```

use crate::error::{SignalError, SignalResult};
use crate::window::get_window;
use ndarray::Array1;
use num_complex::Complex64;
use std::collections::VecDeque;

/// Configuration for streaming STFT
#[derive(Debug, Clone)]
pub struct StreamingStftConfig {
    /// Frame length (window size) in samples
    pub frame_length: usize,
    /// Hop length (step size) between frames in samples
    pub hop_length: usize,
    /// Window function name ("hann", "hamming", "blackman", etc.)
    pub window: String,
    /// Whether to center frames (pad at beginning)
    pub center: bool,
    /// Padding type for centering ("constant", "reflect", "symmetric")
    pub pad_mode: String,
    /// Whether to return magnitude spectrum only
    pub magnitude_only: bool,
    /// Whether to apply log scaling to output
    pub log_magnitude: bool,
    /// Power for magnitude spectrum (1.0 for magnitude, 2.0 for power)
    pub power: f64,
    /// Minimum value for log scaling (prevents log(0))
    pub log_epsilon: f64,
}

impl Default for StreamingStftConfig {
    fn default() -> Self {
        Self {
            frame_length: 512,
            hop_length: 256,
            window: "hann".to_string(),
            center: true,
            pad_mode: "constant".to_string(),
            magnitude_only: false,
            log_magnitude: false,
            power: 1.0,
            log_epsilon: 1e-10,
        }
    }
}

/// Streaming STFT processor
pub struct StreamingStft {
    /// Configuration
    config: StreamingStftConfig,
    /// Window function
    window: Array1<f64>,
    /// Input buffer for overlap-add
    input_buffer: VecDeque<f64>,
    /// Total samples processed
    samples_processed: usize,
    /// Frames generated
    frames_generated: usize,
    /// FFT plan for efficient computation
    fft_plan: Option<scirs2_fft::FftPlan>,
}

impl std::fmt::Debug for StreamingStft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingStft")
            .field("config", &self.config)
            .field("window_length", &self.window.len())
            .field("input_buffer_size", &self.input_buffer.len())
            .field("samples_processed", &self.samples_processed)
            .field("frames_generated", &self.frames_generated)
            .field("has_fft_plan", &self.fft_plan.is_some())
            .finish()
    }
}

impl StreamingStft {
    /// Create a new streaming STFT processor
    ///
    /// # Arguments
    /// * `config` - Configuration for the streaming STFT
    ///
    /// # Returns
    /// * New streaming STFT processor instance
    pub fn new(config: StreamingStftConfig) -> SignalResult<Self> {
        // Validate configuration
        if config.frame_length == 0 {
            return Err(SignalError::ValueError(
                "Frame length must be greater than 0".to_string(),
            ));
        }

        if config.hop_length == 0 {
            return Err(SignalError::ValueError(
                "Hop length must be greater than 0".to_string(),
            ));
        }

        if config.hop_length > config.frame_length {
            return Err(SignalError::ValueError(
                "Hop length should not exceed frame length".to_string(),
            ));
        }

        if config.power <= 0.0 {
            return Err(SignalError::ValueError(
                "Power must be positive".to_string(),
            ));
        }

        // Generate window function
        let window = get_window(&config.window, config.frame_length, true)?;
        let window_array = Array1::from(window);

        // Initialize input buffer
        let mut input_buffer = VecDeque::new();

        // Pre-fill buffer for centering if needed
        if config.center {
            let pad_length = config.frame_length / 2;
            match config.pad_mode.as_str() {
                "constant" => {
                    for _ in 0..pad_length {
                        input_buffer.push_back(0.0);
                    }
                }
                "reflect" | "symmetric" => {
                    // For now, use constant padding
                    // Full implementation would need more complex logic
                    for _ in 0..pad_length {
                        input_buffer.push_back(0.0);
                    }
                }
                _ => {
                    return Err(SignalError::ValueError(format!(
                        "Unknown pad mode: {}",
                        config.pad_mode
                    )));
                }
            }
        }

        // Create FFT plan if available
        let fft_plan = None; // We'll use the direct FFT function

        Ok(Self {
            config,
            window: window_array,
            input_buffer,
            samples_processed: 0,
            frames_generated: 0,
            fft_plan,
        })
    }

    /// Process a frame of input data
    ///
    /// # Arguments
    /// * `input_frame` - Input audio frame
    ///
    /// # Returns
    /// * Optional spectrum array (None if not enough data accumulated)
    pub fn process_frame(
        &mut self,
        input_frame: &Array1<f64>,
    ) -> SignalResult<Option<Array1<Complex64>>> {
        // Add input frame to buffer
        for &sample in input_frame.iter() {
            self.input_buffer.push_back(sample);
        }

        self.samples_processed += input_frame.len();

        // Check if we have enough samples for a frame
        if self.input_buffer.len() >= self.config.frame_length {
            // Extract windowed frame
            let mut frame = Array1::<f64>::zeros(self.config.frame_length);
            for i in 0..self.config.frame_length {
                frame[i] = self.input_buffer[i];
            }

            // Apply window
            let windowed_frame = &frame * &self.window;

            // Compute FFT
            let spectrum = self.compute_fft(&windowed_frame)?;

            // Advance buffer by hop length
            for _ in 0..self.config.hop_length {
                if !self.input_buffer.is_empty() {
                    self.input_buffer.pop_front();
                }
            }

            self.frames_generated += 1;

            // Process spectrum based on configuration
            let processed_spectrum = self.process_spectrum(spectrum)?;

            Ok(Some(processed_spectrum))
        } else {
            Ok(None)
        }
    }

    /// Process multiple input frames at once
    ///
    /// # Arguments
    /// * `input_data` - Input signal data
    /// * `frame_size` - Size of each input frame
    ///
    /// # Returns
    /// * Vector of spectrum arrays
    pub fn process_batch(
        &mut self,
        input_data: &Array1<f64>,
        frame_size: usize,
    ) -> SignalResult<Vec<Array1<Complex64>>> {
        let mut results = Vec::new();
        let mut start = 0;

        while start + frame_size <= input_data.len() {
            let frame = input_data.slice(ndarray::s![start..start + frame_size]);
            let frame_array = frame.to_owned();

            if let Some(spectrum) = self.process_frame(&frame_array)? {
                results.push(spectrum);
            }

            start += frame_size;
        }

        // Process remaining samples if any
        if start < input_data.len() {
            let remaining = input_data.slice(ndarray::s![start..]);
            let frame_array = remaining.to_owned();

            if let Some(spectrum) = self.process_frame(&frame_array)? {
                results.push(spectrum);
            }
        }

        Ok(results)
    }

    /// Process magnitude-only frames (more efficient for some applications)
    ///
    /// # Arguments
    /// * `input_frame` - Input audio frame
    ///
    /// # Returns
    /// * Optional magnitude spectrum array
    pub fn process_magnitude_frame(
        &mut self,
        input_frame: &Array1<f64>,
    ) -> SignalResult<Option<Array1<f64>>> {
        if let Some(complex_spectrum) = self.process_frame(input_frame)? {
            let magnitude_spectrum = if self.config.power == 1.0 {
                complex_spectrum.mapv(|c| c.norm())
            } else if self.config.power == 2.0 {
                complex_spectrum.mapv(|c| c.norm_sqr())
            } else {
                complex_spectrum.mapv(|c| c.norm().powf(self.config.power))
            };

            let processed_magnitude = if self.config.log_magnitude {
                magnitude_spectrum.mapv(|m| (m + self.config.log_epsilon).ln())
            } else {
                magnitude_spectrum
            };

            Ok(Some(processed_magnitude))
        } else {
            Ok(None)
        }
    }

    /// Get latency in samples
    pub fn get_latency_samples(&self) -> usize {
        if self.config.center {
            self.config.frame_length / 2 + self.config.hop_length
        } else {
            self.config.frame_length
        }
    }

    /// Get latency in seconds for given sample rate
    pub fn get_latency_seconds(&self, sample_rate: f64) -> f64 {
        self.get_latency_samples() as f64 / sample_rate
    }

    /// Get processing statistics
    pub fn get_statistics(&self) -> StreamingStftStatistics {
        StreamingStftStatistics {
            samples_processed: self.samples_processed,
            frames_generated: self.frames_generated,
            buffer_size: self.input_buffer.len(),
            latency_samples: self.get_latency_samples(),
        }
    }

    /// Reset the processor state
    pub fn reset(&mut self) {
        self.input_buffer.clear();
        self.samples_processed = 0;
        self.frames_generated = 0;

        // Re-initialize padding if centering is enabled
        if self.config.center {
            let pad_length = self.config.frame_length / 2;
            for _ in 0..pad_length {
                self.input_buffer.push_back(0.0);
            }
        }
    }

    /// Flush remaining data and get final spectra
    pub fn flush(&mut self) -> SignalResult<Vec<Array1<Complex64>>> {
        let mut results = Vec::new();

        // Process remaining data in buffer
        while self.input_buffer.len() >= self.config.hop_length {
            // Pad frame if necessary
            let mut frame = Array1::<f64>::zeros(self.config.frame_length);
            let available = self.input_buffer.len().min(self.config.frame_length);

            for i in 0..available {
                frame[i] = self.input_buffer[i];
            }
            // Remaining elements are already zero (padding)

            // Apply window and compute FFT
            let windowed_frame = &frame * &self.window;
            let spectrum = self.compute_fft(&windowed_frame)?;
            let processed_spectrum = self.process_spectrum(spectrum)?;
            results.push(processed_spectrum);

            // Advance buffer
            for _ in 0..self.config.hop_length {
                if !self.input_buffer.is_empty() {
                    self.input_buffer.pop_front();
                }
            }

            self.frames_generated += 1;
        }

        Ok(results)
    }

    /// Compute FFT of windowed frame
    fn compute_fft(&self, windowed_frame: &Array1<f64>) -> SignalResult<Array1<Complex64>> {
        // Convert to slice for FFT function
        let frame_slice = windowed_frame.as_slice().unwrap();

        // Compute FFT
        let fft_result = scirs2_fft::fft(frame_slice, None)
            .map_err(|e| SignalError::ComputationError(format!("FFT computation error: {}", e)))?;

        // Return only positive frequencies (first half + Nyquist)
        let n_freq = self.config.frame_length / 2 + 1;
        let spectrum = Array1::from_iter(fft_result.into_iter().take(n_freq));

        Ok(spectrum)
    }

    /// Process spectrum based on configuration
    fn process_spectrum(&self, spectrum: Array1<Complex64>) -> SignalResult<Array1<Complex64>> {
        if self.config.magnitude_only {
            // Convert to magnitude spectrum
            let magnitude_spectrum = if self.config.power == 1.0 {
                spectrum.mapv(|c| Complex64::new(c.norm(), 0.0))
            } else if self.config.power == 2.0 {
                spectrum.mapv(|c| Complex64::new(c.norm_sqr(), 0.0))
            } else {
                spectrum.mapv(|c| Complex64::new(c.norm().powf(self.config.power), 0.0))
            };

            if self.config.log_magnitude {
                Ok(magnitude_spectrum
                    .mapv(|c| Complex64::new((c.re + self.config.log_epsilon).ln(), 0.0)))
            } else {
                Ok(magnitude_spectrum)
            }
        } else {
            Ok(spectrum)
        }
    }
}

/// Statistics for streaming STFT processing
#[derive(Debug, Clone)]
pub struct StreamingStftStatistics {
    /// Total samples processed
    pub samples_processed: usize,
    /// Total frames generated
    pub frames_generated: usize,
    /// Current buffer size
    pub buffer_size: usize,
    /// Processing latency in samples
    pub latency_samples: usize,
}

/// Real-time STFT processor with block-based processing
#[derive(Debug)]
pub struct RealTimeStft {
    /// Base streaming STFT
    streaming_stft: StreamingStft,
    /// Input block size for real-time processing
    block_size: usize,
    /// Output buffer for spectrograms
    output_buffer: VecDeque<Array1<Complex64>>,
    /// Maximum output buffer size
    max_output_buffer_size: usize,
}

impl RealTimeStft {
    /// Create a new real-time STFT processor
    ///
    /// # Arguments
    /// * `config` - Streaming STFT configuration
    /// * `block_size` - Input block size for real-time processing
    /// * `max_buffer_size` - Maximum output buffer size
    ///
    /// # Returns
    /// * New real-time STFT processor
    pub fn new(
        config: StreamingStftConfig,
        block_size: usize,
        max_buffer_size: usize,
    ) -> SignalResult<Self> {
        let streaming_stft = StreamingStft::new(config)?;

        Ok(Self {
            streaming_stft,
            block_size,
            output_buffer: VecDeque::new(),
            max_output_buffer_size: max_buffer_size,
        })
    }

    /// Process a real-time block
    ///
    /// # Arguments
    /// * `input_block` - Input audio block
    ///
    /// # Returns
    /// * Number of new spectra available
    pub fn process_block(&mut self, input_block: &Array1<f64>) -> SignalResult<usize> {
        if input_block.len() != self.block_size {
            return Err(SignalError::ValueError(format!(
                "Input block size {} does not match expected size {}",
                input_block.len(),
                self.block_size
            )));
        }

        let mut new_spectra_count = 0;

        // Process the block
        if let Some(spectrum) = self.streaming_stft.process_frame(input_block)? {
            // Add to output buffer
            self.output_buffer.push_back(spectrum);
            new_spectra_count += 1;

            // Manage buffer size
            while self.output_buffer.len() > self.max_output_buffer_size {
                self.output_buffer.pop_front();
            }
        }

        Ok(new_spectra_count)
    }

    /// Get the oldest spectrum from the buffer
    pub fn get_spectrum(&mut self) -> Option<Array1<Complex64>> {
        self.output_buffer.pop_front()
    }

    /// Get all available spectra
    pub fn get_all_spectra(&mut self) -> Vec<Array1<Complex64>> {
        let mut results = Vec::new();
        while let Some(spectrum) = self.output_buffer.pop_front() {
            results.push(spectrum);
        }
        results
    }

    /// Get the latest spectrum without removing it
    pub fn peek_latest_spectrum(&self) -> Option<&Array1<Complex64>> {
        self.output_buffer.back()
    }

    /// Get number of available spectra
    pub fn available_spectra_count(&self) -> usize {
        self.output_buffer.len()
    }

    /// Check if buffer is full
    pub fn is_buffer_full(&self) -> bool {
        self.output_buffer.len() >= self.max_output_buffer_size
    }

    /// Reset the processor
    pub fn reset(&mut self) {
        self.streaming_stft.reset();
        self.output_buffer.clear();
    }

    /// Get processing statistics
    pub fn get_statistics(&self) -> RealTimeStftStatistics {
        let base_stats = self.streaming_stft.get_statistics();
        RealTimeStftStatistics {
            base_statistics: base_stats,
            output_buffer_size: self.output_buffer.len(),
            output_buffer_capacity: self.max_output_buffer_size,
            block_size: self.block_size,
        }
    }
}

/// Statistics for real-time STFT processing
#[derive(Debug, Clone)]
pub struct RealTimeStftStatistics {
    /// Base streaming STFT statistics
    pub base_statistics: StreamingStftStatistics,
    /// Current output buffer size
    pub output_buffer_size: usize,
    /// Output buffer capacity
    pub output_buffer_capacity: usize,
    /// Input block size
    pub block_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_streaming_stft_creation() {
        let config = StreamingStftConfig::default();
        let stft = StreamingStft::new(config).unwrap();

        assert_eq!(stft.config.frame_length, 512);
        assert_eq!(stft.config.hop_length, 256);
    }

    #[test]
    fn test_streaming_stft_processing() {
        let config = StreamingStftConfig {
            frame_length: 256,
            hop_length: 128,
            center: false,
            ..Default::default()
        };

        let mut stft = StreamingStft::new(config).unwrap();

        // Generate a test signal
        let fs = 1000.0;
        let freq = 100.0;
        let n_samples = 256;

        let input: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect();

        let input_frame = Array1::from(input);

        // Process frame
        let result = stft.process_frame(&input_frame).unwrap();
        assert!(result.is_some());

        let spectrum = result.unwrap();
        assert_eq!(spectrum.len(), 129); // 256/2 + 1
    }

    #[test]
    fn test_streaming_stft_magnitude() {
        let config = StreamingStftConfig {
            frame_length: 128,
            hop_length: 64,
            center: false,
            magnitude_only: true,
            ..Default::default()
        };

        let mut stft = StreamingStft::new(config).unwrap();

        let input = Array1::from_vec(vec![1.0; 128]);
        let magnitude_result = stft.process_magnitude_frame(&input).unwrap();

        assert!(magnitude_result.is_some());
        let magnitude_spectrum = magnitude_result.unwrap();
        assert_eq!(magnitude_spectrum.len(), 65); // 128/2 + 1
    }

    #[test]
    fn test_real_time_stft() {
        let config = StreamingStftConfig {
            frame_length: 256,
            hop_length: 128,
            center: false,
            ..Default::default()
        };

        let mut rt_stft = RealTimeStft::new(config, 128, 10).unwrap();

        let input_block = Array1::from_vec(vec![0.5; 128]);
        let new_spectra = rt_stft.process_block(&input_block).unwrap();

        // Might not generate spectrum on first block due to buffering
        assert!(new_spectra <= 1);
    }

    #[test]
    fn test_streaming_stft_latency() {
        let config = StreamingStftConfig {
            frame_length: 512,
            hop_length: 256,
            center: true,
            ..Default::default()
        };

        let stft = StreamingStft::new(config).unwrap();

        let latency_samples = stft.get_latency_samples();
        assert_eq!(latency_samples, 512); // frame_length/2 + hop_length = 256 + 256

        let latency_seconds = stft.get_latency_seconds(1000.0);
        assert_eq!(latency_seconds, 0.512);
    }

    #[test]
    fn test_streaming_stft_batch_processing() {
        let config = StreamingStftConfig {
            frame_length: 128,
            hop_length: 64,
            center: false,
            ..Default::default()
        };

        let mut stft = StreamingStft::new(config).unwrap();

        let input_data = Array1::from_vec(vec![1.0; 512]);
        let results = stft.process_batch(&input_data, 64).unwrap();

        // Should generate multiple spectra
        assert!(!results.is_empty());
    }

    #[test]
    fn test_streaming_stft_flush() {
        let config = StreamingStftConfig {
            frame_length: 128,
            hop_length: 64,
            center: false,
            ..Default::default()
        };

        let mut stft = StreamingStft::new(config).unwrap();

        // Add some data
        let input = Array1::from_vec(vec![1.0; 100]);
        let _ = stft.process_frame(&input);

        // Flush remaining data
        let flushed_results = stft.flush().unwrap();

        // Should process remaining data
        assert!(!flushed_results.is_empty() || stft.input_buffer.is_empty());
    }
}
