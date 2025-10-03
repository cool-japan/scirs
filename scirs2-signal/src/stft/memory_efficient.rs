//! Memory-efficient STFT processing for large signals
//!
//! This module provides memory-optimized STFT computation that can handle
//! very large signals by processing them in chunks.

use super::core::ShortTimeFft;
use super::types::*;
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array2};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{Float, NumCast};
use std::fmt::Debug;

/// Memory-efficient STFT processor for large signals
pub struct MemoryEfficientStft {
    stft: ShortTimeFft,
    config: MemoryEfficientStftConfig,
}

impl MemoryEfficientStft {
    /// Create a new memory-efficient STFT processor
    pub fn new(
        window: &[f64],
        hop_size: usize,
        fs: f64,
        stft_config: Option<StftConfig>,
        memory_config: MemoryEfficientStftConfig,
    ) -> SignalResult<Self> {
        let stft = ShortTimeFft::new(window, hop_size, fs, stft_config)?;

        Ok(Self {
            stft,
            config: memory_config,
        })
    }

    /// Calculate optimal chunk size based on memory constraints
    fn calculate_chunk_size(&self, signal_length: usize) -> usize {
        if let Some(chunk_size) = self.config.chunk_size {
            return chunk_size;
        }

        // Estimate memory usage per sample
        let window_length = self.stft.win.len();
        let fft_size = self.stft.mfft;
        let hop_size = self.stft.hop;

        // Estimate frames per MB
        let frames_per_mb = if self.config.magnitude_only {
            // Only storing magnitude: 8 bytes per complex sample -> 8 bytes per magnitude
            1_000_000 / (fft_size * 8)
        } else {
            // Storing complex values: 16 bytes per complex sample
            1_000_000 / (fft_size * 16)
        };

        let max_frames = frames_per_mb * self.config.memory_limit;
        let samples_per_chunk = max_frames * hop_size + window_length;

        // Ensure chunk size is reasonable
        samples_per_chunk.min(signal_length).max(window_length * 2)
    }

    /// Process STFT in chunks for memory efficiency
    pub fn stft_chunked<T>(&self, signal: &[T]) -> SignalResult<Array2<Complex64>>
    where
        T: Float + NumCast + Debug + Send + Sync,
    {
        let chunk_size = self.calculate_chunk_size(signal.len());
        let window_length = self.stft.win.len();
        let hop_size = self.stft.hop;

        // Calculate overlap needed between chunks
        let overlap = window_length.saturating_sub(hop_size);

        // Estimate total output size
        let total_frames = self.stft.p_max(signal.len()) - self.stft.p_min();
        let mut result = Array2::zeros((self.stft.f_pts(), total_frames as usize));

        let mut frame_offset = 0;
        let mut sample_offset = 0;

        while sample_offset < signal.len() {
            // Calculate chunk boundaries
            let chunk_start = sample_offset.saturating_sub(overlap);
            let chunk_end = (sample_offset + chunk_size).min(signal.len());

            if chunk_end <= chunk_start {
                break;
            }

            // Extract chunk
            let chunk = &signal[chunk_start..chunk_end];

            // Process chunk
            let chunk_stft = self.stft.stft(chunk)?;

            // Calculate where to place results in output array
            let frames_in_chunk = chunk_stft.shape()[1];
            let skip_frames = if sample_offset == 0 {
                0
            } else {
                overlap / hop_size
            };

            let copy_frames = frames_in_chunk.saturating_sub(skip_frames);
            let end_frame = (frame_offset + copy_frames).min(result.shape()[1]);

            if frame_offset < result.shape()[1] && copy_frames > 0 {
                let copy_end = (skip_frames + copy_frames).min(chunk_stft.shape()[1]);

                // Copy data to result array
                for f in 0..self.stft.f_pts() {
                    for t in skip_frames..copy_end {
                        let result_t = frame_offset + t - skip_frames;
                        if result_t < result.shape()[1] {
                            result[[f, result_t]] = chunk_stft[[f, t]];
                        }
                    }
                }

                frame_offset = end_frame;
            }

            // Move to next chunk
            sample_offset += chunk_size;
        }

        Ok(result)
    }

    /// Process spectrogram in chunks (magnitude only for memory efficiency)
    pub fn spectrogram_chunked<T>(&self, signal: &[T]) -> SignalResult<Array2<f64>>
    where
        T: Float + NumCast + Debug + Send + Sync,
    {
        if self.config.magnitude_only {
            // Process directly to magnitude
            let chunk_size = self.calculate_chunk_size(signal.len());
            let window_length = self.stft.win.len();
            let hop_size = self.stft.hop;
            let overlap = window_length.saturating_sub(hop_size);

            let total_frames = self.stft.p_max(signal.len()) - self.stft.p_min();
            let mut result = Array2::zeros((self.stft.f_pts(), total_frames as usize));

            let mut frame_offset = 0;
            let mut sample_offset = 0;

            while sample_offset < signal.len() {
                let chunk_start = sample_offset.saturating_sub(overlap);
                let chunk_end = (sample_offset + chunk_size).min(signal.len());

                if chunk_end <= chunk_start {
                    break;
                }

                let chunk = &signal[chunk_start..chunk_end];
                let chunk_spec = self.stft.spectrogram(chunk)?;

                let frames_in_chunk = chunk_spec.shape()[1];
                let skip_frames = if sample_offset == 0 {
                    0
                } else {
                    overlap / hop_size
                };
                let copy_frames = frames_in_chunk.saturating_sub(skip_frames);
                let end_frame = (frame_offset + copy_frames).min(result.shape()[1]);

                if frame_offset < result.shape()[1] && copy_frames > 0 {
                    let copy_end = (skip_frames + copy_frames).min(chunk_spec.shape()[1]);

                    for f in 0..self.stft.f_pts() {
                        for t in skip_frames..copy_end {
                            let result_t = frame_offset + t - skip_frames;
                            if result_t < result.shape()[1] {
                                result[[f, result_t]] = chunk_spec[[f, t]];
                            }
                        }
                    }

                    frame_offset = end_frame;
                }

                sample_offset += chunk_size;
            }

            Ok(result)
        } else {
            // Use regular STFT then compute magnitude
            let stft_result = self.stft_chunked(signal)?;
            Ok(stft_result.mapv(|c| c.norm_sqr()))
        }
    }

    /// Estimate memory usage for processing a signal of given length
    pub fn memory_estimate(&self, signal_length: usize) -> f64 {
        let window_length = self.stft.win.len();
        let fft_size = self.stft.mfft;
        let hop_size = self.stft.hop;

        // Estimate number of frames
        let frames = (signal_length + hop_size - 1) / hop_size;

        // Memory for STFT matrix
        let stft_memory = if self.config.magnitude_only {
            frames * fft_size * 8 // 8 bytes per f64
        } else {
            frames * fft_size * 16 // 16 bytes per Complex64
        };

        // Memory for chunking buffers
        let chunk_size = self.calculate_chunk_size(signal_length);
        let chunk_memory = chunk_size * 8; // Signal buffer

        (stft_memory + chunk_memory) as f64 / 1_000_000.0 // Convert to MB
    }

    /// Automatically choose processing method based on signal size and memory
    pub fn stft_auto<T>(&self, signal: &[T]) -> SignalResult<Array2<Complex64>>
    where
        T: Float + NumCast + Debug + Send + Sync,
    {
        let estimated_memory = self.memory_estimate(signal.len());

        if estimated_memory <= self.config.memory_limit as f64 {
            // Can fit in memory, use regular STFT
            self.stft.stft(signal)
        } else {
            // Too large, use chunked processing
            self.stft_chunked(signal)
        }
    }

    /// Automatically choose spectrogram processing method
    pub fn spectrogram_auto<T>(&self, signal: &[T]) -> SignalResult<Array2<f64>>
    where
        T: Float + NumCast + Debug + Send + Sync,
    {
        let estimated_memory = self.memory_estimate(signal.len());

        if estimated_memory <= self.config.memory_limit as f64 {
            // Can fit in memory, use regular spectrogram
            self.stft.spectrogram(signal)
        } else {
            // Too large, use chunked processing
            self.spectrogram_chunked(signal)
        }
    }

    /// Process STFT with parallel chunking
    pub fn stft_parallel_chunked<T>(&self, signal: &[T]) -> SignalResult<Array2<Complex64>>
    where
        T: Float + NumCast + Debug + Send + Sync,
    {
        if !self.config.parallel {
            return self.stft_chunked(signal);
        }

        // Determine number of chunks for parallel processing
        let chunk_size = self.calculate_chunk_size(signal.len());
        let window_length = self.stft.win.len();
        let overlap = window_length.saturating_sub(self.stft.hop);

        let num_chunks = (signal.len() + chunk_size - overlap - 1) / (chunk_size - overlap);
        let actual_chunks = num_chunks.min(8); // Limit to reasonable number of threads

        if actual_chunks <= 1 {
            return self.stft_chunked(signal);
        }

        // Implement parallel processing using scirs2_core::parallel_ops
        use scirs2_core::parallel_ops::par_chunks;

        // Calculate chunk boundaries
        let mut chunk_infos = Vec::with_capacity(actual_chunks);
        for i in 0..actual_chunks {
            let start = i * (chunk_size - overlap);
            let end = (start + chunk_size).min(signal.len());
            if end > start {
                let chunk_start = start.saturating_sub(overlap);
                chunk_infos.push((chunk_start, end, i));
            }
        }

        // Process chunks in parallel
        let chunk_results: Vec<_> = par_chunks(&chunk_infos, |chunk_info| {
            let (start, end, _idx) = *chunk_info;
            let chunk_data = &signal[start..end];
            self.stft.stft(chunk_data)
        })
        .collect();

        // Combine results
        let total_frames = self.stft.p_max(signal.len()) - self.stft.p_min();
        let mut result = Array2::zeros((self.stft.f_pts(), total_frames as usize));

        let mut frame_offset = 0;
        for (i, chunk_result) in chunk_results.iter().enumerate() {
            if let Ok(chunk_stft) = chunk_result {
                let frames_in_chunk = chunk_stft.shape()[1];
                let skip_frames = if i == 0 { 0 } else { overlap / self.stft.hop };
                let copy_frames = frames_in_chunk.saturating_sub(skip_frames);
                let end_frame = (frame_offset + copy_frames).min(result.shape()[1]);

                if frame_offset < result.shape()[1] && copy_frames > 0 {
                    let copy_end = (skip_frames + copy_frames).min(chunk_stft.shape()[1]);

                    for f in 0..self.stft.f_pts() {
                        for t in skip_frames..copy_end {
                            let result_t = frame_offset + t - skip_frames;
                            if result_t < result.shape()[1] {
                                result[[f, result_t]] = chunk_stft[[f, t]];
                            }
                        }
                    }

                    frame_offset = end_frame;
                }
            }
        }

        Ok(result)
    }

    /// Get memory usage information
    pub fn memory_info(&self, signal_length: usize) -> MemoryInfo {
        let estimated_memory = self.memory_estimate(signal_length);
        let chunk_size = self.calculate_chunk_size(signal_length);
        let num_chunks = (signal_length + chunk_size - 1) / chunk_size;

        MemoryInfo {
            estimated_memory_mb: estimated_memory,
            chunk_size,
            num_chunks,
            memory_limit_mb: self.config.memory_limit,
            will_use_chunking: estimated_memory > self.config.memory_limit as f64,
        }
    }
}

impl Default for MemoryEfficientStftConfig {
    fn default() -> Self {
        Self {
            memory_limit: 512, // 512 MB default
            chunk_size: None,
            chunk_overlap: 0,
        }
    }
}

/// Memory usage information for STFT processing
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Estimated memory usage in MB
    pub estimated_memory_mb: f64,
    /// Chunk size that will be used
    pub chunk_size: usize,
    /// Number of chunks for processing
    pub num_chunks: usize,
    /// Memory limit in MB
    pub memory_limit_mb: usize,
    /// Whether chunking will be used
    pub will_use_chunking: bool,
}