//! Type definitions for parallel filtering operations
//!
//! This module contains the configuration structures and enums used
//! throughout the parallel filtering framework.

/// Configuration for parallel filter operations
#[derive(Debug, Clone)]
pub struct ParallelFilterConfig {
    /// Chunk size for parallel processing
    pub chunk_size: Option<usize>,
    /// Number of threads to use (None for automatic)
    pub num_threads: Option<usize>,
    /// Enable SIMD optimizations
    pub use_simd: bool,
    /// Memory optimization mode
    pub memory_efficient: bool,
    /// Enable load balancing for uneven workloads
    pub load_balancing: bool,
    /// Prefetch factor for memory optimization
    pub prefetch_factor: usize,
}

impl Default for ParallelFilterConfig {
    fn default() -> Self {
        Self {
            chunk_size: None,
            num_threads: None,
            use_simd: true,
            memory_efficient: false,
            load_balancing: true,
            prefetch_factor: 2,
        }
    }
}

/// Types of parallel filters available
#[derive(Debug, Clone)]
pub enum ParallelFilterType {
    /// FIR filter with coefficients
    FIR { coeffs: Vec<f64> },
    /// IIR filter with numerator and denominator
    IIR {
        numerator: Vec<f64>,
        denominator: Vec<f64>,
    },
    /// Adaptive filter
    Adaptive {
        desired: Vec<f64>,
        filter_length: usize,
        step_size: f64,
    },
    /// FFT-based convolution filter
    FFT { impulse_response: Vec<f64> },
}

/// Types of morphological operations
#[derive(Debug, Clone, Copy)]
pub enum MorphologicalOperation {
    Erosion,
    Dilation,
    Opening,
    Closing,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_filter_config_default() {
        let config = ParallelFilterConfig::default();
        assert!(config.use_simd);
        assert!(!config.memory_efficient);
        assert!(config.load_balancing);
        assert_eq!(config.prefetch_factor, 2);
    }

    #[test]
    fn test_parallel_filter_types() {
        let fir_filter = ParallelFilterType::FIR {
            coeffs: vec![1.0, 2.0, 3.0],
        };

        match fir_filter {
            ParallelFilterType::FIR { coeffs } => {
                assert_eq!(coeffs, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("Expected FIR filter type"),
        }

        let iir_filter = ParallelFilterType::IIR {
            numerator: vec![1.0, 2.0],
            denominator: vec![1.0, -0.5],
        };

        match iir_filter {
            ParallelFilterType::IIR {
                numerator,
                denominator,
            } => {
                assert_eq!(numerator, vec![1.0, 2.0]);
                assert_eq!(denominator, vec![1.0, -0.5]);
            }
            _ => panic!("Expected IIR filter type"),
        }
    }

    #[test]
    fn test_morphological_operations() {
        let operations = [
            MorphologicalOperation::Erosion,
            MorphologicalOperation::Dilation,
            MorphologicalOperation::Opening,
            MorphologicalOperation::Closing,
        ];

        for operation in &operations {
            // Test that operations can be copied and compared
            let copied_op = *operation;
            assert_eq!(format!("{:?}", operation), format!("{:?}", copied_op));
        }
    }
}
