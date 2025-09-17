//! Type definitions and configuration structures for SciPy validation
//!
//! This module contains all the data structures, enums, and configuration types
//! used throughout the SciPy validation system.

use std::collections::HashMap;

/// Configuration for SciPy validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Numerical tolerance for comparisons
    pub tolerance: f64,
    /// Relative tolerance for comparisons
    pub relative_tolerance: f64,
    /// Test signal lengths to use
    pub test_lengths: Vec<usize>,
    /// Sampling frequencies to test
    pub sampling_frequencies: Vec<f64>,
    /// Whether to run extensive tests (slower but more thorough)
    pub extensive: bool,
    /// Maximum allowed relative error percentage
    pub max_error_percent: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            relative_tolerance: 1e-8,
            test_lengths: vec![16, 64, 128, 256, 512, 1024],
            sampling_frequencies: vec![1.0, 44100.0, 48000.0, 100.0],
            extensive: false,
            max_error_percent: 0.1, // 0.1% maximum error
        }
    }
}

/// Results of validation tests
#[derive(Debug, Clone)]
pub struct ValidationResults {
    /// Individual test results
    pub test_results: HashMap<String, ValidationTestResult>,
    /// Overall summary statistics
    pub summary: ValidationSummary,
}

/// Result of a single validation test
#[derive(Debug, Clone)]
pub struct ValidationTestResult {
    /// Test name
    pub test_name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Maximum absolute error found
    pub max_absolute_error: f64,
    /// Maximum relative error found
    pub max_relative_error: f64,
    /// Mean absolute error
    pub mean_absolute_error: f64,
    /// Number of test cases run
    pub num_test_cases: usize,
    /// Error message if test failed
    pub error_message: Option<String>,
}

/// Summary statistics for all validation tests
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    /// Total number of tests run
    pub total_tests: usize,
    /// Number of tests that passed
    pub passed_tests: usize,
    /// Number of tests that failed
    pub failed_tests: usize,
    /// Overall pass rate (0.0 to 1.0)
    pub pass_rate: f64,
    /// Average error across all tests
    pub average_error: f64,
    /// Maximum error found across all tests
    pub max_error: f64,
}

impl ValidationResults {
    /// Create new empty validation results
    pub fn new() -> Self {
        Self {
            test_results: HashMap::new(),
            summary: ValidationSummary {
                total_tests: 0,
                passed_tests: 0,
                failed_tests: 0,
                pass_rate: 0.0,
                average_error: 0.0,
                max_error: 0.0,
            },
        }
    }

    /// Add a test result
    pub fn add_test_result(&mut self, result: ValidationTestResult) {
        self.test_results.insert(result.test_name.clone(), result);
        self.update_summary();
    }

    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.summary.failed_tests == 0 && self.summary.total_tests > 0
    }

    /// Get list of failed tests
    pub fn failures(&self) -> Vec<&ValidationTestResult> {
        self.test_results
            .values()
            .filter(|result| !result.passed)
            .collect()
    }

    /// Update summary statistics
    fn update_summary(&mut self) {
        let total_tests = self.test_results.len();
        let passed_tests = self.test_results.values().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;

        let pass_rate = if total_tests > 0 {
            passed_tests as f64 / total_tests as f64
        } else {
            0.0
        };

        let average_error = if total_tests > 0 {
            self.test_results.values()
                .map(|r| r.mean_absolute_error)
                .sum::<f64>() / total_tests as f64
        } else {
            0.0
        };

        let max_error = self.test_results.values()
            .map(|r| r.max_absolute_error)
            .fold(0.0f64, f64::max);

        self.summary = ValidationSummary {
            total_tests,
            passed_tests,
            failed_tests,
            pass_rate,
            average_error,
            max_error,
        };
    }
}

impl Default for ValidationResults {
    fn default() -> Self {
        Self::new()
    }
}

/// Filter validation specific configuration
#[derive(Debug, Clone)]
pub struct FilterValidationConfig {
    /// Filter orders to test
    pub orders: Vec<usize>,
    /// Cutoff frequencies (relative to Nyquist)
    pub cutoff_frequencies: Vec<f64>,
    /// Filter types to test
    pub filter_types: Vec<FilterTestType>,
    /// Enable Butterworth filter testing
    pub test_butterworth: bool,
    /// Enable Chebyshev Type I filter testing
    pub test_chebyshev1: bool,
    /// Enable Chebyshev Type II filter testing
    pub test_chebyshev2: bool,
    /// Enable Elliptic filter testing
    pub test_elliptic: bool,
    /// Enable Bessel filter testing
    pub test_bessel: bool,
}

/// Filter types for validation testing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterTestType {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
}

impl Default for FilterValidationConfig {
    fn default() -> Self {
        Self {
            orders: vec![2, 4, 6, 8],
            cutoff_frequencies: vec![0.1, 0.25, 0.4],
            filter_types: vec![
                FilterTestType::Lowpass,
                FilterTestType::Highpass,
                FilterTestType::Bandpass,
            ],
            test_butterworth: true,
            test_chebyshev1: true,
            test_chebyshev2: false, // Less commonly used
            test_elliptic: false,   // Complex implementation
            test_bessel: false,     // Less commonly used
        }
    }
}

/// Spectral validation specific configuration
#[derive(Debug, Clone)]
pub struct SpectralValidationConfig {
    /// FFT sizes to test
    pub fft_sizes: Vec<usize>,
    /// Window types to test
    pub window_types: Vec<String>,
    /// Overlap percentages to test
    pub overlap_percentages: Vec<f64>,
    /// Enable periodogram testing
    pub test_periodogram: bool,
    /// Enable Welch method testing
    pub test_welch: bool,
    /// Enable STFT testing
    pub test_stft: bool,
    /// Enable multitaper testing
    pub test_multitaper: bool,
}

impl Default for SpectralValidationConfig {
    fn default() -> Self {
        Self {
            fft_sizes: vec![64, 128, 256, 512],
            window_types: vec!["hann".to_string(), "hamming".to_string(), "blackman".to_string()],
            overlap_percentages: vec![0.0, 0.5, 0.75],
            test_periodogram: true,
            test_welch: true,
            test_stft: true,
            test_multitaper: true,
        }
    }
}

/// Wavelet validation specific configuration
#[derive(Debug, Clone)]
pub struct WaveletValidationConfig {
    /// Wavelet families to test
    pub wavelet_families: Vec<String>,
    /// Decomposition levels to test
    pub decomposition_levels: Vec<usize>,
    /// Enable DWT testing
    pub test_dwt: bool,
    /// Enable CWT testing
    pub test_cwt: bool,
    /// Enable wavelet family validation
    pub test_families: bool,
}

impl Default for WaveletValidationConfig {
    fn default() -> Self {
        Self {
            wavelet_families: vec![
                "db4".to_string(),
                "db8".to_string(),
                "haar".to_string(),
                "bior2.2".to_string(),
            ],
            decomposition_levels: vec![1, 2, 3, 4],
            test_dwt: true,
            test_cwt: false, // More complex, may be slow
            test_families: true,
        }
    }
}

/// Window validation specific configuration
#[derive(Debug, Clone)]
pub struct WindowValidationConfig {
    /// Window types to test
    pub window_types: Vec<String>,
    /// Window lengths to test
    pub window_lengths: Vec<usize>,
    /// Enable symmetric window testing
    pub test_symmetric: bool,
    /// Enable periodic window testing
    pub test_periodic: bool,
}

impl Default for WindowValidationConfig {
    fn default() -> Self {
        Self {
            window_types: vec![
                "hann".to_string(),
                "hamming".to_string(),
                "blackman".to_string(),
                "kaiser".to_string(),
                "tukey".to_string(),
            ],
            window_lengths: vec![16, 32, 64, 128, 256],
            test_symmetric: true,
            test_periodic: true,
        }
    }
}

/// Signal generation validation configuration
#[derive(Debug, Clone)]
pub struct SignalValidationConfig {
    /// Test chirp signals
    pub test_chirp: bool,
    /// Test convolution operations
    pub test_convolution: bool,
    /// Test correlation operations
    pub test_correlation: bool,
    /// Test resampling operations
    pub test_resampling: bool,
    /// Test peak detection
    pub test_peak_detection: bool,
}

impl Default for SignalValidationConfig {
    fn default() -> Self {
        Self {
            test_chirp: true,
            test_convolution: true,
            test_correlation: true,
            test_resampling: false, // Can be complex
            test_peak_detection: false, // Algorithm dependent
        }
    }
}

/// Comprehensive validation configuration
#[derive(Debug, Clone)]
pub struct ComprehensiveValidationConfig {
    /// Base validation configuration
    pub base: ValidationConfig,
    /// Filter validation configuration
    pub filters: FilterValidationConfig,
    /// Spectral validation configuration
    pub spectral: SpectralValidationConfig,
    /// Wavelet validation configuration
    pub wavelets: WaveletValidationConfig,
    /// Window validation configuration
    pub windows: WindowValidationConfig,
    /// Signal validation configuration
    pub signals: SignalValidationConfig,
}

impl Default for ComprehensiveValidationConfig {
    fn default() -> Self {
        Self {
            base: ValidationConfig::default(),
            filters: FilterValidationConfig::default(),
            spectral: SpectralValidationConfig::default(),
            wavelets: WaveletValidationConfig::default(),
            windows: WindowValidationConfig::default(),
            signals: SignalValidationConfig::default(),
        }
    }
}

/// Error analysis results
#[derive(Debug, Clone)]
pub struct ErrorAnalysis {
    /// Absolute errors
    pub absolute_errors: Vec<f64>,
    /// Relative errors
    pub relative_errors: Vec<f64>,
    /// Maximum absolute error
    pub max_absolute_error: f64,
    /// Maximum relative error
    pub max_relative_error: f64,
    /// Mean absolute error
    pub mean_absolute_error: f64,
    /// Mean relative error
    pub mean_relative_error: f64,
    /// Root mean square error
    pub rms_error: f64,
}

impl ErrorAnalysis {
    /// Create new error analysis from two arrays
    pub fn from_arrays(reference: &[f64], test: &[f64]) -> Self {
        let mut absolute_errors = Vec::new();
        let mut relative_errors = Vec::new();

        for (&ref_val, &test_val) in reference.iter().zip(test.iter()) {
            let abs_error = (test_val - ref_val).abs();
            absolute_errors.push(abs_error);

            let rel_error = if ref_val.abs() > f64::EPSILON {
                abs_error / ref_val.abs()
            } else {
                abs_error
            };
            relative_errors.push(rel_error);
        }

        let max_absolute_error = absolute_errors.iter().cloned().fold(0.0f64, f64::max);
        let max_relative_error = relative_errors.iter().cloned().fold(0.0f64, f64::max);

        let mean_absolute_error = if !absolute_errors.is_empty() {
            absolute_errors.iter().sum::<f64>() / absolute_errors.len() as f64
        } else {
            0.0
        };

        let mean_relative_error = if !relative_errors.is_empty() {
            relative_errors.iter().sum::<f64>() / relative_errors.len() as f64
        } else {
            0.0
        };

        let rms_error = if !absolute_errors.is_empty() {
            let sum_squared: f64 = absolute_errors.iter().map(|&e| e * e).sum();
            (sum_squared / absolute_errors.len() as f64).sqrt()
        } else {
            0.0
        };

        Self {
            absolute_errors,
            relative_errors,
            max_absolute_error,
            max_relative_error,
            mean_absolute_error,
            mean_relative_error,
            rms_error,
        }
    }
}

/// Test case parameters
#[derive(Debug, Clone)]
pub struct TestParameters {
    /// Signal length
    pub length: usize,
    /// Sampling frequency
    pub fs: f64,
    /// Additional parameters as key-value pairs
    pub params: HashMap<String, String>,
}

impl TestParameters {
    /// Create new test parameters
    pub fn new(length: usize, fs: f64) -> Self {
        Self {
            length,
            fs,
            params: HashMap::new(),
        }
    }

    /// Add a parameter
    pub fn with_param(mut self, key: String, value: String) -> Self {
        self.params.insert(key, value);
        self
    }

    /// Get parameter as string
    pub fn get_param(&self, key: &str) -> Option<&String> {
        self.params.get(key)
    }

    /// Get parameter as f64
    pub fn get_param_f64(&self, key: &str) -> Option<f64> {
        self.params.get(key)?.parse().ok()
    }

    /// Get parameter as usize
    pub fn get_param_usize(&self, key: &str) -> Option<usize> {
        self.params.get(key)?.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert_eq!(config.tolerance, 1e-10);
        assert!(!config.extensive);
        assert_eq!(config.max_error_percent, 0.1);
    }

    #[test]
    fn test_validation_results() {
        let mut results = ValidationResults::new();
        assert!(results.test_results.is_empty());
        assert_eq!(results.summary.total_tests, 0);

        let test_result = ValidationTestResult {
            test_name: "test1".to_string(),
            passed: true,
            max_absolute_error: 1e-12,
            max_relative_error: 1e-10,
            mean_absolute_error: 5e-13,
            num_test_cases: 10,
            error_message: None,
        };

        results.add_test_result(test_result);
        assert_eq!(results.summary.total_tests, 1);
        assert_eq!(results.summary.passed_tests, 1);
        assert!(results.all_passed());
    }

    #[test]
    fn test_error_analysis() {
        let reference = vec![1.0, 2.0, 3.0, 4.0];
        let test = vec![1.001, 1.998, 3.002, 3.999];

        let analysis = ErrorAnalysis::from_arrays(&reference, &test);
        assert!(analysis.max_absolute_error < 0.01);
        assert!(analysis.mean_absolute_error < 0.01);
        assert_eq!(analysis.absolute_errors.len(), 4);
    }

    #[test]
    fn test_test_parameters() {
        let params = TestParameters::new(1024, 44100.0)
            .with_param("filter_order".to_string(), "4".to_string())
            .with_param("cutoff".to_string(), "0.25".to_string());

        assert_eq!(params.length, 1024);
        assert_eq!(params.fs, 44100.0);
        assert_eq!(params.get_param_usize("filter_order"), Some(4));
        assert_eq!(params.get_param_f64("cutoff"), Some(0.25));
    }
}