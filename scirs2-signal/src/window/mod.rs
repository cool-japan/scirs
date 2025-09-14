//! Window Functions for Signal Processing
//!
//! This module provides a comprehensive collection of window functions organized into:
//! - **Families**: Windows grouped by mathematical characteristics (cosine, triangular, etc.)
//! - **Applications**: Specialized utilities for spectral analysis, filter design, and time-frequency analysis
//! - **Optimization**: SIMD-accelerated implementations and lookup table caching
//!
//! The refactored design improves modularity, performance, and maintainability while
//! preserving full backward compatibility with the original API.

use crate::error::{SignalError, SignalResult};

// Window function families
pub mod families {
    //! Window function families organized by mathematical characteristics

    pub mod cosine;
    pub mod exponential;
    pub mod rectangular;
    pub mod specialized;
    pub mod triangular;

    pub use cosine::*;
    pub use exponential::*;
    pub use rectangular::*;
    pub use specialized::*;
    pub use triangular::*;
}

// Application-specific modules
pub mod applications {
    //! Application-specific window utilities and optimization

    pub mod filter_design;
    pub mod spectral_analysis;
    pub mod time_frequency;

    pub use filter_design::*;
    pub use spectral_analysis::*;
    pub use time_frequency::*;
}

// Performance optimization modules
pub mod optimization {
    //! Performance optimization modules for window generation

    pub mod lookup_tables;
    pub mod simd_implementation;

    pub use lookup_tables::*;
    pub use simd_implementation::*;
}

// Re-export all window families at top level for convenience
pub use families::cosine::*;
pub use families::exponential::*;
pub use families::rectangular::*;
pub use families::specialized::*;
pub use families::triangular::*;

// Re-export optimized window generation
pub use optimization::lookup_tables::cached_windows;

// Include the kaiser.rs file as a module
mod kaiser;

// Re-export kaiser functions for backward compatibility
pub use kaiser::{kaiser, kaiser_bessel_derived};

/// Helper function to handle small or incorrect window lengths
pub(crate) fn _len_guards(m: usize) -> bool {
    // Return true for trivial windows with length 0 or 1
    m <= 1
}

/// Helper function to extend window by 1 sample if needed for DFT-even symmetry
pub(crate) fn _extend(m: usize, sym: bool) -> (usize, bool) {
    if !sym {
        (m + 1, true)
    } else {
        (m, false)
    }
}

/// Helper function to truncate window by 1 sample if needed
pub(crate) fn _truncate(w: Vec<f64>, needed: bool) -> Vec<f64> {
    if needed {
        w[..w.len() - 1].to_vec()
    } else {
        w
    }
}

/// Create a window function of a specified type and length.
///
/// This is the main entry point for window generation, providing backward compatibility
/// with the original API while leveraging the new modular architecture.
///
/// # Arguments
///
/// * `window_type` - Type of window function to create
/// * `length` - Length of the window
/// * `periodic` - If true, the window is periodic, otherwise symmetric
///
/// # Returns
///
/// * Window function of specified type and length
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::get_window;
///
/// // Create a Hamming window of length 10
/// let window = get_window("hamming", 10, false).unwrap();
///
/// assert_eq!(window.len(), 10);
/// assert!(window[0] > 0.0 && window[0] < 1.0);
/// assert!(window[window.len() / 2] > 0.9);
/// ```
pub fn get_window(window_type: &str, length: usize, periodic: bool) -> SignalResult<Vec<f64>> {
    if length == 0 {
        return Err(SignalError::ValueError(
            "Window length must be positive".to_string(),
        ));
    }

    let symmetric = !periodic;

    // Dispatch to specific window function using new modular structure
    match window_type.to_lowercase().as_str() {
        // Cosine family windows
        "hamming" => families::cosine::hamming(length, symmetric),
        "hanning" | "hann" => families::cosine::hann(length, symmetric),
        "blackman" => families::cosine::blackman(length, symmetric),
        "blackmanharris" => families::cosine::blackmanharris(length, symmetric),
        "nuttall" => families::cosine::nuttall(length, symmetric),
        "flattop" => families::cosine::flattop(length, symmetric),
        "cosine" => families::cosine::cosine(length, symmetric),
        "barthann" => families::cosine::barthann(length, symmetric),

        // Triangular family windows
        "bartlett" => families::triangular::bartlett(length, symmetric),
        "triang" => families::triangular::triang(length, symmetric),
        "parzen" => families::triangular::parzen(length, symmetric),
        "welch" => families::triangular::welch(length, symmetric),

        // Rectangular family windows
        "boxcar" | "rectangular" => families::rectangular::boxcar(length, symmetric),

        // Exponential family windows
        "kaiser" => {
            // Default beta value of 8.6 gives sidelobe attenuation of about 60dB
            families::exponential::kaiser(length, 8.6, symmetric)
        }
        "gaussian" => {
            // Default std of 1.0
            families::exponential::gaussian(length, 1.0, symmetric)
        }
        "tukey" => {
            // Default alpha of 0.5
            families::exponential::tukey(length, 0.5, symmetric)
        }
        "exponential" => {
            // Default tau of 2.0
            families::exponential::exponential(length, 2.0, symmetric)
        }
        "lanczos" => {
            // Default parameter a = 2 for Lanczos window
            families::exponential::lanczos(length, 2.0, symmetric)
        }

        // Specialized windows
        "bohman" => families::specialized::bohman(length, symmetric),
        "poisson" => {
            // Default alpha of 1.0
            families::specialized::poisson(length, 1.0, symmetric)
        }
        "dpss" | "slepian" => {
            // Default NW parameter of 3.0 for multitaper
            families::specialized::dpss_approximation(length, 3.0, symmetric)
        }
        "kaiser_bessel_derived" => {
            // Default beta value of 8.6
            crate::window::kaiser::kaiser_bessel_derived(length, 8.6, symmetric)
        }

        _ => Err(SignalError::ValueError(format!(
            "Unknown window type: {}. Available types: hamming, hann, blackman, bartlett, flattop, boxcar, triang, bohman, parzen, nuttall, blackmanharris, cosine, exponential, tukey, barthann, kaiser, gaussian, lanczos, poisson, dpss, kaiser_bessel_derived",
            window_type
        ))),
    }
}

/// Get window with parameters for more advanced usage
///
/// Extended version of get_window that accepts parameters for parameterized windows
///
/// # Arguments
///
/// * `window_type` - Type of window function to create
/// * `length` - Length of the window
/// * `parameters` - Window-specific parameters (e.g., beta for Kaiser)
/// * `symmetric` - If true, generates symmetric window, otherwise periodic
///
/// # Returns
///
/// * Window function of specified type and length
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::get_window_with_params;
///
/// // Create a Kaiser window with specific beta
/// let window = get_window_with_params("kaiser", 64, &[5.0], true).unwrap();
/// assert_eq!(window.len(), 64);
///
/// // Create a Gaussian window with specific standard deviation
/// let window = get_window_with_params("gaussian", 64, &[2.0], true).unwrap();
/// assert_eq!(window.len(), 64);
/// ```
pub fn get_window_with_params(
    window_type: &str,
    length: usize,
    parameters: &[f64],
    symmetric: bool,
) -> SignalResult<Vec<f64>> {
    if length == 0 {
        return Err(SignalError::ValueError(
            "Window length must be positive".to_string(),
        ));
    }

    match window_type.to_lowercase().as_str() {
        // Parameterized windows
        "kaiser" => {
            let beta = parameters.get(0).copied().unwrap_or(8.6);
            families::exponential::kaiser(length, beta, symmetric)
        }
        "gaussian" => {
            let std = parameters.get(0).copied().unwrap_or(1.0);
            families::exponential::gaussian(length, std, symmetric)
        }
        "tukey" => {
            let alpha = parameters.get(0).copied().unwrap_or(0.5);
            families::exponential::tukey(length, alpha, symmetric)
        }
        "exponential" => {
            let tau = parameters.get(0).copied().unwrap_or(2.0);
            families::exponential::exponential(length, tau, symmetric)
        }
        "poisson" => {
            let alpha = parameters.get(0).copied().unwrap_or(1.0);
            families::specialized::poisson(length, alpha, symmetric)
        }
        "dpss" | "slepian" => {
            let nw = parameters.get(0).copied().unwrap_or(3.0);
            families::specialized::dpss_approximation(length, nw, symmetric)
        }
        "lanczos" => {
            let a = parameters.get(0).copied().unwrap_or(2.0);
            families::exponential::lanczos(length, a, symmetric)
        }
        "triangular_general" => {
            let slope_factor = parameters.get(0).copied().unwrap_or(1.0);
            let zero_endpoints = parameters.get(1).copied().unwrap_or(0.0) > 0.5;
            families::triangular::generalized_triangular(
                length,
                slope_factor,
                zero_endpoints,
                symmetric,
            )
        }
        "exponential_general" => {
            let decay_left = parameters.get(0).copied().unwrap_or(2.0);
            let decay_right = parameters.get(1).copied().unwrap_or(2.0);
            let peak_position = parameters.get(2).copied().unwrap_or(0.5);
            families::exponential::generalized_exponential(
                length,
                decay_left,
                decay_right,
                peak_position,
                symmetric,
            )
        }

        // Fall back to standard windows without parameters
        _ => get_window(window_type, length, !symmetric),
    }
}

/// Window generation using optimized lookup tables
///
/// High-performance window generation with automatic caching for frequently used configurations
///
/// # Arguments
///
/// * `window_type` - Type of window function to create  
/// * `length` - Length of the window
/// * `parameters` - Window-specific parameters
/// * `symmetric` - If true, generates symmetric window, otherwise periodic
///
/// # Returns
///
/// * Window function with automatic caching for performance
pub fn get_window_cached(
    window_type: &str,
    length: usize,
    parameters: &[f64],
    symmetric: bool,
) -> SignalResult<Vec<f64>> {
    optimization::lookup_tables::WindowLookupTable::global().get_or_compute_window(
        window_type,
        length,
        parameters,
        symmetric,
    )
}

/// Advanced window analysis and properties
pub mod analysis {
    //! Window analysis utilities for understanding window properties

    use super::*;

    /// Comprehensive window properties analysis
    #[derive(Debug, Clone)]
    pub struct WindowProperties {
        /// Window type name
        pub window_type: String,
        /// Window length
        pub length: usize,
        /// Coherent gain (DC response)
        pub coherent_gain: f64,
        /// Processing gain (equivalent noise bandwidth normalization)
        pub processing_gain: f64,
        /// Scalloping loss (dB)
        pub scalloping_loss: f64,
        /// Main lobe width (bins)
        pub main_lobe_width: f64,
        /// Maximum sidelobe level (dB)
        pub max_sidelobe_level: f64,
        /// Window energy
        pub energy: f64,
        /// Peak value
        pub peak_value: f64,
    }

    /// Analyze window properties
    ///
    /// Computes comprehensive properties of any window function
    ///
    /// # Arguments
    /// * `window` - Window coefficients to analyze
    /// * `window_type` - Optional window type name for reporting
    ///
    /// # Returns
    /// Detailed window properties analysis
    pub fn analyze_window_properties(
        window: &[f64],
        window_type: Option<&str>,
    ) -> WindowProperties {
        let n = window.len();

        // Basic properties
        let energy: f64 = window.iter().map(|&w| w * w).sum();
        let peak_value = window.iter().fold(0.0_f64, |a, &b| a.max(b));

        // Coherent gain (DC response normalized)
        let coherent_gain = window.iter().sum::<f64>() / n as f64;

        // Processing gain (ratio of coherent to incoherent power)
        let processing_gain = window.iter().sum::<f64>().powi(2) / (n as f64 * energy);

        // Estimate frequency domain properties
        let (scalloping_loss, main_lobe_width, max_sidelobe_level) =
            estimate_frequency_properties(window);

        WindowProperties {
            window_type: window_type.unwrap_or("unknown").to_string(),
            length: n,
            coherent_gain,
            processing_gain,
            scalloping_loss,
            main_lobe_width,
            max_sidelobe_level,
            energy,
            peak_value,
        }
    }

    /// Estimate frequency domain properties from window shape
    fn estimate_frequency_properties(window: &[f64]) -> (f64, f64, f64) {
        // These are approximations based on window characteristics
        // In a full implementation, these would be computed via FFT

        let coherent_gain = window.iter().sum::<f64>() / window.len() as f64;
        let energy: f64 = window.iter().map(|&w| w * w).sum();
        let processing_gain = coherent_gain.powi(2) * window.len() as f64 / energy;

        // Scalloping loss estimates based on window shape
        let scalloping_loss = if is_rectangular_like(window) {
            3.92
        } else if is_hann_like(window) {
            1.42
        } else if is_hamming_like(window) {
            1.78
        } else if is_blackman_like(window) {
            1.10
        } else {
            2.0
        };

        // Main lobe width estimates
        let main_lobe_width = if is_rectangular_like(window) {
            2.0
        } else if is_hann_like(window) || is_hamming_like(window) {
            4.0
        } else if is_blackman_like(window) {
            6.0
        } else {
            4.0
        };

        // Maximum sidelobe level estimates
        let max_sidelobe_level = if is_rectangular_like(window) {
            -13.3
        } else if is_hann_like(window) {
            -31.5
        } else if is_hamming_like(window) {
            -42.7
        } else if is_blackman_like(window) {
            -58.1
        } else {
            -30.0
        };

        (scalloping_loss, main_lobe_width, max_sidelobe_level)
    }

    // Helper functions for window classification
    fn is_rectangular_like(window: &[f64]) -> bool {
        let first = window.first().copied().unwrap_or(0.0);
        window.iter().all(|&x| (x - first).abs() < 0.01)
    }

    fn is_hann_like(window: &[f64]) -> bool {
        let n = window.len();
        if n < 3 {
            return false;
        }

        let endpoints_zero = window[0].abs() < 0.01 && window[n - 1].abs() < 0.01;
        let peak_at_center = window[n / 2] > 0.9;

        endpoints_zero && peak_at_center
    }

    fn is_hamming_like(window: &[f64]) -> bool {
        let n = window.len();
        if n < 3 {
            return false;
        }

        let nonzero_endpoints = window[0] > 0.05 && window[n - 1] > 0.05;
        let peak_at_center = window[n / 2] > 0.9;

        nonzero_endpoints && peak_at_center
    }

    fn is_blackman_like(window: &[f64]) -> bool {
        let n = window.len();
        if n < 5 {
            return false;
        }

        let zero_endpoints = window[0].abs() < 0.01 && window[n - 1].abs() < 0.01;
        let quarter_val = window[n / 4] / window[n / 2];

        zero_endpoints && quarter_val > 0.3 && quarter_val < 0.7
    }
}

/// Initialize window system with optimizations
///
/// Initializes SIMD capabilities and populates lookup table cache with common windows
/// Call this once at program startup for optimal performance
pub fn initialize_window_system() -> SignalResult<()> {
    // Initialize lookup table cache
    optimization::lookup_tables::initialize_window_cache()?;

    // SIMD initialization is automatic in SimdWindowGenerator::new()

    Ok(())
}

/// Window system performance benchmarking
pub mod benchmark {
    //! Performance benchmarking utilities for window functions

    use super::*;
    use std::time::{Duration, Instant};

    /// Benchmark results for window generation
    #[derive(Debug)]
    pub struct WindowBenchmarkResults {
        pub window_type: String,
        pub lengths: Vec<usize>,
        pub iterations: usize,
        pub total_duration: Duration,
        pub avg_time_per_window: Duration,
        pub windows_per_second: f64,
    }

    /// Benchmark window generation performance
    ///
    /// # Arguments
    /// * `window_type` - Type of window to benchmark
    /// * `lengths` - Window lengths to test
    /// * `iterations` - Number of iterations per length
    /// * `use_cache` - Whether to use optimized caching
    ///
    /// # Returns
    /// Benchmark results with timing information
    pub fn benchmark_window_generation(
        window_type: &str,
        lengths: &[usize],
        iterations: usize,
        use_cache: bool,
    ) -> SignalResult<WindowBenchmarkResults> {
        let start_time = Instant::now();
        let mut total_windows = 0;

        for _ in 0..iterations {
            for &length in lengths {
                let _window = if use_cache {
                    get_window_cached(window_type, length, &[], true)?
                } else {
                    get_window(window_type, length, false)?
                };
                total_windows += 1;
            }
        }

        let total_duration = start_time.elapsed();
        let avg_time_per_window = total_duration / total_windows as u32;
        let windows_per_second = total_windows as f64 / total_duration.as_secs_f64();

        Ok(WindowBenchmarkResults {
            window_type: window_type.to_string(),
            lengths: lengths.to_vec(),
            iterations,
            total_duration,
            avg_time_per_window,
            windows_per_second,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_window_basic() {
        let window = get_window("hamming", 64, false).unwrap();
        assert_eq!(window.len(), 64);

        let window = get_window("hann", 32, true).unwrap();
        assert_eq!(window.len(), 32);
    }

    #[test]
    fn test_get_window_with_params() {
        let kaiser = get_window_with_params("kaiser", 64, &[5.0], true).unwrap();
        assert_eq!(kaiser.len(), 64);

        let gaussian = get_window_with_params("gaussian", 64, &[2.0], true).unwrap();
        assert_eq!(gaussian.len(), 64);
    }

    #[test]
    fn test_cached_windows() {
        let window1 = get_window_cached("hann", 64, &[], true).unwrap();
        let window2 = get_window_cached("hann", 64, &[], true).unwrap();
        assert_eq!(window1, window2);
    }

    #[test]
    fn test_window_analysis() {
        let window = get_window("blackman", 64, false).unwrap();
        let props = analysis::analyze_window_properties(&window, Some("blackman"));

        assert_eq!(props.length, 64);
        assert_eq!(props.window_type, "blackman");
        assert!(props.coherent_gain > 0.0);
        assert!(props.processing_gain > 0.0);
    }

    #[test]
    fn test_initialize_window_system() {
        let result = initialize_window_system();
        assert!(result.is_ok());
    }

    #[test]
    fn test_all_supported_windows() {
        let windows = [
            "hamming",
            "hann",
            "blackman",
            "bartlett",
            "flattop",
            "boxcar",
            "triang",
            "bohman",
            "parzen",
            "nuttall",
            "blackmanharris",
            "cosine",
            "exponential",
            "tukey",
            "barthann",
            "kaiser",
            "gaussian",
            "lanczos",
            "poisson",
            "dpss",
        ];

        for window_type in &windows {
            let result = get_window(window_type, 64, false);
            assert!(result.is_ok(), "Failed to create {} window", window_type);
        }
    }

    #[test]
    fn test_invalid_window_type() {
        let result = get_window("invalid_window", 64, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_length_window() {
        let result = get_window("hann", 0, false);
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "timeout"]
    fn test_benchmark_functionality() {
        let result = benchmark::benchmark_window_generation("hann", &[32, 64], 10, false);
        assert!(result.is_ok());

        let bench = result.unwrap();
        assert!(bench.total_duration.as_nanos() > 0);
        assert!(bench.windows_per_second > 0.0);
    }
}
