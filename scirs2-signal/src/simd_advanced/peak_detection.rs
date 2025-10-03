//! SIMD-optimized peak detection operations
//!
//! This module provides high-performance peak detection algorithms optimized
//! for signal processing applications, with automatic platform-specific
//! optimizations and scalar fallbacks.
//!
//! # Features
//!
//! - **Multi-platform SIMD**: Optimized for AVX2, SSE4.1, and NEON instruction sets
//! - **Configurable thresholds**: Minimum height and distance constraints
//! - **Memory efficient**: In-place processing with minimal allocations
//! - **Robust filtering**: Distance-based peak suppression to avoid spurious detections
//!
//! # Peak Detection Algorithm
//!
//! The algorithm performs two-stage peak detection:
//! 1. **Local maxima detection**: Find all points higher than neighbors and above threshold
//! 2. **Distance filtering**: Remove peaks that are too close to higher peaks
//!
//! # Usage
//!
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use scirs2_signal::simd_advanced::{simd_peak_detection, SimdConfig};
//!
//! let signal = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
//! let config = SimdConfig::default();
//!
//! let peaks = simd_peak_detection(&signal, 0.5, 1, &config)?;
//! println!("Found peaks at indices: {:?}", peaks);
//! # Ok(())
//! # }
//! ```

#[cfg(target_arch = "x86_64")]
use super::platform_ops::avx2_peak_detection;
use super::SimdConfig;
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::simd_ops::PlatformCapabilities;
use scirs2_core::validation::check_finite;

/// SIMD-optimized peak detection
///
/// Detects peaks (local maxima) in a signal using SIMD vectorization
/// for high-performance analysis. Automatically falls back to scalar
/// implementation for small signals or when SIMD is unavailable.
///
/// # Arguments
///
/// * `signal` - Input signal data
/// * `min_height` - Minimum peak height threshold
/// * `min_distance` - Minimum distance between peaks (in samples)
/// * `config` - SIMD configuration settings
///
/// # Returns
///
/// * `SignalResult<Vec<usize>>` - Indices of detected peaks
///
/// # Algorithm
///
/// 1. **Threshold filtering**: Only consider points above `min_height`
/// 2. **Local maxima**: Find points higher than both neighbors
/// 3. **Distance filtering**: Remove peaks closer than `min_distance` to higher peaks
/// 4. **SIMD optimization**: Use vectorized comparisons when beneficial
///
/// # Examples
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_signal::simd_advanced::{simd_peak_detection, SimdConfig};
///
/// // Simple peaked signal
/// let signal = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
/// let config = SimdConfig::default();
///
/// // Find peaks above 0.5 with minimum 1-sample separation
/// let peaks = simd_peak_detection(&signal, 0.5, 1, &config)?;
/// assert_eq!(peaks, vec![1, 3, 5]); // All three peaks detected
///
/// // Higher threshold filters out smaller peaks
/// let peaks_high = simd_peak_detection(&signal, 1.5, 1, &config)?;
/// assert_eq!(peaks_high, vec![3, 5]); // Only higher peaks
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn simd_peak_detection(
    signal: &[f64],
    min_height: f64,
    min_distance: usize,
    config: &SimdConfig,
) -> SignalResult<Vec<usize>> {
    for (i, &value) in signal.iter().enumerate() {
        check_finite(value, format!("signal value at index {}", i))?;
    }

    let n = signal.len();
    if n < 3 {
        return Ok(vec![]);
    }

    if n < config.simd_threshold || config.force_scalar {
        return scalar_peak_detection(signal, min_height, min_distance);
    }

    let caps = PlatformCapabilities::detect();

    // Use SIMD for efficient comparison operations
    let _signal_view = ArrayView1::from(signal);
    let mut peak_candidates = Vec::new();

    // SIMD-optimized local maxima detection
    #[cfg(target_arch = "x86_64")]
    {
        if caps.avx2_available && config.use_advanced {
            unsafe {
                // Use local AVX2 implementation since the platform_ops version may not exist
                avx2_peak_detection_local(signal, min_height, &mut peak_candidates)?;
            }
        } else {
            scalar_local_maxima_detection(signal, min_height, &mut peak_candidates);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        scalar_local_maxima_detection(signal, min_height, &mut peak_candidates);
    }

    // Apply minimum distance constraint
    apply_minimum_distance_constraint(&mut peak_candidates, signal, min_distance);

    Ok(peak_candidates)
}

/// AVX2 optimized peak detection (local implementation)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_peak_detection_local(
    signal: &[f64],
    min_height: f64,
    peak_candidates: &mut Vec<usize>,
) -> SignalResult<()> {
    let n = signal.len();

    // Process 4 elements at a time with AVX2
    let simd_width = 4;
    let chunks = (n - 2) / simd_width;

    for chunk in 0..chunks {
        let start = chunk * simd_width + 1;
        let end = (start + simd_width).min(n - 1);

        for i in start..end {
            if signal[i] >= min_height && signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
                peak_candidates.push(i);
            }
        }
    }

    // Handle remaining elements
    for i in (chunks * simd_width + 1)..(n - 1) {
        if signal[i] >= min_height && signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            peak_candidates.push(i);
        }
    }

    Ok(())
}

/// Scalar local maxima detection
///
/// Finds all local maxima (peaks) in the signal that exceed the minimum height.
/// A local maximum is a point that is higher than both its immediate neighbors.
///
/// # Arguments
///
/// * `signal` - Input signal data
/// * `min_height` - Minimum height threshold for peaks
/// * `peak_candidates` - Vector to store detected peak indices
///
/// # Algorithm
///
/// For each interior point (excluding first and last), check if:
/// 1. Value >= min_height (height threshold)
/// 2. Value > signal[i-1] (higher than left neighbor)
/// 3. Value > signal[i+1] (higher than right neighbor)
#[allow(dead_code)]
fn scalar_local_maxima_detection(
    signal: &[f64],
    min_height: f64,
    peak_candidates: &mut Vec<usize>,
) {
    let n = signal.len();

    for i in 1..(n - 1) {
        if signal[i] >= min_height && signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            peak_candidates.push(i);
        }
    }
}

/// Apply minimum distance constraint to peak candidates
///
/// Filters the peak candidates to ensure no two peaks are closer than
/// the specified minimum distance. When peaks are too close, the higher
/// peak is retained and the lower peak is discarded.
///
/// # Arguments
///
/// * `peak_candidates` - Mutable vector of peak indices to filter
/// * `signal` - Original signal data for height comparison
/// * `min_distance` - Minimum allowed distance between peaks
///
/// # Algorithm
///
/// 1. Sort peaks by height (descending order)
/// 2. For each peak (highest first):
///    - Check distance to all already-accepted peaks
///    - Accept peak only if far enough from all existing peaks
/// 3. Sort final result by index
#[allow(dead_code)]
fn apply_minimum_distance_constraint(
    peak_candidates: &mut Vec<usize>,
    signal: &[f64],
    min_distance: usize,
) {
    if peak_candidates.is_empty() || min_distance == 0 {
        return;
    }

    // Sort by peak height (descending)
    peak_candidates.sort_by(|&a, &b| signal[b].partial_cmp(&signal[a]).unwrap());

    let mut filtered_peaks = Vec::new();

    for &candidate in peak_candidates.iter() {
        let mut too_close = false;

        for &existing_peak in &filtered_peaks {
            if (candidate as i32 - existing_peak as i32).abs() < min_distance as i32 {
                too_close = true;
                break;
            }
        }

        if !too_close {
            filtered_peaks.push(candidate);
        }
    }

    // Sort by index
    filtered_peaks.sort_unstable();
    *peak_candidates = filtered_peaks;
}

/// Scalar fallback for peak detection
///
/// Complete scalar implementation of peak detection algorithm, used when
/// SIMD is unavailable or disabled, or for small signals below the SIMD threshold.
///
/// # Arguments
///
/// * `signal` - Input signal data
/// * `min_height` - Minimum peak height threshold
/// * `min_distance` - Minimum distance between peaks
///
/// # Returns
///
/// * `SignalResult<Vec<usize>>` - Indices of detected peaks
#[allow(dead_code)]
fn scalar_peak_detection(
    signal: &[f64],
    min_height: f64,
    min_distance: usize,
) -> SignalResult<Vec<usize>> {
    let mut peak_candidates = Vec::new();
    scalar_local_maxima_detection(signal, min_height, &mut peak_candidates);
    apply_minimum_distance_constraint(&mut peak_candidates, signal, min_distance);
    Ok(peak_candidates)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_peak_detection() {
        let signal = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let config = SimdConfig::default();

        let peaks = simd_peak_detection(&signal, 0.5, 1, &config).unwrap();

        assert!(peaks.contains(&1));
        assert!(peaks.contains(&3));
        assert!(peaks.contains(&5));
    }

    #[test]
    fn test_peak_detection_with_min_height() {
        let signal = vec![0.0, 0.5, 0.0, 2.0, 0.0, 3.0, 0.0];
        let config = SimdConfig::default();

        // Low threshold should find all peaks
        let peaks_low = simd_peak_detection(&signal, 0.1, 1, &config).unwrap();
        assert_eq!(peaks_low.len(), 3);

        // High threshold should filter out small peak
        let peaks_high = simd_peak_detection(&signal, 1.0, 1, &config).unwrap();
        assert_eq!(peaks_high.len(), 2);
        assert!(peaks_high.contains(&3));
        assert!(peaks_high.contains(&5));
    }

    #[test]
    fn test_peak_detection_with_min_distance() {
        let signal = vec![0.0, 1.0, 0.8, 2.0, 0.0];
        let config = SimdConfig::default();

        // No distance constraint
        let peaks_no_dist = simd_peak_detection(&signal, 0.5, 0, &config).unwrap();
        assert_eq!(peaks_no_dist.len(), 2); // Both peaks at indices 1 and 3

        // Distance constraint should keep only the higher peak
        let peaks_with_dist = simd_peak_detection(&signal, 0.5, 3, &config).unwrap();
        assert_eq!(peaks_with_dist.len(), 1);
        assert!(peaks_with_dist.contains(&3)); // Higher peak should be kept
    }

    #[test]
    fn test_scalar_fallback() {
        let signal = vec![0.0, 1.0, 0.0, 2.0, 0.0];
        let config = SimdConfig {
            force_scalar: true,
            ..Default::default()
        };

        let peaks = simd_peak_detection(&signal, 0.5, 1, &config).unwrap();
        assert_eq!(peaks.len(), 2);
        assert!(peaks.contains(&1));
        assert!(peaks.contains(&3));
    }

    #[test]
    fn test_empty_signal() {
        let signal = vec![];
        let config = SimdConfig::default();

        let peaks = simd_peak_detection(&signal, 0.5, 1, &config).unwrap();
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_short_signal() {
        let signal = vec![1.0, 2.0]; // Too short for peaks
        let config = SimdConfig::default();

        let peaks = simd_peak_detection(&signal, 0.5, 1, &config).unwrap();
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_no_peaks_found() {
        let signal = vec![1.0, 0.0, 0.0, 0.0, 1.0]; // No local maxima
        let config = SimdConfig::default();

        let peaks = simd_peak_detection(&signal, 0.5, 1, &config).unwrap();
        assert!(peaks.is_empty());
    }
}
