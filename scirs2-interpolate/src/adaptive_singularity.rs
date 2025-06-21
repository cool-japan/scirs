//! Automatic singularity detection and handling for interpolation
//!
//! This module provides algorithms to automatically detect and handle singularities
//! in interpolation problems. Singularities can include:
//!
//! - **Discontinuities**: Jump discontinuities in the function values
//! - **Sharp peaks**: Points where derivatives become very large
//! - **Infinite derivatives**: Points where the function is not smooth
//! - **Oscillatory regions**: Rapid oscillations that require special handling
//! - **Edge artifacts**: Boundary effects that can cause interpolation issues
//!
//! The detection algorithms use multiple criteria:
//! - **Derivative analysis**: Large derivatives indicate potential singularities
//! - **Error concentration**: High interpolation errors clustered in regions
//! - **Curvature analysis**: Rapid changes in curvature
//! - **Spectral analysis**: High-frequency content detection
//! - **Statistical outliers**: Data points that deviate significantly from local trends
//!
//! # Examples
//!
//! ```rust
//! use ndarray::Array1;
//! use scirs2_interpolate::adaptive_singularity::{SingularityDetector, SingularityType};
//!
//! // Create sample data with a singularity (jump discontinuity)
//! let x = Array1::linspace(0.0, 2.0, 100);
//! let y = x.mapv(|x| if x < 1.0 { x } else { x + 10.0 });
//!
//! // Detect singularities
//! let detector = SingularityDetector::new()
//!     .with_derivative_threshold(5.0)
//!     .with_curvature_threshold(10.0);
//!
//! let singularities = detector.detect(&x.view(), &y.view()).unwrap();
//!
//! // Handle the detected singularities during interpolation
//! for singularity in &singularities {
//!     println!("Found {:?} at x = {}", singularity.singularity_type, singularity.location);
//! }
//! ```

use crate::error::{InterpolateError, InterpolateResult};
use crate::multiscale::{MultiscaleBSpline, RefinementCriterion};
use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};

/// Types of singularities that can be detected
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SingularityType {
    /// Jump discontinuity in function values
    Discontinuity,
    /// Rapid change in derivatives (sharp peak)
    SharpPeak,
    /// Infinite or very large derivatives
    InfiniteDerivative,
    /// Rapid oscillations
    Oscillatory,
    /// Statistical outlier point
    Outlier,
    /// Boundary artifact
    BoundaryArtifact,
}

/// Information about a detected singularity
#[derive(Debug, Clone)]
pub struct SingularityInfo<T> {
    /// Type of the singularity
    pub singularity_type: SingularityType,
    /// Location (x-coordinate) of the singularity
    pub location: T,
    /// Index in the original data array
    pub index: usize,
    /// Severity score (higher = more severe)
    pub severity: T,
    /// Recommended treatment strategy
    pub treatment: TreatmentStrategy,
}

/// Strategies for handling detected singularities
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TreatmentStrategy {
    /// Add extra knots around the singularity
    LocalRefinement,
    /// Split the domain at the singularity
    DomainSplit,
    /// Apply smoothing in the local region
    LocalSmoothing,
    /// Remove outlier points
    RemoveOutlier,
    /// Use specialized interpolation method
    SpecialMethod,
    /// No special treatment needed
    NoTreatment,
}

/// Configuration for singularity detection
#[derive(Debug, Clone)]
pub struct SingularityDetectorConfig<T> {
    /// Threshold for derivative-based detection
    pub derivative_threshold: T,
    /// Threshold for curvature-based detection
    pub curvature_threshold: T,
    /// Threshold for discontinuity detection (ratio of jump to local variation)
    pub discontinuity_threshold: T,
    /// Threshold for outlier detection (number of standard deviations)
    pub outlier_threshold: T,
    /// Window size for local analysis (as fraction of total data length)
    pub analysis_window: T,
    /// Minimum separation between detected singularities
    pub min_separation: T,
    /// Enable oscillation detection
    pub detect_oscillations: bool,
    /// Enable boundary artifact detection
    pub detect_boundary_artifacts: bool,
}

impl<T: Float + FromPrimitive> Default for SingularityDetectorConfig<T> {
    fn default() -> Self {
        Self {
            derivative_threshold: T::from(10.0).unwrap(),
            curvature_threshold: T::from(20.0).unwrap(),
            discontinuity_threshold: T::from(5.0).unwrap(),
            outlier_threshold: T::from(3.0).unwrap(),
            analysis_window: T::from(0.1).unwrap(), // 10% of data
            min_separation: T::from(0.01).unwrap(), // 1% of domain
            detect_oscillations: true,
            detect_boundary_artifacts: true,
        }
    }
}

/// Automatic singularity detection for interpolation data
#[derive(Debug, Clone)]
pub struct SingularityDetector<T> {
    config: SingularityDetectorConfig<T>,
}

impl<T> Default for SingularityDetector<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SingularityDetector<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy,
{
    /// Create a new singularity detector with default configuration
    pub fn new() -> Self {
        Self {
            config: SingularityDetectorConfig::default(),
        }
    }

    /// Set the derivative threshold for detection
    pub fn with_derivative_threshold(mut self, threshold: T) -> Self {
        self.config.derivative_threshold = threshold;
        self
    }

    /// Set the curvature threshold for detection
    pub fn with_curvature_threshold(mut self, threshold: T) -> Self {
        self.config.curvature_threshold = threshold;
        self
    }

    /// Set the discontinuity threshold for detection
    pub fn with_discontinuity_threshold(mut self, threshold: T) -> Self {
        self.config.discontinuity_threshold = threshold;
        self
    }

    /// Set the outlier threshold for detection
    pub fn with_outlier_threshold(mut self, threshold: T) -> Self {
        self.config.outlier_threshold = threshold;
        self
    }

    /// Enable or disable oscillation detection
    pub fn with_oscillation_detection(mut self, enable: bool) -> Self {
        self.config.detect_oscillations = enable;
        self
    }

    /// Enable or disable boundary artifact detection
    pub fn with_boundary_detection(mut self, enable: bool) -> Self {
        self.config.detect_boundary_artifacts = enable;
        self
    }

    /// Detect singularities in the given data
    ///
    /// # Arguments
    ///
    /// * `x` - x-coordinates of the data points (must be sorted)
    /// * `y` - y-coordinates of the data points
    ///
    /// # Returns
    ///
    /// A vector of detected singularities with their information
    pub fn detect(
        &self,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
    ) -> InterpolateResult<Vec<SingularityInfo<T>>> {
        if x.len() != y.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "x and y arrays must have the same length, got {} and {}",
                x.len(),
                y.len()
            )));
        }

        if x.len() < 3 {
            return Err(InterpolateError::InvalidValue(
                "At least 3 data points are required for singularity detection".to_string(),
            ));
        }

        let mut singularities = Vec::new();

        // 1. Detect discontinuities
        singularities.extend(self.detect_discontinuities(x, y)?);

        // 2. Detect sharp peaks via derivative analysis
        singularities.extend(self.detect_sharp_peaks(x, y)?);

        // 3. Detect outliers
        singularities.extend(self.detect_outliers(x, y)?);

        // 4. Detect oscillatory regions (optional)
        if self.config.detect_oscillations {
            singularities.extend(self.detect_oscillations(x, y)?);
        }

        // 5. Detect boundary artifacts (optional)
        if self.config.detect_boundary_artifacts {
            singularities.extend(self.detect_boundary_artifacts(x, y)?);
        }

        // Remove duplicates and sort by location
        self.consolidate_singularities(singularities, x)
    }

    /// Detect jump discontinuities in the data
    fn detect_discontinuities(
        &self,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
    ) -> InterpolateResult<Vec<SingularityInfo<T>>> {
        let mut discontinuities = Vec::new();
        let n = x.len();

        if n < 3 {
            return Ok(discontinuities);
        }

        // Calculate local variation for normalization
        let window_size = std::cmp::max(3, (n as f64 * 0.1) as usize);

        for i in 1..n - 1 {
            // Calculate the jump at this point
            let left_diff = y[i] - y[i - 1];
            let right_diff = y[i + 1] - y[i];

            // Check for discontinuities in two ways:
            // 1. Opposite signs with large magnitudes (traditional discontinuity)
            // 2. Large magnitude difference even with same sign (step function)
            let opposite_signs = left_diff * right_diff < T::zero();
            let jump_size = (left_diff.abs() + right_diff.abs()) / T::from(2.0).unwrap();

            // Also check for large difference in magnitude even if same sign
            let magnitude_diff = (left_diff.abs() - right_diff.abs()).abs();
            let is_potential_discontinuity = opposite_signs
                || (jump_size > T::from(0.1).unwrap()
                    && magnitude_diff > jump_size * T::from(0.5).unwrap());

            if is_potential_discontinuity {
                // Calculate local variation for normalization
                let start_idx = if i >= window_size / 2 {
                    i - window_size / 2
                } else {
                    0
                };
                let end_idx = std::cmp::min(i + window_size / 2, n - 1);

                let local_var = self.calculate_local_variation(y, start_idx, end_idx);

                // Check if jump is significant compared to local variation
                if local_var > T::zero()
                    && jump_size / local_var >= self.config.discontinuity_threshold
                {
                    discontinuities.push(SingularityInfo {
                        singularity_type: SingularityType::Discontinuity,
                        location: x[i],
                        index: i,
                        severity: jump_size / local_var,
                        treatment: TreatmentStrategy::DomainSplit,
                    });
                }
            }
        }

        Ok(discontinuities)
    }

    /// Detect sharp peaks via derivative analysis
    fn detect_sharp_peaks(
        &self,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
    ) -> InterpolateResult<Vec<SingularityInfo<T>>> {
        let mut peaks = Vec::new();
        let n = x.len();

        if n < 3 {
            return Ok(peaks);
        }

        // Calculate numerical derivatives
        let mut first_derivs = Array1::zeros(n);
        let mut second_derivs = Array1::zeros(n);

        // Forward difference for first point
        first_derivs[0] = (y[1] - y[0]) / (x[1] - x[0]);

        // Central differences for interior points
        for i in 1..n - 1 {
            let h1 = x[i] - x[i - 1];
            let h2 = x[i + 1] - x[i];
            first_derivs[i] = (y[i + 1] - y[i - 1]) / (h1 + h2);

            // Second derivative
            second_derivs[i] = (y[i + 1] / h2 - y[i] * (T::one() / h1 + T::one() / h2)
                + y[i - 1] / h1)
                * T::from(2.0).unwrap()
                / (h1 + h2);
        }

        // Backward difference for last point
        first_derivs[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);

        // Find points with extremely large derivatives
        for i in 1..n - 1 {
            let deriv_magnitude = first_derivs[i].abs();
            let curvature = second_derivs[i].abs();

            if deriv_magnitude > self.config.derivative_threshold {
                let severity = deriv_magnitude / self.config.derivative_threshold;
                peaks.push(SingularityInfo {
                    singularity_type: if curvature > self.config.curvature_threshold {
                        SingularityType::InfiniteDerivative
                    } else {
                        SingularityType::SharpPeak
                    },
                    location: x[i],
                    index: i,
                    severity,
                    treatment: TreatmentStrategy::LocalRefinement,
                });
            }
        }

        Ok(peaks)
    }

    /// Detect statistical outliers in the data
    fn detect_outliers(
        &self,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
    ) -> InterpolateResult<Vec<SingularityInfo<T>>> {
        let mut outliers = Vec::new();
        let n = x.len();

        if n < 5 {
            return Ok(outliers);
        }

        // Calculate mean and standard deviation
        let mean = y.sum() / T::from(n).unwrap();
        let variance = y.mapv(|val| (val - mean) * (val - mean)).sum() / T::from(n - 1).unwrap();
        let std_dev = variance.sqrt();

        if std_dev <= T::zero() {
            return Ok(outliers); // No variation in data
        }

        // Find points that are more than threshold standard deviations from the mean
        for (i, &y_val) in y.iter().enumerate() {
            let z_score = (y_val - mean).abs() / std_dev;

            if z_score > self.config.outlier_threshold {
                outliers.push(SingularityInfo {
                    singularity_type: SingularityType::Outlier,
                    location: x[i],
                    index: i,
                    severity: z_score,
                    treatment: TreatmentStrategy::RemoveOutlier,
                });
            }
        }

        Ok(outliers)
    }

    /// Detect oscillatory regions in the data
    fn detect_oscillations(
        &self,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
    ) -> InterpolateResult<Vec<SingularityInfo<T>>> {
        let mut oscillations = Vec::new();
        let n = x.len();

        if n < 6 {
            return Ok(oscillations);
        }

        let window_size = std::cmp::max(6, n / 10);

        // Sliding window analysis for oscillations
        for i in window_size / 2..n - window_size / 2 {
            let start = i - window_size / 2;
            let end = i + window_size / 2;

            // Count zero crossings in the local window (using differences)
            let mut zero_crossings = 0;
            let mut last_diff_sign = None;

            for j in start + 1..end {
                let diff = y[j] - y[j - 1];
                if diff.abs() > T::epsilon() {
                    let current_sign = diff > T::zero();
                    if let Some(last_sign) = last_diff_sign {
                        if current_sign != last_sign {
                            zero_crossings += 1;
                        }
                    }
                    last_diff_sign = Some(current_sign);
                }
            }

            // If there are many zero crossings, it's likely oscillatory
            let oscillation_rate =
                T::from(zero_crossings).unwrap() / T::from(window_size - 1).unwrap();
            if oscillation_rate > T::from(0.3).unwrap() {
                // More than 30% zero crossings
                oscillations.push(SingularityInfo {
                    singularity_type: SingularityType::Oscillatory,
                    location: x[i],
                    index: i,
                    severity: oscillation_rate,
                    treatment: TreatmentStrategy::LocalSmoothing,
                });
            }
        }

        Ok(oscillations)
    }

    /// Detect boundary artifacts in the data
    fn detect_boundary_artifacts(
        &self,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
    ) -> InterpolateResult<Vec<SingularityInfo<T>>> {
        let mut artifacts = Vec::new();
        let n = x.len();

        if n < 5 {
            return Ok(artifacts);
        }

        let boundary_region = std::cmp::max(2, n / 20); // Check first/last 5% of points

        // Check left boundary
        for i in 0..boundary_region {
            if i + 2 < n {
                let local_trend = (y[i + 2] - y[i]) / (x[i + 2] - x[i]);
                let boundary_diff = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);

                if (boundary_diff - local_trend).abs() > self.config.derivative_threshold {
                    artifacts.push(SingularityInfo {
                        singularity_type: SingularityType::BoundaryArtifact,
                        location: x[i],
                        index: i,
                        severity: (boundary_diff - local_trend).abs(),
                        treatment: TreatmentStrategy::SpecialMethod,
                    });
                }
            }
        }

        // Check right boundary
        for i in n - boundary_region..n {
            if i >= 2 {
                let local_trend = (y[i] - y[i - 2]) / (x[i] - x[i - 2]);
                let boundary_diff = (y[i] - y[i - 1]) / (x[i] - x[i - 1]);

                if (boundary_diff - local_trend).abs() > self.config.derivative_threshold {
                    artifacts.push(SingularityInfo {
                        singularity_type: SingularityType::BoundaryArtifact,
                        location: x[i],
                        index: i,
                        severity: (boundary_diff - local_trend).abs(),
                        treatment: TreatmentStrategy::SpecialMethod,
                    });
                }
            }
        }

        Ok(artifacts)
    }

    /// Calculate local variation in a window of the data
    fn calculate_local_variation(&self, y: &ArrayView1<T>, start: usize, end: usize) -> T {
        if end <= start + 1 {
            return T::zero();
        }

        let mut sum_diff = T::zero();
        let mut count = 0;

        for i in start + 1..=end {
            if i < y.len() {
                sum_diff += (y[i] - y[i - 1]).abs();
                count += 1;
            }
        }

        if count > 0 {
            sum_diff / T::from(count).unwrap()
        } else {
            T::zero()
        }
    }

    /// Remove duplicate singularities and sort by location
    fn consolidate_singularities(
        &self,
        mut singularities: Vec<SingularityInfo<T>>,
        x: &ArrayView1<T>,
    ) -> InterpolateResult<Vec<SingularityInfo<T>>> {
        if singularities.is_empty() {
            return Ok(singularities);
        }

        // Sort by location
        singularities.sort_by(|a, b| a.location.partial_cmp(&b.location).unwrap());

        let domain_range = x[x.len() - 1] - x[0];
        let min_separation = self.config.min_separation * domain_range;

        // Remove duplicates that are too close together
        let mut consolidated = Vec::new();
        consolidated.push(singularities[0].clone());

        for singularity in singularities.into_iter().skip(1) {
            let last = consolidated.last().unwrap();
            if (singularity.location - last.location).abs() > min_separation {
                consolidated.push(singularity);
            } else {
                // When consolidating, prioritize discontinuities over other types
                let should_replace = match (last.singularity_type, singularity.singularity_type) {
                    // Keep discontinuity over any other type
                    (SingularityType::Discontinuity, _) => false,
                    // Replace any type with discontinuity
                    (_, SingularityType::Discontinuity) => true,
                    // Otherwise, use severity
                    _ => singularity.severity > last.severity,
                };

                if should_replace {
                    *consolidated.last_mut().unwrap() = singularity;
                }
            }
        }

        Ok(consolidated)
    }
}

/// Apply automatic singularity handling to a multiscale B-spline
///
/// This function combines singularity detection with adaptive refinement
/// to automatically handle problematic regions in interpolation data.
///
/// # Arguments
///
/// * `spline` - The multiscale B-spline to refine
/// * `detector` - The singularity detector configuration
/// * `max_refinements` - Maximum number of refinement iterations
///
/// # Returns
///
/// The number of refinements applied and the detected singularities
pub fn apply_singularity_handling<T>(
    spline: &mut MultiscaleBSpline<T>,
    detector: &SingularityDetector<T>,
    max_refinements: usize,
) -> InterpolateResult<(usize, Vec<SingularityInfo<T>>)>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy,
{
    // Get the current spline's domain and evaluation points
    let x_data = spline.x_data();
    let y_data = spline.y_data();

    // Detect singularities
    let singularities = detector.detect(&x_data.view(), &y_data.view())?;

    if singularities.is_empty() {
        return Ok((0, singularities));
    }

    let mut total_refinements = 0;

    // Apply treatments based on singularity types
    for singularity in &singularities {
        match singularity.treatment {
            TreatmentStrategy::LocalRefinement => {
                // Use combined criterion for areas around singularities
                let refined = spline.refine(RefinementCriterion::Combined, 5)?;
                if refined {
                    total_refinements += 1;
                }
            }
            TreatmentStrategy::DomainSplit => {
                // For discontinuities, add extra refinement
                let refined = spline.refine(RefinementCriterion::AbsoluteError, 3)?;
                if refined {
                    total_refinements += 1;
                }
            }
            TreatmentStrategy::LocalSmoothing => {
                // For oscillatory regions, use relative error criterion
                let refined = spline.refine(RefinementCriterion::RelativeError, 2)?;
                if refined {
                    total_refinements += 1;
                }
            }
            _ => {
                // Other treatments might require external handling
                continue;
            }
        }

        if total_refinements >= max_refinements {
            break;
        }
    }

    Ok((total_refinements, singularities))
}

// Helper functions that work with MultiscaleBSpline now that accessor methods are available

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_discontinuity_detection() {
        // Create data with a jump discontinuity
        let x = Array1::linspace(0.0, 2.0, 21);
        let y = x.mapv(|x| if x < 1.0 { x } else { x + 5.0 });

        let detector = SingularityDetector::new().with_discontinuity_threshold(1.0);

        let singularities = detector.detect(&x.view(), &y.view()).unwrap();

        // Should detect the discontinuity around x = 1.0
        assert!(!singularities.is_empty());
        assert!(singularities
            .iter()
            .any(|s| s.singularity_type == SingularityType::Discontinuity));
    }

    #[test]
    fn test_sharp_peak_detection() {
        // Create data with a sharp peak
        let x = Array1::linspace(-2.0, 2.0, 100);
        let y = x.mapv(|x| if x.abs() < 0.1 { 10.0 } else { 0.0 });

        let detector = SingularityDetector::new().with_derivative_threshold(50.0);

        let singularities = detector.detect(&x.view(), &y.view()).unwrap();

        // Should detect sharp peaks
        assert!(!singularities.is_empty());
        assert!(singularities.iter().any(|s| matches!(
            s.singularity_type,
            SingularityType::SharpPeak | SingularityType::InfiniteDerivative
        )));
    }

    #[test]
    fn test_outlier_detection() {
        // Create data with outliers
        let x = Array1::linspace(0.0, 10.0, 50);
        let mut y = x.mapv(|x| x.sin());
        y[25] = 10.0; // Add an outlier

        let detector = SingularityDetector::new().with_outlier_threshold(2.0);

        let singularities = detector.detect(&x.view(), &y.view()).unwrap();

        // Should detect the outlier
        assert!(!singularities.is_empty());
        assert!(singularities
            .iter()
            .any(|s| s.singularity_type == SingularityType::Outlier));
    }
}
