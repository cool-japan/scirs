//! Similarity metrics for image registration
//!
//! This module provides various similarity metrics used to evaluate
//! the quality of image alignment during registration.

use crate::error::{Result, VisionError};
use ndarray::Array2;
use std::collections::HashMap;

/// Available similarity metrics for registration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimilarityMetric {
    /// Sum of Squared Differences (lower is better)
    SumOfSquaredDifferences,
    /// Normalized Cross-Correlation (higher is better, range -1 to 1)
    NormalizedCrossCorrelation,
    /// Mutual Information (higher is better)
    MutualInformation,
    /// Normalized Mutual Information (higher is better, range 0 to 1)
    NormalizedMutualInformation,
    /// Mean Squared Error (lower is better)
    MeanSquaredError,
    /// Peak Signal-to-Noise Ratio (higher is better)
    PeakSignalToNoiseRatio,
    /// Structural Similarity Index (higher is better, range -1 to 1)
    StructuralSimilarity,
    /// Gradient Correlation (higher is better)
    GradientCorrelation,
}

/// Trait for similarity measure implementations
pub trait SimilarityMeasure {
    /// Compute similarity between two images
    fn compute(&self, reference: &Array2<f32>, moving: &Array2<f32>) -> Result<f64>;
    
    /// Whether higher values indicate better similarity
    fn higher_is_better(&self) -> bool;
    
    /// Get the optimal value for this metric
    fn optimal_value(&self) -> f64;
    
    /// Normalize the metric value to [0, 1] range where 1 is best
    fn normalize(&self, value: f64) -> f64;
}

/// Sum of Squared Differences implementation
#[derive(Debug, Clone)]
pub struct SumOfSquaredDifferences;

impl SimilarityMeasure for SumOfSquaredDifferences {
    fn compute(&self, reference: &Array2<f32>, moving: &Array2<f32>) -> Result<f64> {
        validate_dimensions(reference, moving)?;
        
        let sum_sq_diff: f64 = reference
            .iter()
            .zip(moving.iter())
            .map(|(&r, &m)| {
                let diff = r as f64 - m as f64;
                diff * diff
            })
            .sum();
            
        Ok(sum_sq_diff)
    }
    
    fn higher_is_better(&self) -> bool {
        false
    }
    
    fn optimal_value(&self) -> f64 {
        0.0
    }
    
    fn normalize(&self, value: f64) -> f64 {
        // Convert to similarity: lower SSD = higher similarity
        if value <= 0.0 {
            1.0
        } else {
            1.0 / (1.0 + value)
        }
    }
}

/// Normalized Cross-Correlation implementation
#[derive(Debug, Clone)]
pub struct NormalizedCrossCorrelation;

impl SimilarityMeasure for NormalizedCrossCorrelation {
    fn compute(&self, reference: &Array2<f32>, moving: &Array2<f32>) -> Result<f64> {
        validate_dimensions(reference, moving)?;
        
        let n = reference.len() as f64;
        
        // Compute means
        let ref_mean: f64 = reference.iter().map(|&x| x as f64).sum::<f64>() / n;
        let mov_mean: f64 = moving.iter().map(|&x| x as f64).sum::<f64>() / n;
        
        // Compute normalized cross-correlation
        let mut numerator = 0.0;
        let mut ref_var = 0.0;
        let mut mov_var = 0.0;
        
        for (&r, &m) in reference.iter().zip(moving.iter()) {
            let ref_centered = r as f64 - ref_mean;
            let mov_centered = m as f64 - mov_mean;
            
            numerator += ref_centered * mov_centered;
            ref_var += ref_centered * ref_centered;
            mov_var += mov_centered * mov_centered;
        }
        
        if ref_var == 0.0 || mov_var == 0.0 {
            return Ok(0.0);
        }
        
        let ncc = numerator / (ref_var * mov_var).sqrt();
        Ok(ncc)
    }
    
    fn higher_is_better(&self) -> bool {
        true
    }
    
    fn optimal_value(&self) -> f64 {
        1.0
    }
    
    fn normalize(&self, value: f64) -> f64 {
        // NCC is already in [-1, 1], convert to [0, 1]
        (value + 1.0) / 2.0
    }
}

/// Mutual Information implementation
#[derive(Debug, Clone)]
pub struct MutualInformation {
    bins: usize,
}

impl MutualInformation {
    /// Create new mutual information metric with specified number of bins
    pub fn new(bins: usize) -> Self {
        Self { bins }
    }
}

impl Default for MutualInformation {
    fn default() -> Self {
        Self::new(256)
    }
}

impl SimilarityMeasure for MutualInformation {
    fn compute(&self, reference: &Array2<f32>, moving: &Array2<f32>) -> Result<f64> {
        validate_dimensions(reference, moving)?;
        
        let (joint_hist, ref_hist, mov_hist) = compute_histograms(reference, moving, self.bins)?;
        
        let total_samples = reference.len() as f64;
        let mut mi = 0.0;
        
        for i in 0..self.bins {
            for j in 0..self.bins {
                let joint_prob = joint_hist[&(i, j)] / total_samples;
                let ref_prob = ref_hist[&i] / total_samples;
                let mov_prob = mov_hist[&j] / total_samples;
                
                if joint_prob > 0.0 && ref_prob > 0.0 && mov_prob > 0.0 {
                    mi += joint_prob * (joint_prob / (ref_prob * mov_prob)).ln();
                }
            }
        }
        
        Ok(mi)
    }
    
    fn higher_is_better(&self) -> bool {
        true
    }
    
    fn optimal_value(&self) -> f64 {
        // Theoretical maximum depends on image content
        (self.bins as f64).ln()
    }
    
    fn normalize(&self, value: f64) -> f64 {
        // Normalize by theoretical maximum
        let max_mi = self.optimal_value();
        if max_mi > 0.0 {
            (value / max_mi).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Normalized Mutual Information implementation
#[derive(Debug, Clone)]
pub struct NormalizedMutualInformation {
    mi: MutualInformation,
}

impl NormalizedMutualInformation {
    /// Create new normalized mutual information metric
    pub fn new(bins: usize) -> Self {
        Self {
            mi: MutualInformation::new(bins),
        }
    }
}

impl Default for NormalizedMutualInformation {
    fn default() -> Self {
        Self::new(256)
    }
}

impl SimilarityMeasure for NormalizedMutualInformation {
    fn compute(&self, reference: &Array2<f32>, moving: &Array2<f32>) -> Result<f64> {
        validate_dimensions(reference, moving)?;
        
        let (joint_hist, ref_hist, mov_hist) = compute_histograms(reference, moving, self.mi.bins)?;
        
        let total_samples = reference.len() as f64;
        let mut h_ref = 0.0;
        let mut h_mov = 0.0;
        let mut h_joint = 0.0;
        
        // Compute marginal entropies
        for i in 0..self.mi.bins {
            let ref_prob = ref_hist[&i] / total_samples;
            if ref_prob > 0.0 {
                h_ref -= ref_prob * ref_prob.ln();
            }
            
            let mov_prob = mov_hist[&i] / total_samples;
            if mov_prob > 0.0 {
                h_mov -= mov_prob * mov_prob.ln();
            }
        }
        
        // Compute joint entropy
        for i in 0..self.mi.bins {
            for j in 0..self.mi.bins {
                let joint_prob = joint_hist[&(i, j)] / total_samples;
                if joint_prob > 0.0 {
                    h_joint -= joint_prob * joint_prob.ln();
                }
            }
        }
        
        // Normalized MI = (H(ref) + H(mov)) / H(ref, mov)
        if h_joint > 0.0 {
            Ok((h_ref + h_mov) / h_joint)
        } else {
            Ok(0.0)
        }
    }
    
    fn higher_is_better(&self) -> bool {
        true
    }
    
    fn optimal_value(&self) -> f64 {
        2.0 // Maximum NMI is 2.0
    }
    
    fn normalize(&self, value: f64) -> f64 {
        (value / 2.0).clamp(0.0, 1.0)
    }
}

/// Mean Squared Error implementation
#[derive(Debug, Clone)]
pub struct MeanSquaredError;

impl SimilarityMeasure for MeanSquaredError {
    fn compute(&self, reference: &Array2<f32>, moving: &Array2<f32>) -> Result<f64> {
        validate_dimensions(reference, moving)?;
        
        let mse: f64 = reference
            .iter()
            .zip(moving.iter())
            .map(|(&r, &m)| {
                let diff = r as f64 - m as f64;
                diff * diff
            })
            .sum::<f64>() / reference.len() as f64;
            
        Ok(mse)
    }
    
    fn higher_is_better(&self) -> bool {
        false
    }
    
    fn optimal_value(&self) -> f64 {
        0.0
    }
    
    fn normalize(&self, value: f64) -> f64 {
        if value <= 0.0 {
            1.0
        } else {
            1.0 / (1.0 + value)
        }
    }
}

/// Peak Signal-to-Noise Ratio implementation
#[derive(Debug, Clone)]
pub struct PeakSignalToNoiseRatio {
    max_value: f64,
}

impl PeakSignalToNoiseRatio {
    /// Create new PSNR metric with specified maximum pixel value
    pub fn new(max_value: f64) -> Self {
        Self { max_value }
    }
}

impl Default for PeakSignalToNoiseRatio {
    fn default() -> Self {
        Self::new(1.0) // Assume normalized images
    }
}

impl SimilarityMeasure for PeakSignalToNoiseRatio {
    fn compute(&self, reference: &Array2<f32>, moving: &Array2<f32>) -> Result<f64> {
        validate_dimensions(reference, moving)?;
        
        let mse: f64 = reference
            .iter()
            .zip(moving.iter())
            .map(|(&r, &m)| {
                let diff = r as f64 - m as f64;
                diff * diff
            })
            .sum::<f64>() / reference.len() as f64;
        
        if mse == 0.0 {
            Ok(f64::INFINITY)
        } else {
            let psnr = 20.0 * (self.max_value / mse.sqrt()).log10();
            Ok(psnr)
        }
    }
    
    fn higher_is_better(&self) -> bool {
        true
    }
    
    fn optimal_value(&self) -> f64 {
        f64::INFINITY
    }
    
    fn normalize(&self, value: f64) -> f64 {
        if value.is_infinite() {
            1.0
        } else {
            // Normalize PSNR to [0, 1] range (assuming reasonable PSNR values 0-100 dB)
            (value / 100.0).clamp(0.0, 1.0)
        }
    }
}

/// Structural Similarity Index implementation
#[derive(Debug, Clone)]
pub struct StructuralSimilarity {
    window_size: usize,
    k1: f64,
    k2: f64,
    l: f64, // Dynamic range of pixel values
}

impl StructuralSimilarity {
    /// Create new SSIM metric with specified parameters
    pub fn new(window_size: usize, k1: f64, k2: f64, l: f64) -> Self {
        Self { window_size, k1, k2, l }
    }
}

impl Default for StructuralSimilarity {
    fn default() -> Self {
        Self::new(11, 0.01, 0.03, 1.0)
    }
}

impl SimilarityMeasure for StructuralSimilarity {
    fn compute(&self, reference: &Array2<f32>, moving: &Array2<f32>) -> Result<f64> {
        validate_dimensions(reference, moving)?;
        
        let (height, width) = reference.dim();
        let half_window = self.window_size / 2;
        
        if height < self.window_size || width < self.window_size {
            return Err(VisionError::InvalidParameter(
                "Image too small for SSIM window".to_string(),
            ));
        }
        
        let c1 = (self.k1 * self.l).powi(2);
        let c2 = (self.k2 * self.l).powi(2);
        
        let mut ssim_sum = 0.0;
        let mut valid_windows = 0;
        
        for y in half_window..(height - half_window) {
            for x in half_window..(width - half_window) {
                let mut ref_sum = 0.0;
                let mut mov_sum = 0.0;
                let mut ref_sq_sum = 0.0;
                let mut mov_sq_sum = 0.0;
                let mut ref_mov_sum = 0.0;
                let window_pixels = self.window_size * self.window_size;
                
                // Compute statistics over window
                for dy in 0..self.window_size {
                    for dx in 0..self.window_size {
                        let wy = y - half_window + dy;
                        let wx = x - half_window + dx;
                        
                        let ref_val = reference[[wy, wx]] as f64;
                        let mov_val = moving[[wy, wx]] as f64;
                        
                        ref_sum += ref_val;
                        mov_sum += mov_val;
                        ref_sq_sum += ref_val * ref_val;
                        mov_sq_sum += mov_val * mov_val;
                        ref_mov_sum += ref_val * mov_val;
                    }
                }
                
                let n = window_pixels as f64;
                let mu1 = ref_sum / n;
                let mu2 = mov_sum / n;
                let mu1_sq = mu1 * mu1;
                let mu2_sq = mu2 * mu2;
                let mu1_mu2 = mu1 * mu2;
                
                let sigma1_sq = (ref_sq_sum / n) - mu1_sq;
                let sigma2_sq = (mov_sq_sum / n) - mu2_sq;
                let sigma12 = (ref_mov_sum / n) - mu1_mu2;
                
                let numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2);
                let denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2);
                
                if denominator > 0.0 {
                    ssim_sum += numerator / denominator;
                    valid_windows += 1;
                }
            }
        }
        
        if valid_windows > 0 {
            Ok(ssim_sum / valid_windows as f64)
        } else {
            Ok(0.0)
        }
    }
    
    fn higher_is_better(&self) -> bool {
        true
    }
    
    fn optimal_value(&self) -> f64 {
        1.0
    }
    
    fn normalize(&self, value: f64) -> f64 {
        // SSIM is already in [-1, 1], convert to [0, 1]
        (value + 1.0) / 2.0
    }
}

/// Gradient Correlation implementation
#[derive(Debug, Clone)]
pub struct GradientCorrelation;

impl SimilarityMeasure for GradientCorrelation {
    fn compute(&self, reference: &Array2<f32>, moving: &Array2<f32>) -> Result<f64> {
        validate_dimensions(reference, moving)?;
        
        let ref_grad = compute_gradient_magnitude(reference)?;
        let mov_grad = compute_gradient_magnitude(moving)?;
        
        // Compute correlation between gradient magnitudes
        let ncc = NormalizedCrossCorrelation;
        ncc.compute(&ref_grad, &mov_grad)
    }
    
    fn higher_is_better(&self) -> bool {
        true
    }
    
    fn optimal_value(&self) -> f64 {
        1.0
    }
    
    fn normalize(&self, value: f64) -> f64 {
        (value + 1.0) / 2.0
    }
}

/// Factory function to create similarity measure from metric type
pub fn create_similarity_measure(metric: SimilarityMetric) -> Box<dyn SimilarityMeasure> {
    match metric {
        SimilarityMetric::SumOfSquaredDifferences => Box::new(SumOfSquaredDifferences),
        SimilarityMetric::NormalizedCrossCorrelation => Box::new(NormalizedCrossCorrelation),
        SimilarityMetric::MutualInformation => Box::new(MutualInformation::default()),
        SimilarityMetric::NormalizedMutualInformation => Box::new(NormalizedMutualInformation::default()),
        SimilarityMetric::MeanSquaredError => Box::new(MeanSquaredError),
        SimilarityMetric::PeakSignalToNoiseRatio => Box::new(PeakSignalToNoiseRatio::default()),
        SimilarityMetric::StructuralSimilarity => Box::new(StructuralSimilarity::default()),
        SimilarityMetric::GradientCorrelation => Box::new(GradientCorrelation),
    }
}

// Helper functions

fn validate_dimensions(reference: &Array2<f32>, moving: &Array2<f32>) -> Result<()> {
    if reference.dim() != moving.dim() {
        return Err(VisionError::InvalidParameter(
            "Reference and moving images must have the same dimensions".to_string(),
        ));
    }
    
    let (height, width) = reference.dim();
    if height == 0 || width == 0 {
        return Err(VisionError::InvalidParameter(
            "Images must have non-zero dimensions".to_string(),
        ));
    }
    
    Ok(())
}

fn compute_histograms(
    reference: &Array2<f32>,
    moving: &Array2<f32>,
    bins: usize,
) -> Result<(HashMap<(usize, usize), f64>, HashMap<usize, f64>, HashMap<usize, f64>)> {
    // Find min/max values for histogram binning
    let ref_min = reference.iter().fold(f32::INFINITY, |a, &b| a.min(b)) as f64;
    let ref_max = reference.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let mov_min = moving.iter().fold(f32::INFINITY, |a, &b| a.min(b)) as f64;
    let mov_max = moving.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    
    let ref_range = ref_max - ref_min;
    let mov_range = mov_max - mov_min;
    
    if ref_range == 0.0 || mov_range == 0.0 {
        return Err(VisionError::InvalidParameter(
            "Image has no intensity variation".to_string(),
        ));
    }
    
    let mut joint_hist = HashMap::new();
    let mut ref_hist = HashMap::new();
    let mut mov_hist = HashMap::new();
    
    for (&r, &m) in reference.iter().zip(moving.iter()) {
        let ref_bin = ((r as f64 - ref_min) / ref_range * (bins - 1) as f64).round() as usize;
        let mov_bin = ((m as f64 - mov_min) / mov_range * (bins - 1) as f64).round() as usize;
        
        let ref_bin = ref_bin.min(bins - 1);
        let mov_bin = mov_bin.min(bins - 1);
        
        *joint_hist.entry((ref_bin, mov_bin)).or_insert(0.0) += 1.0;
        *ref_hist.entry(ref_bin).or_insert(0.0) += 1.0;
        *mov_hist.entry(mov_bin).or_insert(0.0) += 1.0;
    }
    
    Ok((joint_hist, ref_hist, mov_hist))
}

fn compute_gradient_magnitude(image: &Array2<f32>) -> Result<Array2<f32>> {
    let (height, width) = image.dim();
    let mut gradient = Array2::zeros((height, width));
    
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let gx = image[[y, x + 1]] - image[[y, x - 1]];
            let gy = image[[y + 1, x]] - image[[y - 1, x]];
            gradient[[y, x]] = (gx * gx + gy * gy).sqrt();
        }
    }
    
    Ok(gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_images() -> (Array2<f32>, Array2<f32>) {
        let mut reference = Array2::zeros((50, 50));
        let mut moving = Array2::zeros((50, 50));
        
        // Create a simple pattern
        for i in 10..40 {
            for j in 10..40 {
                reference[[i, j]] = 1.0;
                moving[[i + 2, j + 2]] = 1.0; // Shifted version
            }
        }
        
        (reference, moving)
    }

    #[test]
    fn test_sum_of_squared_differences() {
        let (reference, moving) = create_test_images();
        let ssd = SumOfSquaredDifferences;
        
        let result = ssd.compute(&reference, &moving).unwrap();
        assert!(result > 0.0);
        
        // Test with identical images
        let result_identical = ssd.compute(&reference, &reference).unwrap();
        assert_eq!(result_identical, 0.0);
    }

    #[test]
    fn test_normalized_cross_correlation() {
        let (reference, moving) = create_test_images();
        let ncc = NormalizedCrossCorrelation;
        
        let result = ncc.compute(&reference, &moving).unwrap();
        assert!(result >= -1.0 && result <= 1.0);
        
        // Test with identical images
        let result_identical = ncc.compute(&reference, &reference).unwrap();
        assert!((result_identical - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information() {
        let (reference, moving) = create_test_images();
        let mi = MutualInformation::new(16); // Use fewer bins for test
        
        let result = mi.compute(&reference, &moving).unwrap();
        assert!(result >= 0.0);
        
        // MI should be higher for identical images
        let result_identical = mi.compute(&reference, &reference).unwrap();
        assert!(result_identical >= result);
    }

    #[test]
    fn test_mean_squared_error() {
        let (reference, moving) = create_test_images();
        let mse = MeanSquaredError;
        
        let result = mse.compute(&reference, &moving).unwrap();
        assert!(result >= 0.0);
        
        // MSE should be 0 for identical images
        let result_identical = mse.compute(&reference, &reference).unwrap();
        assert_eq!(result_identical, 0.0);
    }

    #[test]
    fn test_peak_signal_to_noise_ratio() {
        let (reference, moving) = create_test_images();
        let psnr = PeakSignalToNoiseRatio::default();
        
        let result = psnr.compute(&reference, &moving).unwrap();
        assert!(result > 0.0);
        
        // PSNR should be infinite for identical images
        let result_identical = psnr.compute(&reference, &reference).unwrap();
        assert!(result_identical.is_infinite());
    }

    #[test]
    fn test_structural_similarity() {
        let reference = Array2::ones((20, 20));
        let moving = Array2::ones((20, 20));
        
        let ssim = StructuralSimilarity::default();
        let result = ssim.compute(&reference, &moving).unwrap();
        
        // SSIM should be 1 for identical images
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_correlation() {
        let (reference, moving) = create_test_images();
        let gc = GradientCorrelation;
        
        let result = gc.compute(&reference, &moving).unwrap();
        assert!(result >= -1.0 && result <= 1.0);
    }

    #[test]
    fn test_metric_normalization() {
        let (reference, moving) = create_test_images();
        
        let ssd = SumOfSquaredDifferences;
        let result = ssd.compute(&reference, &moving).unwrap();
        let normalized = ssd.normalize(result);
        assert!(normalized >= 0.0 && normalized <= 1.0);
        
        let ncc = NormalizedCrossCorrelation;
        let result = ncc.compute(&reference, &moving).unwrap();
        let normalized = ncc.normalize(result);
        assert!(normalized >= 0.0 && normalized <= 1.0);
    }

    #[test]
    fn test_create_similarity_measure() {
        let metric = SimilarityMetric::NormalizedCrossCorrelation;
        let measure = create_similarity_measure(metric);
        
        let reference = Array2::ones((10, 10));
        let moving = Array2::ones((10, 10));
        
        let result = measure.compute(&reference, &moving).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dimension_validation() {
        let reference = Array2::ones((10, 10));
        let moving = Array2::ones((5, 5));
        
        let ssd = SumOfSquaredDifferences;
        let result = ssd.compute(&reference, &moving);
        assert!(result.is_err());
    }
}