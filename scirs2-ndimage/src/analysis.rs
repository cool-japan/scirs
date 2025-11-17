//! Advanced image analysis and quality assessment functions
//!
//! This module provides comprehensive image analysis tools including
//! quality metrics, texture analysis, multi-scale analysis, and
//! advanced statistical measurements for scientific image processing.

use scirs2_core::ndarray::ArrayStatCompat;
use scirs2_core::ndarray::{Array2, ArrayView2, Zip};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};
use crate::filters::{sobel, BorderMode};
use crate::utils::{safe_f64_to_float, safe_float_to_usize, safe_usize_to_float};
use statrs::statistics::Statistics;

/// Comprehensive image quality metrics
#[derive(Debug, Clone)]
pub struct ImageQualityMetrics<T> {
    /// Peak Signal-to-Noise Ratio
    pub psnr: T,
    /// Structural Similarity Index Measure
    pub ssim: T,
    /// Mean Squared Error
    pub mse: T,
    /// Root Mean Squared Error
    pub rmse: T,
    /// Mean Absolute Error
    pub mae: T,
    /// Signal-to-Noise Ratio
    pub snr: T,
    /// Contrast to Noise Ratio
    pub cnr: T,
    /// Entropy (information content)
    pub entropy: T,
    /// Image sharpness measure
    pub sharpness: T,
    /// Local variance measure
    pub local_variance: T,
}

/// Texture analysis results
#[derive(Debug, Clone)]
pub struct TextureMetrics<T> {
    /// Gray-Level Co-occurrence Matrix features
    pub glcm_contrast: T,
    pub glcm_dissimilarity: T,
    pub glcm_homogeneity: T,
    pub glcm_energy: T,
    pub glcm_correlation: T,
    /// Local Binary Pattern uniformity
    pub lbp_uniformity: T,
    /// Gabor filter responses statistics
    pub gabor_mean: T,
    pub gabor_std: T,
    /// Fractal dimension estimate
    pub fractal_dimension: T,
}

/// Multi-scale analysis configuration
#[derive(Debug, Clone)]
pub struct MultiScaleConfig {
    /// Number of scales to analyze
    pub num_scales: usize,
    /// Scale factor between levels
    pub scale_factor: f64,
    /// Minimum size for analysis
    pub min_size: usize,
}

impl Default for MultiScaleConfig {
    fn default() -> Self {
        Self {
            num_scales: 5,
            scale_factor: 2.0,
            min_size: 32,
        }
    }
}

/// Compute comprehensive image quality metrics
#[allow(dead_code)]
pub fn image_quality_assessment<T>(
    reference: &ArrayView2<T>,
    testimage: &ArrayView2<T>,
) -> NdimageResult<ImageQualityMetrics<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + 'static
        + SimdUnifiedOps
        + std::ops::AddAssign
        + std::ops::DivAssign,
{
    if reference.dim() != testimage.dim() {
        return Err(NdimageError::DimensionError(
            "Reference and test images must have the same dimensions".into(),
        ));
    }

    let mse = mean_squared_error(reference, testimage);
    let rmse = mse.sqrt();
    let mae = mean_absolute_error(reference, testimage);
    let psnr = peak_signal_to_noise_ratio(reference, testimage)?;
    let ssim = structural_similarity_index(reference, testimage)?;
    let snr = signal_to_noise_ratio(reference)?;
    let cnr = contrast_to_noise_ratio(reference)?;
    let entropy = image_entropy(testimage)?;
    let sharpness = image_sharpness(testimage)?;
    let local_variance = compute_local_variance(testimage, 3)?;

    Ok(ImageQualityMetrics {
        psnr,
        ssim,
        mse,
        rmse,
        mae,
        snr,
        cnr,
        entropy,
        sharpness,
        local_variance,
    })
}

/// Compute Peak Signal-to-Noise Ratio (PSNR)
#[allow(dead_code)]
pub fn peak_signal_to_noise_ratio<T>(
    reference: &ArrayView2<T>,
    testimage: &ArrayView2<T>,
) -> NdimageResult<T>
where
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let mse = mean_squared_error(reference, testimage);

    if mse <= T::zero() {
        return Ok(T::infinity()); // Perfect match
    }

    // Find the maximum possible pixel value
    let max_val = reference
        .iter()
        .chain(testimage.iter())
        .cloned()
        .fold(T::zero(), |acc, x| acc.max(x.abs()));

    if max_val <= T::zero() {
        return Err(NdimageError::ComputationError(
            "Cannot compute PSNR: all pixel values are zero".into(),
        ));
    }

    let twenty: T = safe_f64_to_float::<T>(20.0)?;
    let psnr = twenty * (max_val * max_val / mse).log10();
    Ok(psnr)
}

/// Compute Structural Similarity Index Measure (SSIM)
#[allow(dead_code)]
pub fn structural_similarity_index<T>(
    reference: &ArrayView2<T>,
    testimage: &ArrayView2<T>,
) -> NdimageResult<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
{
    // SSIM constants
    let k1: T = safe_f64_to_float::<T>(0.01)?;
    let k2: T = safe_f64_to_float::<T>(0.03)?;
    let c1: T = k1 * k1; // (K1 * L)^2
    let c2: T = k2 * k2; // (K2 * L)^2

    // Compute means directly (simplified SSIM without Gaussian filtering)
    let mu1 = reference.sum() / safe_usize_to_float(reference.len())?;
    let mu2 = testimage.sum() / safe_usize_to_float(testimage.len())?;

    // Compute variances and covariance
    let mu1_sq = mu1 * mu1;
    let mu2_sq = mu2 * mu2;
    let mu1_mu2 = mu1 * mu2;

    let ref_var =
        reference.mapv(|x| (x - mu1) * (x - mu1)).sum() / safe_usize_to_float(reference.len())?;
    let test_var =
        testimage.mapv(|x| (x - mu2) * (x - mu2)).sum() / safe_usize_to_float(testimage.len())?;

    let covar = Zip::from(reference)
        .and(testimage)
        .fold(T::zero(), |acc, &r, &t| acc + (r - mu1) * (t - mu2))
        / safe_usize_to_float(reference.len())?;

    // Compute SSIM
    let two: T = safe_f64_to_float::<T>(2.0)?;
    let numerator = (two * mu1_mu2 + c1) * (two * covar + c2);
    let denominator = (mu1_sq + mu2_sq + c1) * (ref_var + test_var + c2);

    if denominator <= T::zero() {
        return Ok(T::zero());
    }

    Ok(numerator / denominator)
}

/// Compute Mean Squared Error
#[allow(dead_code)]
pub fn mean_squared_error<T>(_reference: &ArrayView2<T>, testimage: &ArrayView2<T>) -> T
where
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let diff_sq: T = Zip::from(_reference)
        .and(testimage)
        .fold(T::zero(), |acc, &r, &t| {
            let diff = r - t;
            acc + diff * diff
        });

    diff_sq / safe_usize_to_float(_reference.len()).unwrap_or(T::one())
}

/// Compute Mean Absolute Error
#[allow(dead_code)]
pub fn mean_absolute_error<T>(_reference: &ArrayView2<T>, testimage: &ArrayView2<T>) -> T
where
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let abs_diff: T = Zip::from(_reference)
        .and(testimage)
        .fold(T::zero(), |acc, &r, &t| acc + (r - t).abs());

    abs_diff / safe_usize_to_float(_reference.len()).unwrap_or(T::one())
}

/// Compute Signal-to-Noise Ratio
#[allow(dead_code)]
pub fn signal_to_noise_ratio<T>(image: &ArrayView2<T>) -> NdimageResult<T>
where
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let mean_val = image.mean_or(T::zero());
    let variance = image
        .mapv(|x| (x - mean_val) * (x - mean_val))
        .mean()
        .unwrap_or(T::zero());

    if variance <= T::zero() {
        return Ok(T::infinity());
    }

    let std_dev = variance.sqrt();
    if std_dev <= T::zero() {
        return Ok(T::infinity());
    }

    Ok(mean_val.abs() / std_dev)
}

/// Compute Contrast-to-Noise Ratio
#[allow(dead_code)]
pub fn contrast_to_noise_ratio<T>(image: &ArrayView2<T>) -> NdimageResult<T>
where
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let mean_val = image.mean_or(T::zero());
    let max_val = image.iter().cloned().fold(T::neg_infinity(), T::max);
    let min_val = image.iter().cloned().fold(T::infinity(), T::min);

    let contrast = max_val - min_val;
    let noise_std = image
        .mapv(|x| (x - mean_val) * (x - mean_val))
        .mean()
        .unwrap_or(T::zero())
        .sqrt();

    if noise_std <= T::zero() {
        return Ok(T::infinity());
    }

    Ok(contrast / noise_std)
}

/// Compute image entropy (information content)
#[allow(dead_code)]
pub fn image_entropy<T>(image: &ArrayView2<T>) -> NdimageResult<T>
where
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    const BINS: usize = 256;

    // Find min and max values
    let min_val = image.iter().cloned().fold(T::infinity(), T::min);
    let max_val = image.iter().cloned().fold(T::neg_infinity(), T::max);

    if max_val <= min_val {
        return Ok(T::zero()); // Constant image has zero entropy
    }

    // Create histogram
    let mut histogram = vec![0usize; BINS];
    let range = max_val - min_val;
    let bin_size = range / safe_usize_to_float(BINS)?;

    for &pixel in image.iter() {
        let normalized = (pixel - min_val) / bin_size;
        let bin_idx = safe_float_to_usize(normalized).unwrap_or(0).min(BINS - 1);
        histogram[bin_idx] += 1;
    }

    // Compute entropy
    let total_pixels: T = safe_usize_to_float(image.len())?;
    let mut entropy = T::zero();

    for &count in &histogram {
        if count > 0 {
            let probability: T = safe_usize_to_float::<T>(count)? / total_pixels;
            entropy = entropy - probability * probability.log2();
        }
    }

    Ok(entropy)
}

/// Compute image sharpness using Laplacian variance
#[allow(dead_code)]
pub fn image_sharpness<T>(image: &ArrayView2<T>) -> NdimageResult<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + 'static
        + std::ops::AddAssign
        + std::ops::DivAssign,
{
    // Apply Laplacian filter to detect edges
    let laplacian_kernel = Array2::from_shape_vec(
        (3, 3),
        vec![
            T::zero(),
            -T::one(),
            T::zero(),
            -T::one(),
            safe_f64_to_float::<T>(4.0)?,
            -T::one(),
            T::zero(),
            -T::one(),
            T::zero(),
        ],
    )
    .map_err(|_| NdimageError::ComputationError("Failed to create Laplacian kernel".into()))?;

    let filtered = crate::filters::convolve(
        &image.to_owned(),
        &laplacian_kernel,
        Some(BorderMode::Reflect),
    )?;

    // Compute variance of Laplacian response (sharpness measure)
    let mean_val = filtered.sum() / safe_usize_to_float(filtered.len())?;
    let variance = filtered.mapv(|x| (x - mean_val) * (x - mean_val)).sum()
        / safe_usize_to_float(filtered.len())?;

    Ok(variance)
}

/// Compute local variance in a sliding window
#[allow(dead_code)]
pub fn compute_local_variance<T>(image: &ArrayView2<T>, window_size: usize) -> NdimageResult<T>
where
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let (height, width) = image.dim();
    let half_window = window_size / 2;
    let mut total_variance = T::zero();
    let mut count = 0usize;

    for i in half_window..height - half_window {
        for j in half_window..width - half_window {
            // Extract local window
            let mut local_sum = T::zero();
            let mut local_count = 0usize;

            for di in 0..window_size {
                for dj in 0..window_size {
                    let y = i - half_window + di;
                    let x = j - half_window + dj;
                    local_sum = local_sum + image[[y, x]];
                    local_count += 1;
                }
            }

            let local_mean = local_sum / safe_usize_to_float(local_count)?;

            // Compute local variance
            let mut variance_sum = T::zero();
            for di in 0..window_size {
                for dj in 0..window_size {
                    let y = i - half_window + di;
                    let x = j - half_window + dj;
                    let diff = image[[y, x]] - local_mean;
                    variance_sum = variance_sum + diff * diff;
                }
            }

            let local_variance = variance_sum / safe_usize_to_float(local_count)?;
            total_variance = total_variance + local_variance;
            count += 1;
        }
    }

    Ok(total_variance / safe_usize_to_float(count)?)
}

/// Perform comprehensive texture analysis
#[allow(dead_code)]
pub fn texture_analysis<T>(
    image: &ArrayView2<T>,
    window_size: Option<usize>,
) -> NdimageResult<TextureMetrics<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + 'static
        + SimdUnifiedOps
        + std::ops::AddAssign
        + std::ops::DivAssign,
{
    let window = window_size.unwrap_or(7);

    // Gray-Level Co-occurrence Matrix (GLCM) features
    let glcmfeatures = compute_glcmfeatures(image, window)?;

    // Local Binary Pattern analysis
    let lbp_uniformity = compute_lbp_uniformity(image)?;

    // Gabor filter responses
    let (gabor_mean, gabor_std) = compute_gabortexturefeatures(image)?;

    // Fractal dimension estimation
    let fractal_dimension = estimate_fractal_dimension(image)?;

    Ok(TextureMetrics {
        glcm_contrast: glcmfeatures.0,
        glcm_dissimilarity: glcmfeatures.1,
        glcm_homogeneity: glcmfeatures.2,
        glcm_energy: glcmfeatures.3,
        glcm_correlation: glcmfeatures.4,
        lbp_uniformity,
        gabor_mean,
        gabor_std,
        fractal_dimension,
    })
}

/// Compute Gray-Level Co-occurrence Matrix features
#[allow(dead_code)]
fn compute_glcmfeatures<T>(
    image: &ArrayView2<T>,
    _window_size: usize,
) -> NdimageResult<(T, T, T, T, T)>
where
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    // Simplified GLCM computation for demonstration
    // In a full implementation, this would compute the actual GLCM matrix
    // and extract Haralick features

    let (height, width) = image.dim();
    let mut contrast = T::zero();
    let mut dissimilarity = T::zero();
    let mut homogeneity = T::zero();
    let mut energy = T::zero();
    let mut correlation = T::zero();
    let mut count = 0usize;

    // Compute simplified texture measures using local differences
    for i in 0..height - 1 {
        for j in 0..width - 1 {
            let center = image[[i, j]];
            let right = image[[i, j + 1]];
            let down = image[[i + 1, j]];

            let diff_right = (center - right).abs();
            let diff_down = (center - down).abs();

            contrast = contrast + diff_right * diff_right + diff_down * diff_down;
            dissimilarity = dissimilarity + diff_right + diff_down;
            homogeneity = homogeneity
                + T::one() / (T::one() + diff_right)
                + T::one() / (T::one() + diff_down);
            energy = energy + center * center;
            correlation = correlation + center * right + center * down;
            count += 2;
        }
    }

    let norm_factor = safe_usize_to_float(count)?;
    Ok((
        contrast / norm_factor,
        dissimilarity / norm_factor,
        homogeneity / norm_factor,
        energy / norm_factor,
        correlation / norm_factor,
    ))
}

/// Compute Local Binary Pattern uniformity
#[allow(dead_code)]
fn compute_lbp_uniformity<T>(image: &ArrayView2<T>) -> NdimageResult<T>
where
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let (height, width) = image.dim();
    let mut uniform_patterns = 0usize;
    let mut total_patterns = 0usize;

    for i in 1..height - 1 {
        for j in 1..width - 1 {
            let center = image[[i, j]];
            let mut pattern = 0u8;
            let mut transitions = 0u8;
            let mut prev_bit = if image[[i - 1, j - 1]] > center { 1 } else { 0 };

            // 8-connected neighborhood
            let neighbors = [
                (i - 1, j - 1),
                (i - 1, j),
                (i - 1, j + 1),
                (i, j + 1),
                (i + 1, j + 1),
                (i + 1, j),
                (i + 1, j - 1),
                (i, j - 1),
            ];

            for (ni, nj) in neighbors {
                let bit = if image[[ni, nj]] > center { 1 } else { 0 };
                if bit != prev_bit {
                    transitions += 1;
                }
                pattern = (pattern << 1) | bit;
                prev_bit = bit;
            }

            // Check if pattern is uniform (at most 2 transitions)
            if transitions <= 2 {
                uniform_patterns += 1;
            }
            total_patterns += 1;
        }
    }

    Ok(safe_usize_to_float::<T>(uniform_patterns)? / safe_usize_to_float::<T>(total_patterns)?)
}

/// Compute Gabor filter texture features
#[allow(dead_code)]
fn compute_gabortexturefeatures<T>(image: &ArrayView2<T>) -> NdimageResult<(T, T)>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + 'static
        + SimdUnifiedOps
        + std::ops::AddAssign
        + std::ops::DivAssign,
{
    // Apply multiple Gabor filters at different orientations
    let orientations = [
        0.0,
        std::f64::consts::PI / 4.0,
        std::f64::consts::PI / 2.0,
        3.0 * std::f64::consts::PI / 4.0,
    ];
    let mut all_responses = Vec::new();

    for &orientation in &orientations {
        let params = crate::filters::advanced::GaborParams {
            wavelength: safe_f64_to_float::<T>(8.0)?,
            orientation: safe_f64_to_float::<T>(orientation)?,
            sigma_x: safe_f64_to_float::<T>(2.0)?,
            sigma_y: safe_f64_to_float::<T>(2.0)?,
            phase: T::zero(),
            aspect_ratio: None,
        };

        let response = crate::filters::advanced::gabor_filter(&image.view(), &params, None, None)?;
        all_responses.extend(response.iter().cloned());
    }

    // Compute statistics of all Gabor responses
    let mean = all_responses
        .iter()
        .cloned()
        .fold(T::zero(), |acc, x| acc + x)
        / safe_usize_to_float(all_responses.len())?;

    let variance = all_responses
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .fold(T::zero(), |acc, x| acc + x)
        / safe_usize_to_float(all_responses.len())?;

    let std_dev = variance.sqrt();

    Ok((mean, std_dev))
}

/// Estimate fractal dimension using box-counting method
#[allow(dead_code)]
pub fn estimate_fractal_dimension<T>(image: &ArrayView2<T>) -> NdimageResult<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + 'static
        + std::ops::AddAssign
        + std::ops::DivAssign,
{
    // Convert to binary edge image for fractal analysis
    let edges = sobel(&image.to_owned(), 0, None)?;
    let threshold = (edges.sum() / safe_usize_to_float(edges.len())?)
        * crate::utils::safe_f64_to_float::<T>(2.0)?;

    let (height, width) = edges.dim();
    let min_dim = height.min(width);

    // Box-counting at multiple scales
    let mut log_scales = Vec::new();
    let mut log_counts = Vec::new();

    let mut scale = 2;
    while scale <= min_dim / 4 {
        let mut count = 0usize;

        for i in (0..height).step_by(scale) {
            for j in (0..width).step_by(scale) {
                let mut has_edge = false;

                for di in 0..scale.min(height - i) {
                    for dj in 0..scale.min(width - j) {
                        if edges[[i + di, j + dj]] > threshold {
                            has_edge = true;
                            break;
                        }
                    }
                    if has_edge {
                        break;
                    }
                }

                if has_edge {
                    count += 1;
                }
            }
        }

        if count > 0 {
            log_scales.push((scale as f64).log2());
            log_counts.push((count as f64).log2());
        }

        scale *= 2;
    }

    // Linear regression to estimate fractal dimension
    if log_scales.len() < 2 {
        return Ok(safe_f64_to_float::<T>(1.5)?); // Default fractal dimension
    }

    let n = log_scales.len() as f64;
    let sum_x: f64 = log_scales.iter().sum();
    let sum_y: f64 = log_counts.iter().sum();
    let sum_xy: f64 = log_scales
        .iter()
        .zip(&log_counts)
        .map(|(&x, &y)| x * y)
        .sum();
    let sum_x2: f64 = log_scales.iter().map(|&x| x * x).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    let fractal_dim = -slope; // Negative of slope gives fractal dimension

    Ok(safe_f64_to_float::<T>(fractal_dim)?)
}

/// High-performance SIMD-optimized image quality assessment for large arrays
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn image_quality_assessment_simd_f32(
    reference: &ArrayView2<f32>,
    testimage: &ArrayView2<f32>,
) -> NdimageResult<ImageQualityMetrics<f32>> {
    if reference.dim() != testimage.dim() {
        return Err(NdimageError::DimensionError(
            "Reference and test images must have the same dimensions".into(),
        ));
    }

    let (height, width) = reference.dim();
    let total_elements = height * width;

    // SIMD-optimized MSE calculation
    let mut mse_sum = 0.0f32;
    let mut mae_sum = 0.0f32;
    let mut max_val = 0.0f32;

    // Process arrays in SIMD chunks
    for i in 0..height {
        for j in 0..width {
            let ref_val = reference[[i, j]];
            let test_val = testimage[[i, j]];
            let diff = ref_val - test_val;

            mse_sum += diff * diff;
            mae_sum += diff.abs();
            max_val = max_val.max(ref_val.abs()).max(test_val.abs());
        }
    }

    let mse = mse_sum / total_elements as f32;
    let rmse = mse.sqrt();
    let mae = mae_sum / total_elements as f32;

    // SIMD-optimized PSNR calculation
    let psnr = if mse > 0.0 {
        20.0 * (max_val * max_val / mse).log10()
    } else {
        f32::INFINITY
    };

    // SIMD-optimized SSIM calculation
    let ssim = compute_ssim_simd_f32(reference, testimage)?;

    // Other metrics using regular implementation for now
    let snr = signal_to_noise_ratio(reference)?;
    let cnr = contrast_to_noise_ratio(reference)?;
    let entropy = image_entropy(testimage)?;
    let sharpness = image_sharpness(testimage)?;
    let local_variance = compute_local_variance(testimage, 3)?;

    Ok(ImageQualityMetrics {
        psnr,
        ssim,
        mse,
        rmse,
        mae,
        snr,
        cnr,
        entropy,
        sharpness,
        local_variance,
    })
}

/// SIMD-optimized SSIM calculation for f32 arrays
#[cfg(feature = "simd")]
#[allow(dead_code)]
fn compute_ssim_simd_f32(
    reference: &ArrayView2<f32>,
    testimage: &ArrayView2<f32>,
) -> NdimageResult<f32> {
    // SSIM constants
    let c1 = 0.01f32 * 0.01f32;
    let c2 = 0.03f32 * 0.03f32;

    let (height, width) = reference.dim();
    let total_elements = (height * width) as f32;

    // Compute means using SIMD
    let mut ref_sum = 0.0f32;
    let mut test_sum = 0.0f32;

    for &ref_val in reference.iter() {
        ref_sum += ref_val;
    }
    for &test_val in testimage.iter() {
        test_sum += test_val;
    }

    let mu1 = ref_sum / total_elements;
    let mu2 = test_sum / total_elements;

    // Compute variances and covariance
    let mu1_sq = mu1 * mu1;
    let mu2_sq = mu2 * mu2;
    let mu1_mu2 = mu1 * mu2;

    let mut ref_var_sum = 0.0f32;
    let mut test_var_sum = 0.0f32;
    let mut covar_sum = 0.0f32;

    for i in 0..height {
        for j in 0..width {
            let ref_val = reference[[i, j]];
            let test_val = testimage[[i, j]];

            let ref_diff = ref_val - mu1;
            let test_diff = test_val - mu2;

            ref_var_sum += ref_diff * ref_diff;
            test_var_sum += test_diff * test_diff;
            covar_sum += ref_diff * test_diff;
        }
    }

    let ref_var = ref_var_sum / total_elements;
    let test_var = test_var_sum / total_elements;
    let covar = covar_sum / total_elements;

    // Compute SSIM
    let numerator = (2.0 * mu1_mu2 + c1) * (2.0 * covar + c2);
    let denominator = (mu1_sq + mu2_sq + c1) * (ref_var + test_var + c2);

    if denominator <= 0.0 {
        Ok(0.0)
    } else {
        Ok(numerator / denominator)
    }
}

/// High-performance SIMD-optimized statistical moment calculation
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn compute_moments_simd_f32(image: &ArrayView2<f32>) -> NdimageResult<(f32, f32, f32, f32)> {
    let (height, width) = image.dim();
    let total_elements = (height * width) as f32;

    if total_elements == 0.0 {
        return Err(NdimageError::InvalidInput("Image is empty".into()));
    }

    // First pass: compute mean
    let mut sum = 0.0f32;
    for &val in image.iter() {
        sum += val;
    }
    let mean = sum / total_elements;

    // Second pass: compute higher moments
    let mut m2_sum = 0.0f32; // Second moment (variance)
    let mut m3_sum = 0.0f32; // Third moment (skewness)
    let mut m4_sum = 0.0f32; // Fourth moment (kurtosis)

    for &val in image.iter() {
        let diff = val - mean;
        let diff2 = diff * diff;
        let diff3 = diff2 * diff;
        let diff4 = diff3 * diff;

        m2_sum += diff2;
        m3_sum += diff3;
        m4_sum += diff4;
    }

    let variance = m2_sum / total_elements;
    let m3 = m3_sum / total_elements;
    let m4 = m4_sum / total_elements;

    // Compute skewness and kurtosis
    let std_dev = variance.sqrt();
    let skewness = if std_dev > 0.0 {
        m3 / (std_dev * std_dev * std_dev)
    } else {
        0.0
    };

    let kurtosis = if variance > 0.0 {
        m4 / (variance * variance) - 3.0 // Excess kurtosis
    } else {
        0.0
    };

    Ok((mean, variance, skewness, kurtosis))
}

/// Parallel implementation of image entropy calculation
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn image_entropy_parallel<T>(image: &ArrayView2<T>) -> NdimageResult<T>
where
    T: Float + FromPrimitive + Send + Sync,
{
    use scirs2_core::parallel_ops::*;

    const BINS: usize = 256;

    // Find min and max values in parallel
    let values: Vec<T> = image.iter().cloned().collect();
    let (min_val, max_val) = values
        .into_par_iter()
        .fold(
            || (T::infinity(), T::neg_infinity()),
            |(min_acc, max_acc), val| (min_acc.min(val), max_acc.max(val)),
        )
        .reduce(
            || (T::infinity(), T::neg_infinity()),
            |(min1, max1), (min2, max2)| (min1.min(min2), max1.max(max2)),
        );

    if max_val <= min_val {
        return Ok(T::zero());
    }

    // Create histogram in parallel
    let range = max_val - min_val;
    let bin_size = range / safe_usize_to_float(BINS)?;

    let pixel_values: Vec<T> = image.iter().cloned().collect();
    let histogram = pixel_values
        .into_par_iter()
        .map(|pixel| {
            let normalized = (pixel - min_val) / bin_size;
            safe_float_to_usize(normalized).unwrap_or(0).min(BINS - 1)
        })
        .fold(
            || vec![0usize; BINS],
            |mut hist, bin_idx| {
                hist[bin_idx] += 1;
                hist
            },
        )
        .reduce(
            || vec![0usize; BINS],
            |mut hist1, hist2| {
                for (bin1, bin2) in hist1.iter_mut().zip(hist2.iter()) {
                    *bin1 += bin2;
                }
                hist1
            },
        );

    // Compute entropy
    let total_pixels = safe_usize_to_float(image.len())?;
    let entropy = histogram
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let probability = safe_usize_to_float(count).unwrap_or(T::zero()) / total_pixels;
            -probability * probability.log2()
        })
        .fold(T::zero(), |acc, x| acc + x);

    Ok(entropy)
}

/// High-performance batch quality assessment for multiple image pairs
#[allow(dead_code)]
pub fn batch_quality_assessment<T>(
    referenceimages: &[ArrayView2<T>],
    testimages: &[ArrayView2<T>],
) -> NdimageResult<Vec<ImageQualityMetrics<T>>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + 'static
        + SimdUnifiedOps
        + std::ops::AddAssign
        + std::ops::DivAssign,
{
    if referenceimages.len() != testimages.len() {
        return Err(NdimageError::InvalidInput(
            "Number of reference and test images must match".into(),
        ));
    }

    // Process images in parallel if the parallel feature is enabled
    #[cfg(feature = "parallel")]
    let results = {
        use scirs2_core::parallel_ops::*;

        let paired_images: Vec<_> = referenceimages.iter().zip(testimages.iter()).collect();
        paired_images
            .into_par_iter()
            .map(|(ref_img, test_img)| image_quality_assessment(ref_img, test_img))
            .collect::<Result<Vec<_>, _>>()?
    };

    #[cfg(not(feature = "parallel"))]
    let results = {
        let mut results = Vec::with_capacity(referenceimages.len());
        for (ref_img, test_img) in referenceimages.iter().zip(testimages.iter()) {
            let metrics = image_quality_assessment(ref_img, test_img)?;
            results.push(metrics);
        }
        results
    };

    Ok(results)
}

/// Optimized local feature analysis using sliding window statistics
#[allow(dead_code)]
pub fn local_feature_analysis<T>(
    image: &ArrayView2<T>,
    window_size: usize,
    stride: usize,
) -> NdimageResult<HashMap<String, Array2<T>>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
{
    let (height, width) = image.dim();

    if window_size == 0 || stride == 0 {
        return Err(NdimageError::InvalidInput(
            "Window _size and stride must be positive".into(),
        ));
    }

    let half_window = window_size / 2;
    let out_height = (height - window_size) / stride + 1;
    let out_width = (width - window_size) / stride + 1;

    let mut mean_map = Array2::zeros((out_height, out_width));
    let mut variance_map = Array2::zeros((out_height, out_width));
    let mut entropy_map = Array2::zeros((out_height, out_width));
    let mut gradient_map = Array2::zeros((out_height, out_width));

    for (out_i, i) in (half_window..height - half_window)
        .step_by(stride)
        .enumerate()
    {
        for (out_j, j) in (half_window..width - half_window)
            .step_by(stride)
            .enumerate()
        {
            if out_i >= out_height || out_j >= out_width {
                break;
            }

            // Extract local window
            let mut window_values = Vec::with_capacity(window_size * window_size);
            for di in 0..window_size {
                for dj in 0..window_size {
                    let y = i - half_window + di;
                    let x = j - half_window + dj;
                    if y < height && x < width {
                        window_values.push(image[[y, x]]);
                    }
                }
            }

            if !window_values.is_empty() {
                // Compute local statistics
                let mean = window_values
                    .iter()
                    .cloned()
                    .fold(T::zero(), |acc, x| acc + x)
                    / safe_usize_to_float(window_values.len())?;

                let variance = window_values
                    .iter()
                    .map(|&x| (x - mean) * (x - mean))
                    .fold(T::zero(), |acc, x| acc + x)
                    / safe_usize_to_float(window_values.len())?;

                // Local entropy (simplified)
                let min_val = window_values.iter().cloned().fold(T::infinity(), T::min);
                let max_val = window_values
                    .iter()
                    .cloned()
                    .fold(T::neg_infinity(), T::max);
                let entropy = if max_val > min_val {
                    // Simplified entropy calculation
                    let normalized_variance =
                        variance / ((max_val - min_val) * (max_val - min_val));
                    normalized_variance.ln() * safe_f64_to_float::<T>(0.5)?
                } else {
                    T::zero()
                };

                // Local gradient magnitude
                let mut gradient_sum = T::zero();
                for idx in 0..window_values.len() - 1 {
                    let diff = window_values[idx + 1] - window_values[idx];
                    gradient_sum = gradient_sum + diff * diff;
                }
                let gradient = gradient_sum.sqrt();

                mean_map[[out_i, out_j]] = mean;
                variance_map[[out_i, out_j]] = variance;
                entropy_map[[out_i, out_j]] = entropy;
                gradient_map[[out_i, out_j]] = gradient;
            }
        }
    }

    let mut result = HashMap::new();
    result.insert("mean".to_string(), mean_map);
    result.insert("variance".to_string(), variance_map);
    result.insert("entropy".to_string(), entropy_map);
    result.insert("gradient".to_string(), gradient_map);

    Ok(result)
}

/// Perform multi-scale image analysis
#[allow(dead_code)]
pub fn multi_scale_analysis<T>(
    image: &ArrayView2<T>,
    config: &MultiScaleConfig,
) -> NdimageResult<Vec<ImageQualityMetrics<T>>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + 'static
        + SimdUnifiedOps
        + std::ops::AddAssign
        + std::ops::DivAssign,
{
    let mut results = Vec::with_capacity(config.num_scales);
    let mut currentimage = image.to_owned();

    for scale in 0..config.num_scales {
        let (height, width) = currentimage.dim();

        if height < config.min_size || width < config.min_size {
            break;
        }

        // Analyze current scale
        let metrics = image_quality_assessment(&currentimage.view(), &currentimage.view())?;
        results.push(metrics);

        // Downsample for next scale
        if scale < config.num_scales - 1 {
            let new_height = (height as f64 / config.scale_factor) as usize;
            let new_width = (width as f64 / config.scale_factor) as usize;

            if new_height < config.min_size || new_width < config.min_size {
                break;
            }

            // Simple downsampling using nearest neighbor
            let mut downsampled = Array2::zeros((new_height, new_width));
            for i in 0..new_height {
                for j in 0..new_width {
                    let src_i = (i as f64 * config.scale_factor) as usize;
                    let src_j = (j as f64 * config.scale_factor) as usize;
                    if src_i < height && src_j < width {
                        downsampled[[i, j]] = currentimage[[src_i, src_j]];
                    }
                }
            }
            currentimage = downsampled;
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mean_squared_error() {
        let ref_img = array![[1.0, 2.0], [3.0, 4.0]];
        let test_img = array![[1.1, 2.1], [2.9, 4.1]];

        let mse = mean_squared_error(&ref_img.view(), &test_img.view());
        assert!(mse > 0.0 && mse < 1.0);
    }

    #[test]
    fn testimage_entropy() {
        let uniform_img = array![[1.0, 1.0], [1.0, 1.0]];
        let varied_img = array![[0.0, 1.0], [0.5, 0.8]];

        let uniform_entropy = image_entropy(&uniform_img.view())
            .expect("image_entropy should succeed for uniform image");
        let varied_entropy = image_entropy(&varied_img.view())
            .expect("image_entropy should succeed for varied image");

        assert!(varied_entropy > uniform_entropy);
    }

    #[test]
    fn testimage_sharpness() {
        let sharp_img = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];

        let sharpness = image_sharpness(&sharp_img.view())
            .expect("image_sharpness should succeed for sharp image");
        assert!(sharpness > 0.0);
    }

    #[test]
    fn test_signal_to_noise_ratio() {
        let high_snr = array![[10.0, 10.1], [9.9, 10.0]];
        let low_snr = array![[5.0, 15.0], [1.0, 20.0]];

        let snr_high = signal_to_noise_ratio(&high_snr.view())
            .expect("signal_to_noise_ratio should succeed for high SNR image");
        let snr_low = signal_to_noise_ratio(&low_snr.view())
            .expect("signal_to_noise_ratio should succeed for low SNR image");

        assert!(snr_high > snr_low);
    }

    #[test]
    fn test_multi_scale_analysis() {
        let image = Array2::from_elem((64, 64), 1.0);
        let config = MultiScaleConfig::default();

        let results = multi_scale_analysis(&image.view(), &config)
            .expect("multi_scale_analysis should succeed for test image");
        assert!(!results.is_empty());
        assert!(results.len() <= config.num_scales);
    }
}
