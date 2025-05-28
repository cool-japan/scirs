//! Template matching for object detection
//!
//! This module provides various template matching methods to find regions
//! in an image that match a template image.

use crate::error::{Result, VisionError};
use image::{DynamicImage, GenericImageView, GrayImage, Rgb, RgbImage};
use ndarray::{s, Array2};
use rayon::prelude::*;

/// Template matching method
#[derive(Debug, Clone, Copy)]
pub enum MatchMethod {
    /// Sum of Squared Differences (SSD)
    SumSquaredDiff,
    /// Normalized Sum of Squared Differences
    NormalizedSumSquaredDiff,
    /// Cross-Correlation
    CrossCorrelation,
    /// Normalized Cross-Correlation
    NormalizedCrossCorrelation,
    /// Correlation Coefficient
    CorrelationCoeff,
    /// Normalized Correlation Coefficient
    NormalizedCorrelationCoeff,
}

/// Match result containing position and score
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// X coordinate of the match
    pub x: u32,
    /// Y coordinate of the match
    pub y: u32,
    /// Match score (higher is better)
    pub score: f32,
}

/// Perform template matching
///
/// # Arguments
///
/// * `img` - Source image to search in
/// * `template` - Template image to find
/// * `method` - Matching method to use
///
/// # Returns
///
/// * Result containing array of match scores
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::{template_match, MatchMethod};
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img = image::open("examples/input/input.jpg").unwrap();
/// let template = img.crop_imm(50, 50, 30, 30);
/// let scores = template_match(&img, &template, MatchMethod::NormalizedCrossCorrelation)?;
/// # Ok(())
/// # }
/// ```
pub fn template_match(
    img: &DynamicImage,
    template: &DynamicImage,
    method: MatchMethod,
) -> Result<Array2<f32>> {
    let gray_img = img.to_luma8();
    let gray_template = template.to_luma8();

    match method {
        MatchMethod::SumSquaredDiff => match_ssd(&gray_img, &gray_template),
        MatchMethod::NormalizedSumSquaredDiff => match_normalized_ssd(&gray_img, &gray_template),
        MatchMethod::CrossCorrelation => match_cross_correlation(&gray_img, &gray_template),
        MatchMethod::NormalizedCrossCorrelation => match_ncc(&gray_img, &gray_template),
        MatchMethod::CorrelationCoeff => match_correlation_coeff(&gray_img, &gray_template),
        MatchMethod::NormalizedCorrelationCoeff => {
            match_normalized_correlation_coeff(&gray_img, &gray_template)
        }
    }
}

/// Sum of Squared Differences matching
fn match_ssd(img: &GrayImage, template: &GrayImage) -> Result<Array2<f32>> {
    let (img_width, img_height) = img.dimensions();
    let (tmpl_width, tmpl_height) = template.dimensions();

    if tmpl_width > img_width || tmpl_height > img_height {
        return Err(VisionError::InvalidParameter(
            "Template larger than image".to_string(),
        ));
    }

    let result_width = (img_width - tmpl_width + 1) as usize;
    let result_height = (img_height - tmpl_height + 1) as usize;
    let mut result = Array2::zeros((result_height, result_width));

    // Convert to arrays for faster access
    let img_array = image_to_array(img);
    let tmpl_array = image_to_array(template);

    // Parallel computation
    let scores: Vec<_> = (0..result_height)
        .into_par_iter()
        .flat_map(|y| {
            let img_slice = img_array.view();
            let tmpl_slice = tmpl_array.view();
            (0..result_width)
                .into_par_iter()
                .map(move |x| {
                    let mut ssd = 0.0f32;
                    for ty in 0..tmpl_height as usize {
                        for tx in 0..tmpl_width as usize {
                            let img_val = img_slice[[y + ty, x + tx]];
                            let tmpl_val = tmpl_slice[[ty, tx]];
                            let diff = img_val - tmpl_val;
                            ssd += diff * diff;
                        }
                    }
                    (y, x, ssd)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Fill result array
    for (y, x, score) in scores {
        result[[y, x]] = score;
    }

    Ok(result)
}

/// Normalized Sum of Squared Differences
fn match_normalized_ssd(img: &GrayImage, template: &GrayImage) -> Result<Array2<f32>> {
    let ssd_result = match_ssd(img, template)?;
    let (height, width) = ssd_result.dim();

    // Compute template norm
    let tmpl_array = image_to_array(template);
    let tmpl_norm: f32 = tmpl_array.iter().map(|&v| v * v).sum();

    let mut result = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            if tmpl_norm > 0.0 {
                result[[y, x]] = ssd_result[[y, x]] / tmpl_norm.sqrt();
            }
        }
    }

    Ok(result)
}

/// Cross-correlation matching
fn match_cross_correlation(img: &GrayImage, template: &GrayImage) -> Result<Array2<f32>> {
    let (img_width, img_height) = img.dimensions();
    let (tmpl_width, tmpl_height) = template.dimensions();

    if tmpl_width > img_width || tmpl_height > img_height {
        return Err(VisionError::InvalidParameter(
            "Template larger than image".to_string(),
        ));
    }

    let result_width = (img_width - tmpl_width + 1) as usize;
    let result_height = (img_height - tmpl_height + 1) as usize;

    let img_array = image_to_array(img);
    let tmpl_array = image_to_array(template);

    // Parallel computation
    let scores: Vec<_> = (0..result_height)
        .into_par_iter()
        .flat_map(|y| {
            let img_slice = img_array.view();
            let tmpl_slice = tmpl_array.view();
            (0..result_width)
                .into_par_iter()
                .map(move |x| {
                    let mut correlation = 0.0f32;
                    for ty in 0..tmpl_height as usize {
                        for tx in 0..tmpl_width as usize {
                            correlation += img_slice[[y + ty, x + tx]] * tmpl_slice[[ty, tx]];
                        }
                    }
                    (y, x, correlation)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut result = Array2::zeros((result_height, result_width));
    for (y, x, score) in scores {
        result[[y, x]] = score;
    }

    Ok(result)
}

/// Normalized Cross-Correlation
fn match_ncc(img: &GrayImage, template: &GrayImage) -> Result<Array2<f32>> {
    let (img_width, img_height) = img.dimensions();
    let (tmpl_width, tmpl_height) = template.dimensions();

    if tmpl_width > img_width || tmpl_height > img_height {
        return Err(VisionError::InvalidParameter(
            "Template larger than image".to_string(),
        ));
    }

    let result_width = (img_width - tmpl_width + 1) as usize;
    let result_height = (img_height - tmpl_height + 1) as usize;

    let img_array = image_to_array(img);
    let tmpl_array = image_to_array(template);

    // Compute template mean and norm
    let tmpl_mean: f32 = tmpl_array.mean().unwrap_or(0.0);
    let tmpl_norm: f32 = tmpl_array
        .iter()
        .map(|&v| {
            let diff = v - tmpl_mean;
            diff * diff
        })
        .sum::<f32>()
        .sqrt();

    // Parallel computation
    let scores: Vec<_> = (0..result_height)
        .into_par_iter()
        .flat_map(|y| {
            let img_slice = img_array.view();
            let tmpl_slice = tmpl_array.view();
            (0..result_width)
                .into_par_iter()
                .map(move |x| {
                    // Extract patch
                    let patch = img_slice
                        .slice(s![y..y + tmpl_height as usize, x..x + tmpl_width as usize]);
                    let patch_mean: f32 = patch.mean().unwrap_or(0.0);

                    let mut correlation = 0.0f32;
                    let mut patch_norm = 0.0f32;

                    for ty in 0..tmpl_height as usize {
                        for tx in 0..tmpl_width as usize {
                            let img_val = img_slice[[y + ty, x + tx]] - patch_mean;
                            let tmpl_val = tmpl_slice[[ty, tx]] - tmpl_mean;
                            correlation += img_val * tmpl_val;
                            patch_norm += img_val * img_val;
                        }
                    }

                    patch_norm = patch_norm.sqrt();

                    let ncc = if patch_norm > 0.0 && tmpl_norm > 0.0 {
                        correlation / (patch_norm * tmpl_norm)
                    } else {
                        0.0
                    };

                    (y, x, ncc)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut result = Array2::zeros((result_height, result_width));
    for (y, x, score) in scores {
        result[[y, x]] = score;
    }

    Ok(result)
}

/// Correlation coefficient matching
fn match_correlation_coeff(img: &GrayImage, template: &GrayImage) -> Result<Array2<f32>> {
    match_ncc(img, template)
}

/// Normalized correlation coefficient
fn match_normalized_correlation_coeff(
    img: &GrayImage,
    template: &GrayImage,
) -> Result<Array2<f32>> {
    let ncc_result = match_ncc(img, template)?;

    // NCC already produces normalized values in [-1, 1]
    // Transform to [0, 1] for consistency
    let mut result = ncc_result.clone();
    result.mapv_inplace(|v| (v + 1.0) / 2.0);

    Ok(result)
}

/// Find best match location
///
/// # Arguments
///
/// * `scores` - Match scores array
/// * `method` - Matching method (to determine if lower or higher is better)
///
/// # Returns
///
/// * Best match result
pub fn find_best_match(scores: &Array2<f32>, method: MatchMethod) -> MatchResult {
    let (height, width) = scores.dim();
    let mut best_score = match method {
        MatchMethod::SumSquaredDiff | MatchMethod::NormalizedSumSquaredDiff => f32::INFINITY,
        _ => f32::NEG_INFINITY,
    };
    let mut best_x = 0;
    let mut best_y = 0;

    for y in 0..height {
        for x in 0..width {
            let score = scores[[y, x]];
            let is_better = match method {
                MatchMethod::SumSquaredDiff | MatchMethod::NormalizedSumSquaredDiff => {
                    score < best_score
                }
                _ => score > best_score,
            };

            if is_better {
                best_score = score;
                best_x = x as u32;
                best_y = y as u32;
            }
        }
    }

    MatchResult {
        x: best_x,
        y: best_y,
        score: best_score,
    }
}

/// Find multiple matches above/below threshold
///
/// # Arguments
///
/// * `scores` - Match scores array
/// * `method` - Matching method
/// * `threshold` - Score threshold
/// * `min_distance` - Minimum distance between matches
///
/// # Returns
///
/// * Vector of match results
pub fn find_matches(
    scores: &Array2<f32>,
    method: MatchMethod,
    threshold: f32,
    min_distance: u32,
) -> Vec<MatchResult> {
    let (height, width) = scores.dim();
    let mut matches = Vec::new();

    // Create a copy for non-maximum suppression
    let mut scores_copy = scores.clone();

    loop {
        let best = find_best_match(&scores_copy, method);

        // Check if match meets threshold
        let meets_threshold = match method {
            MatchMethod::SumSquaredDiff | MatchMethod::NormalizedSumSquaredDiff => {
                best.score <= threshold
            }
            _ => best.score >= threshold,
        };

        if !meets_threshold {
            break;
        }

        matches.push(best.clone());

        // Suppress nearby scores
        let y_start = best.y.saturating_sub(min_distance) as usize;
        let y_end = (best.y + min_distance + 1).min(height as u32) as usize;
        let x_start = best.x.saturating_sub(min_distance) as usize;
        let x_end = (best.x + min_distance + 1).min(width as u32) as usize;

        for y in y_start..y_end {
            for x in x_start..x_end {
                scores_copy[[y, x]] = match method {
                    MatchMethod::SumSquaredDiff | MatchMethod::NormalizedSumSquaredDiff => {
                        f32::INFINITY
                    }
                    _ => f32::NEG_INFINITY,
                };
            }
        }
    }

    matches
}

/// Draw match result on image
pub fn draw_match(
    img: &DynamicImage,
    template: &DynamicImage,
    match_result: &MatchResult,
) -> RgbImage {
    let mut result = img.to_rgb8();
    let (tmpl_width, tmpl_height) = template.dimensions();

    // Draw rectangle around match
    let color = Rgb([0, 255, 0]);
    draw_rectangle(
        &mut result,
        match_result.x,
        match_result.y,
        tmpl_width,
        tmpl_height,
        color,
    );

    result
}

/// Draw rectangle on image
fn draw_rectangle(img: &mut RgbImage, x: u32, y: u32, width: u32, height: u32, color: Rgb<u8>) {
    let (img_width, img_height) = img.dimensions();

    // Top and bottom edges
    for dx in 0..width {
        let px = x + dx;
        if px < img_width {
            if y < img_height {
                img.put_pixel(px, y, color);
            }
            if y + height - 1 < img_height {
                img.put_pixel(px, y + height - 1, color);
            }
        }
    }

    // Left and right edges
    for dy in 0..height {
        let py = y + dy;
        if py < img_height {
            if x < img_width {
                img.put_pixel(x, py, color);
            }
            if x + width - 1 < img_width {
                img.put_pixel(x + width - 1, py, color);
            }
        }
    }
}

/// Convert grayscale image to normalized array
fn image_to_array(img: &GrayImage) -> Array2<f32> {
    let (width, height) = img.dimensions();
    let mut array = Array2::zeros((height as usize, width as usize));

    for y in 0..height {
        for x in 0..width {
            array[[y as usize, x as usize]] = img.get_pixel(x, y)[0] as f32 / 255.0;
        }
    }

    array
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_template_match_basic() {
        let img = DynamicImage::new_luma8(50, 50);
        let template = DynamicImage::new_luma8(10, 10);

        let result = template_match(&img, &template, MatchMethod::CrossCorrelation);
        assert!(result.is_ok());

        let scores = result.unwrap();
        assert_eq!(scores.dim(), (41, 41));
    }

    #[test]
    fn test_template_too_large() {
        let img = DynamicImage::new_luma8(10, 10);
        let template = DynamicImage::new_luma8(20, 20);

        let result = template_match(&img, &template, MatchMethod::CrossCorrelation);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_best_match() {
        let mut scores = Array2::zeros((10, 10));
        scores[[5, 5]] = 0.9;

        let best = find_best_match(&scores, MatchMethod::CrossCorrelation);
        assert_eq!(best.x, 5);
        assert_eq!(best.y, 5);
        assert_eq!(best.score, 0.9);
    }

    #[test]
    fn test_find_multiple_matches() {
        let mut scores = Array2::zeros((20, 20));
        scores[[5, 5]] = 0.9;
        scores[[15, 15]] = 0.8;
        scores[[5, 15]] = 0.7;

        let matches = find_matches(&scores, MatchMethod::CrossCorrelation, 0.6, 3);
        assert_eq!(matches.len(), 3);
    }
}
