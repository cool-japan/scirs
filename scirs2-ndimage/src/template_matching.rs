//! Template Matching and Sliding Window Detection
//!
//! This module provides:
//! - Template matching with multiple similarity metrics (SSD, NCC, coefficient correlation)
//! - Non-maximum suppression to find discrete match locations
//! - Multi-scale (image pyramid) template matching
//!
//! # References
//! - Lewis, J.P. (1995). "Fast Template Matching." Vision Interface.
//! - Briechle, K. & Hanebeck, U.D. (2001). "Template Matching using Fast Normalized
//!   Cross Correlation." Proc. SPIE 4387.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{s, Array2, Array3};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

/// Template matching similarity measure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMethod {
    /// Sum of squared differences (lower is better match)
    SumSquaredDiff,
    /// Normalized sum of squared differences in [0, 1] (lower is better)
    NormalizedSumSquaredDiff,
    /// Normalized cross-correlation in [-1, 1] (higher is better)
    NormalizedCrossCorrelation,
    /// Zero-mean normalized cross-correlation in [-1, 1] (higher is better)
    CoeffCorrelation,
}

// ---------------------------------------------------------------------------
// Core template matching
// ---------------------------------------------------------------------------

/// Compute a template-match response map for the given similarity measure.
///
/// The output array has shape `(image_rows - template_rows + 1,
/// image_cols - template_cols + 1)` — one value per valid placement
/// of the template inside the image.
///
/// For `SumSquaredDiff` / `NormalizedSumSquaredDiff` a **lower** value
/// indicates a better match.  For `NormalizedCrossCorrelation` /
/// `CoeffCorrelation` a **higher** value (closer to 1) is better.
///
/// # Errors
/// Returns `NdimageError::InvalidInput` when the template is larger than
/// the image in either dimension.
pub fn template_match(
    image: &Array2<f64>,
    template: &Array2<f64>,
    method: MatchMethod,
) -> NdimageResult<Array2<f64>> {
    let (ih, iw) = image.dim();
    let (th, tw) = template.dim();

    if th == 0 || tw == 0 {
        return Err(NdimageError::InvalidInput(
            "Template must not be empty".into(),
        ));
    }
    if th > ih || tw > iw {
        return Err(NdimageError::InvalidInput(
            "Template must not be larger than the image".into(),
        ));
    }

    match method {
        MatchMethod::SumSquaredDiff => ssd_map(image, template, false),
        MatchMethod::NormalizedSumSquaredDiff => ssd_map(image, template, true),
        MatchMethod::NormalizedCrossCorrelation => normalized_cross_correlation(image, template),
        MatchMethod::CoeffCorrelation => coeff_correlation(image, template),
    }
}

// ---------------------------------------------------------------------------
// SSD response map
// ---------------------------------------------------------------------------

fn ssd_map(
    image: &Array2<f64>,
    template: &Array2<f64>,
    normalize: bool,
) -> NdimageResult<Array2<f64>> {
    let (ih, iw) = image.dim();
    let (th, tw) = template.dim();
    let out_h = ih - th + 1;
    let out_w = iw - tw + 1;

    // Template sum-of-squares for normalization
    let template_ss: f64 = template.iter().map(|&v| v * v).sum();

    let mut result = Array2::zeros((out_h, out_w));

    for r in 0..out_h {
        for c in 0..out_w {
            let patch = image.slice(s![r..r + th, c..c + tw]);
            let mut ssd = 0.0;
            for (iv, tv) in patch.iter().zip(template.iter()) {
                let d = iv - tv;
                ssd += d * d;
            }

            if normalize {
                // Normalized SSD: SSD / (||patch|| * ||template||)
                let patch_ss: f64 = patch.iter().map(|&v| v * v).sum();
                let denom = (patch_ss * template_ss).sqrt();
                result[[r, c]] = if denom > 1e-12 { ssd / denom } else { 0.0 };
            } else {
                result[[r, c]] = ssd;
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Normalized cross-correlation
// ---------------------------------------------------------------------------

/// Compute the normalized cross-correlation (NCC) response map.
///
/// Each output pixel is
/// ```text
///   NCC(r,c) = sum(patch * template) / (||patch|| * ||template||)
/// ```
/// Values lie in [-1, 1].  A value of 1 means perfect correlation.
///
/// # Errors
/// Returns `NdimageError::InvalidInput` when the template is larger than
/// the image in either dimension.
pub fn normalized_cross_correlation(
    image: &Array2<f64>,
    template: &Array2<f64>,
) -> NdimageResult<Array2<f64>> {
    let (ih, iw) = image.dim();
    let (th, tw) = template.dim();

    if th == 0 || tw == 0 {
        return Err(NdimageError::InvalidInput(
            "Template must not be empty".into(),
        ));
    }
    if th > ih || tw > iw {
        return Err(NdimageError::InvalidInput(
            "Template must not be larger than the image".into(),
        ));
    }

    let out_h = ih - th + 1;
    let out_w = iw - tw + 1;

    let template_norm: f64 = template.iter().map(|&v| v * v).sum::<f64>().sqrt();

    let mut result = Array2::zeros((out_h, out_w));

    for r in 0..out_h {
        for c in 0..out_w {
            let patch = image.slice(s![r..r + th, c..c + tw]);
            let cross: f64 = patch.iter().zip(template.iter()).map(|(a, b)| a * b).sum();
            let patch_norm: f64 = patch.iter().map(|&v| v * v).sum::<f64>().sqrt();
            let denom = patch_norm * template_norm;
            result[[r, c]] = if denom > 1e-12 { cross / denom } else { 0.0 };
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Zero-mean normalized cross-correlation (coefficient correlation)
// ---------------------------------------------------------------------------

fn coeff_correlation(image: &Array2<f64>, template: &Array2<f64>) -> NdimageResult<Array2<f64>> {
    let (ih, iw) = image.dim();
    let (th, tw) = template.dim();
    let out_h = ih - th + 1;
    let out_w = iw - tw + 1;
    let n = (th * tw) as f64;

    // Zero-mean template
    let t_mean: f64 = template.iter().sum::<f64>() / n;
    let t_centered: Vec<f64> = template.iter().map(|&v| v - t_mean).collect();
    let t_std: f64 = t_centered.iter().map(|&v| v * v).sum::<f64>().sqrt();

    let mut result = Array2::zeros((out_h, out_w));

    for r in 0..out_h {
        for c in 0..out_w {
            let patch = image.slice(s![r..r + th, c..c + tw]);
            let p_mean: f64 = patch.iter().sum::<f64>() / n;
            let cross: f64 = patch
                .iter()
                .zip(t_centered.iter())
                .map(|(a, b)| (a - p_mean) * b)
                .sum();
            let p_std: f64 = patch
                .iter()
                .map(|&v| (v - p_mean).powi(2))
                .sum::<f64>()
                .sqrt();
            let denom = p_std * t_std;
            result[[r, c]] = if denom > 1e-12 { cross / denom } else { 0.0 };
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Peak / match extraction with non-maximum suppression
// ---------------------------------------------------------------------------

/// Extract discrete match locations from a response map.
///
/// For `NormalizedCrossCorrelation` / `CoeffCorrelation` maps, peaks **above**
/// `threshold` are returned (score closer to 1 = better).
/// For `SumSquaredDiff` maps, use negated or inverted scores before calling
/// this function, or supply negative scores.
///
/// # Parameters
/// - `correlation_map` – 2-D response map produced by `template_match` (or
///   `normalized_cross_correlation`).
/// - `threshold` – minimum score to be considered a match.
/// - `min_distance` – minimum pixel distance between two accepted peaks
///   (non-maximum suppression radius).
///
/// # Returns
/// A sorted (descending by score) list of `(row, col, score)` tuples.
pub fn find_matches(
    correlation_map: &Array2<f64>,
    threshold: f64,
    min_distance: usize,
) -> NdimageResult<Vec<(usize, usize, f64)>> {
    let (rows, cols) = correlation_map.dim();
    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }

    // Collect all above-threshold positions
    let mut candidates: Vec<(usize, usize, f64)> = correlation_map
        .indexed_iter()
        .filter_map(|((r, c), &score)| {
            if score >= threshold {
                Some((r, c, score))
            } else {
                None
            }
        })
        .collect();

    // Sort descending by score
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Non-maximum suppression: greedily accept peaks that are at least
    // `min_distance` pixels away from all already-accepted peaks.
    let mut accepted: Vec<(usize, usize, f64)> = Vec::new();
    let min_dist_sq = (min_distance as f64) * (min_distance as f64);

    'outer: for (r, c, score) in candidates {
        for &(ar, ac, _) in &accepted {
            let dr = r as f64 - ar as f64;
            let dc = c as f64 - ac as f64;
            if dr * dr + dc * dc < min_dist_sq {
                continue 'outer;
            }
        }
        accepted.push((r, c, score));
    }

    Ok(accepted)
}

// ---------------------------------------------------------------------------
// Multi-scale (image pyramid) template matching
// ---------------------------------------------------------------------------

/// Down-sample a 2-D image by factor 2 using simple 2×2 average pooling.
fn downsample_2x(image: &Array2<f64>) -> Array2<f64> {
    let (h, w) = image.dim();
    let oh = h / 2;
    let ow = w / 2;
    if oh == 0 || ow == 0 {
        return image.clone();
    }
    let mut out = Array2::zeros((oh, ow));
    for r in 0..oh {
        for c in 0..ow {
            out[[r, c]] = 0.25
                * (image[[2 * r, 2 * c]]
                    + image[[2 * r, 2 * c + 1]]
                    + image[[2 * r + 1, 2 * c]]
                    + image[[2 * r + 1, 2 * c + 1]]);
        }
    }
    out
}

/// Multi-scale template matching using a Gaussian image pyramid.
///
/// Builds `n_scales` octave-spaced scales of the image (each half the
/// previous dimensions) and runs normalized cross-correlation at each
/// scale.  Detected peaks are mapped back to the original-image
/// coordinate space and deduplicated with non-maximum suppression.
///
/// # Parameters
/// - `image`    – original grayscale image.
/// - `template` – template to search for.
/// - `n_scales` – number of pyramid levels (≥ 1).
///
/// # Returns
/// List of `(row, col, score, scale)` tuples sorted descending by score,
/// where `scale` is the zoom factor at which the match was found
/// (1.0 = original size, 0.5 = half size, …).
///
/// # Errors
/// Returns `NdimageError::InvalidInput` for degenerate inputs.
pub fn pyramid_template_match(
    image: &Array2<f64>,
    template: &Array2<f64>,
    n_scales: usize,
) -> NdimageResult<Vec<(usize, usize, f64, f64)>> {
    if n_scales == 0 {
        return Err(NdimageError::InvalidInput(
            "n_scales must be at least 1".into(),
        ));
    }
    if template.dim().0 == 0 || template.dim().1 == 0 {
        return Err(NdimageError::InvalidInput(
            "Template must not be empty".into(),
        ));
    }

    let (th, tw) = template.dim();
    let mut results: Vec<(usize, usize, f64, f64)> = Vec::new();

    let mut current_image = image.clone();
    let mut current_template = template.clone();
    let mut scale = 1.0_f64;

    for _lvl in 0..n_scales {
        let (ih, iw) = current_image.dim();
        let (cth, ctw) = current_template.dim();

        // Stop if the template no longer fits the image
        if cth == 0 || ctw == 0 || cth > ih || ctw > iw {
            break;
        }

        let ncc = normalized_cross_correlation(&current_image, &current_template)?;

        // Threshold: accept matches with NCC ≥ 0.5 (reasonable default)
        let threshold = 0.5;
        let min_dist = (th.max(tw) / 2).max(1);
        let local_matches = find_matches(&ncc, threshold, min_dist)?;

        for (r, c, score) in local_matches {
            // Map coordinates back to original image space
            let orig_r = (r as f64 / scale).round() as usize;
            let orig_c = (c as f64 / scale).round() as usize;
            results.push((orig_r, orig_c, score, scale));
        }

        // Build next pyramid level
        current_image = downsample_2x(&current_image);
        current_template = downsample_2x(&current_template);
        scale *= 0.5;
    }

    // Sort by score descending
    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Global NMS across all scales in original coordinates
    let nms_dist: usize = (th.max(tw) / 2).max(1);
    let min_dist_sq = (nms_dist as f64).powi(2);
    let mut accepted: Vec<(usize, usize, f64, f64)> = Vec::new();

    'outer: for (r, c, score, s) in results {
        for &(ar, ac, _, _) in &accepted {
            let dr = r as f64 - ar as f64;
            let dc = c as f64 - ac as f64;
            if dr * dr + dc * dc < min_dist_sq {
                continue 'outer;
            }
        }
        accepted.push((r, c, score, s));
    }

    Ok(accepted)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn checkerboard_image(rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_fn(
            (rows, cols),
            |(r, c)| {
                if (r + c) % 2 == 0 {
                    1.0
                } else {
                    0.0
                }
            },
        )
    }

    #[test]
    fn test_ssd_perfect_match() {
        let image: Array2<f64> = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            ],
        )
        .expect("shape ok");

        let template: Array2<f64> =
            Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).expect("shape ok");

        let map = template_match(&image, &template, MatchMethod::SumSquaredDiff).expect("ssd ok");
        // Perfect match at (0,0): SSD = 0
        assert!(
            map[[0, 0]] < 1e-12,
            "Expected zero SSD at perfect-match location"
        );
    }

    #[test]
    fn test_ncc_perfect_match() {
        let img = checkerboard_image(6, 6);
        let tpl = img.slice(s![1..3, 1..3]).to_owned();
        let ncc = normalized_cross_correlation(&img, &tpl).expect("ncc ok");
        // Find the patch at (1,1) — should give NCC ~ 1
        let score = ncc[[1, 1]];
        assert!(
            score > 0.99,
            "NCC at matching position should be ~1, got {score}"
        );
    }

    #[test]
    fn test_find_matches_basic() {
        let mut map: Array2<f64> = Array2::zeros((10, 10));
        map[[2, 3]] = 0.9;
        map[[7, 8]] = 0.8;
        map[[2, 4]] = 0.85; // close to (2,3); should be suppressed

        let matches = find_matches(&map, 0.7, 3).expect("matches ok");
        assert!(!matches.is_empty());
        // First match should be the highest-score one
        assert_eq!(matches[0], (2, 3, 0.9));
    }

    #[test]
    fn test_pyramid_match_runs() {
        let image = checkerboard_image(32, 32);
        let template: Array2<f64> = image.slice(s![4..8, 4..8]).to_owned();
        let results = pyramid_template_match(&image, &template, 3).expect("pyramid ok");
        // Should produce at least one result
        assert!(!results.is_empty());
    }

    #[test]
    fn test_template_larger_than_image_errors() {
        let small: Array2<f64> = Array2::zeros((3, 3));
        let large: Array2<f64> = Array2::zeros((5, 5));
        let err = template_match(&small, &large, MatchMethod::SumSquaredDiff);
        assert!(err.is_err());
    }
}
