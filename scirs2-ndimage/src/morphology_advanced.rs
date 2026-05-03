//! Advanced mathematical morphology transforms
//!
//! This module provides extended morphological operations beyond the basics:
//! - Rolling-ball background subtraction
//! - White and black top-hat transforms
//! - Morphological gradient
//! - Toggle contrast mapping
//! - Hit-or-miss transform

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array2;
use std::collections::VecDeque;

// ─── Structuring element helpers ─────────────────────────────────────────────

/// A flat 2-D structuring element (SE), represented as a boolean mask.
///
/// The anchor is the centre pixel `(rows/2, cols/2)`.
#[derive(Debug, Clone)]
pub struct StructuringElement {
    /// 2-D boolean mask (true = SE member)
    pub mask: Array2<bool>,
}

impl StructuringElement {
    /// Create a disk (filled circle) structuring element with given `radius`.
    pub fn disk(radius: usize) -> Self {
        let side = 2 * radius + 1;
        let c = radius as f64;
        let mask = Array2::from_shape_fn((side, side), |(r, col)| {
            let dr = r as f64 - c;
            let dc = col as f64 - c;
            dr * dr + dc * dc <= (radius as f64).powi(2) + 1e-9
        });
        StructuringElement { mask }
    }

    /// Create a square (all-ones) structuring element with side length `2*half+1`.
    pub fn square(half: usize) -> Self {
        let side = 2 * half + 1;
        let mask = Array2::from_elem((side, side), true);
        StructuringElement { mask }
    }

    /// Create a cross (plus) structuring element with given `radius`.
    pub fn cross(radius: usize) -> Self {
        let side = 2 * radius + 1;
        let cr = radius;
        let mask = Array2::from_shape_fn((side, side), |(r, c)| r == cr || c == cr);
        StructuringElement { mask }
    }

    /// Number of rows in the SE mask.
    pub fn rows(&self) -> usize {
        self.mask.nrows()
    }

    /// Number of columns in the SE mask.
    pub fn cols(&self) -> usize {
        self.mask.ncols()
    }

    /// Row index of the anchor (centre).
    pub fn anchor_row(&self) -> usize {
        self.rows() / 2
    }

    /// Column index of the anchor (centre).
    pub fn anchor_col(&self) -> usize {
        self.cols() / 2
    }
}

// ─── Core erosion / dilation helpers ────────────────────────────────────────

/// Grayscale erosion: replace each pixel by the minimum value in its SE neighbourhood.
pub fn erode(image: &Array2<f64>, se: &StructuringElement) -> NdimageResult<Array2<f64>> {
    let rows = image.nrows();
    let cols = image.ncols();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    let ar = se.anchor_row() as isize;
    let ac = se.anchor_col() as isize;
    let mut out = Array2::<f64>::from_elem((rows, cols), f64::INFINITY);
    for r in 0..rows {
        for c in 0..cols {
            let mut min_val = f64::INFINITY;
            for sr in 0..se.rows() {
                for sc in 0..se.cols() {
                    if !se.mask[[sr, sc]] {
                        continue;
                    }
                    let nr = r as isize + sr as isize - ar;
                    let nc = c as isize + sc as isize - ac;
                    if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                        // Reflect border: clamp
                        let nr = nr.max(0).min(rows as isize - 1) as usize;
                        let nc = nc.max(0).min(cols as isize - 1) as usize;
                        let v = image[[nr, nc]];
                        if v < min_val {
                            min_val = v;
                        }
                    } else {
                        let v = image[[nr as usize, nc as usize]];
                        if v < min_val {
                            min_val = v;
                        }
                    }
                }
            }
            out[[r, c]] = min_val;
        }
    }
    Ok(out)
}

/// Grayscale dilation: replace each pixel by the maximum value in its SE neighbourhood.
pub fn dilate(image: &Array2<f64>, se: &StructuringElement) -> NdimageResult<Array2<f64>> {
    let rows = image.nrows();
    let cols = image.ncols();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    let ar = se.anchor_row() as isize;
    let ac = se.anchor_col() as isize;
    let mut out = Array2::<f64>::from_elem((rows, cols), f64::NEG_INFINITY);
    for r in 0..rows {
        for c in 0..cols {
            let mut max_val = f64::NEG_INFINITY;
            for sr in 0..se.rows() {
                for sc in 0..se.cols() {
                    if !se.mask[[sr, sc]] {
                        continue;
                    }
                    let nr = r as isize + sr as isize - ar;
                    let nc = c as isize + sc as isize - ac;
                    let nr = nr.max(0).min(rows as isize - 1) as usize;
                    let nc = nc.max(0).min(cols as isize - 1) as usize;
                    let v = image[[nr, nc]];
                    if v > max_val {
                        max_val = v;
                    }
                }
            }
            out[[r, c]] = max_val;
        }
    }
    Ok(out)
}

// ─── Rolling-Ball Background Subtraction ────────────────────────────────────

/// Rolling-ball background estimation for fluorescence microscopy images.
///
/// Estimates a smoothly varying background by rolling a spherical ball of
/// the given `radius` under the image intensity surface. The background is
/// the minimum of the image in a disk neighbourhood, additionally
/// flattened by an opening with a disk of the same radius.
///
/// # Arguments
/// * `image`  – 2-D grayscale image (should be non-negative)
/// * `radius` – ball radius in pixels
///
/// # Returns
/// Background image of same shape; subtract from original to obtain
/// foreground objects.
pub fn rolling_ball_background(image: &Array2<f64>, radius: f64) -> NdimageResult<Array2<f64>> {
    if radius <= 0.0 {
        return Err(NdimageError::InvalidInput("radius must be positive".into()));
    }
    let rows = image.nrows();
    let cols = image.ncols();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }

    let r_int = radius.ceil() as usize;
    let se = StructuringElement::disk(r_int);

    // Opening = dilation of erosion
    let eroded = erode(image, &se)?;
    let background = dilate(&eroded, &se)?;
    Ok(background)
}

// ─── Top-Hat Transforms ─────────────────────────────────────────────────────

/// White top-hat transform: image minus its morphological opening.
///
/// Extracts bright structures smaller than the SE (spots, thin lines).
///
/// # Arguments
/// * `image` – 2-D grayscale image
/// * `se`    – flat structuring element
pub fn top_hat(image: &Array2<f64>, se: &StructuringElement) -> NdimageResult<Array2<f64>> {
    // Opening = dilation(erosion(image))
    let eroded = erode(image, se)?;
    let opened = dilate(&eroded, se)?;
    let rows = image.nrows();
    let cols = image.ncols();
    let mut result = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            result[[r, c]] = (image[[r, c]] - opened[[r, c]]).max(0.0);
        }
    }
    Ok(result)
}

/// Black top-hat transform (bottom-hat): morphological closing minus image.
///
/// Extracts dark structures smaller than the SE (holes, dark spots).
///
/// # Arguments
/// * `image` – 2-D grayscale image
/// * `se`    – flat structuring element
pub fn black_hat(image: &Array2<f64>, se: &StructuringElement) -> NdimageResult<Array2<f64>> {
    // Closing = erosion(dilation(image))
    let dilated = dilate(image, se)?;
    let closed = erode(&dilated, se)?;
    let rows = image.nrows();
    let cols = image.ncols();
    let mut result = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            result[[r, c]] = (closed[[r, c]] - image[[r, c]]).max(0.0);
        }
    }
    Ok(result)
}

// ─── Morphological Gradient ──────────────────────────────────────────────────

/// Morphological gradient: dilation minus erosion.
///
/// Highlights object boundaries and contours in grayscale images.
///
/// # Arguments
/// * `image` – 2-D grayscale image
/// * `se`    – flat structuring element
pub fn morphological_gradient(
    image: &Array2<f64>,
    se: &StructuringElement,
) -> NdimageResult<Array2<f64>> {
    let dil = dilate(image, se)?;
    let ero = erode(image, se)?;
    let rows = image.nrows();
    let cols = image.ncols();
    let mut result = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            result[[r, c]] = dil[[r, c]] - ero[[r, c]];
        }
    }
    Ok(result)
}

// ─── Toggle Contrast ─────────────────────────────────────────────────────────

/// Toggle mapping (contrast enhancement via morphological toggle).
///
/// For each pixel x, assigns `dilation(x)` if the pixel is closer to its
/// dilated value, else `erosion(x)`. Sharpens edges while preserving
/// extremal regions.
///
/// # Arguments
/// * `image`    – 2-D grayscale image
/// * `se_inner` – inner (smaller) structuring element for erosion
/// * `se_outer` – outer (larger) structuring element for dilation
pub fn toggle_contrast(
    image: &Array2<f64>,
    se_inner: &StructuringElement,
    se_outer: &StructuringElement,
) -> NdimageResult<Array2<f64>> {
    let dil = dilate(image, se_outer)?;
    let ero = erode(image, se_inner)?;
    let rows = image.nrows();
    let cols = image.ncols();
    let mut result = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let v = image[[r, c]];
            let d = dil[[r, c]];
            let e = ero[[r, c]];
            result[[r, c]] = if (v - d).abs() <= (v - e).abs() { d } else { e };
        }
    }
    Ok(result)
}

// ─── Hit-or-Miss Transform ───────────────────────────────────────────────────

/// Hit-or-miss transform for template matching in grayscale images.
///
/// Detects pixels where the foreground SE (`fg_se`) fits within the image
/// AND the background SE (`bg_se`) fits within the complement (regions
/// below the image values).
///
/// For binary images, this reduces to the classical hit-or-miss transform.
/// For grayscale images, a pixel at position p hits if:
///   `erosion(image, fg_se)[p] > dilation(image, bg_se)[p]`
///
/// # Arguments
/// * `image` – 2-D grayscale image (values in [0, 1] typical)
/// * `fg_se` – foreground structuring element (fit to bright regions)
/// * `bg_se` – background structuring element (fit to dark regions)
///
/// # Returns
/// Boolean hit mask of same shape.
pub fn hit_or_miss(
    image: &Array2<f64>,
    fg_se: &StructuringElement,
    bg_se: &StructuringElement,
) -> NdimageResult<Array2<bool>> {
    let rows = image.nrows();
    let cols = image.ncols();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }

    let ero_fg = erode(image, fg_se)?;
    let dil_bg = dilate(image, bg_se)?;

    let mut result = Array2::<bool>::from_elem((rows, cols), false);
    for r in 0..rows {
        for c in 0..cols {
            result[[r, c]] = ero_fg[[r, c]] > dil_bg[[r, c]];
        }
    }
    Ok(result)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Create a simple test image with a bright spot in the centre.
    fn bright_spot_image(rows: usize, cols: usize) -> Array2<f64> {
        let cr = rows / 2;
        let cc = cols / 2;
        Array2::from_shape_fn((rows, cols), |(r, c)| {
            let dr = r as f64 - cr as f64;
            let dc = c as f64 - cc as f64;
            if dr * dr + dc * dc < 4.0 {
                1.0
            } else {
                0.1
            }
        })
    }

    fn step_image(rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(_, c)| if c < cols / 2 { 0.0 } else { 1.0 })
    }

    // ── StructuringElement tests ─────────────────────────────────────────────

    #[test]
    fn test_disk_se_centre_true() {
        let se = StructuringElement::disk(2);
        let r = se.anchor_row();
        let c = se.anchor_col();
        assert!(se.mask[[r, c]]);
    }

    #[test]
    fn test_square_se_all_true() {
        let se = StructuringElement::square(1);
        assert_eq!(se.rows(), 3);
        assert_eq!(se.cols(), 3);
        assert!(se.mask.iter().all(|&v| v));
    }

    // ── Rolling-ball tests ───────────────────────────────────────────────────

    #[test]
    fn test_rolling_ball_background_shape() {
        let img = bright_spot_image(16, 16);
        let bg = rolling_ball_background(&img, 3.0).expect("rolling ball failed");
        assert_eq!(bg.shape(), img.shape());
    }

    #[test]
    fn test_rolling_ball_background_lte_image() {
        // Background must be <= original image (opening property)
        let img = bright_spot_image(12, 12);
        let bg = rolling_ball_background(&img, 2.0).expect("rolling ball");
        for r in 0..img.nrows() {
            for c in 0..img.ncols() {
                assert!(
                    bg[[r, c]] <= img[[r, c]] + 1e-9,
                    "bg > img at ({r},{c}): {} > {}",
                    bg[[r, c]],
                    img[[r, c]]
                );
            }
        }
    }

    #[test]
    fn test_rolling_ball_invalid_radius() {
        let img = Array2::<f64>::zeros((4, 4));
        assert!(rolling_ball_background(&img, 0.0).is_err());
        assert!(rolling_ball_background(&img, -1.0).is_err());
    }

    // ── Top-hat tests ────────────────────────────────────────────────────────

    #[test]
    fn test_top_hat_bright_spot_detected() {
        let img = bright_spot_image(16, 16);
        let se = StructuringElement::disk(3);
        let th = top_hat(&img, &se).expect("top hat failed");
        assert_eq!(th.shape(), img.shape());
        // Bright spot should survive top-hat
        let cr = img.nrows() / 2;
        let cc = img.ncols() / 2;
        assert!(th[[cr, cc]] > 0.0, "Centre should be > 0 after top-hat");
    }

    #[test]
    fn test_top_hat_uniform_image_zero() {
        let img = Array2::<f64>::from_elem((8, 8), 0.5);
        let se = StructuringElement::square(1);
        let th = top_hat(&img, &se).expect("top hat uniform");
        // Uniform image: opening = image, so top-hat = 0
        assert!(th.iter().all(|&v| v.abs() < 1e-10));
    }

    // ── Black-hat tests ──────────────────────────────────────────────────────

    #[test]
    fn test_black_hat_dark_hole() {
        // Image with a dark hole in the centre
        let img = Array2::from_shape_fn((16, 16), |(r, c)| {
            let cr = 8usize;
            let cc = 8usize;
            let dr = r as f64 - cr as f64;
            let dc = c as f64 - cc as f64;
            if dr * dr + dc * dc < 4.0 {
                0.0
            } else {
                0.9
            }
        });
        let se = StructuringElement::disk(3);
        let bh = black_hat(&img, &se).expect("black hat failed");
        assert_eq!(bh.shape(), img.shape());
        let cr = img.nrows() / 2;
        let cc = img.ncols() / 2;
        assert!(
            bh[[cr, cc]] > 0.0,
            "Dark hole should be detected by black-hat"
        );
    }

    #[test]
    fn test_black_hat_uniform_zero() {
        let img = Array2::<f64>::from_elem((8, 8), 0.5);
        let se = StructuringElement::square(1);
        let bh = black_hat(&img, &se).expect("black hat uniform");
        assert!(bh.iter().all(|&v| v.abs() < 1e-10));
    }

    // ── Morphological gradient tests ─────────────────────────────────────────

    #[test]
    fn test_morphological_gradient_step_edge() {
        let img = step_image(8, 8);
        let se = StructuringElement::square(1);
        let grad = morphological_gradient(&img, &se).expect("morphological gradient failed");
        assert_eq!(grad.shape(), img.shape());
        // Gradient should be non-zero near the step
        let col = 4; // edge column
        assert!(grad[[4, col]] > 0.0 || grad[[4, col - 1]] > 0.0);
    }

    #[test]
    fn test_morphological_gradient_uniform_zero() {
        let img = Array2::<f64>::from_elem((8, 8), 0.5);
        let se = StructuringElement::square(1);
        let grad = morphological_gradient(&img, &se).expect("gradient uniform");
        // Uniform: dilation == erosion == image, gradient == 0
        assert!(grad.iter().all(|&v| v.abs() < 1e-10));
    }

    // ── Toggle contrast tests ────────────────────────────────────────────────

    #[test]
    fn test_toggle_contrast_shape() {
        let img = bright_spot_image(12, 12);
        let se_inner = StructuringElement::cross(1);
        let se_outer = StructuringElement::disk(2);
        let tc = toggle_contrast(&img, &se_inner, &se_outer).expect("toggle contrast failed");
        assert_eq!(tc.shape(), img.shape());
    }

    #[test]
    fn test_toggle_contrast_extreme_values() {
        // Constant image: toggle should leave it unchanged or produce valid values
        let img = Array2::<f64>::from_elem((6, 6), 0.5);
        let se = StructuringElement::square(1);
        let tc = toggle_contrast(&img, &se, &se).expect("toggle contrast const");
        assert!(tc.iter().all(|&v| v.is_finite()));
    }

    // ── Hit-or-miss tests ────────────────────────────────────────────────────

    #[test]
    fn test_hit_or_miss_bright_peak() {
        // Bright isolated pixel surrounded by dark
        let mut img = Array2::<f64>::from_elem((8, 8), 0.0);
        img[[4, 4]] = 1.0;
        let fg_se = StructuringElement::disk(0); // 1×1 foreground SE
        let bg_se = StructuringElement::disk(0); // 1×1 background SE (will not overlap with same anchor)
                                                 // Use a cross: centre is fg, arms are bg
        let fg_se = StructuringElement {
            mask: {
                let mut m = Array2::from_elem((3, 3), false);
                m[[1, 1]] = true;
                m
            },
        };
        let bg_se = StructuringElement {
            mask: {
                let mut m = Array2::from_elem((3, 3), false);
                m[[0, 1]] = true;
                m[[2, 1]] = true;
                m[[1, 0]] = true;
                m[[1, 2]] = true;
                m
            },
        };
        let hom = hit_or_miss(&img, &fg_se, &bg_se).expect("hit or miss failed");
        assert_eq!(hom.shape(), img.shape());
        // The bright peak at (4,4) surrounded by 0s should hit
        assert!(hom[[4, 4]], "Bright isolated pixel should produce a hit");
    }

    #[test]
    fn test_hit_or_miss_no_hit_uniform() {
        // Uniform image: erosion == dilation, so no hits
        let img = Array2::<f64>::from_elem((8, 8), 0.5);
        let se = StructuringElement::square(1);
        let hom = hit_or_miss(&img, &se, &se).expect("hit or miss uniform");
        // On a uniform image, erosion == dilation, so erosion(fg) <= dilation(bg): no hits
        assert!(!hom.iter().any(|&v| v), "Uniform image should have no hits");
    }
}
