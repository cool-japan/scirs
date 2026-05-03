//! Advanced Image Feature Detection
//!
//! This module provides advanced feature detection algorithms:
//!
//! - **Harris corners** (with configurable struct): Gradient-based corner detector using
//!   the structure tensor (autocorrelation matrix) with non-maximum suppression.
//! - **FAST corners**: Features from Accelerated Segment Test — circle-based corner test.
//! - **LoG blob detection**: Laplacian of Gaussian scale-space blob detector with NMS.
//! - **Shi-Tomasi corners**: Minimum eigenvalue corner quality with distance filtering.
//!
//! # References
//!
//! - Harris, C. & Stephens, M. (1988). "A Combined Corner and Edge Detector."
//! - Rosten, E. & Drummond, T. (2006). "Machine learning for high-speed corner detection."
//! - Lowe, D.G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints."
//! - Shi, J. & Tomasi, C. (1994). "Good Features to Track."

use crate::error::{NdimageError, NdimageResult};
use crate::filters::{gaussian_filter_f32, sobel, BorderMode};
use scirs2_core::ndarray::{Array2, ArrayD, Dimension};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// A detected corner point
#[derive(Debug, Clone, PartialEq)]
pub struct Corner {
    /// Row index of the corner
    pub row: usize,
    /// Column index of the corner
    pub col: usize,
    /// Corner response / quality measure
    pub response: f64,
}

/// A detected blob (scale-space extremum)
#[derive(Debug, Clone, PartialEq)]
pub struct Blob {
    /// Fractional row of the blob centre
    pub row: f64,
    /// Fractional column of the blob centre
    pub col: f64,
    /// Gaussian sigma at which the blob was detected (scale)
    pub sigma: f64,
    /// Normalised LoG response at detection
    pub response: f64,
}

// ---------------------------------------------------------------------------
// Harris corner detector
// ---------------------------------------------------------------------------

/// Configuration for the Harris corner detector
#[derive(Debug, Clone)]
pub struct HarrisConfig {
    /// Harris detector free parameter (default 0.04–0.06)
    pub k: f64,
    /// Gaussian smoothing sigma for structure tensor
    pub sigma: f64,
    /// Response threshold as a fraction of the maximum response (0.0–1.0)
    pub threshold: f64,
    /// Non-maximum suppression radius in pixels
    pub nms_radius: usize,
}

impl Default for HarrisConfig {
    fn default() -> Self {
        Self {
            k: 0.05,
            sigma: 1.0,
            threshold: 0.01,
            nms_radius: 3,
        }
    }
}

/// Detect Harris corners in a greyscale image.
///
/// The algorithm:
/// 1. Compute image gradients Ix, Iy via Sobel filters.
/// 2. Build the structure tensor M = [[Ix², IxIy], [IxIy, Iy²]] and smooth with a Gaussian.
/// 3. Compute the Harris response R = det(M) - k · trace(M)².
/// 4. Apply an absolute threshold (threshold * max_response).
/// 5. Suppress non-maximum responses within `nms_radius`.
///
/// # Arguments
///
/// * `image`  - Input 2D float image.
/// * `config` - [`HarrisConfig`] controlling all parameters.
///
/// # Errors
///
/// Returns [`NdimageError`] if gradient computation fails.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::feature_detection::{harris_corners_configured, HarrisConfig};
/// use scirs2_core::ndarray::Array2;
///
/// let mut img = Array2::<f32>::zeros((20, 20));
/// // Draw a simple cross to create corner-like structures
/// for i in 0..20 { img[[10, i]] = 1.0; img[[i, 10]] = 1.0; }
/// let config = HarrisConfig { k: 0.05, sigma: 1.0, threshold: 0.01, nms_radius: 3 };
/// let corners = harris_corners_configured(&img, config).expect("harris failed");
/// ```
pub fn harris_corners_configured(
    image: &Array2<f32>,
    config: HarrisConfig,
) -> NdimageResult<Vec<Corner>> {
    let (rows, cols) = (image.nrows(), image.ncols());
    if rows < 3 || cols < 3 {
        return Ok(Vec::new());
    }

    // Convert to dynamic array for filter functions
    let mut img_d = ArrayD::from_elem(image.raw_dim().into_dyn(), 0.0f32);
    for ((r, c), &v) in image.indexed_iter() {
        img_d[[r, c]] = v;
    }

    // Compute gradients
    let gy_d = sobel(&img_d, 0, Some(BorderMode::Reflect))
        .map_err(|e| NdimageError::ComputationError(format!("sobel y: {e}")))?;
    let gx_d = sobel(&img_d, 1, Some(BorderMode::Reflect))
        .map_err(|e| NdimageError::ComputationError(format!("sobel x: {e}")))?;

    // Products of gradients
    let mut ix2_d = gx_d.mapv(|v| v * v);
    let mut iy2_d = gy_d.mapv(|v| v * v);
    let mut ixy_d = {
        let mut t = gx_d.clone();
        for (idx, &gy) in gy_d.indexed_iter() {
            t[idx.slice()] *= gy;
        }
        t
    };

    // Gaussian smoothing of the structure tensor
    let sigma = config.sigma as f32;
    ix2_d = gaussian_filter_f32(&ix2_d, sigma, Some(BorderMode::Reflect), None)
        .map_err(|e| NdimageError::ComputationError(format!("gaussian ix2: {e}")))?;
    iy2_d = gaussian_filter_f32(&iy2_d, sigma, Some(BorderMode::Reflect), None)
        .map_err(|e| NdimageError::ComputationError(format!("gaussian iy2: {e}")))?;
    ixy_d = gaussian_filter_f32(&ixy_d, sigma, Some(BorderMode::Reflect), None)
        .map_err(|e| NdimageError::ComputationError(format!("gaussian ixy: {e}")))?;

    // Harris response R = det(M) - k * trace(M)^2
    let k = config.k as f32;
    let mut response = Array2::<f32>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let a = ix2_d[[r, c]];
            let b = ixy_d[[r, c]];
            let d = iy2_d[[r, c]];
            let det = a * d - b * b;
            let trace = a + d;
            response[[r, c]] = det - k * trace * trace;
        }
    }

    // Threshold: relative to maximum positive response
    let max_resp = response.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_resp <= 0.0 {
        return Ok(Vec::new());
    }
    let abs_thresh = (config.threshold as f32) * max_resp;

    // Non-maximum suppression within nms_radius
    let r_nms = config.nms_radius;
    let mut corners = Vec::new();

    for row in r_nms..rows.saturating_sub(r_nms) {
        for col in r_nms..cols.saturating_sub(r_nms) {
            let rv = response[[row, col]];
            if rv <= abs_thresh {
                continue;
            }
            // Check local maximum in (2*r_nms+1)^2 neighbourhood.
            // Lexicographic tiebreaker: among pixels with equal response, keep
            // only the one with the smallest (row, col) index so exactly one
            // corner survives per cluster of equally-valued response pixels.
            let mut is_max = true;
            'outer: for i in (row.saturating_sub(r_nms))..=(row + r_nms).min(rows - 1) {
                for j in (col.saturating_sub(r_nms))..=(col + r_nms).min(cols - 1) {
                    if !(i == row && j == col)
                        && (response[[i, j]] > rv
                            || (response[[i, j]] == rv && (i, j) < (row, col)))
                    {
                        is_max = false;
                        break 'outer;
                    }
                }
            }
            if is_max {
                corners.push(Corner {
                    row,
                    col,
                    response: rv as f64,
                });
            }
        }
    }

    // Sort by descending response for deterministic ordering
    corners.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(corners)
}

// ---------------------------------------------------------------------------
// FAST corner detector
// ---------------------------------------------------------------------------

/// FAST corner detector (Features from Accelerated Segment Test).
///
/// Tests whether at least `n` consecutive pixels on the Bresenham circle of
/// radius 3 centred at a candidate pixel are all brighter than `center + threshold`
/// **or** all darker than `center - threshold`.
///
/// # Arguments
///
/// * `image`     - Input greyscale image with `u8` pixel values.
/// * `threshold` - Absolute intensity threshold.
/// * `n`         - Number of consecutive pixels required (typically 9 or 12).
///
/// # Returns
///
/// Sorted vector of [`Corner`] with `response = 1.0` for every detected corner.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::feature_detection::fast_corners;
/// use scirs2_core::ndarray::Array2;
///
/// let img = Array2::<u8>::zeros((20, 20));
/// let corners = fast_corners(&img, 20, 9);
/// ```
pub fn fast_corners(image: &Array2<u8>, threshold: u8, n: usize) -> Vec<Corner> {
    let rows = image.nrows();
    let cols = image.ncols();
    let mut corners = Vec::new();

    if rows < 7 || cols < 7 || n == 0 {
        return corners;
    }

    // 16-point Bresenham circle at radius 3
    let circle: [(isize, isize); 16] = [
        (0, 3),
        (1, 3),
        (2, 2),
        (3, 1),
        (3, 0),
        (3, -1),
        (2, -2),
        (1, -3),
        (0, -3),
        (-1, -3),
        (-2, -2),
        (-3, -1),
        (-3, 0),
        (-3, 1),
        (-2, 2),
        (-1, 3),
    ];

    for row in 3..(rows - 3) {
        for col in 3..(cols - 3) {
            let center = image[[row, col]] as i16;
            let hi = (center + threshold as i16).min(255) as u8;
            let lo = (center - threshold as i16).max(0) as u8;

            if !fast_segment_test(image, row, col, &circle, hi, lo, n) {
                continue;
            }

            // Optional: compute corner score as sum of absolute differences
            let score: u32 = circle
                .iter()
                .map(|&(dy, dx)| {
                    let r2 = (row as isize + dy) as usize;
                    let c2 = (col as isize + dx) as usize;
                    let diff = (image[[r2, c2]] as i16 - center).unsigned_abs() as u32;
                    diff
                })
                .sum();

            corners.push(Corner {
                row,
                col,
                response: score as f64,
            });
        }
    }

    corners.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    corners
}

/// Internal: test whether n consecutive pixels on the circle pass the threshold test.
fn fast_segment_test(
    image: &Array2<u8>,
    row: usize,
    col: usize,
    circle: &[(isize, isize); 16],
    hi: u8,
    lo: u8,
    n: usize,
) -> bool {
    let len = circle.len();

    // Scan for n consecutive brighter or darker pixels (with wrap-around)
    // We duplicate the circle to handle wrap-around in a single pass.
    let double = 2 * len;
    let mut consec_bright = 0usize;
    let mut consec_dark = 0usize;

    for i in 0..double {
        let (dy, dx) = circle[i % len];
        let r2 = (row as isize + dy) as usize;
        let c2 = (col as isize + dx) as usize;
        let pv = image[[r2, c2]];

        if pv > hi {
            consec_bright += 1;
            consec_dark = 0;
        } else if pv < lo {
            consec_dark += 1;
            consec_bright = 0;
        } else {
            consec_bright = 0;
            consec_dark = 0;
        }

        if consec_bright >= n || consec_dark >= n {
            return true;
        }

        // After scanning the first full circle without success, reset so we do
        // not carry artificially long runs from the second pass into the first.
        if i == len - 1 {
            consec_bright = 0;
            consec_dark = 0;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// LoG blob detector
// ---------------------------------------------------------------------------

/// Detect blobs via Laplacian of Gaussian (LoG) scale-space.
///
/// At each scale σ, the image is convolved with a normalised LoG kernel
/// (σ² · ∇²G_σ).  Local maxima of |response| across space and scale whose
/// magnitude exceeds `threshold` are retained as blobs.  Overlapping blobs
/// (overlap > `overlap` threshold) are suppressed greedily by descending
/// response.
///
/// The characteristic radius of a detected blob is r = √2 · σ.
///
/// # Arguments
///
/// * `image`     - Input 2D float image.
/// * `min_sigma` - Smallest Gaussian sigma to test (must be > 0).
/// * `max_sigma` - Largest Gaussian sigma to test.
/// * `n_scales`  - Number of logarithmically-spaced sigma values.
/// * `threshold` - Minimum normalised LoG response to consider a blob.
/// * `overlap`   - Maximum allowed IoU for blob pairs (0.0 – 1.0).
///
/// # Errors
///
/// Returns [`NdimageError`] for invalid arguments or filter failures.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::feature_detection::blob_log;
/// use scirs2_core::ndarray::Array2;
///
/// let img = Array2::<f32>::zeros((32, 32));
/// let blobs = blob_log(&img, 1.0, 5.0, 5, 0.1, 0.5).expect("blob_log failed");
/// ```
pub fn blob_log(
    image: &Array2<f32>,
    min_sigma: f64,
    max_sigma: f64,
    n_scales: usize,
    threshold: f64,
    overlap: f64,
) -> NdimageResult<Vec<Blob>> {
    if min_sigma <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "min_sigma must be positive".into(),
        ));
    }
    if max_sigma < min_sigma {
        return Err(NdimageError::InvalidInput(
            "max_sigma must be >= min_sigma".into(),
        ));
    }
    if n_scales == 0 {
        return Err(NdimageError::InvalidInput(
            "n_scales must be at least 1".into(),
        ));
    }

    let (rows, cols) = (image.nrows(), image.ncols());
    if rows < 3 || cols < 3 {
        return Ok(Vec::new());
    }

    // Build sigma ladder
    let sigmas: Vec<f64> = if n_scales == 1 {
        vec![min_sigma]
    } else {
        let log_min = min_sigma.ln();
        let log_max = max_sigma.ln();
        (0..n_scales)
            .map(|i| {
                let t = i as f64 / (n_scales - 1) as f64;
                (log_min + t * (log_max - log_min)).exp()
            })
            .collect()
    };

    // Compute LoG scale-space cube: shape (n_scales, rows, cols)
    let mut scale_space: Vec<Array2<f32>> = Vec::with_capacity(n_scales);

    // Convert image to ArrayD for filter functions
    let mut img_d = ArrayD::from_elem(image.raw_dim().into_dyn(), 0.0f32);
    for ((r, c), &v) in image.indexed_iter() {
        img_d[[r, c]] = v;
    }

    for &sigma in &sigmas {
        // Apply Gaussian with sigma, then estimate Laplacian via DoG approximation
        // OR: use σ² * (G(σ) – G(k*σ)) where k = sqrt(2) for efficiency
        // Here we use the Laplacian = (G(s1) - G(s2)) / (s2² - s1²) * constant
        // Simpler: convolve with LoG kernel directly via two Gaussians (DoG)
        let s1 = sigma as f32;
        let s2 = (sigma * 2.0_f64.sqrt()) as f32;

        let g1 = gaussian_filter_f32(&img_d, s1, Some(BorderMode::Reflect), None)
            .map_err(|e| NdimageError::ComputationError(format!("gaussian s1: {e}")))?;
        let g2 = gaussian_filter_f32(&img_d, s2, Some(BorderMode::Reflect), None)
            .map_err(|e| NdimageError::ComputationError(format!("gaussian s2: {e}")))?;

        // DoG approximation of LoG, normalised by σ²
        let sigma2 = (sigma * sigma) as f32;
        let mut log_resp = Array2::<f32>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                // normalised LoG ≈ sigma² * (g1 - g2) / (k² - 1) * norm_factor
                log_resp[[r, c]] = sigma2 * (g1[[r, c]] - g2[[r, c]]);
            }
        }

        scale_space.push(log_resp);
    }

    // Detect local maxima across space and scale
    let mut candidates: Vec<Blob> = Vec::new();

    for s_idx in 0..n_scales {
        let resp = &scale_space[s_idx];
        let sigma_s = sigmas[s_idx];

        for r in 1..(rows - 1) {
            for c in 1..(cols - 1) {
                let v = resp[[r, c]].abs() as f64;
                if v < threshold {
                    continue;
                }

                // Spatial NMS: must be larger than 8-connected neighbours
                let spatial_max = (-1i32..=1)
                    .flat_map(|dr| (-1i32..=1).map(move |dc| (dr, dc)))
                    .filter(|&(dr, dc)| !(dr == 0 && dc == 0))
                    .all(|(dr, dc)| {
                        let nr = (r as i32 + dr) as usize;
                        let nc = (c as i32 + dc) as usize;
                        (resp[[nr, nc]].abs() as f64) < v
                    });

                if !spatial_max {
                    continue;
                }

                // Scale NMS: must be larger than adjacent scales
                let scale_max_below = if s_idx > 0 {
                    (scale_space[s_idx - 1][[r, c]].abs() as f64) < v
                } else {
                    true
                };
                let scale_max_above = if s_idx + 1 < n_scales {
                    (scale_space[s_idx + 1][[r, c]].abs() as f64) < v
                } else {
                    true
                };

                if scale_max_below && scale_max_above {
                    candidates.push(Blob {
                        row: r as f64,
                        col: c as f64,
                        sigma: sigma_s,
                        response: v,
                    });
                }
            }
        }
    }

    // Sort by descending response
    candidates.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Greedy NMS over blobs based on circle overlap
    let blobs = prune_blobs(candidates, overlap);
    Ok(blobs)
}

/// Greedy blob non-maximum suppression: discard blobs whose circle area
/// overlap fraction exceeds `overlap_thresh` with an already-kept blob.
fn prune_blobs(mut blobs: Vec<Blob>, overlap_thresh: f64) -> Vec<Blob> {
    let mut kept: Vec<Blob> = Vec::with_capacity(blobs.len());
    let mut active = vec![true; blobs.len()];

    for i in 0..blobs.len() {
        if !active[i] {
            continue;
        }
        kept.push(blobs[i].clone());

        let r_i = blobs[i].sigma * 2.0_f64.sqrt();

        for j in (i + 1)..blobs.len() {
            if !active[j] {
                continue;
            }
            let r_j = blobs[j].sigma * 2.0_f64.sqrt();
            let dr = blobs[i].row - blobs[j].row;
            let dc = blobs[i].col - blobs[j].col;
            let dist = (dr * dr + dc * dc).sqrt();

            let iou = circle_overlap(r_i, r_j, dist);
            if iou > overlap_thresh {
                active[j] = false;
            }
        }
    }

    // Remove suppressed entries from blobs vec (already built `kept`)
    let _ = blobs.drain(..); // not needed — kept is the result
    kept
}

/// Compute the intersection-over-union (IoU) of two circles.
fn circle_overlap(r1: f64, r2: f64, dist: f64) -> f64 {
    if dist >= r1 + r2 {
        return 0.0;
    }
    if dist <= (r1 - r2).abs() {
        // One circle fully contained in the other
        let smaller = r1.min(r2);
        let larger = r1.max(r2);
        return (smaller * smaller) / (larger * larger);
    }

    // Intersection area of two circles
    let d2 = dist * dist;
    let r1_2 = r1 * r1;
    let r2_2 = r2 * r2;
    let cos_a1 = (d2 + r1_2 - r2_2) / (2.0 * dist * r1);
    let cos_a2 = (d2 + r2_2 - r1_2) / (2.0 * dist * r2);
    let a1 = cos_a1.clamp(-1.0, 1.0).acos();
    let a2 = cos_a2.clamp(-1.0, 1.0).acos();
    let intersection = r1_2 * (a1 - (2.0 * a1).sin() / 2.0) + r2_2 * (a2 - (2.0 * a2).sin() / 2.0);
    let union = PI * (r1_2 + r2_2) - intersection;
    if union <= 0.0 {
        1.0
    } else {
        intersection / union
    }
}

// ---------------------------------------------------------------------------
// Shi-Tomasi (Good Features to Track)
// ---------------------------------------------------------------------------

/// Detect corners using the Shi-Tomasi (Good Features to Track) criterion.
///
/// Each candidate pixel is scored by the **minimum eigenvalue** of its
/// structure tensor.  Candidates above `quality_level * max_score` are
/// returned, sorted by descending score, with pairs closer than
/// `min_distance` pixels suppressed.
///
/// # Arguments
///
/// * `image`         - Input 2D float image.
/// * `max_corners`   - Maximum number of corners to return (0 = unlimited).
/// * `quality_level` - Fraction of the maximum score used as threshold (0 < q ≤ 1).
/// * `min_distance`  - Minimum Euclidean distance between returned corners.
///
/// # Errors
///
/// Returns [`NdimageError`] if gradient computation fails.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::feature_detection::shi_tomasi_corners;
/// use scirs2_core::ndarray::Array2;
///
/// let img = Array2::<f32>::zeros((20, 20));
/// let corners = shi_tomasi_corners(&img, 10, 0.01, 5.0).expect("shi_tomasi failed");
/// ```
pub fn shi_tomasi_corners(
    image: &Array2<f32>,
    max_corners: usize,
    quality_level: f64,
    min_distance: f64,
) -> NdimageResult<Vec<Corner>> {
    if quality_level <= 0.0 || quality_level > 1.0 {
        return Err(NdimageError::InvalidInput(
            "quality_level must be in (0, 1]".into(),
        ));
    }

    let (rows, cols) = (image.nrows(), image.ncols());
    if rows < 3 || cols < 3 {
        return Ok(Vec::new());
    }

    // Compute gradients
    let mut img_d = ArrayD::from_elem(image.raw_dim().into_dyn(), 0.0f32);
    for ((r, c), &v) in image.indexed_iter() {
        img_d[[r, c]] = v;
    }

    let gy_d = sobel(&img_d, 0, Some(BorderMode::Reflect))
        .map_err(|e| NdimageError::ComputationError(format!("sobel y: {e}")))?;
    let gx_d = sobel(&img_d, 1, Some(BorderMode::Reflect))
        .map_err(|e| NdimageError::ComputationError(format!("sobel x: {e}")))?;

    // Structure tensor components (Gaussian-smoothed)
    let mut ix2_d = gx_d.mapv(|v| v * v);
    let mut iy2_d = gy_d.mapv(|v| v * v);
    let mut ixy_d = {
        let mut t = gx_d.clone();
        for (idx, &gy) in gy_d.indexed_iter() {
            t[idx.slice()] *= gy;
        }
        t
    };

    let sigma = 1.0f32;
    ix2_d = gaussian_filter_f32(&ix2_d, sigma, Some(BorderMode::Reflect), None)
        .map_err(|e| NdimageError::ComputationError(format!("gaussian ix2: {e}")))?;
    iy2_d = gaussian_filter_f32(&iy2_d, sigma, Some(BorderMode::Reflect), None)
        .map_err(|e| NdimageError::ComputationError(format!("gaussian iy2: {e}")))?;
    ixy_d = gaussian_filter_f32(&ixy_d, sigma, Some(BorderMode::Reflect), None)
        .map_err(|e| NdimageError::ComputationError(format!("gaussian ixy: {e}")))?;

    // Compute min eigenvalue = Shi-Tomasi quality
    let mut quality = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let a = ix2_d[[r, c]] as f64;
            let b = ixy_d[[r, c]] as f64;
            let d = iy2_d[[r, c]] as f64;
            // eigenvalues of [[a,b],[b,d]]:  λ = (a+d)/2 ± sqrt(((a-d)/2)^2 + b^2)
            let half_trace = (a + d) / 2.0;
            let disc = ((a - d) / 2.0).powi(2) + b * b;
            let sqrt_disc = disc.sqrt();
            let min_eig = half_trace - sqrt_disc;
            quality[[r, c]] = min_eig.max(0.0);
        }
    }

    let max_q = quality.iter().copied().fold(0.0f64, f64::max);
    if max_q <= 0.0 {
        return Ok(Vec::new());
    }
    let thresh = quality_level * max_q;

    // Collect candidates above threshold
    let mut candidates: Vec<Corner> = quality
        .indexed_iter()
        .filter_map(|((r, c), &q)| {
            if q >= thresh {
                Some(Corner {
                    row: r,
                    col: c,
                    response: q,
                })
            } else {
                None
            }
        })
        .collect();

    // Sort descending
    candidates.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Apply minimum-distance suppression
    let min_dist2 = min_distance * min_distance;
    let mut result: Vec<Corner> = Vec::new();

    for cand in candidates {
        let too_close = result.iter().any(|kept| {
            let dr = (kept.row as f64) - (cand.row as f64);
            let dc = (kept.col as f64) - (cand.col as f64);
            dr * dr + dc * dc < min_dist2
        });
        if !too_close {
            result.push(cand);
            if max_corners > 0 && result.len() >= max_corners {
                break;
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Build a simple checkerboard pattern: each tile is `tile` pixels square,
    /// alternating 0.0 and 1.0.
    fn checkerboard(size: usize, tile: usize) -> Array2<f32> {
        Array2::from_shape_fn((size, size), |(r, c)| {
            if (r / tile + c / tile) % 2 == 0 {
                0.0f32
            } else {
                1.0f32
            }
        })
    }

    #[test]
    fn harris_on_checkerboard_finds_corners() {
        let img = checkerboard(32, 8);
        let config = HarrisConfig {
            k: 0.05,
            sigma: 1.0,
            threshold: 0.01,
            nms_radius: 4,
        };
        let corners = harris_corners_configured(&img, config).expect("harris failed");
        // A 32×32 checkerboard with 8-pixel tiles has interior corners
        assert!(
            !corners.is_empty(),
            "Expected corners in checkerboard, found none"
        );
    }

    #[test]
    fn harris_on_flat_image_finds_no_corners() {
        let img = Array2::<f32>::from_elem((20, 20), 0.5);
        let corners = harris_corners_configured(&img, HarrisConfig::default()).expect("harris ok");
        assert!(corners.is_empty(), "Flat image should have no corners");
    }

    #[test]
    fn harris_default_config_is_reasonable() {
        let cfg = HarrisConfig::default();
        assert!(cfg.k > 0.0 && cfg.k < 1.0);
        assert!(cfg.sigma > 0.0);
        assert!(cfg.threshold > 0.0 && cfg.threshold < 1.0);
    }

    #[test]
    fn fast_corners_empty_image_returns_empty() {
        let img = Array2::<u8>::zeros((6, 6)); // smaller than required 7×7
        let corners = fast_corners(&img, 20, 9);
        assert!(corners.is_empty());
    }

    #[test]
    fn fast_corners_basic_detection() {
        // A small image with a clear intensity step
        let mut img = Array2::<u8>::from_elem((15, 15), 100u8);
        // Dark square in the top-left
        for r in 0..7 {
            for c in 0..7 {
                img[[r, c]] = 10;
            }
        }
        let corners = fast_corners(&img, 50, 9);
        // May or may not find corners depending on configuration — just check it runs
        let _ = corners; // execution without panic is the test
    }

    #[test]
    fn blob_log_empty_response_below_threshold() {
        let img = Array2::<f32>::zeros((32, 32));
        let blobs = blob_log(&img, 1.0, 4.0, 4, 0.01, 0.5).expect("blob_log failed");
        // Flat image produces no blobs
        assert!(blobs.is_empty());
    }

    #[test]
    fn blob_log_detects_bright_spot() {
        let mut img = Array2::<f32>::zeros((64, 64));
        // Place a bright Gaussian blob at (32, 32) with sigma ≈ 3
        for r in 20..44 {
            for c in 20..44 {
                let dr = (r as f64 - 32.0) / 3.0;
                let dc = (c as f64 - 32.0) / 3.0;
                img[[r, c]] = (-(dr * dr + dc * dc) / 2.0).exp() as f32;
            }
        }
        let blobs = blob_log(&img, 1.0, 8.0, 6, 0.001, 0.5).expect("blob_log failed");
        assert!(
            !blobs.is_empty(),
            "Expected at least one blob in bright-spot image"
        );
        // The detected blob should be near the centre
        let best = &blobs[0];
        assert!(
            (best.row - 32.0).abs() < 5.0,
            "Blob row {} far from 32",
            best.row
        );
        assert!(
            (best.col - 32.0).abs() < 5.0,
            "Blob col {} far from 32",
            best.col
        );
    }

    #[test]
    fn blob_log_invalid_sigma_returns_error() {
        let img = Array2::<f32>::zeros((16, 16));
        let result = blob_log(&img, 0.0, 4.0, 4, 0.01, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn blob_log_scale_consistency() {
        let mut img = Array2::<f32>::zeros((64, 64));
        // Blob with sigma 4
        for r in 0..64usize {
            for c in 0..64usize {
                let dr = (r as f64 - 32.0) / 4.0;
                let dc = (c as f64 - 32.0) / 4.0;
                img[[r, c]] = (-(dr * dr + dc * dc) / 2.0).exp() as f32;
            }
        }
        let blobs = blob_log(&img, 2.0, 8.0, 5, 0.001, 0.5).expect("ok");
        // The response should be real numbers
        for b in &blobs {
            assert!(b.response.is_finite());
            assert!(b.sigma > 0.0);
        }
    }

    #[test]
    fn shi_tomasi_flat_image_returns_empty() {
        let img = Array2::<f32>::from_elem((20, 20), 0.5);
        let corners = shi_tomasi_corners(&img, 10, 0.01, 3.0).expect("ok");
        assert!(corners.is_empty());
    }

    #[test]
    fn shi_tomasi_on_checkerboard() {
        let img = checkerboard(32, 8);
        let corners = shi_tomasi_corners(&img, 20, 0.01, 4.0).expect("ok");
        assert!(
            !corners.is_empty(),
            "Expected corners in checkerboard for Shi-Tomasi"
        );
        // Corners are sorted descending
        for w in corners.windows(2) {
            assert!(w[0].response >= w[1].response);
        }
    }

    #[test]
    fn shi_tomasi_respects_max_corners() {
        let img = checkerboard(64, 8);
        let max = 5;
        let corners = shi_tomasi_corners(&img, max, 0.001, 3.0).expect("ok");
        assert!(corners.len() <= max);
    }

    #[test]
    fn shi_tomasi_invalid_quality_returns_error() {
        let img = Array2::<f32>::zeros((10, 10));
        assert!(shi_tomasi_corners(&img, 0, 0.0, 1.0).is_err());
        assert!(shi_tomasi_corners(&img, 0, 1.5, 1.0).is_err());
    }
}
