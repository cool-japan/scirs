//! ORB (Oriented FAST and Rotated BRIEF) descriptor.
//!
//! Implements a multi-scale keypoint detector (FAST-style corners) with
//! intensity-centroid orientation estimation and a steered rBRIEF descriptor.
//!
//! Reference: Rublee et al., "ORB: An efficient alternative to SIFT or SURF",
//! ICCV 2011.

use super::brief::{generate_test_pairs, BRIEFPattern};
use crate::error::VisionError;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for ORB keypoint detection and description.
#[derive(Debug, Clone)]
pub struct ORBConfig {
    /// Maximum number of keypoints to retain across all scales.
    pub n_keypoints: usize,
    /// Scale factor between successive pyramid levels (> 1.0).
    pub scale_factor: f32,
    /// Number of pyramid levels.
    pub n_levels: usize,
    /// FAST corner threshold (pixel intensity difference).
    pub fast_threshold: u8,
    /// Border margin (pixels) excluded from detection.
    pub edge_threshold: usize,
    /// Patch radius for descriptor and orientation (pixels).
    pub patch_size: usize,
}

impl Default for ORBConfig {
    fn default() -> Self {
        Self {
            n_keypoints: 500,
            scale_factor: 1.2,
            n_levels: 8,
            fast_threshold: 20,
            edge_threshold: 31,
            patch_size: 31,
        }
    }
}

// ─── Keypoint / Descriptor ───────────────────────────────────────────────────

/// An ORB keypoint.
#[derive(Debug, Clone)]
pub struct ORBKeypoint {
    /// Column coordinate in the original image.
    pub x: f32,
    /// Row coordinate in the original image.
    pub y: f32,
    /// Scale (relative to original).
    pub scale: f32,
    /// Orientation in radians.
    pub angle: f32,
    /// Corner response score.
    pub response: f32,
    /// Pyramid level.
    pub level: usize,
}

/// ORB binary descriptor (256-bit rBRIEF).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ORBDescriptor {
    /// Packed bits (4 × u64 = 256 bits).
    pub bits: Vec<u64>,
}

// ─── Gaussian blur ───────────────────────────────────────────────────────────

/// Very fast 5-tap Gaussian blur (σ ≈ 1.0) applied in-place via separable kernel.
fn gaussian_blur_5(img: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let rows = img.len();
    if rows == 0 {
        return vec![];
    }
    let cols = img[0].len();
    // kernel: [1, 4, 6, 4, 1] / 16
    const K: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
    const K_SUM: f32 = 16.0;

    let get = |r: usize, c: i32| -> f32 {
        let cc = c.clamp(0, cols as i32 - 1) as usize;
        img[r][cc]
    };

    // Horizontal pass.
    let mut tmp = vec![vec![0.0_f32; cols]; rows];
    #[allow(clippy::needless_range_loop)]
    for r in 0..rows {
        for c in 0..cols {
            let mut v = 0.0_f32;
            for (ki, &kv) in K.iter().enumerate() {
                v += get(r, c as i32 + ki as i32 - 2) * kv;
            }
            tmp[r][c] = v / K_SUM;
        }
    }

    let get2 = |r: i32, c: usize| -> f32 {
        let rr = r.clamp(0, rows as i32 - 1) as usize;
        tmp[rr][c]
    };

    // Vertical pass.
    let mut out = vec![vec![0.0_f32; cols]; rows];
    #[allow(clippy::needless_range_loop)]
    for r in 0..rows {
        for c in 0..cols {
            let mut v = 0.0_f32;
            for (ki, &kv) in K.iter().enumerate() {
                v += get2(r as i32 + ki as i32 - 2, c) * kv;
            }
            out[r][c] = v / K_SUM;
        }
    }
    out
}

// ─── Image pyramid ───────────────────────────────────────────────────────────

/// Down-scale `image` by `factor` using bilinear interpolation.
fn downscale(image: &[Vec<f32>], factor: f32) -> Vec<Vec<f32>> {
    if factor <= 1.0 {
        return image.to_vec();
    }
    let src_rows = image.len();
    let src_cols = image.first().map_or(0, |r| r.len());
    let dst_rows = ((src_rows as f32 / factor).floor() as usize).max(1);
    let dst_cols = ((src_cols as f32 / factor).floor() as usize).max(1);

    let mut out = vec![vec![0.0_f32; dst_cols]; dst_rows];
    #[allow(clippy::needless_range_loop)]
    for r in 0..dst_rows {
        for c in 0..dst_cols {
            let sr = r as f32 * factor;
            let sc = c as f32 * factor;
            let r0 = sr.floor() as usize;
            let c0 = sc.floor() as usize;
            let r1 = (r0 + 1).min(src_rows - 1);
            let c1 = (c0 + 1).min(src_cols - 1);
            let dr = sr - sr.floor();
            let dc = sc - sc.floor();
            out[r][c] = image[r0][c0] * (1.0 - dr) * (1.0 - dc)
                + image[r0][c1] * (1.0 - dr) * dc
                + image[r1][c0] * dr * (1.0 - dc)
                + image[r1][c1] * dr * dc;
        }
    }
    out
}

// ─── FAST corner detection ───────────────────────────────────────────────────

/// Bresenham circle of radius 3 used by FAST-9.
const CIRCLE_3: [(i32, i32); 16] = [
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

/// Detect FAST-9 corners in `image`.  Returns `(row, col, score)` triples.
fn detect_fast(image: &[Vec<f32>], threshold: f32, edge: usize) -> Vec<(usize, usize, f32)> {
    let rows = image.len();
    let cols = image.first().map_or(0, |r| r.len());
    let mut corners = Vec::new();

    for r in edge..rows.saturating_sub(edge) {
        for c in edge..cols.saturating_sub(edge) {
            let ip = image[r][c];
            let hi = ip + threshold;
            let lo = ip - threshold;

            // Fast reject: check pixels 1, 5, 9, 13 (compass directions).
            let compass = [CIRCLE_3[0], CIRCLE_3[4], CIRCLE_3[8], CIRCLE_3[12]];
            let n_above = compass
                .iter()
                .filter(|&&(dr, dc)| image[(r as i32 + dr) as usize][(c as i32 + dc) as usize] > hi)
                .count();
            let n_below = compass
                .iter()
                .filter(|&&(dr, dc)| image[(r as i32 + dr) as usize][(c as i32 + dc) as usize] < lo)
                .count();
            if n_above < 2 && n_below < 2 {
                continue;
            }

            // Full FAST-9: check for 9 consecutive bright or dark pixels.
            let vals: Vec<f32> = CIRCLE_3
                .iter()
                .map(|&(dr, dc)| image[(r as i32 + dr) as usize][(c as i32 + dc) as usize])
                .collect();

            let is_corner = {
                let mut found = false;
                'outer: for start in 0..16 {
                    // Check 9 consecutive (wrapping) pixels all brighter.
                    let all_hi = (0..9).all(|k| vals[(start + k) % 16] > hi);
                    let all_lo = (0..9).all(|k| vals[(start + k) % 16] < lo);
                    if all_hi || all_lo {
                        found = true;
                        break 'outer;
                    }
                }
                found
            };

            if is_corner {
                // Score = difference sum.
                let score: f32 = vals.iter().map(|&v| (v - ip).abs()).sum();
                corners.push((r, c, score));
            }
        }
    }
    corners
}

// ─── Orientation via intensity centroid ──────────────────────────────────────

/// Compute patch orientation using the intensity centroid method.
fn compute_orientation(image: &[Vec<f32>], r: usize, c: usize, half_patch: usize) -> f32 {
    let rows = image.len();
    let cols = image.first().map_or(0, |r| r.len());
    let mut m01 = 0.0_f32;
    let mut m10 = 0.0_f32;

    let hp = half_patch as i32;
    for dr in -hp..=hp {
        let rr = (r as i32 + dr).clamp(0, rows as i32 - 1) as usize;
        for dc in -hp..=hp {
            if dr * dr + dc * dc > hp * hp {
                continue;
            }
            let cc = (c as i32 + dc).clamp(0, cols as i32 - 1) as usize;
            let v = image[rr][cc];
            m10 += dc as f32 * v;
            m01 += dr as f32 * v;
        }
    }
    m01.atan2(m10)
}

// ─── Steered BRIEF ───────────────────────────────────────────────────────────

/// Rotate a test pair by `angle` (radians) and round to integer offsets.
#[inline]
fn rotate_pair(dy: i32, dx: i32, angle: f32) -> (i32, i32) {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let dy_r = (sin_a * dx as f32 + cos_a * dy as f32).round() as i32;
    let dx_r = (cos_a * dx as f32 - sin_a * dy as f32).round() as i32;
    (dy_r, dx_r)
}

/// Compute a steered BRIEF descriptor for a keypoint.
#[allow(clippy::type_complexity)]
fn compute_steered_brief(
    image: &[Vec<f32>],
    r: usize,
    c: usize,
    angle: f32,
    pairs: &[((i32, i32), (i32, i32))],
) -> ORBDescriptor {
    let rows = image.len();
    let cols = image.first().map_or(0, |r| r.len());
    let n_bits = pairs.len();
    let n_words = n_bits.div_ceil(64);
    let mut bits = vec![0_u64; n_words];

    let get = |dy: i32, dx: i32| -> f32 {
        let rr = (r as i32 + dy).clamp(0, rows as i32 - 1) as usize;
        let cc = (c as i32 + dx).clamp(0, cols as i32 - 1) as usize;
        image[rr][cc]
    };

    for (i, &((dy_p, dx_p), (dy_q, dx_q))) in pairs.iter().enumerate() {
        let (dy_p_r, dx_p_r) = rotate_pair(dy_p, dx_p, angle);
        let (dy_q_r, dx_q_r) = rotate_pair(dy_q, dx_q, angle);
        if get(dy_p_r, dx_p_r) < get(dy_q_r, dx_q_r) {
            bits[i / 64] |= 1_u64 << (i % 64);
        }
    }

    ORBDescriptor { bits }
}

// ─── Non-maximum suppression ─────────────────────────────────────────────────

/// Retain at most `max_kp` keypoints per level via response-ordered NMS.
fn nms_keypoints(
    keypoints: Vec<(usize, usize, f32)>,
    max_kp: usize,
    radius: usize,
) -> Vec<(usize, usize, f32)> {
    let mut sorted = keypoints;
    sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut retained: Vec<(usize, usize, f32)> = Vec::with_capacity(max_kp);
    for kp in sorted {
        let too_close = retained.iter().any(|&(r2, c2, _)| {
            let dr = (kp.0 as i32 - r2 as i32).unsigned_abs() as usize;
            let dc = (kp.1 as i32 - c2 as i32).unsigned_abs() as usize;
            dr < radius && dc < radius
        });
        if !too_close {
            retained.push(kp);
            if retained.len() >= max_kp {
                break;
            }
        }
    }
    retained
}

// ─── Main API ────────────────────────────────────────────────────────────────

/// Detect and describe ORB keypoints in `image`.
///
/// Returns a list of `(ORBKeypoint, ORBDescriptor)` pairs sorted by response
/// (descending).
///
/// # Errors
/// Returns [`VisionError`] if the image is empty.
pub fn detect_orb(
    image: &[Vec<f32>],
    config: &ORBConfig,
) -> Result<Vec<(ORBKeypoint, ORBDescriptor)>, VisionError> {
    let rows = image.len();
    let cols = image.first().map_or(0, |r| r.len());
    if rows == 0 || cols == 0 {
        return Err(VisionError::InvalidInput("Empty image for ORB".into()));
    }

    // Pre-generate test pairs once.
    let n_bits = 256_usize;
    let half_patch = (config.patch_size / 2) as i32;
    let min_edge = (half_patch as f32 * std::f32::consts::SQRT_2).ceil() as usize + 1;
    let pairs = generate_test_pairs(
        n_bits,
        config.patch_size,
        BRIEFPattern::Gaussian,
        0x0EB31337,
    );
    let threshold = config.fast_threshold as f32 / 255.0;
    let kp_per_level = (config.n_keypoints / config.n_levels).max(1);
    let nms_radius = (config.patch_size / 4).max(3);

    let mut all_kp: Vec<(ORBKeypoint, ORBDescriptor)> = Vec::new();
    let mut current = gaussian_blur_5(image);
    let mut scale = 1.0_f32;

    for level in 0..config.n_levels {
        if current.is_empty() || current[0].is_empty() {
            break;
        }
        let corners = detect_fast(
            &current,
            threshold,
            config.edge_threshold.max(min_edge).max(4),
        );
        let selected = nms_keypoints(corners, kp_per_level, nms_radius);

        for (r, c, response) in selected {
            let angle = compute_orientation(&current, r, c, config.patch_size / 2);
            let desc = compute_steered_brief(&current, r, c, angle, &pairs);
            let kp = ORBKeypoint {
                x: c as f32 * scale,
                y: r as f32 * scale,
                scale,
                angle,
                response,
                level,
            };
            all_kp.push((kp, desc));
        }

        // Build next pyramid level.
        scale *= config.scale_factor;
        let blurred = gaussian_blur_5(&current);
        current = downscale(&blurred, config.scale_factor);
    }

    // Global sort by response and trim to n_keypoints.
    all_kp.sort_by(|a, b| {
        b.0.response
            .partial_cmp(&a.0.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_kp.truncate(config.n_keypoints);

    Ok(all_kp)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard(rows: usize, cols: usize, sq: usize) -> Vec<Vec<f32>> {
        (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| {
                        if (r / sq + c / sq).is_multiple_of(2) {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_orb_returns_keypoints() {
        let img = checkerboard(128, 128, 8);
        let cfg = ORBConfig {
            n_keypoints: 50,
            n_levels: 3,
            ..Default::default()
        };
        let kps = detect_orb(&img, &cfg).expect("detect_orb should succeed on valid image");
        // Should detect some corners on a checkerboard.
        assert!(!kps.is_empty(), "Expected at least one keypoint");
    }

    #[test]
    fn test_orb_descriptor_bits() {
        let img = checkerboard(128, 128, 8);
        let cfg = ORBConfig {
            n_keypoints: 10,
            n_levels: 2,
            ..Default::default()
        };
        let kps = detect_orb(&img, &cfg).expect("detect_orb should succeed on valid image");
        for (_, desc) in &kps {
            assert_eq!(desc.bits.len(), 4, "256-bit descriptor needs 4 words");
        }
    }

    #[test]
    fn test_orb_empty_image() {
        let img: Vec<Vec<f32>> = vec![];
        let cfg = ORBConfig::default();
        assert!(detect_orb(&img, &cfg).is_err());
    }

    #[test]
    fn test_orb_response_sorted() {
        let img = checkerboard(128, 128, 8);
        let cfg = ORBConfig {
            n_keypoints: 20,
            n_levels: 2,
            ..Default::default()
        };
        let kps = detect_orb(&img, &cfg).expect("detect_orb should succeed on valid image");
        for w in kps.windows(2) {
            assert!(w[0].0.response >= w[1].0.response);
        }
    }

    #[test]
    fn test_pyramid_downscale() {
        let img: Vec<Vec<f32>> = (0..64)
            .map(|r| (0..64).map(|c| (r * 64 + c) as f32).collect())
            .collect();
        let out = downscale(&img, 2.0);
        assert_eq!(out.len(), 32);
        assert_eq!(out[0].len(), 32);
    }

    #[test]
    fn test_steered_brief_rotation_invariant_90() {
        let (rows, cols) = (96usize, 80usize);
        let mut s: u32 = 12345;
        let img: Vec<Vec<f32>> = (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|_| {
                        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (s >> 8) as f32 / 16_777_216.0
                    })
                    .collect()
            })
            .collect();

        // Rotate 90° clockwise: point (r, c) moves to (c, rows - 1 - r).
        let rot: Vec<Vec<f32>> = (0..cols)
            .map(|i| (0..rows).map(|j| img[rows - 1 - j][i]).collect())
            .collect();

        let pairs = generate_test_pairs(256, 31, BRIEFPattern::Gaussian, 0x0EB31337);
        let (r, c) = (48, 40);
        let (rb, cb) = (c, rows - 1 - r);

        let ang_a = compute_orientation(&img, r, c, 15);
        let ang_b = compute_orientation(&rot, rb, cb, 15);
        let da = compute_steered_brief(&img, r, c, ang_a, &pairs);
        let db = compute_steered_brief(&rot, rb, cb, ang_b, &pairs);

        let dist: u32 = da
            .bits
            .iter()
            .zip(&db.bits)
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        assert!(
            dist <= 16,
            "descriptor changed under 90° rotation: hamming = {dist}"
        );
    }
}
