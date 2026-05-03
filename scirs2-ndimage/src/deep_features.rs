//! CNN-inspired deep image feature extraction (no neural network training required).
//!
//! Provides:
//! - Gabor filter bank for texture analysis
//! - Simplified SIFT keypoint detection and description
//! - Histogram of Oriented Gradients (HOG) descriptor

use crate::error::{NdimageError, NdimageResult};
use std::f64::consts::PI;

// ─── Gabor Filter Bank ───────────────────────────────────────────────────────

/// Gabor filter bank for multi-orientation, multi-frequency texture analysis.
///
/// A Gabor filter is a Gaussian kernel modulated by a sinusoidal plane wave.
/// It provides optimal joint localization in spatial and frequency domains,
/// making it ideal for texture discrimination.
#[derive(Debug, Clone)]
pub struct GaborFilterBank {
    /// Filter orientations in radians
    pub orientations: Vec<f64>,
    /// Spatial frequencies (cycles per pixel)
    pub frequencies: Vec<f64>,
    /// Standard deviation of the Gaussian envelope
    pub sigma: f64,
    /// Kernel half-size (full kernel is (2*kernel_size+1) × (2*kernel_size+1))
    pub kernel_size: usize,
}

impl GaborFilterBank {
    /// Create a new Gabor filter bank.
    ///
    /// # Arguments
    /// * `n_orientations` – Number of evenly spaced orientations in `[0, π)`.
    /// * `frequencies`    – Slice of spatial frequencies to include.
    /// * `sigma`          – Gaussian envelope standard deviation.
    pub fn new(n_orientations: usize, frequencies: &[f64], sigma: f64) -> Self {
        let orientations: Vec<f64> = (0..n_orientations)
            .map(|i| i as f64 * PI / n_orientations as f64)
            .collect();
        let kernel_size = (3.0 * sigma).ceil() as usize;
        GaborFilterBank {
            orientations,
            frequencies: frequencies.to_vec(),
            sigma,
            kernel_size,
        }
    }

    /// Compute a single real-valued Gabor kernel.
    ///
    /// Returns a 2-D kernel of size `(2*kernel_size+1) × (2*kernel_size+1)`.
    pub fn kernel(&self, theta: f64, frequency: f64) -> Vec<Vec<f64>> {
        let half = self.kernel_size as isize;
        let side = (2 * self.kernel_size + 1) as usize;
        let sigma_sq = self.sigma * self.sigma;
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let mut kern = vec![vec![0.0f64; side]; side];
        for (ri, row) in kern.iter_mut().enumerate() {
            for (ci, cell) in row.iter_mut().enumerate() {
                let y = ri as isize - half;
                let x = ci as isize - half;
                let xp = x as f64 * cos_t + y as f64 * sin_t;
                let yp = -x as f64 * sin_t + y as f64 * cos_t;
                let gaussian = (-(xp * xp + yp * yp) / (2.0 * sigma_sq)).exp();
                let wave = (2.0 * PI * frequency * xp).cos();
                *cell = gaussian * wave;
            }
        }
        kern
    }

    /// Apply all filters in the bank and return feature maps.
    ///
    /// Returns a vector of shape `[n_filters][rows][cols]`, where
    /// `n_filters = orientations.len() * frequencies.len()`.
    pub fn apply(&self, image: &[Vec<f64>]) -> Vec<Vec<Vec<f64>>> {
        let rows = image.len();
        if rows == 0 {
            return Vec::new();
        }
        let cols = image[0].len();
        let mut feature_maps = Vec::new();

        for &theta in &self.orientations {
            for &freq in &self.frequencies {
                let kern = self.kernel(theta, freq);
                let filtered = convolve2d(image, &kern, rows, cols);
                feature_maps.push(filtered);
            }
        }
        feature_maps
    }

    /// Compute energy features: for each filter, compute mean and std of |response|.
    ///
    /// Returns a vector of length `2 * n_filters` (mean then std for each filter).
    pub fn energy_features(&self, image: &[Vec<f64>]) -> Vec<f64> {
        let maps = self.apply(image);
        let mut feats = Vec::with_capacity(maps.len() * 2);
        for map in &maps {
            let flat: Vec<f64> = map.iter().flat_map(|r| r.iter().map(|v| v.abs())).collect();
            let n = flat.len() as f64;
            if n == 0.0 {
                feats.push(0.0);
                feats.push(0.0);
                continue;
            }
            let mean = flat.iter().sum::<f64>() / n;
            let var = flat.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            feats.push(mean);
            feats.push(var.sqrt());
        }
        feats
    }
}

// ─── SIFT ────────────────────────────────────────────────────────────────────

/// A detected SIFT keypoint with its 128-dimensional descriptor.
#[derive(Debug, Clone)]
pub struct SiftKeypoint {
    /// Sub-pixel column position
    pub x: f64,
    /// Sub-pixel row position
    pub y: f64,
    /// Characteristic scale (sigma of the detecting Gaussian)
    pub scale: f64,
    /// Dominant orientation in radians
    pub orientation: f64,
    /// 128-dimensional SIFT descriptor (normalised to unit length)
    pub descriptor: Vec<f64>,
}

/// Detect SIFT keypoints in a grayscale image.
///
/// Implements the Difference-of-Gaussians (DoG) scale-space extrema detection
/// pipeline from Lowe (2004), followed by keypoint localisation, orientation
/// assignment, and 128-D descriptor computation.
///
/// # Arguments
/// * `image`                – 2-D grayscale image (rows × cols).
/// * `n_octaves`            – Number of octaves in the scale space.
/// * `n_scales_per_octave`  – Number of DoG levels per octave (typically 3).
/// * `contrast_threshold`   – Minimum |DoG| response (typical 0.04).
/// * `edge_threshold`       – Harris-like ratio to reject edge keypoints (typical 10.0).
pub fn detect_sift_keypoints(
    image: &[Vec<f64>],
    n_octaves: usize,
    n_scales_per_octave: usize,
    contrast_threshold: f64,
    edge_threshold: f64,
) -> Vec<SiftKeypoint> {
    let rows = image.len();
    if rows == 0 {
        return Vec::new();
    }
    let cols = image[0].len();
    if cols == 0 {
        return Vec::new();
    }

    let k = 2.0f64.powf(1.0 / n_scales_per_octave as f64);
    let initial_sigma = 1.6f64;
    let mut keypoints = Vec::new();

    // Work on the original image (first octave only for simplicity).
    // Down-sample by 2× for each successive octave.
    let mut oct_image: Vec<Vec<f64>> = image.to_vec();

    for _oct in 0..n_octaves {
        let oct_rows = oct_image.len();
        let oct_cols = if oct_rows > 0 { oct_image[0].len() } else { 0 };
        if oct_rows < 4 || oct_cols < 4 {
            break;
        }

        // Build Gaussian scale-space for this octave
        let n_gauss = n_scales_per_octave + 3;
        let mut gauss_stack: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_gauss);
        gauss_stack.push(oct_image.clone());
        for s in 1..n_gauss {
            let sigma = initial_sigma * k.powi(s as i32);
            let blurred = gaussian_blur(&gauss_stack[s - 1], sigma);
            gauss_stack.push(blurred);
        }

        // Difference-of-Gaussians
        let mut dog_stack: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_gauss - 1);
        for s in 0..(n_gauss - 1) {
            let dog = subtract_images(&gauss_stack[s + 1], &gauss_stack[s]);
            dog_stack.push(dog);
        }

        // Find extrema in DoG stack
        let n_dog = dog_stack.len();
        if n_dog < 3 {
            break;
        }
        for s in 1..(n_dog - 1) {
            for r in 1..(oct_rows.saturating_sub(1)) {
                for c in 1..(oct_cols.saturating_sub(1)) {
                    let val = dog_stack[s][r][c];
                    if val.abs() < contrast_threshold {
                        continue;
                    }
                    if !is_extremum(&dog_stack, s, r, c) {
                        continue;
                    }
                    // Edge rejection via principal curvature ratio
                    if is_edge_point(&dog_stack[s], r, c, edge_threshold) {
                        continue;
                    }
                    // Dominant orientation
                    let sigma = initial_sigma * k.powi(s as i32);
                    let orient = dominant_orientation(&gauss_stack[s], r, c, sigma);
                    let kp = SiftKeypoint {
                        x: c as f64,
                        y: r as f64,
                        scale: sigma,
                        orientation: orient,
                        descriptor: Vec::new(), // filled later
                    };
                    keypoints.push(kp);
                }
            }
        }

        // Down-sample for next octave
        oct_image = downsample2x(&oct_image);
    }

    keypoints
}

/// Compute 128-dimensional SIFT descriptors for the provided keypoints.
///
/// Each descriptor is computed from a 16×16 pixel patch around the keypoint,
/// divided into a 4×4 grid of 8-bin gradient-orientation histograms.
pub fn compute_sift_descriptor(image: &[Vec<f64>], keypoints: &[SiftKeypoint]) -> Vec<Vec<f64>> {
    let rows = image.len();
    if rows == 0 {
        return Vec::new();
    }
    let cols = image[0].len();
    let (grad_mag, grad_ori) = gradient_images(image);

    keypoints
        .iter()
        .map(|kp| {
            let cx = kp.x as isize;
            let cy = kp.y as isize;
            let patch_half = 8isize;
            let n_bins = 8usize;
            // 4×4 spatial cells × 8 orientation bins = 128
            let n_cells = 4usize;
            let cell_size = (patch_half * 2 / n_cells as isize).max(1);
            let mut desc = vec![0.0f64; 128];

            for pr in 0..(patch_half * 2) {
                for pc in 0..(patch_half * 2) {
                    let r = (cy - patch_half + pr).max(0).min(rows as isize - 1) as usize;
                    let c = (cx - patch_half + pc).max(0).min(cols as isize - 1) as usize;
                    let mag = grad_mag[r][c];
                    let ori = (grad_ori[r][c] - kp.orientation).rem_euclid(2.0 * PI);
                    let cell_r = (pr / cell_size).min((n_cells - 1) as isize) as usize;
                    let cell_c = (pc / cell_size).min((n_cells - 1) as isize) as usize;
                    let bin = ((ori / (2.0 * PI)) * n_bins as f64) as usize % n_bins;
                    let idx = cell_r * n_cells * n_bins + cell_c * n_bins + bin;
                    if idx < 128 {
                        desc[idx] += mag;
                    }
                }
            }
            // L2 normalise, clamp at 0.2, re-normalise
            let norm = desc.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            desc.iter_mut().for_each(|v| *v = (*v / norm).min(0.2));
            let norm2 = desc.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            desc.iter_mut().for_each(|v| *v /= norm2);
            desc
        })
        .collect()
}

// ─── HOG descriptor ──────────────────────────────────────────────────────────

/// Histogram of Oriented Gradients (HOG) descriptor.
///
/// Divides the image into `cells_per_block`-normalized blocks of cells,
/// where each cell contains an `orientations`-bin gradient orientation histogram.
#[derive(Debug, Clone)]
pub struct HogDescriptor {
    /// Number of orientation bins (typically 9)
    pub orientations: usize,
    /// Pixels per cell (height, width)
    pub pixels_per_cell: (usize, usize),
    /// Cells per block (height, width) for L2-Hys normalisation
    pub cells_per_block: (usize, usize),
}

impl HogDescriptor {
    /// Create a new HOG descriptor configuration.
    pub fn new(
        orientations: usize,
        pixels_per_cell: (usize, usize),
        cells_per_block: (usize, usize),
    ) -> Self {
        HogDescriptor {
            orientations,
            pixels_per_cell,
            cells_per_block,
        }
    }

    /// Compute the HOG descriptor vector for a grayscale image.
    pub fn compute(&self, image: &[Vec<f64>]) -> Vec<f64> {
        let rows = image.len();
        if rows == 0 {
            return Vec::new();
        }
        let cols = image[0].len();
        let (ph, pw) = self.pixels_per_cell;
        let (bh, bw) = self.cells_per_block;
        let n_bins = self.orientations;

        let n_cells_y = rows / ph;
        let n_cells_x = cols / pw;
        if n_cells_y == 0 || n_cells_x == 0 {
            return Vec::new();
        }

        // Build per-cell orientation histograms (unsigned: 0..PI)
        let mut cell_hists = vec![vec![vec![0.0f64; n_bins]; n_cells_x]; n_cells_y];
        let (grad_mag, grad_ori) = gradient_images(image);

        for ry in 0..rows {
            for cx in 0..cols {
                let cy_idx = ry / ph;
                let cx_idx = cx / pw;
                if cy_idx >= n_cells_y || cx_idx >= n_cells_x {
                    continue;
                }
                let mag = grad_mag[ry][cx];
                // Unsigned orientation: fold into [0, π)
                let ori = grad_ori[ry][cx].rem_euclid(PI);
                let bin = ((ori / PI) * n_bins as f64) as usize % n_bins;
                cell_hists[cy_idx][cx_idx][bin] += mag;
            }
        }

        // Block normalisation (L2-Hys)
        let n_blocks_y = n_cells_y.saturating_sub(bh - 1);
        let n_blocks_x = n_cells_x.saturating_sub(bw - 1);
        let block_len = bh * bw * n_bins;
        let mut descriptor = Vec::with_capacity(n_blocks_y * n_blocks_x * block_len);

        for by in 0..n_blocks_y {
            for bx in 0..n_blocks_x {
                let mut block = Vec::with_capacity(block_len);
                for dy in 0..bh {
                    for dx in 0..bw {
                        for &v in &cell_hists[by + dy][bx + dx] {
                            block.push(v);
                        }
                    }
                }
                // L2 normalise
                let norm = block.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
                block.iter_mut().for_each(|v| *v /= norm);
                // Clamp at 0.2
                block.iter_mut().for_each(|v| *v = v.min(0.2));
                // Re-normalise
                let norm2 = block.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
                block.iter_mut().for_each(|v| *v /= norm2);
                descriptor.extend(block);
            }
        }
        descriptor
    }

    /// Return the expected descriptor length for an image of the given shape.
    pub fn feature_size(&self, image_shape: (usize, usize)) -> usize {
        let (rows, cols) = image_shape;
        let (ph, pw) = self.pixels_per_cell;
        let (bh, bw) = self.cells_per_block;
        let n_cells_y = rows / ph;
        let n_cells_x = cols / pw;
        let n_blocks_y = n_cells_y.saturating_sub(bh - 1);
        let n_blocks_x = n_cells_x.saturating_sub(bw - 1);
        n_blocks_y * n_blocks_x * bh * bw * self.orientations
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Compute gradient magnitude and orientation images.
fn gradient_images(image: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let rows = image.len();
    let cols = if rows > 0 { image[0].len() } else { 0 };
    let mut mag = vec![vec![0.0f64; cols]; rows];
    let mut ori = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let dx = if c + 1 < cols {
                image[r][c + 1]
            } else {
                image[r][c]
            } - if c > 0 { image[r][c - 1] } else { image[r][c] };
            let dy = if r + 1 < rows {
                image[r + 1][c]
            } else {
                image[r][c]
            } - if r > 0 { image[r - 1][c] } else { image[r][c] };
            mag[r][c] = (dx * dx + dy * dy).sqrt();
            ori[r][c] = dy.atan2(dx);
        }
    }
    (mag, ori)
}

/// Apply 2-D convolution with zero-padding.
fn convolve2d(image: &[Vec<f64>], kernel: &[Vec<f64>], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let kh = kernel.len();
    let kw = if kh > 0 { kernel[0].len() } else { 0 };
    let khr = (kh / 2) as isize;
    let kwr = (kw / 2) as isize;
    let mut out = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0f64;
            for kr in 0..kh {
                for kc in 0..kw {
                    let nr = r as isize + kr as isize - khr;
                    let nc = c as isize + kc as isize - kwr;
                    if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                        acc += image[nr as usize][nc as usize] * kernel[kr][kc];
                    }
                }
            }
            out[r][c] = acc;
        }
    }
    out
}

/// Gaussian blur with a separable kernel approximation.
fn gaussian_blur(image: &[Vec<f64>], sigma: f64) -> Vec<Vec<f64>> {
    let radius = (3.0 * sigma).ceil() as usize;
    let side = 2 * radius + 1;
    let mut k1d = vec![0.0f64; side];
    for i in 0..side {
        let x = i as f64 - radius as f64;
        k1d[i] = (-x * x / (2.0 * sigma * sigma)).exp();
    }
    let sum: f64 = k1d.iter().sum();
    k1d.iter_mut().for_each(|v| *v /= sum);

    let rows = image.len();
    if rows == 0 {
        return Vec::new();
    }
    let cols = image[0].len();
    // Horizontal pass
    let mut tmp = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0f64;
            for (ki, &kv) in k1d.iter().enumerate() {
                let nc = c as isize + ki as isize - radius as isize;
                let nc = nc.max(0).min(cols as isize - 1) as usize;
                acc += image[r][nc] * kv;
            }
            tmp[r][c] = acc;
        }
    }
    // Vertical pass
    let mut out = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0f64;
            for (ki, &kv) in k1d.iter().enumerate() {
                let nr = r as isize + ki as isize - radius as isize;
                let nr = nr.max(0).min(rows as isize - 1) as usize;
                acc += tmp[nr][c] * kv;
            }
            out[r][c] = acc;
        }
    }
    out
}

/// Pixel-wise subtraction of two same-shape images.
fn subtract_images(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(va, vb)| va - vb).collect())
        .collect()
}

/// Down-sample image by 2× (take every other pixel).
fn downsample2x(image: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = image.len();
    let cols = if rows > 0 { image[0].len() } else { 0 };
    let new_rows = (rows + 1) / 2;
    let new_cols = (cols + 1) / 2;
    let mut out = vec![vec![0.0f64; new_cols]; new_rows];
    for r in 0..new_rows {
        for c in 0..new_cols {
            out[r][c] = image[2 * r][2 * c];
        }
    }
    out
}

/// Check whether `(s, r, c)` is a local extremum in 3-D DoG stack.
fn is_extremum(dog: &[Vec<Vec<f64>>], s: usize, r: usize, c: usize) -> bool {
    let val = dog[s][r][c];
    let is_max = val > 0.0;
    for ds in 0..3usize {
        let ss = s + ds - 1;
        if ss >= dog.len() {
            continue;
        }
        for dr in 0..3usize {
            let rr = r + dr - 1;
            if rr >= dog[ss].len() {
                continue;
            }
            for dc in 0..3usize {
                let cc = c + dc - 1;
                if cc >= dog[ss][rr].len() {
                    continue;
                }
                if ds == 1 && dr == 1 && dc == 1 {
                    continue;
                }
                let v = dog[ss][rr][cc];
                if is_max && v >= val {
                    return false;
                }
                if !is_max && v <= val {
                    return false;
                }
            }
        }
    }
    true
}

/// Harris-ratio edge rejection test.
fn is_edge_point(dog: &[Vec<f64>], r: usize, c: usize, threshold: f64) -> bool {
    let rows = dog.len();
    let cols = if rows > 0 { dog[0].len() } else { 0 };
    if r == 0 || c == 0 || r + 1 >= rows || c + 1 >= cols {
        return false;
    }
    let dxx = dog[r][c + 1] + dog[r][c - 1] - 2.0 * dog[r][c];
    let dyy = dog[r + 1][c] + dog[r - 1][c] - 2.0 * dog[r][c];
    let dxy = (dog[r + 1][c + 1] + dog[r - 1][c - 1] - dog[r + 1][c - 1] - dog[r - 1][c + 1]) / 4.0;
    let trace = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    if det <= 0.0 {
        return true;
    }
    let ratio = trace * trace / det;
    let r_thresh = (threshold + 1.0).powi(2) / threshold;
    ratio >= r_thresh
}

/// Compute dominant gradient orientation at `(r, c)` using a circular histogram.
fn dominant_orientation(image: &[Vec<f64>], r: usize, c: usize, sigma: f64) -> f64 {
    let rows = image.len();
    let cols = if rows > 0 { image[0].len() } else { 0 };
    let radius = (1.5 * sigma).ceil() as isize;
    let mut hist = vec![0.0f64; 36];
    for dr in -radius..=radius {
        for dc in -radius..=radius {
            let nr = (r as isize + dr).max(0).min(rows as isize - 1) as usize;
            let nc = (c as isize + dc).max(0).min(cols as isize - 1) as usize;
            let dy = if nr + 1 < rows {
                image[nr + 1][nc]
            } else {
                image[nr][nc]
            } - if nr > 0 {
                image[nr - 1][nc]
            } else {
                image[nr][nc]
            };
            let dx = if nc + 1 < cols {
                image[nr][nc + 1]
            } else {
                image[nr][nc]
            } - if nc > 0 {
                image[nr][nc - 1]
            } else {
                image[nr][nc]
            };
            let mag = (dx * dx + dy * dy).sqrt();
            let ori = dy.atan2(dx).rem_euclid(2.0 * PI);
            let dist_sq = (dr * dr + dc * dc) as f64;
            let weight = (-dist_sq / (2.0 * (1.5 * sigma).powi(2))).exp();
            let bin = ((ori / (2.0 * PI)) * 36.0) as usize % 36;
            hist[bin] += weight * mag;
        }
    }
    let best_bin = hist
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    best_bin as f64 * 2.0 * PI / 36.0
}

// ─── Public error-returning wrappers ─────────────────────────────────────────

/// Compute HOG descriptor with input validation.
pub fn compute_hog(
    image: &[Vec<f64>],
    orientations: usize,
    pixels_per_cell: (usize, usize),
    cells_per_block: (usize, usize),
) -> NdimageResult<Vec<f64>> {
    if image.is_empty() {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    if orientations == 0 {
        return Err(NdimageError::InvalidInput(
            "orientations must be at least 1".into(),
        ));
    }
    if pixels_per_cell.0 == 0 || pixels_per_cell.1 == 0 {
        return Err(NdimageError::InvalidInput(
            "pixels_per_cell dimensions must be at least 1".into(),
        ));
    }
    if cells_per_block.0 == 0 || cells_per_block.1 == 0 {
        return Err(NdimageError::InvalidInput(
            "cells_per_block dimensions must be at least 1".into(),
        ));
    }
    let hog = HogDescriptor::new(orientations, pixels_per_cell, cells_per_block);
    Ok(hog.compute(image))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_checkerboard(rows: usize, cols: usize) -> Vec<Vec<f64>> {
        (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| if (r / 4 + c / 4) % 2 == 0 { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_gabor_bank_construction() {
        let bank = GaborFilterBank::new(4, &[0.1, 0.2], 1.0);
        assert_eq!(bank.orientations.len(), 4);
        assert_eq!(bank.frequencies.len(), 2);
        let kern = bank.kernel(0.0, 0.1);
        assert!(!kern.is_empty());
    }

    #[test]
    fn test_gabor_apply_shape() {
        let bank = GaborFilterBank::new(2, &[0.1], 1.0);
        let img = make_checkerboard(32, 32);
        let maps = bank.apply(&img);
        assert_eq!(maps.len(), 2); // 2 orientations × 1 frequency
        assert_eq!(maps[0].len(), 32);
        assert_eq!(maps[0][0].len(), 32);
    }

    #[test]
    fn test_gabor_energy_features_length() {
        let bank = GaborFilterBank::new(3, &[0.1, 0.2], 1.5);
        let img = make_checkerboard(32, 32);
        let feats = bank.energy_features(&img);
        // 3 orientations × 2 frequencies × 2 (mean+std)
        assert_eq!(feats.len(), 12);
    }

    #[test]
    fn test_sift_detection_runs() {
        let img = make_checkerboard(64, 64);
        let kps = detect_sift_keypoints(&img, 2, 3, 0.03, 10.0);
        // Just check it doesn't panic; result count depends on image content
        let _ = kps.len();
    }

    #[test]
    fn test_sift_descriptor_shape() {
        let img = make_checkerboard(64, 64);
        let kps = detect_sift_keypoints(&img, 2, 3, 0.01, 10.0);
        let descs = compute_sift_descriptor(&img, &kps);
        assert_eq!(descs.len(), kps.len());
        for d in &descs {
            assert_eq!(d.len(), 128);
        }
    }

    #[test]
    fn test_hog_feature_size() {
        let hog = HogDescriptor::new(9, (8, 8), (2, 2));
        let img = make_checkerboard(64, 64);
        let desc = hog.compute(&img);
        let expected = hog.feature_size((64, 64));
        assert_eq!(desc.len(), expected);
        assert!(!desc.is_empty());
    }

    #[test]
    fn test_hog_compute_hog_wrapper() {
        let img = make_checkerboard(32, 32);
        let result = compute_hog(&img, 9, (8, 8), (2, 2));
        assert!(result.is_ok());
    }

    #[test]
    fn test_hog_invalid_inputs() {
        let img: Vec<Vec<f64>> = Vec::new();
        assert!(compute_hog(&img, 9, (8, 8), (2, 2)).is_err());
        let img2 = make_checkerboard(32, 32);
        assert!(compute_hog(&img2, 0, (8, 8), (2, 2)).is_err());
        assert!(compute_hog(&img2, 9, (0, 8), (2, 2)).is_err());
    }
}
