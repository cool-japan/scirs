//! Texture-Based Image Segmentation
//!
//! This module provides algorithms that segment images based on their
//! textural properties:
//!
//! - **Gabor filter bank** feature maps (multi-frequency, multi-orientation)
//! - **Gabor + k-means** texture segmentation
//! - **LBP-based** (Local Binary Pattern) texture segmentation
//! - **MRF** (Markov Random Field / iterated conditional modes) texture segmentation
//! - Patch-level texture feature extraction
//!
//! # References
//! - Jain, A.K. & Farrokhnia, F. (1991). "Unsupervised texture segmentation using
//!   Gabor filters." Pattern Recognition.
//! - Ojala, T., Pietikainen, M. & Maenpaa, T. (2002). "Multiresolution gray-scale and
//!   rotation invariant texture classification with local binary patterns." IEEE TPAMI.
//! - Besag, J. (1986). "On the statistical analysis of dirty pictures." JRSS-B.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{s, Array1, Array2, Array3};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Gabor filter bank feature map
// ---------------------------------------------------------------------------

/// Compute a Gabor filter bank response volume.
///
/// For every (frequency, orientation) pair a Gabor kernel is constructed
/// and convolved with the image.  Both the real and imaginary response
/// energies (magnitude) are stacked into the output array.
///
/// # Parameters
/// - `image`        – input grayscale image.
/// - `frequencies`  – spatial frequencies in cycles/pixel (e.g. `[0.1, 0.2, 0.3]`).
/// - `orientations` – filter orientations in **radians** (e.g. 4 evenly-spaced from 0 to π).
///
/// # Returns
/// `Array3<f64>` with shape `(rows, cols, n_frequencies * n_orientations)`.
/// Each channel is the magnitude of the complex Gabor response at one
/// (frequency, orientation) pair.
///
/// # Errors
/// Returns `NdimageError::InvalidInput` for an empty image or empty parameter lists.
pub fn gabor_feature_map(
    image: &Array2<f64>,
    frequencies: &[f64],
    orientations: &[f64],
) -> NdimageResult<Array3<f64>> {
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    if frequencies.is_empty() || orientations.is_empty() {
        return Err(NdimageError::InvalidInput(
            "frequencies and orientations must be non-empty".into(),
        ));
    }

    let n_channels = frequencies.len() * orientations.len();
    let mut out = Array3::<f64>::zeros((rows, cols, n_channels));

    let mut ch = 0;
    for &freq in frequencies {
        for &theta in orientations {
            let response = apply_gabor_kernel(image, freq, theta)?;
            for r in 0..rows {
                for c in 0..cols {
                    out[[r, c, ch]] = response[[r, c]];
                }
            }
            ch += 1;
        }
    }

    Ok(out)
}

/// Apply a single Gabor filter and return the response magnitude.
///
/// The kernel size is chosen as `2 * ceil(3 * sigma) + 1` where
/// `sigma = 1.0 / (2.0 * PI * frequency)`.
fn apply_gabor_kernel(
    image: &Array2<f64>,
    frequency: f64,
    theta: f64,
) -> NdimageResult<Array2<f64>> {
    let (rows, cols) = image.dim();

    let sigma = if frequency > 1e-12 {
        1.0 / (2.0 * PI * frequency)
    } else {
        return Err(NdimageError::InvalidInput(
            "Gabor frequency must be positive".into(),
        ));
    };
    let sigma_x = sigma;
    let sigma_y = sigma;

    let half = (3.0 * sigma.max(sigma_x)).ceil() as usize;
    let ksize = 2 * half + 1;

    // Build real & imaginary kernel
    let mut kernel_real = Array2::<f64>::zeros((ksize, ksize));
    let mut kernel_imag = Array2::<f64>::zeros((ksize, ksize));

    let cos_t = theta.cos();
    let sin_t = theta.sin();

    for ky in 0..ksize {
        for kx in 0..ksize {
            let y = ky as f64 - half as f64;
            let x = kx as f64 - half as f64;

            // Rotate
            let x_rot = x * cos_t + y * sin_t;
            let y_rot = -x * sin_t + y * cos_t;

            let gauss = (-0.5
                * (x_rot * x_rot / (sigma_x * sigma_x) + y_rot * y_rot / (sigma_y * sigma_y)))
                .exp();

            let phase = 2.0 * PI * frequency * x_rot;
            kernel_real[[ky, kx]] = gauss * phase.cos();
            kernel_imag[[ky, kx]] = gauss * phase.sin();
        }
    }

    // DC-balance kernels: subtract mean so a uniform image gives zero response
    // for interior pixels when using reflect padding.
    let n_elems = (ksize * ksize) as f64;
    let kr_mean = kernel_real.iter().sum::<f64>() / n_elems;
    let ki_mean = kernel_imag.iter().sum::<f64>() / n_elems;
    kernel_real.iter_mut().for_each(|v| *v -= kr_mean);
    kernel_imag.iter_mut().for_each(|v| *v -= ki_mean);

    // Convolve with reflect padding so that uniform images give near-zero
    // response everywhere (zero-padding would cause non-zero border artefacts
    // for a DC-balanced kernel).
    let resp_real = convolve_reflect(image, &kernel_real)?;
    let resp_imag = convolve_reflect(image, &kernel_imag)?;

    // Magnitude
    let mut magnitude = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            magnitude[[r, c]] = (resp_real[[r, c]].powi(2) + resp_imag[[r, c]].powi(2)).sqrt();
        }
    }

    Ok(magnitude)
}

/// 2-D convolution with zero-padding (same output size).
fn convolve_same(image: &Array2<f64>, kernel: &Array2<f64>) -> NdimageResult<Array2<f64>> {
    let (ih, iw) = image.dim();
    let (kh, kw) = kernel.dim();
    let ph = kh / 2;
    let pw = kw / 2;

    let mut out = Array2::<f64>::zeros((ih, iw));

    for r in 0..ih {
        for c in 0..iw {
            let mut acc = 0.0;
            for kr in 0..kh {
                let ir = r as i64 + kr as i64 - ph as i64;
                if ir < 0 || ir >= ih as i64 {
                    continue;
                }
                for kc in 0..kw {
                    let ic = c as i64 + kc as i64 - pw as i64;
                    if ic < 0 || ic >= iw as i64 {
                        continue;
                    }
                    acc += image[[ir as usize, ic as usize]] * kernel[[kr, kc]];
                }
            }
            out[[r, c]] = acc;
        }
    }
    Ok(out)
}

/// 2-D convolution with reflect-padding (same output size).
///
/// For a DC-balanced kernel (mean = 0), a uniform input image produces a
/// near-zero output everywhere, including border pixels—unlike zero-padding
/// which introduces non-zero responses at the borders.
fn convolve_reflect(image: &Array2<f64>, kernel: &Array2<f64>) -> NdimageResult<Array2<f64>> {
    let (ih, iw) = image.dim();
    let (kh, kw) = kernel.dim();
    let ph = kh / 2;
    let pw = kw / 2;

    // Reflect-clamp: mirrors the coordinate at the boundaries
    let reflect = |idx: i64, size: i64| -> usize {
        if size <= 1 {
            return 0;
        }
        let mut i = idx;
        // Bring into range via periodic reflect
        let period = 2 * (size - 1);
        // Reduce modulo period
        i = ((i % period) + period) % period;
        if i >= size {
            i = period - i;
        }
        i as usize
    };

    let mut out = Array2::<f64>::zeros((ih, iw));

    for r in 0..ih {
        for c in 0..iw {
            let mut acc = 0.0;
            for kr in 0..kh {
                let ir = reflect(r as i64 + kr as i64 - ph as i64, ih as i64);
                for kc in 0..kw {
                    let ic = reflect(c as i64 + kc as i64 - pw as i64, iw as i64);
                    acc += image[[ir, ic]] * kernel[[kr, kc]];
                }
            }
            out[[r, c]] = acc;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// k-means clustering (simple Lloyd's algorithm)
// ---------------------------------------------------------------------------

/// Run k-means on `data` (shape: n_samples × n_features).
///
/// Returns `(labels, centroids)` after convergence or `max_iter` iterations.
fn kmeans(
    data: &[Vec<f64>],
    k: usize,
    max_iter: usize,
) -> NdimageResult<(Vec<usize>, Vec<Vec<f64>>)> {
    if data.is_empty() || k == 0 {
        return Err(NdimageError::InvalidInput(
            "k-means: data must be non-empty and k >= 1".into(),
        ));
    }
    let n = data.len();
    let d = data[0].len();
    let k_actual = k.min(n);

    // Initialise: pick the first k_actual data points as centroids
    let mut centroids: Vec<Vec<f64>> = (0..k_actual).map(|i| data[i].clone()).collect();
    let mut labels = vec![0usize; n];

    for _iter in 0..max_iter {
        let mut changed = false;

        // Assignment step
        for (i, sample) in data.iter().enumerate() {
            let best = nearest_centroid(sample, &centroids);
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step
        let mut sums = vec![vec![0.0f64; d]; k_actual];
        let mut counts = vec![0usize; k_actual];
        for (i, sample) in data.iter().enumerate() {
            let lbl = labels[i];
            counts[lbl] += 1;
            for dim in 0..d {
                sums[lbl][dim] += sample[dim];
            }
        }
        for k_idx in 0..k_actual {
            if counts[k_idx] > 0 {
                let cnt = counts[k_idx] as f64;
                for dim in 0..d {
                    centroids[k_idx][dim] = sums[k_idx][dim] / cnt;
                }
            }
        }
    }

    Ok((labels, centroids))
}

fn nearest_centroid(sample: &[f64], centroids: &[Vec<f64>]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;
    for (idx, centroid) in centroids.iter().enumerate() {
        let dist: f64 = sample
            .iter()
            .zip(centroid.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// Gabor + k-means segmentation
// ---------------------------------------------------------------------------

/// Segment an image by Gabor texture features clustered with k-means.
///
/// # Parameters
/// - `image`        – input grayscale image.
/// - `gabor_params` – `(frequencies, orientations)` slice pair.
/// - `n_clusters`   – number of texture classes.
///
/// # Returns
/// Label image of shape `(rows, cols)` with cluster IDs in `[0, n_clusters)`.
///
/// # Errors
/// See [`gabor_feature_map`] for input constraints.
pub fn texture_segment_kmeans(
    image: &Array2<f64>,
    gabor_params: (&[f64], &[f64]),
    n_clusters: usize,
) -> NdimageResult<Array2<usize>> {
    let (rows, cols) = image.dim();
    let (frequencies, orientations) = gabor_params;

    if n_clusters == 0 {
        return Err(NdimageError::InvalidInput(
            "n_clusters must be at least 1".into(),
        ));
    }

    let feature_map = gabor_feature_map(image, frequencies, orientations)?;
    let n_ch = feature_map.dim().2;

    // Flatten to (n_pixels, n_features)
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let mut feat = Vec::with_capacity(n_ch);
            for ch in 0..n_ch {
                feat.push(feature_map[[r, c, ch]]);
            }
            data.push(feat);
        }
    }

    let (labels, _) = kmeans(&data, n_clusters, 100)?;

    let mut label_image = Array2::<usize>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            label_image[[r, c]] = labels[r * cols + c];
        }
    }

    Ok(label_image)
}

// ---------------------------------------------------------------------------
// LBP-based segmentation
// ---------------------------------------------------------------------------

/// Compute the rotation-invariant LBP code for one pixel at `(row, col)`.
///
/// Uses `n_points` sampling points on a circle of the given `radius`.
fn lbp_code_at(image: &Array2<f64>, row: usize, col: usize, radius: f64, n_points: usize) -> u64 {
    let (rows, cols) = image.dim();
    let center = image[[row, col]];
    let mut code = 0u64;

    for p in 0..n_points {
        let angle = 2.0 * PI * p as f64 / n_points as f64;
        let sample_r = row as f64 - radius * angle.sin();
        let sample_c = col as f64 + radius * angle.cos();

        // Bilinear interpolation
        let r0 = sample_r.floor() as i64;
        let c0 = sample_c.floor() as i64;
        let fr = sample_r - r0 as f64;
        let fc = sample_c - c0 as f64;

        let clamp_r = |r: i64| r.max(0).min(rows as i64 - 1) as usize;
        let clamp_c = |c: i64| c.max(0).min(cols as i64 - 1) as usize;

        let v00 = image[[clamp_r(r0), clamp_c(c0)]];
        let v01 = image[[clamp_r(r0), clamp_c(c0 + 1)]];
        let v10 = image[[clamp_r(r0 + 1), clamp_c(c0)]];
        let v11 = image[[clamp_r(r0 + 1), clamp_c(c0 + 1)]];

        let val = (1.0 - fr) * (1.0 - fc) * v00
            + (1.0 - fr) * fc * v01
            + fr * (1.0 - fc) * v10
            + fr * fc * v11;

        if val >= center {
            code |= 1 << p;
        }
    }

    // Rotation-invariant: take minimum of all bit-rotations
    let mut min_code = code;
    let mask = if n_points < 64 {
        (1u64 << n_points) - 1
    } else {
        u64::MAX
    };
    let mut rotated = code;
    for _ in 1..n_points {
        rotated = ((rotated >> 1) | ((rotated & 1) << (n_points - 1))) & mask;
        if rotated < min_code {
            min_code = rotated;
        }
    }
    min_code
}

/// Segment an image by LBP texture features clustered with k-means.
///
/// # Parameters
/// - `image`      – input grayscale image.
/// - `radius`     – LBP sampling radius in pixels (typical: 1–3).
/// - `n_points`   – number of sampling points on the circle (typical: 8).
/// - `n_clusters` – number of texture classes.
///
/// # Returns
/// Label image of shape `(rows, cols)`.
///
/// # Errors
/// Returns `NdimageError::InvalidInput` for degenerate inputs.
pub fn lbp_segment(
    image: &Array2<f64>,
    radius: f64,
    n_points: usize,
    n_clusters: usize,
) -> NdimageResult<Array2<usize>> {
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    if n_points == 0 {
        return Err(NdimageError::InvalidInput(
            "n_points must be at least 1".into(),
        ));
    }
    if n_clusters == 0 {
        return Err(NdimageError::InvalidInput(
            "n_clusters must be at least 1".into(),
        ));
    }
    if radius <= 0.0 {
        return Err(NdimageError::InvalidInput("radius must be positive".into()));
    }

    // Compute LBP map
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let code = lbp_code_at(image, r, c, radius, n_points);
            data.push(vec![code as f64]);
        }
    }

    let (labels, _) = kmeans(&data, n_clusters, 100)?;

    let mut label_image = Array2::<usize>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            label_image[[r, c]] = labels[r * cols + c];
        }
    }

    Ok(label_image)
}

// ---------------------------------------------------------------------------
// MRF (Markov Random Field) texture segmentation via ICM
// ---------------------------------------------------------------------------

/// Segment an image using a Markov Random Field model with Iterated
/// Conditional Modes (ICM) optimization.
///
/// The data term uses pixel intensity; the MRF prior discourages
/// label discontinuities between neighboring pixels (Potts model).
///
/// # Parameters
/// - `image`      – input grayscale image.
/// - `n_clusters` – number of texture classes.
///
/// # Returns
/// Label image of shape `(rows, cols)`.
///
/// # Errors
/// Returns `NdimageError::InvalidInput` for degenerate inputs.
pub fn mrm_segment(image: &Array2<f64>, n_clusters: usize) -> NdimageResult<Array2<usize>> {
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    if n_clusters == 0 {
        return Err(NdimageError::InvalidInput(
            "n_clusters must be at least 1".into(),
        ));
    }

    let k = n_clusters.min(rows * cols);

    // Initialise: quantize intensity to k classes uniformly
    let i_min = image.iter().cloned().fold(f64::INFINITY, f64::min);
    let i_max = image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (i_max - i_min).max(1e-12);

    let mut labels = Array2::<usize>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            labels[[r, c]] =
                (((image[[r, c]] - i_min) / range * k as f64).floor() as usize).min(k - 1);
        }
    }

    // Estimate class means
    let mut means = vec![0.0f64; k];
    let mut counts = vec![0usize; k];
    for r in 0..rows {
        for c in 0..cols {
            let lbl = labels[[r, c]];
            means[lbl] += image[[r, c]];
            counts[lbl] += 1;
        }
    }
    for ki in 0..k {
        if counts[ki] > 0 {
            means[ki] /= counts[ki] as f64;
        }
    }

    let beta = 0.5; // MRF smoothness weight
    let max_iter = 20;

    let neighbors: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    for _iter in 0..max_iter {
        let mut changed = false;
        let old_labels = labels.clone();

        for r in 0..rows {
            for c in 0..cols {
                let pixel = image[[r, c]];

                let best_label = (0..k)
                    .min_by(|&ka, &kb| {
                        let ea = mrf_energy(
                            pixel,
                            means[ka],
                            ka,
                            r,
                            c,
                            &old_labels,
                            &neighbors,
                            beta,
                            rows,
                            cols,
                        );
                        let eb = mrf_energy(
                            pixel,
                            means[kb],
                            kb,
                            r,
                            c,
                            &old_labels,
                            &neighbors,
                            beta,
                            rows,
                            cols,
                        );
                        ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0);

                if labels[[r, c]] != best_label {
                    labels[[r, c]] = best_label;
                    changed = true;
                }
            }
        }

        // Re-estimate class means
        means = vec![0.0f64; k];
        counts = vec![0usize; k];
        for r in 0..rows {
            for c in 0..cols {
                let lbl = labels[[r, c]];
                means[lbl] += image[[r, c]];
                counts[lbl] += 1;
            }
        }
        for ki in 0..k {
            if counts[ki] > 0 {
                means[ki] /= counts[ki] as f64;
            }
        }

        if !changed {
            break;
        }
    }

    Ok(labels)
}

/// Compute the ICM energy for assigning label `label` to pixel `(r, c)`.
fn mrf_energy(
    pixel: f64,
    mean: f64,
    label: usize,
    r: usize,
    c: usize,
    labels: &Array2<usize>,
    neighbors: &[(i32, i32)],
    beta: f64,
    rows: usize,
    cols: usize,
) -> f64 {
    // Data term: squared deviation from class mean
    let data_term = (pixel - mean).powi(2);

    // Prior term: count neighbors with a different label
    let mut prior_term = 0.0;
    for &(dr, dc) in neighbors {
        let nr = r as i64 + dr as i64;
        let nc = c as i64 + dc as i64;
        if nr >= 0 && nr < rows as i64 && nc >= 0 && nc < cols as i64 {
            if labels[[nr as usize, nc as usize]] != label {
                prior_term += beta;
            }
        }
    }

    data_term + prior_term
}

// ---------------------------------------------------------------------------
// Patch-level texture feature extraction
// ---------------------------------------------------------------------------

/// Extract a texture feature vector for a local patch centred at `(y, x)`.
///
/// The feature vector concatenates:
/// 1. Mean and standard deviation of pixel intensity in the patch.
/// 2. Gabor magnitudes at 4 orientations and 2 frequencies (16 values).
/// 3. LBP histogram (8 bins, radius-1 uniform LBP).
///
/// Total feature dimensionality: **26** elements.
///
/// # Parameters
/// - `image`      – input grayscale image.
/// - `y`, `x`     – centre row and column of the patch.
/// - `patch_size` – patch width/height (must be odd and ≥ 3).
///
/// # Errors
/// Returns `NdimageError::InvalidInput` for an invalid patch size or out-of-bounds centre.
pub fn texture_features_patch(
    image: &Array2<f64>,
    y: usize,
    x: usize,
    patch_size: usize,
) -> NdimageResult<Array1<f64>> {
    let (rows, cols) = image.dim();
    if patch_size < 3 || patch_size % 2 == 0 {
        return Err(NdimageError::InvalidInput(
            "patch_size must be odd and at least 3".into(),
        ));
    }
    let half = patch_size / 2;
    if y < half || x < half || y + half >= rows || x + half >= cols {
        return Err(NdimageError::InvalidInput(
            "Patch extends outside image boundaries".into(),
        ));
    }

    let patch = image.slice(s![y - half..=y + half, x - half..=x + half]);

    // 1. Intensity statistics (2 features)
    let n = patch.len() as f64;
    let mean = patch.iter().sum::<f64>() / n;
    let var = patch.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_dev = var.sqrt();

    // 2. Gabor magnitudes at 4 orientations × 4 frequencies (16 features)
    let freqs = [0.05, 0.1, 0.15, 0.2];
    let thetas = [0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0];
    let patch_owned = patch.to_owned();
    let mut gabor_feats = Vec::with_capacity(16);
    for &freq in &freqs {
        for &theta in &thetas {
            let resp = apply_gabor_kernel(&patch_owned, freq, theta)?;
            let mag: f64 = resp.iter().sum::<f64>() / resp.len() as f64;
            gabor_feats.push(mag);
        }
    }

    // 3. LBP histogram (8 bins for rotation-invariant LBP, 8 features)
    let lbp_n_points = 8usize;
    let lbp_radius = 1.0f64;
    let n_lbp_bins = lbp_n_points + 2; // uniform LBP: 0..P+1 + "non-uniform"
    let mut lbp_hist = vec![0.0f64; n_lbp_bins];
    let ph = patch.nrows();
    let pw = patch.ncols();
    let mut lbp_count = 0.0f64;
    for pr in 0..ph {
        for pc in 0..pw {
            let code = lbp_code_at(&patch_owned, pr, pc, lbp_radius, lbp_n_points);
            let bin = (code as usize).min(n_lbp_bins - 1);
            lbp_hist[bin] += 1.0;
            lbp_count += 1.0;
        }
    }
    if lbp_count > 0.0 {
        for v in lbp_hist.iter_mut() {
            *v /= lbp_count;
        }
    }

    // Assemble feature vector: 2 + 16 + (n_lbp_bins) = 2 + 16 + 10 = 28
    let mut features = Vec::with_capacity(2 + 16 + n_lbp_bins);
    features.push(mean);
    features.push(std_dev);
    features.extend_from_slice(&gabor_feats);
    features.extend_from_slice(&lbp_hist);

    Ok(Array1::from_vec(features))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn uniform_image(rows: usize, cols: usize, val: f64) -> Array2<f64> {
        Array2::from_elem((rows, cols), val)
    }

    fn striped_image(rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(_, c)| if c % 4 < 2 { 1.0 } else { 0.0 })
    }

    #[test]
    fn test_gabor_feature_map_shape() {
        let img = striped_image(16, 16);
        let freqs = vec![0.1, 0.2];
        let thetas = vec![0.0, PI / 2.0];
        let feat = gabor_feature_map(&img, &freqs, &thetas).expect("gabor ok");
        assert_eq!(feat.dim(), (16, 16, 4));
    }

    #[test]
    fn test_gabor_feature_map_uniform_image() {
        // A uniform image should give near-zero Gabor response
        let img = uniform_image(12, 12, 0.5);
        let feat = gabor_feature_map(&img, &[0.15], &[0.0]).expect("gabor ok");
        for v in feat.iter() {
            assert!(
                *v < 0.01,
                "Uniform image Gabor response should be ~0, got {v}"
            );
        }
    }

    #[test]
    fn test_texture_segment_kmeans() {
        let img = striped_image(20, 20);
        let freqs = vec![0.1, 0.2];
        let thetas = vec![0.0, PI / 2.0];
        let labels = texture_segment_kmeans(&img, (&freqs, &thetas), 2).expect("segment ok");
        assert_eq!(labels.dim(), (20, 20));
        // All labels should be in [0, 2)
        for &lbl in labels.iter() {
            assert!(lbl < 2);
        }
    }

    #[test]
    fn test_lbp_segment() {
        let img = striped_image(24, 24);
        let labels = lbp_segment(&img, 1.0, 8, 2).expect("lbp ok");
        assert_eq!(labels.dim(), (24, 24));
    }

    #[test]
    fn test_mrm_segment() {
        let img = striped_image(16, 16);
        let labels = mrm_segment(&img, 2).expect("mrm ok");
        assert_eq!(labels.dim(), (16, 16));
        for &lbl in labels.iter() {
            assert!(lbl < 2);
        }
    }

    #[test]
    fn test_texture_features_patch_shape() {
        let img = striped_image(32, 32);
        let feats = texture_features_patch(&img, 10, 10, 7).expect("patch ok");
        // 2 intensity + 16 Gabor + (8+2) LBP = 28
        assert_eq!(feats.len(), 28);
    }

    #[test]
    fn test_texture_features_patch_invalid_patch_size() {
        let img: Array2<f64> = Array2::zeros((32, 32));
        let err = texture_features_patch(&img, 10, 10, 4); // even size
        assert!(err.is_err());
    }

    #[test]
    fn test_texture_features_patch_out_of_bounds() {
        let img: Array2<f64> = Array2::zeros((16, 16));
        // half = 5; y=2 < half -> out of bounds
        let err = texture_features_patch(&img, 2, 8, 11);
        assert!(err.is_err());
    }
}
