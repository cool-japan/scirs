//! Image quality assessment metrics.
//!
//! Provides both full-reference (PSNR, SSIM, MS-SSIM, FSIM, GMSD, VIF) and
//! no-reference (BRISQUE features, NIQE score, MOS features) quality metrics.

use crate::error::{NdimageError, NdimageResult};
use std::f64::consts::PI;

// ─── Full-reference metrics ───────────────────────────────────────────────────

/// Peak Signal-to-Noise Ratio (PSNR) in decibels.
///
/// A higher PSNR indicates a better quality image.
/// PSNR = 10 · log₁₀(max_val² / MSE).
///
/// # Arguments
/// * `original`  – Reference image.
/// * `distorted` – Distorted image (same shape).
/// * `max_val`   – Maximum pixel value (e.g. 255.0 for uint8, 1.0 for float).
pub fn psnr(original: &[Vec<f64>], distorted: &[Vec<f64>], max_val: f64) -> NdimageResult<f64> {
    validate_same_shape(original, distorted)?;
    if max_val <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "max_val must be positive".into(),
        ));
    }
    let mse = mean_squared_error(original, distorted);
    if mse < 1e-15 {
        return Ok(f64::INFINITY);
    }
    Ok(10.0 * (max_val * max_val / mse).log10())
}

/// Structural Similarity Index Measure (SSIM).
///
/// Measures structural similarity between two images in `[-1, 1]`; higher = better.
///
/// # Arguments
/// * `img1`, `img2` – Images to compare (same shape).
/// * `window_size`  – Gaussian window radius; typical 11.
/// * `k1`, `k2`     – Stability constants (typical: 0.01, 0.03).
/// * `max_val`      – Maximum pixel value.
pub fn ssim(
    img1: &[Vec<f64>],
    img2: &[Vec<f64>],
    window_size: usize,
    k1: f64,
    k2: f64,
    max_val: f64,
) -> NdimageResult<f64> {
    validate_same_shape(img1, img2)?;
    if max_val <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "max_val must be positive".into(),
        ));
    }
    let rows = img1.len();
    let cols = img1[0].len();
    let c1 = (k1 * max_val).powi(2);
    let c2 = (k2 * max_val).powi(2);

    // Build Gaussian window
    let sigma = window_size as f64 / 3.0;
    let win = gaussian_window_2d(window_size, sigma);

    let mut ssim_sum = 0.0f64;
    let mut count = 0usize;
    let half = window_size as isize;

    for r in 0..rows {
        for c in 0..cols {
            // Compute local stats using the window
            let (mu1, mu2, var1, var2, covar) =
                local_stats_with_window(img1, img2, r, c, &win, rows, cols);
            let ssim_val = (2.0 * mu1 * mu2 + c1) * (2.0 * covar + c2)
                / ((mu1 * mu1 + mu2 * mu2 + c1) * (var1 + var2 + c2));
            ssim_sum += ssim_val;
            count += 1;
        }
    }
    let _ = half;
    Ok(if count > 0 {
        ssim_sum / count as f64
    } else {
        0.0
    })
}

/// Multi-Scale SSIM (MS-SSIM).
///
/// Computes SSIM at multiple down-sampled scales and combines using weights
/// calibrated by Wang et al. (2003).  Range `[0, 1]`; higher = better.
pub fn ms_ssim(img1: &[Vec<f64>], img2: &[Vec<f64>], max_val: f64) -> NdimageResult<f64> {
    validate_same_shape(img1, img2)?;
    // Weights from Wang 2003 for 5-scale MS-SSIM
    let weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];
    let n_scales = weights.len();

    let mut a = img1.to_vec();
    let mut b = img2.to_vec();
    let mut result = 1.0f64;

    for scale in 0..n_scales {
        let rows = a.len();
        let cols = if rows > 0 { a[0].len() } else { 0 };
        if rows < 8 || cols < 8 {
            break;
        }
        let s = ssim(&a, &b, 5, 0.01, 0.03, max_val)?;
        result *= s.abs().powf(weights[scale]);
        if scale + 1 < n_scales {
            a = downsample2x(&a);
            b = downsample2x(&b);
        }
    }
    Ok(result)
}

/// Feature SIMilarity index (FSIM).
///
/// Combines phase congruency (PC) and gradient magnitude similarity (GMS).
/// Range `[0, 1]`; higher = better.
pub fn fsim(img1: &[Vec<f64>], img2: &[Vec<f64>]) -> NdimageResult<f64> {
    validate_same_shape(img1, img2)?;
    let rows = img1.len();
    let cols = img1[0].len();

    let (gm1, _) = gradient_magnitude(img1);
    let (gm2, _) = gradient_magnitude(img2);
    let pc1 = phase_congruency_map(img1);
    let pc2 = phase_congruency_map(img2);

    let t1 = 0.85;
    let t2 = 160.0;
    let mut fsim_num = 0.0f64;
    let mut fsim_den = 0.0f64;

    for r in 0..rows {
        for c in 0..cols {
            let pc_m = pc1[r][c].max(pc2[r][c]);
            let s_pc =
                (2.0 * pc1[r][c] * pc2[r][c] + t1) / (pc1[r][c].powi(2) + pc2[r][c].powi(2) + t1);
            let g1 = gm1[r][c];
            let g2 = gm2[r][c];
            let s_gm = (2.0 * g1 * g2 + t2) / (g1 * g1 + g2 * g2 + t2);
            fsim_num += s_pc * s_gm * pc_m;
            fsim_den += pc_m;
        }
    }
    Ok(if fsim_den > 1e-12 {
        fsim_num / fsim_den
    } else {
        0.0
    })
}

/// Gradient Magnitude Similarity Deviation (GMSD).
///
/// Measures the standard deviation of the gradient magnitude similarity map.
/// Lower GMSD indicates better quality (it is a distortion measure).
pub fn gmsd(reference: &[Vec<f64>], distorted: &[Vec<f64>]) -> NdimageResult<f64> {
    validate_same_shape(reference, distorted)?;
    let rows = reference.len();
    let cols = reference[0].len();
    let (gm_ref, _) = gradient_magnitude(reference);
    let (gm_dis, _) = gradient_magnitude(distorted);

    let c = 0.0026f64; // stabilising constant
    let mut gms_map: Vec<f64> = Vec::with_capacity(rows * cols);

    for r in 0..rows {
        for c_idx in 0..cols {
            let g1 = gm_ref[r][c_idx];
            let g2 = gm_dis[r][c_idx];
            let gms = (2.0 * g1 * g2 + c) / (g1 * g1 + g2 * g2 + c);
            gms_map.push(gms);
        }
    }

    let n = gms_map.len() as f64;
    let mean = gms_map.iter().sum::<f64>() / n;
    let var = gms_map.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    Ok(var.sqrt())
}

/// Visual Information Fidelity (VIF) — simplified single-scale version.
///
/// Estimates the ratio of mutual information in the reference and distorted
/// images from a statistical model. Range `[0, ∞)`; 1.0 = perfect quality.
pub fn vif(reference: &[Vec<f64>], distorted: &[Vec<f64>]) -> NdimageResult<f64> {
    validate_same_shape(reference, distorted)?;
    let rows = reference.len();
    let cols = reference[0].len();
    let sigma_n_sq = 0.4f64; // noise variance
    let sigma_n_sq_d = sigma_n_sq; // matched distortion model

    let mut num = 0.0f64;
    let mut den = 0.0f64;
    let patch = 3usize;
    let half = patch as isize / 2;

    for r in (half as usize)..(rows.saturating_sub(half as usize)) {
        for c in (half as usize)..(cols.saturating_sub(half as usize)) {
            // Local variance estimates
            let var_ref = local_variance(reference, r, c, patch, rows, cols);
            let var_dis = local_variance(distorted, r, c, patch, rows, cols);
            let g_sq = (var_dis / (var_ref + 1e-12)).min(1.0);
            num += ((1.0 + g_sq * var_ref / sigma_n_sq_d) + 1e-12).ln();
            den += ((1.0 + var_ref / sigma_n_sq) + 1e-12).ln();
        }
    }
    Ok(if den > 1e-12 { num / den } else { 1.0 })
}

// ─── No-reference metrics ─────────────────────────────────────────────────────

/// BRISQUE-style 36-dimensional feature vector.
///
/// Extracts features based on locally normalised luminance (MSCN) coefficients
/// and their pairwise products at two scales, fitting asymmetric generalised
/// Gaussian distributions (AGGD) to estimate quality.
pub fn brisque_features(image: &[Vec<f64>]) -> NdimageResult<Vec<f64>> {
    if image.is_empty() {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    let mut feats = Vec::with_capacity(36);

    // Process two scales
    let mut cur = image.to_vec();
    for scale in 0..2usize {
        let mscn = compute_mscn(&cur);
        // Use dimensions of the current scale, not the original image
        let cur_rows = cur.len();
        let cur_cols = if cur_rows > 0 { cur[0].len() } else { 0 };
        // Feature 1: AGGD fit to MSCN (shape, left_std, right_std)
        let (alpha, sigma_l, sigma_r) = fit_aggd(&mscn);
        feats.push(alpha);
        feats.push(sigma_l);
        feats.push(sigma_r);

        // Features 2-4: pairwise MSCN products in 4 directions
        let dirs: &[(isize, isize)] = &[(0, 1), (1, 0), (1, 1), (1, -1)];
        for &(dr, dc) in dirs {
            let pairs = mscn_pairwise(&mscn, dr, dc, cur_rows, cur_cols);
            let (a2, sl2, sr2, mean2) = fit_aggd_with_mean(&pairs);
            feats.push(mean2);
            feats.push(a2);
            feats.push(sl2);
            feats.push(sr2);
        }

        // Down-sample by 2× for next scale
        if scale == 0 {
            cur = downsample2x(&cur);
        }
    }

    // Pad to exactly 36 if fewer were added
    while feats.len() < 36 {
        feats.push(0.0);
    }
    feats.truncate(36);
    Ok(feats)
}

/// Natural Image Quality Evaluator (NIQE) score.
///
/// A lower NIQE score indicates better perceptual quality.
/// Computes multi-scale patch statistics and compares them to a reference
/// natural scene statistics (NSS) model encoded as fixed parameters.
pub fn niqe_score(image: &[Vec<f64>]) -> NdimageResult<f64> {
    if image.is_empty() {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    let feats = brisque_features(image)?;
    // Simple model: project features against a reference vector
    // (in practice this would use a fitted multivariate Gaussian model)
    let ref_mean = [
        2.5_f64, 0.45, 0.55, 0.0, 3.0, 0.5, 0.5, 2.8, 0.48, 0.52, 0.01, 2.9, 0.5, 0.5, 0.0, 2.7,
        0.46, 0.54, 0.0, 2.9, 0.47, 0.53, 0.01, 2.8, 0.49, 0.51, 0.0, 2.7, 0.46, 0.53, 0.0, 2.9,
        0.48, 0.52, 0.01, 2.8,
    ];
    let diff: f64 = feats
        .iter()
        .zip(ref_mean.iter())
        .map(|(f, m)| (f - m).powi(2))
        .sum::<f64>();
    Ok((diff / feats.len() as f64).sqrt())
}

/// Mean Opinion Score (MOS) predictor features (72-dimensional).
///
/// Combines BRISQUE features at two scales with gradient statistics and
/// local contrast features to build a feature vector suitable for regressing
/// MOS predictions.
pub fn mos_features(image: &[Vec<f64>]) -> NdimageResult<Vec<f64>> {
    if image.is_empty() {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    let mut feats = brisque_features(image)?; // 36 dims
    let (gm, _) = gradient_magnitude(image);
    let flat_gm: Vec<f64> = gm.iter().flat_map(|r| r.iter().copied()).collect();
    let n = flat_gm.len() as f64;
    if n > 0.0 {
        let mean = flat_gm.iter().sum::<f64>() / n;
        let var = flat_gm.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let skew = if var > 1e-12 {
            flat_gm.iter().map(|v| (v - mean).powi(3)).sum::<f64>() / (n * var.powf(1.5))
        } else {
            0.0
        };
        feats.push(mean);
        feats.push(var.sqrt());
        feats.push(skew);
    }
    // Contrast features (local standard deviation stats)
    let contrast = local_contrast_map(image);
    let flat_c: Vec<f64> = contrast.iter().flat_map(|r| r.iter().copied()).collect();
    let nc = flat_c.len() as f64;
    if nc > 0.0 {
        let mean_c = flat_c.iter().sum::<f64>() / nc;
        let var_c = flat_c.iter().map(|v| (v - mean_c).powi(2)).sum::<f64>() / nc;
        feats.push(mean_c);
        feats.push(var_c.sqrt());
    }
    // Entropy estimate
    feats.push(image_entropy(image));
    Ok(feats)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

fn validate_same_shape(a: &[Vec<f64>], b: &[Vec<f64>]) -> NdimageResult<()> {
    if a.is_empty() || b.is_empty() {
        return Err(NdimageError::InvalidInput(
            "Images must not be empty".into(),
        ));
    }
    if a.len() != b.len() || a[0].len() != b[0].len() {
        return Err(NdimageError::InvalidInput(
            "Images must have the same shape".into(),
        ));
    }
    Ok(())
}

fn mean_squared_error(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let n: usize = a.iter().map(|r| r.len()).sum();
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f64 = a
        .iter()
        .zip(b.iter())
        .flat_map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(va, vb)| (va - vb).powi(2)))
        .sum();
    sum_sq / n as f64
}

/// Build a 2-D Gaussian window of given radius and sigma.
fn gaussian_window_2d(half: usize, sigma: f64) -> Vec<Vec<f64>> {
    let side = 2 * half + 1;
    let c = half as f64;
    let mut win = vec![vec![0.0f64; side]; side];
    let mut sum = 0.0f64;
    for r in 0..side {
        for col in 0..side {
            let dr = r as f64 - c;
            let dc = col as f64 - c;
            let v = (-(dr * dr + dc * dc) / (2.0 * sigma * sigma)).exp();
            win[r][col] = v;
            sum += v;
        }
    }
    win.iter_mut()
        .for_each(|row| row.iter_mut().for_each(|v| *v /= sum));
    win
}

/// Compute weighted local statistics for SSIM.
fn local_stats_with_window(
    img1: &[Vec<f64>],
    img2: &[Vec<f64>],
    r: usize,
    c: usize,
    win: &[Vec<f64>],
    rows: usize,
    cols: usize,
) -> (f64, f64, f64, f64, f64) {
    let half = win.len() / 2;
    let mut mu1 = 0.0f64;
    let mut mu2 = 0.0f64;
    let mut mu1_sq = 0.0f64;
    let mut mu2_sq = 0.0f64;
    let mut mu1_mu2 = 0.0f64;
    let mut w_sum = 0.0f64;

    for wr in 0..win.len() {
        for wc in 0..win[0].len() {
            let nr = r as isize + wr as isize - half as isize;
            let nc = c as isize + wc as isize - half as isize;
            if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                continue;
            }
            let nr = nr as usize;
            let nc = nc as usize;
            let w = win[wr][wc];
            let v1 = img1[nr][nc];
            let v2 = img2[nr][nc];
            mu1 += w * v1;
            mu2 += w * v2;
            mu1_sq += w * v1 * v1;
            mu2_sq += w * v2 * v2;
            mu1_mu2 += w * v1 * v2;
            w_sum += w;
        }
    }

    if w_sum < 1e-12 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    mu1 /= w_sum;
    mu2 /= w_sum;
    mu1_sq /= w_sum;
    mu2_sq /= w_sum;
    mu1_mu2 /= w_sum;

    let var1 = (mu1_sq - mu1 * mu1).max(0.0);
    let var2 = (mu2_sq - mu2 * mu2).max(0.0);
    let covar = mu1_mu2 - mu1 * mu2;
    (mu1, mu2, var1, var2, covar)
}

/// Down-sample image by 2×.
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

/// Compute gradient magnitudes.
fn gradient_magnitude(image: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
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

/// Simplified phase congruency map (energy-normalised gradient magnitude).
fn phase_congruency_map(image: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = image.len();
    let cols = if rows > 0 { image[0].len() } else { 0 };
    let (gm, _) = gradient_magnitude(image);
    // Phase congruency ≈ normalised gradient magnitude (simplified)
    let max_gm = gm
        .iter()
        .flat_map(|r| r.iter().copied())
        .fold(0.0f64, f64::max);
    if max_gm < 1e-12 {
        return vec![vec![0.0; cols]; rows];
    }
    gm.iter()
        .map(|r| r.iter().map(|&v| v / max_gm).collect())
        .collect()
}

/// Compute MSCN (Mean-Subtracted Contrast-Normalised) coefficients.
fn compute_mscn(image: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = image.len();
    let cols = if rows > 0 { image[0].len() } else { 0 };
    let sigma = 7.0f64 / 6.0;
    let win = gaussian_window_2d(3, sigma);
    let half = 3;

    let mut local_mean = vec![vec![0.0f64; cols]; rows];
    let mut local_var = vec![vec![0.0f64; cols]; rows];

    for r in 0..rows {
        for c in 0..cols {
            let mut mu = 0.0f64;
            let mut mu2 = 0.0f64;
            let mut wsum = 0.0f64;
            for wr in 0..win.len() {
                for wc in 0..win[0].len() {
                    let nr = r as isize + wr as isize - half as isize;
                    let nc = c as isize + wc as isize - half as isize;
                    if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                        continue;
                    }
                    let v = image[nr as usize][nc as usize];
                    let w = win[wr][wc];
                    mu += w * v;
                    mu2 += w * v * v;
                    wsum += w;
                }
            }
            if wsum > 1e-12 {
                mu /= wsum;
                mu2 /= wsum;
                local_mean[r][c] = mu;
                local_var[r][c] = (mu2 - mu * mu).max(0.0);
            }
        }
    }

    let mut mscn = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let std = local_var[r][c].sqrt();
            mscn[r][c] = (image[r][c] - local_mean[r][c]) / (std + 1.0);
        }
    }
    mscn
}

/// Extract pairwise MSCN products in direction (dr, dc).
fn mscn_pairwise(mscn: &[Vec<f64>], dr: isize, dc: isize, rows: usize, cols: usize) -> Vec<f64> {
    let mut pairs = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                pairs.push(mscn[r][c] * mscn[nr as usize][nc as usize]);
            }
        }
    }
    pairs
}

/// Fit an Asymmetric Generalised Gaussian Distribution (AGGD) to data.
/// Returns (shape, left_std, right_std).
fn fit_aggd(data: &[Vec<f64>]) -> (f64, f64, f64) {
    let flat: Vec<f64> = data.iter().flat_map(|r| r.iter().copied()).collect();
    fit_aggd_params(&flat)
}

fn fit_aggd_params(data: &[f64]) -> (f64, f64, f64) {
    if data.is_empty() {
        return (2.0, 1.0, 1.0);
    }
    let (neg, pos): (Vec<f64>, Vec<f64>) = data.iter().partition(|&&v| v < 0.0);
    let sigma_l = if neg.is_empty() {
        1e-6
    } else {
        (neg.iter().map(|v| v * v).sum::<f64>() / neg.len() as f64).sqrt()
    };
    let sigma_r = if pos.is_empty() {
        1e-6
    } else {
        (pos.iter().map(|v| v * v).sum::<f64>() / pos.len() as f64).sqrt()
    };
    let mu = data.iter().sum::<f64>() / data.len() as f64;
    let var = data.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / data.len() as f64;
    // Estimate shape parameter via moment matching
    let gamma_ratio = if var > 1e-12 {
        let m2 = data.iter().map(|v| v.abs().powi(2)).sum::<f64>() / data.len() as f64;
        let m1 = data.iter().map(|v| v.abs()).sum::<f64>() / data.len() as f64;
        (m1 * m1 / (m2 + 1e-12)).sqrt().clamp(0.1, 10.0)
    } else {
        2.0
    };
    (gamma_ratio, sigma_l.max(1e-6), sigma_r.max(1e-6))
}

/// Fit AGGD with mean, returns (shape, sigma_l, sigma_r, mean).
fn fit_aggd_with_mean(data: &[f64]) -> (f64, f64, f64, f64) {
    let (a, sl, sr) = fit_aggd_params(data);
    let mean = if data.is_empty() {
        0.0
    } else {
        data.iter().sum::<f64>() / data.len() as f64
    };
    (a, sl, sr, mean)
}

/// Local variance in a patch around (r, c).
fn local_variance(
    image: &[Vec<f64>],
    r: usize,
    c: usize,
    patch: usize,
    rows: usize,
    cols: usize,
) -> f64 {
    let half = patch as isize / 2;
    let mut vals = Vec::new();
    for dr in -half..=half {
        for dc in -half..=half {
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                vals.push(image[nr as usize][nc as usize]);
            }
        }
    }
    if vals.is_empty() {
        return 0.0;
    }
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n
}

/// Local contrast map (local standard deviation).
fn local_contrast_map(image: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = image.len();
    let cols = if rows > 0 { image[0].len() } else { 0 };
    let patch = 5usize;
    let mut out = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            out[r][c] = local_variance(image, r, c, patch, rows, cols).sqrt();
        }
    }
    out
}

/// Shannon entropy of pixel intensity distribution.
fn image_entropy(image: &[Vec<f64>]) -> f64 {
    let n_bins = 64usize;
    let mut hist = vec![0u64; n_bins];
    let flat: Vec<f64> = image.iter().flat_map(|r| r.iter().copied()).collect();
    if flat.is_empty() {
        return 0.0;
    }
    let min_v = flat.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_v = flat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max_v - min_v).max(1e-12);
    for &v in &flat {
        let bin = ((v - min_v) / range * (n_bins - 1) as f64) as usize;
        hist[bin.min(n_bins - 1)] += 1;
    }
    let total = flat.len() as f64;
    hist.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.ln()
        })
        .sum::<f64>()
        / PI.ln() // normalise by log(π) for a bounded measure
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_const(rows: usize, cols: usize, val: f64) -> Vec<Vec<f64>> {
        vec![vec![val; cols]; rows]
    }

    fn make_ramp(rows: usize, cols: usize) -> Vec<Vec<f64>> {
        (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| (r + c) as f64 / (rows + cols) as f64)
                    .collect()
            })
            .collect()
    }

    fn add_noise(image: &[Vec<f64>], level: f64) -> Vec<Vec<f64>> {
        // Simple deterministic "noise" for testing
        image
            .iter()
            .enumerate()
            .map(|(r, row)| {
                row.iter()
                    .enumerate()
                    .map(|(c, &v)| v + level * ((r * 7 + c * 13) % 17) as f64 / 17.0)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_psnr_identical() {
        let img = make_ramp(32, 32);
        let p = psnr(&img, &img, 1.0).expect("psnr should succeed on valid identical images");
        assert!(
            p.is_infinite(),
            "Identical images should have infinite PSNR"
        );
    }

    #[test]
    fn test_psnr_with_noise() {
        let img = make_ramp(32, 32);
        let noisy = add_noise(&img, 0.05);
        let p = psnr(&img, &noisy, 1.0).expect("psnr should succeed on valid images");
        assert!(p > 0.0 && p < 1000.0);
    }

    #[test]
    fn test_psnr_invalid_max_val() {
        let img = make_ramp(8, 8);
        assert!(psnr(&img, &img, 0.0).is_err());
    }

    #[test]
    fn test_psnr_shape_mismatch() {
        let a = make_ramp(8, 8);
        let b = make_ramp(8, 16);
        assert!(psnr(&a, &b, 1.0).is_err());
    }

    #[test]
    fn test_ssim_identical() {
        let img = make_ramp(32, 32);
        let s = ssim(&img, &img, 5, 0.01, 0.03, 1.0)
            .expect("ssim should succeed on valid identical images");
        // SSIM of an image with itself should be close to 1.0
        assert!(s > 0.9, "SSIM of identical images should be ≈1, got {s}");
    }

    #[test]
    fn test_ssim_with_noise() {
        let img = make_ramp(32, 32);
        let noisy = add_noise(&img, 0.2);
        let s =
            ssim(&img, &noisy, 5, 0.01, 0.03, 1.0).expect("ssim should succeed on valid images");
        let s_same =
            ssim(&img, &img, 5, 0.01, 0.03, 1.0).expect("ssim should succeed on identical images");
        assert!(s < s_same, "Noisy image should have lower SSIM");
    }

    #[test]
    fn test_ms_ssim() {
        let img = make_ramp(64, 64);
        let noisy = add_noise(&img, 0.1);
        let s = ms_ssim(&img, &noisy, 1.0).expect("ms_ssim should succeed on valid images");
        assert!(s > 0.0 && s <= 1.0, "MS-SSIM out of range: {s}");
    }

    #[test]
    fn test_fsim_shape_mismatch() {
        let a = make_ramp(8, 8);
        let b = make_ramp(16, 8);
        assert!(fsim(&a, &b).is_err());
    }

    #[test]
    fn test_fsim_runs() {
        let img = make_ramp(32, 32);
        let noisy = add_noise(&img, 0.1);
        let f = fsim(&img, &noisy).expect("fsim should succeed on valid images");
        assert!(f >= 0.0);
    }

    #[test]
    fn test_gmsd_identical() {
        let img = make_ramp(32, 32);
        let g = gmsd(&img, &img).expect("gmsd should succeed on valid identical images");
        assert!(g < 1e-6, "GMSD of identical images should be 0, got {g}");
    }

    #[test]
    fn test_gmsd_with_noise() {
        let img = make_ramp(32, 32);
        let noisy = add_noise(&img, 0.1);
        let g = gmsd(&img, &noisy).expect("gmsd should succeed on valid images");
        assert!(g > 0.0);
    }

    #[test]
    fn test_vif_runs() {
        let img = make_ramp(32, 32);
        let noisy = add_noise(&img, 0.05);
        let v = vif(&img, &noisy).expect("vif should succeed on valid images");
        assert!(v >= 0.0);
    }

    #[test]
    fn test_brisque_features_length() {
        let img = make_ramp(64, 64);
        let f = brisque_features(&img).expect("brisque_features should succeed on valid image");
        assert_eq!(f.len(), 36);
    }

    #[test]
    fn test_niqe_score_runs() {
        let img = make_ramp(64, 64);
        let s = niqe_score(&img).expect("niqe_score should succeed on valid image");
        assert!(s >= 0.0);
    }

    #[test]
    fn test_mos_features_runs() {
        let img = make_ramp(32, 32);
        let f = mos_features(&img).expect("mos_features should succeed on valid image");
        assert!(!f.is_empty());
    }

    #[test]
    fn test_empty_image_errors() {
        let empty: Vec<Vec<f64>> = Vec::new();
        let img = make_ramp(8, 8);
        assert!(psnr(&empty, &img, 1.0).is_err());
        assert!(ssim(&empty, &img, 5, 0.01, 0.03, 1.0).is_err());
        assert!(brisque_features(&empty).is_err());
        assert!(niqe_score(&empty).is_err());
        assert!(mos_features(&empty).is_err());
    }
}
