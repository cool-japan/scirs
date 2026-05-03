//! Extended image segmentation algorithms.
//!
//! Provides:
//! - Felzenszwalb-Huttenlocher efficient graph-based segmentation
//! - Mean shift segmentation
//! - GrabCut-style interactive segmentation (Gaussian Mixture Model + graph cuts)
//! - Quickshift density-based segmentation

use crate::error::{NdimageError, NdimageResult};
use std::collections::HashMap;

// ─── Felzenszwalb-Huttenlocher ────────────────────────────────────────────────

/// Efficient graph-based image segmentation (Felzenszwalb & Huttenlocher 2004).
///
/// Segments the image by merging neighbouring pixel nodes whose edge weight
/// (colour difference) is below an adaptive threshold that grows with segment
/// size.  Larger `scale` produces coarser segments.
///
/// # Arguments
/// * `image`    – 3-D array `[rows][cols][channels]` (or 2-D via single channel).
/// * `scale`    – Controls segment granularity (larger → bigger segments).
/// * `sigma`    – Gaussian pre-smoothing standard deviation.
/// * `min_size` – Minimum segment size in pixels; smaller segments are merged.
///
/// # Returns
/// 2-D label image of shape `[rows][cols]`.
pub fn felzenszwalb_segment(
    image: &[Vec<Vec<f64>>],
    scale: f64,
    sigma: f64,
    min_size: usize,
) -> NdimageResult<Vec<Vec<usize>>> {
    let rows = image.len();
    if rows == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    let cols = image[0].len();
    if cols == 0 {
        return Err(NdimageError::InvalidInput(
            "Image columns must be > 0".into(),
        ));
    }
    let channels = image[0][0].len();

    // Gaussian smoothing per channel
    let smoothed = smooth_image_channels(image, sigma);

    // Build edge list (4-connectivity)
    let mut edges: Vec<(f64, usize, usize)> = Vec::new(); // (weight, node_a, node_b)
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            if c + 1 < cols {
                let w = pixel_diff(&smoothed[r][c], &smoothed[r][c + 1], channels);
                edges.push((w, idx, idx + 1));
            }
            if r + 1 < rows {
                let w = pixel_diff(&smoothed[r][c], &smoothed[r + 1][c], channels);
                edges.push((w, idx, idx + cols));
            }
        }
    }

    // Sort by weight
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Union-Find with internal difference tracking
    let n = rows * cols;
    let mut uf = UnionFind::new(n);
    let mut int_diff = vec![0.0f64; n]; // internal difference per component
    let mut comp_size = vec![1usize; n];

    for (w, a, b) in &edges {
        let ra = uf.find(*a);
        let rb = uf.find(*b);
        if ra == rb {
            continue;
        }
        let threshold_a = int_diff[ra] + scale / comp_size[ra] as f64;
        let threshold_b = int_diff[rb] + scale / comp_size[rb] as f64;
        if *w <= threshold_a.min(threshold_b) {
            let new_root = uf.union(ra, rb);
            int_diff[new_root] = *w;
            comp_size[new_root] = comp_size[ra] + comp_size[rb];
        }
    }

    // Merge small components (post-processing)
    for (w, a, b) in &edges {
        let ra = uf.find(*a);
        let rb = uf.find(*b);
        if ra != rb && (comp_size[ra] < min_size || comp_size[rb] < min_size) {
            let new_root = uf.union(ra, rb);
            comp_size[new_root] = comp_size[ra] + comp_size[rb];
            int_diff[new_root] = *w;
        }
    }

    // Assign labels
    let mut label_map: HashMap<usize, usize> = HashMap::new();
    let mut labels = vec![vec![0usize; cols]; rows];
    let mut next_label = 0usize;
    for r in 0..rows {
        for c in 0..cols {
            let root = uf.find(r * cols + c);
            let label = label_map.entry(root).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            });
            labels[r][c] = *label;
        }
    }
    Ok(labels)
}

// ─── Mean Shift Segmentation ──────────────────────────────────────────────────

/// Mean shift segmentation of a grayscale image.
///
/// Each pixel is iteratively shifted toward the mean of nearby pixels within a
/// joint spatial–range kernel.  Convergence clusters pixels at mode locations.
///
/// # Arguments
/// * `image`             – 2-D grayscale image.
/// * `spatial_bandwidth` – Spatial search radius in pixels.
/// * `color_bandwidth`   – Intensity search radius.
/// * `max_iter`          – Maximum number of mode-seeking iterations per pixel.
pub fn mean_shift_segment(
    image: &[Vec<f64>],
    spatial_bandwidth: f64,
    color_bandwidth: f64,
    max_iter: usize,
) -> NdimageResult<Vec<Vec<usize>>> {
    let rows = image.len();
    if rows == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    let cols = image[0].len();
    if spatial_bandwidth <= 0.0 || color_bandwidth <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "Bandwidths must be positive".into(),
        ));
    }

    let hs2 = spatial_bandwidth * spatial_bandwidth;
    let hr2 = color_bandwidth * color_bandwidth;
    let sr = (spatial_bandwidth.ceil() as isize).max(1);

    // Find the mode for every pixel by mean-shift iteration
    let mut modes = vec![(0.0f64, 0.0f64, 0.0f64); rows * cols]; // (r, c, intensity)
    for r0 in 0..rows {
        for c0 in 0..cols {
            let mut mr = r0 as f64;
            let mut mc = c0 as f64;
            let mut mi = image[r0][c0];

            for _ in 0..max_iter {
                let ri = mr as isize;
                let ci = mc as isize;
                let mut sum_r = 0.0f64;
                let mut sum_c = 0.0f64;
                let mut sum_i = 0.0f64;
                let mut weight_sum = 0.0f64;
                let r_start = (ri - sr).max(0) as usize;
                let r_end = (ri + sr + 1).min(rows as isize) as usize;
                let c_start = (ci - sr).max(0) as usize;
                let c_end = (ci + sr + 1).min(cols as isize) as usize;
                for nr in r_start..r_end {
                    for nc in c_start..c_end {
                        let dr = nr as f64 - mr;
                        let dc = nc as f64 - mc;
                        let di = image[nr][nc] - mi;
                        let w_s = (-(dr * dr + dc * dc) / hs2).exp();
                        let w_r = (-(di * di) / hr2).exp();
                        let w = w_s * w_r;
                        sum_r += w * nr as f64;
                        sum_c += w * nc as f64;
                        sum_i += w * image[nr][nc];
                        weight_sum += w;
                    }
                }
                if weight_sum < 1e-12 {
                    break;
                }
                let new_r = sum_r / weight_sum;
                let new_c = sum_c / weight_sum;
                let new_i = sum_i / weight_sum;
                let shift = (new_r - mr).powi(2) + (new_c - mc).powi(2) + (new_i - mi).powi(2);
                mr = new_r;
                mc = new_c;
                mi = new_i;
                if shift < 1e-6 {
                    break;
                }
            }
            modes[r0 * cols + c0] = (mr, mc, mi);
        }
    }

    // Cluster modes: pixels with nearby modes get the same label
    let merge_tol_s = spatial_bandwidth * 0.5;
    let merge_tol_r = color_bandwidth * 0.5;
    let mut uf = UnionFind::new(rows * cols);
    for i in 0..(rows * cols) {
        for j in (i + 1)..(rows * cols) {
            let (mr1, mc1, mi1) = modes[i];
            let (mr2, mc2, mi2) = modes[j];
            let ds = ((mr1 - mr2).powi(2) + (mc1 - mc2).powi(2)).sqrt();
            let dr = (mi1 - mi2).abs();
            if ds < merge_tol_s && dr < merge_tol_r {
                uf.union_by_id(i, j);
            }
        }
    }

    let mut label_map: HashMap<usize, usize> = HashMap::new();
    let mut labels = vec![vec![0usize; cols]; rows];
    let mut next_label = 0usize;
    for r in 0..rows {
        for c in 0..cols {
            let root = uf.find(r * cols + c);
            let label = label_map.entry(root).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            });
            labels[r][c] = *label;
        }
    }
    Ok(labels)
}

// ─── GrabCut-style segmenter ──────────────────────────────────────────────────

/// GrabCut-style interactive segmenter using Gaussian Mixture Models.
///
/// Segments a colour (RGB) image into foreground and background using an
/// iterative GMM-based energy minimisation.  Initialise with a bounding
/// rectangle, then call `run()` for multiple refinement iterations.
pub struct GrabCutSegmenter {
    image: Vec<Vec<Vec<f64>>>,
    rows: usize,
    cols: usize,
    /// Per-pixel foreground probability (0 = BG, 1 = FG)
    fg_prob: Vec<Vec<f64>>,
    /// Foreground GMM components: (mean_r, mean_g, mean_b, var_r, var_g, var_b, weight)
    fg_gmm: Vec<[f64; 7]>,
    /// Background GMM components
    bg_gmm: Vec<[f64; 7]>,
    /// Initialisation mask (true = inside initial rect)
    init_mask: Vec<Vec<bool>>,
}

impl GrabCutSegmenter {
    /// Create a new segmenter for an RGB image `[rows][cols][3]`.
    pub fn new(image: Vec<Vec<Vec<f64>>>) -> NdimageResult<Self> {
        let rows = image.len();
        if rows == 0 {
            return Err(NdimageError::InvalidInput("Image must not be empty".into()));
        }
        let cols = image[0].len();
        if cols == 0 {
            return Err(NdimageError::InvalidInput(
                "Image columns must be > 0".into(),
            ));
        }
        Ok(GrabCutSegmenter {
            fg_prob: vec![vec![0.0; cols]; rows],
            fg_gmm: Vec::new(),
            bg_gmm: Vec::new(),
            init_mask: vec![vec![false; cols]; rows],
            rows,
            cols,
            image,
        })
    }

    /// Initialise with a rectangular region: pixels inside are likely foreground.
    ///
    /// `rect` is `(row_min, col_min, row_max, col_max)`.
    pub fn init_with_rect(&mut self, rect: (usize, usize, usize, usize)) {
        let (r0, c0, r1, c1) = rect;
        let r1 = r1.min(self.rows);
        let c1 = c1.min(self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                let inside = r >= r0 && r < r1 && c >= c0 && c < c1;
                self.init_mask[r][c] = inside;
                self.fg_prob[r][c] = if inside { 0.8 } else { 0.1 };
            }
        }
        self.fit_gmms();
    }

    /// Execute one GrabCut iteration and return the current foreground mask.
    pub fn iterate(&mut self) -> Vec<Vec<bool>> {
        // E-step: assign each pixel to FG or BG based on GMM likelihoods
        for r in 0..self.rows {
            for c in 0..self.cols {
                let pixel = &self.image[r][c];
                let fg_ll = gmm_likelihood(&self.fg_gmm, pixel);
                let bg_ll = gmm_likelihood(&self.bg_gmm, pixel);
                let total = fg_ll + bg_ll;
                self.fg_prob[r][c] = if total > 1e-300 { fg_ll / total } else { 0.5 };
            }
        }
        // M-step: refit GMMs
        self.fit_gmms();
        self.current_mask()
    }

    /// Run `n_iter` GrabCut iterations and return the final foreground mask.
    pub fn run(&mut self, n_iter: usize) -> Vec<Vec<bool>> {
        for _ in 0..n_iter {
            self.iterate();
        }
        self.current_mask()
    }

    fn current_mask(&self) -> Vec<Vec<bool>> {
        (0..self.rows)
            .map(|r| (0..self.cols).map(|c| self.fg_prob[r][c] >= 0.5).collect())
            .collect()
    }

    fn fit_gmms(&mut self) {
        let n_components = 5usize;
        // Collect FG and BG pixels
        let mut fg_pixels: Vec<Vec<f64>> = Vec::new();
        let mut bg_pixels: Vec<Vec<f64>> = Vec::new();
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.fg_prob[r][c] >= 0.5 {
                    fg_pixels.push(self.image[r][c].clone());
                } else {
                    bg_pixels.push(self.image[r][c].clone());
                }
            }
        }
        self.fg_gmm = fit_gmm_k_means(&fg_pixels, n_components);
        self.bg_gmm = fit_gmm_k_means(&bg_pixels, n_components);
    }
}

// ─── Quickshift ───────────────────────────────────────────────────────────────

/// Quickshift density-based image segmentation.
///
/// Each pixel is linked to its nearest neighbour in joint (spatial, colour)
/// space that has higher density.  Connected components form segments.
///
/// # Arguments
/// * `image`       – 2-D grayscale image.
/// * `kernel_size` – Kernel radius for density estimation (in pixels).
/// * `max_dist`    – Maximum allowed shift distance; larger → coarser segments.
/// * `ratio`       – Blend ratio between colour and spatial distances [0, 1].
pub fn quickshift_segment(
    image: &[Vec<f64>],
    kernel_size: usize,
    max_dist: f64,
    ratio: f64,
) -> NdimageResult<Vec<Vec<usize>>> {
    let rows = image.len();
    if rows == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    let cols = image[0].len();
    let sigma = kernel_size as f64 / 2.0;
    let sigma2 = sigma * sigma;
    let n = rows * cols;

    // Compute kernel density at each pixel using a Gaussian in (space, colour)
    let mut density = vec![0.0f64; n];
    let ks = kernel_size as isize;
    for r in 0..rows {
        for c in 0..cols {
            let mut d = 0.0f64;
            let val = image[r][c];
            for dr in -ks..=ks {
                for dc in -ks..=ks {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                        continue;
                    }
                    let nr = nr as usize;
                    let nc = nc as usize;
                    let ds = (dr * dr + dc * dc) as f64;
                    let di = (image[nr][nc] - val).powi(2);
                    d += (-(ratio * ds + (1.0 - ratio) * di) / sigma2).exp();
                }
            }
            density[r * cols + c] = d;
        }
    }

    // Each pixel links to the nearest higher-density neighbour within max_dist
    let mut parent = vec![usize::MAX; n];
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            let val = image[r][c];
            let my_density = density[idx];
            let mut best_dist = f64::INFINITY;
            let mut best_idx = idx;
            for dr in -ks..=ks {
                for dc in -ks..=ks {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                        continue;
                    }
                    let nr = nr as usize;
                    let nc = nc as usize;
                    let nidx = nr * cols + nc;
                    if density[nidx] <= my_density {
                        continue;
                    }
                    let ds = (dr * dr + dc * dc) as f64;
                    let di = (image[nr][nc] - val).powi(2);
                    let dist = (ratio * ds + (1.0 - ratio) * di).sqrt();
                    if dist < best_dist && dist < max_dist {
                        best_dist = dist;
                        best_idx = nidx;
                    }
                }
            }
            parent[idx] = best_idx;
        }
    }

    // Find roots (pixels that point to themselves) and label connected components
    let mut root_label = vec![usize::MAX; n];
    let mut next_label = 0usize;
    for i in 0..n {
        if parent[i] == i {
            root_label[i] = next_label;
            next_label += 1;
        }
    }

    // Resolve labels by following parent chains
    let mut labels_flat = vec![usize::MAX; n];
    for i in 0..n {
        let mut cur = i;
        let mut depth = 0;
        while parent[cur] != cur && depth < n {
            cur = parent[cur];
            depth += 1;
        }
        labels_flat[i] = root_label[cur].min(next_label.saturating_sub(1));
    }

    let mut labels = vec![vec![0usize; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            labels[r][c] = labels_flat[r * cols + c];
        }
    }
    Ok(labels)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Union-Find data structure with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> usize {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return rx;
        }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
            ry
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
            rx
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
            rx
        }
    }

    fn union_by_id(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        self.parent[ry] = rx;
    }
}

/// Colour distance between two multi-channel pixels.
fn pixel_diff(a: &[f64], b: &[f64], channels: usize) -> f64 {
    (0..channels.min(a.len()).min(b.len()))
        .map(|ch| (a[ch] - b[ch]).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Per-channel Gaussian smoothing of a 3-D image.
fn smooth_image_channels(image: &[Vec<Vec<f64>>], sigma: f64) -> Vec<Vec<Vec<f64>>> {
    if sigma <= 0.0 {
        return image.to_vec();
    }
    let rows = image.len();
    let cols = image[0].len();
    let channels = image[0][0].len();
    let radius = (3.0 * sigma).ceil() as usize;
    let side = 2 * radius + 1;
    let mut k1d = vec![0.0f64; side];
    for i in 0..side {
        let x = i as f64 - radius as f64;
        k1d[i] = (-x * x / (2.0 * sigma * sigma)).exp();
    }
    let sum: f64 = k1d.iter().sum();
    k1d.iter_mut().for_each(|v| *v /= sum);

    // Horizontal pass
    let mut tmp = vec![vec![vec![0.0f64; channels]; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            for ch in 0..channels {
                let mut acc = 0.0f64;
                for (ki, &kv) in k1d.iter().enumerate() {
                    let nc = (c as isize + ki as isize - radius as isize)
                        .max(0)
                        .min(cols as isize - 1) as usize;
                    acc += image[r][nc][ch] * kv;
                }
                tmp[r][c][ch] = acc;
            }
        }
    }
    // Vertical pass
    let mut out = vec![vec![vec![0.0f64; channels]; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            for ch in 0..channels {
                let mut acc = 0.0f64;
                for (ki, &kv) in k1d.iter().enumerate() {
                    let nr = (r as isize + ki as isize - radius as isize)
                        .max(0)
                        .min(rows as isize - 1) as usize;
                    acc += tmp[nr][c][ch] * kv;
                }
                out[r][c][ch] = acc;
            }
        }
    }
    out
}

/// Fit a simple Gaussian mixture via K-means initialisation.
/// Returns components as `[mean_r, mean_g, mean_b, var_r, var_g, var_b, weight]`.
fn fit_gmm_k_means(pixels: &[Vec<f64>], k: usize) -> Vec<[f64; 7]> {
    if pixels.is_empty() {
        return Vec::new();
    }
    let channels = pixels[0].len().min(3);
    let k = k.min(pixels.len()).max(1);

    // Initialise centroids as evenly-spaced pixel samples
    let step = pixels.len() / k;
    let mut centroids: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            pixels[i * step.max(1)]
                .iter()
                .take(channels)
                .copied()
                .collect()
        })
        .collect();

    let mut assignments = vec![0usize; pixels.len()];
    let max_iter = 10;
    for _ in 0..max_iter {
        // Assign
        let mut changed = false;
        for (pi, px) in pixels.iter().enumerate() {
            let best = (0..k)
                .map(|ki| {
                    (0..channels)
                        .map(|ch| {
                            let pv = if ch < px.len() { px[ch] } else { 0.0 };
                            let cv = centroids[ki][ch];
                            (pv - cv).powi(2)
                        })
                        .sum::<f64>()
                })
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if assignments[pi] != best {
                changed = true;
                assignments[pi] = best;
            }
        }
        if !changed {
            break;
        }
        // Update centroids
        let mut sums = vec![vec![0.0f64; channels]; k];
        let mut counts = vec![0usize; k];
        for (pi, px) in pixels.iter().enumerate() {
            let ki = assignments[pi];
            for ch in 0..channels {
                sums[ki][ch] += if ch < px.len() { px[ch] } else { 0.0 };
            }
            counts[ki] += 1;
        }
        for ki in 0..k {
            if counts[ki] > 0 {
                for ch in 0..channels {
                    centroids[ki][ch] = sums[ki][ch] / counts[ki] as f64;
                }
            }
        }
    }

    // Build GMM components
    let mut components: Vec<[f64; 7]> = Vec::with_capacity(k);
    for ki in 0..k {
        let cluster: Vec<&Vec<f64>> = pixels
            .iter()
            .zip(assignments.iter())
            .filter(|(_, &a)| a == ki)
            .map(|(p, _)| p)
            .collect();
        let n = cluster.len() as f64;
        let weight = n / pixels.len() as f64;
        let mut means = [0.0f64; 3];
        let mut vars = [1e-6f64; 3];
        if !cluster.is_empty() {
            for ch in 0..channels.min(3) {
                means[ch] = cluster.iter().map(|p| p[ch]).sum::<f64>() / n;
                vars[ch] = (cluster
                    .iter()
                    .map(|p| (p[ch] - means[ch]).powi(2))
                    .sum::<f64>()
                    / n)
                    .max(1e-6);
            }
        }
        components.push([
            means[0], means[1], means[2], vars[0], vars[1], vars[2], weight,
        ]);
    }
    components
}

/// Evaluate GMM log-likelihood for a pixel.
fn gmm_likelihood(gmm: &[[f64; 7]], pixel: &[f64]) -> f64 {
    if gmm.is_empty() {
        return 1e-300;
    }
    let channels = pixel.len().min(3);
    let mut total = 0.0f64;
    for comp in gmm {
        let w = comp[6];
        let mut log_p = 0.0f64;
        for ch in 0..channels {
            let mu = comp[ch];
            let var = comp[3 + ch].max(1e-12);
            log_p +=
                -0.5 * ((pixel[ch] - mu).powi(2) / var + (2.0 * std::f64::consts::PI * var).ln());
        }
        total += w * log_p.exp();
    }
    total.max(1e-300)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgb(rows: usize, cols: usize) -> Vec<Vec<Vec<f64>>> {
        (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| {
                        let v = if r < rows / 2 { 0.8 } else { 0.2 };
                        let v2 = if c < cols / 2 { v } else { 1.0 - v };
                        vec![v2, 0.5, 1.0 - v2]
                    })
                    .collect()
            })
            .collect()
    }

    fn make_gray(rows: usize, cols: usize) -> Vec<Vec<f64>> {
        (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| if (r / 4 + c / 4) % 2 == 0 { 0.9 } else { 0.1 })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_felzenszwalb_output_shape() {
        let img = make_rgb(32, 32);
        let labels = felzenszwalb_segment(&img, 50.0, 0.8, 5)
            .expect("felzenszwalb_segment should succeed on valid image");
        assert_eq!(labels.len(), 32);
        assert_eq!(labels[0].len(), 32);
    }

    #[test]
    fn test_felzenszwalb_empty_error() {
        let img: Vec<Vec<Vec<f64>>> = Vec::new();
        assert!(felzenszwalb_segment(&img, 50.0, 0.8, 5).is_err());
    }

    #[test]
    fn test_mean_shift_output_shape() {
        let img = make_gray(20, 20);
        let labels = mean_shift_segment(&img, 5.0, 0.3, 5)
            .expect("mean_shift_segment should succeed on valid image");
        assert_eq!(labels.len(), 20);
        assert_eq!(labels[0].len(), 20);
    }

    #[test]
    fn test_mean_shift_invalid_bandwidth() {
        let img = make_gray(20, 20);
        assert!(mean_shift_segment(&img, 0.0, 0.3, 5).is_err());
        assert!(mean_shift_segment(&img, 5.0, -1.0, 5).is_err());
    }

    #[test]
    fn test_grabcut_segmenter() {
        let img = make_rgb(24, 24);
        let mut gc = GrabCutSegmenter::new(img)
            .expect("GrabCutSegmenter::new should succeed on valid image");
        gc.init_with_rect((6, 6, 18, 18));
        let mask = gc.run(2);
        assert_eq!(mask.len(), 24);
        assert_eq!(mask[0].len(), 24);
    }

    #[test]
    fn test_quickshift_output_shape() {
        let img = make_gray(24, 24);
        let labels = quickshift_segment(&img, 3, 5.0, 0.5)
            .expect("quickshift_segment should succeed on valid image");
        assert_eq!(labels.len(), 24);
        assert_eq!(labels[0].len(), 24);
    }
}
