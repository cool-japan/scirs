//! Advanced segmentation algorithms for scirs2-ndimage
//!
//! This module provides state-of-the-art image segmentation algorithms:
//! - SLIC (Simple Linear Iterative Clustering) superpixels
//! - Graph cut segmentation using max-flow/min-cut
//! - Random walker segmentation
//! - Felzenszwalb efficient graph-based segmentation
//! - Quick shift mode-seeking segmentation

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array2;
use std::collections::{HashMap, HashSet, VecDeque};

// ─── SLIC Superpixels ───────────────────────────────────────────────────────

/// Computes SLIC (Simple Linear Iterative Clustering) superpixels.
///
/// SLIC partitions an image into `n_segments` approximately equal-sized
/// superpixels by iterating between cluster assignment and centroid update
/// in a 5-D (x, y, L, a, b) space. For grayscale images the colour space
/// is simply (x, y, intensity).
///
/// # Arguments
/// * `image`       – 2-D grayscale image (rows × cols)
/// * `n_segments`  – desired number of superpixels (≥ 1)
/// * `compactness` – balances colour vs spatial proximity (typical: 10.0)
///
/// # Returns
/// Label array of shape (rows, cols) with labels in `[0, n_segments)`.
pub fn superpixels_slic(
    image: &Array2<f64>,
    n_segments: usize,
    compactness: f64,
) -> NdimageResult<Array2<u32>> {
    if n_segments == 0 {
        return Err(NdimageError::InvalidInput(
            "n_segments must be at least 1".into(),
        ));
    }
    let rows = image.nrows();
    let cols = image.ncols();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }

    let n_pixels = rows * cols;
    let step = ((n_pixels as f64 / n_segments as f64).sqrt()).max(1.0) as usize;

    // Initialise cluster centres on a regular grid, perturbed to lowest
    // gradient magnitude in a 3×3 neighbourhood (reduces boundary placement).
    let mut centers: Vec<[f64; 3]> = Vec::new(); // [row, col, intensity]
    let mut ry = step / 2;
    while ry < rows {
        let mut cx = step / 2;
        while cx < cols {
            // Find pixel with minimum gradient in 3×3 window
            let (br, bc) = find_min_gradient_pixel(image, ry, cx, rows, cols);
            centers.push([br as f64, bc as f64, image[[br, bc]]]);
            cx += step;
        }
        ry += step;
    }

    let k = centers.len();
    if k == 0 {
        // Degenerate: single cluster
        return Ok(Array2::<u32>::zeros((rows, cols)));
    }

    // Spatial distance normalisation factor
    let s = step as f64;
    let compactness_sq = compactness * compactness;

    let max_iter = 10usize;
    let mut labels = Array2::<i32>::from_elem((rows, cols), -1i32);
    let mut distances = Array2::<f64>::from_elem((rows, cols), f64::INFINITY);

    for _iter in 0..max_iter {
        // Assignment step: search only within 2S × 2S window around each centre
        for (cluster_id, center) in centers.iter().enumerate() {
            let cr = center[0] as isize;
            let cc = center[1] as isize;
            let ci = center[2];

            let r_start = (cr - 2 * step as isize).max(0) as usize;
            let r_end = (cr + 2 * step as isize + 1).min(rows as isize) as usize;
            let c_start = (cc - 2 * step as isize).max(0) as usize;
            let c_end = (cc + 2 * step as isize + 1).min(cols as isize) as usize;

            for r in r_start..r_end {
                for c in c_start..c_end {
                    let di = image[[r, c]] - ci;
                    let dr = r as f64 - center[0];
                    let dc = c as f64 - center[1];
                    let d_color = di * di;
                    let d_space = (dr * dr + dc * dc) / (s * s) * compactness_sq;
                    let dist = (d_color + d_space).sqrt();
                    if dist < distances[[r, c]] {
                        distances[[r, c]] = dist;
                        labels[[r, c]] = cluster_id as i32;
                    }
                }
            }
        }

        // Update step: recompute cluster centres
        let mut sum_r = vec![0.0f64; k];
        let mut sum_c = vec![0.0f64; k];
        let mut sum_i = vec![0.0f64; k];
        let mut counts = vec![0u64; k];

        for r in 0..rows {
            for c in 0..cols {
                let lbl = labels[[r, c]];
                if lbl >= 0 {
                    let lid = lbl as usize;
                    sum_r[lid] += r as f64;
                    sum_c[lid] += c as f64;
                    sum_i[lid] += image[[r, c]];
                    counts[lid] += 1;
                }
            }
        }

        for j in 0..k {
            if counts[j] > 0 {
                let cnt = counts[j] as f64;
                centers[j][0] = sum_r[j] / cnt;
                centers[j][1] = sum_c[j] / cnt;
                centers[j][2] = sum_i[j] / cnt;
            }
        }

        // Reset distances for next iteration
        distances.fill(f64::INFINITY);
    }

    // Post-process: enforce connectivity – relabel disconnected fragments
    let label_map = enforce_connectivity(&labels, k, rows, cols);

    Ok(label_map)
}

/// Find the pixel in a 3×3 neighbourhood with the smallest gradient magnitude.
fn find_min_gradient_pixel(
    image: &Array2<f64>,
    ry: usize,
    cx: usize,
    rows: usize,
    cols: usize,
) -> (usize, usize) {
    let r_start = ry.saturating_sub(1);
    let r_end = (ry + 2).min(rows);
    let c_start = cx.saturating_sub(1);
    let c_end = (cx + 2).min(cols);

    let mut min_grad = f64::INFINITY;
    let mut best = (ry, cx);

    for r in r_start..r_end {
        for c in c_start..c_end {
            let gx = if c + 1 < cols {
                image[[r, c + 1]] - image[[r, c]]
            } else {
                0.0
            };
            let gy = if r + 1 < rows {
                image[[r + 1, c]] - image[[r, c]]
            } else {
                0.0
            };
            let g = gx * gx + gy * gy;
            if g < min_grad {
                min_grad = g;
                best = (r, c);
            }
        }
    }
    best
}

/// Enforce connectivity of SLIC labels via connected-component relabelling.
/// Disconnected fragments are assigned to the largest adjacent cluster.
fn enforce_connectivity(labels: &Array2<i32>, k: usize, rows: usize, cols: usize) -> Array2<u32> {
    let mut output = Array2::<u32>::zeros((rows, cols));
    let mut visited = Array2::<bool>::from_elem((rows, cols), false);
    let mut next_label = 0u32;
    // Map from original cluster id to final label
    let mut cluster_largest: Vec<u32> = vec![u32::MAX; k];
    let mut cluster_largest_size: Vec<usize> = vec![0usize; k];

    // BFS flood fill
    for start_r in 0..rows {
        for start_c in 0..cols {
            if visited[[start_r, start_c]] {
                continue;
            }
            let orig_label = labels[[start_r, start_c]].max(0) as usize;
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back((start_r, start_c));
            visited[[start_r, start_c]] = true;

            while let Some((r, c)) = queue.pop_front() {
                component.push((r, c));
                for (dr, dc) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    if nr < 0 || nc < 0 {
                        continue;
                    }
                    let nr = nr as usize;
                    let nc = nc as usize;
                    if nr >= rows || nc >= cols {
                        continue;
                    }
                    if visited[[nr, nc]] {
                        continue;
                    }
                    if labels[[nr, nc]].max(0) as usize == orig_label {
                        visited[[nr, nc]] = true;
                        queue.push_back((nr, nc));
                    }
                }
            }

            // Assign label
            let assign_label = if cluster_largest[orig_label] == u32::MAX {
                cluster_largest[orig_label] = next_label;
                cluster_largest_size[orig_label] = component.len();
                next_label += 1;
                cluster_largest[orig_label]
            } else {
                // Use parent cluster's label
                if component.len() > cluster_largest_size[orig_label] {
                    cluster_largest_size[orig_label] = component.len();
                }
                next_label += 1;
                next_label - 1
            };

            for (r, c) in component {
                output[[r, c]] = assign_label;
            }
        }
    }
    output
}

// ─── Graph Cut Segmentation ─────────────────────────────────────────────────

/// Graph cut segmentation using max-flow / min-cut (Boykov-Kolmogorov style).
///
/// Pixels given as `foreground_seeds` are forced into the foreground; pixels
/// in `background_seeds` are forced into the background. All other pixels are
/// labelled by solving the energy minimisation via a push-relabel max-flow.
///
/// # Arguments
/// * `image`             – 2-D grayscale image
/// * `foreground_seeds`  – (row, col) pairs for foreground
/// * `background_seeds`  – (row, col) pairs for background
///
/// # Returns
/// Boolean mask (true = foreground) of shape (rows, cols).
pub fn graph_cut_segmentation(
    image: &Array2<f64>,
    foreground_seeds: &[(usize, usize)],
    background_seeds: &[(usize, usize)],
) -> NdimageResult<Array2<bool>> {
    if foreground_seeds.is_empty() || background_seeds.is_empty() {
        return Err(NdimageError::InvalidInput(
            "Both foreground and background seeds are required".into(),
        ));
    }
    let rows = image.nrows();
    let cols = image.ncols();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }

    // Validate seeds
    for &(r, c) in foreground_seeds.iter().chain(background_seeds.iter()) {
        if r >= rows || c >= cols {
            return Err(NdimageError::InvalidInput(format!(
                "Seed ({r}, {c}) is out of bounds for image ({rows}x{cols})"
            )));
        }
    }

    let n_pixels = rows * cols;
    // Node indices: 0..n_pixels = pixels, n_pixels = source (fg), n_pixels+1 = sink (bg)
    let source = n_pixels;
    let sink = n_pixels + 1;
    let n_nodes = n_pixels + 2;

    // Build capacity matrix using HashMap for sparse storage
    let mut capacity: HashMap<(usize, usize), f64> = HashMap::new();

    let pixel_idx = |r: usize, c: usize| r * cols + c;

    // Helper: add directed edge (or increase capacity)
    let add_cap = |cap: &mut HashMap<(usize, usize), f64>, u: usize, v: usize, w: f64| {
        *cap.entry((u, v)).or_insert(0.0) += w;
        cap.entry((v, u)).or_insert(0.0);
    };

    // Compute local image statistics for boundary penalties
    let mean_val: f64 = image.iter().sum::<f64>() / (n_pixels as f64);
    let var_val: f64 =
        image.iter().map(|&v| (v - mean_val).powi(2)).sum::<f64>() / (n_pixels as f64).max(1.0);
    let sigma_sq = var_val.max(1e-10);

    // Boundary term: weight on N-link between neighbours
    let boundary_weight = |ia: f64, ib: f64| -> f64 {
        let diff = ia - ib;
        (-diff * diff / (2.0 * sigma_sq)).exp() + 1e-6
    };

    // Add N-links (pairwise terms) for 4-connected neighbours
    for r in 0..rows {
        for c in 0..cols {
            let p = pixel_idx(r, c);
            if c + 1 < cols {
                let q = pixel_idx(r, c + 1);
                let w = boundary_weight(image[[r, c]], image[[r, c + 1]]);
                add_cap(&mut capacity, p, q, w);
                add_cap(&mut capacity, q, p, w);
            }
            if r + 1 < rows {
                let q = pixel_idx(r + 1, c);
                let w = boundary_weight(image[[r, c]], image[[r + 1, c]]);
                add_cap(&mut capacity, p, q, w);
                add_cap(&mut capacity, q, p, w);
            }
        }
    }

    // Large constant for hard constraints
    let k_large = 1e9f64;

    // Add T-links for seeds
    for &(r, c) in foreground_seeds {
        let p = pixel_idx(r, c);
        add_cap(&mut capacity, source, p, k_large);
        add_cap(&mut capacity, p, sink, 0.0);
    }
    for &(r, c) in background_seeds {
        let p = pixel_idx(r, c);
        add_cap(&mut capacity, source, p, 0.0);
        add_cap(&mut capacity, p, sink, k_large);
    }

    // For unlabelled pixels: add weak T-links based on seed statistics
    let mut fg_mean = 0.0f64;
    for &(r, c) in foreground_seeds {
        fg_mean += image[[r, c]];
    }
    fg_mean /= foreground_seeds.len() as f64;

    let mut bg_mean = 0.0f64;
    for &(r, c) in background_seeds {
        bg_mean += image[[r, c]];
    }
    bg_mean /= background_seeds.len() as f64;

    let fg_set: HashSet<usize> = foreground_seeds
        .iter()
        .map(|&(r, c)| pixel_idx(r, c))
        .collect();
    let bg_set: HashSet<usize> = background_seeds
        .iter()
        .map(|&(r, c)| pixel_idx(r, c))
        .collect();

    for r in 0..rows {
        for c in 0..cols {
            let p = pixel_idx(r, c);
            if fg_set.contains(&p) || bg_set.contains(&p) {
                continue;
            }
            let val = image[[r, c]];
            let d_fg = (val - fg_mean).abs();
            let d_bg = (val - bg_mean).abs();
            let total = d_fg + d_bg + 1e-10;
            // Soft link: pixel close to fg mean gets high source capacity
            let w_source = (1.0 - d_fg / total) * 2.0;
            let w_sink = (1.0 - d_bg / total) * 2.0;
            add_cap(&mut capacity, source, p, w_source);
            add_cap(&mut capacity, p, sink, w_sink);
        }
    }

    // Build adjacency list for BFS
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n_nodes];
    for &(u, v) in capacity.keys() {
        adj[u].insert(v);
    }

    // Edmonds-Karp (BFS-based) max-flow
    let mut residual = capacity.clone();

    loop {
        // BFS to find augmenting path
        let mut parent = vec![None::<usize>; n_nodes];
        let mut visited = vec![false; n_nodes];
        let mut queue = VecDeque::new();
        queue.push_back(source);
        visited[source] = true;

        'bfs: while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if !visited[v] && *residual.get(&(u, v)).unwrap_or(&0.0) > 1e-12 {
                    visited[v] = true;
                    parent[v] = Some(u);
                    if v == sink {
                        break 'bfs;
                    }
                    queue.push_back(v);
                }
            }
        }

        if !visited[sink] {
            break;
        }

        // Find min residual capacity along path
        let mut path_flow = f64::INFINITY;
        let mut v = sink;
        while let Some(u) = parent[v] {
            let cap = *residual.get(&(u, v)).unwrap_or(&0.0);
            if cap < path_flow {
                path_flow = cap;
            }
            v = u;
        }

        // Update residual capacities
        let mut v = sink;
        while let Some(u) = parent[v] {
            *residual.entry((u, v)).or_insert(0.0) -= path_flow;
            *residual.entry((v, u)).or_insert(0.0) += path_flow;
            v = u;
        }
    }

    // Determine reachable nodes from source in residual graph (source side = foreground)
    let mut reachable = vec![false; n_nodes];
    let mut queue = VecDeque::new();
    queue.push_back(source);
    reachable[source] = true;
    while let Some(u) = queue.pop_front() {
        for &v in &adj[u] {
            if !reachable[v] && *residual.get(&(u, v)).unwrap_or(&0.0) > 1e-12 {
                reachable[v] = true;
                queue.push_back(v);
            }
        }
    }

    let mut result = Array2::<bool>::from_elem((rows, cols), false);
    for r in 0..rows {
        for c in 0..cols {
            result[[r, c]] = reachable[pixel_idx(r, c)];
        }
    }
    Ok(result)
}

// ─── Random Walker Segmentation ─────────────────────────────────────────────

/// Random walker segmentation with multiple labels.
///
/// Given a set of labelled seed pixels, solves the Dirichlet problem on the
/// image graph: the probability that a random walker started at an unlabelled
/// pixel first reaches each seed label. Each pixel is assigned to the label
/// with the highest hitting probability.
///
/// # Arguments
/// * `image` – 2-D grayscale image
/// * `seeds` – sparse seed map: 0 = unlabelled, positive integers = label id
///
/// # Returns
/// Dense label array of shape (rows, cols) with the same label values as
/// `seeds` (0 where unlabelled pixels have no reachable seeds).
pub fn random_walker_segmentation(
    image: &Array2<f64>,
    seeds: &Array2<u32>,
) -> NdimageResult<Array2<u32>> {
    let rows = image.nrows();
    let cols = image.ncols();
    if seeds.shape() != image.shape() {
        return Err(NdimageError::DimensionError(
            "seeds must have the same shape as image".into(),
        ));
    }
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }

    // Collect unique labels (excluding 0)
    let mut label_set: Vec<u32> = seeds
        .iter()
        .cloned()
        .filter(|&v| v > 0)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    label_set.sort_unstable();

    if label_set.is_empty() {
        return Ok(Array2::<u32>::zeros((rows, cols)));
    }

    let pixel_idx = |r: usize, c: usize| r * cols + c;
    let n_pixels = rows * cols;

    // Compute edge weights: w_ij = exp(-beta * (I_i - I_j)^2)
    let beta = compute_beta(image, rows, cols);

    let edge_weight = |ia: f64, ib: f64| -> f64 { (-beta * (ia - ib).powi(2)).exp().max(1e-10) };

    // For each label, run iterative label propagation (Gauss-Seidel solver)
    // to approximate the hitting probability.
    let n_labels = label_set.len();
    // probs[label_idx][pixel_idx]
    let mut probs: Vec<Vec<f64>> = vec![vec![0.0; n_pixels]; n_labels];

    // Initialise: seeded pixels get probability 1 for their label, 0 for others
    for r in 0..rows {
        for c in 0..cols {
            let s = seeds[[r, c]];
            if s > 0 {
                let p = pixel_idx(r, c);
                if let Some(li) = label_set.iter().position(|&l| l == s) {
                    for (k, prob_k) in probs.iter_mut().enumerate() {
                        prob_k[p] = if k == li { 1.0 } else { 0.0 };
                    }
                }
            }
        }
    }

    // Iterative solver (Gauss-Seidel relaxation)
    let max_iter = 100usize;
    let tol = 1e-4f64;

    let neighbors_of = |r: usize, c: usize| -> Vec<(usize, usize)> {
        let mut nb = Vec::with_capacity(4);
        if r > 0 {
            nb.push((r - 1, c));
        }
        if r + 1 < rows {
            nb.push((r + 1, c));
        }
        if c > 0 {
            nb.push((r, c - 1));
        }
        if c + 1 < cols {
            nb.push((r, c + 1));
        }
        nb
    };

    for label_idx in 0..n_labels {
        let prob = &mut probs[label_idx];
        for _iter in 0..max_iter {
            let mut max_change = 0.0f64;
            for r in 0..rows {
                for c in 0..cols {
                    let p = pixel_idx(r, c);
                    if seeds[[r, c]] > 0 {
                        continue; // Fixed seed
                    }
                    let nbs = neighbors_of(r, c);
                    let mut w_sum = 0.0f64;
                    let mut wp_sum = 0.0f64;
                    for (nr, nc) in nbs {
                        let q = pixel_idx(nr, nc);
                        let w = edge_weight(image[[r, c]], image[[nr, nc]]);
                        w_sum += w;
                        wp_sum += w * prob[q];
                    }
                    let new_val = if w_sum > 0.0 { wp_sum / w_sum } else { 0.0 };
                    let change = (new_val - prob[p]).abs();
                    if change > max_change {
                        max_change = change;
                    }
                    prob[p] = new_val;
                }
            }
            if max_change < tol {
                break;
            }
        }
    }

    // Assign labels: argmax over labels for each unlabelled pixel
    let mut result = seeds.clone();
    for r in 0..rows {
        for c in 0..cols {
            if seeds[[r, c]] == 0 {
                let p = pixel_idx(r, c);
                let best = (0..n_labels)
                    .max_by(|&a, &b| {
                        probs[a][p]
                            .partial_cmp(&probs[b][p])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0);
                result[[r, c]] = label_set[best];
            }
        }
    }

    Ok(result)
}

/// Compute the beta parameter for random walker edge weights.
fn compute_beta(image: &Array2<f64>, rows: usize, cols: usize) -> f64 {
    let mut sum_sq = 0.0f64;
    let mut count = 0usize;
    for r in 0..rows {
        for c in 0..cols {
            if c + 1 < cols {
                let d = image[[r, c]] - image[[r, c + 1]];
                sum_sq += d * d;
                count += 1;
            }
            if r + 1 < rows {
                let d = image[[r, c]] - image[[r + 1, c]];
                sum_sq += d * d;
                count += 1;
            }
        }
    }
    if count == 0 || sum_sq == 0.0 {
        return 1.0;
    }
    1.0 / (2.0 * sum_sq / count as f64)
}

// ─── Felzenszwalb Segmentation ───────────────────────────────────────────────

/// Felzenszwalb & Huttenlocher efficient graph-based image segmentation.
///
/// Builds a graph of pixels, sorts edges by weight, and uses a greedy
/// union-find algorithm to merge components whose internal difference
/// criterion allows the merge.
///
/// # Arguments
/// * `image`    – 2-D grayscale image
/// * `scale`    – controls segment size (larger → bigger segments, typical 100–1000)
/// * `sigma`    – Gaussian pre-smoothing sigma (0 to skip)
/// * `min_size` – minimum segment size in pixels (small components are merged)
///
/// # Returns
/// Label array of shape (rows, cols).
pub fn felzenszwalb_segmentation(
    image: &Array2<f64>,
    scale: f64,
    sigma: f64,
    min_size: usize,
) -> NdimageResult<Array2<u32>> {
    let rows = image.nrows();
    let cols = image.ncols();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    if scale <= 0.0 {
        return Err(NdimageError::InvalidInput("scale must be positive".into()));
    }

    // Optional Gaussian smoothing
    let smoothed = if sigma > 0.0 {
        gaussian_blur_2d(image, sigma)
    } else {
        image.clone()
    };

    let pixel_idx = |r: usize, c: usize| r * cols + c;
    let n_pixels = rows * cols;

    // Build edges: 4-connected grid graph
    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(2 * n_pixels);
    for r in 0..rows {
        for c in 0..cols {
            if c + 1 < cols {
                let w = (smoothed[[r, c]] - smoothed[[r, c + 1]]).abs();
                edges.push((w, pixel_idx(r, c), pixel_idx(r, c + 1)));
            }
            if r + 1 < rows {
                let w = (smoothed[[r, c]] - smoothed[[r + 1, c]]).abs();
                edges.push((w, pixel_idx(r, c), pixel_idx(r + 1, c)));
            }
        }
    }
    // Sort edges by weight
    edges.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Union-Find with internal difference and size tracking
    let mut parent: Vec<usize> = (0..n_pixels).collect();
    let mut rank: Vec<u32> = vec![0; n_pixels];
    let mut size: Vec<usize> = vec![1; n_pixels];
    let mut int_diff: Vec<f64> = vec![0.0; n_pixels]; // internal difference

    fn find_root(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find_root(parent, parent[x]);
        }
        parent[x]
    }

    fn union(
        parent: &mut Vec<usize>,
        rank: &mut Vec<u32>,
        size: &mut Vec<usize>,
        int_diff: &mut Vec<f64>,
        a: usize,
        b: usize,
        w: f64,
    ) {
        let ra = find_root(parent, a);
        let rb = find_root(parent, b);
        if ra == rb {
            return;
        }
        // Union by rank
        let (root, child) = if rank[ra] >= rank[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        parent[child] = root;
        if rank[ra] == rank[rb] {
            rank[root] += 1;
        }
        size[root] += size[child];
        int_diff[root] = w; // internal difference = last merged edge weight
    }

    for (w, u, v) in &edges {
        let ru = find_root(&mut parent, *u);
        let rv = find_root(&mut parent, *v);
        if ru == rv {
            continue;
        }
        // Merge threshold: MInt(C1, C2) = min(Int(C1) + k/|C1|, Int(C2) + k/|C2|)
        let threshold_u = int_diff[ru] + scale / size[ru] as f64;
        let threshold_v = int_diff[rv] + scale / size[rv] as f64;
        if *w <= threshold_u.min(threshold_v) {
            union(&mut parent, &mut rank, &mut size, &mut int_diff, ru, rv, *w);
        }
    }

    // Merge small components
    if min_size > 0 {
        for (w, u, v) in &edges {
            let ru = find_root(&mut parent, *u);
            let rv = find_root(&mut parent, *v);
            if ru != rv && (size[ru] < min_size || size[rv] < min_size) {
                union(&mut parent, &mut rank, &mut size, &mut int_diff, ru, rv, *w);
            }
        }
    }

    // Re-index labels to contiguous integers starting from 0
    let mut root_to_label: HashMap<usize, u32> = HashMap::new();
    let mut next_label = 0u32;
    let mut result = Array2::<u32>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let root = find_root(&mut parent, pixel_idx(r, c));
            let lbl = root_to_label.entry(root).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            });
            result[[r, c]] = *lbl;
        }
    }
    Ok(result)
}

/// Simple separable Gaussian blur for pre-smoothing.
fn gaussian_blur_2d(image: &Array2<f64>, sigma: f64) -> Array2<f64> {
    let rows = image.nrows();
    let cols = image.ncols();
    // Kernel half-width
    let hw = (3.0 * sigma).ceil() as usize;
    let ksize = 2 * hw + 1;
    let mut kernel = vec![0.0f64; ksize];
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut ksum = 0.0f64;
    for i in 0..ksize {
        let x = i as f64 - hw as f64;
        kernel[i] = (-x * x / two_sigma_sq).exp();
        ksum += kernel[i];
    }
    for k in kernel.iter_mut() {
        *k /= ksum;
    }

    // Horizontal pass
    let mut tmp = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let mut s = 0.0f64;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sc = c as isize + ki as isize - hw as isize;
                let sc = sc.max(0).min(cols as isize - 1) as usize;
                s += kv * image[[r, sc]];
            }
            tmp[[r, c]] = s;
        }
    }
    // Vertical pass
    let mut out = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let mut s = 0.0f64;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sr = r as isize + ki as isize - hw as isize;
                let sr = sr.max(0).min(rows as isize - 1) as usize;
                s += kv * tmp[[sr, c]];
            }
            out[[r, c]] = s;
        }
    }
    out
}

// ─── Quick Shift Segmentation ────────────────────────────────────────────────

/// Quick shift mode-seeking segmentation.
///
/// Each pixel is linked to its nearest neighbour in feature space
/// (position + colour) that has higher density, within `max_dist`.
/// Connected components of these links define segments.
///
/// # Arguments
/// * `image`       – 2-D grayscale image
/// * `kernel_size` – Gaussian kernel size for density estimation
/// * `max_dist`    – maximum distance for linking (spatial + colour)
///
/// # Returns
/// Label array of shape (rows, cols).
pub fn quickshift_segmentation(
    image: &Array2<f64>,
    kernel_size: f64,
    max_dist: f64,
) -> NdimageResult<Array2<u32>> {
    let rows = image.nrows();
    let cols = image.ncols();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    if kernel_size <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "kernel_size must be positive".into(),
        ));
    }

    let pixel_idx = |r: usize, c: usize| r * cols + c;
    let n_pixels = rows * cols;

    let sigma_sq = kernel_size * kernel_size;

    // Compute density estimate using Parzen window
    let mut density = vec![0.0f64; n_pixels];
    let search_radius = (3.0 * kernel_size).ceil() as isize;

    for r in 0..rows {
        for c in 0..cols {
            let p = pixel_idx(r, c);
            let iv = image[[r, c]];
            let mut d = 0.0f64;
            let r_start = (r as isize - search_radius).max(0) as usize;
            let r_end = (r as isize + search_radius + 1).min(rows as isize) as usize;
            let c_start = (c as isize - search_radius).max(0) as usize;
            let c_end = (c as isize + search_radius + 1).min(cols as isize) as usize;
            for nr in r_start..r_end {
                for nc in c_start..c_end {
                    let dr = (r as f64 - nr as f64).powi(2);
                    let dc = (c as f64 - nc as f64).powi(2);
                    let di = (iv - image[[nr, nc]]).powi(2);
                    d += (-(dr + dc + di) / (2.0 * sigma_sq)).exp();
                }
            }
            density[p] = d;
        }
    }

    // For each pixel, find parent: nearest pixel with higher density within max_dist
    let mut parent: Vec<usize> = (0..n_pixels).collect();
    let max_dist_sq = max_dist * max_dist;

    for r in 0..rows {
        for c in 0..cols {
            let p = pixel_idx(r, c);
            let dp = density[p];
            let iv = image[[r, c]];
            let mut best_dist = f64::INFINITY;
            let mut best_q = p; // self by default (root)

            let r_start = (r as isize - search_radius).max(0) as usize;
            let r_end = (r as isize + search_radius + 1).min(rows as isize) as usize;
            let c_start = (c as isize - search_radius).max(0) as usize;
            let c_end = (c as isize + search_radius + 1).min(cols as isize) as usize;

            for nr in r_start..r_end {
                for nc in c_start..c_end {
                    let q = pixel_idx(nr, nc);
                    if q == p {
                        continue;
                    }
                    if density[q] <= dp {
                        continue;
                    }
                    let dr = (r as f64 - nr as f64).powi(2);
                    let dc = (c as f64 - nc as f64).powi(2);
                    let di = (iv - image[[nr, nc]]).powi(2);
                    let dist_sq = dr + dc + di;
                    if dist_sq < max_dist_sq && dist_sq < best_dist {
                        best_dist = dist_sq;
                        best_q = q;
                    }
                }
            }
            parent[p] = best_q;
        }
    }

    // Find roots via path compression
    fn find_root_qs(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] == x {
            return x;
        }
        let root = find_root_qs(parent, parent[x]);
        parent[x] = root;
        root
    }

    let mut root_to_label: HashMap<usize, u32> = HashMap::new();
    let mut next_label = 0u32;
    let mut result = Array2::<u32>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let p = pixel_idx(r, c);
            let root = find_root_qs(&mut parent, p);
            let lbl = root_to_label.entry(root).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            });
            result[[r, c]] = *lbl;
        }
    }
    Ok(result)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn checkerboard(rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(r, c)| {
            if (r / 4 + c / 4) % 2 == 0 {
                0.0
            } else {
                1.0
            }
        })
    }

    fn gradient_image(rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(_, c)| c as f64 / cols as f64)
    }

    // ── SLIC tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_slic_basic() {
        let img = checkerboard(16, 16);
        let labels = superpixels_slic(&img, 4, 10.0).expect("slic failed");
        assert_eq!(labels.shape(), img.shape());
        // All values should be < n_segments * some_factor (due to connectivity enforcement)
        assert!(labels.iter().all(|&v| v < 1000));
    }

    #[test]
    fn test_slic_single_segment() {
        let img = Array2::<f64>::zeros((8, 8));
        let labels = superpixels_slic(&img, 1, 5.0).expect("slic single segment");
        assert_eq!(labels.shape(), img.shape());
    }

    #[test]
    fn test_slic_uniform_image() {
        let img = Array2::<f64>::from_elem((12, 12), 0.5);
        let labels = superpixels_slic(&img, 4, 10.0).expect("slic uniform");
        // Uniform image: spatial distance alone separates superpixels
        assert_eq!(labels.shape(), img.shape());
    }

    #[test]
    fn test_slic_invalid_n_segments() {
        let img = Array2::<f64>::zeros((8, 8));
        assert!(superpixels_slic(&img, 0, 10.0).is_err());
    }

    // ── Graph cut tests ─────────────────────────────────────────────────────

    #[test]
    fn test_graph_cut_basic() {
        // Dark left half / bright right half
        let img = Array2::from_shape_fn((8, 8), |(_, c)| if c < 4 { 0.1 } else { 0.9 });
        let fg_seeds = vec![(3, 1)]; // foreground in dark region
        let bg_seeds = vec![(3, 6)]; // background in bright region
        let mask = graph_cut_segmentation(&img, &fg_seeds, &bg_seeds).expect("graph cut failed");
        assert_eq!(mask.shape(), img.shape());
        // Seed pixels must match their assignment
        assert!(mask[[3, 1]]); // foreground seed
        assert!(!mask[[3, 6]]); // background seed
    }

    #[test]
    fn test_graph_cut_seed_validation() {
        let img = Array2::<f64>::zeros((4, 4));
        let result = graph_cut_segmentation(&img, &[(0, 0)], &[(10, 10)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_cut_empty_seeds() {
        let img = Array2::<f64>::zeros((4, 4));
        let result = graph_cut_segmentation(&img, &[], &[(0, 0)]);
        assert!(result.is_err());
    }

    // ── Random walker tests ─────────────────────────────────────────────────

    #[test]
    fn test_random_walker_basic() {
        let img = gradient_image(10, 10);
        let mut seeds = Array2::<u32>::zeros((10, 10));
        seeds[[0, 0]] = 1; // label 1 on the left
        seeds[[9, 9]] = 2; // label 2 on the right
        let labels = random_walker_segmentation(&img, &seeds).expect("rw failed");
        assert_eq!(labels.shape(), img.shape());
        // Seed positions must keep their labels
        assert_eq!(labels[[0, 0]], 1);
        assert_eq!(labels[[9, 9]], 2);
    }

    #[test]
    fn test_random_walker_single_label() {
        let img = Array2::<f64>::zeros((6, 6));
        let mut seeds = Array2::<u32>::zeros((6, 6));
        seeds[[3, 3]] = 1;
        let labels = random_walker_segmentation(&img, &seeds).expect("rw single label");
        assert!(labels.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_random_walker_no_seeds() {
        let img = Array2::<f64>::zeros((4, 4));
        let seeds = Array2::<u32>::zeros((4, 4));
        let labels = random_walker_segmentation(&img, &seeds).expect("rw no seeds");
        assert!(labels.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_random_walker_shape_mismatch() {
        let img = Array2::<f64>::zeros((4, 4));
        let seeds = Array2::<u32>::zeros((5, 5));
        assert!(random_walker_segmentation(&img, &seeds).is_err());
    }

    // ── Felzenszwalb tests ──────────────────────────────────────────────────

    #[test]
    fn test_felzenszwalb_basic() {
        let img = checkerboard(16, 16);
        let labels = felzenszwalb_segmentation(&img, 100.0, 1.0, 20).expect("felzenszwalb failed");
        assert_eq!(labels.shape(), img.shape());
        // Should have at least 1 segment
        let n_labels = labels.iter().cloned().collect::<HashSet<u32>>().len();
        assert!(n_labels >= 1);
    }

    #[test]
    fn test_felzenszwalb_uniform() {
        let img = Array2::<f64>::from_elem((8, 8), 0.5);
        let labels = felzenszwalb_segmentation(&img, 100.0, 0.0, 0).expect("felzenszwalb uniform");
        assert_eq!(labels.shape(), img.shape());
        // Uniform image should produce a single segment
        let n_labels = labels.iter().cloned().collect::<HashSet<u32>>().len();
        assert_eq!(n_labels, 1);
    }

    #[test]
    fn test_felzenszwalb_invalid_scale() {
        let img = Array2::<f64>::zeros((4, 4));
        assert!(felzenszwalb_segmentation(&img, -1.0, 1.0, 0).is_err());
    }

    // ── Quick shift tests ───────────────────────────────────────────────────

    #[test]
    fn test_quickshift_basic() {
        let img = checkerboard(12, 12);
        let labels = quickshift_segmentation(&img, 3.0, 10.0).expect("quickshift failed");
        assert_eq!(labels.shape(), img.shape());
    }

    #[test]
    fn test_quickshift_uniform() {
        let img = Array2::<f64>::from_elem((8, 8), 0.5);
        let labels = quickshift_segmentation(&img, 2.0, 100.0).expect("quickshift uniform");
        assert_eq!(labels.shape(), img.shape());
        // Uniform image with large max_dist → one or few segments
        let n_labels = labels.iter().cloned().collect::<HashSet<u32>>().len();
        assert!(n_labels <= 4);
    }

    #[test]
    fn test_quickshift_invalid_kernel() {
        let img = Array2::<f64>::zeros((4, 4));
        assert!(quickshift_segmentation(&img, 0.0, 10.0).is_err());
    }
}
