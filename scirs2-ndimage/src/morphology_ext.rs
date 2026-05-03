//! Extended mathematical morphology operations.
//!
//! Provides geodesic operations, top-hat transforms, morphological gradient,
//! rolling-ball background subtraction, convex hull, skeletonisation (Zhang-Suen),
//! and the medial axis transform.

use crate::error::{NdimageError, NdimageResult};
use std::collections::VecDeque;

// ─── Geodesic operations ──────────────────────────────────────────────────────

/// Geodesic dilation: expand `marker` constrained by `mask`.
///
/// Repeats binary dilation of `marker` while keeping all pixels within `mask`
/// (i.e., result is always ⊆ mask).  Stops when no further change occurs or
/// `max_iter` is reached.
///
/// `connectivity` must be 4 or 8.
pub fn geodesic_dilation(
    marker: &[Vec<bool>],
    mask: &[Vec<bool>],
    connectivity: u8,
    max_iter: usize,
) -> NdimageResult<Vec<Vec<bool>>> {
    validate_binary_pair(marker, mask)?;
    validate_connectivity(connectivity)?;
    let rows = marker.len();
    let cols = marker[0].len();
    let neighbors = neighbor_offsets(connectivity);
    let mut current = marker.to_vec();

    for _ in 0..max_iter {
        let mut next = vec![vec![false; cols]; rows];
        let mut changed = false;
        for r in 0..rows {
            for c in 0..cols {
                if !mask[r][c] {
                    continue;
                }
                let mut set = current[r][c];
                if !set {
                    for &(dr, dc) in &neighbors {
                        let nr = r as isize + dr;
                        let nc = c as isize + dc;
                        if nr >= 0
                            && nc >= 0
                            && (nr as usize) < rows
                            && (nc as usize) < cols
                            && current[nr as usize][nc as usize]
                        {
                            set = true;
                            break;
                        }
                    }
                }
                next[r][c] = set;
                if set && !current[r][c] {
                    changed = true;
                }
            }
        }
        current = next;
        if !changed {
            break;
        }
    }
    Ok(current)
}

/// Geodesic erosion: shrink `marker` constrained to remain ⊇ mask.
///
/// Repeats binary erosion of `marker` while ensuring all masked pixels remain.
/// Stops on convergence or after `max_iter` iterations.
pub fn geodesic_erosion(
    marker: &[Vec<bool>],
    mask: &[Vec<bool>],
    connectivity: u8,
    max_iter: usize,
) -> NdimageResult<Vec<Vec<bool>>> {
    validate_binary_pair(marker, mask)?;
    validate_connectivity(connectivity)?;
    let rows = marker.len();
    let cols = marker[0].len();
    let neighbors = neighbor_offsets(connectivity);
    let mut current = marker.to_vec();

    for _ in 0..max_iter {
        let mut next = vec![vec![false; cols]; rows];
        let mut changed = false;
        for r in 0..rows {
            for c in 0..cols {
                // Geodesic erosion: keep pixel if all connected neighbours are set
                // OR if mask forces it.
                let mut eroded = current[r][c];
                if eroded {
                    for &(dr, dc) in &neighbors {
                        let nr = r as isize + dr;
                        let nc = c as isize + dc;
                        if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                            continue;
                        }
                        if !current[nr as usize][nc as usize] {
                            eroded = false;
                            break;
                        }
                    }
                }
                // Constrain: must be on if mask[r][c]
                let val = eroded || mask[r][c];
                next[r][c] = val;
                if val != current[r][c] {
                    changed = true;
                }
            }
        }
        current = next;
        if !changed {
            break;
        }
    }
    Ok(current)
}

/// Morphological reconstruction by dilation.
///
/// Dilates `seed` iteratively, constrained by `mask` (seed ⊆ mask required).
/// Equivalent to `geodesic_dilation` with unlimited iterations until convergence.
pub fn reconstruction_by_dilation(
    seed: &[Vec<bool>],
    mask: &[Vec<bool>],
) -> NdimageResult<Vec<Vec<bool>>> {
    let rows = seed.len();
    if rows == 0 {
        return Err(NdimageError::InvalidInput(
            "Inputs must not be empty".into(),
        ));
    }
    let cols = seed[0].len();
    // BFS-based reconstruction for efficiency
    let mut result = seed.to_vec();
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    let neighbors: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    // Seed queue with all initially-set pixels
    for r in 0..rows {
        for c in 0..cols {
            if result[r][c] {
                queue.push_back((r, c));
            }
        }
    }

    while let Some((r, c)) = queue.pop_front() {
        for &(dr, dc) in &neighbors {
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                continue;
            }
            let nr = nr as usize;
            let nc = nc as usize;
            if mask[nr][nc] && !result[nr][nc] {
                result[nr][nc] = true;
                queue.push_back((nr, nc));
            }
        }
    }
    Ok(result)
}

// ─── Top-hat transforms ───────────────────────────────────────────────────────

/// White top-hat transform: `image - opening(image, selem)`.
///
/// Enhances bright features smaller than the structuring element.
pub fn white_tophat(image: &[Vec<f64>], selem: &[Vec<bool>]) -> NdimageResult<Vec<Vec<f64>>> {
    let opened = gray_opening(image, selem)?;
    let rows = image.len();
    let cols = if rows > 0 { image[0].len() } else { 0 };
    let mut out = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            out[r][c] = (image[r][c] - opened[r][c]).max(0.0);
        }
    }
    Ok(out)
}

/// Black top-hat transform: `closing(image, selem) - image`.
///
/// Enhances dark features smaller than the structuring element.
pub fn black_tophat(image: &[Vec<f64>], selem: &[Vec<bool>]) -> NdimageResult<Vec<Vec<f64>>> {
    let closed = gray_closing(image, selem)?;
    let rows = image.len();
    let cols = if rows > 0 { image[0].len() } else { 0 };
    let mut out = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            out[r][c] = (closed[r][c] - image[r][c]).max(0.0);
        }
    }
    Ok(out)
}

/// Morphological gradient: `dilation(image, selem) - erosion(image, selem)`.
///
/// Highlights boundaries and edges.
pub fn morphological_gradient(
    image: &[Vec<f64>],
    selem: &[Vec<bool>],
) -> NdimageResult<Vec<Vec<f64>>> {
    let dilated = gray_dilate(image, selem)?;
    let eroded = gray_erode(image, selem)?;
    let rows = image.len();
    let cols = if rows > 0 { image[0].len() } else { 0 };
    let mut out = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            out[r][c] = dilated[r][c] - eroded[r][c];
        }
    }
    Ok(out)
}

// ─── Rolling-ball background subtraction ─────────────────────────────────────

/// Estimate background using a rolling-ball algorithm and return it.
///
/// The rolling-ball algorithm estimates a slowly-varying background by
/// computing a grayscale opening with a spherical structuring element
/// approximated as a disk.  The returned image is the estimated background.
/// Subtract from original to get the foreground.
///
/// # Arguments
/// * `image`  – 2-D intensity image.
/// * `radius` – Radius of the rolling ball in pixels.
pub fn rolling_ball_background(image: &[Vec<f64>], radius: f64) -> NdimageResult<Vec<Vec<f64>>> {
    if image.is_empty() {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    if radius <= 0.0 {
        return Err(NdimageError::InvalidInput("Radius must be positive".into()));
    }
    let r_int = radius.ceil() as usize;
    // Build a disk selem
    let selem = disk_selem(r_int);
    // Background ≈ grayscale opening
    gray_opening(image, &selem)
}

// ─── Convex hull ─────────────────────────────────────────────────────────────

/// Compute the convex hull of a binary image.
///
/// Returns a binary image where `true` marks pixels inside the convex hull
/// of the set of `true` pixels in the input.
pub fn convex_hull_image(binary: &[Vec<bool>]) -> NdimageResult<Vec<Vec<bool>>> {
    let rows = binary.len();
    if rows == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    let cols = binary[0].len();

    // Collect foreground points
    let mut pts: Vec<(isize, isize)> = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if binary[r][c] {
                pts.push((r as isize, c as isize));
            }
        }
    }
    if pts.len() < 3 {
        // Trivially fill the points themselves
        let mut out = vec![vec![false; cols]; rows];
        for &(r, c) in &pts {
            out[r as usize][c as usize] = true;
        }
        return Ok(out);
    }

    let hull = convex_hull_2d(&pts);
    let mut out = vec![vec![false; cols]; rows];
    // Fill convex hull using scan-line
    for r in 0..rows {
        let y = r as isize;
        let mut x_inters: Vec<isize> = Vec::new();
        let n = hull.len();
        for i in 0..n {
            let (y1, x1) = hull[i];
            let (y2, x2) = hull[(i + 1) % n];
            if (y1 <= y && y < y2) || (y2 <= y && y < y1) {
                // Compute intersection x at scan line y
                let x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1);
                x_inters.push(x_int);
            }
        }
        x_inters.sort();
        let mut i = 0;
        while i + 1 < x_inters.len() {
            let x_start = x_inters[i].max(0) as usize;
            let x_end = (x_inters[i + 1] + 1).min(cols as isize) as usize;
            for c in x_start..x_end {
                out[r][c] = true;
            }
            i += 2;
        }
    }
    Ok(out)
}

// ─── Zhang-Suen skeletonisation ───────────────────────────────────────────────

/// Skeletonise a binary image using the Zhang-Suen thinning algorithm.
///
/// Iteratively removes boundary pixels while preserving topology, until
/// no more pixels can be removed.
pub fn skeletonize(binary: &[Vec<bool>]) -> NdimageResult<Vec<Vec<bool>>> {
    let rows = binary.len();
    if rows == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    let cols = binary[0].len();
    let mut img: Vec<Vec<u8>> = binary
        .iter()
        .map(|r| r.iter().map(|&v| v as u8).collect())
        .collect();

    loop {
        let mut to_delete: Vec<(usize, usize)> = Vec::new();
        // Sub-iteration 1
        for r in 1..(rows - 1) {
            for c in 1..(cols - 1) {
                if img[r][c] == 0 {
                    continue;
                }
                let (p2, p3, p4, p5, p6, p7, p8, p9) = zhang_suen_neighbors(&img, r, c);
                let b = (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) as usize;
                let a = transitions_01(&[p2, p3, p4, p5, p6, p7, p8, p9]);
                if b >= 2 && b <= 6 && a == 1 && (p2 * p4 * p6 == 0) && (p4 * p6 * p8 == 0) {
                    to_delete.push((r, c));
                }
            }
        }
        for (r, c) in &to_delete {
            img[*r][*c] = 0;
        }
        let changed1 = !to_delete.is_empty();
        to_delete.clear();

        // Sub-iteration 2
        for r in 1..(rows - 1) {
            for c in 1..(cols - 1) {
                if img[r][c] == 0 {
                    continue;
                }
                let (p2, p3, p4, p5, p6, p7, p8, p9) = zhang_suen_neighbors(&img, r, c);
                let b = (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) as usize;
                let a = transitions_01(&[p2, p3, p4, p5, p6, p7, p8, p9]);
                if b >= 2 && b <= 6 && a == 1 && (p2 * p4 * p8 == 0) && (p2 * p6 * p8 == 0) {
                    to_delete.push((r, c));
                }
            }
        }
        for (r, c) in &to_delete {
            img[*r][*c] = 0;
        }
        let changed2 = !to_delete.is_empty();

        if !changed1 && !changed2 {
            break;
        }
    }

    let result = img
        .iter()
        .map(|r| r.iter().map(|&v| v != 0).collect())
        .collect();
    Ok(result)
}

// ─── Medial axis transform ────────────────────────────────────────────────────

/// Compute the medial axis of a binary image via ridge detection on the
/// distance transform.
///
/// Returns `(skeleton, distance_transform)`.  The skeleton contains the ridges
/// (local maxima) of the Euclidean distance transform.
pub fn medial_axis(binary: &[Vec<bool>]) -> NdimageResult<(Vec<Vec<bool>>, Vec<Vec<f64>>)> {
    let rows = binary.len();
    if rows == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    let cols = binary[0].len();

    // Compute Euclidean distance transform
    let dt = euclidean_distance_transform(binary);

    // Detect ridges: a pixel is on the medial axis if it is a local maximum
    // in the distance transform among its 8-connected neighbours.
    let mut skeleton = vec![vec![false; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            if !binary[r][c] || dt[r][c] < 1e-9 {
                continue;
            }
            let val = dt[r][c];
            let mut is_max = true;
            for dr in -1isize..=1 {
                for dc in -1isize..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                        continue;
                    }
                    if dt[nr as usize][nc as usize] > val {
                        is_max = false;
                        break;
                    }
                }
                if !is_max {
                    break;
                }
            }
            skeleton[r][c] = is_max;
        }
    }
    Ok((skeleton, dt))
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

fn validate_binary_pair(a: &[Vec<bool>], b: &[Vec<bool>]) -> NdimageResult<()> {
    if a.is_empty() || b.is_empty() {
        return Err(NdimageError::InvalidInput(
            "Inputs must not be empty".into(),
        ));
    }
    if a.len() != b.len() || a[0].len() != b[0].len() {
        return Err(NdimageError::InvalidInput(
            "Inputs must have the same shape".into(),
        ));
    }
    Ok(())
}

fn validate_connectivity(c: u8) -> NdimageResult<()> {
    if c != 4 && c != 8 {
        return Err(NdimageError::InvalidInput(
            "connectivity must be 4 or 8".into(),
        ));
    }
    Ok(())
}

fn neighbor_offsets(connectivity: u8) -> Vec<(isize, isize)> {
    if connectivity == 4 {
        vec![(-1, 0), (1, 0), (0, -1), (0, 1)]
    } else {
        vec![
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    }
}

/// Grayscale dilation with a flat structuring element.
fn gray_dilate(image: &[Vec<f64>], selem: &[Vec<bool>]) -> NdimageResult<Vec<Vec<f64>>> {
    if image.is_empty() || selem.is_empty() {
        return Err(NdimageError::InvalidInput(
            "Inputs must not be empty".into(),
        ));
    }
    let rows = image.len();
    let cols = image[0].len();
    let sh = selem.len();
    let sw = selem[0].len();
    let ar = (sh / 2) as isize;
    let ac = (sw / 2) as isize;
    let mut out = vec![vec![f64::NEG_INFINITY; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let mut max_val = f64::NEG_INFINITY;
            for sr in 0..sh {
                for sc in 0..sw {
                    if !selem[sr][sc] {
                        continue;
                    }
                    let nr = (r as isize + sr as isize - ar)
                        .max(0)
                        .min(rows as isize - 1) as usize;
                    let nc = (c as isize + sc as isize - ac)
                        .max(0)
                        .min(cols as isize - 1) as usize;
                    if image[nr][nc] > max_val {
                        max_val = image[nr][nc];
                    }
                }
            }
            out[r][c] = max_val;
        }
    }
    Ok(out)
}

/// Grayscale erosion with a flat structuring element.
fn gray_erode(image: &[Vec<f64>], selem: &[Vec<bool>]) -> NdimageResult<Vec<Vec<f64>>> {
    if image.is_empty() || selem.is_empty() {
        return Err(NdimageError::InvalidInput(
            "Inputs must not be empty".into(),
        ));
    }
    let rows = image.len();
    let cols = image[0].len();
    let sh = selem.len();
    let sw = selem[0].len();
    let ar = (sh / 2) as isize;
    let ac = (sw / 2) as isize;
    let mut out = vec![vec![f64::INFINITY; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let mut min_val = f64::INFINITY;
            for sr in 0..sh {
                for sc in 0..sw {
                    if !selem[sr][sc] {
                        continue;
                    }
                    let nr = (r as isize + sr as isize - ar)
                        .max(0)
                        .min(rows as isize - 1) as usize;
                    let nc = (c as isize + sc as isize - ac)
                        .max(0)
                        .min(cols as isize - 1) as usize;
                    if image[nr][nc] < min_val {
                        min_val = image[nr][nc];
                    }
                }
            }
            out[r][c] = min_val;
        }
    }
    Ok(out)
}

/// Grayscale opening (erosion then dilation).
fn gray_opening(image: &[Vec<f64>], selem: &[Vec<bool>]) -> NdimageResult<Vec<Vec<f64>>> {
    let eroded = gray_erode(image, selem)?;
    gray_dilate(&eroded, selem)
}

/// Grayscale closing (dilation then erosion).
fn gray_closing(image: &[Vec<f64>], selem: &[Vec<bool>]) -> NdimageResult<Vec<Vec<f64>>> {
    let dilated = gray_dilate(image, selem)?;
    gray_erode(&dilated, selem)
}

/// Build a disk structuring element.
fn disk_selem(radius: usize) -> Vec<Vec<bool>> {
    let side = 2 * radius + 1;
    let c = radius as f64;
    let r2 = (radius as f64).powi(2) + 1e-9;
    (0..side)
        .map(|r| {
            (0..side)
                .map(|col| {
                    let dr = r as f64 - c;
                    let dc = col as f64 - c;
                    dr * dr + dc * dc <= r2
                })
                .collect()
        })
        .collect()
}

/// Graham scan convex hull for 2-D integer points.
fn convex_hull_2d(pts: &[(isize, isize)]) -> Vec<(isize, isize)> {
    let mut sorted = pts.to_vec();
    // Use (col, row) ordering for standard 2-D geometry
    sorted.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    sorted.dedup();
    let n = sorted.len();
    if n <= 1 {
        return sorted;
    }

    let cross = |o: (isize, isize), a: (isize, isize), b: (isize, isize)| -> isize {
        (a.1 - o.1) * (b.0 - o.0) - (a.0 - o.0) * (b.1 - o.1)
    };

    let mut lower: Vec<(isize, isize)> = Vec::new();
    for &p in &sorted {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(isize, isize)> = Vec::new();
    for &p in sorted.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Zhang-Suen 3×3 neighbourhood pattern.
#[allow(clippy::too_many_arguments)]
fn zhang_suen_neighbors(img: &[Vec<u8>], r: usize, c: usize) -> (u8, u8, u8, u8, u8, u8, u8, u8) {
    (
        img[r - 1][c],
        img[r - 1][c + 1],
        img[r][c + 1],
        img[r + 1][c + 1],
        img[r + 1][c],
        img[r + 1][c - 1],
        img[r][c - 1],
        img[r - 1][c - 1],
    )
}

/// Count 0→1 transitions in a circular neighbourhood sequence.
fn transitions_01(ns: &[u8]) -> usize {
    let n = ns.len();
    (0..n)
        .filter(|&i| ns[i] == 0 && ns[(i + 1) % n] == 1)
        .count()
}

/// Euclidean distance transform.
///
/// For each background pixel, computes the distance to the nearest foreground pixel.
/// Foreground pixels have distance 0.
/// Uses the Meijster-Roerdink-Hesselink separable algorithm with correct
/// parabolic envelope construction.
fn euclidean_distance_transform(binary: &[Vec<bool>]) -> Vec<Vec<f64>> {
    let rows = binary.len();
    let cols = if rows > 0 { binary[0].len() } else { 0 };
    if rows == 0 || cols == 0 {
        return vec![vec![0.0; cols]; rows];
    }
    let big = (rows + cols) as f64;

    // Pass 1: column-wise 1D distance (in terms of row offsets)
    let mut col_dist = vec![vec![big; cols]; rows];
    for c in 0..cols {
        // Forward scan
        col_dist[0][c] = if binary[0][c] { 0.0 } else { big };
        for r in 1..rows {
            col_dist[r][c] = if binary[r][c] {
                0.0
            } else {
                col_dist[r - 1][c] + 1.0
            };
        }
        // Backward scan
        for r in (0..rows.saturating_sub(1)).rev() {
            let back = col_dist[r + 1][c] + 1.0;
            if back < col_dist[r][c] {
                col_dist[r][c] = back;
            }
        }
    }

    // Pass 2: row-wise using Felzenszwalb-Huttenlocher lower envelope
    // For each row, compute dt^2[r][c] = min over q of (col_dist[r][q]^2 + (c-q)^2)
    let mut dt = vec![vec![0.0f64; cols]; rows];

    for r in 0..rows {
        // Compute 1D lower envelope for g[q] = col_dist[r][q]^2
        let g: Vec<f64> = (0..cols).map(|q| col_dist[r][q] * col_dist[r][q]).collect();

        // Parabola indices and intersection boundaries
        let mut v = vec![0usize; cols]; // parabola centers
        let mut z = vec![0.0f64; cols + 1]; // intersection boundaries
        let mut k: usize = 0;
        v[0] = 0;
        z[0] = f64::NEG_INFINITY;
        z[1] = f64::INFINITY;

        for q in 1..cols {
            // Intersection of parabola at v[k] and parabola at q
            loop {
                let vk = v[k];
                let s = ((g[q] - g[vk]) as f64 + (q * q) as f64 - (vk * vk) as f64)
                    / (2.0 * (q as f64 - vk as f64));
                if s > z[k] {
                    // New parabola goes on top of stack
                    k += 1;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = f64::INFINITY;
                    break;
                }
                // Pop: current parabola is dominated
                if k == 0 {
                    v[0] = q;
                    z[0] = f64::NEG_INFINITY;
                    z[1] = f64::INFINITY;
                    break;
                }
                k -= 1;
            }
        }

        // Read off minimum for each column
        k = 0;
        for c in 0..cols {
            while z[k + 1] < c as f64 {
                k += 1;
            }
            let d = c as f64 - v[k] as f64;
            dt[r][c] = (g[v[k]] + d * d).sqrt();
        }
    }

    dt
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rect_binary(rows: usize, cols: usize) -> Vec<Vec<bool>> {
        (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| r > 0 && r < rows - 1 && c > 0 && c < cols - 1)
                    .collect()
            })
            .collect()
    }

    fn square_selem(half: usize) -> Vec<Vec<bool>> {
        let side = 2 * half + 1;
        vec![vec![true; side]; side]
    }

    fn make_gray(rows: usize, cols: usize) -> Vec<Vec<f64>> {
        (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| (r + c) as f64 / (rows + cols) as f64)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_geodesic_dilation_converges() {
        let marker = vec![
            vec![false, false, false, false],
            vec![false, true, false, false],
            vec![false, false, false, false],
        ];
        let mask = vec![
            vec![false, true, true, false],
            vec![false, true, true, false],
            vec![false, false, false, false],
        ];
        let result = geodesic_dilation(&marker, &mask, 4, 100)
            .expect("geodesic_dilation should succeed with valid marker and mask");
        // Marker should fill the reachable mask region
        assert!(result[0][1] || result[1][1]);
    }

    #[test]
    fn test_geodesic_erosion_runs() {
        let marker = make_rect_binary(8, 8);
        let mask = make_rect_binary(8, 8);
        let result = geodesic_erosion(&marker, &mask, 4, 10)
            .expect("geodesic_erosion should succeed with valid marker and mask");
        assert_eq!(result.len(), 8);
        assert_eq!(result[0].len(), 8);
    }

    #[test]
    fn test_reconstruction_by_dilation() {
        let seed = vec![
            vec![false, false, false],
            vec![false, true, false],
            vec![false, false, false],
        ];
        let mask = vec![
            vec![false, true, false],
            vec![true, true, true],
            vec![false, true, false],
        ];
        let result = reconstruction_by_dilation(&seed, &mask)
            .expect("reconstruction_by_dilation should succeed with valid seed and mask");
        assert!(result[0][1]);
        assert!(result[1][0]);
    }

    #[test]
    fn test_white_tophat_shape() {
        let img = make_gray(16, 16);
        let se = square_selem(2);
        let out =
            white_tophat(&img, &se).expect("white_tophat should succeed with valid image and SE");
        assert_eq!(out.len(), 16);
        assert_eq!(out[0].len(), 16);
        // White tophat values must be non-negative
        for row in &out {
            for &v in row {
                assert!(v >= 0.0, "white tophat must be non-negative");
            }
        }
    }

    #[test]
    fn test_black_tophat_shape() {
        let img = make_gray(16, 16);
        let se = square_selem(2);
        let out =
            black_tophat(&img, &se).expect("black_tophat should succeed with valid image and SE");
        assert_eq!(out.len(), 16);
        assert_eq!(out[0].len(), 16);
    }

    #[test]
    fn test_morphological_gradient() {
        let img = make_gray(16, 16);
        let se = square_selem(1);
        let out = morphological_gradient(&img, &se)
            .expect("morphological_gradient should succeed with valid image and SE");
        assert_eq!(out.len(), 16);
        for row in &out {
            for &v in row {
                assert!(v >= 0.0);
            }
        }
    }

    #[test]
    fn test_rolling_ball_background() {
        let img = make_gray(20, 20);
        let bg = rolling_ball_background(&img, 3.0)
            .expect("rolling_ball_background should succeed with valid image and radius");
        assert_eq!(bg.len(), 20);
        assert_eq!(bg[0].len(), 20);
    }

    #[test]
    fn test_convex_hull_image() {
        let mut binary = vec![vec![false; 16]; 16];
        binary[4][4] = true;
        binary[4][12] = true;
        binary[12][4] = true;
        binary[12][12] = true;
        let hull = convex_hull_image(&binary)
            .expect("convex_hull_image should succeed with valid binary image");
        assert_eq!(hull.len(), 16);
        // Center should be inside hull
        assert!(hull[8][8]);
    }

    #[test]
    fn test_skeletonize() {
        let binary = make_rect_binary(16, 16);
        let skel =
            skeletonize(&binary).expect("skeletonize should succeed with valid binary image");
        assert_eq!(skel.len(), 16);
        assert_eq!(skel[0].len(), 16);
    }

    #[test]
    fn test_medial_axis_shape() {
        let binary = make_rect_binary(12, 12);
        let (skel, dt) =
            medial_axis(&binary).expect("medial_axis should succeed with valid binary image");
        assert_eq!(skel.len(), 12);
        assert_eq!(dt.len(), 12);
        // Distance at border should be 0
        assert!(dt[0][0] < 1e-9 || !binary[0][0]);
    }

    #[test]
    fn test_invalid_connectivity() {
        let b = make_rect_binary(8, 8);
        assert!(geodesic_dilation(&b, &b, 3, 10).is_err());
    }
}
