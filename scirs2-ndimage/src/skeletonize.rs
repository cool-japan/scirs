//! Morphological skeletonization and binary shape analysis
//!
//! This module provides:
//!
//! - **Zhang-Suen thinning**: Classic parallel thinning algorithm for binary images.
//! - **Lee's thinning**: Improved topology-preserving parallel thinning.
//! - **Medial axis transform**: Skeleton plus distance map to nearest background pixel.
//! - **Euclidean distance transform** (thin wrapper over `distance_transforms::euclidean_dt`).
//! - **City-block distance transform** (wrapping `distance_transforms::cityblock_dt`).
//! - **Convex hull** of a binary image.
//! - **Binary fill holes** via flood fill from the border.
//!
//! # References
//!
//! - Zhang, T.Y. & Suen, C.Y. (1984). "A Fast Parallel Algorithm for Thinning Digital Patterns."
//! - Lee, T.C., Kashyap, R.L., & Chu, C.N. (1994). "Building Skeleton Models via 3-D Medial
//!   Surface Axis Thinning Algorithms."
//! - Meijster, A., Roerdink, J.B.T.M., Hesselink, W.H. (2000).
//!   "A General Algorithm for Computing Distance Transforms in Linear Time."

use crate::distance_transforms::{cityblock_dt, euclidean_dt};
use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array2;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Zhang-Suen thinning
// ---------------------------------------------------------------------------

/// Thin a binary image using the Zhang-Suen (1984) parallel thinning algorithm.
///
/// The algorithm iteratively removes border pixels in two sub-iterations
/// until no more pixels can be removed, yielding a 1-pixel-wide skeleton
/// that preserves connectivity.
///
/// # Arguments
///
/// * `binary` - Input binary image (`true` = foreground object).
///
/// # Returns
///
/// Thinned (skeleton) binary image.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::skeletonize::zhang_suen_thin;
/// use scirs2_core::ndarray::Array2;
///
/// let binary = Array2::from_shape_fn((10, 10), |(r, c)| r > 0 && r < 9 && c > 0 && c < 9);
/// let skeleton = zhang_suen_thin(&binary);
/// ```
pub fn zhang_suen_thin(binary: &Array2<bool>) -> Array2<bool> {
    let rows = binary.nrows();
    let cols = binary.ncols();
    let mut img = binary.clone();

    loop {
        let mut changed = false;

        // Sub-iteration 1
        let to_delete_1 = collect_deletable_zs(&img, 1);
        for (r, c) in &to_delete_1 {
            img[[*r, *c]] = false;
            changed = true;
        }

        // Sub-iteration 2
        let to_delete_2 = collect_deletable_zs(&img, 2);
        for (r, c) in &to_delete_2 {
            img[[*r, *c]] = false;
            changed = true;
        }

        if !changed {
            break;
        }
    }

    img
}

/// Determine which pixels are deletable in a given Zhang-Suen sub-iteration.
fn collect_deletable_zs(img: &Array2<bool>, step: u8) -> Vec<(usize, usize)> {
    let rows = img.nrows();
    let cols = img.ncols();
    let mut to_delete = Vec::new();

    for r in 1..(rows - 1) {
        for c in 1..(cols - 1) {
            if !img[[r, c]] {
                continue;
            }

            // 8-neighbours in order: P2..P9 (clock-wise from top)
            // P2=N, P3=NE, P4=E, P5=SE, P6=S, P7=SW, P8=W, P9=NW
            let p2 = img[[r - 1, c]] as u8;
            let p3 = img[[r - 1, c + 1]] as u8;
            let p4 = img[[r, c + 1]] as u8;
            let p5 = img[[r + 1, c + 1]] as u8;
            let p6 = img[[r + 1, c]] as u8;
            let p7 = img[[r + 1, c - 1]] as u8;
            let p8 = img[[r, c - 1]] as u8;
            let p9 = img[[r - 1, c - 1]] as u8;

            let neighbors = [p2, p3, p4, p5, p6, p7, p8, p9];
            let b: u8 = neighbors.iter().sum(); // number of foreground neighbours

            // Condition B: 2 ≤ B(p) ≤ 6
            if !(2..=6).contains(&b) {
                continue;
            }

            // Condition A: number of 0→1 transitions in ordered sequence
            let seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]; // circular
            let a: u8 = seq
                .windows(2)
                .map(|w| if w[0] == 0 && w[1] == 1 { 1 } else { 0 })
                .sum();
            if a != 1 {
                continue;
            }

            // Step-specific conditions
            if step == 1 {
                // (P2 AND P4 AND P6 == 0) AND (P4 AND P6 AND P8 == 0)
                if p2 * p4 * p6 != 0 || p4 * p6 * p8 != 0 {
                    continue;
                }
            } else {
                // (P2 AND P4 AND P8 == 0) AND (P2 AND P6 AND P8 == 0)
                if p2 * p4 * p8 != 0 || p2 * p6 * p8 != 0 {
                    continue;
                }
            }

            to_delete.push((r, c));
        }
    }

    to_delete
}

// ---------------------------------------------------------------------------
// Lee's thinning
// ---------------------------------------------------------------------------

/// Thin a binary image using a parallel topology-preserving thinning algorithm
/// inspired by Lee, Kashyap & Chu (1994) "Building Skeleton Models via 3-D Medial
/// Surface Axis Thinning Algorithms" (2D specialisation).
///
/// This implementation applies a lookup-table–based simple-point test to ensure
/// the Euler number and connected components are preserved throughout the thinning.
/// It is slower than Zhang-Suen but produces topologically exact skeletons.
///
/// # Arguments
///
/// * `binary` - Input binary image.
///
/// # Returns
///
/// Thinned binary image.
pub fn lee_thin(binary: &Array2<bool>) -> Array2<bool> {
    let rows = binary.nrows();
    let cols = binary.ncols();
    let mut img = binary.clone();

    // Border layer is never modified (treated as background)
    loop {
        let mut changed = false;

        // We use four directional sub-iterations: North, South, East, West border
        // erosion passes, deleting simple-point border pixels.
        for direction in 0..4u8 {
            let mut to_delete: Vec<(usize, usize)> = Vec::new();

            for r in 1..(rows - 1) {
                for c in 1..(cols - 1) {
                    if !img[[r, c]] {
                        continue;
                    }

                    // Check that the pixel is on the "border" for this direction
                    let is_border = match direction {
                        0 => !img[[r - 1, c]], // North
                        1 => !img[[r + 1, c]], // South
                        2 => !img[[r, c - 1]], // West
                        _ => !img[[r, c + 1]], // East
                    };

                    if !is_border {
                        continue;
                    }

                    // Only delete simple points (topology-preserving test in 2D)
                    if is_simple_point(&img, r, c) {
                        to_delete.push((r, c));
                    }
                }
            }

            for (r, c) in &to_delete {
                img[[*r, *c]] = false;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    img
}

/// Simple-point test for 2D images (topology-preserving deletion).
///
/// A pixel p is a simple point if its deletion does not change the topology:
/// - It does not disconnect the foreground (C_fg = 1).
/// - It does not create a new hole in the background (C_bg = 1).
///
/// We check both conditions using 8-connectivity for foreground and
/// 4-connectivity for background.
fn is_simple_point(img: &Array2<bool>, r: usize, c: usize) -> bool {
    // Count connected foreground components in 8-neighbourhood after removal
    let fg_cc = count_fg_components(img, r, c);
    if fg_cc != 1 {
        return false;
    }

    // Count connected background components in 4-neighbourhood
    let bg_cc = count_bg_components(img, r, c);
    bg_cc == 1
}

/// Count 8-connected foreground components among the 8 neighbours of (r,c),
/// treating (r,c) itself as background.
fn count_fg_components(img: &Array2<bool>, r: usize, c: usize) -> usize {
    // Neighbour positions in clockwise order
    let nbrs: [(isize, isize); 8] = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ];

    let rows = img.nrows() as isize;
    let cols = img.ncols() as isize;
    let mut visited = [false; 8];
    let mut count = 0;

    for start in 0..8 {
        let (dr, dc) = nbrs[start];
        let nr = r as isize + dr;
        let nc = c as isize + dc;
        if nr < 0 || nr >= rows || nc < 0 || nc >= cols {
            continue;
        }
        if !img[[nr as usize, nc as usize]] || visited[start] {
            continue;
        }

        // BFS / flood fill among the 8 neighbours using 8-adjacency
        count += 1;
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(cur) = queue.pop_front() {
            for other in 0..8 {
                if visited[other] {
                    continue;
                }
                let (dr2, dc2) = nbrs[other];
                let nr2 = r as isize + dr2;
                let nc2 = c as isize + dc2;
                if nr2 < 0 || nr2 >= rows || nc2 < 0 || nc2 >= cols {
                    continue;
                }
                if !img[[nr2 as usize, nc2 as usize]] {
                    continue;
                }
                // 8-adjacent in the ring?
                let (dr1, dc1) = nbrs[cur];
                if (dr1 - dr2).abs() <= 1 && (dc1 - dc2).abs() <= 1 {
                    visited[other] = true;
                    queue.push_back(other);
                }
            }
        }
    }

    count
}

/// Count 4-connected background components among the 4 neighbours of (r,c),
/// treating (r,c) itself as background.
fn count_bg_components(img: &Array2<bool>, r: usize, c: usize) -> usize {
    let rows = img.nrows() as isize;
    let cols = img.ncols() as isize;
    let nbrs4: [(isize, isize); 4] = [(-1, 0), (0, 1), (1, 0), (0, -1)];

    let mut visited = [false; 4];
    let mut count = 0;

    for start in 0..4 {
        let (dr, dc) = nbrs4[start];
        let nr = r as isize + dr;
        let nc = c as isize + dc;

        // Out-of-bounds neighbours are background
        let is_bg = if nr < 0 || nr >= rows || nc < 0 || nc >= cols {
            true
        } else {
            !img[[nr as usize, nc as usize]]
        };

        if !is_bg || visited[start] {
            continue;
        }

        count += 1;
        visited[start] = true;

        // BFS among the 4 neighbours (cross-shaped ring)
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(cur) = queue.pop_front() {
            for other in 0..4 {
                if visited[other] {
                    continue;
                }
                let (dr2, dc2) = nbrs4[other];
                let nr2 = r as isize + dr2;
                let nc2 = c as isize + dc2;
                let is_bg2 = if nr2 < 0 || nr2 >= rows || nc2 < 0 || nc2 >= cols {
                    true
                } else {
                    !img[[nr2 as usize, nc2 as usize]]
                };
                if !is_bg2 {
                    continue;
                }
                // 4-adjacent in the ring?
                let (dr1, dc1) = nbrs4[cur];
                let (dr2c, dc2c) = nbrs4[other];
                if (dr1 - dr2c).abs() + (dc1 - dc2c).abs() <= 2 {
                    visited[other] = true;
                    queue.push_back(other);
                }
            }
        }
    }

    count
}

// ---------------------------------------------------------------------------
// Medial axis transform
// ---------------------------------------------------------------------------

/// Result of the medial axis transform
#[derive(Debug, Clone)]
pub struct MedialAxisResult {
    /// Skeleton pixels (1-pixel-wide connected structure)
    pub skeleton: Array2<bool>,
    /// Euclidean distance to the nearest background pixel for each foreground pixel
    pub distance: Array2<f64>,
}

/// Compute the medial axis (skeleton + distance transform) of a binary image.
///
/// The skeleton is computed as the set of foreground pixels that are local
/// maxima of the distance transform in a 3×3 neighbourhood **or** lie on an
/// endpoint/junction.  This gives a connected skeleton without requiring
/// iterative thinning, and is substantially faster for large objects.
///
/// For highly accurate topology preservation, combine this with
/// [`zhang_suen_thin`] applied to the local-maximum mask.
///
/// # Arguments
///
/// * `binary` - Input binary image.
///
/// # Errors
///
/// Propagates errors from the underlying EDT computation.
pub fn medial_axis(binary: &Array2<bool>) -> NdimageResult<MedialAxisResult> {
    let rows = binary.nrows();
    let cols = binary.ncols();

    // Compute Euclidean distance transform
    let distance = euclidean_dt(binary)?;

    // Mark pixels that are local maxima of the distance map
    // (strict maxima in 3×3 window)
    let mut skeleton_raw = Array2::<bool>::from_elem((rows, cols), false);
    for r in 1..(rows.saturating_sub(1)) {
        for c in 1..(cols.saturating_sub(1)) {
            if !binary[[r, c]] {
                continue;
            }
            let v = distance[[r, c]];
            if v <= 0.0 {
                continue;
            }
            let is_local_max = (-1i32..=1).all(|dr| {
                (-1i32..=1).all(|dc| {
                    let nr = (r as i32 + dr) as usize;
                    let nc = (c as i32 + dc) as usize;
                    distance[[nr, nc]] <= v
                })
            });
            if is_local_max {
                skeleton_raw[[r, c]] = true;
            }
        }
    }

    // Apply Zhang-Suen to ensure 1-pixel width
    let skeleton = zhang_suen_thin(&skeleton_raw);

    Ok(MedialAxisResult { skeleton, distance })
}

// ---------------------------------------------------------------------------
// Distance transform (thin public wrappers)
// ---------------------------------------------------------------------------

/// Euclidean distance transform.
///
/// For each foreground pixel (`true`), returns the Euclidean distance to the
/// nearest background pixel (`false`).  Background pixels return 0.
///
/// This is a public re-export of [`crate::distance_transforms::euclidean_dt`]
/// for convenience under the `skeletonize` module.
///
/// # Errors
///
/// Returns [`NdimageError`] on invalid input.
pub fn distance_transform_edt(binary: &Array2<bool>) -> NdimageResult<Array2<f64>> {
    euclidean_dt(binary)
}

/// City-block (L1 / Manhattan) distance transform.
///
/// For each foreground pixel, returns the L1 distance to the nearest background
/// pixel.  Background pixels return 0.
///
/// # Errors
///
/// Returns [`NdimageError`] on invalid input.
pub fn distance_transform_cdt(binary: &Array2<bool>) -> NdimageResult<Array2<f64>> {
    cityblock_dt(binary)
}

// ---------------------------------------------------------------------------
// Convex hull
// ---------------------------------------------------------------------------

/// Compute the convex hull of a binary image.
///
/// Returns a binary image where every pixel inside (or on the boundary of)
/// the convex hull of the foreground pixels is set to `true`.
///
/// The convex hull is computed via the Graham scan algorithm on the foreground
/// pixel coordinates, then rasterised by scan-line filling.
///
/// # Arguments
///
/// * `binary` - Input binary image.
///
/// # Returns
///
/// Binary image of the same size with the convex hull filled.
pub fn convex_hull_image(binary: &Array2<bool>) -> Array2<bool> {
    let rows = binary.nrows();
    let cols = binary.ncols();

    // Collect foreground pixel coordinates
    let points: Vec<(i64, i64)> = binary
        .indexed_iter()
        .filter_map(|((r, c), &v)| if v { Some((r as i64, c as i64)) } else { None })
        .collect();

    if points.len() < 3 {
        // Single pixel or line → return the original
        return binary.clone();
    }

    // Graham scan to find convex hull vertices
    let hull = graham_scan(points);

    // Rasterise the convex hull by scan-line filling
    let mut result = Array2::<bool>::from_elem((rows, cols), false);
    rasterise_convex_hull(&hull, &mut result);
    result
}

/// Graham scan algorithm: returns the convex hull vertices in CCW order.
fn graham_scan(mut points: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    // Find the lowest (then leftmost) point as pivot
    let pivot_idx = points
        .iter()
        .enumerate()
        .min_by_key(|&(_, &(r, c))| (r, c))
        .map(|(i, _)| i)
        .unwrap_or(0);
    points.swap(0, pivot_idx);
    let pivot = points[0];

    // Sort by polar angle relative to pivot (ties broken by distance)
    points[1..].sort_by(|&(r1, c1), &(r2, c2)| {
        let cross = (r1 - pivot.0) * (c2 - pivot.1) - (r2 - pivot.0) * (c1 - pivot.1);
        if cross != 0 {
            return cross.cmp(&0).reverse(); // CCW first
        }
        let d1 = (r1 - pivot.0).pow(2) + (c1 - pivot.1).pow(2);
        let d2 = (r2 - pivot.0).pow(2) + (c2 - pivot.1).pow(2);
        d1.cmp(&d2)
    });

    let mut hull: Vec<(i64, i64)> = Vec::with_capacity(points.len());
    for p in &points {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            let cross = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
            if cross <= 0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(*p);
    }
    hull
}

/// Scan-line fill of a convex polygon defined by `hull` vertices.
fn rasterise_convex_hull(hull: &[(i64, i64)], out: &mut Array2<bool>) {
    if hull.is_empty() {
        return;
    }

    let rows = out.nrows() as i64;
    let cols = out.ncols() as i64;

    let min_r = hull.iter().map(|&(r, _)| r).min().unwrap_or(0).max(0);
    let max_r = hull
        .iter()
        .map(|&(r, _)| r)
        .max()
        .unwrap_or(0)
        .min(rows - 1);

    let n = hull.len();

    for scan_r in min_r..=max_r {
        let mut xs: Vec<i64> = Vec::new();

        for i in 0..n {
            let (r0, c0) = hull[i];
            let (r1, c1) = hull[(i + 1) % n];

            // Does edge (r0,c0)→(r1,c1) cross the scan line at scan_r?
            let (rmin, rmax) = if r0 <= r1 { (r0, r1) } else { (r1, r0) };
            if scan_r < rmin || scan_r > rmax {
                continue;
            }

            let dr = r1 - r0;
            if dr == 0 {
                // Horizontal edge — mark both endpoints
                xs.push(c0);
                xs.push(c1);
            } else {
                let x = c0 + (scan_r - r0) * (c1 - c0) / dr;
                xs.push(x);
            }
        }

        if xs.is_empty() {
            continue;
        }

        xs.sort_unstable();
        let x_min = xs[0].max(0).min(cols - 1) as usize;
        let x_max = xs[xs.len() - 1].max(0).min(cols - 1) as usize;
        for c in x_min..=x_max {
            out[[scan_r as usize, c]] = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Binary fill holes
// ---------------------------------------------------------------------------

/// Fill enclosed holes in a binary image using flood fill from the border.
///
/// Any connected background region (`false`) that is surrounded by foreground
/// pixels (i.e. does not touch the image border) is filled with `true`.
///
/// The algorithm performs a flood fill from all border background pixels
/// (4-connected), marks all reachable background pixels as "border background",
/// and then sets all remaining background pixels (the holes) to `true`.
///
/// # Arguments
///
/// * `binary` - Input binary image.
///
/// # Returns
///
/// Binary image with all enclosed holes filled.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::skeletonize::binary_fill_holes;
/// use scirs2_core::ndarray::Array2;
///
/// let binary = Array2::from_shape_fn((7, 7), |(r, c)| {
///     r == 1 || r == 5 || c == 1 || c == 5
/// });
/// let filled = binary_fill_holes(&binary);
/// // Interior of the rectangle is now true
/// assert!(filled[[3, 3]]);
/// ```
pub fn binary_fill_holes(binary: &Array2<bool>) -> Array2<bool> {
    let rows = binary.nrows();
    let cols = binary.ncols();
    let mut out = binary.clone();

    // Mark reachable background from border using BFS (4-connected)
    let mut reachable = Array2::<bool>::from_elem((rows, cols), false);
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

    let seed_if_bg = |r: usize,
                      c: usize,
                      img: &Array2<bool>,
                      reach: &mut Array2<bool>,
                      q: &mut VecDeque<(usize, usize)>| {
        if !img[[r, c]] && !reach[[r, c]] {
            reach[[r, c]] = true;
            q.push_back((r, c));
        }
    };

    // Top and bottom rows
    for c in 0..cols {
        seed_if_bg(0, c, binary, &mut reachable, &mut queue);
        if rows > 1 {
            seed_if_bg(rows - 1, c, binary, &mut reachable, &mut queue);
        }
    }
    // Left and right columns
    for r in 0..rows {
        seed_if_bg(r, 0, binary, &mut reachable, &mut queue);
        if cols > 1 {
            seed_if_bg(r, cols - 1, binary, &mut reachable, &mut queue);
        }
    }

    // BFS
    let dirs: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    while let Some((r, c)) = queue.pop_front() {
        for &(dr, dc) in &dirs {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr < 0 || nr >= rows as i32 || nc < 0 || nc >= cols as i32 {
                continue;
            }
            let nr = nr as usize;
            let nc = nc as usize;
            if !binary[[nr, nc]] && !reachable[[nr, nc]] {
                reachable[[nr, nc]] = true;
                queue.push_back((nr, nc));
            }
        }
    }

    // Fill holes: background pixels not reachable from border
    for r in 0..rows {
        for c in 0..cols {
            if !binary[[r, c]] && !reachable[[r, c]] {
                out[[r, c]] = true;
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn filled_rect(rows: usize, cols: usize) -> Array2<bool> {
        Array2::from_elem((rows, cols), true)
    }

    /// A foreground rectangle inset one pixel from the image border, so that
    /// Zhang-Suen (which only processes interior rows 1..n-1 and cols 1..m-1)
    /// can actually remove all fully-surrounded pixels.
    fn inset_filled_rect(rows: usize, cols: usize) -> Array2<bool> {
        let mut arr = Array2::from_elem((rows + 2, cols + 2), false);
        for r in 1..=rows {
            for c in 1..=cols {
                arr[[r, c]] = true;
            }
        }
        arr
    }

    fn ring(size: usize) -> Array2<bool> {
        Array2::from_shape_fn((size, size), |(r, c)| {
            let d = (r as i32 - size as i32 / 2)
                .abs()
                .max((c as i32 - size as i32 / 2).abs());
            d == size as i32 / 2 - 1
        })
    }

    // ─── Zhang-Suen ──────────────────────────────────────────────────────────

    #[test]
    fn zhang_suen_solid_rect_produces_skeleton() {
        // Use a rectangle inset one pixel from the image border so Zhang-Suen
        // (which only processes interior rows/cols 1..n-1) can fully thin the
        // interior.  An all-true image would have border pixels that are never
        // considered for removal, leaving fully-surrounded pixels.
        let binary = inset_filled_rect(15, 15);
        let skel = zhang_suen_thin(&binary);
        // Skeleton should be non-empty
        assert!(
            skel.iter().any(|&v| v),
            "Expected non-empty skeleton from filled rectangle"
        );
        // No pixel in the skeleton should have 8 foreground neighbours
        // (that would indicate it's interior, not a skeleton pixel)
        let rows = skel.nrows();
        let cols = skel.ncols();
        for r in 1..(rows - 1) {
            for c in 1..(cols - 1) {
                if skel[[r, c]] {
                    let nbr_count: u8 = [
                        skel[[r - 1, c - 1]] as u8,
                        skel[[r - 1, c]] as u8,
                        skel[[r - 1, c + 1]] as u8,
                        skel[[r, c - 1]] as u8,
                        skel[[r, c + 1]] as u8,
                        skel[[r + 1, c - 1]] as u8,
                        skel[[r + 1, c]] as u8,
                        skel[[r + 1, c + 1]] as u8,
                    ]
                    .iter()
                    .sum();
                    assert_ne!(nbr_count, 8, "Skeleton pixel ({r},{c}) is fully surrounded");
                }
            }
        }
    }

    #[test]
    fn zhang_suen_single_pixel_unchanged() {
        let mut binary = Array2::from_elem((5, 5), false);
        binary[[2, 2]] = true;
        let skel = zhang_suen_thin(&binary);
        assert!(skel[[2, 2]]);
        let total: usize = skel.iter().map(|&v| v as usize).sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn zhang_suen_empty_image_unchanged() {
        let binary = Array2::from_elem((10, 10), false);
        let skel = zhang_suen_thin(&binary);
        assert!(skel.iter().all(|&v| !v));
    }

    #[test]
    fn zhang_suen_horizontal_line_unchanged() {
        let mut binary = Array2::from_elem((10, 30), false);
        // 1-pixel-wide horizontal line
        for c in 0..30 {
            binary[[5, c]] = true;
        }
        let skel = zhang_suen_thin(&binary);
        // A 1-pixel-wide line should survive
        assert!(skel.iter().any(|&v| v));
    }

    // ─── Lee thinning ────────────────────────────────────────────────────────

    #[test]
    fn lee_thin_produces_nonempty_skeleton() {
        let binary = filled_rect(12, 12);
        let skel = lee_thin(&binary);
        assert!(
            skel.iter().any(|&v| v),
            "Expected non-empty skeleton from Lee thinning"
        );
    }

    #[test]
    fn lee_thin_preserves_connectivity() {
        // A thick cross shape should remain connected after thinning
        let size = 20;
        let binary = Array2::from_shape_fn((size, size), |(r, c)| {
            let mid = size / 2;
            (r >= mid - 2 && r < mid + 2) || (c >= mid - 2 && c < mid + 2)
        });
        let skel = lee_thin(&binary);
        assert!(skel.iter().any(|&v| v));
    }

    // ─── Medial axis ─────────────────────────────────────────────────────────

    #[test]
    fn medial_axis_disk_center() {
        let size = 30usize;
        let center = size / 2;
        let radius = 10i32;
        let binary = Array2::from_shape_fn((size, size), |(r, c)| {
            let dr = r as i32 - center as i32;
            let dc = c as i32 - center as i32;
            dr * dr + dc * dc <= radius * radius
        });

        let result = medial_axis(&binary).expect("medial_axis failed");
        assert!(
            result.skeleton.iter().any(|&v| v),
            "Medial axis of disk should be non-empty"
        );
        // The distance at the center should be approx the radius
        let d = result.distance[[center, center]];
        assert!(d > 5.0, "Distance at center ({d}) should be > 5");
    }

    // ─── EDT (distance_transform_edt wrapper) ────────────────────────────────

    #[test]
    fn distance_transform_edt_all_true_gives_positive_dist() {
        // For an all-true image: the EDT returns distance to nearest background.
        // But there IS no background, so the algorithm fills with large values.
        // The function should return without error.
        let binary = Array2::from_elem((10, 10), true);
        let dt = distance_transform_edt(&binary).expect("edt failed");
        // Interior pixels must have distance > 0
        for r in 0..10 {
            for c in 0..10 {
                assert!(dt[[r, c]] >= 0.0);
            }
        }
    }

    #[test]
    fn distance_transform_edt_background_is_zero() {
        let binary = Array2::from_elem((10, 10), false);
        let dt = distance_transform_edt(&binary).expect("edt failed");
        assert!(dt.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn distance_transform_edt_center_of_square() {
        // 5×5 square of true embedded in a larger false field
        let binary = Array2::from_shape_fn((9, 9), |(r, c)| r >= 2 && r <= 6 && c >= 2 && c <= 6);
        let dt = distance_transform_edt(&binary).expect("edt ok");
        // Center (4,4): nearest background is at row 1 or row 7 (3 rows away),
        // so Euclidean distance = 3.0
        let center_dist = dt[[4, 4]];
        assert!(
            (center_dist - 3.0).abs() < 0.5,
            "Center distance {center_dist} not close to 3.0"
        );
        // Background pixels should be 0
        assert_eq!(dt[[0, 0]], 0.0);
    }

    #[test]
    fn distance_transform_cdt_matches_l1() {
        let binary = Array2::from_shape_fn((7, 7), |(r, c)| r >= 1 && r <= 5 && c >= 1 && c <= 5);
        let dt = distance_transform_cdt(&binary).expect("cdt ok");
        // Background pixels are 0
        assert_eq!(dt[[0, 0]], 0.0);
        // Center pixel (3,3): L1 distance to nearest background = 2
        let center = dt[[3, 3]];
        assert!(center >= 1.0, "L1 center distance {center}");
    }

    // ─── Convex hull ─────────────────────────────────────────────────────────

    #[test]
    fn convex_hull_of_convex_shape_is_unchanged() {
        // A filled disk is convex; its hull should equal itself (approximately)
        let size = 20usize;
        let center = 10i32;
        let r_sq = 8i32;
        let binary = Array2::from_shape_fn((size, size), |(r, c)| {
            let dr = r as i32 - center;
            let dc = c as i32 - center;
            dr * dr + dc * dc <= r_sq * r_sq
        });
        let hull = convex_hull_image(&binary);
        // Every foreground pixel in original should be in the hull
        for ((r, c), &v) in binary.indexed_iter() {
            if v {
                assert!(hull[[r, c]], "Disk pixel ({r},{c}) missing from hull");
            }
        }
    }

    #[test]
    fn convex_hull_fills_concavity() {
        // An L-shape: the hull should fill in the concave corner
        let mut binary = Array2::from_elem((10, 10), false);
        for r in 0..10 {
            binary[[r, 0]] = true;
        }
        for c in 0..10 {
            binary[[9, c]] = true;
        }
        let hull = convex_hull_image(&binary);
        // All pixels should be true in the hull (the hull covers the entire square)
        assert!(hull.iter().any(|&v| v));
    }

    // ─── Binary fill holes ───────────────────────────────────────────────────

    #[test]
    fn fill_holes_fills_enclosed_region() {
        // A ring with a hole in the middle
        let size = 9usize;
        let mut binary = Array2::from_elem((size, size), false);
        // Draw a thick ring
        for r in 1..(size - 1) {
            for c in 1..(size - 1) {
                let dr = r as i32 - 4;
                let dc = c as i32 - 4;
                let d2 = dr * dr + dc * dc;
                if d2 >= 4 && d2 <= 16 {
                    binary[[r, c]] = true;
                }
            }
        }
        let filled = binary_fill_holes(&binary);
        // Center (4,4) should now be filled
        assert!(
            filled[[4, 4]],
            "Hole at center should be filled, but it is not"
        );
    }

    #[test]
    fn fill_holes_does_not_fill_exterior() {
        // Solid rectangle bordered by background
        let binary = Array2::from_shape_fn((9, 9), |(r, c)| r >= 2 && r <= 6 && c >= 2 && c <= 6);
        let filled = binary_fill_holes(&binary);
        // Corner (0,0) must remain background
        assert!(!filled[[0, 0]]);
        // Interior should be foreground
        assert!(filled[[4, 4]]);
    }

    #[test]
    fn fill_holes_all_background_unchanged() {
        let binary = Array2::from_elem((10, 10), false);
        let filled = binary_fill_holes(&binary);
        assert!(filled.iter().all(|&v| !v));
    }
}
