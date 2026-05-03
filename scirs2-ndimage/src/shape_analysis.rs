//! Shape analysis: convex hull, contour extraction, shape descriptors, ellipse fitting,
//! and minimum bounding rectangle.
//!
//! # Overview
//!
//! This module provides geometric shape analysis tools operating on 2-D binary images
//! and point clouds:
//!
//! - **Convex hull** (Graham scan) for arbitrary 2-D point sets.
//! - **Contour extraction** using Moore-neighbourhood (8-connectivity) tracing.
//! - **Shape descriptors**: area, perimeter, circularity, eccentricity, aspect ratio,
//!   extent, solidity, convexity.
//! - **Ellipse fitting** via the algebraic least-squares method (Fitzgibbon et al.).
//! - **Minimum bounding rectangle** using the rotating calipers technique.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array2, ArrayView2};

// ─────────────────────────────────────────────────────────────────────────────
// Convex Hull – Graham Scan
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the convex hull of a 2-D point set using the Graham scan algorithm.
///
/// Returns the hull vertices in counter-clockwise order.  If fewer than 3 distinct
/// points are supplied the function returns the (deduplicated) input sorted
/// lexicographically so callers always receive a valid polygon-like result.
///
/// # Arguments
///
/// * `points` – Slice of (x, y) coordinates.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when the input is empty.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::shape_analysis::convex_hull_2d;
///
/// let pts = vec![(0.0_f64, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)];
/// let hull = convex_hull_2d(&pts).unwrap();
/// assert_eq!(hull.len(), 4);
/// ```
pub fn convex_hull_2d(points: &[(f64, f64)]) -> NdimageResult<Vec<(f64, f64)>> {
    if points.is_empty() {
        return Err(NdimageError::InvalidInput(
            "convex_hull_2d: point set is empty".to_string(),
        ));
    }

    // Deduplicate
    let mut pts: Vec<(f64, f64)> = points.to_vec();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12);

    if pts.len() < 3 {
        return Ok(pts);
    }

    // Cross product of vectors OA and OB
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };

    let n = pts.len();
    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(2 * n);

    // Lower hull
    for &p in &pts {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }

    // Upper hull
    let lower_len = hull.len() + 1;
    for &p in pts.iter().rev() {
        while hull.len() >= lower_len && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }

    hull.pop(); // Remove last point (same as first)
    Ok(hull)
}

// ─────────────────────────────────────────────────────────────────────────────
// Contour Extraction – Moore Neighbourhood
// ─────────────────────────────────────────────────────────────────────────────

/// Extract outer contours from a binary image.
///
/// Returns one contour per 4-connected foreground component.  Each contour is
/// the ordered sequence of boundary pixels (those with at least one 8-connected
/// background neighbour), collected by BFS through the 8-connected boundary
/// ring of the component.
///
/// # Arguments
///
/// * `binary` – 2-D boolean array where `true` denotes the foreground.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::shape_analysis::contour_extraction;
/// use scirs2_core::ndarray::Array2;
///
/// let mut img = Array2::<bool>::default((6, 6));
/// for r in 1..5 { for c in 1..5 { img[[r, c]] = true; } }
/// let contours = contour_extraction(&img.view()).unwrap();
/// assert_eq!(contours.len(), 1);
/// assert!(!contours[0].is_empty());
/// ```
pub fn contour_extraction(binary: &ArrayView2<bool>) -> NdimageResult<Vec<Vec<(usize, usize)>>> {
    use std::collections::VecDeque;

    let rows = binary.nrows();
    let cols = binary.ncols();

    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }

    // 8-connected neighbourhood offsets
    const N8: [(i32, i32); 8] = [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ];
    // 4-connected neighbourhood offsets
    const N4: [(i32, i32); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];

    let in_bounds = |r: i32, c: i32| r >= 0 && r < rows as i32 && c >= 0 && c < cols as i32;

    let get = |r: usize, c: usize| -> bool { *binary.get((r, c)).unwrap_or(&false) };

    // is_boundary: true if foreground pixel has at least one background 8-neighbour
    let is_boundary = |r: usize, c: usize| -> bool {
        N8.iter().any(|(dr, dc)| {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            !in_bounds(nr, nc) || !get(nr as usize, nc as usize)
        })
    };

    // Mark which foreground pixels have been fully processed
    let mut fg_visited = Array2::<bool>::default((rows, cols));
    let mut contours: Vec<Vec<(usize, usize)>> = Vec::new();

    for start_r in 0..rows {
        for start_c in 0..cols {
            if !get(start_r, start_c) || fg_visited[[start_r, start_c]] {
                continue;
            }

            // BFS to collect all 4-connected foreground pixels of this component
            let mut component: Vec<(usize, usize)> = Vec::new();
            let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
            queue.push_back((start_r, start_c));
            fg_visited[[start_r, start_c]] = true;

            while let Some((r, c)) = queue.pop_front() {
                component.push((r, c));
                for (dr, dc) in &N4 {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if in_bounds(nr, nc) {
                        let nr = nr as usize;
                        let nc = nc as usize;
                        if get(nr, nc) && !fg_visited[[nr, nc]] {
                            fg_visited[[nr, nc]] = true;
                            queue.push_back((nr, nc));
                        }
                    }
                }
            }

            // From the component, collect boundary pixels and order them by
            // BFS traversal through the 8-connected boundary ring.
            let mut boundary_visited = Array2::<bool>::default((rows, cols));
            let mut contour: Vec<(usize, usize)> = Vec::new();

            // Find the first boundary pixel (top-left scanning order)
            let first_boundary = component.iter().find(|&&(r, c)| is_boundary(r, c));

            if let Some(&(br, bc)) = first_boundary {
                let mut bq: VecDeque<(usize, usize)> = VecDeque::new();
                bq.push_back((br, bc));
                boundary_visited[[br, bc]] = true;

                while let Some((r, c)) = bq.pop_front() {
                    contour.push((r, c));
                    for (dr, dc) in &N8 {
                        let nr = r as i32 + dr;
                        let nc = c as i32 + dc;
                        if in_bounds(nr, nc) {
                            let nr = nr as usize;
                            let nc = nc as usize;
                            if get(nr, nc) && is_boundary(nr, nc) && !boundary_visited[[nr, nc]] {
                                boundary_visited[[nr, nc]] = true;
                                bq.push_back((nr, nc));
                            }
                        }
                    }
                }
            }

            if !contour.is_empty() {
                contours.push(contour);
            }
        }
    }

    Ok(contours)
}

// ─────────────────────────────────────────────────────────────────────────────
// Shape Descriptors
// ─────────────────────────────────────────────────────────────────────────────

/// Collection of scalar shape descriptors computed from a contour.
#[derive(Debug, Clone)]
pub struct ShapeDescriptors {
    /// Area enclosed by the contour (shoelace formula, pixels²).
    pub area: f64,
    /// Perimeter (sum of Euclidean distances between consecutive contour points).
    pub perimeter: f64,
    /// Circularity: 4π·area / perimeter².  Perfect circle → 1.
    pub circularity: f64,
    /// Eccentricity of the best-fit ellipse (0 = circle, 1 = line).
    pub eccentricity: f64,
    /// Aspect ratio: major_axis / minor_axis of the bounding box.
    pub aspect_ratio: f64,
    /// Extent: area / bounding_box_area.
    pub extent: f64,
    /// Solidity: area / convex_hull_area.
    pub solidity: f64,
    /// Convexity: convex_hull_perimeter / perimeter.
    pub convexity: f64,
}

/// Compute shape descriptors for a contour given as (row, col) pixel coordinates.
///
/// # Arguments
///
/// * `contour` – Ordered list of boundary pixels (row, col).
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when the contour has fewer than 3 points.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::shape_analysis::shape_descriptors;
///
/// let contour: Vec<(usize, usize)> = vec![(0,0),(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),(1,0)];
/// let d = shape_descriptors(&contour).unwrap();
/// assert!(d.area > 0.0);
/// assert!(d.circularity > 0.0 && d.circularity <= 1.0 + 1e-6);
/// ```
pub fn shape_descriptors(contour: &[(usize, usize)]) -> NdimageResult<ShapeDescriptors> {
    if contour.len() < 3 {
        return Err(NdimageError::InvalidInput(
            "shape_descriptors: contour must have at least 3 points".to_string(),
        ));
    }

    let pts_f: Vec<(f64, f64)> = contour
        .iter()
        .map(|&(r, c)| (c as f64, r as f64)) // (x, y)
        .collect();

    // --- Area (shoelace) ---
    let area = shoelace_area(&pts_f).abs();

    // --- Perimeter ---
    let n = pts_f.len();
    let perimeter: f64 = pts_f
        .iter()
        .zip(pts_f.iter().cycle().skip(1))
        .take(n)
        .map(|(&(x0, y0), &(x1, y1))| ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt())
        .sum();

    // --- Circularity ---
    let circularity = if perimeter > 1e-12 {
        4.0 * std::f64::consts::PI * area / perimeter.powi(2)
    } else {
        0.0
    };

    // --- Bounding box ---
    let (min_x, max_x, min_y, max_y) = pts_f.iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |(mnx, mxx, mny, mxy), &(x, y)| (mnx.min(x), mxx.max(x), mny.min(y), mxy.max(y)),
    );
    let bbox_w = (max_x - min_x).max(1.0);
    let bbox_h = (max_y - min_y).max(1.0);
    let bbox_area = bbox_w * bbox_h;

    // --- Aspect ratio ---
    let aspect_ratio = if bbox_h > 1e-12 { bbox_w / bbox_h } else { 1.0 };

    // --- Extent ---
    let extent = area / bbox_area;

    // --- Convex hull ---
    let hull = convex_hull_2d(&pts_f).unwrap_or_default();

    // Convex hull area
    let hull_area = if hull.len() >= 3 {
        shoelace_area(&hull).abs()
    } else {
        area
    };

    // Convex hull perimeter
    let hull_perim: f64 = if hull.len() >= 2 {
        let hn = hull.len();
        hull.iter()
            .zip(hull.iter().cycle().skip(1))
            .take(hn)
            .map(|(&(x0, y0), &(x1, y1))| ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt())
            .sum()
    } else {
        perimeter
    };

    // --- Solidity ---
    let solidity = if hull_area > 1e-12 {
        (area / hull_area).min(1.0)
    } else {
        1.0
    };

    // --- Convexity ---
    let convexity = if perimeter > 1e-12 {
        (hull_perim / perimeter).min(1.0)
    } else {
        1.0
    };

    // --- Eccentricity via second-order central moments ---
    let (cx, cy) = pts_f
        .iter()
        .fold((0.0f64, 0.0f64), |(ax, ay), &(x, y)| (ax + x, ay + y));
    let cx = cx / n as f64;
    let cy = cy / n as f64;

    let (mu20, mu02, mu11) =
        pts_f
            .iter()
            .fold((0.0f64, 0.0f64, 0.0f64), |(m20, m02, m11), &(x, y)| {
                let dx = x - cx;
                let dy = y - cy;
                (m20 + dx * dx, m02 + dy * dy, m11 + dx * dy)
            });
    let mu20 = mu20 / n as f64;
    let mu02 = mu02 / n as f64;
    let mu11 = mu11 / n as f64;

    let diff = mu20 - mu02;
    let discriminant = (diff * diff + 4.0 * mu11 * mu11).sqrt();
    let lambda1 = (mu20 + mu02 + discriminant) / 2.0;
    let lambda2 = ((mu20 + mu02 - discriminant) / 2.0).max(0.0);

    let eccentricity = if lambda1 > 1e-12 {
        (1.0 - lambda2 / lambda1).sqrt()
    } else {
        0.0
    };

    Ok(ShapeDescriptors {
        area,
        perimeter,
        circularity: circularity.min(1.0 + 1e-9),
        eccentricity,
        aspect_ratio,
        extent: extent.min(1.0),
        solidity,
        convexity,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Ellipse Fitting (Fitzgibbon algebraic least-squares)
// ─────────────────────────────────────────────────────────────────────────────

/// Fit an ellipse to a set of 2-D points using the Fitzgibbon–Pilu–Fisher
/// algebraic least-squares method.
///
/// Returns `(cx, cy, a, b, angle_rad)` where:
/// - `(cx, cy)` is the ellipse centre,
/// - `a` ≥ `b` are the semi-axes,
/// - `angle_rad` is the tilt of the major axis (radians, measured from the
///    positive x-axis counter-clockwise).
///
/// # Arguments
///
/// * `points` – At least 5 (x, y) sample points.
///
/// # Errors
///
/// - [`NdimageError::InvalidInput`] if fewer than 5 points are given.
/// - [`NdimageError::ComputationError`] if the SVD / eigenvalue solver fails.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::shape_analysis::ellipse_fit;
/// use std::f64::consts::PI;
///
/// // Generate points on a 3×2 axis-aligned ellipse
/// let pts: Vec<(f64, f64)> = (0..36)
///     .map(|i| {
///         let t = i as f64 * PI / 18.0;
///         (3.0 * t.cos(), 2.0 * t.sin())
///     })
///     .collect();
/// let (cx, cy, a, b, _angle) = ellipse_fit(&pts).unwrap();
/// assert!((cx).abs() < 0.1);
/// assert!((cy).abs() < 0.1);
/// assert!((a - 3.0).abs() < 0.1);
/// assert!((b - 2.0).abs() < 0.1);
/// ```
pub fn ellipse_fit(points: &[(f64, f64)]) -> NdimageResult<(f64, f64, f64, f64, f64)> {
    if points.len() < 5 {
        return Err(NdimageError::InvalidInput(
            "ellipse_fit: at least 5 points are required".to_string(),
        ));
    }

    let n = points.len();

    // Build the design matrix D (n × 6), columns: x², xy, y², x, y, 1
    // Scatter matrix S = D'D
    let mut s = [[0.0f64; 6]; 6];
    for &(x, y) in points {
        let row = [x * x, x * y, y * y, x, y, 1.0];
        for i in 0..6 {
            for j in 0..6 {
                s[i][j] += row[i] * row[j];
            }
        }
    }

    // Constraint matrix C (only C[0][2], C[1][1], C[2][0] are non-zero for the
    // "bookstein" ellipse-specific constraint 4ac - b² = 1)
    // We use the eigenvector approach: solve (S⁻¹ C) v = λ v, pick the eigenvector
    // with the smallest positive eigenvalue.
    //
    // For numerical stability we use the reduced 3×3 system (Bookstein constraint).
    // The Fitzgibbon direct method:
    //   C = [[0,0,2],[0,-1,0],[2,0,0]]  (top-left 3×3) and zeros elsewhere.

    // Partition S into blocks
    // S = [S11 S12; S21 S22]  where S11, S12, S21, S22 are 3×3
    let s11 = sub_matrix_3x3(&s, 0, 0);
    let s12 = sub_matrix_3x3(&s, 0, 3);
    let s21 = sub_matrix_3x3(&s, 3, 0);
    let s22 = sub_matrix_3x3(&s, 3, 3);

    let s22_inv = mat3_inv(&s22).ok_or_else(|| {
        NdimageError::ComputationError("ellipse_fit: singular scatter sub-matrix".to_string())
    })?;

    // Reduced system M = C11^{-1} (S11 - S12 S22^{-1} S21)
    let tmp = mat3_mul(&s12, &mat3_mul(&s22_inv, &s21));
    let m_raw = mat3_sub(&s11, &tmp);

    // C11 = [[0,0,2],[0,-1,0],[2,0,0]], C11^{-1} = [[0,0,0.5],[0,-1,0],[0.5,0,0]]
    // M = C11^{-1} * m_raw
    let m = [
        [0.5 * m_raw[2][0], 0.5 * m_raw[2][1], 0.5 * m_raw[2][2]],
        [-m_raw[1][0], -m_raw[1][1], -m_raw[1][2]],
        [0.5 * m_raw[0][0], 0.5 * m_raw[0][1], 0.5 * m_raw[0][2]],
    ];

    // Find eigenvalues / eigenvectors of M (3×3 via characteristic polynomial)
    let (eigenvalues, eigenvectors) = mat3_eigen(&m);

    // Pick eigenvector with smallest positive eigenvalue where 4ac - b² > 0
    let mut best: Option<[f64; 3]> = None;
    let mut best_eval = f64::INFINITY;
    for i in 0..3 {
        let ev = eigenvalues[i];
        if ev.is_finite() && ev > -1e-10 {
            let v = eigenvectors[i];
            let cond = 4.0 * v[0] * v[2] - v[1] * v[1];
            if cond > 0.0 && ev < best_eval {
                best_eval = ev;
                best = Some(v);
            }
        }
    }

    let a1 = best.ok_or_else(|| {
        NdimageError::ComputationError(
            "ellipse_fit: no valid ellipse eigenvector found".to_string(),
        )
    })?;

    // Recover full 6-vector: [a, b, c, d, e, f]
    // a2 = -S22^{-1} S21 a1
    let neg_s22inv_s21_a1 = mat3_mul_vec(&mat3_mul(&s22_inv, &s21), &a1);
    let coeffs = [
        a1[0],
        a1[1],
        a1[2],
        -neg_s22inv_s21_a1[0],
        -neg_s22inv_s21_a1[1],
        -neg_s22inv_s21_a1[2],
    ];

    // Convert general conic Ax²+Bxy+Cy²+Dx+Ey+F=0 to geometric parameters
    conic_to_ellipse(coeffs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimum Bounding Rectangle – Rotating Calipers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the minimum-area bounding rectangle of a 2-D point set.
///
/// Returns the four corners of the rectangle in counter-clockwise order.
///
/// Uses the rotating calipers algorithm on the convex hull.
///
/// # Arguments
///
/// * `points` – At least one (x, y) point.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when the input is empty.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::shape_analysis::minimum_bounding_rectangle;
///
/// let pts = vec![(0.0_f64, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)];
/// let rect = minimum_bounding_rectangle(&pts).unwrap();
/// // Area should be close to 12
/// let area = {
///     let (dx1, dy1) = (rect[1].0 - rect[0].0, rect[1].1 - rect[0].1);
///     let (dx2, dy2) = (rect[3].0 - rect[0].0, rect[3].1 - rect[0].1);
///     let l1 = (dx1*dx1 + dy1*dy1).sqrt();
///     let l2 = (dx2*dx2 + dy2*dy2).sqrt();
///     l1 * l2
/// };
/// assert!((area - 12.0).abs() < 0.5);
/// ```
pub fn minimum_bounding_rectangle(points: &[(f64, f64)]) -> NdimageResult<[(f64, f64); 4]> {
    if points.is_empty() {
        return Err(NdimageError::InvalidInput(
            "minimum_bounding_rectangle: point set is empty".to_string(),
        ));
    }

    let hull = convex_hull_2d(points)?;
    if hull.len() == 1 {
        let p = hull[0];
        return Ok([p, p, p, p]);
    }
    if hull.len() == 2 {
        let (x0, y0) = hull[0];
        let (x1, y1) = hull[1];
        return Ok([(x0, y0), (x1, y1), (x1, y1), (x0, y0)]);
    }

    let n = hull.len();
    let mut min_area = f64::INFINITY;
    let mut best_rect = [(0.0f64, 0.0f64); 4];

    // Rotating calipers: iterate over each edge of the hull
    for i in 0..n {
        let j = (i + 1) % n;
        let (ex, ey) = (hull[j].0 - hull[i].0, hull[j].1 - hull[i].1);
        let len_e = (ex * ex + ey * ey).sqrt();
        if len_e < 1e-12 {
            continue;
        }
        let ux = ex / len_e; // unit edge vector
        let uy = ey / len_e;

        // Project all hull points onto (ux, uy) and its perpendicular (-uy, ux)
        let (mut min_u, mut max_u, mut min_v, mut max_v) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );

        for &(px, py) in &hull {
            let u = px * ux + py * uy;
            let v = -px * uy + py * ux;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }

        let area = (max_u - min_u) * (max_v - min_v);
        if area < min_area {
            min_area = area;
            // Reconstruct four corners from (u,v) back to (x,y)
            let corner = |u: f64, v: f64| -> (f64, f64) { (u * ux - v * uy, u * uy + v * ux) };
            best_rect = [
                corner(min_u, min_v),
                corner(max_u, min_v),
                corner(max_u, max_v),
                corner(min_u, max_v),
            ];
        }
    }

    Ok(best_rect)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Shoelace formula for signed polygon area.
fn shoelace_area(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0f64;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i].0 * pts[j].1;
        area -= pts[j].0 * pts[i].1;
    }
    area / 2.0
}

// 3×3 matrix helpers (row-major arrays)

fn sub_matrix_3x3(s: &[[f64; 6]; 6], row_off: usize, col_off: usize) -> [[f64; 3]; 3] {
    let mut m = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = s[row_off + i][col_off + j];
        }
    }
    m
}

fn mat3_det(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn mat3_inv(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = mat3_det(m);
    if det.abs() < 1e-14 {
        return None;
    }
    let inv_det = 1.0 / det;
    let mut inv = [[0.0f64; 3]; 3];
    inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
    Some(inv)
}

fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn mat3_sub(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    c
}

fn mat3_mul_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    let mut out = [0.0f64; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i] += m[i][j] * v[j];
        }
    }
    out
}

/// Compute eigenvalues and eigenvectors of a real 3×3 matrix using the
/// characteristic polynomial (Cardano) + inverse iteration refinement.
/// Returns `(eigenvalues, eigenvectors)` both of length 3.
fn mat3_eigen(m: &[[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    // Characteristic polynomial: λ³ - p·λ² + q·λ - r = 0
    let p = m[0][0] + m[1][1] + m[2][2]; // trace
    let q = m[0][0] * m[1][1] + m[1][1] * m[2][2] + m[0][0] * m[2][2]
        - m[0][1] * m[1][0]
        - m[1][2] * m[2][1]
        - m[0][2] * m[2][0];
    let r = mat3_det(m);

    // Depress the cubic λ³ − p·λ² + q·λ − r = 0 via λ = t + p/3.
    // The resulting depressed cubic is t³ + a·t + b = 0, with:
    //   a = q − p²/3
    //   b = −2p³/27 + p·q/3 − r
    let a = q - p * p / 3.0;
    let b = -2.0 * p * p * p / 27.0 + p * q / 3.0 - r;

    let disc = (b / 2.0).powi(2) + (a / 3.0).powi(3);

    let eigenvalues: [f64; 3] = if disc >= 0.0 {
        // One or two repeated real roots (or complex pair – degenerate case)
        let sqrt_disc = disc.sqrt();
        let u = cbrt(-b / 2.0 + sqrt_disc);
        let v = cbrt(-b / 2.0 - sqrt_disc);
        let t0 = u + v;
        let t1 = -(u + v) / 2.0;
        [t0 + p / 3.0, t1 + p / 3.0, t1 + p / 3.0]
    } else {
        // Three distinct real roots
        let rho = (-a / 3.0).sqrt().max(1e-30);
        let theta = ((-b / 2.0) / (rho * rho * rho)).clamp(-1.0, 1.0).acos();
        [
            2.0 * rho * (theta / 3.0).cos() + p / 3.0,
            2.0 * rho * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos() + p / 3.0,
            2.0 * rho * ((theta + 4.0 * std::f64::consts::PI) / 3.0).cos() + p / 3.0,
        ]
    };

    // Compute eigenvectors via (M - λI) null-space (cross-product method)
    let mut evecs = [[0.0f64; 3]; 3];
    for (i, &lam) in eigenvalues.iter().enumerate() {
        let a_mat = [
            [m[0][0] - lam, m[0][1], m[0][2]],
            [m[1][0], m[1][1] - lam, m[1][2]],
            [m[2][0], m[2][1], m[2][2] - lam],
        ];
        // Take cross products of pairs of rows and pick the largest
        let r0 = a_mat[0];
        let r1 = a_mat[1];
        let r2 = a_mat[2];
        let candidates = [cross3(r0, r1), cross3(r1, r2), cross3(r0, r2)];
        let best = candidates
            .iter()
            .copied()
            .max_by(|x, y| {
                let nx = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                let ny = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
                nx.partial_cmp(&ny).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or([1.0, 0.0, 0.0]);
        let norm = (best[0] * best[0] + best[1] * best[1] + best[2] * best[2])
            .sqrt()
            .max(1e-30);
        evecs[i] = [best[0] / norm, best[1] / norm, best[2] / norm];
    }

    (eigenvalues, evecs)
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn cbrt(x: f64) -> f64 {
    if x >= 0.0 {
        x.powf(1.0 / 3.0)
    } else {
        -(-x).powf(1.0 / 3.0)
    }
}

/// Convert general conic coefficients [A, B, C, D, E, F] to ellipse geometry.
fn conic_to_ellipse(c: [f64; 6]) -> NdimageResult<(f64, f64, f64, f64, f64)> {
    let (a, b, cc, d, e, f) = (c[0], c[1], c[2], c[3], c[4], c[5]);

    // Centre
    let denom = 4.0 * a * cc - b * b;
    if denom.abs() < 1e-14 {
        return Err(NdimageError::ComputationError(
            "ellipse_fit: degenerate conic (not an ellipse)".to_string(),
        ));
    }
    let cx = (b * e - 2.0 * cc * d) / denom;
    let cy = (b * d - 2.0 * a * e) / denom;

    // Axis-aligned form
    let f_prime = a * cx * cx + b * cx * cy + cc * cy * cy + d * cx + e * cy + f;

    let m_11 = a;
    let m_12 = b / 2.0;
    let m_22 = cc;

    let eig_diff = ((m_11 - m_22).powi(2) + m_12 * m_12 * 4.0).sqrt();
    let lam1 = (m_11 + m_22 + eig_diff) / 2.0;
    let lam2 = (m_11 + m_22 - eig_diff) / 2.0;

    if -f_prime / lam1 <= 0.0 || -f_prime / lam2 <= 0.0 {
        return Err(NdimageError::ComputationError(
            "ellipse_fit: conic is not a real ellipse".to_string(),
        ));
    }

    let axis1 = (-f_prime / lam1).sqrt();
    let axis2 = (-f_prime / lam2).sqrt();
    let (semi_major, semi_minor) = if axis1 >= axis2 {
        (axis1, axis2)
    } else {
        (axis2, axis1)
    };

    // Angle of major axis
    let angle = if (m_12).abs() < 1e-14 && m_11 <= m_22 {
        0.0
    } else if (m_12).abs() < 1e-14 {
        std::f64::consts::PI / 2.0
    } else {
        ((lam1 - m_11) / m_12).atan()
    };

    Ok((cx, cy, semi_major, semi_minor, angle))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_convex_hull_square() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)];
        let hull =
            convex_hull_2d(&pts).expect("convex_hull_2d should succeed for square point set");
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_convex_hull_collinear() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)];
        let hull =
            convex_hull_2d(&pts).expect("convex_hull_2d should succeed for collinear points");
        // Collinear – degenerate hull
        assert!(hull.len() >= 2);
    }

    #[test]
    fn test_convex_hull_empty() {
        assert!(convex_hull_2d(&[]).is_err());
    }

    #[test]
    fn test_contour_extraction_filled_square() {
        let mut img = Array2::<bool>::default((8, 8));
        for r in 2..6 {
            for c in 2..6 {
                img[[r, c]] = true;
            }
        }
        let contours = contour_extraction(&img.view())
            .expect("contour_extraction should succeed on valid image");
        assert_eq!(contours.len(), 1);
        assert!(!contours[0].is_empty());
    }

    #[test]
    fn test_contour_extraction_empty_image() {
        let img = Array2::<bool>::default((5, 5));
        let contours = contour_extraction(&img.view())
            .expect("contour_extraction should succeed on empty image");
        assert!(contours.is_empty());
    }

    #[test]
    fn test_shape_descriptors_square() {
        // 4×4 filled square: use the 8-pixel contour
        let contour: Vec<(usize, usize)> = vec![
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 3),
            (2, 3),
            (3, 3),
            (3, 2),
            (3, 1),
            (3, 0),
            (2, 0),
            (1, 0),
        ];
        let d = shape_descriptors(&contour)
            .expect("shape_descriptors should succeed for valid square contour");
        assert!(d.area > 0.0);
        assert!(d.perimeter > 0.0);
        assert!(d.circularity > 0.0);
        assert!(d.circularity <= 1.0 + 1e-9);
        assert!(d.solidity > 0.0 && d.solidity <= 1.0);
    }

    #[test]
    fn test_shape_descriptors_too_few() {
        assert!(shape_descriptors(&[(0, 0), (1, 1)]).is_err());
    }

    #[test]
    fn test_minimum_bounding_rectangle_axis_aligned() {
        let pts = vec![(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)];
        let rect = minimum_bounding_rectangle(&pts)
            .expect("minimum_bounding_rectangle should succeed for axis-aligned rectangle");
        let area = {
            let (dx1, dy1) = (rect[1].0 - rect[0].0, rect[1].1 - rect[0].1);
            let (dx2, dy2) = (rect[3].0 - rect[0].0, rect[3].1 - rect[0].1);
            let l1 = (dx1 * dx1 + dy1 * dy1).sqrt();
            let l2 = (dx2 * dx2 + dy2 * dy2).sqrt();
            l1 * l2
        };
        assert!((area - 12.0).abs() < 0.5, "area = {area}");
    }

    #[test]
    fn test_ellipse_fit_circle() {
        let pts: Vec<(f64, f64)> = (0..36)
            .map(|i| {
                let t = i as f64 * std::f64::consts::PI / 18.0;
                (t.cos(), t.sin())
            })
            .collect();
        let (cx, cy, a, b, _angle) =
            ellipse_fit(&pts).expect("ellipse_fit should succeed for circular point set");
        assert!((cx).abs() < 0.05, "cx={cx}");
        assert!((cy).abs() < 0.05, "cy={cy}");
        assert!((a - 1.0).abs() < 0.05, "a={a}");
        assert!((b - 1.0).abs() < 0.05, "b={b}");
    }
}
