//! Hypervolume indicator computation for multi-objective optimization.
//!
//! Provides exact and approximate algorithms for computing the hypervolume
//! dominated by a Pareto front with respect to a reference point.
//!
//! # Algorithms
//!
//! | Function | Algorithm | Complexity |
//! |----------|-----------|------------|
//! | [`hypervolume_2d`]      | Sweep-line on sorted front    | O(n log n) |
//! | [`hypervolume_3d`]      | Slice-and-sweep               | O(n² log n) |
//! | [`hypervolume_wfg`]     | WFG recursive algorithm       | O(n^(d-1)) |
//! | [`hypervolume_contribution_wfg`] | Remove-and-recompute | O(n^d) |
//! | [`exclusive_hypervolume`] | Per-solution contributions  | O(n^d) |
//!
//! # References
//!
//! - While, L., Hingston, P., Barone, L., & Huband, S. (2006). A faster
//!   algorithm for calculating hypervolume. *IEEE Transactions on Evolutionary
//!   Computation*, 10(1), 29-38.
//! - Emmerich, M., Beume, N., & Naujoks, B. (2005). An EMO algorithm using the
//!   hypervolume measure as selection criterion.
//! - Bradstreet, L., While, L., & Barone, L. (2007). A fast many-objective
//!   hypervolume algorithm. *ISPA*, 3-10.

use crate::error::{OptimizeError, OptimizeResult};

// ─────────────────────────────────────────────────────────────────────────────
// 2-D hypervolume (O(n log n) sweep)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the hypervolume dominated by a 2-D Pareto front.
///
/// The front need not be pre-sorted or pre-filtered; dominated points are
/// handled automatically.
///
/// # Arguments
/// * `front`     — Pareto front as `&[(f1, f2)]`. All objectives minimised.
/// * `reference` — Reference point `(r1, r2)`.  Every front point must
///   satisfy `f_i < r_i` for all `i` for a positive contribution.
///
/// # Returns
/// The hypervolume (area) dominated by the front and bounded by `reference`.
/// Returns `0.0` if the front is empty.
///
/// # Errors
/// Returns an error if `reference` dimensions do not match.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::hypervolume::hypervolume_2d;
/// let front = &[(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)];
/// let hv = hypervolume_2d(front, (2.0, 2.0)).expect("valid input");
/// assert!((hv - 3.25).abs() < 1e-10);
/// ```
pub fn hypervolume_2d(front: &[(f64, f64)], reference: (f64, f64)) -> OptimizeResult<f64> {
    if front.is_empty() {
        return Ok(0.0);
    }

    // Filter points that contribute (both coords less than reference)
    let mut pts: Vec<(f64, f64)> = front
        .iter()
        .filter(|&&(f1, f2)| f1 < reference.0 && f2 < reference.1)
        .copied()
        .collect();

    if pts.is_empty() {
        return Ok(0.0);
    }

    // Sort by first objective ascending
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Remove dominated points (keep only those with strictly decreasing f2)
    let mut non_dom: Vec<(f64, f64)> = Vec::with_capacity(pts.len());
    let mut min_f2 = f64::INFINITY;
    // Process in ascending f1 order: a point is non-dominated iff its f2 < all previously seen f2
    for (f1, f2) in pts {
        if f2 < min_f2 {
            non_dom.push((f1, f2));
            min_f2 = f2;
        }
    }

    // Sweep: area contributed by each point is (next_f1 - curr_f1) * (ref_f2 - curr_f2)
    let mut hv = 0.0_f64;
    let n = non_dom.len();
    for i in 0..n {
        let f1_next = if i + 1 < n {
            non_dom[i + 1].0
        } else {
            reference.0
        };
        let width = f1_next - non_dom[i].0;
        let height = reference.1 - non_dom[i].1;
        if width > 0.0 && height > 0.0 {
            hv += width * height;
        }
    }

    Ok(hv)
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-D hypervolume (slice-and-sweep)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the exact 3-D hypervolume using a slice-and-sweep approach.
///
/// The algorithm iterates over distinct z-values, computes the 2-D hypervolume
/// of the projected front at each slice, and integrates over z.
///
/// # Arguments
/// * `front`     — Slice of 3-element arrays `[f1, f2, f3]`.
/// * `reference` — 3-element reference point array `[r1, r2, r3]`.
///
/// # Errors
/// Returns an error if any point in `front` has length != 3.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::hypervolume::hypervolume_3d;
/// let front = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
/// let hv = hypervolume_3d(&front, &[2.0, 2.0, 2.0]).expect("valid input");
/// assert!(hv > 0.0);
/// ```
pub fn hypervolume_3d(front: &[[f64; 3]], reference: &[f64; 3]) -> OptimizeResult<f64> {
    if front.is_empty() {
        return Ok(0.0);
    }

    // Filter: only points with all coordinates < reference
    let mut pts: Vec<[f64; 3]> = front
        .iter()
        .filter(|p| p[0] < reference[0] && p[1] < reference[1] && p[2] < reference[2])
        .copied()
        .collect();

    if pts.is_empty() {
        return Ok(0.0);
    }

    // Sort by f3 (z-axis) ascending
    pts.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap_or(std::cmp::Ordering::Equal));

    // Collect unique z breakpoints (ascending order)
    let mut z_vals: Vec<f64> = pts.iter().map(|p| p[2]).collect();
    z_vals.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    // Slice-and-sweep: for the slab [z_vals[i], z_vals[i+1]) use active set
    // of all points with p[2] <= z_vals[i].  The first point that becomes
    // active at z_vals[0] contributes from z_vals[0] upward, so there is no
    // slab below z_vals[0].
    let mut hv = 0.0_f64;
    let n_z = z_vals.len();

    for i in 0..n_z {
        // Active set: all points that entered at or before z_vals[i]
        let active: Vec<(f64, f64)> = pts
            .iter()
            .filter(|p| p[2] <= z_vals[i])
            .map(|p| (p[0], p[1]))
            .collect();

        if active.is_empty() {
            continue;
        }

        let hv_2d = hypervolume_2d(&active, (reference[0], reference[1]))?;
        // This 2D cross-section is valid from z_vals[i] up to the next breakpoint
        let z_next = if i + 1 < n_z {
            z_vals[i + 1]
        } else {
            reference[2]
        };
        let dz = z_next - z_vals[i];
        if dz > 0.0 {
            hv += hv_2d * dz;
        }
    }

    Ok(hv)
}

// ─────────────────────────────────────────────────────────────────────────────
// WFG algorithm — arbitrary dimensions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the exact hypervolume using the WFG (While-Hingston-Barone-Huband)
/// recursive algorithm for arbitrary dimensionality.
///
/// WFG is asymptotically optimal for the general case and handles any number
/// of objectives. For 2-D, prefer [`hypervolume_2d`] which is faster in practice.
///
/// # Arguments
/// * `front`     — Pareto front as `&[Vec<f64>]`.  All objectives minimised.
///   Every inner vector must have the same length.
/// * `reference` — Reference point slice, same length as each objective vector.
///
/// # Returns
/// Exact hypervolume dominated by `front` relative to `reference`.
/// Returns `0.0` for an empty front.
///
/// # Errors
/// Returns an error if objective-vector lengths are inconsistent.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::hypervolume::hypervolume_wfg;
/// let front = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
/// let hv = hypervolume_wfg(&front, &[2.0, 2.0]).expect("valid input");
/// assert!((hv - 3.25).abs() < 1e-10);
/// ```
pub fn hypervolume_wfg(front: &[Vec<f64>], reference: &[f64]) -> OptimizeResult<f64> {
    if front.is_empty() {
        return Ok(0.0);
    }
    let n_obj = reference.len();
    for (i, pt) in front.iter().enumerate() {
        if pt.len() != n_obj {
            return Err(OptimizeError::InvalidInput(format!(
                "front[{i}] has length {} but reference has length {n_obj}",
                pt.len()
            )));
        }
    }

    // Filter points that are strictly dominated by the reference (all coords < ref)
    let mut pts: Vec<Vec<f64>> = front
        .iter()
        .filter(|pt| pt.iter().zip(reference.iter()).all(|(f, r)| f < r))
        .cloned()
        .collect();

    if pts.is_empty() {
        return Ok(0.0);
    }

    Ok(wfg_recurse(&mut pts, reference))
}

/// Internal WFG recursive computation. Mutates `pts` (sorting is applied).
fn wfg_recurse(pts: &mut Vec<Vec<f64>>, reference: &[f64]) -> f64 {
    let n = pts.len();
    if n == 0 {
        return 0.0;
    }
    let d = reference.len();

    // Base case: 1-D hypervolume is just (reference - min_val) clamped >= 0
    if d == 1 {
        let min_f: f64 = pts.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        return (reference[0] - min_f).max(0.0);
    }

    // Base case: single point
    if n == 1 {
        let vol: f64 = pts[0]
            .iter()
            .zip(reference.iter())
            .map(|(f, r)| (r - f).max(0.0))
            .product();
        return vol;
    }

    // Sort by last objective ASCENDING (smallest last-obj first).
    // Each point i "owns" the slab from its z up to the next point's z
    // (or the reference if it is the last point).
    pts.sort_by(|a, b| {
        a[d - 1]
            .partial_cmp(&b[d - 1])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Remove points that are dominated (keeps non-dominated wrt all d objectives)
    let non_dom = filter_nd_non_dominated(pts);
    let n_nd = non_dom.len();

    // Slice-sweep: HV(P, R) = Σ_i hv_{d-1}(non_dom[0..=i] projected, R') × dz_i
    // where dz_i is the slice thickness from non_dom[i][d-1] to the next boundary.
    let mut total = 0.0_f64;

    for i in 0..n_nd {
        // Upper boundary of this point's slab in the last dimension
        let z_next = if i + 1 < n_nd {
            non_dom[i + 1][d - 1]
        } else {
            reference[d - 1]
        };

        let dz = z_next - non_dom[i][d - 1];
        if dz <= 0.0 {
            continue;
        }

        // Project to d-1 dimensions: take all points up to and including i
        let sub_ref: Vec<f64> = reference[..d - 1].to_vec();
        let mut sub_pts: Vec<Vec<f64>> =
            non_dom[..=i].iter().map(|p| p[..d - 1].to_vec()).collect();

        let hv_sub = wfg_recurse(&mut sub_pts, &sub_ref);
        total += hv_sub * dz;
    }

    total
}

/// Filter non-dominated points from `pts` (already sorted by last objective desc).
fn filter_nd_non_dominated(pts: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut result: Vec<Vec<f64>> = Vec::with_capacity(pts.len());
    'outer: for pt in pts {
        for existing in &result {
            // If existing dominates pt, skip pt
            if nd_dominates(existing, pt) {
                continue 'outer;
            }
        }
        // Remove from result any points that pt dominates
        result.retain(|existing| !nd_dominates(pt, existing));
        result.push(pt.clone());
    }
    result
}

/// Returns true if `a` weakly dominates `b` in all objectives (a_i <= b_i for all i).
/// For the WFG internal filtering we use weak domination (a_i <= b_i AND at least one strict).
fn nd_dominates(a: &[f64], b: &[f64]) -> bool {
    let mut any_strict = false;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if *ai > *bi {
            return false;
        }
        if *ai < *bi {
            any_strict = true;
        }
    }
    any_strict
}

// ─────────────────────────────────────────────────────────────────────────────
// Hypervolume contribution of a single point
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the hypervolume contribution of solution `idx` to the total hypervolume.
///
/// The contribution is defined as:
/// ```text
/// HVC(x_i) = HV(P) - HV(P \ {x_i})
/// ```
/// where `P` is the full Pareto front.
///
/// # Arguments
/// * `front`     — Full Pareto front as `&[Vec<f64>]`.
/// * `reference` — Reference point.
/// * `idx`       — Index of the solution whose contribution to compute.
///
/// # Errors
/// Returns an error if `idx >= front.len()` or dimensions mismatch.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::hypervolume::hypervolume_contribution_wfg;
/// let front = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
/// let contrib = hypervolume_contribution_wfg(&front, &[2.0, 2.0], 1).expect("valid input");
/// assert!(contrib >= 0.0);
/// ```
pub fn hypervolume_contribution_wfg(
    front: &[Vec<f64>],
    reference: &[f64],
    idx: usize,
) -> OptimizeResult<f64> {
    if idx >= front.len() {
        return Err(OptimizeError::InvalidInput(format!(
            "idx={idx} out of range for front of length {}",
            front.len()
        )));
    }

    let total_hv = hypervolume_wfg(front, reference)?;

    // Front without the idx-th point
    let reduced: Vec<Vec<f64>> = front
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != idx)
        .map(|(_, pt)| pt.clone())
        .collect();

    let reduced_hv = hypervolume_wfg(&reduced, reference)?;
    Ok((total_hv - reduced_hv).max(0.0))
}

// ─────────────────────────────────────────────────────────────────────────────
// Exclusive (per-solution) hypervolume contributions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the exclusive hypervolume contribution of every solution in the front.
///
/// Returns a `Vec<f64>` of the same length as `front`, where each element is
/// the hypervolume contribution of that solution (HVC(x_i) = HV(P) − HV(P \ {x_i})).
///
/// Useful for diversity-aware selection and archiving.
///
/// # Arguments
/// * `front`     — Full Pareto front as `&[Vec<f64>]`.
/// * `reference` — Reference point.
///
/// # Errors
/// Returns an error if objective-vector dimensions are inconsistent.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::hypervolume::exclusive_hypervolume;
/// let front = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
/// let hvc = exclusive_hypervolume(&front, &[2.0, 2.0]).expect("valid input");
/// assert_eq!(hvc.len(), 3);
/// let total: f64 = hvc.iter().sum();
/// assert!(total > 0.0);
/// ```
pub fn exclusive_hypervolume(front: &[Vec<f64>], reference: &[f64]) -> OptimizeResult<Vec<f64>> {
    if front.is_empty() {
        return Ok(vec![]);
    }

    let n = front.len();
    let n_obj = reference.len();
    for (i, pt) in front.iter().enumerate() {
        if pt.len() != n_obj {
            return Err(OptimizeError::InvalidInput(format!(
                "front[{i}] has length {} but reference has length {n_obj}",
                pt.len()
            )));
        }
    }

    let total_hv = hypervolume_wfg(front, reference)?;
    let mut contributions = Vec::with_capacity(n);

    for idx in 0..n {
        let reduced: Vec<Vec<f64>> = front
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, pt)| pt.clone())
            .collect();

        let reduced_hv = hypervolume_wfg(&reduced, reference)?;
        contributions.push((total_hv - reduced_hv).max(0.0));
    }

    Ok(contributions)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── hypervolume_2d ────────────────────────────────────────────────────────

    #[test]
    fn test_hv2d_empty_front() {
        let hv = hypervolume_2d(&[], (2.0, 2.0)).expect("failed to create hv");
        assert_eq!(hv, 0.0);
    }

    #[test]
    fn test_hv2d_single_point() {
        // Point (0,0), reference (2,2): area = 2*2 = 4
        let hv = hypervolume_2d(&[(0.0, 0.0)], (2.0, 2.0)).expect("failed to create hv");
        assert!((hv - 4.0).abs() < 1e-10, "expected 4.0, got {hv}");
    }

    #[test]
    fn test_hv2d_three_points_on_front() {
        // Known: for (0,1), (0.5,0.5), (1,0) with ref (2,2)
        // areas: [0→0.5]*[2-1] + [0.5→1]*[2-0.5] + [1→2]*[2-0] = 0.5 + 0.75 + 2 = 3.25
        let front = &[(0.0f64, 1.0f64), (0.5, 0.5), (1.0, 0.0)];
        let hv = hypervolume_2d(front, (2.0, 2.0)).expect("failed to create hv");
        assert!((hv - 3.25).abs() < 1e-10, "expected 3.25, got {hv}");
    }

    #[test]
    fn test_hv2d_unsorted_input() {
        // Same as above but shuffled
        let front = &[(1.0f64, 0.0f64), (0.0, 1.0), (0.5, 0.5)];
        let hv = hypervolume_2d(front, (2.0, 2.0)).expect("failed to create hv");
        assert!((hv - 3.25).abs() < 1e-10, "expected 3.25, got {hv}");
    }

    #[test]
    fn test_hv2d_with_dominated_points() {
        // Adding a dominated point (0.5, 1.0) shouldn't change HV
        let front = &[(0.0, 1.0), (0.5, 0.5), (1.0, 0.0), (0.5, 1.0)];
        let hv = hypervolume_2d(front, (2.0, 2.0)).expect("failed to create hv");
        assert!((hv - 3.25).abs() < 1e-10, "expected 3.25, got {hv}");
    }

    #[test]
    fn test_hv2d_point_outside_reference() {
        // Point outside reference should not contribute
        let front = &[(3.0, 0.5)]; // f1 > ref[0]
        let hv = hypervolume_2d(front, (2.0, 2.0)).expect("failed to create hv");
        assert_eq!(hv, 0.0);
    }

    // ── hypervolume_3d ────────────────────────────────────────────────────────

    #[test]
    fn test_hv3d_empty_front() {
        let hv = hypervolume_3d(&[], &[2.0, 2.0, 2.0]).expect("failed to create hv");
        assert_eq!(hv, 0.0);
    }

    #[test]
    fn test_hv3d_single_point_unit_cube() {
        // Point (0,0,0) with ref (1,1,1): volume = 1
        let hv = hypervolume_3d(&[[0.0, 0.0, 0.0]], &[1.0, 1.0, 1.0]).expect("failed to create hv");
        assert!((hv - 1.0).abs() < 1e-10, "expected 1.0, got {hv}");
    }

    #[test]
    fn test_hv3d_two_non_dominated_points() {
        // [0,0,1] and [0,1,0] with ref [2,2,2]
        let front = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let hv = hypervolume_3d(&front, &[2.0, 2.0, 2.0]).expect("failed to create hv");
        assert!(hv > 0.0, "hypervolume should be positive, got {hv}");
    }

    #[test]
    fn test_hv3d_matches_wfg() {
        // Verify 3D result matches WFG on the same data
        let front = [[0.2, 0.5, 0.8], [0.5, 0.3, 0.6], [0.8, 0.2, 0.4]];
        let ref3 = [1.5, 1.5, 1.5];
        let hv3d = hypervolume_3d(&front, &ref3).expect("failed to create hv3d");
        let front_vecs: Vec<Vec<f64>> = front.iter().map(|p| p.to_vec()).collect();
        let hv_wfg = hypervolume_wfg(&front_vecs, &ref3).expect("failed to create hv_wfg");
        assert!(
            (hv3d - hv_wfg).abs() < 1e-6,
            "3D ({hv3d}) and WFG ({hv_wfg}) should agree"
        );
    }

    // ── hypervolume_wfg ───────────────────────────────────────────────────────

    #[test]
    fn test_wfg_empty_front() {
        let hv = hypervolume_wfg(&[], &[2.0, 2.0]).expect("failed to create hv");
        assert_eq!(hv, 0.0);
    }

    #[test]
    fn test_wfg_2d_matches_hv2d() {
        let front = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let hv_wfg = hypervolume_wfg(&front, &[2.0, 2.0]).expect("failed to create hv_wfg");
        let front_2d: Vec<(f64, f64)> = front.iter().map(|p| (p[0], p[1])).collect();
        let hv_2d = hypervolume_2d(&front_2d, (2.0, 2.0)).expect("failed to create hv_2d");
        assert!((hv_wfg - hv_2d).abs() < 1e-10, "WFG={hv_wfg}, 2D={hv_2d}");
    }

    #[test]
    fn test_wfg_single_point() {
        let front = vec![vec![1.0, 1.0, 1.0]];
        let hv = hypervolume_wfg(&front, &[3.0, 3.0, 3.0]).expect("failed to create hv");
        assert!((hv - 8.0).abs() < 1e-10, "expected 8.0, got {hv}");
    }

    #[test]
    fn test_wfg_dimension_mismatch_error() {
        let front = vec![vec![1.0, 2.0, 3.0]];
        let result = hypervolume_wfg(&front, &[4.0, 4.0]); // wrong reference dim
        assert!(result.is_err());
    }

    // ── hypervolume_contribution_wfg ─────────────────────────────────────────

    #[test]
    fn test_hvc_sum_leq_total() {
        let front = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let total = hypervolume_wfg(&front, &[2.0, 2.0]).expect("failed to create total");
        let hvc0 =
            hypervolume_contribution_wfg(&front, &[2.0, 2.0], 0).expect("failed to create hvc0");
        let hvc1 =
            hypervolume_contribution_wfg(&front, &[2.0, 2.0], 1).expect("failed to create hvc1");
        let hvc2 =
            hypervolume_contribution_wfg(&front, &[2.0, 2.0], 2).expect("failed to create hvc2");
        // All contributions should be non-negative
        assert!(hvc0 >= 0.0 && hvc1 >= 0.0 && hvc2 >= 0.0);
        // No single contribution exceeds total
        assert!(hvc0 <= total + 1e-10);
        assert!(hvc1 <= total + 1e-10);
        assert!(hvc2 <= total + 1e-10);
    }

    #[test]
    fn test_hvc_out_of_bounds_error() {
        let front = vec![vec![0.0, 1.0]];
        let result = hypervolume_contribution_wfg(&front, &[2.0, 2.0], 5);
        assert!(result.is_err());
    }

    // ── exclusive_hypervolume ─────────────────────────────────────────────────

    #[test]
    fn test_exclusive_hv_correct_length() {
        let front = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let hvc = exclusive_hypervolume(&front, &[2.0, 2.0]).expect("failed to create hvc");
        assert_eq!(hvc.len(), 3);
    }

    #[test]
    fn test_exclusive_hv_non_negative() {
        let front = vec![
            vec![0.1, 0.9],
            vec![0.3, 0.7],
            vec![0.5, 0.5],
            vec![0.7, 0.3],
            vec![0.9, 0.1],
        ];
        let hvc = exclusive_hypervolume(&front, &[1.5, 1.5]).expect("failed to create hvc");
        for (i, &c) in hvc.iter().enumerate() {
            assert!(c >= 0.0, "contribution[{i}] = {c} should be >= 0");
        }
    }

    #[test]
    fn test_exclusive_hv_empty_front() {
        let hvc = exclusive_hypervolume(&[], &[2.0, 2.0]).expect("failed to create hvc");
        assert!(hvc.is_empty());
    }

    #[test]
    fn test_exclusive_hv_dominated_gets_zero() {
        // Point (1.0, 1.0) is dominated by (0.5, 0.5)
        let front = vec![vec![0.5, 0.5], vec![1.0, 1.0]];
        let hvc = exclusive_hypervolume(&front, &[2.0, 2.0]).expect("failed to create hvc");
        // Dominated point should contribute 0 (within floating tolerance)
        assert!(
            hvc[1] < 1e-10,
            "dominated point contribution = {} should ~0",
            hvc[1]
        );
    }

    #[test]
    fn test_exclusive_hv_extreme_points_larger_contribution() {
        // Boundary points often have larger contributions
        let front = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let hvc = exclusive_hypervolume(&front, &[2.0, 2.0]).expect("failed to create hvc");
        // Interior point (0.5, 0.5) should have non-zero contribution
        assert!(hvc[1] >= 0.0, "interior contribution should be >= 0");
    }
}
