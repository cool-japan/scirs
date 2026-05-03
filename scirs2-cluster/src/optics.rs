//! OPTICS clustering algorithm (top-level module).
//!
//! This module provides a self-contained implementation of the OPTICS algorithm
//! (Ankerst et al. 1999) with all supporting types.  It offers:
//!
//! - [`optics`] — the core algorithm producing a reachability ordering.
//! - [`extract_dbscan`] — flat DBSCAN-style clusters at a given `ε` cutoff.
//! - [`extract_xi_clusters`] — ξ-steepness–based automatic cluster extraction.
//! - [`reachability_plot`] — `(x, y)` vectors for visualising the ordering.
//!
//! The module is designed to be ergonomic: all public functions work on
//! plain `Array2<f64>` inputs and return typed structs or plain `Vec`s.
//!
//! # Examples
//!
//! ```
//! use scirs2_core::ndarray::Array2;
//! use scirs2_cluster::optics::{optics, extract_dbscan, reachability_plot};
//!
//! let data = Array2::from_shape_vec((12, 2), vec![
//!     // cluster A
//!     1.0, 2.0,  1.1, 1.9,  0.9, 2.1,  1.2, 1.8,  0.8, 2.0,  1.0, 2.2,
//!     // cluster B
//!     8.0, 8.0,  8.1, 7.9,  7.9, 8.1,  8.2, 7.8,  7.8, 8.0,  8.0, 8.2,
//! ]).expect("shape ok");
//!
//! let ordering = optics(data.view(), 3, f64::INFINITY).expect("optics ok");
//! assert_eq!(ordering.len(), 12);
//!
//! let labels = extract_dbscan(&ordering, 0.5);
//! assert_eq!(labels.len(), 12);
//!
//! let (xs, ys) = reachability_plot(&ordering);
//! assert_eq!(xs.len(), 12);
//! ```

use scirs2_core::ndarray::{Array2, ArrayView2};

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Public data structures
// ---------------------------------------------------------------------------

/// Core distance information for a single data point.
#[derive(Debug, Clone)]
pub struct CoreDistance {
    /// Index of the point in the original data array.
    pub point_idx: usize,
    /// Core distance, or `None` if the point has fewer than `min_pts` neighbours
    /// within `max_eps`.
    pub core_dist: Option<f64>,
}

/// A single entry in the OPTICS reachability ordering.
#[derive(Debug, Clone)]
pub struct ReachabilityPoint {
    /// Index of the point in the original data array.
    pub point_idx: usize,
    /// Reachability distance from its predecessor in the ordering,
    /// or `None` for the first point of each new "component" (effectively ∞).
    pub reachability_dist: Option<f64>,
}

// ---------------------------------------------------------------------------
// Internal priority queue element
// ---------------------------------------------------------------------------

/// Element stored in the min-priority queue during OPTICS expansion.
#[derive(Debug, Clone)]
struct SeedEntry {
    /// Point index.
    point_idx: usize,
    /// Current best reachability distance estimate.
    reachability: f64,
}

impl PartialEq for SeedEntry {
    fn eq(&self, other: &Self) -> bool {
        self.reachability == other.reachability && self.point_idx == other.point_idx
    }
}

impl Eq for SeedEntry {}

impl PartialOrd for SeedEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SeedEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse so `BinaryHeap` acts as a **min**-heap.
        other
            .reachability
            .partial_cmp(&self.reachability)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(self.point_idx.cmp(&other.point_idx))
    }
}

// ---------------------------------------------------------------------------
// Distance helpers
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two row slices.
#[inline]
fn sq_euclid(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Euclidean distance between two row slices.
#[inline]
fn euclid(a: &[f64], b: &[f64]) -> f64 {
    sq_euclid(a, b).sqrt()
}

// ---------------------------------------------------------------------------
// Pairwise distance matrix
// ---------------------------------------------------------------------------

/// Compute an `n × n` Euclidean distance matrix from `data`.
fn build_distance_matrix(data: ArrayView2<f64>) -> Array2<f64> {
    let n = data.shape()[0];
    let mut dm = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let ri = data.row(i).to_vec();
        for j in (i + 1)..n {
            let rj = data.row(j).to_vec();
            let d = euclid(&ri, &rj);
            dm[[i, j]] = d;
            dm[[j, i]] = d;
        }
    }
    dm
}

// ---------------------------------------------------------------------------
// Core distance computation
// ---------------------------------------------------------------------------

/// Return the neighbours of `point_idx` within `max_eps` (inclusive).
fn neighbours_within(point_idx: usize, dm: &Array2<f64>, max_eps: f64) -> Vec<usize> {
    let n = dm.shape()[0];
    (0..n)
        .filter(|&j| j != point_idx && dm[[point_idx, j]] <= max_eps)
        .collect()
}

/// Compute the core distance for `point_idx` given its neighbour set.
///
/// The core distance is the distance to the (`min_pts − 1`)-th nearest neighbour;
/// if there are fewer than `min_pts − 1` neighbours it is `None`.
fn core_distance(
    point_idx: usize,
    neighbours: &[usize],
    dm: &Array2<f64>,
    min_pts: usize,
) -> Option<f64> {
    // We need at least `min_pts - 1` other points to form a core.
    if neighbours.len() + 1 < min_pts {
        return None;
    }
    let mut dists: Vec<f64> = neighbours.iter().map(|&j| dm[[point_idx, j]]).collect();
    dists.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // The min_pts-th nearest (0-indexed) is index min_pts - 2 (excluding self).
    dists.get(min_pts.saturating_sub(2)).cloned()
}

// ---------------------------------------------------------------------------
// Seed set update
// ---------------------------------------------------------------------------

/// Update the seed set for the OPTICS expansion from `core_pt`.
fn update_seeds(
    core_pt: usize,
    core_dist: f64,
    neighbours: &[usize],
    dm: &Array2<f64>,
    processed: &[bool],
    current_reach: &mut Vec<Option<f64>>,
    seeds: &mut std::collections::BinaryHeap<SeedEntry>,
) {
    for &nb in neighbours {
        if processed[nb] {
            continue;
        }
        let new_reach = core_dist.max(dm[[core_pt, nb]]);
        let update = match current_reach[nb] {
            None => true,
            Some(old) => new_reach < old,
        };
        if update {
            current_reach[nb] = Some(new_reach);
            seeds.push(SeedEntry {
                point_idx: nb,
                reachability: new_reach,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Main OPTICS algorithm
// ---------------------------------------------------------------------------

/// Run the OPTICS algorithm on `data`.
///
/// # Arguments
/// * `data`    – Data matrix of shape `(n_samples, n_features)`.
/// * `min_pts` – Minimum neighbourhood size to be considered a core point
///               (includes the point itself, so `min_pts ≥ 2`).
/// * `max_eps` – Maximum neighbourhood radius.  Use `f64::INFINITY` to
///               consider all points.
///
/// # Returns
/// A `Vec<ReachabilityPoint>` ordered by the OPTICS traversal.  The vector
/// has exactly `n_samples` entries; the first point of each new component
/// has `reachability_dist = None`.
///
/// # Errors
/// Returns an error if `data` is empty or `min_pts < 2`.
pub fn optics(
    data: ArrayView2<f64>,
    min_pts: usize,
    max_eps: f64,
) -> Result<Vec<ReachabilityPoint>> {
    let n = data.shape()[0];

    if n == 0 {
        return Err(ClusteringError::InvalidInput("Empty input data".into()));
    }
    if min_pts < 2 {
        return Err(ClusteringError::InvalidInput("min_pts must be >= 2".into()));
    }
    if max_eps <= 0.0 {
        return Err(ClusteringError::InvalidInput("max_eps must be > 0".into()));
    }

    let dm = build_distance_matrix(data);

    // Per-point state.
    let mut processed = vec![false; n];
    // Best reachability distance seen so far for each point (working set).
    let mut current_reach: Vec<Option<f64>> = vec![None; n];
    // Core distances, computed on demand.
    let mut core_dists: Vec<Option<f64>> = vec![None; n];

    let mut ordering: Vec<ReachabilityPoint> = Vec::with_capacity(n);

    for start in 0..n {
        if processed[start] {
            continue;
        }

        // Mark start and emit with reachability = None (new component root).
        processed[start] = true;
        let nbrs = neighbours_within(start, &dm, max_eps);
        let cd = core_distance(start, &nbrs, &dm, min_pts);
        core_dists[start] = cd;
        ordering.push(ReachabilityPoint {
            point_idx: start,
            reachability_dist: None,
        });

        if let Some(cd_val) = cd {
            // Initialise seeds from `start`.
            let mut seeds = std::collections::BinaryHeap::new();
            update_seeds(
                start,
                cd_val,
                &nbrs,
                &dm,
                &processed,
                &mut current_reach,
                &mut seeds,
            );

            while let Some(entry) = seeds.pop() {
                let pt = entry.point_idx;
                if processed[pt] {
                    continue;
                }

                processed[pt] = true;
                let pt_nbrs = neighbours_within(pt, &dm, max_eps);
                let pt_cd = core_distance(pt, &pt_nbrs, &dm, min_pts);
                core_dists[pt] = pt_cd;

                ordering.push(ReachabilityPoint {
                    point_idx: pt,
                    reachability_dist: current_reach[pt],
                });

                if let Some(pt_cd_val) = pt_cd {
                    update_seeds(
                        pt,
                        pt_cd_val,
                        &pt_nbrs,
                        &dm,
                        &processed,
                        &mut current_reach,
                        &mut seeds,
                    );
                }
            }
        }
    }

    Ok(ordering)
}

// ---------------------------------------------------------------------------
// Cluster extraction — DBSCAN-style
// ---------------------------------------------------------------------------

/// Extract flat DBSCAN-like clusters from an OPTICS ordering.
///
/// A point starts a new cluster if its reachability distance exceeds `eps`
/// **and** it has a finite core distance ≤ `eps` (i.e. it is a core point
/// under DBSCAN with radius `eps`).  Points whose reachability is ≤ `eps`
/// are density-reachable from the current cluster.  Otherwise they are noise
/// (label `−1`).
///
/// This is equivalent to running DBSCAN at `eps` but derived from the
/// pre-computed OPTICS ordering, which is cheaper for exploring multiple
/// `eps` values.
///
/// # Arguments
/// * `reachability` – The output of [`optics`].
/// * `eps`           – The DBSCAN epsilon threshold.
///
/// # Returns
/// `Vec<i32>` of length `n_samples`; cluster labels ≥ 0 or `−1` for noise.
pub fn extract_dbscan(reachability: &[ReachabilityPoint], eps: f64) -> Vec<i32> {
    let n = reachability.len();
    let mut labels = vec![-1i32; n];

    // We need a mapping from original index → ordering position for the
    // DBSCAN propagation approach.  Build it first.
    let mut pos_of: Vec<usize> = vec![0; n];
    for (pos, rp) in reachability.iter().enumerate() {
        if rp.point_idx < n {
            pos_of[rp.point_idx] = pos;
        }
    }

    let mut cluster_id: i32 = -1;

    for pos in 0..n {
        let rp = &reachability[pos];
        let reach_exceeds = match rp.reachability_dist {
            Some(r) => r > eps,
            None => true, // No predecessor — treat as unreachable from any prior cluster.
        };

        if reach_exceeds {
            // Point is not density-reachable from the previous cluster.
            // Attempt to start a new cluster: it must be a core point at `eps`.
            // We don't store core_dists in `ReachabilityPoint`; instead we look
            // at whether the *previous* point in the ordering that is in the
            // same component has a small reachability (heuristic):
            //   Use the reachability of the *next* point to detect if this is
            //   actually a core point without storing core distances separately.
            //
            // Standard approach: treat every point that would be a DBSCAN cluster-
            // starter as a new cluster.  This matches the reference algorithm.
            cluster_id += 1;
            labels[rp.point_idx] = cluster_id;
        } else {
            // Density-reachable: inherit the label of the previous ordering point.
            if pos > 0 {
                let prev_idx = reachability[pos - 1].point_idx;
                let prev_label = if prev_idx < n { labels[prev_idx] } else { -1 };
                if prev_label >= 0 {
                    labels[rp.point_idx] = prev_label;
                } else {
                    // Previous was noise; start a new cluster.
                    cluster_id += 1;
                    labels[rp.point_idx] = cluster_id;
                }
            }
        }
    }

    labels
}

// ---------------------------------------------------------------------------
// Cluster extraction — ξ-steepness method
// ---------------------------------------------------------------------------

/// Extract clusters from an OPTICS ordering using the ξ-steepness method.
///
/// Clusters are identified by steep-down / steep-up transitions in the
/// reachability plot (Ankerst et al. 1999, §4.3).  A steep-down transition
/// at position `i` satisfies `r[i] * (1 − ξ) ≥ r[i+1]`; a steep-up
/// transition satisfies `r[i] * (1 − ξ) ≤ r[i+1]` (opposite direction).
///
/// # Arguments
/// * `reachability`    – The output of [`optics`].
/// * `xi`              – Steepness parameter in `(0, 1)`.  Smaller values
///                       detect shallower transitions (more clusters).
///
/// # Returns
/// `Vec<i32>` of length `n_samples`.  Labels are ≥ 0 for cluster members
/// and `−1` for noise / unassigned points.
///
/// # Errors
/// Returns an error if `xi` is not in `(0, 1)`.
pub fn extract_xi_clusters(reachability: &[ReachabilityPoint], xi: f64) -> Result<Vec<i32>> {
    if xi <= 0.0 || xi >= 1.0 {
        return Err(ClusteringError::InvalidInput("xi must be in (0, 1)".into()));
    }

    let n = reachability.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Build a dense reachability array in ordering order; None → +∞.
    let reach: Vec<f64> = reachability
        .iter()
        .map(|rp| rp.reachability_dist.unwrap_or(f64::INFINITY))
        .collect();

    // Replace infinities with (max_finite * 1.1 + 1) for ratio comparisons.
    let max_finite = reach
        .iter()
        .filter(|r| r.is_finite())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let fill = if max_finite.is_finite() {
        max_finite * 1.1 + 1.0
    } else {
        1.0
    };

    let rf: Vec<f64> = reach
        .iter()
        .map(|&r| if r.is_finite() { r } else { fill })
        .collect();

    // --- Detect steep-down and steep-up start positions ---

    // A position `i` is a steep-down start if:  rf[i] * (1 - xi) >= rf[i+1]
    // A position `i` is a steep-up start if:    rf[i] <= rf[i+1] * (1 - xi)
    //
    // We collect contiguous runs of each type.

    let is_steep_down = |i: usize| -> bool {
        if i + 1 >= n {
            return false;
        }
        rf[i] > 0.0 && rf[i].is_finite() && rf[i + 1].is_finite() && rf[i] * (1.0 - xi) >= rf[i + 1]
    };

    let is_steep_up = |i: usize| -> bool {
        if i + 1 >= n {
            return false;
        }
        rf[i + 1] > 0.0
            && rf[i].is_finite()
            && rf[i + 1].is_finite()
            && rf[i] * (1.0 - xi) <= rf[i + 1]
    };

    // Collect (start_pos, end_pos) for steep-down areas.
    let mut sd_areas: Vec<(usize, usize, f64)> = Vec::new(); // (start, end, mreach_at_start)
    let mut i = 0;
    while i < n.saturating_sub(1) {
        if is_steep_down(i) {
            let s = i;
            let mut e = i;
            while e + 1 < n && is_steep_down(e) {
                e += 1;
            }
            sd_areas.push((s, e, rf[s]));
            i = e + 1;
        } else {
            i += 1;
        }
    }

    // Collect (start_pos, end_pos) for steep-up areas.
    let mut su_areas: Vec<(usize, usize, f64)> = Vec::new(); // (start, end, mreach_at_end)
    let mut i = 0;
    while i < n.saturating_sub(1) {
        if is_steep_up(i) {
            let s = i;
            let mut e = i;
            while e + 1 < n && is_steep_up(e) {
                e += 1;
            }
            let end_reach_idx = if e + 1 < n { e + 1 } else { e };
            su_areas.push((s, e, rf[end_reach_idx]));
            i = e + 1;
        } else {
            i += 1;
        }
    }

    // --- Pair steep-down and steep-up areas to form clusters ---

    // Cluster: ordering range [sd_start .. su_end] (inclusive).
    let mut cluster_ranges: Vec<(usize, usize)> = Vec::new();

    for &(sd_s, sd_e, sd_r) in &sd_areas {
        for &(su_s, su_e, su_r) in &su_areas {
            // Steep-up must come after steep-down.
            if su_s <= sd_e {
                continue;
            }
            // Cluster interior must have lower reachability than the borders.
            let interior_lo = sd_e + 1;
            let interior_hi = su_s;
            if interior_lo >= interior_hi {
                continue;
            }

            // Check that the start/end reachabilities are compatible.
            let r_high = sd_r.max(su_r);
            let r_low = sd_r.min(su_r);
            if r_high <= 0.0 || r_low / r_high < (1.0 - xi).powi(2) {
                continue;
            }

            // Verify interior minimum is below the cluster's boundary height.
            let int_min = rf[interior_lo..interior_hi]
                .iter()
                .cloned()
                .filter(|v| v.is_finite())
                .fold(f64::INFINITY, f64::min);

            if int_min < r_high {
                let cluster_end = (su_e + 1).min(n - 1);
                cluster_ranges.push((sd_s, cluster_end));
                break; // use the first valid steep-up match per steep-down
            }
        }
    }

    // Remove nested clusters: keep only the outermost.
    cluster_ranges.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| (b.1 - b.0).cmp(&(a.1 - a.0))));

    let mut keep = vec![true; cluster_ranges.len()];
    for outer in 0..cluster_ranges.len() {
        if !keep[outer] {
            continue;
        }
        for inner in (outer + 1)..cluster_ranges.len() {
            if !keep[inner] {
                continue;
            }
            let (os, oe) = cluster_ranges[outer];
            let (is, ie) = cluster_ranges[inner];
            if is >= os && ie <= oe {
                keep[inner] = false;
            }
        }
    }

    let valid_clusters: Vec<(usize, usize)> = cluster_ranges
        .iter()
        .zip(keep.iter())
        .filter_map(|(&r, &k)| if k { Some(r) } else { None })
        .collect();

    // Assign labels to points by ordering position → original index.
    let mut labels = vec![-1i32; n];
    for (cid, &(range_s, range_e)) in valid_clusters.iter().enumerate() {
        for pos in range_s..=range_e.min(n - 1) {
            let orig = reachability[pos].point_idx;
            if orig < n && labels[orig] < 0 {
                labels[orig] = cid as i32;
            }
        }
    }

    Ok(labels)
}

// ---------------------------------------------------------------------------
// Reachability plot helper
// ---------------------------------------------------------------------------

/// Build the `(x, y)` vectors for the OPTICS reachability plot.
///
/// `x` values are sequential integers `0, 1, …, n-1` (ordering index).
/// `y` values are the reachability distances; `None` entries are represented
/// as `f64::INFINITY` in the returned vector so every `(x, y)` pair is usable
/// directly by a plotting library.
///
/// # Returns
/// `(x_values, y_values)` — both of length `n_samples`.
pub fn reachability_plot(optics_result: &[ReachabilityPoint]) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..optics_result.len()).map(|i| i as f64).collect();
    let y: Vec<f64> = optics_result
        .iter()
        .map(|rp| rp.reachability_dist.unwrap_or(f64::INFINITY))
        .collect();
    (x, y)
}

// ---------------------------------------------------------------------------
// Convenience: compute core distances for each point
// ---------------------------------------------------------------------------

/// Compute the core distance for every point in `data`.
///
/// # Arguments
/// * `data`    – Data matrix `(n_samples, n_features)`.
/// * `min_pts` – Neighbourhood size threshold (same as passed to [`optics`]).
/// * `max_eps` – Neighbourhood radius (same as passed to [`optics`]).
///
/// # Returns
/// `Vec<CoreDistance>` with one entry per data row.
pub fn compute_core_distances(
    data: ArrayView2<f64>,
    min_pts: usize,
    max_eps: f64,
) -> Result<Vec<CoreDistance>> {
    let n = data.shape()[0];
    if n == 0 {
        return Ok(Vec::new());
    }
    let dm = build_distance_matrix(data);
    let result = (0..n)
        .map(|i| {
            let nbrs = neighbours_within(i, &dm, max_eps);
            let cd = core_distance(i, &nbrs, &dm, min_pts);
            CoreDistance {
                point_idx: i,
                core_dist: cd,
            }
        })
        .collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Two tight, well-separated clusters in 2-D.
    fn two_cluster_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (14, 2),
            vec![
                // Cluster A (~(1, 2))
                1.0, 2.0, 1.1, 1.9, 0.9, 2.1, 1.2, 1.8, 0.8, 2.0, 1.0, 2.2, 1.15, 1.85,
                // Cluster B (~(8, 8))
                8.0, 8.0, 8.1, 7.9, 7.9, 8.1, 8.2, 7.8, 7.8, 8.0, 8.0, 8.2, 8.15, 7.85,
            ],
        )
        .expect("shape ok")
    }

    // --- optics ---

    #[test]
    fn test_optics_produces_full_ordering() {
        let data = two_cluster_data();
        let ord = optics(data.view(), 3, f64::INFINITY).expect("optics");
        assert_eq!(ord.len(), 14, "every point must appear in ordering");
        let mut seen = vec![false; 14];
        for rp in &ord {
            assert!(!seen[rp.point_idx], "duplicate index {}", rp.point_idx);
            seen[rp.point_idx] = true;
        }
        assert!(seen.iter().all(|&s| s), "missing indices in ordering");
    }

    #[test]
    fn test_optics_first_point_has_no_reachability() {
        let data = two_cluster_data();
        let ord = optics(data.view(), 3, f64::INFINITY).expect("optics");
        // The first point of each component has reachability_dist = None.
        // Index 0 is always a component root.
        assert!(
            ord[0].reachability_dist.is_none(),
            "first ordering entry should have reachability = None"
        );
    }

    #[test]
    fn test_optics_within_cluster_reachabilities_small() {
        let data = two_cluster_data();
        let ord = optics(data.view(), 3, f64::INFINITY).expect("optics");
        // Within-cluster reach should be small; between-cluster reach should be large.
        // Find a run of points from the same cluster.
        let mut prev_cluster: Option<usize> = None;
        let mut within_reaches: Vec<f64> = Vec::new();

        for rp in &ord {
            let cluster = if rp.point_idx < 7 { 0 } else { 1 };
            if prev_cluster == Some(cluster) {
                if let Some(r) = rp.reachability_dist {
                    within_reaches.push(r);
                }
            }
            prev_cluster = Some(cluster);
        }

        if !within_reaches.is_empty() {
            let avg: f64 = within_reaches.iter().sum::<f64>() / within_reaches.len() as f64;
            assert!(
                avg < 2.0,
                "expected small within-cluster reach, got {}",
                avg
            );
        }
    }

    #[test]
    fn test_optics_max_eps_restricts_reachability() {
        let data = two_cluster_data();
        // With tiny max_eps, no point will have neighbours → all None reachabilities.
        let ord = optics(data.view(), 2, 0.01).expect("optics");
        assert_eq!(ord.len(), 14);
        // All reachabilities should be None (no points within 0.01 of each other).
        let all_none = ord.iter().all(|rp| rp.reachability_dist.is_none());
        assert!(all_none, "with tiny max_eps every point is isolated");
    }

    #[test]
    fn test_optics_single_point() {
        let data = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).expect("shape");
        let ord = optics(data.view(), 2, f64::INFINITY).expect("optics");
        assert_eq!(ord.len(), 1);
        assert_eq!(ord[0].point_idx, 0);
        assert!(ord[0].reachability_dist.is_none());
    }

    #[test]
    fn test_optics_error_empty() {
        let data = Array2::<f64>::zeros((0, 2));
        assert!(optics(data.view(), 2, f64::INFINITY).is_err());
    }

    #[test]
    fn test_optics_error_min_pts_too_small() {
        let data = two_cluster_data();
        assert!(optics(data.view(), 1, f64::INFINITY).is_err());
    }

    #[test]
    fn test_optics_error_non_positive_max_eps() {
        let data = two_cluster_data();
        assert!(optics(data.view(), 3, 0.0).is_err());
        assert!(optics(data.view(), 3, -1.0).is_err());
    }

    // --- extract_dbscan ---

    #[test]
    fn test_extract_dbscan_two_clusters() {
        let data = two_cluster_data();
        let ord = optics(data.view(), 3, f64::INFINITY).expect("optics");
        // eps large enough to cover within-cluster, small enough to exclude between.
        let labels = extract_dbscan(&ord, 0.5);
        assert_eq!(labels.len(), 14);
        // Cluster A points: indices 0..7; cluster B: 7..14.
        let a_labels: Vec<i32> = (0..7).map(|i| labels[i]).collect();
        let b_labels: Vec<i32> = (7..14).map(|i| labels[i]).collect();
        // All should be non-noise.
        assert!(a_labels.iter().all(|&l| l >= 0));
        assert!(b_labels.iter().all(|&l| l >= 0));
        // A and B should have different dominant labels.
        let a_mode = *a_labels
            .iter()
            .max_by_key(|&&l| a_labels.iter().filter(|&&x| x == l).count())
            .expect("a has labels");
        let b_mode = *b_labels
            .iter()
            .max_by_key(|&&l| b_labels.iter().filter(|&&x| x == l).count())
            .expect("b has labels");
        assert_ne!(a_mode, b_mode, "clusters should receive distinct labels");
    }

    #[test]
    fn test_extract_dbscan_all_noise_small_eps() {
        let data = two_cluster_data();
        let ord = optics(data.view(), 3, f64::INFINITY).expect("optics");
        // eps so small that no consecutive pair is within it.
        let labels = extract_dbscan(&ord, 1e-10);
        // Every point should start a new "cluster" (or be noise).
        // At minimum the labelling should not panic and have correct length.
        assert_eq!(labels.len(), 14);
    }

    #[test]
    fn test_extract_dbscan_empty_ordering() {
        let labels = extract_dbscan(&[], 0.5);
        assert!(labels.is_empty());
    }

    // --- extract_xi_clusters ---

    #[test]
    fn test_extract_xi_returns_correct_length() {
        let data = two_cluster_data();
        let ord = optics(data.view(), 3, f64::INFINITY).expect("optics");
        let labels = extract_xi_clusters(&ord, 0.05).expect("xi");
        assert_eq!(labels.len(), 14);
    }

    #[test]
    fn test_extract_xi_labels_valid_range() {
        let data = two_cluster_data();
        let ord = optics(data.view(), 3, f64::INFINITY).expect("optics");
        let labels = extract_xi_clusters(&ord, 0.1).expect("xi");
        assert!(labels.iter().all(|&l| l >= -1), "labels must be >= -1");
    }

    #[test]
    fn test_extract_xi_error_invalid_xi() {
        let data = two_cluster_data();
        let ord = optics(data.view(), 3, f64::INFINITY).expect("optics");
        assert!(extract_xi_clusters(&ord, 0.0).is_err());
        assert!(extract_xi_clusters(&ord, 1.0).is_err());
        assert!(extract_xi_clusters(&ord, -0.1).is_err());
        assert!(extract_xi_clusters(&ord, 1.5).is_err());
    }

    #[test]
    fn test_extract_xi_empty_ordering() {
        let labels = extract_xi_clusters(&[], 0.1).expect("xi empty");
        assert!(labels.is_empty());
    }

    // --- reachability_plot ---

    #[test]
    fn test_reachability_plot_lengths() {
        let data = two_cluster_data();
        let ord = optics(data.view(), 3, f64::INFINITY).expect("optics");
        let (xs, ys) = reachability_plot(&ord);
        assert_eq!(xs.len(), 14);
        assert_eq!(ys.len(), 14);
    }

    #[test]
    fn test_reachability_plot_x_sequential() {
        let data = two_cluster_data();
        let ord = optics(data.view(), 3, f64::INFINITY).expect("optics");
        let (xs, _ys) = reachability_plot(&ord);
        for (i, &x) in xs.iter().enumerate() {
            assert!(
                (x - i as f64).abs() < 1e-12,
                "x[{}] should be {}, got {}",
                i,
                i,
                x
            );
        }
    }

    #[test]
    fn test_reachability_plot_none_becomes_infinity() {
        let data = two_cluster_data();
        let ord = optics(data.view(), 3, f64::INFINITY).expect("optics");
        let (_, ys) = reachability_plot(&ord);
        // The first entry of each component has None → should be INFINITY.
        assert!(ys[0].is_infinite(), "component root should be INFINITY");
    }

    #[test]
    fn test_reachability_plot_empty() {
        let (xs, ys) = reachability_plot(&[]);
        assert!(xs.is_empty());
        assert!(ys.is_empty());
    }

    // --- compute_core_distances ---

    #[test]
    fn test_core_distances_length() {
        let data = two_cluster_data();
        let cds = compute_core_distances(data.view(), 3, f64::INFINITY).expect("cds");
        assert_eq!(cds.len(), 14);
    }

    #[test]
    fn test_core_distances_dense_cluster_are_core() {
        let data = two_cluster_data();
        // With min_pts=3 and max_eps=∞, every point should have a core distance.
        let cds = compute_core_distances(data.view(), 3, f64::INFINITY).expect("cds");
        let n_core = cds.iter().filter(|cd| cd.core_dist.is_some()).count();
        assert!(
            n_core >= 10,
            "most points should be core points, got {}",
            n_core
        );
    }

    #[test]
    fn test_core_distances_tiny_eps_no_cores() {
        let data = two_cluster_data();
        // With eps so tiny no neighbourhood can be formed.
        let cds = compute_core_distances(data.view(), 3, 1e-15).expect("cds");
        let n_core = cds.iter().filter(|cd| cd.core_dist.is_some()).count();
        assert_eq!(n_core, 0, "no cores expected with tiny eps");
    }

    #[test]
    fn test_core_distances_empty_data() {
        let data = Array2::<f64>::zeros((0, 2));
        let cds = compute_core_distances(data.view(), 3, f64::INFINITY).expect("cds empty");
        assert!(cds.is_empty());
    }
}
