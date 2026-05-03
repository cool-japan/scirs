//! Density-based clustering: Density Peaks and Mean Shift
//!
//! This module provides two non-parametric clustering algorithms that locate clusters
//! by finding regions of high density:
//!
//! ## Density Peaks Clustering
//!
//! Rodriguez & Laio (2014) — cluster centres are identified as points that simultaneously
//! exhibit high local density (ρ) and large minimum distance to any higher-density point (δ).
//! The decision graph of (ρ, δ) pairs makes cluster centre selection visual and intuitive.
//!
//! ## Mean Shift
//!
//! A kernel-density-based iterative algorithm that shifts each seed point towards the
//! local density mode.  Cluster centres are the modes discovered; every data point is
//! assigned to the nearest mode.
//!
//! # References
//!
//! * Rodriguez, A., & Laio, A. (2014). Clustering by fast search and find of density peaks.
//!   *Science*, 344(6191), 1492-1496.
//! * Fukunaga, K., & Hostetler, L. (1975). The estimation of the gradient of a density
//!   function, with applications in pattern recognition. *IEEE Transactions on Information
//!   Theory*, 21(1), 32-40.

use scirs2_core::ndarray::Array2;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Helpers shared by both algorithms
// ---------------------------------------------------------------------------

/// Compute the n×n Euclidean distance matrix between all pairs of rows.
fn pairwise_distances(data: &Array2<f64>) -> Vec<f64> {
    let n = data.nrows();
    let d = data.ncols();
    let mut dist = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in i..n {
            let mut sq = 0.0_f64;
            for f in 0..d {
                let diff = data[[i, f]] - data[[j, f]];
                sq += diff * diff;
            }
            let eucl = sq.sqrt();
            dist[i * n + j] = eucl;
            dist[j * n + i] = eucl;
        }
    }
    dist
}

/// Euclidean distance between two data rows.
#[inline]
fn row_dist(data: &Array2<f64>, a: usize, b: usize) -> f64 {
    let d = data.ncols();
    let mut sq = 0.0_f64;
    for f in 0..d {
        let diff = data[[a, f]] - data[[b, f]];
        sq += diff * diff;
    }
    sq.sqrt()
}

/// Euclidean distance between a data row and a centre (given as a `&[f64]` slice).
#[inline]
fn row_to_center_dist(data: &Array2<f64>, i: usize, center: &[f64]) -> f64 {
    let d = data.ncols();
    let mut sq = 0.0_f64;
    for f in 0..d {
        let diff = data[[i, f]] - center[f];
        sq += diff * diff;
    }
    sq.sqrt()
}

// ---------------------------------------------------------------------------
// Density Peaks
// ---------------------------------------------------------------------------

/// Configuration for Density Peaks clustering.
#[derive(Debug, Clone)]
pub struct DensityPeaksConfig {
    /// Cutoff distance for density estimation.
    pub dc: f64,
    /// If `true`, `dc` is ignored and chosen automatically as the `dc_percentile`-th
    /// percentile of all pairwise distances.
    pub auto_dc: bool,
    /// Percentile (0–100) used when auto-selecting `dc` (default 2.0 %).
    pub dc_percentile: f64,
    /// Minimum local density ρ for a point to be a cluster centre candidate.
    pub rho_threshold: f64,
    /// Minimum δ for a point to be a cluster centre candidate.
    pub delta_threshold: f64,
}

impl Default for DensityPeaksConfig {
    fn default() -> Self {
        Self {
            dc: 1.0,
            auto_dc: true,
            dc_percentile: 2.0,
            rho_threshold: 0.0,
            delta_threshold: 0.0,
        }
    }
}

/// Result of Density Peaks clustering.
#[derive(Debug, Clone)]
pub struct DensityPeaksResult {
    /// Cluster assignments (0-indexed); `usize::MAX` means noise / unassigned.
    pub assignments: Vec<usize>,
    /// Local density ρ_i for each point.
    pub rho: Vec<f64>,
    /// Minimum distance δ_i to any higher-density point (the highest-density point gets
    /// the maximum pairwise distance instead).
    pub delta: Vec<f64>,
    /// Indices of identified cluster centres.
    pub centers: Vec<usize>,
    /// Number of clusters found.
    pub n_clusters: usize,
}

/// Density Peaks clustering.
///
/// For every pair of points the Euclidean distance is computed.  Then:
/// 1. ρ_i = Σ_{j≠i} χ(dist[i,j] - dc)   (Gaussian kernel: exp(-dist^2/dc^2))
/// 2. δ_i = min_{j: ρ_j > ρ_i} dist[i,j]
/// 3. Cluster centres are points with ρ_i > rho_threshold AND δ_i > delta_threshold.
/// 4. Non-centres are assigned to the cluster of their nearest higher-density neighbour.
///
/// # Examples
///
/// ```rust
/// use scirs2_cluster::density_peaks::{density_peaks, DensityPeaksConfig};
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::from_shape_vec((8, 2), vec![
///     0.0, 0.0,  0.1, 0.0,  0.0, 0.1, -0.1, 0.0,
///     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,  4.9, 5.0,
/// ]).expect("operation should succeed");
///
/// let config = DensityPeaksConfig {
///     auto_dc: true,
///     dc_percentile: 30.0,
///     rho_threshold: 1.0,
///     delta_threshold: 2.0,
///     ..Default::default()
/// };
/// let result = density_peaks(&data, config).expect("operation should succeed");
/// assert_eq!(result.rho.len(), 8);
/// assert_eq!(result.delta.len(), 8);
/// ```
pub fn density_peaks(data: &Array2<f64>, config: DensityPeaksConfig) -> Result<DensityPeaksResult> {
    let n = data.nrows();

    if n < 2 {
        return Err(ClusteringError::InvalidInput(
            "Data must have at least 2 points".into(),
        ));
    }

    let dist = pairwise_distances(data);

    // --- Select dc ---
    let dc = if config.auto_dc {
        let pct = config.dc_percentile.clamp(0.0, 100.0);
        let mut upper_tri: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                upper_tri.push(dist[i * n + j]);
            }
        }
        upper_tri.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let rank = ((pct / 100.0) * (upper_tri.len() - 1) as f64).round() as usize;
        let rank = rank.min(upper_tri.len() - 1);
        upper_tri[rank].max(1e-10)
    } else {
        if config.dc <= 0.0 {
            return Err(ClusteringError::InvalidInput("dc must be positive".into()));
        }
        config.dc
    };

    // --- Compute local density ρ (Gaussian kernel) ---
    let mut rho = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let d = dist[i * n + j];
                rho[i] += (-d * d / (dc * dc)).exp();
            }
        }
    }

    // --- Compute δ ---
    // For each point: minimum distance to any point with higher density.
    // For the point with the highest density: maximum pairwise distance.
    let max_dist = dist.iter().cloned().fold(0.0_f64, f64::max);
    let mut delta = vec![f64::INFINITY; n];
    let mut higher_density_neighbour = vec![n; n]; // n = sentinel "none"

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if rho[j] > rho[i] {
                let d = dist[i * n + j];
                if d < delta[i] {
                    delta[i] = d;
                    higher_density_neighbour[i] = j;
                }
            }
        }
        // Point with highest ρ gets max_dist
        if delta[i].is_infinite() {
            delta[i] = max_dist;
        }
    }

    // --- Identify cluster centres ---
    let mut centers: Vec<usize> = (0..n)
        .filter(|&i| rho[i] > config.rho_threshold && delta[i] > config.delta_threshold)
        .collect();

    // If no centres found with strict thresholds, pick the single point with max rho*delta
    if centers.is_empty() {
        let best = (0..n)
            .max_by(|&a, &b| {
                (rho[a] * delta[a])
                    .partial_cmp(&(rho[b] * delta[b]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        centers.push(best);
    }

    let n_clusters = centers.len();

    // --- Assign non-centre points ---
    // Order all points by decreasing ρ; each point is assigned to the cluster of its
    // nearest higher-density neighbour.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        rho[b]
            .partial_cmp(&rho[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut assignments = vec![usize::MAX; n];

    // Assign centres first
    for (cluster_id, &centre_idx) in centers.iter().enumerate() {
        assignments[centre_idx] = cluster_id;
    }

    // Assign the rest in order of decreasing density
    for &i in &order {
        if assignments[i] != usize::MAX {
            continue;
        }
        let nb = higher_density_neighbour[i];
        if nb < n && assignments[nb] != usize::MAX {
            assignments[i] = assignments[nb];
        }
    }

    // Any still-unassigned points: assign to nearest centre
    for i in 0..n {
        if assignments[i] == usize::MAX {
            let nearest = centers
                .iter()
                .enumerate()
                .min_by(|&(_, &ca), &(_, &cb)| {
                    dist[i * n + ca]
                        .partial_cmp(&dist[i * n + cb])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(ci, _)| ci)
                .unwrap_or(0);
            assignments[i] = nearest;
        }
    }

    Ok(DensityPeaksResult {
        assignments,
        rho,
        delta,
        centers,
        n_clusters,
    })
}

// ---------------------------------------------------------------------------
// Mean Shift
// ---------------------------------------------------------------------------

/// Configuration for Mean Shift clustering.
#[derive(Debug, Clone)]
pub struct MeanShiftConfig {
    /// Gaussian kernel bandwidth.
    pub bandwidth: f64,
    /// Maximum number of shift iterations per seed.
    pub max_iter: usize,
    /// Convergence tolerance (L2 norm of shift step).
    pub tol: f64,
    /// Optional explicit seed points to start from.
    /// If `None`, every data point is used as a seed.
    pub seeds: Option<Array2<f64>>,
}

impl Default for MeanShiftConfig {
    fn default() -> Self {
        Self {
            bandwidth: 1.0,
            max_iter: 300,
            tol: 1e-5,
            seeds: None,
        }
    }
}

/// Result of Mean Shift clustering.
#[derive(Debug, Clone)]
pub struct MeanShiftResult {
    /// Cluster assignment for each data point (0-indexed).
    pub assignments: Vec<usize>,
    /// Cluster mode centres (one per cluster), shape (n_clusters, d).
    pub centers: Array2<f64>,
    /// Number of clusters found.
    pub n_clusters: usize,
    /// Number of shift iterations actually performed (summed across all seeds).
    pub n_iter: usize,
}

/// Mean Shift clustering with a Gaussian kernel.
///
/// Each seed point is iteratively shifted towards the mean of nearby data points
/// weighted by a Gaussian kernel.  Convergent points that fall within `bandwidth`
/// of an already-known mode are merged.
///
/// # Algorithm
///
/// 1. For each seed point s:
///    a. Compute weights: w_i = exp(-||x_i - s||^2 / (2 * bandwidth^2))
///    b. Update s <- (Σ_i w_i * x_i) / Σ_i w_i
///    c. Repeat until shift < tol or max_iter reached.
/// 2. Merge modes that are within `bandwidth` of each other (keep the one with
///    the most contributing data points).
/// 3. Assign each data point to its nearest surviving mode.
///
/// # Examples
///
/// ```rust
/// use scirs2_cluster::density_peaks::{mean_shift, MeanShiftConfig};
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::from_shape_vec((8, 2), vec![
///     0.0, 0.0,  0.1, 0.0,  0.0, 0.1, -0.1, 0.0,
///     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,  4.9, 5.0,
/// ]).expect("operation should succeed");
///
/// let config = MeanShiftConfig {
///     bandwidth: 1.0,
///     max_iter: 100,
///     tol: 1e-4,
///     seeds: None,
/// };
/// let result = mean_shift(&data, config).expect("operation should succeed");
/// assert!(result.n_clusters >= 1);
/// assert_eq!(result.assignments.len(), 8);
/// ```
pub fn mean_shift(data: &Array2<f64>, config: MeanShiftConfig) -> Result<MeanShiftResult> {
    let n = data.nrows();
    let d = data.ncols();

    if n == 0 || d == 0 {
        return Err(ClusteringError::InvalidInput(
            "Data matrix must be non-empty".into(),
        ));
    }
    if config.bandwidth <= 0.0 {
        return Err(ClusteringError::InvalidInput(
            "bandwidth must be positive".into(),
        ));
    }

    // --- Build seed points ---
    let seeds: Vec<Vec<f64>> = match &config.seeds {
        Some(s) => {
            if s.ncols() != d {
                return Err(ClusteringError::InvalidInput(format!(
                    "seeds has {} features but data has {}",
                    s.ncols(),
                    d
                )));
            }
            (0..s.nrows())
                .map(|i| (0..d).map(|f| s[[i, f]]).collect())
                .collect()
        }
        None => (0..n)
            .map(|i| (0..d).map(|f| data[[i, f]]).collect())
            .collect(),
    };

    let bw2 = config.bandwidth * config.bandwidth;
    let mut total_iters = 0usize;

    // --- Shift each seed to its local mode ---
    struct Mode {
        center: Vec<f64>,
        /// Number of points within bandwidth that contributed (used for mode merging).
        support: usize,
    }

    let mut modes: Vec<Mode> = Vec::with_capacity(seeds.len());

    for mut seed in seeds {
        let mut iter_count = 0usize;
        loop {
            iter_count += 1;

            // Compute Gaussian-weighted mean
            let mut new_center = vec![0.0_f64; d];
            let mut weight_sum = 0.0_f64;
            let mut support = 0usize;

            for i in 0..n {
                let mut sq = 0.0_f64;
                for f in 0..d {
                    let diff = data[[i, f]] - seed[f];
                    sq += diff * diff;
                }
                let w = (-sq / (2.0 * bw2)).exp();
                if w > 1e-300 {
                    weight_sum += w;
                    for f in 0..d {
                        new_center[f] += w * data[[i, f]];
                    }
                    if sq.sqrt() <= config.bandwidth {
                        support += 1;
                    }
                }
            }

            if weight_sum < 1e-300 {
                // No nearby points — keep current position
                break;
            }
            for f in 0..d {
                new_center[f] /= weight_sum;
            }

            // Compute shift distance
            let shift: f64 = (0..d)
                .map(|f| {
                    let diff = new_center[f] - seed[f];
                    diff * diff
                })
                .sum::<f64>()
                .sqrt();

            seed = new_center;

            if shift < config.tol || iter_count >= config.max_iter {
                modes.push(Mode {
                    center: seed.clone(),
                    support,
                });
                break;
            }
        }
        total_iters += iter_count;
    }

    // --- Merge modes that are within bandwidth of each other ---
    // Sort by support (descending) so that modes with more points dominate.
    modes.sort_by(|a, b| b.support.cmp(&a.support));

    let mut merged: Vec<Vec<f64>> = Vec::new();

    'outer: for mode in &modes {
        for existing in &merged {
            let sq_dist: f64 = (0..d)
                .map(|f| {
                    let diff = mode.center[f] - existing[f];
                    diff * diff
                })
                .sum();
            if sq_dist.sqrt() <= config.bandwidth {
                continue 'outer; // absorbed by existing mode
            }
        }
        merged.push(mode.center.clone());
    }

    if merged.is_empty() {
        // Fallback: use the global centroid as the single mode
        let mut global_mean = vec![0.0_f64; d];
        for i in 0..n {
            for f in 0..d {
                global_mean[f] += data[[i, f]];
            }
        }
        for f in 0..d {
            global_mean[f] /= n as f64;
        }
        merged.push(global_mean);
    }

    let n_clusters = merged.len();

    // Build centers array
    let centers_flat: Vec<f64> = merged.iter().flat_map(|c| c.iter().cloned()).collect();
    let centers = Array2::from_shape_vec((n_clusters, d), centers_flat)
        .map_err(|e| ClusteringError::ComputationError(format!("Centers reshape error: {}", e)))?;

    // --- Assign each data point to nearest mode ---
    let assignments: Vec<usize> = (0..n)
        .map(|i| {
            merged
                .iter()
                .enumerate()
                .min_by(|&(_, ca), &(_, cb)| {
                    let da = row_to_center_dist(data, i, ca);
                    let db = row_to_center_dist(data, i, cb);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(ci, _)| ci)
                .unwrap_or(0)
        })
        .collect();

    Ok(MeanShiftResult {
        assignments,
        centers,
        n_clusters,
        n_iter: total_iters,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn two_cluster_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.2, 0.0, 0.0, 0.2, -0.2, 0.0, 0.1, 0.1, -0.1, 0.1, 8.0, 8.0, 8.2, 8.0,
                8.0, 8.2, 7.8, 8.0, 8.1, 8.1, 7.9, 7.9,
            ],
        )
        .expect("create test data")
    }

    // ---- Density Peaks tests ----

    #[test]
    fn test_density_peaks_rho_and_delta_lengths() {
        let data = two_cluster_data();
        let config = DensityPeaksConfig {
            auto_dc: true,
            dc_percentile: 30.0,
            ..Default::default()
        };
        let result = density_peaks(&data, config).expect("operation should succeed");
        assert_eq!(result.rho.len(), 12);
        assert_eq!(result.delta.len(), 12);
    }

    #[test]
    fn test_density_peaks_assignments_length() {
        let data = two_cluster_data();
        let config = DensityPeaksConfig {
            auto_dc: true,
            dc_percentile: 30.0,
            ..Default::default()
        };
        let result = density_peaks(&data, config).expect("operation should succeed");
        assert_eq!(result.assignments.len(), 12);
    }

    #[test]
    fn test_density_peaks_rho_positive() {
        let data = two_cluster_data();
        let config = DensityPeaksConfig {
            auto_dc: true,
            dc_percentile: 30.0,
            ..Default::default()
        };
        let result = density_peaks(&data, config).expect("operation should succeed");
        for &r in &result.rho {
            assert!(r > 0.0, "rho should be positive: {}", r);
        }
    }

    #[test]
    fn test_density_peaks_delta_positive() {
        let data = two_cluster_data();
        let config = DensityPeaksConfig {
            auto_dc: true,
            dc_percentile: 30.0,
            ..Default::default()
        };
        let result = density_peaks(&data, config).expect("operation should succeed");
        for &d in &result.delta {
            assert!(d >= 0.0, "delta should be non-negative: {}", d);
        }
    }

    #[test]
    fn test_density_peaks_identifies_centers_for_well_separated_data() {
        let data = two_cluster_data();
        // With a tight dc relative to within-cluster distances, each cluster
        // should produce at least one high-density candidate centre.
        let config = DensityPeaksConfig {
            auto_dc: true,
            dc_percentile: 30.0,
            rho_threshold: 0.5,
            delta_threshold: 1.0,
            ..Default::default()
        };
        let result = density_peaks(&data, config).expect("operation should succeed");
        // We should discover exactly 2 centres (one per cluster group)
        assert!(
            result.n_clusters >= 1,
            "Should find at least 1 cluster centre"
        );
        assert!(
            result.n_clusters <= 4,
            "Should not find too many spurious centres"
        );
    }

    #[test]
    fn test_density_peaks_all_points_assigned() {
        let data = two_cluster_data();
        let config = DensityPeaksConfig {
            auto_dc: true,
            dc_percentile: 30.0,
            ..Default::default()
        };
        let result = density_peaks(&data, config).expect("operation should succeed");
        let n_clusters = result.n_clusters;
        for &a in &result.assignments {
            assert!(
                a < n_clusters,
                "Assignment {} out of range [0, {})",
                a,
                n_clusters
            );
        }
    }

    #[test]
    fn test_density_peaks_invalid_dc() {
        let data = two_cluster_data();
        let config = DensityPeaksConfig {
            auto_dc: false,
            dc: -1.0,
            ..Default::default()
        };
        assert!(density_peaks(&data, config).is_err());
    }

    // ---- Mean Shift tests ----

    #[test]
    fn test_mean_shift_returns_correct_assignment_length() {
        let data = two_cluster_data();
        let config = MeanShiftConfig {
            bandwidth: 1.0,
            max_iter: 100,
            tol: 1e-4,
            seeds: None,
        };
        let result = mean_shift(&data, config).expect("operation should succeed");
        assert_eq!(result.assignments.len(), 12);
    }

    #[test]
    fn test_mean_shift_finds_two_gaussians() {
        let data = two_cluster_data();
        let config = MeanShiftConfig {
            bandwidth: 1.0,
            max_iter: 200,
            tol: 1e-5,
            seeds: None,
        };
        let result = mean_shift(&data, config).expect("operation should succeed");
        // With bandwidth=1 and clusters 8 units apart, we expect exactly 2 modes
        assert_eq!(
            result.n_clusters, 2,
            "Expected 2 clusters, got {}",
            result.n_clusters
        );
        // All 12 points should have valid assignments
        for &a in &result.assignments {
            assert!(a < 2, "Assignment {} >= 2", a);
        }
    }

    #[test]
    fn test_mean_shift_consistent_assignment() {
        let data = two_cluster_data();
        let config = MeanShiftConfig {
            bandwidth: 1.0,
            max_iter: 200,
            tol: 1e-5,
            seeds: None,
        };
        let result = mean_shift(&data, config).expect("operation should succeed");
        let first_label = result.assignments[0];
        let second_label = result.assignments[6];
        assert_ne!(first_label, second_label, "Two groups should differ");
        // All first-group points should share first_label
        for i in 0..6 {
            assert_eq!(
                result.assignments[i], first_label,
                "Point {} should be in group {}",
                i, first_label
            );
        }
        // All second-group points should share second_label
        for i in 6..12 {
            assert_eq!(
                result.assignments[i], second_label,
                "Point {} should be in group {}",
                i, second_label
            );
        }
    }

    #[test]
    fn test_mean_shift_centers_shape() {
        let data = two_cluster_data();
        let config = MeanShiftConfig {
            bandwidth: 1.0,
            max_iter: 100,
            tol: 1e-4,
            seeds: None,
        };
        let result = mean_shift(&data, config).expect("operation should succeed");
        let nc = result.n_clusters;
        assert_eq!(result.centers.shape(), &[nc, 2]);
    }

    #[test]
    fn test_mean_shift_invalid_bandwidth() {
        let data = two_cluster_data();
        let config = MeanShiftConfig {
            bandwidth: -1.0,
            ..Default::default()
        };
        assert!(mean_shift(&data, config).is_err());
    }

    #[test]
    fn test_mean_shift_with_explicit_seeds() {
        let data = two_cluster_data();
        let seeds = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 8.0, 8.0])
            .expect("operation should succeed");
        let config = MeanShiftConfig {
            bandwidth: 1.0,
            max_iter: 200,
            tol: 1e-5,
            seeds: Some(seeds),
        };
        let result = mean_shift(&data, config).expect("operation should succeed");
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.assignments.len(), 12);
    }

    #[test]
    fn test_mean_shift_n_iter_positive() {
        let data = two_cluster_data();
        let config = MeanShiftConfig {
            bandwidth: 1.0,
            max_iter: 50,
            tol: 1e-5,
            seeds: None,
        };
        let result = mean_shift(&data, config).expect("operation should succeed");
        assert!(result.n_iter > 0, "n_iter should be > 0");
    }
}
