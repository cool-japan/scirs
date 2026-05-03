//! Fuzzy clustering algorithms
//!
//! This module provides:
//! - **Fuzzy C-Means (FCM)** — Bezdek (1981): soft membership based on inverse-distance weighting.
//! - **Possibilistic C-Means (PCM)** — Krishnapuram & Keller (1993): typicality values that are
//!   NOT constrained to sum to 1 per sample, allowing outlier detection.
//!
//! # References
//!
//! * Bezdek, J. C. (1981). *Pattern Recognition with Fuzzy Objective Function Algorithms.*
//!   Kluwer Academic Publishers.
//! * Krishnapuram, R., & Keller, J. M. (1993). A possibilistic approach to clustering.
//!   *IEEE Transactions on Fuzzy Systems*, 1(2), 98-110.

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Common helpers
// ---------------------------------------------------------------------------

/// Linear-congruential PRNG (Knuth MMIX) — avoids any external dependency.
#[derive(Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Compute squared Euclidean distance between data row `i` and center row `k`.
#[inline]
fn sq_dist_row_to_center(
    data: &Array2<f64>,
    i: usize,
    centers: &Array2<f64>,
    k: usize,
    n_features: usize,
) -> f64 {
    let mut d2 = 0.0_f64;
    for f in 0..n_features {
        let diff = data[[i, f]] - centers[[k, f]];
        d2 += diff * diff;
    }
    d2
}

/// Compute weighted cluster centers given a weight matrix.
///
/// `centers[k, f] = sum_i w[i,k]^m * x[i,f] / sum_i w[i,k]^m`
fn compute_weighted_centers(
    data: &Array2<f64>,
    weights: &Array2<f64>,
    m: f64,
    k: usize,
    n_features: usize,
) -> Result<Array2<f64>> {
    let n = data.nrows();
    let mut centers = Array2::<f64>::zeros((k, n_features));

    for c in 0..k {
        let mut wsum = 0.0_f64;
        for i in 0..n {
            let wm = weights[[i, c]].powf(m);
            wsum += wm;
            for f in 0..n_features {
                centers[[c, f]] += wm * data[[i, f]];
            }
        }
        if wsum < 1e-300 {
            return Err(ClusteringError::ComputationError(format!(
                "Cluster {} has near-zero total weight; consider reducing k",
                c
            )));
        }
        for f in 0..n_features {
            centers[[c, f]] /= wsum;
        }
    }

    Ok(centers)
}

// ---------------------------------------------------------------------------
// Fuzzy C-Means (FCM)
// ---------------------------------------------------------------------------

/// Configuration for Fuzzy C-Means.
#[derive(Debug, Clone)]
pub struct FuzzyCMeansConfig {
    /// Number of clusters.
    pub k: usize,
    /// Fuzziness exponent `m > 1` (default 2.0).  Higher values produce softer memberships.
    pub m: f64,
    /// Maximum number of EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance (Frobenius norm of the membership-matrix change).
    pub tol: f64,
    /// Random seed for membership initialisation.
    pub seed: u64,
}

impl Default for FuzzyCMeansConfig {
    fn default() -> Self {
        Self {
            k: 2,
            m: 2.0,
            max_iter: 300,
            tol: 1e-6,
            seed: 42,
        }
    }
}

/// Result of Fuzzy C-Means.
#[derive(Debug, Clone)]
pub struct FuzzyCMeansResult {
    /// n × k membership matrix; rows sum to 1.0.
    pub membership: Array2<f64>,
    /// k × d cluster centres.
    pub centers: Array2<f64>,
    /// Hard cluster assignments (argmax of each membership row).
    pub assignments: Vec<usize>,
    /// Final value of the FCM objective: `J = sum_{i,k} u_{ik}^m * ||x_i - c_k||^2`.
    pub objective: f64,
    /// Number of iterations actually performed.
    pub n_iter: usize,
}

/// Fuzzy C-Means clustering.
///
/// Implements the standard FCM-AO (alternating optimisation) algorithm:
///
/// 1. Randomly initialise membership matrix U (rows sum to 1).
/// 2. Compute cluster centres as weighted centroids.
/// 3. Recompute U from inverse-distance formula.
/// 4. Repeat steps 2-3 until convergence.
///
/// # Examples
///
/// ```rust
/// use scirs2_cluster::fuzzy::{fuzzy_c_means, FuzzyCMeansConfig};
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::from_shape_vec((8, 2), vec![
///     0.0, 0.0,  0.1, 0.1,  -0.1, 0.0,  0.0, -0.1,
///     5.0, 5.0,  5.1, 5.1,   4.9, 5.0,  5.0,  4.9,
/// ]).expect("operation should succeed");
///
/// let config = FuzzyCMeansConfig { k: 2, m: 2.0, max_iter: 100, tol: 1e-5, seed: 1 };
/// let result = fuzzy_c_means(&data, config).expect("operation should succeed");
/// assert_eq!(result.membership.shape(), &[8, 2]);
/// ```
pub fn fuzzy_c_means(data: &Array2<f64>, config: FuzzyCMeansConfig) -> Result<FuzzyCMeansResult> {
    let n = data.nrows();
    let d = data.ncols();
    let k = config.k;

    if n == 0 || d == 0 {
        return Err(ClusteringError::InvalidInput(
            "Data matrix must be non-empty".into(),
        ));
    }
    if k < 2 {
        return Err(ClusteringError::InvalidInput("k must be at least 2".into()));
    }
    if k > n {
        return Err(ClusteringError::InvalidInput(format!(
            "k ({}) must not exceed n_samples ({})",
            k, n
        )));
    }
    if config.m <= 1.0 {
        return Err(ClusteringError::InvalidInput(
            "Fuzziness m must be strictly greater than 1.0".into(),
        ));
    }

    // --- Initialise membership matrix (random, rows sum to 1) ---
    let mut u = {
        let mut rng = Lcg::new(config.seed);
        let mut mat = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let mut vals: Vec<f64> = (0..k).map(|_| rng.next_f64() + 1e-10).collect();
            let s: f64 = vals.iter().sum();
            for (c, v) in vals.iter_mut().enumerate() {
                mat[[i, c]] = *v / s;
            }
        }
        mat
    };

    let exp = 2.0 / (config.m - 1.0);
    let mut n_iter = 0usize;

    for _iter in 0..config.max_iter {
        n_iter += 1;

        // Update centres
        let centers = compute_weighted_centers(data, &u, config.m, k, d)?;

        // Update membership matrix
        let u_new = fcm_update_membership(data, &centers, config.m, exp, k, d)?;

        // Check convergence
        let mut change = 0.0_f64;
        for i in 0..n {
            for c in 0..k {
                let diff = u_new[[i, c]] - u[[i, c]];
                change += diff * diff;
            }
        }
        u = u_new;
        if change.sqrt() < config.tol {
            break;
        }
    }

    let centers = compute_weighted_centers(data, &u, config.m, k, d)?;
    let objective = fcm_objective(data, &u, &centers, config.m, k, d);
    let assignments = hard_assignments(&u, n, k);

    Ok(FuzzyCMeansResult {
        membership: u,
        centers,
        assignments,
        objective,
        n_iter,
    })
}

/// FCM membership update:
/// `u_{ik} = 1 / sum_j ( ||x_i - c_k|| / ||x_i - c_j|| )^exp`
fn fcm_update_membership(
    data: &Array2<f64>,
    centers: &Array2<f64>,
    _m: f64,
    exp: f64,
    k: usize,
    d: usize,
) -> Result<Array2<f64>> {
    let n = data.nrows();
    let mut u = Array2::<f64>::zeros((n, k));

    for i in 0..n {
        let mut dists: Vec<f64> = (0..k)
            .map(|c| sq_dist_row_to_center(data, i, centers, c, d).sqrt())
            .collect();

        let n_zero = dists.iter().filter(|&&v| v < 1e-300).count();

        if n_zero > 0 {
            // Point coincides with one or more centres
            let eq_mem = 1.0 / n_zero as f64;
            for c in 0..k {
                u[[i, c]] = if dists[c] < 1e-300 { eq_mem } else { 0.0 };
            }
            continue;
        }

        // Normal case
        let mut row_sum = 0.0_f64;
        for c in 0..k {
            let inner: f64 = (0..k).map(|j| (dists[c] / dists[j]).powf(exp)).sum();
            u[[i, c]] = 1.0 / inner;
            row_sum += u[[i, c]];
        }
        // Normalise defensively
        if row_sum > 1e-300 {
            for c in 0..k {
                u[[i, c]] /= row_sum;
            }
        }
    }

    Ok(u)
}

/// Compute the FCM objective function.
fn fcm_objective(
    data: &Array2<f64>,
    u: &Array2<f64>,
    centers: &Array2<f64>,
    m: f64,
    k: usize,
    d: usize,
) -> f64 {
    let n = data.nrows();
    let mut obj = 0.0_f64;
    for i in 0..n {
        for c in 0..k {
            let dist2 = sq_dist_row_to_center(data, i, centers, c, d);
            obj += u[[i, c]].powf(m) * dist2;
        }
    }
    obj
}

/// Return hard assignments (argmax per row).
fn hard_assignments(u: &Array2<f64>, n: usize, k: usize) -> Vec<usize> {
    (0..n)
        .map(|i| {
            let mut best_c = 0;
            let mut best_u = -1.0_f64;
            for c in 0..k {
                if u[[i, c]] > best_u {
                    best_u = u[[i, c]];
                    best_c = c;
                }
            }
            best_c
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Possibilistic C-Means (PCM)
// ---------------------------------------------------------------------------

/// Configuration for Possibilistic C-Means.
#[derive(Debug, Clone)]
pub struct PossibilisticConfig {
    /// Number of clusters.
    pub k: usize,
    /// Fuzziness exponent `m > 1` (default 2.0).
    pub m: f64,
    /// Typicality bandwidth — controls the spread of the typicality function.
    /// If set to 0.0 the algorithm initialises η automatically from an FCM pre-run.
    pub eta: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance (Frobenius norm of typicality-matrix change).
    pub tol: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for PossibilisticConfig {
    fn default() -> Self {
        Self {
            k: 2,
            m: 2.0,
            eta: 0.0, // Auto-detect
            max_iter: 300,
            tol: 1e-6,
            seed: 42,
        }
    }
}

/// Result of Possibilistic C-Means.
#[derive(Debug, Clone)]
pub struct PossibilisticResult {
    /// n × k typicality matrix — values are in (0, 1] but do NOT sum to 1 per row.
    pub typicality: Array2<f64>,
    /// k × d cluster centres.
    pub centers: Array2<f64>,
    /// Hard assignments: argmax typicality per sample.
    pub assignments: Vec<usize>,
}

/// Possibilistic C-Means clustering.
///
/// Unlike FCM, the typicality values `t_{ik}` are NOT forced to sum to 1 across clusters.
/// A value close to 1 means the sample is highly typical of cluster k; close to 0 means
/// it is an outlier with respect to cluster k.
///
/// The membership update is:
/// `t_{ik} = 1 / (1 + (||x_i - c_k||^2 / eta_k)^{1/(m-1)})`
///
/// The `eta_k` bandwidth is either supplied explicitly or auto-initialised from an FCM
/// pre-run using the formula: `eta_k = sum_i u_{ik}^m * ||x_i - c_k||^2 / sum_i u_{ik}^m`.
///
/// # Examples
///
/// ```rust
/// use scirs2_cluster::fuzzy::{possibilistic_c_means, PossibilisticConfig};
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::from_shape_vec((8, 2), vec![
///     0.0, 0.0,  0.1, 0.1,  -0.1, 0.0,  0.0, -0.1,
///     5.0, 5.0,  5.1, 5.1,   4.9, 5.0,  5.0,  4.9,
/// ]).expect("operation should succeed");
///
/// let config = PossibilisticConfig { k: 2, m: 2.0, eta: 0.0, max_iter: 200, tol: 1e-5, seed: 1 };
/// let result = possibilistic_c_means(&data, config).expect("operation should succeed");
/// assert_eq!(result.typicality.shape(), &[8, 2]);
/// ```
pub fn possibilistic_c_means(
    data: &Array2<f64>,
    config: PossibilisticConfig,
) -> Result<PossibilisticResult> {
    let n = data.nrows();
    let d = data.ncols();
    let k = config.k;

    if n == 0 || d == 0 {
        return Err(ClusteringError::InvalidInput(
            "Data matrix must be non-empty".into(),
        ));
    }
    if k < 2 {
        return Err(ClusteringError::InvalidInput("k must be at least 2".into()));
    }
    if k > n {
        return Err(ClusteringError::InvalidInput(format!(
            "k ({}) must not exceed n_samples ({})",
            k, n
        )));
    }
    if config.m <= 1.0 {
        return Err(ClusteringError::InvalidInput(
            "Fuzziness m must be strictly greater than 1.0".into(),
        ));
    }

    // --- Obtain initial centres and η via FCM pre-run ---
    let fcm_config = FuzzyCMeansConfig {
        k,
        m: config.m,
        max_iter: 50,
        tol: 1e-4,
        seed: config.seed,
    };
    let fcm_result = fuzzy_c_means(data, fcm_config)?;
    let mut centers = fcm_result.centers.clone();

    // Compute η_k from FCM result (if not supplied)
    let eta_vec: Vec<f64> = if config.eta > 0.0 {
        vec![config.eta; k]
    } else {
        // η_k = sum_i u_{ik}^m * ||x_i - c_k||^2 / sum_i u_{ik}^m
        let u = &fcm_result.membership;
        (0..k)
            .map(|c| {
                let mut num = 0.0_f64;
                let mut den = 0.0_f64;
                for i in 0..n {
                    let wm = u[[i, c]].powf(config.m);
                    num += wm * sq_dist_row_to_center(data, i, &centers, c, d);
                    den += wm;
                }
                if den < 1e-300 {
                    1.0
                } else {
                    (num / den).max(1e-10)
                }
            })
            .collect()
    };

    // Initialise typicality matrix
    let mut t = pcm_update_typicality(data, &centers, &eta_vec, config.m, k, d)?;

    for _iter in 0..config.max_iter {
        // Update centres (same as FCM but using typicality instead of membership)
        let centers_new = compute_weighted_centers(data, &t, config.m, k, d)?;

        // Update typicality
        let t_new = pcm_update_typicality(data, &centers_new, &eta_vec, config.m, k, d)?;

        // Check convergence
        let mut change = 0.0_f64;
        for i in 0..n {
            for c in 0..k {
                let diff = t_new[[i, c]] - t[[i, c]];
                change += diff * diff;
            }
        }
        centers = centers_new;
        t = t_new;
        if change.sqrt() < config.tol {
            break;
        }
    }

    let assignments = hard_assignments(&t, n, k);

    Ok(PossibilisticResult {
        typicality: t,
        centers,
        assignments,
    })
}

/// PCM typicality update:
/// `t_{ik} = 1 / (1 + (dist^2 / eta_k)^{1/(m-1)})`
fn pcm_update_typicality(
    data: &Array2<f64>,
    centers: &Array2<f64>,
    eta: &[f64],
    m: f64,
    k: usize,
    d: usize,
) -> Result<Array2<f64>> {
    let n = data.nrows();
    let exp = 1.0 / (m - 1.0);
    let mut t = Array2::<f64>::zeros((n, k));

    for i in 0..n {
        for c in 0..k {
            let dist2 = sq_dist_row_to_center(data, i, centers, c, d);
            let eta_c = eta[c].max(1e-300);
            t[[i, c]] = 1.0 / (1.0 + (dist2 / eta_c).powf(exp));
        }
    }

    Ok(t)
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
            (16, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, -0.1, 0.0, 0.0, -0.1, 0.2, 0.0, -0.2, 0.0, 0.0, 0.2, 0.0, -0.2,
                8.0, 8.0, 8.1, 8.1, 7.9, 8.0, 8.0, 7.9, 8.2, 8.0, 7.8, 8.0, 8.0, 8.2, 8.0, 7.8,
            ],
        )
        .expect("create test data")
    }

    #[test]
    fn test_fcm_membership_rows_sum_to_one() {
        let data = two_cluster_data();
        let config = FuzzyCMeansConfig {
            k: 2,
            m: 2.0,
            max_iter: 200,
            tol: 1e-6,
            seed: 1,
        };
        let result = fuzzy_c_means(&data, config).expect("operation should succeed");
        let n = result.membership.nrows();
        let k = result.membership.ncols();
        for i in 0..n {
            let row_sum: f64 = (0..k).map(|c| result.membership[[i, c]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-9,
                "Row {} membership sum = {}, expected 1.0",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_fcm_membership_values_in_range() {
        let data = two_cluster_data();
        let config = FuzzyCMeansConfig::default();
        let result = fuzzy_c_means(&data, config).expect("operation should succeed");
        for &v in result.membership.iter() {
            assert!(
                v >= 0.0 && v <= 1.0 + 1e-12,
                "Membership {} out of [0,1]",
                v
            );
        }
    }

    #[test]
    fn test_fcm_well_separated_clusters_membership_sharp() {
        let data = two_cluster_data();
        let config = FuzzyCMeansConfig {
            k: 2,
            m: 2.0,
            max_iter: 300,
            tol: 1e-7,
            seed: 7,
        };
        let result = fuzzy_c_means(&data, config).expect("operation should succeed");
        let n = data.nrows();
        for i in 0..n {
            let max_u = (0..2)
                .map(|c| result.membership[[i, c]])
                .fold(0.0_f64, f64::max);
            // For well-separated clusters, the dominant membership should be very high
            assert!(
                max_u > 0.8,
                "Point {}: max membership {} too low (clusters not well-separated?)",
                i,
                max_u
            );
        }
    }

    #[test]
    fn test_fcm_hard_assignments_correct() {
        let data = two_cluster_data();
        let config = FuzzyCMeansConfig {
            k: 2,
            m: 2.0,
            max_iter: 300,
            tol: 1e-7,
            seed: 3,
        };
        let result = fuzzy_c_means(&data, config).expect("operation should succeed");
        // First 8 points near (0,0), last 8 near (8,8)
        let first_label = result.assignments[0];
        let second_label = result.assignments[8];
        assert_ne!(first_label, second_label, "Two clusters should differ");
        for i in 0..8 {
            assert_eq!(
                result.assignments[i], first_label,
                "Point {} should be in cluster {}",
                i, first_label
            );
        }
        for i in 8..16 {
            assert_eq!(
                result.assignments[i], second_label,
                "Point {} should be in cluster {}",
                i, second_label
            );
        }
    }

    #[test]
    fn test_fcm_centers_shape() {
        let data = two_cluster_data();
        let config = FuzzyCMeansConfig {
            k: 3,
            ..FuzzyCMeansConfig::default()
        };
        let result = fuzzy_c_means(&data, config).expect("operation should succeed");
        assert_eq!(result.centers.shape(), &[3, 2]);
    }

    #[test]
    fn test_fcm_objective_positive() {
        let data = two_cluster_data();
        let config = FuzzyCMeansConfig::default();
        let result = fuzzy_c_means(&data, config).expect("operation should succeed");
        assert!(result.objective >= 0.0, "Objective must be non-negative");
    }

    #[test]
    fn test_fcm_invalid_m() {
        let data = two_cluster_data();
        let config = FuzzyCMeansConfig {
            m: 1.0,
            ..FuzzyCMeansConfig::default()
        };
        assert!(fuzzy_c_means(&data, config).is_err());
    }

    #[test]
    fn test_fcm_k_exceeds_n() {
        let data =
            Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).expect("data");
        let config = FuzzyCMeansConfig {
            k: 5,
            ..FuzzyCMeansConfig::default()
        };
        assert!(fuzzy_c_means(&data, config).is_err());
    }

    // --- Possibilistic C-Means tests ---

    #[test]
    fn test_pcm_typicality_shape() {
        let data = two_cluster_data();
        let config = PossibilisticConfig {
            k: 2,
            ..PossibilisticConfig::default()
        };
        let result = possibilistic_c_means(&data, config).expect("operation should succeed");
        assert_eq!(result.typicality.shape(), &[16, 2]);
    }

    #[test]
    fn test_pcm_typicality_in_01() {
        let data = two_cluster_data();
        let config = PossibilisticConfig {
            k: 2,
            m: 2.0,
            eta: 0.0,
            max_iter: 100,
            tol: 1e-5,
            seed: 5,
        };
        let result = possibilistic_c_means(&data, config).expect("operation should succeed");
        for &v in result.typicality.iter() {
            assert!(
                v >= 0.0 && v <= 1.0 + 1e-10,
                "Typicality {} out of [0,1]",
                v
            );
        }
    }

    #[test]
    fn test_pcm_typicality_does_not_sum_to_one() {
        // For PCM the row sum is NOT constrained to 1; it can take any value in (0, k].
        let data = two_cluster_data();
        let config = PossibilisticConfig {
            k: 2,
            m: 2.0,
            eta: 0.0,
            max_iter: 100,
            tol: 1e-5,
            seed: 8,
        };
        let result = possibilistic_c_means(&data, config).expect("operation should succeed");
        let n = result.typicality.nrows();
        let k = result.typicality.ncols();
        // At least some rows should NOT sum to 1 (within ε).
        // For well-separated data with 2 clusters, points near cluster 0 have high typicality
        // for cluster 0 and low for cluster 1, but the sum is generally not exactly 1.
        // We just verify the sums are not all exactly 1.
        let mut all_one = true;
        for i in 0..n {
            let s: f64 = (0..k).map(|c| result.typicality[[i, c]]).sum();
            if (s - 1.0).abs() > 1e-3 {
                all_one = false;
                break;
            }
        }
        assert!(
            !all_one,
            "PCM typicality rows should generally NOT sum to 1.0 for well-separated data"
        );
    }

    #[test]
    fn test_pcm_centers_shape() {
        let data = two_cluster_data();
        let config = PossibilisticConfig {
            k: 2,
            ..PossibilisticConfig::default()
        };
        let result = possibilistic_c_means(&data, config).expect("operation should succeed");
        assert_eq!(result.centers.shape(), &[2, 2]);
    }

    #[test]
    fn test_pcm_invalid_k() {
        let data = two_cluster_data();
        let config = PossibilisticConfig {
            k: 1,
            ..PossibilisticConfig::default()
        };
        assert!(possibilistic_c_means(&data, config).is_err());
    }

    #[test]
    fn test_pcm_explicit_eta() {
        let data = two_cluster_data();
        let config = PossibilisticConfig {
            k: 2,
            m: 2.0,
            eta: 5.0,
            max_iter: 100,
            tol: 1e-5,
            seed: 9,
        };
        let result = possibilistic_c_means(&data, config);
        assert!(result.is_ok(), "PCM with explicit eta should succeed");
    }
}
