//! Fuzzy C-Means (FCM) clustering implementation
//!
//! Fuzzy C-Means is a soft clustering algorithm where each data point belongs to
//! multiple clusters with varying degrees of membership. This generalization of
//! k-means allows points on cluster boundaries to have partial membership in both.
//!
//! # Algorithm
//!
//! 1. Initialize cluster centers (random or k-means++)
//! 2. Compute membership matrix U using inverse distance weighting
//! 3. Update cluster centers as weighted centroids
//! 4. Repeat 2-3 until convergence (change in U < tol)
//!
//! # Validity Indices
//!
//! - **Partition Coefficient (PC)**: Ranges [1/c, 1]; higher = crisper partition
//! - **Partition Entropy (PE)**: Ranges [0, log(c)]; lower = crisper partition
//!
//! # References
//!
//! Bezdek, J.C. (1981). "Pattern Recognition with Fuzzy Objective Function Algorithms."
//! Kluwer Academic Publishers, Norwell, MA, USA.

use crate::error::{ClusteringError, Result};
use scirs2_core::ndarray::{Array1, Array2};

/// Fuzzy C-Means clustering
///
/// # Examples
///
/// ```rust
/// use scirs2_cluster::fuzzy_cmeans::FuzzyCMeans;
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::from_shape_vec((6, 2), vec![
///     1.0_f64, 1.0, 1.1, 0.9, 0.9, 1.1,
///     5.0, 5.0, 5.1, 4.9, 4.9, 5.1,
/// ]).expect("operation should succeed");
///
/// let mut fcm = FuzzyCMeans::new(2, 2.0, 100, 1e-4);
/// fcm.fit(&data).expect("fit failed");
/// let labels = fcm.predict(&data).expect("predict failed");
/// ```
pub struct FuzzyCMeans {
    /// Number of clusters
    n_clusters: usize,
    /// Fuzziness exponent (m > 1); m=2.0 is typical
    m: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance on the membership matrix change
    tol: f64,
    /// Cluster centers of shape (n_clusters, n_features)
    centers: Array2<f64>,
    /// Membership matrix of shape (n_samples, n_clusters)
    membership: Array2<f64>,
    /// Whether the model has been fitted
    fitted: bool,
    /// Optional random seed for initialization
    pub random_seed: Option<u64>,
}

impl FuzzyCMeans {
    /// Create a new Fuzzy C-Means instance
    ///
    /// # Arguments
    ///
    /// * `n_clusters` - Number of clusters (c >= 2)
    /// * `m` - Fuzziness exponent; must be > 1.0 (typically 2.0)
    /// * `max_iter` - Maximum iterations before stopping
    /// * `tol` - Convergence tolerance (Frobenius norm of membership change)
    pub fn new(n_clusters: usize, m: f64, max_iter: usize, tol: f64) -> Self {
        Self {
            n_clusters,
            m,
            max_iter,
            tol,
            centers: Array2::zeros((0, 0)),
            membership: Array2::zeros((0, 0)),
            fitted: false,
            random_seed: None,
        }
    }

    /// Set a random seed for reproducible initialization
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Fit the model to data
    ///
    /// Uses random initialization of the membership matrix followed by
    /// iterative updates of centers and memberships.
    ///
    /// # Arguments
    ///
    /// * `x` - Training data of shape (n_samples, n_features)
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<&Self> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput(
                "Training data must not be empty".into(),
            ));
        }

        if self.n_clusters < 2 {
            return Err(ClusteringError::InvalidInput(
                "Number of clusters must be at least 2".into(),
            ));
        }

        if self.n_clusters > n_samples {
            return Err(ClusteringError::InvalidInput(format!(
                "n_clusters ({}) must not exceed n_samples ({})",
                self.n_clusters, n_samples
            )));
        }

        if self.m <= 1.0 {
            return Err(ClusteringError::InvalidInput(
                "Fuzziness m must be strictly greater than 1.0".into(),
            ));
        }

        // Initialize membership matrix randomly (rows sum to 1)
        let mut u = self.initialize_membership(n_samples)?;

        // Iterative FCM updates
        for _iter in 0..self.max_iter {
            // Update centers
            let centers = compute_centers(x, &u, self.m, self.n_clusters, n_features)?;

            // Update membership matrix
            let u_new = compute_membership(x, &centers, self.m, self.n_clusters)?;

            // Check convergence: Frobenius norm of change in U
            let mut change = 0.0_f64;
            for i in 0..n_samples {
                for k in 0..self.n_clusters {
                    let diff = u_new[[i, k]] - u[[i, k]];
                    change += diff * diff;
                }
            }
            let change = change.sqrt();

            u = u_new;

            if change < self.tol {
                break;
            }
        }

        // Final center computation
        let centers = compute_centers(x, &u, self.m, self.n_clusters, n_features)?;

        self.centers = centers;
        self.membership = u;
        self.fitted = true;

        Ok(self)
    }

    /// Predict hard cluster labels (argmax of membership) for new data
    ///
    /// # Arguments
    ///
    /// * `x` - Data of shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Array of cluster indices of shape (n_samples,)
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<usize>> {
        let soft = self.predict_soft(x)?;
        let n_samples = soft.shape()[0];
        let mut labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut best_k = 0;
            let mut best_u = -1.0_f64;
            for k in 0..self.n_clusters {
                if soft[[i, k]] > best_u {
                    best_u = soft[[i, k]];
                    best_k = k;
                }
            }
            labels[i] = best_k;
        }

        Ok(labels)
    }

    /// Predict soft (fuzzy) membership matrix for new data
    ///
    /// # Arguments
    ///
    /// * `x` - Data of shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Membership matrix of shape (n_samples, n_clusters)
    pub fn predict_soft(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(ClusteringError::InvalidState(
                "FuzzyCMeans must be fitted before calling predict".into(),
            ));
        }

        let n_features = x.shape()[1];
        if n_features != self.centers.shape()[1] {
            return Err(ClusteringError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.centers.shape()[1],
                n_features
            )));
        }

        compute_membership(x, &self.centers, self.m, self.n_clusters)
    }

    /// Return cluster centers of shape (n_clusters, n_features)
    pub fn centers(&self) -> &Array2<f64> {
        &self.centers
    }

    /// Return the fitted membership matrix of shape (n_samples, n_clusters)
    pub fn membership(&self) -> &Array2<f64> {
        &self.membership
    }

    /// Compute the Partition Coefficient (PC) validity index
    ///
    /// PC = (1/N) * sum_i sum_k u_{ik}^2
    ///
    /// Range: [1/c, 1]; higher = crisper, non-overlapping partition.
    pub fn partition_coefficient(&self) -> f64 {
        if !self.fitted || self.membership.shape()[0] == 0 {
            return 0.0;
        }
        let n_samples = self.membership.shape()[0] as f64;
        let sum_sq: f64 = self.membership.iter().map(|&u| u * u).sum();
        sum_sq / n_samples
    }

    /// Compute the Partition Entropy (PE) validity index
    ///
    /// PE = -(1/N) * sum_i sum_k u_{ik} * log(u_{ik})
    ///
    /// Range: [0, log(c)]; lower = crisper partition.
    pub fn partition_entropy(&self) -> f64 {
        if !self.fitted || self.membership.shape()[0] == 0 {
            return 0.0;
        }
        let n_samples = self.membership.shape()[0] as f64;
        let mut entropy = 0.0_f64;

        for &u in self.membership.iter() {
            if u > 0.0 {
                entropy -= u * u.ln();
            }
        }

        entropy / n_samples
    }

    /// Initialize membership matrix with random values that sum to 1 per row
    fn initialize_membership(&self, n_samples: usize) -> Result<Array2<f64>> {
        let mut u = Array2::zeros((n_samples, self.n_clusters));
        let mut rng_state: u64 = self.random_seed.unwrap_or(42).wrapping_add(9999);

        for i in 0..n_samples {
            let mut row_sum = 0.0_f64;
            let mut raw = vec![0.0_f64; self.n_clusters];

            for k in 0..self.n_clusters {
                rng_state = lcg_next(rng_state);
                // Map u64 to (0, 1)
                let val = (rng_state as f64) / (u64::MAX as f64) + 1e-10;
                raw[k] = val;
                row_sum += val;
            }

            for k in 0..self.n_clusters {
                u[[i, k]] = raw[k] / row_sum;
            }
        }

        Ok(u)
    }
}

/// Compute cluster centers from membership matrix
///
/// center_k = sum_i (u_{ik}^m * x_i) / sum_i (u_{ik}^m)
fn compute_centers(
    x: &Array2<f64>,
    u: &Array2<f64>,
    m: f64,
    n_clusters: usize,
    n_features: usize,
) -> Result<Array2<f64>> {
    let n_samples = x.shape()[0];
    let mut centers = Array2::zeros((n_clusters, n_features));

    for k in 0..n_clusters {
        let mut weight_sum = 0.0_f64;
        for i in 0..n_samples {
            let w = u[[i, k]].powf(m);
            weight_sum += w;
            for f in 0..n_features {
                centers[[k, f]] += w * x[[i, f]];
            }
        }

        if weight_sum < 1e-300 {
            return Err(ClusteringError::ComputationError(format!(
                "Cluster {} has near-zero total membership; consider reducing n_clusters",
                k
            )));
        }

        for f in 0..n_features {
            centers[[k, f]] /= weight_sum;
        }
    }

    Ok(centers)
}

/// Compute membership matrix from cluster centers
///
/// u_{ik} = 1 / sum_j ( ||x_i - c_k|| / ||x_i - c_j|| )^(2/(m-1))
fn compute_membership(
    x: &Array2<f64>,
    centers: &Array2<f64>,
    m: f64,
    n_clusters: usize,
) -> Result<Array2<f64>> {
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];
    let exp = 2.0 / (m - 1.0);

    let mut u = Array2::zeros((n_samples, n_clusters));

    for i in 0..n_samples {
        // Compute distances from x_i to each center
        let mut dists = vec![0.0_f64; n_clusters];
        let mut n_zero = 0usize;

        for k in 0..n_clusters {
            let mut dist_sq = 0.0_f64;
            for f in 0..n_features {
                let diff = x[[i, f]] - centers[[k, f]];
                dist_sq += diff * diff;
            }
            dists[k] = dist_sq.sqrt();
            if dists[k] < 1e-300 {
                n_zero += 1;
            }
        }

        // If point coincides with one or more centers, assign full membership equally
        if n_zero > 0 {
            let eq_mem = 1.0 / n_zero as f64;
            for k in 0..n_clusters {
                u[[i, k]] = if dists[k] < 1e-300 { eq_mem } else { 0.0 };
            }
            continue;
        }

        // Normal case: compute inverse-distance weighted memberships
        let mut row_sum = 0.0_f64;
        for k in 0..n_clusters {
            let mut inner_sum = 0.0_f64;
            for j in 0..n_clusters {
                inner_sum += (dists[k] / dists[j]).powf(exp);
            }
            // u_ik = 1 / inner_sum
            u[[i, k]] = 1.0 / inner_sum;
            row_sum += u[[i, k]];
        }

        // Normalize (defensive)
        if row_sum > 1e-300 {
            for k in 0..n_clusters {
                u[[i, k]] /= row_sum;
            }
        }
    }

    Ok(u)
}

/// Linear Congruential Generator (Knuth MMIX parameters)
#[inline]
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn two_cluster_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (12, 2),
            vec![
                1.0, 1.0, 1.1, 0.9, 0.9, 1.1, 1.2, 1.0, 1.0, 1.2, 0.8, 0.8, 5.0, 5.0, 5.1, 4.9,
                4.9, 5.1, 5.2, 5.0, 5.0, 5.2, 4.8, 4.8,
            ],
        )
        .expect("Failed to create test data")
    }

    #[test]
    fn test_fcm_fit_two_clusters() {
        let data = two_cluster_data();
        let mut fcm = FuzzyCMeans::new(2, 2.0, 100, 1e-5).with_seed(42);
        let result = fcm.fit(&data);
        assert!(result.is_ok(), "FCM fit should succeed: {:?}", result.err());
    }

    #[test]
    fn test_fcm_predict_hard_assignment() {
        let data = two_cluster_data();
        let mut fcm = FuzzyCMeans::new(2, 2.0, 200, 1e-6).with_seed(7);
        fcm.fit(&data).expect("fit failed");

        let labels = fcm.predict(&data).expect("predict failed");
        assert_eq!(labels.len(), 12);

        // All labels should be 0 or 1
        for &l in labels.iter() {
            assert!(l < 2, "label {} out of range", l);
        }

        // Points in the same cluster should get the same label
        let first_half_label = labels[0];
        let second_half_label = labels[6];
        assert_ne!(
            first_half_label, second_half_label,
            "Well-separated clusters should have different labels"
        );

        for i in 0..6 {
            assert_eq!(
                labels[i], first_half_label,
                "First-half point {} mislabeled",
                i
            );
        }
        for i in 6..12 {
            assert_eq!(
                labels[i], second_half_label,
                "Second-half point {} mislabeled",
                i
            );
        }
    }

    #[test]
    fn test_fcm_predict_soft_memberships_sum_to_one() {
        let data = two_cluster_data();
        let mut fcm = FuzzyCMeans::new(2, 2.0, 100, 1e-5).with_seed(3);
        fcm.fit(&data).expect("fit failed");

        let soft = fcm.predict_soft(&data).expect("predict_soft failed");
        assert_eq!(soft.shape(), &[12, 2]);

        for i in 0..12 {
            let row_sum: f64 = (0..2).map(|k| soft[[i, k]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-9,
                "Row {} memberships sum to {}, expected 1.0",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_fcm_membership_matrix_properties() {
        let data = two_cluster_data();
        let mut fcm = FuzzyCMeans::new(2, 2.0, 100, 1e-5).with_seed(11);
        fcm.fit(&data).expect("fit failed");

        let mem = fcm.membership();
        assert_eq!(mem.shape(), &[12, 2]);

        // All membership values in [0, 1]
        for &v in mem.iter() {
            assert!(v >= 0.0 && v <= 1.0 + 1e-9, "Membership {} out of range", v);
        }

        // Each row sums to approximately 1
        for i in 0..12 {
            let s: f64 = (0..2).map(|k| mem[[i, k]]).sum();
            assert!((s - 1.0).abs() < 1e-9, "Row {} sum = {}", i, s);
        }
    }

    #[test]
    fn test_fcm_centers_shape() {
        let data = two_cluster_data();
        let mut fcm = FuzzyCMeans::new(3, 2.0, 100, 1e-5).with_seed(55);
        fcm.fit(&data).expect("fit failed");

        let c = fcm.centers();
        assert_eq!(c.shape(), &[3, 2]);
    }

    #[test]
    fn test_fcm_partition_coefficient() {
        let data = two_cluster_data();
        let mut fcm = FuzzyCMeans::new(2, 2.0, 200, 1e-6).with_seed(42);
        fcm.fit(&data).expect("fit failed");

        let pc = fcm.partition_coefficient();
        // PC in [1/c, 1] = [0.5, 1.0] for c=2
        assert!(pc >= 0.5, "PC should be >= 1/c = 0.5, got {}", pc);
        assert!(pc <= 1.0 + 1e-9, "PC should be <= 1.0, got {}", pc);
    }

    #[test]
    fn test_fcm_partition_entropy() {
        let data = two_cluster_data();
        let mut fcm = FuzzyCMeans::new(2, 2.0, 200, 1e-6).with_seed(42);
        fcm.fit(&data).expect("fit failed");

        let pe = fcm.partition_entropy();
        let max_pe = (2.0_f64).ln(); // log(c) for c=2
                                     // PE in [0, log(c)]
        assert!(pe >= 0.0, "PE should be >= 0, got {}", pe);
        assert!(pe <= max_pe + 1e-9, "PE should be <= log(c), got {}", pe);
    }

    #[test]
    fn test_fcm_invalid_m() {
        let data = two_cluster_data();
        let mut fcm = FuzzyCMeans::new(2, 1.0, 100, 1e-5); // m=1 not allowed
        assert!(fcm.fit(&data).is_err());
    }

    #[test]
    fn test_fcm_too_many_clusters() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("create data");
        let mut fcm = FuzzyCMeans::new(5, 2.0, 100, 1e-5); // more clusters than samples
        assert!(fcm.fit(&data).is_err());
    }

    #[test]
    fn test_fcm_predict_before_fit() {
        let data = two_cluster_data();
        let fcm = FuzzyCMeans::new(2, 2.0, 100, 1e-5);
        assert!(fcm.predict(&data).is_err());
    }
}
