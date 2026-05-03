//! Enhanced hierarchical clustering algorithms
//!
//! This module provides advanced hierarchical clustering methods beyond standard
//! agglomerative clustering, including Ward linkage, divisive (DIANA), and
//! consensus-based hierarchical approaches.
//!
//! # Algorithms
//!
//! - **Ward**: Minimum variance criterion for merging clusters
//! - **Dendrogram**: Full merge-tree structure with flexible cut operations
//! - **Divisive (DIANA)**: Top-down divisive hierarchical clustering
//! - **Consensus Clustering**: Stability-based consensus via repeated subsampling

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use std::collections::{HashMap, HashSet};

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Dendrogram
// ---------------------------------------------------------------------------

/// A merge step in a dendrogram.
#[derive(Debug, Clone)]
pub struct Merge {
    /// Index of the first cluster merged (< n_samples = leaf).
    pub cluster_a: usize,
    /// Index of the second cluster merged.
    pub cluster_b: usize,
    /// Linkage distance at which the merge occurs.
    pub distance: f64,
    /// Total number of original samples in the merged cluster.
    pub size: usize,
}

/// Full merge-tree produced by agglomerative clustering.
///
/// Contains `n - 1` merge records for `n` leaves.
#[derive(Debug, Clone)]
pub struct Dendrogram {
    /// Ordered list of merges (from first/closest to last/farthest).
    pub merges: Vec<Merge>,
    /// Number of original data points (leaves).
    pub n_samples: usize,
}

impl Dendrogram {
    /// Cut the dendrogram to obtain exactly `n_clusters` flat clusters.
    ///
    /// Returns a label vector of length `n_samples` with values in `0..n_clusters`.
    ///
    /// # Errors
    /// Returns an error if `n_clusters` is 0 or larger than `n_samples`.
    pub fn cut_at_n_clusters(&self, n_clusters: usize) -> Result<Vec<usize>> {
        let n = self.n_samples;
        if n_clusters == 0 || n_clusters > n {
            return Err(ClusteringError::InvalidInput(format!(
                "n_clusters must be in 1..={n}, got {n_clusters}"
            )));
        }

        // Perform exactly (n - n_clusters) merges to end up with n_clusters.
        let n_merges_to_do = n.saturating_sub(n_clusters);
        Ok(self.apply_merges_up_to(n_merges_to_do))
    }

    /// Cut the dendrogram at a distance threshold, returning flat cluster labels.
    pub fn cut_at_distance(&self, max_distance: f64) -> Result<Vec<usize>> {
        let n_merges = self
            .merges
            .iter()
            .take_while(|m| m.distance <= max_distance)
            .count();
        Ok(self.apply_merges_up_to(n_merges))
    }

    /// Internal: apply the first `n_merges` merges and return leaf labels.
    ///
    /// The k-th merge (0-indexed) creates internal node `n_samples + k`.
    /// We link the two merged sub-nodes into the new internal node so that
    /// leaf→root path-following gives the correct component root.
    fn apply_merges_up_to(&self, n_merges: usize) -> Vec<usize> {
        let n = self.n_samples;
        // Allocate parent for all possible nodes (leaves + internal).
        let capacity = n + self.merges.len() + 1;
        let mut parent: Vec<usize> = (0..capacity).collect();

        for (k, merge) in self.merges.iter().take(n_merges).enumerate() {
            let new_id = n + k;
            let a_root = find_root(&parent, merge.cluster_a);
            let b_root = find_root(&parent, merge.cluster_b);
            // Link both roots to the new internal node.
            parent[a_root] = new_id;
            parent[b_root] = new_id;
            // The new node is its own root for now.
            parent[new_id] = new_id;
        }

        // For each leaf, find its root cluster.
        let roots: Vec<usize> = (0..n).map(|i| find_root(&parent, i)).collect();

        // Remap root ids to 0-based consecutive labels.
        let mut id_map: HashMap<usize, usize> = HashMap::new();
        let mut next_label = 0usize;
        roots
            .iter()
            .map(|root| {
                let entry = id_map.entry(*root).or_insert_with(|| {
                    let l = next_label;
                    next_label += 1;
                    l
                });
                *entry
            })
            .collect()
    }
}

/// Path-compressed root finder for Union-Find.
fn find_root(parent: &[usize], mut x: usize) -> usize {
    while parent[x] != x {
        x = parent[x];
    }
    x
}

// ---------------------------------------------------------------------------
// Ward's method
// ---------------------------------------------------------------------------

/// Result from Ward hierarchical clustering.
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    /// Flat cluster labels (0-based) after cutting.
    pub labels: Array1<usize>,
    /// Number of clusters requested.
    pub n_clusters: usize,
    /// The full dendrogram (merge tree).
    pub dendrogram: Dendrogram,
    /// Inertia (within-cluster sum of squared deviations from centroids).
    pub inertia: f64,
}

/// Ward hierarchical clustering using the minimum variance criterion.
///
/// Merges pairs of clusters that minimize the total within-cluster variance
/// (Ward's linkage). Uses weighted centroid updates for O(n²) merging.
pub struct Ward;

impl Ward {
    /// Fit Ward hierarchical clustering and return `n_clusters` flat labels.
    ///
    /// # Arguments
    /// * `x` – Data matrix of shape `(n_samples, n_features)`.
    /// * `n_clusters` – Desired number of output clusters.
    ///
    /// # Errors
    /// Returns an error for empty inputs or invalid `n_clusters`.
    pub fn fit(x: ArrayView2<f64>, n_clusters: usize) -> Result<HierarchicalResult> {
        let (n_samples, n_features) = (x.shape()[0], x.shape()[1]);
        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput("Empty input data".into()));
        }
        if n_clusters == 0 || n_clusters > n_samples {
            return Err(ClusteringError::InvalidInput(format!(
                "n_clusters must be in 1..={n_samples}, got {n_clusters}"
            )));
        }

        // Centroid and size storage for all nodes (leaves + internal).
        // Capacity: 2*n_samples - 1 nodes total.
        let capacity = 2 * n_samples - 1;
        let mut all_centroids: Vec<Vec<f64>> = Vec::with_capacity(capacity);
        let mut all_sizes: Vec<f64> = Vec::with_capacity(capacity);

        // Initialise leaf centroids.
        for i in 0..n_samples {
            all_centroids.push(x.row(i).to_vec());
            all_sizes.push(1.0);
        }

        // Ward distance matrix: ward_dist[i][j] = Ward linkage distance.
        // Expanded dynamically as new nodes are created.
        let mut ward_dist: Vec<Vec<f64>> = vec![vec![0.0; n_samples]; n_samples];
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let d = ward_dist_between(&all_centroids[i], &all_centroids[j], 1.0, 1.0);
                ward_dist[i][j] = d;
                ward_dist[j][i] = d;
            }
        }

        let mut merges: Vec<Merge> = Vec::with_capacity(n_samples - 1);
        let mut active_ids: Vec<usize> = (0..n_samples).collect();
        let mut next_node = n_samples;

        for _ in 0..(n_samples - 1) {
            // Find closest pair among active clusters.
            let n_active = active_ids.len();
            let mut min_dist = f64::INFINITY;
            let mut best_ai = 0usize;
            let mut best_aj = 1usize;

            for a in 0..n_active {
                for b in (a + 1)..n_active {
                    let ia = active_ids[a];
                    let ib = active_ids[b];
                    let d = ward_dist[ia][ib];
                    if d < min_dist {
                        min_dist = d;
                        best_ai = a;
                        best_aj = b;
                    }
                }
            }

            let ia = active_ids[best_ai];
            let ib = active_ids[best_aj];
            let sa = all_sizes[ia];
            let sb = all_sizes[ib];
            let new_size = sa + sb;

            // Compute new centroid as weighted average.
            let new_centroid: Vec<f64> = (0..n_features)
                .map(|k| (all_centroids[ia][k] * sa + all_centroids[ib][k] * sb) / new_size)
                .collect();

            // Record merge.
            merges.push(Merge {
                cluster_a: ia,
                cluster_b: ib,
                distance: min_dist,
                size: new_size as usize,
            });

            let new_id = next_node;
            next_node += 1;

            all_centroids.push(new_centroid.clone());
            all_sizes.push(new_size);

            // Expand ward_dist to accommodate new_id.
            let current_len = ward_dist.len();
            for row in ward_dist.iter_mut() {
                row.push(0.0);
            }
            ward_dist.push(vec![0.0; current_len + 1]);

            // Compute Ward distances from new_id to remaining active clusters.
            let remaining: Vec<usize> = active_ids
                .iter()
                .enumerate()
                .filter(|&(idx, _)| idx != best_ai && idx != best_aj)
                .map(|(_, &id)| id)
                .collect();

            for &ik in &remaining {
                let sk = all_sizes[ik];
                let d = ward_dist_between(&new_centroid, &all_centroids[ik], new_size, sk);
                ward_dist[new_id][ik] = d;
                ward_dist[ik][new_id] = d;
            }

            // Update active_ids: remove ia and ib, add new_id.
            let remove_high = best_ai.max(best_aj);
            let remove_low = best_ai.min(best_aj);
            active_ids.remove(remove_high);
            active_ids.remove(remove_low);
            active_ids.push(new_id);
        }

        let dendrogram = Dendrogram { merges, n_samples };

        let label_vec = dendrogram.cut_at_n_clusters(n_clusters)?;
        let labels = Array1::from_vec(label_vec.clone());
        let inertia = compute_inertia(x, &label_vec, n_clusters);

        Ok(HierarchicalResult {
            labels,
            n_clusters,
            dendrogram,
            inertia,
        })
    }
}

/// Squared Ward linkage distance between two clusters with given centroids and sizes.
fn ward_dist_between(ca: &[f64], cb: &[f64], sa: f64, sb: f64) -> f64 {
    let merged = sa + sb;
    if merged == 0.0 {
        return 0.0;
    }
    let factor = (sa * sb) / merged;
    ca.iter()
        .zip(cb.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        * factor
}

/// Compute total within-cluster inertia (sum of squared distances to centroid).
fn compute_inertia(x: ArrayView2<f64>, labels: &[usize], n_clusters: usize) -> f64 {
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];
    let mut cluster_sums: Vec<Vec<f64>> = vec![vec![0.0; n_features]; n_clusters];
    let mut cluster_counts: Vec<usize> = vec![0; n_clusters];

    for i in 0..n_samples {
        let c = labels[i];
        if c < n_clusters {
            cluster_counts[c] += 1;
            for j in 0..n_features {
                cluster_sums[c][j] += x[[i, j]];
            }
        }
    }

    let centroids: Vec<Vec<f64>> = (0..n_clusters)
        .map(|c| {
            if cluster_counts[c] == 0 {
                vec![0.0; n_features]
            } else {
                cluster_sums[c]
                    .iter()
                    .map(|&v| v / cluster_counts[c] as f64)
                    .collect()
            }
        })
        .collect();

    let mut inertia = 0.0;
    for i in 0..n_samples {
        let c = labels[i];
        if c < n_clusters {
            inertia += centroids[c]
                .iter()
                .zip(x.row(i).iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>();
        }
    }
    inertia
}

// ---------------------------------------------------------------------------
// Divisive (DIANA) clustering
// ---------------------------------------------------------------------------

/// Result from divisive hierarchical clustering.
#[derive(Debug, Clone)]
pub struct DivisiveResult {
    /// Flat cluster labels (0-based) for each sample.
    pub labels: Array1<usize>,
    /// Number of final clusters.
    pub n_clusters: usize,
    /// Sequence of splits (cluster sizes) recorded top-down.
    pub split_history: Vec<(usize, usize)>,
    /// Total within-cluster inertia.
    pub inertia: f64,
}

/// Divisive hierarchical clustering (DIANA algorithm).
///
/// Starts with all points in one cluster and recursively splits the cluster
/// with the highest diameter (maximum average dissimilarity). Uses the splinter
/// heuristic from the original DIANA paper (Kaufman & Rousseeuw, 1990).
pub struct Divisive;

impl Divisive {
    /// Run divisive clustering, stopping when `n_clusters` clusters are obtained.
    ///
    /// # Arguments
    /// * `x` – Data matrix `(n_samples, n_features)`.
    /// * `n_clusters` – Target number of clusters.
    pub fn fit(x: ArrayView2<f64>, n_clusters: usize) -> Result<DivisiveResult> {
        let n_samples = x.shape()[0];
        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput("Empty input data".into()));
        }
        if n_clusters == 0 || n_clusters > n_samples {
            return Err(ClusteringError::InvalidInput(format!(
                "n_clusters must be in 1..={n_samples}, got {n_clusters}"
            )));
        }

        // Each cluster is a Vec<usize> of sample indices.
        let mut clusters: Vec<Vec<usize>> = vec![(0..n_samples).collect()];
        let mut split_history: Vec<(usize, usize)> = Vec::new();

        // Pre-compute pairwise Euclidean distances.
        let dist = precompute_distances(x);

        while clusters.len() < n_clusters {
            // Find the cluster with the largest average diameter.
            let split_idx = find_cluster_to_split(&clusters, &dist);

            let old_cluster = clusters.remove(split_idx);
            if old_cluster.len() == 1 {
                // Cannot split a singleton; put it back and stop.
                clusters.push(old_cluster);
                break;
            }

            let (group_a, group_b) = diana_split(&old_cluster, &dist)?;
            let a_size = group_a.len();
            let b_size = group_b.len();
            split_history.push((a_size, b_size));
            clusters.push(group_a);
            clusters.push(group_b);
        }

        // Assign labels.
        let mut labels = vec![0usize; n_samples];
        for (cluster_id, cluster) in clusters.iter().enumerate() {
            for &idx in cluster {
                labels[idx] = cluster_id;
            }
        }

        let actual_n = clusters.len();
        let inertia = compute_inertia(x, &labels, actual_n);

        Ok(DivisiveResult {
            labels: Array1::from_vec(labels),
            n_clusters: actual_n,
            split_history,
            inertia,
        })
    }
}

/// Pre-compute an n×n Euclidean distance matrix.
fn precompute_distances(x: ArrayView2<f64>) -> Vec<Vec<f64>> {
    let n = x.shape()[0];
    let mut dist = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = x
                .row(i)
                .iter()
                .zip(x.row(j).iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

/// Find the index of the cluster to split (largest average internal distance).
fn find_cluster_to_split(clusters: &[Vec<usize>], dist: &[Vec<f64>]) -> usize {
    let mut max_diam = -1.0f64;
    let mut best = 0usize;
    for (idx, cluster) in clusters.iter().enumerate() {
        let diam = average_diameter(cluster, dist);
        if diam > max_diam {
            max_diam = diam;
            best = idx;
        }
    }
    best
}

/// Average distance between all pairs in a cluster (diameter estimate).
fn average_diameter(cluster: &[usize], dist: &[Vec<f64>]) -> f64 {
    let n = cluster.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            total += dist[cluster[i]][cluster[j]];
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// DIANA splinter-based split.
///
/// 1. Find the object with the largest average dissimilarity to others (the first splinter).
/// 2. Iteratively move objects whose average distance to the splinter group is
///    less than their average distance to the main party.
fn diana_split(cluster: &[usize], dist: &[Vec<f64>]) -> Result<(Vec<usize>, Vec<usize>)> {
    if cluster.len() < 2 {
        return Err(ClusteringError::InvalidInput(
            "Cannot split a cluster with fewer than 2 elements".into(),
        ));
    }

    // Average dissimilarity of each object to all others in the cluster.
    let avg_diss: Vec<f64> = cluster
        .iter()
        .map(|&i| {
            let sum: f64 = cluster
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| dist[i][j])
                .sum();
            if cluster.len() <= 1 {
                0.0
            } else {
                sum / (cluster.len() - 1) as f64
            }
        })
        .collect();

    // The first splinter is the object with the highest average dissimilarity.
    let splinter_local_idx = avg_diss
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut main_party: HashSet<usize> = cluster.iter().cloned().collect();
    let mut splinter_group: Vec<usize> = vec![cluster[splinter_local_idx]];
    main_party.remove(&cluster[splinter_local_idx]);

    // Iteratively move objects to splinter group.
    loop {
        let main_vec: Vec<usize> = main_party.iter().cloned().collect();
        let sg_len = splinter_group.len() as f64;
        let main_len = main_vec.len() as f64;

        if main_len == 0.0 {
            break;
        }

        let mut to_move: Vec<usize> = Vec::new();
        for &obj in &main_vec {
            // Average distance from obj to splinter group.
            let d_sg = splinter_group.iter().map(|&s| dist[obj][s]).sum::<f64>() / sg_len;
            // Average distance from obj to main party (excluding itself).
            let other_main_len = (main_len - 1.0).max(1.0);
            let d_main = main_vec
                .iter()
                .filter(|&&o| o != obj)
                .map(|&o| dist[obj][o])
                .sum::<f64>()
                / other_main_len;
            if d_sg < d_main {
                to_move.push(obj);
            }
        }

        if to_move.is_empty() {
            break;
        }

        for obj in to_move {
            main_party.remove(&obj);
            splinter_group.push(obj);
        }
    }

    if splinter_group.is_empty() || main_party.is_empty() {
        // Fallback: split in half.
        let half = cluster.len() / 2;
        return Ok((cluster[..half].to_vec(), cluster[half..].to_vec()));
    }

    let main_vec: Vec<usize> = main_party.into_iter().collect();
    Ok((main_vec, splinter_group))
}

// ---------------------------------------------------------------------------
// Consensus Clustering
// ---------------------------------------------------------------------------

/// The consensus matrix: `M[i][j]` = fraction of resamplings in which i and j
/// co-cluster, divided by the fraction of resamplings in which both were selected.
#[derive(Debug, Clone)]
pub struct ConsensusMatrix {
    /// n_samples × n_samples matrix with values in [0, 1].
    pub matrix: Array2<f64>,
    /// Co-occurrence count (how often i and j co-cluster).
    pub cooccurrence: Array2<f64>,
    /// Selection count (how often i and j are both in the subsample).
    pub selection: Array2<f64>,
    /// Number of samples.
    pub n_samples: usize,
    /// Number of resamplings used.
    pub n_resamples: usize,
}

impl ConsensusMatrix {
    /// Retrieve the consensus score between sample `i` and sample `j`.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.matrix[[i, j]]
    }

    /// Extract flat cluster labels by applying Ward linkage to `1 - M`.
    ///
    /// # Errors
    /// Returns an error if `n_clusters` is invalid.
    pub fn extract_clusters(&self, n_clusters: usize) -> Result<Vec<usize>> {
        let n = self.n_samples;
        let mut dist_data = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                dist_data[[i, j]] = 1.0 - self.matrix[[i, j]].clamp(0.0, 1.0);
            }
        }
        let result = Ward::fit(dist_data.view(), n_clusters)?;
        Ok(result.labels.to_vec())
    }
}

/// A trait for base clusterers usable within consensus clustering.
pub trait BaseClusterer: Send + Sync {
    /// Cluster `x` and return an integer label per sample.
    fn fit_predict(&self, x: ArrayView2<f64>) -> Result<Vec<usize>>;
}

/// Consensus clustering via repeated subsampling.
///
/// Runs a base clusterer on random subsets of the data and builds a consensus
/// (co-association) matrix capturing how often pairs of samples cluster together.
pub struct ConsensusClustering {
    /// Fraction of samples drawn each resample (default 0.8).
    pub subsample_fraction: f64,
    /// Fixed RNG seed (optional).
    pub seed: Option<u64>,
}

impl Default for ConsensusClustering {
    fn default() -> Self {
        Self {
            subsample_fraction: 0.8,
            seed: None,
        }
    }
}

impl ConsensusClustering {
    /// Create with custom subsample fraction.
    pub fn new(subsample_fraction: f64, seed: Option<u64>) -> Self {
        Self {
            subsample_fraction: subsample_fraction.clamp(0.1, 1.0),
            seed,
        }
    }

    /// Run consensus clustering.
    ///
    /// # Arguments
    /// * `x` – Data matrix `(n_samples, n_features)`.
    /// * `base_clusterer` – Any `BaseClusterer` implementation (e.g., wrapping KMeans).
    /// * `n_resamples` – Number of subsampling iterations.
    ///
    /// # Errors
    /// Returns an error for empty data or zero resamples.
    pub fn fit(
        &self,
        x: ArrayView2<f64>,
        base_clusterer: &dyn BaseClusterer,
        n_resamples: usize,
    ) -> Result<ConsensusMatrix> {
        let n_samples = x.shape()[0];
        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput("Empty input data".into()));
        }
        if n_resamples == 0 {
            return Err(ClusteringError::InvalidInput(
                "n_resamples must be at least 1".into(),
            ));
        }

        let mut cooccurrence = Array2::<f64>::zeros((n_samples, n_samples));
        let mut selection = Array2::<f64>::zeros((n_samples, n_samples));

        let subsample_size = ((n_samples as f64 * self.subsample_fraction).ceil() as usize).max(2);

        // Simple LCG-based RNG.
        let mut rng_state = self.seed.unwrap_or(42u64);

        for _ in 0..n_resamples {
            // Draw a random subsample (without replacement via partial Fisher-Yates).
            let indices = lcg_sample_without_replacement(&mut rng_state, n_samples, subsample_size);

            // Build sub-matrix.
            let sub_data = build_submatrix(x, &indices);

            // Run base clusterer.
            let sub_labels = base_clusterer.fit_predict(sub_data.view())?;

            // Update co-occurrence and selection counts.
            for (a, &ia) in indices.iter().enumerate() {
                for (b, &ib) in indices.iter().enumerate() {
                    selection[[ia, ib]] += 1.0;
                    if sub_labels[a] == sub_labels[b] {
                        cooccurrence[[ia, ib]] += 1.0;
                    }
                }
            }
        }

        // Build normalized consensus matrix.
        let mut matrix = Array2::<f64>::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                let sel = selection[[i, j]];
                matrix[[i, j]] = if sel > 0.0 {
                    cooccurrence[[i, j]] / sel
                } else {
                    0.0
                };
            }
        }

        Ok(ConsensusMatrix {
            matrix,
            cooccurrence,
            selection,
            n_samples,
            n_resamples,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// LCG-based pseudo-random partial permutation (sample without replacement).
fn lcg_sample_without_replacement(state: &mut u64, n: usize, k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    let k = k.min(n);
    for i in 0..k {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = i + (*state as usize % (n - i));
        indices.swap(i, j);
    }
    indices[..k].to_vec()
}

/// Extract rows from a data matrix by index list.
fn build_submatrix(x: ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    let n_features = x.shape()[1];
    let k = indices.len();
    let mut sub = Array2::<f64>::zeros((k, n_features));
    for (row, &idx) in indices.iter().enumerate() {
        sub.row_mut(row).assign(&x.row(idx));
    }
    sub
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
            (10, 2),
            vec![
                0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 5.0, 5.1, 5.2, 5.0, 5.1, 5.2,
                5.0, 5.1, 5.2, 5.0,
            ],
        )
        .expect("valid shape")
    }

    fn three_cluster_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 5.0, 0.0, 5.1, 0.1, 5.2, 0.0, 0.0, 5.0, 0.1, 5.1,
                0.2, 5.0, 5.0, 5.0, 5.1, 5.1, 5.2, 5.0,
            ],
        )
        .expect("valid shape")
    }

    #[test]
    fn test_ward_basic_two_clusters() {
        let data = two_cluster_data();
        let result = Ward::fit(data.view(), 2).expect("ward fit");
        assert_eq!(result.labels.len(), 10);
        assert_eq!(result.n_clusters, 2);
        let unique: std::collections::HashSet<usize> = result.labels.iter().cloned().collect();
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn test_ward_single_cluster() {
        let data = two_cluster_data();
        let result = Ward::fit(data.view(), 1).expect("ward fit 1 cluster");
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_ward_returns_correct_number_of_clusters() {
        let data = three_cluster_data();
        let result = Ward::fit(data.view(), 3).expect("ward fit 3 clusters");
        let unique: std::collections::HashSet<usize> = result.labels.iter().cloned().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_dendrogram_cut_at_n_clusters() {
        let data = two_cluster_data();
        let result = Ward::fit(data.view(), 1).expect("ward fit");
        let labels2 = result.dendrogram.cut_at_n_clusters(2).expect("cut 2");
        assert_eq!(labels2.len(), 10);
        let unique: std::collections::HashSet<usize> = labels2.iter().cloned().collect();
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn test_dendrogram_cut_at_distance() {
        let data = two_cluster_data();
        let result = Ward::fit(data.view(), 1).expect("ward fit");
        // Cut at a very large distance should merge everything into 1 cluster.
        let labels = result.dendrogram.cut_at_distance(1e9).expect("cut dist");
        let unique: std::collections::HashSet<usize> = labels.iter().cloned().collect();
        assert_eq!(unique.len(), 1);
    }

    #[test]
    fn test_divisive_basic() {
        let data = two_cluster_data();
        let result = Divisive::fit(data.view(), 2).expect("divisive fit");
        assert_eq!(result.labels.len(), 10);
        let unique: std::collections::HashSet<usize> = result.labels.iter().cloned().collect();
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn test_divisive_single_cluster() {
        let data = two_cluster_data();
        let result = Divisive::fit(data.view(), 1).expect("divisive 1 cluster");
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_consensus_clustering_basic() {
        struct SimpleKMeans {
            k: usize,
        }
        impl BaseClusterer for SimpleKMeans {
            fn fit_predict(&self, x: ArrayView2<f64>) -> Result<Vec<usize>> {
                let n = x.shape()[0];
                if n == 0 {
                    return Ok(vec![]);
                }
                let mut vals: Vec<(f64, usize)> = (0..n).map(|i| (x[[i, 0]], i)).collect();
                vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let mut labels = vec![0usize; n];
                let half = n / self.k.max(1);
                for (rank, (_, orig)) in vals.iter().enumerate() {
                    labels[*orig] = (rank / half.max(1)).min(self.k - 1);
                }
                Ok(labels)
            }
        }

        let data = two_cluster_data();
        let clusterer = SimpleKMeans { k: 2 };
        let cc = ConsensusClustering::new(0.8, Some(7));
        let result = cc.fit(data.view(), &clusterer, 10).expect("consensus fit");

        assert_eq!(result.n_samples, 10);
        assert_eq!(result.n_resamples, 10);
        assert_eq!(result.matrix.shape(), [10, 10]);

        // Diagonal should be 1.
        for i in 0..10 {
            assert!(
                (result.matrix[[i, i]] - 1.0).abs() < 1e-9,
                "diagonal must be 1"
            );
        }
    }

    #[test]
    fn test_consensus_extract_clusters() {
        struct TrivialClusterer;
        impl BaseClusterer for TrivialClusterer {
            fn fit_predict(&self, x: ArrayView2<f64>) -> Result<Vec<usize>> {
                let n = x.shape()[0];
                Ok((0..n).map(|i| i % 2).collect())
            }
        }

        let data = two_cluster_data();
        let clusterer = TrivialClusterer;
        let cc = ConsensusClustering::new(1.0, Some(3));
        let result = cc.fit(data.view(), &clusterer, 5).expect("consensus fit");
        let labels = result.extract_clusters(2).expect("extract clusters");
        assert_eq!(labels.len(), 10);
    }

    #[test]
    fn test_ward_invalid_n_clusters() {
        let data = two_cluster_data();
        assert!(Ward::fit(data.view(), 0).is_err());
        assert!(Ward::fit(data.view(), 100).is_err());
    }
}
