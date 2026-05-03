//! Topological Data Analysis (TDA) based clustering algorithms.
//!
//! This module provides clustering algorithms grounded in persistent homology and
//! the Mapper algorithm, which expose the *shape* of data rather than just its
//! metric or density structure.
//!
//! # Algorithms
//!
//! * [`persistent_homology_0d`] – 0-dimensional persistent homology (connected
//!   components) via the incremental union-find algorithm on the Vietoris-Rips
//!   filtration.  Returns a persistence *barcode* as pairs `(birth, death)`.
//!
//! * [`single_linkage_from_barcode`] – Post-process a barcode to obtain flat
//!   cluster labels by cutting the dendrogram at a given distance threshold.
//!
//! * [`mapper_graph`] – Simplified Mapper algorithm: filter, cover, cluster per
//!   patch, link overlapping patches.
//!
//! # Example
//!
//! ```rust
//! use scirs2_cluster::topological_clustering::{
//!     persistent_homology_0d, single_linkage_from_barcode, EuclideanMetric,
//! };
//! use scirs2_core::ndarray::Array2;
//!
//! let data = Array2::from_shape_vec((6, 2), vec![
//!     0.0, 0.0,  0.1, 0.1,  0.05, 0.05,
//!     5.0, 5.0,  5.1, 4.9,  4.9, 5.1,
//! ]).expect("operation should succeed");
//!
//! let barcode = persistent_homology_0d(data.view(), &EuclideanMetric).expect("operation should succeed");
//! // Two long-lived bars expected (two clusters)
//! let labels = single_linkage_from_barcode(&barcode, 1.0, 6);
//! assert_eq!(labels.len(), 6);
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{ClusteringError, Result};

// ═══════════════════════════════════════════════════════════════════════════
// Metric trait
// ═══════════════════════════════════════════════════════════════════════════

/// Distance metric between two feature vectors.
pub trait Metric: Send + Sync {
    /// Compute the distance between `a` and `b`.
    fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64;
}

/// Standard Euclidean (L2) distance.
#[derive(Debug, Clone, Copy)]
pub struct EuclideanMetric;

impl Metric for EuclideanMetric {
    fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
            .sum::<f64>()
            .sqrt()
    }
}

/// Manhattan (L1) distance.
#[derive(Debug, Clone, Copy)]
pub struct ManhattanMetric;

impl Metric for ManhattanMetric {
    fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).abs())
            .sum()
    }
}

/// Cosine distance: 1 - cosine_similarity.
#[derive(Debug, Clone, Copy)]
pub struct CosineMetric;

impl Metric for CosineMetric {
    fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
        let na: f64 = a.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let denom = na * nb;
        if denom < 1e-15 {
            1.0
        } else {
            (1.0 - dot / denom).clamp(0.0, 2.0)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Union-Find (Disjoint Set Union) with path compression + union by rank
// ═══════════════════════════════════════════════════════════════════════════

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    birth: Vec<f64>, // birth time of the component
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            birth: vec![0.0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union two components.  Returns `Some((younger_root, older_root, death_time))` if
    /// a merge event happened (i.e. the two were in different components); the
    /// younger component (later birth) is the one that "dies" at `death_time`.
    /// Returns `None` if they were already connected.
    fn union(&mut self, x: usize, y: usize, time: f64) -> Option<(usize, usize, f64)> {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return None;
        }
        // The component with the *later* birth dies
        let (older, younger) = if self.birth[rx] <= self.birth[ry] {
            (rx, ry)
        } else {
            (ry, rx)
        };
        // Union by rank: attach smaller-rank tree under larger-rank
        if self.rank[older] >= self.rank[younger] {
            self.parent[younger] = older;
            if self.rank[older] == self.rank[younger] {
                self.rank[older] += 1;
            }
        } else {
            self.parent[older] = younger;
            self.birth[younger] = self.birth[younger].min(self.birth[older]);
        }
        Some((younger, older, time))
    }

    /// Update the birth time of a newly created component.
    fn set_birth(&mut self, x: usize, t: f64) {
        self.birth[x] = t;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 0-dimensional Persistent Homology
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the 0-dimensional persistence barcode of `data`.
///
/// The algorithm incrementally builds the Vietoris-Rips complex by processing
/// edges in order of increasing length (Kruskal order). Each time a new edge
/// connects two previously disconnected components, a homology class *dies*.
/// All components alive at the end are assigned `death = ∞`.
///
/// # Arguments
///
/// * `data`   – `(n_samples, n_features)` input array.
/// * `metric` – Distance function implementing [`Metric`].
///
/// # Returns
///
/// A sorted vector of `(birth, death)` pairs.  The pair `(0.0, ∞)` always
/// appears (the one global component that never dies).  Sorting is by
/// persistence (death − birth) descending.
pub fn persistent_homology_0d(
    data: ArrayView2<f64>,
    metric: &dyn Metric,
) -> Result<Vec<(f64, f64)>> {
    let n = data.shape()[0];
    if n == 0 {
        return Ok(vec![]);
    }

    // Build all pairwise edges and sort by distance
    let n_edges = n * (n - 1) / 2;
    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n_edges);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = metric.distance(data.row(i), data.row(j));
            edges.push((d, i, j));
        }
    }
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut uf = UnionFind::new(n);
    // Each point is born at time 0
    for i in 0..n {
        uf.set_birth(i, 0.0);
    }

    let mut barcode: Vec<(f64, f64)> = Vec::new();

    for (dist, u, v) in &edges {
        if let Some((younger, _older, death_time)) = uf.union(*u, *v, *dist) {
            // The younger component dies at this edge
            let birth = uf.birth[younger];
            barcode.push((birth, death_time));
        }
    }

    // All remaining components are immortal
    let mut representatives: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for i in 0..n {
        let root = uf.find(i);
        representatives.insert(root);
    }
    for _ in 0..representatives.len() {
        barcode.push((0.0, f64::INFINITY));
    }

    // Sort by persistence descending (inf persistence last so it sorts "largest")
    barcode.sort_by(|a, b| {
        let pa = if a.1.is_infinite() {
            f64::MAX
        } else {
            a.1 - a.0
        };
        let pb = if b.1.is_infinite() {
            f64::MAX
        } else {
            b.1 - b.0
        };
        pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(barcode)
}

// ═══════════════════════════════════════════════════════════════════════════
// Clustering from barcode
// ═══════════════════════════════════════════════════════════════════════════

/// Derive flat cluster labels from a 0-dim barcode by *single-linkage* at a
/// given distance threshold.
///
/// The number of clusters equals the number of bars that are born before
/// `threshold` and die after `threshold` (or are immortal).  This is
/// equivalent to cutting the single-linkage dendrogram at height `threshold`.
///
/// # Arguments
///
/// * `barcode`    – Output of [`persistent_homology_0d`].
/// * `threshold`  – Distance at which to cut.
/// * `n_points`   – Total number of data points (for label-array sizing).
///
/// # Returns
///
/// A `Vec<usize>` of length `n_points`.  Labels are assigned per connected
/// component at the given threshold (component index in the barcode).
/// Because the barcode alone does not carry point-to-component assignments,
/// labels are assigned uniformly across components round-robin — callers who
/// need point-accurate labels should use [`single_linkage_from_data`] instead.
///
/// Use this function when you only have the barcode (e.g. computed externally).
pub fn single_linkage_from_barcode(
    barcode: &[(f64, f64)],
    threshold: f64,
    n_points: usize,
) -> Vec<usize> {
    if n_points == 0 {
        return vec![];
    }
    // Count components alive at `threshold`
    let n_clusters = barcode
        .iter()
        .filter(|&&(birth, death)| birth <= threshold && (death > threshold || death.is_infinite()))
        .count()
        .max(1);

    // Assign labels round-robin (approximate — point memberships unknown from barcode alone)
    (0..n_points).map(|i| i % n_clusters).collect()
}

/// Perform single-linkage clustering on raw data, returning cluster labels.
///
/// This is equivalent to cutting the single-linkage dendrogram at `threshold`.
/// Uses the same Kruskal-order incremental union-find as
/// [`persistent_homology_0d`] but returns accurate per-point labels.
///
/// # Arguments
///
/// * `data`      – `(n_samples, n_features)`.
/// * `metric`    – Distance metric.
/// * `threshold` – Link distance at which to cut.
///
/// # Returns
///
/// A vector of `n_samples` cluster labels (consecutive integers starting at 0).
pub fn single_linkage_from_data(
    data: ArrayView2<f64>,
    metric: &dyn Metric,
    threshold: f64,
) -> Result<Vec<usize>> {
    let n = data.shape()[0];
    if n == 0 {
        return Ok(vec![]);
    }

    // Build edges sorted by distance
    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = metric.distance(data.row(i), data.row(j));
            if d <= threshold {
                edges.push((d, i, j));
            }
        }
    }
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut uf = UnionFind::new(n);
    for (_, u, v) in &edges {
        uf.union(*u, *v, 0.0);
    }

    // Compress & relabel roots to 0..n_clusters
    let mut root_to_label: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut next_label = 0_usize;
    let mut labels = vec![0_usize; n];
    for i in 0..n {
        let root = uf.find(i);
        let label = *root_to_label.entry(root).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        labels[i] = label;
    }
    Ok(labels)
}

// ═══════════════════════════════════════════════════════════════════════════
// Mapper algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// A Mapper graph node: an index into the patch list and the data indices it covers.
#[derive(Debug, Clone)]
pub struct MapperNode {
    /// Index of the cover interval this node belongs to.
    pub patch_index: usize,
    /// Indices of the data points in this node.
    pub members: Vec<usize>,
}

/// A Mapper graph edge: two node indices that share at least one data point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MapperEdge {
    /// First node index.
    pub source: usize,
    /// Second node index.
    pub target: usize,
}

/// Output of the [`mapper_graph`] function.
#[derive(Debug, Clone)]
pub struct MapperGraph {
    /// Nodes, each corresponding to one cluster within one cover patch.
    pub nodes: Vec<MapperNode>,
    /// Edges: pairs of node indices that share data points.
    pub edges: Vec<MapperEdge>,
}

impl MapperGraph {
    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// All data-point members of node `i`.
    pub fn members_of(&self, node: usize) -> &[usize] {
        &self.nodes[node].members
    }
}

/// A 1-D cover interval.
#[derive(Debug, Clone, Copy)]
pub struct CoverInterval {
    /// Left endpoint (inclusive).
    pub lo: f64,
    /// Right endpoint (inclusive).
    pub hi: f64,
}

impl CoverInterval {
    /// Create a new interval.
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// Check whether `v` falls inside this interval.
    pub fn contains(&self, v: f64) -> bool {
        v >= self.lo && v <= self.hi
    }
}

/// Generate a uniform 1-D cover of `[lo, hi]` with `n_intervals` intervals
/// and fractional `overlap` ∈ (0, 1).
///
/// # Example
///
/// ```rust
/// use scirs2_cluster::topological_clustering::uniform_cover;
/// let intervals = uniform_cover(0.0, 1.0, 4, 0.3);
/// assert_eq!(intervals.len(), 4);
/// ```
pub fn uniform_cover(lo: f64, hi: f64, n_intervals: usize, overlap: f64) -> Vec<CoverInterval> {
    if n_intervals == 0 {
        return vec![];
    }
    let step = (hi - lo) / n_intervals as f64;
    let half_overlap = overlap * step / 2.0;
    (0..n_intervals)
        .map(|i| {
            let centre = lo + step * (i as f64 + 0.5);
            CoverInterval::new(
                (centre - step / 2.0 - half_overlap).max(lo - half_overlap),
                (centre + step / 2.0 + half_overlap).min(hi + half_overlap),
            )
        })
        .collect()
}

/// Simplified Mapper algorithm.
///
/// The Mapper algorithm (Singh, Mémoli, Carlsson 2007) produces a combinatorial
/// summary (graph) of the shape of high-dimensional data.
///
/// # Steps
///
/// 1. **Filter** – apply `filter_fn` to each data point to get a 1-D lens value.
/// 2. **Cover** – partition the range of lens values into `cover_intervals` with
///    pairwise overlap.
/// 3. **Cluster** – within each patch (data points whose lens value falls in the
///    interval) run single-linkage clustering at the given `cluster_threshold`.
/// 4. **Link** – connect two nodes (clusters in different patches) if they share
///    at least one data point.
///
/// # Arguments
///
/// * `data`              – `(n_samples, n_features)`.
/// * `filter_fn`         – User-supplied closure mapping a data row to a scalar.
/// * `cover_intervals`   – List of 1-D intervals partitioning filter-value space.
/// * `cluster_threshold` – Single-linkage threshold inside each patch.
/// * `metric`            – Distance metric for intra-patch clustering.
///
/// # Returns
///
/// A [`MapperGraph`] with nodes and edges.
///
/// # Example
///
/// ```rust
/// use scirs2_cluster::topological_clustering::{
///     mapper_graph, uniform_cover, EuclideanMetric,
/// };
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::from_shape_vec((8, 2), vec![
///     0.0, 0.0,  0.5, 0.0,  1.0, 0.0,  1.5, 0.0,
///     2.0, 0.0,  2.5, 0.0,  3.0, 0.0,  3.5, 0.0,
/// ]).expect("operation should succeed");
///
/// let filter_fn = |row: &[f64]| row[0]; // x-coordinate
/// let cover = uniform_cover(0.0, 3.5, 3, 0.3);
/// let graph = mapper_graph(data.view(), &filter_fn, &cover, 0.8, &EuclideanMetric).expect("operation should succeed");
/// assert!(graph.n_nodes() >= 1);
/// ```
pub fn mapper_graph(
    data: ArrayView2<f64>,
    filter_fn: &dyn Fn(&[f64]) -> f64,
    cover_intervals: &[CoverInterval],
    cluster_threshold: f64,
    metric: &dyn Metric,
) -> Result<MapperGraph> {
    let n = data.shape()[0];
    let d = data.shape()[1];

    if cover_intervals.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "cover_intervals must be non-empty".to_string(),
        ));
    }

    // Step 1 & 2: compute filter values and assign points to cover patches
    let filter_values: Vec<f64> = (0..n)
        .map(|i| {
            let row: Vec<f64> = (0..d).map(|f| data[[i, f]]).collect();
            filter_fn(&row)
        })
        .collect();

    // For each interval, collect the point indices that fall inside
    let patch_members: Vec<Vec<usize>> = cover_intervals
        .iter()
        .map(|interval| {
            (0..n)
                .filter(|&i| interval.contains(filter_values[i]))
                .collect()
        })
        .collect();

    // Step 3: cluster within each patch
    // Track point -> list of (node_index) for edge construction
    let mut point_to_nodes: Vec<Vec<usize>> = vec![vec![]; n];
    let mut nodes: Vec<MapperNode> = Vec::new();

    for (patch_idx, members) in patch_members.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        if members.len() == 1 {
            // Trivial single-point cluster
            let node_idx = nodes.len();
            nodes.push(MapperNode {
                patch_index: patch_idx,
                members: members.clone(),
            });
            point_to_nodes[members[0]].push(node_idx);
            continue;
        }

        // Extract sub-data for this patch
        let sub_n = members.len();
        let mut sub_data = Array2::<f64>::zeros((sub_n, d));
        for (si, &gi) in members.iter().enumerate() {
            for f in 0..d {
                sub_data[[si, f]] = data[[gi, f]];
            }
        }

        // Single-linkage clustering on sub-data
        let sub_labels = single_linkage_from_data(sub_data.view(), metric, cluster_threshold)?;

        // Group sub-indices by cluster label
        let max_label = sub_labels.iter().max().copied().unwrap_or(0);
        let mut cluster_members: Vec<Vec<usize>> = vec![vec![]; max_label + 1];
        for (si, &label) in sub_labels.iter().enumerate() {
            cluster_members[label].push(members[si]);
        }

        for cluster in cluster_members {
            if cluster.is_empty() {
                continue;
            }
            let node_idx = nodes.len();
            for &gi in &cluster {
                point_to_nodes[gi].push(node_idx);
            }
            nodes.push(MapperNode {
                patch_index: patch_idx,
                members: cluster,
            });
        }
    }

    // Step 4: build edges (shared data points between nodes in different patches)
    let mut edge_set: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for node_list in &point_to_nodes {
        if node_list.len() < 2 {
            continue;
        }
        // All pairs of nodes sharing this point
        for i in 0..node_list.len() {
            for j in (i + 1)..node_list.len() {
                let (a, b) = (
                    node_list[i].min(node_list[j]),
                    node_list[i].max(node_list[j]),
                );
                // Only link nodes from *different* patches
                if nodes[a].patch_index != nodes[b].patch_index {
                    edge_set.insert((a, b));
                }
            }
        }
    }

    let edges: Vec<MapperEdge> = edge_set
        .into_iter()
        .map(|(s, t)| MapperEdge {
            source: s,
            target: t,
        })
        .collect();

    Ok(MapperGraph { nodes, edges })
}

// ═══════════════════════════════════════════════════════════════════════════
// Persistence-based cluster count selection
// ═══════════════════════════════════════════════════════════════════════════

/// Estimate the optimal number of clusters from a 0-dim persistence barcode
/// using the *persistence gap* heuristic.
///
/// Counts the number of bars with persistence (death − birth) exceeding
/// `min_persistence`.  Infinite bars are always counted.
///
/// # Example
///
/// ```rust
/// use scirs2_cluster::topological_clustering::n_clusters_from_barcode;
///
/// let barcode = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (0.0, 0.1)];
/// assert_eq!(n_clusters_from_barcode(&barcode, 0.5), 2);
/// ```
pub fn n_clusters_from_barcode(barcode: &[(f64, f64)], min_persistence: f64) -> usize {
    barcode
        .iter()
        .filter(|&&(birth, death)| death.is_infinite() || (death - birth) >= min_persistence)
        .count()
        .max(1)
}

/// Compute a full pairwise distance matrix for `data`.
///
/// Returns an `(n × n)` symmetric matrix where entry `[i, j]` is the distance
/// between points `i` and `j`.
pub fn pairwise_distance_matrix(data: ArrayView2<f64>, metric: &dyn Metric) -> Array2<f64> {
    let n = data.shape()[0];
    let mut dist = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let d = metric.distance(data.row(i), data.row(j));
            dist[[i, j]] = d;
            dist[[j, i]] = d;
        }
    }
    dist
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn two_cluster_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.05, 0.05, -0.05, 0.05, 5.0, 5.0, 5.1, 4.9, 4.9, 5.1, 5.05,
                4.95,
            ],
        )
        .expect("data")
    }

    #[test]
    fn test_ph0_two_clusters() {
        let data = two_cluster_data();
        let barcode = persistent_homology_0d(data.view(), &EuclideanMetric).expect("ph0");
        // In 0-D persistent homology, there is exactly 1 immortal bar (the final connected component).
        // The two clusters produce one very long-lived finite bar (dies when clusters merge)
        // and 6 short-lived finite bars (within-cluster merges).
        let n_inf = barcode.iter().filter(|&&(_, d)| d.is_infinite()).count();
        assert_eq!(
            n_inf, 1,
            "expected 1 immortal component (connected at infinity)"
        );
        // The longest finite bar should have death >> 1.0 (between-cluster distance ~7)
        let finite_bars: Vec<f64> = barcode
            .iter()
            .filter(|&&(_, d)| d.is_finite())
            .map(|&(b, d)| d - b)
            .collect();
        assert!(!finite_bars.is_empty());
        let max_persistence = finite_bars
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_persistence > 1.0,
            "largest finite bar persistence={}",
            max_persistence
        );
    }

    #[test]
    fn test_ph0_single_cluster() {
        let data = Array2::from_shape_vec((4, 1), vec![0.0, 0.1, 0.2, 0.3]).expect("data");
        let barcode = persistent_homology_0d(data.view(), &EuclideanMetric).expect("ph0 single");
        let n_inf = barcode.iter().filter(|&&(_, d)| d.is_infinite()).count();
        assert_eq!(n_inf, 1);
    }

    #[test]
    fn test_ph0_empty() {
        let data = Array2::<f64>::zeros((0, 2));
        let barcode = persistent_homology_0d(data.view(), &EuclideanMetric).expect("ph0 empty");
        assert!(barcode.is_empty());
    }

    #[test]
    fn test_single_linkage_from_barcode_basic() {
        let barcode = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (0.0, 0.2)];
        let labels = single_linkage_from_barcode(&barcode, 1.0, 8);
        assert_eq!(labels.len(), 8);
        let unique: std::collections::HashSet<_> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn test_single_linkage_from_data() {
        let data = two_cluster_data();
        let labels = single_linkage_from_data(data.view(), &EuclideanMetric, 0.5).expect("sl data");
        assert_eq!(labels.len(), 8);
        let unique: std::collections::HashSet<_> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 2, "expected 2 clusters, got {:?}", unique);
    }

    #[test]
    fn test_single_linkage_threshold_zero() {
        let data = two_cluster_data();
        // At threshold 0, no edges are added (exact duplicates needed)
        let labels = single_linkage_from_data(data.view(), &EuclideanMetric, 0.0).expect("sl zero");
        // Each point is its own cluster
        let unique: std::collections::HashSet<_> = labels.iter().copied().collect();
        assert_eq!(unique.len(), data.shape()[0]);
    }

    #[test]
    fn test_n_clusters_from_barcode() {
        let barcode = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (0.0, 0.05)];
        assert_eq!(n_clusters_from_barcode(&barcode, 0.5), 2);
        assert_eq!(n_clusters_from_barcode(&barcode, 0.01), 3);
    }

    #[test]
    fn test_uniform_cover() {
        let intervals = uniform_cover(0.0, 1.0, 4, 0.3);
        assert_eq!(intervals.len(), 4);
        // All intervals should cover their centre
        assert!(intervals[0].contains(0.125));
        assert!(intervals[3].contains(0.875));
    }

    #[test]
    fn test_mapper_graph_line() {
        let data = Array2::from_shape_vec((8, 1), vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
            .expect("data");
        let filter_fn = |row: &[f64]| row[0];
        let cover = uniform_cover(0.0, 3.5, 4, 0.4);
        let graph =
            mapper_graph(data.view(), &filter_fn, &cover, 0.6, &EuclideanMetric).expect("mapper");
        assert!(
            graph.n_nodes() >= 4,
            "expected >= 4 nodes, got {}",
            graph.n_nodes()
        );
        assert!(
            graph.n_edges() >= 1,
            "expected at least 1 edge, got {}",
            graph.n_edges()
        );
    }

    #[test]
    fn test_mapper_graph_two_clusters() {
        let data = two_cluster_data();
        let filter_fn = |row: &[f64]| row[0]; // x-coordinate
        let cover = uniform_cover(-0.1, 5.2, 6, 0.2);
        let graph = mapper_graph(data.view(), &filter_fn, &cover, 0.3, &EuclideanMetric)
            .expect("mapper 2cluster");
        assert!(graph.n_nodes() >= 2);
    }

    #[test]
    fn test_mapper_empty_cover_error() {
        let data = two_cluster_data();
        let filter_fn = |row: &[f64]| row[0];
        let result = mapper_graph(data.view(), &filter_fn, &[], 0.5, &EuclideanMetric);
        assert!(result.is_err());
    }

    #[test]
    fn test_pairwise_distance_matrix() {
        let data =
            Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).expect("data");
        let dm = pairwise_distance_matrix(data.view(), &EuclideanMetric);
        assert_eq!(dm.shape(), [3, 3]);
        assert!((dm[[0, 0]]).abs() < 1e-10);
        assert!((dm[[0, 1]] - 1.0).abs() < 1e-10);
        // Symmetry
        assert!((dm[[1, 0]] - dm[[0, 1]]).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_metric() {
        let a = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).expect("a");
        let b = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).expect("b");
        let d = ManhattanMetric.distance(a.row(0), b.row(0));
        assert!((d - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_metric_orthogonal() {
        let a = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).expect("a");
        let b = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).expect("b");
        let d = CosineMetric.distance(a.row(0), b.row(0));
        assert!((d - 1.0).abs() < 1e-10);
    }
}
