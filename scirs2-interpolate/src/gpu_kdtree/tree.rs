//! CPU k-d tree for arbitrary dimension, with Rayon parallel batch queries.
//!
//! The tree uses variance-based axis selection at each split and stores
//! multiple point indices in leaf nodes (standard k-d tree leaf-bucket design).

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::parallel_ops::*;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// Internal node type
// ---------------------------------------------------------------------------

/// A node in the k-d tree.
#[derive(Debug)]
enum KdNode {
    /// Leaf node holding one or more point indices.
    Leaf { point_indices: Vec<usize> },
    /// Interior split node.
    Split {
        axis: usize,
        split_val: f64,
        left: Box<KdNode>,
        right: Box<KdNode>,
    },
}

// ---------------------------------------------------------------------------
// Priority queue entry (max-heap by distance, so we can evict the farthest)
// ---------------------------------------------------------------------------

/// Priority-queue entry: `(dist_sq, point_idx)`.
///
/// The `BinaryHeap` in Rust is a max-heap; we want the *farthest* element at
/// the top so we can evict it when a closer point is found.
#[derive(PartialEq)]
struct HeapEntry(f64, usize);

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // We want a max-heap by distance (largest dist at the top),
        // so that we can pop the farthest element when we find something closer.
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(other.1.cmp(&self.1))
    }
}

// ---------------------------------------------------------------------------
// Public API types (re-exported via mod.rs)
// ---------------------------------------------------------------------------

/// Result of a single k-NN query.
#[derive(Debug, Clone)]
pub struct KdQueryResult {
    /// Indices into the original point array (sorted nearest-first).
    pub indices: Vec<usize>,
    /// Squared Euclidean distances (same order as `indices`).
    pub distances_sq: Vec<f64>,
}

// ---------------------------------------------------------------------------
// GpuKdTree
// ---------------------------------------------------------------------------

/// K-d tree for N-dimensional nearest-neighbor queries.
///
/// Optimised for pure-f64 coordinates.  Batch queries are executed in
/// parallel using Rayon when the `parallel` feature of *scirs2-core* is
/// enabled.
///
/// # Notes
///
/// This type is intentionally named `GpuKdTree` to distinguish it from the
/// pre-existing `spatial::kdtree::KdTree<F>` generic tree.  Both types are
/// available in `scirs2-interpolate`; `GpuKdTree` adds the GPU-dispatch
/// overlay for large scattered-data queries.
///
/// # Example
///
/// ```rust
/// use scirs2_interpolate::gpu_kdtree::GpuKdTree;
///
/// let pts = vec![
///     vec![0.0f64, 0.0],
///     vec![1.0, 0.0],
///     vec![0.0, 1.0],
///     vec![1.0, 1.0],
///     vec![0.5, 0.5],
/// ];
/// let tree = GpuKdTree::new(pts).expect("build");
/// let res = tree.knn(&[0.6, 0.6], 1).expect("query");
/// assert_eq!(res.indices[0], 4); // (0.5, 0.5)
/// ```
pub struct GpuKdTree {
    /// Flat storage of all point coordinates.
    points: Vec<Vec<f64>>,
    /// Dimensionality (number of coordinates per point).
    dim: usize,
    /// Root node, `None` only for an empty tree.
    root: Option<KdNode>,
    /// Maximum number of points stored per leaf node.
    leaf_size: usize,
}

impl GpuKdTree {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Build a k-d tree from `points`.
    ///
    /// Each point must have the same number of coordinates.  Returns an error
    /// if any point has a different dimension from the first.
    pub fn new(points: Vec<Vec<f64>>) -> InterpolateResult<Self> {
        Self::with_leaf_size(points, 16)
    }

    /// Build a k-d tree with a custom leaf-bucket size.
    ///
    /// Larger `leaf_size` reduces tree depth at the cost of more linear scans
    /// inside leaves.  The default of 16 is a good general-purpose value.
    pub fn with_leaf_size(points: Vec<Vec<f64>>, leaf_size: usize) -> InterpolateResult<Self> {
        let leaf_size = leaf_size.max(1);

        if points.is_empty() {
            return Ok(Self {
                points: Vec::new(),
                dim: 0,
                root: None,
                leaf_size,
            });
        }

        let dim = points[0].len();
        for (i, p) in points.iter().enumerate() {
            if p.len() != dim {
                return Err(InterpolateError::InvalidInput {
                    message: format!("Point {i} has dimension {} but expected {dim}", p.len()),
                });
            }
        }

        let indices: Vec<usize> = (0..points.len()).collect();
        let root = Some(build_node(&points, indices, dim, leaf_size));

        Ok(Self {
            points,
            dim,
            root,
            leaf_size,
        })
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Find the `k` nearest neighbors to `query`.
    ///
    /// Returns a [`KdQueryResult`] with indices and squared distances sorted
    /// nearest-first.  If `k > n_points()`, all points are returned.
    pub fn knn(&self, query: &[f64], k: usize) -> InterpolateResult<KdQueryResult> {
        if query.len() != self.dim {
            return Err(InterpolateError::InvalidInput {
                message: format!(
                    "Query has dimension {} but tree has dimension {}",
                    query.len(),
                    self.dim
                ),
            });
        }

        if self.points.is_empty() || self.root.is_none() {
            return Ok(KdQueryResult {
                indices: Vec::new(),
                distances_sq: Vec::new(),
            });
        }

        let k_effective = k.min(self.points.len());
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k_effective + 1);

        if let Some(root) = &self.root {
            search_knn(root, &self.points, query, k_effective, &mut heap);
        }

        // Convert heap (max-heap by distance) to sorted nearest-first vec.
        let mut results: Vec<(f64, usize)> = heap.into_iter().map(|e| (e.0, e.1)).collect();
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(KdQueryResult {
            indices: results.iter().map(|(_, i)| *i).collect(),
            distances_sq: results.iter().map(|(d, _)| *d).collect(),
        })
    }

    /// Parallel batch k-NN for many query points.
    ///
    /// Each query is processed independently; results are in the same order as
    /// `queries`.  Uses Rayon parallel iteration when the `parallel` feature
    /// of *scirs2-core* is active; otherwise falls back to sequential.
    pub fn knn_batch(
        &self,
        queries: &[Vec<f64>],
        k: usize,
    ) -> InterpolateResult<Vec<KdQueryResult>> {
        queries.into_par_iter().map(|q| self.knn(q, k)).collect()
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Number of points in the tree.
    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    /// Dimensionality of the points in the tree.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// Tree construction helpers
// ---------------------------------------------------------------------------

fn build_node(points: &[Vec<f64>], indices: Vec<usize>, dim: usize, leaf_size: usize) -> KdNode {
    if indices.len() <= leaf_size {
        return KdNode::Leaf {
            point_indices: indices,
        };
    }

    // Choose the split axis as the dimension with the largest variance.
    let axis = (0..dim)
        .max_by(|&a, &b| {
            let va = variance_along(points, &indices, a);
            let vb = variance_along(points, &indices, b);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);

    // Sort by the chosen axis and split at the median.
    let mut sorted = indices;
    sorted.sort_unstable_by(|&i, &j| {
        points[i][axis]
            .partial_cmp(&points[j][axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mid = sorted.len() / 2;
    let split_val = points[sorted[mid]][axis];

    let right_indices = sorted.split_off(mid);
    let left_indices = sorted;

    KdNode::Split {
        axis,
        split_val,
        left: Box::new(build_node(points, left_indices, dim, leaf_size)),
        right: Box::new(build_node(points, right_indices, dim, leaf_size)),
    }
}

fn variance_along(points: &[Vec<f64>], indices: &[usize], axis: usize) -> f64 {
    let n = indices.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = indices.iter().map(|&i| points[i][axis]).sum::<f64>() / n;
    indices
        .iter()
        .map(|&i| (points[i][axis] - mean).powi(2))
        .sum::<f64>()
        / n
}

// ---------------------------------------------------------------------------
// k-NN search
// ---------------------------------------------------------------------------

fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn search_knn(
    node: &KdNode,
    points: &[Vec<f64>],
    query: &[f64],
    k: usize,
    heap: &mut BinaryHeap<HeapEntry>,
) {
    match node {
        KdNode::Leaf { point_indices } => {
            for &idx in point_indices {
                let d = dist_sq(query, &points[idx]);
                maybe_push(heap, d, idx, k);
            }
        }
        KdNode::Split {
            axis,
            split_val,
            left,
            right,
        } => {
            let diff = query[*axis] - split_val;
            let (near, far) = if diff <= 0.0 {
                (left.as_ref(), right.as_ref())
            } else {
                (right.as_ref(), left.as_ref())
            };

            search_knn(near, points, query, k, heap);

            // Check if the far half-space could contain a closer point.
            let worst_sq = heap.peek().map(|e| e.0).unwrap_or(f64::INFINITY);
            if diff * diff < worst_sq || heap.len() < k {
                search_knn(far, points, query, k, heap);
            }
        }
    }
}

/// Insert `(dist_sq, idx)` into the max-heap if it improves the current k-set.
fn maybe_push(heap: &mut BinaryHeap<HeapEntry>, d: f64, idx: usize, k: usize) {
    if heap.len() < k {
        heap.push(HeapEntry(d, idx));
    } else if let Some(top) = heap.peek() {
        if d < top.0 {
            heap.pop();
            heap.push(HeapEntry(d, idx));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Brute-force k-NN reference implementation for comparison.
    fn brute_force_knn(points: &[Vec<f64>], query: &[f64], k: usize) -> Vec<usize> {
        let mut dists: Vec<(f64, usize)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let d = p.iter().zip(query).map(|(a, b)| (a - b).powi(2)).sum();
                (d, i)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        dists[..k.min(dists.len())]
            .iter()
            .map(|(_, i)| *i)
            .collect()
    }

    #[test]
    fn test_kdtree_empty_returns_empty() {
        let tree = GpuKdTree::new(vec![]).expect("build empty");
        let res = tree.knn(&[], 1).expect("query empty");
        assert!(res.indices.is_empty());
        assert!(res.distances_sq.is_empty());
    }

    #[test]
    fn test_kdtree_single_point() {
        let pts = vec![vec![3.0_f64, 4.0]];
        let tree = GpuKdTree::new(pts).expect("build single");
        let res = tree.knn(&[0.0, 0.0], 1).expect("query");
        assert_eq!(res.indices.len(), 1);
        assert_eq!(res.indices[0], 0);
        // dist² = 9 + 16 = 25
        let expected_d = 25.0_f64;
        assert!((res.distances_sq[0] - expected_d).abs() < 1e-12);
    }

    #[test]
    fn test_kdtree_1d_finds_nearest() {
        let pts: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let tree = GpuKdTree::new(pts.clone()).expect("build 1d");
        let res = tree.knn(&[2.5], 1).expect("query 1d");
        // Nearest to 2.5 should be 2 (dist 0.25) or 3 (dist 0.25) — accept either
        assert!(
            res.indices[0] == 2 || res.indices[0] == 3,
            "expected index 2 or 3, got {}",
            res.indices[0]
        );
        assert!((res.distances_sq[0] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_kdtree_2d_knn_k3() {
        // 3×3 grid: indices 0..8
        // (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2)
        let pts: Vec<Vec<f64>> = (0..3)
            .flat_map(|r: i32| (0..3).map(move |c: i32| vec![c as f64, r as f64]))
            .collect();
        let tree = GpuKdTree::new(pts.clone()).expect("build 2d grid");

        // Query at center (1,1) — index 4
        let res = tree.knn(&[1.0, 1.0], 3).expect("knn 3 at center");
        assert_eq!(res.indices.len(), 3);
        // Nearest should be itself (dist 0) — index 4
        assert_eq!(res.indices[0], 4);
        // Next two must all have dist² = 1.0 (orthogonal neighbors)
        assert!(
            (res.distances_sq[1] - 1.0).abs() < 1e-12 || (res.distances_sq[2] - 1.0).abs() < 1e-12,
            "distances: {:?}",
            res.distances_sq
        );
    }

    #[test]
    fn test_kdtree_knn_batch_matches_brute_force() {
        use fastrand::Rng;
        let mut rng = Rng::with_seed(42);

        let n = 50_usize;
        let dim = 3_usize;
        let pts: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.f64()).collect())
            .collect();

        let tree = GpuKdTree::new(pts.clone()).expect("build 3d");

        let queries: Vec<Vec<f64>> = (0..20)
            .map(|_| (0..dim).map(|_| rng.f64()).collect())
            .collect();

        let k = 5;
        let batch = tree.knn_batch(&queries, k).expect("batch knn");
        assert_eq!(batch.len(), queries.len());

        for (q_idx, (res, q)) in batch.iter().zip(queries.iter()).enumerate() {
            let expected = brute_force_knn(&pts, q, k);
            // Compare sorted index sets (both should agree on the same k points)
            let mut got = res.indices.clone();
            let mut exp = expected.clone();
            got.sort_unstable();
            exp.sort_unstable();
            assert_eq!(got, exp, "query {q_idx}: tree={got:?} brute={exp:?}");
        }
    }

    #[test]
    fn test_kdtree_dimension_mismatch_errors() {
        let pts = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]];
        let tree = GpuKdTree::new(pts).expect("build");
        let err = tree.knn(&[0.0], 1);
        assert!(err.is_err(), "should error on dimension mismatch");
    }

    #[test]
    fn test_kdtree_dimension_mismatch_on_build() {
        let pts = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0, 5.0]];
        let err = GpuKdTree::new(pts);
        assert!(err.is_err(), "should error when points have different dims");
    }

    #[test]
    fn test_kdtree_high_dim_correct() {
        use fastrand::Rng;
        let mut rng = Rng::with_seed(99);

        let n = 100_usize;
        let dim = 10_usize;
        let pts: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.f64()).collect())
            .collect();

        let tree = GpuKdTree::new(pts.clone()).expect("build 10d");
        let q: Vec<f64> = (0..dim).map(|_| rng.f64()).collect();
        let k = 7;

        let tree_res = tree.knn(&q, k).expect("knn 10d");
        let brute_res = brute_force_knn(&pts, &q, k);

        let mut got = tree_res.indices.clone();
        let mut exp = brute_res;
        got.sort_unstable();
        exp.sort_unstable();
        assert_eq!(got, exp, "10-D: tree={got:?} brute={exp:?}");
    }

    #[test]
    fn test_kdtree_k_larger_than_n() {
        let pts = vec![vec![0.0_f64], vec![1.0], vec![2.0]];
        let tree = GpuKdTree::new(pts).expect("build");
        let res = tree.knn(&[1.5], 100).expect("k > n");
        // Should return all 3 points
        assert_eq!(res.indices.len(), 3);
    }

    #[test]
    fn test_kdtree_distances_are_sorted_ascending() {
        use fastrand::Rng;
        let mut rng = Rng::with_seed(7);

        let pts: Vec<Vec<f64>> = (0..30).map(|_| vec![rng.f64(), rng.f64()]).collect();
        let tree = GpuKdTree::new(pts).expect("build");
        let q = vec![0.5, 0.5];
        let res = tree.knn(&q, 10).expect("knn 10");
        for w in res.distances_sq.windows(2) {
            assert!(w[0] <= w[1], "distances not sorted: {:?}", res.distances_sq);
        }
    }
}
