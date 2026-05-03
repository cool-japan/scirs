//! Auto-dispatch layer for GPU vs CPU k-d tree queries.
//!
//! [`knn_auto_dispatch`] inspects the size of the tree and the query batch
//! and, when both exceed the configured thresholds **and** the `gpu_kdtree`
//! cargo feature is active, routes the request to the wgpu linear-scan
//! backend.  On failure, or when thresholds are not met, it falls through to
//! the parallel CPU path.

use super::tree::{GpuKdTree, KdQueryResult};
use crate::error::InterpolateResult;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for [`knn_auto_dispatch`].
///
/// Both thresholds must be exceeded for the GPU path to be attempted.
#[derive(Debug, Clone)]
pub struct KdTreeConfig {
    /// Minimum number of points in the tree before GPU dispatch is tried.
    /// Default: 100 000.
    pub gpu_threshold_points: usize,
    /// Minimum number of query points before GPU dispatch is tried.
    /// Default: 1 000.
    pub gpu_threshold_queries: usize,
}

impl Default for KdTreeConfig {
    fn default() -> Self {
        Self {
            gpu_threshold_points: 100_000,
            gpu_threshold_queries: 1_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Auto-dispatch
// ---------------------------------------------------------------------------

/// Compute batch k-NN, automatically choosing GPU or CPU path.
///
/// GPU dispatch requires:
/// 1. The `gpu_kdtree` cargo feature to be active.
/// 2. `tree.n_points() >= config.gpu_threshold_points`.
/// 3. `queries.len() >= config.gpu_threshold_queries`.
///
/// Any failure in the GPU path causes a silent fallback to the CPU path.
///
/// # Arguments
///
/// * `tree`    – A pre-built [`GpuKdTree`].
/// * `queries` – Batch of query points; each must match `tree.dim()`.
/// * `k`       – Number of nearest neighbors to find per query.
/// * `config`  – Dispatch thresholds.
///
/// # Errors
///
/// Returns an error only when the CPU path fails (e.g. dimension mismatch).
pub fn knn_auto_dispatch(
    tree: &GpuKdTree,
    queries: &[Vec<f64>],
    k: usize,
    config: &KdTreeConfig,
) -> InterpolateResult<Vec<KdQueryResult>> {
    // Attempt GPU path when conditions are met.
    #[cfg(feature = "gpu_kdtree")]
    if tree.n_points() >= config.gpu_threshold_points
        && queries.len() >= config.gpu_threshold_queries
    {
        match super::wgpu_linear_scan::knn_wgpu(tree, queries, k) {
            Ok(results) => return Ok(results),
            // Fall through to CPU on any GPU error.
            Err(_) => {}
        }
    }

    // Suppress "unused variable" warning for `config` in non-gpu builds.
    let _ = config;

    // CPU path: always available.
    tree.knn_batch(queries, k)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_kdtree::GpuKdTree;

    #[test]
    fn test_knn_auto_dispatch_cpu_below_threshold() {
        // Small dataset — should always use the CPU path.
        let pts: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.5]];
        let tree = GpuKdTree::new(pts).expect("build");
        let queries = vec![vec![0.4_f64, 0.4]];
        let cfg = KdTreeConfig::default(); // thresholds far above 3 points

        let results = knn_auto_dispatch(&tree, &queries, 1, &cfg).expect("dispatch");
        assert_eq!(results.len(), 1);
        // Closest to (0.4, 0.4) is (0.5, 0.5) at dist² = 0.02
        assert_eq!(results[0].indices[0], 2);
    }

    #[test]
    fn test_knn_auto_dispatch_returns_correct_count() {
        let pts: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, 0.0]).collect();
        let tree = GpuKdTree::new(pts).expect("build");
        let queries: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 + 0.1, 0.0]).collect();
        let cfg = KdTreeConfig {
            gpu_threshold_points: usize::MAX,
            gpu_threshold_queries: usize::MAX,
        };

        let results = knn_auto_dispatch(&tree, &queries, 3, &cfg).expect("dispatch");
        assert_eq!(results.len(), 5);
        for r in &results {
            assert_eq!(r.indices.len(), 3);
            assert_eq!(r.distances_sq.len(), 3);
        }
    }

    #[test]
    fn test_knn_auto_dispatch_empty_queries() {
        let pts = vec![vec![1.0_f64, 2.0]];
        let tree = GpuKdTree::new(pts).expect("build");
        let results = knn_auto_dispatch(&tree, &[], 1, &KdTreeConfig::default())
            .expect("dispatch empty queries");
        assert!(results.is_empty());
    }

    #[test]
    fn test_kdtree_config_default_thresholds() {
        let cfg = KdTreeConfig::default();
        assert_eq!(cfg.gpu_threshold_points, 100_000);
        assert_eq!(cfg.gpu_threshold_queries, 1_000);
    }
}
