//! Spatial data structures for efficient nearest neighbor search
//!
//! This module provides efficient spatial data structures for nearest neighbor
//! searches, which are fundamental to many interpolation methods. These data
//! structures drastically improve performance for local interpolation techniques
//! that rely on finding nearby points.
//!
//! The implementations include:
//!
//! - **KD-Tree**: A space-partitioning data structure that recursively partitions
//!   points along axis-aligned hyperplanes. KD-trees are particularly efficient
//!   for low to medium dimensional data (up to about 20 dimensions).
//!
//! - **Ball Tree**: A metric tree that partitions points into nested hyperspheres.
//!   Ball trees generally perform better than KD-trees in higher dimensions and
//!   for datasets with non-uniform density.
//!
//! These data structures support:
//! - Finding the single nearest neighbor to a query point
//! - Finding the k nearest neighbors to a query point
//! - Finding all points within a specified radius of a query point
//!
//! # When to use which data structure
//!
//! - **KD-Tree**: Best for low to medium dimensional data (2-20 dimensions) with
//!   relatively uniform density. Performs well when the number of dimensions is
//!   much less than the number of points.
//!
//! - **Ball Tree**: Better for higher dimensional data, datasets with varying density,
//!   or when distance computations are expensive. More robust to the "curse of
//!   dimensionality."
//!
//! # Examples
//!
//! ```rust
//! use scirs2_core::ndarray::Array2;
//! use scirs2_interpolate::spatial::kdtree::KdTree;
//!
//! // Create sample 2D points
//! let points = Array2::from_shape_vec((5, 2), vec![
//!     0.0, 0.0,
//!     1.0, 0.0,
//!     0.0, 1.0,
//!     1.0, 1.0,
//!     0.5, 0.5,
//! ]).unwrap();
//!
//! // Build a KD-tree
//! let kdtree = KdTree::new(points).unwrap();
//!
//! // Find the nearest neighbor to point (0.6, 0.6)
//! let query = vec![0.6, 0.6];
//! let (idx, distance) = kdtree.nearest_neighbor(&query).unwrap();
//!
//! // Find the 3 nearest neighbors
//! let neighbors = kdtree.k_nearest_neighbors(&query, 3).unwrap();
//!
//! // Find all points within radius 0.5
//! let points_in_radius = kdtree.points_within_radius(&query, 0.5).unwrap();
//! ```

pub mod balltree;
pub mod enhanced_search;
pub mod kdtree;
pub mod optimized_search;
pub mod simd_enhancements;

pub use balltree::BallTree;
pub use enhanced_search::{
    make_enhanced_searcher, make_high_performance_searcher, EnhancedNearestNeighborSearcher,
    IndexType, SearchConfig,
};
pub use kdtree::KdTree;
#[cfg(feature = "parallel")]
pub use optimized_search::ParallelQueryProcessor;
pub use optimized_search::{CacheFriendlyKNN, OptimizedSpatialSearch, SimdDistanceOps};
pub use simd_enhancements::{AdvancedSimdOps, SimdBenchmark};
