//! GPU-accelerated k-d tree queries for large scattered datasets.
//!
//! This module provides a k-d tree with Rayon-parallel CPU batch queries and
//! an optional GPU dispatch path (feature `gpu_kdtree`) that routes large
//! query batches to a wgpu linear-scan kernel for small-k nearest-neighbor on
//! GPU hardware.
//!
//! # Architecture
//!
//! - **`GpuKdTree`** – Pure-Rust, generic-float k-d tree with variance-based
//!   axis selection and multi-index leaf nodes.  Batch queries run in parallel
//!   via Rayon (gated on the `parallel` feature of scirs2-core).
//!
//! - **`knn_auto_dispatch`** – Examines the number of tree points vs. number of
//!   queries; if both exceed the configured thresholds **and** the `gpu_kdtree`
//!   feature is active, it attempts to call the wgpu linear-scan backend.  On
//!   failure (no GPU, device lost, etc.) it falls back to the CPU path.
//!
//! - **`wgpu_linear_scan`** (feature `gpu_kdtree` only) – Stub WGSL shader
//!   dispatch that falls through to the CPU path, enabling future real
//!   implementation without API changes.
//!
//! # Example
//!
//! ```rust
//! use scirs2_interpolate::gpu_kdtree::{GpuKdTree, KdTreeConfig, knn_auto_dispatch};
//!
//! let pts: Vec<Vec<f64>> = vec![
//!     vec![0.0, 0.0],
//!     vec![1.0, 0.0],
//!     vec![0.5, 0.5],
//! ];
//! let tree = GpuKdTree::new(pts).expect("build");
//! let q = vec![vec![0.4, 0.4]];
//! let cfg = KdTreeConfig::default();
//! let results = knn_auto_dispatch(&tree, &q, 1, &cfg).expect("query");
//! assert_eq!(results[0].indices[0], 2); // closest to (0.5, 0.5)
//! ```

pub mod dispatch;
pub mod tree;

#[cfg(feature = "gpu_kdtree")]
pub mod wgpu_linear_scan;

pub use dispatch::{knn_auto_dispatch, KdTreeConfig};
pub use tree::{GpuKdTree, KdQueryResult};
