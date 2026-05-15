//! `diffgeom::tensor` — symbolic tensor with mixed valence.
//!
//! A [`Tensor`] is a symbolic container of [`LoweredOp`] components indexed by
//! mixed upper/lower indices. It stores components in an `ndarray::ArrayD`.
//!
//! Indexing convention: upper indices first, then lower.
//! Components shape: `[dim; rank_up + rank_down]`.

use crate::eml::op::LoweredOp;
use ndarray::{ArrayD, IxDyn};

/// Whether a tensor index is contravariant (up) or covariant (down).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexKind {
    /// Contravariant (upper) index.
    Up,
    /// Covariant (lower) index.
    Down,
}

/// A labeled index with name and variance kind.
#[derive(Debug, Clone)]
pub struct IndexLabel {
    /// Display name of this index (e.g. `"^0"`, `"_1"`, or custom).
    pub name: String,
    /// Whether this index is contravariant (`Up`) or covariant (`Down`).
    pub kind: IndexKind,
}

/// A symbolic tensor with mixed valence.
///
/// The components array has shape `[dim; rank_up + rank_down]`.
/// Upper-index positions come first in the multi-index, then lower-index positions.
///
/// All components default to `LoweredOp::Const(0.0)` unless explicitly set.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Number of contravariant (upper) indices.
    pub rank_up: usize,
    /// Number of covariant (lower) indices.
    pub rank_down: usize,
    /// Dimension of each index (all indices range over `0..dim`).
    pub dim: usize,
    /// Component storage. Shape: `[dim; rank_up + rank_down]`.
    pub components: ArrayD<LoweredOp>,
    /// Labels for each index position.
    pub indices: Vec<IndexLabel>,
}

impl Tensor {
    /// Create a zero tensor with all components set to `LoweredOp::Const(0.0)`.
    pub fn zeros(rank_up: usize, rank_down: usize, dim: usize) -> Self {
        let total_rank = rank_up + rank_down;
        let shape = vec![dim; total_rank];
        let components = ArrayD::from_elem(IxDyn(&shape), LoweredOp::Const(0.0));
        let indices = (0..rank_up)
            .map(|i| IndexLabel {
                name: format!("^{i}"),
                kind: IndexKind::Up,
            })
            .chain((0..rank_down).map(|i| IndexLabel {
                name: format!("_{i}"),
                kind: IndexKind::Down,
            }))
            .collect();
        Tensor {
            rank_up,
            rank_down,
            dim,
            components,
            indices,
        }
    }

    /// Create a tensor from a pre-built components array.
    ///
    /// Validates that the shape matches `[dim; rank_up + rank_down]`.
    pub fn from_components(
        rank_up: usize,
        rank_down: usize,
        dim: usize,
        components: ArrayD<LoweredOp>,
    ) -> Self {
        let total_rank = rank_up + rank_down;
        let indices = (0..rank_up)
            .map(|i| IndexLabel {
                name: format!("^{i}"),
                kind: IndexKind::Up,
            })
            .chain((0..rank_down).map(|i| IndexLabel {
                name: format!("_{i}"),
                kind: IndexKind::Down,
            }))
            .collect();
        debug_assert_eq!(
            components.ndim(),
            total_rank,
            "components ndim {} != total_rank {}",
            components.ndim(),
            total_rank,
        );
        debug_assert!(
            components.shape().iter().all(|&s| s == dim),
            "all component dimensions must equal dim={dim}"
        );
        Tensor {
            rank_up,
            rank_down,
            dim,
            components,
            indices,
        }
    }

    /// Total rank (number of indices).
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank_up + self.rank_down
    }

    /// Get a reference to a component by multi-index slice.
    ///
    /// Panics in debug if the index has the wrong length.
    #[inline]
    pub fn get(&self, idx: &[usize]) -> &LoweredOp {
        &self.components[IxDyn(idx)]
    }

    /// Set a component by multi-index slice.
    ///
    /// Panics in debug if the index has the wrong length.
    #[inline]
    pub fn set(&mut self, idx: &[usize], val: LoweredOp) {
        self.components[IxDyn(idx)] = val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_shape_correct() {
        let t = Tensor::zeros(1, 2, 3);
        assert_eq!(t.rank_up, 1);
        assert_eq!(t.rank_down, 2);
        assert_eq!(t.dim, 3);
        assert_eq!(t.rank(), 3);
        assert_eq!(t.components.shape(), &[3, 3, 3]);
    }

    #[test]
    fn get_set_roundtrip() {
        let mut t = Tensor::zeros(1, 1, 2);
        t.set(&[0, 1], LoweredOp::Const(42.0));
        assert_eq!(*t.get(&[0, 1]), LoweredOp::Const(42.0));
        assert_eq!(*t.get(&[1, 0]), LoweredOp::Const(0.0));
    }

    #[test]
    fn index_labels() {
        let t = Tensor::zeros(2, 1, 4);
        assert_eq!(t.indices[0].kind, IndexKind::Up);
        assert_eq!(t.indices[1].kind, IndexKind::Up);
        assert_eq!(t.indices[2].kind, IndexKind::Down);
        assert_eq!(t.indices[0].name, "^0");
        assert_eq!(t.indices[2].name, "_0");
    }
}
