//! `diffgeom::metric` — Riemannian/pseudo-Riemannian metric with symbolic inverse.
//!
//! A [`Metric`] holds the covariant metric tensor `g_ij` as a (0,2) tensor
//! and the contravariant inverse `g^ij` as a (2,0) tensor, computed symbolically
//! using [`crate::cas::matrix_ops`].
//!
//! Supported dimensions: 2, 3, 4.

use crate::cas::{
    canonicalize,
    matrix_ops::{inverse_2x2, inverse_3x3, inverse_4x4, InverseResult},
};
use crate::eml::op::LoweredOp;
use ndarray::{ArrayD, IxDyn};

use super::tensor::Tensor;

/// A Riemannian/pseudo-Riemannian metric.
pub struct Metric {
    /// Covariant metric tensor `g_ij` — a `(0,2)` tensor.
    pub g: Tensor,
    /// Contravariant inverse metric `g^ij` — a `(2,0)` tensor.
    pub g_inv: Tensor,
    /// Variable indices for the coordinate functions `(x^0, x^1, ..., x^{dim-1})`.
    pub coords: Vec<usize>,
}

/// Error type for metric construction.
#[derive(Debug, Clone)]
pub enum MetricError {
    /// Only dimensions 2, 3, 4 are supported.
    UnsupportedDim(usize),
    /// The metric determinant evaluated to a numerical zero — metric is degenerate.
    Singular,
    /// The components array has the wrong shape.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

impl std::fmt::Display for MetricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricError::UnsupportedDim(d) => {
                write!(
                    f,
                    "unsupported metric dimension {d}; only 2, 3, 4 supported"
                )
            }
            MetricError::Singular => write!(f, "metric is singular (det ≈ 0)"),
            MetricError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected:?}, got {got:?}")
            }
        }
    }
}

impl std::error::Error for MetricError {}

impl Metric {
    /// Construct a metric from a `dim × dim` symbolic component array.
    ///
    /// The inverse is computed symbolically and each entry is canonicalized.
    ///
    /// # Errors
    ///
    /// Returns [`MetricError::UnsupportedDim`] if `coords.len()` is not 2, 3, or 4.
    /// Returns [`MetricError::Singular`] if the metric determinant is numerically zero.
    /// Returns [`MetricError::ShapeMismatch`] if `g_components` has wrong shape.
    pub fn new(g_components: ArrayD<LoweredOp>, coords: Vec<usize>) -> Result<Self, MetricError> {
        let dim = coords.len();
        let expected_shape = vec![dim; 2];
        if g_components.shape() != expected_shape.as_slice() {
            return Err(MetricError::ShapeMismatch {
                expected: expected_shape,
                got: g_components.shape().to_vec(),
            });
        }

        // Helper to extract element from ArrayD by [row, col]
        let g_elem = |r: usize, c: usize| -> LoweredOp { g_components[IxDyn(&[r, c])].clone() };

        let inv_components = match dim {
            2 => {
                let mat = [[g_elem(0, 0), g_elem(0, 1)], [g_elem(1, 0), g_elem(1, 1)]];
                match inverse_2x2(&mat) {
                    InverseResult::Singular => return Err(MetricError::Singular),
                    InverseResult::Invertible2(inv) => {
                        let mut arr = ArrayD::from_elem(IxDyn(&[2, 2]), LoweredOp::Const(0.0));
                        for r in 0..2 {
                            for c in 0..2 {
                                arr[IxDyn(&[r, c])] = canonicalize(&inv[r][c]).into_op();
                            }
                        }
                        arr
                    }
                    _ => return Err(MetricError::Singular),
                }
            }
            3 => {
                let mat = [
                    [g_elem(0, 0), g_elem(0, 1), g_elem(0, 2)],
                    [g_elem(1, 0), g_elem(1, 1), g_elem(1, 2)],
                    [g_elem(2, 0), g_elem(2, 1), g_elem(2, 2)],
                ];
                match inverse_3x3(&mat) {
                    InverseResult::Singular => return Err(MetricError::Singular),
                    InverseResult::Invertible3(inv) => {
                        let mut arr = ArrayD::from_elem(IxDyn(&[3, 3]), LoweredOp::Const(0.0));
                        for r in 0..3 {
                            for c in 0..3 {
                                arr[IxDyn(&[r, c])] = canonicalize(&inv[r][c]).into_op();
                            }
                        }
                        arr
                    }
                    _ => return Err(MetricError::Singular),
                }
            }
            4 => {
                let mat = [
                    [g_elem(0, 0), g_elem(0, 1), g_elem(0, 2), g_elem(0, 3)],
                    [g_elem(1, 0), g_elem(1, 1), g_elem(1, 2), g_elem(1, 3)],
                    [g_elem(2, 0), g_elem(2, 1), g_elem(2, 2), g_elem(2, 3)],
                    [g_elem(3, 0), g_elem(3, 1), g_elem(3, 2), g_elem(3, 3)],
                ];
                match inverse_4x4(&mat) {
                    InverseResult::Singular => return Err(MetricError::Singular),
                    InverseResult::Invertible4(inv) => {
                        let mut arr = ArrayD::from_elem(IxDyn(&[4, 4]), LoweredOp::Const(0.0));
                        for r in 0..4 {
                            for c in 0..4 {
                                arr[IxDyn(&[r, c])] = canonicalize(&inv[r][c]).into_op();
                            }
                        }
                        arr
                    }
                    _ => return Err(MetricError::Singular),
                }
            }
            d => return Err(MetricError::UnsupportedDim(d)),
        };

        let g = Tensor::from_components(0, 2, dim, g_components);
        let g_inv = Tensor::from_components(2, 0, dim, inv_components);

        Ok(Metric { g, g_inv, coords })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::eval::{eval_real, EvalCtx};
    use ndarray::ArrayD;

    fn flat_metric_2d() -> Metric {
        let mut g = ArrayD::from_elem(ndarray::IxDyn(&[2, 2]), LoweredOp::Const(0.0));
        g[IxDyn(&[0, 0])] = LoweredOp::Const(1.0);
        g[IxDyn(&[1, 1])] = LoweredOp::Const(1.0);
        Metric::new(g, vec![0, 1]).expect("flat metric")
    }

    #[test]
    fn flat_2d_metric_inverse_is_identity() {
        let metric = flat_metric_2d();
        let ctx = EvalCtx::new(&[]);
        // g_inv[0][0] should be 1, g_inv[0][1] should be 0
        let v00 = eval_real(metric.g_inv.get(&[0, 0]), &ctx).expect("eval");
        let v01 = eval_real(metric.g_inv.get(&[0, 1]), &ctx).expect("eval");
        let v10 = eval_real(metric.g_inv.get(&[1, 0]), &ctx).expect("eval");
        let v11 = eval_real(metric.g_inv.get(&[1, 1]), &ctx).expect("eval");
        assert!((v00 - 1.0).abs() < 1e-10, "g_inv[0,0]={v00}");
        assert!((v01 - 0.0).abs() < 1e-10, "g_inv[0,1]={v01}");
        assert!((v10 - 0.0).abs() < 1e-10, "g_inv[1,0]={v10}");
        assert!((v11 - 1.0).abs() < 1e-10, "g_inv[1,1]={v11}");
    }

    #[test]
    fn minkowski_4d_inverse() {
        // g = diag(-1, 1, 1, 1)
        let mut g = ArrayD::from_elem(IxDyn(&[4, 4]), LoweredOp::Const(0.0));
        g[IxDyn(&[0, 0])] = LoweredOp::Const(-1.0);
        g[IxDyn(&[1, 1])] = LoweredOp::Const(1.0);
        g[IxDyn(&[2, 2])] = LoweredOp::Const(1.0);
        g[IxDyn(&[3, 3])] = LoweredOp::Const(1.0);
        let metric = Metric::new(g, vec![0, 1, 2, 3]).expect("Minkowski metric");
        let ctx = EvalCtx::new(&[]);
        let v00 = eval_real(metric.g_inv.get(&[0, 0]), &ctx).expect("eval");
        assert!(
            (v00 - (-1.0)).abs() < 1e-10,
            "g_inv^00 should be -1, got {v00}"
        );
        let v11 = eval_real(metric.g_inv.get(&[1, 1]), &ctx).expect("eval");
        assert!((v11 - 1.0).abs() < 1e-10, "g_inv^11 should be 1, got {v11}");
    }
}
