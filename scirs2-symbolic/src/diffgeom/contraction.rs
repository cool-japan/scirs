//! `diffgeom::contraction` — index contraction over symbolic tensors.
//!
//! Provides [`contract_indices`] for contracting one upper and one lower index
//! of a [`Tensor`], and [`sum_over_axis`] for building balanced-binary-tree
//! sum expressions used in Christoffel and Ricci computations.

use crate::cas::canonicalize;
use crate::eml::op::LoweredOp;
use ndarray::IxDyn;

use super::tensor::Tensor;

/// Build a balanced binary-tree sum of symbolic expressions.
///
/// This avoids deeply-nested left-skewed trees that would slow canonicalization.
/// Returns `LoweredOp::Const(0.0)` for an empty slice.
pub fn sum_over_axis(terms: &[LoweredOp]) -> LoweredOp {
    match terms.len() {
        0 => LoweredOp::Const(0.0),
        1 => terms[0].clone(),
        _ => {
            let mid = terms.len() / 2;
            let left = sum_over_axis(&terms[..mid]);
            let right = sum_over_axis(&terms[mid..]);
            canonicalize(&LoweredOp::Add(Box::new(left), Box::new(right))).into_op()
        }
    }
}

/// Contract index `up_idx` (an upper/contravariant index) against `down_idx`
/// (a lower/covariant index) of tensor `t`.
///
/// Both `up_idx` and `down_idx` are positions in the full index list of `t`:
/// - `0..rank_up` are upper indices
/// - `rank_up..rank` are lower indices
///
/// The contraction sums over the common dimension, reducing rank by 2.
/// Returns a new tensor with rank `(rank_up - 1, rank_down - 1)`.
///
/// # Panics
///
/// Panics if `up_idx >= rank_up` or `down_idx >= rank_down`.
pub fn contract_indices(t: &Tensor, up_idx: usize, down_idx: usize) -> Tensor {
    assert!(
        up_idx < t.rank_up,
        "up_idx={up_idx} out of bounds, rank_up={}",
        t.rank_up
    );
    assert!(
        down_idx < t.rank_down,
        "down_idx={down_idx} out of bounds, rank_down={}",
        t.rank_down
    );

    let new_rank_up = t.rank_up - 1;
    let new_rank_down = t.rank_down - 1;
    let dim = t.dim;
    let mut result = Tensor::zeros(new_rank_up, new_rank_down, dim);

    // Build iterator over all multi-indices of the output tensor
    let out_rank = new_rank_up + new_rank_down;

    // We iterate all output multi-indices using a flat counter
    let total_out = if out_rank == 0 {
        1
    } else {
        dim.pow(out_rank as u32)
    };

    for flat_out in 0..total_out {
        // Decode flat_out into multi-index of length out_rank
        let mut out_idx = vec![0usize; out_rank];
        let mut tmp = flat_out;
        for pos in (0..out_rank).rev() {
            out_idx[pos] = tmp % dim;
            tmp /= dim;
        }

        // Build input multi-index by inserting the contraction index at the right positions
        let mut terms = Vec::with_capacity(dim);
        for k in 0..dim {
            // Reconstruct the full input index from out_idx by inserting k
            let mut in_idx = Vec::with_capacity(t.rank());
            // Upper indices: insert k at up_idx position
            let mut out_up_iter = out_idx[..new_rank_up].iter();
            for i in 0..t.rank_up {
                if i == up_idx {
                    in_idx.push(k);
                } else {
                    in_idx.push(*out_up_iter.next().unwrap_or(&0));
                }
            }
            // Lower indices: insert k at (rank_up + down_idx) position
            let mut out_down_iter = out_idx[new_rank_up..].iter();
            for i in 0..t.rank_down {
                if i == down_idx {
                    in_idx.push(k);
                } else {
                    in_idx.push(*out_down_iter.next().unwrap_or(&0));
                }
            }
            terms.push(t.components[IxDyn(&in_idx)].clone());
        }

        let contracted = sum_over_axis(&terms);
        result.set(&out_idx, contracted);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffgeom::tensor::Tensor;

    #[test]
    fn sum_over_axis_empty() {
        let result = sum_over_axis(&[]);
        assert_eq!(result, LoweredOp::Const(0.0));
    }

    #[test]
    fn sum_over_axis_single() {
        let result = sum_over_axis(&[LoweredOp::Var(0)]);
        assert_eq!(result, LoweredOp::Var(0));
    }

    #[test]
    fn sum_over_axis_two() {
        let result = sum_over_axis(&[LoweredOp::Const(1.0), LoweredOp::Const(2.0)]);
        // Should evaluate to 3.0 after canonicalization
        use crate::eml::eval::{eval_real, EvalCtx};
        let v = eval_real(&result, &EvalCtx::new(&[])).expect("eval");
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn contract_identity_2x2() {
        // g^ij g_ij = trace = dim (for identity metric)
        let mut t_up = Tensor::zeros(2, 0, 2);
        // delta^ij: diagonal = 1
        t_up.set(&[0, 0], LoweredOp::Const(1.0));
        t_up.set(&[1, 1], LoweredOp::Const(1.0));

        // Contract: first upper index with second upper index — this gives scalar
        // Actually we want to contract a (1,1) tensor's up with its down
        // Let's test (1,1) identity tensor contraction → scalar dim=2
        let mut t_mixed = Tensor::zeros(1, 1, 2);
        t_mixed.set(&[0, 0], LoweredOp::Const(1.0));
        t_mixed.set(&[1, 1], LoweredOp::Const(1.0));

        let scalar = contract_indices(&t_mixed, 0, 0);
        // Trace = 1 + 1 = 2
        assert_eq!(scalar.rank(), 0);
        use crate::eml::eval::{eval_real, EvalCtx};
        let v = eval_real(scalar.get(&[]), &EvalCtx::new(&[])).expect("eval");
        assert!((v - 2.0).abs() < 1e-10, "expected trace=2 but got {v}");
    }
}
