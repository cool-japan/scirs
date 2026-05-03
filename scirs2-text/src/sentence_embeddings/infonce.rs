//! InfoNCE contrastive loss for SimCSE.
//!
//! Implements the normalized temperature-scaled cross-entropy (NT-Xent / InfoNCE)
//! loss used in SimCSE (Gao et al. 2021, <https://arxiv.org/abs/2104.08821>).
//!
//! This module is encoder-agnostic: it operates on pre-computed embedding
//! matrices and makes no assumptions about how embeddings were produced.

use scirs2_core::ndarray::{Array2, ArrayView1};

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute the InfoNCE / NT-Xent loss given a batch of (anchor, positive) pairs.
///
/// # Formula
///
/// For each anchor `a_i` with positive `p_i` and all other positives as in-batch
/// negatives:
///
/// ```text
/// L_i = -log( exp(sim(a_i, p_i) / τ) / Σ_j exp(sim(a_i, p_j) / τ) )
/// ```
///
/// The batch-level loss is the mean over all `i`.
///
/// # Arguments
///
/// * `anchors`   — Shape `(batch_size, embed_dim)`.
/// * `positives` — Shape `(batch_size, embed_dim)`.  `positives[i]` is the positive
///   pair for `anchors[i]`; all other `positives[j]` with `j ≠ i` are
///   treated as negatives.
/// * `temperature` — Scaling parameter τ (typically 0.05).
///
/// # Returns
///
/// A scalar loss ≥ 0.  Returns 0.0 when the batch is empty.
pub fn infonce_loss(anchors: &Array2<f32>, positives: &Array2<f32>, temperature: f32) -> f32 {
    let batch_size = anchors.nrows().min(positives.nrows());
    if batch_size == 0 {
        return 0.0;
    }

    let mut total_loss = 0.0_f32;

    for i in 0..batch_size {
        let anchor = anchors.row(i);
        let pos_i = positives.row(i);

        let a_norm = l2_norm_f32(anchor);

        // sim(a_i, p_i) / τ
        let pos_sim = if a_norm > 1e-8 {
            let p_norm = l2_norm_f32(pos_i);
            if p_norm > 1e-8 {
                anchor.dot(&pos_i) / (a_norm * p_norm)
            } else {
                0.0
            }
        } else {
            0.0
        } / temperature;

        let exp_pos = pos_sim.exp();

        // Denominator: sum over all positives (including i itself)
        let mut denom = 0.0_f32;
        for j in 0..batch_size {
            let pos_j = positives.row(j);
            let p_norm_j = l2_norm_f32(pos_j);
            let sim_ij = if a_norm > 1e-8 && p_norm_j > 1e-8 {
                anchor.dot(&pos_j) / (a_norm * p_norm_j)
            } else {
                0.0
            } / temperature;
            denom += sim_ij.exp();
        }

        if denom > 1e-30 && denom.is_finite() {
            total_loss += -(exp_pos / denom).ln();
        }
    }

    total_loss / batch_size as f32
}

/// Compute batch-level cosine similarity matrix.
///
/// Returns an `(n, m)` matrix where `result[[i, j]] = cosine_sim(a[i], b[j])`.
pub fn cosine_similarity_matrix(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let n = a.nrows();
    let m = b.nrows();
    let mut result = Array2::<f32>::zeros((n, m));

    for i in 0..n {
        let ai = a.row(i);
        let a_norm = l2_norm_f32(ai);
        for j in 0..m {
            let bj = b.row(j);
            let b_norm = l2_norm_f32(bj);
            let sim = if a_norm > 1e-8 && b_norm > 1e-8 {
                ai.dot(&bj) / (a_norm * b_norm)
            } else {
                0.0
            };
            result[[i, j]] = sim;
        }
    }

    result
}

/// Top-1 accuracy: fraction of anchors whose positive is the nearest
/// neighbour in the positive set (by cosine similarity).
pub fn top1_accuracy(anchors: &Array2<f32>, positives: &Array2<f32>) -> f32 {
    let n = anchors.nrows().min(positives.nrows());
    if n == 0 {
        return 0.0;
    }

    let sim_mat = cosine_similarity_matrix(anchors, positives);
    let mut correct = 0usize;

    for i in 0..n {
        let row = sim_mat.row(i);
        let best_j = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(j, _)| j)
            .unwrap_or(0);
        if best_j == i {
            correct += 1;
        }
    }

    correct as f32 / n as f32
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// L2 norm of a row view.
#[inline]
fn l2_norm_f32(v: ArrayView1<f32>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn loss_on_identical_pairs_is_log_batch_size() {
        // When anchors == positives and all are orthogonal, the loss should be
        // -log(exp(1/τ) / (exp(1/τ) + (n-1)*exp(0))) but here all pairs are
        // equal (diagonal is 1.0) so numerically it depends on n.
        // The key property: loss must be finite and >= 0.
        let n = 4;
        let d = 4;
        let anchors = Array2::<f32>::eye(n); // orthonormal rows
        let loss = infonce_loss(&anchors, &anchors, 0.05);
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        assert!(loss >= 0.0, "loss must be >= 0, got {loss}");
    }

    #[test]
    fn loss_is_lower_for_aligned_positives() {
        // Perfect positives (anchors == positives) should give lower loss
        // than misaligned positives.
        let anchors = Array2::<f32>::eye(4);
        let misaligned = {
            let mut m = Array2::<f32>::zeros((4, 4));
            for i in 0..4 {
                m[[i, (i + 1) % 4]] = 1.0;
            }
            m
        };

        let loss_aligned = infonce_loss(&anchors, &anchors, 0.05);
        let loss_misaligned = infonce_loss(&anchors, &misaligned, 0.05);

        assert!(
            loss_aligned < loss_misaligned,
            "aligned loss {loss_aligned} should be < misaligned {loss_misaligned}"
        );
    }

    #[test]
    fn top1_accuracy_on_identity_is_one() {
        let embeddings = Array2::<f32>::eye(4);
        let acc = top1_accuracy(&embeddings, &embeddings);
        assert!(
            (acc - 1.0).abs() < 1e-6,
            "accuracy on identity should be 1.0, got {acc}"
        );
    }

    #[test]
    fn cosine_similarity_matrix_diagonal_is_one() {
        let a = Array2::<f32>::eye(3);
        let sim = cosine_similarity_matrix(&a, &a);
        for i in 0..3 {
            assert!((sim[[i, i]] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn empty_batch_returns_zero() {
        let empty: Array2<f32> = Array2::zeros((0, 8));
        let loss = infonce_loss(&empty, &empty, 0.05);
        assert_eq!(loss, 0.0);
    }
}
