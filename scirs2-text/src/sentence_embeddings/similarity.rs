//! Semantic similarity utilities for sentence embeddings.
//!
//! Provides a metric enum and free functions for computing pairwise and batch
//! similarity between embedding vectors, without requiring any specific encoder
//! implementation.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::ndarray::arr1;
//! use scirs2_text::sentence_embeddings::{PairwiseSimilarityMetric, vector_similarity};
//!
//! let a = arr1(&[1.0f32, 0.0, 0.0]);
//! let b = arr1(&[0.0f32, 1.0, 0.0]);
//! let sim = vector_similarity(&a, &b, PairwiseSimilarityMetric::Cosine);
//! assert!((sim).abs() < 1e-6);
//! ```

use scirs2_core::ndarray::{Array1, Array2, Axis};

// ── SentenceEncoderLike ────────────────────────────────────────────────────────

/// Trait for any encoder that can produce a dense vector from text.
///
/// Implementations must be `Send + Sync` so they can be used in parallel
/// processing pipelines.
pub trait SentenceEncoderLike: Send + Sync {
    /// Encode a text string into a dense embedding vector of length
    /// [`Self::d_model`].
    fn encode_text(&self, text: &str) -> Array1<f32>;

    /// Dimensionality of the output embedding.
    fn d_model(&self) -> usize;
}

// ── PairwiseSimilarityMetric ───────────────────────────────────────────────────

/// Similarity metric used when comparing two embedding vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairwiseSimilarityMetric {
    /// Cosine similarity: in `[-1, 1]`, `1` = identical direction.
    Cosine,
    /// Negative Euclidean distance: `0` for identical vectors, more negative
    /// as vectors diverge — so "higher = more similar" holds uniformly.
    Euclidean,
    /// Raw dot product (useful when embeddings are pre-normalised).
    DotProduct,
    /// Negative L1 (Manhattan) distance (higher = more similar).
    Manhattan,
    /// Pearson correlation coefficient: in `[-1, 1]`.
    Pearson,
}

// ── vector_similarity ─────────────────────────────────────────────────────────

/// Compute the scalar similarity between two f32 embedding vectors.
///
/// The chosen `metric` determines the range and interpretation of the
/// returned value, but for all metrics **higher always means more similar**.
///
/// # Panics
/// Does not panic; zero-norm / zero-variance vectors are handled gracefully
/// (returns `0.0`).
pub fn vector_similarity(
    a: &Array1<f32>,
    b: &Array1<f32>,
    metric: PairwiseSimilarityMetric,
) -> f32 {
    match metric {
        PairwiseSimilarityMetric::Cosine => {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na < 1e-12 || nb < 1e-12 {
                0.0
            } else {
                (dot / (na * nb)).clamp(-1.0, 1.0)
            }
        }

        PairwiseSimilarityMetric::Euclidean => {
            let diff = a - b;
            let dist: f32 = diff.iter().map(|x| x * x).sum::<f32>().sqrt();
            -dist
        }

        PairwiseSimilarityMetric::DotProduct => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),

        PairwiseSimilarityMetric::Manhattan => {
            let dist: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
            -dist
        }

        PairwiseSimilarityMetric::Pearson => {
            let n = a.len() as f32;
            if n < 2.0 {
                return 0.0;
            }
            let mean_a: f32 = a.iter().sum::<f32>() / n;
            let mean_b: f32 = b.iter().sum::<f32>() / n;
            let ca: Vec<f32> = a.iter().map(|x| x - mean_a).collect();
            let cb: Vec<f32> = b.iter().map(|x| x - mean_b).collect();
            let num: f32 = ca.iter().zip(cb.iter()).map(|(x, y)| x * y).sum();
            let var_a: f32 = ca.iter().map(|x| x * x).sum::<f32>();
            let var_b: f32 = cb.iter().map(|x| x * x).sum::<f32>();
            let denom = (var_a * var_b).sqrt();
            if denom < 1e-12 {
                0.0
            } else {
                (num / denom).clamp(-1.0, 1.0)
            }
        }
    }
}

// ── semantic_similarity_vecs ──────────────────────────────────────────────────

/// Compute similarity directly from two pre-computed embedding vectors.
///
/// This is the primary entry point when embeddings are already available.
///
/// ```rust
/// use scirs2_core::ndarray::arr1;
/// use scirs2_text::sentence_embeddings::{PairwiseSimilarityMetric, semantic_similarity_vecs};
///
/// let v = arr1(&[1.0f32, 0.0]);
/// let sim = semantic_similarity_vecs(&v, &v, PairwiseSimilarityMetric::Cosine);
/// assert!((sim - 1.0).abs() < 1e-6);
/// ```
pub fn semantic_similarity_vecs(
    v1: &Array1<f32>,
    v2: &Array1<f32>,
    metric: PairwiseSimilarityMetric,
) -> f32 {
    vector_similarity(v1, v2, metric)
}

// ── semantic_similarity_tokens ────────────────────────────────────────────────

/// Compute similarity between two token-ID sequences using a provided encoder
/// closure.
///
/// # Parameters
/// - `tokens1` / `tokens2`: token-ID slices to encode.
/// - `encoder_fn`: a function mapping a token-ID slice to an `Array1<f32>`.
/// - `metric`: the similarity metric to apply after encoding.
pub fn semantic_similarity_tokens<F>(
    tokens1: &[usize],
    tokens2: &[usize],
    encoder_fn: F,
    metric: PairwiseSimilarityMetric,
) -> f32
where
    F: Fn(&[usize]) -> Array1<f32>,
{
    let v1 = encoder_fn(tokens1);
    let v2 = encoder_fn(tokens2);
    vector_similarity(&v1, &v2, metric)
}

// ── semantic_similarity_matrix ────────────────────────────────────────────────

/// Compute the full `n × n` pairwise similarity matrix for a batch of
/// embedding vectors.
///
/// `embeddings` has shape `[n, d]`.  The returned matrix is symmetric and,
/// for `Cosine` metric, its diagonal is all-ones when embeddings are
/// non-zero.
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_text::sentence_embeddings::{PairwiseSimilarityMetric, semantic_similarity_matrix};
///
/// // 3 two-dimensional embeddings
/// let embs = Array2::from_shape_vec((3, 2), vec![
///     1.0f32, 0.0,   // e0
///     0.0,    1.0,   // e1
///     1.0,    0.0,   // e2 (= e0)
/// ]).unwrap();
///
/// let m = semantic_similarity_matrix(&embs, PairwiseSimilarityMetric::Cosine);
/// assert_eq!(m.shape(), &[3, 3]);
/// // e0 and e2 are identical → cosine 1.0
/// assert!((m[[0, 2]] - 1.0).abs() < 1e-6);
/// ```
pub fn semantic_similarity_matrix(
    embeddings: &Array2<f32>,
    metric: PairwiseSimilarityMetric,
) -> Array2<f32> {
    let n = embeddings.nrows();
    let mut result = Array2::<f32>::zeros((n, n));

    for i in 0..n {
        let row_i = embeddings.index_axis(Axis(0), i).to_owned();
        // diagonal
        result[[i, i]] = vector_similarity(&row_i, &row_i, metric);

        for j in (i + 1)..n {
            let row_j = embeddings.index_axis(Axis(0), j).to_owned();
            let sim = vector_similarity(&row_i, &row_j, metric);
            result[[i, j]] = sim;
            result[[j, i]] = sim;
        }
    }

    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    // ── cosine tests ──────────────────────────────────────────────────────────

    #[test]
    fn cosine_similarity_identical_vectors_is_one() {
        let v = arr1(&[1.0f32, 2.0, 3.0]);
        let sim = vector_similarity(&v, &v, PairwiseSimilarityMetric::Cosine);
        assert!((sim - 1.0).abs() < 1e-6, "expected 1.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors_is_zero() {
        let a = arr1(&[1.0f32, 0.0, 0.0]);
        let b = arr1(&[0.0f32, 1.0, 0.0]);
        let sim = vector_similarity(&a, &b, PairwiseSimilarityMetric::Cosine);
        assert!(sim.abs() < 1e-6, "expected 0.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_opposite_vectors_is_minus_one() {
        let a = arr1(&[1.0f32, 0.0]);
        let b = arr1(&[-1.0f32, 0.0]);
        let sim = vector_similarity(&a, &b, PairwiseSimilarityMetric::Cosine);
        assert!((sim + 1.0).abs() < 1e-6, "expected -1.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_zero_vector_returns_zero() {
        let a = arr1(&[0.0f32, 0.0]);
        let b = arr1(&[1.0f32, 1.0]);
        let sim = vector_similarity(&a, &b, PairwiseSimilarityMetric::Cosine);
        assert_eq!(sim, 0.0);
    }

    // ── euclidean tests ───────────────────────────────────────────────────────

    #[test]
    fn euclidean_of_identical_vectors_is_zero() {
        let v = arr1(&[3.0f32, 4.0]);
        let sim = vector_similarity(&v, &v, PairwiseSimilarityMetric::Euclidean);
        assert!((sim - 0.0).abs() < 1e-6, "expected 0.0, got {sim}");
    }

    #[test]
    fn euclidean_is_negative_distance() {
        let a = arr1(&[0.0f32, 0.0]);
        let b = arr1(&[3.0f32, 4.0]); // distance = 5
        let sim = vector_similarity(&a, &b, PairwiseSimilarityMetric::Euclidean);
        assert!((sim + 5.0).abs() < 1e-5, "expected -5.0, got {sim}");
    }

    // ── dot product tests ─────────────────────────────────────────────────────

    #[test]
    fn dot_product_unit_vectors() {
        let a = arr1(&[1.0f32, 0.0]);
        let b = arr1(&[0.0f32, 1.0]);
        let sim = vector_similarity(&a, &b, PairwiseSimilarityMetric::DotProduct);
        assert!((sim - 0.0).abs() < 1e-6);
    }

    // ── manhattan tests ───────────────────────────────────────────────────────

    #[test]
    fn manhattan_of_identical_vectors_is_zero() {
        let v = arr1(&[1.0f32, 2.0, 3.0]);
        let sim = vector_similarity(&v, &v, PairwiseSimilarityMetric::Manhattan);
        assert!((sim - 0.0).abs() < 1e-6);
    }

    // ── pearson tests ─────────────────────────────────────────────────────────

    #[test]
    fn pearson_of_same_vector_is_one() {
        let v = arr1(&[1.0f32, 2.0, 3.0, 4.0]);
        let sim = vector_similarity(&v, &v, PairwiseSimilarityMetric::Pearson);
        assert!((sim - 1.0).abs() < 1e-6, "expected 1.0, got {sim}");
    }

    #[test]
    fn pearson_of_perfectly_anticorrelated_is_minus_one() {
        let a = arr1(&[1.0f32, 2.0, 3.0]);
        let b = arr1(&[3.0f32, 2.0, 1.0]);
        let sim = vector_similarity(&a, &b, PairwiseSimilarityMetric::Pearson);
        assert!((sim + 1.0).abs() < 1e-5, "expected -1.0, got {sim}");
    }

    // ── semantic_similarity_vecs ──────────────────────────────────────────────

    #[test]
    fn semantic_similarity_vecs_is_alias() {
        let a = arr1(&[1.0f32, 0.0, 0.0]);
        let b = arr1(&[0.0f32, 0.0, 1.0]);
        let direct = vector_similarity(&a, &b, PairwiseSimilarityMetric::Cosine);
        let via_fn = semantic_similarity_vecs(&a, &b, PairwiseSimilarityMetric::Cosine);
        assert!((direct - via_fn).abs() < 1e-9);
    }

    // ── semantic_similarity_tokens ────────────────────────────────────────────

    #[test]
    fn semantic_similarity_tokens_uses_encoder() {
        // encoder: each token → one-hot at position t (mod 4)
        let encoder = |tokens: &[usize]| -> Array1<f32> {
            let mut v = Array1::<f32>::zeros(4);
            for &t in tokens {
                v[t % 4] += 1.0;
            }
            v
        };
        // [0] and [0] should be identical
        let sim = semantic_similarity_tokens(&[0], &[0], encoder, PairwiseSimilarityMetric::Cosine);
        assert!((sim - 1.0).abs() < 1e-6);
        // [0] and [1] are orthogonal
        let sim2 =
            semantic_similarity_tokens(&[0], &[1], encoder, PairwiseSimilarityMetric::Cosine);
        assert!(sim2.abs() < 1e-6);
    }

    // ── semantic_similarity_matrix ────────────────────────────────────────────

    #[test]
    fn semantic_sim_matrix_shape_is_n_by_n() {
        use scirs2_core::ndarray::Array2;
        let embs = Array2::<f32>::from_shape_fn((5, 8), |(i, j)| if j == i { 1.0 } else { 0.0 });
        let mat = semantic_similarity_matrix(&embs, PairwiseSimilarityMetric::Cosine);
        assert_eq!(mat.shape(), &[5, 5]);
    }

    #[test]
    fn semantic_sim_matrix_diagonal_is_one_for_cosine() {
        use scirs2_core::ndarray::Array2;
        let embs = Array2::<f32>::from_shape_fn((4, 3), |(i, j)| lcg_f32(42, (i * 3 + j) as u64));
        let mat = semantic_similarity_matrix(&embs, PairwiseSimilarityMetric::Cosine);
        for i in 0..4 {
            assert!(
                (mat[[i, i]] - 1.0).abs() < 1e-5,
                "diagonal[{i}] = {} ≠ 1.0",
                mat[[i, i]]
            );
        }
    }

    #[test]
    fn semantic_sim_matrix_is_symmetric() {
        use scirs2_core::ndarray::Array2;
        let embs = Array2::<f32>::from_shape_fn((4, 3), |(i, j)| lcg_f32(7, (i * 3 + j) as u64));
        let mat = semantic_similarity_matrix(&embs, PairwiseSimilarityMetric::Cosine);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (mat[[i, j]] - mat[[j, i]]).abs() < 1e-9,
                    "mat[{i},{j}]={} ≠ mat[{j},{i}]={}",
                    mat[[i, j]],
                    mat[[j, i]]
                );
            }
        }
    }

    // ── helper ────────────────────────────────────────────────────────────────

    fn lcg_f32(seed: u64, offset: u64) -> f32 {
        const A: u64 = 6_364_136_223_846_793_005;
        const C: u64 = 1_442_695_040_888_963_407;
        let state = A.wrapping_mul(seed.wrapping_add(offset)).wrapping_add(C);
        (((state >> 12) as f64) / ((1u64 << 52) as f64)) as f32 * 2.0 - 1.0
    }
}
