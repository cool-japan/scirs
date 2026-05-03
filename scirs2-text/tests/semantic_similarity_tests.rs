//! Integration tests for the semantic similarity API.

use scirs2_core::ndarray::{arr1, Array2};
use scirs2_text::sentence_embeddings::{
    semantic_similarity_matrix, semantic_similarity_tokens, semantic_similarity_vecs,
    vector_similarity, PairwiseSimilarityMetric,
};

// ── vector_similarity / cosine ────────────────────────────────────────────────

#[test]
fn cosine_of_identical_vectors_is_one() {
    let v = arr1(&[1.0f32, 2.0, 3.0]);
    let sim = vector_similarity(&v, &v, PairwiseSimilarityMetric::Cosine);
    assert!((sim - 1.0).abs() < 1e-6, "expected 1.0, got {sim}");
}

#[test]
fn cosine_of_orthogonal_vectors_is_zero() {
    let a = arr1(&[1.0f32, 0.0]);
    let b = arr1(&[0.0f32, 1.0]);
    let sim = vector_similarity(&a, &b, PairwiseSimilarityMetric::Cosine);
    assert!(sim.abs() < 1e-6, "expected 0.0, got {sim}");
}

#[test]
fn cosine_of_opposite_vectors_is_minus_one() {
    let a = arr1(&[1.0f32, 0.0]);
    let b = arr1(&[-1.0f32, 0.0]);
    let sim = vector_similarity(&a, &b, PairwiseSimilarityMetric::Cosine);
    assert!((sim + 1.0).abs() < 1e-6, "expected -1.0, got {sim}");
}

#[test]
fn cosine_of_zero_vector_returns_zero() {
    let a = arr1(&[0.0f32, 0.0, 0.0]);
    let b = arr1(&[1.0f32, 2.0, 3.0]);
    let sim = vector_similarity(&a, &b, PairwiseSimilarityMetric::Cosine);
    assert_eq!(sim, 0.0);
}

// ── vector_similarity / euclidean ─────────────────────────────────────────────

#[test]
fn euclidean_of_identical_vectors_is_zero() {
    let v = arr1(&[3.0f32, 4.0, 5.0]);
    let sim = vector_similarity(&v, &v, PairwiseSimilarityMetric::Euclidean);
    assert!((sim - 0.0).abs() < 1e-6, "expected 0.0, got {sim}");
}

#[test]
fn euclidean_is_negative_l2_distance() {
    let a = arr1(&[0.0f32, 0.0]);
    let b = arr1(&[3.0f32, 4.0]); // L2 = 5.0
    let sim = vector_similarity(&a, &b, PairwiseSimilarityMetric::Euclidean);
    assert!((sim + 5.0).abs() < 1e-5, "expected -5.0, got {sim}");
}

// ── vector_similarity / pearson ───────────────────────────────────────────────

#[test]
fn pearson_of_same_vector_is_one() {
    let v = arr1(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let sim = vector_similarity(&v, &v, PairwiseSimilarityMetric::Pearson);
    assert!((sim - 1.0).abs() < 1e-6, "expected 1.0, got {sim}");
}

#[test]
fn pearson_of_anticorrelated_vectors_is_minus_one() {
    let a = arr1(&[1.0f32, 2.0, 3.0]);
    let b = arr1(&[3.0f32, 2.0, 1.0]);
    let sim = vector_similarity(&a, &b, PairwiseSimilarityMetric::Pearson);
    assert!((sim + 1.0).abs() < 1e-5, "expected -1.0, got {sim}");
}

// ── vector_similarity / dot product ──────────────────────────────────────────

#[test]
fn dot_product_matches_manual() {
    let a = arr1(&[2.0f32, 3.0]);
    let b = arr1(&[4.0f32, 5.0]);
    // 2*4 + 3*5 = 23
    let sim = vector_similarity(&a, &b, PairwiseSimilarityMetric::DotProduct);
    assert!((sim - 23.0).abs() < 1e-5, "expected 23.0, got {sim}");
}

// ── vector_similarity / manhattan ────────────────────────────────────────────

#[test]
fn manhattan_of_identical_vectors_is_zero() {
    let v = arr1(&[7.0f32, -3.0, 2.5]);
    let sim = vector_similarity(&v, &v, PairwiseSimilarityMetric::Manhattan);
    assert!((sim - 0.0).abs() < 1e-6, "expected 0.0, got {sim}");
}

// ── semantic_similarity_vecs ──────────────────────────────────────────────────

#[test]
fn semantic_similarity_vecs_cosine_self() {
    let v = arr1(&[0.6f32, 0.8]);
    let sim = semantic_similarity_vecs(&v, &v, PairwiseSimilarityMetric::Cosine);
    assert!((sim - 1.0).abs() < 1e-6);
}

// ── semantic_similarity_tokens ────────────────────────────────────────────────

#[test]
fn semantic_similarity_tokens_same_is_one() {
    let encoder = |tokens: &[usize]| -> scirs2_core::ndarray::Array1<f32> {
        let mut v = scirs2_core::ndarray::Array1::<f32>::zeros(4);
        for &t in tokens {
            v[t % 4] += 1.0;
        }
        v
    };
    let sim =
        semantic_similarity_tokens(&[0, 1], &[0, 1], encoder, PairwiseSimilarityMetric::Cosine);
    assert!((sim - 1.0).abs() < 1e-6, "expected 1.0, got {sim}");
}

#[test]
fn semantic_similarity_tokens_orthogonal_is_zero() {
    let encoder = |tokens: &[usize]| -> scirs2_core::ndarray::Array1<f32> {
        let mut v = scirs2_core::ndarray::Array1::<f32>::zeros(4);
        for &t in tokens {
            v[t % 4] += 1.0;
        }
        v
    };
    // token 0 → dim 0, token 1 → dim 1 — orthogonal
    let sim = semantic_similarity_tokens(&[0], &[1], encoder, PairwiseSimilarityMetric::Cosine);
    assert!(sim.abs() < 1e-6, "expected 0.0, got {sim}");
}

// ── semantic_similarity_matrix ────────────────────────────────────────────────

#[test]
fn semantic_sim_matrix_shape_is_n_by_n() {
    let embs = Array2::<f32>::from_shape_fn((4, 6), |(i, j)| if j == i { 1.0 } else { 0.0 });
    let mat = semantic_similarity_matrix(&embs, PairwiseSimilarityMetric::Cosine);
    assert_eq!(mat.shape(), &[4, 4]);
}

#[test]
fn semantic_sim_matrix_is_symmetric() {
    let embs = Array2::<f32>::from_shape_fn((4, 3), |(i, j)| ((i * 3 + j + 1) as f32).sin());
    let mat = semantic_similarity_matrix(&embs, PairwiseSimilarityMetric::Cosine);
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                (mat[[i, j]] - mat[[j, i]]).abs() < 1e-9,
                "mat[{i},{j}] != mat[{j},{i}]"
            );
        }
    }
}

#[test]
fn semantic_sim_matrix_diagonal_is_one_for_cosine() {
    let embs = Array2::<f32>::from_shape_fn((5, 4), |(i, j)| ((i * 4 + j + 1) as f32).cos());
    let mat = semantic_similarity_matrix(&embs, PairwiseSimilarityMetric::Cosine);
    for i in 0..5 {
        assert!(
            (mat[[i, i]] - 1.0).abs() < 1e-5,
            "diagonal[{i}] = {} ≠ 1.0",
            mat[[i, i]]
        );
    }
}

#[test]
fn semantic_sim_matrix_single_row() {
    let v = arr1(&[1.0f32, 0.0, 0.0]);
    let embs = v.clone().insert_axis(scirs2_core::ndarray::Axis(0));
    let mat = semantic_similarity_matrix(&embs, PairwiseSimilarityMetric::Cosine);
    assert_eq!(mat.shape(), &[1, 1]);
    assert!((mat[[0, 0]] - 1.0).abs() < 1e-6);
}
