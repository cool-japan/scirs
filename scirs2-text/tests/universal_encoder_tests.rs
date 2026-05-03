//! Integration tests for [`UniversalSentenceEncoder`].

use scirs2_core::ndarray::Array2;
use scirs2_text::sentence_embeddings::universal::{
    UniversalPoolingStrategy, UniversalSentenceEncoder,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a small deterministic embedding matrix via a simple LCG.
///
/// Shape: `[vocab_size × d_model]`.  Values are in `[-1, 1)`.
fn make_embeddings(vocab_size: usize, d_model: usize, seed: u64) -> Array2<f32> {
    Array2::from_shape_fn((vocab_size, d_model), |(i, j)| {
        let state: u64 = 6_364_136_223_846_793_005_u64
            .wrapping_mul(seed.wrapping_add(i as u64 * d_model as u64 + j as u64))
            .wrapping_add(1_442_695_040_888_963_407);
        // Map to [-1, 1)
        ((state >> 12) as f32) / ((1u64 << 52) as f32) * 2.0 - 1.0
    })
}

/// Compute the L2 norm of a slice.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn use_mean_pool_equals_average_of_token_vectors() {
    let vocab = 20;
    let d = 16;
    let emb = make_embeddings(vocab, d, 1);
    let encoder = UniversalSentenceEncoder::new(emb.clone(), UniversalPoolingStrategy::Mean, false);

    let tokens = vec![3usize, 7, 11];
    let result = encoder.encode(&tokens);

    // Manually compute mean
    let mut expected = vec![0.0f32; d];
    for &t in &tokens {
        for j in 0..d {
            expected[j] += emb[[t, j]];
        }
    }
    expected.iter_mut().for_each(|v| *v /= tokens.len() as f32);

    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-6, "mean pool mismatch: {r} vs {e}");
    }
}

#[test]
fn use_cls_token_returns_first_token_embedding() {
    let vocab = 10;
    let d = 8;
    let emb = make_embeddings(vocab, d, 2);
    let encoder =
        UniversalSentenceEncoder::new(emb.clone(), UniversalPoolingStrategy::ClsToken, false);

    let tokens = vec![5usize, 2, 9];
    let result = encoder.encode(&tokens);

    // Should equal row 5 of the embedding matrix
    for j in 0..d {
        assert!(
            (result[j] - emb[[5, j]]).abs() < 1e-7,
            "ClsToken mismatch at dim {j}"
        );
    }
}

#[test]
fn use_attention_pool_weights_sum_to_one() {
    let vocab = 15;
    let d = 12;
    let emb = make_embeddings(vocab, d, 3);
    let mut encoder =
        UniversalSentenceEncoder::new(emb, UniversalPoolingStrategy::AttentionPooling, false);

    // Fit attention pooling (1 epoch, small corpus)
    let corpus: Vec<Vec<usize>> = vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]];
    encoder.fit_attention_pooling(&corpus, 1, 0.01);

    // After fitting, encode and verify output is finite and non-zero
    let tokens = vec![2usize, 5, 8];
    let result = encoder.encode(&tokens);
    assert_eq!(result.len(), d);
    let norm = l2_norm(result.as_slice().expect("contiguous"));
    assert!(norm > 1e-6, "AttentionPooling output must be non-trivial");

    // The query must now be set
    assert!(
        encoder.attention_query().is_some(),
        "attention_query must be set after fit_attention_pooling"
    );
}

#[test]
fn use_max_pool_componentwise_max() {
    let vocab = 10;
    let d = 6;
    let emb = make_embeddings(vocab, d, 4);
    let encoder = UniversalSentenceEncoder::new(emb.clone(), UniversalPoolingStrategy::Max, false);

    let tokens = vec![1usize, 4, 7];
    let result = encoder.encode(&tokens);

    // Verify component-wise max
    for j in 0..d {
        let expected_max = tokens
            .iter()
            .map(|&t| emb[[t, j]])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (result[j] - expected_max).abs() < 1e-7,
            "max pool mismatch at dim {j}: {} vs {}",
            result[j],
            expected_max
        );
    }
}

#[test]
fn use_normalized_output_has_unit_norm() {
    let vocab = 20;
    let d = 32;
    let emb = make_embeddings(vocab, d, 5);

    // Test with Mean pooling + normalize
    let encoder = UniversalSentenceEncoder::new(emb, UniversalPoolingStrategy::Mean, true);
    let tokens = vec![2usize, 7, 14, 18];
    let result = encoder.encode(&tokens);

    let norm = l2_norm(result.as_slice().expect("contiguous"));
    assert!(
        (norm - 1.0).abs() < 1e-6,
        "normalized output must have unit norm, got {norm}"
    );
}

#[test]
fn use_weighted_mean_by_idf_downweights_frequent_tokens() {
    let vocab = 10;
    let d = 8;
    let emb = make_embeddings(vocab, d, 6);
    let mut encoder =
        UniversalSentenceEncoder::new(emb, UniversalPoolingStrategy::WeightedMean, false);

    // Build a corpus where token 0 appears in every document (low IDF)
    // and token 9 appears in only one document (high IDF)
    let corpus: Vec<Vec<usize>> = (0..10)
        .map(|i| {
            if i == 5 {
                vec![0usize, 9]
            } else {
                vec![0usize, i % vocab]
            }
        })
        .collect();
    encoder.fit_idf_weights(&corpus, vocab);

    let idf = encoder.idf_weights().expect("IDF must be set after fit");
    // Token 0 appears in all 10 docs: idf[0] = ln(11/11) = 0.0
    // Token 9 appears in 1 doc: idf[9] = ln(11/2) ~ 1.7
    assert!(
        idf[0] < idf[9],
        "frequent token 0 must have lower IDF ({}) than rare token 9 ({})",
        idf[0],
        idf[9]
    );

    // Encode a doc: result should be finite and non-trivial
    let tokens = vec![0usize, 9];
    let result = encoder.encode(&tokens);
    assert_eq!(result.len(), d);
    assert!(result.iter().all(|x| x.is_finite()));
}

#[test]
fn use_mean_sqrt_smaller_norm_than_mean_for_long_seq() {
    let vocab = 20;
    let d = 16;
    let emb = make_embeddings(vocab, d, 7);

    let encoder_mean =
        UniversalSentenceEncoder::new(emb.clone(), UniversalPoolingStrategy::Mean, false);
    let encoder_msqrt =
        UniversalSentenceEncoder::new(emb, UniversalPoolingStrategy::MeanSqrt, false);

    // Long sequence: 16 tokens — MeanSqrt divides by sqrt(16) = 4
    let tokens: Vec<usize> = (0..16).map(|i| i % vocab).collect();

    let mean_result = encoder_mean.encode(&tokens);
    let msqrt_result = encoder_msqrt.encode(&tokens);

    let norm_mean = l2_norm(mean_result.as_slice().expect("contiguous"));
    let norm_msqrt = l2_norm(msqrt_result.as_slice().expect("contiguous"));

    assert!(
        norm_msqrt < norm_mean,
        "MeanSqrt norm ({norm_msqrt}) must be smaller than Mean norm ({norm_mean}) for long seq"
    );
}

#[test]
fn use_empty_tokens_returns_zero_vector() {
    let vocab = 10;
    let d = 8;
    let emb = make_embeddings(vocab, d, 8);
    let encoder = UniversalSentenceEncoder::new(emb, UniversalPoolingStrategy::Mean, false);

    let result = encoder.encode(&[]);
    assert_eq!(result.len(), d);
    assert!(
        result.iter().all(|&x| x == 0.0),
        "empty input must give zero vector"
    );
}

#[test]
fn use_debug_format_works() {
    let emb = make_embeddings(5, 4, 9);
    let encoder = UniversalSentenceEncoder::new(emb, UniversalPoolingStrategy::Mean, true);
    let dbg = format!("{:?}", encoder);
    assert!(dbg.contains("UniversalSentenceEncoder"));
}
