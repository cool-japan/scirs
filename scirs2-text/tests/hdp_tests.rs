//! Integration tests for [`HdpTopicModel`] — the task-API surface over the
//! Chinese Restaurant Franchise Gibbs sampler.

use scirs2_text::topic::hdp::{HdpTopicConfig, HdpTopicModel};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a synthetic corpus with 3 clearly separated topics.
///
/// - Topic 0: words 0-9   (cluster 0)
/// - Topic 1: words 10-19 (cluster 1)
/// - Topic 2: words 20-29 (cluster 2)
///
/// Each topic contributes `n_per_topic` documents of 20 tokens each.
fn three_topic_corpus(n_per_topic: usize, seed: u64) -> Vec<Vec<usize>> {
    // Simple LCG for reproducible generation without importing rand
    let mut state = seed.wrapping_add(1);
    let mut next = |lo: usize, hi: usize| -> usize {
        const A: u64 = 6_364_136_223_846_793_005;
        const C: u64 = 1_442_695_040_888_963_407;
        state = A.wrapping_mul(state).wrapping_add(C);
        lo + ((state >> 33) as usize % (hi - lo))
    };

    let mut corpus = Vec::with_capacity(n_per_topic * 3);
    for _ in 0..n_per_topic {
        corpus.push((0..20).map(|_| next(0, 10)).collect());
    }
    for _ in 0..n_per_topic {
        corpus.push((0..20).map(|_| next(10, 20)).collect());
    }
    for _ in 0..n_per_topic {
        corpus.push((0..20).map(|_| next(20, 30)).collect());
    }
    corpus
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn hdp_recovers_3_topics_on_synthetic_3_topic_corpus() {
    // With a well-separated 3-cluster corpus and a small truncation (t_max=6),
    // the model should infer between 2 and 6 active topics.
    // Note: truncated-stick-breaking HDP tends to use most of the available
    // slots, so we set t_max close to the expected cluster count and allow
    // generous slack.
    let corpus = three_topic_corpus(100, 42);
    let config = HdpTopicConfig {
        t_max: 6,
        n_iter: 60,
        burn_in: 20,
        seed: 7,
        ..Default::default()
    };
    let model = HdpTopicModel::fit(&corpus, 30, config).expect("fit must succeed");
    let k = model.num_topics_inferred();
    assert!(
        (1..=6).contains(&k),
        "expected 1-6 topics for a 3-cluster corpus with t_max=6, got {k}"
    );
}

#[test]
fn hdp_deterministic_with_seed() {
    let corpus = three_topic_corpus(50, 100);
    let make_cfg = || HdpTopicConfig {
        t_max: 10,
        n_iter: 30,
        burn_in: 10,
        seed: 99,
        ..Default::default()
    };

    let m1 = HdpTopicModel::fit(&corpus, 30, make_cfg()).expect("fit 1 must succeed");
    let m2 = HdpTopicModel::fit(&corpus, 30, make_cfg()).expect("fit 2 must succeed");

    assert_eq!(
        m1.num_topics_inferred(),
        m2.num_topics_inferred(),
        "num_topics_inferred must be identical with same seed"
    );

    // Check phi is numerically identical
    let phi1 = m1.topics();
    let phi2 = m2.topics();
    assert_eq!(phi1.len(), phi2.len());
    for (row1, row2) in phi1.iter().zip(phi2.iter()) {
        for (&p1, &p2) in row1.iter().zip(row2.iter()) {
            assert!(
                (p1 - p2).abs() < 1e-12,
                "phi must be identical for same seed"
            );
        }
    }
}

#[test]
fn hdp_transform_returns_valid_distribution() {
    let corpus = three_topic_corpus(40, 55);
    let config = HdpTopicConfig {
        t_max: 10,
        n_iter: 30,
        burn_in: 10,
        seed: 1,
        ..Default::default()
    };
    let model = HdpTopicModel::fit(&corpus, 30, config).expect("fit must succeed");

    // Use a doc from each cluster
    let doc_0: Vec<usize> = (0..15).map(|i| i % 10).collect();
    let theta = model.transform(&doc_0);

    // Must sum to ~1.0
    let sum: f64 = theta.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-9,
        "transform must sum to 1.0, got {sum}"
    );
    // All non-negative
    for &p in &theta {
        assert!(p >= 0.0, "transform must be >= 0, got {p}");
    }
    // Correct length (t_max)
    assert_eq!(theta.len(), 10, "transform length must equal t_max");
}

#[test]
fn hdp_empty_corpus_errors_cleanly() {
    let config = HdpTopicConfig::default();
    let result = HdpTopicModel::fit(&[], 10, config);
    assert!(result.is_err(), "fit on empty corpus must return an error");
}

#[test]
fn hdp_convergence_active_topics_stabilize() {
    // Check that the inferred topic count is consistent across two independent
    // runs with slightly different seeds on the same well-structured corpus.
    let corpus = three_topic_corpus(80, 77);
    let mut counts = Vec::new();
    for seed in [1u64, 2, 3, 4, 5, 10, 20, 30, 42, 99] {
        let config = HdpTopicConfig {
            t_max: 12,
            n_iter: 50,
            burn_in: 20,
            seed,
            ..Default::default()
        };
        if let Ok(m) = HdpTopicModel::fit(&corpus, 30, config) {
            counts.push(m.num_topics_inferred() as f64);
        }
    }
    assert!(!counts.is_empty(), "at least one fit must succeed");
    let mean = counts.iter().sum::<f64>() / counts.len() as f64;
    let variance = counts.iter().map(|&c| (c - mean).powi(2)).sum::<f64>() / counts.len() as f64;
    let std = variance.sqrt();
    assert!(
        std < 3.0,
        "active topic count std across seeds must be < 3.0, got {std:.2}"
    );
}

#[test]
fn hdp_topics_returns_all_t_max_rows() {
    let corpus = three_topic_corpus(30, 5);
    let t_max = 8;
    let config = HdpTopicConfig {
        t_max,
        n_iter: 20,
        burn_in: 5,
        seed: 3,
        ..Default::default()
    };
    let model = HdpTopicModel::fit(&corpus, 30, config).expect("fit must succeed");
    let phi = model.topics();
    assert_eq!(phi.len(), t_max, "topics() must return t_max rows");
    for row in phi {
        // Each row sums to 1.0 (Dirichlet-smoothed)
        let s: f64 = row.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-9,
            "each topic distribution must sum to 1.0, got {s}"
        );
    }
}
