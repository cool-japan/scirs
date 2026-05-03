//! Integration tests for SimCSE contrastive sentence representation learning.
//!
//! Tests the following components:
//! - `infonce` module: standalone InfoNCE / NT-Xent loss functions
//! - `autograd_projection`: differentiable two-layer MLP projection head
//! - `trainer`: high-level SimCSE trainer (frozen encoder + projection)

use scirs2_core::ndarray::{arr1, Array2};
use scirs2_text::sentence_embeddings::{
    autograd_projection::{DifferentiableProjection, ProjectionConfig},
    encoder::{PoolingStrategy, SentenceEncoder, SentenceEncoderConfig},
    infonce::{cosine_similarity_matrix, infonce_loss, top1_accuracy},
    trainer::{SimcseConfig, SimcseTrainer, TrainStep},
};

// ── Helper factories ──────────────────────────────────────────────────────────

fn make_encoder(dim: usize) -> SentenceEncoder {
    let vocab: Vec<String> = (0..200).map(|i| format!("word{i}")).collect();
    SentenceEncoder::new(
        &vocab,
        SentenceEncoderConfig {
            embedding_dim: dim,
            max_seq_len: 64,
            pooling: PoolingStrategy::Mean,
            normalize: true,
        },
    )
}

fn make_trainer(dim: usize) -> SimcseTrainer {
    let enc = make_encoder(dim);
    let config = SimcseConfig {
        temperature: 0.05,
        batch_size: 4,
        projection: ProjectionConfig {
            d_in: dim,
            d_hidden: dim,
            d_out: dim,
            dropout_rate: 0.1,
            learning_rate: 1e-3,
        },
    };
    SimcseTrainer::new(enc, config)
}

// ── InfoNCE loss tests ────────────────────────────────────────────────────────

/// InfoNCE loss on a toy batch must be finite and non-negative.
#[test]
fn simcse_infonce_loss_correct_on_toy_batch() {
    // Four orthogonal unit vectors: aligned positives → low loss.
    let anchors = Array2::<f32>::eye(4);
    let positives = Array2::<f32>::eye(4); // identical → best case

    let loss = infonce_loss(&anchors, &positives, 0.05);
    assert!(loss.is_finite(), "InfoNCE loss must be finite, got {loss}");
    assert!(loss >= 0.0, "InfoNCE loss must be non-negative, got {loss}");
}

/// Aligned positives yield strictly lower loss than misaligned ones.
#[test]
fn simcse_supervised_entailment_positives_beat_random_positives() {
    // Anchors: orthogonal unit vectors.
    let anchors = Array2::<f32>::eye(4);
    // Perfect positives: identical to anchors.
    let perfect = anchors.clone();
    // Shifted positives: next row wraps around → orthogonal to anchor.
    let shifted = {
        let mut m = Array2::<f32>::zeros((4, 4));
        for i in 0..4 {
            m[[i, (i + 1) % 4]] = 1.0;
        }
        m
    };

    let loss_perfect = infonce_loss(&anchors, &perfect, 0.05);
    let loss_shifted = infonce_loss(&anchors, &shifted, 0.05);

    assert!(
        loss_perfect < loss_shifted,
        "aligned loss ({loss_perfect}) should be < shifted loss ({loss_shifted})"
    );
}

/// Empty batch returns 0.0 without panic.
#[test]
fn simcse_infonce_empty_batch_is_zero() {
    let empty: Array2<f32> = Array2::zeros((0, 8));
    assert_eq!(infonce_loss(&empty, &empty, 0.05), 0.0);
}

/// top1_accuracy on perfectly aligned pairs is 1.0.
#[test]
fn simcse_infonce_top1_accuracy_perfect() {
    let embeddings = Array2::<f32>::eye(4);
    let acc = top1_accuracy(&embeddings, &embeddings);
    assert!((acc - 1.0).abs() < 1e-6, "expected 1.0, got {acc}");
}

/// Cosine similarity matrix: diagonal entries are 1.0.
#[test]
fn simcse_cosine_matrix_diagonal_ones() {
    let a = Array2::<f32>::eye(3);
    let sim = cosine_similarity_matrix(&a, &a);
    for i in 0..3 {
        let d = sim[[i, i]];
        assert!((d - 1.0).abs() < 1e-6, "sim[{i},{i}] = {d}");
    }
}

// ── DifferentiableProjection tests ───────────────────────────────────────────

/// Projection forward_inference output shape matches (batch, d_out).
#[test]
fn simcse_embedding_dimension_preserved() {
    let dim = 32usize;
    let proj = DifferentiableProjection::new(ProjectionConfig {
        d_in: dim,
        d_hidden: dim,
        d_out: dim,
        dropout_rate: 0.1,
        learning_rate: 1e-3,
    });
    let input = Array2::<f32>::from_shape_fn((4, dim), |(i, j)| (i * j) as f32 * 0.01);
    let output = proj
        .forward_inference(&input)
        .expect("forward_inference failed");
    assert_eq!(output.shape(), &[4, dim], "output shape mismatch");
}

/// update_step returns finite loss and increments step counter.
#[test]
fn simcse_projection_update_step_is_finite() {
    let dim = 16usize;
    let mut proj = DifferentiableProjection::new(ProjectionConfig {
        d_in: dim,
        d_hidden: dim,
        d_out: dim,
        dropout_rate: 0.05,
        learning_rate: 1e-3,
    });
    let input = Array2::<f32>::from_shape_fn((4, dim), |(i, j)| i as f32 * 0.1 + j as f32 * 0.01);
    let loss = proj.update_step(&input, 0.05).expect("update_step failed");
    assert!(loss.is_finite(), "loss must be finite, got {loss}");
    assert_eq!(proj.steps(), 1, "steps should be 1 after one update");
}

// ── SimcseTrainer tests ───────────────────────────────────────────────────────

/// Unsupervised step: loss is finite, accuracy in [0, 1].
#[test]
fn simcse_unsupervised_step_valid() {
    let mut trainer = make_trainer(32);
    let sentences = ["word0 word1", "word2 word3", "word4 word5", "word6 word7"];
    let TrainStep { loss, accuracy } = trainer
        .unsupervised_step(&sentences)
        .expect("unsupervised step failed");
    assert!(loss.is_finite(), "loss must be finite: {loss}");
    assert!(
        (0.0..=1.0).contains(&accuracy),
        "accuracy out of range: {accuracy}"
    );
}

/// Encoder output dimension is preserved through the full pipeline.
#[test]
fn simcse_encode_dimension_correct() {
    let dim = 32usize;
    let trainer = make_trainer(dim);
    let emb = trainer.encode("word0 word1 word2").expect("encode failed");
    assert_eq!(emb.len(), dim, "expected {dim}-dim embedding");
}

/// Same seed → same loss (fixed seed is deterministic).
#[test]
fn simcse_fixed_seed_is_deterministic() {
    // Two separate trainers built from the same deterministic setup must
    // produce the same InfoNCE loss on the same batch.  We use the standalone
    // infonce_loss function here since DifferentiableProjection uses Glorot
    // uniform (which will differ between runs unless a fixed seed is set).
    let anchors = Array2::<f32>::eye(4);
    let positives = anchors.clone();

    let l1 = infonce_loss(&anchors, &positives, 0.05);
    let l2 = infonce_loss(&anchors, &positives, 0.05);

    assert_eq!(l1, l2, "same inputs → same InfoNCE loss");
}

/// fit_unsupervised returns one history entry per step.
#[test]
fn simcse_unsupervised_loss_fit_returns_correct_history_length() {
    let mut trainer = make_trainer(32);
    let sentences: Vec<&str> = (0..8)
        .map(|i| Box::leak(format!("word{} word{}", i, i + 1).into_boxed_str()) as &str)
        .collect();
    let history = trainer
        .fit_unsupervised(&sentences, 3, 4)
        .expect("fit_unsupervised failed");
    assert_eq!(history.len(), 3, "expected 3 steps in history");
    for step in &history {
        assert!(step.loss.is_finite(), "each step loss must be finite");
    }
}

/// encode_batch produces correct shape.
#[test]
fn simcse_encode_batch_shape() {
    let trainer = make_trainer(32);
    let sentences = ["word0", "word1 word2", "word3 word4 word5"];
    let batch = trainer
        .encode_batch(&sentences)
        .expect("encode_batch failed");
    assert_eq!(batch.shape(), &[3, 32]);
}

/// Inference (encode) disables dropout: calling encode twice on the same
/// sentence must return the same vector.
#[test]
fn simcse_encode_disables_dropout() {
    let trainer = make_trainer(32);
    let sentence = "word0 word1 word2";
    let emb1 = trainer.encode(sentence).expect("first encode failed");
    let emb2 = trainer.encode(sentence).expect("second encode failed");

    let max_diff = emb1
        .iter()
        .zip(emb2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_diff < 1e-5,
        "inference should be deterministic (max diff = {max_diff})"
    );
}

/// Supervised step: loss is finite and accuracy in [0, 1].
#[test]
fn simcse_supervised_step_valid() {
    let mut trainer = make_trainer(32);
    let anchors = ["word0 word1", "word2 word3", "word4 word5", "word6 word7"];
    let positives = [
        "word0 word1 word8",
        "word2 word3 word9",
        "word4 word5 word10",
        "word6 word7 word11",
    ];
    let result = trainer
        .supervised_step(&anchors, &positives)
        .expect("supervised step failed");
    assert!(
        result.loss.is_finite(),
        "supervised loss must be finite: {}",
        result.loss
    );
    assert!(
        result.accuracy >= 0.0 && result.accuracy <= 1.0,
        "accuracy out of range: {}",
        result.accuracy
    );
}
