//! Integration tests for scirs2-text neural_nlp module.

use scirs2_text::neural_nlp::{
    AttentionVisualization, BertClassifier, BertClassifierConfig, NerTag, NeuralNer,
    NeuralNerConfig, TransformerEncoderConfig, TransformerTextEncoder,
};

// ─── TransformerTextEncoder ───────────────────────────────────────────────────

#[test]
fn transformer_encoder_output_shape_correct() {
    let config = TransformerEncoderConfig {
        vocab_size: 50,
        hidden_size: 16,
        num_heads: 2,
        num_layers: 1,
        max_seq_len: 32,
        dropout: 0.0,
        seed: 1,
    };
    let encoder = TransformerTextEncoder::new(config).expect("encoder construction");
    let tokens: Vec<usize> = vec![0, 1, 2, 3, 4];
    let out = encoder.encode_tokens(&tokens).expect("encode_tokens");
    assert_eq!(
        out.shape(),
        &[5, 16],
        "output shape must be [seq_len, hidden_size]"
    );
}

#[test]
fn transformer_encode_sentence_unit_norm_after_normalize() {
    use scirs2_core::ndarray::ArrayView1;

    let config = TransformerEncoderConfig {
        vocab_size: 50,
        hidden_size: 16,
        num_heads: 2,
        num_layers: 1,
        max_seq_len: 32,
        dropout: 0.0,
        seed: 42,
    };
    let encoder = TransformerTextEncoder::new(config).expect("encoder construction");
    let tokens: Vec<usize> = vec![1, 2, 3];
    let sentence_emb = encoder.encode_sentence(&tokens).expect("encode_sentence");

    // Normalise manually and verify unit norm
    let norm: f32 = sentence_emb.iter().map(|&v| v * v).sum::<f32>().sqrt();
    assert!(norm > 0.0, "Norm must be positive");
    let normalized = sentence_emb.mapv(|v| v / norm);
    let unit_norm: f32 = normalized.iter().map(|&v| v * v).sum::<f32>().sqrt();
    assert!(
        (unit_norm - 1.0).abs() < 1e-5,
        "Normalised vector should have unit norm, got {unit_norm}"
    );
}

#[test]
fn transformer_encode_with_attention_returns_correct_shapes() {
    let config = TransformerEncoderConfig {
        vocab_size: 20,
        hidden_size: 8,
        num_heads: 2,
        num_layers: 2,
        max_seq_len: 16,
        dropout: 0.0,
        seed: 99,
    };
    let encoder = TransformerTextEncoder::new(config).expect("encoder construction");
    let tokens: Vec<usize> = vec![0, 1, 2];
    let (emb, attn_weights) = encoder
        .encode_with_attention(&tokens)
        .expect("encode_with_attention");

    assert_eq!(emb.shape(), &[3, 8]);
    assert_eq!(
        attn_weights.len(),
        2,
        "should have n_layers attention tensors"
    );
    // Each tensor: [n_heads, seq, seq]
    for w in &attn_weights {
        assert_eq!(w.shape(), &[2, 3, 3]);
    }
}

// ─── AttentionVisualization ───────────────────────────────────────────────────

#[test]
fn attention_viz_heatmap_rows_sum_to_one() {
    let config = TransformerEncoderConfig {
        vocab_size: 20,
        hidden_size: 8,
        num_heads: 2,
        num_layers: 2,
        max_seq_len: 16,
        dropout: 0.0,
        seed: 7,
    };
    let encoder = TransformerTextEncoder::new(config).expect("encoder");
    let tokens: Vec<usize> = vec![0, 1, 2, 3];
    let (_, attn_weights) = encoder.encode_with_attention(&tokens).expect("attn");
    let token_strs: Vec<String> = tokens.iter().map(|i| format!("tok{i}")).collect();

    let viz = AttentionVisualization::from_attention_weights(token_strs, &attn_weights);

    // Every individual heatmap's rows should sum to 1 (softmax output)
    for hm in &viz.heatmaps {
        let (rows, cols) = hm.weights.dim();
        for r in 0..rows {
            let row_sum: f32 = (0..cols).map(|c| hm.weights[[r, c]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "Row {r} of layer={} head={} sums to {row_sum}, expected 1.0",
                hm.layer_idx,
                hm.head_idx
            );
        }
    }
}

#[test]
fn attention_viz_top_attended_returns_k_tokens() {
    let config = TransformerEncoderConfig {
        vocab_size: 20,
        hidden_size: 8,
        num_heads: 2,
        num_layers: 1,
        max_seq_len: 16,
        dropout: 0.0,
        seed: 5,
    };
    let encoder = TransformerTextEncoder::new(config).expect("encoder");
    let tokens: Vec<usize> = vec![0, 1, 2, 3, 4];
    let (_, attn_weights) = encoder.encode_with_attention(&tokens).expect("attn");
    let token_strs: Vec<String> = tokens.iter().map(|i| format!("t{i}")).collect();

    let viz = AttentionVisualization::from_attention_weights(token_strs, &attn_weights);

    let k = 3;
    let top = viz.top_attended_tokens(k);
    assert_eq!(top.len(), k, "should return exactly k tokens");

    // Ensure results are sorted descending by attention weight
    for w in top.windows(2) {
        assert!(
            w[0].1 >= w[1].1,
            "top_attended_tokens must be sorted descending"
        );
    }
}

#[test]
fn attention_viz_to_flat_vec_has_correct_length() {
    let config = TransformerEncoderConfig {
        vocab_size: 10,
        hidden_size: 8,
        num_heads: 2,
        num_layers: 2,
        max_seq_len: 8,
        dropout: 0.0,
        seed: 3,
    };
    let encoder = TransformerTextEncoder::new(config).expect("encoder");
    let tokens: Vec<usize> = vec![0, 1, 2];
    let (_, attn_weights) = encoder.encode_with_attention(&tokens).expect("attn");
    let token_strs: Vec<String> = vec!["a".into(), "b".into(), "c".into()];

    let viz = AttentionVisualization::from_attention_weights(token_strs, &attn_weights);
    // n_layers=2, n_heads=2, seq=3 → 2*2*3*3 = 36
    let flat = viz.to_flat_vec();
    assert_eq!(
        flat.len(),
        2 * 2 * 3 * 3,
        "flat vec length = n_layers * n_heads * seq * seq"
    );
}

// ─── BertClassifier ───────────────────────────────────────────────────────────

#[test]
fn bert_classifier_fine_tuning_reduces_loss() {
    let enc_config = TransformerEncoderConfig {
        vocab_size: 20,
        hidden_size: 16,
        num_heads: 2,
        num_layers: 1,
        max_seq_len: 16,
        dropout: 0.0,
        seed: 11,
    };
    let cls_config = BertClassifierConfig {
        encoder_config: enc_config,
        num_classes: 2,
        dropout: 0.0,
        learning_rate: 0.1,
        epochs: 20,
        batch_size: 4,
        seed: 22,
    };
    let mut clf = BertClassifier::new(cls_config).expect("bert classifier");

    // Simple data: tokens 0-4 → class 0, tokens 5-9 → class 1
    let data: Vec<(Vec<usize>, usize)> = vec![
        (vec![0, 1, 2], 0),
        (vec![0, 1, 3], 0),
        (vec![0, 2, 4], 0),
        (vec![0, 1, 2, 3], 0),
        (vec![5, 6, 7], 1),
        (vec![5, 6, 8], 1),
        (vec![5, 7, 9], 1),
        (vec![5, 6, 7, 8], 1),
    ];

    let losses = clf.fine_tune(&data).expect("fine_tune");
    assert_eq!(losses.len(), 20, "should return one loss per epoch");

    // Loss should decrease over training
    let first_loss = losses[0];
    let last_loss = losses[losses.len() - 1];
    assert!(
        last_loss < first_loss,
        "Loss should decrease after fine-tuning; got {first_loss} → {last_loss}"
    );
}

#[test]
fn bert_classifier_predicts_correct_class_on_linearly_separable() {
    // Train on a clearly separable problem with enough iterations
    let enc_config = TransformerEncoderConfig {
        vocab_size: 20,
        hidden_size: 16,
        num_heads: 2,
        num_layers: 1,
        max_seq_len: 16,
        dropout: 0.0,
        seed: 55,
    };
    let cls_config = BertClassifierConfig {
        encoder_config: enc_config,
        num_classes: 2,
        dropout: 0.0,
        learning_rate: 0.2,
        epochs: 100,
        batch_size: 4,
        seed: 66,
    };
    let mut clf = BertClassifier::new(cls_config).expect("bert classifier");

    let train_data: Vec<(Vec<usize>, usize)> = vec![
        (vec![0, 1, 2], 0),
        (vec![0, 1, 3], 0),
        (vec![0, 2, 4], 0),
        (vec![5, 6, 7], 1),
        (vec![5, 6, 8], 1),
        (vec![5, 7, 9], 1),
    ];

    clf.fine_tune(&train_data).expect("fine_tune");

    // Evaluate on training set: should have reasonable accuracy after 100 epochs
    let acc = clf.accuracy(&train_data).expect("accuracy");
    // With 100 epochs on 6 samples, accuracy should be high
    assert!(
        acc >= 0.5,
        "accuracy on training set should be at least 50%, got {acc}"
    );
}

// ─── NeuralNer ────────────────────────────────────────────────────────────────

#[test]
fn neural_ner_output_length_matches_input() {
    let config = NeuralNerConfig {
        encoder_config: TransformerEncoderConfig {
            vocab_size: 20,
            hidden_size: 16,
            num_heads: 2,
            num_layers: 1,
            max_seq_len: 16,
            dropout: 0.0,
            seed: 33,
        },
        n_tags: NerTag::N,
        learning_rate: 0.01,
        epochs: 1,
        seed: 44,
    };
    let ner = NeuralNer::new(config).expect("NeuralNer::new");
    let tokens = vec![0usize, 1, 2, 3, 4];
    let tags = ner.predict(&tokens).expect("predict");
    assert_eq!(
        tags.len(),
        tokens.len(),
        "output tag sequence length must equal input length"
    );
}

#[test]
fn neural_ner_tags_are_valid() {
    let config = NeuralNerConfig {
        encoder_config: TransformerEncoderConfig {
            vocab_size: 20,
            hidden_size: 8,
            num_heads: 2,
            num_layers: 1,
            max_seq_len: 16,
            dropout: 0.0,
            seed: 1001,
        },
        n_tags: NerTag::N,
        learning_rate: 0.01,
        epochs: 1,
        seed: 2002,
    };
    let ner = NeuralNer::new(config).expect("NeuralNer::new");
    let tokens = vec![0usize, 5, 10, 15];
    let tags = ner.predict(&tokens).expect("predict");

    // Each tag must round-trip through from_idx / to_idx
    for tag in &tags {
        let idx = tag.to_idx();
        let roundtrip = NerTag::from_idx(idx).expect("round-trip");
        assert_eq!(*tag, roundtrip, "tag round-trip failed for idx {idx}");
    }
}

#[test]
fn neural_ner_fit_reduces_loss() {
    let config = NeuralNerConfig {
        encoder_config: TransformerEncoderConfig {
            vocab_size: 20,
            hidden_size: 16,
            num_heads: 2,
            num_layers: 1,
            max_seq_len: 16,
            dropout: 0.0,
            seed: 77,
        },
        n_tags: NerTag::N,
        learning_rate: 0.1,
        epochs: 20,
        seed: 88,
    };
    let mut ner = NeuralNer::new(config).expect("NeuralNer::new");

    // Small NER training set: token sequences with per-token BIO tags (as indices)
    let o = NerTag::O.to_idx();
    let b_per = NerTag::BPer.to_idx();
    let i_per = NerTag::IPer.to_idx();
    let b_loc = NerTag::BLoc.to_idx();

    let data: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![0, 1, 2, 3], vec![o, b_per, i_per, o]),
        (vec![4, 5, 6], vec![b_loc, o, o]),
        (vec![0, 1, 4, 5], vec![o, b_per, b_loc, o]),
        (vec![2, 3, 6, 7], vec![o, o, o, b_per]),
    ];

    let losses = ner.fit(&data).expect("fit");
    assert_eq!(losses.len(), 20, "should return one loss per epoch");
    assert!(losses[0] > 0.0, "initial loss should be positive");

    let first = losses[0];
    let last = losses[losses.len() - 1];
    assert!(
        last < first,
        "loss should decrease over training; got {first} → {last}"
    );
}
