//! Integration tests for perplexity evaluation (`evaluation/perplexity.rs`).

use scirs2_text::evaluation::perplexity::{
    perplexity_evaluate, LanguageModelLike, PerplexityReport,
};

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

/// A language model that assigns uniform probability 1/V to every token.
struct UniformModel {
    vocab: usize,
}

impl LanguageModelLike for UniformModel {
    fn log_prob_sequence(&self, tokens: &[&str]) -> Option<f64> {
        if tokens.is_empty() {
            return None;
        }
        Some(tokens.len() as f64 * -(self.vocab as f64).ln())
    }

    fn vocabulary_size(&self) -> usize {
        self.vocab
    }
}

/// A perfect predictor that always assigns probability 1.0 (log_prob = 0).
struct PerfectModel;

impl LanguageModelLike for PerfectModel {
    fn log_prob_sequence(&self, tokens: &[&str]) -> Option<f64> {
        if tokens.is_empty() {
            return None;
        }
        Some(0.0_f64)
    }

    fn vocabulary_size(&self) -> usize {
        1
    }
}

/// A model with per-token deterministic probabilities.
struct FixedProbModel {
    /// log-probability assigned to every single token call.
    token_log_prob: f64,
}

impl LanguageModelLike for FixedProbModel {
    fn log_prob_sequence(&self, tokens: &[&str]) -> Option<f64> {
        if tokens.is_empty() {
            return None;
        }
        Some(tokens.len() as f64 * self.token_log_prob)
    }

    fn vocabulary_size(&self) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn perplexity_uniform_model_equals_vocab_size() {
    let model = UniformModel { vocab: 100 };
    let corpus = vec![vec!["a", "b", "c", "d", "e"]];
    let report = perplexity_evaluate(&model, &corpus).expect("evaluate");
    // PP = exp(-1/N * N * ln(1/V)) = exp(ln(V)) = V = 100
    assert!(
        (report.corpus_perplexity - 100.0).abs() < 1e-6,
        "expected 100.0, got {}",
        report.corpus_perplexity
    );
}

#[test]
fn perplexity_of_perfect_predictor_is_one() {
    let model = PerfectModel;
    let corpus = vec![vec!["hello", "world", "foo"]];
    let report = perplexity_evaluate(&model, &corpus).expect("evaluate");
    assert!(
        (report.corpus_perplexity - 1.0).abs() < 1e-9,
        "expected 1.0, got {}",
        report.corpus_perplexity
    );
}

#[test]
fn perplexity_corpus_aggregates_token_log_probs() {
    let vocab = 10usize;
    let model = UniformModel { vocab };
    // Two sentences: 3 and 2 tokens → 5 total tokens
    let corpus = vec![vec!["a", "b", "c"], vec!["d", "e"]];
    let report = perplexity_evaluate(&model, &corpus).expect("evaluate");

    assert_eq!(report.total_tokens, 5, "expected 5 total tokens");

    // total_log_prob = 5 * ln(1/10)
    let expected_lp = 5.0 * -(vocab as f64).ln();
    assert!(
        (report.total_log_prob - expected_lp).abs() < 1e-9,
        "expected total_log_prob {expected_lp}, got {}",
        report.total_log_prob
    );
}

#[test]
fn perplexity_empty_corpus_returns_error() {
    let model = UniformModel { vocab: 10 };
    let result = perplexity_evaluate(&model, &[]);
    assert!(result.is_err(), "expected Err for empty corpus");
}

#[test]
fn perplexity_per_sentence_are_positive() {
    let model = UniformModel { vocab: 5 };
    let corpus = vec![vec!["a"], vec!["b", "c"]];
    let report = perplexity_evaluate(&model, &corpus).expect("evaluate");
    for (i, &ppl) in report.per_sentence_perplexity.iter().enumerate() {
        assert!(
            ppl > 0.0 && ppl.is_finite(),
            "per-sentence ppl[{i}] = {ppl}"
        );
    }
}

#[test]
fn perplexity_all_empty_sentences_returns_error() {
    let model = UniformModel { vocab: 5 };
    // All sentences are empty → total_tokens = 0
    let corpus: Vec<Vec<&str>> = vec![vec![], vec![]];
    let result = perplexity_evaluate(&model, &corpus);
    assert!(result.is_err(), "expected Err when no tokens are present");
}

#[test]
fn perplexity_per_sentence_count_equals_corpus_length() {
    let model = UniformModel { vocab: 4 };
    let corpus = vec![vec!["a", "b"], vec!["c"], vec!["d", "e", "f"]];
    let report = perplexity_evaluate(&model, &corpus).expect("evaluate");
    assert_eq!(
        report.per_sentence_perplexity.len(),
        3,
        "per_sentence_perplexity length must match corpus length"
    );
}

#[test]
fn perplexity_higher_vocab_gives_higher_ppl() {
    let corpus = vec![vec!["a", "b", "c"]];
    let report_small =
        perplexity_evaluate(&UniformModel { vocab: 10 }, &corpus).expect("evaluate small vocab");
    let report_large =
        perplexity_evaluate(&UniformModel { vocab: 1000 }, &corpus).expect("evaluate large vocab");
    assert!(
        report_large.corpus_perplexity > report_small.corpus_perplexity,
        "larger vocab should yield higher PPL: {} vs {}",
        report_large.corpus_perplexity,
        report_small.corpus_perplexity
    );
}

#[test]
fn perplexity_fixed_prob_model_matches_formula() {
    // log p(token) = -2.0 for every token
    let log_p = -2.0_f64;
    let model = FixedProbModel {
        token_log_prob: log_p,
    };
    let corpus = vec![vec!["a", "b", "c"]]; // 3 tokens
    let report = perplexity_evaluate(&model, &corpus).expect("evaluate");
    // PPL = exp(-total_log_prob / n) = exp(-3*(-2)/3) = exp(2)
    let expected = 2.0_f64.exp();
    assert!(
        (report.corpus_perplexity - expected).abs() < 1e-9,
        "expected {expected}, got {}",
        report.corpus_perplexity
    );
}

#[test]
fn perplexity_empty_sentence_yields_nan_in_per_sentence() {
    let model = UniformModel { vocab: 5 };
    // Mix of empty and non-empty
    let corpus: Vec<Vec<&str>> = vec![vec![], vec!["a", "b"]];
    let report = perplexity_evaluate(&model, &corpus).expect("evaluate");
    assert!(
        report.per_sentence_perplexity[0].is_nan(),
        "empty sentence should produce NaN per-sentence PPL"
    );
    assert!(
        report.per_sentence_perplexity[1].is_finite() && report.per_sentence_perplexity[1] > 0.0,
        "non-empty sentence PPL should be finite positive"
    );
}

#[test]
fn perplexity_with_ngram_model_from_language_models() {
    use scirs2_text::language_models::NgramLM;

    let corpus_str = vec![
        "the cat sat on the mat"
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        "the dog ran over the hill"
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>(),
    ];

    let lm = NgramLM::train(2, &corpus_str).expect("train NgramLM");

    // Evaluate on a held-out sentence via the LanguageModelLike trait.
    let test_corpus = vec![vec!["the", "cat", "ran"]];
    let report = perplexity_evaluate(&lm, &test_corpus).expect("perplexity evaluate");
    assert!(
        report.corpus_perplexity > 0.0 && report.corpus_perplexity.is_finite(),
        "PPL = {}",
        report.corpus_perplexity
    );
}

#[test]
fn perplexity_with_ngram_model_from_language_model() {
    use scirs2_text::language_model::{NgramModel, SmoothingMethod};

    let mut model = NgramModel::new(2, SmoothingMethod::Laplace);
    model
        .train(&["the cat sat on the mat", "the dog ran quickly"])
        .expect("train");

    let test_corpus = vec![vec!["the", "cat", "ran"]];
    let report = perplexity_evaluate(&model, &test_corpus).expect("perplexity evaluate");
    assert!(
        report.corpus_perplexity > 0.0 && report.corpus_perplexity.is_finite(),
        "PPL = {}",
        report.corpus_perplexity
    );
}
