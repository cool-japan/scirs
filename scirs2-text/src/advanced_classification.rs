//! Advanced text classification and feature extraction.
//!
//! Provides:
//! - [`NaiveBayesClassifier`]: Multinomial Naive Bayes with Laplace smoothing.
//! - [`FastTextClassifier`]: Averaged word-vector classifier inspired by fastText.
//! - [`CountVectorizer`]: N-gram count matrix builder.
//! - [`TfidfTransformer`]: TF-IDF weighting from count matrices.

use std::collections::HashMap;

use crate::error::{Result, TextError};

// ─────────────────────────────────────────────────────────────────────────────
// NaiveBayesClassifier
// ─────────────────────────────────────────────────────────────────────────────

/// Multinomial Naive Bayes text classifier with Laplace smoothing.
///
/// Uses a bag-of-words representation; each token in a document contributes
/// to the likelihood estimate.
#[derive(Debug, Clone, Default)]
pub struct NaiveBayesClassifier {
    /// log P(class) for each class
    class_log_priors: Vec<f64>,
    /// log P(word | class): indexed as [class_idx][word_idx]
    log_likelihoods: Vec<Vec<f64>>,
    /// Ordered class names
    classes: Vec<String>,
    /// word → index mapping built during fit
    vocabulary: HashMap<String, usize>,
    /// `true` after `fit()` has been called
    fitted: bool,
}

impl NaiveBayesClassifier {
    /// Create an unfitted classifier.
    pub fn new() -> Self {
        Self::default()
    }

    // ── Internal helpers ──────────────────────────────────────────────

    /// Tokenise `text` into words (lower-case, alpha-only).
    fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect()
    }

    /// Convert `text` to a word-count vector aligned with `vocabulary`.
    fn text_to_counts(&self, text: &str) -> Vec<f64> {
        let mut counts = vec![0.0f64; self.vocabulary.len()];
        for word in Self::tokenize(text) {
            if let Some(&idx) = self.vocabulary.get(&word) {
                counts[idx] += 1.0;
            }
        }
        counts
    }

    // ── Public API ────────────────────────────────────────────────────

    /// Train on `(text, label)` pairs.
    ///
    /// `alpha` is the Laplace smoothing factor (typical: 1.0).
    pub fn fit(&mut self, corpus: &[(String, String)], alpha: f64) -> Result<()> {
        if corpus.is_empty() {
            return Err(TextError::InvalidInput("corpus is empty".to_string()));
        }
        if alpha <= 0.0 {
            return Err(TextError::InvalidInput(
                "smoothing parameter alpha must be > 0".to_string(),
            ));
        }

        // Collect unique classes
        let mut class_set: Vec<String> = corpus
            .iter()
            .map(|(_, label)| label.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        class_set.sort();
        self.classes = class_set;
        let n_classes = self.classes.len();
        let class_to_id: HashMap<String, usize> = self
            .classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();

        // Build vocabulary
        let mut vocab_set: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (text, _) in corpus {
            for word in Self::tokenize(text) {
                vocab_set.insert(word);
            }
        }
        let mut vocab_sorted: Vec<String> = vocab_set.into_iter().collect();
        vocab_sorted.sort();
        self.vocabulary = vocab_sorted
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();
        let v = self.vocabulary.len();

        // Count per class
        let mut class_counts = vec![0usize; n_classes];
        let mut word_counts_per_class: Vec<Vec<f64>> = vec![vec![0.0; v]; n_classes];

        for (text, label) in corpus {
            let ci = class_to_id[label];
            class_counts[ci] += 1;
            for word in Self::tokenize(text) {
                if let Some(&wi) = self.vocabulary.get(&word) {
                    word_counts_per_class[ci][wi] += 1.0;
                }
            }
        }

        let total_docs = corpus.len() as f64;
        self.class_log_priors = class_counts
            .iter()
            .map(|&c| (c as f64 / total_docs).ln())
            .collect();

        // Compute log likelihoods with Laplace smoothing
        self.log_likelihoods = word_counts_per_class
            .iter()
            .map(|counts| {
                let total: f64 = counts.iter().sum::<f64>() + alpha * v as f64;
                counts.iter().map(|&c| ((c + alpha) / total).ln()).collect()
            })
            .collect();

        self.fitted = true;
        Ok(())
    }

    /// Compute log-posterior scores for each class.
    fn log_scores(&self, text: &str) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(TextError::ModelNotFitted(
                "NaiveBayesClassifier is not fitted".to_string(),
            ));
        }
        let counts = self.text_to_counts(text);
        let scores: Vec<f64> = self
            .class_log_priors
            .iter()
            .zip(self.log_likelihoods.iter())
            .map(|(&prior, likelihoods)| {
                let ll: f64 = counts
                    .iter()
                    .zip(likelihoods.iter())
                    .map(|(&c, &lp)| c * lp)
                    .sum();
                prior + ll
            })
            .collect();
        Ok(scores)
    }

    /// Predict the most likely class label for `text`.
    pub fn predict(&self, text: &str) -> Result<Option<String>> {
        let scores = self.log_scores(text)?;
        let best = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| self.classes[i].clone());
        Ok(best)
    }

    /// Predict posterior probabilities (softmax of log-scores) per class.
    pub fn predict_proba(&self, text: &str) -> Result<Vec<(String, f64)>> {
        let log_scores = self.log_scores(text)?;
        // Softmax
        let max_s = log_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = log_scores.iter().map(|&s| (s - max_s).exp()).collect();
        let total: f64 = exps.iter().sum();
        Ok(self
            .classes
            .iter()
            .zip(exps.iter())
            .map(|(cls, &e)| (cls.clone(), if total == 0.0 { 0.0 } else { e / total }))
            .collect())
    }

    /// Batch predict over multiple texts.
    pub fn predict_batch(&self, texts: &[String]) -> Result<Vec<Option<String>>> {
        texts.iter().map(|t| self.predict(t)).collect()
    }

    /// Compute accuracy on a labelled test set.
    pub fn accuracy(&self, test_set: &[(String, String)]) -> Result<f64> {
        if test_set.is_empty() {
            return Ok(0.0);
        }
        let mut correct = 0usize;
        for (text, gold) in test_set {
            if let Ok(Some(pred)) = self.predict(text) {
                if &pred == gold {
                    correct += 1;
                }
            }
        }
        Ok(correct as f64 / test_set.len() as f64)
    }

    /// Return ordered class names.
    pub fn class_names(&self) -> &[String] {
        &self.classes
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FastTextClassifier
// ─────────────────────────────────────────────────────────────────────────────

/// FastText-inspired averaged word-vector text classifier.
///
/// Each document is represented as the average of its per-word embeddings.
/// A linear layer maps the averaged vector to class logits.  Training uses
/// SGD with Hogwild-style updates.
#[derive(Debug, Clone)]
pub struct FastTextClassifier {
    n_classes: usize,
    classes: Vec<String>,
    word_vectors: HashMap<String, Vec<f32>>,
    /// Weight matrix [dim × n_classes]
    weights: Vec<Vec<f32>>,
    /// Bias vector [n_classes]
    bias: Vec<f32>,
    dim: usize,
    fitted: bool,
}

impl FastTextClassifier {
    /// Create a new, unfitted classifier.
    ///
    /// * `n_classes` – number of output classes.
    /// * `dim`       – word-embedding dimension.
    /// * `classes`   – ordered class name list.
    pub fn new(n_classes: usize, dim: usize, classes: Vec<String>) -> Self {
        assert_eq!(
            classes.len(),
            n_classes,
            "classes.len() must equal n_classes"
        );
        FastTextClassifier {
            n_classes,
            classes,
            word_vectors: HashMap::new(),
            weights: vec![vec![0.0f32; n_classes]; dim],
            bias: vec![0.0f32; n_classes],
            dim,
            fitted: false,
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────

    /// Initialise a new word vector with small random values via a deterministic
    /// hash-based scheme (avoids external RNG dependency).
    fn init_word_vec(word: &str, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        for (i, c) in word.bytes().enumerate() {
            let idx = i % dim;
            v[idx] += (c as f32 - 64.0) / 128.0;
        }
        // Normalise
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        v
    }

    /// Compute the mean embedding for a token list.
    fn mean_embedding(&self, tokens: &[String]) -> Vec<f32> {
        let mut sum = vec![0.0f32; self.dim];
        let mut count = 0usize;
        for tok in tokens {
            if let Some(vec) = self.word_vectors.get(tok.as_str()) {
                for (s, &v) in sum.iter_mut().zip(vec.iter()) {
                    *s += v;
                }
                count += 1;
            }
        }
        if count > 0 {
            sum.iter_mut().for_each(|s| *s /= count as f32);
        }
        sum
    }

    /// Linear forward: z[k] = sum_d(embedding[d] * weights[d][k]) + bias[k]
    fn forward(&self, embedding: &[f32]) -> Vec<f32> {
        let mut logits = self.bias.clone();
        for (d, &e) in embedding.iter().enumerate() {
            for k in 0..self.n_classes {
                logits[k] += e * self.weights[d][k];
            }
        }
        logits
    }

    /// Softmax in-place.
    fn softmax(logits: &mut [f32]) {
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        logits.iter_mut().for_each(|x| *x = (*x - max_l).exp());
        let sum: f32 = logits.iter().sum();
        if sum > 0.0 {
            logits.iter_mut().for_each(|x| *x /= sum);
        }
    }

    // ── Public API ────────────────────────────────────────────────────

    /// Train the classifier.
    ///
    /// `corpus` is `(tokens, class_id)` pairs.  `lr` is the learning rate.
    pub fn fit(&mut self, corpus: &[(Vec<String>, usize)], n_epochs: usize, lr: f32) -> Result<()> {
        if corpus.is_empty() {
            return Err(TextError::InvalidInput("corpus is empty".to_string()));
        }

        // Ensure all word vectors exist
        for (tokens, _) in corpus {
            for tok in tokens {
                self.word_vectors
                    .entry(tok.clone())
                    .or_insert_with(|| Self::init_word_vec(tok, self.dim));
            }
        }

        for _epoch in 0..n_epochs {
            for (tokens, gold_class) in corpus {
                let gold_class = *gold_class;
                if gold_class >= self.n_classes {
                    continue;
                }
                let emb = self.mean_embedding(tokens);
                let mut probs = self.forward(&emb);
                Self::softmax(&mut probs);

                // Cross-entropy gradient: (probs - one_hot)
                let mut grad = probs.clone();
                grad[gold_class] -= 1.0;

                // Update weights and bias
                for d in 0..self.dim {
                    for k in 0..self.n_classes {
                        self.weights[d][k] -= lr * grad[k] * emb[d];
                    }
                }
                for k in 0..self.n_classes {
                    self.bias[k] -= lr * grad[k];
                }
            }
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict the class ID for a token sequence.
    pub fn predict(&self, tokens: &[String]) -> usize {
        let probs = self.predict_proba(tokens);
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Predict class probability distribution.
    pub fn predict_proba(&self, tokens: &[String]) -> Vec<f32> {
        let emb = self.mean_embedding(tokens);
        let mut logits = self.forward(&emb);
        Self::softmax(&mut logits);
        logits
    }

    /// Return the ordered class names.
    pub fn class_names(&self) -> &[String] {
        &self.classes
    }

    /// `true` if the model has been trained.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CountVectorizer
// ─────────────────────────────────────────────────────────────────────────────

/// Count-based document-feature matrix builder with N-gram support.
#[derive(Debug, Clone)]
pub struct CountVectorizer {
    vocabulary: HashMap<String, usize>,
    max_features: Option<usize>,
    min_df: usize,
    max_df_ratio: f64,
    ngram_range: (usize, usize),
    fitted: bool,
}

impl Default for CountVectorizer {
    fn default() -> Self {
        CountVectorizer {
            vocabulary: HashMap::new(),
            max_features: None,
            min_df: 1,
            max_df_ratio: 1.0,
            ngram_range: (1, 1),
            fitted: false,
        }
    }
}

impl CountVectorizer {
    /// Create a new `CountVectorizer` with default settings (unigrams only).
    pub fn new() -> Self {
        Self::default()
    }

    /// Limit the vocabulary to the `n` most frequent features.
    pub fn with_max_features(mut self, n: usize) -> Self {
        self.max_features = Some(n);
        self
    }

    /// Set N-gram range `(min_n, max_n)`.
    pub fn with_ngram_range(mut self, min: usize, max: usize) -> Self {
        self.ngram_range = (min, max);
        self
    }

    /// Set minimum document frequency (number of documents a token must appear in).
    pub fn with_min_df(mut self, min_df: usize) -> Self {
        self.min_df = min_df;
        self
    }

    /// Set maximum document frequency as a fraction of the corpus (0.0–1.0).
    pub fn with_max_df_ratio(mut self, ratio: f64) -> Self {
        self.max_df_ratio = ratio;
        self
    }

    // ── Internal helpers ──────────────────────────────────────────────

    /// Generate N-grams from a token list according to `ngram_range`.
    fn ngrams(&self, tokens: &[String]) -> Vec<String> {
        let (min_n, max_n) = self.ngram_range;
        let mut grams = Vec::new();
        for n in min_n..=max_n {
            for window in tokens.windows(n) {
                grams.push(window.join(" "));
            }
        }
        grams
    }

    /// Tokenise `text` into lower-case alphanumeric words.
    fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect()
    }

    // ── Public API ────────────────────────────────────────────────────

    /// Fit the vocabulary from `corpus`.
    pub fn fit(&mut self, corpus: &[String]) -> Result<()> {
        if corpus.is_empty() {
            return Err(TextError::InvalidInput("corpus is empty".to_string()));
        }
        let n_docs = corpus.len();

        // Count document frequencies
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut term_freq: HashMap<String, usize> = HashMap::new();

        for doc in corpus {
            let tokens = Self::tokenize(doc);
            let grams = self.ngrams(&tokens);
            let unique: std::collections::HashSet<String> = grams.iter().cloned().collect();
            for gram in unique {
                *df.entry(gram.clone()).or_insert(0) += 1;
                *term_freq.entry(gram).or_insert(0) += 1;
            }
        }

        // Filter by df thresholds
        let max_df_count = (self.max_df_ratio * n_docs as f64).ceil() as usize;
        let mut candidates: Vec<(String, usize)> = df
            .into_iter()
            .filter(|(_, count)| *count >= self.min_df && *count <= max_df_count)
            .collect();

        // Sort by total term frequency descending, then alphabetically
        candidates.sort_by(|a, b| {
            let fa = term_freq.get(&a.0).copied().unwrap_or(0);
            let fb = term_freq.get(&b.0).copied().unwrap_or(0);
            fb.cmp(&fa).then_with(|| a.0.cmp(&b.0))
        });

        // Apply max_features limit
        if let Some(max_f) = self.max_features {
            candidates.truncate(max_f);
        }

        // Build vocabulary
        self.vocabulary = candidates
            .into_iter()
            .enumerate()
            .map(|(i, (gram, _))| (gram, i))
            .collect();

        self.fitted = true;
        Ok(())
    }

    /// Transform `texts` into a count matrix.
    pub fn transform(&self, texts: &[String]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(TextError::ModelNotFitted(
                "CountVectorizer is not fitted".to_string(),
            ));
        }
        let v = self.vocabulary.len();
        texts
            .iter()
            .map(|text| {
                let tokens = Self::tokenize(text);
                let grams = self.ngrams(&tokens);
                let mut counts = vec![0.0f64; v];
                for gram in grams {
                    if let Some(&idx) = self.vocabulary.get(&gram) {
                        counts[idx] += 1.0;
                    }
                }
                Ok(counts)
            })
            .collect()
    }

    /// Fit then transform in one step.
    pub fn fit_transform(&mut self, corpus: &[String]) -> Result<Vec<Vec<f64>>> {
        self.fit(corpus)?;
        self.transform(corpus)
    }

    /// Return the current vocabulary size.
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Borrow the vocabulary map.
    pub fn vocabulary(&self) -> &HashMap<String, usize> {
        &self.vocabulary
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TfidfTransformer
// ─────────────────────────────────────────────────────────────────────────────

/// Transforms a count matrix into a TF-IDF weighted matrix.
///
/// IDF formula (with `smooth_idf = true`):
/// `idf(t) = ln((1 + n) / (1 + df(t))) + 1`
///
/// Each row is L2-normalised after weighting.
#[derive(Debug, Clone)]
pub struct TfidfTransformer {
    /// Per-term IDF values.
    pub idf: Vec<f64>,
    /// Whether to smooth IDF by adding 1 to numerator and denominator.
    pub smooth_idf: bool,
    fitted: bool,
}

impl TfidfTransformer {
    /// Create a new transformer.  `smooth_idf = true` avoids division by zero
    /// for terms seen in all documents.
    pub fn new(smooth_idf: bool) -> Self {
        TfidfTransformer {
            idf: Vec::new(),
            smooth_idf,
            fitted: false,
        }
    }

    /// Compute IDF values from a count matrix (rows = documents, cols = terms).
    pub fn fit(&mut self, count_matrix: &[Vec<f64>]) -> Result<()> {
        if count_matrix.is_empty() {
            return Err(TextError::InvalidInput("count_matrix is empty".to_string()));
        }
        let n_docs = count_matrix.len() as f64;
        let n_features = count_matrix[0].len();

        let mut df = vec![0.0f64; n_features];
        for row in count_matrix {
            for (j, &c) in row.iter().enumerate() {
                if c > 0.0 {
                    df[j] += 1.0;
                }
            }
        }

        self.idf = if self.smooth_idf {
            df.iter()
                .map(|&d| ((1.0 + n_docs) / (1.0 + d)).ln() + 1.0)
                .collect()
        } else {
            df.iter()
                .map(|&d| {
                    if d == 0.0 {
                        0.0
                    } else {
                        (n_docs / d).ln() + 1.0
                    }
                })
                .collect()
        };

        self.fitted = true;
        Ok(())
    }

    /// Apply TF-IDF weighting and L2 normalisation.
    pub fn transform(&self, count_matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(TextError::ModelNotFitted(
                "TfidfTransformer is not fitted".to_string(),
            ));
        }
        count_matrix
            .iter()
            .map(|row| {
                let mut tfidf: Vec<f64> = row
                    .iter()
                    .zip(self.idf.iter())
                    .map(|(&c, &idf)| c * idf)
                    .collect();
                // L2 normalise
                let norm: f64 = tfidf.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if norm > 0.0 {
                    tfidf.iter_mut().for_each(|x| *x /= norm);
                }
                Ok(tfidf)
            })
            .collect()
    }

    /// Fit then transform in one step.
    pub fn fit_transform(&mut self, count_matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        self.fit(count_matrix)?;
        self.transform(count_matrix)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn news_corpus() -> Vec<(String, String)> {
        vec![
            ("football game soccer ball".into(), "sports".into()),
            ("basketball players team score".into(), "sports".into()),
            ("election president vote campaign".into(), "politics".into()),
            ("senate congress legislation bill".into(), "politics".into()),
            ("python rust programming language".into(), "tech".into()),
            ("software compiler code debug".into(), "tech".into()),
        ]
    }

    // ── NaiveBayesClassifier tests ───────────────────────────────────

    #[test]
    fn test_nb_fit_predict() {
        let mut nb = NaiveBayesClassifier::new();
        let corpus = news_corpus();
        nb.fit(&corpus, 1.0).expect("fit failed");
        // Should predict "sports" for a sports-like document
        let pred = nb.predict("soccer football game").expect("predict failed");
        assert!(pred.is_some());
        assert_eq!(pred.unwrap(), "sports");
    }

    #[test]
    fn test_nb_predict_proba_sums_to_one() {
        let mut nb = NaiveBayesClassifier::new();
        let corpus = news_corpus();
        nb.fit(&corpus, 1.0).expect("fit failed");
        let proba = nb.predict_proba("vote election").expect("proba failed");
        let total: f64 = proba.iter().map(|(_, p)| p).sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "probabilities should sum to 1, got {}",
            total
        );
    }

    #[test]
    fn test_nb_accuracy() {
        let mut nb = NaiveBayesClassifier::new();
        let corpus = news_corpus();
        nb.fit(&corpus, 1.0).expect("fit failed");
        let acc = nb.accuracy(&corpus).expect("accuracy failed");
        assert!(acc >= 0.5, "Expected accuracy >= 0.5, got {}", acc);
    }

    #[test]
    fn test_nb_class_names() {
        let mut nb = NaiveBayesClassifier::new();
        nb.fit(&news_corpus(), 1.0).expect("fit failed");
        let classes = nb.class_names();
        assert!(classes.contains(&"sports".to_string()));
        assert!(classes.contains(&"tech".to_string()));
    }

    #[test]
    fn test_nb_not_fitted_error() {
        let nb = NaiveBayesClassifier::new();
        let result = nb.predict("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_nb_batch_predict() {
        let mut nb = NaiveBayesClassifier::new();
        nb.fit(&news_corpus(), 1.0).expect("fit failed");
        let texts = vec!["soccer game".to_string(), "code compiler".to_string()];
        let preds = nb.predict_batch(&texts).expect("batch predict failed");
        assert_eq!(preds.len(), 2);
        assert!(preds[0].is_some());
    }

    // ── FastTextClassifier tests ─────────────────────────────────────

    #[test]
    fn test_fasttext_predict_without_training() {
        let ft = FastTextClassifier::new(2, 16, vec!["sports".to_string(), "tech".to_string()]);
        let tokens: Vec<String> = vec!["soccer".into(), "game".into()];
        let pred = ft.predict(&tokens);
        assert!(pred < 2);
    }

    #[test]
    fn test_fasttext_fit_and_predict() {
        let classes = vec!["pos".to_string(), "neg".to_string()];
        let mut ft = FastTextClassifier::new(2, 8, classes);
        let corpus = vec![
            (vec!["good".to_string(), "great".to_string()], 0usize),
            (vec!["excellent".to_string(), "wonderful".to_string()], 0),
            (vec!["bad".to_string(), "terrible".to_string()], 1),
            (vec!["awful".to_string(), "horrible".to_string()], 1),
        ];
        ft.fit(&corpus, 10, 0.1).expect("fit failed");
        assert!(ft.is_fitted());
        let probs = ft.predict_proba(&["good".to_string()]);
        assert_eq!(probs.len(), 2);
        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    // ── CountVectorizer tests ────────────────────────────────────────

    #[test]
    fn test_count_vectorizer_basic() {
        let mut cv = CountVectorizer::new();
        let corpus: Vec<String> = vec![
            "hello world".to_string(),
            "hello rust".to_string(),
            "world rust".to_string(),
        ];
        let matrix = cv.fit_transform(&corpus).expect("fit_transform failed");
        assert_eq!(matrix.len(), 3);
        assert!(cv.vocabulary_size() > 0);
    }

    #[test]
    fn test_count_vectorizer_ngram() {
        let mut cv = CountVectorizer::new().with_ngram_range(1, 2);
        let corpus: Vec<String> = vec!["the quick fox".to_string(), "the lazy dog".to_string()];
        cv.fit(&corpus).expect("fit failed");
        // Should have unigrams + bigrams
        assert!(cv.vocabulary_size() > 3);
    }

    #[test]
    fn test_count_vectorizer_max_features() {
        let mut cv = CountVectorizer::new().with_max_features(2);
        let corpus: Vec<String> = vec!["a b c d e f".to_string(), "a b c d e f".to_string()];
        cv.fit(&corpus).expect("fit failed");
        assert_eq!(cv.vocabulary_size(), 2);
    }

    #[test]
    fn test_count_vectorizer_not_fitted_error() {
        let cv = CountVectorizer::new();
        let result = cv.transform(&["hello".to_string()]);
        assert!(result.is_err());
    }

    // ── TfidfTransformer tests ───────────────────────────────────────

    #[test]
    fn test_tfidf_transformer_l2_norm() {
        let mut tf = TfidfTransformer::new(true);
        let counts = vec![vec![1.0, 0.0, 2.0], vec![0.0, 3.0, 1.0]];
        let tfidf = tf.fit_transform(&counts).expect("fit_transform failed");
        // Each row should be L2-normalised
        for row in &tfidf {
            let norm: f64 = row.iter().map(|&x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-9, "norm = {}", norm);
        }
    }

    #[test]
    fn test_tfidf_transformer_not_fitted_error() {
        let tf = TfidfTransformer::new(true);
        let result = tf.transform(&[vec![1.0, 2.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tfidf_smooth_vs_no_smooth() {
        let mut tf_smooth = TfidfTransformer::new(true);
        let mut tf_no = TfidfTransformer::new(false);
        let counts = vec![vec![1.0, 2.0], vec![3.0, 0.0]];
        tf_smooth.fit(&counts).expect("fit");
        tf_no.fit(&counts).expect("fit");
        // Smooth IDF should differ from unsmoothed
        assert_ne!(tf_smooth.idf, tf_no.idf);
    }
}
