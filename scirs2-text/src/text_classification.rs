//! Advanced text classification module.
//!
//! This module provides three complementary classifiers:
//!
//! - [`NaiveBayesClassifier`] — multinomial Naïve Bayes with Laplace smoothing.
//! - [`TextCnnLite`] — convolutional n-gram feature extraction followed by
//!   multinomial logistic regression (gradient-descent trained).
//! - [`TfIdfLogisticClassifier`] — TF-IDF features + multinomial logistic
//!   regression trained with mini-batch gradient descent.
//!
//! All classifiers implement the same `fit / predict / predict_proba` pattern
//! and never use `unwrap()`.

use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use crate::vectorize::{TfidfVectorizer, Vectorizer};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Tokenisation helper
// ---------------------------------------------------------------------------

/// Tokenise `text` into lowercased word tokens.
fn tokenize_lower(text: &str) -> Vec<String> {
    let tokenizer = WordTokenizer::default();
    tokenizer
        .tokenize(text)
        .unwrap_or_default()
        .into_iter()
        .map(|t| t.to_lowercase())
        .collect()
}

// ---------------------------------------------------------------------------
// NaiveBayesClassifier
// ---------------------------------------------------------------------------

/// Multinomial Naïve Bayes text classifier with Laplace (additive) smoothing.
///
/// Internally works in log-space to prevent floating-point underflow.
///
/// # Example
///
/// ```rust
/// use scirs2_text::text_classification::NaiveBayesClassifier;
///
/// let mut clf = NaiveBayesClassifier::new(1.0);
/// let texts  = &["spam spam buy now", "hello friend good morning"];
/// let labels = &["spam", "ham"];
/// clf.fit(texts, labels).unwrap();
/// assert_eq!(clf.predict("buy now cheap").unwrap(), "spam");
/// ```
pub struct NaiveBayesClassifier {
    /// Vocabulary: word → column index
    vocabulary: HashMap<String, usize>,
    /// Log prior probability for each class: `log P(class)`
    class_log_priors: Vec<f64>,
    /// Log word probability for each class:
    /// `class_word_log_probs[c][w] = log P(word w | class c)`
    class_word_log_probs: Vec<Vec<f64>>,
    /// Ordered class names
    classes: Vec<String>,
    /// Laplace smoothing parameter α (> 0)
    alpha: f64,
}

impl NaiveBayesClassifier {
    /// Create a new classifier with smoothing parameter `alpha`.
    ///
    /// `alpha = 1.0` is the standard Laplace (add-one) smoothing.
    ///
    /// # Errors
    ///
    /// Returns [`TextError::InvalidInput`] when `alpha <= 0`.
    pub fn new(alpha: f64) -> Self {
        Self {
            vocabulary: HashMap::new(),
            class_log_priors: Vec::new(),
            class_word_log_probs: Vec::new(),
            classes: Vec::new(),
            alpha,
        }
    }

    /// Train the classifier.
    ///
    /// # Errors
    ///
    /// Returns [`TextError::InvalidInput`] when:
    /// - `texts` and `labels` have different lengths.
    /// - The corpus is empty.
    /// - `alpha <= 0`.
    pub fn fit(&mut self, texts: &[&str], labels: &[&str]) -> Result<()> {
        if texts.len() != labels.len() {
            return Err(TextError::InvalidInput(format!(
                "texts ({}) and labels ({}) must have the same length",
                texts.len(),
                labels.len()
            )));
        }
        if texts.is_empty() {
            return Err(TextError::InvalidInput("Empty training corpus".to_string()));
        }
        if self.alpha <= 0.0 {
            return Err(TextError::InvalidInput(
                "alpha must be positive".to_string(),
            ));
        }

        // --- build vocabulary and class index ---
        let mut class_index: HashMap<String, usize> = HashMap::new();
        for &label in labels {
            let n = class_index.len();
            class_index.entry(label.to_string()).or_insert(n);
        }

        let n_classes = class_index.len();
        let mut class_names: Vec<String> = vec![String::new(); n_classes];
        for (name, &idx) in &class_index {
            class_names[idx] = name.clone();
        }

        let mut class_doc_counts = vec![0usize; n_classes];
        // word counts per class: class → (word → count)
        let mut class_word_counts: Vec<HashMap<String, f64>> =
            (0..n_classes).map(|_| HashMap::new()).collect();

        for (&text, &label) in texts.iter().zip(labels.iter()) {
            let class_idx = *class_index
                .get(label)
                .ok_or_else(|| TextError::InvalidInput(format!("Unknown label '{label}'")))?;
            class_doc_counts[class_idx] += 1;

            for word in tokenize_lower(text) {
                // Add to vocabulary
                let n = self.vocabulary.len();
                self.vocabulary.entry(word.clone()).or_insert(n);
                *class_word_counts[class_idx].entry(word).or_insert(0.0) += 1.0;
            }
        }

        let n_docs = texts.len() as f64;
        let vocab_size = self.vocabulary.len();

        // Log priors
        let class_log_priors: Vec<f64> = class_doc_counts
            .iter()
            .map(|&c| (c as f64 / n_docs).ln())
            .collect();

        // Log word probabilities: P(w|c) = (count(w,c) + alpha) / (total_words(c) + alpha * V)
        let mut class_word_log_probs: Vec<Vec<f64>> =
            (0..n_classes).map(|_| vec![0.0; vocab_size]).collect();

        for c in 0..n_classes {
            let total: f64 = class_word_counts[c].values().sum();
            let denom = total + self.alpha * vocab_size as f64;

            for (word, &col_idx) in &self.vocabulary {
                let cnt = class_word_counts[c].get(word).copied().unwrap_or(0.0);
                class_word_log_probs[c][col_idx] = ((cnt + self.alpha) / denom).ln();
            }
        }

        self.classes = class_names;
        self.class_log_priors = class_log_priors;
        self.class_word_log_probs = class_word_log_probs;

        Ok(())
    }

    /// Predict the most probable class for `text`.
    ///
    /// # Errors
    ///
    /// Returns [`TextError::ModelNotFitted`] if `fit` has not been called.
    pub fn predict(&self, text: &str) -> Result<String> {
        let proba = self.predict_log_proba(text)?;
        let best = proba
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| TextError::ModelNotFitted("No classes available".to_string()))?;
        Ok(self.classes[best].clone())
    }

    /// Return `(class_name, probability)` pairs sorted by probability descending.
    ///
    /// Probabilities are derived from log-scores via softmax normalisation.
    ///
    /// # Errors
    ///
    /// Returns [`TextError::ModelNotFitted`] if `fit` has not been called.
    pub fn predict_proba(&self, text: &str) -> Result<Vec<(String, f64)>> {
        let log_proba = self.predict_log_proba(text)?;

        // Softmax
        let max_val = log_proba.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = log_proba.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();

        let mut result: Vec<(String, f64)> = self
            .classes
            .iter()
            .zip(exps.iter())
            .map(|(name, &e)| (name.clone(), e / sum))
            .collect();

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(result)
    }

    /// Compute accuracy on a labelled test set.
    ///
    /// # Errors
    ///
    /// Returns [`TextError::InvalidInput`] when lengths differ.
    pub fn score(&self, texts: &[&str], labels: &[&str]) -> Result<f64> {
        if texts.len() != labels.len() {
            return Err(TextError::InvalidInput(
                "texts and labels must have the same length".to_string(),
            ));
        }
        if texts.is_empty() {
            return Ok(0.0);
        }

        let mut correct = 0usize;
        for (&text, &label) in texts.iter().zip(labels.iter()) {
            if let Ok(pred) = self.predict(text) {
                if pred == label {
                    correct += 1;
                }
            }
        }

        Ok(correct as f64 / texts.len() as f64)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn predict_log_proba(&self, text: &str) -> Result<Vec<f64>> {
        if self.classes.is_empty() {
            return Err(TextError::ModelNotFitted(
                "NaiveBayesClassifier has not been fitted yet".to_string(),
            ));
        }

        let tokens = tokenize_lower(text);
        let n_classes = self.classes.len();
        let mut log_scores: Vec<f64> = self.class_log_priors.clone();

        for word in &tokens {
            if let Some(&col) = self.vocabulary.get(word) {
                for c in 0..n_classes {
                    log_scores[c] += self.class_word_log_probs[c][col];
                }
            }
        }

        Ok(log_scores)
    }
}

// ---------------------------------------------------------------------------
// TextCnnLite
// ---------------------------------------------------------------------------

/// Lightweight text-CNN classifier.
///
/// Extracts n-gram features (bag-of-n-grams) for each filter size and then
/// trains a multinomial logistic regression head using mini-batch stochastic
/// gradient descent.
///
/// This is intentionally a "lite" variant — it approximates a CNN's 1-D
/// convolution by computing frequency counts of character/word n-grams and
/// using the max-pooled (most-frequent) value as the feature.  A proper
/// convolutional network would require a full neural-network framework.
///
/// # Example
///
/// ```rust
/// use scirs2_text::text_classification::TextCnnLite;
///
/// let mut clf = TextCnnLite::new(vec![2, 3], 8);
/// let texts  = &["good movie fun", "bad film boring", "great show entertaining", "terrible awful waste"];
/// let labels = &["pos", "neg", "pos", "neg"];
/// clf.fit(texts, labels, 20).unwrap();
/// ```
pub struct TextCnnLite {
    /// Filter window sizes (in words).
    filter_sizes: Vec<usize>,
    /// Number of feature maps per filter size (for n-gram counting, this sets
    /// the top-k cutoff for the feature vector).
    n_filters: usize,
    /// Vocabulary: n-gram → feature index
    vocab: HashMap<String, usize>,
    /// Logistic regression weights: `weights[c][f]`
    weights: Vec<Vec<f64>>,
    /// Per-class bias terms
    bias: Vec<f64>,
    /// Ordered class names
    classes: Vec<String>,
}

impl TextCnnLite {
    /// Create a new `TextCnnLite`.
    ///
    /// # Parameters
    ///
    /// - `filter_sizes`: n-gram window sizes, e.g. `[2, 3, 4]`.
    /// - `n_filters`: number of top n-gram features to keep per filter size.
    pub fn new(filter_sizes: Vec<usize>, n_filters: usize) -> Self {
        Self {
            filter_sizes,
            n_filters,
            vocab: HashMap::new(),
            weights: Vec::new(),
            bias: Vec::new(),
            classes: Vec::new(),
        }
    }

    /// Train the classifier.
    ///
    /// Internally:
    /// 1. Build the n-gram vocabulary from all documents.
    /// 2. Vectorise each document into a frequency vector.
    /// 3. Train multinomial logistic regression with SGD.
    ///
    /// # Errors
    ///
    /// Returns [`TextError::InvalidInput`] when `texts` and `labels` lengths differ
    /// or when the corpus is empty.
    pub fn fit(&mut self, texts: &[&str], labels: &[&str], epochs: usize) -> Result<()> {
        if texts.len() != labels.len() {
            return Err(TextError::InvalidInput(format!(
                "texts ({}) and labels ({}) must have the same length",
                texts.len(),
                labels.len()
            )));
        }
        if texts.is_empty() {
            return Err(TextError::InvalidInput("Empty training corpus".to_string()));
        }

        // --- class mapping ---
        let mut class_index: HashMap<String, usize> = HashMap::new();
        for &label in labels {
            let n = class_index.len();
            class_index.entry(label.to_string()).or_insert(n);
        }
        let n_classes = class_index.len();
        let mut class_names: Vec<String> = vec![String::new(); n_classes];
        for (name, &idx) in &class_index {
            class_names[idx] = name.clone();
        }

        // --- build n-gram vocabulary ---
        let mut ngram_counts: HashMap<String, usize> = HashMap::new();
        for &text in texts {
            let words = tokenize_lower(text);
            for &size in &self.filter_sizes {
                for ngram in ngrams(&words, size) {
                    *ngram_counts.entry(ngram).or_insert(0) += 1;
                }
            }
        }

        // Keep top n_filters * filter_sizes.len() n-grams by frequency
        let mut ngram_vec: Vec<(String, usize)> = ngram_counts.into_iter().collect();
        ngram_vec.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        let max_feats = self.n_filters * self.filter_sizes.len();
        self.vocab = ngram_vec
            .into_iter()
            .take(max_feats)
            .enumerate()
            .map(|(i, (ng, _))| (ng, i))
            .collect();

        let n_features = self.vocab.len();
        if n_features == 0 {
            return Err(TextError::InvalidInput(
                "No n-gram features found in corpus".to_string(),
            ));
        }

        // --- vectorise documents ---
        let mut x_data: Vec<Vec<f64>> = Vec::with_capacity(texts.len());
        let mut y_labels: Vec<usize> = Vec::with_capacity(texts.len());

        for (&text, &label) in texts.iter().zip(labels.iter()) {
            x_data.push(self.vectorize(text));
            let class_idx = *class_index
                .get(label)
                .ok_or_else(|| TextError::InvalidInput(format!("Unknown label '{label}'")))?;
            y_labels.push(class_idx);
        }

        // --- initialise weights ---
        self.weights = vec![vec![0.0f64; n_features]; n_classes];
        self.bias = vec![0.0f64; n_classes];

        // --- mini-batch SGD ---
        let lr = 0.1_f64;
        let n_samples = texts.len();

        for _epoch in 0..epochs {
            for i in 0..n_samples {
                let x = &x_data[i];
                let y = y_labels[i];

                // Forward: compute logits and softmax
                let logits: Vec<f64> = (0..n_classes)
                    .map(|c| {
                        self.bias[c]
                            + x.iter()
                                .zip(self.weights[c].iter())
                                .map(|(xi, wi)| xi * wi)
                                .sum::<f64>()
                    })
                    .collect();

                let probs = softmax(&logits);

                // Backward: cross-entropy gradient
                for c in 0..n_classes {
                    let delta = probs[c] - if c == y { 1.0 } else { 0.0 };
                    self.bias[c] -= lr * delta;
                    for (j, &xj) in x.iter().enumerate() {
                        self.weights[c][j] -= lr * delta * xj;
                    }
                }
            }
        }

        self.classes = class_names;
        Ok(())
    }

    /// Predict the most probable class for `text`.
    ///
    /// # Errors
    ///
    /// Returns [`TextError::ModelNotFitted`] when the model has not been trained.
    pub fn predict(&self, text: &str) -> Result<String> {
        if self.classes.is_empty() {
            return Err(TextError::ModelNotFitted(
                "TextCnnLite has not been fitted yet".to_string(),
            ));
        }
        let x = self.vectorize(text);
        let n_classes = self.classes.len();
        let logits: Vec<f64> = (0..n_classes)
            .map(|c| {
                self.bias[c]
                    + x.iter()
                        .zip(self.weights[c].iter())
                        .map(|(xi, wi)| xi * wi)
                        .sum::<f64>()
            })
            .collect();

        let best = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| TextError::ModelNotFitted("No classes available".to_string()))?;

        Ok(self.classes[best].clone())
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Compute the n-gram frequency feature vector for `text`.
    fn vectorize(&self, text: &str) -> Vec<f64> {
        let mut v = vec![0.0f64; self.vocab.len()];
        let words = tokenize_lower(text);
        for &size in &self.filter_sizes {
            for ngram in ngrams(&words, size) {
                if let Some(&idx) = self.vocab.get(&ngram) {
                    v[idx] += 1.0;
                }
            }
        }
        v
    }
}

// ---------------------------------------------------------------------------
// TfIdfLogisticClassifier
// ---------------------------------------------------------------------------

/// Logistic regression classifier trained on TF-IDF features.
///
/// Uses the existing [`TfidfVectorizer`] to compute features and mini-batch
/// stochastic gradient descent (SGD) to train the model.
///
/// # Example
///
/// ```rust
/// use scirs2_text::text_classification::TfIdfLogisticClassifier;
///
/// let mut clf = TfIdfLogisticClassifier::new();
/// let texts  = &["good great excellent", "bad terrible awful", "okay decent fine", "poor mediocre subpar"];
/// let labels = &["pos", "neg", "pos", "neg"];
/// clf.fit(texts, labels, 50, 0.1).unwrap();
/// let pred = clf.predict("excellent wonderful").unwrap();
/// assert!(!pred.is_empty());
/// ```
pub struct TfIdfLogisticClassifier {
    /// Underlying TF-IDF vectorizer.
    vectorizer: TfidfVectorizer,
    /// Weight matrix: `weights[class_idx]` is a vector over vocabulary.
    weights: Vec<Vec<f64>>,
    /// Per-class bias terms.
    bias: Vec<f64>,
    /// Ordered class names.
    classes: Vec<String>,
    /// Whether the vectorizer has been fitted.
    fitted: bool,
}

impl Default for TfIdfLogisticClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl TfIdfLogisticClassifier {
    /// Create a new classifier with default TF-IDF settings.
    pub fn new() -> Self {
        Self {
            vectorizer: TfidfVectorizer::new(false, true, Some("l2".to_string())),
            weights: Vec::new(),
            bias: Vec::new(),
            classes: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the classifier.
    ///
    /// # Parameters
    ///
    /// - `texts`: Training documents.
    /// - `labels`: Corresponding class labels.
    /// - `max_iter`: Number of SGD epochs.
    /// - `lr`: Learning rate for SGD.
    ///
    /// # Errors
    ///
    /// Returns [`TextError::InvalidInput`] when inputs are inconsistent.
    pub fn fit(&mut self, texts: &[&str], labels: &[&str], max_iter: usize, lr: f64) -> Result<()> {
        if texts.len() != labels.len() {
            return Err(TextError::InvalidInput(format!(
                "texts ({}) and labels ({}) must have the same length",
                texts.len(),
                labels.len()
            )));
        }
        if texts.is_empty() {
            return Err(TextError::InvalidInput("Empty training corpus".to_string()));
        }
        if lr <= 0.0 {
            return Err(TextError::InvalidInput(
                "Learning rate must be positive".to_string(),
            ));
        }

        // --- build class mapping ---
        let mut class_index: HashMap<String, usize> = HashMap::new();
        for &label in labels {
            let n = class_index.len();
            class_index.entry(label.to_string()).or_insert(n);
        }
        let n_classes = class_index.len();
        let mut class_names: Vec<String> = vec![String::new(); n_classes];
        for (name, &idx) in &class_index {
            class_names[idx] = name.clone();
        }

        // --- vectorise ---
        let x_matrix = self.vectorizer.fit_transform(texts)?;
        let n_features = x_matrix.ncols();

        let y_labels: Vec<usize> = labels
            .iter()
            .map(|&label| {
                class_index
                    .get(label)
                    .copied()
                    .ok_or_else(|| TextError::InvalidInput(format!("Unknown label '{label}'")))
            })
            .collect::<Result<_>>()?;

        // --- initialise weights ---
        self.weights = vec![vec![0.0f64; n_features]; n_classes];
        self.bias = vec![0.0f64; n_classes];

        // --- SGD ---
        let n_samples = x_matrix.nrows();

        for _epoch in 0..max_iter {
            for i in 0..n_samples {
                let x_row = x_matrix.row(i);
                let y = y_labels[i];

                // Logits
                let logits: Vec<f64> = (0..n_classes)
                    .map(|c| {
                        self.bias[c]
                            + x_row
                                .iter()
                                .zip(self.weights[c].iter())
                                .map(|(xi, wi)| xi * wi)
                                .sum::<f64>()
                    })
                    .collect();

                let probs = softmax(&logits);

                // Gradient update
                for c in 0..n_classes {
                    let delta = probs[c] - if c == y { 1.0 } else { 0.0 };
                    self.bias[c] -= lr * delta;
                    for (j, &xj) in x_row.iter().enumerate() {
                        self.weights[c][j] -= lr * delta * xj;
                    }
                }
            }
        }

        self.classes = class_names;
        self.fitted = true;
        Ok(())
    }

    /// Predict the most probable class for `text`.
    ///
    /// # Errors
    ///
    /// Returns [`TextError::ModelNotFitted`] when the model has not been trained.
    pub fn predict(&self, text: &str) -> Result<String> {
        let probs = self.predict_proba(text)?;
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| self.classes[i].clone())
            .ok_or_else(|| TextError::ModelNotFitted("No classes available".to_string()))
    }

    /// Return the probability distribution over classes for `text`.
    ///
    /// # Errors
    ///
    /// Returns [`TextError::ModelNotFitted`] when the model has not been trained.
    pub fn predict_proba(&self, text: &str) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(TextError::ModelNotFitted(
                "TfIdfLogisticClassifier has not been fitted yet".to_string(),
            ));
        }

        let x_vec = self.vectorizer.transform(text)?;
        let n_classes = self.classes.len();

        let logits: Vec<f64> = (0..n_classes)
            .map(|c| {
                self.bias[c]
                    + x_vec
                        .iter()
                        .zip(self.weights[c].iter())
                        .map(|(xi, wi)| xi * wi)
                        .sum::<f64>()
            })
            .collect();

        Ok(softmax(&logits))
    }
}

// ---------------------------------------------------------------------------
// Free helper functions
// ---------------------------------------------------------------------------

/// Compute word n-grams of `size` from `words`.
fn ngrams(words: &[String], size: usize) -> Vec<String> {
    if size == 0 || words.len() < size {
        return Vec::new();
    }
    (0..=words.len() - size)
        .map(|i| words[i..i + size].join(" "))
        .collect()
}

/// Numerically stable softmax.
fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- NaiveBayesClassifier tests ---

    #[test]
    fn test_nb_fit_predict_binary() {
        let mut clf = NaiveBayesClassifier::new(1.0);
        let texts = &[
            "spam buy now cheap discount",
            "spam offer free money",
            "hello how are you doing",
            "good morning friend",
        ];
        let labels = &["spam", "spam", "ham", "ham"];
        clf.fit(texts, labels).expect("fit should succeed");

        let pred = clf
            .predict("buy cheap spam now")
            .expect("predict should succeed");
        assert_eq!(pred, "spam");

        let pred2 = clf
            .predict("good morning hello")
            .expect("predict should succeed");
        assert_eq!(pred2, "ham");
    }

    #[test]
    fn test_nb_predict_proba() {
        let mut clf = NaiveBayesClassifier::new(1.0);
        let texts = &[
            "machine learning data science",
            "deep learning neural network",
            "cooking recipe food delicious",
            "restaurant menu chef dinner",
        ];
        let labels = &["tech", "tech", "food", "food"];
        clf.fit(texts, labels).expect("fit should succeed");

        let proba = clf
            .predict_proba("neural network deep learning")
            .expect("predict_proba should succeed");
        assert_eq!(proba.len(), 2);

        // Probabilities sum to ~1.0
        let total: f64 = proba.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-9, "probabilities should sum to 1");

        // First entry should be "tech" with highest probability
        assert_eq!(proba[0].0, "tech");
        assert!(proba[0].1 > 0.5, "tech probability should exceed 0.5");
    }

    #[test]
    fn test_nb_score_above_chance() {
        let mut clf = NaiveBayesClassifier::new(1.0);
        let train_texts = &[
            "positive great excellent wonderful",
            "positive good happy nice",
            "positive fantastic brilliant awesome",
            "negative bad terrible awful",
            "negative horrible disappointing poor",
            "negative dreadful appalling dire",
        ];
        let train_labels = &[
            "positive", "positive", "positive", "negative", "negative", "negative",
        ];
        clf.fit(train_texts, train_labels)
            .expect("fit should succeed");

        let test_texts = &["excellent wonderful", "terrible awful bad"];
        let test_labels = &["positive", "negative"];
        let acc = clf
            .score(test_texts, test_labels)
            .expect("score should succeed");

        assert!(acc > 0.5, "accuracy should exceed chance: {}", acc);
    }

    #[test]
    fn test_nb_error_on_empty_corpus() {
        let mut clf = NaiveBayesClassifier::new(1.0);
        let result = clf.fit(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_nb_error_on_length_mismatch() {
        let mut clf = NaiveBayesClassifier::new(1.0);
        let result = clf.fit(&["text"], &["label1", "label2"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_nb_not_fitted_error() {
        let clf = NaiveBayesClassifier::new(1.0);
        let result = clf.predict("some text");
        assert!(result.is_err());
    }

    #[test]
    fn test_nb_multiclass() {
        let mut clf = NaiveBayesClassifier::new(0.5);
        let texts = &[
            "soccer football goal kick",
            "basketball dunk three pointer",
            "baseball pitcher batting home run",
            "soccer penalty kick goal",
            "basketball court dribble shoot",
            "baseball strike out innings",
        ];
        let labels = &[
            "soccer",
            "basketball",
            "baseball",
            "soccer",
            "basketball",
            "baseball",
        ];
        clf.fit(texts, labels).expect("fit should succeed");

        let pred = clf
            .predict("goal kick soccer field")
            .expect("predict should succeed");
        assert_eq!(pred, "soccer");
    }

    // --- TextCnnLite tests ---

    #[test]
    fn test_cnn_fit_succeeds() {
        let mut clf = TextCnnLite::new(vec![2, 3], 8);
        let texts = &[
            "good movie fun entertaining",
            "bad film boring terrible",
            "great show exciting wonderful",
            "awful program dull waste",
        ];
        let labels = &["pos", "neg", "pos", "neg"];
        let result = clf.fit(texts, labels, 20);
        assert!(result.is_ok(), "fit should succeed: {:?}", result);
    }

    #[test]
    fn test_cnn_predict_returns_valid_class() {
        let mut clf = TextCnnLite::new(vec![2, 3], 8);
        let texts = &[
            "wonderful excellent amazing brilliant",
            "terrible dreadful awful horrible",
            "great fantastic superb outstanding",
            "poor disappointing bad mediocre",
        ];
        let labels = &["pos", "neg", "pos", "neg"];
        clf.fit(texts, labels, 30).expect("fit should succeed");

        let pred = clf
            .predict("excellent wonderful")
            .expect("predict should succeed");
        assert!(
            pred == "pos" || pred == "neg",
            "prediction should be a valid class"
        );
    }

    #[test]
    fn test_cnn_not_fitted_error() {
        let clf = TextCnnLite::new(vec![2], 4);
        assert!(clf.predict("text").is_err());
    }

    // --- TfIdfLogisticClassifier tests ---

    #[test]
    fn test_tfidf_logistic_fit_and_predict() {
        let mut clf = TfIdfLogisticClassifier::new();
        let texts = &[
            "machine learning artificial intelligence",
            "deep neural network training",
            "cooking recipe baking flour",
            "restaurant food delicious chef",
            "algorithm data science research",
            "dinner menu ingredients spices",
        ];
        let labels = &["tech", "tech", "food", "food", "tech", "food"];
        clf.fit(texts, labels, 50, 0.1).expect("fit should succeed");

        let pred = clf
            .predict("neural network algorithm")
            .expect("predict should succeed");
        assert_eq!(pred, "tech");
    }

    #[test]
    fn test_tfidf_logistic_predict_proba_sums_to_one() {
        let mut clf = TfIdfLogisticClassifier::new();
        let texts = &[
            "positive happy good",
            "negative sad bad",
            "positive great excellent",
            "negative terrible awful",
        ];
        let labels = &["pos", "neg", "pos", "neg"];
        clf.fit(texts, labels, 30, 0.05)
            .expect("fit should succeed");

        let probs = clf
            .predict_proba("happy good great")
            .expect("predict_proba should succeed");
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "probabilities must sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_tfidf_logistic_not_fitted_error() {
        let clf = TfIdfLogisticClassifier::new();
        assert!(clf.predict("text").is_err());
    }

    #[test]
    fn test_tfidf_logistic_error_on_empty_corpus() {
        let mut clf = TfIdfLogisticClassifier::new();
        let result = clf.fit(&[], &[], 10, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_tfidf_logistic_error_on_length_mismatch() {
        let mut clf = TfIdfLogisticClassifier::new();
        let result = clf.fit(&["text"], &["a", "b"], 10, 0.1);
        assert!(result.is_err());
    }

    // --- Utility helpers ---

    #[test]
    fn test_ngrams_bigrams() {
        let words: Vec<String> = ["the", "quick", "brown", "fox"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let bigrams = ngrams(&words, 2);
        assert_eq!(bigrams.len(), 3);
        assert_eq!(bigrams[0], "the quick");
        assert_eq!(bigrams[2], "brown fox");
    }

    #[test]
    fn test_ngrams_empty_when_size_too_large() {
        let words: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let result = ngrams(&words, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_softmax_properties() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_large_values_stable() {
        let logits = vec![1000.0, 1001.0, 999.0];
        let probs = softmax(&logits);
        assert!(probs.iter().all(|&p| p.is_finite()));
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}
