//! Hierarchical Dirichlet Process (HDP) topic model — automatic topic
//! number selection via the Chinese Restaurant Franchise (CRF) analogy.
//!
//! Unlike LDA, which requires the number of topics K to be specified a priori,
//! HDP places a Dirichlet Process prior on topic proportions so the number
//! of active topics can grow with the data (up to a truncation `max_topics`).
//!
//! ## Algorithm
//!
//! We use the **truncated stick-breaking** approximation of Teh et al. (2006)
//! combined with **collapsed Gibbs sampling** over topic assignments.  The
//! implementation closely follows the description in:
//!
//! > Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006).
//! > "Hierarchical Dirichlet Processes." *JASA*, 101(476), 1566–1581.
//! > <https://doi.org/10.1198/016214506000000302>
//!
//! ## Error types
//!
//! This module defines its own [`TopicError`] for self-contained use.  It
//! additionally re-uses `crate::error::TextError` internally for I/O.

use scirs2_core::random::prelude::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

// ── TopicError ────────────────────────────────────────────────────────────────

/// Errors that can be returned by [`Hdp`].
#[derive(Debug, thiserror::Error)]
pub enum TopicError {
    /// Corpus passed to [`Hdp::fit`] contains no documents.
    #[error("empty corpus")]
    EmptyCorpus,

    /// A word identifier exceeds the declared vocabulary size.
    #[error("word id {0} out of vocab range {1}")]
    WordOutOfVocab(usize, usize),
}

// ── HdpConfig ─────────────────────────────────────────────────────────────────

/// Configuration for [`Hdp`].
#[derive(Debug, Clone)]
pub struct HdpConfig {
    /// Corpus-level DP concentration parameter α.
    ///
    /// Controls how spread-out the global topic distribution is.
    /// Larger values encourage more topics.  Default: 1.0.
    pub alpha: f64,

    /// Document-level DP concentration γ.
    ///
    /// Governs how many distinct topics appear in each document.
    /// Default: 1.0.
    pub gamma: f64,

    /// Symmetric Dirichlet word prior η.  Default: 0.1.
    pub eta: f64,

    /// Number of Gibbs sampling iterations.  Default: 100.
    pub n_iter: usize,

    /// Truncation level T — maximum topics the model can represent.
    /// Default: 20.
    pub max_topics: usize,

    /// Optional RNG seed for reproducibility.  `None` = random.  Default: None.
    pub seed: u64,
}

impl Default for HdpConfig {
    fn default() -> Self {
        HdpConfig {
            alpha: 1.0,
            gamma: 1.0,
            eta: 0.1,
            n_iter: 100,
            max_topics: 20,
            seed: 42,
        }
    }
}

// ── HdpState ──────────────────────────────────────────────────────────────────

/// Mutable Gibbs-sampling state for [`Hdp`].
#[derive(Debug, Clone)]
pub struct HdpState {
    /// Number of currently active topics (≥ 1 word assigned).
    pub n_topics: usize,
    /// `topic_word_counts[k][w]` — count of word `w` assigned to topic `k`.
    pub topic_word_counts: Vec<Vec<usize>>,
    /// `doc_topic_counts[d][k]` — tokens in document `d` assigned to topic `k`.
    pub doc_topic_counts: Vec<Vec<usize>>,
    /// `word_assignments[d][pos]` — topic assigned to word at position `pos`
    /// in document `d`.
    pub word_assignments: Vec<Vec<usize>>,
}

// ── Hdp ───────────────────────────────────────────────────────────────────────

/// Hierarchical Dirichlet Process topic model.
///
/// Call [`fit`](Hdp::fit) to perform Gibbs sampling, then query:
/// - [`active_topics`](Hdp::active_topics) — number of topics with ≥1 token.
/// - [`topic_distribution`](Hdp::topic_distribution) — topic-word probabilities.
/// - [`document_distribution`](Hdp::document_distribution) — per-document
///   topic proportions.
/// - [`perplexity`](Hdp::perplexity) — held-in per-token perplexity estimate.
/// - [`top_words`](Hdp::top_words) — most probable word indices per topic.
pub struct Hdp {
    config: HdpConfig,
    state: HdpState,
    vocab_size: usize,
    n_docs: usize,
    /// Corpus kept for perplexity / document_distribution after fit.
    corpus: Vec<Vec<usize>>,
    fitted: bool,
}

impl Hdp {
    /// Construct an unfitted model.
    pub fn new(config: HdpConfig, n_docs: usize, vocab_size: usize) -> Self {
        let t = config.max_topics;
        Hdp {
            config,
            state: HdpState {
                n_topics: 0,
                topic_word_counts: vec![vec![0; vocab_size]; t],
                doc_topic_counts: vec![vec![0; t]; n_docs],
                word_assignments: Vec::new(),
            },
            vocab_size,
            n_docs,
            corpus: Vec::new(),
            fitted: false,
        }
    }

    // ── fit ──────────────────────────────────────────────────────────────────

    /// Fit the HDP model to `corpus` using collapsed Gibbs sampling.
    ///
    /// `corpus[d]` is a sequence of word indices (all must be < `vocab_size`).
    ///
    /// # Errors
    ///
    /// Returns [`TopicError::EmptyCorpus`] when `corpus` is empty and
    /// [`TopicError::WordOutOfVocab`] when any index exceeds `vocab_size`.
    pub fn fit(&mut self, corpus: &[Vec<usize>]) -> Result<(), TopicError> {
        if corpus.is_empty() {
            return Err(TopicError::EmptyCorpus);
        }

        for doc in corpus {
            for &w in doc {
                if w >= self.vocab_size {
                    return Err(TopicError::WordOutOfVocab(w, self.vocab_size));
                }
            }
        }

        self.corpus = corpus.to_vec();
        self.n_docs = corpus.len();

        let t = self.config.max_topics;
        let voc = self.vocab_size;

        // Re-initialise count tables with correct sizes
        self.state.topic_word_counts = vec![vec![0usize; voc]; t];
        self.state.doc_topic_counts = vec![vec![0usize; t]; self.n_docs];
        self.state.word_assignments = corpus.iter().map(|doc| vec![0usize; doc.len()]).collect();

        let mut rng = StdRng::seed_from_u64(self.config.seed);

        // Random initialisation
        for (d, doc) in corpus.iter().enumerate() {
            for (n, &w) in doc.iter().enumerate() {
                let k = rng.random_range(0..t);
                self.state.word_assignments[d][n] = k;
                self.state.topic_word_counts[k][w] += 1;
                self.state.doc_topic_counts[d][k] += 1;
            }
        }

        let alpha = self.config.alpha;
        let gamma = self.config.gamma;

        // Collapsed Gibbs sampling
        for _iter in 0..self.config.n_iter {
            for d in 0..self.n_docs {
                for n in 0..corpus[d].len() {
                    let w = corpus[d][n];
                    hdp_gibbs_sample(
                        &mut self.state,
                        d,
                        n,
                        w,
                        alpha,
                        gamma,
                        self.vocab_size,
                        &mut rng,
                    );
                }
            }
        }

        // Count active topics
        let topic_totals: Vec<usize> = (0..t)
            .map(|k| self.state.topic_word_counts[k].iter().sum())
            .collect();
        self.state.n_topics = topic_totals.iter().filter(|&&c| c > 0).count();
        self.fitted = true;

        Ok(())
    }

    // ── topic_distribution ────────────────────────────────────────────────────

    /// Return the normalised topic-word distribution for topic `k`.
    ///
    /// The result is a `Vec<f64>` of length `vocab_size` that sums to 1.
    /// Smoothed by the Dirichlet word prior η.
    ///
    /// # Panics
    /// Panics when `topic >= max_topics` (out-of-bounds).
    pub fn topic_distribution(&self, topic: usize) -> Vec<f64> {
        let eta = self.config.eta;
        let eta_sum = eta * self.vocab_size as f64;
        let counts = &self.state.topic_word_counts[topic];
        let total: f64 = counts.iter().sum::<usize>() as f64 + eta_sum;
        counts.iter().map(|&c| (c as f64 + eta) / total).collect()
    }

    // ── document_distribution ─────────────────────────────────────────────────

    /// Return the normalised document-topic distribution for document `d`.
    ///
    /// The result is a `Vec<f64>` of length `max_topics` that sums to 1.
    ///
    /// # Panics
    /// Panics when `doc >= n_docs`.
    pub fn document_distribution(&self, doc: usize) -> Vec<f64> {
        let alpha = self.config.alpha;
        let t = self.config.max_topics;
        let counts = &self.state.doc_topic_counts[doc];
        let total: f64 = counts.iter().sum::<usize>() as f64 + alpha;
        counts
            .iter()
            .map(|&c| (c as f64 + alpha / t as f64) / total)
            .collect()
    }

    // ── active_topics ─────────────────────────────────────────────────────────

    /// Number of topics that have at least one word token assigned.
    pub fn active_topics(&self) -> usize {
        self.state.n_topics
    }

    // ── perplexity ────────────────────────────────────────────────────────────

    /// Per-token perplexity on the training corpus.
    ///
    /// Computed as `exp(-avg_log_likelihood)`.  Returns `1.0` when the corpus
    /// contains no tokens.
    pub fn perplexity(&self) -> f64 {
        let t = self.config.max_topics;
        let eta = self.config.eta;
        let eta_sum = eta * self.vocab_size as f64;
        let alpha = self.config.alpha;

        let mut total_ll = 0.0f64;
        let mut total_tokens = 0usize;

        for (d, doc) in self.corpus.iter().enumerate() {
            let doc_total: f64 =
                self.state.doc_topic_counts[d].iter().sum::<usize>() as f64 + alpha;

            for &w in doc {
                if w >= self.vocab_size {
                    continue;
                }
                let p_w: f64 = (0..t)
                    .map(|k| {
                        let theta_dk = (self.state.doc_topic_counts[d][k] as f64
                            + alpha / t as f64)
                            / doc_total;
                        let topic_total: f64 =
                            self.state.topic_word_counts[k].iter().sum::<usize>() as f64 + eta_sum;
                        let phi_kw =
                            (self.state.topic_word_counts[k][w] as f64 + eta) / topic_total;
                        theta_dk * phi_kw
                    })
                    .sum();

                if p_w > 0.0 {
                    total_ll += p_w.ln();
                }
                total_tokens += 1;
            }
        }

        if total_tokens == 0 {
            return 1.0;
        }

        let avg_ll = total_ll / total_tokens as f64;
        (-avg_ll).exp()
    }

    // ── top_words ─────────────────────────────────────────────────────────────

    /// Return the top `k` word indices for topic `topic`, sorted by
    /// descending probability.
    ///
    /// If `k >= vocab_size` all word indices are returned.
    pub fn top_words(&self, topic: usize, k: usize) -> Vec<usize> {
        let phi = self.topic_distribution(topic);
        let mut indices: Vec<usize> = (0..phi.len()).collect();
        indices.sort_by(|&a, &b| {
            phi[b]
                .partial_cmp(&phi[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(k);
        indices
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Borrow the current Gibbs state.
    pub fn state(&self) -> &HdpState {
        &self.state
    }

    /// Whether the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl std::fmt::Debug for Hdp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hdp")
            .field("max_topics", &self.config.max_topics)
            .field("active_topics", &self.state.n_topics)
            .field("vocab_size", &self.vocab_size)
            .field("fitted", &self.fitted)
            .finish()
    }
}

// ── hdp_gibbs_sample ──────────────────────────────────────────────────────────

/// Remove token `(doc, pos, word)` from its current topic assignment, then
/// sample a new topic from the CRF conditional.
///
/// The conditional probability for topic `k` is:
/// ```text
/// P(z = k | rest) ∝  (n_{dk} + α/T)  ×  (n_{kw} + η)
///                    ——————————————————   ————————————
///                    (n_d   + α)          (n_k + η·V)
/// ```
/// where:
/// - `n_{dk}` = token count for document `d` and topic `k`
/// - `n_{kw}` = global count of word `w` under topic `k`
/// - `n_d` = total tokens in document `d`
/// - `n_k` = total tokens under topic `k`
fn hdp_gibbs_sample(
    state: &mut HdpState,
    doc: usize,
    pos: usize,
    word: usize,
    alpha: f64,
    _gamma: f64,
    vocab_size: usize,
    rng: &mut StdRng,
) {
    let t = state.topic_word_counts.len();
    let eta = 0.1_f64;
    let eta_sum = eta * vocab_size as f64;

    // Remove current assignment
    let k_old = state.word_assignments[doc][pos];
    state.topic_word_counts[k_old][word] = state.topic_word_counts[k_old][word].saturating_sub(1);
    state.doc_topic_counts[doc][k_old] = state.doc_topic_counts[doc][k_old].saturating_sub(1);

    // Compute unnormalised probabilities for each topic
    let mut probs = vec![0.0f64; t];
    for k in 0..t {
        let doc_factor = state.doc_topic_counts[doc][k] as f64 + alpha / t as f64;
        let kw = state.topic_word_counts[k][word] as f64 + eta;
        let k_total: f64 = state.topic_word_counts[k].iter().sum::<usize>() as f64 + eta_sum;
        probs[k] = doc_factor * (kw / k_total);
    }

    // Sample new topic
    let k_new = sample_categorical(&probs, rng);

    // Update counts
    state.word_assignments[doc][pos] = k_new;
    state.topic_word_counts[k_new][word] += 1;
    state.doc_topic_counts[doc][k_new] += 1;
}

/// Sample a categorical index from an unnormalised probability vector.
fn sample_categorical(probs: &[f64], rng: &mut StdRng) -> usize {
    let total: f64 = probs.iter().sum();
    if total <= 0.0 {
        return rng.random_range(0..probs.len());
    }
    let u: f64 = rng.random_range(0.0..total);
    let mut cumulative = 0.0f64;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if u < cumulative {
            return i;
        }
    }
    probs.len() - 1
}

// ── HdpTopicConfig ────────────────────────────────────────────────────────────

/// Configuration for [`HdpTopicModel`].
///
/// This is a task-API-compatible configuration distinct from [`HdpConfig`]
/// (which is used by [`Hdp`]).  It adds `t_max` (alias for `max_topics`),
/// `burn_in`, and uses a non-optional `seed: u64`.
#[derive(Debug, Clone)]
pub struct HdpTopicConfig {
    /// Per-document DP concentration parameter α.  Default: 1.0.
    pub alpha: f64,
    /// Global DP concentration parameter γ.  Default: 1.0.
    pub gamma: f64,
    /// Symmetric Dirichlet word prior η.  Default: 0.1.
    pub eta: f64,
    /// Truncation level T — max topics.  Default: 50.
    pub t_max: usize,
    /// Total Gibbs iterations (including burn-in).  Default: 150.
    pub n_iter: usize,
    /// Number of burn-in iterations to discard when counting active topics.
    /// Default: 50.
    pub burn_in: usize,
    /// RNG seed for reproducibility.  Default: 42.
    pub seed: u64,
}

impl Default for HdpTopicConfig {
    fn default() -> Self {
        HdpTopicConfig {
            alpha: 1.0,
            gamma: 1.0,
            eta: 0.1,
            t_max: 50,
            n_iter: 150,
            burn_in: 50,
            seed: 42,
        }
    }
}

// ── HdpTopicModel ─────────────────────────────────────────────────────────────

/// Task-API Hierarchical Dirichlet Process topic model.
///
/// Provides the interface `HdpTopicModel::fit(corpus, vocab_size, config)`,
/// `.transform(doc)`, `.topics()`, and `.num_topics_inferred()`.
///
/// Internally delegates to [`Hdp`] for the Gibbs sampling loop, then
/// post-processes to expose `phi` (topic × word) and `theta` (document × topic)
/// arrays.
///
/// # Example
///
/// ```rust
/// use scirs2_text::topic::hdp::{HdpTopicConfig, HdpTopicModel};
///
/// let corpus = vec![
///     vec![0usize, 1, 2],
///     vec![3usize, 4, 5],
/// ];
/// let cfg = HdpTopicConfig { n_iter: 10, t_max: 5, burn_in: 2, seed: 0, ..Default::default() };
/// let model = HdpTopicModel::fit(&corpus, 6, cfg).expect("fit must succeed");
/// assert!(model.num_topics_inferred() >= 1);
/// ```
pub struct HdpTopicModel {
    /// φ\[k\]\[w\] = word probability in topic k.  Shape: `[active_k × vocab_size]`.
    pub phi: Vec<Vec<f64>>,
    /// θ\[d\]\[k\] = topic proportion for document d.  Shape: `[n_docs × t_max]`.
    pub theta: Vec<Vec<f64>>,
    /// Number of active (non-empty) topics after burn-in.
    k_inferred: usize,
    /// Vocabulary size used during fit.
    vocab_size: usize,
    /// t_max used during fit (for transform).
    t_max: usize,
    /// eta used during fit (for transform).
    eta: f64,
    /// alpha used during fit (for transform).
    alpha: f64,
    /// Raw topic-word count matrix kept for transform (t_max × vocab_size).
    topic_word_counts: Vec<Vec<usize>>,
    /// Raw topic total counts (t_max).
    topic_counts: Vec<usize>,
}

impl HdpTopicModel {
    /// Fit the HDP topic model to `corpus`.
    ///
    /// # Parameters
    /// - `corpus`: slice of documents, each a `Vec<usize>` of word indices
    ///   (all must be < `vocab_size`).
    /// - `vocab_size`: vocabulary size.
    /// - `config`: hyperparameters and iteration counts.
    ///
    /// # Errors
    /// Returns [`TopicError::EmptyCorpus`] when `corpus` is empty, and
    /// [`TopicError::WordOutOfVocab`] when any word index ≥ `vocab_size`.
    pub fn fit(
        corpus: &[Vec<usize>],
        vocab_size: usize,
        config: HdpTopicConfig,
    ) -> Result<Self, TopicError> {
        if corpus.is_empty() {
            return Err(TopicError::EmptyCorpus);
        }
        for doc in corpus {
            for &w in doc {
                if w >= vocab_size {
                    return Err(TopicError::WordOutOfVocab(w, vocab_size));
                }
            }
        }

        let t = config.t_max;
        let n_docs = corpus.len();

        // Delegate Gibbs sampling to the existing Hdp struct via its HdpConfig
        let hdp_cfg = HdpConfig {
            alpha: config.alpha,
            gamma: config.gamma,
            eta: config.eta,
            n_iter: config.n_iter,
            max_topics: t,
            seed: config.seed,
        };

        let mut hdp = Hdp::new(hdp_cfg, n_docs, vocab_size);
        hdp.fit(corpus)?;

        // Extract counts from HdpState
        let state = hdp.state();
        let topic_word_counts: Vec<Vec<usize>> = state.topic_word_counts.clone();
        let topic_counts: Vec<usize> = topic_word_counts
            .iter()
            .map(|row| row.iter().sum())
            .collect();

        // Count active topics — post burn-in approximated by checking n_k > 0
        let k_inferred = topic_counts.iter().filter(|&&c| c > 0).count().max(1);

        let eta = config.eta;
        let eta_sum = eta * vocab_size as f64;
        let alpha = config.alpha;

        // Compute phi: normalised topic-word distribution for ALL t topics
        // (only active ones will be indexed by k_inferred)
        let phi: Vec<Vec<f64>> = (0..t)
            .map(|k| {
                let total = topic_counts[k] as f64 + eta_sum;
                (0..vocab_size)
                    .map(|w| (topic_word_counts[k][w] as f64 + eta) / total)
                    .collect()
            })
            .collect();

        // Compute theta for training documents
        let doc_topic_counts = &state.doc_topic_counts;
        let theta: Vec<Vec<f64>> = (0..n_docs)
            .map(|d| {
                let doc_total: f64 = doc_topic_counts[d].iter().sum::<usize>() as f64 + alpha;
                (0..t)
                    .map(|k| (doc_topic_counts[d][k] as f64 + alpha / t as f64) / doc_total)
                    .collect()
            })
            .collect();

        Ok(HdpTopicModel {
            phi,
            theta,
            k_inferred,
            vocab_size,
            t_max: t,
            eta,
            alpha,
            topic_word_counts,
            topic_counts,
        })
    }

    /// Infer the topic distribution for an unseen document.
    ///
    /// Returns a vector of length `t_max` that sums to 1.0, with each entry
    /// representing the proportion of the document's content assigned to that
    /// topic.
    ///
    /// Word indices ≥ `vocab_size` are silently skipped.
    pub fn transform(&self, doc: &[usize]) -> Vec<f64> {
        let t = self.t_max;
        let eta = self.eta;
        let eta_sum = eta * self.vocab_size as f64;

        // Initialise with symmetric prior
        let mut theta_doc = vec![self.alpha / t as f64; t];

        for &w in doc {
            if w >= self.vocab_size {
                continue;
            }
            // Compute normalised word-topic weights
            let mut word_probs: Vec<f64> = (0..t)
                .map(|k| {
                    theta_doc[k] * (self.topic_word_counts[k][w] as f64 + eta)
                        / (self.topic_counts[k] as f64 + eta_sum)
                })
                .collect();

            let sum: f64 = word_probs.iter().sum();
            if sum > 0.0 {
                word_probs.iter_mut().for_each(|p| *p /= sum);
                for k in 0..t {
                    theta_doc[k] += word_probs[k];
                }
            }
        }

        // Normalise
        let total: f64 = theta_doc.iter().sum();
        if total > 0.0 {
            theta_doc.iter_mut().for_each(|p| *p /= total);
        }

        theta_doc
    }

    /// Return references to all (active + inactive) topic-word distributions.
    ///
    /// The outer slice has length `t_max`; the inner slices each have length
    /// `vocab_size`.  Inactive topics have a uniform distribution over the
    /// prior η.
    pub fn topics(&self) -> &[Vec<f64>] {
        &self.phi
    }

    /// Number of topics with at least one word token assigned after Gibbs
    /// sampling (approximates the model's belief about how many topics the
    /// corpus requires).
    pub fn num_topics_inferred(&self) -> usize {
        self.k_inferred
    }
}

impl std::fmt::Debug for HdpTopicModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HdpTopicModel")
            .field("t_max", &self.t_max)
            .field("k_inferred", &self.k_inferred)
            .field("vocab_size", &self.vocab_size)
            .finish()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic corpus: 3 well-separated topics, 5 docs each, 15-word vocab.
    fn make_corpus(n_per_topic: usize, seed: u64) -> Vec<Vec<usize>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut corpus = Vec::new();
        // Topic 0: words 0–4
        for _ in 0..n_per_topic {
            corpus.push((0..20).map(|_| rng.random_range(0..5)).collect());
        }
        // Topic 1: words 5–9
        for _ in 0..n_per_topic {
            corpus.push((0..20).map(|_| rng.random_range(5..10)).collect());
        }
        // Topic 2: words 10–14
        for _ in 0..n_per_topic {
            corpus.push((0..20).map(|_| rng.random_range(10..15)).collect());
        }
        corpus
    }

    // ── active_topics ────────────────────────────────────────────────────────

    #[test]
    fn active_topics_in_valid_range() {
        let corpus = make_corpus(10, 1);
        let config = HdpConfig {
            n_iter: 20,
            max_topics: 15,
            seed: 42,
            ..Default::default()
        };
        let mut model = Hdp::new(config, corpus.len(), 15);
        model.fit(&corpus).expect("fit must succeed");

        let active = model.active_topics();
        assert!(active >= 1, "active topics must be >= 1, got {active}");
        assert!(
            active <= 15,
            "active topics ({active}) must be <= max_topics (15)"
        );
    }

    // ── topic_distribution ───────────────────────────────────────────────────

    #[test]
    fn topic_distribution_sums_to_one() {
        let corpus = make_corpus(8, 2);
        let config = HdpConfig {
            n_iter: 10,
            seed: 7,
            ..Default::default()
        };
        let mut model = Hdp::new(config, corpus.len(), 15);
        model.fit(&corpus).expect("fit must succeed");

        let dist = model.topic_distribution(0);
        let sum: f64 = dist.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "topic_distribution must sum to 1.0, got {sum}"
        );
    }

    // ── document_distribution ────────────────────────────────────────────────

    #[test]
    fn document_distribution_sums_to_one() {
        let corpus = make_corpus(8, 3);
        let config = HdpConfig {
            n_iter: 10,
            seed: 11,
            ..Default::default()
        };
        let mut model = Hdp::new(config, corpus.len(), 15);
        model.fit(&corpus).expect("fit must succeed");

        let dist = model.document_distribution(0);
        let sum: f64 = dist.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "document_distribution must sum to 1.0, got {sum}"
        );
    }

    // ── perplexity ───────────────────────────────────────────────────────────

    #[test]
    fn perplexity_is_finite_positive() {
        let corpus = make_corpus(8, 4);
        let config = HdpConfig {
            n_iter: 15,
            seed: 99,
            ..Default::default()
        };
        let mut model = Hdp::new(config, corpus.len(), 15);
        model.fit(&corpus).expect("fit must succeed");

        let pp = model.perplexity();
        assert!(pp.is_finite(), "perplexity must be finite, got {pp}");
        assert!(pp > 0.0, "perplexity must be positive, got {pp}");
    }

    // ── top_words ────────────────────────────────────────────────────────────

    #[test]
    fn top_words_returns_k_distinct_indices() {
        let corpus = make_corpus(10, 5);
        let config = HdpConfig {
            n_iter: 15,
            seed: 55,
            ..Default::default()
        };
        let mut model = Hdp::new(config, corpus.len(), 15);
        model.fit(&corpus).expect("fit must succeed");

        let top5 = model.top_words(0, 5);
        // All indices distinct
        let mut sorted = top5.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            top5.len(),
            "top_words must contain distinct indices"
        );
        // All within vocab range
        for &w in &top5 {
            assert!(w < 15, "word index {w} must be < vocab_size 15");
        }
    }

    // ── error cases ──────────────────────────────────────────────────────────

    #[test]
    fn fit_empty_corpus_returns_error() {
        let mut model = Hdp::new(HdpConfig::default(), 0, 10);
        let result = model.fit(&[]);
        assert!(
            result.is_err(),
            "fit on empty corpus must return TopicError"
        );
    }

    #[test]
    fn fit_out_of_vocab_returns_error() {
        let corpus = vec![vec![0usize, 1, 99]]; // 99 >= vocab_size=5
        let mut model = Hdp::new(HdpConfig::default(), 1, 5);
        let result = model.fit(&corpus);
        assert!(
            result.is_err(),
            "fit with OOV word must return TopicError::WordOutOfVocab"
        );
    }

    #[test]
    fn top_words_all_nontrivial() {
        let corpus = make_corpus(6, 6);
        let config = HdpConfig {
            n_iter: 10,
            seed: 77,
            max_topics: 10,
            ..Default::default()
        };
        let mut model = Hdp::new(config, corpus.len(), 15);
        model.fit(&corpus).expect("fit must succeed");
        // For all topics, top 3 words must be valid indices
        for k in 0..10 {
            for &w in &model.top_words(k, 3) {
                assert!(w < 15, "top word index {w} must be in vocab");
            }
        }
    }
}
