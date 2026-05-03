//! Text alignment utilities for parallel corpora
//!
//! This module provides word-level alignment methods for bilingual sentence pairs,
//! including IBM Model 1 EM training, symmetrization (grow-diag-final), and
//! alignment quality metrics (Precision / Recall / F1).

use crate::error::{Result, TextError};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Alignment method selector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlignmentMethod {
    /// Simple word-level co-occurrence baseline
    WordBaseline,
    /// Byte-pair-encoded pair-based alignment
    BpePair,
    /// FastAlign-style approximate IBM Model 1
    FastAlign,
}

/// A directed word alignment: source index → target index
pub type AlignmentPair = (usize, usize);

// ---------------------------------------------------------------------------
// Word-level baseline alignment
// ---------------------------------------------------------------------------

/// Align `source_tokens` to `target_tokens` using a pre-built co-occurrence
/// frequency table.
///
/// `co_occurrence` maps `(source_word, target_word)` → count.  For each source
/// token the target token with the highest co-occurrence is chosen.  Source
/// tokens that have no entry in the table are left unaligned.
///
/// # Errors
/// Returns [`TextError::InvalidInput`] when either token list is empty.
pub fn word_alignment(
    source_tokens: &[String],
    target_tokens: &[String],
    co_occurrence: &HashMap<(String, String), usize>,
) -> Result<Vec<AlignmentPair>> {
    if source_tokens.is_empty() {
        return Err(TextError::InvalidInput(
            "source_tokens must not be empty".to_string(),
        ));
    }
    if target_tokens.is_empty() {
        return Err(TextError::InvalidInput(
            "target_tokens must not be empty".to_string(),
        ));
    }

    let mut alignments: Vec<AlignmentPair> = Vec::new();

    for (si, src) in source_tokens.iter().enumerate() {
        let best = target_tokens
            .iter()
            .enumerate()
            .filter_map(|(ti, tgt)| {
                co_occurrence
                    .get(&(src.clone(), tgt.clone()))
                    .map(|&cnt| (ti, cnt))
            })
            .max_by_key(|&(_, cnt)| cnt);

        if let Some((ti, _)) = best {
            alignments.push((si, ti));
        }
    }

    Ok(alignments)
}

// ---------------------------------------------------------------------------
// IBM Model 1
// ---------------------------------------------------------------------------

/// Train IBM Model 1 translation probabilities via EM.
///
/// Returns a map `(source_word, target_word)` → p(target | source).
///
/// `sentence_pairs` is a slice of `(source_sentence, target_sentence)` pairs,
/// each represented as a `Vec<String>` of tokens.  The NULL token is handled
/// internally; callers should **not** prepend it.
///
/// # Errors
/// Returns [`TextError::InvalidInput`] when `n_iter` is zero or `sentence_pairs`
/// is empty.
pub fn ibm_model1(
    sentence_pairs: &[(Vec<String>, Vec<String>)],
    n_iter: usize,
) -> Result<HashMap<(String, String), f64>> {
    if sentence_pairs.is_empty() {
        return Err(TextError::InvalidInput(
            "sentence_pairs must not be empty".to_string(),
        ));
    }
    if n_iter == 0 {
        return Err(TextError::InvalidInput(
            "n_iter must be at least 1".to_string(),
        ));
    }

    const NULL: &str = "<NULL>";

    // Collect vocabulary
    let mut src_vocab: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut tgt_vocab: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (src_sent, tgt_sent) in sentence_pairs {
        for w in src_sent {
            src_vocab.insert(w.clone());
        }
        for w in tgt_sent {
            tgt_vocab.insert(w.clone());
        }
    }
    src_vocab.insert(NULL.to_string());

    // Uniform initialisation
    let uniform = if tgt_vocab.is_empty() {
        1.0
    } else {
        1.0 / tgt_vocab.len() as f64
    };

    let mut t: HashMap<(String, String), f64> = HashMap::new();
    for s in &src_vocab {
        for e in &tgt_vocab {
            t.insert((s.clone(), e.clone()), uniform);
        }
    }

    // EM iterations
    for _ in 0..n_iter {
        // E-step: accumulate expected counts
        let mut count: HashMap<(String, String), f64> = HashMap::new();
        let mut total_s: HashMap<String, f64> = HashMap::new();

        for (src_sent, tgt_sent) in sentence_pairs {
            // Augment source with NULL
            let augmented_src: Vec<&str> = std::iter::once(NULL)
                .chain(src_sent.iter().map(|s| s.as_str()))
                .collect();

            // Normalise over source words for each target word
            for e in tgt_sent {
                let s_total: f64 = augmented_src
                    .iter()
                    .map(|&s| {
                        t.get(&(s.to_string(), e.clone()))
                            .copied()
                            .unwrap_or(uniform)
                    })
                    .sum();

                if s_total > 0.0 {
                    for &s in &augmented_src {
                        let prob = t
                            .get(&(s.to_string(), e.clone()))
                            .copied()
                            .unwrap_or(uniform);
                        let delta = prob / s_total;
                        *count.entry((s.to_string(), e.clone())).or_insert(0.0) += delta;
                        *total_s.entry(s.to_string()).or_insert(0.0) += delta;
                    }
                }
            }
        }

        // M-step: normalise
        for ((s, e), c) in &count {
            let total = total_s.get(s).copied().unwrap_or(1.0);
            t.insert((s.clone(), e.clone()), c / total);
        }
    }

    // Remove NULL entries from the result
    t.retain(|(s, _), _| s != NULL);
    Ok(t)
}

// ---------------------------------------------------------------------------
// Symmetrization: grow-diag-final
// ---------------------------------------------------------------------------

/// Symmetrize two directed alignments using the *grow-diag-final* heuristic.
///
/// `src_to_tgt` contains alignments in the source→target direction;
/// `tgt_to_src` contains alignments in the target→source direction (stored as
/// `(target_idx, source_idx)` pairs).
///
/// Returns the symmetrized alignment as a set of `(source_idx, target_idx)` pairs.
///
/// # Errors
/// Returns [`TextError::ProcessingError`] when the input alignment vectors are
/// empty at the same time (no alignment signal at all).
pub fn symmetrize_alignments(
    src_to_tgt: &[AlignmentPair],
    tgt_to_src: &[AlignmentPair],
) -> Result<Vec<AlignmentPair>> {
    if src_to_tgt.is_empty() && tgt_to_src.is_empty() {
        return Err(TextError::ProcessingError(
            "Both alignment sets are empty; cannot symmetrize".to_string(),
        ));
    }

    // Build intersection
    let s2t_set: std::collections::HashSet<AlignmentPair> = src_to_tgt.iter().copied().collect();
    // tgt_to_src stores (tgt_idx, src_idx); flip to (src_idx, tgt_idx)
    let t2s_set: std::collections::HashSet<AlignmentPair> =
        tgt_to_src.iter().map(|&(ti, si)| (si, ti)).collect();

    let mut result: std::collections::HashSet<AlignmentPair> =
        s2t_set.intersection(&t2s_set).copied().collect();

    // Track which source/target positions are already aligned
    let aligned_src = |set: &std::collections::HashSet<AlignmentPair>, si: usize| {
        set.iter().any(|&(s, _)| s == si)
    };
    let aligned_tgt = |set: &std::collections::HashSet<AlignmentPair>, ti: usize| {
        set.iter().any(|&(_, t)| t == ti)
    };

    // Union of both directions
    let union: std::collections::HashSet<AlignmentPair> =
        s2t_set.union(&t2s_set).copied().collect();

    // Grow: add neighbouring points from the union when at least one endpoint
    // is already aligned
    let neighbors: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    let mut changed = true;
    while changed {
        changed = false;
        let current: Vec<AlignmentPair> = result.iter().copied().collect();
        for (si, ti) in &current {
            for (ds, dt) in &neighbors {
                let ns = (*si as i32 + ds) as usize;
                let nt = (*ti as i32 + dt) as usize;
                let candidate = (ns, nt);
                if union.contains(&candidate) && !result.contains(&candidate) {
                    result.insert(candidate);
                    changed = true;
                }
            }
        }
    }

    // Final: add unaligned points from union
    for &(si, ti) in &union {
        if !aligned_src(&result, si) || !aligned_tgt(&result, ti) {
            result.insert((si, ti));
        }
    }

    let mut out: Vec<AlignmentPair> = result.into_iter().collect();
    out.sort_unstable();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Alignment evaluation
// ---------------------------------------------------------------------------

/// Compute Precision, Recall, and F1 for predicted alignments against gold.
///
/// Both sets are `(source_idx, target_idx)` pairs.
///
/// Returns `(precision, recall, f1)`.
///
/// # Errors
/// Returns [`TextError::InvalidInput`] when both `pred_alignments` and
/// `gold_alignments` are empty (nothing to evaluate).
pub fn alignment_f1(
    pred_alignments: &[AlignmentPair],
    gold_alignments: &[AlignmentPair],
) -> Result<(f64, f64, f64)> {
    if pred_alignments.is_empty() && gold_alignments.is_empty() {
        return Err(TextError::InvalidInput(
            "Both pred and gold alignment sets are empty".to_string(),
        ));
    }

    let pred_set: std::collections::HashSet<AlignmentPair> =
        pred_alignments.iter().copied().collect();
    let gold_set: std::collections::HashSet<AlignmentPair> =
        gold_alignments.iter().copied().collect();

    let tp = pred_set.intersection(&gold_set).count() as f64;

    let precision = if pred_set.is_empty() {
        0.0
    } else {
        tp / pred_set.len() as f64
    };

    let recall = if gold_set.is_empty() {
        0.0
    } else {
        tp / gold_set.len() as f64
    };

    let f1 = if precision + recall < f64::EPSILON {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    Ok((precision, recall, f1))
}

// ---------------------------------------------------------------------------
// AlignedCorpus helper
// ---------------------------------------------------------------------------

/// A sentence-aligned bilingual corpus together with its IBM Model 1
/// translation table.
#[derive(Debug)]
pub struct AlignedCorpus {
    /// Source sentences (tokenized)
    pub source: Vec<Vec<String>>,
    /// Target sentences (tokenized)
    pub target: Vec<Vec<String>>,
    /// Trained translation probabilities p(target | source)
    pub t_table: HashMap<(String, String), f64>,
}

impl AlignedCorpus {
    /// Build an [`AlignedCorpus`] by training IBM Model 1 on `sentence_pairs`
    /// for `n_iter` EM iterations.
    ///
    /// # Errors
    /// Propagates errors from [`ibm_model1`].
    pub fn train(sentence_pairs: Vec<(Vec<String>, Vec<String>)>, n_iter: usize) -> Result<Self> {
        let t_table = ibm_model1(&sentence_pairs, n_iter)?;
        let (source, target) = sentence_pairs.into_iter().unzip();
        Ok(Self {
            source,
            target,
            t_table,
        })
    }

    /// Viterbi-decode the best source→target alignment for sentence pair `idx`.
    ///
    /// For each target token the source token with the highest `t(tgt | src)` is
    /// chosen (including a virtual NULL source token, which produces no output pair).
    ///
    /// # Errors
    /// Returns [`TextError::InvalidInput`] when `idx` is out of range.
    pub fn viterbi_align(&self, idx: usize) -> Result<Vec<AlignmentPair>> {
        if idx >= self.source.len() {
            return Err(TextError::InvalidInput(format!(
                "Sentence pair index {} is out of range (corpus has {} pairs)",
                idx,
                self.source.len()
            )));
        }

        const NULL: &str = "<NULL>";
        let src = &self.source[idx];
        let tgt = &self.target[idx];

        let mut alignments = Vec::new();

        for (ti, tgt_word) in tgt.iter().enumerate() {
            // Check NULL as a baseline
            let null_prob = self
                .t_table
                .get(&(NULL.to_string(), tgt_word.clone()))
                .copied()
                .unwrap_or(0.0);

            let best = src
                .iter()
                .enumerate()
                .map(|(si, src_word)| {
                    let p = self
                        .t_table
                        .get(&(src_word.clone(), tgt_word.clone()))
                        .copied()
                        .unwrap_or(0.0);
                    (si, p)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((si, best_prob)) = best {
                if best_prob >= null_prob {
                    alignments.push((si, ti));
                }
            }
        }

        Ok(alignments)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn test_word_alignment_basic() {
        let mut cooc: HashMap<(String, String), usize> = HashMap::new();
        cooc.insert(("cat".to_string(), "gato".to_string()), 10);
        cooc.insert(("dog".to_string(), "perro".to_string()), 8);

        let src = tok(&["cat", "dog"]);
        let tgt = tok(&["gato", "perro"]);

        let aligns = word_alignment(&src, &tgt, &cooc).expect("alignment failed");
        assert!(aligns.contains(&(0, 0)));
        assert!(aligns.contains(&(1, 1)));
    }

    #[test]
    fn test_word_alignment_empty_source() {
        let cooc: HashMap<(String, String), usize> = HashMap::new();
        let res = word_alignment(&[], &tok(&["a"]), &cooc);
        assert!(res.is_err());
    }

    #[test]
    fn test_ibm_model1_basic() {
        let pairs = vec![
            (tok(&["the", "cat"]), tok(&["le", "chat"])),
            (tok(&["the", "dog"]), tok(&["le", "chien"])),
            (tok(&["a", "cat"]), tok(&["un", "chat"])),
        ];
        let t = ibm_model1(&pairs, 5).expect("ibm_model1 failed");

        // p(chat | cat) should be relatively high
        let p_chat_cat = t
            .get(&("cat".to_string(), "chat".to_string()))
            .copied()
            .unwrap_or(0.0);
        assert!(
            p_chat_cat > 0.0,
            "Expected positive probability for (cat, chat)"
        );
    }

    #[test]
    fn test_ibm_model1_zero_iters() {
        let pairs = vec![(tok(&["a"]), tok(&["b"]))];
        assert!(ibm_model1(&pairs, 0).is_err());
    }

    #[test]
    fn test_symmetrize_alignments() {
        // s2t: 0→0, 1→1
        let s2t = vec![(0, 0), (1, 1)];
        // t2s stored as (tgt, src): 0→0, 1→1
        let t2s = vec![(0, 0), (1, 1)];
        let sym = symmetrize_alignments(&s2t, &t2s).expect("symmetrize failed");
        assert!(sym.contains(&(0, 0)));
        assert!(sym.contains(&(1, 1)));
    }

    #[test]
    fn test_alignment_f1_perfect() {
        let aligns = vec![(0, 0), (1, 1), (2, 2)];
        let (p, r, f1) = alignment_f1(&aligns, &aligns).expect("f1 failed");
        assert!((p - 1.0).abs() < 1e-9);
        assert!((r - 1.0).abs() < 1e-9);
        assert!((f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_alignment_f1_no_overlap() {
        let pred = vec![(0, 1)];
        let gold = vec![(0, 0)];
        let (p, r, f1) = alignment_f1(&pred, &gold).expect("f1 failed");
        assert!((p - 0.0).abs() < 1e-9);
        assert!((r - 0.0).abs() < 1e-9);
        assert!((f1 - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_aligned_corpus_train_viterbi() {
        let pairs = vec![
            (tok(&["the", "cat"]), tok(&["le", "chat"])),
            (tok(&["the", "dog"]), tok(&["le", "chien"])),
            (tok(&["a", "cat"]), tok(&["un", "chat"])),
        ];
        let corpus = AlignedCorpus::train(pairs, 10).expect("train failed");
        let aligns = corpus.viterbi_align(0).expect("viterbi failed");
        // Should produce some alignments
        assert!(!aligns.is_empty());
    }
}
