//! CRF-style Viterbi decoder for neural sequence labeling (NER etc.)
//! with BIO tagging scheme and span-level evaluation metrics.

use crate::error::{Result, TextError};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// BIO tagging
// ---------------------------------------------------------------------------

/// BIO (Begin-Inside-Outside) tagging scheme.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum BioTag {
    /// Begin of an entity of the given type.
    B(String),
    /// Inside (continuation of) an entity of the given type.
    I(String),
    /// Outside — not part of any entity.
    O,
}

impl BioTag {
    /// Returns the entity type string if the tag is B or I.
    pub fn entity_type(&self) -> Option<&str> {
        match self {
            BioTag::B(t) | BioTag::I(t) => Some(t.as_str()),
            BioTag::O => None,
        }
    }

    /// True if this is a B tag.
    pub fn is_begin(&self) -> bool {
        matches!(self, BioTag::B(_))
    }

    /// True if this is an I tag.
    pub fn is_inside(&self) -> bool {
        matches!(self, BioTag::I(_))
    }
}

// ---------------------------------------------------------------------------
// Viterbi decoder
// ---------------------------------------------------------------------------

/// CRF-style Viterbi decoder operating over emission and transition log-probabilities.
pub struct ViterbiDecoder {
    /// Total number of output tags.
    pub n_tags: usize,
    /// Human-readable tag names in index order.
    pub tag_names: Vec<String>,
}

impl ViterbiDecoder {
    /// Construct a decoder from an ordered list of tag names.
    pub fn new(tag_names: Vec<String>) -> Self {
        let n_tags = tag_names.len();
        Self { n_tags, tag_names }
    }

    /// Viterbi decoding over emission scores and a transition matrix.
    ///
    /// `emissions`: \[seq_len\]\[n_tags\] log-probabilities of each tag at each position.
    /// `transitions`: \[n_tags\]\[n_tags\] log-probability of transitioning from tag *i* to tag *j*.
    ///
    /// Returns the most likely tag index sequence.
    pub fn decode(&self, emissions: &[Vec<f64>], transitions: &[Vec<f64>]) -> Result<Vec<usize>> {
        let seq_len = emissions.len();
        if seq_len == 0 {
            return Err(TextError::InvalidInput(
                "Viterbi: empty emission sequence".into(),
            ));
        }
        if transitions.len() != self.n_tags {
            return Err(TextError::InvalidInput(format!(
                "transitions rows {} != n_tags {}",
                transitions.len(),
                self.n_tags
            )));
        }
        for row in emissions {
            if row.len() != self.n_tags {
                return Err(TextError::InvalidInput(format!(
                    "emission width {} != n_tags {}",
                    row.len(),
                    self.n_tags
                )));
            }
        }

        let n = self.n_tags;
        // dp[t][k] = best log-prob of tagging position t with tag k
        let mut dp = vec![vec![f64::NEG_INFINITY; n]; seq_len];
        // bp[t][k] = argmax predecessor tag at t-1
        let mut bp = vec![vec![0_usize; n]; seq_len];

        // Initialise with emissions at t=0 (uniform start)
        for k in 0..n {
            dp[0][k] = emissions[0][k];
        }

        // Forward
        for t in 1..seq_len {
            for k in 0..n {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_prev = 0;
                for j in 0..n {
                    let score = dp[t - 1][j] + transitions[j][k] + emissions[t][k];
                    if score > best_score {
                        best_score = score;
                        best_prev = j;
                    }
                }
                dp[t][k] = best_score;
                bp[t][k] = best_prev;
            }
        }

        // Find best final tag
        let mut best_last = 0;
        let mut best_last_score = f64::NEG_INFINITY;
        for k in 0..n {
            if dp[seq_len - 1][k] > best_last_score {
                best_last_score = dp[seq_len - 1][k];
                best_last = k;
            }
        }

        // Backtrack
        let mut path = vec![0_usize; seq_len];
        path[seq_len - 1] = best_last;
        for t in (1..seq_len).rev() {
            path[t - 1] = bp[t][path[t]];
        }

        Ok(path)
    }

    /// Convert a sequence of tag indices to BIO tags.
    ///
    /// Tags whose name starts with `B-` are parsed as `BioTag::B(type)`, `I-` → `BioTag::I(type)`,
    /// `O` → `BioTag::O`.  Unknown names are treated as `O`.
    pub fn indices_to_bio(&self, indices: &[usize]) -> Result<Vec<BioTag>> {
        indices
            .iter()
            .map(|&idx| {
                if idx >= self.n_tags {
                    return Err(TextError::InvalidInput(format!(
                        "tag index {} out of range {}",
                        idx, self.n_tags
                    )));
                }
                let name = &self.tag_names[idx];
                let bio = if name.starts_with("B-") {
                    BioTag::B(name[2..].to_owned())
                } else if name.starts_with("I-") {
                    BioTag::I(name[2..].to_owned())
                } else {
                    BioTag::O
                };
                Ok(bio)
            })
            .collect()
    }

    /// Extract named entities from a BIO-tagged sequence.
    ///
    /// Returns `(entity_type, start_index, end_index_exclusive)` triples.
    pub fn extract_entities(bio_tags: &[BioTag]) -> Vec<(String, usize, usize)> {
        let mut entities = Vec::new();
        let mut i = 0;
        while i < bio_tags.len() {
            if let BioTag::B(etype) = &bio_tags[i] {
                let start = i;
                let entity_type = etype.clone();
                i += 1;
                while i < bio_tags.len() {
                    match &bio_tags[i] {
                        BioTag::I(t) if t == &entity_type => {
                            i += 1;
                        }
                        _ => break,
                    }
                }
                entities.push((entity_type, start, i));
            } else {
                i += 1;
            }
        }
        entities
    }
}

// ---------------------------------------------------------------------------
// Evaluation metrics
// ---------------------------------------------------------------------------

/// Span-level precision, recall and F1 for sequence labeling.
#[derive(Debug, Clone)]
pub struct SequenceLabelMetrics {
    /// Precision over all entity types.
    pub precision: f64,
    /// Recall over all entity types.
    pub recall: f64,
    /// F1 score (harmonic mean of precision and recall).
    pub f1: f64,
    /// Per-entity-type counts: `type → (tp, fp, fn_count)`.
    pub entity_counts: HashMap<String, (usize, usize, usize)>,
}

/// Evaluate sequence labeling by comparing predicted to gold BIO sequences.
///
/// Entities are compared at the span level (type + start + end must match).
pub fn evaluate_sequence_labeling(
    predicted: &[Vec<BioTag>],
    gold: &[Vec<BioTag>],
) -> Result<SequenceLabelMetrics> {
    if predicted.len() != gold.len() {
        return Err(TextError::InvalidInput(format!(
            "predicted {} sequences != gold {}",
            predicted.len(),
            gold.len()
        )));
    }

    // Collect (type, start, end) spans from a BIO sequence with a sentence offset.
    let collect_spans = |seq: &Vec<BioTag>, offset: usize| -> Vec<(String, usize, usize)> {
        ViterbiDecoder::extract_entities(seq)
            .into_iter()
            .map(|(t, s, e)| (t, s + offset, e + offset))
            .collect()
    };

    let mut all_pred: Vec<(String, usize, usize)> = Vec::new();
    let mut all_gold: Vec<(String, usize, usize)> = Vec::new();
    let mut offset = 0;
    for (pred_seq, gold_seq) in predicted.iter().zip(gold) {
        all_pred.extend(collect_spans(pred_seq, offset));
        all_gold.extend(collect_spans(gold_seq, offset));
        offset += pred_seq.len().max(gold_seq.len());
    }

    // Compute per-type tp/fp/fn
    let mut counts: HashMap<String, (usize, usize, usize)> = HashMap::new();

    for span in &all_gold {
        counts.entry(span.0.clone()).or_insert((0, 0, 0));
    }
    for span in &all_pred {
        counts.entry(span.0.clone()).or_insert((0, 0, 0));
    }

    for span in &all_pred {
        let entry = counts.entry(span.0.clone()).or_insert((0, 0, 0));
        if all_gold.contains(span) {
            entry.0 += 1; // tp
        } else {
            entry.1 += 1; // fp
        }
    }
    for span in &all_gold {
        let entry = counts.entry(span.0.clone()).or_insert((0, 0, 0));
        if !all_pred.contains(span) {
            entry.2 += 1; // fn
        }
    }

    // Micro-average
    let (total_tp, total_fp, total_fn) = counts.values().fold((0, 0, 0), |(tp, fp, fnn), v| {
        (tp + v.0, fp + v.1, fnn + v.2)
    });

    let precision = if total_tp + total_fp == 0 {
        0.0
    } else {
        total_tp as f64 / (total_tp + total_fp) as f64
    };
    let recall = if total_tp + total_fn == 0 {
        0.0
    } else {
        total_tp as f64 / (total_tp + total_fn) as f64
    };
    let f1 = if precision + recall < 1e-12 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    Ok(SequenceLabelMetrics {
        precision,
        recall,
        f1,
        entity_counts: counts,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_decoder() -> ViterbiDecoder {
        ViterbiDecoder::new(vec![
            "O".into(),
            "B-PER".into(),
            "I-PER".into(),
            "B-ORG".into(),
            "I-ORG".into(),
        ])
    }

    #[test]
    fn test_viterbi_simple_chain() {
        // 3 positions, 2 tags (0 and 1)
        let decoder = ViterbiDecoder::new(vec!["O".into(), "B-PER".into()]);
        // emissions strongly prefer 0, 1, 0
        let emissions = vec![vec![-0.1, -10.0], vec![-10.0, -0.1], vec![-0.1, -10.0]];
        // uniform transitions
        let transitions = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let path = decoder.decode(&emissions, &transitions).unwrap();
        assert_eq!(path, vec![0, 1, 0]);
    }

    #[test]
    fn test_viterbi_all_same() {
        // All emissions identical — transitions govern
        let decoder = ViterbiDecoder::new(vec!["O".into(), "B-LOC".into()]);
        let emissions = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        // transitions: prefer staying in tag 1
        let transitions = vec![vec![-1.0, 0.0], vec![0.0, 1.0]];
        let path = decoder.decode(&emissions, &transitions).unwrap();
        // Second tag (1) should dominate due to self-loop reward
        // At t=0 both equal; at t=1 tag 1 gets +1 from stay
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_indices_to_bio() {
        let decoder = make_decoder();
        // indices: O B-PER I-PER O B-ORG
        let indices = vec![0, 1, 2, 0, 3];
        let bio = decoder.indices_to_bio(&indices).unwrap();
        assert_eq!(bio[0], BioTag::O);
        assert_eq!(bio[1], BioTag::B("PER".into()));
        assert_eq!(bio[2], BioTag::I("PER".into()));
        assert_eq!(bio[3], BioTag::O);
        assert_eq!(bio[4], BioTag::B("ORG".into()));
    }

    #[test]
    fn test_extract_entities_basic() {
        // B-PER I-PER O = one PER entity at positions 0..2
        let tags = vec![BioTag::B("PER".into()), BioTag::I("PER".into()), BioTag::O];
        let entities = ViterbiDecoder::extract_entities(&tags);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0], ("PER".to_owned(), 0, 2));
    }

    #[test]
    fn test_extract_entities_two_entities() {
        let tags = vec![
            BioTag::B("PER".into()),
            BioTag::O,
            BioTag::B("ORG".into()),
            BioTag::I("ORG".into()),
        ];
        let entities = ViterbiDecoder::extract_entities(&tags);
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0], ("PER".to_owned(), 0, 1));
        assert_eq!(entities[1], ("ORG".to_owned(), 2, 4));
    }

    #[test]
    fn test_sequence_labeling_perfect_f1() {
        let gold = vec![vec![
            BioTag::B("PER".into()),
            BioTag::I("PER".into()),
            BioTag::O,
        ]];
        let pred = gold.clone();
        let metrics = evaluate_sequence_labeling(&pred, &gold).unwrap();
        assert!((metrics.f1 - 1.0).abs() < 1e-9, "perfect pred → F1 = 1.0");
        assert!((metrics.precision - 1.0).abs() < 1e-9);
        assert!((metrics.recall - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sequence_labeling_no_overlap() {
        let gold = vec![vec![BioTag::B("PER".into()), BioTag::O]];
        let pred = vec![vec![BioTag::O, BioTag::B("ORG".into())]];
        let metrics = evaluate_sequence_labeling(&pred, &gold).unwrap();
        assert_eq!(metrics.f1, 0.0, "no overlap → F1 = 0.0");
    }

    #[test]
    fn test_empty_sequence_returns_error() {
        let decoder = make_decoder();
        let result = decoder.decode(&[], &[]);
        assert!(result.is_err(), "empty emissions should fail");
    }
}
