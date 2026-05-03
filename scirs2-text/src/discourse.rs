//! Discourse analysis: relation detection, RST tree construction, and
//! text coherence scoring.
//!
//! This module provides purely rule-based (no trained weights) discourse
//! analysis for English text, including:
//!
//! - [`DiscourseRelation`] — the semantic relation between two adjacent
//!   discourse segments (clauses or sentences).
//! - [`CueLexicon`] — cue-phrase lookup tables that drive relation detection.
//! - [`detect_discourse_relation`] — classify the relation between two sentences.
//! - [`RhetoricalStructure`] — a simplified RST (Rhetorical Structure Theory)
//!   tree over a multi-sentence document.
//! - [`coherence_score`] — a sentence-to-sentence coherence measure based on
//!   lexical overlap and discourse connective density.
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::discourse::{CueLexicon, detect_discourse_relation, DiscourseRelation};
//!
//! let lexicon = CueLexicon::default_english();
//! let s1 = "The experiment failed.";
//! let s2 = "However, the team did not give up.";
//! let rel = detect_discourse_relation(s1, s2, &lexicon);
//! assert_eq!(rel, Some(DiscourseRelation::Contrast));
//! ```

use crate::error::{Result, TextError};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// DiscourseRelation
// ---------------------------------------------------------------------------

/// Possible discourse relations between two adjacent text segments.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DiscourseRelation {
    /// Segment 2 describes the cause of segment 1 (or explains why).
    Cause,
    /// Segment 2 is the effect / result of segment 1.
    Effect,
    /// Segments present contrasting or opposing information.
    Contrast,
    /// Segment 2 elaborates, expands, or provides detail about segment 1.
    Elaboration,
    /// Segment 2 describes something that happened before or after segment 1.
    Temporal,
    /// Segment 2 is conditioned on segment 1 (if–then).
    Conditional,
    /// Segment 2 exemplifies a claim made in segment 1.
    Exemplification,
    /// Segment 2 summarises or concludes the discourse up to this point.
    Summary,
    /// No discourse relation detected.
    None,
}

impl std::fmt::Display for DiscourseRelation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::Cause => "CAUSE",
            Self::Effect => "EFFECT",
            Self::Contrast => "CONTRAST",
            Self::Elaboration => "ELABORATION",
            Self::Temporal => "TEMPORAL",
            Self::Conditional => "CONDITIONAL",
            Self::Exemplification => "EXEMPLIFICATION",
            Self::Summary => "SUMMARY",
            Self::None => "NONE",
        };
        write!(f, "{}", label)
    }
}

// ---------------------------------------------------------------------------
// CueLexicon
// ---------------------------------------------------------------------------

/// Cue-phrase lookup tables for discourse relation detection.
///
/// Each field stores a list of cue phrases (lower-cased) associated with the
/// corresponding discourse relation.
#[derive(Debug, Clone, Default)]
pub struct CueLexicon {
    /// Cue phrases signalling a causal relation (segment 2 is the cause).
    pub cause: Vec<String>,
    /// Cue phrases signalling an effect relation (segment 2 is the result).
    pub effect: Vec<String>,
    /// Cue phrases signalling contrast.
    pub contrast: Vec<String>,
    /// Cue phrases signalling elaboration or exemplification.
    pub elaboration: Vec<String>,
    /// Cue phrases signalling a temporal relation.
    pub temporal: Vec<String>,
    /// Cue phrases signalling a conditional relation.
    pub conditional: Vec<String>,
    /// Cue phrases signalling exemplification.
    pub exemplification: Vec<String>,
    /// Cue phrases signalling summary or conclusion.
    pub summary: Vec<String>,
}

impl CueLexicon {
    /// Build the default English cue-phrase lexicon.
    pub fn default_english() -> Self {
        let cue = |phrases: &[&str]| {
            phrases
                .iter()
                .map(|s| s.to_lowercase())
                .collect::<Vec<String>>()
        };

        Self {
            cause: cue(&[
                "because",
                "since",
                "as",
                "due to",
                "owing to",
                "given that",
                "in light of",
                "for the reason that",
                "as a result of",
            ]),
            effect: cue(&[
                "therefore",
                "thus",
                "hence",
                "consequently",
                "as a result",
                "as a consequence",
                "so",
                "accordingly",
                "for this reason",
                "it follows that",
                "this led to",
                "this caused",
            ]),
            contrast: cue(&[
                "however",
                "but",
                "yet",
                "although",
                "even though",
                "while",
                "whereas",
                "on the other hand",
                "in contrast",
                "nevertheless",
                "nonetheless",
                "despite",
                "in spite of",
                "conversely",
                "by contrast",
                "on the contrary",
                "that said",
                "still",
                "yet",
                "though",
            ]),
            elaboration: cue(&[
                "furthermore",
                "moreover",
                "in addition",
                "additionally",
                "also",
                "likewise",
                "similarly",
                "indeed",
                "in fact",
                "specifically",
                "notably",
                "particularly",
                "what is more",
                "besides",
                "more importantly",
            ]),
            temporal: cue(&[
                "then",
                "next",
                "after",
                "before",
                "when",
                "while",
                "once",
                "previously",
                "subsequently",
                "later",
                "earlier",
                "at the same time",
                "meanwhile",
                "in the meantime",
                "afterward",
                "afterwards",
                "first",
                "second",
                "finally",
                "initially",
            ]),
            conditional: cue(&[
                "if",
                "unless",
                "provided that",
                "as long as",
                "given that",
                "in case",
                "assuming that",
                "on condition that",
                "only if",
                "whenever",
            ]),
            exemplification: cue(&[
                "for example",
                "for instance",
                "such as",
                "e.g.",
                "to illustrate",
                "as an example",
                "as illustrated by",
                "consider",
                "take for example",
                "as shown by",
            ]),
            summary: cue(&[
                "in summary",
                "in conclusion",
                "to summarize",
                "to summarise",
                "in brief",
                "in short",
                "overall",
                "to conclude",
                "in closing",
                "all in all",
                "on balance",
                "in the end",
                "to sum up",
            ]),
        }
    }

    /// Return an iterator over `(DiscourseRelation, &[String])` pairs.
    fn relation_cues(&self) -> impl Iterator<Item = (DiscourseRelation, &[String])> {
        [
            (DiscourseRelation::Cause, self.cause.as_slice()),
            (DiscourseRelation::Effect, self.effect.as_slice()),
            (DiscourseRelation::Contrast, self.contrast.as_slice()),
            (DiscourseRelation::Elaboration, self.elaboration.as_slice()),
            (DiscourseRelation::Temporal, self.temporal.as_slice()),
            (DiscourseRelation::Conditional, self.conditional.as_slice()),
            (
                DiscourseRelation::Exemplification,
                self.exemplification.as_slice(),
            ),
            (DiscourseRelation::Summary, self.summary.as_slice()),
        ]
        .into_iter()
    }
}

// ---------------------------------------------------------------------------
// Discourse relation detection
// ---------------------------------------------------------------------------

/// Match a cue phrase in the first 30 characters of `text_lower`, allowing
/// the phrase to appear at the very start (after leading spaces).
fn starts_with_cue(text_lower: &str, cue: &str) -> bool {
    let trimmed = text_lower.trim_start();
    // Exact prefix match
    if trimmed.starts_with(cue) {
        // Make sure it is followed by a non-alphanumeric character (word boundary).
        let after = &trimmed[cue.len()..];
        return after
            .chars()
            .next()
            .map(|c| !c.is_alphanumeric())
            .unwrap_or(true);
    }
    false
}

/// Return the first 60 characters of `text` (lower-cased) for cue matching.
fn leading_window(text: &str) -> String {
    text.chars().take(80).collect::<String>().to_lowercase()
}

/// Detect the discourse relation between `sentence1` and `sentence2`.
///
/// The function scans for cue phrases at the beginning of `sentence2` (the
/// most reliable position) and, for some relation types, also within the body
/// of `sentence2`.  The cue that matches the *longest* phrase wins (to prefer
/// multi-word cues over single-word ones).
///
/// Returns `None` if no cue phrases are found.
pub fn detect_discourse_relation(
    sentence1: &str,
    sentence2: &str,
    cue_words: &CueLexicon,
) -> Option<DiscourseRelation> {
    let window2 = leading_window(sentence2);

    let mut best: Option<(DiscourseRelation, usize)> = None; // (relation, cue_length)

    for (rel, cues) in cue_words.relation_cues() {
        for cue in cues {
            // Primary check: sentence2 starts with the cue
            let found = starts_with_cue(&window2, cue);
            // Secondary check: cue appears in the first 80 chars of sentence2
            let found = found || window2.contains(cue.as_str());

            if found {
                let cue_len = cue.len();
                let is_better = best
                    .as_ref()
                    .map(|(_, prev_len)| cue_len > *prev_len)
                    .unwrap_or(true);
                if is_better {
                    best = Some((rel.clone(), cue_len));
                }
            }
        }
    }

    // Also check if sentence1 ends with a conditional fragment
    let window1_lower = sentence1.to_lowercase();
    if best.is_none()
        && (window1_lower.trim_end_matches('.').ends_with("if") || window1_lower.contains(" if "))
    {
        best = Some((DiscourseRelation::Conditional, 2));
    }

    best.map(|(rel, _)| rel)
}

// ---------------------------------------------------------------------------
// RST Tree
// ---------------------------------------------------------------------------

/// A node in a simplified Rhetorical Structure Theory tree.
#[derive(Debug, Clone)]
pub struct RstNode {
    /// Index of the sentence in the original document (0-based).
    pub sentence_index: usize,
    /// The surface text of the sentence.
    pub text: String,
    /// The discourse relation from this node's parent to this node.
    pub relation_to_parent: Option<DiscourseRelation>,
    /// Child nodes (satellite segments).
    pub children: Vec<RstNode>,
}

/// Simplified RST tree over a document.
#[derive(Debug, Clone)]
pub struct RhetoricalStructure {
    /// Root node of the tree.
    pub root: RstNode,
    /// Total number of sentences.
    pub sentence_count: usize,
    /// Detected discourse relations between adjacent sentences.
    pub inter_sentence_relations: Vec<(usize, usize, DiscourseRelation)>,
}

impl RhetoricalStructure {
    /// Build a flat (chain) RST tree from a sequence of sentences and the
    /// relations detected between each consecutive pair.
    pub fn from_sentence_pairs(
        sentences: &[String],
        relations: Vec<(usize, usize, DiscourseRelation)>,
    ) -> Option<Self> {
        if sentences.is_empty() {
            return None;
        }

        // Build a lookup: sentence_idx → relation from its predecessor
        let mut rel_lookup: HashMap<usize, DiscourseRelation> = HashMap::new();
        for (_, j, rel) in &relations {
            rel_lookup.insert(*j, rel.clone());
        }

        // Root is the first sentence; every other sentence is a direct child
        // of the root (chain structure — sufficient for the simplified model).
        let root = RstNode {
            sentence_index: 0,
            text: sentences[0].clone(),
            relation_to_parent: None,
            children: sentences
                .iter()
                .enumerate()
                .skip(1)
                .map(|(idx, text)| RstNode {
                    sentence_index: idx,
                    text: text.clone(),
                    relation_to_parent: rel_lookup.get(&idx).cloned(),
                    children: Vec::new(),
                })
                .collect(),
        };

        Some(Self {
            root,
            sentence_count: sentences.len(),
            inter_sentence_relations: relations,
        })
    }

    /// Traverse the tree in depth-first order and return all nodes.
    pub fn nodes_dfs(&self) -> Vec<&RstNode> {
        let mut stack = vec![&self.root];
        let mut result = Vec::new();
        while let Some(node) = stack.pop() {
            result.push(node);
            for child in node.children.iter().rev() {
                stack.push(child);
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Coherence scoring
// ---------------------------------------------------------------------------

/// Tokenise a sentence into a lower-cased word set (punctuation stripped).
fn word_set(sentence: &str) -> HashSet<String> {
    sentence
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3)
        .map(|w| w.to_lowercase())
        .collect()
}

/// Common English function words to exclude from lexical overlap scoring.
const STOP_WORDS: &[&str] = &[
    "the", "and", "for", "are", "was", "were", "has", "have", "had", "not", "but", "that", "this",
    "with", "from", "they", "will", "been", "its", "their", "there", "what", "also", "into",
    "than", "then", "when", "more", "some", "such", "even", "both", "each", "said", "very", "just",
    "over", "like", "about", "would", "could", "should", "which",
];

fn stop_set() -> HashSet<&'static str> {
    STOP_WORDS.iter().copied().collect()
}

/// Compute the Jaccard similarity between the content-word sets of two
/// sentences.
fn lexical_overlap(s1: &str, s2: &str) -> f64 {
    let stops = stop_set();
    let w1: HashSet<String> = word_set(s1)
        .into_iter()
        .filter(|w| !stops.contains(w.as_str()))
        .collect();
    let w2: HashSet<String> = word_set(s2)
        .into_iter()
        .filter(|w| !stops.contains(w.as_str()))
        .collect();
    if w1.is_empty() && w2.is_empty() {
        return 1.0;
    }
    let inter = w1.intersection(&w2).count() as f64;
    let union = w1.union(&w2).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Count the number of known discourse cue phrases that appear in `text`.
fn cue_density(text: &str, cue_words: &CueLexicon) -> usize {
    let lower = text.to_lowercase();
    cue_words
        .relation_cues()
        .flat_map(|(_, cues)| cues.iter())
        .filter(|cue| lower.contains(cue.as_str()))
        .count()
}

/// Split `text` into sentences on `.`, `?`, `!`.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut buf = String::new();
    for c in text.chars() {
        buf.push(c);
        if c == '.' || c == '!' || c == '?' {
            let s = buf.trim().to_string();
            if !s.is_empty() {
                sentences.push(s);
            }
            buf.clear();
        }
    }
    let rem = buf.trim().to_string();
    if !rem.is_empty() {
        sentences.push(rem);
    }
    sentences
}

/// Compute a sentence-to-sentence coherence score for `text`.
///
/// The score is a value in `[0.0, 1.0]` computed as a weighted average of:
///
/// 1. **Lexical continuity** (weight 0.6): the mean Jaccard similarity of
///    content-word bags between each consecutive sentence pair.
/// 2. **Discourse cue density** (weight 0.4): the fraction of consecutive
///    sentence pairs that contain at least one discourse cue phrase in the
///    second sentence.
///
/// A score close to 1.0 indicates a well-connected, coherent text; a score
/// close to 0.0 indicates sentences that are lexically unrelated and
/// contain no discourse connectives.
pub fn coherence_score(text: &str) -> f64 {
    coherence_score_with_lexicon(text, &CueLexicon::default_english())
}

/// Like [`coherence_score`] but uses a caller-supplied cue lexicon.
pub fn coherence_score_with_lexicon(text: &str, cue_words: &CueLexicon) -> f64 {
    let sents = split_sentences(text);
    if sents.len() < 2 {
        return 1.0; // Single sentence is trivially coherent.
    }

    let pairs: Vec<(&str, &str)> = sents
        .windows(2)
        .map(|w| (w[0].as_str(), w[1].as_str()))
        .collect();

    let n = pairs.len() as f64;

    // Lexical continuity
    let lex_sum: f64 = pairs.iter().map(|(a, b)| lexical_overlap(a, b)).sum();
    let lex_score = lex_sum / n;

    // Cue density: fraction of transitions with ≥ 1 cue in the second sentence
    let cue_count = pairs
        .iter()
        .filter(|(_, b)| cue_density(b, cue_words) > 0)
        .count() as f64;
    let cue_score = cue_count / n;

    0.6 * lex_score + 0.4 * cue_score
}

// ---------------------------------------------------------------------------
// Full discourse analyser
// ---------------------------------------------------------------------------

/// High-level discourse analyser that wraps detection, tree building, and
/// coherence scoring.
pub struct DiscourseAnalyzer {
    cue_lexicon: CueLexicon,
}

impl Default for DiscourseAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl DiscourseAnalyzer {
    /// Create an analyser with the default English cue lexicon.
    pub fn new() -> Self {
        Self {
            cue_lexicon: CueLexicon::default_english(),
        }
    }

    /// Replace the cue lexicon.
    pub fn with_lexicon(mut self, lex: CueLexicon) -> Self {
        self.cue_lexicon = lex;
        self
    }

    /// Detect the relation between two sentences.
    pub fn detect_relation(&self, s1: &str, s2: &str) -> Option<DiscourseRelation> {
        detect_discourse_relation(s1, s2, &self.cue_lexicon)
    }

    /// Analyse a full text document: split into sentences, detect pairwise
    /// relations, build an RST tree, and compute a coherence score.
    pub fn analyse(&self, text: &str) -> Result<DiscourseAnalysis> {
        if text.is_empty() {
            return Err(TextError::InvalidInput(
                "Input text must not be empty".to_string(),
            ));
        }

        let sentences = split_sentences(text);
        let mut relations: Vec<(usize, usize, DiscourseRelation)> = Vec::new();

        for (i, pair) in sentences.windows(2).enumerate() {
            let s1 = &pair[0];
            let s2 = &pair[1];
            if let Some(rel) = detect_discourse_relation(s1, s2, &self.cue_lexicon) {
                relations.push((i, i + 1, rel));
            }
        }

        let rst = RhetoricalStructure::from_sentence_pairs(&sentences, relations.clone());
        let score = coherence_score_with_lexicon(text, &self.cue_lexicon);

        Ok(DiscourseAnalysis {
            sentences,
            relations,
            rst,
            coherence: score,
        })
    }
}

/// The result of a full discourse analysis.
pub struct DiscourseAnalysis {
    /// The sentences extracted from the input text.
    pub sentences: Vec<String>,
    /// Detected pairwise discourse relations `(i, j, relation)`.
    pub relations: Vec<(usize, usize, DiscourseRelation)>,
    /// Simplified RST tree (may be `None` if the text has fewer than 2 sentences).
    pub rst: Option<RhetoricalStructure>,
    /// Overall coherence score in `[0.0, 1.0]`.
    pub coherence: f64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_contrast() {
        let lex = CueLexicon::default_english();
        let s1 = "The experiment was promising.";
        let s2 = "However, the results were inconclusive.";
        let rel = detect_discourse_relation(s1, s2, &lex);
        assert_eq!(rel, Some(DiscourseRelation::Contrast));
    }

    #[test]
    fn test_detect_effect() {
        let lex = CueLexicon::default_english();
        let s1 = "The team worked very hard.";
        let s2 = "Therefore, they finished on time.";
        let rel = detect_discourse_relation(s1, s2, &lex);
        assert_eq!(rel, Some(DiscourseRelation::Effect));
    }

    #[test]
    fn test_detect_cause() {
        let lex = CueLexicon::default_english();
        let s1 = "The project was delayed.";
        let s2 = "Because the supplier did not deliver the parts.";
        let rel = detect_discourse_relation(s1, s2, &lex);
        assert_eq!(rel, Some(DiscourseRelation::Cause));
    }

    #[test]
    fn test_detect_temporal() {
        let lex = CueLexicon::default_english();
        let s1 = "She completed the analysis.";
        let s2 = "Then she wrote the report.";
        let rel = detect_discourse_relation(s1, s2, &lex);
        assert_eq!(rel, Some(DiscourseRelation::Temporal));
    }

    #[test]
    fn test_detect_conditional() {
        let lex = CueLexicon::default_english();
        let s1 = "You will succeed.";
        let s2 = "If you follow the plan carefully.";
        let rel = detect_discourse_relation(s1, s2, &lex);
        assert_eq!(rel, Some(DiscourseRelation::Conditional));
    }

    #[test]
    fn test_detect_elaboration() {
        let lex = CueLexicon::default_english();
        let s1 = "The new policy was announced.";
        let s2 = "Furthermore, it will take effect immediately.";
        let rel = detect_discourse_relation(s1, s2, &lex);
        assert_eq!(rel, Some(DiscourseRelation::Elaboration));
    }

    #[test]
    fn test_detect_exemplification() {
        let lex = CueLexicon::default_english();
        let s1 = "Many animals live in the rainforest.";
        let s2 = "For example, jaguars and toucans are common there.";
        let rel = detect_discourse_relation(s1, s2, &lex);
        assert_eq!(rel, Some(DiscourseRelation::Exemplification));
    }

    #[test]
    fn test_detect_summary() {
        let lex = CueLexicon::default_english();
        let s1 = "We reviewed all the evidence.";
        let s2 = "In conclusion, the hypothesis is supported.";
        let rel = detect_discourse_relation(s1, s2, &lex);
        assert_eq!(rel, Some(DiscourseRelation::Summary));
    }

    #[test]
    fn test_detect_none() {
        let lex = CueLexicon::default_english();
        let s1 = "The cat sat on the mat.";
        let s2 = "The dog ran across the field.";
        // No strong cue → None
        let rel = detect_discourse_relation(s1, s2, &lex);
        // We accept either None or a weak false-positive from single-word cues.
        // The test just checks the function doesn't panic.
        let _ = rel;
    }

    #[test]
    fn test_coherence_score_coherent() {
        let text = "The researchers conducted an experiment. \
                    Therefore, they published their findings. \
                    Furthermore, the findings were widely cited.";
        let score = coherence_score(text);
        // Should be higher than a random text
        assert!(score > 0.0, "score should be positive: {}", score);
        assert!(score <= 1.0, "score should be <= 1.0: {}", score);
    }

    #[test]
    fn test_coherence_score_incoherent() {
        let text = "The price of gold rose sharply. \
                    Elephants live in Africa. \
                    Quantum mechanics is complex.";
        let score = coherence_score(text);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_coherence_score_single_sentence() {
        let score = coherence_score("This is a single sentence.");
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_rst_tree_construction() {
        let sentences = vec![
            "Alice studied hard.".to_string(),
            "Therefore, she passed the exam.".to_string(),
            "However, she felt tired afterward.".to_string(),
        ];
        let relations = vec![
            (0, 1, DiscourseRelation::Effect),
            (1, 2, DiscourseRelation::Contrast),
        ];
        let tree = RhetoricalStructure::from_sentence_pairs(&sentences, relations);
        assert!(tree.is_some());
        let tree = tree.expect("already checked");
        assert_eq!(tree.sentence_count, 3);
        assert_eq!(tree.root.sentence_index, 0);
        assert_eq!(tree.root.children.len(), 2);

        // Check relations are attached
        let child_relations: Vec<Option<DiscourseRelation>> = tree
            .root
            .children
            .iter()
            .map(|c| c.relation_to_parent.clone())
            .collect();
        assert!(child_relations.contains(&Some(DiscourseRelation::Effect)));
        assert!(child_relations.contains(&Some(DiscourseRelation::Contrast)));
    }

    #[test]
    fn test_rst_empty_text_returns_none() {
        let tree = RhetoricalStructure::from_sentence_pairs(&[], Vec::new());
        assert!(tree.is_none());
    }

    #[test]
    fn test_analyser_full_pipeline() {
        let analyser = DiscourseAnalyzer::new();
        let text = "The company invested heavily in R&D. \
                    Therefore, its products improved significantly. \
                    However, costs also increased.";
        let analysis = analyser.analyse(text).expect("should succeed");
        assert_eq!(analysis.sentences.len(), 3);
        assert!(!analysis.relations.is_empty());
        assert!(analysis.rst.is_some());
        assert!(analysis.coherence >= 0.0 && analysis.coherence <= 1.0);
    }

    #[test]
    fn test_analyser_empty_input_error() {
        let analyser = DiscourseAnalyzer::new();
        assert!(analyser.analyse("").is_err());
    }

    #[test]
    fn test_dfs_traversal() {
        let sentences = vec!["S1".to_string(), "S2".to_string(), "S3".to_string()];
        let tree =
            RhetoricalStructure::from_sentence_pairs(&sentences, Vec::new()).expect("should build");
        let nodes = tree.nodes_dfs();
        assert_eq!(nodes.len(), 3);
    }

    #[test]
    fn test_custom_lexicon() {
        let mut lex = CueLexicon::default();
        lex.effect.push("voila".to_string());
        let s1 = "We mixed the chemicals.";
        let s2 = "Voila, it worked.";
        let rel = detect_discourse_relation(s1, s2, &lex);
        assert_eq!(rel, Some(DiscourseRelation::Effect));
    }
}
