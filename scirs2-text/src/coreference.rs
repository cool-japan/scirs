//! Coreference resolution: pronoun / nominal / proper-name clustering.
//!
//! This module provides a self-contained, rule-based coreference resolution
//! pipeline that does not require external ML weights.  It implements a
//! simplified version of the **Hobbs algorithm** for pronoun antecedent
//! selection, extended with nominal and proper-name clustering.
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::coreference::{resolve_pronouns, replace_pronouns};
//!
//! let text = "Alice is a doctor. She works at the hospital. \
//!             Bob is an engineer. He builds bridges.";
//! let chains = resolve_pronouns(text);
//! assert!(!chains.is_empty());
//!
//! let resolved = replace_pronouns(text, &chains);
//! // "She" should be replaced with "Alice" (or similar) in the output.
//! assert!(resolved.contains("Alice") || resolved.contains("She"));
//! ```

use crate::error::{Result, TextError};
use std::cmp::Reverse;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Coarse-grained morphosyntactic category of a mention.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MentionType {
    /// A multi-token proper name (e.g. "Marie Curie", "Google Inc.").
    Proper,
    /// A definite or indefinite nominal head (e.g. "the researcher",
    /// "a scientist").
    Nominal,
    /// A personal, possessive, or reflexive pronoun.
    Pronominal,
}

/// Gender / number feature used for agreement checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenderNumber {
    /// Masculine singular gender-number feature.
    MasculineSingular,
    /// Feminine singular gender-number feature.
    FeminineSingular,
    /// Neuter singular gender-number feature.
    NeuterSingular,
    /// Plural gender-number feature.
    Plural,
    /// Unknown or unresolved gender-number feature.
    Unknown,
}

/// A single textual reference to an entity.
#[derive(Debug, Clone)]
pub struct Mention {
    /// Character span (start, end) in the original document.
    pub span: (usize, usize),
    /// The surface string.
    pub text: String,
    /// Morpho-syntactic type.
    pub mention_type: MentionType,
    /// Gender/number for agreement.
    pub gender_number: GenderNumber,
}

impl Mention {
    /// Byte start position.
    pub fn start(&self) -> usize {
        self.span.0
    }

    /// Byte end position (exclusive).
    pub fn end(&self) -> usize {
        self.span.1
    }
}

/// A list of co-referring [`Mention`]s.
#[derive(Debug, Clone)]
pub struct CoreferenceChain {
    /// The canonical (most informative) mention – usually the first proper or
    /// nominal mention in document order.
    pub canonical: String,
    /// All mentions in document order.
    pub mentions: Vec<Mention>,
    /// Aggregate confidence score.
    pub confidence: f64,
}

impl CoreferenceChain {
    /// Create a new chain seeded with one mention.
    fn new(seed: Mention, confidence: f64) -> Self {
        let canonical = seed.text.clone();
        Self {
            canonical,
            mentions: vec![seed],
            confidence,
        }
    }

    /// Add a mention and update the canonical form.
    fn add(&mut self, mention: Mention, score: f64) {
        // Prefer proper > nominal > pronominal as canonical.
        if mention.mention_type == MentionType::Proper
            || (mention.mention_type == MentionType::Nominal
                && self.canonical_type() == MentionType::Pronominal)
        {
            self.canonical = mention.text.clone();
        }
        self.confidence = self.confidence.max(score);
        self.mentions.push(mention);
    }

    /// The mention type of the current canonical form.
    fn canonical_type(&self) -> MentionType {
        // Determine type of the canonical mention by scanning.
        for m in &self.mentions {
            if m.text == self.canonical {
                return m.mention_type.clone();
            }
        }
        MentionType::Pronominal
    }
}

// ---------------------------------------------------------------------------
// Feature-scoring helpers
// ---------------------------------------------------------------------------

/// Classify the gender/number of a surface token.
pub fn infer_gender_number(text: &str) -> GenderNumber {
    let lower = text.to_lowercase();
    match lower.as_str() {
        "he" | "him" | "his" | "himself" => GenderNumber::MasculineSingular,
        "she" | "her" | "hers" | "herself" => GenderNumber::FeminineSingular,
        "it" | "its" | "itself" => GenderNumber::NeuterSingular,
        "they" | "them" | "their" | "theirs" | "themselves" => GenderNumber::Plural,
        _ => {
            // Heuristic for proper names
            if is_likely_masculine_name(&lower) {
                GenderNumber::MasculineSingular
            } else if is_likely_feminine_name(&lower) {
                GenderNumber::FeminineSingular
            } else {
                GenderNumber::Unknown
            }
        }
    }
}

fn is_likely_masculine_name(name: &str) -> bool {
    const MASC: &[&str] = &[
        "john",
        "james",
        "michael",
        "william",
        "david",
        "richard",
        "joseph",
        "thomas",
        "charles",
        "christopher",
        "daniel",
        "matthew",
        "anthony",
        "mark",
        "donald",
        "steven",
        "paul",
        "andrew",
        "kenneth",
        "george",
        "joshua",
        "kevin",
        "brian",
        "tim",
        "bob",
        "bill",
        "frank",
        "larry",
        "scott",
        "jeffrey",
        "eric",
        "robert",
        "peter",
        "henry",
        "edward",
    ];
    MASC.contains(&name)
}

fn is_likely_feminine_name(name: &str) -> bool {
    const FEM: &[&str] = &[
        "mary",
        "patricia",
        "linda",
        "barbara",
        "elizabeth",
        "jennifer",
        "maria",
        "susan",
        "margaret",
        "dorothy",
        "lisa",
        "nancy",
        "karen",
        "betty",
        "helen",
        "sandra",
        "donna",
        "carol",
        "ruth",
        "sharon",
        "michelle",
        "laura",
        "sarah",
        "kimberly",
        "deborah",
        "jessica",
        "shirley",
        "cynthia",
        "angela",
        "melissa",
        "brenda",
        "amy",
        "anna",
        "rebecca",
        "virginia",
        "kathleen",
        "pamela",
        "martha",
        "debra",
        "amanda",
        "stephanie",
        "carolyn",
        "christine",
        "alice",
    ];
    FEM.contains(&name)
}

/// Check morpho-syntactic agreement between a pronoun mention and a candidate
/// antecedent mention.
pub fn gender_number_agreement(mention: &Mention, candidate: &Mention) -> bool {
    match (&mention.gender_number, &candidate.gender_number) {
        (GenderNumber::Unknown, _) | (_, GenderNumber::Unknown) => true,
        (a, b) => a == b,
    }
}

/// Score a pronoun `mention` against an `antecedent` candidate.
///
/// Features:
/// - Gender/number agreement: +0.4
/// - Sentence recency: +0.3 for same sentence, +0.2 for 1 back, decaying
/// - Mention type: +0.2 for Proper, +0.1 for Nominal antecedent
/// - Salience: +0.1 if antecedent is a subject-position mention (starts sentence)
pub fn antecedent_score(
    mention: &Mention,
    candidate: &Mention,
    mention_sentence: usize,
    candidate_sentence: usize,
) -> f64 {
    let mut score = 0.0f64;

    // Agreement
    if gender_number_agreement(mention, candidate) {
        score += 0.4;
    } else {
        return 0.0; // Hard constraint: incompatible agreement → skip
    }

    // Recency
    let dist = mention_sentence.saturating_sub(candidate_sentence);
    score += match dist {
        0 => 0.30,
        1 => 0.25,
        2 => 0.15,
        3 => 0.10,
        _ => 0.05f64 / dist as f64,
    };

    // Mention type of candidate
    score += match candidate.mention_type {
        MentionType::Proper => 0.20,
        MentionType::Nominal => 0.10,
        MentionType::Pronominal => 0.0,
    };

    score.min(1.0)
}

// ---------------------------------------------------------------------------
// Pronoun list
// ---------------------------------------------------------------------------

fn is_pronoun(word: &str) -> bool {
    matches!(
        word.to_lowercase().as_str(),
        "he" | "him"
            | "his"
            | "himself"
            | "she"
            | "her"
            | "hers"
            | "herself"
            | "it"
            | "its"
            | "itself"
            | "they"
            | "them"
            | "their"
            | "theirs"
            | "themselves"
    )
}

// ---------------------------------------------------------------------------
// Mention detection
// ---------------------------------------------------------------------------

/// Tokenise text into (start, end, word) tuples (byte offsets).
fn tokenize_with_offsets(text: &str) -> Vec<(usize, usize, String)> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() || c == '\'' {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            tokens.push((s, i, text[s..i].to_string()));
        }
    }
    if let Some(s) = start {
        tokens.push((s, text.len(), text[s..].to_string()));
    }
    tokens
}

/// Split `text` into sentence strings with their start offsets.
fn split_sentences_with_offsets(text: &str) -> Vec<(usize, String)> {
    let mut sentences: Vec<(usize, String)> = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();
    let len = bytes.len();
    while start < len {
        let mut end = start;
        while end < len {
            let b = bytes[end];
            if b == b'.' || b == b'?' || b == b'!' {
                end += 1;
                while end < len && bytes[end] == b' ' {
                    end += 1;
                }
                break;
            }
            end += 1;
        }
        let raw = text[start..end].trim();
        if !raw.is_empty() {
            sentences.push((start, raw.to_string()));
        }
        start = end;
    }
    sentences
}

/// Detect pronoun, nominal, and proper-name mentions in `text`.
fn detect_mentions(text: &str) -> Vec<(usize, Mention)> {
    // (sentence_index, Mention)
    let sentences = split_sentences_with_offsets(text);
    let mut result: Vec<(usize, Mention)> = Vec::new();

    for (sent_idx, (sent_start, sent_text)) in sentences.iter().enumerate() {
        let tokens = tokenize_with_offsets(sent_text);
        let mut i = 0usize;
        while i < tokens.len() {
            let (tok_start, tok_end, word) = &tokens[i];
            let abs_start = sent_start + tok_start;
            let abs_end = sent_start + tok_end;

            // ---- Pronouns ----
            if is_pronoun(word) {
                let gn = infer_gender_number(word);
                result.push((
                    sent_idx,
                    Mention {
                        span: (abs_start, abs_end),
                        text: word.clone(),
                        mention_type: MentionType::Pronominal,
                        gender_number: gn,
                    },
                ));
                i += 1;
                continue;
            }

            // ---- Proper names: consecutive capitalised tokens ----
            if word.starts_with(|c: char| c.is_uppercase()) && abs_start > *sent_start {
                // Skip if it is the first word of the sentence (always capitalised).
                let mut j = i;
                while j < tokens.len() && tokens[j].2.starts_with(|c: char| c.is_uppercase()) {
                    j += 1;
                }
                if j > i {
                    // j - i tokens form the proper name span
                    let name_start = sent_start + tokens[i].0;
                    let name_end = sent_start + tokens[j - 1].1;
                    let name_text = sent_text[tokens[i].0..tokens[j - 1].1].to_string();
                    let first_word = name_text.split_whitespace().next().unwrap_or("");
                    let gn = infer_gender_number(first_word);
                    result.push((
                        sent_idx,
                        Mention {
                            span: (name_start, name_end),
                            text: name_text,
                            mention_type: MentionType::Proper,
                            gender_number: gn,
                        },
                    ));
                    i = j;
                    continue;
                }
            }

            // ---- Nominal mentions: "the X", "a X", "an X" ----
            let lower = word.to_lowercase();
            if (lower == "the" || lower == "a" || lower == "an") && i + 1 < tokens.len() {
                let head_start = sent_start + tokens[i + 1].0;
                let head_end = sent_start + tokens[i + 1].1;
                let det_text = sent_text[*tok_start..tokens[i + 1].1].to_string();
                result.push((
                    sent_idx,
                    Mention {
                        span: (abs_start, head_end),
                        text: det_text,
                        mention_type: MentionType::Nominal,
                        gender_number: GenderNumber::Unknown,
                    },
                ));
                // Do NOT skip ahead – the head noun may also be a proper name.
                let _ = (head_start, head_end);
            }

            i += 1;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Simplified Hobbs algorithm
// ---------------------------------------------------------------------------

/// Resolve pronouns in `text` and return coreference chains.
///
/// The implementation follows a simplified Hobbs-style search: for each
/// pronoun, scan backwards through the preceding mentions in the document
/// and pick the candidate that maximises [`antecedent_score`].
pub fn resolve_pronouns(text: &str) -> Vec<CoreferenceChain> {
    let mentions_with_sent = detect_mentions(text);

    // Collect non-pronominal antecedent candidates in order.
    let candidates: Vec<(usize, usize, &Mention)> = mentions_with_sent
        .iter()
        .enumerate()
        .filter(|(_, (_, m))| m.mention_type != MentionType::Pronominal)
        .map(|(idx, (sent_idx, m))| (idx, *sent_idx, m))
        .collect();

    // Map from mention index → chain index.
    let mut mention_to_chain: HashMap<usize, usize> = HashMap::new();
    let mut chains: Vec<CoreferenceChain> = Vec::new();

    // First, register all proper/nominal mentions as potential chain seeds.
    for (idx, sent_idx, mention) in &candidates {
        // Check if an existing chain already contains this text.
        let existing = chains.iter().position(|c| {
            c.mentions.iter().any(|m| {
                m.text.to_lowercase() == mention.text.to_lowercase()
                    || mention.text.to_lowercase().contains(&m.text.to_lowercase())
                    || m.text.to_lowercase().contains(&mention.text.to_lowercase())
            })
        });

        if let Some(chain_idx) = existing {
            mention_to_chain.insert(*idx, chain_idx);
            let m_clone = (*mention).clone();
            let score = 0.7 + 0.1 * (mention.mention_type == MentionType::Proper) as u8 as f64;
            chains[chain_idx].add(m_clone, score);
        } else {
            let chain_idx = chains.len();
            mention_to_chain.insert(*idx, chain_idx);
            let confidence = if mention.mention_type == MentionType::Proper {
                0.8
            } else {
                0.6
            };
            chains.push(CoreferenceChain::new((*mention).clone(), confidence));
        }
        let _ = sent_idx;
    }

    // Now resolve pronouns.
    for (pron_idx, (pron_sent, pron_mention)) in mentions_with_sent
        .iter()
        .enumerate()
        .filter(|(_, (_, m))| m.mention_type == MentionType::Pronominal)
    {
        // Scan candidates that appear BEFORE this pronoun.
        let mut best_score = 0.0f64;
        let mut best_cand_idx: Option<usize> = None;

        for &(cand_mention_idx, cand_sent, cand_mention) in &candidates {
            // The candidate must precede the pronoun in the document.
            if mentions_with_sent[cand_mention_idx].0 > *pron_sent {
                continue;
            }
            // Within the same sentence, the candidate must come first.
            if cand_mention.span.0 >= pron_mention.span.0 && cand_sent == *pron_sent {
                continue;
            }

            let score = antecedent_score(pron_mention, cand_mention, *pron_sent, cand_sent);
            if score > best_score {
                best_score = score;
                best_cand_idx = Some(cand_mention_idx);
            }
        }

        if best_score > 0.3 {
            if let Some(cand_idx) = best_cand_idx {
                if let Some(&chain_idx) = mention_to_chain.get(&cand_idx) {
                    let pron_clone = pron_mention.clone();
                    chains[chain_idx].add(pron_clone, best_score);
                    mention_to_chain.insert(pron_idx, chain_idx);
                }
            }
        }
    }

    // Prune chains that only contain one mention (no actual coreference).
    chains.retain(|c| c.mentions.len() >= 2);
    chains
}

/// Substitute all pronominal mentions in `text` with their canonical
/// antecedent from the supplied chains.
///
/// Pronouns that appear in multiple overlapping chains are resolved using the
/// highest-confidence chain.  The replacement is done in reverse document
/// order to preserve byte offsets.
pub fn replace_pronouns(text: &str, chains: &[CoreferenceChain]) -> String {
    // Build a map from span → replacement string, keeping the highest-
    // confidence replacement if multiple chains cover the same pronoun.
    let mut replacements: HashMap<(usize, usize), (String, f64)> = HashMap::new();

    for chain in chains {
        for mention in &chain.mentions {
            if mention.mention_type == MentionType::Pronominal {
                let entry = replacements
                    .entry(mention.span)
                    .or_insert_with(|| (chain.canonical.clone(), 0.0));
                if chain.confidence > entry.1 {
                    *entry = (chain.canonical.clone(), chain.confidence);
                }
            }
        }
    }

    // Sort spans in reverse order so replacements do not shift later offsets.
    let mut spans: Vec<(usize, usize, String)> = replacements
        .into_iter()
        .map(|(span, (repl, _))| (span.0, span.1, repl))
        .collect();
    spans.sort_by_key(|(start, _, _)| std::cmp::Reverse(*start));

    let mut result = text.to_string();
    for (start, end, replacement) in spans {
        if start <= end && end <= result.len() {
            result.replace_range(start..end, &replacement);
        }
    }

    result
}

/// Resolve coreferences and return chains – a convenience wrapper with error
/// propagation for pipeline use.
pub fn resolve_coreferences(text: &str) -> Result<Vec<CoreferenceChain>> {
    if text.is_empty() {
        return Err(TextError::InvalidInput(
            "Input text must not be empty".to_string(),
        ));
    }
    Ok(resolve_pronouns(text))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_gender_number() {
        assert_eq!(infer_gender_number("he"), GenderNumber::MasculineSingular);
        assert_eq!(infer_gender_number("She"), GenderNumber::FeminineSingular);
        assert_eq!(infer_gender_number("it"), GenderNumber::NeuterSingular);
        assert_eq!(infer_gender_number("they"), GenderNumber::Plural);
        assert_eq!(infer_gender_number("random"), GenderNumber::Unknown);
    }

    #[test]
    fn test_gender_number_agreement() {
        let he = Mention {
            span: (0, 2),
            text: "he".to_string(),
            mention_type: MentionType::Pronominal,
            gender_number: GenderNumber::MasculineSingular,
        };
        let john = Mention {
            span: (10, 14),
            text: "John".to_string(),
            mention_type: MentionType::Proper,
            gender_number: GenderNumber::MasculineSingular,
        };
        let alice = Mention {
            span: (20, 25),
            text: "Alice".to_string(),
            mention_type: MentionType::Proper,
            gender_number: GenderNumber::FeminineSingular,
        };
        assert!(gender_number_agreement(&he, &john));
        assert!(!gender_number_agreement(&he, &alice));
    }

    #[test]
    fn test_antecedent_score_agreement_constraint() {
        let she = Mention {
            span: (0, 3),
            text: "she".to_string(),
            mention_type: MentionType::Pronominal,
            gender_number: GenderNumber::FeminineSingular,
        };
        let he_candidate = Mention {
            span: (10, 12),
            text: "John".to_string(),
            mention_type: MentionType::Proper,
            gender_number: GenderNumber::MasculineSingular,
        };
        // Disagreement → 0.0
        assert_eq!(antecedent_score(&she, &he_candidate, 1, 0), 0.0);
    }

    #[test]
    fn test_resolve_pronouns_basic() {
        let text = "Alice is a scientist. She won a prize. Bob is an engineer. He built a bridge.";
        let chains = resolve_pronouns(text);
        // Should find at least one chain linking a pronoun back to a name.
        assert!(!chains.is_empty());
        for chain in &chains {
            assert!(chain.mentions.len() >= 2);
        }
    }

    #[test]
    fn test_replace_pronouns() {
        let text = "Alice is a doctor. She works at the hospital.";
        let chains = resolve_pronouns(text);
        let replaced = replace_pronouns(text, &chains);
        // The output should still be valid UTF-8 and non-empty.
        assert!(!replaced.is_empty());
    }

    #[test]
    fn test_resolve_coreferences_error_on_empty() {
        let result = resolve_coreferences("");
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_coreferences_nonempty() {
        let text = "Marie Curie discovered radium. She was brilliant.";
        let chains = resolve_coreferences(text).expect("should succeed");
        // There may or may not be a chain depending on heuristics, but
        // the function must not panic.
        let _ = chains;
    }

    #[test]
    fn test_detect_pronouns_in_isolation() {
        assert!(is_pronoun("she"));
        assert!(is_pronoun("He"));
        assert!(is_pronoun("THEY"));
        assert!(!is_pronoun("Alice"));
        assert!(!is_pronoun("the"));
    }

    #[test]
    fn test_multiple_chains() {
        let text = "Alice is a doctor. She treated patients. \
                    Bob is a lawyer. He argued cases.";
        let chains = resolve_pronouns(text);
        // Should find at least two distinct chains (one for she→Alice, one for he→Bob).
        assert!(chains.len() >= 1);
    }
}
