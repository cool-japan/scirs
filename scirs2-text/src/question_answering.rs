//! Extractive question answering over plain-text documents.
//!
//! This module provides rule-based, overlap-based, and TF-IDF-based extractive
//! QA that locates the most plausible answer span for a given question inside a
//! context document – all without external ML weights.
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::question_answering::{QAContext, QAMethod, extract_answer};
//!
//! let doc = "Marie Curie was born in Warsaw on 7 November 1867. \
//!            She won two Nobel Prizes.";
//! let spans = extract_answer("When was Marie Curie born?", doc, 3);
//! assert!(!spans.is_empty());
//! ```

use crate::error::{Result, TextError};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Method used to score candidate answer spans.
#[derive(Debug, Clone, PartialEq)]
pub enum QAMethod {
    /// Rank candidate sentences by TF-IDF overlap with the question, then
    /// extract the best named-entity-aware span within the top sentence.
    TfIdf,
    /// Rank by exact bigram overlap between the question and each sentence.
    BigramOverlap,
    /// Use cosine similarity of averaged word-embedding vectors (when
    /// embeddings are provided to `QAContext`).
    WordEmbeddingMatch,
}

/// A single question type category inferred from the question wh-word.
#[derive(Debug, Clone, PartialEq)]
pub enum QuestionType {
    /// "Who" – expects a person / organisation
    Who,
    /// "What" – general entity or definition
    What,
    /// "When" – temporal expression
    When,
    /// "Where" – location
    Where,
    /// "Why" – reason / cause
    Why,
    /// "How" – manner / quantity
    How,
    /// Could not be classified
    Unknown,
}

/// An extracted answer span with provenance and confidence.
#[derive(Debug, Clone)]
pub struct AnswerSpan {
    /// Byte-level start offset in the *original* document text.
    pub start: usize,
    /// Byte-level end offset (exclusive) in the *original* document text.
    pub end: usize,
    /// The answer text itself.
    pub text: String,
    /// Confidence score in [0, 1].
    pub confidence: f64,
    /// Which sentence (0-indexed) the span was drawn from.
    pub sentence_index: usize,
}

/// A tokenised, indexed context document ready for repeated QA queries.
pub struct QAContext {
    /// Original document text.
    pub text: String,
    /// Sentences with their byte offsets in the original text.
    sentences: Vec<SentenceRecord>,
    /// Optional word embeddings (word → fixed-length float vector).
    embeddings: Option<HashMap<String, Vec<f64>>>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// A sentence together with its position inside the document.
#[derive(Debug, Clone)]
struct SentenceRecord {
    text: String,
    start: usize,
    tokens: Vec<String>,
}

/// Tokenise a text slice into lowercase alphabetic/numeric tokens.
fn simple_tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// English stop-words used to suppress common terms in TF-IDF.
fn stop_words() -> HashSet<&'static str> {
    [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "shall", "should", "may", "might", "must", "can",
        "could", "to", "of", "in", "on", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "from", "up", "down",
        "out", "off", "over", "under", "again", "further", "then", "once", "and", "but", "or",
        "nor", "so", "yet", "both", "either", "neither", "not", "only", "own", "same", "than",
        "too", "very", "just", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "its", "our", "their", "what", "which", "who", "whom",
        "this", "that", "these", "those", "am", "s", "t",
    ]
    .iter()
    .cloned()
    .collect()
}

/// Classify the question type from the first recognised wh-word.
pub fn classify_question(question: &str) -> QuestionType {
    let lower = question.to_lowercase();
    // Check word boundaries roughly by looking for whole-word occurrences.
    for word in lower.split_whitespace() {
        let w = word.trim_matches(|c: char| !c.is_alphabetic());
        match w {
            "who" | "whose" | "whom" => return QuestionType::Who,
            "when" => return QuestionType::When,
            "where" => return QuestionType::Where,
            "why" => return QuestionType::Why,
            "how" => return QuestionType::How,
            "what" | "which" => return QuestionType::What,
            _ => {}
        }
    }
    QuestionType::Unknown
}

// ---------------------------------------------------------------------------
// QAContext implementation
// ---------------------------------------------------------------------------

impl QAContext {
    /// Build a `QAContext` from a plain document string.
    pub fn new(text: &str) -> Self {
        let sentences = Self::split_sentences(text);
        Self {
            text: text.to_string(),
            sentences,
            embeddings: None,
        }
    }

    /// Attach word embeddings for `WordEmbeddingMatch` queries.
    pub fn with_embeddings(mut self, embeddings: HashMap<String, Vec<f64>>) -> Self {
        self.embeddings = Some(embeddings);
        self
    }

    // ------------------------------------------------------------------
    // Sentence splitting
    // ------------------------------------------------------------------

    fn split_sentences(text: &str) -> Vec<SentenceRecord> {
        let mut records = Vec::new();
        let mut start = 0usize;
        let bytes = text.as_bytes();
        let len = bytes.len();

        while start < len {
            // Find the next sentence boundary: '.', '?', '!' followed by
            // whitespace or end-of-string.
            let mut end = start;
            while end < len {
                let b = bytes[end];
                if b == b'.' || b == b'?' || b == b'!' {
                    // Make sure we are at a valid char boundary before slicing.
                    end += 1;
                    // Consume any trailing whitespace so the next sentence
                    // starts cleanly.
                    while end < len && bytes[end] == b' ' {
                        end += 1;
                    }
                    break;
                }
                end += 1;
            }

            let raw = text[start..end].trim();
            if !raw.is_empty() {
                records.push(SentenceRecord {
                    text: raw.to_string(),
                    start,
                    tokens: simple_tokenize(raw),
                });
            }
            start = end;
        }

        records
    }

    // ------------------------------------------------------------------
    // TF-IDF ranking helpers
    // ------------------------------------------------------------------

    /// Build IDF table over the stored sentences.
    fn build_idf(&self) -> HashMap<String, f64> {
        let n = self.sentences.len() as f64;
        let mut df: HashMap<String, usize> = HashMap::new();
        for sent in &self.sentences {
            let unique: HashSet<&String> = sent.tokens.iter().collect();
            for tok in unique {
                *df.entry(tok.clone()).or_insert(0) += 1;
            }
        }
        df.into_iter()
            .map(|(t, d)| (t, (1.0 + n / (1.0 + d as f64)).ln()))
            .collect()
    }

    /// Score a single sentence against query tokens using TF-IDF cosine.
    fn tfidf_score(
        query_tokens: &[String],
        sentence: &SentenceRecord,
        idf: &HashMap<String, f64>,
        stops: &HashSet<&'static str>,
    ) -> f64 {
        // Build TF for the sentence
        let mut sent_tf: HashMap<String, f64> = HashMap::new();
        for tok in &sentence.tokens {
            *sent_tf.entry(tok.clone()).or_insert(0.0) += 1.0;
        }
        let sent_len = sentence.tokens.len().max(1) as f64;

        let mut dot = 0.0f64;
        let mut q_norm = 0.0f64;
        let mut s_norm = 0.0f64;

        // Compute vectors over query tokens only (sparse dot product)
        let query_freq: HashMap<&String, f64> = {
            let mut m = HashMap::new();
            for t in query_tokens {
                if !stops.contains(t.as_str()) {
                    *m.entry(t).or_insert(0.0) += 1.0;
                }
            }
            m
        };

        for (tok, &qf) in &query_freq {
            let idf_val = idf.get(*tok).copied().unwrap_or(0.0);
            let q_tfidf = (qf / query_tokens.len().max(1) as f64) * idf_val;
            let s_tfidf = sent_tf.get(*tok).copied().unwrap_or(0.0) / sent_len * idf_val;
            dot += q_tfidf * s_tfidf;
            q_norm += q_tfidf * q_tfidf;
            s_norm += s_tfidf * s_tfidf;
        }

        if q_norm > 0.0 && s_norm > 0.0 {
            dot / (q_norm.sqrt() * s_norm.sqrt())
        } else {
            0.0
        }
    }

    // ------------------------------------------------------------------
    // Bigram overlap
    // ------------------------------------------------------------------

    fn bigram_overlap_score(query_tokens: &[String], sentence: &SentenceRecord) -> f64 {
        if query_tokens.len() < 2 || sentence.tokens.len() < 2 {
            // Fall back to unigram overlap
            let q_set: HashSet<&String> = query_tokens.iter().collect();
            let s_set: HashSet<&String> = sentence.tokens.iter().collect();
            let inter = q_set.intersection(&s_set).count();
            return inter as f64 / q_set.len().max(1) as f64;
        }

        let q_bigrams: HashSet<(&String, &String)> =
            query_tokens.windows(2).map(|w| (&w[0], &w[1])).collect();
        let s_bigrams: HashSet<(&String, &String)> =
            sentence.tokens.windows(2).map(|w| (&w[0], &w[1])).collect();

        let inter = q_bigrams.intersection(&s_bigrams).count();
        let union = q_bigrams.union(&s_bigrams).count();
        if union == 0 {
            0.0
        } else {
            inter as f64 / union as f64
        }
    }

    // ------------------------------------------------------------------
    // Embedding cosine
    // ------------------------------------------------------------------

    fn embedding_score(
        query_tokens: &[String],
        sentence: &SentenceRecord,
        embeddings: &HashMap<String, Vec<f64>>,
    ) -> f64 {
        let q_vec = Self::average_embedding(query_tokens, embeddings);
        let s_vec = Self::average_embedding(&sentence.tokens, embeddings);
        match (q_vec, s_vec) {
            (Some(q), Some(s)) => cosine_sim(&q, &s),
            _ => 0.0,
        }
    }

    fn average_embedding(
        tokens: &[String],
        embeddings: &HashMap<String, Vec<f64>>,
    ) -> Option<Vec<f64>> {
        let vecs: Vec<&Vec<f64>> = tokens.iter().filter_map(|t| embeddings.get(t)).collect();
        if vecs.is_empty() {
            return None;
        }
        let dim = vecs[0].len();
        let mut sum = vec![0.0f64; dim];
        for v in &vecs {
            for (s, &x) in sum.iter_mut().zip(v.iter()) {
                *s += x;
            }
        }
        let n = vecs.len() as f64;
        Some(sum.into_iter().map(|x| x / n).collect())
    }

    // ------------------------------------------------------------------
    // Named-entity-aware span extraction
    // ------------------------------------------------------------------

    /// Extract the best sub-span from a sentence for the given question type.
    ///
    /// Applies lightweight regex heuristics to prefer date/time/person/location
    /// phrases when the question type suggests one.
    fn extract_best_span(
        sentence: &SentenceRecord,
        q_type: &QuestionType,
        doc_text: &str,
    ) -> Option<(usize, usize, f64)> {
        // Patterns: (regex_literal, applies_to_question_types, bonus)
        let patterns: &[(&str, &[QuestionType], f64)] = &[
            // Dates / years
            (
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
                &[QuestionType::When],
                0.3,
            ),
            (
                r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
                &[QuestionType::When],
                0.3,
            ),
            (r"\b\d{4}\b", &[QuestionType::When], 0.15),
            // Capitalised phrases (likely NEs)
            (
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
                &[QuestionType::Who, QuestionType::Where, QuestionType::What],
                0.2,
            ),
            // Location indicators
            (
                r"\bin\s+[A-Z][a-zA-Z]+(?:,\s*[A-Z][a-zA-Z]+)*\b",
                &[QuestionType::Where],
                0.25,
            ),
        ];

        // Find the sentence's absolute offset in the original doc.
        let sent_start_in_doc = sentence.start;
        let sent_end_in_doc = sent_start_in_doc + sentence.text.len();
        // Clamp to doc bounds to be safe.
        let sent_end_in_doc = sent_end_in_doc.min(doc_text.len());

        let mut best: Option<(usize, usize, f64)> = None;

        for (pattern_str, qtypes, bonus) in patterns {
            let applies = qtypes.iter().any(|qt| qt == q_type);
            if !applies && *q_type != QuestionType::Unknown && *q_type != QuestionType::What {
                continue;
            }

            // We use a manual match rather than the regex crate to keep
            // compile times down – but we DO use the regex crate which is
            // already a dependency.  Build lazily.
            if let Ok(re) = regex::Regex::new(pattern_str) {
                for m in re.find_iter(&sentence.text) {
                    let abs_start = sent_start_in_doc + m.start();
                    let abs_end = sent_start_in_doc + m.end();
                    // Check bounds
                    if abs_end > sent_end_in_doc {
                        continue;
                    }
                    let score = 0.5 + bonus;
                    if best.is_none_or(|(_, _, s)| score > s) {
                        best = Some((abs_start, abs_end, score));
                    }
                }
            }
        }

        best
    }

    // ------------------------------------------------------------------
    // Core QA routine
    // ------------------------------------------------------------------

    /// Find the single best answer span for `question` using the given method.
    pub fn find_answer_span(&self, question: &str, method: QAMethod) -> Result<Option<AnswerSpan>> {
        if self.sentences.is_empty() {
            return Ok(None);
        }

        let q_tokens = simple_tokenize(question);
        if q_tokens.is_empty() {
            return Err(TextError::InvalidInput(
                "Question must not be empty".to_string(),
            ));
        }

        let q_type = classify_question(question);
        let stops = stop_words();

        // Score every sentence.
        let mut scored: Vec<(usize, f64)> = self
            .sentences
            .iter()
            .enumerate()
            .map(|(i, sent)| {
                let base = match &method {
                    QAMethod::TfIdf => {
                        let idf = self.build_idf();
                        Self::tfidf_score(&q_tokens, sent, &idf, &stops)
                    }
                    QAMethod::BigramOverlap => Self::bigram_overlap_score(&q_tokens, sent),
                    QAMethod::WordEmbeddingMatch => {
                        if let Some(emb) = &self.embeddings {
                            Self::embedding_score(&q_tokens, sent, emb)
                        } else {
                            // Fallback to bigram overlap if no embeddings.
                            Self::bigram_overlap_score(&q_tokens, sent)
                        }
                    }
                };
                (i, base)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (best_idx, base_score) = scored[0];
        if base_score <= 0.0 {
            return Ok(None);
        }

        let best_sent = &self.sentences[best_idx];

        // Try to extract a named-entity-aware sub-span.
        let span = Self::extract_best_span(best_sent, &q_type, &self.text);

        let answer = if let Some((start, end, ne_bonus)) = span {
            if start < end && end <= self.text.len() {
                AnswerSpan {
                    start,
                    end,
                    text: self.text[start..end].to_string(),
                    confidence: (base_score + ne_bonus).min(1.0),
                    sentence_index: best_idx,
                }
            } else {
                // Fall back to full sentence
                let start = best_sent.start;
                let end = (best_sent.start + best_sent.text.len()).min(self.text.len());
                AnswerSpan {
                    start,
                    end,
                    text: best_sent.text.clone(),
                    confidence: base_score,
                    sentence_index: best_idx,
                }
            }
        } else {
            // Return the whole sentence as the answer span.
            let start = best_sent.start;
            let end = (best_sent.start + best_sent.text.len()).min(self.text.len());
            AnswerSpan {
                start,
                end,
                text: best_sent.text.clone(),
                confidence: base_score,
                sentence_index: best_idx,
            }
        };

        Ok(Some(answer))
    }

    // ------------------------------------------------------------------
    // Multi-answer extraction
    // ------------------------------------------------------------------

    /// Rank all sentences and return the top-`k` answer spans.
    pub fn find_top_k(
        &self,
        question: &str,
        method: QAMethod,
        k: usize,
    ) -> Result<Vec<AnswerSpan>> {
        if self.sentences.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let q_tokens = simple_tokenize(question);
        if q_tokens.is_empty() {
            return Err(TextError::InvalidInput(
                "Question must not be empty".to_string(),
            ));
        }

        let q_type = classify_question(question);
        let stops = stop_words();
        let idf = self.build_idf();

        let mut scored: Vec<(usize, f64)> = self
            .sentences
            .iter()
            .enumerate()
            .map(|(i, sent)| {
                let base = match &method {
                    QAMethod::TfIdf => Self::tfidf_score(&q_tokens, sent, &idf, &stops),
                    QAMethod::BigramOverlap => Self::bigram_overlap_score(&q_tokens, sent),
                    QAMethod::WordEmbeddingMatch => {
                        if let Some(emb) = &self.embeddings {
                            Self::embedding_score(&q_tokens, sent, emb)
                        } else {
                            Self::bigram_overlap_score(&q_tokens, sent)
                        }
                    }
                };
                (i, base)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut answers = Vec::new();
        for (idx, base_score) in scored.into_iter().take(k) {
            if base_score <= 0.0 {
                break;
            }
            let sent = &self.sentences[idx];
            let span = Self::extract_best_span(sent, &q_type, &self.text);

            let answer = if let Some((start, end, bonus)) = span {
                if start < end && end <= self.text.len() {
                    AnswerSpan {
                        start,
                        end,
                        text: self.text[start..end].to_string(),
                        confidence: (base_score + bonus).min(1.0),
                        sentence_index: idx,
                    }
                } else {
                    let s = sent.start;
                    let e = (sent.start + sent.text.len()).min(self.text.len());
                    AnswerSpan {
                        start: s,
                        end: e,
                        text: sent.text.clone(),
                        confidence: base_score,
                        sentence_index: idx,
                    }
                }
            } else {
                let s = sent.start;
                let e = (sent.start + sent.text.len()).min(self.text.len());
                AnswerSpan {
                    start: s,
                    end: e,
                    text: sent.text.clone(),
                    confidence: base_score,
                    sentence_index: idx,
                }
            };

            answers.push(answer);
        }

        Ok(answers)
    }
}

// ---------------------------------------------------------------------------
// Free-standing helpers
// ---------------------------------------------------------------------------

/// Rank sentences in `context_sentences` by TF-IDF similarity to `query_tokens`.
///
/// Returns a vector of scores parallel to `context_sentences`.
pub fn tf_idf_similarity(query_tokens: &[String], context_sentences: &[Vec<String>]) -> Vec<f64> {
    if context_sentences.is_empty() || query_tokens.is_empty() {
        return vec![0.0; context_sentences.len()];
    }

    let n = context_sentences.len() as f64;
    let stops = stop_words();

    // Build IDF
    let mut df: HashMap<String, usize> = HashMap::new();
    for sent in context_sentences {
        let unique: HashSet<&String> = sent.iter().collect();
        for tok in unique {
            *df.entry(tok.clone()).or_insert(0) += 1;
        }
    }
    let idf: HashMap<String, f64> = df
        .into_iter()
        .map(|(t, d)| (t, (1.0 + n / (1.0 + d as f64)).ln()))
        .collect();

    context_sentences
        .iter()
        .map(|sent| {
            let record = SentenceRecord {
                text: sent.join(" "),
                start: 0,
                tokens: sent.clone(),
            };
            QAContext::tfidf_score(query_tokens, &record, &idf, &stops)
        })
        .collect()
}

/// Convenience function: extract up to `top_k` answers from `document` for
/// `question` using TF-IDF by default.
pub fn extract_answer(question: &str, document: &str, top_k: usize) -> Vec<AnswerSpan> {
    let ctx = QAContext::new(document);
    ctx.find_top_k(question, QAMethod::TfIdf, top_k)
        .unwrap_or_default()
}

/// Cosine similarity between two float vectors (helper shared with this module).
fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = "Marie Curie was born in Warsaw on 7 November 1867. \
         She conducted pioneering research on radioactivity. \
         In 1903 she won the Nobel Prize in Physics. \
         She also won the Nobel Prize in Chemistry in 1911. \
         Paris became her home after she moved from Poland.";

    #[test]
    fn test_classify_question() {
        assert_eq!(
            classify_question("Who discovered radium?"),
            QuestionType::Who
        );
        assert_eq!(classify_question("When was she born?"), QuestionType::When);
        assert_eq!(
            classify_question("Where did she live?"),
            QuestionType::Where
        );
        assert_eq!(classify_question("How did she win?"), QuestionType::How);
        assert_eq!(
            classify_question("What is radioactivity?"),
            QuestionType::What
        );
        assert_eq!(
            classify_question("Why is science important?"),
            QuestionType::Why
        );
    }

    #[test]
    fn test_extract_answer_tfidf() {
        let answers = extract_answer("When was Marie Curie born?", DOC, 3);
        assert!(!answers.is_empty());
        // The birth-date sentence should be highly ranked.
        assert!(
            answers[0].text.to_lowercase().contains("born")
                || answers[0].text.contains("1867")
                || answers[0].text.to_lowercase().contains("november")
        );
        assert!(answers[0].confidence > 0.0);
    }

    #[test]
    fn test_find_answer_span_bigram() {
        let ctx = QAContext::new(DOC);
        let ans = ctx
            .find_answer_span(
                "What prize did she win in Physics?",
                QAMethod::BigramOverlap,
            )
            .expect("QA failed");
        assert!(ans.is_some());
        let span = ans.expect("should have a span");
        assert!(
            span.text.to_lowercase().contains("physics")
                || span.text.contains("1903")
                || span.text.to_lowercase().contains("prize")
        );
    }

    #[test]
    fn test_find_top_k() {
        let ctx = QAContext::new(DOC);
        let answers = ctx
            .find_top_k("Nobel Prize", QAMethod::TfIdf, 2)
            .expect("top-k failed");
        assert!(answers.len() <= 2);
    }

    #[test]
    fn test_embedding_fallback_without_embeddings() {
        let ctx = QAContext::new(DOC);
        // Without embeddings, WordEmbeddingMatch falls back to bigram overlap.
        let ans = ctx
            .find_answer_span("Where did she live?", QAMethod::WordEmbeddingMatch)
            .expect("QA failed");
        // Just verifies it does not panic and returns a valid result.
        let _ = ans;
    }

    #[test]
    fn test_tf_idf_similarity_standalone() {
        let query = simple_tokenize("Nobel Prize winner");
        let sentences: Vec<Vec<String>> = vec![
            simple_tokenize("She won the Nobel Prize in Physics"),
            simple_tokenize("Marie Curie was born in Warsaw"),
            simple_tokenize("Nobel Prize in Chemistry was awarded"),
        ];
        let scores = tf_idf_similarity(&query, &sentences);
        assert_eq!(scores.len(), 3);
        // Nobel-prize sentences should score higher than the birth sentence.
        assert!(scores[0] > scores[1] || scores[2] > scores[1]);
    }

    #[test]
    fn test_answer_span_bounds() {
        let ctx = QAContext::new(DOC);
        let answers = ctx
            .find_top_k("Marie Curie radioactivity", QAMethod::TfIdf, 5)
            .expect("failed");
        for ans in answers {
            // start and end must be valid byte offsets inside the document.
            assert!(ans.start <= ans.end);
            assert!(ans.end <= DOC.len());
            assert_eq!(ans.text, DOC[ans.start..ans.end]);
        }
    }

    #[test]
    fn test_empty_document() {
        let ctx = QAContext::new("");
        let ans = ctx
            .find_answer_span("Who is here?", QAMethod::TfIdf)
            .expect("QA failed");
        assert!(ans.is_none());
    }
}
