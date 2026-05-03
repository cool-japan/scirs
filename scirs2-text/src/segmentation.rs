//! Text Chunking and Sentence Segmentation (`segmentation.rs`)
//!
//! Provides:
//!
//! - [`SentenceSegmenter`] -- boundary-aware sentence splitter with
//!   abbreviation handling
//! - [`TextChunker`] -- sliding-window chunker with configurable overlap,
//!   optionally respecting sentence boundaries
//! - [`TextChunk`] -- metadata-rich chunk descriptor

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Built-in English abbreviations
// ---------------------------------------------------------------------------

fn builtin_abbreviations() -> HashSet<String> {
    [
        // Titles
        "Mr", "Mrs", "Ms", "Miss", "Dr", "Prof", "Rev", "Gen", "Col", "Capt", "Lt", "Sgt", "Cpl",
        "Pte", "Sr", "Jr", // Geographic
        "St", "Ave", "Blvd", "Rd", "Ln", "Ct", "Pl", "Mt", "Ft", // Time / month
        "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Mon", "Tue",
        "Wed", "Thu", "Fri", "Sat", "Sun", // Miscellaneous
        "etc", "vs", "approx", "est", "dept", "corp", "co", "inc", "Fig", "fig", "Vol", "vol",
        "No", "Nos", "pp", "Ch", "Sec", "e.g", "i.e", "et", "al", "n.b", "N.B", "Esq",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

// ---------------------------------------------------------------------------
// SentenceSegmenter
// ---------------------------------------------------------------------------

/// Sentence boundary detector.
///
/// Uses heuristic rules:
/// 1. `.`, `!`, `?` followed by whitespace and an upper-case letter (or end of
///    string) are candidate boundaries.
/// 2. Tokens ending with a known abbreviation are NOT treated as sentence
///    boundaries.
/// 3. Ellipsis (`...`) is NOT treated as a boundary.
///
/// # Example
///
/// ```rust
/// use scirs2_text::segmentation::SentenceSegmenter;
///
/// let seg = SentenceSegmenter::new();
/// let sentences = seg.segment("Hello, Dr. Smith. How are you today?");
/// assert_eq!(sentences.len(), 2);
/// ```
pub struct SentenceSegmenter {
    /// Known abbreviations (without trailing period).
    abbreviations: HashSet<String>,
    /// Minimum byte length of a sentence (shorter candidates are merged).
    pub min_sentence_len: usize,
}

impl Default for SentenceSegmenter {
    fn default() -> Self {
        Self::new()
    }
}

impl SentenceSegmenter {
    /// Create a new segmenter with built-in English abbreviations.
    pub fn new() -> Self {
        Self {
            abbreviations: builtin_abbreviations(),
            min_sentence_len: 10,
        }
    }

    /// Create a segmenter with a custom abbreviation list.
    pub fn with_abbreviations(abbrevs: Vec<String>) -> Self {
        let mut set = builtin_abbreviations();
        for a in abbrevs {
            set.insert(a);
        }
        Self {
            abbreviations: set,
            min_sentence_len: 10,
        }
    }

    /// Segment `text` into sentence string slices.
    pub fn segment<'a>(&self, text: &'a str) -> Vec<&'a str> {
        if text.trim().is_empty() {
            return Vec::new();
        }
        let boundaries = self.find_boundaries(text);
        let mut result: Vec<&'a str> = Vec::new();
        let mut start = 0;

        for end in boundaries {
            let slice = text[start..end].trim();
            if !slice.is_empty() {
                result.push(slice);
            }
            start = end;
        }

        let tail = text[start..].trim();
        if !tail.is_empty() {
            result.push(tail);
        }

        result
    }

    /// Segment `text` and return owned `String`s.
    pub fn segment_owned(&self, text: &str) -> Vec<String> {
        if text.trim().is_empty() {
            return Vec::new();
        }
        let raw: Vec<String> = self.segment(text).iter().map(|s| s.to_string()).collect();

        // Merge very short fragments into the previous sentence.
        let mut result: Vec<String> = Vec::new();
        for s in raw {
            if s.len() < self.min_sentence_len && !result.is_empty() {
                if let Some(last) = result.last_mut() {
                    last.push(' ');
                    last.push_str(&s);
                }
            } else {
                result.push(s);
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn find_boundaries(&self, text: &str) -> Vec<usize> {
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let n = chars.len();
        let mut boundaries: Vec<usize> = Vec::new();

        let mut i = 0usize;
        while i < n {
            let (byte_pos, ch) = chars[i];

            if ch == '.' || ch == '!' || ch == '?' {
                // Check for ellipsis (...)
                if ch == '.' && i + 2 < n && chars[i + 1].1 == '.' && chars[i + 2].1 == '.' {
                    i += 3;
                    continue;
                }

                // Check if this period follows an abbreviation.
                if ch == '.' && self.is_abbreviation_period(text, byte_pos) {
                    i += 1;
                    continue;
                }

                // Check if period is inside a number like 3.14
                if ch == '.' && self.is_decimal_period(text, byte_pos) {
                    i += 1;
                    continue;
                }

                // Find end of punctuation cluster
                let mut end_i = i + 1;
                while end_i < n
                    && (chars[end_i].1 == '!' || chars[end_i].1 == '?' || chars[end_i].1 == '.')
                {
                    end_i += 1;
                }

                let boundary_byte = if end_i < n {
                    chars[end_i].0
                } else {
                    text.len()
                };

                if self.is_sentence_boundary(text, boundary_byte) {
                    boundaries.push(boundary_byte);
                }

                i = end_i;
                continue;
            }

            i += 1;
        }

        boundaries
    }

    fn is_abbreviation_period(&self, text: &str, period_byte: usize) -> bool {
        let prefix = &text[..period_byte];
        let word = prefix
            .rsplit(|c: char| !c.is_alphabetic() && c != '.')
            .next()
            .unwrap_or("");
        self.abbreviations.contains(word)
            || self.abbreviations.contains(&word.to_lowercase())
            || (word.len() == 1 && word.chars().next().is_some_and(|c| c.is_uppercase()))
    }

    fn is_decimal_period(&self, text: &str, period_byte: usize) -> bool {
        // Check if preceded by a digit and followed by a digit (e.g., 3.14)
        let before = text[..period_byte]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_digit());
        let after = text[period_byte + 1..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_digit());
        before && after
    }

    fn is_sentence_boundary(&self, text: &str, pos: usize) -> bool {
        if pos >= text.len() {
            return true;
        }
        let after = &text[pos..];
        let trimmed = after.trim_start();
        if trimmed.is_empty() {
            return true;
        }
        trimmed.chars().next().is_some_and(|c| {
            c.is_uppercase()
                || c.is_ascii_digit()
                || matches!(c, '"' | '\'' | '(' | '[' | '\u{201C}' | '\u{2018}')
        })
    }
}

// ---------------------------------------------------------------------------
// TextChunker
// ---------------------------------------------------------------------------

/// A chunk of text with positional metadata.
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// The chunk text.
    pub text: String,
    /// Byte offset of the chunk start in the source text.
    pub start: usize,
    /// Byte offset of the chunk end (exclusive) in the source text.
    pub end: usize,
    /// Zero-based index of this chunk.
    pub chunk_index: usize,
    /// Total number of chunks produced.
    pub total_chunks: usize,
}

/// Sliding-window text chunker.
///
/// # Example
///
/// ```rust
/// use scirs2_text::segmentation::TextChunker;
///
/// let chunker = TextChunker::new(10, 2);
/// let chunks = chunker.chunk("Rust is fast. Rust is safe. Rust is fun.");
/// assert!(!chunks.is_empty());
/// ```
pub struct TextChunker {
    /// Number of tokens (words) per chunk.
    pub chunk_size: usize,
    /// Number of tokens of overlap between consecutive chunks.
    pub overlap: usize,
    /// If `true`, try to respect sentence boundaries.
    pub by_sentence: bool,
}

impl Default for TextChunker {
    fn default() -> Self {
        Self::new(512, 50)
    }
}

impl TextChunker {
    /// Create a new chunker.
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        let safe_overlap = if overlap >= chunk_size {
            chunk_size.saturating_sub(1)
        } else {
            overlap
        };
        Self {
            chunk_size,
            overlap: safe_overlap,
            by_sentence: false,
        }
    }

    /// Enable sentence-boundary-respecting mode.
    pub fn with_sentence_boundaries(mut self) -> Self {
        self.by_sentence = true;
        self
    }

    /// Chunk `text` and return plain `String` chunks.
    pub fn chunk(&self, text: &str) -> Vec<String> {
        self.chunk_with_metadata(text)
            .into_iter()
            .map(|c| c.text)
            .collect()
    }

    /// Chunk `text` and return `TextChunk` structs with metadata.
    pub fn chunk_with_metadata(&self, text: &str) -> Vec<TextChunk> {
        if text.is_empty() {
            return Vec::new();
        }

        if self.by_sentence {
            self.chunk_by_sentence(text)
        } else {
            self.chunk_by_tokens(text)
        }
    }

    // -----------------------------------------------------------------------
    // Token-based chunking
    // -----------------------------------------------------------------------

    fn chunk_by_tokens(&self, text: &str) -> Vec<TextChunk> {
        let tokens: Vec<(usize, usize)> = token_byte_ranges(text);

        if tokens.is_empty() {
            return Vec::new();
        }

        let step = self.chunk_size.saturating_sub(self.overlap).max(1);
        let n = tokens.len();

        let mut raw: Vec<(usize, usize)> = Vec::new();
        let mut start_idx = 0usize;
        while start_idx < n {
            let end_idx = (start_idx + self.chunk_size).min(n);
            let chunk_start_byte = tokens[start_idx].0;
            let chunk_end_byte = tokens[end_idx - 1].1;
            raw.push((chunk_start_byte, chunk_end_byte));
            if end_idx >= n {
                break;
            }
            start_idx += step;
        }

        let total = raw.len();
        raw.into_iter()
            .enumerate()
            .map(|(idx, (start, end))| TextChunk {
                text: text[start..end].to_string(),
                start,
                end,
                chunk_index: idx,
                total_chunks: total,
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Sentence-boundary-aware chunking
    // -----------------------------------------------------------------------

    fn chunk_by_sentence(&self, text: &str) -> Vec<TextChunk> {
        let segmenter = SentenceSegmenter::new();
        let sentences = segmenter.segment(text);

        if sentences.is_empty() {
            return Vec::new();
        }

        let mut chunks_data: Vec<(String, usize, usize)> = Vec::new();
        let overlap_sentences = (self.overlap / 10).max(1);
        let mut i = 0;

        while i < sentences.len() {
            let mut word_count = 0;
            let mut j = i;
            let mut chunk_parts: Vec<&str> = Vec::new();

            while j < sentences.len() {
                let sentence = sentences[j];
                let wc = sentence.split_whitespace().count();
                if word_count + wc > self.chunk_size && !chunk_parts.is_empty() {
                    break;
                }
                chunk_parts.push(sentence);
                word_count += wc;
                j += 1;
            }

            if !chunk_parts.is_empty() {
                let combined = chunk_parts.join(" ");
                let start_byte = text.find(chunk_parts[0]).unwrap_or(0);
                let last = chunk_parts[chunk_parts.len() - 1];
                let last_start = text.rfind(last).unwrap_or(start_byte);
                let end_byte = (last_start + last.len()).min(text.len());
                chunks_data.push((combined, start_byte, end_byte));
            }

            let advance = (j - i).saturating_sub(overlap_sentences).max(1);
            i += advance;
        }

        let total = chunks_data.len();
        chunks_data
            .into_iter()
            .enumerate()
            .map(|(idx, (text_s, start, end))| TextChunk {
                text: text_s,
                start,
                end,
                chunk_index: idx,
                total_chunks: total,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return `(start_byte, end_byte)` pairs for each whitespace-delimited token.
pub fn token_byte_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let mut in_token = false;
    let mut token_start = 0usize;

    for (byte_pos, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if in_token {
                result.push((token_start, byte_pos));
                in_token = false;
            }
        } else if !in_token {
            token_start = byte_pos;
            in_token = true;
        }
    }
    if in_token {
        result.push((token_start, text.len()));
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_segmentation() {
        let seg = SentenceSegmenter::new();
        let sentences = seg.segment("Hello world. How are you? I am fine.");
        assert_eq!(
            sentences.len(),
            3,
            "Expected 3 sentences, got {:?}",
            sentences
        );
    }

    #[test]
    fn test_abbreviation_not_split() {
        let seg = SentenceSegmenter::new();
        let sentences = seg.segment("We met Dr. Smith today. He is well.");
        assert_eq!(
            sentences.len(),
            2,
            "Abbreviation should not create extra splits: {:?}",
            sentences
        );
    }

    #[test]
    fn test_exclamation_and_question() {
        let seg = SentenceSegmenter::new();
        let sentences = seg.segment("Amazing! Really? Yes absolutely.");
        assert!(sentences.len() >= 2);
    }

    #[test]
    fn test_segment_owned() {
        let seg = SentenceSegmenter::new();
        let sentences = seg.segment_owned("First sentence. Second sentence. Third sentence.");
        assert!(!sentences.is_empty());
        for s in &sentences {
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_empty_text_returns_empty() {
        let seg = SentenceSegmenter::new();
        assert!(seg.segment("").is_empty());
        assert!(seg.segment_owned("").is_empty());
    }

    #[test]
    fn test_single_sentence() {
        let seg = SentenceSegmenter::new();
        let result = seg.segment("This is just one sentence");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_with_abbreviations() {
        let seg = SentenceSegmenter::with_abbreviations(vec!["Esq".to_string()]);
        let result = seg.segment("John Smith, Esq. is present. He said hello.");
        assert_eq!(result.len(), 2, "Got {:?}", result);
    }

    #[test]
    fn test_no_false_split_on_decimal() {
        let seg = SentenceSegmenter::new();
        let result = seg.segment("Pi is about 3.14159 in value. That is a fact.");
        assert_eq!(result.len(), 2, "Got {:?}", result);
    }

    #[test]
    fn test_chunker_basic() {
        let chunker = TextChunker::new(5, 1);
        let text = "one two three four five six seven eight nine ten";
        let chunks = chunker.chunk(text);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            let wc = chunk.split_whitespace().count();
            assert!(wc <= 5, "Chunk '{}' has {} words", chunk, wc);
        }
    }

    #[test]
    fn test_chunker_overlap() {
        let chunker = TextChunker::new(4, 2);
        let text = "a b c d e f g h";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() >= 2);
        if chunks.len() >= 2 {
            let words_0: Vec<&str> = chunks[0].split_whitespace().collect();
            let words_1: Vec<&str> = chunks[1].split_whitespace().collect();
            let last_two: Vec<&str> = words_0.iter().rev().take(2).rev().copied().collect();
            let first_two: Vec<&str> = words_1.iter().take(2).copied().collect();
            assert_eq!(last_two, first_two, "Overlap should share tokens");
        }
    }

    #[test]
    fn test_chunker_with_metadata() {
        let chunker = TextChunker::new(3, 0);
        let text = "alpha beta gamma delta epsilon";
        let chunks = chunker.chunk_with_metadata(text);
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index, i);
            assert_eq!(chunk.total_chunks, chunks.len());
            assert_eq!(&text[chunk.start..chunk.end], chunk.text.as_str());
        }
    }

    #[test]
    fn test_chunker_empty_text() {
        let chunker = TextChunker::new(10, 2);
        assert!(chunker.chunk("").is_empty());
        assert!(chunker.chunk_with_metadata("").is_empty());
    }

    #[test]
    fn test_chunker_short_text() {
        let chunker = TextChunker::new(100, 10);
        let text = "just three words";
        let chunks = chunker.chunk(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_chunker_by_sentence() {
        let chunker = TextChunker::new(20, 5).with_sentence_boundaries();
        let text = "The quick brown fox jumps. A lazy dog sleeps. The sun is shining.";
        let chunks = chunker.chunk(text);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunker_overlap_clamped() {
        let chunker = TextChunker::new(3, 10);
        assert!(chunker.overlap < chunker.chunk_size);
    }

    #[test]
    fn test_token_byte_ranges() {
        let text = "hello world foo";
        let ranges = token_byte_ranges(text);
        assert_eq!(ranges.len(), 3);
        assert_eq!(&text[ranges[0].0..ranges[0].1], "hello");
        assert_eq!(&text[ranges[1].0..ranges[1].1], "world");
        assert_eq!(&text[ranges[2].0..ranges[2].1], "foo");
    }
}
