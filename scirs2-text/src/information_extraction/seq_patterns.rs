//! Pattern-based information extraction using token sequences.
//!
//! Provides a flexible rule engine operating on tokenized input.  Each pattern
//! is a sequence of [`PatternElement`]s that are matched left-to-right against
//! a slice of [`Token`]s.  Gap elements allow for skip-regions of bounded
//! length.

use regex::Regex;

use crate::error::{Result, TextError};

// ---------------------------------------------------------------------------
// Token type
// ---------------------------------------------------------------------------

/// A single token with optional part-of-speech and lemma annotations.
#[derive(Debug, Clone)]
pub struct Token {
    /// Surface form of the token.
    pub text: String,
    /// Optional part-of-speech tag (e.g. "NN", "VBZ").
    pub pos: Option<String>,
    /// Optional base/lemma form.
    pub lemma: Option<String>,
}

impl Token {
    /// Convenience constructor.
    pub fn new(text: impl Into<String>) -> Token {
        Token {
            text: text.into(),
            pos: None,
            lemma: None,
        }
    }

    /// Builder: set POS tag.
    pub fn with_pos(mut self, pos: impl Into<String>) -> Token {
        self.pos = Some(pos.into());
        self
    }

    /// Builder: set lemma.
    pub fn with_lemma(mut self, lemma: impl Into<String>) -> Token {
        self.lemma = Some(lemma.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Pattern element
// ---------------------------------------------------------------------------

/// A single element in a named extraction pattern.
#[derive(Debug, Clone)]
pub enum PatternElement {
    /// Match a fixed string (case-insensitive).
    Literal(String),
    /// Match a token whose POS tag equals the given string.
    PoS(String),
    /// Match a token whose text matches the given regex string.
    Regex(String),
    /// Match any single token.
    Any,
    /// Match between `min` and `max` tokens (inclusive), skipping them.
    Gap {
        /// Minimum number of tokens to skip.
        min: usize,
        /// Maximum number of tokens to skip.
        max: usize,
    },
}

// ---------------------------------------------------------------------------
// Pattern and result types
// ---------------------------------------------------------------------------

/// A sequence of [`PatternElement`]s representing one extraction rule.
#[derive(Debug, Clone)]
pub struct Pattern {
    /// Ordered list of elements that must match in sequence.
    pub template: Vec<PatternElement>,
}

impl Pattern {
    /// Construct a pattern from a template.
    pub fn new(template: Vec<PatternElement>) -> Pattern {
        Pattern { template }
    }
}

/// A single extraction match.
#[derive(Debug, Clone)]
pub struct Match {
    /// Name of the pattern that produced this match.
    pub pattern_name: String,
    /// Index of the first token in the match.
    pub start: usize,
    /// Index one past the last token in the match.
    pub end: usize,
    /// Captured text fragments (one per non-Gap, non-Any element).
    pub groups: Vec<String>,
}

// ---------------------------------------------------------------------------
// PatternMatcher
// ---------------------------------------------------------------------------

/// A collection of named extraction patterns applied to token sequences.
#[derive(Default)]
pub struct PatternMatcher {
    patterns: Vec<(String, Pattern)>,
    /// Compiled regexes cached by pattern string.
    regex_cache: std::collections::HashMap<String, Regex>,
}

impl PatternMatcher {
    /// Create an empty matcher.
    pub fn new() -> PatternMatcher {
        PatternMatcher::default()
    }

    /// Add a named pattern.
    pub fn add_pattern(&mut self, name: impl Into<String>, pattern: Pattern) -> Result<()> {
        // Pre-compile any Regex elements
        for elem in &pattern.template {
            if let PatternElement::Regex(re_str) = elem {
                if !self.regex_cache.contains_key(re_str) {
                    let compiled = Regex::new(re_str).map_err(|e| {
                        TextError::InvalidInput(format!("Bad regex '{}': {}", re_str, e))
                    })?;
                    self.regex_cache.insert(re_str.clone(), compiled);
                }
            }
        }
        self.patterns.push((name.into(), pattern));
        Ok(())
    }

    /// Find all pattern matches in the token sequence.
    ///
    /// Tries every starting position for every pattern; returns all matches
    /// (possibly overlapping).
    pub fn match_all(&self, tokens: &[Token]) -> Vec<Match> {
        let mut results = Vec::new();
        for (name, pattern) in &self.patterns {
            for start in 0..tokens.len() {
                if let Some((end, groups)) = self.try_match(pattern, tokens, start) {
                    results.push(Match {
                        pattern_name: name.clone(),
                        start,
                        end,
                        groups,
                    });
                }
            }
        }
        results
    }

    /// Attempt to match `pattern` starting at `start` in `tokens`.
    ///
    /// Returns `Some((end_exclusive, captured_groups))` on success.
    fn try_match(
        &self,
        pattern: &Pattern,
        tokens: &[Token],
        start: usize,
    ) -> Option<(usize, Vec<String>)> {
        self.try_match_from(pattern, tokens, start, 0, Vec::new())
    }

    /// Recursive helper that matches pattern elements from `elem_idx` onwards,
    /// starting at token position `pos`, with previously captured `groups`.
    fn try_match_from(
        &self,
        pattern: &Pattern,
        tokens: &[Token],
        pos: usize,
        elem_idx: usize,
        groups: Vec<String>,
    ) -> Option<(usize, Vec<String>)> {
        // All elements matched
        if elem_idx >= pattern.template.len() {
            return Some((pos, groups));
        }

        let elem = &pattern.template[elem_idx];

        match elem {
            PatternElement::Literal(s) => {
                if pos >= tokens.len() {
                    return None;
                }
                if tokens[pos].text.to_lowercase() != s.to_lowercase() {
                    return None;
                }
                let mut new_groups = groups;
                new_groups.push(tokens[pos].text.clone());
                self.try_match_from(pattern, tokens, pos + 1, elem_idx + 1, new_groups)
            }
            PatternElement::PoS(tag) => {
                if pos >= tokens.len() {
                    return None;
                }
                let tok_pos = tokens[pos].pos.as_deref().unwrap_or("");
                if tok_pos != tag.as_str() {
                    return None;
                }
                let mut new_groups = groups;
                new_groups.push(tokens[pos].text.clone());
                self.try_match_from(pattern, tokens, pos + 1, elem_idx + 1, new_groups)
            }
            PatternElement::Regex(re_str) => {
                if pos >= tokens.len() {
                    return None;
                }
                let re = self.regex_cache.get(re_str)?;
                if !re.is_match(&tokens[pos].text) {
                    return None;
                }
                let mut new_groups = groups;
                new_groups.push(tokens[pos].text.clone());
                self.try_match_from(pattern, tokens, pos + 1, elem_idx + 1, new_groups)
            }
            PatternElement::Any => {
                if pos >= tokens.len() {
                    return None;
                }
                let mut new_groups = groups;
                new_groups.push(tokens[pos].text.clone());
                self.try_match_from(pattern, tokens, pos + 1, elem_idx + 1, new_groups)
            }
            PatternElement::Gap { min, max } => {
                // Try each skip length from min to max; return the first
                // that allows the rest of the pattern to match.
                for skip in *min..=*max {
                    let new_pos = pos + skip;
                    if new_pos > tokens.len() {
                        break;
                    }
                    if let Some(result) =
                        self.try_match_from(pattern, tokens, new_pos, elem_idx + 1, groups.clone())
                    {
                        return Some(result);
                    }
                }
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in rule-based NER patterns
// ---------------------------------------------------------------------------

/// Build a `PatternMatcher` pre-loaded with common NER patterns.
///
/// Includes patterns for dates, money, email, URL, and phone numbers.
pub fn build_ner_pattern_matcher() -> Result<PatternMatcher> {
    let mut matcher = PatternMatcher::new();

    // Date: MM/DD/YYYY or YYYY-MM-DD
    matcher.add_pattern(
        "DATE",
        Pattern::new(vec![PatternElement::Regex(
            r"(?:(?:0?[1-9]|1[0-2])[\/\-](?:0?[1-9]|[12][0-9]|3[01])[\/\-](?:19|20)?\d{2}|(?:19|20)\d{2}[\/\-](?:0?[1-9]|1[0-2])[\/\-](?:0?[1-9]|[12][0-9]|3[01]))".to_string(),
        )]),
    )?;

    // Money: $1234.56 or similar
    matcher.add_pattern(
        "MONEY",
        Pattern::new(vec![PatternElement::Regex(
            r"\$[0-9]+(?:\.[0-9]+)?".to_string(),
        )]),
    )?;

    // Email
    matcher.add_pattern(
        "EMAIL",
        Pattern::new(vec![PatternElement::Regex(
            r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}".to_string(),
        )]),
    )?;

    // URL
    matcher.add_pattern(
        "URL",
        Pattern::new(vec![PatternElement::Regex(r"https?://[^\s]+".to_string())]),
    )?;

    // Phone: (NNN) NNN-NNNN or NNN-NNN-NNNN
    matcher.add_pattern(
        "PHONE",
        Pattern::new(vec![PatternElement::Regex(
            r"(?:\+?1[\-.\s]?)?\(?\d{3}\)?[\-.\s]\d{3}[\-.\s]\d{4}".to_string(),
        )]),
    )?;

    Ok(matcher)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenize_simple(text: &str) -> Vec<Token> {
        text.split_whitespace().map(Token::new).collect()
    }

    #[test]
    fn test_literal_match() {
        let mut matcher = PatternMatcher::new();
        matcher
            .add_pattern(
                "greeting",
                Pattern::new(vec![
                    PatternElement::Literal("hello".to_string()),
                    PatternElement::Literal("world".to_string()),
                ]),
            )
            .expect("add_pattern failed");

        let tokens = tokenize_simple("hello world foo bar");
        let matches = matcher.match_all(&tokens);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].start, 0);
        assert_eq!(matches[0].end, 2);
    }

    #[test]
    fn test_pos_match() {
        let mut matcher = PatternMatcher::new();
        matcher
            .add_pattern(
                "dt_nn",
                Pattern::new(vec![
                    PatternElement::PoS("DT".to_string()),
                    PatternElement::PoS("NN".to_string()),
                ]),
            )
            .expect("add_pattern failed");

        let tokens = vec![
            Token::new("the").with_pos("DT"),
            Token::new("dog").with_pos("NN"),
            Token::new("runs").with_pos("VBZ"),
        ];
        let matches = matcher.match_all(&tokens);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].groups, vec!["the", "dog"]);
    }

    #[test]
    fn test_regex_match() {
        let mut matcher = PatternMatcher::new();
        matcher
            .add_pattern(
                "money",
                Pattern::new(vec![PatternElement::Regex(
                    r"\$[0-9]+(?:\.[0-9]+)?".to_string(),
                )]),
            )
            .expect("add_pattern failed");

        let tokens = tokenize_simple("costs $29.99 shipping $5");
        let matches = matcher.match_all(&tokens);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_any_match() {
        let mut matcher = PatternMatcher::new();
        matcher
            .add_pattern(
                "any_word",
                Pattern::new(vec![
                    PatternElement::Literal("the".to_string()),
                    PatternElement::Any,
                ]),
            )
            .expect("add_pattern failed");

        let tokens = tokenize_simple("the cat sat on the mat");
        let matches = matcher.match_all(&tokens);
        // "the cat" and "the mat"
        assert!(matches.len() >= 2);
    }

    #[test]
    fn test_gap_match() {
        let mut matcher = PatternMatcher::new();
        matcher
            .add_pattern(
                "verb_phrase",
                Pattern::new(vec![
                    PatternElement::Literal("john".to_string()),
                    PatternElement::Gap { min: 0, max: 2 },
                    PatternElement::Literal("mary".to_string()),
                ]),
            )
            .expect("add_pattern failed");

        let tokens = tokenize_simple("john loves mary");
        let matches = matcher.match_all(&tokens);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_ner_patterns_email() {
        let matcher = build_ner_pattern_matcher().expect("build failed");
        let tokens = tokenize_simple("contact user@example.com for info");
        let matches = matcher.match_all(&tokens);
        assert!(matches.iter().any(|m| m.pattern_name == "EMAIL"));
    }

    #[test]
    fn test_ner_patterns_money() {
        let matcher = build_ner_pattern_matcher().expect("build failed");
        let tokens = tokenize_simple("costs $100 today");
        let matches = matcher.match_all(&tokens);
        assert!(matches.iter().any(|m| m.pattern_name == "MONEY"));
    }

    #[test]
    fn test_bad_regex_error() {
        let mut matcher = PatternMatcher::new();
        let result = matcher.add_pattern(
            "bad",
            Pattern::new(vec![PatternElement::Regex("[invalid".to_string())]),
        );
        assert!(result.is_err());
    }
}
