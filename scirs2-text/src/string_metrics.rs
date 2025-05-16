//! String metrics module for distance calculations and phonetic algorithms.

use crate::Result;
use std::cmp::{max, min};
use std::collections::HashMap;

/// Trait for string distance metrics
pub trait StringMetric {
    /// Calculate the distance between two strings
    fn distance(&self, s1: &str, s2: &str) -> Result<f64>;

    /// Calculate normalized distance (0.0 to 1.0)
    fn normalized_distance(&self, s1: &str, s2: &str) -> Result<f64> {
        let dist = self.distance(s1, s2)?;
        let max_len = max(s1.len(), s2.len()) as f64;
        Ok(if max_len > 0.0 { dist / max_len } else { 0.0 })
    }

    /// Calculate similarity (1.0 - normalized distance)
    fn similarity(&self, s1: &str, s2: &str) -> Result<f64> {
        Ok(1.0 - self.normalized_distance(s1, s2)?)
    }
}

/// Damerau-Levenshtein distance metric
#[derive(Debug, Clone)]
pub struct DamerauLevenshteinMetric {
    /// Whether to use restricted Damerau-Levenshtein (no substring transpositions)
    restricted: bool,
}

impl DamerauLevenshteinMetric {
    /// Create a new Damerau-Levenshtein metric
    pub fn new() -> Self {
        Self { restricted: false }
    }

    /// Create a restricted variant (OSA - Optimal String Alignment)
    pub fn restricted() -> Self {
        Self { restricted: true }
    }
}

impl Default for DamerauLevenshteinMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl StringMetric for DamerauLevenshteinMetric {
    fn distance(&self, s1: &str, s2: &str) -> Result<f64> {
        if self.restricted {
            Ok(osa_distance(s1, s2) as f64)
        } else {
            Ok(damerau_levenshtein_distance(s1, s2) as f64)
        }
    }
}

/// Calculate restricted Damerau-Levenshtein (OSA) distance
fn osa_distance(s1: &str, s2: &str) -> usize {
    let a: Vec<char> = s1.chars().collect();
    let b: Vec<char> = s2.chars().collect();
    let len_a = a.len();
    let len_b = b.len();

    if len_a == 0 {
        return len_b;
    }
    if len_b == 0 {
        return len_a;
    }

    let mut matrix = vec![vec![0; len_b + 1]; len_a + 1];

    for (i, row) in matrix.iter_mut().enumerate().take(len_a + 1) {
        row[0] = i;
    }
    // For the first row, set each column index as the value
    for j in 0..=len_b {
        matrix[0][j] = j;
    }

    for i in 1..=len_a {
        for j in 1..=len_b {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };

            matrix[i][j] = min(
                min(
                    matrix[i - 1][j] + 1, // deletion
                    matrix[i][j - 1] + 1, // insertion
                ),
                matrix[i - 1][j - 1] + cost, // substitution
            );

            // Transposition
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                matrix[i][j] = min(matrix[i][j], matrix[i - 2][j - 2] + cost);
            }
        }
    }

    matrix[len_a][len_b]
}

/// Calculate full Damerau-Levenshtein distance
fn damerau_levenshtein_distance(s1: &str, s2: &str) -> usize {
    let a: Vec<char> = s1.chars().collect();
    let b: Vec<char> = s2.chars().collect();
    let len_a = a.len();
    let len_b = b.len();

    if len_a == 0 {
        return len_b;
    }
    if len_b == 0 {
        return len_a;
    }

    let max_dist = len_a + len_b;
    let mut h = vec![vec![max_dist; len_b + 2]; len_a + 2];
    let mut da: HashMap<char, usize> = HashMap::new();

    for i in 0..=len_a {
        h[i + 1][0] = max_dist;
        h[i + 1][1] = i;
    }
    for j in 0..=len_b {
        h[0][j + 1] = max_dist;
        h[1][j + 1] = j;
    }

    for i in 1..=len_a {
        let mut db = 0;
        for j in 1..=len_b {
            let k = da.get(&b[j - 1]).copied().unwrap_or(0);
            let l = db;
            let cost = if a[i - 1] == b[j - 1] {
                db = j;
                0
            } else {
                1
            };

            h[i + 1][j + 1] = min(
                min(
                    h[i][j] + cost,  // substitution
                    h[i + 1][j] + 1, // insertion
                ),
                min(
                    h[i][j + 1] + 1,                         // deletion
                    h[k][l] + (i - k - 1) + 1 + (j - l - 1), // transposition
                ),
            );
        }

        da.insert(a[i - 1], i);
    }

    h[len_a + 1][len_b + 1]
}

/// Phonetic algorithms trait
pub trait PhoneticAlgorithm {
    /// Encode a string using the phonetic algorithm
    fn encode(&self, text: &str) -> Result<String>;

    /// Check if two strings are phonetically similar
    fn sounds_like(&self, s1: &str, s2: &str) -> Result<bool> {
        Ok(self.encode(s1)? == self.encode(s2)?)
    }
}

/// Soundex phonetic algorithm
#[derive(Debug, Clone)]
pub struct Soundex {
    /// Length of the code (standard is 4)
    length: usize,
}

impl Soundex {
    /// Create new Soundex encoder with standard length (4)
    pub fn new() -> Self {
        Self { length: 4 }
    }

    /// Create with custom code length
    pub fn with_length(length: usize) -> Self {
        Self { length }
    }
}

impl Default for Soundex {
    fn default() -> Self {
        Self::new()
    }
}

impl PhoneticAlgorithm for Soundex {
    fn encode(&self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }

        let text = text.to_uppercase();
        let chars: Vec<char> = text.chars().filter(|c| c.is_alphabetic()).collect();

        if chars.is_empty() {
            return Ok(String::new());
        }

        let mut code = String::new();
        code.push(chars[0]);

        let mut last_code = encode_char(chars[0]);

        for &ch in &chars[1..] {
            let ch_code = encode_char(ch);
            if ch_code != '0' && ch_code != last_code {
                code.push(ch_code);
                last_code = ch_code;
            }

            if code.len() >= self.length {
                break;
            }
        }

        // Pad with zeros if necessary
        while code.len() < self.length {
            code.push('0');
        }

        Ok(code)
    }
}

/// Encode a character for Soundex
fn encode_char(ch: char) -> char {
    match ch.to_ascii_uppercase() {
        'B' | 'F' | 'P' | 'V' => '1',
        'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => '2',
        'D' | 'T' => '3',
        'L' => '4',
        'M' | 'N' => '5',
        'R' => '6',
        _ => '0',
    }
}

/// Metaphone phonetic algorithm
#[derive(Debug, Clone)]
pub struct Metaphone {
    /// Maximum length of the code
    max_length: usize,
}

impl Metaphone {
    /// Create new Metaphone encoder
    pub fn new() -> Self {
        Self { max_length: 6 }
    }

    /// Create with custom maximum length
    pub fn with_max_length(max_length: usize) -> Self {
        Self { max_length }
    }
}

impl Default for Metaphone {
    fn default() -> Self {
        Self::new()
    }
}

impl PhoneticAlgorithm for Metaphone {
    fn encode(&self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }

        let text = text.to_uppercase();
        let chars: Vec<char> = text.chars().filter(|c| c.is_alphabetic()).collect();

        if chars.is_empty() {
            return Ok(String::new());
        }

        let mut result = String::new();
        let mut i = 0;

        // Initial transformations
        if chars.len() >= 2 {
            match (chars[0], chars[1]) {
                ('A', 'E') | ('G', 'N') | ('K', 'N') | ('P', 'N') | ('W', 'R') => i = 1,
                ('W', 'H') => {
                    result.push('W');
                    i = 2;
                }
                _ => {}
            }
        }

        while i < chars.len() && result.len() < self.max_length {
            let ch = chars[i];
            let next = chars.get(i + 1).copied();
            let prev = if i > 0 {
                chars.get(i - 1).copied()
            } else {
                None
            };

            match ch {
                'A' | 'E' | 'I' | 'O' | 'U' => {
                    if i == 0 {
                        result.push(ch);
                    }
                }
                'B' => {
                    if !result.ends_with('B') {
                        result.push('B');
                    }
                }
                'C' => {
                    if next == Some('H') {
                        result.push('X');
                        i += 1;
                    } else if next == Some('I') || next == Some('E') || next == Some('Y') {
                        result.push('S');
                    } else {
                        result.push('K');
                    }
                }
                'D' => {
                    if next == Some('G') && chars.get(i + 2) == Some(&'E') {
                        result.push('J');
                        i += 2;
                    } else {
                        result.push('T');
                    }
                }
                'F' | 'J' | 'L' | 'M' | 'N' | 'R' => {
                    if !result.ends_with(ch) {
                        result.push(ch);
                    }
                }
                'G' => {
                    if next == Some('H') {
                        i += 1;
                    } else if next == Some('N') {
                        result.push('N');
                        i += 1;
                    } else {
                        result.push('K');
                    }
                }
                'H' => {
                    if prev != Some('C')
                        && prev != Some('S')
                        && prev != Some('P')
                        && prev != Some('T')
                        && prev != Some('G')
                    {
                        result.push('H');
                    }
                }
                'K' => {
                    if prev != Some('C') {
                        result.push('K');
                    }
                }
                'P' => {
                    if next == Some('H') {
                        result.push('F');
                        i += 1;
                    } else {
                        result.push('P');
                    }
                }
                'Q' => result.push('K'),
                'S' => {
                    if next == Some('H') {
                        result.push('X');
                        i += 1;
                    } else if next == Some('I')
                        && (chars.get(i + 2) == Some(&'A') || chars.get(i + 2) == Some(&'O'))
                    {
                        result.push('X');
                    } else {
                        result.push('S');
                    }
                }
                'T' => {
                    if next == Some('H') {
                        result.push('0');
                        i += 1;
                    } else if next != Some('C') || chars.get(i + 2) != Some(&'H') {
                        result.push('T');
                    }
                }
                'V' => result.push('F'),
                'W' | 'Y' => {
                    if next.map(|c| "AEIOU".contains(c)).unwrap_or(false) {
                        result.push(ch);
                    }
                }
                'X' => {
                    result.push('K');
                    result.push('S');
                }
                'Z' => result.push('S'),
                _ => {}
            }

            i += 1;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damerau_levenshtein() {
        let metric = DamerauLevenshteinMetric::new();

        // Basic operations
        assert_eq!(metric.distance("", "").unwrap(), 0.0);
        assert_eq!(metric.distance("abc", "").unwrap(), 3.0);
        assert_eq!(metric.distance("", "abc").unwrap(), 3.0);
        assert_eq!(metric.distance("abc", "abc").unwrap(), 0.0);

        // Single operations
        assert_eq!(metric.distance("abc", "aXc").unwrap(), 1.0); // substitution
        assert_eq!(metric.distance("abc", "ac").unwrap(), 1.0); // deletion
        assert_eq!(metric.distance("ac", "abc").unwrap(), 1.0); // insertion
        assert_eq!(metric.distance("abc", "acb").unwrap(), 1.0); // transposition

        // Multiple operations
        assert_eq!(metric.distance("kitten", "sitting").unwrap(), 3.0);

        // Test normalized distance
        assert!((metric.normalized_distance("abc", "aXc").unwrap() - 0.333).abs() < 0.01);
        assert_eq!(metric.normalized_distance("", "").unwrap(), 0.0);

        // Test similarity
        assert!((metric.similarity("abc", "aXc").unwrap() - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_restricted_damerau_levenshtein() {
        let metric = DamerauLevenshteinMetric::restricted();

        // OSA doesn't allow substring transpositions
        assert_eq!(metric.distance("ca", "abc").unwrap(), 3.0); // Not 2.0 as in full DL
    }

    #[test]
    fn test_soundex() {
        let soundex = Soundex::new();

        assert_eq!(soundex.encode("Robert").unwrap(), "R163");
        assert_eq!(soundex.encode("Rupert").unwrap(), "R163");
        assert_eq!(soundex.encode("SOUNDEX").unwrap(), "S532");
        assert_eq!(soundex.encode("Smith").unwrap(), "S530");
        assert_eq!(soundex.encode("Smythe").unwrap(), "S530");
        assert_eq!(soundex.encode("").unwrap(), "");
        assert_eq!(soundex.encode("123").unwrap(), "");

        // Test sounds_like
        assert!(soundex.sounds_like("Robert", "Rupert").unwrap());
        assert!(soundex.sounds_like("Smith", "Smythe").unwrap());
        assert!(!soundex.sounds_like("Smith", "Jones").unwrap());

        // Test custom length
        let soundex_5 = Soundex::with_length(5);
        assert_eq!(soundex_5.encode("SOUNDEX").unwrap(), "S5320");
    }

    #[test]
    fn test_metaphone() {
        let metaphone = Metaphone::new();

        assert_eq!(metaphone.encode("programming").unwrap(), "PRKRMN");
        assert_eq!(metaphone.encode("programmer").unwrap(), "PRKRMR");
        assert_eq!(metaphone.encode("Wright").unwrap(), "RT");
        assert_eq!(metaphone.encode("White").unwrap(), "WT");
        assert_eq!(metaphone.encode("Knight").unwrap(), "NT");
        assert_eq!(metaphone.encode("").unwrap(), "");
        assert_eq!(metaphone.encode("123").unwrap(), "");

        // Test sounds_like - "Wright" and "Write" actually should sound similar in Metaphone
        assert!(metaphone.sounds_like("Wright", "Write").unwrap());
        assert!(!metaphone.sounds_like("White", "Wright").unwrap());

        // Test custom max length
        let metaphone_3 = Metaphone::with_max_length(3);
        assert_eq!(metaphone_3.encode("programming").unwrap(), "PRK");
    }

    #[test]
    fn test_phonetic_edge_cases() {
        let soundex = Soundex::new();
        let metaphone = Metaphone::new();

        // Test with non-alphabetic characters
        assert_eq!(soundex.encode("O'Brien").unwrap(), "O165");
        assert_eq!(metaphone.encode("O'Brien").unwrap(), "OBRN");

        // Test with mixed case
        assert_eq!(soundex.encode("McDonald").unwrap(), "M235");
        assert_eq!(metaphone.encode("McDonald").unwrap(), "MKTNLT");
    }
}
