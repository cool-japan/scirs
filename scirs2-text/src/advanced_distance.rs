//! Advanced string distance and alignment algorithms.
//!
//! Provides:
//! - Levenshtein edit distance (standard and weighted)
//! - Damerau-Levenshtein (with transpositions)
//! - Jaro and Jaro-Winkler similarity
//! - Normalised edit distance
//! - Longest Common Subsequence (LCS)
//! - Smith-Waterman local sequence alignment
//! - Needleman-Wunsch global sequence alignment
//! - Fuzzy string matching

// ─────────────────────────────────────────────────────────────────────────────
// Levenshtein
// ─────────────────────────────────────────────────────────────────────────────

/// Standard Levenshtein edit distance (insertions, deletions, substitutions,
/// each with cost 1).
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use two rolling rows instead of full matrix to save memory
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1)           // deletion
                .min(curr[j - 1] + 1)          // insertion
                .min(prev[j - 1] + cost); // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Levenshtein edit distance with custom per-operation costs.
///
/// * `ins` – cost of an insertion.
/// * `del` – cost of a deletion.
/// * `sub` – cost of a substitution.
pub fn weighted_levenshtein(a: &str, b: &str, ins: f64, del: f64, sub: f64) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n as f64 * ins;
    }
    if n == 0 {
        return m as f64 * del;
    }

    let mut prev: Vec<f64> = (0..=n).map(|j| j as f64 * ins).collect();
    let mut curr: Vec<f64> = vec![0.0; n + 1];

    for i in 1..=m {
        curr[0] = i as f64 * del;
        for j in 1..=n {
            let sub_cost = if a_chars[i - 1] == b_chars[j - 1] {
                0.0
            } else {
                sub
            };
            curr[j] = (prev[j] + del)              // deletion from a
                .min(curr[j - 1] + ins)             // insertion into a
                .min(prev[j - 1] + sub_cost); // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

// ─────────────────────────────────────────────────────────────────────────────
// Damerau-Levenshtein
// ─────────────────────────────────────────────────────────────────────────────

/// Damerau-Levenshtein edit distance, which additionally allows single-character
/// transpositions (swapping two adjacent characters).
///
/// Uses the optimal string alignment (OSA) variant, which is O(mn) time and
/// space.
pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut d = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        d[i][0] = i;
    }
    for j in 0..=n {
        d[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            d[i][j] = (d[i - 1][j] + 1)                 // deletion
                .min(d[i][j - 1] + 1)                    // insertion
                .min(d[i - 1][j - 1] + cost); // substitution

            // Transposition
            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[m][n]
}

// ─────────────────────────────────────────────────────────────────────────────
// Jaro / Jaro-Winkler
// ─────────────────────────────────────────────────────────────────────────────

/// Jaro similarity between two strings (0.0 = completely different, 1.0 = identical).
pub fn jaro(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m_len = a_chars.len();
    let n_len = b_chars.len();

    let match_dist = (m_len.max(n_len) / 2).saturating_sub(1);

    let mut a_matched = vec![false; m_len];
    let mut b_matched = vec![false; n_len];
    let mut matches = 0usize;

    for i in 0..m_len {
        let lo = i.saturating_sub(match_dist);
        let hi = (i + match_dist + 1).min(n_len);
        for j in lo..hi {
            if !b_matched[j] && a_chars[i] == b_chars[j] {
                a_matched[i] = true;
                b_matched[j] = true;
                matches += 1;
                break;
            }
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions
    let mut transpositions = 0usize;
    let mut k = 0usize;
    for i in 0..m_len {
        if a_matched[i] {
            while !b_matched[k] {
                k += 1;
            }
            if a_chars[i] != b_chars[k] {
                transpositions += 1;
            }
            k += 1;
        }
    }

    let m = matches as f64;
    let t = transpositions as f64 / 2.0;
    (m / m_len as f64 + m / n_len as f64 + (m - t) / m) / 3.0
}

/// Jaro-Winkler similarity, which boosts the score for strings sharing a
/// common prefix.
///
/// `prefix_weight` (standard: 0.1) controls how much the prefix match boosts
/// the score; it must be in (0.0, 0.25] for the result to stay in [0.0, 1.0].
pub fn jaro_winkler(a: &str, b: &str, prefix_weight: f64) -> f64 {
    let j = jaro(a, b);
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let max_prefix = 4.min(a_chars.len().min(b_chars.len()));
    let mut prefix = 0usize;
    for i in 0..max_prefix {
        if a_chars[i] == b_chars[i] {
            prefix += 1;
        } else {
            break;
        }
    }
    j + prefix as f64 * prefix_weight * (1.0 - j)
}

// ─────────────────────────────────────────────────────────────────────────────
// Normalised edit distance
// ─────────────────────────────────────────────────────────────────────────────

/// Levenshtein distance normalised to [0.0, 1.0] by the maximum string length.
///
/// Returns 0.0 for identical strings and 1.0 when one of them is completely
/// unlike the other.
pub fn normalized_levenshtein(a: &str, b: &str) -> f64 {
    let dist = levenshtein(a, b);
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        0.0
    } else {
        dist as f64 / max_len as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LCS
// ─────────────────────────────────────────────────────────────────────────────

/// Length of the Longest Common Subsequence (LCS) of two strings.
pub fn lcs_length(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if a_chars[i - 1] == b_chars[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    dp[m][n]
}

/// Recover the LCS string from two input strings.
pub fn lcs(a: &str, b: &str) -> String {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if a_chars[i - 1] == b_chars[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Traceback
    let mut result = Vec::new();
    let (mut i, mut j) = (m, n);
    while i > 0 && j > 0 {
        if a_chars[i - 1] == b_chars[j - 1] {
            result.push(a_chars[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] >= dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    result.reverse();
    result.into_iter().collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequence alignment helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Reconstruct aligned strings from a direction matrix for Smith-Waterman or
/// Needleman-Wunsch.
///
/// `dir` encodes: 0 = diagonal (match/mismatch), 1 = up (gap in b), 2 = left (gap in a).
fn traceback_alignment(
    a_chars: &[char],
    b_chars: &[char],
    dir: &[Vec<u8>],
    start_i: usize,
    start_j: usize,
    stop_zero: bool, // stop when score hits 0 (Smith-Waterman)
    scores: &[Vec<i32>],
) -> (String, String) {
    let mut aligned_a = Vec::<char>::new();
    let mut aligned_b = Vec::<char>::new();
    let (mut i, mut j) = (start_i, start_j);

    loop {
        if i == 0 && j == 0 {
            break;
        }
        if stop_zero && i < scores.len() && j < scores[i].len() && scores[i][j] == 0 {
            break;
        }
        if i == 0 {
            aligned_a.push('-');
            aligned_b.push(b_chars[j - 1]);
            j -= 1;
        } else if j == 0 {
            aligned_a.push(a_chars[i - 1]);
            aligned_b.push('-');
            i -= 1;
        } else {
            match dir[i][j] {
                0 => {
                    aligned_a.push(a_chars[i - 1]);
                    aligned_b.push(b_chars[j - 1]);
                    i -= 1;
                    j -= 1;
                }
                1 => {
                    aligned_a.push(a_chars[i - 1]);
                    aligned_b.push('-');
                    i -= 1;
                }
                _ => {
                    aligned_a.push('-');
                    aligned_b.push(b_chars[j - 1]);
                    j -= 1;
                }
            }
        }
    }

    aligned_a.reverse();
    aligned_b.reverse();
    (
        aligned_a.into_iter().collect(),
        aligned_b.into_iter().collect(),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Smith-Waterman
// ─────────────────────────────────────────────────────────────────────────────

/// Smith-Waterman local sequence alignment.
///
/// Returns `(best_score, aligned_a, aligned_b)`.
///
/// * `match_score` – reward for matching characters (positive).
/// * `mismatch`    – penalty for mismatches (typically negative).
/// * `gap`         – gap penalty (typically negative).
pub fn smith_waterman(
    a: &str,
    b: &str,
    match_score: i32,
    mismatch: i32,
    gap: i32,
) -> (i32, String, String) {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut score = vec![vec![0i32; n + 1]; m + 1];
    let mut dir = vec![vec![0u8; n + 1]; m + 1];
    let mut best_score = 0i32;
    let mut best_i = 0usize;
    let mut best_j = 0usize;

    for i in 1..=m {
        for j in 1..=n {
            let diag_score = if a_chars[i - 1] == b_chars[j - 1] {
                match_score
            } else {
                mismatch
            };
            let from_diag = score[i - 1][j - 1] + diag_score;
            let from_up = score[i - 1][j] + gap;
            let from_left = score[i][j - 1] + gap;

            let best = from_diag.max(from_up).max(from_left).max(0);
            score[i][j] = best;

            if best == from_diag {
                dir[i][j] = 0;
            } else if best == from_up {
                dir[i][j] = 1;
            } else if best == from_left {
                dir[i][j] = 2;
            }

            if best > best_score {
                best_score = best;
                best_i = i;
                best_j = j;
            }
        }
    }

    if best_score == 0 {
        return (0, String::new(), String::new());
    }

    let (aligned_a, aligned_b) =
        traceback_alignment(&a_chars, &b_chars, &dir, best_i, best_j, true, &score);

    (best_score, aligned_a, aligned_b)
}

// ─────────────────────────────────────────────────────────────────────────────
// Needleman-Wunsch
// ─────────────────────────────────────────────────────────────────────────────

/// Needleman-Wunsch global sequence alignment.
///
/// Returns `(best_score, aligned_a, aligned_b)`.
pub fn needleman_wunsch(
    a: &str,
    b: &str,
    match_score: i32,
    mismatch: i32,
    gap: i32,
) -> (i32, String, String) {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut score = vec![vec![0i32; n + 1]; m + 1];
    let mut dir = vec![vec![0u8; n + 1]; m + 1];

    // Initialise first row and column with gap penalties
    for i in 0..=m {
        score[i][0] = i as i32 * gap;
    }
    for j in 0..=n {
        score[0][j] = j as i32 * gap;
        if j > 0 {
            dir[0][j] = 2; // left
        }
    }
    for i in 1..=m {
        dir[i][0] = 1; // up
    }

    for i in 1..=m {
        for j in 1..=n {
            let diag_score = if a_chars[i - 1] == b_chars[j - 1] {
                match_score
            } else {
                mismatch
            };
            let from_diag = score[i - 1][j - 1] + diag_score;
            let from_up = score[i - 1][j] + gap;
            let from_left = score[i][j - 1] + gap;

            let best = from_diag.max(from_up).max(from_left);
            score[i][j] = best;

            if best == from_diag {
                dir[i][j] = 0;
            } else if best == from_up {
                dir[i][j] = 1;
            } else {
                dir[i][j] = 2;
            }
        }
    }

    let final_score = score[m][n];
    let (aligned_a, aligned_b) = traceback_alignment(&a_chars, &b_chars, &dir, m, n, false, &score);

    (final_score, aligned_a, aligned_b)
}

// ─────────────────────────────────────────────────────────────────────────────
// Fuzzy matching
// ─────────────────────────────────────────────────────────────────────────────

/// Find the candidate from `candidates` whose Jaro-Winkler similarity to
/// `query` is highest.
///
/// Returns `None` if `candidates` is empty.
pub fn fuzzy_match<'a>(query: &str, candidates: &[&'a str]) -> Option<(&'a str, f64)> {
    candidates
        .iter()
        .map(|&c| (c, jaro_winkler(query, c, 0.1)))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

/// Return all candidates whose Jaro-Winkler similarity to `query` is at or
/// above `threshold`, sorted by score descending.
pub fn fuzzy_search<'a>(
    query: &str,
    candidates: &[&'a str],
    threshold: f64,
) -> Vec<(&'a str, f64)> {
    let mut results: Vec<(&'a str, f64)> = candidates
        .iter()
        .map(|&c| (c, jaro_winkler(query, c, 0.1)))
        .filter(|(_, s)| *s >= threshold)
        .collect();
    results.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Levenshtein ──────────────────────────────────────────────────────

    #[test]
    fn test_levenshtein_kitten_sitting() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", ""), 0);
    }

    #[test]
    fn test_weighted_levenshtein_equal_costs() {
        let w = weighted_levenshtein("kitten", "sitting", 1.0, 1.0, 1.0);
        assert!((w - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_levenshtein_custom_costs() {
        // Sub cost of 2.0 should make substitutions more expensive
        let w_custom = weighted_levenshtein("ab", "cd", 1.0, 1.0, 2.0);
        let w_standard = weighted_levenshtein("ab", "cd", 1.0, 1.0, 1.0);
        assert!(w_custom >= w_standard);
    }

    // ── Damerau-Levenshtein ──────────────────────────────────────────────

    #[test]
    fn test_damerau_levenshtein_transposition() {
        // "ab" → "ba" is a single transposition
        assert_eq!(damerau_levenshtein("ab", "ba"), 1);
    }

    #[test]
    fn test_damerau_levenshtein_kitten() {
        // Same as levenshtein for no transpositions
        assert_eq!(damerau_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_damerau_levenshtein_empty() {
        assert_eq!(damerau_levenshtein("", "abc"), 3);
        assert_eq!(damerau_levenshtein("xyz", ""), 3);
    }

    // ── Jaro / Jaro-Winkler ──────────────────────────────────────────────

    #[test]
    fn test_jaro_identical() {
        assert!((jaro("hello", "hello") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaro_empty() {
        assert_eq!(jaro("", ""), 1.0);
        assert_eq!(jaro("", "abc"), 0.0);
        assert_eq!(jaro("abc", ""), 0.0);
    }

    #[test]
    fn test_jaro_dwayne_duane() {
        // Known value from Wikipedia
        let s = jaro("DWAYNE", "DUANE");
        assert!(s > 0.8 && s < 1.0, "jaro DWAYNE/DUANE = {}", s);
    }

    #[test]
    fn test_jaro_winkler_prefix_boost() {
        let j = jaro("PREFIX", "PREFIXY");
        let jw = jaro_winkler("PREFIX", "PREFIXY", 0.1);
        assert!(jw >= j, "JW should be >= Jaro");
    }

    #[test]
    fn test_jaro_winkler_identical() {
        assert!((jaro_winkler("same", "same", 0.1) - 1.0).abs() < 1e-9);
    }

    // ── Normalised Levenshtein ───────────────────────────────────────────

    #[test]
    fn test_normalized_levenshtein_identical() {
        assert_eq!(normalized_levenshtein("abc", "abc"), 0.0);
    }

    #[test]
    fn test_normalized_levenshtein_range() {
        let d = normalized_levenshtein("kitten", "sitting");
        assert!((0.0..=1.0).contains(&d));
    }

    #[test]
    fn test_normalized_levenshtein_empty() {
        assert_eq!(normalized_levenshtein("", ""), 0.0);
        assert_eq!(normalized_levenshtein("", "abc"), 1.0);
    }

    // ── LCS ─────────────────────────────────────────────────────────────

    #[test]
    fn test_lcs_length_basic() {
        assert_eq!(lcs_length("ABCBDAB", "BDCABA"), 4);
    }

    #[test]
    fn test_lcs_length_empty() {
        assert_eq!(lcs_length("", "abc"), 0);
        assert_eq!(lcs_length("abc", ""), 0);
    }

    #[test]
    fn test_lcs_string() {
        let result = lcs("ABCBDAB", "BDCABA");
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_lcs_identical() {
        assert_eq!(lcs("hello", "hello"), "hello");
    }

    // ── Smith-Waterman ───────────────────────────────────────────────────

    #[test]
    fn test_smith_waterman_basic() {
        let (score, a_aligned, b_aligned) = smith_waterman("ACGT", "ACGT", 2, -1, -1);
        assert!(score > 0, "score should be positive");
        assert!(!a_aligned.is_empty());
        assert!(!b_aligned.is_empty());
    }

    #[test]
    fn test_smith_waterman_no_match() {
        let (score, a_aligned, b_aligned) = smith_waterman("AAAA", "TTTT", 2, -5, -5);
        assert_eq!(score, 0);
        assert!(a_aligned.is_empty());
        assert!(b_aligned.is_empty());
    }

    #[test]
    fn test_smith_waterman_partial_match() {
        // Local alignment should find "ACG" common subsequence
        let (score, _, _) = smith_waterman("XACGX", "YACGY", 2, -1, -1);
        assert!(score > 0);
    }

    // ── Needleman-Wunsch ─────────────────────────────────────────────────

    #[test]
    fn test_needleman_wunsch_identical() {
        let (score, a_aligned, b_aligned) = needleman_wunsch("ACG", "ACG", 2, -1, -1);
        assert!(score > 0);
        assert_eq!(a_aligned, b_aligned);
        assert_eq!(a_aligned, "ACG");
    }

    #[test]
    fn test_needleman_wunsch_with_gap() {
        let (score, a_aligned, b_aligned) = needleman_wunsch("AC", "AGC", 2, -1, -1);
        assert_eq!(
            a_aligned.len(),
            b_aligned.len(),
            "aligned lengths must match"
        );
        // Should contain a gap somewhere
        let _ = score;
    }

    #[test]
    fn test_needleman_wunsch_empty_strings() {
        let (score, a_aligned, b_aligned) = needleman_wunsch("", "", 2, -1, -1);
        assert_eq!(score, 0);
        assert!(a_aligned.is_empty());
        assert!(b_aligned.is_empty());
    }

    // ── Fuzzy matching ───────────────────────────────────────────────────

    #[test]
    fn test_fuzzy_match_basic() {
        let candidates = ["apple", "application", "banana", "appetizer"];
        let result = fuzzy_match("appel", &candidates);
        assert!(result.is_some());
        let (best, score) = result.unwrap();
        assert!(!best.is_empty());
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_fuzzy_match_empty_candidates() {
        let result = fuzzy_match("query", &[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_fuzzy_search_threshold() {
        let candidates = ["hello", "helo", "world", "word"];
        let results = fuzzy_search("hello", &candidates, 0.9);
        // At least "hello" itself should be above 0.9
        assert!(!results.is_empty());
        // Results should be sorted descending by score
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn test_fuzzy_search_no_results() {
        let candidates = ["xyz", "abc", "def"];
        let results = fuzzy_search("hello", &candidates, 0.99);
        assert!(results.is_empty());
    }
}
