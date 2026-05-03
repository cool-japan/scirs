//! STS (Semantic Textual Similarity) benchmark evaluation.
//!
//! Provides `sts_evaluate`, `load_sts_from_tsv`, `StsReport`, `StsDatasetFormat`.
//!
//! Protocol-only: this module evaluates pairs already loaded into memory.
//! No dataset is downloaded automatically.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_text::evaluation::sts::{sts_evaluate, StsReport};
//! use scirs2_core::ndarray::Array1;
//!
//! let pairs: Vec<(Vec<String>, Vec<String>, f32)> = vec![
//!     (
//!         vec!["hello".into()],
//!         vec!["hello".into()],
//!         5.0,
//!     ),
//! ];
//! let embed = |tokens: &[String]| {
//!     let mut v = Array1::zeros(tokens.len().max(1));
//!     for (i, _) in tokens.iter().enumerate() {
//!         v[i] = 1.0f32;
//!     }
//!     v
//! };
//! let report = sts_evaluate(&embed, &pairs).unwrap();
//! assert!(report.n_pairs == 1);
//! ```

use crate::error::{Result, TextError};
use scirs2_core::ndarray::Array1;
use std::path::Path;

/// Dataset format variants for documentation / future parsing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StsDatasetFormat {
    /// STS-B format: score 0–5, tab-separated with columns
    /// `idx\tgenre\tfile\tyear\tsid\tscore\tsentence1\tsentence2`
    StsB,
    /// SICK format: score 1–5
    Sick,
    /// STS 2012–2016 format: score 0–5
    Sts12to16,
}

/// Report returned by [`sts_evaluate`].
#[derive(Debug, Clone)]
pub struct StsReport {
    /// Pearson correlation between cosine-similarity predictions and gold scores.
    pub pearson: f32,
    /// Spearman rank correlation between predictions and gold scores.
    pub spearman: f32,
    /// Mean squared error between predictions and gold scores.
    pub mse: f32,
    /// Cosine-similarity prediction for each pair (same order as input).
    pub predictions: Vec<f32>,
    /// Gold similarity labels (same order as input).
    pub gold_labels: Vec<f32>,
    /// Total number of sentence pairs evaluated.
    pub n_pairs: usize,
}

type StsPairs = Vec<(Vec<String>, Vec<String>, f32)>;

/// Load STS sentence pairs from a TSV file.
///
/// The parser tries multiple common column layouts:
/// - 3 columns → `score`, `sentence1`, `sentence2`
/// - 8+ columns (STS-B style) → score at index 4, sentence1 at 5, sentence2 at 6
///
/// Each sentence is tokenized by splitting on whitespace.
///
/// # Errors
///
/// Returns [`TextError::IoError`] if the file cannot be opened or read.
pub fn load_sts_from_tsv(path: impl AsRef<Path>) -> Result<StsPairs> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(path.as_ref()).map_err(|e| TextError::IoError(e.to_string()))?;
    let reader = BufReader::new(file);
    let mut pairs = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| TextError::IoError(e.to_string()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();

        // Determine column layout
        let (score_str, s1, s2) = if fields.len() >= 8 {
            (fields[4], fields[5], fields[6])
        } else if fields.len() >= 3 {
            (fields[0], fields[1], fields[2])
        } else {
            continue;
        };

        let score: f32 = match score_str.trim().parse() {
            Ok(v) => v,
            Err(_) => continue, // skip header or malformed lines
        };

        let tokens1: Vec<String> = s1.split_whitespace().map(str::to_owned).collect();
        let tokens2: Vec<String> = s2.split_whitespace().map(str::to_owned).collect();
        pairs.push((tokens1, tokens2, score));
    }

    Ok(pairs)
}

/// Compute Pearson correlation coefficient between two slices.
fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f32>() / n;
    let my = y.iter().sum::<f32>() / n;
    let num: f32 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let da: f32 = x.iter().map(|a| (a - mx).powi(2)).sum::<f32>().sqrt();
    let db: f32 = y.iter().map(|b| (b - my).powi(2)).sum::<f32>().sqrt();
    if da == 0.0 || db == 0.0 {
        0.0
    } else {
        num / (da * db)
    }
}

/// Compute Spearman rank correlation via rank transform followed by Pearson.
fn spearman_correlation(x: &[f32], y: &[f32]) -> f32 {
    /// Rank a slice (1-based, ties receive fractional average ranks).
    fn rank(v: &[f32]) -> Vec<f32> {
        let mut indexed: Vec<(usize, f32)> = v.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut ranks = vec![0.0f32; v.len()];
        let mut i = 0;
        while i < indexed.len() {
            // Find all ties
            let val = indexed[i].1;
            let mut j = i + 1;
            while j < indexed.len() && indexed[j].1 == val {
                j += 1;
            }
            // Average rank for the tie group
            let avg_rank = (i + j + 1) as f32 / 2.0; // 1-based midpoint
            for item in &indexed[i..j] {
                ranks[item.0] = avg_rank;
            }
            i = j;
        }
        ranks
    }

    let rx = rank(x);
    let ry = rank(y);
    pearson_correlation(&rx, &ry)
}

/// Evaluate semantic textual similarity using cosine similarity of embeddings vs gold labels.
///
/// `embed_fn` maps a token list to a dense embedding vector.
/// `pairs` is a slice of `(tokens1, tokens2, gold_score)`.
///
/// Returns [`StsReport`] with Pearson/Spearman correlations, MSE, and raw predictions.
///
/// # Errors
///
/// Returns [`TextError::InvalidInput`] if `pairs` is empty.
pub fn sts_evaluate(
    embed_fn: &dyn Fn(&[String]) -> Array1<f32>,
    pairs: &[(Vec<String>, Vec<String>, f32)],
) -> Result<StsReport> {
    if pairs.is_empty() {
        return Err(TextError::InvalidInput(
            "STS dataset is empty; at least one pair is required".into(),
        ));
    }

    let mut predictions = Vec::with_capacity(pairs.len());
    let mut gold_labels = Vec::with_capacity(pairs.len());

    for (s1_tokens, s2_tokens, gold) in pairs {
        let e1 = embed_fn(s1_tokens);
        let e2 = embed_fn(s2_tokens);

        let dot = e1.dot(&e2);
        let n1 = e1.dot(&e1).sqrt();
        let n2 = e2.dot(&e2).sqrt();
        let cosine = if n1 == 0.0 || n2 == 0.0 {
            0.0f32
        } else {
            dot / (n1 * n2)
        };

        predictions.push(cosine);
        gold_labels.push(*gold);
    }

    let pearson = pearson_correlation(&predictions, &gold_labels);
    let spearman = spearman_correlation(&predictions, &gold_labels);
    let mse = predictions
        .iter()
        .zip(&gold_labels)
        .map(|(p, g)| (p - g).powi(2))
        .sum::<f32>()
        / predictions.len() as f32;

    Ok(StsReport {
        pearson,
        spearman,
        mse,
        predictions,
        gold_labels,
        n_pairs: pairs.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    /// Simple bag-of-words embed: returns a fixed-dim vector with 1.0 for each token present.
    fn bow_embed(tokens: &[String], dim: usize) -> Array1<f32> {
        let mut v = Array1::zeros(dim);
        for (i, _tok) in tokens.iter().enumerate() {
            let idx = i % dim;
            v[idx] += 1.0;
        }
        v
    }

    #[test]
    fn sts_empty_returns_error() {
        let result = sts_evaluate(&|t| bow_embed(t, 4), &[]);
        assert!(result.is_err());
    }

    #[test]
    fn sts_single_pair_identical_tokens() {
        let pairs = vec![(vec!["cat".to_string()], vec!["cat".to_string()], 5.0f32)];
        let report = sts_evaluate(&|t| bow_embed(t, 4), &pairs).expect("evaluate");
        assert_eq!(report.n_pairs, 1);
        // cosine of identical vectors is 1.0
        assert!((report.predictions[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sts_mse_is_non_negative() {
        let pairs = vec![
            (vec!["a".to_string()], vec!["b".to_string()], 2.5f32),
            (vec!["c".to_string()], vec!["c".to_string()], 4.0f32),
        ];
        let report = sts_evaluate(&|t| bow_embed(t, 4), &pairs).expect("evaluate");
        assert!(report.mse >= 0.0);
    }
}
