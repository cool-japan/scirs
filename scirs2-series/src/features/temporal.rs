//! Temporal pattern features for time series analysis
//!
//! This module provides comprehensive temporal pattern detection including
//! motif discovery, discord detection, SAX representation, and shapelet extraction
//! for discriminative pattern analysis.

use ndarray::{s, Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::utils::{calculate_entropy, euclidean_distance_subsequence, gaussian_breakpoints};
use crate::error::{Result, TimeSeriesError};

/// Temporal pattern features for time series analysis
#[derive(Debug, Clone)]
pub struct TemporalPatternFeatures<F> {
    /// Motifs (frequently occurring patterns)
    pub motifs: Vec<MotifInfo<F>>,
    /// Discord (unusual patterns)
    pub discord_scores: Array1<F>,
    /// SAX representation
    pub sax_symbols: Vec<char>,
    /// Shapelets (discriminative subsequences)
    pub shapelets: Vec<ShapeletInfo<F>>,
}

impl<F> Default for TemporalPatternFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            motifs: Vec::new(),
            discord_scores: Array1::zeros(0),
            sax_symbols: Vec::new(),
            shapelets: Vec::new(),
        }
    }
}

/// Information about discovered motifs
#[derive(Debug, Clone)]
pub struct MotifInfo<F> {
    /// Pattern length
    pub length: usize,
    /// Locations where motif occurs
    pub positions: Vec<usize>,
    /// Frequency of occurrence
    pub frequency: usize,
    /// Average distance between instances
    pub avg_distance: F,
}

impl<F> Default for MotifInfo<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            length: 0,
            positions: Vec::new(),
            frequency: 0,
            avg_distance: F::zero(),
        }
    }
}

/// Information about shapelets
#[derive(Debug, Clone)]
pub struct ShapeletInfo<F> {
    /// Shapelet subsequence
    pub pattern: Array1<F>,
    /// Starting position in original series
    pub position: usize,
    /// Length of shapelet
    pub length: usize,
    /// Information gain or discriminative power
    pub information_gain: F,
}

impl<F> Default for ShapeletInfo<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            pattern: Array1::zeros(0),
            position: 0,
            length: 0,
            information_gain: F::zero(),
        }
    }
}

// =============================================================================
// Main Calculation Functions
// =============================================================================

/// Calculate temporal pattern features
pub fn calculate_temporal_pattern_features<F>(
    ts: &Array1<F>,
    motif_length: Option<usize>,
    max_motifs: usize,
    k_neighbors: usize,
    sax_word_length: usize,
    sax_alphabet_size: usize,
    detect_patterns: bool,
) -> Result<TemporalPatternFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Clone + ndarray::ScalarOperand,
{
    if !detect_patterns {
        return Ok(TemporalPatternFeatures::default());
    }

    let actual_motif_length = motif_length.unwrap_or(ts.len() / 10).max(3);

    // Discover motifs
    let motifs = discover_motifs(ts, actual_motif_length, max_motifs)?;

    // Calculate discord scores
    let discord_scores = calculate_discord_scores(ts, actual_motif_length, k_neighbors)?;

    // Convert to SAX representation
    let sax_symbols = time_series_to_sax(ts, sax_word_length, sax_alphabet_size)?;

    // For shapelets, we would need labeled data from multiple classes
    // For now, return empty shapelets
    let shapelets = Vec::new();

    Ok(TemporalPatternFeatures {
        motifs,
        discord_scores,
        sax_symbols,
        shapelets,
    })
}

// =============================================================================
// Motif Discovery
// =============================================================================

/// Discover motifs (frequently occurring patterns) in time series
///
/// This function uses a brute-force approach to find the most frequently
/// occurring subsequences of a given length in the time series.
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `motif_length` - Length of motifs to discover
/// * `max_motifs` - Maximum number of motifs to return
///
/// # Returns
///
/// * Vector of discovered motifs with their locations and frequencies
pub fn discover_motifs<F>(
    ts: &Array1<F>,
    motif_length: usize,
    max_motifs: usize,
) -> Result<Vec<MotifInfo<F>>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    if n < motif_length * 2 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series too short for motif discovery".to_string(),
        ));
    }

    let num_subsequences = n - motif_length + 1;
    let mut distances = Array2::zeros((num_subsequences, num_subsequences));

    // Calculate distance matrix between all subsequences
    for i in 0..num_subsequences {
        for j in (i + 1)..num_subsequences {
            let dist = euclidean_distance_subsequence(ts, i, j, motif_length);
            distances[[i, j]] = dist;
            distances[[j, i]] = dist;
        }
    }

    let mut motifs = Vec::new();
    let mut used_indices = vec![false; num_subsequences];

    for _ in 0..max_motifs {
        let mut min_dist = F::infinity();
        let mut best_pair = (0, 0);

        // Find the closest pair of unused subsequences
        for i in 0..num_subsequences {
            if used_indices[i] {
                continue;
            }
            for j in (i + motif_length)..num_subsequences {
                if used_indices[j] {
                    continue;
                }
                if distances[[i, j]] < min_dist {
                    min_dist = distances[[i, j]];
                    best_pair = (i, j);
                }
            }
        }

        if min_dist.is_infinite() {
            break;
        }

        // Find all subsequences similar to this motif pair
        let threshold = min_dist * F::from(1.5).unwrap();
        let mut positions = vec![best_pair.0, best_pair.1];

        for k in 0..num_subsequences {
            if used_indices[k] || k == best_pair.0 || k == best_pair.1 {
                continue;
            }

            let dist_to_first = distances[[best_pair.0, k]];
            let dist_to_second = distances[[best_pair.1, k]];

            if dist_to_first <= threshold || dist_to_second <= threshold {
                positions.push(k);
            }
        }

        // Mark these positions as used
        for &pos in &positions {
            for offset in 0..motif_length {
                if pos + offset < used_indices.len() {
                    used_indices[pos + offset] = true;
                }
            }
        }

        // Calculate average distance
        let mut total_dist = F::zero();
        let mut count = 0;
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                total_dist = total_dist + distances[[positions[i], positions[j]]];
                count += 1;
            }
        }

        let avg_distance = if count > 0 {
            total_dist / F::from(count).unwrap()
        } else {
            F::zero()
        };

        motifs.push(MotifInfo {
            length: motif_length,
            frequency: positions.len(),
            positions,
            avg_distance,
        });
    }

    Ok(motifs)
}

// =============================================================================
// Discord Detection
// =============================================================================

/// Calculate discord scores for anomalous subsequences
///
/// Discord scores identify unusual or anomalous patterns in the time series
/// by measuring how far each subsequence is from its nearest neighbors.
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `discord_length` - Length of discord subsequences
/// * `k_neighbors` - Number of nearest neighbors to consider
///
/// # Returns
///
/// * Array of discord scores for each position
pub fn calculate_discord_scores<F>(
    ts: &Array1<F>,
    discord_length: usize,
    k_neighbors: usize,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    if n < discord_length * 2 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series too short for discord detection".to_string(),
        ));
    }

    let num_subsequences = n - discord_length + 1;
    let mut discord_scores = Array1::zeros(num_subsequences);

    for i in 0..num_subsequences {
        let mut distances = Vec::new();

        // Calculate distances to all other subsequences
        for j in 0..num_subsequences {
            if (i as i32 - j as i32).abs() < discord_length as i32 {
                continue; // Skip overlapping subsequences
            }

            let dist = euclidean_distance_subsequence(ts, i, j, discord_length);
            distances.push(dist);
        }

        // Sort distances and take k nearest neighbors
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if distances.len() >= k_neighbors {
            // Discord score is the distance to the k-th nearest neighbor
            discord_scores[i] = distances[k_neighbors - 1];
        } else if !distances.is_empty() {
            discord_scores[i] = distances[distances.len() - 1];
        }
    }

    Ok(discord_scores)
}

// =============================================================================
// SAX (Symbolic Aggregate approXimation)
// =============================================================================

/// Convert time series to SAX (Symbolic Aggregate approXimation) representation
///
/// SAX converts time series data into a symbolic representation using a finite
/// alphabet, which enables efficient pattern matching and data mining.
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `word_length` - Length of SAX words
/// * `alphabet_size` - Size of the alphabet (number of symbols)
///
/// # Returns
///
/// * Vector of SAX symbols
pub fn time_series_to_sax<F>(
    ts: &Array1<F>,
    word_length: usize,
    alphabet_size: usize,
) -> Result<Vec<char>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    if n < word_length {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series too short for SAX conversion".to_string(),
        ));
    }

    if !(2..=26).contains(&alphabet_size) {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Alphabet size must be between 2 and 26".to_string(),
        ));
    }

    // Z-normalize the time series
    let mean = ts.sum() / F::from(n).unwrap();
    let variance = ts.mapv(|x| (x - mean) * (x - mean)).sum() / F::from(n).unwrap();
    let std_dev = variance.sqrt();

    let normalized = if std_dev > F::zero() {
        ts.mapv(|x| (x - mean) / std_dev)
    } else {
        Array1::zeros(n)
    };

    // PAA (Piecewise Aggregate Approximation)
    let segment_size = n / word_length;
    let mut paa = Array1::zeros(word_length);

    for i in 0..word_length {
        let start = i * segment_size;
        let end = if i == word_length - 1 {
            n
        } else {
            (i + 1) * segment_size
        };

        let segment_sum = normalized.slice(s![start..end]).sum();
        let segment_len = end - start;
        paa[i] = segment_sum / F::from(segment_len).unwrap();
    }

    // Convert to symbols using Gaussian breakpoints
    let breakpoints = gaussian_breakpoints(alphabet_size);
    let mut sax_symbols = Vec::with_capacity(word_length);

    for &value in paa.iter() {
        let symbol_index = breakpoints
            .iter()
            .position(|&bp| value.to_f64().unwrap_or(0.0) <= bp)
            .unwrap_or(alphabet_size - 1);

        let symbol = (b'a' + symbol_index as u8) as char;
        sax_symbols.push(symbol);
    }

    Ok(sax_symbols)
}

// =============================================================================
// Shapelet Extraction
// =============================================================================

/// Extract shapelets (discriminative subsequences) from time series
///
/// Shapelets are time series subsequences that are maximally representative
/// of a class. This function requires labeled data from multiple classes.
///
/// # Arguments
///
/// * `ts_class1` - Time series from class 1
/// * `ts_class2` - Time series from class 2  
/// * `min_length` - Minimum shapelet length
/// * `max_length` - Maximum shapelet length
/// * `max_shapelets` - Maximum number of shapelets to return
///
/// # Returns
///
/// * Vector of discovered shapelets
pub fn extract_shapelets<F>(
    ts_class1: &[Array1<F>],
    ts_class2: &[Array1<F>],
    min_length: usize,
    max_length: usize,
    max_shapelets: usize,
) -> Result<Vec<ShapeletInfo<F>>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    if ts_class1.is_empty() || ts_class2.is_empty() {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Need at least one time series from each class".to_string(),
        ));
    }

    let mut all_candidates = Vec::new();

    // Generate candidate shapelets from class 1
    for ts in ts_class1.iter() {
        for length in min_length..=max_length.min(ts.len() / 2) {
            for start in 0..=(ts.len() - length) {
                let shapelet = ts.slice(s![start..start + length]).to_owned();

                // Calculate information gain
                let info_gain =
                    calculate_shapelet_information_gain(&shapelet, ts_class1, ts_class2)?;

                all_candidates.push(ShapeletInfo {
                    pattern: shapelet,
                    position: start,
                    length,
                    information_gain: info_gain,
                });
            }
        }
    }

    // Sort by information gain and take the best ones
    all_candidates.sort_by(|a, b| {
        b.information_gain
            .partial_cmp(&a.information_gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    all_candidates.truncate(max_shapelets);
    Ok(all_candidates)
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Calculate information gain for a shapelet candidate
fn calculate_shapelet_information_gain<F>(
    shapelet: &Array1<F>,
    ts_class1: &[Array1<F>],
    ts_class2: &[Array1<F>],
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let total_count = ts_class1.len() + ts_class2.len();
    let class1_count = ts_class1.len();
    let class2_count = ts_class2.len();

    if total_count == 0 {
        return Ok(F::zero());
    }

    // Calculate original entropy
    let p1 = class1_count as f64 / total_count as f64;
    let p2 = class2_count as f64 / total_count as f64;
    let original_entropy = if p1 > 0.0 && p2 > 0.0 {
        -(p1 * p1.ln() + p2 * p2.ln())
    } else {
        0.0
    };

    // Find best threshold by calculating distances to shapelet
    let mut distances_class1 = Vec::new();
    let mut distances_class2 = Vec::new();

    for ts in ts_class1 {
        let min_dist = find_min_distance_to_shapelet(ts, shapelet);
        distances_class1.push(min_dist);
    }

    for ts in ts_class2 {
        let min_dist = find_min_distance_to_shapelet(ts, shapelet);
        distances_class2.push(min_dist);
    }

    // Try different thresholds to find the best split
    let mut all_distances: Vec<F> = distances_class1
        .iter()
        .cloned()
        .chain(distances_class2.iter().cloned())
        .collect();
    all_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut best_info_gain = F::zero();

    for &threshold in &all_distances {
        // Count instances in each split
        let left_class1 = distances_class1.iter().filter(|&&d| d <= threshold).count();
        let left_class2 = distances_class2.iter().filter(|&&d| d <= threshold).count();
        let right_class1 = class1_count - left_class1;
        let right_class2 = class2_count - left_class2;

        let left_total = left_class1 + left_class2;
        let right_total = right_class1 + right_class2;

        if left_total == 0 || right_total == 0 {
            continue;
        }

        // Calculate weighted entropy for this split
        let left_entropy = calculate_entropy(left_class1, left_class2);
        let right_entropy = calculate_entropy(right_class1, right_class2);

        let weighted_entropy = (left_total as f64 / total_count as f64) * left_entropy
            + (right_total as f64 / total_count as f64) * right_entropy;

        let info_gain = original_entropy - weighted_entropy;

        if F::from(info_gain).unwrap() > best_info_gain {
            best_info_gain = F::from(info_gain).unwrap();
        }
    }

    Ok(best_info_gain)
}

/// Find minimum distance from a time series to a shapelet
fn find_min_distance_to_shapelet<F>(ts: &Array1<F>, shapelet: &Array1<F>) -> F
where
    F: Float + FromPrimitive,
{
    let shapelet_len = shapelet.len();
    let ts_len = ts.len();

    if ts_len < shapelet_len {
        return F::infinity();
    }

    let mut min_distance = F::infinity();

    for start in 0..=(ts_len - shapelet_len) {
        let mut sum = F::zero();
        for i in 0..shapelet_len {
            let diff = ts[start + i] - shapelet[i];
            sum = sum + diff * diff;
        }
        let distance = sum.sqrt();

        if distance < min_distance {
            min_distance = distance;
        }
    }

    min_distance
}

/// Calculate distance matrix for time series subsequences
pub fn calculate_distance_matrix<F>(ts: &Array1<F>, subsequence_length: usize) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    if n < subsequence_length {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series too short for distance matrix calculation".to_string(),
        ));
    }

    let num_subsequences = n - subsequence_length + 1;
    let mut distances = Array2::zeros((num_subsequences, num_subsequences));

    for i in 0..num_subsequences {
        for j in (i + 1)..num_subsequences {
            let dist = euclidean_distance_subsequence(ts, i, j, subsequence_length);
            distances[[i, j]] = dist;
            distances[[j, i]] = dist;
        }
    }

    Ok(distances)
}

/// Find nearest neighbors for each subsequence
pub fn find_nearest_neighbors<F>(distance_matrix: &Array2<F>, k: usize) -> Result<Vec<Vec<usize>>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = distance_matrix.nrows();
    let mut neighbors = Vec::with_capacity(n);

    for i in 0..n {
        let mut distances_with_indices: Vec<(F, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (distance_matrix[[i, j]], j))
            .collect();

        distances_with_indices
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let nearest_k: Vec<usize> = distances_with_indices
            .into_iter()
            .take(k)
            .map(|(_, idx)| idx)
            .collect();

        neighbors.push(nearest_k);
    }

    Ok(neighbors)
}

/// Calculate local intrinsic dimensionality for each subsequence
pub fn calculate_local_intrinsic_dimensionality<F>(
    distance_matrix: &Array2<F>,
    k: usize,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = distance_matrix.nrows();
    let mut lid_values = Vec::with_capacity(n);

    for i in 0..n {
        let mut distances: Vec<F> = (0..n)
            .filter(|&j| j != i)
            .map(|j| distance_matrix[[i, j]])
            .collect();

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if distances.len() < k || distances[k - 1] == F::zero() {
            lid_values.push(F::zero());
            continue;
        }

        // Calculate LID using the maximum likelihood estimator
        let mut sum = F::zero();
        for j in 0..k {
            if distances[j] > F::zero() && distances[k - 1] > F::zero() {
                sum = sum + (distances[k - 1] / distances[j]).ln();
            }
        }

        let lid = F::from(k).unwrap() / sum;
        lid_values.push(lid);
    }

    Ok(lid_values)
}
