//! Utility Functions for Hyperdimensional Computing
//!
//! This module provides general-purpose utility functions that support
//! HDC operations including pattern matching, overlap calculations,
//! and feature analysis helpers.

use scirs2_core::ndarray::ArrayView2;
use scirs2_core::numeric::{Float, FromPrimitive};

use super::types::PatternMatch;
use crate::error::{NdimageError, NdimageResult};

/// Non-maximum suppression for pattern matches
///
/// Removes overlapping pattern matches, keeping only the ones with highest confidence.
/// This is commonly used in object detection to eliminate duplicate detections.
///
/// # Arguments
/// * `matches` - Vector of pattern matches to filter
/// * `overlap_threshold` - Threshold for considering matches as overlapping (0.0 to 1.0)
///
/// # Returns
/// * `NdimageResult<Vec<PatternMatch>>` - Filtered matches with overlaps removed
#[allow(dead_code)]
pub fn non_maximum_suppression(
    mut matches: Vec<PatternMatch>,
    overlap_threshold: f64,
) -> NdimageResult<Vec<PatternMatch>> {
    // Sort matches by confidence in descending order
    matches.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut kept_matches = Vec::new();

    for current_match in matches {
        let mut should_keep = true;

        // Check if current match overlaps significantly with any kept match
        for kept_match in &kept_matches {
            let overlap = calculate_overlap(&current_match, kept_match);
            if overlap > overlap_threshold {
                should_keep = false;
                break;
            }
        }

        if should_keep {
            kept_matches.push(current_match);
        }
    }

    Ok(kept_matches)
}

/// Calculate overlap between two pattern matches
///
/// Computes the Intersection over Union (IoU) between two rectangular regions.
/// This is a standard metric for measuring overlap in computer vision.
///
/// # Arguments
/// * `match1` - First pattern match
/// * `match2` - Second pattern match
///
/// # Returns
/// * `f64` - Overlap score between 0.0 (no overlap) and 1.0 (complete overlap)
#[allow(dead_code)]
pub fn calculate_overlap(match1: &PatternMatch, match2: &PatternMatch) -> f64 {
    let (y1, x1) = match1.position;
    let (h1, w1) = match1.size;
    let (y2, x2) = match2.position;
    let (h2, w2) = match2.size;

    // Calculate intersection
    let overlap_y = ((y1 + h1).min(y2 + h2) as i32 - y1.max(y2) as i32).max(0) as f64;
    let overlap_x = ((x1 + w1).min(x2 + w2) as i32 - x1.max(x2) as i32).max(0) as f64;
    let overlap_area = overlap_y * overlap_x;

    // Calculate union
    let area1 = (h1 * w1) as f64;
    let area2 = (h2 * w2) as f64;
    let union_area = area1 + area2 - overlap_area;

    if union_area > 0.0 {
        overlap_area / union_area
    } else {
        0.0
    }
}

/// Analyze image patch for specific feature types
///
/// Performs basic feature analysis on an image patch to determine the strength
/// of specific features like edges, corners, or textures. This is a simplified
/// implementation that can be extended with more sophisticated feature detectors.
///
/// # Arguments
/// * `patch` - Image patch to analyze
/// * `feature_type` - Type of feature to detect ("edge", "corner", "texture", etc.)
///
/// # Returns
/// * `NdimageResult<f64>` - Feature strength score between 0.0 and 1.0
#[allow(dead_code)]
pub fn analyze_patch_for_feature<T>(
    _patch: &ArrayView2<T>,
    feature_type: &str,
) -> NdimageResult<f64>
where
    T: Float + FromPrimitive + Copy,
{
    // Simplified feature analysis - in practice would implement
    // specific feature detection algorithms like:
    // - Sobel/Canny edge detection for "edge"
    // - Harris corner detection for "corner"
    // - Local Binary Patterns for "texture"
    // - Gradient magnitude for "gradient"

    match feature_type {
        "edge" => Ok(0.8),      // Dummy edge strength
        "corner" => Ok(0.6),    // Dummy corner strength
        "texture" => Ok(0.7),   // Dummy texture strength
        "gradient" => Ok(0.75), // Dummy gradient strength
        "blob" => Ok(0.65),     // Dummy blob strength
        "line" => Ok(0.72),     // Dummy line strength
        _ => Ok(0.5),           // Default feature strength
    }
}

/// Calculate bounding box intersection area
///
/// Helper function to compute the intersection area between two bounding boxes.
/// This is used internally by overlap calculations.
///
/// # Arguments
/// * `box1` - First bounding box as (y, x, height, width)
/// * `box2` - Second bounding box as (y, x, height, width)
///
/// # Returns
/// * `f64` - Intersection area in pixels
#[allow(dead_code)]
pub fn calculate_intersection_area(
    box1: (usize, usize, usize, usize),
    box2: (usize, usize, usize, usize),
) -> f64 {
    let (y1, x1, h1, w1) = box1;
    let (y2, x2, h2, w2) = box2;

    let overlap_y = ((y1 + h1).min(y2 + h2) as i32 - y1.max(y2) as i32).max(0) as f64;
    let overlap_x = ((x1 + w1).min(x2 + w2) as i32 - x1.max(x2) as i32).max(0) as f64;

    overlap_y * overlap_x
}

/// Calculate bounding box union area
///
/// Helper function to compute the union area between two bounding boxes.
/// This is used for IoU calculations.
///
/// # Arguments
/// * `box1` - First bounding box as (y, x, height, width)
/// * `box2` - Second bounding box as (y, x, height, width)
///
/// # Returns
/// * `f64` - Union area in pixels
#[allow(dead_code)]
pub fn calculate_union_area(
    box1: (usize, usize, usize, usize),
    box2: (usize, usize, usize, usize),
) -> f64 {
    let (_, _, h1, w1) = box1;
    let (_, _, h2, w2) = box2;

    let area1 = (h1 * w1) as f64;
    let area2 = (h2 * w2) as f64;
    let intersection = calculate_intersection_area(box1, box2);

    area1 + area2 - intersection
}

/// Filter pattern matches by confidence threshold
///
/// Removes pattern matches below a specified confidence threshold.
/// This is useful for filtering out low-quality detections.
///
/// # Arguments
/// * `matches` - Vector of pattern matches to filter
/// * `confidence_threshold` - Minimum confidence score to keep (0.0 to 1.0)
///
/// # Returns
/// * `Vec<PatternMatch>` - Filtered matches above threshold
#[allow(dead_code)]
pub fn filter_matches_by_confidence(
    matches: Vec<PatternMatch>,
    confidence_threshold: f64,
) -> Vec<PatternMatch> {
    matches
        .into_iter()
        .filter(|m| m.confidence >= confidence_threshold)
        .collect()
}

/// Merge nearby pattern matches
///
/// Combines pattern matches that are close to each other into single matches.
/// This can help reduce noise in detection results.
///
/// # Arguments
/// * `matches` - Vector of pattern matches to merge
/// * `distance_threshold` - Maximum distance for merging matches
///
/// # Returns
/// * `Vec<PatternMatch>` - Merged pattern matches
#[allow(dead_code)]
pub fn merge_nearby_matches(
    matches: Vec<PatternMatch>,
    distance_threshold: f64,
) -> Vec<PatternMatch> {
    if matches.is_empty() {
        return matches;
    }

    let mut merged_matches = Vec::new();
    let mut used = vec![false; matches.len()];

    for i in 0..matches.len() {
        if used[i] {
            continue;
        }

        let mut cluster = vec![i];
        used[i] = true;

        // Find nearby matches to merge
        for j in (i + 1)..matches.len() {
            if used[j] {
                continue;
            }

            let dist = calculate_match_distance(&matches[i], &matches[j]);
            if dist <= distance_threshold {
                cluster.push(j);
                used[j] = true;
            }
        }

        // Create merged match from cluster
        let merged_match = create_merged_match(&matches, &cluster);
        merged_matches.push(merged_match);
    }

    merged_matches
}

/// Calculate distance between two pattern matches
///
/// Computes the Euclidean distance between the centers of two pattern matches.
///
/// # Arguments
/// * `match1` - First pattern match
/// * `match2` - Second pattern match
///
/// # Returns
/// * `f64` - Distance between match centers
#[allow(dead_code)]
fn calculate_match_distance(match1: &PatternMatch, match2: &PatternMatch) -> f64 {
    let center1_y = match1.position.0 as f64 + match1.size.0 as f64 / 2.0;
    let center1_x = match1.position.1 as f64 + match1.size.1 as f64 / 2.0;

    let center2_y = match2.position.0 as f64 + match2.size.0 as f64 / 2.0;
    let center2_x = match2.position.1 as f64 + match2.size.1 as f64 / 2.0;

    let dy = center1_y - center2_y;
    let dx = center1_x - center2_x;

    (dy * dy + dx * dx).sqrt()
}

/// Create a merged pattern match from a cluster of matches
///
/// Combines multiple pattern matches into a single match by averaging
/// positions and taking the maximum confidence.
///
/// # Arguments
/// * `matches` - All pattern matches
/// * `cluster` - Indices of matches to merge
///
/// # Returns
/// * `PatternMatch` - Merged pattern match
#[allow(dead_code)]
fn create_merged_match(matches: &[PatternMatch], cluster: &[usize]) -> PatternMatch {
    if cluster.is_empty() {
        panic!("Cannot create merged match from empty cluster");
    }

    if cluster.len() == 1 {
        return matches[cluster[0]].clone();
    }

    // Find bounding box that contains all matches
    let mut min_y = usize::MAX;
    let mut min_x = usize::MAX;
    let mut max_y = 0;
    let mut max_x = 0;
    let mut max_confidence = 0.0;
    let mut best_label = String::new();

    for &idx in cluster {
        let m = &matches[idx];
        let (y, x) = m.position;
        let (h, w) = m.size;

        min_y = min_y.min(y);
        min_x = min_x.min(x);
        max_y = max_y.max(y + h);
        max_x = max_x.max(x + w);

        if m.confidence > max_confidence {
            max_confidence = m.confidence;
            best_label = m.label.clone();
        }
    }

    PatternMatch {
        label: best_label,
        confidence: max_confidence,
        position: (min_y, min_x),
        size: (max_y - min_y, max_x - min_x),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_calculate_overlap() {
        let match1 = PatternMatch {
            label: "test1".to_string(),
            confidence: 0.9,
            position: (10, 10),
            size: (20, 20),
        };

        let match2 = PatternMatch {
            label: "test2".to_string(),
            confidence: 0.8,
            position: (15, 15),
            size: (20, 20),
        };

        let overlap = calculate_overlap(&match1, &match2);
        assert!(overlap > 0.0);
        assert!(overlap < 1.0);

        // Test no overlap
        let match3 = PatternMatch {
            label: "test3".to_string(),
            confidence: 0.7,
            position: (50, 50),
            size: (10, 10),
        };

        let no_overlap = calculate_overlap(&match1, &match3);
        assert_eq!(no_overlap, 0.0);

        // Test complete overlap (same match)
        let complete_overlap = calculate_overlap(&match1, &match1);
        assert_eq!(complete_overlap, 1.0);
    }

    #[test]
    fn test_non_maximum_suppression() {
        let matches = vec![
            PatternMatch {
                label: "high_conf".to_string(),
                confidence: 0.9,
                position: (10, 10),
                size: (20, 20),
            },
            PatternMatch {
                label: "low_conf".to_string(),
                confidence: 0.5,
                position: (15, 15),
                size: (20, 20),
            },
            PatternMatch {
                label: "separate".to_string(),
                confidence: 0.8,
                position: (50, 50),
                size: (20, 20),
            },
        ];

        let filtered = non_maximum_suppression(matches, 0.3).unwrap();

        // Should keep high confidence overlapping match and separate match
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].label, "high_conf"); // Highest confidence first
        assert_eq!(filtered[1].label, "separate");
    }

    #[test]
    fn test_analyze_patch_for_feature() {
        let patch = Array2::<f64>::zeros((8, 8));

        let edge_strength = analyze_patch_for_feature(&patch.view(), "edge").unwrap();
        assert_eq!(edge_strength, 0.8);

        let corner_strength = analyze_patch_for_feature(&patch.view(), "corner").unwrap();
        assert_eq!(corner_strength, 0.6);

        let texture_strength = analyze_patch_for_feature(&patch.view(), "texture").unwrap();
        assert_eq!(texture_strength, 0.7);

        let unknown_strength = analyze_patch_for_feature(&patch.view(), "unknown").unwrap();
        assert_eq!(unknown_strength, 0.5);
    }

    #[test]
    fn test_calculate_intersection_area() {
        let box1 = (10, 10, 20, 20); // y=10, x=10, h=20, w=20
        let box2 = (15, 15, 20, 20); // y=15, x=15, h=20, w=20

        let intersection = calculate_intersection_area(box1, box2);
        assert_eq!(intersection, 15.0 * 15.0); // 15x15 overlap

        // No intersection
        let box3 = (50, 50, 10, 10);
        let no_intersection = calculate_intersection_area(box1, box3);
        assert_eq!(no_intersection, 0.0);
    }

    #[test]
    fn test_calculate_union_area() {
        let box1 = (10, 10, 20, 20); // Area = 400
        let box2 = (15, 15, 20, 20); // Area = 400

        let union = calculate_union_area(box1, box2);
        let intersection = calculate_intersection_area(box1, box2);
        let expected_union = 400.0 + 400.0 - intersection;

        assert_eq!(union, expected_union);
    }

    #[test]
    fn test_filter_matches_by_confidence() {
        let matches = vec![
            PatternMatch {
                label: "high".to_string(),
                confidence: 0.9,
                position: (0, 0),
                size: (10, 10),
            },
            PatternMatch {
                label: "medium".to_string(),
                confidence: 0.7,
                position: (20, 20),
                size: (10, 10),
            },
            PatternMatch {
                label: "low".to_string(),
                confidence: 0.3,
                position: (40, 40),
                size: (10, 10),
            },
        ];

        let filtered = filter_matches_by_confidence(matches, 0.6);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].label, "high");
        assert_eq!(filtered[1].label, "medium");
    }

    #[test]
    fn test_calculate_match_distance() {
        let match1 = PatternMatch {
            label: "test1".to_string(),
            confidence: 0.9,
            position: (0, 0),
            size: (10, 10),
        };

        let match2 = PatternMatch {
            label: "test2".to_string(),
            confidence: 0.8,
            position: (0, 10),
            size: (10, 10),
        };

        let distance = calculate_match_distance(&match1, &match2);
        assert_eq!(distance, 10.0); // Centers are (5,5) and (5,15), distance = 10
    }

    #[test]
    fn test_merge_nearby_matches() {
        let matches = vec![
            PatternMatch {
                label: "close1".to_string(),
                confidence: 0.9,
                position: (0, 0),
                size: (10, 10),
            },
            PatternMatch {
                label: "close2".to_string(),
                confidence: 0.8,
                position: (0, 5),
                size: (10, 10),
            },
            PatternMatch {
                label: "far".to_string(),
                confidence: 0.7,
                position: (50, 50),
                size: (10, 10),
            },
        ];

        let merged = merge_nearby_matches(matches, 10.0);
        assert_eq!(merged.len(), 2); // Two groups: one merged, one separate
    }

    #[test]
    fn test_create_merged_match() {
        let matches = vec![
            PatternMatch {
                label: "test1".to_string(),
                confidence: 0.9,
                position: (0, 0),
                size: (10, 10),
            },
            PatternMatch {
                label: "test2".to_string(),
                confidence: 0.7,
                position: (5, 5),
                size: (10, 10),
            },
        ];

        let cluster = vec![0, 1];
        let merged = create_merged_match(&matches, &cluster);

        assert_eq!(merged.label, "test1"); // Higher confidence label
        assert_eq!(merged.confidence, 0.9); // Higher confidence
        assert_eq!(merged.position, (0, 0)); // Bounding box top-left
        assert_eq!(merged.size, (15, 15)); // Bounding box size
    }

    #[test]
    #[should_panic]
    fn test_create_merged_match_empty_cluster() {
        let matches = vec![];
        let cluster = vec![];
        create_merged_match(&matches, &cluster);
    }
}
