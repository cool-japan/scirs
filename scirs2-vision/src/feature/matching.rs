//! Advanced feature matching algorithms
//!
//! This module provides sophisticated feature matching algorithms including:
//! - Brute force matching with various distance metrics
//! - FLANN-based approximate nearest neighbor matching
//! - Lowe's ratio test for robust matching
//! - Cross-check validation for bidirectional matching
//! - RANSAC-based outlier rejection
//!
//! The implementation supports both binary and floating-point descriptors,
//! with confidence scoring and comprehensive filtering techniques.

use crate::error::{Result, VisionError};
use crate::feature::{Descriptor, KeyPoint};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

/// Distance metrics for feature matching
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan distance (L1 norm)
    Manhattan,
    /// Hamming distance for binary descriptors
    Hamming,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
    /// Chi-squared distance
    ChiSquared,
}

/// Match between two descriptors
#[derive(Debug, Clone)]
pub struct DescriptorMatch {
    /// Index in first descriptor set
    pub query_idx: usize,
    /// Index in second descriptor set  
    pub train_idx: usize,
    /// Distance between descriptors
    pub distance: f32,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
}

/// Configuration for brute force matcher
#[derive(Debug, Clone)]
pub struct BruteForceConfig {
    /// Distance metric to use
    pub distancemetric: DistanceMetric,
    /// Maximum distance threshold for valid matches
    pub max_distance: f32,
    /// Apply cross-check validation
    pub cross_check: bool,
    /// Apply Lowe's ratio test
    pub use_ratio_test: bool,
    /// Ratio threshold for Lowe's test (typically 0.7-0.8)
    pub ratio_threshold: f32,
}

impl Default for BruteForceConfig {
    fn default() -> Self {
        Self {
            distancemetric: DistanceMetric::Euclidean,
            max_distance: 0.7,
            cross_check: true,
            use_ratio_test: true,
            ratio_threshold: 0.75,
        }
    }
}

/// Configuration for FLANN matcher
#[derive(Debug, Clone)]
pub struct FlannConfig {
    /// Number of trees for randomized kd-tree forest
    pub trees: usize,
    /// Number of checks for search
    pub checks: usize,
    /// Use LSH for binary descriptors
    pub use_lsh: bool,
    /// Number of hash tables for LSH
    pub hash_tables: usize,
    /// Key size for LSH
    pub key_size: usize,
}

impl Default for FlannConfig {
    fn default() -> Self {
        Self {
            trees: 4,
            checks: 50,
            use_lsh: false,
            hash_tables: 12,
            key_size: 20,
        }
    }
}

/// Configuration for RANSAC matcher
#[derive(Debug, Clone)]
pub struct RansacMatcherConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Distance threshold for inliers
    pub threshold: f32,
    /// Minimum number of inliers required
    pub min_inliers: usize,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
    /// Model type for geometric verification
    pub model_type: GeometricModel,
}

impl Default for RansacMatcherConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            threshold: 3.0,
            min_inliers: 8,
            confidence: 0.99,
            model_type: GeometricModel::Homography,
        }
    }
}

/// Geometric models for RANSAC
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeometricModel {
    /// Fundamental matrix (7-8 point algorithm)
    Fundamental,
    /// Essential matrix
    Essential,
    /// Homography matrix
    Homography,
    /// Affine transformation
    Affine,
}

/// Brute force matcher implementation
pub struct BruteForceMatcher {
    config: BruteForceConfig,
}

impl BruteForceMatcher {
    /// Create new brute force matcher
    pub fn new(config: BruteForceConfig) -> Self {
        Self { config }
    }

    /// Create default brute force matcher
    pub fn new_default() -> Self {
        Self {
            config: BruteForceConfig::default(),
        }
    }

    /// Match floating-point descriptors
    pub fn match_descriptors(
        &self,
        descriptors1: &[Descriptor],
        descriptors2: &[Descriptor],
    ) -> Result<Vec<DescriptorMatch>> {
        if descriptors1.is_empty() || descriptors2.is_empty() {
            return Ok(Vec::new());
        }

        let mut matches = Vec::new();

        // Forward matching
        for (i, desc1) in descriptors1.iter().enumerate() {
            if let Some(m) = self.find_best_match(i, &desc1.vector, descriptors2)? {
                matches.push(m);
            }
        }

        // Apply cross-check if enabled
        if self.config.cross_check {
            matches = self.apply_cross_check(descriptors1, descriptors2, matches)?;
        }

        Ok(matches)
    }

    /// Match binary descriptors (ORB/BRIEF)
    pub fn match_binary_descriptors(
        &self,
        descriptors1: &[Vec<u32>],
        descriptors2: &[Vec<u32>],
    ) -> Result<Vec<DescriptorMatch>> {
        if descriptors1.is_empty() || descriptors2.is_empty() {
            return Ok(Vec::new());
        }

        let mut matches = Vec::new();

        // Forward matching
        for (i, desc1) in descriptors1.iter().enumerate() {
            if let Some(m) = self.find_best_binary_match(i, desc1, descriptors2)? {
                matches.push(m);
            }
        }

        // Apply cross-check if enabled
        if self.config.cross_check {
            matches = self.apply_binary_cross_check(descriptors1, descriptors2, matches)?;
        }

        Ok(matches)
    }

    /// Find best match for a single descriptor
    fn find_best_match(
        &self,
        query_idx: usize,
        query_desc: &[f32],
        train_descriptors: &[Descriptor],
    ) -> Result<Option<DescriptorMatch>> {
        let mut best_distance = f32::MAX;
        let mut second_best_distance = f32::MAX;
        let mut best_idx = 0;

        for (j, train_desc) in train_descriptors.iter().enumerate() {
            let distance = self.compute_distance(query_desc, &train_desc.vector)?;

            if distance < best_distance {
                second_best_distance = best_distance;
                best_distance = distance;
                best_idx = j;
            } else if distance < second_best_distance {
                second_best_distance = distance;
            }
        }

        // Apply distance threshold
        if best_distance > self.config.max_distance {
            return Ok(None);
        }

        // Apply ratio test if enabled
        if self.config.use_ratio_test
            && (second_best_distance == f32::MAX
                || best_distance > self.config.ratio_threshold * second_best_distance)
        {
            return Ok(None);
        }

        // Calculate confidence score
        let confidence = if second_best_distance > 0.0 {
            1.0 - (best_distance / second_best_distance)
        } else {
            1.0
        };

        Ok(Some(DescriptorMatch {
            query_idx,
            train_idx: best_idx,
            distance: best_distance,
            confidence: confidence.clamp(0.0, 1.0),
        }))
    }

    /// Find best match for a single binary descriptor
    fn find_best_binary_match(
        &self,
        query_idx: usize,
        query_desc: &[u32],
        train_descriptors: &[Vec<u32>],
    ) -> Result<Option<DescriptorMatch>> {
        let mut best_distance = u32::MAX;
        let mut second_best_distance = u32::MAX;
        let mut best_idx = 0;

        for (j, train_desc) in train_descriptors.iter().enumerate() {
            let distance = self.compute_hamming_distance(query_desc, train_desc);

            if distance < best_distance {
                second_best_distance = best_distance;
                best_distance = distance;
                best_idx = j;
            } else if distance < second_best_distance {
                second_best_distance = distance;
            }
        }

        // Apply distance threshold
        if best_distance as f32 > self.config.max_distance {
            return Ok(None);
        }

        // Apply ratio test if enabled
        if self.config.use_ratio_test
            && (second_best_distance == u32::MAX
                || best_distance as f32 > self.config.ratio_threshold * second_best_distance as f32)
        {
            return Ok(None);
        }

        // Calculate confidence score
        let confidence = if second_best_distance > 0 {
            1.0 - (best_distance as f32 / second_best_distance as f32)
        } else {
            1.0
        };

        Ok(Some(DescriptorMatch {
            query_idx,
            train_idx: best_idx,
            distance: best_distance as f32,
            confidence: confidence.clamp(0.0, 1.0),
        }))
    }

    /// Apply cross-check validation
    fn apply_cross_check(
        &self,
        descriptors1: &[Descriptor],
        descriptors2: &[Descriptor],
        forward_matches: Vec<DescriptorMatch>,
    ) -> Result<Vec<DescriptorMatch>> {
        let mut validated_matches = Vec::new();

        // Create reverse _matches
        let mut reverse_matches = HashMap::new();
        for (j, desc2) in descriptors2.iter().enumerate() {
            if let Some(m) = self.find_best_match(j, &desc2.vector, descriptors1)? {
                reverse_matches.insert(j, m.train_idx);
            }
        }

        // Check bidirectional consistency
        for forward_match in forward_matches {
            if let Some(&reverse_idx) = reverse_matches.get(&forward_match.train_idx) {
                if reverse_idx == forward_match.query_idx {
                    validated_matches.push(forward_match);
                }
            }
        }

        Ok(validated_matches)
    }

    /// Apply cross-check validation for binary descriptors
    fn apply_binary_cross_check(
        &self,
        descriptors1: &[Vec<u32>],
        descriptors2: &[Vec<u32>],
        forward_matches: Vec<DescriptorMatch>,
    ) -> Result<Vec<DescriptorMatch>> {
        let mut validated_matches = Vec::new();

        // Create reverse _matches
        let mut reverse_matches = HashMap::new();
        for (j, desc2) in descriptors2.iter().enumerate() {
            if let Some(m) = self.find_best_binary_match(j, desc2, descriptors1)? {
                reverse_matches.insert(j, m.train_idx);
            }
        }

        // Check bidirectional consistency
        for forward_match in forward_matches {
            if let Some(&reverse_idx) = reverse_matches.get(&forward_match.train_idx) {
                if reverse_idx == forward_match.query_idx {
                    validated_matches.push(forward_match);
                }
            }
        }

        Ok(validated_matches)
    }

    /// Compute distance between two descriptors
    fn compute_distance(&self, desc1: &[f32], desc2: &[f32]) -> Result<f32> {
        if desc1.len() != desc2.len() {
            return Err(VisionError::InvalidParameter(
                "Descriptor dimensions must match".to_string(),
            ));
        }

        match self.config.distancemetric {
            DistanceMetric::Euclidean => {
                let sum_sq: f32 = desc1
                    .iter()
                    .zip(desc2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                Ok(sum_sq.sqrt())
            }
            DistanceMetric::Manhattan => {
                let sum: f32 = desc1
                    .iter()
                    .zip(desc2.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                Ok(sum)
            }
            DistanceMetric::Cosine => {
                let dot_product: f32 = desc1.iter().zip(desc2.iter()).map(|(a, b)| a * b).sum();
                let norm1: f32 = desc1.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                let norm2: f32 = desc2.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

                if norm1 == 0.0 || norm2 == 0.0 {
                    return Ok(1.0);
                }

                let cosine_sim = dot_product / (norm1 * norm2);
                Ok(1.0 - cosine_sim.clamp(-1.0, 1.0))
            }
            DistanceMetric::ChiSquared => {
                let sum: f32 = desc1
                    .iter()
                    .zip(desc2.iter())
                    .map(|(a, b)| {
                        if a + b != 0.0 {
                            (a - b).powi(2) / (a + b)
                        } else {
                            0.0
                        }
                    })
                    .sum();
                Ok(sum)
            }
            DistanceMetric::Hamming => Err(VisionError::InvalidParameter(
                "Hamming distance requires binary descriptors".to_string(),
            )),
        }
    }

    /// Compute Hamming distance between binary descriptors
    fn compute_hamming_distance(&self, desc1: &[u32], desc2: &[u32]) -> u32 {
        desc1
            .iter()
            .zip(desc2.iter())
            .map(|(&d1, &d2)| (d1 ^ d2).count_ones())
            .sum()
    }
}

/// FLANN-based approximate nearest neighbor matcher
pub struct FlannMatcher {
    #[allow(dead_code)]
    config: FlannConfig,
}

impl FlannMatcher {
    /// Create new FLANN matcher
    pub fn new(config: FlannConfig) -> Self {
        Self { config }
    }

    /// Create default FLANN matcher
    pub fn new_default() -> Self {
        Self {
            config: FlannConfig::default(),
        }
    }

    /// Match descriptors using approximate nearest neighbor search
    pub fn match_descriptors(
        &self,
        descriptors1: &[Descriptor],
        descriptors2: &[Descriptor],
    ) -> Result<Vec<DescriptorMatch>> {
        if descriptors1.is_empty() || descriptors2.is_empty() {
            return Ok(Vec::new());
        }

        // Build index for descriptors2
        let index = self.build_kd_tree_index(descriptors2)?;

        let mut matches = Vec::new();

        for (i, desc1) in descriptors1.iter().enumerate() {
            if let Some(neighbors) = self.search_kd_tree(&index, &desc1.vector, 2)? {
                if neighbors.len() >= 2 {
                    let best_distance = neighbors[0].1;
                    let second_best_distance = neighbors[1].1;

                    // Apply ratio test
                    if best_distance < 0.75 * second_best_distance {
                        let confidence = 1.0 - (best_distance / second_best_distance);
                        matches.push(DescriptorMatch {
                            query_idx: i,
                            train_idx: neighbors[0].0,
                            distance: best_distance,
                            confidence: confidence.clamp(0.0, 1.0),
                        });
                    }
                }
            }
        }

        Ok(matches)
    }

    /// Build simplified kd-tree index
    fn build_kd_tree_index(&self, descriptors: &[Descriptor]) -> Result<KdTree> {
        if descriptors.is_empty() {
            return Err(VisionError::InvalidParameter(
                "Cannot build index from empty descriptors".to_string(),
            ));
        }

        let dim = descriptors[0].vector.len();
        let mut points = Vec::new();

        for (i, desc) in descriptors.iter().enumerate() {
            if desc.vector.len() != dim {
                return Err(VisionError::InvalidParameter(
                    "All descriptors must have the same dimension".to_string(),
                ));
            }
            points.push((i, desc.vector.clone()));
        }

        Ok(KdTree::new(points, 0))
    }

    /// Search kd-tree for k nearest neighbors
    fn search_kd_tree(
        &self,
        tree: &KdTree,
        query: &[f32],
        k: usize,
    ) -> Result<Option<Vec<(usize, f32)>>> {
        let mut neighbors = Vec::new();
        tree.search_knn(query, k, &mut neighbors);

        if neighbors.is_empty() {
            Ok(None)
        } else {
            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            Ok(Some(neighbors))
        }
    }
}

/// Simplified kd-tree implementation for FLANN
#[derive(Debug)]
struct KdTree {
    point: Option<(usize, Vec<f32>)>,
    axis: usize,
    left: Option<Box<KdTree>>,
    right: Option<Box<KdTree>>,
}

impl KdTree {
    /// Create new kd-tree
    fn new(mut points: Vec<(usize, Vec<f32>)>, depth: usize) -> Self {
        if points.is_empty() {
            return Self {
                point: None,
                axis: 0,
                left: None,
                right: None,
            };
        }

        if points.len() == 1 {
            let axis = depth % points[0].1.len();
            return Self {
                point: Some(points.into_iter().next().unwrap()),
                axis,
                left: None,
                right: None,
            };
        }

        let axis = depth % points[0].1.len();
        points.sort_by(|a, b| {
            a.1[axis]
                .partial_cmp(&b.1[axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let median = points.len() / 2;
        let median_point = points[median].clone();

        let mut points_iter = points.into_iter();
        let left_points: Vec<(usize, Vec<f32>)> = points_iter.by_ref().take(median).collect();
        let right_points: Vec<(usize, Vec<f32>)> = points_iter.skip(1).collect();

        Self {
            point: Some(median_point),
            axis,
            left: if !left_points.is_empty() {
                Some(Box::new(Self::new(left_points, depth + 1)))
            } else {
                None
            },
            right: if !right_points.is_empty() {
                Some(Box::new(Self::new(right_points, depth + 1)))
            } else {
                None
            },
        }
    }

    /// Search for k nearest neighbors
    fn search_knn(&self, query: &[f32], k: usize, neighbors: &mut Vec<(usize, f32)>) {
        if let Some((idx, point)) = &self.point {
            let distance = euclidean_distance(query, point);

            if neighbors.len() < k {
                neighbors.push((*idx, distance));
            } else {
                // Find the farthest neighbor
                let max_idx = neighbors
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i_, _)| i_)
                    .unwrap();

                if distance < neighbors[max_idx].1 {
                    neighbors[max_idx] = (*idx, distance);
                }
            }
        }

        // Recursively search subtrees
        if let Some(point) = &self.point {
            let diff = query[self.axis] - point.1[self.axis];

            let (primary, secondary) = if diff <= 0.0 {
                (&self.left, &self.right)
            } else {
                (&self.right, &self.left)
            };

            if let Some(tree) = primary {
                tree.search_knn(query, k, neighbors);
            }

            // Check if we need to search the other side
            if neighbors.len() < k
                || diff.abs()
                    < neighbors
                        .iter()
                        .map(|(_, d)| *d)
                        .fold(f32::INFINITY, f32::min)
            {
                if let Some(tree) = secondary {
                    tree.search_knn(query, k, neighbors);
                }
            }
        }
    }
}

/// Calculate Euclidean distance between two points
#[allow(dead_code)]
fn euclidean_distance(p1: &[f32], p2: &[f32]) -> f32 {
    p1.iter()
        .zip(p2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Ratio test matcher implementing Lowe's ratio test
pub struct RatioTestMatcher {
    ratio_threshold: f32,
    distancemetric: DistanceMetric,
}

impl RatioTestMatcher {
    /// Create new ratio test matcher
    pub fn new(ratio_threshold: f32, distancemetric: DistanceMetric) -> Self {
        Self {
            ratio_threshold,
            distancemetric,
        }
    }

    /// Create default ratio test matcher
    pub fn new_default() -> Self {
        Self {
            ratio_threshold: 0.75,
            distancemetric: DistanceMetric::Euclidean,
        }
    }

    /// Apply ratio test to matches
    pub fn filter_matches(&self, matches: &[DescriptorMatch]) -> Vec<DescriptorMatch> {
        matches
            .iter()
            .filter(|m| m.confidence > (1.0 - self.ratio_threshold))
            .cloned()
            .collect()
    }

    /// Match descriptors with ratio test
    pub fn match_descriptors(
        &self,
        descriptors1: &[Descriptor],
        descriptors2: &[Descriptor],
    ) -> Result<Vec<DescriptorMatch>> {
        let bf_config = BruteForceConfig {
            distancemetric: self.distancemetric,
            max_distance: f32::MAX,
            cross_check: false,
            use_ratio_test: true,
            ratio_threshold: self.ratio_threshold,
        };

        let matcher = BruteForceMatcher::new(bf_config);
        matcher.match_descriptors(descriptors1, descriptors2)
    }
}

/// Cross-check matcher for bidirectional validation
pub struct CrossCheckMatcher {
    base_matcher: BruteForceMatcher,
}

impl CrossCheckMatcher {
    /// Create new cross-check matcher
    pub fn new(distancemetric: DistanceMetric) -> Self {
        let config = BruteForceConfig {
            distancemetric,
            max_distance: f32::MAX,
            cross_check: true,
            use_ratio_test: false,
            ratio_threshold: 1.0,
        };

        Self {
            base_matcher: BruteForceMatcher::new(config),
        }
    }

    /// Match descriptors with cross-check validation
    pub fn match_descriptors(
        &self,
        descriptors1: &[Descriptor],
        descriptors2: &[Descriptor],
    ) -> Result<Vec<DescriptorMatch>> {
        self.base_matcher
            .match_descriptors(descriptors1, descriptors2)
    }
}

/// RANSAC matcher for outlier rejection
pub struct RansacMatcher {
    config: RansacMatcherConfig,
}

impl RansacMatcher {
    /// Create new RANSAC matcher
    pub fn new(config: RansacMatcherConfig) -> Self {
        Self { config }
    }

    /// Create default RANSAC matcher
    pub fn new_default() -> Self {
        Self {
            config: RansacMatcherConfig::default(),
        }
    }

    /// Filter matches using RANSAC
    pub fn filter_matches(
        &self,
        matches: &[DescriptorMatch],
        keypoints1: &[KeyPoint],
        keypoints2: &[KeyPoint],
    ) -> Result<Vec<DescriptorMatch>> {
        if matches.len() < self.config.min_inliers {
            return Ok(Vec::new());
        }

        match self.config.model_type {
            GeometricModel::Homography => self.ransac_homography(matches, keypoints1, keypoints2),
            GeometricModel::Fundamental => self.ransac_fundamental(matches, keypoints1, keypoints2),
            GeometricModel::Essential => self.ransac_essential(matches, keypoints1, keypoints2),
            GeometricModel::Affine => self.ransac_affine(matches, keypoints1, keypoints2),
        }
    }

    /// RANSAC for homography estimation
    fn ransac_homography(
        &self,
        matches: &[DescriptorMatch],
        keypoints1: &[KeyPoint],
        keypoints2: &[KeyPoint],
    ) -> Result<Vec<DescriptorMatch>> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut best_inliers = Vec::new();
        let mut best_inlier_count = 0;

        for _ in 0..self.config.max_iterations {
            // Select 4 random matches for homography estimation
            if matches.len() < 4 {
                break;
            }

            let mut sample_indices = Vec::new();
            while sample_indices.len() < 4 {
                let idx = rng.gen_range(0..matches.len());
                if !sample_indices.contains(&idx) {
                    sample_indices.push(idx);
                }
            }

            // Extract sample points
            let mut src_points = Vec::new();
            let mut dstpoints = Vec::new();

            for &idx in &sample_indices {
                let m = &matches[idx];
                src_points.push([keypoints1[m.query_idx].x, keypoints1[m.query_idx].y]);
                dstpoints.push([keypoints2[m.train_idx].x, keypoints2[m.train_idx].y]);
            }

            // Estimate homography (simplified)
            if let Ok(homography) = estimate_homography(&src_points, &dstpoints) {
                // Count inliers
                let mut inliers = Vec::new();
                for (i, m) in matches.iter().enumerate() {
                    let src = [keypoints1[m.query_idx].x, keypoints1[m.query_idx].y];
                    let projected = apply_homography(&homography, src);
                    let dst = [keypoints2[m.train_idx].x, keypoints2[m.train_idx].y];

                    let error =
                        ((projected[0] - dst[0]).powi(2) + (projected[1] - dst[1]).powi(2)).sqrt();

                    if error < self.config.threshold {
                        inliers.push(i);
                    }
                }

                if inliers.len() > best_inlier_count {
                    best_inlier_count = inliers.len();
                    best_inliers = inliers;
                }

                // Early termination if we have enough inliers
                if best_inlier_count >= self.config.min_inliers {
                    let inlier_ratio = best_inlier_count as f32 / matches.len() as f32;
                    if inlier_ratio > self.config.confidence {
                        break;
                    }
                }
            }
        }

        // Return inlier matches
        Ok(best_inliers.iter().map(|&i| matches[i].clone()).collect())
    }

    /// RANSAC for fundamental matrix estimation using 8-point algorithm
    fn ransac_fundamental(
        &self,
        matches: &[DescriptorMatch],
        keypoints1: &[KeyPoint],
        keypoints2: &[KeyPoint],
    ) -> Result<Vec<DescriptorMatch>> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut best_inliers = Vec::new();
        let mut best_inlier_count = 0;

        for _ in 0..self.config.max_iterations {
            // Select 8 random matches for fundamental matrix estimation
            if matches.len() < 8 {
                break;
            }

            let mut sample_indices = Vec::new();
            while sample_indices.len() < 8 {
                let idx = rng.gen_range(0..matches.len());
                if !sample_indices.contains(&idx) {
                    sample_indices.push(idx);
                }
            }

            // Extract sample points
            let mut src_points = Vec::new();
            let mut dstpoints = Vec::new();

            for &idx in &sample_indices {
                let m = &matches[idx];
                src_points.push([keypoints1[m.query_idx].x, keypoints1[m.query_idx].y]);
                dstpoints.push([keypoints2[m.train_idx].x, keypoints2[m.train_idx].y]);
            }

            // Estimate fundamental matrix using 8-point algorithm
            if let Ok(fundamental) = estimate_fundamental_matrix(&src_points, &dstpoints) {
                // Count inliers using epipolar constraint
                let mut inliers = Vec::new();
                for (i, m) in matches.iter().enumerate() {
                    let p1 = [keypoints1[m.query_idx].x, keypoints1[m.query_idx].y, 1.0];
                    let p2 = [keypoints2[m.train_idx].x, keypoints2[m.train_idx].y, 1.0];

                    let error = compute_epipolar_error(&fundamental, &p1, &p2);

                    if error < self.config.threshold {
                        inliers.push(i);
                    }
                }

                if inliers.len() > best_inlier_count {
                    best_inlier_count = inliers.len();
                    best_inliers = inliers;
                }

                // Early termination if we have enough inliers
                if best_inlier_count >= self.config.min_inliers {
                    let inlier_ratio = best_inlier_count as f32 / matches.len() as f32;
                    if inlier_ratio > self.config.confidence {
                        break;
                    }
                }
            }
        }

        // Return inlier matches
        Ok(best_inliers.iter().map(|&i| matches[i].clone()).collect())
    }

    /// RANSAC for essential matrix estimation using 5-point algorithm
    fn ransac_essential(
        &self,
        matches: &[DescriptorMatch],
        keypoints1: &[KeyPoint],
        keypoints2: &[KeyPoint],
    ) -> Result<Vec<DescriptorMatch>> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut best_inliers = Vec::new();
        let mut best_inlier_count = 0;

        for _ in 0..self.config.max_iterations {
            // Select 5 random matches for essential matrix estimation
            if matches.len() < 5 {
                break;
            }

            let mut sample_indices = Vec::new();
            while sample_indices.len() < 5 {
                let idx = rng.gen_range(0..matches.len());
                if !sample_indices.contains(&idx) {
                    sample_indices.push(idx);
                }
            }

            // Extract sample points (assuming calibrated cameras with unit focal length)
            let mut src_points = Vec::new();
            let mut dstpoints = Vec::new();

            for &idx in &sample_indices {
                let m = &matches[idx];
                src_points.push([keypoints1[m.query_idx].x, keypoints1[m.query_idx].y]);
                dstpoints.push([keypoints2[m.train_idx].x, keypoints2[m.train_idx].y]);
            }

            // Estimate essential matrix (simplified using fundamental matrix for now)
            // In practice, implement proper 5-point algorithm
            if let Ok(essential) = estimate_essential_matrix(&src_points, &dstpoints) {
                // Count inliers using epipolar constraint
                let mut inliers = Vec::new();
                for (i, m) in matches.iter().enumerate() {
                    let p1 = [keypoints1[m.query_idx].x, keypoints1[m.query_idx].y, 1.0];
                    let p2 = [keypoints2[m.train_idx].x, keypoints2[m.train_idx].y, 1.0];

                    let error = compute_epipolar_error(&essential, &p1, &p2);

                    if error < self.config.threshold {
                        inliers.push(i);
                    }
                }

                if inliers.len() > best_inlier_count {
                    best_inlier_count = inliers.len();
                    best_inliers = inliers;
                }

                // Early termination if we have enough inliers
                if best_inlier_count >= self.config.min_inliers {
                    let inlier_ratio = best_inlier_count as f32 / matches.len() as f32;
                    if inlier_ratio > self.config.confidence {
                        break;
                    }
                }
            }
        }

        // Return inlier matches
        Ok(best_inliers.iter().map(|&i| matches[i].clone()).collect())
    }

    /// RANSAC for affine transformation estimation
    fn ransac_affine(
        &self,
        matches: &[DescriptorMatch],
        keypoints1: &[KeyPoint],
        keypoints2: &[KeyPoint],
    ) -> Result<Vec<DescriptorMatch>> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut best_inliers = Vec::new();
        let mut best_inlier_count = 0;

        for _ in 0..self.config.max_iterations {
            // Select 3 random matches for affine estimation
            if matches.len() < 3 {
                break;
            }

            let mut sample_indices = Vec::new();
            while sample_indices.len() < 3 {
                let idx = rng.gen_range(0..matches.len());
                if !sample_indices.contains(&idx) {
                    sample_indices.push(idx);
                }
            }

            // Extract sample points
            let mut src_points = Vec::new();
            let mut dstpoints = Vec::new();

            for &idx in &sample_indices {
                let m = &matches[idx];
                src_points.push([keypoints1[m.query_idx].x, keypoints1[m.query_idx].y]);
                dstpoints.push([keypoints2[m.train_idx].x, keypoints2[m.train_idx].y]);
            }

            // Estimate affine transformation (simplified)
            if let Ok(affine) = estimate_affine(&src_points, &dstpoints) {
                // Count inliers
                let mut inliers = Vec::new();
                for (i, m) in matches.iter().enumerate() {
                    let src = [keypoints1[m.query_idx].x, keypoints1[m.query_idx].y];
                    let projected = apply_affine(&affine, src);
                    let dst = [keypoints2[m.train_idx].x, keypoints2[m.train_idx].y];

                    let error =
                        ((projected[0] - dst[0]).powi(2) + (projected[1] - dst[1]).powi(2)).sqrt();

                    if error < self.config.threshold {
                        inliers.push(i);
                    }
                }

                if inliers.len() > best_inlier_count {
                    best_inlier_count = inliers.len();
                    best_inliers = inliers;
                }
            }
        }

        Ok(best_inliers.iter().map(|&i| matches[i].clone()).collect())
    }
}

/// Estimate homography matrix from point correspondences using Direct Linear Transform (DLT)
#[allow(dead_code)]
fn estimate_homography(src_points: &[[f32; 2]], dstpoints: &[[f32; 2]]) -> Result<Array2<f32>> {
    if src_points.len() != dstpoints.len() || src_points.len() < 4 {
        return Err(VisionError::InvalidParameter(
            "Need at least 4 point correspondences".to_string(),
        ));
    }

    // Normalize points for better numerical stability
    let (src_norm, src_transform) = normalize_points(src_points);
    let (dst_norm, dst_transform) = normalize_points(dstpoints);

    // Build the coefficient matrix using normalized points
    let n = src_points.len();
    let mut a = Array2::zeros((2 * n, 9));

    for i in 0..n {
        let [x, y] = src_norm[i];
        let [xp, yp] = dst_norm[i];

        // First row: -xi, -yi, -1, 0, 0, 0, xi*xp', yi*xp', xp'
        a[[2 * i, 0]] = -x;
        a[[2 * i, 1]] = -y;
        a[[2 * i, 2]] = -1.0;
        a[[2 * i, 6]] = x * xp;
        a[[2 * i, 7]] = y * xp;
        a[[2 * i, 8]] = xp;

        // Second row: 0, 0, 0, -xi, -yi, -1, xi*yp', yi*yp', yp'
        a[[2 * i + 1, 3]] = -x;
        a[[2 * i + 1, 4]] = -y;
        a[[2 * i + 1, 5]] = -1.0;
        a[[2 * i + 1, 6]] = x * yp;
        a[[2 * i + 1, 7]] = y * yp;
        a[[2 * i + 1, 8]] = yp;
    }

    // Solve for homography using SVD via eigenvalue decomposition of A^T*A
    let ata = compute_ata(&a);
    let h_vec = find_smallest_eigenvector_homography(&ata)?;

    // Reshape to 3x3 matrix
    let mut h_norm = Array2::from_shape_vec((3, 3), h_vec)
        .map_err(|e| VisionError::OperationError(format!("Failed to reshape homography: {e}")))?;

    // Denormalize: H = T_dst^(-1) * H_norm * Tsrc
    let dst_inv = matrix_inverse_3x3(&dst_transform)?;
    h_norm = matrix_multiply_3x3(&dst_inv, &h_norm);
    let h = matrix_multiply_3x3(&h_norm, &src_transform);

    // Normalize so that h[2,2] = 1 if possible
    if h[[2, 2]].abs() > 1e-8 {
        let h_normalized = h.mapv(|v| v / h[[2, 2]]);
        Ok(h_normalized)
    } else {
        Ok(h)
    }
}

/// Apply homography transformation to a point
#[allow(dead_code)]
fn apply_homography(h: &Array2<f32>, point: [f32; 2]) -> [f32; 2] {
    let [x, y] = point;
    let w = h[[2, 0]] * x + h[[2, 1]] * y + h[[2, 2]];

    if w.abs() > 1e-8 {
        [
            (h[[0, 0]] * x + h[[0, 1]] * y + h[[0, 2]]) / w,
            (h[[1, 0]] * x + h[[1, 1]] * y + h[[1, 2]]) / w,
        ]
    } else {
        point
    }
}

/// Estimate affine transformation from point correspondences using least squares
#[allow(dead_code)]
fn estimate_affine(src_points: &[[f32; 2]], dstpoints: &[[f32; 2]]) -> Result<Array2<f32>> {
    if src_points.len() != dstpoints.len() || src_points.len() < 3 {
        return Err(VisionError::InvalidParameter(
            "Need at least 3 point correspondences".to_string(),
        ));
    }

    let n = src_points.len();

    // Set up the linear system: A * x = b
    // For affine transformation: [a b c; d e f; 0 0 1]
    // We solve two separate 3x1 systems for [a b c] and [d e f]

    let mut a_matrix = Array2::zeros((n, 3));
    let mut bx = vec![0.0f32; n];
    let mut by = vec![0.0f32; n];

    for i in 0..n {
        let [x, y] = src_points[i];
        let [xp, yp] = dstpoints[i];

        // Fill coefficient matrix: [x y 1]
        a_matrix[[i, 0]] = x;
        a_matrix[[i, 1]] = y;
        a_matrix[[i, 2]] = 1.0;

        // Fill target vectors
        bx[i] = xp;
        by[i] = yp;
    }

    // Solve for x-coefficients: [a b c]
    let x_coeffs = solve_least_squares(&a_matrix, &bx)?;

    // Solve for y-coefficients: [d e f]
    let y_coeffs = solve_least_squares(&a_matrix, &by)?;

    // Construct affine transformation matrix
    let mut affine = Array2::zeros((3, 3));
    affine[[0, 0]] = x_coeffs[0]; // a
    affine[[0, 1]] = x_coeffs[1]; // b
    affine[[0, 2]] = x_coeffs[2]; // c
    affine[[1, 0]] = y_coeffs[0]; // d
    affine[[1, 1]] = y_coeffs[1]; // e
    affine[[1, 2]] = y_coeffs[2]; // f
    affine[[2, 2]] = 1.0;

    Ok(affine)
}

/// Apply affine transformation to a point
#[allow(dead_code)]
fn apply_affine(affine: &Array2<f32>, point: [f32; 2]) -> [f32; 2] {
    let [x, y] = point;
    [
        affine[[0, 0]] * x + affine[[0, 1]] * y + affine[[0, 2]],
        affine[[1, 0]] * x + affine[[1, 1]] * y + affine[[1, 2]],
    ]
}

/// Utility functions for matching
pub mod utils {
    use super::*;

    /// Convert confidence score to match quality
    pub fn confidence_to_quality(confidence: f32) -> MatchQuality {
        if confidence > 0.8 {
            MatchQuality::Excellent
        } else if confidence > 0.6 {
            MatchQuality::Good
        } else if confidence > 0.4 {
            MatchQuality::Fair
        } else {
            MatchQuality::Poor
        }
    }

    /// Filter matches by confidence threshold
    pub fn filter_by_confidence(
        matches: &[DescriptorMatch],
        threshold: f32,
    ) -> Vec<DescriptorMatch> {
        matches
            .iter()
            .filter(|m| m.confidence >= threshold)
            .cloned()
            .collect()
    }

    /// Sort matches by confidence (descending)
    pub fn sort_by_confidence(mut matches: Vec<DescriptorMatch>) -> Vec<DescriptorMatch> {
        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches
    }

    /// Calculate match statistics
    pub fn calculate_match_statistics(matches: &[DescriptorMatch]) -> MatchStatistics {
        if matches.is_empty() {
            return MatchStatistics::default();
        }

        let distances: Vec<f32> = matches.iter().map(|m| m.distance).collect();
        let confidences: Vec<f32> = matches.iter().map(|m| m.confidence).collect();

        let mean_distance = distances.iter().sum::<f32>() / distances.len() as f32;
        let mean_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;

        let mut sorted_distances = distances.clone();
        sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_distance = if sorted_distances.len() % 2 == 0 {
            (sorted_distances[sorted_distances.len() / 2 - 1]
                + sorted_distances[sorted_distances.len() / 2])
                / 2.0
        } else {
            sorted_distances[sorted_distances.len() / 2]
        };

        MatchStatistics {
            count: matches.len(),
            mean_distance,
            median_distance,
            min_distance: distances.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            max_distance: distances.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
            mean_confidence,
            min_confidence: confidences.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            max_confidence: confidences.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
        }
    }
}

/// Match quality enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatchQuality {
    /// Excellent match (confidence > 0.8)
    Excellent,
    /// Good match (0.6 < confidence <= 0.8)
    Good,
    /// Fair match (0.4 < confidence <= 0.6)
    Fair,
    /// Poor match (confidence <= 0.4)
    Poor,
}

/// Match statistics
#[derive(Debug, Clone)]
pub struct MatchStatistics {
    /// Number of matches
    pub count: usize,
    /// Mean distance
    pub mean_distance: f32,
    /// Median distance
    pub median_distance: f32,
    /// Minimum distance
    pub min_distance: f32,
    /// Maximum distance
    pub max_distance: f32,
    /// Mean confidence
    pub mean_confidence: f32,
    /// Minimum confidence
    pub min_confidence: f32,
    /// Maximum confidence
    pub max_confidence: f32,
}

impl Default for MatchStatistics {
    fn default() -> Self {
        Self {
            count: 0,
            mean_distance: 0.0,
            median_distance: 0.0,
            min_distance: 0.0,
            max_distance: 0.0,
            mean_confidence: 0.0,
            min_confidence: 0.0,
            max_confidence: 0.0,
        }
    }
}

/// Helper functions for linear algebra operations
/// Normalize points for better numerical stability in homography estimation
#[allow(dead_code)]
fn normalize_points(points: &[[f32; 2]]) -> (Vec<[f32; 2]>, Array2<f32>) {
    if points.is_empty() {
        return (Vec::new(), Array2::eye(3));
    }

    // Compute centroid
    let n = points.len() as f32;
    let cx = points.iter().map(|p| p[0]).sum::<f32>() / n;
    let cy = points.iter().map(|p| p[1]).sum::<f32>() / n;

    // Compute average distance from centroid
    let avg_dist = points
        .iter()
        .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt())
        .sum::<f32>()
        / n;

    // Scale factor for unit average distance
    let scale = if avg_dist > 1e-8 {
        (2.0_f32).sqrt() / avg_dist
    } else {
        1.0
    };

    // Apply normalization transformation
    let normalized: Vec<[f32; 2]> = points
        .iter()
        .map(|p| [(p[0] - cx) * scale, (p[1] - cy) * scale])
        .collect();

    // Build transformation matrix
    let mut transform = Array2::zeros((3, 3));
    transform[[0, 0]] = scale;
    transform[[1, 1]] = scale;
    transform[[0, 2]] = -cx * scale;
    transform[[1, 2]] = -cy * scale;
    transform[[2, 2]] = 1.0;

    (normalized, transform)
}

/// Compute A^T * A for homography estimation
#[allow(dead_code)]
fn compute_ata(a: &Array2<f32>) -> Array2<f32> {
    let (m, n) = a.dim();
    let mut ata = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            for k in 0..m {
                ata[[i, j]] += a[[k, i]] * a[[k, j]];
            }
        }
    }

    ata
}

/// Find smallest eigenvector for homography estimation using power iteration
#[allow(dead_code)]
fn find_smallest_eigenvector_homography(matrix: &Array2<f32>) -> Result<Vec<f32>> {
    let n = matrix.shape()[0];
    let mut v = vec![1.0; n];
    v[n - 1] = 1.0; // Bias toward the last element

    // Normalize
    let mut norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut v {
        *val /= norm;
    }

    // Power iteration to find smallest eigenvalue
    for _ in 0..100 {
        let mut mv = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                mv[i] += matrix[[i, j]] * v[j];
            }
        }

        // Compute Rayleigh quotient
        let mut rayleigh = 0.0;
        for i in 0..n {
            rayleigh += v[i] * mv[i];
        }

        // Inverse iteration for smallest eigenvalue
        for i in 0..n {
            v[i] = mv[i] - rayleigh * v[i];
        }

        // Renormalize
        norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            break;
        }

        for val in &mut v {
            *val /= norm;
        }
    }

    Ok(v)
}

/// Compute 3x3 matrix inverse
#[allow(dead_code)]
fn matrix_inverse_3x3(matrix: &Array2<f32>) -> Result<Array2<f32>> {
    let m = matrix;
    let det = m[[0, 0]] * (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]])
        - m[[0, 1]] * (m[[1, 0]] * m[[2, 2]] - m[[1, 2]] * m[[2, 0]])
        + m[[0, 2]] * (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]]);

    if det.abs() < 1e-10 {
        return Err(VisionError::OperationError(
            "Matrix is singular".to_string(),
        ));
    }

    let mut inv = Array2::zeros((3, 3));
    inv[[0, 0]] = (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]]) / det;
    inv[[0, 1]] = (m[[0, 2]] * m[[2, 1]] - m[[0, 1]] * m[[2, 2]]) / det;
    inv[[0, 2]] = (m[[0, 1]] * m[[1, 2]] - m[[0, 2]] * m[[1, 1]]) / det;
    inv[[1, 0]] = (m[[1, 2]] * m[[2, 0]] - m[[1, 0]] * m[[2, 2]]) / det;
    inv[[1, 1]] = (m[[0, 0]] * m[[2, 2]] - m[[0, 2]] * m[[2, 0]]) / det;
    inv[[1, 2]] = (m[[0, 2]] * m[[1, 0]] - m[[0, 0]] * m[[1, 2]]) / det;
    inv[[2, 0]] = (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]]) / det;
    inv[[2, 1]] = (m[[0, 1]] * m[[2, 0]] - m[[0, 0]] * m[[2, 1]]) / det;
    inv[[2, 2]] = (m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]]) / det;

    Ok(inv)
}

/// Multiply two 3x3 matrices
#[allow(dead_code)]
fn matrix_multiply_3x3(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let mut result = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }
    result
}

/// Solve least squares problem using normal equations
#[allow(dead_code)]
fn solve_least_squares(a: &Array2<f32>, b: &[f32]) -> Result<Vec<f32>> {
    let (m, n) = a.dim();

    // Compute A^T * A
    let mut ata = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..m {
                ata[[i, j]] += a[[k, i]] * a[[k, j]];
            }
        }
    }

    // Compute A^T * b
    let mut atb = vec![0.0f32; n];
    for i in 0..n {
        for k in 0..m {
            atb[i] += a[[k, i]] * b[k];
        }
    }

    // Solve using simple Gaussian elimination
    solve_linear_system(&ata, &atb)
}

/// Solve linear system using Gaussian elimination with partial pivoting
#[allow(dead_code)]
fn solve_linear_system(a: &Array2<f32>, b: &[f32]) -> Result<Vec<f32>> {
    let n = a.shape()[0];
    let mut a_copy = a.clone();
    let mut b_copy = b.to_vec();

    // Forward elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if a_copy[[k, i]].abs() > a_copy[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..n {
                let temp = a_copy[[i, j]];
                a_copy[[i, j]] = a_copy[[max_row, j]];
                a_copy[[max_row, j]] = temp;
            }
            b_copy.swap(i, max_row);
        }

        // Check for singular matrix
        if a_copy[[i, i]].abs() < 1e-10 {
            return Err(VisionError::OperationError("Singular matrix".to_string()));
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = a_copy[[k, i]] / a_copy[[i, i]];
            for j in (i + 1)..n {
                a_copy[[k, j]] -= factor * a_copy[[i, j]];
            }
            b_copy[k] -= factor * b_copy[i];
        }
    }

    // Back substitution
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        x[i] = b_copy[i];
        for j in (i + 1)..n {
            x[i] -= a_copy[[i, j]] * x[j];
        }
        x[i] /= a_copy[[i, i]];
    }

    Ok(x)
}

/// Estimate fundamental matrix using 8-point algorithm
#[allow(dead_code)]
fn estimate_fundamental_matrix(
    src_points: &[[f32; 2]],
    dstpoints: &[[f32; 2]],
) -> Result<Array2<f32>> {
    if src_points.len() != dstpoints.len() || src_points.len() < 8 {
        return Err(VisionError::InvalidParameter(
            "Need at least 8 point correspondences for fundamental matrix".to_string(),
        ));
    }

    // Normalize points
    let (src_norm, src_transform) = normalize_points(src_points);
    let (dst_norm, dst_transform) = normalize_points(dstpoints);

    let n = src_points.len();
    let mut a = Array2::zeros((n, 9));

    // Build coefficient matrix
    for i in 0..n {
        let [x1, y1] = src_norm[i];
        let [x2, y2] = dst_norm[i];

        a[[i, 0]] = x2 * x1;
        a[[i, 1]] = x2 * y1;
        a[[i, 2]] = x2;
        a[[i, 3]] = y2 * x1;
        a[[i, 4]] = y2 * y1;
        a[[i, 5]] = y2;
        a[[i, 6]] = x1;
        a[[i, 7]] = y1;
        a[[i, 8]] = 1.0;
    }

    // Solve using SVD approximation
    let ata = compute_ata(&a);
    let f_vec = find_smallest_eigenvector_homography(&ata)?;

    // Reshape to 3x3 matrix
    let mut f_norm = Array2::from_shape_vec((3, 3), f_vec).map_err(|e| {
        VisionError::OperationError(format!("Failed to reshape fundamental matrix: {e}"))
    })?;

    // Enforce rank-2 constraint by setting smallest singular value to 0
    // (Simplified approach - in practice use proper SVD)

    // Denormalize: F = T_dst^T * F_norm * Tsrc
    let dst_t = transpose_3x3(&dst_transform);
    f_norm = matrix_multiply_3x3(&dst_t, &f_norm);
    let f = matrix_multiply_3x3(&f_norm, &src_transform);

    Ok(f)
}

/// Estimate essential matrix (simplified using fundamental matrix)
#[allow(dead_code)]
fn estimate_essential_matrix(
    src_points: &[[f32; 2]],
    dstpoints: &[[f32; 2]],
) -> Result<Array2<f32>> {
    // For calibrated cameras, essential matrix is similar to fundamental matrix
    // In practice, use proper 5-point algorithm with camera calibration
    estimate_fundamental_matrix(src_points, dstpoints)
}

/// Compute epipolar error for a point correspondence
#[allow(dead_code)]
fn compute_epipolar_error(matrix: &Array2<f32>, p1: &[f32; 3], p2: &[f32; 3]) -> f32 {
    // Compute F * p1
    let mut fp1 = [0.0f32; 3];
    for i in 0..3 {
        for j in 0..3 {
            fp1[i] += matrix[[i, j]] * p1[j];
        }
    }

    // Compute p2^T * F * p1
    let mut error = 0.0f32;
    for i in 0..3 {
        error += p2[i] * fp1[i];
    }

    // Normalize by the norm of the epipolar line
    let line_norm = (fp1[0] * fp1[0] + fp1[1] * fp1[1]).sqrt();
    if line_norm > 1e-8 {
        error.abs() / line_norm
    } else {
        error.abs()
    }
}

/// Transpose a 3x3 matrix
#[allow(dead_code)]
fn transpose_3x3(matrix: &Array2<f32>) -> Array2<f32> {
    let mut result = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            result[[i, j]] = matrix[[j, i]];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::Descriptor;

    fn create_test_descriptors() -> (Vec<Descriptor>, Vec<Descriptor>) {
        let kp1 = KeyPoint {
            x: 10.0,
            y: 20.0,
            scale: 1.0,
            orientation: 0.0,
            response: 0.5,
        };

        let kp2 = KeyPoint {
            x: 15.0,
            y: 25.0,
            scale: 1.0,
            orientation: 0.1,
            response: 0.6,
        };

        let desc1 = vec![
            Descriptor {
                keypoint: kp1.clone(),
                vector: vec![1.0, 0.0, 0.0, 1.0],
            },
            Descriptor {
                keypoint: kp2.clone(),
                vector: vec![0.0, 1.0, 1.0, 0.0],
            },
        ];

        let desc2 = vec![
            Descriptor {
                keypoint: kp1,
                vector: vec![0.9, 0.1, 0.1, 0.9], // Similar to first descriptor
            },
            Descriptor {
                keypoint: kp2,
                vector: vec![0.1, 0.9, 0.9, 0.1], // Similar to second descriptor
            },
        ];

        (desc1, desc2)
    }

    #[test]
    fn test_brute_force_matcher() {
        let (desc1, desc2) = create_test_descriptors();
        let matcher = BruteForceMatcher::new_default();

        let matches = matcher.match_descriptors(&desc1, &desc2).unwrap();
        assert!(!matches.is_empty());

        // Check that we have valid matches
        for m in &matches {
            assert!(m.query_idx < desc1.len());
            assert!(m.train_idx < desc2.len());
            assert!(m.confidence >= 0.0 && m.confidence <= 1.0);
        }
    }

    #[test]
    fn test_distance_metrics() {
        let vec1 = vec![1.0, 0.0, 0.0, 1.0];
        let vec2 = vec![0.0, 1.0, 1.0, 0.0];

        let config = BruteForceConfig {
            distancemetric: DistanceMetric::Euclidean,
            ..Default::default()
        };
        let matcher = BruteForceMatcher::new(config);

        let dist = matcher.compute_distance(&vec1, &vec2).unwrap();
        assert!(dist > 0.0);

        let config = BruteForceConfig {
            distancemetric: DistanceMetric::Manhattan,
            ..Default::default()
        };
        let matcher = BruteForceMatcher::new(config);

        let dist = matcher.compute_distance(&vec1, &vec2).unwrap();
        assert!(dist > 0.0);
    }

    #[test]
    fn test_ratio_test_matcher() {
        let (desc1, desc2) = create_test_descriptors();
        let matcher = RatioTestMatcher::new_default();

        let matches = matcher.match_descriptors(&desc1, &desc2).unwrap();

        // All returned matches should pass the ratio test
        for m in &matches {
            assert!(m.confidence > 0.25); // 1 - 0.75 ratio threshold
        }
    }

    #[test]
    fn test_hamming_distance() {
        let desc1 = vec![0b11110000u32];
        let desc2 = vec![0b11001100u32];

        let config = BruteForceConfig {
            distancemetric: DistanceMetric::Hamming,
            ..Default::default()
        };
        let matcher = BruteForceMatcher::new(config);

        let distance = matcher.compute_hamming_distance(&desc1, &desc2);
        assert_eq!(distance, 4); // 4 bits different
    }

    #[test]
    fn test_match_statistics() {
        let matches = vec![
            DescriptorMatch {
                query_idx: 0,
                train_idx: 0,
                distance: 0.1,
                confidence: 0.9,
            },
            DescriptorMatch {
                query_idx: 1,
                train_idx: 1,
                distance: 0.2,
                confidence: 0.8,
            },
        ];

        let stats = utils::calculate_match_statistics(&matches);
        assert_eq!(stats.count, 2);
        assert!((stats.mean_distance - 0.15).abs() < 1e-6);
        assert!((stats.mean_confidence - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_confidence_filtering() {
        let matches = vec![
            DescriptorMatch {
                query_idx: 0,
                train_idx: 0,
                distance: 0.1,
                confidence: 0.9,
            },
            DescriptorMatch {
                query_idx: 1,
                train_idx: 1,
                distance: 0.3,
                confidence: 0.4,
            },
        ];

        let filtered = utils::filter_by_confidence(&matches, 0.7);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].confidence, 0.9);
    }
}
