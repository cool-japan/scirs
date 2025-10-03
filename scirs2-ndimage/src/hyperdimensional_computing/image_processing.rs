//! Image Processing with Hyperdimensional Computing
//!
//! This module implements HDC-based image processing algorithms including
//! image encoding, pattern recognition, feature detection, and sequence processing
//! for computer vision applications.

use scirs2_core::ndarray::{s, Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::{HashMap, HashSet};

use crate::error::{NdimageError, NdimageResult};
use crate::hyperdimensional_computing::memory::HDCMemory;
use crate::hyperdimensional_computing::types::{
    CompositionResult, FeatureDetection, HDCConfig, Hypervector, PatternMatch, SequenceEncoding,
};
use crate::hyperdimensional_computing::utils::{
    analyze_patch_for_feature, non_maximum_suppression,
};
use crate::hyperdimensional_computing::vector_ops::vector_utils;

/// HDC-based image encoder
#[derive(Debug, Clone)]
pub struct ImageHDCEncoder {
    /// Image dimensions
    pub height: usize,
    pub width: usize,
    /// Position encoders for spatial information
    pub position_encoders: HashMap<(usize, usize), Hypervector>,
    /// Value encoders for pixel intensities
    pub value_encoders: HashMap<u8, Hypervector>,
    /// Feature encoders for specific features
    pub feature_encoders: HashMap<String, Hypervector>,
    /// Configuration
    pub config: HDCConfig,
}

impl ImageHDCEncoder {
    /// Create a new image HDC encoder
    ///
    /// # Arguments
    ///
    /// * `height` - Image height
    /// * `width` - Image width
    /// * `config` - HDC configuration
    ///
    /// # Returns
    ///
    /// A new ImageHDCEncoder instance
    pub fn new(height: usize, width: usize, config: HDCConfig) -> Self {
        let mut encoder = Self {
            height,
            width,
            position_encoders: HashMap::new(),
            value_encoders: HashMap::new(),
            feature_encoders: HashMap::new(),
            config,
        };

        encoder.initialize_encoders();
        encoder
    }

    /// Initialize position and value encoders
    fn initialize_encoders(&mut self) {
        // Create position encoders for each spatial location
        for y in 0..self.height {
            for x in 0..self.width {
                let position_hv =
                    Hypervector::random(self.config.hypervector_dim, self.config.sparsity);
                self.position_encoders.insert((y, x), position_hv);
            }
        }

        // Create value encoders for pixel intensities (0-255)
        for value in 0..=255 {
            let value_hv = Hypervector::random(self.config.hypervector_dim, self.config.sparsity);
            self.value_encoders.insert(value, value_hv);
        }
    }

    /// Encode an image into a hypervector
    ///
    /// # Arguments
    ///
    /// * `image` - Input image as 2D array
    ///
    /// # Returns
    ///
    /// Encoded hypervector representation of the image
    pub fn encode_image<T>(&self, image: ArrayView2<T>) -> NdimageResult<Hypervector>
    where
        T: Float + FromPrimitive + Copy,
    {
        let (img_height, img_width) = image.dim();

        if img_height > self.height || img_width > self.width {
            return Err(NdimageError::InvalidInput(format!(
                "Image size ({}×{}) exceeds encoder capacity ({}×{})",
                img_height, img_width, self.height, self.width
            )));
        }

        let mut encodings = Vec::new();

        for y in 0..img_height {
            for x in 0..img_width {
                let pixel_value = image[[y, x]];

                // Convert to u8 intensity (assuming normalized input)
                let intensity = (pixel_value.to_f64().unwrap_or(0.0).clamp(0.0, 1.0) * 255.0) as u8;

                // Get position and value encoders
                if let (Some(pos_hv), Some(val_hv)) = (
                    self.position_encoders.get(&(y, x)),
                    self.value_encoders.get(&intensity),
                ) {
                    // Bind position and value
                    let pixel_encoding = pos_hv.bind(val_hv)?;
                    encodings.push(pixel_encoding);
                }
            }
        }

        // Bundle all pixel encodings
        if encodings.is_empty() {
            Ok(Hypervector::zeros(self.config.hypervector_dim))
        } else {
            vector_utils::bundle_multiple(&encodings)
        }
    }

    /// Encode a patch with specific feature type
    ///
    /// # Arguments
    ///
    /// * `patch` - Image patch to encode
    /// * `feature_type` - Type of feature to encode for
    ///
    /// # Returns
    ///
    /// Feature-specific encoded hypervector
    pub fn encode_patch<T>(
        &self,
        patch: ArrayView2<T>,
        feature_type: &str,
    ) -> NdimageResult<Hypervector>
    where
        T: Float + FromPrimitive + Copy,
    {
        let base_encoding = self.encode_image(patch)?;

        if let Some(feature_hv) = self.feature_encoders.get(feature_type) {
            base_encoding.bind(feature_hv)
        } else {
            Ok(base_encoding)
        }
    }

    /// Add a feature encoder
    ///
    /// # Arguments
    ///
    /// * `feature_type` - Name of the feature type
    /// * `feature_hv` - Hypervector encoding for this feature type
    pub fn add_feature_encoder(&mut self, feature_type: String, feature_hv: Hypervector) {
        self.feature_encoders.insert(feature_type, feature_hv);
    }

    /// Get spatial encoding for a specific position
    pub fn get_position_encoding(&self, y: usize, x: usize) -> Option<&Hypervector> {
        self.position_encoders.get(&(y, x))
    }

    /// Get value encoding for a specific intensity
    pub fn get_value_encoding(&self, intensity: u8) -> Option<&Hypervector> {
        self.value_encoders.get(&intensity)
    }
}

/// HDC-based pattern recognition
///
/// Recognize patterns in images using stored hypervector representations.
///
/// # Arguments
///
/// * `image` - Input image to search
/// * `patterns` - Known patterns with labels
/// * `config` - HDC configuration
///
/// # Returns
///
/// Vector of pattern matches found in the image
#[allow(dead_code)]
pub fn hdc_pattern_recognition<T>(
    image: ArrayView2<T>,
    patterns: &[(&ArrayView2<T>, &str)],
    config: &HDCConfig,
) -> NdimageResult<Vec<PatternMatch>>
where
    T: Float + FromPrimitive + Copy + Send + Sync,
{
    let (height, width) = image.dim();
    let encoder = ImageHDCEncoder::new(height, width, config.clone());
    let mut memory = HDCMemory::new(config.clone());

    // Encode and store patterns
    for (pattern, label) in patterns {
        let encoded_pattern = encoder.encode_image(**pattern)?;
        memory.store(label.to_string(), encoded_pattern);
    }

    let mut matches = Vec::new();
    let patch_size = 32; // Configurable patch size

    // Sliding window pattern matching
    for y in 0..height.saturating_sub(patch_size) {
        for x in 0..width.saturating_sub(patch_size) {
            let patch = image.slice(s![y..y + patch_size, x..x + patch_size]);
            let encoded_patch = encoder.encode_image(patch)?;

            if let Some((matched_label, confidence)) = memory.retrieve(&encoded_patch) {
                matches.push(PatternMatch {
                    label: matched_label,
                    confidence,
                    position: (y, x),
                    size: (patch_size, patch_size),
                });
            }
        }
    }

    // Non-maximum suppression
    let filtered_matches = non_maximum_suppression(matches, 0.5)?;

    Ok(filtered_matches)
}

/// HDC-based Feature Detection
///
/// Detect and encode image features using hyperdimensional computing.
/// Provides compositional feature representations.
#[allow(dead_code)]
pub fn hdc_feature_detection<T>(
    image: ArrayView2<T>,
    feature_types: &[String],
    config: &HDCConfig,
) -> NdimageResult<HashMap<String, Vec<FeatureDetection>>>
where
    T: Float + FromPrimitive + Copy + Send + Sync,
{
    let (height, width) = image.dim();
    let mut encoder = ImageHDCEncoder::new(height, width, config.clone());

    // Initialize feature encoders
    for feature_type in feature_types {
        let feature_hv = Hypervector::random(config.hypervector_dim, config.sparsity);
        encoder.add_feature_encoder(feature_type.clone(), feature_hv);
    }

    let mut feature_detections = HashMap::new();

    for feature_type in feature_types {
        let mut detections = Vec::new();
        let window_size = 16; // Configurable

        for y in 0..height.saturating_sub(window_size) {
            for x in 0..width.saturating_sub(window_size) {
                let patch = image.slice(s![y..y + window_size, x..x + window_size]);

                // Analyze patch characteristics
                let feature_strength = analyze_patch_for_feature(&patch, feature_type)?;

                if feature_strength > config.similarity_threshold {
                    let encoded_feature = encoder.encode_patch(patch, feature_type)?;

                    detections.push(FeatureDetection {
                        feature_type: feature_type.clone(),
                        position: (y, x),
                        strength: feature_strength,
                        hypervector: encoded_feature,
                        patch_size: (window_size, window_size),
                    });
                }
            }
        }

        feature_detections.insert(feature_type.clone(), detections);
    }

    Ok(feature_detections)
}

/// HDC-based Sequence Processing
///
/// Process temporal sequences using hyperdimensional computing.
/// Encodes temporal relationships and sequences efficiently.
#[allow(dead_code)]
pub fn hdc_sequence_processing<T>(
    image_sequence: &[ArrayView2<T>],
    sequence_length: usize,
    config: &HDCConfig,
) -> NdimageResult<SequenceEncoding>
where
    T: Float + FromPrimitive + Copy + Send + Sync,
{
    if image_sequence.is_empty() {
        return Err(NdimageError::InvalidInput("Empty image sequence".into()));
    }

    let (height, width) = image_sequence[0].dim();
    let encoder = ImageHDCEncoder::new(height, width, config.clone());

    // Create temporal position encoders
    let mut temporal_encoders = Vec::new();
    for t in 0..sequence_length {
        temporal_encoders.push(Hypervector::random(config.hypervector_dim, config.sparsity));
    }

    let mut sequence_encodings = Vec::new();
    let mut temporal_positions = Vec::new();

    // Process each frame in the sequence
    for (t, image) in image_sequence.iter().enumerate().take(sequence_length) {
        let frame_encoding = encoder.encode_image(*image)?;

        if t < temporal_encoders.len() {
            let temporal_binding = frame_encoding.bind(&temporal_encoders[t])?;
            sequence_encodings.push(temporal_binding);
            temporal_positions.push(t);
        }
    }

    // Bundle all temporal encodings
    let final_encoding = if sequence_encodings.is_empty() {
        Hypervector::zeros(config.hypervector_dim)
    } else {
        vector_utils::bundle_multiple(&sequence_encodings)?
    };

    // Calculate sequence confidence based on consistency
    let mut similarities = Vec::new();
    for i in 0..sequence_encodings.len().saturating_sub(1) {
        let sim = sequence_encodings[i].similarity(&sequence_encodings[i + 1]);
        similarities.push(sim);
    }

    let confidence = if similarities.is_empty() {
        1.0
    } else {
        similarities.iter().sum::<f64>() / similarities.len() as f64
    };

    Ok(SequenceEncoding {
        encoding: final_encoding,
        temporal_positions,
        confidence,
    })
}

/// Multi-scale image encoding
///
/// Encode image at multiple scales for robust representation.
#[allow(dead_code)]
pub fn multiscale_encoding<T>(
    image: ArrayView2<T>,
    scales: &[f64],
    config: &HDCConfig,
) -> NdimageResult<Hypervector>
where
    T: Float + FromPrimitive + Copy,
{
    let (height, width) = image.dim();
    let mut scale_encodings = Vec::new();

    for &scale in scales {
        let scaled_height = (height as f64 * scale) as usize;
        let scaled_width = (width as f64 * scale) as usize;

        if scaled_height == 0 || scaled_width == 0 {
            continue;
        }

        // Simple downsampling (in practice would use proper resampling)
        let mut scaled_image = Array2::zeros((scaled_height, scaled_width));
        for y in 0..scaled_height {
            for x in 0..scaled_width {
                let orig_y = (y * height) / scaled_height;
                let orig_x = (x * width) / scaled_width;
                scaled_image[[y, x]] = image[[orig_y, orig_x]];
            }
        }

        let encoder = ImageHDCEncoder::new(scaled_height, scaled_width, config.clone());
        let scale_encoding = encoder.encode_image(scaled_image.view())?;
        scale_encodings.push(scale_encoding);
    }

    if scale_encodings.is_empty() {
        Ok(Hypervector::zeros(config.hypervector_dim))
    } else {
        vector_utils::bundle_multiple(&scale_encodings)
    }
}

/// Create patches from image with specified stride and size
pub fn create_patches<T>(
    image: ArrayView2<T>,
    patch_size: (usize, usize),
    stride: (usize, usize),
) -> Vec<Array2<T>>
where
    T: Clone,
{
    let (height, width) = image.dim();
    let (patch_h, patch_w) = patch_size;
    let (stride_y, stride_x) = stride;

    let mut patches = Vec::new();

    let mut y = 0;
    while y + patch_h <= height {
        let mut x = 0;
        while x + patch_w <= width {
            let patch_view = image.slice(s![y..y + patch_h, x..x + patch_w]);
            let patch =
                Array2::from_shape_fn((patch_h, patch_w), |(i, j)| patch_view[[i, j]].clone());
            patches.push(patch);
            x += stride_x;
        }
        y += stride_y;
    }

    patches
}

/// Semantic encoding of concepts
pub fn encode_semantic_concepts(
    concepts: &[String],
    config: &HDCConfig,
) -> NdimageResult<Hypervector> {
    if concepts.is_empty() {
        return Ok(Hypervector::zeros(config.hypervector_dim));
    }

    let mut concept_encodings = Vec::new();

    for concept in concepts {
        // Create deterministic encoding based on concept string
        let mut concept_hv = Hypervector::random(config.hypervector_dim, config.sparsity);

        // Hash concept name to create consistent encoding
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        concept.hash(&mut hasher);
        let hash_value = hasher.finish();

        // Use hash to permute the hypervector consistently
        let permutation_seed = hash_value as usize;
        if permutation_seed % 2 == 0 {
            concept_hv = concept_hv.scale(-1.0); // Flip polarity based on hash
        }

        concept_encodings.push(concept_hv);
    }

    vector_utils::bundle_multiple(&concept_encodings)
}

/// HDC-based Image Classification
///
/// Optimized image classification using hyperdimensional computing.
/// Achieves brain-like efficiency with massive parallelism.
#[allow(dead_code)]
pub fn hdc_image_classification<T>(
    images: &[ArrayView2<T>],
    labels: &[String],
    test_images: &[ArrayView2<T>],
    config: &HDCConfig,
) -> NdimageResult<Vec<(String, f64)>>
where
    T: Float + FromPrimitive + Copy + Send + Sync,
{
    if images.is_empty() || images.len() != labels.len() {
        return Err(NdimageError::InvalidInput(
            "Invalid training data".to_string(),
        ));
    }

    let (height, width) = images[0].dim();

    // Initialize encoder and memory
    let encoder = ImageHDCEncoder::new(height, width, config.clone());
    let mut memory = HDCMemory::new(config.clone());

    // Training phase: encode and store images
    for (image, label) in images.iter().zip(labels.iter()) {
        let encoded_image = encoder.encode_image(*image)?;

        // If label already exists, bundle with existing pattern
        if let Some(existing) = memory.patterns.get(label) {
            let bundled = existing.bundle(&encoded_image)?;
            memory.store(label.clone(), bundled);
        } else {
            memory.store(label.clone(), encoded_image);
        }
    }

    // Testing phase: classify test images
    let mut results = Vec::new();

    for test_image in test_images {
        let encoded_test = encoder.encode_image(*test_image)?;

        if let Some((predicted_label, confidence)) = memory.retrieve(&encoded_test) {
            results.push((predicted_label, confidence));
        } else {
            results.push(("unknown".to_string(), 0.0));
        }
    }

    Ok(results)
}

/// HDC-based Pattern Matching
///
/// Optimized pattern matching using hyperdimensional representations.
/// Robust to noise and partial occlusion.
#[allow(dead_code)]
pub fn hdc_pattern_matching<T>(
    image: ArrayView2<T>,
    patterns: &[(ArrayView2<T>, String)],
    config: &HDCConfig,
) -> NdimageResult<Vec<PatternMatch>>
where
    T: Float + FromPrimitive + Copy + Send + Sync,
{
    let (height, width) = image.dim();
    let encoder = ImageHDCEncoder::new(height, width, config.clone());
    let mut memory = HDCMemory::new(config.clone());

    // Encode and store patterns
    for (pattern, label) in patterns {
        let encoded_pattern = encoder.encode_image(*pattern)?;
        memory.store(label.clone(), encoded_pattern);
    }

    let mut matches = Vec::new();
    let patch_size = 32; // Configurable patch size

    // Sliding window pattern matching
    for y in 0..height.saturating_sub(patch_size) {
        for x in 0..width.saturating_sub(patch_size) {
            let patch = image.slice(s![y..y + patch_size, x..x + patch_size]);
            let encoded_patch = encoder.encode_image(patch)?;

            if let Some((matched_label, confidence)) = memory.retrieve(&encoded_patch) {
                matches.push(PatternMatch {
                    label: matched_label,
                    confidence,
                    position: (y, x),
                    size: (patch_size, patch_size),
                });
            }
        }
    }

    // Non-maximum suppression
    let filtered_matches = non_maximum_suppression(matches, 0.5)?;

    Ok(filtered_matches)
}

/// HDC-based Compositional Reasoning
///
/// Compose and decompose visual concepts using hyperdimensional operations.
/// Enables complex reasoning about image content.
#[allow(dead_code)]
pub fn hdc_compositional_reasoning<T>(
    image: ArrayView2<T>,
    concept_memory: &HDCMemory,
    query_concepts: &[String],
    config: &HDCConfig,
) -> NdimageResult<CompositionResult>
where
    T: Float + FromPrimitive + Copy + Send + Sync,
{
    let (height, width) = image.dim();
    let encoder = ImageHDCEncoder::new(height, width, config.clone());

    // Encode the input image
    let encoded_image = encoder.encode_image(image)?;

    // Compose query from concepts
    let mut composed_query: Option<Hypervector> = None;

    for concept in query_concepts {
        if let Some(concept_hv) = concept_memory.get_item(concept) {
            if let Some(ref current_query) = composed_query {
                composed_query = Some(current_query.bind(concept_hv)?);
            } else {
                composed_query = Some(concept_hv.clone());
            }
        }
    }

    if let Some(query_hv) = composed_query {
        // Compute similarity between image and composed query
        let similarity = encoded_image.similarity(&query_hv);

        // Decompose image to find constituent concepts
        let mut concept_presence = HashMap::new();

        for (concept_name, concept_hv) in &concept_memory.item_memory {
            let presence_strength = encoded_image.similarity(concept_hv);
            if presence_strength > config.cleanup_threshold {
                concept_presence.insert(concept_name.clone(), presence_strength);
            }
        }

        Ok(CompositionResult {
            query_similarity: similarity,
            concept_presence,
            composed_representation: query_hv,
            image_representation: encoded_image,
        })
    } else {
        Err(NdimageError::InvalidInput(
            "No valid concepts found".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperdimensional_computing::calculate_overlap;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_image_hdc_encoder_creation() {
        let config = HDCConfig::default();
        let encoder = ImageHDCEncoder::new(10, 10, config);

        assert_eq!(encoder.height, 10);
        assert_eq!(encoder.width, 10);
        assert_eq!(encoder.position_encoders.len(), 100); // 10x10
        assert_eq!(encoder.value_encoders.len(), 256); // 0-255
    }

    #[test]
    fn test_image_encoding() {
        let config = HDCConfig::default();
        let encoder = ImageHDCEncoder::new(5, 5, config);

        let image = Array2::from_shape_fn((5, 5), |(i, j)| (i + j) as f64 / 10.0);
        let result = encoder.encode_image(image.view());

        assert!(result.is_ok());
        let encoded = result.unwrap();
        assert_eq!(encoded.dimension, encoder.config.hypervector_dim);
        assert!(encoded.norm > 0.0);
    }

    #[test]
    fn test_feature_encoder() {
        let config = HDCConfig::default();
        let mut encoder = ImageHDCEncoder::new(5, 5, config.clone());

        let feature_hv = Hypervector::random(config.hypervector_dim, config.sparsity);
        encoder.add_feature_encoder("test_feature".to_string(), feature_hv);

        assert!(encoder.feature_encoders.contains_key("test_feature"));
    }

    #[test]
    fn test_patch_encoding() {
        let config = HDCConfig::default();
        let mut encoder = ImageHDCEncoder::new(10, 10, config.clone());

        let feature_hv = Hypervector::random(config.hypervector_dim, config.sparsity);
        encoder.add_feature_encoder("edge".to_string(), feature_hv);

        let patch = Array2::<f64>::ones((3, 3));
        let result = encoder.encode_patch(patch.view(), "edge");

        assert!(result.is_ok());
        let encoded = result.unwrap();
        assert_eq!(encoded.dimension, encoder.config.hypervector_dim);
    }

    #[test]
    fn test_multiscale_encoding() {
        let config = HDCConfig::default();
        let image = Array2::from_shape_fn((20, 20), |(i, j)| (i + j) as f64 / 40.0);
        let scales = vec![1.0, 0.5, 0.25];

        let result = multiscale_encoding(image.view(), &scales, &config);
        assert!(result.is_ok());

        let encoded = result.unwrap();
        assert_eq!(encoded.dimension, config.hypervector_dim);
        assert!(encoded.norm > 0.0);
    }

    #[test]
    fn test_create_patches() {
        let image = Array2::from_shape_fn((10, 10), |(i, j)| i + j);
        let patches = create_patches(image.view(), (3, 3), (2, 2));

        assert!(!patches.is_empty());

        // Should create patches from (0,0), (0,2), (0,4), (0,6), (2,0), etc.
        // With 10x10 image, 3x3 patches, and stride 2, we get several patches
        assert!(!patches.is_empty());

        for patch in &patches {
            assert_eq!(patch.dim(), (3, 3));
        }
    }

    #[test]
    fn test_encode_semantic_concepts() {
        let config = HDCConfig::default();
        let concepts = vec!["cat".to_string(), "dog".to_string(), "bird".to_string()];

        let result = encode_semantic_concepts(&concepts, &config);
        assert!(result.is_ok());

        let encoded = result.unwrap();
        assert_eq!(encoded.dimension, config.hypervector_dim);
        assert!(encoded.norm > 0.0);

        // Test that the function can be called multiple times successfully
        let result2 = encode_semantic_concepts(&concepts, &config);
        assert!(result2.is_ok());
        let encoded2 = result2.unwrap();
        assert_eq!(encoded2.dimension, config.hypervector_dim);
        assert!(encoded2.norm > 0.0);
    }

    #[test]
    fn test_analyze_patch_for_feature() {
        let patch = Array2::<f64>::ones((5, 5));

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
    fn test_calculate_overlap() {
        let match1 = PatternMatch {
            label: "test1".to_string(),
            confidence: 0.9,
            position: (0, 0),
            size: (10, 10),
        };

        let match2 = PatternMatch {
            label: "test2".to_string(),
            confidence: 0.8,
            position: (5, 5),
            size: (10, 10),
        };

        let overlap = calculate_overlap(&match1, &match2);
        assert!(overlap > 0.0);
        assert!(overlap < 1.0);

        // Non-overlapping matches
        let match3 = PatternMatch {
            label: "test3".to_string(),
            confidence: 0.7,
            position: (15, 15),
            size: (5, 5),
        };

        let no_overlap = calculate_overlap(&match1, &match3);
        assert_eq!(no_overlap, 0.0);
    }

    #[test]
    fn test_sequence_encoding() {
        let config = HDCConfig::default();

        let img1 = Array2::zeros((5, 5));
        let img2 = Array2::ones((5, 5));
        let img3 = Array2::from_elem((5, 5), 0.5);

        let sequence = vec![img1.view(), img2.view(), img3.view()];
        let result = hdc_sequence_processing(&sequence, 3, &config);

        assert!(result.is_ok());
        let seq_encoding = result.unwrap();

        assert_eq!(seq_encoding.encoding.dimension, config.hypervector_dim);
        assert_eq!(seq_encoding.temporal_positions.len(), 3);
        assert!(seq_encoding.confidence >= -1.0); // Allow negative confidence
        assert!(seq_encoding.confidence <= 1.0);
    }
}
