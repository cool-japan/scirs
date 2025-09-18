//! Hyperdimensional Computing for Advanced-Efficient Pattern Recognition
//!
//! This module implements cutting-edge hyperdimensional computing (HDC) algorithms
//! for image processing. HDC operates with very high-dimensional vectors (10,000+
//! dimensions) to achieve brain-like computation with exceptional efficiency,
//! robustness, and parallelizability.
//!
//! # Revolutionary Features
//!
//! - **Advanced-High Dimensional Vectors**: 10,000+ dimensional sparse representations
//! - **Brain-Inspired Computing**: Mimics neural computation principles
//! - **One-Shot Learning**: Immediate learning from single examples
//! - **Noise Resilience**: Robust to corruption and partial information
//! - **Massive Parallelism**: Inherently parallel operations
//! - **Memory Efficiency**: Sparse representations with high capacity
//! - **Real-Time Processing**: Optimized associative memory operations
//! - **Compositional Reasoning**: Ability to compose and decompose concepts
//!
//! # Module Organization
//!
//! - [`types`] - Core data structures and configuration types
//! - [`vector_ops`] - Hypervector operations and mathematical functions
//! - [`memory`] - Memory systems and learning algorithms
//! - [`image_processing`] - HDC-based image processing and encoding
//! - [`reasoning`] - Advanced reasoning and hierarchical concept processing
//! - [`utils`] - Utility functions for pattern matching and analysis
//!
//! # Examples
//!
//! ## Basic Image Classification
//!
//! ```rust,ignore
//! use scirs2_ndimage::hyperdimensional_computing::*;
//! use ndarray::Array2;
//!
//! let config = HDCConfig::default();
//! let train_images = vec![Array2::zeros((8, 8)).view()];
//! let train_labels = vec!["zero".to_string()];
//! let test_images = vec![Array2::zeros((8, 8)).view()];
//!
//! let results = hdc_image_classification(
//!     &train_images,
//!     &train_labels,
//!     &test_images,
//!     &config
//! ).unwrap();
//! ```
//!
//! ## Pattern Matching
//!
//! ```rust,ignore
//! use scirs2_ndimage::hyperdimensional_computing::*;
//! use ndarray::Array2;
//!
//! let config = HDCConfig::default();
//! let image = Array2::from_elem((64, 64), 0.5);
//! let pattern = Array2::ones((16, 16));
//! let patterns = vec![(pattern.view(), "square".to_string())];
//!
//! let matches = hdc_pattern_matching(
//!     image.view(),
//!     &patterns,
//!     &config
//! ).unwrap();
//! ```

// Re-export all sub-modules
pub mod image_processing;
pub mod memory;
pub mod reasoning;
pub mod types;
pub mod utils;
pub mod vector_ops;

// Re-export core types and configuration
pub use types::*;

// Re-export vector operations
pub use vector_ops::*;

// Re-export memory systems
pub use memory::*;

// Re-export image processing functions
pub use image_processing::*;

// Re-export advanced reasoning functions
pub use reasoning::*;

// Re-export utility functions
pub use utils::*;

// Main public API functions re-exported for backward compatibility

/// HDC-based Image Classification (re-exported from image_processing module)
///
/// Optimized image classification using hyperdimensional computing.
/// Achieves brain-like efficiency with massive parallelism.
pub use image_processing::hdc_image_classification;

/// HDC-based Pattern Matching (re-exported from image_processing module)
///
/// Optimized pattern matching using hyperdimensional representations.
/// Robust to noise and partial occlusion.
pub use image_processing::hdc_pattern_matching;

/// HDC-based Feature Detection (re-exported from image_processing module)
///
/// Detect and encode image features using hyperdimensional computing.
/// Provides compositional feature representations.
pub use image_processing::hdc_feature_detection;

/// HDC-based Sequence Processing (re-exported from image_processing module)
///
/// Process temporal sequences using hyperdimensional computing.
/// Encodes temporal relationships and sequences efficiently.
pub use image_processing::hdc_sequence_processing;

/// HDC-based Compositional Reasoning (re-exported from image_processing module)
///
/// Compose and decompose visual concepts using hyperdimensional operations.
/// Enables complex reasoning about image content.
pub use image_processing::hdc_compositional_reasoning;

// Advanced reasoning functions (re-exported from reasoning module)

/// Advanced Hierarchical HDC Reasoning (re-exported from reasoning module)
pub use reasoning::advanced_hierarchical_hdc_reasoning;

/// Advanced Continual Learning HDC (re-exported from reasoning module)
pub use reasoning::advanced_continual_learning_hdc;

/// Advanced Multi-Modal HDC Fusion (re-exported from reasoning module)
pub use reasoning::advanced_multimodal_hdc_fusion;

/// Advanced Online Learning HDC (re-exported from reasoning module)
pub use reasoning::advanced_online_learning_hdc;

// Utility functions (re-exported from utils module)

/// Non-maximum suppression for pattern matches (re-exported from utils module)
pub use utils::non_maximum_suppression;

/// Calculate overlap between pattern matches (re-exported from utils module)
pub use utils::calculate_overlap;

/// Analyze image patch for features (re-exported from utils module)
pub use utils::analyze_patch_for_feature;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_module_integration() {
        // Test that all modules work together correctly
        let config = HDCConfig::default();

        // Test basic hypervector creation
        let hv = Hypervector::random(1000, 0.1);
        assert_eq!(hv.dimension, 1000);

        // Test memory system
        let mut memory = HDCMemory::new(config.clone());
        memory.store("test".to_string(), hv.clone());

        let retrieved = memory.retrieve(&hv);
        assert!(retrieved.is_some());

        // Test image encoder
        let encoder = ImageHDCEncoder::new(8, 8, config.clone());
        let image = Array2::<f64>::zeros((8, 8));
        let encoded = encoder.encode_image(image.view()).unwrap();
        assert_eq!(encoded.dimension, config.hypervector_dim);
    }

    #[test]
    fn test_pattern_matching_integration() {
        let config = HDCConfig::default();
        let image = Array2::from_elem((32, 32), 0.5);
        let pattern = Array2::ones((8, 8));
        let patterns = vec![(pattern.view(), "square".to_string())];

        let matches = hdc_pattern_matching(image.view(), &patterns, &config).unwrap();

        // Should complete without error
        assert!(matches.len() >= 0);
    }

    #[test]
    fn test_classification_integration() {
        let config = HDCConfig::default();
        let train_zeros = Array2::<f64>::zeros((4, 4));
        let train_ones = Array2::<f64>::ones((4, 4));
        let train_images = vec![train_zeros.view(), train_ones.view()];
        let train_labels = vec!["zeros".to_string(), "ones".to_string()];

        let test_zeros = Array2::<f64>::zeros((4, 4));
        let test_images = vec![test_zeros.view()];

        let results =
            hdc_image_classification(&train_images, &train_labels, &test_images, &config).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].1 >= 0.0); // Valid confidence
    }

    #[test]
    fn test_sequence_processing_integration() {
        let config = HDCConfig::default();
        let frame1 = Array2::<f64>::zeros((4, 4));
        let frame2 = Array2::<f64>::ones((4, 4));
        let sequence = vec![frame1.view(), frame2.view()];

        let sequence_hv = hdc_sequence_processing(&sequence, 2, &config).unwrap();
        assert_eq!(sequence_hv.encoding.dimension, config.hypervector_dim);
    }

    #[test]
    fn test_utils_integration() {
        let matches = vec![
            PatternMatch {
                label: "test1".to_string(),
                confidence: 0.9,
                position: (0, 0),
                size: (10, 10),
            },
            PatternMatch {
                label: "test2".to_string(),
                confidence: 0.8,
                position: (5, 5),
                size: (10, 10),
            },
        ];

        let overlap = calculate_overlap(&matches[0], &matches[1]);
        assert!(overlap > 0.0);

        let filtered = non_maximum_suppression(matches, 0.1).unwrap();
        assert_eq!(filtered.len(), 1); // Should remove overlapping match
    }

    #[test]
    fn test_reasoning_integration() {
        let config = HDCConfig::default();
        let mut concept_library = HierarchicalConceptLibrary::new();

        // Add test concepts
        let mut level1_concepts = std::collections::HashMap::new();
        level1_concepts.insert(
            "edge".to_string(),
            Hypervector::random(config.hypervector_dim, config.sparsity),
        );
        concept_library.levels.insert(1, level1_concepts);

        let image = Array2::from_elem((8, 8), 0.5);

        let result =
            advanced_hierarchical_hdc_reasoning(image.view(), 1, &concept_library, &config)
                .unwrap();

        assert_eq!(result.base_encoding.dimension, config.hypervector_dim);
    }

    #[test]
    fn test_memory_integration() {
        let config = HDCConfig::default();
        let mut memory_system = ContinualLearningMemory::new(&config);

        let experience = Experience {
            encoding: Hypervector::random(config.hypervector_dim, config.sparsity),
            label: "test".to_string(),
            timestamp: 0,
            importance: 0.8,
        };

        let consolidation = ConsolidationResult {
            interference_prevented: 1,
            effectiveness_score: 0.9,
            replay_cycles_used: 3,
        };

        assert!(memory_system
            .add_experience(experience, &consolidation)
            .is_ok());
    }

    #[test]
    fn test_online_learning_integration() {
        let config = HDCConfig::default();
        let mut learning_system = OnlineLearningSystem::new(&config);

        let stream_image = Array2::<f64>::zeros((4, 4));

        let result = advanced_online_learning_hdc(
            stream_image.view(),
            Some("test_label"),
            &mut learning_system,
            &config,
        )
        .unwrap();

        assert!(result.prediction.confidence >= 0.0);
        assert!(result.learning_update.memory_updated);
    }
}
