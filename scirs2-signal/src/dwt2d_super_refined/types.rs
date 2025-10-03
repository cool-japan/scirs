//! Type definitions and configuration structures for advanced-refined 2D wavelet transforms
//!
//! This module contains all the data structures, enums, and configuration types
//! used throughout the advanced-refined 2D wavelet transform implementation.

use crate::dwt2d_enhanced::Dwt2dConfig;
use scirs2_core::ndarray::{Array2, Array3};

/// Advanced-refined 2D wavelet packet decomposition result
#[derive(Debug, Clone)]
pub struct AdvancedRefinedWaveletPacketResult {
    /// Wavelet packet coefficients organized by level and orientation
    pub coefficients: Array3<f64>, // [level][subband][data]
    /// Subband energy distribution
    pub energy_map: Array2<f64>,
    /// Optimal decomposition tree structure
    pub decomposition_tree: DecompositionTree,
    /// Advanced quality metrics
    pub quality_metrics: AdvancedRefinedQualityMetrics,
    /// Memory usage statistics
    pub memory_stats: MemoryStatistics,
    /// Processing performance metrics
    pub performance_metrics: ProcessingMetrics,
}

/// Advanced decomposition tree for wavelet packets
#[derive(Debug, Clone)]
pub struct DecompositionTree {
    /// Tree structure representing the decomposition
    pub nodes: Vec<TreeNode>,
    /// Optimal basis selection
    pub optimal_basis: Vec<usize>,
    /// Cost function used for basis selection
    pub cost_function: CostFunction,
    /// Tree traversal statistics
    pub traversal_stats: TreeTraversalStats,
}

/// Tree node for wavelet packet decomposition
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub level: usize,
    pub index: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub energy: f64,
    pub entropy: f64,
    pub is_leaf: bool,
    pub subband_type: SubbandType,
}

/// Subband classification for wavelet packets
#[derive(Debug, Clone, PartialEq)]
pub enum SubbandType {
    Approximation,
    HorizontalDetail,
    VerticalDetail,
    DiagonalDetail,
    Mixed(Vec<SubbandType>),
}

/// Cost functions for basis selection
#[derive(Debug, Clone, Copy)]
pub enum CostFunction {
    /// Shannon entropy
    Entropy,
    /// Threshold-based energy
    Energy,
    /// Log-energy entropy
    LogEntropy,
    /// Sure (Stein's unbiased risk estimate)
    Sure,
    /// Minimax
    Minimax,
    /// Custom adaptive cost
    Adaptive,
}

/// Tree traversal statistics
#[derive(Debug, Clone)]
pub struct TreeTraversalStats {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub max_depth: usize,
    pub avg_branching_factor: f64,
}

/// Advanced quality metrics for wavelet analysis
#[derive(Debug, Clone)]
pub struct AdvancedRefinedQualityMetrics {
    /// Overall perceptual quality score
    pub perceptual_quality: f64,
    /// Energy preservation ratio
    pub energy_preservation: f64,
    /// Coefficient sparsity measure
    pub sparsity: f64,
    /// Compression efficiency
    pub compression_ratio: f64,
    /// Edge preservation quality
    pub edge_preservation: f64,
    /// Frequency domain analysis
    pub frequency_analysis: FrequencyAnalysis,
    /// Compression-related metrics
    pub compression_metrics: CompressionMetrics,
}

/// Frequency domain analysis results
#[derive(Debug, Clone)]
pub struct FrequencyAnalysis {
    pub spectral_energy_distribution: Vec<f64>,
    pub dominant_frequencies: Vec<f64>,
    pub frequency_content_preservation: f64,
    pub aliasing_artifacts: f64,
}

/// Compression performance metrics
#[derive(Debug, Clone)]
pub struct CompressionMetrics {
    pub theoretical_compression_ratio: f64,
    pub effective_compression_ratio: f64,
    pub rate_distortion_score: f64,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
    pub memory_efficiency: f64,
    pub allocation_count: usize,
}

/// Processing performance metrics
#[derive(Debug, Clone)]
pub struct ProcessingMetrics {
    pub total_time_ms: f64,
    pub decomposition_time_ms: f64,
    pub simd_acceleration_factor: f64,
    pub parallel_efficiency: f64,
    pub cache_hit_ratio: f64,
}

/// Advanced-refined configuration for 2D wavelet operations
#[derive(Debug, Clone)]
pub struct AdvancedRefinedConfig {
    /// Base 2D wavelet configuration
    pub base_config: Dwt2dConfig,
    /// Maximum decomposition levels
    pub max_levels: usize,
    /// Minimum subband size to process
    pub min_subband_size: usize,
    /// Cost function for basis selection
    pub cost_function: CostFunction,
    /// Enable adaptive decomposition strategies
    pub adaptive_decomposition: bool,
    /// Use memory-efficient processing
    pub memory_efficient: bool,
    /// Tile size for tiled processing
    pub tile_size: (usize, usize),
    /// Overlap between tiles
    pub tile_overlap: usize,
    /// SIMD optimization level
    pub simd_level: SimdLevel,
    /// Quality computation configuration
    pub quality_config: QualityConfig,
}

/// SIMD optimization levels
#[derive(Debug, Clone, Copy)]
pub enum SimdLevel {
    None,
    Basic,
    Advanced,
    Aggressive,
}

/// Quality computation configuration
#[derive(Debug, Clone)]
pub struct QualityConfig {
    pub compute_perceptual_metrics: bool,
    pub compute_compression_metrics: bool,
    pub compute_frequency_analysis: bool,
    pub reference_image: Option<Array2<f64>>,
}

impl Default for AdvancedRefinedConfig {
    fn default() -> Self {
        Self {
            base_config: Dwt2dConfig::default(),
            max_levels: 6,
            min_subband_size: 4,
            cost_function: CostFunction::Adaptive,
            adaptive_decomposition: true,
            memory_efficient: true,
            tile_size: (256, 256),
            tile_overlap: 16,
            simd_level: SimdLevel::Advanced,
            quality_config: QualityConfig {
                compute_perceptual_metrics: true,
                compute_compression_metrics: true,
                compute_frequency_analysis: true,
                reference_image: None,
            },
        }
    }
}

/// Advanced reconstruction result
#[derive(Debug, Clone)]
pub struct AdvancedRefinedReconstructionResult {
    pub image: Array2<f64>,
    pub reconstruction_time_ms: f64,
    pub quality_metrics: ReconstructionQualityMetrics,
    pub coefficient_utilization: f64,
}

/// Reconstruction quality metrics
#[derive(Debug, Clone)]
pub struct ReconstructionQualityMetrics {
    pub reconstruction_error: f64,
    pub energy_preservation: f64,
    pub perceptual_similarity: f64,
}

/// Advanced denoising configuration
#[derive(Debug, Clone)]
pub struct AdvancedRefinedDenoisingConfig {
    pub noise_variance: Option<f64>,
    pub threshold_method: ThresholdMethod,
    pub edge_preservation: f64,
    pub perceptual_weighting: bool,
}

/// Threshold methods for denoising
#[derive(Debug, Clone, Copy)]
pub enum ThresholdMethod {
    Soft,
    Hard,
    Garrotte,
    Greater,
    Less,
    Adaptive,
}

/// Advanced denoising result
#[derive(Debug, Clone)]
pub struct AdvancedRefinedDenoisingResult {
    pub denoised_image: Array2<f64>,
    pub noise_analysis: NoiseAnalysis,
    pub denoising_time_ms: f64,
    pub quality_metrics: DenoisingQualityMetrics,
    pub coefficient_statistics: CoefficientStatistics,
}

/// Noise characteristic analysis
#[derive(Debug, Clone)]
pub struct NoiseAnalysis {
    pub noise_type: NoiseType,
    pub estimated_variance: f64,
    pub spatial_correlation: f64,
    pub frequency_characteristics: Vec<f64>,
}

/// Types of noise patterns
#[derive(Debug, Clone)]
pub enum NoiseType {
    Gaussian,
    Poisson,
    Impulse,
    Uniform,
    Mixed,
}

/// Denoising quality assessment
#[derive(Debug, Clone)]
pub struct DenoisingQualityMetrics {
    pub psnr: f64,
    pub ssim: f64,
    pub edge_preservation_index: f64,
    pub artifacts_score: f64,
}

/// Coefficient statistical properties
#[derive(Debug, Clone)]
pub struct CoefficientStatistics {
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Internal processing result structure
#[derive(Debug, Clone)]
pub(crate) struct ProcessingResult {
    pub coefficients: Array3<f64>,
    pub energy_map: Array2<f64>,
    pub parallel_efficiency: f64,
}

/// SIMD configuration for optimization
#[derive(Debug, Clone)]
pub(crate) struct SimdConfiguration {
    pub use_avx2: bool,
    pub use_sse: bool,
    pub acceleration_factor: f64,
}

/// Memory tracking for optimization
#[derive(Debug, Clone)]
pub(crate) struct MemoryTracker {
    allocations: std::collections::HashMap<String, f64>,
    peak_usage: f64,
    current_usage: f64,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            allocations: std::collections::HashMap::new(),
            peak_usage: 0.0,
            current_usage: 0.0,
        }
    }

    pub fn track_allocation(&mut self, name: &str, size_mb: f64) {
        self.allocations.insert(name.to_string(), size_mb);
        self.current_usage += size_mb;
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
    }

    pub fn finalize(self) -> MemoryStatistics {
        let avg_memory = if !self.allocations.is_empty() {
            self.allocations.values().sum::<f64>() / self.allocations.len() as f64
        } else {
            0.0
        };

        MemoryStatistics {
            peak_memory_mb: self.peak_usage,
            average_memory_mb: avg_memory,
            memory_efficiency: if self.peak_usage > 0.0 {
                avg_memory / self.peak_usage
            } else {
                0.0
            },
            allocation_count: self.allocations.len(),
        }
    }
}

/// Perceptual reconstruction engine
#[derive(Debug)]
pub(crate) struct PerceptualReconstructionEngine {
    config: AdvancedRefinedConfig,
}

impl PerceptualReconstructionEngine {
    pub fn new(config: &AdvancedRefinedConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}
