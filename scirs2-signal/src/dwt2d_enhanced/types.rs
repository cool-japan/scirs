//! Type definitions for enhanced 2D DWT functionality
//!
//! This module contains all the type definitions used throughout the enhanced 2D DWT
//! implementation, including:
//! - Result structures for decomposition operations
//! - Quality metrics for analysis
//! - Configuration types and boundary handling modes
//! - Statistics and coefficient analysis types
//! - Denoising method enumerations

use crate::dwt::Wavelet;
use scirs2_core::ndarray::Array2;

/// Enhanced 2D DWT decomposition result
///
/// This structure contains the complete result of a 2D DWT decomposition,
/// including all coefficient subbands, metadata, and optional quality metrics.
#[derive(Debug, Clone)]
pub struct EnhancedDwt2dResult {
    /// Approximation coefficients (LL)
    pub approx: Array2<f64>,
    /// Horizontal detail coefficients (LH)
    pub detail_h: Array2<f64>,
    /// Vertical detail coefficients (HL)
    pub detail_v: Array2<f64>,
    /// Diagonal detail coefficients (HH)
    pub detail_d: Array2<f64>,
    /// Original shape for perfect reconstruction
    pub originalshape: (usize, usize),
    /// Boundary mode used
    pub boundary_mode: BoundaryMode,
    /// Quality metrics (if computed)
    pub metrics: Option<Dwt2dQualityMetrics>,
}

/// Quality metrics for 2D DWT analysis
///
/// Provides comprehensive quality assessment metrics for DWT decompositions,
/// including energy preservation, compression characteristics, and edge preservation.
#[derive(Debug, Clone)]
pub struct Dwt2dQualityMetrics {
    /// Energy in approximation band
    pub approx_energy: f64,
    /// Energy in detail bands
    pub detail_energy: f64,
    /// Total energy preservation
    pub energy_preservation: f64,
    /// Compression ratio estimate
    pub compression_ratio: f64,
    /// Sparsity measure
    pub sparsity: f64,
    /// Edge preservation metric
    pub edge_preservation: f64,
}

/// Boundary handling modes
///
/// Defines various strategies for handling image boundaries during DWT operations.
/// Different modes provide different trade-offs between computational efficiency,
/// edge artifacts, and reconstruction accuracy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryMode {
    /// Zero padding
    Zero,
    /// Symmetric extension (reflect)
    Symmetric,
    /// Periodic extension (wrap)
    Periodic,
    /// Constant extension
    Constant(f64),
    /// Anti-symmetric extension
    AntiSymmetric,
    /// Smooth extension (polynomial)
    Smooth,
    /// Adaptive extension based on local characteristics
    Adaptive,
    /// Extrapolation using local gradients
    Extrapolate,
    /// Mirroring with edge correction
    MirrorCorrect,
    /// Content-aware padding
    ContentAware,
}

/// Configuration for enhanced 2D DWT
///
/// Comprehensive configuration structure that controls all aspects of the
/// enhanced 2D DWT computation, including performance optimizations,
/// boundary handling, and quality analysis options.
#[derive(Debug, Clone)]
pub struct Dwt2dConfig {
    /// Boundary handling mode
    pub boundary_mode: BoundaryMode,
    /// Use SIMD optimization
    pub use_simd: bool,
    /// Use parallel processing
    pub use_parallel: bool,
    /// Minimum size for parallel processing
    pub parallel_threshold: usize,
    /// Precision tolerance
    pub tolerance: f64,
    /// Enable memory optimization for large images
    pub memory_optimized: bool,
    /// Block size for memory-optimized processing
    pub block_size: usize,
    /// Enable advanced quality metrics
    pub compute_metrics: bool,
}

impl Default for Dwt2dConfig {
    fn default() -> Self {
        Self {
            boundary_mode: BoundaryMode::Symmetric,
            use_simd: true,
            use_parallel: true,
            parallel_threshold: 64,
            tolerance: 1e-12,
            memory_optimized: false,
            block_size: 512,
            compute_metrics: false,
        }
    }
}

/// Multilevel 2D DWT decomposition
///
/// Contains the complete multilevel decomposition result, including
/// the coarsest approximation and all detail coefficients at each level.
pub struct MultilevelDwt2d {
    /// Approximation at coarsest level
    pub approx: Array2<f64>,
    /// Detail coefficients at each level
    pub details: Vec<(Array2<f64>, Array2<f64>, Array2<f64>)>,
    /// Original shape
    pub originalshape: (usize, usize),
    /// Wavelet used
    pub wavelet: Wavelet,
    /// Configuration
    pub config: Dwt2dConfig,
}

/// Statistics for 2D DWT analysis
///
/// Provides detailed statistical analysis of DWT coefficients across
/// multiple decomposition levels, including energy, entropy, and sparsity measures.
#[derive(Debug, Clone)]
pub struct Dwt2dStatistics {
    /// Energy at each decomposition level
    pub level_energies: Vec<f64>,
    /// Entropy at each decomposition level
    pub level_entropies: Vec<f64>,
    /// Sparsity at each decomposition level
    pub level_sparsities: Vec<f64>,
    /// Energy in approximation coefficients
    pub approx_energy: f64,
    /// Total number of decomposition levels
    pub total_levels: usize,
}

/// Denoising methods
///
/// Enumeration of available denoising algorithms for wavelet-based
/// noise reduction, each with different characteristics and use cases.
#[derive(Debug, Clone)]
pub enum DenoisingMethod {
    /// Soft thresholding
    Soft,
    /// Hard thresholding
    Hard,
    /// SURE (Stein's Unbiased Risk Estimator)
    Sure,
    /// BayesShrink
    BayesShrink,
    /// BiShrink (bivariate shrinkage)
    BiShrink,
    /// Non-local means in wavelet domain
    NonLocalMeans,
}

/// Statistics for wavelet coefficient validation
///
/// Internal structure used for tracking coefficient statistics during
/// validation and quality assessment procedures.
#[derive(Debug)]
struct WaveletCoefficientStats {
    approx_max: f64,
    detail_max: f64,
    total_energy: f64,
    approx_energy: f64,
    detail_energy: f64,
}

impl WaveletCoefficientStats {
    fn new() -> Self {
        Self {
            approx_max: f64::NEG_INFINITY,
            detail_max: f64::NEG_INFINITY,
            total_energy: 0.0,
            approx_energy: 0.0,
            detail_energy: 0.0,
        }
    }

    fn update(
        &mut self,
        subband_name: &str,
        max_val: f64,
        _min_val: f64,
        _mean: f64,
        variance: f64,
    ) {
        let energy = variance;

        match subband_name {
            "approximation" => {
                self.approx_max = max_val;
                self.approx_energy = energy;
            }
            _ => {
                self.detail_max = self.detail_max.max(max_val);
                self.detail_energy += energy;
            }
        }

        self.total_energy += energy;
    }
}
