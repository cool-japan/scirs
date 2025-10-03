//! Configuration types for dataset generators

use crate::error::{DatasetsError, Result};

/// Missing data patterns for noise injection
#[derive(Debug, Clone, Copy)]
pub enum MissingPattern {
    /// Missing Completely At Random
    MCAR,
    /// Missing At Random
    MAR,
    /// Missing Not At Random
    MNAR,
    /// Block missing pattern
    Block,
}

/// Outlier types for injection
#[derive(Debug, Clone, Copy)]
pub enum OutlierType {
    /// Individual point outliers
    Point,
    /// Context-dependent outliers
    Contextual,
    /// Collective outliers (groups of anomalous points)
    Collective,
}

/// GPU-accelerated data generation configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
    /// GPU device index (0 for default)
    pub device_id: usize,
    /// Whether to use single precision (f32) instead of double (f64)
    pub use_single_precision: bool,
    /// Chunk size for GPU operations
    pub chunk_size: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            use_gpu: true,
            device_id: 0,
            use_single_precision: false,
            chunk_size: 10000,
        }
    }
}

impl GpuConfig {
    /// Create a new GPU configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to use GPU
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set GPU device ID
    pub fn with_device(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }

    /// Set precision mode
    pub fn with_single_precision(mut self, single_precision: bool) -> Self {
        self.use_single_precision = single_precision;
        self
    }

    /// Set chunk size
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }
}

/// Advanced manifold configuration for complex datasets
#[derive(Debug, Clone)]
pub struct ManifoldConfig {
    /// Type of manifold to generate
    pub manifold_type: ManifoldType,
    /// Number of samples
    pub n_samples: usize,
    /// Noise level
    pub noise: f64,
    /// Random seed
    pub randomseed: Option<u64>,
    /// Manifold-specific parameters
    pub parameters: std::collections::HashMap<String, f64>,
}

/// Types of manifolds that can be generated
#[derive(Debug, Clone)]
pub enum ManifoldType {
    /// S-curve manifold
    SCurve,
    /// Swiss roll (with optional hole)
    SwissRoll {
        /// Whether to create a hole in the middle
        hole: bool,
    },
    /// Severed sphere
    SeveredSphere,
    /// Twin peaks
    TwinPeaks,
    /// Helix with specified turns
    Helix {
        /// Number of turns in the helix
        n_turns: f64,
    },
    /// Intersecting manifolds
    IntersectingManifolds,
    /// Torus with major and minor radii
    Torus {
        /// Major radius of the torus
        major_radius: f64,
        /// Minor radius of the torus
        minor_radius: f64,
    },
}

impl Default for ManifoldConfig {
    fn default() -> Self {
        Self {
            manifold_type: ManifoldType::SCurve,
            n_samples: 1000,
            noise: 0.1,
            randomseed: None,
            parameters: std::collections::HashMap::new(),
        }
    }
}

impl ManifoldConfig {
    /// Create a new manifold configuration
    pub fn new(manifold_type: ManifoldType) -> Self {
        Self {
            manifold_type,
            ..Default::default()
        }
    }

    /// Set number of samples
    pub fn with_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set noise level
    pub fn with_noise(mut self, noise: f64) -> Self {
        self.noise = noise;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.randomseed = Some(seed);
        self
    }

    /// Add a parameter
    pub fn with_parameter(mut self, name: String, value: f64) -> Self {
        self.parameters.insert(name, value);
        self
    }
}
