//! # Advanced Fusion Core - Ultimate Image Processing Engine
//!
//! This module represents the pinnacle of image processing technology, combining:
//! - **Quantum-Classical Hybrid Computing**: Seamless integration of quantum and classical algorithms
//! - **Bio-Inspired Meta-Learning**: Self-evolving algorithms that adapt like biological systems
//! - **Consciousness-Level Processing**: Human-like attention and awareness mechanisms
//! - **Advanced-Dimensional Analysis**: Processing beyond traditional spatial dimensions
//! - **Temporal-Causal Intelligence**: Understanding of time and causality in image sequences
//! - **Self-Organizing Neural Architectures**: Networks that redesign themselves
//! - **Quantum Consciousness Simulation**: Computational models of awareness and perception
//! - **Advanced-Efficient Resource Management**: Optimal utilization of all available compute resources
//!
//! ## Module Architecture
//!
//! The advanced fusion algorithms have been refactored into focused modules for better maintainability:
//!
//! - **`config`**: Configuration types and data structures
//! - **`core_processing`**: Main fusion processing pipeline and orchestration
//! - **`feature_extraction`**: Multi-dimensional feature extraction capabilities
//! - **`quantum_consciousness`**: Quantum consciousness simulation and evolution
//! - **`neural_processing`**: Self-organizing neural networks and biological processing
//! - **`temporal_causality`**: Temporal pattern analysis and causal inference
//! - **`meta_learning`**: Adaptive meta-learning with memory consolidation
//! - **`resource_scheduling`**: Quantum-aware resource scheduling and optimization
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use scirs2_ndimage::advanced_fusion_algorithms::*;
//! use scirs2_core::ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create configuration
//! let config = AdvancedConfig::default();
//!
//! // Process image with advanced fusion
//! let image = Array2::zeros((256, 256));
//! let (result, final_state) = fusion_processing(image.view(), &config, None)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Characteristics
//!
//! The advanced fusion algorithms are designed for high-performance processing with:
//! - Parallel processing capabilities
//! - Memory-efficient operations
//! - Adaptive resource utilization
//! - Real-time processing support
//! - Quantum-enhanced acceleration (when available)

// Re-export all module components for backward compatibility and ease of use
// Note: Some functions are exported from specific modules to avoid conflicts
pub use self::{
    config::*,
    core_processing::fusion_processing,
    feature_extraction::*,
    meta_learning::{enhanced_meta_learning_with_temporal_fusion, meta_learning_adaptation},
    neural_processing::*,
    quantum_consciousness::{
        enhanced_quantum_consciousness_evolution, simulate_quantum_consciousness,
        QuantumConsciousnessEvolution,
    },
    resource_scheduling::quantum_aware_resource_scheduling_optimization,
    temporal_causality::*,
};

// Re-export remaining core_processing functions to maintain compatibility
pub use self::core_processing::{
    generate_consciousness_guided_output, initialize_or_updatestate, multi_scale_integration,
    optimize_resource_allocation, predict_future_load, update_efficiencymetrics,
};

// Module declarations
pub mod config;
pub mod core_processing;
pub mod feature_extraction;
pub mod meta_learning;
pub mod neural_processing;
pub mod quantum_consciousness;
pub mod resource_scheduling;
pub mod temporal_causality;
