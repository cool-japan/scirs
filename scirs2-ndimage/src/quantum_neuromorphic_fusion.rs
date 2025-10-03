//! Quantum-Neuromorphic Fusion for Image Processing
//!
//! This module implements next-generation algorithms that fuse quantum computing
//! principles with neuromorphic processing for unprecedented image processing
//! capabilities. It represents the cutting edge of bio-quantum computation.
//!
//! # Revolutionary Features
//!
//! - **Quantum Spiking Neural Networks**: Fusion of quantum superposition with spike-based processing
//! - **Neuromorphic Quantum Entanglement**: Bio-inspired quantum correlation processing
//! - **Quantum-Enhanced Synaptic Plasticity**: STDP with quantum coherence effects
//! - **Bio-Quantum Reservoir Computing**: Quantum liquid state machines with biological dynamics
//! - **Quantum Homeostatic Adaptation**: Self-organizing quantum-bio systems
//! - **Quantum Memory Consolidation**: Sleep-inspired quantum state optimization
//! - **Quantum Attention Mechanisms**: Bio-inspired quantum attention for feature selection
//! - **Quantum-Enhanced Temporal Coding**: Temporal spike patterns with quantum interference
//! - **Advanced Quantum-Classical Hybrid Processing**: Sophisticated integration algorithms
//! - **Quantum Error Correction for Classical Systems**: Quantum ECC integrated with classical processing
//! - **Quantum-Classical Meta-Learning**: Hybrid learning across quantum and classical domains
//!
//! # Module Architecture
//!
//! The quantum-neuromorphic fusion algorithms have been refactored into focused modules:
//!
//! - **`config`**: Configuration structures and data types for all quantum-neuromorphic processing
//! - **`quantum_neural`**: Core quantum spiking neural networks, entanglement, and bio-quantum processing
//! - **`consciousness`**: Consciousness-inspired processing based on leading neuroscience theories
//! - **`hybrid_processing`**: Quantum-classical hybrid processing with error correction and optimization
//!
//! # Usage Examples
//!
//! ## Quantum Spiking Neural Network Processing
//!
//! ```rust,ignore
//! use scirs2_ndimage::quantum_neuromorphic_fusion::*;
//! use scirs2_core::ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create configuration
//! let config = QuantumNeuromorphicConfig::default();
//!
//! // Process image with quantum spiking neural network
//! let image = Array2::ones((64, 64));
//! let result = quantum_spiking_neural_network(image.view(), &config)?;
//! println!("Quantum neural processing result: {:?}", result.dim());
//! # Ok(())
//! # }
//! ```
//!
//! ## Consciousness-Inspired Global Workspace
//!
//! ```rust,ignore
//! use scirs2_ndimage::quantum_neuromorphic_fusion::*;
//! use scirs2_core::ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create consciousness configuration
//! let config = ConsciousnessConfig::default();
//!
//! // Process with consciousness-inspired global workspace
//! let image = Array2::ones((128, 128));
//! let result = consciousness_inspired_global_workspace(image.view(), &config)?;
//! println!("Consciousness processing result: {:?}", result.dim());
//! # Ok(())
//! # }
//! ```
//!
//! ## Quantum-Classical Hybrid Processing
//!
//! ```rust,ignore
//! use scirs2_ndimage::quantum_neuromorphic_fusion::*;
//! use scirs2_core::ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create hybrid configuration
//! let config = QuantumClassicalHybridConfig::default();
//!
//! // Process with quantum-classical hybrid system
//! let image = Array2::ones((256, 256));
//! let result = advanced_quantum_classical_hybrid_processing(image.view(), &config)?;
//! println!("Hybrid processing completed with quantum speedup: {:.2}x",
//!          result.quantum_contribution / result.classical_contribution);
//! # Ok(())
//! # }
//! ```
//!
//! # Performance Characteristics
//!
//! The quantum-neuromorphic fusion algorithms provide:
//! - **Bio-inspired Efficiency**: Processing patterns inspired by biological neural networks
//! - **Quantum Enhancement**: Leveraging quantum superposition and entanglement for speed
//! - **Self-Organization**: Adaptive networks that evolve based on input patterns
//! - **Consciousness Integration**: Global workspace theory for coherent information processing
//! - **Hybrid Optimization**: Intelligent switching between quantum and classical processing
//! - **Error Resilience**: Quantum error correction integrated with classical fault tolerance
//!
//! # Theoretical Foundation
//!
//! This module implements cutting-edge theories from:
//! - **Quantum Computing**: Superposition, entanglement, and quantum information theory
//! - **Neuromorphic Engineering**: Spike-based processing and synaptic plasticity
//! - **Consciousness Research**: Global Workspace Theory, Integrated Information Theory
//! - **Computational Neuroscience**: Predictive coding, meta-cognitive monitoring
//! - **Bio-Quantum Interface**: Quantum effects in biological systems

// Re-export all module components for backward compatibility and ease of use
pub use self::{
    config::{ConsciousnessConfig, QuantumClassicalHybridConfig, QuantumNeuromorphicConfig},
    consciousness::*,
    hybrid_processing::{
        advanced_quantum_classical_hybrid_processing, QuantumClassicalHybridProcessor,
    },
    quantum_neural::*,
};

// Module declarations
pub mod config;
pub mod consciousness;
pub mod hybrid_processing;
pub mod quantum_neural;
