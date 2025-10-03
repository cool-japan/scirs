//! Configuration Structures for Quantum-Neuromorphic Fusion
//!
//! This module contains all configuration structures and supporting types
//! for quantum-neuromorphic fusion algorithms. These configurations enable
//! customization of quantum computing principles fused with neuromorphic
//! processing for advanced image processing capabilities.
//!
//! # Configuration Categories
//!
//! - **Core Configurations**: Main fusion algorithm configurations
//! - **Interface Configurations**: Quantum and classical interface settings
//! - **Resource Configurations**: Resource requirement and constraint specifications
//! - **Data Pipeline Configurations**: Conversion and processing pipeline settings
//! - **Consciousness Configurations**: Bio-inspired consciousness processing settings
//!
//! # Features
//!
//! - Quantum spiking neural network configuration
//! - Quantum-classical hybrid processing settings
//! - Consciousness-inspired processing parameters
//! - Resource management and constraint specifications
//! - Data conversion pipeline configurations

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex;
use std::collections::VecDeque;

// Import from parent modules
use crate::neuromorphic_computing::{NeuromorphicConfig, SpikingNeuron};
use crate::quantum_inspired::QuantumConfig;

/// Configuration for quantum-neuromorphic fusion algorithms
#[derive(Debug, Clone)]
pub struct QuantumNeuromorphicConfig {
    /// Quantum configuration parameters
    pub quantum: QuantumConfig,
    /// Neuromorphic configuration parameters
    pub neuromorphic: NeuromorphicConfig,
    /// Quantum coherence preservation time
    pub coherence_time: f64,
    /// Strength of quantum-biological coupling
    pub quantum_bio_coupling: f64,
    /// Quantum decoherence rate
    pub decoherence_rate: f64,
    /// Number of quantum states per neuron
    pub quantumstates_per_neuron: usize,
    /// Quantum memory consolidation cycles
    pub consolidation_cycles: usize,
    /// Attention gate quantum threshold
    pub attention_threshold: f64,
}

impl Default for QuantumNeuromorphicConfig {
    fn default() -> Self {
        Self {
            quantum: QuantumConfig::default(),
            neuromorphic: NeuromorphicConfig::default(),
            coherence_time: 50.0,
            quantum_bio_coupling: 0.3,
            decoherence_rate: 0.02,
            quantumstates_per_neuron: 4,
            consolidation_cycles: 10,
            attention_threshold: 0.7,
        }
    }
}

/// Quantum spiking neuron with superposition states
#[derive(Debug, Clone)]
pub struct QuantumSpikingNeuron {
    /// Classical spiking neuron properties
    pub classical_neuron: SpikingNeuron,
    /// Quantum state amplitudes for different neural states
    pub quantum_amplitudes: Array1<Complex<f64>>,
    /// Quantum coherence matrix
    pub coherence_matrix: Array2<Complex<f64>>,
    /// Entanglement connections to other neurons
    pub entanglement_partners: Vec<(usize, f64)>,
    /// Quantum memory traces
    pub quantum_memory: VecDeque<Array1<Complex<f64>>>,
    /// Attention gate activation
    pub attention_gate: f64,
}

impl Default for QuantumSpikingNeuron {
    fn default() -> Self {
        let numstates = 4; // |ground⟩, |excited⟩, |superposition⟩, |entangled⟩
        Self {
            classical_neuron: SpikingNeuron::default(),
            quantum_amplitudes: Array1::from_elem(numstates, Complex::new(0.5, 0.0)),
            coherence_matrix: Array2::from_elem((numstates, numstates), Complex::new(0.0, 0.0)),
            entanglement_partners: Vec::new(),
            quantum_memory: VecDeque::new(),
            attention_gate: 0.0,
        }
    }
}

/// Configuration for consciousness-inspired processing
#[derive(Debug, Clone)]
pub struct ConsciousnessConfig {
    /// Global workspace broadcast threshold
    pub broadcast_threshold: f64,
    /// Attention schema strength
    pub attention_schema_strength: f64,
    /// Temporal binding window size (time steps)
    pub temporal_binding_window: usize,
    /// Meta-cognitive monitoring sensitivity
    pub metacognitive_sensitivity: f64,
    /// Integrated information complexity parameter
    pub phi_complexity_factor: f64,
    /// Predictive coding precision weights
    pub precision_weights: Array1<f64>,
}

impl Default for ConsciousnessConfig {
    fn default() -> Self {
        Self {
            broadcast_threshold: 0.6,
            attention_schema_strength: 0.8,
            temporal_binding_window: 40,
            metacognitive_sensitivity: 0.3,
            phi_complexity_factor: 2.0,
            precision_weights: Array1::from_vec(vec![1.0, 0.8, 0.6, 0.4]),
        }
    }
}

/// Quantum Interface Configuration
#[derive(Debug, Clone)]
pub struct QuantumInterfaceConfig {
    /// State preparation method
    pub state_preparation: String,
    /// Measurement strategy
    pub measurement_strategy: String,
    /// Decoherence mitigation
    pub decoherence_mitigation: bool,
}

/// Classical Interface Configuration
#[derive(Debug, Clone)]
pub struct ClassicalInterfaceConfig {
    /// Data format
    pub data_format: String,
    /// Precision level
    pub precision: usize,
    /// Buffer size
    pub buffer_size: usize,
}

/// Quantum-Classical Hybrid Configuration
#[derive(Debug, Clone)]
pub struct QuantumClassicalHybridConfig {
    /// Quantum processing weight
    pub quantum_weight: f64,
    /// Classical processing weight
    pub classical_weight: f64,
    /// Error correction enabled
    pub error_correction: bool,
    /// Performance optimization enabled
    pub performance_optimization: bool,
    /// Adaptive algorithm selection
    pub adaptive_selection: bool,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
}

impl Default for QuantumClassicalHybridConfig {
    fn default() -> Self {
        Self {
            quantum_weight: 0.6,
            classical_weight: 0.4,
            error_correction: true,
            performance_optimization: true,
            adaptive_selection: true,
            resource_constraints: ResourceConstraints::default(),
        }
    }
}

/// Resource Constraints
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum quantum resources
    pub max_quantum_resources: QuantumResourceReq,
    /// Maximum classical resources
    pub max_classical_resources: ClassicalResourceReq,
    /// Maximum processing time
    pub max_processing_time: f64,
    /// Maximum energy consumption
    pub max_energy: f64,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_quantum_resources: QuantumResourceReq {
                qubits: 100,
                gates: 10000,
                coherence_time: 100.0,
                fidelity: 0.99,
            },
            max_classical_resources: ClassicalResourceReq {
                cpu_cores: 8,
                memory_mb: 8192,
                storage_mb: 1024,
                bandwidth_mbps: 100.0,
            },
            max_processing_time: 60.0,
            max_energy: 1000.0,
        }
    }
}

/// Quantum Resource Requirements
#[derive(Debug, Clone)]
pub struct QuantumResourceReq {
    /// Number of qubits
    pub qubits: usize,
    /// Gate count
    pub gates: usize,
    /// Coherence time required
    pub coherence_time: f64,
    /// Fidelity requirements
    pub fidelity: f64,
}

/// Classical Resource Requirements
#[derive(Debug, Clone)]
pub struct ClassicalResourceReq {
    /// CPU cores
    pub cpu_cores: usize,
    /// Memory (in MB)
    pub memory_mb: usize,
    /// Storage (in MB)
    pub storage_mb: usize,
    /// Network bandwidth (in Mbps)
    pub bandwidth_mbps: f64,
}

/// Data Conversion Pipeline
#[derive(Debug, Clone)]
pub struct DataConversionPipeline {
    /// Pipeline ID
    pub id: String,
    /// Conversion stages
    pub stages: Vec<ConversionStage>,
    /// Error handling strategy
    pub error_handling: ErrorHandlingStrategy,
    /// Performance metrics
    pub metrics: ConversionMetrics,
}

/// Conversion Stage
#[derive(Debug, Clone)]
pub struct ConversionStage {
    /// Stage name
    pub name: String,
    /// Conversion function
    pub function_type: ConversionFunction,
    /// Input format
    pub input_format: DataFormat,
    /// Output format
    pub output_format: DataFormat,
}

/// Error Handling Strategy
#[derive(Debug, Clone)]
pub enum ErrorHandlingStrategy {
    Retry { max_attempts: usize },
    Fallback { fallback_method: String },
    Graceful { degradation_factor: f64 },
    Abort,
}

/// Conversion Metrics
#[derive(Debug, Clone)]
pub struct ConversionMetrics {
    /// Conversion accuracy
    pub accuracy: f64,
    /// Processing time
    pub processing_time: f64,
    /// Resource usage
    pub resource_usage: f64,
    /// Error rate
    pub error_rate: f64,
}

/// Conversion Function Types
#[derive(Debug, Clone)]
pub enum ConversionFunction {
    QuantumToClassical { method: String },
    ClassicalToQuantum { encoding: String },
    QuantumToQuantum { transformation: String },
    ClassicalToClassical { preprocessing: String },
}

/// Data Format Types
#[derive(Debug, Clone)]
pub enum DataFormat {
    QuantumState {
        dimensions: usize,
    },
    ClassicalArray {
        dtype: String,
        shape: Vec<usize>,
    },
    CompressedQuantum {
        compression_ratio: f64,
    },
    HybridRepresentation {
        quantum_part: f64,
        classical_part: f64,
    },
}
