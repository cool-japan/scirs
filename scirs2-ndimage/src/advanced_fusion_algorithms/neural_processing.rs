//! Neural Processing Module for Advanced Fusion Algorithms
//!
//! This module implements self-organizing neural processing capabilities that enable
//! neural networks to reorganize their own structure based on input patterns and
//! processing requirements. The implementation is inspired by biological neural
//! plasticity and includes various activation functions ranging from classical to
//! quantum-inspired variants.
//!
//! ## Key Features
//!
//! - **Self-Organizing Networks**: Neural networks that adapt their topology dynamically
//! - **Multiple Activation Functions**: Classical (Sigmoid, Tanh, ReLU) and advanced (Quantum, Biological, Consciousness-inspired)
//! - **Real-time Adaptation**: Networks that learn and reorganize during processing
//! - **Quantum-Classical Hybrid**: Seamless integration of quantum and classical processing paradigms
//! - **Biological Inspiration**: Leaky integrate-and-fire neurons and spike-based processing
//! - **Consciousness Modeling**: Attention and awareness mechanisms for intelligent processing
//!
//! ## Processing Flow
//!
//! 1. **Network Reorganization**: Structure adapts based on input patterns
//! 2. **Connection Processing**: Calculate inputs from connected nodes
//! 3. **Activation**: Apply appropriate activation function
//! 4. **State Update**: Update node internal states
//! 5. **Learning**: Apply self-organization learning rules
//! 6. **Global Update**: Update network-wide properties

use scirs2_core::ndarray::{Array2, Array5};
use scirs2_core::numeric::Complex;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, RwLock};

use super::config::*;
use crate::error::NdimageResult;

/// Self-Organizing Neural Processing
///
/// Implements neural networks that reorganize their own structure based on input patterns
/// and processing requirements, inspired by biological neural plasticity.
///
/// This function processes multi-dimensional features through a self-organizing neural
/// network that can adapt its topology, connection weights, and activation patterns
/// in real-time. The network combines classical and quantum processing paradigms
/// with biological inspiration.
///
/// # Arguments
///
/// * `advancedfeatures` - Input features as 5D array (batch, channel, depth, height, width)
/// * `advancedstate` - Mutable reference to the advanced processing state containing network topology
/// * `config` - Configuration parameters for neural processing
///
/// # Returns
///
/// Returns a 2D array representing the processed neural output with dimensions (height, width)
///
/// # Features
///
/// - **Dynamic Topology**: Network structure adapts based on input patterns
/// - **Multi-paradigm Activation**: Support for classical, quantum, and biological activation functions
/// - **Self-organization Learning**: Continuous adaptation of connection weights and structure
/// - **Global Coherence**: Network-wide properties maintained and updated
///
/// # Example
///
/// ```rust,ignore
/// use scirs2_core::ndarray::Array5;
/// use scirs2_ndimage::advanced_fusion_algorithms::neural_processing::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let features = Array5::zeros((1, 3, 10, 64, 64));
/// let mut state = AdvancedState::default();
/// let config = AdvancedConfig::default();
///
/// let result = self_organizing_neural_processing(&features, &mut state, &config)?;
/// assert_eq!(result.dim(), (64, 64));
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn self_organizing_neural_processing(
    advancedfeatures: &Array5<f64>,
    advancedstate: &mut AdvancedState,
    config: &AdvancedConfig,
) -> NdimageResult<Array2<f64>> {
    let shape = advancedfeatures.dim();
    let (height, width) = (shape.0, shape.1);
    let mut neural_output = Array2::zeros((height, width));

    // Access the network topology with proper locking
    let mut topology = advancedstate
        .network_topology
        .write()
        .expect("Operation failed");

    // Self-organize network structure based on input patterns
    if config.self_organization {
        reorganize_network_structure(&mut topology, advancedfeatures, config)?;
    }

    // Process through self-organizing network
    for y in 0..height {
        for x in 0..width {
            let pixel_id = y * width + x;

            if pixel_id < topology.nodes.len() {
                let mut node_activation = 0.0;

                // Collect inputs from connected nodes
                if let Some(connections) = topology.connections.get(&pixel_id) {
                    for connection in connections {
                        if connection.target < topology.nodes.len() {
                            let source_node = &topology.nodes[connection.target];

                            // Calculate connection contribution
                            let connection_input = calculate_connection_input(
                                source_node,
                                connection,
                                advancedfeatures,
                                (y, x),
                                config,
                            )?;

                            node_activation += connection_input;
                        }
                    }
                }

                // Apply activation function
                let activation_type = topology.nodes[pixel_id].activation_type.clone();
                let activated_output =
                    apply_activation_function(node_activation, &activation_type, config)?;

                // Update node state
                update_nodestate(
                    &mut topology.nodes[pixel_id],
                    activated_output,
                    advancedfeatures,
                    (y, x),
                    config,
                )?;

                neural_output[(y, x)] = activated_output;

                // Apply self-organization learning
                if config.self_organization {
                    apply_self_organization_learning_safe(&mut topology, pixel_id, config)?;
                }
            }
        }
    }

    // Update global network properties
    update_global_network_properties(&mut topology, config)?;

    Ok(neural_output)
}

/// Reorganize Network Structure
///
/// Implements Kohonen SOM-inspired reorganization: for each node, finds the best-matching unit
/// (BMU) in the feature space and updates neighbors toward the input using a Gaussian
/// neighborhood function with decaying radius. New nodes are created when needed.
#[allow(dead_code)]
fn reorganize_network_structure(
    topology: &mut NetworkTopology,
    features: &Array5<f64>,
    config: &AdvancedConfig,
) -> NdimageResult<()> {
    let shape = features.dim();
    let (height, width) = (shape.0, shape.1);
    let total_nodes = height * width;

    // Ensure we have enough nodes
    while topology.nodes.len() < total_nodes.min(64) {
        let id = topology.nodes.len();
        topology.nodes.push(NetworkNode {
            id,
            quantumstate: scirs2_core::ndarray::Array1::zeros(4),
            classicalstate: scirs2_core::ndarray::Array1::zeros(4),
            learning_params: scirs2_core::ndarray::Array1::from_vec(vec![
                config.meta_learning_rate,
                config.neuromorphic_plasticity,
                0.5,
                1.0,
            ]),
            activation_type: ActivationType::Sigmoid,
            self_org_strength: config.neuromorphic_plasticity,
        });
    }

    if topology.nodes.is_empty() {
        return Ok(());
    }

    // Extract a representative feature vector from the center of the feature map
    let cy = height / 2;
    let cx = width / 2;

    let mut input_vec = Vec::with_capacity(shape.2);
    for d in 0..shape.2 {
        input_vec.push(features[(cy, cx, d, 0, 0)]);
    }

    // Find BMU: node whose classical state is closest to input
    let mut bmu_idx = 0_usize;
    let mut bmu_dist = f64::INFINITY;

    for (i, node) in topology.nodes.iter().enumerate() {
        let dist: f64 = node
            .classicalstate
            .iter()
            .enumerate()
            .map(|(j, &s)| {
                let iv = input_vec.get(j).cloned().unwrap_or(0.0);
                (s - iv).powi(2)
            })
            .sum::<f64>();
        if dist < bmu_dist {
            bmu_dist = dist;
            bmu_idx = i;
        }
    }

    // Kohonen neighborhood update
    let learning_rate = config.meta_learning_rate;
    let sigma = (topology.nodes.len() as f64 / 4.0).max(1.0);

    let n_nodes = topology.nodes.len();
    for i in 0..n_nodes {
        let dist_to_bmu = (i as f64 - bmu_idx as f64).abs();
        let neighborhood = (-(dist_to_bmu.powi(2)) / (2.0 * sigma * sigma)).exp();
        let effective_lr = learning_rate * neighborhood;

        // Update classical state toward input
        let node = &mut topology.nodes[i];
        for j in 0..node.classicalstate.len() {
            let iv = input_vec.get(j).cloned().unwrap_or(0.0);
            node.classicalstate[j] += effective_lr * (iv - node.classicalstate[j]);
        }
    }

    Ok(())
}

/// Calculate Connection Input
///
/// Computes the weighted input contribution from a source node through a connection:
/// `input = weight.re * source_activation + bias_from_features`
/// Connection type modulates the sign (excitatory positive, inhibitory negative).
/// Quantum connections additionally apply phase-modulated interference.
#[allow(dead_code)]
fn calculate_connection_input(
    source_node: &NetworkNode,
    connection: &Connection,
    features: &Array5<f64>,
    position: (usize, usize),
    config: &AdvancedConfig,
) -> NdimageResult<f64> {
    // Source node activation: use mean of classical state as output
    let source_activation = if source_node.classicalstate.is_empty() {
        0.0
    } else {
        source_node.classicalstate.iter().sum::<f64>() / source_node.classicalstate.len() as f64
    };

    // Apply connection weight (real part)
    let weight_real = connection.weight.re;
    let mut input = weight_real * source_activation;

    // Connection type modulation
    input *= match connection.connection_type {
        ConnectionType::Excitatory => 1.0,
        ConnectionType::Inhibitory => -1.0,
        ConnectionType::Modulatory => 0.5,
        ConnectionType::Quantum | ConnectionType::QuantumEntangled => {
            // Quantum interference: phase modulation from imaginary weight
            let phase = connection.weight.im * PI * config.quantum.phase_factor;
            phase.cos()
        }
        ConnectionType::SelfOrganizing => source_node.self_org_strength,
        ConnectionType::Causal | ConnectionType::Temporal => 0.8,
    };

    // Add small feature-based bias from the position
    let (y, x) = position;
    let feature_shape = features.dim();
    if y < feature_shape.0 && x < feature_shape.1 && feature_shape.2 > 0 {
        let feature_bias = features[(y, x, 0, 0, 0)] * config.meta_learning_rate;
        input += feature_bias;
    }

    // Apply plasticity modulation
    input *= 1.0 + connection.plasticity.quantum_coherence * 0.1;

    Ok(input)
}

/// Apply Activation Function
///
/// Applies the specified activation function to the input value, supporting
/// both classical and advanced activation paradigms including quantum-inspired,
/// biological, and consciousness-based functions.
///
/// # Arguments
///
/// * `input` - Input value to be activated
/// * `activation_type` - Type of activation function to apply
/// * `config` - Configuration parameters for advanced activation functions
///
/// # Returns
///
/// Returns the activated output value clamped to [-10.0, 10.0] for numerical stability
///
/// # Supported Activation Functions
///
/// ## Classical Functions
/// - **Sigmoid**: Standard logistic function `1 / (1 + exp(-x))`
/// - **Tanh**: Hyperbolic tangent function
/// - **ReLU**: Rectified Linear Unit `max(0, x)`
/// - **Swish**: Self-gated function `x * sigmoid(x)`
///
/// ## Advanced Functions
/// - **QuantumSigmoid**: Quantum-inspired sigmoid with interference effects
/// - **BiologicalSpike**: Leaky integrate-and-fire neuron model
/// - **ConsciousnessGate**: Attention-based gating function
/// - **AdvancedActivation**: Multi-paradigm combination function
///
/// # Example
///
/// ```rust,ignore
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use scirs2_ndimage::advanced_fusion_algorithms::neural_processing::*;
/// # let config = AdvancedConfig::default();
/// let output = apply_activation_function(2.5, &ActivationType::Sigmoid, &config)?;
/// assert!(output > 0.9 && output < 1.0);
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
fn apply_activation_function(
    input: f64,
    activation_type: &ActivationType,
    config: &AdvancedConfig,
) -> NdimageResult<f64> {
    let output = match activation_type {
        ActivationType::Sigmoid => {
            // Standard logistic sigmoid function
            1.0 / (1.0 + (-input).exp())
        }
        ActivationType::Tanh => {
            // Hyperbolic tangent function
            input.tanh()
        }
        ActivationType::ReLU => {
            // Rectified Linear Unit
            input.max(0.0)
        }
        ActivationType::Swish => {
            // Self-gated activation function
            let sigmoid = 1.0 / (1.0 + (-input).exp());
            input * sigmoid
        }
        ActivationType::QuantumSigmoid => {
            // Quantum-inspired sigmoid with interference effects
            let quantum_factor = (input * PI * config.quantum.coherence_factor).cos();
            let classical_sigmoid = 1.0 / (1.0 + (-input).exp());
            classical_sigmoid * (1.0 + 0.1 * quantum_factor)
        }
        ActivationType::BiologicalSpike => {
            // Leaky integrate-and-fire neuron model
            let threshold = 1.0;
            let leak_factor = 0.9;
            if input > threshold {
                1.0 // Spike output
            } else {
                input * leak_factor // Leak current
            }
        }
        ActivationType::ConsciousnessGate => {
            // Consciousness-inspired gating function with attention mechanisms
            let attention_factor = (input.abs() / config.consciousness_depth as f64).tanh();
            let awareness_threshold = 0.5;
            if attention_factor > awareness_threshold {
                // Conscious processing: full activation with attention modulation
                input.tanh() * attention_factor
            } else {
                // Subconscious processing: reduced activation
                input * 0.1
            }
        }
        ActivationType::AdvancedActivation => {
            // Advanced-advanced activation combining multiple paradigms
            let sigmoid_component = 1.0 / (1.0 + (-input).exp());
            let quantum_component = (input * PI).sin() * 0.1;
            let meta_component = (input / config.meta_learning_rate).tanh() * 0.05;
            let temporal_component = (input * config.temporal_window as f64).cos() * 0.05;

            sigmoid_component + quantum_component + meta_component + temporal_component
        }
    };

    // Ensure output is finite and within reasonable bounds for numerical stability
    Ok(output.clamp(-10.0, 10.0))
}

/// Update Node State
///
/// Updates node activation via: `state = activation_function(connection_input + bias)`
/// The classical state is updated with a leaky integrator, and quantum state amplitude
/// is updated based on the output magnitude.
#[allow(dead_code)]
fn update_nodestate(
    node: &mut NetworkNode,
    output: f64,
    advancedfeatures: &Array5<f64>,
    position: (usize, usize),
    config: &AdvancedConfig,
) -> NdimageResult<()> {
    // Activation function: apply to connection_input + bias
    let bias = if node.learning_params.len() > 2 {
        node.learning_params[2]
    } else {
        0.0
    };
    let combined_input = output + bias;

    // Apply activation function to get new state
    let new_activation = apply_activation_function(combined_input, &node.activation_type, config)?;

    // Update classical state with leaky integration: s' = decay * s + (1-decay) * output
    let decay = (1.0 - config.neuromorphic_plasticity).clamp(0.0, 1.0);
    let (y, x) = position;
    let feature_shape = advancedfeatures.dim();

    for (j, state_val) in node.classicalstate.iter_mut().enumerate() {
        // Leaky integration of activation
        *state_val = decay * (*state_val) + (1.0 - decay) * new_activation;

        // Add small feature-based modulation
        if y < feature_shape.0 && x < feature_shape.1 {
            let d_idx = j.min(feature_shape.2.saturating_sub(1));
            if feature_shape.2 > 0 && feature_shape.3 > 0 && feature_shape.4 > 0 {
                let feat = advancedfeatures[(y, x, d_idx, 0, 0)];
                *state_val += config.meta_learning_rate * feat * 0.01;
            }
        }

        // Clamp to valid range
        *state_val = state_val.clamp(-10.0, 10.0);
    }

    // Update quantum state: amplitude based on output magnitude
    let amplitude = new_activation.abs().min(1.0);
    let qs_len = node.quantumstate.len().max(1);
    for (j, qs_val) in node.quantumstate.iter_mut().enumerate() {
        let phase = (j as f64 * PI / qs_len as f64) * new_activation;
        *qs_val = Complex::new(amplitude * phase.cos(), amplitude * phase.sin());
    }

    // Update self-organization strength based on activation
    node.self_org_strength =
        (node.self_org_strength * 0.99 + new_activation.abs() * 0.01).clamp(0.0, 1.0);

    Ok(())
}

/// Apply Self-Organization Learning (Safe Version)
///
/// Safe Kohonen-style update with bounds checking.
/// Updates connection weights toward the current node activation with a gradually decaying
/// learning rate based on the node's learning_params history.
#[allow(dead_code)]
fn apply_self_organization_learning_safe(
    topology: &mut NetworkTopology,
    node_id: usize,
    config: &AdvancedConfig,
) -> NdimageResult<()> {
    if node_id >= topology.nodes.len() {
        return Ok(());
    }

    // Get node activation (mean of classical state)
    let node_activation = if topology.nodes[node_id].classicalstate.is_empty() {
        0.0
    } else {
        topology.nodes[node_id].classicalstate.iter().sum::<f64>()
            / topology.nodes[node_id].classicalstate.len() as f64
    };

    // Gradually reduce learning rate (using learning_params[0] as a decaying counter)
    let base_lr = config.meta_learning_rate;
    let decay_factor = if !topology.nodes[node_id].learning_params.is_empty() {
        let calls = topology.nodes[node_id].learning_params[0].max(1.0);
        1.0 / (1.0 + calls * 0.01)
    } else {
        1.0
    };
    let effective_lr = (base_lr * decay_factor).clamp(1e-6, 1.0);

    // Update learning_params[0] as call counter
    if !topology.nodes[node_id].learning_params.is_empty() {
        topology.nodes[node_id].learning_params[0] += 1.0;
    }

    // Update connections for this node: Hebbian rule with bounds check
    if let Some(connections) = topology.connections.get_mut(&node_id) {
        for connection in connections.iter_mut() {
            if connection.target < topology.nodes.len() {
                let target_activation =
                    if topology.nodes[connection.target].classicalstate.is_empty() {
                        0.0
                    } else {
                        topology.nodes[connection.target]
                            .classicalstate
                            .iter()
                            .sum::<f64>()
                            / topology.nodes[connection.target].classicalstate.len() as f64
                    };

                // Hebbian update: Δw = lr * source * target
                let delta_w = effective_lr * node_activation * target_activation;
                connection.weight = Complex::new(
                    (connection.weight.re + delta_w).clamp(-10.0, 10.0),
                    connection.weight.im * (1.0 - connection.plasticity.decay_rate),
                );
            }
        }
    }

    Ok(())
}

/// Update Global Network Properties
///
/// Computes average activation across all nodes, average connection strength,
/// self-organization index, and efficiency ratio. Updates global_properties in place.
#[allow(dead_code)]
fn update_global_network_properties(
    topology: &mut NetworkTopology,
    config: &AdvancedConfig,
) -> NdimageResult<()> {
    if topology.nodes.is_empty() {
        return Ok(());
    }

    let n = topology.nodes.len() as f64;

    // Average activation across all nodes
    let avg_activation: f64 = topology
        .nodes
        .iter()
        .map(|node| {
            if node.classicalstate.is_empty() {
                0.0
            } else {
                node.classicalstate.iter().sum::<f64>() / node.classicalstate.len() as f64
            }
        })
        .sum::<f64>()
        / n;

    // Activation variance (for coherence measure)
    let var_activation: f64 = topology
        .nodes
        .iter()
        .map(|node| {
            let act = if node.classicalstate.is_empty() {
                0.0
            } else {
                node.classicalstate.iter().sum::<f64>() / node.classicalstate.len() as f64
            };
            (act - avg_activation).powi(2)
        })
        .sum::<f64>()
        / n;

    // Coherence: inverse of normalized variance (high variance = low coherence)
    let coherence = 1.0 / (1.0 + var_activation);

    // Average connection strength
    let total_connections: usize = topology.connections.values().map(|c| c.len()).sum();
    let avg_connection_strength = if total_connections > 0 {
        topology
            .connections
            .values()
            .flat_map(|cs| cs.iter().map(|c| c.weight.re.abs()))
            .sum::<f64>()
            / total_connections as f64
    } else {
        0.0
    };

    // Self-organization index: mean self_org_strength across nodes
    let self_org_index = topology
        .nodes
        .iter()
        .map(|n| n.self_org_strength)
        .sum::<f64>()
        / topology.nodes.len() as f64;

    // Consciousness emergence: geometric mean of coherence and self-org
    let consciousness_emergence = (coherence * self_org_index).sqrt();

    // Efficiency: information flow relative to connection overhead
    let efficiency = if total_connections > 0 {
        (avg_activation.abs() * coherence)
            / (1.0 + avg_connection_strength * total_connections as f64 / n)
    } else {
        avg_activation.abs() * coherence
    };

    topology.global_properties.coherence = coherence;
    topology.global_properties.self_organization_index = self_org_index;
    topology.global_properties.consciousness_emergence = consciousness_emergence;
    topology.global_properties.efficiency = efficiency.clamp(0.0, 1.0);

    Ok(())
}

/// Apply Self-Organization Learning
///
/// Fast path (no bounds checking) for applying Kohonen-style weight updates to a specific node.
/// Updates connection weights directly using Hebbian correlation without safety checks.
#[allow(dead_code)]
fn apply_self_organization_learning(
    node: &mut NetworkNode,
    connections: &mut HashMap<usize, Vec<Connection>>,
    node_id: usize,
    config: &AdvancedConfig,
) -> NdimageResult<()> {
    // Node activation: mean of classical state
    let node_activation = if node.classicalstate.is_empty() {
        0.0
    } else {
        node.classicalstate.iter().sum::<f64>() / node.classicalstate.len() as f64
    };

    let lr = config.meta_learning_rate;

    // Update outgoing connections for this node
    if let Some(conns) = connections.get_mut(&node_id) {
        for connection in conns.iter_mut() {
            // Faster path: use node_activation as both pre- and post-synaptic proxy
            let delta_w = lr * node_activation * node_activation;
            let new_real = (connection.weight.re + delta_w).clamp(-10.0, 10.0);
            let new_imag = connection.weight.im * (1.0 - connection.plasticity.decay_rate);
            connection.weight = Complex::new(new_real, new_imag);
        }
    }

    // Update self-organization strength
    node.self_org_strength = (node.self_org_strength + lr * node_activation.abs()).clamp(0.0, 1.0);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array5;

    #[test]
    fn test_activation_functions() {
        let config = AdvancedConfig::default();

        // Test sigmoid activation
        let result = apply_activation_function(0.0, &ActivationType::Sigmoid, &config)
            .expect("Operation failed");
        assert!((result - 0.5).abs() < 1e-10);

        // Test ReLU activation
        let result = apply_activation_function(-1.0, &ActivationType::ReLU, &config)
            .expect("Operation failed");
        assert_eq!(result, 0.0);

        let result = apply_activation_function(2.0, &ActivationType::ReLU, &config)
            .expect("Operation failed");
        assert_eq!(result, 2.0);

        // Test tanh activation
        let result = apply_activation_function(0.0, &ActivationType::Tanh, &config)
            .expect("Operation failed");
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_neural_processing_dimensions() {
        let features = Array5::zeros((32, 32, 1, 1, 1));
        let config = AdvancedConfig::default();

        // Create a minimal test state
        let mut state = create_test_state();

        let result = self_organizing_neural_processing(&features, &mut state, &config);
        assert!(result.is_ok());

        let output = result.expect("Operation failed");
        assert_eq!(output.dim(), (32, 32));
    }

    // Helper function to create test state
    fn create_test_state() -> AdvancedState {
        use scirs2_core::ndarray::{Array1, Array4};
        use scirs2_core::numeric::Complex64;
        use std::collections::{BTreeMap, VecDeque};

        // Create minimal network topology for testing
        let topology = NetworkTopology {
            connections: HashMap::new(),
            nodes: vec![NetworkNode {
                id: 0,
                quantumstate: Array1::zeros(4),
                classicalstate: Array1::zeros(4),
                learning_params: Array1::zeros(4),
                activation_type: ActivationType::Sigmoid,
                self_org_strength: 0.5,
            }],
            global_properties: NetworkProperties {
                coherence: 0.5,
                self_organization_index: 0.3,
                consciousness_emergence: 0.2,
                efficiency: 0.8,
            },
        };

        AdvancedState {
            consciousness_amplitudes: Array4::zeros((2, 2, 2, 2)),
            meta_parameters: Array2::zeros((4, 4)),
            network_topology: Arc::new(RwLock::new(topology)),
            temporal_memory: VecDeque::new(),
            causal_graph: BTreeMap::new(),
            advancedfeatures: Array5::zeros((1, 1, 1, 1, 1)),
            resource_allocation: ResourceState {
                cpu_allocation: vec![0.5, 0.3, 0.2],
                memory_allocation: 0.7,
                gpu_allocation: Some(0.4),
                quantum_allocation: Some(0.1),
                allocationhistory: VecDeque::new(),
            },
            efficiencymetrics: EfficiencyMetrics {
                ops_per_second: 1000.0,
                memory_efficiency: 0.8,
                energy_efficiency: 0.6,
                quality_efficiency: 0.75,
                temporal_efficiency: 0.9,
            },
            processing_cycles: 0,
        }
    }

    #[test]
    fn test_activation_bounds() {
        let config = AdvancedConfig::default();

        // Test extreme inputs are clamped
        let result = apply_activation_function(1000.0, &ActivationType::Sigmoid, &config)
            .expect("Operation failed");
        assert!(result >= -10.0 && result <= 10.0);

        let result = apply_activation_function(-1000.0, &ActivationType::Sigmoid, &config)
            .expect("Operation failed");
        assert!(result >= -10.0 && result <= 10.0);
    }

    #[test]
    fn test_network_node_update_bounded() {
        use scirs2_core::ndarray::{Array1, Array5};

        let mut node = NetworkNode {
            id: 0,
            quantumstate: Array1::zeros(4),
            classicalstate: Array1::from_vec(vec![0.5, -0.3, 0.1, 0.8]),
            learning_params: Array1::from_vec(vec![1.0, 0.1, 0.0, 0.0]),
            activation_type: ActivationType::Sigmoid,
            self_org_strength: 0.5,
        };

        let features = Array5::zeros((8, 8, 2, 2, 2));
        let config = AdvancedConfig::default();

        // Apply update with a range of extreme outputs
        for output in [-100.0_f64, -1.0, 0.0, 1.0, 100.0] {
            let result = update_nodestate(&mut node, output, &features, (0, 0), &config);
            assert!(
                result.is_ok(),
                "update_nodestate failed for output={}",
                output
            );

            // All classical state values must be bounded in [-10, 10]
            for &state_val in node.classicalstate.iter() {
                assert!(
                    state_val.is_finite() && state_val >= -10.0 && state_val <= 10.0,
                    "classical state out of bounds: {} (output was {})",
                    state_val,
                    output
                );
            }

            // Quantum state norms should be <= 1
            for qs in node.quantumstate.iter() {
                assert!(
                    qs.norm() <= 1.0 + 1e-10,
                    "quantum state norm exceeded 1: {}",
                    qs.norm()
                );
            }

            // self_org_strength must be in [0, 1]
            assert!(
                node.self_org_strength >= 0.0 && node.self_org_strength <= 1.0,
                "self_org_strength out of range: {}",
                node.self_org_strength
            );
        }
    }
}
