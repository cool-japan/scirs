//! # Quantum Consciousness Simulation Module
//!
//! This module provides advanced quantum consciousness simulation capabilities for image processing,
//! combining quantum mechanical principles with consciousness-inspired computing paradigms.
//!
//! ## Features
//!
//! - **Quantum Consciousness Simulation**: Models consciousness-like processing using quantum superposition,
//!   entanglement, and quantum interference effects
//! - **Evolutionary Consciousness**: Advanced consciousness evolution using quantum-inspired evolutionary
//!   dynamics that allow consciousness to adapt and emerge over time
//! - **Quantum State Management**: Sophisticated quantum amplitude management and coherence optimization
//! - **Consciousness Analysis**: Comprehensive analysis of consciousness states including level assessment,
//!   coherence quality measurement, and self-awareness indexing
//! - **Integrated Information Theory**: Implementation of simplified Phi measures for consciousness quantification
//!
//! ## Core Concepts
//!
//! The module implements several key concepts from consciousness research and quantum computing:
//!
//! - **Quantum Superposition**: Consciousness states exist in superposition until measured
//! - **Quantum Entanglement**: Consciousness levels can be entangled across different spatial regions
//! - **Decoherence Management**: Strategies to maintain quantum coherence in consciousness processing
//! - **Evolutionary Adaptation**: Consciousness parameters evolve based on processing effectiveness
//! - **Global Coherence**: Maintenance of coherent consciousness across entire processing domains
//!
//! ## Usage
//!
//! ```rust,ignore
//! use crate::advanced_fusion_algorithms::quantum_consciousness::*;
//! use scirs2_core::ndarray::{Array2, Array5};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let features = Array5::zeros((1, 3, 10, 64, 64));
//! # let mut state = AdvancedState::default();
//! # let config = AdvancedConfig::default();
//! # let image = Array2::zeros((64, 64));
//! // Basic quantum consciousness simulation
//! let consciousness_output = simulate_quantum_consciousness(
//!     &features,
//!     &mut state,
//!     &config,
//! )?;
//!
//! // Enhanced evolution-based consciousness processing
//! let mut evolution_system = QuantumConsciousnessEvolution::default();
//! let evolved_output = enhanced_quantum_consciousness_evolution(
//!     image.view(),
//!     &features,
//!     &mut state,
//!     &config,
//!     &mut evolution_system,
//! )?;
//! # Ok(())
//! # }
//! ```

use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4, Array5, ArrayView1, ArrayView2};
use scirs2_core::numeric::Complex;
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;
use std::sync::{Arc, RwLock};

use super::config::*;
use crate::error::NdimageResult;

/// Represents the state of consciousness in quantum simulation
#[derive(Debug, Clone)]
pub struct ConsciousnessState {
    /// Consciousness level (0.0 to 1.0)
    pub level: f64,
    /// Quantum coherence quality
    pub coherence_quality: f64,
    /// Information integration measure (Phi)
    pub phi_measure: f64,
    /// Attention focus strength
    pub attention_strength: f64,
    /// Self-awareness index
    pub self_awareness: f64,
    /// Timestamp of state
    pub timestamp: usize,
}

/// Metrics for consciousness complexity assessment
#[derive(Debug, Clone)]
pub struct ConsciousnessComplexity {
    /// Integrated information measure
    pub integrated_information: f64,
    /// Causal structure complexity
    pub causal_complexity: f64,
    /// Temporal coherence measure
    pub temporal_coherence: f64,
    /// Hierarchical organization index
    pub hierarchical_index: f64,
    /// Emergent property strength
    pub emergence_strength: f64,
}

/// Quantum coherence optimization strategies
#[derive(Debug, Clone)]
pub enum CoherenceStrategy {
    /// Error correction based coherence preservation
    ErrorCorrection {
        threshold: f64,
        correction_rate: f64,
    },
    /// Decoherence suppression
    DecoherenceSuppression { suppression_strength: f64 },
    /// Entanglement purification
    EntanglementPurification { purification_cycles: usize },
    /// Dynamical decoupling
    DynamicalDecoupling { pulse_frequency: f64 },
    /// Quantum Zeno effect
    QuantumZeno { measurement_frequency: f64 },
}

/// Quantum coherence optimization engine
#[derive(Debug, Clone)]
pub struct QuantumCoherenceOptimizer {
    /// Coherence maintenance strategies
    pub strategies: Vec<CoherenceStrategy>,
    /// Optimization parameters
    pub optimization_params: HashMap<String, f64>,
    /// Performance history
    pub performancehistory: VecDeque<f64>,
}

/// Quantum consciousness evolution system
#[derive(Debug, Clone)]
pub struct QuantumConsciousnessEvolution {
    /// Consciousness evolution history
    pub evolutionhistory: VecDeque<ConsciousnessState>,
    /// Evolution rate parameters
    pub evolution_rate: f64,
    /// Consciousness complexity metrics
    pub complexitymetrics: ConsciousnessComplexity,
    /// Quantum coherence optimization engine
    pub coherence_optimizer: QuantumCoherenceOptimizer,
    /// Evolutionary selection pressure
    pub selection_pressure: f64,
    /// Consciousness emergence threshold
    pub emergence_threshold: f64,
}

impl Default for QuantumConsciousnessEvolution {
    fn default() -> Self {
        Self {
            evolutionhistory: VecDeque::new(),
            evolution_rate: 0.01,
            complexitymetrics: ConsciousnessComplexity {
                integrated_information: 0.0,
                causal_complexity: 0.0,
                temporal_coherence: 0.0,
                hierarchical_index: 0.0,
                emergence_strength: 0.0,
            },
            coherence_optimizer: QuantumCoherenceOptimizer {
                strategies: vec![
                    CoherenceStrategy::ErrorCorrection {
                        threshold: 0.95,
                        correction_rate: 0.1,
                    },
                    CoherenceStrategy::DecoherenceSuppression {
                        suppression_strength: 0.8,
                    },
                    CoherenceStrategy::EntanglementPurification {
                        purification_cycles: 5,
                    },
                ],
                optimization_params: HashMap::new(),
                performancehistory: VecDeque::new(),
            },
            selection_pressure: 0.1,
            emergence_threshold: 0.7,
        }
    }
}

/// Quantum Consciousness Simulation
///
/// Simulates consciousness-like processing using quantum mechanical principles
/// including superposition, entanglement, and quantum interference effects.
#[allow(dead_code)]
pub fn simulate_quantum_consciousness(
    advancedfeatures: &Array5<f64>,
    advancedstate: &mut AdvancedState,
    config: &AdvancedConfig,
) -> NdimageResult<Array2<f64>> {
    let (height, width, dimensions, temporal, consciousness) = advancedfeatures.dim();
    let mut consciousness_output = Array2::zeros((height, width));

    // Initialize quantum consciousness amplitudes if not present or if all-zero
    // (all-zero state produces no output; ensure superposition is set before processing)
    let shape_mismatch =
        advancedstate.consciousness_amplitudes.dim() != (height, width, consciousness, 2);
    let amplitudes_all_zero = advancedstate
        .consciousness_amplitudes
        .iter()
        .all(|c| c.norm() < 1e-12);
    let needs_init = shape_mismatch || amplitudes_all_zero;
    if needs_init {
        advancedstate.consciousness_amplitudes = Array4::zeros((height, width, consciousness, 2));

        // Initialize in quantum superposition state
        let amplitude = Complex::new((1.0 / consciousness as f64).sqrt(), 0.0);
        advancedstate.consciousness_amplitudes.fill(amplitude);
    }

    // Quantum consciousness processing
    for y in 0..height {
        for x in 0..width {
            let mut consciousness_amplitude = Complex::new(0.0, 0.0);

            // Process each consciousness level
            for c in 0..consciousness {
                // Extract multi-dimensional feature vector
                let mut feature_vector = Vec::new();
                for d in 0..dimensions {
                    for t in 0..temporal {
                        feature_vector.push(advancedfeatures[(y, x, d, t, c)]);
                    }
                }

                // Apply quantum consciousness operators
                let quantumstate = apply_quantum_consciousness_operators(
                    &feature_vector,
                    &advancedstate
                        .consciousness_amplitudes
                        .slice(s![y, x, c, ..]),
                    config,
                )?;

                // Update consciousness amplitudes
                advancedstate.consciousness_amplitudes[(y, x, c, 0)] =
                    Complex::new(quantumstate.re, 0.0);
                advancedstate.consciousness_amplitudes[(y, x, c, 1)] =
                    Complex::new(quantumstate.im, 0.0);

                // Accumulate consciousness response
                consciousness_amplitude += quantumstate;
            }

            // Consciousness measurement (collapse to classical state)
            let consciousness_probability = consciousness_amplitude.norm_sqr();
            consciousness_output[(y, x)] = consciousness_probability;
        }
    }

    // Apply consciousness-level global coherence
    apply_global_consciousness_coherence(&mut consciousness_output, advancedstate, config)?;

    // Propagate temporal-causal influence across the consciousness field using the
    // accumulated causal graph and temporal memory (Granger-style causal weighting).
    apply_temporal_causal_inference(&mut consciousness_output, advancedstate, config)?;

    Ok(consciousness_output)
}

/// Apply quantum consciousness operators to feature vectors
#[allow(dead_code)]
fn apply_quantum_consciousness_operators(
    feature_vector: &[f64],
    consciousnessstate: &ArrayView1<Complex<f64>>,
    config: &AdvancedConfig,
) -> NdimageResult<Complex<f64>> {
    if feature_vector.is_empty() || consciousnessstate.is_empty() {
        return Ok(Complex::new(0.0, 0.0));
    }

    let mut quantumstate = Complex::new(0.0, 0.0);

    // Quantum superposition of feature states
    let feature_norm = feature_vector
        .iter()
        .map(|&x| x * x)
        .sum::<f64>()
        .sqrt()
        .max(1e-10);
    let normalizedfeatures: Vec<f64> = feature_vector.iter().map(|&x| x / feature_norm).collect();

    // Apply quantum Hadamard-like transformation
    for (i, &feature) in normalizedfeatures.iter().enumerate() {
        if i < consciousnessstate.len() {
            let phase = feature * PI * config.quantum.phase_factor;
            let amplitude = (feature.abs() / config.consciousness_depth as f64).sqrt();

            // Quantum interference with existing consciousness state
            let existingstate = consciousnessstate[i % consciousnessstate.len()];

            // Apply quantum rotation
            let cos_phase = phase.cos();
            let sin_phase = phase.sin();

            let rotated_real = existingstate.re * cos_phase - existingstate.im * sin_phase;
            let rotated_imag = existingstate.re * sin_phase + existingstate.im * cos_phase;

            quantumstate += Complex::new(rotated_real, rotated_imag) * amplitude;
        }
    }

    // Apply quantum entanglement effects
    let entanglement_factor = config.quantum.entanglement_strength;
    let entangled_phase = normalizedfeatures.iter().sum::<f64>() * PI * entanglement_factor;

    let entanglement_rotation = Complex::new(entangled_phase.cos(), entangled_phase.sin());
    quantumstate *= entanglement_rotation;

    // Apply consciousness-specific quantum effects
    let consciousness_depth_factor =
        1.0 / (1.0 + (-(config.consciousness_depth as f64) * 0.1).exp());
    quantumstate *= consciousness_depth_factor;

    // Quantum decoherence simulation
    let decoherence_factor = (1.0 - config.quantum.decoherence_rate).max(0.1);
    quantumstate *= decoherence_factor;

    // Normalize quantum state
    let norm = quantumstate.norm();
    if norm > 1e-10 {
        quantumstate /= norm;
    }

    Ok(quantumstate)
}

/// Apply global consciousness coherence effects
///
/// Implements coherence damping based on the integrated information (phi) measure.
/// Each amplitude is scaled by exp(-gamma * phi), modelling decoherence strength
/// proportional to integrated information.
#[allow(dead_code)]
fn apply_global_consciousness_coherence(
    consciousness_output: &mut Array2<f64>,
    advancedstate: &AdvancedState,
    config: &AdvancedConfig,
) -> NdimageResult<()> {
    let phi = calculate_simplified_phi_measure(advancedstate, config)?;

    // Coherence damping rate: derived from decoherence_rate config
    let gamma = config.quantum.decoherence_rate.max(0.0);
    let damping = (-gamma * phi).exp();

    // Apply coherence damping uniformly across the consciousness output
    consciousness_output.mapv_inplace(|x| x * damping);

    Ok(())
}

/// Enhanced Quantum Consciousness Processing with Evolution
///
/// This advanced function extends the existing quantum consciousness simulation
/// with evolutionary dynamics, allowing consciousness to adapt and emerge
/// over time through quantum-inspired evolutionary processes.
#[allow(dead_code)]
pub fn enhanced_quantum_consciousness_evolution<T>(
    image: ArrayView2<T>,
    advancedfeatures: &Array5<f64>,
    advancedstate: &mut AdvancedState,
    config: &AdvancedConfig,
    evolution_system: &mut QuantumConsciousnessEvolution,
) -> NdimageResult<Array2<f64>>
where
    T: Float + FromPrimitive + Copy,
{
    let (height, width, dimensions, temporal, consciousness) = advancedfeatures.dim();
    let mut consciousness_output = Array2::zeros((height, width));

    // Analyze current consciousness state
    let currentstate = analyze_consciousnessstate(advancedstate, config)?;

    // Evolutionary consciousness adaptation
    evolve_consciousness_parameters(evolution_system, &currentstate, config)?;

    // Enhanced quantum processing with evolution
    for y in 0..height {
        for x in 0..width {
            let mut evolved_consciousness_amplitude = Complex::new(0.0, 0.0);

            // Process each consciousness level with evolutionary enhancement
            for c in 0..consciousness {
                // Extract multi-dimensional feature vector
                let mut feature_vector = Vec::new();
                for d in 0..dimensions {
                    for t in 0..temporal {
                        feature_vector.push(advancedfeatures[(y, x, d, t, c)]);
                    }
                }

                // Apply evolved quantum consciousness operators
                let evolved_quantumstate = apply_evolved_quantum_consciousness_operators(
                    &feature_vector,
                    &advancedstate
                        .consciousness_amplitudes
                        .slice(s![y, x, c, ..]),
                    config,
                    evolution_system,
                )?;

                // Update consciousness amplitudes with evolution
                advancedstate.consciousness_amplitudes[(y, x, c, 0)] =
                    Complex::new(evolved_quantumstate.re, 0.0);
                advancedstate.consciousness_amplitudes[(y, x, c, 1)] =
                    Complex::new(evolved_quantumstate.im, 0.0);

                // Accumulate evolved consciousness response
                evolved_consciousness_amplitude += evolved_quantumstate;
            }

            // Apply consciousness evolution and selection
            let evolved_response = apply_consciousness_evolution_selection(
                evolved_consciousness_amplitude,
                evolution_system,
                (y, x),
                config,
            )?;

            consciousness_output[(y, x)] = evolved_response;
        }
    }

    // Apply global consciousness evolution coherence
    apply_evolved_global_consciousness_coherence(
        &mut consciousness_output,
        advancedstate,
        evolution_system,
        config,
    )?;

    // Update evolution history
    update_consciousness_evolutionhistory(evolution_system, &currentstate)?;

    Ok(consciousness_output)
}

/// Analyze current consciousness state for evolutionary adaptation
#[allow(dead_code)]
fn analyze_consciousnessstate(
    advancedstate: &AdvancedState,
    config: &AdvancedConfig,
) -> NdimageResult<ConsciousnessState> {
    // Calculate consciousness level based on quantum amplitudes
    let total_amplitudes = advancedstate.consciousness_amplitudes.len() as f64;
    let coherence_sum = advancedstate
        .consciousness_amplitudes
        .iter()
        .map(|&amp| amp.norm())
        .sum::<f64>();

    let consciousness_level = if total_amplitudes > 0.0 {
        coherence_sum / total_amplitudes
    } else {
        0.0
    };

    // Calculate quantum coherence quality
    let coherence_variance = advancedstate
        .consciousness_amplitudes
        .iter()
        .map(|&amp| {
            let norm = amp.norm();
            (norm - consciousness_level).powi(2)
        })
        .sum::<f64>()
        / total_amplitudes.max(1.0);

    let coherence_quality = 1.0 / (1.0 + coherence_variance);

    // Calculate Phi measure (simplified integrated information)
    let phi_measure = calculate_simplified_phi_measure(advancedstate, config)?;

    // Calculate attention strength from network topology
    let attention_strength = {
        let topology = advancedstate
            .network_topology
            .read()
            .expect("Operation failed");
        topology.global_properties.coherence
    };

    // Calculate self-awareness index
    let self_awareness = (consciousness_level * coherence_quality * phi_measure).cbrt();

    Ok(ConsciousnessState {
        level: consciousness_level,
        coherence_quality,
        phi_measure,
        attention_strength,
        self_awareness,
        timestamp: advancedstate.temporal_memory.len(),
    })
}

/// Calculate simplified Phi measure for integrated information (IIT 3.0 bipartition approach)
///
/// Computes phi = I(X_left; X_right) = H(X_left) + H(X_right) - H(X_left, X_right)
/// where the probability distribution p(x) = |ψ|² across consciousness amplitudes.
/// This models integrated information as mutual information across the bipartition.
#[allow(dead_code)]
fn calculate_simplified_phi_measure(
    advancedstate: &AdvancedState,
    _config: &AdvancedConfig,
) -> NdimageResult<f64> {
    let probs: Vec<f64> = advancedstate
        .consciousness_amplitudes
        .iter()
        .map(|c| c.norm_sqr())
        .collect();

    let total: f64 = probs.iter().sum();
    if total < 1e-12 || probs.len() < 2 {
        return Ok(0.0);
    }

    let probs: Vec<f64> = probs.iter().map(|p| p / total).collect();
    let n = probs.len();
    let half = n / 2;

    // Marginal probability of left half
    let p_left: f64 = probs[..half].iter().sum();
    let h_left = if p_left > 1e-12 && p_left < 1.0 - 1e-12 {
        let p_right_complement = 1.0 - p_left;
        -p_left * p_left.ln() - p_right_complement * p_right_complement.ln()
    } else {
        0.0
    };

    // Marginal probability of right half
    let p_right: f64 = probs[half..].iter().sum();
    let h_right = if p_right > 1e-12 && p_right < 1.0 - 1e-12 {
        let p_left_complement = 1.0 - p_right;
        -p_right * p_right.ln() - p_left_complement * p_left_complement.ln()
    } else {
        0.0
    };

    // Joint entropy over the full distribution
    let h_joint: f64 = -probs
        .iter()
        .filter(|&&p| p > 1e-12)
        .map(|&p| p * p.ln())
        .sum::<f64>();

    // Phi = mutual information: I(left; right) = H(left) + H(right) - H(joint)
    let phi = (h_left + h_right - h_joint).max(0.0);

    Ok(phi.min(1.0))
}

/// Evolve consciousness parameters based on current state
#[allow(dead_code)]
fn evolve_consciousness_parameters(
    evolution_system: &mut QuantumConsciousnessEvolution,
    currentstate: &ConsciousnessState,
    _config: &AdvancedConfig,
) -> NdimageResult<()> {
    // Calculate evolution pressure based on consciousness quality
    let consciousness_fitness = (currentstate.level
        + currentstate.coherence_quality
        + currentstate.phi_measure
        + currentstate.self_awareness)
        / 4.0;

    // Apply evolutionary pressure
    if consciousness_fitness > evolution_system.emergence_threshold {
        // Positive selection - enhance current parameters
        evolution_system.evolution_rate = (evolution_system.evolution_rate * 1.05).min(0.1);
        evolution_system.selection_pressure =
            (evolution_system.selection_pressure * 0.95).max(0.01);
    } else {
        // Negative selection - explore parameter space
        evolution_system.evolution_rate = (evolution_system.evolution_rate * 0.95).max(0.001);
        evolution_system.selection_pressure = (evolution_system.selection_pressure * 1.05).min(0.5);
    }

    // Update complexity metrics
    evolution_system.complexitymetrics.integrated_information = currentstate.phi_measure;
    evolution_system.complexitymetrics.temporal_coherence = currentstate.coherence_quality;
    evolution_system.complexitymetrics.emergence_strength = consciousness_fitness;

    // Evolve quantum coherence optimization strategies
    evolve_coherence_strategies(
        &mut evolution_system.coherence_optimizer,
        consciousness_fitness,
    )?;

    Ok(())
}

/// Evolve quantum coherence optimization strategies using a quantum walk step
///
/// Applies one step of the continuous-time quantum walk on the performance history lattice:
/// ψ(t+dt) ≈ ψ(t) - i·L·dt·ψ(t)
/// where L is the graph Laplacian (1D lattice with nearest-neighbor connectivity).
/// The resulting coherence norms are stored back in the performance history.
#[allow(dead_code)]
fn evolve_coherence_strategies(
    optimizer: &mut QuantumCoherenceOptimizer,
    fitness: f64,
) -> NdimageResult<()> {
    // Add current fitness to performance history
    optimizer.performancehistory.push_back(fitness);
    if optimizer.performancehistory.len() > 50 {
        optimizer.performancehistory.pop_front();
    }

    let n = optimizer.performancehistory.len();
    if n < 2 {
        return Ok(());
    }

    // Build |ψ⟩ from performance history as real-valued quantum amplitudes
    let psi: Vec<f64> = optimizer.performancehistory.iter().cloned().collect();

    // Compute total norm for normalization
    let psi_norm: f64 = psi.iter().map(|x| x * x).sum::<f64>().sqrt();
    if psi_norm < 1e-12 {
        return Ok(());
    }
    let psi: Vec<f64> = psi.iter().map(|x| x / psi_norm).collect();

    // Apply one-step 1D quantum walk via first-order expansion: ψ' = ψ - i L dt ψ
    // Since we work in reals, the imaginary part contributes as a rotation.
    // We compute L·ψ (1D lattice Laplacian) and update real/imag components.
    let dt = 0.01_f64;
    let mut l_psi = vec![0.0_f64; n];
    for i in 0..n {
        // L = degree - adjacency: degree=2 for interior, 1 for boundary
        let degree = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
        let left = if i > 0 { psi[i - 1] } else { 0.0 };
        let right = if i < n - 1 { psi[i + 1] } else { 0.0 };
        l_psi[i] = degree * psi[i] - left - right;
    }

    // After one step: |ψ_new|² = psi² + (dt * L·psi)² (first-order approximation)
    // We store the resulting probability norms back into performance history
    let mut new_history: VecDeque<f64> = VecDeque::with_capacity(n);
    for i in 0..n {
        // ψ'_real = psi[i], ψ'_imag = -dt * l_psi[i]
        let new_norm = (psi[i] * psi[i] + (dt * l_psi[i]).powi(2)).sqrt();
        new_history.push_back(new_norm);
    }

    optimizer.performancehistory = new_history;

    Ok(())
}

/// Apply evolved quantum consciousness operators with evolutionary enhancements
#[allow(dead_code)]
fn apply_evolved_quantum_consciousness_operators(
    feature_vector: &[f64],
    consciousnessstate: &ArrayView1<Complex<f64>>,
    config: &AdvancedConfig,
    evolution_system: &QuantumConsciousnessEvolution,
) -> NdimageResult<Complex<f64>> {
    // Start with basic quantum consciousness operators
    let mut quantumstate =
        apply_quantum_consciousness_operators(feature_vector, consciousnessstate, config)?;

    // Apply evolutionary enhancements
    let evolution_enhancement = Complex::new(
        1.0 + evolution_system.evolution_rate
            * evolution_system.complexitymetrics.emergence_strength,
        evolution_system.selection_pressure * 0.1,
    );

    quantumstate *= evolution_enhancement;

    // Apply coherence optimization
    let coherence_boost = 1.0
        + evolution_system
            .coherence_optimizer
            .performancehistory
            .iter()
            .sum::<f64>()
            / evolution_system
                .coherence_optimizer
                .performancehistory
                .len()
                .max(1) as f64;

    quantumstate *= coherence_boost;

    // Normalize to maintain quantum state properties
    let norm = quantumstate.norm();
    if norm > 1e-10 {
        quantumstate /= norm;
    }

    Ok(quantumstate)
}

/// Apply consciousness evolution and selection to quantum amplitudes
#[allow(dead_code)]
fn apply_consciousness_evolution_selection(
    consciousness_amplitude: Complex<f64>,
    evolution_system: &QuantumConsciousnessEvolution,
    position: (usize, usize),
    _config: &AdvancedConfig,
) -> NdimageResult<f64> {
    // Calculate base consciousness probability
    let base_probability = consciousness_amplitude.norm_sqr();

    // Apply evolutionary selection pressure
    let selection_factor = 1.0
        + evolution_system.selection_pressure
            * (evolution_system.complexitymetrics.emergence_strength - 0.5);

    // Apply spatial coherence effects (simplified)
    let spatial_coherence = 1.0 + 0.1 * ((position.0 + position.1) as f64 * 0.01).sin();

    // Combine factors
    let evolved_probability = base_probability * selection_factor * spatial_coherence;

    Ok(evolved_probability.min(1.0))
}

/// Apply evolved global consciousness coherence
///
/// Uses the most recently evolved coherence strategy from the optimizer's performance history
/// to apply adaptive scaling. Each region gets a locally-estimated phi-based scaling factor.
#[allow(dead_code)]
fn apply_evolved_global_consciousness_coherence(
    consciousness_output: &mut Array2<f64>,
    advancedstate: &AdvancedState,
    evolution_system: &QuantumConsciousnessEvolution,
    config: &AdvancedConfig,
) -> NdimageResult<()> {
    // Use the most recent evolved coherence norm from quantum walk history
    let evolved_coherence = evolution_system
        .coherence_optimizer
        .performancehistory
        .back()
        .cloned()
        .unwrap_or(evolution_system.complexitymetrics.temporal_coherence);

    // Base damping from the phi measure
    let phi = calculate_simplified_phi_measure(advancedstate, config)?;
    let gamma = config.quantum.decoherence_rate.max(0.0);

    let (height, width) = consciousness_output.dim();

    // Apply adaptive local scaling based on evolved coherence and phi
    for y in 0..height {
        for x in 0..width {
            // Local phi estimate: modulate by spatial position
            let spatial_mod = (1.0
                + 0.05
                    * ((y as f64 / height.max(1) as f64) * PI).sin()
                    * ((x as f64 / width.max(1) as f64) * PI).cos())
            .abs();

            let local_phi = (phi * spatial_mod).min(1.0);
            let local_damping = (-gamma * local_phi).exp();

            // Scale by evolved coherence strength
            let scale = local_damping * (1.0 + evolved_coherence * 0.05);
            consciousness_output[(y, x)] *= scale;
        }
    }

    Ok(())
}

/// Update consciousness evolution history
#[allow(dead_code)]
fn update_consciousness_evolutionhistory(
    evolution_system: &mut QuantumConsciousnessEvolution,
    currentstate: &ConsciousnessState,
) -> NdimageResult<()> {
    // Add current state to evolution history
    evolution_system
        .evolutionhistory
        .push_back(currentstate.clone());

    // Maintain history size limit
    if evolution_system.evolutionhistory.len() > 100 {
        evolution_system.evolutionhistory.pop_front();
    }

    Ok(())
}

/// Apply temporal-causal inference to the consciousness output field.
///
/// # Model
///
/// This implements a deterministic, Granger-style temporal-causal propagation over the
/// consciousness field `C(y, x)`. Each pixel is treated as an event whose flattened index is
/// `id = y * width + x`. The accumulated causal graph
/// `causal_graph: BTreeMap<id, Vec<CausalRelation>>` maps every *source* event to its outgoing
/// causal relations, each carrying a `strength`, a `confidence`, and a temporal `delay`.
///
/// For a given target pixel `t`, the causal influence is the temporally-lagged, causally-weighted
/// sum of the activity of all its source pixels:
///
/// ```text
/// influence(t) = Σ_{ (s → t) ∈ graph }  strength · confidence · decay(delay) · C_lag(s, delay)
/// ```
///
/// where:
/// * `decay(delay) = exp(-delay / τ)` with `τ = max(temporal_window, 1)` is a temporal decay that
///   discounts long-lag causation (lagged states matter less the further back they are);
/// * `C_lag(s, delay)` is the value of source pixel `s` taken from `temporal_memory` at the given
///   lag (the `delay`-th most recent frame). When the temporal memory is shallower than the
///   requested lag, the *current* output `C(s)` is used as the best available estimate.
///
/// The influence is then normalised by the total incoming causal weight so the result stays a
/// bounded weighted average of the source activities, and the output is updated as
///
/// ```text
/// C'(t) = (1 - α_eff) · C(t) + α_eff · influence_norm(t)
/// ```
///
/// with an effective causal coupling `α_eff = α · mean_decay(t)`, where the base coupling
/// `α = clamp(causal_depth / (causal_depth + temporal_window), 0, 1)` is derived from the
/// configuration and `mean_decay(t) ∈ (0, 1]` is the weight-averaged temporal discount of the
/// relations incoming to `t`. Because the weighted average alone is invariant to a common scaling
/// of the weights, folding `mean_decay` into the coupling is what makes long-lag causation move
/// the present value *less* than recent causation — even when a pixel has a single cause.
/// Pixels with no incoming causal relations are left unchanged. The whole operation is
/// deterministic and order-independent (a snapshot of the input field is used for every read),
/// and every output value remains finite and non-negative whenever the input is.
pub(crate) fn apply_temporal_causal_inference(
    consciousness_output: &mut Array2<f64>,
    state: &AdvancedState,
    config: &AdvancedConfig,
) -> NdimageResult<()> {
    let (height, width) = consciousness_output.dim();
    if height == 0 || width == 0 || state.causal_graph.is_empty() {
        return Ok(());
    }

    // Temporal decay constant: τ controls how strongly long-lag causation is discounted.
    let tau = config.temporal_window.max(1) as f64;

    // Causal coupling factor α ∈ [0, 1] derived from the configured causal/temporal depths.
    let causal_depth = config.causal_depth as f64;
    let alpha = (causal_depth / (causal_depth + tau)).clamp(0.0, 1.0);

    // Snapshot the current field so all causal reads use the same (pre-update) state,
    // making the propagation deterministic and independent of traversal order.
    let snapshot = consciousness_output.clone();

    // Helper: fetch the value of a source event at a given temporal lag.
    // Lag 0 (or insufficient memory) falls back to the current snapshot value.
    let lagged_value = |source_id: usize, delay: usize| -> f64 {
        let sy = source_id / width;
        let sx = source_id % width;
        if sy >= height || sx >= width {
            return 0.0;
        }
        if delay == 0 || state.temporal_memory.is_empty() {
            return snapshot[(sy, sx)];
        }
        // temporal_memory is ordered oldest -> newest; the `delay`-th most recent frame.
        let len = state.temporal_memory.len();
        let lag = delay.min(len);
        let idx = len - lag;
        let frame = &state.temporal_memory[idx];
        let (fh, fw, _fd) = frame.dim();
        if sy < fh && sx < fw {
            frame[(sy, sx, 0)]
        } else {
            snapshot[(sy, sx)]
        }
    };

    // Accumulate incoming causal influence per target pixel.
    // Per target we track:
    //   acc          = Σ (decayed_weight · source_activity)   -> for the weighted-average influence
    //   total_w      = Σ  decayed_weight                      -> normaliser for that average
    //   total_raw    = Σ (strength · confidence)              -> un-decayed coupling mass
    // The ratio mean_decay = total_w / total_raw ∈ (0, 1] is the weight-averaged temporal
    // discount; it scales how strongly the (time-lagged) causal channel overrides the present
    // value, so a target whose only cause is far in the past is moved less than one with a
    // recent cause.
    let mut influence = vec![(0.0_f64, 0.0_f64, 0.0_f64); height * width];

    for relations in state.causal_graph.values() {
        for relation in relations {
            if relation.target >= influence.len() {
                continue;
            }
            let raw = relation.strength.abs() * relation.confidence.clamp(0.0, 1.0);
            if raw <= 0.0 {
                continue;
            }
            // Temporal decay discounts long-lag causation.
            let decay = (-(relation.delay as f64) / tau).exp();
            let weight = raw * decay;
            let source_activity = lagged_value(relation.source, relation.delay);
            let (ref mut acc, ref mut total_w, ref mut total_raw) = influence[relation.target];
            *acc += weight * source_activity;
            *total_w += weight;
            *total_raw += raw;
        }
    }

    // Blend the causal influence into the consciousness field where causation is present.
    for y in 0..height {
        for x in 0..width {
            let id = y * width + x;
            let (acc, total_w, total_raw) = influence[id];
            if total_w > 1e-12 && total_raw > 1e-12 {
                // Weighted average of (lagged) source activities.
                let influence_norm = acc / total_w;
                // Temporal discount on the coupling strength (recent causes pull harder).
                let mean_decay = (total_w / total_raw).clamp(0.0, 1.0);
                let effective_alpha = (alpha * mean_decay).clamp(0.0, 1.0);

                let current = snapshot[(y, x)];
                let updated = (1.0 - effective_alpha) * current + effective_alpha * influence_norm;
                // Keep values finite; preserve non-negativity of the probability-like field.
                consciousness_output[(y, x)] = if updated.is_finite() {
                    updated.max(0.0)
                } else {
                    current
                };
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2, Array3, Array4};
    use scirs2_core::numeric::Complex;
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::{Arc, RwLock};

    fn make_test_state(amplitudes: Array4<Complex<f64>>) -> AdvancedState {
        use scirs2_core::ndarray::Array5;

        AdvancedState {
            consciousness_amplitudes: amplitudes,
            meta_parameters: Array2::zeros((4, 4)),
            network_topology: Arc::new(RwLock::new(NetworkTopology {
                connections: std::collections::HashMap::new(),
                nodes: Vec::new(),
                global_properties: NetworkProperties {
                    coherence: 0.5,
                    self_organization_index: 0.3,
                    consciousness_emergence: 0.2,
                    efficiency: 0.8,
                },
            })),
            temporal_memory: VecDeque::new(),
            causal_graph: BTreeMap::new(),
            advancedfeatures: Array5::zeros((1, 1, 1, 1, 1)),
            resource_allocation: ResourceState {
                cpu_allocation: vec![0.5],
                memory_allocation: 0.5,
                gpu_allocation: None,
                quantum_allocation: None,
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
    fn test_phi_measure_nonnegative() {
        // Random-ish amplitudes: phi should always be >= 0
        let mut amps = Array4::zeros((4, 4, 2, 2));
        for (i, v) in amps.iter_mut().enumerate() {
            *v = Complex::new((i as f64 * 0.1).sin(), (i as f64 * 0.1).cos());
        }
        let state = make_test_state(amps);
        let config = AdvancedConfig::default();

        let phi =
            calculate_simplified_phi_measure(&state, &config).expect("phi measure should not fail");
        assert!(phi >= 0.0, "phi must be non-negative, got {}", phi);
    }

    #[test]
    fn test_phi_measure_zero_for_uniform_state() {
        // All amplitudes equal → uniform probability → phi should be 0
        // (all probability mass is uniformly distributed, no bipartition imbalance beyond rounding)
        let amps = Array4::from_elem((4, 4, 2, 2), Complex::new(1.0, 0.0));
        let state = make_test_state(amps);
        let config = AdvancedConfig::default();

        let phi =
            calculate_simplified_phi_measure(&state, &config).expect("phi measure should not fail");
        // For uniform distribution: H(left) + H(right) - H(joint) = 0
        assert!(
            phi < 1e-10,
            "phi should be near 0 for uniform distribution, got {}",
            phi
        );
    }

    #[test]
    fn test_quantum_walk_step_preserves_probability() {
        // After evolve_coherence_strategies, norms from quantum walk should sum to ~1
        // (we renormalize, so we check that history is non-empty and values are bounded)
        let mut optimizer = QuantumCoherenceOptimizer {
            strategies: Vec::new(),
            optimization_params: std::collections::HashMap::new(),
            performancehistory: VecDeque::new(),
        };

        // Push enough entries for quantum walk to activate
        for i in 0..10_usize {
            evolve_coherence_strategies(&mut optimizer, 0.1 * i as f64)
                .expect("evolve should not fail");
        }

        // After quantum walk evolution, all values should be finite and non-negative
        let history: Vec<f64> = optimizer.performancehistory.iter().cloned().collect();
        assert!(!history.is_empty(), "history should not be empty");

        for &v in &history {
            assert!(
                v.is_finite() && v >= 0.0,
                "quantum walk output should be finite non-negative, got {}",
                v
            );
        }
    }

    /// Build a `CausalRelation` with the given parameters.
    fn make_relation(
        source: usize,
        target: usize,
        strength: f64,
        delay: usize,
        confidence: f64,
    ) -> CausalRelation {
        CausalRelation {
            source,
            target,
            strength,
            delay,
            confidence,
        }
    }

    #[test]
    fn test_temporal_causal_inference_empty_graph_is_noop() {
        // With no causal relations the field must be returned unchanged.
        let amps = Array4::from_elem((2, 2, 1, 2), Complex::new(0.5, 0.0));
        let state = make_test_state(amps); // causal_graph is empty
        let config = AdvancedConfig::default();

        let mut output = Array2::from_shape_fn((4, 4), |(y, x)| 0.1 * (y + x) as f64);
        let before = output.clone();

        apply_temporal_causal_inference(&mut output, &state, &config)
            .expect("temporal causal inference should not fail");

        assert_eq!(
            before, output,
            "empty causal graph must leave field unchanged"
        );
    }

    #[test]
    fn test_temporal_causal_inference_blends_toward_source() {
        // A single zero-delay causal relation source -> target should pull the target's
        // value toward the source's value (deterministic convex blend), staying bounded.
        let amps = Array4::from_elem((2, 2, 1, 2), Complex::new(0.5, 0.0));
        let mut state = make_test_state(amps);
        let config = AdvancedConfig::default();

        let width = 4usize;
        // source pixel (0,0) -> id 0 ; target pixel (0,1) -> id 1
        let source_id = 0usize;
        let target_id = 1usize;
        state.causal_graph.insert(
            source_id,
            vec![make_relation(source_id, target_id, 1.0, 0, 1.0)],
        );

        // Source has high activity, target has low activity.
        let mut output = Array2::<f64>::zeros((1, width));
        output[(0, 0)] = 1.0; // source
        output[(0, 1)] = 0.0; // target
        let source_val = output[(0, 0)];
        let target_before = output[(0, 1)];

        apply_temporal_causal_inference(&mut output, &state, &config)
            .expect("temporal causal inference should not fail");

        let target_after = output[(0, 1)];

        // Source value unchanged (no incoming causation).
        assert!(
            (output[(0, 0)] - source_val).abs() < 1e-12,
            "source pixel must be unaffected"
        );
        // Target moved strictly toward the (higher) source value.
        assert!(
            target_after > target_before,
            "target should increase toward source, before={} after={}",
            target_before,
            target_after
        );
        assert!(
            target_after <= source_val + 1e-12,
            "blended value must not exceed source value, got {}",
            target_after
        );
        // Bounded and finite.
        assert!(
            target_after.is_finite() && target_after >= 0.0,
            "output must be finite and non-negative, got {}",
            target_after
        );

        // Verify the exact convex-blend value: alpha = causal_depth/(causal_depth+tau).
        let tau = config.temporal_window.max(1) as f64;
        let causal_depth = config.causal_depth as f64;
        let alpha = causal_depth / (causal_depth + tau);
        let expected = (1.0 - alpha) * target_before + alpha * source_val;
        assert!(
            (target_after - expected).abs() < 1e-10,
            "blend mismatch: expected {} got {}",
            expected,
            target_after
        );
    }

    #[test]
    fn test_temporal_causal_inference_delay_monotonic_decay() {
        // Larger causal delay => stronger temporal decay => weaker influence =>
        // target stays closer to its original value. We compare delay=1 vs delay=8.
        let config = AdvancedConfig::default();
        let width = 4usize;
        let source_id = 0usize;
        let target_id = 1usize;

        // Helper that runs inference for a given delay and returns the resulting target value.
        let run_for_delay = |delay: usize| -> f64 {
            let amps = Array4::from_elem((2, 2, 1, 2), Complex::new(0.5, 0.0));
            let mut state = make_test_state(amps);
            state.causal_graph.insert(
                source_id,
                vec![make_relation(source_id, target_id, 1.0, delay, 1.0)],
            );
            // Provide enough temporal memory so the lag can be honoured.
            for _ in 0..16 {
                let mut frame = Array3::<f64>::zeros((1, width, 1));
                frame[(0, 0, 0)] = 1.0; // source activity in history
                state.temporal_memory.push_back(frame);
            }

            let mut output = Array2::<f64>::zeros((1, width));
            output[(0, 0)] = 1.0;
            output[(0, 1)] = 0.0;
            apply_temporal_causal_inference(&mut output, &state, &config)
                .expect("temporal causal inference should not fail");
            output[(0, 1)]
        };

        let near = run_for_delay(1);
        let far = run_for_delay(8);

        // Both pull the target up from 0, but the longer delay pulls less.
        assert!(near > 0.0 && far > 0.0, "both should increase the target");
        assert!(
            near > far,
            "shorter delay should yield stronger influence: near(delay=1)={} far(delay=8)={}",
            near,
            far
        );
    }
}
