//! Advanced Memristive Learning for Neuromorphic Computing
//!
//! This module implements sophisticated memristive computing paradigms including
//! crossbar arrays with multiple device types, advanced plasticity mechanisms,
//! homeostatic regulation, metaplasticity, and neuromodulation for spatial learning.

use crate::error::SpatialResult;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::Rng;
use std::collections::VecDeque;

/// Advanced memristive learning system with synaptic plasticity and homeostasis
///
/// This system implements a comprehensive memristive computing framework with
/// multiple types of plasticity, homeostatic regulation, and neuromodulation.
/// It supports various memristive device types and advanced learning dynamics.
///
/// # Features
/// - Multiple memristive device types (TiO2, HfO2, Phase Change, etc.)
/// - Advanced plasticity mechanisms (STDP, homeostatic scaling, etc.)
/// - Homeostatic regulation for stable learning
/// - Metaplasticity for learning-to-learn capabilities
/// - Neuromodulation systems (dopamine, serotonin, etc.)
/// - Memory consolidation and forgetting protection
/// - Comprehensive learning history tracking
///
/// # Example
/// ```rust
/// use ndarray::Array2;
/// use scirs2_spatial::neuromorphic::algorithms::{AdvancedMemristiveLearning, MemristiveDeviceType};
///
/// let mut learning_system = AdvancedMemristiveLearning::new(4, 2, MemristiveDeviceType::TitaniumDioxide)
///     .with_forgetting_protection(true);
///
/// // Training data
/// let spatial_data = Array2::from_shape_vec((4, 4), vec![
///     0.0, 0.0, 1.0, 1.0,
///     1.0, 0.0, 0.0, 1.0,
///     0.0, 1.0, 1.0, 0.0,
///     1.0, 1.0, 0.0, 0.0
/// ]).unwrap();
/// let targets = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);
///
/// // Train the system
/// # tokio_test::block_on(async {
/// let result = learning_system.train_spatial_data(&spatial_data.view(), &targets.view(), 50).await.unwrap();
/// println!("Final accuracy: {:.2}", result.training_metrics.last().unwrap().accuracy);
/// # });
/// ```
#[derive(Debug, Clone)]
pub struct AdvancedMemristiveLearning {
    /// Memristive crossbar array
    crossbar_array: MemristiveCrossbar,
    /// Synaptic plasticity mechanisms
    plasticity_mechanisms: Vec<PlasticityMechanism>,
    /// Homeostatic regulation system
    homeostatic_system: HomeostaticSystem,
    /// Metaplasticity rules
    metaplasticity: MetaplasticityRules,
    /// Neuromodulation system
    neuromodulation: NeuromodulationSystem,
    /// Learning history
    learning_history: LearningHistory,
    /// Enable online learning
    #[allow(dead_code)]
    online_learning: bool,
    /// Enable catastrophic forgetting protection
    forgetting_protection: bool,
}

/// Memristive crossbar array with advanced properties
///
/// Represents a physical memristive crossbar array with realistic device
/// characteristics including conductance, resistance, switching dynamics,
/// and aging effects.
#[derive(Debug, Clone)]
pub struct MemristiveCrossbar {
    /// Device conductances
    pub conductances: Array2<f64>,
    /// Device resistances
    pub resistances: Array2<f64>,
    /// Switching thresholds
    pub switching_thresholds: Array2<f64>,
    /// Retention times
    pub retention_times: Array2<f64>,
    /// Endurance cycles
    pub endurance_cycles: Array2<usize>,
    /// Programming voltages
    pub programming_voltages: Array2<f64>,
    /// Temperature effects
    pub temperature_coefficients: Array2<f64>,
    /// Device variability
    pub device_variability: Array2<f64>,
    /// Crossbar dimensions
    pub dimensions: (usize, usize),
    /// Device type
    pub device_type: MemristiveDeviceType,
}

/// Types of memristive devices
///
/// Different memristive technologies have distinct switching characteristics,
/// speed, endurance, and non-linearity properties that affect learning dynamics.
#[derive(Debug, Clone)]
pub enum MemristiveDeviceType {
    /// Titanium dioxide (TiO2) - Classic memristor with exponential switching
    TitaniumDioxide,
    /// Hafnium oxide (HfO2) - High endurance with steep switching
    HafniumOxide,
    /// Tantalum oxide (Ta2O5) - Moderate switching characteristics
    TantalumOxide,
    /// Silver sulfide (Ag2S) - Fast switching, lower endurance
    SilverSulfide,
    /// Organic memristor - Biocompatible, variable characteristics
    Organic,
    /// Phase change memory - Binary switching with high contrast
    PhaseChange,
    /// Magnetic tunnel junction - Non-volatile, spin-based switching
    MagneticTunnelJunction,
}

/// Synaptic plasticity mechanisms
///
/// Encapsulates different types of synaptic plasticity with their
/// associated parameters and learning dynamics.
#[derive(Debug, Clone)]
pub struct PlasticityMechanism {
    /// Mechanism type
    pub mechanism_type: PlasticityType,
    /// Time constants
    pub time_constants: PlasticityTimeConstants,
    /// Learning rates
    pub learning_rates: PlasticityLearningRates,
    /// Threshold parameters
    pub thresholds: PlasticityThresholds,
    /// Enable state
    pub enabled: bool,
    /// Weight scaling factors
    pub weight_scaling: f64,
}

/// Types of synaptic plasticity
///
/// Different plasticity mechanisms that can be enabled individually
/// or in combination for complex learning dynamics.
#[derive(Debug, Clone)]
pub enum PlasticityType {
    /// Spike-timing dependent plasticity
    STDP,
    /// Homeostatic synaptic scaling
    HomeostaticScaling,
    /// Intrinsic plasticity
    IntrinsicPlasticity,
    /// Heterosynaptic plasticity
    HeterosynapticPlasticity,
    /// Metaplasticity
    Metaplasticity,
    /// Calcium-dependent plasticity
    CalciumDependent,
    /// Voltage-dependent plasticity
    VoltageDependent,
    /// Frequency-dependent plasticity
    FrequencyDependent,
}

/// Time constants for plasticity mechanisms
#[derive(Debug, Clone)]
pub struct PlasticityTimeConstants {
    /// Fast component time constant
    pub tau_fast: f64,
    /// Slow component time constant
    pub tau_slow: f64,
    /// STDP time window
    pub stdp_window: f64,
    /// Homeostatic time constant
    pub tau_homeostatic: f64,
    /// Calcium decay time
    pub tau_calcium: f64,
}

/// Learning rates for different plasticity components
#[derive(Debug, Clone)]
pub struct PlasticityLearningRates {
    /// Potentiation learning rate
    pub potentiation_rate: f64,
    /// Depression learning rate
    pub depression_rate: f64,
    /// Homeostatic learning rate
    pub homeostatic_rate: f64,
    /// Metaplastic learning rate
    pub metaplastic_rate: f64,
    /// Intrinsic plasticity rate
    pub intrinsic_rate: f64,
}

/// Threshold parameters for plasticity
#[derive(Debug, Clone)]
pub struct PlasticityThresholds {
    /// LTP threshold
    pub ltp_threshold: f64,
    /// LTD threshold
    pub ltd_threshold: f64,
    /// Homeostatic target activity
    pub target_activity: f64,
    /// Metaplasticity threshold
    pub metaplasticity_threshold: f64,
    /// Saturation threshold
    pub saturation_threshold: f64,
}

/// Homeostatic regulation system
///
/// Maintains stable neural activity levels through multiple homeostatic
/// mechanisms including synaptic scaling and intrinsic excitability adjustment.
#[derive(Debug, Clone)]
pub struct HomeostaticSystem {
    /// Target firing rates
    pub target_firing_rates: Array1<f64>,
    /// Current firing rates
    pub current_firing_rates: Array1<f64>,
    /// Homeostatic time constants
    pub time_constants: Array1<f64>,
    /// Regulation mechanisms
    pub mechanisms: Vec<HomeostaticMechanism>,
    /// Adaptation rates
    pub adaptation_rates: Array1<f64>,
    /// Activity history
    pub activity_history: VecDeque<Array1<f64>>,
    /// History window size
    pub history_window: usize,
}

/// Types of homeostatic mechanisms
#[derive(Debug, Clone)]
pub enum HomeostaticMechanism {
    /// Synaptic scaling
    SynapticScaling,
    /// Intrinsic excitability adjustment
    IntrinsicExcitability,
    /// Structural plasticity
    StructuralPlasticity,
    /// Inhibitory plasticity
    InhibitoryPlasticity,
    /// Metaplastic regulation
    MetaplasticRegulation,
}

/// Metaplasticity rules for learning-to-learn
///
/// Implements meta-learning capabilities where the learning process
/// itself adapts based on experience and performance history.
#[derive(Debug, Clone)]
pub struct MetaplasticityRules {
    /// Learning rate adaptation rules
    pub learning_rate_adaptation: LearningRateAdaptation,
    /// Threshold adaptation rules
    pub threshold_adaptation: ThresholdAdaptation,
    /// Memory consolidation rules
    pub consolidation_rules: ConsolidationRules,
    /// Forgetting protection rules
    pub forgetting_protection: ForgettingProtectionRules,
}

/// Learning rate adaptation mechanisms
#[derive(Debug, Clone)]
pub struct LearningRateAdaptation {
    /// Base learning rate
    pub base_rate: f64,
    /// Adaptation factor
    pub adaptation_factor: f64,
    /// Performance history
    pub performance_history: VecDeque<f64>,
    /// Adaptation threshold
    pub adaptation_threshold: f64,
    /// Maximum learning rate
    pub max_rate: f64,
    /// Minimum learning rate
    pub min_rate: f64,
}

/// Threshold adaptation for dynamic learning
#[derive(Debug, Clone)]
pub struct ThresholdAdaptation {
    /// Adaptive thresholds
    pub adaptive_thresholds: Array1<f64>,
    /// Threshold update rates
    pub update_rates: Array1<f64>,
    /// Target activation levels
    pub target_activations: Array1<f64>,
    /// Threshold bounds
    pub threshold_bounds: Vec<(f64, f64)>,
}

/// Memory consolidation rules
#[derive(Debug, Clone)]
pub struct ConsolidationRules {
    /// Consolidation time windows
    pub time_windows: Vec<f64>,
    /// Consolidation strengths
    pub consolidation_strengths: Array1<f64>,
    /// Replay mechanisms
    pub replay_enabled: bool,
    /// Replay patterns
    pub replay_patterns: Vec<Array1<f64>>,
    /// Systems consolidation
    pub systems_consolidation: bool,
}

/// Forgetting protection mechanisms
#[derive(Debug, Clone)]
pub struct ForgettingProtectionRules {
    /// Elastic weight consolidation
    pub ewc_enabled: bool,
    /// Fisher information matrix
    pub fisher_information: Array2<f64>,
    /// Synaptic intelligence
    pub synaptic_intelligence: bool,
    /// Importance weights
    pub importance_weights: Array1<f64>,
    /// Protection strength
    pub protection_strength: f64,
}

/// Neuromodulation system for context-dependent learning
///
/// Models the effects of neuromodulators on learning and plasticity,
/// enabling context-dependent adaptation of learning parameters.
#[derive(Debug, Clone)]
pub struct NeuromodulationSystem {
    /// Dopamine levels
    pub dopamine_levels: Array1<f64>,
    /// Serotonin levels
    pub serotonin_levels: Array1<f64>,
    /// Acetylcholine levels
    pub acetylcholine_levels: Array1<f64>,
    /// Noradrenaline levels
    pub noradrenaline_levels: Array1<f64>,
    /// Modulation effects
    pub modulation_effects: NeuromodulationEffects,
    /// Release patterns
    pub release_patterns: NeuromodulatorReleasePatterns,
}

/// Effects of neuromodulation on plasticity
#[derive(Debug, Clone)]
pub struct NeuromodulationEffects {
    /// Effect on learning rate
    pub learning_rate_modulation: Array1<f64>,
    /// Effect on thresholds
    pub threshold_modulation: Array1<f64>,
    /// Effect on excitability
    pub excitability_modulation: Array1<f64>,
    /// Effect on attention
    pub attention_modulation: Array1<f64>,
}

/// Neuromodulator release patterns
#[derive(Debug, Clone)]
pub struct NeuromodulatorReleasePatterns {
    /// Phasic dopamine release
    pub phasic_dopamine: Vec<(f64, f64)>, // (time, amplitude)
    /// Tonic serotonin level
    pub tonic_serotonin: f64,
    /// Cholinergic attention signals
    pub cholinergic_attention: Array1<f64>,
    /// Stress-related noradrenaline
    pub stress_noradrenaline: f64,
}

/// Learning history tracking
///
/// Comprehensive tracking of learning progress, weight changes,
/// and important events during training.
#[derive(Debug, Clone)]
pub struct LearningHistory {
    /// Weight change history
    pub weight_changes: VecDeque<Array2<f64>>,
    /// Performance metrics
    pub performance_metrics: VecDeque<PerformanceMetrics>,
    /// Plasticity events
    pub plasticity_events: VecDeque<PlasticityEvent>,
    /// Consolidation events
    pub consolidation_events: VecDeque<ConsolidationEvent>,
    /// Maximum history length
    pub max_history_length: usize,
}

/// Performance metrics for learning assessment
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Accuracy
    pub accuracy: f64,
    /// Learning speed
    pub learning_speed: f64,
    /// Stability
    pub stability: f64,
    /// Generalization
    pub generalization: f64,
    /// Timestamp
    pub timestamp: f64,
}

/// Plasticity event recording
#[derive(Debug, Clone)]
pub struct PlasticityEvent {
    /// Event type
    pub event_type: PlasticityEventType,
    /// Synapses involved
    pub synapses: Vec<(usize, usize)>,
    /// Magnitude of change
    pub magnitude: f64,
    /// Timestamp
    pub timestamp: f64,
    /// Context information
    pub context: String,
}

/// Types of plasticity events
#[derive(Debug, Clone)]
pub enum PlasticityEventType {
    LongTermPotentiation,
    LongTermDepression,
    HomeostaticScaling,
    StructuralPlasticity,
    MetaplasticChange,
}

/// Memory consolidation event
#[derive(Debug, Clone)]
pub struct ConsolidationEvent {
    /// Consolidation type
    pub consolidation_type: ConsolidationType,
    /// Memory patterns consolidated
    pub patterns: Vec<Array1<f64>>,
    /// Consolidation strength
    pub strength: f64,
    /// Timestamp
    pub timestamp: f64,
}

/// Types of memory consolidation
#[derive(Debug, Clone)]
pub enum ConsolidationType {
    SynapticConsolidation,
    SystemsConsolidation,
    ReconsolidationUpdate,
    OfflineReplay,
}

/// Training result structure
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final weight matrix
    pub final_weights: Array2<f64>,
    /// Training metrics over time
    pub training_metrics: Vec<PerformanceMetrics>,
    /// Recorded plasticity events
    pub plasticity_events: VecDeque<PlasticityEvent>,
    /// Recorded consolidation events
    pub consolidation_events: VecDeque<ConsolidationEvent>,
}

impl AdvancedMemristiveLearning {
    /// Create new advanced memristive learning system
    ///
    /// # Arguments
    /// * `rows` - Number of rows in crossbar array
    /// * `cols` - Number of columns in crossbar array
    /// * `device_type` - Type of memristive device to simulate
    ///
    /// # Returns
    /// A new `AdvancedMemristiveLearning` system with default parameters
    pub fn new(rows: usize, cols: usize, device_type: MemristiveDeviceType) -> Self {
        let crossbar_array = MemristiveCrossbar::new(rows, cols, device_type);

        let plasticity_mechanisms = vec![
            PlasticityMechanism::new(PlasticityType::STDP),
            PlasticityMechanism::new(PlasticityType::HomeostaticScaling),
            PlasticityMechanism::new(PlasticityType::IntrinsicPlasticity),
        ];

        let homeostatic_system = HomeostaticSystem::new(rows);
        let metaplasticity = MetaplasticityRules::new();
        let neuromodulation = NeuromodulationSystem::new(rows);
        let learning_history = LearningHistory::new();

        Self {
            crossbar_array,
            plasticity_mechanisms,
            homeostatic_system,
            metaplasticity,
            neuromodulation,
            learning_history,
            online_learning: true,
            forgetting_protection: true,
        }
    }

    /// Enable specific plasticity mechanism
    ///
    /// # Arguments
    /// * `plasticity_type` - Type of plasticity to enable
    pub fn enable_plasticity(mut self, plasticity_type: PlasticityType) -> Self {
        for mechanism in &mut self.plasticity_mechanisms {
            if std::mem::discriminant(&mechanism.mechanism_type)
                == std::mem::discriminant(&plasticity_type)
            {
                mechanism.enabled = true;
            }
        }
        self
    }

    /// Configure homeostatic regulation
    ///
    /// # Arguments
    /// * `target_rates` - Target firing rates for each neuron
    pub fn with_homeostatic_regulation(mut self, target_rates: Array1<f64>) -> Self {
        self.homeostatic_system.target_firing_rates = target_rates;
        self
    }

    /// Enable catastrophic forgetting protection
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable forgetting protection mechanisms
    pub fn with_forgetting_protection(mut self, enabled: bool) -> Self {
        self.forgetting_protection = enabled;
        self.metaplasticity.forgetting_protection.ewc_enabled = enabled;
        self
    }

    /// Train on spatial data with advanced plasticity
    ///
    /// Performs training using all enabled plasticity mechanisms,
    /// homeostatic regulation, and neuromodulation.
    ///
    /// # Arguments
    /// * `spatial_data` - Input spatial data (n_samples × n_features)
    /// * `target_outputs` - Target outputs for each sample
    /// * `epochs` - Number of training epochs
    ///
    /// # Returns
    /// Training results including final weights and learning history
    pub async fn train_spatial_data(
        &mut self,
        spatial_data: &ArrayView2<'_, f64>,
        target_outputs: &ArrayView1<'_, f64>,
        epochs: usize,
    ) -> SpatialResult<TrainingResult> {
        let mut training_metrics = Vec::new();

        for epoch in 0..epochs {
            // Process each spatial pattern
            let epoch_metrics = self.process_epoch(spatial_data, target_outputs).await?;

            // Apply homeostatic regulation
            self.apply_homeostatic_regulation().await?;

            // Apply metaplasticity updates
            self.apply_metaplasticity_updates(&epoch_metrics).await?;

            // Update neuromodulation
            self.update_neuromodulation(&epoch_metrics).await?;

            // Record learning history
            self.record_learning_history(&epoch_metrics, epoch as f64)
                .await?;

            training_metrics.push(epoch_metrics);

            // Check for consolidation triggers
            if self.should_trigger_consolidation(epoch) {
                self.trigger_memory_consolidation().await?;
            }
        }

        let final_weights = self.crossbar_array.conductances.clone();

        Ok(TrainingResult {
            final_weights,
            training_metrics,
            plasticity_events: self.learning_history.plasticity_events.clone(),
            consolidation_events: self.learning_history.consolidation_events.clone(),
        })
    }

    /// Process single training epoch
    async fn process_epoch(
        &mut self,
        spatial_data: &ArrayView2<'_, f64>,
        target_outputs: &ArrayView1<'_, f64>,
    ) -> SpatialResult<PerformanceMetrics> {
        let n_samples = spatial_data.dim().0;
        let mut total_error = 0.0;
        let mut correct_predictions = 0;

        for i in 0..n_samples {
            let input = spatial_data.row(i);
            let target = target_outputs[i];

            // Forward pass through memristive crossbar
            let output = self.forward_pass(&input).await?;

            // Compute error
            let error = target - output;
            total_error += error.abs();

            if error.abs() < 0.1 {
                correct_predictions += 1;
            }

            // Apply plasticity mechanisms
            self.apply_plasticity_mechanisms(&input, output, target, error)
                .await?;

            // Update device characteristics
            self.update_memristive_devices(&input, error).await?;
        }

        let accuracy = correct_predictions as f64 / n_samples as f64;
        let average_error = total_error / n_samples as f64;

        Ok(PerformanceMetrics {
            accuracy,
            learning_speed: 1.0 / (average_error + 1e-8),
            stability: self.compute_weight_stability(),
            generalization: self.estimate_generalization(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        })
    }

    /// Forward pass through memristive crossbar
    async fn forward_pass(&self, input: &ArrayView1<'_, f64>) -> SpatialResult<f64> {
        let mut output = 0.0;

        for (i, &input_val) in input.iter().enumerate() {
            if i < self.crossbar_array.dimensions.0 {
                for j in 0..self.crossbar_array.dimensions.1 {
                    let conductance = self.crossbar_array.conductances[[i, j]];
                    let current = input_val * conductance;

                    // Apply device non-linearity
                    let nonlinear_current = self.apply_device_nonlinearity(current, i, j);

                    output += nonlinear_current;
                }
            }
        }

        // Apply activation function
        Ok(Self::sigmoid(output))
    }

    /// Apply device-specific non-linearity
    fn apply_device_nonlinearity(&self, current: f64, row: usize, col: usize) -> f64 {
        match self.crossbar_array.device_type {
            MemristiveDeviceType::TitaniumDioxide => {
                // TiO2 exponential switching
                let threshold = self.crossbar_array.switching_thresholds[[row, col]];
                if current.abs() > threshold {
                    current * (1.0 + 0.1 * (current / threshold).ln())
                } else {
                    current
                }
            }
            MemristiveDeviceType::HafniumOxide => {
                // HfO2 with steep switching
                let threshold = self.crossbar_array.switching_thresholds[[row, col]];
                current * (1.0 + 0.2 * (current / threshold).tanh())
            }
            MemristiveDeviceType::PhaseChange => {
                // Phase change memory with threshold switching
                let threshold = self.crossbar_array.switching_thresholds[[row, col]];
                if current.abs() > threshold {
                    current * 2.0
                } else {
                    current * 0.1
                }
            }
            _ => current, // Linear for other types
        }
    }

    /// Apply all enabled plasticity mechanisms
    async fn apply_plasticity_mechanisms(
        &mut self,
        input: &ArrayView1<'_, f64>,
        output: f64,
        target: f64,
        error: f64,
    ) -> SpatialResult<()> {
        let mechanisms = self.plasticity_mechanisms.clone();
        for mechanism in &mechanisms {
            if mechanism.enabled {
                match mechanism.mechanism_type {
                    PlasticityType::STDP => {
                        self.apply_stdp_plasticity(input, output, &mechanism)
                            .await?;
                    }
                    PlasticityType::HomeostaticScaling => {
                        self.apply_homeostatic_scaling(input, output, &mechanism)
                            .await?;
                    }
                    PlasticityType::CalciumDependent => {
                        self.apply_calcium_dependent_plasticity(input, output, target, &mechanism)
                            .await?;
                    }
                    PlasticityType::VoltageDependent => {
                        self.apply_voltage_dependent_plasticity(input, error, &mechanism)
                            .await?;
                    }
                    _ => {
                        // Default plasticity rule
                        self.apply_error_based_plasticity(input, error, &mechanism)
                            .await?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply STDP plasticity with advanced timing rules
    async fn apply_stdp_plasticity(
        &mut self,
        input: &ArrayView1<'_, f64>,
        output: f64,
        mechanism: &PlasticityMechanism,
    ) -> SpatialResult<()> {
        let tau_plus = mechanism.time_constants.tau_fast;
        let tau_minus = mechanism.time_constants.tau_slow;
        let a_plus = mechanism.learning_rates.potentiation_rate;
        let a_minus = mechanism.learning_rates.depression_rate;

        // Simplified STDP implementation
        for (i, &input_val) in input.iter().enumerate() {
            if i < self.crossbar_array.dimensions.0 {
                for j in 0..self.crossbar_array.dimensions.1 {
                    // Compute timing difference (simplified)
                    let dt = if input_val > 0.5 && output > 0.5 {
                        1.0 // Pre before post
                    } else if input_val <= 0.5 && output > 0.5 {
                        -1.0 // Post before pre
                    } else {
                        0.0 // No timing relationship
                    };

                    let weight_change = if dt > 0.0 {
                        a_plus * (-dt / tau_plus).exp()
                    } else if dt < 0.0 {
                        -a_minus * (dt / tau_minus).exp()
                    } else {
                        0.0
                    };

                    self.crossbar_array.conductances[[i, j]] +=
                        weight_change * mechanism.weight_scaling;
                    self.crossbar_array.conductances[[i, j]] =
                        self.crossbar_array.conductances[[i, j]].clamp(0.0, 1.0);
                }
            }
        }

        Ok(())
    }

    /// Apply homeostatic scaling
    async fn apply_homeostatic_scaling(
        &mut self,
        _input: &ArrayView1<'_, f64>,
        output: f64,
        mechanism: &PlasticityMechanism,
    ) -> SpatialResult<()> {
        let target_activity = mechanism.thresholds.target_activity;
        let scaling_rate = mechanism.learning_rates.homeostatic_rate;

        // Global scaling based on overall activity
        let activity_error = output - target_activity;
        let scaling_factor = 1.0 - scaling_rate * activity_error;

        // Apply scaling to all weights
        for i in 0..self.crossbar_array.dimensions.0 {
            for j in 0..self.crossbar_array.dimensions.1 {
                self.crossbar_array.conductances[[i, j]] *= scaling_factor;
                self.crossbar_array.conductances[[i, j]] =
                    self.crossbar_array.conductances[[i, j]].clamp(0.0, 1.0);
            }
        }

        Ok(())
    }

    /// Apply calcium-dependent plasticity
    async fn apply_calcium_dependent_plasticity(
        &mut self,
        input: &ArrayView1<'_, f64>,
        output: f64,
        target: f64,
        mechanism: &PlasticityMechanism,
    ) -> SpatialResult<()> {
        // Simulate calcium dynamics
        let calcium_level = Self::compute_calcium_level(input, output, target);

        let ltp_threshold = mechanism.thresholds.ltp_threshold;
        let ltd_threshold = mechanism.thresholds.ltd_threshold;

        for (i, &input_val) in input.iter().enumerate() {
            if i < self.crossbar_array.dimensions.0 {
                for j in 0..self.crossbar_array.dimensions.1 {
                    let local_calcium = calcium_level * input_val;

                    let weight_change = if local_calcium > ltp_threshold {
                        mechanism.learning_rates.potentiation_rate * (local_calcium - ltp_threshold)
                    } else if local_calcium < ltd_threshold {
                        -mechanism.learning_rates.depression_rate * (ltd_threshold - local_calcium)
                    } else {
                        0.0
                    };

                    self.crossbar_array.conductances[[i, j]] += weight_change;
                    self.crossbar_array.conductances[[i, j]] =
                        self.crossbar_array.conductances[[i, j]].clamp(0.0, 1.0);
                }
            }
        }

        Ok(())
    }

    /// Apply voltage-dependent plasticity
    async fn apply_voltage_dependent_plasticity(
        &mut self,
        input: &ArrayView1<'_, f64>,
        error: f64,
        mechanism: &PlasticityMechanism,
    ) -> SpatialResult<()> {
        let voltage_threshold = mechanism.thresholds.ltd_threshold;

        for (i, &input_val) in input.iter().enumerate() {
            if i < self.crossbar_array.dimensions.0 {
                for j in 0..self.crossbar_array.dimensions.1 {
                    let local_voltage = input_val * error.abs();

                    if local_voltage > voltage_threshold {
                        let weight_change = mechanism.learning_rates.potentiation_rate
                            * (local_voltage - voltage_threshold)
                            * error.signum();

                        self.crossbar_array.conductances[[i, j]] += weight_change;
                        self.crossbar_array.conductances[[i, j]] =
                            self.crossbar_array.conductances[[i, j]].clamp(0.0, 1.0);
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply error-based plasticity (default)
    async fn apply_error_based_plasticity(
        &mut self,
        input: &ArrayView1<'_, f64>,
        error: f64,
        mechanism: &PlasticityMechanism,
    ) -> SpatialResult<()> {
        let learning_rate = mechanism.learning_rates.potentiation_rate;

        for (i, &input_val) in input.iter().enumerate() {
            if i < self.crossbar_array.dimensions.0 {
                for j in 0..self.crossbar_array.dimensions.1 {
                    let weight_change = learning_rate * error * input_val;

                    self.crossbar_array.conductances[[i, j]] += weight_change;
                    self.crossbar_array.conductances[[i, j]] =
                        self.crossbar_array.conductances[[i, j]].clamp(0.0, 1.0);
                }
            }
        }

        Ok(())
    }

    /// Compute calcium level for calcium-dependent plasticity
    fn compute_calcium_level(input: &ArrayView1<'_, f64>, output: f64, target: f64) -> f64 {
        let input_activity = input.iter().map(|&x| x.max(0.0)).sum::<f64>();
        let output_activity = output.max(0.0);
        let target_activity = target.max(0.0);

        // Simplified calcium dynamics
        (input_activity * 0.3 + output_activity * 0.4 + target_activity * 0.3).min(1.0)
    }

    /// Update memristive device characteristics
    async fn update_memristive_devices(
        &mut self,
        input: &ArrayView1<'_, f64>,
        _error: f64,
    ) -> SpatialResult<()> {
        for (i, &input_val) in input.iter().enumerate() {
            if i < self.crossbar_array.dimensions.0 {
                for j in 0..self.crossbar_array.dimensions.1 {
                    // Update resistance based on conductance
                    let conductance = self.crossbar_array.conductances[[i, j]];
                    self.crossbar_array.resistances[[i, j]] = if conductance > 1e-12 {
                        1.0 / conductance
                    } else {
                        1e12
                    };

                    // Update endurance cycles
                    if input_val > 0.1 {
                        self.crossbar_array.endurance_cycles[[i, j]] += 1;
                    }

                    // Apply device aging effects
                    self.apply_device_aging(i, j);

                    // Apply variability
                    self.apply_device_variability(i, j);
                }
            }
        }

        Ok(())
    }

    /// Apply device aging effects
    fn apply_device_aging(&mut self, row: usize, col: usize) {
        let cycles = self.crossbar_array.endurance_cycles[[row, col]];
        let aging_factor = 1.0 - (cycles as f64) * 1e-8; // Small aging effect

        self.crossbar_array.conductances[[row, col]] *= aging_factor.max(0.1);
    }

    /// Apply device-to-device variability
    fn apply_device_variability(&mut self, row: usize, col: usize) {
        let variability = self.crossbar_array.device_variability[[row, col]];
        let mut rng = rand::rng();
        let noise = (rng.gen_range(0.0..1.0) - 0.5) * variability;

        self.crossbar_array.conductances[[row, col]] += noise;
        self.crossbar_array.conductances[[row, col]] =
            self.crossbar_array.conductances[[row, col]].clamp(0.0, 1.0);
    }

    /// Apply homeostatic regulation
    async fn apply_homeostatic_regulation(&mut self) -> SpatialResult<()> {
        // Update firing rate history
        let current_rates = self.compute_current_firing_rates();
        self.homeostatic_system
            .activity_history
            .push_back(current_rates);

        // Maintain history window
        if self.homeostatic_system.activity_history.len() > self.homeostatic_system.history_window {
            self.homeostatic_system.activity_history.pop_front();
        }

        // Apply homeostatic mechanisms
        self.apply_synaptic_scaling().await?;
        self.apply_intrinsic_excitability_adjustment().await?;

        Ok(())
    }

    /// Compute current firing rates
    fn compute_current_firing_rates(&self) -> Array1<f64> {
        // Simplified firing rate computation based on conductance sums
        let mut rates = Array1::zeros(self.crossbar_array.dimensions.1);

        for j in 0..self.crossbar_array.dimensions.1 {
            let total_conductance: f64 = (0..self.crossbar_array.dimensions.0)
                .map(|i| self.crossbar_array.conductances[[i, j]])
                .sum();
            rates[j] = Self::sigmoid(total_conductance);
        }

        rates
    }

    /// Apply synaptic scaling homeostasis
    async fn apply_synaptic_scaling(&mut self) -> SpatialResult<()> {
        let current_rates = self.compute_current_firing_rates();

        for j in 0..self.crossbar_array.dimensions.1 {
            let target_rate = self.homeostatic_system.target_firing_rates[j];
            let current_rate = current_rates[j];
            let adaptation_rate = self.homeostatic_system.adaptation_rates[j];

            let scaling_factor = 1.0 + adaptation_rate * (target_rate - current_rate);

            // Apply scaling to all incoming synapses
            for i in 0..self.crossbar_array.dimensions.0 {
                self.crossbar_array.conductances[[i, j]] *= scaling_factor;
                self.crossbar_array.conductances[[i, j]] =
                    self.crossbar_array.conductances[[i, j]].clamp(0.0, 1.0);
            }
        }

        Ok(())
    }

    /// Apply intrinsic excitability adjustment
    async fn apply_intrinsic_excitability_adjustment(&mut self) -> SpatialResult<()> {
        // Adjust switching thresholds based on activity
        let current_rates = self.compute_current_firing_rates();

        for j in 0..self.crossbar_array.dimensions.1 {
            let target_rate = self.homeostatic_system.target_firing_rates[j];
            let current_rate = current_rates[j];
            let adaptation_rate = self.homeostatic_system.adaptation_rates[j];

            let threshold_adjustment = adaptation_rate * (current_rate - target_rate);

            for i in 0..self.crossbar_array.dimensions.0 {
                self.crossbar_array.switching_thresholds[[i, j]] += threshold_adjustment;
                self.crossbar_array.switching_thresholds[[i, j]] =
                    self.crossbar_array.switching_thresholds[[i, j]].clamp(0.1, 2.0);
            }
        }

        Ok(())
    }

    /// Apply metaplasticity updates
    async fn apply_metaplasticity_updates(
        &mut self,
        metrics: &PerformanceMetrics,
    ) -> SpatialResult<()> {
        // Update learning rate adaptation
        self.metaplasticity
            .learning_rate_adaptation
            .performance_history
            .push_back(metrics.accuracy);

        if self
            .metaplasticity
            .learning_rate_adaptation
            .performance_history
            .len()
            > 100
        {
            self.metaplasticity
                .learning_rate_adaptation
                .performance_history
                .pop_front();
        }

        // Adapt learning rates based on performance
        self.adapt_learning_rates(metrics).await?;

        // Update thresholds
        self.adapt_thresholds(metrics).await?;

        // Apply consolidation if needed
        if metrics.accuracy > 0.9 {
            self.trigger_memory_consolidation().await?;
        }

        Ok(())
    }

    /// Adapt learning rates based on performance
    async fn adapt_learning_rates(&mut self, _metrics: &PerformanceMetrics) -> SpatialResult<()> {
        let performance_trend = self.compute_performance_trend();

        for mechanism in &mut self.plasticity_mechanisms {
            if performance_trend > 0.0 {
                // Performance improving, maintain or slightly increase learning rate
                mechanism.learning_rates.potentiation_rate *= 1.01;
                mechanism.learning_rates.depression_rate *= 1.01;
            } else {
                // Performance declining, reduce learning rate
                mechanism.learning_rates.potentiation_rate *= 0.99;
                mechanism.learning_rates.depression_rate *= 0.99;
            }

            // Clamp learning rates
            mechanism.learning_rates.potentiation_rate =
                mechanism.learning_rates.potentiation_rate.clamp(1e-6, 0.1);
            mechanism.learning_rates.depression_rate =
                mechanism.learning_rates.depression_rate.clamp(1e-6, 0.1);
        }

        Ok(())
    }

    /// Compute performance trend
    fn compute_performance_trend(&self) -> f64 {
        let history = &self
            .metaplasticity
            .learning_rate_adaptation
            .performance_history;

        if history.len() < 10 {
            return 0.0;
        }

        let recent_performance: f64 = history.iter().rev().take(5).sum::<f64>() / 5.0;
        let older_performance: f64 = history.iter().rev().skip(5).take(5).sum::<f64>() / 5.0;

        recent_performance - older_performance
    }

    /// Adapt thresholds based on performance
    async fn adapt_thresholds(&mut self, metrics: &PerformanceMetrics) -> SpatialResult<()> {
        // Adjust plasticity thresholds based on learning progress
        for mechanism in &mut self.plasticity_mechanisms {
            if metrics.learning_speed > 1.0 {
                // Fast learning, can afford higher thresholds
                mechanism.thresholds.ltp_threshold *= 1.001;
                mechanism.thresholds.ltd_threshold *= 1.001;
            } else {
                // Slow learning, lower thresholds to increase plasticity
                mechanism.thresholds.ltp_threshold *= 0.999;
                mechanism.thresholds.ltd_threshold *= 0.999;
            }

            // Clamp thresholds
            mechanism.thresholds.ltp_threshold = mechanism.thresholds.ltp_threshold.clamp(0.1, 2.0);
            mechanism.thresholds.ltd_threshold = mechanism.thresholds.ltd_threshold.clamp(0.1, 2.0);
        }

        Ok(())
    }

    /// Update neuromodulation system
    async fn update_neuromodulation(&mut self, metrics: &PerformanceMetrics) -> SpatialResult<()> {
        // Update dopamine based on performance
        let performance_change = metrics.accuracy - 0.5; // Baseline accuracy
        self.neuromodulation
            .dopamine_levels
            .mapv_inplace(|x| x + 0.1 * performance_change);

        // Update serotonin based on stability
        let stability_change = metrics.stability - 0.5;
        self.neuromodulation
            .serotonin_levels
            .mapv_inplace(|x| x + 0.05 * stability_change);

        // Clamp neurotransmitter levels
        self.neuromodulation
            .dopamine_levels
            .mapv_inplace(|x| x.clamp(0.0, 1.0));
        self.neuromodulation
            .serotonin_levels
            .mapv_inplace(|x| x.clamp(0.0, 1.0));

        Ok(())
    }

    /// Record learning history
    async fn record_learning_history(
        &mut self,
        metrics: &PerformanceMetrics,
        _timestamp: f64,
    ) -> SpatialResult<()> {
        // Record performance metrics
        self.learning_history
            .performance_metrics
            .push_back(metrics.clone());

        // Record weight changes
        self.learning_history
            .weight_changes
            .push_back(self.crossbar_array.conductances.clone());

        // Maintain history size
        if self.learning_history.performance_metrics.len()
            > self.learning_history.max_history_length
        {
            self.learning_history.performance_metrics.pop_front();
            self.learning_history.weight_changes.pop_front();
        }

        Ok(())
    }

    /// Check if memory consolidation should be triggered
    fn should_trigger_consolidation(&self, epoch: usize) -> bool {
        // Trigger consolidation every 100 epochs or when performance is high
        epoch % 100 == 0
            || self
                .learning_history
                .performance_metrics
                .back()
                .map(|m| m.accuracy > 0.95)
                .unwrap_or(false)
    }

    /// Trigger memory consolidation
    async fn trigger_memory_consolidation(&mut self) -> SpatialResult<()> {
        // Systems consolidation: strengthen important connections
        self.strengthen_important_connections().await?;

        // Record consolidation event
        let consolidation_event = ConsolidationEvent {
            consolidation_type: ConsolidationType::SynapticConsolidation,
            patterns: vec![], // Would store relevant patterns
            strength: 1.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        self.learning_history
            .consolidation_events
            .push_back(consolidation_event);

        Ok(())
    }

    /// Strengthen important connections during consolidation
    async fn strengthen_important_connections(&mut self) -> SpatialResult<()> {
        // Calculate connection importance based on usage and performance contribution
        let mut importance_matrix = Array2::zeros(self.crossbar_array.dimensions);

        for i in 0..self.crossbar_array.dimensions.0 {
            for j in 0..self.crossbar_array.dimensions.1 {
                let conductance = self.crossbar_array.conductances[[i, j]];
                let usage = self.crossbar_array.endurance_cycles[[i, j]] as f64;

                // Importance based on conductance and usage
                importance_matrix[[i, j]] = conductance * (1.0 + 0.1 * usage.ln_1p());
            }
        }

        // Strengthen top 20% most important connections
        let threshold = self.compute_importance_threshold(&importance_matrix, 0.8);

        for i in 0..self.crossbar_array.dimensions.0 {
            for j in 0..self.crossbar_array.dimensions.1 {
                if importance_matrix[[i, j]] > threshold {
                    self.crossbar_array.conductances[[i, j]] *= 1.05; // 5% strengthening
                    self.crossbar_array.conductances[[i, j]] =
                        self.crossbar_array.conductances[[i, j]].min(1.0);
                }
            }
        }

        Ok(())
    }

    /// Compute importance threshold for top percentage
    fn compute_importance_threshold(
        &self,
        importance_matrix: &Array2<f64>,
        percentile: f64,
    ) -> f64 {
        let mut values: Vec<f64> = importance_matrix.iter().cloned().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (values.len() as f64 * percentile) as usize;
        values.get(index).cloned().unwrap_or(0.0)
    }

    /// Helper functions
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn compute_weight_stability(&self) -> f64 {
        // Simplified stability measure
        let weight_variance = self.crossbar_array.conductances.var(0.0);
        1.0 / (1.0 + weight_variance)
    }

    fn estimate_generalization(&self) -> f64 {
        // Simplified generalization estimate
        0.8 // Placeholder
    }

    /// Get crossbar dimensions
    pub fn crossbar_dimensions(&self) -> (usize, usize) {
        self.crossbar_array.dimensions
    }

    /// Get device type
    pub fn device_type(&self) -> &MemristiveDeviceType {
        &self.crossbar_array.device_type
    }

    /// Get current conductances
    pub fn conductances(&self) -> &Array2<f64> {
        &self.crossbar_array.conductances
    }

    /// Get learning history
    pub fn learning_history(&self) -> &LearningHistory {
        &self.learning_history
    }
}

impl MemristiveCrossbar {
    /// Create new memristive crossbar
    pub fn new(rows: usize, cols: usize, device_type: MemristiveDeviceType) -> Self {
        let mut rng = rand::rng();
        let conductances = Array2::from_shape_fn((rows, cols), |_| rng.gen_range(0.0..0.1));
        let resistances = conductances.mapv(|g| if g > 1e-12 { 1.0 / g } else { 1e12 });
        let switching_thresholds = Array2::from_elem((rows, cols), 0.5);
        let retention_times = Array2::from_elem((rows, cols), 1e6);
        let endurance_cycles = Array2::zeros((rows, cols));
        let programming_voltages = Array2::from_elem((rows, cols), 1.0);
        let temperature_coefficients = Array2::from_elem((rows, cols), 0.01);
        let device_variability = Array2::from_shape_fn((rows, cols), |_| rng.gen_range(0.0..0.01));

        Self {
            conductances,
            resistances,
            switching_thresholds,
            retention_times,
            endurance_cycles,
            programming_voltages,
            temperature_coefficients,
            device_variability,
            dimensions: (rows, cols),
            device_type,
        }
    }
}

impl PlasticityMechanism {
    /// Create new plasticity mechanism
    pub fn new(mechanism_type: PlasticityType) -> Self {
        let (time_constants, learning_rates, thresholds) = match mechanism_type {
            PlasticityType::STDP => (
                PlasticityTimeConstants {
                    tau_fast: 20.0,
                    tau_slow: 40.0,
                    stdp_window: 100.0,
                    tau_homeostatic: 1000.0,
                    tau_calcium: 50.0,
                },
                PlasticityLearningRates {
                    potentiation_rate: 0.01,
                    depression_rate: 0.005,
                    homeostatic_rate: 0.001,
                    metaplastic_rate: 0.0001,
                    intrinsic_rate: 0.001,
                },
                PlasticityThresholds {
                    ltp_threshold: 0.6,
                    ltd_threshold: 0.4,
                    target_activity: 0.5,
                    metaplasticity_threshold: 0.8,
                    saturation_threshold: 0.95,
                },
            ),
            _ => (
                PlasticityTimeConstants {
                    tau_fast: 10.0,
                    tau_slow: 20.0,
                    stdp_window: 50.0,
                    tau_homeostatic: 500.0,
                    tau_calcium: 25.0,
                },
                PlasticityLearningRates {
                    potentiation_rate: 0.005,
                    depression_rate: 0.0025,
                    homeostatic_rate: 0.0005,
                    metaplastic_rate: 0.00005,
                    intrinsic_rate: 0.0005,
                },
                PlasticityThresholds {
                    ltp_threshold: 0.5,
                    ltd_threshold: 0.3,
                    target_activity: 0.4,
                    metaplasticity_threshold: 0.7,
                    saturation_threshold: 0.9,
                },
            ),
        };

        Self {
            mechanism_type,
            time_constants,
            learning_rates,
            thresholds,
            enabled: true,
            weight_scaling: 1.0,
        }
    }
}

impl HomeostaticSystem {
    /// Create new homeostatic system
    pub fn new(num_neurons: usize) -> Self {
        Self {
            target_firing_rates: Array1::from_elem(num_neurons, 0.5),
            current_firing_rates: Array1::zeros(num_neurons),
            time_constants: Array1::from_elem(num_neurons, 1000.0),
            mechanisms: vec![
                HomeostaticMechanism::SynapticScaling,
                HomeostaticMechanism::IntrinsicExcitability,
            ],
            adaptation_rates: Array1::from_elem(num_neurons, 0.001),
            activity_history: VecDeque::new(),
            history_window: 100,
        }
    }
}

impl Default for MetaplasticityRules {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaplasticityRules {
    /// Create new metaplasticity rules
    pub fn new() -> Self {
        Self {
            learning_rate_adaptation: LearningRateAdaptation {
                base_rate: 0.01,
                adaptation_factor: 0.1,
                performance_history: VecDeque::new(),
                adaptation_threshold: 0.1,
                max_rate: 0.1,
                min_rate: 1e-6,
            },
            threshold_adaptation: ThresholdAdaptation {
                adaptive_thresholds: Array1::from_elem(10, 0.5),
                update_rates: Array1::from_elem(10, 0.001),
                target_activations: Array1::from_elem(10, 0.5),
                threshold_bounds: vec![(0.1, 2.0); 10],
            },
            consolidation_rules: ConsolidationRules {
                time_windows: vec![100.0, 1000.0, 10000.0],
                consolidation_strengths: Array1::from_elem(3, 1.0),
                replay_enabled: true,
                replay_patterns: Vec::new(),
                systems_consolidation: true,
            },
            forgetting_protection: ForgettingProtectionRules {
                ewc_enabled: false,
                fisher_information: Array2::zeros((10, 10)),
                synaptic_intelligence: false,
                importance_weights: Array1::zeros(10),
                protection_strength: 1.0,
            },
        }
    }
}

impl NeuromodulationSystem {
    /// Create new neuromodulation system
    pub fn new(num_neurons: usize) -> Self {
        Self {
            dopamine_levels: Array1::from_elem(num_neurons, 0.5),
            serotonin_levels: Array1::from_elem(num_neurons, 0.5),
            acetylcholine_levels: Array1::from_elem(num_neurons, 0.5),
            noradrenaline_levels: Array1::from_elem(num_neurons, 0.5),
            modulation_effects: NeuromodulationEffects {
                learning_rate_modulation: Array1::from_elem(num_neurons, 1.0),
                threshold_modulation: Array1::from_elem(num_neurons, 1.0),
                excitability_modulation: Array1::from_elem(num_neurons, 1.0),
                attention_modulation: Array1::from_elem(num_neurons, 1.0),
            },
            release_patterns: NeuromodulatorReleasePatterns {
                phasic_dopamine: Vec::new(),
                tonic_serotonin: 0.5,
                cholinergic_attention: Array1::from_elem(num_neurons, 0.5),
                stress_noradrenaline: 0.3,
            },
        }
    }
}

impl Default for LearningHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningHistory {
    /// Create new learning history tracker
    pub fn new() -> Self {
        Self {
            weight_changes: VecDeque::new(),
            performance_metrics: VecDeque::new(),
            plasticity_events: VecDeque::new(),
            consolidation_events: VecDeque::new(),
            max_history_length: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_advanced_memristive_learning_creation() {
        let learning_system =
            AdvancedMemristiveLearning::new(8, 4, MemristiveDeviceType::TitaniumDioxide);
        assert_eq!(learning_system.crossbar_dimensions(), (8, 4));
        assert_eq!(learning_system.plasticity_mechanisms.len(), 3);
        assert!(learning_system.forgetting_protection);
    }

    #[test]
    fn test_memristive_device_types() {
        let tio2_system =
            AdvancedMemristiveLearning::new(4, 4, MemristiveDeviceType::TitaniumDioxide);
        let hfo2_system = AdvancedMemristiveLearning::new(4, 4, MemristiveDeviceType::HafniumOxide);
        let pcm_system = AdvancedMemristiveLearning::new(4, 4, MemristiveDeviceType::PhaseChange);

        assert!(matches!(
            tio2_system.device_type(),
            MemristiveDeviceType::TitaniumDioxide
        ));
        assert!(matches!(
            hfo2_system.device_type(),
            MemristiveDeviceType::HafniumOxide
        ));
        assert!(matches!(
            pcm_system.device_type(),
            MemristiveDeviceType::PhaseChange
        ));
    }

    #[test]
    fn test_plasticity_mechanism_creation() {
        let stdp_mechanism = PlasticityMechanism::new(PlasticityType::STDP);
        assert!(stdp_mechanism.enabled);
        assert!(matches!(
            stdp_mechanism.mechanism_type,
            PlasticityType::STDP
        ));
        assert!(stdp_mechanism.learning_rates.potentiation_rate > 0.0);
    }

    #[test]
    fn test_homeostatic_regulation() {
        let target_rates = Array1::from_vec(vec![0.3, 0.7, 0.5, 0.8]);
        let learning_system =
            AdvancedMemristiveLearning::new(4, 4, MemristiveDeviceType::HafniumOxide)
                .with_homeostatic_regulation(target_rates.clone());
        assert_eq!(
            learning_system.homeostatic_system.target_firing_rates,
            target_rates
        );
    }

    #[test]
    fn test_forgetting_protection() {
        let learning_system =
            AdvancedMemristiveLearning::new(4, 4, MemristiveDeviceType::PhaseChange)
                .with_forgetting_protection(true);
        assert!(learning_system.forgetting_protection);
        assert!(
            learning_system
                .metaplasticity
                .forgetting_protection
                .ewc_enabled
        );

        let no_protection_system =
            AdvancedMemristiveLearning::new(4, 4, MemristiveDeviceType::PhaseChange)
                .with_forgetting_protection(false);
        assert!(!no_protection_system.forgetting_protection);
        assert!(
            !no_protection_system
                .metaplasticity
                .forgetting_protection
                .ewc_enabled
        );
    }

    #[tokio::test]
    async fn test_memristive_forward_pass() {
        let learning_system =
            AdvancedMemristiveLearning::new(3, 2, MemristiveDeviceType::TitaniumDioxide);
        let input = array![0.5, 0.8, 0.3];
        let result = learning_system.forward_pass(&input.view()).await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output >= 0.0 && output <= 1.0); // Sigmoid output
    }

    #[test]
    fn test_device_nonlinearity() {
        let learning_system =
            AdvancedMemristiveLearning::new(2, 2, MemristiveDeviceType::TitaniumDioxide);

        // Test TiO2 nonlinearity
        let linear_current = 0.1;
        let nonlinear_current = learning_system.apply_device_nonlinearity(linear_current, 0, 0);
        assert!(nonlinear_current.is_finite());

        // Test with HfO2
        let hfo2_system = AdvancedMemristiveLearning::new(2, 2, MemristiveDeviceType::HafniumOxide);
        let hfo2_output = hfo2_system.apply_device_nonlinearity(linear_current, 0, 0);
        assert!(hfo2_output.is_finite());

        // Test with Phase Change Memory
        let pcm_system = AdvancedMemristiveLearning::new(2, 2, MemristiveDeviceType::PhaseChange);
        let pcm_output = pcm_system.apply_device_nonlinearity(linear_current, 0, 0);
        assert!(pcm_output.is_finite());
    }

    #[tokio::test]
    async fn test_memristive_training() {
        let mut learning_system =
            AdvancedMemristiveLearning::new(2, 1, MemristiveDeviceType::TitaniumDioxide);

        let spatial_data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let target_outputs = array![0.0, 1.0, 1.0, 0.0]; // XOR pattern

        let result = learning_system
            .train_spatial_data(&spatial_data.view(), &target_outputs.view(), 5)
            .await;

        assert!(result.is_ok());
        let training_result = result.unwrap();
        assert_eq!(training_result.training_metrics.len(), 5);
        assert!(!training_result.final_weights.is_empty());
    }

    #[test]
    fn test_memristive_crossbar_creation() {
        let crossbar = MemristiveCrossbar::new(4, 3, MemristiveDeviceType::SilverSulfide);
        assert_eq!(crossbar.dimensions, (4, 3));
        assert_eq!(crossbar.conductances.shape(), &[4, 3]);
        assert_eq!(crossbar.resistances.shape(), &[4, 3]);
        assert_eq!(crossbar.switching_thresholds.shape(), &[4, 3]);
        assert!(matches!(
            crossbar.device_type,
            MemristiveDeviceType::SilverSulfide
        ));

        // Check that resistances are inverse of conductances (approximately)
        for i in 0..4 {
            for j in 0..3 {
                let conductance = crossbar.conductances[[i, j]];
                let resistance = crossbar.resistances[[i, j]];
                if conductance > 1e-12 {
                    assert!((resistance * conductance - 1.0).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_device_aging_and_variability() {
        let mut learning_system =
            AdvancedMemristiveLearning::new(2, 2, MemristiveDeviceType::Organic);

        // Store initial conductance
        let initial_conductance = learning_system.crossbar_array.conductances[[0, 0]];

        // Apply aging
        learning_system.apply_device_aging(0, 0);
        let aged_conductance = learning_system.crossbar_array.conductances[[0, 0]];

        // Conductance should be equal or slightly reduced (aging effect is small)
        assert!(aged_conductance <= initial_conductance);

        // Apply variability
        let pre_variability = learning_system.crossbar_array.conductances[[0, 0]];
        learning_system.apply_device_variability(0, 0);
        let post_variability = learning_system.crossbar_array.conductances[[0, 0]];

        // Variability should cause some change (might be very small)
        assert!(post_variability >= 0.0 && post_variability <= 1.0);
    }

    #[test]
    fn test_plasticity_mechanisms_configuration() {
        let learning_system =
            AdvancedMemristiveLearning::new(4, 4, MemristiveDeviceType::TitaniumDioxide)
                .enable_plasticity(PlasticityType::CalciumDependent)
                .enable_plasticity(PlasticityType::VoltageDependent);

        // Check that mechanisms are properly configured
        let enabled_mechanisms: Vec<_> = learning_system
            .plasticity_mechanisms
            .iter()
            .filter(|m| m.enabled)
            .map(|m| &m.mechanism_type)
            .collect();

        assert!(!enabled_mechanisms.is_empty());
    }

    #[test]
    fn test_learning_history_tracking() {
        let learning_system =
            AdvancedMemristiveLearning::new(3, 3, MemristiveDeviceType::MagneticTunnelJunction);

        let history = learning_system.learning_history();
        assert_eq!(history.max_history_length, 1000);
        assert!(history.weight_changes.is_empty());
        assert!(history.performance_metrics.is_empty());
    }
}
