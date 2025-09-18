//! Neuromorphic Computing Optimization
//!
//! This module implements optimization algorithms specifically designed for neuromorphic
//! computing platforms, including spike-based optimization, event-driven parameter updates,
//! and energy-efficient optimization strategies for neuromorphic chips.

use crate::error::Result;
use crate::optimizers::Optimizer;
use ndarray::{Array1, Array2, ArrayBase, Data, DataMut, Dimension};
use num_traits::Float;
use std::fmt::Debug;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

pub mod spike_based;
pub mod event_driven;
pub mod energy_efficient;

// Re-export key types
pub use spike_based::{SpikingOptimizer, SpikingConfig, SpikeTrainOptimizer};
pub use event_driven::{EventDrivenOptimizer, EventDrivenConfig, EventType};
pub use energy_efficient::{EnergyEfficientOptimizer, EnergyOptimizationStrategy, EnergyBudget};

/// Neuromorphic computing platform types
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum NeuromorphicPlatform {
    /// Intel Loihi neuromorphic chip
    IntelLoihi,
    
    /// SpiNNaker platform
    SpiNNaker,
    
    /// IBM TrueNorth
    IBMTrueNorth,
    
    /// BrainChip Akida
    BrainChipAkida,
    
    /// University research platforms
    Research,
    
    /// Custom neuromorphic hardware
    Custom(String)}

/// Neuromorphic optimization configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct NeuromorphicConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Target neuromorphic platform
    pub platform: NeuromorphicPlatform,
    
    /// Spike-based optimization settings
    pub spike_config: SpikingConfig<T>,
    
    /// Event-driven optimization settings
    pub event_config: EventDrivenConfig<T>,
    
    /// Energy optimization settings
    pub energy_config: EnergyOptimizationConfig<T>,
    
    /// Enable temporal coding
    pub temporal_coding: bool,
    
    /// Enable rate coding
    pub rate_coding: bool,
    
    /// Spike timing dependent plasticity (STDP) parameters
    pub stdp_config: STDPConfig<T>,
    
    /// Membrane potential dynamics
    pub membrane_dynamics: MembraneDynamicsConfig<T>,
    
    /// Synaptic plasticity model
    pub plasticity_model: PlasticityModel,
    
    /// Enable homeostatic mechanisms
    pub homeostatic_plasticity: bool,
    
    /// Enable metaplasticity
    pub metaplasticity: bool,
    
    /// Population dynamics configuration
    pub population_config: PopulationConfig}

/// Spike Timing Dependent Plasticity configuration
#[derive(Debug, Clone)]
pub struct STDPConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Learning rate for potentiation
    pub learning_rate_pot: T,
    
    /// Learning rate for depression
    pub learning_rate_dep: T,
    
    /// Time constant for potentiation (ms)
    pub tau_pot: T,
    
    /// Time constant for depression (ms)
    pub tau_dep: T,
    
    /// Maximum weight value
    pub weight_max: T,
    
    /// Minimum weight value
    pub weight_min: T,
    
    /// Enable triplet STDP
    pub enable_triplet: bool,
    
    /// Triplet learning rate
    pub triplet_learning_rate: T}

/// Membrane potential dynamics configuration
#[derive(Debug, Clone)]
pub struct MembraneDynamicsConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Membrane time constant (ms)
    pub tau_membrane: T,
    
    /// Resting potential (mV)
    pub resting_potential: T,
    
    /// Threshold potential (mV)
    pub threshold_potential: T,
    
    /// Reset potential (mV)
    pub reset_potential: T,
    
    /// Refractory period (ms)
    pub refractory_period: T,
    
    /// Capacitance (pF)
    pub capacitance: T,
    
    /// Leak conductance (nS)
    pub leak_conductance: T,
    
    /// Enable adaptive threshold
    pub adaptive_threshold: bool,
    
    /// Threshold adaptation time constant
    pub threshold_adaptation_tau: T}

/// Synaptic plasticity models
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum PlasticityModel {
    /// Hebbian plasticity
    Hebbian,
    
    /// Anti-Hebbian plasticity
    AntiHebbian,
    
    /// Spike Timing Dependent Plasticity
    STDP,
    
    /// Triplet STDP
    TripletSTDP,
    
    /// Voltage-dependent plasticity
    VoltageDependentSTDP,
    
    /// Calcium-based plasticity
    CalciumBased,
    
    /// BCM (Bienenstock-Cooper-Munro) rule
    BCM,
    
    /// Oja's rule
    Oja}

/// Population-level configuration
#[derive(Debug, Clone)]
pub struct PopulationConfig {
    /// Population size
    pub population_size: usize,
    
    /// Enable lateral inhibition
    pub lateral_inhibition: bool,
    
    /// Inhibition strength
    pub inhibition_strength: f64,
    
    /// Enable winner-take-all dynamics
    pub winner_take_all: bool,
    
    /// Population coding strategy
    pub coding_strategy: PopulationCodingStrategy,
    
    /// Enable population bursting
    pub enable_bursting: bool,
    
    /// Synchronization mechanisms
    pub synchronization: SynchronizationMechanism}

/// Population coding strategies
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum PopulationCodingStrategy {
    /// Distributed coding
    Distributed,
    
    /// Sparse coding
    Sparse,
    
    /// Local coding
    Local,
    
    /// Vector coding
    Vector,
    
    /// Rank order coding
    RankOrder}

/// Synchronization mechanisms
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum SynchronizationMechanism {
    /// No synchronization
    None,
    
    /// Global clock
    GlobalClock,
    
    /// Phase-locked loops
    PhaseLocked,
    
    /// Adaptive synchronization
    Adaptive,
    
    /// Network oscillations
    NetworkOscillations}

/// Energy optimization configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct EnergyOptimizationConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Energy budget (nJ per operation)
    pub energy_budget: T,
    
    /// Energy optimization strategy
    pub strategy: EnergyOptimizationStrategy,
    
    /// Enable dynamic voltage scaling
    pub dynamic_voltage_scaling: bool,
    
    /// Enable clock gating
    pub clock_gating: bool,
    
    /// Enable power gating
    pub power_gating: bool,
    
    /// Sleep mode configuration
    pub sleep_mode_config: SleepModeConfig<T>,
    
    /// Energy monitoring frequency
    pub monitoring_frequency: Duration,
    
    /// Thermal management
    pub thermal_management: ThermalManagementConfig<T>}

/// Sleep mode configuration for energy efficiency
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SleepModeConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Enable sleep mode
    pub enable_sleep_mode: bool,
    
    /// Sleep threshold (inactivity time)
    pub sleep_threshold: Duration,
    
    /// Wake-up time (ms)
    pub wakeup_time: T,
    
    /// Sleep energy consumption (nW)
    pub sleep_power: T,
    
    /// Wake-up energy cost (nJ)
    pub wakeup_energy: T}

/// Thermal management configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ThermalManagementConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Enable thermal management
    pub enable_thermal_management: bool,
    
    /// Target temperature (°C)
    pub target_temperature: T,
    
    /// Maximum temperature (°C)
    pub max_temperature: T,
    
    /// Thermal time constant (s)
    pub thermal_time_constant: T,
    
    /// Thermal throttling strategy
    pub throttling_strategy: ThermalThrottlingStrategy}

/// Thermal throttling strategies
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum ThermalThrottlingStrategy {
    /// Frequency scaling
    FrequencyScaling,
    
    /// Voltage scaling
    VoltageScaling,
    
    /// Activity reduction
    ActivityReduction,
    
    /// Selective shutdown
    SelectiveShutdown,
    
    /// Dynamic load balancing
    DynamicLoadBalancing}

/// Spike representation for neuromorphic optimization
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Spike<T: Float + Debug + Send + Sync + 'static> {
    /// Neuron ID
    pub neuron_id: usize,
    
    /// Spike time (ms)
    pub time: T,
    
    /// Spike amplitude (mV)
    pub amplitude: T,
    
    /// Spike width (ms)
    pub width: Option<T>,
    
    /// Associated synapse weight
    pub weight: T,
    
    /// Presynaptic neuron ID
    pub presynaptic_id: Option<usize>,
    
    /// Postsynaptic neuron ID
    pub postsynaptic_id: Option<usize>}

/// Spike train representation
#[derive(Debug, Clone)]
pub struct SpikeTrain<T: Float + Debug + Send + Sync + 'static> {
    /// Neuron ID
    pub neuron_id: usize,
    
    /// Spike times
    pub spike_times: Vec<T>,
    
    /// Inter-spike intervals
    pub inter_spike_intervals: Vec<T>,
    
    /// Firing rate (Hz)
    pub firing_rate: T,
    
    /// Spike train duration (ms)
    pub duration: T,
    
    /// Spike count
    pub spike_count: usize}

impl<T: Float + Debug + Send + Sync + 'static> SpikeTrain<T> {
    /// Create a new spike train from spike times
    pub fn new(_neuron_id: usize, spiketimes: Vec<T>) -> Self {
        let spike_count = spike_times.len();
        let duration = if spike_count > 0 {
            spike_times[spike_count - 1] - spike_times[0]
        } else {
            T::zero()
        };
        
        let firing_rate = if duration > T::zero() {
            num_traits::cast::cast(spike_count).unwrap_or_else(|| T::zero()) / (duration / num_traits::cast::cast(1000.0).unwrap_or_else(|| T::zero()))
        } else {
            T::zero()
        };
        
        let inter_spike_intervals = if spike_count > 1 {
            spike_times.windows(2)
                .map(|w| w[1] - w[0])
                .collect()
        } else {
            Vec::new()
        };
        
        Self {
            neuron_id,
            spike_times,
            inter_spike_intervals,
            firing_rate,
            duration,
            spike_count}
    }
    
    /// Calculate coefficient of variation of inter-spike intervals
    pub fn coefficient_of_variation(&self) -> T {
        if self.inter_spike_intervals.len() < 2 {
            return T::zero();
        }
        
        let mean = self.inter_spike_intervals.iter().cloned().sum::<T>() / 
            T::from(self.inter_spike_intervals.len()).unwrap();
        
        let variance = self.inter_spike_intervals.iter()
            .map(|&isi| (isi - mean) * (isi - mean))
            .sum::<T>() / T::from(self.inter_spike_intervals.len()).unwrap();
        
        variance.sqrt() / mean
    }
    
    /// Calculate local variation measure
    pub fn local_variation(&self) -> T {
        if self.inter_spike_intervals.len() < 2 {
            return T::zero();
        }
        
        let mut lv_sum = T::zero();
        for window in self.inter_spike_intervals.windows(2) {
            let isi1 = window[0];
            let isi2 = window[1];
            let diff = isi1 - isi2;
            let sum = isi1 + isi2;
            
            if sum > T::zero() {
                lv_sum = lv_sum + (diff * diff) / (sum * sum);
            }
        }
        
        let three = num_traits::cast::cast(3.0).unwrap_or_else(|| T::zero());
        three * lv_sum / T::from(self.inter_spike_intervals.len() - 1).unwrap()
    }
}

/// Event-driven update representation
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct NeuromorphicEvent<T: Float + Debug + Send + Sync + 'static> {
    /// Event type
    pub event_type: EventType,
    
    /// Event timestamp
    pub timestamp: Instant,
    
    /// Source neuron
    pub source_neuron: usize,
    
    /// Target neuron
    pub target_neuron: Option<usize>,
    
    /// Event value/weight
    pub value: T,
    
    /// Energy cost of processing this event
    pub energy_cost: T,
    
    /// Priority level
    pub priority: EventPriority}

/// Event priority levels for neuromorphic processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[allow(dead_code)]
pub enum EventPriority {
    Low,
    Normal,
    High,
    Critical,
    RealTime}

/// Neuromorphic optimization metrics
#[derive(Debug, Clone)]
pub struct NeuromorphicMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Total spikes processed
    pub total_spikes: usize,
    
    /// Average firing rate (Hz)
    pub average_firing_rate: T,
    
    /// Energy consumption (nJ)
    pub energy_consumption: T,
    
    /// Power consumption (nW)
    pub power_consumption: T,
    
    /// Spike timing precision (ms)
    pub timing_precision: T,
    
    /// Synaptic operations per second
    pub synaptic_ops_per_sec: T,
    
    /// Plasticity events per second
    pub plasticity_events_per_sec: T,
    
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: T,
    
    /// Thermal efficiency score
    pub thermal_efficiency: T,
    
    /// Network synchronization measure
    pub network_synchronization: T}

impl<T: Float + Debug + Send + Sync + 'static> Default for NeuromorphicMetrics<T> {
    fn default() -> Self {
        Self {
            total_spikes: 0,
            average_firing_rate: T::zero(),
            energy_consumption: T::zero(),
            power_consumption: T::zero(),
            timing_precision: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()), // 0.1ms default
            synaptic_ops_per_sec: T::zero(),
            plasticity_events_per_sec: T::zero(),
            memory_bandwidth_utilization: T::zero(),
            thermal_efficiency: T::one(),
            network_synchronization: T::zero()}
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for NeuromorphicConfig<T> {
    fn default() -> Self {
        Self {
            platform: NeuromorphicPlatform::IntelLoihi,
            spike_config: SpikingConfig::default(),
            event_config: EventDrivenConfig::default(),
            energy_config: EnergyOptimizationConfig::default(),
            temporal_coding: true,
            rate_coding: false,
            stdp_config: STDPConfig::default(),
            membrane_dynamics: MembraneDynamicsConfig::default(),
            plasticity_model: PlasticityModel::STDP,
            homeostatic_plasticity: false,
            metaplasticity: false,
            population_config: PopulationConfig::default()}
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for STDPConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate_pot: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            learning_rate_dep: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            tau_pot: num_traits::cast::cast(20.0).unwrap_or_else(|| T::zero()),
            tau_dep: num_traits::cast::cast(20.0).unwrap_or_else(|| T::zero()),
            weight_max: T::one(),
            weight_min: T::zero(),
            enable_triplet: false,
            triplet_learning_rate: num_traits::cast::cast(0.001).unwrap_or_else(|| T::zero())}
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for MembraneDynamicsConfig<T> {
    fn default() -> Self {
        Self {
            tau_membrane: num_traits::cast::cast(20.0).unwrap_or_else(|| T::zero()),
            resting_potential: num_traits::cast::cast(-70.0).unwrap_or_else(|| T::zero()),
            threshold_potential: num_traits::cast::cast(-55.0).unwrap_or_else(|| T::zero()),
            reset_potential: num_traits::cast::cast(-80.0).unwrap_or_else(|| T::zero()),
            refractory_period: num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()),
            capacitance: num_traits::cast::cast(100.0).unwrap_or_else(|| T::zero()),
            leak_conductance: num_traits::cast::cast(10.0).unwrap_or_else(|| T::zero()),
            adaptive_threshold: false,
            threshold_adaptation_tau: num_traits::cast::cast(100.0).unwrap_or_else(|| T::zero())}
    }
}

impl Default for PopulationConfig {
    fn default() -> Self {
        Self {
            population_size: 1000,
            lateral_inhibition: false,
            inhibition_strength: 0.1,
            winner_take_all: false,
            coding_strategy: PopulationCodingStrategy::Distributed,
            enable_bursting: false,
            synchronization: SynchronizationMechanism::None}
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for EnergyOptimizationConfig<T> {
    fn default() -> Self {
        Self {
            energy_budget: num_traits::cast::cast(10.0).unwrap_or_else(|| T::zero()), // 10 nJ per operation
            strategy: EnergyOptimizationStrategy::DynamicVoltageScaling,
            dynamic_voltage_scaling: true,
            clock_gating: true,
            power_gating: false,
            sleep_mode_config: SleepModeConfig::default(),
            monitoring_frequency: Duration::from_millis(100),
            thermal_management: ThermalManagementConfig::default()}
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for SleepModeConfig<T> {
    fn default() -> Self {
        Self {
            enable_sleep_mode: true,
            sleep_threshold: Duration::from_millis(100),
            wakeup_time: num_traits::cast::cast(1.0).unwrap_or_else(|| T::zero()),
            sleep_power: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            wakeup_energy: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero())}
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for ThermalManagementConfig<T> {
    fn default() -> Self {
        Self {
            enable_thermal_management: true,
            target_temperature: num_traits::cast::cast(65.0).unwrap_or_else(|| T::zero()),
            max_temperature: num_traits::cast::cast(85.0).unwrap_or_else(|| T::zero()),
            thermal_time_constant: num_traits::cast::cast(10.0).unwrap_or_else(|| T::zero()),
            throttling_strategy: ThermalThrottlingStrategy::FrequencyScaling}
    }
}
