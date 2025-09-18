//! Topology Configuration Types and Enums
//!
//! This module contains all configuration-related types, enums, and constants
//! for TPU pod topology management. It provides the foundational types used
//! throughout the topology management system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::tpu::tpu_backend::DeviceId;

// Type aliases for topology management
pub type NodeId = u32;
pub type LinkLatency = f64;
pub type LinkBandwidth = f64;

/// Configuration for topology management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    /// Pod topology type
    pub topology_type: TopologyType,
    /// Number of devices in the pod
    pub device_count: usize,
    /// Number of nodes in the topology
    pub node_count: usize,
    /// Devices per node
    pub devices_per_node: usize,
    /// Inter-node connection type
    pub inter_node_connection: InterNodeConnection,
    /// Intra-node connection type
    pub intra_node_connection: IntraNodeConnection,
    /// Enable topology optimization
    pub enable_optimization: bool,
    /// Enable dynamic reconfiguration
    pub enable_dynamic_reconfig: bool,
    /// Network redundancy level
    pub redundancy_level: RedundancyLevel,
    /// Quality of service settings
    pub qos_settings: TopologyQoSSettings,
    /// Topology constraints
    pub constraints: TopologyConstraints,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
}

/// Types of pod topologies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TopologyType {
    /// Linear topology
    Linear,
    /// Ring topology
    Ring,
    /// Mesh topology
    Mesh { dimension: usize },
    /// Tree topology
    Tree { branching_factor: usize },
    /// Hypercube topology
    Hypercube { dimension: usize },
    /// Torus topology
    Torus { dimensions: Vec<usize> },
    /// Fat tree topology
    FatTree { levels: usize, branching: usize },
    /// Dragonfly topology
    Dragonfly { groups: usize, routers_per_group: usize },
    /// Custom topology with user-defined connections
    Custom { adjacency_matrix: Vec<Vec<bool>> },
}

/// Inter-node connection types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterNodeConnection {
    /// Ethernet connection
    Ethernet { speed_gbps: f64 },
    /// InfiniBand connection
    InfiniBand { speed_gbps: f64 },
    /// NVLink connection
    NVLink { speed_gbps: f64 },
    /// Custom high-speed interconnect
    CustomInterconnect { speed_gbps: f64, protocol: String },
}

/// Intra-node connection types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntraNodeConnection {
    /// PCIe connection
    PCIe { version: String, lanes: usize },
    /// NVLink connection
    NVLink { version: String, speed_gbps: f64 },
    /// Custom high-speed bus
    CustomBus { speed_gbps: f64, protocol: String },
}

/// Network redundancy levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RedundancyLevel {
    /// No redundancy
    None,
    /// Single path redundancy
    SinglePath,
    /// Dual path redundancy
    DualPath,
    /// Multiple path redundancy
    MultiPath { path_count: usize },
    /// Full mesh redundancy
    FullMesh,
}

/// Quality of service settings for topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyQoSSettings {
    /// Maximum acceptable latency (microseconds)
    pub max_latency: f64,
    /// Minimum required bandwidth (Gbps)
    pub min_bandwidth: f64,
    /// Required reliability (0.0 to 1.0)
    pub reliability: f64,
    /// Jitter tolerance (microseconds)
    pub jitter_tolerance: f64,
    /// Packet loss tolerance (0.0 to 1.0)
    pub packet_loss_tolerance: f64,
}

/// Topology constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConstraints {
    /// Maximum number of hops
    pub max_hops: Option<usize>,
    /// Maximum cable length (meters)
    pub max_cable_length: Option<f64>,
    /// Power budget constraints
    pub power_budget: PowerBudgetConstraints,
    /// Thermal constraints
    pub thermal_constraints: ThermalConstraints,
    /// Physical space constraints
    pub space_constraints: SpaceConstraints,
}

/// Power budget constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerBudgetConstraints {
    /// Maximum total power consumption (watts)
    pub max_total_power: f64,
    /// Maximum power per node (watts)
    pub max_power_per_node: f64,
    /// Power efficiency requirements
    pub efficiency_requirements: f64,
    /// Emergency power limits
    pub emergency_limits: EmergencyPowerLimits,
}

/// Emergency power limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyPowerLimits {
    /// Critical power level (watts)
    pub critical_level: f64,
    /// Emergency shutdown threshold (watts)
    pub shutdown_threshold: f64,
    /// Power reduction strategy
    pub reduction_strategy: PowerReductionStrategy,
}

/// Power reduction strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerReductionStrategy {
    /// Reduce frequency
    ReduceFrequency { target_reduction: f64 },
    /// Disable non-critical devices
    DisableDevices { priority_order: Vec<String> },
    /// Throttle computation
    ThrottleComputation { throttle_factor: f64 },
    /// Custom strategy
    Custom { strategy: String },
}

/// Thermal constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalConstraints {
    /// Maximum operating temperature (Celsius)
    pub max_temperature: f64,
    /// Maximum temperature gradient (Celsius/meter)
    pub max_temperature_gradient: f64,
    /// Cooling requirements
    pub cooling_requirements: CoolingRequirements,
    /// Thermal zones
    pub thermal_zones: Vec<ThermalZoneConstraints>,
}

/// Cooling requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingRequirements {
    /// Required cooling capacity (watts)
    pub required_capacity: f64,
    /// Cooling type
    pub cooling_type: CoolingType,
    /// Airflow requirements (CFM)
    pub airflow_cfm: f64,
    /// Redundancy requirements
    pub redundancy: CoolingRedundancy,
}

/// Cooling types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoolingType {
    /// Air cooling
    Air { fan_count: usize, cfm_per_fan: f64 },
    /// Liquid cooling
    Liquid { pump_capacity: f64, radiator_size: f64 },
    /// Immersion cooling
    Immersion { fluid_type: String, flow_rate: f64 },
    /// Hybrid cooling
    Hybrid { primary: Box<CoolingType>, secondary: Box<CoolingType> },
}

/// Cooling redundancy levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoolingRedundancy {
    /// No redundancy
    None,
    /// N+1 redundancy
    NPlusOne,
    /// N+2 redundancy
    NPlusTwo,
    /// Full redundancy
    Full,
}

/// Thermal zone constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalZoneConstraints {
    /// Zone identifier
    pub zone_id: String,
    /// Maximum temperature for this zone
    pub max_temp: f64,
    /// Target temperature for this zone
    pub target_temp: f64,
    /// Temperature tolerance
    pub tolerance: f64,
    /// Critical temperature threshold
    pub critical_threshold: f64,
}

/// Physical space constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceConstraints {
    /// Maximum rack units
    pub max_rack_units: usize,
    /// Available floor space (square meters)
    pub floor_space_sqm: f64,
    /// Maximum height (meters)
    pub max_height: f64,
    /// Weight constraints
    pub weight_constraints: WeightConstraints,
    /// Cable management requirements
    pub cable_management: CableManagementRequirements,
}

/// Weight constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightConstraints {
    /// Maximum total weight (kg)
    pub max_total_weight: f64,
    /// Maximum weight per rack unit (kg)
    pub max_weight_per_ru: f64,
    /// Floor loading capacity (kg/sqm)
    pub floor_loading_capacity: f64,
}

/// Cable management requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CableManagementRequirements {
    /// Cable tray capacity
    pub tray_capacity: usize,
    /// Maximum cable length
    pub max_cable_length: f64,
    /// Cable separation requirements
    pub separation_requirements: f64,
    /// Cable routing constraints
    pub routing_constraints: Vec<String>,
}

/// Performance requirements for topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Bandwidth requirements
    pub bandwidth: BandwidthRequirements,
    /// Latency requirements
    pub latency: LatencyRequirements,
    /// Scalability requirements
    pub scalability: ScalabilityRequirements,
    /// Fault tolerance requirements
    pub fault_tolerance: FaultToleranceRequirements,
}

/// Bandwidth requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthRequirements {
    /// Minimum aggregate bandwidth (Gbps)
    pub min_aggregate: f64,
    /// Minimum per-device bandwidth (Gbps)
    pub min_per_device: f64,
    /// Peak bandwidth requirements (Gbps)
    pub peak_bandwidth: f64,
    /// Sustained bandwidth requirements (Gbps)
    pub sustained_bandwidth: f64,
    /// Bandwidth utilization targets
    pub utilization_targets: BandwidthUtilizationTargets,
}

/// Bandwidth utilization targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthUtilizationTargets {
    /// Target utilization (0.0 to 1.0)
    pub target: f64,
    /// Warning threshold (0.0 to 1.0)
    pub warning_threshold: f64,
    /// Critical threshold (0.0 to 1.0)
    pub critical_threshold: f64,
    /// Maximum utilization (0.0 to 1.0)
    pub max_utilization: f64,
}

/// Latency requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyRequirements {
    /// Maximum end-to-end latency (microseconds)
    pub max_end_to_end: f64,
    /// Maximum hop latency (microseconds)
    pub max_hop_latency: f64,
    /// Jitter tolerance (microseconds)
    pub jitter_tolerance: f64,
    /// Latency SLA requirements
    pub sla_requirements: LatencySLA,
}

/// Latency SLA requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencySLA {
    /// P50 latency target (microseconds)
    pub p50_target: f64,
    /// P95 latency target (microseconds)
    pub p95_target: f64,
    /// P99 latency target (microseconds)
    pub p99_target: f64,
    /// P99.9 latency target (microseconds)
    pub p999_target: f64,
}

/// Scalability requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityRequirements {
    /// Maximum supported devices
    pub max_devices: usize,
    /// Scaling factor
    pub scaling_factor: f64,
    /// Linear scalability requirements
    pub linear_scaling: bool,
    /// Horizontal scaling support
    pub horizontal_scaling: HorizontalScalingRequirements,
    /// Vertical scaling support
    pub vertical_scaling: VerticalScalingRequirements,
}

/// Horizontal scaling requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizontalScalingRequirements {
    /// Support for adding nodes
    pub support_node_addition: bool,
    /// Dynamic reconfiguration support
    pub dynamic_reconfig: bool,
    /// Hot-plug support
    pub hot_plug_support: bool,
    /// Maximum scaling rate (nodes/hour)
    pub max_scaling_rate: f64,
}

/// Vertical scaling requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerticalScalingRequirements {
    /// Support for device upgrades
    pub support_device_upgrade: bool,
    /// Memory expansion support
    pub memory_expansion: bool,
    /// Compute enhancement support
    pub compute_enhancement: bool,
    /// Maximum upgrade frequency (upgrades/year)
    pub max_upgrade_frequency: f64,
}

/// Fault tolerance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceRequirements {
    /// Target availability (0.0 to 1.0)
    pub target_availability: f64,
    /// Maximum tolerable failures
    pub max_tolerable_failures: usize,
    /// Recovery time objectives
    pub recovery_objectives: RecoveryObjectives,
    /// Failure isolation requirements
    pub isolation_requirements: FailureIsolationRequirements,
}

/// Recovery time objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryObjectives {
    /// Recovery time objective (seconds)
    pub rto_seconds: f64,
    /// Recovery point objective (seconds)
    pub rpo_seconds: f64,
    /// Mean time to recovery (seconds)
    pub mttr_seconds: f64,
    /// Maximum downtime per year (seconds)
    pub max_downtime_per_year: f64,
}

/// Failure isolation requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureIsolationRequirements {
    /// Blast radius limitation
    pub blast_radius_limit: usize,
    /// Fault containment support
    pub fault_containment: bool,
    /// Graceful degradation support
    pub graceful_degradation: bool,
    /// Automatic failover support
    pub automatic_failover: bool,
}

/// Node types in the topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    /// Compute node with TPU devices
    Compute {
        tpu_count: usize,
        memory_gb: u64,
        cpu_cores: usize,
    },
    /// Storage node
    Storage {
        storage_capacity_tb: f64,
        storage_type: StorageType,
    },
    /// Network node/switch
    Network {
        port_count: usize,
        switching_capacity: f64,
    },
    /// Management node
    Management {
        services: Vec<String>,
    },
}

/// Storage types for storage nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageType {
    /// Solid state drive
    SSD,
    /// Non-volatile memory express
    NVMe,
    /// Hard disk drive
    HDD,
    /// Memory-based storage
    Memory,
    /// Distributed storage
    Distributed { replication_factor: usize },
    /// Hybrid storage
    Hybrid { primary: Box<StorageType>, secondary: Box<StorageType> },
}

/// Network interface types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterfaceType {
    /// Ethernet interface
    Ethernet { speed_gbps: f64 },
    /// InfiniBand interface
    InfiniBand { speed_gbps: f64 },
    /// Fibre Channel interface
    FibreChannel { speed_gbps: f64 },
    /// NVLink interface
    NVLink { version: String, speed_gbps: f64 },
    /// Custom interface
    Custom { name: String, speed_gbps: f64 },
}

/// Network interface status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterfaceStatus {
    /// Interface is up and operational
    Up,
    /// Interface is down
    Down,
    /// Interface is in testing mode
    Testing,
    /// Interface has errors
    Error { error_count: usize },
    /// Interface is being configured
    Configuring,
}

/// Network protocols supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkProtocol {
    /// TCP/IP protocol stack
    TCP,
    /// UDP protocol
    UDP,
    /// RDMA over Converged Ethernet
    RoCE,
    /// RDMA over InfiniBand
    IB,
    /// NVLink protocol
    NVLink,
    /// Custom protocol
    Custom { name: String, version: String },
}

/// Topology validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyValidationConfig {
    /// Enable connectivity validation
    pub validate_connectivity: bool,
    /// Enable performance validation
    pub validate_performance: bool,
    /// Enable constraint validation
    pub validate_constraints: bool,
    /// Validation timeout (seconds)
    pub validation_timeout: f64,
    /// Validation criteria
    pub validation_criteria: ValidationCriteria,
}

/// Validation criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCriteria {
    /// Minimum connectivity percentage
    pub min_connectivity: f64,
    /// Maximum acceptable latency increase
    pub max_latency_increase: f64,
    /// Minimum bandwidth efficiency
    pub min_bandwidth_efficiency: f64,
    /// Maximum constraint violations
    pub max_constraint_violations: usize,
}

/// Topology optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyOptimizationConfig {
    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,
    /// Optimization constraints
    pub constraints: Vec<OptimizationConstraint>,
    /// Optimization algorithm
    pub algorithm: OptimizationAlgorithm,
    /// Optimization parameters
    pub parameters: OptimizationParameters,
}

/// Optimization objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Minimize total latency
    MinimizeLatency,
    /// Maximize bandwidth utilization
    MaximizeBandwidth,
    /// Minimize power consumption
    MinimizePower,
    /// Maximize fault tolerance
    MaximizeFaultTolerance,
    /// Minimize cost
    MinimizeCost,
    /// Multi-objective optimization
    MultiObjective { weights: HashMap<String, f64> },
}

/// Optimization constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationConstraint {
    /// Latency constraint
    Latency { max_latency: f64 },
    /// Bandwidth constraint
    Bandwidth { min_bandwidth: f64 },
    /// Power constraint
    Power { max_power: f64 },
    /// Reliability constraint
    Reliability { min_reliability: f64 },
    /// Custom constraint
    Custom { name: String, parameters: HashMap<String, f64> },
}

/// Optimization algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    /// Genetic algorithm
    Genetic,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Particle swarm optimization
    ParticleSwarm,
    /// Gradient descent
    GradientDescent,
    /// Custom algorithm
    Custom { name: String },
}

/// Optimization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationParameters {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Population size (for evolutionary algorithms)
    pub population_size: Option<usize>,
    /// Learning rate (for gradient-based algorithms)
    pub learning_rate: Option<f64>,
    /// Custom parameters
    pub custom_parameters: HashMap<String, f64>,
}

// Default implementations
impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            topology_type: TopologyType::Mesh { dimension: 2 },
            device_count: 8,
            node_count: 2,
            devices_per_node: 4,
            inter_node_connection: InterNodeConnection::InfiniBand { speed_gbps: 100.0 },
            intra_node_connection: IntraNodeConnection::NVLink {
                version: "4.0".to_string(),
                speed_gbps: 600.0
            },
            enable_optimization: true,
            enable_dynamic_reconfig: false,
            redundancy_level: RedundancyLevel::DualPath,
            qos_settings: TopologyQoSSettings::default(),
            constraints: TopologyConstraints::default(),
            performance_requirements: PerformanceRequirements::default(),
        }
    }
}

impl Default for TopologyQoSSettings {
    fn default() -> Self {
        Self {
            max_latency: 10.0,  // 10 microseconds
            min_bandwidth: 100.0,  // 100 Gbps
            reliability: 0.999,  // 99.9% reliability
            jitter_tolerance: 1.0,  // 1 microsecond
            packet_loss_tolerance: 0.0001,  // 0.01% packet loss
        }
    }
}

impl Default for TopologyConstraints {
    fn default() -> Self {
        Self {
            max_hops: Some(3),
            max_cable_length: Some(100.0),  // 100 meters
            power_budget: PowerBudgetConstraints::default(),
            thermal_constraints: ThermalConstraints::default(),
            space_constraints: SpaceConstraints::default(),
        }
    }
}

impl Default for PowerBudgetConstraints {
    fn default() -> Self {
        Self {
            max_total_power: 10000.0,  // 10 kW
            max_power_per_node: 2000.0,  // 2 kW per node
            efficiency_requirements: 0.85,  // 85% efficiency
            emergency_limits: EmergencyPowerLimits::default(),
        }
    }
}

impl Default for EmergencyPowerLimits {
    fn default() -> Self {
        Self {
            critical_level: 8000.0,  // 8 kW
            shutdown_threshold: 12000.0,  // 12 kW
            reduction_strategy: PowerReductionStrategy::ReduceFrequency { target_reduction: 0.2 },
        }
    }
}

impl Default for ThermalConstraints {
    fn default() -> Self {
        Self {
            max_temperature: 85.0,  // 85°C
            max_temperature_gradient: 10.0,  // 10°C/m
            cooling_requirements: CoolingRequirements::default(),
            thermal_zones: Vec::new(),
        }
    }
}

impl Default for CoolingRequirements {
    fn default() -> Self {
        Self {
            required_capacity: 8000.0,  // 8 kW cooling
            cooling_type: CoolingType::Air { fan_count: 8, cfm_per_fan: 200.0 },
            airflow_cfm: 1600.0,  // 1600 CFM
            redundancy: CoolingRedundancy::NPlusOne,
        }
    }
}

impl Default for SpaceConstraints {
    fn default() -> Self {
        Self {
            max_rack_units: 42,
            floor_space_sqm: 4.0,  // 4 square meters
            max_height: 2.1,  // 2.1 meters
            weight_constraints: WeightConstraints::default(),
            cable_management: CableManagementRequirements::default(),
        }
    }
}

impl Default for WeightConstraints {
    fn default() -> Self {
        Self {
            max_total_weight: 1000.0,  // 1000 kg
            max_weight_per_ru: 25.0,  // 25 kg per RU
            floor_loading_capacity: 500.0,  // 500 kg/sqm
        }
    }
}

impl Default for CableManagementRequirements {
    fn default() -> Self {
        Self {
            tray_capacity: 100,
            max_cable_length: 50.0,  // 50 meters
            separation_requirements: 0.1,  // 10 cm separation
            routing_constraints: Vec::new(),
        }
    }
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            bandwidth: BandwidthRequirements::default(),
            latency: LatencyRequirements::default(),
            scalability: ScalabilityRequirements::default(),
            fault_tolerance: FaultToleranceRequirements::default(),
        }
    }
}

impl Default for BandwidthRequirements {
    fn default() -> Self {
        Self {
            min_aggregate: 800.0,  // 800 Gbps aggregate
            min_per_device: 100.0,  // 100 Gbps per device
            peak_bandwidth: 1000.0,  // 1 Tbps peak
            sustained_bandwidth: 800.0,  // 800 Gbps sustained
            utilization_targets: BandwidthUtilizationTargets::default(),
        }
    }
}

impl Default for BandwidthUtilizationTargets {
    fn default() -> Self {
        Self {
            target: 0.7,  // 70% target utilization
            warning_threshold: 0.8,  // 80% warning
            critical_threshold: 0.9,  // 90% critical
            max_utilization: 0.95,  // 95% maximum
        }
    }
}

impl Default for LatencyRequirements {
    fn default() -> Self {
        Self {
            max_end_to_end: 20.0,  // 20 microseconds
            max_hop_latency: 5.0,  // 5 microseconds per hop
            jitter_tolerance: 1.0,  // 1 microsecond jitter
            sla_requirements: LatencySLA::default(),
        }
    }
}

impl Default for LatencySLA {
    fn default() -> Self {
        Self {
            p50_target: 5.0,  // 5 μs P50
            p95_target: 15.0,  // 15 μs P95
            p99_target: 25.0,  // 25 μs P99
            p999_target: 50.0,  // 50 μs P99.9
        }
    }
}

impl Default for ScalabilityRequirements {
    fn default() -> Self {
        Self {
            max_devices: 1024,
            scaling_factor: 1.5,
            linear_scaling: true,
            horizontal_scaling: HorizontalScalingRequirements::default(),
            vertical_scaling: VerticalScalingRequirements::default(),
        }
    }
}

impl Default for HorizontalScalingRequirements {
    fn default() -> Self {
        Self {
            support_node_addition: true,
            dynamic_reconfig: false,
            hot_plug_support: false,
            max_scaling_rate: 10.0,  // 10 nodes/hour
        }
    }
}

impl Default for VerticalScalingRequirements {
    fn default() -> Self {
        Self {
            support_device_upgrade: true,
            memory_expansion: true,
            compute_enhancement: true,
            max_upgrade_frequency: 2.0,  // 2 upgrades/year
        }
    }
}

impl Default for FaultToleranceRequirements {
    fn default() -> Self {
        Self {
            target_availability: 0.9999,  // 99.99% availability
            max_tolerable_failures: 2,
            recovery_objectives: RecoveryObjectives::default(),
            isolation_requirements: FailureIsolationRequirements::default(),
        }
    }
}

impl Default for RecoveryObjectives {
    fn default() -> Self {
        Self {
            rto_seconds: 300.0,  // 5 minutes RTO
            rpo_seconds: 60.0,  // 1 minute RPO
            mttr_seconds: 600.0,  // 10 minutes MTTR
            max_downtime_per_year: 3153.6,  // 52.56 minutes/year for 99.99%
        }
    }
}

impl Default for FailureIsolationRequirements {
    fn default() -> Self {
        Self {
            blast_radius_limit: 4,  // Limit failures to 4 devices
            fault_containment: true,
            graceful_degradation: true,
            automatic_failover: true,
        }
    }
}

impl Default for TopologyValidationConfig {
    fn default() -> Self {
        Self {
            validate_connectivity: true,
            validate_performance: true,
            validate_constraints: true,
            validation_timeout: 300.0,  // 5 minutes
            validation_criteria: ValidationCriteria::default(),
        }
    }
}

impl Default for ValidationCriteria {
    fn default() -> Self {
        Self {
            min_connectivity: 0.95,  // 95% connectivity
            max_latency_increase: 0.1,  // 10% latency increase
            min_bandwidth_efficiency: 0.8,  // 80% bandwidth efficiency
            max_constraint_violations: 0,
        }
    }
}

impl Default for TopologyOptimizationConfig {
    fn default() -> Self {
        Self {
            objectives: vec![OptimizationObjective::MinimizeLatency],
            constraints: Vec::new(),
            algorithm: OptimizationAlgorithm::Genetic,
            parameters: OptimizationParameters::default(),
        }
    }
}

impl Default for OptimizationParameters {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            convergence_tolerance: 0.001,
            population_size: Some(50),
            learning_rate: Some(0.01),
            custom_parameters: HashMap::new(),
        }
    }
}