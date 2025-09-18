//! Resource allocation and monitoring for streaming optimization
//!
//! This module provides comprehensive resource management capabilities including
//! dynamic resource allocation, monitoring, budgeting, and optimization for
//! streaming optimization workloads.

use super::config::*;
use super::optimizer::{Adaptation, AdaptationPriority, AdaptationType};

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Current resource usage information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Memory usage in MB
    pub memory_usage_mb: usize,
    /// CPU usage percentage (0-100)
    pub cpu_usage_percent: f64,
    /// GPU usage percentage (0-100) if applicable
    pub gpu_usage_percent: Option<f64>,
    /// Network I/O rate in MB/s
    pub network_io_mbps: f64,
    /// Disk I/O rate in MB/s
    pub disk_io_mbps: f64,
    /// Number of active threads
    pub active_threads: usize,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

/// Resource budget and constraints
#[derive(Debug, Clone)]
pub struct ResourceBudget {
    /// Memory budget constraints
    pub memory_budget: MemoryBudget,
    /// CPU budget constraints
    pub cpu_budget: CpuBudget,
    /// Network budget constraints
    pub network_budget: NetworkBudget,
    /// Time budget constraints
    pub time_budget: TimeBudget,
    /// Enforcement strategy
    pub enforcement_strategy: BudgetEnforcementStrategy,
    /// Budget flexibility (0.0 = strict, 1.0 = flexible)
    pub flexibility: f64,
}

/// Memory budget configuration
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Maximum memory allocation in MB
    pub max_allocation_mb: usize,
    /// Soft limit for memory usage in MB
    pub soft_limit_mb: usize,
    /// Memory cleanup threshold (percentage)
    pub cleanup_threshold: f64,
    /// Enable memory compression
    pub enable_compression: bool,
    /// Memory priority levels
    pub priority_levels: Vec<MemoryPriority>,
}

/// Memory allocation priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPriority {
    /// Critical memory for core operations
    Critical,
    /// High priority memory for optimization
    High,
    /// Normal priority memory for buffering
    Normal,
    /// Low priority memory for caching
    Low,
    /// Temporary memory that can be freed immediately
    Temporary,
}

/// CPU budget configuration
#[derive(Debug, Clone)]
pub struct CpuBudget {
    /// Maximum CPU utilization percentage
    pub max_utilization: f64,
    /// Target CPU utilization percentage
    pub target_utilization: f64,
    /// Maximum number of worker threads
    pub max_threads: usize,
    /// Thread priority management
    pub thread_priority: ThreadPriorityConfig,
    /// CPU affinity settings
    pub cpu_affinity: Option<Vec<usize>>,
}

/// Thread priority configuration
#[derive(Debug, Clone)]
pub struct ThreadPriorityConfig {
    /// High priority thread count
    pub high_priority_threads: usize,
    /// Normal priority thread count
    pub normal_priority_threads: usize,
    /// Background thread count
    pub background_threads: usize,
    /// Enable dynamic priority adjustment
    pub dynamic_priority: bool,
}

/// Network budget configuration
#[derive(Debug, Clone)]
pub struct NetworkBudget {
    /// Maximum bandwidth usage in MB/s
    pub max_bandwidth_mbps: f64,
    /// Bandwidth priority allocation
    pub priority_allocation: HashMap<String, f64>,
    /// Enable traffic shaping
    pub enable_traffic_shaping: bool,
    /// Quality of Service settings
    pub qos_settings: QoSSettings,
}

/// Quality of Service settings for network traffic
#[derive(Debug, Clone)]
pub struct QoSSettings {
    /// Latency requirements in milliseconds
    pub max_latency_ms: u64,
    /// Jitter tolerance in milliseconds
    pub jitter_tolerance_ms: u64,
    /// Packet loss tolerance (percentage)
    pub packet_loss_tolerance: f64,
    /// Traffic classes
    pub traffic_classes: Vec<TrafficClass>,
}

/// Network traffic classification
#[derive(Debug, Clone)]
pub struct TrafficClass {
    /// Class name
    pub name: String,
    /// Priority level (0 = highest)
    pub priority: u8,
    /// Bandwidth guarantee (percentage)
    pub bandwidth_guarantee: f64,
    /// Maximum bandwidth (percentage)
    pub max_bandwidth: f64,
}

/// Time budget configuration
#[derive(Debug, Clone)]
pub struct TimeBudget {
    /// Maximum processing time per batch
    pub max_batch_processing_time: Duration,
    /// Target processing time per batch
    pub target_batch_processing_time: Duration,
    /// Timeout for long-running operations
    pub operation_timeout: Duration,
    /// Deadline enforcement strategy
    pub deadline_enforcement: DeadlineEnforcement,
}

/// Deadline enforcement strategies
#[derive(Debug, Clone)]
pub enum DeadlineEnforcement {
    /// Strict deadline enforcement (fail if exceeded)
    Strict,
    /// Soft deadline with warnings
    Soft,
    /// Best effort (informational only)
    BestEffort,
    /// Adaptive deadline based on system load
    Adaptive,
}

/// Budget enforcement strategies
#[derive(Debug, Clone)]
pub enum BudgetEnforcementStrategy {
    /// Strict enforcement (fail if budget exceeded)
    Strict,
    /// Throttling (reduce resource usage)
    Throttling,
    /// Load shedding (drop low priority work)
    LoadShedding,
    /// Graceful degradation
    GracefulDegradation,
    /// Adaptive enforcement based on system state
    Adaptive,
}

/// Resource manager for streaming optimization
pub struct ResourceManager {
    /// Resource configuration
    config: ResourceConfig,
    /// Current resource usage
    current_usage: Arc<Mutex<ResourceUsage>>,
    /// Resource usage history
    usage_history: Arc<Mutex<VecDeque<ResourceUsage>>>,
    /// Resource budget
    budget: ResourceBudget,
    /// Resource allocations by component
    allocations: Arc<Mutex<HashMap<String, ResourceAllocation>>>,
    /// Resource monitoring thread handle
    monitoring_handle: Option<std::thread::JoinHandle<()>>,
    /// Resource prediction model
    predictor: ResourcePredictor,
    /// Resource optimizer
    optimizer: ResourceOptimizer,
    /// Alert system
    alert_system: ResourceAlertSystem,
}

/// Resource allocation for a specific component
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Component name
    pub component_name: String,
    /// Allocated memory in MB
    pub allocated_memory_mb: usize,
    /// Allocated CPU percentage
    pub allocated_cpu_percent: f64,
    /// Allocated network bandwidth in MB/s
    pub allocated_bandwidth_mbps: f64,
    /// Priority level
    pub priority: ResourcePriority,
    /// Allocation timestamp
    pub allocation_time: Instant,
    /// Last access timestamp
    pub last_access: Instant,
    /// Usage statistics
    pub usage_stats: ComponentUsageStats,
}

/// Resource priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResourcePriority {
    /// Critical system resources
    Critical = 0,
    /// High priority operations
    High = 1,
    /// Normal priority operations
    Normal = 2,
    /// Low priority background operations
    Low = 3,
    /// Temporary or cache operations
    Temporary = 4,
}

/// Usage statistics for a component
#[derive(Debug, Clone)]
pub struct ComponentUsageStats {
    /// Peak memory usage
    pub peak_memory_mb: usize,
    /// Average memory usage
    pub avg_memory_mb: usize,
    /// Peak CPU usage
    pub peak_cpu_percent: f64,
    /// Average CPU usage
    pub avg_cpu_percent: f64,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Number of operations performed
    pub operation_count: u64,
    /// Efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,
}

/// Resource usage prediction model
pub struct ResourcePredictor {
    /// Historical usage patterns
    usage_patterns: VecDeque<ResourceUsage>,
    /// Prediction horizon (steps ahead)
    prediction_horizon: usize,
    /// Prediction accuracy tracking
    prediction_accuracy: HashMap<String, f64>,
    /// Seasonal patterns
    seasonal_patterns: HashMap<String, Vec<f64>>,
    /// Trend analysis
    trend_analysis: ResourceTrendAnalysis,
}

/// Resource trend analysis
#[derive(Debug, Clone)]
pub struct ResourceTrendAnalysis {
    /// Memory usage trend
    pub memory_trend: TrendDirection,
    /// CPU usage trend
    pub cpu_trend: TrendDirection,
    /// Network usage trend
    pub network_trend: TrendDirection,
    /// Trend confidence
    pub trend_confidence: f64,
    /// Trend stability
    pub trend_stability: f64,
}

/// Trend direction indicators
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,
    /// Decreasing trend
    Decreasing,
    /// Stable trend
    Stable,
    /// Oscillating trend
    Oscillating,
    /// Unknown trend
    Unknown,
}

/// Resource optimizer for dynamic allocation
pub struct ResourceOptimizer {
    /// Optimization strategy
    strategy: ResourceOptimizationStrategy,
    /// Optimization history
    optimization_history: VecDeque<OptimizationEvent>,
    /// Performance impact tracking
    performance_impact: HashMap<String, f64>,
    /// Optimization constraints
    constraints: OptimizationConstraints,
}

/// Resource optimization strategies
#[derive(Debug, Clone)]
pub enum ResourceOptimizationStrategy {
    /// Conservative optimization (minimize changes)
    Conservative,
    /// Aggressive optimization (maximize performance)
    Aggressive,
    /// Balanced optimization
    Balanced,
    /// Power-efficient optimization
    PowerEfficient,
    /// Latency-optimized
    LatencyOptimized,
    /// Throughput-optimized
    ThroughputOptimized,
}

/// Resource optimization event
#[derive(Debug, Clone)]
pub struct OptimizationEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Optimization type
    pub optimization_type: String,
    /// Resources affected
    pub affected_resources: Vec<String>,
    /// Resource deltas
    pub resource_deltas: HashMap<String, f64>,
    /// Performance impact
    pub performance_impact: f64,
    /// Success indicator
    pub success: bool,
}

/// Constraints for resource optimization
#[derive(Debug, Clone)]
pub struct OptimizationConstraints {
    /// Minimum resource guarantees
    pub min_guarantees: HashMap<String, f64>,
    /// Maximum resource limits
    pub max_limits: HashMap<String, f64>,
    /// Resource change rate limits
    pub change_rate_limits: HashMap<String, f64>,
    /// Stability requirements
    pub stability_requirements: StabilityRequirements,
}

/// Stability requirements for resource allocation
#[derive(Debug, Clone)]
pub struct StabilityRequirements {
    /// Minimum stable period before changes
    pub min_stable_period: Duration,
    /// Maximum change frequency
    pub max_change_frequency: f64,
    /// Oscillation prevention
    pub prevent_oscillation: bool,
    /// Hysteresis factor (0.0 to 1.0)
    pub hysteresis_factor: f64,
}

/// Resource alert system
pub struct ResourceAlertSystem {
    /// Alert thresholds
    thresholds: ResourceThresholds,
    /// Active alerts
    active_alerts: VecDeque<ResourceAlert>,
    /// Alert history
    alert_history: VecDeque<ResourceAlert>,
    /// Alert handlers
    alert_handlers: Vec<Box<dyn AlertHandler>>,
}

/// Resource alert thresholds
#[derive(Debug, Clone)]
pub struct ResourceThresholds {
    /// Memory usage thresholds
    pub memory_thresholds: ThresholdSet,
    /// CPU usage thresholds
    pub cpu_thresholds: ThresholdSet,
    /// Network usage thresholds
    pub network_thresholds: ThresholdSet,
    /// Response time thresholds
    pub response_time_thresholds: ThresholdSet,
}

/// Threshold set for a resource type
#[derive(Debug, Clone)]
pub struct ThresholdSet {
    /// Warning threshold
    pub warning: f64,
    /// Critical threshold
    pub critical: f64,
    /// Emergency threshold
    pub emergency: f64,
    /// Recovery threshold (for clearing alerts)
    pub recovery: f64,
}

/// Resource alert
#[derive(Debug, Clone)]
pub struct ResourceAlert {
    /// Alert ID
    pub id: String,
    /// Alert timestamp
    pub timestamp: Instant,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Resource type
    pub resource_type: String,
    /// Current value
    pub current_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Alert message
    pub message: String,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
    /// Auto-resolution attempts
    pub auto_resolution_attempts: u32,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Error alert
    Error,
    /// Critical alert
    Critical,
    /// Emergency alert
    Emergency,
}

/// Trait for handling resource alerts
pub trait AlertHandler: Send + Sync {
    /// Handles a resource alert
    fn handle_alert(&self, alert: &ResourceAlert) -> Result<(), String>;

    /// Gets handler priority (lower number = higher priority)
    fn priority(&self) -> u32;

    /// Checks if this handler can handle the given alert
    fn can_handle(&self, alert: &ResourceAlert) -> bool;
}

impl ResourceManager {
    /// Creates a new resource manager
    pub fn new(config: &StreamingConfig) -> Result<Self, String> {
        let resource_config = config.resource_config.clone();

        let budget = ResourceBudget {
            memory_budget: MemoryBudget {
                max_allocation_mb: resource_config.max_memory_mb,
                soft_limit_mb: (resource_config.max_memory_mb as f64 * 0.8) as usize,
                cleanup_threshold: resource_config.cleanup_threshold,
                enable_compression: true,
                priority_levels: vec![
                    MemoryPriority::Critical,
                    MemoryPriority::High,
                    MemoryPriority::Normal,
                    MemoryPriority::Low,
                ],
            },
            cpu_budget: CpuBudget {
                max_utilization: resource_config.max_cpu_percent,
                target_utilization: resource_config.max_cpu_percent * 0.8,
                max_threads: num_cpus::get(),
                thread_priority: ThreadPriorityConfig {
                    high_priority_threads: 2,
                    normal_priority_threads: num_cpus::get() - 2,
                    background_threads: 1,
                    dynamic_priority: true,
                },
                cpu_affinity: None,
            },
            network_budget: NetworkBudget {
                max_bandwidth_mbps: 100.0, // Default limit
                priority_allocation: HashMap::new(),
                enable_traffic_shaping: false,
                qos_settings: QoSSettings {
                    max_latency_ms: 100,
                    jitter_tolerance_ms: 10,
                    packet_loss_tolerance: 0.1,
                    traffic_classes: Vec::new(),
                },
            },
            time_budget: TimeBudget {
                max_batch_processing_time: Duration::from_secs(30),
                target_batch_processing_time: Duration::from_secs(10),
                operation_timeout: Duration::from_secs(60),
                deadline_enforcement: DeadlineEnforcement::Soft,
            },
            enforcement_strategy: match resource_config.allocation_strategy {
                ResourceAllocationStrategy::Static => BudgetEnforcementStrategy::Strict,
                ResourceAllocationStrategy::Dynamic => BudgetEnforcementStrategy::Throttling,
                ResourceAllocationStrategy::Adaptive => BudgetEnforcementStrategy::Adaptive,
                _ => BudgetEnforcementStrategy::GracefulDegradation,
            },
            flexibility: 0.2,
        };

        let predictor = ResourcePredictor::new();
        let optimizer = ResourceOptimizer::new(ResourceOptimizationStrategy::Balanced);
        let alert_system = ResourceAlertSystem::new();

        Ok(Self {
            config: resource_config,
            current_usage: Arc::new(Mutex::new(ResourceUsage::default())),
            usage_history: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            budget,
            allocations: Arc::new(Mutex::new(HashMap::new())),
            monitoring_handle: None,
            predictor,
            optimizer,
            alert_system,
        })
    }

    /// Starts resource monitoring
    pub fn start_monitoring(&mut self) -> Result<(), String> {
        if self.monitoring_handle.is_some() {
            return Ok(()); // Already monitoring
        }

        let current_usage = Arc::clone(&self.current_usage);
        let usage_history = Arc::clone(&self.usage_history);
        let monitoring_frequency = self.config.monitoring_frequency;

        let handle = std::thread::spawn(move || {
            loop {
                if let Ok(usage) = Self::collect_resource_usage() {
                    // Update current usage
                    {
                        let mut current = current_usage.lock().unwrap();
                        *current = usage.clone();
                    }

                    // Add to history
                    {
                        let mut history = usage_history.lock().unwrap();
                        if history.len() >= 1000 {
                            history.pop_front();
                        }
                        history.push_back(usage);
                    }
                }

                std::thread::sleep(monitoring_frequency);
            }
        });

        self.monitoring_handle = Some(handle);
        Ok(())
    }

    /// Collects current resource usage
    fn collect_resource_usage() -> Result<ResourceUsage, String> {
        // Simplified resource collection - in practice would use system APIs
        let mut usage = ResourceUsage::default();
        usage.timestamp = Instant::now();

        // Memory usage (simplified)
        if let Ok(info) = sysinfo::System::new_all().used_memory() {
            usage.memory_usage_mb = (info / 1024 / 1024) as usize;
        }

        // CPU usage (simplified - would need proper measurement)
        usage.cpu_usage_percent = 50.0; // Placeholder

        // Network I/O (simplified)
        usage.network_io_mbps = 1.0; // Placeholder

        // Disk I/O (simplified)
        usage.disk_io_mbps = 5.0; // Placeholder

        // Active threads
        usage.active_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        Ok(usage)
    }

    /// Allocates resources for a component
    pub fn allocate_resources(
        &mut self,
        component_name: &str,
        memory_mb: usize,
        cpu_percent: f64,
        priority: ResourcePriority,
    ) -> Result<(), String> {
        // Check budget constraints
        self.check_budget_constraints(memory_mb, cpu_percent)?;

        let allocation = ResourceAllocation {
            component_name: component_name.to_string(),
            allocated_memory_mb: memory_mb,
            allocated_cpu_percent: cpu_percent,
            allocated_bandwidth_mbps: 0.0, // Default
            priority,
            allocation_time: Instant::now(),
            last_access: Instant::now(),
            usage_stats: ComponentUsageStats {
                peak_memory_mb: 0,
                avg_memory_mb: 0,
                peak_cpu_percent: 0.0,
                avg_cpu_percent: 0.0,
                total_processing_time: Duration::ZERO,
                operation_count: 0,
                efficiency_score: 1.0,
            },
        };

        let mut allocations = self.allocations.lock().unwrap();
        allocations.insert(component_name.to_string(), allocation);

        Ok(())
    }

    /// Checks budget constraints for resource allocation
    fn check_budget_constraints(&self, memory_mb: usize, cpu_percent: f64) -> Result<(), String> {
        let allocations = self.allocations.lock().unwrap();

        // Calculate total allocated resources
        let total_memory: usize = allocations
            .values()
            .map(|a| a.allocated_memory_mb)
            .sum::<usize>()
            + memory_mb;

        let total_cpu: f64 = allocations
            .values()
            .map(|a| a.allocated_cpu_percent)
            .sum::<f64>()
            + cpu_percent;

        // Check constraints
        if total_memory > self.budget.memory_budget.max_allocation_mb {
            return Err(format!(
                "Memory allocation would exceed budget: {} MB > {} MB",
                total_memory, self.budget.memory_budget.max_allocation_mb
            ));
        }

        if total_cpu > self.budget.cpu_budget.max_utilization {
            return Err(format!(
                "CPU allocation would exceed budget: {:.2}% > {:.2}%",
                total_cpu, self.budget.cpu_budget.max_utilization
            ));
        }

        Ok(())
    }

    /// Updates resource utilization tracking
    pub fn update_utilization(&mut self) -> Result<(), String> {
        let current_usage = self.current_usage.lock().unwrap().clone();

        // Check for alerts
        let alerts = self.alert_system.check_thresholds(&current_usage)?;
        for alert in alerts {
            self.alert_system.handle_alert(alert)?;
        }

        // Update predictor
        self.predictor.update(&current_usage)?;

        // Check for optimization opportunities
        if self.config.enable_dynamic_allocation {
            self.optimizer
                .check_optimization_opportunities(&current_usage, &self.allocations)?;
        }

        Ok(())
    }

    /// Checks if sufficient resources are available for processing
    pub fn has_sufficient_resources_for_processing(&self) -> Result<bool, String> {
        let current_usage = self.current_usage.lock().unwrap();

        // Check memory availability
        let memory_available = current_usage.memory_usage_mb
            < (self.budget.memory_budget.soft_limit_mb as f64 * 0.9) as usize;

        // Check CPU availability
        let cpu_available =
            current_usage.cpu_usage_percent < self.budget.cpu_budget.target_utilization * 0.9;

        Ok(memory_available && cpu_available)
    }

    /// Computes resource allocation adaptation
    pub fn compute_allocation_adaptation(&mut self) -> Result<Option<Adaptation<f32>>, String> {
        let current_usage = self.current_usage.lock().unwrap();

        // Check if we need to adapt resource allocation
        if current_usage.memory_usage_mb > self.budget.memory_budget.soft_limit_mb {
            // Memory pressure - suggest reducing buffer sizes
            let adaptation = Adaptation {
                adaptation_type: AdaptationType::ResourceAllocation,
                magnitude: -0.2, // Reduce by 20%
                target_component: "memory_manager".to_string(),
                parameters: std::collections::HashMap::new(),
                priority: AdaptationPriority::High,
                timestamp: Instant::now(),
            };

            return Ok(Some(adaptation));
        }

        if current_usage.cpu_usage_percent > self.budget.cpu_budget.target_utilization {
            // CPU pressure - suggest reducing processing frequency
            let adaptation = Adaptation {
                adaptation_type: AdaptationType::ResourceAllocation,
                magnitude: -0.15, // Reduce by 15%
                target_component: "cpu_manager".to_string(),
                parameters: std::collections::HashMap::new(),
                priority: AdaptationPriority::High,
                timestamp: Instant::now(),
            };

            return Ok(Some(adaptation));
        }

        Ok(None)
    }

    /// Applies resource allocation adaptation
    pub fn apply_allocation_adaptation(
        &mut self,
        adaptation: &Adaptation<f32>,
    ) -> Result<(), String> {
        if adaptation.adaptation_type == AdaptationType::ResourceAllocation {
            match adaptation.target_component.as_str() {
                "memory_manager" => {
                    // Adjust memory allocations
                    let factor = 1.0 + adaptation.magnitude;
                    let mut allocations = self.allocations.lock().unwrap();

                    for allocation in allocations.values_mut() {
                        if allocation.priority >= ResourcePriority::Normal {
                            allocation.allocated_memory_mb =
                                ((allocation.allocated_memory_mb as f32) * factor) as usize;
                        }
                    }
                }
                "cpu_manager" => {
                    // Adjust CPU allocations
                    let factor = 1.0 + adaptation.magnitude;
                    let mut allocations = self.allocations.lock().unwrap();

                    for allocation in allocations.values_mut() {
                        if allocation.priority >= ResourcePriority::Normal {
                            allocation.allocated_cpu_percent *= factor as f64;
                        }
                    }
                }
                _ => {
                    // Handle other resource adaptations
                }
            }
        }

        Ok(())
    }

    /// Gets current resource usage
    pub fn current_usage(&self) -> Result<ResourceUsage, String> {
        Ok(self.current_usage.lock().unwrap().clone())
    }

    /// Gets resource usage history
    pub fn get_usage_history(&self, count: usize) -> Vec<ResourceUsage> {
        let history = self.usage_history.lock().unwrap();
        history.iter().rev().take(count).cloned().collect()
    }

    /// Gets diagnostic information
    pub fn get_diagnostics(&self) -> ResourceDiagnostics {
        let current_usage = self.current_usage.lock().unwrap();
        let allocations = self.allocations.lock().unwrap();

        ResourceDiagnostics {
            current_usage: current_usage.clone(),
            total_allocations: allocations.len(),
            memory_utilization: (current_usage.memory_usage_mb as f64
                / self.budget.memory_budget.max_allocation_mb as f64)
                * 100.0,
            cpu_utilization: current_usage.cpu_usage_percent,
            active_alerts: self.alert_system.active_alerts.len(),
            budget_violations: 0, // Would be calculated from history
        }
    }
}

impl ResourcePredictor {
    fn new() -> Self {
        Self {
            usage_patterns: VecDeque::with_capacity(1000),
            prediction_horizon: 10,
            prediction_accuracy: HashMap::new(),
            seasonal_patterns: HashMap::new(),
            trend_analysis: ResourceTrendAnalysis {
                memory_trend: TrendDirection::Unknown,
                cpu_trend: TrendDirection::Unknown,
                network_trend: TrendDirection::Unknown,
                trend_confidence: 0.0,
                trend_stability: 0.0,
            },
        }
    }

    fn update(&mut self, usage: &ResourceUsage) -> Result<(), String> {
        if self.usage_patterns.len() >= 1000 {
            self.usage_patterns.pop_front();
        }
        self.usage_patterns.push_back(usage.clone());

        // Update trend analysis
        if self.usage_patterns.len() >= 10 {
            self.update_trend_analysis()?;
        }

        Ok(())
    }

    fn update_trend_analysis(&mut self) -> Result<(), String> {
        let recent_patterns: Vec<_> = self.usage_patterns.iter().rev().take(10).collect();

        // Analyze memory trend
        let memory_values: Vec<f64> = recent_patterns
            .iter()
            .map(|u| u.memory_usage_mb as f64)
            .collect();
        self.trend_analysis.memory_trend = self.analyze_trend(&memory_values);

        // Analyze CPU trend
        let cpu_values: Vec<f64> = recent_patterns
            .iter()
            .map(|u| u.cpu_usage_percent)
            .collect();
        self.trend_analysis.cpu_trend = self.analyze_trend(&cpu_values);

        // Calculate trend confidence
        self.trend_analysis.trend_confidence =
            self.calculate_trend_confidence(&memory_values, &cpu_values);

        Ok(())
    }

    fn analyze_trend(&self, values: &[f64]) -> TrendDirection {
        if values.len() < 3 {
            return TrendDirection::Unknown;
        }

        let first_half: f64 =
            values.iter().take(values.len() / 2).sum::<f64>() / (values.len() / 2) as f64;
        let second_half: f64 = values.iter().skip(values.len() / 2).sum::<f64>()
            / (values.len() - values.len() / 2) as f64;

        let change_threshold = 0.05; // 5% change threshold
        let relative_change = (second_half - first_half) / first_half.max(1.0);

        if relative_change > change_threshold {
            TrendDirection::Increasing
        } else if relative_change < -change_threshold {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }

    fn calculate_trend_confidence(&self, memory_values: &[f64], cpu_values: &[f64]) -> f64 {
        // Simple confidence calculation based on trend consistency
        let memory_variance = self.calculate_variance(memory_values);
        let cpu_variance = self.calculate_variance(cpu_values);

        // Lower variance = higher confidence
        let memory_confidence = 1.0 / (1.0 + memory_variance / 100.0);
        let cpu_confidence = 1.0 / (1.0 + cpu_variance / 100.0);

        (memory_confidence + cpu_confidence) / 2.0
    }

    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

        variance
    }
}

impl ResourceOptimizer {
    fn new(strategy: ResourceOptimizationStrategy) -> Self {
        Self {
            strategy,
            optimization_history: VecDeque::with_capacity(100),
            performance_impact: HashMap::new(),
            constraints: OptimizationConstraints {
                min_guarantees: HashMap::new(),
                max_limits: HashMap::new(),
                change_rate_limits: HashMap::new(),
                stability_requirements: StabilityRequirements {
                    min_stable_period: Duration::from_secs(60),
                    max_change_frequency: 0.1, // 10% per minute
                    prevent_oscillation: true,
                    hysteresis_factor: 0.1,
                },
            },
        }
    }

    fn check_optimization_opportunities(
        &mut self,
        current_usage: &ResourceUsage,
        allocations: &Arc<Mutex<HashMap<String, ResourceAllocation>>>,
    ) -> Result<(), String> {
        // Check for optimization opportunities based on current strategy
        match self.strategy {
            ResourceOptimizationStrategy::Balanced => {
                self.check_balanced_optimization(current_usage, allocations)?;
            }
            ResourceOptimizationStrategy::PowerEfficient => {
                self.check_power_optimization(current_usage, allocations)?;
            }
            ResourceOptimizationStrategy::LatencyOptimized => {
                self.check_latency_optimization(current_usage, allocations)?;
            }
            _ => {
                // Handle other strategies
            }
        }

        Ok(())
    }

    fn check_balanced_optimization(
        &mut self,
        current_usage: &ResourceUsage,
        _allocations: &Arc<Mutex<HashMap<String, ResourceAllocation>>>,
    ) -> Result<(), String> {
        // Check for resource imbalances
        let memory_utilization = current_usage.memory_usage_mb as f64 / 1024.0; // Simplified
        let cpu_utilization = current_usage.cpu_usage_percent;

        // If one resource is heavily utilized while others are underutilized, suggest rebalancing
        if (memory_utilization > 80.0 && cpu_utilization < 40.0)
            || (cpu_utilization > 80.0 && memory_utilization < 40.0)
        {
            let optimization_event = OptimizationEvent {
                timestamp: Instant::now(),
                optimization_type: "resource_rebalancing".to_string(),
                affected_resources: vec!["memory".to_string(), "cpu".to_string()],
                resource_deltas: HashMap::new(),
                performance_impact: 0.05, // Expected 5% improvement
                success: false,           // Will be updated after application
            };

            if self.optimization_history.len() >= 100 {
                self.optimization_history.pop_front();
            }
            self.optimization_history.push_back(optimization_event);
        }

        Ok(())
    }

    fn check_power_optimization(
        &mut self,
        _current_usage: &ResourceUsage,
        _allocations: &Arc<Mutex<HashMap<String, ResourceAllocation>>>,
    ) -> Result<(), String> {
        // Power optimization logic would go here
        Ok(())
    }

    fn check_latency_optimization(
        &mut self,
        _current_usage: &ResourceUsage,
        _allocations: &Arc<Mutex<HashMap<String, ResourceAllocation>>>,
    ) -> Result<(), String> {
        // Latency optimization logic would go here
        Ok(())
    }
}

impl ResourceAlertSystem {
    fn new() -> Self {
        Self {
            thresholds: ResourceThresholds {
                memory_thresholds: ThresholdSet {
                    warning: 70.0,
                    critical: 85.0,
                    emergency: 95.0,
                    recovery: 65.0,
                },
                cpu_thresholds: ThresholdSet {
                    warning: 75.0,
                    critical: 90.0,
                    emergency: 98.0,
                    recovery: 70.0,
                },
                network_thresholds: ThresholdSet {
                    warning: 80.0,
                    critical: 95.0,
                    emergency: 99.0,
                    recovery: 75.0,
                },
                response_time_thresholds: ThresholdSet {
                    warning: 1000.0,    // 1 second
                    critical: 5000.0,   // 5 seconds
                    emergency: 10000.0, // 10 seconds
                    recovery: 500.0,    // 0.5 seconds
                },
            },
            active_alerts: VecDeque::new(),
            alert_history: VecDeque::with_capacity(1000),
            alert_handlers: Vec::new(),
        }
    }

    fn check_thresholds(&mut self, usage: &ResourceUsage) -> Result<Vec<ResourceAlert>, String> {
        let mut alerts = Vec::new();

        // Check memory thresholds
        let memory_percent = (usage.memory_usage_mb as f64 / 1024.0) * 100.0; // Simplified
        if let Some(alert) =
            self.check_threshold("memory", memory_percent, &self.thresholds.memory_thresholds)?
        {
            alerts.push(alert);
        }

        // Check CPU thresholds
        if let Some(alert) = self.check_threshold(
            "cpu",
            usage.cpu_usage_percent,
            &self.thresholds.cpu_thresholds,
        )? {
            alerts.push(alert);
        }

        Ok(alerts)
    }

    fn check_threshold(
        &self,
        resource_type: &str,
        current_value: f64,
        thresholds: &ThresholdSet,
    ) -> Result<Option<ResourceAlert>, String> {
        let severity = if current_value >= thresholds.emergency {
            AlertSeverity::Emergency
        } else if current_value >= thresholds.critical {
            AlertSeverity::Critical
        } else if current_value >= thresholds.warning {
            AlertSeverity::Warning
        } else {
            return Ok(None);
        };

        let alert = ResourceAlert {
            id: format!("{}_{}", resource_type, Instant::now().elapsed().as_nanos()),
            timestamp: Instant::now(),
            severity,
            resource_type: resource_type.to_string(),
            current_value,
            threshold_value: match severity {
                AlertSeverity::Emergency => thresholds.emergency,
                AlertSeverity::Critical => thresholds.critical,
                AlertSeverity::Warning => thresholds.warning,
                _ => thresholds.warning,
            },
            message: format!(
                "{} usage is {:.2}% (threshold: {:.2}%)",
                resource_type,
                current_value,
                match severity {
                    AlertSeverity::Emergency => thresholds.emergency,
                    AlertSeverity::Critical => thresholds.critical,
                    AlertSeverity::Warning => thresholds.warning,
                    _ => thresholds.warning,
                }
            ),
            suggested_actions: self.generate_suggested_actions(resource_type, &severity),
            auto_resolution_attempts: 0,
        };

        Ok(Some(alert))
    }

    fn generate_suggested_actions(
        &self,
        resource_type: &str,
        severity: &AlertSeverity,
    ) -> Vec<String> {
        match (resource_type, severity) {
            ("memory", AlertSeverity::Critical | AlertSeverity::Emergency) => vec![
                "Reduce buffer sizes".to_string(),
                "Clear caches".to_string(),
                "Reduce batch sizes".to_string(),
            ],
            ("memory", AlertSeverity::Warning) => vec![
                "Monitor memory usage trends".to_string(),
                "Consider reducing buffer sizes".to_string(),
            ],
            ("cpu", AlertSeverity::Critical | AlertSeverity::Emergency) => vec![
                "Reduce processing frequency".to_string(),
                "Lower thread count".to_string(),
                "Defer non-critical operations".to_string(),
            ],
            ("cpu", AlertSeverity::Warning) => vec![
                "Monitor CPU usage patterns".to_string(),
                "Consider load balancing".to_string(),
            ],
            _ => vec!["Monitor resource usage".to_string()],
        }
    }

    fn handle_alert(&mut self, alert: ResourceAlert) -> Result<(), String> {
        // Add to active alerts
        self.active_alerts.push_back(alert.clone());

        // Add to history
        if self.alert_history.len() >= 1000 {
            self.alert_history.pop_front();
        }
        self.alert_history.push_back(alert.clone());

        // Notify handlers
        for handler in &self.alert_handlers {
            if handler.can_handle(&alert) {
                handler.handle_alert(&alert)?;
            }
        }

        Ok(())
    }
}

/// Diagnostic information for resource management
#[derive(Debug, Clone)]
pub struct ResourceDiagnostics {
    pub current_usage: ResourceUsage,
    pub total_allocations: usize,
    pub memory_utilization: f64,
    pub cpu_utilization: f64,
    pub active_alerts: usize,
    pub budget_violations: usize,
}
