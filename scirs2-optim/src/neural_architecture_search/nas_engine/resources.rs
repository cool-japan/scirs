//! Resource Monitoring and Management for Neural Architecture Search
//!
//! This module provides comprehensive resource monitoring, constraint enforcement,
//! and resource optimization functionality for the NAS system.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use std::thread;
use std::marker::PhantomData;
use num_traits::Float;

use crate::error::{OptimError, Result};
use super::config::{ResourceConstraints, HardwareResources, TimeConstraints, ResourceViolationHandling};
use super::results::ResourceUsage;

/// Resource monitor for tracking and managing system resources
#[derive(Debug)]
pub struct ResourceMonitor<T: Float> {
    /// Resource constraints
    constraints: ResourceConstraints<T>,

    /// Current resource usage
    current_usage: Arc<Mutex<ResourceUsage<T>>>,

    /// Resource usage history
    usage_history: Arc<Mutex<Vec<ResourceSnapshot<T>>>>,

    /// Monitoring configuration
    monitoring_config: MonitoringConfig,

    /// Resource monitors
    monitors: Vec<Box<dyn ResourceTracker<T>>>,

    /// Alert handlers
    alert_handlers: Vec<Box<dyn AlertHandler<T>>>,

    /// Monitoring state
    monitoring_state: MonitoringState,

    /// Resource optimization strategies
    optimization_strategies: Vec<Box<dyn ResourceOptimizer<T>>>,
}

/// Resource snapshot for history tracking
#[derive(Debug, Clone)]
pub struct ResourceSnapshot<T: Float> {
    /// Timestamp
    pub timestamp: Instant,

    /// Resource usage at this time
    pub usage: ResourceUsage<T>,

    /// System metrics
    pub system_metrics: SystemMetrics<T>,

    /// Active processes
    pub active_processes: usize,

    /// Memory pressure level
    pub memory_pressure: MemoryPressure,

    /// CPU load average
    pub cpu_load_average: T,

    /// GPU utilization
    pub gpu_utilization: T,
}

/// System metrics
#[derive(Debug, Clone)]
pub struct SystemMetrics<T: Float> {
    /// Available memory (GB)
    pub available_memory_gb: T,

    /// Available CPU cores
    pub available_cpu_cores: usize,

    /// Available GPU devices
    pub available_gpu_devices: usize,

    /// Available disk space (GB)
    pub available_disk_gb: T,

    /// Network bandwidth (MB/s)
    pub network_bandwidth: T,

    /// System temperature (Celsius)
    pub system_temperature: T,

    /// Power consumption (watts)
    pub power_consumption: T,
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy)]
pub enum MemoryPressure {
    Low,
    Medium,
    High,
    Critical,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Monitoring interval
    pub monitoring_interval: Duration,

    /// History retention period
    pub history_retention: Duration,

    /// Enable detailed monitoring
    pub enable_detailed_monitoring: bool,

    /// Enable predictive monitoring
    pub enable_predictive_monitoring: bool,

    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,

    /// Enable automatic optimization
    pub enable_auto_optimization: bool,
}

/// Alert thresholds
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Memory usage threshold (0.0 to 1.0)
    pub memory_threshold: f64,

    /// CPU usage threshold (0.0 to 1.0)
    pub cpu_threshold: f64,

    /// GPU usage threshold (0.0 to 1.0)
    pub gpu_threshold: f64,

    /// Disk usage threshold (0.0 to 1.0)
    pub disk_threshold: f64,

    /// Temperature threshold (Celsius)
    pub temperature_threshold: f64,

    /// Power consumption threshold (watts)
    pub power_threshold: f64,
}

/// Monitoring state
#[derive(Debug, Clone, Copy)]
pub enum MonitoringState {
    Stopped,
    Starting,
    Running,
    Paused,
    Stopping,
    Error,
}

/// Resource tracker trait
pub trait ResourceTracker<T: Float>: Send + Sync {
    /// Get current resource usage
    fn get_current_usage(&self) -> Result<ResourceUsage<T>>;

    /// Get system metrics
    fn get_system_metrics(&self) -> Result<SystemMetrics<T>>;

    /// Check if resource limits are exceeded
    fn check_limits(&self, constraints: &ResourceConstraints<T>) -> Result<Vec<ResourceViolation<T>>>;

    /// Get tracker name
    fn name(&self) -> &str;

    /// Initialize tracker
    fn initialize(&mut self) -> Result<()>;

    /// Cleanup tracker
    fn cleanup(&mut self) -> Result<()>;
}

/// Resource violation
#[derive(Debug, Clone)]
pub struct ResourceViolation<T: Float> {
    /// Violation type
    pub violation_type: ViolationType,

    /// Current value
    pub current_value: T,

    /// Limit value
    pub limit_value: T,

    /// Severity
    pub severity: ViolationSeverity,

    /// Violation time
    pub violation_time: Instant,

    /// Affected resources
    pub affected_resources: Vec<String>,

    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Types of resource violations
#[derive(Debug, Clone, Copy)]
pub enum ViolationType {
    MemoryExceeded,
    CPUExceeded,
    GPUExceeded,
    DiskExceeded,
    TimeExceeded,
    EnergyExceeded,
    TemperatureExceeded,
    PowerExceeded,
    NetworkExceeded,
}

/// Violation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Alert handler trait
pub trait AlertHandler<T: Float>: Send + Sync {
    /// Handle resource violation
    fn handle_violation(&self, violation: &ResourceViolation<T>) -> Result<()>;

    /// Handle resource warning
    fn handle_warning(&self, warning: &ResourceWarning<T>) -> Result<()>;

    /// Get handler name
    fn name(&self) -> &str;
}

/// Resource warning
#[derive(Debug, Clone)]
pub struct ResourceWarning<T: Float> {
    /// Warning type
    pub warning_type: WarningType,

    /// Current value
    pub current_value: T,

    /// Threshold value
    pub threshold_value: T,

    /// Trend direction
    pub trend: TrendDirection,

    /// Estimated time to violation
    pub time_to_violation: Option<Duration>,

    /// Warning message
    pub message: String,
}

/// Types of resource warnings
#[derive(Debug, Clone, Copy)]
pub enum WarningType {
    MemoryApproachingLimit,
    CPUApproachingLimit,
    GPUApproachingLimit,
    DiskApproachingLimit,
    TimeApproachingLimit,
    EnergyApproachingLimit,
    TemperatureRising,
    PowerIncreasing,
}

/// Trend directions
#[derive(Debug, Clone, Copy)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Resource optimizer trait
pub trait ResourceOptimizer<T: Float>: Send + Sync {
    /// Optimize resource usage
    fn optimize(&self, current_usage: &ResourceUsage<T>, constraints: &ResourceConstraints<T>) -> Result<OptimizationAction<T>>;

    /// Get optimizer name
    fn name(&self) -> &str;

    /// Get optimization priority
    fn priority(&self) -> OptimizationPriority;
}

/// Optimization actions
#[derive(Debug, Clone)]
pub struct OptimizationAction<T: Float> {
    /// Action type
    pub action_type: ActionType,

    /// Action parameters
    pub parameters: HashMap<String, T>,

    /// Expected resource savings
    pub expected_savings: ResourceUsage<T>,

    /// Implementation cost
    pub implementation_cost: T,

    /// Action description
    pub description: String,

    /// Priority
    pub priority: OptimizationPriority,
}

/// Types of optimization actions
#[derive(Debug, Clone, Copy)]
pub enum ActionType {
    ReduceMemoryUsage,
    OptimizeCPUUsage,
    OptimizeGPUUsage,
    ClearCaches,
    GarbageCollection,
    ProcessThrottling,
    ResourceReallocation,
    TaskMigration,
    PowerManagement,
    TemperatureControl,
}

/// Optimization priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// System resource tracker implementation
pub struct SystemResourceTracker {
    /// Tracker name
    name: String,

    /// Monitoring interval
    monitoring_interval: Duration,

    /// Last update time
    last_update: Instant,

    /// Cached metrics
    cached_metrics: Option<SystemMetrics<f64>>,
}

impl SystemResourceTracker {
    /// Create a new system resource tracker
    pub fn new(name: String, monitoring_interval: Duration) -> Self {
        Self {
            name,
            monitoring_interval,
            last_update: Instant::now(),
            cached_metrics: None,
        }
    }

    /// Get memory information
    fn get_memory_info(&self) -> Result<(f64, f64)> {
        // In a real implementation, this would query system APIs
        // For now, return simulated values
        let total_memory = 32.0; // 32 GB
        let used_memory = 16.0; // 16 GB used
        Ok((total_memory, used_memory))
    }

    /// Get CPU information
    fn get_cpu_info(&self) -> Result<(usize, f64)> {
        // In a real implementation, this would query system APIs
        let total_cores = num_cpus::get();
        let cpu_usage = 0.6; // 60% usage
        Ok((total_cores, cpu_usage))
    }

    /// Get GPU information
    fn get_gpu_info(&self) -> Result<(usize, f64)> {
        // In a real implementation, this would query GPU APIs (CUDA, OpenCL, etc.)
        let gpu_devices = 4;
        let gpu_usage = 0.8; // 80% usage
        Ok((gpu_devices, gpu_usage))
    }

    /// Get disk information
    fn get_disk_info(&self) -> Result<(f64, f64)> {
        // In a real implementation, this would query filesystem APIs
        let total_disk = 1000.0; // 1TB
        let used_disk = 500.0; // 500GB used
        Ok((total_disk, used_disk))
    }

    /// Get network information
    fn get_network_info(&self) -> Result<f64> {
        // In a real implementation, this would query network interfaces
        let bandwidth = 1000.0; // 1 GB/s
        Ok(bandwidth)
    }

    /// Get temperature information
    fn get_temperature_info(&self) -> Result<f64> {
        // In a real implementation, this would query thermal sensors
        let temperature = 65.0; // 65°C
        Ok(temperature)
    }

    /// Get power information
    fn get_power_info(&self) -> Result<f64> {
        // In a real implementation, this would query power management APIs
        let power = 250.0; // 250W
        Ok(power)
    }
}

impl<T: Float> ResourceTracker<T> for SystemResourceTracker
where
    T: From<f64> + std::fmt::Debug,
{
    fn get_current_usage(&self) -> Result<ResourceUsage<T>> {
        let (total_memory, used_memory) = self.get_memory_info()?;
        let (total_cores, cpu_usage) = self.get_cpu_info()?;
        let (gpu_devices, gpu_usage) = self.get_gpu_info()?;

        Ok(ResourceUsage {
            memory_gb: num_traits::cast::cast(used_memory).unwrap_or_else(|| T::zero()),
            cpu_time_seconds: num_traits::cast::cast(cpu_usage * 3600.0).unwrap_or_else(|| T::zero()), // Convert to CPU-hours equivalent
            gpu_time_seconds: num_traits::cast::cast(gpu_usage * 3600.0).unwrap_or_else(|| T::zero()), // Convert to GPU-hours equivalent
            energy_kwh: num_traits::cast::cast(0.25).unwrap_or_else(|| T::zero()), // 0.25 kWh estimated
            network_io_gb: T::from(1.0).unwrap_or_else(|| T::zero()), // 1 GB network I/O
            disk_io_gb: T::from(2.0).unwrap_or_else(|| T::zero()), // 2 GB disk I/O
            peak_memory_gb: T::from(used_memory * 1.2).unwrap_or_else(|| T::zero()), // 20% overhead
            efficiency_score: T::from(0.8), // 80% efficiency
        })
    }

    fn get_system_metrics(&self) -> Result<SystemMetrics<T>> {
        let (total_memory, used_memory) = self.get_memory_info()?;
        let (total_cores, _cpu_usage) = self.get_cpu_info()?;
        let (gpu_devices, _gpu_usage) = self.get_gpu_info()?;
        let (total_disk, used_disk) = self.get_disk_info()?;
        let bandwidth = self.get_network_info()?;
        let temperature = self.get_temperature_info()?;
        let power = self.get_power_info()?;

        Ok(SystemMetrics {
            available_memory_gb: T::from(total_memory - used_memory),
            available_cpu_cores: total_cores,
            available_gpu_devices: gpu_devices,
            available_disk_gb: T::from(total_disk - used_disk),
            network_bandwidth: T::from(bandwidth),
            system_temperature: T::from(temperature),
            power_consumption: T::from(power),
        })
    }

    fn check_limits(&self, constraints: &ResourceConstraints<T>) -> Result<Vec<ResourceViolation<T>>> {
        let current_usage = self.get_current_usage()?;
        let mut violations = Vec::new();

        // Check memory limit
        let memory_limit = T::from(constraints.hardware_resources.max_memory_gb);
        if current_usage.memory_gb > memory_limit {
            violations.push(ResourceViolation {
                violation_type: ViolationType::MemoryExceeded,
                current_value: current_usage.memory_gb,
                limit_value: memory_limit,
                severity: ViolationSeverity::High,
                violation_time: Instant::now(),
                affected_resources: vec!["Memory".to_string()],
                suggested_actions: vec![
                    "Clear caches".to_string(),
                    "Reduce batch size".to_string(),
                    "Enable memory optimization".to_string(),
                ],
            });
        }

        // Check power limit
        let system_metrics = self.get_system_metrics()?;
        let power_limit = T::from(1000.0); // 1000W limit
        if system_metrics.power_consumption > power_limit {
            violations.push(ResourceViolation {
                violation_type: ViolationType::PowerExceeded,
                current_value: system_metrics.power_consumption,
                limit_value: power_limit,
                severity: ViolationSeverity::Medium,
                violation_time: Instant::now(),
                affected_resources: vec!["Power".to_string()],
                suggested_actions: vec![
                    "Reduce clock speeds".to_string(),
                    "Throttle processes".to_string(),
                    "Enable power saving mode".to_string(),
                ],
            });
        }

        // Check temperature limit
        let temp_limit = T::from(80.0); // 80°C limit
        if system_metrics.system_temperature > temp_limit {
            violations.push(ResourceViolation {
                violation_type: ViolationType::TemperatureExceeded,
                current_value: system_metrics.system_temperature,
                limit_value: temp_limit,
                severity: ViolationSeverity::Critical,
                violation_time: Instant::now(),
                affected_resources: vec!["CPU".to_string(), "GPU".to_string()],
                suggested_actions: vec![
                    "Increase cooling".to_string(),
                    "Reduce workload".to_string(),
                    "Enable thermal throttling".to_string(),
                ],
            });
        }

        Ok(violations)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn initialize(&mut self) -> Result<()> {
        // Initialize system monitoring
        self.last_update = Instant::now();
        println!("Initialized system resource tracker: {}", self.name);
        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        // Cleanup monitoring resources
        self.cached_metrics = None;
        println!("Cleaned up system resource tracker: {}", self.name);
        Ok(())
    }
}

/// Console alert handler implementation
pub struct ConsoleAlertHandler {
    /// Handler name
    name: String,

    /// Enable verbose logging
    verbose: bool,
}

impl ConsoleAlertHandler {
    /// Create a new console alert handler
    pub fn new(name: String, verbose: bool) -> Self {
        Self { name, verbose }
    }
}

impl<T: Float> AlertHandler<T> for ConsoleAlertHandler
where
    T: std::fmt::Display + std::fmt::Debug,
{
    fn handle_violation(&self, violation: &ResourceViolation<T>) -> Result<()> {
        println!(
            "[VIOLATION] {:?}: Current={}, Limit={}, Severity={:?}",
            violation.violation_type,
            violation.current_value,
            violation.limit_value,
            violation.severity
        );

        if self.verbose {
            println!("  Time: {:?}", violation.violation_time);
            println!("  Affected resources: {:?}", violation.affected_resources);
            println!("  Suggested actions: {:?}", violation.suggested_actions);
        }

        Ok(())
    }

    fn handle_warning(&self, warning: &ResourceWarning<T>) -> Result<()> {
        println!(
            "[WARNING] {:?}: Current={}, Threshold={}, Trend={:?}",
            warning.warning_type,
            warning.current_value,
            warning.threshold_value,
            warning.trend
        );

        if self.verbose {
            if let Some(time_to_violation) = warning.time_to_violation {
                println!("  Time to violation: {:?}", time_to_violation);
            }
            println!("  Message: {}", warning.message);
        }

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Memory optimizer implementation
pub struct MemoryOptimizer {
    /// Optimizer name
    name: String,

    /// Optimization aggressiveness
    aggressiveness: f64,
}

impl MemoryOptimizer {
    /// Create a new memory optimizer
    pub fn new(name: String, aggressiveness: f64) -> Self {
        Self { name, aggressiveness }
    }
}

impl<T: Float> ResourceOptimizer<T> for MemoryOptimizer
where
    T: From<f64> + std::cmp::PartialOrd,
{
    fn optimize(&self, current_usage: &ResourceUsage<T>, constraints: &ResourceConstraints<T>) -> Result<OptimizationAction<T>> {
        let memory_limit = T::from(constraints.hardware_resources.max_memory_gb);

        if current_usage.memory_gb > memory_limit * T::from(0.8) {
            let reduction_target = current_usage.memory_gb * T::from(self.aggressiveness * 0.2);

            let mut parameters = HashMap::new();
            parameters.insert("memory_reduction_gb".to_string(), reduction_target);
            parameters.insert("aggressiveness".to_string(), T::from(self.aggressiveness));

            Ok(OptimizationAction {
                action_type: ActionType::ReduceMemoryUsage,
                parameters,
                expected_savings: ResourceUsage {
                    memory_gb: reduction_target,
                    cpu_time_seconds: T::from(0.0),
                    gpu_time_seconds: T::from(0.0),
                    energy_kwh: T::from(0.0),
                    network_io_gb: T::from(0.0),
                    disk_io_gb: T::from(0.0),
                    peak_memory_gb: reduction_target,
                    efficiency_score: T::from(0.1),
                },
                implementation_cost: T::from(0.05),
                description: "Reduce memory usage through cache clearing and optimization".to_string(),
                priority: OptimizationPriority::High,
            })
        } else {
            Ok(OptimizationAction {
                action_type: ActionType::ReduceMemoryUsage,
                parameters: HashMap::new(),
                expected_savings: ResourceUsage::default(),
                implementation_cost: T::from(0.0),
                description: "No memory optimization needed".to_string(),
                priority: OptimizationPriority::Low,
            })
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> OptimizationPriority {
        OptimizationPriority::High
    }
}

impl<T: Float + Send + Sync + From<f64> + std::fmt::Debug> ResourceMonitor<T>
where
    T: std::cmp::PartialOrd + std::fmt::Display,
{
    /// Create a new resource monitor
    pub fn new(constraints: ResourceConstraints<T>) -> Self {
        let monitoring_config = MonitoringConfig {
            monitoring_interval: Duration::from_secs(5),
            history_retention: Duration::from_secs(3600), // 1 hour
            enable_detailed_monitoring: true,
            enable_predictive_monitoring: false,
            alert_thresholds: AlertThresholds {
                memory_threshold: 0.8,
                cpu_threshold: 0.9,
                gpu_threshold: 0.9,
                disk_threshold: 0.9,
                temperature_threshold: 75.0,
                power_threshold: 800.0,
            },
            enable_auto_optimization: true,
        };

        let mut monitors: Vec<Box<dyn ResourceTracker<T>>> = Vec::new();
        monitors.push(Box::new(SystemResourceTracker::new(
            "SystemTracker".to_string(),
            monitoring_config.monitoring_interval,
        )));

        let mut alert_handlers: Vec<Box<dyn AlertHandler<T>>> = Vec::new();
        alert_handlers.push(Box::new(ConsoleAlertHandler::new(
            "ConsoleHandler".to_string(),
            true,
        )));

        let mut optimization_strategies: Vec<Box<dyn ResourceOptimizer<T>>> = Vec::new();
        optimization_strategies.push(Box::new(MemoryOptimizer::new(
            "MemoryOptimizer".to_string(),
            0.8,
        )));

        Self {
            constraints,
            current_usage: Arc::new(Mutex::new(ResourceUsage::default())),
            usage_history: Arc::new(Mutex::new(Vec::new())),
            monitoring_config,
            monitors,
            alert_handlers,
            monitoring_state: MonitoringState::Stopped,
            optimization_strategies,
        }
    }

    /// Start resource monitoring
    pub fn start_monitoring(&mut self) -> Result<()> {
        self.monitoring_state = MonitoringState::Starting;

        // Initialize all monitors
        for monitor in &mut self.monitors {
            monitor.initialize()?;
        }

        self.monitoring_state = MonitoringState::Running;
        println!("Resource monitoring started");
        Ok(())
    }

    /// Stop resource monitoring
    pub fn stop_monitoring(&mut self) -> Result<()> {
        self.monitoring_state = MonitoringState::Stopping;

        // Cleanup all monitors
        for monitor in &mut self.monitors {
            monitor.cleanup()?;
        }

        self.monitoring_state = MonitoringState::Stopped;
        println!("Resource monitoring stopped");
        Ok(())
    }

    /// Update resource usage
    pub fn update_usage(&self) -> Result<()> {
        if self.monitoring_state != MonitoringState::Running {
            return Ok(());
        }

        // Get current usage from all monitors
        let mut total_usage = ResourceUsage::default();
        let mut system_metrics = None;

        for monitor in &self.monitors {
            let usage = monitor.get_current_usage()?;
            total_usage.memory_gb = total_usage.memory_gb + usage.memory_gb;
            total_usage.cpu_time_seconds = total_usage.cpu_time_seconds + usage.cpu_time_seconds;
            total_usage.gpu_time_seconds = total_usage.gpu_time_seconds + usage.gpu_time_seconds;
            total_usage.energy_kwh = total_usage.energy_kwh + usage.energy_kwh;

            if system_metrics.is_none() {
                system_metrics = Some(monitor.get_system_metrics()?);
            }
        }

        // Update current usage
        {
            let mut current = self.current_usage.lock().unwrap();
            *current = total_usage.clone();
        }

        // Add to history
        if let Some(metrics) = system_metrics {
            let snapshot = ResourceSnapshot {
                timestamp: Instant::now(),
                usage: total_usage,
                system_metrics: metrics,
                active_processes: 1, // Simplified
                memory_pressure: MemoryPressure::Low, // Simplified
                cpu_load_average: T::from(0.6),
                gpu_utilization: T::from(0.8),
            };

            {
                let mut history = self.usage_history.lock().unwrap();
                history.push(snapshot);

                // Clean old history
                let cutoff_time = Instant::now() - self.monitoring_config.history_retention;
                history.retain(|s| s.timestamp > cutoff_time);
            }
        }

        Ok(())
    }

    /// Check for resource violations
    pub fn check_violations(&self) -> Result<Vec<ResourceViolation<T>>> {
        let mut all_violations = Vec::new();

        for monitor in &self.monitors {
            let violations = monitor.check_limits(&self.constraints)?;
            all_violations.extend(violations);
        }

        // Handle violations
        for violation in &all_violations {
            for handler in &self.alert_handlers {
                handler.handle_violation(violation)?;
            }
        }

        Ok(all_violations)
    }

    /// Optimize resource usage
    pub fn optimize_resources(&self) -> Result<Vec<OptimizationAction<T>>> {
        if !self.monitoring_config.enable_auto_optimization {
            return Ok(Vec::new());
        }

        let current_usage = {
            let usage = self.current_usage.lock().unwrap();
            usage.clone()
        };

        let mut optimization_actions = Vec::new();

        for optimizer in &self.optimization_strategies {
            let action = optimizer.optimize(&current_usage, &self.constraints)?;
            if action.priority >= OptimizationPriority::Medium {
                optimization_actions.push(action);
            }
        }

        // Sort by priority
        optimization_actions.sort_by_key(|a| a.priority);

        Ok(optimization_actions)
    }

    /// Get current resource usage
    pub fn get_current_usage(&self) -> ResourceUsage<T> {
        let usage = self.current_usage.lock().unwrap();
        usage.clone()
    }

    /// Get usage history
    pub fn get_usage_history(&self) -> Vec<ResourceSnapshot<T>> {
        let history = self.usage_history.lock().unwrap();
        history.clone()
    }

    /// Get monitoring state
    pub fn get_monitoring_state(&self) -> MonitoringState {
        self.monitoring_state
    }

    /// Update constraints
    pub fn update_constraints(&mut self, constraints: ResourceConstraints<T>) {
        self.constraints = constraints;
    }

    /// Generate resource report
    pub fn generate_report(&self) -> ResourceReport<T> {
        let current_usage = self.get_current_usage();
        let history = self.get_usage_history();

        let usage_trend = self.calculate_usage_trend(&history);
        let efficiency_score = self.calculate_efficiency_score(&current_usage);
        let recommendations = self.generate_recommendations(&current_usage, &history);

        ResourceReport {
            timestamp: Instant::now(),
            current_usage,
            usage_trend,
            efficiency_score,
            total_samples: history.len(),
            monitoring_duration: if !history.is_empty() {
                history.last().unwrap().timestamp.duration_since(history.first().unwrap().timestamp)
            } else {
                Duration::from_secs(0)
            },
            recommendations,
            constraint_violations: self.check_violations().unwrap_or_default(),
        }
    }

    /// Calculate usage trend
    fn calculate_usage_trend(&self, history: &[ResourceSnapshot<T>]) -> UsageTrend<T> {
        if history.len() < 2 {
            return UsageTrend::default();
        }

        let recent_window = history.len().min(10);
        let recent = &history[history.len() - recent_window..];

        let memory_trend = self.calculate_metric_trend(recent.iter().map(|s| s.usage.memory_gb).collect());
        let cpu_trend = self.calculate_metric_trend(recent.iter().map(|s| s.usage.cpu_time_seconds).collect());
        let gpu_trend = self.calculate_metric_trend(recent.iter().map(|s| s.usage.gpu_time_seconds).collect());

        UsageTrend {
            memory_trend,
            cpu_trend,
            gpu_trend,
            energy_trend: TrendDirection::Stable, // Simplified
            overall_trend: TrendDirection::Stable, // Simplified
        }
    }

    /// Calculate metric trend
    fn calculate_metric_trend(&self, values: Vec<T>) -> TrendDirection {
        if values.len() < 2 {
            return TrendDirection::Stable;
        }

        let first_half_avg = values.iter().take(values.len() / 2).fold(T::from(0.0), |acc, &x| acc + x) / T::from(values.len() / 2);
        let second_half_avg = values.iter().skip(values.len() / 2).fold(T::from(0.0), |acc, &x| acc + x) / T::from(values.len() - values.len() / 2);

        let change_threshold = T::from(0.05); // 5% change threshold

        if second_half_avg > first_half_avg + change_threshold {
            TrendDirection::Increasing
        } else if second_half_avg < first_half_avg - change_threshold {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }

    /// Calculate efficiency score
    fn calculate_efficiency_score(&self, usage: &ResourceUsage<T>) -> T {
        // Simplified efficiency calculation
        usage.efficiency_score
    }

    /// Generate recommendations
    fn generate_recommendations(&self, _usage: &ResourceUsage<T>, _history: &[ResourceSnapshot<T>]) -> Vec<String> {
        vec![
            "Consider enabling memory optimization".to_string(),
            "Monitor GPU utilization for potential improvements".to_string(),
            "Review energy consumption patterns".to_string(),
        ]
    }
}

/// Resource usage trend
#[derive(Debug, Clone)]
pub struct UsageTrend<T: Float> {
    pub memory_trend: TrendDirection,
    pub cpu_trend: TrendDirection,
    pub gpu_trend: TrendDirection,
    pub energy_trend: TrendDirection,
    pub overall_trend: TrendDirection,
    _phantom: PhantomData<T>,
}

impl<T: Float> Default for UsageTrend<T> {
    fn default() -> Self {
        Self {
            memory_trend: TrendDirection::Stable,
            cpu_trend: TrendDirection::Stable,
            gpu_trend: TrendDirection::Stable,
            energy_trend: TrendDirection::Stable,
            overall_trend: TrendDirection::Stable,
            _phantom: PhantomData,
        }
    }
}

/// Resource monitoring report
#[derive(Debug, Clone)]
pub struct ResourceReport<T: Float> {
    /// Report timestamp
    pub timestamp: Instant,

    /// Current resource usage
    pub current_usage: ResourceUsage<T>,

    /// Usage trends
    pub usage_trend: UsageTrend<T>,

    /// Efficiency score
    pub efficiency_score: T,

    /// Total samples in history
    pub total_samples: usize,

    /// Monitoring duration
    pub monitoring_duration: Duration,

    /// Optimization recommendations
    pub recommendations: Vec<String>,

    /// Current constraint violations
    pub constraint_violations: Vec<ResourceViolation<T>>,
}

// Default implementations
impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_secs(5),
            history_retention: Duration::from_secs(3600),
            enable_detailed_monitoring: true,
            enable_predictive_monitoring: false,
            alert_thresholds: AlertThresholds::default(),
            enable_auto_optimization: true,
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            memory_threshold: 0.8,
            cpu_threshold: 0.9,
            gpu_threshold: 0.9,
            disk_threshold: 0.9,
            temperature_threshold: 75.0,
            power_threshold: 800.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_monitor_creation() {
        let constraints = ResourceConstraints::default();
        let monitor = ResourceMonitor::<f32>::new(constraints);
        assert_eq!(monitor.get_monitoring_state(), MonitoringState::Stopped);
    }

    #[test]
    fn test_system_resource_tracker() {
        let mut tracker = SystemResourceTracker::new(
            "TestTracker".to_string(),
            Duration::from_secs(1),
        );

        assert!(tracker.initialize().is_ok());

        let usage = tracker.get_current_usage::<f32>();
        assert!(usage.is_ok());

        let metrics = tracker.get_system_metrics::<f32>();
        assert!(metrics.is_ok());

        assert!(tracker.cleanup().is_ok());
    }

    #[test]
    fn test_memory_optimizer() {
        let optimizer = MemoryOptimizer::new("TestOptimizer".to_string(), 0.5);

        let usage = ResourceUsage {
            memory_gb: 20.0,
            cpu_time_seconds: 100.0,
            gpu_time_seconds: 50.0,
            energy_kwh: 1.0,
            network_io_gb: 1.0,
            disk_io_gb: 2.0,
            peak_memory_gb: 22.0,
            efficiency_score: 0.8,
        };

        let constraints = ResourceConstraints::default();
        let action = optimizer.optimize(&usage, &constraints);
        assert!(action.is_ok());
    }

    #[test]
    fn test_alert_handler() {
        let handler = ConsoleAlertHandler::new("TestHandler".to_string(), false);

        let violation = ResourceViolation {
            violation_type: ViolationType::MemoryExceeded,
            current_value: 25.0,
            limit_value: 20.0,
            severity: ViolationSeverity::High,
            violation_time: Instant::now(),
            affected_resources: vec!["Memory".to_string()],
            suggested_actions: vec!["Clear cache".to_string()],
        };

        assert!(handler.handle_violation(&violation).is_ok());
    }

    #[test]
    fn test_trend_calculation() {
        let constraints = ResourceConstraints::default();
        let monitor = ResourceMonitor::<f32>::new(constraints);

        let values = vec![1.0, 1.1, 1.2, 1.3, 1.4];
        let trend = monitor.calculate_metric_trend(values);
        assert!(matches!(trend, TrendDirection::Increasing));

        let stable_values = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let stable_trend = monitor.calculate_metric_trend(stable_values);
        assert!(matches!(stable_trend, TrendDirection::Stable));
    }
}