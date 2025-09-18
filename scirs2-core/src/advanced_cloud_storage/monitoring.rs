//! Performance monitoring and alerting system
//!
//! This module provides comprehensive monitoring, alerting, and dashboard
//! capabilities for cloud storage operations and performance tracking.

use crate::error::{CoreError, CoreResult};
use super::types::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Cloud storage monitoring
#[derive(Debug)]
pub struct CloudStorageMonitoring {
    /// Metrics collectors
    metrics_collectors: Vec<MetricsCollector>,
    /// Alert manager
    alert_manager: AlertManager,
    /// Performance dashboard
    dashboard: PerformanceDashboard,
    /// Health checks
    health_checks: Vec<HealthCheck>,
}

/// Metrics collector
#[derive(Debug)]
pub struct MetricsCollector {
    /// Collector name
    pub name: String,
    /// Metric types
    pub metric_types: Vec<MetricType>,
    /// Collection interval
    pub collection_interval: Duration,
    /// Data retention
    pub data_retention: Duration,
}

/// Metric types
#[derive(Debug, Clone)]
pub enum MetricType {
    Latency,
    Throughput,
    ErrorRate,
    Cost,
    Availability,
    Storage,
    Bandwidth,
}

/// Alert manager
#[derive(Debug)]
pub struct AlertManager {
    /// Active alerts
    active_alerts: Vec<Alert>,
    /// Alert rules
    alert_rules: Vec<AlertRule>,
    /// Notification channels
    notification_channels: Vec<NotificationChannel>,
}

/// Alert
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Alert level
    pub level: AlertLevel,
    /// Alert message
    pub message: String,
    /// Source
    pub source: String,
    /// Timestamp
    pub timestamp: Instant,
    /// Acknowledged
    pub acknowledged: bool,
}

/// Alert levels
#[derive(Debug, Clone)]
pub enum AlertLevel {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert rule
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Metric condition
    pub condition: AlertCondition,
    /// Threshold
    pub threshold: f64,
    /// Evaluation interval
    pub evaluation_interval: Duration,
}

/// Alert condition
#[derive(Debug, Clone)]
pub struct AlertCondition {
    /// Metric name
    pub metric: String,
    /// Operator
    pub operator: ComparisonOperator,
    /// Time window
    pub time_window: Duration,
}

/// Comparison operators
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    GreaterOrEqual,
    LessOrEqual,
}

/// Notification channel
#[derive(Debug, Clone)]
pub struct NotificationChannel {
    /// Channel type
    pub channel_type: NotificationChannelType,
    /// Configuration
    pub config: HashMap<String, String>,
    /// Enabled
    pub enabled: bool,
}

/// Notification channel types
#[derive(Debug, Clone)]
pub enum NotificationChannelType {
    Email,
    Slack,
    Webhook,
    SMS,
    PagerDuty,
}

/// Performance dashboard
#[derive(Debug)]
pub struct PerformanceDashboard {
    /// Dashboard widgets
    widgets: Vec<DashboardWidget>,
    /// Update interval
    update_interval: Duration,
    /// Data sources
    data_sources: Vec<DataSource>,
}

/// Dashboard widget
#[derive(Debug, Clone)]
pub struct DashboardWidget {
    /// Widget type
    pub widget_type: WidgetType,
    /// Title
    pub title: String,
    /// Metrics
    pub metrics: Vec<String>,
    /// Time range
    pub time_range: TimeRange,
}

/// Widget types
#[derive(Debug, Clone)]
pub enum WidgetType {
    LineChart,
    BarChart,
    Gauge,
    Table,
    Heatmap,
    Counter,
}

/// Time range
#[derive(Debug, Clone)]
pub struct TimeRange {
    /// Start time
    pub start: Instant,
    /// End time
    pub end: Instant,
    /// Interval
    pub interval: Duration,
}

/// Data source
#[derive(Debug, Clone)]
pub struct DataSource {
    /// Source name
    pub name: String,
    /// Source type
    pub source_type: DataSourceType,
    /// Connection config
    pub config: HashMap<String, String>,
}

/// Data source types
#[derive(Debug, Clone)]
pub enum DataSourceType {
    Prometheus,
    InfluxDB,
    CloudWatch,
    Datadog,
    Custom,
}

/// Health check
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Check name
    pub name: String,
    /// Check type
    pub check_type: HealthCheckType,
    /// Interval
    pub interval: Duration,
    /// Timeout
    pub timeout: Duration,
    /// Enabled
    pub enabled: bool,
}

/// Health check types
#[derive(Debug, Clone)]
pub enum HealthCheckType {
    Ping,
    HTTPGet,
    TCPConnect,
    Custom,
}

/// Transfer manager for parallel operations
#[derive(Debug)]
pub struct ParallelTransferManager {
    /// Active transfers
    active_transfers: HashMap<String, TransferJob>,
    /// Transfer queue
    transfer_queue: Vec<TransferJob>,
    /// Thread pool
    thread_pool: ThreadPool,
    /// Performance metrics
    performance_metrics: TransferManagerMetrics,
}

/// Transfer job
#[derive(Debug, Clone)]
pub struct TransferJob {
    /// Job ID
    pub id: String,
    /// Job type
    pub job_type: TransferJobType,
    /// Priority
    pub priority: TransferPriority,
    /// Progress
    pub progress: TransferProgress,
    /// Status
    pub status: TransferStatus,
    /// Created timestamp
    pub created: Instant,
    /// Estimated completion
    pub estimated_completion: Option<Instant>,
}

/// Transfer job types
#[derive(Debug, Clone)]
pub enum TransferJobType {
    Upload,
    Download,
    Copy,
    Sync,
    Backup,
}

/// Transfer priorities
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransferPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Transfer progress
#[derive(Debug, Clone)]
pub struct TransferProgress {
    /// Bytes transferred
    pub bytes_transferred: u64,
    /// Total bytes
    pub total_bytes: u64,
    /// Progress percentage
    pub percentage: f64,
    /// Transfer rate (MB/s)
    pub transfer_rate_mbps: f64,
    /// Estimated time remaining
    pub eta: Option<Duration>,
}

/// Transfer status
#[derive(Debug, Clone)]
pub enum TransferStatus {
    Queued,
    Running,
    Paused,
    Completed,
    Failed(String),
    Cancelled,
}

/// Thread pool
#[derive(Debug)]
pub struct ThreadPool {
    /// Number of threads
    pub thread_count: usize,
    /// Queue size
    pub queue_size: usize,
    /// Active tasks
    pub active_tasks: usize,
}

/// Transfer manager metrics
#[derive(Debug, Clone)]
pub struct TransferManagerMetrics {
    /// Total transfers
    pub total_transfers: u64,
    /// Successful transfers
    pub successful_transfers: u64,
    /// Failed transfers
    pub failed_transfers: u64,
    /// Average transfer rate
    pub avg_transfer_rate_mbps: f64,
    /// Queue efficiency
    pub queue_efficiency: f64,
}

/// Monitoring statistics
#[derive(Debug, Clone)]
pub struct MonitoringStatistics {
    /// Active alerts count
    pub active_alerts: u32,
    /// Metrics collected per minute
    pub metrics_per_minute: u64,
    /// Dashboard widgets count
    pub dashboard_widgets: u32,
    /// Health checks count
    pub health_checks: u32,
    /// Data sources count
    pub data_sources: u32,
}

// Implementations

impl Default for CloudStorageMonitoring {
    fn default() -> Self {
        Self::new()
    }
}

impl CloudStorageMonitoring {
    pub fn new() -> Self {
        Self {
            metrics_collectors: vec![MetricsCollector {
                name: "performance_collector".to_string(),
                metric_types: vec![MetricType::Latency, MetricType::Throughput],
                collection_interval: Duration::from_secs(60),
                data_retention: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
            }],
            alert_manager: AlertManager {
                active_alerts: Vec::new(),
                alert_rules: vec![AlertRule {
                    name: "high_latency".to_string(),
                    condition: AlertCondition {
                        metric: "latency".to_string(),
                        operator: ComparisonOperator::GreaterThan,
                        time_window: Duration::from_secs(300),
                    },
                    threshold: 1000.0, // 1 second
                    evaluation_interval: Duration::from_secs(60),
                }],
                notification_channels: vec![NotificationChannel {
                    channel_type: NotificationChannelType::Email,
                    config: {
                        let mut config = HashMap::new();
                        config.insert("address".to_string(), "admin@example.com".to_string());
                        config
                    },
                    enabled: true,
                }],
            },
            dashboard: PerformanceDashboard {
                widgets: vec![DashboardWidget {
                    widget_type: WidgetType::LineChart,
                    title: "Response Time".to_string(),
                    metrics: vec!["latency".to_string()],
                    time_range: TimeRange {
                        start: Instant::now() - Duration::from_secs(3600),
                        end: Instant::now(),
                        interval: Duration::from_secs(60),
                    },
                }],
                update_interval: Duration::from_secs(30),
                data_sources: vec![DataSource {
                    name: "prometheus".to_string(),
                    source_type: DataSourceType::Prometheus,
                    config: HashMap::new(),
                }],
            },
            health_checks: vec![HealthCheck {
                name: "endpoint_health".to_string(),
                check_type: HealthCheckType::HTTPGet,
                interval: Duration::from_secs(30),
                timeout: Duration::from_secs(10),
                enabled: true,
            }],
        }
    }

    /// Start monitoring system
    pub fn start_monitoring(&mut self) -> CoreResult<()> {
        // Initialize metrics collection
        for collector in &mut self.metrics_collectors {
            self.start_metrics_collector(collector)?;
        }

        // Start health checks
        for health_check in &mut self.health_checks {
            if health_check.enabled {
                self.start_health_check(health_check)?;
            }
        }

        // Initialize dashboard
        self.start_dashboard()?;

        println!("✅ Cloud storage monitoring started");
        Ok(())
    }

    /// Stop monitoring system
    pub fn stop_monitoring(&mut self) -> CoreResult<()> {
        // Stop all active monitoring
        println!("🛑 Cloud storage monitoring stopped");
        Ok(())
    }

    /// Record a metric
    pub fn record_metric(&mut self, metric_type: MetricType, value: f64, tags: HashMap<String, String>) -> CoreResult<()> {
        // Evaluate alert rules
        self.evaluate_alerts(&metric_type, value)?;

        // Store metric (in real implementation, would send to time series database)
        println!("📊 Recorded metric {:?}: {} with tags {:?}", metric_type, value, tags);
        Ok(())
    }

    /// Create an alert
    pub fn create_alert(&mut self, level: AlertLevel, message: String, source: String) -> CoreResult<String> {
        let alert_id = format!("alert_{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos());

        let alert = Alert {
            id: alert_id.clone(),
            level,
            message: message.clone(),
            source,
            timestamp: Instant::now(),
            acknowledged: false,
        };

        self.alert_manager.active_alerts.push(alert);

        // Send notifications
        self.send_alert_notifications(&message)?;

        Ok(alert_id)
    }

    /// Acknowledge an alert
    pub fn acknowledge_alert(&mut self, alert_id: &str) -> CoreResult<bool> {
        for alert in &mut self.alert_manager.active_alerts {
            if alert.id == alert_id {
                alert.acknowledged = true;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<Alert> {
        self.alert_manager.active_alerts.clone()
    }

    /// Add a new alert rule
    pub fn add_alert_rule(&mut self, rule: AlertRule) -> CoreResult<()> {
        self.alert_manager.alert_rules.push(rule);
        Ok(())
    }

    /// Add a dashboard widget
    pub fn add_dashboard_widget(&mut self, widget: DashboardWidget) -> CoreResult<()> {
        self.dashboard.widgets.push(widget);
        Ok(())
    }

    /// Get monitoring statistics
    pub fn get_monitoring_statistics(&self) -> MonitoringStatistics {
        MonitoringStatistics {
            active_alerts: self.alert_manager.active_alerts.len() as u32,
            metrics_per_minute: 60, // Simulated
            dashboard_widgets: self.dashboard.widgets.len() as u32,
            health_checks: self.health_checks.len() as u32,
            data_sources: self.dashboard.data_sources.len() as u32,
        }
    }

    /// Perform health check
    pub fn perform_health_check(&self, check_name: &str) -> CoreResult<HealthCheckResult> {
        for health_check in &self.health_checks {
            if health_check.name == check_name {
                return self.execute_health_check(health_check);
            }
        }

        Err(CoreError::InvalidArgument(
            crate::error::ErrorContext::new(
                format!("Health check '{}' not found", check_name)
            )
        ))
    }

    // Private helper methods

    fn start_metrics_collector(&self, _collector: &MetricsCollector) -> CoreResult<()> {
        // Start metrics collection (would run in background thread)
        Ok(())
    }

    fn start_health_check(&self, _health_check: &HealthCheck) -> CoreResult<()> {
        // Start health check (would run in background thread)
        Ok(())
    }

    fn start_dashboard(&self) -> CoreResult<()> {
        // Initialize dashboard (would start web server)
        Ok(())
    }

    fn evaluate_alerts(&mut self, metric_type: &MetricType, value: f64) -> CoreResult<()> {
        for rule in &self.alert_manager.alert_rules.clone() {
            if self.rule_matches_metric(rule, metric_type) && self.evaluate_threshold(rule, value) {
                self.create_alert(
                    AlertLevel::Warning,
                    format!("Threshold exceeded: {} = {}", rule.condition.metric, value),
                    "monitoring_system".to_string(),
                )?;
            }
        }
        Ok(())
    }

    fn rule_matches_metric(&self, rule: &AlertRule, metric_type: &MetricType) -> bool {
        match metric_type {
            MetricType::Latency => rule.condition.metric == "latency",
            MetricType::Throughput => rule.condition.metric == "throughput",
            MetricType::ErrorRate => rule.condition.metric == "error_rate",
            MetricType::Cost => rule.condition.metric == "cost",
            MetricType::Availability => rule.condition.metric == "availability",
            MetricType::Storage => rule.condition.metric == "storage",
            MetricType::Bandwidth => rule.condition.metric == "bandwidth",
        }
    }

    fn evaluate_threshold(&self, rule: &AlertRule, value: f64) -> bool {
        match rule.condition.operator {
            ComparisonOperator::GreaterThan => value > rule.threshold,
            ComparisonOperator::LessThan => value < rule.threshold,
            ComparisonOperator::Equal => (value - rule.threshold).abs() < f64::EPSILON,
            ComparisonOperator::NotEqual => (value - rule.threshold).abs() >= f64::EPSILON,
            ComparisonOperator::GreaterOrEqual => value >= rule.threshold,
            ComparisonOperator::LessOrEqual => value <= rule.threshold,
        }
    }

    fn send_alert_notifications(&self, message: &str) -> CoreResult<()> {
        for channel in &self.alert_manager.notification_channels {
            if channel.enabled {
                self.send_notification(channel, message)?;
            }
        }
        Ok(())
    }

    fn send_notification(&self, channel: &NotificationChannel, message: &str) -> CoreResult<()> {
        match channel.channel_type {
            NotificationChannelType::Email => {
                println!("📧 Email notification: {}", message);
            }
            NotificationChannelType::Slack => {
                println!("💬 Slack notification: {}", message);
            }
            NotificationChannelType::Webhook => {
                println!("🔗 Webhook notification: {}", message);
            }
            NotificationChannelType::SMS => {
                println!("📱 SMS notification: {}", message);
            }
            NotificationChannelType::PagerDuty => {
                println!("📟 PagerDuty notification: {}", message);
            }
        }
        Ok(())
    }

    fn execute_health_check(&self, health_check: &HealthCheck) -> CoreResult<HealthCheckResult> {
        // Simulate health check execution
        let success = true; // In real implementation, would perform actual check

        Ok(HealthCheckResult {
            check_name: health_check.name.clone(),
            success,
            response_time: Duration::from_millis(50),
            message: if success { "OK".to_string() } else { "Failed".to_string() },
            timestamp: Instant::now(),
        })
    }
}

impl Default for ParallelTransferManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelTransferManager {
    pub fn new() -> Self {
        Self {
            active_transfers: HashMap::new(),
            transfer_queue: Vec::new(),
            thread_pool: ThreadPool {
                thread_count: 8,
                queue_size: 100,
                active_tasks: 0,
            },
            performance_metrics: TransferManagerMetrics {
                total_transfers: 0,
                successful_transfers: 0,
                failed_transfers: 0,
                avg_transfer_rate_mbps: 0.0,
                queue_efficiency: 0.0,
            },
        }
    }

    /// Submit a transfer job
    pub fn submit_transfer(&mut self, job: TransferJob) -> CoreResult<String> {
        let job_id = job.id.clone();
        self.transfer_queue.push(job);
        self.performance_metrics.total_transfers += 1;
        Ok(job_id)
    }

    /// Get transfer status
    pub fn get_transfer_status(&self, job_id: &str) -> Option<TransferStatus> {
        self.active_transfers.get(job_id).map(|job| job.status.clone())
    }

    /// Cancel a transfer
    pub fn cancel_transfer(&mut self, job_id: &str) -> CoreResult<bool> {
        if let Some(job) = self.active_transfers.get_mut(job_id) {
            job.status = TransferStatus::Cancelled;
            return Ok(true);
        }
        Ok(false)
    }

    /// Get transfer metrics
    pub fn get_transfer_metrics(&self) -> TransferManagerMetrics {
        self.performance_metrics.clone()
    }
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Check name
    pub check_name: String,
    /// Success flag
    pub success: bool,
    /// Response time
    pub response_time: Duration,
    /// Result message
    pub message: String,
    /// Timestamp
    pub timestamp: Instant,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitoring_system_creation() {
        let monitoring = CloudStorageMonitoring::new();
        assert!(!monitoring.metrics_collectors.is_empty());
        assert!(!monitoring.alert_manager.alert_rules.is_empty());
        assert!(!monitoring.dashboard.widgets.is_empty());
    }

    #[test]
    fn test_alert_creation() {
        let mut monitoring = CloudStorageMonitoring::new();

        let alert_id = monitoring.create_alert(
            AlertLevel::Warning,
            "Test alert".to_string(),
            "test_source".to_string(),
        ).unwrap();

        assert!(!alert_id.is_empty());
        assert_eq!(monitoring.get_active_alerts().len(), 1);
    }

    #[test]
    fn test_alert_acknowledgment() {
        let mut monitoring = CloudStorageMonitoring::new();

        let alert_id = monitoring.create_alert(
            AlertLevel::Warning,
            "Test alert".to_string(),
            "test_source".to_string(),
        ).unwrap();

        let acknowledged = monitoring.acknowledge_alert(&alert_id).unwrap();
        assert!(acknowledged);

        let alerts = monitoring.get_active_alerts();
        assert!(alerts[0].acknowledged);
    }

    #[test]
    fn test_metric_recording() {
        let mut monitoring = CloudStorageMonitoring::new();
        let mut tags = HashMap::new();
        tags.insert("provider".to_string(), "test".to_string());

        monitoring.record_metric(MetricType::Latency, 150.0, tags).unwrap();
        // Should not panic or error
    }

    #[test]
    fn test_parallel_transfer_manager() {
        let mut manager = ParallelTransferManager::new();

        let job = TransferJob {
            id: "test_job".to_string(),
            job_type: TransferJobType::Upload,
            priority: TransferPriority::Normal,
            progress: TransferProgress {
                bytes_transferred: 0,
                total_bytes: 1000,
                percentage: 0.0,
                transfer_rate_mbps: 0.0,
                eta: None,
            },
            status: TransferStatus::Queued,
            created: Instant::now(),
            estimated_completion: None,
        };

        let job_id = manager.submit_transfer(job).unwrap();
        assert_eq!(job_id, "test_job");
        assert_eq!(manager.performance_metrics.total_transfers, 1);
    }
}