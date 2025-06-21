//! # System Resource Monitoring
//!
//! This module provides comprehensive system resource monitoring capabilities
//! for correlating application performance with system-level metrics.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Error types for system monitoring
#[derive(Error, Debug)]
pub enum SystemMonitorError {
    /// Failed to read system information
    #[error("Failed to read system information: {0}")]
    SystemReadError(String),

    /// Monitoring not supported on this platform
    #[error("System monitoring not supported on this platform")]
    UnsupportedPlatform,

    /// Permission denied for system monitoring
    #[error("Permission denied for system monitoring")]
    PermissionDenied,

    /// Monitor not running
    #[error("System monitor is not running")]
    NotRunning,
}

/// System resource metrics
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    /// Timestamp when metrics were collected
    pub timestamp: Instant,
    /// CPU usage percentage (0.0 to 100.0)
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Available memory in bytes
    pub memory_available: usize,
    /// Total memory in bytes
    pub memory_total: usize,
    /// Disk I/O read bytes per second
    pub disk_read_bps: u64,
    /// Disk I/O write bytes per second
    pub disk_write_bps: u64,
    /// Network received bytes per second
    pub network_rx_bps: u64,
    /// Network transmitted bytes per second
    pub network_tx_bps: u64,
    /// Number of running processes
    pub process_count: usize,
    /// System load average (1-minute)
    pub load_average: f64,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            cpu_usage: 0.0,
            memory_usage: 0,
            memory_available: 0,
            memory_total: 0,
            disk_read_bps: 0,
            disk_write_bps: 0,
            network_rx_bps: 0,
            network_tx_bps: 0,
            process_count: 0,
            load_average: 0.0,
        }
    }
}

/// System monitoring configuration
#[derive(Debug, Clone)]
pub struct SystemMonitorConfig {
    /// Sampling interval for system metrics
    pub sampling_interval: Duration,
    /// Maximum number of samples to keep in memory
    pub max_samples: usize,
    /// Enable CPU monitoring
    pub monitor_cpu: bool,
    /// Enable memory monitoring
    pub monitor_memory: bool,
    /// Enable disk I/O monitoring
    pub monitor_disk: bool,
    /// Enable network monitoring
    pub monitor_network: bool,
    /// Enable process monitoring
    pub monitor_processes: bool,
}

impl Default for SystemMonitorConfig {
    fn default() -> Self {
        Self {
            sampling_interval: Duration::from_millis(500),
            max_samples: 1000,
            monitor_cpu: true,
            monitor_memory: true,
            monitor_disk: true,
            monitor_network: true,
            monitor_processes: true,
        }
    }
}

/// System resource monitor
pub struct SystemMonitor {
    config: SystemMonitorConfig,
    metrics_history: Arc<Mutex<VecDeque<SystemMetrics>>>,
    running: Arc<Mutex<bool>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl SystemMonitor {
    /// Create a new system monitor
    pub fn new(config: SystemMonitorConfig) -> Self {
        Self {
            config,
            metrics_history: Arc::new(Mutex::new(VecDeque::new())),
            running: Arc::new(Mutex::new(false)),
            handle: None,
        }
    }

    /// Start monitoring system resources
    pub fn start(&mut self) -> Result<(), SystemMonitorError> {
        let mut running = self.running.lock().unwrap();
        if *running {
            return Ok(()); // Already running
        }
        *running = true;

        let config = self.config.clone();
        let metrics_history = Arc::clone(&self.metrics_history);
        let running_flag = Arc::clone(&self.running);

        self.handle = Some(thread::spawn(move || {
            Self::monitoring_loop(config, metrics_history, running_flag);
        }));

        Ok(())
    }

    /// Stop monitoring
    pub fn stop(&mut self) {
        if let Ok(mut running) = self.running.lock() {
            *running = false;
        }

        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// Get current system metrics
    pub fn get_current_metrics(&self) -> Result<SystemMetrics, SystemMonitorError> {
        Self::collect_system_metrics(&self.config)
    }

    /// Get metrics history
    pub fn get_metrics_history(&self) -> Vec<SystemMetrics> {
        self.metrics_history
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .collect()
    }

    /// Get latest N metrics
    pub fn get_latest_metrics(&self, n: usize) -> Vec<SystemMetrics> {
        let history = self.metrics_history.lock().unwrap();
        history.iter().rev().take(n).cloned().collect()
    }

    /// Get metrics within time range
    pub fn get_metrics_in_range(&self, start: Instant, end: Instant) -> Vec<SystemMetrics> {
        self.metrics_history
            .lock()
            .unwrap()
            .iter()
            .filter(|m| m.timestamp >= start && m.timestamp <= end)
            .cloned()
            .collect()
    }

    /// Calculate average metrics over time period
    pub fn get_average_metrics(&self, duration: Duration) -> Option<SystemMetrics> {
        let now = Instant::now();
        let start = now - duration;
        let metrics = self.get_metrics_in_range(start, now);

        if metrics.is_empty() {
            return None;
        }

        let count = metrics.len() as f64;
        let avg_cpu = metrics.iter().map(|m| m.cpu_usage).sum::<f64>() / count;
        let avg_memory =
            (metrics.iter().map(|m| m.memory_usage).sum::<usize>() as f64 / count) as usize;
        let avg_disk_read =
            (metrics.iter().map(|m| m.disk_read_bps).sum::<u64>() as f64 / count) as u64;
        let avg_disk_write =
            (metrics.iter().map(|m| m.disk_write_bps).sum::<u64>() as f64 / count) as u64;
        let avg_network_rx =
            (metrics.iter().map(|m| m.network_rx_bps).sum::<u64>() as f64 / count) as u64;
        let avg_network_tx =
            (metrics.iter().map(|m| m.network_tx_bps).sum::<u64>() as f64 / count) as u64;
        let avg_processes =
            (metrics.iter().map(|m| m.process_count).sum::<usize>() as f64 / count) as usize;
        let avg_load = metrics.iter().map(|m| m.load_average).sum::<f64>() / count;

        Some(SystemMetrics {
            timestamp: now,
            cpu_usage: avg_cpu,
            memory_usage: avg_memory,
            memory_available: metrics.last()?.memory_available,
            memory_total: metrics.last()?.memory_total,
            disk_read_bps: avg_disk_read,
            disk_write_bps: avg_disk_write,
            network_rx_bps: avg_network_rx,
            network_tx_bps: avg_network_tx,
            process_count: avg_processes,
            load_average: avg_load,
        })
    }

    /// Monitoring loop (runs in background thread)
    fn monitoring_loop(
        config: SystemMonitorConfig,
        metrics_history: Arc<Mutex<VecDeque<SystemMetrics>>>,
        running: Arc<Mutex<bool>>,
    ) {
        while *running.lock().unwrap() {
            if let Ok(metrics) = Self::collect_system_metrics(&config) {
                let mut history = metrics_history.lock().unwrap();
                history.push_back(metrics);

                // Keep only the last max_samples
                while history.len() > config.max_samples {
                    history.pop_front();
                }
            }

            thread::sleep(config.sampling_interval);
        }
    }

    /// Collect current system metrics
    fn collect_system_metrics(
        config: &SystemMonitorConfig,
    ) -> Result<SystemMetrics, SystemMonitorError> {
        let mut metrics = SystemMetrics::default();

        if config.monitor_cpu {
            metrics.cpu_usage = Self::get_cpu_usage()?;
        }

        if config.monitor_memory {
            let (used, available, total) = Self::get_memory_info()?;
            metrics.memory_usage = used;
            metrics.memory_available = available;
            metrics.memory_total = total;
        }

        if config.monitor_disk {
            let (read_bps, write_bps) = Self::get_disk_io()?;
            metrics.disk_read_bps = read_bps;
            metrics.disk_write_bps = write_bps;
        }

        if config.monitor_network {
            let (rx_bps, tx_bps) = Self::get_network_io()?;
            metrics.network_rx_bps = rx_bps;
            metrics.network_tx_bps = tx_bps;
        }

        if config.monitor_processes {
            metrics.process_count = Self::get_process_count()?;
        }

        metrics.load_average = Self::get_load_average()?;

        Ok(metrics)
    }

    /// Get CPU usage percentage
    fn get_cpu_usage() -> Result<f64, SystemMonitorError> {
        #[cfg(target_os = "linux")]
        {
            Self::get_cpu_usage_linux()
        }

        #[cfg(target_os = "macos")]
        {
            Self::get_cpu_usage_macos()
        }

        #[cfg(target_os = "windows")]
        {
            Self::get_cpu_usage_windows()
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            // Fallback for unsupported platforms
            Ok(0.0)
        }
    }

    /// Get memory information (used, available, total)
    fn get_memory_info() -> Result<(usize, usize, usize), SystemMonitorError> {
        #[cfg(target_os = "linux")]
        {
            Self::get_memory_info_linux()
        }

        #[cfg(target_os = "macos")]
        {
            Self::get_memory_info_macos()
        }

        #[cfg(target_os = "windows")]
        {
            Self::get_memory_info_windows()
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            Ok((0, 0, 0))
        }
    }

    /// Get disk I/O rates (read bps, write bps)
    fn get_disk_io() -> Result<(u64, u64), SystemMonitorError> {
        // Simplified implementation - real version would track deltas
        Ok((0, 0))
    }

    /// Get network I/O rates (rx bps, tx bps)
    fn get_network_io() -> Result<(u64, u64), SystemMonitorError> {
        // Simplified implementation - real version would track deltas
        Ok((0, 0))
    }

    /// Get number of running processes
    fn get_process_count() -> Result<usize, SystemMonitorError> {
        #[cfg(target_os = "linux")]
        {
            match std::fs::read_dir("/proc") {
                Ok(entries) => {
                    let count = entries
                        .filter_map(|entry| entry.ok())
                        .filter(|entry| {
                            entry
                                .file_name()
                                .to_string_lossy()
                                .chars()
                                .all(|c| c.is_ascii_digit())
                        })
                        .count();
                    Ok(count)
                }
                Err(e) => Err(SystemMonitorError::SystemReadError(e.to_string())),
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            Ok(0)
        }
    }

    /// Get system load average
    fn get_load_average() -> Result<f64, SystemMonitorError> {
        #[cfg(target_os = "linux")]
        {
            match std::fs::read_to_string("/proc/loadavg") {
                Ok(content) => {
                    let load = content
                        .split_whitespace()
                        .next()
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    Ok(load)
                }
                Err(e) => Err(SystemMonitorError::SystemReadError(e.to_string())),
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            Ok(0.0)
        }
    }

    // Platform-specific implementations

    #[cfg(target_os = "linux")]
    fn get_cpu_usage_linux() -> Result<f64, SystemMonitorError> {
        use std::fs;

        // Read /proc/stat for CPU usage
        let stat1 = fs::read_to_string("/proc/stat")
            .map_err(|e| SystemMonitorError::SystemReadError(e.to_string()))?;

        thread::sleep(Duration::from_millis(100));

        let stat2 = fs::read_to_string("/proc/stat")
            .map_err(|e| SystemMonitorError::SystemReadError(e.to_string()))?;

        let cpu1 = Self::parse_cpu_line(&stat1)?;
        let cpu2 = Self::parse_cpu_line(&stat2)?;

        let total1 = cpu1.iter().sum::<u64>();
        let total2 = cpu2.iter().sum::<u64>();
        let idle1 = cpu1[3]; // idle time
        let idle2 = cpu2[3];

        let total_diff = total2 - total1;
        let idle_diff = idle2 - idle1;

        if total_diff == 0 {
            Ok(0.0)
        } else {
            let usage = 100.0 - (idle_diff as f64 / total_diff as f64) * 100.0;
            Ok(usage.clamp(0.0, 100.0))
        }
    }

    #[cfg(target_os = "linux")]
    fn parse_cpu_line(stat: &str) -> Result<Vec<u64>, SystemMonitorError> {
        let first_line = stat
            .lines()
            .next()
            .ok_or_else(|| SystemMonitorError::SystemReadError("Empty /proc/stat".to_string()))?;

        let values: Result<Vec<u64>, _> = first_line
            .split_whitespace()
            .skip(1) // Skip "cpu"
            .map(|s| s.parse::<u64>())
            .collect();

        values.map_err(|e| SystemMonitorError::SystemReadError(e.to_string()))
    }

    #[cfg(target_os = "linux")]
    fn get_memory_info_linux() -> Result<(usize, usize, usize), SystemMonitorError> {
        use std::fs;

        let meminfo = fs::read_to_string("/proc/meminfo")
            .map_err(|e| SystemMonitorError::SystemReadError(e.to_string()))?;

        let mut mem_total = 0;
        let mut mem_available = 0;

        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                mem_total = Self::parse_memory_line(line)?;
            } else if line.starts_with("MemAvailable:") {
                mem_available = Self::parse_memory_line(line)?;
            }
        }

        let mem_used = mem_total.saturating_sub(mem_available);
        Ok((mem_used, mem_available, mem_total))
    }

    #[cfg(target_os = "linux")]
    fn parse_memory_line(line: &str) -> Result<usize, SystemMonitorError> {
        let kb = line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| {
                SystemMonitorError::SystemReadError("Invalid memory line".to_string())
            })?;

        Ok(kb * 1024) // Convert from KB to bytes
    }

    #[cfg(target_os = "macos")]
    fn get_cpu_usage_macos() -> Result<f64, SystemMonitorError> {
        // Would use system APIs like host_processor_info
        Ok(0.0)
    }

    #[cfg(target_os = "macos")]
    fn get_memory_info_macos() -> Result<(usize, usize, usize), SystemMonitorError> {
        // Would use system APIs like vm_statistics64
        Ok((0, 0, 0))
    }

    #[cfg(target_os = "windows")]
    fn get_cpu_usage_windows() -> Result<f64, SystemMonitorError> {
        // Would use Windows APIs like GetSystemTimes
        Ok(0.0)
    }

    #[cfg(target_os = "windows")]
    fn get_memory_info_windows() -> Result<(usize, usize, usize), SystemMonitorError> {
        // Would use Windows APIs like GlobalMemoryStatusEx
        Ok((0, 0, 0))
    }
}

impl Drop for SystemMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

/// System resource alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// CPU usage threshold for alerts (percentage)
    pub cpu_threshold: f64,
    /// Memory usage threshold for alerts (percentage)
    pub memory_threshold: f64,
    /// Disk I/O threshold for alerts (bytes per second)
    pub disk_io_threshold: u64,
    /// Network I/O threshold for alerts (bytes per second)
    pub network_io_threshold: u64,
    /// Load average threshold for alerts
    pub load_threshold: f64,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            cpu_threshold: 80.0,
            memory_threshold: 85.0,
            disk_io_threshold: 100 * 1024 * 1024,   // 100 MB/s
            network_io_threshold: 50 * 1024 * 1024, // 50 MB/s
            load_threshold: 2.0,
        }
    }
}

/// System resource alert
#[derive(Debug, Clone)]
pub struct SystemAlert {
    /// Alert type
    pub alert_type: AlertType,
    /// Current value that triggered the alert
    pub current_value: f64,
    /// Threshold that was exceeded
    pub threshold: f64,
    /// Timestamp when alert was triggered
    pub timestamp: Instant,
    /// Severity level
    pub severity: AlertSeverity,
    /// Human-readable message
    pub message: String,
}

/// Alert types
#[derive(Debug, Clone, PartialEq)]
pub enum AlertType {
    HighCpuUsage,
    HighMemoryUsage,
    HighDiskIo,
    HighNetworkIo,
    HighLoadAverage,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// System resource alerting system
pub struct SystemAlerter {
    config: AlertConfig,
    alert_history: VecDeque<SystemAlert>,
    max_alert_history: usize,
}

impl SystemAlerter {
    /// Create a new system alerter
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            alert_history: VecDeque::new(),
            max_alert_history: 1000,
        }
    }

    /// Check metrics against thresholds and generate alerts
    pub fn check_alerts(&mut self, metrics: &SystemMetrics) -> Vec<SystemAlert> {
        let mut alerts = Vec::new();

        // Check CPU usage
        if metrics.cpu_usage > self.config.cpu_threshold {
            alerts.push(self.create_alert(
                AlertType::HighCpuUsage,
                metrics.cpu_usage,
                self.config.cpu_threshold,
                format!("High CPU usage: {:.1}%", metrics.cpu_usage),
            ));
        }

        // Check memory usage
        if metrics.memory_total > 0 {
            let memory_percent =
                (metrics.memory_usage as f64 / metrics.memory_total as f64) * 100.0;
            if memory_percent > self.config.memory_threshold {
                alerts.push(self.create_alert(
                    AlertType::HighMemoryUsage,
                    memory_percent,
                    self.config.memory_threshold,
                    format!("High memory usage: {:.1}%", memory_percent),
                ));
            }
        }

        // Check disk I/O
        let total_disk_io = metrics.disk_read_bps + metrics.disk_write_bps;
        if total_disk_io > self.config.disk_io_threshold {
            alerts.push(self.create_alert(
                AlertType::HighDiskIo,
                total_disk_io as f64,
                self.config.disk_io_threshold as f64,
                format!(
                    "High disk I/O: {:.1} MB/s",
                    total_disk_io as f64 / (1024.0 * 1024.0)
                ),
            ));
        }

        // Check network I/O
        let total_network_io = metrics.network_rx_bps + metrics.network_tx_bps;
        if total_network_io > self.config.network_io_threshold {
            alerts.push(self.create_alert(
                AlertType::HighNetworkIo,
                total_network_io as f64,
                self.config.network_io_threshold as f64,
                format!(
                    "High network I/O: {:.1} MB/s",
                    total_network_io as f64 / (1024.0 * 1024.0)
                ),
            ));
        }

        // Check load average
        if metrics.load_average > self.config.load_threshold {
            alerts.push(self.create_alert(
                AlertType::HighLoadAverage,
                metrics.load_average,
                self.config.load_threshold,
                format!("High load average: {:.2}", metrics.load_average),
            ));
        }

        // Store alerts in history
        for alert in &alerts {
            self.alert_history.push_back(alert.clone());
            while self.alert_history.len() > self.max_alert_history {
                self.alert_history.pop_front();
            }
        }

        alerts
    }

    /// Create an alert with appropriate severity
    fn create_alert(
        &self,
        alert_type: AlertType,
        current: f64,
        threshold: f64,
        message: String,
    ) -> SystemAlert {
        let severity = if current > threshold * 2.0 {
            AlertSeverity::Critical
        } else if current > threshold * 1.5 {
            AlertSeverity::Warning
        } else {
            AlertSeverity::Info
        };

        SystemAlert {
            alert_type,
            current_value: current,
            threshold,
            timestamp: Instant::now(),
            severity,
            message,
        }
    }

    /// Get alert history
    pub fn get_alert_history(&self) -> Vec<SystemAlert> {
        self.alert_history.iter().cloned().collect()
    }

    /// Get recent alerts
    pub fn get_recent_alerts(&self, duration: Duration) -> Vec<SystemAlert> {
        let cutoff = Instant::now() - duration;
        self.alert_history
            .iter()
            .filter(|alert| alert.timestamp >= cutoff)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_monitor_creation() {
        let config = SystemMonitorConfig::default();
        let monitor = SystemMonitor::new(config);
        assert!(!*monitor.running.lock().unwrap());
    }

    #[test]
    fn test_alert_creation() {
        let config = AlertConfig::default();
        let mut alerter = SystemAlerter::new(config);

        let metrics = SystemMetrics {
            cpu_usage: 90.0, // Above threshold
            ..Default::default()
        };

        let alerts = alerter.check_alerts(&metrics);
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].alert_type, AlertType::HighCpuUsage);
    }

    #[test]
    fn test_metrics_averaging() {
        let config = SystemMonitorConfig::default();
        let monitor = SystemMonitor::new(config);

        // Simulate some metrics
        {
            let mut history = monitor.metrics_history.lock().unwrap();
            for i in 0..10 {
                let metrics = SystemMetrics {
                    cpu_usage: i as f64 * 10.0,
                    timestamp: Instant::now() - Duration::from_secs(i),
                    ..Default::default()
                };
                history.push_back(metrics);
            }
        }

        let avg = monitor.get_average_metrics(Duration::from_secs(100));
        assert!(avg.is_some());
    }
}
