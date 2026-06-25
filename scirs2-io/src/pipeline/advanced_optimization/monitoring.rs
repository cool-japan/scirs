//! System resource monitoring for pipeline optimization
//!
//! This module provides real-time monitoring of system resources including
//! CPU usage, memory utilization, I/O performance, and cache efficiency.

use crate::error::{IoError, Result};
use std::cell::Cell;
use std::time::{Duration, Instant};

use super::config::{CachePerformance, MemoryUsage, NumaNode, NumaTopology, SystemMetrics};

/// Cumulative network byte counters read from `/proc/net/dev`, paired with the
/// instant they were sampled. Used to derive a real transfer rate from the
/// delta between two reads.
#[derive(Debug, Clone, Copy)]
struct NetworkSample {
    /// Total received + transmitted bytes across all non-loopback interfaces.
    total_bytes: u64,
    /// When this sample was taken.
    sampled_at: Instant,
}

/// Real-time system resource monitor.
///
/// Where a real OS data source exists (Linux `/proc`), metrics are measured.
/// On platforms without an available data source the corresponding accessor
/// reports a documented "not measured" sentinel rather than a fabricated
/// value; see the individual methods.
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Last update timestamp
    last_update: Instant,
    /// Update frequency
    update_frequency: Duration,
    /// Cached metrics to avoid frequent system calls
    cached_metrics: Option<SystemMetrics>,
    /// Monitoring history for trend analysis
    metrics_history: Vec<SystemMetrics>,
    /// Capture instant for each entry in `metrics_history` (parallel vector),
    /// so trend queries can be filtered by a real time window.
    history_timestamps: Vec<Instant>,
    /// Maximum history size
    max_history_size: usize,
    /// Previous network byte sample, used to compute a real bandwidth rate.
    prev_network_sample: Cell<Option<NetworkSample>>,
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceMonitor {
    pub fn new() -> Self {
        Self {
            last_update: Instant::now(),
            update_frequency: Duration::from_millis(500), // Update every 500ms
            cached_metrics: None,
            metrics_history: Vec::new(),
            history_timestamps: Vec::new(),
            max_history_size: 100, // Keep last 100 samples
            prev_network_sample: Cell::new(None),
        }
    }

    /// Get current system metrics with caching
    pub fn get_current_metrics(&mut self) -> Result<SystemMetrics> {
        let now = Instant::now();

        // Check if we need to update cached metrics
        if self.cached_metrics.is_none()
            || now.duration_since(self.last_update) >= self.update_frequency
        {
            let metrics = self.collect_system_metrics()?;
            self.cached_metrics = Some(metrics.clone());
            self.last_update = now;

            // Add to history (metrics and their capture instant stay in lockstep).
            self.metrics_history.push(metrics.clone());
            self.history_timestamps.push(now);
            if self.metrics_history.len() > self.max_history_size {
                self.metrics_history.remove(0);
                self.history_timestamps.remove(0);
            }

            Ok(metrics)
        } else {
            // Safe: we know cached_metrics is Some because we checked is_none() above
            Ok(self
                .cached_metrics
                .clone()
                .expect("cached_metrics should be Some here"))
        }
    }

    /// Collect fresh system metrics
    fn collect_system_metrics(&self) -> Result<SystemMetrics> {
        Ok(SystemMetrics {
            cpu_usage: self.get_cpu_usage()?,
            memory_usage: self.get_memory_usage()?,
            io_utilization: self.get_io_utilization()?,
            network_bandwidth_usage: self.get_network_usage()?,
            cache_performance: self.get_cache_performance()?,
            numa_topology: self.get_numa_topology()?,
        })
    }

    /// Get CPU usage percentage
    fn get_cpu_usage(&self) -> Result<f64> {
        #[cfg(target_os = "linux")]
        {
            self.get_linux_cpu_usage()
        }
        #[cfg(target_os = "windows")]
        {
            self.get_windows_cpu_usage()
        }
        #[cfg(target_os = "macos")]
        {
            self.get_macos_cpu_usage()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            // No CPU-usage data source is available on this platform. Report a
            // neutral, explicitly-unmeasured value (0.5) rather than pretending
            // to have taken a reading. Callers treating 0.5 as "unknown midpoint"
            // (as SystemMetrics::default does) remain correct.
            Ok(0.5)
        }
    }

    #[cfg(target_os = "linux")]
    fn get_linux_cpu_usage(&self) -> Result<f64> {
        // Read /proc/stat for CPU usage
        let stat_content = std::fs::read_to_string("/proc/stat")
            .map_err(|e| IoError::Other(format!("Failed to read /proc/stat: {}", e)))?;

        if let Some(cpu_line) = stat_content.lines().next() {
            let values: Vec<u64> = cpu_line
                .split_whitespace()
                .skip(1)
                .take(4)
                .filter_map(|s| s.parse().ok())
                .collect();

            if values.len() >= 4 {
                let idle = values[3];
                let total: u64 = values.iter().sum();
                return Ok(1.0 - (idle as f64) / (total as f64));
            }
        }

        // `/proc/stat` was readable but its first line was not the expected
        // aggregate "cpu" row; we cannot derive a reading, so report the neutral
        // unmeasured midpoint rather than fabricating one.
        Ok(0.5)
    }

    #[cfg(target_os = "windows")]
    fn get_windows_cpu_usage(&self) -> Result<f64> {
        // No Windows CPU-counter integration is wired up in this build (would
        // require GetSystemTimes via a platform crate). Report a neutral,
        // explicitly-unmeasured value rather than a fabricated reading.
        Ok(0.5)
    }

    #[cfg(target_os = "macos")]
    fn get_macos_cpu_usage(&self) -> Result<f64> {
        // No macOS CPU-counter integration is wired up in this build (would
        // require host_processor_info via a platform crate). Report a neutral,
        // explicitly-unmeasured value rather than a fabricated reading.
        Ok(0.5)
    }

    /// Get memory usage information
    fn get_memory_usage(&self) -> Result<MemoryUsage> {
        #[cfg(target_os = "linux")]
        {
            self.get_linux_memory_usage()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Ok(MemoryUsage {
                total: 8 * 1024 * 1024 * 1024,     // 8GB fallback
                available: 4 * 1024 * 1024 * 1024, // 4GB fallback
                used: 4 * 1024 * 1024 * 1024,
                utilization: 0.5,
            })
        }
    }

    #[cfg(target_os = "linux")]
    fn get_linux_memory_usage(&self) -> Result<MemoryUsage> {
        let meminfo_content = std::fs::read_to_string("/proc/meminfo")
            .map_err(|e| IoError::Other(format!("Failed to read /proc/meminfo: {}", e)))?;

        let mut total = 0u64;
        let mut available = 0u64;

        for line in meminfo_content.lines() {
            if line.starts_with("MemTotal:") {
                total = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0)
                    * 1024; // Convert KB to bytes
            } else if line.starts_with("MemAvailable:") {
                available = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0)
                    * 1024; // Convert KB to bytes
            }
        }

        let used = total - available;
        let utilization = if total > 0 {
            used as f64 / total as f64
        } else {
            0.0
        };

        Ok(MemoryUsage {
            total,
            available,
            used,
            utilization,
        })
    }

    /// Get I/O utilization (0.0 to 1.0).
    ///
    /// On Linux this is a real measurement of disk queue pressure derived from
    /// `/proc/diskstats`. On other platforms no data source is available, so a
    /// neutral, explicitly-unmeasured value is reported.
    fn get_io_utilization(&self) -> Result<f64> {
        #[cfg(target_os = "linux")]
        {
            self.get_linux_io_utilization()
        }
        #[cfg(not(target_os = "linux"))]
        {
            // No I/O statistics source available on this platform; neutral,
            // explicitly-unmeasured value rather than a fabricated reading.
            Ok(0.3)
        }
    }

    /// Measure disk I/O pressure from `/proc/diskstats`.
    ///
    /// The 12th per-device field is "I/Os currently in progress" (an
    /// instantaneous queue depth). We sum it across physical block devices and
    /// map the aggregate queue depth to a 0..=1 saturation score: a single
    /// in-flight request per CPU is treated as light load, scaling up to fully
    /// saturated as the queue grows. This is a genuine instantaneous reading,
    /// not a constant.
    #[cfg(target_os = "linux")]
    fn get_linux_io_utilization(&self) -> Result<f64> {
        let content = match std::fs::read_to_string("/proc/diskstats") {
            Ok(content) => content,
            // The file is unavailable (e.g. restricted container); we have no
            // measurement, so report the neutral unmeasured value.
            Err(_) => return Ok(0.3),
        };

        let mut in_flight_total: u64 = 0;
        for line in content.lines() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            // Fields: major minor name reads ... (>= 14 expected). Field index
            // 11 is "I/Os currently in progress".
            if fields.len() < 14 {
                continue;
            }
            let name = fields[2];
            // Skip partitions and virtual devices; only aggregate whole physical
            // disks (sdX, nvmeXnY, vdX, hdX, mmcblkX) to avoid double counting.
            let is_physical =
                (name.starts_with("sd") || name.starts_with("vd") || name.starts_with("hd"))
                    && !name
                        .chars()
                        .last()
                        .map(|c| c.is_ascii_digit())
                        .unwrap_or(false)
                    || (name.starts_with("nvme") && !name.contains('p'))
                    || (name.starts_with("mmcblk") && !name.contains('p'));
            if !is_physical {
                continue;
            }
            if let Ok(in_flight) = fields[11].parse::<u64>() {
                in_flight_total += in_flight;
            }
        }

        // Normalize: treat one in-flight request per logical CPU as the point of
        // meaningful pressure; saturate at full when the queue is several times
        // that. This yields 0.0 for an idle disk and approaches 1.0 under load.
        let cpus = num_cpus::get().max(1) as f64;
        let utilization = (in_flight_total as f64 / cpus).tanh();
        Ok(utilization.clamp(0.0, 1.0))
    }

    /// Get network bandwidth usage (0.0 to 1.0).
    ///
    /// On Linux this is a real measurement of throughput derived from the byte
    /// counters in `/proc/net/dev` between successive calls. On other platforms
    /// no data source is available, so a neutral, explicitly-unmeasured value is
    /// reported.
    fn get_network_usage(&self) -> Result<f64> {
        #[cfg(target_os = "linux")]
        {
            self.get_linux_network_usage()
        }
        #[cfg(not(target_os = "linux"))]
        {
            // No network statistics source available on this platform; neutral,
            // explicitly-unmeasured value rather than a fabricated reading.
            Ok(0.2)
        }
    }

    /// Measure network throughput from `/proc/net/dev`.
    ///
    /// Sums received + transmitted bytes across all non-loopback interfaces,
    /// then computes the byte rate against the previously stored sample. The
    /// rate is mapped onto a 0..=1 saturation score relative to a 1 Gbit/s
    /// reference link. The first call has no prior sample, so it records the
    /// baseline and returns 0.0 (no rate observable yet) rather than a
    /// fabricated value.
    #[cfg(target_os = "linux")]
    fn get_linux_network_usage(&self) -> Result<f64> {
        let content = match std::fs::read_to_string("/proc/net/dev") {
            Ok(content) => content,
            // No data source available; record nothing and report the neutral
            // unmeasured value.
            Err(_) => return Ok(0.2),
        };

        let mut total_bytes: u64 = 0;
        for line in content.lines() {
            // Lines look like: "  eth0: <rx_bytes> <rx_pkts> ... <tx_bytes> ...".
            let Some((iface, rest)) = line.split_once(':') else {
                continue;
            };
            let iface = iface.trim();
            // Skip the loopback interface and the two header lines (no ':').
            if iface == "lo" || iface.is_empty() {
                continue;
            }
            let fields: Vec<&str> = rest.split_whitespace().collect();
            // Field 0 is received bytes, field 8 is transmitted bytes.
            if fields.len() < 9 {
                continue;
            }
            let rx = fields[0].parse::<u64>().unwrap_or(0);
            let tx = fields[8].parse::<u64>().unwrap_or(0);
            total_bytes = total_bytes.saturating_add(rx).saturating_add(tx);
        }

        let now = Instant::now();
        let current = NetworkSample {
            total_bytes,
            sampled_at: now,
        };

        let utilization = match self.prev_network_sample.get() {
            Some(prev) => {
                let elapsed = now.duration_since(prev.sampled_at).as_secs_f64();
                if elapsed <= 0.0 {
                    // No measurable interval elapsed; cannot compute a rate.
                    0.0
                } else {
                    let delta_bytes = total_bytes.saturating_sub(prev.total_bytes);
                    let bytes_per_sec = delta_bytes as f64 / elapsed;
                    // Reference: 1 Gbit/s == 125_000_000 bytes/s as full saturation.
                    const REFERENCE_BYTES_PER_SEC: f64 = 125_000_000.0;
                    (bytes_per_sec / REFERENCE_BYTES_PER_SEC).clamp(0.0, 1.0)
                }
            }
            // First observation: no rate is computable yet.
            None => 0.0,
        };

        self.prev_network_sample.set(Some(current));
        Ok(utilization)
    }

    /// Get cache performance metrics.
    ///
    /// Per-level cache hit rates can only be *measured* via hardware
    /// performance counters (`perf_event_open` on Linux, which additionally
    /// needs elevated privileges) and no such integration is wired up here.
    /// These values are therefore NOT measurements: they are nominal hit-rate
    /// assumptions representative of a well-behaved workload on a modern CPU,
    /// used so the optimizer has a sane prior. They are intentionally constant
    /// and should not be interpreted as reflecting the current machine state.
    fn get_cache_performance(&self) -> Result<CachePerformance> {
        Ok(CachePerformance {
            l1_hit_rate: 0.95,
            l2_hit_rate: 0.85,
            l3_hit_rate: 0.75,
            tlb_hit_rate: 0.99,
        })
    }

    /// Get NUMA topology information
    fn get_numa_topology(&self) -> Result<NumaTopology> {
        #[cfg(target_os = "linux")]
        {
            self.get_linux_numa_topology()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Ok(NumaTopology::default())
        }
    }

    /// Read the real NUMA topology from `/sys/devices/system/node/`.
    ///
    /// Each `nodeN` directory describes one NUMA node; we parse its `cpulist`
    /// for the CPUs local to the node and its `meminfo` for the node's total
    /// memory. If sysfs does not expose NUMA information (non-NUMA kernel or a
    /// restricted container) we fall back to a single synthetic node covering
    /// all logical CPUs, which is the correct description of a flat memory
    /// system.
    #[cfg(target_os = "linux")]
    fn get_linux_numa_topology(&self) -> Result<NumaTopology> {
        let entries = match std::fs::read_dir("/sys/devices/system/node/") {
            Ok(entries) => entries,
            Err(_) => return Ok(Self::flat_numa_topology()),
        };

        let mut nodes: Vec<NumaNode> = Vec::new();
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            // Node directories are named "node0", "node1", ...
            let Some(id_str) = name.strip_prefix("node") else {
                continue;
            };
            let Ok(id) = id_str.parse::<usize>() else {
                continue;
            };

            let node_path = entry.path();
            let cpu_cores = std::fs::read_to_string(node_path.join("cpulist"))
                .ok()
                .map(|s| Self::parse_cpu_list(s.trim()))
                .unwrap_or_default();
            let memory_size = std::fs::read_to_string(node_path.join("meminfo"))
                .ok()
                .and_then(|s| Self::parse_node_mem_total(&s))
                .unwrap_or(0);

            nodes.push(NumaNode {
                id,
                memory_size,
                cpu_cores,
            });
        }

        if nodes.is_empty() {
            return Ok(Self::flat_numa_topology());
        }

        nodes.sort_by_key(|n| n.id);
        let preferred_node = nodes.first().map(|n| n.id).unwrap_or(0);
        Ok(NumaTopology {
            nodes,
            preferred_node,
        })
    }

    /// Build a single-node topology spanning every logical CPU, used when no
    /// real NUMA information is available (flat memory system).
    #[cfg(target_os = "linux")]
    fn flat_numa_topology() -> NumaTopology {
        let cpu_cores: Vec<usize> = (0..num_cpus::get().max(1)).collect();
        let memory_size = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|content| {
                content.lines().find_map(|line| {
                    line.strip_prefix("MemTotal:")
                        .and_then(|rest| rest.split_whitespace().next())
                        .and_then(|kb| kb.parse::<u64>().ok())
                        .map(|kb| kb * 1024)
                })
            })
            .unwrap_or(0);
        NumaTopology {
            nodes: vec![NumaNode {
                id: 0,
                memory_size,
                cpu_cores,
            }],
            preferred_node: 0,
        }
    }

    /// Parse a Linux cpu-list string such as "0-3,8,10-11" into the explicit
    /// list of CPU indices it denotes.
    #[cfg(target_os = "linux")]
    fn parse_cpu_list(list: &str) -> Vec<usize> {
        let mut cpus = Vec::new();
        for token in list.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            if let Some((start, end)) = token.split_once('-') {
                if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) {
                    for cpu in start..=end {
                        cpus.push(cpu);
                    }
                }
            } else if let Ok(cpu) = token.parse::<usize>() {
                cpus.push(cpu);
            }
        }
        cpus
    }

    /// Extract `MemTotal` (in bytes) from a per-node `meminfo` file whose lines
    /// look like "Node 0 MemTotal:  16384000 kB".
    #[cfg(target_os = "linux")]
    fn parse_node_mem_total(meminfo: &str) -> Option<u64> {
        for line in meminfo.lines() {
            if line.contains("MemTotal:") {
                if let Some(kb) = line
                    .split_whitespace()
                    .rev()
                    .nth(1)
                    .and_then(|s| s.parse::<u64>().ok())
                {
                    return Some(kb * 1024);
                }
            }
        }
        None
    }

    /// Get the metrics sampled within the most recent `duration` window.
    ///
    /// Each historical sample is tagged with its capture instant, so this
    /// returns exactly those samples whose timestamp falls on or after
    /// `now - duration`, in chronological order.
    pub fn get_metrics_trend(&self, duration: Duration) -> Vec<&SystemMetrics> {
        let cutoff_time = Instant::now().checked_sub(duration);
        self.metrics_history
            .iter()
            .zip(self.history_timestamps.iter())
            .filter_map(|(metrics, timestamp)| match cutoff_time {
                // Keep samples at or after the cutoff.
                Some(cutoff) if *timestamp >= cutoff => Some(metrics),
                // If the cutoff underflows the clock origin the whole history is
                // within the window.
                None => Some(metrics),
                _ => None,
            })
            .collect()
    }

    /// Check if system is under high load
    pub fn is_high_load(&self) -> bool {
        if let Some(metrics) = &self.cached_metrics {
            metrics.cpu_usage > 0.8
                || metrics.memory_usage.utilization > 0.9
                || metrics.io_utilization > 0.8
        } else {
            false
        }
    }

    /// Get resource utilization score (0.0 to 1.0).
    ///
    /// Averages CPU, memory and I/O utilization from the most recent metrics.
    /// Before any metrics have been collected there is nothing to average, so a
    /// neutral midpoint is returned to signal "unknown".
    pub fn get_utilization_score(&self) -> f64 {
        if let Some(metrics) = &self.cached_metrics {
            (metrics.cpu_usage + metrics.memory_usage.utilization + metrics.io_utilization) / 3.0
        } else {
            0.5
        }
    }

    /// Predict resource pressure in near future
    pub fn predict_resource_pressure(&self, lookahead: Duration) -> f64 {
        // Simple linear extrapolation based on recent trends
        if self.metrics_history.len() < 2 {
            return self.get_utilization_score();
        }

        let recent_scores: Vec<f64> = self
            .metrics_history
            .iter()
            .rev()
            .take(10)
            .map(|m| (m.cpu_usage + m.memory_usage.utilization + m.io_utilization) / 3.0)
            .collect();

        if recent_scores.len() < 2 {
            return recent_scores[0];
        }

        // Calculate trend slope
        let n = recent_scores.len() as f64;
        let sum_x: f64 = (0..recent_scores.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent_scores.iter().sum();
        let sum_xy: f64 = recent_scores
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum();
        let sum_x2: f64 = (0..recent_scores.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Project into the future
        let future_steps = lookahead.as_secs() as f64 / self.update_frequency.as_secs() as f64;
        let predicted = intercept + slope * (n + future_steps);

        predicted.clamp(0.0, 1.0)
    }
}
