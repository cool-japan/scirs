//! Health monitoring system for cluster nodes
//!
//! This module provides comprehensive health monitoring capabilities
//! for cluster nodes, including various health checks and scoring.
//!
//! # Design notes: what is actually checkable
//!
//! This crate has no remote metrics-collection agent/RPC: a
//! [`HealthMonitor`] runs in one process (typically the cluster
//! coordinator) and is handed a [`NodeInfo`] describing some other machine.
//! That means only two classes of check are honestly implementable here:
//!
//! - Checks that only require *network* access to `node.address` (`Ping`,
//!   `NetworkConnectivity`) are always performed for real, against any node.
//! - Checks that require reading the *target* machine's own resource
//!   counters (`CpuLoad`, `MemoryUsage`, `DiskSpace`) can only be answered
//!   for real when `node.address` refers to the local host (loopback):
//!   there is no transport to ask a remote node for its CPU/memory/disk
//!   state. For a non-loopback node these checks honestly report
//!   [`HealthCheckStatus::Unknown`] rather than fabricating a verdict from
//!   this process's own local readings (which would silently mislabel the
//!   coordinator's stats as belonging to the remote node).
//!
//! `Unknown` never counts as a failure (it does not reduce the health
//! score) and never counts as a pass either: [`NodeStatus::Unknown`] is
//! reported distinctly from [`NodeStatus::Healthy`] whenever any check
//! could not be evaluated with confidence.

use crate::error::CoreResult;
use std::io;
use std::net::{SocketAddr, TcpStream};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::types::{
    HealthCheck, HealthCheckResult, HealthCheckStatus, NodeHealthStatus, NodeInfo, NodeStatus,
};

/// Default timeout applied to network-based probes (`Ping` /
/// `NetworkConnectivity`).
const DEFAULT_PROBE_TIMEOUT: Duration = Duration::from_millis(750);

/// Above this 1-minute load average per logical core, a node is considered
/// unhealthy (sustained overload).
const CPU_LOAD_PER_CORE_UNHEALTHY: f64 = 2.0;

/// Above this percentage of memory in use, a node is considered unhealthy.
const MEMORY_USAGE_UNHEALTHY_PERCENT: f64 = 90.0;

/// Below this percentage of free disk space (only measurable with the
/// `sysinfo` feature), a node is considered unhealthy even if the
/// functional write probe succeeded.
#[cfg(feature = "sysinfo")]
const DISK_FREE_UNHEALTHY_PERCENT: f64 = 5.0;

/// Health monitoring system
#[derive(Debug)]
pub struct HealthMonitor {
    health_checks: Vec<HealthCheck>,
    check_interval: Duration,
    /// Timeout applied to TCP-based probes (`Ping` / `NetworkConnectivity`).
    probe_timeout: Duration,
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self {
            health_checks: Self::default_health_checks(),
            check_interval: Duration::from_secs(30),
            probe_timeout: DEFAULT_PROBE_TIMEOUT,
        }
    }
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new() -> CoreResult<Self> {
        Ok(Self::default())
    }

    /// Get the default set of health checks
    fn default_health_checks() -> Vec<HealthCheck> {
        vec![
            HealthCheck::Ping,
            HealthCheck::CpuLoad,
            HealthCheck::MemoryUsage,
            HealthCheck::DiskSpace,
            HealthCheck::NetworkConnectivity,
        ]
    }

    /// Check the health of a specific node
    pub fn check_node_health(&mut self, node: &NodeInfo) -> CoreResult<NodeHealthStatus> {
        let mut health_score = 100.0f64;
        let mut failing_checks = Vec::new();
        let mut unknown_checks = Vec::new();

        for check in &self.health_checks {
            match self.execute_health_check(check, node) {
                Ok(result) => match result.status {
                    HealthCheckStatus::Healthy => {}
                    HealthCheckStatus::Unhealthy => {
                        health_score -= result.impact_score;
                        failing_checks.push(check.clone());
                    }
                    HealthCheckStatus::Unknown => {
                        unknown_checks.push(check.clone());
                    }
                },
                Err(_) => {
                    health_score -= 20.0f64; // Penalty for a check that errored unexpectedly
                    failing_checks.push(check.clone());
                }
            }
        }

        let status = Self::classify_status(health_score, &unknown_checks, 80.0, 50.0);

        Ok(NodeHealthStatus {
            status,
            health_score,
            failing_checks,
            unknown_checks,
            last_checked: Instant::now(),
        })
    }

    /// Turn a health score plus the set of inconclusive checks into an
    /// overall [`NodeStatus`], honoring the given healthy/degraded score
    /// thresholds.
    ///
    /// A node only earns [`NodeStatus::Healthy`] when its score clears
    /// `healthy_threshold` *and* every check was actually evaluated (no
    /// `Unknown`s); otherwise it is reported as [`NodeStatus::Unknown`] so
    /// callers cannot mistake "nothing confirmed broken" for "confirmed
    /// fine".
    fn classify_status(
        health_score: f64,
        unknown_checks: &[HealthCheck],
        healthy_threshold: f64,
        degraded_threshold: f64,
    ) -> NodeStatus {
        if health_score >= healthy_threshold {
            if unknown_checks.is_empty() {
                NodeStatus::Healthy
            } else {
                NodeStatus::Unknown
            }
        } else if health_score >= degraded_threshold {
            NodeStatus::Degraded
        } else {
            NodeStatus::Unhealthy
        }
    }

    /// Execute a specific health check on a node
    fn execute_health_check(
        &self,
        check: &HealthCheck,
        node: &NodeInfo,
    ) -> CoreResult<HealthCheckResult> {
        match check {
            HealthCheck::Ping => Ok(Self::check_ping(node.address, self.probe_timeout)),
            HealthCheck::NetworkConnectivity => Ok(Self::check_network_connectivity(
                node.address,
                self.probe_timeout,
            )),
            HealthCheck::CpuLoad => Ok(Self::check_cpu_load(node)),
            HealthCheck::MemoryUsage => Ok(Self::check_memory_usage(node)),
            HealthCheck::DiskSpace => Ok(Self::check_disk_space(node)),
        }
    }

    /// Real TCP-connect liveness probe: is `address` accepting connections?
    fn check_ping(address: SocketAddr, timeout: Duration) -> HealthCheckResult {
        let start = Instant::now();
        match TcpStream::connect_timeout(&address, timeout) {
            Ok(_stream) => HealthCheckResult {
                status: HealthCheckStatus::Healthy,
                impact_score: 10.0,
                details: format!(
                    "TCP connect to {address} succeeded in {:?}",
                    start.elapsed()
                ),
            },
            Err(e) => HealthCheckResult {
                status: HealthCheckStatus::Unhealthy,
                impact_score: 10.0,
                details: format!("TCP connect to {address} failed: {e}"),
            },
        }
    }

    /// Real network-quality probe: like `Ping`, but also requires the TCP
    /// handshake to complete comfortably inside the configured timeout
    /// (a connection that barely makes it under the wire indicates a
    /// degraded network path, not a healthy one).
    fn check_network_connectivity(address: SocketAddr, timeout: Duration) -> HealthCheckResult {
        let acceptable = timeout / 2;
        let start = Instant::now();
        match TcpStream::connect_timeout(&address, timeout) {
            Ok(_stream) => {
                let elapsed = start.elapsed();
                let healthy = elapsed <= acceptable;
                HealthCheckResult {
                    status: if healthy {
                        HealthCheckStatus::Healthy
                    } else {
                        HealthCheckStatus::Unhealthy
                    },
                    impact_score: 15.0,
                    details: format!(
                        "Network round-trip to {address} took {elapsed:?} (acceptable <= {acceptable:?})"
                    ),
                }
            }
            Err(e) => HealthCheckResult {
                status: HealthCheckStatus::Unhealthy,
                impact_score: 15.0,
                details: format!("Network connectivity check to {address} failed: {e}"),
            },
        }
    }

    /// CPU load check. Only answerable for the local host (see module docs).
    fn check_cpu_load(node: &NodeInfo) -> HealthCheckResult {
        if !Self::node_is_local(node) {
            return HealthCheckResult {
                status: HealthCheckStatus::Unknown,
                impact_score: 15.0,
                details: format!(
                    "CPU load of remote node {} ({}) cannot be measured: no metrics-collection \
                     transport is implemented for remote nodes",
                    node.id, node.address
                ),
            };
        }

        match Self::read_local_load_average() {
            Some(load_one) => {
                let cores = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1);
                let per_core = load_one / cores as f64;
                let healthy = per_core < CPU_LOAD_PER_CORE_UNHEALTHY;
                HealthCheckResult {
                    status: if healthy {
                        HealthCheckStatus::Healthy
                    } else {
                        HealthCheckStatus::Unhealthy
                    },
                    impact_score: 15.0,
                    details: format!(
                        "1-minute load average {load_one:.2} across {cores} logical core(s) \
                         ({per_core:.2}/core, unhealthy at >= {CPU_LOAD_PER_CORE_UNHEALTHY:.2}/core)"
                    ),
                }
            }
            None => HealthCheckResult {
                status: HealthCheckStatus::Unknown,
                impact_score: 15.0,
                details: "CPU load average is unavailable on this platform build (Linux reads \
                          /proc/loadavg natively; other platforms require the `sysinfo` feature)"
                    .to_string(),
            },
        }
    }

    /// Memory usage check. Only answerable for the local host (see module
    /// docs).
    fn check_memory_usage(node: &NodeInfo) -> HealthCheckResult {
        if !Self::node_is_local(node) {
            return HealthCheckResult {
                status: HealthCheckStatus::Unknown,
                impact_score: 20.0,
                details: format!(
                    "Memory usage of remote node {} ({}) cannot be measured: no \
                     metrics-collection transport is implemented for remote nodes",
                    node.id, node.address
                ),
            };
        }

        match Self::read_local_memory_kb() {
            Some((total_kb, available_kb)) if total_kb > 0 => {
                let used_percent =
                    100.0 * (total_kb.saturating_sub(available_kb)) as f64 / total_kb as f64;
                let healthy = used_percent < MEMORY_USAGE_UNHEALTHY_PERCENT;
                HealthCheckResult {
                    status: if healthy {
                        HealthCheckStatus::Healthy
                    } else {
                        HealthCheckStatus::Unhealthy
                    },
                    impact_score: 20.0,
                    details: format!(
                        "Memory in use: {used_percent:.1}% of {:.2} GiB (unhealthy at >= \
                         {MEMORY_USAGE_UNHEALTHY_PERCENT:.1}%)",
                        total_kb as f64 / (1024.0 * 1024.0)
                    ),
                }
            }
            _ => HealthCheckResult {
                status: HealthCheckStatus::Unknown,
                impact_score: 20.0,
                details: "Memory usage is unavailable on this platform build (Linux reads \
                          /proc/meminfo natively; other platforms require the `sysinfo` feature)"
                    .to_string(),
            },
        }
    }

    /// Disk space check: a genuine local write/read/delete probe (works
    /// everywhere, no platform-specific code required), optionally
    /// enriched with a real free-space percentage when the `sysinfo`
    /// feature is enabled. Only answerable for the local host (see module
    /// docs).
    fn check_disk_space(node: &NodeInfo) -> HealthCheckResult {
        if !Self::node_is_local(node) {
            return HealthCheckResult {
                status: HealthCheckStatus::Unknown,
                impact_score: 10.0,
                details: format!(
                    "Disk space of remote node {} ({}) cannot be measured: no \
                     metrics-collection transport is implemented for remote nodes",
                    node.id, node.address
                ),
            };
        }

        let dir = std::env::temp_dir();
        match Self::probe_local_disk_write(&dir) {
            Ok(bytes_written) => {
                #[cfg(feature = "sysinfo")]
                {
                    if let Some(free_percent) = Self::read_local_disk_free_percent(&dir) {
                        let healthy = free_percent >= DISK_FREE_UNHEALTHY_PERCENT;
                        return HealthCheckResult {
                            status: if healthy {
                                HealthCheckStatus::Healthy
                            } else {
                                HealthCheckStatus::Unhealthy
                            },
                            impact_score: 10.0,
                            details: format!(
                                "Write/read probe of {bytes_written} bytes to {} succeeded; \
                                 {free_percent:.1}% free space remaining (unhealthy at < \
                                 {DISK_FREE_UNHEALTHY_PERCENT:.1}%)",
                                dir.display()
                            ),
                        };
                    }
                }

                HealthCheckResult {
                    status: HealthCheckStatus::Healthy,
                    impact_score: 10.0,
                    details: format!(
                        "Write/read/delete probe of {bytes_written} bytes to {} succeeded",
                        dir.display()
                    ),
                }
            }
            Err(e) => HealthCheckResult {
                status: HealthCheckStatus::Unhealthy,
                impact_score: 10.0,
                details: format!(
                    "Disk write/read probe against {} failed: {e}",
                    dir.display()
                ),
            },
        }
    }

    /// Whether `node` refers to the machine this `HealthMonitor` runs on.
    ///
    /// We can only be certain of this for loopback addresses: pure-`std`
    /// code has no portable way to enumerate this host's non-loopback
    /// interface addresses, so a non-loopback address is conservatively
    /// treated as "possibly remote" rather than guessed at.
    fn node_is_local(node: &NodeInfo) -> bool {
        node.address.ip().is_loopback()
    }

    /// Perform a real write + read-back + delete against `dir`, returning
    /// the number of bytes round-tripped on success.
    fn probe_local_disk_write(dir: &std::path::Path) -> io::Result<usize> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let marker = dir.join(format!(
            "scirs2_health_probe_{}_{nanos}.tmp",
            std::process::id()
        ));
        let payload = b"scirs2-core cluster health disk probe";

        std::fs::write(&marker, payload)?;
        let readback = std::fs::read(&marker);
        // Always attempt cleanup, even if the read-back failed.
        let _ = std::fs::remove_file(&marker);
        let readback = readback?;

        if readback != payload {
            return Err(io::Error::other(format!(
                "read-back mismatch: wrote {} bytes, read back {} bytes",
                payload.len(),
                readback.len()
            )));
        }

        Ok(payload.len())
    }

    /// Read the 1-minute load average for the local host, if possible.
    #[cfg(target_os = "linux")]
    fn read_local_load_average() -> Option<f64> {
        let content = std::fs::read_to_string("/proc/loadavg").ok()?;
        content.split_whitespace().next()?.parse::<f64>().ok()
    }

    #[cfg(all(not(target_os = "linux"), feature = "sysinfo"))]
    fn read_local_load_average() -> Option<f64> {
        Some(sysinfo::System::load_average().one)
    }

    #[cfg(all(not(target_os = "linux"), not(feature = "sysinfo")))]
    fn read_local_load_average() -> Option<f64> {
        None
    }

    /// Read `(total_kb, available_kb)` for the local host, if possible.
    #[cfg(target_os = "linux")]
    fn read_local_memory_kb() -> Option<(u64, u64)> {
        let content = std::fs::read_to_string("/proc/meminfo").ok()?;
        let field = |key: &str| -> Option<u64> {
            content.lines().find_map(|line| {
                line.strip_prefix(key)
                    .and_then(|rest| rest.split_whitespace().next())
                    .and_then(|value| value.parse::<u64>().ok())
            })
        };
        let total = field("MemTotal:")?;
        let available = field("MemAvailable:")?;
        Some((total, available))
    }

    #[cfg(all(not(target_os = "linux"), feature = "sysinfo"))]
    fn read_local_memory_kb() -> Option<(u64, u64)> {
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        let total = system.total_memory() / 1024;
        let available = system.available_memory() / 1024;
        Some((total, available))
    }

    #[cfg(all(not(target_os = "linux"), not(feature = "sysinfo")))]
    fn read_local_memory_kb() -> Option<(u64, u64)> {
        None
    }

    /// Real free-space percentage for the disk backing `path`, using the
    /// `sysinfo` crate's disk listing (matches the mount point with the
    /// longest path prefix, i.e. the most specific mount for `path`).
    #[cfg(feature = "sysinfo")]
    fn read_local_disk_free_percent(path: &std::path::Path) -> Option<f64> {
        let disks = sysinfo::Disks::new_with_refreshed_list();
        let mut best: Option<(&std::path::Path, u64, u64)> = None;

        for disk in disks.list() {
            let mount = disk.mount_point();
            if path.starts_with(mount) {
                let is_better_match = best.is_none_or(|(current, _, _)| {
                    mount.as_os_str().len() > current.as_os_str().len()
                });
                if is_better_match {
                    best = Some((mount, disk.total_space(), disk.available_space()));
                }
            }
        }

        best.and_then(|(_, total, available)| {
            if total == 0 {
                None
            } else {
                Some(100.0 * available as f64 / total as f64)
            }
        })
    }

    /// Add a custom health check
    pub fn add_health_check(&mut self, check: HealthCheck) {
        if !self.health_checks.contains(&check) {
            self.health_checks.push(check);
        }
    }

    /// Remove a health check
    pub fn remove_health_check(&mut self, check: &HealthCheck) {
        self.health_checks.retain(|c| c != check);
    }

    /// Get the list of configured health checks
    pub fn get_health_checks(&self) -> &[HealthCheck] {
        &self.health_checks
    }

    /// Set the check interval
    pub fn set_check_interval(&mut self, interval: Duration) {
        self.check_interval = interval;
    }

    /// Get the check interval
    pub fn get_check_interval(&self) -> Duration {
        self.check_interval
    }

    /// Set the timeout applied to TCP-based probes (`Ping` /
    /// `NetworkConnectivity`).
    pub fn set_probe_timeout(&mut self, timeout: Duration) {
        self.probe_timeout = timeout;
    }

    /// Get the timeout applied to TCP-based probes.
    pub fn get_probe_timeout(&self) -> Duration {
        self.probe_timeout
    }

    /// Perform a quick health check (subset of full checks)
    pub fn quick_health_check(&mut self, node: &NodeInfo) -> CoreResult<NodeHealthStatus> {
        let quick_checks = vec![HealthCheck::Ping, HealthCheck::NetworkConnectivity];
        let mut health_score = 100.0f64;
        let mut failing_checks = Vec::new();
        let mut unknown_checks = Vec::new();

        for check in &quick_checks {
            match self.execute_health_check(check, node) {
                Ok(result) => match result.status {
                    HealthCheckStatus::Healthy => {}
                    HealthCheckStatus::Unhealthy => {
                        health_score -= result.impact_score;
                        failing_checks.push(check.clone());
                    }
                    HealthCheckStatus::Unknown => {
                        unknown_checks.push(check.clone());
                    }
                },
                Err(_) => {
                    health_score -= 30.0f64; // Higher penalty for quick checks
                    failing_checks.push(check.clone());
                }
            }
        }

        // `quick_health_check` only ever ran two checks in the original
        // design and never had a "Degraded" tier, so (unlike
        // `check_node_health`'s weighted score) it must not tolerate a
        // partial-credit reading: with only Ping (10) + NetworkConnectivity
        // (15) contributing, a node failing *both* would still score 75,
        // clearing a naive ">= 70" cutoff and reporting Healthy for a
        // provably dead node. Require every quick check to have actually
        // passed instead.
        let status = if !failing_checks.is_empty() {
            NodeStatus::Unhealthy
        } else if !unknown_checks.is_empty() {
            NodeStatus::Unknown
        } else {
            NodeStatus::Healthy
        };

        Ok(NodeHealthStatus {
            status,
            health_score,
            failing_checks,
            unknown_checks,
            last_checked: Instant::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, TcpListener};

    fn make_node(address: SocketAddr) -> NodeInfo {
        NodeInfo {
            id: format!("node_{address}"),
            address,
            node_type: super::super::types::NodeType::Worker,
            capabilities: super::super::types::NodeCapabilities::default(),
            status: NodeStatus::Unknown,
            last_seen: Instant::now(),
            metadata: super::super::types::NodeMetadata::default(),
        }
    }

    /// Bind an ephemeral loopback port, then immediately drop the listener.
    /// The returned address is guaranteed to have nothing listening on it,
    /// so a connection attempt reliably fails fast instead of timing out.
    fn dead_loopback_address() -> SocketAddr {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local_addr");
        drop(listener);
        addr
    }

    #[test]
    fn ping_against_dead_endpoint_is_not_healthy() {
        let addr = dead_loopback_address();
        let result = HealthMonitor::check_ping(addr, Duration::from_millis(200));
        assert_eq!(result.status, HealthCheckStatus::Unhealthy);
        assert!(!result.status.is_healthy());
    }

    #[test]
    fn network_connectivity_against_dead_endpoint_is_not_healthy() {
        let addr = dead_loopback_address();
        let result = HealthMonitor::check_network_connectivity(addr, Duration::from_millis(200));
        assert_eq!(result.status, HealthCheckStatus::Unhealthy);
    }

    #[test]
    fn ping_against_live_listener_is_healthy() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local_addr");
        // Accept in the background so the connecting side completes its
        // handshake instead of blocking on a full backlog.
        let handle = std::thread::spawn(move || {
            let _ = listener.accept();
        });
        let result = HealthMonitor::check_ping(addr, Duration::from_millis(500));
        assert_eq!(result.status, HealthCheckStatus::Healthy);
        assert!(result.status.is_healthy());
        handle.join().expect("acceptor thread panicked");
    }

    #[test]
    fn full_node_health_check_never_fabricates_healthy_for_dead_node() {
        let addr = dead_loopback_address();
        let mut monitor = HealthMonitor::new().expect("construct monitor");
        let node = make_node(addr);

        let health = monitor.check_node_health(&node).expect("check health");
        // Ping and NetworkConnectivity both fail for real against a dead
        // port; the aggregate status must reflect that, never Healthy.
        assert_ne!(health.status, NodeStatus::Healthy);
        assert!(health.failing_checks.contains(&HealthCheck::Ping));
        assert!(health
            .failing_checks
            .contains(&HealthCheck::NetworkConnectivity));
        assert!(health.health_score < 100.0);
    }

    #[test]
    fn quick_health_check_never_fabricates_healthy_for_dead_node() {
        let addr = dead_loopback_address();
        let mut monitor = HealthMonitor::new().expect("construct monitor");
        let node = make_node(addr);

        let health = monitor.quick_health_check(&node).expect("quick check");
        assert_eq!(health.status, NodeStatus::Unhealthy);
    }

    #[test]
    fn remote_resource_checks_report_unknown_not_healthy() {
        // A non-loopback address: this process has no transport to query
        // that machine's CPU/memory/disk, so these must come back Unknown
        // rather than a fabricated Healthy (the exact bug being fixed).
        let remote = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 42, 42, 42)), 9); // discard port
        let node = make_node(remote);

        for result in [
            HealthMonitor::check_cpu_load(&node),
            HealthMonitor::check_memory_usage(&node),
            HealthMonitor::check_disk_space(&node),
        ] {
            assert_eq!(result.status, HealthCheckStatus::Unknown);
            assert!(!result.status.is_healthy());
        }
    }

    #[test]
    fn local_resource_checks_perform_a_real_measurement() {
        let local = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let node = make_node(local);

        let cpu = HealthMonitor::check_cpu_load(&node);
        let mem = HealthMonitor::check_memory_usage(&node);
        let disk = HealthMonitor::check_disk_space(&node);

        // The disk probe is real std::fs I/O on every platform: it must
        // never be Unknown for the local host.
        assert_ne!(disk.status, HealthCheckStatus::Unknown);
        assert!(disk.details.contains("bytes"));

        // CPU/memory readings require Linux's /proc or the `sysinfo`
        // feature; under the verification profile (--all-features) one of
        // those is always available, so assert a real (non-Unknown)
        // reading whenever we know one is obtainable.
        if cfg!(any(target_os = "linux", feature = "sysinfo")) {
            assert_ne!(cpu.status, HealthCheckStatus::Unknown);
            assert_ne!(mem.status, HealthCheckStatus::Unknown);
            assert!(cpu.details.contains("load average"));
            assert!(mem.details.contains('%'));
        }
    }

    #[test]
    fn classify_status_distinguishes_unknown_from_healthy() {
        // Perfect score but an inconclusive check present: must not claim
        // Healthy.
        let with_unknown =
            HealthMonitor::classify_status(100.0, &[HealthCheck::CpuLoad], 80.0, 50.0);
        assert_eq!(with_unknown, NodeStatus::Unknown);
        assert_ne!(with_unknown, NodeStatus::Healthy);

        // Perfect score, nothing unknown: genuinely healthy.
        let fully_confirmed = HealthMonitor::classify_status(100.0, &[], 80.0, 50.0);
        assert_eq!(fully_confirmed, NodeStatus::Healthy);

        // Low score always wins regardless of unknown checks.
        let unhealthy = HealthMonitor::classify_status(10.0, &[HealthCheck::CpuLoad], 80.0, 50.0);
        assert_eq!(unhealthy, NodeStatus::Unhealthy);
    }

    #[test]
    fn probe_timeout_is_configurable() {
        let mut monitor = HealthMonitor::new().expect("construct monitor");
        let custom = Duration::from_millis(42);
        monitor.set_probe_timeout(custom);
        assert_eq!(monitor.get_probe_timeout(), custom);
    }
}
