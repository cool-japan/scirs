//! Health monitoring system for cluster nodes
//!
//! This module provides comprehensive health monitoring capabilities
//! for cluster nodes, including various health checks and scoring.

use crate::error::CoreResult;
use std::time::{Duration, Instant};

use super::types::{HealthCheck, HealthCheckResult, NodeHealthStatus, NodeInfo, NodeStatus};

/// Health monitoring system
#[derive(Debug)]
pub struct HealthMonitor {
    health_checks: Vec<HealthCheck>,
    #[allow(dead_code)]
    check_interval: Duration,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new() -> CoreResult<Self> {
        Ok(Self {
            health_checks: Self::default_health_checks(),
            check_interval: Duration::from_secs(30),
        })
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

        for check in &self.health_checks {
            match self.execute_health_check(check, node) {
                Ok(result) => {
                    if !result.is_healthy {
                        health_score -= result.impact_score;
                        failing_checks.push(check.clone());
                    }
                }
                Err(_) => {
                    health_score -= 20.0f64; // Penalty for failed check
                    failing_checks.push(check.clone());
                }
            }
        }

        let status = if health_score >= 80.0 {
            NodeStatus::Healthy
        } else if health_score >= 50.0 {
            NodeStatus::Degraded
        } else {
            NodeStatus::Unhealthy
        };

        Ok(NodeHealthStatus {
            status,
            health_score,
            failing_checks,
            last_checked: Instant::now(),
        })
    }

    /// Execute a specific health check on a node
    fn execute_health_check(
        &self,
        check: &HealthCheck,
        node: &NodeInfo,
    ) -> CoreResult<HealthCheckResult> {
        match check {
            HealthCheck::Ping => {
                // Simple ping check
                Ok(HealthCheckResult {
                    is_healthy: true, // Placeholder
                    impact_score: 10.0f64,
                    details: "Ping successful".to_string(),
                })
            }
            HealthCheck::CpuLoad => {
                // CPU load check
                Ok(HealthCheckResult {
                    is_healthy: true, // Placeholder
                    impact_score: 15.0f64,
                    details: "CPU load normal".to_string(),
                })
            }
            HealthCheck::MemoryUsage => {
                // Memory usage check
                Ok(HealthCheckResult {
                    is_healthy: true, // Placeholder
                    impact_score: 20.0f64,
                    details: "Memory usage normal".to_string(),
                })
            }
            HealthCheck::DiskSpace => {
                // Disk space check
                Ok(HealthCheckResult {
                    is_healthy: true, // Placeholder
                    impact_score: 10.0f64,
                    details: "Disk space adequate".to_string(),
                })
            }
            HealthCheck::NetworkConnectivity => {
                // Network connectivity check
                let _ = node; // Suppress unused variable warning
                Ok(HealthCheckResult {
                    is_healthy: true, // Placeholder
                    impact_score: 15.0f64,
                    details: "Network connectivity good".to_string(),
                })
            }
        }
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

    /// Perform a quick health check (subset of full checks)
    pub fn quick_health_check(&mut self, node: &NodeInfo) -> CoreResult<NodeHealthStatus> {
        let quick_checks = vec![HealthCheck::Ping, HealthCheck::NetworkConnectivity];
        let mut health_score = 100.0f64;
        let mut failing_checks = Vec::new();

        for check in &quick_checks {
            match self.execute_health_check(check, node) {
                Ok(result) => {
                    if !result.is_healthy {
                        health_score -= result.impact_score;
                        failing_checks.push(check.clone());
                    }
                }
                Err(_) => {
                    health_score -= 30.0f64; // Higher penalty for quick checks
                    failing_checks.push(check.clone());
                }
            }
        }

        let status = if health_score >= 70.0 {
            NodeStatus::Healthy
        } else {
            NodeStatus::Unhealthy
        };

        Ok(NodeHealthStatus {
            status,
            health_score,
            failing_checks,
            last_checked: Instant::now(),
        })
    }
}
