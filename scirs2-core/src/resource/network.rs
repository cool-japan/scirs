//! # Network Detection and Capabilities
//!
//! This module provides network interface detection for I/O optimization.

use crate::error::CoreResult;

/// Network interface information
#[derive(Debug, Clone)]
pub struct NetworkInfo {
    /// Network interfaces
    pub interfaces: Vec<NetworkInterface>,
    /// Maximum transmission unit
    pub mtu: usize,
    /// Network bandwidth estimate (Mbps)
    pub bandwidth_mbps: f64,
    /// Network latency estimate (milliseconds)
    pub latency_ms: f64,
}

impl Default for NetworkInfo {
    fn default() -> Self {
        Self {
            interfaces: vec![NetworkInterface::default()],
            mtu: 1500,
            bandwidth_mbps: 1000.0, // 1 Gbps default
            latency_ms: 1.0,
        }
    }
}

impl NetworkInfo {
    /// Detect network information
    pub fn detect() -> CoreResult<Self> {
        // Simplified implementation
        Ok(Self::default())
    }
}

/// Network interface information
#[derive(Debug, Clone)]
pub struct NetworkInterface {
    /// Interface name
    pub name: String,
    /// Interface type
    pub interface_type: NetworkInterfaceType,
    /// MAC address
    pub mac_address: String,
    /// IP addresses
    pub ip_addresses: Vec<String>,
    /// Interface status
    pub is_up: bool,
}

impl Default for NetworkInterface {
    fn default() -> Self {
        Self {
            name: "eth0".to_string(),
            interface_type: NetworkInterfaceType::Ethernet,
            mac_address: "00:00:00:00:00:00".to_string(),
            ip_addresses: vec!["127.0.0.1".to_string()],
            is_up: true,
        }
    }
}

/// Network interface types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkInterfaceType {
    /// Ethernet
    Ethernet,
    /// WiFi
    WiFi,
    /// Loopback
    Loopback,
    /// Infiniband
    Infiniband,
    /// Unknown
    Unknown,
}
