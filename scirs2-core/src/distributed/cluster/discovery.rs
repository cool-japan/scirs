//! Node discovery implementations for cluster management
//!
//! This module provides various methods for discovering nodes in the cluster,
//! including static configuration, multicast discovery, DNS service discovery,
//! and Consul-based discovery.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::net::{IpAddr, SocketAddr};
use std::time::{Duration, Instant};

use super::types::{
    NodeCapabilities, NodeDiscoveryMethod, NodeInfo, NodeMetadata, NodeStatus, NodeType,
};

/// Node discovery implementation
pub struct NodeDiscovery;

impl NodeDiscovery {
    /// Discover nodes using a specific discovery method
    pub fn discover_nodes(method: &NodeDiscoveryMethod) -> CoreResult<Vec<NodeInfo>> {
        match method {
            NodeDiscoveryMethod::Static(addresses) => Self::discover_static_nodes(addresses),
            NodeDiscoveryMethod::Multicast { group, port } => {
                Self::discover_multicast_nodes(group, *port)
            }
            NodeDiscoveryMethod::DnsService { service_name } => {
                Self::discover_dns_service_nodes(service_name)
            }
            NodeDiscoveryMethod::Consul { endpoint } => Self::discover_consul_nodes(endpoint),
        }
    }

    /// Discover nodes from static address list
    fn discover_static_nodes(addresses: &[SocketAddr]) -> CoreResult<Vec<NodeInfo>> {
        let mut nodes = Vec::new();
        for address in addresses {
            if Self::is_node_reachable(*address)? {
                nodes.push(NodeInfo {
                    id: format!("node_{address}"),
                    address: *address,
                    node_type: NodeType::Worker,
                    capabilities: NodeCapabilities::default(),
                    status: NodeStatus::Unknown,
                    last_seen: Instant::now(),
                    metadata: NodeMetadata::default(),
                });
            }
        }
        Ok(nodes)
    }

    /// Discover nodes via multicast
    fn discover_multicast_nodes(group: &IpAddr, port: u16) -> CoreResult<Vec<NodeInfo>> {
        use std::net::{SocketAddr, UdpSocket};
        use std::time::Duration;

        let mut discovered_nodes = Vec::new();

        // Create a UDP socket for multicast discovery
        let socket = UdpSocket::bind(SocketAddr::new(*group, port)).map_err(|e| {
            CoreError::IoError(crate::error::ErrorContext::new(format!(
                "Failed to bind multicast socket: {e}"
            )))
        })?;

        // Set socket timeout for non-blocking operation
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .map_err(|e| {
                CoreError::IoError(crate::error::ErrorContext::new(format!(
                    "Failed to set socket timeout: {e}"
                )))
            })?;

        // Send discovery broadcast
        let discovery_message = b"SCIRS2_NODE_DISCOVERY";
        let broadcast_addr = SocketAddr::new(*group, port);

        match socket.send_to(discovery_message, broadcast_addr) {
            Ok(_) => {
                // Listen for responses
                let mut buffer = [0u8; 1024];
                let start_time = std::time::Instant::now();

                while start_time.elapsed() < Duration::from_secs(3) {
                    match socket.recv_from(&mut buffer) {
                        Ok((size, addr)) => {
                            let response = String::from_utf8_lossy(&buffer[..size]);
                            if response.starts_with("SCIRS2_NODE_RESPONSE") {
                                // Parse node information from response
                                let parts: Vec<&str> = response.split(':').collect();
                                if parts.len() >= 3 {
                                    let nodeid = parts[1usize].to_string();
                                    let node_type = match parts[2usize] {
                                        "master" => NodeType::Master,
                                        "worker" => NodeType::Worker,
                                        "storage" => NodeType::Storage,
                                        "compute" => NodeType::Compute,
                                        _ => NodeType::Worker,
                                    };

                                    discovered_nodes.push(NodeInfo {
                                        id: nodeid,
                                        address: addr,
                                        node_type,
                                        capabilities: NodeCapabilities::default(),
                                        status: NodeStatus::Unknown,
                                        last_seen: Instant::now(),
                                        metadata: NodeMetadata::default(),
                                    });
                                }
                            }
                        }
                        Err(_) => break, // Timeout or error, exit loop
                    }
                }
            }
            Err(e) => {
                return Err(CoreError::IoError(crate::error::ErrorContext::new(
                    format!("Failed to send discovery broadcast: {e}"),
                )));
            }
        }

        Ok(discovered_nodes)
    }

    /// Discover nodes via DNS service discovery
    fn discover_dns_service_nodes(service_name: &str) -> CoreResult<Vec<NodeInfo>> {
        // DNS-SD discovery implementation
        // This would typically use DNS SRV records to discover services
        #[allow(unused_mut)]
        let mut discovered_nodes = Vec::new();

        #[cfg(target_os = "linux")]
        {
            use std::process::Command;
            use std::str;
            // Try to use avahi-browse for DNS-SD discovery on Linux
            match Command::new("avahi-browse")
                .arg("-t")  // Terminate after cache is exhausted
                .arg("-r")  // Resolve found services
                .arg("-p")  // Parseable output
                .arg(service_name)
                .output()
            {
                Ok(output) => {
                    let output_str = str::from_utf8(&output.stdout).map_err(|e| {
                        CoreError::ValidationError(ErrorContext::new(format!(
                            "Failed to parse avahi output: {e}"
                        )))
                    })?;

                    // Parse avahi-browse output format
                    for line in output_str.lines() {
                        let parts: Vec<&str> = line.split(';').collect();
                        if parts.len() >= 9 && parts[0usize] == "=" {
                            // Format: =;interface;protocol;name;type;domain;hostname;address;port;txt
                            let hostname = parts[6usize];
                            let address_str = parts[7usize];
                            let port_str = parts[8usize];

                            if let Ok(port) = port_str.parse::<u16>() {
                                // Try to parse IP address
                                if let Ok(ip) = address_str.parse::<IpAddr>() {
                                    let socket_addr = SocketAddr::new(ip, port);
                                    let nodeid = format!("dns_{hostname}_{port}");

                                    discovered_nodes.push(NodeInfo {
                                        id: nodeid,
                                        address: socket_addr,
                                        node_type: NodeType::Worker,
                                        capabilities: NodeCapabilities::default(),
                                        status: NodeStatus::Unknown,
                                        last_seen: Instant::now(),
                                        metadata: NodeMetadata {
                                            hostname: hostname.to_string(),
                                            operating_system: "unknown".to_string(),
                                            kernel_version: "unknown".to_string(),
                                            container_runtime: None,
                                            labels: std::collections::HashMap::new(),
                                        },
                                    });
                                }
                            }
                        }
                    }
                }
                Err(_) => {
                    // avahi-browse not available, try nslookup for basic SRV record resolution
                    match Command::new("nslookup")
                        .arg("-type=SRV")
                        .arg(service_name)
                        .output()
                    {
                        Ok(output) => {
                            let output_str = str::from_utf8(&output.stdout).map_err(|e| {
                                CoreError::ValidationError(ErrorContext::new(format!(
                                    "Failed to parse nslookup output: {e}"
                                )))
                            })?;

                            // Parse SRV records (simplified)
                            for line in output_str.lines() {
                                if line.contains("service =") {
                                    // Extract port and hostname from SRV record
                                    let parts: Vec<&str> = line.split_whitespace().collect();
                                    if parts.len() >= 4 {
                                        if let Ok(port) = parts[2usize].parse::<u16>() {
                                            let hostname = parts[3usize].trim_end_matches('.');
                                            let nodeid = format!("srv_{hostname}_{port}");

                                            // Try to resolve hostname to IP
                                            if let Ok(mut addrs) =
                                                std::net::ToSocketAddrs::to_socket_addrs(&format!(
                                                    "{hostname}:{port}"
                                                ))
                                            {
                                                if let Some(addr) = addrs.next() {
                                                    discovered_nodes.push(NodeInfo {
                                                        id: nodeid,
                                                        address: addr,
                                                        node_type: NodeType::Worker,
                                                        capabilities: NodeCapabilities::default(),
                                                        status: NodeStatus::Unknown,
                                                        last_seen: Instant::now(),
                                                        metadata: NodeMetadata {
                                                            hostname: hostname.to_string(),
                                                            operating_system: "unknown".to_string(),
                                                            kernel_version: "unknown".to_string(),
                                                            container_runtime: None,
                                                            labels: std::collections::HashMap::new(
                                                            ),
                                                        },
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            // Both avahi-browse and nslookup failed, return empty list
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            use std::process::Command;
            use std::str;
            // On Windows, try to use dns-sd command if available
            match Command::new("dns-sd")
                .arg("-B")  // Browse for services
                .arg(service_name)
                .output()
            {
                Ok(output) => {
                    let output_str = str::from_utf8(&output.stdout).map_err(|e| {
                        CoreError::ValidationError(ErrorContext::new(format!(
                            "Failed to parse dns-sd output: {e}"
                        )))
                    })?;

                    // Parse dns-sd output (simplified implementation)
                    for line in output_str.lines() {
                        if line.contains(service_name) {
                            // Extract service information
                            // This is a simplified parser - real implementation would be more robust
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() >= 2 {
                                let service_instance = parts[1usize];
                                let nodeid = format!("dnssd_{service_instance}");

                                // For now, use a default port and localhost
                                // Real implementation would resolve the service
                                let socket_addr = SocketAddr::new(
                                    IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)),
                                    8080,
                                );

                                discovered_nodes.push(NodeInfo {
                                    id: nodeid,
                                    address: socket_addr,
                                    node_type: NodeType::Worker,
                                    capabilities: NodeCapabilities::default(),
                                    status: NodeStatus::Unknown,
                                    last_seen: Instant::now(),
                                    metadata: NodeMetadata::default(),
                                });
                            }
                        }
                    }
                }
                Err(_) => {
                    // dns-sd not available
                }
            }
        }

        Ok(discovered_nodes)
    }

    /// Discover nodes via Consul service registry
    fn discover_consul_nodes(endpoint: &str) -> CoreResult<Vec<NodeInfo>> {
        // Consul discovery implementation via HTTP API
        use std::process::Command;
        use std::str;

        let mut discovered_nodes = Vec::new();

        // Try to query Consul catalog API for services
        let consul_url = if endpoint.starts_with("http") {
            format!("{endpoint}/v1/catalog/services")
        } else {
            format!("http://{endpoint}/v1/catalog/services")
        };

        // Use curl to query Consul API (most portable approach)
        match Command::new("curl")
            .arg("-s")  // Silent mode
            .arg("-f")  // Fail silently on HTTP errors
            .arg("--connect-timeout")
            .arg("5")   // 5 second timeout
            .arg(&consul_url)
            .output()
        {
            Ok(output) => {
                if output.status.success() {
                    let json_str = str::from_utf8(&output.stdout).map_err(|e| {
                        CoreError::ValidationError(ErrorContext::new(format!(
                            "Failed to parse Consul response: {e}"
                        )))
                    })?;

                    // Parse JSON response (simplified - would use serde_json in real implementation)
                    // Looking for service names in the format: {"service_name": ["tag1", "tag2"]}
                    if json_str.trim().starts_with('{') {
                        // Extract service names from JSON
                        let cleaned = json_str.replace(['{', '}'], "");
                        for service_entry in cleaned.split(',') {
                            let service_parts: Vec<&str> = service_entry.split(':').collect();
                            if service_parts.len() >= 2 {
                                let service_name = service_parts[0usize].trim().trim_matches('"');

                                // Query specific service details
                                let service_url = if endpoint.starts_with("http") {
                                    format!("{endpoint}/v1/catalog/service/{service_name}")
                                } else {
                                    format!("http://{endpoint}/v1/catalog/service/{service_name}")
                                };

                                match Command::new("curl")
                                    .arg("-s")
                                    .arg("-f")
                                    .arg("--connect-timeout")
                                    .arg("3")
                                    .arg(&service_url)
                                    .output()
                                {
                                    Ok(service_output) => {
                                        if service_output.status.success() {
                                            let service_json =
                                                str::from_utf8(&service_output.stdout)
                                                    .unwrap_or("");

                                            // Simple JSON parsing to extract Address and ServicePort
                                            // In real implementation, would use proper JSON parsing
                                            if service_json.contains("\"Address\"")
                                                && service_json.contains("\"ServicePort\"")
                                            {
                                                // Extract address and port (very simplified)
                                                let lines: Vec<&str> =
                                                    service_json.lines().collect();
                                                let mut address_str = "";
                                                let mut port_str = "";

                                                for line in lines {
                                                    if line.contains("\"Address\"") {
                                                        if let Some(addr_part) =
                                                            line.split(':').nth(1)
                                                        {
                                                            address_str = addr_part
                                                                .trim()
                                                                .trim_matches('"')
                                                                .trim_matches(',');
                                                        }
                                                    }
                                                    if line.contains("\"ServicePort\"") {
                                                        if let Some(port_part) =
                                                            line.split(':').nth(1)
                                                        {
                                                            port_str =
                                                                port_part.trim().trim_matches(',');
                                                        }
                                                    }
                                                }

                                                // Create node info if we have both address and port
                                                if !address_str.is_empty() && !port_str.is_empty() {
                                                    if let (Ok(ip), Ok(port)) = (
                                                        address_str.parse::<IpAddr>(),
                                                        port_str.parse::<u16>(),
                                                    ) {
                                                        let socket_addr = SocketAddr::new(ip, port);
                                                        let nodeid = format!(
                                                            "consul_{service_name}_{address_str}"
                                                        );

                                                        discovered_nodes.push(NodeInfo {
                                                            id: nodeid,
                                                            address: socket_addr,
                                                            node_type: NodeType::Worker,
                                                            capabilities: NodeCapabilities::default(),
                                                            status: NodeStatus::Unknown,
                                                            last_seen: Instant::now(),
                                                            metadata: NodeMetadata {
                                                                hostname: address_str.to_string(),
                                                                operating_system: "unknown".to_string(),
                                                                kernel_version: "unknown".to_string(),
                                                                container_runtime: Some("consul".to_string()),
                                                                labels: {
                                                                    let mut labels = std::collections::HashMap::new();
                                                                    labels.insert("service".to_string(), service_name.to_string());
                                                                    labels.insert("discovery".to_string(), "consul".to_string());
                                                                    labels
                                                                },
                                                            },
                                                        });
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Err(_) => continue, // Skip this service if query fails
                                }
                            }
                        }
                    }
                } else {
                    return Err(CoreError::IoError(ErrorContext::new(format!(
                        "Failed to connect to Consul at {endpoint}"
                    ))));
                }
            }
            Err(_) => {
                return Err(CoreError::InvalidState(ErrorContext::new(
                    "curl command not available for Consul discovery",
                )));
            }
        }

        Ok(discovered_nodes)
    }

    /// Check if a node is reachable
    fn is_node_reachable(address: SocketAddr) -> CoreResult<bool> {
        // Simple reachability check
        // In a real implementation, this would do proper health checking
        Ok(true) // Placeholder
    }
}
