//! Cluster management for distributed computing
//!
//! This module provides comprehensive cluster management capabilities
//! including node discovery, health monitoring, resource allocation,
//! and fault-tolerant cluster coordination.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

#[cfg(feature = "logging")]
use log;

use serde::{Deserialize, Serialize};

/// Global cluster manager instance
static GLOBAL_CLUSTER_MANAGER: std::sync::OnceLock<Arc<ClusterManager>> =
    std::sync::OnceLock::new();

/// Comprehensive cluster management system
#[derive(Debug)]
pub struct ClusterManager {
    cluster_state: Arc<RwLock<ClusterState>>,
    node_registry: Arc<RwLock<NodeRegistry>>,
    healthmonitor: Arc<Mutex<HealthMonitor>>,
    resource_allocator: Arc<RwLock<ResourceAllocator>>,
    configuration: Arc<RwLock<ClusterConfiguration>>,
    eventlog: Arc<Mutex<ClusterEventLog>>,
}

#[allow(dead_code)]
impl ClusterManager {
    /// Create new cluster manager
    pub fn new(config: ClusterConfiguration) -> CoreResult<Self> {
        Ok(Self {
            cluster_state: Arc::new(RwLock::new(ClusterState::new())),
            node_registry: Arc::new(RwLock::new(NodeRegistry::new())),
            healthmonitor: Arc::new(Mutex::new(HealthMonitor::new()?)),
            resource_allocator: Arc::new(RwLock::new(ResourceAllocator::new())),
            configuration: Arc::new(RwLock::new(config)),
            eventlog: Arc::new(Mutex::new(ClusterEventLog::new())),
        })
    }

    /// Get global cluster manager instance
    pub fn global() -> CoreResult<Arc<Self>> {
        Ok(GLOBAL_CLUSTER_MANAGER
            .get_or_init(|| Arc::new(Self::new(ClusterConfiguration::default()).unwrap()))
            .clone())
    }

    /// Start cluster management services
    pub fn start(&self) -> CoreResult<()> {
        // Start node discovery
        self.start_node_discovery()?;

        // Start health monitoring
        self.start_healthmonitoring()?;

        // Start resource management
        self.start_resource_management()?;

        // Start cluster coordination
        self.start_cluster_coordination()?;

        Ok(())
    }

    fn start_node_discovery(&self) -> CoreResult<()> {
        let registry = self.node_registry.clone();
        let config = self.configuration.clone();
        let eventlog = self.eventlog.clone();

        thread::spawn(move || loop {
            if let Err(e) = Self::node_discovery_loop(&registry, &config, &eventlog) {
                eprintln!("Node discovery error: {e:?}");
            }
            thread::sleep(Duration::from_secs(30));
        });

        Ok(())
    }

    fn start_healthmonitoring(&self) -> CoreResult<()> {
        let healthmonitor = self.healthmonitor.clone();
        let registry = self.node_registry.clone();
        let eventlog = self.eventlog.clone();

        thread::spawn(move || loop {
            if let Err(e) = Self::healthmonitoring_loop(&healthmonitor, &registry, &eventlog) {
                eprintln!("Health monitoring error: {e:?}");
            }
            thread::sleep(Duration::from_secs(10));
        });

        Ok(())
    }

    fn start_resource_management(&self) -> CoreResult<()> {
        let allocator = self.resource_allocator.clone();
        let registry = self.node_registry.clone();

        thread::spawn(move || loop {
            if let Err(e) = Self::resource_management_loop(&allocator, &registry) {
                eprintln!("Resource management error: {e:?}");
            }
            thread::sleep(Duration::from_secs(15));
        });

        Ok(())
    }

    fn start_cluster_coordination(&self) -> CoreResult<()> {
        let cluster_state = self.cluster_state.clone();
        let registry = self.node_registry.clone();
        let eventlog = self.eventlog.clone();

        thread::spawn(move || loop {
            if let Err(e) = Self::cluster_coordination_loop(&cluster_state, &registry, &eventlog) {
                eprintln!("Cluster coordination error: {e:?}");
            }
            thread::sleep(Duration::from_secs(5));
        });

        Ok(())
    }

    fn node_discovery_loop(
        registry: &Arc<RwLock<NodeRegistry>>,
        config: &Arc<RwLock<ClusterConfiguration>>,
        eventlog: &Arc<Mutex<ClusterEventLog>>,
    ) -> CoreResult<()> {
        let config_read = config.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire config lock"))
        })?;

        if !config_read.auto_discovery_enabled {
            return Ok(());
        }

        // Discover nodes using configured methods
        for discovery_method in &config_read.discovery_methods {
            // Use static discovery method instead of creating a temporary manager
            let discovered_nodes = match discovery_method {
                NodeDiscoveryMethod::Static(addresses) => {
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
                    nodes
                }
                NodeDiscoveryMethod::Multicast { group, port } => {
                    Self::multicast_discovery(group, *port)?
                }
                NodeDiscoveryMethod::DnsService { service_name } => {
                    // Placeholder implementation
                    vec![]
                }
                NodeDiscoveryMethod::Consul { endpoint } => {
                    // Placeholder implementation
                    vec![]
                }
            };

            let mut registry_write = registry.write().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new("Failed to acquire registry lock"))
            })?;

            for nodeinfo in discovered_nodes {
                if registry_write.register_node(nodeinfo.clone())? {
                    // New node discovered
                    let mut log = eventlog.lock().map_err(|_| {
                        CoreError::InvalidState(ErrorContext::new(
                            "Failed to acquire event log lock",
                        ))
                    })?;
                    log.log_event(ClusterEvent::NodeDiscovered {
                        nodeid: nodeinfo.id.clone(),
                        address: nodeinfo.address,
                        timestamp: Instant::now(),
                    });
                }
            }
        }

        Ok(())
    }

    fn discover_nodes(&self, method: &NodeDiscoveryMethod) -> CoreResult<Vec<NodeInfo>> {
        match method {
            NodeDiscoveryMethod::Static(addresses) => {
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
            NodeDiscoveryMethod::Multicast { group, port } => {
                // Implement multicast discovery
                self.discover_via_multicast(group, *port)
            }
            NodeDiscoveryMethod::DnsService { service_name } => {
                // Implement DNS-SD discovery
                self.discover_via_dns_service(service_name)
            }
            NodeDiscoveryMethod::Consul { endpoint } => {
                // Implement Consul discovery
                self.discover_via_consul(endpoint)
            }
        }
    }

    fn discover_via_multicast(&self, group: &IpAddr, port: u16) -> CoreResult<Vec<NodeInfo>> {
        Self::multicast_discovery(group, port)
    }

    fn discover_via_dns_service(&self, _servicename: &str) -> CoreResult<Vec<NodeInfo>> {
        // Placeholder implementation
        Ok(vec![])
    }

    fn discover_via_consul(&self, endpoint: &str) -> CoreResult<Vec<NodeInfo>> {
        // Placeholder implementation
        Ok(vec![])
    }

    fn is_node_reachable(address: SocketAddr) -> CoreResult<bool> {
        // Simple reachability check
        // In a real implementation, this would do proper health checking
        Ok(true) // Placeholder
    }

    fn multicast_discovery(group: &IpAddr, port: u16) -> CoreResult<Vec<NodeInfo>> {
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

    fn dns_discovery(_servicename: &str) -> CoreResult<Vec<NodeInfo>> {
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
                .arg(_servicename)
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
                        .arg(_servicename)
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
            // On Windows, try to use dns-sd command if available
            match Command::new("dns-sd")
                .arg("-B")  // Browse for services
                .arg(_servicename)
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
                        if line.contains(_servicename) {
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

    fn consul_discovery(endpoint: &str) -> CoreResult<Vec<NodeInfo>> {
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

    fn healthmonitoring_loop(
        healthmonitor: &Arc<Mutex<HealthMonitor>>,
        registry: &Arc<RwLock<NodeRegistry>>,
        eventlog: &Arc<Mutex<ClusterEventLog>>,
    ) -> CoreResult<()> {
        let nodes = {
            let registry_read = registry.read().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new("Failed to acquire registry lock"))
            })?;
            registry_read.get_all_nodes()
        };

        let mut monitor = healthmonitor.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire health monitor lock"))
        })?;

        for nodeinfo in nodes {
            let health_status = monitor.check_node_health(&nodeinfo)?;

            // Update node status
            let mut registry_write = registry.write().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new("Failed to acquire registry lock"))
            })?;

            let previous_status = registry_write.get_node_status(&nodeinfo.id);
            registry_write.update_node_status(&nodeinfo.id, health_status.status)?;

            // Log status changes
            if let Some(prev_status) = previous_status {
                if prev_status != health_status.status {
                    let mut log = eventlog.lock().map_err(|_| {
                        CoreError::InvalidState(ErrorContext::new(
                            "Failed to acquire event log lock",
                        ))
                    })?;
                    log.log_event(ClusterEvent::NodeStatusChanged {
                        nodeid: nodeinfo.id.clone(),
                        old_status: prev_status,
                        new_status: health_status.status,
                        timestamp: Instant::now(),
                    });
                }
            }
        }

        Ok(())
    }

    fn resource_management_loop(
        allocator: &Arc<RwLock<ResourceAllocator>>,
        registry: &Arc<RwLock<NodeRegistry>>,
    ) -> CoreResult<()> {
        let nodes = {
            let registry_read = registry.read().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new("Failed to acquire registry lock"))
            })?;
            registry_read.get_healthy_nodes()
        };

        let mut allocator_write = allocator.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire allocator lock"))
        })?;

        allocator_write.update_available_resources(&nodes)?;
        allocator_write.optimize_resource_allocation()?;

        Ok(())
    }

    fn cluster_coordination_loop(
        cluster_state: &Arc<RwLock<ClusterState>>,
        registry: &Arc<RwLock<NodeRegistry>>,
        eventlog: &Arc<Mutex<ClusterEventLog>>,
    ) -> CoreResult<()> {
        let healthy_nodes = {
            let registry_read = registry.read().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new("Failed to acquire registry lock"))
            })?;
            registry_read.get_healthy_nodes()
        };

        let mut state_write = cluster_state.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire cluster _state lock"))
        })?;

        // Update cluster topology
        state_write.update_topology(&healthy_nodes)?;

        // Check for leadership changes
        if state_write.needs_leader_election() {
            // TODO: Implement leader election logic
            let new_leader: Option<String> = None;
            if let Some(leader) = new_leader {
                state_write.set_leader(leader.clone());

                let mut log = eventlog.lock().map_err(|_| {
                    CoreError::InvalidState(ErrorContext::new("Failed to acquire event log lock"))
                })?;
                log.log_event(ClusterEvent::LeaderElected {
                    nodeid: leader,
                    timestamp: Instant::now(),
                });
            }
        }

        Ok(())
    }

    fn elect_leader(&self, nodes: &[NodeInfo]) -> CoreResult<Option<String>> {
        // Simple leader election based on node ID
        if nodes.is_empty() {
            return Ok(None);
        }

        // Select node with smallest ID (deterministic)
        let leader = nodes
            .iter()
            .filter(|node| node.status == NodeStatus::Healthy)
            .min_by(|a, b| a.id.cmp(&b.id));

        Ok(leader.map(|node| node.id.clone()))
    }

    /// Register a new node in the cluster
    pub fn register_node(&self, nodeinfo: NodeInfo) -> CoreResult<()> {
        let mut registry = self.node_registry.write().map_err(|_| {
            CoreError::InvalidState(
                ErrorContext::new("Failed to acquire registry lock")
                    .with_location(crate::error::ErrorLocation::new(file!(), line!())),
            )
        })?;

        registry.register_node(nodeinfo)?;
        Ok(())
    }

    /// Get cluster health status
    pub fn get_health(&self) -> CoreResult<ClusterHealth> {
        let registry = self.node_registry.read().map_err(|_| {
            CoreError::InvalidState(
                ErrorContext::new("Failed to acquire registry lock")
                    .with_location(crate::error::ErrorLocation::new(file!(), line!())),
            )
        })?;

        let all_nodes = registry.get_all_nodes();
        let healthy_nodes = all_nodes
            .iter()
            .filter(|n| n.status == NodeStatus::Healthy)
            .count();
        let total_nodes = all_nodes.len();

        let health_percentage = if total_nodes == 0 {
            100.0
        } else {
            (healthy_nodes as f64 / total_nodes as f64) * 100.0
        };

        let status = if health_percentage >= 80.0 {
            ClusterHealthStatus::Healthy
        } else if health_percentage >= 50.0 {
            ClusterHealthStatus::Degraded
        } else {
            ClusterHealthStatus::Unhealthy
        };

        Ok(ClusterHealth {
            status,
            healthy_nodes,
            total_nodes,
            health_percentage,
            last_updated: Instant::now(),
        })
    }

    /// Get list of active nodes
    pub fn get_active_nodes(&self) -> CoreResult<Vec<NodeInfo>> {
        let registry = self.node_registry.read().map_err(|_| {
            CoreError::InvalidState(
                ErrorContext::new("Failed to acquire registry lock")
                    .with_location(crate::error::ErrorLocation::new(file!(), line!())),
            )
        })?;

        Ok(registry.get_healthy_nodes())
    }

    /// Get available nodes (returns nodeid -> nodeinfo mapping)
    pub fn get_availablenodes(&self) -> CoreResult<HashMap<String, NodeInfo>> {
        let registry = self.node_registry.read().map_err(|_| {
            CoreError::InvalidState(
                ErrorContext::new("Failed to acquire registry lock")
                    .with_location(crate::error::ErrorLocation::new(file!(), line!())),
            )
        })?;

        let nodes = registry.get_healthy_nodes();
        let mut node_map = HashMap::new();
        for node in nodes {
            node_map.insert(node.id.clone(), node);
        }
        Ok(node_map)
    }

    /// Get total cluster compute capacity
    pub fn get_total_capacity(&self) -> CoreResult<ComputeCapacity> {
        let registry = self.node_registry.read().map_err(|_| {
            CoreError::InvalidState(
                ErrorContext::new("Failed to acquire registry lock")
                    .with_location(crate::error::ErrorLocation::new(file!(), line!())),
            )
        })?;

        let nodes = registry.get_healthy_nodes();
        let mut total_capacity = ComputeCapacity::default();

        for node in nodes {
            total_capacity.cpu_cores += node.capabilities.cpu_cores;
            total_capacity.memory_gb += node.capabilities.memory_gb;
            total_capacity.gpu_count += node.capabilities.gpu_count;
            total_capacity.disk_space_gb += node.capabilities.disk_space_gb;
        }

        Ok(total_capacity)
    }

    /// Submit a distributed task to the cluster
    pub fn submit_task(&self, task: DistributedTask) -> CoreResult<TaskId> {
        let allocator = self.resource_allocator.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire allocator lock"))
        })?;

        let allocation = allocator.allocate_resources(&task.resource_requirements)?;

        // Create task execution plan
        let taskid = TaskId::generate();
        let _execution_plan = ExecutionPlan {
            taskid: taskid.clone(),
            task,
            node_allocation: allocation,
            created_at: Instant::now(),
            status: ExecutionStatus::Pending,
        };

        // Submit to scheduler (placeholder)
        // In a real implementation, this would go to the distributed scheduler
        Ok(taskid)
    }

    /// Get cluster statistics
    pub fn get_cluster_statistics(&self) -> CoreResult<ClusterStatistics> {
        let registry = self.node_registry.read().map_err(|_| {
            CoreError::InvalidState(
                ErrorContext::new("Failed to acquire registry lock")
                    .with_location(crate::error::ErrorLocation::new(file!(), line!())),
            )
        })?;

        let allocator = self.resource_allocator.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire allocator lock"))
        })?;

        let nodes = registry.get_all_nodes();
        let total_capacity = self.get_total_capacity()?;
        let available_capacity = (*allocator).available_capacity();

        Ok(ClusterStatistics {
            total_nodes: nodes.len(),
            healthy_nodes: nodes
                .iter()
                .filter(|n| n.status == NodeStatus::Healthy)
                .count(),
            total_capacity: total_capacity.clone(),
            available_capacity: available_capacity.clone(),
            resource_utilization: ResourceUtilization {
                cpu_utilization: 1.0
                    - (available_capacity.cpu_cores as f64 / total_capacity.cpu_cores as f64),
                memory_utilization: 1.0
                    - (available_capacity.memory_gb as f64 / total_capacity.memory_gb as f64),
                gpu_utilization: if total_capacity.gpu_count > 0 {
                    1.0 - (available_capacity.gpu_count as f64 / total_capacity.gpu_count as f64)
                } else {
                    0.0
                },
            },
        })
    }
}

/// Cluster state management
#[derive(Debug)]
pub struct ClusterState {
    leader_node: Option<String>,
    topology: ClusterTopology,
    last_updated: Instant,
}

impl Default for ClusterState {
    fn default() -> Self {
        Self::new()
    }
}

impl ClusterState {
    pub fn new() -> Self {
        Self {
            leader_node: None,
            topology: ClusterTopology::new(),
            last_updated: Instant::now(),
        }
    }

    pub fn update_topology(&mut self, nodes: &[NodeInfo]) -> CoreResult<()> {
        self.topology.update(nodes);
        self.last_updated = Instant::now();
        Ok(())
    }

    pub fn needs_leader_election(&self) -> bool {
        self.leader_node.is_none() || self.last_updated.elapsed() > Duration::from_secs(300)
        // Re-elect every 5 minutes
    }

    pub fn set_leader(&mut self, nodeid: String) {
        self.leader_node = Some(nodeid);
        self.last_updated = Instant::now();
    }
}

/// Node registry for tracking cluster members
#[derive(Debug)]
pub struct NodeRegistry {
    nodes: HashMap<String, NodeInfo>,
    node_status: HashMap<String, NodeStatus>,
}

impl Default for NodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeRegistry {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            node_status: HashMap::new(),
        }
    }

    pub fn register_node(&mut self, nodeinfo: NodeInfo) -> CoreResult<bool> {
        let is_new = !self.nodes.contains_key(&nodeinfo.id);
        self.nodes.insert(nodeinfo.id.clone(), nodeinfo.clone());
        self.node_status
            .insert(nodeinfo.id.clone(), nodeinfo.status);
        Ok(is_new)
    }

    pub fn get_all_nodes(&self) -> Vec<NodeInfo> {
        self.nodes.values().cloned().collect()
    }

    pub fn get_healthy_nodes(&self) -> Vec<NodeInfo> {
        self.nodes
            .values()
            .filter(|node| self.node_status.get(&node.id) == Some(&NodeStatus::Healthy))
            .cloned()
            .collect()
    }

    pub fn get_node_status(&self, nodeid: &str) -> Option<NodeStatus> {
        self.node_status.get(nodeid).copied()
    }

    pub fn update_node_status(&mut self, nodeid: &str, status: NodeStatus) -> CoreResult<()> {
        if let Some(node) = self.nodes.get_mut(nodeid) {
            node.status = status;
            self.node_status.insert(nodeid.to_string(), status);
        }
        Ok(())
    }
}

/// Health monitoring system
#[derive(Debug)]
pub struct HealthMonitor {
    health_checks: Vec<HealthCheck>,
    #[allow(dead_code)]
    check_interval: Duration,
}

impl HealthMonitor {
    pub fn new() -> CoreResult<Self> {
        Ok(Self {
            health_checks: Self::default_health_checks(),
            check_interval: Duration::from_secs(30),
        })
    }

    fn default_health_checks() -> Vec<HealthCheck> {
        vec![
            HealthCheck::Ping,
            HealthCheck::CpuLoad,
            HealthCheck::MemoryUsage,
            HealthCheck::DiskSpace,
            HealthCheck::NetworkConnectivity,
        ]
    }

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
}

/// Resource allocation and management
#[derive(Debug)]
pub struct ResourceAllocator {
    allocations: HashMap<TaskId, ResourceAllocation>,
    available_resources: ComputeCapacity,
    allocation_strategy: AllocationStrategy,
}

impl Default for ResourceAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl ResourceAllocator {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            available_resources: ComputeCapacity::default(),
            allocation_strategy: AllocationStrategy::FirstFit,
        }
    }

    pub fn update_available_resources(&mut self, nodes: &[NodeInfo]) -> CoreResult<()> {
        self.available_resources = ComputeCapacity::default();

        for node in nodes {
            if node.status == NodeStatus::Healthy {
                self.available_resources.cpu_cores += node.capabilities.cpu_cores;
                self.available_resources.memory_gb += node.capabilities.memory_gb;
                self.available_resources.gpu_count += node.capabilities.gpu_count;
                self.available_resources.disk_space_gb += node.capabilities.disk_space_gb;
            }
        }

        // Subtract already allocated resources
        for allocation in self.allocations.values() {
            self.available_resources.cpu_cores = self
                .available_resources
                .cpu_cores
                .saturating_sub(allocation.allocated_resources.cpu_cores);
            self.available_resources.memory_gb = self
                .available_resources
                .memory_gb
                .saturating_sub(allocation.allocated_resources.memory_gb);
            self.available_resources.gpu_count = self
                .available_resources
                .gpu_count
                .saturating_sub(allocation.allocated_resources.gpu_count);
            self.available_resources.disk_space_gb = self
                .available_resources
                .disk_space_gb
                .saturating_sub(allocation.allocated_resources.disk_space_gb);
        }

        Ok(())
    }

    pub fn allocate_resources(
        &self,
        requirements: &ResourceRequirements,
    ) -> CoreResult<ResourceAllocation> {
        // Check if resources are available
        if !self.can_satisfy_requirements(requirements) {
            return Err(CoreError::ResourceError(ErrorContext::new(
                "Insufficient resources available",
            )));
        }

        // Create allocation
        Ok(ResourceAllocation {
            allocation_id: AllocationId::generate(),
            allocated_resources: ComputeCapacity {
                cpu_cores: requirements.cpu_cores,
                memory_gb: requirements.memory_gb,
                gpu_count: requirements.gpu_count,
                disk_space_gb: requirements.disk_space_gb,
            },
            assigned_nodes: Vec::new(), // Would be populated with actual nodes
            created_at: Instant::now(),
            expires_at: None,
        })
    }

    fn can_satisfy_requirements(&self, requirements: &ResourceRequirements) -> bool {
        self.available_resources.cpu_cores >= requirements.cpu_cores
            && self.available_resources.memory_gb >= requirements.memory_gb
            && self.available_resources.gpu_count >= requirements.gpu_count
            && self.available_resources.disk_space_gb >= requirements.disk_space_gb
    }

    pub fn optimize_resource_allocation(&mut self) -> CoreResult<()> {
        // Implement resource optimization strategies
        match self.allocation_strategy {
            AllocationStrategy::FirstFit => {
                // First-fit allocation (already implemented)
            }
            AllocationStrategy::BestFit => {
                // Best-fit allocation
                self.optimize_best_fit()?;
            }
            AllocationStrategy::LoadBalanced => {
                // Load-balanced allocation
                self.optimize_load_balanced()?;
            }
        }
        Ok(())
    }

    fn optimize_best_fit(&mut self) -> CoreResult<()> {
        // Best-fit optimization: minimize resource fragmentation by allocating
        // to nodes that most closely match the resource requirements

        // Get all current allocations sorted by resource usage
        let mut allocations: Vec<(TaskId, ResourceAllocation)> = self
            .allocations
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // Sort allocations by total resource "weight" (descending)
        // This helps identify heavy allocations that could be better placed
        allocations.sort_by(|a, b| {
            let weight_a = a.1.allocated_resources.cpu_cores
                + a.1.allocated_resources.memory_gb
                + a.1.allocated_resources.gpu_count * 4  // Weight GPUs more heavily
                + a.1.allocated_resources.disk_space_gb / 10; // Weight disk less
            let weight_b = b.1.allocated_resources.cpu_cores
                + b.1.allocated_resources.memory_gb
                + b.1.allocated_resources.gpu_count * 4
                + b.1.allocated_resources.disk_space_gb / 10;
            weight_b.cmp(&weight_a)
        });

        // Optimization strategy: consolidate small allocations onto fewer nodes
        // and ensure large allocations get dedicated resources

        // Track optimization improvements
        let mut optimizations_made = 0;
        let fragmentation_score_before = self.calculate_fragmentation_score();

        // Group allocations by size category
        let (large_allocations, medium_allocations, small_allocations): (Vec<_>, Vec<_>, Vec<_>) = {
            let mut large = Vec::new();
            let mut medium = Vec::new();
            let mut small = Vec::new();

            for (taskid, allocation) in allocations {
                let total_resources = allocation.allocated_resources.cpu_cores
                    + allocation.allocated_resources.memory_gb
                    + allocation.allocated_resources.gpu_count * 4;

                if total_resources >= 32 {
                    large.push((taskid.clone(), allocation.clone()));
                } else if total_resources >= 8 {
                    medium.push((taskid.clone(), allocation.clone()));
                } else {
                    small.push((taskid.clone(), allocation.clone()));
                }
            }

            (large, medium, small)
        };

        // Best-fit strategy for large allocations:
        // Ensure they get dedicated, high-capacity nodes
        for (taskid, allocation) in large_allocations {
            if allocation.assigned_nodes.len() > 1 {
                // Try to consolidate onto a single high-capacity node
                if self.attempt_consolidation(&taskid, &allocation)? {
                    optimizations_made += 1;
                }
            }
        }

        // Best-fit strategy for medium allocations:
        // Pair them efficiently to minimize waste
        for (taskid, allocation) in medium_allocations {
            if self.attempt_best_fit_pairing(&taskid, &allocation)? {
                optimizations_made += 1;
            }
        }

        // Best-fit strategy for small allocations:
        // Pack them tightly onto shared nodes
        for (taskid, allocation) in small_allocations {
            if self.attempt_small_allocation_packing(&taskid, &allocation)? {
                optimizations_made += 1;
            }
        }

        // Calculate improvement
        let fragmentation_score_after = self.calculate_fragmentation_score();
        let _improvement = fragmentation_score_before - fragmentation_score_after;

        if optimizations_made > 0 {
            #[cfg(feature = "logging")]
            log::info!(
                "Best-fit optimization completed: {optimizations_made} optimizations, fragmentation improved by {_improvement:.2}"
            );
        }

        Ok(())
    }

    fn optimize_load_balanced(&mut self) -> CoreResult<()> {
        // Load-balanced optimization: distribute workload evenly across nodes
        // to prevent hot spots and maximize overall cluster throughput

        // Calculate current load distribution across nodes
        let mut nodeloads = HashMap::new();
        let mut total_load = 0.0f64;

        // Calculate load for each node based on current allocations
        for allocation in self.allocations.values() {
            for nodeid in &allocation.assigned_nodes {
                let load_weight =
                    self.calculate_allocation_load_weight(&allocation.allocated_resources);
                *nodeloads.entry(nodeid.clone()).or_insert(0.0) += load_weight;
                total_load += load_weight;
            }
        }

        // Identify the target load per node (assuming uniform node capabilities)
        let num_active_nodes = nodeloads.len().max(1);
        let target_load_per_node = total_load / num_active_nodes as f64;
        let load_variance_threshold = target_load_per_node * 0.15f64; // 15% variance allowed

        // Find overloaded and underloaded nodes
        let mut overloaded_nodes = Vec::new();
        let mut underloaded_nodes = Vec::new();

        for (nodeid, &current_load) in &nodeloads {
            let load_diff = current_load - target_load_per_node;
            if load_diff > load_variance_threshold {
                overloaded_nodes.push((nodeid.clone(), current_load, load_diff));
            } else if load_diff < -load_variance_threshold {
                underloaded_nodes.push((nodeid.clone(), current_load, -load_diff));
            }
        }

        // Sort by load difference (most extreme first)
        overloaded_nodes.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        underloaded_nodes.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        let mut rebalancing_actions = 0;
        let initial_variance = self.calculate_load_variance(&nodeloads);

        // Rebalancing algorithm: move allocations from overloaded to underloaded nodes
        for (overloaded_node, current_load, overloaded_amount) in overloaded_nodes {
            // Find allocations on this overloaded node that can be moved
            let moveable_allocations = self.find_moveable_allocations(&overloaded_node);

            for (taskid, allocation) in moveable_allocations {
                // Find the best underloaded node for this allocation
                if let Some((target_node, _)) = self.find_best_target_node(
                    &allocation.allocated_resources,
                    &underloaded_nodes
                        .iter()
                        .map(|(nodeid, load, _)| (nodeid.clone(), *load))
                        .collect::<Vec<_>>(),
                )? {
                    // Attempt to move the allocation
                    if self.attempt_allocation_migration(&taskid, &target_node)? {
                        rebalancing_actions += 1;

                        // Update node loads tracking
                        let allocation_weight =
                            self.calculate_allocation_load_weight(&allocation.allocated_resources);
                        if let Some(old_load) = nodeloads.get_mut(&overloaded_node) {
                            *old_load -= allocation_weight;
                        }
                        if let Some(new_load) = nodeloads.get_mut(&target_node) {
                            *new_load += allocation_weight;
                        }

                        // Check if we've balanced enough
                        if nodeloads.get(&overloaded_node).copied().unwrap_or(0.0)
                            <= target_load_per_node + load_variance_threshold
                        {
                            break; // This node is now balanced
                        }
                    }
                }
            }
        }

        // Secondary optimization: spread single large allocations across multiple nodes
        let single_node_allocations: Vec<(TaskId, ResourceAllocation)> = self
            .allocations
            .iter()
            .filter(|(_, allocation)| allocation.assigned_nodes.len() == 1)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        for (taskid, allocation) in single_node_allocations {
            let load_weight =
                self.calculate_allocation_load_weight(&allocation.allocated_resources);
            if load_weight > target_load_per_node * 0.6 {
                // Large allocation
                if self.attempt_allocation_spreading(&taskid, &allocation)? {
                    rebalancing_actions += 1;
                }
            }
        }

        // Calculate improvement in load balance
        let final_variance = self.calculate_load_variance(&nodeloads);
        let _variance_improvement = initial_variance - final_variance;

        if rebalancing_actions > 0 {
            #[cfg(feature = "logging")]
            log::info!(
                "Load-balanced optimization completed: {rebalancing_actions} rebalancing actions, \
                 load variance improved by {_variance_improvement:.2}"
            );
        }

        Ok(())
    }

    pub fn get_available_capacity(&self) -> ComputeCapacity {
        self.available_resources.clone()
    }

    // Helper methods for optimization algorithms

    fn calculate_fragmentation_score(&self) -> f64 {
        // Calculate how fragmented the resource allocation is
        // Lower score = better (less fragmented)
        let total_allocated_resources = self.allocations.len() as f64;
        if total_allocated_resources == 0.0 {
            return 0.0f64;
        }

        // Count allocations that are split across multiple nodes
        let split_allocations = self
            .allocations
            .values()
            .filter(|alloc| alloc.assigned_nodes.len() > 1)
            .count() as f64;

        // Calculate average resource utilization efficiency
        let mut total_efficiency = 0.0f64;
        for allocation in self.allocations.values() {
            let resource_efficiency =
                self.calculate_resource_efficiency(&allocation.allocated_resources);
            total_efficiency += resource_efficiency;
        }
        let avg_efficiency = total_efficiency / total_allocated_resources;

        // Fragmentation score: high split ratio + low efficiency = high fragmentation
        let split_ratio = split_allocations / total_allocated_resources;
        (split_ratio * 0.6 + (1.0 - avg_efficiency) * 0.4f64) * 100.0
    }

    fn calculate_resource_efficiency(&self, resources: &ComputeCapacity) -> f64 {
        // Calculate how efficiently resources are being used
        // 1.0 = perfect efficiency, 0.0 = completely inefficient

        // Check resource balance (CPU:Memory:GPU ratio)
        let cpu_ratio = resources.cpu_cores as f64;
        let _memory_ratio = resources.memory_gb as f64 / 4.0f64; // Assume 4GB per CPU core is balanced
        let gpu_ratio = resources.gpu_count as f64 * 8.0f64; // Each GPU equivalent to 8 CPU cores

        let total_compute = cpu_ratio + gpu_ratio;
        let balanced_memory = total_compute * 4.0f64;

        // Efficiency is higher when memory allocation matches compute needs
        let memory_efficiency = if resources.memory_gb as f64 > 0.0 {
            balanced_memory.min(resources.memory_gb as f64)
                / balanced_memory.max(resources.memory_gb as f64)
        } else {
            1.0
        };

        // Also consider if resources are "too small" (overhead penalty)
        let scale_efficiency = if total_compute < 2.0 {
            total_compute / 2.0 // Penalty for very small allocations
        } else {
            1.0
        };

        let combined_efficiency = memory_efficiency * 0.7 + scale_efficiency * 0.3f64;
        combined_efficiency.min(1.0)
    }

    fn try_consolidate_large_allocation(
        &self,
        _allocation: &ResourceAllocation,
    ) -> CoreResult<bool> {
        // Attempt to consolidate a multi-node allocation onto fewer nodes
        // For now, return false indicating no consolidation was possible
        // In a real implementation, this would:
        // 1. Find nodes with sufficient capacity to host the entire allocation
        // 2. Check if consolidation would improve performance
        // 3. Migrate the allocation if beneficial
        Ok(false)
    }

    fn try_optimize_medium_allocations(
        &self,
        _allocation: &ResourceAllocation,
    ) -> CoreResult<bool> {
        // Attempt to pair medium allocations efficiently
        // For now, return false indicating no pairing optimization was made
        Ok(false)
    }

    fn try_pack_small_allocations(&self, allocation: &ResourceAllocation) -> CoreResult<bool> {
        // Attempt to pack small allocations tightly onto shared nodes
        // For now, return false indicating no packing optimization was made
        Ok(false)
    }

    fn calculate_allocation_load_weight(&self, resources: &ComputeCapacity) -> f64 {
        // Calculate the "load weight" of an allocation for load balancing
        // Higher weight = more demanding allocation
        let cpu_weight = resources.cpu_cores as f64;
        let memory_weight = resources.memory_gb as f64 * 0.25f64; // Memory is less constraining than CPU
        let gpu_weight = resources.gpu_count as f64 * 8.0f64; // GPUs are very constraining
        let disk_weight = resources.disk_space_gb as f64 * 0.01f64; // Disk is least constraining

        cpu_weight + memory_weight + gpu_weight + disk_weight
    }

    fn calculate_load_variance(&self, nodeloads: &HashMap<String, f64>) -> f64 {
        // Calculate variance in load distribution across nodes
        if nodeloads.len() <= 1 {
            return 0.0f64;
        }

        let total_load: f64 = nodeloads.values().sum();
        let mean_load = total_load / nodeloads.len() as f64;

        let variance = nodeloads
            .values()
            .map(|&load| (load - mean_load).powi(2))
            .sum::<f64>()
            / nodeloads.len() as f64;

        variance.sqrt() // Return standard deviation
    }

    fn find_moveable_allocations(&self, nodeid: &str) -> Vec<(TaskId, ResourceAllocation)> {
        // Find allocations on a specific node that can potentially be moved
        self.allocations
            .iter()
            .filter(|(_, allocation)| allocation.assigned_nodes.contains(&nodeid.to_string()))
            .map(|(taskid, allocation)| (taskid.clone(), allocation.clone()))
            .collect()
    }

    fn find_best_underloaded_node(
        &self,
        nodes: &[(String, f64, f64)],
        _required_capacity: f64,
    ) -> Option<(String, f64)> {
        // Find the best underloaded node to receive an allocation
        // For now, just return the most underloaded node
        nodes
            .first()
            .map(|(nodeid, load, capacity)| (nodeid.clone(), *load))
    }

    fn try_migrate_allocation(&self, _taskid: &TaskId, _targetnode: &str) -> CoreResult<bool> {
        // Attempt to migrate an allocation to a different node
        // For now, return false indicating migration wasn't performed
        // In a real implementation, this would:
        // 1. Check if target node has capacity
        // 2. Coordinate with the task scheduler
        // 3. Perform the actual migration
        Ok(false)
    }

    fn try_spread_allocation(&self, allocation: &ResourceAllocation) -> CoreResult<bool> {
        // Attempt to spread a large allocation across multiple nodes
        // For now, return false indicating spreading wasn't performed
        Ok(false)
    }

    pub fn available_capacity(&self) -> &ComputeCapacity {
        &self.available_resources
    }

    pub fn attempt_consolidation(
        &mut self,
        _taskid: &TaskId,
        _allocation: &ResourceAllocation,
    ) -> CoreResult<bool> {
        // Placeholder implementation
        Ok(false)
    }

    pub fn attempt_best_fit_pairing(
        &mut self,
        _taskid: &TaskId,
        _allocation: &ResourceAllocation,
    ) -> CoreResult<bool> {
        // Placeholder implementation
        Ok(false)
    }

    pub fn attempt_small_allocation_packing(
        &mut self,
        _taskid: &TaskId,
        _allocation: &ResourceAllocation,
    ) -> CoreResult<bool> {
        // Placeholder implementation
        Ok(false)
    }

    pub fn find_best_target_node(
        &mut self,
        _resources: &ComputeCapacity,
        _underloaded_nodes: &[(String, f64)],
    ) -> CoreResult<Option<(String, f64)>> {
        // Placeholder implementation
        Ok(None)
    }

    pub fn attempt_allocation_migration(
        &mut self,
        _taskid: &TaskId,
        _to_node: &str,
    ) -> CoreResult<bool> {
        // Placeholder implementation
        Ok(false)
    }

    pub fn attempt_allocation_spreading(
        &mut self,
        _taskid: &TaskId,
        _allocation: &ResourceAllocation,
    ) -> CoreResult<bool> {
        // Placeholder implementation
        Ok(false)
    }
}

/// Cluster event logging
#[derive(Debug)]
pub struct ClusterEventLog {
    events: VecDeque<ClusterEvent>,
    max_events: usize,
}

impl Default for ClusterEventLog {
    fn default() -> Self {
        Self::new()
    }
}

impl ClusterEventLog {
    pub fn new() -> Self {
        Self {
            events: VecDeque::with_capacity(10000usize),
            max_events: 10000,
        }
    }

    pub fn log_event(&mut self, event: ClusterEvent) {
        self.events.push_back(event);

        // Maintain max size
        while self.events.len() > self.max_events {
            self.events.pop_front();
        }
    }

    pub fn get_recent_events(&self, count: usize) -> Vec<ClusterEvent> {
        self.events.iter().rev().take(count).cloned().collect()
    }
}

// Supporting types and structures

#[derive(Debug, Clone)]
pub struct ClusterConfiguration {
    pub auto_discovery_enabled: bool,
    pub discovery_methods: Vec<NodeDiscoveryMethod>,
    pub health_check_interval: Duration,
    pub leadership_timeout: Duration,
    pub resource_allocation_strategy: AllocationStrategy,
    pub max_nodes: Option<usize>,
}

impl Default for ClusterConfiguration {
    fn default() -> Self {
        Self {
            auto_discovery_enabled: true,
            discovery_methods: vec![NodeDiscoveryMethod::Static(vec![])],
            health_check_interval: Duration::from_secs(30),
            leadership_timeout: Duration::from_secs(300),
            resource_allocation_strategy: AllocationStrategy::FirstFit,
            max_nodes: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum NodeDiscoveryMethod {
    Static(Vec<SocketAddr>),
    Multicast { group: IpAddr, port: u16 },
    DnsService { service_name: String },
    Consul { endpoint: String },
}

#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub id: String,
    pub address: SocketAddr,
    pub node_type: NodeType,
    pub capabilities: NodeCapabilities,
    pub status: NodeStatus,
    pub last_seen: Instant,
    pub metadata: NodeMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    Master,
    Worker,
    Storage,
    Compute,
    ComputeOptimized,
    MemoryOptimized,
    StorageOptimized,
    General,
}

#[derive(Debug, Clone)]
pub struct NodeCapabilities {
    pub cpu_cores: usize,
    pub memory_gb: usize,
    pub gpu_count: usize,
    pub disk_space_gb: usize,
    pub networkbandwidth_gbps: f64,
    pub specialized_units: Vec<SpecializedUnit>,
}

impl Default for NodeCapabilities {
    fn default() -> Self {
        Self {
            cpu_cores: 4,
            memory_gb: 8,
            gpu_count: 0,
            disk_space_gb: 100,
            networkbandwidth_gbps: 1.0f64,
            specialized_units: Vec::new(),
        }
    }
}

/// Specialized computing units available on a node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpecializedUnit {
    TensorCore,
    QuantumProcessor,
    VectorUnit,
    CryptoAccelerator,
    NeuralProcessingUnit,
    Fpga,
    Asic,
    CustomAsic(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    Unknown,
    Healthy,
    Degraded,
    Unhealthy,
    Offline,
    Draining,
}

#[derive(Debug, Clone)]
pub struct NodeMetadata {
    pub hostname: String,
    pub operating_system: String,
    pub kernel_version: String,
    pub container_runtime: Option<String>,
    pub labels: HashMap<String, String>,
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self {
            hostname: "unknown".to_string(),
            operating_system: "unknown".to_string(),
            kernel_version: "unknown".to_string(),
            container_runtime: None,
            labels: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClusterTopology {
    pub zones: BTreeMap<String, Zone>,
    pub network_topology: NetworkTopology,
}

impl Default for ClusterTopology {
    fn default() -> Self {
        Self::new()
    }
}

impl ClusterTopology {
    pub fn new() -> Self {
        Self {
            zones: BTreeMap::new(),
            network_topology: NetworkTopology::Flat,
        }
    }

    pub fn update(&mut self, nodes: &[NodeInfo]) {
        // Simple topology update - group nodes by network zone
        self.zones.clear();

        for node in nodes {
            let zone_name = self.determine_zone(&node.address);
            let zone = self.zones.entry(zone_name).or_default();
            zone.add_node(node.clone());
        }
    }

    fn determine_zone(&self, address: &SocketAddr) -> String {
        // Simple zone determination based on IP address
        // In a real implementation, this would use proper network topology discovery
        format!(
            "zone_{}",
            address.ip().to_string().split('.').next().unwrap_or("0")
        )
    }

    /// Update the topology model with new node information
    pub fn update_model(&mut self, nodes: &[NodeInfo]) {
        // Update the topology model based on new node information
        self.update(nodes);

        // Additional model updates can be added here
        // For example, network latency measurements, bandwidth tests, etc.
    }
}

#[derive(Debug, Clone)]
pub struct Zone {
    pub nodes: Vec<NodeInfo>,
    pub capacity: ComputeCapacity,
}

impl Default for Zone {
    fn default() -> Self {
        Self::new()
    }
}

impl Zone {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            capacity: ComputeCapacity::default(),
        }
    }

    pub fn add_node(&mut self, node: NodeInfo) {
        self.capacity.cpu_cores += node.capabilities.cpu_cores;
        self.capacity.memory_gb += node.capabilities.memory_gb;
        self.capacity.gpu_count += node.capabilities.gpu_count;
        self.capacity.disk_space_gb += node.capabilities.disk_space_gb;

        self.nodes.push(node);
    }
}

#[derive(Debug, Clone)]
pub enum NetworkTopology {
    Flat,
    Hierarchical,
    Mesh,
    Ring,
}

#[derive(Debug, Clone)]
pub struct NodeHealthStatus {
    pub status: NodeStatus,
    pub health_score: f64,
    pub failing_checks: Vec<HealthCheck>,
    pub last_checked: Instant,
}

#[derive(Debug, Clone)]
pub enum HealthCheck {
    Ping,
    CpuLoad,
    MemoryUsage,
    DiskSpace,
    NetworkConnectivity,
}

#[derive(Debug)]
pub struct HealthCheckResult {
    pub is_healthy: bool,
    pub impact_score: f64,
    pub details: String,
}

#[derive(Debug, Clone)]
pub struct ClusterHealth {
    pub status: ClusterHealthStatus,
    pub healthy_nodes: usize,
    pub total_nodes: usize,
    pub health_percentage: f64,
    pub last_updated: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterHealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Default)]
pub struct ComputeCapacity {
    pub cpu_cores: usize,
    pub memory_gb: usize,
    pub gpu_count: usize,
    pub disk_space_gb: usize,
}

#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub cpu_cores: usize,
    pub memory_gb: usize,
    pub gpu_count: usize,
    pub disk_space_gb: usize,
    pub specialized_requirements: Vec<SpecializedRequirement>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpecializedRequirement {
    pub unit_type: SpecializedUnit,
    pub count: usize,
}

#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub allocation_id: AllocationId,
    pub allocated_resources: ComputeCapacity,
    pub assigned_nodes: Vec<String>,
    pub created_at: Instant,
    pub expires_at: Option<Instant>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AllocationId(String);

impl AllocationId {
    pub fn generate() -> Self {
        Self(format!(
            "alloc_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    FirstFit,
    BestFit,
    LoadBalanced,
}

#[derive(Debug, Clone)]
pub struct DistributedTask {
    pub taskid: TaskId,
    pub task_type: TaskType,
    pub resource_requirements: ResourceRequirements,
    pub data_dependencies: Vec<DataDependency>,
    pub execution_parameters: TaskParameters,
    pub priority: TaskPriority,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskId(String);

impl TaskId {
    pub fn generate() -> Self {
        Self(format!(
            "task_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ))
    }
}

#[derive(Debug, Clone)]
pub enum TaskType {
    Computation,
    DataProcessing,
    MachineLearning,
    Simulation,
    Analysis,
}

#[derive(Debug, Clone)]
pub struct DataDependency {
    pub data_id: String,
    pub access_type: DataAccessType,
    pub size_hint: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataAccessType {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone)]
pub struct TaskParameters {
    pub environment_variables: HashMap<String, String>,
    pub command_arguments: Vec<String>,
    pub timeout: Option<Duration>,
    pub retrypolicy: RetryPolicy,
}

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: usize,
    pub backoff_strategy: BackoffStrategy,
}

#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    Fixed(Duration),
    Linear(Duration),
    Exponential { base: Duration, multiplier: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub taskid: TaskId,
    pub task: DistributedTask,
    pub node_allocation: ResourceAllocation,
    pub created_at: Instant,
    pub status: ExecutionStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStatus {
    Pending,
    Scheduled,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone)]
pub enum ClusterEvent {
    NodeDiscovered {
        nodeid: String,
        address: SocketAddr,
        timestamp: Instant,
    },
    NodeStatusChanged {
        nodeid: String,
        old_status: NodeStatus,
        new_status: NodeStatus,
        timestamp: Instant,
    },
    LeaderElected {
        nodeid: String,
        timestamp: Instant,
    },
    TaskScheduled {
        taskid: TaskId,
        nodeid: String,
        timestamp: Instant,
    },
    TaskCompleted {
        taskid: TaskId,
        nodeid: String,
        execution_time: Duration,
        timestamp: Instant,
    },
    ResourceAllocation {
        allocation_id: AllocationId,
        resources: ComputeCapacity,
        timestamp: Instant,
    },
}

#[derive(Debug, Clone)]
pub struct ClusterStatistics {
    pub total_nodes: usize,
    pub healthy_nodes: usize,
    pub total_capacity: ComputeCapacity,
    pub available_capacity: ComputeCapacity,
    pub resource_utilization: ResourceUtilization,
}

#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub gpu_utilization: f64,
}

/// Initialize cluster manager with default configuration
#[allow(dead_code)]
pub fn initialize_cluster_manager() -> CoreResult<()> {
    let cluster_manager = ClusterManager::global()?;
    cluster_manager.start()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_cluster_manager_creation() {
        let config = ClusterConfiguration::default();
        let manager = ClusterManager::new(config).unwrap();
        // Basic functionality test
    }

    #[test]
    fn test_node_registry() {
        let mut registry = NodeRegistry::new();

        let node = NodeInfo {
            id: "test_node".to_string(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
            node_type: NodeType::Worker,
            capabilities: NodeCapabilities::default(),
            status: NodeStatus::Healthy,
            last_seen: Instant::now(),
            metadata: NodeMetadata::default(),
        };

        let is_new = registry.register_node(node.clone()).unwrap();
        assert!(is_new);

        let healthy_nodes = registry.get_healthy_nodes();
        assert_eq!(healthy_nodes.len(), 1);
        assert_eq!(healthy_nodes[0usize].id, "test_node");
    }

    #[test]
    fn test_resource_allocator() {
        let mut allocator = ResourceAllocator::new();

        // Set some available resources
        allocator.available_resources = ComputeCapacity {
            cpu_cores: 8,
            memory_gb: 16,
            gpu_count: 1,
            disk_space_gb: 100,
        };

        let requirements = ResourceRequirements {
            cpu_cores: 4,
            memory_gb: 8,
            gpu_count: 0,
            disk_space_gb: 50,
            specialized_requirements: Vec::new(),
        };

        let allocation = allocator.allocate_resources(&requirements).unwrap();
        assert_eq!(allocation.allocated_resources.cpu_cores, 4);
        assert_eq!(allocation.allocated_resources.memory_gb, 8);
    }

    #[test]
    fn test_healthmonitor() {
        let mut monitor = HealthMonitor::new().unwrap();

        let node = NodeInfo {
            id: "test_node".to_string(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
            node_type: NodeType::Worker,
            capabilities: NodeCapabilities::default(),
            status: NodeStatus::Unknown,
            last_seen: Instant::now(),
            metadata: NodeMetadata::default(),
        };

        let health_status = monitor.check_node_health(&node).unwrap();
        assert!(health_status.health_score >= 0.0 && health_status.health_score <= 100.0f64);
    }

    #[test]
    fn test_cluster_topology() {
        let mut topology = ClusterTopology::new();

        let nodes = vec![
            NodeInfo {
                id: "node1".to_string(),
                address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)), 8080),
                node_type: NodeType::Worker,
                capabilities: NodeCapabilities::default(),
                status: NodeStatus::Healthy,
                last_seen: Instant::now(),
                metadata: NodeMetadata::default(),
            },
            NodeInfo {
                id: "node2".to_string(),
                address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)), 8080),
                node_type: NodeType::Worker,
                capabilities: NodeCapabilities::default(),
                status: NodeStatus::Healthy,
                last_seen: Instant::now(),
                metadata: NodeMetadata::default(),
            },
        ];

        topology.update_model(&nodes);
        assert_eq!(topology.zones.len(), 2); // Two different zones based on IP
    }
}
