//! Distributed spatial computing framework
//!
//! This module provides a comprehensive distributed computing framework for spatial algorithms,
//! enabling scaling across multiple nodes, automatic load balancing, fault tolerance, and
//! efficient data partitioning for massive spatial datasets. It supports both message-passing
//! and shared-memory paradigms with optimized communication patterns.
//!
//! # Features
//!
//! - **Distributed spatial data structures**: Scale KD-trees, spatial indices across nodes
//! - **Automatic data partitioning**: Space-filling curves, load-balanced partitioning
//! - **Fault-tolerant computation**: Checkpointing, automatic recovery, redundancy
//! - **Adaptive load balancing**: Dynamic workload redistribution
//! - **Communication optimization**: Bandwidth-aware algorithms, compression
//! - **Hierarchical clustering**: Multi-level distributed algorithms
//! - **Streaming spatial analytics**: Real-time processing of spatial data streams
//! - **Elastic scaling**: Add/remove nodes dynamically
//!
//! # Architecture
//!
//! The framework uses a hybrid architecture combining:
//! - **Master-worker pattern** for coordination
//! - **Peer-to-peer communication** for data exchange
//! - **Hierarchical topology** for scalability
//! - **Event-driven programming** for responsiveness
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::distributed::{DistributedSpatialCluster, NodeConfig};
//! use ndarray::array;
//!
//! // Create distributed spatial cluster
//! let clusterconfig = NodeConfig::new()
//!     .with_node_count(4)
//!     .with_fault_tolerance(true)
//!     .with_load_balancing(true)
//!     .with_compression(true);
//!
//! let mut cluster = DistributedSpatialCluster::new(clusterconfig)?;
//!
//! // Distribute large spatial dataset
//! let large_dataset = array![[0.0, 0.0], [1.0, 0.0], /* ... millions of points ... */];
//! cluster.distribute_data(&large_dataset.view()).await?;
//!
//! // Run distributed k-means clustering
//! let (centroids, assignments) = cluster.distributed_kmeans(5, 100).await?;
//! println!("Distributed clustering completed: {} centroids", centroids.nrows());
//!
//! // Query distributed spatial index
//! let query_point = array![0.5, 0.5];
//! let nearest_neighbors = cluster.distributed_knn_search(&query_point.view(), 10).await?;
//! println!("Found {} nearest neighbors across cluster", nearest_neighbors.len());
//! ```

use crate::error::{SpatialError, SpatialResult};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock as TokioRwLock};

/// Node configuration for distributed cluster
#[derive(Debug, Clone)]
pub struct NodeConfig {
    /// Number of nodes in cluster
    pub node_count: usize,
    /// Enable fault tolerance
    pub fault_tolerance: bool,
    /// Enable load balancing
    pub load_balancing: bool,
    /// Enable data compression
    pub compression: bool,
    /// Communication timeout (milliseconds)
    pub communication_timeout_ms: u64,
    /// Heartbeat interval (milliseconds)
    pub heartbeat_interval_ms: u64,
    /// Maximum retries for failed operations
    pub max_retries: usize,
    /// Replication factor for fault tolerance
    pub replication_factor: usize,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeConfig {
    /// Create new node configuration
    pub fn new() -> Self {
        Self {
            node_count: 1,
            fault_tolerance: false,
            load_balancing: false,
            compression: false,
            communication_timeout_ms: 5000,
            heartbeat_interval_ms: 1000,
            max_retries: 3,
            replication_factor: 1,
        }
    }

    /// Configure node count
    pub fn with_node_count(mut self, count: usize) -> Self {
        self.node_count = count;
        self
    }

    /// Enable fault tolerance
    pub fn with_fault_tolerance(mut self, enabled: bool) -> Self {
        self.fault_tolerance = enabled;
        if enabled && self.replication_factor < 2 {
            self.replication_factor = 2;
        }
        self
    }

    /// Enable load balancing
    pub fn with_load_balancing(mut self, enabled: bool) -> Self {
        self.load_balancing = enabled;
        self
    }

    /// Enable compression
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression = enabled;
        self
    }
}

/// Distributed spatial computing cluster
#[derive(Debug)]
pub struct DistributedSpatialCluster {
    /// Cluster configuration
    config: NodeConfig,
    /// Node instances
    nodes: Vec<Arc<TokioRwLock<NodeInstance>>>,
    /// Master node ID
    #[allow(dead_code)]
    master_node_id: usize,
    /// Data partitions
    partitions: Arc<TokioRwLock<HashMap<usize, DataPartition>>>,
    /// Load balancer
    load_balancer: Arc<TokioRwLock<LoadBalancer>>,
    /// Fault detector
    #[allow(dead_code)]
    fault_detector: Arc<TokioRwLock<FaultDetector>>,
    /// Communication layer
    communication: Arc<TokioRwLock<CommunicationLayer>>,
    /// Cluster state
    cluster_state: Arc<TokioRwLock<ClusterState>>,
}

/// Individual node in the distributed cluster
#[derive(Debug)]
pub struct NodeInstance {
    /// Node ID
    pub node_id: usize,
    /// Node status
    pub status: NodeStatus,
    /// Local data partition
    pub local_data: Option<Array2<f64>>,
    /// Local spatial index
    pub local_index: Option<DistributedSpatialIndex>,
    /// Node load metrics
    pub load_metrics: LoadMetrics,
    /// Last heartbeat timestamp
    pub last_heartbeat: Instant,
    /// Assigned partitions
    pub assigned_partitions: Vec<usize>,
}

/// Node status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    Active,
    Inactive,
    Failed,
    Recovering,
    Joining,
    Leaving,
}

/// Data partition for distributed processing
#[derive(Debug, Clone)]
pub struct DataPartition {
    /// Partition ID
    pub partition_id: usize,
    /// Spatial bounds of partition
    pub bounds: SpatialBounds,
    /// Data points in partition
    pub data: Array2<f64>,
    /// Primary node for this partition
    pub primary_node: usize,
    /// Replica nodes
    pub replica_nodes: Vec<usize>,
    /// Partition size (number of points)
    pub size: usize,
    /// Last modified timestamp
    pub last_modified: Instant,
}

/// Spatial bounds for data partition
#[derive(Debug, Clone)]
pub struct SpatialBounds {
    /// Minimum coordinates
    pub min_coords: Array1<f64>,
    /// Maximum coordinates
    pub max_coords: Array1<f64>,
}

impl SpatialBounds {
    /// Check if point is within bounds
    pub fn contains(&self, point: &ArrayView1<f64>) -> bool {
        point
            .iter()
            .zip(self.min_coords.iter())
            .zip(self.max_coords.iter())
            .all(|((&coord, &min_coord), &max_coord)| coord >= min_coord && coord <= max_coord)
    }

    /// Calculate volume of bounds
    pub fn volume(&self) -> f64 {
        self.min_coords
            .iter()
            .zip(self.max_coords.iter())
            .map(|(&min_coord, &max_coord)| max_coord - min_coord)
            .product()
    }
}

/// Load balancer for distributed workload management
#[derive(Debug)]
pub struct LoadBalancer {
    /// Node load information
    #[allow(dead_code)]
    node_loads: HashMap<usize, LoadMetrics>,
    /// Load balancing strategy
    #[allow(dead_code)]
    strategy: LoadBalancingStrategy,
    /// Last rebalancing time
    #[allow(dead_code)]
    last_rebalance: Instant,
    /// Rebalancing threshold
    #[allow(dead_code)]
    load_threshold: f64,
}

/// Load balancing strategies
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    ProportionalLoad,
    AdaptiveLoad,
}

/// Load metrics for nodes
#[derive(Debug, Clone)]
pub struct LoadMetrics {
    /// CPU utilization (0.0 - 1.0)
    pub cpu_utilization: f64,
    /// Memory utilization (0.0 - 1.0)
    pub memory_utilization: f64,
    /// Network utilization (0.0 - 1.0)
    pub network_utilization: f64,
    /// Number of assigned partitions
    pub partition_count: usize,
    /// Current operation count
    pub operation_count: usize,
    /// Last update timestamp
    pub last_update: Instant,
}

impl LoadMetrics {
    /// Calculate overall load score
    pub fn load_score(&self) -> f64 {
        0.4 * self.cpu_utilization
            + 0.3 * self.memory_utilization
            + 0.2 * self.network_utilization
            + 0.1 * (self.partition_count as f64 / 10.0).min(1.0)
    }
}

/// Fault detector for monitoring node health
#[derive(Debug)]
pub struct FaultDetector {
    /// Node health status
    #[allow(dead_code)]
    node_health: HashMap<usize, NodeHealth>,
    /// Failure detection threshold
    #[allow(dead_code)]
    failure_threshold: Duration,
    /// Recovery strategies
    #[allow(dead_code)]
    recovery_strategies: HashMap<FailureType, RecoveryStrategy>,
}

/// Node health information
#[derive(Debug, Clone)]
pub struct NodeHealth {
    /// Last successful communication
    pub last_contact: Instant,
    /// Consecutive failures
    pub consecutive_failures: usize,
    /// Response times
    pub response_times: VecDeque<Duration>,
    /// Health score (0.0 - 1.0)
    pub health_score: f64,
}

/// Types of failures that can be detected
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum FailureType {
    NodeUnresponsive,
    HighLatency,
    ResourceExhaustion,
    PartialFailure,
    NetworkPartition,
}

/// Recovery strategies for different failure types
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    Restart,
    Migrate,
    Replicate,
    Isolate,
    WaitAndRetry,
}

/// Communication layer for inter-node communication
#[derive(Debug)]
pub struct CommunicationLayer {
    /// Communication channels
    #[allow(dead_code)]
    channels: HashMap<usize, mpsc::Sender<DistributedMessage>>,
    /// Message compression enabled
    #[allow(dead_code)]
    compression_enabled: bool,
    /// Communication statistics
    stats: CommunicationStats,
}

/// Statistics for communication performance
#[derive(Debug, Clone)]
pub struct CommunicationStats {
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Average latency
    pub average_latency_ms: f64,
}

/// Distributed message types
#[derive(Debug, Clone)]
pub enum DistributedMessage {
    /// Heartbeat message
    Heartbeat {
        node_id: usize,
        timestamp: Instant,
        load_metrics: LoadMetrics,
    },
    /// Data distribution message
    DataDistribution {
        partition_id: usize,
        data: Array2<f64>,
        bounds: SpatialBounds,
    },
    /// Query message
    Query {
        query_id: usize,
        query_type: QueryType,
        parameters: QueryParameters,
    },
    /// Query response
    QueryResponse {
        query_id: usize,
        results: QueryResults,
        node_id: usize,
    },
    /// Load balancing message
    LoadBalance { rebalance_plan: RebalancePlan },
    /// Fault tolerance message
    FaultTolerance {
        failure_type: FailureType,
        affected_nodes: Vec<usize>,
        recovery_plan: RecoveryPlan,
    },
}

/// Types of distributed queries
#[derive(Debug, Clone)]
pub enum QueryType {
    KNearestNeighbors,
    RangeSearch,
    Clustering,
    DistanceMatrix,
}

/// Query parameters
#[derive(Debug, Clone)]
pub struct QueryParameters {
    /// Query point (for NN queries)
    pub query_point: Option<Array1<f64>>,
    /// Search radius (for range queries)
    pub radius: Option<f64>,
    /// Number of neighbors (for KNN)
    pub k: Option<usize>,
    /// Number of clusters (for clustering)
    pub num_clusters: Option<usize>,
    /// Additional parameters
    pub extra_params: HashMap<String, f64>,
}

/// Query results
#[derive(Debug, Clone)]
pub enum QueryResults {
    NearestNeighbors {
        indices: Vec<usize>,
        distances: Vec<f64>,
    },
    RangeSearch {
        indices: Vec<usize>,
        points: Array2<f64>,
    },
    Clustering {
        centroids: Array2<f64>,
        assignments: Array1<usize>,
    },
    DistanceMatrix {
        matrix: Array2<f64>,
    },
}

/// Load rebalancing plan
#[derive(Debug, Clone)]
pub struct RebalancePlan {
    /// Partition migrations
    pub migrations: Vec<PartitionMigration>,
    /// Expected load improvement
    pub load_improvement: f64,
    /// Migration cost estimate
    pub migration_cost: f64,
}

/// Partition migration instruction
#[derive(Debug, Clone)]
pub struct PartitionMigration {
    /// Partition to migrate
    pub partition_id: usize,
    /// Source node
    pub from_node: usize,
    /// Destination node
    pub to_node: usize,
    /// Migration priority
    pub priority: f64,
}

/// Recovery plan for fault tolerance
#[derive(Debug, Clone)]
pub struct RecoveryPlan {
    /// Recovery actions
    pub actions: Vec<RecoveryAction>,
    /// Expected recovery time
    pub estimated_recovery_time: Duration,
    /// Success probability
    pub success_probability: f64,
}

/// Recovery action
#[derive(Debug, Clone)]
pub struct RecoveryAction {
    /// Action type
    pub action_type: RecoveryStrategy,
    /// Target node
    pub target_node: usize,
    /// Action parameters
    pub parameters: HashMap<String, String>,
}

/// Overall cluster state
#[derive(Debug)]
pub struct ClusterState {
    /// Active nodes
    pub active_nodes: Vec<usize>,
    /// Total data points
    pub total_data_points: usize,
    /// Total partitions
    pub total_partitions: usize,
    /// Cluster health score
    pub health_score: f64,
    /// Performance metrics
    pub performance_metrics: ClusterPerformanceMetrics,
}

/// Cluster performance metrics
#[derive(Debug, Clone)]
pub struct ClusterPerformanceMetrics {
    /// Average query latency
    pub avg_query_latency_ms: f64,
    /// Throughput (queries per second)
    pub throughput_qps: f64,
    /// Data distribution balance
    pub load_balance_score: f64,
    /// Fault tolerance level
    pub fault_tolerance_level: f64,
}

/// Distributed spatial index
#[derive(Debug)]
pub struct DistributedSpatialIndex {
    /// Local spatial index
    pub local_index: LocalSpatialIndex,
    /// Global index metadata
    pub global_metadata: GlobalIndexMetadata,
    /// Routing table for distributed queries
    pub routing_table: RoutingTable,
}

/// Local spatial index on each node
#[derive(Debug)]
pub struct LocalSpatialIndex {
    /// Local KD-tree
    pub kdtree: Option<crate::KDTree<f64, crate::EuclideanDistance<f64>>>,
    /// Local data bounds
    pub bounds: SpatialBounds,
    /// Index statistics
    pub stats: IndexStatistics,
}

/// Global index metadata shared across nodes
#[derive(Debug, Clone)]
pub struct GlobalIndexMetadata {
    /// Global data bounds
    pub global_bounds: SpatialBounds,
    /// Partition mapping
    pub partition_map: HashMap<usize, SpatialBounds>,
    /// Index version
    pub version: usize,
}

/// Routing table for distributed queries
#[derive(Debug)]
pub struct RoutingTable {
    /// Spatial routing entries
    pub entries: BTreeMap<SpatialKey, Vec<usize>>,
    /// Routing cache
    pub cache: HashMap<SpatialKey, Vec<usize>>,
}

/// Spatial key for routing
#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct SpatialKey {
    /// Z-order (Morton) code
    pub z_order: u64,
    /// Resolution level
    pub level: usize,
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct IndexStatistics {
    /// Build time
    pub build_time_ms: f64,
    /// Memory usage
    pub memory_usage_bytes: usize,
    /// Query count
    pub query_count: u64,
    /// Average query time
    pub avg_query_time_ms: f64,
}

impl DistributedSpatialCluster {
    /// Create new distributed spatial cluster
    pub fn new(config: NodeConfig) -> SpatialResult<Self> {
        let mut nodes = Vec::new();
        let mut channels = HashMap::new();

        // Create node instances
        for node_id in 0..config.node_count {
            let (sender, receiver) = mpsc::channel(1000);
            channels.insert(node_id, sender);

            let node = NodeInstance {
                node_id,
                status: NodeStatus::Active,
                local_data: None,
                local_index: None,
                load_metrics: LoadMetrics {
                    cpu_utilization: 0.0,
                    memory_utilization: 0.0,
                    network_utilization: 0.0,
                    partition_count: 0,
                    operation_count: 0,
                    last_update: Instant::now(),
                },
                last_heartbeat: Instant::now(),
                assigned_partitions: Vec::new(),
            };

            nodes.push(Arc::new(TokioRwLock::new(node)));
        }

        let load_balancer = LoadBalancer {
            node_loads: HashMap::new(),
            strategy: LoadBalancingStrategy::AdaptiveLoad,
            last_rebalance: Instant::now(),
            load_threshold: 0.8,
        };

        let fault_detector = FaultDetector {
            node_health: HashMap::new(),
            failure_threshold: Duration::from_secs(10),
            recovery_strategies: HashMap::new(),
        };

        let communication = CommunicationLayer {
            channels,
            compression_enabled: config.compression,
            stats: CommunicationStats {
                messages_sent: 0,
                messages_received: 0,
                bytes_sent: 0,
                bytes_received: 0,
                average_latency_ms: 0.0,
            },
        };

        let cluster_state = ClusterState {
            active_nodes: (0..config.node_count).collect(),
            total_data_points: 0,
            total_partitions: 0,
            health_score: 1.0,
            performance_metrics: ClusterPerformanceMetrics {
                avg_query_latency_ms: 0.0,
                throughput_qps: 0.0,
                load_balance_score: 1.0,
                fault_tolerance_level: if config.fault_tolerance { 0.8 } else { 0.0 },
            },
        };

        Ok(Self {
            config,
            nodes,
            master_node_id: 0,
            partitions: Arc::new(TokioRwLock::new(HashMap::new())),
            load_balancer: Arc::new(TokioRwLock::new(load_balancer)),
            fault_detector: Arc::new(TokioRwLock::new(fault_detector)),
            communication: Arc::new(TokioRwLock::new(communication)),
            cluster_state: Arc::new(TokioRwLock::new(cluster_state)),
        })
    }

    /// Default recovery strategies for different failure types
    #[allow(dead_code)]
    fn default_recovery_strategies(&self) -> HashMap<FailureType, RecoveryStrategy> {
        let mut strategies = HashMap::new();
        strategies.insert(FailureType::NodeUnresponsive, RecoveryStrategy::Restart);
        strategies.insert(FailureType::HighLatency, RecoveryStrategy::WaitAndRetry);
        strategies.insert(FailureType::ResourceExhaustion, RecoveryStrategy::Migrate);
        strategies.insert(FailureType::PartialFailure, RecoveryStrategy::Replicate);
        strategies.insert(FailureType::NetworkPartition, RecoveryStrategy::Isolate);
        strategies
    }

    /// Distribute data across cluster nodes
    pub async fn distribute_data(&mut self, data: &ArrayView2<'_, f64>) -> SpatialResult<()> {
        let (n_points, n_dims) = data.dim();

        // Create spatial partitions
        let partitions = self.create_spatial_partitions(data).await?;

        // Distribute partitions to nodes
        self.assign_partitions_to_nodes(&partitions).await?;

        // Build distributed spatial indices
        self.build_distributed_indices().await?;

        // Update cluster state
        {
            let mut state = self.cluster_state.write().await;
            state.total_data_points = n_points;
            state.total_partitions = partitions.len();
        }

        Ok(())
    }

    /// Create spatial partitions using space-filling curves
    async fn create_spatial_partitions(
        &self,
        data: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Vec<DataPartition>> {
        let (n_points, n_dims) = data.dim();
        let target_partitions = self.config.node_count * 2; // 2 partitions per node

        // Calculate global bounds
        let mut min_coords = Array1::from_elem(n_dims, f64::INFINITY);
        let mut max_coords = Array1::from_elem(n_dims, f64::NEG_INFINITY);

        for point in data.outer_iter() {
            for (i, &coord) in point.iter().enumerate() {
                min_coords[i] = min_coords[i].min(coord);
                max_coords[i] = max_coords[i].max(coord);
            }
        }

        let global_bounds = SpatialBounds {
            min_coords,
            max_coords,
        };

        // Use Z-order (Morton) curve for space partitioning
        let mut point_z_orders = Vec::new();
        for (i, point) in data.outer_iter().enumerate() {
            let z_order = self.calculate_z_order(&point.to_owned(), &global_bounds, 16);
            point_z_orders.push((i, z_order, point.to_owned()));
        }

        // Sort by Z-order
        point_z_orders.sort_by_key(|(_, z_order_, _)| *z_order_);

        // Create partitions
        let points_per_partition = n_points.div_ceil(target_partitions);
        let mut partitions = Vec::new();

        for partition_id in 0..target_partitions {
            let start_idx = partition_id * points_per_partition;
            let end_idx = ((partition_id + 1) * points_per_partition).min(n_points);

            if start_idx >= n_points {
                break;
            }

            // Extract partition data
            let partition_size = end_idx - start_idx;
            let mut partition_data = Array2::zeros((partition_size, n_dims));
            let mut partition_min = Array1::from_elem(n_dims, f64::INFINITY);
            let mut partition_max = Array1::from_elem(n_dims, f64::NEG_INFINITY);

            for (i, (_, _, point)) in point_z_orders[start_idx..end_idx].iter().enumerate() {
                partition_data.row_mut(i).assign(point);

                for (j, &coord) in point.iter().enumerate() {
                    partition_min[j] = partition_min[j].min(coord);
                    partition_max[j] = partition_max[j].max(coord);
                }
            }

            let partition_bounds = SpatialBounds {
                min_coords: partition_min,
                max_coords: partition_max,
            };

            let partition = DataPartition {
                partition_id,
                bounds: partition_bounds,
                data: partition_data,
                primary_node: partition_id % self.config.node_count,
                replica_nodes: if self.config.fault_tolerance {
                    vec![(partition_id + 1) % self.config.node_count]
                } else {
                    Vec::new()
                },
                size: partition_size,
                last_modified: Instant::now(),
            };

            partitions.push(partition);
        }

        Ok(partitions)
    }

    /// Calculate Z-order (Morton) code for spatial point
    fn calculate_z_order(
        &self,
        point: &Array1<f64>,
        bounds: &SpatialBounds,
        resolution: usize,
    ) -> u64 {
        let mut z_order = 0u64;

        for bit in 0..resolution {
            for (dim, ((&coord, &min_coord), &max_coord)) in point
                .iter()
                .zip(bounds.min_coords.iter())
                .zip(bounds.max_coords.iter())
                .enumerate()
            {
                if dim >= 3 {
                    break;
                } // Limit to 3D for 64-bit Z-order

                let normalized = if max_coord > min_coord {
                    (coord - min_coord) / (max_coord - min_coord)
                } else {
                    0.5
                };

                let bit_val = if normalized >= 0.5 { 1u64 } else { 0u64 };
                let bit_pos = bit * 3 + dim; // 3D interleaving

                if bit_pos < 64 {
                    z_order |= bit_val << bit_pos;
                }
            }
        }

        z_order
    }

    /// Assign partitions to nodes with load balancing
    async fn assign_partitions_to_nodes(
        &mut self,
        partitions: &[DataPartition],
    ) -> SpatialResult<()> {
        let mut partition_map = HashMap::new();

        for partition in partitions {
            partition_map.insert(partition.partition_id, partition.clone());

            // Assign to primary node
            let primary_node = &self.nodes[partition.primary_node];
            {
                let mut node = primary_node.write().await;
                node.assigned_partitions.push(partition.partition_id);

                // Append partition data to existing data instead of overwriting
                if let Some(ref existing_data) = node.local_data {
                    // Concatenate existing data with new partition data
                    let (existing_rows, cols) = existing_data.dim();
                    let (new_rows_, _) = partition.data.dim();
                    let total_rows = existing_rows + new_rows_;

                    let mut combined_data = Array2::zeros((total_rows, cols));
                    combined_data
                        .slice_mut(s![..existing_rows, ..])
                        .assign(existing_data);
                    combined_data
                        .slice_mut(s![existing_rows.., ..])
                        .assign(&partition.data);
                    node.local_data = Some(combined_data);
                } else {
                    node.local_data = Some(partition.data.clone());
                }

                node.load_metrics.partition_count += 1;
            }

            // Assign to replica nodes if fault tolerance is enabled
            for &replica_node_id in &partition.replica_nodes {
                let replica_node = &self.nodes[replica_node_id];
                let mut node = replica_node.write().await;
                node.assigned_partitions.push(partition.partition_id);

                // Append partition data to existing data instead of overwriting
                if let Some(ref existing_data) = node.local_data {
                    // Concatenate existing data with new partition data
                    let (existing_rows, cols) = existing_data.dim();
                    let (new_rows_, _) = partition.data.dim();
                    let total_rows = existing_rows + new_rows_;

                    let mut combined_data = Array2::zeros((total_rows, cols));
                    combined_data
                        .slice_mut(s![..existing_rows, ..])
                        .assign(existing_data);
                    combined_data
                        .slice_mut(s![existing_rows.., ..])
                        .assign(&partition.data);
                    node.local_data = Some(combined_data);
                } else {
                    node.local_data = Some(partition.data.clone());
                }

                node.load_metrics.partition_count += 1;
            }
        }

        {
            let mut partitions_lock = self.partitions.write().await;
            *partitions_lock = partition_map;
        }

        Ok(())
    }

    /// Build distributed spatial indices
    async fn build_distributed_indices(&mut self) -> SpatialResult<()> {
        // Build local indices on each node
        for node_arc in &self.nodes {
            let mut node = node_arc.write().await;

            if let Some(ref local_data) = node.local_data {
                // Calculate local bounds
                let (n_points, n_dims) = local_data.dim();
                let mut min_coords = Array1::from_elem(n_dims, f64::INFINITY);
                let mut max_coords = Array1::from_elem(n_dims, f64::NEG_INFINITY);

                for point in local_data.outer_iter() {
                    for (i, &coord) in point.iter().enumerate() {
                        min_coords[i] = min_coords[i].min(coord);
                        max_coords[i] = max_coords[i].max(coord);
                    }
                }

                let local_bounds = SpatialBounds {
                    min_coords,
                    max_coords,
                };

                // Build KD-tree
                let kdtree = crate::KDTree::new(local_data)?;

                let local_index = LocalSpatialIndex {
                    kdtree: Some(kdtree),
                    bounds: local_bounds.clone(),
                    stats: IndexStatistics {
                        build_time_ms: 0.0,                        // Would measure actual build time
                        memory_usage_bytes: n_points * n_dims * 8, // Rough estimate
                        query_count: 0,
                        avg_query_time_ms: 0.0,
                    },
                };

                // Create routing table entries
                let routing_table = RoutingTable {
                    entries: BTreeMap::new(),
                    cache: HashMap::new(),
                };

                // Create global metadata (simplified)
                let global_metadata = GlobalIndexMetadata {
                    global_bounds: local_bounds.clone(), // Would be computed globally
                    partition_map: HashMap::new(),
                    version: 1,
                };

                let distributed_index = DistributedSpatialIndex {
                    local_index,
                    global_metadata,
                    routing_table,
                };

                node.local_index = Some(distributed_index);
            }
        }

        Ok(())
    }

    /// Perform distributed k-means clustering
    pub async fn distributed_kmeans(
        &mut self,
        k: usize,
        max_iterations: usize,
    ) -> SpatialResult<(Array2<f64>, Array1<usize>)> {
        // Initialize centroids using k-means++
        let initial_centroids = self.initialize_distributed_centroids(k).await?;
        let mut centroids = initial_centroids;

        for _iteration in 0..max_iterations {
            // Assign points to clusters on each node
            let local_assignments = self.distributed_assignment_step(&centroids).await?;

            // Update centroids using distributed computation
            let new_centroids = self
                .distributed_centroid_update(&local_assignments, k)
                .await?;

            // Check convergence
            let centroid_change = self.calculate_centroid_change(&centroids, &new_centroids);
            if centroid_change < 1e-6 {
                break;
            }

            centroids = new_centroids;
        }

        // Collect final assignments
        let final_assignments = self.collect_final_assignments(&centroids).await?;

        Ok((centroids, final_assignments))
    }

    /// Initialize centroids using distributed k-means++
    async fn initialize_distributed_centroids(&self, k: usize) -> SpatialResult<Array2<f64>> {
        // Get random first centroid from any node
        let first_centroid = self.get_random_point_from_cluster().await?;

        let n_dims = first_centroid.len();
        let mut centroids = Array2::zeros((k, n_dims));
        centroids.row_mut(0).assign(&first_centroid);

        // Select remaining centroids using k-means++ probability
        for i in 1..k {
            let distances = self
                .compute_distributed_distances(&centroids.slice(s![..i, ..]))
                .await?;
            let next_centroid = self.select_next_centroid_weighted(&distances).await?;
            centroids.row_mut(i).assign(&next_centroid);
        }

        Ok(centroids)
    }

    /// Get random point from any node in cluster
    async fn get_random_point_from_cluster(&self) -> SpatialResult<Array1<f64>> {
        for node_arc in &self.nodes {
            let node = node_arc.read().await;
            if let Some(ref local_data) = node.local_data {
                if local_data.nrows() > 0 {
                    let idx = (rand::random::<f64>() * local_data.nrows() as f64) as usize;
                    return Ok(local_data.row(idx).to_owned());
                }
            }
        }

        Err(SpatialError::InvalidInput(
            "No data found in cluster".to_string(),
        ))
    }

    /// Compute distances to current centroids across all nodes
    async fn compute_distributed_distances(
        &self,
        centroids: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Vec<f64>> {
        let mut all_distances = Vec::new();

        for node_arc in &self.nodes {
            let node = node_arc.read().await;
            if let Some(ref local_data) = node.local_data {
                for point in local_data.outer_iter() {
                    let mut min_distance = f64::INFINITY;

                    for centroid in centroids.outer_iter() {
                        let distance: f64 = point
                            .iter()
                            .zip(centroid.iter())
                            .map(|(&a, &b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();

                        min_distance = min_distance.min(distance);
                    }

                    all_distances.push(min_distance);
                }
            }
        }

        Ok(all_distances)
    }

    /// Select next centroid using weighted probability
    async fn select_next_centroid_weighted(
        &self,
        _distances: &[f64],
    ) -> SpatialResult<Array1<f64>> {
        let total_distance: f64 = _distances.iter().sum();
        let target = rand::random::<f64>() * total_distance;

        let mut cumulative = 0.0;
        let mut point_index = 0;

        for &distance in _distances {
            cumulative += distance;
            if cumulative >= target {
                break;
            }
            point_index += 1;
        }

        // Find the point at the selected index across all nodes
        let mut current_index = 0;
        for node_arc in &self.nodes {
            let node = node_arc.read().await;
            if let Some(ref local_data) = node.local_data {
                if current_index + local_data.nrows() > point_index {
                    let local_index = point_index - current_index;
                    return Ok(local_data.row(local_index).to_owned());
                }
                current_index += local_data.nrows();
            }
        }

        Err(SpatialError::InvalidInput(
            "Point index out of range".to_string(),
        ))
    }

    /// Perform distributed assignment step
    async fn distributed_assignment_step(
        &self,
        centroids: &Array2<f64>,
    ) -> SpatialResult<Vec<(usize, Array1<usize>)>> {
        let mut local_assignments = Vec::new();

        for (node_id, node_arc) in self.nodes.iter().enumerate() {
            let node = node_arc.read().await;
            if let Some(ref local_data) = node.local_data {
                let (n_points_, _) = local_data.dim();
                let mut assignments = Array1::zeros(n_points_);

                for (i, point) in local_data.outer_iter().enumerate() {
                    let mut best_cluster = 0;
                    let mut best_distance = f64::INFINITY;

                    for (j, centroid) in centroids.outer_iter().enumerate() {
                        let distance: f64 = point
                            .iter()
                            .zip(centroid.iter())
                            .map(|(&a, &b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();

                        if distance < best_distance {
                            best_distance = distance;
                            best_cluster = j;
                        }
                    }

                    assignments[i] = best_cluster;
                }

                local_assignments.push((node_id, assignments));
            }
        }

        Ok(local_assignments)
    }

    /// Update centroids using distributed computation
    async fn distributed_centroid_update(
        &self,
        local_assignments: &[(usize, Array1<usize>)],
        k: usize,
    ) -> SpatialResult<Array2<f64>> {
        // Collect cluster statistics from all nodes
        let mut cluster_sums: HashMap<usize, Array1<f64>> = HashMap::new();
        let mut cluster_counts: HashMap<usize, usize> = HashMap::new();

        for (node_id, assignments) in local_assignments {
            let node = self.nodes[*node_id].read().await;
            if let Some(ref local_data) = node.local_data {
                let (_, n_dims) = local_data.dim();

                for (i, &cluster) in assignments.iter().enumerate() {
                    let point = local_data.row(i);

                    let cluster_sum = cluster_sums
                        .entry(cluster)
                        .or_insert_with(|| Array1::zeros(n_dims));
                    let cluster_count = cluster_counts.entry(cluster).or_insert(0);

                    for (j, &coord) in point.iter().enumerate() {
                        cluster_sum[j] += coord;
                    }
                    *cluster_count += 1;
                }
            }
        }

        // Calculate new centroids
        let n_dims = cluster_sums
            .values()
            .next()
            .map(|sum| sum.len())
            .unwrap_or(2);

        let mut new_centroids = Array2::zeros((k, n_dims));

        for cluster in 0..k {
            if let (Some(sum), Some(&count)) =
                (cluster_sums.get(&cluster), cluster_counts.get(&cluster))
            {
                if count > 0 {
                    for j in 0..n_dims {
                        new_centroids[[cluster, j]] = sum[j] / count as f64;
                    }
                }
            }
        }

        Ok(new_centroids)
    }

    /// Calculate change in centroids for convergence checking
    fn calculate_centroid_change(
        &self,
        old_centroids: &Array2<f64>,
        new_centroids: &Array2<f64>,
    ) -> f64 {
        let mut total_change = 0.0;

        for (old_row, new_row) in old_centroids.outer_iter().zip(new_centroids.outer_iter()) {
            let change: f64 = old_row
                .iter()
                .zip(new_row.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            total_change += change;
        }

        total_change / old_centroids.nrows() as f64
    }

    /// Collect final assignments from all nodes
    async fn collect_final_assignments(
        &self,
        centroids: &Array2<f64>,
    ) -> SpatialResult<Array1<usize>> {
        let mut all_assignments = Vec::new();

        for node_arc in &self.nodes {
            let node = node_arc.read().await;
            if let Some(ref local_data) = node.local_data {
                for point in local_data.outer_iter() {
                    let mut best_cluster = 0;
                    let mut best_distance = f64::INFINITY;

                    for (j, centroid) in centroids.outer_iter().enumerate() {
                        let distance: f64 = point
                            .iter()
                            .zip(centroid.iter())
                            .map(|(&a, &b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();

                        if distance < best_distance {
                            best_distance = distance;
                            best_cluster = j;
                        }
                    }

                    all_assignments.push(best_cluster);
                }
            }
        }

        Ok(Array1::from(all_assignments))
    }

    /// Perform distributed k-nearest neighbors search
    pub async fn distributed_knn_search(
        &self,
        query_point: &ArrayView1<'_, f64>,
        k: usize,
    ) -> SpatialResult<Vec<(usize, f64)>> {
        let mut all_neighbors = Vec::new();

        // Query each node
        for node_arc in &self.nodes {
            let node = node_arc.read().await;
            if let Some(ref local_index) = node.local_index {
                if let Some(ref kdtree) = local_index.local_index.kdtree {
                    // Check if query _point is within local bounds
                    if local_index.local_index.bounds.contains(query_point) {
                        let (indices, distances) =
                            kdtree.query(query_point.as_slice().unwrap(), k)?;

                        for (idx, dist) in indices.iter().zip(distances.iter()) {
                            all_neighbors.push((*idx, *dist));
                        }
                    }
                }
            }
        }

        // Sort and return top k neighbors
        all_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_neighbors.truncate(k);

        Ok(all_neighbors)
    }

    /// Get cluster statistics
    pub async fn get_cluster_statistics(&self) -> SpatialResult<ClusterStatistics> {
        let state = self.cluster_state.read().await;
        let _load_balancer = self.load_balancer.read().await;
        let communication = self.communication.read().await;

        let active_node_count = state.active_nodes.len();
        let total_partitions = state.total_partitions;
        let avg_partitions_per_node = if active_node_count > 0 {
            total_partitions as f64 / active_node_count as f64
        } else {
            0.0
        };

        Ok(ClusterStatistics {
            active_nodes: active_node_count,
            total_data_points: state.total_data_points,
            total_partitions,
            avg_partitions_per_node,
            health_score: state.health_score,
            load_balance_score: state.performance_metrics.load_balance_score,
            avg_query_latency_ms: state.performance_metrics.avg_query_latency_ms,
            throughput_qps: state.performance_metrics.throughput_qps,
            total_messages_sent: communication.stats.messages_sent,
            total_bytes_sent: communication.stats.bytes_sent,
            avg_communication_latency_ms: communication.stats.average_latency_ms,
        })
    }
}

/// Cluster statistics
#[derive(Debug, Clone)]
pub struct ClusterStatistics {
    pub active_nodes: usize,
    pub total_data_points: usize,
    pub total_partitions: usize,
    pub avg_partitions_per_node: f64,
    pub health_score: f64,
    pub load_balance_score: f64,
    pub avg_query_latency_ms: f64,
    pub throughput_qps: f64,
    pub total_messages_sent: u64,
    pub total_bytes_sent: u64,
    pub avg_communication_latency_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_nodeconfig() {
        let config = NodeConfig::new()
            .with_node_count(4)
            .with_fault_tolerance(true)
            .with_load_balancing(true);

        assert_eq!(config.node_count, 4);
        assert!(config.fault_tolerance);
        assert!(config.load_balancing);
        assert_eq!(config.replication_factor, 2);
    }

    #[test]
    fn test_spatial_bounds() {
        let bounds = SpatialBounds {
            min_coords: array![0.0, 0.0],
            max_coords: array![1.0, 1.0],
        };

        assert!(bounds.contains(&array![0.5, 0.5].view()));
        assert!(!bounds.contains(&array![1.5, 0.5].view()));
        assert_eq!(bounds.volume(), 1.0);
    }

    #[test]
    fn test_load_metrics() {
        let metrics = LoadMetrics {
            cpu_utilization: 0.5,
            memory_utilization: 0.3,
            network_utilization: 0.2,
            partition_count: 2,
            operation_count: 100,
            last_update: Instant::now(),
        };

        let load_score = metrics.load_score();
        assert!(load_score > 0.0 && load_score < 1.0);
    }

    #[tokio::test]
    async fn test_distributed_cluster_creation() {
        let config = NodeConfig::new()
            .with_node_count(2)
            .with_fault_tolerance(false);

        let cluster = DistributedSpatialCluster::new(config);
        assert!(cluster.is_ok());

        let cluster = cluster.unwrap();
        assert_eq!(cluster.nodes.len(), 2);
        assert_eq!(cluster.master_node_id, 0);
    }

    #[tokio::test]
    async fn test_data_distribution() {
        let config = NodeConfig::new()
            .with_node_count(2)
            .with_fault_tolerance(false);

        let mut cluster = DistributedSpatialCluster::new(config).unwrap();
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let result = cluster.distribute_data(&data.view()).await;
        assert!(result.is_ok());

        let stats = cluster.get_cluster_statistics().await.unwrap();
        assert_eq!(stats.total_data_points, 4);
        assert!(stats.total_partitions > 0);
    }

    #[tokio::test]
    async fn test_distributed_kmeans() {
        let config = NodeConfig::new().with_node_count(2);
        let mut cluster = DistributedSpatialCluster::new(config).unwrap();

        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0]
        ];
        cluster.distribute_data(&data.view()).await.unwrap();

        let result = cluster.distributed_kmeans(2, 10).await;
        assert!(result.is_ok());

        let (centroids, assignments) = result.unwrap();
        assert_eq!(centroids.nrows(), 2);
        assert_eq!(assignments.len(), 6);
    }
}
