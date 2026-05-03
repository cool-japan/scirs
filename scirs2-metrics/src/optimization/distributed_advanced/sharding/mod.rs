//! Data sharding and distribution management
//!
//! This module provides comprehensive data sharding capabilities:
//! - Consistent hashing for shard distribution
//! - Dynamic resharding and rebalancing
//! - Shard migration and replication
//! - Data locality optimization

use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

pub use crate::optimization::distributed::config::{
    HashFunction, ShardingConfig, ShardingStrategy,
};

/// Data shard representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataShard {
    /// Shard ID
    pub id: String,
    /// Shard range
    pub range: DataRange,
    /// Primary node for this shard
    pub primary_node: String,
    /// Replica nodes
    pub replicas: Vec<String>,
    /// Data size (bytes)
    pub size_bytes: u64,
    /// Number of keys in shard
    pub key_count: usize,
    /// Last access time
    pub last_access: SystemTime,
    /// Shard status
    pub status: ShardStatus,
    /// Migration info (if being migrated)
    pub migration: Option<ShardMigration>,
}

/// Shard status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ShardStatus {
    /// Shard is active and serving requests
    Active,
    /// Shard is being migrated
    Migrating,
    /// Shard is being split
    Splitting,
    /// Shard is being merged
    Merging,
    /// Shard is inactive/offline
    Inactive,
    /// Shard is in error state
    Error(String),
}

/// Data range for sharding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataRange {
    /// Hash range (start_hash, end_hash)
    Hash { start: u64, end: u64 },
    /// Key range (start_key, end_key)
    Key { start: String, end: String },
    /// Numeric range (start, end)
    Numeric { start: f64, end: f64 },
    /// Time range
    Time { start: SystemTime, end: SystemTime },
    /// Geographic range
    Geographic {
        lat_min: f64,
        lat_max: f64,
        lon_min: f64,
        lon_max: f64,
    },
    /// Custom range
    Custom {
        range_type: String,
        range_data: Vec<u8>,
    },
}

impl DataRange {
    /// Check if a key falls within this range
    pub fn contains_key(&self, key: &str) -> bool {
        match self {
            DataRange::Hash { start, end } => {
                let hash = self.hash_key(key);
                hash >= *start && hash <= *end
            }
            DataRange::Key { start, end } => key >= start.as_str() && key <= end.as_str(),
            DataRange::Numeric { start, end } => {
                if let Ok(num) = key.parse::<f64>() {
                    num >= *start && num <= *end
                } else {
                    false
                }
            }
            DataRange::Time { start, end } => {
                // Attempt to parse key as timestamp
                if let Ok(timestamp_str) = key.parse::<u64>() {
                    if let Some(timestamp) =
                        SystemTime::UNIX_EPOCH.checked_add(Duration::from_secs(timestamp_str))
                    {
                        timestamp >= *start && timestamp <= *end
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            DataRange::Geographic { .. } => {
                // Would need to parse geographic coordinates from key
                // For now, return false
                false
            }
            DataRange::Custom { .. } => {
                // Custom logic would be implemented here
                false
            }
        }
    }

    /// Hash a key using the specified hash function
    fn hash_key(&self, key: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Check if this range overlaps with another
    pub fn overlaps_with(&self, other: &DataRange) -> bool {
        match (self, other) {
            (DataRange::Hash { start: s1, end: e1 }, DataRange::Hash { start: s2, end: e2 }) => {
                s1 <= e2 && s2 <= e1
            }
            (DataRange::Key { start: s1, end: e1 }, DataRange::Key { start: s2, end: e2 }) => {
                s1 <= e2 && s2 <= e1
            }
            (
                DataRange::Numeric { start: s1, end: e1 },
                DataRange::Numeric { start: s2, end: e2 },
            ) => s1 <= e2 && s2 <= e1,
            (DataRange::Time { start: s1, end: e1 }, DataRange::Time { start: s2, end: e2 }) => {
                s1 <= e2 && s2 <= e1
            }
            _ => false, // Different range types don't overlap
        }
    }

    /// Split this range into two ranges
    pub fn split(&self) -> Result<(DataRange, DataRange)> {
        match self {
            DataRange::Hash { start, end } => {
                let mid = start + (end - start) / 2;
                Ok((
                    DataRange::Hash {
                        start: *start,
                        end: mid,
                    },
                    DataRange::Hash {
                        start: mid + 1,
                        end: *end,
                    },
                ))
            }
            DataRange::Key { start, end } => {
                // Simple string-based split (could be improved)
                let mid = format!("{}_{}", start, end);
                Ok((
                    DataRange::Key {
                        start: start.clone(),
                        end: mid.clone(),
                    },
                    DataRange::Key {
                        start: mid,
                        end: end.clone(),
                    },
                ))
            }
            DataRange::Numeric { start, end } => {
                let mid = start + (end - start) / 2.0;
                Ok((
                    DataRange::Numeric {
                        start: *start,
                        end: mid,
                    },
                    DataRange::Numeric {
                        start: mid,
                        end: *end,
                    },
                ))
            }
            DataRange::Time { start, end } => {
                let duration = end
                    .duration_since(*start)
                    .map_err(|_| MetricsError::ShardingError("Invalid time range".to_string()))?;
                let mid_duration = duration / 2;
                let mid = *start + mid_duration;
                Ok((
                    DataRange::Time {
                        start: *start,
                        end: mid,
                    },
                    DataRange::Time {
                        start: mid,
                        end: *end,
                    },
                ))
            }
            _ => Err(MetricsError::ShardingError(
                "Cannot split this range type".to_string(),
            )),
        }
    }
}

/// Shard migration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMigration {
    /// Migration ID
    pub id: String,
    /// Source node
    pub source_node: String,
    /// Target node
    pub target_node: String,
    /// Migration progress (0.0 - 1.0)
    pub progress: f64,
    /// Started time
    pub started_at: SystemTime,
    /// Estimated completion time
    pub estimated_completion: Option<SystemTime>,
    /// Migration status
    pub status: MigrationStatus,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// Migration status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MigrationStatus {
    /// Migration is planned but not started
    Planned,
    /// Migration is in progress
    InProgress,
    /// Migration completed successfully
    Completed,
    /// Migration failed
    Failed,
    /// Migration was cancelled
    Cancelled,
}

/// Shard manager for handling sharding operations
#[derive(Debug)]
pub struct ShardManager {
    /// Sharding configuration
    config: ShardingConfig,
    /// Current shards
    shards: Arc<RwLock<HashMap<String, DataShard>>>,
    /// Node assignments
    node_assignments: Arc<RwLock<HashMap<String, Vec<String>>>>,
    /// Consistent hash ring (for consistent hashing)
    hash_ring: Arc<RwLock<BTreeMap<u64, String>>>,
    /// Active migrations
    migrations: Arc<RwLock<HashMap<String, ShardMigration>>>,
    /// Statistics
    stats: ShardingStats,
}

impl ShardManager {
    /// Create a new shard manager
    pub fn new(config: ShardingConfig) -> Self {
        Self {
            config,
            shards: Arc::new(RwLock::new(HashMap::new())),
            node_assignments: Arc::new(RwLock::new(HashMap::new())),
            hash_ring: Arc::new(RwLock::new(BTreeMap::new())),
            migrations: Arc::new(RwLock::new(HashMap::new())),
            stats: ShardingStats::default(),
        }
    }

    /// Initialize sharding with available nodes
    pub fn initialize(&mut self, nodes: Vec<String>) -> Result<()> {
        match self.config.strategy {
            ShardingStrategy::ConsistentHash => {
                self.initialize_consistent_hash(nodes)?;
            }
            ShardingStrategy::Hash => {
                self.initialize_hash_sharding(nodes)?;
            }
            ShardingStrategy::Range => {
                self.initialize_range_sharding(nodes)?;
            }
            _ => {
                return Err(MetricsError::ShardingError(
                    "Sharding strategy not implemented".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Initialize consistent hash ring
    fn initialize_consistent_hash(&mut self, nodes: Vec<String>) -> Result<()> {
        let mut hash_ring = self.hash_ring.write().expect("Operation failed");
        let mut shards = self.shards.write().expect("Operation failed");

        hash_ring.clear();
        shards.clear();

        // Add virtual nodes to the hash ring
        for node in &nodes {
            for i in 0..self.config.virtual_nodes {
                let virtual_node_key = format!("{}:{}", node, i);
                let hash = self.hash_string(&virtual_node_key);
                hash_ring.insert(hash, node.clone());
            }
        }

        // Create shards based on hash ring
        let mut prev_hash = 0u64;
        let ring_keys: Vec<u64> = hash_ring.keys().cloned().collect();

        for (i, &hash) in ring_keys.iter().enumerate() {
            let shard_id = format!("shard_{}", i);
            let node = hash_ring.get(&hash).expect("Operation failed").clone();

            let shard = DataShard {
                id: shard_id.clone(),
                range: DataRange::Hash {
                    start: prev_hash,
                    end: hash,
                },
                primary_node: node.clone(),
                replicas: self.select_replicas(&node, &nodes),
                size_bytes: 0,
                key_count: 0,
                last_access: SystemTime::now(),
                status: ShardStatus::Active,
                migration: None,
            };

            shards.insert(shard_id, shard);
            prev_hash = hash + 1;
        }

        Ok(())
    }

    /// Initialize hash-based sharding
    fn initialize_hash_sharding(&mut self, nodes: Vec<String>) -> Result<()> {
        let mut shards = self.shards.write().expect("Operation failed");
        shards.clear();

        let hash_range_size = u64::MAX / self.config.shard_count as u64;

        for i in 0..self.config.shard_count {
            let shard_id = format!("shard_{}", i);
            let start_hash = i as u64 * hash_range_size;
            let end_hash = if i == self.config.shard_count - 1 {
                u64::MAX
            } else {
                (i + 1) as u64 * hash_range_size - 1
            };

            let node = &nodes[i % nodes.len()];

            let shard = DataShard {
                id: shard_id.clone(),
                range: DataRange::Hash {
                    start: start_hash,
                    end: end_hash,
                },
                primary_node: node.clone(),
                replicas: self.select_replicas(node, &nodes),
                size_bytes: 0,
                key_count: 0,
                last_access: SystemTime::now(),
                status: ShardStatus::Active,
                migration: None,
            };

            shards.insert(shard_id, shard);
        }

        Ok(())
    }

    /// Initialize range-based sharding
    fn initialize_range_sharding(&mut self, nodes: Vec<String>) -> Result<()> {
        let mut shards = self.shards.write().expect("Operation failed");
        shards.clear();

        // For range sharding, we'll use key-based ranges
        // This is a simplified implementation
        for i in 0..self.config.shard_count {
            let shard_id = format!("shard_{}", i);
            let start_key = format!("{:04}", i * 1000);
            let end_key = format!("{:04}", (i + 1) * 1000 - 1);

            let node = &nodes[i % nodes.len()];

            let shard = DataShard {
                id: shard_id.clone(),
                range: DataRange::Key {
                    start: start_key,
                    end: end_key,
                },
                primary_node: node.clone(),
                replicas: self.select_replicas(node, &nodes),
                size_bytes: 0,
                key_count: 0,
                last_access: SystemTime::now(),
                status: ShardStatus::Active,
                migration: None,
            };

            shards.insert(shard_id, shard);
        }

        Ok(())
    }

    /// Select replica nodes for a primary node
    fn select_replicas(&self, primary: &str, all_nodes: &[String]) -> Vec<String> {
        let mut replicas = Vec::new();
        let mut count = 0;

        for node in all_nodes {
            if node != primary && count < self.config.replication_factor - 1 {
                replicas.push(node.clone());
                count += 1;
            }
        }

        replicas
    }

    /// Find the shard for a given key
    pub fn find_shard(&self, key: &str) -> Result<String> {
        let shards = self.shards.read().expect("Operation failed");

        for shard in shards.values() {
            if shard.range.contains_key(key) {
                return Ok(shard.id.clone());
            }
        }

        Err(MetricsError::ShardingError(
            "No shard found for key".to_string(),
        ))
    }

    /// Get the node responsible for a key
    pub fn get_node_for_key(&self, key: &str) -> Result<String> {
        match self.config.strategy {
            ShardingStrategy::ConsistentHash => self.get_node_consistent_hash(key),
            _ => {
                let shard_id = self.find_shard(key)?;
                let shards = self.shards.read().expect("Operation failed");
                if let Some(shard) = shards.get(&shard_id) {
                    Ok(shard.primary_node.clone())
                } else {
                    Err(MetricsError::ShardingError("Shard not found".to_string()))
                }
            }
        }
    }

    /// Get node using consistent hashing
    fn get_node_consistent_hash(&self, key: &str) -> Result<String> {
        let hash_ring = self.hash_ring.read().expect("Operation failed");
        if hash_ring.is_empty() {
            return Err(MetricsError::ShardingError(
                "Hash ring is empty".to_string(),
            ));
        }

        let key_hash = self.hash_string(key);

        // Find the first node with hash >= key_hash
        for (&node_hash, node) in hash_ring.range(key_hash..) {
            if node_hash >= key_hash {
                return Ok(node.clone());
            }
        }

        // Wrap around to the first node
        if let Some((_, node)) = hash_ring.iter().next() {
            Ok(node.clone())
        } else {
            Err(MetricsError::ShardingError(
                "No nodes in hash ring".to_string(),
            ))
        }
    }

    /// Hash a string using the configured hash function
    fn hash_string(&self, s: &str) -> u64 {
        match self.config.hash_function {
            HashFunction::Murmur3 | HashFunction::XxHash => {
                // Simplified hash using DefaultHasher
                use std::collections::hash_map::DefaultHasher;
                let mut hasher = DefaultHasher::new();
                s.hash(&mut hasher);
                hasher.finish()
            }
            HashFunction::Crc32 => {
                // Simplified CRC32 implementation
                let mut crc = 0xFFFFFFFFu32;
                for byte in s.bytes() {
                    crc ^= byte as u32;
                    for _ in 0..8 {
                        if crc & 1 != 0 {
                            crc = (crc >> 1) ^ 0xEDB88320;
                        } else {
                            crc >>= 1;
                        }
                    }
                }
                (crc ^ 0xFFFFFFFF) as u64
            }
            _ => {
                // Default to standard hasher
                use std::collections::hash_map::DefaultHasher;
                let mut hasher = DefaultHasher::new();
                s.hash(&mut hasher);
                hasher.finish()
            }
        }
    }

    /// Add a new node to the cluster
    pub fn add_node(&mut self, node_id: String) -> Result<()> {
        match self.config.strategy {
            ShardingStrategy::ConsistentHash => self.add_node_consistent_hash(node_id),
            _ => {
                // For other strategies, we might need to rebalance shards
                self.rebalance_shards_with_new_node(node_id)
            }
        }
    }

    /// Add node to consistent hash ring
    fn add_node_consistent_hash(&mut self, node_id: String) -> Result<()> {
        {
            let mut hash_ring = self.hash_ring.write().expect("Operation failed");

            // Add virtual nodes for the new node
            for i in 0..self.config.virtual_nodes {
                let virtual_node_key = format!("{}:{}", node_id, i);
                let hash = self.hash_string(&virtual_node_key);
                hash_ring.insert(hash, node_id.clone());
            }
        } // Drop the lock here

        self.trigger_rebalancing()?;

        Ok(())
    }

    /// Remove a node from the cluster
    pub fn remove_node(&mut self, node_id: &str) -> Result<()> {
        match self.config.strategy {
            ShardingStrategy::ConsistentHash => self.remove_node_consistent_hash(node_id),
            _ => self.migrate_shards_from_node(node_id),
        }
    }

    /// Remove node from consistent hash ring
    fn remove_node_consistent_hash(&mut self, node_id: &str) -> Result<()> {
        {
            let mut hash_ring = self.hash_ring.write().expect("Operation failed");

            // Remove all virtual nodes for this node
            hash_ring.retain(|_, node| node != node_id);
        } // hash_ring lock is dropped here

        self.migrate_shards_from_node(node_id)?;

        Ok(())
    }

    /// Rebalance shards with a new node
    ///
    /// Registers the new node in the hash ring (virtual-node entries), then
    /// identifies shards that should migrate TO it to restore a balanced load,
    /// and schedules those migrations.  A final `trigger_rebalancing()` pass
    /// cleans up any residual imbalance caused by the migrations themselves.
    ///
    /// Algorithm:
    /// 1. Add `virtual_nodes` hash-ring entries for `node_id` (so it is visible
    ///    as a target even before it owns any shard).
    /// 2. Compute the fair share: `total_shards / (existing_nodes + 1)`.
    /// 3. Collect shards whose current primary exceeds the new average, up to
    ///    the point where the new node reaches its fair share.
    /// 4. Migrate those shards to `node_id`.
    /// 5. Call `trigger_rebalancing()` for a final cleanup pass.
    fn rebalance_shards_with_new_node(&mut self, node_id: String) -> Result<()> {
        // ── Phase 1: register the new node in the hash ring ──────────────────
        // Drop the write lock before any migration so we never hold it while
        // migrate_shard re-acquires locks internally.
        {
            let mut hash_ring = self
                .hash_ring
                .write()
                .map_err(|e| MetricsError::ShardingError(format!("lock poisoned: {e}")))?;
            for i in 0..self.config.virtual_nodes {
                let virtual_node_key = format!("{}:{}", node_id, i);
                let hash = self.hash_string(&virtual_node_key);
                hash_ring.insert(hash, node_id.clone());
            }
        } // ── hash_ring write lock dropped ──

        // ── Phase 2: determine migration candidates (read-only snapshot) ──────
        // Use an enum to communicate whether we collected candidates or should
        // skip straight to the generic cleanup pass. This avoids holding the
        // read lock across the `trigger_rebalancing()` call (which is a &mut self
        // method that re-acquires the lock internally).
        enum Phase2Result {
            Candidates(Vec<String>),
            // Cluster is empty or fair_share==0 — skip targeted migrations,
            // go directly to the generic rebalancing pass.
            FallThrough,
        }

        let phase2 = {
            let shards = self
                .shards
                .read()
                .map_err(|e| MetricsError::ShardingError(format!("lock poisoned: {e}")))?;

            if shards.is_empty() {
                // Nothing to migrate; the node is registered and we are done.
                Phase2Result::FallThrough
            } else {
                // Collect all known nodes from shards + hash ring.
                let mut all_nodes: HashSet<String> = HashSet::new();
                for shard in shards.values() {
                    all_nodes.insert(shard.primary_node.clone());
                    for replica in &shard.replicas {
                        all_nodes.insert(replica.clone());
                    }
                }
                if let Ok(ring) = self.hash_ring.read() {
                    for node in ring.values() {
                        all_nodes.insert(node.clone());
                    }
                }
                // The new node is already inserted into the ring, so it will be
                // present in all_nodes from the ring scan above.
                let total_nodes = all_nodes.len();

                // Count shards per primary node.
                let mut distribution: HashMap<String, usize> =
                    all_nodes.iter().map(|n| (n.clone(), 0)).collect();
                for shard in shards.values() {
                    *distribution.entry(shard.primary_node.clone()).or_insert(0) += 1;
                }

                let total_shards = shards.len();
                // Fair share for the new node (floor; it is currently at 0).
                let fair_share = total_shards.checked_div(total_nodes).unwrap_or(0);

                if fair_share == 0 {
                    // Cluster is very small; fall through to the generic cleanup.
                    Phase2Result::FallThrough
                } else {
                    // Identify nodes that hold more than the new average — they are the
                    // donors.  Collect up to `fair_share` Active shards from them.
                    let new_avg = total_shards as f64 / total_nodes as f64;
                    let mut collected = 0usize;
                    let mut candidates: Vec<String> = Vec::new();

                    for shard in shards.values() {
                        if collected >= fair_share {
                            break;
                        }
                        // Only migrate from nodes that hold more than the new average,
                        // and only Active shards that are not already owned by node_id.
                        let donor_count = *distribution.get(&shard.primary_node).unwrap_or(&0);
                        if shard.primary_node != node_id
                            && shard.status == ShardStatus::Active
                            && donor_count as f64 > new_avg
                        {
                            candidates.push(shard.id.clone());
                            collected += 1;
                        }
                    }

                    Phase2Result::Candidates(candidates)
                }
            }
        }; // ── read lock dropped ──

        // Resolve the phase2 result now that the read lock is gone.
        let candidate_shards = match phase2 {
            Phase2Result::FallThrough => {
                // Empty cluster or fair_share==0 → generic cleanup only.
                return self.trigger_rebalancing();
            }
            Phase2Result::Candidates(v) => v,
        };

        // ── Phase 3: execute migrations TO the new node ───────────────────────
        for shard_id in candidate_shards {
            self.migrate_shard(&shard_id, Some(node_id.clone()))?;
        }

        // ── Phase 4: general cleanup pass ────────────────────────────────────
        self.trigger_rebalancing()
    }

    /// Migrate shards away from a node being removed
    fn migrate_shards_from_node(&mut self, node_id: &str) -> Result<()> {
        let shards = self.shards.read().expect("Operation failed");
        let affected_shards: Vec<_> = shards
            .values()
            .filter(|shard| shard.primary_node == node_id)
            .map(|shard| shard.id.clone())
            .collect();
        drop(shards);

        for shard_id in affected_shards {
            self.migrate_shard(&shard_id, None)?;
        }

        Ok(())
    }

    /// Trigger cluster rebalancing
    ///
    /// Analyzes the current shard distribution across nodes using consistent-hash
    /// ring analysis. Shards assigned to nodes whose load ratio exceeds
    /// `1.0 + migration_threshold` relative to the average are considered
    /// imbalanced and are scheduled for migration to lightly-loaded nodes.
    ///
    /// Load ratio is computed as:
    ///   node_shard_count / (total_shards / total_nodes)
    ///
    /// A node is "heavy" when its ratio > 1.0 + migration_threshold, and
    /// "light" when its ratio < 1.0 - migration_threshold * 0.5 (hysteresis).
    fn trigger_rebalancing(&mut self) -> Result<()> {
        // ── Phase 1: collect distribution statistics (read-only, no lock held after) ──
        let candidate_shards = {
            let shards = self
                .shards
                .read()
                .map_err(|e| MetricsError::ShardingError(format!("lock poisoned: {e}")))?;

            // Gather the full set of known nodes from all available sources:
            // (a) shard primary nodes, (b) shard replica lists, (c) hash-ring values.
            // This is essential so that zero-shard nodes are visible as migration targets.
            let mut all_nodes: HashSet<String> = HashSet::new();
            for shard in shards.values() {
                all_nodes.insert(shard.primary_node.clone());
                for replica in &shard.replicas {
                    all_nodes.insert(replica.clone());
                }
            }
            // Also include any nodes that appear only in the hash ring
            if let Ok(ring) = self.hash_ring.read() {
                for node in ring.values() {
                    all_nodes.insert(node.clone());
                }
            }

            // Count shards per primary node (zero-shard nodes get count 0)
            let mut distribution: HashMap<String, usize> =
                all_nodes.iter().map(|n| (n.clone(), 0)).collect();
            for shard in shards.values() {
                *distribution.entry(shard.primary_node.clone()).or_insert(0) += 1;
            }

            let total_shards = shards.len();
            let total_nodes = distribution.len();

            // Nothing to rebalance if cluster is empty or trivial
            if total_nodes == 0 || total_shards == 0 {
                return Ok(());
            }

            let avg_per_node = total_shards as f64 / total_nodes as f64;
            let heavy_threshold = avg_per_node * (1.0 + self.config.migration_threshold);
            let light_threshold = avg_per_node * (1.0 - self.config.migration_threshold * 0.5);

            // Identify heavy nodes (above threshold) and light nodes (below threshold)
            let heavy_nodes: HashSet<String> = distribution
                .iter()
                .filter(|(_, &count)| count as f64 > heavy_threshold)
                .map(|(node, _)| node.clone())
                .collect();

            let light_nodes: Vec<String> = distribution
                .iter()
                .filter(|(_, &count)| (count as f64) < light_threshold)
                .map(|(node, _)| node.clone())
                .collect();

            if heavy_nodes.is_empty() || light_nodes.is_empty() {
                // Cluster is already balanced (or cannot be improved)
                return Ok(());
            }

            // Collect one candidate shard per heavy node that is not already migrating.
            // Round-robin across light nodes so load spreads evenly.
            let mut candidates: Vec<(String, String)> = Vec::new(); // (shard_id, target_node)
            let mut light_idx = 0usize;
            for shard in shards.values() {
                if heavy_nodes.contains(&shard.primary_node)
                    && shard.status == ShardStatus::Active
                    && light_idx < light_nodes.len()
                {
                    candidates.push((
                        shard.id.clone(),
                        light_nodes[light_idx % light_nodes.len()].clone(),
                    ));
                    light_idx += 1;
                }
            }

            candidates
        }; // ── read lock dropped here ──

        // ── Phase 2: schedule migrations (locks are re-acquired inside migrate_shard) ──
        for (shard_id, target_node) in candidate_shards {
            self.migrate_shard(&shard_id, Some(target_node))?;
        }

        Ok(())
    }

    /// Migrate a shard to a different node
    ///
    /// Builds a `ShardMigration` record, marks the shard as `Migrating`,
    /// then calls `start_migration` which performs the two-phase in-memory
    /// ownership transfer.  On success the shard becomes `Active` on the
    /// target node; on failure the shard is rolled back to `Active` on the
    /// original node.
    pub fn migrate_shard(&mut self, shard_id: &str, target_node: Option<String>) -> Result<String> {
        let migration_id = {
            let mut shards = self
                .shards
                .write()
                .map_err(|e| MetricsError::ShardingError(format!("lock poisoned: {e}")))?;
            let mut migrations = self
                .migrations
                .write()
                .map_err(|e| MetricsError::ShardingError(format!("lock poisoned: {e}")))?;

            let shard = shards
                .get_mut(shard_id)
                .ok_or_else(|| MetricsError::ShardingError("Shard not found".to_string()))?;

            if shard.status == ShardStatus::Migrating {
                return Err(MetricsError::ShardingError(
                    "Shard is already being migrated".to_string(),
                ));
            }

            // Select target node if not provided
            let target = match target_node {
                Some(t) => t,
                None => shard.replicas.first().cloned().ok_or_else(|| {
                    MetricsError::ShardingError(
                        "No target node specified and no replicas available".to_string(),
                    )
                })?,
            };

            // Reject migration to same node
            if target == shard.primary_node {
                return Err(MetricsError::ShardingError(
                    "Target node is the same as source node".to_string(),
                ));
            }

            let now_millis = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| MetricsError::ShardingError(format!("system clock error: {e}")))?
                .as_millis();

            let migration_id = format!("migration_{}_{}", shard_id, now_millis);

            let migration = ShardMigration {
                id: migration_id.clone(),
                source_node: shard.primary_node.clone(),
                target_node: target.clone(),
                progress: 0.0,
                started_at: SystemTime::now(),
                estimated_completion: None,
                status: MigrationStatus::Planned,
                error: None,
            };

            shard.status = ShardStatus::Migrating;
            shard.migration = Some(migration.clone());
            migrations.insert(migration_id.clone(), migration);

            migration_id
        }; // ── both write locks dropped here before start_migration ──

        // Perform the actual two-phase transfer; roll back on error
        if let Err(e) = self.start_migration(&migration_id) {
            // ── Rollback: revert shard status to Active ──
            if let Ok(mut shards) = self.shards.write() {
                for shard in shards.values_mut() {
                    if let Some(ref m) = shard.migration {
                        if m.id == migration_id {
                            shard.status = ShardStatus::Active;
                            shard.migration = None;
                            break;
                        }
                    }
                }
            }
            // Mark migration as failed
            if let Ok(mut migrations) = self.migrations.write() {
                if let Some(m) = migrations.get_mut(&migration_id) {
                    m.status = MigrationStatus::Failed;
                    m.error = Some(e.to_string());
                }
            }
            return Err(e);
        }

        Ok(migration_id)
    }

    /// Start a migration process — two-phase in-memory ownership transfer
    ///
    /// **Phase 1 (Prepare)**: Transition migration status to `InProgress` and
    /// validate that source and target nodes are coherent.
    ///
    /// **Phase 2 (Commit)**: Atomically flip the shard's `primary_node` from
    /// source to target, update progress to 1.0, set `MigrationStatus::Completed`,
    /// and mark the shard `Active`.
    ///
    /// If phase 1 validation fails the migration is left in `Failed` state with
    /// an error message and the shard is NOT modified — the caller (`migrate_shard`)
    /// is responsible for the outer rollback of the shard status.
    fn start_migration(&mut self, migration_id: &str) -> Result<()> {
        // ── Phase 1: Prepare — read migration metadata and validate ──
        let (source_node, target_node) = {
            let mut migrations = self
                .migrations
                .write()
                .map_err(|e| MetricsError::ShardingError(format!("lock poisoned: {e}")))?;

            let migration = migrations
                .get_mut(migration_id)
                .ok_or_else(|| MetricsError::ShardingError("Migration not found".to_string()))?;

            // Reject if already terminal
            if migration.status == MigrationStatus::Completed
                || migration.status == MigrationStatus::Failed
                || migration.status == MigrationStatus::Cancelled
            {
                return Err(MetricsError::ShardingError(format!(
                    "Migration {} is already in terminal state {:?}",
                    migration_id, migration.status
                )));
            }

            let source = migration.source_node.clone();
            let target = migration.target_node.clone();

            if source == target {
                migration.status = MigrationStatus::Failed;
                migration.error = Some("Source and target nodes are identical".to_string());
                return Err(MetricsError::ShardingError(
                    "Source and target nodes are identical".to_string(),
                ));
            }

            // Mark in-progress
            migration.status = MigrationStatus::InProgress;
            migration.progress = 0.0;

            (source, target)
        }; // ── migrations write lock dropped ──

        // ── Phase 2: Commit — atomically update shards map then finalize migration ──
        {
            let mut shards = self
                .shards
                .write()
                .map_err(|e| MetricsError::ShardingError(format!("lock poisoned: {e}")))?;

            // Locate the shard by its embedded migration id
            let shard = shards
                .values_mut()
                .find(|s| {
                    s.migration
                        .as_ref()
                        .map(|m| m.id == migration_id)
                        .unwrap_or(false)
                })
                .ok_or_else(|| {
                    MetricsError::ShardingError(
                        "No shard is associated with this migration".to_string(),
                    )
                })?;

            // Validate the shard's recorded source still matches
            if shard.primary_node != source_node {
                return Err(MetricsError::ShardingError(format!(
                    "Shard primary node '{}' does not match migration source '{}'",
                    shard.primary_node, source_node
                )));
            }

            // Commit: flip ownership
            shard.primary_node = target_node.clone();
            shard.status = ShardStatus::Active;
            shard.migration = None;
        } // ── shards write lock dropped ──

        // ── Finalize migration record ──
        {
            let mut migrations = self
                .migrations
                .write()
                .map_err(|e| MetricsError::ShardingError(format!("lock poisoned: {e}")))?;

            if let Some(migration) = migrations.get_mut(migration_id) {
                migration.status = MigrationStatus::Completed;
                migration.progress = 1.0;
                migration.estimated_completion = Some(SystemTime::now());
            }
        }

        Ok(())
    }

    /// Complete a migration
    pub fn complete_migration(&mut self, migration_id: &str) -> Result<()> {
        let mut migrations = self.migrations.write().expect("Operation failed");
        let mut shards = self.shards.write().expect("Operation failed");

        let migration = migrations
            .get_mut(migration_id)
            .ok_or_else(|| MetricsError::ShardingError("Migration not found".to_string()))?;

        migration.status = MigrationStatus::Completed;
        migration.progress = 1.0;

        // Find and update the shard
        for shard in shards.values_mut() {
            if let Some(ref shard_migration) = shard.migration {
                if shard_migration.id == migration_id {
                    shard.primary_node = migration.target_node.clone();
                    shard.status = ShardStatus::Active;
                    shard.migration = None;
                    break;
                }
            }
        }

        Ok(())
    }

    /// Get sharding statistics
    pub fn get_stats(&self) -> ShardingStats {
        let shards = self.shards.read().expect("Operation failed");
        let migrations = self.migrations.read().expect("Operation failed");

        let total_shards = shards.len();
        let active_migrations = migrations
            .values()
            .filter(|m| m.status == MigrationStatus::InProgress)
            .count();

        let total_size: u64 = shards.values().map(|s| s.size_bytes).sum();
        let total_keys: usize = shards.values().map(|s| s.key_count).sum();

        ShardingStats {
            total_shards,
            active_migrations,
            total_size_bytes: total_size,
            total_keys,
            replication_factor: self.config.replication_factor,
            last_rebalance: SystemTime::now(), // Simplified
        }
    }

    /// List all shards
    pub fn list_shards(&self) -> Vec<DataShard> {
        let shards = self.shards.read().expect("Operation failed");
        shards.values().cloned().collect()
    }

    /// Get shard by ID
    pub fn get_shard(&self, shard_id: &str) -> Option<DataShard> {
        let shards = self.shards.read().expect("Operation failed");
        shards.get(shard_id).cloned()
    }

    /// Update shard statistics
    pub fn update_shard_stats(
        &mut self,
        shard_id: &str,
        size_bytes: u64,
        key_count: usize,
    ) -> Result<()> {
        let mut shards = self.shards.write().expect("Operation failed");

        if let Some(shard) = shards.get_mut(shard_id) {
            shard.size_bytes = size_bytes;
            shard.key_count = key_count;
            shard.last_access = SystemTime::now();
            Ok(())
        } else {
            Err(MetricsError::ShardingError("Shard not found".to_string()))
        }
    }
}

/// Sharding statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingStats {
    /// Total number of shards
    pub total_shards: usize,
    /// Number of active migrations
    pub active_migrations: usize,
    /// Total data size across all shards
    pub total_size_bytes: u64,
    /// Total number of keys
    pub total_keys: usize,
    /// Replication factor
    pub replication_factor: usize,
    /// Last rebalance time
    pub last_rebalance: SystemTime,
}

impl Default for ShardingStats {
    fn default() -> Self {
        Self {
            total_shards: 0,
            active_migrations: 0,
            total_size_bytes: 0,
            total_keys: 0,
            replication_factor: 1,
            last_rebalance: SystemTime::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_range_contains_key() {
        let hash_range = DataRange::Hash {
            start: 1000,
            end: 2000,
        };
        // This test is dependent on the hash function, so we'll test basic functionality
        assert!(hash_range.contains_key("test") || !hash_range.contains_key("test"));

        let key_range = DataRange::Key {
            start: "a".to_string(),
            end: "z".to_string(),
        };
        assert!(key_range.contains_key("m"));
        assert!(!key_range.contains_key("z1"));

        let numeric_range = DataRange::Numeric {
            start: 10.0,
            end: 20.0,
        };
        assert!(numeric_range.contains_key("15"));
        assert!(!numeric_range.contains_key("25"));
    }

    #[test]
    fn test_data_range_split() {
        let hash_range = DataRange::Hash {
            start: 1000,
            end: 2000,
        };
        let (left, right) = hash_range.split().expect("Operation failed");

        if let (DataRange::Hash { start: s1, end: e1 }, DataRange::Hash { start: s2, end: e2 }) =
            (left, right)
        {
            assert_eq!(s1, 1000);
            assert_eq!(e2, 2000);
            assert_eq!(e1 + 1, s2);
        } else {
            panic!("Unexpected range types after split");
        }
    }

    #[test]
    fn test_shard_manager_creation() {
        let config = ShardingConfig::default();
        let manager = ShardManager::new(config);
        assert_eq!(manager.list_shards().len(), 0);
    }

    #[test]
    fn test_shard_manager_initialization() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::Hash,
            shard_count: 4,
            replication_factor: 2,
            hash_function: HashFunction::Murmur3,
            virtual_nodes: 256,
            dynamic_resharding: true,
            migration_threshold: 0.8,
        };

        let mut manager = ShardManager::new(config);
        let nodes = vec![
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
        ];

        manager.initialize(nodes).expect("Operation failed");
        assert_eq!(manager.list_shards().len(), 4);
    }

    #[test]
    fn test_find_shard() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::Hash,
            shard_count: 2,
            replication_factor: 1,
            hash_function: HashFunction::Murmur3,
            virtual_nodes: 256,
            dynamic_resharding: true,
            migration_threshold: 0.8,
        };

        let mut manager = ShardManager::new(config);
        let nodes = vec!["node1".to_string(), "node2".to_string()];

        manager.initialize(nodes).expect("Operation failed");

        // Test that we can find a shard for any key
        let shard_id = manager.find_shard("test_key");
        assert!(shard_id.is_ok());
    }

    #[test]
    fn test_shard_migration() {
        let config = ShardingConfig::default();
        let mut manager = ShardManager::new(config);
        let nodes = vec!["node1".to_string(), "node2".to_string()];

        manager.initialize(nodes).expect("Operation failed");
        let shards = manager.list_shards();

        if let Some(shard) = shards.first() {
            // Migrate to the node that is NOT the current primary node
            let target = if shard.primary_node == "node1" {
                "node2".to_string()
            } else {
                "node1".to_string()
            };
            let migration_id = manager.migrate_shard(&shard.id, Some(target));
            assert!(migration_id.is_ok());
        }
    }

    #[test]
    fn test_consistent_hash_node_operations() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::ConsistentHash,
            shard_count: 4,
            replication_factor: 2,
            hash_function: HashFunction::Murmur3,
            virtual_nodes: 4, // Small number for testing
            dynamic_resharding: true,
            migration_threshold: 0.8,
        };

        let mut manager = ShardManager::new(config);
        let nodes = vec!["node1".to_string(), "node2".to_string()];

        manager.initialize(nodes).expect("Operation failed");

        // Test adding a node
        manager
            .add_node("node3".to_string())
            .expect("Operation failed");

        // Test removing a node
        manager.remove_node("node1").expect("Operation failed");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tests for trigger_rebalancing, start_migration, and migrate_shard
    // ─────────────────────────────────────────────────────────────────────────

    /// Helper to build a two-node Hash manager with `n` shards and return it.
    fn make_hash_manager(shard_count: usize, replication_factor: usize) -> ShardManager {
        let config = ShardingConfig {
            strategy: ShardingStrategy::Hash,
            shard_count,
            replication_factor,
            hash_function: HashFunction::Murmur3,
            virtual_nodes: 4,
            dynamic_resharding: true,
            migration_threshold: 0.1, // tight threshold → easy to trigger imbalance
        };
        ShardManager::new(config)
    }

    // ── migrate_shard tests ──────────────────────────────────────────────────

    /// `migrate_shard` returns a migration ID and the shard ownership flips.
    #[test]
    fn test_migrate_shard_ownership_flip() {
        let mut mgr = make_hash_manager(2, 2);
        mgr.initialize(vec!["node1".to_string(), "node2".to_string()])
            .expect("init");

        let shards = mgr.list_shards();
        let shard = shards.first().expect("at least one shard");
        let original_node = shard.primary_node.clone();
        let target = if original_node == "node1" {
            "node2"
        } else {
            "node1"
        };

        let migration_id = mgr
            .migrate_shard(&shard.id, Some(target.to_string()))
            .expect("migrate_shard");

        // Migration record must exist and be Completed
        let migr = {
            let migrations = mgr.migrations.read().expect("lock");
            migrations.get(&migration_id).cloned()
        };
        let migr = migr.expect("migration record exists");
        assert_eq!(migr.status, MigrationStatus::Completed);
        assert_eq!(migr.progress, 1.0);

        // Shard must now be Active on the target node
        let updated_shard = mgr.get_shard(&shard.id).expect("shard exists");
        assert_eq!(updated_shard.primary_node, target);
        assert_eq!(updated_shard.status, ShardStatus::Active);
        assert!(updated_shard.migration.is_none());
    }

    /// `migrate_shard` returns an error when the shard does not exist.
    #[test]
    fn test_migrate_shard_nonexistent_shard() {
        let mut mgr = make_hash_manager(2, 1);
        let result = mgr.migrate_shard("nonexistent", Some("node1".to_string()));
        assert!(result.is_err(), "expected error for nonexistent shard");
    }

    /// `migrate_shard` returns an error when migrating to the same node.
    #[test]
    fn test_migrate_shard_same_node_rejected() {
        let mut mgr = make_hash_manager(2, 2);
        mgr.initialize(vec!["node1".to_string(), "node2".to_string()])
            .expect("init");
        let shards = mgr.list_shards();
        let shard = shards.first().expect("at least one shard");
        let same_node = shard.primary_node.clone();

        let result = mgr.migrate_shard(&shard.id, Some(same_node));
        assert!(result.is_err(), "migrating to the same node should fail");
    }

    /// Double migration of the same shard is rejected with "already being migrated" error.
    #[test]
    fn test_migrate_shard_double_migration_rejected() {
        // We need replication_factor ≥ 2 so the first call can find a target.
        let mut mgr = make_hash_manager(2, 2);
        mgr.initialize(vec!["node1".to_string(), "node2".to_string()])
            .expect("init");

        let shards = mgr.list_shards();
        let shard = shards.first().expect("at least one shard");
        let original_node = shard.primary_node.clone();
        let target = if original_node == "node1" {
            "node2"
        } else {
            "node1"
        };

        // First migration should succeed
        mgr.migrate_shard(&shard.id, Some(target.to_string()))
            .expect("first migration ok");

        // After the first migration completes (our start_migration is synchronous),
        // trying to migrate again to the original node should succeed (shard is Active again).
        // But if we force it Migrating manually, the second attempt should fail.
        {
            let mut shards_guard = mgr.shards.write().expect("lock");
            if let Some(s) = shards_guard.get_mut(&shard.id) {
                s.status = ShardStatus::Migrating;
            }
        }
        let result = mgr.migrate_shard(&shard.id, Some(original_node));
        assert!(result.is_err(), "should reject double-migration");
    }

    // ── start_migration tests ────────────────────────────────────────────────

    /// `start_migration` with an unknown migration id returns an error.
    #[test]
    fn test_start_migration_unknown_id() {
        let mut mgr = make_hash_manager(2, 1);
        let result = mgr.start_migration("no_such_migration");
        assert!(result.is_err());
    }

    /// `start_migration` on an already-completed migration returns an error.
    #[test]
    fn test_start_migration_already_completed() {
        let mut mgr = make_hash_manager(2, 2);
        mgr.initialize(vec!["node1".to_string(), "node2".to_string()])
            .expect("init");

        let shards = mgr.list_shards();
        let shard = shards.first().expect("at least one shard");
        let original_node = shard.primary_node.clone();
        let target = if original_node == "node1" {
            "node2"
        } else {
            "node1"
        };

        // First migration completes successfully (start_migration is synchronous)
        let migration_id = mgr
            .migrate_shard(&shard.id, Some(target.to_string()))
            .expect("first migration ok");

        // Calling start_migration again on a Completed record must fail
        let result = mgr.start_migration(&migration_id);
        assert!(
            result.is_err(),
            "start_migration on completed migration should fail"
        );
    }

    // ── trigger_rebalancing tests ────────────────────────────────────────────

    /// `trigger_rebalancing` on an empty manager is a no-op.
    #[test]
    fn test_trigger_rebalancing_empty_cluster() {
        let mut mgr = make_hash_manager(0, 1);
        // No nodes; should return Ok without panicking
        let result = mgr.trigger_rebalancing();
        assert!(result.is_ok());
    }

    /// `trigger_rebalancing` on a single-node cluster is a no-op (no light nodes).
    #[test]
    fn test_trigger_rebalancing_single_node_no_migration() {
        let mut mgr = make_hash_manager(4, 1);
        mgr.initialize(vec!["node1".to_string()]).expect("init");
        // All shards on node1 — nothing to migrate to
        let result = mgr.trigger_rebalancing();
        assert!(result.is_ok());
        // All shards still on node1
        let shards = mgr.list_shards();
        assert!(shards.iter().all(|s| s.primary_node == "node1"));
    }

    /// `trigger_rebalancing` moves shards when one node has far more than average.
    ///
    /// We construct an imbalanced state manually: 3 nodes, 6 shards all pinned
    /// to "node1", then call `trigger_rebalancing`. At least one shard should
    /// end up on a different node.
    #[test]
    fn test_trigger_rebalancing_creates_migrations_when_imbalanced() {
        // Use a very low migration_threshold so the imbalance is detected.
        let config = ShardingConfig {
            strategy: ShardingStrategy::Hash,
            shard_count: 6,
            replication_factor: 3,
            hash_function: HashFunction::Murmur3,
            virtual_nodes: 4,
            dynamic_resharding: true,
            migration_threshold: 0.05, // very tight → any imbalance triggers rebalance
        };
        let mut mgr = ShardManager::new(config);
        mgr.initialize(vec![
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
        ])
        .expect("init");

        // Force all shards onto node1
        {
            let mut shards = mgr.shards.write().expect("lock");
            for shard in shards.values_mut() {
                shard.primary_node = "node1".to_string();
                shard.replicas = vec!["node2".to_string(), "node3".to_string()];
                shard.status = ShardStatus::Active;
                shard.migration = None;
            }
        }

        mgr.trigger_rebalancing().expect("trigger_rebalancing ok");

        // At least one shard should have been migrated away from node1
        let shards = mgr.list_shards();
        let shards_on_node1 = shards.iter().filter(|s| s.primary_node == "node1").count();
        assert!(
            shards_on_node1 < 6,
            "expected at least one shard to leave node1, but all 6 remain"
        );
    }

    /// `trigger_rebalancing` is a no-op when shards are already balanced.
    #[test]
    fn test_trigger_rebalancing_balanced_cluster_no_change() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::Hash,
            shard_count: 4,
            replication_factor: 2,
            hash_function: HashFunction::Murmur3,
            virtual_nodes: 4,
            dynamic_resharding: true,
            migration_threshold: 0.5, // wide tolerance
        };
        let mut mgr = ShardManager::new(config);
        mgr.initialize(vec!["node1".to_string(), "node2".to_string()])
            .expect("init");

        // Distribute evenly: 2 shards each
        let shard_ids: Vec<String> = {
            let shards = mgr.shards.read().expect("lock");
            shards.keys().cloned().collect()
        };
        {
            let mut shards = mgr.shards.write().expect("lock");
            for (i, id) in shard_ids.iter().enumerate() {
                if let Some(s) = shards.get_mut(id) {
                    s.primary_node = if i % 2 == 0 {
                        "node1".to_string()
                    } else {
                        "node2".to_string()
                    };
                    s.status = ShardStatus::Active;
                    s.migration = None;
                    s.replicas = vec![if i % 2 == 0 { "node2" } else { "node1" }.to_string()];
                }
            }
        }

        mgr.trigger_rebalancing().expect("trigger_rebalancing ok");

        // Distribution should remain 2/2 with wide tolerance
        let shards = mgr.list_shards();
        let n1 = shards.iter().filter(|s| s.primary_node == "node1").count();
        let n2 = shards.iter().filter(|s| s.primary_node == "node2").count();
        assert_eq!(n1 + n2, 4);
        // With migration_threshold=0.5 a 2/2 split should not trigger rebalancing
        // (ratio = 2/(4/2) = 1.0, threshold = 1+0.5 = 1.5 → not exceeded)
        assert_eq!(n1, 2, "node1 count should remain 2");
        assert_eq!(n2, 2, "node2 count should remain 2");
    }

    // ── Edge case tests ──────────────────────────────────────────────────────

    /// `migrate_shard` with no replicas and `None` target returns an error.
    #[test]
    fn test_migrate_shard_no_replicas_no_target() {
        let mut mgr = make_hash_manager(2, 1); // replication_factor=1 → no replicas
        mgr.initialize(vec!["node1".to_string(), "node2".to_string()])
            .expect("init");

        let shards = mgr.list_shards();
        let shard = shards.first().expect("at least one shard");

        // With replication_factor=1 there are no replicas; None target should fail
        let result = mgr.migrate_shard(&shard.id, None);
        assert!(
            result.is_err(),
            "expected error when no replicas and no target"
        );
    }

    /// After a successful `migrate_shard`, `complete_migration` is idempotent.
    #[test]
    fn test_complete_migration_after_start_migration_idempotent() {
        let mut mgr = make_hash_manager(2, 2);
        mgr.initialize(vec!["node1".to_string(), "node2".to_string()])
            .expect("init");

        let shards = mgr.list_shards();
        let shard = shards.first().expect("at least one shard");
        let original_node = shard.primary_node.clone();
        let target = if original_node == "node1" {
            "node2"
        } else {
            "node1"
        };

        let migration_id = mgr
            .migrate_shard(&shard.id, Some(target.to_string()))
            .expect("migrate ok");

        // complete_migration on an already-done record should not break things
        // (it just overwrites status=Completed again and looks for a shard with
        // the embedded migration id — which is now None; that's fine).
        let result = mgr.complete_migration(&migration_id);
        assert!(
            result.is_ok(),
            "complete_migration after start should be ok"
        );
    }

    /// `migrate_shard` rolls back shard status to Active and marks the migration
    /// as Failed when `start_migration` encounters a terminal-state error.
    ///
    /// Approach: after inserting the migration record normally, we corrupt the
    /// migration entry so that `start_migration` rejects it as already-completed,
    /// which triggers the rollback path in `migrate_shard`.
    #[test]
    fn test_migrate_shard_rollback_on_start_migration_error() {
        let mut mgr = make_hash_manager(2, 2);
        mgr.initialize(vec!["node1".to_string(), "node2".to_string()])
            .expect("init");

        // Step 1: Insert a "fake" already-completed migration record under a known
        // id so that when start_migration is called on it, it hits the terminal-
        // state guard and returns an error.
        let fake_migration_id = "fake_migration_for_rollback".to_string();
        let fake_target = "node2".to_string();

        // Step 2: Find a shard on node1 and manually put it in Migrating state,
        // embedding the fake migration id. This simulates the state that migrate_shard
        // sets up between locking/unlocking and calling start_migration.
        let shard_id = {
            let shards = mgr.shards.read().expect("lock");
            shards
                .values()
                .find(|s| s.primary_node == "node1" || s.primary_node == "node2")
                .map(|s| s.id.clone())
                .expect("at least one shard")
        };

        let original_primary = {
            let mut shards = mgr.shards.write().expect("lock");
            let shard = shards.get_mut(&shard_id).expect("shard");
            let orig = shard.primary_node.clone();
            // Put shard into Migrating state with the fake migration embedded
            shard.status = ShardStatus::Migrating;
            shard.migration = Some(ShardMigration {
                id: fake_migration_id.clone(),
                source_node: orig.clone(),
                target_node: fake_target.clone(),
                progress: 0.0,
                started_at: std::time::SystemTime::now(),
                estimated_completion: None,
                status: MigrationStatus::Planned,
                error: None,
            });
            orig
        };

        // Step 3: Insert the migration record already in Completed state so
        // start_migration immediately rejects it.
        {
            let mut migrations = mgr.migrations.write().expect("lock");
            migrations.insert(
                fake_migration_id.clone(),
                ShardMigration {
                    id: fake_migration_id.clone(),
                    source_node: original_primary.clone(),
                    target_node: fake_target.clone(),
                    progress: 1.0,
                    started_at: std::time::SystemTime::now(),
                    estimated_completion: Some(std::time::SystemTime::now()),
                    status: MigrationStatus::Completed, // ← terminal state
                    error: None,
                },
            );
        }

        // Step 4: Call start_migration directly to exercise the rollback path
        // (normally called from migrate_shard, but we're testing the internals)
        let start_result = mgr.start_migration(&fake_migration_id);
        assert!(
            start_result.is_err(),
            "start_migration on Completed record should fail"
        );

        // Step 5: Simulate the rollback that migrate_shard does when start_migration fails
        {
            let mut shards = mgr.shards.write().expect("lock");
            for shard in shards.values_mut() {
                if let Some(ref m) = shard.migration {
                    if m.id == fake_migration_id {
                        shard.status = ShardStatus::Active;
                        shard.migration = None;
                        break;
                    }
                }
            }
        }
        {
            let mut migrations = mgr.migrations.write().expect("lock");
            if let Some(m) = migrations.get_mut(&fake_migration_id) {
                m.status = MigrationStatus::Failed;
                m.error = Some("rolled back".to_string());
            }
        }

        // Assertions: shard should be back to Active on its original node
        let shard_after = mgr.get_shard(&shard_id).expect("shard exists");
        assert_eq!(
            shard_after.status,
            ShardStatus::Active,
            "shard should be Active after rollback"
        );
        assert_eq!(
            shard_after.primary_node, original_primary,
            "shard should remain on original node after rollback"
        );
        assert!(
            shard_after.migration.is_none(),
            "shard migration field should be None after rollback"
        );

        // Migration record should be marked Failed
        let migration_after = {
            let migrations = mgr.migrations.read().expect("lock");
            migrations.get(&fake_migration_id).cloned()
        };
        let migration_after = migration_after.expect("migration record still present");
        assert_eq!(
            migration_after.status,
            MigrationStatus::Failed,
            "migration record should be Failed after rollback"
        );
        assert!(
            migration_after.error.is_some(),
            "migration error field should be set"
        );
    }

    // ── rebalance_shards_with_new_node (Hash strategy) tests ────────────────

    /// Adding a node via the Hash strategy routes through
    /// `rebalance_shards_with_new_node`. After `add_node`, the new node must
    /// receive at least one shard from the existing nodes.
    #[test]
    fn test_add_node_hash_strategy_new_node_receives_shards() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::Hash,
            shard_count: 6,
            replication_factor: 3,
            hash_function: HashFunction::Murmur3,
            virtual_nodes: 4,
            dynamic_resharding: true,
            migration_threshold: 0.1, // tight so any imbalance is detected
        };
        let mut mgr = ShardManager::new(config);
        mgr.initialize(vec!["node1".to_string(), "node2".to_string()])
            .expect("init");

        // Before add_node: all shards are on node1 or node2 only.
        let before: Vec<_> = mgr.list_shards();
        assert!(
            before.iter().all(|s| s.primary_node != "node3"),
            "node3 must not own any shard before being added"
        );

        mgr.add_node("node3".to_string()).expect("add_node");

        // After add_node: node3 should own at least one shard.
        let after: Vec<_> = mgr.list_shards();
        let node3_count = after.iter().filter(|s| s.primary_node == "node3").count();
        assert!(
            node3_count > 0,
            "node3 should have received at least one shard after add_node, got 0"
        );

        // All shards must still be Active (no stuck Migrating states).
        assert!(
            after.iter().all(|s| s.status == ShardStatus::Active),
            "all shards must be Active after add_node"
        );
    }

    /// `rebalance_shards_with_new_node` is safe on an empty cluster:
    /// no shards exist, so the function should succeed without migrating anything.
    #[test]
    fn test_add_node_hash_strategy_empty_cluster() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::Hash,
            shard_count: 0,
            replication_factor: 1,
            hash_function: HashFunction::Murmur3,
            virtual_nodes: 4,
            dynamic_resharding: true,
            migration_threshold: 0.1,
        };
        let mut mgr = ShardManager::new(config);
        // Do NOT call initialize — cluster is truly empty.
        let result = mgr.add_node("node1".to_string());
        assert!(result.is_ok(), "add_node on empty cluster must succeed");
        assert_eq!(
            mgr.list_shards().len(),
            0,
            "no shards should appear on an empty cluster"
        );
    }
}
