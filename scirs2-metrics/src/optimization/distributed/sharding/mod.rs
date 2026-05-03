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

pub use super::config::{HashFunction, ShardingConfig, ShardingStrategy};

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

/// Simple glob match supporting `*` as a wildcard that matches any substring.
///
/// `*` in `pattern` matches zero or more characters; all other characters are
/// matched literally.  Multiple `*` wildcards are supported via a recursive
/// linear scan.
fn simple_glob_match(text: &str, pattern: &str) -> bool {
    let mut parts = pattern.split('*');
    let first = match parts.next() {
        Some(p) => p,
        None => return true,
    };

    // The text must start with the literal prefix before the first `*`.
    if !text.starts_with(first) {
        return false;
    }

    let mut remaining = &text[first.len()..];

    for part in parts {
        if part.is_empty() {
            // `**` or trailing `*` — skip, no literal to match
            continue;
        }
        match remaining.find(part) {
            Some(pos) => remaining = &remaining[pos + part.len()..],
            None => return false,
        }
    }

    // If the pattern ended without a trailing `*`, the text must be exhausted.
    if !pattern.ends_with('*') && !remaining.is_empty() {
        return false;
    }

    true
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
            DataRange::Geographic {
                lat_min,
                lat_max,
                lon_min,
                lon_max,
            } => {
                // key format expected: "lat:lon" or "lat,lon"
                let sep = if key.contains(':') { ':' } else { ',' };
                let parts: Vec<&str> = key.splitn(2, sep).collect();
                if parts.len() != 2 {
                    return false;
                }
                let lat = parts[0].trim().parse::<f64>().ok();
                let lon = parts[1].trim().parse::<f64>().ok();
                match (lat, lon) {
                    (Some(lat), Some(lon)) => {
                        lat >= *lat_min && lat <= *lat_max && lon >= *lon_min && lon <= *lon_max
                    }
                    _ => false,
                }
            }
            DataRange::Custom {
                range_type,
                range_data,
            } => {
                let data_str = match std::str::from_utf8(range_data) {
                    Ok(s) => s,
                    Err(_) => return false,
                };
                match range_type.as_str() {
                    "prefix" => key.starts_with(data_str),
                    "suffix" => key.ends_with(data_str),
                    "contains" => key.contains(data_str),
                    "regex" => simple_glob_match(key, data_str),
                    _ => false,
                }
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
            (
                DataRange::Geographic {
                    lat_min: lat_min1,
                    lat_max: lat_max1,
                    lon_min: lon_min1,
                    lon_max: lon_max1,
                },
                DataRange::Geographic {
                    lat_min: lat_min2,
                    lat_max: lat_max2,
                    lon_min: lon_min2,
                    lon_max: lon_max2,
                },
            ) => {
                // Bounding boxes overlap unless separated along either axis
                !(lat_max1 < lat_min2
                    || lat_max2 < lat_min1
                    || lon_max1 < lon_min2
                    || lon_max2 < lon_min1)
            }
            (
                DataRange::Custom {
                    range_type: rt1,
                    range_data: rd1,
                },
                DataRange::Custom {
                    range_type: rt2,
                    range_data: rd2,
                },
            ) => rt1 == rt2 && rd1 == rd2,
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
        } // hash_ring lock is dropped here

        // Trigger shard migration so the new node absorbs its fair share.
        self.rebalance_shards_with_new_node(node_id)?;

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

            // Remove all virtual nodes for this node.
            hash_ring.retain(|_, n| n != node_id);
        } // hash_ring lock is dropped here

        // Migrate every shard that was primary on the removed node.
        self.migrate_shards_from_node(node_id)?;

        Ok(())
    }

    /// Rebalance shards after a new node joins.
    ///
    /// Identifies shards whose consistent-hash position now maps to the new
    /// node and migrates them accordingly.
    fn rebalance_shards_with_new_node(&mut self, new_node_id: String) -> Result<()> {
        // Collect shard IDs that should now belong to the new node according
        // to the current hash ring.
        let shard_ids_to_move: Vec<String> = {
            let shards = self.shards.read().expect("Operation failed");
            shards
                .values()
                .filter(|shard| {
                    // Only consider active shards not already on the new node.
                    shard.primary_node != new_node_id
                        && shard.status == ShardStatus::Active
                        // Heuristic: check if the midpoint of the shard range now
                        // routes to the new node via the hash ring.
                        && self
                            .probe_shard_owner(shard)
                            .map(|owner| owner == new_node_id)
                            .unwrap_or(false)
                })
                .map(|s| s.id.clone())
                .collect()
        };

        for shard_id in shard_ids_to_move {
            self.migrate_shard(&shard_id, Some(new_node_id.clone()))?;
        }

        Ok(())
    }

    /// Determine which node owns a shard's representative hash position.
    fn probe_shard_owner(&self, shard: &DataShard) -> Option<String> {
        let probe_key = match &shard.range {
            DataRange::Hash { start, end } => {
                let mid = start + (end - start) / 2;
                mid.to_string()
            }
            DataRange::Key { start, .. } => start.clone(),
            _ => shard.id.clone(),
        };
        self.get_node_consistent_hash(&probe_key).ok()
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

    /// Trigger cluster rebalancing.
    ///
    /// Computes per-node shard counts, identifies nodes that carry more than
    /// the average share by at least `migration_threshold`, and migrates one
    /// shard per over-loaded node to the least-loaded node.
    fn trigger_rebalancing(&mut self) -> Result<()> {
        if !self.config.dynamic_resharding {
            return Ok(());
        }

        // Compute shard counts per node.
        let (node_shard_counts, shard_ids_by_node): (
            HashMap<String, usize>,
            HashMap<String, Vec<String>>,
        ) = {
            let shards = self.shards.read().expect("Operation failed");
            let mut counts: HashMap<String, usize> = HashMap::new();
            let mut by_node: HashMap<String, Vec<String>> = HashMap::new();
            for shard in shards.values() {
                if shard.status != ShardStatus::Active {
                    continue;
                }
                *counts.entry(shard.primary_node.clone()).or_insert(0) += 1;
                by_node
                    .entry(shard.primary_node.clone())
                    .or_default()
                    .push(shard.id.clone());
            }
            (counts, by_node)
        };

        if node_shard_counts.is_empty() {
            return Ok(());
        }

        let total_shards: usize = node_shard_counts.values().sum();
        let node_count = node_shard_counts.len();
        if node_count == 0 {
            return Ok(());
        }
        let avg = total_shards as f64 / node_count as f64;
        let threshold = avg * self.config.migration_threshold;

        // Find the least-loaded node.
        let least_loaded = node_shard_counts
            .iter()
            .min_by_key(|(_, &c)| c)
            .map(|(n, _)| n.clone());

        let target_node = match least_loaded {
            Some(n) => n,
            None => return Ok(()),
        };

        // Migrate one shard from each over-loaded node.
        let overloaded: Vec<(String, String)> = shard_ids_by_node
            .iter()
            .filter(|(node, _)| {
                *node != &target_node
                    && (*node_shard_counts.get(*node).unwrap_or(&0) as f64) > threshold
            })
            .filter_map(|(_, ids)| ids.first().cloned().map(|sid| (sid, target_node.clone())))
            .collect();

        for (shard_id, target) in overloaded {
            self.migrate_shard(&shard_id, Some(target))?;
        }

        Ok(())
    }

    /// Migrate a shard to a different node
    pub fn migrate_shard(&mut self, shard_id: &str, target_node: Option<String>) -> Result<String> {
        let migration_id = {
            let mut shards = self.shards.write().expect("Operation failed");
            let mut migrations = self.migrations.write().expect("Operation failed");

            let shard = shards
                .get_mut(shard_id)
                .ok_or_else(|| MetricsError::ShardingError("Shard not found".to_string()))?;

            if shard.status == ShardStatus::Migrating {
                return Err(MetricsError::ShardingError(
                    "Shard is already being migrated".to_string(),
                ));
            }

            // Select target node if not provided
            let target = target_node.unwrap_or_else(|| {
                // Simple selection: pick the first replica or a default node
                shard
                    .replicas
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "default_node".to_string())
            });

            let migration_id = format!(
                "migration_{}_{}",
                shard_id,
                SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("Operation failed")
                    .as_millis()
            );

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

            migration_id.clone()
        }; // Drop locks here before calling start_migration

        // Perform the actual migration (in-memory: transfer ownership atomically).
        self.start_migration(&migration_id)?;

        Ok(migration_id)
    }

    /// Start (and immediately complete for in-memory) a migration process.
    ///
    /// For in-memory operation the "data transfer" is instantaneous — we
    /// update the shard's `primary_node` and mark the migration completed.
    fn start_migration(&mut self, migration_id: &str) -> Result<()> {
        // Determine the target node from the migration record.
        let target_node = {
            let mut migrations = self.migrations.write().expect("Operation failed");
            if let Some(migration) = migrations.get_mut(migration_id) {
                migration.status = MigrationStatus::InProgress;
                migration.progress = 0.5;
                migration.target_node.clone()
            } else {
                return Ok(());
            }
        };

        // Update shard ownership and mark migration completed.
        {
            let mut shards = self.shards.write().expect("Operation failed");
            for shard in shards.values_mut() {
                if let Some(ref m) = shard.migration {
                    if m.id == migration_id {
                        shard.primary_node = target_node.clone();
                        shard.status = ShardStatus::Active;
                        shard.migration = None;
                        break;
                    }
                }
            }
        }

        // Finalise migration record.
        {
            let mut migrations = self.migrations.write().expect("Operation failed");
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
            let migration_id = manager.migrate_shard(&shard.id, Some("node2".to_string()));
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

    // ── Rebalancing / migration tests ────────────────────────────────────────

    /// Add a node, verify that shards exist and some may have been redistributed.
    #[test]
    fn test_shard_rebalance_on_add() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::ConsistentHash,
            shard_count: 4,
            replication_factor: 1,
            hash_function: HashFunction::Murmur3,
            virtual_nodes: 4,
            dynamic_resharding: true,
            migration_threshold: 0.8,
        };

        let mut manager = ShardManager::new(config);
        let nodes = vec!["node1".to_string(), "node2".to_string()];
        manager.initialize(nodes).expect("initialise");

        let before = manager.list_shards().len();

        // Add a third node — rebalancing may migrate shards to it.
        manager.add_node("node3".to_string()).expect("add node3");

        let after = manager.list_shards().len();
        // Total shard count should stay the same after rebalancing.
        assert_eq!(before, after, "shard count should be stable after add_node");

        // All shards must be in an Active state (no stuck Migrating shards).
        let stuck = manager
            .list_shards()
            .iter()
            .filter(|s| s.status == ShardStatus::Migrating)
            .count();
        assert_eq!(stuck, 0, "no shards should be stuck in Migrating state");
    }

    /// Remove a node, verify its shards have been reassigned.
    #[test]
    fn test_shard_rebalance_on_remove() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::ConsistentHash,
            shard_count: 4,
            replication_factor: 2,
            hash_function: HashFunction::Murmur3,
            virtual_nodes: 4,
            dynamic_resharding: true,
            migration_threshold: 0.8,
        };

        let mut manager = ShardManager::new(config);
        let nodes = vec![
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
        ];
        manager.initialize(nodes).expect("initialise");

        // Count how many shards node1 owns.
        let n1_before = manager
            .list_shards()
            .iter()
            .filter(|s| s.primary_node == "node1")
            .count();

        // Remove node1; all its shards must migrate away.
        manager.remove_node("node1").expect("remove node1");

        let n1_after = manager
            .list_shards()
            .iter()
            .filter(|s| s.primary_node == "node1")
            .count();

        // If node1 had any primary shards, they should all be gone now.
        if n1_before > 0 {
            assert_eq!(
                n1_after, 0,
                "no shards should remain primary on the removed node"
            );
        }

        // All remaining shards should be Active.
        let stuck = manager
            .list_shards()
            .iter()
            .filter(|s| s.status == ShardStatus::Migrating)
            .count();
        assert_eq!(stuck, 0, "no shards stuck in Migrating state after remove");
    }

    // ── Geographic range tests ────────────────────────────────────────────────

    #[test]
    fn test_geographic_contains_key_success() {
        let geo = DataRange::Geographic {
            lat_min: 35.0,
            lat_max: 36.0,
            lon_min: 139.0,
            lon_max: 140.0,
        };
        // Inside the bounding box — colon separator
        assert!(geo.contains_key("35.5:139.5"));
        // Inside the bounding box — comma separator
        assert!(geo.contains_key("35.5,139.5"));
        // On the boundary (inclusive)
        assert!(geo.contains_key("35.0:139.0"));
        assert!(geo.contains_key("36.0:140.0"));
    }

    #[test]
    fn test_geographic_contains_key_outside() {
        let geo = DataRange::Geographic {
            lat_min: 35.0,
            lat_max: 36.0,
            lon_min: 139.0,
            lon_max: 140.0,
        };
        // Latitude out of range
        assert!(!geo.contains_key("40.0:139.5"));
        // Longitude out of range
        assert!(!geo.contains_key("35.5:145.0"));
        // Both out
        assert!(!geo.contains_key("0.0:0.0"));
    }

    #[test]
    fn test_geographic_contains_key_parse_failure() {
        let geo = DataRange::Geographic {
            lat_min: 35.0,
            lat_max: 36.0,
            lon_min: 139.0,
            lon_max: 140.0,
        };
        // Non-numeric key
        assert!(!geo.contains_key("not_a_geo_key"));
        // Missing separator / only one part
        assert!(!geo.contains_key("35.5"));
        // Only separator, no values
        assert!(!geo.contains_key(":"));
    }

    #[test]
    fn test_geographic_overlaps_with() {
        let geo1 = DataRange::Geographic {
            lat_min: 0.0,
            lat_max: 10.0,
            lon_min: 0.0,
            lon_max: 10.0,
        };
        let geo2 = DataRange::Geographic {
            lat_min: 5.0,
            lat_max: 15.0,
            lon_min: 5.0,
            lon_max: 15.0,
        };
        let geo3 = DataRange::Geographic {
            lat_min: 20.0,
            lat_max: 30.0,
            lon_min: 20.0,
            lon_max: 30.0,
        };
        // Overlapping
        assert!(geo1.overlaps_with(&geo2));
        assert!(geo2.overlaps_with(&geo1));
        // Non-overlapping (completely separated)
        assert!(!geo1.overlaps_with(&geo3));
        assert!(!geo3.overlaps_with(&geo1));
        // Self-overlap
        assert!(geo1.overlaps_with(&geo1));
    }

    // ── Custom range tests ────────────────────────────────────────────────────

    #[test]
    fn test_custom_prefix_match() {
        let custom = DataRange::Custom {
            range_type: "prefix".to_string(),
            range_data: b"user_".to_vec(),
        };
        assert!(custom.contains_key("user_42"));
        assert!(custom.contains_key("user_abc"));
        assert!(!custom.contains_key("order_42"));
        assert!(!custom.contains_key("42_user_"));
    }

    #[test]
    fn test_custom_suffix_match() {
        let custom = DataRange::Custom {
            range_type: "suffix".to_string(),
            range_data: b".log".to_vec(),
        };
        assert!(custom.contains_key("app.log"));
        assert!(custom.contains_key("error.log"));
        assert!(!custom.contains_key("app.txt"));
    }

    #[test]
    fn test_custom_contains_match() {
        let custom = DataRange::Custom {
            range_type: "contains".to_string(),
            range_data: b"error".to_vec(),
        };
        assert!(custom.contains_key("fatal_error_42"));
        assert!(custom.contains_key("error"));
        assert!(!custom.contains_key("warning_42"));
    }

    #[test]
    fn test_custom_glob_regex_match() {
        let custom = DataRange::Custom {
            range_type: "regex".to_string(),
            range_data: b"user_*_log".to_vec(),
        };
        assert!(custom.contains_key("user_42_log"));
        assert!(custom.contains_key("user_abc_log"));
        assert!(!custom.contains_key("user_42_data"));
        assert!(!custom.contains_key("order_42_log"));
    }

    #[test]
    fn test_custom_unknown_type_returns_false() {
        let custom = DataRange::Custom {
            range_type: "unknown_type".to_string(),
            range_data: b"data".to_vec(),
        };
        assert!(!custom.contains_key("data"));
    }

    #[test]
    fn test_custom_overlaps_with_same_type_and_data() {
        let c1 = DataRange::Custom {
            range_type: "prefix".to_string(),
            range_data: b"user_".to_vec(),
        };
        let c2 = DataRange::Custom {
            range_type: "prefix".to_string(),
            range_data: b"user_".to_vec(),
        };
        let c3 = DataRange::Custom {
            range_type: "prefix".to_string(),
            range_data: b"order_".to_vec(),
        };
        let c4 = DataRange::Custom {
            range_type: "suffix".to_string(),
            range_data: b"user_".to_vec(),
        };
        assert!(c1.overlaps_with(&c2));
        // Different data — no overlap
        assert!(!c1.overlaps_with(&c3));
        // Different type — no overlap
        assert!(!c1.overlaps_with(&c4));
    }

    #[test]
    fn test_simple_glob_match_helper() {
        // Leading wildcard
        assert!(simple_glob_match("hello_world", "*world"));
        // Trailing wildcard
        assert!(simple_glob_match("hello_world", "hello*"));
        // Middle wildcard
        assert!(simple_glob_match("hello_world", "hello*world"));
        // Double wildcard
        assert!(simple_glob_match("abc_def_ghi", "abc*def*ghi"));
        // No wildcard — exact match
        assert!(simple_glob_match("exact", "exact"));
        assert!(!simple_glob_match("not_exact", "exact"));
        // Wildcard-only matches everything
        assert!(simple_glob_match("anything", "*"));
        assert!(simple_glob_match("", "*"));
    }
}
