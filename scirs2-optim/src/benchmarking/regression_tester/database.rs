//! Performance database operations for regression testing
//!
//! This module handles persistence and retrieval of performance history data,
//! including database management, record storage, and history maintenance.

use crate::benchmarking::regression_tester::types::{DatabaseMetadata, PerformanceRecord};
use crate::error::Result;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Historical performance database
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceDatabase<A: Float> {
    /// Performance history by optimizer and test
    history: HashMap<String, VecDeque<PerformanceRecord<A>>>,
    /// Database metadata
    metadata: DatabaseMetadata,
}

impl<A: Float + Debug + Serialize + for<'de> Deserialize<'de>> PerformanceDatabase<A> {
    /// Create a new empty performance database
    pub fn new() -> Self {
        Self {
            history: HashMap::new(),
            metadata: DatabaseMetadata {
                version: "1.0".to_string(),
                created_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                last_updated: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                total_records: 0,
            },
        }
    }

    /// Load database from disk
    pub fn load(basedir: &Path) -> Result<Self> {
        let db_path = basedir.join("performance_db.json");
        if db_path.exists() {
            let data = fs::read_to_string(&db_path)?;
            let db = serde_json::from_str(&data)?;
            Ok(db)
        } else {
            Ok(Self::new())
        }
    }

    /// Save database to disk
    pub fn save(&self, basedir: &Path) -> Result<()> {
        fs::create_dir_all(basedir)?;
        let db_path = basedir.join("performance_db.json");
        let data = serde_json::to_string_pretty(self)?;
        fs::write(&db_path, data)?;
        Ok(())
    }

    /// Add a performance record with history size management
    pub fn add_record(&mut self, key: String, record: PerformanceRecord<A>) {
        self.add_record_with_limit(key, record, 1000);
    }

    /// Add a performance record with custom history limit
    pub fn add_record_with_limit(
        &mut self,
        key: String,
        record: PerformanceRecord<A>,
        max_history: usize,
    ) {
        let history = self.history.entry(key).or_insert_with(VecDeque::new);
        history.push_back(record);

        // Maintain reasonable history size
        while history.len() > max_history {
            history.pop_front();
        }

        self.metadata.total_records += 1;
        self.metadata.last_updated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Get performance history for a specific key
    pub fn get_history(&self, key: &str) -> Option<&VecDeque<PerformanceRecord<A>>> {
        self.history.get(key)
    }

    /// Get mutable performance history for a specific key
    pub fn get_history_mut(&mut self, key: &str) -> Option<&mut VecDeque<PerformanceRecord<A>>> {
        self.history.get_mut(key)
    }

    /// Get all keys with performance history
    pub fn get_keys(&self) -> Vec<&String> {
        self.history.keys().collect()
    }

    /// Get recent records for a key (last N records)
    pub fn get_recent_records(&self, key: &str, count: usize) -> Vec<&PerformanceRecord<A>> {
        if let Some(history) = self.history.get(key) {
            history.iter().rev().take(count).collect()
        } else {
            Vec::new()
        }
    }

    /// Remove old records based on timestamp (older than given timestamp)
    pub fn cleanup_old_records(&mut self, cutoff_timestamp: u64) -> usize {
        let mut removed_count = 0;

        for history in self.history.values_mut() {
            let original_len = history.len();
            history.retain(|record| record.timestamp >= cutoff_timestamp);
            removed_count += original_len - history.len();
        }

        // Remove empty entries
        self.history.retain(|_, history| !history.is_empty());

        // Update metadata
        self.metadata.total_records = self.metadata.total_records.saturating_sub(removed_count);
        self.metadata.last_updated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        removed_count
    }

    /// Get total number of records across all keys
    pub fn total_records(&self) -> usize {
        self.history.values().map(|h| h.len()).sum()
    }

    /// Get number of unique keys
    pub fn key_count(&self) -> usize {
        self.history.len()
    }

    /// Get database metadata
    pub fn metadata(&self) -> &DatabaseMetadata {
        &self.metadata
    }

    /// Check if database has any data
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Get records for a specific commit hash
    pub fn get_records_by_commit(&self, commit_hash: &str) -> Vec<(&String, &PerformanceRecord<A>)> {
        let mut results = Vec::new();
        for (key, history) in &self.history {
            for record in history {
                if let Some(ref hash) = record.commit_hash {
                    if hash == commit_hash {
                        results.push((key, record));
                    }
                }
            }
        }
        results
    }

    /// Get records for a specific branch
    pub fn get_records_by_branch(&self, branch: &str) -> Vec<(&String, &PerformanceRecord<A>)> {
        let mut results = Vec::new();
        for (key, history) in &self.history {
            for record in history {
                if let Some(ref record_branch) = record.branch {
                    if record_branch == branch {
                        results.push((key, record));
                    }
                }
            }
        }
        results
    }

    /// Get records within a time range
    pub fn get_records_by_time_range(
        &self,
        start_timestamp: u64,
        end_timestamp: u64,
    ) -> Vec<(&String, &PerformanceRecord<A>)> {
        let mut results = Vec::new();
        for (key, history) in &self.history {
            for record in history {
                if record.timestamp >= start_timestamp && record.timestamp <= end_timestamp {
                    results.push((key, record));
                }
            }
        }
        results
    }

    /// Export database to JSON string
    pub fn export_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Import database from JSON string
    pub fn import_json(json_data: &str) -> Result<Self> {
        Ok(serde_json::from_str(json_data)?)
    }

    /// Merge another database into this one
    pub fn merge(&mut self, other: PerformanceDatabase<A>) {
        for (key, mut other_history) in other.history {
            let history = self.history.entry(key).or_insert_with(VecDeque::new);

            // Merge histories and sort by timestamp
            history.append(&mut other_history);
            let mut sorted_records: Vec<_> = history.drain(..).collect();
            sorted_records.sort_by_key(|record| record.timestamp);

            // Rebuild the deque
            *history = sorted_records.into();
        }

        // Update metadata
        self.metadata.total_records = self.total_records();
        self.metadata.last_updated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Compact database by removing duplicate records
    pub fn compact(&mut self) -> usize {
        let mut removed_count = 0;

        for history in self.history.values_mut() {
            let original_len = history.len();

            // Remove duplicates based on timestamp and commit hash
            let mut seen = std::collections::HashSet::new();
            history.retain(|record| {
                let key = (record.timestamp, &record.commit_hash);
                if seen.contains(&key) {
                    false
                } else {
                    seen.insert(key);
                    true
                }
            });

            removed_count += original_len - history.len();
        }

        // Update metadata
        self.metadata.total_records = self.total_records();
        self.metadata.last_updated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        removed_count
    }
}

impl<A: Float + Debug + Serialize + for<'de> Deserialize<'de>> Default for PerformanceDatabase<A> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarking::regression_tester::config::TestEnvironment;
    use crate::benchmarking::regression_tester::types::PerformanceMetrics;

    fn create_test_record() -> PerformanceRecord<f64> {
        PerformanceRecord {
            timestamp: 1000000,
            commit_hash: Some("abc123".to_string()),
            branch: Some("main".to_string()),
            environment: TestEnvironment::default(),
            metrics: PerformanceMetrics::default(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_database_creation() {
        let db: PerformanceDatabase<f64> = PerformanceDatabase::new();
        assert!(db.is_empty());
        assert_eq!(db.total_records(), 0);
        assert_eq!(db.key_count(), 0);
    }

    #[test]
    fn test_add_record() {
        let mut db: PerformanceDatabase<f64> = PerformanceDatabase::new();
        let record = create_test_record();

        db.add_record("test_key".to_string(), record);

        assert!(!db.is_empty());
        assert_eq!(db.total_records(), 1);
        assert_eq!(db.key_count(), 1);
        assert!(db.get_history("test_key").is_some());
    }

    #[test]
    fn test_history_limit() {
        let mut db: PerformanceDatabase<f64> = PerformanceDatabase::new();

        // Add records beyond the limit
        for i in 0..1005 {
            let mut record = create_test_record();
            record.timestamp = i;
            db.add_record_with_limit("test_key".to_string(), record, 1000);
        }

        // Should maintain limit
        assert_eq!(db.get_history("test_key").unwrap().len(), 1000);
    }

    #[test]
    fn test_get_recent_records() {
        let mut db: PerformanceDatabase<f64> = PerformanceDatabase::new();

        // Add 5 records
        for i in 0..5 {
            let mut record = create_test_record();
            record.timestamp = i;
            db.add_record("test_key".to_string(), record);
        }

        let recent = db.get_recent_records("test_key", 3);
        assert_eq!(recent.len(), 3);
        // Should be in reverse order (most recent first)
        assert_eq!(recent[0].timestamp, 4);
        assert_eq!(recent[1].timestamp, 3);
        assert_eq!(recent[2].timestamp, 2);
    }

    #[test]
    fn test_cleanup_old_records() {
        let mut db: PerformanceDatabase<f64> = PerformanceDatabase::new();

        // Add records with different timestamps
        for i in 0..10 {
            let mut record = create_test_record();
            record.timestamp = i * 1000;
            db.add_record("test_key".to_string(), record);
        }

        // Remove records older than 5000
        let removed = db.cleanup_old_records(5000);
        assert_eq!(removed, 5); // Records 0-4000 should be removed
        assert_eq!(db.total_records(), 5); // Records 5000-9000 should remain
    }

    #[test]
    fn test_get_records_by_commit() {
        let mut db: PerformanceDatabase<f64> = PerformanceDatabase::new();

        let mut record1 = create_test_record();
        record1.commit_hash = Some("commit1".to_string());
        db.add_record("key1".to_string(), record1);

        let mut record2 = create_test_record();
        record2.commit_hash = Some("commit2".to_string());
        db.add_record("key2".to_string(), record2);

        let results = db.get_records_by_commit("commit1");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "key1");
    }

    #[test]
    fn test_compact_database() {
        let mut db: PerformanceDatabase<f64> = PerformanceDatabase::new();

        // Add duplicate records (same timestamp and commit)
        for _ in 0..3 {
            let record = create_test_record();
            db.add_record("test_key".to_string(), record);
        }

        assert_eq!(db.total_records(), 3);

        let removed = db.compact();
        assert_eq!(removed, 2); // 2 duplicates removed
        assert_eq!(db.total_records(), 1); // 1 unique record remains
    }
}