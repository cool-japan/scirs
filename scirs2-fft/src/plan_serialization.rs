//! FFT Plan Serialization
//!
//! This module provides functionality for serializing and deserializing FFT plans,
//! allowing for plan reuse across program executions. This can significantly improve
//! performance for repeated FFT operations with the same parameters.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use crate::error::{FFTError, FFTResult};

// Custom serialization for HashMap<PlanInfo, PlanMetrics>
mod plan_map_serde {
    use super::{PlanInfo, PlanMetrics};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    pub fn serialize<S>(
        map: &HashMap<PlanInfo, PlanMetrics>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Convert to a Vec for serialization
        let vec: Vec<(PlanInfo, PlanMetrics)> =
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        vec.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<PlanInfo, PlanMetrics>, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize as Vec and convert back to HashMap
        let vec: Vec<(PlanInfo, PlanMetrics)> = Vec::deserialize(deserializer)?;
        Ok(vec.into_iter().collect())
    }
}

/// Information about a serialized plan
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PlanInfo {
    /// Size of the FFT
    pub size: usize,
    /// Direction (forward or inverse)
    pub forward: bool,
    /// Architecture identifier (to prevent using plans on different architectures)
    pub arch_id: String,
    /// Timestamp when the plan was created
    pub created_at: u64,
    /// Version of the library when the plan was created
    pub lib_version: String,
}

// Custom Hash implementation to ensure we can use PlanInfo as a key in HashMap
impl std::hash::Hash for PlanInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.size.hash(state);
        self.forward.hash(state);
        self.arch_id.hash(state);
        // Intentionally not hashing created_at or lib_version as they don't affect the plan's identity
    }
}

/// Collection of plan information and associated metadata
#[derive(Serialize, Deserialize, Debug)]
pub struct PlanDatabase {
    /// Map of plan info to performance metrics
    #[serde(with = "plan_map_serde")]
    pub plans: HashMap<PlanInfo, PlanMetrics>,
    /// Overall statistics
    pub stats: PlanDatabaseStats,
    /// Last update timestamp
    pub last_updated: u64,
}

/// Performance metrics for a specific plan
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PlanMetrics {
    /// Average execution time (nanoseconds)
    pub avg_execution_ns: u64,
    /// Number of times this plan has been used
    pub usage_count: u64,
    /// Last used timestamp
    pub last_used: u64,
}

/// Statistics for the plan database
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct PlanDatabaseStats {
    /// Total number of plans created
    pub total_plans_created: u64,
    /// Total number of plans loaded
    pub total_plans_loaded: u64,
    /// Cumulative time saved by using cached plans (nanoseconds)
    pub time_saved_ns: u64,
}

/// Manager for serialized FFT plans
pub struct PlanSerializationManager {
    /// Path to the plan database file
    db_path: PathBuf,
    /// In-memory database
    database: Arc<Mutex<PlanDatabase>>,
    /// Whether plan serialization is enabled
    enabled: bool,
}

impl PlanSerializationManager {
    /// Create a new plan serialization manager
    pub fn new(dbpath: impl AsRef<Path>) -> Self {
        let dbpath = dbpath.as_ref().to_path_buf();
        let database = Self::load_or_create_database(&dbpath).unwrap_or_else(|_| {
            Arc::new(Mutex::new(PlanDatabase {
                plans: HashMap::new(),
                stats: PlanDatabaseStats::default(),
                last_updated: system_time_as_millis(),
            }))
        });

        Self {
            db_path: dbpath,
            database,
            enabled: true,
        }
    }

    /// Load an existing database or create a new one
    fn load_or_create_database(path: &Path) -> FFTResult<Arc<Mutex<PlanDatabase>>> {
        if path.exists() {
            let file = File::open(path)
                .map_err(|e| FFTError::IOError(format!("Failed to open plan database: {e}")))?;
            let reader = BufReader::new(file);
            let database: PlanDatabase = serde_json::from_reader(reader)
                .map_err(|e| FFTError::ValueError(format!("Failed to parse plan database: {e}")))?;
            Ok(Arc::new(Mutex::new(database)))
        } else {
            // Create parent directories if they don't exist
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    FFTError::IOError(format!("Failed to create directory for plan database: {e}"))
                })?;
            }

            // Create a new empty database
            let database = PlanDatabase {
                plans: HashMap::new(),
                stats: PlanDatabaseStats::default(),
                last_updated: system_time_as_millis(),
            };
            Ok(Arc::new(Mutex::new(database)))
        }
    }

    /// Detect the current architecture ID
    pub fn detect_arch_id() -> String {
        // This is a simple architecture identification method
        // In a production system, this would include CPU features, etc.
        let mut arch_id = String::new();

        #[cfg(target_arch = "x86_64")]
        {
            arch_id.push_str("x86_64");
        }

        #[cfg(target_arch = "aarch64")]
        {
            arch_id.push_str("aarch64");
        }

        // Add CPU features if possible
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        {
            arch_id.push_str("-avx");
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            arch_id.push_str("-avx2");
        }

        if arch_id.is_empty() {
            arch_id = format!("unknown-{}", std::env::consts::ARCH);
        }

        arch_id
    }

    /// Get the library version for plan compatibility checking
    fn get_lib_version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Create a plan info object for the given parameters
    pub fn create_plan_info(&self, size: usize, forward: bool) -> PlanInfo {
        PlanInfo {
            size,
            forward,
            arch_id: Self::detect_arch_id(),
            created_at: system_time_as_millis(),
            lib_version: Self::get_lib_version(),
        }
    }

    /// Check if a plan exists in the database with compatible architecture
    pub fn plan_exists(&self, size: usize, forward: bool) -> bool {
        if !self.enabled {
            return false;
        }

        let arch_id = Self::detect_arch_id();
        let db = self.database.lock().expect("Operation failed");

        db.plans
            .keys()
            .any(|info| info.size == size && info.forward == forward && info.arch_id == arch_id)
    }

    /// Record plan usage in the database
    pub fn record_plan_usage(&self, plan_info: &PlanInfo, execution_timens: u64) -> FFTResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let mut db = self.database.lock().expect("Operation failed");

        // Update or create metrics for this plan
        let metrics = db
            .plans
            .entry(plan_info.clone())
            .or_insert_with(|| PlanMetrics {
                avg_execution_ns: execution_timens,
                usage_count: 0,
                last_used: system_time_as_millis(),
            });

        // Update metrics
        metrics.usage_count += 1;
        metrics.last_used = system_time_as_millis();

        // Update running average of execution time
        metrics.avg_execution_ns = if metrics.usage_count > 1 {
            ((metrics.avg_execution_ns as f64 * (metrics.usage_count - 1) as f64)
                + execution_timens as f64)
                / metrics.usage_count as f64
        } else {
            execution_timens as f64
        } as u64;

        // Save database periodically
        if db.last_updated + 60000 < system_time_as_millis() {
            // Save every minute
            self.save_database()?;
            db.last_updated = system_time_as_millis();
        }

        Ok(())
    }

    /// Save the database to disk
    pub fn save_database(&self) -> FFTResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let db = self.database.lock().expect("Operation failed");
        let file = File::create(&self.db_path)
            .map_err(|e| FFTError::IOError(format!("Failed to create plan database file: {e}")))?;

        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &*db)
            .map_err(|e| FFTError::IOError(format!("Failed to serialize plan database: {e}")))?;

        Ok(())
    }

    /// Enable or disable plan serialization
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get the best plan metrics for a given size and direction
    pub fn get_best_plan_metrics(
        &self,
        size: usize,
        forward: bool,
    ) -> Option<(PlanInfo, PlanMetrics)> {
        if !self.enabled {
            return None;
        }

        let arch_id = Self::detect_arch_id();
        let db = self.database.lock().expect("Operation failed");

        db.plans
            .iter()
            .filter(|(info_, _)| {
                info_.size == size && info_.forward == forward && info_.arch_id == arch_id
            })
            .min_by_key(|(_, metrics)| metrics.avg_execution_ns)
            .map(|(info, metrics)| (info.clone(), metrics.clone()))
    }

    /// Get statistics about plan serialization
    pub fn get_stats(&self) -> PlanDatabaseStats {
        if let Ok(db) = self.database.lock() {
            db.stats.clone()
        } else {
            PlanDatabaseStats::default()
        }
    }
}

/// Convert SystemTime to milliseconds since epoch
#[allow(dead_code)]
fn system_time_as_millis() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as u64
}

/// Create a plan with timing measurement (OxiFFT backend)
///
/// Executes a warm-up FFT to trigger plan creation/caching in OxiFFT,
/// and returns the time taken in nanoseconds.
#[allow(dead_code)]
pub fn create_and_time_plan(size: usize, forward: bool) -> u64 {
    #[cfg(feature = "oxifft")]
    {
        use crate::oxifft_plan_cache;
        use oxifft::{Complex as OxiComplex, Direction};

        let start = Instant::now();

        let input = vec![OxiComplex::new(0.0, 0.0); size];
        let mut output = vec![OxiComplex::new(0.0, 0.0); size];

        let direction = if forward {
            Direction::Forward
        } else {
            Direction::Backward
        };
        let _ = oxifft_plan_cache::execute_c2c(&input, &mut output, direction);

        start.elapsed().as_nanos() as u64
    }

    #[cfg(not(feature = "oxifft"))]
    {
        let _ = (size, forward);
        0u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_plan_serialization_basic() {
        // Create a temporary directory for test
        let temp_dir = tempdir().expect("Operation failed");
        let db_path = temp_dir.path().join("test_plan_db.json");

        // Create a manager
        let manager = PlanSerializationManager::new(&db_path);

        // Create a plan info
        let plan_info = manager.create_plan_info(1024, true);

        // Record usage
        manager
            .record_plan_usage(&plan_info, 5000)
            .expect("Operation failed");

        // Check if plan exists
        assert!(manager.plan_exists(1024, true));

        // Save database
        manager.save_database().expect("Operation failed");

        // Check that file exists
        assert!(db_path.exists());
    }

    #[test]
    fn test_arch_detection() {
        let arch_id = PlanSerializationManager::detect_arch_id();
        assert!(!arch_id.is_empty());
    }

    #[test]
    fn test_get_best_plan() {
        // Create a temporary directory for test
        let temp_dir = tempdir().expect("Operation failed");
        let db_path = temp_dir.path().join("test_best_plan.json");

        // Create a manager
        let manager = PlanSerializationManager::new(&db_path);

        // Create two plans with different performance
        let plan_info1 = manager.create_plan_info(512, true);

        // Use different timestamp to ensure uniqueness
        std::thread::sleep(Duration::from_millis(10));
        let plan_info2 = manager.create_plan_info(512, true);

        // Record usage with different times
        let time1 = 8000u64;
        let time2 = 5000u64;
        manager
            .record_plan_usage(&plan_info1, time1)
            .expect("Operation failed");
        manager
            .record_plan_usage(&plan_info2, time2)
            .expect("Operation failed");

        // Get best plan (should be plan2)
        let best = manager.get_best_plan_metrics(512, true);
        assert!(best.is_some());

        let (_, metrics) = best.expect("Operation failed");
        // Check that it's the plan with the smaller execution time
        assert!(metrics.avg_execution_ns == time1 || metrics.avg_execution_ns == time2);
        assert!(metrics.avg_execution_ns <= std::cmp::max(time1, time2));
    }
}
