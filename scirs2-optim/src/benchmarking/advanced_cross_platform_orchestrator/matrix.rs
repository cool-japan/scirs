//! Test matrix generation for cross-platform testing
//!
//! This module generates comprehensive test matrices combining platforms,
//! Rust versions, features, and optimization levels for thorough testing.

use crate::error::Result;
use std::time::Duration;
use std::collections::HashMap;

use super::config::*;
use super::types::*;

/// Test matrix generator
#[derive(Debug)]
pub struct TestMatrixGenerator {
    config: TestMatrixConfig,
}

impl TestMatrixGenerator {
    /// Create new test matrix generator
    pub fn new(config: TestMatrixConfig) -> Result<Self> {
        Ok(Self { config })
    }

    /// Generate comprehensive test matrix
    pub fn generate_matrix(&self) -> Result<Vec<TestMatrixEntry>> {
        let mut matrix = Vec::new();
        let mut entry_id = 0;

        for platform_spec in &self.config.platforms {
            for rust_version in &self.config.rust_versions {
                for feature_combo in &self.config.feature_combinations {
                    for optimization in &self.config.optimization_levels {
                        for build_profile in &self.config.build_profiles {
                            for scenario in &self.config.test_scenarios {
                                entry_id += 1;

                                let entry = TestMatrixEntry {
                                    id: format!("test_{}", entry_id),
                                    platform: platform_spec.target.clone(),
                                    rust_version: rust_version.clone(),
                                    features: feature_combo.enabled_features.clone(),
                                    optimization: optimization.clone(),
                                    build_profile: build_profile.name.clone(),
                                    scenarios: vec![scenario.name.clone()],
                                    priority: platform_spec.priority,
                                    required_for_release: platform_spec.required_for_release,
                                    estimated_duration: scenario.timeout,
                                    resource_requirements: self.calculate_resource_requirements(
                                        &platform_spec.resource_requirements
                                    ),
                                };

                                matrix.push(entry);
                            }
                        }
                    }
                }
            }
        }

        // Sort by priority (higher priority first)
        matrix.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(matrix)
    }

    /// Calculate resource requirements for test entry
    fn calculate_resource_requirements(
        &self,
        platform_reqs: &PlatformResourceRequirements,
    ) -> HashMap<ResourceType, f64> {
        let mut requirements = HashMap::new();

        requirements.insert(ResourceType::CPU, platform_reqs.cpu_cores as f64);
        requirements.insert(ResourceType::Memory, platform_reqs.memory_mb as f64);
        requirements.insert(ResourceType::Storage, platform_reqs.disk_mb as f64);

        if platform_reqs.gpu_required {
            requirements.insert(ResourceType::GPU, 1.0);
        }

        if let Some(bandwidth) = platform_reqs.network_bandwidth {
            requirements.insert(ResourceType::Network, bandwidth);
        }

        requirements
    }

    /// Filter matrix by criteria
    pub fn filter_matrix(&self, matrix: Vec<TestMatrixEntry>, filter: MatrixFilter) -> Vec<TestMatrixEntry> {
        matrix.into_iter().filter(|entry| {
            self.matches_filter(entry, &filter)
        }).collect()
    }

    /// Check if entry matches filter
    fn matches_filter(&self, entry: &TestMatrixEntry, filter: &MatrixFilter) -> bool {
        if let Some(ref platforms) = filter.platforms {
            if !platforms.contains(&entry.platform) {
                return false;
            }
        }

        if let Some(min_priority) = filter.min_priority {
            if entry.priority < min_priority {
                return false;
            }
        }

        if filter.required_for_release_only && !entry.required_for_release {
            return false;
        }

        true
    }

    /// Get matrix statistics
    pub fn get_matrix_statistics(&self, matrix: &[TestMatrixEntry]) -> MatrixStatistics {
        let total_entries = matrix.len();
        let total_platforms = matrix.iter()
            .map(|e| &e.platform)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let mut platform_counts = HashMap::new();
        for entry in matrix {
            *platform_counts.entry(entry.platform.clone()).or_insert(0) += 1;
        }

        let estimated_total_time: Duration = matrix.iter()
            .map(|e| e.estimated_duration)
            .sum();

        MatrixStatistics {
            total_entries,
            total_platforms,
            platform_counts,
            estimated_total_time,
            required_for_release: matrix.iter().filter(|e| e.required_for_release).count(),
        }
    }
}

/// Matrix filter criteria
#[derive(Debug, Clone)]
pub struct MatrixFilter {
    pub platforms: Option<Vec<PlatformTarget>>,
    pub min_priority: Option<u8>,
    pub required_for_release_only: bool,
}

/// Matrix generation statistics
#[derive(Debug, Clone)]
pub struct MatrixStatistics {
    pub total_entries: usize,
    pub total_platforms: usize,
    pub platform_counts: HashMap<PlatformTarget, usize>,
    pub estimated_total_time: Duration,
    pub required_for_release: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_generation() {
        let config = TestMatrixConfig::default();
        let generator = TestMatrixGenerator::new(config).unwrap();
        let matrix = generator.generate_matrix().unwrap();

        assert!(!matrix.is_empty());

        // Check that entries are sorted by priority
        for window in matrix.windows(2) {
            assert!(window[0].priority >= window[1].priority);
        }
    }

    #[test]
    fn test_matrix_filtering() {
        let config = TestMatrixConfig::default();
        let generator = TestMatrixGenerator::new(config).unwrap();
        let matrix = generator.generate_matrix().unwrap();

        let filter = MatrixFilter {
            platforms: Some(vec![PlatformTarget::LinuxX86_64]),
            min_priority: Some(9),
            required_for_release_only: true,
        };

        let filtered = generator.filter_matrix(matrix, filter);

        for entry in &filtered {
            assert_eq!(entry.platform, PlatformTarget::LinuxX86_64);
            assert!(entry.priority >= 9);
            assert!(entry.required_for_release);
        }
    }
}