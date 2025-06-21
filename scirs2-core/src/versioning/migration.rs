//! # Migration Management
//!
//! Comprehensive migration assistance system for API upgrades
//! and version transitions in production environments.

use super::Version;
use crate::error::CoreError;
use std::collections::{HashMap, VecDeque};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Migration plan for upgrading between versions
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MigrationPlan {
    /// Source version
    pub from_version: Version,
    /// Target version
    pub to_version: Version,
    /// Ordered migration steps
    pub steps: Vec<MigrationStep>,
    /// Estimated total effort in hours
    pub estimated_effort: u32,
    /// Risk level of the migration
    pub risk_level: RiskLevel,
    /// Required rollback plan
    pub rollback_plan: Option<RollbackPlan>,
    /// Prerequisites before migration
    pub prerequisites: Vec<String>,
    /// Post-migration validation steps
    pub validation_steps: Vec<String>,
}

/// Individual migration step
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MigrationStep {
    /// Step identifier
    pub id: String,
    /// Step name
    pub name: String,
    /// Step description
    pub description: String,
    /// Step type
    pub step_type: StepType,
    /// Estimated effort in hours
    pub estimated_effort: u32,
    /// Step priority
    pub priority: StepPriority,
    /// Dependencies on other steps
    pub dependencies: Vec<String>,
    /// Automation script (if available)
    pub automation_script: Option<String>,
    /// Manual instructions
    pub manual_instructions: Option<String>,
    /// Validation criteria
    pub validation_criteria: Vec<String>,
    /// Rollback instructions
    pub rollback_instructions: Option<String>,
}

/// Types of migration steps
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StepType {
    /// Code changes required
    CodeChange,
    /// Configuration update
    ConfigurationUpdate,
    /// Database migration
    DatabaseMigration,
    /// Dependency update
    DependencyUpdate,
    /// Feature removal
    FeatureRemoval,
    /// API endpoint change
    ApiChange,
    /// Data format change
    DataFormatChange,
    /// Testing and validation
    Testing,
    /// Documentation update
    Documentation,
}

/// Step priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StepPriority {
    /// Optional step
    Optional,
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical - must be completed
    Critical,
}

/// Risk levels for migrations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RiskLevel {
    /// Low risk migration
    Low,
    /// Medium risk migration
    Medium,
    /// High risk migration
    High,
    /// Critical risk migration
    Critical,
}

/// Rollback plan for migration failures
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollbackPlan {
    /// Rollback steps in order
    pub steps: Vec<RollbackStep>,
    /// Estimated rollback time
    pub estimated_time: u32,
    /// Data backup requirements
    pub backup_requirements: Vec<String>,
    /// Recovery validation steps
    pub recovery_validation: Vec<String>,
}

/// Individual rollback step
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollbackStep {
    /// Step identifier
    pub id: String,
    /// Step description
    pub description: String,
    /// Rollback commands or instructions
    pub instructions: String,
    /// Validation after rollback
    pub validation: Option<String>,
}

/// Migration execution status
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MigrationExecution {
    /// Migration plan being executed
    pub plan: MigrationPlan,
    /// Current step being executed
    pub current_step: Option<String>,
    /// Completed steps
    pub completed_steps: Vec<String>,
    /// Failed steps
    pub failed_steps: Vec<String>,
    /// Execution status
    pub status: ExecutionStatus,
    /// Start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// End time (if completed)
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    /// Execution log
    pub execution_log: Vec<LogEntry>,
}

/// Execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ExecutionStatus {
    /// Migration not started
    NotStarted,
    /// Migration in progress
    InProgress,
    /// Migration completed successfully
    Completed,
    /// Migration failed
    Failed,
    /// Migration paused
    Paused,
    /// Migration rolled back
    RolledBack,
}

/// Log entry for migration execution
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LogEntry {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Log level
    pub level: LogLevel,
    /// Step ID (if applicable)
    pub step_id: Option<String>,
    /// Log message
    pub message: String,
    /// Additional data
    pub data: Option<String>,
}

/// Log levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LogLevel {
    /// Debug information
    Debug,
    /// Informational message
    Info,
    /// Warning message
    Warning,
    /// Error message
    Error,
}

/// Migration manager implementation
pub struct MigrationManager {
    /// Migration templates by version transitions
    migration_templates: HashMap<(Version, Version), MigrationTemplate>,
    /// Active migrations
    active_migrations: HashMap<String, MigrationExecution>,
    /// Migration history
    migration_history: Vec<MigrationExecution>,
}

impl MigrationManager {
    /// Create a new migration manager
    pub fn new() -> Self {
        Self {
            migration_templates: HashMap::new(),
            active_migrations: HashMap::new(),
            migration_history: Vec::new(),
        }
    }

    /// Register a version for migration planning
    pub fn register_version(&mut self, _api_version: &super::ApiVersion) -> Result<(), CoreError> {
        // This would register version-specific migration information
        Ok(())
    }

    /// Create a migration plan between versions
    pub fn create_migration_plan(
        &self,
        from_version: &Version,
        to_version: &Version,
    ) -> Result<MigrationPlan, CoreError> {
        // Check if direct migration template exists
        if let Some(template) = self
            .migration_templates
            .get(&(from_version.clone(), to_version.clone()))
        {
            return Ok(template.create_plan(from_version.clone(), to_version.clone()));
        }

        // Try to find a path through intermediate versions
        if let Some(path) = self.find_migration_path(from_version, to_version)? {
            self.create_multi_step_plan(&path)
        } else {
            self.create_default_plan(from_version, to_version)
        }
    }

    /// Check if a migration path exists
    pub fn has_migration_path(&self, from_version: &Version, to_version: &Version) -> bool {
        // Check direct path
        if self
            .migration_templates
            .contains_key(&(from_version.clone(), to_version.clone()))
        {
            return true;
        }

        // Check if we can find an indirect path
        self.find_migration_path(from_version, to_version)
            .unwrap_or(None)
            .is_some()
    }

    /// Find migration path through intermediate versions
    fn find_migration_path(
        &self,
        from_version: &Version,
        to_version: &Version,
    ) -> Result<Option<Vec<Version>>, CoreError> {
        // BFS to find shortest path
        let mut queue = VecDeque::new();
        let mut visited = std::collections::HashSet::new();
        let mut parent: HashMap<Version, Version> = HashMap::new();

        queue.push_back(from_version.clone());
        visited.insert(from_version.clone());

        while let Some(current) = queue.pop_front() {
            if current == *to_version {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = current;

                while let Some(p) = parent.get(&node) {
                    path.push(node);
                    node = p.clone();
                }
                path.push(from_version.clone());
                path.reverse();

                return Ok(Some(path));
            }

            // Find all versions reachable from current
            for (from, to) in self.migration_templates.keys() {
                if *from == current && !visited.contains(to) {
                    visited.insert(to.clone());
                    parent.insert(to.clone(), current.clone());
                    queue.push_back(to.clone());
                }
            }
        }

        Ok(None)
    }

    /// Create multi-step migration plan
    fn create_multi_step_plan(&self, path: &[Version]) -> Result<MigrationPlan, CoreError> {
        let mut all_steps = Vec::new();
        let mut total_effort = 0;
        let mut max_risk = RiskLevel::Low;

        for window in path.windows(2) {
            if let Some(template) = self
                .migration_templates
                .get(&(window[0].clone(), window[1].clone()))
            {
                let plan = template.create_plan(window[0].clone(), window[1].clone());
                all_steps.extend(plan.steps);
                total_effort += plan.estimated_effort;
                max_risk = max_risk.max(plan.risk_level);
            }
        }

        Ok(MigrationPlan {
            from_version: path.first().unwrap().clone(),
            to_version: path.last().unwrap().clone(),
            steps: all_steps,
            estimated_effort: total_effort,
            risk_level: max_risk,
            rollback_plan: None, // Would be constructed from individual rollback plans
            prerequisites: Vec::new(),
            validation_steps: Vec::new(),
        })
    }

    /// Create default migration plan when no template exists
    fn create_default_plan(
        &self,
        from_version: &Version,
        to_version: &Version,
    ) -> Result<MigrationPlan, CoreError> {
        let mut steps = Vec::new();
        let risk_level = if to_version.major() > from_version.major() {
            RiskLevel::High
        } else if to_version.minor() > from_version.minor() {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        // Add default steps based on version difference
        if to_version.major() > from_version.major() {
            steps.push(MigrationStep {
                id: "major_version_review".to_string(),
                name: "Major Version Review".to_string(),
                description: "Review all breaking changes in major version upgrade".to_string(),
                step_type: StepType::CodeChange,
                estimated_effort: 40,
                priority: StepPriority::Critical,
                dependencies: Vec::new(),
                automation_script: None,
                manual_instructions: Some(
                    "Review changelog and identify all breaking changes".to_string(),
                ),
                validation_criteria: vec![
                    "All breaking changes identified".to_string(),
                    "Migration strategy defined".to_string(),
                ],
                rollback_instructions: Some("Revert to previous version".to_string()),
            });
        }

        if to_version.minor() > from_version.minor() {
            steps.push(MigrationStep {
                id: "minor_version_update".to_string(),
                name: "Minor Version Update".to_string(),
                description: "Update to new minor version with backward compatibility".to_string(),
                step_type: StepType::DependencyUpdate,
                estimated_effort: 8,
                priority: StepPriority::Medium,
                dependencies: Vec::new(),
                automation_script: Some("update_dependencies.sh".to_string()),
                manual_instructions: None,
                validation_criteria: vec!["All tests pass".to_string()],
                rollback_instructions: Some("Revert dependency versions".to_string()),
            });
        }

        // Always add testing step
        steps.push(MigrationStep {
            id: "comprehensive_testing".to_string(),
            name: "Comprehensive Testing".to_string(),
            description: "Run full test suite and integration tests".to_string(),
            step_type: StepType::Testing,
            estimated_effort: 16,
            priority: StepPriority::Critical,
            dependencies: steps.iter().map(|s| s.id.clone()).collect(),
            automation_script: Some("run_tests.sh".to_string()),
            manual_instructions: Some("Verify all functionality works as expected".to_string()),
            validation_criteria: vec![
                "All unit tests pass".to_string(),
                "All integration tests pass".to_string(),
                "Performance benchmarks met".to_string(),
            ],
            rollback_instructions: None,
        });

        let estimated_effort = steps.iter().map(|s| s.estimated_effort).sum();

        Ok(MigrationPlan {
            from_version: from_version.clone(),
            to_version: to_version.clone(),
            steps,
            estimated_effort,
            risk_level,
            rollback_plan: Some(self.create_default_rollback_plan()),
            prerequisites: vec![
                "Create backup of current system".to_string(),
                "Ensure rollback plan is tested".to_string(),
            ],
            validation_steps: vec![
                "Verify system functionality".to_string(),
                "Check performance metrics".to_string(),
                "Validate data integrity".to_string(),
            ],
        })
    }

    /// Create default rollback plan
    fn create_default_rollback_plan(&self) -> RollbackPlan {
        RollbackPlan {
            steps: vec![RollbackStep {
                id: "restore_backup".to_string(),
                description: "Restore from backup".to_string(),
                instructions: "Restore system from pre-migration backup".to_string(),
                validation: Some("Verify system is functioning".to_string()),
            }],
            estimated_time: 30, // 30 minutes
            backup_requirements: vec![
                "Full system backup".to_string(),
                "Database backup".to_string(),
                "Configuration backup".to_string(),
            ],
            recovery_validation: vec![
                "System startup successful".to_string(),
                "All services running".to_string(),
                "Data integrity verified".to_string(),
            ],
        }
    }

    /// Start migration execution
    pub fn start_migration(
        &mut self,
        plan: MigrationPlan,
        execution_id: String,
    ) -> Result<(), CoreError> {
        let execution = MigrationExecution {
            plan,
            current_step: None,
            completed_steps: Vec::new(),
            failed_steps: Vec::new(),
            status: ExecutionStatus::NotStarted,
            start_time: chrono::Utc::now(),
            end_time: None,
            execution_log: Vec::new(),
        };

        self.active_migrations.insert(execution_id, execution);
        Ok(())
    }

    /// Get migration execution status
    pub fn get_migration_status(&self, execution_id: &str) -> Option<&MigrationExecution> {
        self.active_migrations.get(execution_id)
    }

    /// Clean up old migration plans
    pub fn cleanup_old_plans(&mut self) -> Result<usize, CoreError> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(30);
        let initial_count = self.migration_history.len();

        self.migration_history
            .retain(|execution| execution.start_time > cutoff);

        Ok(initial_count - self.migration_history.len())
    }
}

impl Default for MigrationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Migration template for version transitions
struct MigrationTemplate {
    /// Template steps
    steps: Vec<MigrationStepTemplate>,
    /// Base effort estimate
    base_effort: u32,
    /// Risk level
    risk_level: RiskLevel,
}

impl MigrationTemplate {
    fn create_plan(&self, from_version: Version, to_version: Version) -> MigrationPlan {
        let steps = self
            .steps
            .iter()
            .map(|template| template.create_step())
            .collect();

        MigrationPlan {
            from_version,
            to_version,
            steps,
            estimated_effort: self.base_effort,
            risk_level: self.risk_level,
            rollback_plan: None,
            prerequisites: Vec::new(),
            validation_steps: Vec::new(),
        }
    }
}

/// Template for migration steps
struct MigrationStepTemplate {
    id: String,
    name: String,
    description: String,
    step_type: StepType,
    estimated_effort: u32,
    priority: StepPriority,
}

impl MigrationStepTemplate {
    fn create_step(&self) -> MigrationStep {
        MigrationStep {
            id: self.id.clone(),
            name: self.name.clone(),
            description: self.description.clone(),
            step_type: self.step_type,
            estimated_effort: self.estimated_effort,
            priority: self.priority,
            dependencies: Vec::new(),
            automation_script: None,
            manual_instructions: None,
            validation_criteria: Vec::new(),
            rollback_instructions: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_manager_creation() {
        let manager = MigrationManager::new();
        assert!(manager.migration_templates.is_empty());
        assert!(manager.active_migrations.is_empty());
    }

    #[test]
    fn test_default_migration_plan() {
        let manager = MigrationManager::new();
        let from_version = Version::new(1, 0, 0);
        let to_version = Version::new(2, 0, 0);

        let plan = manager
            .create_migration_plan(&from_version, &to_version)
            .unwrap();
        assert_eq!(plan.from_version, from_version);
        assert_eq!(plan.to_version, to_version);
        assert!(!plan.steps.is_empty());
        assert_eq!(plan.risk_level, RiskLevel::High); // Major version change
    }

    #[test]
    fn test_minor_version_migration() {
        let manager = MigrationManager::new();
        let from_version = Version::new(1, 0, 0);
        let to_version = Version::new(1, 1, 0);

        let plan = manager
            .create_migration_plan(&from_version, &to_version)
            .unwrap();
        assert_eq!(plan.risk_level, RiskLevel::Medium); // Minor version change
    }

    #[test]
    fn test_patch_version_migration() {
        let manager = MigrationManager::new();
        let from_version = Version::new(1, 0, 0);
        let to_version = Version::new(1, 0, 1);

        let plan = manager
            .create_migration_plan(&from_version, &to_version)
            .unwrap();
        assert_eq!(plan.risk_level, RiskLevel::Low); // Patch version change
    }

    #[test]
    fn test_migration_execution() {
        let mut manager = MigrationManager::new();
        let plan = MigrationPlan {
            from_version: Version::new(1, 0, 0),
            to_version: Version::new(1, 1, 0),
            steps: Vec::new(),
            estimated_effort: 8,
            risk_level: RiskLevel::Medium,
            rollback_plan: None,
            prerequisites: Vec::new(),
            validation_steps: Vec::new(),
        };

        let execution_id = "test_migration_123".to_string();
        manager.start_migration(plan, execution_id.clone()).unwrap();

        let status = manager.get_migration_status(&execution_id);
        assert!(status.is_some());
        assert_eq!(status.unwrap().status, ExecutionStatus::NotStarted);
    }

    #[test]
    fn test_step_priority_ordering() {
        assert!(StepPriority::Critical > StepPriority::High);
        assert!(StepPriority::High > StepPriority::Medium);
        assert!(StepPriority::Medium > StepPriority::Low);
        assert!(StepPriority::Low > StepPriority::Optional);
    }

    #[test]
    fn test_risk_level_ordering() {
        assert!(RiskLevel::Critical > RiskLevel::High);
        assert!(RiskLevel::High > RiskLevel::Medium);
        assert!(RiskLevel::Medium > RiskLevel::Low);
    }
}
