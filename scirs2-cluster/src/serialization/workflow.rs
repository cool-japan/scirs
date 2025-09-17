//! Workflow management for clustering operations
//!
//! This module provides workflow management capabilities for complex
//! clustering pipelines with automatic saving, loading, and state management.

use crate::error::{ClusteringError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::core::SerializableModel;

/// Comprehensive clustering workflow with state management
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClusteringWorkflow {
    /// Workflow identifier
    pub workflow_id: String,
    /// Current step in the workflow
    pub current_step: usize,
    /// All steps in the workflow
    pub steps: Vec<TrainingStep>,
    /// Current algorithm state
    pub current_state: AlgorithmState,
    /// Workflow configuration
    pub config: WorkflowConfig,
    /// Execution history
    pub execution_history: Vec<ExecutionRecord>,
    /// Intermediate results
    pub intermediate_results: HashMap<String, serde_json::Value>,
}

/// State of algorithm execution
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum AlgorithmState {
    /// Not started
    NotStarted,
    /// Currently running
    Running {
        /// Current iteration
        iteration: usize,
        /// Start time
        start_time: u64,
        /// Progress percentage (0-100)
        progress: f32,
    },
    /// Completed successfully
    Completed {
        /// Total iterations
        iterations: usize,
        /// Total execution time in seconds
        execution_time: f64,
        /// Final metrics
        final_metrics: HashMap<String, f64>,
    },
    /// Failed with error
    Failed {
        /// Error message
        error: String,
        /// Failure time
        failure_time: u64,
    },
    /// Paused
    Paused {
        /// Pause time
        pause_time: u64,
        /// Current iteration when paused
        paused_at_iteration: usize,
    },
}

/// Individual training step in workflow
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingStep {
    /// Step name/identifier
    pub name: String,
    /// Algorithm to use
    pub algorithm: String,
    /// Step parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Dependencies on other steps
    pub dependencies: Vec<String>,
    /// Whether step is completed
    pub completed: bool,
    /// Step execution time
    pub execution_time: Option<f64>,
    /// Step results
    pub results: Option<serde_json::Value>,
}

/// Workflow configuration
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct WorkflowConfig {
    /// Auto-save interval in seconds
    pub auto_save_interval: Option<u64>,
    /// Maximum retries for failed steps
    pub max_retries: usize,
    /// Timeout for individual steps in seconds
    pub step_timeout: Option<u64>,
    /// Enable parallel execution where possible
    pub parallel_execution: bool,
    /// Checkpoint directory
    pub checkpoint_dir: Option<PathBuf>,
}

/// Execution record for audit trail
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExecutionRecord {
    /// Timestamp of execution
    pub timestamp: u64,
    /// Step that was executed
    pub step_name: String,
    /// Action performed
    pub action: String,
    /// Execution result
    pub result: ExecutionResult,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Result of step execution
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ExecutionResult {
    /// Successful execution
    Success {
        /// Execution time in seconds
        duration: f64,
        /// Output data
        output: Option<serde_json::Value>,
    },
    /// Failed execution
    Failure {
        /// Error message
        error: String,
        /// Error code
        error_code: Option<String>,
    },
    /// Skipped execution
    Skipped {
        /// Reason for skipping
        reason: String,
    },
}

impl ClusteringWorkflow {
    /// Create a new workflow
    pub fn new(workflow_id: String, config: WorkflowConfig) -> Self {
        Self {
            workflow_id,
            current_step: 0,
            steps: Vec::new(),
            current_state: AlgorithmState::NotStarted,
            config,
            execution_history: Vec::new(),
            intermediate_results: HashMap::new(),
        }
    }

    /// Add a step to the workflow
    pub fn add_step(&mut self, step: TrainingStep) {
        self.steps.push(step);
    }

    /// Execute the workflow
    pub fn execute(&mut self) -> Result<()> {
        self.current_state = AlgorithmState::Running {
            iteration: 0,
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            progress: 0.0,
        };

        let start_time = std::time::Instant::now();

        let steps_len = self.steps.len();
        for i in 0..steps_len {
            self.current_step = i;

            // Check dependencies (clone to avoid borrow issues)
            let dependencies = self.steps[i].dependencies.clone();
            if !self.check_dependencies(&dependencies)? {
                return Err(ClusteringError::InvalidInput(format!(
                    "Dependencies not satisfied for step: {}",
                    self.steps[i].name
                )));
            }

            // Execute step
            let step_start = std::time::Instant::now();
            let step_clone = self.steps[i].clone();
            let result = self.execute_step(&step_clone)?;
            let step_duration = step_start.elapsed().as_secs_f64();

            // Update step results
            self.steps[i].completed = true;
            self.steps[i].execution_time = Some(step_duration);
            self.steps[i].results = Some(result.clone());

            // Record execution
            let step_name = self.steps[i].name.clone();
            self.record_execution(
                &step_name,
                "execute",
                ExecutionResult::Success {
                    duration: step_duration,
                    output: Some(result),
                },
            );

            // Update progress
            let progress = ((i + 1) as f32 / steps_len as f32) * 100.0;
            self.update_progress(progress);

            // Auto-save if configured
            if let Some(interval) = self.config.auto_save_interval {
                if step_duration > interval as f64 {
                    self.save_checkpoint()?;
                }
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();
        self.current_state = AlgorithmState::Completed {
            iterations: self.steps.len(),
            execution_time: total_time,
            final_metrics: self.collect_final_metrics(),
        };

        Ok(())
    }

    /// Execute a single step
    fn execute_step(&mut self, step: &TrainingStep) -> Result<serde_json::Value> {
        // This would dispatch to the appropriate algorithm implementation
        // For now, return a placeholder result
        use serde_json::json;

        let result = match step.algorithm.as_str() {
            "kmeans" => {
                json!({
                    "algorithm": "kmeans",
                    "centroids": [[0.0, 0.0], [1.0, 1.0]],
                    "inertia": 0.5,
                    "iterations": 10
                })
            }
            "dbscan" => {
                json!({
                    "algorithm": "dbscan",
                    "n_clusters": 2,
                    "core_samples": [0, 1, 2],
                    "noise_points": []
                })
            }
            _ => {
                return Err(ClusteringError::InvalidInput(format!(
                    "Unknown algorithm: {}",
                    step.algorithm
                )));
            }
        };

        // Store intermediate result
        self.intermediate_results
            .insert(step.name.clone(), result.clone());

        Ok(result)
    }

    /// Check if step dependencies are satisfied
    fn check_dependencies(&self, dependencies: &[String]) -> Result<bool> {
        for dep in dependencies {
            if !self.steps.iter().any(|s| s.name == *dep && s.completed) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Update workflow progress
    fn update_progress(&mut self, progress: f32) {
        if let AlgorithmState::Running {
            iteration,
            start_time,
            ..
        } = &mut self.current_state
        {
            self.current_state = AlgorithmState::Running {
                iteration: *iteration + 1,
                start_time: *start_time,
                progress,
            };
        }
    }

    /// Record execution event
    fn record_execution(&mut self, step_name: &str, action: &str, result: ExecutionResult) {
        let record = ExecutionRecord {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            step_name: step_name.to_string(),
            action: action.to_string(),
            result,
            metadata: HashMap::new(),
        };

        self.execution_history.push(record);
    }

    /// Collect final metrics from all completed steps
    fn collect_final_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        let total_steps = self.steps.len() as f64;
        let completed_steps = self.steps.iter().filter(|s| s.completed).count() as f64;
        let total_time: f64 = self.steps.iter().filter_map(|s| s.execution_time).sum();

        metrics.insert("total_steps".to_string(), total_steps);
        metrics.insert("completed_steps".to_string(), completed_steps);
        metrics.insert("completion_rate".to_string(), completed_steps / total_steps);
        metrics.insert("total_execution_time".to_string(), total_time);

        metrics
    }

    /// Save workflow checkpoint
    pub fn save_checkpoint(&self) -> Result<()> {
        if let Some(ref checkpoint_dir) = self.config.checkpoint_dir {
            std::fs::create_dir_all(checkpoint_dir)
                .map_err(|e| ClusteringError::InvalidInput(e.to_string()))?;

            let checkpoint_file =
                checkpoint_dir.join(format!("{}_checkpoint.json", self.workflow_id));
            self.save_to_file(checkpoint_file)?;
        }

        Ok(())
    }

    /// Load workflow from checkpoint
    pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::load_from_file(path)
    }

    /// Pause workflow execution
    pub fn pause(&mut self) {
        let current_iteration = match &self.current_state {
            AlgorithmState::Running { iteration, .. } => *iteration,
            _ => 0,
        };

        self.current_state = AlgorithmState::Paused {
            pause_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            paused_at_iteration: current_iteration,
        };
    }

    /// Resume workflow execution
    pub fn resume(&mut self) -> Result<()> {
        if let AlgorithmState::Paused {
            paused_at_iteration,
            ..
        } = &self.current_state
        {
            self.current_state = AlgorithmState::Running {
                iteration: *paused_at_iteration,
                start_time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                progress: (*paused_at_iteration as f32 / self.steps.len() as f32) * 100.0,
            };

            // Continue execution from where we left off
            self.execute_remaining_steps()
        } else {
            Err(ClusteringError::InvalidInput(
                "Workflow is not in paused state".to_string(),
            ))
        }
    }

    /// Execute remaining steps after resume
    fn execute_remaining_steps(&mut self) -> Result<()> {
        let start_index = self.current_step;

        let steps_len = self.steps.len();
        for i in start_index..steps_len {
            if !self.steps[i].completed {
                self.current_step = i;
                let step_start = std::time::Instant::now();
                let step_clone = self.steps[i].clone();
                let result = self.execute_step(&step_clone)?;
                let step_duration = step_start.elapsed().as_secs_f64();

                self.steps[i].completed = true;
                self.steps[i].execution_time = Some(step_duration);
                self.steps[i].results = Some(result.clone());

                let step_name = self.steps[i].name.clone();
                self.record_execution(
                    &step_name,
                    "resume_execute",
                    ExecutionResult::Success {
                        duration: step_duration,
                        output: Some(result),
                    },
                );
            }
        }

        let final_metrics = self.collect_final_metrics();
        self.current_state = AlgorithmState::Completed {
            iterations: self.steps.len(),
            execution_time: final_metrics
                .get("total_execution_time")
                .copied()
                .unwrap_or(0.0),
            final_metrics,
        };

        Ok(())
    }

    /// Get workflow progress as percentage
    pub fn get_progress(&self) -> f32 {
        match &self.current_state {
            AlgorithmState::Running { progress, .. } => *progress,
            AlgorithmState::Completed { .. } => 100.0,
            AlgorithmState::Failed { .. } => 0.0,
            AlgorithmState::Paused {
                paused_at_iteration,
                ..
            } => (*paused_at_iteration as f32 / self.steps.len() as f32) * 100.0,
            AlgorithmState::NotStarted => 0.0,
        }
    }

    /// Get detailed workflow status
    pub fn get_status(&self) -> WorkflowStatus {
        WorkflowStatus {
            workflow_id: self.workflow_id.clone(),
            current_step: self.current_step,
            total_steps: self.steps.len(),
            state: self.current_state.clone(),
            progress: self.get_progress(),
            completed_steps: self.steps.iter().filter(|s| s.completed).count(),
        }
    }
}

impl SerializableModel for ClusteringWorkflow {}

/// Workflow status information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorkflowStatus {
    /// Workflow identifier
    pub workflow_id: String,
    /// Current step index
    pub current_step: usize,
    /// Total number of steps
    pub total_steps: usize,
    /// Current algorithm state
    pub state: AlgorithmState,
    /// Progress percentage
    pub progress: f32,
    /// Number of completed steps
    pub completed_steps: usize,
}

/// Workflow manager for handling multiple workflows
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClusteringWorkflowManager {
    /// Active workflows
    pub workflows: HashMap<String, ClusteringWorkflow>,
    /// Manager configuration
    pub config: ManagerConfig,
}

/// Configuration for workflow manager
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ManagerConfig {
    /// Maximum number of concurrent workflows
    pub max_concurrent_workflows: usize,
    /// Default checkpoint directory
    pub default_checkpoint_dir: Option<PathBuf>,
    /// Auto-save interval for all workflows
    pub global_auto_save_interval: Option<u64>,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_workflows: 10,
            default_checkpoint_dir: None,
            global_auto_save_interval: Some(300), // 5 minutes
        }
    }
}

impl ClusteringWorkflowManager {
    /// Create a new workflow manager
    pub fn new(config: ManagerConfig) -> Self {
        Self {
            workflows: HashMap::new(),
            config,
        }
    }

    /// Add a workflow to the manager
    pub fn add_workflow(&mut self, workflow: ClusteringWorkflow) -> Result<()> {
        if self.workflows.len() >= self.config.max_concurrent_workflows {
            return Err(ClusteringError::InvalidInput(
                "Maximum number of concurrent workflows reached".to_string(),
            ));
        }

        self.workflows
            .insert(workflow.workflow_id.clone(), workflow);
        Ok(())
    }

    /// Get workflow by ID
    pub fn get_workflow(&self, workflow_id: &str) -> Option<&ClusteringWorkflow> {
        self.workflows.get(workflow_id)
    }

    /// Get mutable workflow by ID
    pub fn get_workflow_mut(&mut self, workflow_id: &str) -> Option<&mut ClusteringWorkflow> {
        self.workflows.get_mut(workflow_id)
    }

    /// Execute a specific workflow
    pub fn execute_workflow(&mut self, workflow_id: &str) -> Result<()> {
        if let Some(workflow) = self.workflows.get_mut(workflow_id) {
            workflow.execute()
        } else {
            Err(ClusteringError::InvalidInput(format!(
                "Workflow not found: {}",
                workflow_id
            )))
        }
    }

    /// Get status of all workflows
    pub fn get_all_statuses(&self) -> HashMap<String, WorkflowStatus> {
        self.workflows
            .iter()
            .map(|(id, workflow)| (id.clone(), workflow.get_status()))
            .collect()
    }

    /// Remove completed workflows
    pub fn cleanup_completed(&mut self) {
        self.workflows.retain(|_, workflow| {
            !matches!(workflow.current_state, AlgorithmState::Completed { .. })
        });
    }
}

/// Auto-save configuration
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AutoSaveConfig {
    /// Enable auto-save
    pub enabled: bool,
    /// Save interval in seconds
    pub interval_seconds: u64,
    /// Directory for auto-save files
    pub save_directory: PathBuf,
}

impl Default for AutoSaveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_seconds: 300, // 5 minutes
            save_directory: PathBuf::from("./checkpoints"),
        }
    }
}

/// Workflow execution state
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum WorkflowState {
    /// Workflow created but not started
    Created,
    /// Workflow is running
    Running,
    /// Workflow paused
    Paused,
    /// Workflow completed successfully
    Completed,
    /// Workflow failed with error
    Failed(String),
    /// Workflow was cancelled
    Cancelled,
}

/// Result of a workflow step
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum StepResult {
    /// Step completed successfully
    Success {
        /// Output data from the step
        output: serde_json::Value,
        /// Execution metrics
        metrics: HashMap<String, f64>,
    },
    /// Step failed
    Failure {
        /// Error message
        error: String,
        /// Error details
        details: Option<serde_json::Value>,
    },
    /// Step was skipped
    Skipped {
        /// Reason for skipping
        reason: String,
    },
}

/// Workflow step definition
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorkflowStep {
    /// Step name/identifier
    pub name: String,
    /// Step type/algorithm
    pub step_type: String,
    /// Step parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Dependencies on other steps
    pub dependencies: Vec<String>,
    /// Expected execution time (optional)
    pub expected_duration: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_creation() {
        let config = WorkflowConfig {
            auto_save_interval: Some(60),
            max_retries: 3,
            step_timeout: Some(300),
            parallel_execution: false,
            checkpoint_dir: None,
        };

        let workflow = ClusteringWorkflow::new("test_workflow".to_string(), config);
        assert_eq!(workflow.workflow_id, "test_workflow");
        assert_eq!(workflow.current_step, 0);
        assert!(workflow.steps.is_empty());
    }

    #[test]
    fn test_workflow_step_addition() {
        let config = WorkflowConfig {
            auto_save_interval: None,
            max_retries: 1,
            step_timeout: None,
            parallel_execution: false,
            checkpoint_dir: None,
        };

        let mut workflow = ClusteringWorkflow::new("test".to_string(), config);

        let step = TrainingStep {
            name: "kmeans_step".to_string(),
            algorithm: "kmeans".to_string(),
            parameters: HashMap::new(),
            dependencies: Vec::new(),
            completed: false,
            execution_time: None,
            results: None,
        };

        workflow.add_step(step);
        assert_eq!(workflow.steps.len(), 1);
        assert_eq!(workflow.steps[0].name, "kmeans_step");
    }

    #[test]
    fn test_workflow_manager() {
        let config = ManagerConfig::default();
        let mut manager = ClusteringWorkflowManager::new(config);

        let workflow_config = WorkflowConfig {
            auto_save_interval: None,
            max_retries: 1,
            step_timeout: None,
            parallel_execution: false,
            checkpoint_dir: None,
        };

        let workflow = ClusteringWorkflow::new("test_workflow".to_string(), workflow_config);
        manager.add_workflow(workflow).unwrap();

        assert!(manager.get_workflow("test_workflow").is_some());
        assert_eq!(manager.workflows.len(), 1);
    }
}
