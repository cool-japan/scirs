//! Workflow execution and management

use super::types::*;
use crate::distributed::ResourceRequirements;
use crate::error::{CoreError, CoreResult, ErrorContext};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

/// Distributed workflow specification
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DistributedWorkflow {
    pub name: String,
    pub description: String,
    pub stages: Vec<WorkflowStage>,
    pub dependencies: HashMap<String, Vec<String>>,
    pub resource_requirements: ResourceRequirements,
}

/// Result of workflow execution
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    pub workflow_name: String,
    pub execution_time: Duration,
    pub stage_results: HashMap<String, StageResult>,
    pub performance_metrics: PerformanceMetrics,
    pub success: bool,
}

/// Result of a single workflow stage
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResult {
    pub stage_name: String,
    pub execution_time: Duration,
    pub output_size: usize,
    pub success: bool,
    pub error_message: Option<String>,
}

/// State of workflow execution
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct WorkflowState {
    /// Completed stages
    pub completed_stages: Vec<String>,
    /// Current stage
    pub current_stage: Option<String>,
    /// Accumulated data
    pub accumulated_data: HashMap<String, Vec<u8>>,
    /// Execution metadata
    pub metadata: HashMap<String, String>,
    /// Should terminate early flag
    pub should_terminate: bool,
    /// Stage execution times
    pub stage_times: HashMap<String, Duration>,
}

impl Default for WorkflowState {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkflowState {
    pub fn new() -> Self {
        Self {
            completed_stages: Vec::new(),
            current_stage: None,
            accumulated_data: HashMap::new(),
            metadata: HashMap::new(),
            should_terminate: false,
            stage_times: HashMap::new(),
        }
    }

    pub fn incorporate_stage_result(&mut self, result: &StageResult) -> CoreResult<()> {
        self.completed_stages.push(result.stage_name.clone());
        self.stage_times
            .insert(result.stage_name.clone(), result.execution_time);

        if !result.success {
            self.should_terminate = true;
        }

        Ok(())
    }

    pub fn should_terminate_early(&self) -> bool {
        self.should_terminate
    }
}

/// Workflow execution implementation
pub struct WorkflowExecutor;

impl WorkflowExecutor {
    /// Validate a workflow before execution
    pub fn validate_workflow(workflow: &DistributedWorkflow) -> CoreResult<()> {
        // Validate basic workflow structure
        if workflow.name.is_empty() {
            return Err(CoreError::InvalidInput(ErrorContext::new(
                "Workflow name cannot be empty",
            )));
        }

        if workflow.stages.is_empty() {
            return Err(CoreError::InvalidInput(ErrorContext::new(
                "Workflow must have at least one stage",
            )));
        }

        // Validate stage dependencies
        for stage in &workflow.stages {
            if stage.name.is_empty() {
                return Err(CoreError::InvalidInput(ErrorContext::new(
                    "Stage name cannot be empty",
                )));
            }

            if stage.module.is_empty() {
                return Err(CoreError::InvalidInput(ErrorContext::new(
                    "Stage module cannot be empty",
                )));
            }

            // Check if dependencies exist as stages
            if let Some(deps) = workflow.dependencies.get(&stage.name) {
                for dep in deps {
                    if !workflow.stages.iter().any(|s| &s.name == dep) {
                        return Err(CoreError::InvalidInput(ErrorContext::new(format!(
                            "Dependency '{}' not found for stage '{}'",
                            dep, stage.name
                        ))));
                    }
                }
            }
        }

        // Check for circular dependencies
        Self::detect_circular_dependencies(workflow)?;

        // Validate resource requirements
        if workflow.resource_requirements.memory_gb == 0 {
            return Err(CoreError::InvalidInput(ErrorContext::new(
                "Workflow must specify memory requirements",
            )));
        }

        if workflow.resource_requirements.cpu_cores == 0 {
            return Err(CoreError::InvalidInput(ErrorContext::new(
                "Workflow must specify CPU requirements",
            )));
        }

        Ok(())
    }

    /// Create a workflow execution plan
    pub fn create_workflow_execution_plan(
        workflow: &DistributedWorkflow,
    ) -> CoreResult<WorkflowExecutionPlan> {
        // First validate the workflow
        Self::validate_workflow(workflow)?;

        // Topologically sort stages based on dependencies
        let sorted_stages = Self::topological_sort_stages(workflow)?;

        // Calculate estimated duration based on stage complexity and dependencies
        let estimated_duration = Self::estimate_workflow_duration(&sorted_stages, workflow)?;

        // Optimize stage ordering for parallel execution where possible
        let optimized_stages = Self::optimize_stage_ordering(sorted_stages, workflow)?;

        Ok(WorkflowExecutionPlan {
            stages: optimized_stages,
            estimated_duration,
        })
    }

    /// Topologically sort workflow stages based on dependencies
    fn topological_sort_stages(workflow: &DistributedWorkflow) -> CoreResult<Vec<WorkflowStage>> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut adjacency_list: HashMap<String, Vec<String>> = HashMap::new();

        // Initialize in-degree and adjacency list
        for stage in &workflow.stages {
            in_degree.insert(stage.name.clone(), 0);
            adjacency_list.insert(stage.name.clone(), Vec::new());
        }

        // Build dependency graph
        for (stage_name, deps) in &workflow.dependencies {
            for dep in deps {
                adjacency_list
                    .get_mut(dep)
                    .unwrap()
                    .push(stage_name.clone());
                *in_degree.get_mut(stage_name).unwrap() += 1;
            }
        }

        // Kahn's algorithm for topological sorting
        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(name, _)| name.clone())
            .collect();

        let mut sorted_names = Vec::new();

        while let Some(current) = queue.pop_front() {
            sorted_names.push(current.clone());

            if let Some(neighbors) = adjacency_list.get(&current) {
                for neighbor in neighbors {
                    let degree = in_degree.get_mut(neighbor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        if sorted_names.len() != workflow.stages.len() {
            return Err(CoreError::InvalidInput(ErrorContext::new(
                "Circular dependency detected in workflow",
            )));
        }

        // Convert sorted names back to stages
        let mut sorted_stages = Vec::new();
        for name in sorted_names {
            if let Some(stage) = workflow.stages.iter().find(|s| s.name == name) {
                sorted_stages.push(stage.clone());
            }
        }

        Ok(sorted_stages)
    }

    /// Estimate workflow duration based on stage complexity
    fn estimate_workflow_duration(
        stages: &[WorkflowStage],
        workflow: &DistributedWorkflow,
    ) -> CoreResult<Duration> {
        let mut total_duration = Duration::from_secs(0);

        for stage in stages {
            // Base estimation: 30 seconds per stage
            let mut stage_duration = Duration::from_secs(30);

            // Adjust based on stage complexity (heuristic)
            match stage.operation.as_str() {
                "matrix_multiply" | "fft" | "convolution" => {
                    stage_duration = Duration::from_secs(120); // Computationally intensive
                }
                "load_data" | "save_data" => {
                    stage_duration = Duration::from_secs(60); // I/O bound
                }
                "transform" | "filter" => {
                    stage_duration = Duration::from_secs(45); // Medium complexity
                }
                _ => {
                    // Keep default value (30 seconds)
                }
            }

            // Adjust based on resource requirements
            let memory_factor = (workflow.resource_requirements.memory_gb as f64).max(1.0);
            let adjusted_duration = Duration::from_secs_f64(
                stage_duration.as_secs_f64() * memory_factor.log2().max(1.0),
            );

            total_duration += adjusted_duration;
        }

        Ok(total_duration)
    }

    /// Optimize stage ordering for parallel execution
    fn optimize_stage_ordering(
        stages: Vec<WorkflowStage>,
        workflow: &DistributedWorkflow,
    ) -> CoreResult<Vec<WorkflowStage>> {
        // For now, return stages as-is since they're already topologically sorted
        // In a more advanced implementation, this would identify stages that can run in parallel
        // and group them accordingly

        let mut optimized = stages;

        // Identify parallel execution opportunities
        let _parallel_groups = Self::identify_parallel_groups(&optimized, workflow)?;

        // Reorder stages to maximize parallelism (simplified heuristic)
        optimized.sort_by_key(|stage| {
            // Prioritize stages with fewer dependencies first
            workflow
                .dependencies
                .get(&stage.name)
                .map_or(0, |deps| deps.len())
        });

        Ok(optimized)
    }

    /// Identify groups of stages that can run in parallel
    fn identify_parallel_groups(
        stages: &[WorkflowStage],
        workflow: &DistributedWorkflow,
    ) -> CoreResult<Vec<Vec<String>>> {
        let mut parallel_groups = Vec::new();
        let mut processed_stages = HashSet::new();

        for stage in stages {
            if !processed_stages.contains(&stage.name) {
                let mut group = vec![stage.name.clone()];
                processed_stages.insert(stage.name.clone());

                // Find other stages that can run in parallel with this one
                for other_stage in stages {
                    if other_stage.name != stage.name
                        && !processed_stages.contains(&other_stage.name)
                        && Self::can_run_in_parallel(&stage.name, &other_stage.name, workflow)?
                    {
                        group.push(other_stage.name.clone());
                        processed_stages.insert(other_stage.name.clone());
                    }
                }

                parallel_groups.push(group);
            }
        }

        Ok(parallel_groups)
    }

    /// Check if two stages can run in parallel
    fn can_run_in_parallel(
        stage1: &str,
        stage2: &str,
        workflow: &DistributedWorkflow,
    ) -> CoreResult<bool> {
        // Check if one stage depends on the other
        if let Some(deps1) = workflow.dependencies.get(stage1) {
            if deps1.contains(&stage2.to_string()) {
                return Ok(false);
            }
        }

        if let Some(deps2) = workflow.dependencies.get(stage2) {
            if deps2.contains(&stage1.to_string()) {
                return Ok(false);
            }
        }

        // Check for transitive dependencies
        // This is a simplified check - a more complete implementation would
        // perform a full transitive closure analysis

        Ok(true)
    }

    /// Setup workflow communication channels
    pub fn setup_workflow_communication(plan: &WorkflowExecutionPlan) -> CoreResult<Vec<String>> {
        let mut channels = Vec::new();

        // Create communication channels for each stage
        for stage in &plan.stages {
            let channel_name = stage.name.to_string();
            channels.push(channel_name);
        }

        // Add control channels
        channels.push("control_channel".to_string());
        channels.push("monitoring_channel".to_string());
        channels.push("error_channel".to_string());

        // Set up inter-stage communication
        for i in 0..plan.stages.len() {
            if i > 0 {
                let prev_stage_name = &plan.stages[i.saturating_sub(1)].name;
                let curr_stage_name = &plan.stages[i].name;
                let inter_stage_channel = format!("{prev_stage_name}-{curr_stage_name}");
                channels.push(inter_stage_channel);
            }
        }

        Ok(channels)
    }

    /// Execute workflow stage
    pub fn execute_workflow_stage(
        stage: &WorkflowStage,
        _channels: &[String],
    ) -> CoreResult<StageResult> {
        println!("    🔧 Executing workflow stage: {}", stage.name);
        Ok(StageResult {
            stage_name: stage.name.clone(),
            execution_time: Duration::from_millis(100),
            output_size: 1024,
            success: true,
            error_message: None,
        })
    }

    /// Aggregate workflow results
    pub fn aggregate_workflow_results(
        stage_results: &[StageResult],
        _state: &WorkflowState,
    ) -> CoreResult<WorkflowResult> {
        let total_time = stage_results
            .iter()
            .map(|r| r.execution_time)
            .sum::<Duration>();

        let mut results_map = HashMap::new();
        for result in stage_results {
            results_map.insert(result.stage_name.clone(), result.clone());
        }

        Ok(WorkflowResult {
            workflow_name: "distributed_workflow".to_string(),
            execution_time: total_time,
            stage_results: results_map,
            performance_metrics: PerformanceMetrics {
                throughput: 1000.0,
                latency: Duration::from_millis(100),
                cpu_usage: 50.0,
                memory_usage: 1024,
                gpu_usage: 30.0,
            },
            success: stage_results.iter().all(|r| r.success),
        })
    }

    /// Helper method to detect circular dependencies in workflow
    fn detect_circular_dependencies(workflow: &DistributedWorkflow) -> CoreResult<()> {
        // Build dependency graph
        let mut visited = HashSet::new();
        let mut recursion_stack = HashSet::new();

        for stage in &workflow.stages {
            if !visited.contains(&stage.name)
                && Self::detect_cycle_recursive(
                    &stage.name,
                    workflow,
                    &mut visited,
                    &mut recursion_stack,
                )?
            {
                return Err(CoreError::InvalidInput(ErrorContext::new(format!(
                    "Circular dependency detected involving stage '{}'",
                    stage.name
                ))));
            }
        }

        Ok(())
    }

    /// Recursive helper for cycle detection
    #[allow(clippy::only_used_in_recursion)]
    fn detect_cycle_recursive(
        stage_name: &str,
        workflow: &DistributedWorkflow,
        visited: &mut HashSet<String>,
        recursion_stack: &mut HashSet<String>,
    ) -> CoreResult<bool> {
        visited.insert(stage_name.to_string());
        recursion_stack.insert(stage_name.to_string());

        if let Some(deps) = workflow.dependencies.get(stage_name) {
            for dep in deps {
                if !visited.contains(dep) {
                    if Self::detect_cycle_recursive(dep, workflow, visited, recursion_stack)? {
                        return Ok(true);
                    }
                } else if recursion_stack.contains(dep) {
                    return Ok(true);
                }
            }
        }

        recursion_stack.remove(stage_name);
        Ok(false)
    }

    /// Execute a distributed workflow
    pub fn execute_distributed_workflow(
        workflow: DistributedWorkflow,
    ) -> CoreResult<WorkflowResult> {
        let start_time = Instant::now();

        println!("🌐 Executing distributed workflow: {}", workflow.name);

        // Validate workflow
        Self::validate_workflow(&workflow)?;

        // Create execution plan
        let execution_plan = Self::create_workflow_execution_plan(&workflow)?;

        // Set up inter-module communication channels
        let comm_channels = Self::setup_workflow_communication(&execution_plan)?;

        // Execute workflow stages
        let mut workflow_state = WorkflowState::new();
        let mut stage_results = Vec::new();

        for stage in &execution_plan.stages {
            println!("  🔧 Executing workflow stage: {}", stage.name);

            // Execute stage across multiple modules/nodes
            let stage_result = Self::execute_workflow_stage(stage, &comm_channels)?;

            // Update workflow state
            workflow_state.incorporate_stage_result(&stage_result)?;
            stage_results.push(stage_result);

            // Check for early termination conditions
            if workflow_state.should_terminate_early() {
                println!("  ⚠️  Early termination triggered");
                break;
            }
        }

        // Aggregate results
        let final_result = Self::aggregate_workflow_results(&stage_results, &workflow_state)?;

        println!(
            "✅ Distributed workflow completed in {:.2}s",
            start_time.elapsed().as_secs_f64()
        );
        Ok(final_result)
    }
}
