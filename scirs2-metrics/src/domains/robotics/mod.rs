//! Robotics and autonomous systems evaluation metrics
//!
//! This module provides specialized metrics for evaluating robotic systems and
//! autonomous agents across various tasks including:
//! - Motion planning and trajectory evaluation
//! - Localization and mapping (SLAM) metrics
//! - Object detection and tracking for robotics
//! - Manipulation task evaluation
//! - Navigation and path planning metrics
//! - Human-robot interaction assessment
//! - Multi-robot coordination metrics
//! - Safety and reliability evaluation

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::{DomainEvaluationResult, DomainMetrics};
use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// Module declarations
pub mod core;
pub mod human_robot_interaction;
pub mod manipulation;
pub mod motion_planning;
pub mod multi_robot;
pub mod navigation;
pub mod perception;
pub mod safety_reliability;
pub mod slam_localization;

// Re-export core types
pub use core::*;

// Re-export from motion planning module
pub use motion_planning::{
    ConstraintSatisfactionMetrics, CurvatureMetrics, MotionPlanningMetrics, Obstacle,
    ObstacleClearanceMetrics, PathOptimalityMetrics, PlanningAlgorithm, PlanningConstraints,
    PlanningEfficiencyMetrics, TrajectoryQuality, TrajectorySmoothnessMetrics,
};

// Re-export from SLAM/localization module
pub use slam_localization::{
    LocalizationAccuracyMetrics, LoopClosureMetrics, MapConsistencyMetrics, MappingQualityMetrics,
    SlamComputationalMetrics, SlamMetrics,
};

// Re-export from manipulation module
pub use manipulation::{
    ForceContactMetrics, GraspingMetrics, ManipulationAccuracyMetrics, ManipulationMetrics,
    TaskCompletionMetrics,
};

// Re-export from navigation module
pub use navigation::{
    DynamicAdaptationMetrics, GoalReachingMetrics, NavigationMetrics, ObstacleAvoidanceMetrics,
    PathPlanningMetrics,
};

// Re-export from human-robot interaction module
pub use human_robot_interaction::{
    CollaborationEfficiencyMetrics, CommunicationMetrics, HriSafetyMetrics,
    HumanRobotInteractionMetrics, UserSatisfactionMetrics,
};

// Re-export from multi-robot module
pub use multi_robot::{
    CollectiveBehaviorMetrics, FormationControlMetrics, MultiRobotMetrics,
    NetworkPerformanceMetrics, TaskAllocationMetrics,
};

// Re-export from safety/reliability module
pub use safety_reliability::{
    FailureMetrics, MaintenanceMetrics, RedundancyMetrics, RiskAssessmentMetrics,
    SafetyReliabilityMetrics,
};

// Re-export from perception module
pub use perception::{
    ObjectDetectionMetrics, RoboticPerceptionMetrics, SceneUnderstandingMetrics,
    SensorFusionMetrics,
};

/// Comprehensive robotics evaluation metrics suite
#[derive(Debug)]
pub struct RoboticsMetrics {
    /// Motion planning and control metrics
    pub motion_metrics: MotionPlanningMetrics,
    /// SLAM and localization metrics
    pub slam_metrics: SlamMetrics,
    /// Manipulation task metrics
    pub manipulation_metrics: ManipulationMetrics,
    /// Navigation metrics
    pub navigation_metrics: NavigationMetrics,
    /// Human-robot interaction metrics
    pub hri_metrics: HumanRobotInteractionMetrics,
    /// Multi-robot coordination metrics
    pub multi_robot_metrics: MultiRobotMetrics,
    /// Safety and reliability metrics
    pub safety_metrics: SafetyReliabilityMetrics,
    /// Perception metrics for robotics
    pub perception_metrics: RoboticPerceptionMetrics,
}

impl RoboticsMetrics {
    /// Create new robotics metrics
    pub fn new() -> Self {
        Self {
            motion_metrics: MotionPlanningMetrics::new(),
            slam_metrics: SlamMetrics::new(),
            manipulation_metrics: ManipulationMetrics::new(),
            navigation_metrics: NavigationMetrics::new(),
            hri_metrics: HumanRobotInteractionMetrics::new(),
            multi_robot_metrics: MultiRobotMetrics::new(),
            safety_metrics: SafetyReliabilityMetrics::new(),
            perception_metrics: RoboticPerceptionMetrics::new(),
        }
    }
}

impl Default for RoboticsMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Robotics evaluation computer for integrated metrics
///
/// # Scope
///
/// This aggregator mirrors the flat `(predicted, actual)` 2D-array interface
/// used by the other domain "computer" wrappers in this crate (e.g.
/// [`super::graph_neural_networks::GraphNeuralNetworkMetricsComputer`]). Real
/// per-domain robotics evaluation needs richly structured input that a flat
/// array cannot carry -- timestamped trajectories, obstacle maps, grasp/force
/// logs, human-distance streams, formation targets, failure event logs, and
/// so on. Because of that structural mismatch, each `evaluate_*` method below
/// only returns the sub-metric's neutral defaults rather than fabricating a
/// score from data it cannot actually interpret.
///
/// For real analysis, call the corresponding submodule's own `evaluate_*`
/// methods directly with properly-typed inputs, e.g.
/// [`MotionPlanningMetrics::evaluate_constraint_satisfaction`],
/// [`SlamMetrics::evaluate_localization_accuracy`],
/// [`ManipulationMetrics::evaluate_grasping`],
/// [`NavigationMetrics::evaluate_goal_reaching`],
/// [`RoboticPerceptionMetrics::evaluate_object_detection`],
/// [`HumanRobotInteractionMetrics::evaluate_safety`],
/// [`MultiRobotMetrics::evaluate_formation_control`], and
/// [`SafetyReliabilityMetrics::evaluate_failure_metrics`].
pub struct RoboticsMetricsComputer {
    config: RoboticsEvaluationConfig,
}

/// Configuration for robotics evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoboticsEvaluationConfig {
    /// Enable motion planning evaluation
    pub enable_motion_planning: bool,
    /// Enable SLAM evaluation
    pub enable_slam: bool,
    /// Enable manipulation evaluation
    pub enable_manipulation: bool,
    /// Enable navigation evaluation
    pub enable_navigation: bool,
    /// Enable HRI evaluation
    pub enable_hri: bool,
    /// Enable multi-robot evaluation
    pub enable_multi_robot: bool,
    /// Enable safety evaluation
    pub enable_safety: bool,
    /// Enable perception evaluation
    pub enable_perception: bool,
    /// Float-time performance requirements
    pub real_time_requirements: RealTimeRequirements,
}

/// Float-time performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeRequirements {
    /// Maximum allowed latency
    pub max_latency: Duration,
    /// Required update frequency
    pub update_frequency: f64,
    /// CPU usage threshold
    pub cpu_threshold: f64,
    /// Memory usage threshold
    pub memory_threshold: f64,
}

impl Default for RoboticsEvaluationConfig {
    fn default() -> Self {
        Self {
            enable_motion_planning: true,
            enable_slam: true,
            enable_manipulation: true,
            enable_navigation: true,
            enable_hri: true,
            enable_multi_robot: false,
            enable_safety: true,
            enable_perception: true,
            real_time_requirements: RealTimeRequirements::default(),
        }
    }
}

impl Default for RealTimeRequirements {
    fn default() -> Self {
        Self {
            max_latency: Duration::from_millis(100),
            update_frequency: 10.0,
            cpu_threshold: 0.8,
            memory_threshold: 0.8,
        }
    }
}

impl RoboticsMetricsComputer {
    /// Create new robotics metrics computer
    pub fn new(config: RoboticsEvaluationConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(RoboticsEvaluationConfig::default())
    }

    /// Compute comprehensive robotics evaluation
    pub fn compute_metrics<F: Float + 'static>(
        &mut self,
        predicted: &ArrayView2<F>,
        actual: &ArrayView2<F>,
        metadata: Option<&HashMap<String, String>>,
    ) -> Result<RoboticsMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let mut metrics = RoboticsMetrics::new();

        // Motion planning evaluation
        if self.config.enable_motion_planning {
            metrics.motion_metrics = self.evaluate_motion_planning(predicted, actual, metadata)?;
        }

        // SLAM evaluation
        if self.config.enable_slam {
            metrics.slam_metrics = self.evaluate_slam(predicted, actual, metadata)?;
        }

        // Manipulation evaluation
        if self.config.enable_manipulation {
            metrics.manipulation_metrics =
                self.evaluate_manipulation(predicted, actual, metadata)?;
        }

        // Navigation evaluation
        if self.config.enable_navigation {
            metrics.navigation_metrics = self.evaluate_navigation(predicted, actual, metadata)?;
        }

        // HRI evaluation
        if self.config.enable_hri {
            metrics.hri_metrics = self.evaluate_hri(predicted, actual, metadata)?;
        }

        // Multi-robot evaluation
        if self.config.enable_multi_robot {
            metrics.multi_robot_metrics = self.evaluate_multi_robot(predicted, actual, metadata)?;
        }

        // Safety evaluation
        if self.config.enable_safety {
            metrics.safety_metrics = self.evaluate_safety(predicted, actual, metadata)?;
        }

        // Perception evaluation
        if self.config.enable_perception {
            metrics.perception_metrics = self.evaluate_perception(predicted, actual, metadata)?;
        }

        Ok(metrics)
    }

    fn evaluate_motion_planning<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<MotionPlanningMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(MotionPlanningMetrics::new())
    }

    fn evaluate_slam<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<SlamMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(SlamMetrics::new())
    }

    fn evaluate_manipulation<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<ManipulationMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(ManipulationMetrics::new())
    }

    fn evaluate_navigation<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<NavigationMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(NavigationMetrics::new())
    }

    fn evaluate_hri<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<HumanRobotInteractionMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(HumanRobotInteractionMetrics::new())
    }

    fn evaluate_multi_robot<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<MultiRobotMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(MultiRobotMetrics::new())
    }

    fn evaluate_safety<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<SafetyReliabilityMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(SafetyReliabilityMetrics::new())
    }

    fn evaluate_perception<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<RoboticPerceptionMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(RoboticPerceptionMetrics::new())
    }
}

impl DomainMetrics for RoboticsMetrics {
    type Result = DomainEvaluationResult;

    fn domain_name(&self) -> &'static str {
        "Robotics"
    }

    fn available_metrics(&self) -> Vec<&'static str> {
        vec![
            "motion_smoothness",
            "path_optimality",
            "localization_accuracy",
            "mapping_quality",
            "grasp_success_rate",
            "manipulation_precision",
            "navigation_success_rate",
            "path_planning_efficiency",
            "hri_safety_score",
            "user_satisfaction",
            "formation_accuracy",
            "task_allocation_optimality",
            "system_availability",
            "risk_score",
            "detection_accuracy",
            "sensor_fusion_quality",
        ]
    }

    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str> {
        let mut descriptions = HashMap::new();
        descriptions.insert(
            "motion_smoothness",
            "Average jerk of executed trajectories (lower is smoother)",
        );
        descriptions.insert(
            "path_optimality",
            "Ratio of executed path length to the optimal path length",
        );
        descriptions.insert(
            "localization_accuracy",
            "Absolute Trajectory Error (RMSE of position error) for SLAM localization",
        );
        descriptions.insert(
            "mapping_quality",
            "Accuracy of the built map versus ground truth",
        );
        descriptions.insert(
            "grasp_success_rate",
            "Fraction of grasp attempts that succeeded",
        );
        descriptions.insert(
            "manipulation_precision",
            "Task-execution position precision during manipulation",
        );
        descriptions.insert(
            "navigation_success_rate",
            "Fraction of navigation goals successfully reached",
        );
        descriptions.insert(
            "path_planning_efficiency",
            "Computational efficiency of the path planner",
        );
        descriptions.insert(
            "hri_safety_score",
            "Collision-avoidance success rate during human-robot interaction",
        );
        descriptions.insert(
            "user_satisfaction",
            "Overall user satisfaction with the interaction",
        );
        descriptions.insert(
            "formation_accuracy",
            "Accuracy with which a multi-robot formation is maintained",
        );
        descriptions.insert(
            "task_allocation_optimality",
            "Optimality of task allocation across a robot team",
        );
        descriptions.insert(
            "system_availability",
            "Fraction of time the system is operational",
        );
        descriptions.insert(
            "risk_score",
            "Overall assessed operational risk (lower is safer)",
        );
        descriptions.insert(
            "detection_accuracy",
            "Object detection accuracy (mean average precision) for robotic perception",
        );
        descriptions.insert(
            "sensor_fusion_quality",
            "Accuracy improvement gained from fusing multiple sensor modalities",
        );
        descriptions
    }
}

impl RoboticsMetrics {
    /// Flatten every sub-domain's headline metric into a single map.
    ///
    /// This is the "most important metric per sub-domain" view; see
    /// [`Self::secondary_metrics`] for supporting metrics.
    pub fn primary_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        // Motion planning metrics
        metrics.insert(
            "motion_smoothness".to_string(),
            self.motion_metrics.smoothness_metrics.average_jerk,
        );
        metrics.insert(
            "path_optimality".to_string(),
            self.motion_metrics
                .optimality_metrics
                .length_optimality_ratio,
        );

        // SLAM metrics
        metrics.insert(
            "localization_accuracy".to_string(),
            self.slam_metrics
                .localization_metrics
                .absolute_trajectory_error,
        );
        metrics.insert(
            "mapping_quality".to_string(),
            self.slam_metrics.mapping_metrics.map_accuracy,
        );

        // Manipulation metrics
        metrics.insert(
            "grasp_success_rate".to_string(),
            self.manipulation_metrics.grasping_metrics.success_rate,
        );
        metrics.insert(
            "manipulation_precision".to_string(),
            self.manipulation_metrics
                .manipulation_accuracy
                .trajectory_accuracy,
        );

        // Navigation metrics
        metrics.insert(
            "navigation_success_rate".to_string(),
            self.navigation_metrics.goal_reaching.success_rate,
        );
        metrics.insert(
            "path_planning_efficiency".to_string(),
            self.navigation_metrics
                .path_planning
                .computational_efficiency,
        );

        // HRI metrics
        metrics.insert(
            "hri_safety_score".to_string(),
            self.hri_metrics.safety_metrics.collision_avoidance_rate,
        );
        metrics.insert(
            "user_satisfaction".to_string(),
            self.hri_metrics.user_satisfaction.overall_satisfaction,
        );

        // Multi-robot metrics
        metrics.insert(
            "formation_accuracy".to_string(),
            self.multi_robot_metrics
                .formation_control
                .formation_accuracy,
        );
        metrics.insert(
            "task_allocation_optimality".to_string(),
            self.multi_robot_metrics
                .task_allocation
                .allocation_optimality,
        );

        // Safety metrics
        metrics.insert(
            "system_availability".to_string(),
            self.safety_metrics.failure_metrics.availability,
        );
        metrics.insert(
            "risk_score".to_string(),
            self.safety_metrics.risk_assessment.overall_risk_score,
        );

        // Perception metrics
        metrics.insert(
            "detection_accuracy".to_string(),
            self.perception_metrics.object_detection.detection_accuracy,
        );
        metrics.insert(
            "sensor_fusion_quality".to_string(),
            self.perception_metrics.sensor_fusion.accuracy_improvement,
        );

        metrics
    }

    /// Additional supporting metrics per sub-domain; see [`Self::primary_metrics`]
    /// for each sub-domain's headline metric.
    pub fn secondary_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        // Additional motion planning metrics
        metrics.insert(
            "max_jerk".to_string(),
            self.motion_metrics.smoothness_metrics.max_jerk,
        );
        metrics.insert(
            "energy_optimality".to_string(),
            self.motion_metrics
                .optimality_metrics
                .energy_optimality_ratio,
        );

        // Additional SLAM metrics
        metrics.insert(
            "loop_closure_accuracy".to_string(),
            self.slam_metrics.loop_closure_metrics.closure_accuracy,
        );
        metrics.insert(
            "mapping_consistency".to_string(),
            self.slam_metrics
                .mapping_metrics
                .consistency_metrics
                .global_consistency,
        );

        // Additional manipulation metrics
        metrics.insert(
            "grasp_stability".to_string(),
            self.manipulation_metrics.grasping_metrics.stability_score,
        );
        metrics.insert(
            "force_control_accuracy".to_string(),
            self.manipulation_metrics.force_metrics.force_accuracy.rmse,
        );

        // Additional navigation metrics
        metrics.insert(
            "obstacle_avoidance_rate".to_string(),
            self.navigation_metrics
                .obstacle_avoidance
                .collision_avoidance_rate,
        );
        metrics.insert(
            "adaptation_success_rate".to_string(),
            self.navigation_metrics
                .dynamic_adaptation
                .adaptation_success_rate,
        );

        // Additional HRI metrics
        metrics.insert(
            "communication_accuracy".to_string(),
            self.hri_metrics.communication_metrics.command_accuracy,
        );
        metrics.insert(
            "collaboration_efficiency".to_string(),
            self.hri_metrics
                .collaboration_efficiency
                .coordination_effectiveness,
        );

        // Additional multi-robot metrics
        metrics.insert(
            "network_reliability".to_string(),
            self.multi_robot_metrics
                .network_performance
                .network_reliability,
        );
        metrics.insert(
            "swarm_cohesion".to_string(),
            self.multi_robot_metrics.collective_behavior.swarm_cohesion,
        );

        // Additional safety metrics
        metrics.insert(
            "failure_rate".to_string(),
            self.safety_metrics.failure_metrics.failure_rate,
        );
        metrics.insert(
            "redundancy_level".to_string(),
            self.safety_metrics.redundancy_metrics.redundancy_level as f64,
        );

        // Additional perception metrics
        metrics.insert(
            "false_positive_rate".to_string(),
            self.perception_metrics.object_detection.false_positive_rate,
        );
        metrics.insert(
            "segmentation_accuracy".to_string(),
            self.perception_metrics
                .scene_understanding
                .segmentation_accuracy,
        );

        metrics
    }

    /// Build a complete [`DomainEvaluationResult`] from the current metrics:
    /// every primary/secondary metric, plus a human-readable summary with an
    /// overall score, a qualitative performance level, and any recommendations.
    pub fn evaluation_summary(&self) -> DomainEvaluationResult {
        let primary = self.primary_metrics();
        let secondary = self.secondary_metrics();

        // Calculate overall performance score
        let motion_score = (primary.get("motion_smoothness").unwrap_or(&0.0)
            + primary.get("path_optimality").unwrap_or(&0.0))
            / 2.0;
        let slam_score = (primary.get("localization_accuracy").unwrap_or(&0.0)
            + primary.get("mapping_quality").unwrap_or(&0.0))
            / 2.0;
        let manipulation_score = (primary.get("grasp_success_rate").unwrap_or(&0.0)
            + primary.get("manipulation_precision").unwrap_or(&0.0))
            / 2.0;
        let navigation_score = (primary.get("navigation_success_rate").unwrap_or(&0.0)
            + primary.get("path_planning_efficiency").unwrap_or(&0.0))
            / 2.0;
        let hri_score = (primary.get("hri_safety_score").unwrap_or(&0.0)
            + primary.get("user_satisfaction").unwrap_or(&0.0))
            / 2.0;
        let multi_robot_score = (primary.get("formation_accuracy").unwrap_or(&0.0)
            + primary.get("task_allocation_optimality").unwrap_or(&0.0))
            / 2.0;
        let safety_score = primary.get("system_availability").unwrap_or(&0.0)
            * (1.0 - primary.get("risk_score").unwrap_or(&0.0));
        let perception_score = (primary.get("detection_accuracy").unwrap_or(&0.0)
            + primary.get("sensor_fusion_quality").unwrap_or(&0.0))
            / 2.0;

        let overall_score = (motion_score
            + slam_score
            + manipulation_score
            + navigation_score
            + hri_score
            + multi_robot_score
            + safety_score
            + perception_score)
            / 8.0;

        // Determine performance level
        let performance_level = if overall_score >= 0.9 {
            "Excellent"
        } else if overall_score >= 0.8 {
            "Good"
        } else if overall_score >= 0.7 {
            "Fair"
        } else if overall_score >= 0.6 {
            "Poor"
        } else {
            "Critical"
        };

        let mut recommendations = Vec::new();

        if motion_score < 0.8 {
            recommendations
                .push("Improve motion planning algorithms and trajectory smoothness".to_string());
        }
        if slam_score < 0.8 {
            recommendations.push("Enhance SLAM accuracy and mapping consistency".to_string());
        }
        if manipulation_score < 0.8 {
            recommendations.push("Optimize grasping strategies and force control".to_string());
        }
        if navigation_score < 0.8 {
            recommendations.push("Improve path planning and obstacle avoidance".to_string());
        }
        if hri_score < 0.8 {
            recommendations
                .push("Enhance human-robot interaction safety and communication".to_string());
        }
        if safety_score < 0.9 {
            recommendations.push("Critical: Address safety and reliability concerns".to_string());
        }
        if perception_score < 0.8 {
            recommendations.push("Improve perception accuracy and sensor fusion".to_string());
        }

        let mut result = DomainEvaluationResult::new();
        for (name, value) in primary {
            result.add_primary_metric(name, value);
        }
        for (name, value) in secondary {
            result.add_secondary_metric(name, value);
        }
        result.add_primary_metric("overall_score".to_string(), overall_score);

        let recommendation_text = if recommendations.is_empty() {
            "No specific recommendations; all sub-domains meet their target thresholds.".to_string()
        } else {
            recommendations.join(" ")
        };
        result.set_summary(format!(
            "Robotics System Score: {overall_score:.2} ({performance_level}). {recommendation_text}"
        ));

        result
    }
}
