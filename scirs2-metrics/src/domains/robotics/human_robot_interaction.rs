//! Human-Robot Interaction (HRI) metrics
//!
//! This module provides metrics for evaluating human-robot interaction quality,
//! safety, communication effectiveness, and collaboration efficiency.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Human-Robot Interaction evaluation metrics
#[derive(Debug, Clone)]
pub struct HumanRobotInteractionMetrics {
    /// Safety metrics for HRI
    pub safety_metrics: HriSafetyMetrics,
    /// Communication effectiveness
    pub communication_metrics: CommunicationMetrics,
    /// User satisfaction measures
    pub user_satisfaction: UserSatisfactionMetrics,
    /// Collaboration efficiency
    pub collaboration_efficiency: CollaborationEfficiencyMetrics,
}

/// HRI safety evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HriSafetyMetrics {
    /// Minimum distance maintained
    pub min_safe_distance: f64,
    /// Number of safety violations
    pub safety_violations: usize,
    /// Emergency stop response time
    pub emergency_response_time: Duration,
    /// Collision avoidance success rate
    pub collision_avoidance_rate: f64,
}

/// Communication effectiveness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationMetrics {
    /// Command understanding accuracy
    pub command_accuracy: f64,
    /// Response time to commands
    pub response_time: Duration,
    /// Feedback quality score
    pub feedback_quality: f64,
    /// Multimodal communication success
    pub multimodal_success_rate: f64,
}

/// User satisfaction assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSatisfactionMetrics {
    /// Overall satisfaction score
    pub overall_satisfaction: f64,
    /// Ease of use rating
    pub ease_of_use: f64,
    /// Trust level in robot
    pub trust_level: f64,
    /// Task completion satisfaction
    pub task_satisfaction: f64,
}

/// Collaboration efficiency evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationEfficiencyMetrics {
    /// Task completion time improvement
    pub time_improvement: f64,
    /// Workload distribution balance
    pub workload_balance: f64,
    /// Coordination effectiveness
    pub coordination_effectiveness: f64,
    /// Human cognitive load
    pub cognitive_load: f64,
}

impl HumanRobotInteractionMetrics {
    /// Create new HRI metrics
    pub fn new() -> Self {
        Self {
            safety_metrics: HriSafetyMetrics::default(),
            communication_metrics: CommunicationMetrics::default(),
            user_satisfaction: UserSatisfactionMetrics::default(),
            collaboration_efficiency: CollaborationEfficiencyMetrics::default(),
        }
    }
}

// Default implementations
impl Default for HriSafetyMetrics {
    fn default() -> Self {
        Self {
            min_safe_distance: 0.0,
            safety_violations: 0,
            emergency_response_time: Duration::from_millis(0),
            collision_avoidance_rate: 1.0,
        }
    }
}

impl Default for CommunicationMetrics {
    fn default() -> Self {
        Self {
            command_accuracy: 1.0,
            response_time: Duration::from_millis(0),
            feedback_quality: 1.0,
            multimodal_success_rate: 1.0,
        }
    }
}

impl Default for UserSatisfactionMetrics {
    fn default() -> Self {
        Self {
            overall_satisfaction: 1.0,
            ease_of_use: 1.0,
            trust_level: 1.0,
            task_satisfaction: 1.0,
        }
    }
}

impl Default for CollaborationEfficiencyMetrics {
    fn default() -> Self {
        Self {
            time_improvement: 0.0,
            workload_balance: 1.0,
            coordination_effectiveness: 1.0,
            cognitive_load: 0.5,
        }
    }
}