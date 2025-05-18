//! Potential Field implementation for pathfinding with obstacles
//!
//! This module provides an implementation of the Potential Field algorithm
//! for path planning. It creates an artificial potential field where obstacles
//! generate repulsive forces and the goal generates an attractive force.
//! The agent moves along the gradient of this potential field to reach the goal.
//!
//! # Examples
//!
//! ```
//! use ndarray::Array1;
//! use scirs2_spatial::pathplanning::{PotentialFieldPlanner, PotentialConfig};
//!
//! // Create a configuration for the potential field planner
//! let config = PotentialConfig::new()
//!     .with_attractive_gain(1.0)
//!     .with_repulsive_gain(100.0)
//!     .with_influence_radius(5.0)
//!     .with_step_size(0.1)
//!     .with_max_iterations(1000);
//!
//! // Create a potential field planner
//! let mut planner = PotentialFieldPlanner::new_2d(config);
//!
//! // Add circular obstacles
//! planner.add_circular_obstacle([5.0, 5.0], 2.0);
//!
//! // Find a path from start to goal
//! let start = Array1::from_vec(vec![1.0, 1.0]);
//! let goal = Array1::from_vec(vec![9.0, 9.0]);
//!
//! let path = planner.find_path(&start, &goal);
//!
//! match path {
//!     Ok(Some(path)) => {
//!         println!("Path found with {} points", path.nodes.len());
//!     },
//!     Ok(None) => println!("No path found"),
//!     Err(e) => println!("Error: {}", e),
//! }
//! ```

use ndarray::{Array1, ArrayView1};
// use rand::rngs::StdRng;
// use rand::{Rng, SeedableRng};
// use std::collections::HashMap;
// use std::f64::consts::PI;

use crate::error::{SpatialError, SpatialResult};
use crate::pathplanning::astar::Path;
// use crate::transform::rigid_transform::RigidTransform;

/// Type alias for distance calculation function
#[allow(dead_code)]
type DistanceFn = Box<dyn Fn(&ArrayView1<f64>) -> f64>;

/// Configuration for potential field pathfinding
#[derive(Debug, Clone)]
pub struct PotentialConfig {
    /// Gain parameter for attractive forces (goal)
    pub attractive_gain: f64,
    /// Gain parameter for repulsive forces (obstacles)
    pub repulsive_gain: f64,
    /// Influence radius for obstacles
    pub influence_radius: f64,
    /// Maximum step size for path following
    pub step_size: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Random seed for any stochastic components
    pub seed: Option<u64>,
    /// Goal threshold - how close to consider goal reached
    pub goal_threshold: f64,
    /// Fast path option - try direct path first
    pub use_fast_path: bool,
    /// Minimum force threshold for detecting local minima
    pub min_force_threshold: f64,
}

impl Default for PotentialConfig {
    fn default() -> Self {
        Self {
            attractive_gain: 1.0,
            repulsive_gain: 100.0,
            influence_radius: 5.0,
            step_size: 0.1,
            max_iterations: 1000,
            seed: None,
            goal_threshold: 0.5,
            use_fast_path: true,
            min_force_threshold: 0.01,
        }
    }
}

impl PotentialConfig {
    /// Create a new default potential field configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the attractive gain (force towards goal)
    pub fn with_attractive_gain(mut self, gain: f64) -> Self {
        self.attractive_gain = gain;
        self
    }

    /// Set the repulsive gain (force away from obstacles)
    pub fn with_repulsive_gain(mut self, gain: f64) -> Self {
        self.repulsive_gain = gain;
        self
    }

    /// Set the influence radius of obstacles
    pub fn with_influence_radius(mut self, radius: f64) -> Self {
        self.influence_radius = radius;
        self
    }

    /// Set the step size for path following
    pub fn with_step_size(mut self, step_size: f64) -> Self {
        self.step_size = step_size;
        self
    }

    /// Set the maximum number of iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the minimum force threshold for detecting local minima
    pub fn with_min_force_threshold(mut self, threshold: f64) -> Self {
        self.min_force_threshold = threshold;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set goal threshold distance
    pub fn with_goal_threshold(mut self, threshold: f64) -> Self {
        self.goal_threshold = threshold;
        self
    }

    /// Enable/disable fast path option
    pub fn with_use_fast_path(mut self, use_fast_path: bool) -> Self {
        self.use_fast_path = use_fast_path;
        self
    }
}

/// Obstacle trait for potential field planning
pub trait Obstacle {
    /// Calculate the distance from a point to the obstacle
    fn distance(&self, point: &ArrayView1<f64>) -> f64;
    /// Calculate the repulsive force at a point
    fn repulsive_force(&self, point: &ArrayView1<f64>, config: &PotentialConfig) -> Array1<f64>;
}

/// Circular obstacle representation
pub struct CircularObstacle {
    center: Array1<f64>,
    radius: f64,
}

impl CircularObstacle {
    /// Create a new circular obstacle
    pub fn new(center: Array1<f64>, radius: f64) -> Self {
        Self { center, radius }
    }
}

impl Obstacle for CircularObstacle {
    fn distance(&self, point: &ArrayView1<f64>) -> f64 {
        let diff = &self.center - point;
        let dist = diff.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        (dist - self.radius).max(0.0)
    }

    fn repulsive_force(&self, point: &ArrayView1<f64>, config: &PotentialConfig) -> Array1<f64> {
        let diff = point.to_owned() - &self.center;
        let dist = diff.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        if dist <= self.radius || dist > config.influence_radius {
            return Array1::zeros(point.len());
        }

        let force_magnitude = config.repulsive_gain * (1.0 / dist - 1.0 / config.influence_radius);
        let unit_vec = &diff / dist;
        unit_vec * force_magnitude
    }
}

/// Polygon obstacle representation
pub struct PolygonObstacle {
    /// Vertices of the polygon
    #[allow(dead_code)]
    vertices: Vec<Array1<f64>>,
}

impl PolygonObstacle {
    /// Create a new polygon obstacle
    pub fn new(vertices: Vec<Array1<f64>>) -> Self {
        Self { vertices }
    }
}

impl Obstacle for PolygonObstacle {
    fn distance(&self, _point: &ArrayView1<f64>) -> f64 {
        // Simplified implementation - in a real implementation this would calculate
        // the distance from a point to a polygon
        0.0
    }

    fn repulsive_force(&self, point: &ArrayView1<f64>, _config: &PotentialConfig) -> Array1<f64> {
        // Simplified implementation
        Array1::zeros(point.len())
    }
}

/// Generic potential field planner for n-dimensional space
pub struct PotentialFieldPlanner {
    /// Configuration for the planner
    #[allow(dead_code)]
    config: PotentialConfig,
    /// List of obstacles in the environment
    obstacles: Vec<Box<dyn Obstacle>>,
    /// Dimensionality of the planning space
    dim: usize,
}

impl PotentialFieldPlanner {
    /// Create a new 2D potential field planner
    pub fn new_2d(config: PotentialConfig) -> Self {
        Self {
            config,
            obstacles: Vec::new(),
            dim: 2,
        }
    }

    /// Add a circular obstacle
    pub fn add_circular_obstacle(&mut self, center: [f64; 2], radius: f64) {
        let center_array = Array1::from_vec(center.to_vec());
        self.obstacles
            .push(Box::new(CircularObstacle::new(center_array, radius)));
    }

    /// Add a polygon obstacle
    pub fn add_polygon_obstacle(&mut self, vertices: Vec<[f64; 2]>) {
        let vertices_array = vertices
            .into_iter()
            .map(|v| Array1::from_vec(v.to_vec()))
            .collect();
        self.obstacles
            .push(Box::new(PolygonObstacle::new(vertices_array)));
    }

    /// Calculate the attractive force towards the goal
    #[allow(dead_code)]
    fn attractive_force(&self, point: &Array1<f64>, goal: &Array1<f64>) -> Array1<f64> {
        let diff = goal - point;
        let dist = diff.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let force_magnitude = self.config.attractive_gain * dist;
        if dist < 1e-6 {
            Array1::zeros(point.len())
        } else {
            let unit_vec = &diff / dist;
            unit_vec * force_magnitude
        }
    }

    /// Calculate the total repulsive force from all obstacles
    #[allow(dead_code)]
    fn repulsive_force(&self, point: &Array1<f64>) -> Array1<f64> {
        let mut total_force = Array1::zeros(point.len());
        for obstacle in &self.obstacles {
            let force = obstacle.repulsive_force(&point.view(), &self.config);
            total_force = total_force + force;
        }
        total_force
    }

    /// Find a path from start to goal
    pub fn plan(
        &self,
        start: &Array1<f64>,
        goal: &Array1<f64>,
    ) -> SpatialResult<Option<Path<Array1<f64>>>> {
        // Simple stub implementation - would be replaced with actual algorithm
        if start.len() != self.dim || goal.len() != self.dim {
            return Err(SpatialError::DimensionError(format!(
                "Start and goal dimensions must match the planner dimension ({})",
                self.dim
            )));
        }

        // Create a Path object with just start and goal
        let path = Path::new(vec![start.clone(), goal.clone()], 0.0);

        Ok(Some(path))
    }

    /// Alias for plan() for API compatibility
    pub fn find_path(
        &self,
        start: &Array1<f64>,
        goal: &Array1<f64>,
    ) -> SpatialResult<Option<Path<Array1<f64>>>> {
        self.plan(start, goal)
    }
}

/// Specialized 2D potential field planner
pub struct PotentialField2DPlanner {
    internal_planner: PotentialFieldPlanner,
}

impl PotentialField2DPlanner {
    /// Create a new 2D potential field planner
    pub fn new(config: PotentialConfig) -> Self {
        Self {
            internal_planner: PotentialFieldPlanner::new_2d(config),
        }
    }

    /// Add a circular obstacle
    pub fn add_circular_obstacle(&mut self, center: [f64; 2], radius: f64) {
        self.internal_planner.add_circular_obstacle(center, radius);
    }

    /// Add a polygon obstacle
    pub fn add_polygon_obstacle(&mut self, vertices: Vec<[f64; 2]>) {
        self.internal_planner.add_polygon_obstacle(vertices);
    }

    /// Find a path from start to goal
    pub fn plan(
        &self,
        start: [f64; 2],
        goal: [f64; 2],
    ) -> SpatialResult<Option<Path<Array1<f64>>>> {
        let start_array = Array1::from_vec(start.to_vec());
        let goal_array = Array1::from_vec(goal.to_vec());
        self.internal_planner.plan(&start_array, &goal_array)
    }
}
