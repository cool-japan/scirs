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
use num_traits::Float;
// use rand::rngs::StdRng;
// use rand::{Rng, SeedableRng};
// use std::collections::HashMap;

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
    pub fn new(&mut self) -> Self {
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
    pub fn with_step_size(mut self, stepsize: f64) -> Self {
        self.step_size = stepsize;
        self
    }

    /// Set the maximum number of iterations
    pub fn with_max_iterations(mut self, maxiterations: usize) -> Self {
        self.max_iterations = maxiterations;
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
    pub fn with_use_fast_path(mut self, use_fastpath: bool) -> Self {
        self.use_fast_path = use_fastpath;
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
    vertices: Vec<Array1<f64>>,
}

impl PolygonObstacle {
    /// Create a new polygon obstacle
    pub fn new(vertices: Vec<Array1<f64>>) -> Self {
        Self { vertices }
    }

    /// Check if a point is inside the polygon using the ray casting algorithm
    fn is_point_inside(&self, point: &ArrayView1<f64>) -> bool {
        if self.vertices.len() < 3 || point.len() != 2 {
            return false; // Only support 2D polygons
        }

        let px = point[0];
        let py = point[1];
        let mut inside = false;
        let n = self.vertices.len();

        for i in 0..n {
            let j = (i + 1) % n;
            let xi = self.vertices[i][0];
            let yi = self.vertices[i][1];
            let xj = self.vertices[j][0];
            let yj = self.vertices[j][1];

            if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
        }

        inside
    }

    /// Calculate the minimum distance from a point to any edge of the polygon
    fn distance_to_polygon_boundary(&self, point: &ArrayView1<f64>) -> f64 {
        if self.vertices.len() < 2 || point.len() != 2 {
            return 0.0; // Only support 2D polygons
        }

        let mut min_distance = f64::INFINITY;
        let n = self.vertices.len();

        for i in 0..n {
            let j = (i + 1) % n;
            let edge_dist =
                self.distance_point_to_line_segment(point, &self.vertices[i], &self.vertices[j]);
            min_distance = min_distance.min(edge_dist);
        }

        min_distance
    }

    /// Calculate the distance from a point to a line segment
    fn distance_point_to_line_segment(
        &self,
        point: &ArrayView1<f64>,
        line_start: &Array1<f64>,
        line_end: &Array1<f64>,
    ) -> f64 {
        let px = point[0];
        let py = point[1];
        let x1 = line_start[0];
        let y1 = line_start[1];
        let x2 = line_end[0];
        let y2 = line_end[1];

        let dx = x2 - x1;
        let dy = y2 - y1;
        let length_squared = dx * dx + dy * dy;

        if length_squared < 1e-10 {
            // Line segment is actually a point
            let diff_x = px - x1;
            let diff_y = py - y1;
            return (diff_x * diff_x + diff_y * diff_y).sqrt();
        }

        // Parameter t represents position along the line segment
        let t = ((px - x1) * dx + (py - y1) * dy) / length_squared;
        let t_clamped = t.clamp(0.0, 1.0);

        // Find the closest point on the line segment
        let closest_x = x1 + t_clamped * dx;
        let closest_y = y1 + t_clamped * dy;

        // Calculate distance to closest point
        let diff_x = px - closest_x;
        let diff_y = py - closest_y;
        (diff_x * diff_x + diff_y * diff_y).sqrt()
    }

    /// Find the closest point on the polygon boundary to the given point
    fn closest_point_on_boundary(&self, point: &ArrayView1<f64>) -> Array1<f64> {
        if self.vertices.len() < 2 || point.len() != 2 {
            return Array1::from_vec(vec![0.0, 0.0]);
        }

        let mut min_distance = f64::INFINITY;
        let mut closest_point = Array1::from_vec(vec![0.0, 0.0]);
        let n = self.vertices.len();

        for i in 0..n {
            let j = (i + 1) % n;
            let edge_point =
                self.closest_point_on_line_segment(point, &self.vertices[i], &self.vertices[j]);

            let dist =
                self.distance_point_to_line_segment(point, &self.vertices[i], &self.vertices[j]);
            if dist < min_distance {
                min_distance = dist;
                closest_point = edge_point;
            }
        }

        closest_point
    }

    /// Find the closest point on a line segment to the given point
    fn closest_point_on_line_segment(
        &self,
        point: &ArrayView1<f64>,
        line_start: &Array1<f64>,
        line_end: &Array1<f64>,
    ) -> Array1<f64> {
        let px = point[0];
        let py = point[1];
        let x1 = line_start[0];
        let y1 = line_start[1];
        let x2 = line_end[0];
        let y2 = line_end[1];

        let dx = x2 - x1;
        let dy = y2 - y1;
        let length_squared = dx * dx + dy * dy;

        if length_squared < 1e-10 {
            // Line segment is actually a point
            return line_start.clone();
        }

        // Parameter t represents position along the line segment
        let t = ((px - x1) * dx + (py - y1) * dy) / length_squared;
        let t_clamped = t.clamp(0.0, 1.0);

        // Find the closest point on the line segment
        let closest_x = x1 + t_clamped * dx;
        let closest_y = y1 + t_clamped * dy;

        Array1::from_vec(vec![closest_x, closest_y])
    }
}

impl Obstacle for PolygonObstacle {
    fn distance(&self, point: &ArrayView1<f64>) -> f64 {
        if point.len() != 2 || self.vertices.len() < 3 {
            return 0.0; // Only support 2D polygons
        }

        let boundary_distance = self.distance_to_polygon_boundary(point);

        // If point is inside the polygon, return 0 (collision)
        if self.is_point_inside(point) {
            0.0
        } else {
            boundary_distance
        }
    }

    fn repulsive_force(&self, point: &ArrayView1<f64>, config: &PotentialConfig) -> Array1<f64> {
        if point.len() != 2 || self.vertices.len() < 3 {
            return Array1::zeros(point.len()); // Only support 2D polygons
        }

        let distance = self.distance_to_polygon_boundary(point);

        // No force if point is too far away or inside the polygon
        if distance > config.influence_radius || self.is_point_inside(point) {
            return Array1::zeros(point.len());
        }

        // Find the closest point on the polygon boundary
        let closest_point = self.closest_point_on_boundary(point);

        // Calculate direction from closest point to the point (repulsive direction)
        let direction_x = point[0] - closest_point[0];
        let direction_y = point[1] - closest_point[1];
        let direction_magnitude = (direction_x * direction_x + direction_y * direction_y).sqrt();

        if direction_magnitude < 1e-10 {
            // Point is exactly on the boundary, push in arbitrary direction
            return Array1::from_vec(vec![1.0, 0.0]) * config.repulsive_gain;
        }

        // Normalize direction vector
        let unit_direction = Array1::from_vec(vec![
            direction_x / direction_magnitude,
            direction_y / direction_magnitude,
        ]);

        // Calculate force magnitude based on distance
        // Force increases as distance decreases, following potential field formula
        let force_magnitude = if distance > 1e-6 {
            config.repulsive_gain * (1.0 / distance - 1.0 / config.influence_radius)
        } else {
            config.repulsive_gain * 1000.0 // Very large force when very close
        };

        // Apply force in the repulsive direction
        unit_direction * force_magnitude.max(0.0)
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
    fn repulsive_force(&self, point: &Array1<f64>) -> Array1<f64> {
        let mut total_force = Array1::zeros(point.len());
        for obstacle in &self.obstacles {
            let force = obstacle.repulsive_force(&point.view(), &self.config);
            total_force = total_force + force;
        }
        total_force
    }

    /// Calculate the total force (attractive + repulsive) at a point
    fn total_force(&self, point: &Array1<f64>, goal: &Array1<f64>) -> Array1<f64> {
        let attractive = self.attractive_force(point, goal);
        let repulsive = self.repulsive_force(point);
        attractive + repulsive
    }

    /// Calculate the distance between two points
    fn distance(p1: &Array1<f64>, p2: &Array1<f64>) -> f64 {
        let diff = p1 - p2;
        diff.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
    }

    /// Check if the point is in collision with any obstacle
    fn is_collision(&self, point: &Array1<f64>) -> bool {
        for obstacle in &self.obstacles {
            // Use distance to check collision - if distance is very small, consider it inside
            let dist = obstacle.distance(&point.view());
            if dist < 1e-6 {
                return true;
            }
        }
        false
    }

    /// Find a path from start to goal using potential field method
    pub fn plan(
        &self,
        start: &Array1<f64>,
        goal: &Array1<f64>,
    ) -> SpatialResult<Option<Path<Array1<f64>>>> {
        // Validate dimensions
        if start.len() != self.dim || goal.len() != self.dim {
            return Err(SpatialError::DimensionError(format!(
                "Start and goal dimensions must match the planner dimension ({})",
                self.dim
            )));
        }

        // Check if start or goal are in collision
        if self.is_collision(start) {
            return Err(SpatialError::ValueError(
                "Start position is in collision with obstacle".to_string(),
            ));
        }
        if self.is_collision(goal) {
            return Err(SpatialError::ValueError(
                "Goal position is in collision with obstacle".to_string(),
            ));
        }

        // Try fast path first if enabled
        if self.config.use_fast_path && self.is_direct_path_clear(start, goal) {
            let distance = PotentialFieldPlanner::distance(start, goal);
            let path = Path::new(vec![start.clone(), goal.clone()], distance);
            return Ok(Some(path));
        }

        // Initialize path tracking
        let mut path_points = vec![start.clone()];
        let mut current_pos = start.clone();
        let mut total_distance = 0.0;
        let mut iteration = 0;
        let mut stuck_counter = 0;
        let mut previous_pos = current_pos.clone();

        while iteration < self.config.max_iterations {
            iteration += 1;

            // Check if goal is reached
            let goal_distance = PotentialFieldPlanner::distance(&current_pos, goal);
            if goal_distance < self.config.goal_threshold {
                path_points.push(goal.clone());
                total_distance += PotentialFieldPlanner::distance(&current_pos, goal);
                let path = Path::new(path_points, total_distance);
                return Ok(Some(path));
            }

            // Calculate total force at current position
            let force = self.total_force(&current_pos, goal);
            let force_magnitude = force.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

            // Check for local minimum (very small force)
            if force_magnitude < self.config.min_force_threshold {
                // Try to escape local minimum by adding random perturbation
                let escape_success = self.escape_local_minimum(
                    &mut current_pos,
                    goal,
                    &mut path_points,
                    &mut total_distance,
                );
                if !escape_success {
                    break; // Could not escape local minimum
                }
                stuck_counter = 0;
                continue;
            }

            // Normalize force and take a step
            let force_unit = &force / force_magnitude;
            let step = &force_unit * self.config.step_size;
            let next_pos = &current_pos + &step;

            // Check if next position is in collision
            if self.is_collision(&next_pos) {
                // Try to move along the boundary or find alternative direction
                let adjusted_pos = self.adjust_for_collision(&current_pos, &step, &force_unit);
                if self.is_collision(&adjusted_pos) {
                    // Still in collision, try escape mechanism
                    let escape_success = self.escape_local_minimum(
                        &mut current_pos,
                        goal,
                        &mut path_points,
                        &mut total_distance,
                    );
                    if !escape_success {
                        break;
                    }
                    continue;
                } else {
                    current_pos = adjusted_pos;
                }
            } else {
                current_pos = next_pos;
            }

            // Check if we're stuck (moving very little)
            let movement = PotentialFieldPlanner::distance(&current_pos, &previous_pos);
            if movement < self.config.step_size * 0.1 {
                stuck_counter += 1;
                if stuck_counter > 10 {
                    // Try escape mechanism
                    let escape_success = self.escape_local_minimum(
                        &mut current_pos,
                        goal,
                        &mut path_points,
                        &mut total_distance,
                    );
                    if !escape_success {
                        break;
                    }
                    stuck_counter = 0;
                }
            } else {
                stuck_counter = 0;
            }

            // Add point to path and update distance
            total_distance += PotentialFieldPlanner::distance(&previous_pos, &current_pos);
            path_points.push(current_pos.clone());
            previous_pos = current_pos.clone();
        }

        // Return partial path if we couldn't reach the goal
        if !path_points.is_empty() {
            let path = Path::new(path_points, total_distance);
            Ok(Some(path))
        } else {
            Ok(None)
        }
    }

    /// Check if a direct path from start to goal is clear of obstacles
    fn is_direct_path_clear(&self, start: &Array1<f64>, goal: &Array1<f64>) -> bool {
        let num_checks = 20;
        for i in 0..=num_checks {
            let t = i as f64 / num_checks as f64;
            let point = start * (1.0 - t) + goal * t;
            if self.is_collision(&point) {
                return false;
            }
        }
        true
    }

    /// Attempt to escape a local minimum by trying alternative directions
    fn escape_local_minimum(
        &self,
        current_pos: &mut Array1<f64>,
        goal: &Array1<f64>,
        path_points: &mut Vec<Array1<f64>>,
        total_distance: &mut f64,
    ) -> bool {
        use rand::Rng;
        let mut rng = rand::rng();

        // Try multiple random directions
        for _ in 0..8 {
            let mut random_direction = Array1::zeros(current_pos.len());
            for i in 0..random_direction.len() {
                random_direction[i] = rng.gen_range(-1.0..1.0);
            }

            // Normalize the random direction
            let magnitude = random_direction
                .iter()
                .map(|x| x.powi(2))
                .sum::<f64>()
                .sqrt();
            if magnitude > 1e-6 {
                random_direction /= magnitude;
            }

            // Try a larger step in the random direction
            let escape_step = random_direction * (self.config.step_size * 3.0);
            let candidate_pos = &*current_pos + &escape_step;

            // Check if this position is valid and makes progress toward goal
            if !self.is_collision(&candidate_pos) {
                let old_goal_distance = PotentialFieldPlanner::distance(current_pos, goal);
                let new_goal_distance = PotentialFieldPlanner::distance(&candidate_pos, goal);

                // Accept if it gets us closer to the goal or at least doesn't move us much farther
                if new_goal_distance <= old_goal_distance * 1.2 {
                    *total_distance += PotentialFieldPlanner::distance(current_pos, &candidate_pos);
                    *current_pos = candidate_pos;
                    path_points.push(current_pos.clone());
                    return true;
                }
            }
        }

        false // Could not escape
    }

    /// Adjust movement direction to avoid collision
    fn adjust_for_collision(
        &self,
        current_pos: &Array1<f64>,
        step: &Array1<f64>,
        force_unit: &Array1<f64>,
    ) -> Array1<f64> {
        // Try moving with a smaller step
        let reduced_step = step * 0.5;
        let candidate1 = current_pos + &reduced_step;
        if !self.is_collision(&candidate1) {
            return candidate1;
        }

        // Try moving perpendicular to the force direction (wall following)
        if current_pos.len() == 2 {
            // For 2D, rotate force vector by 90 degrees
            let perpendicular = Array1::from_vec(vec![-force_unit[1], force_unit[0]]);
            let side_step = &perpendicular * self.config.step_size * 0.5;

            let candidate2 = current_pos + &side_step;
            if !self.is_collision(&candidate2) {
                return candidate2;
            }

            // Try the other perpendicular direction
            let candidate3 = current_pos - &side_step;
            if !self.is_collision(&candidate3) {
                return candidate3;
            }
        }

        // If all else fails, return current position (no movement)
        current_pos.clone()
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
