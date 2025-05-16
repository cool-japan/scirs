//! Rapidly-exploring Random Tree (RRT) implementation
//!
//! The RRT algorithm incrementally builds a space-filling tree that
//! efficiently explores high-dimensional spaces. It is particularly useful
//! for motion planning in robotics and other applications with complex
//! configuration spaces.
//!
//! This implementation includes:
//! - Basic RRT for pathfinding
//! - RRT* for optimal pathfinding
//! - RRT-Connect for bi-directional search
//!
//! The algorithm works by:
//! 1. Sampling random points in the space
//! 2. Finding the nearest node in the tree to the sampled point
//! 3. Extending the tree toward the sampled point
//! 4. Repeating until the goal is reached or max iterations are exceeded

use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt::Debug;

use crate::distance::EuclideanDistance;
use crate::error::{SpatialError, SpatialResult};
use crate::kdtree::KDTree;
use crate::pathplanning::astar::Path;

/// Type alias for the collision checking function
type CollisionCheckFn = Box<dyn Fn(&Array1<f64>, &Array1<f64>) -> bool>;

/// Configuration options for the RRT algorithm
#[derive(Clone, Debug)]
pub struct RRTConfig {
    /// Maximum number of iterations before giving up
    pub max_iterations: usize,
    /// Maximum step size for tree extension
    pub step_size: f64,
    /// Goal bias (probability of sampling the goal directly)
    pub goal_bias: f64,
    /// Optional random seed for reproducibility
    pub seed: Option<u64>,
    /// Whether to use RRT* algorithm for optimality
    pub use_rrt_star: bool,
    /// Neighborhood radius for RRT* rewiring
    pub neighborhood_radius: Option<f64>,
    /// Whether to use bi-directional RRT (RRT-Connect)
    pub bidirectional: bool,
}

impl Default for RRTConfig {
    fn default() -> Self {
        RRTConfig {
            max_iterations: 10000,
            step_size: 0.5,
            goal_bias: 0.05,
            seed: None,
            use_rrt_star: false,
            neighborhood_radius: None,
            bidirectional: false,
        }
    }
}

/// Tree node for RRT algorithm
#[derive(Clone, Debug)]
struct RRTNode {
    /// Position in the configuration space
    position: Array1<f64>,
    /// Index of parent node
    parent: Option<usize>,
    /// Cost from the start (used in RRT*)
    cost: f64,
}

/// Rapidly-exploring Random Tree (RRT) planner
pub struct RRTPlanner {
    /// Configuration options
    config: RRTConfig,
    /// Collision checking function
    collision_checker: Option<CollisionCheckFn>,
    /// Random number generator
    rng: StdRng,
    /// Dimension of the configuration space
    dimension: usize,
    /// Bounds of the configuration space (min, max)
    bounds: Option<(Array1<f64>, Array1<f64>)>,
}

impl Debug for RRTPlanner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RRTPlanner")
            .field("config", &self.config)
            .field("dimension", &self.dimension)
            .field("bounds", &self.bounds)
            .field("collision_checker", &"<function>")
            .finish()
    }
}

impl Clone for RRTPlanner {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            collision_checker: None, // We can't clone the collision checker function
            rng: StdRng::seed_from_u64(rand::random()), // Create a new random number generator
            dimension: self.dimension,
            bounds: self.bounds.clone(),
        }
    }
}

impl RRTPlanner {
    /// Create a new RRT planner with the given configuration
    pub fn new(config: RRTConfig, dimension: usize) -> Self {
        let seed = config.seed.unwrap_or_else(rand::random);
        let rng = StdRng::seed_from_u64(seed);

        RRTPlanner {
            config,
            collision_checker: None,
            rng,
            dimension,
            bounds: None,
        }
    }

    /// Set the collision checking function
    ///
    /// The function should return true if there is a collision between the two points
    pub fn with_collision_checker<F>(mut self, collision_checker: F) -> Self
    where
        F: Fn(&Array1<f64>, &Array1<f64>) -> bool + 'static,
    {
        self.collision_checker = Some(Box::new(collision_checker));
        self
    }

    /// Set the bounds of the configuration space
    pub fn with_bounds(
        mut self,
        min_bounds: Array1<f64>,
        max_bounds: Array1<f64>,
    ) -> SpatialResult<Self> {
        if min_bounds.len() != self.dimension || max_bounds.len() != self.dimension {
            return Err(SpatialError::DimensionError(format!(
                "Bounds dimensions ({}, {}) don't match planner dimension ({})",
                min_bounds.len(),
                max_bounds.len(),
                self.dimension
            )));
        }

        // Ensure min_bounds are actually less than max_bounds
        for i in 0..self.dimension {
            if min_bounds[i] >= max_bounds[i] {
                return Err(SpatialError::ValueError(format!(
                    "Min bound {} is not less than max bound {} at index {}",
                    min_bounds[i], max_bounds[i], i
                )));
            }
        }

        self.bounds = Some((min_bounds, max_bounds));
        Ok(self)
    }

    /// Sample a random point in the configuration space
    fn sample_random_point(&mut self) -> Array1<f64> {
        let (min_bounds, max_bounds) = self
            .bounds
            .as_ref()
            .expect("Bounds must be set before sampling");
        let mut point = Array1::zeros(self.dimension);

        for i in 0..self.dimension {
            point[i] = self.rng.random_range(min_bounds[i]..max_bounds[i]);
        }

        point
    }

    /// Sample a random point with goal bias
    fn sample_with_goal_bias(&mut self, goal: &ArrayView1<f64>) -> Array1<f64> {
        if self.rng.random_range(0.0..1.0) < self.config.goal_bias {
            goal.to_owned()
        } else {
            self.sample_random_point()
        }
    }

    /// Find the nearest node in the tree to the given point
    fn find_nearest_node(
        &self,
        point: &ArrayView1<f64>,
        _nodes: &[RRTNode],
        kdtree: &KDTree<f64, EuclideanDistance<f64>>,
    ) -> usize {
        let (indices, _) = kdtree
            .query(point.as_slice().unwrap(), 1)
            .expect("KDTree query failed");
        indices[0]
    }

    /// Compute a new point that is step_size distance from nearest toward random_point
    fn steer(&self, nearest: &ArrayView1<f64>, random_point: &ArrayView1<f64>) -> Array1<f64> {
        let mut direction = random_point - nearest;
        let norm = direction.iter().map(|&x| x * x).sum::<f64>().sqrt();

        if norm < 1e-10 {
            return nearest.to_owned();
        }

        // Scale to step_size
        if norm > self.config.step_size {
            direction *= self.config.step_size / norm;
        }

        nearest + direction
    }

    /// Check if there is a valid path between two points
    fn is_valid_connection(&self, from: &ArrayView1<f64>, to: &ArrayView1<f64>) -> bool {
        if let Some(ref collision_checker) = self.collision_checker {
            !collision_checker(&from.to_owned(), &to.to_owned())
        } else {
            true // No collision checker provided, assume valid
        }
    }

    /// Find the path from start to goal using RRT
    pub fn find_path(
        &mut self,
        start: ArrayView1<f64>,
        goal: ArrayView1<f64>,
        goal_threshold: f64,
    ) -> SpatialResult<Option<Path<Array1<f64>>>> {
        if start.len() != self.dimension || goal.len() != self.dimension {
            return Err(SpatialError::DimensionError(format!(
                "Start or goal dimensions ({}, {}) don't match planner dimension ({})",
                start.len(),
                goal.len(),
                self.dimension
            )));
        }

        if self.bounds.is_none() {
            return Err(SpatialError::ValueError(
                "Bounds must be set before planning".to_string(),
            ));
        }

        if self.config.bidirectional {
            self.find_path_bidirectional(start, goal, goal_threshold)
        } else if self.config.use_rrt_star {
            self.find_path_rrt_star(start, goal, goal_threshold)
        } else {
            self.find_path_basic_rrt(start, goal, goal_threshold)
        }
    }

    /// Find the path using basic RRT algorithm
    fn find_path_basic_rrt(
        &mut self,
        start: ArrayView1<f64>,
        goal: ArrayView1<f64>,
        goal_threshold: f64,
    ) -> SpatialResult<Option<Path<Array1<f64>>>> {
        // Initialize tree with start node
        let mut nodes = vec![RRTNode {
            position: start.to_owned(),
            parent: None,
            cost: 0.0,
        }];

        for _ in 0..self.config.max_iterations {
            // Sample random point with goal bias
            let random_point = self.sample_with_goal_bias(&goal);

            // Build KDTree for nearest neighbor search
            let points: Vec<_> = nodes.iter().map(|node| node.position.clone()).collect();
            let points_array = ndarray::stack(
                ndarray::Axis(0),
                &points.iter().map(|p| p.view()).collect::<Vec<_>>(),
            )
            .expect("Failed to stack points");
            let kdtree = KDTree::<f64, EuclideanDistance<f64>>::new(&points_array)
                .expect("Failed to build KDTree");

            // Find nearest node
            let nearest_idx = self.find_nearest_node(&random_point.view(), &nodes, &kdtree);

            // Create temporary copies to avoid borrowing conflicts
            let nearest_position = nodes[nearest_idx].position.clone();
            let nearest_cost = nodes[nearest_idx].cost;

            // Steer toward random point
            let new_point = self.steer(&nearest_position.view(), &random_point.view());

            // Check if the connection is valid
            if self.is_valid_connection(&nearest_position.view(), &new_point.view()) {
                // Add new node to the tree
                let new_node = RRTNode {
                    position: new_point.clone(),
                    parent: Some(nearest_idx),
                    cost: nearest_cost
                        + euclidean_distance(&nearest_position.view(), &new_point.view()),
                };
                nodes.push(new_node);

                // Check if we've reached the goal
                if euclidean_distance(&new_point.view(), &goal) <= goal_threshold {
                    // Extract the path
                    return Ok(Some(self.extract_path(&nodes, nodes.len() - 1)));
                }
            }
        }

        // Failed to find a path within max_iterations
        Ok(None)
    }

    /// Find the path using RRT* algorithm (optimized version of RRT)
    fn find_path_rrt_star(
        &mut self,
        start: ArrayView1<f64>,
        goal: ArrayView1<f64>,
        goal_threshold: f64,
    ) -> SpatialResult<Option<Path<Array1<f64>>>> {
        // Initialize tree with start node
        let mut nodes = vec![RRTNode {
            position: start.to_owned(),
            parent: None,
            cost: 0.0,
        }];

        // Goal node index, if found
        let mut goal_idx: Option<usize> = None;
        let neighborhood_radius = self
            .config
            .neighborhood_radius
            .unwrap_or(self.config.step_size * 2.0);

        for _ in 0..self.config.max_iterations {
            // Sample random point with goal bias
            let random_point = self.sample_with_goal_bias(&goal);

            // Build KDTree for nearest neighbor search
            let points: Vec<_> = nodes.iter().map(|node| node.position.clone()).collect();
            let points_array = ndarray::stack(
                ndarray::Axis(0),
                &points.iter().map(|p| p.view()).collect::<Vec<_>>(),
            )
            .expect("Failed to stack points");
            let kdtree = KDTree::<f64, EuclideanDistance<f64>>::new(&points_array)
                .expect("Failed to build KDTree");

            // Find nearest node
            let nearest_idx = self.find_nearest_node(&random_point.view(), &nodes, &kdtree);

            // Create temporary copies to avoid borrowing conflicts
            let nearest_position = nodes[nearest_idx].position.clone();

            // Steer toward random point
            let new_point = self.steer(&nearest_position.view(), &random_point.view());

            // Check if the connection is valid
            if self.is_valid_connection(&nearest_position.view(), &new_point.view()) {
                // Find the best parent for the new node
                let (parent_idx, cost_from_parent) = self.find_best_parent(
                    &new_point,
                    &nodes,
                    &kdtree,
                    nearest_idx,
                    neighborhood_radius,
                );

                // Add new node to the tree
                let new_node_idx = nodes.len();
                let parent_cost = nodes[parent_idx].cost;
                let new_node = RRTNode {
                    position: new_point.clone(),
                    parent: Some(parent_idx),
                    cost: parent_cost + cost_from_parent,
                };
                nodes.push(new_node);

                // Rewire the tree (RRT* optimization)
                self.rewire_tree(&mut nodes, new_node_idx, &kdtree, neighborhood_radius);

                // Check if we've reached the goal
                let dist_to_goal = euclidean_distance(&new_point.view(), &goal);
                if dist_to_goal <= goal_threshold {
                    // Update goal index if we found a better path
                    let new_cost = nodes[new_node_idx].cost + dist_to_goal;
                    if let Some(idx) = goal_idx {
                        if new_cost < nodes[idx].cost {
                            goal_idx = Some(new_node_idx);
                        }
                    } else {
                        goal_idx = Some(new_node_idx);
                    }
                }
            }
        }

        // Extract the path if goal was reached
        if let Some(idx) = goal_idx {
            Ok(Some(self.extract_path(&nodes, idx)))
        } else {
            Ok(None)
        }
    }

    /// Find the best parent for a new node
    fn find_best_parent(
        &self,
        new_point: &Array1<f64>,
        nodes: &[RRTNode],
        kdtree: &KDTree<f64, EuclideanDistance<f64>>,
        nearest_idx: usize,
        radius: f64,
    ) -> (usize, f64) {
        let mut best_parent_idx = nearest_idx;
        let mut best_cost = nodes[nearest_idx].cost
            + euclidean_distance(&nodes[nearest_idx].position.view(), &new_point.view());

        // Find all nodes within the neighborhood
        let (near_indices, near_distances) = kdtree
            .query_radius(new_point.as_slice().unwrap(), radius)
            .expect("KDTree query failed");

        // Check each nearby node as a potential parent
        for (idx, &node_idx) in near_indices.iter().enumerate() {
            let node = &nodes[node_idx];
            let dist = near_distances[idx];

            // Calculate the cost if we came through this node
            let cost_from_start = node.cost + dist;

            // Update best parent if this path is cheaper
            if cost_from_start < best_cost
                && self.is_valid_connection(&node.position.view(), &new_point.view())
            {
                best_parent_idx = node_idx;
                best_cost = cost_from_start;
            }
        }

        // Return the best parent and the cost from that parent
        let cost_from_parent =
            euclidean_distance(&nodes[best_parent_idx].position.view(), &new_point.view());
        (best_parent_idx, cost_from_parent)
    }

    /// Rewire the tree to optimize paths (RRT* step)
    fn rewire_tree(
        &self,
        nodes: &mut [RRTNode],
        new_node_idx: usize,
        kdtree: &KDTree<f64, EuclideanDistance<f64>>,
        radius: f64,
    ) {
        // Create temporary copies to avoid borrowing conflicts
        let new_point = nodes[new_node_idx].position.clone();
        let new_cost = nodes[new_node_idx].cost;

        // Find all nodes within the neighborhood
        let (near_indices, near_distances) = kdtree
            .query_radius(new_point.as_slice().unwrap(), radius)
            .expect("KDTree query failed");

        // Check if we can improve the path to any nearby node by going through the new node
        for (idx, &node_idx) in near_indices.iter().enumerate() {
            // Skip the node itself
            if node_idx == new_node_idx {
                continue;
            }

            let dist = near_distances[idx];
            let cost_through_new = new_cost + dist;

            // Create temporary copy of the position to avoid borrowing conflicts
            let node_position = nodes[node_idx].position.clone();

            // If the path through the new node is better, rewire
            if cost_through_new < nodes[node_idx].cost
                && self.is_valid_connection(&new_point.view(), &node_position.view())
            {
                nodes[node_idx].parent = Some(new_node_idx);
                nodes[node_idx].cost = cost_through_new;

                // Recursively update costs in the subtree
                self.update_subtree_costs(nodes, node_idx);
            }
        }
    }

    /// Update costs in a subtree after rewiring
    #[allow(clippy::only_used_in_recursion)]
    fn update_subtree_costs(&self, nodes: &mut [RRTNode], node_idx: usize) {
        // Find all children of this node
        let children: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.parent == Some(node_idx))
            .map(|(idx, _)| idx)
            .collect();

        // Update each child's cost and recursively update its subtree
        for &child_idx in &children {
            // Create temporary copies to avoid borrowing conflicts
            let parent_cost = nodes[node_idx].cost;
            let parent_position = nodes[node_idx].position.clone();
            let child_position = nodes[child_idx].position.clone();

            let edge_cost = euclidean_distance(&parent_position.view(), &child_position.view());
            nodes[child_idx].cost = parent_cost + edge_cost;

            // Recursively update this child's subtree
            self.update_subtree_costs(nodes, child_idx);
        }
    }

    /// Find the path using bi-directional RRT (RRT-Connect)
    fn find_path_bidirectional(
        &mut self,
        start: ArrayView1<f64>,
        goal: ArrayView1<f64>,
        goal_threshold: f64,
    ) -> SpatialResult<Option<Path<Array1<f64>>>> {
        // Initialize two trees (one from start, one from goal)
        let mut start_tree = vec![RRTNode {
            position: start.to_owned(),
            parent: None,
            cost: 0.0,
        }];

        let mut goal_tree = vec![RRTNode {
            position: goal.to_owned(),
            parent: None,
            cost: 0.0,
        }];

        // Tree A starts as the start tree, Tree B as the goal tree
        let mut tree_a = &mut start_tree;
        let mut tree_b = &mut goal_tree;
        let mut a_is_start = true;

        // Indices of connecting nodes between trees, if found
        let mut connection: Option<(usize, usize)> = None;

        for _ in 0..self.config.max_iterations {
            // Swap trees every iteration
            std::mem::swap(&mut tree_a, &mut tree_b);
            a_is_start = !a_is_start;

            // Sample random point (from tree A's perspective)
            let target = if a_is_start {
                goal.to_owned()
            } else {
                start.to_owned()
            };
            let random_point = self.sample_with_goal_bias(&target.view());

            // Build KDTree for tree A
            let points_a: Vec<_> = tree_a.iter().map(|node| node.position.clone()).collect();
            let points_array_a = ndarray::stack(
                ndarray::Axis(0),
                &points_a.iter().map(|p| p.view()).collect::<Vec<_>>(),
            )
            .expect("Failed to stack points");
            let kdtree_a = KDTree::<f64, EuclideanDistance<f64>>::new(&points_array_a)
                .expect("Failed to build KDTree");

            // Find nearest node in tree A
            let nearest_idx_a = self.find_nearest_node(&random_point.view(), tree_a, &kdtree_a);

            // Create temporary copy to avoid borrowing conflicts
            let nearest_position = tree_a[nearest_idx_a].position.clone();
            let nearest_cost = tree_a[nearest_idx_a].cost;

            // Steer from nearest in A toward random point
            let new_point = self.steer(&nearest_position.view(), &random_point.view());

            // Check if the connection is valid
            if self.is_valid_connection(&nearest_position.view(), &new_point.view()) {
                // Add new node to tree A
                let new_cost =
                    nearest_cost + euclidean_distance(&nearest_position.view(), &new_point.view());
                let new_node_idx_a = tree_a.len();
                tree_a.push(RRTNode {
                    position: new_point.clone(),
                    parent: Some(nearest_idx_a),
                    cost: new_cost,
                });

                // Build KDTree for tree B
                let points_b: Vec<_> = tree_b.iter().map(|node| node.position.clone()).collect();
                let points_array_b = ndarray::stack(
                    ndarray::Axis(0),
                    &points_b.iter().map(|p| p.view()).collect::<Vec<_>>(),
                )
                .expect("Failed to stack points");
                let kdtree_b = KDTree::<f64, EuclideanDistance<f64>>::new(&points_array_b)
                    .expect("Failed to build KDTree");

                // Find nearest node in tree B
                let nearest_idx_b = self.find_nearest_node(&new_point.view(), tree_b, &kdtree_b);

                // Create temporary copy to avoid borrowing conflicts
                let nearest_position_b = tree_b[nearest_idx_b].position.clone();

                // Check if trees can be connected
                let dist_between_trees =
                    euclidean_distance(&new_point.view(), &nearest_position_b.view());
                if dist_between_trees <= goal_threshold
                    && self.is_valid_connection(&new_point.view(), &nearest_position_b.view())
                {
                    // Trees can be connected! Store the connection indices
                    connection = if a_is_start {
                        Some((new_node_idx_a, nearest_idx_b))
                    } else {
                        Some((nearest_idx_b, new_node_idx_a))
                    };
                    break;
                }
            }
        }

        // Extract the path if trees were connected
        if let Some((start_idx, goal_idx)) = connection {
            let path =
                self.extract_bidirectional_path(&start_tree, &goal_tree, start_idx, goal_idx);
            Ok(Some(path))
        } else {
            Ok(None)
        }
    }

    /// Extract the path from a bi-directional search
    fn extract_bidirectional_path(
        &self,
        start_tree: &[RRTNode],
        goal_tree: &[RRTNode],
        start_idx: usize,
        goal_idx: usize,
    ) -> Path<Array1<f64>> {
        // Extract path from start to connection point
        let mut forward_path = Vec::new();
        let mut current_idx = Some(start_idx);
        while let Some(idx) = current_idx {
            forward_path.push(start_tree[idx].position.clone());
            current_idx = start_tree[idx].parent;
        }
        forward_path.reverse(); // Reverse to get start to connection

        // Extract path from goal to connection point
        let mut backward_path = Vec::new();
        let mut current_idx = Some(goal_idx);
        while let Some(idx) = current_idx {
            backward_path.push(goal_tree[idx].position.clone());
            current_idx = goal_tree[idx].parent;
        }
        // No need to reverse - we want connection to goal

        // Combine paths
        let mut full_path = forward_path;
        full_path.extend(backward_path);

        // Calculate total cost
        let mut total_cost = 0.0;
        for i in 1..full_path.len() {
            total_cost += euclidean_distance(&full_path[i - 1].view(), &full_path[i].view());
        }

        Path::new(full_path, total_cost)
    }

    /// Extract the path from the RRT tree
    fn extract_path(&self, nodes: &[RRTNode], goal_idx: usize) -> Path<Array1<f64>> {
        let mut path = Vec::new();
        let mut current_idx = Some(goal_idx);
        let cost = nodes[goal_idx].cost;

        while let Some(idx) = current_idx {
            path.push(nodes[idx].position.clone());
            current_idx = nodes[idx].parent;
        }

        // Reverse to get start to goal
        path.reverse();

        Path::new(path, cost)
    }
}

/// Helper function to calculate Euclidean distance between points
fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

/// A 2D RRT planner that works with polygon obstacles
#[derive(Clone)]
pub struct RRT2DPlanner {
    /// The RRT planner
    planner: RRTPlanner,
    /// Obstacle polygons (each polygon is a vector of 2D points)
    obstacles: Vec<Vec<[f64; 2]>>,
    /// Step size for collision checking
    _collision_step_size: f64,
}

impl RRT2DPlanner {
    /// Create a new 2D RRT planner
    pub fn new(
        config: RRTConfig,
        obstacles: Vec<Vec<[f64; 2]>>,
        min_bounds: [f64; 2],
        max_bounds: [f64; 2],
        collision_step_size: f64,
    ) -> SpatialResult<Self> {
        let mut planner = RRTPlanner::new(config, 2);
        planner = planner.with_bounds(ndarray::arr1(&min_bounds), ndarray::arr1(&max_bounds))?;

        let obstacles_clone = obstacles.clone();
        planner = planner.with_collision_checker(move |from, to| {
            Self::check_collision_with_obstacles(from, to, &obstacles_clone, collision_step_size)
        });

        Ok(RRT2DPlanner {
            planner,
            obstacles,
            _collision_step_size: collision_step_size,
        })
    }

    /// Find a path from start to goal
    pub fn find_path(
        &mut self,
        start: [f64; 2],
        goal: [f64; 2],
        goal_threshold: f64,
    ) -> SpatialResult<Option<Path<[f64; 2]>>> {
        let start_arr = ndarray::arr1(&start);
        let goal_arr = ndarray::arr1(&goal);

        // Check if start or goal are in collision
        for obstacle in &self.obstacles {
            if Self::point_in_polygon(&start, obstacle) {
                return Err(SpatialError::ValueError(
                    "Start point is inside an obstacle".to_string(),
                ));
            }
            if Self::point_in_polygon(&goal, obstacle) {
                return Err(SpatialError::ValueError(
                    "Goal point is inside an obstacle".to_string(),
                ));
            }
        }

        // Find path using RRT
        let result = self
            .planner
            .find_path(start_arr.view(), goal_arr.view(), goal_threshold)?;

        // Convert path to [f64; 2] format
        if let Some(path) = result {
            let nodes: Vec<[f64; 2]> = path.nodes.iter().map(|p| [p[0], p[1]]).collect();
            Ok(Some(Path::new(nodes, path.cost)))
        } else {
            Ok(None)
        }
    }

    /// Check if a line segment collides with any obstacle
    fn check_collision_with_obstacles(
        from: &Array1<f64>,
        to: &Array1<f64>,
        obstacles: &[Vec<[f64; 2]>],
        step_size: f64,
    ) -> bool {
        let from_point = [from[0], from[1]];
        let to_point = [to[0], to[1]];

        // First, check if either endpoint is inside an obstacle
        for obstacle in obstacles {
            if Self::point_in_polygon(&from_point, obstacle)
                || Self::point_in_polygon(&to_point, obstacle)
            {
                return true;
            }
        }

        // Check if the line segment intersects any obstacle
        let dx = to[0] - from[0];
        let dy = to[1] - from[1];
        let distance = (dx * dx + dy * dy).sqrt();

        if distance < 1e-6 {
            return false; // Points are too close
        }

        let steps = (distance / step_size).ceil() as usize;

        for i in 1..steps {
            let t = i as f64 / steps as f64;
            let x = from[0] + dx * t;
            let y = from[1] + dy * t;
            let point = [x, y];

            for obstacle in obstacles {
                if Self::point_in_polygon(&point, obstacle) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if a point is inside a polygon using ray casting algorithm
    fn point_in_polygon(point: &[f64; 2], polygon: &[[f64; 2]]) -> bool {
        if polygon.len() < 3 {
            return false;
        }

        let mut inside = false;
        let mut j = polygon.len() - 1;

        for i in 0..polygon.len() {
            let xi = polygon[i][0];
            let yi = polygon[i][1];
            let xj = polygon[j][0];
            let yj = polygon[j][1];

            let intersect = ((yi > point[1]) != (yj > point[1]))
                && (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi);

            if intersect {
                inside = !inside;
            }

            j = i;
        }

        inside
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrt_empty_space() {
        // Create an RRT planner in an empty 2D space
        let config = RRTConfig {
            max_iterations: 1000,
            step_size: 0.5,
            goal_bias: 0.1,
            seed: Some(42), // For reproducibility
            use_rrt_star: false,
            neighborhood_radius: None,
            bidirectional: false,
        };

        let mut planner = RRT2DPlanner::new(
            config,
            vec![],       // No obstacles
            [0.0, 0.0],   // Min bounds
            [10.0, 10.0], // Max bounds
            0.1,          // Collision step size
        )
        .unwrap();

        // Find a path from (1,1) to (9,9)
        let start = [1.0, 1.0];
        let goal = [9.0, 9.0];
        let goal_threshold = 0.5;

        let path = planner.find_path(start, goal, goal_threshold).unwrap();

        // A path should be found
        assert!(path.is_some());
        let path = path.unwrap();

        // Path should start at start and end near goal
        assert_eq!(path.nodes[0], start);
        let last_node = path.nodes.last().unwrap();
        let dist_to_goal =
            ((last_node[0] - goal[0]).powi(2) + (last_node[1] - goal[1]).powi(2)).sqrt();
        assert!(dist_to_goal <= goal_threshold);
    }

    #[test]
    fn test_rrt_star_optimization() {
        // Create an RRT* planner in an empty 2D space
        let config = RRTConfig {
            max_iterations: 1000,
            step_size: 0.5,
            goal_bias: 0.1,
            seed: Some(42), // For reproducibility
            use_rrt_star: true,
            neighborhood_radius: Some(1.0),
            bidirectional: false,
        };

        let mut planner = RRT2DPlanner::new(
            config,
            vec![],       // No obstacles
            [0.0, 0.0],   // Min bounds
            [10.0, 10.0], // Max bounds
            0.1,          // Collision step size
        )
        .unwrap();

        // Find a path from (1,1) to (9,9)
        let start = [1.0, 1.0];
        let goal = [9.0, 9.0];
        let goal_threshold = 0.5;

        let path = planner.find_path(start, goal, goal_threshold).unwrap();

        // A path should be found
        assert!(path.is_some());
        let path = path.unwrap();

        // Path should start at start and end near goal
        assert_eq!(path.nodes[0], start);
        let last_node = path.nodes.last().unwrap();
        let dist_to_goal =
            ((last_node[0] - goal[0]).powi(2) + (last_node[1] - goal[1]).powi(2)).sqrt();
        assert!(dist_to_goal <= goal_threshold);

        // RRT* should produce a reasonably direct path
        // Check that the path cost is not too much longer than the direct distance
        let direct_distance = ((goal[0] - start[0]).powi(2) + (goal[1] - start[1]).powi(2)).sqrt();
        assert!(path.cost <= direct_distance * 1.5);
    }

    #[test]
    fn test_rrt_bidirectional() {
        // Create a bidirectional RRT planner in an empty 2D space
        let config = RRTConfig {
            max_iterations: 1000,
            step_size: 0.5,
            goal_bias: 0.1,
            seed: Some(42), // For reproducibility
            use_rrt_star: false,
            neighborhood_radius: None,
            bidirectional: true,
        };

        let mut planner = RRT2DPlanner::new(
            config,
            vec![],       // No obstacles
            [0.0, 0.0],   // Min bounds
            [10.0, 10.0], // Max bounds
            0.1,          // Collision step size
        )
        .unwrap();

        // Find a path from (1,1) to (9,9)
        let start = [1.0, 1.0];
        let goal = [9.0, 9.0];
        let goal_threshold = 0.5;

        let path = planner.find_path(start, goal, goal_threshold).unwrap();

        // A path should be found
        assert!(path.is_some());
        let path = path.unwrap();

        // Path should start at start and end near goal
        assert_eq!(path.nodes[0], start);
        let last_node = path.nodes.last().unwrap();
        let dist_to_goal =
            ((last_node[0] - goal[0]).powi(2) + (last_node[1] - goal[1]).powi(2)).sqrt();
        assert!(dist_to_goal <= goal_threshold);
    }

    #[test]
    fn test_rrt_with_obstacles() {
        // Create an RRT planner with obstacles
        let config = RRTConfig {
            max_iterations: 2000,
            step_size: 0.3,
            goal_bias: 0.1,
            seed: Some(42), // For reproducibility
            use_rrt_star: false,
            neighborhood_radius: None,
            bidirectional: false,
        };

        // Define a wall obstacle that divides the space
        let obstacles = vec![vec![[4.0, 0.0], [5.0, 0.0], [5.0, 8.0], [4.0, 8.0]]];

        let mut planner = RRT2DPlanner::new(
            config,
            obstacles,
            [0.0, 0.0],   // Min bounds
            [10.0, 10.0], // Max bounds
            0.1,          // Collision step size
        )
        .unwrap();

        // Find a path from left side to right side of the wall
        let start = [2.0, 5.0];
        let goal = [7.0, 5.0];
        let goal_threshold = 0.5;

        let path = planner.find_path(start, goal, goal_threshold).unwrap();

        // A path should be found
        assert!(path.is_some());
        let path = path.unwrap();

        // Path should start at start and end near goal
        assert_eq!(path.nodes[0], start);
        let last_node = path.nodes.last().unwrap();
        let dist_to_goal =
            ((last_node[0] - goal[0]).powi(2) + (last_node[1] - goal[1]).powi(2)).sqrt();
        assert!(dist_to_goal <= goal_threshold);

        // The path should go around the wall (y < 0 or y > 8)
        // Check that no point in the path is inside the wall
        for node in &path.nodes {
            assert!(!(node[0] >= 4.0 && node[0] <= 5.0 && node[1] >= 0.0 && node[1] <= 8.0));
        }
    }
}
