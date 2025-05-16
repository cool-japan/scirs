//! Visibility graph implementation for pathfinding with polygon obstacles
//!
//! This module provides an implementation of the visibility graph algorithm
//! for pathfinding in 2D environments with polygon obstacles. The algorithm
//! works by connecting mutually visible points (start, goal, and obstacle vertices)
//! and then finding the shortest path through this graph.
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use scirs2_spatial::pathplanning::VisibilityGraphPlanner;
//!
//! // Create some polygon obstacles
//! let obstacles = vec![
//!     array![[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]],  // Square obstacle
//!     array![[3.0, 3.0], [4.0, 3.0], [3.5, 4.0]],              // Triangle obstacle
//! ];
//!
//! // Create a visibility graph planner
//! let planner = VisibilityGraphPlanner::new(obstacles);
//!
//! // Find a path from start to goal
//! let start = [0.0, 0.0];
//! let goal = [5.0, 5.0];
//! let path = planner.find_path(start, goal).unwrap().unwrap();
//!
//! // The path contains waypoints around the obstacles
//! assert!(path.len() > 2); // More than just start and goal
//! assert_eq!(path.nodes[0], start);
//! assert_eq!(*path.nodes.last().unwrap(), goal);
//! ```

use ndarray::{Array2, ArrayView2};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::error::{SpatialError, SpatialResult};
use crate::pathplanning::astar::{AStarPlanner, HashableFloat2D, Path};
use crate::polygon;

/// A 2D point used in the visibility graph
#[derive(Debug, Clone, Copy)]
pub struct Point2D {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
}

impl Point2D {
    /// Create a new 2D point
    pub fn new(x: f64, y: f64) -> Self {
        Point2D { x, y }
    }

    /// Convert from array representation
    pub fn from_array(arr: [f64; 2]) -> Self {
        Point2D {
            x: arr[0],
            y: arr[1],
        }
    }

    /// Convert to array representation
    pub fn to_array(&self) -> [f64; 2] {
        [self.x, self.y]
    }

    /// Calculate Euclidean distance to another point
    pub fn distance(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl PartialEq for Point2D {
    fn eq(&self, other: &Self) -> bool {
        // Use high precision for point equality to avoid floating point issues
        const EPSILON: f64 = 1e-10;
        (self.x - other.x).abs() < EPSILON && (self.y - other.y).abs() < EPSILON
    }
}

impl Eq for Point2D {}

impl Hash for Point2D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Use rounded values for hashing to handle floating point imprecision
        let precision = 1_000_000.0; // 6 decimal places
        let x_rounded = (self.x * precision).round() as i64;
        let y_rounded = (self.y * precision).round() as i64;

        x_rounded.hash(state);
        y_rounded.hash(state);
    }
}

/// An edge in the visibility graph, connecting two points
#[derive(Debug, Clone)]
struct Edge {
    /// Start point of the edge
    pub start: Point2D,
    /// End point of the edge
    pub end: Point2D,
    /// Weight/cost of the edge (Euclidean distance)
    pub weight: f64,
}

impl Edge {
    /// Create a new edge between two points
    fn new(start: Point2D, end: Point2D) -> Self {
        let weight = start.distance(&end);
        Edge { start, end, weight }
    }

    /// Check if this edge intersects with a polygon edge
    fn intersects_segment(&self, p1: &Point2D, p2: &Point2D) -> bool {
        // Don't consider intersection if the segments share an endpoint
        if self.start == *p1 || self.start == *p2 || self.end == *p1 || self.end == *p2 {
            return false;
        }

        // Convert to arrays for the polygon module
        let a1 = [self.start.x, self.start.y];
        let a2 = [self.end.x, self.end.y];
        let b1 = [p1.x, p1.y];
        let b2 = [p2.x, p2.y];

        segments_intersect(&a1, &a2, &b1, &b2)
    }
}

/// A visibility graph for pathfinding with polygon obstacles
#[derive(Debug, Clone)]
pub struct VisibilityGraph {
    /// Vertices of the graph (including start, goal, and obstacle vertices)
    pub vertices: Vec<Point2D>,
    /// Adjacency list representation of the graph
    pub adjacency_list: HashMap<Point2D, Vec<(Point2D, f64)>>,
}

impl VisibilityGraph {
    /// Create a new empty visibility graph
    pub fn new() -> Self {
        VisibilityGraph {
            vertices: Vec::new(),
            adjacency_list: HashMap::new(),
        }
    }

    /// Add a vertex to the graph
    pub fn add_vertex(&mut self, vertex: Point2D) {
        if !self.vertices.contains(&vertex) {
            self.vertices.push(vertex);
            self.adjacency_list.entry(vertex).or_insert_with(Vec::new);
        }
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, from: Point2D, to: Point2D, weight: f64) {
        // Make sure both vertices exist
        self.add_vertex(from);
        self.add_vertex(to);

        // Add the edge
        self.adjacency_list
            .get_mut(&from)
            .unwrap()
            .push((to, weight));
    }

    /// Get all neighbors of a vertex
    pub fn get_neighbors(&self, vertex: &Point2D) -> Vec<(Point2D, f64)> {
        match self.adjacency_list.get(vertex) {
            Some(neighbors) => neighbors.clone(),
            None => Vec::new(),
        }
    }

    /// Find the shortest path between two points using A* search
    pub fn find_path(&self, start: Point2D, goal: Point2D) -> Option<Path<[f64; 2]>> {
        // Make sure start and goal are in the graph
        if !self.adjacency_list.contains_key(&start) || !self.adjacency_list.contains_key(&goal) {
            return None;
        }

        // Create a heuristic function (Euclidean distance)
        let heuristic = |a: &HashableFloat2D, b: &HashableFloat2D| a.distance(b);

        // Create A* planner
        let planner = AStarPlanner::new();
        let graph = self.clone();

        // Convert Point2D to HashableFloat2D for A* search
        let start_hashable = HashableFloat2D::from_array(start.to_array());
        let goal_hashable = HashableFloat2D::from_array(goal.to_array());

        // Create point to hashable mappings
        let mut point_to_hashable = HashMap::new();
        let mut hashable_to_point = HashMap::new();

        for point in &self.vertices {
            let hashable = HashableFloat2D::from_array(point.to_array());
            point_to_hashable.insert(*point, hashable);
            hashable_to_point.insert(hashable, *point);
        }

        // Create neighbor function for A*
        let neighbors_fn = move |pos: &HashableFloat2D| {
            if let Some(point) = hashable_to_point.get(pos) {
                graph
                    .get_neighbors(point)
                    .into_iter()
                    .map(|(neighbor, cost)| (point_to_hashable[&neighbor], cost))
                    .collect()
            } else {
                Vec::new()
            }
        };

        // Create heuristic function for A*
        let heuristic_fn = move |a: &HashableFloat2D, b: &HashableFloat2D| heuristic(a, b);

        // Convert path to use arrays
        match planner.search(start_hashable, goal_hashable, &neighbors_fn, &heuristic_fn) {
            Ok(Some(path)) => {
                // Convert HashableFloat2D path to array path
                let array_path = Path::new(
                    path.nodes.into_iter().map(|p| p.to_array()).collect(),
                    path.cost,
                );
                Some(array_path)
            }
            _ => None,
        }
    }

    /// Check if two points are mutually visible
    fn are_points_visible(&self, p1: &Point2D, p2: &Point2D, obstacles: &[Vec<Point2D>]) -> bool {
        if p1 == p2 {
            return true;
        }

        let edge = Edge::new(*p1, *p2);

        // Check if the edge intersects with any obstacle edge
        for obstacle in obstacles {
            let n = obstacle.len();
            for i in 0..n {
                let j = (i + 1) % n;
                if edge.intersects_segment(&obstacle[i], &obstacle[j]) {
                    return false;
                }
            }
        }

        // Check if the edge passes through any obstacle
        for obstacle in obstacles {
            // Skip if either point is a vertex of this obstacle
            if obstacle.contains(p1) || obstacle.contains(p2) {
                continue;
            }

            // Convert to array representation for the polygon check
            let middle_point = Point2D::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);

            let middle_arr = [middle_point.x, middle_point.y];

            // Convert obstacle to ndarray for point_in_polygon check
            let mut obstacle_array = Array2::zeros((obstacle.len(), 2));
            for (i, p) in obstacle.iter().enumerate() {
                obstacle_array[[i, 0]] = p.x;
                obstacle_array[[i, 1]] = p.y;
            }

            // If the midpoint is inside the obstacle, the edge passes through it
            if polygon::point_in_polygon(&middle_arr, &obstacle_array.view()) {
                return false;
            }
        }

        true
    }
}

impl Default for VisibilityGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// A pathplanning planner that uses visibility graphs to find paths
#[derive(Debug, Clone)]
pub struct VisibilityGraphPlanner {
    /// The obstacles in the environment (polygons)
    pub obstacles: Vec<Array2<f64>>,
    /// Pre-constructed visibility graph (if available)
    visibility_graph: Option<VisibilityGraph>,
    /// Whether to use the start-goal fast path optimization
    use_fast_path: bool,
}

impl VisibilityGraphPlanner {
    /// Create a new visibility graph planner with the given obstacles
    pub fn new(obstacles: Vec<Array2<f64>>) -> Self {
        VisibilityGraphPlanner {
            obstacles,
            visibility_graph: None,
            use_fast_path: true,
        }
    }

    /// Set whether to use the fast path optimization
    ///
    /// When enabled, the planner first checks if there's a direct path
    /// from start to goal before building the full visibility graph.
    pub fn with_fast_path(mut self, use_fast_path: bool) -> Self {
        self.use_fast_path = use_fast_path;
        self
    }

    /// Pre-compute the visibility graph for the given obstacles
    ///
    /// This can be called explicitly to build the graph once and reuse it
    /// for multiple path queries.
    pub fn build_graph(&mut self) -> SpatialResult<()> {
        let mut graph = VisibilityGraph::new();
        let mut obstacle_vertices = Vec::new();

        // Extract all obstacle vertices
        for obstacle in &self.obstacles {
            let mut obstacle_points = Vec::new();

            for i in 0..obstacle.shape()[0] {
                let vertex = Point2D::new(obstacle[[i, 0]], obstacle[[i, 1]]);
                obstacle_points.push(vertex);
                graph.add_vertex(vertex);
            }

            obstacle_vertices.push(obstacle_points);
        }

        // Connect mutually visible vertices
        for (i, obstacle_i) in obstacle_vertices.iter().enumerate() {
            // Connect vertices within the same obstacle to their adjacent vertices
            let n_i = obstacle_i.len();
            for j in 0..n_i {
                let v1 = obstacle_i[j];
                let v2 = obstacle_i[(j + 1) % n_i];
                let weight = v1.distance(&v2);

                graph.add_edge(v1, v2, weight);
                graph.add_edge(v2, v1, weight);
            }

            // Connect vertices between different obstacles
            for k in i + 1..obstacle_vertices.len() {
                let obstacle_k = &obstacle_vertices[k];

                for &v1 in obstacle_i {
                    for &v2 in obstacle_k {
                        if graph.are_points_visible(&v1, &v2, &obstacle_vertices) {
                            let weight = v1.distance(&v2);
                            graph.add_edge(v1, v2, weight);
                            graph.add_edge(v2, v1, weight);
                        }
                    }
                }
            }

            // Connect vertices within the same obstacle that are not adjacent
            for j in 0..n_i {
                for k in j + 2..n_i {
                    // Skip adjacent vertices (already connected)
                    if (k == j + 1) || (j == 0 && k == n_i - 1) {
                        continue;
                    }

                    let v1 = obstacle_i[j];
                    let v2 = obstacle_i[k];

                    if graph.are_points_visible(&v1, &v2, &obstacle_vertices) {
                        let weight = v1.distance(&v2);
                        graph.add_edge(v1, v2, weight);
                        graph.add_edge(v2, v1, weight);
                    }
                }
            }
        }

        self.visibility_graph = Some(graph);
        Ok(())
    }

    /// Find a path from start to goal
    ///
    /// # Arguments
    ///
    /// * `start` - The start coordinates [x, y]
    /// * `goal` - The goal coordinates [x, y]
    ///
    /// # Returns
    ///
    /// * `Ok(Some(Path))` - A path was found
    /// * `Ok(None)` - No path was found
    /// * `Err(SpatialError)` - An error occurred
    pub fn find_path(
        &mut self,
        start: [f64; 2],
        goal: [f64; 2],
    ) -> SpatialResult<Option<Path<[f64; 2]>>> {
        let start_point = Point2D::from_array(start);
        let goal_point = Point2D::from_array(goal);

        // Try direct path first (optimization)
        if self.use_fast_path {
            let direct_edge = Edge::new(start_point, goal_point);
            let mut is_visible = true;

            // Convert obstacles to Point2D vectors for visibility check
            let mut obstacle_vertices = Vec::new();
            for obstacle in &self.obstacles {
                let mut obstacle_points = Vec::new();
                for i in 0..obstacle.shape()[0] {
                    obstacle_points.push(Point2D::new(obstacle[[i, 0]], obstacle[[i, 1]]));
                }
                obstacle_vertices.push(obstacle_points);
            }

            // Check if any obstacle edge intersects with the direct path
            for obstacle in &obstacle_vertices {
                let n = obstacle.len();
                for i in 0..n {
                    let j = (i + 1) % n;
                    if direct_edge.intersects_segment(&obstacle[i], &obstacle[j]) {
                        is_visible = false;
                        break;
                    }
                }
                if !is_visible {
                    break;
                }

                // Check if the path passes through an obstacle
                let middle_point = Point2D::new(
                    (start_point.x + goal_point.x) / 2.0,
                    (start_point.y + goal_point.y) / 2.0,
                );

                let middle_arr = [middle_point.x, middle_point.y];

                // Convert obstacle to ndarray for point_in_polygon check
                let mut obstacle_array = Array2::zeros((obstacle.len(), 2));
                for (i, p) in obstacle.iter().enumerate() {
                    obstacle_array[[i, 0]] = p.x;
                    obstacle_array[[i, 1]] = p.y;
                }

                if polygon::point_in_polygon(&middle_arr, &obstacle_array.view()) {
                    is_visible = false;
                    break;
                }
            }

            if is_visible {
                let mut path = Vec::new();
                path.push(start);
                path.push(goal);
                let cost = start_point.distance(&goal_point);
                return Ok(Some(Path::new(path, cost)));
            }
        }

        // If there's no direct path or fast path is disabled, use the visibility graph
        let mut graph = match &self.visibility_graph {
            Some(g) => g.clone(),
            None => {
                self.build_graph()?;
                self.visibility_graph.as_ref().unwrap().clone()
            }
        };

        // Add start and goal points to the graph
        graph.add_vertex(start_point);
        graph.add_vertex(goal_point);

        // Connect start and goal to visible obstacle vertices
        let mut obstacle_vertices = Vec::new();
        for obstacle in &self.obstacles {
            let mut obstacle_points = Vec::new();
            for i in 0..obstacle.shape()[0] {
                obstacle_points.push(Point2D::new(obstacle[[i, 0]], obstacle[[i, 1]]));
            }
            obstacle_vertices.push(obstacle_points);
        }

        // Connect start to visible vertices
        for obstacle in &obstacle_vertices {
            for &vertex in obstacle {
                if graph.are_points_visible(&start_point, &vertex, &obstacle_vertices) {
                    let weight = start_point.distance(&vertex);
                    graph.add_edge(start_point, vertex, weight);
                    graph.add_edge(vertex, start_point, weight);
                }
            }
        }

        // Connect goal to visible vertices
        for obstacle in &obstacle_vertices {
            for &vertex in obstacle {
                if graph.are_points_visible(&goal_point, &vertex, &obstacle_vertices) {
                    let weight = goal_point.distance(&vertex);
                    graph.add_edge(goal_point, vertex, weight);
                    graph.add_edge(vertex, goal_point, weight);
                }
            }
        }

        // Connect start and goal if they're mutually visible
        if graph.are_points_visible(&start_point, &goal_point, &obstacle_vertices) {
            let weight = start_point.distance(&goal_point);
            graph.add_edge(start_point, goal_point, weight);
            graph.add_edge(goal_point, start_point, weight);
        }

        // Find path
        match graph.find_path(start_point, goal_point) {
            Some(path) => Ok(Some(path)),
            None => Ok(None),
        }
    }
}

/// Check if two line segments intersect.
///
/// # Arguments
///
/// * `a1`, `a2` - The endpoints of the first segment
/// * `b1`, `b2` - The endpoints of the second segment
///
/// # Returns
///
/// * `true` if the segments intersect, `false` otherwise
fn segments_intersect(a1: &[f64], a2: &[f64], b1: &[f64], b2: &[f64]) -> bool {
    // Function to compute orientation of triplet (p, q, r)
    // Returns:
    // 0 -> collinear
    // 1 -> clockwise
    // 2 -> counterclockwise
    let orientation = |p: &[f64], q: &[f64], r: &[f64]| -> i32 {
        let val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]);

        if val < 0.0 {
            return 1; // clockwise
        } else if val > 0.0 {
            return 2; // counterclockwise
        }

        0 // collinear
    };

    // Function to check if point q is on segment pr
    let on_segment = |p: &[f64], q: &[f64], r: &[f64]| -> bool {
        q[0] <= p[0].max(r[0])
            && q[0] >= p[0].min(r[0])
            && q[1] <= p[1].max(r[1])
            && q[1] >= p[1].min(r[1])
    };

    let o1 = orientation(a1, a2, b1);
    let o2 = orientation(a1, a2, b2);
    let o3 = orientation(b1, b2, a1);
    let o4 = orientation(b1, b2, a2);

    // General case
    if o1 != o2 && o3 != o4 {
        return true;
    }

    // Special cases
    if o1 == 0 && on_segment(a1, b1, a2) {
        return true;
    }

    if o2 == 0 && on_segment(a1, b2, a2) {
        return true;
    }

    if o3 == 0 && on_segment(b1, a1, b2) {
        return true;
    }

    if o4 == 0 && on_segment(b1, a2, b2) {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_point_equality() {
        let p1 = Point2D::new(1.0, 2.0);
        let p2 = Point2D::new(1.0, 2.0);
        let p3 = Point2D::new(1.0, 2.000000001); // Very close, should be equal
        let p4 = Point2D::new(1.1, 2.0);

        assert_eq!(p1, p2);
        assert_eq!(p1, p3);
        assert_ne!(p1, p4);
    }

    #[test]
    fn test_edge_intersection() {
        let e1 = Edge::new(Point2D::new(0.0, 0.0), Point2D::new(1.0, 1.0));
        let e2 = Edge::new(Point2D::new(0.0, 1.0), Point2D::new(1.0, 0.0));
        let e3 = Edge::new(Point2D::new(0.0, 0.0), Point2D::new(0.5, 0.5));

        // Intersecting diagonal edges
        assert!(e1.intersects_segment(&Point2D::new(0.0, 1.0), &Point2D::new(1.0, 0.0)));

        // Non-intersecting edges
        assert!(!e3.intersects_segment(&Point2D::new(0.0, 1.0), &Point2D::new(1.0, 1.0)));

        // Edges that share an endpoint should not be considered intersecting
        assert!(!e1.intersects_segment(&Point2D::new(0.0, 0.0), &Point2D::new(0.0, 1.0)));
    }

    #[test]
    fn test_visibility_graph_creation() {
        let mut graph = VisibilityGraph::new();

        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(1.0, 0.0);
        let p3 = Point2D::new(0.0, 1.0);

        graph.add_vertex(p1);
        graph.add_vertex(p2);
        graph.add_vertex(p3);

        graph.add_edge(p1, p2, p1.distance(&p2));
        graph.add_edge(p2, p3, p2.distance(&p3));

        assert_eq!(graph.vertices.len(), 3);
        assert_eq!(graph.get_neighbors(&p1).len(), 1);
        assert_eq!(graph.get_neighbors(&p2).len(), 1);
        assert_eq!(graph.get_neighbors(&p3).len(), 0);
    }

    #[test]
    fn test_visibility_check() {
        let mut graph = VisibilityGraph::new();

        // Create a square obstacle
        let obstacle = vec![
            Point2D::new(1.0, 1.0),
            Point2D::new(2.0, 1.0),
            Point2D::new(2.0, 2.0),
            Point2D::new(1.0, 2.0),
        ];

        let obstacles = vec![obstacle];

        // Points on opposite sides of the obstacle
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 3.0);

        // Points that can see each other
        let p3 = Point2D::new(0.0, 0.0);
        let p4 = Point2D::new(0.0, 3.0);

        assert!(!graph.are_points_visible(&p1, &p2, &obstacles));
        assert!(graph.are_points_visible(&p3, &p4, &obstacles));
    }

    #[test]
    fn test_simple_path() {
        // Create a square obstacle
        let obstacles = vec![array![[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0],]];

        let mut planner = VisibilityGraphPlanner::new(obstacles);

        // Path that should go around the obstacle
        let start = [0.0, 0.0];
        let goal = [3.0, 3.0];

        let path = planner.find_path(start, goal).unwrap().unwrap();

        // Path should exist and go from start to goal
        assert!(path.len() > 2);
        assert_eq!(path.nodes[0], start);
        assert_eq!(*path.nodes.last().unwrap(), goal);

        // Path should not intersect the obstacle
        for i in 0..path.nodes.len() - 1 {
            let edge = Edge::new(
                Point2D::from_array(path.nodes[i]),
                Point2D::from_array(path.nodes[i + 1]),
            );

            for j in 0..4 {
                let k = (j + 1) % 4;
                let p1 = Point2D::new(planner.obstacles[0][[j, 0]], planner.obstacles[0][[j, 1]]);
                let p2 = Point2D::new(planner.obstacles[0][[k, 0]], planner.obstacles[0][[k, 1]]);

                assert!(!edge.intersects_segment(&p1, &p2));
            }
        }
    }

    #[test]
    fn test_direct_path() {
        // Create obstacles that don't block the direct path
        let obstacles = vec![
            array![[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0],],
            array![[3.0, 3.0], [4.0, 3.0], [4.0, 4.0], [3.0, 4.0],],
        ];

        let mut planner = VisibilityGraphPlanner::new(obstacles);

        // Path that should go directly without going through obstacles
        let start = [0.0, 0.0];
        let goal = [5.0, 0.0];

        let path = planner.find_path(start, goal).unwrap().unwrap();

        // Direct path should have exactly 2 points
        assert_eq!(path.len(), 2);
        assert_eq!(path.nodes[0], start);
        assert_eq!(path.nodes[1], goal);

        // Cost should be the Euclidean distance
        assert_relative_eq!(path.cost, 5.0);
    }

    #[test]
    fn test_no_path() {
        // Create a wall of obstacles that completely block the path
        let mut obstacles = Vec::new();

        // Create a wall from x=2 to x=3, spanning y from -10 to 10
        for i in -10..10 {
            obstacles.push(array![
                [2.0, i as f64],
                [3.0, i as f64],
                [3.0, (i + 1) as f64],
                [2.0, (i + 1) as f64],
            ]);
        }

        let mut planner = VisibilityGraphPlanner::new(obstacles);

        // Path that should be impossible
        let start = [0.0, 0.0];
        let goal = [5.0, 0.0];

        let path = planner.find_path(start, goal).unwrap();

        // There should be no path
        assert!(path.is_none());
    }
}
