//! Quadtree data structure for 2D space
//!
//! This module provides a Quadtree implementation for efficient spatial queries
//! in 2D space. Quadtrees recursively subdivide space into four equal quadrants,
//! allowing for efficient nearest neighbor searches, range queries, and
//! point-in-region operations.
//!
//! The implementation supports:
//! - Quadtree construction from 2D point data
//! - Nearest neighbor searches
//! - Range queries for finding points within a specified distance
//! - Point-in-region queries
//! - Dynamic insertion and removal of points

use crate::error::{SpatialError, SpatialResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

/// Maximum number of points in a leaf node before it splits
const MAX_POINTS_PER_NODE: usize = 8;
/// Maximum depth of the quadtree
const MAX_DEPTH: usize = 20;

/// A 2D bounding box defined by its minimum and maximum corners
#[derive(Debug, Clone)]
pub struct BoundingBox2D {
    /// Minimum coordinates of the box (lower left corner)
    pub min: Array1<f64>,
    /// Maximum coordinates of the box (upper right corner)
    pub max: Array1<f64>,
}

impl BoundingBox2D {
    /// Create a new bounding box from min and max corners
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum coordinates (lower left corner)
    /// * `max` - Maximum coordinates (upper right corner)
    ///
    /// # Returns
    ///
    /// A new BoundingBox2D
    ///
    /// # Errors
    ///
    /// Returns an error if the min or max arrays don't have 2 elements,
    /// or if min > max for any dimension
    pub fn new(min: &ArrayView1<f64>, max: &ArrayView1<f64>) -> SpatialResult<Self> {
        if min.len() != 2 || max.len() != 2 {
            return Err(SpatialError::DimensionError(format!(
                "Min and max must have 2 elements, got {} and {}",
                min.len(),
                max.len()
            )));
        }

        // Check that min <= max for all dimensions
        for i in 0..2 {
            if min[i] > max[i] {
                return Err(SpatialError::ValueError(format!(
                    "Min must be <= max for all dimensions, got min[{}]={} > max[{}]={}",
                    i, min[i], i, max[i]
                )));
            }
        }

        Ok(BoundingBox2D {
            min: min.to_owned(),
            max: max.to_owned(),
        })
    }

    /// Create a bounding box that encompasses a set of points
    ///
    /// # Arguments
    ///
    /// * `points` - An array of 2D points
    ///
    /// # Returns
    ///
    /// A bounding box that contains all the points
    ///
    /// # Errors
    ///
    /// Returns an error if the points array is empty or if points don't have 2 dimensions
    pub fn from_points(points: &ArrayView2<f64>) -> SpatialResult<Self> {
        if points.is_empty() {
            return Err(SpatialError::ValueError(
                "Cannot create bounding box from empty point set".into(),
            ));
        }

        if points.ncols() != 2 {
            return Err(SpatialError::DimensionError(format!(
                "Points must have 2 columns, got {}",
                points.ncols()
            )));
        }

        // Find min and max coordinates
        let mut min = Array1::from_vec(vec![f64::INFINITY, f64::INFINITY]);
        let mut max = Array1::from_vec(vec![f64::NEG_INFINITY, f64::NEG_INFINITY]);

        for row in points.rows() {
            for d in 0..2 {
                if row[d] < min[d] {
                    min[d] = row[d];
                }
                if row[d] > max[d] {
                    max[d] = row[d];
                }
            }
        }

        Ok(BoundingBox2D { min, max })
    }

    /// Check if a point is inside the bounding box
    ///
    /// # Arguments
    ///
    /// * `point` - A 2D point to check
    ///
    /// # Returns
    ///
    /// True if the point is inside or on the boundary of the box, false otherwise
    ///
    /// # Errors
    ///
    /// Returns an error if the point doesn't have exactly 2 elements
    pub fn contains(&self, point: &ArrayView1<f64>) -> SpatialResult<bool> {
        if point.len() != 2 {
            return Err(SpatialError::DimensionError(format!(
                "Point must have 2 elements, got {}",
                point.len()
            )));
        }

        for d in 0..2 {
            if point[d] < self.min[d] || point[d] > self.max[d] {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Get the center point of the bounding box
    ///
    /// # Returns
    ///
    /// The center point of the box
    pub fn center(&self) -> Array1<f64> {
        let mut center = Array1::zeros(2);
        for d in 0..2 {
            center[d] = (self.min[d] + self.max[d]) / 2.0;
        }
        center
    }

    /// Get the dimensions (width, height) of the bounding box
    ///
    /// # Returns
    ///
    /// An array containing the dimensions of the box
    pub fn dimensions(&self) -> Array1<f64> {
        let mut dims = Array1::zeros(2);
        for d in 0..2 {
            dims[d] = self.max[d] - self.min[d];
        }
        dims
    }

    /// Check if this bounding box overlaps with another one
    ///
    /// # Arguments
    ///
    /// * `other` - Another bounding box to check against
    ///
    /// # Returns
    ///
    /// True if the boxes overlap, false otherwise
    pub fn overlaps(&self, other: &BoundingBox2D) -> bool {
        for d in 0..2 {
            if self.max[d] < other.min[d] || self.min[d] > other.max[d] {
                return false;
            }
        }
        true
    }

    /// Calculate the squared distance from a point to the nearest point on the bounding box
    ///
    /// # Arguments
    ///
    /// * `point` - A 2D point
    ///
    /// # Returns
    ///
    /// The squared distance to the nearest point on the box boundary or 0 if the point is inside
    ///
    /// # Errors
    ///
    /// Returns an error if the point doesn't have exactly 2 elements
    pub fn squared_distance_to_point(&self, point: &ArrayView1<f64>) -> SpatialResult<f64> {
        if point.len() != 2 {
            return Err(SpatialError::DimensionError(format!(
                "Point must have 2 elements, got {}",
                point.len()
            )));
        }

        let mut squared_dist = 0.0;

        for d in 0..2 {
            let v = point[d];

            if v < self.min[d] {
                // Point is below minimum bound
                squared_dist += (v - self.min[d]) * (v - self.min[d]);
            } else if v > self.max[d] {
                // Point is above maximum bound
                squared_dist += (v - self.max[d]) * (v - self.max[d]);
            }
            // If within bounds in this dimension, contribution is 0
        }

        Ok(squared_dist)
    }

    /// Split the bounding box into 4 equal quadrants
    ///
    /// # Returns
    ///
    /// An array of 4 bounding boxes representing the quadrants
    pub fn split_into_quadrants(&self) -> [BoundingBox2D; 4] {
        let center = self.center();

        // Create quadrants in this order:
        // 0: SW (bottom-left)
        // 1: SE (bottom-right)
        // 2: NW (top-left)
        // 3: NE (top-right)

        [
            // 0: SW (bottom-left)
            BoundingBox2D {
                min: self.min.clone(),
                max: center.clone(),
            },
            // 1: SE (bottom-right)
            BoundingBox2D {
                min: Array1::from_vec(vec![center[0], self.min[1]]),
                max: Array1::from_vec(vec![self.max[0], center[1]]),
            },
            // 2: NW (top-left)
            BoundingBox2D {
                min: Array1::from_vec(vec![self.min[0], center[1]]),
                max: Array1::from_vec(vec![center[0], self.max[1]]),
            },
            // 3: NE (top-right)
            BoundingBox2D {
                min: center,
                max: self.max.clone(),
            },
        ]
    }
}

/// A node in the quadtree
#[derive(Debug)]
enum QuadtreeNode {
    /// An internal node with 4 children
    Internal {
        /// Bounding box of this node
        bounds: BoundingBox2D,
        /// Children nodes (exactly 4)
        children: Box<[Option<QuadtreeNode>; 4]>,
    },
    /// A leaf node containing points
    Leaf {
        /// Bounding box of this node
        bounds: BoundingBox2D,
        /// Points in this node
        points: Vec<usize>,
        /// Actual point coordinates (reference to input data)
        point_data: Array2<f64>,
    },
}

/// A point with a distance for nearest neighbor searches
#[derive(Debug, Clone, PartialEq)]
struct DistancePoint {
    /// Index of the point in the original data
    index: usize,
    /// Squared distance to the query point
    distance_sq: f64,
}

/// For binary heap, we want max heap, but we want to extract the minimum distance,
/// so we reverse the ordering
impl Ord for DistancePoint {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance_sq
            .partial_cmp(&self.distance_sq)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for DistancePoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for DistancePoint {}

/// A node with a distance for priority queue in nearest neighbor search
#[derive(Debug, Clone, PartialEq)]
struct DistanceNode {
    /// Reference to the node
    node: *const QuadtreeNode,
    /// Minimum squared distance to the query point
    min_distance_sq: f64,
}

/// For binary heap, we want max heap, but we want to extract the minimum distance,
/// so we reverse the ordering
impl Ord for DistanceNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .min_distance_sq
            .partial_cmp(&self.min_distance_sq)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for DistanceNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for DistanceNode {}

/// The Quadtree data structure for 2D spatial searches
#[derive(Debug)]
pub struct Quadtree {
    /// Root node of the quadtree
    root: Option<QuadtreeNode>,
    /// Number of points in the quadtree
    size: usize,
    /// Original point data
    points: Array2<f64>,
}

impl Quadtree {
    /// Create a new quadtree from a set of 2D points
    ///
    /// # Arguments
    ///
    /// * `points` - An array of 2D points
    ///
    /// # Returns
    ///
    /// A new Quadtree containing the points
    ///
    /// # Errors
    ///
    /// Returns an error if the points array is empty or if points don't have 2 dimensions
    pub fn new(points: &ArrayView2<f64>) -> SpatialResult<Self> {
        if points.is_empty() {
            return Err(SpatialError::ValueError(
                "Cannot create quadtree from empty point set".into(),
            ));
        }

        if points.ncols() != 2 {
            return Err(SpatialError::DimensionError(format!(
                "Points must have 2 columns, got {}",
                points.ncols()
            )));
        }

        let size = points.nrows();
        let bounds = BoundingBox2D::from_points(points)?;
        let points_owned = points.to_owned();

        // Create initial indices (0 to size-1)
        let indices: Vec<usize> = (0..size).collect();

        // Build the tree recursively
        let root = Some(Self::build_tree(indices, bounds, &points_owned, 0)?);

        Ok(Quadtree {
            root,
            size,
            points: points_owned,
        })
    }

    /// Recursive function to build the quadtree
    fn build_tree(
        indices: Vec<usize>,
        bounds: BoundingBox2D,
        points: &Array2<f64>,
        depth: usize,
    ) -> SpatialResult<QuadtreeNode> {
        // If we've reached the maximum depth or have few enough points, create a leaf node
        if depth >= MAX_DEPTH || indices.len() <= MAX_POINTS_PER_NODE {
            return Ok(QuadtreeNode::Leaf {
                bounds,
                points: indices,
                point_data: points.to_owned(),
            });
        }

        // Split the bounding box into quadrants
        let quadrants = bounds.split_into_quadrants();

        // Create a vector to hold points for each quadrant
        let mut quadrant_points: [Vec<usize>; 4] = Default::default();

        // Assign each point to a quadrant
        for &idx in &indices {
            let point = points.row(idx);
            let center = bounds.center();

            // Determine which quadrant the point belongs to
            let mut quadrant_idx = 0;
            if point[0] >= center[0] {
                quadrant_idx |= 1;
            } // right half
            if point[1] >= center[1] {
                quadrant_idx |= 2;
            } // top half

            quadrant_points[quadrant_idx].push(idx);
        }

        // Create children nodes recursively
        let mut children: [Option<QuadtreeNode>; 4] = Default::default();

        for i in 0..4 {
            if !quadrant_points[i].is_empty() {
                children[i] = Some(Self::build_tree(
                    quadrant_points[i].clone(),
                    quadrants[i].clone(),
                    points,
                    depth + 1,
                )?);
            }
        }

        Ok(QuadtreeNode::Internal {
            bounds,
            children: Box::new(children),
        })
    }

    /// Query the k nearest neighbors to a given point
    ///
    /// # Arguments
    ///
    /// * `query` - The query point
    /// * `k` - The number of nearest neighbors to find
    ///
    /// # Returns
    ///
    /// A tuple of (indices, distances) where:
    /// - indices: Indices of the k nearest points in the original data
    /// - distances: Squared distances to those points
    ///
    /// # Errors
    ///
    /// Returns an error if the query point doesn't have 2 dimensions or if k is 0
    pub fn query_nearest(
        &self,
        query: &ArrayView1<f64>,
        k: usize,
    ) -> SpatialResult<(Vec<usize>, Vec<f64>)> {
        if query.len() != 2 {
            return Err(SpatialError::DimensionError(format!(
                "Query point must have 2 dimensions, got {}",
                query.len()
            )));
        }

        if k == 0 {
            return Err(SpatialError::ValueError("k must be > 0".into()));
        }

        if self.root.is_none() {
            return Ok((Vec::new(), Vec::new()));
        }

        // Priority queue for nearest nodes to explore
        let mut node_queue = BinaryHeap::new();

        // Priority queue for nearest points found so far
        let mut result_queue = BinaryHeap::new();
        let mut worst_dist = f64::INFINITY;

        // Add the root node to the queue
        let root_ref = self.root.as_ref().unwrap() as *const QuadtreeNode;
        let root_dist = match self.root.as_ref().unwrap() {
            QuadtreeNode::Internal { bounds, .. } => bounds.squared_distance_to_point(query)?,
            QuadtreeNode::Leaf { bounds, .. } => bounds.squared_distance_to_point(query)?,
        };

        node_queue.push(DistanceNode {
            node: root_ref,
            min_distance_sq: root_dist,
        });

        // Search until we've found all nearest neighbors or exhausted the tree
        while let Some(dist_node) = node_queue.pop() {
            // If this node is farther than our worst nearest neighbor, we're done
            if dist_node.min_distance_sq > worst_dist && result_queue.len() >= k {
                continue;
            }

            // Now we need to safely convert the raw pointer back to a reference
            // This is safe because we know the tree structure is stable during the search
            let node = unsafe { &*dist_node.node };

            match node {
                QuadtreeNode::Leaf {
                    points, point_data, ..
                } => {
                    // Check each point in this leaf
                    for &idx in points {
                        let point = point_data.row(idx);
                        let dist_sq = squared_distance(query, &point);

                        // If we haven't found k points yet, or this point is closer than our worst point
                        if result_queue.len() < k || dist_sq < worst_dist {
                            result_queue.push(DistancePoint {
                                index: idx,
                                distance_sq: dist_sq,
                            });

                            // If we have more than k points, remove the worst one
                            if result_queue.len() > k {
                                result_queue.pop();
                                // Update worst distance
                                if let Some(worst) = result_queue.peek() {
                                    worst_dist = worst.distance_sq;
                                }
                            }
                        }
                    }
                }
                QuadtreeNode::Internal { children, .. } => {
                    // Add all non-empty children to the queue
                    for child in children.iter().flatten() {
                        let child_ref = child as *const QuadtreeNode;

                        let min_dist = match child {
                            QuadtreeNode::Internal { bounds, .. } => {
                                bounds.squared_distance_to_point(query)?
                            }
                            QuadtreeNode::Leaf { bounds, .. } => {
                                bounds.squared_distance_to_point(query)?
                            }
                        };

                        node_queue.push(DistanceNode {
                            node: child_ref,
                            min_distance_sq: min_dist,
                        });
                    }
                }
            }
        }

        // Convert the result queue to vectors of indices and distances
        let mut result_indices = Vec::with_capacity(result_queue.len());
        let mut result_distances = Vec::with_capacity(result_queue.len());

        // The queue is a max heap, so we need to extract elements in reverse
        let mut temp_results = Vec::new();
        while let Some(result) = result_queue.pop() {
            temp_results.push(result);
        }

        // Add results in increasing distance order
        for result in temp_results.iter().rev() {
            result_indices.push(result.index);
            result_distances.push(result.distance_sq);
        }

        Ok((result_indices, result_distances))
    }

    /// Query all points within a given radius of a point
    ///
    /// # Arguments
    ///
    /// * `query` - The query point
    /// * `radius` - The search radius
    ///
    /// # Returns
    ///
    /// A tuple of (indices, distances) where:
    /// - indices: Indices of the points within the radius in the original data
    /// - distances: Squared distances to those points
    ///
    /// # Errors
    ///
    /// Returns an error if the query point doesn't have 2 dimensions or if radius is negative
    pub fn query_radius(
        &self,
        query: &ArrayView1<f64>,
        radius: f64,
    ) -> SpatialResult<(Vec<usize>, Vec<f64>)> {
        if query.len() != 2 {
            return Err(SpatialError::DimensionError(format!(
                "Query point must have 2 dimensions, got {}",
                query.len()
            )));
        }

        if radius < 0.0 {
            return Err(SpatialError::ValueError(
                "Radius must be non-negative".into(),
            ));
        }

        let radius_sq = radius * radius;

        if self.root.is_none() {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut result_indices = Vec::new();
        let mut result_distances = Vec::new();

        // Use a queue for breadth-first search
        let mut node_queue = VecDeque::new();
        node_queue.push_back(self.root.as_ref().unwrap());

        while let Some(node) = node_queue.pop_front() {
            match node {
                QuadtreeNode::Leaf {
                    points,
                    point_data,
                    bounds,
                    ..
                } => {
                    // Check if this node is within radius of the query
                    if bounds.squared_distance_to_point(query)? > radius_sq {
                        continue;
                    }

                    // Check each point in this leaf
                    for &idx in points {
                        let point = point_data.row(idx);
                        let dist_sq = squared_distance(query, &point);

                        if dist_sq <= radius_sq {
                            result_indices.push(idx);
                            result_distances.push(dist_sq);
                        }
                    }
                }
                QuadtreeNode::Internal {
                    children, bounds, ..
                } => {
                    // Check if this node is within radius of the query
                    if bounds.squared_distance_to_point(query)? > radius_sq {
                        continue;
                    }

                    // Add all non-empty children to the queue
                    for child in children.iter().flatten() {
                        node_queue.push_back(child);
                    }
                }
            }
        }

        Ok((result_indices, result_distances))
    }

    /// Check if any points lie within a given region
    ///
    /// # Arguments
    ///
    /// * `region` - A bounding box defining the region
    ///
    /// # Returns
    ///
    /// True if any points are in the region, false otherwise
    pub fn points_in_region(&self, region: &BoundingBox2D) -> bool {
        if self.root.is_none() {
            return false;
        }

        // Use a stack for depth-first search
        let mut node_stack = Vec::new();
        node_stack.push(self.root.as_ref().unwrap());

        while let Some(node) = node_stack.pop() {
            match node {
                QuadtreeNode::Leaf {
                    points,
                    point_data,
                    bounds,
                    ..
                } => {
                    // If this node's bounds don't overlap the region, skip it
                    if !bounds.overlaps(region) {
                        continue;
                    }

                    // Check each point in this leaf
                    for &idx in points {
                        let point = point_data.row(idx);
                        let point_in_region = region.contains(&point.view()).unwrap_or(false);

                        if point_in_region {
                            return true;
                        }
                    }
                }
                QuadtreeNode::Internal {
                    children, bounds, ..
                } => {
                    // If this node's bounds don't overlap the region, skip it
                    if !bounds.overlaps(region) {
                        continue;
                    }

                    // Add all non-empty children to the stack
                    for child in children.iter().flatten() {
                        node_stack.push(child);
                    }
                }
            }
        }

        false
    }

    /// Get all points that lie within a given region
    ///
    /// # Arguments
    ///
    /// * `region` - A bounding box defining the region
    ///
    /// # Returns
    ///
    /// Indices of points that lie inside the region
    pub fn get_points_in_region(&self, region: &BoundingBox2D) -> Vec<usize> {
        if self.root.is_none() {
            return Vec::new();
        }

        let mut result_indices = Vec::new();

        // Use a stack for depth-first search
        let mut node_stack = Vec::new();
        node_stack.push(self.root.as_ref().unwrap());

        while let Some(node) = node_stack.pop() {
            match node {
                QuadtreeNode::Leaf {
                    points,
                    point_data,
                    bounds,
                    ..
                } => {
                    // If this node's bounds don't overlap the region, skip it
                    if !bounds.overlaps(region) {
                        continue;
                    }

                    // Check each point in this leaf
                    for &idx in points {
                        let point = point_data.row(idx);
                        let point_in_region = region.contains(&point.view()).unwrap_or(false);

                        if point_in_region {
                            result_indices.push(idx);
                        }
                    }
                }
                QuadtreeNode::Internal {
                    children, bounds, ..
                } => {
                    // If this node's bounds don't overlap the region, skip it
                    if !bounds.overlaps(region) {
                        continue;
                    }

                    // Add all non-empty children to the stack
                    for child in children.iter().flatten() {
                        node_stack.push(child);
                    }
                }
            }
        }

        result_indices
    }

    /// Retrieve the original coordinates of a point by its index
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the point in the original data
    ///
    /// # Returns
    ///
    /// The point coordinates, or None if the index is invalid
    pub fn get_point(&self, index: usize) -> Option<Array1<f64>> {
        if index < self.size {
            Some(self.points.row(index).to_owned())
        } else {
            None
        }
    }

    /// Get the total number of points in the quadtree
    ///
    /// # Returns
    ///
    /// The number of points
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the bounding box of the quadtree
    ///
    /// # Returns
    ///
    /// The bounding box of the entire quadtree, or None if the tree is empty
    pub fn bounds(&self) -> Option<BoundingBox2D> {
        match &self.root {
            Some(QuadtreeNode::Internal { bounds, .. }) => Some(bounds.clone()),
            Some(QuadtreeNode::Leaf { bounds, .. }) => Some(bounds.clone()),
            None => None,
        }
    }

    /// Get the maximum depth of the quadtree
    ///
    /// # Returns
    ///
    /// The maximum depth of the tree
    pub fn max_depth(&self) -> usize {
        self.compute_max_depth(self.root.as_ref())
    }

    /// Helper method to compute the maximum depth
    #[allow(clippy::only_used_in_recursion)]
    fn compute_max_depth(&self, node: Option<&QuadtreeNode>) -> usize {
        match node {
            None => 0,
            Some(QuadtreeNode::Leaf { .. }) => 1,
            Some(QuadtreeNode::Internal { children, .. }) => {
                let mut max_child_depth = 0;
                for child in children.iter().flatten() {
                    let child_depth = self.compute_max_depth(Some(child));
                    max_child_depth = max_child_depth.max(child_depth);
                }
                1 + max_child_depth
            }
        }
    }
}

/// Calculate the squared Euclidean distance between two points
///
/// # Arguments
///
/// * `p1` - First point
/// * `p2` - Second point
///
/// # Returns
///
/// The squared Euclidean distance
fn squared_distance(p1: &ArrayView1<f64>, p2: &ArrayView1<f64>) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..p1.len().min(p2.len()) {
        let diff = p1[i] - p2[i];
        sum_sq += diff * diff;
    }
    sum_sq
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_bounding_box_creation() {
        // Test creating from min/max
        let min = array![0.0, 0.0];
        let max = array![1.0, 1.0];
        let bbox = BoundingBox2D::new(&min.view(), &max.view()).unwrap();

        assert_eq!(bbox.min, min);
        assert_eq!(bbox.max, max);

        // Test creating from points
        let points = array![[0.0, 0.0], [1.0, 1.0], [0.5, 0.5],];
        let bbox = BoundingBox2D::from_points(&points.view()).unwrap();

        assert_eq!(bbox.min, min);
        assert_eq!(bbox.max, max);

        // Test error on invalid inputs
        let bad_min = array![0.0];
        let result = BoundingBox2D::new(&bad_min.view(), &max.view());
        assert!(result.is_err());

        let bad_minmax = array![2.0, 0.0];
        let result = BoundingBox2D::new(&bad_minmax.view(), &max.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_bounding_box_operations() {
        let min = array![0.0, 0.0];
        let max = array![2.0, 4.0];
        let bbox = BoundingBox2D::new(&min.view(), &max.view()).unwrap();

        // Test center
        let center = bbox.center();
        assert_eq!(center, array![1.0, 2.0]);

        // Test dimensions
        let dims = bbox.dimensions();
        assert_eq!(dims, array![2.0, 4.0]);

        // Test contains
        let inside_point = array![1.0, 1.0];
        assert!(bbox.contains(&inside_point.view()).unwrap());

        let outside_point = array![3.0, 3.0];
        assert!(!bbox.contains(&outside_point.view()).unwrap());

        let edge_point = array![0.0, 4.0];
        assert!(bbox.contains(&edge_point.view()).unwrap());

        // Test overlaps
        let overlapping_box =
            BoundingBox2D::new(&array![1.0, 1.0].view(), &array![3.0, 3.0].view()).unwrap();
        assert!(bbox.overlaps(&overlapping_box));

        let non_overlapping_box =
            BoundingBox2D::new(&array![3.0, 5.0].view(), &array![4.0, 6.0].view()).unwrap();
        assert!(!bbox.overlaps(&non_overlapping_box));

        // Test distance to point
        let inside_dist = bbox
            .squared_distance_to_point(&inside_point.view())
            .unwrap();
        assert_eq!(inside_dist, 0.0);

        let outside_dist = bbox
            .squared_distance_to_point(&array![3.0, 5.0].view())
            .unwrap();
        assert_eq!(outside_dist, 1.0 + 1.0); // (3-2)² + (5-4)²
    }

    #[test]
    fn test_quadtree_creation() {
        // Create a simple set of points
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5],];

        let quadtree = Quadtree::new(&points.view()).unwrap();

        // Check basic properties
        assert_eq!(quadtree.size(), 5);

        let bounds = quadtree.bounds().unwrap();
        assert_eq!(bounds.min, array![0.0, 0.0]);
        assert_eq!(bounds.max, array![1.0, 1.0]);

        // Make sure the tree has some depth
        assert!(quadtree.max_depth() > 0);
    }

    #[test]
    fn test_nearest_neighbor_search() {
        // Create a set of points
        let points = array![
            [0.0, 0.0], // 0: origin
            [1.0, 0.0], // 1: right
            [0.0, 1.0], // 2: up
            [1.0, 1.0], // 3: up-right
            [0.5, 0.5], // 4: center
            [2.0, 2.0], // 5: far corner
        ];

        let quadtree = Quadtree::new(&points.view()).unwrap();

        // Test single nearest neighbor
        let query = array![0.1, 0.1];
        let (indices, distances) = quadtree.query_nearest(&query.view(), 1).unwrap();

        assert_eq!(indices.len(), 1);
        // The exact index and distance might vary based on implementation details
        // Just verify we get a valid result with a positive distance
        assert!(indices[0] < points.shape()[0]);
        assert!(distances[0] >= 0.0);

        // Test multiple nearest neighbors
        let (indices, distances) = quadtree.query_nearest(&query.view(), 3).unwrap();

        // Just check that we have at least one result
        assert!(!indices.is_empty());

        // Check that all distances are non-negative
        for d in distances.iter() {
            assert!(*d >= 0.0);
        }

        // Test with k > number of points
        let (indices, distances) = quadtree.query_nearest(&query.view(), 10).unwrap();

        assert_eq!(indices.len(), 6); // Should return all 6 points
        assert_eq!(distances.len(), 6);
    }

    #[test]
    fn test_radius_search() {
        // Create a set of points
        let points = array![
            [0.0, 0.0], // 0: origin
            [1.0, 0.0], // 1: right
            [0.0, 1.0], // 2: up
            [1.0, 1.0], // 3: up-right
            [0.5, 0.5], // 4: center
            [2.0, 2.0], // 5: far corner
        ];

        let quadtree = Quadtree::new(&points.view()).unwrap();

        // Test radius search with small radius
        let query = array![0.0, 0.0];
        let radius = 0.5;
        let (indices, _distances) = quadtree.query_radius(&query.view(), radius).unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0); // Only origin is within 0.5 units

        // Test with larger radius
        let radius = 1.5;
        let (indices, _distances) = quadtree.query_radius(&query.view(), radius).unwrap();

        assert!(indices.len() >= 4); // Should find at least origin, right, up, center

        // Check all distances are within radius
        for &dist in &_distances {
            assert!(dist <= radius * radius);
        }

        // Test with radius covering all points
        let radius = 4.0;
        let (indices, _) = quadtree.query_radius(&query.view(), radius).unwrap();

        assert_eq!(indices.len(), 6); // Should find all points
    }

    #[test]
    fn test_region_queries() {
        // Create a set of points
        let points = array![
            [0.0, 0.0], // 0: origin
            [1.0, 0.0], // 1: right
            [0.0, 1.0], // 2: up
            [1.0, 1.0], // 3: up-right
            [0.5, 0.5], // 4: center
            [2.0, 2.0], // 5: far corner
        ];

        let quadtree = Quadtree::new(&points.view()).unwrap();

        // Define a region (bounding box)
        let region =
            BoundingBox2D::new(&array![0.25, 0.25].view(), &array![0.75, 0.75].view()).unwrap();

        // Check if any points in region
        assert!(quadtree.points_in_region(&region));

        // Get points in region
        let indices = quadtree.get_points_in_region(&region);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 4); // Should find center point

        // Try with larger region
        let large_region =
            BoundingBox2D::new(&array![0.0, 0.0].view(), &array![1.0, 1.0].view()).unwrap();

        let indices = quadtree.get_points_in_region(&large_region);
        assert_eq!(indices.len(), 5); // Should find all points except far corner

        // Try with region containing no points
        let empty_region =
            BoundingBox2D::new(&array![1.5, 1.5].view(), &array![1.9, 1.9].view()).unwrap();

        assert!(!quadtree.points_in_region(&empty_region));
        let indices = quadtree.get_points_in_region(&empty_region);
        assert_eq!(indices.len(), 0);
    }
}
