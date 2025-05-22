//! Spatial algorithms module
//!
//! This module provides implementations of various spatial algorithms,
//! similar to SciPy's `spatial` module.
//!
//! ## Overview
//!
//! * Distance computations and metrics
//! * KD-trees for efficient nearest neighbor searches
//! * Ball trees for high-dimensional nearest neighbor searches
//! * Voronoi diagrams and Delaunay triangulation
//! * Convex hulls
//! * Set-based distances (Hausdorff, Wasserstein)
//! * Polygon operations
//! * Spatial transformations
//! * Path planning algorithms
//!
//! ## Features
//!
//! * Efficient nearest-neighbor queries with KD-Tree and Ball Tree data structures
//! * Comprehensive set of distance metrics (Euclidean, Manhattan, Minkowski, etc.)
//! * Distance matrix computations (similar to SciPy's cdist and pdist)
//! * Convex hull computation using the Qhull library
//! * Delaunay triangulation for 2D and higher dimensions
//! * Customizable distance metrics for spatial data structures
//! * Advanced query capabilities (k-nearest neighbors, radius search)
//! * Set-based distances (Hausdorff, Wasserstein)
//! * Polygon operations (point-in-polygon, area, centroid)
//! * Path planning algorithms (A*, RRT, visibility graphs)
//!
//! ## Examples
//!
//! ### Distance Metrics
//!
//! ```
//! use scirs2_spatial::distance::euclidean;
//!
//! let point1 = &[1.0, 2.0, 3.0];
//! let point2 = &[4.0, 5.0, 6.0];
//!
//! let dist = euclidean(point1, point2);
//! println!("Euclidean distance: {}", dist);
//! ```
//!
//! ### KD-Tree for Nearest Neighbor Searches
//!
//! ```
//! use scirs2_spatial::KDTree;
//! use ndarray::array;
//!
//! // Create points
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
//!
//! // Build KD-Tree
//! let kdtree = KDTree::new(&points).unwrap();
//!
//! // Find 2 nearest neighbors to [0.5, 0.5]
//! let (indices, distances) = kdtree.query(&[0.5, 0.5], 2).unwrap();
//! println!("Indices of 2 nearest points: {:?}", indices);
//! println!("Distances to 2 nearest points: {:?}", distances);
//!
//! // Find all points within radius 0.7
//! let (idx_radius, dist_radius) = kdtree.query_radius(&[0.5, 0.5], 0.7).unwrap();
//! println!("Found {} points within radius 0.7", idx_radius.len());
//! ```
//!
//! ### Distance Matrices
//!
//! ```
//! use scirs2_spatial::distance::{pdist, euclidean};
//! use ndarray::array;
//!
//! // Create points
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
//!
//! // Calculate pairwise distance matrix
//! let dist_matrix = pdist(&points, euclidean);
//! println!("Distance matrix shape: {:?}", dist_matrix.shape());
//! ```
//!
//! ### Convex Hull
//!
//! ```
//! use scirs2_spatial::convex_hull::ConvexHull;
//! use ndarray::array;
//!
//! // Create points
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
//!
//! // Compute convex hull
//! let hull = ConvexHull::new(&points.view()).unwrap();
//!
//! // Get the hull vertices
//! let vertices = hull.vertices();
//! println!("Hull vertices: {:?}", vertices);
//!
//! // Check if a point is inside the hull
//! let is_inside = hull.contains(&[0.25, 0.25]).unwrap();
//! println!("Is point [0.25, 0.25] inside? {}", is_inside);
//! ```
//!
//! ### Delaunay Triangulation
//!
//! ```
//! use scirs2_spatial::delaunay::Delaunay;
//! use ndarray::array;
//!
//! // Create points
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
//!
//! // Compute Delaunay triangulation
//! let tri = Delaunay::new(&points).unwrap();
//!
//! // Get the simplices (triangles in 2D)
//! let simplices = tri.simplices();
//! println!("Triangles: {:?}", simplices);
//!
//! // Find which triangle contains a point
//! if let Some(idx) = tri.find_simplex(&[0.25, 0.25]) {
//!     println!("Point [0.25, 0.25] is in triangle {}", idx);
//! }
//! ```
//!
//! ### Set-Based Distances
//!
//! ```
//! use scirs2_spatial::set_distance::hausdorff_distance;
//! use ndarray::array;
//!
//! // Create two point sets
//! let set1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
//! let set2 = array![[0.0, 0.5], [1.0, 0.5], [0.5, 1.0]];
//!
//! // Compute the Hausdorff distance
//! let dist = hausdorff_distance(&set1.view(), &set2.view(), None);
//! println!("Hausdorff distance: {}", dist);
//! ```
//!
//! ### Polygon Operations
//!
//! ```
//! use scirs2_spatial::polygon::{point_in_polygon, polygon_area, polygon_centroid};
//! use ndarray::array;
//!
//! // Create a polygon (square)
//! let polygon = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
//!
//! // Check if a point is inside
//! let inside = point_in_polygon(&[0.5, 0.5], &polygon.view());
//! println!("Is point [0.5, 0.5] inside? {}", inside);
//!
//! // Calculate polygon area
//! let area = polygon_area(&polygon.view());
//! println!("Polygon area: {}", area);
//!
//! // Calculate centroid
//! let centroid = polygon_centroid(&polygon.view());
//! println!("Polygon centroid: ({}, {})", centroid[0], centroid[1]);
//! ```
//!
//! ### Ball Tree for Nearest Neighbor Searches
//!
//! ```
//! use scirs2_spatial::BallTree;
//! use ndarray::array;
//!
//! // Create points
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
//!
//! // Build Ball Tree
//! let ball_tree = BallTree::with_euclidean_distance(&points.view(), 2).unwrap();
//!
//! // Find 2 nearest neighbors to [0.5, 0.5]
//! let (indices, distances) = ball_tree.query(&[0.5, 0.5], 2, true).unwrap();
//! println!("Indices of 2 nearest points: {:?}", indices);
//! println!("Distances to 2 nearest points: {:?}", distances.unwrap());
//!
//! // Find all points within radius 0.7
//! let (idx_radius, dist_radius) = ball_tree.query_radius(&[0.5, 0.5], 0.7, true).unwrap();
//! println!("Found {} points within radius 0.7", idx_radius.len());
//! ```
//!
//! ### A* Pathfinding
//!
//! ```
//! use scirs2_spatial::pathplanning::GridAStarPlanner;
//!
//! // Create a grid with some obstacles (true = obstacle, false = free space)
//! let grid = vec![
//!     vec![false, false, false, false, false],
//!     vec![false, false, false, false, false],
//!     vec![false, true, true, true, false],  // A wall of obstacles
//!     vec![false, false, false, false, false],
//!     vec![false, false, false, false, false],
//! ];
//!
//! // Create an A* planner with the grid
//! let planner = GridAStarPlanner::new(grid, false);
//!
//! // Find a path from top-left to bottom-right
//! let start = [0, 0];
//! let goal = [4, 4];
//!
//! let path = planner.find_path(start, goal).unwrap().unwrap();
//!
//! println!("Found a path with {} steps:", path.len() - 1);
//! for (i, pos) in path.nodes.iter().enumerate() {
//!     println!("  Step {}: {:?}", i, pos);
//! }
//! ```
//!
//! ### RRT Pathfinding
//!
//! ```
//! use scirs2_spatial::pathplanning::{RRTConfig, RRT2DPlanner};
//!
//! // Create a configuration for RRT
//! let config = RRTConfig {
//!     max_iterations: 1000,
//!     step_size: 0.3,
//!     goal_bias: 0.1,
//!     seed: Some(42),
//!     use_rrt_star: false,
//!     neighborhood_radius: None,
//!     bidirectional: false,
//! };
//!
//! // Define obstacles as polygons
//! let obstacles = vec![
//!     // Rectangle obstacle
//!     vec![[3.0, 2.0], [3.0, 4.0], [4.0, 4.0], [4.0, 2.0]],
//! ];
//!
//! // Create RRT planner
//! let mut planner = RRT2DPlanner::new(
//!     config,
//!     obstacles,
//!     [0.0, 0.0],   // Min bounds
//!     [10.0, 10.0], // Max bounds
//!     0.1,          // Collision checking step size
//! ).unwrap();
//!
//! // Find a path from start to goal
//! let start = [1.0, 3.0];
//! let goal = [8.0, 3.0];
//! let goal_threshold = 0.5;
//!
//! let path = planner.find_path(start, goal, goal_threshold).unwrap().unwrap();
//!
//! println!("Found a path with {} segments:", path.len() - 1);
//! for (i, pos) in path.nodes.iter().enumerate() {
//!     println!("  Point {}: [{:.2}, {:.2}]", i, pos[0], pos[1]);
//! }
//! ```

// Export error types
pub mod error;
pub use error::{SpatialError, SpatialResult};

// Distance metrics
pub mod distance;
pub use distance::{
    canberra,
    cdist,
    chebyshev,
    correlation,
    cosine,
    // Convenience functions
    euclidean,
    is_valid_condensed_distance_matrix,
    jaccard,

    manhattan,
    minkowski,
    // Distance matrix computation
    pdist,
    sqeuclidean,
    squareform,
    squareform_to_condensed,
    ChebyshevDistance,
    // Core distance traits and structs
    Distance,
    EuclideanDistance,
    ManhattanDistance,
    MinkowskiDistance,
};

// KD-Tree for efficient nearest neighbor searches
pub mod kdtree;
pub use kdtree::{KDTree, Rectangle};

// KD-Tree optimizations for spatial operations
pub mod kdtree_optimized;
pub use kdtree_optimized::KDTreeOptimized;

// Ball-Tree for efficient nearest neighbor searches in high dimensions
pub mod balltree;
pub use balltree::BallTree;

// Delaunay triangulation
pub mod delaunay;
pub use delaunay::Delaunay;

// Voronoi diagrams
pub mod voronoi;
pub use voronoi::{voronoi, Voronoi};

// Spherical Voronoi diagrams
pub mod spherical_voronoi;
pub use spherical_voronoi::SphericalVoronoi;

// Procrustes analysis
pub mod procrustes;
pub use procrustes::{procrustes, procrustes_extended, ProcrustesParams};

// Convex hull computation
pub mod convex_hull;
pub use convex_hull::{convex_hull, ConvexHull};

// Set-based distance metrics
pub mod set_distance;
pub use set_distance::{
    directed_hausdorff, gromov_hausdorff_distance, hausdorff_distance, wasserstein_distance,
};

// Polygon operations
pub mod polygon;
pub use polygon::{
    convex_hull_graham, is_simple_polygon, point_in_polygon, point_on_boundary, polygon_area,
    polygon_centroid, polygon_contains_polygon,
};

// R-tree for efficient spatial indexing
pub mod rtree;
pub use rtree::{RTree, Rectangle as RTreeRectangle};

// Octree for 3D spatial searches
pub mod octree;
pub use octree::{BoundingBox, Octree};

// Quadtree for 2D spatial searches
pub mod quadtree;
pub use quadtree::{BoundingBox2D, Quadtree};

// Spatial interpolation methods
pub mod interpolate;
pub use interpolate::{IDWInterpolator, NaturalNeighborInterpolator, RBFInterpolator, RBFKernel};

// Path planning algorithms
pub mod pathplanning;
pub use pathplanning::astar::{AStarPlanner, ContinuousAStarPlanner, GridAStarPlanner, Node, Path};
pub use pathplanning::rrt::{RRT2DPlanner, RRTConfig, RRTPlanner};

// Spatial transformations
pub mod transform;

// Collision detection
pub mod collision;
// Re-export shapes for convenience
pub use collision::shapes::{
    Box2D, Box3D, Circle, LineSegment2D, LineSegment3D, Sphere, Triangle2D, Triangle3D,
};
// Re-export narrowphase collision functions
pub use collision::narrowphase::{
    box2d_box2d_collision, box3d_box3d_collision, circle_box2d_collision, circle_circle_collision,
    point_box2d_collision, point_box3d_collision, point_circle_collision, point_sphere_collision,
    point_triangle2d_collision, ray_box3d_collision, ray_sphere_collision,
    ray_triangle3d_collision, sphere_box3d_collision, sphere_sphere_collision,
};
// Re-export continuous collision functions
pub use collision::continuous::continuous_sphere_sphere_collision;

// Utility functions
mod utils;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
