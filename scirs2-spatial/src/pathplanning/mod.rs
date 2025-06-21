//! Path planning algorithms
//!
//! This module provides various path planning algorithms for finding optimal
//! paths in 2D and 3D space. These algorithms are useful for robotics, game
//! development, logistics, and any application that requires navigating through
//! environments with obstacles.
//!
//! The available algorithms include:
//!
//! - A* search: An efficient, best-first search algorithm that uses a heuristic
//!   to guide the search towards the goal. Guarantees the optimal path if the
//!   heuristic is admissible.
//!
//! - RRT (Rapidly-exploring Random Tree): A sampling-based algorithm that
//!   efficiently explores high-dimensional spaces by incrementally building a
//!   space-filling tree.
//!
//! - Visibility graphs: A graph-based approach that connects visible vertices
//!   of obstacles, providing optimal paths in 2D polygonal environments.
//!
//! - Probabilistic roadmaps (PRM): A sampling-based method that constructs a roadmap
//!   of the free configuration space, then uses graph search to find paths.
//!
//! - Potential fields: A reactive approach that uses artificial potential
//!   fields to guide the agent towards the goal while avoiding obstacles.
//!
//! - Dubins paths: Shortest paths for vehicles with minimum turning radius constraint
//!   (forward motion only).
//!
//! - Reeds-Shepp paths: Shortest paths for vehicles with minimum turning radius
//!   constraint that can move both forward and backward.
//!
//! - Trajectory optimization: Smooth trajectory generation with kinematic and
//!   dynamic constraints.

// Re-export public modules
pub mod astar;
pub mod dubins;
pub mod potentialfield;
pub mod prm;
pub mod reedshepp;
pub mod rrt;
pub mod trajectory;
pub mod visibility;

// Re-export key types and functions
pub use astar::{
    AStarPlanner, ContinuousAStarPlanner, GridAStarPlanner, HashableFloat2D, Node, Path,
};
pub use dubins::{DubinsPath, DubinsPathType, DubinsPlanner, DubinsSegment, Pose2D, SegmentType};
pub use potentialfield::{
    CircularObstacle, Obstacle, PolygonObstacle, PotentialConfig, PotentialField2DPlanner,
    PotentialFieldPlanner,
};
pub use prm::{PRM2DPlanner, PRMConfig, PRMPlanner};
pub use reedshepp::{
    Motion, ReedsSheppPath, ReedsSheppPathType, ReedsSheppPlanner, ReedsSheppSegment, Turn,
};
pub use rrt::{RRT2DPlanner, RRTConfig, RRTPlanner};
pub use trajectory::{
    CircularObstacle as TrajectoryObstacle, OptimizationMethod, Trajectory, TrajectoryConstraints,
    TrajectoryOptimizer, TrajectoryPoint,
};
pub use visibility::{Point2D, VisibilityGraphPlanner};
