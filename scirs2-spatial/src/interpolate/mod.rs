//! Spatial interpolation methods
//!
//! This module provides various methods for interpolating scattered data in
//! 2D and 3D space. These interpolation methods are useful for reconstructing
//! continuous fields from discrete sample points, filling gaps in data, and
//! generating smooth surfaces from irregularly sampled points.
//!
//! The available interpolation methods include:
//!
//! - Natural Neighbor interpolation: Local method that creates a weighted average
//!   of neighboring points based on their Voronoi cells. Produces smooth surfaces
//!   that respect the local structure of the data.
//!
//! - Radial Basis Function (RBF) interpolation: Uses radial basis functions to
//!   create a global interpolation that can represent complex surfaces. Various
//!   kernel functions can be selected to control the smoothness and locality of
//!   the interpolation.
//!
//! - Inverse Distance Weighting (IDW): Simple interpolation method that weights
//!   neighboring points by the inverse of their distance raised to a power. Fast
//!   but can create "bull's-eye" patterns around sample points.
//!
//! - Kriging (planned): Geostatistical method that accounts for the spatial
//!   correlation of data. Produces an interpolated surface along with an estimate
//!   of the prediction error.

// Re-export public modules
pub mod idw;
pub mod natural_neighbor;
pub mod rbf;

// Re-export key types and functions
pub use idw::IDWInterpolator;
pub use natural_neighbor::NaturalNeighborInterpolator;
pub use rbf::{RBFInterpolator, RBFKernel};
