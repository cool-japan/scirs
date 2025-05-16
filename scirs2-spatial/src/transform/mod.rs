//! Spatial transformations module
//!
//! This module provides functionality for spatial transformations including rotations,
//! rigid transforms (rotation + translation), interpolation between rotations,
//! and conversions between Cartesian and spherical coordinate systems.
//!
//! The module is inspired by SciPy's `scipy.spatial.transform` module and provides similar
//! functionality in a Rust-idiomatic way.

// Public modules
mod rigid_transform;
mod rotation;
mod rotation_spline;
mod slerp;
pub mod spherical;

// Re-exports for public API
pub use rigid_transform::RigidTransform;
pub use rotation::Rotation;
pub use rotation_spline::RotationSpline;
pub use slerp::Slerp;
