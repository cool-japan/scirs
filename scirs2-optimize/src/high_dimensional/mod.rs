//! High-dimensional optimization methods
//!
//! This module provides optimization algorithms specifically designed for
//! high-dimensional problems where standard methods may be too expensive.
//!
//! ## Modules
//!
//! - [`coordinate_descent`]: Coordinate descent variants (cyclic, randomized, greedy, proximal, block)
//! - [`kaczmarz`]: Kaczmarz iteration for solving linear systems Ax = b
//! - [`sketched_gd`]: Sketched gradient descent with dimensionality reduction

pub mod coordinate_descent;
pub mod kaczmarz;
pub mod randomized_svd;
pub mod sketched_gd;
pub mod stochastic_coordinate;

pub use coordinate_descent::{
    BlockCoordinateDescent, CoordinateDescentConfig, CoordinateDescentResult,
    CoordinateDescentSolver, CoordinateSelectionStrategy, ProximalCoordinateDescent,
    RegularizationType,
};
pub use kaczmarz::{
    BlockKaczmarz, ExtendedKaczmarz, KaczmarzConfig, KaczmarzResult, KaczmarzSolver,
    KaczmarzVariant,
};
pub use sketched_gd::{SketchType, SketchedGdConfig, SketchedGdResult, SketchedGradientDescent};
