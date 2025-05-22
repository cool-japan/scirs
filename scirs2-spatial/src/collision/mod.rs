//! Collision detection algorithms for various geometric primitives
//!
//! This module provides implementations of collision detection algorithms
//! for common geometric primitives in 2D and 3D space. It supports both
//! discrete collision detection (testing if two objects intersect at a given
//! moment) and continuous collision detection (testing if two moving objects
//! will collide during a time interval).
//!
//! ## Features
//!
//! * Point collision tests with various shapes
//! * Line segment intersection tests
//! * Ray casting and intersection
//! * Collision detection between various geometric primitives
//! * Bounding volumes (spheres, axis-aligned bounding boxes)
//! * Continuous collision detection for moving objects
//!
//! ## Examples
//!
//! ### Testing if a point is inside a sphere
//!
//! ```
//! use scirs2_spatial::collision::{Sphere, point_sphere_collision};
//!
//! let sphere = Sphere {
//!     center: [0.0, 0.0, 0.0],
//!     radius: 2.0,
//! };
//!
//! let point = [1.0, 1.0, 1.0];
//! let inside = point_sphere_collision(&point, &sphere);
//!
//! println!("Is the point inside the sphere? {}", inside);
//! ```
//!
//! ### Testing if two circles collide
//!
//! ```
//! use scirs2_spatial::collision::{Circle, circle_circle_collision};
//!
//! let circle1 = Circle {
//!     center: [0.0, 0.0],
//!     radius: 2.0,
//! };
//!
//! let circle2 = Circle {
//!     center: [3.0, 0.0],
//!     radius: 1.5,
//! };
//!
//! let collide = circle_circle_collision(&circle1, &circle2);
//! println!("Do the circles collide? {}", collide);
//! ```

// Re-export all public items from submodules
pub use self::broadphase::*;
pub use self::continuous::*;
pub use self::narrowphase::*;
pub use self::response::*;
pub use self::shapes::*;

// Public modules
pub mod broadphase;
pub mod continuous;
pub mod narrowphase;
pub mod response;
pub mod shapes;

// Tests
mod tests;
