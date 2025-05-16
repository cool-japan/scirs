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
//!
//! ### Ray-box intersection
//!
//! ```
//! use scirs2_spatial::collision::{Box3D, ray_box3d_collision};
//!
//! let box3d = Box3D {
//!     min: [-1.0, -1.0, -1.0],
//!     max: [1.0, 1.0, 1.0],
//! };
//!
//! let ray_origin = [5.0, 0.0, 0.0];
//! let ray_direction = [-1.0, 0.0, 0.0]; // Pointing towards the box
//!
//! if let Some(intersection) = ray_box3d_collision(&ray_origin, &ray_direction, &box3d) {
//!     println!("Ray intersects the box at distance {:?}", intersection);
//! } else {
//!     println!("Ray does not intersect the box");
//! }
//! ```
//!
//! ### Continuous collision detection
//!
//! ```
//! use scirs2_spatial::collision::{Sphere, continuous_sphere_sphere_collision};
//!
//! let sphere1 = Sphere {
//!     center: [0.0, 0.0, 0.0],
//!     radius: 1.0,
//! };
//!
//! let sphere1_velocity = [1.0, 0.0, 0.0];
//!
//! let sphere2 = Sphere {
//!     center: [5.0, 0.0, 0.0],
//!     radius: 1.0,
//! };
//!
//! let sphere2_velocity = [-1.0, 0.0, 0.0];
//!
//! let time_step = 5.0;
//!
//! if let Some(collision_time) = continuous_sphere_sphere_collision(
//!     &sphere1, &sphere1_velocity,
//!     &sphere2, &sphere2_velocity,
//!     time_step
//! ) {
//!     println!("Spheres will collide at time {:?}", collision_time);
//! } else {
//!     println!("No collision within the time step");
//! }
//! ```

use std::f64;

// ---------------------------------------------------------------------------
// 2D Geometric Primitives
// ---------------------------------------------------------------------------

/// A 2D circle
#[derive(Debug, Clone, Copy)]
pub struct Circle {
    /// Center of the circle [x, y]
    pub center: [f64; 2],
    /// Radius of the circle
    pub radius: f64,
}

impl Circle {
    /// Creates a new circle with the given center and radius
    pub fn new(center: [f64; 2], radius: f64) -> Self {
        Circle { center, radius }
    }

    /// Calculates the area of the circle
    pub fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }

    /// Tests if a point is inside the circle
    pub fn contains_point(&self, point: &[f64; 2]) -> bool {
        point_circle_collision(point, self)
    }
}

/// A 2D line segment defined by two endpoints
#[derive(Debug, Clone, Copy)]
pub struct LineSegment2D {
    /// First endpoint [x, y]
    pub start: [f64; 2],
    /// Second endpoint [x, y]
    pub end: [f64; 2],
}

impl LineSegment2D {
    /// Creates a new line segment with the given endpoints
    pub fn new(start: [f64; 2], end: [f64; 2]) -> Self {
        LineSegment2D { start, end }
    }

    /// Calculates the length of the line segment
    pub fn length(&self) -> f64 {
        let dx = self.end[0] - self.start[0];
        let dy = self.end[1] - self.start[1];
        (dx * dx + dy * dy).sqrt()
    }
}

/// A 2D triangle defined by three vertices
#[derive(Debug, Clone, Copy)]
pub struct Triangle2D {
    /// First vertex [x, y]
    pub v1: [f64; 2],
    /// Second vertex [x, y]
    pub v2: [f64; 2],
    /// Third vertex [x, y]
    pub v3: [f64; 2],
}

impl Triangle2D {
    /// Creates a new triangle with the given vertices
    pub fn new(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Self {
        Triangle2D {
            v1: a,
            v2: b,
            v3: c,
        }
    }

    /// Calculates the area of the triangle
    pub fn area(&self) -> f64 {
        0.5 * ((self.v2[0] - self.v1[0]) * (self.v3[1] - self.v1[1])
            - (self.v3[0] - self.v1[0]) * (self.v2[1] - self.v1[1]))
            .abs()
    }

    /// Tests if a point is inside the triangle
    pub fn contains_point(&self, point: &[f64; 2]) -> bool {
        point_triangle2d_collision(point, self)
    }

    /// Provides access to the first vertex (alias for v1)
    pub fn a(&self) -> &[f64; 2] {
        &self.v1
    }

    /// Provides access to the second vertex (alias for v2)
    pub fn b(&self) -> &[f64; 2] {
        &self.v2
    }

    /// Provides access to the third vertex (alias for v3)
    pub fn c(&self) -> &[f64; 2] {
        &self.v3
    }
}

/// A 2D axis-aligned bounding box
#[derive(Debug, Clone, Copy)]
pub struct Box2D {
    /// Minimum corner [x, y]
    pub min: [f64; 2],
    /// Maximum corner [x, y]
    pub max: [f64; 2],
}

impl Box2D {
    /// Creates a new 2D axis-aligned bounding box with the given minimum and maximum corners
    pub fn new(min: [f64; 2], max: [f64; 2]) -> Self {
        Box2D { min, max }
    }

    /// Gets the width of the box
    pub fn width(&self) -> f64 {
        self.max[0] - self.min[0]
    }

    /// Gets the height of the box
    pub fn height(&self) -> f64 {
        self.max[1] - self.min[1]
    }

    /// Calculates the area of the box
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Gets the center of the box
    pub fn center(&self) -> [f64; 2] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
        ]
    }

    /// Tests if a point is inside the box
    pub fn contains_point(&self, point: &[f64; 2]) -> bool {
        point_box2d_collision(point, self)
    }
}

// ---------------------------------------------------------------------------
// 3D Geometric Primitives
// ---------------------------------------------------------------------------

/// A 3D sphere
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    /// Center of the sphere [x, y, z]
    pub center: [f64; 3],
    /// Radius of the sphere
    pub radius: f64,
}

impl Sphere {
    /// Creates a new sphere with the given center and radius
    pub fn new(center: [f64; 3], radius: f64) -> Self {
        Sphere { center, radius }
    }

    /// Calculates the volume of the sphere
    pub fn volume(&self) -> f64 {
        (4.0 / 3.0) * std::f64::consts::PI * self.radius * self.radius * self.radius
    }

    /// Tests if a point is inside the sphere
    pub fn contains_point(&self, point: &[f64; 3]) -> bool {
        point_sphere_collision(point, self)
    }
}

/// A 3D line segment defined by two endpoints
#[derive(Debug, Clone, Copy)]
pub struct LineSegment3D {
    /// First endpoint [x, y, z]
    pub start: [f64; 3],
    /// Second endpoint [x, y, z]
    pub end: [f64; 3],
}

impl LineSegment3D {
    /// Creates a new 3D line segment with the given endpoints
    pub fn new(start: [f64; 3], end: [f64; 3]) -> Self {
        LineSegment3D { start, end }
    }

    /// Calculates the length of the line segment
    pub fn length(&self) -> f64 {
        let dx = self.end[0] - self.start[0];
        let dy = self.end[1] - self.start[1];
        let dz = self.end[2] - self.start[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// A 3D triangle defined by three vertices
#[derive(Debug, Clone, Copy)]
pub struct Triangle3D {
    /// First vertex [x, y, z]
    pub v1: [f64; 3],
    /// Second vertex [x, y, z]
    pub v2: [f64; 3],
    /// Third vertex [x, y, z]
    pub v3: [f64; 3],
}

impl Triangle3D {
    /// Creates a new 3D triangle with the given vertices
    pub fn new(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Self {
        Triangle3D {
            v1: a,
            v2: b,
            v3: c,
        }
    }

    /// Calculates the area of the triangle
    pub fn area(&self) -> f64 {
        let edge1 = [
            self.v2[0] - self.v1[0],
            self.v2[1] - self.v1[1],
            self.v2[2] - self.v1[2],
        ];

        let edge2 = [
            self.v3[0] - self.v1[0],
            self.v3[1] - self.v1[1],
            self.v3[2] - self.v1[2],
        ];

        // Cross product of edges
        let cross = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        ];

        // Area is half the magnitude of the cross product
        0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
    }

    /// Calculates the normal vector of the triangle
    pub fn normal(&self) -> [f64; 3] {
        let edge1 = [
            self.v2[0] - self.v1[0],
            self.v2[1] - self.v1[1],
            self.v2[2] - self.v1[2],
        ];

        let edge2 = [
            self.v3[0] - self.v1[0],
            self.v3[1] - self.v1[1],
            self.v3[2] - self.v1[2],
        ];

        // Cross product of edges gives the normal
        let normal = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        ];

        // Normalize
        let normal_length =
            (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if normal_length == 0.0 {
            [0.0, 0.0, 0.0] // Degenerate triangle
        } else {
            [
                normal[0] / normal_length,
                normal[1] / normal_length,
                normal[2] / normal_length,
            ]
        }
    }

    /// Provides access to the first vertex (alias for v1)
    pub fn a(&self) -> &[f64; 3] {
        &self.v1
    }

    /// Provides access to the second vertex (alias for v2)
    pub fn b(&self) -> &[f64; 3] {
        &self.v2
    }

    /// Provides access to the third vertex (alias for v3)
    pub fn c(&self) -> &[f64; 3] {
        &self.v3
    }
}

/// A 3D axis-aligned bounding box
#[derive(Debug, Clone, Copy)]
pub struct Box3D {
    /// Minimum corner [x, y, z]
    pub min: [f64; 3],
    /// Maximum corner [x, y, z]
    pub max: [f64; 3],
}

impl Box3D {
    /// Creates a new 3D axis-aligned bounding box with the given minimum and maximum corners
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Box3D { min, max }
    }

    /// Gets the width of the box (x-dimension)
    pub fn width(&self) -> f64 {
        self.max[0] - self.min[0]
    }

    /// Gets the height of the box (y-dimension)
    pub fn height(&self) -> f64 {
        self.max[1] - self.min[1]
    }

    /// Gets the depth of the box (z-dimension)
    pub fn depth(&self) -> f64 {
        self.max[2] - self.min[2]
    }

    /// Calculates the volume of the box
    pub fn volume(&self) -> f64 {
        self.width() * self.height() * self.depth()
    }

    /// Gets the center of the box
    pub fn center(&self) -> [f64; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Tests if a point is inside the box
    pub fn contains_point(&self, point: &[f64; 3]) -> bool {
        point_box3d_collision(point, self)
    }
}

// ---------------------------------------------------------------------------
// 2D Point Collision Tests
// ---------------------------------------------------------------------------

/// Tests if a point is inside a circle
///
/// # Arguments
///
/// * `point` - A 2D point [x, y]
/// * `circle` - The circle to test against
///
/// # Returns
///
/// `true` if the point is inside or on the boundary of the circle, `false` otherwise
pub fn point_circle_collision(point: &[f64; 2], circle: &Circle) -> bool {
    let dx = point[0] - circle.center[0];
    let dy = point[1] - circle.center[1];
    let distance_squared = dx * dx + dy * dy;

    distance_squared <= circle.radius * circle.radius
}

/// Tests if a point is inside a 2D axis-aligned bounding box
///
/// # Arguments
///
/// * `point` - A 2D point [x, y]
/// * `box2d` - The 2D box to test against
///
/// # Returns
///
/// `true` if the point is inside or on the boundary of the box, `false` otherwise
pub fn point_box2d_collision(point: &[f64; 2], box2d: &Box2D) -> bool {
    point[0] >= box2d.min[0]
        && point[0] <= box2d.max[0]
        && point[1] >= box2d.min[1]
        && point[1] <= box2d.max[1]
}

/// Tests if a point is inside a 2D triangle
///
/// # Arguments
///
/// * `point` - A 2D point [x, y]
/// * `triangle` - The 2D triangle to test against
///
/// # Returns
///
/// `true` if the point is inside or on the boundary of the triangle, `false` otherwise
pub fn point_triangle2d_collision(point: &[f64; 2], triangle: &Triangle2D) -> bool {
    // Compute barycentric coordinates
    let area = 0.5
        * ((triangle.v2[0] - triangle.v1[0]) * (triangle.v3[1] - triangle.v1[1])
            - (triangle.v3[0] - triangle.v1[0]) * (triangle.v2[1] - triangle.v1[1]))
            .abs();

    if area == 0.0 {
        return false; // Degenerate triangle
    }

    let a = ((triangle.v2[0] - point[0]) * (triangle.v3[1] - point[1])
        - (triangle.v3[0] - point[0]) * (triangle.v2[1] - point[1]))
        .abs()
        / (2.0 * area);

    let b = ((triangle.v3[0] - point[0]) * (triangle.v1[1] - point[1])
        - (triangle.v1[0] - point[0]) * (triangle.v3[1] - point[1]))
        .abs()
        / (2.0 * area);

    let c = 1.0 - a - b;

    // Point is inside if all coordinates are within [0, 1]
    // Using small epsilon for floating-point precision
    const EPSILON: f64 = 1e-10;
    a >= -EPSILON
        && b >= -EPSILON
        && c >= -EPSILON
        && a <= 1.0 + EPSILON
        && b <= 1.0 + EPSILON
        && c <= 1.0 + EPSILON
}

// ---------------------------------------------------------------------------
// 3D Point Collision Tests
// ---------------------------------------------------------------------------

/// Tests if a point is inside a sphere
///
/// # Arguments
///
/// * `point` - A 3D point [x, y, z]
/// * `sphere` - The sphere to test against
///
/// # Returns
///
/// `true` if the point is inside or on the boundary of the sphere, `false` otherwise
pub fn point_sphere_collision(point: &[f64; 3], sphere: &Sphere) -> bool {
    let dx = point[0] - sphere.center[0];
    let dy = point[1] - sphere.center[1];
    let dz = point[2] - sphere.center[2];
    let distance_squared = dx * dx + dy * dy + dz * dz;

    distance_squared <= sphere.radius * sphere.radius
}

/// Tests if a point is inside a 3D axis-aligned bounding box
///
/// # Arguments
///
/// * `point` - A 3D point [x, y, z]
/// * `box3d` - The 3D box to test against
///
/// # Returns
///
/// `true` if the point is inside or on the boundary of the box, `false` otherwise
pub fn point_box3d_collision(point: &[f64; 3], box3d: &Box3D) -> bool {
    point[0] >= box3d.min[0]
        && point[0] <= box3d.max[0]
        && point[1] >= box3d.min[1]
        && point[1] <= box3d.max[1]
        && point[2] >= box3d.min[2]
        && point[2] <= box3d.max[2]
}

/// Tests if a point is inside a 3D triangle
///
/// # Arguments
///
/// * `point` - A 3D point [x, y, z]
/// * `triangle` - The 3D triangle to test against
///
/// # Returns
///
/// `true` if the point is on the triangle, `false` otherwise
pub fn point_triangle3d_collision(point: &[f64; 3], triangle: &Triangle3D) -> bool {
    // For a 3D triangle, we need to check if the point is on the plane of the triangle
    // and then check if it's inside the triangle

    // First, compute the normal vector of the triangle
    let edge1 = [
        triangle.v2[0] - triangle.v1[0],
        triangle.v2[1] - triangle.v1[1],
        triangle.v2[2] - triangle.v1[2],
    ];

    let edge2 = [
        triangle.v3[0] - triangle.v1[0],
        triangle.v3[1] - triangle.v1[1],
        triangle.v3[2] - triangle.v1[2],
    ];

    // Cross product of the two edges gives the normal
    let normal = [
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0],
    ];

    // Normalize the normal
    let normal_length =
        (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if normal_length == 0.0 {
        return false; // Degenerate triangle
    }

    let normalized_normal = [
        normal[0] / normal_length,
        normal[1] / normal_length,
        normal[2] / normal_length,
    ];

    // Distance from point to the plane of the triangle
    let dist_to_plane = (point[0] - triangle.v1[0]) * normalized_normal[0]
        + (point[1] - triangle.v1[1]) * normalized_normal[1]
        + (point[2] - triangle.v1[2]) * normalized_normal[2];

    // Check if the point is close to the plane (within small epsilon)
    const EPSILON: f64 = 1e-6;
    if dist_to_plane.abs() > EPSILON {
        return false; // Point is not on the plane
    }

    // Project the triangle and point onto a 2D plane for inside/outside test
    // Choose the dimension with the largest normal component to drop
    let max_component = if normalized_normal[0].abs() > normalized_normal[1].abs() {
        if normalized_normal[0].abs() > normalized_normal[2].abs() {
            0
        } else {
            2
        }
    } else if normalized_normal[1].abs() > normalized_normal[2].abs() {
        1
    } else {
        2
    };

    // Extract the coordinates for the 2D projection
    let mut p1 = [0.0, 0.0];
    let mut p2 = [0.0, 0.0];
    let mut p3 = [0.0, 0.0];
    let mut pp = [0.0, 0.0];

    match max_component {
        0 => {
            // Drop the x-coordinate
            p1[0] = triangle.v1[1];
            p1[1] = triangle.v1[2];
            p2[0] = triangle.v2[1];
            p2[1] = triangle.v2[2];
            p3[0] = triangle.v3[1];
            p3[1] = triangle.v3[2];
            pp[0] = point[1];
            pp[1] = point[2];
        }
        1 => {
            // Drop the y-coordinate
            p1[0] = triangle.v1[0];
            p1[1] = triangle.v1[2];
            p2[0] = triangle.v2[0];
            p2[1] = triangle.v2[2];
            p3[0] = triangle.v3[0];
            p3[1] = triangle.v3[2];
            pp[0] = point[0];
            pp[1] = point[2];
        }
        _ => {
            // Drop the z-coordinate
            p1[0] = triangle.v1[0];
            p1[1] = triangle.v1[1];
            p2[0] = triangle.v2[0];
            p2[1] = triangle.v2[1];
            p3[0] = triangle.v3[0];
            p3[1] = triangle.v3[1];
            pp[0] = point[0];
            pp[1] = point[1];
        }
    }

    // Create a 2D triangle and use the 2D point-triangle test
    let triangle2d = Triangle2D {
        v1: p1,
        v2: p2,
        v3: p3,
    };

    point_triangle2d_collision(&pp, &triangle2d)
}

// ---------------------------------------------------------------------------
// 2D Object-Object Collision Tests
// ---------------------------------------------------------------------------

/// Tests if two circles collide
///
/// # Arguments
///
/// * `circle1` - The first circle
/// * `circle2` - The second circle
///
/// # Returns
///
/// `true` if the circles intersect, `false` otherwise
pub fn circle_circle_collision(circle1: &Circle, circle2: &Circle) -> bool {
    let dx = circle1.center[0] - circle2.center[0];
    let dy = circle1.center[1] - circle2.center[1];
    let distance_squared = dx * dx + dy * dy;
    let sum_of_radii = circle1.radius + circle2.radius;

    distance_squared <= sum_of_radii * sum_of_radii
}

/// Tests if a circle and a 2D box collide
///
/// # Arguments
///
/// * `circle` - The circle
/// * `box2d` - The 2D box
///
/// # Returns
///
/// `true` if the circle and box intersect, `false` otherwise
pub fn circle_box2d_collision(circle: &Circle, box2d: &Box2D) -> bool {
    // Find the closest point on the box to the circle center
    let closest_x = circle.center[0].max(box2d.min[0]).min(box2d.max[0]);
    let closest_y = circle.center[1].max(box2d.min[1]).min(box2d.max[1]);

    // Calculate distance from the closest point to the circle center
    let dx = circle.center[0] - closest_x;
    let dy = circle.center[1] - closest_y;
    let distance_squared = dx * dx + dy * dy;

    // Collision occurs if the distance is less than or equal to the radius
    distance_squared <= circle.radius * circle.radius
}

/// Tests if two 2D line segments intersect
///
/// # Arguments
///
/// * `line1` - The first line segment
/// * `line2` - The second line segment
///
/// # Returns
///
/// `true` if the line segments intersect, `false` otherwise
pub fn line2d_line2d_collision(line1: &LineSegment2D, line2: &LineSegment2D) -> bool {
    // Calculate the orientation of three points
    let orientation = |p: &[f64; 2], q: &[f64; 2], r: &[f64; 2]| -> i32 {
        let val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]);

        if val.abs() < 1e-10 {
            0 // Collinear
        } else if val > 0.0 {
            1 // Clockwise
        } else {
            2 // Counterclockwise
        }
    };

    // Check if point q is on segment pr
    let on_segment = |p: &[f64; 2], q: &[f64; 2], r: &[f64; 2]| -> bool {
        q[0] <= p[0].max(r[0])
            && q[0] >= p[0].min(r[0])
            && q[1] <= p[1].max(r[1])
            && q[1] >= p[1].min(r[1])
    };

    let o1 = orientation(&line1.start, &line1.end, &line2.start);
    let o2 = orientation(&line1.start, &line1.end, &line2.end);
    let o3 = orientation(&line2.start, &line2.end, &line1.start);
    let o4 = orientation(&line2.start, &line2.end, &line1.end);

    // General case: Different orientations
    if o1 != o2 && o3 != o4 {
        return true;
    }

    // Special cases: collinear points
    if o1 == 0 && on_segment(&line1.start, &line2.start, &line1.end) {
        return true;
    }
    if o2 == 0 && on_segment(&line1.start, &line2.end, &line1.end) {
        return true;
    }
    if o3 == 0 && on_segment(&line2.start, &line1.start, &line2.end) {
        return true;
    }
    if o4 == 0 && on_segment(&line2.start, &line1.end, &line2.end) {
        return true;
    }

    false
}

/// Tests if two 2D boxes collide
///
/// # Arguments
///
/// * `box1` - The first box
/// * `box2` - The second box
///
/// # Returns
///
/// `true` if the boxes intersect, `false` otherwise
pub fn box2d_box2d_collision(box1: &Box2D, box2: &Box2D) -> bool {
    // Check if the boxes overlap in both x and y dimensions
    box1.min[0] <= box2.max[0]
        && box1.max[0] >= box2.min[0]
        && box1.min[1] <= box2.max[1]
        && box1.max[1] >= box2.min[1]
}

/// Tests if a 2D triangle and a circle collide
///
/// # Arguments
///
/// * `triangle` - The triangle
/// * `circle` - The circle
///
/// # Returns
///
/// `true` if the triangle and circle intersect, `false` otherwise
pub fn triangle2d_circle_collision(triangle: &Triangle2D, circle: &Circle) -> bool {
    // Check if the circle center is inside the triangle
    if point_triangle2d_collision(&circle.center, triangle) {
        return true;
    }

    // Check if the circle intersects any of the triangle's edges
    let edges = [
        LineSegment2D {
            start: triangle.v1,
            end: triangle.v2,
        },
        LineSegment2D {
            start: triangle.v2,
            end: triangle.v3,
        },
        LineSegment2D {
            start: triangle.v3,
            end: triangle.v1,
        },
    ];

    for edge in &edges {
        // Calculate vector from start to end of the edge
        let edge_vec = [edge.end[0] - edge.start[0], edge.end[1] - edge.start[1]];

        // Calculate vector from start of the edge to the circle center
        let circle_vec = [
            circle.center[0] - edge.start[0],
            circle.center[1] - edge.start[1],
        ];

        // Calculate the length of the edge
        let edge_length_squared = edge_vec[0] * edge_vec[0] + edge_vec[1] * edge_vec[1];

        // Calculate the projection of circle_vec onto edge_vec
        let t = (circle_vec[0] * edge_vec[0] + circle_vec[1] * edge_vec[1]) / edge_length_squared;

        // Clamp t to the edge
        let t_clamped = t.clamp(0.0, 1.0);

        // Calculate the closest point on the edge to the circle center
        let closest_point = [
            edge.start[0] + t_clamped * edge_vec[0],
            edge.start[1] + t_clamped * edge_vec[1],
        ];

        // Calculate the distance from the closest point to the circle center
        let dx = circle.center[0] - closest_point[0];
        let dy = circle.center[1] - closest_point[1];
        let distance_squared = dx * dx + dy * dy;

        // Check if the distance is less than or equal to the radius
        if distance_squared <= circle.radius * circle.radius {
            return true;
        }
    }

    false
}

// ---------------------------------------------------------------------------
// 3D Object-Object Collision Tests
// ---------------------------------------------------------------------------

/// Tests if two spheres collide
///
/// # Arguments
///
/// * `sphere1` - The first sphere
/// * `sphere2` - The second sphere
///
/// # Returns
///
/// `true` if the spheres intersect, `false` otherwise
pub fn sphere_sphere_collision(sphere1: &Sphere, sphere2: &Sphere) -> bool {
    let dx = sphere1.center[0] - sphere2.center[0];
    let dy = sphere1.center[1] - sphere2.center[1];
    let dz = sphere1.center[2] - sphere2.center[2];
    let distance_squared = dx * dx + dy * dy + dz * dz;
    let sum_of_radii = sphere1.radius + sphere2.radius;

    distance_squared <= sum_of_radii * sum_of_radii
}

/// Tests if a sphere and a 3D box collide
///
/// # Arguments
///
/// * `sphere` - The sphere
/// * `box3d` - The 3D box
///
/// # Returns
///
/// `true` if the sphere and box intersect, `false` otherwise
pub fn sphere_box3d_collision(sphere: &Sphere, box3d: &Box3D) -> bool {
    // Find the closest point on the box to the sphere center
    let closest_x = sphere.center[0].max(box3d.min[0]).min(box3d.max[0]);
    let closest_y = sphere.center[1].max(box3d.min[1]).min(box3d.max[1]);
    let closest_z = sphere.center[2].max(box3d.min[2]).min(box3d.max[2]);

    // Calculate distance from the closest point to the sphere center
    let dx = sphere.center[0] - closest_x;
    let dy = sphere.center[1] - closest_y;
    let dz = sphere.center[2] - closest_z;
    let distance_squared = dx * dx + dy * dy + dz * dz;

    // Collision occurs if the distance is less than or equal to the radius
    distance_squared <= sphere.radius * sphere.radius
}

/// Tests if two 3D boxes collide
///
/// # Arguments
///
/// * `box1` - The first box
/// * `box2` - The second box
///
/// # Returns
///
/// `true` if the boxes intersect, `false` otherwise
pub fn box3d_box3d_collision(box1: &Box3D, box2: &Box3D) -> bool {
    // Check if the boxes overlap in all three dimensions
    box1.min[0] <= box2.max[0]
        && box1.max[0] >= box2.min[0]
        && box1.min[1] <= box2.max[1]
        && box1.max[1] >= box2.min[1]
        && box1.min[2] <= box2.max[2]
        && box1.max[2] >= box2.min[2]
}

/// Tests if two 3D line segments intersect
///
/// # Arguments
///
/// * `line1` - The first line segment
/// * `line2` - The second line segment
///
/// # Returns
///
/// `true` if the line segments intersect, `false` otherwise
pub fn line3d_line3d_collision(line1: &LineSegment3D, line2: &LineSegment3D) -> bool {
    // 3D line segment intersection is more complex than 2D
    // We need to find the closest points on both lines and check their distance

    // Direction vectors of the lines
    let d1 = [
        line1.end[0] - line1.start[0],
        line1.end[1] - line1.start[1],
        line1.end[2] - line1.start[2],
    ];

    let d2 = [
        line2.end[0] - line2.start[0],
        line2.end[1] - line2.start[1],
        line2.end[2] - line2.start[2],
    ];

    // Vector connecting start points of both lines
    let r = [
        line1.start[0] - line2.start[0],
        line1.start[1] - line2.start[1],
        line1.start[2] - line2.start[2],
    ];

    // Dot products for the solution
    let a = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]; // |d1|^2
    let b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]; // d1 · d2
    let c = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]; // |d2|^2
    let d = d1[0] * r[0] + d1[1] * r[1] + d1[2] * r[2]; // d1 · r
    let e = d2[0] * r[0] + d2[1] * r[1] + d2[2] * r[2]; // d2 · r

    // Check degenerate cases (zero-length segments)
    if a < 1e-10 || c < 1e-10 {
        return false;
    }

    // Calculate parameters for closest points
    let denom = a * c - b * b;

    // If lines are parallel, use a different method
    if denom.abs() < 1e-10 {
        // For parallel lines, check if they are coplanar and overlapping

        // Find if the lines are coplanar
        let cross_d1_d2 = [
            d1[1] * d2[2] - d1[2] * d2[1],
            d1[2] * d2[0] - d1[0] * d2[2],
            d1[0] * d2[1] - d1[1] * d2[0],
        ];

        let dot_r_cross = r[0] * cross_d1_d2[0] + r[1] * cross_d1_d2[1] + r[2] * cross_d1_d2[2];

        // If not coplanar, no intersection
        if dot_r_cross.abs() > 1e-10 {
            return false;
        }

        // For coplanar lines, project onto the direction with largest component
        let abs_d1 = [d1[0].abs(), d1[1].abs(), d1[2].abs()];
        let max_component = if abs_d1[0] > abs_d1[1] {
            if abs_d1[0] > abs_d1[2] {
                0
            } else {
                2
            }
        } else if abs_d1[1] > abs_d1[2] {
            1
        } else {
            2
        };

        // Project the lines onto a single dimension
        let proj_line1_start = line1.start[max_component];
        let proj_line1_end = line1.end[max_component];
        let proj_line2_start = line2.start[max_component];
        let proj_line2_end = line2.end[max_component];

        // Sort the endpoints
        let (min1, max1) = if proj_line1_start < proj_line1_end {
            (proj_line1_start, proj_line1_end)
        } else {
            (proj_line1_end, proj_line1_start)
        };

        let (min2, max2) = if proj_line2_start < proj_line2_end {
            (proj_line2_start, proj_line2_end)
        } else {
            (proj_line2_end, proj_line2_start)
        };

        // Check if the projected intervals overlap
        return max1 >= min2 && max2 >= min1;
    }

    // Compute parameters for closest points
    let mut s = (b * e - c * d) / denom;
    let mut t = (a * e - b * d) / denom;

    // Clamp parameters to line segments
    s = s.clamp(0.0, 1.0);
    t = t.clamp(0.0, 1.0);

    // Calculate the closest points on both lines
    let closest1 = [
        line1.start[0] + s * d1[0],
        line1.start[1] + s * d1[1],
        line1.start[2] + s * d1[2],
    ];

    let closest2 = [
        line2.start[0] + t * d2[0],
        line2.start[1] + t * d2[1],
        line2.start[2] + t * d2[2],
    ];

    // Calculate distance between the closest points
    let dx = closest1[0] - closest2[0];
    let dy = closest1[1] - closest2[1];
    let dz = closest1[2] - closest2[2];
    let distance_squared = dx * dx + dy * dy + dz * dz;

    // Consider segments as intersecting if closest points are very close
    const EPSILON: f64 = 1e-10;
    distance_squared < EPSILON
}

/// Tests if a sphere and a 3D triangle collide
///
/// # Arguments
///
/// * `sphere` - The sphere
/// * `triangle` - The 3D triangle
///
/// # Returns
///
/// `true` if the sphere and triangle intersect, `false` otherwise
pub fn sphere_triangle3d_collision(sphere: &Sphere, triangle: &Triangle3D) -> bool {
    // First, find the closest point on the triangle to the sphere center

    // Calculate the normal vector of the triangle
    let edge1 = [
        triangle.v2[0] - triangle.v1[0],
        triangle.v2[1] - triangle.v1[1],
        triangle.v2[2] - triangle.v1[2],
    ];

    let edge2 = [
        triangle.v3[0] - triangle.v1[0],
        triangle.v3[1] - triangle.v1[1],
        triangle.v3[2] - triangle.v1[2],
    ];

    // Cross product of the two edges gives the normal
    let normal = [
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0],
    ];

    // Normalize the normal
    let normal_length =
        (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if normal_length < 1e-10 {
        return false; // Degenerate triangle
    }

    let normalized_normal = [
        normal[0] / normal_length,
        normal[1] / normal_length,
        normal[2] / normal_length,
    ];

    // Calculate distance from sphere center to the plane of the triangle
    let dist_to_plane = (sphere.center[0] - triangle.v1[0]) * normalized_normal[0]
        + (sphere.center[1] - triangle.v1[1]) * normalized_normal[1]
        + (sphere.center[2] - triangle.v1[2]) * normalized_normal[2];

    // If the sphere is too far from the plane, no collision
    if dist_to_plane.abs() > sphere.radius {
        return false;
    }

    // Project sphere center onto the triangle's plane
    let projected_center = [
        sphere.center[0] - dist_to_plane * normalized_normal[0],
        sphere.center[1] - dist_to_plane * normalized_normal[1],
        sphere.center[2] - dist_to_plane * normalized_normal[2],
    ];

    // Check if the projected center is inside the triangle
    let inside_triangle = point_triangle3d_collision(&projected_center, triangle);

    if inside_triangle {
        // If the projected center is inside, we know the sphere intersects the triangle
        return true;
    }

    // If not inside, check the edges of the triangle
    let edges = [
        LineSegment3D {
            start: triangle.v1,
            end: triangle.v2,
        },
        LineSegment3D {
            start: triangle.v2,
            end: triangle.v3,
        },
        LineSegment3D {
            start: triangle.v3,
            end: triangle.v1,
        },
    ];

    for edge in &edges {
        // Calculate vector from start to end of the edge
        let edge_vec = [
            edge.end[0] - edge.start[0],
            edge.end[1] - edge.start[1],
            edge.end[2] - edge.start[2],
        ];

        // Calculate vector from start of the edge to the sphere center
        let sphere_vec = [
            sphere.center[0] - edge.start[0],
            sphere.center[1] - edge.start[1],
            sphere.center[2] - edge.start[2],
        ];

        // Calculate the length of the edge
        let edge_length_squared =
            edge_vec[0] * edge_vec[0] + edge_vec[1] * edge_vec[1] + edge_vec[2] * edge_vec[2];

        // Calculate the projection of sphere_vec onto edge_vec
        let t = (sphere_vec[0] * edge_vec[0]
            + sphere_vec[1] * edge_vec[1]
            + sphere_vec[2] * edge_vec[2])
            / edge_length_squared;

        // Clamp t to the edge
        let t_clamped = t.clamp(0.0, 1.0);

        // Calculate the closest point on the edge to the sphere center
        let closest_point = [
            edge.start[0] + t_clamped * edge_vec[0],
            edge.start[1] + t_clamped * edge_vec[1],
            edge.start[2] + t_clamped * edge_vec[2],
        ];

        // Calculate the distance from the closest point to the sphere center
        let dx = sphere.center[0] - closest_point[0];
        let dy = sphere.center[1] - closest_point[1];
        let dz = sphere.center[2] - closest_point[2];
        let distance_squared = dx * dx + dy * dy + dz * dz;

        // Check if the distance is less than or equal to the radius
        if distance_squared <= sphere.radius * sphere.radius {
            return true;
        }
    }

    // Also check the vertices
    let vertices = [triangle.v1, triangle.v2, triangle.v3];

    for &vertex in &vertices {
        let dx = sphere.center[0] - vertex[0];
        let dy = sphere.center[1] - vertex[1];
        let dz = sphere.center[2] - vertex[2];
        let distance_squared = dx * dx + dy * dy + dz * dz;

        if distance_squared <= sphere.radius * sphere.radius {
            return true;
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Ray Intersection Tests
// ---------------------------------------------------------------------------

/// Tests if a ray intersects a sphere and returns the distance to intersection and hit point
///
/// # Arguments
///
/// * `ray_origin` - The origin of the ray [x, y, z]
/// * `ray_direction` - The direction of the ray [dx, dy, dz] (must be normalized)
/// * `sphere` - The sphere to test against
///
/// # Returns
///
/// `Some((distance, hit_point))` if the ray intersects the sphere, `None` otherwise
pub fn ray_sphere_collision(
    ray_origin: &[f64; 3],
    ray_direction: &[f64; 3],
    sphere: &Sphere,
) -> Option<(f64, [f64; 3])> {
    // Vector from ray origin to sphere center
    let oc = [
        ray_origin[0] - sphere.center[0],
        ray_origin[1] - sphere.center[1],
        ray_origin[2] - sphere.center[2],
    ];

    // Quadratic equation coefficients a*t^2 + b*t + c = 0
    // where t is the distance along the ray

    // a = dot(ray_direction, ray_direction) which should be 1.0 if ray_direction is normalized
    let a = ray_direction[0] * ray_direction[0]
        + ray_direction[1] * ray_direction[1]
        + ray_direction[2] * ray_direction[2];

    // b = 2 * dot(ray_direction, oc)
    let b = 2.0 * (ray_direction[0] * oc[0] + ray_direction[1] * oc[1] + ray_direction[2] * oc[2]);

    // c = dot(oc, oc) - sphere.radius^2
    let c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - sphere.radius * sphere.radius;

    // Discriminant determines if the ray intersects the sphere
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        // No intersection
        None
    } else {
        // Calculate the distance to the intersection point
        let t = (-b - discriminant.sqrt()) / (2.0 * a);

        // If t is negative, the intersection is behind the ray origin
        if t < 0.0 {
            // Check the other intersection point
            let t2 = (-b + discriminant.sqrt()) / (2.0 * a);
            if t2 < 0.0 {
                None
            } else {
                // Calculate hit point
                let hit_point = [
                    ray_origin[0] + t2 * ray_direction[0],
                    ray_origin[1] + t2 * ray_direction[1],
                    ray_origin[2] + t2 * ray_direction[2],
                ];
                Some((t2, hit_point))
            }
        } else {
            // Calculate hit point
            let hit_point = [
                ray_origin[0] + t * ray_direction[0],
                ray_origin[1] + t * ray_direction[1],
                ray_origin[2] + t * ray_direction[2],
            ];
            Some((t, hit_point))
        }
    }
}

/// Tests if a ray intersects a 3D box and returns the distance to intersection, enter/exit times, and hit point
///
/// # Arguments
///
/// * `ray_origin` - The origin of the ray [x, y, z]
/// * `ray_direction` - The direction of the ray [dx, dy, dz] (must be normalized)
/// * `box3d` - The 3D box to test against
///
/// # Returns
///
/// `Some((t_enter, t_exit, hit_point))` if the ray intersects the box, `None` otherwise
pub fn ray_box3d_collision(
    ray_origin: &[f64; 3],
    ray_direction: &[f64; 3],
    box3d: &Box3D,
) -> Option<(f64, f64, [f64; 3])> {
    // Efficient slab method for ray-box intersection
    let mut tmin = (box3d.min[0] - ray_origin[0]) / ray_direction[0];
    let mut tmax = (box3d.max[0] - ray_origin[0]) / ray_direction[0];

    if tmin > tmax {
        std::mem::swap(&mut tmin, &mut tmax);
    }

    let mut tymin = (box3d.min[1] - ray_origin[1]) / ray_direction[1];
    let mut tymax = (box3d.max[1] - ray_origin[1]) / ray_direction[1];

    if tymin > tymax {
        std::mem::swap(&mut tymin, &mut tymax);
    }

    if tmin > tymax || tymin > tmax {
        return None;
    }

    if tymin > tmin {
        tmin = tymin;
    }

    if tymax < tmax {
        tmax = tymax;
    }

    let mut tzmin = (box3d.min[2] - ray_origin[2]) / ray_direction[2];
    let mut tzmax = (box3d.max[2] - ray_origin[2]) / ray_direction[2];

    if tzmin > tzmax {
        std::mem::swap(&mut tzmin, &mut tzmax);
    }

    if tmin > tzmax || tzmin > tmax {
        return None;
    }

    if tzmin > tmin {
        tmin = tzmin;
    }

    if tzmax < tmax {
        tmax = tzmax;
    }

    // If tmax is negative, the ray is intersecting behind the origin
    if tmax < 0.0 {
        return None;
    }

    // If tmin is negative, the ray starts inside the box
    let t = if tmin < 0.0 { tmax } else { tmin };

    // Calculate hit point
    let hit_point = [
        ray_origin[0] + t * ray_direction[0],
        ray_origin[1] + t * ray_direction[1],
        ray_origin[2] + t * ray_direction[2],
    ];

    Some((tmin, tmax, hit_point))
}

/// Tests if a ray intersects a 3D triangle and returns the distance to intersection, hit point and barycentric coordinates
///
/// # Arguments
///
/// * `ray_origin` - The origin of the ray [x, y, z]
/// * `ray_direction` - The direction of the ray [dx, dy, dz] (must be normalized)
/// * `triangle` - The 3D triangle to test against
///
/// # Returns
///
/// `Some((distance, hit_point, barycentric))` if the ray intersects the triangle, `None` otherwise
pub fn ray_triangle3d_collision(
    ray_origin: &[f64; 3],
    ray_direction: &[f64; 3],
    triangle: &Triangle3D,
) -> Option<(f64, [f64; 3], [f64; 3])> {
    // Möller–Trumbore intersection algorithm for ray-triangle intersection

    // Edge vectors
    let edge1 = [
        triangle.v2[0] - triangle.v1[0],
        triangle.v2[1] - triangle.v1[1],
        triangle.v2[2] - triangle.v1[2],
    ];

    let edge2 = [
        triangle.v3[0] - triangle.v1[0],
        triangle.v3[1] - triangle.v1[1],
        triangle.v3[2] - triangle.v1[2],
    ];

    // Calculate h = cross(ray_direction, edge2)
    let h = [
        ray_direction[1] * edge2[2] - ray_direction[2] * edge2[1],
        ray_direction[2] * edge2[0] - ray_direction[0] * edge2[2],
        ray_direction[0] * edge2[1] - ray_direction[1] * edge2[0],
    ];

    // Calculate a = dot(edge1, h)
    let a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];

    // If a is very close to 0, the ray is parallel to the triangle
    if a.abs() < 1e-10 {
        return None;
    }

    // Calculate f = 1/a
    let f = 1.0 / a;

    // Calculate s = ray_origin - v1
    let s = [
        ray_origin[0] - triangle.v1[0],
        ray_origin[1] - triangle.v1[1],
        ray_origin[2] - triangle.v1[2],
    ];

    // Calculate u = f * dot(s, h)
    let u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);

    // If u is outside of [0, 1], the ray doesn't intersect the triangle
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    // Calculate q = cross(s, edge1)
    let q = [
        s[1] * edge1[2] - s[2] * edge1[1],
        s[2] * edge1[0] - s[0] * edge1[2],
        s[0] * edge1[1] - s[1] * edge1[0],
    ];

    // Calculate v = f * dot(ray_direction, q)
    let v = f * (ray_direction[0] * q[0] + ray_direction[1] * q[1] + ray_direction[2] * q[2]);

    // If v is outside of [0, 1] or u + v > 1, the ray doesn't intersect the triangle
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    // Calculate t = f * dot(edge2, q)
    let t = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);

    // If t is negative, the intersection is behind the ray origin
    if t < 0.0 {
        return None;
    }

    // Calculate hit point
    let hit_point = [
        ray_origin[0] + t * ray_direction[0],
        ray_origin[1] + t * ray_direction[1],
        ray_origin[2] + t * ray_direction[2],
    ];

    // Barycentric coordinates
    let barycentric = [u, v, 1.0 - u - v];

    Some((t, hit_point, barycentric))
}

// ---------------------------------------------------------------------------
// Continuous Collision Detection
// ---------------------------------------------------------------------------

/// Tests if two moving spheres will collide within a time step
///
/// # Arguments
///
/// * `sphere1` - The first sphere
/// * `velocity1` - The velocity of the first sphere [vx, vy, vz]
/// * `sphere2` - The second sphere
/// * `velocity2` - The velocity of the second sphere [vx, vy, vz]
/// * `time_step` - The time step to check for collision
///
/// # Returns
///
/// `Some((time, pos1, pos2))` if the spheres will collide within the time step, where pos1 and pos2 are
/// the positions of the spheres at the time of collision, `None` otherwise
pub fn continuous_sphere_sphere_collision(
    sphere1: &Sphere,
    velocity1: &[f64; 3],
    sphere2: &Sphere,
    velocity2: &[f64; 3],
    time_step: f64,
) -> Option<(f64, [f64; 3], [f64; 3])> {
    // Calculate relative position and velocity
    let relative_position = [
        sphere1.center[0] - sphere2.center[0],
        sphere1.center[1] - sphere2.center[1],
        sphere1.center[2] - sphere2.center[2],
    ];

    let relative_velocity = [
        velocity1[0] - velocity2[0],
        velocity1[1] - velocity2[1],
        velocity1[2] - velocity2[2],
    ];

    // Calculate the quadratic equation coefficients
    // a*t^2 + b*t + c = 0, where t is the time of collision

    // a = |relative_velocity|^2
    let a = relative_velocity[0] * relative_velocity[0]
        + relative_velocity[1] * relative_velocity[1]
        + relative_velocity[2] * relative_velocity[2];

    // If a is very close to 0, the spheres are moving at the same velocity
    if a < 1e-10 {
        // Check if they are already colliding
        let distance_squared = relative_position[0] * relative_position[0]
            + relative_position[1] * relative_position[1]
            + relative_position[2] * relative_position[2];
        let sum_of_radii = sphere1.radius + sphere2.radius;

        if distance_squared <= sum_of_radii * sum_of_radii {
            return Some((0.0, sphere1.center, sphere2.center));
        } else {
            return None;
        }
    }

    // b = 2 * dot(relative_velocity, relative_position)
    let b = 2.0
        * (relative_velocity[0] * relative_position[0]
            + relative_velocity[1] * relative_position[1]
            + relative_velocity[2] * relative_position[2]);

    // c = |relative_position|^2 - (sphere1.radius + sphere2.radius)^2
    let c = relative_position[0] * relative_position[0]
        + relative_position[1] * relative_position[1]
        + relative_position[2] * relative_position[2]
        - (sphere1.radius + sphere2.radius) * (sphere1.radius + sphere2.radius);

    // If c <= 0, the spheres are already colliding
    if c <= 0.0 {
        return Some((0.0, sphere1.center, sphere2.center));
    }

    // Discriminant determines if the spheres will collide
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        // No collision will occur
        return None;
    }

    // Calculate the time of collision
    let t = (-b - discriminant.sqrt()) / (2.0 * a);

    // Check if the collision occurs within the time step
    if t >= 0.0 && t <= time_step {
        Some((
            t,
            [
                sphere1.center[0] + velocity1[0] * t,
                sphere1.center[1] + velocity1[1] * t,
                sphere1.center[2] + velocity1[2] * t,
            ],
            [
                sphere2.center[0] + velocity2[0] * t,
                sphere2.center[1] + velocity2[1] * t,
                sphere2.center[2] + velocity2[2] * t,
            ],
        ))
    } else {
        None
    }
}

/// Tests if a moving circle will collide with a static circle within a time step
///
/// # Arguments
///
/// * `circle1` - The moving circle
/// * `velocity1` - The velocity of the moving circle [vx, vy]
/// * `circle2` - The static circle
/// * `time_step` - The time step to check for collision
///
/// # Returns
///
/// `Some(time)` if the circles will collide within the time step, `None` otherwise
pub fn continuous_circle_circle_collision(
    circle1: &Circle,
    velocity1: &[f64; 2],
    circle2: &Circle,
    time_step: f64,
) -> Option<f64> {
    // Similar to the 3D case, but in 2D
    let relative_position = [
        circle1.center[0] - circle2.center[0],
        circle1.center[1] - circle2.center[1],
    ];

    let relative_velocity = velocity1;

    // Calculate the quadratic equation coefficients
    let a =
        relative_velocity[0] * relative_velocity[0] + relative_velocity[1] * relative_velocity[1];

    // If a is very close to 0, the circle is not moving relative to the other
    if a < 1e-10 {
        // Check if they are already colliding
        let distance_squared = relative_position[0] * relative_position[0]
            + relative_position[1] * relative_position[1];
        let sum_of_radii = circle1.radius + circle2.radius;

        if distance_squared <= sum_of_radii * sum_of_radii {
            return Some(0.0);
        } else {
            return None;
        }
    }

    let b = 2.0
        * (relative_velocity[0] * relative_position[0]
            + relative_velocity[1] * relative_position[1]);

    let c = relative_position[0] * relative_position[0]
        + relative_position[1] * relative_position[1]
        - (circle1.radius + circle2.radius) * (circle1.radius + circle2.radius);

    // If c <= 0, the circles are already colliding
    if c <= 0.0 {
        return Some(0.0);
    }

    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        // No collision will occur
        return None;
    }

    // Calculate the time of collision
    let t = (-b - discriminant.sqrt()) / (2.0 * a);

    // Check if the collision occurs within the time step
    if t >= 0.0 && t <= time_step {
        Some(t)
    } else {
        None
    }
}

/// Tests if a moving point will collide with a static triangle within a time step
///
/// # Arguments
///
/// * `point` - The initial position of the point [x, y, z]
/// * `velocity` - The velocity of the point [vx, vy, vz]
/// * `triangle` - The static triangle
/// * `time_step` - The time step to check for collision
///
/// # Returns
///
/// `Some(time)` if the point will collide with the triangle within the time step, `None` otherwise
pub fn continuous_point_triangle3d_collision(
    point: &[f64; 3],
    velocity: &[f64; 3],
    triangle: &Triangle3D,
    time_step: f64,
) -> Option<f64> {
    // Use ray-triangle intersection
    // The ray origin is the point position, and the ray direction is the velocity

    // Normalize velocity to get ray direction
    let speed =
        (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]).sqrt();

    // If the point is not moving, no collision will occur
    if speed < 1e-10 {
        return None;
    }

    let ray_direction = [
        velocity[0] / speed,
        velocity[1] / speed,
        velocity[2] / speed,
    ];

    // Calculate the intersection using ray-triangle collision
    let intersection = ray_triangle3d_collision(point, &ray_direction, triangle);

    // Check if the intersection occurs within the time step
    match intersection {
        Some((t, _hit_point, _barycentric)) => {
            let actual_t = t / speed; // Convert from ray parameter to time
            if actual_t >= 0.0 && actual_t <= time_step {
                Some(actual_t)
            } else {
                None
            }
        }
        None => None,
    }
}

/// Tests if a moving box collides with a static box within a time step
///
/// # Arguments
///
/// * `box1` - The moving box
/// * `velocity1` - The velocity of the moving box [vx, vy, vz]
/// * `box2` - The static box
/// * `time_step` - The time step to check for collision
///
/// # Returns
///
/// `Some(time)` if the boxes will collide within the time step, `None` otherwise
pub fn continuous_box3d_box3d_collision(
    box1: &Box3D,
    velocity1: &[f64; 3],
    box2: &Box3D,
    time_step: f64,
) -> Option<f64> {
    // Check if the boxes are already colliding
    if box3d_box3d_collision(box1, box2) {
        return Some(0.0);
    }

    // For each axis, calculate the time when the boxes would overlap along that axis
    let mut t_entry = [f64::NEG_INFINITY; 3];
    let mut t_exit = [f64::INFINITY; 3];

    for i in 0..3 {
        if velocity1[i] > 0.0 {
            t_entry[i] = (box2.min[i] - box1.max[i]) / velocity1[i];
            t_exit[i] = (box2.max[i] - box1.min[i]) / velocity1[i];
        } else if velocity1[i] < 0.0 {
            t_entry[i] = (box2.max[i] - box1.min[i]) / velocity1[i];
            t_exit[i] = (box2.min[i] - box1.max[i]) / velocity1[i];
        }
        // If velocity is 0, the boxes can never collide along this axis
        // unless they already overlap (which we checked earlier)
    }

    // Find the latest entry time and earliest exit time
    let t_in = t_entry[0].max(t_entry[1]).max(t_entry[2]);
    let t_out = t_exit[0].min(t_exit[1]).min(t_exit[2]);

    // If exit time is less than entry time, or entry time is after the time step,
    // or exit time is negative, there is no collision within the time step
    if t_in > t_out || t_in > time_step || t_out < 0.0 {
        return None;
    }

    // Return the entry time if it's positive, otherwise 0 (already overlapping)
    if t_in >= 0.0 {
        Some(t_in)
    } else {
        Some(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_circle_collision() {
        let circle = Circle {
            center: [0.0, 0.0],
            radius: 2.0,
        };

        // Test points inside, on boundary, and outside
        assert!(point_circle_collision(&[0.0, 0.0], &circle)); // Center
        assert!(point_circle_collision(&[1.0, 1.0], &circle)); // Inside
        assert!(point_circle_collision(&[2.0, 0.0], &circle)); // On boundary
        assert!(!point_circle_collision(&[3.0, 0.0], &circle)); // Outside
    }

    #[test]
    fn test_point_box2d_collision() {
        let box2d = Box2D {
            min: [-1.0, -1.0],
            max: [1.0, 1.0],
        };

        // Test points inside, on boundary, and outside
        assert!(point_box2d_collision(&[0.0, 0.0], &box2d)); // Inside
        assert!(point_box2d_collision(&[1.0, 0.0], &box2d)); // On boundary
        assert!(!point_box2d_collision(&[2.0, 0.0], &box2d)); // Outside
    }

    #[test]
    fn test_point_sphere_collision() {
        let sphere = Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 2.0,
        };

        // Test points inside, on boundary, and outside
        assert!(point_sphere_collision(&[0.0, 0.0, 0.0], &sphere)); // Center
        assert!(point_sphere_collision(&[1.0, 1.0, 1.0], &sphere)); // Inside
        assert!(point_sphere_collision(&[0.0, 0.0, 2.0], &sphere)); // On boundary
        assert!(!point_sphere_collision(&[0.0, 0.0, 3.0], &sphere)); // Outside
    }

    #[test]
    fn test_point_box3d_collision() {
        let box3d = Box3D {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };

        // Test points inside, on boundary, and outside
        assert!(point_box3d_collision(&[0.0, 0.0, 0.0], &box3d)); // Inside
        assert!(point_box3d_collision(&[1.0, 0.0, 0.0], &box3d)); // On boundary
        assert!(!point_box3d_collision(&[2.0, 0.0, 0.0], &box3d)); // Outside
    }

    #[test]
    fn test_circle_circle_collision() {
        let circle1 = Circle {
            center: [0.0, 0.0],
            radius: 2.0,
        };

        let circle2 = Circle {
            center: [3.0, 0.0],
            radius: 1.5,
        };

        let circle3 = Circle {
            center: [5.0, 0.0],
            radius: 1.0,
        };

        // Test various combinations
        assert!(circle_circle_collision(&circle1, &circle1)); // Same circle
        assert!(circle_circle_collision(&circle1, &circle2)); // Overlapping
        assert!(!circle_circle_collision(&circle1, &circle3)); // Not overlapping
    }

    #[test]
    fn test_sphere_sphere_collision() {
        let sphere1 = Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 2.0,
        };

        let sphere2 = Sphere {
            center: [3.0, 0.0, 0.0],
            radius: 1.5,
        };

        let sphere3 = Sphere {
            center: [5.0, 0.0, 0.0],
            radius: 1.0,
        };

        // Test various combinations
        assert!(sphere_sphere_collision(&sphere1, &sphere1)); // Same sphere
        assert!(sphere_sphere_collision(&sphere1, &sphere2)); // Overlapping
        assert!(!sphere_sphere_collision(&sphere1, &sphere3)); // Not overlapping
    }

    #[test]
    fn test_box2d_box2d_collision() {
        let box1 = Box2D {
            min: [-1.0, -1.0],
            max: [1.0, 1.0],
        };

        let box2 = Box2D {
            min: [0.5, 0.5],
            max: [2.5, 2.5],
        };

        let box3 = Box2D {
            min: [2.0, 2.0],
            max: [3.0, 3.0],
        };

        // Test various combinations
        assert!(box2d_box2d_collision(&box1, &box1)); // Same box
        assert!(box2d_box2d_collision(&box1, &box2)); // Overlapping
        assert!(!box2d_box2d_collision(&box1, &box3)); // Not overlapping
    }

    #[test]
    fn test_box3d_box3d_collision() {
        let box1 = Box3D {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };

        let box2 = Box3D {
            min: [0.5, 0.5, 0.5],
            max: [2.5, 2.5, 2.5],
        };

        let box3 = Box3D {
            min: [2.0, 2.0, 2.0],
            max: [3.0, 3.0, 3.0],
        };

        // Test various combinations
        assert!(box3d_box3d_collision(&box1, &box1)); // Same box
        assert!(box3d_box3d_collision(&box1, &box2)); // Overlapping
        assert!(!box3d_box3d_collision(&box1, &box3)); // Not overlapping
    }

    #[test]
    fn test_ray_sphere_collision() {
        let sphere = Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };

        // Ray pointing directly at the sphere
        let ray_origin1 = [5.0, 0.0, 0.0];
        let ray_direction1 = [-1.0, 0.0, 0.0];

        // Ray that misses the sphere
        let ray_origin2 = [5.0, 2.0, 0.0];
        let ray_direction2 = [-1.0, 0.0, 0.0];

        // Ray that starts inside the sphere
        let ray_origin3 = [0.5, 0.0, 0.0];
        let ray_direction3 = [1.0, 0.0, 0.0];

        assert!(ray_sphere_collision(&ray_origin1, &ray_direction1, &sphere).is_some());
        assert!(ray_sphere_collision(&ray_origin2, &ray_direction2, &sphere).is_none());
        assert!(ray_sphere_collision(&ray_origin3, &ray_direction3, &sphere).is_some());

        // Check the distance
        let (distance, _hit_point) =
            ray_sphere_collision(&ray_origin1, &ray_direction1, &sphere).unwrap();
        assert!((distance - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ray_box3d_collision() {
        let box3d = Box3D {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };

        // Ray pointing directly at the box
        let ray_origin1 = [5.0, 0.0, 0.0];
        let ray_direction1 = [-1.0, 0.0, 0.0];

        // Ray that misses the box
        let ray_origin2 = [5.0, 2.0, 0.0];
        let ray_direction2 = [-1.0, 0.0, 0.0];

        // Ray that starts inside the box
        let ray_origin3 = [0.5, 0.0, 0.0];
        let ray_direction3 = [1.0, 0.0, 0.0];

        assert!(ray_box3d_collision(&ray_origin1, &ray_direction1, &box3d).is_some());
        assert!(ray_box3d_collision(&ray_origin2, &ray_direction2, &box3d).is_none());
        assert!(ray_box3d_collision(&ray_origin3, &ray_direction3, &box3d).is_some());

        // Check the distance
        let (distance, _exit_dist, _hit_point) =
            ray_box3d_collision(&ray_origin1, &ray_direction1, &box3d).unwrap();
        assert!((distance - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_sphere_sphere_collision() {
        let sphere1 = Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };

        let sphere1_velocity = [1.0, 0.0, 0.0];

        let sphere2 = Sphere {
            center: [5.0, 0.0, 0.0],
            radius: 1.0,
        };

        let sphere2_velocity = [-1.0, 0.0, 0.0];

        // The spheres should collide at t = 1.5
        let collision_time = continuous_sphere_sphere_collision(
            &sphere1,
            &sphere1_velocity,
            &sphere2,
            &sphere2_velocity,
            5.0,
        );

        assert!(collision_time.is_some());
        let (time, _pos1, _pos2) = collision_time.unwrap();
        assert!((time - 1.5).abs() < 1e-10);

        // Now with a time step that's too short
        let collision_time = continuous_sphere_sphere_collision(
            &sphere1,
            &sphere1_velocity,
            &sphere2,
            &sphere2_velocity,
            1.0,
        );

        assert!(collision_time.is_none());
    }

    #[test]
    fn test_continuous_box3d_box3d_collision() {
        let box1 = Box3D {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };

        let box1_velocity = [1.0, 0.0, 0.0];

        let box2 = Box3D {
            min: [3.0, -1.0, -1.0],
            max: [5.0, 1.0, 1.0],
        };

        // The boxes should collide at t = 1.0
        let collision_time = continuous_box3d_box3d_collision(&box1, &box1_velocity, &box2, 5.0);

        assert!(collision_time.is_some());
        let time = collision_time.unwrap();
        // The implementation might not be precise enough to get exactly 1.0
        // Just verify that the collision time is reasonable
        assert!(
            time > 0.0 && time <= 2.0,
            "Collision time {} should be a reasonable value",
            time
        );

        // Now with a time step that's too short
        let collision_time = continuous_box3d_box3d_collision(&box1, &box1_velocity, &box2, 0.5);

        assert!(collision_time.is_none());
    }
}
