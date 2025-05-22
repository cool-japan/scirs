//! Geometric primitives used for collision detection
//!
//! This module provides the geometric primitives that are used for collision
//! detection algorithms, including circles, triangles, boxes in 2D and 3D.

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
        super::narrowphase::point_circle_collision(point, self)
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
        super::narrowphase::point_triangle2d_collision(point, self)
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
        super::narrowphase::point_box2d_collision(point, self)
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
        super::narrowphase::point_sphere_collision(point, self)
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
        super::narrowphase::point_box3d_collision(point, self)
    }
}
