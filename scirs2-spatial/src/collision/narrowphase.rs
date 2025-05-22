//! Narrow-phase collision detection algorithms
//!
//! This module provides detailed collision detection tests between different geometric objects.
//! These tests determine precise collision information after broad-phase culling has been applied.

use super::shapes::{
    Box2D, Box3D, Circle, LineSegment2D, LineSegment3D, Sphere, Triangle2D, Triangle3D,
};

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
    // M�llerTrumbore intersection algorithm for ray-triangle intersection

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
