//! Continuous collision detection algorithms
//!
//! This module provides algorithms for detecting collisions between moving objects
//! over a time interval, rather than just testing for overlap at a specific moment.

use super::shapes::{Box3D, Circle, Sphere, Triangle3D};

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
    let intersection =
        super::narrowphase::ray_triangle3d_collision(point, &ray_direction, triangle);

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
    if super::narrowphase::box3d_box3d_collision(box1, box2) {
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

/// Tests if a moving triangle collides with a static sphere within a time step
///
/// # Arguments
///
/// * `triangle` - The triangle
/// * `velocity` - The velocity of the triangle [vx, vy, vz]
/// * `sphere` - The static sphere
/// * `time_step` - The time step to check for collision
///
/// # Returns
///
/// `Some(time)` if the triangle will collide with the sphere within the time step, `None` otherwise
pub fn continuous_triangle3d_sphere_collision(
    triangle: &Triangle3D,
    velocity: &[f64; 3],
    sphere: &Sphere,
    time_step: f64,
) -> Option<f64> {
    // Use the inverse problem: a static triangle and a moving sphere
    // with the negative velocity of the triangle
    let inverse_velocity = [-velocity[0], -velocity[1], -velocity[2]];

    // Adjust the starting position of the sphere to compensate for the triangle's movement
    let sphere_adjusted = Sphere {
        center: sphere.center,
        radius: sphere.radius,
    };

    // Check for initial collision
    if super::narrowphase::sphere_triangle3d_collision(&sphere_adjusted, triangle) {
        return Some(0.0);
    }

    // For simplicity, we'll break this down into three separate collision tests,
    // one for each vertex of the triangle and one for each edge

    // First, check for collisions between the sphere and each vertex of the triangle
    let vertices = [triangle.v1, triangle.v2, triangle.v3];
    let mut earliest_time = f64::INFINITY;
    let mut collision_found = false;

    for vertex in &vertices {
        // Create a point representing the vertex
        let point = *vertex;

        // Calculate distance from the point to the sphere center
        let dx = point[0] - sphere.center[0];
        let dy = point[1] - sphere.center[1];
        let dz = point[2] - sphere.center[2];
        let dist_squared = dx * dx + dy * dy + dz * dz;

        // If the point is already inside the sphere, collision at t=0
        if dist_squared <= sphere.radius * sphere.radius {
            return Some(0.0);
        }

        // Calculate quadratic equation for time of collision
        let a = inverse_velocity[0] * inverse_velocity[0]
            + inverse_velocity[1] * inverse_velocity[1]
            + inverse_velocity[2] * inverse_velocity[2];

        if a < 1e-10 {
            continue; // Velocity is too small to consider
        }

        let b =
            2.0 * (dx * inverse_velocity[0] + dy * inverse_velocity[1] + dz * inverse_velocity[2]);

        let c = dist_squared - sphere.radius * sphere.radius;

        let discriminant = b * b - 4.0 * a * c;

        if discriminant >= 0.0 {
            let t = (-b - discriminant.sqrt()) / (2.0 * a);

            if t >= 0.0 && t <= time_step && t < earliest_time {
                earliest_time = t;
                collision_found = true;
            }
        }
    }

    // Check for collisions with the edges of the triangle
    let edges = [
        (triangle.v1, triangle.v2),
        (triangle.v2, triangle.v3),
        (triangle.v3, triangle.v1),
    ];

    for &(start, end) in &edges {
        // Create a vector representing the edge
        let edge_vector = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];

        let edge_length_squared = edge_vector[0] * edge_vector[0]
            + edge_vector[1] * edge_vector[1]
            + edge_vector[2] * edge_vector[2];

        // Calculate the closest point on the line to the sphere center
        let t0 = {
            let v = [
                sphere.center[0] - start[0],
                sphere.center[1] - start[1],
                sphere.center[2] - start[2],
            ];

            let dot = v[0] * edge_vector[0] + v[1] * edge_vector[1] + v[2] * edge_vector[2];

            dot / edge_length_squared
        };

        // Clamp t0 to [0, 1] to get the closest point on the segment
        let t0_clamped = t0.clamp(0.0, 1.0);

        // Calculate the closest point on the edge
        let closest_point = [
            start[0] + t0_clamped * edge_vector[0],
            start[1] + t0_clamped * edge_vector[1],
            start[2] + t0_clamped * edge_vector[2],
        ];

        // Calculate distance from the closest point to the sphere center
        let dx = closest_point[0] - sphere.center[0];
        let dy = closest_point[1] - sphere.center[1];
        let dz = closest_point[2] - sphere.center[2];
        let dist_squared = dx * dx + dy * dy + dz * dz;

        // If the closest point is already inside the sphere, collision at t=0
        if dist_squared <= sphere.radius * sphere.radius {
            return Some(0.0);
        }

        // Now check for future collisions
        // We need to solve a more complex problem for edges, but for simplicity,
        // we'll treat it approximately using the closest point

        let a = inverse_velocity[0] * inverse_velocity[0]
            + inverse_velocity[1] * inverse_velocity[1]
            + inverse_velocity[2] * inverse_velocity[2];

        if a < 1e-10 {
            continue; // Velocity is too small to consider
        }

        let b =
            2.0 * (dx * inverse_velocity[0] + dy * inverse_velocity[1] + dz * inverse_velocity[2]);

        let c = dist_squared - sphere.radius * sphere.radius;

        let discriminant = b * b - 4.0 * a * c;

        if discriminant >= 0.0 {
            let t = (-b - discriminant.sqrt()) / (2.0 * a);

            if t >= 0.0 && t <= time_step && t < earliest_time {
                earliest_time = t;
                collision_found = true;
            }
        }
    }

    // Now check for collision with the triangle face
    // This is a complex problem, but we'll use the approach of finding when
    // the sphere's center crosses the triangle's plane minus the radius

    // Calculate the triangle's normal
    let normal = {
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

        // Cross product
        let cross = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        ];

        // Normalize
        let length = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        if length < 1e-10 {
            [0.0, 0.0, 0.0] // Degenerate triangle
        } else {
            [cross[0] / length, cross[1] / length, cross[2] / length]
        }
    };

    if normal[0] == 0.0 && normal[1] == 0.0 && normal[2] == 0.0 {
        // Degenerate triangle
        if collision_found {
            return Some(earliest_time);
        }
        return None;
    }

    // Calculate distance from sphere center to triangle plane
    let plane_dist = (sphere.center[0] - triangle.v1[0]) * normal[0]
        + (sphere.center[1] - triangle.v1[1]) * normal[1]
        + (sphere.center[2] - triangle.v1[2]) * normal[2];

    // Calculate speed of approach to the plane
    let approach_speed = -(inverse_velocity[0] * normal[0]
        + inverse_velocity[1] * normal[1]
        + inverse_velocity[2] * normal[2]);

    if approach_speed.abs() > 1e-10 {
        // Time until the sphere center is at a distance of radius from the plane
        let t_plane = (plane_dist.abs() - sphere.radius) / approach_speed;

        if t_plane >= 0.0 && t_plane <= time_step && t_plane < earliest_time {
            // At this time, the sphere is at a distance of radius from the plane
            // Now check if the projection of the sphere center onto the plane is inside the triangle

            let projected_center = [
                sphere.center[0] + inverse_velocity[0] * t_plane
                    - normal[0] * sphere.radius * plane_dist.signum(),
                sphere.center[1] + inverse_velocity[1] * t_plane
                    - normal[1] * sphere.radius * plane_dist.signum(),
                sphere.center[2] + inverse_velocity[2] * t_plane
                    - normal[2] * sphere.radius * plane_dist.signum(),
            ];

            if super::narrowphase::point_triangle3d_collision(&projected_center, triangle) {
                earliest_time = t_plane;
                collision_found = true;
            }
        }
    }

    if collision_found {
        Some(earliest_time)
    } else {
        None
    }
}

/// Tests if a moving sphere collides with a static triangle within a time step
///
/// # Arguments
///
/// * `sphere` - The sphere
/// * `velocity` - The velocity of the sphere [vx, vy, vz]
/// * `triangle` - The static triangle
/// * `time_step` - The time step to check for collision
///
/// # Returns
///
/// `Some(time)` if the sphere will collide with the triangle within the time step, `None` otherwise
pub fn continuous_sphere_triangle3d_collision(
    sphere: &Sphere,
    velocity: &[f64; 3],
    triangle: &Triangle3D,
    time_step: f64,
) -> Option<f64> {
    // This is the same problem as a moving triangle with the negative velocity
    // and a static sphere
    let inverse_velocity = [-velocity[0], -velocity[1], -velocity[2]];
    continuous_triangle3d_sphere_collision(triangle, &inverse_velocity, sphere, time_step)
}
