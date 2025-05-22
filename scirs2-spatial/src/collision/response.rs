//! Collision response algorithms
//!
//! This module provides algorithms for calculating appropriate reactions
//! after a collision has been detected, such as computing impulses and
//! resolving penetration.

use super::shapes::{Box3D, Sphere, Triangle3D};

/// Information about a collision
#[derive(Debug, Clone)]
pub struct CollisionInfo {
    /// Point of collision in world space
    pub contact_point: [f64; 3],
    /// Normal vector at the point of collision (pointing from first object to second)
    pub normal: [f64; 3],
    /// Penetration depth (positive indicates overlap)
    pub penetration: f64,
}

/// Calculates the collision response impulse for two spheres
///
/// # Arguments
///
/// * `sphere1_pos` - Position of the first sphere
/// * `sphere1_vel` - Velocity of the first sphere
/// * `sphere1_mass` - Mass of the first sphere
/// * `sphere2_pos` - Position of the second sphere
/// * `sphere2_vel` - Velocity of the second sphere
/// * `sphere2_mass` - Mass of the second sphere
/// * `restitution` - Coefficient of restitution (0 = inelastic, 1 = elastic)
///
/// # Returns
///
/// A tuple containing the impulse vectors for the first and second spheres
pub fn sphere_sphere_impulse(
    sphere1_pos: &[f64; 3],
    sphere1_vel: &[f64; 3],
    sphere1_mass: f64,
    sphere2_pos: &[f64; 3],
    sphere2_vel: &[f64; 3],
    sphere2_mass: f64,
    restitution: f64,
) -> ([f64; 3], [f64; 3]) {
    // Calculate collision normal
    let normal = [
        sphere2_pos[0] - sphere1_pos[0],
        sphere2_pos[1] - sphere1_pos[1],
        sphere2_pos[2] - sphere1_pos[2],
    ];

    // Normalize the normal
    let normal_length =
        (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if normal_length < 1e-10 {
        // Spheres are at the same position, use an arbitrary normal
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    }

    let unit_normal = [
        normal[0] / normal_length,
        normal[1] / normal_length,
        normal[2] / normal_length,
    ];

    // Calculate relative velocity
    let relative_velocity = [
        sphere2_vel[0] - sphere1_vel[0],
        sphere2_vel[1] - sphere1_vel[1],
        sphere2_vel[2] - sphere1_vel[2],
    ];

    // Calculate velocity along the normal
    let velocity_along_normal = relative_velocity[0] * unit_normal[0]
        + relative_velocity[1] * unit_normal[1]
        + relative_velocity[2] * unit_normal[2];

    // If the spheres are moving away from each other, no impulse is needed
    if velocity_along_normal > 0.0 {
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    }

    // Calculate the impulse scalar
    let inverse_mass1 = if sphere1_mass == 0.0 {
        0.0
    } else {
        1.0 / sphere1_mass
    };
    let inverse_mass2 = if sphere2_mass == 0.0 {
        0.0
    } else {
        1.0 / sphere2_mass
    };

    let impulse_scalar =
        -(1.0 + restitution) * velocity_along_normal / (inverse_mass1 + inverse_mass2);

    // Calculate the impulse vectors
    let impulse1 = [
        -impulse_scalar * unit_normal[0] * inverse_mass1,
        -impulse_scalar * unit_normal[1] * inverse_mass1,
        -impulse_scalar * unit_normal[2] * inverse_mass1,
    ];

    let impulse2 = [
        impulse_scalar * unit_normal[0] * inverse_mass2,
        impulse_scalar * unit_normal[1] * inverse_mass2,
        impulse_scalar * unit_normal[2] * inverse_mass2,
    ];

    (impulse1, impulse2)
}

/// Resolves the penetration between two spheres
///
/// # Arguments
///
/// * `sphere1` - The first sphere
/// * `sphere1_mass` - Mass of the first sphere
/// * `sphere2` - The second sphere
/// * `sphere2_mass` - Mass of the second sphere
///
/// # Returns
///
/// A tuple containing the position adjustments for the first and second spheres
pub fn resolve_sphere_sphere_penetration(
    sphere1: &Sphere,
    sphere1_mass: f64,
    sphere2: &Sphere,
    sphere2_mass: f64,
) -> ([f64; 3], [f64; 3]) {
    // Calculate distance between spheres
    let normal = [
        sphere2.center[0] - sphere1.center[0],
        sphere2.center[1] - sphere1.center[1],
        sphere2.center[2] - sphere1.center[2],
    ];

    let distance = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();

    // If no penetration, no adjustment needed
    if distance >= sphere1.radius + sphere2.radius {
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    }

    // Calculate penetration depth
    let penetration = sphere1.radius + sphere2.radius - distance;

    // Normalize the normal
    let unit_normal = if distance < 1e-10 {
        // Spheres are at the same position, use an arbitrary normal
        [1.0, 0.0, 0.0]
    } else {
        [
            normal[0] / distance,
            normal[1] / distance,
            normal[2] / distance,
        ]
    };

    // Calculate mass factors
    let total_mass = sphere1_mass + sphere2_mass;
    let mass_ratio1 = if total_mass == 0.0 {
        0.5
    } else {
        sphere2_mass / total_mass
    };
    let mass_ratio2 = if total_mass == 0.0 {
        0.5
    } else {
        sphere1_mass / total_mass
    };

    // Adjust positions based on mass ratio
    let adjustment1 = [
        -unit_normal[0] * penetration * mass_ratio1,
        -unit_normal[1] * penetration * mass_ratio1,
        -unit_normal[2] * penetration * mass_ratio1,
    ];

    let adjustment2 = [
        unit_normal[0] * penetration * mass_ratio2,
        unit_normal[1] * penetration * mass_ratio2,
        unit_normal[2] * penetration * mass_ratio2,
    ];

    (adjustment1, adjustment2)
}

/// Calculates the collision response impulse for a sphere and a box
///
/// # Arguments
///
/// * `sphere_pos` - Position of the sphere
/// * `sphere_vel` - Velocity of the sphere
/// * `sphere_mass` - Mass of the sphere
/// * `box_pos` - Position of the box (center)
/// * `box_vel` - Velocity of the box
/// * `box_mass` - Mass of the box
/// * `box_dims` - Dimensions of the box [width, height, depth]
/// * `collision_info` - Information about the collision
/// * `restitution` - Coefficient of restitution (0 = inelastic, 1 = elastic)
///
/// # Returns
///
/// A tuple containing the impulse vectors for the sphere and the box
#[allow(clippy::too_many_arguments)]
pub fn sphere_box_impulse(
    _sphere_pos: &[f64; 3],
    sphere_vel: &[f64; 3],
    sphere_mass: f64,
    _box_pos: &[f64; 3],
    box_vel: &[f64; 3],
    box_mass: f64,
    _box_dims: &[f64; 3],
    collision_info: &CollisionInfo,
    restitution: f64,
) -> ([f64; 3], [f64; 3]) {
    // Calculate relative velocity
    let relative_velocity = [
        sphere_vel[0] - box_vel[0],
        sphere_vel[1] - box_vel[1],
        sphere_vel[2] - box_vel[2],
    ];

    // Calculate velocity along the normal
    let velocity_along_normal = relative_velocity[0] * collision_info.normal[0]
        + relative_velocity[1] * collision_info.normal[1]
        + relative_velocity[2] * collision_info.normal[2];

    // If the objects are moving away from each other, no impulse is needed
    if velocity_along_normal > 0.0 {
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    }

    // Calculate the impulse scalar
    let inverse_mass1 = if sphere_mass == 0.0 {
        0.0
    } else {
        1.0 / sphere_mass
    };
    let inverse_mass2 = if box_mass == 0.0 { 0.0 } else { 1.0 / box_mass };

    let impulse_scalar =
        -(1.0 + restitution) * velocity_along_normal / (inverse_mass1 + inverse_mass2);

    // Calculate the impulse vectors
    let impulse1 = [
        impulse_scalar * collision_info.normal[0] * inverse_mass1,
        impulse_scalar * collision_info.normal[1] * inverse_mass1,
        impulse_scalar * collision_info.normal[2] * inverse_mass1,
    ];

    let impulse2 = [
        -impulse_scalar * collision_info.normal[0] * inverse_mass2,
        -impulse_scalar * collision_info.normal[1] * inverse_mass2,
        -impulse_scalar * collision_info.normal[2] * inverse_mass2,
    ];

    (impulse1, impulse2)
}

/// Finds the collision information for a sphere and a box
///
/// # Arguments
///
/// * `sphere` - The sphere
/// * `box3d` - The box
///
/// # Returns
///
/// Collision information if a collision occurs, None otherwise
pub fn find_sphere_box_collision(sphere: &Sphere, box3d: &Box3D) -> Option<CollisionInfo> {
    // Find the closest point on the box to the sphere center
    let closest_x = sphere.center[0].max(box3d.min[0]).min(box3d.max[0]);
    let closest_y = sphere.center[1].max(box3d.min[1]).min(box3d.max[1]);
    let closest_z = sphere.center[2].max(box3d.min[2]).min(box3d.max[2]);

    let closest_point = [closest_x, closest_y, closest_z];

    // Calculate distance from the closest point to the sphere center
    let dx = sphere.center[0] - closest_point[0];
    let dy = sphere.center[1] - closest_point[1];
    let dz = sphere.center[2] - closest_point[2];
    let distance_squared = dx * dx + dy * dy + dz * dz;

    // Check if collision occurs
    if distance_squared > sphere.radius * sphere.radius {
        return None;
    }

    let distance = distance_squared.sqrt();

    // Normal points from the box to the sphere
    let normal = if distance < 1e-10 {
        // Sphere center is on the box surface, use the face normal
        // Determine which face is closest to the sphere center
        let box_center = [
            (box3d.min[0] + box3d.max[0]) * 0.5,
            (box3d.min[1] + box3d.max[1]) * 0.5,
            (box3d.min[2] + box3d.max[2]) * 0.5,
        ];

        let half_width = (box3d.max[0] - box3d.min[0]) * 0.5;
        let half_height = (box3d.max[1] - box3d.min[1]) * 0.5;
        let half_depth = (box3d.max[2] - box3d.min[2]) * 0.5;

        let dx = (sphere.center[0] - box_center[0]).abs() / half_width;
        let dy = (sphere.center[1] - box_center[1]).abs() / half_height;
        let dz = (sphere.center[2] - box_center[2]).abs() / half_depth;

        if dx > dy && dx > dz {
            [sphere.center[0] - box_center[0].signum(), 0.0, 0.0]
        } else if dy > dz {
            [0.0, sphere.center[1] - box_center[1].signum(), 0.0]
        } else {
            [0.0, 0.0, sphere.center[2] - box_center[2].signum()]
        }
    } else {
        [dx / distance, dy / distance, dz / distance]
    };

    let penetration = sphere.radius - distance;

    Some(CollisionInfo {
        contact_point: closest_point,
        normal,
        penetration,
    })
}

/// Finds the collision information for a sphere and a triangle
///
/// # Arguments
///
/// * `sphere` - The sphere
/// * `triangle` - The triangle
///
/// # Returns
///
/// Collision information if a collision occurs, None otherwise
pub fn find_sphere_triangle_collision(
    sphere: &Sphere,
    triangle: &Triangle3D,
) -> Option<CollisionInfo> {
    // First, calculate the normal vector of the triangle
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
        return None; // Degenerate triangle
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
        return None;
    }

    // Project sphere center onto the triangle's plane
    let projected_center = [
        sphere.center[0] - dist_to_plane * normalized_normal[0],
        sphere.center[1] - dist_to_plane * normalized_normal[1],
        sphere.center[2] - dist_to_plane * normalized_normal[2],
    ];

    // Check if the projected center is inside the triangle
    let inside_triangle =
        super::narrowphase::point_triangle3d_collision(&projected_center, triangle);

    // If the projected center is inside the triangle, collision normal is the plane normal
    if inside_triangle {
        let contact_normal = if dist_to_plane > 0.0 {
            normalized_normal
        } else {
            [
                -normalized_normal[0],
                -normalized_normal[1],
                -normalized_normal[2],
            ]
        };

        return Some(CollisionInfo {
            contact_point: [
                sphere.center[0] - contact_normal[0] * sphere.radius,
                sphere.center[1] - contact_normal[1] * sphere.radius,
                sphere.center[2] - contact_normal[2] * sphere.radius,
            ],
            normal: contact_normal,
            penetration: sphere.radius - dist_to_plane.abs(),
        });
    }

    // If not inside, find the closest point on the triangle edges
    let edges = [
        (triangle.v1, triangle.v2),
        (triangle.v2, triangle.v3),
        (triangle.v3, triangle.v1),
    ];

    let mut closest_point = [0.0, 0.0, 0.0];
    let mut min_distance = f64::INFINITY;

    for &(start, end) in &edges {
        // Vector from start to end of the edge
        let edge_vec = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];

        // Vector from start of the edge to the sphere center
        let sphere_vec = [
            sphere.center[0] - start[0],
            sphere.center[1] - start[1],
            sphere.center[2] - start[2],
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
        let point = [
            start[0] + t_clamped * edge_vec[0],
            start[1] + t_clamped * edge_vec[1],
            start[2] + t_clamped * edge_vec[2],
        ];

        // Calculate the distance from the closest point to the sphere center
        let dx = sphere.center[0] - point[0];
        let dy = sphere.center[1] - point[1];
        let dz = sphere.center[2] - point[2];
        let distance_squared = dx * dx + dy * dy + dz * dz;

        if distance_squared < min_distance {
            min_distance = distance_squared;
            closest_point = point;
        }
    }

    // Also check the vertices
    let vertices = [triangle.v1, triangle.v2, triangle.v3];

    for &vertex in &vertices {
        let dx = sphere.center[0] - vertex[0];
        let dy = sphere.center[1] - vertex[1];
        let dz = sphere.center[2] - vertex[2];
        let distance_squared = dx * dx + dy * dy + dz * dz;

        if distance_squared < min_distance {
            min_distance = distance_squared;
            closest_point = vertex;
        }
    }

    // Check if the closest point is within the sphere's radius
    if min_distance > sphere.radius * sphere.radius {
        return None;
    }

    let distance = min_distance.sqrt();
    let normal = [
        (sphere.center[0] - closest_point[0]) / distance,
        (sphere.center[1] - closest_point[1]) / distance,
        (sphere.center[2] - closest_point[2]) / distance,
    ];

    Some(CollisionInfo {
        contact_point: closest_point,
        normal,
        penetration: sphere.radius - distance,
    })
}
