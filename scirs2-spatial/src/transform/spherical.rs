//! Spherical coordinate transformations
//!
//! This module provides functions for transforming between Cartesian and
//! spherical coordinate systems, as well as utilities for working with
//! spherical coordinates.
//!
//! Spherical coordinates use the following convention:
//! - r (radius): Distance from the origin to the point
//! - theta (polar angle): Angle from the positive z-axis (0 to π)
//! - phi (azimuthal angle): Angle from the positive x-axis in the xy-plane (0 to 2π)
//!
//! Note: This convention (r, theta, phi) is used in physics. In mathematics, the
//! convention (r, phi, theta) is sometimes used. We follow the physics convention.

use crate::error::{SpatialError, SpatialResult};
use ndarray::{array, Array1, Array2, ArrayView1, ArrayView2};
use std::f64::consts::{PI, TAU};

/// Converts Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi)
///
/// The spherical coordinates use the following convention:
/// - r: Distance from the origin to the point (0 to ∞)
/// - theta: Angle from the positive z-axis (0 to π)
/// - phi: Angle from the positive x-axis in the xy-plane (0 to 2π)
///
/// # Arguments
///
/// * `cart` - Cartesian coordinates as a 3-element array [x, y, z]
///
/// # Returns
///
/// A 3-element array containing [r, theta, phi]
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::transform::spherical::cart_to_spherical;
///
/// let cart = array![1.0, 1.0, 1.0]; // Point (1, 1, 1)
/// let spherical = cart_to_spherical(&cart.view()).unwrap();
///
/// // r = sqrt(3)
/// // theta = arccos(1/sqrt(3)) = 0.9553 radians (≈54.7°)
/// // phi = arctan(1/1) = 0.7854 radians (45°)
/// ```
///
/// # Errors
///
/// Returns an error if the input array doesn't have exactly 3 elements
pub fn cart_to_spherical(cart: &ArrayView1<f64>) -> SpatialResult<Array1<f64>> {
    if cart.len() != 3 {
        return Err(SpatialError::DimensionError(format!(
            "Cartesian coordinates must have 3 elements, got {}",
            cart.len()
        )));
    }

    let x = cart[0];
    let y = cart[1];
    let z = cart[2];

    // Compute r (radius)
    let r = (x * x + y * y + z * z).sqrt();

    // Handle the case where r is close to zero
    if r < 1e-10 {
        // Return [0, 0, 0] for the origin
        return Ok(Array1::zeros(3));
    }

    // Compute theta (polar angle)
    let theta = if r < 1e-10 {
        0.0 // Default to 0 when at origin
    } else {
        (z / r).acos()
    };

    // Compute phi (azimuthal angle)
    let phi = if x.abs() < 1e-10 && y.abs() < 1e-10 {
        0.0 // Default to 0 when on z-axis
    } else {
        let raw_phi = y.atan2(x);
        // Ensure phi is in [0, 2π)
        if raw_phi < 0.0 {
            raw_phi + TAU
        } else {
            raw_phi
        }
    };

    Ok(array![r, theta, phi])
}

/// Converts spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z)
///
/// The spherical coordinates use the following convention:
/// - r: Distance from the origin to the point (0 to ∞)
/// - theta: Angle from the positive z-axis (0 to π)
/// - phi: Angle from the positive x-axis in the xy-plane (0 to 2π)
///
/// # Arguments
///
/// * `spherical` - Spherical coordinates as a 3-element array [r, theta, phi]
///
/// # Returns
///
/// A 3-element array containing [x, y, z]
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::transform::spherical::spherical_to_cart;
/// use std::f64::consts::PI;
///
/// // Point at r=2, theta=π/4 (45°), phi=π/3 (60°)
/// let spherical = array![2.0, PI/4.0, PI/3.0];
/// let cart = spherical_to_cart(&spherical.view()).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the input array doesn't have exactly 3 elements
pub fn spherical_to_cart(spherical: &ArrayView1<f64>) -> SpatialResult<Array1<f64>> {
    if spherical.len() != 3 {
        return Err(SpatialError::DimensionError(format!(
            "Spherical coordinates must have 3 elements, got {}",
            spherical.len()
        )));
    }

    let r = spherical[0];
    let theta = spherical[1];
    let phi = spherical[2];

    // Check that r is non-negative
    if r < 0.0 {
        return Err(SpatialError::ValueError(format!(
            "Radius r must be non-negative, got {}",
            r
        )));
    }

    // Check that theta is within valid range
    if !(0.0..=PI).contains(&theta) {
        return Err(SpatialError::ValueError(format!(
            "Polar angle theta must be in [0, π], got {}",
            theta
        )));
    }

    // Convert to Cartesian coordinates
    let x = r * theta.sin() * phi.cos();
    let y = r * theta.sin() * phi.sin();
    let z = r * theta.cos();

    Ok(array![x, y, z])
}

/// Converts multiple Cartesian coordinates to spherical coordinates
///
/// # Arguments
///
/// * `cart` - A 2D array of Cartesian coordinates, each row is a 3D point [x, y, z]
///
/// # Returns
///
/// A 2D array of spherical coordinates, each row is [r, theta, phi]
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::transform::spherical::cart_to_spherical_batch;
///
/// let cart = array![
///     [1.0, 0.0, 0.0],  // Point on x-axis
///     [0.0, 1.0, 0.0],  // Point on y-axis
///     [0.0, 0.0, 1.0],  // Point on z-axis
/// ];
/// let spherical = cart_to_spherical_batch(&cart.view()).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if any row of the input array doesn't have exactly 3 elements
pub fn cart_to_spherical_batch(cart: &ArrayView2<f64>) -> SpatialResult<Array2<f64>> {
    if cart.ncols() != 3 {
        return Err(SpatialError::DimensionError(format!(
            "Cartesian coordinates must have 3 columns, got {}",
            cart.ncols()
        )));
    }

    let n_points = cart.nrows();
    let mut result = Array2::zeros((n_points, 3));

    for (i, row) in cart.rows().into_iter().enumerate() {
        let spherical = cart_to_spherical(&row)?;
        result.row_mut(i).assign(&spherical);
    }

    Ok(result)
}

/// Converts multiple spherical coordinates to Cartesian coordinates
///
/// # Arguments
///
/// * `spherical` - A 2D array of spherical coordinates, each row is [r, theta, phi]
///
/// # Returns
///
/// A 2D array of Cartesian coordinates, each row is [x, y, z]
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::transform::spherical::spherical_to_cart_batch;
/// use std::f64::consts::PI;
///
/// let spherical = array![
///     [1.0, PI/2.0, 0.0],      // Point on x-axis
///     [1.0, PI/2.0, PI/2.0],   // Point on y-axis
///     [1.0, 0.0, 0.0],         // Point on z-axis
/// ];
/// let cart = spherical_to_cart_batch(&spherical.view()).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if any row of the input array doesn't have exactly 3 elements
pub fn spherical_to_cart_batch(spherical: &ArrayView2<f64>) -> SpatialResult<Array2<f64>> {
    if spherical.ncols() != 3 {
        return Err(SpatialError::DimensionError(format!(
            "Spherical coordinates must have 3 columns, got {}",
            spherical.ncols()
        )));
    }

    let n_points = spherical.nrows();
    let mut result = Array2::zeros((n_points, 3));

    for (i, row) in spherical.rows().into_iter().enumerate() {
        let cart = spherical_to_cart(&row)?;
        result.row_mut(i).assign(&cart);
    }

    Ok(result)
}

/// Calculates the geodesic distance between two points on a sphere
///
/// # Arguments
///
/// * `spherical1` - Spherical coordinates of the first point [r, theta, phi]
/// * `spherical2` - Spherical coordinates of the second point [r, theta, phi]
///
/// # Returns
///
/// The geodesic distance
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::transform::spherical::geodesic_distance;
/// use std::f64::consts::PI;
///
/// // North pole and a point on the equator
/// let point1 = array![1.0, 0.0, 0.0];         // North pole
/// let point2 = array![1.0, PI/2.0, 0.0];      // Point on equator
///
/// // Distance should be π/2 radians (90°) * radius (1.0)
/// let distance = geodesic_distance(&point1.view(), &point2.view()).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the input arrays don't have exactly 3 elements
pub fn geodesic_distance(
    spherical1: &ArrayView1<f64>,
    spherical2: &ArrayView1<f64>,
) -> SpatialResult<f64> {
    if spherical1.len() != 3 || spherical2.len() != 3 {
        return Err(SpatialError::DimensionError(
            "Spherical coordinates must have 3 elements".into(),
        ));
    }

    let r1 = spherical1[0];
    let theta1 = spherical1[1];
    let phi1 = spherical1[2];

    let r2 = spherical2[0];
    let theta2 = spherical2[1];
    let phi2 = spherical2[2];

    // Check that radii are non-negative
    if r1 < 0.0 || r2 < 0.0 {
        return Err(SpatialError::ValueError(
            "Radius must be non-negative".into(),
        ));
    }

    // For points on spheres with different radii, error is returned
    if (r1 - r2).abs() > 1e-10 {
        return Err(SpatialError::ValueError(
            "Geodesic distance is only defined for points on the same sphere".into(),
        ));
    }

    // Calculate the central angle using the Vincenty formula
    let cos_theta1 = theta1.cos();
    let sin_theta1 = theta1.sin();
    let cos_theta2 = theta2.cos();
    let sin_theta2 = theta2.sin();
    let cos_dphi = (phi1 - phi2).cos();

    // Calculate the angular distance
    let cos_angle = cos_theta1 * cos_theta2 + sin_theta1 * sin_theta2 * cos_dphi;
    // Clamp to valid range for acos
    let cos_angle = cos_angle.clamp(-1.0, 1.0);
    let angle = cos_angle.acos();

    // Calculate the geodesic distance
    let distance = r1 * angle;

    Ok(distance)
}

/// Calculates the area of a spherical triangle given by three points on a unit sphere
///
/// # Arguments
///
/// * `p1` - First point in spherical coordinates [r, theta, phi]
/// * `p2` - Second point in spherical coordinates [r, theta, phi]
/// * `p3` - Third point in spherical coordinates [r, theta, phi]
///
/// # Returns
///
/// The area of the spherical triangle
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::transform::spherical::spherical_triangle_area;
/// use std::f64::consts::PI;
///
/// // Three points on a unit sphere
/// let p1 = array![1.0, 0.0, 0.0];         // North pole
/// let p2 = array![1.0, PI/2.0, 0.0];      // Point on equator, phi=0
/// let p3 = array![1.0, PI/2.0, PI/2.0];   // Point on equator, phi=π/2
///
/// // This forms a spherical triangle with area π/2 steradians
/// let area = spherical_triangle_area(&p1.view(), &p2.view(), &p3.view()).unwrap();
/// ```
pub fn spherical_triangle_area(
    p1: &ArrayView1<f64>,
    p2: &ArrayView1<f64>,
    p3: &ArrayView1<f64>,
) -> SpatialResult<f64> {
    // Ensure we're working with unit vectors on the sphere
    let r1 = p1[0];
    let r2 = p2[0];
    let r3 = p3[0];

    // Convert spherical to Cartesian for easier calculations
    let cart1 = spherical_to_cart(p1)?;
    let cart2 = spherical_to_cart(p2)?;
    let cart3 = spherical_to_cart(p3)?;

    // Normalize to unit vectors
    let v1 = &cart1 / r1;
    let v2 = &cart2 / r2;
    let v3 = &cart3 / r3;

    // Calculate dot products
    let dot12 = v1.dot(&v2);
    let dot23 = v2.dot(&v3);
    let dot31 = v3.dot(&v1);

    // Clamp to valid range for acos
    let dot12 = dot12.clamp(-1.0, 1.0);
    let dot23 = dot23.clamp(-1.0, 1.0);
    let dot31 = dot31.clamp(-1.0, 1.0);

    // Calculate the angles of the geodesic triangle
    let a12 = dot12.acos();
    let a23 = dot23.acos();
    let a31 = dot31.acos();

    // Calculate the area using the spherical excess formula
    // E = A + B + C - π
    let excess = a12 + a23 + a31 - PI;

    // Area = radius^2 * excess
    let area = r1 * r1 * excess;

    Ok(area)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_cart_to_spherical() {
        // Origin
        let cart = array![0.0, 0.0, 0.0];
        let spherical = cart_to_spherical(&cart.view()).unwrap();
        assert_relative_eq!(spherical[0], 0.0);
        assert_relative_eq!(spherical[1], 0.0);
        assert_relative_eq!(spherical[2], 0.0);

        // Point on positive x-axis
        let cart = array![1.0, 0.0, 0.0];
        let spherical = cart_to_spherical(&cart.view()).unwrap();
        assert_relative_eq!(spherical[0], 1.0); // r
        assert_relative_eq!(spherical[1], PI / 2.0); // theta
        assert_relative_eq!(spherical[2], 0.0); // phi

        // Point on positive y-axis
        let cart = array![0.0, 1.0, 0.0];
        let spherical = cart_to_spherical(&cart.view()).unwrap();
        assert_relative_eq!(spherical[0], 1.0); // r
        assert_relative_eq!(spherical[1], PI / 2.0); // theta
        assert_relative_eq!(spherical[2], PI / 2.0); // phi

        // Point on positive z-axis
        let cart = array![0.0, 0.0, 1.0];
        let spherical = cart_to_spherical(&cart.view()).unwrap();
        assert_relative_eq!(spherical[0], 1.0); // r
        assert_relative_eq!(spherical[1], 0.0); // theta
        assert_relative_eq!(spherical[2], 0.0); // phi

        // Point in octant with all positive coordinates
        let cart = array![1.0, 1.0, 1.0];
        let spherical = cart_to_spherical(&cart.view()).unwrap();
        assert_relative_eq!(spherical[0], 3.0_f64.sqrt(), epsilon = 1e-6); // r = sqrt(3)
        assert_relative_eq!(spherical[1], (1.0 / 3.0_f64.sqrt()).acos(), epsilon = 1e-6); // theta = acos(1/sqrt(3))
        assert_relative_eq!(spherical[2], PI / 4.0, epsilon = 1e-6); // phi = π/4
    }

    #[test]
    fn test_spherical_to_cart() {
        // Origin
        let spherical = array![0.0, 0.0, 0.0];
        let cart = spherical_to_cart(&spherical.view()).unwrap();
        assert_relative_eq!(cart[0], 0.0);
        assert_relative_eq!(cart[1], 0.0);
        assert_relative_eq!(cart[2], 0.0);

        // Point on positive x-axis
        let spherical = array![1.0, PI / 2.0, 0.0];
        let cart = spherical_to_cart(&spherical.view()).unwrap();
        assert_relative_eq!(cart[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(cart[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(cart[2], 0.0, epsilon = 1e-6);

        // Point on positive y-axis
        let spherical = array![1.0, PI / 2.0, PI / 2.0];
        let cart = spherical_to_cart(&spherical.view()).unwrap();
        assert_relative_eq!(cart[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(cart[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(cart[2], 0.0, epsilon = 1e-6);

        // Point on positive z-axis
        let spherical = array![1.0, 0.0, 0.0];
        let cart = spherical_to_cart(&spherical.view()).unwrap();
        assert_relative_eq!(cart[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(cart[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(cart[2], 1.0, epsilon = 1e-6);

        // Point with r=2, theta=π/4, phi=π/3
        let spherical = array![2.0, PI / 4.0, PI / 3.0];
        let cart = spherical_to_cart(&spherical.view()).unwrap();
        assert_relative_eq!(
            cart[0],
            2.0 * (PI / 4.0).sin() * (PI / 3.0).cos(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            cart[1],
            2.0 * (PI / 4.0).sin() * (PI / 3.0).sin(),
            epsilon = 1e-6
        );
        assert_relative_eq!(cart[2], 2.0 * (PI / 4.0).cos(), epsilon = 1e-6);
    }

    #[test]
    fn test_roundtrip() {
        // Define a set of points to test
        let cart_points = array![
            [1.0, 0.0, 0.0],  // x-axis
            [0.0, 1.0, 0.0],  // y-axis
            [0.0, 0.0, 1.0],  // z-axis
            [1.0, 1.0, 1.0],  // General point
            [3.0, -2.0, 4.0], // Another general point
        ];

        for row in cart_points.rows() {
            let cart_original = row.to_owned();

            // Convert to spherical
            let spherical = cart_to_spherical(&cart_original.view()).unwrap();

            // Convert back to Cartesian
            let cart_roundtrip = spherical_to_cart(&spherical.view()).unwrap();

            // Check that we get the original point back
            for i in 0..3 {
                assert_relative_eq!(cart_original[i], cart_roundtrip[i], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_batch_conversions() {
        // Test batch conversion from Cartesian to spherical
        let cart = array![
            [1.0, 0.0, 0.0], // x-axis
            [0.0, 1.0, 0.0], // y-axis
            [0.0, 0.0, 1.0], // z-axis
        ];

        let spherical = cart_to_spherical_batch(&cart.view()).unwrap();

        // Check dimensions
        assert_eq!(spherical.shape(), &[3, 3]);

        // Check individual values for x-axis point
        assert_relative_eq!(spherical[[0, 0]], 1.0); // r
        assert_relative_eq!(spherical[[0, 1]], PI / 2.0); // theta
        assert_relative_eq!(spherical[[0, 2]], 0.0); // phi

        // Test batch conversion from spherical to Cartesian
        let cart_roundtrip = spherical_to_cart_batch(&spherical.view()).unwrap();

        // Check that we get the original points back
        assert_eq!(cart_roundtrip.shape(), &[3, 3]);

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(cart[[i, j]], cart_roundtrip[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_geodesic_distance() {
        // North pole to point on equator
        let north_pole = array![1.0, 0.0, 0.0];
        let equator_point = array![1.0, PI / 2.0, 0.0];

        let distance = geodesic_distance(&north_pole.view(), &equator_point.view()).unwrap();
        assert_relative_eq!(distance, PI / 2.0, epsilon = 1e-6); // 90° arc = π/2 radians

        // Two antipodal points on the equator
        let point1 = array![1.0, PI / 2.0, 0.0];
        let point2 = array![1.0, PI / 2.0, PI];

        let distance = geodesic_distance(&point1.view(), &point2.view()).unwrap();
        assert_relative_eq!(distance, PI, epsilon = 1e-6); // 180° arc = π radians

        // Two points 60° apart on a sphere of radius 2
        let point1 = array![2.0, PI / 2.0, 0.0];
        let point2 = array![2.0, PI / 2.0, PI / 3.0];

        let distance = geodesic_distance(&point1.view(), &point2.view()).unwrap();
        assert_relative_eq!(distance, 2.0 * PI / 3.0, epsilon = 1e-6); // 60° arc * radius 2 = 2π/3
    }

    #[test]
    fn test_spherical_triangle_area() {
        // Octant triangle (1/8 of the sphere)
        let p1 = array![1.0, 0.0, 0.0]; // North pole
        let p2 = array![1.0, PI / 2.0, 0.0]; // Point on equator at phi=0
        let p3 = array![1.0, PI / 2.0, PI / 2.0]; // Point on equator at phi=π/2

        let area = spherical_triangle_area(&p1.view(), &p2.view(), &p3.view()).unwrap();
        assert_relative_eq!(area, PI / 2.0, epsilon = 1e-6); // Area = π/2 steradians

        // 90° wedge on the sphere
        let p1 = array![1.0, 0.0, 0.0]; // North pole
        let p2 = array![1.0, PI, 0.0]; // South pole
        let p3 = array![1.0, PI / 2.0, 0.0]; // Point on equator at phi=0

        let area = spherical_triangle_area(&p1.view(), &p2.view(), &p3.view()).unwrap();
        assert_relative_eq!(area, PI, epsilon = 1e-6); // Area = π steradians (1/4 of sphere)
    }

    #[test]
    fn test_error_cases() {
        // Test error for wrong dimension in Cartesian to spherical
        let bad_cart = array![1.0, 2.0];
        assert!(cart_to_spherical(&bad_cart.view()).is_err());

        // Test error for wrong dimension in spherical to Cartesian
        let bad_spherical = array![1.0, 2.0];
        assert!(spherical_to_cart(&bad_spherical.view()).is_err());

        // Test error for negative radius
        let neg_radius = array![-1.0, PI / 2.0, 0.0];
        assert!(spherical_to_cart(&neg_radius.view()).is_err());

        // Test error for invalid theta (polar angle)
        let bad_theta = array![1.0, -0.1, 0.0];
        assert!(spherical_to_cart(&bad_theta.view()).is_err());

        let bad_theta = array![1.0, PI + 0.1, 0.0];
        assert!(spherical_to_cart(&bad_theta.view()).is_err());

        // Test error for geodesic distance with different radii
        let p1 = array![1.0, 0.0, 0.0];
        let p2 = array![2.0, 0.0, 0.0];
        assert!(geodesic_distance(&p1.view(), &p2.view()).is_err());
    }
}
