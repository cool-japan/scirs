//! Geospatial functionality for working with geographic coordinates
//!
//! This module provides basic geospatial operations including:
//! - Coordinate system transformations
//! - Great circle distance calculations (Haversine formula)
//! - Bearing and azimuth calculations
//! - Geodesic operations on the sphere
//!
//! # Coordinate Systems
//!
//! The module supports common geographic coordinate systems:
//! - **WGS84**: World Geodetic System 1984 (GPS standard)
//! - **Geographic**: Latitude/Longitude coordinates
//! - **UTM**: Universal Transverse Mercator projections
//! - **Web Mercator**: Spherical Mercator (EPSG:3857)
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::geospatial::{haversine_distance, initial_bearing, destination_point};
//!
//! // Calculate distance between two cities
//! let london = (51.5074, -0.1278);  // Latitude, Longitude
//! let paris = (48.8566, 2.3522);
//!
//! let distance = haversine_distance(london, paris);
//! println!("Distance from London to Paris: {:.1} km", distance / 1000.0);
//!
//! // Calculate bearing
//! let bearing = initial_bearing(london, paris);
//! println!("Initial bearing: {:.1}°", bearing.to_degrees());
//!
//! // Find destination point
//! let destination = destination_point(london, 100000.0, bearing); // 100 km
//! println!("100km from London: ({:.4}, {:.4})", destination.0, destination.1);
//! ```

use crate::error::{SpatialError, SpatialResult};
use std::f64::consts::PI;

/// Earth radius in meters (WGS84 mean radius)
pub const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// Earth radius in kilometers
pub const EARTH_RADIUS_KM: f64 = EARTH_RADIUS_M / 1000.0;

/// Earth's equatorial radius in meters (WGS84)
pub const EARTH_EQUATORIAL_RADIUS_M: f64 = 6_378_137.0;

/// Earth's polar radius in meters (WGS84)
pub const EARTH_POLAR_RADIUS_M: f64 = 6_356_752.314245;

/// Earth's flattening (WGS84)
pub const EARTH_FLATTENING: f64 = 1.0 / 298.257223563;

/// Earth's eccentricity squared (WGS84)
pub const EARTH_ECCENTRICITY_SQ: f64 = 2.0 * EARTH_FLATTENING - EARTH_FLATTENING * EARTH_FLATTENING;

/// Convert degrees to radians
pub fn deg_to_rad(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

/// Convert radians to degrees
pub fn rad_to_deg(radians: f64) -> f64 {
    radians * 180.0 / PI
}

/// Normalize angle to [0, 2π) range
pub fn normalize_angle(angle: f64) -> f64 {
    let normalized = angle % (2.0 * PI);
    if normalized < 0.0 {
        normalized + 2.0 * PI
    } else {
        normalized
    }
}

/// Normalize bearing to [0°, 360°) range
pub fn normalize_bearing(bearing_deg: f64) -> f64 {
    let normalized = bearing_deg % 360.0;
    if normalized < 0.0 {
        normalized + 360.0
    } else {
        normalized
    }
}

/// Calculate the great circle distance between two points using the Haversine formula
///
/// This is the most common method for calculating distances on a sphere.
/// The Haversine formula is numerically stable for small distances.
///
/// # Arguments
///
/// * `point1` - First point as (latitude, longitude) in degrees
/// * `point2` - Second point as (latitude, longitude) in degrees
///
/// # Returns
///
/// * Distance in meters
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geospatial::haversine_distance;
///
/// let new_york = (40.7128, -74.0060);
/// let london = (51.5074, -0.1278);
///
/// let distance = haversine_distance(new_york, london);
/// println!("Distance: {:.1} km", distance / 1000.0);
/// ```
pub fn haversine_distance(point1: (f64, f64), point2: (f64, f64)) -> f64 {
    let (lat1, lon1) = (deg_to_rad(point1.0), deg_to_rad(point1.1));
    let (lat2, lon2) = (deg_to_rad(point2.0), deg_to_rad(point2.1));

    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    EARTH_RADIUS_M * c
}

/// Calculate the initial bearing (forward azimuth) from point1 to point2
///
/// # Arguments
///
/// * `point1` - Starting point as (latitude, longitude) in degrees
/// * `point2` - End point as (latitude, longitude) in degrees
///
/// # Returns
///
/// * Initial bearing in radians (0 = North, π/2 = East, π = South, 3π/2 = West)
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geospatial::initial_bearing;
///
/// let start = (40.7128, -74.0060);  // New York
/// let end = (51.5074, -0.1278);     // London
///
/// let bearing = initial_bearing(start, end);
/// println!("Bearing: {:.1}°", bearing.to_degrees());
/// ```
pub fn initial_bearing(point1: (f64, f64), point2: (f64, f64)) -> f64 {
    let (lat1, lon1) = (deg_to_rad(point1.0), deg_to_rad(point1.1));
    let (lat2, lon2) = (deg_to_rad(point2.0), deg_to_rad(point2.1));

    let dlon = lon2 - lon1;

    let y = dlon.sin() * lat2.cos();
    let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();

    normalize_angle(y.atan2(x))
}

/// Calculate the final bearing (back azimuth) when arriving at point2 from point1
///
/// # Arguments
///
/// * `point1` - Starting point as (latitude, longitude) in degrees
/// * `point2` - End point as (latitude, longitude) in degrees
///
/// # Returns
///
/// * Final bearing in radians
pub fn final_bearing(point1: (f64, f64), point2: (f64, f64)) -> f64 {
    let reverse_bearing = initial_bearing(point2, point1);
    normalize_angle(reverse_bearing + PI)
}

/// Calculate the destination point given a starting point, distance, and bearing
///
/// # Arguments
///
/// * `start` - Starting point as (latitude, longitude) in degrees
/// * `distance` - Distance to travel in meters
/// * `bearing` - Bearing to travel in radians (0 = North)
///
/// # Returns
///
/// * Destination point as (latitude, longitude) in degrees
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geospatial::destination_point;
///
/// let start = (40.7128, -74.0060);  // New York
/// let distance = 100_000.0;         // 100 km
/// let bearing = std::f64::consts::PI / 4.0;  // 45° (Northeast)
///
/// let destination = destination_point(start, distance, bearing);
/// println!("Destination: ({:.4}, {:.4})", destination.0, destination.1);
/// ```
pub fn destination_point(start: (f64, f64), distance: f64, bearing: f64) -> (f64, f64) {
    let (lat1, lon1) = (deg_to_rad(start.0), deg_to_rad(start.1));

    let angular_distance = distance / EARTH_RADIUS_M;

    let lat2 = (lat1.sin() * angular_distance.cos()
        + lat1.cos() * angular_distance.sin() * bearing.cos())
    .asin();

    let lon2 = lon1
        + (bearing.sin() * angular_distance.sin() * lat1.cos())
            .atan2(angular_distance.cos() - lat1.sin() * lat2.sin());

    (rad_to_deg(lat2), rad_to_deg(lon2))
}

/// Calculate the midpoint between two geographic points
///
/// # Arguments
///
/// * `point1` - First point as (latitude, longitude) in degrees
/// * `point2` - Second point as (latitude, longitude) in degrees
///
/// # Returns
///
/// * Midpoint as (latitude, longitude) in degrees
pub fn midpoint(point1: (f64, f64), point2: (f64, f64)) -> (f64, f64) {
    let (lat1, lon1) = (deg_to_rad(point1.0), deg_to_rad(point1.1));
    let (lat2, lon2) = (deg_to_rad(point2.0), deg_to_rad(point2.1));

    let dlon = lon2 - lon1;

    let bx = lat2.cos() * dlon.cos();
    let by = lat2.cos() * dlon.sin();

    let lat_mid = (lat1.sin() + lat2.sin()).atan2(((lat1.cos() + bx).powi(2) + by.powi(2)).sqrt());

    let lon_mid = lon1 + by.atan2(lat1.cos() + bx);

    (rad_to_deg(lat_mid), rad_to_deg(lon_mid))
}

/// Calculate the cross-track distance (distance from a point to a great circle path)
///
/// # Arguments
///
/// * `point` - Point to measure distance from, as (latitude, longitude) in degrees
/// * `path_start` - Start of the great circle path, as (latitude, longitude) in degrees
/// * `path_end` - End of the great circle path, as (latitude, longitude) in degrees
///
/// # Returns
///
/// * Cross-track distance in meters (positive if point is to the right of the path)
pub fn cross_track_distance(
    point: (f64, f64),
    path_start: (f64, f64),
    path_end: (f64, f64),
) -> f64 {
    let distance_to_start = haversine_distance(path_start, point) / EARTH_RADIUS_M;
    let bearing_to_point = initial_bearing(path_start, point);
    let bearing_to_end = initial_bearing(path_start, path_end);

    let cross_track_angular =
        (distance_to_start.sin() * (bearing_to_point - bearing_to_end).sin()).asin();

    EARTH_RADIUS_M * cross_track_angular
}

/// Calculate the along-track distance (distance along a great circle path to the closest point)
///
/// # Arguments
///
/// * `point` - Point to project onto the path, as (latitude, longitude) in degrees
/// * `path_start` - Start of the great circle path, as (latitude, longitude) in degrees
/// * `path_end` - End of the great circle path, as (latitude, longitude) in degrees
///
/// # Returns
///
/// * Along-track distance in meters from path_start to the closest point on the path
pub fn along_track_distance(
    point: (f64, f64),
    path_start: (f64, f64),
    path_end: (f64, f64),
) -> f64 {
    let distance_to_start = haversine_distance(path_start, point) / EARTH_RADIUS_M;
    let cross_track_angular = cross_track_distance(point, path_start, path_end) / EARTH_RADIUS_M;

    let along_track_angular = (distance_to_start.powi(2) - cross_track_angular.powi(2))
        .sqrt()
        .acos();

    EARTH_RADIUS_M * along_track_angular
}

/// Calculate the area of a polygon on the sphere using spherical excess
///
/// # Arguments
///
/// * `polygon` - Vector of points as (latitude, longitude) in degrees
///
/// # Returns
///
/// * Area in square meters
///
/// # Note
///
/// This uses the spherical excess method. For very large polygons, more sophisticated
/// methods may be needed to handle numerical precision issues.
pub fn spherical_polygon_area(polygon: &[(f64, f64)]) -> SpatialResult<f64> {
    if polygon.len() < 3 {
        return Err(SpatialError::ValueError(
            "Polygon must have at least 3 vertices".to_string(),
        ));
    }

    let n = polygon.len();
    let mut sum = 0.0;

    for i in 0..n {
        let j = (i + 1) % n;
        let (lat1, lon1) = (deg_to_rad(polygon[i].0), deg_to_rad(polygon[i].1));
        let (lat2, lon2) = (deg_to_rad(polygon[j].0), deg_to_rad(polygon[j].1));

        sum += (lon2 - lon1) * (2.0 + lat1.sin() + lat2.sin());
    }

    let area = (sum.abs() / 2.0) * EARTH_RADIUS_M * EARTH_RADIUS_M;
    Ok(area)
}

/// Check if a point is inside a spherical polygon using the winding number method
///
/// # Arguments
///
/// * `point` - Point to test as (latitude, longitude) in degrees
/// * `polygon` - Polygon vertices as (latitude, longitude) in degrees
///
/// # Returns
///
/// * true if point is inside the polygon, false otherwise
pub fn point_in_spherical_polygon(point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    // For small polygons (< 10 degrees), use planar approximation for better numerical stability
    let max_extent = polygon
        .iter()
        .flat_map(|(lat, lon)| [lat.abs(), lon.abs()])
        .fold(0.0, f64::max);

    if max_extent < 10.0 {
        // Use planar point-in-polygon algorithm (ray casting)
        let (x, y) = point;
        let mut inside = false;
        let n = polygon.len();

        for i in 0..n {
            let j = (i + 1) % n;
            let (xi, yi) = polygon[i];
            let (xj, yj) = polygon[j];

            if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
        }
        return inside;
    }

    // For larger polygons, use proper spherical calculation
    let (test_lat, test_lon) = (deg_to_rad(point.0), deg_to_rad(point.1));
    let mut angle_sum = 0.0;

    for i in 0..polygon.len() {
        let j = (i + 1) % polygon.len();
        let (lat1, lon1) = (deg_to_rad(polygon[i].0), deg_to_rad(polygon[i].1));
        let (lat2, lon2) = (deg_to_rad(polygon[j].0), deg_to_rad(polygon[j].1));

        // Convert to 3D Cartesian coordinates on unit sphere
        let x1 = lat1.cos() * lon1.cos();
        let y1 = lat1.cos() * lon1.sin();
        let z1 = lat1.sin();

        let x2 = lat2.cos() * lon2.cos();
        let y2 = lat2.cos() * lon2.sin();
        let z2 = lat2.sin();

        let xt = test_lat.cos() * test_lon.cos();
        let yt = test_lat.cos() * test_lon.sin();
        let zt = test_lat.sin();

        // Vectors from test point to polygon vertices
        let v1x = x1 - xt;
        let v1y = y1 - yt;
        let v1z = z1 - zt;

        let v2x = x2 - xt;
        let v2y = y2 - yt;
        let v2z = z2 - zt;

        // Normalize vectors
        let v1_len = (v1x * v1x + v1y * v1y + v1z * v1z).sqrt();
        let v2_len = (v2x * v2x + v2y * v2y + v2z * v2z).sqrt();

        if v1_len < 1e-10 || v2_len < 1e-10 {
            continue; // Point is on a vertex
        }

        let v1x_norm = v1x / v1_len;
        let v1y_norm = v1y / v1_len;
        let v1z_norm = v1z / v1_len;

        let v2x_norm = v2x / v2_len;
        let v2y_norm = v2y / v2_len;
        let v2z_norm = v2z / v2_len;

        // Calculate angle between vectors
        let dot = v1x_norm * v2x_norm + v1y_norm * v2y_norm + v1z_norm * v2z_norm;
        let dot = dot.clamp(-1.0, 1.0); // Handle numerical errors

        // Cross product for sign
        let cross_x = v1y_norm * v2z_norm - v1z_norm * v2y_norm;
        let cross_y = v1z_norm * v2x_norm - v1x_norm * v2z_norm;
        let cross_z = v1x_norm * v2y_norm - v1y_norm * v2x_norm;

        // Project cross product onto normal at test point to get sign
        let normal_dot = cross_x * xt + cross_y * yt + cross_z * zt;
        let angle = dot.acos();

        if normal_dot < 0.0 {
            angle_sum -= angle;
        } else {
            angle_sum += angle;
        }
    }

    (angle_sum.abs() / (2.0 * PI)) > 0.5
}

/// Convert geographic coordinates to UTM coordinates
///
/// # Arguments
///
/// * `lat` - Latitude in degrees
/// * `lon` - Longitude in degrees
///
/// # Returns
///
/// * (easting, northing, zone_number, zone_letter)
///
/// # Note
///
/// This is a simplified UTM conversion. For high-precision applications,
/// use specialized geospatial libraries like PROJ.
pub fn geographic_to_utm(lat: f64, lon: f64) -> SpatialResult<(f64, f64, i32, char)> {
    if !(-80.0..=84.0).contains(&lat) {
        return Err(SpatialError::ValueError(
            "Latitude must be between -80° and 84° for UTM".to_string(),
        ));
    }

    let zone_number = ((lon + 180.0) / 6.0).floor() as i32 + 1;
    let zone_letter = utm_zone_letter(lat)?;

    let lat_rad = deg_to_rad(lat);
    let lon_rad = deg_to_rad(lon);
    let central_meridian = deg_to_rad(((zone_number - 1) * 6 - 177) as f64);

    let k0 = 0.9996; // UTM scale factor
    let a = EARTH_EQUATORIAL_RADIUS_M;
    let e_sq = EARTH_ECCENTRICITY_SQ;

    let n = a / (1.0 - e_sq * lat_rad.sin().powi(2)).sqrt();
    let t = lat_rad.tan().powi(2);
    let c = EARTH_ECCENTRICITY_SQ * lat_rad.cos().powi(2) / (1.0 - EARTH_ECCENTRICITY_SQ);
    let a_coeff = lat_rad.cos() * (lon_rad - central_meridian);

    let m = a
        * ((1.0 - e_sq / 4.0 - 3.0 * e_sq.powi(2) / 64.0 - 5.0 * e_sq.powi(3) / 256.0) * lat_rad
            - (3.0 * e_sq / 8.0 + 3.0 * e_sq.powi(2) / 32.0 + 45.0 * e_sq.powi(3) / 1024.0)
                * (2.0 * lat_rad).sin()
            + (15.0 * e_sq.powi(2) / 256.0 + 45.0 * e_sq.powi(3) / 1024.0) * (4.0 * lat_rad).sin()
            - (35.0 * e_sq.powi(3) / 3072.0) * (6.0 * lat_rad).sin());

    let easting = k0
        * n
        * (a_coeff
            + (1.0 - t + c) * a_coeff.powi(3) / 6.0
            + (5.0 - 18.0 * t + t.powi(2) + 72.0 * c - 58.0 * EARTH_ECCENTRICITY_SQ)
                * a_coeff.powi(5)
                / 120.0)
        + 500000.0;

    let northing = k0
        * (m + n
            * lat_rad.tan()
            * (a_coeff.powi(2) / 2.0
                + (5.0 - t + 9.0 * c + 4.0 * c.powi(2)) * a_coeff.powi(4) / 24.0
                + (61.0 - 58.0 * t + t.powi(2) + 600.0 * c - 330.0 * EARTH_ECCENTRICITY_SQ)
                    * a_coeff.powi(6)
                    / 720.0));

    let final_northing = if lat < 0.0 {
        northing + 10000000.0
    } else {
        northing
    };

    Ok((easting, final_northing, zone_number, zone_letter))
}

/// Get UTM zone letter from latitude
fn utm_zone_letter(lat: f64) -> SpatialResult<char> {
    let letters = [
        'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
        'W', 'X',
    ];

    if !(-80.0..=84.0).contains(&lat) {
        return Err(SpatialError::ValueError(
            "Latitude out of UTM range".to_string(),
        ));
    }

    let index = ((lat + 80.0) / 8.0).floor() as usize;
    if index < letters.len() {
        Ok(letters[index])
    } else {
        Ok('X') // Special case for 72°-84°N
    }
}

/// Convert geographic coordinates to Web Mercator (EPSG:3857)
///
/// # Arguments
///
/// * `lat` - Latitude in degrees
/// * `lon` - Longitude in degrees
///
/// # Returns
///
/// * (x, y) in Web Mercator coordinates (meters)
pub fn geographic_to_web_mercator(lat: f64, lon: f64) -> SpatialResult<(f64, f64)> {
    if lat.abs() >= 85.051_128_779_806_59 {
        return Err(SpatialError::ValueError(
            "Latitude must be between -85.051° and 85.051° for Web Mercator".to_string(),
        ));
    }

    let x = deg_to_rad(lon) * EARTH_EQUATORIAL_RADIUS_M;
    let y = ((deg_to_rad(lat) / 2.0 + PI / 4.0).tan()).ln() * EARTH_EQUATORIAL_RADIUS_M;

    Ok((x, y))
}

/// Convert Web Mercator coordinates to geographic coordinates
///
/// # Arguments
///
/// * `x` - X coordinate in Web Mercator (meters)
/// * `y` - Y coordinate in Web Mercator (meters)
///
/// # Returns
///
/// * (latitude, longitude) in degrees
pub fn web_mercator_to_geographic(x: f64, y: f64) -> (f64, f64) {
    let lon = rad_to_deg(x / EARTH_EQUATORIAL_RADIUS_M);
    let lat = rad_to_deg(2.0 * ((y / EARTH_EQUATORIAL_RADIUS_M).exp().atan() - PI / 4.0));

    (lat, lon)
}

/// Calculate the vincenty distance between two points (more accurate than Haversine)
///
/// This uses Vincenty's inverse formula for ellipsoidal calculations.
/// More accurate than Haversine for long distances.
///
/// # Arguments
///
/// * `point1` - First point as (latitude, longitude) in degrees
/// * `point2` - Second point as (latitude, longitude) in degrees
///
/// # Returns
///
/// * Distance in meters
pub fn vincenty_distance(point1: (f64, f64), point2: (f64, f64)) -> SpatialResult<f64> {
    let (lat1, lon1) = (deg_to_rad(point1.0), deg_to_rad(point1.1));
    let (lat2, lon2) = (deg_to_rad(point2.0), deg_to_rad(point2.1));

    let a = EARTH_EQUATORIAL_RADIUS_M;
    let b = EARTH_POLAR_RADIUS_M;
    let f = EARTH_FLATTENING;

    let l = lon2 - lon1;
    let u1 = ((1.0 - f) * lat1.tan()).atan();
    let u2 = ((1.0 - f) * lat2.tan()).atan();

    let sin_u1 = u1.sin();
    let cos_u1 = u1.cos();
    let sin_u2 = u2.sin();
    let cos_u2 = u2.cos();

    let mut lambda = l;
    let mut lambda_prev;
    let mut iteration_limit = 100;

    let (cos_sq_alpha, sin_sigma, cos_sigma, sigma, cos_2sigma_m) = loop {
        iteration_limit -= 1;
        if iteration_limit == 0 {
            return Err(SpatialError::ComputationError(
                "Vincenty formula failed to converge".to_string(),
            ));
        }

        let sin_lambda = lambda.sin();
        let cos_lambda = lambda.cos();

        let sin_sigma = ((cos_u2 * sin_lambda).powi(2)
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2))
        .sqrt();

        if sin_sigma == 0.0 {
            return Ok(0.0); // Co-incident points
        }

        let cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
        let sigma = sin_sigma.atan2(cos_sigma);

        let sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
        let cos_sq_alpha = 1.0 - sin_alpha.powi(2);

        let cos_2sigma_m = if cos_sq_alpha == 0.0 {
            0.0 // Equatorial line
        } else {
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
        };

        let c = f / 16.0 * cos_sq_alpha * (4.0 + f * (4.0 - 3.0 * cos_sq_alpha));

        lambda_prev = lambda;
        lambda = l
            + (1.0 - c)
                * f
                * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos_2sigma_m + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m.powi(2))));

        if (lambda - lambda_prev).abs() < 1e-12 {
            break (cos_sq_alpha, sin_sigma, cos_sigma, sigma, cos_2sigma_m);
        }
    };

    let u_sq = cos_sq_alpha * (a.powi(2) - b.powi(2)) / b.powi(2);
    let a_coeff = 1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
    let b_coeff = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));

    let delta_sigma = b_coeff
        * sin_sigma
        * (cos_2sigma_m
            + b_coeff / 4.0
                * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m.powi(2))
                    - b_coeff / 6.0
                        * cos_2sigma_m
                        * (-3.0 + 4.0 * sin_sigma.powi(2))
                        * (-3.0 + 4.0 * cos_2sigma_m.powi(2))));

    let distance = b * a_coeff * (sigma - delta_sigma);

    Ok(distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_degree_radian_conversion() {
        assert_relative_eq!(deg_to_rad(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(deg_to_rad(90.0), PI / 2.0, epsilon = 1e-10);
        assert_relative_eq!(deg_to_rad(180.0), PI, epsilon = 1e-10);
        assert_relative_eq!(deg_to_rad(360.0), 2.0 * PI, epsilon = 1e-10);

        assert_relative_eq!(rad_to_deg(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(rad_to_deg(PI / 2.0), 90.0, epsilon = 1e-10);
        assert_relative_eq!(rad_to_deg(PI), 180.0, epsilon = 1e-10);
        assert_relative_eq!(rad_to_deg(2.0 * PI), 360.0, epsilon = 1e-10);
    }

    #[test]
    fn test_haversine_distance() {
        // Distance between London and Paris (approximately 344 km)
        let london = (51.5074, -0.1278);
        let paris = (48.8566, 2.3522);
        let distance = haversine_distance(london, paris);
        assert!((distance - 344_000.0).abs() < 5_000.0); // Within 5km tolerance

        // Distance from a point to itself should be 0
        assert_relative_eq!(haversine_distance(london, london), 0.0, epsilon = 1e-6);

        // Antipodal points (opposite sides of Earth)
        let north_pole = (90.0, 0.0);
        let south_pole = (-90.0, 0.0);
        let antipodal_distance = haversine_distance(north_pole, south_pole);
        let expected_distance = PI * EARTH_RADIUS_M;
        assert_relative_eq!(antipodal_distance, expected_distance, epsilon = 1000.0);
    }

    #[test]
    fn test_initial_bearing() {
        // Bearing from London to Paris should be roughly southeast
        let london = (51.5074, -0.1278);
        let paris = (48.8566, 2.3522);
        let bearing = initial_bearing(london, paris);
        let bearing_deg = rad_to_deg(bearing);

        // Should be roughly in southeast direction (around 120-150 degrees)
        assert!(bearing_deg > 100.0 && bearing_deg < 180.0);

        // Bearing due north
        let start = (0.0, 0.0);
        let north = (1.0, 0.0);
        let north_bearing = initial_bearing(start, north);
        assert_relative_eq!(north_bearing, 0.0, epsilon = 1e-6);

        // Bearing due east
        let east = (0.0, 1.0);
        let east_bearing = initial_bearing(start, east);
        assert_relative_eq!(east_bearing, PI / 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_destination_point() {
        let start = (51.5074, -0.1278); // London
        let distance = 100_000.0; // 100 km
        let bearing = 0.0; // Due north

        let destination = destination_point(start, distance, bearing);

        // Should be roughly north of London
        assert!(destination.0 > start.0); // Latitude should increase
        assert!((destination.1 - start.1).abs() < 0.1); // Longitude should change little

        // Verify round trip
        let calculated_distance = haversine_distance(start, destination);
        assert_relative_eq!(calculated_distance, distance, epsilon = 1000.0); // Within 1km
    }

    #[test]
    fn test_midpoint() {
        let london = (51.5074, -0.1278);
        let paris = (48.8566, 2.3522);
        let mid = midpoint(london, paris);

        // Midpoint should be between the two cities
        assert!(mid.0 < london.0 && mid.0 > paris.0); // Latitude between
        assert!(mid.1 > london.1 && mid.1 < paris.1); // Longitude between

        // Distance from midpoint to each city should be roughly equal
        let dist_to_london = haversine_distance(mid, london);
        let dist_to_paris = haversine_distance(mid, paris);
        assert_relative_eq!(dist_to_london, dist_to_paris, epsilon = 1000.0);
    }

    #[test]
    fn test_normalize_angle() {
        assert_relative_eq!(normalize_angle(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(normalize_angle(PI), PI, epsilon = 1e-10);
        assert_relative_eq!(normalize_angle(2.0 * PI), 0.0, epsilon = 1e-10);
        assert_relative_eq!(normalize_angle(-PI), PI, epsilon = 1e-10);
        assert_relative_eq!(normalize_angle(3.0 * PI), PI, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_bearing() {
        assert_relative_eq!(normalize_bearing(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(normalize_bearing(180.0), 180.0, epsilon = 1e-10);
        assert_relative_eq!(normalize_bearing(360.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(normalize_bearing(-90.0), 270.0, epsilon = 1e-10);
        assert_relative_eq!(normalize_bearing(450.0), 90.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spherical_polygon_area() {
        // Simple triangle
        let triangle = vec![
            (0.0, 0.0), // Equator, Greenwich
            (0.0, 1.0), // Equator, 1° East
            (1.0, 0.0), // 1° North, Greenwich
        ];

        let area = spherical_polygon_area(&triangle).unwrap();
        assert!(area > 0.0);

        // Area should be reasonable for a 1°×1° triangle
        // Expected area is roughly (π/180)² * R² / 2
        let expected = (PI / 180.0).powi(2) * EARTH_RADIUS_M.powi(2) / 2.0;
        assert_relative_eq!(area, expected, epsilon = expected * 0.1);
    }

    #[test]
    fn test_geographic_to_web_mercator() {
        // Test equator and prime meridian
        let (x, y) = geographic_to_web_mercator(0.0, 0.0).unwrap();
        assert_relative_eq!(x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(y, 0.0, epsilon = 1e-6);

        // Test round trip
        let original = (45.0, -90.0);
        let (x, y) = geographic_to_web_mercator(original.0, original.1).unwrap();
        let back = web_mercator_to_geographic(x, y);
        assert_relative_eq!(back.0, original.0, epsilon = 1e-6);
        assert_relative_eq!(back.1, original.1, epsilon = 1e-6);

        // Test error case
        let result = geographic_to_web_mercator(86.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_geographic_to_utm() {
        // Test a known location (London)
        let london = (51.5074, -0.1278);
        let (easting, northing, zone, letter) = geographic_to_utm(london.0, london.1).unwrap();

        // London should be in UTM zone 30 or 31
        assert!(zone == 30 || zone == 31);
        assert!(letter == 'U' || letter == 'V');

        // Coordinates should be reasonable
        assert!(easting > 400_000.0 && easting < 700_000.0);
        assert!(northing > 5_700_000.0 && northing < 5_800_000.0);

        // Test error cases
        assert!(geographic_to_utm(85.0, 0.0).is_err()); // Latitude too high
        assert!(geographic_to_utm(-85.0, 0.0).is_err()); // Latitude too low
    }

    #[test]
    fn test_cross_track_distance() {
        let start = (51.0, 0.0);
        let end = (52.0, 1.0);
        let point = (51.5, 0.0); // Point on the same meridian as start

        let cross_track = cross_track_distance(point, start, end);

        // Should be relatively small since point is close to the great circle
        assert!(cross_track.abs() < 50_000.0); // Within 50km
    }

    #[test]
    fn test_vincenty_distance() {
        // Test against Haversine for short distance
        let london = (51.5074, -0.1278);
        let paris = (48.8566, 2.3522);

        let haversine_dist = haversine_distance(london, paris);
        let vincenty_dist = vincenty_distance(london, paris).unwrap();

        // Should be very close for moderate distances
        let diff_percent = ((vincenty_dist - haversine_dist) / haversine_dist * 100.0).abs();
        assert!(diff_percent < 1.0); // Within 1%

        // Test identical points
        let same_point_dist = vincenty_distance(london, london).unwrap();
        assert_relative_eq!(same_point_dist, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_point_in_spherical_polygon() {
        // Simple square around equator
        let square = vec![(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

        // Point inside
        assert!(point_in_spherical_polygon((0.0, 0.0), &square));

        // Point outside
        assert!(!point_in_spherical_polygon((2.0, 2.0), &square));

        // Point on edge (may be unstable, so just ensure it doesn't crash)
        let _ = point_in_spherical_polygon((1.0, 0.0), &square);
    }
}
