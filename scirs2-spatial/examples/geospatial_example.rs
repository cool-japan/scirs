//! Geospatial Functionality Example
//!
//! This example demonstrates various geospatial operations including:
//! - Distance calculations (Haversine and Vincenty)
//! - Bearing and navigation calculations
//! - Coordinate system transformations
//! - Spherical geometry operations

use scirs2_spatial::geospatial::*;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Geospatial Functionality Example ===\n");

    // Example 1: Distance calculations between cities
    println!("1. Distance Calculations Between Major Cities");
    distance_calculations_example()?;
    println!();

    // Example 2: Navigation and bearing calculations
    println!("2. Navigation and Bearing Calculations");
    navigation_example()?;
    println!();

    // Example 3: Coordinate system transformations
    println!("3. Coordinate System Transformations");
    coordinate_transformations_example()?;
    println!();

    // Example 4: Flight path analysis
    println!("4. Flight Path Analysis");
    flight_path_example()?;
    println!();

    // Example 5: Spherical geometry
    println!("5. Spherical Geometry Operations");
    spherical_geometry_example()?;
    println!();

    // Example 6: GPS tracking simulation
    println!("6. GPS Tracking Simulation");
    gps_tracking_example()?;
    println!();

    // Example 7: Geospatial analysis
    println!("7. Geospatial Analysis");
    geospatial_analysis_example()?;

    Ok(())
}

fn distance_calculations_example() -> Result<(), Box<dyn std::error::Error>> {
    // Define major world cities
    let cities = vec![
        ("London", (51.5074, -0.1278)),
        ("Paris", (48.8566, 2.3522)),
        ("New York", (40.7128, -74.0060)),
        ("Tokyo", (35.6762, 139.6503)),
        ("Sydney", (-33.8688, 151.2093)),
        ("São Paulo", (-23.5505, -46.6333)),
        ("Cairo", (30.0444, 31.2357)),
        ("Mumbai", (19.0760, 72.8777)),
    ];

    println!("Distance matrix between major cities (km):");
    println!(
        "{:>12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "City", "London", "Paris", "New York", "Tokyo", "Sydney", "São Paulo", "Cairo", "Mumbai"
    );

    for (i, (city1, coord1)) in cities.iter().enumerate() {
        print!("{:>12}", city1);
        for (j, (_, coord2)) in cities.iter().enumerate() {
            if i <= j {
                let distance_m = haversine_distance(*coord1, *coord2);
                let distance_km = distance_m / 1000.0;
                print!(" {:>8.0}", distance_km);
            } else {
                print!(" {:>8}", "-");
            }
        }
        println!();
    }

    // Compare Haversine vs Vincenty for long distances
    println!("\nHaversine vs Vincenty comparison for long distances:");
    let long_distance_pairs = vec![
        ("London", "Sydney", cities[0].1, cities[4].1),
        ("New York", "Tokyo", cities[2].1, cities[3].1),
        ("Cairo", "São Paulo", cities[6].1, cities[5].1),
    ];

    for (city1, city2, coord1, coord2) in long_distance_pairs {
        let haversine_km = haversine_distance(coord1, coord2) / 1000.0;
        let vincenty_km = vincenty_distance(coord1, coord2)? / 1000.0;
        let difference = (vincenty_km - haversine_km).abs();

        println!("{} to {}:", city1, city2);
        println!("  Haversine: {:.1} km", haversine_km);
        println!("  Vincenty:  {:.1} km", vincenty_km);
        println!(
            "  Difference: {:.1} km ({:.2}%)",
            difference,
            difference / vincenty_km * 100.0
        );
    }

    Ok(())
}

fn navigation_example() -> Result<(), Box<dyn std::error::Error>> {
    let departure = (40.6413, -73.7781); // JFK Airport, New York
    let destination = (51.4700, -0.4543); // Heathrow Airport, London

    println!("Flight from JFK (New York) to Heathrow (London):");
    println!("Departure: ({:.4}°, {:.4}°)", departure.0, departure.1);
    println!(
        "Destination: ({:.4}°, {:.4}°)",
        destination.0, destination.1
    );

    // Calculate distance and bearings
    let distance_m = haversine_distance(departure, destination);
    let distance_km = distance_m / 1000.0;
    let distance_nm = distance_km * 0.539957; // Convert to nautical miles

    let initial_bearing_rad = initial_bearing(departure, destination);
    let initial_bearing_deg = initial_bearing_rad * 180.0 / PI;

    let final_bearing_rad = final_bearing(departure, destination);
    let final_bearing_deg = final_bearing_rad * 180.0 / PI;

    println!(
        "Distance: {:.1} km ({:.1} nautical miles)",
        distance_km, distance_nm
    );
    println!(
        "Initial bearing: {:.1}°",
        normalize_bearing(initial_bearing_deg)
    );
    println!(
        "Final bearing: {:.1}°",
        normalize_bearing(final_bearing_deg)
    );

    // Calculate waypoints along the great circle route
    println!("\nWaypoints along the great circle route:");
    let num_waypoints = 5;
    for i in 0..=num_waypoints {
        let fraction = i as f64 / num_waypoints as f64;
        let waypoint_distance = distance_m * fraction;
        let waypoint = destination_point(departure, waypoint_distance, initial_bearing_rad);

        println!(
            "  {:.0}%: ({:.4}°, {:.4}°) - {:.0} km from departure",
            fraction * 100.0,
            waypoint.0,
            waypoint.1,
            waypoint_distance / 1000.0
        );
    }

    // Calculate midpoint
    let mid = midpoint(departure, destination);
    let mid_distance_from_departure = haversine_distance(departure, mid) / 1000.0;
    let mid_distance_from_destination = haversine_distance(destination, mid) / 1000.0;

    println!("\nMidpoint: ({:.4}°, {:.4}°)", mid.0, mid.1);
    println!(
        "Distance from departure: {:.1} km",
        mid_distance_from_departure
    );
    println!(
        "Distance from destination: {:.1} km",
        mid_distance_from_destination
    );

    Ok(())
}

fn coordinate_transformations_example() -> Result<(), Box<dyn std::error::Error>> {
    let locations = vec![
        ("Greenwich Observatory", (51.4769, 0.0)),
        ("Equator/Prime Meridian", (0.0, 0.0)),
        ("North Pole", (90.0, 0.0)),
        ("Sydney Opera House", (-33.8568, 151.2153)),
        ("Statue of Liberty", (40.6892, -74.0445)),
        ("Eiffel Tower", (48.8584, 2.2945)),
    ];

    println!("Coordinate transformations for various locations:");
    println!(
        "{:>20} {:>12} {:>12} {:>15} {:>15} {:>6} {:>3}",
        "Location", "Latitude", "Longitude", "UTM Easting", "UTM Northing", "Zone", "Let"
    );

    for (name, (lat, lon)) in &locations {
        // UTM conversion (skip poles)
        let lat_val: f64 = *lat;
        if lat_val.abs() < 84.0 {
            let utm_result = geographic_to_utm(*lat, *lon);
            match utm_result {
                Ok((easting, northing, zone, letter)) => {
                    println!(
                        "{:>20} {:>12.4} {:>12.4} {:>15.0} {:>15.0} {:>6} {:>3}",
                        name, lat, lon, easting, northing, zone, letter
                    );
                }
                Err(_) => {
                    println!(
                        "{:>20} {:>12.4} {:>12.4} {:>15} {:>15} {:>6} {:>3}",
                        name, lat, lon, "Error", "Error", "-", "-"
                    );
                }
            }
        } else {
            println!(
                "{:>20} {:>12.4} {:>12.4} {:>15} {:>15} {:>6} {:>3}",
                name, lat, lon, "N/A (Polar)", "N/A (Polar)", "-", "-"
            );
        }
    }

    println!("\nWeb Mercator transformations:");
    println!(
        "{:>20} {:>15} {:>15}",
        "Location", "Web Merc X", "Web Merc Y"
    );

    for (name, (lat, lon)) in &locations {
        let lat_val: f64 = *lat;
        if lat_val.abs() < 85.05 {
            // Web Mercator limit
            let web_merc_result = geographic_to_web_mercator(*lat, *lon);
            match web_merc_result {
                Ok((x, y)) => {
                    println!("{:>20} {:>15.0} {:>15.0}", name, x, y);

                    // Test round trip
                    let (back_lat, back_lon) = web_mercator_to_geographic(x, y);
                    let lat_diff = (back_lat - lat).abs();
                    let lon_diff = (back_lon - lon).abs();

                    if lat_diff > 1e-6 || lon_diff > 1e-6 {
                        println!(
                            "    Round trip error: Δlat={:.8}, Δlon={:.8}",
                            lat_diff, lon_diff
                        );
                    }
                }
                Err(_) => {
                    println!("{:>20} {:>15} {:>15}", name, "Error", "Error");
                }
            }
        } else {
            println!("{:>20} {:>15} {:>15}", name, "Out of range", "Out of range");
        }
    }

    Ok(())
}

fn flight_path_example() -> Result<(), Box<dyn std::error::Error>> {
    // Simulate an aircraft flying from Los Angeles to London
    let departure = (34.0522, -118.2437); // Los Angeles
    let destination = (51.5074, -0.1278); // London

    // Define a planned route with waypoints
    let planned_route = [
        departure,
        (40.0, -100.0), // Over central US
        (50.0, -60.0),  // Over Atlantic
        (55.0, -20.0),  // Approaching Europe
        destination,
    ];

    println!("Flight path analysis: Los Angeles to London");
    println!("Planned route waypoints:");

    let mut total_distance = 0.0;
    for (i, waypoint) in planned_route.iter().enumerate() {
        if i > 0 {
            let segment_distance = haversine_distance(planned_route[i - 1], *waypoint);
            total_distance += segment_distance;
            let bearing = initial_bearing(planned_route[i - 1], *waypoint);

            println!(
                "  Waypoint {}: ({:.2}°, {:.2}°) - {:.0} km, bearing {:.0}°",
                i,
                waypoint.0,
                waypoint.1,
                segment_distance / 1000.0,
                normalize_bearing(bearing * 180.0 / PI)
            );
        } else {
            println!(
                "  Waypoint {}: ({:.2}°, {:.2}°) - Departure",
                i, waypoint.0, waypoint.1
            );
        }
    }

    println!("Total planned distance: {:.0} km", total_distance / 1000.0);

    // Compare with great circle route
    let great_circle_distance = haversine_distance(departure, destination);
    let route_efficiency = great_circle_distance / total_distance * 100.0;

    println!(
        "Great circle distance: {:.0} km",
        great_circle_distance / 1000.0
    );
    println!("Route efficiency: {:.1}%", route_efficiency);

    // Simulate aircraft deviating from planned route
    println!("\nSimulating aircraft tracking:");
    let current_position = (45.0, -75.0); // Somewhere over the Atlantic

    // Calculate cross-track and along-track distances
    let cross_track = cross_track_distance(current_position, departure, destination);
    let along_track = along_track_distance(current_position, departure, destination);

    println!(
        "Current position: ({:.2}°, {:.2}°)",
        current_position.0, current_position.1
    );
    println!("Cross-track distance: {:.1} km", cross_track / 1000.0);
    println!(
        "Along-track distance: {:.0} km from departure",
        along_track / 1000.0
    );

    let remaining_distance = great_circle_distance - along_track;
    println!(
        "Estimated remaining distance: {:.0} km",
        remaining_distance / 1000.0
    );

    Ok(())
}

fn spherical_geometry_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a polygon representing the Bermuda Triangle
    let bermuda_triangle = vec![
        (25.7617, -80.1918), // Miami
        (32.3078, -64.7505), // Bermuda
        (18.2208, -66.5901), // San Juan, Puerto Rico
    ];

    println!("Bermuda Triangle analysis:");
    println!("Vertices:");
    for (i, (lat, lon)) in bermuda_triangle.iter().enumerate() {
        let name = match i {
            0 => "Miami",
            1 => "Bermuda",
            2 => "San Juan",
            _ => "Unknown",
        };
        println!("  {}: ({:.4}°, {:.4}°)", name, lat, lon);
    }

    // Calculate area
    let area_sq_m = spherical_polygon_area(&bermuda_triangle)?;
    let area_sq_km = area_sq_m / 1_000_000.0;

    println!("Area: {:.0} km²", area_sq_km);

    // Calculate perimeter
    let mut perimeter = 0.0;
    for i in 0..bermuda_triangle.len() {
        let j = (i + 1) % bermuda_triangle.len();
        perimeter += haversine_distance(bermuda_triangle[i], bermuda_triangle[j]);
    }

    println!("Perimeter: {:.0} km", perimeter / 1000.0);

    // Test points inside and outside the triangle
    let test_points = vec![
        (25.0, -75.0, "Center of triangle"),
        (30.0, -70.0, "Inside triangle"),
        (20.0, -60.0, "Outside triangle (east)"),
        (35.0, -80.0, "Outside triangle (north)"),
        (15.0, -75.0, "Outside triangle (south)"),
    ];

    println!("\nPoint-in-polygon tests:");
    for (lat, lon, description) in test_points {
        let inside = point_in_spherical_polygon((lat, lon), &bermuda_triangle);
        println!(
            "  ({:.1}°, {:.1}°) - {}: {}",
            lat,
            lon,
            description,
            if inside { "Inside" } else { "Outside" }
        );
    }

    // Create a larger polygon (approximate Caribbean Sea boundary)
    let caribbean = vec![
        (10.0, -85.0), // Costa Rica
        (20.0, -88.0), // Yucatan
        (25.0, -85.0), // Florida
        (25.0, -75.0), // Bahamas
        (18.0, -65.0), // Puerto Rico
        (12.0, -62.0), // Trinidad
        (10.0, -65.0), // Venezuela
        (8.0, -75.0),  // Colombia
        (8.0, -82.0),  // Panama
    ];

    let caribbean_area = spherical_polygon_area(&caribbean)? / 1_000_000.0;
    println!(
        "\nCaribbean Sea (approximate area): {:.0} km²",
        caribbean_area
    );

    Ok(())
}

fn gps_tracking_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("GPS tracking simulation: Hiking trail");

    // Simulate GPS coordinates along a hiking trail
    let trail_points = [
        (37.8651, -119.5383), // Yosemite Valley start
        (37.8701, -119.5234), // Moving northeast
        (37.8781, -119.5145), // Continuing
        (37.8834, -119.5089), // Elevation gain
        (37.8889, -119.5012), // Near peak
        (37.8923, -119.4967), // Peak
        (37.8889, -119.5012), // Return
        (37.8834, -119.5089), // Descent
        (37.8781, -119.5145), // Continuing descent
        (37.8651, -119.5383), // Back to start
    ];

    println!("Trail GPS coordinates:");
    let mut total_distance = 0.0;
    let elevations = [
        1200.0, 1250.0, 1320.0, 1400.0, 1520.0, 1650.0, 1520.0, 1400.0, 1320.0, 1200.0,
    ];

    for (i, ((lat, lon), elevation)) in trail_points.iter().zip(elevations.iter()).enumerate() {
        let time = format!("{}:{:02}", 9 + i / 2, (i % 2) * 30); // Simulate time every 30 minutes

        if i > 0 {
            let segment_distance = haversine_distance(trail_points[i - 1], trail_points[i]);
            let elevation_change = elevation - elevations[i - 1];
            total_distance += segment_distance;

            // Calculate speed (assuming 30 minutes between points)
            let speed_kmh = (segment_distance / 1000.0) / 0.5; // 0.5 hours

            println!("  {}: ({:.4}°, {:.4}°) - {:.0}m - {:.1}m segment, {:.1}m elevation change, {:.1} km/h", 
                     time, lat, lon, elevation, segment_distance, elevation_change, speed_kmh);
        } else {
            println!(
                "  {}: ({:.4}°, {:.4}°) - {:.0}m - Start",
                time, lat, lon, elevation
            );
        }
    }

    let total_km = total_distance / 1000.0;
    let total_elevation_gain = elevations
        .iter()
        .zip(elevations.iter().skip(1))
        .map(|(prev, curr)| if curr > prev { curr - prev } else { 0.0 })
        .sum::<f64>();

    println!("\nHike summary:");
    println!("  Total distance: {:.2} km", total_km);
    println!("  Total elevation gain: {:.0} m", total_elevation_gain);
    println!("  Average speed: {:.1} km/h", total_km / 4.5); // 4.5 hours total

    // Calculate some statistics
    let max_elevation = elevations.iter().fold(0.0f64, |acc, &x| acc.max(x));
    let min_elevation = elevations.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));

    println!(
        "  Elevation range: {:.0}m - {:.0}m",
        min_elevation, max_elevation
    );

    Ok(())
}

fn geospatial_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Geospatial analysis: Earthquake monitoring network");

    // Simulate seismic monitoring stations around the San Andreas Fault
    let monitoring_stations = vec![
        ("Berkeley", (37.8719, -122.2585)),
        ("Stanford", (37.4419, -122.1430)),
        ("Parkfield", (35.8997, -120.4339)),
        ("Paso Robles", (35.6266, -120.6909)),
        ("Cholame", (35.7214, -120.3085)),
        ("San Juan Bautista", (36.8455, -121.5374)),
    ];

    // Simulate earthquake epicenter
    let earthquake = (35.7000, -120.4000); // Near Parkfield
    let magnitude = 5.2;

    println!("Monitoring stations:");
    for (name, (lat, lon)) in &monitoring_stations {
        println!("  {}: ({:.4}°, {:.4}°)", name, lat, lon);
    }

    println!("\nEarthquake event:");
    println!("  Epicenter: ({:.4}°, {:.4}°)", earthquake.0, earthquake.1);
    println!("  Magnitude: M{:.1}", magnitude);

    // Calculate distances from each station to earthquake
    println!("\nStation distances from epicenter:");
    let mut station_distances: Vec<_> = monitoring_stations
        .iter()
        .map(|(name, coord)| {
            let distance_km = haversine_distance(*coord, earthquake) / 1000.0;
            (name, distance_km)
        })
        .collect();

    // Sort by distance
    station_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for (name, distance) in &station_distances {
        // Estimate P-wave arrival time (approximately 6 km/s)
        let p_wave_time = distance / 6.0; // seconds

        // Estimate S-wave arrival time (approximately 3.5 km/s)
        let s_wave_time = distance / 3.5; // seconds

        println!(
            "  {}: {:.1} km - P-wave: {:.1}s, S-wave: {:.1}s",
            name, distance, p_wave_time, s_wave_time
        );
    }

    // Find the centroid of the monitoring network
    let centroid_lat = monitoring_stations
        .iter()
        .map(|(_, (lat, _))| lat)
        .sum::<f64>()
        / monitoring_stations.len() as f64;
    let centroid_lon = monitoring_stations
        .iter()
        .map(|(_, (_, lon))| lon)
        .sum::<f64>()
        / monitoring_stations.len() as f64;
    let centroid = (centroid_lat, centroid_lon);

    println!("\nNetwork analysis:");
    println!(
        "  Network centroid: ({:.4}°, {:.4}°)",
        centroid.0, centroid.1
    );

    let distance_to_centroid = haversine_distance(earthquake, centroid) / 1000.0;
    println!(
        "  Distance from earthquake to network centroid: {:.1} km",
        distance_to_centroid
    );

    // Calculate network coverage
    let mut max_station_distance = 0.0f64;
    for i in 0..monitoring_stations.len() {
        for j in (i + 1)..monitoring_stations.len() {
            let distance =
                haversine_distance(monitoring_stations[i].1, monitoring_stations[j].1) / 1000.0;
            max_station_distance = max_station_distance.max(distance);
        }
    }

    println!(
        "  Maximum inter-station distance: {:.1} km",
        max_station_distance
    );

    // Estimate detection threshold
    let closest_station_distance = station_distances[0].1;
    let estimated_magnitude_threshold = 1.0 + 2.0 * (closest_station_distance / 100.0).log10();

    println!(
        "  Estimated detection threshold at closest station: M{:.1}",
        estimated_magnitude_threshold
    );

    Ok(())
}

/// Helper function to format coordinates in degrees, minutes, seconds
#[allow(dead_code)]
fn format_dms(decimal_degrees: f64, is_latitude: bool) -> String {
    let abs_degrees = decimal_degrees.abs();
    let degrees = abs_degrees.floor() as i32;
    let minutes = ((abs_degrees - degrees as f64) * 60.0).floor() as i32;
    let seconds = ((abs_degrees - degrees as f64) * 60.0 - minutes as f64) * 60.0;

    let direction = if is_latitude {
        if decimal_degrees >= 0.0 {
            "N"
        } else {
            "S"
        }
    } else if decimal_degrees >= 0.0 {
        "E"
    } else {
        "W"
    };

    format!("{}°{}'{:.1}\"{}", degrees, minutes, seconds, direction)
}

/// Calculate the centroid of a set of geographic points
#[allow(dead_code)]
fn geographic_centroid(points: &[(f64, f64)]) -> (f64, f64) {
    if points.is_empty() {
        return (0.0, 0.0);
    }

    let mut x_sum = 0.0;
    let mut y_sum = 0.0;
    let mut z_sum = 0.0;

    for (lat, lon) in points {
        let lat_rad = lat * PI / 180.0;
        let lon_rad = lon * PI / 180.0;

        x_sum += lat_rad.cos() * lon_rad.cos();
        y_sum += lat_rad.cos() * lon_rad.sin();
        z_sum += lat_rad.sin();
    }

    let n = points.len() as f64;
    let x_avg = x_sum / n;
    let y_avg = y_sum / n;
    let z_avg = z_sum / n;

    let lon_avg = y_avg.atan2(x_avg);
    let hyp = (x_avg.powi(2) + y_avg.powi(2)).sqrt();
    let lat_avg = z_avg.atan2(hyp);

    (lat_avg * 180.0 / PI, lon_avg * 180.0 / PI)
}
