use ndarray::array;
use scirs2_spatial::transform::spherical::{
    cart_to_spherical, cart_to_spherical_batch, geodesic_distance, spherical_to_cart,
    spherical_to_cart_batch, spherical_triangle_area,
};
use scirs2_spatial::SpatialResult;
use std::f64::consts::PI;

fn main() -> SpatialResult<()> {
    println!("Spherical Coordinate Transformations Example");
    println!("===========================================\n");

    // Single point conversion examples
    println!("Single Point Conversions");
    println!("-----------------------");

    // Define some Cartesian points
    let points_cart = [
        array![1.0, 0.0, 0.0], // Point on x-axis
        array![0.0, 1.0, 0.0], // Point on y-axis
        array![0.0, 0.0, 1.0], // Point on z-axis
        array![1.0, 1.0, 1.0], // Point in 1st octant
    ];

    let point_names = ["X-axis", "Y-axis", "Z-axis", "Octant (1,1,1)"];

    // Convert and display
    for (i, point) in points_cart.iter().enumerate() {
        println!("\n{}. {} point:", i + 1, point_names[i]);
        println!(
            "   Cartesian coordinates: [{:.4}, {:.4}, {:.4}]",
            point[0], point[1], point[2]
        );

        let spherical = cart_to_spherical(&point.view())?;

        println!("   Spherical coordinates (r, θ, φ):");
        println!("     r     = {:.4}", spherical[0]);
        println!(
            "     θ (radians) = {:.4}, θ (degrees) = {:.2}°",
            spherical[1],
            spherical[1] * 180.0 / PI
        );
        println!(
            "     φ (radians) = {:.4}, φ (degrees) = {:.2}°",
            spherical[2],
            spherical[2] * 180.0 / PI
        );

        // Convert back to verify
        let cart_again = spherical_to_cart(&spherical.view())?;
        println!(
            "   Back to Cartesian: [{:.4}, {:.4}, {:.4}]",
            cart_again[0], cart_again[1], cart_again[2]
        );
    }

    // Batch conversion example
    println!("\n\nBatch Conversion");
    println!("----------------");

    let batch_cart = array![
        [1.0, 0.0, 0.0], // Point on x-axis
        [0.0, 1.0, 0.0], // Point on y-axis
        [0.0, 0.0, 1.0], // Point on z-axis
        [1.0, 1.0, 1.0], // Point in 1st octant
    ];

    println!("Cartesian coordinates (batch):");
    for (i, row) in batch_cart.rows().into_iter().enumerate() {
        println!("  {}: [{:.4}, {:.4}, {:.4}]", i + 1, row[0], row[1], row[2]);
    }

    let batch_spherical = cart_to_spherical_batch(&batch_cart.view())?;

    println!("\nSpherical coordinates (batch):");
    println!("  [r, θ (rad), φ (rad)]");
    for (i, row) in batch_spherical.rows().into_iter().enumerate() {
        println!("  {}: [{:.4}, {:.4}, {:.4}]", i + 1, row[0], row[1], row[2]);
    }

    let batch_cart_again = spherical_to_cart_batch(&batch_spherical.view())?;

    println!("\nBack to Cartesian (batch):");
    for (i, row) in batch_cart_again.rows().into_iter().enumerate() {
        println!("  {}: [{:.4}, {:.4}, {:.4}]", i + 1, row[0], row[1], row[2]);
    }

    // Geodesic distance examples
    println!("\n\nGeodesic Distance on a Sphere");
    println!("---------------------------");

    // Create spherical coordinates for some points on a unit sphere
    let north_pole = array![1.0, 0.0, 0.0]; // North pole (θ=0)
    let south_pole = array![1.0, PI, 0.0]; // South pole (θ=π)
    let equator_0 = array![1.0, PI / 2.0, 0.0]; // Point on equator at φ=0
    let equator_90 = array![1.0, PI / 2.0, PI / 2.0]; // Point on equator at φ=π/2

    println!("Points on a unit sphere:");
    println!("  1. North pole: (r=1, θ=0°, φ=0°)");
    println!("  2. South pole: (r=1, θ=180°, φ=0°)");
    println!("  3. Equator at φ=0°: (r=1, θ=90°, φ=0°)");
    println!("  4. Equator at φ=90°: (r=1, θ=90°, φ=90°)");

    println!("\nGeodetic distances:");

    let distance = geodesic_distance(&north_pole.view(), &south_pole.view())?;
    println!(
        "  North pole to South pole: {:.4} radians = {:.2}°",
        distance,
        distance * 180.0 / PI
    );

    let distance = geodesic_distance(&north_pole.view(), &equator_0.view())?;
    println!(
        "  North pole to Equator: {:.4} radians = {:.2}°",
        distance,
        distance * 180.0 / PI
    );

    let distance = geodesic_distance(&equator_0.view(), &equator_90.view())?;
    println!(
        "  Between points 90° apart on equator: {:.4} radians = {:.2}°",
        distance,
        distance * 180.0 / PI
    );

    // Spherical triangle area
    println!("\n\nSpherical Triangle Area");
    println!("----------------------");

    // North pole and two points on the equator 90° apart
    let area = spherical_triangle_area(&north_pole.view(), &equator_0.view(), &equator_90.view())?;

    println!("Triangle: North pole and two points 90° apart on the equator");
    println!(
        "  Area: {:.4} steradians (out of 4π = {:.4} for the whole sphere)",
        area,
        4.0 * PI
    );
    println!(
        "  This is 1/8 of the sphere's surface, or {:.2}%",
        100.0 * area / (4.0 * PI)
    );

    // Another example: 1/4 of a hemisphere
    let p1 = array![1.0, 0.0, 0.0]; // North pole
    let p2 = array![1.0, PI / 2.0, 0.0]; // Equator at φ=0
    let p3 = array![1.0, PI / 2.0, PI / 2.0]; // Equator at φ=π/2

    let area = spherical_triangle_area(&p1.view(), &p2.view(), &p3.view())?;
    println!("\nTriangle: North pole and two points on the equator 90° apart");
    println!("  Area: {:.4} steradians", area);

    // Error cases
    println!("\n\nError Handling Examples");
    println!("---------------------");

    // Try to calculate geodesic distance between points on spheres with different radii
    let p1 = array![1.0, 0.0, 0.0]; // Point on unit sphere
    let p2 = array![2.0, 0.0, 0.0]; // Point on sphere with radius 2

    println!("Points on different spheres:");
    println!("  p1: (r=1, θ=0°, φ=0°)");
    println!("  p2: (r=2, θ=0°, φ=0°)");

    match geodesic_distance(&p1.view(), &p2.view()) {
        Ok(distance) => println!("  Distance: {:.4}", distance),
        Err(e) => println!("  Error: {}", e),
    }

    // Try with invalid spherical coordinates
    let invalid = array![1.0, -0.1, 0.0]; // θ should be in [0, π]

    println!("\nInvalid spherical coordinates:");
    println!("  (r=1, θ=-0.1, φ=0°)");

    match spherical_to_cart(&invalid.view()) {
        Ok(cart) => println!("  Converted to: {:?}", cart),
        Err(e) => println!("  Error: {}", e),
    }

    Ok(())
}
