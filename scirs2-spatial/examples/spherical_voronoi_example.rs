use scirs2_core::ndarray::array;
use scirs2_spatial::spherical_voronoi::SphericalVoronoi;
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Spherical Voronoi Diagram Example");
    println!("=================================\n");

    // Create points at the vertices of an octahedron
    let points = array![
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ];

    println!("Points on the sphere (octahedron vertices):");
    for (i, row) in points.rows().into_iter().enumerate() {
        println!(
            "  Point {}: [{:.2}, {:.2}, {:.2}]",
            i, row[0], row[1], row[2]
        );
    }
    println!();

    // Create a SphericalVoronoi diagram with radius 1.0 centered at the origin
    let radius = 1.0;
    let center = array![0.0, 0.0, 0.0];

    println!("Creating spherical Voronoi diagram with:");
    println!("  Radius: {radius}");
    println!("  Center: [{}, {}, {}]", center[0], center[1], center[2]);
    println!();

    let mut sv = SphericalVoronoi::new(&points.view(), radius, Some(&center), None)?;

    // Sort vertices for visualization purposes
    sv.sort_vertices_of_regions()?;

    // Display information about the Voronoi diagram
    println!("Spherical Voronoi diagram information:");
    println!("  Number of regions: {}", sv.regions().len());
    println!("  Number of vertices: {}", sv.vertices().nrows());
    println!("  Number of Delaunay simplices: {}", sv.simplices().len());
    println!();

    // Show the vertices
    println!("Voronoi vertices on the sphere:");
    for (i, row) in sv.vertices().rows().into_iter().enumerate() {
        println!(
            "  Vertex {}: [{:.6}, {:.6}, {:.6}]",
            i, row[0], row[1], row[2]
        );
    }
    println!();

    // Show the regions
    println!("Voronoi regions (point index -> vertex indices):");
    for (i, region) in sv.regions().iter().enumerate() {
        println!("  Region for point {i}: {region:?}");
    }
    println!();

    // Get regions length first before any other borrow
    let num_regions = sv.regions().len();

    // Calculate region areas
    println!("Calculating region areas...");
    let areas = sv.areas()?;
    let total_area = 4.0 * PI * radius * radius;
    let expected_area_per_region = total_area / (num_regions as f64);

    println!("Region areas on the unit sphere:");
    for (i, &area) in areas.iter().enumerate() {
        println!("  Region {i}: {area:.6} (expected: {expected_area_per_region:.6})");
    }

    println!(
        "\nTotal area of all regions: {:.6}",
        areas.iter().sum::<f64>()
    );
    println!("Expected total area (4πr²): {total_area:.6}");
    println!();

    // Demonstrate geodesic distance calculation
    println!("Geodesic distance calculations:");

    // North pole to south pole (should be π radians)
    let north_pole = array![0.0, 0.0, 1.0];
    let south_pole = array![0.0, 0.0, -1.0];
    let dist = sv.geodesic_distance(&north_pole.view(), &south_pole.view())?;
    println!("  North pole to South pole: {dist:.6} (expected: π = {PI:.6})");

    // North pole to equator (should be π/2 radians)
    let equator_point = array![1.0, 0.0, 0.0];
    let dist = sv.geodesic_distance(&north_pole.view(), &equator_point.view())?;
    println!(
        "  North pole to Equator: {:.6} (expected: π/2 = {:.6})",
        dist,
        PI / 2.0
    );

    // Find nearest generator to a point
    let query_point = array![0.5, 0.5, std::f64::consts::FRAC_1_SQRT_2]; // Approximately normalized
    let (nearest_idx, dist) = sv.nearest_generator(&query_point.view())?;
    println!("\nNearest generator to [0.5, 0.5, std::f64::consts::FRAC_1_SQRT_2]:");
    println!("  Generator index: {nearest_idx}");
    println!("  Distance: {dist:.6}");

    // Calculate distances to all generators
    let distances = sv.geodesic_distances_to_generators(&query_point.view())?;
    println!("\nDistances to all generators:");
    for (i, &dist) in distances.iter().enumerate() {
        println!("  Distance to generator {i}: {dist:.6}");
    }
    println!();

    // Create another example with a more complex point set
    println!("Creating a more complex example with a dodecahedron-based point set...");
    let dodeca_points = generate_dodecahedron_points();

    let mut complex_sv = SphericalVoronoi::new(&dodeca_points.view(), radius, Some(&center), None)?;
    complex_sv.sort_vertices_of_regions()?;

    println!("Dodecahedron Voronoi diagram information:");
    println!("  Number of regions: {}", complex_sv.regions().len());
    println!("  Number of vertices: {}", complex_sv.vertices().nrows());
    println!(
        "  Number of Delaunay simplices: {}",
        complex_sv.simplices().len()
    );

    // Calculate areas of the complex regions
    let complex_areas = complex_sv.areas()?;
    let total_complex_area: f64 = complex_areas.iter().sum();
    println!("  Total area of all regions: {total_complex_area:.6}");
    println!("  Expected total area (4πr²): {total_area:.6}");

    println!("\nVoronoi diagram calculation completed successfully!");

    Ok(())
}

/// Generate vertices of a dodecahedron as generator points on a unit sphere
#[allow(dead_code)]
fn generate_dodecahedron_points() -> scirs2_core::ndarray::Array2<f64> {
    use scirs2_core::ndarray::Array2;

    // The golden ratio, used in dodecahedron construction
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;

    // Coordinates of vertices (before normalization)
    let vertices = vec![
        // (±1, ±1, ±1)
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0],
        // (0, ±1/φ, ±φ)
        [0.0, 1.0 / phi, phi],
        [0.0, 1.0 / phi, -phi],
        [0.0, -1.0 / phi, phi],
        [0.0, -1.0 / phi, -phi],
        // (±1/φ, ±φ, 0)
        [1.0 / phi, phi, 0.0],
        [1.0 / phi, -phi, 0.0],
        [-1.0 / phi, phi, 0.0],
        [-1.0 / phi, -phi, 0.0],
        // (±φ, 0, ±1/φ)
        [phi, 0.0, 1.0 / phi],
        [phi, 0.0, -1.0 / phi],
        [-phi, 0.0, 1.0 / phi],
        [-phi, 0.0, -1.0 / phi],
    ];

    // Create array from vertices
    let mut points = Array2::zeros((vertices.len(), 3));

    // Normalize all vertices to the unit sphere
    for (i, vertex) in vertices.iter().enumerate() {
        let norm = (vertex[0].powi(2) + vertex[1].powi(2) + vertex[2].powi(2)).sqrt();
        points[[i, 0]] = vertex[0] / norm;
        points[[i, 1]] = vertex[1] / norm;
        points[[i, 2]] = vertex[2] / norm;
    }

    points
}

// This function is kept here for future reference but not currently used
#[allow(dead_code)]
/// Generate random points uniformly distributed on a sphere.
fn generate_random_points_on_sphere(
    n: usize,
    radius: f64,
    center: &scirs2_core::ndarray::Array1<f64>,
) -> scirs2_core::ndarray::Array2<f64> {
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::Rng;

    let mut rng = scirs2_core::random::rng();
    let mut points = Array2::zeros((n, 3));

    for i in 0..n {
        // Generate random points according to the uniform distribution on the sphere
        // using the Marsaglia method

        // Generate points in the unit cube between -1 and 1
        let mut valid = false;
        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;

        while !valid {
            x = rng.random_range(-1.0..1.0);
            y = rng.random_range(-1.0..1.0);
            z = rng.random_range(-1.0..1.0);

            let d2: f64 = x * x + y * y + z * z;
            if d2 > 0.0 && d2 <= 1.0 {
                // Normalize to the sphere surface
                let scale = radius / d2.sqrt();
                x *= scale;
                y *= scale;
                z *= scale;
                valid = true;
            }
        }

        // Add the center coordinates to get the final point
        points[[i, 0]] = x + center[0];
        points[[i, 1]] = y + center[1];
        points[[i, 2]] = z + center[2];
    }

    points
}
