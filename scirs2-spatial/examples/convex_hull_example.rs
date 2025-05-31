use ndarray::array;
use scirs2_spatial::convex_hull::{convex_hull, ConvexHull};

fn main() {
    println!("=== SciRS2 Spatial - Convex Hull Example ===\n");

    // Create a set of 2D points
    let points_2d = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],   // Interior point
        [0.25, 0.75], // Interior point
        [0.75, 0.25], // Interior point
    ];

    println!("2D Points:");
    for (i, row) in points_2d.outer_iter().enumerate() {
        println!("  Point {}: [{:.2}, {:.2}]", i, row[0], row[1]);
    }
    println!();

    // Method 1: Using the simple function
    println!("Method 1: Using convex_hull() function");
    let hull_vertices = convex_hull(&points_2d.view()).unwrap();

    println!("Hull Vertices:");
    for (i, row) in hull_vertices.outer_iter().enumerate() {
        println!("  Vertex {}: [{:.2}, {:.2}]", i, row[0], row[1]);
    }
    println!();

    // Method 2: Using the ConvexHull class
    println!("Method 2: Using ConvexHull struct");
    let hull = ConvexHull::new(&points_2d.view()).unwrap();

    println!("Hull Vertex Indices: {:?}", hull.vertex_indices());
    println!("Hull Simplices: {:?}", hull.simplices());
    println!();

    // Check if points are inside the hull
    println!("Point Inside Check:");
    let test_points = [
        ([0.5, 0.5], "center"),
        ([0.1, 0.1], "near corner"),
        ([2.0, 2.0], "outside"),
    ];

    for (point, desc) in test_points.iter() {
        let inside = hull.contains(point).unwrap();
        println!(
            "  Point {} at {:?}: {}",
            desc,
            point,
            if inside { "INSIDE" } else { "OUTSIDE" }
        );
    }
    println!();

    // Create a 3D example
    let points_3d = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5], // Interior point
    ];

    println!("3D Example:");
    let hull_3d = ConvexHull::new(&points_3d.view()).unwrap();
    println!("  3D Hull has {} vertices", hull_3d.vertex_indices().len());
    println!("  3D Hull has {} facets", hull_3d.simplices().len());

    // Check if the center point is inside
    let inside_3d = hull_3d.contains([0.5, 0.5, 0.5]).unwrap();
    println!(
        "  Center point [0.5, 0.5, 0.5] is {}",
        if inside_3d { "INSIDE" } else { "OUTSIDE" }
    );
    println!();
}
