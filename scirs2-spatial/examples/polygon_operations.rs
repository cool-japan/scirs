//! Examples of polygon operations
//!
//! This example demonstrates the use of the polygon module from the `scirs2-spatial` crate,
//! showing how to work with polygons including point-in-polygon tests, calculating areas,
//! finding centroids, and creating convex hulls.

#![allow(clippy::needless_range_loop)]

use ndarray::{array, ArrayView2};
use scirs2_spatial::polygon::{
    convex_hull_graham, is_simple_polygon, point_in_polygon, point_on_boundary, polygon_area,
    polygon_centroid, polygon_contains_polygon,
};

/// Visualize a polygon and highlight points inside/outside/on boundary
fn visualize_polygon(polygon: &ArrayView2<f64>, title: &str) {
    let grid_size = 21;
    let mut grid = vec![vec!['.'; grid_size]; grid_size];

    // Scale factor to fit the polygon in the grid
    let min_x = polygon.column(0).fold(f64::INFINITY, |a, &b| a.min(b));
    let max_x = polygon.column(0).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_y = polygon.column(1).fold(f64::INFINITY, |a, &b| a.min(b));
    let max_y = polygon.column(1).fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let width = max_x - min_x;
    let height = max_y - min_y;
    let scale = ((grid_size - 3) as f64) / width.max(height);

    // Function to map real coordinates to grid coordinates
    let to_grid = |x: f64, y: f64| -> (usize, usize) {
        let grid_x = ((x - min_x) * scale + 1.0).round() as usize;
        let grid_y = ((y - min_y) * scale + 1.0).round() as usize;
        (grid_x, grid_y)
    };

    // Draw the polygon vertices with 'V'
    for i in 0..polygon.shape()[0] {
        let (x, y) = to_grid(polygon[[i, 0]], polygon[[i, 1]]);
        grid[y][x] = 'V';
    }

    // Draw the polygon edges with 'E'
    let n = polygon.shape()[0];
    for i in 0..n {
        let j = (i + 1) % n;
        let (x1, y1) = to_grid(polygon[[i, 0]], polygon[[i, 1]]);
        let (x2, y2) = to_grid(polygon[[j, 0]], polygon[[j, 1]]);

        // Draw line between vertices
        let steps = ((x2 as i32 - x1 as i32)
            .abs()
            .max((y2 as i32 - y1 as i32).abs())
            + 1) as usize;
        for k in 0..steps {
            let t = k as f64 / steps as f64;
            let x = (x1 as f64 * (1.0 - t) + x2 as f64 * t).round() as usize;
            let y = (y1 as f64 * (1.0 - t) + y2 as f64 * t).round() as usize;
            if grid[y][x] == '.' {
                grid[y][x] = 'E';
            }
        }
    }

    // Test points in the grid for inside/outside
    for y in 0..grid_size {
        for x in 0..grid_size {
            if grid[y][x] != '.' {
                continue; // Skip points already marked
            }

            // Convert grid coordinates back to real coordinates
            let real_x = (x as f64 - 1.0) / scale + min_x;
            let real_y = (y as f64 - 1.0) / scale + min_y;

            // Mark inside points with 'I', outside with 'O', boundary with 'B'
            if point_on_boundary(&[real_x, real_y], polygon, 0.1 / scale) {
                grid[y][x] = 'B';
            } else if point_in_polygon(&[real_x, real_y], polygon) {
                grid[y][x] = 'I';
            } else {
                grid[y][x] = 'O';
            }
        }
    }

    // Print the visualization
    println!("{}", title);
    println!("{}", "-".repeat(title.len()));

    // Print the grid
    for row in &grid {
        println!("{}", row.iter().collect::<String>());
    }

    println!("V: Vertex, E: Edge, I: Inside, O: Outside, B: Boundary\n");
}

fn main() {
    println!("Polygon Operations Examples");
    println!("==========================");
    println!();

    // Example 1: Point-in-polygon testing
    println!("Example 1: Point-in-Polygon Testing");
    println!("----------------------------------");

    // Create a simple polygon (a square)
    let square = array![[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0],];

    // Test points
    let points = [
        ([3.0, 3.0], "Inside center"),
        ([1.0, 1.0], "On vertex"),
        ([3.0, 1.0], "On edge"),
        ([0.0, 3.0], "Outside left"),
        ([6.0, 3.0], "Outside right"),
        ([3.0, 0.0], "Outside bottom"),
        ([3.0, 6.0], "Outside top"),
    ];

    for (point, desc) in &points {
        let inside = point_in_polygon(point, &square.view());
        let on_boundary = point_on_boundary(point, &square.view(), 1e-10);

        println!("Point {:?} ({}): ", point, desc);
        println!("  Inside: {}", inside);
        println!("  On boundary: {}", on_boundary);
    }

    println!();
    visualize_polygon(&square.view(), "Square Polygon");

    // Example 2: Area and centroid
    println!("Example 2: Area and Centroid");
    println!("--------------------------");

    // Different polygons to test
    let shapes = [
        (
            array![[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0]],
            "Square",
        ),
        (
            array![[1.0, 1.0], [6.0, 1.0], [6.0, 3.0], [1.0, 3.0]],
            "Rectangle",
        ),
        (array![[1.0, 1.0], [5.0, 1.0], [3.0, 5.0]], "Triangle"),
        (
            array![
                [1.0, 1.0],
                [5.0, 1.0],
                [5.0, 3.0],
                [3.0, 3.0],
                [3.0, 5.0],
                [1.0, 5.0]
            ],
            "L-Shape",
        ),
    ];

    for (shape, name) in &shapes {
        let area = polygon_area(&shape.view());
        let centroid = polygon_centroid(&shape.view());

        println!("{} Polygon:", name);
        println!("  Vertices: {}", shape.shape()[0]);
        println!("  Area: {:.2}", area);
        println!("  Centroid: ({:.2}, {:.2})", centroid[0], centroid[1]);
        println!();

        visualize_polygon(&shape.view(), &format!("{} Polygon", name));
    }

    // Example 3: Complex polygons
    println!("Example 3: Complex Polygons");
    println!("-------------------------");

    // Create a complex concave polygon
    let concave = array![
        [2.0, 1.0],
        [5.0, 1.0],
        [5.0, 5.0],
        [3.0, 3.0], // This makes it concave
        [1.0, 5.0],
        [1.0, 3.0],
    ];

    println!("Concave Polygon:");
    println!("  Area: {:.2}", polygon_area(&concave.view()));
    println!(
        "  Is simple (non-self-intersecting): {}",
        is_simple_polygon(&concave.view())
    );
    println!();

    visualize_polygon(&concave.view(), "Concave Polygon");

    // Create a self-intersecting polygon (bow tie)
    let bowtie = array![[1.0, 1.0], [5.0, 5.0], [1.0, 5.0], [5.0, 1.0],];

    println!("Self-intersecting Polygon (Bow Tie):");
    println!("  Area: {:.2}", polygon_area(&bowtie.view()));
    println!(
        "  Is simple (non-self-intersecting): {}",
        is_simple_polygon(&bowtie.view())
    );
    println!();

    visualize_polygon(&bowtie.view(), "Self-intersecting Polygon");

    // Example 4: Polygon contains polygon
    println!("Example 4: Polygon Contains Polygon");
    println!("--------------------------------");

    // Outer and inner polygons
    let outer = array![[1.0, 1.0], [6.0, 1.0], [6.0, 6.0], [1.0, 6.0],];

    let inner = array![[2.0, 2.0], [5.0, 2.0], [5.0, 5.0], [2.0, 5.0],];

    // Polygon that overlaps but doesn't contain
    let overlap = array![[4.0, 4.0], [8.0, 4.0], [8.0, 8.0], [4.0, 8.0],];

    println!(
        "Outer contains Inner: {}",
        polygon_contains_polygon(&outer.view(), &inner.view())
    );
    println!(
        "Inner contains Outer: {}",
        polygon_contains_polygon(&inner.view(), &outer.view())
    );
    println!(
        "Outer contains Overlap: {}",
        polygon_contains_polygon(&outer.view(), &overlap.view())
    );
    println!();

    // Example 5: Convex Hull
    println!("Example 5: Convex Hull");
    println!("--------------------");

    // Create points in various arrangements
    let points = array![
        [3.0, 3.0], // Center point
        [1.0, 1.0], // Corner points
        [5.0, 1.0],
        [5.0, 5.0],
        [1.0, 5.0],
        [2.0, 3.0], // Interior points
        [3.0, 2.0],
        [4.0, 3.0],
        [3.0, 4.0],
    ];

    println!("Original Points: {} points", points.shape()[0]);
    for i in 0..points.shape()[0] {
        println!("  ({:.1}, {:.1})", points[[i, 0]], points[[i, 1]]);
    }
    println!();

    // Compute the convex hull
    let hull = convex_hull_graham(&points.view());

    println!("Convex Hull: {} points", hull.shape()[0]);
    for i in 0..hull.shape()[0] {
        println!("  ({:.1}, {:.1})", hull[[i, 0]], hull[[i, 1]]);
    }
    println!();

    // Visualize the points and hull
    let max_hull_size = 30;
    let mut hull_viz = vec![vec![' '; max_hull_size]; max_hull_size];

    // Find bounds
    let min_x = points.column(0).fold(f64::INFINITY, |a, &b| a.min(b));
    let max_x = points.column(0).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_y = points.column(1).fold(f64::INFINITY, |a, &b| a.min(b));
    let max_y = points.column(1).fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let buffer = 2.0;
    let width = max_x - min_x + buffer * 2.0;
    let height = max_y - min_y + buffer * 2.0;
    let scale = (max_hull_size as f64 - 1.0) / width.max(height);

    // Map coordinates to grid
    let to_grid = |x: f64, y: f64| -> (usize, usize) {
        let grid_x = ((x - min_x + buffer) * scale).round() as usize;
        let grid_y = ((y - min_y + buffer) * scale).round() as usize;
        (grid_x, grid_y)
    };

    // Mark all original points with '.'
    for i in 0..points.shape()[0] {
        let (x, y) = to_grid(points[[i, 0]], points[[i, 1]]);
        hull_viz[y][x] = '.';
    }

    // Mark hull vertices with 'H'
    for i in 0..hull.shape()[0] {
        let (x, y) = to_grid(hull[[i, 0]], hull[[i, 1]]);
        hull_viz[y][x] = 'H';
    }

    // Draw hull edges
    for i in 0..hull.shape()[0] {
        let j = (i + 1) % hull.shape()[0];
        let (x1, y1) = to_grid(hull[[i, 0]], hull[[i, 1]]);
        let (x2, y2) = to_grid(hull[[j, 0]], hull[[j, 1]]);

        // Draw line between hull vertices
        let steps = ((x2 as i32 - x1 as i32)
            .abs()
            .max((y2 as i32 - y1 as i32).abs())
            + 1) as usize;
        for k in 1..steps - 1 {
            // Skip endpoints as they're already marked
            let t = k as f64 / steps as f64;
            let x = (x1 as f64 * (1.0 - t) + x2 as f64 * t).round() as usize;
            let y = (y1 as f64 * (1.0 - t) + y2 as f64 * t).round() as usize;
            if hull_viz[y][x] == ' ' {
                hull_viz[y][x] = '-';
            }
        }
    }

    // Print hull visualization
    println!("Convex Hull Visualization:");
    println!("H: Hull vertex, -: Hull edge, .: Interior point\n");

    for row in &hull_viz {
        println!("{}", row.iter().collect::<String>());
    }

    println!("\nPolygon operations are useful in computational geometry, computer graphics,");
    println!("geographic information systems, and many other applications.");
}
