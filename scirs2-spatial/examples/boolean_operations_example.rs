//! Boolean Operations Example
//!
//! This example demonstrates Boolean operations on polygons including union,
//! intersection, difference, and symmetric difference. These operations are
//! fundamental in computational geometry and are used in CAD systems, GIS
//! applications, and computer graphics.

use ndarray::{array, Array2};
use scirs2_spatial::boolean_ops::{
    compute_polygon_area, is_convex_polygon, is_self_intersecting, polygon_difference,
    polygon_intersection, polygon_symmetric_difference, polygon_union,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Boolean Operations Example ===\n");

    // Example 1: Two overlapping squares
    println!("1. Overlapping Squares");
    overlapping_squares_example()?;
    println!();

    // Example 2: Non-overlapping polygons
    println!("2. Non-overlapping Polygons");
    non_overlapping_example()?;
    println!();

    // Example 3: Complex polygon operations
    println!("3. Complex Polygon Operations");
    complex_polygon_example()?;
    println!();

    // Example 4: Polygon property checking
    println!("4. Polygon Property Checking");
    polygon_properties_example()?;
    println!();

    // Example 5: Triangle operations
    println!("5. Triangle Operations");
    triangle_operations_example()?;
    println!();

    // Example 6: L-shaped polygon operations
    println!("6. L-shaped Polygon Operations");
    l_shape_operations_example()?;
    println!();

    // Example 7: Self-intersection detection
    println!("7. Self-intersection Detection");
    self_intersection_example()?;

    Ok(())
}

fn overlapping_squares_example() -> Result<(), Box<dyn std::error::Error>> {
    // Define two overlapping squares
    let poly1 = array![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];

    let poly2 = array![[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];

    println!("First square: [0,0] to [2,2]");
    println!("Second square: [1,1] to [3,3]");

    let area1 = compute_polygon_area(&poly1.view())?;
    let area2 = compute_polygon_area(&poly2.view())?;
    println!("Areas: {:.1}, {:.1}", area1, area2);

    // Union
    let union_result = polygon_union(&poly1.view(), &poly2.view())?;
    let union_area = compute_polygon_area(&union_result.view())?;
    println!("Union:");
    println!("  Vertices: {}", union_result.nrows());
    println!("  Area: {:.3}", union_area);
    print_vertices(&union_result);

    // Intersection
    let intersection_result = polygon_intersection(&poly1.view(), &poly2.view())?;
    println!("Intersection:");
    println!("  Vertices: {}", intersection_result.nrows());
    if intersection_result.nrows() > 0 {
        let intersection_area = compute_polygon_area(&intersection_result.view())?;
        println!("  Area: {:.3}", intersection_area);
        print_vertices(&intersection_result);
    } else {
        println!("  No intersection");
    }

    // Difference (poly1 - poly2)
    let difference_result = polygon_difference(&poly1.view(), &poly2.view())?;
    println!("Difference (poly1 - poly2):");
    println!("  Vertices: {}", difference_result.nrows());
    if difference_result.nrows() > 0 {
        let difference_area = compute_polygon_area(&difference_result.view())?;
        println!("  Area: {:.3}", difference_area);
        print_vertices(&difference_result);
    }

    // Symmetric difference
    let sym_diff_result = polygon_symmetric_difference(&poly1.view(), &poly2.view())?;
    println!("Symmetric difference:");
    println!("  Vertices: {}", sym_diff_result.nrows());
    if sym_diff_result.nrows() > 0 {
        let sym_diff_area = compute_polygon_area(&sym_diff_result.view())?;
        println!("  Area: {:.3}", sym_diff_area);
    }

    Ok(())
}

fn non_overlapping_example() -> Result<(), Box<dyn std::error::Error>> {
    // Two non-overlapping squares
    let poly1 = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    let poly2 = array![[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]];

    println!("Two non-overlapping unit squares");

    let area1 = compute_polygon_area(&poly1.view())?;
    let area2 = compute_polygon_area(&poly2.view())?;
    println!("Individual areas: {:.1}, {:.1}", area1, area2);

    // Union should have both polygons
    let union_result = polygon_union(&poly1.view(), &poly2.view())?;
    println!("Union result:");
    println!("  Vertices: {}", union_result.nrows());
    let union_area = compute_polygon_area(&union_result.view())?;
    println!(
        "  Area: {:.3} (should be close to {:.1})",
        union_area,
        area1 + area2
    );

    // Intersection should be empty
    let intersection_result = polygon_intersection(&poly1.view(), &poly2.view())?;
    println!("Intersection result:");
    println!("  Vertices: {}", intersection_result.nrows());
    if intersection_result.nrows() > 0 {
        let intersection_area = compute_polygon_area(&intersection_result.view())?;
        println!("  Area: {:.6}", intersection_area);
    } else {
        println!("  Empty (as expected)");
    }

    Ok(())
}

fn complex_polygon_example() -> Result<(), Box<dyn std::error::Error>> {
    // Hexagon and triangle
    let hexagon = regular_hexagon(2.0, 0.0, 0.0);
    let triangle = array![[-1.0, -1.0], [1.0, -1.0], [0.0, 2.0]];

    println!("Regular hexagon and triangle");

    let hex_area = compute_polygon_area(&hexagon.view())?;
    let tri_area = compute_polygon_area(&triangle.view())?;
    println!(
        "Areas: hexagon = {:.3}, triangle = {:.3}",
        hex_area, tri_area
    );

    // Check if polygons are convex
    let hex_convex = is_convex_polygon(&hexagon.view())?;
    let tri_convex = is_convex_polygon(&triangle.view())?;
    println!(
        "Convex: hexagon = {}, triangle = {}",
        hex_convex, tri_convex
    );

    // Union
    let union_result = polygon_union(&hexagon.view(), &triangle.view())?;
    println!("Union:");
    println!("  Vertices: {}", union_result.nrows());
    let union_area = compute_polygon_area(&union_result.view())?;
    println!("  Area: {:.3}", union_area);

    // Intersection
    let intersection_result = polygon_intersection(&hexagon.view(), &triangle.view())?;
    println!("Intersection:");
    println!("  Vertices: {}", intersection_result.nrows());
    if intersection_result.nrows() > 0 {
        let intersection_area = compute_polygon_area(&intersection_result.view())?;
        println!("  Area: {:.3}", intersection_area);
    }

    Ok(())
}

fn polygon_properties_example() -> Result<(), Box<dyn std::error::Error>> {
    // Test different polygon types
    let polygons = vec![
        (
            "Square",
            array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ),
        ("Triangle", array![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
        (
            "L-shape",
            array![
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [0.0, 2.0]
            ],
        ),
        ("Star", create_star_polygon()),
    ];

    for (name, poly) in polygons {
        println!("{}:", name);

        let area = compute_polygon_area(&poly.view())?;
        let is_convex = is_convex_polygon(&poly.view())?;
        let is_self_intersecting = is_self_intersecting(&poly.view())?;

        println!("  Area: {:.3}", area);
        println!("  Convex: {}", is_convex);
        println!("  Self-intersecting: {}", is_self_intersecting);
        println!("  Vertices: {}", poly.nrows());
        println!();
    }

    Ok(())
}

fn triangle_operations_example() -> Result<(), Box<dyn std::error::Error>> {
    let triangle1 = array![[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]];

    let triangle2 = array![[1.0, 0.0], [3.0, 0.0], [2.0, 2.0]];

    println!("Two overlapping triangles");

    let area1 = compute_polygon_area(&triangle1.view())?;
    let area2 = compute_polygon_area(&triangle2.view())?;
    println!("Areas: {:.3}, {:.3}", area1, area2);

    // Union
    let union_result = polygon_union(&triangle1.view(), &triangle2.view())?;
    println!("Union:");
    println!("  Vertices: {}", union_result.nrows());
    let union_area = compute_polygon_area(&union_result.view())?;
    println!("  Area: {:.3}", union_area);

    // Intersection
    let intersection_result = polygon_intersection(&triangle1.view(), &triangle2.view())?;
    println!("Intersection:");
    println!("  Vertices: {}", intersection_result.nrows());
    if intersection_result.nrows() > 0 {
        let intersection_area = compute_polygon_area(&intersection_result.view())?;
        println!("  Area: {:.3}", intersection_area);
    }

    Ok(())
}

fn l_shape_operations_example() -> Result<(), Box<dyn std::error::Error>> {
    // L-shaped polygon
    let l_shape = array![
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [0.0, 2.0]
    ];

    // Square
    let square = array![[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]];

    println!("L-shaped polygon and square");

    let l_area = compute_polygon_area(&l_shape.view())?;
    let sq_area = compute_polygon_area(&square.view())?;
    println!("Areas: L-shape = {:.3}, square = {:.3}", l_area, sq_area);

    let l_convex = is_convex_polygon(&l_shape.view())?;
    let sq_convex = is_convex_polygon(&square.view())?;
    println!("Convex: L-shape = {}, square = {}", l_convex, sq_convex);

    // Union
    let union_result = polygon_union(&l_shape.view(), &square.view())?;
    println!("Union:");
    println!("  Vertices: {}", union_result.nrows());
    let union_area = compute_polygon_area(&union_result.view())?;
    println!("  Area: {:.3}", union_area);

    // Intersection
    let intersection_result = polygon_intersection(&l_shape.view(), &square.view())?;
    println!("Intersection:");
    println!("  Vertices: {}", intersection_result.nrows());
    if intersection_result.nrows() > 0 {
        let intersection_area = compute_polygon_area(&intersection_result.view())?;
        println!("  Area: {:.3}", intersection_area);
        print_vertices(&intersection_result);
    }

    Ok(())
}

fn self_intersection_example() -> Result<(), Box<dyn std::error::Error>> {
    // Simple polygon
    let simple = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    // Self-intersecting polygon (bowtie)
    let bowtie = array![[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]];

    // Complex self-intersecting shape
    let complex_self_intersecting =
        array![[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]];

    let shapes = vec![
        ("Simple square", simple),
        ("Bowtie", bowtie),
        ("Complex self-intersecting", complex_self_intersecting),
    ];

    for (name, shape) in shapes {
        let is_self_intersecting = is_self_intersecting(&shape.view())?;
        let is_convex = is_convex_polygon(&shape.view())?;

        println!("{}:", name);
        println!("  Self-intersecting: {}", is_self_intersecting);
        println!("  Convex: {}", is_convex);

        if !is_self_intersecting {
            let area = compute_polygon_area(&shape.view())?;
            println!("  Area: {:.3}", area);
        } else {
            println!("  Area: (undefined for self-intersecting polygon)");
        }
        println!();
    }

    Ok(())
}

/// Create a regular hexagon with given radius and center
fn regular_hexagon(radius: f64, center_x: f64, center_y: f64) -> Array2<f64> {
    let mut vertices = Vec::with_capacity(12); // 6 vertices * 2 coordinates

    for i in 0..6 {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / 6.0;
        let x = center_x + radius * angle.cos();
        let y = center_y + radius * angle.sin();
        vertices.push(x);
        vertices.push(y);
    }

    Array2::from_shape_vec((6, 2), vertices).unwrap()
}

/// Create a simple star polygon (5-pointed)
fn create_star_polygon() -> Array2<f64> {
    let outer_radius = 1.0;
    let inner_radius = 0.4;
    let mut vertices = Vec::with_capacity(20); // 10 vertices * 2 coordinates

    for i in 0..10 {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / 10.0;
        let radius = if i % 2 == 0 {
            outer_radius
        } else {
            inner_radius
        };
        let x = radius * angle.cos();
        let y = radius * angle.sin();
        vertices.push(x);
        vertices.push(y);
    }

    Array2::from_shape_vec((10, 2), vertices).unwrap()
}

/// Print polygon vertices (limited to first few vertices for readability)
fn print_vertices(poly: &Array2<f64>) {
    let max_vertices = 6; // Limit output for readability
    let n = poly.nrows().min(max_vertices);

    print!("  Vertices:");
    for i in 0..n {
        print!(" [{:.2}, {:.2}]", poly[[i, 0]], poly[[i, 1]]);
    }
    if poly.nrows() > max_vertices {
        print!(" ... ({} more)", poly.nrows() - max_vertices);
    }
    println!();
}
