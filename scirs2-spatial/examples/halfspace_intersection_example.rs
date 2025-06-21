//! Halfspace Intersection Example
//!
//! This example demonstrates the use of halfspace intersection to construct
//! convex polytopes from linear constraints. Halfspace intersection is the
//! dual problem to convex hull computation and is useful in optimization,
//! computational geometry, and constraint satisfaction.

use ndarray::arr1;
use scirs2_spatial::halfspace::{Halfspace, HalfspaceIntersection};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Halfspace Intersection Example ===\n");

    // Example 1: Unit square
    println!("1. Unit Square from Halfspaces");
    unit_square_example()?;
    println!();

    // Example 2: Triangle
    println!("2. Triangle from Halfspaces");
    triangle_example()?;
    println!();

    // Example 3: Regular polygon
    println!("3. Regular Hexagon");
    regular_polygon_example()?;
    println!();

    // Example 4: 3D cube
    println!("4. 3D Unit Cube");
    cube_3d_example()?;
    println!();

    // Example 5: Unbounded region
    println!("5. Unbounded Region");
    unbounded_example()?;
    println!();

    // Example 6: Empty intersection
    println!("6. Empty Intersection");
    empty_intersection_example()?;
    println!();

    // Example 7: Custom polytope with interior point
    println!("7. Custom Polytope with Interior Point");
    custom_polytope_example()?;

    Ok(())
}

fn unit_square_example() -> Result<(), Box<dyn std::error::Error>> {
    // Define halfspaces for unit square: 0 ≤ x ≤ 1, 0 ≤ y ≤ 1
    let halfspaces = vec![
        Halfspace::new(arr1(&[-1.0, 0.0]), 0.0), // -x ≤ 0  =>  x ≥ 0
        Halfspace::new(arr1(&[0.0, -1.0]), 0.0), // -y ≤ 0  =>  y ≥ 0
        Halfspace::new(arr1(&[1.0, 0.0]), 1.0),  //  x ≤ 1
        Halfspace::new(arr1(&[0.0, 1.0]), 1.0),  //  y ≤ 1
    ];

    println!("Halfspaces for unit square:");
    for (i, hs) in halfspaces.iter().enumerate() {
        println!("  {}: {:?} · x ≤ {:.1}", i + 1, hs.normal(), hs.offset());
    }

    let intersection = HalfspaceIntersection::new(&halfspaces, None)?;

    println!("Results:");
    println!("  Vertices: {}", intersection.num_vertices());
    println!("  Faces: {}", intersection.num_faces());
    println!("  Is bounded: {}", intersection.is_bounded());
    println!("  Is feasible: {}", intersection.is_feasible());

    if intersection.is_feasible() {
        let area = intersection.volume()?;
        println!("  Area: {:.3}", area);

        println!("  Vertex coordinates:");
        for (i, vertex) in intersection.vertices().outer_iter().enumerate() {
            println!("    Vertex {}: [{:.3}, {:.3}]", i, vertex[0], vertex[1]);
        }
    }

    Ok(())
}

fn triangle_example() -> Result<(), Box<dyn std::error::Error>> {
    // Define halfspaces for triangle: x ≥ 0, y ≥ 0, x + y ≤ 1
    let halfspaces = vec![
        Halfspace::new(arr1(&[-1.0, 0.0]), 0.0), // x ≥ 0
        Halfspace::new(arr1(&[0.0, -1.0]), 0.0), // y ≥ 0
        Halfspace::new(arr1(&[1.0, 1.0]), 1.0),  // x + y ≤ 1
    ];

    println!("Halfspaces for triangle:");
    for (i, hs) in halfspaces.iter().enumerate() {
        println!("  {}: {:?} · x ≤ {:.1}", i + 1, hs.normal(), hs.offset());
    }

    let intersection = HalfspaceIntersection::new(&halfspaces, None)?;

    println!("Results:");
    println!("  Vertices: {}", intersection.num_vertices());
    println!("  Is bounded: {}", intersection.is_bounded());

    if intersection.is_feasible() {
        let area = intersection.volume()?;
        println!("  Area: {:.3}", area);

        println!("  Vertex coordinates:");
        for (i, vertex) in intersection.vertices().outer_iter().enumerate() {
            println!("    Vertex {}: [{:.3}, {:.3}]", i, vertex[0], vertex[1]);
        }
    }

    Ok(())
}

fn regular_polygon_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a regular hexagon using halfspaces
    let n_sides = 6;
    let radius = 2.0;
    let mut halfspaces = Vec::new();

    println!(
        "Regular hexagon with {} sides and radius {:.1}:",
        n_sides, radius
    );

    for i in 0..n_sides {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_sides as f64);

        // Normal vector pointing inward
        let normal_x = angle.cos();
        let normal_y = angle.sin();

        // Distance from origin to edge
        let offset = radius;

        let halfspace = Halfspace::new(arr1(&[normal_x, normal_y]), offset);
        println!(
            "  Side {}: [{:.3}, {:.3}] · x ≤ {:.3}",
            i + 1,
            normal_x,
            normal_y,
            offset
        );

        halfspaces.push(halfspace);
    }

    let intersection = HalfspaceIntersection::new(&halfspaces, None)?;

    println!("Results:");
    println!("  Vertices: {}", intersection.num_vertices());
    println!("  Is bounded: {}", intersection.is_bounded());

    if intersection.is_feasible() {
        let area = intersection.volume()?;
        println!("  Area: {:.3}", area);

        // Theoretical area of regular hexagon: 3*sqrt(3)/2 * radius^2
        let theoretical_area = 3.0 * 3.0_f64.sqrt() / 2.0 * radius * radius;
        println!("  Theoretical area: {:.3}", theoretical_area);

        if intersection.num_vertices() <= 8 {
            println!("  Vertex coordinates:");
            for (i, vertex) in intersection.vertices().outer_iter().enumerate() {
                println!("    Vertex {}: [{:.3}, {:.3}]", i, vertex[0], vertex[1]);
            }
        }
    }

    Ok(())
}

fn cube_3d_example() -> Result<(), Box<dyn std::error::Error>> {
    // Define halfspaces for 3D unit cube: 0 ≤ x ≤ 1, 0 ≤ y ≤ 1, 0 ≤ z ≤ 1
    let halfspaces = vec![
        Halfspace::new(arr1(&[-1.0, 0.0, 0.0]), 0.0), // x ≥ 0
        Halfspace::new(arr1(&[0.0, -1.0, 0.0]), 0.0), // y ≥ 0
        Halfspace::new(arr1(&[0.0, 0.0, -1.0]), 0.0), // z ≥ 0
        Halfspace::new(arr1(&[1.0, 0.0, 0.0]), 1.0),  // x ≤ 1
        Halfspace::new(arr1(&[0.0, 1.0, 0.0]), 1.0),  // y ≤ 1
        Halfspace::new(arr1(&[0.0, 0.0, 1.0]), 1.0),  // z ≤ 1
    ];

    println!("Halfspaces for 3D unit cube:");
    let constraints = ["x ≥ 0", "y ≥ 0", "z ≥ 0", "x ≤ 1", "y ≤ 1", "z ≤ 1"];
    for (i, constraint) in constraints.iter().enumerate() {
        println!("  {}: {}", i + 1, constraint);
    }

    let intersection = HalfspaceIntersection::new(&halfspaces, None)?;

    println!("Results:");
    println!("  Vertices: {}", intersection.num_vertices());
    println!("  Faces: {}", intersection.num_faces());
    println!("  Is bounded: {}", intersection.is_bounded());
    println!("  Is feasible: {}", intersection.is_feasible());

    if intersection.is_feasible() {
        if intersection.dim() == 3 && intersection.is_bounded() {
            let volume = intersection.volume()?;
            println!("  Volume: {:.3}", volume);
        }

        if intersection.num_vertices() <= 12 {
            println!("  Vertex coordinates:");
            for (i, vertex) in intersection.vertices().outer_iter().enumerate() {
                println!(
                    "    Vertex {}: [{:.3}, {:.3}, {:.3}]",
                    i, vertex[0], vertex[1], vertex[2]
                );
            }
        } else {
            println!("  (Too many vertices to display)");
        }
    }

    Ok(())
}

fn unbounded_example() -> Result<(), Box<dyn std::error::Error>> {
    // Define halfspaces that form an unbounded region: x ≥ 0, y ≥ 0
    let halfspaces = vec![
        Halfspace::new(arr1(&[-1.0, 0.0]), 0.0), // x ≥ 0
        Halfspace::new(arr1(&[0.0, -1.0]), 0.0), // y ≥ 0
    ];

    println!("Halfspaces for unbounded region (first quadrant):");
    for (i, hs) in halfspaces.iter().enumerate() {
        println!("  {}: {:?} · x ≤ {:.1}", i + 1, hs.normal(), hs.offset());
    }

    // For unbounded regions, we might need an interior point
    let interior_point = Some(arr1(&[1.0, 1.0]));
    let result = HalfspaceIntersection::new(&halfspaces, interior_point);

    match result {
        Ok(intersection) => {
            println!("Results:");
            println!("  Is bounded: {}", intersection.is_bounded());
            println!("  Is feasible: {}", intersection.is_feasible());
            println!("  Vertices: {}", intersection.num_vertices());

            if !intersection.is_bounded() {
                println!("  Note: This region extends to infinity");
            }
        }
        Err(e) => {
            println!("Could not compute intersection: {}", e);
            println!("This is expected for unbounded regions without proper constraints");
        }
    }

    Ok(())
}

fn empty_intersection_example() -> Result<(), Box<dyn std::error::Error>> {
    // Define contradictory halfspaces: x ≤ 0 and x ≥ 1
    let halfspaces = vec![
        Halfspace::new(arr1(&[1.0, 0.0]), 0.0),   // x ≤ 0
        Halfspace::new(arr1(&[-1.0, 0.0]), -1.0), // x ≥ 1
    ];

    println!("Contradictory halfspaces:");
    println!("  1: x ≤ 0");
    println!("  2: x ≥ 1");

    let result = HalfspaceIntersection::new(&halfspaces, None);

    match result {
        Ok(intersection) => {
            println!("Results:");
            println!("  Is feasible: {}", intersection.is_feasible());
            println!("  Vertices: {}", intersection.num_vertices());

            if !intersection.is_feasible() {
                println!("  The intersection is empty (no solution exists)");
            }
        }
        Err(e) => {
            println!("Error computing intersection: {}", e);
            println!("This indicates the constraints are contradictory");
        }
    }

    Ok(())
}

fn custom_polytope_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a more complex polytope
    let halfspaces = vec![
        Halfspace::new(arr1(&[-1.0, 0.0]), 0.0), // x ≥ 0
        Halfspace::new(arr1(&[0.0, -1.0]), 0.0), // y ≥ 0
        Halfspace::new(arr1(&[1.0, 1.0]), 3.0),  // x + y ≤ 3
        Halfspace::new(arr1(&[1.0, -1.0]), 1.0), // x - y ≤ 1
        Halfspace::new(arr1(&[-1.0, 1.0]), 1.0), // -x + y ≤ 1  =>  y - x ≤ 1
    ];

    println!("Custom polytope halfspaces:");
    let descriptions = ["x ≥ 0", "y ≥ 0", "x + y ≤ 3", "x - y ≤ 1", "y - x ≤ 1"];
    for (i, desc) in descriptions.iter().enumerate() {
        println!("  {}: {}", i + 1, desc);
    }

    // Provide an interior point
    let interior_point = Some(arr1(&[1.0, 1.0]));

    // Verify the interior point satisfies all constraints
    println!("Checking interior point [1.0, 1.0]:");
    for (i, hs) in halfspaces.iter().enumerate() {
        let satisfies = hs.contains(&interior_point.as_ref().unwrap().view());
        println!(
            "  Constraint {}: {}",
            i + 1,
            if satisfies { "✓" } else { "✗" }
        );
    }

    let intersection = HalfspaceIntersection::new(&halfspaces, interior_point)?;

    println!("Results:");
    println!("  Vertices: {}", intersection.num_vertices());
    println!("  Faces: {}", intersection.num_faces());
    println!("  Is bounded: {}", intersection.is_bounded());
    println!("  Is feasible: {}", intersection.is_feasible());

    if intersection.is_feasible() {
        let area = intersection.volume()?;
        println!("  Area: {:.3}", area);

        println!("  Vertex coordinates:");
        for (i, vertex) in intersection.vertices().outer_iter().enumerate() {
            println!("    Vertex {}: [{:.3}, {:.3}]", i, vertex[0], vertex[1]);
        }
    }

    Ok(())
}
