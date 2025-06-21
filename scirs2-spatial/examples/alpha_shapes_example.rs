//! Alpha Shapes Example
//!
//! This example demonstrates the use of alpha shapes for analyzing point sets.
//! Alpha shapes provide a generalization of convex hulls that can capture
//! non-convex boundaries and holes in data.

use ndarray::array;
use scirs2_spatial::alpha_shapes::AlphaShape;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Alpha Shapes Example ===\n");

    // Example 1: Basic 2D alpha shape
    println!("1. Basic 2D Alpha Shape");
    basic_2d_alpha_shape()?;
    println!();

    // Example 2: Multiple alpha values comparison
    println!("2. Multiple Alpha Values Comparison");
    alpha_spectrum_analysis()?;
    println!();

    // Example 3: Non-convex shape (C-shape)
    println!("3. Non-Convex Shape Analysis");
    non_convex_shape_example()?;
    println!();

    // Example 4: Point cloud with outliers
    println!("4. Point Cloud with Outliers");
    outlier_detection_example()?;
    println!();

    // Example 5: 3D alpha shape
    println!("5. 3D Alpha Shape");
    alpha_shape_3d_example()?;
    println!();

    // Example 6: Optimal alpha finding
    println!("6. Finding Optimal Alpha");
    optimal_alpha_example()?;
    println!();

    // Example 7: Circular point arrangement
    println!("7. Circular Point Arrangement");
    circular_points_example()?;

    Ok(())
}

fn basic_2d_alpha_shape() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple square with an interior point
    let points = array![
        [0.0, 0.0], // Bottom-left
        [2.0, 0.0], // Bottom-right
        [2.0, 2.0], // Top-right
        [0.0, 2.0], // Top-left
        [1.0, 1.0], // Center point
    ];

    println!("Points: {:?}", points);

    // Compute alpha shape with moderate alpha
    let alpha = 1.5;
    let alpha_shape = AlphaShape::new(&points, alpha)?;

    println!("Alpha: {}", alpha);
    println!(
        "Number of simplices in complex: {}",
        alpha_shape.complex().len()
    );
    println!(
        "Number of boundary elements: {}",
        alpha_shape.boundary().len()
    );
    println!("Area: {:.3}", alpha_shape.measure()?);

    // Show the boundary edges
    println!("Boundary edges:");
    for (i, edge) in alpha_shape.boundary().iter().enumerate() {
        let p1 = &points.row(edge[0]);
        let p2 = &points.row(edge[1]);
        println!(
            "  Edge {}: [{:.1}, {:.1}] -> [{:.1}, {:.1}]",
            i, p1[0], p1[1], p2[0], p2[1]
        );
    }

    Ok(())
}

fn alpha_spectrum_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Create a more complex point set
    let points = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0],
        [3.0, 1.0], // Outlier
    ];

    let alphas = vec![0.5, 1.0, 1.5, 2.0, 3.0];
    let shapes = AlphaShape::multi_alpha(&points, &alphas)?;

    println!("Alpha spectrum analysis:");
    println!("Alpha\t| Complex Size\t| Boundary Size\t| Area");
    println!("-----\t| -----------\t| ------------\t| ----");

    for (i, shape) in shapes.iter().enumerate() {
        let area = shape.measure().unwrap_or(0.0);
        println!(
            "{:.1}\t| {}\t\t| {}\t\t| {:.3}",
            alphas[i],
            shape.complex().len(),
            shape.boundary().len(),
            area
        );
    }

    Ok(())
}

fn non_convex_shape_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a C-shaped point set
    let mut points_vec = Vec::new();

    // Bottom horizontal line
    for i in 0..=10 {
        points_vec.push([i as f64 * 0.5, 0.0]);
    }

    // Left vertical line
    for i in 1..=10 {
        points_vec.push([0.0, i as f64 * 0.5]);
    }

    // Top horizontal line
    for i in 1..=10 {
        points_vec.push([i as f64 * 0.5, 5.0]);
    }

    // Add some interior points to make it interesting
    points_vec.push([1.0, 2.5]);
    points_vec.push([2.0, 2.5]);
    points_vec.push([1.5, 1.0]);
    points_vec.push([1.5, 4.0]);

    let n = points_vec.len();
    let mut flat_data = Vec::with_capacity(n * 2);
    for point in &points_vec {
        flat_data.push(point[0]);
        flat_data.push(point[1]);
    }

    let points = ndarray::Array2::from_shape_vec((n, 2), flat_data)?;

    println!("C-shaped point cloud with {} points", n);

    // Try different alpha values
    let alphas = vec![0.3, 0.7, 1.0, 2.0];

    for alpha in alphas {
        let alpha_shape = AlphaShape::new(&points, alpha)?;
        let area = alpha_shape.measure()?;

        println!(
            "Alpha {:.1}: {} simplices, {} boundary edges, area {:.2}",
            alpha,
            alpha_shape.complex().len(),
            alpha_shape.boundary().len(),
            area
        );
    }

    Ok(())
}

fn outlier_detection_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a main cluster with some outliers
    let mut points_vec = Vec::new();

    // Main cluster (roughly circular)
    for i in 0..=20 {
        let angle = 2.0 * PI * (i as f64) / 20.0;
        let radius = 1.0 + 0.1 * (3.0 * angle).sin(); // Slight variation
        points_vec.push([radius * angle.cos(), radius * angle.sin()]);
    }

    // Add some interior points
    points_vec.push([0.0, 0.0]);
    points_vec.push([0.3, 0.2]);
    points_vec.push([-0.2, 0.4]);
    points_vec.push([0.1, -0.3]);

    // Add outliers
    points_vec.push([3.0, 0.0]); // Far outlier
    points_vec.push([2.5, 1.0]); // Medium outlier
    points_vec.push([-2.0, -1.0]); // Far outlier

    let n = points_vec.len();
    let mut flat_data = Vec::with_capacity(n * 2);
    for point in &points_vec {
        flat_data.push(point[0]);
        flat_data.push(point[1]);
    }

    let points = ndarray::Array2::from_shape_vec((n, 2), flat_data)?;

    println!("Point cloud with {} points including outliers", n);

    // Use alpha shapes to identify core vs outlier structure
    let alpha_small = AlphaShape::new(&points, 0.5)?;
    let alpha_large = AlphaShape::new(&points, 2.0)?;

    println!("Small alpha (0.5):");
    println!("  Complex: {} simplices", alpha_small.complex().len());
    println!("  Boundary: {} edges", alpha_small.boundary().len());
    println!("  Area: {:.3}", alpha_small.measure()?);

    println!("Large alpha (2.0):");
    println!("  Complex: {} simplices", alpha_large.complex().len());
    println!("  Boundary: {} edges", alpha_large.boundary().len());
    println!("  Area: {:.3}", alpha_large.measure()?);

    // Find optimal alpha
    let (optimal_alpha, optimal_shape) = AlphaShape::find_optimal_alpha(&points, "area")?;
    println!("Optimal alpha: {:.3}", optimal_alpha);
    println!("Optimal shape area: {:.3}", optimal_shape.measure()?);

    Ok(())
}

fn alpha_shape_3d_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 3D point set (vertices of a cube plus center)
    let points = array![
        [0.0, 0.0, 0.0], // Cube vertices
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5], // Center point
        [2.0, 0.5, 0.5], // Outlier
    ];

    println!("3D point cloud with {} points", points.nrows());

    let alpha = 1.0;
    let alpha_shape = AlphaShape::new(&points, alpha)?;

    println!("Alpha: {}", alpha);
    println!(
        "Number of tetrahedra in complex: {}",
        alpha_shape.complex().len()
    );
    println!(
        "Number of boundary triangles: {}",
        alpha_shape.boundary().len()
    );
    println!("Volume: {:.3}", alpha_shape.measure()?);

    // Show some circumradii
    println!("Circumradii of simplices:");
    for (i, &radius) in alpha_shape.circumradii().iter().enumerate() {
        if i < 5 {
            // Show first few
            println!("  Simplex {}: radius = {:.3}", i, radius);
        }
    }
    if alpha_shape.circumradii().len() > 5 {
        println!("  ... and {} more", alpha_shape.circumradii().len() - 5);
    }

    Ok(())
}

fn optimal_alpha_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a point set with clear structure
    let mut points_vec = Vec::new();

    // Create two clusters
    // Cluster 1: around (0, 0)
    for i in 0..8 {
        let angle = 2.0 * PI * (i as f64) / 8.0;
        points_vec.push([0.8 * angle.cos(), 0.8 * angle.sin()]);
    }
    points_vec.push([0.0, 0.0]); // Center of cluster 1

    // Cluster 2: around (4, 0)
    for i in 0..8 {
        let angle = 2.0 * PI * (i as f64) / 8.0;
        points_vec.push([4.0 + 0.8 * angle.cos(), 0.8 * angle.sin()]);
    }
    points_vec.push([4.0, 0.0]); // Center of cluster 2

    let n = points_vec.len();
    let mut flat_data = Vec::with_capacity(n * 2);
    for point in &points_vec {
        flat_data.push(point[0]);
        flat_data.push(point[1]);
    }

    let points = ndarray::Array2::from_shape_vec((n, 2), flat_data)?;

    println!("Two-cluster point set with {} points", n);

    // Find optimal alpha using different criteria
    let criteria = ["area", "boundary"];

    for criterion in &criteria {
        let (optimal_alpha, optimal_shape) = AlphaShape::find_optimal_alpha(&points, criterion)?;
        let area = optimal_shape.measure()?;

        println!("Optimal alpha ({}): {:.3}", criterion, optimal_alpha);
        println!("  Complex size: {}", optimal_shape.complex().len());
        println!("  Boundary edges: {}", optimal_shape.boundary().len());
        println!("  Total area: {:.3}", area);
    }

    Ok(())
}

fn circular_points_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create points arranged in a circle (ring shape)
    let mut points_vec = Vec::new();
    let num_points = 20;

    // Outer ring
    for i in 0..num_points {
        let angle = 2.0 * PI * (i as f64) / (num_points as f64);
        points_vec.push([2.0 * angle.cos(), 2.0 * angle.sin()]);
    }

    // Inner ring
    for i in 0..num_points {
        let angle = 2.0 * PI * (i as f64) / (num_points as f64);
        points_vec.push([1.0 * angle.cos(), 1.0 * angle.sin()]);
    }

    let n = points_vec.len();
    let mut flat_data = Vec::with_capacity(n * 2);
    for point in &points_vec {
        flat_data.push(point[0]);
        flat_data.push(point[1]);
    }

    let points = ndarray::Array2::from_shape_vec((n, 2), flat_data)?;

    println!("Circular ring with {} points", n);

    // Test different alpha values to see the ring structure
    let alphas = vec![0.5, 1.0, 1.5, 2.0, 3.0];

    println!("Alpha analysis for ring structure:");
    for alpha in alphas {
        let alpha_shape = AlphaShape::new(&points, alpha)?;
        let area = alpha_shape.measure()?;

        println!(
            "α = {:.1}: {} triangles, {} edges, area = {:.2}",
            alpha,
            alpha_shape.complex().len(),
            alpha_shape.boundary().len(),
            area
        );
    }

    // Find the optimal alpha that captures the ring without connecting inner and outer
    let (optimal_alpha, optimal_shape) = AlphaShape::find_optimal_alpha(&points, "boundary")?;
    println!("\nOptimal alpha for ring: {:.3}", optimal_alpha);
    println!(
        "Captures ring with {} boundary edges",
        optimal_shape.boundary().len()
    );

    Ok(())
}
