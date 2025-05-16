use ndarray::{array, Array2};
use rand::prelude::*;
use scirs2_spatial::{BoundingBox2D, Quadtree};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Quadtree Example");
    println!("===============\n");

    // Create a simple set of points
    println!("Creating a simple data set with 10 points...");
    let points = array![
        [0.0, 0.0],   // Origin
        [1.0, 0.0],   // Right
        [0.0, 1.0],   // Up
        [1.0, 1.0],   // Up-right
        [0.5, 0.5],   // Center
        [0.25, 0.25], // Bottom-left quadrant
        [0.75, 0.25], // Bottom-right quadrant
        [0.25, 0.75], // Top-left quadrant
        [0.75, 0.75], // Top-right quadrant
        [2.0, 2.0],   // Outside the unit square
    ];

    // Build the quadtree
    println!("Building quadtree...");
    let quadtree = Quadtree::new(&points.view())?;

    println!("Quadtree properties:");
    println!("  - Number of points: {}", quadtree.size());
    println!("  - Maximum depth: {}", quadtree.max_depth());

    if let Some(bounds) = quadtree.bounds() {
        println!("  - Bounding box:");
        println!("      Min: [{:.1}, {:.1}]", bounds.min[0], bounds.min[1]);
        println!("      Max: [{:.1}, {:.1}]", bounds.max[0], bounds.max[1]);
        println!(
            "      Dimensions: [{:.1}, {:.1}]",
            bounds.dimensions()[0],
            bounds.dimensions()[1]
        );
    }

    // Perform nearest neighbor search
    println!("\nNearest Neighbor Search");
    println!("----------------------");

    let query_point = array![0.4, 0.4];
    println!(
        "Query point: [{:.1}, {:.1}]",
        query_point[0], query_point[1]
    );

    println!("\nFinding 3 nearest neighbors:");
    let (indices, distances) = quadtree.query_nearest(&query_point.view(), 3)?;

    for i in 0..indices.len() {
        let idx = indices[i];
        let point = points.row(idx);
        println!(
            "  #{}: Point #{} at [{:.1}, {:.1}], distance: {:.4}",
            i + 1,
            idx,
            point[0],
            point[1],
            distances[i].sqrt()
        );
    }

    // Perform radius search
    println!("\nRadius Search");
    println!("------------");

    let search_radius = 0.3;
    println!(
        "Finding points within radius {:.1} of query point:",
        search_radius
    );

    let (indices, distances) = quadtree.query_radius(&query_point.view(), search_radius)?;

    if indices.is_empty() {
        println!("  No points found within the radius");
    } else {
        for i in 0..indices.len() {
            let idx = indices[i];
            let point = points.row(idx);
            println!(
                "  Point #{} at [{:.1}, {:.1}], distance: {:.4}",
                idx,
                point[0],
                point[1],
                distances[i].sqrt()
            );
        }
    }

    // Demonstrate region queries
    println!("\nRegion Queries");
    println!("-------------");

    // Define regions for testing
    let regions = [
        (
            "Unit square (0,0) to (1,1)",
            BoundingBox2D::new(&array![0.0, 0.0].view(), &array![1.0, 1.0].view())?,
        ),
        (
            "Bottom-right quadrant of unit square",
            BoundingBox2D::new(&array![0.5, 0.0].view(), &array![1.0, 0.5].view())?,
        ),
        (
            "Top-left quadrant of unit square",
            BoundingBox2D::new(&array![0.0, 0.5].view(), &array![0.5, 1.0].view())?,
        ),
        (
            "Small region around query point",
            BoundingBox2D::new(&array![0.35, 0.35].view(), &array![0.45, 0.45].view())?,
        ),
        (
            "Region outside unit square",
            BoundingBox2D::new(&array![1.5, 1.5].view(), &array![2.5, 2.5].view())?,
        ),
    ];

    for (name, region) in &regions {
        println!("\nRegion: {}", name);
        println!("  Min: [{:.2}, {:.2}]", region.min[0], region.min[1]);
        println!("  Max: [{:.2}, {:.2}]", region.max[0], region.max[1]);

        // Check if any points in region
        let has_points = quadtree.points_in_region(region);
        println!("  Contains points: {}", has_points);

        // Get all points in region
        let indices = quadtree.get_points_in_region(region);
        println!("  Number of points in region: {}", indices.len());

        if !indices.is_empty() {
            println!("  Points in region:");
            for &idx in &indices {
                let point = points.row(idx);
                println!("    Point #{} at [{:.2}, {:.2}]", idx, point[0], point[1]);
            }
        }
    }

    // Demonstrate performance with larger dataset
    println!("\nPerformance with Larger Dataset");
    println!("-----------------------------");

    let n_points = 10000;
    println!("Creating a random dataset with {} points...", n_points);

    let mut rng = rand::rng();
    let mut large_points = Array2::zeros((n_points, 2));

    for i in 0..n_points {
        for j in 0..2 {
            large_points[[i, j]] = rng.random_range(-100.0..100.0);
        }
    }

    println!("Building quadtree...");
    let start = std::time::Instant::now();
    let large_quadtree = Quadtree::new(&large_points.view())?;
    let build_time = start.elapsed();

    println!("  Built quadtree in {:.2?}", build_time);
    println!("  Maximum depth: {}", large_quadtree.max_depth());

    // Test nearest neighbor query performance
    println!("\nTesting query performance...");
    let query_point = array![0.0, 0.0];

    let start = std::time::Instant::now();
    let (_indices, _) = large_quadtree.query_nearest(&query_point.view(), 10)?;
    let query_time = start.elapsed();

    println!("  Found 10 nearest neighbors in {:.2?}", query_time);

    // Test radius search performance
    let start = std::time::Instant::now();
    let (indices, _) = large_quadtree.query_radius(&query_point.view(), 10.0)?;
    let radius_time = start.elapsed();

    println!(
        "  Found {} points within radius 10.0 in {:.2?}",
        indices.len(),
        radius_time
    );

    // Test region query performance
    let region = BoundingBox2D::new(&array![-10.0, -10.0].view(), &array![10.0, 10.0].view())?;

    let start = std::time::Instant::now();
    let indices = large_quadtree.get_points_in_region(&region);
    let region_time = start.elapsed();

    println!(
        "  Found {} points in region in {:.2?}",
        indices.len(),
        region_time
    );

    println!("\nExample completed successfully!");
    Ok(())
}
