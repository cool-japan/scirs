use ndarray::{array, Array2};
use rand::prelude::*;
use scirs2_spatial::{BoundingBox, Octree};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Octree Example");
    println!("==============\n");

    // Create a simple set of points
    println!("Creating a simple data set with 10 points...");
    let points = array![
        [0.0, 0.0, 0.0], // Origin
        [1.0, 0.0, 0.0], // Unit points along axes
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], // Unit points in coordinate planes
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0], // Unit cube corner
        [0.5, 0.5, 0.5], // Center of unit cube
        [2.0, 2.0, 2.0], // Point outside unit cube
    ];

    // Build the octree
    println!("Building octree...");
    let octree = Octree::new(&points.view())?;

    println!("Octree properties:");
    println!("  - Number of points: {}", octree.size());
    println!("  - Maximum depth: {}", octree.max_depth());

    if let Some(bounds) = octree.bounds() {
        println!("  - Bounding box:");
        println!(
            "      Min: [{:.1}, {:.1}, {:.1}]",
            bounds.min[0], bounds.min[1], bounds.min[2]
        );
        println!(
            "      Max: [{:.1}, {:.1}, {:.1}]",
            bounds.max[0], bounds.max[1], bounds.max[2]
        );
        println!(
            "      Dimensions: [{:.1}, {:.1}, {:.1}]",
            bounds.dimensions()[0],
            bounds.dimensions()[1],
            bounds.dimensions()[2]
        );
    }

    // Perform nearest neighbor search
    println!("\nNearest Neighbor Search");
    println!("----------------------");

    let query_point = array![0.4, 0.4, 0.4];
    println!(
        "Query point: [{:.1}, {:.1}, {:.1}]",
        query_point[0], query_point[1], query_point[2]
    );

    println!("\nFinding 3 nearest neighbors:");
    let (indices, distances) = octree.query_nearest(&query_point.view(), 3)?;

    for i in 0..indices.len() {
        let idx = indices[i];
        let point = points.row(idx);
        println!(
            "  #{}: Point #{} at [{:.1}, {:.1}, {:.1}], distance: {:.4}",
            i + 1,
            idx,
            point[0],
            point[1],
            point[2],
            distances[i].sqrt()
        );
    }

    // Perform radius search
    println!("\nRadius Search");
    println!("------------");

    let search_radius = 0.7;
    println!(
        "Finding points within radius {:.1} of query point:",
        search_radius
    );

    let (indices, distances) = octree.query_radius(&query_point.view(), search_radius)?;

    if indices.is_empty() {
        println!("  No points found within the radius");
    } else {
        for i in 0..indices.len() {
            let idx = indices[i];
            let point = points.row(idx);
            println!(
                "  Point #{} at [{:.1}, {:.1}, {:.1}], distance: {:.4}",
                idx,
                point[0],
                point[1],
                point[2],
                distances[i].sqrt()
            );
        }
    }

    // Demonstrate collision detection
    println!("\nCollision Detection");
    println!("------------------");

    // Create a second set of points representing another object
    let object_points = array![
        [1.9, 1.9, 1.9], // Close to the point at [2.0, 2.0, 2.0]
        [2.1, 2.1, 2.1],
    ];

    let collision_threshold = 0.2;
    println!(
        "Checking for collision with threshold {:.1}:",
        collision_threshold
    );
    println!("  Object points:");
    for i in 0..object_points.nrows() {
        let point = object_points.row(i);
        println!("    [{:.1}, {:.1}, {:.1}]", point[0], point[1], point[2]);
    }

    let collision = octree.check_collision(&object_points.view(), collision_threshold)?;
    println!("  Collision detected: {}", collision);

    // Demonstrate performance with larger dataset
    println!("\nPerformance with Larger Dataset");
    println!("-----------------------------");

    let n_points = 100000;
    println!("Creating a random dataset with {} points...", n_points);

    let mut rng = rand::rng();
    let mut large_points = Array2::zeros((n_points, 3));

    for i in 0..n_points {
        for j in 0..3 {
            large_points[[i, j]] = rng.random_range(-100.0..100.0);
        }
    }

    println!("Building octree...");
    let start = std::time::Instant::now();
    let large_octree = Octree::new(&large_points.view())?;
    let build_time = start.elapsed();

    println!("  Built octree in {:.2?}", build_time);
    println!("  Maximum depth: {}", large_octree.max_depth());

    // Test nearest neighbor query performance
    println!("\nTesting query performance...");
    let query_point = array![0.0, 0.0, 0.0];

    let start = std::time::Instant::now();
    let (_indices, _) = large_octree.query_nearest(&query_point.view(), 10)?;
    let query_time = start.elapsed();

    println!("  Found 10 nearest neighbors in {:.2?}", query_time);

    // Test radius search performance
    let start = std::time::Instant::now();
    let (indices, _) = large_octree.query_radius(&query_point.view(), 10.0)?;
    let radius_time = start.elapsed();

    println!(
        "  Found {} points within radius 10.0 in {:.2?}",
        indices.len(),
        radius_time
    );

    // Create custom bounding box
    println!("\nCustom Bounding Box Operations");
    println!("---------------------------");

    let min = array![-1.0, -1.0, -1.0];
    let max = array![2.0, 2.0, 2.0];

    let bbox = BoundingBox::new(&min.view(), &max.view())?;
    println!("Created bounding box:");
    println!(
        "  Min: [{:.1}, {:.1}, {:.1}]",
        bbox.min[0], bbox.min[1], bbox.min[2]
    );
    println!(
        "  Max: [{:.1}, {:.1}, {:.1}]",
        bbox.max[0], bbox.max[1], bbox.max[2]
    );

    let center = bbox.center();
    println!(
        "  Center: [{:.1}, {:.1}, {:.1}]",
        center[0], center[1], center[2]
    );

    let dimensions = bbox.dimensions();
    println!(
        "  Dimensions: [{:.1}, {:.1}, {:.1}]",
        dimensions[0], dimensions[1], dimensions[2]
    );

    let test_point = array![0.5, 0.5, 0.5];
    let contains = bbox.contains(&test_point.view())?;
    println!(
        "  Contains point [{:.1}, {:.1}, {:.1}]: {}",
        test_point[0], test_point[1], test_point[2], contains
    );

    println!("\nExample completed successfully!");
    Ok(())
}
