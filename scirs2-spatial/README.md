# SciRS2 Spatial

[![crates.io](https://img.shields.io/crates/v/scirs2-spatial.svg)](https://crates.io/crates/scirs2-spatial)
[![License](https://img.shields.io/crates/l/scirs2-spatial.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-spatial)](https://docs.rs/scirs2-spatial)

Spatial algorithms and data structures for the SciRS2 scientific computing library. This module provides tools for spatial queries, distance calculations, and geometric algorithms.

## Features

- **k-d Tree Implementation**: Efficient spatial indexing for nearest neighbor queries
- **Distance Functions**: Various distance metrics for spatial data
- **Convex Hull Algorithms**: Algorithms for computing convex hulls
- **Voronoi Diagrams**: Functions for generating Voronoi diagrams
- **Utility Functions**: Helper functions for spatial data processing

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-spatial = "0.1.0-alpha.1"
```

To enable optimizations through the core module, add feature flags:

```toml
[dependencies]
scirs2-spatial = { version = "0.1.0-alpha.1", features = ["parallel"] }
```

## Usage

Basic usage examples:

```rust
use scirs2_spatial::{kdtree, distance, convex_hull, voronoi};
use scirs2_core::error::CoreResult;
use ndarray::array;

// k-d Tree for nearest neighbor searches
fn kdtree_example() -> CoreResult<()> {
    // Create some 2D points
    let points = array![
        [2.0, 3.0],
        [5.0, 4.0],
        [9.0, 6.0],
        [4.0, 7.0],
        [8.0, 1.0],
        [7.0, 2.0]
    ];
    
    // Build a k-d tree
    let tree = kdtree::KDTree::build(&points, None)?;
    
    // Query point
    let query = array![6.0, 5.0];
    
    // Find nearest neighbor
    let (idx, dist) = tree.query(&query, 1, None)?;
    println!("Nearest neighbor index: {}, distance: {}", idx[0], dist[0]);
    println!("Nearest point: {:?}", points.row(idx[0]));
    
    // Find k nearest neighbors
    let k = 3;
    let (indices, distances) = tree.query(&query, k, None)?;
    println!("k-nearest neighbor indices: {:?}, distances: {:?}", indices, distances);
    
    // Find all points within a certain radius
    let radius = 3.0;
    let (indices, distances) = tree.query_radius(&query, radius, None)?;
    println!("Points within radius {}: {:?}", radius, indices.len());
    for i in 0..indices.len() {
        println!("  Point {:?}, distance: {}", points.row(indices[i]), distances[i]);
    }
    
    Ok(())
}

// Distance calculations
fn distance_example() -> CoreResult<()> {
    // Create two points
    let p1 = array![1.0, 2.0, 3.0];
    let p2 = array![4.0, 5.0, 6.0];
    
    // Calculate various distances
    let euclidean = distance::euclidean(&p1, &p2)?;
    let manhattan = distance::manhattan(&p1, &p2)?;
    let chebyshev = distance::chebyshev(&p1, &p2)?;
    let minkowski = distance::minkowski(&p1, &p2, 3.0)?;
    
    println!("Euclidean distance: {}", euclidean);
    println!("Manhattan distance: {}", manhattan);
    println!("Chebyshev distance: {}", chebyshev);
    println!("Minkowski distance (p=3): {}", minkowski);
    
    // Calculate distance matrices
    let points = array![
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ];
    
    let dist_matrix = distance::pdist(&points, "euclidean")?;
    println!("Pairwise distance matrix:\n{:?}", dist_matrix);
    
    Ok(())
}

// Convex hull computation
fn convex_hull_example() -> CoreResult<()> {
    // Create 2D points
    let points = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.3, 0.2],
        [0.8, 0.7],
        [0.2, 0.8]
    ];
    
    // Compute the convex hull
    let hull = convex_hull::convex_hull(&points, false)?;
    
    println!("Convex hull indices: {:?}", hull);
    println!("Hull points:");
    for &idx in &hull {
        println!("  {:?}", points.row(idx));
    }
    
    Ok(())
}

// Voronoi diagram
fn voronoi_example() -> CoreResult<()> {
    // Create 2D points
    let points = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5]
    ];
    
    // Generate Voronoi diagram
    let diagram = voronoi::voronoi_diagram(&points, None)?;
    
    println!("Voronoi vertices: {:?}", diagram.vertices);
    println!("Voronoi regions: {:?}", diagram.regions);
    
    Ok(())
}
```

## Components

### k-d Tree

Efficient spatial indexing:

```rust
use scirs2_spatial::kdtree::{
    KDTree,                 // k-d tree implementation
    build,                  // Build a k-d tree from points
    query,                  // Query nearest neighbors
    query_radius,           // Query points within a radius
    query_ball_point,       // Query all points within a ball
    query_pairs,            // Query all pairs of points within distance
};
```

### Distance Functions

Distance metrics and utilities:

```rust
use scirs2_spatial::distance::{
    // Point-to-point distances
    euclidean,              // Euclidean (L2) distance
    manhattan,              // Manhattan (L1) distance
    chebyshev,              // Chebyshev (L∞) distance
    minkowski,              // Minkowski distance
    mahalanobis,            // Mahalanobis distance
    hamming,                // Hamming distance
    cosine,                 // Cosine distance
    jaccard,                // Jaccard distance
    
    // Distance matrices
    pdist,                  // Pairwise distances between points
    cdist,                  // Distances between two sets of points
    squareform,             // Convert between distance vector and matrix
    
    // Other distance utilities
    directed_hausdorff,     // Directed Hausdorff distance
    distance_matrix,        // Compute distance matrix
};
```

### Convex Hull

Convex hull algorithms:

```rust
use scirs2_spatial::convex_hull::{
    convex_hull,            // Compute convex hull of points
    convex_hull_plot,       // Generate points for plotting convex hull
    is_point_in_hull,       // Check if point is in convex hull
};
```

### Voronoi Diagrams

Functions for Voronoi diagrams:

```rust
use scirs2_spatial::voronoi::{
    voronoi_diagram,        // Generate Voronoi diagram
    voronoi_plot,           // Generate points for plotting Voronoi diagram
    VoronoiDiagram,         // Voronoi diagram structure
    VoronoiRegion,          // Voronoi region structure
};
```

### Utilities

Helper functions for spatial data:

```rust
use scirs2_spatial::utils::{
    cartesian_product,      // Generate Cartesian product of arrays
    delaunay_triangulation, // Generate Delaunay triangulation
    triangulate,            // Triangulate a polygon
    orient,                 // Orient points (clockwise/counterclockwise)
    point_in_polygon,       // Check if point is in polygon
};
```

## Advanced Features

### Optimized Ball Tree for High-Dimensional Data

The module includes an optimized Ball Tree implementation for high-dimensional nearest neighbor searches:

```rust
use scirs2_spatial::kdtree::KDTree;
use ndarray::Array2;

// Load high-dimensional data
let data = Array2::<f64>::zeros((1000, 100)); // 1000 points in 100 dimensions

// Create tree with leaf_size optimization and using "ball_tree" algorithm
let leaf_size = 30;
let tree = KDTree::build(&data, Some("ball_tree"), Some(leaf_size)).unwrap();

// Query is more efficient than standard k-d tree for high dimensions
let query = Array2::<f64>::zeros((1, 100));
let (indices, distances) = tree.query(&query, 5, None).unwrap();
```

### Custom Distance Metrics

Support for custom distance metrics:

```rust
use scirs2_spatial::distance::Distance;
use ndarray::ArrayView1;

// Create a custom distance metric
struct MyCustomDistance;

impl Distance for MyCustomDistance {
    fn compute(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        // Custom distance calculation
        let mut max_diff = 0.0;
        for i in 0..a.len() {
            let diff = (a[i] - b[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        max_diff
    }
}

// Use custom distance with k-d tree
let points = Array2::<f64>::zeros((100, 3));
let tree = KDTree::build_with_distance(&points, MyCustomDistance {}).unwrap();
```

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](../LICENSE) file for details.