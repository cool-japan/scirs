# SciRS2 Spatial

[![crates.io](https://img.shields.io/crates/v/scirs2-spatial.svg)](https://crates.io/crates/scirs2-spatial)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-spatial)](https://docs.rs/scirs2-spatial)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)]()

**scirs2-spatial** is the spatial algorithms and computational geometry crate for the [SciRS2](https://github.com/cool-japan/scirs) scientific computing library. It provides spatial data structures, distance metrics, geometric algorithms, geospatial utilities, and path planning tools modeled after SciPy's `spatial` module.

Tested: 854/854 tests passing with default features, 890/890 with `--all-features` (verified 2026-07-15; the 0.6.5 fixes below changed test assertions/logic, not test counts).

**v0.6.5:** fixed an inverted `Ord` implementation on `octree`'s and `quadtree`'s k-nearest-neighbor `BinaryHeap` (`DistancePoint` in `src/octree.rs`/`src/quadtree.rs`) that evicted the *closest* candidate instead of the *farthest* once the result set grew beyond k, silently returning the farthest points instead of the nearest; also fixed `pathplanning::astar::AStarPlanner::reconstruct_path` (`src/pathplanning/astar.rs`), which reassigned `cost` from every node's g-value while walking backward from goal to start, so it always ended on the start node's own g-value of `0.0` regardless of the real path cost — `cost` is now captured once from the goal node before the backward walk.

**v0.6.3:** fixed undefined behavior in `DistancePool::create_aligned_buffer`/`create_numa_aware_buffer` (`src/memory_pool.rs`) — both allocated cache-aligned memory via raw `System.alloc` then freed it through `Box<[f64]>`'s default-aligned `Drop`, an alloc/dealloc layout mismatch that corrupted the heap on Windows (`STATUS_HEAP_CORRUPTION`); both now allocate through plain `Vec`/`Box<[f64]>`.

## What scirs2-spatial Provides

Use scirs2-spatial when you need to:

- Query nearest neighbors efficiently in 2D, 3D, or high-dimensional spaces
- Compute pairwise distance matrices with many distance metrics
- Build Voronoi diagrams, Delaunay triangulations, or convex hulls
- Work with geospatial data (geodesic distances, map projections, Moran's I)
- Perform spatial joins or grid-based indexing
- Analyze trajectories or point clouds
- Plan paths in continuous space (A*, RRT)
- Apply 3D transformations (quaternions, rigid transforms, SLERP)

## Features (v0.6.6)

### Spatial Data Structures
- **KD-Tree**: Efficient k-nearest neighbor and radius search in any dimension
- **Ball Tree**: Optimized for high-dimensional data (>10D)
- **R*-Tree**: Improved R-tree variant for spatial indexing of 2D/3D bounding boxes
- **Octree**: 3D spatial partitioning for point clouds
- **Quadtree**: 2D spatial partitioning
- **Grid Index**: Hash-grid for fast spatial lookup at fixed resolution

### Distance Metrics (20+ metrics)
- Euclidean, Manhattan (L1), Chebyshev (L-inf), Minkowski
- Mahalanobis (covariance-weighted)
- Cosine, correlation, Canberra
- Hamming, Jaccard, Bray-Curtis
- SIMD-accelerated computation for f32 and f64
- Pairwise distance matrices (`pdist`, `cdist`, `squareform`)

### Set-Based Distances
- Hausdorff distance (directed and symmetric)
- Wasserstein distance (Earth Mover's Distance)
- Gromov-Hausdorff distance (metric space comparison)

### Computational Geometry
- Convex hull (2D and 3D) with degenerate case handling
- Delaunay triangulation with numerical stability
- Voronoi diagrams via Fortune's sweep line algorithm
- Alpha shapes and halfspace intersection
- Polygon operations (point-in-polygon, area, centroid, boolean ops)
- 3D convex hull (incremental algorithm)

### Geospatial Data
- Geodesic distance calculations (Haversine, Vincenty)
- Map projections (Mercator, Lambert, equirectangular)
- Geographic coordinate system conversions (WGS84, UTM)
- Topography analysis (slope, aspect, curvature)
- Spatial statistics: Moran's I (spatial autocorrelation), Ripley's K function

### Spatial Join Operations
- Point-in-polygon join
- Distance-based spatial join
- Nearest-neighbor spatial join between two datasets

### Voronoi Diagrams
- Fortune's sweep line algorithm for exact 2D Voronoi
- Handles degenerate inputs (collinear points, coincident points)
- Furthest-site Voronoi variant

### Trajectory Analysis
- Trajectory smoothing and simplification (Ramer-Douglas-Peucker)
- Frechet distance between trajectories
- Dynamic time warping (DTW) for trajectory similarity
- Speed, acceleration, and curvature analysis

### Point Cloud Processing
- Normal estimation via PCA
- Outlier removal (statistical, radius-based)
- Voxel grid downsampling
- Point cloud registration (ICP - see scirs2-vision for full pipeline)

### Path Planning
- A* on grids and continuous spaces
- RRT (Rapidly-exploring Random Tree)
- RRT* (asymptotically optimal)
- Probabilistic Roadmap Method (PRM)
- Visibility graphs
- Dubins and Reeds-Shepp paths for car-like robots

### 3D Transformations
- Rotation representations: quaternions, rotation matrices, Euler angles
- Rigid body transforms and pose composition
- Spherical coordinate transformations
- SLERP rotation interpolation and rotation splines
- Procrustes analysis (shape alignment)

### Spatial Interpolation
- Kriging (Simple and Ordinary)
- Inverse Distance Weighting (IDW)
- Radial Basis Functions (RBF)
- Natural neighbor interpolation
- Shepard's method

### Geometric Algorithms
- Sweep line algorithms for line segment intersection
- Point location in planar subdivisions
- Arrangement computation
- Simplification of polylines and polygons (Visvalingam-Whyatt)

### Collision Detection
- Primitive shape collision (circles, spheres, AABB, OBB)
- Continuous collision detection (swept volumes)
- Broad-phase (BVH) and narrow-phase algorithms

## Installation

```toml
[dependencies]
scirs2-spatial = "0.6.6"
```

For parallel processing:

```toml
[dependencies]
scirs2-spatial = { version = "0.6.6", features = ["parallel"] }
```

## Quick Start

### KD-Tree Nearest Neighbor Search

```rust
use scirs2_spatial::KDTree;
use scirs2_core::ndarray::array;
use scirs2_core::error::CoreResult;

fn kdtree_example() -> CoreResult<()> {
    let points = array![
        [2.0f64, 3.0],
        [5.0, 4.0],
        [9.0, 6.0],
        [4.0, 7.0],
        [8.0, 1.0],
    ];

    let tree = KDTree::new(&points)?;

    // Single nearest neighbor
    let (indices, distances) = tree.query(&[6.0, 5.0], 1)?;
    println!("Nearest: index={}, dist={}", indices[0], distances[0]);

    // k nearest neighbors
    let (indices, distances) = tree.query(&[6.0, 5.0], 3)?;
    println!("3-nearest indices: {:?}", indices);

    // Radius search
    let (indices, distances) = tree.query_radius(&[6.0, 5.0], 3.0)?;
    println!("Points within radius 3: {}", indices.len());

    Ok(())
}
```

### Distance Metrics

```rust
use scirs2_spatial::distance;
use scirs2_core::error::CoreResult;

fn distance_example() -> CoreResult<()> {
    let p1 = [1.0f64, 2.0, 3.0];
    let p2 = [4.0, 5.0, 6.0];

    let euclidean = distance::euclidean(&p1, &p2);
    let manhattan = distance::manhattan(&p1, &p2);
    let cosine = distance::cosine(&p1, &p2);

    println!("Euclidean: {:.4}", euclidean);
    println!("Manhattan: {:.4}", manhattan);
    println!("Cosine: {:.4}", cosine);

    Ok(())
}
```

### Pairwise Distance Matrix

```rust
use scirs2_spatial::distance;
use scirs2_core::ndarray::array;
use scirs2_core::error::CoreResult;

fn cdist_example() -> CoreResult<()> {
    let set_a = array![[0.0f64, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let set_b = array![[1.0f64, 1.0], [2.0, 2.0]];

    let dist_matrix = distance::cdist(&set_a.view(), &set_b.view(), "euclidean")?;
    println!("Distance matrix shape: {:?}", dist_matrix.shape());

    Ok(())
}
```

### Voronoi Diagram

```rust
use scirs2_spatial::voronoi::Voronoi;
use scirs2_core::ndarray::array;
use scirs2_core::error::CoreResult;

fn voronoi_example() -> CoreResult<()> {
    let points = array![
        [0.0f64, 0.0],
        [1.0, 0.0],
        [0.5, 0.866],
        [0.5, 0.5],
    ];

    let vor = Voronoi::new(&points.view(), false)?;
    println!("Voronoi vertices: {}", vor.vertices().len());
    println!("Ridge count: {}", vor.ridge_points().len());

    Ok(())
}
```

### Geospatial Distance

```rust
use scirs2_spatial::geo::{haversine_distance, vincenty_distance};
use scirs2_core::error::CoreResult;

fn geo_example() -> CoreResult<()> {
    // Tokyo coordinates
    let lat1 = 35.6762_f64;
    let lon1 = 139.6503_f64;
    // New York coordinates
    let lat2 = 40.7128_f64;
    let lon2 = -74.0060_f64;

    let dist_km = haversine_distance(lat1, lon1, lat2, lon2);
    println!("Haversine distance: {:.1} km", dist_km / 1000.0);

    Ok(())
}
```

### Spatial Statistics (Moran's I)

```rust
use scirs2_spatial::advanced_spatial_stats::morans_i;
use scirs2_core::error::CoreResult;

fn spatial_stats_example() -> CoreResult<()> {
    // let values = ...;  // observed values at each location
    // let weights = ...; // spatial weights matrix (e.g., contiguity or distance-based)
    // let (i_stat, p_value) = morans_i(&values, &weights)?;
    // println!("Moran's I = {:.4}, p = {:.4}", i_stat, p_value);
    Ok(())
}
```

### R*-Tree Spatial Index

```rust
use scirs2_spatial::rtree::RStarTree;
use scirs2_core::error::CoreResult;

fn rtree_example() -> CoreResult<()> {
    let mut rtree = RStarTree::new();
    // Insert bounding boxes (min_x, min_y, max_x, max_y) with IDs
    // rtree.insert([0.0, 0.0, 1.0, 1.0], 0)?;
    // rtree.insert([1.5, 1.5, 2.5, 2.5], 1)?;

    // Query intersecting bounding boxes
    // let results = rtree.search([0.5, 0.5, 2.0, 2.0])?;
    Ok(())
}
```

## Feature Flags

| Flag | Description |
|------|-------------|
| `parallel` | Enable Rayon-based parallel distance matrix computation — in `default` |
| `async` | Async APIs via `tokio` |
| `cuda` / `rocm` / `vulkan` | GPU **capability detection only** (nvidia-smi/rocm-smi/vulkaninfo probes in `gpu_accel.rs`); does not unlock real GPU compute, which always falls back to CPU SIMD |
| `experimental` | Research-fringe APIs that are NOT production-ready — currently gates `SelfOptimizingAlgorithm`, which honestly returns `NotImplementedError` rather than fabricating results; excluded from `default` |

Corrected 2026-07-15: table previously listed only `parallel`; the four flags above also exist in `Cargo.toml` and are now documented.

## Performance

- SIMD-accelerated distance metrics (Euclidean, Manhattan, Chebyshev): up to 2x speedup for f32
- 1.5-25 million distance calculations per second (hardware dependent)
- KD-Tree: 20K+ nearest-neighbor queries per second (10K point dataset)
- Linear memory scaling with point count

## Documentation

Full API reference: [docs.rs/scirs2-spatial](https://docs.rs/scirs2-spatial)

## Compatibility

The API is modeled after SciPy's `scipy.spatial` module. Key equivalents:

| SciRS2 | SciPy |
|--------|-------|
| `KDTree::new()` | `scipy.spatial.KDTree()` |
| `distance::pdist()` | `scipy.spatial.distance.pdist()` |
| `distance::cdist()` | `scipy.spatial.distance.cdist()` |
| `Voronoi::new()` | `scipy.spatial.Voronoi()` |
| `ConvexHull::new()` | `scipy.spatial.ConvexHull()` |
| `Delaunay::new()` | `scipy.spatial.Delaunay()` |

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
