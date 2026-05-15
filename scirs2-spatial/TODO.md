# scirs2-spatial TODO

## Status: v0.4.3 Released (May 3, 2026)

## v0.3.3 Completed

### Spatial Data Structures
- KD-Tree with SIMD-accelerated distance computations
- Ball Tree for high-dimensional nearest neighbor search
- R*-Tree (improved R-tree variant with forced reinsertion)
- Octree for 3D point cloud spatial indexing
- Quadtree for 2D spatial indexing
- Grid index for fixed-resolution spatial hashing

### Distance Metrics
- 20+ distance functions: Euclidean, Manhattan, Chebyshev, Minkowski, Mahalanobis, Hamming, Jaccard, Cosine, Correlation, Canberra, Bray-Curtis
- SIMD-accelerated Euclidean, Manhattan, Chebyshev (2x speedup on f32)
- Pairwise distance matrices: `pdist`, `cdist`, `squareform`
- Set-based distances: Hausdorff (directed and symmetric), Wasserstein, Gromov-Hausdorff

### Computational Geometry
- Convex hull (2D Graham scan, 3D incremental) with degenerate case handling
- Delaunay triangulation with numerical stability (near-collinear point handling)
- Voronoi diagrams via Fortune's sweep line algorithm
- Alpha shapes and halfspace intersection
- Boolean polygon operations (union, intersection, difference)

### Geospatial Data
- Haversine and Vincenty geodesic distance formulas
- Map projections: Mercator, Lambert conformal conic, equirectangular
- WGS84/UTM coordinate conversions
- Topography analysis: slope, aspect, curvature, TRI, TPI
- Spatial statistics: Moran's I, Geary's C, Ripley's K, G function

### Spatial Join Operations
- Point-in-polygon join
- Distance-based join (all pairs within threshold)
- Nearest-neighbor join between two datasets

### Trajectory Analysis
- Ramer-Douglas-Peucker simplification
- Frechet distance
- Dynamic Time Warping (DTW)
- Speed, acceleration, curvature computation

### Point Cloud Processing
- Normal estimation (PCA per neighborhood)
- Statistical outlier removal
- Radius outlier removal
- Voxel grid downsampling

### Path Planning
- A* (grid and continuous)
- RRT and RRT* with configurable sampling
- PRM (Probabilistic Roadmap Method)
- Visibility graphs for polygonal environments
- Dubins paths, Reeds-Shepp paths

### 3D Transformations
- Quaternion arithmetic, conjugation, normalization
- Rotation matrices and Euler angle conversions
- Rigid transforms (SE(3)) and pose composition
- SLERP and rotation splines (Squad, cubic)
- Procrustes analysis (orthogonal and extended)

### Spatial Interpolation
- Simple Kriging and Ordinary Kriging with variogram fitting
- Inverse Distance Weighting (IDW)
- Radial Basis Functions (RBF) with multiple kernel choices
- Natural neighbor interpolation (Sibson)
- Shepard's method (generalized IDW)

### Geometric Algorithms
- Sweep line for line segment intersection (Bentley-Ottmann)
- Trapezoidal map for point location
- Arrangement computation (planar subdivisions)
- Ramer-Douglas-Peucker and Visvalingam-Whyatt simplification

### Collision Detection
- Circle vs circle, sphere vs sphere, AABB vs AABB, OBB vs OBB
- Continuous collision detection for moving objects
- BVH broad-phase with SAT narrow-phase

## v0.4.0 Roadmap

### GPU-Accelerated Spatial Indexing
- GPU-based KD-Tree construction and traversal
- GPU batch nearest-neighbor queries (millions of queries per second)
- GPU distance matrix computation via OxiBLAS
- CUDA/Metal backend selection through scirs2-core GPU abstractions

### Real-Time Streaming Spatial Queries
- Incremental insertion and deletion in KD-Tree and R*-Tree (kinetic data structures)
- Sliding window spatial queries for streaming point clouds
- Online Voronoi diagram updates with vertex event processing
- Pub/sub interface for spatial change notifications

### Advanced Spatial Statistics — Implemented in v0.4.0
- [x] Local Indicators of Spatial Association (LISA)
- [x] Kernel density estimation (KDE) with spatial bandwidth selection
- Spatial regression models (spatial lag, spatial error)
- [x] Spatial scan statistics for cluster detection
- [x] Getis-Ord Gi* statistic

### Large-Scale Geospatial Processing
- Efficient handling of billion-point datasets via chunked R*-Tree
- [x] Hilbert curve spatial sorting for cache-efficient access — see `src/hilbert.rs`
- GeoParquet and GeoArrow format support (via scirs2-io)

## Known Issues

- Voronoi construction for >100K seed points may be slow; use grid-based approximation for large inputs
- Ball Tree does not yet support user-defined distance functions with non-Euclidean metrics in all cases
- [x] R*-Tree deletion is implemented — see `src/rtree/deletion.rs` (lines 17-400) and `src/rect_rtree.rs:747`
- Kriging variogram fitting may diverge without good initial parameter estimates for poorly sampled data
