# scirs2-spatial TODO

This module provides spatial algorithms and data structures similar to SciPy's spatial module.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] Distance computations
  - [x] Euclidean distance
  - [x] Manhattan distance
  - [x] Chebyshev distance
  - [x] Minkowski distance
  - [x] Hamming distance
  - [x] Canberra distance
  - [x] Cosine distance
  - [x] Correlation distance
  - [x] Jaccard distance
  - [x] Pairwise and cross-distance matrices
- [x] Spatial data structures
  - [x] KD-tree implementation
  - [x] Nearest neighbor queries
  - [x] Range queries
- [x] Computational Geometry
  - [x] Convex hull (using qhull)
    - [x] Support for 2D and 3D convex hulls
    - [x] Handling of degenerate cases
    - [x] Function signatures with ArrayView2 support
  - [x] Delaunay triangulation (using qhull)
    - [x] Robust implementation with updated QHull API usage
    - [x] Point perturbation for numerical stability
    - [x] Special case handling for degenerate geometries
  - [x] Voronoi diagrams (using Delaunay triangulation)
    - [x] Special case handlers for triangles and squares
    - [x] Robust implementation with proper error handling
    - [x] Function signatures with ArrayView2 support

## Current Priorities (based on SciPy comparison)

- [x] Spherical Voronoi implementation
  - [x] Construction algorithm for points on a sphere
  - [x] Region area calculations
  - [x] Geodesic distance calculations
  - [x] Visualization utilities
- [x] Spatial transformations module (similar to scipy.spatial.transform)
  - [x] Rotation class with multiple representations
    - [x] Quaternions
    - [x] Rotation matrices
    - [x] Euler angles (with support for all conventions: XYZ, ZYX, XYX, XZX, YXY, YZY, ZXZ, ZYZ)
    - [x] Axis-angle representation
  - [x] RigidTransform class (rotation + translation)
  - [x] Slerp (Spherical Linear Interpolation)
  - [x] RotationSpline (with support for both SLERP and cubic interpolation)
- [x] Procrustes analysis
  - [x] Orthogonal Procrustes
  - [x] Extended Procrustes

## Distance Metrics

- [x] Complete collection of distance metrics
  - [x] Numeric vector metrics
    - [x] Euclidean
    - [x] Manhattan/cityblock
    - [x] Chebyshev/chessboard
    - [x] Minkowski
    - [x] Mahalanobis
    - [x] Canberra
    - [x] Cosine
    - [x] Correlation
    - [x] Bray-Curtis
    - [x] Seuclidean (standardized Euclidean)
  - [x] Boolean vector metrics
    - [x] Hamming
    - [x] Jaccard
    - [x] Dice
    - [x] Kulsinski
    - [x] Rogers-Tanimoto
    - [x] Russell-Rao
    - [x] Sokal-Michener
    - [x] Sokal-Sneath
    - [x] Yule
  - [x] Set-based distances
    - [x] Earth Mover's distance (Wasserstein)
    - [x] Hausdorff distance
    - [x] Gromov-Hausdorff distance

## Spatial Data Structures

- [x] Complete existing data structures
  - [x] Improve KD-tree performance
    - [x] Optimized construction algorithms
    - [x] Balanced tree construction
    - [x] Parallelization of search operations
    - [x] Batch query optimization
  - [x] Enhance nearest neighbor functionality
    - [x] K-nearest neighbors
    - [x] Radius-based neighbor finding
    - [x] Approximate nearest neighbors
    - [x] Priority queue-based algorithms
- [x] Add more spatial data structures
  - [x] Ball tree
    - [x] Construction algorithm
    - [x] Neighbor search
    - [x] Range queries
  - [x] R-tree
    - [x] Insertion and deletion algorithms
    - [x] Range queries
    - [x] Spatial joins
  - [x] Octree for 3D data
    - [x] Construction algorithm
    - [x] Neighbor search
    - [x] Collision detection
  - [x] Quad tree for 2D data
    - [x] Region-based subdivision
    - [x] Point-based queries and searches

## Computational Geometry

- [ ] Complete placeholder implementations
  - [x] Full convex hull algorithm
    - [x] 2D implementation
    - [x] 3D implementation
    - [x] N-dimensional hull
    - [x] Qhull integration
    - [ ] Additional convex hull algorithms (Graham scan, Jarvis march)
    - [ ] Comprehensive volume/area calculations
  - [x] Proper Voronoi diagram construction
    - [x] 2D Voronoi diagrams
    - [x] Robust handling of degenerate inputs
    - [x] Special case handling for triangles/squares
    - [x] 3D Voronoi diagrams
    - [x] Fortune's algorithm implementation
    - [x] Integration with visualization tools
  - [x] Spherical Voronoi diagrams
    - [x] Construction algorithm
    - [x] Geodesic distance calculations
  - [x] Delaunay triangulation
    - [x] 2D triangulation with proper error handling
    - [x] 3D triangulation
    - [x] Robust handling of degenerate cases
    - [x] Point perturbation for numerical stability
    - [ ] Constrained Delaunay triangulation
- [ ] Add complex geometric algorithms
  - [ ] Alpha shapes
    - [ ] 2D implementation
    - [ ] 3D implementation
  - [ ] Halfspace intersection
    - [ ] Convex polytope construction
    - [ ] Incremental construction algorithm
  - [x] Procrustes analysis
    - [x] Orthogonal Procrustes (implemented with SVD-based optimal rotation calculation)
    - [x] Extended Procrustes (with options for scaling, reflection, and translation)
  - [x] Polygon/polyhedron operations
    - [x] Point in polygon tests
    - [x] Area and volume calculations
    - [x] Intersection tests
    - [ ] Boolean operations (union, difference, intersection)

## Spatial Interpolation and Transforms

- [x] Spatial interpolation methods
  - [x] Natural neighbor interpolation
  - [x] Radial basis function interpolation
  - [x] Inverse distance weighting
  - [ ] Kriging (Gaussian process regression)
- [x] Spatial transformations
  - [x] 3D rotations
    - [x] Euler angles
    - [x] Quaternions
    - [x] Rotation matrices
    - [x] Axis-angle representation
  - [x] Rigid transforms
    - [x] 4x4 transform matrices
    - [x] Pose composition
    - [x] Interpolation between poses (Slerp and RotationSpline)
  - [x] Spherical coordinate transformations
    - [x] Cartesian to spherical (with batch support)
    - [x] Spherical to cartesian (with batch support)
    - [x] Geodesic calculations (distance and spherical triangle area)

## Path Planning and Navigation

- [x] Path planning algorithms
  - [x] A* search in continuous space
  - [x] RRT (Rapidly-exploring Random Tree)
    - [x] Basic RRT implementation
    - [x] RRT* for optimal path planning
    - [x] Bidirectional RRT (RRT-Connect)
  - [x] Visibility graphs
  - [x] Probabilistic roadmaps (PRM)
    - [x] Basic PRM implementation
    - [x] PRM with specialized 2D polygon collision detection
    - [x] Configurable connection strategies and sampling
  - [x] Potential field methods
    - [x] Attractive and repulsive force calculation
    - [x] Support for various obstacle types (circles, polygons, custom)
    - [x] Local minimum detection and handling
    - [x] 2D-specific optimizations
- [ ] Motion planning
  - [x] Collision detection
  - [x] Trajectory optimization
  - [x] Dubins paths
  - [x] Reeds-Shepp paths

## Geospatial Functionality

- [ ] Geographic coordinate systems
  - [ ] Coordinate transformations
  - [ ] Datum conversions
  - [ ] Map projections
- [ ] Geospatial distance metrics
  - [ ] Haversine distance
  - [ ] Vincenty distance
  - [ ] Great circle distance
  - [ ] Geodesic calculations on ellipsoids

## Implementation Strategies

- [ ] Performance optimization
  - [ ] SIMD-accelerated distance calculations
  - [ ] GPU-accelerated spatial queries
  - [ ] Parallel construction of spatial data structures
  - [ ] Multi-threaded batch operations
- [ ] Memory efficiency
  - [ ] Compact data representations
  - [ ] Caching strategies for repeated queries
  - [ ] Lazy evaluation for distance matrices
- [x] API design
  - [x] Consistent interface across all structures
  - [x] Flexible query parameters
  - [x] Function signatures with ArrayView2 support
  - [x] Proper error handling and type inference
  - [ ] Generic type parameters for custom point types
  - [ ] Integration with array ecosystem

## Documentation and Examples

- [x] Add more examples and documentation
  - [x] Proper doctest examples with correct type annotations
  - [x] Updated example code with ArrayView2 usage
  - [x] Error handling examples in documentation
  - [ ] Tutorial for spatial data analysis
  - [ ] Visual examples for different algorithms
  - [ ] Performance comparison of different data structures
  - [ ] Use case demonstrations

## Long-term Goals

- [ ] Performance comparable to or better than SciPy's spatial
- [ ] Integration with clustering and machine learning modules
- [ ] Support for large-scale spatial databases
- [ ] GPU-accelerated implementations for computationally intensive operations
- [ ] Specialized algorithms for robotics and computer vision
- [ ] Advanced visualization tools for spatial data
- [ ] Integration with geographic information systems (GIS)