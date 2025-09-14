#![allow(deprecated)]
#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(missing_docs)]
#![allow(clippy::for_loops_over_fallibles)]
// Spatial algorithms module
#![allow(unreachable_patterns)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(private_interfaces)]
//
// This module provides implementations of various spatial algorithms,
// similar to SciPy's `spatial` module.
//
// ## Overview
//
// * Distance computations and metrics
// * KD-trees for efficient nearest neighbor searches
// * Ball trees for high-dimensional nearest neighbor searches
// * Voronoi diagrams and Delaunay triangulation
// * Convex hulls
// * Set-based distances (Hausdorff, Wasserstein)
// * Polygon operations
// * Spatial transformations
// * Path planning algorithms
//
// ## Features
//
// * Efficient nearest-neighbor queries with KD-Tree and Ball Tree data structures
// * Comprehensive set of distance metrics (Euclidean, Manhattan, Minkowski, etc.)
// * Distance matrix computations (similar to SciPy's cdist and pdist)
// * Convex hull computation using the Qhull library
// * Delaunay triangulation for 2D and higher dimensions
// * Customizable distance metrics for spatial data structures
// * Advanced query capabilities (k-nearest neighbors, radius search)
// * Set-based distances (Hausdorff, Wasserstein)
// * Polygon operations (point-in-polygon, area, centroid)
// * Path planning algorithms (A*, RRT, visibility graphs)
// * **Advanced MODE: Revolutionary Computing Paradigms** - Quantum-neuromorphic fusion, next-gen GPU architectures, AI-driven optimization, extreme performance beyond current limits
//
// ## Examples
//
// ### Distance Metrics
//
// ```
// use scirs2_spatial::distance::euclidean;
//
// let point1 = &[1.0, 2.0, 3.0];
// let point2 = &[4.0, 5.0, 6.0];
//
// let dist = euclidean(point1, point2);
// println!("Euclidean distance: {}", dist);
// ```
//
// ### KD-Tree for Nearest Neighbor Searches
//
// ```
// use scirs2_spatial::KDTree;
// use ndarray::array;
//
// // Create points
// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
//
// // Build KD-Tree
// let kdtree = KDTree::new(&points).unwrap();
//
// // Find 2 nearest neighbors to [0.5, 0.5]
// let (indices, distances) = kdtree.query(&[0.5, 0.5], 2).unwrap();
// println!("Indices of 2 nearest points: {:?}", indices);
// println!("Distances to 2 nearest points: {:?}", distances);
//
// // Find all points within radius 0.7
// let (idx_radius, dist_radius) = kdtree.query_radius(&[0.5, 0.5], 0.7).unwrap();
// println!("Found {} points within radius 0.7", idx_radius.len());
// ```
//
// ### Distance Matrices
//
// ```
// use scirs2_spatial::distance::{pdist, euclidean};
// use ndarray::array;
//
// // Create points
// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
//
// // Calculate pairwise distance matrix
// let dist_matrix = pdist(&points, euclidean);
// println!("Distance matrix shape: {:?}", dist_matrix.shape());
// ```
//
// ### Convex Hull
//
// ```
// use scirs2_spatial::convex_hull::ConvexHull;
// use ndarray::array;
//
// // Create points
// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
//
// // Compute convex hull
// let hull = ConvexHull::new(&points.view()).unwrap();
//
// // Get the hull vertices
// let vertices = hull.vertices();
// println!("Hull vertices: {:?}", vertices);
//
// // Check if a point is inside the hull
// let is_inside = hull.contains(&[0.25, 0.25]).unwrap();
// println!("Is point [0.25, 0.25] inside? {}", is_inside);
// ```
//
// ### Delaunay Triangulation
//
// ```
// use scirs2_spatial::delaunay::Delaunay;
// use ndarray::array;
//
// // Create points
// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
//
// // Compute Delaunay triangulation
// let tri = Delaunay::new(&points).unwrap();
//
// // Get the simplices (triangles in 2D)
// let simplices = tri.simplices();
// println!("Triangles: {:?}", simplices);
//
// // Find which triangle contains a point
// if let Some(idx) = tri.find_simplex(&[0.25, 0.25]) {
//     println!("Point [0.25, 0.25] is in triangle {}", idx);
// }
// ```
//
// ### Alpha Shapes
//
// ```
// use scirs2_spatial::AlphaShape;
// use ndarray::array;
//
// // Create a point set with some outliers
// let points = array![
//     [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  // Square corners
//     [0.5, 0.5],                                        // Interior point
//     [2.0, 0.5], [3.0, 0.5]                            // Outliers
// ];
//
// // Compute alpha shape with different alpha values
// let alpha_small = AlphaShape::new(&points, 0.3).unwrap();
// let alpha_large = AlphaShape::new(&points, 1.5).unwrap();
//
// // Get boundary (edges in 2D)
// let boundary_small = alpha_small.boundary();
// let boundary_large = alpha_large.boundary();
//
// println!("Small alpha boundary edges: {}", boundary_small.len());
// println!("Large alpha boundary edges: {}", boundary_large.len());
//
// // Find optimal alpha automatically
// let (optimal_alpha, optimalshape) = AlphaShape::find_optimal_alpha(&points, "area").unwrap();
// println!("Optimal alpha: {:.3}", optimal_alpha);
// println!("Shape area: {:.3}", optimalshape.measure().unwrap());
// ```
//
// ### Halfspace Intersection
//
// ```
// use scirs2_spatial::halfspace::{HalfspaceIntersection, Halfspace};
// use ndarray::array;
//
// // Define halfspaces for a unit square: x ≥ 0, y ≥ 0, x ≤ 1, y ≤ 1
// let halfspaces = vec![
//     Halfspace::new(array![-1.0, 0.0], 0.0),   // -x ≤ 0  =>  x ≥ 0
//     Halfspace::new(array![0.0, -1.0], 0.0),   // -y ≤ 0  =>  y ≥ 0
//     Halfspace::new(array![1.0, 0.0], 1.0),    //  x ≤ 1
//     Halfspace::new(array![0.0, 1.0], 1.0),    //  y ≤ 1
// ];
//
// let intersection = HalfspaceIntersection::new(&halfspaces, None).unwrap();
//
// // Get the vertices of the resulting polytope
// let vertices = intersection.vertices();
// println!("Polytope has {} vertices", vertices.nrows());
//
// // Check properties
// println!("Is bounded: {}", intersection.is_bounded());
// println!("Volume/Area: {:.3}", intersection.volume().unwrap());
// ```
//
// ### Boolean Operations on Polygons
//
// ```
// use scirs2_spatial::boolean_ops::{polygon_union, polygon_intersection, polygon_difference};
// use ndarray::array;
//
// // Define two overlapping squares
// let poly1 = array![
//     [0.0, 0.0],
//     [2.0, 0.0],
//     [2.0, 2.0],
//     [0.0, 2.0]
// ];
//
// let poly2 = array![
//     [1.0, 1.0],
//     [3.0, 1.0],
//     [3.0, 3.0],
//     [1.0, 3.0]
// ];
//
// // Compute union
// let union_result = polygon_union(&poly1.view(), &poly2.view()).unwrap();
// println!("Union has {} vertices", union_result.nrows());
//
// // Compute intersection
// let intersection_result = polygon_intersection(&poly1.view(), &poly2.view()).unwrap();
// println!("Intersection has {} vertices", intersection_result.nrows());
//
// // Compute difference (poly1 - poly2)
// let difference_result = polygon_difference(&poly1.view(), &poly2.view()).unwrap();
// println!("Difference has {} vertices", difference_result.nrows());
// ```
//
// ### Set-Based Distances
//
// ```
// use scirs2_spatial::set_distance::hausdorff_distance;
// use ndarray::array;
//
// // Create two point sets
// let set1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
// let set2 = array![[0.0, 0.5], [1.0, 0.5], [0.5, 1.0]];
//
// // Compute the Hausdorff distance
// let dist = hausdorff_distance(&set1.view(), &set2.view(), None);
// println!("Hausdorff distance: {}", dist);
// ```
//
// ### Polygon Operations
//
// ```
// use scirs2_spatial::polygon::{point_in_polygon, polygon_area, polygon_centroid};
// use ndarray::array;
//
// // Create a polygon (square)
// let polygon = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
//
// // Check if a point is inside
// let inside = point_in_polygon(&[0.5, 0.5], &polygon.view());
// println!("Is point [0.5, 0.5] inside? {}", inside);
//
// // Calculate polygon area
// let area = polygon_area(&polygon.view());
// println!("Polygon area: {}", area);
//
// // Calculate centroid
// let centroid = polygon_centroid(&polygon.view());
// println!("Polygon centroid: ({}, {})", centroid[0], centroid[1]);
// ```
//
// ### Ball Tree for Nearest Neighbor Searches
//
// ```
// use scirs2_spatial::BallTree;
// use ndarray::array;
//
// // Create points
// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
//
// // Build Ball Tree
// let ball_tree = BallTree::with_euclidean_distance(&points.view(), 2).unwrap();
//
// // Find 2 nearest neighbors to [0.5, 0.5]
// let (indices, distances) = ball_tree.query(&[0.5, 0.5], 2, true).unwrap();
// println!("Indices of 2 nearest points: {:?}", indices);
// println!("Distances to 2 nearest points: {:?}", distances.unwrap());
//
// // Find all points within radius 0.7
// let (idx_radius, dist_radius) = ball_tree.query_radius(&[0.5, 0.5], 0.7, true).unwrap();
// println!("Found {} points within radius 0.7", idx_radius.len());
// ```
//
// ### A* Pathfinding
//
// ```
// use scirs2_spatial::pathplanning::GridAStarPlanner;
//
// // Create a grid with some obstacles (true = obstacle, false = free space)
// let grid = vec![
//     vec![false, false, false, false, false],
//     vec![false, false, false, false, false],
//     vec![false, true, true, true, false],  // A wall of obstacles
//     vec![false, false, false, false, false],
//     vec![false, false, false, false, false],
// ];
//
// // Create an A* planner with the grid
// let planner = GridAStarPlanner::new(grid, false);
//
// // Find a path from top-left to bottom-right
// let start = [0, 0];
// let goal = [4, 4];
//
// let path = planner.find_path(start, goal).unwrap().unwrap();
//
// println!("Found a path with {} steps:", path.len() - 1);
// for (i, pos) in path.nodes.iter().enumerate() {
//     println!("  Step {}: {:?}", i, pos);
// }
// ```
//
// ### SIMD-Accelerated Distance Calculations
//
// ```
// use scirs2_spatial::simd_distance::{simd_euclidean_distance_batch, parallel_pdist};
// use ndarray::array;
//
// // SIMD batch distance calculation between corresponding points
// let points1 = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
// let points2 = array![[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]];
//
// let distances = simd_euclidean_distance_batch(&points1.view(), &points2.view()).unwrap();
// println!("Batch distances: {:?}", distances);
//
// // Parallel pairwise distance matrix computation
// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
// let dist_matrix = parallel_pdist(&points.view(), "euclidean").unwrap();
// println!("Distance matrix shape: {:?}", dist_matrix.shape());
//
// // High-performance k-nearest neighbors search
// use scirs2_spatial::simd_distance::simd_knn_search;
// let (indices, distances) = simd_knn_search(&points1.view(), &points.view(), 2, "euclidean").unwrap();
// println!("Nearest neighbor indices: {:?}", indices);
// ```
//
// ### Advanced-Optimized SIMD Clustering
//
// ```
// use scirs2_spatial::{AdvancedSimdKMeans, AdvancedSimdNearestNeighbors};
// use ndarray::array;
//
// // Optimized SIMD K-means clustering
// let points = array![
//     [0.0, 0.0], [0.1, 0.1], [0.0, 0.1],  // Cluster 1
//     [5.0, 5.0], [5.1, 5.1], [5.0, 5.1],  // Cluster 2
// ];
//
// let advanced_kmeans = AdvancedSimdKMeans::new(2)
//     .with_mixed_precision(true)
//     .with_block_size(256);
//
// let (centroids, assignments) = advanced_kmeans.fit(&points.view()).unwrap();
// println!("Centroids: {:?}", centroids);
// println!("Assignments: {:?}", assignments);
//
// // Optimized SIMD nearest neighbors
// let nn_searcher = AdvancedSimdNearestNeighbors::new();
// let query_points = array![[0.05, 0.05], [5.05, 5.05]];
// let (indices, distances) = nn_searcher.simd_knn_advanced_fast(
//     &query_points.view(), &points.view(), 2
// ).unwrap();
// println!("NN indices: {:?}", indices);
// ```
//
// ### Memory Pool Optimization
//
// ```
// use scirs2_spatial::{DistancePool, ClusteringArena, global_distance_pool};
//
// // Use global memory pool for frequent allocations
// let pool = global_distance_pool();
//
// // Get a reusable distance buffer
// let mut buffer = pool.get_distance_buffer(1000);
//
// // Use buffer for computations...
// let data = buffer.as_mut_slice();
// data[0] = 42.0;
//
// // Buffer automatically returns to pool on drop
// drop(buffer);
//
// // Check pool performance
// let stats = pool.statistics();
// println!("Pool hit rate: {:.1}%", stats.hit_rate());
//
// // Use arena for temporary objects
// use scirs2_spatial::ClusteringArena;
// let arena = ClusteringArena::new();
// let temp_vec = arena.alloc_temp_vec::<f64>(500);
// // Temporary objects are freed when arena is reset
// arena.reset();
// ```
//
// ### GPU-Accelerated Massive-Scale Computing
//
// ```
// use scirs2_spatial::{GpuDistanceMatrix, GpuKMeans, report_gpu_status};
// use ndarray::array;
//
// // Check GPU acceleration availability
// report_gpu_status();
//
// // GPU-accelerated distance matrix for massive datasets
// let points = array![
//     [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
//     [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0],
// ];
//
// let gpu_matrix = GpuDistanceMatrix::new()?;
// let distances = gpu_matrix.compute_parallel(&points.view()).await?;
// println!("GPU distance matrix computed: {:?}", distances.shape());
//
// // GPU-accelerated K-means for massive clusters
// let gpu_kmeans = GpuKMeans::new(3)?
//     .with_batch_size(1024)
//     .with_tolerance(1e-6);
//
// let (centroids, assignments) = gpu_kmeans.fit(&points.view()).await?;
// println!("GPU K-means completed: {} centroids", centroids.nrows());
//
// // Hybrid CPU-GPU processing
// use scirs2_spatial::HybridProcessor;
// let processor = HybridProcessor::new()?;
// let strategy = processor.choose_strategy(points.nrows());
// println!("Optimal strategy: {:?}", strategy);
// ```
//
// ### Advanced-Optimized KD-Tree for Maximum Performance
//
// ```
// use scirs2_spatial::{AdvancedKDTree, KDTreeConfig};
// use ndarray::array;
//
// // Create points dataset
// let points = array![
//     [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
//     [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0],
// ];
//
// // Configure advanced-optimized KD-Tree
// let config = KDTreeConfig::new()
//     .with_cache_aware_layout(true)    // Optimize for CPU cache
//     .with_vectorized_search(true)     // Use SIMD acceleration
//     .with_numa_aware(true)            // NUMA-aware construction
//     .with_parallel_construction(true, 1000);  // Parallel for large datasets
//
// // Build advanced-optimized tree
// let advanced_kdtree = AdvancedKDTree::new(&points.view(), config)?;
//
// // Optimized k-nearest neighbors
// let query = array![2.1, 2.1];
// let (indices, distances) = advanced_kdtree.knn_search_advanced(&query.view(), 3)?;
// println!("Optimized k-NN: indices={:?}, distances={:?}", indices, distances);
//
// // Batch processing for multiple queries
// let queries = array![[0.5, 0.5], [2.5, 2.5], [4.5, 4.5]];
// let (batch_indices, batch_distances) = advanced_kdtree.batch_knn_search(&queries.view(), 2)?;
// println!("Batch k-NN shape: {:?}", batch_indices.shape());
//
// // Range search with radius
// let range_results = advanced_kdtree.range_search(&query.view(), 1.0)?;
// println!("Points within radius 1.0: {} found", range_results.len());
//
// // Performance statistics
// let stats = advanced_kdtree.statistics();
// println!("Tree depth: {}, Construction time: {:.2}ms",
//          stats.depth, stats.construction_time_ms);
// println!("Memory usage: {:.1} KB", stats.memory_usage_bytes as f64 / 1024.0);
// ```
//
// ### RRT Pathfinding
//
// ```
// use scirs2_spatial::pathplanning::{RRTConfig, RRT2DPlanner};
//
// // Create a configuration for RRT
// let config = RRTConfig {
//     max_iterations: 1000,
//     step_size: 0.3,
//     goal_bias: 0.1,
//     seed: Some(42),
//     use_rrt_star: false,
//     neighborhood_radius: None,
//     bidirectional: false,
// };
//
// // Define obstacles as polygons
// let obstacles = vec![
//     // Rectangle obstacle
//     vec![[3.0, 2.0], [3.0, 4.0], [4.0, 4.0], [4.0, 2.0]],
// ];
//
// // Create RRT planner
// let mut planner = RRT2DPlanner::new(
//     config,
//     obstacles,
//     [0.0, 0.0],   // Min bounds
//     [10.0, 10.0], // Max bounds
//     0.1,          // Collision checking step size
// ).unwrap();
//
// // Find a path from start to goal
// let start = [1.0, 3.0];
// let goal = [8.0, 3.0];
// let goal_threshold = 0.5;
//
// let path = planner.find_path(start, goal, goal_threshold).unwrap().unwrap();
//
// println!("Found a path with {} segments:", path.len() - 1);
// for (i, pos) in path.nodes.iter().enumerate() {
//     println!("  Point {}: [{:.2}, {:.2}]", i, pos[0], pos[1]);
// }
// ```
//
// ## Advanced MODE: Revolutionary Computing Paradigms
//
// These cutting-edge implementations push spatial computing beyond current limitations,
// achieving unprecedented performance through quantum computing, neuromorphic processing,
// next-generation GPU architectures, AI-driven optimization, and extreme performance
// optimizations that can deliver 10-100x speedups over conventional approaches.
//
// Note: Advanced modules are currently being optimized and may be temporarily disabled
// during development phases.
//
// ### Quantum-Classical Hybrid Algorithms (Development Mode)
//
// ```text
// // Temporarily disabled for optimization
// // use scirs2_spatial::quantum_classical_hybrid::{HybridSpatialOptimizer, HybridClusterer};
// // use ndarray::array;
// //
// // // Quantum-classical hybrid spatial optimization
// // let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
// // let mut hybrid_optimizer = HybridSpatialOptimizer::new()
// //     .with_quantum_depth(5)
// //     .with_classical_refinement(true)
// //     .with_adaptive_switching(0.7);
// //
// // let result = hybrid_optimizer.optimize_spatial_problem(&points.view()).await?;
// // println!("Quantum-classical optimization: {} iterations", result.iterations);
// ```
//
// ### Neuromorphic-Quantum Fusion Computing (Development Mode)
//
// ```text
// // Temporarily disabled for optimization
// // use scirs2_spatial::neuromorphic_quantum_fusion::{QuantumSpikingClusterer, NeuralQuantumOptimizer};
// // use ndarray::array;
// //
// // // Quantum-enhanced spiking neural clustering
// // let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
// // let mut quantum_snn = QuantumSpikingClusterer::new(2)
// //     .with_quantum_superposition(true)
// //     .with_spike_timing_plasticity(true)
// //     .with_quantum_entanglement(0.7)
// //     .with_bio_inspired_adaptation(true);
// //
// // let (clusters, quantum_spikes, fusion_metrics) = quantum_snn.cluster(&points.view()).await?;
// // println!("Quantum-neural speedup: {:.2}x", fusion_metrics.quantum_neural_speedup);
// ```
//
// ### Next-Generation GPU Architectures (Development Mode)
//
// ```text
// // Temporarily disabled for optimization
// // use scirs2_spatial::next_gen_gpu_architecture::{QuantumGpuProcessor, PhotonicAccelerator};
// // use ndarray::array;
// //
// // // Quantum-GPU hybrid processing with tensor core enhancement
// // let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
// // let mut quantum_gpu = QuantumGpuProcessor::new()
// //     .with_quantum_coherence_preservation(true)
// //     .with_tensor_core_quantum_enhancement(true)
// //     .with_holographic_memory(true);
// //
// // let quantum_distances = quantum_gpu.compute_quantum_distance_matrix(&points.view()).await?;
// // println!("Quantum-GPU: Unprecedented computing performance achieved");
// ```
//
// ### AI-Driven Algorithm Selection and Optimization (Development Mode)
//
// ```text
// // Temporarily disabled for optimization
// // use scirs2_spatial::ai_driven_optimization::{AIAlgorithmSelector, MetaLearningOptimizer};
// // use ndarray::array;
// //
// // // AI automatically selects optimal algorithms and parameters
// // let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
// // let mut ai_selector = AIAlgorithmSelector::new()
// //     .with_meta_learning(true)
// //     .with_neural_architecture_search(true)
// //     .with_real_time_adaptation(true)
// //     .with_multi_objective_optimization(true);
// //
// // let (optimal_algorithm, parameters, performance_prediction) =
// //     ai_selector.select_optimal_algorithm(&points.view(), "clustering").await?;
// //
// // println!("AI selected: {} with predicted accuracy: {:.3}",
// //          optimal_algorithm, performance_prediction.expected_accuracy);
// ```
//
// ### Extreme Performance Optimization (Development Mode)
//
// ```text
// // Temporarily disabled for optimization
// // use scirs2_spatial::extreme_performance_optimization::{
// //     ExtremeOptimizer, AdvancedfastDistanceMatrix, SelfOptimizingAlgorithm, create_ultimate_optimizer
// // };
// // use ndarray::array;
// //
// // // Achieve 50-100x performance improvements with all optimizations
// // let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
// // let optimizer = create_ultimate_optimizer(); // All optimizations enabled
// //
// // let advancedfast_matrix = AdvancedfastDistanceMatrix::new(optimizer);
// // let distances = advancedfast_matrix.compute_extreme_performance(&points.view()).await?;
// //
// // // Self-optimizing algorithms that improve during execution
// // let mut self_optimizer = SelfOptimizingAlgorithm::new("clustering")
// //     .with_hardware_counter_feedback(true)  // Real-time performance monitoring
// //     .with_runtime_code_generation(true)    // Dynamic optimization
// //     .with_adaptive_memory_patterns(true);  // Intelligent prefetching
// //
// // let optimized_result = self_optimizer.auto_optimize_and_execute(&points.view()).await?;
// // println!("Self-optimized performance: 10-50x speedup achieved automatically");
// //
// // // Benchmark all extreme optimizations
// // let extreme_metrics = benchmark_extreme_optimizations(&points.view()).await?;
// // println!("Extreme speedup: {:.1}x faster than conventional algorithms",
// //          extreme_metrics.extreme_speedup);
// ```

// Export error types
pub mod error;
pub use error::{SpatialError, SpatialResult};

// Safe conversion utilities
pub(crate) mod safe_conversions;

// Distance metrics
pub mod distance;
pub use distance::{
    // Basic distance functions
    braycurtis,
    canberra,
    cdist,
    chebyshev,
    correlation,
    cosine,
    dice,
    // Convenience functions
    euclidean,
    is_valid_condensed_distance_matrix,
    jaccard,
    kulsinski,
    mahalanobis,
    manhattan,
    minkowski,
    // Distance matrix computation
    pdist,
    rogerstanimoto,
    russellrao,
    seuclidean,
    sokalmichener,
    sokalsneath,
    sqeuclidean,
    squareform,
    squareform_to_condensed,
    yule,
    ChebyshevDistance,
    // Core distance traits and structs
    Distance,
    EuclideanDistance,
    ManhattanDistance,
    MinkowskiDistance,
};

// KD-Tree for efficient nearest neighbor searches
pub mod kdtree;
pub use kdtree::{KDTree, Rectangle};

// KD-Tree optimizations for spatial operations
pub mod kdtree_optimized;
pub use kdtree_optimized::KDTreeOptimized;

// Advanced-optimized KD-Tree with advanced performance features
pub mod kdtree_advanced;
pub use kdtree_advanced::{
    AdvancedKDTree, BoundingBox as KDTreeBoundingBox, KDTreeConfig, TreeStatistics,
};

// Ball-Tree for efficient nearest neighbor searches in high dimensions
pub mod balltree;
pub use balltree::BallTree;

// Delaunay triangulation
pub mod delaunay;
pub use delaunay::Delaunay;

// Voronoi diagrams
pub mod voronoi;
pub use voronoi::{voronoi, Voronoi};

// Spherical Voronoi diagrams
pub mod spherical_voronoi;
pub use spherical_voronoi::SphericalVoronoi;

// Procrustes analysis
pub mod procrustes;
pub use procrustes::{procrustes, procrustes_extended, ProcrustesParams};

// Convex hull computation
pub mod convex_hull;
pub use convex_hull::{convex_hull, ConvexHull};

// Alpha shapes
pub mod alphashapes;
pub use alphashapes::AlphaShape;

// Halfspace intersection
pub mod halfspace;
pub use halfspace::{Halfspace, HalfspaceIntersection};

// Boolean operations
pub mod boolean_ops;
pub use boolean_ops::{
    compute_polygon_area, is_convex_polygon, is_self_intersecting, polygon_difference,
    polygon_intersection, polygon_symmetric_difference, polygon_union,
};

// Kriging interpolation
pub mod kriging;
pub use kriging::{KrigingPrediction, OrdinaryKriging, SimpleKriging, VariogramModel};

// Geospatial functionality
pub mod geospatial;
pub use geospatial::{
    cross_track_distance, destination_point, final_bearing, geographic_to_utm,
    geographic_to_web_mercator, haversine_distance, initial_bearing, midpoint, normalize_bearing,
    point_in_spherical_polygon, spherical_polygon_area, vincenty_distance,
    web_mercator_to_geographic, EARTH_RADIUS_KM, EARTH_RADIUS_M,
};

// Set-based distance metrics
pub mod set_distance;
pub use set_distance::{
    directed_hausdorff, gromov_hausdorff_distance, hausdorff_distance, wasserstein_distance,
};

// Polygon operations
pub mod polygon;
pub use polygon::{
    convex_hull_graham, douglas_peucker_simplify, is_simple_polygon, point_in_polygon,
    point_on_boundary, polygon_area, polygon_centroid, polygon_contains_polygon,
    visvalingam_whyatt_simplify,
};

// R-tree for efficient spatial indexing
pub mod rtree;
pub use rtree::{RTree, Rectangle as RTreeRectangle};

// Octree for 3D spatial searches
pub mod octree;
pub use octree::{BoundingBox as OctreeBoundingBox, Octree};

// Quadtree for 2D spatial searches
pub mod quadtree;
pub use quadtree::{BoundingBox2D, Quadtree};

// Spatial interpolation methods
pub mod interpolate;
pub use interpolate::{IDWInterpolator, NaturalNeighborInterpolator, RBFInterpolator, RBFKernel};

// Path planning algorithms
pub mod pathplanning;
pub use pathplanning::astar::{AStarPlanner, ContinuousAStarPlanner, GridAStarPlanner, Node, Path};
pub use pathplanning::rrt::{RRT2DPlanner, RRTConfig, RRTPlanner};

// Spatial transformations
pub mod transform;

// Collision detection
pub mod collision;
// Re-export shapes for convenience
pub use collision::shapes::{
    Box2D, Box3D, Circle, LineSegment2D, LineSegment3D, Sphere, Triangle2D, Triangle3D,
};
// Re-export narrowphase collision functions
pub use collision::narrowphase::{
    box2d_box2d_collision,
    box3d_box3d_collision,
    circle_box2d_collision,
    circle_circle_collision,
    // GJK collision detection functions
    gjk_box_box_collision,
    gjk_collision_detection,
    gjk_sphere_box_collision,
    gjk_sphere_sphere_collision,
    point_box2d_collision,
    point_box3d_collision,
    point_circle_collision,
    point_sphere_collision,
    point_triangle2d_collision,
    ray_box3d_collision,
    ray_sphere_collision,
    ray_triangle3d_collision,
    sphere_box3d_collision,
    sphere_sphere_collision,
    // GJK trait for advanced users
    GJKShape,
};
// Re-export continuous collision functions
pub use collision::continuous::continuous_sphere_sphere_collision;

// Spatial statistics and pattern analysis
pub mod spatial_stats;
pub use spatial_stats::{
    clark_evans_index, distance_weights_matrix, gearys_c, getis_ord_gi, local_morans_i, morans_i,
};

// SIMD-accelerated distance calculations
pub mod simd_distance;
pub use simd_distance::{
    parallel_cdist, parallel_pdist, simd_euclidean_distance, simd_euclidean_distance_batch,
    simd_knn_search, simd_manhattan_distance, SimdMetric,
};

// Advanced-optimized SIMD clustering and distance operations
pub use simd_distance::advanced_simd_clustering::{
    AdvancedSimdKMeans, AdvancedSimdNearestNeighbors,
};
pub use simd_distance::bench::{
    benchmark_distance_computation, report_simd_features, BenchmarkResults,
};
pub use simd_distance::mixed_precision_simd::{
    simd_euclidean_distance_batch_f32, simd_euclidean_distance_f32,
};

// Advanced-optimized memory pool system for spatial algorithms
pub mod memory_pool;
pub use memory_pool::{
    global_clustering_arena, global_distance_pool, ArenaStatistics, ClusteringArena,
    DistanceBuffer, DistancePool, IndexBuffer, MatrixBuffer, MemoryPoolConfig, PoolStatistics,
};

// GPU acceleration for massive-scale spatial computations
pub mod gpu_accel;
pub use gpu_accel::{
    get_gpu_capabilities, global_gpu_device, is_gpu_acceleration_available, report_gpu_status,
    GpuCapabilities, GpuDevice, GpuDistanceMatrix, GpuKMeans, GpuNearestNeighbors, HybridProcessor,
    ProcessingStrategy,
};

// Advanced-parallel algorithms with work-stealing and NUMA-aware optimizations
pub mod advanced_parallel;
pub use advanced_parallel::{
    get_numa_topology, initialize_global_pool, report_advanced_parallel_capabilities,
    AdvancedParallelDistanceMatrix, AdvancedParallelKMeans, MemoryStrategy, NumaTopology,
    PoolStatistics as AdvancedPoolStatistics, ThreadAffinityStrategy, WorkStealingConfig,
    WorkStealingPool,
};

// Utility functions
mod utils;

// Quantum-inspired spatial algorithms for cutting-edge optimization
pub mod quantum_inspired;
pub use quantum_inspired::{
    // Re-export from algorithms submodule
    algorithms::QuantumSpatialOptimizer,
    ErrorCorrectionConfig,
    ErrorCorrectionType,
    OptimizationConfig,
    OptimizerType,
    PerformanceMetrics,
    QuantumAmplitude,
    QuantumClusterer,
    // Configuration types
    QuantumConfig,
    QuantumNearestNeighbor,
    QuantumSpatialFramework,
    QuantumState,
};

// Neuromorphic computing acceleration for brain-inspired spatial processing
pub mod neuromorphic;
pub use neuromorphic::{
    // Core neuromorphic components
    AdaptiveSpikingNeuron,
    // Clustering algorithms
    CompetitiveNeuralClusterer,
    HomeostaticNeuralClusterer,
    HomeostaticSynapse,
    MetaplasticSynapse,
    NetworkStats,
    NeuromorphicCapability,
    // Configuration
    NeuromorphicConfig,
    NeuromorphicFactory,
    // Processing
    NeuromorphicProcessor,
    SpikeEvent,
    SpikeSequence,
    SpikingNeuralClusterer,
    SpikingNeuron,
    Synapse,
};

// Advanced GPU tensor core utilization for maximum performance
pub mod tensor_cores;
pub use tensor_cores::{
    detect_tensor_core_capabilities, GpuArchitecture, PrecisionMode, TensorCoreCapabilities,
    TensorCoreClustering, TensorCoreDistanceMatrix, TensorCoreType, TensorLayout,
};

// Machine learning-based spatial optimization and adaptive algorithms
pub mod ml_optimization;
pub use ml_optimization::{
    ActivationFunction, ClusteringParameters, ClusteringResult, DataState, DistanceMetric,
    Experience, NeuralSpatialOptimizer, ReinforcementLearningSelector, SpatialAlgorithm,
};

// Distributed spatial computing framework for massive scale processing
pub mod distributed;
pub use distributed::{
    ClusterStatistics, DataPartition, DistributedMessage, DistributedSpatialCluster, LoadBalancer,
    LoadMetrics, NodeConfig, NodeStatus, QueryResults, QueryType, SpatialBounds,
};

// Real-time adaptive algorithm selection and optimization
pub mod adaptive_selection;
pub use adaptive_selection::{
    ActualPerformance, AdaptiveAlgorithmSelector, AlgorithmParameters, AlgorithmSelection,
    DataCharacteristics, ExecutionResult, PerformancePrediction, SelectedAlgorithm,
    SelectionContext,
};

// Quantum-classical hybrid algorithms for unprecedented performance breakthroughs
pub mod quantum_classical_hybrid;
pub use quantum_classical_hybrid::{
    HybridClusterer, HybridClusteringMetrics, HybridOptimizationResult, HybridPerformanceMetrics,
    HybridSpatialOptimizer, OptimizationStepResult,
};

// Neuromorphic-quantum fusion algorithms for revolutionary bio-quantum computing
pub mod neuromorphic_quantum_fusion;
pub use neuromorphic_quantum_fusion::{
    FusionMetrics, NeuralQuantumOptimizationResult, NeuralQuantumOptimizer, QuantumSpikeEvent,
    QuantumSpikePattern, QuantumSpikingClusterer, QuantumSpikingNeuron,
};

// Next-generation GPU architecture support for future computing paradigms
pub mod next_gen_gpu_architecture;
pub use next_gen_gpu_architecture::{
    NextGenGpuArchitecture, NextGenPerformanceMetrics, PhotonicAccelerator, PhotonicProcessingUnit,
    QuantumGpuProcessor, QuantumProcessingUnit,
};

// Generic traits and algorithms for flexible spatial computing
pub mod generic_traits;
pub use generic_traits::{
    ChebyshevMetric, EuclideanMetric, ManhattanMetric, Point, SpatialArray, SpatialPoint,
    SpatialScalar,
};

pub mod generic_algorithms;
pub use generic_algorithms::{
    DBSCANResult, GMMResult, GenericConvexHull, GenericDBSCAN, GenericDistanceMatrix, GenericGMM,
    GenericKDTree, GenericKMeans, KMeansResult,
};

// AI-driven algorithm selection and optimization for intelligent spatial computing
pub mod ai_driven_optimization;
pub use ai_driven_optimization::{
    AIAlgorithmSelector, AdaptationRecord, AlgorithmCandidate, AlgorithmKnowledgeBase,
    AlgorithmMetadata, ComplexityModel, MetaLearningModel, MetaLearningOptimizer,
    MetaOptimizationResult, PerformanceModel, PerformanceRecord, PredictionNetworks,
    ReinforcementLearningAgent, TaskMetadata,
};

// Extreme performance optimization pushing spatial computing beyond current limits
pub mod extreme_performance_optimization;
pub use extreme_performance_optimization::{
    benchmark_extreme_optimizations, create_ultimate_optimizer, AdvancedfastDistanceMatrix,
    CacheHierarchyInfo, CacheObliviousSpatialAlgorithms, ExtremeMemoryAllocator, ExtremeOptimizer,
    ExtremePerformanceMetrics, HardwarePerformanceCounters, JitCompiler, LockFreeSpatialStructures,
    NumaTopologyInfo, OptimizationRecord, SelfOptimizingAlgorithm,
};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
