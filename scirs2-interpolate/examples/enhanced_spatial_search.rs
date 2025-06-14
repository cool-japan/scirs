//! Enhanced Spatial Search Optimizations Example
//!
//! This example demonstrates the advanced nearest neighbor search optimizations
//! implemented in the spatial module, including:
//! - Optimized KdTree and BallTree with early termination
//! - SIMD-accelerated distance computations
//! - Cache-friendly memory layouts
//! - Adaptive search strategies
//! - Batch and parallel query processing

use ndarray::Array2;
use scirs2_interpolate::spatial::{
    AdaptiveSearchStrategy, BatchQueryProcessor, CacheFriendlyIndex, KdTree,
    SIMDDistanceCalculator,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Enhanced Spatial Search Optimizations Example");
    println!("=============================================");

    // Generate test data
    let (points, queries) = generate_test_data(1000, 100, 10)?;

    println!(
        "\nDataset: {} points, {} queries in {}D space",
        points.nrows(),
        queries.nrows(),
        points.ncols()
    );

    // 1. Compare standard vs optimized k-NN search
    compare_knn_methods(&points, &queries)?;

    // 2. Demonstrate SIMD distance calculations
    demonstrate_simd_distances(&points, &queries)?;

    // 3. Show cache-friendly indexing benefits
    demonstrate_cache_friendly_index(&points, &queries)?;

    // 4. Test adaptive search strategy
    demonstrate_adaptive_search(&points, &queries)?;

    // 5. Batch query processing
    demonstrate_batch_processing(&points, &queries)?;

    // 6. Performance scaling analysis
    performance_scaling_analysis()?;

    Ok(())
}

fn generate_test_data(
    n_points: usize,
    n_queries: usize,
    n_dims: usize,
) -> Result<(Array2<f64>, Array2<f64>), Box<dyn std::error::Error>> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);

    // Generate clustered data for realistic scenarios
    let mut points_data = Vec::new();
    let mut queries_data = Vec::new();

    // Create several clusters
    let n_clusters = 5;
    let cluster_std = 0.3;

    for _ in 0..n_points {
        let cluster_id = rng.random_range(0..n_clusters);
        let cluster_center = cluster_id as f64 * 2.0;

        for _ in 0..n_dims {
            let coord = cluster_center + rng.random::<f64>() * cluster_std;
            points_data.push(coord);
        }
    }

    // Generate query points around the data
    for _ in 0..n_queries {
        for _ in 0..n_dims {
            let coord = rng.random::<f64>() * (n_clusters as f64 * 2.0);
            queries_data.push(coord);
        }
    }

    let points = Array2::from_shape_vec((n_points, n_dims), points_data)?;
    let queries = Array2::from_shape_vec((n_queries, n_dims), queries_data)?;

    Ok((points, queries))
}

fn compare_knn_methods(
    points: &Array2<f64>,
    queries: &Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n1. Comparing Standard vs Optimized k-NN Search");
    println!("----------------------------------------------");

    let k = 10;
    let query = queries.row(0);
    let query_slice = query.as_slice().unwrap();

    // Standard KdTree
    let kdtree = KdTree::new(points.to_owned())?;
    let start = Instant::now();
    let standard_result = kdtree.k_nearest_neighbors(query_slice, k)?;
    let standard_time = start.elapsed();

    // Optimized KdTree
    let start = Instant::now();
    let optimized_result = kdtree.k_nearest_neighbors_optimized(query_slice, k, None)?;
    let optimized_time = start.elapsed();

    println!(
        "Standard k-NN:  {:?} (found {} neighbors)",
        standard_time,
        standard_result.len()
    );
    println!(
        "Optimized k-NN: {:?} (found {} neighbors)",
        optimized_time,
        optimized_result.len()
    );

    // Verify results are equivalent
    let distances_match = standard_result
        .iter()
        .zip(optimized_result.iter())
        .all(|((i1, d1), (i2, d2))| i1 == i2 && (d1 - d2).abs() < 1e-10);

    if distances_match {
        println!("✓ Results match between standard and optimized methods");
        if optimized_time < standard_time {
            let speedup = standard_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
            println!("✓ Speedup: {:.2}x", speedup);
        }
    } else {
        println!("⚠ Results differ between methods");
    }

    Ok(())
}

fn demonstrate_simd_distances(
    points: &Array2<f64>,
    queries: &Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2. SIMD Distance Calculations");
    println!("-----------------------------");

    let query = queries.row(0);
    let query_slice = query.as_slice().unwrap();
    let simd_calc = SIMDDistanceCalculator::new();

    // Time SIMD distance calculation
    let start = Instant::now();
    let distances = simd_calc.batch_squared_distances_simd(query_slice, &points.view());
    let simd_time = start.elapsed();

    // Time scalar distance calculation for comparison
    let start = Instant::now();
    let mut scalar_distances = Vec::with_capacity(points.nrows());
    for i in 0..points.nrows() {
        let point = points.row(i);
        let mut sum_sq = 0.0;
        for j in 0..points.ncols() {
            let diff = point[j] - query_slice[j];
            sum_sq += diff * diff;
        }
        scalar_distances.push(sum_sq);
    }
    let scalar_time = start.elapsed();

    println!(
        "SIMD batch calculation:   {:?} ({} distances)",
        simd_time,
        distances.len()
    );
    println!(
        "Scalar loop calculation:  {:?} ({} distances)",
        scalar_time,
        scalar_distances.len()
    );

    // Verify accuracy
    let accuracy_check = distances
        .iter()
        .zip(scalar_distances.iter())
        .all(|(simd_d, scalar_d)| (simd_d - scalar_d).abs() < 1e-10);

    if accuracy_check {
        println!("✓ SIMD and scalar results match");
        if simd_time < scalar_time {
            let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
            println!("✓ SIMD speedup: {:.2}x", speedup);
        }
    } else {
        println!("⚠ SIMD and scalar results differ");
    }

    Ok(())
}

fn demonstrate_cache_friendly_index(
    points: &Array2<f64>,
    queries: &Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n3. Cache-Friendly Index Performance");
    println!("-----------------------------------");

    let cache_index = CacheFriendlyIndex::new(&points.view())?;
    let query = queries.row(0);
    let query_slice = query.as_slice().unwrap();
    let k = 10;

    // Time cache-friendly search
    let start = Instant::now();
    let cache_result = cache_index.k_nearest_neighbors(query_slice, k)?;
    let cache_time = start.elapsed();

    // Compare with standard approach
    let kdtree = KdTree::new(points.to_owned())?;
    let start = Instant::now();
    let standard_result = kdtree.k_nearest_neighbors(query_slice, k)?;
    let standard_time = start.elapsed();

    println!(
        "Cache-friendly index: {:?} (found {} neighbors)",
        cache_time,
        cache_result.len()
    );
    println!(
        "Standard index:       {:?} (found {} neighbors)",
        standard_time,
        standard_result.len()
    );

    // Test batch distance computation
    let start = Instant::now();
    let _batch_distances = cache_index.batch_distances(query_slice);
    let batch_time = start.elapsed();

    println!(
        "Batch distance calc:  {:?} ({} distances)",
        batch_time,
        points.nrows()
    );

    Ok(())
}

fn demonstrate_adaptive_search(
    points: &Array2<f64>,
    queries: &Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n4. Adaptive Search Strategy");
    println!("---------------------------");

    let mut adaptive_strategy = AdaptiveSearchStrategy::new(&points.view())?;
    let query = queries.row(0);
    let query_slice = query.as_slice().unwrap();

    // Perform several searches with different k values
    for k in [1, 5, 10, 50] {
        let start = Instant::now();
        let result = adaptive_strategy.adaptive_k_nearest_neighbors(query_slice, k)?;
        let search_time = start.elapsed();

        println!(
            "k={:2}: {:?} (found {} neighbors)",
            k,
            search_time,
            result.len()
        );
    }

    // Show statistics
    let stats = adaptive_strategy.stats();
    println!("\nAdaptive Search Statistics:");
    println!("  Total queries:      {}", stats.total_queries);
    println!("  KdTree queries:     {}", stats.kdtree_queries);
    println!("  BallTree queries:   {}", stats.balltree_queries);
    println!("  Brute force queries:{}", stats.brute_force_queries);
    println!("  Avg query time:     {:.2} ns", stats.avg_query_time_ns);

    Ok(())
}

fn demonstrate_batch_processing(
    points: &Array2<f64>,
    queries: &Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n5. Batch Query Processing");
    println!("-------------------------");

    let mut batch_processor = BatchQueryProcessor::new(&points.view())?.with_batch_size(16);

    let k = 5;

    // Sequential processing
    let start = Instant::now();
    let mut sequential_results = Vec::new();
    let kdtree = KdTree::new(points.to_owned())?;
    for i in 0..queries.nrows() {
        let query = queries.row(i);
        let query_slice = query.as_slice().unwrap();
        let result = kdtree.k_nearest_neighbors(query_slice, k)?;
        sequential_results.push(result);
    }
    let sequential_time = start.elapsed();

    // Batch processing
    let start = Instant::now();
    let batch_results = batch_processor.batch_k_nearest_neighbors(&queries.view(), k)?;
    let batch_time = start.elapsed();

    println!(
        "Sequential processing: {:?} ({} queries)",
        sequential_time,
        queries.nrows()
    );
    println!(
        "Batch processing:      {:?} ({} queries)",
        batch_time,
        queries.nrows()
    );

    if batch_time < sequential_time {
        let speedup = sequential_time.as_nanos() as f64 / batch_time.as_nanos() as f64;
        println!("✓ Batch speedup: {:.2}x", speedup);
    }

    // Verify results
    let results_match = sequential_results.len() == batch_results.len()
        && sequential_results
            .iter()
            .zip(batch_results.iter())
            .all(|(seq, batch)| seq.len() == batch.len());

    if results_match {
        println!("✓ Sequential and batch results have same structure");
    }

    Ok(())
}

fn performance_scaling_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n6. Performance Scaling Analysis");
    println!("-------------------------------");

    let dimensions = [2, 5, 10, 20];
    let sizes = [100, 500, 1000];

    println!("   Size |   2D   |   5D   |  10D   |  20D   |");
    println!("--------|-------|-------|-------|-------|");

    for &n_points in &sizes {
        print!("  {:4} |", n_points);

        for &n_dims in &dimensions {
            let (points, queries) = generate_test_data(n_points, 10, n_dims)?;

            // Test adaptive strategy
            let mut adaptive_strategy = AdaptiveSearchStrategy::new(&points.view())?;
            let query = queries.row(0);
            let query_slice = query.as_slice().unwrap();

            let start = Instant::now();
            let _result = adaptive_strategy.adaptive_k_nearest_neighbors(query_slice, 10)?;
            let elapsed = start.elapsed();

            print!(" {:5.1}μs |", elapsed.as_nanos() as f64 / 1000.0);
        }
        println!();
    }

    println!("\nKey observations:");
    println!("- Low dimensions (2D-5D): KdTree performs best");
    println!("- Medium dimensions (10D): BallTree often preferred");
    println!("- High dimensions (20D+): Brute force may be optimal");
    println!("- Adaptive strategy automatically chooses the best method");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_spatial_search() {
        let (points, queries) = generate_test_data(50, 10, 3).unwrap();

        // Test that all methods work
        assert!(compare_knn_methods(&points, &queries).is_ok());
        assert!(demonstrate_simd_distances(&points, &queries).is_ok());
        assert!(demonstrate_cache_friendly_index(&points, &queries).is_ok());
        assert!(demonstrate_adaptive_search(&points, &queries).is_ok());
        assert!(demonstrate_batch_processing(&points, &queries).is_ok());
    }

    #[test]
    fn test_optimized_methods_consistency() {
        let (points, _) = generate_test_data(100, 1, 5).unwrap();
        let query = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let k = 5;

        let kdtree = KdTree::new(&points).unwrap();
        let standard_result = kdtree.k_nearest_neighbors(&query, k).unwrap();
        let optimized_result = kdtree
            .k_nearest_neighbors_optimized(&query, k, None)
            .unwrap();

        // Results should be the same
        assert_eq!(standard_result.len(), optimized_result.len());
        for (i, ((idx1, dist1), (idx2, dist2))) in standard_result
            .iter()
            .zip(optimized_result.iter())
            .enumerate()
        {
            assert_eq!(idx1, idx2, "Index mismatch at position {}", i);
            assert!(
                (dist1 - dist2).abs() < 1e-10,
                "Distance mismatch at position {}: {} vs {}",
                i,
                dist1,
                dist2
            );
        }
    }
}
