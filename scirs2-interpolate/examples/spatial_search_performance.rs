use ndarray::{Array1, Array2};
use scirs2_interpolate::local::mls::{MovingLeastSquares, PolynomialBasis, WeightFunction};
use scirs2_interpolate::spatial::{BallTree, KdTree};
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Spatial Search Performance Comparison");
    println!("-------------------------------------\n");

    // Generate test data
    println!("Generating test data...");
    let n_points = 10000;
    let dim = 3;

    let mut points_vec = Vec::with_capacity(n_points * dim);
    let mut values_vec = Vec::with_capacity(n_points);

    let _rng = rand::rng();

    for _ in 0..n_points {
        // Generate random points in [0, 1]^dim
        for _ in 0..dim {
            points_vec.push(rand::random::<f64>());
        }

        // Generate a test function: f(x,y,z) = sin(2πx) * cos(2πy) * z^2
        let x = points_vec[points_vec.len() - dim];
        let y = points_vec[points_vec.len() - dim + 1];
        let z = points_vec[points_vec.len() - dim + 2];

        let value = (2.0 * std::f64::consts::PI * x).sin()
            * (2.0 * std::f64::consts::PI * y).cos()
            * z.powi(2);

        values_vec.push(value);
    }

    let points = Array2::from_shape_vec((n_points, dim), points_vec)?;
    let values = Array1::from_vec(values_vec);

    // Generate query points
    let n_queries = 100;
    let mut query_vec = Vec::with_capacity(n_queries * dim);

    for _ in 0..n_queries {
        for _ in 0..dim {
            query_vec.push(rand::random::<f64>());
        }
    }

    let query_points = Array2::from_shape_vec((n_queries, dim), query_vec)?;

    // Build spatial data structures
    println!("Building spatial data structures...");

    let build_start = Instant::now();
    let kdtree = KdTree::new(points.clone())?;
    let kdtree_build_time = build_start.elapsed();

    let build_start = Instant::now();
    let balltree = BallTree::new(points.clone())?;
    let balltree_build_time = build_start.elapsed();

    println!("  KD-Tree build time:    {:?}", kdtree_build_time);
    println!("  Ball Tree build time:  {:?}", balltree_build_time);

    // Compare nearest neighbor search performance
    println!("\nNearest neighbor search performance (100 queries):");

    // Linear search (brute force)
    let start = Instant::now();
    let mut linear_results = Vec::with_capacity(n_queries);

    for i in 0..n_queries {
        let query = query_points.slice(ndarray::s![i, ..]).to_vec();
        let (idx, _) = linear_nearest_neighbor(&points, &query)?;
        linear_results.push(idx);
    }

    let linear_time = start.elapsed();

    // KD-Tree search
    let start = Instant::now();
    let mut kdtree_results = Vec::with_capacity(n_queries);

    for i in 0..n_queries {
        let query = query_points.slice(ndarray::s![i, ..]).to_vec();
        let (idx, _) = kdtree.nearest_neighbor(&query)?;
        kdtree_results.push(idx);
    }

    let kdtree_time = start.elapsed();

    // Ball Tree search
    let start = Instant::now();
    let mut balltree_results = Vec::with_capacity(n_queries);

    for i in 0..n_queries {
        let query = query_points.slice(ndarray::s![i, ..]).to_vec();
        let (idx, _) = balltree.nearest_neighbor(&query)?;
        balltree_results.push(idx);
    }

    let balltree_time = start.elapsed();

    println!("  Linear search:   {:?}", linear_time);
    println!("  KD-Tree search:  {:?}", kdtree_time);
    println!("  Ball Tree search: {:?}", balltree_time);

    // Calculate speedups
    let kdtree_speedup = linear_time.as_secs_f64() / kdtree_time.as_secs_f64();
    let balltree_speedup = linear_time.as_secs_f64() / balltree_time.as_secs_f64();

    println!("  KD-Tree speedup: {:.1}x", kdtree_speedup);
    println!("  Ball Tree speedup: {:.1}x", balltree_speedup);

    // Verify that all methods return the same results
    let mut all_match = true;
    for i in 0..n_queries {
        if kdtree_results[i] != linear_results[i] || balltree_results[i] != linear_results[i] {
            all_match = false;
            break;
        }
    }

    println!("  All results match: {}", all_match);

    // Compare k nearest neighbors search performance
    println!("\nk-nearest neighbors search performance (k=10, 100 queries):");

    // Linear search
    let start = Instant::now();

    for i in 0..n_queries {
        let query = query_points.slice(ndarray::s![i, ..]).to_vec();
        let _ = linear_k_nearest_neighbors(&points, &query, 10)?;
    }

    let linear_time = start.elapsed();

    // KD-Tree search
    let start = Instant::now();

    for i in 0..n_queries {
        let query = query_points.slice(ndarray::s![i, ..]).to_vec();
        let _ = kdtree.k_nearest_neighbors(&query, 10)?;
    }

    let kdtree_time = start.elapsed();

    // Ball Tree search
    let start = Instant::now();

    for i in 0..n_queries {
        let query = query_points.slice(ndarray::s![i, ..]).to_vec();
        let _ = balltree.k_nearest_neighbors(&query, 10)?;
    }

    let balltree_time = start.elapsed();

    println!("  Linear search:    {:?}", linear_time);
    println!("  KD-Tree search:   {:?}", kdtree_time);
    println!("  Ball Tree search: {:?}", balltree_time);

    // Calculate speedups
    let kdtree_speedup = linear_time.as_secs_f64() / kdtree_time.as_secs_f64();
    let balltree_speedup = linear_time.as_secs_f64() / balltree_time.as_secs_f64();

    println!("  KD-Tree speedup:  {:.1}x", kdtree_speedup);
    println!("  Ball Tree speedup: {:.1}x", balltree_speedup);

    // Compare radius search performance
    let radius = 0.2;
    println!("\nRadius search performance (r={}, 100 queries):", radius);

    // Linear search
    let start = Instant::now();

    for i in 0..n_queries {
        let query = query_points.slice(ndarray::s![i, ..]).to_vec();
        let _ = linear_points_within_radius(&points, &query, radius)?;
    }

    let linear_time = start.elapsed();

    // KD-Tree search
    let start = Instant::now();

    for i in 0..n_queries {
        let query = query_points.slice(ndarray::s![i, ..]).to_vec();
        let _ = kdtree.points_within_radius(&query, radius)?;
    }

    let kdtree_time = start.elapsed();

    // Ball Tree search
    let start = Instant::now();

    for i in 0..n_queries {
        let query = query_points.slice(ndarray::s![i, ..]).to_vec();
        let _ = balltree.points_within_radius(&query, radius)?;
    }

    let balltree_time = start.elapsed();

    println!("  Linear search:    {:?}", linear_time);
    println!("  KD-Tree search:   {:?}", kdtree_time);
    println!("  Ball Tree search: {:?}", balltree_time);

    // Calculate speedups
    let kdtree_speedup = linear_time.as_secs_f64() / kdtree_time.as_secs_f64();
    let balltree_speedup = linear_time.as_secs_f64() / balltree_time.as_secs_f64();

    println!("  KD-Tree speedup:  {:.1}x", kdtree_speedup);
    println!("  Ball Tree speedup: {:.1}x", balltree_speedup);

    // Demonstrate performance improvement in MLS interpolation
    println!("\nMoving Least Squares interpolation performance:");

    // Create MLS interpolator with linear search
    let mls_standard = MovingLeastSquares::new(
        points.clone(),
        values.clone(),
        WeightFunction::Gaussian,
        PolynomialBasis::Linear,
        0.1,
    )?;

    // Create MLS that manually uses KD-Tree
    let kdtree = KdTree::new(points.clone())?;

    // Run interpolation with standard MLS (linear search)
    let start = Instant::now();

    for i in 0..n_queries.min(10) {
        // Using fewer queries for MLS to keep runtime reasonable
        let query = query_points.slice(ndarray::s![i, ..]);
        let _ = mls_standard.evaluate(&query)?;
    }

    let standard_time = start.elapsed();

    // Run interpolation with MLS manually using KD-Tree
    let start = Instant::now();

    for i in 0..n_queries.min(10) {
        let query = query_points.slice(ndarray::s![i, ..]);
        let query_vec = query.to_vec();

        // Find neighbors using KD-Tree (much faster than linear search)
        let neighbors = kdtree.k_nearest_neighbors(&query_vec, 50)?;

        // Extract indices and distances
        let indices: Vec<usize> = neighbors.iter().map(|&(idx, _)| idx).collect();
        let distances: Vec<f64> = neighbors.iter().map(|&(_, dist)| dist).collect();

        // Manually implement weighted interpolation (simplified)
        let mut weight_sum = 0.0;
        let mut weighted_value_sum = 0.0;

        for (i, &idx) in indices.iter().enumerate() {
            // Compute Gaussian weight
            let dist = distances[i];
            let bandwidth = 0.1;
            let weight = (-(dist / bandwidth).powi(2)).exp();

            weight_sum += weight;
            weighted_value_sum += weight * values[idx];
        }

        let _ = if weight_sum > 0.0 {
            weighted_value_sum / weight_sum
        } else {
            0.0
        };
    }

    let kdtree_time = start.elapsed();

    println!("  Standard MLS (linear search): {:?}", standard_time);
    println!("  MLS with KD-Tree:           {:?}", kdtree_time);

    // Calculate speedup
    let speedup = standard_time.as_secs_f64() / kdtree_time.as_secs_f64();
    println!("  Speedup: {:.1}x", speedup);

    println!("\nConclusion:");
    println!("  Spatial data structures like KD-Trees and Ball Trees can");
    println!("  significantly accelerate nearest neighbor searches, which");
    println!("  are critical for many interpolation methods.");

    Ok(())
}

/// Linear (brute force) nearest neighbor search
fn linear_nearest_neighbor(
    points: &Array2<f64>,
    query: &[f64],
) -> Result<(usize, f64), Box<dyn Error>> {
    let n_points = points.shape()[0];

    let mut min_dist = f64::INFINITY;
    let mut min_idx = 0;

    for i in 0..n_points {
        let point = points.row(i);
        let mut dist_sq = 0.0;

        for j in 0..query.len() {
            let diff = point[j] - query[j];
            dist_sq += diff * diff;
        }

        let dist = dist_sq.sqrt();

        if dist < min_dist {
            min_dist = dist;
            min_idx = i;
        }
    }

    Ok((min_idx, min_dist))
}

/// Linear (brute force) k-nearest neighbors search
fn linear_k_nearest_neighbors(
    points: &Array2<f64>,
    query: &[f64],
    k: usize,
) -> Result<Vec<(usize, f64)>, Box<dyn Error>> {
    let n_points = points.shape()[0];
    let k = k.min(n_points);

    // Calculate all distances
    let mut distances: Vec<(usize, f64)> = (0..n_points)
        .map(|i| {
            let point = points.row(i);
            let mut dist_sq = 0.0;

            for j in 0..query.len() {
                let diff = point[j] - query[j];
                dist_sq += diff * diff;
            }

            (i, dist_sq.sqrt())
        })
        .collect();

    // Sort by distance
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Return k nearest
    distances.truncate(k);
    Ok(distances)
}

/// Linear (brute force) search for points within radius
fn linear_points_within_radius(
    points: &Array2<f64>,
    query: &[f64],
    radius: f64,
) -> Result<Vec<(usize, f64)>, Box<dyn Error>> {
    let n_points = points.shape()[0];

    // Calculate all distances and filter by radius
    let mut results: Vec<(usize, f64)> = (0..n_points)
        .filter_map(|i| {
            let point = points.row(i);
            let mut dist_sq = 0.0;

            for j in 0..query.len() {
                let diff = point[j] - query[j];
                dist_sq += diff * diff;
            }

            let dist = dist_sq.sqrt();

            if dist <= radius {
                Some((i, dist))
            } else {
                None
            }
        })
        .collect();

    // Sort by distance
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(results)
}
