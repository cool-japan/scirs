/// Fast Approximate Kriging Example for Large Datasets
///
/// This example demonstrates fast approximation techniques for applying
/// kriging to large spatial datasets (with thousands to millions of points).
/// Standard kriging methods scale poorly (O(n³) for fitting, O(n²) for prediction),
/// making them impractical for large datasets.
///
/// This example demonstrates four different approximation techniques:
///
/// 1. **Local Kriging**: Uses only nearby points (typically 30-100) for each prediction,
///    reducing complexity to O(k³) per prediction where k << n is the neighborhood size.
///
/// 2. **Fixed Rank Kriging**: Approximates the covariance matrix with a low-rank
///    decomposition, reducing complexity to O(nr²) for fitting and O(r²) for prediction,
///    where r is the rank parameter.
///
/// 3. **Covariance Tapering**: Introduces sparsity by setting small covariances to zero,
///    enabling sparse matrix operations for improved efficiency.
///
/// 4. **Performance Benchmarking**: Compares speed and accuracy of different methods
///    on synthetic datasets of various sizes.
///
/// Each method offers different trade-offs between computational efficiency and
/// prediction accuracy, enabling kriging to scale to much larger datasets.
use ndarray::{Array1, Array2};
use scirs2_interpolate::advanced::fast_kriging::{
    make_fixed_rank_kriging, make_hodlr_kriging, make_local_kriging, FastKrigingBuilder,
    FastKrigingMethod,
};
use scirs2_interpolate::advanced::kriging::CovarianceFunction;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fast Approximate Kriging Example");
    println!("================================\n");

    // Generate sample datasets of different sizes
    let small_dataset = generate_synthetic_data(100, 2, 0.5)?;
    let medium_dataset = generate_synthetic_data(1000, 2, 0.5)?;
    let large_dataset = generate_synthetic_data(5000, 2, 0.5)?;

    println!("Generated synthetic datasets:");
    println!("  Small: {} points", small_dataset.0.shape()[0]);
    println!("  Medium: {} points", medium_dataset.0.shape()[0]);
    println!("  Large: {} points", large_dataset.0.shape()[0]);

    // Example 1: Local Kriging for different neighborhood sizes
    println!("\n1. Local Kriging with Different Neighborhood Sizes");
    println!("------------------------------------------------");
    local_kriging_example(&medium_dataset.0, &medium_dataset.1)?;

    // Example 2: Fixed Rank Kriging with different ranks
    println!("\n2. Fixed Rank Kriging with Different Ranks");
    println!("----------------------------------------");
    fixed_rank_kriging_example(&medium_dataset.0, &medium_dataset.1)?;

    // Example 3: HODLR Kriging with different leaf sizes
    println!("\n3. HODLR Kriging with Different Leaf Sizes");
    println!("----------------------------------------");
    hodlr_kriging_example(&medium_dataset.0, &medium_dataset.1)?;

    // Example 4: Performance comparison on increasingly large datasets
    println!("\n4. Performance Scaling with Dataset Size");
    println!("-------------------------------------");
    performance_scaling_example(&small_dataset, &medium_dataset, &large_dataset)?;

    // Example 5: Accuracy comparison between methods
    println!("\n5. Accuracy Comparison Between Methods");
    println!("------------------------------------");
    accuracy_comparison_example(&medium_dataset.0, &medium_dataset.1)?;

    Ok(())
}

/// Example demonstrating HODLR kriging with different leaf sizes
fn hodlr_kriging_example(
    points: &Array2<f64>,
    values: &Array1<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing different leaf sizes for HODLR kriging...");

    // Create test points for prediction
    let test_points = generate_prediction_grid(5, points.shape()[1]);
    println!("Generated {} test points", test_points.shape()[0]);

    // Create HODLR kriging models with different leaf sizes
    let leaf_sizes = [16, 32, 64, 128];

    println!("\nLeaf Size | Build Time (ms) | Prediction Time (ms) | Memory Usage");
    println!("----------|-----------------|---------------------|-------------");

    for &leaf_size in &leaf_sizes {
        // Create model
        let start_time = Instant::now();

        let kriging = make_hodlr_kriging(
            &points.view(),
            &values.view(),
            CovarianceFunction::Matern52,
            1.0,
            leaf_size,
        )?;

        let build_time = start_time.elapsed().as_micros() as f64 / 1000.0;

        // Make predictions
        let pred_start = Instant::now();
        let predictions = kriging.predict(&test_points.view())?;
        let pred_time = pred_start.elapsed().as_micros() as f64 / 1000.0;

        // Print results
        println!(
            "{:10} | {:16.2} | {:19.2} | O(n log n)",
            leaf_size, build_time, pred_time
        );

        // Show sample prediction for first test point
        println!(
            "  Sample prediction at ({:.1}, {:.1}): {:.4}",
            test_points[[0, 0]],
            test_points[[0, 1]],
            predictions.value[0]
        );
    }

    // Explanation of the HODLR kriging approach
    println!("\nThe HODLR (Hierarchical Off-Diagonal Low-Rank) approach:");
    println!("- Divides points into a hierarchical tree structure");
    println!("- Uses full-rank computation for nearby points (diagonal blocks)");
    println!("- Approximates distant interactions with low-rank representations");
    println!("- Reduces complexity from O(n³) to O(n log n)");
    println!("- Smaller leaf size = more approximation but faster computation");
    println!("- Balances accuracy and computational efficiency");
    println!("- Good choice for very large datasets with millions of points");

    Ok(())
}

/// Generate a synthetic spatial dataset with underlying pattern plus noise
fn generate_synthetic_data(
    n_points: usize,
    n_dims: usize,
    noise_level: f64,
) -> Result<(Array2<f64>, Array1<f64>), Box<dyn std::error::Error>> {
    use scirs2_core::random::Random;
    let mut rng = Random::default();

    // Create points using Latin Hypercube Sampling for better space filling
    let mut points = Array2::zeros((n_points, n_dims));

    // Simple LHS implementation
    for d in 0..n_dims {
        let mut values: Vec<f64> = (0..n_points)
            .map(|i| {
                let bin_width = 1.0 / (n_points as f64);
                let bin_min = i as f64 * bin_width;
                bin_min + rng.random_range(0.0, bin_width)
            })
            .collect();

        // Shuffle the dimension values
        for i in (1..values.len()).rev() {
            let j = rng.random_range(0usize, i + 1);
            values.swap(i, j);
        }

        // Assign to points array
        for i in 0..n_points {
            points[[i, d]] = values[i] * 10.0; // Scale to [0, 10]
        }
    }

    // Create values based on underlying pattern plus noise
    let mut values = Array1::zeros(n_points);

    // Generate values with spatial pattern (2D example)
    for i in 0..n_points {
        if n_dims >= 2 {
            let x = points[[i, 0]];
            let y = points[[i, 1]];

            // Pattern: f(x,y) = sin(x/2) + cos(y/3) + x*y/50
            values[i] = f64::sin(x / 2.0) + f64::cos(y / 3.0) + (x * y / 50.0);
        } else {
            // 1D fallback
            let x = points[[i, 0]];
            values[i] = f64::sin(x / 2.0);
        }

        // Add noise
        values[i] += noise_level * scirs2_core::random::sampling::random_standard_normal(&mut rng);
    }

    Ok((points, values))
}

/// Generate a grid of points for prediction
fn generate_prediction_grid(n_grid: usize, n_dims: usize) -> Array2<f64> {
    // For dimensions > 2, we'll just create a line through the space
    if n_dims > 2 {
        let mut grid_points = Array2::zeros((n_grid, n_dims));
        for i in 0..n_grid {
            let t = 10.0 * (i as f64) / ((n_grid - 1) as f64);
            for d in 0..n_dims {
                grid_points[[i, d]] = t;
            }
        }
        return grid_points;
    }

    // For 1D, create a line
    if n_dims == 1 {
        let mut grid_points = Array2::zeros((n_grid, 1));
        for i in 0..n_grid {
            grid_points[[i, 0]] = 10.0 * (i as f64) / ((n_grid - 1) as f64);
        }
        return grid_points;
    }

    // For 2D, create a regular grid
    let grid_size = n_grid * n_grid;
    let mut grid_points = Array2::zeros((grid_size, 2));

    let mut idx = 0;
    for i in 0..n_grid {
        let x = 10.0 * (i as f64) / ((n_grid - 1) as f64);
        for j in 0..n_grid {
            let y = 10.0 * (j as f64) / ((n_grid - 1) as f64);
            grid_points[[idx, 0]] = x;
            grid_points[[idx, 1]] = y;
            idx += 1;
        }
    }

    grid_points
}

/// Example demonstrating local kriging with different neighborhood sizes
fn local_kriging_example(
    points: &Array2<f64>,
    values: &Array1<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing different neighborhood sizes for local kriging...");

    // Create test points for prediction
    let test_points = generate_prediction_grid(5, points.shape()[1]);
    println!("Generated {} test points", test_points.shape()[0]);

    // Create local kriging models with different neighborhood sizes
    let neighborhood_sizes = [10, 30, 50, 100];

    println!("\nNeighborhood Size | Prediction Time (ms) | Memory Usage");
    println!("----------------|---------------------|-------------");

    for &size in &neighborhood_sizes {
        // Create model
        let _start_time = Instant::now();

        let kriging = make_local_kriging(
            &points.view(),
            &values.view(),
            CovarianceFunction::Matern52,
            1.0,
            size,
        )?;

        // Make predictions
        let pred_start = Instant::now();
        let predictions = kriging.predict(&test_points.view())?;
        let pred_time = pred_start.elapsed().as_micros() as f64 / 1000.0;

        // Print results
        println!(
            "{:16} | {:19.2} | O({})",
            size,
            pred_time,
            size * size * size
        );

        // Show sample prediction for first test point
        println!(
            "  Sample prediction at ({:.1}, {:.1}): {:.4}",
            test_points[[0, 0]],
            test_points[[0, 1]],
            predictions.value[0]
        );
    }

    // Explanation of the local kriging approach
    println!("\nThe local kriging approach:");
    println!("- Uses only the K nearest neighbors for each prediction");
    println!("- Reduces complexity from O(n³) to O(K³) where K << n");
    println!("- Makes each prediction independent (parallelization friendly)");
    println!("- Trades global optimality for computational efficiency");
    println!("- Larger neighborhoods = more accurate but slower predictions");

    Ok(())
}

/// Example demonstrating fixed rank kriging with different ranks
fn fixed_rank_kriging_example(
    points: &Array2<f64>,
    values: &Array1<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing different ranks for fixed rank kriging...");

    // Create test points for prediction
    let test_points = generate_prediction_grid(5, points.shape()[1]);
    println!("Generated {} test points", test_points.shape()[0]);

    // Create fixed rank kriging models with different ranks
    let ranks = [10, 20, 50, 100];

    println!("\nRank | Build Time (ms) | Prediction Time (ms) | Memory Usage");
    println!("-----|-----------------|---------------------|-------------");

    for &rank in &ranks {
        // Create model
        let start_time = Instant::now();

        let kriging = make_fixed_rank_kriging(
            &points.view(),
            &values.view(),
            CovarianceFunction::Matern52,
            1.0,
            rank,
        )?;

        let build_time = start_time.elapsed().as_micros() as f64 / 1000.0;

        // Make predictions
        let pred_start = Instant::now();
        let predictions = kriging.predict(&test_points.view())?;
        let pred_time = pred_start.elapsed().as_micros() as f64 / 1000.0;

        // Print results
        println!(
            "{:5} | {:16.2} | {:19.2} | O({})",
            rank,
            build_time,
            pred_time,
            rank * rank
        );

        // Show sample prediction for first test point
        println!(
            "  Sample prediction at ({:.1}, {:.1}): {:.4}",
            test_points[[0, 0]],
            test_points[[0, 1]],
            predictions.value[0]
        );
    }

    // Explanation of the fixed rank kriging approach
    println!("\nThe fixed rank kriging approach:");
    println!("- Approximates the covariance matrix K ≈ QΛQ^T with rank r << n");
    println!("- Reduces complexity from O(n³) to O(nr²) fitting, O(r²) prediction");
    println!("- One-time upfront cost, fast predictions");
    println!("- Higher rank = better approximation but more memory and computation");
    println!("- Good choice when predicting at many locations");

    Ok(())
}

/// Example demonstrating performance scaling with dataset size
fn performance_scaling_example(
    small_dataset: &(Array2<f64>, Array1<f64>),
    medium_dataset: &(Array2<f64>, Array1<f64>),
    large_dataset: &(Array2<f64>, Array1<f64>),
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Measuring performance scaling across dataset sizes...");

    // Create fixed test points for prediction
    let test_points = generate_prediction_grid(3, small_dataset.0.shape()[1]);
    println!("Using {} test points", test_points.shape()[0]);

    // Define methods to test
    let methods = [
        (FastKrigingMethod::Local, "Local (50 neighbors)"),
        (FastKrigingMethod::FixedRank(50), "Fixed Rank (r=50)"),
        (FastKrigingMethod::Tapering(2.0), "Tapering (range=2.0)"),
        (FastKrigingMethod::HODLR(64), "HODLR (leaf=64)"),
    ];

    // Test datasets
    let datasets = [
        ("Small (n=100)", small_dataset),
        ("Medium (n=1000)", medium_dataset),
        ("Large (n=5000)", large_dataset),
    ];

    println!("\nMethod | Dataset | Build Time (ms) | Predict Time (ms)");
    println!("-------|---------|-----------------|------------------");

    for (method, method_name) in &methods {
        for (dataset_name, (points, values)) in &datasets {
            // Create model
            let start_time = Instant::now();

            let kriging = FastKrigingBuilder::new()
                .points(points.clone())
                .values(values.clone())
                .covariance_function(CovarianceFunction::Matern52)
                .approximation_method(*method)
                .max_neighbors(50) // Only relevant for Local method
                .build()?;

            let build_time = start_time.elapsed().as_micros() as f64 / 1000.0;

            // Make predictions
            let pred_start = Instant::now();
            let _predictions = kriging.predict(&test_points.view())?;
            let pred_time = pred_start.elapsed().as_micros() as f64 / 1000.0;

            // Print results
            println!(
                "{:7} | {:9} | {:16.2} | {:16.2}",
                method_name, dataset_name, build_time, pred_time
            );
        }
        println!("-------|---------|-----------------|------------------");
    }

    // Explanation of scaling behavior
    println!("\nScaling behavior observations:");
    println!("- Local kriging: Build time is O(1), prediction scales with neighborhood size");
    println!("- Fixed rank: Build time scales with dataset size, prediction time is constant");
    println!("- Tapering: Both build and prediction scale better than full kriging");
    println!("- The optimal method depends on dataset size and prediction pattern");

    Ok(())
}

/// Example demonstrating accuracy comparison between methods
fn accuracy_comparison_example(
    points: &Array2<f64>,
    values: &Array1<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing accuracy of different approximation methods...");

    // Use a subset of the data for training, hold out some for testing
    let n_points = points.shape()[0];
    let n_train = (n_points * 80) / 100; // Use 80% for training

    let train_points = points.slice(ndarray::s![0..n_train, ..]).to_owned();
    let train_values = values.slice(ndarray::s![0..n_train]).to_owned();

    let test_points = points.slice(ndarray::s![n_train.., ..]).to_owned();
    let test_values = values.slice(ndarray::s![n_train..]).to_owned();

    println!(
        "Using {} points for training, {} points for testing",
        train_points.shape()[0],
        test_points.shape()[0]
    );

    // Define methods to test
    let methods = [
        (FastKrigingMethod::Local, "Local (30 neighbors)"),
        (FastKrigingMethod::Local, "Local (100 neighbors)"),
        (FastKrigingMethod::FixedRank(20), "Fixed Rank (r=20)"),
        (FastKrigingMethod::FixedRank(50), "Fixed Rank (r=50)"),
        (FastKrigingMethod::Tapering(2.0), "Tapering (range=2.0)"),
        (FastKrigingMethod::HODLR(32), "HODLR (leaf=32)"),
        (FastKrigingMethod::HODLR(128), "HODLR (leaf=128)"),
    ];

    println!("\nMethod | Mean Sq. Error | Compute Time (ms)");
    println!("-------|----------------|------------------");

    for (i, &(method, method_name)) in methods.iter().enumerate() {
        // Create model with appropriate parameters
        let start_time = Instant::now();

        let kriging = if i == 0 {
            // Local with 30 neighbors
            FastKrigingBuilder::new()
                .points(train_points.clone())
                .values(train_values.clone())
                .covariance_function(CovarianceFunction::Matern52)
                .approximation_method(method)
                .max_neighbors(30)
                .build()?
        } else if i == 1 {
            // Local with 100 neighbors
            FastKrigingBuilder::new()
                .points(train_points.clone())
                .values(train_values.clone())
                .covariance_function(CovarianceFunction::Matern52)
                .approximation_method(method)
                .max_neighbors(100)
                .build()?
        } else {
            // Other methods
            FastKrigingBuilder::new()
                .points(train_points.clone())
                .values(train_values.clone())
                .covariance_function(CovarianceFunction::Matern52)
                .approximation_method(method)
                .build()?
        };

        // Make predictions on test set
        let predictions = kriging.predict(&test_points.view())?;
        let compute_time = start_time.elapsed().as_micros() as f64 / 1000.0;

        // Calculate mean squared error
        let mut sum_sq_error = 0.0;
        for j in 0..test_values.len() {
            let error = predictions.value[j] - test_values[j];
            sum_sq_error += error * error;
        }
        let mse = sum_sq_error / test_values.len() as f64;

        // Print results
        println!("{:20} | {:16.6} | {:16.2}", method_name, mse, compute_time);
    }

    // Explanation of the accuracy comparison
    println!("\nAccuracy comparison observations:");
    println!("- Local kriging with more neighbors generally improves accuracy");
    println!("- Fixed rank kriging accuracy improves with higher rank");
    println!("- HODLR kriging offers good balance between accuracy and computation");
    println!("- Larger leaf sizes in HODLR generally improve accuracy but increase computation");
    println!("- Each method offers a different trade-off between speed and accuracy");
    println!("- The best method depends on your specific requirements");
    println!("- Consider combining methods for an optimal approach");

    Ok(())
}
