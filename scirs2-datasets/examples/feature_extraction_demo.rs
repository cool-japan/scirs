//! Feature extraction utilities demonstration
//!
//! This example demonstrates the use of feature extraction and transformation utilities
//! for preprocessing datasets before machine learning model training.

use ndarray::{Array1, Array2};
use scirs2_datasets::{
    create_binned_features, load_iris, min_max_scale, polynomial_features, robust_scale,
    statistical_features, BinningStrategy,
};

fn main() {
    println!("=== Feature Extraction Utilities Demonstration ===\n");

    // Create a sample dataset for demonstration
    let data = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 10.0, // Normal data
            2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 100.0, 500.0, // Outlier
        ],
    )
    .unwrap();

    println!("Original dataset:");
    print_data_summary(&data, "Original");
    println!();

    // Demonstrate Min-Max Scaling
    println!("=== Min-Max Scaling ============================");
    let mut data_minmax = data.clone();
    min_max_scale(&mut data_minmax, (0.0, 1.0));
    print_data_summary(&data_minmax, "Min-Max Scaled [0, 1]");

    let mut data_custom_range = data.clone();
    min_max_scale(&mut data_custom_range, (-1.0, 1.0));
    print_data_summary(&data_custom_range, "Min-Max Scaled [-1, 1]");
    println!();

    // Demonstrate Robust Scaling
    println!("=== Robust Scaling ==============================");
    let mut data_robust = data.clone();
    robust_scale(&mut data_robust);
    print_data_summary(&data_robust, "Robust Scaled (Median/IQR)");
    println!();

    // Demonstrate Polynomial Features
    println!("=== Polynomial Feature Generation ==============");
    let small_data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0]).unwrap();

    println!("Small dataset for polynomial demonstration:");
    print_data_matrix(&small_data, &["x1", "x2"]);

    let poly_with_bias = polynomial_features(&small_data, 2, true).unwrap();
    println!("Polynomial features (degree=2, with bias):");
    print_data_matrix(&poly_with_bias, &["1", "x1", "x2", "x1²", "x1*x2", "x2²"]);

    let poly_no_bias = polynomial_features(&small_data, 2, false).unwrap();
    println!("Polynomial features (degree=2, no bias):");
    print_data_matrix(&poly_no_bias, &["x1", "x2", "x1²", "x1*x2", "x2²"]);
    println!();

    // Demonstrate Statistical Feature Extraction
    println!("=== Statistical Feature Extraction =============");
    let stats_data = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

    let stats_features = statistical_features(&stats_data).unwrap();
    println!("Statistical features for data [1, 2, 3, 4, 5]:");
    println!("(Each sample gets the same global statistics)");
    print_statistical_features(stats_features.row(0).to_owned());
    println!();

    // Demonstrate Binning/Discretization
    println!("=== Feature Binning/Discretization =============");
    let binning_data =
        Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

    println!("Original data for binning: [1, 2, 3, 4, 5, 6, 7, 8]");

    let uniform_binned =
        create_binned_features(&binning_data, 3, BinningStrategy::Uniform).unwrap();
    println!(
        "Uniform binning (3 bins): {:?}",
        uniform_binned
            .column(0)
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>()
    );

    let quantile_binned =
        create_binned_features(&binning_data, 4, BinningStrategy::Quantile).unwrap();
    println!(
        "Quantile binning (4 bins): {:?}",
        quantile_binned
            .column(0)
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>()
    );
    println!();

    // Demonstrate Feature Extraction Pipeline
    println!("=== Complete Feature Extraction Pipeline =======");
    let iris = load_iris().unwrap();
    println!(
        "Using Iris dataset ({} samples, {} features)",
        iris.n_samples(),
        iris.n_features()
    );

    // Step 1: Robust scaling (handles outliers better)
    let mut scaled_iris = iris.data.clone();
    robust_scale(&mut scaled_iris);
    println!("Step 1: Applied robust scaling");

    // Step 2: Generate polynomial features (degree 2)
    let poly_iris = polynomial_features(&scaled_iris, 2, false).unwrap();
    println!("Step 2: Generated polynomial features");
    println!("  Original features: {}", scaled_iris.ncols());
    println!("  Polynomial features: {}", poly_iris.ncols());

    // Step 3: Create binned features for non-linearity
    let binned_iris = create_binned_features(&scaled_iris, 5, BinningStrategy::Quantile).unwrap();
    println!("Step 3: Created binned features");
    println!("  Binned features: {}", binned_iris.ncols());

    // Step 4: Extract statistical features
    let stats_iris =
        statistical_features(&iris.data.slice(ndarray::s![0..20, ..]).to_owned()).unwrap();
    println!("Step 4: Extracted statistical features (from first 20 samples)");
    println!("  Statistical features: {}", stats_iris.ncols());
    println!();

    // Comparison of scaling methods with outliers
    println!("=== Scaling Methods Comparison (with outliers) =");
    let outlier_data = Array2::from_shape_vec(
        (5, 1),
        vec![1.0, 2.0, 3.0, 4.0, 100.0], // 100.0 is a severe outlier
    )
    .unwrap();

    println!("Original data with outlier: [1, 2, 3, 4, 100]");

    let mut minmax_outlier = outlier_data.clone();
    min_max_scale(&mut minmax_outlier, (0.0, 1.0));
    println!(
        "Min-Max scaled: {:?}",
        minmax_outlier
            .column(0)
            .iter()
            .map(|&x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );

    let mut robust_outlier = outlier_data.clone();
    robust_scale(&mut robust_outlier);
    println!(
        "Robust scaled: {:?}",
        robust_outlier
            .column(0)
            .iter()
            .map(|&x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );

    println!("\nNotice how robust scaling is less affected by the outlier!");
    println!();

    // Feature engineering recommendations
    println!("=== Feature Engineering Recommendations ========");
    println!("1. **Scaling**: Use robust scaling when outliers are present");
    println!("2. **Polynomial**: Use degree 2-3 for non-linear relationships");
    println!("3. **Binning**: Use quantile binning for better distribution");
    println!("4. **Statistical**: Extract global statistics for context");
    println!("5. **Pipeline**: Always scale → transform → engineer → validate");
    println!();

    println!("=== Feature Extraction Demo Complete ===========");
}

/// Print a summary of data statistics
fn print_data_summary(data: &Array2<f64>, title: &str) {
    println!("{}: shape=({}, {})", title, data.nrows(), data.ncols());
    for j in 0..data.ncols() {
        let col = data.column(j);
        let min_val = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean = col.iter().sum::<f64>() / col.len() as f64;
        println!(
            "  Feature {}: min={:.3}, max={:.3}, mean={:.3}",
            j, min_val, max_val, mean
        );
    }
}

/// Print a data matrix with feature names
fn print_data_matrix(data: &Array2<f64>, feature_names: &[&str]) {
    // Print header
    print!("     ");
    for name in feature_names {
        print!("{:>8}", name);
    }
    println!();

    // Print data
    for i in 0..data.nrows() {
        print!("  {}: ", i);
        for j in 0..data.ncols() {
            print!("{:8.3}", data[[i, j]]);
        }
        println!();
    }
}

/// Print statistical features with labels
fn print_statistical_features(stats: Array1<f64>) {
    let labels = [
        "mean", "std", "min", "max", "median", "q25", "q75", "skewness", "kurtosis",
    ];
    println!("  Statistical measures:");
    for (i, &value) in stats.iter().enumerate() {
        if i < labels.len() {
            println!("    {}: {:.3}", labels[i], value);
        }
    }
}
