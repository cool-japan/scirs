use scirs2_datasets::{
    make_blobs, make_classification, make_regression, make_time_series, utils::normalize,
    utils::train_test_split,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating synthetic datasets...\n");

    // Generate classification dataset
    let n_samples = 100;
    let n_features = 5;

    let classification_data = make_classification(
        n_samples,
        n_features,
        3,        // 3 classes
        2,        // 2 clusters per class
        3,        // 3 informative features
        Some(42), // random seed
    )?;

    // Train-test split
    let (train, test) = train_test_split(&classification_data, 0.2, Some(42))?;

    println!("Classification dataset:");
    println!("  Total samples: {}", classification_data.n_samples());
    println!("  Features: {}", classification_data.n_features());
    println!("  Training samples: {}", train.n_samples());
    println!("  Test samples: {}", test.n_samples());

    // Generate regression dataset
    let regression_data = make_regression(
        n_samples,
        n_features,
        3,   // 3 informative features
        0.5, // noise level
        Some(42),
    )?;

    println!("\nRegression dataset:");
    println!("  Samples: {}", regression_data.n_samples());
    println!("  Features: {}", regression_data.n_features());

    // Normalize the data (in-place)
    let mut data_copy = regression_data.data.clone();
    normalize(&mut data_copy);
    println!("  Data normalized successfully");

    // Generate clustering data (blobs)
    let clustering_data = make_blobs(
        n_samples,
        2,   // 2 features for easy visualization
        4,   // 4 clusters
        0.8, // cluster standard deviation
        Some(42),
    )?;

    println!("\nClustering dataset (blobs):");
    println!("  Samples: {}", clustering_data.n_samples());
    println!("  Features: {}", clustering_data.n_features());

    // Find the number of clusters by finding the max value of target
    let num_clusters = clustering_data.target.as_ref().map_or(0, |t| {
        let mut max_val = -1.0;
        for &val in t.iter() {
            if val > max_val {
                max_val = val;
            }
        }
        (max_val as usize) + 1
    });

    println!("  Clusters: {}", num_clusters);

    // Generate time series data
    let time_series = make_time_series(
        100,  // 100 time steps
        3,    // 3 features/variables
        true, // with trend
        true, // with seasonality
        0.2,  // noise level
        Some(42),
    )?;

    println!("\nTime series dataset:");
    println!("  Time steps: {}", time_series.n_samples());
    println!("  Features: {}", time_series.n_features());

    Ok(())
}
