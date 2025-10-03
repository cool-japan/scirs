use scirs2_core::array::{mask_array, MaskedArray};
use scirs2_core::memory_efficient::{create_disk_array, ChunkingStrategy};
use scirs2_core::ndarray_ext::Array2;
use scirs2_core::random::Rng;
// Removed unused statrs import
use std::time::Instant;
use tempfile::tempdir;

/// Simulates loading a chunk of a large dataset
#[allow(dead_code)]
fn load_dataset_chunk(chunk_size: usize, nfeatures: usize) -> Array2<f64> {
    let mut rng = scirs2_core::random::thread_rng();
    Array2::from_shape_fn((chunk_size, nfeatures), |_| rng.random_range(0.0..100.0))
}

/// Normalizes data (center and scale)
#[allow(dead_code)]
fn normalize_chunk(chunk: &Array2<f64>) -> Array2<f64> {
    let mut normalized = chunk.clone();
    for col in 0..chunk.shape()[1] {
        let column = chunk.slice(scirs2_core::ndarray::s![.., col]);
        let mean = column.mean().unwrap_or(0.0);
        let std_dev = column.std(0.0);

        if std_dev > 1e-10 {
            for (i, val) in column.iter().enumerate() {
                normalized[[i, col]] = (val - mean) / std_dev;
            }
        }
    }
    normalized
}

/// Find outliers in data (values > threshold standard deviations from mean)
#[allow(dead_code)]
fn mask_outliers(
    chunk: &Array2<f64>,
    threshold: f64,
) -> MaskedArray<f64, scirs2_core::ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> {
    // For each feature, mask values that are more than threshold standard deviations from the mean
    let mut mask = Array2::from_elem(chunk.raw_dim(), false);

    for col in 0..chunk.shape()[1] {
        let column = chunk.slice(scirs2_core::ndarray::s![.., col]);
        let mean = column.mean().unwrap_or(0.0);
        let std_dev = column.std(0.0);

        if std_dev > 1e-10 {
            for (i, val) in column.iter().enumerate() {
                if (*val - mean).abs() > threshold * std_dev {
                    mask[[i, col]] = true;
                }
            }
        }
    }

    // Create a masked array with the outliers masked
    mask_array(chunk.clone(), Some(mask), Some(f64::NAN)).expect("Failed to create masked array")
}

/// Compute correlation matrix for a dataset
#[allow(dead_code)]
fn compute_correlation(data: &Array2<f64>) -> Array2<f64> {
    let n_features = data.shape()[1];
    let mut corr = Array2::zeros((n_features, n_features));

    for i in 0..n_features {
        for j in 0..n_features {
            let col_i = data.slice(scirs2_core::ndarray::s![.., i]);
            let col_j = data.slice(scirs2_core::ndarray::s![.., j]);

            let mean_i = col_i.mean().unwrap_or(0.0);
            let mean_j = col_j.mean().unwrap_or(0.0);

            let std_i = col_i.std(0.0);
            let std_j = col_j.std(0.0);

            if std_i > 1e-10 && std_j > 1e-10 {
                let mut cov = 0.0;
                for k in 0..data.shape()[0] {
                    cov += (data[[k, i]] - mean_i) * (data[[k, j]] - mean_j);
                }
                cov /= data.shape()[0] as f64;

                corr[[i, j]] = cov / (std_i * std_j);
            } else {
                corr[[i, j]] = 0.0;
            }
        }
    }

    corr
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Large Dataset Processing Example");
    println!("================================\n");

    // Simulate a large dataset that wouldn't fit in memory
    let total_samples = 1_000_000; // 1 million samples
    let n_features = 100; // 100 features
    let chunk_size = 10_000; // Process in chunks of 10,000 samples
    let n_chunks = total_samples / chunk_size;

    println!(
        "Dataset: {} samples × {} features",
        total_samples, n_features
    );
    println!(
        "Processing in {} chunks of {} samples each\n",
        n_chunks, chunk_size
    );

    // Create a temporary directory for out-of-core storage
    let temp_dir = tempdir()?;
    let normalized_path = temp_dir.path().join("normalized.bin");
    let outliers_path = temp_dir.path().join("outliers.bin");

    // Step 1: Load and normalize data in chunks
    println!("Step 1: Normalize data in chunks and store on disk");
    let start = Instant::now();

    // Process the first chunk to initialize the disk-backed array
    println!("  Processing chunk 1/{}", n_chunks);
    let chunk1 = load_dataset_chunk(chunk_size, n_features);
    let normalized1 = normalize_chunk(&chunk1);

    // Create disk-backed array
    let disk_array = create_disk_array(
        &normalized1,
        &normalized_path,
        ChunkingStrategy::Fixed(chunk_size),
        false,
    )?;

    // Now process remaining chunks
    for i in 1..n_chunks {
        println!("  Processing chunk {}/{}", i + 1, n_chunks);
        let chunk = load_dataset_chunk(chunk_size, n_features);
        let normalized = normalize_chunk(&chunk);

        // In a real implementation, we would append this to the disk array
        // For this example, we'll just simulate the process
    }

    println!("  Normalization completed in {:?}", start.elapsed());

    // Step 2: Detect outliers and create a masked array
    println!("\nStep 2: Detect outliers in normalized data");
    let start = Instant::now();

    // Load some chunks back from disk to detect outliers
    let normalized_data = disk_array.load()?;
    let outlier_threshold = 3.0; // 3 standard deviations

    println!(
        "  Detecting outliers (threshold: {} std dev)",
        outlier_threshold
    );
    let mask_start = Instant::now();
    let masked_data = mask_outliers(&normalized_data, outlier_threshold);
    println!(
        "  Outlier detection completed in {:?}",
        mask_start.elapsed()
    );

    // Count outliers
    let mask = &masked_data.mask;
    let outlier_count = mask.iter().filter(|&&m| m).count();
    println!(
        "  Found {} outliers out of {} values ({:.2}%)",
        outlier_count,
        mask.len(),
        100.0 * outlier_count as f64 / mask.len() as f64
    );

    // Step 3: Compute correlation matrix on cleaned data
    println!("\nStep 3: Compute correlation matrix");
    let start = Instant::now();

    let correlation = compute_correlation(&normalized_data);

    println!(
        "  Correlation computation completed in {:?}",
        start.elapsed()
    );
    println!("  Correlation matrix shape: {:?}", correlation.shape());

    // Display a small section of the correlation matrix
    let n = std::cmp::min(5, correlation.shape()[0]);
    println!("  Top-left corner of correlation matrix:");
    for i in 0..n {
        for j in 0..n {
            print!("{:8.3} ", correlation[[i, j]]);
        }
        println!();
    }

    // Step 4: Find highly correlated features
    println!("\nStep 4: Find highly correlated features");
    let corr_threshold = 0.8;

    let mut highly_correlated = Vec::new();
    for i in 0..n_features {
        for j in (i + 1)..n_features {
            if correlation[[i, j]].abs() >= corr_threshold {
                highly_correlated.push((i, j, correlation[[i, j]]));
            }
        }
    }

    println!(
        "  Found {} feature pairs with correlation ≥ {}",
        highly_correlated.len(),
        corr_threshold
    );

    // Print a few examples
    let n_examples = std::cmp::min(5, highly_correlated.len());
    for &(f1, f2, corr) in highly_correlated.iter().take(n_examples) {
        println!("  Features {} and {}: correlation = {:.3}", f1, f2, corr);
    }

    println!("\nAnalysis completed successfully!");
    Ok(())
}
