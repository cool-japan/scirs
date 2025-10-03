use scirs2_core::error::CoreError;
use scirs2_core::memory_efficient::{
    chunk_wise_op, create_disk_array, diagonal_view, evaluate, transpose_view, ChunkingStrategy,
    LazyArray,
};
use scirs2_core::ndarray_ext::Array2;
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<(), CoreError> {
    println!("Memory-Efficient Operations Example");
    println!("===================================\n");

    // Create a large array (500MB if using f64)
    let n = 8000;
    println!("Creating a {}x{} array...", n, n);
    let start = Instant::now();
    let data = Array2::from_shape_fn((n, n), |(i, j)| (i as f64 + j as f64) / (n as f64));
    println!("Array created in {:?}", start.elapsed());
    println!("Array shape: {:?}", data.shape());
    println!("Array memory usage: ~{} MB\n", n * n * 8 / (1024 * 1024));

    // Example 1: Chunked processing
    println!("Example 1: Chunked Processing");
    println!("--------------------------");

    // Define a simple operation - compute the square of each element
    let start = Instant::now();
    let result = chunk_wise_op(&data, |chunk| chunk.map(|x| x * x), ChunkingStrategy::Auto)?;
    println!("Chunked operation completed in {:?}", start.elapsed());
    println!("Result shape: {:?}", result.shape());
    println!("First few values: {:?}\n", get_corner(&result, 3));

    // Example 2: Lazy evaluation
    println!("Example 2: Lazy Evaluation");
    println!("----------------------");

    let start = Instant::now();
    // Create a lazy array
    let lazy_array = LazyArray::new(data.clone());

    // Define operations without executing them
    let lazy_result = lazy_array.map(|x| x * x);

    // Evaluate the lazy operations
    let result = evaluate(&lazy_result)?;
    println!("Lazy evaluation completed in {:?}", start.elapsed());
    println!("Result shape: {:?}", result.shape());
    println!("First few values: {:?}\n", get_corner(&result, 3));

    // Example 3: Array views
    println!("Example 3: Array Views");
    println!("------------------");

    let start = Instant::now();
    // Create a smaller array for demonstration
    let small_data = Array2::from_shape_fn((10, 10), |(i, j)| i as f64 + j as f64);

    // Create transposed and diagonal views
    let transposed = transpose_view(&small_data)?;
    let diagonal = diagonal_view(&small_data)?;

    println!("View operations completed in {:?}", start.elapsed());
    println!("Original array: \n{:?}", small_data);
    println!("Transposed view: \n{:?}", transposed);
    println!("Diagonal view: \n{:?}", diagonal);

    // Example 4: Out-of-core processing
    println!("\nExample 4: Out-of-Core Processing");
    println!("----------------------------");

    // Create a temporary file for the out-of-core array
    let temp_dir = tempfile::tempdir()?;
    let file_path = temp_dir.path().join("array.bin");

    let start = Instant::now();
    // Create a disk-backed array
    let disk_array = create_disk_array(&data, &file_path, ChunkingStrategy::Fixed(1000), false)?;

    // Load the array back from disk
    let loaded_data = disk_array.load()?;
    println!("Out-of-core operations completed in {:?}", start.elapsed());
    println!("Loaded data shape: {:?}", loaded_data.shape());
    println!("First few values: {:?}", get_corner(&loaded_data, 3));

    Ok(())
}

// Helper function to get a corner of an array for display
#[allow(dead_code)]
fn get_corner<T: Clone>(arr: &Array2<T>, size: usize) -> Array2<T> {
    let s = std::cmp::min(size, arr.shape()[0]);
    arr.slice(scirs2_core::ndarray::s![0..s, 0..s]).to_owned()
}
