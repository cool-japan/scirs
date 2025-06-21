use ndarray::Array;
use scirs2_core::memory_efficient::{
    ArithmeticOps, BroadcastOps, ChunkingStrategy, MemoryMappedArray, MemoryMappedChunks,
    ZeroCopyOps,
};
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Memory-Mapped Array Zero-Copy Operations Example");
    println!("===============================================\n");

    // Create a temporary directory for our example files
    let dir = tempdir()?;
    let file_path = dir.path().join("large_array.bin");
    println!("Creating a test file at: {}", file_path.display());

    // Create a large test array (10 million elements)
    let size = 10_000_000;
    println!("Creating a 1D array with {} elements", size);

    // Create and save the array in chunks to avoid excessive memory usage
    let mut file = File::create(&file_path)?;
    let chunk_size = 1_000_000;

    for chunk_idx in 0..(size / chunk_size) {
        let start = chunk_idx * chunk_size;
        let chunk: Vec<f64> = (0..chunk_size).map(|i| (start + i) as f64).collect();

        for val in &chunk {
            file.write_all(&val.to_ne_bytes())?;
        }
    }
    drop(file);

    println!(
        "Array saved to file (size: {} bytes)",
        size * std::mem::size_of::<f64>()
    );

    // Open as memory-mapped array
    let array = MemoryMappedArray::<f64>::open(&file_path, &[size])?;
    println!(
        "Opened as memory-mapped array with shape: {:?}",
        array.shape
    );

    // Demonstrate various zero-copy operations
    println!("\n1. Basic Statistics Using Zero-Copy Operations");
    println!("--------------------------------------------");

    // Calculate statistics
    let start = Instant::now();
    let sum = array.sum_zero_copy()?;
    let elapsed = start.elapsed();
    println!("Sum: {:.0} (calculated in {:.2?})", sum, elapsed);

    // Note: mean_zero_copy requires From<usize> trait bound
    // let start = Instant::now();
    // let mean = array.mean_zero_copy()?;
    // let elapsed = start.elapsed();
    // println!("Mean: {:.2} (calculated in {:.2?})", mean, elapsed);

    // Alternative: calculate mean manually
    let start = Instant::now();
    let data = array.as_array::<ndarray::Ix1>()?;
    let mean = data.mean().unwrap_or(0.0);
    let elapsed = start.elapsed();
    println!("Mean: {:.2} (calculated in {:.2?})", mean, elapsed);

    let start = Instant::now();
    let min = array.min_zero_copy()?;
    let max = array.max_zero_copy()?;
    let elapsed = start.elapsed();
    println!(
        "Min: {:.0}, Max: {:.0} (calculated in {:.2?})",
        min, max, elapsed
    );

    // Compare with loading the entire array
    println!("\nComparison with loading the entire array:");
    let start = Instant::now();
    let loaded_array = array.readonly_array::<ndarray::Ix1>()?;
    let loaded_sum: f64 = loaded_array.iter().sum();
    let elapsed = start.elapsed();
    println!(
        "Sum (loaded): {:.0} (calculated in {:.2?})",
        loaded_sum, elapsed
    );

    // Demonstrate mapping operation
    println!("\n2. Mapping Operations");
    println!("-------------------");

    // Time the mapping operation
    let start = Instant::now();
    let squared = array.map_zero_copy(|x| x * x)?;
    let elapsed = start.elapsed();
    println!("Squared all values in {:.2?}", elapsed);

    // Calculate statistics on the mapped array
    // Note: mean_zero_copy requires From<usize> trait bound
    // let squared_mean = squared.mean_zero_copy()?;
    // println!("Mean of squared values: {:.2}", squared_mean);

    // Alternative: calculate mean manually
    let squared_data = squared.as_array::<ndarray::Ix1>()?;
    let squared_mean = squared_data.mean().unwrap_or(0.0);
    println!("Mean of squared values: {:.2}", squared_mean);

    // Compare with conventional approach
    println!("\nComparison with conventional approach:");
    let start = Instant::now();
    let loaded_array = array.readonly_array::<ndarray::Ix1>()?;
    let squared_loaded: Array<f64, _> = loaded_array.map(|&x| x * x);
    let squared_loaded_mean = squared_loaded.mean().unwrap();
    let elapsed = start.elapsed();
    println!(
        "Mean of squared values (loaded): {:.2} (calculated in {:.2?})",
        squared_loaded_mean, elapsed
    );

    // Demonstrate combining arrays
    println!("\n3. Combining Arrays");
    println!("-----------------");

    // Create another array for combining
    let file_path2 = dir.path().join("array2.bin");
    let mut file = File::create(&file_path2)?;

    // Initialize with values 1000, 1001, 1002, ...
    for chunk_idx in 0..(size / chunk_size) {
        let start = chunk_idx * chunk_size;
        let chunk: Vec<f64> = (0..chunk_size).map(|i| (start + i + 1000) as f64).collect();

        for val in &chunk {
            file.write_all(&val.to_ne_bytes())?;
        }
    }
    drop(file);

    let array2 = MemoryMappedArray::<f64>::open(&file_path2, &[size])?;
    println!("Created second array with values offset by 1000");

    // Perform arithmetic operations
    let start = Instant::now();
    let sum_array = array.add(&array2)?;
    let elapsed = start.elapsed();
    println!("Added arrays in {:.2?}", elapsed);

    // Calculate statistics on the combined array
    // Note: mean_zero_copy requires From<usize> trait bound
    // let sum_mean = sum_array.mean_zero_copy()?;

    // Alternative: calculate mean manually
    let sum_data = sum_array.as_array::<ndarray::Ix1>()?;
    let sum_mean = sum_data.mean().unwrap_or(0.0);
    println!(
        "Mean of sum array: {:.2} (expected: {:.2})",
        sum_mean,
        (mean + 1000.0)
    );

    // Demonstrate filtering
    println!("\n4. Filtering Operations");
    println!("---------------------");

    // Filter for values divisible by 1000
    let start = Instant::now();
    let filtered = array.filter_zero_copy(|&x| (x as usize) % 1000 == 0)?;
    let elapsed = start.elapsed();

    println!("Filtered for values divisible by 1000 in {:.2?}", elapsed);
    println!("Found {} values", filtered.len());
    println!(
        "First few filtered values: {:?}",
        &filtered[..5.min(filtered.len())]
    );

    // Demonstrate broadcasting
    println!("\n5. Broadcasting Operations");
    println!("------------------------");

    // Create a small array for broadcasting
    let file_path3 = dir.path().join("small_array.bin");
    let factor_data: Vec<f64> = (0..5).map(|i| (i + 1) as f64).collect();

    let mut file = File::create(&file_path3)?;
    for val in &factor_data {
        file.write_all(&val.to_ne_bytes())?;
    }
    drop(file);

    // Create a 2D array for the example
    let file_path4 = dir.path().join("matrix.bin");
    let rows = 5;
    let cols = 5;
    let matrix_data: Vec<f64> = (0..(rows * cols)).map(|i| i as f64).collect();

    let mut file = File::create(&file_path4)?;
    for val in &matrix_data {
        file.write_all(&val.to_ne_bytes())?;
    }
    drop(file);

    let matrix = MemoryMappedArray::<f64>::open(&file_path4, &[rows, cols])?;
    let factors = MemoryMappedArray::<f64>::open(&file_path3, &[cols])?;

    println!(
        "Created matrix with shape {:?} and factors with shape {:?}",
        matrix.shape, factors.shape
    );

    // Perform broadcasting
    let start = Instant::now();
    let broadcast_result = matrix.broadcast_op(&factors, |a, b| a * b)?;
    let elapsed = start.elapsed();

    println!("Applied broadcasting operation in {:.2?}", elapsed);
    println!("Result shape: {:?}", broadcast_result.shape);

    // Display a small sample of the result
    let result_array = broadcast_result.readonly_array()?;
    println!("\nSample of the broadcast result:");
    for i in 0..3 {
        for j in 0..3 {
            print!("{:.0} ", result_array[(i, j)]);
        }
        println!();
    }

    // Performance comparison with different methods
    println!("\n6. Performance Comparison");
    println!("-----------------------");

    // Setup for benchmark
    let n_runs = 5;
    let benchmark_size = 5_000_000; // 5M elements
    let benchmark_path = dir.path().join("benchmark.bin");

    let mut file = File::create(&benchmark_path)?;
    for i in 0..benchmark_size {
        let val = i as f64;
        file.write_all(&val.to_ne_bytes())?;
    }
    drop(file);

    let bench_array = MemoryMappedArray::<f64>::open(&benchmark_path, &[benchmark_size])?;

    // Function to time: Calculate sum
    println!(
        "Calculating sum of {} elements with different methods:",
        benchmark_size
    );
    println!("{:-^60}", "");
    println!("{:<20} {:<20} {:<15}", "Method", "Time (ms)", "Result");
    println!("{:-^60}", "");

    // 1. Full array loading
    let mut total_time = 0;
    let mut result = 0.0;
    for _ in 0..n_runs {
        let start = Instant::now();
        let full_array = bench_array.readonly_array::<ndarray::Ix1>()?;
        result = full_array.iter().sum();
        total_time += start.elapsed().as_millis();
    }
    println!(
        "{:<20} {:<20.2} {:<15.0}",
        "Full loading",
        total_time as f64 / n_runs as f64,
        result
    );

    // 2. Zero-copy method
    total_time = 0;
    for _ in 0..n_runs {
        let start = Instant::now();
        result = bench_array.sum_zero_copy()?;
        total_time += start.elapsed().as_millis();
    }
    println!(
        "{:<20} {:<20.2} {:<15.0}",
        "Zero-copy",
        total_time as f64 / n_runs as f64,
        result
    );

    // 3. Chunk-wise manual
    total_time = 0;
    for _ in 0..n_runs {
        let start = Instant::now();
        let chunk_size = 1_000_000;
        let strategy = ChunkingStrategy::Fixed(chunk_size);

        // Process chunks
        let chunk_sums = bench_array.process_chunks(strategy, |chunk, _| chunk.iter().sum::<f64>());

        // Calculate final sum
        result = chunk_sums.iter().sum();
        total_time += start.elapsed().as_millis();
    }
    println!(
        "{:<20} {:<20.2} {:<15.0}",
        "Manual chunking",
        total_time as f64 / n_runs as f64,
        result
    );

    #[cfg(feature = "parallel")]
    {
        // 4. Parallel zero-copy
        use scirs2_core::memory_efficient::MemoryMappedChunksParallel;

        total_time = 0;
        for _ in 0..n_runs {
            let start = Instant::now();
            let chunk_size = 1_000_000;
            let strategy = ChunkingStrategy::Fixed(chunk_size);

            // Process chunks in parallel
            let chunk_sums =
                bench_array.process_chunks_parallel(strategy, |chunk, _| chunk.iter().sum::<f64>());

            // Calculate final sum
            result = chunk_sums.iter().sum();
            total_time += start.elapsed().as_millis();
        }
        println!(
            "{:<20} {:<20.2} {:<15.0}",
            "Parallel chunking",
            total_time as f64 / n_runs as f64,
            result
        );
    }

    println!("\nAll examples completed successfully!");

    Ok(())
}
