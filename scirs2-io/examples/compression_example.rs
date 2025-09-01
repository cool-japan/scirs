use ndarray::{Array, ArrayBase, IxDyn, OwnedRepr};
use rand::Rng;
use scirs2_io::compression::{
    algorithm_info, compress_data, compress_file, decompress_data, decompress_file,
    ndarray::{
        compare_compression_algorithms, compress_array, compress_array_chunked, decompress_array,
        decompress_array_chunked,
    },
    CompressionAlgorithm,
};
use std::fs::File;
use std::io::Write;

#[allow(dead_code)]
fn main() {
    // Basic data compression example
    if let Err(e) = basic_compression_example() {
        println!("Basic compression example failed: {}", e);
    }

    // File compression example
    if let Err(e) = file_compression_example() {
        println!("File compression example failed: {}", e);
    }

    // Array compression example
    if let Err(e) = array_compression_example() {
        println!("Array compression example failed: {}", e);
    }

    // Advanced chunked array compression example
    if let Err(e) = chunked_array_compression_example() {
        println!("Chunked array compression example failed: {}", e);
    }

    // Compression algorithm comparison
    if let Err(e) = compare_algorithms() {
        println!("Compression algorithm comparison failed: {}", e);
    }

    println!("Compression examples completed!");
}

#[allow(dead_code)]
fn basic_compression_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Basic Compression Example ===");

    // Create some sample data
    let original_data =
        b"This is a sample string that will be compressed using different algorithms.
    Scientific data often contains patterns and redundancy that can be effectively compressed.
    Let's see how well different compression algorithms perform on this text.";

    // Try different compression algorithms
    for algorithm in &[
        CompressionAlgorithm::Gzip,
        CompressionAlgorithm::Zstd,
        CompressionAlgorithm::Lz4,
        CompressionAlgorithm::Bzip2,
    ] {
        // Get algorithm info
        let info = algorithm_info(*algorithm);
        println!("\nCompressing with {}", info.name);

        // Compress with this algorithm
        let compressed = compress_data(original_data, *algorithm, None)?;

        // Calculate and print compression statistics
        let ratio = original_data.len() as f64 / compressed.len() as f64;
        println!("  Original size: {} bytes", original_data.len());
        println!("  Compressed size: {} bytes", compressed.len());
        println!("  Compression ratio: {:.2}x", ratio);

        // Verify we can decompress correctly
        let decompressed = decompress_data(&compressed, *algorithm)?;
        assert_eq!(decompressed, original_data);
        println!("  Decompression verified successfully");
    }

    Ok(())
}

#[allow(dead_code)]
fn file_compression_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== File Compression Example ===");

    // Create a test file with repeating content to demonstrate compression
    let test_file_path = "test_data.txt";
    let mut test_file = File::create(test_file_path)?;

    // Write some repetitive data to make compression effective
    for i in 0..1000 {
        writeln!(
            test_file,
            "Line {} with some repeated content for compression testing",
            i
        )?;
    }
    test_file.flush()?;

    // Compress the file with various algorithms
    for algorithm in &[
        CompressionAlgorithm::Gzip,
        CompressionAlgorithm::Zstd,
        CompressionAlgorithm::Lz4,
        CompressionAlgorithm::Bzip2,
    ] {
        let output_path = compress_file(test_file_path, None, *algorithm, Some(6))?;

        println!("Compressed with {}: {}", algorithm.extension(), output_path);

        // Decompress and verify
        let decompressed_path = decompress_file(&output_path, None, Some(*algorithm))?;
        println!("Decompressed to: {}", decompressed_path);

        // Clean up compressed and decompressed files
        std::fs::remove_file(output_path)?;
        std::fs::remove_file(decompressed_path)?;
    }

    // Clean up the original test file
    std::fs::remove_file(test_file_path)?;

    Ok(())
}

#[allow(dead_code)]
fn array_compression_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Array Compression Example ===");

    // Create a simple 3D array with repeating patterns
    let shape = vec![100, 100, 10];
    let mut data = Vec::with_capacity(shape.iter().product());

    // Fill with repeating patterns (good for compression)
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                // Create some spatial patterns
                data.push((i as f64).sin() + (j as f64).cos() + (k as f64 / 5.0).sin());
            }
        }
    }

    // Create ndarray from the data
    let array = Array::from_shape_vec(IxDyn(&shape), data)?;

    println!("Created array with shape: {:?}", array.shape());

    // Compress the array with different algorithms
    for algorithm in &[
        CompressionAlgorithm::Gzip,
        CompressionAlgorithm::Zstd,
        CompressionAlgorithm::Lz4,
        CompressionAlgorithm::Bzip2,
    ] {
        // Define output path
        let extension = algorithm.extension();
        let output_path = format!("array_compressed.{extension}");

        // Compress the array
        compress_array(&output_path, &array, *algorithm, Some(6), None)?;

        // Get file size
        let file_size = std::fs::metadata(&output_path)?.len();
        let original_size = array.len() * std::mem::size_of::<f64>();

        println!(
            "Compressed with {}: {:.2}MB → {:.2}MB (ratio: {:.2}x)",
            algorithm.extension(),
            original_size as f64 / 1024.0 / 1024.0,
            file_size as f64 / 1024.0 / 1024.0,
            original_size as f64 / file_size as f64
        );

        // Decompress the array
        let decompressed_array: ArrayBase<OwnedRepr<f64>, IxDyn> = decompress_array(&output_path)?;

        // Verify the shape
        assert_eq!(decompressed_array.shape(), array.shape());

        // Verify a few elements
        let indices_to_check = [(0, 0, 0), (50, 50, 5), (99, 99, 9)];

        for &(i, j, k) in &indices_to_check {
            let original = array[[i, j, k]];
            let decompressed = decompressed_array[[i, j, k]];
            assert!((original - decompressed).abs() < 1e-10);
        }

        println!("  Decompression verified successfully");

        // Clean up
        std::fs::remove_file(output_path)?;
    }

    Ok(())
}

#[allow(dead_code)]
fn chunked_array_compression_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Chunked Array Compression Example ===");

    // Create a larger array to demonstrate chunked compression
    let shape = vec![200, 200, 20]; // 8 million elements
    let mut data = Vec::with_capacity(shape.iter().product());

    // Fill with repeating patterns (good for compression)
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                // Create some spatial patterns
                data.push(
                    (i as f64 / 20.0).sin() + (j as f64 / 20.0).cos() + (k as f64 / 5.0).sin(),
                );
            }
        }
    }

    // Create ndarray from the data
    let array = Array::from_shape_vec(IxDyn(&shape), data)?;

    println!("Created large array with shape: {:?}", array.shape());
    println!("Number of elements: {}", array.len());
    println!(
        "Memory size: {:.2} MB",
        (array.len() * std::mem::size_of::<f64>()) as f64 / 1024.0 / 1024.0
    );

    // Compress the array in chunks
    let output_path = "array_compressed_chunked.zst";
    let chunk_size = 100_000; // Process 100k elements at a time

    println!("Compressing in chunks of {} elements", chunk_size);

    compress_array_chunked(
        output_path,
        &array,
        CompressionAlgorithm::Zstd,
        Some(6),
        chunk_size,
    )?;

    // Get file size
    let file_size = std::fs::metadata(output_path)?.len();
    let original_size = array.len() * std::mem::size_of::<f64>();

    println!(
        "Compressed with chunked ZSTD: {:.2}MB → {:.2}MB (ratio: {:.2}x)",
        original_size as f64 / 1024.0 / 1024.0,
        file_size as f64 / 1024.0 / 1024.0,
        original_size as f64 / file_size as f64
    );

    // Decompress the array
    let (decompressed_array, metadata) = decompress_array_chunked::<_, f64>(output_path)?;

    // Print metadata
    println!("Decompressed with metadata:");
    println!("  Shape: {:?}", metadata.shape);
    println!("  Element type: {}", metadata.dtype);
    println!("  Compression ratio: {:.2}x", metadata.compression_ratio);
    println!("  Additional info: {:?}", metadata.additional_metadata);

    // Verify the shape
    assert_eq!(decompressed_array.shape(), array.shape());

    // Verify a few elements
    let indices_to_check = [(0, 0, 0), (100, 100, 10), (199, 199, 19)];

    for &(i, j, k) in &indices_to_check {
        let original = array[[i, j, k]];
        let decompressed = decompressed_array[[i, j, k]];
        assert!((original - decompressed).abs() < 1e-10);
    }

    println!("  Chunked decompression verified successfully");

    // Clean up
    std::fs::remove_file(output_path)?;

    Ok(())
}

#[allow(dead_code)]
fn compare_algorithms() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Compression Algorithm Comparison ===");

    // Create an array with different patterns to test compression performance
    let shape = vec![100, 100];

    // Create three different types of patterns
    let patterns = vec![
        ("Sine waves", create_sine_wave_array(&shape)),
        ("Random noise", create_random_array(&shape)),
        ("Mixed data", create_mixed_array(&shape)),
    ];

    // Define algorithms to compare
    let algorithms = vec![
        CompressionAlgorithm::Gzip,
        CompressionAlgorithm::Zstd,
        CompressionAlgorithm::Lz4,
        CompressionAlgorithm::Bzip2,
    ];

    // Compare algorithms for each pattern
    for (pattern_name, array) in patterns {
        println!("\nPattern: {}", pattern_name);

        // Get comparison results
        let results = compare_compression_algorithms(&array, &algorithms, Some(6))?;

        // Print results
        println!(
            "  {:<10} | {:<12} | {:<12}",
            "Algorithm", "Ratio", "Size (KB)"
        );
        println!("  {:-<10} | {:-<12} | {:-<12}", "", "", "");

        for (algorithm, ratio, size) in results {
            println!(
                "  {:<10} | {:<12.2} | {:<12.2}",
                format!("{:?}", algorithm),
                ratio,
                size as f64 / 1024.0
            );
        }
    }

    Ok(())
}

// Helper functions to create different test arrays

#[allow(dead_code)]
fn create_sine_wave_array(shape: &[usize]) -> Array<f64, IxDyn> {
    let mut data = Vec::with_capacity(shape.iter().product());

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            // Create smooth sine wave patterns
            data.push((i as f64 / 10.0).sin() * (j as f64 / 15.0).cos());
        }
    }

    Array::from_shape_vec(IxDyn(shape), data).unwrap()
}

#[allow(dead_code)]
fn create_random_array(shape: &[usize]) -> Array<f64, IxDyn> {
    let mut data = Vec::with_capacity(shape.iter().product());

    // Use a proper RNG for randomness
    let mut rng = rand::rng();

    for _ in 0..shape[0] {
        for _ in 0..shape[1] {
            // Random values don't compress well typically
            data.push(rng.random_range(-1.0..1.0));
        }
    }

    Array::from_shape_vec(IxDyn(shape), data).unwrap()
}

#[allow(dead_code)]
fn create_mixed_array(shape: &[usize]) -> Array<f64, IxDyn> {
    let mut data = Vec::with_capacity(shape.iter().product());
    let mut rng = rand::rng();

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            if i < shape[0] / 2 {
                // Half the array is patterned
                data.push((i as f64 / 20.0).sin() + (j as f64 / 20.0).cos());
            } else {
                // Half is random
                data.push(rng.random_range(-1.0..1.0));
            }
        }
    }

    Array::from_shape_vec(IxDyn(shape), data).unwrap()
}
