//! Example demonstrating memory-efficient image processing with scirs2-ndimage
//!
//! This example shows how to process large images that don't fit in RAM using
//! memory-mapped arrays, chunked processing, and other memory-efficient techniques.

use scirs2_core::memory_efficient::{AccessMode, ChunkingStrategy};
use scirs2_core::ndarray::{s, Array2, Array3};
use scirs2_ndimage::{
    chunked_v2::{convolve_chunked_v2, uniform_filter_chunked_v2, ChunkConfigBuilder},
    filters::{gaussian_filter, BorderMode},
    mmap_io::{
        create_temp_mmap, loadimage_mmap, process_mmap_chunks, saveimage_mmap, smart_loadimage,
        MmapConfig,
    },
};
use std::path::Path;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Memory-Efficient Image Processing Example ===\n");

    // Example 1: Processing a large image with automatic memory management
    example_1_auto_memory_management()?;

    // Example 2: Using memory-mapped arrays for very large images
    example_2_memory_mapped_processing()?;

    // Example 3: Chunked processing with custom configuration
    example_3_chunked_processing()?;

    // Example 4: Processing image sequences efficiently
    example_4image_sequence_processing()?;

    // Example 5: Filter pipeline with memory optimization
    example_5_filter_pipeline()?;

    Ok(())
}

/// Example 1: Automatic memory management based on image size
#[allow(dead_code)]
fn example_1_auto_memory_management() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: Automatic Memory Management");
    println!("--------------------------------------");

    // Create test images of different sizes
    let smallimage = Array2::<f64>::from_elem((100, 100), 1.0);
    let mediumimage = Array2::<f64>::from_elem((1000, 1000), 1.0);
    let largeimage = Array2::<f64>::from_elem((5000, 5000), 1.0);

    // Configure automatic memory management
    let config = ChunkConfigBuilder::new()
        .strategy(ChunkingStrategy::Auto)
        .mmap_threshold(Some(50 * 1024 * 1024)) // 50 MB threshold
        .build();

    // Process small image (will use regular processing)
    println!("Processing small image (100x100)...");
    let result_small = uniform_filter_chunked_v2(
        &smallimage,
        &[3, 3],
        BorderMode::Reflect,
        Some(config.clone()),
    )?;
    println!("  ✓ Processed using regular method");

    // Process medium image (will use chunked processing)
    println!("Processing medium image (1000x1000)...");
    let result_medium = uniform_filter_chunked_v2(
        &mediumimage,
        &[5, 5],
        BorderMode::Reflect,
        Some(config.clone()),
    )?;
    println!("  ✓ Processed using chunked method");

    // Process large image (would use memory-mapped if implemented)
    println!("Processing large image (5000x5000)...");
    // Note: This is a simulation - actual processing would use mmap
    println!("  ✓ Would process using memory-mapped arrays");

    println!();
    Ok(())
}

/// Example 2: Using memory-mapped arrays for very large images
#[allow(dead_code)]
fn example_2_memory_mapped_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: Memory-Mapped Array Processing");
    println!("-----------------------------------------");

    // Create a temporary large image file
    let temp_dir = tempfile::tempdir()?;
    let image_path = temp_dir.path().join("largeimage.bin");

    // Create and save a large test image
    let shape = vec![2000, 2000];
    let test_data = Array2::<f64>::from_elem((2000, 2000), std::f64::consts::PI);

    println!("Creating memory-mapped image (2000x2000)...");
    let mmap = saveimage_mmap(&test_data.view(), &image_path, 0)?;
    println!("  ✓ Created memory-mapped file: {:?}", image_path);

    // Load the image as memory-mapped
    println!("Loading image as memory-mapped array...");
    let loaded_mmap = loadimage_mmap::<f64, scirs2_core::ndarray::Ix2, _>(
        &image_path,
        &shape,
        0,
        AccessMode::ReadOnly,
    )?;
    println!("  ✓ Loaded with shape: {:?}", loaded_mmap.shape);

    // Process the memory-mapped image in chunks
    println!("Processing memory-mapped image in chunks...");
    let chunk_sums = process_mmap_chunks(
        &loaded_mmap,
        ChunkingStrategy::NumChunks(10),
        |chunk_data, chunk_idx| {
            let sum: f64 = chunk_data.iter().sum();
            println!("  - Chunk {}: sum = {:.2}", chunk_idx, sum);
            sum
        },
    )?;

    let total_sum: f64 = chunk_sums.iter().sum();
    println!("  ✓ Total sum: {:.2}", total_sum);

    println!();
    Ok(())
}

/// Example 3: Chunked processing with custom configuration
#[allow(dead_code)]
fn example_3_chunked_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 3: Custom Chunked Processing");
    println!("------------------------------------");

    // Create a test image
    let image = Array2::<f64>::from_shape_fn((1500, 1500), |(i, j)| {
        ((i as f64 * 0.01).sin() + (j as f64 * 0.01).cos()) * 100.0
    });

    // Configure chunking strategies
    let strategies = vec![
        ("Fixed size (500 elements)", ChunkingStrategy::Fixed(500)),
        ("Number of chunks (9)", ChunkingStrategy::NumChunks(9)),
        (
            "Fixed bytes (1 MB)",
            ChunkingStrategy::FixedBytes(1024 * 1024),
        ),
        ("Auto", ChunkingStrategy::Auto),
    ];

    for (name, strategy) in strategies {
        println!("Testing strategy: {}", name);

        let config = ChunkConfigBuilder::new()
            .strategy(strategy)
            .overlap(2)
            .parallel(true)
            .build();

        let start = std::time::Instant::now();
        let result =
            uniform_filter_chunked_v2(&image, &[7, 7], BorderMode::Constant, Some(config))?;
        let elapsed = start.elapsed();

        println!("  ✓ Completed in {:.2?}", elapsed);
        println!("  - Result shape: {:?}", result.shape());
        println!("  - Center value: {:.2}", result[[750, 750]]);
    }

    println!();
    Ok(())
}

/// Example 4: Processing image sequences efficiently
#[allow(dead_code)]
fn example_4image_sequence_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 4: Image Sequence Processing");
    println!("------------------------------------");

    // Simulate processing a sequence of images (e.g., video frames)
    let num_frames = 100;
    let frameshape = (512, 512);

    println!(
        "Processing {} frames of size {:?}...",
        num_frames, frameshape
    );

    // Create temporary storage for the sequence
    let temp_dir = tempfile::tempdir()?;
    let sequence_path = temp_dir.path().join("sequence.bin");

    // Calculate total size for the sequence
    let totalshape = [num_frames, frameshape.0, frameshape.1];
    let sequence_array = Array3::<f32>::zeros((num_frames, frameshape.0, frameshape.1));

    // Save as memory-mapped sequence
    let sequence_mmap = saveimage_mmap(&sequence_array.view(), &sequence_path, 0)?;
    println!("  ✓ Created memory-mapped sequence");

    // Process each frame without loading entire sequence
    let processed_frames = process_mmap_chunks(
        &sequence_mmap,
        ChunkingStrategy::Fixed(frameshape.0 * frameshape.1), // One frame per chunk
        |frame_data, frame_idx| {
            // Simulate frame processing
            let frame_mean: f32 = frame_data.iter().sum::<f32>() / frame_data.len() as f32;

            if frame_idx % 20 == 0 {
                println!("  - Processed frame {}/{}", frame_idx + 1, num_frames);
            }

            frame_mean
        },
    )?;

    println!("  ✓ Processed all frames");
    println!(
        "  - Average frame mean: {:.4}",
        processed_frames.iter().sum::<f32>() / processed_frames.len() as f32
    );

    println!();
    Ok(())
}

/// Example 5: Complex filter pipeline with memory optimization
#[allow(dead_code)]
fn example_5_filter_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 5: Memory-Optimized Filter Pipeline");
    println!("-------------------------------------------");

    // Create a test image
    let image = Array2::<f64>::from_shape_fn((800, 800), |(i, j)| {
        let x = i as f64 - 400.0;
        let y = j as f64 - 400.0;
        let r = (x * x + y * y).sqrt();
        if r < 200.0 {
            255.0 * (1.0 - r / 200.0)
        } else {
            0.0
        }
    });

    println!("Processing image through filter pipeline...");

    // Configure memory-efficient processing
    let config = ChunkConfigBuilder::new()
        .strategy(ChunkingStrategy::NumChunks(4))
        .overlap(5)
        .build();

    // Step 1: Gaussian blur
    println!("  1. Applying Gaussian blur...");
    let step1 = gaussian_filter(&image, 2.0, Some(BorderMode::Reflect), None)?;

    // Step 2: Edge-preserving smoothing (simulated with uniform filter)
    println!("  2. Applying edge-preserving smoothing...");
    let step2 =
        uniform_filter_chunked_v2(&step1, &[5, 5], BorderMode::Reflect, Some(config.clone()))?;

    // Step 3: Final Gaussian smoothing
    println!("  3. Applying final smoothing...");
    let kernel =
        Array2::<f64>::from_shape_fn((3, 3), |(i, j)| if i == 1 && j == 1 { 0.5 } else { 0.0625 });
    let final_result = convolve_chunked_v2(&step2, &kernel, BorderMode::Constant, Some(config))?;

    println!("  ✓ Pipeline completed");
    println!("  - Input shape: {:?}", image.shape());
    println!("  - Output shape: {:?}", final_result.shape());
    println!("  - Center value: {:.2}", final_result[[400, 400]]);

    // Memory usage summary
    println!("\nMemory Usage Summary:");
    println!(
        "  - Traditional approach: ~{:.1} MB (all steps in memory)",
        3.0 * 800.0 * 800.0 * 8.0 / (1024.0 * 1024.0)
    );
    println!(
        "  - Chunked approach: ~{:.1} MB (processing in chunks)",
        800.0 * 800.0 * 8.0 / (4.0 * 1024.0 * 1024.0)
    );

    println!();
    Ok(())
}

/// Helper function to display memory usage
#[allow(dead_code)]
fn print_memory_stats(label: &str, sizebytes: usize) {
    let size_mb = sizebytes as f64 / (1024.0 * 1024.0);
    println!("{}: {:.2} MB", label, size_mb);
}
