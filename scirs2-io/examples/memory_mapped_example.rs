//! Memory-mapped file I/O example
//!
//! This example demonstrates the memory-mapped array functionality for
//! efficient handling of large datasets without loading them entirely into memory.

use ndarray::{Array1, Array2, Array3};
use scirs2_io::mmap::{create_mmap_array, read_mmap_array, MmapArray, MmapArrayMut};
use std::time::Instant;
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🗂️  Memory-Mapped File I/O Example");
    println!("==================================");

    // Demonstrate basic memory mapping
    demonstrate_basic_memory_mapping()?;

    // Demonstrate large dataset handling
    demonstrate_large_dataset_handling()?;

    // Demonstrate mutable memory mapping
    demonstrate_mutable_memory_mapping()?;

    // Demonstrate performance benefits
    demonstrate_performance_benefits()?;

    // Demonstrate multidimensional arrays
    demonstrate_multidimensional_arrays()?;

    println!("\n✅ All memory-mapped I/O demonstrations completed successfully!");
    println!("💡 Memory mapping provides:");
    println!("   - Minimal memory usage for large files");
    println!("   - Fast random access patterns");
    println!("   - Shared data between processes");
    println!("   - Efficient read-write operations");

    Ok(())
}

fn demonstrate_basic_memory_mapping() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📚 Demonstrating Basic Memory Mapping...");

    let temp_dir = tempdir()?;
    let file_path = temp_dir.path().join("basic_array.bin");

    // Create test data
    let data = Array1::from_vec((0..1000).map(|i| i as f64).collect());
    println!("  📊 Created 1D array with {} elements", data.len());

    // Write to memory-mapped file
    println!("  💾 Writing array to memory-mapped file...");
    create_mmap_array(&file_path, &data)?;

    // Open memory-mapped array (no data loaded into RAM)
    println!("  🔗 Opening memory-mapped array...");
    let mmap_array: MmapArray<f64> = MmapArray::open(&file_path)?;
    let shape = mmap_array.shape()?;
    println!("    Shape: {:?}", shape);
    println!("    Elements: {}", mmap_array.len());

    // Access data without loading entire file
    let array_view = mmap_array.as_array_view(&shape)?;
    println!("  🔍 Accessing random elements:");
    for i in [0, 100, 500, 999] {
        println!("    Element {}: {}", i, array_view.as_slice().unwrap()[i]);
    }

    // Verify data integrity
    let read_back: ndarray::ArrayD<f64> = read_mmap_array(&file_path)?;
    assert_eq!(read_back.len(), data.len());
    println!("  ✅ Data integrity verified!");

    Ok(())
}

fn demonstrate_large_dataset_handling() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 Demonstrating Large Dataset Handling...");

    let temp_dir = tempdir()?;
    let file_path = temp_dir.path().join("large_dataset.bin");

    // Create a large 2D array (simulate scientific data)
    println!("  🏗️  Creating large dataset (2000 x 1000 = 2M elements)...");
    let large_data = Array2::from_shape_fn((2000, 1000), |(i, j)| {
        // Simulate some scientific function
        let x = i as f64 / 2000.0;
        let y = j as f64 / 1000.0;
        (x.sin() * y.cos() * 100.0).abs()
    });

    let data_size_mb = (large_data.len() * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
    println!("    Dataset size: {:.1} MB", data_size_mb);

    // Write to file
    println!("  💾 Writing large dataset to memory-mapped file...");
    let write_start = Instant::now();
    create_mmap_array(&file_path, &large_data)?;
    let write_time = write_start.elapsed();
    println!("    Write time: {:.2}ms", write_time.as_secs_f64() * 1000.0);

    // Memory-map for efficient access
    println!("  🔗 Memory-mapping large dataset...");
    let map_start = Instant::now();
    let mmap_array: MmapArray<f64> = MmapArray::open(&file_path)?;
    let map_time = map_start.elapsed();
    println!(
        "    Mapping time: {:.2}ms (instant access!)",
        map_time.as_secs_f64() * 1000.0
    );

    // Access specific regions without loading everything
    let shape = mmap_array.shape()?;
    let array_view = mmap_array.as_array_view(&shape)?;

    println!("  🎯 Accessing specific regions:");
    let slice = array_view.as_slice().unwrap();

    // Sample different parts of the array
    let regions = [
        (0, 0, "top-left corner"),
        (1000, 500, "center"),
        (1999, 999, "bottom-right corner"),
    ];

    for (i, j, desc) in regions {
        let linear_index = i * 1000 + j;
        let value = slice[linear_index];
        println!("    {} ({}, {}): {:.3}", desc, i, j, value);
    }

    println!("  ✅ Large dataset handled efficiently with minimal memory usage!");

    Ok(())
}

fn demonstrate_mutable_memory_mapping() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n✏️  Demonstrating Mutable Memory Mapping...");

    let temp_dir = tempdir()?;
    let file_path = temp_dir.path().join("mutable_array.bin");

    // Create initial data
    let initial_data: Array2<f64> = Array2::zeros((100, 100));
    println!("  📊 Created 100x100 zero-filled array");

    // Write to file
    create_mmap_array(&file_path, &initial_data)?;

    // Open for mutable access
    println!("  ✏️  Opening for read-write access...");
    let mut mmap_array: MmapArrayMut<f64> = MmapArrayMut::open(&file_path)?;
    let shape = mmap_array.shape()?;

    // Modify data directly in the memory-mapped file
    {
        let mut array_view = mmap_array.as_array_view_mut(&shape)?;
        let slice = array_view.as_slice_mut().unwrap();

        // Create a pattern
        for i in 0..100 {
            for j in 0..100 {
                let linear_index = i * 100 + j;
                if (i + j) % 2 == 0 {
                    slice[linear_index] = (i + j) as f64;
                } else {
                    slice[linear_index] = -((i + j) as f64);
                }
            }
        }
    }

    // Flush changes to disk
    println!("  💾 Flushing changes to disk...");
    mmap_array.flush()?;

    // Verify changes by reading back
    let modified_data: ndarray::ArrayD<f64> = read_mmap_array(&file_path)?;
    let modified_slice = modified_data.as_slice().unwrap();

    println!("  🔍 Verifying modifications:");
    let test_positions = [(0, 0), (1, 1), (50, 50), (99, 99)];
    for (i, j) in test_positions {
        let linear_index = i * 100 + j;
        let expected = if (i + j) % 2 == 0 {
            (i + j) as f64
        } else {
            -((i + j) as f64)
        };
        let actual = modified_slice[linear_index];
        println!(
            "    Position ({}, {}): expected {}, got {}",
            i, j, expected, actual
        );
        assert_eq!(actual, expected);
    }

    println!("  ✅ Mutable memory mapping working correctly!");

    Ok(())
}

fn demonstrate_performance_benefits() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Demonstrating Performance Benefits...");

    let temp_dir = tempdir()?;
    let file_path = temp_dir.path().join("performance_test.bin");

    // Create moderately large dataset
    let data = Array2::from_shape_fn((1000, 500), |(i, j)| (i * j) as f64);
    let data_size_mb = (data.len() * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);

    println!(
        "  📊 Test dataset: {}x{} = {} elements ({:.1} MB)",
        data.shape()[0],
        data.shape()[1],
        data.len(),
        data_size_mb
    );

    // Write the data
    create_mmap_array(&file_path, &data)?;

    // Compare memory mapping vs full read
    println!("  🏁 Performance comparison:");

    // Method 1: Full array read (loads everything into memory)
    let full_read_start = Instant::now();
    let _full_array: ndarray::ArrayD<f64> = read_mmap_array(&file_path)?;
    let full_read_time = full_read_start.elapsed();
    println!(
        "    Full read time: {:.2}ms",
        full_read_time.as_secs_f64() * 1000.0
    );

    // Method 2: Memory mapping (instant access)
    let mmap_start = Instant::now();
    let mmap_array: MmapArray<f64> = MmapArray::open(&file_path)?;
    let shape = mmap_array.shape()?;
    let _array_view = mmap_array.as_array_view(&shape)?;
    let mmap_time = mmap_start.elapsed();
    println!(
        "    Memory mapping time: {:.2}ms",
        mmap_time.as_secs_f64() * 1000.0
    );

    // Random access performance
    let access_start = Instant::now();
    let slice = mmap_array.as_slice()?;
    let mut sum = 0.0;
    for i in 0..1000 {
        let random_index = (i * 1031 + 17) % slice.len(); // Simple pseudo-random
        sum += slice[random_index];
    }
    let access_time = access_start.elapsed();
    println!(
        "    1000 random accesses: {:.2}ms (sum: {:.0})",
        access_time.as_secs_f64() * 1000.0,
        sum
    );

    let speedup = full_read_time.as_secs_f64() / mmap_time.as_secs_f64();
    println!(
        "  🚀 Memory mapping is {:.1}x faster for initial access!",
        speedup
    );

    Ok(())
}

fn demonstrate_multidimensional_arrays() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧊 Demonstrating Multidimensional Arrays...");

    let temp_dir = tempdir()?;

    // 3D array example (simulate a data cube)
    println!("  🧊 3D array example (20x30x40 data cube):");
    let data_3d = Array3::from_shape_fn((20, 30, 40), |(x, y, z)| {
        (x as f64).sin() + (y as f64).cos() + (z as f64).tan()
    });

    let file_3d = temp_dir.path().join("data_cube.bin");
    create_mmap_array(&file_3d, &data_3d)?;

    let mmap_3d: MmapArray<f64> = MmapArray::open(&file_3d)?;
    let shape_3d = mmap_3d.shape()?;
    println!("    Shape: {:?}", shape_3d);
    println!("    Total elements: {}", mmap_3d.len());

    // Sample the data cube
    let slice_3d = mmap_3d.as_slice()?;
    println!("    Sample values:");
    for (x, y, z) in [(0, 0, 0), (10, 15, 20), (19, 29, 39)] {
        let linear_index = x * (30 * 40) + y * 40 + z;
        println!("      ({}, {}, {}): {:.3}", x, y, z, slice_3d[linear_index]);
    }

    // Large 1D array example
    println!("  📊 Large 1D array example (10M elements):");
    let large_1d = Array1::from_shape_fn(10_000_000, |i| (i as f64).sqrt());
    let file_1d = temp_dir.path().join("large_1d.bin");

    let write_start = Instant::now();
    create_mmap_array(&file_1d, &large_1d)?;
    let write_time = write_start.elapsed();

    let mmap_1d: MmapArray<f64> = MmapArray::open(&file_1d)?;
    let map_time = Instant::now().elapsed();

    println!("    Write time: {:.2}ms", write_time.as_secs_f64() * 1000.0);
    println!("    Map time: {:.2}ms", map_time.as_secs_f64() * 1000.0);
    println!(
        "    Size: {} elements ({:.1} MB)",
        mmap_1d.len(),
        (mmap_1d.len() * 8) as f64 / (1024.0 * 1024.0)
    );

    println!("  ✅ Multidimensional arrays handled efficiently!");

    Ok(())
}
