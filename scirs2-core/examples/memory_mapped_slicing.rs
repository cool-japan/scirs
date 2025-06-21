use ndarray::{s, Array2};
use scirs2_core::memory_efficient::{MemoryMappedArray, MemoryMappedSlicing};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Memory-Mapped Array Slicing Example");
    println!("=====================================\n");

    // Create a temporary directory for our example files
    let dir = tempdir()?;
    let file_path = dir.path().join("large_matrix.bin");
    println!("Creating a test file at: {}", file_path.display());

    // Create a large test matrix (100x100)
    let rows = 100;
    let cols = 100;
    let size = rows * cols;
    println!("Creating a {}x{} matrix with {} elements", rows, cols, size);

    // Initialize with data (i*100 + j pattern)
    let matrix = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i * 100 + j) as f64);

    // Save the matrix to a file
    let mut file = File::create(&file_path)?;
    for val in matrix.iter() {
        file.write_all(&val.to_ne_bytes())?;
    }
    drop(file);
    println!(
        "Matrix saved to file (size: {} bytes)",
        rows * cols * std::mem::size_of::<f64>()
    );

    // Open as memory-mapped array
    let mmap = MemoryMappedArray::<f64>::open(&file_path, &[rows, cols])?;
    println!("\nOpened as memory-mapped array");
    println!("Full array shape: {:?}", mmap.shape);

    // Demonstrate different slicing operations
    demo_basic_slicing(&mmap)?;
    demo_complex_slicing(&mmap)?;
    demo_ndarray_syntax(&mmap)?;
    demo_slice_chaining(&mmap, &file_path)?;

    println!("\nAll examples completed successfully!");

    Ok(())
}

fn demo_basic_slicing(mmap: &MemoryMappedArray<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n1. Basic Slicing Example");
    println!("------------------------");

    // Create a small slice from the middle of the array
    let row_start = 25;
    let row_end = 30;
    let col_start = 25;
    let col_end = 30;

    println!(
        "Creating slice of rows {}..{} and columns {}..{}",
        row_start, row_end, col_start, col_end
    );

    let slice = mmap.slice_2d(row_start..row_end, col_start..col_end)?;
    println!("Slice shape: {:?}", slice.shape());

    // Load the slice into memory
    let array = slice.load()?;

    // Display the slice data
    println!("\nSlice content:");
    for i in 0..array.shape()[0] {
        print!("Row {}: ", i + row_start);
        for j in 0..array.shape()[1] {
            print!("{:.0} ", array[[i, j]]);
        }
        println!();
    }

    println!("\nNotice that only the slice was loaded into memory, not the entire array.");

    Ok(())
}

fn demo_complex_slicing(mmap: &MemoryMappedArray<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2. Complex Slicing Example");
    println!("--------------------------");

    // Demonstrate different range types
    println!("Using different range types:");

    // Inclusive range (..=)
    println!("\nInclusive range (rows 40..=45, cols 10..=15):");
    let slice1 = mmap.slice_2d(40..=45, 10..=15)?;
    let array1 = slice1.load()?;
    println!("Shape: {:?}", array1.shape());
    println!(
        "First element: {:.0}, Last element: {:.0}",
        array1[(0, 0)],
        array1[(array1.shape()[0] - 1, array1.shape()[1] - 1)]
    );

    // Open-ended range (..)
    println!("\nOpen-ended range (first 5 rows, all columns):");
    let slice2 = mmap.slice_2d(0..5, ..)?;
    let array2 = slice2.load()?;
    println!("Shape: {:?}", array2.shape());
    println!(
        "Sample values: {:.0}, {:.0}, {:.0}",
        array2[[0, 0]],
        array2[[2, 50]],
        array2[[4, 99]]
    );

    // Range from
    println!("\nRange from (rows from 95, columns from 95):");
    let slice3 = mmap.slice_2d(95.., 95..)?;
    let array3 = slice3.load()?;
    println!("Shape: {:?}", array3.shape());
    println!(
        "Sample values: {:.0}, {:.0}, {:.0}",
        array3[[0, 0]],
        array3[[2, 2]],
        array3[[4, 4]]
    );

    Ok(())
}

fn demo_ndarray_syntax(mmap: &MemoryMappedArray<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n3. Using ndarray Slice Syntax");
    println!("-----------------------------");

    // Using ndarray's s![] macro
    println!("Using ndarray's s![] macro for slicing:");

    // Regular slice with s![]
    println!("\nRegular slice with s![10..15, 20..25]:");
    let slice1 = mmap.slice(s![10..15, 20..25])?;
    let array1 = slice1.load()?;
    println!("Shape: {:?}", array1.shape());
    println!(
        "First element: {:.0}, Last element: {:.0}",
        array1[(0, 0)],
        array1[(array1.shape()[0] - 1, array1.shape()[1] - 1)]
    );

    // Stride with s![]
    println!("\nStride with s![50..70;2, 50..70;2] (every other element):");
    let slice2 = mmap.slice(s![50..70;2, 50..70;2])?;
    let array2 = slice2.load()?;
    println!("Shape: {:?}", array2.shape());
    println!("Values should increase by 2 in each dimension:");
    for i in 0..3 {
        print!("Row {}: ", i);
        for j in 0..3 {
            print!("{:.0} ", array2[(i, j)]);
        }
        println!();
    }

    Ok(())
}

fn demo_slice_chaining(
    mmap: &MemoryMappedArray<f64>,
    base_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n4. Slice Chaining Example");
    println!("-------------------------");

    // First slice the array
    println!("First slice: rows 10..60, columns 10..60");
    let slice1 = mmap.slice_2d(10..60, 10..60)?;
    println!("First slice shape: {:?}", slice1.shape());

    // Save the first slice to a file to demonstrate memory mapping it
    let slice1_path = base_path.with_file_name("slice1.bin");
    let slice1_data = slice1.load()?;

    let mut file = File::create(&slice1_path)?;
    for val in slice1_data.iter() {
        file.write_all(&val.to_ne_bytes())?;
    }
    drop(file);
    println!("First slice saved to: {}", slice1_path.display());

    // Memory map the first slice
    let slice1_mmap = MemoryMappedArray::<f64>::open(&slice1_path, &[50, 50])?;

    // Create a further slice from the first slice
    println!("\nSecond slice: rows 10..20, columns 10..20 (from the first slice)");
    let slice2 = slice1_mmap.slice_2d(10..20, 10..20)?;
    println!("Second slice shape: {:?}", slice2.shape());

    // Load and display the second slice
    let array2 = slice2.load()?;
    println!("\nSecond slice content (first few elements):");
    for i in 0..3 {
        print!("Row {}: ", i);
        for j in 0..3 {
            print!("{:.0} ", array2[(i, j)]);
        }
        println!();
    }

    // Calculate the original coordinates
    println!("\nCorresponding to original coordinates:");
    println!("Second slice coordinates (0,0) -> Original coordinates (20,20)");
    println!(
        "Value at original[20,20]: {:.0}",
        mmap.readonly_array()?[(20, 20)]
    );
    println!("Value at slice2[0,0]: {:.0}", array2[(0, 0)]);

    Ok(())
}
