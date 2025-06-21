//! Example demonstrating memory management utilities for scientific computing
//!
//! This example shows:
//! - Using the global buffer pool for efficient memory allocation
//! - Zero-copy views for efficient data access
//! - 2D chunk processing for memory-efficient operations on large arrays

#[cfg(not(feature = "memory_management"))]
fn main() {
    println!("This example requires the 'memory_management' feature to be enabled.");
    println!("Run with: cargo run --example memory_management --features memory_management");
}

#[cfg(feature = "memory_management")]
use ndarray::Array2;
#[cfg(feature = "memory_management")]
use scirs2_core::memory::{global_buffer_pool, BufferPool, ChunkProcessor2D, ZeroCopyView};

#[cfg(feature = "memory_management")]
fn main() {
    println!("Memory Management Example");

    println!("\n--- Chunk Processing Example ---");
    chunk_processing_example();

    println!("\n--- Buffer Pool Example ---");
    buffer_pool_example();

    println!("\n--- Zero-Copy Example ---");
    zero_copy_example();

    println!("\n--- Global Buffer Pool Example ---");
    global_buffer_pool_example();
}

#[cfg(feature = "memory_management")]
fn chunk_processing_example() {
    // Create a large 2D array
    let rows = 10;
    let cols = 10;
    let array = Array2::from_shape_fn((rows, cols), |(i, j)| i * cols + j);

    println!("Original array:");
    print_array(&array);

    // Process the array in chunks
    let chunk_size = (3, 3);
    let mut processor = ChunkProcessor2D::new(&array, chunk_size);

    println!("\nProcessing in chunks of size {:?}:", chunk_size);
    processor.process_chunks(|chunk, (row, col)| {
        println!("Processing chunk at position ({}, {}):", row, col);
        print_array(chunk);
    });
}

#[cfg(feature = "memory_management")]
fn buffer_pool_example() {
    // Create a buffer pool
    let mut pool = BufferPool::<f64>::new();

    // Acquire buffers
    println!("Acquiring buffers from the pool...");
    let mut buffer1 = pool.acquire_vec(5);
    let mut buffer2 = pool.acquire_vec(10);

    // Fill buffers with data
    for (i, elem) in buffer1.iter_mut().enumerate() {
        *elem = i as f64 * 2.0;
    }

    for (i, elem) in buffer2.iter_mut().enumerate() {
        *elem = i as f64 * 3.0;
    }

    println!("Buffer 1: {:?}", buffer1);
    println!("Buffer 2: {:?}", buffer2);

    // Release buffers back to the pool
    println!("Releasing buffers back to the pool...");
    pool.release_vec(buffer1);
    pool.release_vec(buffer2);

    // Acquire a buffer again
    let buffer3 = pool.acquire_vec(8);
    println!(
        "Acquired buffer from pool: capacity = {}",
        buffer3.capacity()
    );
}

#[cfg(feature = "memory_management")]
fn zero_copy_example() {
    // Create an array
    let array = Array2::from_shape_fn((4, 4), |(i, j)| i * 4 + j);

    println!("Original array:");
    print_array(&array);

    // Create a zero-copy view
    let view = ZeroCopyView::new(&array);

    // Transform the data without making a copy
    let transformed = view.transform(|&x| x * 2);

    println!("\nTransformed array (x2):");
    print_array(&transformed);
}

#[cfg(feature = "memory_management")]
fn global_buffer_pool_example() {
    // Get a reference to the global buffer pool
    let pool = global_buffer_pool();

    // Get a type-specific pool
    let f32_pool = pool.get_pool::<f32>();

    // Acquire a buffer
    let mut buffer = f32_pool.lock().unwrap().acquire_vec(5);

    // Fill the buffer
    for (i, elem) in buffer.iter_mut().enumerate() {
        *elem = i as f32 * 1.5;
    }

    println!("Buffer from global pool: {:?}", buffer);

    // Release the buffer
    f32_pool.lock().unwrap().release_vec(buffer);
    println!("Buffer released back to the global pool");
}

#[cfg(feature = "memory_management")]
fn print_array<T: std::fmt::Display + Copy>(
    array: &ndarray::ArrayBase<impl ndarray::Data<Elem = T>, ndarray::Dim<[usize; 2]>>,
) {
    for row in array.rows() {
        for &value in row.iter() {
            print!("{:4} ", value);
        }
        println!();
    }
}
