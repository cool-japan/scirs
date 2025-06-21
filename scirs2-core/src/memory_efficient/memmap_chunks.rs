//! Chunked processing extension for memory-mapped arrays.
//!
//! This module provides utilities for working with large memory-mapped arrays
//! in a memory-efficient manner by processing them in smaller chunks.
//!
//! ## Overview
//!
//! Memory-mapped arrays allow working with data that doesn't fit entirely in RAM,
//! but processing such large arrays can still be challenging. This module extends
//! `MemoryMappedArray` with chunked processing capabilities, enabling efficient
//! processing of large datasets through a combination of:
//!
//! - Processing data in manageable chunks to control memory usage
//! - Providing both iterator-based and callback-based processing APIs
//! - Supporting various chunking strategies to optimize for different workloads
//! - Ensuring mutations are properly persisted to the underlying memory-mapped file
//!
//! ## Usage
//!
//! There are three main ways to process chunks:
//!
//! 1. Using `process_chunks` for reading/processing chunks:
//!    ```
//!    // Process chunks of a large array and collect results
//!    let results = mmap.process_chunks(
//!        ChunkingStrategy::Fixed(1000),
//!        |chunk_data, chunk_idx| {
//!            // Process this chunk and return a result
//!            chunk_data.iter().sum::<f64>()
//!        }
//!    );
//!    ```
//!
//! 2. Using `process_chunks_mut` for mutating chunks:
//!    ```
//!    // Modify each chunk in-place
//!    mmap.process_chunks_mut(
//!        ChunkingStrategy::NumChunks(10),
//!        |chunk_data, chunk_idx| {
//!            // Modify the chunk data in-place
//!            for i in 0..chunk_data.len() {
//!                chunk_data[i] *= 2.0;
//!            }
//!        }
//!    );
//!    ```
//!
//! 3. Using the `chunks` iterator for element-by-element processing:
//!    ```
//!    // Process chunks using iterator
//!    for chunk in mmap.chunks(ChunkingStrategy::Auto) {
//!        // Each chunk is an Array1 of the appropriate type
//!        println!("Chunk sum: {}", chunk.sum());
//!    }
//!    ```
//!
//! 4. If you have the `parallel` feature enabled, you can also use parallel processing:
//!    ```
//!    // Process chunks in parallel
//!    let results = mmap.process_chunks_parallel(
//!        ChunkingStrategy::Fixed(1000),
//!        |chunk_data, chunk_idx| {
//!            chunk_data.iter().sum::<f64>()
//!        }
//!    );
//!    ```
//!
//! ## Chunking Strategies
//!
//! This module supports different chunking strategies:
//!
//! - `ChunkingStrategy::Fixed(size)`: Fixed-size chunks
//! - `ChunkingStrategy::NumChunks(n)`: Divide the array into a specific number of chunks
//! - `ChunkingStrategy::Auto`: Automatically determine a reasonable chunk size
//! - `ChunkingStrategy::FixedBytes(bytes)`: Chunks with a specific size in bytes
//!
//! ## Limitations
//!
//! - Currently only works with 1D arrays (1-dimensional arrays only)
//! - For mutating operations, the module uses direct file I/O to ensure changes are
//!   properly persisted to disk, which may be slower than memory-only operations

use crate::memory_efficient::chunked::ChunkingStrategy;
use crate::memory_efficient::memmap::MemoryMappedArray;
use ndarray::Array1;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Extension trait for MemoryMappedArray to enable chunked processing of large datasets.
///
/// This trait extends `MemoryMappedArray` with methods for processing large datasets
/// in manageable chunks, which helps to control memory usage and enables working with
/// arrays that might be too large to fit entirely in memory.
pub trait MemoryMappedChunks<A: Clone + Copy + 'static + Send + Sync> {
    /// Get the number of chunks for the given chunking strategy.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The chunking strategy to determine chunk sizes
    ///
    /// # Returns
    ///
    /// The number of chunks that the array will be divided into
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::memory_efficient::{create_mmap, AccessMode, ChunkingStrategy, MemoryMappedChunks};
    /// use ndarray::Array1;
    ///
    /// // Create a memory-mapped array with 100 elements
    /// let data = Array1::<f64>::linspace(0., 99., 100);
    /// let file_path = "example.bin";  // In practice, use a proper temporary path
    /// let mmap = create_mmap(&data, file_path.as_ref(), AccessMode::Write, 0).unwrap();
    ///
    /// // Check how many chunks we'll get with different strategies
    /// assert_eq!(mmap.chunk_count(ChunkingStrategy::Fixed(10)), 10);  // 10 chunks of 10 elements each
    /// assert_eq!(mmap.chunk_count(ChunkingStrategy::NumChunks(5)), 5);  // 5 chunks of 20 elements each
    /// ```
    fn chunk_count(&self, strategy: ChunkingStrategy) -> usize;

    /// Process each chunk with a function and collect the results.
    ///
    /// This method divides the array into chunks according to the given strategy,
    /// applies the provided function to each chunk, and collects the results into a vector.
    /// It is efficient for read-only operations on large arrays.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The chunking strategy to determine chunk sizes
    /// * `f` - A function that processes each chunk and returns a result
    ///
    /// # Returns
    ///
    /// A vector containing the results from processing each chunk
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::memory_efficient::{create_mmap, AccessMode, ChunkingStrategy, MemoryMappedChunks};
    /// use ndarray::Array1;
    ///
    /// // Create a memory-mapped array with 100 elements
    /// let data = Array1::<i32>::from_vec((0..100).collect());
    /// let file_path = "example.bin";  // In practice, use a proper temporary path
    /// let mmap = create_mmap(&data, file_path.as_ref(), AccessMode::Write, 0).unwrap();
    ///
    /// // Calculate the sum of each chunk
    /// let chunk_sums = mmap.process_chunks(
    ///     ChunkingStrategy::Fixed(25),
    ///     |chunk, chunk_idx| chunk.iter().sum::<i32>()
    /// );
    ///
    /// // We should have 4 chunks with sums of 0-24, 25-49, 50-74, 75-99
    /// assert_eq!(chunk_sums.len(), 4);
    /// ```
    fn process_chunks<F, R>(&self, strategy: ChunkingStrategy, f: F) -> Vec<R>
    where
        F: Fn(&[A], usize) -> R;

    /// Process each chunk with a mutable function that modifies the data in-place.
    ///
    /// This method divides the array into chunks according to the given strategy
    /// and applies the provided mutable function to each chunk. The function can
    /// modify the chunk data in-place, and the changes will be saved to the
    /// underlying memory-mapped file.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The chunking strategy to determine chunk sizes
    /// * `f` - A function that processes and potentially modifies each chunk
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::memory_efficient::{create_mmap, AccessMode, ChunkingStrategy, MemoryMappedChunks};
    /// use ndarray::Array1;
    ///
    /// // Create a memory-mapped array with 100 zeros
    /// let data = Array1::<f64>::zeros(100);
    /// let file_path = "example.bin";  // In practice, use a proper temporary path
    /// let mut mmap = create_mmap(&data, file_path.as_ref(), AccessMode::Write, 0).unwrap();
    ///
    /// // Modify each chunk: set elements to their index
    /// mmap.process_chunks_mut(
    ///     ChunkingStrategy::Fixed(10),
    ///     |chunk, chunk_idx| {
    ///         for (i, elem) in chunk.iter_mut().enumerate() {
    ///             *elem = (chunk_idx * 10 + i) as f64;
    ///         }
    ///     }
    /// );
    ///
    /// // Now the array contains [0, 1, 2, ..., 99]
    /// ```
    ///
    /// # Notes
    ///
    /// This method uses direct file I/O to ensure changes are properly persisted to disk,
    /// which may be slower than memory-only operations but is more reliable for ensuring
    /// data is properly saved, especially with large datasets.
    fn process_chunks_mut<F>(&mut self, strategy: ChunkingStrategy, f: F)
    where
        F: Fn(&mut [A], usize);
}

/// Extension trait for parallel processing of memory-mapped arrays.
///
/// This trait is only available when the 'parallel' feature is enabled.
/// It extends the `MemoryMappedChunks` trait with parallel processing capabilities.
#[cfg(feature = "parallel")]
pub trait MemoryMappedChunksParallel<A: Clone + Copy + 'static + Send + Sync + Send + Sync>:
    MemoryMappedChunks<A>
{
    /// Process chunks in parallel and collect the results.
    ///
    /// This method works like `process_chunks` but processes the chunks in parallel using Rayon.
    /// It's useful for computationally intensive operations on large datasets.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The chunking strategy to determine chunk sizes
    /// * `f` - A function that processes each chunk and returns a result
    ///
    /// # Returns
    ///
    /// A vector containing the results from processing each chunk, in chunk order
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "parallel")]
    /// # {
    /// use scirs2_core::memory_efficient::{create_mmap, AccessMode, ChunkingStrategy, MemoryMappedChunks, MemoryMappedChunksParallel};
    /// use ndarray::Array1;
    ///
    /// // Create a memory-mapped array with 100 elements
    /// let data = Array1::<i32>::from_vec((0..100).collect());
    /// let file_path = "example.bin";  // In practice, use a proper temporary path
    /// let mmap = create_mmap(&data, file_path.as_ref(), AccessMode::Write, 0).unwrap();
    ///
    /// // Calculate the sum of each chunk in parallel
    /// let chunk_sums = mmap.process_chunks_parallel(
    ///     ChunkingStrategy::Fixed(25),
    ///     |chunk, chunk_idx| chunk.iter().sum::<i32>()
    /// );
    ///
    /// // We should have 4 chunks with sums of 0-24, 25-49, 50-74, 75-99
    /// assert_eq!(chunk_sums.len(), 4);
    /// # }
    /// ```
    fn process_chunks_parallel<F, R>(&self, strategy: ChunkingStrategy, f: F) -> Vec<R>
    where
        F: Fn(&[A], usize) -> R + Send + Sync,
        R: Send;

    /// Process chunks in parallel with a mutable function.
    ///
    /// This method works like `process_chunks_mut` but processes the chunks in parallel using Rayon.
    /// It's useful for computationally intensive operations on large datasets.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The chunking strategy to determine chunk sizes
    /// * `f` - A function that processes and potentially modifies each chunk
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "parallel")]
    /// # {
    /// use scirs2_core::memory_efficient::{create_mmap, AccessMode, ChunkingStrategy, MemoryMappedChunks, MemoryMappedChunksParallel};
    /// use ndarray::Array1;
    ///
    /// // Create a memory-mapped array with 100 zeros
    /// let data = Array1::<f64>::zeros(100);
    /// let file_path = "example.bin";  // In practice, use a proper temporary path
    /// let mut mmap = create_mmap(&data, file_path.as_ref(), AccessMode::Write, 0).unwrap();
    ///
    /// // Modify each chunk in parallel: set elements to their index
    /// mmap.process_chunks_mut_parallel(
    ///     ChunkingStrategy::Fixed(10),
    ///     |chunk, chunk_idx| {
    ///         for (i, elem) in chunk.iter_mut().enumerate() {
    ///             *elem = (chunk_idx * 10 + i) as f64;
    ///         }
    ///     }
    /// );
    /// # }
    /// ```
    ///
    /// # Notes
    ///
    /// Even when used in parallel, this method ensures that file writes are safe
    /// and do not conflict with each other by collecting all modifications and
    /// applying them sequentially.
    fn process_chunks_mut_parallel<F>(&mut self, strategy: ChunkingStrategy, f: F)
    where
        F: Fn(&mut [A], usize) + Send + Sync;
}

/// Iterator over chunks of a memory-mapped array (for 1D arrays only).
///
/// This iterator provides a convenient way to process a memory-mapped array in chunks,
/// returning each chunk as an `Array1<A>`. It's particularly useful for operations where
/// you want to process chunks sequentially and don't need to collect results.
///
/// # Examples
///
/// ```
/// use scirs2_core::memory_efficient::{create_mmap, AccessMode, ChunkingStrategy, MemoryMappedChunkIter};
/// use ndarray::Array1;
///
/// // Create a memory-mapped array
/// let data = Array1::<f64>::linspace(0., 99., 100);
/// let file_path = "example.bin";  // In practice, use a proper temporary path
/// let mmap = create_mmap(&data, file_path.as_ref(), AccessMode::Write, 0).unwrap();
///
/// // Process chunks using iterator
/// let mut sum = 0.0;
/// for chunk in mmap.chunks(ChunkingStrategy::Fixed(10)) {
///     // Each chunk is an Array1<f64>
///     sum += chunk.sum();
/// }
///
/// // The sum should be the same as summing the original array
/// assert!((sum - data.sum()).abs() < 1e-10);
/// ```
pub struct ChunkIter<'a, A>
where
    A: Clone + Copy + 'static + Send + Sync + Send + Sync,
{
    /// Reference to the memory-mapped array
    array: &'a MemoryMappedArray<A>,
    /// Current chunk index
    current_idx: usize,
    /// Total number of chunks
    total_chunks: usize,
    /// Chunking strategy
    strategy: ChunkingStrategy,
}

impl<A> Iterator for ChunkIter<'_, A>
where
    A: Clone + Copy + 'static + Send + Sync,
{
    type Item = Array1<A>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.total_chunks {
            None
        } else {
            let chunk_idx = self.current_idx;
            self.current_idx += 1;

            // Get chunk start/end indices
            let chunk_size = match self.strategy {
                ChunkingStrategy::Fixed(size) => size,
                ChunkingStrategy::NumChunks(n) => self.array.size.div_ceil(n),
                ChunkingStrategy::Auto => (self.array.size / 100).max(1),
                ChunkingStrategy::FixedBytes(bytes) => {
                    let element_size = std::mem::size_of::<A>();
                    let elements_per_chunk = bytes / element_size;
                    elements_per_chunk.max(1)
                }
            };

            let start_idx = chunk_idx * chunk_size;
            let end_idx = (start_idx + chunk_size).min(self.array.size);

            // Get the array data to return a chunk
            if let Ok(array_1d) = self.array.as_array::<ndarray::Ix1>() {
                Some(array_1d.slice(ndarray::s![start_idx..end_idx]).to_owned())
            } else {
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_chunks - self.current_idx;
        (remaining, Some(remaining))
    }
}

impl<A> ExactSizeIterator for ChunkIter<'_, A> where A: Clone + Copy + 'static + Send + Sync {}

/// Extension trait for MemoryMappedArray to enable chunked iteration.
///
/// This trait extends `MemoryMappedArray` with the ability to iterate over chunks
/// of the array, which provides a convenient way to process large arrays sequentially
/// in smaller, manageable pieces.
pub trait MemoryMappedChunkIter<A: Clone + Copy + 'static + Send + Sync> {
    /// Create an iterator over chunks of the array (for 1D arrays only).
    ///
    /// This method returns an iterator that yields chunks of the array as
    /// `Array1<A>` values, making it easy to process the array in smaller pieces.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The chunking strategy to determine chunk sizes
    ///
    /// # Returns
    ///
    /// An iterator that yields `Array1<A>` chunks of the memory-mapped array
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::memory_efficient::{create_mmap, AccessMode, ChunkingStrategy, MemoryMappedChunkIter};
    /// use ndarray::Array1;
    ///
    /// // Create a memory-mapped array
    /// let data = Array1::<f64>::linspace(0., 99., 100);
    /// let file_path = "example.bin";  // In practice, use a proper temporary path
    /// let mmap = create_mmap(&data, file_path.as_ref(), AccessMode::Write, 0).unwrap();
    ///
    /// // Create a chunk iterator with chunks of size 25
    /// let mut iter = mmap.chunks(ChunkingStrategy::Fixed(25));
    ///
    /// // Get the first chunk (elements 0-24)
    /// let chunk1 = iter.next().unwrap();
    /// assert_eq!(chunk1.len(), 25);
    /// assert_eq!(chunk1[0], 0.0);
    /// assert_eq!(chunk1[24], 24.0);
    /// ```
    fn chunks(&self, strategy: ChunkingStrategy) -> ChunkIter<A>;
}

impl<A: Clone + Copy + 'static + Send + Sync + Send + Sync> MemoryMappedChunks<A>
    for MemoryMappedArray<A>
{
    fn chunk_count(&self, strategy: ChunkingStrategy) -> usize {
        match strategy {
            ChunkingStrategy::Fixed(size) => {
                // Calculate how many chunks of the given size we need
                self.size.div_ceil(size)
            }
            ChunkingStrategy::NumChunks(n) => {
                // Number of chunks is explicitly specified
                n
            }
            ChunkingStrategy::Auto => {
                // Determine a reasonable chunk size based on the array size
                let total_elements = self.size;
                let optimal_chunk_size = (total_elements / 100).max(1);
                total_elements.div_ceil(optimal_chunk_size)
            }
            ChunkingStrategy::FixedBytes(bytes) => {
                // Calculate how many chunks based on bytes
                let element_size = std::mem::size_of::<A>();
                let elements_per_chunk = bytes / element_size;
                let elements_per_chunk = elements_per_chunk.max(1); // Ensure at least 1 element per chunk
                self.size.div_ceil(elements_per_chunk)
            }
        }
    }

    fn process_chunks<F, R>(&self, strategy: ChunkingStrategy, f: F) -> Vec<R>
    where
        F: Fn(&[A], usize) -> R,
    {
        let total_chunks = self.chunk_count(strategy);
        let mut results = Vec::with_capacity(total_chunks);

        // Process each chunk by copying the data
        for chunk_idx in 0..total_chunks {
            // Calculate chunk size and indices
            let chunk_size = match strategy {
                ChunkingStrategy::Fixed(size) => size,
                ChunkingStrategy::NumChunks(n) => self.size.div_ceil(n),
                ChunkingStrategy::Auto => {
                    let total_elements = self.size;
                    (total_elements / 100).max(1)
                }
                ChunkingStrategy::FixedBytes(bytes) => {
                    let element_size = std::mem::size_of::<A>();
                    let elements_per_chunk = bytes / element_size;
                    elements_per_chunk.max(1)
                }
            };

            let start_idx = chunk_idx * chunk_size;
            let end_idx = (start_idx + chunk_size).min(self.size);

            // Get the data for this chunk
            if let Ok(array_1d) = self.as_array::<ndarray::Ix1>() {
                // Copy the data to a new array to avoid lifetime issues
                let chunk_data = array_1d.slice(ndarray::s![start_idx..end_idx]).to_vec();

                // Process the chunk data
                results.push(f(&chunk_data, chunk_idx));
            }
        }

        results
    }

    fn process_chunks_mut<F>(&mut self, strategy: ChunkingStrategy, f: F)
    where
        F: Fn(&mut [A], usize),
    {
        let total_chunks = self.chunk_count(strategy);
        let element_size = std::mem::size_of::<A>();

        // Process each chunk
        for chunk_idx in 0..total_chunks {
            // Calculate chunk size and indices
            let chunk_size = match strategy {
                ChunkingStrategy::Fixed(size) => size,
                ChunkingStrategy::NumChunks(n) => self.size.div_ceil(n),
                ChunkingStrategy::Auto => {
                    let total_elements = self.size;
                    (total_elements / 100).max(1)
                }
                ChunkingStrategy::FixedBytes(bytes) => {
                    let elements_per_chunk = bytes / element_size;
                    elements_per_chunk.max(1)
                }
            };

            let start_idx = chunk_idx * chunk_size;
            let end_idx = (start_idx + chunk_size).min(self.size);

            // Get a copy of the data for this chunk
            let mut chunk_data = Vec::with_capacity(end_idx - start_idx);

            // Obtain the data safely through the memory mapping
            if let Ok(array_1d) = self.as_array::<ndarray::Ix1>() {
                chunk_data.extend_from_slice(
                    array_1d
                        .slice(ndarray::s![start_idx..end_idx])
                        .as_slice()
                        .unwrap(),
                );
            } else {
                continue;
            }

            // Process the chunk data with the provided function
            f(&mut chunk_data, chunk_idx);

            // Write the modified data back to the file directly
            // This is the most reliable way to ensure changes are persisted
            let file_path = &self.file_path;

            if let Ok(mut file) = OpenOptions::new().write(true).open(file_path) {
                // Calculate the effective offset (header + data offset + element position)
                let effective_offset = self.offset + start_idx * element_size;

                // Seek to the position and write the data
                if file.seek(SeekFrom::Start(effective_offset as u64)).is_ok() {
                    // Convert the chunk data to bytes
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            chunk_data.as_ptr() as *const u8,
                            chunk_data.len() * element_size,
                        )
                    };

                    // Write the bytes to the file
                    let _ = file.write_all(bytes);
                    let _ = file.flush();
                }
            }
        }

        // Reload the memory mapping to ensure changes are visible
        let _ = self.reload();
    }
}

// Add the parallel methods directly to the existing implementation
#[cfg(feature = "parallel")]
impl<A: Clone + Copy + 'static + Send + Sync + Send + Sync> MemoryMappedChunksParallel<A>
    for MemoryMappedArray<A>
{
    fn process_chunks_parallel<F, R>(&self, strategy: ChunkingStrategy, f: F) -> Vec<R>
    where
        F: Fn(&[A], usize) -> R + Send + Sync,
        R: Send,
    {
        // First, generate all the chunk indices and sizes
        let total_chunks = self.chunk_count(strategy);
        let chunks_info: Vec<_> = (0..total_chunks)
            .map(|chunk_idx| {
                let chunk_size = match strategy {
                    ChunkingStrategy::Fixed(size) => size,
                    ChunkingStrategy::NumChunks(n) => self.size.div_ceil(n),
                    ChunkingStrategy::Auto => {
                        let total_elements = self.size;
                        (total_elements / 100).max(1)
                    }
                    ChunkingStrategy::FixedBytes(bytes) => {
                        let element_size = std::mem::size_of::<A>();
                        let elements_per_chunk = bytes / element_size;
                        elements_per_chunk.max(1)
                    }
                };

                let start_idx = chunk_idx * chunk_size;
                let end_idx = (start_idx + chunk_size).min(self.size);

                (chunk_idx, start_idx, end_idx)
            })
            .collect();

        // Get the full array data
        let array_1d = match self.as_array::<ndarray::Ix1>() {
            Ok(arr) => arr,
            Err(_) => return Vec::new(),
        };

        // Process chunks in parallel
        let results: Vec<_> = chunks_info
            .into_par_iter()
            .map(|(chunk_idx, start_idx, end_idx)| {
                // Copy the data for this chunk
                let chunk_data = array_1d.slice(ndarray::s![start_idx..end_idx]).to_vec();

                // Process the chunk and return the result
                f(&chunk_data, chunk_idx)
            })
            .collect();

        results
    }

    fn process_chunks_mut_parallel<F>(&mut self, strategy: ChunkingStrategy, f: F)
    where
        F: Fn(&mut [A], usize) + Send + Sync,
    {
        let total_chunks = self.chunk_count(strategy);
        let element_size = std::mem::size_of::<A>();

        // First, generate all the chunk indices and sizes
        let chunks_info: Vec<_> = (0..total_chunks)
            .map(|chunk_idx| {
                let chunk_size = match strategy {
                    ChunkingStrategy::Fixed(size) => size,
                    ChunkingStrategy::NumChunks(n) => self.size.div_ceil(n),
                    ChunkingStrategy::Auto => {
                        let total_elements = self.size;
                        (total_elements / 100).max(1)
                    }
                    ChunkingStrategy::FixedBytes(bytes) => {
                        let elements_per_chunk = bytes / element_size;
                        elements_per_chunk.max(1)
                    }
                };

                let start_idx = chunk_idx * chunk_size;
                let end_idx = (start_idx + chunk_size).min(self.size);

                (chunk_idx, start_idx, end_idx)
            })
            .collect();

        // Get reference to the file path for the closures
        let file_path = self.file_path.clone();
        let offset = self.offset;

        // Get the full array data
        let array_1d = match self.as_array::<ndarray::Ix1>() {
            Ok(arr) => arr,
            Err(_) => return,
        };

        // Process chunks in parallel and collect the modified data
        let modifications: Vec<_> = chunks_info
            .into_par_iter()
            .map(|(chunk_idx, start_idx, end_idx)| {
                // Copy the data for this chunk
                let mut chunk_data = array_1d.slice(ndarray::s![start_idx..end_idx]).to_vec();

                // Process the chunk data with the provided function
                f(&mut chunk_data, chunk_idx);

                // Return the chunk index, start index, and modified data
                (chunk_idx, start_idx, chunk_data)
            })
            .collect();

        // Apply all modifications to the file sequentially to avoid conflicts
        for (_, start_idx, chunk_data) in modifications {
            if let Ok(mut file) = OpenOptions::new().write(true).open(&file_path) {
                // Calculate the effective offset
                let effective_offset = offset + start_idx * element_size;

                // Seek to the position and write the data
                if file.seek(SeekFrom::Start(effective_offset as u64)).is_ok() {
                    // Convert the chunk data to bytes
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            chunk_data.as_ptr() as *const u8,
                            chunk_data.len() * element_size,
                        )
                    };

                    // Write the bytes to the file
                    let _ = file.write_all(bytes);
                    let _ = file.flush();
                }
            }
        }

        // Reload the memory mapping to ensure changes are visible
        let _ = self.reload();
    }
}

impl<A: Clone + Copy + 'static + Send + Sync> MemoryMappedChunkIter<A> for MemoryMappedArray<A> {
    fn chunks(&self, strategy: ChunkingStrategy) -> ChunkIter<A> {
        ChunkIter {
            array: self,
            current_idx: 0,
            total_chunks: self.chunk_count(strategy),
            strategy,
        }
    }
}
