#[cfg(feature = "memory_efficient")]
mod tests {
    use ndarray::Array1;
    use scirs2_core::memory_efficient::{
        create_mmap, AccessMode, ChunkingStrategy, MemoryMappedArray, MemoryMappedChunkIter,
        MemoryMappedChunks,
    };
    use tempfile::tempdir;

    #[test]
    fn test_chunk_count() {
        // Create test data
        let data = Array1::<f64>::linspace(0., 99., 100);

        // Create temporary memory-mapped array
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_chunk_count.bin");
        let mmap = create_mmap(&data, &file_path, AccessMode::Write, 0).unwrap();

        // Test different chunking strategies
        assert_eq!(mmap.chunk_count(ChunkingStrategy::Fixed(10)), 10);
        assert_eq!(mmap.chunk_count(ChunkingStrategy::Fixed(25)), 4);
        assert_eq!(mmap.chunk_count(ChunkingStrategy::NumChunks(5)), 5);

        // Auto should give a reasonable number of chunks
        let auto_chunks = mmap.chunk_count(ChunkingStrategy::Auto);
        assert!(auto_chunks > 0);
        assert!(auto_chunks <= 100);
    }

    #[test]
    fn test_process_chunks() {
        // Create test data
        let data = Array1::<i32>::from_vec((0..100).collect());

        // Create temporary memory-mapped array
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_process_chunks.bin");
        let mmap = create_mmap(&data, &file_path, AccessMode::Write, 0).unwrap();

        // Use process_chunks to sum each chunk
        let chunk_sums = mmap.process_chunks(ChunkingStrategy::Fixed(25), |chunk, _| {
            chunk.iter().map(|&x| x as i64).sum::<i64>()
        });

        assert_eq!(chunk_sums.len(), 4);

        // Expected sums for chunks of 25 elements starting at 0, 25, 50, 75
        let expected_sums = vec![
            (0..25).sum::<i32>() as i64,
            (25..50).sum::<i32>() as i64,
            (50..75).sum::<i32>() as i64,
            (75..100).sum::<i32>() as i64,
        ];

        assert_eq!(chunk_sums, expected_sums);
    }

    #[test]
    fn test_simple_mutation() {
        // Create test data - simple array of zeros
        let data = Array1::<i32>::from_vec(vec![0; 10]);
        println!("Original data: {:?}", data);

        // Create memory-mapped array
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_simple_mutation.bin");
        let mut mmap = create_mmap(&data, &file_path, AccessMode::Write, 0).unwrap();

        // Process chunks to make a simple mutation
        // This is more reliable than direct mutation through array view
        mmap.process_chunks_mut(ChunkingStrategy::Fixed(10), |chunk_data, _| {
            chunk_data[0] = 42;
        });

        // Verify changes persisted
        let array = mmap.as_array::<ndarray::Ix1>().unwrap();
        println!("After mutation through process_chunks_mut: {:?}", array);

        // Test should pass if mutation worked
        assert_eq!(array[0], 42);
    }

    #[test]
    fn test_process_chunks_mut() {
        // Create test data - simple array of zeros
        let data = Array1::<i32>::from_vec(vec![0; 100]);

        // Create memory-mapped array
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_chunks_mutation.bin");

        // Use save_array to create the file with proper header
        MemoryMappedArray::<i32>::save_array(&data, &file_path, None).unwrap();

        // Open for read-write access
        let mut mmap =
            MemoryMappedArray::<i32>::open_zero_copy(&file_path, AccessMode::ReadWrite).unwrap();

        // Get the original data
        let original = mmap.as_array::<ndarray::Ix1>().unwrap();
        println!(
            "Original data (first 5 elements): {:?}",
            original.slice(ndarray::s![0..5])
        );

        // Process chunks and modify each chunk
        mmap.process_chunks_mut(ChunkingStrategy::Fixed(10), |chunk_data, chunk_idx| {
            // Modify each element in the chunk
            for (i, item) in chunk_data.iter_mut().enumerate() {
                *item = (chunk_idx * 10 + i) as i32;
            }
        });

        // Flush changes to disk
        mmap.flush().unwrap();

        // Verify changes persisted by reading directly from the file
        // Drop the mmap to ensure file is closed
        drop(mmap);

        // Reopen the file
        let reopened_mmap =
            MemoryMappedArray::<i32>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwrap();
        let modified = reopened_mmap.as_array::<ndarray::Ix1>().unwrap();

        println!(
            "Modified data (first 15 elements): {:?}",
            modified.slice(ndarray::s![0..15])
        );

        // Check the values in each chunk
        for chunk_idx in 0..10 {
            for i in 0..10 {
                let expected = (chunk_idx * 10 + i) as i32;
                let actual = modified[chunk_idx * 10 + i];
                assert_eq!(
                    actual,
                    expected,
                    "Element at position {} should be {}",
                    chunk_idx * 10 + i,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_chunks_iterator() {
        // Create test data
        let data = Array1::<f64>::linspace(0., 99., 100);

        // Create temporary memory-mapped array
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_chunks_iterator.bin");
        let mmap = create_mmap(&data, &file_path, AccessMode::Write, 0).unwrap();

        // Use chunks iterator
        let mut count = 0;
        let mut sum = 0.0;

        for chunk in mmap.chunks(ChunkingStrategy::Fixed(10)) {
            sum += chunk.sum();
            count += 1;
        }

        assert_eq!(count, 10);
        assert!((sum - data.sum()).abs() < 1e-10);
    }
}
