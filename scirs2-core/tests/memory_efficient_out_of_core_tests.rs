#[cfg(all(feature = "memory_efficient", test))]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::Array2;
    use scirs2_core::memory_efficient::{
        create_disk_array, ChunkingStrategy, DiskBackedArray, OutOfCoreArray,
    };
    use tempfile::NamedTempFile;

    #[test]
    fn test_out_of_core_array_creation() {
        // Create test data
        let data = Array2::from_shape_fn((10, 10), |(i, j)| i as f64 + j as f64);

        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path();

        // Create an out-of-core array
        let result = OutOfCoreArray::new(&data, file_path, ChunkingStrategy::Fixed(5));

        assert!(result.is_ok());

        let array = result.unwrap();

        // Check properties
        assert_eq!(array.shape, data.shape());
        assert_eq!(array.size, data.len());
        assert_eq!(array.file_path, file_path);
        assert!(!array.is_temp()); // Should not be marked as temporary
    }

    #[test]
    fn test_out_of_core_array_temp() {
        // Create test data
        let data = Array2::from_shape_fn((10, 10), |(i, j)| i as f64 + j as f64);

        // Create a temporary out-of-core array
        let result = OutOfCoreArray::new_temp(&data, ChunkingStrategy::Fixed(5));

        assert!(result.is_ok());

        let array = result.unwrap();

        // Check properties
        assert_eq!(array.shape, data.shape());
        assert_eq!(array.size, data.len());
        assert!(array.is_temp()); // Should be marked as temporary

        // Verify the file exists
        assert!(array.file_path.exists());

        // The file should be deleted when the array is dropped
        let file_path = array.file_path.clone();
        drop(array);
        assert!(!file_path.exists());
    }

    #[test]
    fn test_out_of_core_array_load() {
        // Create test data with a specific pattern
        let data = Array2::from_shape_fn((5, 5), |(i, j)| (i * 10 + j) as f64);

        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path();

        // Create an out-of-core array
        let array = OutOfCoreArray::new(&data, file_path, ChunkingStrategy::Fixed(2)).unwrap();

        // Load the data back
        let loaded = array.load().unwrap();

        // Check that the loaded data matches the original
        assert_eq!(loaded.shape(), data.shape());

        for i in 0..data.shape()[0] {
            for j in 0..data.shape()[1] {
                assert_relative_eq!(loaded[[i, j]], data[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_disk_backed_array() {
        // Create test data
        let data = Array2::from_shape_fn((10, 10), |(i, j)| i as f64 + j as f64);

        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path();

        // Create a disk-backed array
        let result = create_disk_array(
            &data,
            file_path,
            ChunkingStrategy::Fixed(5),
            true, // read-only
        );

        assert!(result.is_ok());

        let array = result.unwrap();

        // Check properties
        assert_eq!(array.array.shape, data.shape());
        assert!(array.read_only);

        // Load the data back
        let loaded = array.load().unwrap();

        // Check that the loaded data matches the original
        assert_eq!(loaded.shape(), data.shape());

        for i in 0..data.shape()[0] {
            for j in 0..data.shape()[1] {
                assert_relative_eq!(loaded[[i, j]], data[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_disk_backed_array_temp() {
        // Create test data
        let data = Array2::from_shape_fn((10, 10), |(i, j)| i as f64 + j as f64);

        // Create a temporary disk-backed array
        let result = DiskBackedArray::new_temp(
            &data,
            ChunkingStrategy::Fixed(5),
            false, // not read-only
        );

        assert!(result.is_ok());

        let array = result.unwrap();

        // Check properties
        assert_eq!(array.array.shape, data.shape());
        assert!(!array.read_only);
        assert!(array.is_temp());

        // The file should be deleted when the array is dropped
        let file_path = array.array.file_path.clone();
        assert!(file_path.exists());
        drop(array);
        assert!(!file_path.exists());
    }

    #[test]
    fn test_out_of_core_array_num_chunks() {
        // Create test data
        let data = Array2::from_shape_fn((100, 10), |(i, j)| i as f64 + j as f64);

        // Create a temporary out-of-core array with different chunking strategies

        // Fixed size
        let chunk_size = 20;
        let array1 = OutOfCoreArray::new_temp(&data, ChunkingStrategy::Fixed(chunk_size)).unwrap();

        // Expected number of chunks is ceil(total_size / chunk_size)
        let expected_chunks = data.len().div_ceil(chunk_size);
        assert_eq!(array1.num_chunks(), expected_chunks);

        // Fixed number of chunks
        let num_chunks = 5;
        let array2 =
            OutOfCoreArray::new_temp(&data, ChunkingStrategy::NumChunks(num_chunks)).unwrap();

        assert_eq!(array2.num_chunks(), num_chunks);

        // Auto (based on OPTIMAL_CHUNK_SIZE)
        let array3 = OutOfCoreArray::new_temp(&data, ChunkingStrategy::Auto).unwrap();

        // Check that the number of chunks is reasonable
        assert!(array3.num_chunks() > 0);
        assert!(array3.num_chunks() <= data.len());
    }

    // Note: The load_chunks function is not fully implemented in our placeholder
    // implementation, so we don't test it extensively.
    #[test]
    #[should_panic(expected = "OutOfCoreArray::map is not yet implemented")]
    fn test_out_of_core_array_map_unimplemented() {
        // Create test data
        let data = Array2::from_shape_fn((10, 10), |(i, j)| i as f64 + j as f64);

        // Create a temporary out-of-core array
        let array = OutOfCoreArray::new_temp(&data, ChunkingStrategy::Fixed(5)).unwrap();

        // This should panic with "not yet implemented"
        let _: Vec<f64> = array.map(|_| 0.0).unwrap();
    }
}
