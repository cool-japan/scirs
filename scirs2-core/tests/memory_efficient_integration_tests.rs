#[cfg(feature = "memory_efficient")]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::Array2;
    use scirs2_core::array::{mask_array, masked_invalid};
    use scirs2_core::memory_efficient::{
        chunk_wise_op, create_disk_array, diagonal_view, evaluate, transpose_view,
        ChunkingStrategy, LazyArray,
    };
    use tempfile::NamedTempFile;

    #[test]
    fn test_memory_efficient_integration() {
        // Create test data
        let data =
            Array2::from_shape_fn((10, 10), |(i, j)| if i == j { (i + 1) as f64 } else { 0.0 });

        // 1. Use chunk-wise operation to double all values
        let doubled = chunk_wise_op(
            &data,
            |chunk| chunk.map(|&x| x * 2.0),
            ChunkingStrategy::Fixed(5),
        )
        .unwrap();

        // 2. Create a lazy array from the result
        let lazy = LazyArray::new(doubled.clone());

        // 3. Define a lazy operation (identity in this case)
        let lazy_result = lazy.map(|&x| x);

        // 4. Evaluate the lazy operation
        let evaluated = evaluate(&lazy_result).unwrap();

        // 5. Create a transpose view
        let transposed = transpose_view(&evaluated).unwrap();

        // 6. Get the diagonal view (should contain the doubled diagonal values)
        let diagonal = diagonal_view(&evaluated).unwrap();

        // 7. Create a masked array to mask out zeros
        let mask = Array2::from_shape_fn(evaluated.raw_dim(), |(i, j)| evaluated[[i, j]] == 0.0);
        let masked = mask_array(evaluated.clone(), Some(mask), Some(0.0)).unwrap();

        // 8. Store the result in a disk-backed array
        let temp_file = NamedTempFile::new().unwrap();
        let disk_array = create_disk_array(
            &masked.data,
            temp_file.path(),
            ChunkingStrategy::Fixed(5),
            false,
        )
        .unwrap();

        // 9. Load back from disk
        let loaded = disk_array.load().unwrap();

        // Verify the results

        // The doubled diagonal should have values 2, 4, 6, ..., 20
        for i in 0..10 {
            assert_relative_eq!(diagonal[i], 2.0 * (i + 1) as f64, epsilon = 1e-10);
        }

        // The transposed array should be the same (since the matrix is symmetric)
        for i in 0..10 {
            for j in 0..10 {
                assert_relative_eq!(transposed[[i, j]], evaluated[[j, i]], epsilon = 1e-10);
            }
        }

        // The masked array should have zeros everywhere except the diagonal
        for i in 0..10 {
            for j in 0..10 {
                if i == j {
                    assert_relative_eq!(masked.data[[i, j]], 2.0 * (i + 1) as f64, epsilon = 1e-10);
                    assert!(!masked.mask[[i, j]]);
                } else {
                    assert!(masked.mask[[i, j]]);
                }
            }
        }

        // The loaded array should match the original data
        for i in 0..10 {
            for j in 0..10 {
                if i == j {
                    assert_relative_eq!(loaded[[i, j]], 2.0 * (i + 1) as f64, epsilon = 1e-10);
                } else {
                    assert_relative_eq!(loaded[[i, j]], 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_chunked_lazy_disk_workflow() {
        // Create a larger test matrix
        let n = 20;
        let data =
            Array2::from_shape_fn((n, n), |(i, j)| if i == j { (i + 1) as f64 } else { 0.0 });

        // Workflow test:
        // 1. Process in chunks
        // 2. Store as lazy computation
        // 3. Save to disk
        // 4. Load back and verify

        // Process in chunks - add 10 to all elements
        let added = chunk_wise_op(
            &data,
            |chunk| chunk.map(|&x| x + 10.0),
            ChunkingStrategy::Fixed(5),
        )
        .unwrap();

        // Create a lazy array for another operation - multiply by 2
        let lazy = LazyArray::new(added);
        let lazy_doubled = lazy.map(|&x| x * 2.0);

        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();

        // Store the result in a disk-backed array
        // In a real implementation, we would be able to evaluate directly to disk,
        // but in our placeholder we need to evaluate first
        let evaluated = evaluate(&lazy_doubled).unwrap();

        let disk_array = create_disk_array(
            &evaluated,
            temp_file.path(),
            ChunkingStrategy::Fixed(5),
            false,
        )
        .unwrap();

        // Load back from disk
        let loaded = disk_array.load().unwrap();

        // Verify the results
        // The diagonal should have values (1+10)*2, (2+10)*2, ..., (n+10)*2
        // Other elements should be 10*2 = 20
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    assert_relative_eq!(
                        loaded[[i, j]],
                        2.0 * ((i + 1) + 10) as f64,
                        epsilon = 1e-10
                    );
                } else {
                    assert_relative_eq!(loaded[[i, j]], 20.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_masked_arrays_with_chunking() {
        // Create test data with NaN values
        let mut data = Array2::from_shape_fn((10, 10), |(i, j)| (i * 10 + j) as f64);

        // Add some NaN values
        data[[0, 0]] = f64::NAN;
        data[[3, 4]] = f64::NAN;
        data[[7, 9]] = f64::NAN;

        // 1. Create a masked array that masks out NaN values
        let masked = masked_invalid(data.clone());

        // 2. Process the masked array in chunks
        let result = chunk_wise_op(
            &masked.data,
            |chunk| chunk.map(|&x| x * 2.0),
            ChunkingStrategy::Fixed(3),
        )
        .unwrap();

        // 3. Create a new masked array with the result but keep the original mask
        let doubled_masked = mask_array(result, Some(masked.mask.clone()), Some(0.0)).unwrap();

        // Verify the results
        for i in 0..10 {
            for j in 0..10 {
                if (i == 0 && j == 0) || (i == 3 && j == 4) || (i == 7 && j == 9) {
                    assert!(doubled_masked.mask[[i, j]]);
                } else {
                    assert!(!doubled_masked.mask[[i, j]]);
                    assert_relative_eq!(
                        doubled_masked.data[[i, j]],
                        2.0 * (i * 10 + j) as f64,
                        epsilon = 1e-10
                    );
                }
            }
        }
    }
}
