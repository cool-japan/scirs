#[cfg(feature = "memory_efficient")]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array1, Array2};
    use scirs2_core::memory_efficient::{
        chunk_wise_binary_op, chunk_wise_op, chunk_wise_reduce, ChunkedArray, ChunkingStrategy,
        OPTIMAL_CHUNK_SIZE,
    };

    #[test]
    fn test_chunked_array_creation() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Test with auto strategy
        let _chunked_auto = ChunkedArray::new(data.clone(), ChunkingStrategy::Auto);

        // Test with fixed size strategy
        let chunk_size = 3;
        let chunked_fixed = ChunkedArray::new(data.clone(), ChunkingStrategy::Fixed(chunk_size));
        assert_eq!(chunked_fixed.chunk_size(), chunk_size);

        // Number of chunks should be ceil(total_size / chunk_size)
        let expected_chunks = data.len().div_ceil(chunk_size);
        assert_eq!(chunked_fixed.num_chunks(), expected_chunks);

        // Test with fixed bytes strategy
        let elem_size = std::mem::size_of::<f64>();
        let chunked_bytes = ChunkedArray::new(
            data.clone(),
            ChunkingStrategy::FixedBytes(chunk_size * elem_size),
        );
        assert_eq!(chunked_bytes.chunk_size(), chunk_size);

        // Test with num chunks strategy
        let num_chunks = 2;
        let chunked_num = ChunkedArray::new(data.clone(), ChunkingStrategy::NumChunks(num_chunks));
        assert_eq!(chunked_num.num_chunks(), num_chunks);
    }

    #[test]
    fn test_chunk_wise_op() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Test squaring all elements
        let result = chunk_wise_op(
            &data,
            |chunk| chunk.map(|&x| x * x),
            ChunkingStrategy::Fixed(3),
        )
        .unwrap();

        // Check that all elements are correctly squared
        let expected = Array1::from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_chunk_wise_binary_op() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

        // Test adding two arrays
        let result = chunk_wise_binary_op(
            &a,
            &b,
            |chunk_a, chunk_b| chunk_a + chunk_b,
            ChunkingStrategy::Fixed(2),
        )
        .unwrap();

        // Check that all elements are correctly added
        let expected = Array1::from_vec(vec![6.0, 8.0, 10.0, 12.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_chunk_wise_reduce() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Test summing all elements
        let result = chunk_wise_reduce(
            &data,
            |chunk| chunk.sum(),
            |partial_sums| partial_sums.into_iter().sum(),
            ChunkingStrategy::Fixed(3),
        )
        .unwrap();

        // Check that all elements are correctly summed
        let expected: f64 = 36.0; // 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_optimal_chunk_size() {
        // Just verify that OPTIMAL_CHUNK_SIZE is a reasonable value
        // OPTIMAL_CHUNK_SIZE is guaranteed to be > 0 by its definition

        // Check that a 2D array with dimensions that would fit in the optimal chunk size
        // is correctly processed
        let n = ((OPTIMAL_CHUNK_SIZE / std::mem::size_of::<f64>()) as f64).sqrt() as usize;
        let data = Array2::from_elem((n, n), 1.0);

        let result = chunk_wise_op(
            &data,
            |chunk| chunk.map(|&x| x * 2.0),
            ChunkingStrategy::Auto,
        )
        .unwrap();

        assert_eq!(result.shape(), data.shape());
        assert_relative_eq!(result[[0, 0]], 2.0, epsilon = 1e-10);
    }
}
