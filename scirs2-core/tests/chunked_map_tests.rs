//! Tests for ChunkedArray map and par_map methods

#[cfg(feature = "memory_efficient")]
mod tests {
    use ndarray::Array1;
    use scirs2_core::memory_efficient::{ChunkedArray, ChunkingStrategy};

    #[test]
    fn test_chunked_array_map_1d() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let chunked = ChunkedArray::new(data.clone(), ChunkingStrategy::Fixed(3));

        // Map function that computes the sum of each chunk
        let results = chunked.map(|chunk| chunk.sum());

        // Expected results:
        // Chunk 1: [1.0, 2.0, 3.0] -> sum = 6.0
        // Chunk 2: [4.0, 5.0, 6.0] -> sum = 15.0
        // Chunk 3: [7.0, 8.0] -> sum = 15.0
        let expected = Array1::from_vec(vec![6.0, 15.0, 15.0]);
        assert_eq!(results, expected);
    }

    #[test]
    fn test_chunked_array_map_with_different_strategies() {
        let data = Array1::from_vec(vec![1.0; 10]);

        // Test with NumChunks strategy
        let chunked = ChunkedArray::new(data.clone(), ChunkingStrategy::NumChunks(5));
        let results = chunked.map(|chunk| chunk.len());

        // Each chunk should have 2 elements
        let expected = Array1::from_vec(vec![2; 5]);
        assert_eq!(results, expected);

        // Test with Fixed strategy
        let chunked = ChunkedArray::new(data.clone(), ChunkingStrategy::Fixed(4));
        let results = chunked.map(|chunk| chunk.len());

        // First two chunks have 4 elements, last chunk has 2
        let expected = Array1::from_vec(vec![4, 4, 2]);
        assert_eq!(results, expected);
    }

    #[test]
    fn test_chunked_array_map_mean() {
        let data = Array1::linspace(0.0, 100.0, 100);
        let chunked = ChunkedArray::new(data, ChunkingStrategy::Fixed(25));

        // Compute mean of each chunk
        let results = chunked.map(|chunk| chunk.mean().unwrap());

        // We should have 4 chunks
        assert_eq!(results.len(), 4);

        // The means should be approximately: 12.0, 37.0, 62.0, 87.0
        // (centers of each quarter of the range)
        assert!((results[0] - 12.12f64).abs() < 1.0);
        assert!((results[1] - 37.37f64).abs() < 1.0);
        assert!((results[2] - 62.63f64).abs() < 1.0);
        assert!((results[3] - 87.88f64).abs() < 1.0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_chunked_array_par_map() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let chunked = ChunkedArray::new(data.clone(), ChunkingStrategy::Fixed(3));

        // Parallel map function that computes the sum of each chunk
        let results = chunked.par_map(|chunk| chunk.sum());

        // Expected results (same as sequential version):
        let expected = Array1::from_vec(vec![6.0, 15.0, 15.0]);
        assert_eq!(results, expected);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_chunked_array_par_map_large() {
        // Create a large array to benefit from parallelization
        let size = 10_000;
        let data = Array1::linspace(0.0, 1000.0, size);
        let chunked = ChunkedArray::new(data.clone(), ChunkingStrategy::Fixed(1000));

        // Compute a somewhat expensive operation on each chunk
        let results =
            chunked.par_map(|chunk| chunk.iter().map(|&x: &f64| x.sin() * x.cos()).sum::<f64>());

        // We should have 10 chunks
        assert_eq!(results.len(), 10);

        // Compare with sequential version to ensure correctness
        let sequential_results =
            chunked.map(|chunk| chunk.iter().map(|&x: &f64| x.sin() * x.cos()).sum::<f64>());

        // Results should be the same (within floating point tolerance)
        for (par, seq) in results.iter().zip(sequential_results.iter()) {
            assert!((par - seq).abs() < 1e-10);
        }
    }

    #[test]
    fn test_chunked_array_map_different_output_type() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let chunked = ChunkedArray::new(data, ChunkingStrategy::Fixed(2));

        // Map to a different type (bool indicating if sum > 5)
        let results = chunked.map(|chunk| chunk.sum() > 5.0);

        // Expected: [false, true, true] (sums are 3, 7, 11)
        let expected = Array1::from_vec(vec![false, true, true]);
        assert_eq!(results, expected);
    }

    #[test]
    fn test_chunked_array_map_single_chunk() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let chunked = ChunkedArray::new(data.clone(), ChunkingStrategy::Fixed(10));

        // With chunk size larger than data, we should have just one chunk
        assert_eq!(chunked.num_chunks(), 1);

        let results = chunked.map(|chunk| chunk.sum());
        let expected = Array1::from_vec(vec![10.0]);
        assert_eq!(results, expected);
    }

    #[test]
    fn test_chunked_array_map_empty_edge_case() {
        let data = Array1::from_vec(vec![1.0]);
        let chunked = ChunkedArray::new(data, ChunkingStrategy::Fixed(1));

        let results = chunked.map(|chunk| chunk.sum());
        let expected = Array1::from_vec(vec![1.0]);
        assert_eq!(results, expected);
    }
}
