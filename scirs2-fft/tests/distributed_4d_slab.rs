/// Tests for 4D slab decomposition in the distributed FFT module.
///
/// The `distributed` module is gated by `feature = "never"` (legacy gate).
/// Run these tests with: `cargo nextest run --features never -p scirs2-fft`
#[cfg(feature = "never")]
mod tests {
    use scirs2_core::ndarray::{ArrayD, IxDyn};
    use scirs2_core::numeric::Complex64;
    use scirs2_fft::distributed::{
        CommunicationPattern, DecompositionStrategy, DistributedConfig, DistributedFFT,
        MockCommunicator,
    };
    use std::sync::Arc;

    fn make_dfft(strategy: DecompositionStrategy, process_grid: Vec<usize>) -> DistributedFFT {
        let config = DistributedConfig {
            node_count: 1,
            rank: 0,
            decomposition: strategy,
            communication: CommunicationPattern::AllToAll,
            process_grid,
            local_size: vec![],
            max_local_size: 64,
        };
        let comm = Arc::new(MockCommunicator::new(1, 0));
        DistributedFFT::new(config, comm)
    }

    /// Build a 4D complex array of shape [d0,d1,d2,d3] with values index * 0.1 + 0.0i.
    fn make_4d_input(shape: [usize; 4]) -> ArrayD<Complex64> {
        let total: usize = shape.iter().product();
        let data: Vec<Complex64> = (0..total)
            .map(|i| Complex64::new(i as f64 * 0.1, 0.0))
            .collect();
        ArrayD::from_shape_vec(IxDyn(&shape), data).expect("shape/data mismatch")
    }

    #[test]
    fn test_slab_4d_no_error() {
        let dfft = make_dfft(DecompositionStrategy::Slab, vec![1]);
        let input = make_4d_input([4, 4, 4, 4]);
        // Must not return Err — previously returned DimensionError for ndim > 3
        let result = dfft.decompose_data(&input);
        assert!(
            result.is_ok(),
            "slab decomposition on 4D input should succeed, got: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_slab_4d_output_shape() {
        let dfft = make_dfft(DecompositionStrategy::Slab, vec![1]);
        let input = make_4d_input([4, 4, 4, 4]);
        let local = dfft.decompose_data(&input).expect("decompose_data failed");
        // With node_count=1, rank=0: the whole array is slab[0]
        assert_eq!(
            local.shape(),
            &[4, 4, 4, 4],
            "slab output shape should equal input shape for single node"
        );
    }

    #[test]
    fn test_slab_4d_values_correct() {
        let dfft = make_dfft(DecompositionStrategy::Slab, vec![1]);
        let input = make_4d_input([4, 4, 4, 4]);
        let local = dfft.decompose_data(&input).expect("decompose_data failed");
        // Every element should match the input value exactly
        let total: usize = 4 * 4 * 4 * 4;
        for i in 0..total {
            let expected = Complex64::new(i as f64 * 0.1, 0.0);
            let got = local.iter().nth(i).copied().expect("element missing");
            assert!(
                (got.re - expected.re).abs() < 1e-10 && (got.im - expected.im).abs() < 1e-10,
                "element {i}: expected {expected:?}, got {got:?}"
            );
        }
    }

    #[test]
    fn test_slab_4d_two_node_first_rank() {
        // Simulate rank 0 of 2 nodes: receives the first half of slab dimension
        let config = DistributedConfig {
            node_count: 2,
            rank: 0,
            decomposition: DecompositionStrategy::Slab,
            communication: CommunicationPattern::AllToAll,
            process_grid: vec![2],
            local_size: vec![],
            max_local_size: 64,
        };
        let comm = Arc::new(MockCommunicator::new(2, 0));
        let dfft = DistributedFFT::new(config, comm);
        let input = make_4d_input([4, 4, 4, 4]);
        let local = dfft.decompose_data(&input).expect("decompose_data failed");
        // Slab axis 0: 4 slabs / 2 nodes = 2 slabs per node; rank 0 gets slabs 0..2
        assert_eq!(local.shape()[0], 2, "rank 0 of 2 should get first 2 slabs");
        assert_eq!(local.shape()[1..], [4, 4, 4]);
    }

    #[test]
    fn test_slab_4d_two_node_second_rank() {
        // Simulate rank 1 of 2 nodes: receives the second half
        let config = DistributedConfig {
            node_count: 2,
            rank: 1,
            decomposition: DecompositionStrategy::Slab,
            communication: CommunicationPattern::AllToAll,
            process_grid: vec![2],
            local_size: vec![],
            max_local_size: 64,
        };
        let comm = Arc::new(MockCommunicator::new(2, 1));
        let dfft = DistributedFFT::new(config, comm);
        let input = make_4d_input([4, 4, 4, 4]);
        let local = dfft.decompose_data(&input).expect("decompose_data failed");
        // rank 1 gets slabs 2..4
        assert_eq!(local.shape()[0], 2, "rank 1 of 2 should get last 2 slabs");
        assert_eq!(local.shape()[1..], [4, 4, 4]);
        // First element of rank-1 slab should equal input[2,0,0,0]
        let first_val = local[[0, 0, 0, 0]];
        // input[2,0,0,0] = flat index 2*4*4*4 = 128; value = 128 * 0.1
        let expected = Complex64::new(128.0 * 0.1, 0.0);
        assert!(
            (first_val.re - expected.re).abs() < 1e-10,
            "first element of rank-1 slab: expected {expected:?}, got {first_val:?}"
        );
    }

    #[test]
    fn test_slab_5d_no_error() {
        // Verify 5D also works (exercises the same n-D branch)
        let dfft = make_dfft(DecompositionStrategy::Slab, vec![1]);
        let total = 3usize.pow(5);
        let data: Vec<Complex64> = (0..total).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let input =
            ArrayD::from_shape_vec(IxDyn(&[3, 3, 3, 3, 3]), data).expect("shape/data mismatch");
        let result = dfft.decompose_data(&input);
        assert!(result.is_ok(), "5D slab decomposition must not error");
        let local = result.expect("decompose_data failed");
        assert_eq!(local.shape(), &[3, 3, 3, 3, 3]);
    }
}
