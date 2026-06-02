/// Tests for 5D pencil decomposition in the distributed FFT module.
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

    fn make_pencil_dfft(
        node_count: usize,
        rank: usize,
        process_grid: Vec<usize>,
    ) -> DistributedFFT {
        let config = DistributedConfig {
            node_count,
            rank,
            decomposition: DecompositionStrategy::Pencil,
            communication: CommunicationPattern::AllToAll,
            process_grid,
            local_size: vec![],
            max_local_size: 64,
        };
        let comm = Arc::new(MockCommunicator::new(node_count, rank));
        DistributedFFT::new(config, comm)
    }

    /// Build an n-D complex array with sequential float values.
    fn make_nd_input(shape: &[usize]) -> ArrayD<Complex64> {
        let total: usize = shape.iter().product();
        let data: Vec<Complex64> = (0..total)
            .map(|i| Complex64::new(i as f64 * 0.1, 0.0))
            .collect();
        ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape/data mismatch")
    }

    #[test]
    fn test_pencil_5d_single_node_no_error() {
        // node_count=1 with 1x1 process grid; 5D input must not error
        let dfft = make_pencil_dfft(1, 0, vec![1, 1]);
        let input = make_nd_input(&[4, 4, 4, 4, 4]);
        let result = dfft.decompose_data(&input);
        assert!(
            result.is_ok(),
            "pencil decomposition on 5D input should succeed, got: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_pencil_5d_single_node_output_shape() {
        let dfft = make_pencil_dfft(1, 0, vec![1, 1]);
        let input = make_nd_input(&[4, 4, 4, 4, 4]);
        let local = dfft.decompose_data(&input).expect("decompose_data failed");
        // Single node: all data assigned to rank 0
        assert_eq!(
            local.shape(),
            &[4, 4, 4, 4, 4],
            "pencil single-node output shape must match input"
        );
    }

    #[test]
    fn test_pencil_5d_values_correct() {
        let dfft = make_pencil_dfft(1, 0, vec![1, 1]);
        let input = make_nd_input(&[4, 4, 4, 4, 4]);
        let local = dfft.decompose_data(&input).expect("decompose_data failed");
        let total: usize = 4usize.pow(5);
        for i in 0..total {
            let expected = Complex64::new(i as f64 * 0.1, 0.0);
            let got = local.iter().nth(i).copied().expect("element missing");
            assert!(
                (got.re - expected.re).abs() < 1e-10 && got.im.abs() < 1e-10,
                "element {i}: expected {expected:?}, got {got:?}"
            );
        }
    }

    #[test]
    fn test_pencil_5d_four_node_rank0_shape() {
        // 4-node 2x2 grid; rank 0 owns rows 0..2 of axis-0, cols 0..2 of axis-1
        let dfft = make_pencil_dfft(4, 0, vec![2, 2]);
        let input = make_nd_input(&[4, 4, 4, 4, 4]);
        let local = dfft.decompose_data(&input).expect("decompose_data failed");
        assert_eq!(local.shape()[0], 2, "rank 0 gets rows 0..2 on axis 0");
        assert_eq!(local.shape()[1], 2, "rank 0 gets cols 0..2 on axis 1");
        assert_eq!(&local.shape()[2..], &[4, 4, 4], "remaining axes unchanged");
    }

    #[test]
    fn test_pencil_5d_four_node_rank3_shape() {
        // Rank 3 in 2x2 grid: my_row=1, my_col=1 → rows 2..4, cols 2..4
        let dfft = make_pencil_dfft(4, 3, vec![2, 2]);
        let input = make_nd_input(&[4, 4, 4, 4, 4]);
        let local = dfft.decompose_data(&input).expect("decompose_data failed");
        assert_eq!(local.shape()[0], 2, "rank 3 gets rows 2..4 on axis 0");
        assert_eq!(local.shape()[1], 2, "rank 3 gets cols 2..4 on axis 1");
        assert_eq!(&local.shape()[2..], &[4, 4, 4]);
    }

    #[test]
    fn test_pencil_5d_four_node_rank3_values() {
        // Rank 3: my_start_row=2, my_start_col=2
        // local[0,0,0,0,0] corresponds to input[2,2,0,0,0]
        let dfft = make_pencil_dfft(4, 3, vec![2, 2]);
        let input = make_nd_input(&[4, 4, 4, 4, 4]);
        let local = dfft.decompose_data(&input).expect("decompose_data failed");
        // Flat index of input[2,2,0,0,0] = (2*4 + 2)*4*4*4 = 10 * 64 = 640
        let expected = Complex64::new(640.0 * 0.1, 0.0);
        let got = local[[0, 0, 0, 0, 0]];
        assert!(
            (got.re - expected.re).abs() < 1e-10,
            "rank-3 first element: expected {expected:?}, got {got:?}"
        );
    }

    #[test]
    fn test_pencil_6d_no_error() {
        // Verify 6D also works
        let dfft = make_pencil_dfft(1, 0, vec![1, 1]);
        let input = make_nd_input(&[3, 3, 3, 3, 3, 3]);
        let result = dfft.decompose_data(&input);
        assert!(result.is_ok(), "6D pencil decomposition must not error");
        let local = result.expect("decompose_data failed");
        assert_eq!(local.shape(), &[3, 3, 3, 3, 3, 3]);
    }
}
