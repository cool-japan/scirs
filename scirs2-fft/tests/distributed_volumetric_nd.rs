/// Tests for n-D volumetric decomposition (n >= 4) in the distributed FFT module.
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

    fn make_vol_dfft(node_count: usize, rank: usize, process_grid: Vec<usize>) -> DistributedFFT {
        let config = DistributedConfig {
            node_count,
            rank,
            decomposition: DecompositionStrategy::Volumetric,
            communication: CommunicationPattern::AllToAll,
            process_grid,
            local_size: vec![],
            max_local_size: 64,
        };
        let comm = Arc::new(MockCommunicator::new(node_count, rank));
        DistributedFFT::new(config, comm)
    }

    /// Build a flat-index-valued complex array.
    fn make_nd_input(shape: &[usize]) -> ArrayD<Complex64> {
        let total: usize = shape.iter().product();
        let data: Vec<Complex64> = (0..total)
            .map(|i| Complex64::new(i as f64 * 0.1, 0.0))
            .collect();
        ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape/data mismatch")
    }

    #[test]
    fn test_volumetric_4d_single_node_no_error() {
        // Single-node 1x1x1 grid; 4D input must not error (previously returned DimensionError)
        let dfft = make_vol_dfft(1, 0, vec![1, 1, 1]);
        let input = make_nd_input(&[4, 4, 4, 4]);
        let result = dfft.decompose_data(&input);
        assert!(
            result.is_ok(),
            "volumetric decomposition on 4D input must succeed, got: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_volumetric_4d_single_node_shape() {
        let dfft = make_vol_dfft(1, 0, vec![1, 1, 1]);
        let input = make_nd_input(&[4, 4, 4, 4]);
        let local = dfft.decompose_data(&input).expect("decompose_data failed");
        assert_eq!(
            local.shape(),
            &[4, 4, 4, 4],
            "single-node volumetric 4D output shape must equal input shape"
        );
    }

    #[test]
    fn test_volumetric_4d_values_correct() {
        let dfft = make_vol_dfft(1, 0, vec![1, 1, 1]);
        let input = make_nd_input(&[4, 4, 4, 4]);
        let local = dfft.decompose_data(&input).expect("decompose_data failed");
        let total = 4usize.pow(4);
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
    fn test_volumetric_6d_single_node_no_error() {
        // 6D input: exercises the n-D general branch (ndim=6 > 3)
        let dfft = make_vol_dfft(1, 0, vec![1, 1, 1]);
        let input = make_nd_input(&[3, 3, 3, 3, 3, 3]);
        let result = dfft.decompose_data(&input);
        assert!(result.is_ok(), "6D volumetric decomposition must not error");
        let local = result.expect("decompose_data failed");
        assert_eq!(local.shape(), &[3, 3, 3, 3, 3, 3]);
    }

    #[test]
    fn test_volumetric_6d_values_correct() {
        let dfft = make_vol_dfft(1, 0, vec![1, 1, 1]);
        let input = make_nd_input(&[3, 3, 3, 3, 3, 3]);
        let local = dfft.decompose_data(&input).expect("decompose_data failed");
        let total = 3usize.pow(6);
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
    fn test_volumetric_7d_single_node_no_error() {
        // 7D input: deep n-D test
        let dfft = make_vol_dfft(1, 0, vec![1, 1, 1]);
        let input = make_nd_input(&[2, 2, 2, 2, 2, 2, 2]);
        let result = dfft.decompose_data(&input);
        assert!(result.is_ok(), "7D volumetric decomposition must not error");
        let local = result.expect("decompose_data failed");
        assert_eq!(local.shape(), &[2, 2, 2, 2, 2, 2, 2]);
    }

    #[test]
    fn test_volumetric_8_node_4d_rank0_shape() {
        // 8 nodes with 2x2x2 grid: rank 0 owns first half of each partitioned axis
        let dfft = make_vol_dfft(8, 0, vec![2, 2, 2]);
        let input = make_nd_input(&[4, 4, 4, 4]);
        let local = dfft.decompose_data(&input).expect("decompose_data failed");
        // rank 0: plane 0, row 0, col 0 → my_start_plane=0, my_start_row=0, my_start_col=0
        // each partition: 4/2 = 2 elements
        assert_eq!(
            local.shape()[0],
            2,
            "axis 0 should have 2 elements for rank 0"
        );
        assert_eq!(
            local.shape()[1],
            2,
            "axis 1 should have 2 elements for rank 0"
        );
        assert_eq!(
            local.shape()[2],
            2,
            "axis 2 should have 2 elements for rank 0"
        );
        assert_eq!(local.shape()[3], 4, "axis 3 (unpartitioned) unchanged");
    }

    #[test]
    fn test_volumetric_8_node_4d_rank7_values() {
        // 8 nodes 2x2x2 grid: rank 7 → my_plane=1, my_row=1, my_col=1
        // my_start_plane=2, my_start_row=2, my_start_col=2
        let dfft = make_vol_dfft(8, 7, vec![2, 2, 2]);
        let input = make_nd_input(&[4, 4, 4, 4]);
        let local = dfft.decompose_data(&input).expect("decompose_data failed");
        // First element: input[2,2,2,0]
        // flat index: ((2*4 + 2)*4 + 2)*4 + 0 = (10*4+2)*4 = 42*4 = 168
        let expected = Complex64::new(168.0 * 0.1, 0.0);
        let got = local[[0, 0, 0, 0]];
        assert!(
            (got.re - expected.re).abs() < 1e-10,
            "rank-7 first element: expected {expected:?}, got {got:?}"
        );
    }
}
