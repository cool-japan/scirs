use ag::ndarray::array;
use ag::tensor_ops as T;
use scirs2_autograd as ag;

#[test]
fn test_minimal_matmul() {
    ag::run::<f32, _, _>(|ctx| {
        // Create two simple test matrices
        let a_data = array![[1.0, 2.0], [3.0, 4.0]];
        let b_data = array![[5.0, 6.0], [7.0, 8.0]];

        println!("Matrix A:\n{:?}", a_data);
        println!("Matrix B:\n{:?}", b_data);

        // Create placeholders
        let a = ctx.placeholder("a", &[2, 2]);
        let b = ctx.placeholder("b", &[2, 2]);

        // Compute matrix multiplication
        let c = T::matmul(a, b);

        // Evaluate using evaluator.feed() approach instead of Feeder
        let result = ctx
            .evaluator()
            .push(&c)
            .feed(a, a_data.view().into_dyn())
            .feed(b, b_data.view().into_dyn())
            .run()[0]
            .clone()
            .unwrap();

        println!("Result of A * B:\n{:?}", result);

        // Expected result: [[19, 22], [43, 50]]
        let expected = array![[19.0, 22.0], [43.0, 50.0]];
        let result_2d = result.into_dimensionality::<ag::ndarray::Ix2>().unwrap();
        let diff = (result_2d - &expected).mapv(|x| x.abs()).sum();

        println!("Difference from expected: {}", diff);
        assert!(diff < 1e-5, "Matrix multiplication failed, diff: {}", diff);
    });
}
