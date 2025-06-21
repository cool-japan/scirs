use ag::tensor_ops::*;
use ndarray::array;
use scirs2_autograd as ag;

fn main() {
    ag::run(|ctx| {
        let a_data = array![[3.0f32, 4.0], [0.0, 1.0]];
        println!("Input matrix A:\n{:?}", a_data);

        let a = ctx.placeholder("a", &[2, 2]);
        let feeder = ag::Feeder::new().push(a, a_data.view().into_dyn());

        // Test QR decomposition
        let (q, r) = qr(a);

        // Evaluate
        let results = ctx
            .evaluator()
            .push(&q)
            .push(&r)
            .set_feeder(feeder.clone())
            .run();

        let q_result = results[0].clone().unwrap();
        let r_result = results[1].clone().unwrap();

        println!("Q result:\n{:?}", q_result);
        println!("R result:\n{:?}", r_result);

        // Verify Q is orthogonal
        let q_2d = q_result.into_dimensionality::<ag::ndarray::Ix2>().unwrap();
        let r_2d = r_result.into_dimensionality::<ag::ndarray::Ix2>().unwrap();

        let qt_q = q_2d.t().dot(&q_2d);
        println!("Q^T * Q:\n{:?}", qt_q);

        // Verify A = Q * R
        let reconstructed = q_2d.dot(&r_2d);
        println!("Q * R:\n{:?}", reconstructed);

        println!("Original A:\n{:?}", a_data);
    });
}
