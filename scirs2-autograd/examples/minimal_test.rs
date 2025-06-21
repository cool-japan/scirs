extern crate scirs2_autograd as ag;
use ag::tensor_ops as T;
use ndarray::array;

fn main() {
    println!("Running basic shape tests");

    ag::run::<f64, _, _>(|ctx| {
        // Test scalar
        let scalar = T::scalar(5.0, ctx);
        println!("Scalar shape: {:?}", scalar.eval(ctx).unwrap().shape());

        // Test vector
        let vector = T::ones(&[3], ctx);
        println!("Vector shape: {:?}", vector.eval(ctx).unwrap().shape());

        // Test matrix
        let matrix = T::ones(&[2, 2], ctx);
        println!("Matrix shape: {:?}", matrix.eval(ctx).unwrap().shape());

        // Test matrix operations
        let matrix_a = T::convert_to_tensor(array![[1.0, 2.0], [3.0, 4.0]], ctx);
        let det = T::determinant(matrix_a);
        println!("Determinant: {}", det.eval(ctx).unwrap()[[]]);

        let inv = T::matrix_inverse(matrix_a);
        println!("Inverse shape: {:?}", inv.eval(ctx).unwrap().shape());

        println!("All tests passed!");
    });
}
