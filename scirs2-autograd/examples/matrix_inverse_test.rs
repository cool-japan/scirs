extern crate scirs2_autograd as ag;
use ag::ndarray;
use ag::tensor_ops as T;

fn main() {
    println!("Testing matrix inverse operation");

    // Create and evaluate a 2x2 matrix
    ag::run::<f64, _, _>(|ctx| {
        // Create a 2x2 matrix with specific values
        println!("Creating a 2x2 matrix");
        let matrix_data = ndarray::arr2(&[[2.0, 0.0], [0.0, 2.0]]);

        // Convert to tensor
        let matrix = T::convert_to_tensor(matrix_data.into_dyn(), ctx);

        // Print the matrix
        println!("Original matrix:");
        let result = matrix.eval(ctx).unwrap();
        println!("{:?}", result);

        // Compute the inverse
        println!("Computing inverse");
        let inverse = T::matrix_inverse(matrix);

        // Print the inverse
        println!("Matrix inverse:");
        match inverse.eval(ctx) {
            Ok(inv) => println!("{:?}", inv),
            Err(e) => println!("Error computing inverse: {:?}", e),
        }

        // Try a determinant
        println!("Computing determinant");
        let det = T::determinant(matrix);

        // Print the determinant
        println!("Matrix determinant:");
        match det.eval(ctx) {
            Ok(d) => println!("{:?}", d),
            Err(e) => println!("Error computing determinant: {:?}", e),
        }
    });
}
