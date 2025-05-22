use ag::tensor_ops::*;
use scirs2_autograd as ag;

fn main() {
    println!("Minimal matrix multiplication example");

    // Create tensors and perform matrix multiplication
    ag::run(|g| {
        // Create input matrices
        let a = convert_to_tensor(ag::ndarray::array![[1.0, 2.0], [3.0, 4.0]], g);
        let b = convert_to_tensor(ag::ndarray::array![[5.0, 6.0], [7.0, 8.0]], g);

        // Transpose b
        let b_t = transpose(b, &[1, 0]);

        // Perform matrix multiplication
        let c = matmul(a, b_t);

        // Show result
        println!("a * b^T = {:?}", c.eval(g).unwrap());

        println!("Matrix multiplication completed successfully!");
    });
}
