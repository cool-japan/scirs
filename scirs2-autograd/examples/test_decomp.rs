use ag::tensor_ops::*;
use ndarray::array;
use scirs2_autograd as ag;

fn main() {
    ag::run::<f64, _, _>(|g| {
        // Test 2x2 matrix
        let matrix = array![[4.0, 3.0], [0.0, -1.0]];
        let matrix_tensor = convert_to_tensor(matrix, g);

        // Test QR decomposition
        println!("Testing QR decomposition...");
        let (q, r) = qr(matrix_tensor);
        println!("Q shape: {:?}", q.eval(g).unwrap().shape());
        println!("R shape: {:?}", r.eval(g).unwrap().shape());

        // Verify Q*R = original matrix
        let reconstructed = matmul(q, r);
        println!("Original matrix:");
        println!("{:?}", matrix_tensor.eval(g).unwrap());
        println!("Q*R reconstruction:");
        println!("{:?}", reconstructed.eval(g).unwrap());

        // Test SVD decomposition
        println!("\nTesting SVD decomposition...");
        let (u, s, vt) = svd(matrix_tensor);
        println!("U shape: {:?}", u.eval(g).unwrap().shape());
        println!("S shape: {:?}", s.eval(g).unwrap().shape());
        println!("Vt shape: {:?}", vt.eval(g).unwrap().shape());

        println!("\nDecomposition tests completed successfully!");
    });
}
