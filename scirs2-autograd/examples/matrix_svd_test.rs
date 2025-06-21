use ag::tensor_ops::*;
use ndarray::array;
use scirs2_autograd as ag;

fn main() {
    println!("Testing SVD with autograd");

    // Test 1: Use a simple manual SVD test
    let mat = array![[1.0, 2.0], [3.0, 4.0]];
    println!("Original matrix: {:?}", mat);

    // For this example, we'll use the autograd SVD implementation
    ag::run::<f64, _, _>(|g| {
        let mat_tensor = convert_to_tensor(mat.clone().into_dyn(), g);

        // Compute SVD using autograd
        let (u, s, v) = svd(mat_tensor);

        println!("SVD result from autograd:");
        let u_val = u.eval(g).unwrap();
        let s_val = s.eval(g).unwrap();
        let v_val = v.eval(g).unwrap();

        println!("U shape: {:?}", u_val.shape());
        println!("S shape: {:?}", s_val.shape());
        println!("V shape: {:?}", v_val.shape());

        // For now, just test that the decomposition worked without errors
        println!("SVD decomposition completed successfully");
    });

    // Test 2: Test gradients through SVD
    println!("\nTesting simple SVD gradient implementation in autograd");

    // The SVD wrapper function
    fn simple_svd<'g>(
        matrix: &ag::tensor::Tensor<'g, f64>,
    ) -> (
        ag::tensor::Tensor<'g, f64>,
        ag::tensor::Tensor<'g, f64>,
        ag::tensor::Tensor<'g, f64>,
    ) {
        // For this example, we'll just use the built-in SVD
        svd(matrix)
    }

    // Test the function in autograd context
    ag::run::<f64, _, _>(|g| {
        // Create the input tensor
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let x = variable(data.clone(), g);

        // Apply our SVD function
        let (u, s, vt) = simple_svd(&x);

        // Test reconstruction
        let s_diag = diag(s);
        let reconstructed = matmul(matmul(u, s_diag), vt);

        // Calculate the reconstruction error
        let diff = sub(reconstructed, x);
        let error = sum_all(square(diff));

        // Check gradients
        println!("Testing gradients through reconstruction error");
        let error_grads = grad(&[error], &[&x]);
        let grad_x = &error_grads[0];

        match grad_x.eval(g) {
            Ok(arr) => {
                println!("Gradient of reconstruction error w.r.t input:");
                println!("Shape: {:?}", arr.shape());
                println!("Values: {:?}", arr);
            }
            Err(e) => println!("Error computing gradient: {:?}", e),
        }

        // Test gradients through sum of U
        println!("\nTesting gradients through sum of U");
        let sum_u = sum_all(u);
        let sum_u_grads = grad(&[sum_u], &[&x]);
        let grad_x_u = &sum_u_grads[0];

        match grad_x_u.eval(g) {
            Ok(arr) => {
                println!("Gradient of sum(U) w.r.t input:");
                println!("Shape: {:?}", arr.shape());
                println!("Values: {:?}", arr);
            }
            Err(e) => println!("Error computing gradient: {:?}", e),
        }

        // Test gradients through sum of S
        println!("\nTesting gradients through sum of S");
        let sum_s = sum_all(s);
        let sum_s_grads = grad(&[sum_s], &[&x]);
        let grad_x_s = &sum_s_grads[0];

        match grad_x_s.eval(g) {
            Ok(arr) => {
                println!("Gradient of sum(S) w.r.t input:");
                println!("Shape: {:?}", arr.shape());
                println!("Values: {:?}", arr);
            }
            Err(e) => println!("Error computing gradient: {:?}", e),
        }

        // Test gradients through sum of V^T
        println!("\nTesting gradients through sum of V^T");
        let sum_vt = sum_all(vt);
        let sum_vt_grads = grad(&[sum_vt], &[&x]);
        let grad_x_vt = &sum_vt_grads[0];

        match grad_x_vt.eval(g) {
            Ok(arr) => {
                println!("Gradient of sum(V^T) w.r.t input:");
                println!("Shape: {:?}", arr.shape());
                println!("Values: {:?}", arr);
            }
            Err(e) => println!("Error computing gradient: {:?}", e),
        }
    });
}
