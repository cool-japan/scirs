use ag::prelude::*;
use ag::tensor_ops::*;
use scirs2_autograd as ag;

fn main() {
    let mut env = ag::VariableEnvironment::new();

    // Create simple 2x2 matrices for clarity
    let a_array = ndarray::arr2(&[[1.0f32, 2.0], [3.0, 4.0]]).into_dyn();
    let b_array = ndarray::arr2(&[[5.0f32, 6.0], [7.0, 8.0]]).into_dyn();

    let a_id = env.set(a_array);
    let b_id = env.set(b_array);

    env.run(|g| {
        let a = g.variable(a_id);
        let b = g.variable(b_id);

        // c = matmul(a, b), shape should be 2x2
        let c = matmul(a, b);

        println!("Testing matmul gradients:");
        println!("a shape: {:?}", a.eval(g).unwrap().shape());
        println!("b shape: {:?}", b.eval(g).unwrap().shape());
        println!("c shape: {:?}", c.eval(g).unwrap().shape());
        println!("c =\n{:?}", c.eval(g).unwrap());

        // Test gradient of entire c with respect to a and b
        println!("\nGradient of entire c:");
        let grads = grad(&[c], &[a, b]);
        for (i, grad_tensor) in grads.iter().enumerate() {
            match grad_tensor.eval(g) {
                Ok(arr) => {
                    println!(
                        "grad[{}] shape: {:?} (should be same as input shape)",
                        i,
                        arr.shape()
                    );
                    println!("grad[{}] =\n{:?}", i, arr);
                }
                Err(e) => println!("grad[{}] error: {:?}", i, e),
            }
        }

        // Test gradient of a single element
        println!("\nGradient of c[0,0]:");
        let c_00 = c.access_elem(0);
        let grads_00 = grad(&[c_00], &[a, b]);
        for (i, grad_tensor) in grads_00.iter().enumerate() {
            match grad_tensor.eval(g) {
                Ok(arr) => {
                    println!("grad[{}] shape: {:?}", i, arr.shape());
                    println!("grad[{}] =\n{:?}", i, arr);
                }
                Err(e) => println!("grad[{}] error: {:?}", i, e),
            }
        }

        // Now manually build the jacobian to see what happens
        println!("\nManual jacobian construction:");
        let mut jac_rows = Vec::new();
        for i in 0..4 {
            let c_i = c.access_elem(i);
            let grads_i = grad(&[c_i], &[a]);
            let grad_a = &grads_i[0];

            // Check what we get
            match grad_a.eval(g) {
                Ok(arr) => {
                    println!("grad of c[{}] w.r.t. a: shape {:?}", i, arr.shape());

                    // Flatten it
                    let flattened = flatten(grad_a);
                    match flattened.eval(g) {
                        Ok(f_arr) => {
                            println!("  flattened shape: {:?}", f_arr.shape());
                            // Expand dims to make it a row
                            let row = expand_dims(flattened, &[0]);
                            jac_rows.push(row);
                        }
                        Err(e) => println!("  flatten error: {:?}", e),
                    }
                }
                Err(e) => println!("grad error: {:?}", e),
            }
        }

        // Try to concat the rows
        if jac_rows.len() == 4 {
            let jac = concat(&jac_rows, 0);
            match jac.eval(g) {
                Ok(arr) => {
                    println!("\nFinal jacobian shape: {:?}", arr.shape());
                    println!("Expected shape: [4, 4]");
                }
                Err(e) => println!("Concat error: {:?}", e),
            }
        }
    });
}
