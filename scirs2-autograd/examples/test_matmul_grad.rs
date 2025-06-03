use ag::prelude::*;
use ag::tensor_ops::*;
use ndarray;
use scirs2_autograd as ag;

fn main() {
    let mut env = ag::VariableEnvironment::new();

    // Create matrices
    let a_array = ndarray::arr2(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).into_dyn();
    let b_array = ndarray::arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();

    let a_id = env.set(a_array);
    let b_id = env.set(b_array);

    env.run(|g| {
        let a = g.variable(a_id);
        let b = g.variable(b_id);

        // c = matmul(a, b), shape should be 4x3
        let c = matmul(a, b);

        println!("a shape: {:?}", a.eval(g).unwrap().shape());
        println!("b shape: {:?}", b.eval(g).unwrap().shape());
        println!("c shape: {:?}", c.eval(g).unwrap().shape());

        // Test element access
        let c_00 = c.access_elem(0);
        println!("\nc[0,0] = {:?}", c_00.eval(g).unwrap());

        // Test grad with single output element
        println!("\nTesting grad of c[0,0] w.r.t. a and b:");
        let grads = grad(&[c_00], &[a, b]);

        for (i, grad_tensor) in grads.iter().enumerate() {
            match grad_tensor.eval(g) {
                Ok(arr) => {
                    println!("grad[{}] shape: {:?}", i, arr.shape());
                    println!("grad[{}]:\n{:?}", i, arr);

                    // Test flatten
                    let flattened = flatten(grad_tensor);
                    match flattened.eval(g) {
                        Ok(f_arr) => println!("flattened grad[{}] shape: {:?}", i, f_arr.shape()),
                        Err(e) => println!("flatten error: {:?}", e),
                    }
                }
                Err(e) => println!("grad[{}] error: {:?}", i, e),
            }
        }
    });
}
