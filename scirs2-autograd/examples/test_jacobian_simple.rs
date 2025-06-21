use ag::prelude::*;
use ag::tensor_ops::*;
use scirs2_autograd as ag;

fn main() {
    let mut env = ag::VariableEnvironment::new();

    // Create simple arrays directly with ndarray
    let a_array = ndarray::arr2(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
    let b_array = ndarray::arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    println!("Original arrays:");
    println!("a_array shape: {:?}", a_array.shape());
    println!("b_array shape: {:?}", b_array.shape());

    // Convert to dynamic arrays
    let a_dyn = a_array.into_dyn();
    let b_dyn = b_array.into_dyn();

    let a_id = env.set(a_dyn);
    let b_id = env.set(b_dyn);

    env.run(|g| {
        let a = g.variable(a_id);
        let b = g.variable(b_id);

        // Check actual arrays
        println!("\nVariable arrays:");
        let a_val = a.eval(g).unwrap();
        let b_val = b.eval(g).unwrap();
        println!("a shape: {:?}", a_val.shape());
        println!("b shape: {:?}", b_val.shape());

        // c = a * b, shape should be 4x3
        let c = matmul(a, b);
        let c_val = c.eval(g).unwrap();
        println!("c shape: {:?}", c_val.shape());
        println!("c:\n{:?}", c_val);

        // Test grad
        println!("\nTesting grad():");
        let grads = grad(&[c], &[a, b]);
        for (i, grad_tensor) in grads.iter().enumerate() {
            match grad_tensor.eval(g) {
                Ok(arr) => println!("grad[{}] shape: {:?}", i, arr.shape()),
                Err(e) => println!("grad[{}] error: {:?}", i, e),
            }
        }

        // Jacobian computation
        println!("\nJacobian computation:");
        let j = jacobians(c, &[a, b], 4 * 3);

        println!("Jacobians computed: {} items", j.len());

        // Check shapes
        for (i, jac) in j.iter().enumerate() {
            match jac.eval(g) {
                Ok(arr) => {
                    println!("j[{}] shape: {:?}", i, arr.shape());
                    if i == 0 {
                        println!("Expected shape for j[0]: [12, 8] (got {:?})", arr.shape());
                    } else {
                        println!("Expected shape for j[1]: [12, 6] (got {:?})", arr.shape());
                    }
                }
                Err(e) => println!("j[{}] evaluation error: {:?}", i, e),
            }
        }
    });
}
