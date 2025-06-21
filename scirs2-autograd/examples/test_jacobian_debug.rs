use ag::prelude::*;
use ag::tensor_ops::*;
use scirs2_autograd as ag;

fn main() {
    let mut env = ag::VariableEnvironment::new();

    // Create simple arrays
    let a_array = ndarray::arr2(&[[1.0f32, 2.0], [3.0, 4.0]]).into_dyn();
    let b_array = ndarray::arr2(&[[1.0f32, 2.0], [3.0, 4.0]]).into_dyn();

    let a_id = env.set(a_array);
    let b_id = env.set(b_array);

    env.run(|g| {
        let a = g.variable(a_id);
        let b = g.variable(b_id);

        // c = a * b (element-wise), shape should be 2x2
        let c = a * b;

        println!("c shape: {:?}", c.eval(g).unwrap().shape());
        println!("c:\n{:?}", c.eval(g).unwrap());

        // Test element access
        println!("\nTesting element access:");
        let c_00 = c.access_elem(0);
        let c_01 = c.access_elem(1);
        let c_10 = c.access_elem(2);
        let c_11 = c.access_elem(3);

        println!("c[0,0] = {:?}", c_00.eval(g).unwrap());
        println!("c[0,1] = {:?}", c_01.eval(g).unwrap());
        println!("c[1,0] = {:?}", c_10.eval(g).unwrap());
        println!("c[1,1] = {:?}", c_11.eval(g).unwrap());

        // Test grad with single output
        println!("\nTesting grad with c[0,0]:");
        let grads = grad(&[c_00], &[a, b]);
        for (i, grad_tensor) in grads.iter().enumerate() {
            match grad_tensor.eval(g) {
                Ok(arr) => {
                    println!("grad[{}] shape: {:?}", i, arr.shape());
                    println!("grad[{}]:\n{:?}", i, arr);
                }
                Err(e) => println!("grad[{}] error: {:?}", i, e),
            }
        }

        // Now test jacobians with smaller size
        println!("\nTesting jacobians with size 4:");
        let j = jacobians(c, &[a, b], 4);

        for (i, jac) in j.iter().enumerate() {
            match jac.eval(g) {
                Ok(arr) => {
                    println!("j[{}] shape: {:?}", i, arr.shape());
                    println!("j[{}]:\n{:?}", i, arr);
                }
                Err(e) => println!("j[{}] error: {:?}", i, e),
            }
        }
    });
}
