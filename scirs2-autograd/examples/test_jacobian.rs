use ag::prelude::*;
use ag::tensor_ops::*;
use scirs2_autograd as ag;

fn main() {
    let mut env = ag::VariableEnvironment::new();

    let mut rng = ag::ndarray_ext::ArrayRng::<f32>::default();
    // Create variables with known shapes
    let a = env.set(rng.standard_normal(&[4, 2])); // 4x2 matrix
    let b = env.set(rng.standard_normal(&[2, 3])); // 2x3 matrix

    env.run(|g| {
        let a = g.variable(a);
        let b = g.variable(b);

        // Check the actual arrays
        println!("a array shape: {:?}", a.eval(g).unwrap().shape());
        println!("b array shape: {:?}", b.eval(g).unwrap().shape());

        // c = a * b, shape should be 4x3
        let c = matmul(a, b);
        println!("c array shape: {:?}", c.eval(g).unwrap().shape());

        println!("\nShape tensors:");
        println!("a shape: {:?}", shape(a).eval(g).unwrap());
        println!("b shape: {:?}", shape(b).eval(g).unwrap());
        println!("c shape: {:?}", shape(c).eval(g).unwrap());

        // Jacobian computation
        let j = jacobians(c, &[a, b], 4 * 3);

        println!("\nJacobians computed: {} items", j.len());

        // Check shapes
        for (i, jac) in j.iter().enumerate() {
            let shape_result = shape(jac).eval(g);
            println!("j[{}] shape result: {:?}", i, shape_result);

            if let Ok(shape_arr) = shape_result {
                println!("j[{}] shape: {:?}", i, shape_arr);
            }
        }

        // Try to evaluate the jacobians
        println!("\nEvaluating jacobians:");
        for (i, jac) in j.iter().enumerate() {
            match jac.eval(g) {
                Ok(arr) => {
                    println!("j[{}] evaluated shape: {:?}", i, arr.shape());
                    if i == 0 {
                        println!("Expected shape for j[0]: [12, 8] (got {:?})", arr.shape());
                    } else {
                        println!("Expected shape for j[1]: [12, 6] (got {:?})", arr.shape());
                    }
                }
                Err(e) => println!("j[{}] evaluation error: {:?}", i, e),
            }
        }

        // Let's also test grad() function to see what it returns
        println!("\nTesting grad() directly:");
        let grads = grad(&[c], &[a, b]);
        for (i, grad_tensor) in grads.iter().enumerate() {
            match grad_tensor.eval(g) {
                Ok(arr) => println!("grad[{}] shape: {:?}", i, arr.shape()),
                Err(e) => println!("grad[{}] error: {:?}", i, e),
            }
        }
    });
}
