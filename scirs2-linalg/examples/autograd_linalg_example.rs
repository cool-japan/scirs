//! Linear algebra automatic differentiation example
//!
//! This example demonstrates how to use automatic differentiation with linear
//! algebra operations in scirs2-linalg, including workarounds for operations
//! that are not yet available in scirs2-autograd.

#[cfg(feature = "autograd")]
use ag::tensor_ops::*;
#[cfg(feature = "autograd")]
use scirs2_autograd as ag;
#[cfg(feature = "autograd")]
use scirs2_linalg::autograd::helpers;

#[cfg(not(feature = "autograd"))]
#[allow(dead_code)]
fn main() {
    println!("This example requires the 'autograd' feature. Run with:");
    println!("cargo run --example autograd_linalg_example --features=autograd");
}

#[cfg(feature = "autograd")]
#[allow(dead_code)]
fn main() {
    println!("SciRS2 Linear Algebra Automatic Differentiation Example");
    println!("=====================================================\n");

    // Example 1: Basic matrix operations
    demo_basicmatrix_ops();

    // Example 2: Trace and matrix functions
    demo_trace_and_functions();

    // Example 3: Advanced operations
    demo_advanced_ops();
}

#[cfg(feature = "autograd")]
#[allow(dead_code)]
fn demo_basicmatrix_ops() {
    println!("1. Basic Matrix Operations");
    println!("-------------------------");

    ag::run(|ctx: &mut ag::Context<f64>| {
        // Create matrix placeholders
        let a = ctx.placeholder("a", &[2, 2]);
        let b = ctx.placeholder("b", &[2, 2]);

        // Matrix multiplication
        let c = matmul(a, b);

        // Frobenius norm squared
        let c_flat = flatten(c);
        let frobenius_sq = sum_all(square(c_flat));

        // Compute gradients
        let grads = grad(&[frobenius_sq], &[a, b]);

        // Feed concrete values
        let a_val = ag::ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b_val = ag::ndarray::arr2(&[[5.0, 6.0], [7.0, 8.0]]);

        // Evaluate
        let results = ctx
            .evaluator()
            .extend(&grads)
            .feed(a, a_val.view().into_dyn())
            .feed(b, b_val.view().into_dyn())
            .run();

        println!("A = \n{}", a_val);
        println!("B = \n{}", b_val);
        println!("grad(||A*B||²)/dA = \n{:?}", results[0]);
        println!("grad(||A*B||²)/dB = \n{:?}", results[1]);
    });

    println!();
}

#[cfg(feature = "autograd")]
#[allow(dead_code)]
fn demo_trace_and_functions() {
    println!("2. Trace and Matrix Functions");
    println!("----------------------------");

    ag::run(|ctx: &mut ag::Context<f64>| {
        // Create a matrix
        let a = ctx.placeholder("a", &[3, 3]);

        // Compute trace using workaround
        let trace_a = helpers::trace_workaround(&a, 3, ctx);

        // Create another operation: trace(A² + A)
        let a_squared = matmul(a, a);
        let a_plus_a2 = a + a_squared;
        let trace_result = helpers::trace_workaround(&a_plus_a2, 3, ctx);

        // Gradient
        let grad_trace = &grad(&[trace_result], &[a])[0];

        // Feed value
        let a_val = ag::ndarray::arr2(&[[1.0, 2.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 2.0]]);

        let result = ctx
            .evaluator()
            .push(grad_trace)
            .feed(a, a_val.view().into_dyn())
            .run()[0]
            .clone();

        println!("A = \n{}", a_val);
        println!("grad(trace(A + A²))/dA = \n{:?}", result);

        // Also evaluate the trace itself
        let trace_val = ctx
            .evaluator()
            .push(&trace_result)
            .feed(a, a_val.view().into_dyn())
            .run()[0]
            .clone();

        println!("trace(A + A²) = {:?}", trace_val);
    });

    println!();
}

#[cfg(feature = "autograd")]
#[allow(dead_code)]
fn demo_advanced_ops() {
    println!("3. Advanced Operations");
    println!("---------------------");

    ag::run(|ctx: &mut ag::Context<f64>| {
        // Create matrices
        let x = ctx.placeholder("x", &[3, 2]);
        let w = ctx.placeholder("w", &[2, 4]);
        let b = ctx.placeholder("b", &[1, 4]);

        // Linear transformation: Y = X*W + B (with broadcasting)
        let y = matmul(x, w) + b;

        // Apply activation (tanh)
        let activated = tanh(y);

        // Loss: mean squared norm
        let flat = flatten(activated);
        let loss = reduce_mean(square(flat), &[0], false);

        // Gradients
        let grads = grad(&[loss], &[x, w, b]);

        // Feed values
        let x_val = ag::ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let w_val = ag::ndarray::arr2(&[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]);
        let b_val = ag::ndarray::arr2(&[[0.1, 0.2, 0.3, 0.4]]);

        let results = ctx
            .evaluator()
            .extend(&grads)
            .feed(x, x_val.view().into_dyn())
            .feed(w, w_val.view().into_dyn())
            .feed(b, b_val.view().into_dyn())
            .run();

        println!("Neural network layer gradient computation:");
        println!("X shape: {:?}", x_val.shape());
        println!("W shape: {:?}", w_val.shape());
        println!("b shape: {:?}", b_val.shape());
        println!("\nGradients:");
        println!("dL/dX shape: {:?}", results[0].as_ref().unwrap().shape());
        println!("dL/dW shape: {:?}", results[1].as_ref().unwrap().shape());
        println!("dL/db shape: {:?}", results[2].as_ref().unwrap().shape());

        // Create identity for another example
        let eye_3 = helpers::eye_workaround(3, ctx);
        let eye_val = eye_3.eval(ctx).unwrap();
        println!("\nIdentity matrix (3x3):\n{:?}", eye_val);
    });

    println!();
}
