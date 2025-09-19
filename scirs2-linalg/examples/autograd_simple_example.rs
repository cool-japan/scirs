//! Simple automatic differentiation example using scirs2-autograd directly
//!
//! This example demonstrates basic autodiff capabilities with scirs2-autograd

#[cfg(feature = "autograd")]
use ag::prelude::*;
#[cfg(feature = "autograd")]
use ag::tensor_ops as T;
#[cfg(feature = "autograd")]
use scirs2_autograd as ag;

#[cfg(not(feature = "autograd"))]
#[allow(dead_code)]
fn main() {
    println!("This example requires the 'autograd' feature. Run with:");
    println!("cargo run --example autograd_simple_example --features=autograd");
}

#[cfg(feature = "autograd")]
#[allow(dead_code)]
fn main() {
    println!("SciRS2 Automatic Differentiation Simple Example");
    println!("============================================\n");

    // Example 1: Basic derivatives
    demo_basic_derivatives();

    // Example 2: Matrix operations
    demomatrix_operations();

    // Example 3: Composite functions
    demo_composite_functions();
}

#[cfg(feature = "autograd")]
#[allow(dead_code)]
fn demo_basic_derivatives() {
    println!("1. Basic Derivatives");
    println!("-------------------");

    ag::run(|ctx: &mut ag::Context<f64>| {
        // Define variables
        let x = ctx.placeholder("x", &[]);
        let y = ctx.placeholder("y", &[]);

        // Define function: z = 2x^2 + 3y + 1
        let z = 2. * x * x + 3. * y + 1.;

        // Compute gradients
        let grads = T::grad(&[z], &[x, y]);

        // dz/dy = 3
        println!("dz/dy = {:?}", grads[1].eval(ctx));

        // dz/dx = 4x (requires feeding x value)
        let feed = ag::ndarray::arr0(2.0).into_dyn();
        let gx_val = ctx.evaluator().push(&grads[0]).feed(x, feed.view()).run()[0].clone();
        println!("dz/dx at x=2 = {:?}", gx_val);

        // Second derivative: d²z/dx² = 4
        let d2z_dx2 = &T::grad(&[grads[0]], &[x])[0];
        println!("d²z/dx² = {:?}", d2z_dx2.eval(ctx));
    });

    println!();
}

#[cfg(feature = "autograd")]
#[allow(dead_code)]
fn demomatrix_operations() {
    println!("2. Matrix Operations");
    println!("-------------------");

    ag::run(|ctx: &mut ag::Context<f64>| {
        // Create matrix placeholders
        let a = ctx.placeholder("a", &[2, 2]);
        let b = ctx.placeholder("b", &[2, 2]);

        // Matrix multiplication
        let c = T::matmul(a, b);

        // Trace of the result
        let trace_c = T::trace(c);

        // Gradient of trace w.r.t. A and B
        let grads = T::grad(&[trace_c], &[a, b]);

        // Feed concrete values
        let a_val = ag::ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b_val = ag::ndarray::arr2(&[[5.0, 6.0], [7.0, 8.0]]);

        // Evaluate gradients
        let grad_results = ctx
            .evaluator()
            .extend(&grads)
            .feed(a, a_val.view().into_dyn())
            .feed(b, b_val.view().into_dyn())
            .run();

        println!("grad(trace(A*B))/dA = \n{:?}", grad_results[0]);
        println!("grad(trace(A*B))/dB = \n{:?}", grad_results[1]);

        // Frobenius norm squared
        let norm_squared = T::sum_all(T::square(c));
        let norm_grads = T::grad(&[norm_squared], &[a, b]);

        let norm_grad_results = ctx
            .evaluator()
            .extend(&norm_grads)
            .feed(a, a_val.view().into_dyn())
            .feed(b, b_val.view().into_dyn())
            .run();

        println!("\ngrad(||A*B||²)/dA = \n{:?}", norm_grad_results[0]);
        println!("grad(||A*B||²)/dB = \n{:?}", norm_grad_results[1]);
    });

    println!();
}

#[cfg(feature = "autograd")]
#[allow(dead_code)]
fn demo_composite_functions() {
    println!("3. Composite Functions");
    println!("---------------------");

    ag::run(|ctx: &mut ag::Context<f64>| {
        // Create a matrix
        let a = ctx.placeholder("a", &[2, 2]);

        // Composite function: f(A) = trace(A^T * A)
        let a_t = T::transpose(a, &[1, 0]);
        let ata = T::matmul(a_t, a);
        let f = T::trace(ata);

        // Gradient
        let grad_f = &T::grad(&[f], &[a])[0];

        // Feed value
        let a_val = ag::ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        let result = ctx
            .evaluator()
            .push(grad_f)
            .feed(a, a_val.view().into_dyn())
            .run()[0]
            .clone();

        println!("grad(trace(A^T * A))/dA = \n{:?}", result);

        // Note: Determinant gradient example disabled due to element access API changes
        // The 'at' method for accessing individual elements is not currently available
        println!("\nDeterminant gradient example skipped (element access API not available)");

        // Note: Matrix inverse gradient example disabled due to element access API changes
        // The 'at' method for accessing individual elements is not currently available
        println!("\nMatrix inverse gradient example skipped (element access API not available)");
    });

    println!();
}
