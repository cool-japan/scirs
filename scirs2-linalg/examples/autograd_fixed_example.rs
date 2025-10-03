//! Working automatic differentiation example using scirs2-autograd
//!
//! This example demonstrates basic autodiff capabilities with scirs2-autograd

#[cfg(feature = "autograd")]
use ag::tensor_ops as T;
#[cfg(feature = "autograd")]
use scirs2_autograd as ag;

#[cfg(not(feature = "autograd"))]
#[allow(dead_code)]
fn main() {
    println!("This example requires the 'autograd' feature. Run with:");
    println!("cargo run --example autograd_fixed_example --features=autograd");
}

#[cfg(feature = "autograd")]
#[allow(dead_code)]
fn main() {
    println!("SciRS2 Automatic Differentiation Example");
    println!("======================================\n");

    // Example 1: Basic derivatives
    demo_basic_derivatives();

    // Example 2: Matrix operations
    demomatrix_operations();

    // Example 3: Linear algebra operations
    demo_linalg_ops();
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
        let feed = ag::ndarray::arr0(2.0);
        let gx_val = ctx
            .evaluator()
            .push(&grads[0])
            .feed(x, feed.view().into_dyn())
            .run()[0]
            .clone();
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

        // Trace equivalent using sum of diagonal
        let diag_sum = T::extract_diag(c);
        let trace_c = T::sum_all(diag_sum);

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
        let c_flat = T::flatten(c);
        let norm_squared = T::sum_all(T::square(c_flat));
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
fn demo_linalg_ops() {
    println!("3. Linear Algebra Operations");
    println!("---------------------------");

    ag::run(|ctx: &mut ag::Context<f64>| {
        // Create a matrix
        let a = ctx.placeholder("a", &[2, 2]);

        // Matrix operations
        let a_t = T::transpose(a, &[1, 0]);
        let ata = T::matmul(a_t, a);

        // Compute trace
        let diag = T::extract_diag(ata);
        let trace_ata = T::sum_all(diag);

        // Gradient
        let grad_trace = &T::grad(&[trace_ata], &[a])[0];

        // Feed value
        let a_val = ag::ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        let result = ctx
            .evaluator()
            .push(grad_trace)
            .feed(a, a_val.view().into_dyn())
            .run()[0]
            .clone();

        println!("grad(trace(A^T * A))/dA = \n{:?}", result);

        // Another example: matrix norm
        let a_flat = T::flatten(a);
        let a_norm = T::sqrt(T::sum_all(T::square(a_flat)));
        let grad_norm = &T::grad(&[a_norm], &[a])[0];

        let norm_result = ctx
            .evaluator()
            .push(grad_norm)
            .feed(a, a_val.view().into_dyn())
            .run()[0]
            .clone();

        println!("\ngrad(||A||)/dA = \n{:?}", norm_result);

        // Matrix exponential approximation (first 3 terms of Taylor series)
        let i = T::eye(2, ctx);
        let a_sq = T::matmul(a, a);
        let exp_approx = i + a + T::scalar_mul(a_sq, 0.5);

        // Sum all elements
        let exp_sum = T::sum_all(T::flatten(exp_approx));
        let grad_exp = &T::grad(&[exp_sum], &[a])[0];

        let exp_result = ctx
            .evaluator()
            .push(grad_exp)
            .feed(a, a_val.view().into_dyn())
            .run()[0]
            .clone();

        println!("\ngrad(sum(exp(A) approx))/dA = \n{:?}", exp_result);
    });

    println!();
}
