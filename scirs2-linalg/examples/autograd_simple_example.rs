//! Simple automatic differentiation example using scirs2-autograd directly
//!
//! This example demonstrates basic autodiff capabilities with scirs2-autograd

#[cfg(feature = "autograd")]
use scirs2_autograd as ag;
#[cfg(feature = "autograd")]
use ag::tensor_ops as T;
#[cfg(feature = "autograd")]
use ag::prelude::*;

#[cfg(not(feature = "autograd"))]
fn main() {
    println!("This example requires the 'autograd' feature. Run with:");
    println!("cargo run --example autograd_simple_example --features=autograd");
}

#[cfg(feature = "autograd")]
fn main() {
    println!("SciRS2 Automatic Differentiation Simple Example");
    println!("============================================\n");

    // Example 1: Basic derivatives
    demo_basic_derivatives();

    // Example 2: Matrix operations
    demo_matrix_operations();

    // Example 3: Composite functions
    demo_composite_functions();
}

#[cfg(feature = "autograd")]
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
        let gx_val = ctx.evaluator()
            .push(&grads[0])
            .feed(x, feed.view())
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
fn demo_matrix_operations() {
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
        let grad_results = ctx.evaluator()
            .extend(&grads)
            .feed(a, a_val.view())
            .feed(b, b_val.view())
            .run();
        
        println!("grad(trace(A*B))/dA = \n{:?}", grad_results[0]);
        println!("grad(trace(A*B))/dB = \n{:?}", grad_results[1]);
        
        // Frobenius norm squared
        let norm_squared = T::scalar_sum(T::square(c));
        let norm_grads = T::grad(&[norm_squared], &[a, b]);
        
        let norm_grad_results = ctx.evaluator()
            .extend(&norm_grads)
            .feed(a, a_val.view())
            .feed(b, b_val.view())
            .run();
        
        println!("\ngrad(||A*B||²)/dA = \n{:?}", norm_grad_results[0]);
        println!("grad(||A*B||²)/dB = \n{:?}", norm_grad_results[1]);
    });
    
    println!();
}

#[cfg(feature = "autograd")]
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
        
        let result = ctx.evaluator()
            .push(grad_f)
            .feed(a, a_val.view())
            .run()[0]
            .clone();
        
        println!("grad(trace(A^T * A))/dA = \n{:?}", result);
        
        // Another composite: g(A) = det(A) (for 2x2 only)
        // Using a manual implementation since autograd might not have det
        let det_a = a.at(&[0, 0]) * a.at(&[1, 1]) - a.at(&[0, 1]) * a.at(&[1, 0]);
        let grad_det = &T::grad(&[det_a], &[a])[0];
        
        let det_grad_result = ctx.evaluator()
            .push(grad_det)
            .feed(a, a_val.view())
            .run()[0]
            .clone();
        
        println!("\ngrad(det(A))/dA = \n{:?}", det_grad_result);
        
        // Matrix inverse gradient (simplified for 2x2)
        let det_inv = T::scalar_inv(det_a);
        let adj_00 = a.at(&[1, 1]);
        let adj_01 = -a.at(&[0, 1]);
        let adj_10 = -a.at(&[1, 0]);
        let adj_11 = a.at(&[0, 0]);
        
        // Just trace of inverse as example
        let inv_trace = det_inv * (adj_00 + adj_11);
        let grad_inv_trace = &T::grad(&[inv_trace], &[a])[0];
        
        let inv_trace_result = ctx.evaluator()
            .push(grad_inv_trace)
            .feed(a, a_val.view())
            .run()[0]
            .clone();
        
        println!("\ngrad(trace(A^(-1)))/dA = \n{:?}", inv_trace_result);
    });
    
    println!();
}