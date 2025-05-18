//! Examples of scalar optimization using various methods

extern crate openblas_src;
use scirs2_optimize::scalar::{minimize_scalar, Method, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Simple quadratic function
    println!("Example 1: Minimize (x - 2)^2");
    let f1 = |x: f64| (x - 2.0).powi(2);

    // Using Brent method
    let result = minimize_scalar(f1, None, Method::Brent, None)?;
    println!("Brent method:");
    println!("  Minimum at x = {:.6}", result.x);
    println!("  Function value = {:.6}", result.fun);
    println!("  Iterations: {}", result.iterations);
    println!();

    // Example 2: Function with multiple local minima
    println!("Example 2: Minimize (x - 2) * x * (x + 2)^2");
    let f2 = |x: f64| (x - 2.0) * x * (x + 2.0).powi(2);

    // Using Brent method
    let result = minimize_scalar(f2, None, Method::Brent, None)?;
    println!("Brent method:");
    println!("  Minimum at x = {:.6}", result.x);
    println!("  Function value = {:.6}", result.fun);
    println!();

    // Example 3: Constrained optimization
    println!("Example 3: Minimize (x - 2)^2 within bounds [-1, 1]");
    let f3 = |x: f64| (x - 2.0).powi(2);

    // Using bounded method
    let result = minimize_scalar(f3, Some((-1.0, 1.0)), Method::Bounded, None)?;
    println!("Bounded method:");
    println!("  Minimum at x = {:.6}", result.x);
    println!("  Function value = {:.6}", result.fun);
    println!();

    // Example 4: Using custom options
    println!("Example 4: Minimize x^4 - 2x^2 + x with custom tolerance");
    let f4 = |x: f64| x.powi(4) - 2.0 * x.powi(2) + x;

    let mut options = Options::default();
    options.xatol = 1e-8;
    options.max_iter = 100;

    // Using Golden section search
    let result = minimize_scalar(f4, None, Method::Golden, Some(options))?;
    println!("Golden section method:");
    println!("  Minimum at x = {:.6}", result.x);
    println!("  Function value = {:.6}", result.fun);
    println!("  Iterations: {}", result.iterations);
    println!();

    // Example 5: Rosenbrock function in 1D
    println!("Example 5: Minimize 100*(1 - x)^2");
    let f5 = |x: f64| 100.0 * (1.0 - x).powi(2);

    let result = minimize_scalar(f5, None, Method::Brent, None)?;
    println!("Brent method:");
    println!("  Minimum at x = {:.6}", result.x);
    println!("  Function value = {:.6}", result.fun);
    println!();

    // Example 6: Trigonometric function
    println!("Example 6: Minimize sin(x) + 0.1*x^2 within [0, 2π]");
    let f6 = |x: f64| x.sin() + 0.1 * x.powi(2);

    let result = minimize_scalar(
        f6,
        Some((0.0, 2.0 * std::f64::consts::PI)),
        Method::Bounded,
        None,
    )?;
    println!("Bounded method:");
    println!("  Minimum at x = {:.6}", result.x);
    println!("  Function value = {:.6}", result.fun);
    println!();

    Ok(())
}
