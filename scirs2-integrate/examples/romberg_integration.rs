use ndarray::ArrayView1;
use scirs2_integrate::romberg::{multi_romberg, romberg, RombergOptions};
use std::f64::consts::PI;

fn main() {
    println!("Romberg Integration Examples\n");

    // Example 1: Integrate x^2 over [0, 1]
    // Exact result: 1/3
    println!("Example 1: Integrate x^2 from 0 to 1");
    println!("Exact value: {}", 1.0 / 3.0);

    let result = romberg(|x: f64| x * x, 0.0, 1.0, None).unwrap();
    println!("Romberg result: {:.15}", result.value);
    println!("Estimated error: {:.1e}", result.abs_error);
    println!("Number of iterations: {}", result.n_iters);
    println!("Converged: {}", result.converged);

    // Print the Romberg table
    println!("\nRomberg table:");
    for i in 0..result.table.shape()[0] {
        for j in 0..=i {
            print!("{:15.10}  ", result.table[[i, j]]);
        }
        println!();
    }
    println!();

    // Example 2: Integrate sin(x) over [0, π]
    // Exact result: 2
    println!("Example 2: Integrate sin(x) from 0 to π");
    println!("Exact value: {}", 2.0);

    let result = romberg(|x: f64| x.sin(), 0.0, PI, None).unwrap();
    println!("Romberg result: {:.15}", result.value);
    println!("Absolute error: {:.1e}", (result.value - 2.0).abs());
    println!("Estimated error: {:.1e}", result.abs_error);
    println!("Number of iterations: {}", result.n_iters);
    println!("Converged: {}", result.converged);
    println!();

    // Example 3: Integrating a function with a singularity using custom options
    // int_0^1 ln(x) dx = -1
    println!("Example 3: Integrate ln(x) from 0 to 1");
    println!("Exact value: {}", -1.0);

    // Use custom options to handle the challenging integral
    let options = RombergOptions {
        max_iters: 16,                  // More iterations for difficult integrals
        abs_tol: 1e-10,                 // Tighter absolute tolerance
        rel_tol: 1e-10,                 // Tighter relative tolerance
        max_true_dimension: 3,          // Default setting
        min_monte_carlo_samples: 10000, // Default setting
    };

    // Avoid the singularity at x=0 by starting from a small positive value
    let epsilon = 1e-10;
    let result = romberg(|x: f64| x.ln(), epsilon, 1.0, Some(options)).unwrap();

    println!("Romberg result: {:.15}", result.value);
    println!("Absolute error: {:.1e}", (result.value - (-1.0)).abs());
    println!("Estimated error: {:.1e}", result.abs_error);
    println!("Number of iterations: {}", result.n_iters);
    println!("Converged: {}", result.converged);
    println!();

    // Example 4: Multi-dimensional integration
    println!("Example 4: Multi-dimensional integration");

    // Integrate f(x,y) = cos(x+y) over [0,π/2]×[0,π/2]
    // Exact result: sin(π) - sin(0) - sin(π) + sin(0) = 0
    let result_2d = multi_romberg(
        |x: ArrayView1<f64>| (x[0] + x[1]).cos(),
        &[(0.0, PI / 2.0), (0.0, PI / 2.0)],
        None,
    )
    .unwrap();

    println!("∫∫cos(x+y) dxdy over [0,π/2]²:");
    println!("  Calculated: {:.15}", result_2d);
    println!("  Exact:      {:.15}", 0.0);
    println!("  Error:      {:.1e}", result_2d.abs());
    println!();

    // Example 5: Integrating a rapidly oscillating function
    println!("Example 5: Integrate cos(10x) from 0 to 2π");
    println!("Exact value: {}", 0.0); // Integral of cos(10x) over a full period is 0

    let result = romberg(|x: f64| (10.0 * x).cos(), 0.0, 2.0 * PI, None).unwrap();
    println!("Romberg result: {:.15}", result.value);
    println!("Absolute error: {:.1e}", result.value.abs());
    println!("Estimated error: {:.1e}", result.abs_error);
    println!("Number of iterations: {}", result.n_iters);
    println!("Converged: {}", result.converged);
}
