use ndarray::{array, ArrayView1};
use scirs2_integrate::bvp::solve_bvp;
use std::f64::consts::PI;

fn main() {
    println!("Boundary Value Problem Solver Example");
    println!("--------------------------------------");

    // Example 1: Solve the harmonic oscillator ODE: y'' + y = 0
    // as a first-order system: y0' = y1, y1' = -y0
    // with boundary conditions y0(0) = 0, y0(π) = 0
    println!("Example 1: Harmonic Oscillator");

    let fun = |_x: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

    let bc = |ya: ArrayView1<f64>, yb: ArrayView1<f64>| {
        // Boundary conditions: y0(0) = 0, y0(π) = 0
        array![ya[0], yb[0]]
    };

    // Initial mesh: 10 points from 0 to π
    let n_points = 10;
    let mut x = Vec::with_capacity(n_points);
    for i in 0..n_points {
        x.push(PI * (i as f64) / (n_points as f64 - 1.0));
    }

    // Initial guess: Simple sine function to help convergence
    let mut y_init = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let t = i as f64 / (n_points as f64 - 1.0);
        let y_val = (PI * t).sin() * 0.1; // Small amplitude sine function
        y_init.push(array![y_val, y_val * PI * (PI * t).cos()]); // y and y'
    }

    match solve_bvp(fun, bc, Some(x.clone()), y_init, None) {
        Ok(result) => {
            // The solution should be proportional to sin(x)
            // Scale the solution for comparison with sin(x)
            let idx_mid = result.x.len() / 2;
            let scale = if result.y[idx_mid][0].abs() > 1e-10 {
                result.y[idx_mid][0] / (PI / 2.0).sin()
            } else {
                1.0
            };

            println!("Solution for harmonic oscillator (should be proportional to sin(x)):");
            println!(
                "{:>10} {:>15} {:>15} {:>15}",
                "x", "y_numerical", "y_exact", "error"
            );
            println!("{:->10} {:->15} {:->15} {:->15}", "", "", "", "");

            for i in 0..result.x.len() {
                let x_val = result.x[i];
                let y_val = result.y[i][0];
                let sin_val = scale * x_val.sin();
                let error = (y_val - sin_val).abs();

                println!(
                    "{:10.6} {:15.6} {:15.6} {:15.6e}",
                    x_val, y_val, sin_val, error
                );
            }

            println!("");
            println!("Number of iterations: {}", result.n_iter);
            println!("Residual norm: {:.6e}", result.residual_norm);
            println!("Successful convergence: {}", result.success);

            // Example 2: Second Order ODE with Non-zero Boundary Conditions
            println!("\nExample 2: Second Order ODE with Non-zero Boundary Conditions");

            // Solve y'' = -y with boundary conditions y(0) = 1, y(π) = -1
            // This has solution y = cos(x)

            let fun2 = |_x: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

            let bc2 = |ya: ArrayView1<f64>, yb: ArrayView1<f64>| {
                // Boundary conditions: y0(0) = 1, y0(π) = -1
                array![ya[0] - 1.0, yb[0] + 1.0]
            };

            // Initial guess: cosine function to help convergence
            let mut y_init2 = Vec::with_capacity(n_points);
            for i in 0..n_points {
                let t = i as f64 / (n_points as f64 - 1.0);
                let x_val = PI * t;
                let y_val = x_val.cos(); // Exact solution as initial guess
                y_init2.push(array![y_val, -x_val.sin()]); // y and y'
            }

            match solve_bvp(fun2, bc2, Some(x), y_init2, None) {
                Ok(result2) => {
                    println!("Solution for y'' = -y with y(0) = 1, y(π) = -1:");
                    println!(
                        "{:>10} {:>15} {:>15} {:>15}",
                        "x", "y_numerical", "y_exact", "error"
                    );
                    println!("{:->10} {:->15} {:->15} {:->15}", "", "", "", "");

                    for i in 0..result2.x.len() {
                        let x_val = result2.x[i];
                        let y_val = result2.y[i][0];
                        let cos_val = x_val.cos();
                        let error = (y_val - cos_val).abs();

                        println!(
                            "{:10.6} {:15.6} {:15.6} {:15.6e}",
                            x_val, y_val, cos_val, error
                        );
                    }

                    println!("");
                    println!("Number of iterations: {}", result2.n_iter);
                    println!("Residual norm: {:.6e}", result2.residual_norm);
                    println!("Successful convergence: {}", result2.success);
                }
                Err(e) => {
                    println!("Error solving second BVP: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Error solving first BVP: {}", e);
        }
    }
}
