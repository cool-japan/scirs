use scirs2_sparse::linalg::{cgs, CGSOptions, DiagonalOperator};

fn main() {
    // Simple diagonal matrix test
    let diag = DiagonalOperator::new(vec![2.0, 3.0, 4.0]);
    let b = vec![2.0, 6.0, 12.0];

    let mut options = CGSOptions::default();
    options.atol = 1e-12;
    options.rtol = 1e-8;
    options.max_iter = 10;

    let result = cgs(&diag, &b, options).unwrap();

    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("Residual norm: {}", result.residual_norm);
    println!("Solution: {:?}", result.x);
    println!("Expected: [1.0, 2.0, 3.0]");

    // Check residual
    let mut residual = vec![0.0; 3];
    for i in 0..3 {
        residual[i] = b[i] - diag.diagonal()[i] * result.x[i];
    }
    println!("Actual residual: {:?}", residual);

    // Show iteration steps
    println!("\nIteration trace:");
    let mut x = vec![0.0; 3];
    let mut r = b.clone();
    let r_tilde = r.clone();

    let mut p = vec![0.0; 3];
    let mut u = vec![0.0; 3];
    let mut q = vec![0.0; 3];
    let mut rho_old = 1.0;

    for iter in 0..10 {
        println!("\n--- Iteration {} ---", iter);

        // Compute rho = (r_tilde, r)
        let rho: f64 = r_tilde.iter().zip(&r).map(|(a, b)| a * b).sum();
        println!("rho = {}", rho);

        let beta = if iter == 0 { 0.0 } else { rho / rho_old };
        println!("beta = {}", beta);

        // Update u = r + beta * q
        for i in 0..3 {
            u[i] = r[i] + beta * q[i];
        }
        println!("u = {:?}", u);

        // Update p = u + beta * (q + beta * p)
        for i in 0..3 {
            p[i] = u[i] + beta * (q[i] + beta * p[i]);
        }
        println!("p = {:?}", p);

        // Compute v = A * p
        let mut v = vec![0.0; 3];
        for i in 0..3 {
            v[i] = diag.diagonal()[i] * p[i];
        }
        println!("v = A*p = {:?}", v);

        // Compute alpha = rho / (r_tilde, v)
        let sigma: f64 = r_tilde.iter().zip(&v).map(|(a, b)| a * b).sum();
        println!("sigma = (r_tilde, v) = {}", sigma);

        if sigma.abs() < 1e-10 {
            println!("Breakdown: sigma ≈ 0");
            break;
        }

        let alpha = rho / sigma;
        println!("alpha = {}", alpha);

        // Update q = u - alpha * v
        for i in 0..3 {
            q[i] = u[i] - alpha * v[i];
        }
        println!("q = {:?}", q);

        // Update solution x = x + alpha * (u + q)
        println!("Before update: x = {:?}", x);
        for i in 0..3 {
            x[i] = x[i] + alpha * (u[i] + q[i]);
        }
        println!("After update: x = {:?}", x);

        // Compute w = A * q
        let mut w = vec![0.0; 3];
        for i in 0..3 {
            w[i] = diag.diagonal()[i] * q[i];
        }
        println!("w = A*q = {:?}", w);

        // Update residual r = r - alpha * (v + w)
        println!("Before residual update: r = {:?}", r);
        for i in 0..3 {
            r[i] = r[i] - alpha * (v[i] + w[i]);
        }
        println!("After residual update: r = {:?}", r);

        let rnorm: f64 = r.iter().map(|&x| x * x).sum::<f64>().sqrt();
        println!("||r|| = {}", rnorm);

        if rnorm < 1e-10 {
            println!("Converged!");
            break;
        }

        rho_old = rho;
    }

    println!("\nFinal solution: {:?}", x);
}
