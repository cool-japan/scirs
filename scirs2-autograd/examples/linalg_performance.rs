use ag::tensor_ops::*;
use ndarray::Array2;
use scirs2_autograd as ag;
use std::time::Instant;

macro_rules! benchmark {
    ($name:expr, $code:block) => {{
        let start = Instant::now();
        let result = $code;
        let duration = start.elapsed();
        println!("{}: {:?}", $name, duration);
        result
    }};
}

fn main() {
    println!("=== scirs2-autograd Linear Algebra Performance Benchmark ===\n");

    ag::run(|g| {
        // Create test matrices of various sizes
        let sizes = vec![10, 50, 100, 200];

        for &n in &sizes {
            println!("\n--- Matrix size: {}x{} ---", n, n);

            // Generate random positive definite matrix
            let random_data: Vec<f64> = (0..n * n).map(|_| rand::random::<f64>()).collect();
            let a_raw = Array2::from_shape_vec((n, n), random_data).unwrap();
            let a_sym = &a_raw + &a_raw.t();
            let a_pd = &a_sym + Array2::eye(n) * (n as f64);

            let a = convert_to_tensor(a_pd.clone(), g);

            // Benchmark basic operations
            benchmark!("Identity matrix", {
                let _ = eye(n, g).eval(g).unwrap();
            });

            benchmark!("Matrix trace", {
                let _ = trace(a).eval(g).unwrap();
            });

            benchmark!("Determinant", {
                let _ = determinant(a).eval(g).unwrap();
            });

            benchmark!("Frobenius norm", {
                let _ = frobenius_norm(a).eval(g).unwrap();
            });

            // Benchmark decompositions
            benchmark!("QR decomposition", {
                let (q, r) = qr(a);
                let _ = q.eval(g).unwrap();
                let _ = r.eval(g).unwrap();
            });

            if n <= 100 {
                // SVD is expensive for large matrices
                benchmark!("SVD", {
                    let (u, s, v) = svd(a);
                    let _ = u.eval(g).unwrap();
                    let _ = s.eval(g).unwrap();
                    let _ = v.eval(g).unwrap();
                });
            }

            benchmark!("Cholesky decomposition", {
                let _ = cholesky(&a).eval(g).unwrap();
            });

            if n <= 100 {
                // Eigendecomposition is expensive
                benchmark!("Eigendecomposition", {
                    let (vals, vecs) = eigen(a);
                    let _ = vals.eval(g).unwrap();
                    let _ = vecs.eval(g).unwrap();
                });
            }

            // Benchmark matrix operations
            benchmark!("Matrix inverse", {
                let _ = matrix_inverse(a).eval(g).unwrap();
            });

            // Benchmark linear solver
            let b = convert_to_tensor(Array2::ones((n, 1)), g);
            benchmark!("Linear solve (Ax=b)", {
                let _ = solve(a, b).eval(g).unwrap();
            });

            // Benchmark with gradients
            if n <= 50 {
                // Gradient computation is memory intensive
                let a_var = variable(a_pd.clone(), g);
                benchmark!("Determinant with gradient", {
                    let det = determinant(a_var);
                    let grads = grad(&[&det], &[&a_var]);
                    let _ = grads[0].eval(g).unwrap();
                });

                benchmark!("Solve with gradient", {
                    let x = solve(a_var, b);
                    let loss = sum_all(square(x));
                    let grads = grad(&[&loss], &[&a_var]);
                    let _ = grads[0].eval(g).unwrap();
                });
            }
        }

        println!("\n--- Memory-efficient operations ---");

        // Test large matrix operations
        let large_n = 500;
        println!("\nLarge matrix ({}x{})", large_n, large_n);

        let large_data: Vec<f64> = (0..large_n * large_n)
            .map(|i| (i as f64) / (large_n * large_n) as f64)
            .collect();
        let large_matrix = Array2::from_shape_vec((large_n, large_n), large_data).unwrap();
        let large_a = convert_to_tensor(large_matrix, g);

        benchmark!("Large matrix trace", {
            let _ = trace(large_a).eval(g).unwrap();
        });

        benchmark!("Large matrix norm", {
            let _ = frobenius_norm(large_a).eval(g).unwrap();
        });

        // Demonstrate operation chaining
        println!("\n--- Operation chaining performance ---");

        let n = 50;
        let chain_data = Array2::eye(n) * 2.0;
        let chain_a = convert_to_tensor(chain_data, g);

        benchmark!("Complex operation chain", {
            let result = matrix_exp(&matrix_inverse(chain_a)); // Changed from matrix_sqrt (not implemented)
            let norm = frobenius_norm(result);
            let det = determinant(result);
            let combined = add(norm, det);
            let _ = combined.eval(g).unwrap();
        });

        println!("\n=== Benchmark completed ===");
    });
}
