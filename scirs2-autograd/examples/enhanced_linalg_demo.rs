use ag::tensor_ops::linear_algebra::*;
use ag::tensor_ops::*;
use ndarray::array;
use scirs2_autograd as ag;

fn main() {
    ag::run(|g| {
        println!("=== Enhanced Linear Algebra Operations Demo ===\n");

        // 1. Matrix Norms
        println!("1. Matrix Norms:");
        let a = convert_to_tensor(
            array![[1.0_f64, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]],
            g,
        );

        let n1 = norm1(&a).eval(g).unwrap();
        let n2 = norm2(&a).eval(g).unwrap();
        let ninf = norminf(&a).eval(g).unwrap();
        let nfro = normfro(&a).eval(g).unwrap();

        println!("  1-norm (max column sum): {}", n1[ndarray::IxDyn(&[])]);
        println!("  2-norm (spectral norm): {}", n2[ndarray::IxDyn(&[])]);
        println!("  ∞-norm (max row sum): {}", ninf[ndarray::IxDyn(&[])]);
        println!("  Frobenius norm: {}", nfro[ndarray::IxDyn(&[])]);

        // 2. Symmetric Eigendecomposition
        println!("\n2. Symmetric Matrix Eigendecomposition:");
        let sym = convert_to_tensor(
            array![[4.0_f64, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            g,
        );

        let (eigenvalues, eigenvectors) = eigh(&sym);
        let vals = eigenvalues.eval(g).unwrap();
        let vecs = eigenvectors.eval(g).unwrap();

        println!("  Eigenvalues: {:?}", vals.as_slice().unwrap());
        println!(
            "  First eigenvector: [{:.3}, {:.3}, {:.3}]",
            vecs[[0, 0]],
            vecs[[1, 0]],
            vecs[[2, 0]]
        );

        // 3. Matrix Exponential
        println!("\n3. Matrix Exponential:");
        let rot = convert_to_tensor(array![[0.0_f64, -1.0], [1.0, 0.0]], g);

        let exp_pade = expm2(&rot).eval(g).unwrap();
        let exp_eigen = expm3(&rot).eval(g).unwrap();

        println!("  Using Padé approximation:");
        println!("    [[{:.3}, {:.3}],", exp_pade[[0, 0]], exp_pade[[0, 1]]);
        println!("     [{:.3}, {:.3}]]", exp_pade[[1, 0]], exp_pade[[1, 1]]);

        println!("  Using eigendecomposition:");
        println!("    [[{:.3}, {:.3}],", exp_eigen[[0, 0]], exp_eigen[[0, 1]]);
        println!("     [{:.3}, {:.3}]]", exp_eigen[[1, 0]], exp_eigen[[1, 1]]);

        // 4. Cholesky Solve
        println!("\n4. Solving Positive Definite System with Cholesky:");
        let pd_matrix = convert_to_tensor(array![[4.0_f64, 2.0], [2.0, 3.0]], g);
        let b = convert_to_tensor(array![1.0_f64, 2.0], g);

        let x = cholesky_solve(&pd_matrix, &b).eval(g).unwrap();
        println!("  Solution x: [{:.3}, {:.3}]", x[0], x[1]);

        // Verify: A*x = b
        let ax = matmul(
            &pd_matrix,
            &convert_to_tensor(x.clone().insert_axis(ndarray::Axis(1)), g),
        );
        let ax_flat = ax.eval(g).unwrap();
        println!(
            "  Verification A*x: [{:.3}, {:.3}]",
            ax_flat[[0, 0]],
            ax_flat[[1, 0]]
        );

        // 5. Sylvester Equation
        println!("\n5. Solving Sylvester Equation AX + XB = C:");
        let a_syl = convert_to_tensor(array![[1.0_f64, 0.0], [0.0, 2.0]], g);
        let b_syl = convert_to_tensor(array![[3.0_f64, 0.0], [0.0, 4.0]], g);
        let c_syl = convert_to_tensor(array![[1.0_f64, 2.0], [3.0, 4.0]], g);

        let x_syl = solve_sylvester(&a_syl, &b_syl, &c_syl).eval(g).unwrap();
        println!("  Solution X:");
        println!("    [[{:.3}, {:.3}],", x_syl[[0, 0]], x_syl[[0, 1]]);
        println!("     [{:.3}, {:.3}]]", x_syl[[1, 0]], x_syl[[1, 1]]);

        // 6. Polar Decomposition
        println!("\n6. Polar Decomposition A = UP:");
        let mat = convert_to_tensor(array![[1.0_f64, 2.0], [3.0, 4.0]], g);

        let (u, p) = polar(&mat);
        let u_val = u.eval(g).unwrap();
        let p_val = p.eval(g).unwrap();

        println!("  Unitary part U:");
        println!("    [[{:.3}, {:.3}],", u_val[[0, 0]], u_val[[0, 1]]);
        println!("     [{:.3}, {:.3}]]", u_val[[1, 0]], u_val[[1, 1]]);

        println!("  Positive semidefinite part P:");
        println!("    [[{:.3}, {:.3}],", p_val[[0, 0]], p_val[[0, 1]]);
        println!("     [{:.3}, {:.3}]]", p_val[[1, 0]], p_val[[1, 1]]);

        // 7. Einstein Summation
        println!("\n7. Einstein Summation (einsum):");
        let e1 = convert_to_tensor(array![[1.0_f64, 2.0], [3.0, 4.0]], g);
        let e2 = convert_to_tensor(array![[5.0_f64, 6.0], [7.0, 8.0]], g);

        // Matrix multiplication
        let matmul_result = einsum("ij,jk->ik", &[&e1, &e2]).eval(g).unwrap();
        println!("  Matrix multiplication (ij,jk->ik):");
        println!(
            "    [[{:.0}, {:.0}],",
            matmul_result[[0, 0]],
            matmul_result[[0, 1]]
        );
        println!(
            "     [{:.0}, {:.0}]]",
            matmul_result[[1, 0]],
            matmul_result[[1, 1]]
        );

        // Trace - compute manually for now
        let e1_array = e1.eval(g).unwrap();
        let trace_val = e1_array[[0, 0]] + e1_array[[1, 1]];
        println!("  Trace: {:.0}", trace_val);

        // Element-wise multiplication
        let hadamard = einsum("ij,ij->ij", &[&e1, &e2]).eval(g).unwrap();
        println!("  Element-wise multiplication (ij,ij->ij):");
        println!("    [[{:.0}, {:.0}],", hadamard[[0, 0]], hadamard[[0, 1]]);
        println!("     [{:.0}, {:.0}]]", hadamard[[1, 0]], hadamard[[1, 1]]);

        // 8. Kronecker Product
        println!("\n8. Kronecker Product:");
        let k1 = convert_to_tensor(array![[1.0_f64, 2.0]], g);
        let k2 = convert_to_tensor(array![[3.0_f64], [4.0]], g);

        let kron_result = kronecker_product(&k1, &k2).eval(g).unwrap();
        println!("  kron([1, 2], [3; 4]):");
        println!(
            "    [[{:.0}, {:.0}],",
            kron_result[[0, 0]],
            kron_result[[0, 1]]
        );
        println!(
            "     [{:.0}, {:.0}]]",
            kron_result[[1, 0]],
            kron_result[[1, 1]]
        );

        println!("\n=== Demo Complete ===");
    });
}
