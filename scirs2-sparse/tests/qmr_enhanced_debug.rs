use scirs2_sparse::error::SparseError;
use scirs2_sparse::linalg::{qmr, IdentityOperator, LinearOperator, QMROptions};

#[test]
fn test_qmr_identity_step_by_step() {
    // Test that QMR is not implemented yet, showing how it would fail on identity
    println!("=== QMR Step-by-Step Debug ===");

    // Test QMR on identity matrix
    let identity = IdentityOperator::<f64>::new(3);
    let b = vec![1.0, 2.0, 3.0];
    let x0 = vec![0.0, 0.0, 0.0];

    // Show how the algorithm would break down
    let r_initial = b.clone();
    println!("Initial residual: {:?}", r_initial);

    let r_star = r_initial.clone();
    println!("r_star: {:?}", r_star);

    let rho_initial: f64 = r_initial.iter().zip(&r_star).map(|(a, b)| a * b).sum();
    println!("Initial rho = dot(r, r_star) = {}", rho_initial);

    // Show the steps that would lead to breakdown
    let v_tilde_0: Vec<f64> = r_initial.iter().map(|&x| x / rho_initial).collect();
    println!("v_tilde_0 = r / rho = {:?}", v_tilde_0);

    let y_0: Vec<f64> = r_star.iter().map(|&x| x / rho_initial).collect();
    println!("y_0 = r_star / rho = {:?}", y_0);

    // This demonstrates the mathematical issue with the identity matrix
    println!("\n=== Running actual QMR function ===");
    let options = QMROptions {
        x0: Some(x0),
        max_iter: 1,
        rtol: 1e-8,
        atol: 1e-12,
        ..Default::default()
    };

    // Test that QMR is not implemented
    match qmr(&identity, &b, options) {
        Err(SparseError::NotImplemented(msg)) => {
            println!("As expected: {}", msg);
            assert!(msg.contains("QMR solver is not yet implemented"));
        }
        _ => panic!("Expected NotImplemented error"),
    }
}
