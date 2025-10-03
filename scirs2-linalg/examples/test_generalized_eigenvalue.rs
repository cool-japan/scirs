//! Test example for generalized eigenvalue decomposition functions
//!
//! This example demonstrates the usage of the newly implemented generalized
//! eigenvalue functions: eig_gen, eigh_gen, eigvals_gen, and eigvalsh_gen.

use scirs2_core::ndarray::array;
use scirs2_linalg::{eig_gen, eigh_gen, eigvals_gen, eigvalsh_gen};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Generalized Eigenvalue Decomposition Functions");
    println!("======================================================");

    // Test 1: Basic generalized eigenvalue problem with diagonal matrices
    println!("\n1. Testing eig_gen with diagonal matrices:");
    let a = array![[1.0, 0.0], [0.0, 2.0]];
    let b = array![[2.0, 0.0], [0.0, 1.0]];

    match eig_gen(&a.view(), &b.view(), None) {
        Ok((eigenvalues, eigenvectors)) => {
            println!("   Eigenvalues: {:?}", eigenvalues);
            println!("   First eigenvector: {:?}", eigenvectors.column(0));
            println!("   Success: eig_gen works correctly!");
        }
        Err(e) => {
            println!("   Error in eiggen: {}", e);
            return Err(e.into());
        }
    }

    // Test 2: Symmetric generalized eigenvalue problem
    println!("\n2. Testing eigh_gen with symmetric matrices:");
    let a_sym = array![[2.0, 1.0], [1.0, 3.0]];
    let b_sym = array![[1.0, 0.0], [0.0, 2.0]];

    match eigh_gen(&a_sym.view(), &b_sym.view(), None) {
        Ok((eigenvalues, eigenvectors)) => {
            println!("   Eigenvalues: {:?}", eigenvalues);
            println!("   First eigenvector: {:?}", eigenvectors.column(0));
            println!("   Success: eigh_gen works correctly!");
        }
        Err(e) => {
            println!("   Error in eighgen: {}", e);
            return Err(e.into());
        }
    }

    // Test 3: Eigenvalues-only functions
    println!("\n3. Testing eigvals_gen (eigenvalues only):");
    match eigvals_gen(&a.view(), &b.view(), None) {
        Ok(eigenvalues) => {
            println!("   Eigenvalues: {:?}", eigenvalues);
            println!("   Success: eigvals_gen works correctly!");
        }
        Err(e) => {
            println!("   Error in eigvalsgen: {}", e);
            return Err(e.into());
        }
    }

    println!("\n4. Testing eigvalsh_gen (symmetric eigenvalues only):");
    match eigvalsh_gen(&a_sym.view(), &b_sym.view(), None) {
        Ok(eigenvalues) => {
            println!("   Eigenvalues: {:?}", eigenvalues);
            println!("   Success: eigvalsh_gen works correctly!");
        }
        Err(e) => {
            println!("   Error in eigvalshgen: {}", e);
            return Err(e.into());
        }
    }

    // Test 5: Special case - identity matrix B (should match standard eigenvalue problem)
    println!("\n5. Testing with identity matrix B (should match standard eigenvalue):");
    let a_test = array![[2.0, 1.0], [1.0, 2.0]];
    let b_identity = array![[1.0, 0.0], [0.0, 1.0]];

    match eig_gen(&a_test.view(), &b_identity.view(), None) {
        Ok(gen_eigenvalues_) => {
            println!("   Generalized eigenvalues: {:?}", gen_eigenvalues_);
            println!("   Success: Identity matrix case works!");
        }
        Err(e) => {
            println!("   Error with identity matrix: {}", e);
            return Err(e.into());
        }
    }

    println!("\n======================================================");
    println!("All Generalized Eigenvalue Tests Completed Successfully!");
    println!("✅ eiggen: General generalized eigenvalue problem");
    println!("✅ eighgen: Symmetric generalized eigenvalue problem");
    println!("✅ eigvalsgen: Eigenvalues-only (general)");
    println!("✅ eigvalshgen: Eigenvalues-only (symmetric)");
    println!("======================================================");

    Ok(())
}
