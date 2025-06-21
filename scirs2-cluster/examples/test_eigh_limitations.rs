// No imports needed for this documentation example

fn main() {
    println!("Testing eigh function limitations for spectral clustering");
    println!("Based on code analysis of scirs2-linalg/src/eigen.rs");

    println!("\n=== Matrix Size Support Summary ===");
    println!("1x1 matrices: ✓ SUPPORTED (direct analytical solution)");
    println!("2x2 matrices: ✓ SUPPORTED (quadratic formula)");
    println!("3x3 matrices: ✓ SUPPORTED (cubic formula with high precision)");
    println!("4x4+ matrices: ✗ NOT SUPPORTED (returns NotImplementedError)");

    println!("\n=== Key Findings ===");
    println!("1. The eigh function in scirs2-linalg is limited to matrices up to 3x3");
    println!("2. For matrices 4x4 and larger, it calls solve_with_power_iteration()");
    println!("3. solve_with_power_iteration() explicitly returns NotImplementedError:");
    println!("   'Eigenvalue decomposition for NxN matrices not fully implemented yet'");

    println!("\n=== Impact on Spectral Clustering ===");
    println!("- Spectral clustering typically needs eigendecomposition of:");
    println!("  * Laplacian matrices (size n_samples x n_samples)");
    println!("  * For real datasets, this is often 100x100, 1000x1000, or larger");
    println!("- Current limitation means spectral clustering will fail for:");
    println!("  * Any dataset with more than 3 samples");
    println!("  * This explains why tests with 4+ samples are failing");

    println!("\n=== Recommended Solutions ===");
    println!("1. Use specialized sparse eigenvalue solvers for larger matrices");
    println!("2. Consider using functions like largest_k_eigh/smallest_k_eigh");
    println!("3. Implement iterative methods like Lanczos algorithm");
    println!("4. For spectral clustering, only need k smallest non-zero eigenvalues");

    println!("\n=== Code Reference ===");
    println!("File: scirs2-linalg/src/eigen.rs");
    println!("Function: eigh() line 391");
    println!("Limitation: solve_with_power_iteration() line 1206-1214");
}
