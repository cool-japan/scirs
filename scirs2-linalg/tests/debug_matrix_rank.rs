use ndarray::array;
use scirs2_linalg::{matrix_rank, svd};

#[test]
fn debug_nearly_singular_matrix_rank() {
    // Create a nearly singular matrix (rank should be 1)
    let matrix = array![
        [1.0, 2.0],
        [1.0, 2.0 + 1e-15] // Second row is almost identical to first
    ];

    println!("Matrix:");
    println!("{:?}", matrix);

    // Test with different tolerance values
    let default_rank = matrix_rank(&matrix.view(), None, None).unwrap();
    println!("Rank with default tolerance: {}", default_rank);

    let tight_tol_rank = matrix_rank(&matrix.view(), Some(1e-14), None).unwrap();
    println!("Rank with tolerance 1e-14: {}", tight_tol_rank);

    let loose_tol_rank = matrix_rank(&matrix.view(), Some(1e-12), None).unwrap();
    println!("Rank with tolerance 1e-12: {}", loose_tol_rank);

    // Let's also check what SVD gives us
    let (_, s, _) = svd(&matrix.view(), false, None).unwrap();
    println!("Singular values: {:?}", s);

    // Calculate the default tolerance manually
    let max_dim = std::cmp::max(matrix.nrows(), matrix.ncols());
    let eps = f64::EPSILON;
    let sigma_max = s[0];
    let default_tolerance = (max_dim as f64) * eps * sigma_max;
    println!("Default tolerance would be: {:.2e}", default_tolerance);
    println!("Machine epsilon: {:.2e}", eps);
    println!("Smallest singular value: {:.2e}", s[s.len() - 1]);

    // Check which singular values are above the default tolerance
    for (i, &sv) in s.iter().enumerate() {
        let above_threshold = sv > default_tolerance;
        println!(
            "Singular value {}: {:.2e} (above threshold: {})",
            i, sv, above_threshold
        );
    }

    // Let's test another nearly singular matrix example
    println!("\n--- Testing another nearly singular matrix ---");
    let matrix2 = array![[1.0, 1.0], [1.0, 1.0 + 1e-16]];

    println!("Matrix2:");
    println!("{:?}", matrix2);

    let rank2 = matrix_rank(&matrix2.view(), None, None).unwrap();
    println!("Rank with default tolerance: {}", rank2);

    let (_, s2, _) = svd(&matrix2.view(), false, None).unwrap();
    println!("Singular values: {:?}", s2);

    let default_tolerance2 = (max_dim as f64) * eps * s2[0];
    println!("Default tolerance would be: {:.2e}", default_tolerance2);
    println!("Smallest singular value: {:.2e}", s2[s2.len() - 1]);

    for (i, &sv) in s2.iter().enumerate() {
        let above_threshold = sv > default_tolerance2;
        println!(
            "Singular value {}: {:.2e} (above threshold: {})",
            i, sv, above_threshold
        );
    }

    // Test a matrix that might incorrectly return rank 2 instead of 1
    println!("\n--- Testing matrix that might give wrong rank ---");
    let matrix3 = array![
        [1.0, 2.0],
        [1.0 + 1e-13, 2.0 + 2e-13] // Small perturbation that might be above tolerance
    ];

    println!("Matrix3:");
    println!("{:?}", matrix3);

    let rank3 = matrix_rank(&matrix3.view(), None, None).unwrap();
    println!("Rank with default tolerance: {}", rank3);

    let (_, s3, _) = svd(&matrix3.view(), false, None).unwrap();
    println!("Singular values: {:?}", s3);

    let default_tolerance3 = (max_dim as f64) * eps * s3[0];
    println!("Default tolerance would be: {:.2e}", default_tolerance3);
    println!("Smallest singular value: {:.2e}", s3[s3.len() - 1]);

    for (i, &sv) in s3.iter().enumerate() {
        let above_threshold = sv > default_tolerance3;
        println!(
            "Singular value {}: {:.2e} (above threshold: {})",
            i, sv, above_threshold
        );
    }

    // Compare with a stricter tolerance
    let strict_rank3 = matrix_rank(&matrix3.view(), Some(1e-12), None).unwrap();
    println!("Rank with tolerance 1e-12: {}", strict_rank3);
}
