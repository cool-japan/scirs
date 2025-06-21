use approx::assert_relative_eq;
use ndarray::array;
use scirs2_linalg::{largest_k_eigh, smallest_k_eigh};

#[test]
fn test_largest_k_eigh_diagonal() {
    // Simple diagonal matrix
    let a = array![
        [5.0_f64, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 3.0]
    ];

    // Get 2 largest eigenvalues
    let (eigenvalues, eigenvectors) = largest_k_eigh(&a.view(), 2, 100, 1e-10).unwrap();

    // Eigenvalues should be 5.0 and 3.0
    assert_relative_eq!(eigenvalues[0], 5.0, epsilon = 1e-10);
    assert_relative_eq!(eigenvalues[1], 3.0, epsilon = 1e-10);

    // Verify eigenvectors
    for i in 0..2 {
        let v = eigenvectors.column(i).to_owned();
        let av = a.dot(&v);
        let lambda_v = v * eigenvalues[i];

        for j in 0..4 {
            assert_relative_eq!(av[j], lambda_v[j], epsilon = 1e-4, max_relative = 1e-3);
        }
    }
}

#[test]
fn test_smallest_k_eigh_diagonal() {
    // Simple diagonal matrix
    let a = array![
        [5.0_f64, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 3.0]
    ];

    // Get 2 smallest eigenvalues
    let (eigenvalues, eigenvectors) = smallest_k_eigh(&a.view(), 2, 100, 1e-10).unwrap();

    // Eigenvalues should be 1.0 and 2.0
    assert_relative_eq!(eigenvalues[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(eigenvalues[1], 2.0, epsilon = 1e-10);

    // Verify eigenvectors
    for i in 0..2 {
        let v = eigenvectors.column(i).to_owned();
        let av = a.dot(&v);
        let lambda_v = v * eigenvalues[i];

        for j in 0..4 {
            assert_relative_eq!(av[j], lambda_v[j], epsilon = 1e-4, max_relative = 1e-3);
        }
    }
}

#[test]
fn test_largest_k_eigh_symmetric() {
    // Symmetric matrix with known eigenvalues
    let a = array![
        [1.0_f64, 0.5, 0.0, 0.0],
        [0.5, 2.0, 0.5, 0.0],
        [0.0, 0.5, 3.0, 0.5],
        [0.0, 0.0, 0.5, 4.0]
    ];

    // Get 2 largest eigenvalues
    let (eigenvalues, eigenvectors) = largest_k_eigh(&a.view(), 2, 100, 1e-10).unwrap();

    // Verify eigenvectors satisfy Av = λv
    for i in 0..2 {
        let v = eigenvectors.column(i).to_owned();
        let av = a.dot(&v);
        let lambda_v = v * eigenvalues[i];

        for j in 0..4 {
            assert_relative_eq!(av[j], lambda_v[j], epsilon = 1e-4, max_relative = 1e-3);
        }
    }

    // The largest eigenvalue should be greater than 4.0 due to off-diagonal elements
    assert!(eigenvalues[0] > 4.0);
    // The second largest eigenvalue should be between 3.0 and 4.0
    assert!(eigenvalues[1] > 3.0 && eigenvalues[1] < 4.0);
}

#[test]
fn test_smallest_k_eigh_symmetric() {
    // Symmetric matrix with known eigenvalues
    let a = array![
        [1.0_f64, 0.5, 0.0, 0.0],
        [0.5, 2.0, 0.5, 0.0],
        [0.0, 0.5, 3.0, 0.5],
        [0.0, 0.0, 0.5, 4.0]
    ];

    // Get 2 smallest eigenvalues
    let (eigenvalues, eigenvectors) = smallest_k_eigh(&a.view(), 2, 100, 1e-10).unwrap();

    // Verify eigenvectors satisfy Av = λv
    for i in 0..2 {
        let v = eigenvectors.column(i).to_owned();
        let av = a.dot(&v);
        let lambda_v = v * eigenvalues[i];

        for j in 0..4 {
            assert_relative_eq!(av[j], lambda_v[j], epsilon = 1e-4, max_relative = 1e-3);
        }
    }

    // The smallest eigenvalue should be less than 1.0 due to off-diagonal elements
    assert!(eigenvalues[0] < 1.0);
    // The second smallest eigenvalue should be between 1.0 and 2.0
    assert!(eigenvalues[1] > 1.0 && eigenvalues[1] < 2.0);
}

#[test]
fn test_k_equal_zero() {
    let a = array![[1.0_f64, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];

    // Request 0 eigenvalues
    let (eigenvalues, eigenvectors) = largest_k_eigh(&a.view(), 0, 100, 1e-10).unwrap();

    assert_eq!(eigenvalues.len(), 0);
    assert_eq!(eigenvectors.shape(), &[3, 0]);

    let (eigenvalues, eigenvectors) = smallest_k_eigh(&a.view(), 0, 100, 1e-10).unwrap();

    assert_eq!(eigenvalues.len(), 0);
    assert_eq!(eigenvectors.shape(), &[3, 0]);
}

#[test]
fn test_k_equal_n() {
    let a = array![
        [1.0_f64, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 4.0]
    ];

    // Request all eigenvalues
    let (eigenvalues, eigenvectors) = largest_k_eigh(&a.view(), 4, 100, 1e-10).unwrap();

    assert_eq!(eigenvalues.len(), 4);
    assert_eq!(eigenvectors.shape(), &[4, 4]);

    // Sort in descending order for verification
    assert_relative_eq!(eigenvalues[0], 4.0, epsilon = 1e-10);
    assert_relative_eq!(eigenvalues[1], 3.0, epsilon = 1e-10);
    assert_relative_eq!(eigenvalues[2], 2.0, epsilon = 1e-10);
    assert_relative_eq!(eigenvalues[3], 1.0, epsilon = 1e-10);

    let (eigenvalues, eigenvectors) = smallest_k_eigh(&a.view(), 4, 100, 1e-10).unwrap();

    assert_eq!(eigenvalues.len(), 4);
    assert_eq!(eigenvectors.shape(), &[4, 4]);

    // Should be in ascending order
    assert_relative_eq!(eigenvalues[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(eigenvalues[1], 2.0, epsilon = 1e-10);
    assert_relative_eq!(eigenvalues[2], 3.0, epsilon = 1e-10);
    assert_relative_eq!(eigenvalues[3], 4.0, epsilon = 1e-10);
}

#[test]
fn test_invalid_input() {
    // Non-square matrix
    let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let result = largest_k_eigh(&a.view(), 1, 100, 1e-10);
    assert!(result.is_err());

    let result = smallest_k_eigh(&a.view(), 1, 100, 1e-10);
    assert!(result.is_err());

    // Non-symmetric matrix
    let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

    let result = largest_k_eigh(&a.view(), 1, 100, 1e-10);
    assert!(result.is_err());

    let result = smallest_k_eigh(&a.view(), 1, 100, 1e-10);
    assert!(result.is_err());

    // k too large
    let a = array![[1.0_f64, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];

    let result = largest_k_eigh(&a.view(), 4, 100, 1e-10);
    assert!(result.is_err());

    let result = smallest_k_eigh(&a.view(), 4, 100, 1e-10);
    assert!(result.is_err());
}
