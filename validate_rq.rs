use ndarray::{array, Array2};
use scirs2_linalg::compat;

fn is_upper_triangular(matrix: &Array2<f64>, tol: f64) -> bool {
    let (nrows, ncols) = matrix.dim();
    for i in 0..nrows {
        for j in 0..ncols {
            if i > j && matrix[(i, j)].abs() > tol {
                return false;
            }
        }
    }
    true
}

fn main() {
    let test_matrices = vec![
        array![[1.0, 2.0], [3.0, 4.0]],
        array![[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]],
        array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], // Identity
        array![[5.0, 2.0, 1.0], [2.0, 3.0, 1.0], [1.0, 1.0, 2.0]], // Another test
    ];

    for (i, matrix) in test_matrices.iter().enumerate() {
        println!("Testing matrix {}:", i + 1);
        println!("Original matrix A:\n{:?}", matrix);

        let (r, q) = compat::rq(&matrix.view(), false, None, "full", true).unwrap();
        
        println!("R is upper triangular: {}", is_upper_triangular(&r, 1e-12));
        
        // Verify A = R * Q
        let reconstructed = r.dot(&q);
        let error_matrix = matrix - &reconstructed;
        let max_error = error_matrix.iter().map(|x| x.abs()).fold(0.0, f64::max);
        println!("Reconstruction error: {:.2e}", max_error);
        
        // Verify Q is orthogonal (Q^T * Q = I)
        let qtq = q.t().dot(&q);
        let identity: Array2<f64> = Array2::eye(q.ncols());
        let q_error = &qtq - &identity;
        let max_q_error = q_error.iter().map(|x| x.abs()).fold(0.0, f64::max);
        println!("Q orthogonality error: {:.2e}", max_q_error);
        
        println!("Test passed: {}", max_error < 1e-10 && max_q_error < 1e-10);
        println!("---");
    }
}