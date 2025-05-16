use scirs2_sparse::csr::CsrMatrix;
use scirs2_sparse::linalg::{expm_multiply, AsLinearOperator};

fn main() {
    // Create a diagonal matrix: [[1, 0], [0, -1]]
    let rows = vec![0, 1];
    let cols = vec![0, 1];
    let data = vec![1.0, -1.0];
    let matrix = CsrMatrix::new(data, rows, cols, (2, 2)).unwrap();

    // Debug: Print the matrix
    println!("Matrix contents:");
    for i in 0..2 {
        for j in 0..2 {
            print!("{:.1} ", matrix.get(i, j));
        }
        println!();
    }
    println!();

    // Use AsLinearOperator trait to convert to LinearOperator
    let op = matrix.as_linear_operator();

    // Vector to multiply
    let v = vec![1.0, 1.0];

    // Test operator directly
    let av = op.matvec(&v).unwrap();
    println!("A*v = {:?}", av);
    println!();

    // Compute exp(A) * v with different options
    println!("Using default m:");
    let result1 = expm_multiply(op.as_ref(), &v, 1.0, None, None).unwrap();
    println!("exp(A)*v = {:?}", result1);
    println!();

    println!("Using small m=3:");
    let result2 = expm_multiply(op.as_ref(), &v, 1.0, Some(3), None).unwrap();
    println!("exp(A)*v = {:?}", result2);
    println!();

    // Use DiagonalOperator directly
    use scirs2_sparse::linalg::DiagonalOperator;
    use scirs2_sparse::linalg::LinearOperator;
    let diag_op = DiagonalOperator::new(vec![1.0, -1.0]);
    println!("Using DiagonalOperator directly:");
    let result3 = expm_multiply(&diag_op, &v, 1.0, None, None).unwrap();
    println!("exp(A)*v = {:?}", result3);
    println!();

    // For diagonal matrix, exp(diag([a, b])) = diag([exp(a), exp(b)])
    // So exp(diag([1, -1])) = diag([e, 1/e])
    // And exp(A) * [1, 1] = [e, 1/e]
    let e = std::f64::consts::E;
    println!("Expected: [{}, {}]", e, 1.0 / e);
}
