use scirs2_core::ndarray::{array, Array1, ArrayView1};
use scirs2_linalg::{
    block_diagonal_operator, jacobi_preconditioner, matrix_free_conjugate_gradient,
    matrix_free_gmres, matrix_free_preconditioned_conjugate_gradient, LinearOperator, MatrixFreeOp,
};

#[allow(dead_code)]
fn main() {
    // Example 1: Using LinearOperator to represent a matrix without storing it
    println!("Example 1: Matrix-free operator representation");

    // Create a linear operator representing a symmetric positive definite matrix
    // [4.0, 1.0]
    // [1.0, 3.0]
    let spd_op = LinearOperator::new(2, |v: &ArrayView1<f64>| {
        let mut result = Array1::zeros(2);
        result[0] = 4.0 * v[0] + 1.0 * v[1];
        result[1] = 1.0 * v[0] + 3.0 * v[1];
        result
    })
    .symmetric()
    .positive_definite();

    // Create a vector and apply the operator
    let x = array![1.0, 2.0];
    let y = spd_op.apply(&x.view()).unwrap();

    println!("Matrix-vector product:");
    println!("x = [{}, {}]", x[0], x[1]);
    println!("Ax = [{}, {}]", y[0], y[1]);
    println!();

    // Example 2: Using matrix-free conjugate gradient to solve a linear system
    println!("Example 2: Matrix-free conjugate gradient");

    // Define the right-hand side
    let b = array![5.0, 7.0];

    // Solve using matrix-free conjugate gradient
    let solution = matrix_free_conjugate_gradient(&spd_op, &b, 10, 1e-10).unwrap();

    println!("Solving Ax = b with conjugate gradient:");
    println!("b = [{}, {}]", b[0], b[1]);
    println!("x = [{:.6}, {:.6}]", solution[0], solution[1]);

    // Verify solution
    let ax = spd_op.apply(&solution.view()).unwrap();
    println!("Ax = [{:.6}, {:.6}]", ax[0], ax[1]);
    println!("Error: {:.6e}", vector_norm(&b, &ax));
    println!();

    // Example 3: GMRES for non-symmetric systems
    println!("Example 3: Matrix-free GMRES for non-symmetric systems");

    // Create a non-symmetric operator
    let nonsym_op = LinearOperator::new(2, |v: &ArrayView1<f64>| {
        let mut result = Array1::zeros(2);
        result[0] = 3.0 * v[0] + 1.0 * v[1];
        result[1] = 2.0 * v[0] + 4.0 * v[1];
        result
    });

    // Define a right-hand side vector
    let b2 = array![4.0, 10.0];

    // Solve using GMRES
    let solution2 = matrix_free_gmres(&nonsym_op, &b2, 10, 1e-10, None).unwrap();

    println!("Solving Ax = b with GMRES:");
    println!("b = [{}, {}]", b2[0], b2[1]);
    println!("x = [{:.6}, {:.6}]", solution2[0], solution2[1]);

    // Verify solution
    let ax2 = nonsym_op.apply(&solution2.view()).unwrap();
    println!("Ax = [{:.6}, {:.6}]", ax2[0], ax2[1]);
    println!("Error: {:.6e}", vector_norm(&b2, &ax2));
    println!();

    // Example 4: Preconditioned conjugate gradient
    println!("Example 4: Preconditioned conjugate gradient");

    // Create a Jacobi preconditioner
    let precond = jacobi_preconditioner(&spd_op).unwrap();

    // Solve using preconditioned conjugate gradient
    let solution3 =
        matrix_free_preconditioned_conjugate_gradient(&spd_op, &precond, &b, 10, 1e-10).unwrap();

    println!("Solving Ax = b with preconditioned conjugate gradient:");
    println!("b = [{}, {}]", b[0], b[1]);
    println!("x = [{:.6}, {:.6}]", solution3[0], solution3[1]);

    // Verify solution
    let ax3 = spd_op.apply(&solution3.view()).unwrap();
    println!("Ax = [{:.6}, {:.6}]", ax3[0], ax3[1]);
    println!("Error: {:.6e}", vector_norm(&b, &ax3));
    println!();

    // Example 5: Block diagonal operator
    println!("Example 5: Block diagonal operator");

    // Create two smaller operators
    let op1 = LinearOperator::new(2, |v: &ArrayView1<f64>| {
        let mut result = Array1::zeros(2);
        result[0] = 2.0 * v[0];
        result[1] = 3.0 * v[1];
        result
    });

    let op2 = LinearOperator::new(1, |v: &ArrayView1<f64>| {
        let mut result = Array1::zeros(1);
        result[0] = 4.0 * v[0];
        result
    });

    // Create a block diagonal operator
    let block_op = block_diagonal_operator(vec![op1, op2]);

    // Create a vector and apply the block operator
    let x5 = array![1.0, 2.0, 3.0];
    let y5 = block_op.apply(&x5.view()).unwrap();

    println!("Block diagonal matrix-vector product:");
    println!("x = [{}, {}, {}]", x5[0], x5[1], x5[2]);
    println!("Ax = [{}, {}, {}]", y5[0], y5[1], y5[2]);
}

// Helper function to compute the norm of the difference vector
#[allow(dead_code)]
fn vector_norm(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum.sqrt()
}
