//! Block-structured Preconditioners for DAE Systems
//!
//! This module provides block-structured preconditioners for improving the
//! convergence of Krylov subspace methods when solving large DAE systems.
//! These preconditioners exploit the natural block structure of DAE systems,
//! especially for semi-explicit DAEs, which have a clear separation between
//! differential and algebraic variables.
//!
//! The preconditioners in this module are designed to be used with the
//! Krylov-based DAE solvers in the `krylov_dae` module.

use crate::IntegrateFloat;
use ndarray::{Array1, Array2};

/// Block ILU(0) Preconditioner for Semi-explicit DAE Systems
///
/// This struct implements an incomplete LU factorization preconditioner that
/// exploits the block structure of semi-explicit DAE systems.
///
/// The semi-explicit DAE system:
/// ```ignore
/// x' = f(x, y, t)
/// 0 = g(x, y, t)
/// ```
///
/// leads to a Jacobian matrix of the form:
/// ```ignore
/// | I - h*β*∂f/∂x  -h*β*∂f/∂y |
/// | ∂g/∂x          ∂g/∂y      |
/// ```
///
/// This block structure is exploited by the preconditioner for more efficient
/// solution of the linear systems arising in Newton iterations.
pub struct BlockILUPreconditioner<F: IntegrateFloat> {
    // Sizes of differential and algebraic parts
    n_x: usize,
    n_y: usize,

    // Block components
    #[allow(dead_code)]
    a_block: Array2<F>, // I - h*β*∂f/∂x
    #[allow(dead_code)]
    b_block: Array2<F>, // -h*β*∂f/∂y
    #[allow(dead_code)]
    c_block: Array2<F>, // ∂g/∂x
    #[allow(dead_code)]
    d_block: Array2<F>, // ∂g/∂y

    // ILU(0) factors for the blocks
    l_factors: Array2<F>,
    u_factors: Array2<F>,

    // Diagonal scaling factors
    d_scaling: Array1<F>,
}

impl<F: IntegrateFloat> BlockILUPreconditioner<F> {
    /// Create a new block ILU(0) preconditioner for semi-explicit DAE systems
    ///
    /// # Arguments
    /// * `f_x` - Jacobian of f with respect to x variables (∂f/∂x)
    /// * `f_y` - Jacobian of f with respect to y variables (∂f/∂y)
    /// * `g_x` - Jacobian of g with respect to x variables (∂g/∂x)
    /// * `g_y` - Jacobian of g with respect to y variables (∂g/∂y)
    /// * `h` - Step size
    /// * `beta` - BDF method coefficient
    pub fn new(
        f_x: &Array2<F>,
        f_y: &Array2<F>,
        g_x: &Array2<F>,
        g_y: &Array2<F>,
        h: F,
        beta: F,
    ) -> Self {
        let n_x = f_x.shape()[0];
        let n_y = g_y.shape()[0];

        // Compute the blocks of the Jacobian matrix
        let mut a_block = Array2::<F>::eye(n_x); // Identity matrix

        // Subtract h*β*∂f/∂x from the identity
        for i in 0..n_x {
            for j in 0..n_x {
                a_block[[i, j]] -= h * beta * f_x[[i, j]];
            }
        }

        // Compute -h*β*∂f/∂y
        let b_block = f_y.mapv(|x| -h * beta * x);

        // Direct copies of the constraint Jacobians
        let c_block = g_x.clone();
        let d_block = g_y.clone();

        // Compute ILU(0) factorization for the blocks
        let (l_factors, u_factors, d_scaling) =
            Self::compute_block_ilu0(&a_block, &b_block, &c_block, &d_block);

        BlockILUPreconditioner {
            n_x,
            n_y,
            a_block,
            b_block,
            c_block,
            d_block,
            l_factors,
            u_factors,
            d_scaling,
        }
    }

    /// Compute ILU(0) factorization for the block Jacobian
    ///
    /// Uses an approximate factorization that preserves the block structure
    /// and only computes entries where the original matrix has nonzeros.
    fn compute_block_ilu0(
        a: &Array2<F>,
        b: &Array2<F>,
        c: &Array2<F>,
        d: &Array2<F>,
    ) -> (Array2<F>, Array2<F>, Array1<F>) {
        let n_x = a.shape()[0];
        let n_y = d.shape()[0];
        let n_total = n_x + n_y;

        // Allocate storage for L and U factors
        let mut l = Array2::<F>::zeros((n_total, n_total));
        let mut u = Array2::<F>::zeros((n_total, n_total));

        // Initialize L and U with the blocks
        // L gets the lower triangular part of the block structure
        // U gets the upper triangular part of the block structure

        // Fill in the A block (split between L and U)
        for i in 0..n_x {
            for j in 0..n_x {
                match i.cmp(&j) {
                    std::cmp::Ordering::Greater => {
                        // Lower triangular part goes to L
                        l[[i, j]] = a[[i, j]];
                    }
                    std::cmp::Ordering::Equal => {
                        // Diagonal goes to U
                        u[[i, j]] = a[[i, j]];
                    }
                    std::cmp::Ordering::Less => {
                        // Upper triangular part goes to U
                        u[[i, j]] = a[[i, j]];
                    }
                }
            }
        }

        // Fill in the B block (goes to U as it's in the upper-right)
        for i in 0..n_x {
            for j in 0..n_y {
                u[[i, n_x + j]] = b[[i, j]];
            }
        }

        // Fill in the C block (goes to L as it's in the lower-left)
        for i in 0..n_y {
            for j in 0..n_x {
                l[[n_x + i, j]] = c[[i, j]];
            }
        }

        // Fill in the D block (split between L and U)
        for i in 0..n_y {
            for j in 0..n_y {
                match i.cmp(&j) {
                    std::cmp::Ordering::Greater => {
                        // Lower triangular part goes to L
                        l[[n_x + i, n_x + j]] = d[[i, j]];
                    }
                    std::cmp::Ordering::Equal => {
                        // Diagonal goes to U
                        u[[n_x + i, n_x + j]] = d[[i, j]];
                    }
                    std::cmp::Ordering::Less => {
                        // Upper triangular part goes to U
                        u[[n_x + i, n_x + j]] = d[[i, j]];
                    }
                }
            }
        }

        // Set the unit diagonal in L
        for i in 0..n_total {
            l[[i, i]] = F::one();
        }

        // Perform the incomplete LU factorization (ILU(0))
        // We only update entries that are non-zero in the original matrix
        for k in 0..n_total {
            // Process the entries below the diagonal in column k
            for i in (k + 1)..n_total {
                // Only process entries that are non-zero in L
                if l[[i, k]].abs() > F::from_f64(1e-14).unwrap() {
                    l[[i, k]] /= u[[k, k]];

                    // Update the remaining entries in row i
                    for j in (k + 1)..n_total {
                        // Only update entries that are non-zero in U
                        if u[[k, j]].abs() > F::from_f64(1e-14).unwrap() {
                            // Only update if the entry is non-zero in L or U
                            if l[[i, j]].abs() > F::from_f64(1e-14).unwrap()
                                || u[[i, j]].abs() > F::from_f64(1e-14).unwrap()
                            {
                                l[[i, j]] = l[[i, j]] - l[[i, k]] * u[[k, j]];
                            }
                        }
                    }
                }
            }
        }

        // Extract diagonal scaling factors for better conditioning
        let mut d_scaling = Array1::<F>::ones(n_total);
        for i in 0..n_total {
            // Ensure diagonal elements are not too small
            if u[[i, i]].abs() < F::from_f64(1e-14).unwrap() {
                u[[i, i]] = F::from_f64(1e-14).unwrap() * u[[i, i]].signum();
            }
            // Store inverse of diagonal for faster application
            d_scaling[i] = F::one() / u[[i, i]];
        }

        (l, u, d_scaling)
    }

    /// Apply the preconditioner: P⁻¹v
    ///
    /// Solves the system P * z = v for z, where P is the block ILU(0) preconditioner
    pub fn apply(&self, v: &Array1<F>) -> Array1<F> {
        let n_total = self.n_x + self.n_y;
        let mut result = Array1::<F>::zeros(n_total);

        // First solve L * y = v for y
        let mut y = Array1::<F>::zeros(n_total);

        // Forward substitution
        for i in 0..n_total {
            y[i] = v[i];
            for j in 0..i {
                y[i] = y[i] - self.l_factors[[i, j]] * y[j];
            }
        }

        // Then solve U * z = y for z
        // Backward substitution, but using the pre-computed diagonal scaling
        for i in (0..n_total).rev() {
            result[i] = y[i];
            for j in (i + 1)..n_total {
                result[i] = result[i] - self.u_factors[[i, j]] * result[j];
            }
            // Apply diagonal scaling
            result[i] *= self.d_scaling[i];
        }

        result
    }
}

/// Block Jacobi Preconditioner for General DAE Systems
///
/// This struct implements a block Jacobi preconditioner that exploits the natural
/// block structure in DAE systems. The preconditioner uses approximate inverses
/// of the diagonal blocks, which is computationally efficient and preserves sparsity.
pub struct BlockJacobiPreconditioner<F: IntegrateFloat> {
    // Total size of the system
    n: usize,

    // Block size (for block diagonal structure)
    block_size: usize,

    // Number of blocks
    n_blocks: usize,

    // Block diagonal inverses
    block_inverses: Vec<Array2<F>>,
}

impl<F: IntegrateFloat> BlockJacobiPreconditioner<F> {
    /// Create a new block Jacobi preconditioner with automatic block detection
    ///
    /// This constructor attempts to identify natural blocks in the Jacobian
    /// based on the sparsity pattern and the structure of the DAE system.
    ///
    /// # Arguments
    /// * `jacobian` - The full Jacobian matrix to be preconditioned
    /// * `block_size` - Size of the diagonal blocks (typically related to the physics of the problem)
    pub fn new(jacobian: &Array2<F>, block_size: usize) -> Self {
        let n = jacobian.shape()[0];

        // Ensure block size divides the matrix size evenly
        assert!(
            n % block_size == 0,
            "Matrix size must be divisible by block size"
        );

        let n_blocks = n / block_size;
        let mut block_inverses = Vec::with_capacity(n_blocks);

        // Extract and invert each diagonal block
        for i in 0..n_blocks {
            let start = i * block_size;
            let end = start + block_size;

            // Extract the diagonal block
            let block = jacobian
                .slice(ndarray::s![start..end, start..end])
                .to_owned();

            // Compute the inverse of the block (or an approximation)
            let block_inv = Self::compute_block_inverse(&block);

            block_inverses.push(block_inv);
        }

        BlockJacobiPreconditioner {
            n,
            block_size,
            n_blocks,
            block_inverses,
        }
    }

    /// Compute the inverse of a small block matrix
    ///
    /// For small blocks, we can compute the exact inverse.
    /// For larger blocks, we might use an approximate inverse.
    fn compute_block_inverse(block: &Array2<F>) -> Array2<F> {
        let n = block.shape()[0];
        let mut result = Array2::<F>::zeros((n, n));

        // For 1x1 blocks, just take the reciprocal
        if n == 1 {
            let val = block[[0, 0]];
            if val.abs() < F::from_f64(1e-14).unwrap() {
                result[[0, 0]] = F::from_f64(1e-14).unwrap() * val.signum();
            } else {
                result[[0, 0]] = F::one() / val;
            }
            return result;
        }

        // For 2x2 blocks, use the analytic formula
        if n == 2 {
            let a = block[[0, 0]];
            let b = block[[0, 1]];
            let c = block[[1, 0]];
            let d = block[[1, 1]];

            let det = a * d - b * c;
            if det.abs() < F::from_f64(1e-14).unwrap() {
                // If determinant is too small, use regularization
                let reg = F::from_f64(1e-14).unwrap() * det.signum();
                result[[0, 0]] = d / (det + reg);
                result[[0, 1]] = -b / (det + reg);
                result[[1, 0]] = -c / (det + reg);
                result[[1, 1]] = a / (det + reg);
            } else {
                result[[0, 0]] = d / det;
                result[[0, 1]] = -b / det;
                result[[1, 0]] = -c / det;
                result[[1, 1]] = a / det;
            }
            return result;
        }

        // For 3x3 blocks, still use the analytic formula
        if n == 3 {
            // Extract the block elements
            let a = block[[0, 0]];
            let b = block[[0, 1]];
            let c = block[[0, 2]];
            let d = block[[1, 0]];
            let e = block[[1, 1]];
            let f = block[[1, 2]];
            let g = block[[2, 0]];
            let h = block[[2, 1]];
            let i = block[[2, 2]];

            // Compute the determinant
            let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);

            if det.abs() < F::from_f64(1e-14).unwrap() {
                // If determinant is too small, use regularization
                let reg = F::from_f64(1e-14).unwrap() * det.signum();

                // Compute the adjugate matrix elements
                let a11 = (e * i - f * h) / (det + reg);
                let a12 = (c * h - b * i) / (det + reg);
                let a13 = (b * f - c * e) / (det + reg);
                let a21 = (f * g - d * i) / (det + reg);
                let a22 = (a * i - c * g) / (det + reg);
                let a23 = (c * d - a * f) / (det + reg);
                let a31 = (d * h - e * g) / (det + reg);
                let a32 = (b * g - a * h) / (det + reg);
                let a33 = (a * e - b * d) / (det + reg);

                // Store the inverse
                result[[0, 0]] = a11;
                result[[0, 1]] = a12;
                result[[0, 2]] = a13;
                result[[1, 0]] = a21;
                result[[1, 1]] = a22;
                result[[1, 2]] = a23;
                result[[2, 0]] = a31;
                result[[2, 1]] = a32;
                result[[2, 2]] = a33;
            } else {
                // Compute the adjugate matrix elements
                let a11 = (e * i - f * h) / det;
                let a12 = (c * h - b * i) / det;
                let a13 = (b * f - c * e) / det;
                let a21 = (f * g - d * i) / det;
                let a22 = (a * i - c * g) / det;
                let a23 = (c * d - a * f) / det;
                let a31 = (d * h - e * g) / det;
                let a32 = (b * g - a * h) / det;
                let a33 = (a * e - b * d) / det;

                // Store the inverse
                result[[0, 0]] = a11;
                result[[0, 1]] = a12;
                result[[0, 2]] = a13;
                result[[1, 0]] = a21;
                result[[1, 1]] = a22;
                result[[1, 2]] = a23;
                result[[2, 0]] = a31;
                result[[2, 1]] = a32;
                result[[2, 2]] = a33;
            }
            return result;
        }

        // For larger blocks, use a more general approach
        // This is a simple Gauss-Jordan elimination for demonstration
        // In a production setting, consider using a more robust method

        // Start with the identity matrix
        let mut inv = Array2::<F>::eye(n);
        let mut lu = block.clone();

        // Perform Gaussian elimination with pivoting
        for k in 0..n {
            // Find the pivot
            let mut p = k;
            let mut max_val = lu[[k, k]].abs();

            for i in (k + 1)..n {
                if lu[[i, k]].abs() > max_val {
                    max_val = lu[[i, k]].abs();
                    p = i;
                }
            }

            // If the pivot is too small, use a small value
            if max_val < F::from_f64(1e-14).unwrap() {
                lu[[p, k]] = F::from_f64(1e-14).unwrap() * lu[[p, k]].signum();
            }

            // Swap rows if necessary
            if p != k {
                for j in 0..n {
                    let temp = lu[[k, j]];
                    lu[[k, j]] = lu[[p, j]];
                    lu[[p, j]] = temp;

                    let temp = inv[[k, j]];
                    inv[[k, j]] = inv[[p, j]];
                    inv[[p, j]] = temp;
                }
            }

            // Scale the pivot row
            let pivot = lu[[k, k]];
            for j in 0..n {
                lu[[k, j]] /= pivot;
                inv[[k, j]] /= pivot;
            }

            // Eliminate other rows
            for i in 0..n {
                if i != k {
                    let factor = lu[[i, k]];
                    for j in 0..n {
                        lu[[i, j]] = lu[[i, j]] - factor * lu[[k, j]];
                        inv[[i, j]] = inv[[i, j]] - factor * inv[[k, j]];
                    }
                }
            }
        }

        inv
    }

    /// Apply the preconditioner: P⁻¹v
    ///
    /// Multiplies the input vector by the block diagonal inverse
    pub fn apply(&self, v: &Array1<F>) -> Array1<F> {
        let mut result = Array1::<F>::zeros(self.n);

        // Apply each block inverse to the corresponding segment of the input vector
        for i in 0..self.n_blocks {
            let start = i * self.block_size;
            let end = start + self.block_size;

            // Extract the segment of the input vector
            let v_segment = v.slice(ndarray::s![start..end]).to_owned();

            // Apply the block inverse
            let block_result = &self.block_inverses[i].dot(&v_segment);

            // Copy the result to the output vector
            for j in 0..self.block_size {
                result[start + j] = block_result[j];
            }
        }

        result
    }
}

/// Create a block ILU preconditioner for semi-explicit DAE systems
///
/// This is a factory function that creates a block ILU preconditioner
/// for semi-explicit DAE systems from the Jacobian components.
///
/// # Arguments
/// * `f_x` - Jacobian of f with respect to x variables (∂f/∂x)
/// * `f_y` - Jacobian of f with respect to y variables (∂f/∂y)
/// * `g_x` - Jacobian of g with respect to x variables (∂g/∂x)
/// * `g_y` - Jacobian of g with respect to y variables (∂g/∂y)
/// * `h` - Step size
/// * `beta` - BDF method coefficient
///
/// # Returns
/// * A function that applies the preconditioner to a vector
pub fn create_block_ilu_preconditioner<F>(
    f_x: &Array2<F>,
    f_y: &Array2<F>,
    g_x: &Array2<F>,
    g_y: &Array2<F>,
    h: F,
    beta: F,
) -> impl Fn(&Array1<F>) -> Array1<F>
where
    F: IntegrateFloat,
{
    // Create the preconditioner
    let preconditioner = BlockILUPreconditioner::new(f_x, f_y, g_x, g_y, h, beta);

    // Return a closure that applies the preconditioner
    move |v: &Array1<F>| preconditioner.apply(v)
}

/// Create a block Jacobi preconditioner for a general DAE system
///
/// This is a factory function that creates a block Jacobi preconditioner
/// for general DAE systems from the full Jacobian matrix.
///
/// # Arguments
/// * `jacobian` - The full Jacobian matrix of the DAE system
/// * `block_size` - Size of the diagonal blocks (typically related to the physics of the problem)
///
/// # Returns
/// * A function that applies the preconditioner to a vector
pub fn create_block_jacobi_preconditioner<F>(
    jacobian: &Array2<F>,
    block_size: usize,
) -> impl Fn(&Array1<F>) -> Array1<F>
where
    F: IntegrateFloat,
{
    // Create the preconditioner
    let preconditioner = BlockJacobiPreconditioner::new(jacobian, block_size);

    // Return a closure that applies the preconditioner
    move |v: &Array1<F>| preconditioner.apply(v)
}
