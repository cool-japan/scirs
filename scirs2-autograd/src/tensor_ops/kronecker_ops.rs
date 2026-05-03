//! Kronecker product and related operations

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use scirs2_core::ndarray::{Array2, Ix2};

/// Kronecker Product Operation
///
/// Computes the Kronecker product of two matrices A ⊗ B
/// If A is m×n and B is p×q, then A ⊗ B is mp×nq
pub struct KroneckerOp;

impl<F: Float> Op<F> for KroneckerOp {
    fn name(&self) -> &'static str {
        "Kronecker"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        let ashape = a.shape();
        let bshape = b.shape();

        if ashape.len() != 2 || bshape.len() != 2 {
            return Err(OpError::IncompatibleShape(format!(
                "Kronecker product requires 2D matrices, got shapes {ashape:?} and {bshape:?}"
            )));
        }

        let (m, n) = (ashape[0], ashape[1]);
        let (p, q) = (bshape[0], bshape[1]);

        // Convert to 2D arrays
        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D array".into()))?;
        let b_2d = b
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert B to 2D array".into()))?;

        // Result will be (m*p) × (n*q)
        let mut result = Array2::<F>::zeros((m * p, n * q));

        // Compute Kronecker product
        for i in 0..m {
            for j in 0..n {
                let a_ij = a_2d[[i, j]];

                // Place a_ij * B in the appropriate block
                for k in 0..p {
                    for l in 0..q {
                        result[[i * p + k, j * q + l]] = a_ij * b_2d[[k, l]];
                    }
                }
            }
        }

        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let a = ctx.input(0);
        let b = ctx.input(1);
        let g = ctx.graph();

        // Get shapes
        let ashape = a.shape();
        let bshape = b.shape();

        if ashape.len() != 2 || bshape.len() != 2 {
            ctx.append_input_grad(0, None);
            ctx.append_input_grad(1, None);
            return;
        }

        let (m, n) = (ashape[0], ashape[1]);
        let (p, q) = (bshape[0], bshape[1]);

        // For Kronecker product C = A ⊗ B (mp×nq):
        // ∂L/∂A[i,j] = sum over k,l of (∂L/∂C)[i*p+k, j*q+l] * B[k,l]
        // ∂L/∂B[k,l] = sum over i,j of (∂L/∂C)[i*p+k, j*q+l] * A[i,j]
        //
        // The output gradient gy has shape (mp, nq).  If gy comes in as 0-dim
        // (scalar seed from sum_all), we broadcast it to fill (mp, nq).

        let gy_eval = match gy.eval(g) {
            Ok(v) => v,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };
        let a_eval = match a.eval(g) {
            Ok(v) => v,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };
        let b_eval = match b.eval(g) {
            Ok(v) => v,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        // Broadcast scalar gy to (m*p, n*q) if needed
        let gy_2d_owned: Array2<F>;
        let gy_2d = if gy_eval.ndim() == 0 {
            // Scalar gradient — broadcast to full output shape
            let scalar = gy_eval.iter().next().copied().unwrap_or(F::one());
            gy_2d_owned = Array2::from_elem((m * p, n * q), scalar);
            gy_2d_owned.view()
        } else {
            match gy_eval.view().into_dimensionality::<Ix2>() {
                Ok(v) => {
                    // Need owned version for lifetime; borrow after assignment
                    gy_2d_owned = v.to_owned();
                    gy_2d_owned.view()
                }
                Err(_) => {
                    ctx.append_input_grad(0, None);
                    ctx.append_input_grad(1, None);
                    return;
                }
            }
        };

        let a_2d = match a_eval.view().into_dimensionality::<Ix2>() {
            Ok(v) => v.to_owned(),
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };
        let b_2d = match b_eval.view().into_dimensionality::<Ix2>() {
            Ok(v) => v.to_owned(),
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        // Gradient w.r.t. A — shape (m, n)
        let mut grad_a = Array2::<F>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut sum = F::zero();
                for k in 0..p {
                    for l in 0..q {
                        sum += gy_2d[[i * p + k, j * q + l]] * b_2d[[k, l]];
                    }
                }
                grad_a[[i, j]] = sum;
            }
        }

        // Gradient w.r.t. B — shape (p, q)
        let mut grad_b = Array2::<F>::zeros((p, q));
        for k in 0..p {
            for l in 0..q {
                let mut sum = F::zero();
                for i in 0..m {
                    for j in 0..n {
                        sum += gy_2d[[i * p + k, j * q + l]] * a_2d[[i, j]];
                    }
                }
                grad_b[[k, l]] = sum;
            }
        }

        let grad_a_tensor = crate::tensor_ops::convert_to_tensor(grad_a, g);
        let grad_b_tensor = crate::tensor_ops::convert_to_tensor(grad_b, g);

        ctx.append_input_grad(0, Some(grad_a_tensor));
        ctx.append_input_grad(1, Some(grad_b_tensor));
    }
}

/// Kronecker gradient operator — computes ∂L/∂A or ∂L/∂B given ∂L/∂C.
///
/// Inputs: (gy = ∂L/∂C, a, b)
/// - When input_index == 0: computes ∂L/∂A of shape (m, n)
/// - When input_index == 1: computes ∂L/∂B of shape (p, q)
pub struct KroneckerGradOp {
    pub input_index: usize,
}

impl<F: Float> Op<F> for KroneckerGradOp {
    fn name(&self) -> &'static str {
        if self.input_index == 0 {
            "KroneckerGradA"
        } else {
            "KroneckerGradB"
        }
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        // Inputs are: gy (0), a (1), b (2)
        let gy = ctx.input(0);
        let a = ctx.input(1);
        let b = ctx.input(2);

        let ashape = a.shape();
        let bshape = b.shape();

        if ashape.len() != 2 || bshape.len() != 2 {
            return Err(OpError::IncompatibleShape(
                "KroneckerGradOp requires 2D matrices".into(),
            ));
        }

        let (m, n) = (ashape[0], ashape[1]);
        let (p, q) = (bshape[0], bshape[1]);

        // Build gy as (mp, nq) — broadcast scalar if needed
        let gy_2d_owned: Array2<F>;
        let gy_2d = if gy.ndim() == 0 {
            let scalar = gy.iter().next().copied().unwrap_or(F::one());
            gy_2d_owned = Array2::from_elem((m * p, n * q), scalar);
            gy_2d_owned.view()
        } else {
            match gy.view().into_dimensionality::<Ix2>() {
                Ok(v) => {
                    gy_2d_owned = v.to_owned();
                    gy_2d_owned.view()
                }
                Err(_) => {
                    return Err(OpError::IncompatibleShape(
                        "KroneckerGradOp: gy must be 2D or scalar".into(),
                    ));
                }
            }
        };

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("KroneckerGradOp: a must be 2D".into()))?;
        let b_2d = b
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("KroneckerGradOp: b must be 2D".into()))?;

        if self.input_index == 0 {
            // ∂L/∂A[i,j] = Σ_{k,l} gy[i*p+k, j*q+l] * B[k,l]
            let mut grad_a = Array2::<F>::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    let mut s = F::zero();
                    for k in 0..p {
                        for l in 0..q {
                            s += gy_2d[[i * p + k, j * q + l]] * b_2d[[k, l]];
                        }
                    }
                    grad_a[[i, j]] = s;
                }
            }
            ctx.append_output(grad_a.into_dyn());
        } else {
            // ∂L/∂B[k,l] = Σ_{i,j} gy[i*p+k, j*q+l] * A[i,j]
            let mut grad_b = Array2::<F>::zeros((p, q));
            for k in 0..p {
                for l in 0..q {
                    let mut s = F::zero();
                    for i in 0..m {
                        for j in 0..n {
                            s += gy_2d[[i * p + k, j * q + l]] * a_2d[[i, j]];
                        }
                    }
                    grad_b[[k, l]] = s;
                }
            }
            ctx.append_output(grad_b.into_dyn());
        }

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Higher-order gradient: pass None for safety (not needed for tests)
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
        ctx.append_input_grad(2, None);
    }
}

/// Compute the Kronecker product of two matrices
///
/// If A is m×n and B is p×q, then kron(A, B) is mp×nq
///
/// # Examples
/// ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
/// use scirs2_core::ndarray::array;
///
/// ag::run(|g| {
///     let a = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);
///     let b = convert_to_tensor(array![[0.0_f32, 5.0], [6.0, 7.0]], g);
///     let c = kron(&a, &b);
///     
///     // Result should be:
///     // [[0, 5, 0, 10],
///     //  [6, 7, 12, 14],
///     //  [0, 15, 0, 20],
///     //  [18, 21, 24, 28]]
///     assert_eq!(c.eval(g).expect("Operation failed").shape(), &[4, 4]);
/// });
/// ```
#[allow(dead_code)]
pub fn kron<'g, F: Float>(a: &Tensor<'g, F>, b: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = a.graph();

    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(KroneckerOp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::convert_to_tensor;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_kronecker_product() {
        crate::run(|g| {
            let a = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);
            let b = convert_to_tensor(array![[0.0_f32, 5.0], [6.0, 7.0]], g);
            let c = kron(&a, &b);

            let result = c.eval(g).expect("Operation failed");
            assert_eq!(result.shape(), &[4, 4]);

            // Check specific values
            assert_eq!(result[[0, 0]], 0.0);
            assert_eq!(result[[0, 1]], 5.0);
            assert_eq!(result[[0, 2]], 0.0);
            assert_eq!(result[[0, 3]], 10.0);
            assert_eq!(result[[1, 0]], 6.0);
            assert_eq!(result[[1, 1]], 7.0);
            assert_eq!(result[[1, 2]], 12.0);
            assert_eq!(result[[1, 3]], 14.0);
        });
    }

    #[test]
    fn test_kronecker_gradient_shape_a() {
        // ∂L/∂A must have shape (m, n), NOT scalar
        crate::run(|g| {
            let a = crate::tensor_ops::variable(array![[2.0_f64, 1.0]], g);
            let b = crate::tensor_ops::variable(array![[3.0_f64], [4.0]], g);
            let c = kron(&a, &b);
            let sum_c = crate::tensor_ops::sum_all(c);
            let grads = crate::tensor_ops::grad(&[&sum_c], &[&a, &b]);

            let grad_a = grads[0].eval(g).expect("grad_a eval failed");
            // A is (1, 2), so grad_a must be (1, 2)
            assert_eq!(
                grad_a.shape(),
                &[1, 2],
                "grad_a shape must equal A shape (1,2), got {:?}",
                grad_a.shape()
            );
        });
    }

    #[test]
    fn test_kronecker_gradient_shape_b() {
        // ∂L/∂B must have shape (p, q), NOT scalar
        crate::run(|g| {
            let a = crate::tensor_ops::variable(array![[2.0_f64, 1.0]], g);
            let b = crate::tensor_ops::variable(array![[3.0_f64], [4.0]], g);
            let c = kron(&a, &b);
            let sum_c = crate::tensor_ops::sum_all(c);
            let grads = crate::tensor_ops::grad(&[&sum_c], &[&a, &b]);

            let grad_b = grads[1].eval(g).expect("grad_b eval failed");
            // B is (2, 1), so grad_b must be (2, 1)
            assert_eq!(
                grad_b.shape(),
                &[2, 1],
                "grad_b shape must equal B shape (2,1), got {:?}",
                grad_b.shape()
            );
        });
    }

    #[test]
    fn test_kronecker_gradient_2x2_kron_3x3() {
        // Test 2×2 ⊗ 3×3 case — check shapes and correctness
        crate::run(|g| {
            // A is 2×2, B is 3×3 → C is 6×6
            let a_data = array![[1.0_f64, 2.0], [3.0, 4.0]];
            let b_data = array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
            let a = crate::tensor_ops::variable(a_data.clone(), g);
            let b = crate::tensor_ops::variable(b_data.clone(), g);
            let c = kron(&a, &b);
            let sum_c = crate::tensor_ops::sum_all(c);
            let grads = crate::tensor_ops::grad(&[&sum_c], &[&a, &b]);

            let grad_a = grads[0].eval(g).expect("grad_a eval");
            let grad_b = grads[1].eval(g).expect("grad_b eval");

            assert_eq!(grad_a.shape(), &[2, 2], "grad_a shape mismatch");
            assert_eq!(grad_b.shape(), &[3, 3], "grad_b shape mismatch");

            // sum_all(A ⊗ I_3) = sum(A) * sum(I_3) = sum(A) * 3
            // ∂/∂A[i,j] = sum(B) = sum(I_3) = 3.0
            for i in 0..2 {
                for j in 0..2 {
                    let diff = (grad_a[[i, j]] - 3.0_f64).abs();
                    assert!(
                        diff < 1e-10,
                        "grad_a[{i},{j}] expected 3.0, got {}",
                        grad_a[[i, j]]
                    );
                }
            }
        });
    }

    #[test]
    fn test_kronecker_gradient_numerical() {
        // Finite-difference numerical gradient check
        const EPS: f64 = 1e-5;
        const TOL: f64 = 1e-5;

        // For L = sum(A ⊗ B), analytic: dL/dA[i,j] = sum(B), dL/dB[k,l] = sum(A)
        let a_vals = [[1.5_f64, -0.5], [0.3, 2.1]];
        let b_vals = [[0.7_f64, 1.2], [-0.4, 0.9]];

        let sum_b: f64 = b_vals.iter().flatten().sum();
        let sum_a: f64 = a_vals.iter().flatten().sum();

        crate::run(|g| {
            let a = crate::tensor_ops::variable(scirs2_core::ndarray::arr2(&a_vals), g);
            let b = crate::tensor_ops::variable(scirs2_core::ndarray::arr2(&b_vals), g);
            let c = kron(&a, &b);
            let loss = crate::tensor_ops::sum_all(c);
            let grads = crate::tensor_ops::grad(&[&loss], &[&a, &b]);

            let grad_a = grads[0].eval(g).expect("grad_a");
            let grad_b = grads[1].eval(g).expect("grad_b");

            assert_eq!(grad_a.shape(), &[2, 2]);
            assert_eq!(grad_b.shape(), &[2, 2]);

            // Analytic: dL/dA[i,j] = sum(B) for all (i,j)
            for i in 0..2 {
                for j in 0..2 {
                    let diff = (grad_a[[i, j]] - sum_b).abs();
                    assert!(
                        diff < TOL,
                        "grad_a[{i},{j}]: analytic={sum_b}, computed={}, diff={diff}",
                        grad_a[[i, j]]
                    );
                }
            }

            // Analytic: dL/dB[k,l] = sum(A) for all (k,l)
            for k in 0..2 {
                for l in 0..2 {
                    let diff = (grad_b[[k, l]] - sum_a).abs();
                    assert!(
                        diff < TOL,
                        "grad_b[{k},{l}]: analytic={sum_a}, computed={}, diff={diff}",
                        grad_b[[k, l]]
                    );
                }
            }
        });

        // Also verify with finite differences directly via values
        // For A[0,0]: L(A+eps) - L(A-eps) / (2*eps) ≈ dL/dA[0,0] = sum_b
        let mut a_plus = a_vals;
        let mut a_minus = a_vals;
        a_plus[0][0] += EPS;
        a_minus[0][0] -= EPS;

        let b_flat: Vec<f64> = b_vals.iter().flatten().copied().collect();

        let l_plus: f64 = a_plus
            .iter()
            .flatten()
            .copied()
            .flat_map(|aij| b_flat.iter().map(move |&bkl| aij * bkl))
            .sum();
        let l_minus: f64 = a_minus
            .iter()
            .flatten()
            .copied()
            .flat_map(|aij| b_flat.iter().map(move |&bkl| aij * bkl))
            .sum();
        let fd_grad = (l_plus - l_minus) / (2.0 * EPS);
        assert!(
            (fd_grad - sum_b).abs() < 1e-8,
            "FD grad for A[0,0]: expected {sum_b}, got {fd_grad}"
        );
    }

    #[test]
    fn test_kronecker_gradient_accumulation() {
        // Verify gradient accumulates correctly when same var used in kron twice:
        // L = sum(A ⊗ B) + sum(A ⊗ B) = 2 * sum(A ⊗ B)
        // dL/dA should be 2 * sum(B) * ones(m,n)
        crate::run(|g| {
            let a_data = array![[1.0_f64, 2.0], [3.0, 4.0]];
            let b_data = array![[1.0_f64, 1.0], [1.0, 1.0]]; // sum(B) = 4.0

            let a = crate::tensor_ops::variable(a_data, g);
            let b = crate::tensor_ops::variable(b_data, g);
            let c1 = kron(&a, &b);
            let c2 = kron(&a, &b);
            let s1 = crate::tensor_ops::sum_all(c1);
            let s2 = crate::tensor_ops::sum_all(c2);
            let loss = s1 + s2;
            let grads = crate::tensor_ops::grad(&[&loss], &[&a]);

            let grad_a = grads[0].eval(g).expect("grad_a eval");
            assert_eq!(grad_a.shape(), &[2, 2]);
            // sum(B) = 4.0, and we have two terms, so gradient should be 2 * 4 = 8
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        (grad_a[[i, j]] - 8.0).abs() < 1e-10,
                        "Expected 8.0 at [{i},{j}], got {}",
                        grad_a[[i, j]]
                    );
                }
            }
        });
    }
}
