//! QR decomposition operations — QROp and QRExtractOp.

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops::convert_to_tensor;
use crate::tensor_ops::decomposition_backward::qr_backward;
use crate::Float;
use scirs2_core::ndarray::{Array2, Ix2};

/// QR Decomposition
pub struct QROp;

impl<F: Float> Op<F> for QROp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape("QR requires 2D matrix".into()));
        }

        let m = shape[0];
        let n = shape[1];
        let k = m.min(n);

        println!("Computing QR decomposition for matrix of shape: [{m}, {n}]");

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        // Gram-Schmidt orthogonalization
        let mut q = Array2::<F>::zeros((m, k));
        let mut r = Array2::<F>::zeros((k, n));

        for j in 0..k {
            // Copy column j of input to column j of Q
            for i in 0..m {
                q[[i, j]] = input_2d[[i, j]];
            }

            // Orthogonalize against previous columns
            for i in 0..j {
                let mut dot_product = F::zero();
                for row in 0..m {
                    dot_product += q[[row, i]] * q[[row, j]];
                }
                r[[i, j]] = dot_product;

                for row in 0..m {
                    q[[row, j]] = q[[row, j]] - dot_product * q[[row, i]];
                }
            }

            // Normalize
            let mut norm = F::zero();
            for row in 0..m {
                norm += q[[row, j]] * q[[row, j]];
            }
            norm = norm.sqrt();

            if norm > F::epsilon() {
                r[[j, j]] = norm;
                for row in 0..m {
                    q[[row, j]] /= norm;
                }
            }

            // Fill rest of R
            for col in (j + 1)..n {
                let mut dot_product = F::zero();
                for row in 0..m {
                    dot_product += q[[row, j]] * input_2d[[row, col]];
                }
                r[[j, col]] = dot_product;
            }
        }

        // Debug output
        println!("QR decomposition results:");
        println!("Q shape: {:?}, R shape: {:?}", q.shape(), r.shape());

        // Append the outputs with their shapes preserved
        ctx.append_output(q.into_dyn());
        ctx.append_output(r.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // QROp is no longer used directly (qr() calls QRExtractOp).
        // Fallback: pass zero gradient to avoid panics.
        let input = ctx.input(0);
        let g = ctx.graph();

        let input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let zero_grad = Array2::<F>::zeros((input_array.shape()[0], input_array.shape()[1]));
        let grad_tensor = convert_to_tensor(zero_grad.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

/// QR component extraction
pub struct QRExtractOp {
    pub component: usize,
}

impl<F: Float> Op<F> for QRExtractOp {
    fn name(&self) -> &'static str {
        match self.component {
            0 => "QRExtractQ",
            1 => "QRExtractR",
            _ => "QRExtractUnknown",
        }
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape("QR requires 2D matrix".into()));
        }

        let m = shape[0];
        let n = shape[1];
        let k = m.min(n); // thin QR: Q is m×k, R is k×n

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        // Thin QR via Gram-Schmidt.
        let mut q = Array2::<F>::zeros((m, k));
        let mut r = Array2::<F>::zeros((k, n));

        for j in 0..k {
            for i in 0..m {
                q[[i, j]] = input_2d[[i, j]];
            }
            for i in 0..j {
                let mut dot = F::zero();
                for row in 0..m {
                    dot += q[[row, i]] * q[[row, j]];
                }
                r[[i, j]] = dot;
                for row in 0..m {
                    q[[row, j]] = q[[row, j]] - dot * q[[row, i]];
                }
            }
            let mut norm = F::zero();
            for row in 0..m {
                norm += q[[row, j]] * q[[row, j]];
            }
            norm = norm.sqrt();
            if norm > F::epsilon() {
                r[[j, j]] = norm;
                for row in 0..m {
                    q[[row, j]] /= norm;
                }
            }
            for col in (j + 1)..n {
                let mut dot = F::zero();
                for row in 0..m {
                    dot += q[[row, j]] * input_2d[[row, col]];
                }
                r[[j, col]] = dot;
            }
        }

        match self.component {
            0 => ctx.append_output(q.into_dyn()),
            1 => ctx.append_output(r.into_dyn()),
            _ => return Err(OpError::IncompatibleShape("Invalid component index".into())),
        }

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Townsend / Murray (2016) QR backward, exploiting linearity:
        //   QRExtractOp{0} (Q component): grad_a = qr_backward(full_q, full_r, embed(gQ), 0)
        //   QRExtractOp{1} (R component): grad_a = qr_backward(full_q, full_r, 0, embed(gR))
        //
        // grad() always works with full QR (Q m×m, R m×n) for correctness.
        // The upstream gradient has shape corresponding to the thin output:
        //   component=0: grad_2d is m×k, embed in m×m zero matrix (pad with zeros)
        //   component=1: grad_2d is k×n, embed in m×n zero matrix (top k rows)

        let gy = ctx.output_grad();
        let input = ctx.input(0);
        let g = ctx.graph();

        let input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };
        let grad_array = match gy.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let input_2d = match input_array.view().into_dimensionality::<Ix2>() {
            Ok(v) => v.to_owned(),
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };
        let grad_2d = match grad_array.view().into_dimensionality::<Ix2>() {
            Ok(v) => v.to_owned(),
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let m = input_2d.nrows();
        let n = input_2d.ncols();

        // Re-compute full QR (Q is m×m, R is m×n) for backward.
        let mut q_full = Array2::<F>::zeros((m, m));
        let mut r_full = Array2::<F>::zeros((m, n));

        let mut a_aug = Array2::<F>::zeros((m, m));
        for i in 0..m {
            for j in 0..n {
                a_aug[[i, j]] = input_2d[[i, j]];
            }
            if i >= n {
                a_aug[[i, i]] = F::one();
            }
        }
        for j in 0..m {
            for i in 0..m {
                q_full[[i, j]] = a_aug[[i, j]];
            }
            for i in 0..j {
                let mut dot = F::zero();
                for row in 0..m {
                    dot += q_full[[row, i]] * q_full[[row, j]];
                }
                for row in 0..m {
                    q_full[[row, j]] = q_full[[row, j]] - dot * q_full[[row, i]];
                }
            }
            let mut norm = F::zero();
            for row in 0..m {
                norm += q_full[[row, j]] * q_full[[row, j]];
            }
            norm = norm.sqrt();
            if norm > F::epsilon() {
                for row in 0..m {
                    q_full[[row, j]] /= norm;
                }
            }
        }
        for i in 0..m {
            for j in 0..n {
                let mut dot = F::zero();
                for row in 0..m {
                    dot += q_full[[row, i]] * input_2d[[row, j]];
                }
                r_full[[i, j]] = dot;
            }
        }

        // Embed the thin upstream gradient into the full-QR shape expected by qr_backward.
        // qr_backward takes grad_q as m×m and grad_r as m×n.
        let grad_q_full;
        let grad_r_full;
        match self.component {
            0 => {
                // Upstream dQ is m×k (thin). Embed it in m×m by padding columns with zeros.
                let k = grad_2d.ncols();
                let mut gq = Array2::<F>::zeros((m, m));
                for i in 0..m {
                    for j in 0..k {
                        gq[[i, j]] = grad_2d[[i, j]];
                    }
                }
                grad_q_full = gq;
                grad_r_full = Array2::<F>::zeros((m, n));
            }
            1 => {
                // Upstream dR is k×n (thin). Embed it in m×n by padding rows with zeros.
                let k = grad_2d.nrows();
                let mut gr = Array2::<F>::zeros((m, n));
                for i in 0..k {
                    for j in 0..n {
                        gr[[i, j]] = grad_2d[[i, j]];
                    }
                }
                grad_q_full = Array2::<F>::zeros((m, m));
                grad_r_full = gr;
            }
            _ => {
                ctx.append_input_grad(0, None);
                return;
            }
        }

        let grad_a = qr_backward(&q_full, &r_full, &grad_q_full, &grad_r_full);
        let grad_tensor = convert_to_tensor(grad_a.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

/// QR decomposition of a matrix.
///
/// Decomposes a matrix A into Q and R matrices such that A = Q * R, where:
/// - Q is an orthogonal matrix (Q^T * Q = I)
/// - R is an upper triangular matrix
///
/// # Arguments
/// * `matrix` - The input tensor to decompose
///
/// # Returns
/// A tuple of tensors (Q, R) representing the decomposition
#[allow(dead_code)]
pub fn qr<'g, F: Float>(matrix: &Tensor<'g, F>) -> (Tensor<'g, F>, Tensor<'g, F>) {
    let g = matrix.graph();

    // Create component ops directly using extraction operators
    let q = Tensor::builder(g)
        .append_input(matrix, false)
        .build(QRExtractOp { component: 0 });

    let r = Tensor::builder(g)
        .append_input(matrix, false)
        .build(QRExtractOp { component: 1 });

    (q, r)
}
