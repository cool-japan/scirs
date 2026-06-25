use crate::ndarray_ext::NdArray;
use crate::op::OpError;
use crate::Float;

// GitHub Issue #98: Helper functions for scalar/1×1 parameter handling

/// Check if an array is scalar-like (0-D or 1-element)
fn is_scalar<F: Float>(arr: &NdArray<F>) -> bool {
    arr.shape().is_empty() || (arr.len() == 1)
}

/// Extract scalar value from 0-D or 1-element arrays
/// Handles both shape [] and shape [1] (GitHub Issue #98)
fn extract_scalar<F: Float>(arr: &NdArray<F>) -> Result<F, OpError> {
    if arr.shape().is_empty() {
        // 0-D array: use IxDyn(&[])
        Ok(arr[scirs2_core::ndarray::IxDyn(&[])])
    } else if arr.len() == 1 {
        // 1-element array: use iter().next()
        arr.iter().next().copied().ok_or_else(|| {
            OpError::IncompatibleShape("Failed to extract scalar from 1-element array".to_string())
        })
    } else {
        Err(OpError::IncompatibleShape(format!(
            "Expected scalar or 1-element array, got shape {:?}",
            arr.shape()
        )))
    }
}

pub(crate) struct AdamOp<F: Float> {
    pub(crate) alpha: F,
    pub(crate) eps: F,
    pub(crate) b1: F,
    pub(crate) b2: F,
}

impl<F: Float> crate::op::Op<F> for AdamOp<F> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<F>) -> Result<(), OpError> {
        // Since we can't modify inputs directly with input_mut, we need to
        // create new arrays for all our outputs and return them

        // Check if we have all the inputs we need
        if ctx.inputs().len() < 5 {
            return Err(OpError::IncompatibleShape(format!(
                "AdamOp requires 5 inputs, but got {}",
                ctx.inputs().len()
            )));
        }

        // Get all the inputs we need (clone them to avoid borrowing issues)
        let param = ctx.input(0).to_owned(); // The parameter to update
        let grad_raw = ctx.input(1).to_owned(); // The gradient
        let m = ctx.input(2).to_owned(); // First moment estimate
        let v = ctx.input(3).to_owned(); // Second moment estimate
        let t_array = ctx.input(4).to_owned(); // Timestep

        // When the parameter is scalar-like but the gradient has more elements
        // (e.g., from broadcasting during the forward pass), reduce the gradient
        // by summing to match the parameter's shape. This is standard behavior
        // for gradient accumulation across broadcast dimensions.
        let param_is_scalar = is_scalar(&param);
        let grad = if param_is_scalar && grad_raw.len() > 1 {
            // Sum all gradient elements to produce a scalar gradient
            let sum_val = grad_raw.iter().fold(F::zero(), |acc, &x| acc + x);
            if param.shape().is_empty() {
                NdArray::from_elem(scirs2_core::ndarray::IxDyn(&[]), sum_val)
            } else {
                NdArray::from_elem(scirs2_core::ndarray::IxDyn(&[1]), sum_val)
            }
        } else if !param_is_scalar && param.shape() != grad_raw.shape() {
            // For non-scalar params, try to reduce gradient to param shape
            // by summing over extra dimensions
            let param_len = param.len();
            let grad_len = grad_raw.len();
            if grad_len > param_len && grad_len.is_multiple_of(param_len) {
                // Sum grad elements in groups to match param size
                let mut reduced = NdArray::zeros(param.raw_dim());
                let chunks = grad_len / param_len;
                let grad_flat = grad_raw.iter().copied().collect::<Vec<_>>();
                for (i, elem) in reduced.iter_mut().enumerate() {
                    let mut sum = F::zero();
                    for c in 0..chunks {
                        sum += grad_flat[c * param_len + i];
                    }
                    *elem = sum;
                }
                reduced
            } else {
                grad_raw
            }
        } else {
            grad_raw
        };

        // Handle shape mismatches: ensure arrays have compatible shapes
        // We need to create arrays of matching shapes for operations to work
        let gradshape = grad.shape().to_vec();

        // Get the current timestep value and increment it (GitHub Issue #98: handle 1-element arrays)
        let t_val = extract_scalar(&t_array)?;
        let new_t = t_val + F::one();
        let new_t_array = if is_scalar(&t_array) && t_array.shape().is_empty() {
            NdArray::from_elem(scirs2_core::ndarray::IxDyn(&[]), new_t)
        } else {
            // Preserve original shape (e.g., [1])
            NdArray::from_elem(scirs2_core::ndarray::IxDyn(&[1]), new_t)
        };

        // Create new momentum and velocity arrays with the same shape as grad
        // If original arrays are scalar and grad is not, we need to broadcast
        let mut new_m: NdArray<F>;
        let mut new_v: NdArray<F>;

        // Check if we need to broadcast scalar arrays to match grad shape (GitHub Issue #98)
        if is_scalar(&m) && !gradshape.is_empty() {
            // If m is scalar but grad is not, create a new array with m's value broadcast to grad's shape
            let m_val = extract_scalar(&m)?;
            new_m = NdArray::from_elem(scirs2_core::ndarray::IxDyn(&gradshape), m_val);
        } else {
            new_m = m.to_owned();
        }

        if is_scalar(&v) && !gradshape.is_empty() {
            // If v is scalar but grad is not, create a new array with v's value broadcast to grad's shape
            let v_val = extract_scalar(&v)?;
            new_v = NdArray::from_elem(scirs2_core::ndarray::IxDyn(&gradshape), v_val);
        } else {
            new_v = v.to_owned();
        }

        // Also handle param broadcasting if needed (GitHub Issue #98)
        let mut new_param: NdArray<F>;
        if is_scalar(&param) && !gradshape.is_empty() {
            let param_val = extract_scalar(&param)?;
            new_param = NdArray::from_elem(scirs2_core::ndarray::IxDyn(&gradshape), param_val);
        } else {
            new_param = param.to_owned();
        }

        // Compute new first moment estimate
        let tmp_b1 = F::one() - self.b1;
        new_m.zip_mut_with(&grad, move |m_val, g_val| {
            *m_val = *m_val * self.b1 + tmp_b1 * *g_val
        });

        // Compute new second moment estimate
        let tmp_b2 = F::one() - self.b2;
        new_v.zip_mut_with(&grad, move |v_val, g_val| {
            *v_val = *v_val * self.b2 + tmp_b2 * *g_val * *g_val
        });

        // Compute bias-corrected estimates
        let m_correction = F::one() / (F::one() - self.b1.powf(new_t));
        let v_correction = F::one() / (F::one() - self.b2.powf(new_t));

        let m_hat = new_m.mapv(move |m_val| m_val * m_correction);
        let v_hat = new_v.mapv(move |v_val| v_val * v_correction);

        // Compute the parameter update
        let mut update = m_hat.to_owned();
        update.zip_mut_with(&v_hat, move |m_hat_val, v_hat_val| {
            *m_hat_val /= v_hat_val.sqrt() + self.eps;
        });

        // Apply updates to parameters
        new_param.zip_mut_with(&update, move |param_val, update_val| {
            *param_val -= self.alpha * *update_val
        });

        // Append all outputs to the context
        ctx.append_output(new_param); // Updated parameter
        ctx.append_output(grad); // Gradient (unchanged)
        ctx.append_output(new_m); // Updated first moment
        ctx.append_output(new_v); // Updated second moment
        ctx.append_output(new_t_array); // Updated timestep

        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<F>) {
        // Since this is an optimizer operation, we don't propagate gradients
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
        ctx.append_input_grad(2, None);
        ctx.append_input_grad(3, None);
        ctx.append_input_grad(4, None);
    }
}
