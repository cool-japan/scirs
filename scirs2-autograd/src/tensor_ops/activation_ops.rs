use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;

use crate::tensor::Tensor;

use crate::ndarray;
use crate::tensor_ops::*;
use crate::Float;

pub struct Elu<T> {
    pub alpha: T,
}

pub struct EluGrad<T> {
    pub alpha: T,
}

pub struct Identity;

pub struct ReLU;

pub struct Sigmoid;

pub struct Softplus;

pub struct Softmax {
    pub axis: isize,
}

#[cfg(feature = "blas")]
#[allow(dead_code)]
fn fast_sigmoid_impl<F: Float>(x: &NdArrayView<F>) -> NdArray<F> {
    use crate::same_type;

    let half = F::from(0.5).expect("Failed to convert constant to float");

    // Use standard tanh implementation since MKL vectorized functions are no longer available
    // This provides a fast sigmoid approximation: sigmoid(x) ≈ 0.5 * (tanh(0.5 * x) + 1)
    let y = x.mapv(move |x_val| {
        let tanh_result = (x_val * half).tanh();
        half * (tanh_result + F::one())
    });
    y
}

#[inline]
#[allow(dead_code)]
pub fn softmax_impl<T: Float>(x: &NdArrayView<T>, axis: isize) -> NdArray<T> {
    let axis = if axis < 0 {
        (x.ndim() as isize + axis) as usize
    } else {
        axis as usize
    };

    let mut a = x.shape().to_vec();
    a[axis] = 1;
    let reducedshape = a.as_slice();
    let max_fn = T::max;
    // unwrap is safe
    let max = &x
        .fold_axis(
            scirs2_core::ndarray::Axis(axis),
            T::min_value(),
            move |&a, &b| max_fn(a, b),
        )
        .into_shape_with_order(scirs2_core::ndarray::IxDyn(reducedshape))
        .expect("Failed to create array");
    // subtract `max` to prevent overflow
    let mut tmp = x - max;
    tmp.mapv_inplace(move |a| a.exp());
    // unwrap is safe
    let sum = tmp
        .sum_axis(scirs2_core::ndarray::Axis(axis))
        .into_shape_with_order(scirs2_core::ndarray::IxDyn(reducedshape))
        .expect("Failed to create array");
    tmp /= &sum;
    tmp
}

impl<T: Float> op::Op<T> for Softmax {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let ret = softmax_impl(&ctx.input(0), self.axis);
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        let y = ctx.output();
        let gy = ctx.output_grad();
        let sum = reduce_sum(y * gy, &[self.axis], true);
        ctx.append_input_grad(0, Some((gy - sum) * y))
    }
}

impl<T: Float> op::Op<T> for Softplus {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let ret = ctx.input(0).map(move |a| (a.exp() + T::one()).ln());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        let gy = ctx.output_grad();
        let a = exp(ctx.input(0));
        let b = a + scalar(T::one(), ctx.graph());
        let gx = gy * (a / b);
        ctx.append_input_grad(0, Some(gx))
    }
}

impl<T: Float> op::Op<T> for Sigmoid {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let half = T::from(0.5).expect("Operation failed");
        let ret = ctx
            .input(0)
            .mapv(move |a| ((a * half).tanh() * half) + half);
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        let gy = ctx.output_grad();
        let y = ctx.output();
        ctx.append_input_grad(0, Some(gy * (y - square(y))));
    }
}

impl<T: Float> op::Op<T> for ReLU {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let ret = ctx.input(0).map(|a| a.max(T::zero()));
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        let s = ctx.graph();
        let gy = ctx.output_grad();
        let bin = greater(ctx.input(0), scalar(T::zero(), s));
        ctx.append_input_grad(0, Some(mul(bin, gy)))
    }
}

impl<T: Float> op::Op<T> for Identity {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        // do nothing
        let ret = ctx.input(0);
        ctx.append_output(ret.to_owned());
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        let gy = ctx.output_grad();
        ctx.append_input_grad(0, Some(gy.to_owned()))
    }
}

impl<T: Float> op::Op<T> for Elu<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let ret = ctx.input(0).mapv(move |a| {
            if a > T::zero() {
                a
            } else {
                self.alpha * (a.exp() - T::one())
            }
        });
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        let gy = &ctx.output_grad();
        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.input(0), false)
            .append_input(gy, false)
            .setshape(&shape(gy))
            .build(EluGrad { alpha: self.alpha });
        ctx.append_input_grad(0, Some(gx))
    }
}

impl<T: Float> op::Op<T> for EluGrad<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let a = x.mapv(move |a| {
            if a > T::zero() {
                T::one()
            } else {
                self.alpha * (a.exp() - T::one()) + self.alpha
            }
        });
        let ret = a * &ctx.input(1);
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

/// Swish activation function: x * sigmoid(x)
pub struct Swish;

impl<T: Float> op::Op<T> for Swish {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        // Compute sigmoid(x) first
        let half = T::from(0.5).expect("Operation failed");
        let sigmoid_x = x.mapv(move |a| ((a * half).tanh() * half) + half);
        // Swish = x * sigmoid(x)
        let ret = x * &sigmoid_x;
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        let gy = ctx.output_grad();
        let x = ctx.input(0);

        // Compute sigmoid(x)
        let sigmoid_x = sigmoid(x);

        // Derivative of swish: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        // = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        let one = scalar(T::one(), ctx.graph());
        let grad_factor = sigmoid_x * (one + x * (scalar(T::one(), ctx.graph()) - sigmoid_x));

        ctx.append_input_grad(0, Some(gy * grad_factor));
    }
}

/// GELU (Gaussian Error Linear Unit) activation function
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
pub struct Gelu;

impl<T: Float> op::Op<T> for Gelu {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);

        // Constants
        let half = T::from(0.5).expect("Operation failed");
        let sqrt_2_pi = T::from(0.7978845608028654).expect("Operation failed"); // sqrt(2/π)
        let c = T::from(0.044715).expect("Operation failed");
        let one = T::one();

        // Inner expression: sqrt(2/π) * (x + 0.044715 * x³)
        let inner = x.mapv(|val| sqrt_2_pi * (val + c * val * val * val));

        // tanh of inner expression
        let tanh_inner = inner.mapv(|a| a.tanh());

        // Final GELU: 0.5 * x * (1 + tanh(inner))
        let ret = x.mapv(|val| val * half) * &tanh_inner.mapv(|a| one + a);

        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        let gy = ctx.output_grad();
        let x = ctx.input(0);

        // For gradient computation, we use the derivative formula
        // This is a simplified approximation
        let half = scalar(T::from(0.5).expect("Operation failed"), ctx.graph());
        let sqrt_2_pi = scalar(
            T::from(0.7978845608028654).expect("Operation failed"),
            ctx.graph(),
        );
        let c = scalar(T::from(0.044715).expect("Operation failed"), ctx.graph());
        let one = scalar(T::one(), ctx.graph());

        // Approximation: use tanh derivative for gradient
        let x_squared = square(x);
        let x_cubed = x * x_squared;
        let inner = sqrt_2_pi * (x + c * x_cubed);
        let tanh_inner = tanh(inner);
        let sech_squared = one - square(tanh_inner);

        // Gradient approximation
        let grad = half
            * (one
                + tanh_inner
                + x * sqrt_2_pi
                    * sech_squared
                    * (one
                        + scalar(T::from(3.0).expect("Operation failed"), ctx.graph())
                            * c
                            * x_squared));

        ctx.append_input_grad(0, Some(gy * grad));
    }
}

/// Mish activation function: x * tanh(softplus(x))
/// Mish(x) = x * tanh(ln(1 + exp(x)))
pub struct Mish;

impl<T: Float> op::Op<T> for Mish {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);

        // Compute softplus(x) = ln(1 + exp(x))
        let softplus_x = x.mapv(move |a| {
            // Use log1p for numerical stability when possible
            if a > T::from(20.0).expect("Operation failed") {
                // For large x, softplus(x) ≈ x
                a
            } else {
                (a.exp() + T::one()).ln()
            }
        });

        // Compute tanh(softplus(x))
        let tanh_softplus = softplus_x.mapv(|a| a.tanh());

        // Mish = x * tanh(softplus(x))
        let ret = x * &tanh_softplus;
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        let gy = ctx.output_grad();
        let x = ctx.input(0);

        // For gradient, we compute d/dx[x * tanh(softplus(x))]
        // This involves the product rule and chain rule

        // Compute softplus and its components
        let exp_x = exp(x);
        let one = scalar(T::one(), ctx.graph());
        let softplus_x = ln(one + exp_x);
        let tanh_softplus = tanh(softplus_x);

        // Derivative of softplus: sigmoid(x)
        let sigmoid_x = exp_x / (one + exp_x);

        // Derivative of tanh: sech²(x) = 1 - tanh²(x)
        let sech_squared = one - square(tanh_softplus);

        // Complete derivative: tanh(softplus(x)) + x * sech²(softplus(x)) * sigmoid(x)
        let grad = tanh_softplus + x * sech_squared * sigmoid_x;

        ctx.append_input_grad(0, Some(gy * grad));
    }
}

/// Parametric ReLU (PReLU) activation function
/// PReLU(x) = x if x > 0, else alpha * x
/// where alpha is a learnable parameter
pub struct PReLU<T> {
    pub alpha: T,
}

impl<T: Float> op::Op<T> for PReLU<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = ctx.input(0);
        let ret = x.mapv(|val| {
            if val > T::zero() {
                val
            } else {
                self.alpha * val
            }
        });
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        let gy = ctx.output_grad();
        let x = ctx.input(0);
        let g = ctx.graph();

        // Gradient w.r.t. x: 1 if x > 0, else alpha
        let grad_x = Tensor::builder(g)
            .append_input(x, false)
            .append_input(gy, false)
            .setshape(&shape(gy))
            .build(PReLUGrad { alpha: self.alpha });

        ctx.append_input_grad(0, Some(grad_x));
    }
}

/// Gradient operation for PReLU
pub struct PReLUGrad<T> {
    pub alpha: T,
}

impl<T: Float> op::Op<T> for PReLUGrad<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = ctx.input(0);
        let gy = ctx.input(1);
        let ret = x.mapv(|val| {
            if val > T::zero() {
                T::one()
            } else {
                self.alpha
            }
        }) * gy;
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        // PReLU''(x) = 0 everywhere (piecewise-constant first derivative).
        // So grad w.r.t. x = output_grad * 0 = None.
        // Grad w.r.t. gy = output_grad * PReLU'(x) — reuse PReLUGrad with new gy.
        let gy_out = ctx.output_grad();
        let x = ctx.input(0);
        let g = ctx.graph();

        let grad_gy = Tensor::builder(g)
            .append_input(x, false)
            .append_input(gy_out, false)
            .setshape(&shape(gy_out))
            .build(PReLUGrad { alpha: self.alpha });

        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, Some(grad_gy));
    }
}

/// Second-order gradient operation for LearnableELU.
///
/// Computes `output_grad * f''(x) * gy` where `f''(x) = alpha * exp(x)` if x < 0, else 0.
/// This is built by `LearnableELUGrad::grad` and placed on the x-input path.
pub struct LearnableELUGradGrad<T> {
    pub alpha: T,
}

impl<T: Float> op::Op<T> for LearnableELUGradGrad<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        // inputs: 0 = x, 1 = gy (original upstream), 2 = gy_out (second-order upstream)
        let x = ctx.input(0);
        let gy = ctx.input(1);
        let gy_out = ctx.input(2);
        // f''(x) = alpha * exp(x) if x < 0, else 0
        let d2 = x.mapv(|val| {
            if val > T::zero() {
                T::zero()
            } else {
                self.alpha * val.exp()
            }
        });
        let ret = d2 * gy * gy_out;
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        // Third-order gradients not required; stop the chain.
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
        ctx.append_input_grad(2, None);
    }
}

/// Second-order gradient (w.r.t. x) for LearnableSwish.
///
/// LearnableSwish(x) = x * σ(β·x), where σ is sigmoid.
/// Let s = σ(β·x), s' = β·s·(1-s).
/// First derivative:  f'(x) = s + x·s'
/// Second derivative: f''(x) = 2·s' + x·β·s'·(1 - 2s)
///                           = β·s·(1-s)·(2 + β·x·(1-2s))
///
/// This op computes `output_grad * f''(x) * gy`.
pub struct LearnableSwishGradGrad<T> {
    pub beta: T,
}

impl<T: Float> op::Op<T> for LearnableSwishGradGrad<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        // inputs: 0 = x, 1 = gy (original upstream), 2 = gy_out (second-order upstream)
        let x = ctx.input(0);
        let gy = ctx.input(1);
        let gy_out = ctx.input(2);
        let half = T::from(0.5).expect("constant conversion");
        let one = T::one();
        let two = T::from(2.0).expect("constant conversion");

        let d2 = x.mapv(|val| {
            let bx = self.beta * val;
            // s = sigmoid(beta * x) using tanh identity
            let s = (bx * half).tanh() * half + half;
            let s_deriv = self.beta * s * (one - s); // β·s·(1-s)
                                                     // f''(x) = 2·s' + x·β·s'·(1 - 2s) = s'·(2 + β·x·(1 - 2s))
            s_deriv * (two + self.beta * val * (one - two * s))
        });
        let ret = d2 * gy * gy_out;
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
        ctx.append_input_grad(2, None);
    }
}

/// Second-order gradient (w.r.t. x) for AdaptiveActivation.
///
/// AdaptiveActivation(x) = a·x + b·tanh(c·x) + d·σ(e·x).
/// f'(x) = a + b·c·sech²(c·x) + d·e·σ_e·(1-σ_e)
/// f''(x) = -2·b·c²·tanh(c·x)·sech²(c·x)
///         + d·e²·σ_e·(1-σ_e)·(1-2σ_e)
///
/// This op computes `output_grad * f''(x) * gy`.
pub struct AdaptiveActivationGradGrad<T> {
    pub b: T,
    pub c: T,
    pub d: T,
    pub e: T,
}

impl<T: Float> op::Op<T> for AdaptiveActivationGradGrad<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        // inputs: 0 = x, 1 = gy (original upstream), 2 = gy_out (second-order upstream)
        let x = ctx.input(0);
        let gy = ctx.input(1);
        let gy_out = ctx.input(2);
        let half = T::from(0.5).expect("constant conversion");
        let one = T::one();
        let two = T::from(2.0).expect("constant conversion");

        let d2 = x.mapv(|val| {
            // tanh branch
            let cx = self.c * val;
            let t = cx.tanh();
            let sech2 = one - t * t; // sech²(c·x)
            let tanh_term = -two * self.b * self.c * self.c * t * sech2;

            // sigmoid branch
            let ex = self.e * val;
            let s = (ex * half).tanh() * half + half; // σ(e·x)
            let sig_term = self.d * self.e * self.e * s * (one - s) * (one - two * s);

            tanh_term + sig_term
        });
        let ret = d2 * gy * gy_out;
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
        ctx.append_input_grad(2, None);
    }
}

/// Learnable ELU activation function
/// ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
/// where alpha is a learnable parameter
pub struct LearnableELU<T> {
    pub alpha: T,
}

impl<T: Float> op::Op<T> for LearnableELU<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = ctx.input(0);
        let ret = x.mapv(|val| {
            if val > T::zero() {
                val
            } else {
                self.alpha * (val.exp() - T::one())
            }
        });
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        let gy = ctx.output_grad();
        let x = ctx.input(0);
        let g = ctx.graph();

        // Gradient w.r.t. x: 1 if x > 0, else alpha * exp(x)
        let grad_x = Tensor::builder(g)
            .append_input(x, false)
            .append_input(gy, false)
            .setshape(&shape(gy))
            .build(LearnableELUGrad { alpha: self.alpha });

        ctx.append_input_grad(0, Some(grad_x));
    }
}

/// Gradient operation for LearnableELU
pub struct LearnableELUGrad<T> {
    pub alpha: T,
}

impl<T: Float> op::Op<T> for LearnableELUGrad<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = ctx.input(0);
        let gy = ctx.input(1);
        let ret = x.mapv(|val| {
            if val > T::zero() {
                T::one()
            } else {
                self.alpha * val.exp()
            }
        }) * gy;
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        // LearnableELU''(x) = alpha * exp(x) if x < 0, else 0.
        // Grad w.r.t. x   = output_grad * LearnableELU''(x) * gy  → LearnableELUGradGrad
        // Grad w.r.t. gy  = output_grad * LearnableELU'(x)        → re-apply LearnableELUGrad
        let gy_out = ctx.output_grad();
        let x = ctx.input(0);
        let gy = ctx.input(1);
        let g = ctx.graph();

        // grad w.r.t. x: uses LearnableELUGradGrad (second derivative times gy times gy_out)
        let grad_x = Tensor::builder(g)
            .append_input(x, false)
            .append_input(gy, false)
            .append_input(gy_out, false)
            .setshape(&shape(gy_out))
            .build(LearnableELUGradGrad { alpha: self.alpha });

        // grad w.r.t. gy: re-apply LearnableELUGrad with gy_out as new upstream
        let grad_gy = Tensor::builder(g)
            .append_input(x, false)
            .append_input(gy_out, false)
            .setshape(&shape(gy_out))
            .build(LearnableELUGrad { alpha: self.alpha });

        ctx.append_input_grad(0, Some(grad_x));
        ctx.append_input_grad(1, Some(grad_gy));
    }
}

/// Learnable Swish activation function
/// Swish(x) = x * sigmoid(beta * x)
/// where beta is a learnable parameter (typically initialized to 1.0)
pub struct LearnableSwish<T> {
    pub beta: T,
}

impl<T: Float> op::Op<T> for LearnableSwish<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = ctx.input(0);
        let half = T::from(0.5).expect("Operation failed");

        // Compute sigmoid(beta * x)
        let beta_x = x.mapv(|val| self.beta * val);
        let sigmoid_beta_x = beta_x.mapv(|val| ((val * half).tanh() * half) + half);

        // Swish = x * sigmoid(beta * x)
        let ret = &x.to_owned() * &sigmoid_beta_x;
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        let gy = ctx.output_grad();
        let x = ctx.input(0);
        let g = ctx.graph();

        // Gradient computation for learnable swish
        let grad_x = Tensor::builder(g)
            .append_input(x, false)
            .append_input(gy, false)
            .setshape(&shape(gy))
            .build(LearnableSwishGrad { beta: self.beta });

        ctx.append_input_grad(0, Some(grad_x));
    }
}

/// Gradient operation for LearnableSwish
pub struct LearnableSwishGrad<T> {
    pub beta: T,
}

impl<T: Float> op::Op<T> for LearnableSwishGrad<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = ctx.input(0);
        let gy = ctx.input(1);
        let half = T::from(0.5).expect("Operation failed");

        // Compute sigmoid(beta * x)
        let beta_x = x.mapv(|val| self.beta * val);
        let sigmoid_beta_x = beta_x.mapv(|val| ((val * half).tanh() * half) + half);

        // Derivative: sigmoid(beta * x) + x * beta * sigmoid(beta * x) * (1 - sigmoid(beta * x))
        let one = T::one();
        let derivative = sigmoid_beta_x.mapv(|s_val| s_val)
            + x.mapv(|x_val| x_val)
                * sigmoid_beta_x.mapv(|s_val| self.beta * s_val * (one - s_val));

        let ret = derivative * gy;
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        // LearnableSwish''(x) = β·s·(1-s)·(2 + β·x·(1-2s)), where s = σ(β·x).
        // Grad w.r.t. x  = output_grad * swish''(x) * gy  → LearnableSwishGradGrad
        // Grad w.r.t. gy = output_grad * swish'(x)        → re-apply LearnableSwishGrad
        let gy_out = ctx.output_grad();
        let x = ctx.input(0);
        let gy = ctx.input(1);
        let g = ctx.graph();

        let grad_x = Tensor::builder(g)
            .append_input(x, false)
            .append_input(gy, false)
            .append_input(gy_out, false)
            .setshape(&shape(gy_out))
            .build(LearnableSwishGradGrad { beta: self.beta });

        let grad_gy = Tensor::builder(g)
            .append_input(x, false)
            .append_input(gy_out, false)
            .setshape(&shape(gy_out))
            .build(LearnableSwishGrad { beta: self.beta });

        ctx.append_input_grad(0, Some(grad_x));
        ctx.append_input_grad(1, Some(grad_gy));
    }
}

/// Adaptive activation function with learnable parameters
/// AdaAct(x) = a * x + b * tanh(c * x) + d * sigmoid(e * x)
/// where a, b, c, d, e are learnable parameters
pub struct AdaptiveActivation<T> {
    pub a: T, // linear coefficient
    pub b: T, // tanh coefficient
    pub c: T, // tanh scaling
    pub d: T, // sigmoid coefficient
    pub e: T, // sigmoid scaling
}

impl<T: Float> op::Op<T> for AdaptiveActivation<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = ctx.input(0);
        let half = T::from(0.5).expect("Operation failed");

        // Compute each component
        let linear_part = x.mapv(|val| self.a * val);
        let tanh_part = x.mapv(|val| self.b * (self.c * val).tanh());
        let sigmoid_part = x.mapv(|val| {
            let sigmoid_val = ((self.e * val * half).tanh() * half) + half;
            self.d * sigmoid_val
        });

        // Combine all components
        let ret = linear_part + &tanh_part + &sigmoid_part;
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        let gy = ctx.output_grad();
        let x = ctx.input(0);
        let g = ctx.graph();

        // Gradient computation for adaptive activation
        let grad_x = Tensor::builder(g)
            .append_input(x, false)
            .append_input(gy, false)
            .setshape(&shape(gy))
            .build(AdaptiveActivationGrad {
                a: self.a,
                b: self.b,
                c: self.c,
                d: self.d,
                e: self.e,
            });

        ctx.append_input_grad(0, Some(grad_x));
    }
}

/// Gradient operation for AdaptiveActivation
pub struct AdaptiveActivationGrad<T> {
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
    pub e: T,
}

impl<T: Float> op::Op<T> for AdaptiveActivationGrad<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = ctx.input(0);
        let gy = ctx.input(1);
        let half = T::from(0.5).expect("Operation failed");
        let one = T::one();

        // Compute derivatives of each component
        let linear_grad = x.mapv(|_| self.a);

        let tanh_grad = x.mapv(|val| {
            let tanh_val = (self.c * val).tanh();
            self.b * self.c * (one - tanh_val * tanh_val)
        });

        let sigmoid_grad = x.mapv(|val| {
            let sigmoid_val = ((self.e * val * half).tanh() * half) + half;
            self.d * self.e * sigmoid_val * (one - sigmoid_val)
        });

        // Total gradient
        let total_grad = linear_grad + &tanh_grad + &sigmoid_grad;
        let ret = total_grad * gy;
        ctx.append_output(ret);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        // AdaptiveActivation''(x) = -2b·c²·tanh(c·x)·sech²(c·x) + d·e²·σ_e·(1-σ_e)·(1-2σ_e)
        // Grad w.r.t. x  = output_grad * f''(x) * gy  → AdaptiveActivationGradGrad
        // Grad w.r.t. gy = output_grad * f'(x)        → re-apply AdaptiveActivationGrad
        let gy_out = ctx.output_grad();
        let x = ctx.input(0);
        let gy = ctx.input(1);
        let g = ctx.graph();

        let grad_x = Tensor::builder(g)
            .append_input(x, false)
            .append_input(gy, false)
            .append_input(gy_out, false)
            .setshape(&shape(gy_out))
            .build(AdaptiveActivationGradGrad {
                b: self.b,
                c: self.c,
                d: self.d,
                e: self.e,
            });

        let grad_gy = Tensor::builder(g)
            .append_input(x, false)
            .append_input(gy_out, false)
            .setshape(&shape(gy_out))
            .build(AdaptiveActivationGrad {
                a: self.a,
                b: self.b,
                c: self.c,
                d: self.d,
                e: self.e,
            });

        ctx.append_input_grad(0, Some(grad_x));
        ctx.append_input_grad(1, Some(grad_gy));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests — second-derivative formulas validated against central differences
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    // Central-difference step size
    const H: f64 = 1e-4;

    // Tolerance for central-difference comparison (h=1e-4, smooth activations → error ~1e-8)
    const TOL: f64 = 1e-5;

    fn central_diff2(f: impl Fn(f64) -> f64, x: f64) -> f64 {
        (f(x + H) - 2.0 * f(x) + f(x - H)) / (H * H)
    }

    // ── Helpers for the activation forward passes ────────────────────────────

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn prelu_forward(x: f64, alpha: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            alpha * x
        }
    }

    fn learnable_elu_forward(x: f64, alpha: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            alpha * (x.exp() - 1.0)
        }
    }

    fn learnable_swish_forward(x: f64, beta: f64) -> f64 {
        x * sigmoid(beta * x)
    }

    fn adaptive_activation_forward(x: f64, a: f64, b: f64, c: f64, d: f64, e: f64) -> f64 {
        a * x + b * (c * x).tanh() + d * sigmoid(e * x)
    }

    // ── Analytical second derivatives (same formulas as the *GradGrad ops) ──

    fn prelu_d2(_x: f64, _alpha: f64) -> f64 {
        // PReLU' is piecewise constant → second derivative is 0 everywhere
        0.0
    }

    fn learnable_elu_d2(x: f64, alpha: f64) -> f64 {
        if x > 0.0 {
            0.0
        } else {
            alpha * x.exp()
        }
    }

    fn learnable_swish_d2(x: f64, beta: f64) -> f64 {
        let s = sigmoid(beta * x);
        let s_deriv = beta * s * (1.0 - s);
        s_deriv * (2.0 + beta * x * (1.0 - 2.0 * s))
    }

    fn adaptive_activation_d2(x: f64, b: f64, c: f64, d: f64, e: f64) -> f64 {
        let t = (c * x).tanh();
        let sech2 = 1.0 - t * t;
        let tanh_term = -2.0 * b * c * c * t * sech2;

        let s = sigmoid(e * x);
        let sig_term = d * e * e * s * (1.0 - s) * (1.0 - 2.0 * s);

        tanh_term + sig_term
    }

    fn check_close(analytical: f64, numerical: f64, label: &str, x: f64) {
        let err = (analytical - numerical).abs();
        assert!(
            err < TOL,
            "{label} at x={x}: analytical={analytical:.8}, numerical={numerical:.8}, err={err:.2e}"
        );
    }

    // ── PReLU second derivative ──────────────────────────────────────────────

    #[test]
    fn test_prelu_second_derivative() {
        let alpha = 0.1f64;
        // Skip x=0 (kink) — central difference is unreliable there
        for &x in &[-2.0f64, -0.5, 0.5, 2.0] {
            let analytical = prelu_d2(x, alpha);
            let numerical = central_diff2(|v| prelu_forward(v, alpha), x);
            check_close(analytical, numerical, "PReLU d2", x);
        }
    }

    // ── LearnableELU second derivative ──────────────────────────────────────

    #[test]
    fn test_learnable_elu_second_derivative_alpha1() {
        let alpha = 1.0f64;
        for &x in &[-2.0f64, -0.5, 0.5, 2.0] {
            let analytical = learnable_elu_d2(x, alpha);
            let numerical = central_diff2(|v| learnable_elu_forward(v, alpha), x);
            check_close(analytical, numerical, "LearnableELU(α=1) d2", x);
        }
    }

    #[test]
    fn test_learnable_elu_second_derivative_alpha05() {
        let alpha = 0.5f64;
        for &x in &[-2.0f64, -1.0, -0.5] {
            let analytical = learnable_elu_d2(x, alpha);
            let numerical = central_diff2(|v| learnable_elu_forward(v, alpha), x);
            check_close(analytical, numerical, "LearnableELU(α=0.5) d2", x);
        }
    }

    // ── LearnableSwish second derivative ─────────────────────────────────────

    #[test]
    fn test_learnable_swish_second_derivative_beta1() {
        let beta = 1.0f64;
        for &x in &[-2.0f64, -0.5, 0.0, 0.5, 2.0] {
            let analytical = learnable_swish_d2(x, beta);
            let numerical = central_diff2(|v| learnable_swish_forward(v, beta), x);
            check_close(analytical, numerical, "LearnableSwish(β=1) d2", x);
        }
    }

    #[test]
    fn test_learnable_swish_second_derivative_beta2() {
        let beta = 2.0f64;
        for &x in &[-2.0f64, -0.5, 0.0, 0.5, 2.0] {
            let analytical = learnable_swish_d2(x, beta);
            let numerical = central_diff2(|v| learnable_swish_forward(v, beta), x);
            check_close(analytical, numerical, "LearnableSwish(β=2) d2", x);
        }
    }

    // ── AdaptiveActivation second derivative ─────────────────────────────────

    #[test]
    fn test_adaptive_activation_second_derivative() {
        let (a, b, c, d, e) = (1.0f64, 0.5, 2.0, 0.3, 1.5);
        for &x in &[-2.0f64, -0.5, 0.0, 0.5, 2.0] {
            let analytical = adaptive_activation_d2(x, b, c, d, e);
            let numerical = central_diff2(|v| adaptive_activation_forward(v, a, b, c, d, e), x);
            check_close(analytical, numerical, "AdaptiveActivation d2", x);
        }
    }

    #[test]
    fn test_adaptive_activation_second_derivative_linear() {
        // With b=0, d=0 the second derivative should be 0 everywhere (pure linear).
        let (a, b, c, d, e) = (1.0f64, 0.0, 1.0, 0.0, 1.0);
        for &x in &[-2.0f64, -0.5, 0.0, 0.5, 2.0] {
            let analytical = adaptive_activation_d2(x, b, c, d, e);
            assert!(
                analytical.abs() < 1e-12,
                "Linear AdaptiveActivation d2 should be 0, got {analytical} at x={x}"
            );
            let numerical = central_diff2(|v| adaptive_activation_forward(v, a, b, c, d, e), x);
            check_close(analytical, numerical, "AdaptiveActivation(linear) d2", x);
        }
    }
}
