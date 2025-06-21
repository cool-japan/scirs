use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
#[cfg(feature = "mkl")]
use crate::same_type;
use crate::tensor::Tensor;
#[cfg(feature = "mkl")]
use crate::tensor_ops::blas_ffi::*;
use crate::tensor_ops::*;
use crate::Float;
use ndarray;

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

#[cfg(feature = "mkl")]
fn fast_sigmoid_impl<F: Float>(x: &NdArrayView<F>) -> NdArray<F> {
    let half = F::from(0.5).unwrap();
    unsafe {
        if same_type::<F, f32>() {
            let mut y = x.mapv(move |x| x * half);
            vsTanh(
                y.len() as MklInt,
                y.as_ptr() as *const f32,
                y.as_mut_ptr() as *mut f32,
            );
            y.mapv_inplace(move |x2| half * (x2 + F::one()));
            y
        } else if same_type::<F, f64>() {
            let mut y = x.mapv(move |x| x * half);
            vdTanh(
                y.len() as MklInt,
                y.as_ptr() as *const f64,
                y.as_mut_ptr() as *mut f64,
            );
            y.mapv_inplace(move |x2| half * (x2 + F::one()));
            y
        } else {
            x.mapv(move |a| ((a * half).tanh() * half) + half)
        }
    }
}

#[inline]
pub fn softmax_impl<T: Float>(x: &NdArrayView<T>, axis: isize) -> NdArray<T> {
    let axis = if axis < 0 {
        (x.ndim() as isize + axis) as usize
    } else {
        axis as usize
    };

    let mut a = x.shape().to_vec();
    a[axis] = 1;
    let reduced_shape = a.as_slice();
    let max_fn = T::max;
    // unwrap is safe
    let max = &x
        .fold_axis(ndarray::Axis(axis), T::min_value(), move |&a, &b| {
            max_fn(a, b)
        })
        .into_shape_with_order(ndarray::IxDyn(reduced_shape))
        .unwrap();
    // subtract `max` to prevent overflow
    let mut tmp = x - max;
    #[cfg(feature = "mkl")]
    {
        crate::tensor_ops::math_ops::fast_inplace_exp_impl(&mut tmp);
    }
    #[cfg(not(feature = "mkl"))]
    {
        tmp.mapv_inplace(move |a| a.exp());
    }
    // unwrap is safe
    let sum = tmp
        .sum_axis(ndarray::Axis(axis))
        .into_shape_with_order(ndarray::IxDyn(reduced_shape))
        .unwrap();
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
        let ret;
        #[cfg(feature = "mkl")]
        {
            ret = fast_sigmoid_impl(&ctx.input(0));
        }
        #[cfg(not(feature = "mkl"))]
        {
            let half = T::from(0.5).unwrap();
            ret = ctx
                .input(0)
                .mapv(move |a| ((a * half).tanh() * half) + half);
        }
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
            .set_shape(&shape(gy))
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
        let half = T::from(0.5).unwrap();
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
        let half = T::from(0.5).unwrap();
        let sqrt_2_pi = T::from(0.7978845608028654).unwrap(); // sqrt(2/π)
        let c = T::from(0.044715).unwrap();
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
        let half = scalar(T::from(0.5).unwrap(), ctx.graph());
        let sqrt_2_pi = scalar(T::from(0.7978845608028654).unwrap(), ctx.graph());
        let c = scalar(T::from(0.044715).unwrap(), ctx.graph());
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
                    * (one + scalar(T::from(3.0).unwrap(), ctx.graph()) * c * x_squared));

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
            if a > T::from(20.0).unwrap() {
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
            .set_shape(&shape(gy))
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

    fn grad<'a>(&self, _ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        // Second-order gradients not implemented
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
            .set_shape(&shape(gy))
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

    fn grad<'a>(&self, _ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        // Second-order gradients not implemented
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
        let half = T::from(0.5).unwrap();

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
            .set_shape(&shape(gy))
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
        let half = T::from(0.5).unwrap();

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

    fn grad<'a>(&self, _ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        // Second-order gradients not implemented
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
        let half = T::from(0.5).unwrap();

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
            .set_shape(&shape(gy))
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
        let half = T::from(0.5).unwrap();
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

    fn grad<'a>(&self, _ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        // Second-order gradients not implemented
    }
}
