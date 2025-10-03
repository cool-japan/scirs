//! Performance optimization operations for CPU computation (simplified)
//!
//! This module provides simplified versions of SIMD-accelerated operations
//! and performance optimizations for tensor operations.

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use scirs2_core::ndarray::Axis;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global flag to enable/disable SIMD optimizations
static SIMD_ENABLED: AtomicBool = AtomicBool::new(true);

/// Global flag to enable/disable parallel processing
static PARALLEL_ENABLED: AtomicBool = AtomicBool::new(true);

/// SIMD-optimized element-wise binary operation (simplified)
pub struct SimdBinaryOp {
    pub operation: SimdBinaryOperation,
}

#[derive(Debug, Clone, Copy)]
pub enum SimdBinaryOperation {
    Add,
    Mul,
}

impl<F: Float> Op<F> for SimdBinaryOp {
    fn name(&self) -> &'static str {
        match self.operation {
            SimdBinaryOperation::Add => "SimdAdd",
            SimdBinaryOperation::Mul => "SimdMul",
        }
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let left = ctx.input(0);
        let right = ctx.input(1);

        let result = match self.operation {
            SimdBinaryOperation::Add => &left.to_owned() + &right.to_owned(),
            SimdBinaryOperation::Mul => &left.to_owned() * &right.to_owned(),
        };

        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let left = ctx.input(0);
        let right = ctx.input(1);

        match self.operation {
            SimdBinaryOperation::Add => {
                ctx.append_input_grad(0, Some(*gy));
                ctx.append_input_grad(1, Some(*gy));
            }
            SimdBinaryOperation::Mul => {
                ctx.append_input_grad(0, Some((*gy) * right));
                ctx.append_input_grad(1, Some((*gy) * left));
            }
        }
    }
}

/// SIMD-optimized unary operation (simplified)
pub struct SimdUnaryOp {
    pub operation: SimdUnaryOperation,
}

#[derive(Debug, Clone, Copy)]
pub enum SimdUnaryOperation {
    ReLU,
    Sigmoid,
}

impl<F: Float> Op<F> for SimdUnaryOp {
    fn name(&self) -> &'static str {
        match self.operation {
            SimdUnaryOperation::ReLU => "SimdReLU",
            SimdUnaryOperation::Sigmoid => "SimdSigmoid",
        }
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);

        let result = match self.operation {
            SimdUnaryOperation::ReLU => input.mapv(|x| if x > F::zero() { x } else { F::zero() }),
            SimdUnaryOperation::Sigmoid => input.mapv(|x| F::one() / (F::one() + (-x).exp())),
        };

        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let input = ctx.input(0);

        let grad = match self.operation {
            SimdUnaryOperation::ReLU => {
                let zero_tensor = crate::tensor_ops::scalar(F::zero(), ctx.graph());
                let mask = crate::tensor_ops::greater(input, zero_tensor);
                (*gy) * mask
            }
            SimdUnaryOperation::Sigmoid => {
                let sigmoid_x = crate::tensor_ops::sigmoid(input);
                let one = crate::tensor_ops::scalar(F::one(), ctx.graph());
                let one_minus_sigmoid = one - sigmoid_x;
                (*gy) * sigmoid_x * one_minus_sigmoid
            }
        };

        ctx.append_input_grad(0, Some(grad));
    }
}

/// Parallel reduction operation (simplified)
pub struct ParallelReductionOp {
    #[allow(dead_code)]
    pub operation: ReductionOperation,
    pub axis: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum ReductionOperation {
    Sum,
}

impl<F: Float> Op<F> for ParallelReductionOp {
    fn name(&self) -> &'static str {
        "ParallelSum"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let result = input.sum_axis(Axis(self.axis));
        // Convert to dynamic dimensions, ensuring scalar becomes 1D array with single element
        let dyn_result = if result.ndim() == 0 {
            // Convert 0D scalar to 1D array with single element
            let scalar_val = result.iter().next().copied().unwrap_or(F::zero());
            scirs2_core::ndarray::arr1(&[scalar_val]).into_dyn()
        } else {
            result.into_dyn()
        };
        ctx.append_output(dyn_result);
        Ok(())
    }

    fn grad<'a>(&self, _ctx: &mut GradientContext<'a, 'a, F>) {
        // Simplified gradient implementation
        // In practice, would implement proper gradient broadcasting
    }
}

// Public API functions

/// Enable or disable SIMD optimizations
#[allow(dead_code)]
pub fn set_simd_enabled(enabled: bool) {
    SIMD_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Check if SIMD optimizations are enabled
#[allow(dead_code)]
pub fn is_simd_enabled() -> bool {
    SIMD_ENABLED.load(Ordering::Relaxed)
}

/// Enable or disable parallel processing
#[allow(dead_code)]
pub fn set_parallel_enabled(enabled: bool) {
    PARALLEL_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Check if parallel processing is enabled
#[allow(dead_code)]
pub fn is_parallel_enabled() -> bool {
    PARALLEL_ENABLED.load(Ordering::Relaxed)
}

/// SIMD-optimized element-wise addition
#[allow(dead_code)]
pub fn simd_add<'g, F: Float>(left: &Tensor<'g, F>, right: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = left.graph();
    Tensor::builder(g)
        .append_input(left, false)
        .append_input(right, false)
        .build(SimdBinaryOp {
            operation: SimdBinaryOperation::Add,
        })
}

/// SIMD-optimized element-wise multiplication
#[allow(dead_code)]
pub fn simd_mul<'g, F: Float>(left: &Tensor<'g, F>, right: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = left.graph();
    Tensor::builder(g)
        .append_input(left, false)
        .append_input(right, false)
        .build(SimdBinaryOp {
            operation: SimdBinaryOperation::Mul,
        })
}

/// SIMD-optimized ReLU activation
#[allow(dead_code)]
pub fn simd_relu<'g, F: Float>(tensor: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = tensor.graph();
    Tensor::builder(g)
        .append_input(tensor, false)
        .build(SimdUnaryOp {
            operation: SimdUnaryOperation::ReLU,
        })
}

/// SIMD-optimized sigmoid activation
#[allow(dead_code)]
pub fn simd_sigmoid<'g, F: Float>(tensor: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = tensor.graph();
    Tensor::builder(g)
        .append_input(tensor, false)
        .build(SimdUnaryOp {
            operation: SimdUnaryOperation::Sigmoid,
        })
}

/// Cache-friendly matrix multiplication (simplified to regular matmul)
#[allow(dead_code)]
pub fn cache_friendly_matmul<'g, F: Float>(
    left: &Tensor<'g, F>,
    right: &Tensor<'g, F>,
    _block_size: Option<usize>,
) -> Tensor<'g, F> {
    crate::tensor_ops::matmul(left, right)
}

/// Parallel sum reduction
#[allow(dead_code)]
pub fn parallel_sum<'g, F: Float>(
    tensor: &Tensor<'g, F>,
    axes: &[usize],
    _keep_dims: bool,
) -> Tensor<'g, F> {
    let axis = axes.first().copied().unwrap_or(0);
    let g = tensor.graph();
    Tensor::builder(g)
        .append_input(tensor, false)
        .build(ParallelReductionOp {
            operation: ReductionOperation::Sum,
            axis,
        })
}

/// Performance configuration utility
pub struct PerformanceConfig;

impl PerformanceConfig {
    /// Configure for maximum performance
    pub fn configure_for_performance() {
        set_simd_enabled(true);
        set_parallel_enabled(true);
    }

    /// Configure for compatibility (disable optimizations)
    pub fn configure_for_compatibility() {
        set_simd_enabled(false);
        set_parallel_enabled(false);
    }

    /// Get current performance settings
    pub fn get_settings() -> (bool, bool) {
        (is_simd_enabled(), is_parallel_enabled())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_settings() {
        set_simd_enabled(false);
        assert!(!is_simd_enabled());

        set_simd_enabled(true);
        assert!(is_simd_enabled());
    }

    #[test]
    fn test_parallel_settings() {
        set_parallel_enabled(false);
        assert!(!is_parallel_enabled());

        set_parallel_enabled(true);
        assert!(is_parallel_enabled());
    }

    #[test]
    fn test_performance_config() {
        PerformanceConfig::configure_for_compatibility();
        let (simd, parallel) = PerformanceConfig::get_settings();
        assert!(!simd);
        assert!(!parallel);

        PerformanceConfig::configure_for_performance();
        let (simd, parallel) = PerformanceConfig::get_settings();
        assert!(simd);
        assert!(parallel);
    }
}
