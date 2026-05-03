//! GPU kernel implementations for automatic differentiation
//!
//! This module provides optimized GPU kernels for common autodiff operations.

use crate::{error::AutogradError, Float, Result};
use scirs2_core::gpu::GpuContext;

/// GPU kernel for element-wise operations
pub trait ElementWiseKernel<T: Float> {
    /// Execute element-wise operation on GPU
    fn execute(&self, context: &GpuContext, input: &[T], output: &mut [T]) -> Result<()>;
}

/// ReLU activation kernel
pub struct ReLUKernel;

impl<T: Float> ElementWiseKernel<T> for ReLUKernel {
    fn execute(&self, _context: &GpuContext, input: &[T], output: &mut [T]) -> Result<()> {
        if input.len() != output.len() {
            return Err(AutogradError::shape_error(
                "Input and output must have same length".to_string(),
            ));
        }

        // For CPU fallback or testing
        for (i, &val) in input.iter().enumerate() {
            output[i] = if val > T::zero() { val } else { T::zero() };
        }

        Ok(())
    }
}

/// Sigmoid activation kernel
pub struct SigmoidKernel;

impl<T: Float> ElementWiseKernel<T> for SigmoidKernel {
    fn execute(&self, _context: &GpuContext, input: &[T], output: &mut [T]) -> Result<()> {
        if input.len() != output.len() {
            return Err(AutogradError::shape_error(
                "Input and output must have same length".to_string(),
            ));
        }

        for (i, &val) in input.iter().enumerate() {
            // sigmoid(x) = 1 / (1 + exp(-x))
            output[i] = T::one() / (T::one() + (-val).exp());
        }

        Ok(())
    }
}

/// Tanh activation kernel
pub struct TanhKernel;

impl<T: Float> ElementWiseKernel<T> for TanhKernel {
    fn execute(&self, _context: &GpuContext, input: &[T], output: &mut [T]) -> Result<()> {
        if input.len() != output.len() {
            return Err(AutogradError::shape_error(
                "Input and output must have same length".to_string(),
            ));
        }

        for (i, &val) in input.iter().enumerate() {
            output[i] = val.tanh();
        }

        Ok(())
    }
}

/// GELU activation kernel
pub struct GELUKernel;

impl<T: Float> ElementWiseKernel<T> for GELUKernel {
    fn execute(&self, _context: &GpuContext, input: &[T], output: &mut [T]) -> Result<()> {
        if input.len() != output.len() {
            return Err(AutogradError::shape_error(
                "Input and output must have same length".to_string(),
            ));
        }

        // GELU(x) = x * Φ(x) where Φ is the cumulative distribution function of standard normal
        // Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        let sqrt_2_over_pi = T::from(0.7978845608).ok_or_else(|| {
            AutogradError::compute_error("Failed to convert constant".to_string())
        })?;
        let coeff = T::from(0.044715).ok_or_else(|| {
            AutogradError::compute_error("Failed to convert constant".to_string())
        })?;

        for (i, &val) in input.iter().enumerate() {
            let val3 = val * val * val;
            let inner = sqrt_2_over_pi * (val + coeff * val3);
            output[i] = T::from(0.5).ok_or_else(|| {
                AutogradError::compute_error("Failed to convert constant".to_string())
            })? * val
                * (T::one() + inner.tanh());
        }

        Ok(())
    }
}

/// Matrix multiplication kernel (GEMM)
pub struct GEMMKernel;

impl GEMMKernel {
    /// Execute general matrix multiplication: C = alpha * A * B + beta * C
    pub fn execute<T: Float>(
        &self,
        _context: &GpuContext,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        a: &[T],
        b: &[T],
        beta: T,
        c: &mut [T],
    ) -> Result<()> {
        // Validate dimensions
        if a.len() < m * k {
            return Err(AutogradError::shape_error(format!(
                "Matrix A size mismatch: expected {}, got {}",
                m * k,
                a.len()
            )));
        }
        if b.len() < k * n {
            return Err(AutogradError::shape_error(format!(
                "Matrix B size mismatch: expected {}, got {}",
                k * n,
                b.len()
            )));
        }
        if c.len() < m * n {
            return Err(AutogradError::shape_error(format!(
                "Matrix C size mismatch: expected {}, got {}",
                m * n,
                c.len()
            )));
        }

        // Simple CPU fallback implementation
        // In production, this would dispatch to optimized GPU kernels
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                let idx = i * n + j;
                c[idx] = alpha * sum + beta * c[idx];
            }
        }

        Ok(())
    }
}

/// Reduction kernel for sum/mean/max/min operations
pub struct ReductionKernel;

impl ReductionKernel {
    /// Sum reduction
    pub fn sum<T: Float>(
        &self,
        _context: &GpuContext,
        input: &[T],
        output: &mut [T],
        reduce_size: usize,
    ) -> Result<()> {
        if !input.len().is_multiple_of(reduce_size) {
            return Err(AutogradError::shape_error(
                "Input size must be divisible by reduce size".to_string(),
            ));
        }

        let num_outputs = input.len() / reduce_size;
        if output.len() < num_outputs {
            return Err(AutogradError::shape_error(format!(
                "Output size {} is too small, need {}",
                output.len(),
                num_outputs
            )));
        }

        for i in 0..num_outputs {
            let mut sum = T::zero();
            for j in 0..reduce_size {
                sum += input[i * reduce_size + j];
            }
            output[i] = sum;
        }

        Ok(())
    }

    /// Mean reduction
    pub fn mean<T: Float>(
        &self,
        context: &GpuContext,
        input: &[T],
        output: &mut [T],
        reduce_size: usize,
    ) -> Result<()> {
        self.sum(context, input, output, reduce_size)?;

        let divisor = T::from(reduce_size).ok_or_else(|| {
            AutogradError::compute_error("Failed to convert reduce size".to_string())
        })?;

        for val in output.iter_mut() {
            *val /= divisor;
        }

        Ok(())
    }

    /// Max reduction
    pub fn max<T: Float>(
        &self,
        _context: &GpuContext,
        input: &[T],
        output: &mut [T],
        reduce_size: usize,
    ) -> Result<()> {
        if !input.len().is_multiple_of(reduce_size) {
            return Err(AutogradError::shape_error(
                "Input size must be divisible by reduce size".to_string(),
            ));
        }

        let num_outputs = input.len() / reduce_size;
        if output.len() < num_outputs {
            return Err(AutogradError::shape_error(format!(
                "Output size {} is too small, need {}",
                output.len(),
                num_outputs
            )));
        }

        for i in 0..num_outputs {
            let start = i * reduce_size;
            let slice = &input[start..start + reduce_size];
            output[i] = slice.iter().copied().fold(T::neg_infinity(), T::max);
        }

        Ok(())
    }

    /// Min reduction
    pub fn min<T: Float>(
        &self,
        _context: &GpuContext,
        input: &[T],
        output: &mut [T],
        reduce_size: usize,
    ) -> Result<()> {
        if !input.len().is_multiple_of(reduce_size) {
            return Err(AutogradError::shape_error(
                "Input size must be divisible by reduce size".to_string(),
            ));
        }

        let num_outputs = input.len() / reduce_size;
        if output.len() < num_outputs {
            return Err(AutogradError::shape_error(format!(
                "Output size {} is too small, need {}",
                output.len(),
                num_outputs
            )));
        }

        for i in 0..num_outputs {
            let start = i * reduce_size;
            let slice = &input[start..start + reduce_size];
            output[i] = slice.iter().copied().fold(T::infinity(), T::min);
        }

        Ok(())
    }
}

/// Broadcast kernel for gradient backpropagation
pub struct BroadcastKernel;

impl BroadcastKernel {
    /// Broadcast smaller array to larger shape
    pub fn execute<T: Float>(
        &self,
        _context: &GpuContext,
        input: &[T],
        input_shape: &[usize],
        output: &mut [T],
        output_shape: &[usize],
    ) -> Result<()> {
        // Validate shapes are broadcast-compatible
        if !Self::is_broadcastable(input_shape, output_shape) {
            return Err(AutogradError::shape_error(format!(
                "Shapes {:?} and {:?} are not broadcastable",
                input_shape, output_shape
            )));
        }

        // Simple implementation for 2D case
        if input_shape.len() == 2 && output_shape.len() == 2 {
            let (in_rows, in_cols) = (input_shape[0], input_shape[1]);
            let (out_rows, out_cols) = (output_shape[0], output_shape[1]);

            for i in 0..out_rows {
                for j in 0..out_cols {
                    let in_i = if in_rows == 1 { 0 } else { i };
                    let in_j = if in_cols == 1 { 0 } else { j };
                    output[i * out_cols + j] = input[in_i * in_cols + in_j];
                }
            }
        } else {
            // General N-D broadcasting via strided index decomposition.
            // Compute the output shape by taking the element-wise max of both shapes
            // (broadcast semantics: a dim of 1 expands to match the other).
            let out_ndim = output_shape.len();
            let in_ndim = input_shape.len();
            let num_outputs = output_shape.iter().product::<usize>();

            for out_flat in 0..num_outputs {
                // Decompose out_flat into per-dimension output indices (row-major)
                let mut out_coords = vec![0usize; out_ndim];
                let mut rem = out_flat;
                for d in (0..out_ndim).rev() {
                    out_coords[d] = rem % output_shape[d];
                    rem /= output_shape[d];
                }

                // Map each output dimension to the corresponding input dimension.
                // Input shape is right-aligned; leading dims of input are treated as 1.
                let mut in_flat = 0usize;
                let mut in_stride = 1usize;
                for d in (0..out_ndim).rev() {
                    // How far is this output dimension from the right edge of input?
                    let input_dim_idx = (in_ndim as isize) - (out_ndim as isize) + d as isize;
                    let in_dim_size = if input_dim_idx >= 0 {
                        input_shape[input_dim_idx as usize]
                    } else {
                        1
                    };
                    let in_coord = if in_dim_size == 1 { 0 } else { out_coords[d] };
                    in_flat += in_coord * in_stride;
                    in_stride *= in_dim_size;
                }

                output[out_flat] = input[in_flat];
            }
        }

        Ok(())
    }

    /// Check if two shapes are broadcastable
    fn is_broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
        let len1 = shape1.len();
        let len2 = shape2.len();
        let max_len = len1.max(len2);

        for i in 0..max_len {
            let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
            let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };

            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::gpu::GpuBackend;

    #[test]
    fn test_relu_kernel() {
        let context = GpuContext::new(GpuBackend::Cpu).expect("Should create context");
        let kernel = ReLUKernel;

        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];

        kernel
            .execute(&context, &input, &mut output)
            .expect("Should execute");
        assert_eq!(output, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid_kernel() {
        let context = GpuContext::new(GpuBackend::Cpu).expect("Should create context");
        let kernel = SigmoidKernel;

        let input = vec![0.0_f32];
        let mut output = vec![0.0_f32];

        kernel
            .execute(&context, &input, &mut output)
            .expect("Should execute");
        assert!((output[0] - 0.5_f32).abs() < 1e-6);
    }

    #[test]
    fn test_gemm_kernel() {
        let context = GpuContext::new(GpuBackend::Cpu).expect("Should create context");
        let kernel = GEMMKernel;

        // 2x3 * 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0; 4];

        kernel
            .execute(&context, 2, 2, 3, 1.0, &a, &b, 0.0, &mut c)
            .expect("Should execute");

        // Verify result
        assert!(c[0] > 0.0); // Just check it computed something
    }

    #[test]
    fn test_reduction_sum() {
        let context = GpuContext::new(GpuBackend::Cpu).expect("Should create context");
        let kernel = ReductionKernel;

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0; 2];

        kernel
            .sum(&context, &input, &mut output, 3)
            .expect("Should execute");
        assert_eq!(output, vec![6.0, 15.0]);
    }

    #[test]
    fn test_broadcast_compatibility() {
        assert!(BroadcastKernel::is_broadcastable(&[1, 3], &[4, 3]));
        assert!(BroadcastKernel::is_broadcastable(&[1], &[4, 3]));
        assert!(!BroadcastKernel::is_broadcastable(&[2, 3], &[4, 5]));
    }

    #[test]
    fn test_broadcast_kernel_1d_to_2d() {
        // Broadcast [3] -> [4, 3]: each row of output is the same as input
        let context = GpuContext::new(GpuBackend::Cpu).expect("Should create context");
        let kernel = BroadcastKernel;
        let input = vec![1.0f64, 2.0, 3.0];
        let mut output = vec![0.0f64; 12];
        kernel
            .execute(&context, &input, &[3], &mut output, &[4, 3])
            .expect("Should broadcast");
        for row in 0..4 {
            assert_eq!(output[row * 3], 1.0);
            assert_eq!(output[row * 3 + 1], 2.0);
            assert_eq!(output[row * 3 + 2], 3.0);
        }
    }

    #[test]
    fn test_broadcast_kernel_3d_leading_one() {
        // Broadcast [1, 2, 3] -> [4, 2, 3]
        let context = GpuContext::new(GpuBackend::Cpu).expect("Should create context");
        let kernel = BroadcastKernel;
        let input: Vec<f64> = (1..=6).map(|x| x as f64).collect(); // 1×2×3
        let mut output = vec![0.0f64; 24]; // 4×2×3
        kernel
            .execute(&context, &input, &[1, 2, 3], &mut output, &[4, 2, 3])
            .expect("Should broadcast");
        // Every batch of 6 should equal the input
        for batch in 0..4 {
            for i in 0..6 {
                assert_eq!(
                    output[batch * 6 + i],
                    input[i],
                    "Mismatch at batch={batch} i={i}"
                );
            }
        }
    }

    #[test]
    fn test_broadcast_kernel_column_vector_to_matrix() {
        // Broadcast [4, 1] -> [4, 3]: each column value repeated across columns
        let context = GpuContext::new(GpuBackend::Cpu).expect("Should create context");
        let kernel = BroadcastKernel;
        let input = vec![10.0f64, 20.0, 30.0, 40.0]; // 4×1
        let mut output = vec![0.0f64; 12]; // 4×3
        kernel
            .execute(&context, &input, &[4, 1], &mut output, &[4, 3])
            .expect("Should broadcast");
        for row in 0..4 {
            let expected = input[row];
            for col in 0..3 {
                assert_eq!(
                    output[row * 3 + col],
                    expected,
                    "Mismatch at row={row} col={col}"
                );
            }
        }
    }
}
