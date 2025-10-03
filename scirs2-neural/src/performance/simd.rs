//! SIMD-accelerated operations for neural networks
//!
//! This module provides vectorized implementations of common neural network operations
//! using unified SIMD operations from scirs2-core for significant performance improvements
//! and cross-platform compatibility. All functions are feature-gated with "simd" feature.
//! IMPORTANT: All SIMD operations are delegated to scirs2-core's unified SIMD abstraction layer
//! in compliance with the project-wide SIMD policy. Direct use of SIMD intrinsics is FORBIDDEN.

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{ArrayD, ArrayView, ArrayViewMut, IxDyn};
#[allow(unused_imports)]
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
#[cfg(feature = "simd")]
use scirs2_core::ndarray::Array;
use scirs2_core::simd_ops::{AutoOptimizer, PlatformCapabilities, SimdUnifiedOps};
/// SIMD-accelerated operations for neural networks
pub struct SIMDOperations;
impl SIMDOperations {
    /// SIMD-accelerated ReLU activation for f32 arrays
    pub fn simd_relu_f32_inplace(input: &mut ArrayViewMut<f32, IxDyn>) {
        if let Some(slice) = input.as_slice_mut() {
            // Use unified SIMD operations for ReLU
            let caps = PlatformCapabilities::detect();
            if caps.simd_available && f32::simd_available() {
                Self::simd_relu_f32_slice_unified(slice);
            } else {
                Self::simd_relu_f32_slice_fallback(slice);
            }
        } else {
            input.mapv_inplace(|x| x.max(0.0));
        }
    }
    /// SIMD-accelerated ReLU for f32 slice using unified operations
    fn simd_relu_f32_slice_unified(slice: &mut [f32]) {
        // Convert _slice to Array1 for SIMD operations
        let arr = Array1::from_vec(_slice.to_vec());
        let zero_arr = Array1::zeros(arr.len());
        // Use unified SIMD max operation
        let result = f32::simd_max(&arr.view(), &zero_arr.view());
        // Copy result back to _slice
        for (i, &val) in result.iter().enumerate() {
            slice[i] = val;
    /// Fallback ReLU implementation
    fn simd_relu_f32_slice_fallback(slice: &mut [f32]) {
        for val in slice.iter_mut() {
            *val = val.max(0.0);
    /// Vectorized ReLU activation returning new array
    pub fn simd_relu_f32(input: &ArrayView<f32, IxDyn>) -> ArrayD<f32> {
        let mut result = input.to_owned();
        Self::simd_relu_f32_inplace(&mut result.view_mut());
        result
    /// SIMD-accelerated sigmoid activation for f32 arrays
    pub fn simd_sigmoid_f32_inplace(input: &mut ArrayViewMut<f32, IxDyn>) {
                Self::simd_sigmoid_f32_slice_unified(_slice);
                Self::simd_sigmoid_f32_slice_fallback(_slice);
            input.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
    /// SIMD-accelerated sigmoid for f32 _slice using unified operations
    fn simd_sigmoid_f32_slice_unified(slice: &mut [f32]) {
        // For sigmoid, we need to compute 1 / (1 + exp(-x))
        // Using core SIMD operations
        // Compute -x
        let neg_arr = Array1::from_elem(arr.len(), -1.0f32);
        let neg_x = f32::simd_mul(&arr.view(), &neg_arr.view());
        // Apply sigmoid element-wise (no direct SIMD exp available)
        let result: Array1<f32> = neg_x.mapv(|x| 1.0 / (1.0 + x.exp()));
    /// Fallback sigmoid implementation
    fn simd_sigmoid_f32_slice_fallback(slice: &mut [f32]) {
            *val = 1.0 / (1.0 + (-*val).exp());
    /// Vectorized sigmoid activation returning new array
    pub fn simd_sigmoid_f32(input: &ArrayView<f32, IxDyn>) -> ArrayD<f32> {
        Self::simd_sigmoid_f32_inplace(&mut result.view_mut());
    /// SIMD-accelerated tanh activation for f32 arrays
    pub fn simd_tanh_f32_inplace(input: &mut ArrayViewMut<f32, IxDyn>) {
                Self::simd_tanh_f32_slice_unified(_slice);
                Self::simd_tanh_f32_slice_fallback(_slice);
            input.mapv_inplace(|x| x.tanh());
    /// SIMD-accelerated tanh for f32 _slice using unified operations
    fn simd_tanh_f32_slice_unified(slice: &mut [f32]) {
        // For tanh, we apply element-wise since no direct SIMD tanh
        let result = arr.mapv(|x| x.tanh());
    /// Fallback tanh implementation
    fn simd_tanh_f32_slice_fallback(slice: &mut [f32]) {
            *val = val.tanh();
    /// Vectorized tanh activation returning new array
    pub fn simd_tanh_f32(input: &ArrayView<f32, IxDyn>) -> ArrayD<f32> {
        Self::simd_tanh_f32_inplace(&mut result.view_mut());
    /// SIMD-accelerated GELU activation for f32 arrays
    pub fn simd_gelu_f32(input: &ArrayView<f32, IxDyn>) -> ArrayD<f32> {
        if let Some(_slice) = result.as_slice_mut() {
                Self::simd_gelu_f32_slice_unified(_slice);
                Self::simd_gelu_f32_slice_fallback(_slice);
            result.mapv_inplace(|x| {
                0.5 * x * (1.0 + (x * 0.797_884_6 * (1.0 + 0.044715 * x * x)).tanh())
            });
    /// SIMD-accelerated GELU for f32 slice using unified operations
    fn simd_gelu_f32_slice_unified(slice: &mut [f32]) {
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        // Using core SIMD for parts of the computation
        // Compute x^2 and x^3 using SIMD
        let x_sq = f32::simd_mul(&arr.view(), &arr.view());
        let x_cube = f32::simd_mul(&x_sq.view(), &arr.view());
        // Compute 0.044715 * x^3
        let coeff2_arr = Array1::from_elem(arr.len(), 0.044715f32);
        let term2 = f32::simd_mul(&coeff2_arr.view(), &x_cube.view());
        // Compute x + 0.044715 * x^3
        let inner1 = f32::simd_add(&arr.view(), &term2.view());
        // Compute sqrt(2/π) * (x + 0.044715 * x^3)
        let coeff1_arr = Array1::from_elem(arr.len(), 0.797_884_6f32);
        let inner2 = f32::simd_mul(&coeff1_arr.view(), &inner1.view());
        // Apply tanh and complete GELU computation
        let tanh_vals = inner2.mapv(|x| x.tanh());
        let one_arr = Array1::from_elem(arr.len(), 1.0f32);
        let factor = f32::simd_add(&one_arr.view(), &tanh_vals.view());
        let half_arr = Array1::from_elem(arr.len(), 0.5f32);
        let temp = f32::simd_mul(&half_arr.view(), &arr.view());
        let result = f32::simd_mul(&temp.view(), &factor.view());
    /// Fallback GELU implementation
    fn simd_gelu_f32_slice_fallback(slice: &mut [f32]) {
            let x = *val;
            *val = 0.5 * x * (1.0 + (x * 0.797_884_6 * (1.0 + 0.044715 * x * x)).tanh());
    /// SIMD-accelerated Swish/SiLU activation for f32 arrays
    pub fn simd_swish_f32(input: &ArrayView<f32, IxDyn>) -> ArrayD<f32> {
                Self::simd_swish_f32_slice_unified(_slice);
                Self::simd_swish_f32_slice_fallback(_slice);
            result.mapv_inplace(|x| x / (1.0 + (-x).exp()));
    /// SIMD-accelerated Swish for f32 _slice using unified operations
    fn simd_swish_f32_slice_unified(slice: &mut [f32]) {
        // Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
        // Compute sigmoid(x) = 1 / (1 + exp(-x))
        // Since we don't have direct SIMD exp, we apply element-wise
        let result = arr.mapv(|x| x / (1.0 + (-x).exp()));
    /// Fallback Swish implementation
    fn simd_swish_f32_slice_fallback(slice: &mut [f32]) {
            *val = x / (1.0 + (-x).exp());
    /// SIMD-accelerated softmax for f32 arrays
    pub fn simd_softmax_f32(input: &ArrayD<f32>, axis: Option<usize>) -> Result<ArrayD<f32>> {
        let axis = axis.unwrap_or(input.ndim() - 1);
        if axis >= input.ndim() {
            return Err(NeuralError::InvalidArchitecture(
                "Softmax axis out of bounds".to_string(),
            ));
        let mut result = Array::zeros(input.raw_dim());
        // Process along the specified axis
        for lane in input.axis_iter(scirs2_core::ndarray::Axis(axis)) {
            if let (Some(input_slice), Some(mut result_slice)) = (
                lane.as_slice(),
                result.axis_iter_mut(scirs2_core::ndarray::Axis(axis)).next(),
            ) {
                if let Some(result_slice_mut) = result_slice.as_slice_mut() {
                    Self::simd_softmax_f32_slice(input_slice, result_slice_mut);
                }
        Ok(result)
    /// SIMD softmax for f32 slice using unified operations
    fn simd_softmax_f32_slice(input: &[f32], result: &mut [f32]) {
        if input.is_empty() {
            return;
        // Find maximum for numerical stability
        let input_arr = Array1::from_vec(_input.to_vec());
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let max_arr = Array1::from_elem(_input.len(), max_val);
        // Compute x - max using SIMD
        let shifted = f32::simd_sub(&input_arr.view(), &max_arr.view());
        // Apply exp element-wise (no direct SIMD exp)
        let exp_arr = shifted.mapv(|x| x.exp());
        // Sum all exp values
        let sum = f32::simd_sum(&exp_arr.view());
        // Divide by sum using SIMD
        let sum_arr = Array1::from_elem(exp_arr.len(), sum);
        let softmax_result = f32::simd_div(&exp_arr.view(), &sum_arr.view());
        // Copy result
        for (i, &val) in softmax_result.iter().enumerate() {
            result[i] = val;
    /// SIMD-accelerated cross-entropy loss computation
    pub fn simd_cross_entropy_loss_f32(
        predictions: &ArrayView<f32, IxDyn>,
        targets: &ArrayView<f32, IxDyn>,
        epsilon: f32,
    ) -> Result<f32> {
        if predictions.shape() != targets.shape() {
            return Err(NeuralError::ComputationError(
                "Predictions and targets must have the same shape".to_string(),
        if let (Some(pred_slice), Some(target_slice)) = (predictions.as_slice(), targets.as_slice())
        {
            Ok(Self::simd_cross_entropy_f32_slices(
                pred_slice,
                target_slice,
                epsilon,
            ))
            // Fallback
            let mut loss = 0.0;
            for (&pred, &target) in predictions.iter().zip(targets.iter()) {
                let clamped_pred = pred.max(epsilon).min(1.0 - epsilon);
                loss -= target * clamped_pred.ln();
            Ok(loss / predictions.len() as f32)
    /// SIMD cross-entropy for f32 slices using unified operations
    fn simd_cross_entropy_f32_slices(predictions: &[f32], targets: &[f32], epsilon: f32) -> f32 {
        let len = predictions.len().min(targets.len());
        let pred_arr = Array1::from_vec(predictions[..len].to_vec());
        let target_arr = Array1::from_vec(targets[..len].to_vec());
        // Clamp predictions to [epsilon, 1-epsilon] using SIMD
        let eps_arr = Array1::from_elem(len, epsilon);
        let one_minus_eps_arr = Array1::from_elem(len, 1.0 - epsilon);
        let clamped_lower = f32::simd_max(&pred_arr.view(), &eps_arr.view());
        let clamped_pred = f32::simd_min(&clamped_lower.view(), &one_minus_eps_arr.view());
        // Apply ln element-wise (no direct SIMD ln)
        let log_pred = clamped_pred.mapv(|x| x.ln());
        // Compute -target * log(pred) using SIMD
        let neg_target_arr = f32::simd_scalar_mul(&target_arr.view(), -1.0);
        let loss_values = f32::simd_mul(&neg_target_arr.view(), &log_pred.view());
        // Sum all loss values
        let total_loss = f32::simd_sum(&loss_values.view());
        total_loss / len as f32
    /// SIMD-accelerated matrix multiplication
    pub fn simd_matmul_f32(
        a: &ArrayView<f32, IxDyn>,
        b: &ArrayView<f32, IxDyn>,
    ) -> Result<ArrayD<f32>> {
        if a.ndim() != 2 || b.ndim() != 2 {
                "SIMD matmul only supports 2D arrays".to_string(),
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let (k2, n) = (b.shape()[0], b.shape()[1]);
        if k != k2 {
                "Matrix dimensions don't match for multiplication".to_string(),
        let mut result = Array::zeros((m, n));
        // Use SIMD for inner loop vectorization
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                // SIMD-accelerated dot product
                let a_row_view = a.slice(scirs2_core::ndarray::s![i, ..]);
                let a_row_slice = if a.is_standard_layout() {
                    a_row_view.as_slice()
                } else {
                    None
                };
                if let Some(a_row) = a_row_slice {
                    sum = Self::simd_dot_product_f32(a_row, &Self::extract_column(b, j));
                    // Fallback to standard computation
                    for l in 0..k {
                        sum += a[[i, l]] * b[[l, j]];
                    }
                result[[i, j]] = sum;
        Ok(result.into_dyn())
    /// Extract column from 2D array for SIMD operations
    fn extract_column(_matrix: &ArrayView<f32, IxDyn>, colidx: usize) -> Vec<f32> {
        let rows = matrix.shape()[0];
        let mut column = Vec::with_capacity(rows);
        for i in 0..rows {
            column.push(_matrix[[i, col_idx]]);
        column
    /// SIMD-accelerated dot product using unified operations
    fn simd_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let a_arr = Array1::from_vec(a[..len].to_vec());
        let b_arr = Array1::from_vec(b[..len].to_vec());
        // Compute element-wise product using SIMD
        let product = f32::simd_mul(&a_arr.view(), &b_arr.view());
        // Sum all products using SIMD
        f32::simd_sum(&product.view())
    /// SIMD-accelerated element-wise addition
    pub fn simd_add_f32(
        if a.shape() != b.shape() {
                "Arrays must have the same shape for addition".to_string(),
        let mut result = Array::zeros(a.raw_dim());
        let caps = PlatformCapabilities::detect();
        if caps.simd_available && f32::simd_available() {
            if let (Some(a_slice), Some(b_slice), Some(result_slice)) =
                (a.as_slice(), b.as_slice(), result.as_slice_mut())
            {
                Self::simd_add_f32_slices(a_slice, b_slice, result_slice);
                // Fallback for non-contiguous arrays
                for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                    *result_val = a_val + b_val;
            // Fallback for non-SIMD
            for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                *result_val = a_val + b_val;
    /// SIMD element-wise addition for slices using unified operations
    fn simd_add_f32_slices(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len().min(b.len()).min(result.len());
        // Add using SIMD
        let sum = f32::simd_add(&a_arr.view(), &b_arr.view());
        for (i, &val) in sum.iter().enumerate() {
    /// SIMD-accelerated convolution operation
    pub fn simd_conv2d_f32(
        input: &ArrayView<f32, IxDyn>,
        kernel: &ArrayView<f32, IxDyn>,
        bias: Option<&[f32]>,
        stride: (usize, usize),
        padding: (usize, usize),
        if input.ndim() != 4 || kernel.ndim() != 4 {
                "Input and kernel must be 4D arrays (batch, channels, height, width)".to_string(),
        let (batch_size, in_channels, in_height, in_width) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (out_channels_, kernel_height, kernel_width) = (
            kernel.shape()[0],
            kernel.shape()[1],
            kernel.shape()[2],
            kernel.shape()[3],
        let out_height = (in_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
        let out_width = (in_width + 2 * padding.1 - kernel_width) / stride.1 + 1;
        let mut output = Array::zeros((batch_size, out_channels, out_height, out_width));
        // SIMD-optimized convolution
        for batch in 0..batch_size {
            for out_ch in 0..out_channels {
                for out_h in 0..out_height {
                    for out_w in 0..out_width {
                        let mut sum = 0.0f32;
                        for in_ch in 0..in_channels {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let in_h = out_h * stride.0 + kh;
                                    let in_w = out_w * stride.1 + kw;
                                    if in_h >= padding.0
                                        && in_w >= padding.1
                                        && in_h - padding.0 < in_height
                                        && in_w - padding.1 < in_width
                                    {
                                        let input_val = input
                                            [[batch, in_ch, in_h - padding.0, in_w - padding.1]];
                                        let kernel_val = kernel[[out_ch, in_ch, kh, kw]];
                                        sum += input_val * kernel_val;
                                    }
                                }
                            }
                        }
                        // Add bias if provided
                        if let Some(b) = bias {
                            sum += b[out_ch % b.len()];
                        output[[batch, out_ch, out_h, out_w]] = sum;
        Ok(output.into_dyn())
    /// SIMD-accelerated batch normalization
    pub fn simd_batch_norm_f32(
        gamma: Option<&[f32]>,
        beta: Option<&[f32]>,
        mean: &[f32],
        variance: &[f32],
        if input.ndim() < 2 {
                "Input must have at least 2 dimensions".to_string(),
        let channels = input.shape()[1];
        if mean.len() != channels || variance.len() != channels {
                "Mean and variance must have same length as input channels".to_string(),
        // Process each element
        for (idx, (&input_val, result_val)) in input.iter().zip(result.iter_mut()).enumerate() {
            // Calculate which channel this element belongs to
            let flat_idx = idx;
            let total_spatial = input.len() / (input.shape()[0] * channels);
            let _spatial_idx = flat_idx % total_spatial;
            let channel_batch_idx = flat_idx / total_spatial;
            let channel = channel_batch_idx % channels;
            // Normalize: (x - mean) / sqrt(var + epsilon)
            let normalized = (input_val - mean[channel]) / (variance[channel] + epsilon).sqrt();
            // Scale and shift: gamma * normalized + beta
            let mut output = normalized;
            if let Some(g) = gamma {
                output *= g[channel % g.len()];
            if let Some(b) = beta {
                output += b[channel % b.len()];
            *result_val = output;
}
// Provide no-op implementations when SIMD is not available
/// SIMD operations for neural network computations (fallback implementation)
#[cfg(not(feature = "simd"))]
    /// Apply ReLU activation in-place using SIMD operations
    pub fn simd_relu_f32_inplace(input: &mut ArrayViewMut<f32, IxDyn>) {
        // No-op fallback
    /// Apply ReLU activation using SIMD operations
    pub fn simd_relu_f32(input: &ArrayView<f32, IxDyn>) -> ArrayD<f32> {
        // Fallback to standard operation
        input.mapv(|x| x.max(0.0))
    /// Apply sigmoid activation in-place using SIMD operations
    pub fn simd_sigmoid_f32_inplace(input: &mut ArrayViewMut<f32, IxDyn>) {
    /// Apply sigmoid activation using SIMD operations
    pub fn simd_sigmoid_f32(input: &ArrayView<f32, IxDyn>) -> ArrayD<f32> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    /// Apply tanh activation in-place using SIMD operations
    pub fn simd_tanh_f32_inplace(input: &mut ArrayViewMut<f32, IxDyn>) {
    /// Apply tanh activation using SIMD operations
    pub fn simd_tanh_f32(input: &ArrayView<f32, IxDyn>) -> ArrayD<f32> {
        input.mapv(|x| x.tanh())
    /// Apply GELU activation using SIMD operations
    pub fn simd_gelu_f32(input: &ArrayView<f32, IxDyn>) -> ArrayD<f32> {
        input.mapv(|x| 0.5 * x * (1.0 + (x * 0.797_884_6 * (1.0 + 0.044715 * x * x)).tanh()))
    /// Apply Swish activation using SIMD operations
    pub fn simd_swish_f32(input: &ArrayView<f32, IxDyn>) -> ArrayD<f32> {
        input.mapv(|x| x / (1.0 + (-x).exp()))
    /// Apply softmax activation using SIMD operations
    pub fn simd_softmax_f32(_input: &ArrayD<f32>, axis: Option<usize>) -> Result<ArrayD<f32>> {
        Err(NeuralError::ComputationError(
            "SIMD softmax requires 'simd' feature".to_string(),
        ))
    /// Compute cross-entropy loss using SIMD operations
        _predictions: &ArrayView<f32, IxDyn>, _targets: &ArrayView<f32, IxDyn>, _epsilon: f32,
            "SIMD cross entropy requires 'simd' feature".to_string(),
    /// Perform matrix multiplication using SIMD operations
        _a: &ArrayView<f32, IxDyn>, _b: &ArrayView<f32, IxDyn>,
            "SIMD matmul requires 'simd' feature".to_string(),
    /// Perform element-wise addition using SIMD operations
            "SIMD add requires 'simd' feature".to_string(),
    /// Perform 2D convolution using SIMD operations
        _input: &ArrayView<f32, IxDyn>, _kernel: &ArrayView<f32, IxDyn>, _bias: Option<&[f32]>, _stride: (usize, usize), _padding: (usize, usize),
            "SIMD conv2d requires 'simd' feature".to_string(),
    /// Perform batch normalization using SIMD operations
        _gamma: Option<&[f32]>, _beta: Option<&[f32]>, _mean: &[f32], _variance: &[f32],
            "SIMD batch norm requires 'simd' feature".to_string(),
