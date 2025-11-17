//! MPSGraph (Metal Performance Shaders Graph) integration for high-performance operations
//!
//! This module provides access to Apple's MPSGraph framework, which offers automatic
//! optimization and kernel fusion for graph-based operations. MPSGraph is used by
//! PyTorch, MLX, and other ML frameworks for optimal performance on Apple Silicon.
//!
//! ## Performance Characteristics
//!
//! MPSGraph provides 10-50x performance improvements over naive Metal kernels through:
//! - Automatic kernel fusion (e.g., matmul + softmax + matmul → single fused kernel)
//! - Platform-specific optimizations for M1/M2/M3 architectures
//! - Intelligent memory bandwidth management
//! - Operator stitching (e.g., GeLU can be 10-50x faster)
//!
//! ## Reference
//!
//! Based on PyTorch's MPS implementation:
//! - `aten/src/ATen/native/mps/operations/Attention.mm`
//! - Apple WWDC 2024: "Accelerate machine learning with Metal"

#![cfg(all(feature = "mpsgraph", target_os = "macos"))]
#![allow(dead_code)]

use crate::gpu::GpuError;
use std::sync::Arc;

// MPSGraph imports
#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice};

#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphExecutable, MPSGraphTensor, MPSGraphTensorData,
};

#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
use objc2::runtime::ProtocolObject;

#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
use objc2::rc::Retained;

#[cfg(all(feature = "mpsgraph", target_os = "macos"))]
use objc2::{msg_send, msg_send_id, ClassType};

// Fallback types for non-macOS platforms
#[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
type MTLDevice = ();
#[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
type MTLCommandQueue = ();
#[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
type MTLBuffer = ();

/// MPSGraph context for high-performance graph operations
pub struct MPSGraphContext {
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    graph: Retained<MPSGraph>,
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    device: MTLDevice,
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    command_queue: MTLCommandQueue,
}

// SAFETY: Metal devices, command queues, and graphs are thread-safe
unsafe impl Send for MPSGraphContext {}
unsafe impl Sync for MPSGraphContext {}

impl MPSGraphContext {
    /// Create a new MPSGraph context
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> Self {
        // Create MPSGraph instance
        let graph = unsafe { MPSGraph::new() };

        Self {
            device,
            command_queue,
            graph,
        }
    }

    /// Create a new MPSGraph context (fallback for non-macOS)
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    pub fn new(device: MTLDevice, command_queue: MTLCommandQueue) -> Self {
        Self {
            device,
            command_queue,
        }
    }

    /// Scaled Dot-Product Attention using MPSGraph
    ///
    /// Implements: softmax(Q @ K^T / sqrt(d_k)) @ V
    ///
    /// This is the primary attention operation used by PyTorch on MPS.
    /// MPSGraph automatically fuses the operations for optimal performance.
    ///
    /// # Arguments
    ///
    /// * `q_buffer` - Query tensor [batch, num_heads, q_seq_len, head_dim]
    /// * `k_buffer` - Key tensor [batch, num_heads, kv_seq_len, head_dim]
    /// * `v_buffer` - Value tensor [batch, num_heads, kv_seq_len, head_dim]
    /// * `batch_size` - Batch size
    /// * `num_heads` - Number of attention heads
    /// * `q_seq_len` - Query sequence length
    /// * `kv_seq_len` - Key/Value sequence length
    /// * `head_dim` - Dimension per head
    /// * `use_causal_mask` - Whether to apply causal masking
    ///
    /// # Returns
    ///
    /// Output buffer [batch, num_heads, q_seq_len, head_dim]
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    pub fn scaled_dot_product_attention(
        &self,
        q_buffer: &ProtocolObject<dyn MTLBuffer>,
        k_buffer: &ProtocolObject<dyn MTLBuffer>,
        v_buffer: &ProtocolObject<dyn MTLBuffer>,
        batch_size: usize,
        num_heads: usize,
        q_seq_len: usize,
        kv_seq_len: usize,
        head_dim: usize,
        use_causal_mask: bool,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSArray, NSNumber};
        use objc2_metal_performance_shaders_graph::MPSDataType;

        // Calculate scale factor: 1/sqrt(head_dim)
        let scale_factor = 1.0 / (head_dim as f32).sqrt();

        // Shape: [batch, num_heads, seq_len, head_dim]
        let q_shape = vec![
            batch_size as i64,
            num_heads as i64,
            q_seq_len as i64,
            head_dim as i64,
        ];
        let k_shape = vec![
            batch_size as i64,
            num_heads as i64,
            kv_seq_len as i64,
            head_dim as i64,
        ];
        let v_shape = vec![
            batch_size as i64,
            num_heads as i64,
            kv_seq_len as i64,
            head_dim as i64,
        ];

        let result = autoreleasepool(|_| {
            // Create NSArray for shapes
            let q_shape_ns = unsafe {
                NSArray::from_vec(q_shape.iter().map(|&x| NSNumber::new_i64(x)).collect())
            };
            let k_shape_ns = unsafe {
                NSArray::from_vec(k_shape.iter().map(|&x| NSNumber::new_i64(x)).collect())
            };
            let v_shape_ns = unsafe {
                NSArray::from_vec(v_shape.iter().map(|&x| NSNumber::new_i64(x)).collect())
            };

            // Create placeholder tensors for Q, K, V using msg_send
            let q_placeholder: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    &self.graph,
                    placeholderWithShape: &*q_shape_ns,
                    dataType: MPSDataType::Float32,
                    name: None::<&objc2_foundation::NSString>
                ]
            };

            let k_placeholder: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    &self.graph,
                    placeholderWithShape: &*k_shape_ns,
                    dataType: MPSDataType::Float32,
                    name: None::<&objc2_foundation::NSString>
                ]
            };

            let v_placeholder: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    &self.graph,
                    placeholderWithShape: &*v_shape_ns,
                    dataType: MPSDataType::Float32,
                    name: None::<&objc2_foundation::NSString>
                ]
            };

            // Build computation graph
            // 1. Transpose K: [batch, num_heads, kv_seq, head_dim] → [batch, num_heads, head_dim, kv_seq]
            let k_transposed: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    &self.graph,
                    transposeTensor: &*k_placeholder,
                    dimension: 2i64,
                    withDimension: 3i64,
                    name: None::<&objc2_foundation::NSString>
                ]
            };

            // 2. MatMul: Q @ K^T
            let scores: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    &self.graph,
                    matrixMultiplicationWithPrimaryTensor: &*q_placeholder,
                    secondaryTensor: &*k_transposed,
                    name: None::<&objc2_foundation::NSString>
                ]
            };

            // 3. Scale by 1/sqrt(head_dim)
            let scale_tensor: Retained<MPSGraphTensor> = unsafe {
                let scale_shape_ns = NSArray::from_vec(vec![NSNumber::new_i64(1)]);
                msg_send_id![
                    &self.graph,
                    constantWithScalar: scale_factor as f64,
                    shape: &*scale_shape_ns,
                    dataType: MPSDataType::Float32
                ]
            };

            let scaled_scores: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    &self.graph,
                    multiplicationWithPrimaryTensor: &*scores,
                    secondaryTensor: &*scale_tensor,
                    name: None::<&objc2_foundation::NSString>
                ]
            };

            // 4. Apply causal mask if needed
            let masked_scores = if use_causal_mask {
                // Create causal mask using bandPart
                let mask_shape_ns = unsafe {
                    NSArray::from_vec(vec![
                        NSNumber::new_i64(q_seq_len as i64),
                        NSNumber::new_i64(kv_seq_len as i64),
                    ])
                };

                let causal_mask: Retained<MPSGraphTensor> = unsafe {
                    msg_send_id![
                        &self.graph,
                        constantWithScalar: 1.0f64,
                        shape: &*mask_shape_ns,
                        dataType: MPSDataType::Bool
                    ]
                };

                let causal_mask: Retained<MPSGraphTensor> = unsafe {
                    msg_send_id![
                        &self.graph,
                        bandPartWithTensor: &*causal_mask,
                        numLower: -1i64,
                        numUpper: 0i64,
                        name: None::<&objc2_foundation::NSString>
                    ]
                };

                // Create -inf tensor for masked positions
                let minus_inf: Retained<MPSGraphTensor> = unsafe {
                    let scores_shape: Retained<NSArray<NSNumber>> =
                        msg_send_id![&*scaled_scores, shape];
                    msg_send_id![
                        &self.graph,
                        constantWithScalar: -1e20f64,
                        shape: &*scores_shape,
                        dataType: MPSDataType::Float32
                    ]
                };

                // Select: causal_mask ? scaled_scores : -inf
                unsafe {
                    msg_send_id![
                        &self.graph,
                        selectWithPredicateTensor: &*causal_mask,
                        truePredicateTensor: &*scaled_scores,
                        falsePredicateTensor: &*minus_inf,
                        name: None::<&objc2_foundation::NSString>
                    ]
                }
            } else {
                scaled_scores
            };

            // 5. Softmax along last dimension (axis=3)
            let attention_weights: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    &self.graph,
                    softMaxWithTensor: &*masked_scores,
                    axis: 3i64,
                    name: None::<&objc2_foundation::NSString>
                ]
            };

            // 6. Final MatMul: attention_weights @ V
            let output: Retained<MPSGraphTensor> = unsafe {
                msg_send_id![
                    &self.graph,
                    matrixMultiplicationWithPrimaryTensor: &*attention_weights,
                    secondaryTensor: &*v_placeholder,
                    name: None::<&objc2_foundation::NSString>
                ]
            };

            // Create output buffer
            let output_size = batch_size * num_heads * q_seq_len * head_dim;
            let output_buffer = self.device.new_buffer(
                (output_size * 4) as u64, // 4 bytes per f32
                objc2_metal::MTLResourceOptions::StorageModeShared,
            );

            // TODO: Create MPSGraphTensorData for inputs and execute graph
            // For now, return error - need to implement graph execution
            Err(GpuError::Other(
                "MPSGraph execution not yet implemented - graph built successfully".to_string(),
            ))
        });

        result
    }

    /// Scaled Dot-Product Attention (fallback for non-macOS)
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    pub fn scaled_dot_product_attention(
        &self,
        _q_buffer: &MTLBuffer,
        _k_buffer: &MTLBuffer,
        _v_buffer: &MTLBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _q_seq_len: usize,
        _kv_seq_len: usize,
        _head_dim: usize,
        _use_causal_mask: bool,
    ) -> Result<MTLBuffer, GpuError> {
        Err(GpuError::Other(
            "MPSGraph is only available on macOS".to_string(),
        ))
    }

    /// Matrix multiplication using MPSGraph
    ///
    /// Computes: C = A @ B
    ///
    /// MPSGraph provides highly optimized matmul that's often faster than
    /// custom Metal kernels.
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    pub fn matmul(
        &self,
        a_buffer: &ProtocolObject<dyn MTLBuffer>,
        b_buffer: &ProtocolObject<dyn MTLBuffer>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        // TODO: Implement using MPSGraph matmul operation
        Err(GpuError::Other(
            "MPSGraph matmul not yet implemented".to_string(),
        ))
    }

    /// Matrix multiplication (fallback)
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    pub fn matmul(
        &self,
        _a_buffer: &MTLBuffer,
        _b_buffer: &MTLBuffer,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<MTLBuffer, GpuError> {
        Err(GpuError::Other(
            "MPSGraph is only available on macOS".to_string(),
        ))
    }

    /// Softmax using MPSGraph
    ///
    /// Computes: softmax(x, axis)
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    pub fn softmax(
        &self,
        input_buffer: &ProtocolObject<dyn MTLBuffer>,
        shape: &[usize],
        axis: isize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        // TODO: Implement using MPSGraph softmax operation
        Err(GpuError::Other(
            "MPSGraph softmax not yet implemented".to_string(),
        ))
    }

    /// Softmax (fallback)
    #[cfg(not(all(feature = "mpsgraph", target_os = "macos")))]
    pub fn softmax(
        &self,
        _input_buffer: &MTLBuffer,
        _shape: &[usize],
        _axis: isize,
    ) -> Result<MTLBuffer, GpuError> {
        Err(GpuError::Other(
            "MPSGraph is only available on macOS".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(feature = "mpsgraph", target_os = "macos"))]
    fn test_mpsgraph_context_creation() {
        // This test requires macOS environment
        // TODO: Add actual test when MPSGraph operations are implemented
    }
}
