//! Metal Performance Shaders (MPS) integration for accelerated operations
//!
//! This module provides access to Apple's optimized GPU primitives through
//! Metal Performance Shaders, offering high-performance implementations of
//! common operations like matrix multiplication, convolution, and more.
//!
//! ## Implementation Status (objc2 API Migration)
//!
//! This module has been updated to use the objc2-metal and objc2-metal-performance-shaders
//! crates (v0.3.1). All operations are currently stub implementations that require:
//! - macOS development environment for testing
//! - Implementation of actual MPS operations using objc2 bindings
//!
//! ### Required objc2 Types:
//! - `MTLDevice`, `MTLCommandQueue`, `MTLBuffer` from objc2-metal
//! - `MPSMatrixDescriptor`, `MPSMatrix`, `MPSMatrixMultiplication` from objc2-metal-performance-shaders
//! - `MPSNNOptimizer` family for neural network operations
//! - `MPSImageConvolution`, `MPSImageGaussianBlur` for image operations

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(dead_code)]

use crate::gpu::GpuError;
use std::sync::Arc;

// objc2 API imports for Metal and Metal Performance Shaders
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice};

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal_performance_shaders::{
    MPSDataType as MPSDataTypeEnum, MPSImageConvolution, MPSImageGaussianBlur, MPSMatrix,
    MPSMatrixDescriptor, MPSMatrixMultiplication,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::runtime::ProtocolObject;

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::rc::Retained;

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::{msg_send, msg_send_id, ClassType};

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::runtime::AnyObject;

// Import macOS-specific MPS class types
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal_performance_shaders::MPSKernel;

// Fallback type aliases when not on macOS
#[cfg(not(all(feature = "metal", target_os = "macos")))]
type MTLDevice = ();
#[cfg(not(all(feature = "metal", target_os = "macos")))]
type MTLCommandQueue = ();
#[cfg(not(all(feature = "metal", target_os = "macos")))]
type MTLBuffer = ();

/// Metal Performance Shaders context (using objc2 API)
pub struct MPSContext {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    device: MTLDevice,
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    command_queue: MTLCommandQueue,
}

// SAFETY: Metal devices and command queues are inherently thread-safe.
// The Retained pointers don't implement Sync because they're trait objects,
// but the underlying Metal objects are designed for multi-threaded access.
unsafe impl Send for MPSContext {}
unsafe impl Sync for MPSContext {}

impl MPSContext {
    /// Create a new MPS context (objc2 API)
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> Self {
        Self {
            device,
            command_queue,
        }
    }

    /// Create a new MPS context (fallback)
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn new(device: MTLDevice, command_queue: MTLCommandQueue) -> Self {
        Self {
            device,
            command_queue,
        }
    }

    /// Create a matrix multiplication operation
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn create_matmul(
        &self,
        transpose_left: bool,
        transpose_right: bool,
        result_rows: usize,
        result_columns: usize,
        interior_columns: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<Retained<MPSMatrixMultiplication>, GpuError> {
        use objc2_metal_performance_shaders::MPSMatrixMultiplication;

        // Create matrix multiplication kernel using msg_send (handles trait objects properly)
        let matmul = unsafe {
            let cls = MPSMatrixMultiplication::class();
            let alloc = msg_send_id![cls, alloc];
            msg_send_id![
                alloc,
                initWithDevice: &*self.device,
                transposeLeft: transpose_left,
                transposeRight: transpose_right,
                resultRows: result_rows,
                resultColumns: result_columns,
                interiorColumns: interior_columns,
                alpha: alpha,
                beta: beta
            ]
        };

        Ok(matmul)
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn create_matmul(
        &self,
        _transpose_left: bool,
        _transpose_right: bool,
        _result_rows: usize,
        _result_columns: usize,
        _interior_columns: usize,
        _alpha: f64,
        _beta: f64,
    ) -> Result<(), GpuError> {
        Err(GpuError::Other(
            "Metal not available on this platform".to_string(),
        ))
    }

    /// Create a matrix descriptor
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn create_descriptor(
        rows: usize,
        columns: usize,
        row_bytes: usize,
        datatype: MPSDataType,
    ) -> Result<Retained<MPSMatrixDescriptor>, GpuError> {
        use objc2_metal_performance_shaders::MPSMatrixDescriptor;

        // Map our datatype to MPS data type enum
        let mps_datatype = match datatype {
            MPSDataType::Float32 => MPSDataTypeEnum::Float32,
            MPSDataType::Float16 => MPSDataTypeEnum::Float16,
            MPSDataType::Int32 => MPSDataTypeEnum::Int32,
            _ => {
                return Err(GpuError::Other(format!(
                    "Unsupported datatype: {:?}",
                    datatype
                )))
            }
        };

        // Create matrix descriptor using msg_send
        let descriptor = unsafe {
            let cls = MPSMatrixDescriptor::class();
            msg_send_id![
                cls,
                matrixDescriptorWithRows: rows,
                columns: columns,
                rowBytes: row_bytes,
                dataType: mps_datatype
            ]
        };

        Ok(descriptor)
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn create_descriptor(
        _rows: usize,
        _columns: usize,
        _row_bytes: usize,
        _datatype: MPSDataType,
    ) -> Result<(), GpuError> {
        Err(GpuError::Other(
            "Metal not available on this platform".to_string(),
        ))
    }

    /// Create an MPS matrix from a Metal buffer
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn create_matrix(
        &self,
        buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        descriptor: &Retained<MPSMatrixDescriptor>,
    ) -> Result<Retained<MPSMatrix>, GpuError> {
        use objc2_metal_performance_shaders::MPSMatrix;

        // Create MPSMatrix wrapping the MTLBuffer using msg_send (handles trait objects)
        let matrix = unsafe {
            let cls = MPSMatrix::class();
            let alloc = msg_send_id![cls, alloc];
            msg_send_id![
                alloc,
                initWithBuffer: &**buffer,
                descriptor: &**descriptor
            ]
        };

        Ok(matrix)
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn create_matrix(&self, _buffer: &MTLBuffer, _descriptor: &()) -> Result<(), GpuError> {
        Err(GpuError::Other(
            "Metal not available on this platform".to_string(),
        ))
    }

    /// Create a command buffer for batching operations
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn create_command_buffer(&self) -> Result<Retained<AnyObject>, GpuError> {
        let command_buffer: Option<Retained<AnyObject>> =
            unsafe { msg_send_id![&self.command_queue, commandBuffer] };

        command_buffer.ok_or_else(|| GpuError::Other("Failed to create command buffer".to_string()))
    }

    /// Commit a command buffer (non-blocking)
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn commit_command_buffer(&self, command_buffer: &Retained<AnyObject>) {
        unsafe {
            let _: () = msg_send![&**command_buffer, commit];
        }
    }

    /// Wait for a command buffer to complete
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn wait_for_command_buffer(&self, command_buffer: &Retained<AnyObject>) {
        unsafe {
            let _: () = msg_send![&**command_buffer, waitUntilCompleted];
        }
    }

    /// Encode matrix multiplication to an existing command buffer (non-blocking, for batching)
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn encode_matrix_multiply(
        &self,
        command_buffer: &Retained<AnyObject>,
        left_matrix: &Retained<MPSMatrix>,
        right_matrix: &Retained<MPSMatrix>,
        result_matrix: &Retained<MPSMatrix>,
        matmul: &Retained<MPSMatrixMultiplication>,
    ) -> Result<(), GpuError> {
        // Encode matrix multiplication operation using msg_send
        unsafe {
            let _: () = msg_send![
                &**matmul,
                encodeToCommandBuffer:&**command_buffer
                leftMatrix:&**left_matrix
                rightMatrix:&**right_matrix
                resultMatrix:&**result_matrix
            ];
        }

        Ok(())
    }

    /// Perform matrix multiplication using MPS (creates own command buffer and waits)
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn matrix_multiply(
        &self,
        left_matrix: &Retained<MPSMatrix>,
        right_matrix: &Retained<MPSMatrix>,
        result_matrix: &Retained<MPSMatrix>,
        matmul: &Retained<MPSMatrixMultiplication>,
    ) -> Result<(), GpuError> {
        use objc2_metal::MTLCommandBuffer;

        // Create command buffer using msg_send! (trait object requires dynamic dispatch)
        let command_buffer = self.create_command_buffer()?;

        // Encode operation
        self.encode_matrix_multiply(
            &command_buffer,
            left_matrix,
            right_matrix,
            result_matrix,
            matmul,
        )?;

        // Commit and wait for completion
        self.commit_command_buffer(&command_buffer);
        self.wait_for_command_buffer(&command_buffer);

        Ok(())
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn matrix_multiply(
        &self,
        _left: &(),
        _right: &(),
        _result: &(),
        _matmul: &(),
    ) -> Result<(), GpuError> {
        Err(GpuError::Other(
            "Metal not available on this platform".to_string(),
        ))
    }

    /// Create a softmax operation (stub)
    pub fn create_softmax(&self, _axis: i32) -> Result<(), GpuError> {
        // TODO: Implement proper MPS softmax with updated objc2 API
        Err(GpuError::Other(
            "MPS softmax not yet implemented with new objc2 API".to_string(),
        ))
    }

    /// Create a sum reduction operation (stub)
    pub fn create_sum(&self) -> Result<(), GpuError> {
        // TODO: Implement proper MPS sum with updated objc2 API
        Err(GpuError::Other(
            "MPS sum not yet implemented with new objc2 API".to_string(),
        ))
    }

    /// Create a top-k operation (stub)
    pub fn create_find_top_k(&self, _k: usize) -> Result<(), GpuError> {
        // TODO: Implement proper MPS top-k with updated objc2 API
        Err(GpuError::Other(
            "MPS top-k not yet implemented with new objc2 API".to_string(),
        ))
    }
}

/// MPS-accelerated convolution operation (stub)
pub struct MPSConvolution {
    pub(crate) context: Arc<MPSContext>,
}

impl MPSConvolution {
    /// Create a new MPS convolution operation (stub)
    pub fn new(_context: Arc<MPSContext>) -> Result<Self, GpuError> {
        // TODO: Implement proper MPS convolution with updated objc2 API
        Err(GpuError::Other(
            "MPS convolution not yet implemented with new objc2 API".to_string(),
        ))
    }

    /// Execute convolution (stub - objc2 API)
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn execute(
        &self,
        _input: &objc2::rc::Retained<dyn MTLBuffer>,
        _weights: &objc2::rc::Retained<dyn MTLBuffer>,
        _output: &mut objc2::rc::Retained<dyn MTLBuffer>,
    ) -> Result<(), GpuError> {
        // TODO: Implement using objc2 MPSCNNConvolution::encode_to_command_buffer
        Err(GpuError::Other(
            "MPS convolution execution not yet implemented with new objc2 API".to_string(),
        ))
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn execute(
        &self,
        _input: &MTLBuffer,
        _weights: &MTLBuffer,
        _output: &mut MTLBuffer,
    ) -> Result<(), GpuError> {
        Err(GpuError::Other(
            "Metal not available on this platform".to_string(),
        ))
    }
}

/// MPS-accelerated pooling operations (stub)
pub struct MPSPooling {
    pub(crate) context: Arc<MPSContext>,
    pub(crate) pool_type: PoolType,
}

/// Pooling type
#[derive(Clone, Copy, Debug)]
pub enum PoolType {
    Max,
    Average,
}

impl MPSPooling {
    /// Create a new MPS pooling operation (stub)
    pub fn new(_context: Arc<MPSContext>, _pool_type: PoolType) -> Result<Self, GpuError> {
        // TODO: Implement proper MPS pooling with updated objc2 API
        Err(GpuError::Other(
            "MPS pooling not yet implemented with new objc2 API".to_string(),
        ))
    }

    /// Execute pooling (stub - objc2 API)
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn execute(
        &self,
        _input: &objc2::rc::Retained<dyn MTLBuffer>,
        _output: &mut objc2::rc::Retained<dyn MTLBuffer>,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<(), GpuError> {
        // TODO: Implement using objc2 MPSCNNPooling::encode_to_command_buffer
        Err(GpuError::Other(
            "MPS pooling execution not yet implemented with new objc2 API".to_string(),
        ))
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn execute(
        &self,
        _input: &MTLBuffer,
        _output: &mut MTLBuffer,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<(), GpuError> {
        Err(GpuError::Other(
            "Metal not available on this platform".to_string(),
        ))
    }
}

/// MPS data type
#[derive(Clone, Copy, Debug)]
pub enum MPSDataType {
    Float32,
    Float16,
    Int32,
    Int16,
    Int8,
    UInt8,
}

impl MPSDataType {
    /// Convert to MPS data type value (stub)
    pub fn to_mps_datatype(self) -> u32 {
        // TODO: Return proper MPS data type values
        match self {
            MPSDataType::Float32 => 0x10000 | 32, // Placeholder value
            MPSDataType::Float16 => 0x10000 | 16, // Placeholder value
            MPSDataType::Int32 => 0x20000 | 32,   // Placeholder value
            MPSDataType::Int16 => 0x20000 | 16,   // Placeholder value
            MPSDataType::Int8 => 0x20000 | 8,     // Placeholder value
            MPSDataType::UInt8 => 0x30000 | 8,    // Placeholder value
        }
    }
}

/// MPS operations wrapper for high-level operations (using objc2 API)
pub struct MPSOperations {
    context: Arc<MPSContext>,
}

// SAFETY: MPSOperations only contains Arc<MPSContext>, and MPSContext is Send + Sync.
// Arc itself is Send + Sync when T is Send + Sync.
unsafe impl Send for MPSOperations {}
unsafe impl Sync for MPSOperations {}

impl MPSOperations {
    /// Create new MPS operations instance (objc2 API)
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> Self {
        Self {
            context: Arc::new(MPSContext::new(device, command_queue)),
        }
    }

    /// Create new MPS operations instance (fallback)
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn new(device: MTLDevice, command_queue: MTLCommandQueue) -> Self {
        Self {
            context: Arc::new(MPSContext::new(device, command_queue)),
        }
    }

    /// Get the underlying context
    pub fn context(&self) -> &Arc<MPSContext> {
        &self.context
    }

    /// Encode matrix multiplication to an existing command buffer (for batching)
    ///
    /// This variant doesn't commit or wait, allowing multiple operations to be batched.
    /// Expected speedup: 2-3x when batching multiple operations.
    ///
    /// # Arguments
    /// * `command_buffer` - Existing command buffer to encode into
    /// * `a_buffer` - Left matrix buffer (M x K)
    /// * `b_buffer` - Right matrix buffer (K x N)
    /// * `c_buffer` - Result matrix buffer (M x N)
    /// * `m` - Number of rows in A and C
    /// * `k` - Number of columns in A and rows in B
    /// * `n` - Number of columns in B and C
    ///
    /// # Returns
    /// Ok(()) if operation encoded successfully
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn encode_matmul_f32(
        &self,
        command_buffer: &Retained<AnyObject>,
        a_buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        b_buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        c_buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        // Create matrix descriptors
        let a_desc = MPSContext::create_descriptor(m, k, k * 4, MPSDataType::Float32)?;
        let b_desc = MPSContext::create_descriptor(k, n, n * 4, MPSDataType::Float32)?;
        let c_desc = MPSContext::create_descriptor(m, n, n * 4, MPSDataType::Float32)?;

        // Create MPS matrices
        let a_matrix = self.context.create_matrix(a_buffer, &a_desc)?;
        let b_matrix = self.context.create_matrix(b_buffer, &b_desc)?;
        let c_matrix = self.context.create_matrix(c_buffer, &c_desc)?;

        // Create matmul kernel (alpha=1.0, beta=0.0 for C = A*B)
        let matmul = self.context.create_matmul(
            false, // No transpose for A
            false, // No transpose for B
            m,     // Result rows
            n,     // Result columns
            k,     // Interior dimension
            1.0,   // alpha
            0.0,   // beta
        )?;

        // Encode multiplication (don't commit/wait)
        self.context.encode_matrix_multiply(
            command_buffer,
            &a_matrix,
            &b_matrix,
            &c_matrix,
            &matmul,
        )?;

        Ok(())
    }

    /// High-level matrix multiplication for f32 data (C = A * B)
    ///
    /// Performs optimized matrix multiplication using Metal Performance Shaders.
    /// Expected speedup: 100-500x over naive Metal kernels.
    ///
    /// # Arguments
    /// * `a_buffer` - Left matrix buffer (M x K)
    /// * `b_buffer` - Right matrix buffer (K x N)
    /// * `c_buffer` - Result matrix buffer (M x N)
    /// * `m` - Number of rows in A and C
    /// * `k` - Number of columns in A and rows in B
    /// * `n` - Number of columns in B and C
    ///
    /// # Returns
    /// Ok(()) if operation completed successfully
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn matmul_f32(
        &self,
        a_buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        b_buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        c_buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        // Create matrix descriptors
        let a_desc = MPSContext::create_descriptor(m, k, k * 4, MPSDataType::Float32)?;
        let b_desc = MPSContext::create_descriptor(k, n, n * 4, MPSDataType::Float32)?;
        let c_desc = MPSContext::create_descriptor(m, n, n * 4, MPSDataType::Float32)?;

        // Create MPS matrices
        let a_matrix = self.context.create_matrix(a_buffer, &a_desc)?;
        let b_matrix = self.context.create_matrix(b_buffer, &b_desc)?;
        let c_matrix = self.context.create_matrix(c_buffer, &c_desc)?;

        // Create matmul kernel (alpha=1.0, beta=0.0 for C = A*B)
        let matmul = self.context.create_matmul(
            false, // No transpose for A
            false, // No transpose for B
            m,     // Result rows
            n,     // Result columns
            k,     // Interior dimension
            1.0,   // alpha
            0.0,   // beta
        )?;

        // Execute multiplication
        self.context
            .matrix_multiply(&a_matrix, &b_matrix, &c_matrix, &matmul)?;

        Ok(())
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn matmul_f32(
        &self,
        _a_buffer: &(),
        _b_buffer: &(),
        _c_buffer: &(),
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<(), GpuError> {
        Err(GpuError::Other(
            "Metal not available on this platform".to_string(),
        ))
    }

    /// High-level scaled matrix multiplication for f32 data (C = alpha * A * B)
    ///
    /// Performs optimized scaled matrix multiplication using Metal Performance Shaders.
    /// This fuses the scaling operation into the matmul, eliminating a separate kernel dispatch.
    /// Expected speedup: 1.5-2x over separate matmul + scale operations.
    ///
    /// # Arguments
    /// * `a_buffer` - Left matrix buffer (M x K)
    /// * `b_buffer` - Right matrix buffer (K x N)
    /// * `c_buffer` - Result matrix buffer (M x N)
    /// * `m` - Number of rows in A and C
    /// * `k` - Number of columns in A and rows in B
    /// * `n` - Number of columns in B and C
    /// * `alpha` - Scaling factor for the result (C = alpha * A * B)
    ///
    /// # Returns
    /// Ok(()) if operation completed successfully
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn matmul_f32_scaled(
        &self,
        a_buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        b_buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        c_buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
    ) -> Result<(), GpuError> {
        // Create matrix descriptors
        let a_desc = MPSContext::create_descriptor(m, k, k * 4, MPSDataType::Float32)?;
        let b_desc = MPSContext::create_descriptor(k, n, n * 4, MPSDataType::Float32)?;
        let c_desc = MPSContext::create_descriptor(m, n, n * 4, MPSDataType::Float32)?;

        // Create MPS matrices
        let a_matrix = self.context.create_matrix(a_buffer, &a_desc)?;
        let b_matrix = self.context.create_matrix(b_buffer, &b_desc)?;
        let c_matrix = self.context.create_matrix(c_buffer, &c_desc)?;

        // Create matmul kernel with custom alpha (C = alpha * A * B)
        let matmul = self.context.create_matmul(
            false,        // No transpose for A
            false,        // No transpose for B
            m,            // Result rows
            n,            // Result columns
            k,            // Interior dimension
            alpha as f64, // alpha (scaling factor)
            0.0,          // beta
        )?;

        // Execute multiplication
        self.context
            .matrix_multiply(&a_matrix, &b_matrix, &c_matrix, &matmul)?;

        Ok(())
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn matmul_f32_scaled(
        &self,
        _a_buffer: &(),
        _b_buffer: &(),
        _c_buffer: &(),
        _m: usize,
        _k: usize,
        _n: usize,
        _alpha: f32,
    ) -> Result<(), GpuError> {
        Err(GpuError::Other(
            "Metal not available on this platform".to_string(),
        ))
    }
}

/// MPS-accelerated image operations (stub)
pub struct MPSImageOps {
    pub(crate) context: Arc<MPSContext>,
}

impl MPSImageOps {
    /// Create a new MPS image operations handler (stub)
    pub fn new(_context: Arc<MPSContext>) -> Result<Self, GpuError> {
        // TODO: Implement proper MPS image operations with updated objc2 API
        Err(GpuError::Other(
            "MPS image operations not yet implemented with new objc2 API".to_string(),
        ))
    }

    /// Apply Gaussian blur (stub - objc2 API)
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn gaussian_blur(
        &self,
        _input: &objc2::rc::Retained<dyn MTLBuffer>,
        _output: &mut objc2::rc::Retained<dyn MTLBuffer>,
        _sigma: f32,
    ) -> Result<(), GpuError> {
        // TODO: Implement using objc2 MPSImageGaussianBlur::encode_to_command_buffer
        Err(GpuError::Other(
            "MPS Gaussian blur not yet implemented with new objc2 API".to_string(),
        ))
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn gaussian_blur(
        &self,
        _input: &MTLBuffer,
        _output: &mut MTLBuffer,
        _sigma: f32,
    ) -> Result<(), GpuError> {
        Err(GpuError::Other(
            "Metal not available on this platform".to_string(),
        ))
    }

    /// Apply edge detection (stub - objc2 API)
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn edge_detection(
        &self,
        _input: &objc2::rc::Retained<dyn MTLBuffer>,
        _output: &mut objc2::rc::Retained<dyn MTLBuffer>,
        _threshold: f32,
    ) -> Result<(), GpuError> {
        // TODO: Implement using objc2 MPS image edge detection filters
        Err(GpuError::Other(
            "MPS edge detection not yet implemented with new objc2 API".to_string(),
        ))
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn edge_detection(
        &self,
        _input: &MTLBuffer,
        _output: &mut MTLBuffer,
        _threshold: f32,
    ) -> Result<(), GpuError> {
        Err(GpuError::Other(
            "Metal not available on this platform".to_string(),
        ))
    }
}
